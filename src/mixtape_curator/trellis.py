"""
trellis.py — Phase C (Deterministic Algorithm Engine)

Responsibilities:
  1. For each Segment in a Blueprint, retrieve candidate tracks from the library
  2. Enforce hard constraints at retrieval time (not post-hoc)
  3. Guarantee ≥ 20 candidates per segment where feasible via progressive relaxation
  4. Use the similarity graph to expand and rerank candidate pools
  5. Produce a SegmentTrellis for the arc optimizer to consume

§3C contract:
  - Input : Blueprint + offline library + offline similarity graph
  - Output: SegmentTrellis with deterministic candidate pools
  - Must enforce hard constraints at retrieval time
  - Candidate pools ≥ 20 tracks per segment when feasible

§2 offline invariant: no external API calls. All data from library.json + similarity_graph.json.
"""

import json
import logging
import math
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .config import config
from .diagnostics import DiagnosticCode
from .library import library
from .models import Blueprint, Segment, SegmentTrellis, Track, UserProfile

logger = logging.getLogger("mixtape_curator.trellis")

# Minimum candidate pool size per §3C
MIN_POOL_SIZE = 20

# Relaxation stages applied when pool is undersized
# Each stage value is the fraction of constraints to relax
_RELAXATION_STAGES = [
    "strict",       # stage 0: all hard constraints applied
    "relax_sub",    # stage 1: subgenre match → primary genre only
    "relax_desc",   # stage 2: also ignore descriptor constraints
    "relax_genre",  # stage 3: drop genre constraint entirely (audio only)
    "unconstrained",# stage 4: return all library tracks that pass duration filter
]


# ---------------------------------------------------------------------------
# Similarity Graph Loader (lazy, cached)
# ---------------------------------------------------------------------------

class _SimilarityGraph:
    """Thin wrapper around similarity_graph.json for neighbour lookup."""

    _instance: Optional["_SimilarityGraph"] = None
    _edges: Dict[str, Dict[str, float]] = {}    # track_id → {neighbour_id: weight}

    @classmethod
    def instance(cls) -> "_SimilarityGraph":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        root = Path(__file__).parent.parent.parent
        path = root / "data" / "similarity_graph.json"
        if not path.exists():
            logger.warning("similarity_graph.json not found at %s; graph disabled.", path)
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Support two formats:
            #  {track_id: {neighbour_id: weight}}  (preferred)
            #  {track_id: [neighbour_id, ...]}       (legacy list)
            for tid, neighbours in raw.items():
                if isinstance(neighbours, dict):
                    self._edges[tid] = neighbours
                elif isinstance(neighbours, list):
                    self._edges[tid] = {n: 1.0 for n in neighbours}
            logger.info("Similarity graph loaded: %d nodes.", len(self._edges))
        except Exception as exc:
            logger.warning("Failed to load similarity_graph.json: %s", exc)

    def get_neighbours(self, track_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return up to top_k neighbours sorted by descending edge weight."""
        nbrs = self._edges.get(track_id, {})
        return sorted(nbrs.items(), key=lambda x: -x[1])[:top_k]

    def has_edge(self, a: str, b: str) -> bool:
        return b in self._edges.get(a, {})

    def edge_weight(self, a: str, b: str) -> float:
        return self._edges.get(a, {}).get(b, 0.0)


similarity_graph = _SimilarityGraph  # accessed via .instance()


# ---------------------------------------------------------------------------
# Candidate Scoring
# ---------------------------------------------------------------------------

def _score_candidate(track: Track, segment: Segment, profile: UserProfile) -> float:
    """
    Relevance score for a candidate track against a segment's targets.
    Higher = better fit. Used to rank the candidate pool.
    """
    score = 0.0

    # 1. Audio target alignment (if specified)
    at = segment.audio_targets or {}
    for feat, target_val in at.items():
        actual = getattr(track, feat, None)
        if actual is not None:
            delta = abs(actual - target_val)
            score += 1.0 - delta   # max contribution 1.0 per feature

    # 2. Semantic target alignment
    st = segment.semantic_targets or {}
    track_genres = set(track.rym_data.primary_genres + track.rym_data.subgenres)
    track_descs  = set(track.rym_data.descriptors)

    for g in st.get("genres", []):
        if g.lower() in {x.lower() for x in track_genres}:
            score += 1.5
    for d in st.get("descriptors", []):
        if d.lower() in {x.lower() for x in track_descs}:
            score += 0.75

    # 3. RYM rating (quality signal)
    score += track.rating * 1.5

    # 4. Mix prominence signal (social proof from existing mixes)
    score += track.mix_prominence * 0.5

    return score


# ---------------------------------------------------------------------------
# Trellis Builder
# ---------------------------------------------------------------------------

class TrellisBuilder:
    """
    §3C — Builds a SegmentTrellis by retrieving deterministic candidate pools
    for each segment in the Blueprint.
    """

    def __init__(self, profile: UserProfile):
        self.profile   = profile
        self.graph     = similarity_graph.instance()
        self.min_dur   = config.get("track_min_duration_s", 60)
        self.max_dur   = config.get("track_max_duration_s", 600)

    # ------------------------------------------------------------------ public

    def build(self, blueprint: Blueprint) -> SegmentTrellis:
        """
        Main entry point.
        Processes every segment in the blueprint and returns a SegmentTrellis.
        """
        trellis = SegmentTrellis(
            trellis_id   = str(uuid.uuid4()),
            blueprint_id = blueprint.blueprint_id,
        )

        # Pre-fetch base candidate set (respects hard profile exclusions)
        base_candidates = library.filter_candidates(self.profile)
        base_candidates = [
            t for t in base_candidates
            if self.min_dur <= t.duration_s <= self.max_dur
        ]
        base_id_set = {t.id for t in base_candidates}

        for segment in blueprint.segments:
            pool, stage, codes = self._build_pool(segment, base_candidates)

            if stage > 0:
                logger.info(
                    "Segment '%s': reached stage-%d relaxation, pool=%d",
                    segment.segment_id, stage, len(pool),
                )

            if len(pool) < MIN_POOL_SIZE:
                trellis.undersized_segments.append(segment.segment_id)
                if not pool:
                    trellis.diagnostic_codes[segment.segment_id] = (
                        DiagnosticCode.NO_CANDIDATES_FOR_SEGMENT
                    )
                    logger.warning("Segment '%s' has zero candidates.", segment.segment_id)
                else:
                    trellis.diagnostic_codes[segment.segment_id] = (
                        DiagnosticCode.INSUFFICIENT_CANDIDATES
                    )

            # Expand pool via similarity graph
            pool = self._expand_via_graph(pool, base_id_set)

            # Rank and deduplicate
            scored = [(self._score_candidate(t, segment), t) for t in pool]
            scored.sort(key=lambda x: -x[0])
            deduped_ids = list(dict.fromkeys(t.id for _, t in scored))

            trellis.segment_pools[segment.segment_id] = deduped_ids
            logger.debug(
                "Segment '%s' final pool: %d candidates (stage=%d)",
                segment.segment_id, len(deduped_ids), stage,
            )

        return trellis

    # ----------------------------------------------------------------- private

    def _build_pool(
        self,
        segment: Segment,
        base_candidates: List[Track],
    ) -> Tuple[List[Track], int, List[str]]:
        """
        Progressive relaxation: try each stage until MIN_POOL_SIZE is reached.
        Returns (pool, stage_reached, diagnostic_codes).
        """
        codes: List[str] = []

        for stage_idx, stage in enumerate(_RELAXATION_STAGES):
            pool = self._filter_for_stage(segment, base_candidates, stage)
            if len(pool) >= MIN_POOL_SIZE:
                return pool, stage_idx, codes
            # Record that we had to relax
            if stage_idx > 0:
                codes.append(DiagnosticCode.INSUFFICIENT_CANDIDATES)

        return pool, len(_RELAXATION_STAGES) - 1, codes

    def _filter_for_stage(
        self,
        segment: Segment,
        base: List[Track],
        stage: str,
    ) -> List[Track]:
        """Apply constraints appropriate for the given relaxation stage."""
        if stage == "unconstrained":
            return list(base)

        pool: List[Track] = []
        st = segment.semantic_targets or {}
        at = segment.audio_targets or {}

        target_genres   = {g.lower() for g in st.get("genres", [])}
        target_descs    = {d.lower() for d in st.get("descriptors", [])}

        for track in base:
            # --- Semantic matching ---
            if stage in ("strict", "relax_sub") and target_genres:
                track_primary = {g.lower() for g in track.rym_data.primary_genres}
                track_all     = track_primary | {g.lower() for g in track.rym_data.subgenres}

                if stage == "strict":
                    if not (target_genres & track_all):
                        continue
                else:  # relax_sub: primary genre only
                    if not (target_genres & track_primary):
                        continue

            if stage == "strict" and target_descs:
                track_descs = {d.lower() for d in track.rym_data.descriptors}
                if not (target_descs & track_descs):
                    continue

            # --- Audio target matching (applied unless relax_genre/unconstrained) ---
            if stage not in ("relax_genre", "unconstrained") and at:
                if not self._audio_in_range(track, at, tolerance=0.25):
                    continue

            pool.append(track)

        return pool

    def _audio_in_range(
        self,
        track: Track,
        audio_targets: Dict[str, float],
        tolerance: float = 0.25,
    ) -> bool:
        """
        Return True if all specified audio features are within ±tolerance of the target.
        """
        for feat, target_val in audio_targets.items():
            actual = getattr(track, feat, None)
            if actual is None:
                continue   # missing feature: don't penalise
            if abs(actual - target_val) > tolerance:
                return False
        return True

    def _expand_via_graph(
        self,
        pool: List[Track],
        all_track_ids: Set[str],
        top_k_neighbours: int = 5,
        max_additions: int = 30,
    ) -> List[Track]:
        """
        §3C — Use the similarity graph to expand the candidate pool.
        For each track already in the pool, fetch its top-k neighbours and
        add them if they are in the base candidate set and not already in the pool.
        """
        existing_ids = {t.id for t in pool}
        additions: List[Track] = []

        for seed in pool:
            if len(additions) >= max_additions:
                break
            for nbr_id, _weight in self.graph.get_neighbours(seed.id, top_k=top_k_neighbours):
                if nbr_id in existing_ids or nbr_id not in all_track_ids:
                    continue
                neighbour = library.get_track(nbr_id)
                if neighbour is None:
                    continue
                if not (self.min_dur <= neighbour.duration_s <= self.max_dur):
                    continue
                additions.append(neighbour)
                existing_ids.add(nbr_id)

        return pool + additions

    @staticmethod
    def _score_candidate(track: Track, segment: Segment) -> float:
        """Delegate to module-level scorer."""
        return _score_candidate(track, segment, profile=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def build_trellis(blueprint: Blueprint, profile: UserProfile) -> SegmentTrellis:
    """Factory: build a SegmentTrellis for the given Blueprint and UserProfile."""
    return TrellisBuilder(profile).build(blueprint)
