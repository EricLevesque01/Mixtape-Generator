"""
arc_optimizer.py — Phase D (Deterministic Algorithm Engine)

Responsibilities:
  1. Select 3–5 segments from the 7–8 segment trellis pool
  2. Fill tracks per segment from the candidate pools
  3. Order segments + tracks globally using flow scoring
  4. Produce two distinct ArcDrafts (A and B) satisfying §7 distinctness guarantee
  5. Enforce duration constraints: hard cap + soft floor (§3D)
  6. Score each draft deterministically (§5 objective ordering)

§5 Optimization objective (priority order):
  1. Maximize total_score (fit, flow, variety, accessibility, quality)
  2. Minimize duration error
  3. Maximize arc coherence (segment-boundary flow)

§7 A/B distinctness:
  - At least one different segment_id
  - Jaccard(segment_id sets of A, B) ≤ 0.8

§6 Segment-boundary flow:
  - Measure transition quality at each segment boundary
  - Used as tiebreaker / additive scorer

§9 Determinism:
  - All operations use self.rng (seeded from config.rng_seed)
  - No random.random() calls — only self.rng
"""

import logging
import math
import uuid
from typing import Dict, List, Optional, Set, Tuple

from .config import config
from .diagnostics import DiagnosticCode, RelaxationMenu
from .library import library
from .models import (
    ArcDraft,
    Playlist,
    PlaylistScores,
    Segment,
    SegmentTrellis,
    Track,
    UserProfile,
)
from .scoring import scorer

import random

logger = logging.getLogger("mixtape_curator.arc_optimizer")

# How many segment combinations to evaluate
_MAX_COMBINATIONS = 50
# Minimum segments in each draft
_MIN_SEGMENTS = 3
# Maximum segments in each draft
_MAX_SEGMENTS = 5
# Soft floor fraction (§3D)
_DURATION_FLOOR_RATIO = 0.9


# ---------------------------------------------------------------------------
# Segment-Boundary Flow (§6)
# ---------------------------------------------------------------------------

def compute_boundary_flow(
    seg_a_tracks: List[Track],
    seg_b_tracks: List[Track],
) -> float:
    """
    §6 — Score the transition between the final track(s) of segment A
    and the first track(s) of segment B.

    Implementation: measure the sonic distance between the last track
    of A and the first track of B. Lower sonic distance = better flow.
    Returns a score in [0, 1] where 1 = perfect transition.
    """
    if not seg_a_tracks or not seg_b_tracks:
        return 1.0  # no transition to score

    last  = seg_a_tracks[-1]
    first = seg_b_tracks[0]

    d2 = (
        (last.energy    - first.energy)    ** 2 +
        (last.valence   - first.valence)   ** 2 +
        (last.intensity - first.intensity) ** 2
    )
    sonic_distance = math.sqrt(d2 / 3.0)   # normalised in [0, 1]
    return round(1.0 - sonic_distance, 4)


def compute_arc_boundary_flow(
    segment_order: List[str],
    tracks_per_segment: Dict[str, List[Track]],
) -> float:
    """Average boundary flow across all consecutive segment pairs in the arc."""
    if len(segment_order) < 2:
        return 1.0
    scores = []
    for i in range(len(segment_order) - 1):
        a = tracks_per_segment.get(segment_order[i], [])
        b = tracks_per_segment.get(segment_order[i + 1], [])
        scores.append(compute_boundary_flow(a, b))
    return round(sum(scores) / len(scores), 4) if scores else 1.0


# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------

def _draft_duration(segment_ids: List[str], tracks_by_seg: Dict[str, List[Track]]) -> int:
    return sum(t.duration_s for sid in segment_ids for t in tracks_by_seg.get(sid, []))


# ---------------------------------------------------------------------------
# Track Filling
# ---------------------------------------------------------------------------

def _fill_segment_tracks(
    segment: Segment,
    candidate_ids: List[str],
    already_used: Set[str],
    profile: UserProfile,
    duration_budget_s: Optional[int] = None,
) -> List[Track]:
    """
    Select tracks from a segment's candidate pool.

    Uses:
      - target_duration_s  — fill until this budget is met (preferred).
      - target_track_count — hard count if no duration specified.

    Respects max_tracks_per_artist and duration budget.
    Returns an ordered list of Track objects.
    """
    max_per_artist = config.max_tracks_per_artist
    min_dur = config.get("track_min_duration_s", 60)
    max_dur = config.get("track_max_duration_s", 600)

    # Determine budget
    if segment.target_duration_s is not None:
        budget = duration_budget_s if duration_budget_s is not None else segment.target_duration_s
        use_count_mode = False
    elif segment.target_track_count is not None:
        budget = None
        target_count = segment.target_track_count
        use_count_mode = True
    else:
        budget = config.duration_target_s // 7
        use_count_mode = False

    artist_counts: Dict[str, int] = {}
    selected: List[Track] = []
    current_dur = 0

    def _try_fill(pool: List[str], per_artist_limit: int) -> None:
        """Inner fill loop with the given per-artist cap."""
        nonlocal current_dur
        for tid in pool:
            if tid in already_used:
                continue
            track = library.get_track(tid)
            if track is None:
                continue
            if not (min_dur <= track.duration_s <= max_dur):
                continue
            if artist_counts.get(track.artist, 0) >= per_artist_limit:
                continue

            if use_count_mode:
                selected.append(track)
                already_used.add(tid)
                artist_counts[track.artist] = artist_counts.get(track.artist, 0) + 1
                if len(selected) >= target_count:
                    return
            else:
                if budget is not None and current_dur >= budget:
                    return
                selected.append(track)
                already_used.add(tid)
                current_dur += track.duration_s
                artist_counts[track.artist] = artist_counts.get(track.artist, 0) + 1

    # Pass 1: strongly prefer 1 track per artist
    _try_fill(candidate_ids, 1)

    # Pass 2: allow up to max_tracks_per_artist only if budget/count not yet met
    budget_met = (current_dur >= (budget or 0)) if not use_count_mode else (len(selected) >= target_count)
    if not budget_met and max_per_artist > 1:
        _try_fill(candidate_ids, max_per_artist)

    return selected


# ---------------------------------------------------------------------------
# Sequence Optimiser (within a segment)
# ---------------------------------------------------------------------------

def _greedy_sequence(tracks: List[Track]) -> List[Track]:
    """
    Simple greedy nearest-neighbour flow sequence within a segment.
    Identical logic to Generator._optimize_sequence_greedy but operates
    on Track objects directly and is deterministic (no RNG, single start).
    """
    if len(tracks) <= 2:
        return tracks

    remaining = list(tracks)
    # Seed: prefer track_number == 1 or earliest in album
    remaining.sort(key=lambda t: (t.track_number or 99))
    seq = [remaining.pop(0)]

    while remaining:
        last = seq[-1]
        best_idx, best_score = 0, -float("inf")
        for i, candidate in enumerate(remaining):
            d2 = (
                (last.energy    - candidate.energy)    ** 2 +
                (last.valence   - candidate.valence)   ** 2 +
                (last.intensity - candidate.intensity) ** 2
            )
            score = 1.0 - math.sqrt(d2 / 3.0)
            if score > best_score:
                best_score = score
                best_idx   = i
        seq.append(remaining.pop(best_idx))

    return seq


# ---------------------------------------------------------------------------
# Arc Optimizer
# ---------------------------------------------------------------------------

class ArcOptimizer:
    """
    §3D — Selects 3–5 segments from the trellis, fills tracks, and produces
    two distinct ArcDrafts (A and B).
    """

    def __init__(self, profile: UserProfile, rng_seed: Optional[int] = None):
        self.profile   = profile
        self.rng       = random.Random(rng_seed if rng_seed is not None else config.get("rng_seed", 12345))

    # ------------------------------------------------------------------ public

    def optimize(
        self,
        trellis: SegmentTrellis,
        blueprint_segments: List[Segment],
        duration_target_s: Optional[int] = None,
    ) -> Tuple[ArcDraft, ArcDraft]:
        """
        Produce ArcDraft A and ArcDraft B.

        Args:
            trellis            : SegmentTrellis from Phase C.
            blueprint_segments : The full Segment list from the Blueprint (for metadata).
            duration_target_s  : Mix length target (s). Falls back to config or profile.

        Returns:
            (draft_a, draft_b) — two distinct drafts satisfying §7.
        """
        target = duration_target_s or self.profile.duration_target_s or config.duration_target_s
        cap    = config.duration_cap_s

        # Map segment_id → Segment object for later lookup
        seg_map: Dict[str, Segment] = {s.segment_id: s for s in blueprint_segments}

        # Available segment ids (only those with non-empty pools)
        available = [
            sid for sid, pool in trellis.segment_pools.items() if pool
        ]

        if len(available) < _MIN_SEGMENTS:
            logger.error(
                "Only %d segments have candidates; need at least %d.",
                len(available), _MIN_SEGMENTS,
            )
            # Return best-effort empty drafts with diagnostics
            return self._empty_draft("A"), self._empty_draft("B")

        # Generate candidate arc combinations and score them
        combos = self._enumerate_combinations(available, seg_map, target)
        combos.sort(key=lambda x: x[1], reverse=True)   # highest total_score first

        if not combos:
            return self._empty_draft("A"), self._empty_draft("B")

        # Build Draft A from best combo
        best_combo_ids, _ = combos[0]
        draft_a = self._build_draft("A", best_combo_ids, seg_map, trellis, target, cap)

        # Build Draft B: find first combo with sufficient distinctness from A (§7)
        draft_b = self._build_draft_b(draft_a, combos, seg_map, trellis, target, cap)

        return draft_a, draft_b

    # ------------------------------------------------------------------ A/B

    def _build_draft_b(
        self,
        draft_a: ArcDraft,
        combos: List[Tuple[List[str], float]],
        seg_map: Dict[str, Segment],
        trellis: SegmentTrellis,
        target: int,
        cap: int,
    ) -> ArcDraft:
        """
        §7 — Select the highest-scoring combo that is distinct from A.
        Distinctness: ≥1 different segment_id AND Jaccard ≤ 0.8.
        Falls back to best available if no fully-distinct combo exists.
        """
        a_set = set(draft_a.selected_segment_ids)

        for combo_ids, _score in combos[1:]:   # skip index 0 (already used for A)
            b_set = set(combo_ids)
            if a_set == b_set:
                continue
            jaccard = len(a_set & b_set) / max(len(a_set | b_set), 1)
            if jaccard <= 0.8:
                return self._build_draft("B", combo_ids, seg_map, trellis, target, cap)

        # Fallback: force at least one segment swap
        logger.warning(
            "Could not find fully-distinct B combo; forcing segment swap on best combo."
        )
        return self._force_distinct_b(draft_a, combos, seg_map, trellis, target, cap)

    def _force_distinct_b(
        self,
        draft_a: ArcDraft,
        combos: List[Tuple[List[str], float]],
        seg_map: Dict[str, Segment],
        trellis: SegmentTrellis,
        target: int,
        cap: int,
    ) -> ArcDraft:
        """Replace one segment from A with one not in A to create a distinct B."""
        a_ids = list(draft_a.selected_segment_ids)
        all_seg_ids   = list(trellis.segment_pools.keys())
        outside_a     = [s for s in all_seg_ids if s not in set(a_ids) and trellis.segment_pools.get(s)]

        if not outside_a:
            # Truly no alternative — B = A with a diagnostic flag
            draft_b = self._build_draft("B", a_ids, seg_map, trellis, target, cap)
            draft_b.diagnostic_codes.append(DiagnosticCode.AB_SEGMENT_REUSE)
            return draft_b

        # Swap the weakest segment in A for the best available outside A
        swap_out = a_ids[-1]    # simplistic: last segment
        swap_in  = outside_a[0]
        new_ids  = a_ids[:-1] + [swap_in]

        draft_b = self._build_draft("B", new_ids, seg_map, trellis, target, cap)
        a_set = set(a_ids)
        b_set = set(new_ids)
        jaccard = len(a_set & b_set) / max(len(a_set | b_set), 1)
        if jaccard > 0.8:
            draft_b.diagnostic_codes.append(DiagnosticCode.AB_JACCARD_VIOLATION)

        return draft_b

    # ------------------------------------------------------------------ draft building

    def _build_draft(
        self,
        label: str,
        segment_ids: List[str],
        seg_map: Dict[str, Segment],
        trellis: SegmentTrellis,
        target: int,
        cap: int,
    ) -> ArcDraft:
        """
        Assemble a complete ArcDraft: fill tracks per segment, sequence them,
        score everything, enforce duration constraints.
        """
        draft = ArcDraft(
            draft_id             = str(uuid.uuid4()),
            label                = label,
            selected_segment_ids = list(segment_ids),
        )

        # Per-segment duration budget
        per_seg_budget = target // max(len(segment_ids), 1)

        used_track_ids: Set[str] = set()
        tracks_per_seg_obj: Dict[str, List[Track]] = {}

        for sid in segment_ids:
            seg = seg_map.get(sid)
            if seg is None:
                continue
            pool = trellis.segment_pools.get(sid, [])
            seg_tracks = _fill_segment_tracks(
                segment        = seg,
                candidate_ids  = pool,
                already_used   = used_track_ids,
                profile        = self.profile,
                duration_budget_s = per_seg_budget,
            )
            # Sequence within segment
            seg_tracks = _greedy_sequence(seg_tracks)
            tracks_per_seg_obj[sid] = seg_tracks
            draft.tracks_per_segment[sid] = [t.id for t in seg_tracks]

        # Global order: concatenate segments as-is (segment order IS the arc order)
        global_tracks: List[Track] = []
        for sid in segment_ids:
            global_tracks.extend(tracks_per_seg_obj.get(sid, []))

        draft.global_order = [t.id for t in global_tracks]
        draft.total_duration_s = sum(t.duration_s for t in global_tracks)

        # Duration hard cap enforcement
        if draft.total_duration_s > cap:
            draft, global_tracks = self._trim_to_cap(draft, global_tracks, cap)

        # Duration soft floor diagnostic
        floor = int(_DURATION_FLOOR_RATIO * target)
        if draft.total_duration_s < floor:
            draft.diagnostic_codes.append(DiagnosticCode.DURATION_INFEASIBLE)

        draft.duration_error_s = abs(draft.total_duration_s - target)

        # Score
        if global_tracks:
            draft.scores = scorer.score_playlist(global_tracks, self.profile)

        # §6 Boundary flow
        draft.boundary_flow_score = compute_arc_boundary_flow(segment_ids, tracks_per_seg_obj)

        return draft

    def _trim_to_cap(
        self,
        draft: ArcDraft,
        tracks: List[Track],
        cap: int,
    ) -> Tuple[ArcDraft, List[Track]]:
        """Remove tracks from the end until total_duration_s <= cap."""
        while tracks and sum(t.duration_s for t in tracks) > cap:
            removed = tracks.pop()
            draft.global_order = [tid for tid in draft.global_order if tid != removed.id]
            # Remove from tracks_per_segment
            for sid in list(draft.tracks_per_segment):
                if removed.id in draft.tracks_per_segment[sid]:
                    draft.tracks_per_segment[sid].remove(removed.id)
        draft.total_duration_s = sum(t.duration_s for t in tracks)
        return draft, tracks

    # ------------------------------------------------------------------ combo enumeration

    def _enumerate_combinations(
        self,
        available: List[str],
        seg_map: Dict[str, Segment],
        target: int,
    ) -> List[Tuple[List[str], float]]:
        """
        Enumerate ordered subsets of 3–5 segments and score each.
        Limits total enumeration to _MAX_COMBINATIONS to remain fast.
        """
        from itertools import combinations

        cap = config.duration_cap_s
        floor = int(_DURATION_FLOOR_RATIO * target)
        results: List[Tuple[List[str], float]] = []

        n = len(available)
        for size in range(_MIN_SEGMENTS, min(_MAX_SEGMENTS, n) + 1):
            all_combos = list(combinations(available, size))
            # Shuffle deterministically then cap
            self.rng.shuffle(all_combos)
            all_combos = all_combos[:_MAX_COMBINATIONS]

            for combo in all_combos:
                combo_list = list(combo)
                # Fast duration estimate using target_duration_s from segments
                est_dur = self._estimate_duration(combo_list, seg_map, target)

                if est_dur > cap:
                    continue   # would exceed hard cap before even filling
                if est_dur < floor * 0.5:
                    continue   # obviously too short

                score = self._score_combo(combo_list, seg_map, est_dur, target)
                results.append((combo_list, score))

                if len(results) >= _MAX_COMBINATIONS:
                    return results

        return results

    def _estimate_duration(
        self,
        combo: List[str],
        seg_map: Dict[str, Segment],
        overall_target: int,
    ) -> int:
        """Fast estimate: sum target_duration_s from each segment's definition."""
        total = 0
        per_seg_default = overall_target // max(len(combo), 1)
        for sid in combo:
            seg = seg_map.get(sid)
            if seg and seg.target_duration_s:
                total += seg.target_duration_s
            else:
                total += per_seg_default
        return total

    def _score_combo(
        self,
        combo: List[str],
        seg_map: Dict[str, Segment],
        est_dur: int,
        target: int,
    ) -> float:
        """
        Cheap proxy score for combo ranking before actual track filling.
        §5 objective (priority proxy):
          1. Segment diversity (uniformity coverage)
          2. Duration error
          3. Variety (number of distinct themes)
        """
        diversity_bonus = len(set(combo)) / max(len(combo), 1)
        dur_error_penalty = abs(est_dur - target) / max(target, 1)
        return diversity_bonus - dur_error_penalty * 0.5

    # ------------------------------------------------------------------ empty draft helper

    def _empty_draft(self, label: str) -> ArcDraft:
        """Return an empty ArcDraft with a failure diagnostic."""
        d = ArcDraft(
            draft_id = str(uuid.uuid4()),
            label    = label,
        )
        d.diagnostic_codes.append(DiagnosticCode.INSUFFICIENT_CANDIDATES)
        return d


# ---------------------------------------------------------------------------
# ArcDraft → Playlist adapter
# ---------------------------------------------------------------------------

def arc_draft_to_playlist(draft: ArcDraft) -> Playlist:
    """
    Convert an ArcDraft to the legacy Playlist format used by the ReActAgent,
    exporter, and UI. Preserves segment metadata in the new fields.
    """
    from .models import Playlist, PlaylistScores

    pl = Playlist(
        id             = draft.draft_id,
        track_ids      = list(draft.global_order),
        total_duration_s = draft.total_duration_s,
        scores         = draft.scores,
        violations     = [str(c) for c in draft.diagnostic_codes],
        segments_used  = list(draft.selected_segment_ids),
    )
    return pl


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def optimize_arc(
    trellis: "SegmentTrellis",
    blueprint_segments: List[Segment],
    profile: UserProfile,
    duration_target_s: Optional[int] = None,
) -> Tuple[ArcDraft, ArcDraft]:
    """Factory: build an ArcOptimizer and run it."""
    optimizer = ArcOptimizer(profile=profile)
    return optimizer.optimize(trellis, blueprint_segments, duration_target_s)
