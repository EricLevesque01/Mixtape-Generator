"""
blueprint.py — Phase B (LLM Creative Director)

Responsibilities:
  1. Build a LibraryVocabularySummary from library.json
  2. Call the LLM to produce a Blueprint (7–8 Segments with explicit duration targets)
  3. Validate and repair the output before returning

§3B contract:
  - Input : PersonaProfile + LibraryVocabularySummary
  - Output: Blueprint with 7–8 Segments, each having target_duration_s or target_track_count
  - Blueprint describes *intent*, not exact tracks (algorithm does the fetching)

§2 offline invariant:
  - No external API calls during generation.
  - LibraryVocabularySummary is built entirely from the cached library.
"""

import json
import logging
import uuid
from typing import List, Dict, Tuple, Optional

from .models import PersonaProfile, Blueprint, Segment
from .library import library
from .config import config
from .llm.interface import LLMProvider

logger = logging.getLogger("mixtape_curator.blueprint")


# ---------------------------------------------------------------------------
# Library Vocabulary Summary
# ---------------------------------------------------------------------------

class LibraryVocabularySummary:
    """
    Static snapshot of the library's content vocabulary.
    Passed to the Blueprint LLM prompt so it can make grounded segment decisions.
    Built once and reused across generation runs (cached as a class variable).
    """
    _cached: Optional["LibraryVocabularySummary"] = None

    def __init__(self):
        self.genre_distribution: Dict[str, int] = {}      # genre -> track count
        self.descriptor_cloud: Dict[str, int] = {}        # descriptor -> track count
        self.era_histogram: Dict[str, int] = {}           # decade -> track count  e.g. "1990s": 47
        self.energy_range: Tuple[float, float] = (0.0, 1.0)
        self.valence_range: Tuple[float, float] = (0.0, 1.0)
        self.total_tracks: int = 0
        self.total_duration_s: int = 0

    @classmethod
    def build(cls) -> "LibraryVocabularySummary":
        """Build summary from the loaded library (deterministic, offline)."""
        if cls._cached is not None:
            return cls._cached

        inst = cls()
        df = library.df
        if df.empty:
            logger.warning("LibraryVocabularySummary built on empty library.")
            cls._cached = inst
            return inst

        inst.total_tracks = len(df)
        inst.total_duration_s = int(df["duration_s"].sum()) if "duration_s" in df.columns else 0

        # Genre / descriptor tallies from RYM data
        for _, row in df.iterrows():
            rym = row.get("rym_data", {}) or {}
            for g in (rym.get("primary_genres") or []):
                inst.genre_distribution[g] = inst.genre_distribution.get(g, 0) + 1
            for d in (rym.get("descriptors") or []):
                inst.descriptor_cloud[d] = inst.descriptor_cloud.get(d, 0) + 1

            # Era histogram (bucket by decade)
            year = row.get("release_year")
            if year and isinstance(year, (int, float)) and year > 1900:
                decade = f"{int(year) // 10 * 10}s"
                inst.era_histogram[decade] = inst.era_histogram.get(decade, 0) + 1

        # Energy / valence range
        if "energy" in df.columns and len(df) > 0:
            inst.energy_range = (float(df["energy"].min()), float(df["energy"].max()))
        if "valence" in df.columns and len(df) > 0:
            inst.valence_range = (float(df["valence"].min()), float(df["valence"].max()))

        cls._cached = inst
        logger.info(
            "LibraryVocabularySummary built: %d tracks, %d genres, %d descriptors",
            inst.total_tracks, len(inst.genre_distribution), len(inst.descriptor_cloud),
        )
        return inst

    def to_prompt_block(self, top_n: int = 20) -> str:
        """Render a compact text block for LLM prompts."""
        top_genres = sorted(self.genre_distribution.items(), key=lambda x: -x[1])[:top_n]
        top_descs  = sorted(self.descriptor_cloud.items(),    key=lambda x: -x[1])[:top_n]
        top_eras   = sorted(self.era_histogram.items(),        key=lambda x:  x[0])

        lines = [
            f"LIBRARY SNAPSHOT ({self.total_tracks} tracks, "
            f"{self.total_duration_s // 3600}h {(self.total_duration_s % 3600) // 60}m total):",
            f"  Top genres : {', '.join(f'{g}({c})' for g, c in top_genres)}",
            f"  Descriptors: {', '.join(f'{d}({c})' for d, c in top_descs)}",
            f"  Eras       : {', '.join(f'{e}:{c}' for e, c in top_eras)}",
            f"  Energy     : {self.energy_range[0]:.2f} – {self.energy_range[1]:.2f}",
            f"  Valence    : {self.valence_range[0]:.2f} – {self.valence_range[1]:.2f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Blueprint Generator
# ---------------------------------------------------------------------------

_BLUEPRINT_SYSTEM_PROMPT = """\
You are the Creative Director of a mixtape curation engine.
Your role is ONLY to design a narrative blueprint — a sequence of 7 to 8 thematic chapters \
(called Segments) for a mixtape.

RULES:
1. Produce EXACTLY 7 or 8 segments.
2. Each segment MUST include:
   - segment_id   : short slugified string (e.g. "seg_01_warmup")
   - theme        : 1-2 sentence narrative description
   - target_duration_s : integer seconds (preferred), OR
   - target_track_count: integer (if duration is unknowable)
3. Each segment MAY include:
   - audio_targets  : dict with keys from {energy, valence, intensity, tempo} (0.0–1.0)
   - semantic_targets: dict with keys from {genres: [...], descriptors: [...]}
   - uniformity_bias : float 0.0–1.0 (0=eclectic, 1=uniform within segment)
   - notes          : plain-text for context
4. Total target_duration_s across all segments MUST approximately equal the requested duration.
5. Base your segment design on the LIBRARY SNAPSHOT — your segment themes must be feasible \
   given the available genres and descriptors.
6. The blueprint describes INTENT, not specific tracks. Do not name tracks or artists.
7. Respond ONLY with valid JSON matching this schema (no markdown fences, no extra text):

{
  "narrative_arc": "string — one paragraph describing the overall emotional journey",
  "segments": [
    {
      "segment_id": "string",
      "theme": "string",
      "target_duration_s": int_or_null,
      "target_track_count": int_or_null,
      "audio_targets": { "energy": float, "valence": float } or null,
      "semantic_targets": { "genres": [...], "descriptors": [...] } or null,
      "uniformity_bias": float_or_null,
      "notes": "string_or_null"
    }
  ]
}
"""

def _build_user_prompt(
    persona: PersonaProfile,
    vocab: LibraryVocabularySummary,
    duration_target_s: int,
) -> str:
    confidence_lines = "\n".join(
        f"  {axis}: {score:.2f}" for axis, score in sorted(persona.confidence.items())
    )
    return (
        f"PERSONA PROFILE:\n"
        f"  Occasion   : {persona.occasion or '(not set)'}\n"
        f"  Personality: {', '.join(persona.personality_words) or '(not set)'}\n"
        f"  Mood today : {persona.mood_today or '(not set)'}\n"
        f"  Aesthetic  : {persona.aesthetic_choice or '(not set)'}\n"
        f"  Era pref.  : {persona.era_preference or '(not set)'}\n"
        f"  Wildcard   : {persona.wildcard or '(not set)'}\n"
        f"\nAXIS CONFIDENCE (0=no signal, 1=fully populated):\n{confidence_lines or '  (none)'}\n"
        f"\n{vocab.to_prompt_block()}\n"
        f"\nDURATION TARGET: {duration_target_s} seconds "
        f"({duration_target_s // 60} minutes)\n"
        f"\nDesign the blueprint now."
    )


class BlueprintGenerator:
    """
    §3B — Calls the LLM to produce a Blueprint from a PersonaProfile.
    Falls back to a heuristic blueprint if the LLM is unavailable.
    """

    def __init__(self, llm: Optional[LLMProvider] = None):
        self.llm = llm

    # ------------------------------------------------------------------ public

    def generate(
        self,
        persona: PersonaProfile,
        persona_id: Optional[str] = None,
        duration_target_s: Optional[int] = None,
    ) -> Blueprint:
        """
        Main entry point.

        Args:
            persona          : Completed PersonaProfile from the interview.
            persona_id       : Traceability ID (defaults to new UUID).
            duration_target_s: Target mix length in seconds (defaults to config).

        Returns:
            A validated Blueprint with 7–8 segments.
        """
        target = duration_target_s or config.duration_target_s
        vocab  = LibraryVocabularySummary.build()

        if self.llm is not None:
            try:
                blueprint = self._generate_with_llm(persona, vocab, target)
            except Exception as exc:
                logger.warning("Blueprint LLM call failed (%s); falling back to heuristic.", exc)
                blueprint = self._generate_heuristic(persona, vocab, target)
        else:
            blueprint = self._generate_heuristic(persona, vocab, target)

        blueprint.persona_state_id = persona_id or str(uuid.uuid4())
        return blueprint

    # ----------------------------------------------------------------- private

    def _generate_with_llm(
        self,
        persona: PersonaProfile,
        vocab: LibraryVocabularySummary,
        target: int,
    ) -> Blueprint:
        user_prompt = _build_user_prompt(persona, vocab, target)
        messages = [
            {"role": "system", "content": _BLUEPRINT_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]
        model = config.get("smart_llm_model", config.get("default_llm_model", "gpt-4o"))
        data = self.llm.json(
            messages,
            model=model,
            temperature=config.get("llm_temperature_default", 0.3),
        )
        blueprint = self._parse_and_validate(data, target)
        logger.info(
            "Blueprint generated via LLM: %d segments, arc='%s'",
            len(blueprint.segments), blueprint.narrative_arc[:60],
        )
        return blueprint

    def _parse_and_validate(self, data: dict, target: int) -> Blueprint:
        """Parse LLM JSON and enforce the §3B locked rules."""
        narrative = data.get("narrative_arc", "A curated musical journey.")
        raw_segs  = data.get("segments", [])

        segments: List[Segment] = []
        for s in raw_segs:
            seg = Segment(
                segment_id        = s.get("segment_id", f"seg_{len(segments):02d}"),
                theme             = s.get("theme", "Untitled segment"),
                target_duration_s = s.get("target_duration_s"),
                target_track_count= s.get("target_track_count"),
                audio_targets     = s.get("audio_targets"),
                semantic_targets  = s.get("semantic_targets"),
                uniformity_bias   = s.get("uniformity_bias"),
                notes             = s.get("notes"),
            )
            # Locked rule: must have at least one duration specifier
            if seg.target_duration_s is None and seg.target_track_count is None:
                seg.target_duration_s = target // max(len(raw_segs), 1)
            segments.append(seg)

        # Clamp to 7–8 segments
        segments = self._repair_segment_count(segments, target)

        return Blueprint(
            blueprint_id   = str(uuid.uuid4()),
            persona_state_id = "",       # set by caller
            narrative_arc  = narrative,
            segments       = segments,
        )

    def _repair_segment_count(self, segments: List[Segment], target: int) -> List[Segment]:
        """Ensure 7–8 segments; pad or trim as needed."""
        target_count = 7          # minimum per contract
        max_count    = 8

        while len(segments) < target_count:
            idx = len(segments) + 1
            segments.append(Segment(
                segment_id        = f"seg_{idx:02d}_fill",
                theme             = "Atmospheric interlude",
                target_duration_s = target // target_count,
            ))

        if len(segments) > max_count:
            excess       = len(segments) - max_count
            removed_dur  = sum(s.target_duration_s or 0 for s in segments[-excess:])
            segments     = segments[:max_count]
            # Redistribute excess duration to last kept segment
            if removed_dur and segments[-1].target_duration_s:
                segments[-1].target_duration_s += removed_dur

        return segments

    def _generate_heuristic(
        self,
        persona: PersonaProfile,
        vocab: LibraryVocabularySummary,
        target: int,
    ) -> Blueprint:
        """
        Offline heuristic fallback when the LLM is unavailable.
        Builds a reasonable 7-segment arc from persona data.
        """
        seg_dur = target // 7
        top_genres = sorted(vocab.genre_distribution.items(), key=lambda x: -x[1])

        def pick_genre(rank: int) -> Optional[str]:
            return top_genres[rank][0] if len(top_genres) > rank else None

        # Map persona to broad arc shape
        energy_target = 0.5
        if persona.mood_today:
            m = persona.mood_today.lower()
            if any(w in m for w in ["happy", "excited", "hyped"]):
                energy_target = 0.75
            elif any(w in m for w in ["sad", "down", "melancholic"]):
                energy_target = 0.25

        segments = [
            Segment(
                segment_id        = "seg_01_opening",
                theme             = "Ease into the journey — familiar textures, moderate energy.",
                target_duration_s = seg_dur,
                audio_targets     = {"energy": max(0.1, energy_target - 0.2), "valence": 0.5},
                semantic_targets  = {"genres": [pick_genre(0)] if pick_genre(0) else []},
            ),
            Segment(
                segment_id        = "seg_02_build",
                theme             = "Energy building — the mix picks up momentum.",
                target_duration_s = seg_dur,
                audio_targets     = {"energy": energy_target, "valence": 0.55},
                semantic_targets  = {"genres": [pick_genre(1)] if pick_genre(1) else []},
            ),
            Segment(
                segment_id        = "seg_03_peak",
                theme             = "Peak energy — the heart of the mix.",
                target_duration_s = seg_dur,
                audio_targets     = {"energy": min(1.0, energy_target + 0.2), "valence": 0.6},
            ),
            Segment(
                segment_id        = "seg_04_interlude",
                theme             = "Midpoint breath — a brief tonal shift or slower moment.",
                target_duration_s = seg_dur,
                audio_targets     = {"energy": energy_target * 0.7, "valence": 0.45},
            ),
            Segment(
                segment_id        = "seg_05_resurgence",
                theme             = "Second wind — energy returns with a new flavor.",
                target_duration_s = seg_dur,
                audio_targets     = {"energy": energy_target + 0.1, "valence": 0.6},
                semantic_targets  = {"genres": [pick_genre(2)] if pick_genre(2) else []},
            ),
            Segment(
                segment_id        = "seg_06_descent",
                theme             = "Winding down — softer textures, reflecting on the journey.",
                target_duration_s = seg_dur,
                audio_targets     = {"energy": energy_target * 0.65, "valence": 0.5},
            ),
            Segment(
                segment_id        = "seg_07_close",
                theme             = "Final note — a gentle, memorable close.",
                target_duration_s = target - seg_dur * 6,  # absorb remainder
                audio_targets     = {"energy": max(0.1, energy_target - 0.3), "valence": 0.45},
            ),
        ]

        return Blueprint(
            blueprint_id     = str(uuid.uuid4()),
            persona_state_id = "",
            narrative_arc    = (
                f"A {len(segments)}-chapter arc shaped by the persona's "
                f"'{persona.occasion}' occasion and '{persona.mood_today}' mood, "
                f"moving from build to peak to resolution."
            ),
            segments = segments,
        )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def build_blueprint_generator(llm: Optional[LLMProvider] = None) -> BlueprintGenerator:
    """Factory: returns a BlueprintGenerator wired to the given LLM (or heuristic if None)."""
    return BlueprintGenerator(llm=llm)
