"""
diagnostics.py — §10 Failure Mode Handling

Defines:
  - DiagnosticCode  : enum of all structured reason codes
  - ArcFailure      : structured failure result (best-effort draft + codes)
  - RelaxationMenu  : generates user-facing relaxation options for consult_user()
"""

from enum import Enum
from typing import List, Optional


class DiagnosticCode(str, Enum):
    """
    §10 — Structured reason codes.
    All failure paths in the arc optimizer must report at least one of these.
    """

    # Trellis / candidate retrieval failures
    INSUFFICIENT_CANDIDATES   = "INSUFFICIENT_CANDIDATES"    # segment pool < 20 after all relaxations
    NO_CANDIDATES_FOR_SEGMENT = "NO_CANDIDATES_FOR_SEGMENT"  # pool is completely empty

    # Duration arithmetic failures
    DURATION_INFEASIBLE        = "DURATION_INFEASIBLE"        # cannot reach 0.9 * duration_target_s
    DURATION_CAP_EXCEEDED      = "DURATION_CAP_EXCEEDED"      # hard cap violated (should never happen)

    # Constraint conflicts
    CONSTRAINT_CONFLICT        = "CONSTRAINT_CONFLICT"        # mutually exclusive constraints
    ARTIST_CAP_CONFLICT        = "ARTIST_CAP_CONFLICT"        # max_tracks_per_artist too restrictive
    GENRE_BLACKLIST_CONFLICT   = "GENRE_BLACKLIST_CONFLICT"   # all candidates excluded by genre block

    # A/B distinctness
    AB_JACCARD_VIOLATION       = "AB_JACCARD_VIOLATION"       # could not achieve Jaccard ≤ 0.8
    AB_SEGMENT_REUSE           = "AB_SEGMENT_REUSE"           # B shares all segments with A

    # Refinement
    REFINEMENT_NO_RATIONALE    = "REFINEMENT_NO_RATIONALE"    # finalize_playlist not called
    TOOL_VALIDATION_FAILURE    = "TOOL_VALIDATION_FAILURE"    # post-tool constraint check failed

    # General
    UNKNOWN                    = "UNKNOWN"


class RelaxationOption:
    """A single row in the relaxation menu shown to the user via consult_user()."""

    def __init__(self, code: str, description: str, suggestion: str):
        self.code        = code         # e.g. "relax_genre"
        self.description = description  # what the problem is
        self.suggestion  = suggestion   # what the user can do about it

    def __str__(self) -> str:
        return f"[{self.code}] {self.description} — {self.suggestion}"


class RelaxationMenu:
    """
    §10 — Generates a user-facing relaxation menu from a list of diagnostic codes.
    The menu is passed to consult_user() when arc generation fails.
    """

    _CODE_TO_OPTIONS = {
        DiagnosticCode.INSUFFICIENT_CANDIDATES: RelaxationOption(
            "relax_genre",
            "Not enough tracks matched the requested genre/style.",
            "Broaden the genre or add related styles (e.g. 'post-punk' → 'indie rock').",
        ),
        DiagnosticCode.NO_CANDIDATES_FOR_SEGMENT: RelaxationOption(
            "skip_segment",
            "One segment had zero matching tracks.",
            "Remove that segment or replace it with a more common style.",
        ),
        DiagnosticCode.DURATION_INFEASIBLE: RelaxationOption(
            "shorten_mix",
            "Library doesn't have enough content to fill the requested duration.",
            "Reduce the target length (e.g. from 77 min to 55 min).",
        ),
        DiagnosticCode.ARTIST_CAP_CONFLICT: RelaxationOption(
            "relax_artist_cap",
            "Artist diversity constraint too strict for available library.",
            "Allow more tracks per artist (e.g. increase from 2 to 3).",
        ),
        DiagnosticCode.GENRE_BLACKLIST_CONFLICT: RelaxationOption(
            "remove_exclusion",
            "Excluded genres blocked all viable candidates.",
            "Remove one of the genre exclusions.",
        ),
        DiagnosticCode.AB_JACCARD_VIOLATION: RelaxationOption(
            "accept_similar_b",
            "Could not create a sufficiently different B-side.",
            "Accept a B-side with higher overlap, or allow the system to use a different segment strategy.",
        ),
        DiagnosticCode.CONSTRAINT_CONFLICT: RelaxationOption(
            "review_constraints",
            "Some constraints are mutually exclusive.",
            "Review must-include and exclude lists for conflicts.",
        ),
    }

    @classmethod
    def from_codes(cls, codes: List[str]) -> "RelaxationMenu":
        menu = cls()
        seen = set()
        for code_str in codes:
            try:
                code = DiagnosticCode(code_str)
            except ValueError:
                code = DiagnosticCode.UNKNOWN
            if code not in seen and code in cls._CODE_TO_OPTIONS:
                menu._options.append(cls._CODE_TO_OPTIONS[code])
                seen.add(code)
        return menu

    def __init__(self):
        self._options: List[RelaxationOption] = []

    def is_empty(self) -> bool:
        return len(self._options) == 0

    def to_consult_question(self) -> str:
        """
        Render the menu as a question string suitable for passing to consult_user().
        §10: no silent failure — always surfaces a recovery path.
        """
        if not self._options:
            return (
                "I couldn't build a valid arc with the current constraints. "
                "Would you like to relax any requirements? "
                "Reply 'yes' to try with looser settings, or 'no' to keep the best-effort draft."
            )

        lines = [
            "I ran into some issues building your mixtape. Here's what I found:\n"
        ]
        for i, opt in enumerate(self._options, 1):
            lines.append(f"  {i}. {opt}")

        lines.append(
            "\nWhich would you like to try? "
            "Reply with one or more option numbers, or 'skip' to keep the best-effort draft."
        )
        return "\n".join(lines)
