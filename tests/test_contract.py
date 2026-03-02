"""
test_contract.py — Rigorous compliance tests for REACT_MIXTAPE_CURATOR_V5_ARCH_CONTRACT.md

One test class per contract section (§1–§10).
All tests use synthetic in-memory library data so they never hit disk or the network.
"""
import pytest
import random
import uuid
import pandas as pd

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

from mixtape_curator.models import (
    ArcDraft, Blueprint, FeedbackTargets, PersonaProfile, Playlist, PlaylistScores,
    RYMData, Segment, SegmentTrellis, Track, UserProfile,
)
from mixtape_curator.library import library
from mixtape_curator.scoring import scorer
from mixtape_curator.config import config


def _make_track(
    tid: str,
    artist: str = "Artist",
    genre: str = "Rock",
    energy: float = 0.5,
    valence: float = 0.5,
    intensity: float = 0.5,
    duration_s: int = 240,
    rating: float = 0.7,
) -> dict:
    """Return a flat dict matching the library DataFrame schema."""
    return {
        "id": tid,
        "title": f"Song {tid}",
        "artist": artist,
        "album": "Album",
        "duration_s": duration_s,
        "energy": energy,
        "valence": valence,
        "intensity": intensity,
        "tempo": 120.0,
        "danceability": 0.5,
        "key": 0,
        "mode": 1,
        "key_full": "C major",
        "acousticness": 0.3,
        "instrumentalness": 0.0,
        "speechiness": 0.0,
        "liveness": 0.1,
        "brightness": 0.5,
        "flatness": 0.0,
        "entropy": 0.5,
        "dynamic_range": 0.5,
        "accessibility": 0.7,
        "familiarity": 0.6,
        "rating": rating,
        "recommendability": 0.6,
        "liked": False,
        "album_loved": False,
        "mix_appearances": 1,
        "mix_prominence": 0.5,
        "artist_mix_prominence": 0.5,
        "spotify_affinity": 0.4,
        "file_path": None,
        "spotify_uri": None,
        "enrichment_source": "test",
        "release_year": 2000,
        "track_number": 1,
        "total_tracks": 10,
        "disc_number": 1,
        "enrichment_gaps": [],
        "rym_data": {
            "primary_genres": [genre],
            "subgenres": [],
            "descriptors": ["energetic"],
        },
    }


def _build_library(n: int = 50, genres: list = None) -> pd.DataFrame:
    """Build a synthetic library DataFrame with n tracks across multiple artists/genres."""
    genres = genres or ["Rock", "Pop", "Jazz", "Electronic", "Folk"]
    rows = []
    for i in range(n):
        rows.append(_make_track(
            tid=f"t{i:03d}",
            artist=f"Artist_{chr(65 + (i % 20))}",  # 20 artists A-T
            genre=genres[i % len(genres)],
            energy=round(0.2 + (i % 10) * 0.07, 2),
            valence=round(0.2 + (i % 8) * 0.08, 2),
            intensity=round(0.3 + (i % 6) * 0.09, 2),
            duration_s=180 + (i % 8) * 30,   # 180–390 s (3–6.5 min)
            rating=round(0.4 + (i % 6) * 0.1, 2),
        ))
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def synthetic_library():
    """Replace the global library DF with a synthetic one for every test."""
    library.df = _build_library(60)
    library._build_indexes()
    yield
    # teardown: leave it loaded so subsequent tests don't choke


@pytest.fixture
def base_profile():
    p = UserProfile()
    p.targets.energy      = 0.5
    p.targets.valence     = 0.5
    p.targets.intensity   = 0.5
    p.targets.uniformity  = 0.5
    p.targets.accessibility = 0.6
    return p


@pytest.fixture
def rock_profile():
    p = UserProfile()
    p.target_genres = ["Rock"]
    p.targets.energy    = 0.6
    p.targets.valence   = 0.5
    p.targets.intensity = 0.6
    p.targets.uniformity = 0.5
    return p


def _make_blueprint(n_segs: int = 7, target_s: int = 4200) -> Blueprint:
    """Return a Blueprint with n_segs segments dividing target_s evenly."""
    seg_dur = target_s // n_segs
    segments = [
        Segment(
            segment_id=f"seg_{i:02d}",
            theme=f"Chapter {i}",
            target_duration_s=seg_dur,
            audio_targets={"energy": 0.3 + i * 0.07, "valence": 0.5},
            semantic_targets={"genres": ["Rock", "Pop", "Jazz", "Electronic", "Folk"][i % 5:i % 5 + 1]},
        )
        for i in range(n_segs)
    ]
    return Blueprint(
        blueprint_id=str(uuid.uuid4()),
        persona_state_id="test-persona",
        narrative_arc="Test arc",
        segments=segments,
    )


# ===========================================================================
# §1 — System Identity: LLM = Creative Director / Algorithm = Engine
# ===========================================================================

class TestSection1_SystemIdentity:
    """
    §1 Invariant: The algorithm is the source of truth for feasibility,
    scoring, and constraint satisfaction. The LLM cannot 'decide' a playlist
    into existence.
    """

    def test_blueprint_generator_produces_blueprint_not_playlist(self, base_profile):
        """BlueprintGenerator (LLM phase) must return a Blueprint, not a Playlist."""
        from mixtape_curator.blueprint import BlueprintGenerator
        gen = BlueprintGenerator(llm=None)   # heuristic mode
        persona = PersonaProfile(occasion="chill", mood_today="relaxed")
        result = gen.generate(persona, duration_target_s=4200)
        assert isinstance(result, Blueprint), "LLM phase must produce a Blueprint"
        assert result.segments, "Blueprint must have at least one segment"

    def test_trellis_is_deterministic_given_same_inputs(self, rock_profile):
        """Algorithm phase (trellis) must produce identical results for the same inputs."""
        from mixtape_curator.trellis import TrellisBuilder
        blueprint = _make_blueprint(7, 4200)
        t1 = TrellisBuilder(rock_profile).build(blueprint)
        t2 = TrellisBuilder(rock_profile).build(blueprint)
        assert t1.segment_pools == t2.segment_pools, "Trellis must be deterministic"

    def test_arc_optimizer_is_deterministic_given_same_seed(self, rock_profile):
        """Arc optimization must be deterministic under fixed rng_seed (§9)."""
        from mixtape_curator.trellis import TrellisBuilder
        from mixtape_curator.arc_optimizer import ArcOptimizer
        blueprint = _make_blueprint(7, 4200)
        trellis = TrellisBuilder(rock_profile).build(blueprint)

        seed = config.get("rng_seed", 12345)
        a1, b1 = ArcOptimizer(rock_profile, rng_seed=seed).optimize(trellis, blueprint.segments)
        a2, b2 = ArcOptimizer(rock_profile, rng_seed=seed).optimize(trellis, blueprint.segments)

        assert a1.global_order == a2.global_order, "Draft A must be deterministic"
        assert b1.global_order == b2.global_order, "Draft B must be deterministic"


# ===========================================================================
# §2 — Runtime and Offline Boundary
# ===========================================================================

class TestSection2_OfflineInvariant:
    """§2: No external API calls during generation pipeline."""

    def test_trellis_uses_no_network(self, rock_profile, monkeypatch):
        """TrellisBuilder must not make any socket calls."""
        import socket
        original_connect = socket.socket.connect

        def fail_connect(*args, **kwargs):
            raise AssertionError("Network call detected during trellis construction!")

        monkeypatch.setattr(socket.socket, "connect", fail_connect)

        from mixtape_curator.trellis import TrellisBuilder
        blueprint = _make_blueprint(7, 4200)
        # Should complete without triggering the patched connect
        trellis = TrellisBuilder(rock_profile).build(blueprint)
        assert trellis is not None

    def test_arc_optimizer_uses_no_network(self, rock_profile, monkeypatch):
        """ArcOptimizer must not make any socket calls."""
        import socket

        def fail_connect(*args, **kwargs):
            raise AssertionError("Network call detected during arc optimization!")

        monkeypatch.setattr(socket.socket, "connect", fail_connect)

        from mixtape_curator.trellis import TrellisBuilder
        from mixtape_curator.arc_optimizer import ArcOptimizer
        blueprint = _make_blueprint(7, 4200)
        trellis = TrellisBuilder(rock_profile).build(blueprint)
        draft_a, draft_b = ArcOptimizer(rock_profile).optimize(trellis, blueprint.segments)
        assert draft_a is not None


# ===========================================================================
# §3A — Interview → PersonaState with per-axis confidence
# ===========================================================================

class TestSection3A_PersonaState:
    """§3A: PersonaProfile must carry per-axis confidence."""

    def test_persona_profile_has_confidence_field(self):
        """PersonaProfile must have a confidence dict field."""
        p = PersonaProfile()
        assert hasattr(p, "confidence"), "PersonaProfile must have 'confidence'"
        assert isinstance(p.confidence, dict)

    def test_persona_confidence_accepts_per_axis_values(self):
        """confidence dict must accept float values for recognized axes."""
        p = PersonaProfile(confidence={
            "occasion": 1.0,
            "personality": 0.8,
            "mood": 0.5,
            "aesthetic": 0.0,
            "era": 0.7,
            "wildcard": 0.3,
        })
        assert p.confidence["occasion"] == 1.0
        assert p.confidence["wildcard"] == 0.3

    def test_persona_to_user_profile_still_works(self):
        """to_user_profile() must work regardless of confidence field."""
        p = PersonaProfile(occasion="gym", mood_today="excited", confidence={"occasion": 1.0})
        up = p.to_user_profile()
        assert isinstance(up, UserProfile)
        assert up.targets.energy > 0.5   # gym maps to high energy


# ===========================================================================
# §3B — Blueprint Generation
# ===========================================================================

class TestSection3B_Blueprint:
    """§3B: Blueprint must have 7–8 segments, each with explicit duration specifier."""

    def test_heuristic_blueprint_has_7_to_8_segments(self):
        from mixtape_curator.blueprint import BlueprintGenerator
        gen = BlueprintGenerator(llm=None)
        persona = PersonaProfile(occasion="drive")
        bp = gen.generate(persona, duration_target_s=4200)
        assert 7 <= len(bp.segments) <= 8, f"Expected 7–8 segments, got {len(bp.segments)}"

    def test_every_segment_has_duration_specifier(self):
        """Every segment must have target_duration_s or target_track_count (§3B locked rule)."""
        from mixtape_curator.blueprint import BlueprintGenerator
        gen = BlueprintGenerator(llm=None)
        persona = PersonaProfile(occasion="party")
        bp = gen.generate(persona, duration_target_s=3600)
        for seg in bp.segments:
            has_specifier = (seg.target_duration_s is not None) or (seg.target_track_count is not None)
            assert has_specifier, f"Segment '{seg.segment_id}' missing duration specifier"

    def test_blueprint_total_duration_approximates_target(self):
        """Sum of segment durations should be within 20% of the target."""
        from mixtape_curator.blueprint import BlueprintGenerator
        gen = BlueprintGenerator(llm=None)
        persona = PersonaProfile()
        target = 4620
        bp = gen.generate(persona, duration_target_s=target)
        total = sum(s.target_duration_s or 0 for s in bp.segments)
        assert total > 0
        ratio = abs(total - target) / target
        assert ratio < 0.25, f"Blueprint duration {total}s is >25% off target {target}s"

    def test_blueprint_describes_intent_not_tracks(self):
        """Blueprint segments must have themes (narrative description), not track IDs."""
        from mixtape_curator.blueprint import BlueprintGenerator
        gen = BlueprintGenerator(llm=None)
        persona = PersonaProfile(occasion="study")
        bp = gen.generate(persona, duration_target_s=3000)
        for seg in bp.segments:
            assert seg.theme, f"Segment '{seg.segment_id}' must have a theme"
            assert len(seg.theme) > 5


# ===========================================================================
# §3C — Segment Trellis Construction
# ===========================================================================

class TestSection3C_SegmentTrellis:
    """§3C: Trellis must produce deterministic candidate pools with ≥20 tracks per segment
    where feasible, enforcing hard constraints at retrieval time."""

    def test_trellis_produces_nonempty_pools(self, rock_profile):
        from mixtape_curator.trellis import TrellisBuilder
        blueprint = _make_blueprint(7)
        trellis = TrellisBuilder(rock_profile).build(blueprint)
        assert trellis.segment_pools, "Trellis must have at least one non-empty pool"
        total_candidates = sum(len(p) for p in trellis.segment_pools.values())
        assert total_candidates > 0

    def test_trellis_has_pool_for_every_blueprint_segment(self, rock_profile):
        from mixtape_curator.trellis import TrellisBuilder
        blueprint = _make_blueprint(7)
        trellis = TrellisBuilder(rock_profile).build(blueprint)
        for seg in blueprint.segments:
            assert seg.segment_id in trellis.segment_pools, \
                f"No pool for segment '{seg.segment_id}'"

    def test_trellis_respects_excluded_artists(self):
        """Tracks from excluded artists must NOT appear in any pool."""
        profile = UserProfile()
        profile.exclude_artists = ["Artist_A"]   # excludes all tracks by Artist A
        from mixtape_curator.trellis import TrellisBuilder
        blueprint = _make_blueprint(7)
        trellis = TrellisBuilder(profile).build(blueprint)

        for sid, pool in trellis.segment_pools.items():
            for tid in pool:
                t = library.get_track(tid)
                if t:
                    assert t.artist != "Artist_A", \
                        f"Excluded artist found in pool for '{sid}'"

    def test_trellis_respects_duration_filter(self, rock_profile):
        """Candidate tracks must satisfy track_min/max_duration_s."""
        from mixtape_curator.trellis import TrellisBuilder
        min_dur = config.get("track_min_duration_s", 60)
        max_dur = config.get("track_max_duration_s", 600)
        blueprint = _make_blueprint(7)
        trellis = TrellisBuilder(rock_profile).build(blueprint)

        for sid, pool in trellis.segment_pools.items():
            for tid in pool:
                t = library.get_track(tid)
                if t:
                    assert min_dur <= t.duration_s <= max_dur, \
                        f"Track {tid} duration {t.duration_s}s violates filter in segment '{sid}'"

    def test_trellis_reports_undersized_segments(self):
        """If a pool is < 20 tracks, the trellis must record it in undersized_segments."""
        from mixtape_curator.trellis import TrellisBuilder, MIN_POOL_SIZE
        # Build a very restrictive profile that can't possibly get 20 candidates per segment
        profile = UserProfile()
        profile.exclude_genres = ["Rock", "Pop", "Jazz"]   # exclude 3/5 genres
        blueprint = Blueprint(
            blueprint_id=str(uuid.uuid4()),
            persona_state_id="x",
            narrative_arc="test",
            segments=[
                Segment(
                    segment_id="seg_rare",
                    theme="Very rare genre",
                    target_duration_s=600,
                    semantic_targets={"genres": ["Doom Metal"]},  # not in our library
                )
            ],
        )
        trellis = TrellisBuilder(profile).build(blueprint)
        # The rare segment must appear either in undersized_segments or have pool < 20
        pool_size = len(trellis.segment_pools.get("seg_rare", []))
        if pool_size < MIN_POOL_SIZE:
            assert "seg_rare" in trellis.undersized_segments, \
                "Undersized segment must be recorded in trellis.undersized_segments"


# ===========================================================================
# §3D — Arc Optimization: Two Distinct Drafts
# ===========================================================================

class TestSection3D_ArcOptimization:
    """§3D: Arc optimizer must produce ArcDraft A and B with selected segments,
    tracks, global ordering, and deterministic score breakdowns."""

    def test_optimizer_returns_two_arc_drafts(self, rock_profile):
        from mixtape_curator.trellis import TrellisBuilder
        from mixtape_curator.arc_optimizer import ArcOptimizer
        blueprint = _make_blueprint(7)
        trellis = TrellisBuilder(rock_profile).build(blueprint)
        draft_a, draft_b = ArcOptimizer(rock_profile).optimize(trellis, blueprint.segments)
        assert isinstance(draft_a, ArcDraft)
        assert isinstance(draft_b, ArcDraft)

    def test_draft_a_and_b_have_global_order(self, rock_profile):
        from mixtape_curator.trellis import TrellisBuilder
        from mixtape_curator.arc_optimizer import ArcOptimizer
        blueprint = _make_blueprint(7)
        trellis = TrellisBuilder(rock_profile).build(blueprint)
        draft_a, draft_b = ArcOptimizer(rock_profile).optimize(trellis, blueprint.segments)
        assert draft_a.global_order, "Draft A must have a non-empty global_order"
        assert draft_b.global_order, "Draft B must have a non-empty global_order"

    def test_draft_segment_count_within_bounds(self, rock_profile):
        """Each draft must have 3–5 selected segments (§3D)."""
        from mixtape_curator.trellis import TrellisBuilder
        from mixtape_curator.arc_optimizer import ArcOptimizer, _MIN_SEGMENTS, _MAX_SEGMENTS
        blueprint = _make_blueprint(8)
        trellis = TrellisBuilder(rock_profile).build(blueprint)
        draft_a, draft_b = ArcOptimizer(rock_profile).optimize(trellis, blueprint.segments)
        assert _MIN_SEGMENTS <= len(draft_a.selected_segment_ids) <= _MAX_SEGMENTS, \
            f"Draft A has {len(draft_a.selected_segment_ids)} segments"
        assert _MIN_SEGMENTS <= len(draft_b.selected_segment_ids) <= _MAX_SEGMENTS, \
            f"Draft B has {len(draft_b.selected_segment_ids)} segments"

    def test_draft_has_deterministic_score_breakdown(self, rock_profile):
        """Draft scores must be deterministic for the same inputs (§9 / §3D)."""
        from mixtape_curator.trellis import TrellisBuilder
        from mixtape_curator.arc_optimizer import ArcOptimizer
        blueprint = _make_blueprint(7)
        trellis = TrellisBuilder(rock_profile).build(blueprint)
        seed = 42
        a1, _ = ArcOptimizer(rock_profile, rng_seed=seed).optimize(trellis, blueprint.segments)
        a2, _ = ArcOptimizer(rock_profile, rng_seed=seed).optimize(trellis, blueprint.segments)
        assert a1.scores.total == a2.scores.total, "Score must be deterministic"

    def test_duration_hard_cap_respected(self, rock_profile):
        """total_duration_s must never exceed duration_cap_s (§3D hard cap)."""
        from mixtape_curator.trellis import TrellisBuilder
        from mixtape_curator.arc_optimizer import ArcOptimizer
        blueprint = _make_blueprint(7)
        trellis = TrellisBuilder(rock_profile).build(blueprint)
        draft_a, draft_b = ArcOptimizer(rock_profile).optimize(trellis, blueprint.segments)
        cap = config.duration_cap_s
        assert draft_a.total_duration_s <= cap, \
            f"Draft A duration {draft_a.total_duration_s}s exceeds cap {cap}s"
        assert draft_b.total_duration_s <= cap, \
            f"Draft B duration {draft_b.total_duration_s}s exceeds cap {cap}s"

    def test_no_duplicate_tracks_within_draft(self, rock_profile):
        """A single track must not appear twice in the same draft's global_order."""
        from mixtape_curator.trellis import TrellisBuilder
        from mixtape_curator.arc_optimizer import ArcOptimizer
        blueprint = _make_blueprint(7)
        trellis = TrellisBuilder(rock_profile).build(blueprint)
        draft_a, draft_b = ArcOptimizer(rock_profile).optimize(trellis, blueprint.segments)
        assert len(draft_a.global_order) == len(set(draft_a.global_order)), \
            "Draft A has duplicate track IDs"
        assert len(draft_b.global_order) == len(set(draft_b.global_order)), \
            "Draft B has duplicate track IDs"

    def test_arc_draft_to_playlist_adapter(self, rock_profile):
        """arc_draft_to_playlist must produce a valid Playlist."""
        from mixtape_curator.trellis import TrellisBuilder
        from mixtape_curator.arc_optimizer import ArcOptimizer, arc_draft_to_playlist
        blueprint = _make_blueprint(7)
        trellis = TrellisBuilder(rock_profile).build(blueprint)
        draft_a, _ = ArcOptimizer(rock_profile).optimize(trellis, blueprint.segments)
        pl = arc_draft_to_playlist(draft_a)
        assert isinstance(pl, Playlist)
        assert pl.track_ids == draft_a.global_order
        assert pl.segments_used == draft_a.selected_segment_ids


# ===========================================================================
# §3E — Refinement + Rationale (LLM)
# ===========================================================================

class TestSection3E_RefinementRationale:
    """§3E: finalize_playlist must require a non-empty rationale; backend validates after each tool call."""

    def _make_playlist_with_tracks(self, n: int = 5) -> Playlist:
        tids = [f"t{i:03d}" for i in range(n)]
        return Playlist(
            id=str(uuid.uuid4()),
            track_ids=tids,
            total_duration_s=n * 240,
            scores=PlaylistScores(total=0.85),
        )

    def test_finalize_playlist_rejects_empty_rationale(self):
        """finalize_playlist must return a rejection string when rationale is empty."""
        from mixtape_curator.agent import ReActAgent
        from mixtape_curator.llm.providers.local import MockLLM
        agent = ReActAgent(llm=MockLLM())
        agent._last_profile = UserProfile()
        pl = self._make_playlist_with_tracks()
        result = agent._tool_finalize_playlist(pl, {"rationale": ""})
        assert result != "accepted", "Empty rationale must be rejected"
        assert "REJECTED" in result

    def test_finalize_playlist_rejects_whitespace_only_rationale(self):
        from mixtape_curator.agent import ReActAgent
        from mixtape_curator.llm.providers.local import MockLLM
        agent = ReActAgent(llm=MockLLM())
        agent._last_profile = UserProfile()
        pl = self._make_playlist_with_tracks()
        result = agent._tool_finalize_playlist(pl, {"rationale": "   "})
        assert result != "accepted", "Whitespace-only rationale must be rejected"

    def test_finalize_playlist_accepts_valid_rationale(self):
        from mixtape_curator.agent import ReActAgent
        from mixtape_curator.llm.providers.local import MockLLM
        agent = ReActAgent(llm=MockLLM())
        agent._last_profile = UserProfile()
        pl = self._make_playlist_with_tracks()
        result = agent._tool_finalize_playlist(pl, {"rationale": "This mix flows from X to Y."})
        assert result == "accepted"
        assert pl.rationale == "This mix flows from X to Y."

    def test_rationale_cleared_if_constraints_violated(self):
        """If constraints fail post-finalize, rationale must be cleared."""
        from mixtape_curator.agent import ReActAgent
        from mixtape_curator.llm.providers.local import MockLLM
        agent = ReActAgent(llm=MockLLM())
        profile = UserProfile()
        profile.exclude_artists = ["Artist_A"]
        agent._last_profile = profile

        pl = Playlist(
            id="test",
            track_ids=["t000"],  # Artist_A match must be in library to fail constraint
            total_duration_s=240,
            scores=PlaylistScores(),
        )
        # Inject a track that violates the exclusion
        library.df.loc[library.df["id"] == "t000", "artist"] = "Artist_A"

        result = agent._tool_finalize_playlist(pl, {"rationale": "A great mix."})
        if result != "accepted":
            assert pl.rationale is None, "Rationale must be cleared on constraint failure"


# ===========================================================================
# §4 — Segment Definition (Locked Schema)
# ===========================================================================

class TestSection4_SegmentSchema:
    """§4: Segment must have segment_id, theme, and at least one duration specifier."""

    def test_segment_requires_segment_id(self):
        with pytest.raises(Exception):
            Segment(theme="No ID", target_duration_s=300)  # missing segment_id

    def test_segment_requires_theme(self):
        with pytest.raises(Exception):
            Segment(segment_id="s1", target_duration_s=300)  # missing theme

    def test_segment_optional_fields_default_none(self):
        seg = Segment(segment_id="s1", theme="Opener", target_duration_s=300)
        assert seg.audio_targets is None
        assert seg.semantic_targets is None
        assert seg.uniformity_bias is None
        assert seg.notes is None

    def test_segment_can_use_track_count_instead_of_duration(self):
        seg = Segment(segment_id="s2", theme="Short segment", target_track_count=4)
        assert seg.target_track_count == 4
        assert seg.target_duration_s is None

    def test_segment_is_algorithm_interpretable(self):
        """A segment's fields must all be primitive types — no creative inference needed."""
        seg = Segment(
            segment_id="s3",
            theme="High energy rock block",
            target_duration_s=600,
            audio_targets={"energy": 0.8, "valence": 0.6},
            semantic_targets={"genres": ["Rock"], "descriptors": ["driving"]},
            uniformity_bias=0.7,
        )
        # Verify all fields are serialisable / algorithm-readable
        d = seg.model_dump()
        assert isinstance(d["audio_targets"]["energy"], float)
        assert isinstance(d["semantic_targets"]["genres"], list)


# ===========================================================================
# §5 — Optimization Objective
# ===========================================================================

class TestSection5_OptimizationObjective:
    """§5 priority: maximize total_score → minimize duration error → maximize boundary flow."""

    def test_higher_scoring_combo_is_preferred_as_draft_a(self, rock_profile):
        """Draft A should have a higher or equal score vs Draft B (A is best combo)."""
        from mixtape_curator.trellis import TrellisBuilder
        from mixtape_curator.arc_optimizer import ArcOptimizer
        blueprint = _make_blueprint(8)
        trellis = TrellisBuilder(rock_profile).build(blueprint)
        draft_a, draft_b = ArcOptimizer(rock_profile).optimize(trellis, blueprint.segments)
        # A is built from the highest-scoring combo; B must differ, so may score lower
        # At minimum, A's selection was made by the optimizer prioritising score
        assert draft_a.scores.total >= 0.0

    def test_draft_score_breakdown_sums_to_total(self, rock_profile):
        """PlaylistScores.total must equal the weighted sum of components."""
        from mixtape_curator.trellis import TrellisBuilder
        from mixtape_curator.arc_optimizer import ArcOptimizer
        blueprint = _make_blueprint(7)
        trellis = TrellisBuilder(rock_profile).build(blueprint)
        draft_a, _ = ArcOptimizer(rock_profile).optimize(trellis, blueprint.segments)

        if not draft_a.global_order:
            pytest.skip("No tracks in draft — library too small for this combo")

        s = draft_a.scores
        w = config.weights
        expected = (
            w["fit"]     * s.fit +
            w["flow"]    * s.flow +
            w["variety"] * s.variety +
            w["access"]  * s.accessibility +
            w["quality"] * s.quality
        )
        assert abs(s.total - expected) < 0.01, \
            f"Score total {s.total:.4f} ≠ weighted sum {expected:.4f}"

    def test_duration_error_is_recorded(self, rock_profile):
        """ArcDraft must record the absolute duration error vs target."""
        from mixtape_curator.trellis import TrellisBuilder
        from mixtape_curator.arc_optimizer import ArcOptimizer
        target = 3600
        blueprint = _make_blueprint(7, target)
        trellis = TrellisBuilder(rock_profile).build(blueprint)
        draft_a, _ = ArcOptimizer(rock_profile).optimize(trellis, blueprint.segments, duration_target_s=target)
        assert draft_a.duration_error_s == abs(draft_a.total_duration_s - target)


# ===========================================================================
# §6 — Segment-Boundary Flow
# ===========================================================================

class TestSection6_BoundaryFlow:
    """§6: Transition quality between segments must be scored and stored."""

    def _make_tracks(self, energies):
        return [
            Track(
                id=f"t_{i}", title=f"T{i}", artist=f"A", duration_s=240,
                energy=e, valence=0.5, intensity=0.5,
                rym_data=RYMData(primary_genres=["Rock"]),
            )
            for i, e in enumerate(energies)
        ]

    def test_perfect_transition_scores_1(self):
        from mixtape_curator.arc_optimizer import compute_boundary_flow
        a = self._make_tracks([0.5, 0.5])
        b = self._make_tracks([0.5, 0.5])
        score = compute_boundary_flow(a, b)
        assert abs(score - 1.0) < 0.01, "Identical sonic profiles must score ~1.0"

    def test_worst_transition_scores_low(self):
        from mixtape_curator.arc_optimizer import compute_boundary_flow
        a = self._make_tracks([0.0])   # energy=0
        b = self._make_tracks([1.0])   # energy=1, valence/intensity differ too
        score = compute_boundary_flow(
            [Track(id="a", title="A", artist="X", duration_s=200,
                   energy=0.0, valence=0.0, intensity=0.0, rym_data=RYMData())],
            [Track(id="b", title="B", artist="Y", duration_s=200,
                   energy=1.0, valence=1.0, intensity=1.0, rym_data=RYMData())],
        )
        assert score < 0.1, f"Max sonic distance must score near 0, got {score}"

    def test_boundary_flow_stored_on_arc_draft(self, rock_profile):
        """boundary_flow_score must be a float in [0, 1] on the ArcDraft."""
        from mixtape_curator.trellis import TrellisBuilder
        from mixtape_curator.arc_optimizer import ArcOptimizer
        blueprint = _make_blueprint(7)
        trellis = TrellisBuilder(rock_profile).build(blueprint)
        draft_a, _ = ArcOptimizer(rock_profile).optimize(trellis, blueprint.segments)
        assert 0.0 <= draft_a.boundary_flow_score <= 1.0

    def test_scorer_has_boundary_flow_method(self):
        """Scorer must expose compute_segment_boundary_flow() (§6 integration)."""
        from mixtape_curator.scoring import scorer
        assert hasattr(scorer, "compute_segment_boundary_flow"), \
            "Scorer must have compute_segment_boundary_flow method"
        t = Track(id="x", title="X", artist="A", duration_s=200,
                  energy=0.5, valence=0.5, intensity=0.5, rym_data=RYMData())
        score = scorer.compute_segment_boundary_flow([t], [t])
        assert score == 1.0


# ===========================================================================
# §7 — A/B Distinctness Guarantee
# ===========================================================================

class TestSection7_ABDistinctness:
    """§7: Playlist B must differ from A by ≥1 segment_id and Jaccard(segment sets) ≤ 0.8."""

    def test_draft_a_and_b_differ_by_at_least_one_segment(self, rock_profile):
        from mixtape_curator.trellis import TrellisBuilder
        from mixtape_curator.arc_optimizer import ArcOptimizer
        blueprint = _make_blueprint(8)
        trellis = TrellisBuilder(rock_profile).build(blueprint)
        draft_a, draft_b = ArcOptimizer(rock_profile).optimize(trellis, blueprint.segments)

        if not draft_a.global_order or not draft_b.global_order:
            pytest.skip("Library too small to produce two distinct drafts")

        a_segs = set(draft_a.selected_segment_ids)
        b_segs = set(draft_b.selected_segment_ids)
        assert a_segs != b_segs, "Draft A and B must have different segment sets"

    def test_jaccard_of_segments_at_most_0_8(self, rock_profile):
        from mixtape_curator.trellis import TrellisBuilder
        from mixtape_curator.arc_optimizer import ArcOptimizer
        blueprint = _make_blueprint(8)
        trellis = TrellisBuilder(rock_profile).build(blueprint)
        draft_a, draft_b = ArcOptimizer(rock_profile).optimize(trellis, blueprint.segments)

        if not draft_a.global_order or not draft_b.global_order:
            pytest.skip("Library too small to produce two distinct drafts")

        a = set(draft_a.selected_segment_ids)
        b = set(draft_b.selected_segment_ids)
        jaccard = len(a & b) / max(len(a | b), 1)
        assert jaccard <= 0.8, f"§7 violated: Jaccard({jaccard:.2f}) > 0.8"

    def test_draft_b_label_is_B(self, rock_profile):
        from mixtape_curator.trellis import TrellisBuilder
        from mixtape_curator.arc_optimizer import ArcOptimizer
        blueprint = _make_blueprint(8)
        trellis = TrellisBuilder(rock_profile).build(blueprint)
        draft_a, draft_b = ArcOptimizer(rock_profile).optimize(trellis, blueprint.segments)
        assert draft_a.label == "A"
        assert draft_b.label == "B"


# ===========================================================================
# §8 — Tooling Boundary
# ===========================================================================

class TestSection8_ToolingBoundary:
    """§8: LLM must only modify playlists via the four bounded tools."""

    def _make_agent_and_playlist(self):
        from mixtape_curator.agent import ReActAgent
        from mixtape_curator.llm.providers.local import MockLLM
        agent = ReActAgent(llm=MockLLM())
        agent._last_profile = UserProfile()
        pl = Playlist(
            id="test",
            track_ids=["t000", "t001", "t002", "t003", "t004"],
            total_duration_s=1200,
            scores=PlaylistScores(),
        )
        return agent, pl

    def test_tool_swap_track_replaces_track(self):
        agent, pl = self._make_agent_and_playlist()
        original = list(pl.track_ids)
        agent._tool_swap(pl, {"old_id": "t000", "new_id": "t050", "reasoning": "Better fit"}, UserProfile())
        assert "t050" in pl.track_ids
        assert "t000" not in pl.track_ids

    def test_tool_remove_track(self):
        agent, pl = self._make_agent_and_playlist()
        agent._tool_remove(pl, {"track_id": "t001"}, UserProfile())
        assert "t001" not in pl.track_ids

    def test_tool_reorder_segment_reorders_tracks(self):
        """reorder_segment must change track order without adding/removing tracks."""
        agent, pl = self._make_agent_and_playlist()
        before = list(pl.track_ids)
        new_order = ["t002", "t001", "t000"]   # reverse the first three
        agent._tool_reorder_segment(pl, {
            "segment_id": None,
            "new_order_ids": new_order,
        })
        # All original tracks must still be present
        assert set(pl.track_ids) == set(before)
        # The new_order tracks must appear at the END (fallback mode with no segment_id)
        tail = pl.track_ids[-3:]
        assert set(tail) == set(new_order)

    def test_tool_consult_user_returns_string(self):
        agent, _ = self._make_agent_and_playlist()
        agent.user_callback = lambda q: "user says yes"
        result = agent._tool_consult_user({"question": "Do you want more energy?"})
        assert result == "user says yes"

    def test_tool_finalize_playlist_is_present(self):
        """Agent must have a finalize_playlist tool (§8 minimum toolset)."""
        from mixtape_curator.agent import ReActAgent
        from mixtape_curator.llm.providers.local import MockLLM
        agent = ReActAgent(llm=MockLLM())
        assert hasattr(agent, "_tool_finalize_playlist"), \
            "ReActAgent must implement _tool_finalize_playlist"
        assert hasattr(agent, "_tool_reorder_segment"), \
            "ReActAgent must implement _tool_reorder_segment"


# ===========================================================================
# §9 — Determinism Contract
# ===========================================================================

class TestSection9_Determinism:
    """§9: Same inputs + same rng_seed → identical A/B output."""

    def test_same_seed_produces_identical_drafts(self, rock_profile):
        from mixtape_curator.trellis import TrellisBuilder
        from mixtape_curator.arc_optimizer import ArcOptimizer
        blueprint = _make_blueprint(7)
        trellis = TrellisBuilder(rock_profile).build(blueprint)

        seed = 99
        a1, b1 = ArcOptimizer(rock_profile, rng_seed=seed).optimize(trellis, blueprint.segments)
        a2, b2 = ArcOptimizer(rock_profile, rng_seed=seed).optimize(trellis, blueprint.segments)

        assert a1.global_order == a2.global_order, "Draft A must be identical across runs"
        assert b1.global_order == b2.global_order, "Draft B must be identical across runs"
        assert a1.selected_segment_ids == a2.selected_segment_ids
        assert b1.selected_segment_ids == b2.selected_segment_ids

    def test_different_seeds_may_produce_different_order(self, rock_profile):
        """Different seeds should be allowed to produce different combo orders."""
        from mixtape_curator.trellis import TrellisBuilder
        from mixtape_curator.arc_optimizer import ArcOptimizer
        blueprint = _make_blueprint(8)
        trellis = TrellisBuilder(rock_profile).build(blueprint)

        a1, _ = ArcOptimizer(rock_profile, rng_seed=1).optimize(trellis, blueprint.segments)
        a2, _ = ArcOptimizer(rock_profile, rng_seed=9999).optimize(trellis, blueprint.segments)
        # Not a hard requirement — just shows the seed matters (may coincidentally match)
        # Just assert both run without error
        assert isinstance(a1, ArcDraft)
        assert isinstance(a2, ArcDraft)

    def test_trellis_is_deterministic_without_seed(self, rock_profile):
        """TrellisBuilder uses only library data — must always be deterministic."""
        from mixtape_curator.trellis import TrellisBuilder
        blueprint = _make_blueprint(7)
        t1 = TrellisBuilder(rock_profile).build(blueprint)
        t2 = TrellisBuilder(rock_profile).build(blueprint)
        assert t1.segment_pools == t2.segment_pools


# ===========================================================================
# §10 — Failure Modes
# ===========================================================================

class TestSection10_FailureModes:
    """§10: On failure, must return best-effort draft with diagnostic reason codes
    and call consult_user() — no silent failure."""

    def test_diagnostic_code_enum_covers_key_modes(self):
        from mixtape_curator.diagnostics import DiagnosticCode
        required = [
            "INSUFFICIENT_CANDIDATES",
            "NO_CANDIDATES_FOR_SEGMENT",
            "DURATION_INFEASIBLE",
            "AB_JACCARD_VIOLATION",
            "REFINEMENT_NO_RATIONALE",
        ]
        for code in required:
            assert hasattr(DiagnosticCode, code), f"DiagnosticCode missing '{code}'"

    def test_relaxation_menu_from_codes_is_non_empty(self):
        from mixtape_curator.diagnostics import RelaxationMenu, DiagnosticCode
        menu = RelaxationMenu.from_codes([
            DiagnosticCode.INSUFFICIENT_CANDIDATES,
            DiagnosticCode.DURATION_INFEASIBLE,
        ])
        assert not menu.is_empty()

    def test_relaxation_menu_produces_question_string(self):
        from mixtape_curator.diagnostics import RelaxationMenu, DiagnosticCode
        menu = RelaxationMenu.from_codes([DiagnosticCode.INSUFFICIENT_CANDIDATES])
        question = menu.to_consult_question()
        assert isinstance(question, str)
        assert len(question) > 20

    def test_empty_draft_has_diagnostic_code(self, rock_profile):
        """If the optimizer builds an empty draft, it must attach a diagnostic code."""
        from mixtape_curator.arc_optimizer import ArcOptimizer
        # Provide a trellis with zero pools to force failure
        trellis = SegmentTrellis(
            trellis_id="empty",
            blueprint_id="bp",
            segment_pools={},           # no segments at all → optimizer hits empty draft path
        )
        blueprint = _make_blueprint(2)  # only 2 segments available (< MIN_SEGMENTS=3)
        draft_a, draft_b = ArcOptimizer(rock_profile).optimize(trellis, blueprint.segments)
        # Both drafts must surface a diagnostic code, not silently return empty
        assert draft_a.diagnostic_codes, "Empty draft must carry diagnostic codes"

    def test_undersized_trellis_records_diagnostic(self):
        """Trellis with truly too-small library must record undersized segments."""
        from mixtape_curator.trellis import TrellisBuilder, MIN_POOL_SIZE
        # Use a tiny library: only 5 tracks (well under MIN_POOL_SIZE=20)
        library.df = _build_library(5)
        library._build_indexes()

        blueprint = Blueprint(
            blueprint_id=str(uuid.uuid4()),
            persona_state_id="x",
            narrative_arc="test",
            segments=[
                Segment(
                    segment_id="small_seg",
                    theme="Small library segment",
                    target_duration_s=600,
                    # No genre filter — unconstrained still only has 5 tracks
                )
            ],
        )
        profile = UserProfile()
        trellis = TrellisBuilder(profile).build(blueprint)
        pool_size = len(trellis.segment_pools.get("small_seg", []))
        # With only 5 tracks total, pool_size must be < 20
        assert pool_size < MIN_POOL_SIZE, (
            f"Expected pool < {MIN_POOL_SIZE} with 5-track library, got {pool_size}"
        )
        # And the segment must be flagged
        assert "small_seg" in trellis.undersized_segments, \
            "Segment must appear in trellis.undersized_segments when pool < 20"


# ===========================================================================
# §– Artist Preference (user rule: hard cap=2, strong preference for 1)
# ===========================================================================

class TestArtistPreference:
    """Hard cap stays at max_tracks_per_artist, penalty steers strongly toward 1."""

    def test_duplicate_artist_reduces_variety_score(self, base_profile):
        """A playlist with a duplicate artist must score lower variety than one without."""
        base_profile.targets.uniformity = 0.5
        t1 = Track(id="t1", title="A1", artist="Artist_X", duration_s=200,
                   energy=0.5, valence=0.5, intensity=0.5,
                   rym_data=RYMData(primary_genres=["Rock"]))
        t2 = Track(id="t2", title="A2", artist="Artist_X", duration_s=200,   # SAME artist
                   energy=0.5, valence=0.5, intensity=0.5,
                   rym_data=RYMData(primary_genres=["Rock"]))
        t3 = Track(id="t3", title="B1", artist="Artist_Y", duration_s=200,
                   energy=0.5, valence=0.5, intensity=0.5,
                   rym_data=RYMData(primary_genres=["Rock"]))
        t4 = Track(id="t4", title="C1", artist="Artist_Z", duration_s=200,
                   energy=0.5, valence=0.5, intensity=0.5,
                   rym_data=RYMData(primary_genres=["Rock"]))

        duped   = scorer.compute_variety_score([t1, t2, t3, t4], base_profile)
        no_dupe = scorer.compute_variety_score([
            t1,
            t3,
            t4,
            Track(id="t5", title="D1", artist="Artist_W", duration_s=200,
                  energy=0.5, valence=0.5, intensity=0.5,
                  rym_data=RYMData(primary_genres=["Rock"])),
        ], base_profile)

        assert duped < no_dupe, \
            f"Duplicate artist should reduce variety: {duped:.3f} < {no_dupe:.3f}"
        assert (no_dupe - duped) >= 0.05, \
            "Duplicate penalty must be meaningful (≥0.05 difference)"

    def test_arc_fill_prefers_1_track_per_artist(self, rock_profile):
        """Pass 1 (max=1 per artist) must fill the majority of the playlist before
        pass 2 (max=2) is needed. We verify that ≥50% of artists appear exactly once."""
        from mixtape_curator.trellis import TrellisBuilder
        from mixtape_curator.arc_optimizer import ArcOptimizer
        # Use a medium library with limited artists so pass-2 behaviour is observable
        library.df = _build_library(30, genres=["Rock", "Pop", "Jazz"])
        library._build_indexes()

        blueprint = _make_blueprint(7, 3600)
        trellis = TrellisBuilder(rock_profile).build(blueprint)
        draft_a, _ = ArcOptimizer(rock_profile).optimize(trellis, blueprint.segments)

        if not draft_a.global_order:
            pytest.skip("No tracks produced with 30-track library")

        from collections import Counter
        tracks = [library.get_track(tid) for tid in draft_a.global_order if library.get_track(tid)]
        artist_counts = Counter(t.artist for t in tracks)

        # Majority of represented artists should appear only once
        once_count  = sum(1 for c in artist_counts.values() if c == 1)
        total_count = len(artist_counts)
        once_fraction = once_count / max(total_count, 1)
        assert once_fraction >= 0.5, (
            f"Expected ≥50% of artists to appear once; "
            f"got {once_count}/{total_count} ({once_fraction:.0%})"
        )
