import pytest
from src.mixtape_curator.generator import generator
from src.mixtape_curator.models import UserProfile, Track, RYMData
from src.mixtape_curator.library import library
from src.mixtape_curator.config import config
import pandas as pd

# Mock Data Injection for Tests
@pytest.fixture(autouse=True)
def mock_library_data():
    # Create 10 fake tracks
    tracks = []
    for i in range(10):
        tracks.append({
            "id": f"t{i}",
            "title": f"Track {i}",
            "artist": "Artist A" if i < 5 else "Artist B",
            "duration_s": 300, # 5 mins
            "energy": 0.1 * i,
            "valence": 0.1 * i,
            "intensity": 0.5,
            "rym_data_primary_genres": ["Pop"],
            "rym_data_subgenres": [],
            "rym_data_descriptors": []
        })
    library.df = pd.DataFrame(tracks)

def test_draft_constraints():
    profile = UserProfile()
    # Exclude Artist A
    profile.exclude_artists = ["Artist A"]
    
    pl = generator.create_draft(profile)
    
    # Check exclusion
    # Artist A was t0-t4. Artist B was t5-t9.
    for tid in pl.track_ids:
        # We need to map back to artist to verify, but simple ID check works for this mock:
        assert int(tid[1:]) >= 5 # Should only have tracks 5-9

def test_draft_duration_target():
    profile = UserProfile()
    # Config target is 4620s. 
    # Mock tracks are 300s.
    # It should fit ~15 tracks. We only have 10.
    # So it should take all 10.
    pl = generator.create_draft(profile)
    assert len(pl.track_ids) == 10
    assert pl.total_duration_s == 3000

def test_sequencing_improvement():
    profile = UserProfile()
    # Create a playlist with jagged energy: 0.0, 0.9, 0.1, 0.8...
    # The optimizer should smooth this.
    
    # Manually forcing ids for the test
    # t0 (0.0), t9 (0.9), t1 (0.1), t8 (0.8)
    
    from src.mixtape_curator.models import Playlist, PlaylistScores
    pl = Playlist(
        id="test",
        track_ids=["t0", "t9", "t1", "t8"],
        track_scores=PlaylistScores()
    )
    
    # Score before
    from src.mixtape_curator.scoring import scorer
    tracks_before = [library.get_track(tid) for tid in pl.track_ids]
    score_before = scorer.compute_flow_score(tracks_before)
    
    # Optimize
    pl_opt = generator.optimize_flow(pl, profile)
    
    # Score after
    tracks_after = [library.get_track(tid) for tid in pl_opt.track_ids]
    score_after = scorer.compute_flow_score(tracks_after)
    
    # Flow score should be higher (impliying lower distance/error)
    # Note depending on the greedy start, it might find 0->1 or 9->8 pairs.
    assert score_after >= score_before
