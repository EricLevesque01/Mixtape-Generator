import pytest
from mixtape_curator.generator import generator
from mixtape_curator.models import UserProfile, Playlist, PlaylistScores
from mixtape_curator.library import library
import pandas as pd

# Mock Data Injection for Tests
@pytest.fixture(autouse=True)
def mock_library_data():
    # Create 10 fake tracks with diverse artists to avoid max_tracks_per_artist limit
    tracks = []
    for i in range(10):
        tracks.append({
            "id": f"t{i}",
            "title": f"Track {i}",
            "artist": f"Artist {chr(65 + i)}",  # Artist A through J (unique artists)
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
    # Exclude Artist A (only t0)
    profile.exclude_artists = ["Artist A"]
    
    pl = generator.create_draft(profile)
    
    # Check exclusion - t0 has "Artist A", all others are different
    for tid in pl.track_ids:
        assert tid != "t0"

def test_draft_duration_target():
    profile = UserProfile()
    # Config target is 4620s. 
    # Mock tracks are 300s each, all unique artists.
    # It should take all 10 (3000s < 4620s target).
    pl = generator.create_draft(profile)
    assert len(pl.track_ids) == 10
    assert pl.total_duration_s == 3000

def test_sequencing_improvement():
    profile = UserProfile()
    
    pl = Playlist(
        id="test",
        track_ids=["t0", "t9", "t1", "t8"],
        scores=PlaylistScores()
    )
    
    # Score before
    from mixtape_curator.scoring import scorer
    tracks_before = [library.get_track(tid) for tid in pl.track_ids]
    score_before = scorer.compute_flow_score(tracks_before)
    
    # Optimize
    pl_opt = generator.optimize_flow(pl, profile)
    
    # Score after
    tracks_after = [library.get_track(tid) for tid in pl_opt.track_ids]
    score_after = scorer.compute_flow_score(tracks_after)
    
    # Flow score should be higher (implying lower distance/error)
    assert score_after >= score_before
