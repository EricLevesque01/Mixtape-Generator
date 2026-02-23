import pytest
from mixtape_curator.ab_test import ab_tester
from mixtape_curator.models import UserProfile, Playlist, PlaylistScores
from mixtape_curator.library import library
import pandas as pd

# Reuse mock data setup
@pytest.fixture(autouse=True)
def mock_library_data():
    tracks = []
    for i in range(20): # Need enough for pool
        tracks.append({
            "id": f"t{i}",
            "title": f"Track {i}",
            "artist": f"Artist {chr(65 + i % 10)}",  # 10 different artists (A-J)
            "duration_s": 300,
            "energy": 0.5,
            "valence": 0.5,
            "intensity": 0.5,
            "rym_data_primary_genres": ["Pop"],
            "rym_data_subgenres": [],
            "rym_data_descriptors": []
        })
    library.df = pd.DataFrame(tracks)

def test_b_side_swaps():
    profile = UserProfile()
    # Create valid Playlist A with 10 tracks (ids t0-t9)
    pl_a = Playlist(
        id="test_A",
        track_ids=[f"t{i}" for i in range(10)],
        total_duration_s=3000,
        scores=PlaylistScores()
    )
    
    # Generate B
    pl_b = ab_tester.generate_b_side(pl_a, profile)
    
    # Verify strict difference
    # 20% of 10 tracks = 2 tracks swapped.
    # Config clamped min=2.
    
    set_a = set(pl_a.track_ids)
    set_b = set(pl_b.track_ids)
    
    diff = set_a - set_b
    assert len(diff) >= 2 # At least 2 tracks removed
    assert len(set_b) == 10 # Length maintained
