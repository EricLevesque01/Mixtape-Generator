import pytest
from mixtape_curator.agent import ReActAgent
from mixtape_curator.llm.providers.local import MockLLM
from mixtape_curator.models import Playlist, UserProfile, PlaylistScores
from mixtape_curator.library import library
import pandas as pd

@pytest.fixture(autouse=True)
def mock_setup():
    # Setup a library with enough tracks for the agent to work with
    tracks = []
    for i in range(15):
        tracks.append({
            "id": f"t{i}", "title": f"Track {i}", "artist": f"Artist {chr(65+i)}",
            "duration_s": 300, "energy": 0.5, "valence": 0.5, "intensity": 0.5,
            "rym_data_primary_genres": ["Pop"], "rym_data_subgenres": [], "rym_data_descriptors": []
        })
    library.df = pd.DataFrame(tracks)
    library._build_indexes()

def test_repair_loop_runs():
    """Test that the agent repair loop completes without errors."""
    pl = Playlist(id="test", track_ids=["t0", "t1", "t2"], total_duration_s=900, scores=PlaylistScores())
    profile = UserProfile()
    profile.target_genres = ["Pop"]
    
    agent = ReActAgent(llm=MockLLM())
    repaired = agent.repair_playlist(pl, profile)
    
    # The agent should produce a result — it may add tracks (growth phase)
    assert repaired is not None
    assert isinstance(repaired, Playlist)
    # Should have at least the original tracks (may grow depending on MockLLM)
    assert len(repaired.track_ids) >= 3

def test_constraint_validation():
    """Test that constraint validation detects violations."""
    # Playlist exceeding duration cap
    pl = Playlist(id="test", track_ids=["t0"], total_duration_s=8000, scores=PlaylistScores())
    profile = UserProfile()
    
    agent = ReActAgent(llm=MockLLM())
    violations = agent._validate_constraints(pl, profile)
    
    assert any("Duration" in v for v in violations)

def test_artist_limit_constraint():
    """Test max tracks per artist constraint."""
    # Make t0, t1, t2 same artist
    tracks = []
    for i in range(5):
        tracks.append({
            "id": f"t{i}", "title": f"Track {i}", "artist": "Same Artist",
            "duration_s": 300, "energy": 0.5, "valence": 0.5, "intensity": 0.5,
            "rym_data_primary_genres": ["Pop"], "rym_data_subgenres": [], "rym_data_descriptors": []
        })
    library.df = pd.DataFrame(tracks)
    library._build_indexes()
    
    pl = Playlist(id="test", track_ids=["t0", "t1", "t2"], total_duration_s=900, scores=PlaylistScores())
    profile = UserProfile()
    
    agent = ReActAgent(llm=MockLLM())
    violations = agent._validate_constraints(pl, profile)
    
    # 3 tracks from "Same Artist" > max of 2
    assert any("Same Artist" in v for v in violations)
