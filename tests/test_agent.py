import pytest
from src.mixtape_curator.agent import ReActAgent, MockLLM
from src.mixtape_curator.models import Playlist, UserProfile, PlaylistScores
from src.mixtape_curator.config import config
from src.mixtape_curator.library import library
import pandas as pd

@pytest.fixture(autouse=True)
def mock_setup():
    # Setup track t0 with 1 hour duration to force violation tests easier
    tracks = [{"id": "t0", "title": "Long Song", "artist": "A", "duration_s": 7300, "energy":0.5, "valence":0.5, "intensity":0.5}]
    library.df = pd.DataFrame(tracks)

def test_repair_loop_duration_constraint():
    # Playlist with 7300s > 7200s cap
    pl = Playlist(id="test", track_ids=["t0"], total_duration_s=7300, scores=PlaylistScores())
    profile = UserProfile()
    
    agent = ReActAgent(llm=MockLLM())
    
    # Run repair
    # MockLLM is hardcoded to call 'remove_track' if it sees "Duration"
    repaired = agent.repair_playlist(pl, profile)
    
    # Should have removed t0
    assert "t0" not in repaired.track_ids
    # And technically duration should update if we ran re-score, 
    # but our mock agent logic in _tool_remove is simple list op.
    # The agent loop calls generator.optimize_flow which effectively re-syncs state.
    # But since t0 is removed, list is empty.
    assert len(repaired.track_ids) == 0
