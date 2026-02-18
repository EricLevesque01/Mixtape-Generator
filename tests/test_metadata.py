
import unittest
import unittest
import pandas as pd
from unittest.mock import MagicMock
from mixtape_curator.models import Track, UserProfile
from mixtape_curator.library import Library, library
from mixtape_curator.agent import ReActAgent

class TestMetadata(unittest.TestCase):
    def setUp(self):
        # Reset library
        library.df = pd.DataFrame()
        
    def test_schema_updates(self):
        """Verify new metadata fields exist."""
        t = Track(
            id="t1", title="Test", artist="  The Test Artist  ", duration_s=60,
            release_year=1999
        )
        self.assertEqual(t.release_year, 1999)
        self.assertEqual(t.normalized_artist, "the test artist")
        
        # Test default is None
        t2 = Track(id="t2", title="Test2", artist="Test", duration_s=10)
        self.assertIsNone(t2.release_year)

    def test_fuzzy_search(self):
        """Verify library.search_artist finds fuzzy matches."""
        # Setup mock library content
        data = [
            {"id": "t1", "title": "Song", "artist": "AC/DC"},
            {"id": "t2", "title": "Other", "artist": "The Beatles"},
             {"id": "t3", "title": "Foo", "artist": "Foo Fighters"}
        ]
        library.df = pd.DataFrame(data)
        
        # Exact match
        self.assertEqual(library.search_artist("AC/DC"), "AC/DC")
        # Diff case
        self.assertEqual(library.search_artist("ac/dc"), "AC/DC")
        # Fuzzy (missing slash)
        self.assertEqual(library.search_artist("acdc"), "AC/DC")
        # Fuzzy (The vs no The)
        self.assertEqual(library.search_artist("beatles"), "The Beatles")
        
    def test_agent_search_grounding(self):
        """Verify agent uses library grounding."""
        # Setup mock
        data = [
            {"id": "t1", "title": "Back in Black", "artist": "AC/DC", "rym_data_primary_genres": ["Rock"]},
            {"id": "t2", "title": "Yesterday", "artist": "The Beatles", "rym_data_primary_genres": ["Pop"]}
        ]
        library.df = pd.DataFrame(data)
        
        agent = ReActAgent(llm=MagicMock())
        
        # Search "acdc" -> Should return AC/DC tracks
        res = agent._tool_search_library({"query": "acdc"}, UserProfile())
        self.assertIn("Back in Black", res)
        self.assertIn("Found fuzzy match for artist 'AC/DC'", res)
        
        # Search "yesterday" (Broad match, no artist fuzzy match)
        res = agent._tool_search_library({"query": "yesterday"}, UserProfile())
        self.assertIn("Yesterday", res)
        self.assertNotIn("Found fuzzy match", res)

if __name__ == '__main__':
    unittest.main()
