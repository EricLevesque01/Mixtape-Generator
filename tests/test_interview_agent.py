import unittest
import pandas as pd
from mixtape_curator.interview_agent import InterviewAgent
from mixtape_curator.library import library

class TestInterviewAgent(unittest.TestCase):
    def setUp(self):
        # Ensure library has data so _get_library_snapshot doesn't crash on empty DF
        if library.df.empty:
            library.df = pd.DataFrame([
                {"id": "t1", "title": "Party Rock", "artist": "LMFAO", "duration_s": 200,
                 "energy": 0.9, "valence": 0.8, "intensity": 0.7,
                 "rym_data_primary_genres": ["Pop"], "rym_data_subgenres": [], "rym_data_descriptors": []},
                {"id": "t2", "title": "Jazz Cafe", "artist": "Miles Davis", "duration_s": 300,
                 "energy": 0.3, "valence": 0.4, "intensity": 0.2,
                 "rym_data_primary_genres": ["Jazz"], "rym_data_subgenres": [], "rym_data_descriptors": []},
            ])

    def test_transparency_callback(self):
        """Verify thoughts are streamed to callback."""
        agent = InterviewAgent()
        thoughts = []
        
        def mock_callback(t):
            thoughts.append(t)
            
        agent.process_input("For a party", user_callback=mock_callback)
        
        # Check if we got any callback invocations
        self.assertTrue(len(thoughts) > 0)
        # The callback should contain some kind of thought/status message
        self.assertTrue(any("[" in t for t in thoughts))
        
    def test_sufficiency_check(self):
        """Verify agent continues until profile is sufficient."""
        agent = InterviewAgent()
        
        # 1. Provide only recipient
        resp, done = agent.process_input("It's for my friend")
        self.assertFalse(done)
        
        # 2. Provide genre/vibe
        resp, done = agent.process_input("She likes upbeat pop music")
        # Should have extracted genres
        self.assertTrue(len(agent.profile.target_genres) > 0 or agent.completed)
        
    def test_mock_extraction(self):
        """Verify basic entity extraction (mock logic)."""
        agent = InterviewAgent()
        agent.process_input("I want a slow jazz mix for studying")
        
        self.assertIn("jazz", agent.profile.target_genres)
        self.assertEqual(agent.profile.targets.energy, 0.3) # "slow"

if __name__ == '__main__':
    unittest.main()
