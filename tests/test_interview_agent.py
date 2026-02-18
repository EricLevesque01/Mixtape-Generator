
import unittest
from unittest.mock import MagicMock
from mixtape_curator.interview_agent import InterviewAgent
from mixtape_curator.models import UserProfile

class TestInterviewAgent(unittest.TestCase):
    def test_transparency_callback(self):
        """Verify thoughts are streamed to callback."""
        agent = InterviewAgent()
        thoughts = []
        
        def mock_callback(t):
            thoughts.append(t)
            
        agent.process_input("For a party", user_callback=mock_callback)
        
        # Check if we got thoughts
        self.assertTrue(len(thoughts) > 0)
        self.assertIn("[Thought]", thoughts[0])
        
    def test_sufficiency_check(self):
        """Verify agent continues until profile is sufficient."""
        agent = InterviewAgent()
        
        # 1. Provide only recipient
        resp, done = agent.process_input("It's for my friend")
        self.assertFalse(done)
        self.assertEqual(agent.profile.recipient, "Friend")
        
        # 2. Provide genre/vibe
        resp, done = agent.process_input("She likes upbeat pop music")
        self.assertTrue(done) # Should be sufficient now (Recipient + Genre + Vibe inferred)
        self.assertIn("pop", agent.profile.target_genres)
        self.assertTrue(agent.profile.targets.valence >= 0.6) # "upbeat" -> happy
        
    def test_mock_extraction(self):
        """Verify basic entity extraction (mock logic)."""
        agent = InterviewAgent()
        agent.process_input("I want a slow jazz mix for studying")
        
        self.assertIn("jazz", agent.profile.target_genres)
        self.assertEqual(agent.profile.targets.energy, 0.3) # "slow"

if __name__ == '__main__':
    unittest.main()
