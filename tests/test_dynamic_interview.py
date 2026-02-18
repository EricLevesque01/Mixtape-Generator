
import unittest
from mixtape_curator.interview import Interviewer, InterviewState

class TestDynamicInterview(unittest.TestCase):
    def test_standard_flow(self):
        """Verify the standard linear path."""
        i = Interviewer()
        self.assertEqual(i.state, InterviewState.INTAKE_RECIPIENT)
        
        # 1. Recipient
        resp, done = i.process_input("For me")
        self.assertIn("context or vibe", resp)
        self.assertEqual(i.state, InterviewState.INTAKE_CONTEXT)
        
        # 2. Context
        resp, done = i.process_input("Chilling at home")
        self.assertIn("MUST include", resp)
        self.assertEqual(i.state, InterviewState.INTAKE_CONSTRAINTS)
        
        # 3. Constraints
        resp, done = i.process_input("No Nickelback")
        self.assertIn("Nickelback", i.profile.exclude_artists)
        self.assertEqual(i.state, InterviewState.CALIBRATION_GENRE)

    def test_fast_track_skip(self):
        """Verify skipping Context question if intent is clear."""
        i = Interviewer()
        
        # User provides context ("workout") in first answer
        resp, done = i.process_input("High intensity workout mix for me")
        
        # Should skip to Constraints, skipping "What's the context?"
        self.assertIn("MUST include", resp) 
        self.assertEqual(i.state, InterviewState.INTAKE_CONSTRAINTS)
        self.assertEqual(i.profile.context_notes, "High intensity workout mix for me")

    def test_clarification_loop(self):
        """Verify clarification question on vague input."""
        i = Interviewer()
        i.state = InterviewState.CALIBRATION_COHESION
        
        # Vague answer
        resp, done = i.process_input("idk whatever")
        
        # Should remain in Cohesion state and ask specific follow-up
        self.assertIn("To clarify", resp)
        self.assertEqual(i.state, InterviewState.CALIBRATION_COHESION)
        self.assertEqual(i.confidence["cohesion"], 0.2)
        
        # Valid answer resolves it
        resp, done = i.process_input("uniform please")
        self.assertEqual(i.state, InterviewState.CALIBRATION_ENERGY)
        self.assertEqual(i.profile.targets.uniformity, 0.9)

    def test_global_entity_extraction(self):
        """Verify artists mentioned out of context are captured."""
        i = Interviewer()
        i.state = InterviewState.CALIBRATION_GENRE
        
        # User answers genre question but mentions an artist
        # "pop, rock, definitely include Taylor Swift"
        resp, done = i.process_input("pop, rock, definitely include Taylor Swift")
        
        self.assertIn("pop", i.profile.target_genres)
        self.assertIn("Taylor Swift", i.profile.must_include_artists)
        # Should still advance state
        self.assertEqual(i.state, InterviewState.CALIBRATION_DESCRIPTOR)

if __name__ == '__main__':
    unittest.main()
