"""
Tests for the persona-based PersonaInterviewer.

Covers: standard linear flow, aesthetic shortcut, vague-answer handling,
PersonaProfile.to_user_profile() mapping, and the refine_profile() feedback hook.
"""
import unittest
from mixtape_curator.interview import PersonaInterviewer, PersonaState
from mixtape_curator.models import PersonaProfile


def _run_full_interview(answers: list[str]) -> PersonaInterviewer:
    """Helper: run a complete interview with a list of canned answers."""
    agent = PersonaInterviewer()
    agent.start()
    for answer in answers:
        _, done = agent.process_input(answer)
        if done:
            break
    return agent


class TestPersonaInterviewFlow(unittest.TestCase):

    def test_standard_linear_flow(self):
        """Verify the standard linear path through all 7 states."""
        agent = PersonaInterviewer()
        agent.start()

        # 1. Occasion
        resp, done = agent.process_input("Late-night drive")
        self.assertFalse(done)
        self.assertEqual(agent.state, PersonaState.PERSONALITY)
        self.assertEqual(agent.persona.occasion, "Late-night drive")

        # 2. Personality — use scenario shortcut
        resp, done = agent.process_input("1")
        self.assertFalse(done)
        self.assertEqual(agent.state, PersonaState.MOOD_TODAY)
        # Scenario 1 maps to adventurous/curious/creative
        self.assertIn("adventurous", agent.persona.personality_words)

        # 3. Mood
        resp, done = agent.process_input("relaxed but a bit nostalgic")
        self.assertFalse(done)
        self.assertEqual(agent.state, PersonaState.AESTHETIC)

        # 4. Aesthetic — letter shortcut
        resp, done = agent.process_input("A")
        self.assertFalse(done)
        self.assertEqual(agent.state, PersonaState.ERA_TASTE)
        self.assertIn("dark room", agent.persona.aesthetic_choice)

        # 5. Era
        resp, done = agent.process_input("I love nostalgia, classic stuff")
        self.assertFalse(done)
        self.assertEqual(agent.state, PersonaState.WILDCARD)

        # 6. Wildcard
        resp, done = agent.process_input("Just finished reading a dark novel")
        self.assertFalse(done)
        self.assertEqual(agent.state, PersonaState.CONFIRMATION)

        # 7. Confirm
        resp, done = agent.process_input("yes")
        self.assertTrue(done)
        self.assertTrue(agent.completed)

    def test_aesthetic_letter_shortcuts(self):
        """A/B/C single-letter inputs are resolved to descriptive strings."""
        for letter, expected in [("a", "dark room"), ("b", "rooftop"), ("c", "basement")]:
            agent = PersonaInterviewer()
            agent.start()
            agent.process_input("driving")           # occasion
            agent.process_input("1")                 # personality
            agent.process_input("calm")              # mood → now in AESTHETIC
            self.assertEqual(agent.state, PersonaState.AESTHETIC)
            agent.process_input(letter)
            self.assertIn(expected, agent.persona.aesthetic_choice.lower(),
                          msg=f"Letter '{letter}' should map to '{expected}'")

    def test_vague_answer_rephrases(self):
        """Vague answers get a rephrase without advancing the state."""
        agent = PersonaInterviewer()
        agent.state = PersonaState.MOOD_TODAY

        resp, done = agent.process_input("idk")
        self.assertFalse(done)
        self.assertEqual(agent.state, PersonaState.MOOD_TODAY)
        self.assertEqual(agent.persona.mood_today, "")  # not yet set

    def test_confirmation_no_restarts(self):
        """Answering 'no' at confirmation loops back to occasion."""
        agent = _run_full_interview([
            "gym session",
            "3",             # personality: "both" → open, balanced, social
            "hyped",
            "C",
            "new stuff",
            "watching a sports doc",
        ])
        self.assertEqual(agent.state, PersonaState.CONFIRMATION)
        _, done = agent.process_input("no")
        self.assertFalse(done)
        self.assertEqual(agent.state, PersonaState.OCCASION)


class TestPersonaToUserProfile(unittest.TestCase):

    def test_gym_occasion_sets_high_energy(self):
        p = PersonaProfile(occasion="gym session", mood_today="hyped")
        up = p.to_user_profile()
        self.assertGreaterEqual(up.targets.energy, 0.70)

    def test_late_night_sets_low_accessibility(self):
        p = PersonaProfile(occasion="late night listening")
        up = p.to_user_profile()
        self.assertLessEqual(up.targets.accessibility, 0.45)

    def test_dark_room_aesthetic_sets_intensity(self):
        p = PersonaProfile(aesthetic_choice="dark room with headphones")
        up = p.to_user_profile()
        self.assertGreaterEqual(up.targets.intensity, 0.60)

    def test_nostalgia_era_sets_high_familiarity(self):
        p = PersonaProfile(era_preference="nostalgic, old school classics")
        up = p.to_user_profile()
        self.assertGreaterEqual(up.targets.familiarity, 0.65)

    def test_new_era_sets_low_familiarity(self):
        p = PersonaProfile(era_preference="discovering new things")
        up = p.to_user_profile()
        self.assertLessEqual(up.targets.familiarity, 0.40)

    def test_adventurous_personality_sets_low_uniformity(self):
        p = PersonaProfile(personality_words=["adventurous", "chaotic"])
        up = p.to_user_profile()
        self.assertLessEqual(up.targets.uniformity, 0.30)

    def test_calm_personality_sets_high_uniformity(self):
        p = PersonaProfile(personality_words=["chill", "calm"])
        up = p.to_user_profile()
        self.assertGreaterEqual(up.targets.uniformity, 0.60)

    def test_descriptors_populated_from_personality(self):
        p = PersonaProfile(personality_words=["introspective", "dark"])
        up = p.to_user_profile()
        self.assertTrue(len(up.target_descriptors) > 0)
        # introspective → melancholic / pensive; dark → brooding / moody
        has_expected = any(d in up.target_descriptors for d in ["melancholic", "brooding", "moody", "pensive"])
        self.assertTrue(
            has_expected,
            msg=f"Expected rich descriptors, got: {up.target_descriptors}"
        )

    def test_context_notes_built_from_occasion_and_mood(self):
        p = PersonaProfile(occasion="road trip", mood_today="excited")
        up = p.to_user_profile()
        self.assertIn("road trip", up.context_notes)
        self.assertIn("excited", up.context_notes)


class TestRefineFeedback(unittest.TestCase):

    def test_refine_slow_increases_energy(self):
        agent = _run_full_interview([
            "just chilling", "3", "calm", "B", "both", "read a book"
        ])
        original_energy = agent.profile.targets.energy
        agent.refine_profile("too slow, needs more energy")
        self.assertGreater(agent.profile.targets.energy, original_energy)

    def test_refine_too_eclectic_adds_focused(self):
        agent = _run_full_interview([
            "study session", "1", "bored", "C", "new", "wild documentary"
        ])
        agent.refine_profile("all over the place, too eclectic")
        self.assertIn("focused", agent.persona.personality_words)


if __name__ == "__main__":
    unittest.main()
