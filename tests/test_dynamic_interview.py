"""
Tests for the persona-based PersonaInterviewer (V5 two-tier LLM architecture).

Covers:
  - Standard linear flow through all 7 open-ended questions
  - Vague answer handling (no state advance)
  - Confirmation "no" restarts at Q1_TONE
  - Keyword fallback (_keyword_interpret) for all 7 Q names
  - MockLLM full interview path (end-to-end with fallback interpretation)
  - refine_profile() feedback hook
  - PersonaProfile.to_user_profile() mappings (unchanged — still pass)
"""
import unittest
from mixtape_curator.interview import (
    PersonaInterviewer,
    PersonaState,
    _keyword_interpret,
    _keyword_ack,
)
from mixtape_curator.models import PersonaProfile
from mixtape_curator.llm.providers.local import MockLLM


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_full_interview(answers: list[str], use_mock_llm: bool = False) -> PersonaInterviewer:
    """Run a complete interview with canned answers. Uses MockLLM when specified."""
    llm = MockLLM() if use_mock_llm else None
    agent = PersonaInterviewer(cheap_llm=llm, main_llm=llm)
    agent.start()
    for answer in answers:
        _, done = agent.process_input(answer)
        if done:
            break
    return agent


# ---------------------------------------------------------------------------
# Interview Flow
# ---------------------------------------------------------------------------

class TestPersonaInterviewFlow(unittest.TestCase):

    def test_standard_linear_flow(self):
        """Verify the standard linear path through all 7 questions."""
        agent = PersonaInterviewer()
        agent.start()

        # Q1: tone of day
        resp, done = agent.process_input("heavy, chaotic")
        self.assertFalse(done)
        self.assertEqual(agent.state, PersonaState.Q2_ENERGY)
        self.assertNotEqual(agent.raw_answers.get("Q1_TONE"), None)

        # Q2: energy
        resp, done = agent.process_input("drained, running on fumes")
        self.assertFalse(done)
        self.assertEqual(agent.state, PersonaState.Q3_ATTENTION)

        # Q3: attention
        resp, done = agent.process_input("unwinding, nothing in particular")
        self.assertFalse(done)
        self.assertEqual(agent.state, PersonaState.Q4_SPACE)

        # Q4: space
        resp, done = agent.process_input("just mine")
        self.assertFalse(done)
        self.assertEqual(agent.state, PersonaState.Q5_FLOW)

        # Q5: flow
        resp, done = agent.process_input("something focused and cohesive")
        self.assertFalse(done)
        self.assertEqual(agent.state, PersonaState.Q6_DIRECTION)

        # Q6: direction
        resp, done = agent.process_input("stay in this feeling")
        self.assertFalse(done)
        self.assertEqual(agent.state, PersonaState.Q7_TEXTURE)

        # Q7: texture
        resp, done = agent.process_input("overcast, cool, still")
        self.assertFalse(done)
        self.assertEqual(agent.state, PersonaState.CONFIRMATION)
        self.assertIn("Does that feel right?", resp)

        # Confirm
        resp, done = agent.process_input("yes")
        self.assertTrue(done)
        self.assertTrue(agent.completed)

    def test_all_7_answers_stored(self):
        """All 7 raw answers are stored before confirmation."""
        ANSWERS = [
            "steady, smooth",
            "calm, settled",
            "background tasks",
            "shared with others",
            "eclectic and wide-ranging",
            "head somewhere different",
            "bright, open, electric",
        ]
        agent = _run_full_interview(ANSWERS)
        self.assertEqual(agent.state, PersonaState.CONFIRMATION)
        for key in ["Q1_TONE","Q2_ENERGY","Q3_ATTENTION","Q4_SPACE","Q5_FLOW","Q6_DIRECTION","Q7_TEXTURE"]:
            self.assertIn(key, agent.raw_answers, msg=f"{key} not in raw_answers")

    def test_vague_answer_rephrases_without_advancing(self):
        """Vague answers don't advance the state."""
        agent = PersonaInterviewer()
        agent.start()
        initial_state = agent.state  # Q1_TONE
        resp, done = agent.process_input("idk")
        self.assertFalse(done)
        self.assertEqual(agent.state, initial_state)
        self.assertNotIn("Q1_TONE", agent.raw_answers)

    def test_confirmation_no_restarts_at_q1(self):
        """Answering 'no' at confirmation loops back to Q1_TONE."""
        agent = _run_full_interview([
            "light", "calm", "focused", "just mine",
            "cohesive", "stay here", "soft and warm",
        ])
        self.assertEqual(agent.state, PersonaState.CONFIRMATION)
        _, done = agent.process_input("no")
        self.assertFalse(done)
        self.assertEqual(agent.state, PersonaState.Q1_TONE)
        # raw_answers should be cleared
        self.assertEqual(agent.raw_answers, {})

    def test_confirmation_yes_marks_completed(self):
        """Answering 'yes' at confirmation sets completed=True."""
        agent = _run_full_interview([
            "tense", "buzzing", "deep focus", "mine only",
            "focused", "lean into it", "sharp and electric",
        ])
        self.assertEqual(agent.state, PersonaState.CONFIRMATION)
        _, done = agent.process_input("yes")
        self.assertTrue(done)
        self.assertTrue(agent.completed)


# ---------------------------------------------------------------------------
# Keyword fallback (_keyword_interpret)
# ---------------------------------------------------------------------------

class TestKeywordInterpret(unittest.TestCase):
    """Verify _keyword_interpret maps all 7 Q keys to a valid PersonaProfile."""

    def _interp(self, **kwargs) -> PersonaProfile:
        return _keyword_interpret({k: v for k, v in kwargs.items()})

    def test_heavy_tone_maps_frustrated(self):
        p = self._interp(Q1_TONE="heavy, a bit chaotic")
        self.assertEqual(p.mood_today, "frustrated")

    def test_light_tone_maps_happy(self):
        p = self._interp(Q1_TONE="light and smooth")
        self.assertEqual(p.mood_today, "happy")

    def test_drained_energy_maps_late_night(self):
        p = self._interp(Q1_TONE="steady", Q2_ENERGY="drained and slow")
        self.assertEqual(p.occasion, "late night")
        self.assertEqual(p.mood_today, "melancholic")

    def test_wired_energy_maps_restless(self):
        p = self._interp(Q1_TONE="steady", Q2_ENERGY="wired, buzzing")
        self.assertEqual(p.mood_today, "restless")

    def test_focused_attention_maps_headphones(self):
        p = self._interp(Q3_ATTENTION="deep focus, problem solving")
        self.assertIn("headphones", p.aesthetic_choice)

    def test_background_attention_maps_rooftop(self):
        p = self._interp(Q3_ATTENTION="background tasks")
        self.assertIn("rooftop", p.aesthetic_choice)

    def test_alone_space_maps_introspective(self):
        p = self._interp(Q4_SPACE="just mine, alone")
        self.assertIn("introspective", p.personality_words)

    def test_shared_space_maps_social(self):
        p = self._interp(Q4_SPACE="shared with others")
        self.assertIn("social", p.personality_words)

    def test_cohesive_flow_adds_focused(self):
        p = self._interp(Q5_FLOW="something focused and cohesive")
        self.assertIn("focused", p.personality_words)

    def test_eclectic_flow_adds_adventurous(self):
        p = self._interp(Q5_FLOW="eclectic and wide-ranging")
        self.assertIn("adventurous", p.personality_words)

    def test_stay_direction_maps_nostalgia(self):
        p = self._interp(Q6_DIRECTION="stay in this feeling")
        self.assertEqual(p.era_preference, "nostalgia")

    def test_shift_direction_maps_new(self):
        p = self._interp(Q6_DIRECTION="head somewhere different")
        self.assertEqual(p.era_preference, "new")

    def test_texture_stored_as_wildcard(self):
        p = self._interp(Q7_TEXTURE="overcast, cool, still")
        self.assertEqual(p.wildcard, "overcast, cool, still")

    def test_empty_raw_returns_valid_profile(self):
        """Empty raw_answers should still produce a usable PersonaProfile."""
        p = _keyword_interpret({})
        up = p.to_user_profile()
        self.assertIsNotNone(up.targets)
        self.assertGreaterEqual(up.targets.energy, 0.0)
        self.assertLessEqual(up.targets.energy, 1.0)


# ---------------------------------------------------------------------------
# MockLLM end-to-end
# ---------------------------------------------------------------------------

class TestMockLLMInterview(unittest.TestCase):

    def test_full_flow_with_mock_llm(self):
        """Complete interview with MockLLM: falls back to keyword interpret, exits cleanly."""
        agent = _run_full_interview([
            "steady", "calm", "unwinding", "just mine",
            "cohesive", "stay here", "soft and warm",
            "yes",
        ], use_mock_llm=True)
        # Should complete or be at confirmation
        self.assertTrue(agent.completed or agent.state == PersonaState.COMPLETED)

    def test_mock_llm_profile_is_usable(self):
        """Profile from MockLLM fallback maps to valid UserProfile targets."""
        agent = _run_full_interview([
            "tense", "buzzing", "deep focus", "mine only",
            "focused", "lean into it", "sharp, electric",
            "yes",
        ], use_mock_llm=True)
        up = agent.persona.to_user_profile()
        self.assertIsNotNone(up.targets)
        self.assertGreater(up.targets.energy, 0.0)


# ---------------------------------------------------------------------------
# Keyword acknowledgments
# ---------------------------------------------------------------------------

class TestKeywordAck(unittest.TestCase):

    def test_heavy_ack(self):
        r = _keyword_ack("heavy and slow")
        self.assertIn(r, ["Noted.", "Got it.", "Good to hear.", "Nice.", "Understood."])

    def test_electric_ack(self):
        r = _keyword_ack("bright and electric")
        self.assertEqual(r, "Good to hear.")

    def test_settled_ack(self):
        r = _keyword_ack("settled and calm")
        self.assertEqual(r, "Nice.")

    def test_default_ack(self):
        r = _keyword_ack("moderately middling")
        self.assertEqual(r, "Got it.")


# ---------------------------------------------------------------------------
# PersonaProfile → UserProfile mappings (unchanged, still valid)
# ---------------------------------------------------------------------------

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
        has_expected = any(d in up.target_descriptors for d in ["melancholic", "brooding", "moody", "pensive"])
        self.assertTrue(has_expected, msg=f"Got: {up.target_descriptors}")

    def test_context_notes_built_from_occasion_and_mood(self):
        p = PersonaProfile(occasion="road trip", mood_today="excited")
        up = p.to_user_profile()
        self.assertIn("road trip", up.context_notes)
        self.assertIn("excited", up.context_notes)


# ---------------------------------------------------------------------------
# refine_profile feedback
# ---------------------------------------------------------------------------

class TestRefineFeedback(unittest.TestCase):

    def test_refine_slow_increases_energy_word(self):
        """refine_profile 'too slow' appends 'energetic' to personality_words."""
        agent = PersonaInterviewer()
        agent.persona = PersonaProfile(personality_words=["calm"])
        agent.refine_profile("too slow, needs more energy")
        self.assertIn("energetic", agent.persona.personality_words)

    def test_refine_too_eclectic_adds_focused(self):
        """refine_profile 'too eclectic' appends 'focused' to personality_words."""
        agent = PersonaInterviewer()
        agent.persona = PersonaProfile(personality_words=["adventurous"])
        agent.refine_profile("all over the place, too eclectic")
        self.assertIn("focused", agent.persona.personality_words)

    def test_refine_too_intense_adds_chill(self):
        """refine_profile 'too loud' appends 'chill'."""
        agent = PersonaInterviewer()
        agent.persona = PersonaProfile(personality_words=["intense"])
        agent.refine_profile("too loud, I want something softer")
        self.assertIn("chill", agent.persona.personality_words)


if __name__ == "__main__":
    unittest.main()
