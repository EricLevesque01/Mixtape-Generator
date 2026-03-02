"""
Persona-based interview module — two-tier LLM architecture.

Design:
  - 5 open-ended questions inspired by evidence-based frameworks
    (Russell's Circumplex, PANAS, MUSIC Model, ISMUS, Big Five × Music)
  - Each answer gets a brief conversational acknowledgment from cheap model
    (gpt-4o-mini, ~5 tokens input / ~20 tokens output per turn)
  - After all answers: one call to the main model (gpt-4o) interprets
    the full conversation as structured PersonaProfile JSON
  - Falls back to keyword matching if no LLM is available (MockLLM)
"""
import json
import logging
import re
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum, auto
from .models import PersonaProfile, UserProfile
from .config import config

logger = logging.getLogger("mixtape_curator")


# ── Models ──────────────────────────────────────────────────────────────────
CHEAP_MODEL = "gpt-4o-mini"   # per-turn acknowledgments
MAIN_MODEL  = "gpt-4o"        # final interpretation


class PersonaState(Enum):
    Q1_MOOD      = auto()   # energy + emotional state (Russell + PANAS)
    Q2_SOUND     = auto()   # sonic preference (MUSIC Model)
    Q3_LISTENING = auto()   # listening motivation (ISMUS)
    Q4_FAMILIAR  = auto()   # familiarity preference (Big Five × Music)
    Q5_WILDCARD  = auto()   # freeform cultural cue
    CONFIRMATION = auto()
    COMPLETED    = auto()


_VAGUE_KEYWORDS = {"idk", "unsure", "whatever", "doesn't matter", "not sure", "dunno", "n/a", "nothing"}

# ── Question prompts ─────────────────────────────────────────────────────────
_QUESTIONS = {
    PersonaState.Q1_MOOD:      "How are you feeling right now — energy and mood?",
    PersonaState.Q2_SOUND:     "What kind of music sounds right for that?",
    PersonaState.Q3_LISTENING: "Are you planning to really sit with the music, or more as background while you do something else?",
    PersonaState.Q4_FAMILIAR:  "Are you in the mood for songs that already feel familiar, or open to discovering something you haven't heard?",
    PersonaState.Q5_WILDCARD:  "Anything you've been into lately — a show, album, place, feeling — that might color this? (Or just skip it.)",
}

_QUESTION_ORDER = [
    PersonaState.Q1_MOOD,
    PersonaState.Q2_SOUND,
    PersonaState.Q3_LISTENING,
    PersonaState.Q4_FAMILIAR,
    PersonaState.Q5_WILDCARD,
]

# ── System prompt for cheap model (per-turn ack) ─────────────────────────────
_ACK_SYSTEM = (
    "You are helping build a mixtape for someone. They are answering short questions. "
    "Your job: give a ONE sentence acknowledgment of their answer (max 12 words). "
    "No questions. Just a natural, warm, brief confirmation that you heard them."
)

# ── System prompt for main model (final interpretation) ─────────────────────
_INTERPRET_SYSTEM = (
    "You are a music curation assistant. Based on a short interview, extract the user's "
    "musical preferences and current state as a JSON object with exactly these fields:\n\n"
    "{\n"
    "  \"occasion\": <str: one of: workout, party, drive, study, late night, chill, dinner, pregame>,\n"
    "  \"mood_today\": <str: one of: happy, joyful, excited, calm, relaxed, focused, nostalgic, "
    "reflective, introspective, anxious, frustrated, angry, sad, melancholic, bored, restless, numb>,\n"
    "  \"personality_words\": <list of 2-3 str from: adventurous, curious, creative, nostalgic, "
    "passionate, introspective, calm, chill, intense, energetic, focused, adventurous, open, balanced>,\n"
    "  \"aesthetic_choice\": <str: one of: dark room with headphones, rooftop at sunset, "
    "sweaty basement show, commute>,\n"
    "  \"era_preference\": <str: one of: nostalgia, new, both>,\n"
    "  \"wildcard\": <str: brief description of their cultural cue, or empty string>\n"
    "}\n\n"
    "Return ONLY the JSON object. No explanation."
)


class PersonaInterviewer:
    """
    Two-tier LLM interview agent.
    - cheap_llm: used for per-turn conversational acknowledgments
    - main_llm: used once at the end to interpret all answers as PersonaProfile JSON
    Both default to None (keyword fallback mode).
    """

    def __init__(self, cheap_llm=None, main_llm=None):
        self.cheap_llm = cheap_llm
        self.main_llm  = main_llm
        self.history: List[Dict[str, str]] = []
        self.persona = PersonaProfile()
        self.completed = False
        self.state = PersonaState.Q1_MOOD
        self._off_topic_count = 0
        self._off_topic_max = config.get("off_topic_max", 3)
        self.raw_answers: Dict[str, str] = {}   # state_name → user answer

        self.confidence = {
            "constraints": 1.0,
            "intent": 0.0,
            "cohesion": 0.0,
            "energy_mood": 0.0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def profile(self) -> UserProfile:
        return self.persona.to_user_profile()

    def start(self) -> str:
        self.history = []
        self.state = PersonaState.Q1_MOOD
        self.raw_answers = {}
        opening = (
            "Hey — I'll ask you 5 quick questions to get a read on you, "
            "then build a mixtape from Eric's library that fits.\n\n"
            + _QUESTIONS[PersonaState.Q1_MOOD]
        )
        self._log_reply(opening)
        return opening

    def process_input(self, user_input: str) -> Tuple[str, bool]:
        self.history.append({"role": "user", "content": user_input})
        reply, done = self._dispatch(user_input.strip())
        self._log_reply(reply)
        return reply, done

    def process_input_compat(self, user_input: str, user_callback=None) -> Tuple[str, bool]:
        return self.process_input(user_input)

    def refine_profile(self, feedback: str):
        fb = feedback.lower()
        if any(w in fb for w in ["slow", "boring", "low energy", "faster", "more energy"]):
            self.persona.occasion = "workout " + self.persona.occasion
            self.persona.personality_words.append("energetic")
        if any(w in fb for w in ["too intense", "chill", "softer"]):
            self.persona.occasion = "chill " + self.persona.occasion
            self.persona.personality_words.append("chill")
        if any(w in fb for w in ["samey", "repetitive", "more variety"]):
            self.persona.personality_words.append("adventurous")
        if any(w in fb for w in ["all over", "too eclectic", "more cohesive"]):
            self.persona.personality_words.append("focused")

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, user_input: str) -> Tuple[str, bool]:
        if self.state in (PersonaState.CONFIRMATION, PersonaState.COMPLETED):
            return self._handle_confirmation(user_input)

        if self._is_vague(user_input) and self.state != PersonaState.Q5_WILDCARD:
            self._off_topic_count += 1
            if self._off_topic_count >= self._off_topic_max:
                return self._skip_current(), False
            return "Can you say a bit more? Even a few words is fine.", False

        self._off_topic_count = 0

        # Store answer
        self.raw_answers[self.state.name] = user_input

        # Generate acknowledgment
        ack = self._acknowledge(user_input)

        # Advance state
        next_q = self._advance_state()

        if next_q is None:
            # All questions answered — run interpretation
            return self._run_interpretation(ack)

        return f"{ack}\n\n{next_q}", False

    # ------------------------------------------------------------------
    # State advancement
    # ------------------------------------------------------------------

    def _advance_state(self) -> Optional[str]:
        """Move to next state. Return next question text, or None if done."""
        idx = _QUESTION_ORDER.index(self.state)
        if idx + 1 < len(_QUESTION_ORDER):
            self.state = _QUESTION_ORDER[idx + 1]
            return _QUESTIONS[self.state]
        else:
            self.state = PersonaState.CONFIRMATION
            return None

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _acknowledge(self, answer: str) -> str:
        """Cheap model: one-sentence acknowledgment of the user's answer."""
        if self.cheap_llm is None or isinstance(self.cheap_llm, _MockLLMCheck):
            return _keyword_ack(answer)

        try:
            msgs = [
                {"role": "system", "content": _ACK_SYSTEM},
                {"role": "user", "content": answer},
            ]
            return self.cheap_llm.generate(msgs, model=CHEAP_MODEL, temperature=0.7).strip()
        except Exception as e:
            logger.warning("Cheap model ack failed: %s", e)
            return _keyword_ack(answer)

    def _run_interpretation(self, ack: str) -> Tuple[str, bool]:
        """Main model: interpret all raw answers into PersonaProfile JSON."""
        conversation_text = "\n".join(
            f"Q: {_QUESTIONS[PersonaState[k]]}\nA: {v}"
            for k, v in self.raw_answers.items()
            if k in {s.name for s in _QUESTION_ORDER}
        )

        persona_data = None
        if self.main_llm is not None and not isinstance(self.main_llm, _MockLLMCheck):
            try:
                msgs = [
                    {"role": "system", "content": _INTERPRET_SYSTEM},
                    {"role": "user", "content": conversation_text},
                ]
                persona_data = self.main_llm.json(msgs, model=MAIN_MODEL, temperature=0.1)
            except Exception as e:
                logger.warning("Main model interpretation failed: %s", e)

        if persona_data:
            self.persona = _json_to_persona(persona_data, self.raw_answers.get("Q5_WILDCARD", ""))
        else:
            # Keyword fallback
            self.persona = _keyword_interpret(self.raw_answers)

        self.confidence["intent"] = 1.0
        self.confidence["energy_mood"] = 1.0
        self.confidence["cohesion"] = 1.0

        summary = self._generate_summary()
        return f"{ack}\n\n{summary}\n\nSound right? [yes / no]", False

    # ------------------------------------------------------------------
    # Confirmation
    # ------------------------------------------------------------------

    def _handle_confirmation(self, text: str) -> Tuple[str, bool]:
        tl = text.lower()
        if any(w in tl for w in ["yes", "yeah", "yep", "go", "do it", "sure", "sounds", "good"]):
            self.completed = True
            self.state = PersonaState.COMPLETED
            return "Let's go. Pulling tracks now...", True
        elif "no" in tl:
            self.state = PersonaState.Q1_MOOD
            self.raw_answers = {}
            return f"No problem — let's try again.\n\n{_QUESTIONS[PersonaState.Q1_MOOD]}", False
        else:
            return "Type 'yes' to build the tape, or 'no' to start over.", False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_summary(self) -> str:
        up = self.persona.to_user_profile()
        e_label = (
            "high energy" if up.targets.energy >= 0.70
            else "pretty relaxed" if up.targets.energy <= 0.35
            else "somewhere in the middle"
        )
        v_label = (
            "upbeat and bright" if up.targets.valence >= 0.60
            else "darker, more introspective" if up.targets.valence <= 0.30
            else "pretty balanced"
        )
        era_label = (
            "familiar songs you already know" if up.targets.familiarity >= 0.65
            else "new discoveries" if up.targets.familiarity <= 0.35
            else "mix of familiar and new"
        )
        return (
            f"  Mood: {self.persona.mood_today or 'neutral'}\n"
            f"  Listening mode: {self.persona.aesthetic_choice or '—'}\n"
            f"  Era: {era_label}\n"
            f"  Energy: {e_label}, {v_label}"
        )

    def _is_vague(self, text: str) -> bool:
        stripped = text.strip().lower()
        return any(k in stripped for k in _VAGUE_KEYWORDS) or len(stripped) < 3

    def _skip_current(self) -> str:
        """Skip current question with a default and advance."""
        defaults = {
            PersonaState.Q1_MOOD:      {"mood_today": "calm", "occasion": "chill"},
            PersonaState.Q2_SOUND:     {"personality_words": ["open", "balanced"]},
            PersonaState.Q3_LISTENING: {"aesthetic_choice": "rooftop at sunset"},
            PersonaState.Q4_FAMILIAR:  {"era_preference": "both"},
        }
        d = defaults.get(self.state, {})
        for k, v in d.items():
            setattr(self.persona, k, v)
        self._off_topic_count = 0
        next_q = self._advance_state()
        return next_q or "Alright, I think I have enough. Ready? [yes / no]"

    def _log_reply(self, text: str):
        self.history.append({"role": "assistant", "content": text})


# ── Helpers outside the class ─────────────────────────────────────────────────

class _MockLLMCheck:
    """Sentinel for isinstance checks — not an actual class used at runtime."""
    pass


def _keyword_ack(answer: str) -> str:
    """Simple keyword-based acknowledgment when no LLM is available."""
    a = answer.lower()
    if any(w in a for w in ["tired", "drained", "exhausted", "low"]):
        return "Noted — sounds like you need something that meets you where you're at."
    if any(w in a for w in ["frustrated", "angry", "stressed", "anxious"]):
        return "Got it — something with some edge to it."
    if any(w in a for w in ["happy", "great", "good", "excited", "upbeat"]):
        return "Good to hear — we'll keep that energy going."
    if any(w in a for w in ["calm", "chill", "relaxed", "peaceful"]):
        return "Nice. We'll keep things easy."
    if any(w in a for w in ["sad", "down", "melancholic", "low"]):
        return "Understood — something to sit with."
    return "Got it."


def _keyword_interpret(raw: Dict[str, str]) -> PersonaProfile:
    """Fallback: keyword-based mapping of raw answers → PersonaProfile."""
    p = PersonaProfile()

    q1 = raw.get("Q1_MOOD", "").lower()
    if any(w in q1 for w in ["tired", "drained", "low", "exhausted"]):
        p.occasion, p.mood_today = "late night", "melancholic"
    elif any(w in q1 for w in ["frustrated", "angry", "tense"]):
        p.occasion, p.mood_today = "commute", "frustrated"
    elif any(w in q1 for w in ["happy", "great", "excited", "upbeat"]):
        p.occasion, p.mood_today = "party", "happy"
    elif any(w in q1 for w in ["calm", "chill", "relaxed"]):
        p.occasion, p.mood_today = "chill", "calm"
    else:
        p.occasion, p.mood_today = "commute", "calm"

    q2 = raw.get("Q2_SOUND", "").lower()
    if any(w in q2 for w in ["soft", "mellow", "quiet", "slow"]):
        p.personality_words = ["introspective", "calm"]
    elif any(w in q2 for w in ["complex", "jazz", "classical", "depth"]):
        p.personality_words = ["curious", "creative"]
    elif any(w in q2 for w in ["intense", "loud", "heavy", "hard"]):
        p.personality_words = ["intense", "energetic"]
    else:
        p.personality_words = ["open", "balanced"]

    q3 = raw.get("Q3_LISTENING", "").lower()
    if any(w in q3 for w in ["really", "sit", "focus", "attention", "headphone"]):
        p.aesthetic_choice = "dark room with headphones"
    elif any(w in q3 for w in ["background", "while", "doing", "work"]):
        p.aesthetic_choice = "rooftop at sunset"
    else:
        p.aesthetic_choice = "rooftop at sunset"

    q4 = raw.get("Q4_FAMILIAR", "").lower()
    if any(w in q4 for w in ["familiar", "know", "love", "already", "nostalgia"]):
        p.era_preference = "nostalgia"
    elif any(w in q4 for w in ["new", "discover", "never", "fresh", "open"]):
        p.era_preference = "new"
    else:
        p.era_preference = "both"

    p.wildcard = raw.get("Q5_WILDCARD", "")
    return p


def _json_to_persona(data: Dict[str, Any], wildcard_raw: str) -> PersonaProfile:
    """Convert LLM JSON output to PersonaProfile, with safe fallbacks."""
    return PersonaProfile(
        occasion         = data.get("occasion", "chill"),
        mood_today       = data.get("mood_today", "calm"),
        personality_words= data.get("personality_words", ["open", "balanced"]),
        aesthetic_choice = data.get("aesthetic_choice", "rooftop at sunset"),
        era_preference   = data.get("era_preference", "both"),
        wildcard         = data.get("wildcard", wildcard_raw),
    )


# ── Legacy shims ──────────────────────────────────────────────────────────────

class Interviewer(PersonaInterviewer):
    """Backward-compatible alias."""
    pass

interviewer = PersonaInterviewer()
