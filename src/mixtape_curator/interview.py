"""
Persona-based interview module.

Instead of asking about music preferences directly, the PersonaInterviewer
asks about the USER AS A PERSON — their occasion, personality, mood, aesthetic
taste, era preference, and a wildcard cultural cue. Those answers are invisibly
mapped to a UserProfile via PersonaProfile.to_user_profile().
"""
import logging
from typing import List, Dict, Tuple
from enum import Enum, auto
from .models import PersonaProfile, UserProfile
from .config import config

logger = logging.getLogger("mixtape_curator")


class PersonaState(Enum):
    OCCASION      = auto()  # What's the occasion / context?
    PERSONALITY   = auto()  # 3 words friends would use to describe you
    MOOD_TODAY    = auto()  # How are you feeling right now?
    AESTHETIC     = auto()  # Multiple-choice vibe pick
    ERA_TASTE     = auto()  # Nostalgia vs. discovering new things
    WILDCARD      = auto()  # Last thing that stuck with you (cultural cue)
    CONFIRMATION  = auto()  # Summary + "ready?"
    COMPLETED     = auto()


# Canned follow-up for vague answers
_VAGUE_KEYWORDS = {"idk", "unsure", "whatever", "doesn't matter", "not sure", "dunno", "hmm"}

# Aesthetic option hints shown to the user
_AESTHETIC_QUESTION = (
    "What kind of environment fits where you're at right now?\n"
    "  A) Headphones on, lights low — just you and the music\n"
    "  B) Outside somewhere, golden hour kind of feeling\n"
    "  C) Loud, packed room, everyone's feeling it\n"
    "(A, B, C, or just describe it)"
)

_AESTHETIC_CHOICE_MAP: Dict[str, str] = {
    "a": "dark room with headphones",
    "b": "rooftop at sunset",
    "c": "sweaty basement show",
}


class PersonaInterviewer:
    """
    Conversational interview agent that learns about the user as a person,
    then maps their profile onto a UserProfile for the mixtape generator.
    """

    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.persona = PersonaProfile()
        self.completed = False
        self.state = PersonaState.OCCASION
        self._off_topic_count = 0
        self._off_topic_max = config.get("off_topic_max", 3)

        # Confidence axes (for compatibility with existing UI checks)
        self.confidence = {
            "constraints": 1.0,   # no hard constraints in persona mode
            "intent": 0.0,
            "cohesion": 0.0,
            "energy_mood": 0.0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def profile(self) -> UserProfile:
        """Return a UserProfile derived from the current persona answers."""
        return self.persona.to_user_profile()

    def start(self) -> str:
        self.history = []
        self.state = PersonaState.OCCASION
        opening = (
            "Hey — I'll ask you a few quick questions to get a sense of who you are, "
            "then build you a mixtape from your library.\n\n"
            "What's the occasion? What are you doing while you listen?"
        )
        self._log_reply(opening)
        return opening

    def process_input(self, user_input: str) -> Tuple[str, bool]:
        """
        Process one user turn.
        Returns (next_question_or_summary, is_complete).
        """
        self.history.append({"role": "user", "content": user_input})
        reply, done = self._dispatch(user_input.strip())
        self._log_reply(reply)
        return reply, done

    # Alias so existing call-sites that pass user_callback= still work
    def process_input_compat(self, user_input: str, user_callback=None) -> Tuple[str, bool]:
        return self.process_input(user_input)

    def refine_profile(self, feedback: str):
        """
        Post-generation feedback hook (used by Neither loop in ui_cli).
        Re-interprets free-text feedback to nudge the persona.
        """
        fb = feedback.lower()
        if any(w in fb for w in ["slow", "boring", "low energy", "faster", "more energy"]):
            # Boost energy by shifting occasion toward high-energy keyword
            self.persona.occasion = "workout " + self.persona.occasion
            self.persona.personality_words.append("energetic")
            logger.info("Persona refinement: boosting energy signal")
        if any(w in fb for w in ["too intense", "too loud", "chill", "softer"]):
            # Reduce energy by shifting occasion toward chill keyword
            self.persona.occasion = "chill " + self.persona.occasion
            self.persona.personality_words.append("chill")
            logger.info("Persona refinement: reducing energy signal")
        if any(w in fb for w in ["samey", "repetitive", "boring", "more variety"]):
            self.persona.personality_words.append("adventurous")
            logger.info("Persona refinement: added 'adventurous' personality word")
        if any(w in fb for w in ["all over", "too eclectic", "more cohesive", "focus"]):
            self.persona.personality_words.append("focused")
            logger.info("Persona refinement: added 'focused' personality word")

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, user_input: str) -> Tuple[str, bool]:
        if self._is_vague(user_input) and self.state not in (
            PersonaState.CONFIRMATION, PersonaState.COMPLETED
        ):
            self._off_topic_count += 1
            if self._off_topic_count >= self._off_topic_max:
                # Move on with defaults, don't stall forever
                logger.info("Off-topic limit hit — advancing with defaults")
                return self._advance_with_default()
            return self._rephrase_current(), False

        self._off_topic_count = 0  # reset on meaningful answer

        if self.state == PersonaState.OCCASION:
            return self._handle_occasion(user_input)
        elif self.state == PersonaState.PERSONALITY:
            return self._handle_personality(user_input)
        elif self.state == PersonaState.MOOD_TODAY:
            return self._handle_mood(user_input)
        elif self.state == PersonaState.AESTHETIC:
            return self._handle_aesthetic(user_input)
        elif self.state == PersonaState.ERA_TASTE:
            return self._handle_era(user_input)
        elif self.state == PersonaState.WILDCARD:
            return self._handle_wildcard(user_input)
        elif self.state == PersonaState.CONFIRMATION:
            return self._handle_confirmation(user_input)

        return "Something went sideways. Type 'start' to restart.", True

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_occasion(self, text: str) -> Tuple[str, bool]:
        self.persona.occasion = text
        self.confidence["intent"] += 0.25
        self.state = PersonaState.PERSONALITY
        reply = (
            "Got it. Are you usually the person in the room playing stuff no one's heard, "
            "or do you know every word to everything? Or somewhere in between?"
        )
        return reply, False

    def _handle_personality(self, text: str) -> Tuple[str, bool]:
        import re
        text_lower = text.lower()
        # Map scenario numbers to Big Five signals
        if text_lower.strip() in ("1", "1)") or "introducing" in text_lower or "never heard" in text_lower:
            self.persona.personality_words = ["adventurous", "curious", "creative"]
        elif text_lower.strip() in ("2", "2)") or "every word" in text_lower or "know every" in text_lower:
            self.persona.personality_words = ["nostalgic", "passionate", "loyal"]
        elif text_lower.strip() in ("3", "3)") or "both" in text_lower or "depends" in text_lower or "between" in text_lower:
            self.persona.personality_words = ["open", "balanced", "social"]
        else:
            # Free-text: split on commas / 'and' / spaces — grab up to 5 words/phrases
            words = re.split(r"[,\s]+and\s+|,\s*", text)
            self.persona.personality_words = [w.strip() for w in words if len(w.strip()) > 1][:5]
        self.confidence["intent"] += 0.25
        self.state = PersonaState.MOOD_TODAY
        reply = "How are you feeling right now?"
        return reply, False

    def _handle_mood(self, text: str) -> Tuple[str, bool]:
        self.persona.mood_today = text
        self.confidence["energy_mood"] = 1.0
        self.state = PersonaState.AESTHETIC
        return _AESTHETIC_QUESTION, False

    def _handle_aesthetic(self, text: str) -> Tuple[str, bool]:
        text_lower = text.lower().strip()
        # Check single-letter shortcut first
        if text_lower in _AESTHETIC_CHOICE_MAP:
            self.persona.aesthetic_choice = _AESTHETIC_CHOICE_MAP[text_lower]
        else:
            # Accept free-text description
            self.persona.aesthetic_choice = text
        self.confidence["cohesion"] = 1.0
        self.state = PersonaState.ERA_TASTE
        reply = (
            "Are you feeling more like familiar stuff right now — things you know — "
            "or do you want to hear something new?"
        )
        return reply, False

    def _handle_era(self, text: str) -> Tuple[str, bool]:
        self.persona.era_preference = text
        self.state = PersonaState.WILDCARD
        reply = "What's something you've been into lately — a show, album, place, anything?"
        return reply, False

    def _handle_wildcard(self, text: str) -> Tuple[str, bool]:
        self.persona.wildcard = text
        self.state = PersonaState.CONFIRMATION
        summary = self._generate_summary()
        reply = f"{summary}\n\nSound right? [yes / no]"
        return reply, False

    def _handle_confirmation(self, text: str) -> Tuple[str, bool]:
        text_lower = text.lower()
        if "yes" in text_lower or "yeah" in text_lower or "go" in text_lower or "do it" in text_lower:
            self.completed = True
            self.state = PersonaState.COMPLETED
            return "Let's go. Pulling tracks now...", True
        elif "no" in text_lower:
            # Drop back to the beginning
            self.state = PersonaState.OCCASION
            return (
                "No problem — let's start fresh. "
                "What's the occasion for this tape?"
            ), False
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
            else "darker, heavier" if up.targets.valence <= 0.30
            else "pretty balanced"
        )
        era_label = (
            "familiar stuff, things you know" if up.targets.familiarity >= 0.65
            else "new discoveries" if up.targets.familiarity <= 0.35
            else "mix of old and new"
        )
        occasion = self.persona.occasion or "just listening"
        mood = self.persona.mood_today or "neutral"
        aesthetic = self.persona.aesthetic_choice or "—"

        return (
            f"  Occasion: {occasion}\n"
            f"  Mood: {mood}\n"
            f"  Setting: {aesthetic}\n"
            f"  Era: {era_label}\n"
            f"  Energy: {e_label}, {v_label}"
        )

    def _is_vague(self, text: str) -> bool:
        # Single-letter A/B/C are valid answers in AESTHETIC state
        if self.state == PersonaState.AESTHETIC and text.strip().lower() in _AESTHETIC_CHOICE_MAP:
            return False
        # Single-digit 1/2/3 are valid answers in PERSONALITY state
        if self.state == PersonaState.PERSONALITY and text.strip() in ("1", "2", "3"):
            return False
        return any(k in text.lower() for k in _VAGUE_KEYWORDS) or len(text.strip()) < 3

    def _rephrase_current(self) -> str:
        rephrases = {
            PersonaState.OCCASION:    "Even something like 'driving around' or 'just hanging at home' works.",
            PersonaState.PERSONALITY: "Think about it like this — do you usually put people on to new music, or are you the one who knows every word?",
            PersonaState.MOOD_TODAY:  "Even vague is fine. Tired? Restless? Good? Something in between?",
            PersonaState.AESTHETIC:   "Just go with A, B, or C if that's easier.",
            PersonaState.ERA_TASTE:   "Are you more in the mood for songs you already know, or something you've never heard?",
            PersonaState.WILDCARD:    "Anything — a show you're watching, somewhere you went, something you heard recently.",
        }
        return rephrases.get(self.state, "Can you say a bit more?")

    def _advance_with_default(self) -> Tuple[str, bool]:
        """Skip the current state with a sensible default and move on."""
        defaults = {
            PersonaState.OCCASION:    ("just listening",      PersonaState.PERSONALITY),
            PersonaState.PERSONALITY: ("open, balanced",      PersonaState.MOOD_TODAY),
            PersonaState.MOOD_TODAY:  ("neutral",             PersonaState.AESTHETIC),
            PersonaState.AESTHETIC:   ("rooftop at sunset",   PersonaState.ERA_TASTE),
            PersonaState.ERA_TASTE:   ("both",                PersonaState.WILDCARD),
            PersonaState.WILDCARD:    ("",                    PersonaState.CONFIRMATION),
        }
        if self.state not in defaults:
            return "Alright, let's just go for it. Ready? [yes / no]", False

        default_val, next_state = defaults[self.state]
        # Apply default to persona
        if self.state == PersonaState.PERSONALITY:
            self.persona.personality_words = ["open", "balanced"]
        elif self.state == PersonaState.OCCASION:
            self.persona.occasion = default_val
        elif self.state == PersonaState.MOOD_TODAY:
            self.persona.mood_today = default_val
        elif self.state == PersonaState.AESTHETIC:
            self.persona.aesthetic_choice = default_val
        elif self.state == PersonaState.ERA_TASTE:
            self.persona.era_preference = default_val
        elif self.state == PersonaState.WILDCARD:
            self.persona.wildcard = default_val

        self.state = next_state
        self._off_topic_count = 0  # reset so next state isn't immediately skipped

        # Return the next question directly without re-dispatching
        next_questions = {
            PersonaState.PERSONALITY: (
                "Which fits you more?\n"
                "  1) You introduce people to artists they've never heard\n"
                "  2) You know every word to every song in the room\n"
                "  3) Depends on the night\n"
                "(Or just describe yourself.)"
            ),
            PersonaState.MOOD_TODAY:  "What's your mood or headspace right now?",
            PersonaState.AESTHETIC:   _AESTHETIC_QUESTION,
            PersonaState.ERA_TASTE:   "Are you more drawn to nostalgia or discovering something new?",
            PersonaState.WILDCARD:    "Last thing you watched, read, or did that genuinely stuck with you?",
            PersonaState.CONFIRMATION: self._generate_summary() + "\n\nReady for me to build your tape? [yes / no]",
        }
        return next_questions.get(next_state, "Ready? [yes / no]"), False

    def _log_reply(self, text: str):
        self.history.append({"role": "assistant", "content": text})


# ---------------------------------------------------------------------------
# Legacy shim — keeps old import paths working (interview_agent, tests, etc.)
# ---------------------------------------------------------------------------

class Interviewer(PersonaInterviewer):
    """Backward-compatible alias for PersonaInterviewer."""
    pass


# Module-level singleton (used by some test imports)
interviewer = PersonaInterviewer()
