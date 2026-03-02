"""
Persona-based interview module.

Asks personality and emotional-state questions (no specific songs, no occasion)
to build a PersonaProfile, which is then mapped to a UserProfile for generation.

Question flow:
  1. ENERGY      — current energy/state (maps to occasion + mood signals)
  2. EMOTIONAL   — emotional tone right now (maps to mood_today + valence)
  3. LISTENING   — how they engage with music (maps to personality_words + uniformity)
  4. FAMILIARITY — nostalgia vs discovery (maps to era_preference + familiarity)
  5. TEXTURE     — sonic texture preference (maps to aesthetic_choice + descriptors)
  6. WILDCARD    — anything they've been into lately (cultural cue)
  7. CONFIRMATION
"""
import logging
from typing import List, Dict, Tuple
from enum import Enum, auto
from .models import PersonaProfile, UserProfile
from .config import config

logger = logging.getLogger("mixtape_curator")


class PersonaState(Enum):
    ENERGY        = auto()  # How's your energy right now?
    EMOTIONAL     = auto()  # Emotional tone / headspace
    LISTENING     = auto()  # How do you engage with music?
    FAMILIARITY   = auto()  # Nostalgia vs discovering new things
    TEXTURE       = auto()  # Sonic texture / intensity preference
    WILDCARD      = auto()  # Freeform cultural cue
    CONFIRMATION  = auto()
    COMPLETED     = auto()


_VAGUE_KEYWORDS = {"idk", "unsure", "whatever", "doesn't matter", "not sure", "dunno", "hmm", "ok"}


class PersonaInterviewer:
    """
    Conversational interview that builds a PersonaProfile from personality
    and emotional-state questions, then maps it to a UserProfile for generation.
    """

    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.persona = PersonaProfile()
        self.completed = False
        self.state = PersonaState.ENERGY
        self._off_topic_count = 0
        self._off_topic_max = config.get("off_topic_max", 3)

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
        self.state = PersonaState.ENERGY
        opening = (
            "Hey — I'll ask you a few quick questions to learn a bit about you, "
            "then I'll build a mixtape from Eric's library that fits.\n\n"
            "First: how's your energy right now? Are you winding down, "
            "mid-stride, or do you need something to pick you up?"
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
        if any(w in fb for w in ["too intense", "too loud", "chill", "softer"]):
            self.persona.occasion = "chill " + self.persona.occasion
            self.persona.personality_words.append("chill")
        if any(w in fb for w in ["samey", "repetitive", "boring", "more variety"]):
            self.persona.personality_words.append("adventurous")
        if any(w in fb for w in ["all over", "too eclectic", "more cohesive", "focus"]):
            self.persona.personality_words.append("focused")

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, user_input: str) -> Tuple[str, bool]:
        if self._is_vague(user_input) and self.state not in (
            PersonaState.CONFIRMATION, PersonaState.COMPLETED
        ):
            self._off_topic_count += 1
            if self._off_topic_count >= self._off_topic_max:
                logger.info("Off-topic limit hit — advancing with defaults")
                return self._advance_with_default()
            return self._rephrase_current(), False

        self._off_topic_count = 0

        handlers = {
            PersonaState.ENERGY:      self._handle_energy,
            PersonaState.EMOTIONAL:   self._handle_emotional,
            PersonaState.LISTENING:   self._handle_listening,
            PersonaState.FAMILIARITY: self._handle_familiarity,
            PersonaState.TEXTURE:     self._handle_texture,
            PersonaState.WILDCARD:    self._handle_wildcard,
            PersonaState.CONFIRMATION: self._handle_confirmation,
        }
        handler = handlers.get(self.state)
        if handler:
            return handler(user_input)
        return "Something went sideways. Type 'start' to restart.", True

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_energy(self, text: str) -> Tuple[str, bool]:
        """Maps energy level → occasion field (drives energy + accessibility targets)."""
        tl = text.lower()
        if any(w in tl for w in ["winding down", "tired", "low", "exhausted", "relaxed", "slow"]):
            self.persona.occasion = "winding down at home"
        elif any(w in tl for w in ["charged", "pick up", "need energy", "hyped", "pumped", "motivated"]):
            self.persona.occasion = "workout"
        elif any(w in tl for w in ["mid", "normal", "fine", "okay", "alright", "good"]):
            self.persona.occasion = "commute"
        else:
            # Free-text — store raw and let the mood map do the lifting
            self.persona.occasion = text
        self.confidence["intent"] += 0.25
        self.state = PersonaState.EMOTIONAL
        return (
            "What's the emotional tone right now — are you in your feelings, "
            "feeling good, or somewhere more neutral?"
        ), False

    def _handle_emotional(self, text: str) -> Tuple[str, bool]:
        """Stores mood directly — the mood-energy map in models.py handles the rest."""
        self.persona.mood_today = text
        self.confidence["energy_mood"] = 1.0
        self.state = PersonaState.LISTENING
        return (
            "When you listen to music, do you sit with it and really pay attention — "
            "or is it more in the background of whatever you're doing?"
        ), False

    def _handle_listening(self, text: str) -> Tuple[str, bool]:
        """Listening mode → personality_words (drives uniformity / cohesion targets)."""
        tl = text.lower()
        if any(w in tl for w in ["sit", "pay attention", "really listen", "focus", "headphones", "alone"]):
            self.persona.personality_words = ["introspective", "attentive", "curious"]
        elif any(w in tl for w in ["background", "coding", "working", "driving", "doing something"]):
            self.persona.personality_words = ["focused", "productive", "balanced"]
        elif any(w in tl for w in ["both", "depends", "either", "mix"]):
            self.persona.personality_words = ["open", "balanced", "social"]
        else:
            import re
            words = re.split(r"[,\s]+and\s+|,\s*|\s+", text)
            self.persona.personality_words = [w.strip() for w in words if len(w.strip()) > 1][:5]
        self.confidence["intent"] += 0.25
        self.state = PersonaState.FAMILIARITY
        return (
            "Are you in the mood for music that already feels familiar — "
            "songs you know — or do you want to discover something?"
        ), False

    def _handle_familiarity(self, text: str) -> Tuple[str, bool]:
        """Maps answer → era_preference field."""
        tl = text.lower()
        if any(w in tl for w in ["familiar", "know", "nostalgia", "classic", "love", "already"]):
            self.persona.era_preference = "nostalgia"
        elif any(w in tl for w in ["new", "discover", "never heard", "something different", "fresh"]):
            self.persona.era_preference = "new"
        else:
            self.persona.era_preference = "both"
        self.state = PersonaState.TEXTURE
        return (
            "When music really gets you — what does that feel like? "
            "Is it something soft and introspective, or loud and overwhelming, "
            "or something in between?"
        ), False

    def _handle_texture(self, text: str) -> Tuple[str, bool]:
        """Sonic texture preference → aesthetic_choice field."""
        tl = text.lower()
        if any(w in tl for w in ["soft", "quiet", "introspective", "gentle", "calm", "mellow"]):
            self.persona.aesthetic_choice = "dark room with headphones"
        elif any(w in tl for w in ["loud", "overwhelming", "intense", "heavy", "big", "powerful"]):
            self.persona.aesthetic_choice = "sweaty basement show"
        elif any(w in tl for w in ["between", "both", "medium", "somewhere", "depends"]):
            self.persona.aesthetic_choice = "rooftop at sunset"
        else:
            self.persona.aesthetic_choice = text
        self.confidence["cohesion"] = 1.0
        self.state = PersonaState.WILDCARD
        return "Anything you've been into lately — a show, album, place, feeling — that might color this?", False

    def _handle_wildcard(self, text: str) -> Tuple[str, bool]:
        self.persona.wildcard = text
        self.state = PersonaState.CONFIRMATION
        summary = self._generate_summary()
        return f"{summary}\n\nSound right? [yes / no]", False

    def _handle_confirmation(self, text: str) -> Tuple[str, bool]:
        tl = text.lower()
        if any(w in tl for w in ["yes", "yeah", "yep", "go", "do it", "sure", "sounds"]):
            self.completed = True
            self.state = PersonaState.COMPLETED
            return "Let's go. Pulling tracks now...", True
        elif "no" in tl:
            self.state = PersonaState.ENERGY
            return "No problem — let's start over. How's your energy right now?", False
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
            "familiar stuff you know" if up.targets.familiarity >= 0.65
            else "new discoveries" if up.targets.familiarity <= 0.35
            else "mix of old and new"
        )
        mood = self.persona.mood_today or "neutral"
        aesthetic = self.persona.aesthetic_choice or "—"

        return (
            f"  Mood: {mood}\n"
            f"  Vibe: {aesthetic}\n"
            f"  Era: {era_label}\n"
            f"  Energy: {e_label}, {v_label}"
        )

    def _is_vague(self, text: str) -> bool:
        return any(k in text.lower() for k in _VAGUE_KEYWORDS) or len(text.strip()) < 2

    def _rephrase_current(self) -> str:
        rephrases = {
            PersonaState.ENERGY:      "Even simple works — winding down, mid-stride, need a boost?",
            PersonaState.EMOTIONAL:   "Like — are you feeling good, down, restless, somewhere in between?",
            PersonaState.LISTENING:   "Do you tend to really focus on music, or is it more background noise?",
            PersonaState.FAMILIARITY: "Want songs you already know and love, or something you've never heard?",
            PersonaState.TEXTURE:     "Soft and quiet, loud and intense, or somewhere in between?",
            PersonaState.WILDCARD:    "Anything — a show you're watching, somewhere you went, a song recently stuck in your head.",
        }
        return rephrases.get(self.state, "Can you say a bit more?")

    def _advance_with_default(self) -> Tuple[str, bool]:
        defaults = {
            PersonaState.ENERGY:      ("commute",               PersonaState.EMOTIONAL),
            PersonaState.EMOTIONAL:   ("neutral",               PersonaState.LISTENING),
            PersonaState.LISTENING:   ("open, balanced",        PersonaState.FAMILIARITY),
            PersonaState.FAMILIARITY: ("both",                  PersonaState.TEXTURE),
            PersonaState.TEXTURE:     ("rooftop at sunset",     PersonaState.WILDCARD),
            PersonaState.WILDCARD:    ("",                      PersonaState.CONFIRMATION),
        }
        if self.state not in defaults:
            return "Alright, let's just go for it. Sound good? [yes / no]", False

        default_val, next_state = defaults[self.state]
        if self.state == PersonaState.ENERGY:
            self.persona.occasion = default_val
        elif self.state == PersonaState.EMOTIONAL:
            self.persona.mood_today = default_val
        elif self.state == PersonaState.LISTENING:
            self.persona.personality_words = ["open", "balanced"]
        elif self.state == PersonaState.FAMILIARITY:
            self.persona.era_preference = default_val
        elif self.state == PersonaState.TEXTURE:
            self.persona.aesthetic_choice = default_val
        elif self.state == PersonaState.WILDCARD:
            self.persona.wildcard = default_val

        self.state = next_state
        self._off_topic_count = 0

        next_questions = {
            PersonaState.EMOTIONAL:   "What's the emotional tone right now — in your feelings, feeling good, or more neutral?",
            PersonaState.LISTENING:   "When you listen to music, do you really sit with it, or is it more background?",
            PersonaState.FAMILIARITY: "Familiar songs you already know, or open to discovering something?",
            PersonaState.TEXTURE:     "Do you want something soft and introspective, loud and intense, or in between?",
            PersonaState.WILDCARD:    "Anything you've been into lately that might color this?",
            PersonaState.CONFIRMATION: self._generate_summary() + "\n\nSound right? [yes / no]",
        }
        return next_questions.get(next_state, "Sound right? [yes / no]"), False

    def _log_reply(self, text: str):
        self.history.append({"role": "assistant", "content": text})


# ---------------------------------------------------------------------------
# Legacy shims
# ---------------------------------------------------------------------------

class Interviewer(PersonaInterviewer):
    """Backward-compatible alias."""
    pass

interviewer = PersonaInterviewer()
