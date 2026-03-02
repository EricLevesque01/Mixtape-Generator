"""
Persona-based interview module.

Uses evidence-based music psychology frameworks to learn about the user,
then maps their answers to a PersonaProfile → UserProfile for generation.

Framework sources:
  Q1 — Russell's Circumplex (1980): arousal axis → energy signal
  Q2 — PANAS (Watson et al., 1988): positive/negative affect → valence + energy
  Q3 — MUSIC Model (Rentfrow et al., 2011): 5 music preference dimensions
       (Mellow, Unpretentious, Sophisticated, Intense, Contemporary)
  Q4 — ISMUS (Schäfer et al., 2013): listening motivation → intensity + uniformity
  Q5 — Big Five × Music (Rentfrow & Gosling, 2003): openness → familiarity/discovery
  Q6 — Wildcard: freeform cultural cue → extra descriptors
"""
import logging
import re
from typing import List, Dict, Tuple
from enum import Enum, auto
from .models import PersonaProfile, UserProfile
from .config import config

logger = logging.getLogger("mixtape_curator")


class PersonaState(Enum):
    AROUSAL       = auto()  # Q1: Russell's Circumplex — current arousal level
    VALENCE       = auto()  # Q2: PANAS — positive/negative affect
    MUSIC_PREF    = auto()  # Q3: MUSIC Model — dimension preference
    MOTIVATION    = auto()  # Q4: ISMUS — why are you listening?
    OPENNESS      = auto()  # Q5: Big Five Openness — familiarity vs discovery
    WILDCARD      = auto()  # Q6: freeform cultural cue
    CONFIRMATION  = auto()
    COMPLETED     = auto()


_VAGUE_KEYWORDS = {"idk", "unsure", "whatever", "doesn't matter", "not sure", "dunno", "hmm", "ok"}

# ── Q1: Arousal options → occasion field ────────────────────────────────────
_Q1 = (
    "First, where are you right now energy-wise?\n"
    "  A) Energized and alert\n"
    "  B) Calm and at ease\n"
    "  C) Drained or low\n"
    "  D) Restless or tense"
)
_Q1_MAP = {
    "a": "workout",        # high energy + high accessibility
    "b": "chill",          # low energy, peaceful
    "c": "late night",     # very low energy, introspective
    "d": "commute",        # mid-high energy, tense
}

# ── Q2: Valence options → mood_today field ───────────────────────────────────
_Q2 = (
    "How's your emotional state right now?\n"
    "  A) Positive — feeling good, upbeat\n"
    "  B) Negative — heavy, frustrated, or sad\n"
    "  C) Neutral — nothing in particular\n"
    "  D) Mixed — hard to say"
)
_Q2_MAP = {
    "a": "happy",
    "b": "frustrated",
    "c": "calm",
    "d": "reflective",
}

# ── Q3: MUSIC Model → personality_words field ────────────────────────────────
_Q3 = (
    "What sounds good to you right now?\n"
    "  A) Something soft, emotional, easy to sink into  [Mellow]\n"
    "  B) Something with depth and complexity  [Sophisticated]\n"
    "  C) Something intense, loud, high-energy  [Intense]\n"
    "  D) Something upbeat and feel-good  [Contemporary]"
)
_Q3_MAP = {
    "a": ["introspective", "calm", "chill"],          # Mellow dimension
    "b": ["curious", "creative", "introspective"],    # Sophisticated dimension
    "c": ["intense", "energetic", "adventurous"],     # Intense dimension
    "d": ["happy", "energetic", "adventurous"],       # Contemporary/Unpretentious
}

# ── Q4: ISMUS listening motivation → aesthetic_choice field ─────────────────
_Q4 = (
    "Why are you listening right now?\n"
    "  A) To match the mood I'm already in\n"
    "  B) To shift my mood — get somewhere different\n"
    "  C) As background while I do something else\n"
    "  D) To sit with it and really pay attention"
)
_Q4_MAP = {
    "a": "rooftop at sunset",          # mirror → moderate uniformity
    "b": "sweaty basement show",       # regulation → push energy/valence
    "c": "commute",                    # background → low uniformity
    "d": "dark room with headphones",  # attentive → high uniformity, introspective
}

# ── Q5: Openness → era_preference field ─────────────────────────────────────
_Q5 = (
    "Are you more in the mood for something familiar — songs that already "
    "feel like home — or open to discovering something you've never heard?"
)


class PersonaInterviewer:
    """
    Conversational interview agent grounded in music psychology research.
    Builds a PersonaProfile from 6 short questions, then maps it to a
    UserProfile for the mixtape generator.
    """

    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.persona = PersonaProfile()
        self.completed = False
        self.state = PersonaState.AROUSAL
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
        self.state = PersonaState.AROUSAL
        opening = (
            "Hey — I'll ask you 6 quick questions to get a read on you, "
            "then build a mixtape from Eric's library that fits.\n\n"
            + _Q1
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
                return self._advance_with_default()
            return self._rephrase_current(), False

        self._off_topic_count = 0

        handlers = {
            PersonaState.AROUSAL:    self._handle_arousal,
            PersonaState.VALENCE:    self._handle_valence,
            PersonaState.MUSIC_PREF: self._handle_music_pref,
            PersonaState.MOTIVATION: self._handle_motivation,
            PersonaState.OPENNESS:   self._handle_openness,
            PersonaState.WILDCARD:   self._handle_wildcard,
            PersonaState.CONFIRMATION: self._handle_confirmation,
        }
        handler = handlers.get(self.state)
        if handler:
            return handler(user_input)
        return "Something went sideways. Type 'start' to restart.", True

    # ------------------------------------------------------------------
    # Q1 — Arousal (Russell's Circumplex)
    # ------------------------------------------------------------------

    def _handle_arousal(self, text: str) -> Tuple[str, bool]:
        tl = text.lower().strip()
        letter = tl[0] if tl and tl[0] in _Q1_MAP else None

        if letter:
            self.persona.occasion = _Q1_MAP[letter]
        else:
            # Free-text fallback
            if any(w in tl for w in ["energy", "alert", "active", "hyped", "charged"]):
                self.persona.occasion = "workout"
            elif any(w in tl for w in ["calm", "ease", "peaceful", "relaxed"]):
                self.persona.occasion = "chill"
            elif any(w in tl for w in ["drain", "low", "tired", "exhausted"]):
                self.persona.occasion = "late night"
            else:
                self.persona.occasion = "commute"

        self.confidence["intent"] = 0.5
        self.state = PersonaState.VALENCE
        return _Q2, False

    # ------------------------------------------------------------------
    # Q2 — Valence (PANAS)
    # ------------------------------------------------------------------

    def _handle_valence(self, text: str) -> Tuple[str, bool]:
        tl = text.lower().strip()
        letter = tl[0] if tl and tl[0] in _Q2_MAP else None

        if letter:
            self.persona.mood_today = _Q2_MAP[letter]
        else:
            # Free-text fallback — store raw (mood-energy map in models.py handles it)
            if any(w in tl for w in ["good", "great", "happy", "positive", "upbeat"]):
                self.persona.mood_today = "happy"
            elif any(w in tl for w in ["bad", "heavy", "sad", "frustrated", "negative"]):
                self.persona.mood_today = "frustrated"
            elif any(w in tl for w in ["mixed", "complicated", "hard to say"]):
                self.persona.mood_today = "reflective"
            else:
                self.persona.mood_today = text

        self.confidence["energy_mood"] = 1.0
        self.state = PersonaState.MUSIC_PREF
        return _Q3, False

    # ------------------------------------------------------------------
    # Q3 — Music Preference (MUSIC Model)
    # ------------------------------------------------------------------

    def _handle_music_pref(self, text: str) -> Tuple[str, bool]:
        tl = text.lower().strip()
        letter = tl[0] if tl and tl[0] in _Q3_MAP else None

        if letter:
            self.persona.personality_words = _Q3_MAP[letter]
        else:
            # Free-text fallback
            if any(w in tl for w in ["soft", "mellow", "emotional", "easy"]):
                self.persona.personality_words = _Q3_MAP["a"]
            elif any(w in tl for w in ["complex", "depth", "jazz", "sophisticated"]):
                self.persona.personality_words = _Q3_MAP["b"]
            elif any(w in tl for w in ["intense", "loud", "heavy", "energy"]):
                self.persona.personality_words = _Q3_MAP["c"]
            else:
                self.persona.personality_words = _Q3_MAP["d"]

        self.confidence["intent"] = 1.0
        self.state = PersonaState.MOTIVATION
        return _Q4, False

    # ------------------------------------------------------------------
    # Q4 — Listening Motivation (ISMUS)
    # ------------------------------------------------------------------

    def _handle_motivation(self, text: str) -> Tuple[str, bool]:
        tl = text.lower().strip()
        letter = tl[0] if tl and tl[0] in _Q4_MAP else None

        if letter:
            self.persona.aesthetic_choice = _Q4_MAP[letter]
        else:
            if any(w in tl for w in ["match", "same", "already"]):
                self.persona.aesthetic_choice = _Q4_MAP["a"]
            elif any(w in tl for w in ["shift", "change", "different", "out of"]):
                self.persona.aesthetic_choice = _Q4_MAP["b"]
            elif any(w in tl for w in ["background", "while", "doing"]):
                self.persona.aesthetic_choice = _Q4_MAP["c"]
            else:
                self.persona.aesthetic_choice = _Q4_MAP["d"]

        self.confidence["cohesion"] = 1.0
        self.state = PersonaState.OPENNESS
        return _Q5, False

    # ------------------------------------------------------------------
    # Q5 — Openness / Familiarity (Big Five × Music)
    # ------------------------------------------------------------------

    def _handle_openness(self, text: str) -> Tuple[str, bool]:
        tl = text.lower()
        if any(w in tl for w in ["familiar", "know", "home", "love", "already", "nostalgia"]):
            self.persona.era_preference = "nostalgia"
        elif any(w in tl for w in ["new", "discover", "never", "fresh", "open", "different"]):
            self.persona.era_preference = "new"
        else:
            self.persona.era_preference = "both"

        self.state = PersonaState.WILDCARD
        return "Anything you've been into lately — a show, album, place, feeling — that might color this?", False

    # ------------------------------------------------------------------
    # Q6 — Wildcard
    # ------------------------------------------------------------------

    def _handle_wildcard(self, text: str) -> Tuple[str, bool]:
        self.persona.wildcard = text if text.lower() not in {"nothing", "nope", "no", "n/a", "-"} else ""
        self.state = PersonaState.CONFIRMATION
        return f"{self._generate_summary()}\n\nSound right? [yes / no]", False

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
            self.state = PersonaState.AROUSAL
            return f"No problem — let's try again.\n\n{_Q1}", False
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
        mood = self.persona.mood_today or "neutral"
        motivation = self.persona.aesthetic_choice or "—"

        return (
            f"  Mood: {mood}\n"
            f"  Listening mode: {motivation}\n"
            f"  Era: {era_label}\n"
            f"  Energy: {e_label}, {v_label}"
        )

    def _is_vague(self, text: str) -> bool:
        stripped = text.strip().lower()
        # Single letter A-D is always a valid answer
        if stripped in ("a", "b", "c", "d"):
            return False
        return any(k in stripped for k in _VAGUE_KEYWORDS) or len(stripped) < 2

    def _rephrase_current(self) -> str:
        rephrases = {
            PersonaState.AROUSAL:    f"Just pick whichever is closest:\n{_Q1}",
            PersonaState.VALENCE:    f"Even roughly:\n{_Q2}",
            PersonaState.MUSIC_PREF: f"Go with your gut:\n{_Q3}",
            PersonaState.MOTIVATION: f"Pick the closest:\n{_Q4}",
            PersonaState.OPENNESS:   "Familiar songs you know, or open to something new?",
            PersonaState.WILDCARD:   "Anything — a show, a trip, something you heard recently. (Or just skip it.)",
        }
        return rephrases.get(self.state, "Can you say a bit more?")

    def _advance_with_default(self) -> Tuple[str, bool]:
        defaults = {
            PersonaState.AROUSAL:    ("commute",                   PersonaState.VALENCE),
            PersonaState.VALENCE:    ("calm",                      PersonaState.MUSIC_PREF),
            PersonaState.MUSIC_PREF: (["open", "balanced"],        PersonaState.MOTIVATION),
            PersonaState.MOTIVATION: ("rooftop at sunset",         PersonaState.OPENNESS),
            PersonaState.OPENNESS:   ("both",                      PersonaState.WILDCARD),
            PersonaState.WILDCARD:   ("",                          PersonaState.CONFIRMATION),
        }
        if self.state not in defaults:
            return f"{self._generate_summary()}\n\nSound right? [yes / no]", False

        default_val, next_state = defaults[self.state]

        if self.state == PersonaState.AROUSAL:
            self.persona.occasion = default_val
        elif self.state == PersonaState.VALENCE:
            self.persona.mood_today = default_val
        elif self.state == PersonaState.MUSIC_PREF:
            self.persona.personality_words = default_val
        elif self.state == PersonaState.MOTIVATION:
            self.persona.aesthetic_choice = default_val
        elif self.state == PersonaState.OPENNESS:
            self.persona.era_preference = default_val
        elif self.state == PersonaState.WILDCARD:
            self.persona.wildcard = default_val

        self.state = next_state
        self._off_topic_count = 0

        next_questions = {
            PersonaState.VALENCE:    _Q2,
            PersonaState.MUSIC_PREF: _Q3,
            PersonaState.MOTIVATION: _Q4,
            PersonaState.OPENNESS:   _Q5,
            PersonaState.WILDCARD:   "Anything you've been into lately?",
            PersonaState.CONFIRMATION: self._generate_summary() + "\n\nSound right? [yes / no]",
        }
        return next_questions.get(next_state, "Sound right? [yes / no]"), False

    def _log_reply(self, text: str):
        self.history.append({"role": "assistant", "content": text})


# ── Legacy shims ────────────────────────────────────────────────────────────

class Interviewer(PersonaInterviewer):
    """Backward-compatible alias."""
    pass

interviewer = PersonaInterviewer()
