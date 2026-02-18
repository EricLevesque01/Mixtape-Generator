
from typing import List, Dict, Optional, Tuple, Callable
import json
import logging
from .models import UserProfile
from .config import config
from .llm.interface import LLMProvider
from .llm.providers.local import MockLLM
from .library import library

logger = logging.getLogger(__name__)

# System prompt for the Interview Agent
SYSTEM_PROMPT = """You are a Mixtape Curator AI conducting a conversational interview to build a playlist.
Your goal is to gather enough information to create a perfect mixtape.

You need to determine:
1. WHO is this for? (recipient / occasion / context)
2. WHAT kind of music? (genres, artists, era/decade)
3. WHAT VIBE? (energy level, mood/valence, tempo preference)

Be conversational, warm, and enthusiastic about music. Ask follow-up questions naturally.
If the user provides a lot of info at once, acknowledge it and ask about what's still missing.
If the user seems impatient or says "just go" or "that's enough", respect that and wrap up.

IMPORTANT: After each user message, you must also extract structured data.
Always respond with valid JSON in this format:
{
    "response": "Your conversational response to the user",
    "extracted": {
        "recipient": "who the mix is for (or null)",
        "context": "occasion/setting (or null)",
        "genres": ["list of mentioned genres"],
        "artists_include": ["artists to include"],
        "artists_exclude": ["artists to exclude"],
        "descriptors": ["mood/vibe words like 'chill', 'energetic'"],
        "energy": 0.0-1.0 or null,
        "valence": 0.0-1.0 or null,
        "era": "e.g. '90s' or '1980-1995' or null",
        "is_sufficient": false,
        "user_wants_to_proceed": false
    }
}

Set "is_sufficient" to true when you have at least a recipient/context AND some musical direction.
Set "user_wants_to_proceed" to true if the user explicitly wants to skip ahead.
"""


# Guardrail constants
MAX_INTERVIEW_TURNS = 10       # Hard cap on conversation rounds
MAX_TOKENS_PER_CALL = 300      # Limit response length to control cost
MAX_LLM_CALLS_TOTAL = 15       # Total LLM calls across the session


class InterviewAgent:
    def __init__(self, llm: LLMProvider = None):
        self.profile = UserProfile()
        self.history: List[Dict[str, str]] = []
        self.completed = False
        self.use_llm = False  # Flag to track if real LLM is available
        self.turn_count = 0
        self.llm_call_count = 0
        self.consecutive_failures = 0  # Track failed extractions
        self.escalated = False         # Whether we've upgraded to smart model
        
        # Dependency Injection
        if llm:
            self.llm = llm
            self.use_llm = True
        else:
            # Try to auto-detect a real provider
            try:
                if config.openai_api_key:
                    from .llm.providers.openai import OpenAIProvider
                    self.llm = OpenAIProvider()
                    self.use_llm = True
                    logger.info("Using OpenAI for interview.")
                else:
                    self.llm = MockLLM()
                    logger.info("No API key found. Using MockLLM for interview.")
            except Exception as e:
                logger.warning(f"Failed to init OpenAI provider: {e}. Falling back to Mock.")
                self.llm = MockLLM()
                
        self.model_fast = config.get("fast_llm_model", "gpt-4o-mini")
        self.model_smart = config.get("smart_llm_model", "gpt-4o")

    def start(self) -> str:
        """Begin the interview."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        greeting = "Hi! I'm your Mixtape Curator. Who is this mix for and what's the occasion?"
        self.history.append({"role": "assistant", "content": greeting})
        return greeting

    def process_input(self, user_input: str, user_callback: Callable[[str], None] = None) -> Tuple[str, bool]:
        """
        Process user input, update profile, check sufficiency, and respond.
        Returns: (Response text, Is Complete)
        """
        self.history.append({"role": "user", "content": user_input})
        self.turn_count += 1
        
        # 1. Guardrail: Turn limit
        if self.turn_count >= MAX_INTERVIEW_TURNS:
            if user_callback:
                user_callback(f"[Thought] Reached max interview turns ({MAX_INTERVIEW_TURNS}). Wrapping up.")
            summary = self._generate_summary()
            self.completed = True
            return f"I think I have a good picture! Let me work with what we've got.\n{summary}\n\nStarting generation now...", True
        
        # 2. Transparency Callback (Thought)
        if user_callback:
            user_callback(f"[Thought] Analyzing input: '{user_input}'... (Turn {self.turn_count}/{MAX_INTERVIEW_TURNS})")

        if self.use_llm and self.llm_call_count < MAX_LLM_CALLS_TOTAL:
            return self._process_with_llm(user_input, user_callback)
        else:
            if self.use_llm and self.llm_call_count >= MAX_LLM_CALLS_TOTAL:
                if user_callback:
                    user_callback(f"[Thought] LLM call limit reached ({MAX_LLM_CALLS_TOTAL}). Using local logic.")
            return self._process_with_mock(user_input, user_callback)

    # ─── Real LLM Path ──────────────────────────────────────────────
    def _process_with_llm(self, user_input: str, user_callback=None) -> Tuple[str, bool]:
        """Use the real LLM to extract entities and generate response."""
        self.llm_call_count += 1
        
        # Model escalation: use smart model if we've had consecutive failures
        current_model = self.model_smart if self.escalated else self.model_fast
        
        try:
            result = self.llm.json(
                self.history, 
                model=current_model,
                temperature=0.4
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            if user_callback:
                user_callback(f"[Thought] LLM call failed, falling back to mock logic.")
            return self._process_with_mock(user_input, user_callback)
        
        # Parse structured extraction
        extracted = result.get("extracted", {})
        response_text = result.get("response", "Could you tell me more about what you're looking for?")
        
        # Check if extraction was useful (escalation trigger)
        has_useful_data = any([
            extracted.get("recipient"),
            extracted.get("genres"),
            extracted.get("artists_include"),
            extracted.get("descriptors"),
            extracted.get("energy") is not None,
            extracted.get("valence") is not None,
        ])
        
        if not has_useful_data:
            self.consecutive_failures += 1
            if self.consecutive_failures >= 2 and not self.escalated:
                self.escalated = True
                if user_callback:
                    user_callback(f"[Thought] Extraction quality low. Escalating to {self.model_smart} for better understanding.")
        else:
            self.consecutive_failures = 0  # Reset on success
        
        # Apply extracted data to profile
        self._apply_extraction(extracted)
        
        # Library grounding: check if requested artists exist
        grounding_notes = self._ground_artists(user_callback)
        if grounding_notes:
            # Inject grounding info and ask LLM to regenerate response with this context
            self.history.append({"role": "system", "content": grounding_notes})
            self.llm_call_count += 1
            try:
                regen = self.llm.json(self.history, model=current_model, temperature=0.4)
                response_text = regen.get("response", response_text)
            except Exception:
                pass  # Keep original response if regen fails
        
        # Check sufficiency
        is_sufficient = extracted.get("is_sufficient", False)
        user_wants_proceed = extracted.get("user_wants_to_proceed", False)
        missing = self._check_missing_fields()
        
        if user_callback:
            if missing:
                user_callback(f"[Thought] Still need: {', '.join(missing)}")
            else:
                user_callback("[Thought] Profile looks complete!")
        
        if (is_sufficient or user_wants_proceed) and not missing:
            summary = self._generate_summary()
            self.completed = True
            self.history.append({"role": "assistant", "content": response_text})
            return f"{response_text}\n\n{summary}\n\nStarting generation now...", True
        
        # If LLM thinks sufficient but we still have missing fields, keep going
        if is_sufficient and missing:
            if user_callback:
                user_callback(f"[Thought] LLM thinks we're ready, but I still need: {', '.join(missing)}")
        
        self.history.append({"role": "assistant", "content": response_text})
        return response_text, False

    def _apply_extraction(self, extracted: dict):
        """Apply LLM-extracted entities to the UserProfile."""
        if extracted.get("recipient"):
            self.profile.recipient = extracted["recipient"]
        if extracted.get("context"):
            self.profile.context_notes = extracted["context"]
            
        for g in extracted.get("genres", []):
            if g and g not in self.profile.target_genres:
                self.profile.target_genres.append(g)
                
        for a in extracted.get("artists_include", []):
            if a and a not in self.profile.must_include_artists:
                self.profile.must_include_artists.append(a)
                
        for a in extracted.get("artists_exclude", []):
            if a and a not in self.profile.exclude_artists:
                self.profile.exclude_artists.append(a)
                
        for d in extracted.get("descriptors", []):
            if d and d not in self.profile.target_descriptors:
                self.profile.target_descriptors.append(d)
        
        if extracted.get("energy") is not None:
            self.profile.targets.energy = float(extracted["energy"])
        if extracted.get("valence") is not None:
            self.profile.targets.valence = float(extracted["valence"])

    def _ground_artists(self, user_callback=None) -> Optional[str]:
        """
        Check requested artists against the library.
        - Found (exact/fuzzy): correct the name, report track count
        - Not found: search for similar artists by genre
        Returns a grounding note string or None.
        """
        if library.df.empty:
            return None
            
        parts = []
        corrected_artists = []
        missing_artists = []
        
        for artist in list(self.profile.must_include_artists):
            match = library.search_artist(artist)
            if match:
                # Found (possibly fuzzy-corrected)
                if match != artist:
                    if user_callback:
                        user_callback(f"[Thought] '{artist}' → matched to '{match}' in library.")
                    self.profile.must_include_artists.remove(artist)
                    if match not in self.profile.must_include_artists:
                        self.profile.must_include_artists.append(match)
                
                # Check track count
                tracks = library.get_artist_tracks(match)
                track_count = len(tracks)
                if track_count <= 2:
                    track_names = ', '.join([f"'{t[1]}'" for t in tracks])
                    parts.append(f"'{match}' has only {track_count} track(s) in the library: {track_names}. Mention this to the user and ask if they want to include it.")
                    if user_callback:
                        user_callback(f"[Thought] '{match}' has only {track_count} track(s): {track_names}")
                else:
                    if user_callback:
                        user_callback(f"[Thought] '{match}' found with {track_count} tracks. Good coverage.")
                
                corrected_artists.append(match)
            else:
                missing_artists.append(artist)
        
        # For missing artists, find alternatives by genre
        if missing_artists:
            if user_callback:
                user_callback(f"[Thought] Not in library: {', '.join(missing_artists)}. Searching for similar artists...")
            
            for artist in missing_artists:
                if artist in self.profile.must_include_artists:
                    self.profile.must_include_artists.remove(artist)
                
                # Search by each target genre for alternatives
                found_alts = False
                for genre in self.profile.target_genres:
                    similar = library.search_artists_by_genre(genre, limit=5)
                    if similar:
                        alts = ', '.join(similar[:3])
                        parts.append(f"'{artist}' is not in the library. Similar {genre} artists available: {alts}")
                        found_alts = True
                        break
                
                if not found_alts:
                    parts.append(f"'{artist}' is not in the library and no similar artists were found by genre.")
        
        if not parts:
            return None
            
        note = "LIBRARY GROUNDING: " + ". ".join(parts) + ". Please naturally inform the user about library availability and suggest alternatives where needed."
        if user_callback:
            user_callback(f"[Thought] {note}")
        return note

    # ─── Mock Path (Fallback) ────────────────────────────────────────
    def _process_with_mock(self, user_input: str, user_callback=None) -> Tuple[str, bool]:
        """Fallback: use keyword-based extraction."""
        self._extract_entities_mock(user_input)
        
        missing = self._check_missing_fields()
        
        if not missing:
            if user_callback:
                user_callback("[Thought] Profile sufficient. Generating summary...")
            summary = self._generate_summary()
            self.completed = True
            return f"Great! I have everything I need.\n{summary}\n\nStarting generation now...", True
        
        if user_callback:
            user_callback(f"[Thought] Missing info: {', '.join(missing)}. Formulating question...")

        response = self._generate_question_mock(missing)
        self.history.append({"role": "assistant", "content": response})
        return response, False

    def _extract_entities_mock(self, text: str):
        """Mock extraction logic using keyword matching."""
        text_lower = text.lower()
        
        # Recipient
        if "friend" in text_lower: self.profile.recipient = "Friend"
        elif "workout" in text_lower: self.profile.recipient = "Self (Workout)"
        elif "party" in text_lower: self.profile.recipient = "Party Guests"
        elif self.profile.recipient == "self" and len(self.history) > 2:
             pass

        # Genres
        common_genres = ["rock", "pop", "jazz", "electronic", "hip hop", "metal", "classical"]
        for g in common_genres:
            if g in text_lower and g not in self.profile.target_genres:
                self.profile.target_genres.append(g)

        # Vibe / Content
        if "fast" in text_lower: 
            self.profile.targets.energy = 0.8
            self.profile.target_descriptors.append("fast")
        if "slow" in text_lower: 
            self.profile.targets.energy = 0.3
            self.profile.target_descriptors.append("slow")
        if "happy" in text_lower or "upbeat" in text_lower: 
            self.profile.targets.valence = 0.8
            self.profile.target_descriptors.append("happy")
        if "sad" in text_lower: 
            self.profile.targets.valence = 0.2
            self.profile.target_descriptors.append("sad")

    def _generate_question_mock(self, missing: List[str]) -> str:
        """Mock question generation."""
        if "Recipient/Context" in missing:
            return "Who are we making this mix for today? Or is it for a specific event?"
        if "Musical Direction (Genre/Vibe)" in missing:
            return "Got it. What kind of vibe or genres are we looking for? (e.g. 'Upbeat Pop', 'Mellow Jazz')"
        return "Is there anything else you'd like to add? Specific artists to include or avoid?"

    # ─── Shared Logic ────────────────────────────────────────────────
    def _check_missing_fields(self) -> List[str]:
        """Return list of fields that need more info."""
        missing = []
        
        if not self.profile.recipient:
            missing.append("Recipient/Context")
            
        has_content = (self.profile.target_genres or 
                       self.profile.must_include_artists or 
                       self.profile.targets.energy != 0.5)
                       
        if not has_content:
            missing.append("Musical Direction (Genre/Vibe)")
            
        return missing

    def _generate_summary(self) -> str:
        parts = [f"Context: {self.profile.recipient}"]
        if self.profile.context_notes:
            parts.append(f"Occasion: {self.profile.context_notes}")
        if self.profile.target_genres:
            parts.append(f"Genres: {', '.join(self.profile.target_genres)}")
        if self.profile.must_include_artists:
            parts.append(f"Must Include: {', '.join(self.profile.must_include_artists)}")
        if self.profile.exclude_artists:
            parts.append(f"Exclude: {', '.join(self.profile.exclude_artists)}")
        if self.profile.target_descriptors:
            parts.append(f"Vibe: {', '.join(self.profile.target_descriptors)}")
        parts.append(f"Energy: {self.profile.targets.energy:.1f}, Valence: {self.profile.targets.valence:.1f}")
        return "\n".join(parts)
