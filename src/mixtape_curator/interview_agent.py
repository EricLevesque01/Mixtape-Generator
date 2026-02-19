
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
SYSTEM_PROMPT = """You are a professional Mixtape Curator, A&R, and Creative Director. 
Your goal is to help the user craft a cohesive musical experience, not just a list of songs.
You are building something special, like a physical CD-R with a curated journey.

Think about the "DNA" of the project:
1. The Mission Statement: What's the mood? (e.g. "Main Character", "Late Night Drive", "Focus")
2. Anchor Artist/Song: Is there a centerpiece we should build around?
3. The Texture: Crisp and modern? Warm and lo-fi? Esoteric?
4. The Journey Length: CD-R (80 mins)? EP style?

Be sophisticated, warm, and highly knowledgeable about music. Use terms like "A&R", "Creative Director", "Centerpiece", "Vibe", and "Sonic Texture".

IMPORTANT: Even if you have enough info, check if there are specific "must-haves" or "exclusions" to ensure better grounding.

Always respond with valid JSON:
{
    "response": "Your conversational response as a curator",
    "extracted": {
        "recipient": "who (or null)",
        "context": "occasion (or null)",
        "genres": ["ordered list of genres for progression"],
        "artists_include": ["artists"],
        "artists_exclude": ["artists"],
        "tracks_include": ["titles"],
        "tracks_exclude": ["titles"],
        "descriptors": ["vibe words like 'crooner', 'glitchy'"],
        "energy": 0.0-1.0 or null,
        "valence": 0.0-1.0 or null,
        "era": "e.g. '80s' or null",
        "is_sufficient": false,
        "user_wants_to_proceed": false
    }
}

If the user requests a specific progression (e.g., "Start with Rock then go into Folk"), ensure the "genres" list reflects that order: ["Rock", "Folk"]. This order will be used to structure the mixtape segments.

Set "is_sufficient" to true only when you have a strong vision for the mix.
Set "user_wants_to_proceed" to true ONLY if the user explicitly says they are ready to generate.
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
        
        # Library grounding: check if requested artists/tracks exist
        notes_artists = self._ground_artists(user_callback)
        notes_tracks = self._ground_tracks(extracted.get("tracks_include", []), user_callback)
        
        grounding_notes = ""
        if notes_artists: grounding_notes += notes_artists + " "
        if notes_tracks: grounding_notes += notes_tracks
        
        if grounding_notes:
            # Inject grounding info and ask LLM to regenerate response with this context
            self.history.append({"role": "system", "content": grounding_notes.strip()})
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
                user_callback("[Thought] Profile looks sufficient, but checking if user wants more.")
        
        # We only finish if the user explicitly wants to proceed or if we hit the limit
        # or if we are sufficient AND the user gives a positive signal (handled by LLM setting user_wants_to_proceed)
        if user_wants_proceed and not missing:
            summary = self._generate_summary()
            self.completed = True
            self.history.append({"role": "assistant", "content": response_text})
            return f"{response_text}\n\n{summary}\n\nStarting generation now...", True
        
        # If we hit max turns, force wrap-up
        if self.turn_count >= MAX_INTERVIEW_TURNS:
            summary = self._generate_summary()
            self.completed = True
            msg = f"I've got quite a bit of info now! Let's get started on the music.\n\n{summary}"
            self.history.append({"role": "assistant", "content": msg})
            return msg, True
        
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
                
        # Tracks are handled via grounding to map to IDs
        # (Already handled in _ground_tracks if they exist)

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
            
        note = "LIBRARY GROUNDING (ARTISTS): " + ". ".join(parts) + ". Please naturally inform the user about library availability (especially mention if an artist is completely missing) and suggest alternatives where needed."
        if user_callback:
            user_callback(f"[Thought] {note}")
        return note

    def _ground_tracks(self, tracks: List[str], user_callback=None) -> Optional[str]:
        """Match requested song titles to specific track IDs."""
        if not tracks or library.df.empty:
            return None
            
        parts = []
        for title in tracks:
            # Search library for title
            # (Case insensitive search in DataFrame)
            matches = library.df[library.df['title'].str.lower() == title.lower()]
            
            if matches.empty:
                # Try partial match or fuzzy?
                mask = library.df['title'].str.lower().str.contains(title.lower(), na=False)
                matches = library.df[mask]

            if not matches.empty:
                if len(matches) == 1:
                    track_id = matches.iloc[0]['id']
                    track_title = matches.iloc[0]['title']
                    artist = matches.iloc[0]['artist']
                    if track_id not in self.profile.must_include_track_ids:
                        self.profile.must_include_track_ids.append(track_id)
                    if user_callback:
                        user_callback(f"[Thought] Track '{title}' matched to '{track_title}' by {artist} ({track_id}).")
                else:
                    # Multiple matches (Disambiguation needed - like Clarity)
                    options = [f"{m['title']} by {m['artist']}" for _, m in matches.head(3).iterrows()]
                    parts.append(f"Found multiple versions of '{title}': {', '.join(options)}. Ask the user to specify which one they want.")
                    if user_callback:
                        user_callback(f"[Thought] Multiple matches for '{title}': {len(matches)} found.")
            else:
                parts.append(f"Track '{title}' was not found in the library.")
                if user_callback:
                    user_callback(f"[Thought] Track '{title}' not found.")

        if not parts:
            return None
        return "LIBRARY GROUNDING (TRACKS): " + ". ".join(parts) + ". Be sure to ask for clarification if there are multiple versions of a song."

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

    # ─── Refinement (Neither Loop) ───────────────────────────────────
    def refine_profile(self, feedback: str, user_callback: Callable[[str], None] = None):
        """
        Update the profile based on user feedback when they reject a draft.
        Uses the LLM for intelligent extraction from the feedback string.
        """
        if user_callback:
            user_callback(f"[Thought] Processing refinement feedback: '{feedback}'")
            
        if not self.use_llm:
            # Fallback to simple keyword logic if no LLM
            self._refine_mock(feedback)
            return

        refine_prompt = f"""
        The user rejected the generated mixtape drafts and provided this feedback: "{feedback}"
        
        Update the structured profile based on this feedback. 
        If they want it faster, increase energy. If they want it more similar, increase uniformity.
        Extract any new artists to include or exclude.
        
        Respond with valid JSON in this format:
        {{
            "extracted": {{
                "genres": ["new genres to add"],
                "artists_include": ["new artists"],
                "artists_exclude": ["artists to now avoid"],
                "descriptors": ["new vibe words"],
                "energy_delta": 0.0 (e.g. +0.2 or -0.2),
                "uniformity_delta": 0.0 (e.g. +0.2 or -0.2),
                "valence_delta": 0.0
            }}
        }}
        """
        
        try:
            current_model = self.model_smart if self.escalated else self.model_fast
            result = self.llm.json([{"role": "user", "content": refine_prompt}], model=current_model)
            ext = result.get("extracted", {})
            
            # Apply changes
            self._apply_refinement(ext)
            
            if user_callback:
                user_callback(f"[Thought] Profile updated based on feedback.")
                
        except Exception as e:
            logger.error(f"Refinement LLM call failed: {e}")
            self._refine_mock(feedback)

    def _apply_refinement(self, ext: dict):
        """Helper to apply LLM-extracted refinement deltas."""
        for g in ext.get("genres", []):
            if g not in self.profile.target_genres:
                self.profile.target_genres.append(g)
        
        for a in ext.get("artists_include", []):
            if a not in self.profile.must_include_artists:
                self.profile.must_include_artists.append(a)
                
        for a in ext.get("artists_exclude", []):
            if a not in self.profile.exclude_artists:
                self.profile.exclude_artists.append(a)
                
        for d in ext.get("descriptors", []):
            if d not in self.profile.target_descriptors:
                self.profile.target_descriptors.append(d)
                
        # Deltas
        self.profile.targets.energy = max(0.0, min(1.0, self.profile.targets.energy + ext.get("energy_delta", 0.0)))
        self.profile.targets.uniformity = max(0.0, min(1.0, self.profile.targets.uniformity + ext.get("uniformity_delta", 0.0)))
        self.profile.targets.valence = max(0.0, min(1.0, self.profile.targets.valence + ext.get("valence_delta", 0.0)))

    def _refine_mock(self, feedback: str):
        """Fallback keyword-based refinement (ported from interview.py)."""
        feedback = feedback.lower()
        if any(w in feedback for w in ["slow", "sleepy", "boring", "low energy", "faster"]):
            self.profile.targets.energy = min(1.0, self.profile.targets.energy + 0.2)
        if any(w in feedback for w in ["fast", "intense", "aggressive", "too hard", "slower"]):
            self.profile.targets.energy = max(0.0, self.profile.targets.energy - 0.2)
        if any(w in feedback for w in ["messy", "random", "all over", "inconsistent", "too eclectic"]):
            self.profile.targets.uniformity = min(1.0, self.profile.targets.uniformity + 0.2)
        if any(w in feedback for w in ["samey", "repetitive", "boring", "too similar", "vary"]):
            self.profile.targets.uniformity = max(0.0, self.profile.targets.uniformity - 0.2)
        
        # Simple exclusion
        if "exclude" in feedback or "no " in feedback:
            for word in ["exclude", "no ", "not ", "dont want ", "don't want "]:
                if word in feedback:
                    parts = feedback.split(word)
                    if len(parts) > 1:
                        artist = parts[1].split(',')[0].strip().title()
                        if artist and artist not in self.profile.exclude_artists:
                            self.profile.exclude_artists.append(artist)

    # ─── Shared Logic ────────────────────────────────────────────────
    def _check_missing_fields(self) -> List[str]:
        """Return list of fields that need more info."""
        missing = []
        
        if not self.profile.recipient or self.profile.recipient == "self":
            # If it's just 'self', check if we have context
            if not self.profile.context_notes:
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
