
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
SYSTEM_PROMPT = """You are a professional Mixtape Curator. 

MISSION: Efficiently extract Recipient, Context, Genres, and 3-4 FOUND Artists.

STRICT GROUNDING RULES:
- Only mention artists or tracks explicitly marked as "FOUND" in the LIBRARY SNAPSHOT.
- If a user mentions a friend's name (e.g. "Tracy"), do NOT treat it as an artist unless it matches a library artist perfectly.
- Artist Limit: Do NOT suggest more than 2 songs per artist.
- NO HALLUCINATIONS. If you don't see it in the snapshot, don't pretend it's there.

ARTIST LIST HANDLING:
- Differentiate clearly between "Must Include" (artists the user LOVES) and "Exclude" (artists they HATE).
- If the user provides a long list of artists (e.g. 5+), simply add them to 'artists_include'. Do NOT ask them to cut the list down yourself; the system will handle that.
- If the user provides a list mixed with "except" or "but not", parse carefully into 'artists_include' and 'artists_exclude'.

OUTPUT FORMAT (JSON ONLY):
{
        "user_wants_to_proceed": bool
    }
}

Set "is_sufficient" to true only if you have the 4 pillars + confirmed library matches for artists.
Set "user_wants_to_proceed" to true ONLY if the user says "Go", "Yes", or "Generate".
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
        self.user_wants_to_proceed_mock = False # Initialize for mock path
        self.ready_to_generate = False
        
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
        self._callback: Optional[Callable[[str], None]] = None

    def _log_thought(self, message: str):
        """Helper to safely call the user callback if it exists."""
        if self._callback is not None:
            self._callback(message)

    def start(self) -> str:
        """Begin the interview."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        greeting = "Hi! I'm your Mixtape Curator. Who is this mix for and what's the occasion?"
        self.history.append({"role": "assistant", "content": greeting})
        return greeting

    def process_input(self, user_input: str, user_callback: Optional[Callable[[str], None]] = None) -> Tuple[str, bool]:
        """
        Process user input, update profile, check sufficiency, and respond.
        Returns: (Response text, Is Complete)
        """
        self._callback = user_callback
        self.history.append({"role": "user", "content": user_input})
        self.turn_count += 1
        
        # 1. Guardrail: Turn limit
        if self.turn_count >= MAX_INTERVIEW_TURNS:
            self._log_thought(f"[Thought] Reached max interview turns ({MAX_INTERVIEW_TURNS}). Wrapping up.")
            summary = self._generate_summary()
            self.completed = True
            return f"I think I have a good picture! Let me work with what we've got.\n{summary}\n\nStarting generation now...", True
        
        # 2. Transparency Callback (Thought)
        self._log_thought(f"[Thought] Analyzing input: '{user_input}'... (Turn {self.turn_count}/{MAX_INTERVIEW_TURNS})")

        if self.use_llm and self.llm_call_count < MAX_LLM_CALLS_TOTAL:
            return self._process_with_llm(user_input, user_callback)
        else:
            if self.use_llm and self.llm_call_count >= MAX_LLM_CALLS_TOTAL:
                self._log_thought(f"[Thought] LLM call limit reached ({MAX_LLM_CALLS_TOTAL}). Using local logic.")
            return self._process_with_mock(user_input, user_callback)

    # ─── Real LLM Path ──────────────────────────────────────────────
    def _process_with_llm(self, user_input: str, user_callback: Optional[Callable[[str], None]] = None) -> Tuple[str, bool]:
        """Use the real LLM with a library snapshot to reduce turns and hallucinations."""
        self.llm_call_count += 1
        current_model = self.model_smart if self.escalated else self.model_fast
        
        # 1. PRE-GROUNDING: Find what's actually in the library related to this turn
        snapshot = self._get_library_snapshot(user_input, user_callback)
        
        # 2. Inject Snapshot into History for this turn only
        prompt_with_grounding = list(self.history)
        if snapshot:
            prompt_with_grounding.append({"role": "system", "content": snapshot})

        try:
            result = self.llm.json(
                prompt_with_grounding, 
                model=current_model,
                temperature=0.2
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._process_with_mock(user_input, user_callback)
        
        # Parse structured extraction
        extracted = result.get("extracted", {})
        response_text = result.get("response", "I'm listening. Tell me more about the vibe.")
        
        # Update profile with extraction
        self._apply_extraction(extracted)
        
        # Check efficiency (escalation trigger)
        missing = self._check_missing_fields()
        is_sufficient = extracted.get("is_sufficient", False)
        user_wants_proceed = extracted.get("user_wants_to_proceed", False)

        if missing:
            self._log_thought(f"[Thought] Still need: {', '.join(missing)}")
        elif is_sufficient and not user_wants_proceed:
            self._log_thought(f"[Thought] Vision complete. Waiting for user to say go.")
        elif user_wants_proceed:
             self._log_thought(f"[Thought] User confirmed readiness. Proceeding.")

        # Finalize if user is ready
        if user_wants_proceed:
             summary = self._generate_summary()
             self.completed = True
             self.history.append({"role": "assistant", "content": response_text})
             return f"{response_text}\n\n(Starting generation...)", True
             
        # If we hit max turns, force wrap-up
        if self.turn_count >= MAX_INTERVIEW_TURNS:
            summary = self._generate_summary()
            self.completed = True
            msg = f"I've got quite a bit of info now! Let's get started on the music.\n\n{summary}"
            self.history.append({"role": "assistant", "content": msg})
            return msg, True

        self.history.append({"role": "assistant", "content": response_text})
        return response_text, False

    def _get_library_snapshot(self, user_input: str, user_callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
        """Quickly search the library for artists/tracks mentioned today or in profile."""
        import re
        # Find entities in current turn
        entities = re.findall(r'"([^"]+)"', user_input) or re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', user_input)
        entities = set(entities)
        
        # If no entities in turn, check some from the profile to keep them in focus
        if not entities:
            entities = set(self.profile.must_include_artists[:3])
            
        if not entities and not self.profile.target_genres:
            return None
            
        found = []
        missing = []
        
        for entity in entities:
            if not entity: continue
            # Artist search - stricter threshold for conversational input
            artist_match = library.search_artist(str(entity), threshold=0.85)
            if artist_match:
                track_count = len(library.df[library.df['artist'] == artist_match])
                note = f"Artist '{artist_match}' FOUND ({track_count} tracks)."
                if artist_match != str(entity):
                    note += f" (Matched from user text '{entity}')"
                found.append(note)
                continue
            
            # Track search
            mask = library.df['title'].str.lower().str.contains(str(entity).lower(), na=False, regex=False)
            matches = library.df[mask]
            if not matches.empty:
                t = matches.iloc[0]
                found.append(f"Track '{t['title']}' by {t['artist']} FOUND.")
                continue
                
            missing.append(f"'{entity}' NOT in library.")
            
        # Genre check & Suggestions
        suggestions = []
        for genre in self.profile.target_genres[-3:]:
            # Get some valid artists for these genres to give the AI a palette
            artists = library.search_artists_by_genre(genre, limit=5)
            if artists:
                found.append(f"Genre '{genre}' has artists in library: {', '.join(artists[:3])}")
            else:
                 missing.append(f"Genre '{genre}' has NO artists in the current library.")

        if not found and not missing:
            return None
            
        snapshot = "LIBRARY SNAPSHOT:\n" + "\n".join(set(found + missing))
        snapshot += "\nSTRICT RULE: Do NOT invent tracklists. Do NOT mention artists marked 'NOT in library'. Use the 'FOUND' artists or 'Genre artists' listed above."
        self._log_thought(f"[Thought] Library Check: {len(found)} available, {len(missing)} missing.")
        return snapshot

    def _validate_constraints(self) -> Optional[str]:
        """
        Check for logic conflicts, caps, or flow requirements.
        Returns a question string if intervention is needed, else None.
        """
        p = self.profile
        
        # 1. Conflict Check: Artist in both Include AND Exclude
        # normalize for comparison
        inc_norm = {a.lower().strip() for a in p.must_include_artists}
        exc_norm = {a.lower().strip() for a in p.exclude_artists}
        
        conflict = inc_norm.intersection(exc_norm)
        if conflict:
            c_artist = list(conflict)[0].title() # pick one
            return f"I noticed a mix-up: {c_artist} is marked as both 'Must Include' and 'Exclude'. Which should it be?"

        # 2. Soft Cap Check (Eco-Modular Trigger)
        # If > 8 artists, confirm Segmented approach
        if len(p.must_include_artists) > 8:
            # Only trigger this once
            if not getattr(self, "_cap_triggered", False):
                self._cap_triggered = True
                self._flow_question_asked = True # Implicitly handled
                p.targets.uniformity = 0.35 # Ensure Eco-Modular
                return "That's a lot of artists! To fit them all, I'll structure this as a 'Segmented Journey' moving through their different vibes. Does that sound good?"

        # 3. Staged Interview Flow (Content -> Structure)
        # If we have enough content (>2 artists) but haven't discussed Flow/Uniformity
        has_content = len(p.must_include_artists) >= 2
        # We check if uniformity is still default (0.35) and if we haven't asked about it
        # Actually, if it IS default 0.35, it means "Eclectic".
        # We want to ask IF the user might want Uniformity.
        # Let's trigger this question if we have content and haven't set a explicit "strictness" or discussed flow.
        # For simplicity, let's use a flag.
        if has_content and not getattr(self, "_flow_question_asked", False):
            self._flow_question_asked = True
            return "We have some great artists lined up. Do you want a consistent vibe for these, or a journey through different styles?"
            
        return None

    def _apply_extraction(self, extraction: dict):
        """Update UserProfile with extracted entities."""
        if not extraction: return

        # Recipient / Context
        if extraction.get("recipient"):
            self.profile.recipient = extraction["recipient"]
        if extraction.get("context_notes"):
            self.profile.context_notes = extraction["context_notes"]
            
        # Artists Include (Append, don't overwrite, to allow accumulation)
        new_includes = extraction.get("artists_include", [])
        for artist in new_includes:
            # Conflict pre-check logic could go here, but we do it in _validate_constraints
            if artist not in self.profile.must_include_artists:
                 self.profile.must_include_artists.append(artist)

        # Artists Exclude
        new_excludes = extraction.get("artists_exclude", [])
        for artist in new_excludes:
             if artist not in self.profile.exclude_artists:
                 self.profile.exclude_artists.append(artist)
                 
        # Genres
        if extraction.get("genres"):
            # Update target genres
            for g in extraction["genres"]:
                if g not in self.profile.target_genres:
                    self.profile.target_genres.append(g)

        # Proceed flag
        if extraction.get("user_wants_to_proceed"):
            # We only proceed if we actually have enough info
            if self._is_sufficient():
                 self.ready_to_generate = True
        
        # Descriptors
        for d in extraction.get("descriptors", []):
            if d and d not in self.profile.target_descriptors:
                self.profile.target_descriptors.append(d)
        
        # Energy and Valence
        if extraction.get("energy") is not None:
            self.profile.targets.energy = float(extraction["energy"])
        if extraction.get("valence") is not None:
            self.profile.targets.valence = float(extraction["valence"])

    def _ground_artists(self, user_callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
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
                    self._log_thought(f"[Thought] '{artist}' → matched to '{match}' in library.")
                    self.profile.must_include_artists.remove(artist)
                    if match not in self.profile.must_include_artists:
                        self.profile.must_include_artists.append(match)
                
                # Check track count
                tracks = library.get_artist_tracks(match)
                track_count = len(tracks)
                if track_count <= 2:
                    track_names = ', '.join([f"'{t[1]}'" for t in tracks])
                    parts.append(f"'{match}' has only {track_count} track(s) in the library: {track_names}. Mention this to the user and ask if they want to include it.")
                    self._log_thought(f"[Thought] '{match}' has only {track_count} track(s): {track_names}")
                else:
                    self._log_thought(f"[Thought] '{match}' found with {track_count} tracks. Good coverage.")
                
                corrected_artists.append(match)
            else:
                missing_artists.append(artist)
        
        # For missing artists, find alternatives by genre
        if missing_artists:
            self._log_thought(f"[Thought] Not in library: {', '.join(missing_artists)}. Searching for similar artists...")
            
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
        self._log_thought(f"[Thought] {note}")
        return note

    def _ground_tracks(self, tracks: List[str], user_callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
        """Match requested song titles to specific track IDs."""
        if not tracks or library.df.empty:
            return None
            
        parts = []
        for title in tracks:
            # Search library for title (exact match first)
            matches = library.df[library.df['title'].str.lower() == title.lower()]
            
            if matches.empty:
                # Try partial match (safe from regex warnings)
                mask = library.df['title'].str.lower().str.contains(title.lower(), na=False, regex=False)
                matches = library.df[mask]

            if not matches.empty:
                # If multiple matches, check if they are all by the same artist
                unique_artists = matches['artist'].unique()
                
                if len(unique_artists) == 1 or len(matches) <= 2:
                    # Auto-select the first one if it's the same artist or only a couple of options
                    # This avoids disambiguation "choice overload"
                    row = matches.iloc[0]
                    track_id = row['id']
                    track_title = row['title']
                    artist = row['artist']
                    
                    if track_id not in self.profile.must_include_track_ids:
                        self.profile.must_include_track_ids.append(track_id)
                    self._log_thought(f"[Thought] Track '{title}' auto-matched to '{track_title}' by {artist} ({track_id}).")
                else:
                    # Truly ambiguous (different artists)
                    options = [f"{m['title']} by {m['artist']}" for _, m in matches.head(3).iterrows()]
                    parts.append(f"Found multiple versions of '{title}': {', '.join(options)}. Please determine which one fits best or ask the user if unsure.")
                    self._log_thought(f"[Thought] Multiple matches for '{title}': {len(matches)} found across {len(unique_artists)} artists.")
            else:
                parts.append(f"Track '{title}' was NOT found in the library. Do NOT recommend it in your tracklist; suggest a similar available song instead.")
                self._log_thought(f"[Thought] Track '{title}' not found.")

        if not parts:
            return None
        return "LIBRARY GROUNDING (TRACKS): " + ". ".join(parts)

    # ─── Mock Path (Fallback) ────────────────────────────────────────
    def _process_with_mock(self, user_input: str, user_callback: Optional[Callable[[str], None]] = None) -> Tuple[str, bool]:
        """Fallback: use keyword-based extraction."""
        self._extract_entities_mock(user_input)
        
        missing = self._check_missing_fields()
        
        # Check readiness
        user_ready = getattr(self, "user_wants_to_proceed_mock", False)

        if not missing:
            if user_ready:
                if user_callback:
                    user_callback("[Thought] Profile sufficient and User ready. Generating summary...")
                summary = self._generate_summary()
                self.completed = True
                return f"Great! I have everything I need.\n{summary}\n\nStarting generation now...", True
            else:
                if user_callback:
                    user_callback("[Thought] Profile sufficient. Waiting for confirmation.")
                return "I think I have a solid idea for the mix. Are you ready for me to generate it?", False
        
        # If missing critical info, but user forces start
        if user_ready and len(missing) < 2: # Allow if only 1 thing missing? No, let's strict.
            # Actually, if user says "generate", we should probably try.
            if user_callback:
                user_callback("[Thought] User forced generation despite missing info.")
            summary = self._generate_summary()
            self.completed = True
            return f"Alright, I'll do my best with what we have!\n{summary}\n\nStarting generation now...", True
        
        self._log_thought(f"[Thought] Missing info: {', '.join(missing)}. Formulating question...")

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
             
        # Readiness
        if any(w in text_lower for w in ["yes", "ready", "generate", "go ahead", "sure"]):
            self.user_wants_to_proceed_mock = True
        else:
            self.user_wants_to_proceed_mock = False

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
    def refine_profile(self, feedback: str, user_callback: Optional[Callable[[str], None]] = None):
        """
        Update the profile based on user feedback when they reject a draft.
        Uses the LLM for intelligent extraction from the feedback string.
        """
        self._callback = user_callback
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
    def _is_sufficient(self) -> bool:
        """Check if we have enough info to generate a draft."""
        # We need at least:
        # 1. Recipient or Context
        # 2. Some musical direction (Genres OR Artists OR Energy/Valence)
        
        has_context = bool(self.profile.recipient and self.profile.recipient != "self") or bool(self.profile.context_notes)
        has_music = bool(self.profile.target_genres or self.profile.must_include_artists or self.profile.targets.energy != 0.5)
        
        return has_context and has_music

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
