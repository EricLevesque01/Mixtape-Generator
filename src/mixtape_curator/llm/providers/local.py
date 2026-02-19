from typing import List, Dict, Any, Optional
import json
import re
import re
import random
from ..interface import LLMProvider
from mixtape_curator.library import library

class MockLLM(LLMProvider):
    """
    A sophisticated Mock LLM that follows Spec v2.1.1.
    Handles both the ReAct repair loop and the conversational interview.
    """
    
    def generate(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.2, seed: Optional[int] = None) -> str:
        # Simple text generation for conversational parts if needed
        return "Mock response"

    def json(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.2, seed: Optional[int] = None) -> Dict[str, Any]:
        prompt = messages[-1]["content"].lower()
        
        # --- Interview Classification Mock ---
        if "classify" in prompt or "extract" in prompt:
            return self._mock_interview_extraction(prompt)
            
        # --- ReAct Repair Loop Mock ---
        if "fix the current playlist" in prompt or "action" in prompt:
            return self._mock_repair_action(messages[-1]["content"])
            
        return {"error": "Mock LLM doesn't understand this prompt."}

    def classify(self, messages: List[Dict[str, str]], model: str) -> Dict[str, bool]:
        # Simple relevance check
        return {"is_relevant": True}

    def _mock_interview_extraction(self, prompt: str) -> Dict[str, Any]:
        """Simulates extracting structured data from user interview responses."""
        
        # Check if it's a refinement prompt
        if "rejected" in prompt or "refinement" in prompt:
            return {
                "extracted": {
                    "genres": [],
                    "artists_include": [],
                    "artists_exclude": [],
                    "energy_delta": 0.1,
                    "uniformity_delta": -0.1
                }
            }

        # Handle start of interview
        if "who is this for" in prompt:
             return {
                "response": "Nice to meet you! What kind of music should we include?",
                "extracted": {
                    "recipient": "Self",
                    "is_sufficient": False,
                    "user_wants_to_proceed": False
                }
            }
            
        return {
            "response": "That sounds great. Anything else you want to add?",
            "extracted": {
                "recipient": "Self",
                "genres": ["Rock"],
                "is_sufficient": True,
                "user_wants_to_proceed": True if "yes" in prompt or "go" in prompt else False
            }
        }

    def _mock_repair_action(self, prompt: str) -> Dict[str, Any]:
        """Existing repair logic moved here."""
        # Parse State from prompt
        try:
            state_str = prompt.split("State: ")[1].split("\n")[0]
            state = json.loads(state_str)
        except (IndexError, json.JSONDecodeError):
            return {"action": "finalize", "params": {}, "reasoning": "Parse failure, finalizing."}

        violations = state.get("violations", [])
        artist_map = state.get("artist_map", {})
        track_count = state.get("track_count", 0)
        phase = state.get("phase", "")

        # 1. Growth Phase Priority
        target_count = 12
        if "GROWTH" in phase or track_count < target_count:
            # Try to parse genres from state if available
            genres = state.get("genres", [])
            target_genre = genres[0] if genres else "Pop"
            
            # Use library to find relevant tracks
            # If complex query, we might want to split it. For now, use primary.
            artists = library.search_artists_by_genre(target_genre, limit=20)
            
            candidates = []
            if artists:
                for artist in artists:
                    tracks = library.get_artist_tracks(artist) # returns (id, title)
                    candidates.extend(tracks)
            else:
                # Fallback to general filtered candidates if no specific genre match
                # This simulates "I don't know that genre, here's some good stuff"
                # For mock purposes, just pick some randoms from library if empty
                 pass

            if not candidates:
                 # Last resort fallback if library search failed (e.g. "Modern Indie" might not match "Indie")
                 # Try searching for "Indie" or "Rock" as fallback
                 for fallback in ["Indie", "Rock", "Pop", "Alternative"]:
                     artists = library.search_artists_by_genre(fallback, limit=10)
                     if artists:
                         for artist in artists:
                             candidates.extend(library.get_artist_tracks(artist))
                         if candidates: break
            
            # Shuffle and pick target_count unique artists if possible
            random.shuffle(candidates)
            
            # Filter for unique artists
            selected_tracks = []
            seen_artists = set()
            
            for tid, title in candidates:
                t = library.get_track(tid)
                if not t: continue
                if t.artist in seen_artists: continue
                
                selected_tracks.append(t)
                seen_artists.add(t.artist)
                if len(selected_tracks) >= target_count: break
            
            # If still need more, allow repeats
            if len(selected_tracks) < target_count:
                remaining_needed = target_count - len(selected_tracks)
                others = [t for tid, t_title in candidates if tid not in [x.id for x in selected_tracks]]
                for i in range(min(len(others), remaining_needed)):
                    t = library.get_track(others[i][0])
                    if t: selected_tracks.append(t)

            track_ids = [t.id for t in selected_tracks]
            reasonings = {}
            for t in selected_tracks:
                # Dynamic Logic for "Unique but True"
                vibe_parts = []
                if t.energy > 0.8: vibe_parts.append("high-energy")
                elif t.energy < 0.4: vibe_parts.append("mellow")
                else: vibe_parts.append("mid-tempo")
                
                if t.valence > 0.6: vibe_parts.append("uplifting")
                elif t.valence < 0.4: vibe_parts.append("moody")
                
                desc = " ".join(vibe_parts)
                reasonings[t.id] = f"A {desc} track that aligns with the {target_genre} vibe."

            return {
                "action": "add_track",
                "params": {
                    "track_ids": track_ids,
                    "reasonings": reasonings
                },
                "reasoning": f"Adding {len(track_ids)} tracks matching the {target_genre} vibe."
            }

        # 2. Refinement Phase (Standard Repair)
        if violations:
             # Fix violations (Duration, Artist)
             if "Duration" in str(violations):
                  return {"action": "remove_track", "params": {"track_ids": ["t0"]}, "reasoning": "Removing t0 to fix duration."}
             return {"action": "remove_track", "params": {"track_ids": ["t1"]}, "reasoning": "Fixing violation."}
        
        return {"action": "finalize", "params": {}, "reasoning": "All constraints met and duration reached."}
