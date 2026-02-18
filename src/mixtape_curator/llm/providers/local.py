from typing import List, Dict, Any, Optional
import json
import re
from ..interface import LLMProvider

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
        # This allows the Interviewer to offload 'understanding' to the mock
        
        # We'll just return a dummy confidence and some parsed fields
        # In a real one, we'd use regex or keyword matching on the previous user message
        return {
            "confidence": 0.85,
            "fields": {
                "recipient": "self",
                "target_genres": ["rock"],
                "uniformity": 0.5
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

        # 1. Fix Duration
        if "Duration" in str(violations):
             return {"action": "remove_track", "params": {"track_id": "t0"}, "reasoning": "Removing t0 to fix duration exceedance."}
        
        # 2. Fix Artist Violations
        for v in violations:
            if "Artist" in v and "limit" in v:
                match = re.search(r"Artist (.+) has (\d+) tracks", v)
                if match:
                    artist = match.group(1)
                    tracks = artist_map.get(artist, [])
                    if len(tracks) > 2:
                        return {
                            "action": "remove_track", 
                            "params": {"track_ids": list(tracks[2:])}, 
                            "reasoning": f"Removing {len(tracks)-2} excess tracks from {artist}."
                        }

        # 3. Handle Short Playlist
        if track_count < 10:
             # Look at history to see what we've tried
             if "consult_user" not in prompt:
                 return {
                     "action": "consult_user", 
                     "params": {"question": f"Playlist is too short ({track_count} tracks). Search for more?"}, 
                     "reasoning": "Playlist length below target."
                 }
             elif "search_library" not in prompt:
                 return {
                     "action": "search_library",
                     "params": {"query": "Pop", "limit": 10}, 
                     "reasoning": "User approved expansion."
                 }
             elif "add_track" not in prompt:
                 # Cheat and find some IDs if history doesn't have them easily parseable
                 return {
                     "action": "add_track",
                     "params": {"track_ids": ["t1", "t2", "t3", "t4", "t5"]},
                     "reasoning": "Adding discovered tracks."
                 }
        
        return {"action": "finalize", "params": {}, "reasoning": "All constraints met."}
