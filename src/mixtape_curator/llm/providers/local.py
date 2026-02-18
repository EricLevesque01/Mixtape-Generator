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
        if "GROWTH" in phase or track_count < 8:
            # Need to build. Check if we've searched.
            if "search" not in str(prompt).lower():
                return {
                    "action": "search_library",
                    "params": {"query": "Rock", "limit": 10}, 
                    "reasoning": "Building playlist from seed. Searching for candidates."
                }
            return {
                "action": "add_track",
                "params": {
                    "track_ids": ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"],
                    "reasonings": {
                        "t1": "A perfect opener for the vibe.",
                        "t2": "Bridges the gap to the next section.",
                        "t3": "Adds a nice experimental texture.",
                        "t4": "The emotional centerpiece.",
                        "t5": "Keep the energy high here.",
                        "t6": "A deep cut for variety.",
                        "t7": "A dreamy transition.",
                        "t8": "The ultimate finale track."
                    }
                },
                "reasoning": "Adding discovered tracks to reach duration target and build the journey."
            }

        # 2. Refinement Phase (Standard Repair)
        if violations:
             # Fix violations (Duration, Artist)
             if "Duration" in str(violations):
                  return {"action": "remove_track", "params": {"track_ids": ["t0"]}, "reasoning": "Removing t0 to fix duration."}
             return {"action": "remove_track", "params": {"track_ids": ["t1"]}, "reasoning": "Fixing violation."}
        
        return {"action": "finalize", "params": {}, "reasoning": "All constraints met and duration reached."}
