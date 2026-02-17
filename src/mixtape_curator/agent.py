import logging
import json
from typing import List, Dict, Any, Optional
from .models import Playlist, UserProfile
from .library import library
from .scoring import scorer
from .generator import generator
from .llm.interface import LLMProvider
from .config import config

logger = logging.getLogger("mixtape_curator")

class ReActAgent:
    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.max_iters = config.get("max_repair_iters", 8)
        self.stall_iters = config.get("stall_iters", 2)
        
    def repair_playlist(self, playlist: Playlist, profile: UserProfile) -> Playlist:
        """
        Execute ReAct loop to repair valid playlist.
        """
        history = []
        best_playlist = playlist
        best_score = playlist.scores.total
        stall_count = 0
        
        for i in range(self.max_iters):
            # 1. Validate constraints
            violations = self._validate_constraints(playlist, profile)
            playlist.violations = violations
            
            # Check acceptance (No violations + high score)
            # Spec says: loop until valid OR max iters.
            if not violations and playlist.scores.total >= config.get("accept_threshold", 0.70):
                logger.info("Playlist accepted.")
                return playlist
                
            # Track stalling
            if playlist.scores.total > best_score:
                best_score = playlist.scores.total
                best_playlist = playlist
                stall_count = 0
            else:
                stall_count += 1
                
            if stall_count >= self.stall_iters:
                logger.warning("Agent stalled. detailed logic TODO: consult user or force diversity.")
                # For now, break and return best
                break

            # 2. Plan (Thought)
            # Construct prompt with state
            prompt = self._construct_prompt(playlist, violations, profile, history)
            
            # Call LLM for Action
            # Expected format: {"action": "tool_name", "params": {...}}
            try:
                response = self.llm.json([{"role": "user", "content": prompt}], model=config.get("default_llm_model", "gpt-4o"))
                action = response.get("action")
                params = response.get("params", {})
                reasoning = response.get("reasoning", "")
                
                history.append(f"Iter {i}: Thought: {reasoning} -> Action: {action}")
                
                # 3. Act (Tool Execution)
                if action == "swap_track":
                    self._tool_swap(playlist, params, profile)
                elif action == "remove_track":
                     self._tool_remove(playlist, params, profile)
                elif action == "finalize":
                    return playlist
                else:
                    logger.warning(f"Unknown action: {action}")
                    
            except Exception as e:
                logger.error(f"Agent error: {e}")
                break
                
            # 4. Resequence & Score
            generator.optimize_flow(playlist, profile)
            # Scoring happens inside optimize_flow
            
        return best_playlist if not violations else playlist # Return valid if possible

    def _validate_constraints(self, playlist: Playlist, profile: UserProfile) -> List[str]:
        violations = []
        # Duration Cap
        if playlist.total_duration_s > config.duration_cap_s:
            violations.append(f"Duration {playlist.total_duration_s}s exceeds cap {config.duration_cap_s}s")
            
        # Artist Limit (Max 2)
        artist_counts = {}
        for tid in playlist.track_ids:
            t = library.get_track(tid)
            if t:
                artist_counts[t.artist] = artist_counts.get(t.artist, 0) + 1
        
        for artist, count in artist_counts.items():
            if count > config.max_tracks_per_artist:
                violations.append(f"Artist {artist} has {count} tracks (limit {config.max_tracks_per_artist})")
                
        return violations

    def _construct_prompt(self, playlist: Playlist, violations: List[str], profile: UserProfile, history: List[str]) -> str:
        # Simplified prompt construction
        state = {
            "duration": playlist.total_duration_s,
            "score": playlist.scores.total,
            "violations": violations,
            "track_count": len(playlist.track_ids)
        }
        return f"""
        You are a playlist curator agent. Fix the current playlist.
        State: {json.dumps(state)}
        Violations: {violations}
        History: {history[-3:]}
        
        Available Tools:
        - swap_track(old_id, new_id): Replace a track. 
        - remove_track(track_id): Remove a track (useful for duration/artist count).
        - finalize(): If valid and good score.
        
        Respond JSON: {{ "reasoning": "...", "action": "...", "params": {{...}} }}
        """

    def _tool_swap(self, playlist: Playlist, params: Dict, profile: UserProfile):
        old_id = params.get("old_id")
        new_id = params.get("new_id")
        if old_id in playlist.track_ids:
            idx = playlist.track_ids.index(old_id)
            playlist.track_ids[idx] = new_id
            
    def _tool_remove(self, playlist: Playlist, params: Dict, profile: UserProfile):
        tid = params.get("track_id")
        if tid in playlist.track_ids:
            playlist.track_ids.remove(tid)

# Need a mock LLM for now since providers aren't implemented
class MockLLM(LLMProvider):
    def json(self, messages, model, **kwargs):
        # deterministically fix simple violations for Checkpoint 5
        prompt = messages[0]["content"]
        if "Duration" in prompt and "exceeds" in prompt:
             # Find a track to remove? Or just pretend.
             # In a real mock we'd parse the state.
             return {"action": "remove_track", "params": {"track_id": "t0"}, "reasoning": "Removing t0 to fix duration."}
        return {"action": "finalize", "params": {}, "reasoning": "Looks good."}

