import logging
import json
from typing import List, Dict, Any, Optional
import pandas as pd
from .models import Playlist, UserProfile
from .library import library
from .scoring import scorer
from .generator import generator
from .llm.interface import LLMProvider
from .llm.providers.local import MockLLM
from .config import config

logger = logging.getLogger("mixtape_curator")

class ReActAgent:
    def __init__(self, llm: LLMProvider, user_callback=None):
        self.llm = llm
        self.max_iters = config.get("max_repair_iters", 8)
        self.stall_iters = config.get("stall_iters", 2)
        self.user_callback = user_callback if user_callback else (lambda q: input(f"AGENT: {q}\n> "))
        self.model_fast = config.get("fast_llm_model", "gpt-4o-mini")
        self.model_smart = config.get("smart_llm_model", "gpt-4o")
        self.escalated = False
        
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
            
            # Check acceptance (No violations + high score + sufficient length)
            # Spec says: loop until valid OR max iters.
            # We add a check for length < 10 to allow the agent to fix short playlists
            is_valid = not violations
            score_good = playlist.scores.total >= config.get("accept_threshold", 0.70)
            length_good = len(playlist.track_ids) >= 10
            
            if is_valid and score_good and length_good:
                logger.info("Playlist accepted.")
                return playlist
            elif is_valid and not length_good:
                logger.info("Playlist valid but too short. Continuing to agent...")
                
            # Track stalling
            # If we are fixing violations, do not stall based on score
            if len(violations) < len(playlist.violations):
                 # We made progress on violations (not perfect check but okay for now)
                 stall_count = 0
            elif not violations:
                # No violations, check score
                if playlist.scores.total > best_score:
                    best_score = playlist.scores.total
                    best_playlist = playlist
                    stall_count = 0
                else:
                    stall_count += 1
            else:
                 # Still have violations and count didn't naturally decrease?
                 # (Start of loop violations vs current are same object reference in code above, so this logic is tricky)
                 # Simpler: If violations exist, we don't care about score stalling yet.
                 # But we must ensure we aren't looping forever doing nothing.
                 if violations:
                     stall_count = 0 # Assume we are trying to fix them.
                 else:
                     stall_count += 1
                
            if stall_count >= self.stall_iters:
                if not self.escalated:
                    # Escalate to smarter model before giving up
                    self.escalated = True
                    stall_count = 0
                    logger.info(f"Agent stalled. Escalating to {self.model_smart}.")
                    continue
                else:
                    logger.warning("Agent stalled even with smart model. Stopping.")
                    break

            # 2. Plan (Thought)
            # Construct prompt with state
            prompt = self._construct_prompt(playlist, violations, profile, history)
            
            # Call LLM for Action
            # Expected format: {"action": "tool_name", "params": {...}}
            current_model = self.model_smart if self.escalated else self.model_fast
            try:
                response = self.llm.json([{"role": "user", "content": prompt}], model=current_model)
                action = response.get("action")
                params = response.get("params", {})
                reasoning = response.get("reasoning", "")
                
                history.append(f"Iter {i}: Thought: {reasoning} -> Action: {action}")
                
                # 3. Act (Tool Execution)
                if action == "swap_track":
                    self._tool_swap(playlist, params, profile)
                elif action == "remove_track":
                     self._tool_remove(playlist, params, profile)
                elif action == "add_track":
                     self._tool_add_track(playlist, params, profile)
                elif action == "search_library":
                     results = self._tool_search_library(params, profile)
                     history.append(f"Search Results: {results}")
                     stall_count = 0 # Interaction shouldn't count as stalling
                elif action == "consult_user":
                     answer = self._tool_consult_user(params)
                     history.append(f"User Answer: {answer}")
                     stall_count = 0 # Interaction shouldn't count as stalling
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
        exclude_artists_lower = [a.lower() for a in profile.exclude_artists]
        
        for tid in playlist.track_ids:
            t = library.get_track(tid)
            if t:
                # Check Excluded Artist
                if t.artist.lower() in exclude_artists_lower:
                    violations.append(f"Forbidden Artist: {t.artist}")
                
                # Check Excluded Genres
                if profile.exclude_genres:
                    genres = set(t.rym_data.primary_genres + t.rym_data.subgenres)
                    for eg in profile.exclude_genres:
                        if eg.lower() in [g.lower() for g in genres]:
                            violations.append(f"Forbidden Genre '{eg}' on {t.title}")

                # Artist concentration
                artist_counts[t.artist] = artist_counts.get(t.artist, 0) + 1
        
        # Check Must-Include Artists
        if profile.must_include_artists:
            present_artists = set(a.lower() for a in artist_counts.keys())
            for ma in profile.must_include_artists:
                if ma.lower() not in present_artists:
                    violations.append(f"Missing Must-Include Artist: {ma}")
        
        for artist, count in artist_counts.items():
            if count > config.max_tracks_per_artist:
                violations.append(f"Artist {artist} has {count} tracks (limit {config.max_tracks_per_artist})")
                
        return violations
    
    def _update_playlist_stats(self, playlist: Playlist):
        """Recalculate metadata like duration after edits."""
        total_s = 0
        for tid in playlist.track_ids:
            t = library.get_track(tid)
            if t:
                total_s += t.duration_s
        playlist.total_duration_s = total_s

    def _construct_prompt(self, playlist: Playlist, violations: List[str], profile: UserProfile, history: List[str]) -> str:
        # Simplified prompt construction

        # Helper to group tracks by artist for prompt context
        artist_map = {}
        for tid in playlist.track_ids:
            t = library.get_track(tid)
            if t:
                # Type safe access
                artist = getattr(t, 'artist', 'Unknown')
                if artist not in artist_map:
                    artist_map[artist] = []
                artist_map[artist].append(tid)

        state = {
            "duration": playlist.total_duration_s,
            "score": playlist.scores.total,
            "violations": violations,
            "track_count": len(playlist.track_ids),
            "artist_map": artist_map
        }
        return f"""
        You are a playlist curator agent. Fix the current playlist.
        State: {json.dumps(state)}
        Violations: {violations}
        History: {history[-10:]}
        
        Available Tools:
        - swap_track(old_id, new_id): Replace a track. 
        - remove_track(track_id): Remove a track (useful for duration/artist count).
        - add_track(track_ids): Add one or more tracks (list of IDs).
        - consult_user(question): Ask the user for input/decision.
        - search_library(query, limit=5): Search for tracks. Returns list of IDs and Titles.
        - finalize(): If valid and good score.
        
        Respond JSON: {{ "reasoning": "...", "action": "...", "params": {{...}} }}
        """

    def _tool_swap(self, playlist: Playlist, params: Dict, profile: UserProfile):
        old_id = params.get("old_id")
        new_id = params.get("new_id")
        if old_id in playlist.track_ids:
            idx = playlist.track_ids.index(old_id)
            playlist.track_ids[idx] = new_id
            self._update_playlist_stats(playlist)
            
    def _tool_remove(self, playlist: Playlist, params: Dict, profile: UserProfile):
        tids = params.get("track_ids", [])
        if not tids and "track_id" in params:
            tids = [params["track_id"]]
            
        for tid in tids:
            if tid in playlist.track_ids:
                playlist.track_ids.remove(tid)
        self._update_playlist_stats(playlist)

    def _tool_add_track(self, playlist: Playlist, params: Dict, profile: UserProfile):
        tids = params.get("track_ids", [])
        if not tids and "track_id" in params:
            tids = [params["track_id"]]
            
        for tid in tids:
            if tid not in playlist.track_ids:
                playlist.track_ids.append(tid)
        self._update_playlist_stats(playlist)

    def _tool_consult_user(self, params: Dict) -> str:
        question = params.get("question", "Verification needed?")
        return self.user_callback(question)

    def _tool_search_library(self, params: Dict, profile: UserProfile) -> str:
        query = params.get("query", "")
        limit = params.get("limit", 5)
        
        # 1. Fuzzy Artist Search (Grounding)
        artist_match = library.search_artist(query)
        
        results_msg = ""
        found_matches = []
        
        # If artist match found, prioritize those tracks
        if artist_match:
            results_msg = f"Found fuzzy match for artist '{artist_match}'. "
            # Exact lookup on normalized name
            artist_tracks = library.df[library.df['artist'] == artist_match]
            if not artist_tracks.empty:
                # Take top N
                found_matches = artist_tracks.head(limit)
        
        # If no fuzzy match or not enough, do broad search
        if len(found_matches) < limit:
             # Broad filter: Title OR Artist OR Genre
            mask = library.df.apply(lambda row: 
                query.lower() in row['title'].lower() or 
                query.lower() in row['artist'].lower() or
                any(query.lower() in g.lower() for g in row['rym_data_primary_genres']), axis=1)
            
            broad_matches = library.df[mask]
            
            # If we had no artist matches, just take the broad ones
            if len(found_matches) == 0:
                 found_matches = broad_matches.head(limit)
            else:
                 # We have some artist matches, but need more to fill limit
                 # Append broad matches that aren't already included
                 needed = limit - len(found_matches)
                 current_ids = set(found_matches['id'])
                 new_matches = broad_matches[~broad_matches['id'].isin(current_ids)].head(needed)
                 if not new_matches.empty:
                    found_matches = pd.concat([found_matches, new_matches])
        
        if found_matches.empty:
            return "No matches found."
            
        results = []
        for _, row in found_matches.iterrows():
            results.append(f"{row['id']}: {row['title']} by {row['artist']}")
            
        return results_msg + "\n".join(results)

