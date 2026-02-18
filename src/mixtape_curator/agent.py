import logging
import json
import copy
from typing import List, Dict, Any, Optional
import pandas as pd
from mixtape_curator.models import Playlist, UserProfile
from mixtape_curator.library import library
from mixtape_curator.scoring import scorer
from mixtape_curator.generator import generator
from mixtape_curator.llm.interface import LLMProvider
from mixtape_curator.llm.providers.local import MockLLM
from mixtape_curator.config import config

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
        Build and repair a playlist.
        Supports Incremental Phase (Growth) and Optimization Phase (Refinement).
        """
        history = []
        best_playlist = playlist
        best_score = playlist.scores.total
        stall_count = 0
        
        # Thresholds from config
        ambitious_thresh = config.get("ambitious_threshold", 0.88)
        accept_thresh = config.get("accept_threshold", 0.80)
        
        # Phase detection
        target_duration = config.duration_target_s
        is_growth_phase = len(playlist.track_ids) < 8 or playlist.total_duration_s < (target_duration * 0.7)
        
        current_threshold = accept_thresh if is_growth_phase else ambitious_thresh
        
        self.escalated = True # Use smart model for construction

        for i in range(self.max_iters):
            # 1. Validate constraints
            violations = self._validate_constraints(playlist, profile)
            playlist.violations = violations
            
            # Update Phase
            is_growth_phase = playlist.total_duration_s < (target_duration * 0.85)
            
            # Check acceptance 
            is_valid = not violations
            score_good = playlist.scores.total >= current_threshold
            duration_good = not is_growth_phase # If we are out of growth phase, duration is good
            
            if is_valid and score_good and duration_good:
                logger.info(f"Playlist accepted! Score: {playlist.scores.total:.2f}, Duration: {playlist.total_duration_s}s")
                return playlist
            
            # 2. Plan (Thought)
            phase_msg = "GROWTH PHASE: Focus on finding and adding relevant tracks to meet duration target." if is_growth_phase else "REFINEMENT PHASE: Optimize flow and fit to reach high score."
            prompt = self._construct_prompt(playlist, violations, profile, history, phase_msg)
            
            try:
                response = self.llm.json([{"role": "user", "content": prompt}], model=self.model_smart)
                action = response.get("action")
                params = response.get("params", {})
                reasoning = response.get("reasoning", "")
                
                history.append(f"Iter {i}: Thought: {reasoning} -> Action: {action}")
                
                # 3. Act
                if action == "swap_track":
                    self._tool_swap(playlist, params, profile)
                elif action == "remove_track":
                    self._tool_remove(playlist, params, profile)
                elif action == "add_track":
                    self._tool_add_track(playlist, params, profile)
                elif action == "search_library":
                    results = self._tool_search_library(params, profile)
                    history.append(f"Search Results: {results}")
                elif action == "consult_user":
                    answer = self._tool_consult_user(params)
                    history.append(f"User Answer: {answer}")
                elif action == "finalize":
                    if is_valid and duration_good: return playlist
                    else: history.append(f"Finalize rejected: valid={is_valid}, dur_good={duration_good}")
                
                # Resequence and update stats
                generator.optimize_flow(playlist, profile)
                
                # Track best
                if not violations and playlist.scores.total > best_score:
                    best_score = playlist.scores.total
                    best_playlist = copy.deepcopy(playlist)
                    stall_count = 0
                else:
                    stall_count += 1
                    
                if stall_count >= self.stall_iters:
                    if current_threshold > ambitious_thresh: # If we were aiming high, lower it
                        current_threshold = accept_thresh
                        stall_count = 0
                    else:
                        break

            except Exception as e:
                logger.error(f"ReAct error: {e}")
                break
                
        return best_playlist if not best_playlist.violations else playlist

    def _validate_constraints(self, playlist: Playlist, profile: UserProfile) -> List[str]:
        violations = []
        # Duration Cap
        if playlist.total_duration_s > config.duration_cap_s:
            violations.append(f"Duration {playlist.total_duration_s}s exceeds cap {config.duration_cap_s}s")
            
        # Artist Limit (Max 2)
        artist_counts = {}
        exclude_artists_lower = [a.lower() for a in profile.exclude_artists]
        exclude_genres_lower = [g.lower() for g in profile.exclude_genres]
        exclude_descriptors_lower = [d.lower() for d in profile.exclude_descriptors]
        exclude_track_ids = set(profile.exclude_track_ids)
        
        present_ids = set(playlist.track_ids)
        
        for tid in playlist.track_ids:
            if tid in exclude_track_ids:
                violations.append(f"Forbidden Track ID: {tid}")
                
            t = library.get_track(tid)
            if t:
                # Check Excluded Artist
                if t.artist.lower() in exclude_artists_lower:
                    violations.append(f"Forbidden Artist: {t.artist}")
                
                # Check Excluded Genres
                if exclude_genres_lower:
                    genres = set(t.rym_data.primary_genres + t.rym_data.subgenres)
                    for eg in exclude_genres_lower:
                        if eg in [g.lower() for g in genres]:
                            violations.append(f"Forbidden Genre '{eg}' on {t.title}")

                # Check Excluded Descriptors
                if exclude_descriptors_lower:
                    descriptors = [d.lower() for d in t.rym_data.descriptors]
                    for ed in exclude_descriptors_lower:
                        if ed in descriptors:
                            violations.append(f"Forbidden Descriptor '{ed}' on {t.title}")

                # Artist concentration
                artist_counts[t.artist] = artist_counts.get(t.artist, 0) + 1
        
        # Check Must-Include Artists
        if profile.must_include_artists:
            present_artists = set(a.lower() for a in artist_counts.keys())
            for ma in profile.must_include_artists:
                if ma.lower() not in present_artists:
                    violations.append(f"Missing Must-Include Artist: {ma}")
        
        # Check Must-Include Track IDs
        if profile.must_include_track_ids:
            for mit in profile.must_include_track_ids:
                if mit not in present_ids:
                    violations.append(f"Missing Must-Include Track: {mit}")
        
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

    def _construct_prompt(self, playlist: Playlist, violations: List[str], profile: UserProfile, history: List[str], phase_msg: str) -> str:
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
            "artist_map": artist_map,
            "phase": phase_msg
        }
        return f"""
        You are a Mixtape Curator and A&R agent. Your goal is to build a cohesive musical journey.
        We are not just matching genres; we are creating a VIBE (like an expert human curator).
        
        State: {json.dumps(state)}
        Violations: {violations}
        History: {history[-10:]}
        
        Available Tools:
        - swap_track(old_id, new_id, reasoning): Replace a track. Provide "reasoning" for why the NEW track fits the vibe/journey.
        - remove_track(track_id): Remove a track.
        - add_track(track_ids, reasonings): Add one or more tracks. Provide a dictionary "reasonings" mapping track_id to a short "Why it fits" string.
        - consult_user(question): Ask the user for input.
        - search_library(query, limit=5): Search for tracks.
        - finalize(): If valid and you've provided reasoning for all tracks.
        
        Respond JSON: {{ "reasoning": "thought process", "action": "...", "params": {{...}} }}
        """

    def _tool_swap(self, playlist: Playlist, params: Dict, profile: UserProfile):
        old_id = params.get("old_id")
        new_id = params.get("new_id")
        reasoning = params.get("reasoning", "Fits the curated journey.")
        if old_id in playlist.track_ids:
            idx = playlist.track_ids.index(old_id)
            playlist.track_ids[idx] = new_id
            playlist.track_notes[new_id] = reasoning
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
        reasonings = params.get("reasonings", {})
        if not tids and "track_id" in params:
            tids = [params["track_id"]]
            
        for tid in tids:
            if tid not in playlist.track_ids:
                playlist.track_ids.append(tid)
                playlist.track_notes[tid] = reasonings.get(tid, reasoning if (reasoning := params.get("reasoning")) else "Carefully selected for this mix.")
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
            genres = ", ".join(row['rym_data_primary_genres'][:2])
            descriptors = ", ".join(row['rym_data_descriptors'][:3])
            results.append(
                f"{row['id']}: {row['title']} by {row['artist']} [{genres}] [{descriptors}] ({row['duration_s']}s)"
            )
            
        return results_msg + "\n".join(results)
