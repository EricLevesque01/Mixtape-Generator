import random
import uuid
import numpy as np
import copy
from typing import List, Optional, Tuple
from .models import Track, UserProfile, Playlist, PlaylistScores
from .library import library
from .scoring import scorer
from .config import config

class Generator:
    def __init__(self):
        self.rng = random.Random(config.get("rng_seed", 12345))

    def create_draft(self, profile: UserProfile) -> Playlist:
        """
        Create an initial draft playlist satisfying hard constraints
        and aiming for duration target.
        """
        # 1. candidate selection
        candidates = library.filter_candidates(profile)
        
        # 1b. Per-track duration filter — skip intros/skits and epic-length tracks
        min_dur = config.get("track_min_duration_s", 60)
        max_dur = config.get("track_max_duration_s", 600)
        candidates = [t for t in candidates if min_dur <= t.duration_s <= max_dur]
        
        # 2. must-include handling
        draft_tracks: List[Track] = []
        must_have_ids = set(profile.must_include_track_ids)
        
        # Add specific track IDs first (must-includes bypass duration filter)
        all_candidates_unfiltered = library.filter_candidates(profile)
        for t in all_candidates_unfiltered:
            if t.id in must_have_ids:
                draft_tracks.append(t)
                
        # TODO: Add logic for must_include_artists
        
        # Remove already added from pool
        current_ids = {t.id for t in draft_tracks}
        pool = [t for t in candidates if t.id not in current_ids]
        
        # 3. Fill to duration target (Greedy fit by highest fit score)
        # Calculate fit scores for entire pool effectively efficiently
        # For MVP, just loop.
        pool_with_scores = []
        for t in pool:
            # Quick fit score check (just individual track fit)
            score = scorer.compute_fit_score([t], profile)
            pool_with_scores.append((score, t))
            
        # Sort by best fit descending
        pool_with_scores.sort(key=lambda x: x[0], reverse=True)
        
        current_duration = sum(t.duration_s for t in draft_tracks)
        target = config.duration_target_s
        cap = config.duration_cap_s
        
        # 2-Pass Selection to prioritize diversity
        # Pass 1: Add best track from each artist (max 1 per artist)
        draft_artists = {t.artist: 1 for t in draft_tracks}
        
        for score, t in pool_with_scores:
            if current_duration >= target:
                break
                
            if current_duration + t.duration_s <= cap:
                # Check artist count
                count = draft_artists.get(t.artist, 0)
                if count < 1:
                    draft_tracks.append(t)
                    current_duration += t.duration_s
                    draft_artists[t.artist] = count + 1

        # Pass 2: Fill remaining space up to max_tracks_per_artist (2)
        if current_duration < target:
            limit = config.max_tracks_per_artist
            for score, t in pool_with_scores:
                if current_duration >= target:
                    break
                    
                if t.id not in [x.id for x in draft_tracks]: # Avoid duplicates
                     if current_duration + t.duration_s <= cap:
                        count = draft_artists.get(t.artist, 0)
                        if count < limit:
                            draft_tracks.append(t)
                            current_duration += t.duration_s
                            draft_artists[t.artist] = count + 1
                
        # 4. Create Playlist Object
        # Note: Order is currently just "Must Haves" + "Best Fit Descending"
        # Sequencing happens next step.
        pl = Playlist(
            id=str(uuid.uuid4()),
            track_ids=[t.id for t in draft_tracks],
            total_duration_s=current_duration,
            scores=scorer.score_playlist(draft_tracks, profile)
        )
        return pl

    def optimize_flow(self, playlist: Playlist, profile: UserProfile) -> Playlist:
        """
        Reorder tracks using Greedy Multi-Start algorithm to minimize flow error.
        """
        tracks = [library.get_track(tid) for tid in playlist.track_ids]
        tracks = [t for t in tracks if t] # filter nones
        
        if len(tracks) < 3:
            return playlist
            
        best_sequence = tracks
        best_flow_score = scorer.compute_flow_score(tracks)
        
        # Multi-start: Try starting with different tracks
        # Limit starts to avoid N^2 on large lists
        num_starts = min(len(tracks), 10) 
        
        for i in range(num_starts):
            # Pick a seed (rotate through first 10 high-fit tracks maybe? or random?)
            # Spec says "Iteratively append best transition"
            
            remaining = tracks.copy()
            # Try starting with track i
            current_seq = [remaining.pop(i)]
            
            while remaining:
                last_track = current_seq[-1]
                # Find best next track
                best_next_idx = -1
                min_dist = float('inf')
                
                for idx, candidate in enumerate(remaining):
                    # Distance metric from spec flow score logic
                    d = (
                        (last_track.energy - candidate.energy)**2 +
                        (last_track.valence - candidate.valence)**2 +
                        (last_track.intensity - candidate.intensity)**2
                    )
                    if d < min_dist:
                        min_dist = d
                        best_next_idx = idx
                
                current_seq.append(remaining.pop(best_next_idx))
            
            # Score this sequence
            flow_score = scorer.compute_flow_score(current_seq)
            if flow_score > best_flow_score:
                best_flow_score = flow_score
                best_sequence = current_seq
                
        # Update playlist
        playlist.track_ids = [t.id for t in best_sequence]
        playlist.scores = scorer.score_playlist(best_sequence, profile)
        
        return playlist
        
generator = Generator()
