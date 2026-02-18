import random
import uuid
import numpy as np
import copy
import math
from typing import List, Optional, Tuple
from mixtape_curator.models import Track, UserProfile, Playlist, PlaylistScores
from mixtape_curator.library import library
from mixtape_curator.scoring import scorer
from mixtape_curator.config import config

class Generator:
    def __init__(self):
        self.rng = random.Random(config.get("rng_seed", 12345))

    def create_draft(self, profile: UserProfile, incremental: bool = False) -> Playlist:
        """
        Create an initial draft playlist.
        If incremental is True, only includes must-have tracks/artists as a 'seed'.
        """
        # 1. Start with must-includes
        draft_tracks: List[Track] = []
        must_have_ids = set(profile.must_include_track_ids)
        all_candidates_unfiltered = library.filter_candidates(profile)
        
        # Must-Include Track IDs
        for t in all_candidates_unfiltered:
            if t.id in must_have_ids:
                draft_tracks.append(t)
                
        # Must-Include Artists
        if profile.must_include_artists:
            for artist in profile.must_include_artists:
                if any(t.artist == artist for t in draft_tracks):
                    continue
                artist_tracks = [t for t in all_candidates_unfiltered if t.artist == artist]
                if artist_tracks:
                    # Sort by fit
                    artist_tracks_scored = [(scorer.compute_fit_score([t], profile), t) for t in artist_tracks]
                    artist_tracks_scored.sort(key=lambda x: x[0], reverse=True)
                    best_track = artist_tracks_scored[0][1]
                    draft_tracks.append(best_track)
                    if best_track.id not in profile.must_include_track_ids:
                        profile.must_include_track_ids.append(best_track.id)

        if not incremental:
            # Fill remaining space
            draft_tracks = self._fill_remaining(draft_tracks, profile)

        current_duration = sum(t.duration_s for t in draft_tracks)
        
        pl = Playlist(
            id=str(uuid.uuid4()),
            track_ids=[t.id for t in draft_tracks],
            total_duration_s=current_duration,
            scores=scorer.score_playlist(draft_tracks, profile)
        )
        return pl

    def _fill_remaining(self, draft_tracks: List[Track], profile: UserProfile) -> List[Track]:
        """Greedy fill to duration target."""
        candidates = library.filter_candidates(profile)
        min_dur = config.get("track_min_duration_s", 60)
        max_dur = config.get("track_max_duration_s", 600)
        candidates = [t for t in candidates if min_dur <= t.duration_s <= max_dur]
        
        current_ids = {t.id for t in draft_tracks}
        pool = [t for t in candidates if t.id not in current_ids]
        
        pool_with_scores = []
        for t in pool:
            score = scorer.compute_fit_score([t], profile)
            pool_with_scores.append((score, t))
        pool_with_scores.sort(key=lambda x: x[0], reverse=True)
        
        current_duration = sum(t.duration_s for t in draft_tracks)
        target = config.duration_target_s
        cap = config.duration_cap_s
        
        draft_artists = {}
        for t in draft_tracks:
             draft_artists[t.artist] = draft_artists.get(t.artist, 0) + 1
        
        # Pass 1: Max 1 per artist
        for score, t in pool_with_scores:
            if current_duration >= target: break
            if current_duration + t.duration_s <= cap:
                if draft_artists.get(t.artist, 0) < 1:
                    draft_tracks.append(t)
                    current_duration += t.duration_s
                    draft_artists[t.artist] = draft_artists.get(t.artist, 0) + 1

        # Pass 2: Fill remaining space up to max_tracks_per_artist (2)
        if current_duration < target:
            limit = config.max_tracks_per_artist
            for score, t in pool_with_scores:
                if current_duration >= target: break
                if t.id not in [x.id for x in draft_tracks]:
                     if current_duration + t.duration_s <= cap:
                        if draft_artists.get(t.artist, 0) < limit:
                            draft_tracks.append(t)
                            current_duration += t.duration_s
                            draft_artists[t.artist] = draft_artists.get(t.artist, 0) + 1
        return draft_tracks

    def optimize_flow(self, playlist: Playlist, profile: UserProfile) -> Playlist:
        """
        Reorder tracks using Greedy Multi-Start algorithm to minimize flow error.
        Now incorporates original album track position as a structural signal.
        """
        tracks = [library.get_track(tid) for tid in playlist.track_ids]
        tracks = [t for t in tracks if t]
        
        if len(tracks) < 3:
            return playlist
            
        # 1. Smarter Seed Selection (Section 7 of Spec)
        # Prioritize tracks that were original album openers (Track 1 or 2)
        opener_indices = [i for i, t in enumerate(tracks) if (t.track_number or 99) <= 2]
        
        # Decide which starting points to try
        num_starts = min(len(tracks), 8)
        if len(opener_indices) > 0:
            # Mix of openers and random candidates
            start_indices = list(set(opener_indices[:4] + self.rng.sample(range(len(tracks)), max(0, num_starts - len(opener_indices[:4])))))
        else:
            start_indices = self.rng.sample(range(len(tracks)), num_starts)
            
        best_sequence = tracks
        best_flow_score = -1.0 # Initialize to force first run
        
        for i in start_indices:
            remaining = tracks.copy()
            current_seq = [remaining.pop(i)]
            
            while remaining:
                last_track = current_seq[-1]
                best_next_idx = -1
                max_score = -float('inf')
                
                # Progress through the playlist (0.0 at start, 1.0 at end)
                playlist_progress = len(current_seq) / len(tracks)
                
                for idx, candidate in enumerate(remaining):
                    # A. Sonic Distance (Primary Signal)
                    d2 = (
                        (last_track.energy - candidate.energy)**2 +
                        (last_track.valence - candidate.valence)**2 +
                        (last_track.intensity - candidate.intensity)**2
                    )
                    sonic_score = 1.0 - math.sqrt(d2 / 3.0)
                    
                    # B. Structural Alignment (The "Original Intent" Signal)
                    # Does this track's original album position match its new playlist position?
                    struct_score = 0.0
                    track_pos = candidate.track_number or 5 # assume middle if unknown
                    total = candidate.total_tracks or 10
                    album_progress = track_pos / total
                    
                    # Bonus if album progress aligns with playlist progress
                    # (e.g. tracks from the end of an album fit better at the end of the mix)
                    alignment = 1.0 - abs(album_progress - playlist_progress)
                    struct_score = 0.15 * alignment # 15% weight for structural alignment
                    
                    # C. Closer Bonus
                    if len(remaining) == 1: # This is the final track
                         if (candidate.track_number and candidate.total_tracks and 
                             candidate.track_number >= candidate.total_tracks):
                             struct_score += 0.2 # Extra 20% boost for true album closers
                    
                    total_score = sonic_score + struct_score
                    
                    if total_score > max_score:
                        max_score = total_score
                        best_next_idx = idx
                
                current_seq.append(remaining.pop(best_next_idx))
            
            # Score this sequence (using standard flow math for comparability)
            flow_score = scorer.compute_flow_score(current_seq)
            if flow_score > best_flow_score:
                best_flow_score = flow_score
                best_sequence = current_seq
                
        # Update playlist
        playlist.track_ids = [t.id for t in best_sequence]
        playlist.scores = scorer.score_playlist(best_sequence, profile)
        
        return playlist
        
generator = Generator()
