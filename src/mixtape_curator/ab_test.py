import copy
from typing import List, Tuple
from .models import Track, Playlist, UserProfile
from .library import library
from .scoring import scorer
from .generator import generator
from .config import config

class ABTester:
    def __init__(self):
        self.min_swap = config.get("branch_swap_min", 2)
        self.max_swap = config.get("branch_swap_max", 5)

    def generate_b_side(self, playlist_a: Playlist, profile: UserProfile) -> Playlist:
        """
        Generate Playlist B from A using Base-and-Branch strategy.
        Replaces 20% of tracks (clamped) in weakest scoring dimension.
        """
        # 1. Clone A
        playlist_b = copy.deepcopy(playlist_a)
        original_ids = set(playlist_a.track_ids)
        must_haves = set(profile.must_include_track_ids)
        
        # 2. Determine swap count
        total = len(playlist_b.track_ids)
        swap_count = max(self.min_swap, min(self.max_swap, round(0.2 * total)))
        
        # 3. Identify weakest dimension (simplified)
        # We look at individual track fit scores vs variety contribution
        # For v2.1.1, we'll just swap the Lowest Fit Score tracks that aren't locked.
        
        # Get track objects
        tracks = [library.get_track(tid) for tid in playlist_b.track_ids]
        tracks = [t for t in tracks if t]
        
        # Score each track individually
        track_scores = []
        for i, t in enumerate(tracks):
            if t.id in must_haves:
                continue
            score = scorer.compute_fit_score([t], profile)
            track_scores.append((score, i, t))
            
        # Sort by lowest score
        track_scores.sort(key=lambda x: x[0])
        
        # Select indices to swap
        indices_to_swap = [x[1] for x in track_scores[:swap_count]]
        
        # 4. Find Replacements
        # Get candidates excluding current playlist
        current_b_ids = set(playlist_b.track_ids)
        candidates = library.filter_candidates(profile)
        pool = [t for t in candidates if t.id not in current_b_ids]
        
        # Sort pool by fit score descending
        pool_with_scores = [(scorer.compute_fit_score([t], profile), t) for t in pool]
        pool_with_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Swap
        for idx in indices_to_swap:
            if not pool_with_scores:
                break
            
            # Take best available replacement
            new_score, new_track = pool_with_scores.pop(0)
            
            # Replace in list works because indices are from original list
            # But we must be careful if indices shift. Here we just replace by ID in the list.
            old_id = playlist_b.track_ids[idx]
            playlist_b.track_ids[idx] = new_track.id
            
        # 5. Resequence B
        playlist_b = generator.optimize_flow(playlist_b, profile)
        
        # 6. Final Score B
        tracks_b = [library.get_track(tid) for tid in playlist_b.track_ids if tid]
        playlist_b.scores = scorer.score_playlist(tracks_b, profile)
        playlist_b.id = playlist_a.id + "_B" # simplistic ID
        
        return playlist_b

ab_tester = ABTester()
