import random
import uuid
import math
from typing import List, Dict
from mixtape_curator.models import Track, UserProfile, Playlist
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
        track_notes: Dict[str, str] = {}
        must_have_ids = set(profile.must_include_track_ids)
        all_candidates_unfiltered = library.filter_candidates(profile)
        
        # Must-Include Track IDs
        for t in all_candidates_unfiltered:
            if t.id in must_have_ids:
                draft_tracks.append(t)
                track_notes[t.id] = "Must-include: requested by the curator."
                
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
                    track_notes[best_track.id] = f"Anchor artist pick — {artist} sets the DNA for this mix."
                    if best_track.id not in profile.must_include_track_ids:
                        profile.must_include_track_ids.append(best_track.id)

        if not incremental:
            # Check for Segmented Generation Trigger (Eclectic + Multiple Genres)
            is_segmented = profile.targets.uniformity < 0.6 and len(profile.target_genres) > 1
            
            if is_segmented:
                 # "Eco-Modular" Strategy
                 draft_tracks = self._create_segmented_draft(draft_tracks, profile)
            else:
                 # Standard Greedy Fill
                 draft_tracks = self._fill_remaining(draft_tracks, profile)

        # Auto-generate reasoning for any tracks that don't have notes yet
        target_genres = profile.target_genres
        for t in draft_tracks:
            if t.id not in track_notes:
                track_notes[t.id] = self._generate_track_reasoning(t, target_genres)

        current_duration = sum(t.duration_s for t in draft_tracks)
        
        # Use profile duration if set, else config default
        target_s = profile.duration_target_s or config.duration_target_s
        cap_s = min(config.duration_cap_s, max(target_s + 1800, config.duration_cap_s))
        
        pl = Playlist(
            id=str(uuid.uuid4()),
            track_ids=[t.id for t in draft_tracks],
            total_duration_s=current_duration,
            scores=scorer.score_playlist(draft_tracks, profile),
            track_notes=track_notes
        )
        return pl


    def _generate_track_reasoning(self, track: Track, target_genres: List[str]) -> str:
        """Generate a 'Why it fits' reasoning string for a track based on its attributes."""
        parts = []
        
        # Sonic description
        if track.energy > 0.7:
            parts.append("high-energy")
        elif track.energy < 0.35:
            parts.append("mellow")
        else:
            parts.append("mid-tempo")
        
        if track.valence > 0.65:
            parts.append("uplifting")
        elif track.valence < 0.35:
            parts.append("moody")
        
        if track.intensity > 0.7:
            parts.append("intense")
        
        # Genre match
        t_genres = track.rym_data.primary_genres + track.rym_data.subgenres
        matched = [g for g in t_genres if g in target_genres]
        if matched:
            parts.append(f"{matched[0]} fit")
        elif t_genres:
            parts.append(f"{t_genres[0]} texture")
        
        # Descriptor flavor
        if track.rym_data.descriptors:
            top_desc = track.rym_data.descriptors[:2]
            parts.append(", ".join(top_desc))
        
        desc = ", ".join(parts) if parts else "solid sonic match"
        return f"Brings {desc} to the journey."

    def _create_segmented_draft(self, draft_tracks: List[Track], profile: UserProfile) -> List[Track]:
        """
        Eco-Modular Generation:
        Builds the playlist in distinct blocks based on the order of `profile.target_genres`.
        Used for "Eclectic" mixes where the user wants a journey (e.g. Rock -> Folk).
        """
        import difflib # Import for fuzzy matching
        
        target = config.duration_target_s
        cap = config.duration_cap_s
        current_duration = sum(t.duration_s for t in draft_tracks)
        
        # 1. Determine Segments
        # If user provided ordered genres ["Rock", "Folk"], we follow that.
        genres = profile.target_genres
        num_segments = len(genres)
        if num_segments == 0: return self._fill_remaining(draft_tracks, profile)
        
        # Calculate time remaining and split per segment
        remaining_time = max(0, target - current_duration)
        time_per_segment = remaining_time / num_segments
        
        current_ids = {t.id for t in draft_tracks}
        draft_artists = {}
        for t in draft_tracks:
             draft_artists[t.artist] = draft_artists.get(t.artist, 0) + 1
             
        # 2. Fill Each Segment
        for i, genre in enumerate(genres):
            # Find candidates used for this genre
            segment_candidates = []
            
            # Fuzzy match helper
            def is_match(g_target, g_candidate):
                if not g_candidate: return False
                g_target = g_target.lower()
                g_candidate = g_candidate.lower()
                if g_target in g_candidate: return True
                # Fuzzy ratio
                return difflib.SequenceMatcher(None, g_target, g_candidate).ratio() > 0.8

            all_candidates = library.filter_candidates(profile)
            
            # Try specific genre match first
            for t in all_candidates:
                if t.id in current_ids: continue
                
                # Check genre match (Primary or Sub)
                t_genres = t.rym_data.primary_genres + t.rym_data.subgenres
                if any(is_match(genre, g) for g in t_genres):
                    segment_candidates.append(t)
            
            # Fallback: If no tracks found for this genre, use general fit
            if not segment_candidates:
                # print(f"Warning: No tracks found for genre '{genre}'. Using general fit fallback.")
                pool = [t for t in all_candidates if t.id not in current_ids]
                # Sort by general fit to profile
                pool.sort(key=lambda t: scorer.compute_fit_score([t], profile), reverse=True)
                segment_candidates = pool # Take everything as potential candidates
            else:
                # Sort by fit + RYM rating (heavily weighted)
                segment_candidates.sort(key=lambda t: scorer.compute_fit_score([t], profile) * 0.6 + t.rating * 0.4, reverse=True)
            
            # Fill segment - Two Pass Approach
            segment_fill = 0  # Track duration added in this segment
            
            # Pass 1: Strict Unique Artists (Max 1 per artist)
            for t in segment_candidates:
                if segment_fill >= time_per_segment: break
                if current_duration >= target: break
                
                if current_duration + t.duration_s <= cap:
                    if draft_artists.get(t.artist, 0) < 1: # Strict limit 1
                        draft_tracks.append(t)
                        current_ids.add(t.id)
                        current_duration += t.duration_s
                        segment_fill += t.duration_s
                        draft_artists[t.artist] = draft_artists.get(t.artist, 0) + 1

            # Pass 2: Relaxed Limit (Fill up to max_tracks_per_artist) if segment not full
            if segment_fill < time_per_segment:
                for t in segment_candidates:
                    if segment_fill >= time_per_segment: break
                    if current_duration >= target: break
                    if t.id in current_ids: continue # Skip already added
                    
                    if current_duration + t.duration_s <= cap:
                        if draft_artists.get(t.artist, 0) < config.max_tracks_per_artist:
                            draft_tracks.append(t)
                            current_ids.add(t.id)
                            current_duration += t.duration_s
                            segment_fill += t.duration_s
                            draft_artists[t.artist] = draft_artists.get(t.artist, 0) + 1
                        
        return draft_tracks

    def _fill_remaining(self, draft_tracks: List[Track], profile: UserProfile) -> List[Track]:
        """Greedy fill to duration target, anchor-artists first."""
        candidates = library.filter_candidates(profile)
        min_dur = config.get("track_min_duration_s", 60)
        max_dur = config.get("track_max_duration_s", 600)
        candidates = [t for t in candidates if min_dur <= t.duration_s <= max_dur]
        
        current_ids = {t.id for t in draft_tracks}
        draft_artists: dict = {}
        for t in draft_tracks:
            draft_artists[t.artist] = draft_artists.get(t.artist, 0) + 1

        # Use profile duration target if set, else config default
        target = profile.duration_target_s or config.duration_target_s
        cap = config.duration_cap_s
        current_duration = sum(t.duration_s for t in draft_tracks)
        limit = config.max_tracks_per_artist

        # Score all candidates (heavily weighted by RYM rating)
        pool = [t for t in candidates if t.id not in current_ids]
        pool_with_scores = [(scorer.compute_fit_score([t], profile) * 0.6 + t.rating * 0.4, t) for t in pool]
        pool_with_scores.sort(key=lambda x: x[0], reverse=True)

        # --- Anchor Pass: prefer tracks from must_include_artists ---
        anchor_artists = set(a.lower() for a in profile.must_include_artists)
        anchor_pool = [(s, t) for s, t in pool_with_scores if t.artist.lower() in anchor_artists]
        general_pool = [(s, t) for s, t in pool_with_scores if t.artist.lower() not in anchor_artists]
        ordered_pool = anchor_pool + general_pool

        def try_fill(pool_subset, max_per_artist):
            nonlocal current_duration
            for score, t in pool_subset:
                if current_duration >= target:
                    break
                if t.id in current_ids:
                    continue
                if current_duration + t.duration_s > cap:
                    continue
                if draft_artists.get(t.artist, 0) < max_per_artist:
                    draft_tracks.append(t)
                    current_ids.add(t.id)
                    current_duration += t.duration_s
                    draft_artists[t.artist] = draft_artists.get(t.artist, 0) + 1

        # Pass 1: Max 1 per artist (diversity first)
        try_fill(ordered_pool, 1)
        # Pass 2: Allow up to max_tracks_per_artist if still under target
        if current_duration < target:
            try_fill(ordered_pool, limit)

        return draft_tracks

    def optimize_flow(self, playlist: Playlist, profile: UserProfile) -> Playlist:
        """
        Reorder tracks using Greedy Multi-Start algorithm to minimize flow error.
        Now incorporates Intra-Segment Optimization for Eclectic mixes.
        """
        tracks = [library.get_track(tid) for tid in playlist.track_ids]
        tracks_valid = [t for t in tracks if t]
        
        if len(tracks_valid) < 3:
            return playlist

        # Check for Segmented flag (Uniformity < 0.6 + Multiple Genres)
        is_segmented = (profile.targets.uniformity < 0.6 and len(profile.target_genres) > 1)
        
        if is_segmented:
            return self._optimize_flow_segmented(playlist, tracks_valid, profile)
        
        # Standard Global Optimization
        return self._optimize_flow_global(playlist, tracks_valid, profile)

    def _optimize_flow_segmented(self, playlist: Playlist, tracks: List[Track], profile: UserProfile) -> Playlist:
        """
        Optimize flow *within* genre blocks, preserving the macro-structure.
        """
        optimized_tracks = []
        ordered_genres = profile.target_genres
        
        # Group tracks by assumed segment (first matching genre in order)
        # Tracks that match multiple will be assigned to the first one in the list they match
        segments: Dict[str, List[Track]] = {g: [] for g in ordered_genres}
        leftover = []
        
        for t in tracks:
            assigned = False
            t_genres = set(t.rym_data.primary_genres + t.rym_data.subgenres)
            for g in ordered_genres:
                if g in t_genres:
                    segments[g].append(t)
                    assigned = True
                    break # Assign to first matching block
            if not assigned:
                leftover.append(t)
                
        # Optimize each segment independently
        for g in ordered_genres:
             seg_tracks = segments[g]
             if seg_tracks:
                 # Run mini-optimization on this block
                 # We simply sort by energy/key or use a mini-greedy
                 seg_sorted = self._optimize_sequence_greedy(seg_tracks)
                 optimized_tracks.extend(seg_sorted)
                 
        # Append leftovers (maybe at the end or distributed? For now, end)
        if leftover:
            optimized_tracks.extend(self._optimize_sequence_greedy(leftover))
            
        playlist.track_ids = [t.id for t in optimized_tracks]
        playlist.scores = scorer.score_playlist(optimized_tracks, profile)
        return playlist

    def _optimize_flow_global(self, playlist: Playlist, tracks: List[Track], profile: UserProfile) -> Playlist:
        """Wrapper for the original global optimization logic."""
        best_sequence = self._optimize_sequence_greedy(tracks)
        playlist.track_ids = [t.id for t in best_sequence]
        playlist.scores = scorer.score_playlist(best_sequence, profile)
        return playlist
        
    def _optimize_sequence_greedy(self, tracks: List[Track]) -> List[Track]:
        """Core Greedy Logic extracted for reuse."""
        if not tracks: return []
        
        # 1. Smarter Seed Selection
        opener_indices = [i for i, t in enumerate(tracks) if (t.track_number or 99) <= 2]
        num_starts = min(len(tracks), 8)
        
        if len(opener_indices) > 0:
             start_indices = list(set(opener_indices[:4] + self.rng.sample(range(len(tracks)), max(0, num_starts - len(opener_indices[:4])))))
        else:
             start_indices = self.rng.sample(range(len(tracks)), num_starts)
             
        best_sequence = tracks
        best_flow_score = -1.0
        
        for i in start_indices:
            remaining = tracks.copy()
            current_seq = [remaining.pop(i)]
            
            while remaining:
                last_track = current_seq[-1]
                best_next_idx = -1
                max_score = -float('inf')
                
                playlist_progress = len(current_seq) / len(tracks)
                
                for idx, candidate in enumerate(remaining):
                    # A. Sonic Distance
                    d2 = (
                        (last_track.energy - candidate.energy)**2 +
                        (last_track.valence - candidate.valence)**2 +
                        (last_track.intensity - candidate.intensity)**2
                    )
                    sonic_score = 1.0 - math.sqrt(d2 / 3.0)
                    
                    # B. Structural Alignment
                    struct_score = 0.0
                    track_pos = candidate.track_number or 5
                    total = candidate.total_tracks or 10
                    album_progress = track_pos / total
                    alignment = 1.0 - abs(album_progress - playlist_progress)
                    struct_score = 0.15 * alignment
                    
                    # C. Closer Bonus
                    if len(remaining) == 1:
                         if (candidate.track_number and candidate.total_tracks and 
                             candidate.track_number >= candidate.total_tracks):
                             struct_score += 0.2
                    
                    total_score = sonic_score + struct_score
                    
                    if total_score > max_score:
                        max_score = total_score
                        best_next_idx = idx
                
                current_seq.append(remaining.pop(best_next_idx))
            
            flow_score = scorer.compute_flow_score(current_seq)
            if flow_score > best_flow_score:
                best_flow_score = flow_score
                best_sequence = current_seq
                
        return best_sequence

generator = Generator()
