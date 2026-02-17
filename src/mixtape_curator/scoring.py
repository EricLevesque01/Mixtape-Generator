import math
import numpy as np
from typing import List, Dict
from collections import Counter
from .models import Track, UserProfile, PlaylistScores
from .config import config

class Scorer:
    def __init__(self):
        self.weights = config.weights

    def compute_fit_score(self, tracks: List[Track], profile: UserProfile) -> float:
        """
        Compute Fit Score using Gaussian decay for sonic features
        and strict/loose blending for genre features.
        """
        if not tracks:
            return 0.0

        scores = []
        
        # Pre-compute target vectors
        target_sonic = {
            "energy": profile.targets.energy,
            "valence": profile.targets.valence,
            "intensity": profile.targets.intensity
        }
        
        target_genres = set(profile.target_genres)
        
        for t in tracks:
            # 1. Sonic Fit (Gaussian Decay)
            # Distance squared
            d2 = (
                (t.energy - target_sonic["energy"])**2 +
                (t.valence - target_sonic["valence"])**2 +
                (t.intensity - target_sonic["intensity"])**2
            )
            tolerance = 0.25 # From Spec v2.1.1
            sonic_fit = math.exp(-d2 / (2 * tolerance**2))
            
            # 2. Genre Fit
            # Full credit if primary or subgenre matches any target
            t_genres = set(t.rym_data.primary_genres + t.rym_data.subgenres)
            if not target_genres:
                genre_fit = 1.0 # No preference expressed
            else:
                genre_fit = 1.0 if not t_genres.isdisjoint(target_genres) else 0.0
                
            # Blend based on strictness
            strictness = profile.genre_strictness
            final_fit = (sonic_fit * (1 - strictness)) + (genre_fit * strictness)
            scores.append(final_fit)
            
        return float(np.mean(scores))

    def compute_flow_score(self, tracks: List[Track]) -> float:
        """
        Compute Flow Score.
        Minimizes deltas between adjacent tracks.
        """
        if len(tracks) < 2:
            return 1.0
            
        deltas = []
        for i in range(len(tracks) - 1):
            t1 = tracks[i]
            t2 = tracks[i+1]
            
            # Euclidian distance of key features
            d = math.sqrt(
                (t1.energy - t2.energy)**2 +
                (t1.valence - t2.valence)**2 +
                (t1.intensity - t2.intensity)**2
            )
            # Normalize: max possible distance is sqrt(1+1+1) = 1.732
            # We want score 1.0 for distance 0
            score = max(0.0, 1.0 - (d / 1.732))
            deltas.append(score)
            
        return float(np.mean(deltas))

    def compute_variety_score(self, tracks: List[Track], profile: UserProfile) -> float:
        """
        Compute Variety Score via Entropy alignment to Uniformity target.
        """
        if not tracks:
            return 0.0
            
        # Collect all tokens
        tokens = []
        for t in tracks:
            # Weighted tokens per track
            # This is a simplified implementation of the "per-track distribution" 
            # described in spec. We aggregate all tokens and weight them.
            for g in t.rym_data.primary_genres:
                tokens.extend([g] * 10) # weight 1.0 -> 10 counts
            for s in t.rym_data.subgenres:
                tokens.extend([s] * 7)  # weight 0.7 -> 7 counts
            for d in t.rym_data.descriptors[:5]:
                tokens.extend([d] * 3)  # weight 0.3 -> 3 counts
                
        if not tokens:
            return 0.0
            
        # Calculate Entropy
        counts = Counter(tokens)
        total_tokens = sum(counts.values())
        probs = [c / total_tokens for c in counts.values()]
        
        # Shannon Entropy H
        H = -sum(p * math.log(p) for p in probs)
        
        # Max Possible Entropy (log K)
        K = len(counts)
        if K <= 1:
            measured_diversity = 0.0
        else:
            measured_diversity = H / math.log(K)
            
        # Uniformity Mapping
        # Spec: ideal_diversity = 1 - targets.uniformity
        ideal_diversity = 1.0 - profile.targets.uniformity
        
        # Variety score = proximity to ideal
        variety_score = 1.0 - abs(measured_diversity - ideal_diversity)
        return float(variety_score)

    def score_playlist(self, tracks: List[Track], profile: UserProfile) -> PlaylistScores:
        fit = self.compute_fit_score(tracks, profile)
        flow = self.compute_flow_score(tracks)
        variety = self.compute_variety_score(tracks, profile)
        
        # Placeholders for access/quality (v1 logic)
        access = 0.8 # TODO: Implement actual logic
        quality = 0.8 # TODO: Implement actual logic
        
        total = (
            self.weights["fit"] * fit +
            self.weights["flow"] * flow +
            self.weights["variety"] * variety +
            self.weights["access"] * access +
            self.weights["quality"] * quality
        )
        
        return PlaylistScores(
            fit=round(fit, 2),
            flow=round(flow, 2),
            variety=round(variety, 2),
            accessibility=round(access, 2),
            quality=round(quality, 2),
            total=round(total, 2)
        )

scorer = Scorer()
