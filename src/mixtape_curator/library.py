import pandas as pd
import json
from typing import List, Optional
from pathlib import Path
from .models import Track, RYMData, UserProfile
from .config import config

class Library:
    _instance = None
    df: pd.DataFrame = pd.DataFrame()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Library, cls).__new__(cls)
        return cls._instance

    def load(self, data_path: str = "data/library.json"):
        """Load library from JSON into Pandas DataFrame."""
        root_dir = Path(__file__).parent.parent.parent
        path = root_dir / data_path
        
        if not path.exists():
             # Fallback
            path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(f"Library file not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert to DataFrame
        self.df = pd.json_normalize(data, sep='_')
        
        # Rename normalized columns back to structure if needed, or just keep flat
        # Structure is rym_data.primary_genres -> rym_data_primary_genres
        
        print(f"Loaded {len(self.df)} tracks.")

    def get_track(self, track_id: str) -> Optional[Track]:
        """Convert a row to a Track object."""
        if self.df.empty:
            return None
            
        row = self.df[self.df['id'] == track_id]
        if row.empty:
            return None
        
        row = row.iloc[0]
        return self._row_to_track(row)

    def _row_to_track(self, row: pd.Series) -> Track:
        """Helper to convert DF row to Pydantic model."""
        # reconstruct rym_data
        rym = RYMData(
            primary_genres=row.get('rym_data_primary_genres', []),
            subgenres=row.get('rym_data_subgenres', []),
            descriptors=row.get('rym_data_descriptors', [])
        )
        
        return Track(
            id=row['id'],
            title=row['title'],
            artist=row['artist'],
            duration_s=row['duration_s'],
            energy=row.get('energy', 0.0),
            valence=row.get('valence', 0.0),
            intensity=row.get('intensity', 0.0),
            tempo=row.get('tempo', 120.0),
            danceability=row.get('danceability', 0.5),
            key=row.get('key', 0),
            mode=row.get('mode', 1),
            key_full=row.get('key_full', "Unknown"),
            acousticness=row.get('acousticness', 0.0),
            instrumentalness=row.get('instrumentalness', 0.0),
            speechiness=row.get('speechiness', 0.0),
            liveness=row.get('liveness', 0.0),
            brightness=row.get('brightness', 0.0),
            flatness=row.get('flatness', 0.0),
            entropy=row.get('entropy', 0.0),
            dynamic_range=row.get('dynamic_range', 0.0),
            accessibility=row.get('accessibility', 0.6),
            familiarity=row.get('familiarity', 0.5),
            rating=row.get('rating', 0.0),
            recommendability=row.get('recommendability', 0.5),
            rym_data=rym,
            file_path=row.get('file_path'),
            spotify_uri=row.get('spotify_uri'),
            enrichment_source=row.get('enrichment_source')
        )

    def filter_candidates(self, profile: UserProfile) -> List[Track]:
        """Filter library based on hard constraints."""
        if self.df.empty:
            return []
            
        # Start with all tracks
        candidates = self.df.copy()
        
        # 1. Exclude Artists
        if profile.exclude_artists:
            exclude_lower = [a.lower() for a in profile.exclude_artists]
            candidates = candidates[~candidates['artist'].str.lower().isin(exclude_lower)]
            
        # 2. Exclude Genres (Broad match against primary and sub)
        if profile.exclude_genres:
            # This is tricky with lists in pandas. 
            # We explode or use apply. Apply is easier for now given scale < 10k.
            def has_excluded_genre(row):
                genres = set(row.get('rym_data_primary_genres', []) + row.get('rym_data_subgenres', []))
                return not genres.isdisjoint(profile.exclude_genres)
            
            mask = candidates.apply(has_excluded_genre, axis=1)
            candidates = candidates[~mask]

        # 3. Exclude Descriptors
        if profile.exclude_descriptors:
            def has_excluded_descriptor(row):
                descs = set(row.get('rym_data_descriptors', []))
                return not descs.isdisjoint(profile.exclude_descriptors)
            
            mask = candidates.apply(has_excluded_descriptor, axis=1)
            candidates = candidates[~mask]
            
        # Convert remaining rows to Tracks
        return [self._row_to_track(row) for _, row in candidates.iterrows()]

# Global instance
library = Library()

if __name__ == "__main__":
    # Validation Checkpoint 1
    library.load()
    print("Checkpoint 1 Passed: Library loaded successfully.")
    
    # Test fetch
    t = library.get_track("t1")
    if t and t.title == "Midnight City":
        print("Checkpoint 1 Passed: Track retrieval verification.")
    else:
        print("Checkpoint 1 Failed: Track retrieval mismatch.")
