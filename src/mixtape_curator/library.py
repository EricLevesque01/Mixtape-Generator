import pandas as pd
import json
from typing import List, Optional, Tuple
from pathlib import Path
import difflib
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

    def save(self, data_path: str = "data/library.json"):
        """Save current DataFrame back to JSON file."""
        if self.df.empty:
            print("Warning: DataFrame is empty, skipping save.")
            return

        # Convert back to list of dicts
        # Must handle the flattened rym_data structure manually or via json_normalize inverse logic
        # Since json_normalize flattens with '_', we need to unflatten.
        
        records = self.df.to_dict(orient='records')
        output_data = []
        
        for row in records:
            # Reconstruct rym_data dict
            rym = {
                "primary_genres": row.get('rym_data_primary_genres', []),
                "subgenres": row.get('rym_data_subgenres', []),
                "descriptors": row.get('rym_data_descriptors', [])
            }
            
            # Base dict
            item = {k: v for k, v in row.items() if not k.startswith('rym_data_')}
            item['rym_data'] = rym
            output_data.append(item)
            
        root_dir = Path(__file__).parent.parent.parent
        path = root_dir / data_path
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)
        print(f"Saved {len(output_data)} tracks to {path}")
        
    @staticmethod
    def _normalize(s: str) -> str:
        """Normalize artist name: lowercase, strip articles, remove punctuation."""
        import re
        s = s.lower().strip()
        for prefix in ['the ', 'a ', 'an ']:
            if s.startswith(prefix):
                s = s[len(prefix):]
                break
        s = re.sub(r'[^a-z0-9 ]', '', s)
        return re.sub(r'\s+', ' ', s).strip()

    def search_artist(self, query: str, threshold: float = 0.7) -> Optional[str]:
        """
        Fuzzy search for an artist in the library.
        Uses a 3-tier strategy:
          1. Exact case-insensitive match
          2. Exact normalized match (strips articles + punctuation)
          3. Fuzzy match on normalized names
        """
        if self.df.empty:
            return None
            
        all_artists = self.df['artist'].unique().tolist()
        
        # 1. Exact match (case insensitive)
        query_lower = query.lower().strip()
        for a in all_artists:
            if a.lower().strip() == query_lower:
                return a
        
        # 2. Build normalized maps
        query_norm = self._normalize(query)
        
        norm_map = {}
        for a in all_artists:
            key = self._normalize(a)
            norm_map[key] = a
        
        # Exact normalized match
        if query_norm in norm_map:
            return norm_map[query_norm]
        
        # 3. Fuzzy match on normalized names
        norm_candidates = list(norm_map.keys())
        matches = difflib.get_close_matches(query_norm, norm_candidates, n=1, cutoff=threshold)
        if matches:
            return norm_map[matches[0]]
            
        return None

    def search_artists_by_genre(self, genre: str, limit: int = 5) -> List[str]:
        """Find artists in the library that match a genre (primary or sub). Uses substring matching."""
        if self.df.empty:
            return []
        
        genre_lower = genre.lower().strip()
        
        def has_genre(row):
            primary = [g.lower() for g in row.get('rym_data_primary_genres', [])]
            sub = [g.lower() for g in row.get('rym_data_subgenres', [])]
            all_genres = primary + sub
            # Substring match: 'grunge' matches 'Grunge Pop', 'Post-Grunge', etc.
            return any(genre_lower in g or g in genre_lower for g in all_genres)
        
        mask = self.df.apply(has_genre, axis=1)
        matched = self.df[mask]
        
        if matched.empty:
            return []
        
        # Return unique artists, sorted by frequency (most tracks first)
        artist_counts = matched['artist'].value_counts()
        return artist_counts.head(limit).index.tolist()

    def get_artist_tracks(self, artist: str) -> List[Tuple[str, str]]:
        """Return list of (track_id, title) for a given artist."""
        if self.df.empty:
            return []
        matched = self.df[self.df['artist'].str.lower() == artist.lower()]
        return [(row['id'], row['title']) for _, row in matched.iterrows()]

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
            enrichment_source=row.get('enrichment_source'),
            release_year=row.get('release_year')
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
