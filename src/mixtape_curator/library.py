import pandas as pd
import json
from typing import List, Optional, Tuple, Set, Dict
from collections import defaultdict
from pathlib import Path
import difflib
from .models import Track, RYMData, UserProfile

class Library:
    _instance = None
    df: pd.DataFrame = pd.DataFrame()
    # Inverted indexes for O(1) genre/descriptor lookups
    genre_index: Dict[str, Set[str]] = {}      # genre_lower -> set(track_ids)
    subgenre_index: Dict[str, Set[str]] = {}   # subgenre_lower -> set(track_ids)
    descriptor_index: Dict[str, Set[str]] = {}  # descriptor_lower -> set(track_ids)
    artist_genre_index: Dict[str, Set[str]] = {} # genre_lower -> set(artist_names)

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
        
        # Build inverted indexes for fast genre/descriptor lookups
        self._build_indexes()
        
        print(f"Loaded {len(self.df)} tracks.")

    def _build_indexes(self):
        """Build inverted indexes from the DataFrame for O(1) tag lookups."""
        self.genre_index = defaultdict(set)
        self.subgenre_index = defaultdict(set)
        self.descriptor_index = defaultdict(set)
        self.artist_genre_index = defaultdict(set)
        
        for _, row in self.df.iterrows():
            tid = row['id']
            artist = row.get('artist', '')
            
            genres = row.get('rym_data_primary_genres', [])
            if isinstance(genres, list):
                for g in genres:
                    g_lower = g.lower()
                    self.genre_index[g_lower].add(tid)
                    self.artist_genre_index[g_lower].add(artist)
            
            subs = row.get('rym_data_subgenres', [])
            if isinstance(subs, list):
                for s in subs:
                    s_lower = s.lower()
                    self.subgenre_index[s_lower].add(tid)
                    self.artist_genre_index[s_lower].add(artist)
            
            descs = row.get('rym_data_descriptors', [])
            if isinstance(descs, list):
                for d in descs:
                    self.descriptor_index[d.lower()].add(tid)

    def search_by_tags(self, genres: List[str] = None, descriptors: List[str] = None,
                       limit: int = 20) -> List[str]:
        """Find track IDs matching any of the given genres/descriptors using the inverted index.
        Returns up to `limit` track IDs sorted by number of matching tags (most relevant first)."""
        from collections import Counter
        tag_hits: Counter = Counter()
        
        for g in (genres or []):
            g_lower = g.lower()
            # Check primary genres
            tag_hits.update(self.genre_index.get(g_lower, set()))
            # Check subgenres
            tag_hits.update(self.subgenre_index.get(g_lower, set()))
        
        for d in (descriptors or []):
            tag_hits.update(self.descriptor_index.get(d.lower(), set()))
        
        # Return track IDs sorted by hit count (most matching tags first)
        return [tid for tid, _ in tag_hits.most_common(limit)]

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
        """Find artists matching a genre using the inverted index. O(1) lookup."""
        if self.df.empty:
            return []
        
        genre_lower = genre.lower().strip()
        artists = self.artist_genre_index.get(genre_lower, set())
        
        if not artists:
            # Substring fallback for partial matches (e.g. 'indie' matches 'indie folk')
            for key, artist_set in self.artist_genre_index.items():
                if genre_lower in key or key in genre_lower:
                    artists = artists | artist_set
        
        if not artists:
            return []
        
        # Sort by track count (most tracks first)
        artist_counts = self.df[self.df['artist'].isin(artists)]['artist'].value_counts()
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
        import math
        
        gaps = []  # Track which fields were missing/defaulted
        
        def _sf(field_name, val, default=0.0):
            """Safe float: coerce NaN/None to default, record gap."""
            if val is None:
                gaps.append(field_name)
                return default
            try:
                f = float(val)
                if math.isnan(f):
                    gaps.append(field_name)
                    return default
                return f
            except (ValueError, TypeError):
                gaps.append(field_name)
                return default
        
        def _si(field_name, val, default=None):
            """Safe int: coerce NaN/None to default, record gap."""
            if val is None:
                gaps.append(field_name)
                return default
            try:
                f = float(val)
                if math.isnan(f):
                    gaps.append(field_name)
                    return default
                return int(f)
            except (ValueError, TypeError):
                gaps.append(field_name)
                return default
        
        def _sl(field_name, val):
            """Safe list: ensure we get a list, not NaN, record gap."""
            if isinstance(val, list):
                if not val:
                    gaps.append(field_name)
                return val
            gaps.append(field_name)
            return []
        
        def _ss(val, default=None):
            """Safe string: coerce NaN to default (None)."""
            if val is None:
                return default
            try:
                if isinstance(val, float) and math.isnan(val):
                    return default
            except (ValueError, TypeError):
                pass
            return str(val) if val != "" else default
        
        rym = RYMData(
            primary_genres=_sl('primary_genres', row.get('rym_data_primary_genres', [])),
            subgenres=_sl('subgenres', row.get('rym_data_subgenres', [])),
            descriptors=_sl('descriptors', row.get('rym_data_descriptors', []))
        )
        
        return Track(
            id=row['id'],
            title=_ss(row.get('title'), 'Unknown Title') or 'Unknown Title',
            artist=_ss(row.get('artist'), 'Unknown Artist') or 'Unknown Artist',
            album=_ss(row.get('album')),
            duration_s=_si('duration_s', row.get('duration_s', 0), 0),
            energy=_sf('energy', row.get('energy'), 0.5),
            valence=_sf('valence', row.get('valence'), 0.5),
            intensity=_sf('intensity', row.get('intensity'), 0.5),
            tempo=_sf('tempo', row.get('tempo'), 120.0),
            danceability=_sf('danceability', row.get('danceability'), 0.5),
            key=_si('key', row.get('key'), 0),
            mode=_si('mode', row.get('mode'), 1),
            key_full=_ss(row.get('key_full'), "Unknown") or "Unknown",
            acousticness=_sf('acousticness', row.get('acousticness'), 0.0),
            instrumentalness=_sf('instrumentalness', row.get('instrumentalness'), 0.0),
            speechiness=_sf('speechiness', row.get('speechiness'), 0.0),
            liveness=_sf('liveness', row.get('liveness'), 0.0),
            brightness=_sf('brightness', row.get('brightness'), 0.0),
            flatness=_sf('flatness', row.get('flatness'), 0.0),
            entropy=_sf('entropy', row.get('entropy'), 0.0),
            dynamic_range=_sf('dynamic_range', row.get('dynamic_range'), 0.0),
            accessibility=_sf('accessibility', row.get('accessibility'), 0.6),
            familiarity=_sf('familiarity', row.get('familiarity'), 0.5),
            rating=_sf('rating', row.get('rating'), 0.0),
            recommendability=_sf('recommendability', row.get('recommendability'), 0.5),
            rym_data=rym,
            file_path=_ss(row.get('file_path')),
            spotify_uri=_ss(row.get('spotify_uri')),
            enrichment_source=_ss(row.get('enrichment_source')),
            release_year=_si('release_year', row.get('release_year')),
            track_number=_si('track_number', row.get('track_number')),
            total_tracks=_si('total_tracks', row.get('total_tracks')),
            disc_number=_si('disc_number', row.get('disc_number')),
            enrichment_gaps=gaps
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
