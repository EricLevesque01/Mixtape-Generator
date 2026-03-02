import sys
sys.path.insert(0, 'src')
from mixtape_curator.library import library

library.load()
df = library.df

print("Columns:", list(df.columns))
print()

# Check if any tracks have non-default values
non_default_energy = df[df['energy'] != 0.5]
print(f"Tracks with non-default energy: {len(non_default_energy)}")

non_default_valence = df[df['valence'] != 0.5]
print(f"Tracks with non-default valence: {len(non_default_valence)}")

# Check descriptors
has_descriptors = df[df['rym_data_descriptors'].apply(len) > 0]
print(f"Tracks with descriptors: {len(has_descriptors)}")

# Sample descriptors
print()
print("Sample descriptors:")
for _, row in has_descriptors.head(5).iterrows():
    print(f"  {row['artist']} - {row['title']}: {row['rym_data_descriptors'][:5]}")

# Genre distribution
print()
from collections import Counter
all_genres = []
for genres in df['rym_data_primary_genres']:
    all_genres.extend(genres)
print("Top 20 genres:", Counter(all_genres).most_common(20))
