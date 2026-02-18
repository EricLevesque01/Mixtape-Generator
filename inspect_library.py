
import sys
import os
sys.path.append(os.path.abspath('src'))
from mixtape_curator.library import library

library.load()
df = library.df
print(f"Total tracks: {len(df)}")


# potential matches for "Synthpop" (primary + sub)
def has_genre(row, genre):
    return genre in row.get('rym_data_primary_genres', []) or genre in row.get('rym_data_subgenres', [])



mask = df.apply(lambda x: has_genre(x, "Synthpop"), axis=1)
synthpop = df[mask]
print(f"Tracks with 'Synthpop' in genres (primary/sub): {len(synthpop)}")

# Count Synthpop by artist
print("Synthpop counts by artist:")
synthpop_counts = synthpop['artist'].value_counts()
print(synthpop_counts)

# Check specifically which 1975 tracks are synthpop
the1975_synth = synthpop[synthpop['artist'] == "The 1975"]
print("\nThe 1975 Synthpop Tracks:")
for title in the1975_synth['title']:
    print(f"- {title}")


