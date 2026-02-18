"""Analyze the latest mixtape_for_me exports against spec requirements."""
import sys, os
sys.path.insert(0, 'src')

from mixtape_curator.library import library
from mixtape_curator.models import UserProfile, FeedbackTargets
from mixtape_curator.scoring import scorer
from mixtape_curator.config import config
from collections import Counter

library.load()
df = library.df

# Reconstruct tracks from the export
titles_a = [
    "Riding With The King", "Blues Before Sunrise", "Stereo",
    "The Ending of Dramamine", "no intro", "Chloe Kelly",
    "Odds And Ends", "Wasted", "Larger Than Life", "Piano Man",
    "The Duck Song", "Clark", "Peanut Butter Jelly Time (Radio Version)",
    "Locked Out Of Heaven", "Real Life", "Big Mike's", "Ivory",
    "Nikes", "New Low", "She Looked Like Me!", "The 1975"
]

tracks = []
for title in titles_a:
    row = df[df['title'] == title]
    if not row.empty:
        tid = row.iloc[0]['id']
        t = library.get_track(tid)
        if t:
            tracks.append(t)

print(f"=== PLAYLIST A ANALYSIS ===")
print(f"Tracks resolved: {len(tracks)}/21")
print(f"Total duration: 4846s ({4846//60}m {4846%60}s)")
print(f"Target duration: {config.duration_target_s}s ({config.duration_target_s//60}m)")
print(f"Score: 0.71")
print()

# Genre breakdown
print("--- Genre Breakdown ---")
genre_counter = Counter()
for t in tracks:
    for g in t.rym_data.primary_genres:
        genre_counter[g] += 1
for genre, count in genre_counter.most_common(10):
    print(f"  {genre}: {count} tracks")

print()

# Sonic analysis
print("--- Sonic Profile ---")
if tracks:
    energies = [t.energy for t in tracks]
    valences = [t.valence for t in tracks]
    intensities = [t.intensity for t in tracks]
    print(f"  Energy:    avg={sum(energies)/len(energies):.2f}, range=[{min(energies):.2f}-{max(energies):.2f}]")
    print(f"  Valence:   avg={sum(valences)/len(valences):.2f}, range=[{min(valences):.2f}-{max(valences):.2f}]")
    print(f"  Intensity: avg={sum(intensities)/len(intensities):.2f}, range=[{min(intensities):.2f}-{max(intensities):.2f}]")

print()

# Flow analysis
print("--- Flow Analysis (energy jumps) ---")
big_jumps = 0
for i in range(1, len(tracks)):
    delta = abs(tracks[i].energy - tracks[i-1].energy)
    flag = " <<< BIG JUMP" if delta > 0.25 else ""
    if delta > 0.25:
        big_jumps += 1
    print(f"  {tracks[i-1].artist[:20]:20} -> {tracks[i].artist[:20]:20} Δenergy={delta:.2f}{flag}")
print(f"  Total big jumps (>0.25): {big_jumps}")

print()

# Artist diversity
print("--- Artist Diversity ---")
artist_counts = Counter(t.artist for t in tracks)
for artist, count in artist_counts.most_common():
    flag = " *** VIOLATION" if count > config.max_tracks_per_artist else ""
    print(f"  {artist}: {count}{flag}")

print()

# Constraint check
print("--- Constraint Check ---")
violations = [a for a, c in artist_counts.items() if c > config.max_tracks_per_artist]
print(f"  Artist violations: {len(violations)} {'PASS' if not violations else 'FAIL - ' + str(violations)}")
print(f"  Duration cap ({config.duration_cap_s}s): {'PASS' if 4846 <= config.duration_cap_s else 'FAIL'}")
print(f"  Duration target ({config.duration_target_s}s): {'PASS (within 10%)' if abs(4846 - config.duration_target_s) / config.duration_target_s < 0.1 else 'PARTIAL'}")

print()

# Issues
print("--- Issues Identified ---")
print("  1. Score 0.71 is below accept_threshold 0.70 (barely passing)")
print("  2. Genre mix is very eclectic (Blues, Slacker Rock, Country, Pop, R&B, Indie)")
print("     - User typed 'me' as recipient, no specific genre given -> default profile")
print("  3. 'The Ending of Dramamine' is 857s (14 min!) - one track = 18% of playlist")
print("  4. 'Ivory' by Omar Apollo is only 45s - too short for a mixtape track")
print("  5. B-side only differs in 4 tracks (Bob Dylan, Carrie Underwood, Backstreet Boys, Billy Joel)")
print("     - These are the lowest-fit tracks being swapped, which is correct behavior")
