"""
Patch missing/weak sonic features in library.json.

Fixes:
1. danceability: 1,926 tracks stuck at 0.0 because beat_strength * (1 - flatness) ≈ 0
   for noisy tracks. We add a tempo-based floor so danceable-tempo tracks don't get 0.
2. acousticness: 3,382 tracks at 0.0 — needs the formula from the enrichment script
   reapplied. But wait, these ARE computed. Let's check if they're genuinely non-acoustic.
3. key/mode: best-effort from chroma; hard to improve without re-analysis.
"""
import json
import sys
from pathlib import Path
from collections import Counter

LIBRARY_PATH = Path(__file__).parent.parent / "data" / "library.json"

with open(LIBRARY_PATH, "r", encoding="utf-8") as f:
    tracks = json.load(f)

print(f"Loaded {len(tracks)} tracks.")

# --- Analyze the problem tracks ---
dance_zero = [t for t in tracks if t.get('danceability', 0) == 0.0]
acoust_zero = [t for t in tracks if t.get('acousticness', 0) == 0.0]

print(f"\nDanceability = 0.0: {len(dance_zero)} tracks")
print(f"Acousticness = 0.0: {len(acoust_zero)} tracks")

# Sample some danceability=0 tracks to understand why
print("\n=== Sample danceability=0 tracks ===")
for t in dance_zero[:10]:
    print(f"  {t['artist']:25s} - {t['title']:30s}  tempo={t.get('tempo'):6.1f}  energy={t.get('energy'):.3f}  flatness={t.get('flatness'):.3f}")

# Check: are acousticness=0 tracks actually bright/noisy? That would explain it.
print("\n=== Sample acousticness=0 tracks ===")
for t in acoust_zero[:10]:
    print(f"  {t['artist']:25s} - {t['title']:30s}  brightness={t.get('brightness'):.3f}  flatness={t.get('flatness'):.3f}")

# --- Fix danceability ---
# Better formula: incorporate tempo proximity to dance range (100-130 BPM)
# and energy level, not just beat_strength * (1-flatness)
patched_dance = 0
for t in tracks:
    if t.get('danceability', 0) == 0.0 and t.get('enrichment_source') == 'librosa':
        tempo = t.get('tempo', 120.0)
        energy = t.get('energy', 0.5)
        flatness = t.get('flatness', 0.5)

        # Tempo contribution: peaks at 120 BPM, falls off outside 80-160
        tempo_score = max(0.0, 1.0 - abs(tempo - 120.0) / 60.0)
        
        # Energy contribution: danceable = energetic
        energy_score = energy
        
        # Flatness penalty: very noisy/monotone = less danceable
        flatness_penalty = min(flatness, 1.0) * 0.3
        
        new_dance = round(float(max(0.0, min(1.0, 
            tempo_score * 0.4 + energy_score * 0.4 - flatness_penalty + 0.1
        ))), 3)
        
        t['danceability'] = new_dance
        patched_dance += 1

print(f"\n=== Results ===")
print(f"  Patched danceability: {patched_dance} tracks")

# Save
if patched_dance > 0:
    import shutil
    backup = LIBRARY_PATH.with_suffix(".json.bak")
    shutil.copy(LIBRARY_PATH, backup)
    with open(LIBRARY_PATH, "w", encoding="utf-8") as f:
        json.dump(tracks, f, indent=2)
    print(f"  Library saved. Backup at {backup}")

# Final audit
dance_zero_after = sum(1 for t in tracks if t.get('danceability', 0) == 0.0)
print(f"  Danceability = 0.0 after patch: {dance_zero_after}")
