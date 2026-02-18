"""
enrich_sonic_from_genres.py

Derives synthetic energy/valence/intensity/accessibility values from RYM genre
taxonomy since we have no Spotify audio features. This makes scoring and flow
optimization actually meaningful.

Run once: python scripts/enrich_sonic_from_genres.py
Updates data/library.json in-place.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Genre → (energy, valence, intensity) mappings
# Values in [0, 1]. Based on general musical knowledge of genre characteristics.
GENRE_SONIC = {
    # High energy, high intensity
    "Hard Rock":        (0.85, 0.55, 0.85),
    "Heavy Metal":      (0.90, 0.40, 0.95),
    "Punk Rock":        (0.90, 0.60, 0.90),
    "Metal":            (0.90, 0.40, 0.95),
    "Thrash Metal":     (0.95, 0.35, 0.98),
    "Death Metal":      (0.95, 0.25, 0.99),
    "Metalcore":        (0.90, 0.40, 0.92),
    "Electronic":       (0.80, 0.60, 0.75),
    "Dance":            (0.85, 0.70, 0.75),
    "Techno":           (0.85, 0.50, 0.80),
    "House":            (0.80, 0.65, 0.75),
    "Drum and Bass":    (0.90, 0.55, 0.85),
    "Dubstep":          (0.85, 0.50, 0.88),
    "EDM":              (0.85, 0.65, 0.80),
    "Trance":           (0.80, 0.60, 0.78),
    "Electropop":       (0.75, 0.70, 0.65),
    "Synthpop":         (0.70, 0.65, 0.60),
    "New Wave":         (0.65, 0.60, 0.55),
    "Post-Punk":        (0.65, 0.45, 0.60),
    "Industrial":       (0.80, 0.35, 0.85),
    "EBM":              (0.80, 0.40, 0.82),

    # Mid energy
    "Rock":             (0.70, 0.55, 0.65),
    "Alternative Rock": (0.65, 0.50, 0.60),
    "Indie Rock":       (0.60, 0.55, 0.55),
    "Pop Rock":         (0.65, 0.65, 0.55),
    "Pop":              (0.60, 0.70, 0.50),
    "Teen Pop":         (0.65, 0.75, 0.50),
    "Country Pop":      (0.55, 0.65, 0.45),
    "Country Rock":     (0.65, 0.55, 0.55),
    "Country":          (0.50, 0.60, 0.45),
    "Funk":             (0.75, 0.75, 0.65),
    "Funk Rock":        (0.75, 0.65, 0.65),
    "R&B":              (0.60, 0.65, 0.50),
    "Contemporary R&B": (0.60, 0.65, 0.50),
    "Alternative R&B":  (0.55, 0.55, 0.50),
    "Hip Hop":          (0.70, 0.55, 0.65),
    "Rap":              (0.70, 0.50, 0.65),
    "Trap":             (0.72, 0.45, 0.70),
    "Soul":             (0.55, 0.65, 0.50),
    "Pop Soul":         (0.55, 0.70, 0.48),
    "Neo-Soul":         (0.50, 0.60, 0.45),
    "Gospel":           (0.60, 0.75, 0.55),
    "Reggae":           (0.55, 0.70, 0.45),
    "Ska":              (0.70, 0.75, 0.60),
    "Jazz":             (0.45, 0.60, 0.40),
    "Blues":            (0.50, 0.45, 0.50),
    "Blues Rock":       (0.65, 0.50, 0.60),
    "Soft Rock":        (0.45, 0.65, 0.35),
    "Piano Rock":       (0.55, 0.60, 0.50),
    "Glam Rock":        (0.75, 0.65, 0.70),
    "Psychedelic Rock": (0.65, 0.55, 0.60),
    "Progressive Rock": (0.65, 0.50, 0.60),
    "Folk Rock":        (0.50, 0.60, 0.45),
    "Indie Folk":       (0.40, 0.60, 0.35),
    "Slacker Rock":     (0.50, 0.55, 0.45),

    # Low energy, chill
    "Singer-Songwriter":(0.35, 0.60, 0.30),
    "Folk":             (0.35, 0.60, 0.30),
    "Acoustic":         (0.30, 0.60, 0.25),
    "Ambient":          (0.20, 0.55, 0.15),
    "Chillout":         (0.25, 0.60, 0.20),
    "Lo-Fi":            (0.30, 0.55, 0.25),
    "Bedroom Pop":      (0.35, 0.60, 0.30),
    "Dream Pop":        (0.35, 0.55, 0.30),
    "Shoegaze":         (0.45, 0.45, 0.50),
    "Post-Rock":        (0.50, 0.45, 0.55),
    "Classical":        (0.30, 0.55, 0.25),
    "Opera":            (0.45, 0.55, 0.45),
    "Soundtrack":       (0.45, 0.55, 0.40),
    "Christmas":        (0.45, 0.70, 0.35),
    "Holiday":          (0.45, 0.70, 0.35),
    "Easy Listening":   (0.30, 0.65, 0.20),
    "Adult Contemporary":(0.35, 0.65, 0.28),
    "Soft Pop":         (0.35, 0.65, 0.28),
}

# Accessibility: how mainstream/accessible a genre is
GENRE_ACCESSIBILITY = {
    "Pop": 0.90, "Teen Pop": 0.90, "Pop Rock": 0.80, "Country Pop": 0.80,
    "Synthpop": 0.75, "Electropop": 0.75, "Dance": 0.80, "EDM": 0.75,
    "R&B": 0.75, "Contemporary R&B": 0.75, "Hip Hop": 0.70, "Rap": 0.65,
    "Rock": 0.70, "Alternative Rock": 0.65, "Indie Rock": 0.55,
    "Hard Rock": 0.65, "Heavy Metal": 0.45, "Metal": 0.40,
    "Singer-Songwriter": 0.65, "Folk": 0.55, "Indie Folk": 0.55,
    "Country": 0.70, "Soul": 0.70, "Neo-Soul": 0.60,
    "Jazz": 0.50, "Blues": 0.50, "Classical": 0.55, "Opera": 0.45,
    "Ambient": 0.40, "Electronic": 0.55, "Techno": 0.45, "House": 0.60,
    "Soundtrack": 0.65, "Christmas": 0.85,
}

DEFAULT_SONIC = (0.55, 0.60, 0.50)
DEFAULT_ACCESSIBILITY = 0.60

def get_sonic_for_genres(genres: list) -> tuple:
    """Average sonic values across matched genres, fallback to default."""
    matched = []
    for g in genres:
        if g in GENRE_SONIC:
            matched.append(GENRE_SONIC[g])
        else:
            # Try partial match
            for key, val in GENRE_SONIC.items():
                if key.lower() in g.lower() or g.lower() in key.lower():
                    matched.append(val)
                    break
    
    if not matched:
        return DEFAULT_SONIC
    
    avg_e = sum(m[0] for m in matched) / len(matched)
    avg_v = sum(m[1] for m in matched) / len(matched)
    avg_i = sum(m[2] for m in matched) / len(matched)
    return (round(avg_e, 3), round(avg_v, 3), round(avg_i, 3))


def get_accessibility_for_genres(genres: list) -> float:
    matched = []
    for g in genres:
        if g in GENRE_ACCESSIBILITY:
            matched.append(GENRE_ACCESSIBILITY[g])
    return round(sum(matched) / len(matched), 3) if matched else DEFAULT_ACCESSIBILITY


def main():
    lib_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'library.json')
    
    if not os.path.exists(lib_path):
        print(f"ERROR: library.json not found at {lib_path}")
        sys.exit(1)
    
    print(f"Loading {lib_path}...")
    with open(lib_path, 'r', encoding='utf-8') as f:
        tracks = json.load(f)
    
    print(f"Enriching {len(tracks)} tracks with synthetic sonic data...")
    
    enriched = 0
    genre_miss = 0
    
    for track in tracks:
        primary = track.get('rym_data', {}).get('primary_genres', [])
        subgenres = track.get('rym_data', {}).get('subgenres', [])
        all_genres = primary + subgenres
        
        energy, valence, intensity = get_sonic_for_genres(all_genres)
        accessibility = get_accessibility_for_genres(all_genres)
        
        # Only update if still at default (0.5) — don't overwrite real data
        if track.get('energy', 0.5) == 0.5:
            track['energy'] = energy
        if track.get('valence', 0.5) == 0.5:
            track['valence'] = valence
        if track.get('intensity', 0.5) == 0.5:
            track['intensity'] = intensity
        if track.get('accessibility', 0.5) == 0.5:
            track['accessibility'] = accessibility
        
        if all_genres:
            enriched += 1
        else:
            genre_miss += 1
    
    print(f"  Enriched: {enriched} tracks")
    print(f"  No genres (kept default): {genre_miss} tracks")
    
    print(f"Writing back to {lib_path}...")
    with open(lib_path, 'w', encoding='utf-8') as f:
        json.dump(tracks, f, indent=2, ensure_ascii=False)
    
    print("Done!")
    
    # Quick sanity check
    energies = [t['energy'] for t in tracks]
    print(f"\nEnergy distribution after enrichment:")
    print(f"  Min: {min(energies):.3f}, Max: {max(energies):.3f}, Avg: {sum(energies)/len(energies):.3f}")
    
    # Show some examples
    print("\nSample enriched tracks:")
    for t in tracks[:5]:
        print(f"  {t['artist']} - {t['title']}: genres={t.get('rym_data',{}).get('primary_genres',[])} -> energy={t['energy']}, valence={t['valence']}, intensity={t['intensity']}")


if __name__ == '__main__':
    main()
