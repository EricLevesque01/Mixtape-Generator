"""
enrich_mix_signals.py

Enriches library.json with signals derived from iTunes playlist co-membership:
  - mix_appearances: how many playlists a track appears in (raw count)
  - mix_prominence: normalized 0-1 score (capped at 10 appearances)
  - artist_mix_appearances: total appearances across all tracks by this artist

These signals favor tracks and artists that the user has repeatedly curated.

Usage:
    python scripts/enrich_mix_signals.py
"""
import json
import plistlib
import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).parent.parent
LIBRARY_XML = r"C:\Users\ericl\Music\Library.xml"
LIBRARY_JSON = PROJECT_ROOT / "data" / "library.json"

SKIP_NAMES = {
    "Library", "Music", "Downloaded", "Movies", "TV Shows",
    "Podcasts", "Audiobooks", "Genius", "Purchased", "kids",
}

# Playlists considered "heavyweight" (large, deliberate, canonical)
# These count double toward mix prominence
HEAVYWEIGHT_PLAYLISTS = {
    "no skips", "core", "modern", "BT", "((d[-_-]b))", "lead tracks",
}

def loc_to_path(loc: str) -> str:
    path = unquote(urlparse(loc).path)
    if path.startswith("/"):
        path = path[1:]
    return path.replace("/", "\\")


def main():
    print("Loading iTunes XML...")
    plist = plistlib.load(open(LIBRARY_XML, "rb"))
    itunes_tracks = {str(tid): t for tid, t in plist.get("Tracks", {}).items()}
    playlists = plist.get("Playlists", [])

    # Collect music playlists
    music_playlists = []
    for p in playlists:
        name = p.get("Name", "")
        if name in SKIP_NAMES or p.get("Folder") or p.get("Smart Info"):
            continue
        if len(p.get("Playlist Items", [])) < 3:
            continue
        music_playlists.append(p)

    print(f"  {len(music_playlists)} playlists found")

    # Build path -> weighted appearance count
    path_appearances: dict[str, float] = defaultdict(float)
    path_playlist_names: dict[str, list] = defaultdict(list)

    for p in music_playlists:
        name = p.get("Name", "")
        weight = 2.0 if name in HEAVYWEIGHT_PLAYLISTS else 1.0
        for item in p.get("Playlist Items", []):
            tid = str(item.get("Track ID"))
            it = itunes_tracks.get(tid)
            if not it or not it.get("Location"):
                continue
            path = loc_to_path(it["Location"])
            path_appearances[path] += weight
            path_playlist_names[path].append(name)

    print(f"  {len(path_appearances)} tracks appear in at least one playlist")

    # Load library
    print("Loading library.json...")
    with open(LIBRARY_JSON, encoding="utf-8") as f:
        library = json.load(f)

    # Cap for normalization: 10 weighted appearances = 1.0
    # (a track in 5 heavyweight playlists = 10 weighted = 1.0)
    PROMINENCE_CAP = 10.0

    # Build artist -> total weighted appearances
    artist_appearances: dict[str, float] = defaultdict(float)

    enriched = 0
    for track in library:
        path = track.get("file_path", "")
        raw_count = path_appearances.get(path, 0.0)
        playlist_names = path_playlist_names.get(path, [])
        n_raw_playlists = len(playlist_names)

        # Compute normalized prominence
        prominence = round(min(1.0, raw_count / PROMINENCE_CAP), 4)

        prev_appearances = track.get("mix_appearances", 0)
        prev_prominence  = track.get("mix_prominence", 0.0)

        if raw_count != prev_appearances or prominence != prev_prominence:
            track["mix_appearances"] = n_raw_playlists  # raw playlist count (unweighted)
            track["mix_prominence"]  = prominence         # weighted + normalized
            enriched += 1

        # Accumulate for artist-level signal
        artist = track.get("artist", "").split(",")[0].split(" feat")[0].strip()
        artist_appearances[artist] += raw_count

    # Normalize artist appearances (cap at 100 weighted = 1.0)
    ARTIST_CAP = 100.0
    enriched_artist = 0
    for track in library:
        artist = track.get("artist", "").split(",")[0].split(" feat")[0].strip()
        val = round(min(1.0, artist_appearances.get(artist, 0.0) / ARTIST_CAP), 4)
        if track.get("artist_mix_prominence", 0.0) != val:
            track["artist_mix_prominence"] = val
            enriched_artist += 1

    print(f"  Tracks updated with mix_appearances:       {enriched}")
    print(f"  Tracks updated with artist_mix_prominence: {enriched_artist}")

    # Show top tracks
    top_tracks = sorted(library, key=lambda t: t.get("mix_prominence", 0), reverse=True)[:15]
    print(f"\nTop 15 most-prominent tracks:")
    for t in top_tracks:
        liked = "[L]" if t.get("liked") or t.get("album_loved") else "   "
        print(f"  {t['mix_appearances']:2}x {liked}  prom={t['mix_prominence']:.3f}  "
              f"{t['title'][:35]:<36} {t['artist']}")

    # Top artists
    top_artists = sorted(artist_appearances.items(), key=lambda x: -x[1])[:10]
    print(f"\nTop 10 most-prominent artists (by weighted appearances):")
    for artist, score in top_artists:
        norm = round(min(1.0, score / ARTIST_CAP), 3)
        print(f"  {score:6.1f} weighted  artist_prom={norm:.3f}  {artist}")

    # Save
    import shutil
    shutil.copy(LIBRARY_JSON, LIBRARY_JSON.with_suffix(".json.bak"))
    with open(LIBRARY_JSON, "w", encoding="utf-8") as f:
        json.dump(library, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {LIBRARY_JSON}")


if __name__ == "__main__":
    main()
