"""
sync_itunes.py — Authoritative library sync from iTunes Library.xml

Parses Library.xml, adds missing tracks, enriches existing ones with
iTunes metadata (genre, year, disc/track numbers). Never removes tracks.
"""

import os
import sys
import json
import plistlib
from pathlib import Path
from urllib.parse import unquote, urlparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

LIBRARY_XML = r"C:\Users\ericl\Music\Library.xml"
LIBRARY_JSON = Path(__file__).parent.parent / "data" / "library.json"
SKIP_EXTENSIONS = {".m4b", ".aa", ".aax"}  # audiobooks


def parse_itunes_xml(xml_path: str) -> dict:
    """Parse iTunes Library.xml and return dict of path -> metadata."""
    with open(xml_path, "rb") as f:
        plist = plistlib.load(f)

    tracks = plist.get("Tracks", {})
    result = {}

    for _tid, t in tracks.items():
        loc = t.get("Location")
        if not loc:
            continue

        # Decode file:// URL to local path
        path = unquote(urlparse(loc).path)
        if path.startswith("/"):
            path = path[1:]
        path = path.replace("/", os.sep)

        ext = Path(path).suffix.lower()
        if ext in SKIP_EXTENSIONS:
            continue

        # Extract metadata
        genres = []
        genre = t.get("Genre", "")
        if genre:
            genres = [genre]

        result[path] = {
            "name": t.get("Name", Path(path).stem),
            "artist": t.get("Artist", "Unknown Artist"),
            "album": t.get("Album", ""),
            "genre": genres,
            "year": t.get("Year"),
            "disc_number": t.get("Disc Number"),
            "disc_count": t.get("Disc Count"),
            "track_number": t.get("Track Number"),
            "track_count": t.get("Track Count"),
            "duration_ms": t.get("Total Time"),  # milliseconds
            "play_count": t.get("Play Count", 0),
            "rating": t.get("Rating"),  # 0-100 scale in iTunes
            "loved": bool(t.get("Loved", False)),
            "album_loved": bool(t.get("Album Loved", False)),
        }

    return result


def load_library_json(json_path: Path) -> list:
    """Load existing library.json."""
    if not json_path.exists():
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sync(dry_run: bool = False):
    print(f"Parsing iTunes XML: {LIBRARY_XML}")
    itunes = parse_itunes_xml(LIBRARY_XML)
    print(f"  iTunes tracks (music only): {len(itunes)}")

    print(f"Loading library: {LIBRARY_JSON}")
    library = load_library_json(LIBRARY_JSON)
    print(f"  Existing tracks: {len(library)}")

    # Index existing library by file_path
    path_index = {}
    for i, track in enumerate(library):
        fp = track.get("file_path", "")
        if fp:
            path_index[fp] = i

    # Also index by (artist_lower, title_lower) for dedup
    key_index = {}
    for i, track in enumerate(library):
        key = (track.get("artist", "").lower(), track.get("title", "").lower())
        key_index[key] = i

    added = 0
    enriched = 0
    skipped_no_file = 0
    skipped_audiobook = 0
    next_id = len(library) + 1

    for file_path, meta in itunes.items():
        # Check if already in library by path
        if file_path in path_index:
            # Enrich existing track with iTunes metadata
            idx = path_index[file_path]
            track = library[idx]
            changed = False

            # Add genre if we only have empty genres
            existing_genres = track.get("rym_data", {}).get("primary_genres", [])
            if not existing_genres and meta["genre"]:
                track.setdefault("rym_data", {})["primary_genres"] = meta["genre"]
                changed = True

            # Add year if missing
            if not track.get("release_year") and meta["year"]:
                track["release_year"] = meta["year"]
                changed = True

            # Add disc/track numbers if missing
            if not track.get("track_number") and meta["track_number"]:
                track["track_number"] = meta["track_number"]
                changed = True
            if not track.get("total_tracks") and meta["track_count"]:
                track["total_tracks"] = meta["track_count"]
                changed = True
            if not track.get("disc_number") and meta["disc_number"]:
                track["disc_number"] = meta["disc_number"]
                changed = True

            # Boost familiarity/rating from play count
            if meta["play_count"] and meta["play_count"] > 0:
                # Scale play count to 0-1 (cap at 50 plays = 1.0)
                fam = min(1.0, meta["play_count"] / 50.0)
                if fam > track.get("familiarity", 0):
                    track["familiarity"] = round(fam, 3)
                    changed = True

            # iTunes rating (0-100) to 0-1
            if meta["rating"] and meta["rating"] > 0:
                rating = meta["rating"] / 100.0
                if rating > track.get("rating", 0):
                    track["rating"] = round(rating, 2)
                    changed = True

            # iTunes Loved — always sync (user may have recently hearted a track/album)
            new_liked = meta["loved"]
            if track.get("liked") != new_liked:
                track["liked"] = new_liked
                changed = True

            new_album_loved = meta["album_loved"]
            if track.get("album_loved") != new_album_loved:
                track["album_loved"] = new_album_loved
                changed = True

            if changed:
                enriched += 1
            continue

        # Check if same (artist, title) already exists (different path)
        dedup_key = (meta["artist"].lower(), meta["name"].lower())
        if dedup_key in key_index:
            continue

        # New track — check if file exists on disk
        if not os.path.exists(file_path):
            skipped_no_file += 1
            continue

        # Get duration from iTunes metadata or fallback
        duration_s = 0
        if meta["duration_ms"]:
            duration_s = int(meta["duration_ms"] / 1000)

        # If no duration from iTunes, try mutagen
        if duration_s == 0:
            try:
                import mutagen
                audio = mutagen.File(file_path)
                if audio and hasattr(audio.info, "length"):
                    duration_s = int(audio.info.length)
            except Exception:
                pass

        track_id = f"t{next_id}"
        next_id += 1

        new_track = {
            "id": track_id,
            "title": meta["name"],
            "artist": meta["artist"],
            "duration_s": duration_s,
            "rym_data": {
                "primary_genres": meta["genre"],
                "subgenres": [],
                "descriptors": [],
            },
            "energy": 0.5,
            "valence": 0.5,
            "intensity": 0.5,
            "accessibility": 0.5,
            "familiarity": round(min(1.0, meta["play_count"] / 50.0), 3) if meta["play_count"] else 0.5,
            "rating": round(meta["rating"] / 100.0, 2) if meta.get("rating") else 0.0,
            "liked": meta["loved"],
            "album_loved": meta["album_loved"],
            "recommendability": 0.5,
            "file_path": file_path,
            "spotify_uri": None,
            "release_year": meta["year"],
            "track_number": meta["track_number"],
            "total_tracks": meta["track_count"],
            "disc_number": meta["disc_number"],
        }

        library.append(new_track)
        key_index[dedup_key] = len(library) - 1
        path_index[file_path] = len(library) - 1
        added += 1

    print(f"\n--- Sync Results ---")
    print(f"  Tracks added:    {added}")
    print(f"  Tracks enriched: {enriched}")
    print(f"  Skipped (file missing): {skipped_no_file}")
    print(f"  Final library size: {len(library)}")

    if dry_run:
        print("\n(DRY RUN — no changes written)")
        return

    # Backup and save
    if LIBRARY_JSON.exists():
        import shutil
        shutil.copy(LIBRARY_JSON, LIBRARY_JSON.with_suffix(".json.bak"))

    with open(LIBRARY_JSON, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2)

    print(f"  Saved to {LIBRARY_JSON}")

    # Show new artists added
    if added > 0:
        new_artists = set()
        for track in library[-added:]:
            new_artists.add(track["artist"])
        print(f"\n  New artists: {sorted(new_artists)}")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    sync(dry_run=dry)
