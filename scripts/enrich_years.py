"""
Enrich library.json with release_year extracted from audio file metadata tags.
Uses mutagen to read 'date' or 'year' tags from the actual files.
"""
import json
import sys
import re
from pathlib import Path

try:
    import mutagen
    from mutagen.easyid3 import EasyID3
    from mutagen.mp4 import MP4
except ImportError:
    print("ERROR: mutagen not installed. Run: pip install mutagen")
    sys.exit(1)

LIBRARY_PATH = Path(__file__).parent.parent / "data" / "library.json"


def extract_year(file_path: str) -> int | None:
    """Extract release year from audio file metadata."""
    try:
        path = Path(file_path)
        if not path.exists():
            return None

        audio = mutagen.File(file_path, easy=True)
        if audio is None:
            audio = mutagen.File(file_path)

        if audio is None:
            return None

        # Try common date/year tags
        year_str = None

        # EasyID3 / EasyMP4 style
        for tag in ['date', 'year', 'originaldate', 'TDRC', 'TYER']:
            val = audio.get(tag)
            if val:
                if isinstance(val, list):
                    year_str = str(val[0])
                else:
                    year_str = str(val)
                break

        # For MP4 files, try the raw tag
        if not year_str and hasattr(audio, 'tags') and audio.tags:
            # MP4 uses '\xa9day' for date
            for tag in ['\xa9day', 'date', 'year']:
                val = audio.tags.get(tag)
                if val:
                    if isinstance(val, list):
                        year_str = str(val[0])
                    else:
                        year_str = str(val)
                    break

        if not year_str:
            return None

        # Parse year from various formats: "2005", "2005-03-01", "2005/03/01", etc.
        match = re.match(r'(\d{4})', year_str.strip())
        if match:
            year = int(match.group(1))
            if 1900 <= year <= 2030:
                return year

        return None

    except Exception:
        return None


def main():
    with open(LIBRARY_PATH, "r", encoding="utf-8") as f:
        tracks = json.load(f)

    print(f"Loaded {len(tracks)} tracks from library.")

    updated = 0
    missing_file = 0
    no_year_in_tags = 0
    already_has_year = 0

    for i, track in enumerate(tracks):
        if i % 500 == 0 and i > 0:
            print(f"  Processed {i}/{len(tracks)}... ({updated} years found so far)")

        # Skip if already has a year
        if track.get("release_year"):
            already_has_year += 1
            continue

        file_path = track.get("file_path")
        if not file_path:
            missing_file += 1
            continue

        year = extract_year(file_path)
        if year:
            track["release_year"] = year
            updated += 1
        else:
            no_year_in_tags += 1

    print(f"\n=== Results ===")
    print(f"  Already had year:    {already_has_year}")
    print(f"  Newly extracted:     {updated}")
    print(f"  No year in tags:     {no_year_in_tags}")
    print(f"  No file path:        {missing_file}")
    print(f"  Total coverage:      {already_has_year + updated}/{len(tracks)} ({(already_has_year + updated) / len(tracks) * 100:.1f}%)")

    if updated > 0:
        # Backup
        backup = LIBRARY_PATH.with_suffix(".json.bak")
        import shutil
        shutil.copy(LIBRARY_PATH, backup)
        print(f"\n  Backup saved to {backup}")

        with open(LIBRARY_PATH, "w", encoding="utf-8") as f:
            json.dump(tracks, f, indent=2)
        print(f"  Library updated with {updated} new release years.")
    else:
        print("\n  No updates needed.")


if __name__ == "__main__":
    main()
