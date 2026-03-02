import os
import sys
from pathlib import Path
import json
import traceback

print("DEBUG: Starting scan_library.py", flush=True)

try:
    import mutagen
    print("DEBUG: mutagen imported", flush=True)
except ImportError:
    print("ERROR: mutagen not installed", flush=True)
    sys.exit(1)

# Add src to path to import config
sys.path.append(str(Path(__file__).parent.parent / "src"))
try:
    from mixtape_curator.config import config
    print("DEBUG: config imported", flush=True)
except ImportError:
    print("ERROR: Could not import config", flush=True)
    traceback.print_exc()
    sys.exit(1)

def scan_library(limit=None):
    # Hardcoded list of paths to scan based on user request
    # NOTE: In a real app we might support a list in config, but for now we iterate here or update config
    # The config only supports one string for now.
    # To support multiple without changing config schema too much, I'll just iterate here manually
    # or better, let's just create a list of paths to scan.
    
    # User asked to scan "C:\Users\ericl\Music" but SKIP "C:\Users\ericl\Music\tPod V3" 
    # because it was already scanned (to avoid duplicates? or just efficiency?)
    # actually, "we already added the songs from there" implies we should KEEP existing library and ADD to it.
    
    paths_to_scan = [
        "C:\\Users\\ericl\\Music",
        "C:\\Users\\ericl\\Downloads\\zips",
        "C:\\Users\\ericl\\Music\\tPod V3\\iTunes Media\\Music\\Geese\\Getting Killed"
    ]
    
    exclude_paths = []
    
    print(f"Scanning paths: {paths_to_scan}\n", flush=True)
    
    # Load existing library if it exists to append?
    # User said "we already added the songs". So we should load existing.
    output_path = Path(__file__).parent.parent / "data" / "library.json"
    tracks = []
    existing_paths = set()
    
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                tracks = json.load(f)
            print(f"Loaded {len(tracks)} existing tracks.", flush=True)
            for t in tracks:
                if t.get("file_path"):
                     existing_paths.add(t["file_path"])
        except Exception as e:
            print(f"Error loading existing library: {e}", flush=True)
            tracks = []

    supported_exts = {'.mp3', '.m4a', '.flac', '.ogg', '.wav'}
    
    # Count needs to continue from existing
    count = len(tracks)
    scanned_count = 0
    
    for music_path in paths_to_scan:
        music_dir = Path(music_path)
        if not music_dir.exists():
             print(f"Warning: Path not found {music_dir}", flush=True)
             continue
             
        try:
            for root, dirs, files in os.walk(music_dir):
                # Check exclusions
                # If root starts with excluded path, skip files
                # Normalize paths for comparison
                root_path = Path(root)
                skip = False
                for exc in exclude_paths:
                    if str(root_path).startswith(str(Path(exc))):
                        skip = True
                        break
                if skip:
                    continue
                
                for file in files:
                    scanned_count += 1
                    if scanned_count % 100 == 0:
                        print(f"Parsed {scanned_count} files...", flush=True)
                        
                    if limit and count >= limit:
                        break
                        
                    file_path = Path(root) / file
                    
                    # Skip if already in library
                    if str(file_path) in existing_paths:
                        continue
                        
                    if file_path.suffix.lower() in supported_exts:
                        try:
                            audio = mutagen.File(file_path, easy=True)
                            if audio is None:
                                audio = mutagen.File(file_path)
                            
                            if not audio:
                                continue
                            
                            title = "Unknown Title"
                            artist = "Unknown Artist"
                            primary_genres = []
                            
                            try:
                                # Safely get tags
                                t = audio.get('title', [file_path.stem])
                                title = t[0] if isinstance(t, list) else str(t)
                                
                                a = audio.get('artist', ['Unknown Artist'])
                                artist = a[0] if isinstance(a, list) else str(a)
                                
                                g = audio.get('genre', [])
                                if isinstance(g, list):
                                    primary_genres = [str(x) for x in g]
                                elif g:
                                    primary_genres = [str(g)]
                                    
                            except Exception as e:
                                title = file_path.stem
    
                            duration_s = 0
                            if hasattr(audio.info, 'length'):
                                duration_s = int(audio.info.length)
                            
                            count += 1
                            track_id = f"t{count}"
                            
                            track = {
                                "id": track_id,
                                "title": title,
                                "artist": artist,
                                "duration_s": duration_s,
                                "rym_data": {
                                    "primary_genres": primary_genres,
                                    "subgenres": [],
                                    "descriptors": []
                                },
                                "energy": 0.5,
                                "valence": 0.5,
                                "intensity": 0.5,
                                "accessibility": 0.5,
                                "familiarity": 0.5,
                                "rating": 0.0,
                                "recommendability": 0.5,
                                "file_path": str(file_path),
                                "spotify_uri": None
                            }
                            tracks.append(track)
                            existing_paths.add(str(file_path))
                            
                        except Exception as e:
                            pass
                
                if limit and count >= limit:
                    break
        except KeyboardInterrupt:
            print("Interrupted scan.", flush=True)

    print(f"Total tracks now: {len(tracks)}.", flush=True)
    
    # Check if we found broad genres?
    genres_found = set()
    for t in tracks:
        for g in t["rym_data"]["primary_genres"]:
            genres_found.add(g)
    
    print(f"Found {len(genres_found)} unique genres...", flush=True)
    
    if output_path.exists():
        backup_path = output_path.with_suffix(".json.bak")
        try:
            import shutil
            shutil.copy(output_path, backup_path)
        except:
            pass

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tracks, f, indent=2)
    
    print(f"Library saved to {output_path}", flush=True)

if __name__ == "__main__":
    scan_library()
