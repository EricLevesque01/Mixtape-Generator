"""
Enrich library.json with Spotify URIs via the Spotify Search API.

Usage:
    python scripts/enrich_spotify_uris.py

Requires SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env file.
Processes tracks incrementally, skipping those already resolved.
"""
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

DATA_PATH = project_root / "data" / "library.json"
BATCH_SIZE = 50
SAVE_EVERY = 5


def load_library():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_library(data):
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def search_spotify(sp, artist, title):
    """Search Spotify for a track and return its URI."""
    query = f"track:{title} artist:{artist}"
    try:
        results = sp.search(q=query, type="track", limit=1)
        items = results.get("tracks", {}).get("items", [])
        if items:
            return items[0]["uri"]
    except Exception as e:
        # Rate limit or other error
        if "429" in str(e):
            time.sleep(5)
        pass
    return None


def main():
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("Error: SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in .env")
        sys.exit(1)
    
    auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth)
    
    data = load_library()
    
    # Find tracks without Spotify URIs
    to_resolve = [(i, t) for i, t in enumerate(data) if not t.get("spotify_uri")]
    total = len(to_resolve)
    print(f"Found {total} tracks without Spotify URIs out of {len(data)} total.")
    
    if total == 0:
        print("All tracks already have Spotify URIs!")
        return
    
    resolved = 0
    not_found = 0
    batch_count = 0
    
    for batch_start in range(0, total, BATCH_SIZE):
        batch = to_resolve[batch_start:batch_start + BATCH_SIZE]
        
        print(f"  Batch {batch_count + 1}/{(total + BATCH_SIZE - 1) // BATCH_SIZE}: "
              f"tracks {batch_start + 1}-{min(batch_start + BATCH_SIZE, total)}...")
        
        for idx, track in batch:
            uri = search_spotify(sp, track["artist"], track["title"])
            if uri:
                data[idx]["spotify_uri"] = uri
                resolved += 1
            else:
                not_found += 1
            
            time.sleep(0.1)  # Rate limiting
        
        batch_count += 1
        
        if batch_count % SAVE_EVERY == 0:
            save_library(data)
            print(f"  [SAVE] Progress saved. {resolved} resolved, {not_found} not found.")
    
    save_library(data)
    print(f"\nDone! Resolved {resolved}/{total} tracks. {not_found} not found on Spotify.")


if __name__ == "__main__":
    main()
