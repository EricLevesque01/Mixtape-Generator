"""
enrich_spotify_listening.py

Pulls your Spotify listening data and writes spotify_affinity (0-1)
to each matched track in library.json.

Data pulled (with point weights):
  - Top tracks short-term  (4 weeks)  × 3
  - Top tracks medium-term (6 months) × 2
  - Top tracks long-term   (all-time) × 1
  - Recently played                   × 1
  - Saved tracks (Spotify likes)      × 2

Run once to authorize (browser opens), token is cached after that.

Usage:
    python scripts/enrich_spotify_listening.py
"""
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
except ImportError:
    print("ERROR: spotipy not installed. Run: pip install spotipy")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
LIBRARY_JSON = PROJECT_ROOT / "data" / "library.json"

SCOPES = (
    "user-top-read "
    "user-read-recently-played "
    "user-library-read "
    "playlist-read-private"
)

# Point weights per data source
SOURCE_WEIGHTS = {
    "playlists":  3,   # your curated playlists — strongest signal
    "top_short":  2,   # what you're streaming right now
    "top_medium": 1,   # 6-month listening
    "top_long":   1,   # all-time listening
    "recent":     1,   # last 50 plays
    "saved":      1,   # Spotify likes
    "top_artists": 2,  # artist-level affinity from listening
}

# Max points possible (top-50 × 3 + top-50 × 2 + top-50 × 1 + 50 × 1 + 50 × 2) = 450
# We cap normalization at 10 total points → 1.0 affinity
AFFINITY_CAP = 10.0


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def get_spotify() -> spotipy.Spotify:
    client_id     = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri  = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")

    if not client_id or not client_secret:
        print("ERROR: SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET not set in .env")
        sys.exit(1)

    auth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=SCOPES,
        cache_path=str(PROJECT_ROOT / ".spotify_token_cache"),
        open_browser=True,
    )
    sp = spotipy.Spotify(auth_manager=auth)
    user = sp.current_user()
    print(f"  Authenticated as: {user['display_name']} ({user['id']})")
    return sp


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def fetch_top_tracks(sp: spotipy.Spotify, time_range: str) -> list[dict]:
    """Fetch up to 50 top tracks for a given time range."""
    results = sp.current_user_top_tracks(limit=50, time_range=time_range)
    return results.get("items", [])


def fetch_top_artists(sp: spotipy.Spotify) -> list[str]:
    """Fetch top artist names across all time ranges."""
    artists = set()
    for tr in ["short_term", "medium_term", "long_term"]:
        results = sp.current_user_top_artists(limit=50, time_range=tr)
        for a in results.get("items", []):
            artists.add(a["name"].lower())
    return list(artists)


def fetch_recently_played(sp: spotipy.Spotify) -> list[dict]:
    results = sp.current_user_recently_played(limit=50)
    return [item["track"] for item in results.get("items", [])]


def fetch_saved_tracks(sp: spotipy.Spotify, max_tracks: int = 500) -> list[dict]:
    """Fetch saved (liked) tracks with pagination — up to max_tracks."""
    all_tracks = []
    offset = 0
    while offset < max_tracks:
        results = sp.current_user_saved_tracks(limit=50, offset=offset)
        items = results.get("items", [])
        if not items:
            break
        all_tracks.extend(item["track"] for item in items if item.get("track"))
        offset += 50
    return all_tracks


def fetch_playlist_tracks(sp: spotipy.Spotify) -> list[dict]:
    """Fetch all tracks from user's own Spotify playlists."""
    user_id = sp.current_user()["id"]
    all_tracks = []
    playlists = sp.current_user_playlists(limit=50)
    for pl in playlists.get("items", []):
        # Only include playlists owned by the user
        if pl.get("owner", {}).get("id") != user_id:
            continue
        try:
            results = sp.playlist_tracks(pl["id"], limit=100)
            for item in results.get("items", []):
                track = item.get("track")
                if track and track.get("id"):  # skip local files
                    all_tracks.append(track)
        except Exception:
            continue
    return all_tracks


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Lowercase, strip punctuation for fuzzy matching."""
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def build_library_index(library: list[dict]) -> tuple[dict, dict, dict]:
    """Return three lookup dicts: by_uri, by_title_artist, by_title."""
    by_uri: dict[str, int] = {}
    by_title_artist: dict[tuple, int] = {}
    by_title: dict[str, list[int]] = defaultdict(list)

    for i, t in enumerate(library):
        uri = t.get("spotify_uri")
        if uri:
            by_uri[uri] = i
        key = (normalize(t.get("title", "")), normalize(t.get("artist", "").split(",")[0].split(" feat")[0]))
        by_title_artist[key] = i
        by_title[normalize(t.get("title", ""))].append(i)

    return by_uri, by_title_artist, by_title


def match_track(
    sp_track: dict,
    by_uri: dict,
    by_title_artist: dict,
    by_title: dict,
) -> int | None:
    """Return library index for a Spotify track, or None."""
    uri = sp_track.get("uri")
    if uri and uri in by_uri:
        return by_uri[uri]

    title  = normalize(sp_track.get("name", ""))
    artist = normalize(sp_track.get("artists", [{}])[0].get("name", ""))

    # Exact title+artist
    if (title, artist) in by_title_artist:
        return by_title_artist[(title, artist)]

    # Title-only if unique
    candidates = by_title.get(title, [])
    if len(candidates) == 1:
        return candidates[0]

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Spotify Listening Enrichment ===\n")

    print("Step 1: Authenticating with Spotify...")
    sp = get_spotify()

    print("\nStep 2: Fetching listening data...")
    sources: dict[str, list[dict]] = {}

    sources["top_short"]  = fetch_top_tracks(sp, "short_term")
    print(f"  Top tracks (4 weeks):   {len(sources['top_short'])}")

    sources["top_medium"] = fetch_top_tracks(sp, "medium_term")
    print(f"  Top tracks (6 months):  {len(sources['top_medium'])}")

    sources["top_long"]   = fetch_top_tracks(sp, "long_term")
    print(f"  Top tracks (all-time):  {len(sources['top_long'])}")

    sources["recent"]     = fetch_recently_played(sp)
    print(f"  Recently played:        {len(sources['recent'])}")

    sources["saved"]      = fetch_saved_tracks(sp)
    print(f"  Saved tracks:           {len(sources['saved'])}")

    sources["playlists"]  = fetch_playlist_tracks(sp)
    print(f"  Playlist tracks:        {len(sources['playlists'])}")

    top_artists = fetch_top_artists(sp)
    print(f"  Top artists:            {len(top_artists)}")

    print("\nStep 3: Loading library...")
    with open(LIBRARY_JSON, encoding="utf-8") as f:
        library = json.load(f)
    print(f"  {len(library)} tracks")

    by_uri, by_title_artist, by_title = build_library_index(library)

    print("\nStep 4: Matching and scoring...")
    # Accumulate weighted points per library index
    points: dict[int, float] = defaultdict(float)
    matched_uris: dict[int, str] = {}  # store spotify URIs we discover

    for source_name, tracks in sources.items():
        weight = SOURCE_WEIGHTS[source_name]
        matched = 0
        for sp_track in tracks:
            idx = match_track(sp_track, by_uri, by_title_artist, by_title)
            if idx is not None:
                points[idx] += weight
                matched += 1
                # Store URI for future fast matching
                uri = sp_track.get("uri")
                if uri and not library[idx].get("spotify_uri"):
                    matched_uris[idx] = uri
        print(f"  {source_name:15} matched {matched}/{len(tracks)}")

    # Artist-level matching: every track by a top artist gets points
    artist_weight = SOURCE_WEIGHTS["top_artists"]
    artist_matched = 0
    for i, t in enumerate(library):
        artist_name = normalize(t.get("artist", "").split(",")[0].split(" feat")[0])
        if artist_name in [normalize(a) for a in top_artists]:
            points[i] += artist_weight
            artist_matched += 1
    print(f"  {'top_artists':15} matched {artist_matched}/{len(library)} library tracks")

    print("\nStep 5: Writing spotify_affinity to library...")
    enriched = 0
    for idx, pts in points.items():
        affinity = round(min(1.0, pts / AFFINITY_CAP), 4)
        library[idx]["spotify_affinity"] = affinity
        if idx in matched_uris:
            library[idx]["spotify_uri"] = matched_uris[idx]
        enriched += 1

    # Zero out tracks not in Spotify data (preserve existing if re-running)
    for i, t in enumerate(library):
        if i not in points and "spotify_affinity" not in t:
            t["spotify_affinity"] = 0.0

    print(f"  {enriched} tracks enriched with spotify_affinity")

    # Show top matched tracks
    top = sorted(points.items(), key=lambda x: -x[1])[:15]
    print(f"\nTop 15 by Spotify affinity:")
    for idx, pts in top:
        t = library[idx]
        aff = round(min(1.0, pts / AFFINITY_CAP), 3)
        liked = "[L]" if t.get("liked") or t.get("album_loved") else "   "
        print(f"  pts={pts:5.1f}  aff={aff:.3f}  {liked}  {t['title'][:35]:<36} {t['artist']}")

    # Save
    shutil.copy(LIBRARY_JSON, LIBRARY_JSON.with_suffix(".json.bak"))
    with open(LIBRARY_JSON, "w", encoding="utf-8") as f:
        json.dump(library, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {LIBRARY_JSON}")
    print(f"\nDone! Run: python scripts/build_similarity_graph.py to rebuild the graph.")


if __name__ == "__main__":
    main()
