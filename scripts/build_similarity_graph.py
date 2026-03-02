"""
build_similarity_graph.py

Builds a K-Nearest Neighbor similarity graph from library.json.

Signals:
  - Sonic features (50%): energy, valence, intensity, tempo, danceability,
    acousticness, brightness, dynamic_range
  - Genre/descriptor Jaccard (30%): primary_genres + subgenres + descriptors
  - Mix co-occurrence (20%): tracks appearing together in your iTunes 'mixes' playlists
  - Liked boost (x1.15): neighbors of liked/album-loved tracks get a relevance boost

Output: data/similarity_graph.json
  top-25 neighbors per track, with per-signal score breakdown.

Usage:
    python scripts/build_similarity_graph.py
"""
import json
import plistlib
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from sklearn.preprocessing import normalize

# RYM genre hierarchy for hierarchical one-hot encoding
sys.path.insert(0, str(Path(__file__).parent))
from rym_genre_hierarchy import get_genre_chain

PROJECT_ROOT = Path(__file__).parent.parent
LIBRARY_XML = r"C:\Users\ericl\Music\Library.xml"
LIBRARY_JSON = PROJECT_ROOT / "data" / "library.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "similarity_graph.json"

# Weights — validated against mixtape co-occurrence pairs vs random pairs
# Best grid search: sep=+0.1945 (vs +0.1197 for prior heuristic weights)
W_SONIC = 0.15   # audio fingerprint cosine similarity
W_GENRE = 0.15   # hierarchical RYM genre cosine similarity
W_MIX   = 0.35   # playlist co-occurrence (strongest signal)
W_RYM   = 0.35   # RYM community rating

# Genre feature weights within combined vector
SOURCE_WEIGHTS = {
    "primary_genres": 1.0,   # full weight
    "subgenres":      0.7,   # slightly lower
    "descriptors":    0.4,   # mood/vibe descriptors, lower weight
}

# Liked neighbor boost multiplier
LIKED_BOOST = 1.15

# Neighbors to store per track
TOP_K = 25

# Sonic feature keys and their normalization ranges
SONIC_FEATURES = [
    "energy",       # 0-1
    "valence",      # 0-1
    "intensity",    # 0-1
    "danceability", # 0-1
    "acousticness", # 0-1
    "brightness",   # 0-1
    "dynamic_range",# 0-1
    "tempo",        # normalize 0-250 bpm -> 0-1
]


# ---------------------------------------------------------------------------
# 1. Parse iTunes XML — extract mixes folder playlists
# ---------------------------------------------------------------------------

def loc_to_path(loc: str) -> str:
    path = unquote(urlparse(loc).path)
    if path.startswith("/"):
        path = path[1:]
    return path.replace("/", "\\")


def load_mix_playlists(xml_path: str) -> dict[str, set[str]]:
    """
    Returns: track_path -> set of mix playlist names the track appears in.
    Only playlists inside the 'mixes' folder (or named like mixes).
    """
    with open(xml_path, "rb") as f:
        plist = plistlib.load(f)

    playlists = plist.get("Playlists", [])
    tracks_by_id = {str(tid): t for tid, t in plist.get("Tracks", {}).items()}

    # Find 'mixes' folder PID
    mixes_folder_pid = None
    for p in playlists:
        if p.get("Name") == "mixes" and p.get("Folder"):
            mixes_folder_pid = p.get("Playlist Persistent ID")
            break

    # iTunes doesn't reliably serialize folder parent IDs into the XML,
    # so we include all non-system music playlists as co-occurrence training data.
    skip_names = {
        "Library", "Music", "Downloaded", "Movies", "TV Shows",
        "Podcasts", "Audiobooks", "Genius", "Purchased",
        "kids",  # children's music — different taste profile
    }

    mix_playlists = []
    for p in playlists:
        name = p.get("Name", "")
        if name in skip_names:
            continue
        if p.get("Folder") or p.get("Smart Info"):
            continue
        items = p.get("Playlist Items", [])
        if len(items) < 3:  # skip tiny playlists
            continue
        mix_playlists.append(p)

    print(f"  Found {len(mix_playlists)} mix playlists:")
    for p in mix_playlists:
        print(f"    '{p['Name']}': {len(p.get('Playlist Items', []))} tracks")

    # Build track_path -> set[playlist_name]
    track_to_mixes: dict[str, set[str]] = defaultdict(set)
    for p in mix_playlists:
        playlist_name = p.get("Name", "")
        for item in p.get("Playlist Items", []):
            tid = str(item.get("Track ID"))
            track = tracks_by_id.get(tid)
            if not track:
                continue
            loc = track.get("Location")
            if not loc:
                continue
            path = loc_to_path(loc)
            track_to_mixes[path].add(playlist_name)

    return dict(track_to_mixes)


# ---------------------------------------------------------------------------
# 2. Build feature vectors
# ---------------------------------------------------------------------------

def build_genre_vocabulary(tracks: list[dict]) -> dict[str, int]:
    """
    Build vocabulary from all genres AND their ancestors per the RYM hierarchy.
    This ensures parent-genre dimensions exist in the matrix.
    """
    vocab: dict[str, int] = {}
    for t in tracks:
        rym = t.get("rym_data", {})
        all_genres = (
            list(rym.get("primary_genres", [])) +
            list(rym.get("subgenres", [])) +
            list(rym.get("descriptors", []))[:5]
        )
        for g in all_genres:
            for ancestor in get_genre_chain(g):  # includes genre + parents
                key = ancestor.lower()
                if key not in vocab:
                    vocab[key] = len(vocab)
    return vocab


def build_feature_matrices(
    tracks: list[dict],
    vocab: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (sonic_matrix, genre_matrix), each L2-normalized.
    Kept separate so they can be weighted independently.
    """
    N = len(tracks)
    V = len(vocab)

    # --- Sonic sub-matrix (N x 8) ---
    sonic_rows = []
    for t in tracks:
        vec = [
            t.get("energy", 0.5),
            t.get("valence", 0.5),
            t.get("intensity", 0.5),
            t.get("danceability", 0.5),
            t.get("acousticness", 0.0),
            t.get("brightness", 0.5),
            t.get("dynamic_range", 0.5),
            min(1.0, t.get("tempo", 120) / 250.0),
        ]
        sonic_rows.append(vec)
    sonic_mat = np.array(sonic_rows, dtype=np.float32)

    # --- Genre hierarchical sub-matrix (N x V) ---
    SOURCE_W = {"primary": 1.0, "subgenre": 0.7, "descriptor": 0.4}
    genre_mat = np.zeros((N, V), dtype=np.float32)
    for i, t in enumerate(tracks):
        rym = t.get("rym_data", {})
        genre_sources = (
            [(g, SOURCE_W["primary"])   for g in rym.get("primary_genres", [])] +
            [(g, SOURCE_W["subgenre"])  for g in rym.get("subgenres", [])] +
            [(g, SOURCE_W["descriptor"]) for g in rym.get("descriptors", [])[:5]]
        )
        for genre, src_weight in genre_sources:
            chain = get_genre_chain(genre)
            for ancestor, decay_weight in chain.items():
                j = vocab.get(ancestor.lower())
                if j is not None:
                    val = src_weight * decay_weight
                    if genre_mat[i, j] < val:
                        genre_mat[i, j] = val

    def l2norm(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms

    return l2norm(sonic_mat), l2norm(genre_mat)




# ---------------------------------------------------------------------------
# 3. Compute full similarity matrix (chunked to avoid memory explosion)
# ---------------------------------------------------------------------------

def compute_similarity_graph(
    tracks: list[dict],
    sonic_matrix: np.ndarray,
    genre_matrix: np.ndarray,
    track_to_mixes: dict[str, set[str]],
) -> dict[str, list[dict]]:
    """
    Returns: track_id -> list of top-K neighbor dicts.
    Sonic and genre cosine similarity are weighted independently.
    """
    N = len(tracks)
    print(f"\n  Computing similarities for {N} tracks...")

    # Lookup helpers
    id_to_idx = {t["id"]: i for i, t in enumerate(tracks)}
    path_to_idx = {t.get("file_path", ""): i for i, t in enumerate(tracks)}

    # Build co-occurrence lookup: (i, j) -> # shared mixes
    # First build idx -> set of mix playlist names
    idx_to_mixes: dict[int, set[str]] = {}
    for path, mixes in track_to_mixes.items():
        idx = path_to_idx.get(path)
        if idx is not None:
            idx_to_mixes[idx] = mixes

    max_mixes = max((len(v) for v in idx_to_mixes.values()), default=1)

    # Liked lookup
    liked_set = {i for i, t in enumerate(tracks) if t.get("liked") or t.get("album_loved")}

    # Process in chunks to avoid N^2 memory
    CHUNK = 500
    graph: dict[str, list] = {}

    start = time.time()
    for chunk_start in range(0, N, CHUNK):
        chunk_end = min(chunk_start + CHUNK, N)
        chunk_size = chunk_end - chunk_start

        # Cosine similarity chunks
        sonic_chunk = sonic_matrix[chunk_start:chunk_end]
        sonic_sims  = sonic_chunk @ sonic_matrix.T
        genre_chunk = genre_matrix[chunk_start:chunk_end]
        genre_sims  = genre_chunk @ genre_matrix.T

        for local_i, global_i in enumerate(range(chunk_start, chunk_end)):
            t_i = tracks[global_i]
            mixes_i = idx_to_mixes.get(global_i, set())

            row_scores = []
            sonic_row = sonic_sims[local_i]
            genre_row = genre_sims[local_i]

            for global_j in range(N):
                if global_j == global_i:
                    continue

                # --- Sonic cosine ---
                s_sonic = max(0.0, float(sonic_row[global_j]))

                # --- Genre cosine ---
                s_genre = max(0.0, float(genre_row[global_j]))

                # --- Mix co-occurrence ---
                mixes_j = idx_to_mixes.get(global_j, set())
                shared = len(mixes_i & mixes_j)
                denom = max(len(mixes_i), len(mixes_j), 1)
                s_mix = shared / denom if (mixes_i or mixes_j) else 0.0

                # --- RYM quality ---
                raw_rym_j = tracks[global_j].get("rym_rating", 3.0) or 3.0
                s_rym = max(0.0, min(1.0, (raw_rym_j - 0.5) / 4.5))

                # --- Blend ---
                score = W_SONIC * s_sonic + W_GENRE * s_genre + W_MIX * s_mix + W_RYM * s_rym

                # --- Liked boost ---
                if global_j in liked_set:
                    score *= LIKED_BOOST

                row_scores.append((global_j, score, s_sonic, s_genre, s_mix, s_rym))

            # Sort by score descending, keep top-K
            row_scores.sort(key=lambda x: x[1], reverse=True)
            top = row_scores[:TOP_K]

            t_j_list = []
            for global_j, score, s_sonic, s_genre, s_mix, s_rym in top:
                tj = tracks[global_j]
                t_j_list.append({
                    "id": tj["id"],
                    "title": tj["title"],
                    "artist": tj["artist"],
                    "score": round(score, 4),
                    "sonic": round(s_sonic, 3),
                    "genre": round(s_genre, 3),
                    "mix": round(s_mix, 3),
                    "rym": round(s_rym, 3),
                    "in_mix": s_mix > 0,
                    "liked": bool(tj.get("liked") or tj.get("album_loved")),
                })

            graph[t_i["id"]] = t_j_list

        elapsed = time.time() - start
        pct = chunk_end / N * 100
        rate = chunk_end / elapsed if elapsed > 0 else 0
        eta = (N - chunk_end) / rate if rate > 0 else 0
        print(f"  [{pct:5.1f}%] {chunk_end}/{N} tracks | {rate:.0f} t/s | ETA {eta:.0f}s")

    return graph


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    print("=== Building Song Similarity Graph ===\n")

    print("Step 1: Loading mix playlists from iTunes XML...")
    track_to_mixes = load_mix_playlists(LIBRARY_XML)
    print(f"  {len(track_to_mixes)} tracks appear in at least one mix\n")

    print("Step 2: Loading library.json...")
    with open(LIBRARY_JSON, encoding="utf-8") as f:
        library = json.load(f)
    print(f"  {len(library)} tracks\n")

    print("Step 3: Building feature matrices...")
    print("  Building genre vocabulary...")
    vocab = build_genre_vocabulary(library)
    print(f"  Genre vocabulary: {len(vocab)} unique genres/descriptors")
    sonic_matrix, genre_matrix = build_feature_matrices(library, vocab)
    liked_count = sum(1 for t in library if t.get("liked") or t.get("album_loved"))
    mix_count = sum(1 for t in library if t.get("file_path", "") in track_to_mixes)
    print(f"  Sonic matrix: {sonic_matrix.shape}  |  Genre matrix: {genre_matrix.shape}")
    print(f"  Liked/album-loved tracks: {liked_count}")
    print(f"  Tracks in mixes: {mix_count}\n")

    print("Step 4: Computing similarity graph (top-25 per track)...")
    graph = compute_similarity_graph(library, sonic_matrix, genre_matrix, track_to_mixes)

    print(f"\nStep 5: Saving to {OUTPUT_PATH}...")
    # Wrap with metadata
    output = {
        "_meta": {
            "total_tracks": len(library),
            "top_k": TOP_K,
            "weights": {"sonic": W_SONIC, "genre": W_GENRE, "mix": W_MIX, "rym": W_RYM},
            "liked_boost": LIKED_BOOST,
            "mix_playlists_used": len(set(
                p for mixes in track_to_mixes.values() for p in mixes
            )),
        },
        "tracks": {}
    }
    for t in library:
        tid = t["id"]
        output["tracks"][tid] = {
            "title": t["title"],
            "artist": t["artist"],
            "liked": bool(t.get("liked") or t.get("album_loved")),
            "neighbors": graph.get(tid, []),
        }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    size_mb = OUTPUT_PATH.stat().st_size / 1_000_000
    print(f"  Saved! {size_mb:.1f} MB")
    print(f"\nDone. Graph built: {len(library)} tracks, top-{TOP_K} neighbors each.")
    print(f"Run: python scripts/recommend.py \"Song Title\" \"Another Song\" --n 10")


if __name__ == "__main__":
    main()
