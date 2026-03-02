"""
recommend.py — Seed-based recommendations from the similarity graph.

Usage:
    python scripts/recommend.py "Song Title" "Another Song" --n 20
    python scripts/recommend.py "Fix You" "Lucky" "Paranoid Android" --n 15 --liked-only

Options:
    --n N           Number of recommendations to return (default: 10)
    --liked-only    Only return liked/album-loved tracks
    --show-scores   Show individual signal scores (sonic, genre, mix)
"""
import json
import sys
import argparse
import io
from pathlib import Path

# Force UTF-8 output on Windows to handle extended characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent.parent
GRAPH_PATH = PROJECT_ROOT / "data" / "similarity_graph.json"


def load_graph():
    if not GRAPH_PATH.exists():
        print(f"ERROR: {GRAPH_PATH} not found.")
        print("Run: python scripts/build_similarity_graph.py")
        sys.exit(1)
    with open(GRAPH_PATH, encoding="utf-8") as f:
        return json.load(f)


def find_seeds(graph_tracks: dict, queries: list[str]) -> list[str]:
    """Find track IDs matching the query strings (case-insensitive title or artist)."""
    seeds = []
    for query in queries:
        q = query.lower().strip()
        matches = [
            tid for tid, t in graph_tracks.items()
            if q in t["title"].lower() or q in t["artist"].lower()
        ]
        if not matches:
            print(f"  [WARN] No match for '{query}' — skipping")
            continue
        if len(matches) > 1:
            # Prefer exact title match
            exact = [m for m in matches if graph_tracks[m]["title"].lower() == q]
            matches = exact if exact else matches[:1]
        tid = matches[0]
        t = graph_tracks[tid]
        heart = "(liked)" if t["liked"] else ""
        print(f"  Seed: '{t['title']}' by {t['artist']} {heart}")
        seeds.append(tid)
    return seeds


def aggregate_recommendations(
    graph_tracks: dict,
    seed_ids: list[str],
    n: int,
    liked_only: bool,
) -> list[dict]:
    """
    Walk the neighbor lists of all seeds, aggregate scores.
    Tracks appearing as a neighbor of multiple seeds get a bonus.
    """
    seed_set = set(seed_ids)
    score_map: dict[str, float] = {}
    appearances: dict[str, int] = {}
    neighbor_data: dict[str, dict] = {}

    for seed_id in seed_ids:
        seed_entry = graph_tracks.get(seed_id)
        if not seed_entry:
            continue
        for neighbor in seed_entry.get("neighbors", []):
            nid = neighbor["id"]
            if nid in seed_set:
                continue  # don't recommend a seed
            if liked_only and not neighbor.get("liked"):
                continue
            score_map[nid] = score_map.get(nid, 0.0) + neighbor["score"]
            appearances[nid] = appearances.get(nid, 0) + 1
            neighbor_data[nid] = neighbor  # keep latest metadata

    # Bonus for appearing near multiple seeds
    results = []
    for nid, score in score_map.items():
        count = appearances[nid]
        # Multi-seed bonus: sqrt(count) scale
        boosted = score * (count ** 0.5)
        nd = neighbor_data[nid]
        results.append({
            "id": nid,
            "title": nd["title"],
            "artist": nd["artist"],
            "score": round(boosted, 4),
            "raw_score": round(score, 4),
            "seed_count": count,
            "sonic": nd.get("sonic", 0),
            "genre": nd.get("genre", 0),
            "mix": nd.get("mix", 0),
            "rym": nd.get("rym", 0),
            "in_mix": nd.get("in_mix", False),
            "liked": nd.get("liked", False),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:n]


def main():
    parser = argparse.ArgumentParser(description="Seed-based song recommendations")
    parser.add_argument("seeds", nargs="+", help="Seed song titles or artist names")
    parser.add_argument("--n", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--liked-only", action="store_true", help="Only return liked tracks")
    parser.add_argument("--show-scores", action="store_true", help="Show signal breakdown")
    args = parser.parse_args()

    print(f"\nLoading similarity graph...")
    data = load_graph()
    graph_tracks = data["tracks"]
    meta = data.get("_meta", {})
    print(f"  {meta.get('total_tracks', '?')} tracks, top-{meta.get('top_k', '?')} neighbors each")
    weights = meta.get('weights', {})
    print(f"  Weights: sonic={weights.get('sonic','?')}, genre={weights.get('genre','?')}, mix={weights.get('mix','?')}, rym={weights.get('rym','?')}")
    print(f"  Mix playlists used: {meta.get('mix_playlists_used', '?')}\n")

    print("Finding seed tracks:")
    seed_ids = find_seeds(graph_tracks, args.seeds)
    if not seed_ids:
        print("No seeds found. Exiting.")
        sys.exit(1)

    print(f"\nGenerating top-{args.n} recommendations{'  (liked only)' if args.liked_only else ''}...\n")
    recs = aggregate_recommendations(graph_tracks, seed_ids, args.n, args.liked_only)

    if not recs:
        print("No recommendations found.")
        sys.exit(0)

    print(f"{'#':>3}  {'Score':>6}  {'Liked':>5}  {'Mix':>3}  {'Title':<40}  Artist")
    print("-" * 80)
    for rank, r in enumerate(recs, 1):
        heart = "[L]" if r["liked"] else "   "
        mix_flag = "[M]" if r["in_mix"] else "   "
        title = r['title'][:38] + ".." if len(r['title']) > 40 else r['title']
        print(f"{rank:>3}. {r['score']:>6.4f}  {heart:>5}  {mix_flag}  {title:<40}  {r['artist']}")
        if args.show_scores:
            seeds_str = f"x{r['seed_count']}" if r['seed_count'] > 1 else "   "
            print(f"       sonic={r['sonic']:.3f}  genre={r['genre']:.3f}  mix={r['mix']:.3f}  rym={r['rym']:.3f}  seeds={seeds_str}")

    print(f"\n[L] = liked/album-loved   [M] = appeared in your mixes together")


if __name__ == "__main__":
    main()
