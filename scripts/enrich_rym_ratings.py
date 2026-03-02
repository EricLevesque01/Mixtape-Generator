"""
Enrich library.json with RYM-style album and track ratings via LLM recall.

RYM blocks scraping (403 + CloudFlare), so we use GPT-4o's training data
knowledge of RYM ratings. GPT-4o has seen most popular album pages.

Usage:
    python scripts/enrich_rym_ratings.py
"""
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from openai import OpenAI

DATA_PATH = project_root / "data" / "library.json"
BATCH_SIZE = 5  # albums per batch
SAVE_EVERY = 10


def load_library():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_library(data):
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def group_by_album(data):
    """Group tracks by (artist, album_folder) and return album groups."""
    albums = defaultdict(list)
    for i, t in enumerate(data):
        fp = t.get("file_path", "")
        if not fp:
            continue
        p = Path(fp)
        if len(p.parts) >= 3:
            album_name = p.parent.name
            artist = t.get("artist", "Unknown")
            # Use primary artist (strip feat. etc for grouping)
            primary_artist = artist.split(",")[0].split(" feat")[0].split(" Feat")[0].strip()
            albums[(primary_artist, album_name)].append(i)
    return albums


def needs_rating(data, track_indices):
    """Check if any track in the album group needs a rating."""
    return any(data[i].get("rating", 0) == 0 for i in track_indices)


def build_prompt(album_batch):
    """Build prompt for a batch of albums."""
    lines = []
    for batch_idx, (artist, album, tracks) in enumerate(album_batch):
        track_list = "\n".join(
            f"    {j}. {t['title']}" for j, t in enumerate(tracks)
        )
        lines.append(f"Album {batch_idx}:\n  Artist: {artist}\n  Album: {album}\n  Tracks:\n{track_list}")

    albums_text = "\n\n".join(lines)

    return f"""You are a music expert with deep knowledge of RateYourMusic (RYM) ratings.

For each album below, provide:
1. "album_rating": The RYM community rating (0.50 to 5.00 scale). Use your best knowledge.
   - If you know the actual RYM rating, use it.
   - If unsure, estimate based on critical reception and the artist's catalog.
   - Most albums fall between 2.50-4.00. Only true classics exceed 4.00.
2. "track_ratings": For each track, a relative rating (0.50 to 5.00).
   - Fan-favorite / standout tracks should be rated higher.
   - Deep cuts and filler should be rated lower.
   - Most tracks cluster around the album rating +/- 0.3.

{albums_text}

Respond as a JSON object where keys are album numbers (as strings).
Example: {{"0": {{"album_rating": 3.85, "track_ratings": [3.9, 3.7, 4.1, 3.5]}}, "1": {{...}}}}"""


def enrich_batch(client, album_batch):
    """Call GPT-4o to get ratings for a batch of albums."""
    prompt = build_prompt(album_batch)

    response = client.chat.completions.create(
        model="gpt-4o",  # Use the smarter model for accurate rating recall
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # Low temp for factual recall
        max_tokens=4000,
        response_format={"type": "json_object"},
    )

    text = response.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [WARN] Failed to parse JSON: {e}")
        return {}


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    data = load_library()

    # Group by album
    album_groups = group_by_album(data)
    print(f"Found {len(album_groups)} unique albums across {len(data)} tracks.")

    # Filter to albums needing ratings
    to_enrich = []
    for (artist, album), indices in album_groups.items():
        if needs_rating(data, indices):
            tracks = [data[i] for i in indices]
            to_enrich.append((artist, album, tracks, indices))

    total = len(to_enrich)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Albums needing ratings: {total}")
    print(f"Running {total_batches} batches of {BATCH_SIZE}...\n")

    if total == 0:
        print("All albums already rated!")
        return

    enriched_albums = 0
    enriched_tracks = 0
    batch_count = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch_items = to_enrich[batch_start:batch_start + BATCH_SIZE]
        album_batch = [(artist, album, tracks) for artist, album, tracks, _ in batch_items]

        batch_num = batch_count + 1
        print(f"  Batch {batch_num}/{total_batches}: "
              f"albums {batch_start + 1}-{min(batch_start + BATCH_SIZE, total)} "
              f"| {enriched_albums} albums done...")

        try:
            results = enrich_batch(client, album_batch)

            for local_idx, (artist, album, tracks, indices) in enumerate(batch_items):
                result = results.get(str(local_idx))
                if not result:
                    continue

                album_rating = result.get("album_rating", 0)
                track_ratings = result.get("track_ratings", [])

                # Apply album rating to all tracks, use track-specific if available
                for j, lib_idx in enumerate(indices):
                    if j < len(track_ratings):
                        rating = track_ratings[j]
                    else:
                        rating = album_rating

                    # Normalize to 0-1 scale (RYM is 0.5-5.0)
                    normalized = round(min(1.0, max(0.0, (rating - 0.5) / 4.5)), 3)
                    data[lib_idx]["rating"] = normalized

                    # Also store raw RYM rating for reference
                    data[lib_idx]["rym_rating"] = round(rating, 2)
                    enriched_tracks += 1

                enriched_albums += 1

        except Exception as e:
            print(f"  [ERROR] Batch {batch_num} failed: {e}")
            time.sleep(5)

        batch_count += 1

        if batch_count % SAVE_EVERY == 0:
            save_library(data)
            print(f"  [SAVE] Checkpoint. {enriched_albums}/{total} albums, "
                  f"{enriched_tracks} tracks.\n")

        time.sleep(0.5)  # Rate limit

    save_library(data)
    print(f"\nDone! Enriched {enriched_albums}/{total} albums, "
          f"{enriched_tracks} tracks with RYM ratings.")


if __name__ == "__main__":
    main()
