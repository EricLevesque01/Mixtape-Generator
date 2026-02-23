"""
Enrich library.json with subgenres and descriptors using LLM inference.

Usage:
    python scripts/enrich_rym_taxonomy.py

Requires OPENAI_API_KEY in .env file.
Processes tracks one at a time in small batches, saving progress incrementally.
Designed to restart safely — skips already-enriched tracks.
"""
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from openai import OpenAI

DATA_PATH = project_root / "data" / "library.json"
BATCH_SIZE = 10   # Smaller batches for better accuracy
SAVE_EVERY = 10   # Save every N batches


def load_library():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_library(data):
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def needs_enrichment(track):
    """Check if a track is missing subgenres or descriptors."""
    rym = track.get("rym_data", {})
    return (not rym.get("subgenres")) or (not rym.get("descriptors"))


def build_prompt(batch):
    """
    Build a prompt that returns a JSON OBJECT (required by OpenAI json_object mode).
    Keys are 0-indexed track numbers so we can match by key, not position.
    """
    lines = []
    for i, t in enumerate(batch):
        primary = ", ".join(t.get("rym_data", {}).get("primary_genres", []))
        lines.append(f'{i}: {t["artist"]} - {t["title"]} (Primary: {primary or "Unknown"})')

    tracks_text = "\n".join(lines)

    return f"""You are a music taxonomy expert familiar with RateYourMusic (RYM) genres and descriptors.

For each numbered track below, infer:
1. subgenres (1-3 specific subgenre names from RYM, e.g. "Dream Pop", "Post-Punk Revival", "Indie Folk")
2. descriptors (3-5 mood/style words from RYM, e.g. "melancholic", "ethereal", "driving", "lo-fi", "atmospheric")

Tracks:
{tracks_text}

Respond as a JSON object where keys are the track numbers (as strings) and values have "subgenres" and "descriptors" arrays.
Example: {{"0": {{"subgenres": ["Indie Pop"], "descriptors": ["dreamy", "lo-fi"]}}, "1": {{...}}}}"""


def enrich_batch(client, batch, model="gpt-4o-mini"):
    """Call the LLM to enrich a batch. Returns dict keyed by track index (str)."""
    prompt = build_prompt(batch)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"}
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

    # Find tracks needing enrichment
    to_enrich = [(i, t) for i, t in enumerate(data) if needs_enrichment(t)]
    total = len(to_enrich)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Found {total} tracks needing enrichment out of {len(data)} total.")
    print(f"Running {total_batches} batches of {BATCH_SIZE}...\n")

    if total == 0:
        print("All tracks already enriched!")
        return

    enriched_count = 0
    failed_count = 0
    batch_count = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch_items = to_enrich[batch_start:batch_start + BATCH_SIZE]
        batch_tracks = [t for _, t in batch_items]
        batch_indices = [i for i, _ in batch_items]

        batch_num = batch_count + 1
        print(f"  Batch {batch_num}/{total_batches}: "
              f"tracks {batch_start + 1}-{min(batch_start + BATCH_SIZE, total)} "
              f"| {enriched_count} enriched so far...")

        try:
            results = enrich_batch(client, batch_tracks)

            # Match by key (track index within batch), not by position
            batch_enriched = 0
            for local_idx, (lib_idx, _) in enumerate(zip(batch_indices, batch_tracks)):
                enrichment = results.get(str(local_idx)) or results.get(local_idx)
                if not enrichment:
                    failed_count += 1
                    continue

                rym = data[lib_idx].setdefault("rym_data", {})
                if enrichment.get("subgenres"):
                    rym["subgenres"] = enrichment["subgenres"]
                if enrichment.get("descriptors"):
                    rym["descriptors"] = enrichment["descriptors"]
                enriched_count += 1
                batch_enriched += 1

            if batch_enriched < len(batch_tracks):
                print(f"    [INFO] Enriched {batch_enriched}/{len(batch_tracks)} in this batch.")

        except Exception as e:
            print(f"  [ERROR] Batch {batch_num} failed: {e}")
            time.sleep(5)  # Back off on error

        batch_count += 1

        # Save periodically
        if batch_count % SAVE_EVERY == 0:
            save_library(data)
            pct = (enriched_count / total * 100) if total else 0
            print(f"  [SAVE] Checkpoint. {enriched_count}/{total} enriched ({pct:.1f}%). "
                  f"{failed_count} failed.\n")

        # Gentle rate limiting
        time.sleep(0.3)

    # Final save
    save_library(data)
    pct = (enriched_count / total * 100) if total else 0
    print(f"\nDone! Enriched {enriched_count}/{total} tracks ({pct:.1f}%). "
          f"{failed_count} had no LLM result.")


if __name__ == "__main__":
    main()
