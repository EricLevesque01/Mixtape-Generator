"""
Library data quality audit.
Shows which fields are missing across the library so you know what to enrich.

Usage: python scripts/audit_gaps.py
"""
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mixtape_curator.library import library

library.load()

# Build gap summary
gap_counter = Counter()
tracks_with_gaps = 0
total_tracks = 0

for _, row in library.df.iterrows():
    track = library._row_to_track(row)
    total_tracks += 1
    if track.enrichment_gaps:
        tracks_with_gaps += 1
        for gap in track.enrichment_gaps:
            gap_counter[gap] += 1

print(f"=== Library Data Quality Audit ===")
print(f"Total tracks:      {total_tracks}")
print(f"Tracks with gaps:  {tracks_with_gaps} ({tracks_with_gaps/total_tracks*100:.1f}%)")
print(f"Fully enriched:    {total_tracks - tracks_with_gaps} ({(total_tracks - tracks_with_gaps)/total_tracks*100:.1f}%)")
print()

print(f"{'Field':<20} {'Missing':>8} {'Coverage':>10}")
print("-" * 40)
for field, count in gap_counter.most_common():
    coverage = (total_tracks - count) / total_tracks * 100
    print(f"{field:<20} {count:>8} {coverage:>9.1f}%")
