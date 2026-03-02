"""
rym_genre_hierarchy.py

RateYourMusic genre taxonomy — maps each genre/subgenre to its parent chain.
Used by build_similarity_graph.py to propagate genre membership up the tree
with decaying weights, so tracks sharing a parent genre are more similar
even when their leaf genres differ (e.g. Dream Pop ≈ Shoegaze via Indie Rock).

Structure: GENRE_PARENTS = { child: [immediate_parent, grandparent, ...] }
Weight decay per level: 1.0 → 0.5 → 0.25 → 0.125
"""

# Top-level roots (no parent)
ROOTS = {
    "Rock", "Pop", "Electronic", "Hip Hop", "Jazz", "Classical",
    "Folk", "R&B", "Soul", "Country", "Metal",
}

# Genre → [parent, grandparent, ...] — ordered leaf → root
GENRE_PARENTS: dict[str, list[str]] = {

    # ── ROCK TREE ────────────────────────────────────────────────────────────
    "Alternative Rock":     ["Rock"],
    "Indie Rock":           ["Alternative Rock", "Rock"],
    "Art Rock":             ["Alternative Rock", "Rock"],
    "Post-Punk":            ["Alternative Rock", "Rock"],
    "Post-Punk Revival":    ["Post-Punk", "Alternative Rock", "Rock"],
    "Garage Rock Revival":  ["Alternative Rock", "Rock"],
    "Post-Grunge":          ["Alternative Rock", "Rock"],
    "Lo-Fi":                ["Alternative Rock", "Rock"],

    # Indie subgenres
    "Dream Pop":            ["Indie Rock", "Alternative Rock", "Rock"],
    "Shoegaze":             ["Indie Rock", "Alternative Rock", "Rock"],
    "Jangle Pop":           ["Indie Rock", "Alternative Rock", "Rock"],
    "Slacker Rock":         ["Indie Rock", "Alternative Rock", "Rock"],
    "Sadcore":              ["Indie Rock", "Alternative Rock", "Rock"],
    "Noise Pop":            ["Indie Rock", "Alternative Rock", "Rock"],
    "Indie Pop":            ["Indie Rock", "Alternative Rock", "Rock"],
    "Bedroom Pop":          ["Indie Rock", "Alternative Rock", "Rock"],
    "bedroom pop":          ["Indie Rock", "Alternative Rock", "Rock"],
    "Lo-Fi Indie":          ["Indie Rock", "Alternative Rock", "Rock"],

    # Power / Punk-adjacent
    "Power Pop":            ["Indie Rock", "Alternative Rock", "Rock"],
    "Pop Punk":             ["Alternative Rock", "Rock"],
    "Midwest Emo":          ["Emo", "Alternative Rock", "Rock"],
    "Emo":                  ["Alternative Rock", "Rock"],
    "Punk Rock":            ["Rock"],

    # Classic rock branches
    "Piano Rock":           ["Rock"],
    "Acoustic Rock":        ["Rock"],
    "Roots Rock":           ["Rock"],
    "Classic Rock":         ["Rock"],
    "Singer-Songwriter":    ["Rock", "Folk"],
    "Folk Rock":            ["Rock", "Folk"],
    "Psychedelic Rock":     ["Rock"],
    "Britpop":              ["Alternative Rock", "Rock"],
    "Post-Britpop":         ["Alternative Rock", "Rock"],

    # ── POP TREE ─────────────────────────────────────────────────────────────
    "Alt Pop":              ["Pop"],
    "Art Pop":              ["Pop"],
    "Synthpop":             ["Pop", "Electronic"],
    "Dance Pop":            ["Pop"],
    "Electropop":           ["Synthpop", "Pop", "Electronic"],
    "Chamber Pop":          ["Pop"],
    "Baroque Pop":          ["Chamber Pop", "Pop"],
    "Sophisti-Pop":         ["Pop"],
    "Teen Pop":             ["Pop"],
    "Boy Band":             ["Pop"],
    "Pop Rock":             ["Pop", "Rock"],
    "Pop Soul":             ["Pop", "Soul"],
    "Jazz Pop":             ["Pop", "Jazz"],
    "Country Pop":          ["Pop", "Country"],
    "Dream Pop":            ["Indie Rock", "Alternative Rock", "Rock"],
    "Singer-Songwriter":    ["Rock", "Folk"],

    # ── ELECTRONIC TREE ──────────────────────────────────────────────────────
    "Electronic":           [],
    "Ambient":              ["Electronic"],
    "IDM":                  ["Electronic"],
    "Downtempo":            ["Electronic"],
    "Trip Hop":             ["Hip Hop", "Electronic"],
    "UK Garage":            ["Electronic"],
    "Synth Funk":           ["Electronic", "R&B"],

    # ── HIP HOP TREE ─────────────────────────────────────────────────────────
    "Hip Hop":              [],
    "Pop Rap":              ["Hip Hop", "Pop"],
    "East Coast Hip Hop":   ["Hip Hop"],
    "Conscious Hip Hop":    ["Hip Hop"],
    "Consious Hip Hop":     ["Hip Hop"],   # common misspelling in RYM
    "Alternative Hip Hop":  ["Hip Hop"],
    "Trap":                 ["Hip Hop"],

    # ── R&B / SOUL TREE ──────────────────────────────────────────────────────
    "R&B":                  [],
    "Soul":                 [],
    "Neo-Soul":             ["R&B", "Soul"],
    "Contemporary R&B":     ["R&B"],
    "Alternative R&B":      ["R&B"],
    "Quiet Storm":          ["R&B", "Soul"],
    "Funk":                 ["R&B", "Soul"],

    # ── JAZZ TREE ────────────────────────────────────────────────────────────
    "Jazz":                 [],
    "Vocal Jazz":           ["Jazz"],
    "Jazz Fusion":          ["Jazz"],
    "Bebop":                ["Jazz"],
    "Cool Jazz":            ["Jazz"],
    "Bossa Nova":           ["Jazz"],

    # ── FOLK TREE ────────────────────────────────────────────────────────────
    "Folk":                 [],
    "Indie Folk":           ["Folk", "Indie Rock"],
    "Alt Country":          ["Country", "Folk"],
    "Americana":            ["Folk", "Country"],
    "Traditional Folk":     ["Folk"],
    "Contemporary Folk":    ["Folk"],

    # ── COUNTRY TREE ─────────────────────────────────────────────────────────
    "Country":              [],

    # ── OTHER ────────────────────────────────────────────────────────────────
    "Alternative":          ["Alternative Rock", "Rock"],
    "Sound Collage":        [],
}

# Propagation decay factor per level (child=1.0, parent=DECAY, grandparent=DECAY^2...)
DECAY = 0.45


def get_genre_chain(genre: str) -> dict[str, float]:
    """
    Returns {genre: weight} for the genre and all its ancestors,
    with exponentially decaying weights up the tree.
    """
    result: dict[str, float] = {genre: 1.0}
    parents = GENRE_PARENTS.get(genre, [])
    for level, parent in enumerate(parents, start=1):
        weight = DECAY ** level
        if parent not in result or result[parent] < weight:
            result[parent] = weight
    return result
