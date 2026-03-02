# Mixtape Generator — Specification V5.1

> **Scope:** Pure CLI tool. No web app, no public API. All interaction through terminal.
> **Freeze:** This document incorporates V5.1 Freeze Patch contracts (FP-1–FP-10). Changes to frozen behavior require a V5.2 bump.

---

## 1. Overview

A personal mixtape generation system that produces curated, coherent playlists from a local iTunes library. Generation is guided by a persona interview, enriched by Spotify listening data, and ranked using signals empirically validated against the user's existing mixtapes.

**Stack:**
- Python 3.11+, Pydantic v2, NumPy, LiteLLM
- iTunes XML + Librosa for audio features
- RateYourMusic data (scraped genres, ratings, descriptors)
- Spotify Web API (OAuth, listening data enrichment)

---

## 2. Data Pipeline (Offline, Run Once)

```
Library.xml + audio files
       │
       ▼
scan_library.py          → data/library.json     (base metadata)
enrich_librosa.py        → adds sonic features   (energy, valence, tempo, …)
enrich_rym_taxonomy.py   → adds rym_data         (genres, subgenres, descriptors)
enrich_rym_ratings.py    → adds rym_rating        (0.5–5.0)
enrich_years.py          → adds release_year
enrich_mix_signals.py    → adds mix_prominence, artist_mix_prominence
enrich_spotify_listening.py → adds spotify_affinity, spotify_uri
build_similarity_graph.py → data/similarity_graph.json
```

### Key Library Fields (per track in `library.json`)

| Field | Type | Description |
|---|---|---|
| `energy`, `valence`, `intensity`, `tempo`, `danceability` | float | Librosa-derived sonic features (0–1) |
| `acousticness`, `brightness`, `dynamic_range` | float | Extended sonic features |
| `rym_data` | object | `primary_genres`, `subgenres`, `descriptors[]` |
| `rym_rating` | float | RYM community rating (0.5–5.0) |
| `rating` | float | Normalized RYM rating (0–1) |
| `liked` / `album_loved` | bool | iTunes heart (track or album level) |
| `mix_appearances` | int | Raw playlist count |
| `mix_prominence` | float | Weighted + normalized playlist presence (0–1) |
| `artist_mix_prominence` | float | Artist-level aggregate (0–1) |
| `spotify_affinity` | float | Spotify listening signal (0–1) |
| `spotify_uri` | str | Linked Spotify track URI |

---

## 3. Similarity Graph (`build_similarity_graph.py`)

### 3.1 Feature Matrices

Two independently L2-normalized matrices are built:

- **Sonic matrix** (N × 8): energy, valence, intensity, danceability, acousticness, brightness, dynamic_range, tempo/250
- **Genre matrix** (N × V): Hierarchical one-hot encoding using the **RYM genre taxonomy** (`rym_genre_hierarchy.py`). Each genre propagates to its ancestors with **0.45× decay per level** (e.g. Dream Pop → Indie Rock → Alternative Rock → Rock). Primary genres weight 1.0, subgenres 0.7, descriptors 0.4.

### 3.2 Similarity Score (per neighbor pair)

```
score = 0.15 × sonic_cosine
      + 0.15 × genre_cosine
      + 0.35 × mix_cooccurrence     ← strongest signal
      + 0.35 × rym_rating_of_neighbor
```

> **Validated:** Grid search over 153,782 mixtape co-occurrence pairs yielded sep=+0.1945 vs +0.1197 for prior heuristic weights.

**Liked neighbor boost:** ×1.15 if neighbor is iTunes-liked or album-loved.

**Mix co-occurrence formula:**
```
shared_playlists / max(|playlists_i|, |playlists_j|)
```

### 3.3 Graph Framing (FP-1)

`similarity_graph.json` is **not a pure similarity graph** — it is a **retrieval bias graph** that intentionally blends:
- relational similarity (sonic + genre cosine)
- taste evidence (mix co-occurrence)
- quality prior (neighbor RYM rating)

This "desirability leakage" is intentional. The graph is used for candidate retrieval, not scientific similarity measurement. The formula in §3.2 is frozen.

### 3.4 Output

`data/similarity_graph.json` — top-25 neighbors per track, with `sonic`, `genre`, `mix`, `rym`, `in_mix`, `liked` scores stored per neighbor.

---

## 4. Spotify Enrichment (`enrich_spotify_listening.py`)

### 4.1 Data Sources & Weights

| Source | Weight | Description |
|---|---|---|
| User playlists | 3 | Tracks in your own Spotify playlists |
| Top tracks (4-week) | 2 | Current listening |
| Top artists | 2 | All tracks by top artists in library |
| Top tracks (6-month) | 1 | Medium-term listening |
| Top tracks (all-time) | 1 | All-time listening |
| Recently played | 1 | Last 50 plays |
| Saved tracks | 1 | Spotify liked tracks (up to 500, paginated) |

### 4.2 `spotify_affinity` Formula
```
affinity = min(1.0, total_points / 10.0)
```
Points accumulate per track across all matched sources. Artist-level matching gives all library tracks by a top artist the artist weight (2 pts).

**OAuth scopes:** `user-top-read`, `user-read-recently-played`, `user-library-read`, `playlist-read-private`

---

## 5. Quality Score (used in generation ranking)

```
quality = rating × 0.30
        + liked  × 0.30
        + mix_prominence × 0.25
        + spotify_affinity × 0.15
```

> **Validated:** Grid search over 1,057 in-mix vs 5,191 out-of-mix tracks. Best config sep=+0.2593 (vs +0.2478 at prior weights).

`liked` = 1.0 if `track.liked OR track.album_loved`, else 0.0.

---

## 6. Scoring Engine (`scoring.py`)

### 6.1 Playlist Scores

```python
PlaylistScores:
  fit         # Gaussian sonic fit to profile targets
  flow        # Sonic transition smoothness
  variety     # Genre diversity + artist variety
  accessibility
  quality     # Per-track quality avg (§5)
  total       # Weighted composite
```

### 6.2 Fit Score
Gaussian decay on energy/valence/intensity distance from `UserProfile.targets`:
```
sonic_fit = exp(-d² / 2σ²)  where σ = 0.25
```
Genre fit: full credit if primary/subgenre overlaps target genres.

**Descriptor fit credit (FP-6 — frozen):**
- Only the top 5 descriptors per track contribute
- Credit: `+0.05` per matching descriptor, max 3 matches
- Hard cap: `descriptor_bonus_cap = 0.15`
- `track_fit = clamp(base_fit + descriptor_bonus, 0.0, 1.0)`

### 6.3 Flow Score
Energy/valence/intensity transition smoothness across track pairs. Weighted: energy 50%, valence 30%, intensity 20%.

### 6.4 Variety Score
```
variety = 0.70 × genre_variety + 0.30 × artist_variety
```
**Artist variety:** base ratio (unique/total) minus duplicate penalty (−0.15 per extra appearance of any artist, capped at 0 floor). Strongly steers toward ≤2 tracks per artist.

### 6.5 Segment Boundary Flow (§6)
Scores the sonic transition between the last track of segment A and first track of segment B:
```
1.0 - sqrt((ΔE² + ΔV² + ΔI²) / 3)
```
Range: 0.0 (jarring) → 1.0 (seamless).

---

## 7. Persona System

### 7.1 PersonaProfile (§3A)

Collected by the interview agent:

| Field | Maps to |
|---|---|
| `occasion` | energy, accessibility targets |
| `mood_today` | energy blend (60% occasion + 40% mood) + valence |
| `aesthetic_choice` | intensity, valence refinement, genre_strictness |
| `personality_words` | RYM descriptors + uniformity bias |
| `era_preference` | familiarity target |
| `wildcard` | extra descriptors |
| `confidence` | Per-axis confidence (0–1), populated by interview |

`PersonaProfile.to_user_profile()` maps all signals to a `UserProfile` for the generator.

### 7.2 UserProfile

```python
UserProfile:
  recipient, context_notes
  must_include_track_ids / must_include_artists
  exclude_track_ids / exclude_artists / exclude_genres / exclude_descriptors
  target_genres, target_descriptors
  genre_strictness: float       # 0.0 = open, 1.0 = strict
  targets: FeedbackTargets      # energy, valence, intensity, uniformity, familiarity, accessibility
  duration_target_s: Optional[int]
```

---

## 8. V5 Architecture Contract

### 8.1 Generation Pipeline

```
PersonaProfile
  └─ §3A  Interview / Persona Mapping
       │
       ▼
  Blueprint (§3B)         LLM Creative Director
  7–8 Segments with themes, audio/semantic targets
       │
       ▼
  SegmentTrellis (§3C)    Algorithm: ≥20 candidates per segment
       │
       ▼
  ArcDraft ×2 (§3D)       Arc Optimizer: selects 3–5 segments, orders tracks
  Draft A / Draft B       Differ by ≥1 segment; Jaccard(segs) ≤ 0.8
       │
       ▼
  Playlist + Rationale (§3E)  LLM Refiner adds rationale before exit
```

### 8.2 Segment (§4)

A narrative chapter within the blueprint. **Exactly one** of `target_duration_s` or `target_track_count` must be set.

```python
Segment:
  segment_id: str
  theme: str
  target_duration_s: Optional[int]   # XOR
  target_track_count: Optional[int]  # XOR
  audio_targets: Dict[str, float]    # algorithm-readable only
  semantic_targets: Dict[str, List[str]]
  uniformity_bias: float             # 0.0 eclectic → 1.0 uniform
  notes: str
```

### 8.3 Blueprint (§3B)

LLM output: 7–8 segments + a plain-text `narrative_arc` description. Links to the `PersonaProfile` via `persona_state_id`.

### 8.4 SegmentTrellis (§3C)

Deterministic algorithm output. Maps each `segment_id` → sorted list of candidate `track_id`s (target ≥ 20). Records `undersized_segments` and `diagnostic_codes` for any failures.

### 8.5 ArcDraft (§3D)

```python
ArcDraft:
  draft_id, label       # "A" or "B"
  selected_segment_ids  # 3–5 chosen
  tracks_per_segment    # seg_id → [track_id]
  global_order          # flat final track order
  scores: PlaylistScores
  boundary_flow_score   # §6 cross-segment transition quality
  duration_error_s      # |actual - target| in seconds
  diagnostic_codes      # failure codes if arc unsatisfied
```

### 8.6 Arc Optimizer Priority (FP-7 — frozen)

Draft A and Draft B are selected using this strict priority order:
1. Satisfy all hard constraints
2. Maximize `total_score`
3. Minimize `duration_error_s = |actual - duration_target_s|`
4. Maximize `boundary_flow_score`

### 8.7 A/B Diversity Guarantee (FP-8 — frozen)

- Draft A and Draft B must differ by **≥1 segment**
- `Jaccard(A.selected_segments, B.selected_segments) ≤ 0.8`
- If no distinct B can be found that meets constraints: emit `diagnostic_codes += ["AB_DISTINCTNESS_INFEASIBLE"]` and return best available alternative.

### 8.8 Segment Candidate Pool Construction (FP-4 — frozen)

**Step 1 — Hard filter:**
- Excluded artists / genres / descriptors / tracks
- Track duration min/max
- Optional liked-only gate (if requested)
- Must-include handling

**Step 2 — Rank candidates:**
```
segment_candidate_score =
    0.70 × segment_fit(track, segment.audio_targets + semantic_targets)
  + 0.30 × quality(track)
```

**Pool size targets:**
- Target: ≥ 20 candidates per segment
- Minimum acceptable: ≥ 10
- If < 10: trigger undersized recovery (§8.9)

### 8.9 Undersized Segment Recovery (FP-5 — frozen)

If a segment yields < 10 candidates after filtering:
1. Relax semantic strictness once (exclusions are never relaxed):
   - Widen genre matching to include ancestor genres via RYM hierarchy, and/or
   - Lower `genre_strictness` by 0.15 (floored at 0.0)
2. Rebuild candidate pool.

If still < 10 after relaxation:
- `undersized_segments += [segment_id]`
- `diagnostic_codes += ["SEGMENT_POOL_UNDERSIZED:<segment_id>"]`

Arc optimizer must avoid infeasible segments unless no valid arc exists.

### 8.10 Determinism Contract (FP-9 — frozen)

In test mode, with the same `library.json`, `similarity_graph.json`, config, and seed:
- All algorithmic outputs are deterministic
- LLM variability controlled by: recorded fixtures for `PersonaProfile` / `Blueprint`, or `temperature=0` with fixed model/prompt versions
- Acceptance: identical fixtures ⇒ identical A/B tracklists and ordering

---

## 9. CLI Scripts

| Script | Purpose |
|---|---|
| `scripts/scan_library.py` | Initial library scan from iTunes XML |
| `scripts/enrich_librosa.py` | Audio feature extraction |
| `scripts/enrich_rym_taxonomy.py` | Genre/descriptor scrape from RYM |
| `scripts/enrich_rym_ratings.py` | RYM rating scrape |
| `scripts/enrich_years.py` | Release year enrichment |
| `scripts/enrich_mix_signals.py` | Mix co-occurrence signals |
| `scripts/enrich_spotify_listening.py` | Spotify affinity enrichment |
| `scripts/enrich_spotify_uris.py` | Spotify URI linking |
| `scripts/build_similarity_graph.py` | Build similarity graph |
| `scripts/recommend.py` | Seed-based recommendations |
| `scripts/rym_genre_hierarchy.py` | RYM genre taxonomy (imported) |
| `src/mixtape_curator/` | Core library (generator, scoring, interview, agent) |
| `run.py` | Main entrypoint |

### Recommend script
```bash
python scripts/recommend.py "Song Title" "Another Song" --n 20 --show-scores
python scripts/recommend.py "Paprika" "Nomad" --n 15 --liked-only
```

---

## 10. Hard Constraints (Generation)

### 10.1 Artist Cap
- Max 2 tracks per artist per playlist

### 10.2 Duration Semantics (FP-2 — frozen)
- **Hard cap:** `total_duration_s <= duration_cap_s`
- **Soft floor:** `total_duration_s >= floor_target_s` where `floor_target_s = 0.90 × duration_target_s`
- **Infeasibility exception:** If the floor is unreachable, the generator may return a shorter playlist **only if** it emits:
  - `diagnostic_codes += ["DURATION_FLOOR_INFEASIBLE"]`
  - `duration_shortfall_s = floor_target_s - total_duration_s`
- **Non-goal:** The system must not add low-quality filler tracks to hit the floor.

### 10.3 Segment Duration Tolerance (FP-3 — frozen)
- Segment assembly must hit within ±10% of `target_duration_s` *when feasible*
- If infeasible, deviation beyond ±10% is allowed only if: `diagnostic_codes += ["SEGMENT_DURATION_INFEASIBLE:<segment_id>"]`
- Fallback: arc optimizer may replace infeasible segment with alternate from blueprint pool

### 10.4 Exclusions
- No excluded artists / genres / tracks
- Genre strictness threshold respected when `target_genres` is set

### 10.5 Must-Include Conflict Resolution (FP-10 — frozen)
If a must-include track conflicts with an exclusion, the system must **not** silently ignore it:
- **Interactive (CLI):** Prompt the user with a forced choice:
  1. Drop the must-include
  2. Drop the exclusion
  3. Replace must-include with a similar alternative
- **Non-interactive:** Default to dropping must-include and emit `diagnostic_codes += ["MUST_INCLUDE_CONFLICT_DROPPED"]`

---

## 11. Configuration (`config.yaml`)

Key settings:
- `duration_target_s` — default mixtape length (seconds)
- `weights` — scorer component weights (fit, flow, variety, accessibility, quality)
- `top_k` — neighbors per track in graph (default: 25)
- `liked_boost` — score multiplier for liked neighbors (default: 1.15)
