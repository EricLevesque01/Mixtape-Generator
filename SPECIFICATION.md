# ReAct Mixtape Curator — AI Implementation Specification

**Version:** v2.2.0 (Curator Persona & Incremental Growth)

This specification is intended for direct consumption by an AI coding agent.

All required system behaviors, schemas, constraints, and execution rules are explicitly defined. This version incorporates the "A&R Curator" persona, incremental "Seed -> Growth -> Refinement" generation, and reasoning transparency.

---

## 0. System Summary

ReAct Mixtape Curator is a conversational AI application that acts as a **professional Mixtape Curator / Creative Director**. It generates curated playlists (“mixtapes”) from a user’s personal music library with a focus on cohesive journeys, specific vibes, and clear artistic reasoning.

The system is intentionally modeled after physical mixtape media:

-   Standard CD-length mixtapes ≈ 74–80 minutes.
-   Extended cassette formats allow up to ≈120 minutes.

### Execution Flow

1.  **Curator Interview**: Dynamically interview the user with an "A&R" persona to gather the "DNA" of the project (Mission Statement, Anchor Artist, Texture, Length).
2.  **Profile Extraction**: Convert responses into a structured JSON preference profile.
3.  **Phase 1: Seed**: Create an initial draft containing *only* the user's "must-have" tracks and artists.
4.  **Phase 2: Growth**: The ReAct agent actively searches the library for tracks that fit the vibe and expands the playlist to near the target duration.
5.  **Phase 3: Refinement**: The agent optimizes sequencing and flow, swapping tracks to maximize the score (target > 0.80).
6.  **Reasoning**: Every track decision (add/swap) must include a "Why it fits" reasoning string.
7.  **Production**: Produce two viable candidates (A/B).
8.  **Presentation**: Present choices (A, B, or Neither) with a rich table display showing curator notes.
9.  **Feedback Loop**: If "Neither" is selected, intelligently refine the profile based on specific feedback (e.g., "too slow", "more funk") and regenerate.
10. **Output**: Finalize playlist via local exports (TXT, M3U8) or optional Spotify sync.

**Goal:** Playlists should feel intentionally curated with deep semantic and sonic awareness.

---

## 1. Required Configuration Keys

Loaded from `config.yaml`.

```yaml
duration_target_s: 4620
duration_cap_s: 7200
max_tracks_per_artist: 2

max_questions: 10
conf_thresh: 0.7
ambitious_threshold: 0.88
accept_threshold: 0.80
max_repair_iters: 8
eps: 0.01
stall_iters: 2
off_topic_max: 3
max_ab_rounds: 3

w_fit: 0.35
w_flow: 0.25
w_variety: 0.15
w_access: 0.15
w_quality: 0.10

branch_swap_min: 2
branch_swap_max: 5
branch_variety_boost: 0.2

rng_seed: 12345
spotify_add_chunk_size: 100
spotify_playlist_public_default: false
log_sim_user_transcripts: true
```

Testing uses deterministic seeds; production mode switches to random seeds.

---

## 2. Hard Constraints

Playlist duration must satisfy:

    total_duration_s <= duration_cap_s

Default generation aims near `duration_target_s` but must not degrade quality to hit target.

Additional constraints:

-   Maximum two tracks per artist.
-   Excluded artists/genres/descriptors must not appear.
-   Must-include artists/tracks must appear if feasible.

Constraint conflicts must trigger `consult_user()`.

---

## 3. Soft Objectives & Scoring Math

Scores are normalized to `[0,1]`.

### 3.1 Fit Score

Track fit combines sonic similarity and semantic genre alignment.

#### Sonic Fit (Frozen v1)

Dimensions: energy, valence, intensity.

    d^2 = (Δenergy)^2 + (Δvalence)^2 + (Δintensity)^2
    tolerance = 0.25
    sonic_fit = exp(-d^2 / (2 * tolerance^2))

#### Genre Fit

Uses RateYourMusic taxonomy fields:

-   primary_genres
-   subgenres
-   descriptors

Full-credit genre match occurs if **either** primary or subgenre matches target genres.

Descriptors contribute lower-weight additive alignment.

Genre fit blends with sonic fit via `genre_strictness`:

    track_fit =
        sonic_fit * (1 - genre_strictness)
      + genre_fit * genre_strictness

### 3.2 Flow Score

Minimizes energy, valence, and intensity deltas while maximizing genre continuity between adjacent tracks.

### 3.3 Variety Score (Uniformity Alignment)

Variety aligns to user cohesion target.

Token construction per track:

-   primary genres weight = 1.0
-   subgenres weight = 0.7
-   descriptors weight = 0.3 (top 5 descriptors only)

Per-track tokens are normalized to sum to 1.0 ensuring equal contribution.

Playlist distribution is the average of per-track distributions.

Entropy computation:

    H = -Σ p_i log(p_i)
    K = count(tokens with p_i > 0)
    measured_diversity = 0 if K <= 1 else H / log(K)

Uniformity mapping:

    ideal_diversity = 1 - targets.uniformity
    variety_score = 1 - abs(measured_diversity - ideal_diversity)

Uniformity = 1 aims for maximal genre homogeneity.

---

## 4. Conversational Interview Framework

### Persona
The agent adopts the persona of a **Creative Director / A&R**. It uses sophisticated language ("Texture", "DNA", "Anchor Artist") and seeks to build a "Journey".

### Interview Stages

1.  **Mission Statement**: Determine mood, context, and recipient.
2.  **Anchor & Texture**: Identify key artists/songs and the sonic aesthetic.
3.  **Journey Length**: confirm approximate duration (CD-R vs EP).
4.  **Grounding**: Check for specific exclusions or must-haves.

### Confidence Aggregation

Per-axis confidences in `[0,1]`.

Required axes:

1.  Constraints
2.  Semantic intent
3.  Cohesion target
4.  Energy/mood targets

Interview stops when all required axes and global mean confidence exceed threshold or question limit reached.

---

## 5. Guardrails & Profile Integrity

Only relevant responses modify structured profile fields.

Repeated irrelevance triggers restart or default progression.

---

## 6. Library Preprocessing & Search

Metadata source: `data/library.json`

Offline enrichment requirements:

-   Spotify URIs cached offline.
-   RYM taxonomy enrichment required offline.
-   Runtime must not query external APIs.

---

## 7. Flow Optimization

Sequencing uses Greedy Multi-Start:

1.  Choose seed track (prioritizing "Opener" candidates if available, e.g. track 1/2 of an album).
2.  Iteratively append best transition.
3.  Repeat across seeds.
4.  Keep best scoring ordering.

---

## 8. Agentic ReAct Repair Loop

### Phases

1.  **Growth Phase**: If playlist is short (< 8 tracks) or short on duration (< 70% target), focus on `search_library` and `add_track` to build volume.
2.  **Refinement Phase**: Once sufficient length is reached, switch to `swap_track` and `remove_track` to optimize Score and Flow.

### Process

Draft → sequence → score → validate.

Repair continues until playlist valid, threshold met, or improvement stalls.

If stalled and invalid, agent must consult user.

---

## 9. Base-and-Branch A/B Generation

Playlist B derives from A.

    swap_count = clamp(
        round(0.2 * playlist_length),
        branch_swap_min,
        branch_swap_max
    )

Swaps occur preferentially in weakest scoring dimension.

Must-includes never replaced.

Resequence and minimally repair.

---

## 10. "Neither" Retry Loop

If "Neither" is selected:
1.  Ask user for specific feedback (e.g., "Too slow", "Remove Artist X").
2.  **Refine Profile**: Update `UserProfile` based on feedback.
3.  Regenerate A/B playlists until retry cap reached.

---

## 11. Exports

Local: console (Table format), TXT (with reasoning), M3U8.

Spotify: OAuth → create playlist → add tracks → report failures.

---

## 12. Tool Definitions

Agent may call only:

-   `search_library(query, limit)`: Returns tracks with Genre/Descriptor metadata.
-   `swap_track(old_id, new_id, reasoning)`: Replace a track. **Must provide reasoning.**
-   `add_track(track_ids, reasonings)`: Add tracks. **Must provide reasoning dict.**
-   `remove_track(track_id)`: Remove a track.
-   `consult_user(question)`: Ask the user.
-   `finalize()`: Finish the loop.

---

## 13. Core Data Schemas

### Track Object

```json
{
  "id": "uuid",
  "title": "string",
  "artist": "string",
  "duration_s": 0,
  "rym_data": {
    "primary_genres": [],
    "subgenres": [],
    "descriptors": []
  },
  "energy": 0.0,
  "valence": 0.0,
  "intensity": 0.0,
  "accessibility": 0.0,
  "familiarity": 0.0,
  "rating": 0.0,
  "recommendability": 0.0,
  "file_path": "string|null",
  "spotify_uri": "string|null"
}
```

### Profile State

```json
{
  "must_include_track_ids": [],
  "must_include_artists": [],
  "exclude_track_ids": [],
  "exclude_artists": [],
  "exclude_genres": [],
  "exclude_descriptors": [],
  "target_genres": [],
  "target_descriptors": [],
  "genre_strictness": 0.0,
  "targets": {
    "uniformity": 0.0,
    "energy": 0.0,
    "valence": 0.0,
    "intensity": 0.0,
    "accessibility": 0.0,
    "familiarity": 0.0
  },
  "context_notes": null
}
```

### Playlist State

```json
{
  "track_ids": [],
  "total_duration_s": 0,
  "scores": {},
  "violations": [],
  "track_notes": { "track_id": "Reasoning string" }
}
```

### Playlist State

```json
{
  "track_ids": [],
  "total_duration_s": 0,
  "scores": {},
  "violations": []
}
```

---

## 14. Multi-LLM Evaluation

Supports simulated user evaluation. Logged metrics include:

-   question_count
-   repair_iterations
-   constraint_pass
-   final_score
-   ab_difference

---

## 15. Repository Structure

```
mixtape_curator/
  config.yaml
  data/
    library.json
    feedback.jsonl
    eval_cases.jsonl
    eval_results.jsonl
  scripts/
    enrich_spotify_uris.py
    enrich_rym_taxonomy.py
  src/
    mixtape_curator/
      models.py
      library.py
      interview.py
      scoring.py
      generator.py
      ab_test.py
      export.py
      spotify_export.py
      ui_cli.py
      llm/
        interface.py
        providers/
          openai.py
          anthropic.py
          local.py
  tests/
```

---

## 16. Acceptance Criteria

System must:

-   Complete interview with cohesion/genre guardrails.
-   Generate valid A/B playlists utilizing semantic genre matching.
-   Score variety via uniformity alignment.
-   Export playlists locally and optionally sync with Spotify.
-   Produce reproducible results in testing mode.

---

## Implementation Freeze

Specification is stable for implementation. Future changes require version increment beyond v2.1.1.
