# Mixtape Generator

A CLI tool that builds personalized mixtapes from a local music library. It uses a short conversational interview to understand your mood and the moment, then runs a multi-stage pipeline to curate two playlist variants (A and B) for you to choose from.

---

## How it works

1. **Interview** — 7 open-ended questions capture your current state (tone of day, energy, attention, space, flow, direction, atmosphere)
2. **Blueprint** — An LLM acts as creative director and designs a narrative arc (segments with themes and energy targets)
3. **Trellis** — Tracks from the library are matched to each segment using genre, descriptor, and audio feature similarity
4. **Arc Optimization** — A deterministic algorithm selects and sequences the best tracks, producing two distinct mixes
5. **Refinement** — A second LLM pass fine-tunes the final playlist
6. **Output** — Two clean tracklists displayed in the terminal; choose A, B, or Neither to iterate

---

## Prerequisites

- Python 3.10+
- An OpenAI API key (for the full experience — runs without one using a built-in fallback)
- Optional: Spotify credentials (for exporting directly to a Spotify playlist)

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/EricLevesque01/Mixtape-Generator.git
cd Mixtape-Generator

# 2. Install
pip install -e .

# 3. Set up your environment file
cp .env.example .env
# Then open .env and fill in your API key(s)

# 4. Run
python run.py
```

At the `(mixtape) >` prompt, type `start` to begin the interview.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values you need:

```bash
cp .env.example .env
```

| Variable | Required? | Description |
|---|---|---|
| `OPENAI_API_KEY` | Recommended | Powers interview acknowledgments (`gpt-4o-mini`) and final interpretation + blueprint generation (`gpt-4o`). Without it, the app falls back to keyword matching — functional but less nuanced. |
| `SPOTIFY_CLIENT_ID` | Optional | Enables direct export to a Spotify playlist. |
| `SPOTIFY_CLIENT_SECRET` | Optional | See above. |
| `SPOTIFY_REDIRECT_URI` | Optional | Default: `http://localhost:8888/callback` |

### Getting an OpenAI API key

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign in → **API Keys** → **Create new secret key**
3. Paste it into your `.env` file:
   ```
   OPENAI_API_KEY=sk-...
   ```

### Getting Spotify credentials (optional)

1. Go to [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard)
2. Create an app → copy the **Client ID** and **Client Secret**
3. In the app settings, add `http://localhost:8888/callback` as a Redirect URI
4. Fill the three `SPOTIFY_*` values in your `.env`

---

## Running without an API key

The app works fully without any API key — it uses keyword-based fallbacks:

- Interview acknowledgments are brief fixed phrases ("Got it.", "Nice.", etc.)
- The persona interpretation uses keyword matching instead of GPT-4o
- Blueprint generation uses a heuristic arc instead of the LLM

The tracklist quality will be slightly less personalized but the pipeline runs end-to-end.

---

## CLI Commands

| Command | Description |
|---|---|
| `start` | Begin the interview and generate a mixtape |
| `help` | Show available commands |
| `quit` | Exit |

---

## Configuration (`config.yaml`)

Key parameters you might want to adjust:

| Setting | Default | Description |
|---|---|---|
| `duration_target_s` | `4620` | Target length (~77 min) |
| `duration_cap_s` | `4800` | Hard max (1h20m) — tracks trimmed from end if exceeded |
| `target_track_count` | `16` | Approximate track count target |
| `max_tracks_per_artist` | `2` | Artist diversity cap |
| `rng_seed` | `42` | Seed for reproducible results |

---

## Project Structure

```
Mixtape-Generator/
├── run.py                      # Entry point
├── config.yaml                 # All tunable parameters
├── .env.example                # Template for API keys
├── data/
│   ├── library.json            # The music library (tracks, scores, metadata)
│   └── similarity_graph.json   # Pre-computed track similarity graph
├── scripts/                    # Data enrichment + library scanning scripts
├── src/mixtape_curator/
│   ├── interview.py            # 7-question conversational interview
│   ├── blueprint.py            # LLM creative director (arc design)
│   ├── trellis.py              # Track-to-segment matching
│   ├── arc_optimizer.py        # Deterministic A/B playlist builder
│   ├── agent.py                # ReAct refinement loop
│   ├── scoring.py              # 5-dimension scoring engine
│   ├── models.py               # Data models (Track, UserProfile, etc.)
│   ├── library.py              # Library loader + search index
│   ├── exporter.py             # .m3u8 / .txt export
│   ├── spotify_export.py       # Spotify playlist sync
│   ├── ui_cli.py               # Terminal interface (Rich)
│   └── llm/
│       ├── interface.py        # LLMProvider protocol
│       └── providers/
│           ├── openai.py       # GPT-4o / GPT-4o-mini
│           ├── local.py        # MockLLM (keyword fallback)
│           └── anthropic.py    # Claude (optional)
└── tests/                      # 46+ tests
```

---

## Running the tests

```bash
python -m pytest tests/ -q
```

---

## About the library

The included `data/library.json` is Eric's personal music library — ~5,900 tracks with enriched metadata (genres, descriptors, audio features, RYM ratings). If you want to use your own library, see the scripts in `scripts/` for how to scan and enrich a library from Spotify or a local collection.

---

## License

MIT
