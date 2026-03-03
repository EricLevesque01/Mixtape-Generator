# Mixtape Generator

A command-line tool that curates personalized mixtapes from Eric's music library. A short conversational interview captures your current mood and moment, then a multi-stage pipeline builds two distinct playlist variants (Mix A and Mix B) for you to choose from.

---

## How it works

1. **Interview** — 7 open-ended questions understand the tone of your day, your energy, attention level, space, flow preference, direction, and atmosphere
2. **Blueprint** — An LLM acts as creative director, designing a narrative arc with themes and energy targets per segment
3. **Trellis** — Tracks from the library are matched to each segment using genre, descriptor, and audio feature similarity
4. **Arc Optimization** — A deterministic algorithm selects and sequences the best tracks into two distinct mixes (A and B)
5. **Refinement** — A second LLM pass fine-tunes the final playlist
6. **Output** — Two clean tracklists in the terminal; choose A, B, or Neither to iterate

---

## Prerequisites

- Python 3.10+
- An OpenAI API key *(optional — the app has a built-in fallback that works without one)*

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/EricLevesque01/Mixtape-Generator.git
cd Mixtape-Generator

# 2. Install dependencies (either works)
pip install -e .
# or: pip install -r requirements.txt

# 3. Set up your API key  (see below)
cp .env.example .env
# Open .env and paste in your OpenAI key

# 4. Run
python run.py
```

At the `(mixtape) >` prompt, type **`start`** to begin.

---

## Setting up your API key

Copy the example env file and fill it in:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your real key:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### How to get an OpenAI API key

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Sign in → **API Keys** → **Create new secret key**
3. Copy the key and paste it into your `.env` file

> **No key?** The app still works. It uses short fixed-phrase acknowledgments and keyword-based interpretation instead of GPT-4o. The pipeline runs end-to-end — results are just slightly less nuanced.

---

## CLI Commands

| Command | Description |
|---|---|
| `start` | Begin the interview and generate your mixtape |
| `help` | Show available commands |
| `quit` | Exit |

---

## Configuration (`config.yaml`)

Key parameters you may want to adjust:

| Setting | Default | Description |
|---|---|---|
| `duration_target_s` | `4620` | Target mix length (~77 min) |
| `duration_cap_s` | `4800` | Hard maximum (1h 20m) |
| `target_track_count` | `16` | Approximate number of tracks |
| `max_tracks_per_artist` | `1` | Max times any one artist appears |
| `rng_seed` | `12345` | Seed for reproducible results |

---

## Project Structure

```
Mixtape-Generator/
├── run.py                      # Entry point
├── config.yaml                 # All tunable parameters
├── .env.example                # API key template
├── pyproject.toml              # Dependencies
├── data/
│   ├── library.json            # Music library (tracks + metadata) — not in repo
│   └── similarity_graph.json   # Pre-computed track similarity — not in repo
├── src/mixtape_curator/
│   ├── interview.py            # 7-question conversational interview
│   ├── blueprint.py            # LLM creative director (arc design)
│   ├── trellis.py              # Track-to-segment candidate matching
│   ├── arc_optimizer.py        # Deterministic A/B playlist builder
│   ├── agent.py                # ReAct refinement loop
│   ├── scoring.py              # 5-dimension scoring engine
│   ├── models.py               # Data models (Track, UserProfile, etc.)
│   ├── library.py              # Library loader + search index
│   ├── exporter.py             # .m3u8 / .txt playlist export
│   ├── ui_cli.py               # Terminal interface (Rich)
│   └── llm/
│       ├── interface.py        # LLMProvider protocol
│       └── providers/
│           ├── openai.py       # GPT-4o / GPT-4o-mini
│           ├── local.py        # MockLLM (keyword fallback)
│           └── anthropic.py    # Claude (optional)
└── tests/                      # Test suite (48 tests)
```

---

## Running the tests

```bash
python -m pytest tests/ -q
```

---

## About the library

`data/library.json` and `data/similarity_graph.json` are not included in this repo — they're large, machine-specific files built from a personal music collection. The app won't generate mixtapes without them.

If you want to adapt this to your own library, the `scripts/` folder (not tracked in git) contains the enrichment pipeline used to build these files from an iTunes/local library + Spotify + RateYourMusic data.

---

## License

MIT
