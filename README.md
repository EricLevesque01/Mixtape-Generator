# 🎵 ReAct Mixtape Curator

A conversational AI agent that curates personalized mixtapes from your music library using a **ReAct (Reasoning + Acting)** loop with semantic genre matching, flow optimization, and iterative refinement.

## Features

- **Curator Persona** — Acts as a professional A&R / Creative Director with deep musical awareness
- **Conversational Interview** — Dynamically gathers preferences through natural conversation
- **ReAct Agent Loop** — Validates constraints, searches the library, and intelligently adds/swaps tracks with reasoning
- **Flow Optimization** — Greedy multi-start algorithm for smooth sonic transitions
- **A/B Generation** — Produces two playlist variants for comparison
- **"Neither" Feedback Loop** — Reject both and provide specific refinement feedback
- **Multi-Format Export** — Local `.m3u8`, `.txt` (with reasoning), and optional Spotify sync
- **Genre/Descriptor Index** — O(1) inverted index for fast tag-based track discovery

## Architecture

```
Interview → Profile Extraction → Seed → Growth → Refinement → A/B → Export
     ↑                                                              |
     └──────────────── "Neither" Feedback Loop ←────────────────────┘
```

### Scoring (§3)

| Weight | Dimension | Method |
|--------|-----------|--------|
| 0.35 | **Fit** | Sonic distance (energy/valence/intensity) + genre/descriptor overlap |
| 0.25 | **Flow** | Adjacent-track transition smoothness |
| 0.15 | **Variety** | Per-track normalized genre distribution uniformity |
| 0.15 | **Accessibility** | Average track accessibility score |
| 0.10 | **Quality** | Average rating × recommendability |

## Setup

```bash
# Clone and install
git clone https://github.com/yourusername/Mixtape-Generator.git
cd Mixtape-Generator
pip install -e .
```

### Environment Variables

Create a `.env` file:

```env
# Required for Spotify export
SPOTIFY_CLIENT_ID=your_id
SPOTIFY_CLIENT_SECRET=your_secret
SPOTIFY_REDIRECT_URI=http://localhost:8888/callback

# Optional: Use real LLM instead of MockLLM
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### Interactive CLI

```bash
python -m mixtape_curator.ui_cli
```

Follow the prompts to describe the vibe, anchor artists, and duration. The curator generates two playlists (A/B) and you can choose, reject, or refine.

### Eval Harness

```bash
python -m mixtape_curator.eval_harness
```

Runs automated test cases from `data/eval_cases.jsonl` and outputs metrics to `eval_results.jsonl`.

### Data Enrichment

```bash
# Enrich subgenres & descriptors (requires OPENAI_API_KEY)
python scripts/enrich_rym_taxonomy.py

# Enrich Spotify URIs (requires Spotify credentials)
python scripts/enrich_spotify_uris.py
```

## Testing

```bash
python -m pytest tests/ -v --tb=short
```

## Project Structure

```
Mixtape-Generator/
├── config.yaml              # All tunable parameters
├── data/
│   ├── library.json          # Music library (5,965 tracks)
│   ├── eval_cases.jsonl      # Evaluation scenarios
│   └── feedback.jsonl        # User feedback log
├── scripts/
│   ├── enrich_rym_taxonomy.py
│   └── enrich_spotify_uris.py
├── src/mixtape_curator/
│   ├── models.py             # Track, Playlist, UserProfile
│   ├── library.py            # Library loader + inverted index
│   ├── interview.py          # Interview agent
│   ├── scoring.py            # 5-dimension scoring engine
│   ├── generator.py          # Draft + flow optimization
│   ├── agent.py              # ReAct repair loop
│   ├── ab_test.py            # A/B variant generation
│   ├── export.py             # TXT/M3U8 export
│   ├── spotify_export.py     # Spotify playlist sync
│   ├── ui_cli.py             # Rich CLI interface
│   └── llm/
│       ├── interface.py      # LLMProvider protocol
│       └── providers/
│           ├── local.py      # MockLLM (testing)
│           ├── openai.py     # GPT-4
│           └── anthropic.py  # Claude
└── tests/                    # 23 test suite
```

## License

MIT
