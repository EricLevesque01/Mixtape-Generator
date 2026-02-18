# Mixtape Generator

A ReAct-based AI agent that curates personalized mixtapes with iterative refinement.

## Features
- **ReAct Agent:** Validates constraints like "No heavy metal" or "Must include Frank Ocean".
- **Flow Optimization:** Uses greedy multi-start algorithms to ensure smooth transitions between tracks.
- **Interactive Refinement:** Reject entire drafts ("Neither A nor B") and provide feedback ("Too slow", "Too varied") to steer the AI.
- **Multi-Format Export:**
  - Local `.m3u8` playlists.
  - Text summaries.
  - Direct **Spotify Export** (requires API credentials).

## Setup
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Environment Variables:**
    Create a `.env` file with your Spotify credentials for export support:
    ```env
    SPOTIFY_CLIENT_ID=your_id
    SPOTIFY_CLIENT_SECRET=your_secret
    SPOTIFY_REDIRECT_URI=http://localhost:8888/callback
    ```

## Usage
Run the interactive CLI:
```bash
python -m mixtape_curator.ui_cli
```

Follow the prompts to calibrate Energy, Uniformity, and Genre. After generation, you can choose A, B, or Neither to refine.

## Testing
Run the verification suite:
```bash
python -m unittest discover tests
```
