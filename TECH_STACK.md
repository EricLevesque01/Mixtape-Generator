# Technology Stack — ReAct Mixtape Curator v2.1.1

Based on the [Specification v2.1.1](SPECIFICATION.md), the following technology stack is selected to prioritize **deterministic control**, **semantic depth**, and **comparative evaluation**.

## 1. Core Runtime
*   **Language:** **Python 3.10+** (Required for modern type hinting and library support).
*   **Dependency Management:** `poetry` (Recommended) or `venv` + `pip`.
*   **Linting/Formatting:** `ruff` (Fast, comprehensive compliance).

## 2. Agent Framework & Logic
*   **Orchestration:** **LangGraph** (or Custom State Machine)
    *   *Justification:* The "ReAct Repair Loop" (Spec §8) and "Base-and-Branch" (Spec §9) strategies require a stateful, cyclic execution graph. LangGraph provides the ideal primitive for this `Draft -> Validate -> Repair -> Repeat` cycle.
*   **LLM Interface:** **Custom Abstraction Layer** (`src/mixtape_curator/llm/`)
    *   *Justification:* Spec §14 requires benchmarking multiple models. We cannot rely on a single vendor SDK. We will implement a lightweight adapter pattern supporting:
        *   **OpenAI** (GPT-4o)
        *   **Anthropic** (Claude 3.5 Sonnet)
        *   **Local/Open-Weights** (via Llama.cpp/Ollama standard endpoints)

## 3. Data Engineering & Scoring
*   **Data Processing:** **Pandas**
    *   *Justification:* High-performance filtering and vector operations for the 5k+ track library. Essential for computing the **Entropy-based Variety Score** (Spec §3.3) and **Gaussian Decay Fit Score** (Spec §3.1) efficiently across thousands of candidates.
*   **Validation:** **Pydantic**
    *   *Justification:* Strict schema enforcement for `Track`, `Profile`, and `Playlist` objects (Spec §13). Ensures data integrity before it reaches the LLM or export layer.
*   **Taxonomy:** **RateYourMusic (RYM)**
    *   *Justification:* Spec §2 requires deep semantic genre modeling (Primary/Subgenres/Descriptors). This data is ingested into `library.json` offline.

## 4. User Interface
*   **Framework:** **CLI (Standard Library `cmd` or `argparse`)**
    *   *Justification:* Spec §18 explicit mandates a CLI. This is critical for the **Evaluation Harness**, which acts as a "Simulated User" piping text inputs into the interface programmatically.
*   **Output Formatting:** **Rich**
    *   *Justification:* Provides beautiful terminal output (tables, progress bars, colored diffs) to make the text-based interface feel polished and "curated".

## 5. Integrations (Offline-First)
*   **Spotify:** **Spotipy**
    *   *Justification:* Used **ONLY** for the final export step (Spec §11).
    *   *Constraint:* No runtime API reliance. URIs are resolved via an offline lookup table during library enrichment.

## 6. Architecture Map

```mermaid
graph TD
    User[User / Sim User] <--> CLI[CLI Interface (Rich)]
    CLI --> Agent[LangGraph ReAct Loop]
    Agent --> Abstraction[LLM Abstraction Layer]
    Abstraction --> Models[OpenAI / Anthropic / Local]
    
    Agent --> Tools[Tool Layer]
    Tools --> Logic[Generator & Scoring]
    Logic --> Pandas[Pandas DataFrame]
    Pandas --> Data[(library.json)]
    
    Logic --> Validator[Constraint Validator]
    
    Agent --> Export[Export Module]
    Export --> Files[TXT/M3U8]
    Export --> Spotify[Spotipy API]
```
