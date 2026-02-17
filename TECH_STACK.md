# Technology Stack

Based on the requirements in `SPECIFICATION.md`, the following technology stack is selected to prioritize rapid development, agentic capabilities, and academic demonstration of the ReAct framework.

## 1. Core Runtime
*   **Language:** **Python 3.10+**
    *   *Justification:* The de facto standard for AI/LLM development. Excellent support for data manipulation (Pandas) and agent frameworks.

## 2. Agent Framework & Logic
*   **Orchestration:** **LangGraph**
    *   *Justification:* The specification (Section 5.6) explicitly describes a cyclic "Reason -> Act -> Observe -> Repair" loop. LangGraph is specifically designed for stateful, cyclic agent workflows, making it a better fit than linear chains.
*   **LLM Interface:** **LangChain**
    *   *Justification:* Provides standard abstractions for prompt templates and tool binding.
*   **Model Provider:** **OpenAI API (GPT-4o)** or **Anthropic API (Claude 3.5 Sonnet)**
    *   *Justification:* High reasoning capability is required for the "Tape Viability Validator" and flow optimization logic.

## 3. Data & Storage
*   **Data Processing:** **Pandas**
    *   *Justification:* The library contains ~5,000 songs. Pandas is highly efficient for filtering, sorting, and scoring this volume of data in-memory based on metadata (energy, valence, etc.).
*   **Data Store:** **Local JSON / CSV**
    *   *Justification:* As per Spec Section 7 (Component 1), a local file store is sufficient and simplifies architecture without needing a dedicated SQL/NoSQL database.

## 4. User Interface
*   **Framework:** **Chainlit**
    *   *Justification:* Specifies a "lightweight web chat" (Section 7, Component 5). Chainlit provides a production-ready chat interface out-of-the-box with built-in support for displaying "Thought/Action" steps, which is crucial for demonstrating the ReAct loop to the academic instructor (Section 4).

## 5. Development Tools
*   **Environment Management:** `venv` or `poetry`
*   **Linting/Formatting:** `ruff` (fast Python linter/formatter)
*   **Version Control:** Git + GitHub

## 6. Architecture Map
```mermaid
graph TD
    User[User via Chainlit UI] <--> Agent[LangGraph Agent]
    Agent --> LLM[LLM (GPT-4/Claude)]
    Agent --> Tools[Tool Layer]
    Tools --> Pandas[Pandas DataFrame]
    Pandas --> Data[library.json]
    Tools --> Validator[Scoring Engines]
```
