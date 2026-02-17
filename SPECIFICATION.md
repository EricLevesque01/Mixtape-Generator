# Product Specification
## Project: ReAct Mixtape Curator
**Version:** v1.0 (Class Project Scope)

## 1. Product Overview
ReAct Mixtape Curator is a conversational AI application that interviews a user and generates a curated 120-minute mixtape from the user’s personal music library.

The system behaves like a thoughtful mixtape maker: it asks context questions, reasons about mood and audience, constructs a playlist, evaluates its quality, and iteratively improves it until it meets defined viability standards.

The application demonstrates an implementation of the ReAct (Reason + Act) framework using structured tool calls and external scoring functions.

## 2. Goals
**Primary Goal**
Create an AI agent that can:
* Interview a user about mixtape intent
* Generate a playlist from a personal library
* Optimize sequencing and flow
* Ensure playlist viability using constraints and scoring
* Output a final curated mixtape

**Academic Goal**
Demonstrate:
* Agentic reasoning
* Tool use via ReAct framework
* Constraint-based generation
* Iterative improvement loops
* Structured evaluation metrics

## 3. Non-Goals
Out of scope for v1:
* Real-time streaming platform integration
* Social playlist sharing
* Multi-user recommendation modeling
* Audio playback
* Music discovery outside the personal library
* Production-scale scalability

## 4. Target Users
**Primary User**
The library owner creating tapes for:
* personal listening
* friends
* romantic interests
* parties
* travel
* study or work

**Secondary User (Academic)**
Instructor evaluating:
* agent design
* reasoning process
* system architecture
* reproducibility

## 5. Core Features
### 5.1 Conversational Tape Interview
Agent asks targeted questions such as:
* Who is the tape for?
* When will it be played?
* Desired mood?
* Familiar vs discovery?
* Artists to include/exclude?
* Energy arc preference?

Agent gathers constraints before building.

### 5.2 Personal Library Search
System searches ~5,000 songs using metadata:
* genres
* moods
* energy
* valence
* agreeableness
* pretentiousness
* personal ratings
* play history

### 5.3 Tape Construction Engine
System selects tracks to:
* satisfy constraints
* maintain duration ≤120 min
* limit artist repetition
* match requested vibe
* produce coherent sequencing

### 5.4 Flow Optimization
Songs are ordered to minimize harsh transitions using:
* energy continuity
* emotional continuity
* genre proximity
* intensity continuity

### 5.5 Tape Viability Validator
Playlist must pass viability checks:

**Hard constraints:**
* Duration ≤120 min
* Respect user exclusions
* Artist repetition limits

**Soft constraints:**
* Mood/genre fit
* Flow smoothness
* Arc consistency
* Accessibility match
* Variety balance

Only viable tapes are finalized.

### 5.6 ReAct Agent Loop
Agent repeatedly:
* Reasons about next step
* Calls a tool
* Observes results
* Updates plan
* Repairs playlist if needed

Cycle repeats until viable.

### 5.7 Final Output
System outputs:
* Ordered tracklist
* Duration per track
* Total runtime
* Optional liner notes
* Score summary

## 6. Metadata Requirements
Tracks contain:

**Required**
* Title
* Artist
* Album
* Duration
* Track position
* Year

**Personal signals**
* Personal rating
* Play count
* Recommendation likelihood

**Musical features**
* Genres
* Moods
* Energy
* Valence
* Intensity
* Social accessibility
* Agreeableness
* Pretentiousness
* Familiarity

## 7. Architecture Overview
**Components**

1. **Library Data Store**
   Local metadata JSON.

2. **Tool Layer**
   Functions callable by agent:
   * ask user
   * filter library
   * build draft
   * order tape
   * validate tape
   * repair tape

3. **Agent Engine**
   LLM chooses next action.

4. **Validator**
   Computes scores and constraints.

5. **UI Layer**
   CLI or lightweight web chat.

## 8. System Flow
User → Agent Interview → Library Search → Draft Tape → Validate → Repair → Finalize → Output

## 9. User Experience Flow
1. User opens app.
2. Agent asks purpose questions.
3. Agent builds draft.
4. Agent repairs playlist if needed.
5. Final tape presented.
6. User optionally reruns or tweaks constraints.

## 10. Success Metrics
Project success measured by:

**Functional:**
* Agent produces ≤120-minute tapes
* System respects constraints
* Agent repairs invalid drafts

**Qualitative:**
* Tape flow feels coherent
* Playlist matches requested vibe

**Academic:**
* Demonstrates ReAct loop
* Tool use visible
* Iterative improvement shown

## 11. Risks and Mitigation
| Risk | Mitigation |
| :--- | :--- |
| Poor metadata quality | Hybrid tagging + manual correction |
| Agent loops excessively | Max iteration cap |
| Flow scoring too strict | Adjustable thresholds |
| Slow tagging process | Tag popular tracks first |

## 12. Future Extensions (Post-Class)
Potential improvements:
* Automatic audio feature extraction
* Tape cover art generation
* Playlist export integrations
* Personalized recommendation learning
* Multi-user tapes
* Genre embedding models
* Long-term taste evolution modeling

## 13. MVP Definition
Minimum viable product:
* Library loaded
* Agent asks questions
* Playlist constructed
* Validator enforces duration + repetition
* Ordered tape output

Everything else is enhancement.
