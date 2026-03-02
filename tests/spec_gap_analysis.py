"""Gap analysis: spec vs implementation."""
import sys
import os
import inspect
sys.path.insert(0, 'src')

from mixtape_curator.library import library
from mixtape_curator.generator import generator
from mixtape_curator.agent import ReActAgent
from mixtape_curator.config import config
from mixtape_curator.scoring import scorer

library.load()

OK = "✅"
MISS = "❌"
PART = "⚠️ "

print("=== SPEC ALIGNMENT AUDIT ===\n")

# Section 1: Config
print("--- Section 1: Required Config Keys ---")
required_keys = [
    'duration_target_s','duration_cap_s','max_tracks_per_artist','max_questions',
    'conf_thresh','accept_threshold','max_repair_iters','eps','stall_iters',
    'off_topic_max','max_ab_rounds','w_fit','w_flow','w_variety','w_access',
    'w_quality','branch_swap_min','branch_swap_max','branch_variety_boost','rng_seed'
]
for k in required_keys:
    val = config.get(k, 'MISSING')
    status = OK if val != 'MISSING' else MISS
    print(f"  {status} {k} = {val}")

# Section 2: Hard Constraints
print("\n--- Section 2: Hard Constraints ---")
print(f"  {OK} duration_cap_s enforced in agent._validate_constraints")
print(f"  {OK} max_tracks_per_artist enforced in agent._validate_constraints")
print(f"  {PART} excluded artists/genres: library.filter_candidates checks exclusions but not all profile fields")
print(f"  {MISS} must_include_artists not enforced in generator.create_draft")

# Section 3: Scoring
print("\n--- Section 3: Scoring Math ---")
scorer_src = inspect.getsource(scorer.compute_fit_score)
print(f"  {OK} Sonic fit: Gaussian decay exp(-d^2 / 2*tol^2)")
print(f"  {OK} Genre fit: primary or subgenre match = full credit")
print(f"  {OK} Blend: sonic*(1-strictness) + genre*strictness")
print(f"  {OK} Flow score: energy/valence/intensity delta minimization")
print(f"  {OK} Variety score: entropy-based uniformity alignment")
print(f"  {MISS} Accessibility score: field exists but not computed from real data")
print(f"  {MISS} Quality score: field exists but not computed from real data")

# Section 4: Interview
print("\n--- Section 4: Interview ---")
print(f"  {OK} Stage 1: Recipient/context intake")
print(f"  {OK} Stage 2: Genre & descriptor calibration")
print(f"  {OK} Stage 3: Cohesion calibration")
print(f"  {OK} Stage 4: Energy & mood calibration")
print(f"  {OK} Stage 5: Confirmation")
print(f"  {MISS} Confidence aggregation per axis (conf_thresh not used)")
print(f"  {MISS} max_questions limit not enforced (hardcoded 5 questions)")
print(f"  {MISS} Dynamic follow-up questions based on confidence")
print(f"  {MISS} off_topic_max guardrail not implemented")

# Section 7: Flow Optimization
print("\n--- Section 7: Flow Optimization ---")
gen_src = inspect.getsource(generator.optimize_flow)
print(f"  {PART} Greedy multi-start: only 1 seed tried (not multi-start)")
print(f"  {OK} Iterative best-transition append")
print(f"  {OK} Best ordering kept")

# Section 8: ReAct Tools
print("\n--- Section 8: ReAct Agent Tools ---")
spec_tools = ['search_library', 'swap_track', 'consult_user', 'finalize']
agent_src = inspect.getsource(ReActAgent)
for t in spec_tools:
    present = t in agent_src
    status = OK if present else MISS
    print(f"  {status} {t}")
print(f"  {PART} remove_track: implemented but not in spec")
print(f"  {PART} add_track: implemented but not in spec")
print(f"  {MISS} MockLLM uses hardcoded logic — real LLM not wired")

# Section 9: A/B
print("\n--- Section 9: A/B Generation ---")
print(f"  {OK} swap_count = clamp(round(0.2*n), min, max)")
print(f"  {OK} must_includes never replaced")
print(f"  {OK} resequence after swap")
print(f"  {PART} weakest scoring dimension: uses lowest fit score (not full dimension analysis)")
print(f"  {OK} No duplicate artists in B-side (fixed)")

# Section 10: Neither Retry
print("\n--- Section 10: Neither Retry Loop ---")
print(f"  {MISS} Neither/retry loop not in ui_cli.py (max_ab_rounds unused)")
print(f"  {MISS} Feedback collection not implemented")
print(f"  {MISS} Profile update on retry not implemented")

# Section 11: Exports
print("\n--- Section 11: Exports ---")
print(f"  {OK} TXT export")
print(f"  {OK} M3U8 export")
print(f"  {MISS} CSV export")
print(f"  {MISS} Spotify sync (spotify_export.py missing)")

# Section 14: Eval Harness
print("\n--- Section 14: Eval Harness ---")
print(f"  {OK if os.path.exists('src/mixtape_curator/eval_harness.py') else MISS} eval_harness.py")
print(f"  {OK if os.path.exists('data/eval_cases.jsonl') else MISS} data/eval_cases.jsonl")
print(f"  {OK if os.path.exists('data/feedback.jsonl') else MISS} data/feedback.jsonl")
print(f"  {OK if os.path.exists('data/eval_results.jsonl') else MISS} data/eval_results.jsonl")

# Section 15: Repo structure
print("\n--- Section 15: Repo Structure ---")
expected = [
    'config.yaml', 'data/library.json', 'src/mixtape_curator/models.py',
    'src/mixtape_curator/library.py', 'src/mixtape_curator/interview.py',
    'src/mixtape_curator/scoring.py', 'src/mixtape_curator/generator.py',
    'src/mixtape_curator/ab_test.py', 'src/mixtape_curator/ui_cli.py',
]
for path in expected:
    status = OK if os.path.exists(path) else MISS
    print(f"  {status} {path}")

missing_spec = [
    'src/mixtape_curator/export.py',
    'src/mixtape_curator/spotify_export.py',
    'src/mixtape_curator/llm/providers/openai.py',
    'src/mixtape_curator/llm/providers/anthropic.py',
    'data/feedback.jsonl',
    'data/eval_cases.jsonl',
]
for path in missing_spec:
    status = OK if os.path.exists(path) else MISS
    print(f"  {status} {path} (spec required)")

print("\n=== SUMMARY ===")
print("IMPLEMENTED: Core pipeline (interview→draft→flow→agent→A/B→export)")
print("PARTIAL:     Flow multi-start, weakest-dimension swap, exclusion enforcement")
print("MISSING:     Confidence-based interview, Neither loop, Spotify, CSV, eval harness, real LLM")
