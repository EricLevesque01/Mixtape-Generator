import json
import logging
import time
from typing import Dict, Any
from .models import UserProfile
from .generator import generator
from .agent import ReActAgent
from .llm.providers.local import MockLLM
from .library import library

logger = logging.getLogger("mixtape_curator")

class EvalHarness:
    def __init__(self, output_file: str = "eval_results.jsonl"):
        self.output_file = output_file

    def run_suite(self, cases_file: str = "data/eval_cases.jsonl"):
        """Run a suite of test cases."""
        logger.info(f"Starting evaluation suite from {cases_file}")
        
        # Load cases
        with open(cases_file, "r") as f:
            cases = [json.loads(line) for line in f if line.strip()]

        results = []
        for case in cases:
            res = self._run_case(case)
            results.append(res)
            
        # Write results
        with open(self.output_file, "w") as f:
            for res in results:
                f.write(json.dumps(res) + "\n")
                
        logger.info(f"Evaluation complete. Results written to {self.output_file}")
        return results

    def _run_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case."""
        case_id = case.get("id")
        desc = case.get("description")
        
        # Build Profile from case data
        profile = UserProfile()
        profile.recipient = case.get("recipient", "Test User")
        profile.target_genres = case.get("genres", [])
        
        # Set targets manually if provided
        targets = case.get("targets", {})
        if targets:
            profile.targets.energy = targets.get("energy", 0.5)
            profile.targets.valence = targets.get("valence", 0.5)
            profile.targets.intensity = targets.get("intensity", 0.5)
            # profile.targets.tempo = targets.get("tempo", 0.5) # Not in model
            # profile.targets.popularity = targets.get("popularity", 0.5) # Not in model
            profile.targets.uniformity = targets.get("uniformity", 0.5) # Should align with spec v2.1.1
            
        start_time = time.time()
        
        # 1. Generate Draft
        draft = generator.create_draft(profile)
        
        # 2. Sequence
        sequenced = generator.optimize_flow(draft, profile)
        
        # 3. Repair (Agent)
        agent = ReActAgent(llm=MockLLM())
        final_pl = agent.repair_playlist(sequenced, profile)
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Calculate Metrics
        final_score = final_pl.scores.total
        # Check against expected spec constraints (e.g. uniformity)
        # Spec v2.1.1: If user wants uniformity > 0.8, variety score should be high if uniform.
        
        log_entry = {
            "case_id": case_id,
            "description": desc,
            "profile_genres": profile.target_genres,
            "playlist_duration_s": final_pl.total_duration_s,
            "track_count": len(final_pl.track_ids),
            "final_score": final_score,
            "processing_time_ms": duration_ms,
            "violations": final_pl.violations
        }
        
        return log_entry

harness = EvalHarness()

if __name__ == "__main__":
    from .library import library
    library.load() # Ensure data loaded
    
    harness.run_suite("data/eval_cases.jsonl")
