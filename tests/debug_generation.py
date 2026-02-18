import sys
import logging
import os

# Add src to path
sys.path.append(os.path.abspath('src'))

from mixtape_curator.models import UserProfile
from mixtape_curator.library import library
from mixtape_curator.generator import generator
from mixtape_curator.agent import ReActAgent, MockLLM
from mixtape_curator.ab_test import ab_tester

# Setup logging
logging.basicConfig(level=logging.INFO)

def run_debug():
    # Load library
    print("Loading library...")
    library.load()
    print(f"Library loaded with {len(library.df)} tracks.")

    # Create profile manually (simulating interview result)
    profile = UserProfile()
    profile.recipient = "My Best Friend"
    profile.target_genres = ["Synthpop", "80s"]
    profile.targets.uniformity = 0.8
    profile.targets.energy = 0.8
    profile.targets.intensity = 0.7

    print("Starting generation...")

    # 1. Draft
    print("Creating draft...")
    draft = generator.create_draft(profile)
    print(f"Draft Initialized: {len(draft.track_ids)} tracks, {draft.total_duration_s}s duration.")

    # 2. Sequence
    print("Optimizing Flow...")
    sequenced = generator.optimize_flow(draft, profile)
    print("Flow optimized.")

    # 3. Agent Repair
    print("Agent Reviewing Constraints...")
    agent = ReActAgent(llm=MockLLM())
    playlist_a = agent.repair_playlist(sequenced, profile)
    print(f"Playlist A Finalized: {len(playlist_a.track_ids)} tracks.")

    # 4. A/B Generation
    print("Generating B-Side Variation...")
    playlist_b = ab_tester.generate_b_side(playlist_a, profile)
    print("B-Side generated.")

    print("Done.")

if __name__ == "__main__":
    run_debug()
