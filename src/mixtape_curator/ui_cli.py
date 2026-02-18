import sys
import cmd
from .interview import Interviewer
from .generator import generator
from .agent import ReActAgent, MockLLM
from .ab_test import ab_tester
from .library import library
from .exporter import exporter
from .config import config

class MixtapeCLI(cmd.Cmd):
    intro = 'Welcome to the ReAct Mixtape Curator. Type "start" to begin.'
    prompt = '(mixtape) '

    def do_start(self, arg):
        """Start the interview process."""
        interviewer = Interviewer() # New instance per chat
        print(interviewer.start())
        
        while not interviewer.completed:
            try:
                user_input = input("> ")
                response, done = interviewer.process_input(user_input)
                print(f"\nAI: {response}\n")
                if done:
                    break
            except EOFError:
                return True
            
        if interviewer.completed:
            self._run_generation(interviewer.profile)

    def _run_generation(self, profile):
        print("Drafting playlist...")
        # 1. Draft
        draft = generator.create_draft(profile)
        print(f"Draft Initialized: {len(draft.track_ids)} tracks, {draft.total_duration_s}s duration.")
        
        # 2. Sequence
        print("Optimizing Flow...")
        sequenced = generator.optimize_flow(draft, profile)
        
        # 3. Agent Repair
        print("Agent Reviewing Constraints...")
        # TODO: wire real LLM here later
        agent = ReActAgent(llm=MockLLM())
        playlist_a = agent.repair_playlist(sequenced, profile)
        
        print(f"Playlist A Finalized: {len(playlist_a.track_ids)} tracks.")
        
        # 4. A/B Generation
        print("Generating B-Side Variation...")
        playlist_b = ab_tester.generate_b_side(playlist_a, profile)
        
        print("\n=== GENERATION COMPLETE ===")
        print(f"Playlist A: {len(playlist_a.track_ids)} tracks (Score: {playlist_a.scores.total})")
        # print specific tracks
        for tid in playlist_a.track_ids:
            t = library.get_track(tid)
            if t:
                print(f"  - {t.title} ({t.artist})")
        
        print(f"\nPlaylist B: {len(playlist_b.track_ids)} tracks (Score: {playlist_b.scores.total})")
        for tid in playlist_b.track_ids:
            t = library.get_track(tid)
            if t:
                print(f"  - {t.title} ({t.artist})")
        
        print(f"\n(Export features coming in next phase)")
        
        # 5. Export
        print("\n=== WRITING FILES ===")
        # Simple file name generation from profile recipient
        base_a = f"mixtape_for_{profile.recipient.replace(' ', '_')}_A"
        base_b = f"mixtape_for_{profile.recipient.replace(' ', '_')}_B"
        
        print(exporter.export_playlist(playlist_a, base_a))
        print(exporter.export_playlist(playlist_b, base_b))

    def do_quit(self, arg):
        """Exit the program."""
        return True

    def do_EOF(self, arg):
        """Handle EOF to exit cleanly."""
        print()
        return True

if __name__ == '__main__':
    # Load data first
    try:
        library.load()
        MixtapeCLI().cmdloop()
    except Exception as e:
        print(f"Startup Error: {e}")
