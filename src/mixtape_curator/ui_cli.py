import sys
import cmd
from .interview import Interviewer
from .generator import generator
from .agent import ReActAgent, MockLLM
from .ab_test import ab_tester
from .library import library
from .exporter import exporter
from .spotify_export import spotify_exporter
from .config import config

class MixtapeCLI(cmd.Cmd):
    intro = 'Welcome to the ReAct Mixtape Curator. Type "start" to begin.'
    prompt = '(mixtape) '

    def do_start(self, arg):
        """Start the interview process with the new Agentic Interviewer."""
        from .interview_agent import InterviewAgent
        
        # Transparency callback
        def show_thought(thought: str):
            print(f"\n{thought}")
            
        agent = InterviewAgent() # New conversational agent
        print(agent.start())
        
        while not agent.completed:
            try:
                user_input = input("> ")
                response, done = agent.process_input(user_input, user_callback=show_thought)
                print(f"\nAI: {response}\n")
                if done:
                    break
            except EOFError:
                return True
            
        if agent.completed:
            self._run_generation(agent)

    def _run_generation(self, interviewer):
        profile = interviewer.profile
        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            print(f"\n--- Round {retry_count + 1} / {max_retries + 1} ---")
            print("Drafting playlist...")
            # 1. Draft
            draft = generator.create_draft(profile)
            print(f"Draft Initialized: {len(draft.track_ids)} tracks, {draft.total_duration_s}s duration.")
            
            # 2. Sequence
            print("Optimizing Flow...")
            sequenced = generator.optimize_flow(draft, profile)
            
            # 3. Agent Repair
            print("Agent Reviewing Constraints...")
            
            # Callback for agent to ask user
            def ask_user(question):
                print(f"\n[Agent Question]: {question}")
                return input("> ")
    
            # TODO: wire real LLM here later
            agent = ReActAgent(llm=MockLLM(), user_callback=ask_user)
            playlist_a = agent.repair_playlist(sequenced, profile)
            
            print(f"Playlist A Finalized: {len(playlist_a.track_ids)} tracks.")
            
            # 4. A/B Generation
            print("Generating B-Side Variation...")
            playlist_b = ab_tester.generate_b_side(playlist_a, profile)
            
            print("\n=== GENERATION COMPLETE ===")
            print(f"Playlist A: {len(playlist_a.track_ids)} tracks ({playlist_a.total_duration_s}s) (Score: {playlist_a.scores.total:.2f})")
            # print specific tracks with new audio features
            for tid in playlist_a.track_ids:
                t = library.get_track(tid)
                if t:
                    key = getattr(t, 'key_full', 'Unknown Key')
                    energy = getattr(t, 'energy', 0.0)
                    print(f"  - {t.title} ({t.artist}) [{key}] [E:{energy:.2f}]")
            
            print(f"\nPlaylist B: {len(playlist_b.track_ids)} tracks ({playlist_b.total_duration_s}s) (Score: {playlist_b.scores.total:.2f})")
            for tid in playlist_b.track_ids:
                t = library.get_track(tid)
                if t:
                    key = getattr(t, 'key_full', 'Unknown Key')
                    energy = getattr(t, 'energy', 0.0)
                    print(f"  - {t.title} ({t.artist}) [{key}] [E:{energy:.2f}]")
            
            # 5. Selection Loop
            print(f"\n(Export features coming in next phase)")
            choice = input("\nWhich mix do you prefer? [A / B / Neither]: ").strip().upper()
            
            if choice == 'A':
                print(f"\nSelected Playlist A!")
                self._export(playlist_a, profile, "A")
                return
            elif choice == 'B':
                print(f"\nSelected Playlist B!")
                self._export(playlist_b, profile, "B")
                return
                return
            elif choice == 'NEITHER':
                retry_count += 1
                if retry_count < max_retries:
                   feedback = input("What would you like to change? (e.g., 'too slow', 'too eclectic', 'exclude Taylor Swift'): ")
                   interviewer.refine_profile(feedback)
                   # Loop continues with updated profile
                else:
                    print("Max retries reached. Exporting A by default.")
                    self._export(playlist_a, profile, "Final")
                    return
            else:
                 print("Invalid choice. defaulting to A.")
                 self._export(playlist_a, profile, "A")
                 return

    def _export(self, playlist, profile, suffix):
        print("\n=== WRITING FILES ===")
        base = f"mixtape_for_{profile.recipient.replace(' ', '_')}_{suffix}"
        
        # 1. Local Export
        print(exporter.export_playlist(playlist, base))
        
        # 2. Spotify Export (Optional)
        if config.spotify_client_id and config.spotify_client_secret:
            choice = input("Export to Spotify? (y/N): ").strip().lower()
            if choice == 'y':
                print("Connecting to Spotify...")
                result = spotify_exporter.export_playlist(playlist, base.replace('_', ' '))
                print(result)
        else:
            print("(Spotify export skipped: credentials not set)")

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
