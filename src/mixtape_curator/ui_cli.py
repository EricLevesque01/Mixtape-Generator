import sys
import cmd
from mixtape_curator.generator import generator
from mixtape_curator.agent import ReActAgent
from mixtape_curator.ab_test import ab_tester
from mixtape_curator.library import library
from mixtape_curator.exporter import exporter
from mixtape_curator.spotify_export import spotify_exporter
from mixtape_curator.config import config
from mixtape_curator.llm.providers.local import MockLLM

class MixtapeCLI(cmd.Cmd):
    intro = 'Welcome to the ReAct Mixtape Curator. Type "start" to begin.'
    prompt = '(mixtape) '

    def do_start(self, arg):
        """Start the interview process with the new Agentic Interviewer."""
        from mixtape_curator.interview_agent import InterviewAgent
        
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
            print("Seeding playlist with must-haves...")
            # 1. Draft (Seed Phase)
            draft = generator.create_draft(profile, incremental=True)
            print(f"Seed Created: {len(draft.track_ids)} tracks.")
            
            # 2. Sequence (Initial)
            print("Optimizing Initial Flow...")
            sequenced = generator.optimize_flow(draft, profile)
            
            # 3. Agent Repair
            print("Agent Reviewing Constraints...")
            
            # Callback for agent to ask user
            def ask_user(question):
                print(f"\n[Agent Question]: {question}")
                return input("> ")
    
            # Use real LLM provider if key is available
            llm_provider = MockLLM()
            if hasattr(config, "openai_api_key") and config.openai_api_key:
                from mixtape_curator.llm.providers.openai import OpenAIProvider
                llm_provider = OpenAIProvider()
            
            agent = ReActAgent(llm=llm_provider, user_callback=ask_user)
            playlist_a = agent.repair_playlist(sequenced, profile)
            
            print(f"Playlist A Finalized: {len(playlist_a.track_ids)} tracks.")
            
            # 4. A/B Generation
            print("Generating B-Side Variation...")
            playlist_b = ab_tester.generate_b_side(playlist_a, profile)
            
            print("\n=== GENERATION COMPLETE ===")
            headers = f"{'Track':<5} | {'Artist':<20} | {'Song Title':<30} | {'Vibe / Why it fits'}"
            separator = "-" * len(headers)
            
            print(f"\n### Playlist A (Score: {playlist_a.scores.total:.2f})")
            print(headers)
            print(separator)
            for i, tid in enumerate(playlist_a.track_ids):
                t = library.get_track(tid)
                if t:
                    note = playlist_a.track_notes.get(tid, "Fits the curated journey.")
                    print(f"{i+1:<5} | {t.artist[:20]:<20} | {t.title[:30]:<30} | {note}")
            
            print(f"\n### Playlist B (Score: {playlist_b.scores.total:.2f})")
            print(headers)
            print(separator)
            for i, tid in enumerate(playlist_b.track_ids):
                t = library.get_track(tid)
                if t:
                    note = playlist_b.track_notes.get(tid, "Fits the curated journey.")
                    print(f"{i+1:<5} | {t.artist[:20]:<20} | {t.title[:30]:<30} | {note}")
            
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
