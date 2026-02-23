import cmd
from .generator import generator
from .agent import ReActAgent
from .ab_test import ab_tester
from .library import library
from .exporter import exporter
from .spotify_export import spotify_exporter
from .config import config
from .llm.providers.local import MockLLM
import time
from rich.console import Console
from rich.table import Table

console = Console()

def typewriter_print(text: str, delay: float = 0.01):
    """Prints text one character at a time for better UX."""
    print("AI: ", end="", flush=True)
    # Split by lines to preserve structure nicely
    lines = text.split('\n')
    for i, line in enumerate(lines):
        line_content = line.strip()
        if not line_content and i > 0: continue # Skip empty lines inside
        
        for char in line:
            print(char, end="", flush=True)
            time.sleep(delay)
        if i < len(lines) - 1:
            print('\n    ', end="", flush=True) # Indent wrapped lines slightly
    print("\n")

class MixtapeCLI(cmd.Cmd):
    prompt = '(mixtape) '

    def __init__(self):
        super().__init__()
        self.intro = 'Welcome to the ReAct Mixtape Curator. Type "start" to begin.'

    def _display_playlist(self, playlist, label: str):
        """Display a playlist using Rich styled tables."""
        score = playlist.scores.total
        # Color the score: green if >= 0.80, yellow if >= 0.60, red otherwise
        if score >= 0.80:
            score_style = "bold green"
        elif score >= 0.60:
            score_style = "bold yellow"
        else:
            score_style = "bold red"
        
        table = Table(
            title=f"Playlist {label}",
            title_style="bold cyan",
            caption=f"Score: [{score_style}]{score:.2f}[/{score_style}] | "
                    f"{len(playlist.track_ids)} tracks | "
                    f"{playlist.total_duration_s // 60}m {playlist.total_duration_s % 60}s",
            show_lines=False,
            pad_edge=True,
        )
        
        table.add_column("#", style="dim", width=3, justify="right")
        table.add_column("Artist", style="cyan", max_width=22)
        table.add_column("Title", max_width=30)
        table.add_column("Dur", style="dim", width=5, justify="right")
        table.add_column("Why it fits", style="italic", max_width=50)
        
        for i, tid in enumerate(playlist.track_ids):
            t = library.get_track(tid)
            if t:
                dur_str = f"{t.duration_s // 60}:{t.duration_s % 60:02d}"
                note = playlist.track_notes.get(tid, "Fits the curated journey.")
                table.add_row(
                    str(i + 1),
                    t.artist[:22],
                    t.title[:30],
                    dur_str,
                    note[:50]
                )
        
        console.print()
        console.print(table)

    def do_start(self, arg):
        """Start the interview process with the new Agentic Interviewer."""
        from .interview_agent import InterviewAgent
        
        # Transparency callback
        def show_thought(thought: str):
            print(f"\n{thought}")
            
        agent = InterviewAgent()  # New conversational agent
        print(f"AI: {agent.start()}\n")
        
        while not agent.completed:
            try:
                user_input = input("> ")
                response, done = agent.process_input(user_input, user_callback=show_thought)
                print()  # Space after user input
                typewriter_print(response)
                if done:
                    break
            except EOFError:
                return True
            
        if agent.completed:
            self._run_generation(agent)


    def _run_generation(self, interviewer):
        profile = interviewer.profile
        retry_count: int = 0
        max_retries = 3

        while retry_count < max_retries:
            print(f"\n--- Round {retry_count + 1} / {max_retries + 1} ---")
            print("Seeding playlist with must-haves...")
            # 1. Draft (Seed Phase)
            # Use segmented generation if eclectic + multi-genre (disable incremental)
            use_incremental = True
            if profile.targets.uniformity < 0.6 and len(profile.target_genres) > 1:
                 use_incremental = False
                 print(f"Triggering Eco-Modular Generation for {len(profile.target_genres)} segments...")
            
            draft = generator.create_draft(profile, incremental=use_incremental)
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
                from .llm.providers.openai import OpenAIProvider
                llm_provider = OpenAIProvider()
            
            agent = ReActAgent(llm=llm_provider, user_callback=ask_user)
            playlist_a = agent.repair_playlist(sequenced, profile)
            
            print(f"Playlist A Finalized: {len(playlist_a.track_ids)} tracks.")
            
            # 4. A/B Generation
            print("Generating B-Side Variation...")
            playlist_b = ab_tester.generate_b_side(playlist_a, profile)
            
            console.print("\n[bold green]=== GENERATION COMPLETE ===[/bold green]")
            
            self._display_playlist(playlist_a, "A")
            self._display_playlist(playlist_b, "B")

            
            # 5. Selection Loop
            print("\n(Export features coming in next phase)")
            choice = input("\nWhich mix do you prefer? [A / B / Neither]: ").strip().upper()
            
            if choice == 'A':
                print("\nSelected Playlist A!")
                self._export(playlist_a, profile, "A")
                return
            elif choice == 'B':
                print("\nSelected Playlist B!")
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
