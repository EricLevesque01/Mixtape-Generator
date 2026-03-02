import cmd
from .generator import generator
from .agent import ReActAgent
from .blueprint import build_blueprint_generator
from .trellis import build_trellis
from .arc_optimizer import ArcOptimizer, arc_draft_to_playlist
from .diagnostics import RelaxationMenu
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
        """Start the persona-based interview to build a mixtape from Eric's library."""
        from .interview import PersonaInterviewer

        llm_provider = self._get_llm_provider()
        cheap_llm    = self._get_cheap_llm_provider()

        agent = PersonaInterviewer(cheap_llm=cheap_llm, main_llm=llm_provider)
        print()
        typewriter_print(agent.start())

        while not agent.completed:
            try:
                user_input = input("> ")
                if not user_input.strip():
                    continue
                response, done = agent.process_input(user_input)
                print()
                typewriter_print(response)
                if done:
                    break
            except EOFError:
                return True

        if agent.completed:
            self._run_generation(agent)



    def _run_generation(self, interviewer):
        """V5 pipeline: Interview → Blueprint → Trellis → ArcOptimizer → Refinement."""
        profile = interviewer.profile
        # Convert PersonaProfile -> UserProfile if needed
        if hasattr(profile, 'to_user_profile'):
            persona = profile
            profile = profile.to_user_profile()
        else:
            persona = None

        # Cap duration at 1h20m (4800s)
        MAX_DURATION_S = 4800
        duration_s = min(profile.duration_target_s or config.duration_target_s, MAX_DURATION_S)

        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            console.print()
            console.print("[dim]Building your mixtape...[/dim]")

            # ----------------------------------------------------------------
            # PHASE B — Blueprint (LLM Creative Director)
            # ----------------------------------------------------------------
            llm_provider = self._get_llm_provider()
            bp_gen = build_blueprint_generator(llm_provider)
            persona_obj = persona or profile
            blueprint = bp_gen.generate(
                persona           = persona_obj if hasattr(persona_obj, 'confidence') else _FakePersona(profile),
                duration_target_s = duration_s,
            )

            # ----------------------------------------------------------------
            # PHASE C — Segment Trellis (Deterministic Algorithm)
            # ----------------------------------------------------------------
            trellis = build_trellis(blueprint, profile)

            # ----------------------------------------------------------------
            # PHASE D — Arc Optimization (A/B Dual Draft)
            # ----------------------------------------------------------------
            optimizer = ArcOptimizer(profile=profile)
            draft_a, draft_b = optimizer.optimize(
                trellis            = trellis,
                blueprint_segments = blueprint.segments,
                duration_target_s  = duration_s,
            )

            # Handle failure modes (§10)
            if not draft_a.global_order:
                codes = draft_a.diagnostic_codes + draft_b.diagnostic_codes
                menu  = RelaxationMenu.from_codes([str(c) for c in codes])
                print(menu.to_consult_question())
                user_choice = input("> ").strip().lower()
                if user_choice in ('no', 'skip', ''):
                    print("Maximum relaxation reached. Exiting.")
                    return
                retry_count += 1
                continue

            # Convert to Playlist for ReActAgent refinement
            playlist_a = arc_draft_to_playlist(draft_a)
            playlist_b = arc_draft_to_playlist(draft_b)

            # ----------------------------------------------------------------
            # PHASE E — Refinement (LLM, bounded tools)
            # ----------------------------------------------------------------
            def ask_user(question):
                print(f"\n{question}")
                return input("> ")

            react_agent = ReActAgent(llm=llm_provider, user_callback=ask_user)
            playlist_a  = react_agent.repair_playlist(playlist_a, profile)

            # ----------------------------------------------------------------
            # Display
            # ----------------------------------------------------------------
            self._display_arc_draft(playlist_a, draft_a, blueprint, "A")
            self._display_arc_draft(playlist_b, draft_b, blueprint, "B")

            if playlist_a.rationale:
                console.print(f"\n[italic dim]{playlist_a.rationale}[/italic dim]")

            # ----------------------------------------------------------------
            # Selection loop
            # ----------------------------------------------------------------
            choice = input("\nWhich mix? [A / B / Neither]: ").strip().upper()
            if choice == 'A':
                self._export(playlist_a, profile, "A")
                return
            elif choice == 'B':
                self._export(playlist_b, profile, "B")
                return
            elif choice == 'NEITHER':
                retry_count += 1
                if retry_count < max_retries:
                    feedback = input("What should change? (e.g. 'more energy', 'no rap'): ")
                    if interviewer and hasattr(interviewer, 'refine_profile'):
                        interviewer.refine_profile(feedback)
                        if hasattr(interviewer, 'to_user_profile'):
                            profile = interviewer.to_user_profile()
                        elif hasattr(interviewer, 'profile'):
                            profile = interviewer.profile
                else:
                    print("Max retries reached. Exporting A by default.")
                    self._export(playlist_a, profile, "Final")
                    return
            else:
                print("Invalid choice. Defaulting to A.")
                self._export(playlist_a, profile, "A")
                return

    def _get_llm_provider(self):
        """Return main LLM (gpt-4o) if key available, else MockLLM."""
        if hasattr(config, 'openai_api_key') and config.openai_api_key:
            from .llm.providers.openai import OpenAIProvider
            return OpenAIProvider()
        return MockLLM()

    def _get_cheap_llm_provider(self):
        """Return cheap LLM (gpt-4o-mini) if key available, else MockLLM.
        Uses the same OpenAIProvider — the model string is passed at call time."""
        if hasattr(config, 'openai_api_key') and config.openai_api_key:
            from .llm.providers.openai import OpenAIProvider
            return OpenAIProvider()
        return MockLLM()


    def _display_arc_draft(self, playlist, draft, blueprint, label: str):
        """Display a playlist as a clean tracklist."""
        total_s = playlist.total_duration_s
        hrs  = total_s // 3600
        mins = (total_s % 3600) // 60
        dur_str = f"{hrs}h {mins}m" if hrs > 0 else f"{mins}m"

        table = Table(
            title       = f"Mix {label}",
            title_style = "bold cyan",
            caption     = f"{len(playlist.track_ids)} tracks  ·  {dur_str}",
            show_lines  = False,
            pad_edge    = True,
        )
        table.add_column("#",       style="dim",  width=3, justify="right")
        table.add_column("Artist",  style="cyan", max_width=24)
        table.add_column("Title",   max_width=34)
        table.add_column("Dur",     style="dim",  width=5, justify="right")

        for i, tid in enumerate(playlist.track_ids):
            t = library.get_track(tid)
            if t:
                d = t.duration_s
                table.add_row(
                    str(i + 1),
                    t.artist[:24],
                    t.title[:34],
                    f"{d // 60}:{d % 60:02d}",
                )

        console.print()
        console.print(table)

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


class _FakePersona:
    """
    Adapter: wraps a UserProfile as a minimal PersonaProfile duck-type
    so BlueprintGenerator can accept it without a full interview.
    Used when a UserProfile is passed directly to _run_generation.
    """
    def __init__(self, profile):
        self.occasion           = profile.context_notes or "personal listening"
        self.personality_words  = profile.target_descriptors[:3]
        self.mood_today         = ""
        self.aesthetic_choice   = ""
        self.era_preference     = ""
        self.wildcard           = ""
        self.confidence         = {}   # §3A — no per-axis confidence for fake persona


if __name__ == '__main__':
    # Load data first
    try:
        library.load()
        MixtapeCLI().cmdloop()
    except Exception as e:
        print(f"Startup Error: {e}")

