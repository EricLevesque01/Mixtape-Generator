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

        agent = PersonaInterviewer()
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

        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            print(f"\n--- Round {retry_count + 1} / {max_retries} ---")

            # ----------------------------------------------------------------
            # PHASE B — Blueprint (LLM Creative Director)
            # ----------------------------------------------------------------
            print("▶ Generating narrative blueprint...")
            llm_provider = self._get_llm_provider()
            bp_gen = build_blueprint_generator(llm_provider)
            persona_obj = persona or profile   # fallback if already a UserProfile
            blueprint = bp_gen.generate(
                persona     = persona_obj if hasattr(persona_obj, 'confidence') else _FakePersona(profile),
                duration_target_s = profile.duration_target_s or config.duration_target_s,
            )
            console.print(
                f"  [dim]Blueprint:[/dim] [bold]{len(blueprint.segments)} segments[/bold] — "
                f"[italic]{blueprint.narrative_arc[:80]}...[/italic]"
            )

            # ----------------------------------------------------------------
            # PHASE C — Segment Trellis (Deterministic Algorithm)
            # ----------------------------------------------------------------
            print("▶ Building segment trellis...")
            trellis = build_trellis(blueprint, profile)
            if trellis.undersized_segments:
                console.print(
                    f"  [yellow]Warning:[/yellow] {len(trellis.undersized_segments)} segment(s) "
                    f"have fewer than 20 candidates: {trellis.undersized_segments}"
                )

            # ----------------------------------------------------------------
            # PHASE D — Arc Optimization (A/B Dual Draft)
            # ----------------------------------------------------------------
            print("▶ Optimising arc (A/B)...")
            optimizer = ArcOptimizer(profile=profile)
            draft_a, draft_b = optimizer.optimize(
                trellis            = trellis,
                blueprint_segments = blueprint.segments,
                duration_target_s  = profile.duration_target_s or config.duration_target_s,
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
            print("▶ Agent refining playlist A...")
            def ask_user(question):
                print(f"\n[Agent]: {question}")
                return input("> ")

            react_agent = ReActAgent(llm=llm_provider, user_callback=ask_user)
            playlist_a  = react_agent.repair_playlist(playlist_a, profile)

            # ----------------------------------------------------------------
            # Display
            # ----------------------------------------------------------------
            console.print("\n[bold green]=== GENERATION COMPLETE ===[/bold green]")
            self._display_arc_draft(playlist_a, draft_a, blueprint, "A")
            self._display_arc_draft(playlist_b, draft_b, blueprint, "B")

            # Show rationale if available
            if playlist_a.rationale:
                console.print(f"\n[bold cyan]Curator Rationale:[/bold cyan] {playlist_a.rationale}")

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
        """Return real LLM if key available, else MockLLM."""
        if hasattr(config, 'openai_api_key') and config.openai_api_key:
            from .llm.providers.openai import OpenAIProvider
            return OpenAIProvider()
        return MockLLM()

    def _display_arc_draft(self, playlist, draft, blueprint, label: str):
        """Display a playlist with segment structure using Rich."""
        score  = playlist.scores.total
        if score >= 0.80:
            score_style = "bold green"
        elif score >= 0.60:
            score_style = "bold yellow"
        else:
            score_style = "bold red"

        # Build segment lookup: track_id -> segment theme
        seg_by_track = {}
        seg_map = {s.segment_id: s for s in blueprint.segments}
        for sid, tids in (draft.tracks_per_segment.items() if draft else {}).items():
            seg = seg_map.get(sid)
            theme = seg.theme[:30] if seg else sid
            for tid in tids:
                seg_by_track[tid] = theme

        table = Table(
            title       = f"Playlist {label}",
            title_style = "bold cyan",
            caption     = (
                f"Score: [{score_style}]{score:.2f}[/{score_style}] | "
                f"{len(playlist.track_ids)} tracks | "
                f"{playlist.total_duration_s // 60}m {playlist.total_duration_s % 60}s | "
                f"Boundary flow: {draft.boundary_flow_score:.2f}"
            ) if draft else f"Playlist {label}",
            show_lines  = False,
            pad_edge    = True,
        )
        table.add_column("#",  style="dim", width=3, justify="right")
        table.add_column("Artist",  style="cyan",   max_width=22)
        table.add_column("Title",   max_width=28)
        table.add_column("Dur",     style="dim",    width=5, justify="right")
        table.add_column("Segment", style="magenta", max_width=22)
        table.add_column("Why it fits", style="italic", max_width=40)

        for i, tid in enumerate(playlist.track_ids):
            t = library.get_track(tid)
            if t:
                dur_str  = f"{t.duration_s // 60}:{t.duration_s % 60:02d}"
                note     = playlist.track_notes.get(tid, "Fits the journey.")
                segment  = seg_by_track.get(tid, "")
                table.add_row(
                    str(i + 1),
                    t.artist[:22],
                    t.title[:28],
                    dur_str,
                    segment[:22],
                    note[:40],
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

