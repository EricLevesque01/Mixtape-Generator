
import sys
import os

# Add src to the python path so 'mixtape_curator' can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from mixtape_curator.ui_cli import MixtapeCLI
    from mixtape_curator.library import library
except ImportError as e:
    print(f"Startup Error: Could not import application modules.\n{e}")
    sys.exit(1)

if __name__ == "__main__":
    print("Initializing Mixtape Curator...")
    try:
        # Load library data
        library.load()
        
        # Start CLI
        print("Starting CLI Interface...")
        MixtapeCLI().cmdloop()
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\nFatal Error: {e}")
