import sys
from mixtape_curator.ui_cli import MixtapeCLI
from mixtape_curator.library import library
from io import StringIO
import time

# Simulate user inputs
inputs = [
    "start",
    "For Testing",
    "Testing Neither Loop",
    "", # Context
    "", # Exclude/Include
    "Indie, Rock", # Genres
    "Balanced", # Mood
    "Uniform", # Cohesion
    "Low", # Energy
    "yes", # Confirm
    "Neither", # Reject 1
    "too slow", # Feedback 1
    "Neither", # Reject 2
    "too slow", # Feedback 2
    "A", # Accept 3
    "quit"
]

def mock_input(prompt=""):
    if not inputs:
        raise EOFError
    val = inputs.pop(0)
    print(f"{prompt}{val}")
    return val

# Monkey patch input
import builtins
builtins.input = mock_input

if __name__ == "__main__":
    library.load()
    cli = MixtapeCLI()
    try:
        cli.cmdloop()
    except Exception as e:
        print(e)
