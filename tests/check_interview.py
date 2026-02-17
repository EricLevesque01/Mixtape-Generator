from mixtape_curator.interview import Interviewer
from mixtape_curator.models import UserProfile
from mixtape_curator.generator import generator
from mixtape_curator.library import library
from mixtape_curator.config import config
import pandas as pd
from unittest.mock import Mock

def test_interview_flow():
    # Setup mock data 
    tracks = []
    for i in range(10):
        tracks.append({
            "id": f"t{i}",
            "title": f"Song {i}",
            "artist": "Artist A",
            "duration_s": 200,
            "energy": 0.8,
            "valence": 0.8,
            "intensity": 0.8,
            "rym_data_primary_genres": ["Synthpop"],
            "rym_data_subgenres": [],
            "rym_data_descriptors": []
        })
    library.df = pd.DataFrame(tracks)
    
    # 1. Start Interview
    interviewer = Interviewer()
    start_msg = interviewer.start()
    assert "Who am I making a mixtape for" in start_msg
    
    # 2. Answer Intake
    reply, done = interviewer.process_input("For my friend")
    assert not done
    assert "genres" in reply.lower()
    assert interviewer.profile.recipient == "For my friend"
    
    # 3. Answer Genres
    reply, done = interviewer.process_input("Synthpop, New Wave")
    assert not done
    assert "Synthpop" in interviewer.profile.target_genres
    assert "consistent" in reply.lower() or "uniform" in reply.lower()
    
    # 4. Answer Cohesion (Testing Spec v2.1.1 logic)
    reply, done = interviewer.process_input("Make it uniform")
    assert not done
    assert interviewer.profile.targets.uniformity > 0.6
    
    # 5. Answer Energy
    reply, done = interviewer.process_input("high energy")
    assert not done
    assert interviewer.profile.targets.energy == 0.8
    assert "Ready to generate" in reply
    
    # 6. Confirm
    reply, done = interviewer.process_input("yes")
    assert done
    assert interviewer.completed
    
    print("Checkpoint 7 Passed: Interview loop completes and updates profile correctly.")
    
    # 7. Trigger Logic (Integration check)
    draft = generator.create_draft(interviewer.profile)
    assert len(draft.track_ids) > 0
    print(f"Checkpoint 7 Passed: Profile successfully triggers generator ({len(draft.track_ids)} tracks).")

if __name__ == "__main__":
    test_interview_flow()
