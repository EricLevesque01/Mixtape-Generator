from mixtape_curator.exporter import exporter
from mixtape_curator.models import Playlist, PlaylistScores
from mixtape_curator.library import library
from pathlib import Path
import pandas as pd
import shutil

def test_export_functionality():
    # Setup mock data 
    tracks = [{
        "id": "t1",
        "title": "Test Song",
        "artist": "Test Artist",
        "duration_s": 300,
        "energy": 0.5,
        "valence": 0.5,
        "intensity": 0.5,
        "spotify_uri": "spotify:track:123",
        "rym_data_primary_genres": [],
        "rym_data_subgenres": [],
        "rym_data_descriptors": []
    }]
    library.df = pd.DataFrame(tracks)
    
    # Create Dummy Playlist
    pl = Playlist(
        id="test_export_pl",
        track_ids=["t1"],
        total_duration_s=300,
        scores=PlaylistScores()
    )
    
    # Run Export
    base_name = "test_mixtape"
    exporter.export_playlist(pl, base_name)
    
    # Verify Files Exist
    txt_path = Path("exports") / f"{base_name}.txt"
    m3u8_path = Path("exports") / f"{base_name}.m3u8"
    
    assert txt_path.exists(), "TXT file not created"
    assert m3u8_path.exists(), "M3U8 file not created"
    
    # Check content
    content = m3u8_path.read_text(encoding="utf-8")
    assert "#EXTM3U" in content
    assert "#EXTINF:300,Test Artist - Test Song" in content
    assert "# Spotify URI: spotify:track:123" in content
    
    print("Checkpoint 8 Passed: Export files created and verified.")
    
    # Cleanup
    shutil.rmtree("exports")

if __name__ == "__main__":
    test_export_functionality()
