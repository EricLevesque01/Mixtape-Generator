
import unittest
from unittest.mock import patch
from mixtape_curator.ui_cli import MixtapeCLI
from mixtape_curator.models import UserProfile, Playlist, PlaylistScores

class TestSpotifyMock(unittest.TestCase):
    def test_spotify_export_call(self):
        # 1. Setup Mock
        with patch('mixtape_curator.ui_cli.spotify_exporter') as mock_exporter, \
             patch('mixtape_curator.ui_cli.config') as mock_config, \
             patch('builtins.input', return_value='y'):
            
            # Pretend we have creds
            mock_config.spotify_client_id = "fake_id"
            mock_config.spotify_client_secret = "fake_secret"
            
            mock_exporter.export_playlist.return_value = "Success: https://spotify.com/playlist/123"
            
            # 2. Instantiate CLI
            cli = MixtapeCLI()
            
            # 3. Create Dummy Data
            profile = UserProfile(recipient="Test User")
            playlist = Playlist(
                id="p1", 
                track_ids=["t1"], 
                total_duration_s=200, 
                scores=PlaylistScores(total=0.8)
            )
            
            # 4. Call _export
            # Capture print output? Not strictly necessary if we check mock call
            cli._export(playlist, profile, "Test")
            
            # 5. Verify
            mock_exporter.export_playlist.assert_called_once()
            args, _ = mock_exporter.export_playlist.call_args
            self.assertEqual(args[0], playlist)
            self.assertIn("Test User", args[1])

if __name__ == '__main__':
    unittest.main()
