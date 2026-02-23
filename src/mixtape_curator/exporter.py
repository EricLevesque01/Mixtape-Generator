from pathlib import Path
from typing import List
from mixtape_curator.models import Playlist, Track
from mixtape_curator.library import library

class Exporter:
    def __init__(self, export_dir: str = "exports"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)

    def export_playlist(self, playlist: Playlist, filename_base: str):
        """
        Export playlist to M3U8 and TXT formats.
        """
        tracks = [library.get_track(tid) for tid in playlist.track_ids]
        tracks = [t for t in tracks if t] # filter nones
        
        # 1. TXT Summary
        self._write_txt(tracks, filename_base, playlist)
        
        # 2. M3U8 Playlist
        self._write_m3u8(tracks, filename_base)
        
        return f"Exported to {self.export_dir}/{filename_base}.[txt|m3u8]"

    def _write_txt(self, tracks: List[Track], base: str, playlist: Playlist):
        path = self.export_dir / f"{base}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Mixtape ID: {playlist.id}\n")
            f.write(f"Total Duration: {playlist.total_duration_s}s\n")
            f.write(f"Score: {playlist.scores.total:.2f}\n")
            f.write("-" * 40 + "\n")
            for i, t in enumerate(tracks):
                reason = playlist.track_notes.get(t.id, "Selected for its fit in the journey.")
                f.write(f"{i+1}. {t.artist} - {t.title} ({t.duration_s}s)\n")
                f.write(f"   Why it fits: {reason}\n")

    def _write_m3u8(self, tracks: List[Track], base: str):
        """
        Write standard M3U8 file. 
        Note: Since we don't have local paths, we just write EXTINF metadata.
        If we had file paths, they would go here.
        """
        path = self.export_dir / f"{base}.m3u8"
        with open(path, "w", encoding="utf-8") as f:
            f.write("#EXTM3U\n")
            for t in tracks:
                f.write(f"#EXTINF:{t.duration_s},{t.artist} - {t.title}\n")
                # Write file path if available, otherwise fallback
                if t.file_path:
                   f.write(f"{t.file_path}\n")
                else:
                   f.write(f"# Spotify URI: {t.spotify_uri}\n")

exporter = Exporter()
