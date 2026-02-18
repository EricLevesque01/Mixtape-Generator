"""
enrich_album_metadata.py — Extract track numbers, total tracks, and disc numbers from audio files.
Uses mutagen to read native metadata tags.
"""
import os
import json
import logging
from pathlib import Path
from mutagen import File

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
LIBRARY_PATH = ROOT / "data" / "library.json"

def get_album_metadata(file_path):
    """
    Extract track/disc metadata using mutagen.
    Handles MP4 (M4A) specifically but handles others too.
    """
    try:
        audio = File(file_path)
        if audio is None:
            return None
        
        meta = {
            "track_number": None,
            "total_tracks": None,
            "disc_number": None
        }
        
        # M4A/MP4 tags
        if "trkn" in audio:
            trkn = audio["trkn"]
            if isinstance(trkn, list) and len(trkn) > 0:
                val = trkn[0]
                if isinstance(val, tuple):
                    meta["track_number"] = val[0]
                    if len(val) > 1:
                        meta["total_tracks"] = val[1]
        
        if "disk" in audio:
            disk = audio["disk"]
            if isinstance(disk, list) and len(disk) > 0:
                val = disk[0]
                if isinstance(val, tuple):
                    meta["disc_number"] = val[0]
                    
        # ID3 tags (MP3)
        if hasattr(audio, 'tags') and audio.tags:
            # Track number
            if 'TRCK' in audio.tags:
                trck = str(audio.tags['TRCK'])
                if '/' in trck:
                    parts = trck.split('/')
                    try:
                        meta["track_number"] = int(parts[0])
                        meta["total_tracks"] = int(parts[1])
                    except ValueError: pass
                else:
                    try: meta["track_number"] = int(trck)
                    except ValueError: pass
            
            # Disc number
            if 'TPOS' in audio.tags:
                tpos = str(audio.tags['TPOS'])
                if '/' in tpos:
                    try: meta["disc_number"] = int(tpos.split('/')[0])
                    except ValueError: pass
                else:
                    try: meta["disc_number"] = int(tpos)
                    except ValueError: pass

        return meta
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None

def main():
    if not LIBRARY_PATH.exists():
        logger.error(f"Library not found at {LIBRARY_PATH}")
        return

    with open(LIBRARY_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Enriching {len(data)} tracks with album metadata...")
    
    updated_count = 0
    missing_path = 0
    fail_count = 0

    for i, track in enumerate(data):
        file_path = track.get("file_path")
        if not file_path:
            missing_path += 1
            continue
            
        if not os.path.exists(file_path):
            # Try to fix path if it was copied from another system
            # (Assuming user might have moved library)
            pass
            
        meta = get_album_metadata(file_path)
        if meta:
            # Only update if we found something
            if meta["track_number"] is not None:
                track["track_number"] = meta["track_number"]
            if meta["total_tracks"] is not None:
                track["total_tracks"] = meta["total_tracks"]
            if meta["disc_number"] is not None:
                track["disc_number"] = meta["disc_number"]
            
            updated_count += 1
        else:
            fail_count += 1

        if (i + 1) % 500 == 0:
            logger.info(f"Processed {i + 1}/{len(data)} tracks...")

    # Save backup
    backup_path = LIBRARY_PATH.with_suffix(".json.bak_meta")
    import shutil
    shutil.copy(LIBRARY_PATH, backup_path)
    logger.info(f"Backup created at {backup_path}")

    with open(LIBRARY_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Finished! Updated {updated_count} tracks.")
    logger.info(f"Gaps: {missing_path} missing paths, {fail_count} failed reads.")

if __name__ == "__main__":
    main()
