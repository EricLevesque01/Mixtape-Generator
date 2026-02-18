"""
enrich_librosa.py — Overnight audio feature extraction using librosa.

Reads each track's local .m4a file, extracts:
  - energy    (RMS energy, normalized 0-1)
  - valence   (derived from mode major/minor + spectral brightness)
  - intensity (spectral centroid proxy, normalized 0-1)
  - tempo     (BPM, stored for future use)

Updates library.json in-place, only overwriting tracks still at
synthetic defaults (energy == 0.5 from genre enrichment).

Usage:
  python scripts/enrich_librosa.py [--force] [--limit N] [--dry-run]

  --force     Re-analyze even tracks already enriched
  --limit N   Only process first N tracks (for testing)
  --dry-run   Print what would happen, don't write anything

Runtime estimate: ~2-4 seconds per track → ~4-6 hours for 5,966 tracks.
Run overnight. Progress is saved every 50 tracks so it's resumable.
"""

import sys
import os
# Prevent numba/coverage conflicts that can occur on some Python installs
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

# Ensure we can find the project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.path.join(ROOT, "data", "library.json")
MUSIC_ROOT = r"C:\Users\ericl\Music\tPod V3\iTunes Media\Music"

def check_ffmpeg():
    """Check if FFmpeg is available for .m4a decoding."""
    import shutil
    if shutil.which("ffmpeg"):
        return True
    # Check common iTunes/Apple Music install locations
    candidates = [
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        os.path.expanduser(r"~\ffmpeg\bin\ffmpeg.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            os.environ["PATH"] += os.pathsep + os.path.dirname(c)
            return True
    return False

def extract_features(filepath: str) -> dict | None:
    """
    'Kitchen Sink' audio feature extraction using soundfile + scipy + numpy.
    Extracts all standard librosa-style descriptors without the numba JIT dependency.
    """
    try:
        import soundfile as sf
        import numpy as np
        from scipy import signal
        from scipy.fft import rfft, rfftfreq

        # --- Data Loading ---
        try:
            y, sr = sf.read(filepath, dtype='float32', always_2d=False)
        except Exception:
            import subprocess
            cmd = [
                'ffmpeg', '-i', filepath,
                '-f', 'f32le', '-acodec', 'pcm_f32le',
                '-ar', '22050', '-ac', '1', '-t', '90',
                '-loglevel', 'error', 'pipe:1'
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=120)
            if proc.returncode != 0 or len(proc.stdout) < 1000:
                return None
            y = np.frombuffer(proc.stdout, dtype=np.float32)
            sr = 22050

        if y.ndim > 1:
            y = y.mean(axis=1)

        if sr != 22050:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(22050, sr)
            y = resample_poly(y, 22050 // g, sr // g)
            sr = 22050

        max_samples = 90 * sr
        if len(y) > max_samples:
            y = y[:max_samples]

        if len(y) < sr * 2:
            return None

        # --- Signal Processing Setup ---
        frame_len = 2048
        hop_len = 512
        frames = [y[i:i+frame_len] for i in range(0, len(y)-frame_len, hop_len)]
        num_frames = len(frames)
        
        window = np.hanning(frame_len)
        specs = [np.abs(rfft(f * window)) for f in frames[:400]]
        freqs = rfftfreq(frame_len, d=1.0/sr)

        # --- Core Metrics ---
        
        # 1. Energy & Dynamics
        rms_vals = np.array([np.sqrt(np.mean(f**2)) for f in frames])
        peak_val = np.max(np.abs(y))
        raw_energy = float(np.mean(rms_vals))
        energy = float(np.clip(raw_energy / 0.45, 0.0, 1.0))
        # Dynamic Range: Ratio of peak to average RMS (higher = more dynamic/less compressed)
        dynamic_range = float(np.clip(peak_val / (raw_energy + 1e-8) / 10.0, 0.0, 1.0))

        # 2. Spectral Features
        centroids = []
        flatness = []
        roll_off = []
        flux = []
        entropy = []
        prev_spec = None
        
        for s in specs:
            spec_sum = s.sum() + 1e-8
            # Centroid (Brightness)
            centroids.append(np.sum(freqs * s) / spec_sum)
            # Flatness (Noisiness)
            log_mean = np.mean(np.log(s + 1e-8))
            flatness.append(np.exp(log_mean) / (spec_sum / len(s)))
            # Roll-off (85% freq)
            cum_sum = np.cumsum(s)
            roll_off.append(freqs[np.searchsorted(cum_sum, 0.85 * cum_sum[-1])])
            # Flux (Change between frames - proxy for Liveness/Complexity)
            if prev_spec is not None:
                flux.append(np.sqrt(np.mean((s - prev_spec)**2)))
            prev_spec = s
            # Entropy (Information density)
            psd = s**2 / spec_sum**2
            entropy.append(-np.sum(psd * np.log2(psd + 1e-8)))

        raw_centroid = np.mean(centroids)
        raw_flatness = np.mean(flatness)
        raw_rolloff = np.mean(roll_off)
        raw_flux = np.mean(flux) if flux else 0.0
        raw_entropy = np.mean(entropy)

        # 3. Temporal Features (Tempo, Danceability, Speechiness)
        # ZCR (Speechiness proxy)
        zcr = np.mean([np.mean(np.abs(np.diff(np.sign(f)))) / 2 for f in frames])
        speechiness = float(np.clip(zcr * 2.0, 0.0, 1.0)) # Higher ZCR = more speech-like/noisy
        
        # Tempo & Pulse
        onset_env = np.maximum(0, np.diff(rms_vals))
        tempo = 120.0
        danceability = 0.5
        beat_strength = 0.5
        if len(onset_env) > 100:
            ac = np.correlate(onset_env, onset_env, mode='full')
            ac = ac[len(ac)//2:]
            min_lag = int(60.0 / 220 * sr / hop_len)
            max_lag = int(60.0 / 40 * sr / hop_len)
            min_lag = max(1, min_lag)
            max_lag = min(len(ac)-1, max_lag)
            if max_lag > min_lag:
                peak_idx = np.argmax(ac[min_lag:max_lag]) + min_lag
                peak_val_ac = ac[peak_idx]
                avg_val_ac = np.mean(ac[min_lag:max_lag])
                tempo = 60.0 / (peak_idx * hop_len / sr)
                beat_strength = float(np.clip(peak_val_ac / (avg_val_ac + 1e-8) / 5.0, 0.0, 1.0))
                danceability = float(np.clip(beat_strength * (1.0 - raw_flatness), 0.0, 1.0))

        # 4. Harmonic Features (Key, Tonality)
        chroma = np.zeros(12)
        for s in specs:
            for pitch in range(12):
                f_ref = 440.0 * (2 ** ((pitch - 9) / 12.0))
                for octave in [-2, -1, 0, 1, 2]:
                    f_target = f_ref * (2 ** octave)
                    bin_idx = int(round(f_target * frame_len / sr))
                    if 0 < bin_idx < len(s):
                        chroma[pitch] += s[bin_idx]
        chroma /= (chroma.max() + 1e-8)
        
        major_t = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_t = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        best_key, best_mode, max_corr = 0, 1, -1.0
        for p in range(12):
            rotated = np.roll(chroma, -p)
            maj_c = np.corrcoef(rotated, major_t)[0,1]
            min_c = np.corrcoef(rotated, minor_t)[0,1]
            if maj_c > max_corr: max_corr, best_key, best_mode = maj_c, p, 1
            if min_c > max_corr: max_corr, best_key, best_mode = min_c, p, 0

        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_full = f"{keys[best_key]} {'Major' if best_mode == 1 else 'Minor'}"
        tonality = float(np.clip(max_corr, 0.0, 1.0))

        # --- Aggregated Descriptors ---
        
        # Intensity: Brightness + Noisiness + Compression
        intensity = float(np.clip((raw_centroid / 4000) + (raw_flatness * 0.5) + (1.0 - dynamic_range), 0.0, 1.0))
        
        # Valence: Mode + Tempo + Energy - Complexity
        val_base = 0.6 if best_mode == 1 else 0.4
        valence = float(np.clip(val_base + (tempo-110)/200 + (energy-0.5)*0.2 - (raw_entropy/20), 0.0, 1.0))
        
        # Acousticness: Inverse of brightness and noisiness
        acousticness = float(np.clip(1.0 - (raw_rolloff / 6000) - (raw_flatness), 0.0, 1.0))
        
        # Instrumentalness: Low ZCR + High tonality stability
        instrumentalness = float(np.clip(tonality * (1.0 - zcr*5), 0.0, 1.0))

        # Liveness: High spectral flux + low tonality
        liveness = float(np.clip(raw_flux * 5.0 + (1.0 - tonality)*0.2, 0.0, 1.0))

        return {
            "energy": round(energy, 3),
            "valence": round(valence, 3),
            "intensity": round(intensity, 3),
            "tempo": round(float(tempo), 1),
            "danceability": round(danceability, 3),
            "beat_strength": round(beat_strength, 3),
            "key": best_key,
            "mode": best_mode,
            "key_full": key_full,
            "tonality": round(tonality, 3),
            "acousticness": round(acousticness, 3),
            "instrumentalness": round(instrumentalness, 3),
            "speechiness": round(speechiness, 3),
            "liveness": round(liveness, 3),
            "brightness": round(float(raw_centroid / 5000), 3),
            "flatness": round(float(raw_flatness), 3),
            "entropy": round(float(raw_entropy), 3),
            "dynamic_range": round(dynamic_range, 3)
        }

    except Exception as e:
        print(f"    extract_features error: {type(e).__name__}: {e}")
        return None

    except Exception as e:
        print(f"    extract_features error: {type(e).__name__}: {e}")
        return None


def find_audio_file(track: dict) -> str | None:
    """
    Find the local audio file for a track.
    Tries the track's file_path first, then searches MUSIC_ROOT.
    """
    fp = track.get("file_path", "")
    if fp and os.path.exists(fp):
        return fp

    artist = track.get("artist", "").replace("/", "-").replace(":", "-")
    album = track.get("album", "").replace("/", "-").replace(":", "-")
    title = track.get("title", "").replace("/", "-").replace(":", "-")

    for ext in [".m4a", ".mp3", ".flac", ".wav"]:
        candidate = os.path.join(MUSIC_ROOT, artist, album, title + ext)
        if os.path.exists(candidate):
            return candidate

    return None


def main():
    print("KITCHEN SINK MODE ACTIVE (18 descriptors)")
    parser = argparse.ArgumentParser(description="Enrich library.json with audio features")
    parser.add_argument("--force", action="store_true", help="Re-analyze all tracks")
    parser.add_argument("--limit", type=int, default=None, help="Only process N tracks")
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes")
    parser.add_argument("--music-root", type=str, default=MUSIC_ROOT, help="Root of music library")
    args = parser.parse_args()

    print(f"Loading {LIB_PATH}...")
    with open(LIB_PATH, "r", encoding="utf-8") as f:
        tracks = json.load(f)
    print(f"Loaded {len(tracks)} tracks.\n")

    to_process = []
    for t in tracks:
        # Check if already has the NEW features (danceability is a good indicator)
        is_fully_enriched = t.get("enrichment_source") == "librosa" and "danceability" in t
        if args.force or not is_fully_enriched:
            to_process.append(t)

    if args.limit:
        to_process = to_process[:args.limit]

    print(f"Tracks to process: {len(to_process)}")
    print(f"Estimated time: {len(to_process) * 4 / 60:.0f} minutes\n")

    track_by_id = {t["id"]: t for t in tracks}
    success = 0
    skipped_no_file = 0
    skipped_error = 0
    start_time = time.time()

    for i, track in enumerate(to_process):
        tid = track["id"]
        artist = track.get("artist", "?")
        title = track.get("title", "?")

        filepath = find_audio_file(track)
        if not filepath:
            skipped_no_file += 1
            continue

        features = extract_features(filepath)
        if features is None:
            skipped_error += 1
            continue

        if not args.dry_run:
            # Update all fields
            for k, v in features.items():
                track_by_id[tid][k] = v
            track_by_id[tid]["enrichment_source"] = "librosa"

        success += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(to_process) - i - 1) / rate / 60
            print(f"  [{i+1}/{len(to_process)}] OK: {artist} - {title}")
            print(f"    {features['key_full']} | Energy: {features['energy']:.2f} | Dance: {features['danceability']:.2f} | Mood: {features['valence']:.2f}")
            print(f"    Progress: {success} enriched | ~{remaining:.0f} min remaining\n")

            if not args.dry_run:
                with open(LIB_PATH, "w", encoding="utf-8") as f:
                    json.dump(tracks, f, indent=2, ensure_ascii=False)

    if not args.dry_run:
        with open(LIB_PATH, "w", encoding="utf-8") as f:
            json.dump(tracks, f, indent=2, ensure_ascii=False)

    print(f"\n=== DONE in {(time.time()-start_time)/60:.1f} minutes ===")
    print(f"  Enriched: {success} | No file: {skipped_no_file} | Errors: {skipped_error}")


if __name__ == "__main__":
    main()
