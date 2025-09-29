#!/usr/bin/env python3
"""
Speech Segmentation from Audio/Video

Features:
- Extract audio from video (e.g., .mp4) and save as mono 16 kHz WAV
- Convert any audio input to standard format (mono, 16 kHz)
- Detect speech segments using short-time energy (no ML)
- Save detected segments as JSON: [{"start": s, "end": e}, ...] (seconds)
- Export segmented audio clips as WAV files

Dependencies:
- numpy, pydub, moviepy (only needed for video inputs)

Notes for Windows:
- ffmpeg is required by pydub/moviepy. Install and ensure ffmpeg is in PATH or set pydub.AudioSegment.converter.

Example usage (PowerShell):
  # Create venv and install deps
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt

  # Run on a video input
  python segment_speech.py input.mp4 --out_dir output_mp4

  # Run on an audio input
  python segment_speech.py input.wav --out_dir output_wav

  # Tweak detection sensitivity (threshold_scale, pad, mins)
  python segment_speech.py input.wav --threshold_scale 0.25 --pad_ms 150 --min_speech_ms 300 --min_gap_ms 150

Deliverables produced per run in the output directory:
- audio.wav             (standardized mono 16 kHz audio)
- segments.json         (array of start/end times in seconds)
- segments/segment_XX.wav (individual segment files)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from pydub import AudioSegment

# moviepy is optional; only required for video input
try:
    from moviepy.editor import VideoFileClip  # type: ignore
    MOVIEPY_AVAILABLE = True
except Exception:
    MOVIEPY_AVAILABLE = False


# -----------------------------
# Utility helpers
# -----------------------------

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTS


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------
# I/O: Extract / Load / Standardize
# -----------------------------

def extract_audio_from_video(video_path: Path, tmp_wav_path: Path) -> Path:
    """Extract audio track from a video into a temporary WAV file.
    Returns the path to the extracted WAV file.
    """
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError(
            "moviepy is required to process video inputs. Install with `pip install moviepy`."
        )
    # Use moviepy to read audio and write PCM 16-bit WAV
    with VideoFileClip(str(video_path)) as clip:
        if clip.audio is None:
            raise RuntimeError("No audio track found in the video.")
        # Write as 16-bit PCM at original rate; we'll standardize later to 16 kHz mono
        # moviepy will use ffmpeg under the hood
        clip.audio.write_audiofile(
            str(tmp_wav_path),
            fps=None,  # keep original sr first; we'll resample later
            nbytes=2,  # 16-bit PCM
            codec="pcm_s16le",
            verbose=False,
            logger=None,
        )
    return tmp_wav_path


def load_and_standardize_audio(input_audio_path: Path, target_sr: int = 16000) -> AudioSegment:
    """Load audio with pydub and convert to mono, target sample rate."""
    audio = AudioSegment.from_file(str(input_audio_path))
    # Convert to mono 16 kHz
    audio = audio.set_frame_rate(target_sr).set_channels(1)
    return audio


# -----------------------------
# Speech Activity Detection (Energy-based)
# -----------------------------

def audiosegment_to_float32(audio: AudioSegment) -> np.ndarray:
    """Convert pydub AudioSegment (mono) to float32 numpy array in [-1, 1]."""
    samples = np.array(audio.get_array_of_samples())
    if audio.channels != 1:
        # Should not happen for standardized audio, but guard anyway
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)
    # Normalize to [-1, 1] based on sample width
    max_val = float(1 << (8 * audio.sample_width - 1))
    data = samples.astype(np.float32) / max_val
    return data


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    ma = (cumsum[win:] - cumsum[:-win]) / float(win)
    # pad to original length
    pad_left = win // 2
    pad_right = len(x) - len(ma) - pad_left
    return np.pad(ma, (pad_left, pad_right), mode="edge")


def detect_speech_segments(
    audio: AudioSegment,
    frame_ms: int = 20,
    hop_ms: int = 10,
    min_speech_ms: int = 300,
    min_gap_ms: int = 150,
    pad_ms: int = 100,
    abs_threshold: Optional[float] = None,
    threshold_scale: float = 0.2,
) -> List[Tuple[int, int]]:
    """
    Detect speech segments using short-time RMS energy with a dynamic threshold.

    Parameters:
    - frame_ms: analysis frame size in ms
    - hop_ms: hop size in ms
    - min_speech_ms: drop segments shorter than this
    - min_gap_ms: merge neighboring segments separated by gaps shorter than this
    - pad_ms: expand each segment by this padding on both ends
    - abs_threshold: absolute threshold on RMS (0..1 in float32 domain). If None, auto-threshold
    - threshold_scale: controls auto-threshold = median + scale * (p95 - median)

    Returns list of (start_ms, end_ms)
    """
    sr = audio.frame_rate
    data = audiosegment_to_float32(audio)

    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))

    # Frame the signal and compute RMS per frame
    n_frames = 1 + max(0, (len(data) - frame_len) // hop_len)
    if n_frames <= 0:
        return []

    rms_vals = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len
        frame = data[start:end]
        # Zero-pad last frame if needed
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        rms = math.sqrt(float(np.mean(frame * frame)) + 1e-12)
        rms_vals[i] = rms

    # Smooth RMS slightly to reduce spurious spikes
    rms_smooth = moving_average(rms_vals, win=max(1, int(50 / hop_ms)))  # ~50ms smoothing

    # Determine threshold
    if abs_threshold is not None:
        thr = float(abs_threshold)
    else:
        med = float(np.median(rms_smooth))
        p95 = float(np.percentile(rms_smooth, 95))
        thr = med + threshold_scale * (p95 - med)
        thr = max(thr, 1e-4)  # guard against too-low values

    # Binary mask of speech frames
    speech_mask = rms_smooth > thr

    # Convert frame indices to ms ranges
    times_ms = np.arange(n_frames) * hop_ms

    segments: List[Tuple[int, int]] = []
    in_seg = False
    seg_start_ms = 0

    for i, is_speech in enumerate(speech_mask):
        t = int(times_ms[i])
        if is_speech and not in_seg:
            in_seg = True
            seg_start_ms = t
        elif (not is_speech) and in_seg:
            in_seg = False
            seg_end_ms = t + frame_ms  # close at end of frame window
            segments.append((seg_start_ms, seg_end_ms))

    if in_seg:
        # Close final segment at end of audio
        segments.append((seg_start_ms, len(audio)))

    if not segments:
        return []

    # Merge segments with small gaps and apply padding
    merged: List[Tuple[int, int]] = []
    cur_start, cur_end = segments[0]
    for s, e in segments[1:]:
        if s - cur_end <= min_gap_ms:
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))

    # Apply padding and clamp to audio bounds
    padded: List[Tuple[int, int]] = []
    for s, e in merged:
        s2 = max(0, s - pad_ms)
        e2 = min(len(audio), e + pad_ms)
        padded.append((s2, e2))

    # Filter out too-short segments
    final_segs = [(s, e) for (s, e) in padded if (e - s) >= min_speech_ms]

    return final_segs


# -----------------------------
# Exporters
# -----------------------------

def write_segments_json(segments_ms: List[Tuple[int, int]], json_path: Path) -> None:
    as_seconds = [
        {"start": round(s / 1000.0, 3), "end": round(e / 1000.0, 3)} for (s, e) in segments_ms
    ]
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(as_seconds, f, indent=2)


def export_segment_wavs(audio: AudioSegment, segments_ms: List[Tuple[int, int]], out_dir: Path) -> None:
    seg_dir = out_dir / "segments"
    ensure_dir(seg_dir)
    width = max(2, len(str(len(segments_ms))))
    for idx, (s, e) in enumerate(segments_ms, start=1):
        clip = audio[s:e]
        name = f"segment_{idx:0{width}d}.wav"
        out_path = seg_dir / name
        clip.export(str(out_path), format="wav")


# -----------------------------
# Main pipeline
# -----------------------------

def process_file(
    input_path: Path,
    out_dir: Path,
    frame_ms: int = 20,
    hop_ms: int = 10,
    min_speech_ms: int = 300,
    min_gap_ms: int = 150,
    pad_ms: int = 100,
    abs_threshold: Optional[float] = None,
    threshold_scale: float = 0.2,
) -> None:
    ensure_dir(out_dir)

    tmp_extracted_wav = out_dir / "_extracted_tmp.wav"
    standardized_wav = out_dir / "audio.wav"  # deliverable name

    # Step 1: Ingest
    if is_video_file(input_path):
        print(f"[1/4] Extracting audio from video: {input_path}")
        extracted_path = extract_audio_from_video(input_path, tmp_extracted_wav)
        print("       Loading and standardizing (mono, 16 kHz)...")
        audio = load_and_standardize_audio(extracted_path, target_sr=16000)
    else:
        print(f"[1/4] Loading audio: {input_path}")
        audio = load_and_standardize_audio(input_path, target_sr=16000)

    # Save standardized audio
    audio.export(str(standardized_wav), format="wav")
    print(f"[2/4] Saved standardized audio: {standardized_wav}")

    # Cleanup temp
    if tmp_extracted_wav.exists():
        try:
            tmp_extracted_wav.unlink()
        except Exception:
            pass

    # Step 2: Detect segments
    print("[3/4] Detecting speech segments (energy-based)...")
    segments_ms = detect_speech_segments(
        audio,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        min_speech_ms=min_speech_ms,
        min_gap_ms=min_gap_ms,
        pad_ms=pad_ms,
        abs_threshold=abs_threshold,
        threshold_scale=threshold_scale,
    )

    json_path = out_dir / "segments.json"
    write_segments_json(segments_ms, json_path)
    print(f"       Found {len(segments_ms)} segments. Wrote: {json_path}")

    # Step 3: Export per-segment WAV files
    print("[4/4] Exporting segment WAV files...")
    export_segment_wavs(audio, segments_ms, out_dir)
    print(f"       Exported segments to: {out_dir / 'segments'}")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Speech Segmentation from Audio/Video (energy-based)")
    p.add_argument("input", type=str, help="Path to input audio/video file")
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: <input_stem>_out)",
    )
    p.add_argument("--frame_ms", type=int, default=20, help="Frame size in ms (default: 20)")
    p.add_argument("--hop_ms", type=int, default=10, help="Hop size in ms (default: 10)")
    p.add_argument(
        "--min_speech_ms",
        type=int,
        default=300,
        help="Minimum speech segment length in ms (default: 300)",
    )
    p.add_argument(
        "--min_gap_ms",
        type=int,
        default=150,
        help="Merge segments closer than this gap in ms (default: 150)",
    )
    p.add_argument("--pad_ms", type=int, default=100, help="Padding around each segment in ms (default: 100)")
    p.add_argument(
        "--abs_threshold",
        type=float,
        default=None,
        help="Absolute RMS threshold in [0..1]. If provided, overrides auto-threshold.",
    )
    p.add_argument(
        "--threshold_scale",
        type=float,
        default=0.2,
        help="For auto-threshold: thr = median + scale * (p95 - median) (default: 0.2)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    out_dir = Path(args.out_dir) if args.out_dir else input_path.with_suffix("").parent / f"{input_path.stem}_out"
    process_file(
        input_path=input_path,
        out_dir=out_dir,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        min_speech_ms=args.min_speech_ms,
        min_gap_ms=args.min_gap_ms,
        pad_ms=args.pad_ms,
        abs_threshold=args.abs_threshold,
        threshold_scale=args.threshold_scale,
    )


if __name__ == "__main__":
    main()
