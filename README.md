# Speech Segmentation from Audio/Video

Energy-based speech activity detection pipeline with a simple web demo for visualization.

## Features
- Extract audio from video (mp4, mkv, mov, avi, webm, m4v)
- Standardize audio to mono, 16 kHz
- Detect speech segments via short-time RMS energy (no ML)
- Export timestamps to JSON and per-segment WAV clips
- Website to visualize waveform and segments with per-segment playback

## Directory
- `segment_speech.py` — main pipeline script
- `requirements.txt` — Python dependencies (requires ffmpeg for video inputs)
- `website/` — static site (index.html, CSS, JS) to visualize results

## Quickstart (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run on audio
python segment_speech.py input.wav --out_dir output_wav

# Run on video (requires ffmpeg)
python segment_speech.py input.mp4 --out_dir output_mp4

# Adjust sensitivity
python segment_speech.py input.wav --threshold_scale 0.25 --pad_ms 150 --min_speech_ms 300 --min_gap_ms 150
```

Outputs in the `--out_dir`:
- `audio.wav` — standardized audio
- `segments.json` — `[ { "start": s, "end": e }, ... ]` in seconds
- `segments/segment_XX.wav` — segment clips

## Website (local)
Serve the site and open http://localhost:8000
```powershell
python -m http.server 8000 --directory website
```
Then upload `audio.wav` and `segments.json` to visualize and play segments.

## License
MIT
