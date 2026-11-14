"""
Stage 1 only (YOLO detect/track -> raw `_data.csv`), headless, for whole sessions.

Use it when the raw detections are wanted but not the slow Stage 3 render, e.g. to
re-extract sessions with a new detector version, or to build the data the Stage 2
parameters are tuned on. Output lands exactly where main.py would put it
(pipeline_outputs/<session>_<DETECTOR_VERSION>/<video>_data.csv), and a video whose
CSV already exists is skipped, so an interrupted run can simply be started again.

Usage:  .venv/Scripts/python.exe tools/extract_data_only.py <session> [<session> ...]
        (default sessions: the two May ones the detector was never trained on)
"""
import glob
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch  # noqa: F401  (must be imported before pandas, see main.py)
from audio_utils import audio_path_for, extract_audio_features
from config import DETECTOR_VERSION
from logging_utils import log
from video_processor import VideoProcessor

DEFAULT_SESSIONS = ["2026-05-01_morning", "2026-05-05_evening"]

if __name__ == "__main__":
    sessions = sys.argv[1:] or DEFAULT_SESSIONS
    t0 = time.time()
    for session in sessions:
        out_dir = os.path.join(PROJECT_ROOT, "pipeline_outputs", f"{session}_{DETECTOR_VERSION}")
        os.makedirs(out_dir, exist_ok=True)
        for video in sorted(glob.glob(os.path.join(PROJECT_ROOT, "raw_footage", session, "*.MP4"))):
            base = os.path.splitext(os.path.basename(video))[0]
            csv_path = os.path.join(out_dir, f"{base}_data.csv")
            audio_npz = audio_path_for(csv_path)
            if not os.path.exists(audio_npz):
                extract_audio_features(video, audio_npz)   # kick-sound cue for shots (see src/audio_utils.py)
            if os.path.exists(csv_path):
                log(f"{session}/{base}: data CSV exists, skipping.")
                continue
            log(f"{session}/{base}: extracting...")
            partial = csv_path + ".partial"  # renamed only when the whole video finished, so a killed run never leaves a truncated CSV that looks complete
            VideoProcessor(video, partial, show_preview=os.environ.get("FOOTBALL_LIVE_PREVIEW", "0") == "1").process_video()   # live view only when asked for: FOOTBALL_LIVE_PREVIEW=1
            os.replace(partial, csv_path)
    log(f"extract_data_only finished in {(time.time() - t0) / 60:.1f} min.")
