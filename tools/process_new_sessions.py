"""
Runs the full pipeline (via main.process_session) on specific raw_footage/
sessions only, instead of main.main()'s "every session under raw_footage/".

Used here to process just the two brand-new, never-trained-on sessions
(2026-05-01 morning, 2026-05-05 evening) without also reprocessing
2026-01-30, which has no pipeline_outputs yet (it was only ever used to pull
frames for the bootstrap dataset, not run through the full pipeline) and
would add ~81 minutes of unrelated processing time to this run.

Usage:  .venv/Scripts/python.exe tools/process_new_sessions.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from config import DETECTOR_VERSION
from main import process_session

SESSIONS = ["2026-05-01_morning", "2026-05-05_evening"]

if __name__ == "__main__":
    for session_name in SESSIONS:
        input_dir = os.path.join(PROJECT_ROOT, "raw_footage", session_name)
        output_dir = os.path.join(PROJECT_ROOT, "pipeline_outputs", f"{session_name}_{DETECTOR_VERSION}")  # same naming as main.main()
        print(f"\n{'#' * 60}\nSESSION: {session_name}\n{'#' * 60}")
        process_session(input_dir, output_dir)
