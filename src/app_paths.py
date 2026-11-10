"""
Paths and the detector version, in a module that imports NOTHING heavy (no torch, no cv2): the app window (app.py) needs them at start-up
and must not pay for importing torch. config.py re-exports all of it, so every other module keeps using `from config import ...`.

Two roots: the bundle (read-only things that ship with the program: models/) and the app folder (things the user owns and edits:
team_colors.json, output/). Identical when running from source; different in the PyInstaller app (tools/build_exe.py), where the bundle
is the exe's _internal folder and the app folder is the folder that holds the exe.
"""
import os
import sys

# Which trained detector the whole pipeline uses (models/match_detector_<version>.pt, see the long note in config.py). Output folders
# get this as a suffix (pipeline_outputs/<session>_<version>), so runs with different versions never reuse each other's CSVs.
DETECTOR_VERSION = "v2"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = bool(getattr(sys, "frozen", False))
BUNDLE_ROOT = getattr(sys, "_MEIPASS", PROJECT_ROOT) if FROZEN else PROJECT_ROOT
APP_ROOT = os.path.dirname(os.path.abspath(sys.executable)) if FROZEN else PROJECT_ROOT
