"""
Shared constants for the football-analysis pipeline: AI processing resolution,
disk-write batching, and the UI color/font palette used by every rendering stage.
Centralized here so every stage reads the exact same values instead of each
file redefining its own copy.
"""
import os
import cv2
import torch

_HAS_CUDA = torch.cuda.is_available()  # checked once at import time, reused by every constant below

# What it does: internal processing resolution fed to the detection model.
# This is purely an AI-accuracy/speed knob - the live preview window is a
# separate, always-crisp upscale of whatever this is set to (see
# PREVIEW_WIDTH below), so lowering this never makes the preview blockier.
# Hardware-adaptive: a GPU can afford 960. v1 was trained at 640, but measured
# on our own out-of-sample footage, inferring it at 960 found the ball in
# 33.7% of sampled frames vs 29.2% at 640 (640->1280 showed no further gain
# past 960), at a modest ~45% per-frame cost increase - worth it since ball
# recall is this pipeline's weakest spot. v2 is trained at 960 on frames shrunk
# exactly like video_reader.py shrinks them (INTER_NEAREST), so 960 is now its
# native scale. A CPU (OpenVINO or plain PyTorch) stays at 480 - the lowest
# we're willing to drop to - to keep per-frame cost manageable now that no
# frame is skipped; that is below v2's training scale, so expect weaker ball
# recall there.
TARGET_WIDTH = 960 if _HAS_CUDA else 480

# What it does: the resolution the live preview window is built at, by
# upscaling the (small, AI-resolution) frame with high-quality interpolation
# before drawing any boxes/text on it - never by letting the OS window
# stretch an already-small, already-labeled image (which is what caused the
# old blocky/pixelated look: text drawn at 480-640px width, then stretched).
# Height is derived per-video to preserve its native aspect ratio.
PREVIEW_WIDTH = 1280

# What it does: show the live "Football Video" window while Stage 1 extracts data (boxes, team colours, Players/Ball/Goal read-out).
# Measured on this machine: with the window ~8 frames/s, without it ~48 frames/s (a 5-minute video takes ~3.5 min instead of ~19).
# Set False for batch runs; tools/extract_data_only.py always runs without it. The CSV is identical either way.
LIVE_PREVIEW = True

# What it does: how many extracted-data rows accumulate in RAM before being
# flushed to the CSV on disk.
# Impact of change: higher -> fewer disk writes (faster) but more RAM used and
# more data lost if the process crashes mid-run; lower -> safer but causes
# small stutters from frequent disk I/O.
CSV_WRITE_CHUNK = 1800

# Unified person+ball+goal detector, fine-tuned on our own chest-cam footage.
# This REPLACES the old pair of models (a generic COCO person/ball model + a
# separately-trained, domain-mismatched goal model): one weight file, one class
# map, one YOLO call per frame instead of two. Both hardware tiers load the same
# file; only the backend (raw PyTorch on GPU vs OpenVINO on CPU) adapts.
#
# Every trained version lives side by side in models/ as match_detector_<version>.pt,
# and DETECTOR_VERSION is the ONE switch that picks which one the whole pipeline
# uses (switching back to an older version is a one-word change):
#   "v1" - 2907 frames, original labels (wearer's hands labelled as people), trained at 640
#          (datasets/match_dataset_final, training_runs/match_detector_v1)
#   "v2" - same 2907 frames re-reviewed (hands removed, goals doubled), video-level split,
#          trained at 960 on NEAREST-resized frames (datasets/match_dataset_v2,
#          training_runs/match_detector_v2). Held-out test mAP50 0.830 vs v1's 0.755.
# The version is also appended to every output folder name (pipeline_outputs/<session>_<version>),
# so a run with one version can never be mistaken for, or silently reuse, another version's CSVs.
# Paths are resolved relative to this file, not the process's working directory, so they load
# correctly no matter where the pipeline is launched from.
DETECTOR_VERSION = "v2"
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECTOR_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", f"match_detector_{DETECTOR_VERSION}.pt")
DETECTOR_MODEL_OPENVINO_DIR = os.path.join(_PROJECT_ROOT, "models", f"match_detector_{DETECTOR_VERSION}_openvino_model")
# Every trained version uses this same class map (data.yaml: person/ball/goal) -
# fixed by training, not COCO's ids anymore.
PERSON_CLASS_ID = 0
BALL_CLASS_ID = 1
GOAL_CLASS_ID = 2
DETECTOR_CONF_THRESHOLD = 0.25

# BGR color palette (OpenCV draws in BGR, not RGB) shared by every rendering stage.
UI_COLORS = {
    "blue": (0, 191, 255),
    "red": (255, 0, 100),
    "white": (255, 255, 255),
    "dark_green": (0, 180, 45),
    "dark_red": (180, 0, 60),
    "orange": (255, 160, 0),
    "cyan": (0, 255, 255),
    "light_gray": (170, 170, 170),
    "yellow": (255, 255, 90),
    "black": (0, 0, 0),
    "stats_card_bg": (20, 20, 20),
}

# ---------------------------------------------------------------------------
# ONE colour system for everything that is drawn, shared by the Stage 1 live preview and the
# Stage 3 video. All values are BGR, which is what OpenCV draws in, so the name says what you
# actually see on screen (UI_COLORS above predates this and its names do not match what it draws).
# ---------------------------------------------------------------------------
DETECTION_COLORS = {
    "goal": (170, 40, 0),              # dark blue
    "team_a": (0, 255, 255),           # yellow   - OUR team (see event_params.OUR_SHADE_BY_SESSION)
    "team_b": (255, 0, 255),           # magenta  - the opponents
    "player_unassigned": (170, 170, 170),  # grey - detected, but no team yet / no stable track
    "ball": (255, 255, 255),           # white
}
EVENT_COLORS = {
    "PASS_SUCCESS": (0, 200, 0),        # green
    "PASS_INCOMPLETE": (0, 0, 230),     # red
    "SHOT_ON_TARGET": (190, 255, 190),  # very light green
    "SHOT_OFF_TARGET": (193, 182, 255), # light pink
    "DRIBBLE_SHORT": (120, 200, 255),   # three oranges: the longer the dribble, the darker
    "DRIBBLE_MEDIUM": (0, 140, 255),
    "DRIBBLE_LONG": (0, 80, 190),
}

# Typography/line-thickness used for every on-video text and box overlay.
UI_SETTINGS = {
    "box_thickness": 2,
    "font_style": cv2.FONT_HERSHEY_DUPLEX,
    "font_scale_large": 1.0,
    "font_thickness": 1,
    "font_scale_small": 0.8,
}
