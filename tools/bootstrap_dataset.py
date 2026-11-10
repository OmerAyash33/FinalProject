"""
Builds/extends a bootstrap-labeled YOLO dataset (person + ball + goal) from a
raw_footage/<session> folder, to be hand-corrected in labelImg before training.
Re-run this once per session (change SESSION_NAME below) to keep adding more
sessions into the same datasets/match_dataset - it never touches or re-exports
images already written for a different session (filenames are prefixed by
their own source video name, so there's no collision risk).

Three-source design per session:
  1. GOAL candidates:
     - If the session already has production `_data.csv` files (a prior real
       pipeline run) in pipeline_outputs/<session>/, pull exact frame numbers
       straight from wherever type=="goal" was logged - goal sightings are
       rare and a coarse time-stride scan has poor odds of landing on one by
       chance, so for this class we go straight to known-positive frames.
     - Otherwise (a brand new session with no prior pipeline run), the scan
       pass below also runs the goal model on every candidate and the same
       confidence-based event selection is applied directly to scan results.
     Either way, each distinct goal EVENT is expanded into GOAL_FRAMES_PER_EVENT
     nearby frames (not just the single best one), for more real diversity
     without inventing new non-existent sightings.
  2. BALL-event candidates: a stride-sampled scan across the session, kept
     only where the existing person/ball model is already fairly confident -
     these are easy, fast-to-confirm positives.
  3. UNIFORM candidates: evenly spaced across the whole session, for general
     diversity (poses, formations, true "no ball visible" negatives) - and,
     incidentally, the best chance of catching a ball the model MISSED
     entirely, since those frames aren't selected by any confidence filter.

Only genuine camera recordings are ever read from raw_footage/<session>/ -
that folder holds nothing else by construction (see project layout note in
src/main.py), so there is no risk of this script training on its own
pipeline's previously-rendered output.

Output: datasets/match_dataset/images/{train,val}/*.jpg + labels/{train,val}/*.txt
        datasets/match_dataset/data.yaml
Unified class scheme (remapped from the two source models):
  0 = person   (was COCO class 0)
  1 = ball     (was COCO class 32)
  2 = goal     (was goal_detection_best.pt class 1)
"""
import os
import sys
import glob
import random
import json
import csv as csv_module

import cv2

PROJECT_ROOT = r"C:\Users\OmerAyash\PycharmProjects\FinalProject"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from yolo_utils import load_yolo_model
from config import GOAL_MODEL_PATH, GOAL_MODEL_OPENVINO_DIR, GOAL_CLASS_ID, GPU_MODEL_PATH

# --- change these two per run; everything else is session-agnostic ---
SESSION_NAME = "2026-01-30"
HAS_PRODUCTION_CSV = False   # True only if pipeline_outputs/<session>/*_data.csv already exists

VIDEO_DIR = os.path.join(PROJECT_ROOT, "raw_footage", SESSION_NAME)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "pipeline_outputs", SESSION_NAME)
OUT_DIR = os.path.join(PROJECT_ROOT, "datasets", "match_dataset")

SCAN_STRIDE_SEC = 1.5
TARGET_UNIFORM = 1000
TARGET_BALL_EVENTS = 250
GOAL_FRAMES_PER_EVENT = 5
GOAL_EVENT_OFFSETS = [-10, -5, 0, 5, 10]  # frames (~0.33s spread @ 30fps) - close enough the goal stays in similar framing
MIN_EVENT_SPACING_SEC = 4.0
VAL_FRACTION = 0.15
SCAN_CONF_PERSON_BALL = 0.15
SCAN_CONF_GOAL = 0.10
EXPORT_CONF_PERSON_BALL = 0.15
EXPORT_CONF_GOAL = 0.10
RANDOM_SEED = 42

video_paths = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.MP4")))
print(f"Session '{SESSION_NAME}': {len(video_paths)} source clips in {VIDEO_DIR}")

print("Loading models...")
person_ball_model, pb_fp16 = load_yolo_model(gpu_model_path=GPU_MODEL_PATH, task='detect')
goal_model, goal_fp16 = load_yolo_model(cpu_model_path=GOAL_MODEL_PATH, gpu_model_path=GOAL_MODEL_PATH,
                                        ov_dir=GOAL_MODEL_OPENVINO_DIR, task='segment')


def dedup_by_spacing(items, time_key, spacing):
    items_sorted = sorted(items, key=lambda c: c.get("conf", 0), reverse=True)
    picked, picked_keys = [], []
    for c in items_sorted:
        key = (c["video"], c[time_key])
        if any(c["video"] == v and abs(c[time_key] - t) < spacing for v, t in picked_keys):
            continue
        picked.append(c)
        picked_keys.append(key)
    return picked


# ---------------------------------------------------------------------------
# SOURCE 1: goal events, either from existing production CSVs or from the scan below
# ---------------------------------------------------------------------------
goal_events = []
if HAS_PRODUCTION_CSV:
    goal_candidates = []
    for video_path in video_paths:
        base = os.path.splitext(os.path.basename(video_path))[0]
        data_csv = os.path.join(OUTPUTS_DIR, f"{base}_data.csv")
        if not os.path.exists(data_csv):
            continue
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        with open(data_csv, newline="") as f:
            for row in csv_module.DictReader(f):
                if row.get("type") == "goal":
                    # VideoProcessor's frame_counter is 1-indexed (incremented before first use);
                    # cv2.CAP_PROP_POS_FRAMES is 0-indexed - off by one here lands one frame AFTER
                    # the real detection, which is often a single-frame flicker that doesn't persist.
                    frame_idx = int(float(row["frame"])) - 1
                    goal_candidates.append({"video": video_path, "frame_idx": frame_idx, "n_frames": n_frames,
                                            "t_video": frame_idx / fps, "conf": float(row.get("confidence", 0))})
    goal_events = dedup_by_spacing(goal_candidates, "t_video", MIN_EVENT_SPACING_SEC)
    print(f"Goal candidates from existing production CSVs: {len(goal_candidates)} "
          f"-> {len(goal_events)} distinct events")
# else: filled in from the scan pass below, since there's no shortcut CSV yet for this session.

# ---------------------------------------------------------------------------
# SOURCE 2 & 3 (+ goal scan when SOURCE 1 had no CSV): stride-sampled scan
# ---------------------------------------------------------------------------
candidates = []
t_global = 0.0
for vid_idx, video_path in enumerate(video_paths):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride_frames = max(1, int(round(fps * SCAN_STRIDE_SEC)))

    print(f"[{vid_idx+1}/{len(video_paths)}] Scanning {os.path.basename(video_path)} "
          f"({n_frames} frames, stride {stride_frames})...")

    for frame_idx in range(0, n_frames, stride_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        pb_results = person_ball_model.predict(frame, classes=[0, 32], conf=SCAN_CONF_PERSON_BALL,
                                               imgsz=640, verbose=False, half=pb_fp16)
        ball_conf = 0.0
        if len(pb_results) > 0 and pb_results[0].boxes is not None:
            for b in pb_results[0].boxes:
                if int(b.cls[0]) == 32:
                    ball_conf = max(ball_conf, float(b.conf[0]))

        goal_conf = 0.0
        if not HAS_PRODUCTION_CSV:
            goal_results = goal_model.predict(frame, classes=[GOAL_CLASS_ID], conf=SCAN_CONF_GOAL,
                                              imgsz=640, verbose=False, half=goal_fp16)
            if len(goal_results) > 0 and goal_results[0].boxes is not None:
                for b in goal_results[0].boxes:
                    goal_conf = max(goal_conf, float(b.conf[0]))

        candidates.append({"video": video_path, "frame_idx": frame_idx, "n_frames": n_frames,
                           "t_global": t_global + frame_idx / fps, "t_video": frame_idx / fps,
                           "ball_conf": ball_conf, "goal_conf": goal_conf})
    t_global += n_frames / fps
    cap.release()

print(f"\nScan complete: {len(candidates)} candidate frames. "
      f"with ball>=0.3 conf: {sum(1 for c in candidates if c['ball_conf'] >= 0.3)}"
      + ("" if HAS_PRODUCTION_CSV else f", with goal>=0.3 conf: {sum(1 for c in candidates if c['goal_conf'] >= 0.3)}"))

if not HAS_PRODUCTION_CSV:
    goal_candidates_from_scan = [{**c, "conf": c["goal_conf"]} for c in candidates if c["goal_conf"] >= 0.3]
    goal_events = dedup_by_spacing(goal_candidates_from_scan, "t_video", MIN_EVENT_SPACING_SEC)
    print(f"Goal events found directly in scan: {len(goal_events)}")

if len(goal_events) < 15:
    print("  [NOTE] Very few distinct goal sightings exist in this footage (chest-cam rarely frames "
        "the goal). The 'goal' class will stay sparse from this session alone.")

selected = {}
ball_events = [c for c in candidates if c["ball_conf"] >= 0.3]
ball_events = dedup_by_spacing([{**c, "conf": c["ball_conf"]} for c in ball_events],
                               "t_global", MIN_EVENT_SPACING_SEC)[:TARGET_BALL_EVENTS]
for c in ball_events:
    selected[(c["video"], c["frame_idx"])] = c
print(f"Ball-confident picks: {len(ball_events)}")

if candidates:
    span = candidates[-1]["t_global"] - candidates[0]["t_global"]
    step = span / TARGET_UNIFORM
    next_t = candidates[0]["t_global"]
    for c in candidates:
        key = (c["video"], c["frame_idx"])
        if c["t_global"] >= next_t and key not in selected:
            selected[key] = c
            next_t += step

goal_frames_added = 0
for c in goal_events:
    for offset in GOAL_EVENT_OFFSETS[:GOAL_FRAMES_PER_EVENT]:
        frame_idx = min(max(c["frame_idx"] + offset, 0), c["n_frames"] - 1)
        key = (c["video"], frame_idx)
        if key not in selected:
            selected[key] = {**c, "frame_idx": frame_idx}
            goal_frames_added += 1

final_selection = list(selected.values())
print(f"\nFinal dataset size for this session: {len(final_selection)} images "
      f"({goal_frames_added} goal frames from {len(goal_events)} events x up to {GOAL_FRAMES_PER_EVENT} offsets, "
      f"{len(ball_events)} ball-event, rest uniform).")

# ---------------------------------------------------------------------------
# EXPORT: re-seek each selected frame, run fresh full inference, save image + label
# ---------------------------------------------------------------------------
random.seed(RANDOM_SEED)
random.shuffle(final_selection)
n_val = int(round(len(final_selection) * VAL_FRACTION))
val_ids = set(id(c) for c in final_selection[:n_val])

manifest_path = os.path.join(OUT_DIR, "manifest.json")
manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else []
caps_cache = {}
for i, c in enumerate(final_selection):
    split = "val" if id(c) in val_ids else "train"
    if c["video"] not in caps_cache:
        caps_cache[c["video"]] = cv2.VideoCapture(c["video"])
    cap = caps_cache[c["video"]]
    cap.set(cv2.CAP_PROP_POS_FRAMES, c["frame_idx"])
    ret, frame = cap.read()
    if not ret:
        print(f"  [WARN] could not re-read frame {c['frame_idx']} of {c['video']}, skipping.")
        continue
    h, w = frame.shape[:2]

    boxes = []
    pb_results = person_ball_model.predict(frame, classes=[0, 32], conf=EXPORT_CONF_PERSON_BALL,
                                           imgsz=640, verbose=False, half=pb_fp16)
    if len(pb_results) > 0 and pb_results[0].boxes is not None:
        for b in pb_results[0].boxes:
            unified_cls = 0 if int(b.cls[0]) == 0 else 1
            boxes.append((unified_cls, *b.xyxy[0].tolist()))
    goal_results = goal_model.predict(frame, classes=[GOAL_CLASS_ID], conf=EXPORT_CONF_GOAL,
                                      imgsz=640, verbose=False, half=goal_fp16)
    if len(goal_results) > 0 and goal_results[0].boxes is not None:
        for b in goal_results[0].boxes:
            boxes.append((2, *b.xyxy[0].tolist()))

    stem = f"{os.path.splitext(os.path.basename(c['video']))[0]}_f{c['frame_idx']:06d}"
    img_path = os.path.join(OUT_DIR, "images", split, f"{stem}.jpg")
    lbl_path = os.path.join(OUT_DIR, "labels", split, f"{stem}.txt")
    cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    lines = []
    for (cls, x1, y1, x2, y2) in boxes:
        xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    with open(lbl_path, "w") as f:
        f.write("\n".join(lines))

    manifest.append({"image": img_path, "label": lbl_path, "split": split, "session": SESSION_NAME,
                     "n_boxes": len(boxes), "n_goal_boxes": sum(1 for b in boxes if b[0] == 2)})
    if (i + 1) % 50 == 0:
        print(f"  exported {i+1}/{len(final_selection)}...")

for cap in caps_cache.values():
    cap.release()

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

data_yaml = f"""path: {OUT_DIR}
train: images/train
val: images/val
names:
  0: person
  1: ball
  2: goal
"""
with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
    f.write(data_yaml)

n_train_total = sum(1 for m in manifest if m["split"] == "train")
n_val_total = sum(1 for m in manifest if m["split"] == "val")
n_this_session = len(final_selection)
n_goal_imgs_this_session = sum(1 for m in manifest if m.get("session") == SESSION_NAME and m["n_goal_boxes"] > 0)
print(f"\nDone. Added {n_this_session} images from session '{SESSION_NAME}'.")
print(f"Dataset totals now: {n_train_total} train / {n_val_total} val / {n_train_total + n_val_total} overall.")
print(f"This session's images containing at least one goal box: {n_goal_imgs_this_session}")
print("Boxes are BOOTSTRAP ONLY (auto-detected) - review/fix in labelImg before training.")
