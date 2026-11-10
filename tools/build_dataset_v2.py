"""
Builds datasets/match_dataset_v2 directly from the re-reviewed Label Studio
project ("Football Match Labeling - Review 2"), reading the annotations
straight out of Label Studio's sqlite database - no zip export, so the
alphabetical-class-order landmine that build_final_split.py had to undo
cannot happen here: class ids come from the label names (person=0, ball=1,
goal=2).

Differences from build_final_split.py (which built match_dataset_final for v1):
  * The split is done per SOURCE VIDEO, not per frame. The dataset contains
    runs of near-identical frames (goal events were expanded into +-10 frames
    around each sighting), so a per-frame random split lets almost the same
    picture sit in train and test and inflates every metric. Whole videos stay
    together instead.
  * Both recording dates are forced into val and test, so lighting/venue
    differences are measured, not just memorised.
  * Rotated boxes (the goals were tilted to fit perspective) are converted to
    their axis-aligned bounding box, since YOLO detection has no rotation.
  * Images are pre-shrunk to IMAGE_WIDTH with the SAME resize the pipeline
    applies before detection (src/video_reader.py: cv2.INTER_NEAREST), so the
    model trains on the pixels it will actually see at inference. Ultralytics
    would otherwise shrink the 4K originals itself with a different filter.
    As a side effect each sample decodes ~10x faster, which is what lets
    training run with only a few loader processes (each costs ~1GB on Windows).

Input:  .label_studio_data/label_studio.sqlite3  (project LS_PROJECT_ID)
        datasets/match_dataset/ls_import/images/*.jpg  (4K originals, never modified)
Output: datasets/match_dataset_v2/{images,labels}/{train,val,test}, data.yaml,
        split_manifest.json   (images are lossless PNG copies at IMAGE_WIDTH)
Usage:  .venv/Scripts/python.exe tools/build_dataset_v2.py
"""
import collections
import json
import math
import os
import random
import re
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LS_DB = os.path.join(PROJECT_ROOT, ".label_studio_data", "label_studio.sqlite3")
LS_PROJECT_ID = 2
IMAGES_DIR = os.path.join(PROJECT_ROOT, "datasets", "match_dataset", "ls_import", "images")
OUT_DIR = os.path.join(PROJECT_ROOT, "datasets", "match_dataset_v2")

IMAGE_WIDTH = 960                          # keep equal to config.TARGET_WIDTH on CUDA and to imgsz in train_model_v2.py
RESIZE_INTERPOLATION = cv2.INTER_NEAREST   # keep equal to src/video_reader.py; switch both together if the reader ever changes

CLASS_IDS = {"person": 0, "ball": 1, "goal": 2}
SPLIT_RATIOS = {"train": 0.75, "val": 0.15, "test": 0.10}
RANDOM_SEED = 42
SEARCH_TRIALS = 50000
METRICS = ("images", "person", "ball", "goal")
STEM_RE = re.compile(r"^(?P<video>.+)_f(?P<frame>\d+)$")


def load_annotations():
    # Snapshot through the sqlite backup API so a running Label Studio can't hand us a half-written read.
    src = sqlite3.connect(f"file:{LS_DB}?mode=ro", uri=True)
    con = sqlite3.connect(":memory:")
    src.backup(con)
    src.close()
    rows = con.execute(
        "SELECT t.data, c.result FROM task_completion c JOIN task t ON t.id = c.task_id "
        "WHERE c.project_id = ? AND c.was_cancelled = 0 ORDER BY c.id", (LS_PROJECT_ID,)).fetchall()
    con.close()
    latest = {}  # one annotation per image; a later row (re-submitted) wins
    for data, result in rows:
        stem = re.search(r"images/(.+)\.jpg", json.loads(data)["image"]).group(1)
        latest[stem] = json.loads(result)
    return latest


def rect_to_yolo(item):
    """Label Studio rectangle (percent, rotated about its top-left corner) -> normalised YOLO xywh, or None."""
    v = item["value"]
    W, H = item["original_width"], item["original_height"]
    x, y = v["x"] / 100 * W, v["y"] / 100 * H
    w, h = v["width"] / 100 * W, v["height"] / 100 * H
    angle = math.radians(v.get("rotation") or 0)
    if angle:
        c, s = math.cos(angle), math.sin(angle)
        pts = [(x + dx * c - dy * s, y + dx * s + dy * c) for dx, dy in ((0, 0), (w, 0), (w, h), (0, h))]
        x1, x2 = min(p[0] for p in pts), max(p[0] for p in pts)
        y1, y2 = min(p[1] for p in pts), max(p[1] for p in pts)
    else:
        x1, y1, x2, y2 = x, y, x + w, y + h
    x1, x2 = max(0.0, x1), min(float(W), x2)
    y1, y2 = max(0.0, y1), min(float(H), y2)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H


annotations = load_annotations()
records = []
skipped_boxes = 0
for stem, items in sorted(annotations.items()):
    if not os.path.exists(os.path.join(IMAGES_DIR, stem + ".jpg")):
        raise SystemExit(f"Image missing for annotated task: {stem}")
    lines, counts = [], collections.Counter()
    for item in items:
        if item.get("type") != "rectanglelabels":
            continue
        name = item["value"]["rectanglelabels"][0]
        box = rect_to_yolo(item)
        if box is None:
            skipped_boxes += 1
            continue
        lines.append(f"{CLASS_IDS[name]} " + " ".join(f"{n:.6f}" for n in box))
        counts[name] += 1
    video = STEM_RE.match(stem)["video"]
    records.append({"stem": stem, "video": video, "lines": lines, "counts": counts})
print(f"Loaded {len(records)} annotated images from Label Studio project {LS_PROJECT_ID} "
      f"({skipped_boxes} degenerate boxes dropped).")

# ---------------------------------------------------------------------------
# Split by video: random search for the assignment whose per-split share of
# images / person / ball / goal boxes is closest to the target ratios.
# ---------------------------------------------------------------------------
videos = collections.defaultdict(lambda: collections.Counter())
for r in records:
    v = videos[r["video"]]
    v["images"] += 1
    for name, n in r["counts"].items():
        v[name] += n
totals = {m: sum(v[m] for v in videos.values()) for m in METRICS}
dates = {name: name[:8] for name in videos}
all_dates = set(dates.values())
names = sorted(videos)

rng = random.Random(RANDOM_SEED)
best_score, best_assign = float("inf"), None
for _ in range(SEARCH_TRIALS):
    order = names[:]
    rng.shuffle(order)
    assign, acc = {}, {"test": 0, "val": 0}
    for name in order:
        for split in ("test", "val"):
            if acc[split] < SPLIT_RATIOS[split] * totals["images"]:
                assign[name] = split
                acc[split] += videos[name]["images"]
                break
        else:
            assign[name] = "train"
    if any({dates[n] for n in names if assign[n] == s} != all_dates for s in ("val", "test")):
        continue
    score = 0.0
    for split, target in SPLIT_RATIOS.items():
        for m in METRICS:
            share = sum(videos[n][m] for n in names if assign[n] == split) / totals[m]
            score += abs(share - target) / target
    if score < best_score:
        best_score, best_assign = score, assign
if best_assign is None:
    raise SystemExit("No video split put both recording dates into val and test; add more videos or relax the rule.")

# ---------------------------------------------------------------------------
# Write out. Derived folders are wiped first so a re-run with a different
# assignment can never leave a stale copy of an image in a second split.
# ---------------------------------------------------------------------------
for sub in ("images", "labels"):
    shutil.rmtree(os.path.join(OUT_DIR, sub), ignore_errors=True)
for split in SPLIT_RATIOS:
    os.makedirs(os.path.join(OUT_DIR, "images", split))
    os.makedirs(os.path.join(OUT_DIR, "labels", split))

def write_image(r):
    img = cv2.imread(os.path.join(IMAGES_DIR, r["stem"] + ".jpg"))
    h, w = img.shape[:2]
    small = cv2.resize(img, (IMAGE_WIDTH, int(h * (IMAGE_WIDTH / float(w)))), interpolation=RESIZE_INTERPOLATION)
    dst = os.path.join(OUT_DIR, "images", best_assign[r["video"]], r["stem"] + ".png")
    if not cv2.imwrite(dst, small, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise SystemExit(f"Failed to write {dst}")


for r in records:
    with open(os.path.join(OUT_DIR, "labels", best_assign[r["video"]], r["stem"] + ".txt"), "w") as f:
        f.write("\n".join(r["lines"]))
with ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(write_image, records))

with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
    f.write(f"path: {OUT_DIR}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            "names:\n  0: person\n  1: ball\n  2: goal\n")
with open(os.path.join(OUT_DIR, "split_manifest.json"), "w") as f:
    json.dump({"source": f"label_studio_project_{LS_PROJECT_ID}", "seed": RANDOM_SEED,
               "video_to_split": dict(sorted(best_assign.items()))}, f, indent=2)

print(f"\nPer-split totals (split chosen per video, score {best_score:.2f}, lower = closer to {SPLIT_RATIOS}):")
for split in SPLIT_RATIOS:
    members = [n for n in names if best_assign[n] == split]
    row = {m: sum(videos[n][m] for n in members) for m in METRICS}
    share = {m: f"{row[m] / totals[m]:.0%}" for m in METRICS}
    print(f"  {split:5} {len(members):2} videos | images={row['images']} ({share['images']}) "
          f"person={row['person']} ({share['person']}) ball={row['ball']} ({share['ball']}) "
          f"goal={row['goal']} ({share['goal']})")
    print(f"        videos: {', '.join(members)}")
print(f"\nWritten to {OUT_DIR}")
