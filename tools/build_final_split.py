"""
Builds the final train/val/test split from the Label-Studio-corrected labels,
stratified so rare classes (ball, goal) are proportionally represented in
every split - a plain random split risks val/test ending up with few or no
goal examples, since goal is only ~550 boxes out of 2907 images.

Also fixes a real landmine: Label Studio's YOLO export always sorts class
names ALPHABETICALLY (ball=0, goal=1, person=2), ignoring the original
label config / hotkey order (1=person, 2=ball, 3=goal) used while
annotating. This remaps back to this project's scheme: 0=person, 1=ball,
2=goal - the same scheme used throughout the rest of the project.

Input:  datasets/match_dataset/ls_export_check/labels/*.txt  (Label Studio export, alphabetical classes)
        datasets/match_dataset/ls_import/images/*.jpg         (original images, untouched by Label Studio)
Output: datasets/match_dataset_final/images/{train,val,test}/*.jpg  (hardlinked, no duplication)
        datasets/match_dataset_final/labels/{train,val,test}/*.txt  (remapped class indices)
        datasets/match_dataset_final/data.yaml
"""
import os
import random
import glob

PROJECT_ROOT = r"C:\Users\OmerAyash\PycharmProjects\FinalProject"
EXPORTED_LABELS_DIR = os.path.join(PROJECT_ROOT, "datasets", "match_dataset", "ls_export_check", "labels")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "datasets", "match_dataset", "ls_import", "images")
OUT_DIR = os.path.join(PROJECT_ROOT, "datasets", "match_dataset_final")

# Label Studio exported classes.txt order -> this project's canonical order.
# Exported order was alphabetical: 0=ball, 1=goal, 2=person.
REMAP = {0: 1, 1: 2, 2: 0}  # old_class_index -> new_class_index (ball->1, goal->2, person->0)
CANONICAL_NAMES = {0: "person", 1: "ball", 2: "goal"}

SPLIT_RATIOS = {"train": 0.75, "val": 0.15, "test": 0.10}
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Load + remap every label file; bucket images by which rare classes they contain.
# ---------------------------------------------------------------------------
records = []  # {stem, remapped_lines, has_ball, has_goal}
for lbl_path in sorted(glob.glob(os.path.join(EXPORTED_LABELS_DIR, "*.txt"))):
    if os.path.basename(lbl_path) == "classes.txt":
        continue
    stem = os.path.splitext(os.path.basename(lbl_path))[0]
    remapped_lines = []
    has_ball, has_goal = False, False
    with open(lbl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_cls = int(parts[0])
            new_cls = REMAP[old_cls]
            remapped_lines.append(" ".join([str(new_cls)] + parts[1:]))
            if new_cls == 1:
                has_ball = True
            elif new_cls == 2:
                has_goal = True
    records.append({"stem": stem, "lines": remapped_lines, "has_ball": has_ball, "has_goal": has_goal})

print(f"Loaded {len(records)} labeled images.")

# ---------------------------------------------------------------------------
# Stratify: bucket by (has_ball, has_goal) combination, split each bucket
# independently so every split gets a proportional share of rare classes.
# ---------------------------------------------------------------------------
buckets = {}
for r in records:
    key = (r["has_ball"], r["has_goal"])
    buckets.setdefault(key, []).append(r)

split_assignment = {"train": [], "val": [], "test": []}
for key, items in buckets.items():
    random.shuffle(items)
    n = len(items)
    n_train = round(n * SPLIT_RATIOS["train"])
    n_val = round(n * SPLIT_RATIOS["val"])
    split_assignment["train"].extend(items[:n_train])
    split_assignment["val"].extend(items[n_train:n_train + n_val])
    split_assignment["test"].extend(items[n_train + n_val:])
    print(f"  bucket has_ball={key[0]} has_goal={key[1]}: {n} images -> "
          f"train {n_train}, val {n_val}, test {n - n_train - n_val}")

# ---------------------------------------------------------------------------
# Write out: hardlink images (zero extra disk), write remapped label files.
# ---------------------------------------------------------------------------
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)

for split, items in split_assignment.items():
    for r in items:
        src_img = os.path.join(IMAGES_DIR, r["stem"] + ".jpg")
        dst_img = os.path.join(OUT_DIR, "images", split, r["stem"] + ".jpg")
        if not os.path.exists(dst_img):
            os.link(src_img, dst_img)
        dst_lbl = os.path.join(OUT_DIR, "labels", split, r["stem"] + ".txt")
        with open(dst_lbl, "w") as f:
            f.write("\n".join(r["lines"]))

data_yaml = f"""path: {OUT_DIR}
train: images/train
val: images/val
test: images/test
names:
  0: person
  1: ball
  2: goal
"""
with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
    f.write(data_yaml)

# ---------------------------------------------------------------------------
# Report final per-split, per-class stats to confirm stratification worked.
# ---------------------------------------------------------------------------
print("\nFinal split sizes and class representation:")
for split, items in split_assignment.items():
    n_ball_imgs = sum(1 for r in items if r["has_ball"])
    n_goal_imgs = sum(1 for r in items if r["has_goal"])
    n_person_boxes = sum(1 for r in items for l in r["lines"] if l.startswith("0 "))
    n_ball_boxes = sum(1 for r in items for l in r["lines"] if l.startswith("1 "))
    n_goal_boxes = sum(1 for r in items for l in r["lines"] if l.startswith("2 "))
    print(f"  {split}: {len(items)} images | person boxes={n_person_boxes}, "
          f"ball boxes={n_ball_boxes} ({n_ball_imgs} imgs), goal boxes={n_goal_boxes} ({n_goal_imgs} imgs)")

print(f"\nWritten to {OUT_DIR}")
