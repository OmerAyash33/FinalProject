"""
Fine-tunes yolo11m.pt on match_dataset_v2 (built by tools/build_dataset_v2.py from
the re-reviewed Label Studio project): person + ball + goal, camera-wearer's hands
no longer labelled as people.

Augmentation: deliberately NOT pre-generated on disk. Ultralytics re-augments every
image on the fly each epoch (mosaic, horizontal flip 0.5, HSV jitter, scale/translate),
so the ~2,100 training images are seen as a different variant every epoch. The
defaults are kept on purpose:
  * no grayscale  - jersey colour is the signal the team clustering depends on, and
                    inference runs on colour frames
  * no vertical flip / rotation - the chest camera is always upright, and rotating
                    a box loosens it, which hurts the tiny ball most
Change one knob at a time against val if the ball is still the weak class.

imgsz=960 (v1 used 640): the pipeline runs inference at 960 wide (config.TARGET_WIDTH),
and at 960 the median ball is ~18px instead of ~12px (smallest decile 7.8px vs 5.2px).
Training at the scale the model is deployed at removes a train/inference mismatch.

The test split is NOT touched here. After picking a model on val, run once:
  .venv/Scripts/python.exe tools/evaluate_on_test.py <best.pt> datasets/match_dataset_v2/data.yaml

Usage:  .venv/Scripts/python.exe tools/train_model_v2.py
"""
import os
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(PROJECT_ROOT, "datasets", "match_dataset_v2", "data.yaml")
RUNS_DIR = os.path.join(PROJECT_ROOT, "training_runs")

if __name__ == "__main__":
    model = YOLO(os.path.join(PROJECT_ROOT, "yolo11m.pt"))  # COCO-pretrained, same start as v1 so the comparison isolates data + resolution
    model.train(
        data=DATA_YAML,
        epochs=100,
        patience=20,         # keeps best.pt (by val), not last.pt
        imgsz=960,
        batch=4,             # measured on the 8GB RTX 5070: batch=-1 crashes autobatch at 960, batch 6 needs 6.9GB, and batch 8 needs
                             # 8.9GB so it spills into shared system RAM (~6s/iteration). 4 uses 4.8GB at the same ~9 img/s.
        cache=False,         # not "ram": on Windows every loader worker would hold its own copy of the cache
        workers=2,           # Windows spawns `workers` train + 2x`workers` val loader processes (~0.6GB RAM each). The default 8 meant 24
                             # processes, ~15GB RAM and a 5 minute start. The 960px PNGs from build_dataset_v2.py decode ~10x faster than
                             # the 4K originals, so 2 workers keep up with the GPU.
        device=0,
        project=RUNS_DIR,
        name="match_detector_v2",
        amp=True,
        plots=True,
    )
