"""
Fine-tunes yolo11m.pt on the hand-corrected match dataset (person + ball + goal).

Uses datasets/match_dataset_final, built by tools/build_final_split.py from the
Label-Studio-corrected labels: a stratified 75/15/10 train/val/test split (not
a plain random split - ball and goal are rare enough that a random split could
easily starve val/test of them) with class indices remapped back to this
project's scheme (Label Studio's YOLO export alphabetizes class names, so its
raw output had them as 0=ball,1=goal,2=person - already corrected on disk).

The test split is intentionally held out here and NOT touched by this script -
it only gets evaluated once, after training is fully done and the val-based
model selection is finalized. See tools/evaluate_on_test.py for that step.

Usage:  .venv/Scripts/python.exe tools/train_model.py
"""
import os
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(PROJECT_ROOT, "datasets", "match_dataset_final", "data.yaml")
RUNS_DIR = os.path.join(PROJECT_ROOT, "training_runs")

if __name__ == "__main__":
    model = YOLO("yolo11m.pt")  # COCO-pretrained; we fine-tune its person/ball knowledge + teach it "goal" from scratch
    model.train(
        data=DATA_YAML,
        epochs=100,
        patience=20,         # stop early if val metrics haven't improved in 20 epochs; keeps best.pt, not last.pt
        imgsz=640,
        batch=-1,            # auto-sized to ~60% of free VRAM - safest default on an unfamiliar 8GB GPU
        device=0,            # the RTX 5070
        project=RUNS_DIR,
        name="match_detector_v1",
        amp=True,            # automatic mixed precision during training (separate from inference-time half=)
        plots=True,          # generates results.png, confusion_matrix.png, PR curves, etc.
    )
