"""
Final, one-time evaluation on the held-out test split.

Run this ONCE, after you're done looking at val metrics / training curves and
have picked a model you're happy with. The test split was never touched during
training or model selection (that's what val is for), so this is the only
genuinely unbiased number to report. Running it repeatedly and tuning based on
its result defeats the point - if you need to iterate, iterate against val.

Usage:  .venv/Scripts/python.exe tools/evaluate_on_test.py [path/to/best.pt] [path/to/data.yaml]
        (data.yaml defaults to match_dataset_final, i.e. the v1 split)
"""
import os
import sys
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(PROJECT_ROOT, "datasets", "match_dataset_final", "data.yaml")
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "training_runs", "match_detector_v1", "weights", "best.pt")

if __name__ == "__main__":
    weights = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WEIGHTS
    data_yaml = sys.argv[2] if len(sys.argv) > 2 else DATA_YAML
    print(f"Evaluating {weights} on the held-out TEST split of {data_yaml} (not seen during training or model selection)...")
    model = YOLO(weights)
    metrics = model.val(data=data_yaml, split="test", plots=True)
    print("\nPer-class results (precision, recall, mAP50, mAP50-95):")
    for i, name in metrics.names.items():
        print(f"  {name}: P={metrics.box.p[i]:.3f} R={metrics.box.r[i]:.3f} "
              f"mAP50={metrics.box.ap50[i]:.3f} mAP50-95={metrics.box.ap[i]:.3f}")
    print(f"\nOverall mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")
