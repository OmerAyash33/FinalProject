"""
Runs the pipeline on the 10 videos that neither the detector (trained on 2026-01-30 + 2026-07-28 only) nor the event labels
(raw_footage/ only) have ever seen: 5 by day and 5 by night, picked 2026-11-14 from the archive
`New folder\\VideosByDate` (see the project memory "video-archive-unseen"). They are NOT copied: the pipeline reads them in place.

Outputs go to pipeline_outputs/unseen_day_<detector>/ and pipeline_outputs/unseen_night_<detector>/ (the session name `unseen_day` /
`unseen_night` also selects the shirt-colour rule: no picked reference -> OUR_SHADE_DEFAULT).

Usage:  .venv/Scripts/python.exe tools/run_unseen_set.py extract      # Stage 1 + audio cue (slow, ~4 h on the GPU, resumable)
        .venv/Scripts/python.exe tools/run_unseen_set.py analyze      # Stage 2 + selection + Stage 3 (needs the extraction)
        .venv/Scripts/python.exe tools/run_unseen_set.py stage2       # Stage 2 only (events CSVs; fast)
Every stage skips files that already exist, so it can be stopped and restarted.
"""
import glob
import os
import sys

os.environ.setdefault("FOOTBALL_LIVE_PREVIEW", "1")          # live view of the detections (drawn 6x a second, costs ~5% speed; FOOTBALL_LIVE_PREVIEW=0 turns it off, "q" in the window closes just the window)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch  # noqa: F401,E402  (before pandas, see main.py)
from config import DETECTOR_VERSION  # noqa: E402
from main import process_session  # noqa: E402

ARCHIVE = r"C:\Users\OmerAyash\PycharmProjects\New folder\VideosByDate"
SETS = {
    "unseen_night": [("2026-05-05", n) for n in ("20260505194324_000015", "20260505200324_000019", "20260505201109_000021",
                                                 "20260505205610_000030", "20260505210110_000031")],
    "unseen_day": [("2026-02-20", "20260220084805_000005"), ("2026-02-20", "20260220091505_000008"), ("2025-12-16", "20251216084606_000006"),
                   ("2025-10-31", "20251031091117_000006"), ("2025-10-06", "20251006133243_000045")],
}


def video_paths(name):
    return [os.path.join(ARCHIVE, d, f"{v}.MP4") for d, v in SETS[name]]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "extract"
    for name in SETS:
        vids = video_paths(name)
        missing = [v for v in vids if not os.path.exists(v)]
        if missing:
            sys.exit(f"missing videos: {missing}")
        out_dir = os.path.join(PROJECT_ROOT, "pipeline_outputs", f"{name}_{DETECTOR_VERSION}")
        print(f"\n{'#' * 60}\nSET: {name}  ({len(vids)} videos) -> {out_dir}\n{'#' * 60}", flush=True)
        if mode == "stage2":
            from tactical_analyzer import DataAnalyzer
            for v in vids:
                base = os.path.splitext(os.path.basename(v))[0]
                raw, ev = os.path.join(out_dir, f"{base}_data.csv"), os.path.join(out_dir, f"{base}_events.csv")
                if os.path.exists(raw) and not os.path.exists(ev):
                    DataAnalyzer(raw, ev).run_advanced_analytics()
            continue
        process_session(ARCHIVE, out_dir, video_files=vids, stages="extract" if mode == "extract" else "analyze")


if __name__ == "__main__":
    main()
