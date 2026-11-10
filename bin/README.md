# Archive — safe-to-ignore, kept only for recovery

Everything here was removed from the live project during the 2026-09-25 cleanup.
Nothing here is needed to run `src/main.py`. Kept only in case something is
needed later — delete this whole folder once you're confident you won't.

## old_code/newMain050826.py
The older, more feature-rich pipeline (goal/shot/turnover detection, awareness
score, radar overlay) that used to live in `final project 18.8/`. It was
**not runnable as-is**: it requires `ball_detection.pt` and
`bytetrack_custom.yaml`, neither of which exist anywhere in the project (not
even in the old venvs) — they likely only exist on a teammate's machine
("benbe", per the hardcoded OneDrive path in its `main()`). `src/main.py`
(formerly `stabilzer.py`) was chosen as the canonical pipeline instead because
it actually runs today. If you recover the missing model + tracker config,
this file has meaningfully more tactical intelligence and is worth merging
back in.

## old_code/stabilization.py
`VideoStabilizer` — the optical-flow camera-stabilization stage that used to
run before YOLO tracking (added 2026-09-26, when `src/main.py` was split into
one file per stage). Removed from the pipeline after a real measurement
(15sec.mp4, 478 frames) showed it was a net negative: it cut YOLO's
ball-detection recall by ~22% (73 -> 57 frames with any detection) with only
a marginal confidence gain on frames it did catch (0.252 -> 0.268) — because
`cv2.warpAffine` without cropping pushes real frame content into black
undefined borders, and per this project's own footage that can wipe out a
large fraction of the screen on a hard camera jolt. It did meaningfully help
person-tracking continuity (127 -> 93 unique ByteTrack IDs, i.e. fewer ID
switches), but not enough to be worth the ball-detection cost. If you want
stabilization back without the black-border problem, the older
`newMain050826.py` (below) crops a 5% margin and resizes back before writing
each frame — that fix was never ported to this version.

## old_code/pitch_detector.py
`PitchDetector` — the HSV-color grass detector that used to gate YOLO (skip
frames judged "not looking at the pitch") and tag each detection with an
`in_pitch` column. Removed 2026-09-26 at the user's request: every frame is
now processed and logged, no exceptions, so the tactical state machine gets a
complete timeline. Also, `in_pitch` was never actually read by
`tactical_analyzer.py` or `highlight_generator.py` — it was pure dead weight
even before this change.

## superseded_models/goal_detection_best.pt, goal_detection_full_checkpoint.pt
The old two-model setup (a generic COCO person/ball model + this separately-
trained, domain-mismatched goal model) - replaced 2026-10-25 by
`models/match_detector_v1.pt`, one model fine-tuned on 2907 hand-corrected
frames from our own two chest-cam sessions, covering person+ball+goal in a
single YOLO call per frame. Test-set results: mAP50=0.779, mAP50-95=0.591
overall (person 0.921/0.728, ball 0.713/0.533, goal 0.702/0.510). Kept here in
case the new model needs to be compared back against the old one.

## duplicate_weights/goal_detection_duplicate_of_best.pt
Byte-identical (MD5-verified) to `models/goal_detection_best.pt`. It was
present under two different names in two different folders; this is just the
redundant copy.

## incomplete_training_runs/goal_seg_v1/
A YOLO training run that only ever produced `args.yaml` — it was aborted
before any weights or metrics were saved. Kept only as a record that this
config was tried.

## lost_bytecode/newMain090826.cpython-313.pyc
Compiled bytecode for a script named `newMain090826.py` that no longer exists
anywhere as source — apparently an intermediate version between
`newMain050826.py` and `stabilzer.py` that was deleted before being committed
anywhere. The `.py` source is gone; only this compiled `.pyc` survives.
Recovering readable source from it would require a Python 3.13 bytecode
decompiler (e.g. `decompyle3`/`pycdc`) and won't be perfect (comments and
exact formatting are lost), but it may still be worth it if this
intermediate version had features neither of the other two files have.

## Not kept here (deleted outright, not archived)
- Both old virtual environments (`.venv/`, `final project 18.8/venv/`, ~2.1GB
  combined) — 100% reproducible via `pip install -r requirements.txt`.
- `final project 18.8/yolo11n.pt` — stock Ultralytics YOLO11-nano checkpoint,
  auto-downloads on first run of any script that calls `YOLO("yolo11n.pt")`.
- `__pycache__/` folders and `labels.cache` — regenerated automatically.
- An empty `videos/` folder at the project root that was never used.
