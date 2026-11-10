# Football chest-cam highlights

Takes chest-camera football footage and produces a highlight reel: dribbles (3 lengths), passes (completed / incomplete) and shots
(on / off target), each scored 0-100, the best 3 of every kind rendered with unified colours (goal dark blue, our team yellow,
opponents magenta, ball white).

## Pipeline

| Stage | File | What it does |
|---|---|---|
| 1 | `src/video_processor.py` | YOLO detect + track (players, ball, goal) -> `<video>_data.csv`; camera motion per frame |
| 1.5 | `src/audio_utils.py` | kick-sound cue from the audio track -> `<video>_audio.npz` |
| 2 | `src/tactical_analyzer.py`, `src/event_detection.py`, `src/event_scoring.py` | teams, holds, dribbles, passes, shots, scores, top-N selection |
| 3 | `src/highlight_generator.py` | renders the chosen events onto the original footage |

`src/main.py` runs everything for every session folder under `raw_footage/<session>/*.MP4` and writes to
`pipeline_outputs/<session>_<detector version>/`. **Every tunable number lives in `src/event_params.py`.**

## Setup

```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
python src/main.py
```

Needs an NVIDIA GPU for reasonable speed. The trained detectors are in `models/` (`match_detector_v2.pt` is the default,
see `src/config.py`). Footage, extracted data and rendered videos are NOT in the repository.

## The app (drop videos in, get highlights)

`src/app.py` is a desktop window: drag videos in, press start. It extracts the detections, then asks you to click a few players of
YOUR team (`src/team_color_picker.py`: left click = ours, right click = opponent, N = next player; the median of the clicked shirts
becomes the reference colour of the session, stored in `team_colors.json`, so any colour works - yellow, blue, red), then renders the
highlight videos. Without a picked colour the old light/dark rule is used (`event_params.OUR_SHADE_*`).

```
python src/app.py                  # run from source
python tools/build_exe.py          # -> dist/FootballHighlights/FootballHighlights.exe (several GB, copy the whole folder)
python src/team_color_picker.py <session folder in raw_footage>    # only the colour picker (needs extracted data)
```

## Measuring progress

`tools/review_tool/review_tool.py` is a desktop tool to label moments (pass completed / incomplete, shot, goal, dribble) while
watching the footage; labels go to `tools/review_tool/labels.csv`.
`tools/benchmark.py "note"` compares Stage 2 with those labels and appends a row to
`pipeline_outputs/_analysis/benchmark_history.csv`, so every change can be judged against the earlier ones. Labelled videos listed in
`HELDOUT_VIDEOS` are never to be tuned on.

## Status (2026-11)

Passes are found for ~36% of the labelled ones (81% of those with the right class); the audio+goal shot rule finds 4 of 8 labelled shots
but did not generalise to a second day, see `pipeline_outputs/_analysis/heldout_audit_2026-11-08.md`.
