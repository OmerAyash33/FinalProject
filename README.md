# Football chest-cam highlights

> ## Free for everyone
> **No cost, no account, no subscription, no ads.** Anyone can download it, run it on their own computer and use it for their own team, club or school.
> Your videos are processed on your own machine and are never uploaded anywhere. Open source under the [GNU AGPL-3.0](LICENSE).

Film a match with a **chest camera**, drop the videos in, and get a **highlight reel**: the best dribbles (3 lengths), passes (completed / incomplete)
and shots (on / off target), each scored 0-100, the top 3 of every kind rendered onto your own footage (goal dark blue, your team yellow, opponents
magenta, ball white) and finished with a short coach tip on the end card.

## Quick start (Windows)

**You need:** Windows 10/11, **Python 3.11 or newer**, about 6 GB of free disk space (PyTorch is large), and the videos of your match.
A graphics card is **not** required - see [Do I need a graphics card?](#do-i-need-a-graphics-card).

```
git clone https://github.com/OmerAyash33/FinalProject.git C:\football
cd C:\football
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
python src/app.py
```

The trained detector is included in the repository (`models/match_detector_v2.pt`, 40 MB), so there is nothing else to download.

- Clone into a **short folder** such as `C:\football`: Windows' 260-character path limit can make `pip` fail inside deeply nested folders.
- The `+cu128` PyTorch build also runs on a computer without an NVIDIA card (it simply uses the processor). On macOS, install `torch` and
  `torchvision` without the `+cu128` pin; macOS and Linux are untested (the code has fallbacks for them) - Windows is what it is built and tested on.

## Do I need a graphics card?

No. The program picks the fastest thing available on its own:

| Your computer | What it uses | Speed (measured, 1080p) | Quality |
|---|---|---|---|
| NVIDIA GPU (RTX 20-series or newer recommended) | PyTorch on the GPU, half precision, frames analysed at 960 px | about 40 frames/s: a 5-minute video takes about 4 minutes | full |
| No NVIDIA GPU | **Intel OpenVINO** on the processor (compiled automatically on the first run, a few seconds), frames analysed at 480 px | about 7-10 frames/s on a fast laptop CPU (i7-13650HX): about 15-20 minutes per 5-minute video, slower on a weaker CPU | the ball is found in about 3/4 as many frames and about 87% of the players, so fewer passes and shots are detected |
| No OpenVINO (unsupported processor) | plain PyTorch on the processor | slowest, but runs on any machine | same as above |

The slow part is always the first stage (detecting players, ball and goal); finding the events and rendering the highlights takes about a minute.
Leave it running - the window shows how many videos are done and about how long is left.

## Using the app

`python src/app.py` opens a window with five numbered steps:

1. **Add your videos** - drag them into the list or press "Add videos".
2. **Name the session and choose the output folder.**
3. **Shirt colour** - tick "pick our team's shirt colour (recommended)".
4. **Start processing.** The app runs three stages: (1) extract the detections (a live view shows what the detector sees; a bar and a line such as
   "Extracting: 2 of 5 videos done - about 11 min left" show the progress), (2) the colour picker, (3) find the events and render the highlights.
5. **Open your results** - clickable links to the highlight reel and the results folder appear when the run is finished.

The colour picker shows players one at a time: left click = ours, right click = opponent (optional), N = next player; at most 5 per colour, and after
7 clicks in total it saves and continues by itself. The median of the clicked shirts becomes the reference colour of the session (stored in
`team_colors.json`), so any colour works - yellow, blue, red. Without a picked colour the program guesses from light / dark shirts.

Box colours in the preview and in the rendered videos: goal dark blue, **ball white (the only white box)**, our team yellow, opponents magenta,
players without a team orange.

```
python src/app.py                  # the window
python src/main.py                 # no window: every session folder under raw_footage/<session>/*.MP4
python tools/build_exe.py          # -> dist/FootballHighlights/FootballHighlights.exe (several GB, copy the whole folder)
python src/team_color_picker.py <session folder in raw_footage>    # only the colour picker (needs extracted data)
```

## How it works

| Stage | File | What it does |
|---|---|---|
| 1 | `src/video_processor.py` | YOLO detect + track (players, ball, goal) -> `<video>_data.csv`; camera motion per frame |
| 1.5 | `src/audio_utils.py` | kick-sound cue from the audio track -> `<video>_audio.npz` |
| 2 | `src/tactical_analyzer.py`, `src/event_detection.py`, `src/event_scoring.py` | teams, holds, dribbles, passes, shots, scores, top-3 selection |
| 3 | `src/highlight_generator.py`, `src/coach.py` | renders the chosen events onto the original footage, with the coach tip on the end card |

Results go to `<output folder>/<session>_<detector version>/`. **Every tunable number lives in `src/event_params.py`.** The coach tip is built from
rules over what the camera saw - there is no cloud service and no AI chat behind it.

## Honest limits

This is a research-grade project, not a finished product. The ball is small and often hidden or blurred: it is seen in roughly 7-30% of frames, so
passes are found for about a third of the real ones, and a shot is often recognised by the kick sound rather than by the ball. Team assignment from
the shirt colour is the weakest link. Treat the output as "good moments to look at", not as match statistics. Details and numbers:
`pipeline_outputs/_analysis/progress_report_2026-11-14.md`.

## Measuring progress (for contributors)

`tools/review_tool/review_tool.py` is a desktop tool to label moments (pass completed / incomplete, shot, goal, dribble) while watching the footage;
labels go to `tools/review_tool/labels.csv`. `tools/benchmark.py "note"` compares Stage 2 with those labels and appends a row to
`pipeline_outputs/_analysis/benchmark_history.csv`, so every change can be judged against the earlier ones. Labelled videos listed in `HELDOUT_VIDEOS`
are never to be tuned on. Footage, extracted data and rendered videos are not in the repository.

## Privacy

Your videos never leave your computer: the program uploads nothing and needs no account. The detector library (Ultralytics) may send anonymous usage
statistics (never videos or frames) when there is an internet connection; switch that off with `yolo settings sync=False`.

## License

[GNU Affero General Public License v3.0](LICENSE). The detector is built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is
AGPL-3.0 as well, so the whole project is released under the same terms: use it, change it and share it freely, and keep it open when you share it.
