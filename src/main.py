"""
Orchestrator for the football chest-cam analysis pipeline.

Full pipeline, one file per stage:
  1. video_processor.py     - VideoProcessor:     YOLO detect/track -> raw CSV
  2. tactical_analyzer.py   - DataAnalyzer:        team clustering + event state machine
  3. highlight_generator.py - HighlightGenerator:  render events onto the original footage
  4. (this file)            - loop over every session under raw_footage/, mirror outputs
                               into pipeline_outputs/, then stitch a per-session master reel

Project layout this relies on (see project root):
  raw_footage/<session>/*.MP4        - camera originals ONLY, never written to by this pipeline
  pipeline_outputs/<session>_<ver>/  - everything this pipeline generates for that session, where
                                        <ver> is config.DETECTOR_VERSION (e.g. 2026-05-05_evening_v2),
                                        so outputs of different detector versions sit side by side
                                        and never reuse each other's CSVs. (Folders without a
                                        version suffix are older runs made before the switch existed:
                                        they were produced by the v1 detector.)
Raw input and generated output are kept in physically separate trees on purpose, so a glob
over raw_footage/ can never accidentally pick up this pipeline's own prior output as if it
were new footage (that used to require a manual "skip anything with 'highlights' in the name"
filter - a real bug magnet, since Windows path matching is case-insensitive and once silently
matched both *.MP4 and *.mp4 in the same glob). With the trees separated, that filter is gone;
there is simply nothing to filter.

There used to be a Stage 0 optical-flow stabilizer before Stage 1 (still kept,
unused, at bin/old_code/stabilization.py). It was removed after measurement
showed it was a net negative: it cut YOLO's ball-detection recall by ~22%
(warping the frame without cropping pushes real content into black borders -
sometimes covering a large fraction of the screen), for only a partial gain
in person-tracking ID stability. Stage 1 now runs directly on the source
video.

There also used to be a grass/pitch validity check that skipped YOLO
entirely on frames judged to be "not looking at the pitch" (still kept,
unused, at bin/old_code/pitch_detector.py). It was removed because every
frame is now wanted for a gap-free state-machine timeline, and the one piece
of data it fed into the CSV (`in_pitch`) was never actually read by any
later stage anyway.
"""
import os
import glob
import time
import warnings

# Imported first and explicitly, before anything else in this process ever gets a
# chance to import pandas: on this Windows setup, pandas-before-torch reliably
# breaks torch's native DLL init (c10.dll). video_processor.py and
# highlight_generator.py also guard their own internal import order for the same
# reason, but this is the one guarantee that holds no matter how main.py's own
# import list is reordered later.
import torch  # noqa: F401

import event_params as P
from audio_utils import audio_path_for, extract_audio_features
from config import DETECTOR_VERSION, LIVE_PREVIEW
from event_scoring import select_events
from logging_utils import log
from video_processor import VideoProcessor
from tactical_analyzer import DataAnalyzer
from highlight_generator import HighlightGenerator, concatenate_highlights

import pandas as pd

# Pandas FutureWarnings are noisy and not actionable here; silenced to keep the console log readable.
warnings.filterwarnings("ignore", category=FutureWarning)
# Works around an OpenMP runtime conflict between OpenCV/PyTorch that otherwise crashes on some machines.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def process_session(input_dir, output_dir, video_files=None, stages="all"):
    """
    Push every video in `input_dir` (a single raw_footage/<session> folder)
    through all three stages, writing every generated file into `output_dir`
    (the pipeline_outputs/<session>_<detector version> folder) instead of next to the
    source video.

    Stages 1 and 2 run for EVERY video first. Only then are the events of all videos
    ranked against each other and the best TOP_N_PER_EVENT of each kind chosen
    (event_params.SELECTION_SCOPE says whether "against each other" means the whole
    session or each video on its own), and Stage 3 renders just those. The choice is
    saved to <output_dir>/selected_events.csv.

    Idempotent by design: every intermediate/output file is named
    deterministically from the source video's own filename, and each stage
    is skipped if its output file already exists. This means:
      - Re-running on a session that's already been processed does no
        redundant work — it just re-collects the existing outputs.
      - Forcing a specific video (or a specific stage) to re-run is just a
        matter of deleting that one output file first; the re-run then
        writes to that exact same path again (never a new/renamed file),
        so the previous and new outputs can be diffed to check determinism.
      - A crash partway through (e.g. stage 2 fails) can be resumed without
        re-doing the already-finished stage(s) for that video.
    After changing event_params.py, delete the *_events.csv files (and the
    *_highlights.mp4 of the videos whose selection changed) to apply it.

    `video_files`: an explicit list of video paths (any folder, any extension) instead of every *.MP4 of `input_dir`.
    `stages`: "all" (default), "extract" (Stage 1 + the audio cue only) or "analyze" (Stage 2, the selection and Stage 3 for videos that
    are already extracted). The packaged app (src/app.py) runs "extract", lets the user pick the team colour from the extracted
    detections, then runs "analyze".
    """
    os.makedirs(output_dir, exist_ok=True)
    video_files = list(video_files) if video_files else sorted(glob.glob(os.path.join(input_dir, "*.MP4")))

    if not video_files:
        print(f"No videos found in {input_dir}")
        return

    # ---- Stages 1 + 2 for every video (or only one of them, see `stages`) ----
    paths = {}
    for video_path in video_files:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        raw_csv = os.path.join(output_dir, f"{base_name}_data.csv")
        events_csv = os.path.join(output_dir, f"{base_name}_events.csv")
        highlights_vid = os.path.join(output_dir, f"{base_name}_highlights.mp4")
        paths[base_name] = (video_path, raw_csv, events_csv, highlights_vid)

        print(f"\n{'=' * 50}\nProcessing Video: {base_name}\n{'=' * 50}")
        video_start_time = time.time()
        if not os.path.exists(raw_csv):
            if stages == "analyze":
                log(f"{base_name}: no extracted data (run the extraction first), skipping this video.")
                paths.pop(base_name)
                continue
            VideoProcessor(video_path, raw_csv, show_preview=LIVE_PREVIEW).process_video(chunk_seconds=5)
        else:
            log(f"{base_name}: raw data CSV already exists, skipping extraction (Stage 1).")

        audio_npz = audio_path_for(raw_csv)
        if P.AUDIO_ENABLED and not os.path.exists(audio_npz):   # kick-sound cue for shots (small .npz next to the CSV, so Stage 2 never needs the video)
            log(f"{base_name}: extracting the kick-sound cue from the audio track..." if extract_audio_features(video_path, audio_npz)
                else f"{base_name}: no audio track or PyAV missing (pip install av): kick-sound cue skipped.")

        if stages == "extract":
            log(f"Video {base_name}: extraction done in {time.time() - video_start_time:.1f} seconds.")
            continue

        if not os.path.exists(events_csv):
            DataAnalyzer(raw_csv, events_csv).run_advanced_analytics()
        else:
            log(f"{base_name}: events CSV already exists, skipping tactical analysis (Stage 2).")
        log(f"Video {base_name}: stages 1-2 done in {time.time() - video_start_time:.1f} seconds.")

    if stages == "extract":
        return

    # ---- choose the best events across the session (or per video) ----
    per_video = []
    for base_name, (_, _, events_csv, _) in paths.items():
        if os.path.exists(events_csv):
            ev = pd.read_csv(events_csv)
            if not ev.empty:
                ev.insert(0, "video", base_name)
                ev.insert(1, "pool", base_name if P.SELECTION_SCOPE == "video" else "session")
                per_video.append(ev)
    selected = select_events(pd.concat(per_video, ignore_index=True)) if per_video else pd.DataFrame()
    selected.to_csv(os.path.join(output_dir, "selected_events.csv"), index=False)
    log(f"Selection ({P.SELECTION_SCOPE} scope, top {P.TOP_N_PER_EVENT} per event kind): {len(selected)} events chosen"
        + (f": {selected['event_type'].value_counts().to_dict()}" if len(selected) else "."))

    # ---- Stage 3: render only the chosen events ----
    generated_highlights = []
    for base_name, (video_path, raw_csv, events_csv, highlights_vid) in paths.items():
        chosen = selected[selected["video"] == base_name] if len(selected) else selected
        if chosen.empty:
            continue
        if os.path.exists(highlights_vid):
            # Already rendered in a previous run - delete it yourself first if you want this video redone.
            log(f"{base_name}: highlights already exist, skipping render.")
        else:
            HighlightGenerator(video_path, raw_csv, events_csv, highlights_vid, events_df=chosen).create_highlights()
        if os.path.exists(highlights_vid):
            generated_highlights.append(highlights_vid)

    master_output = os.path.join(output_dir, "All_Matches_Master_Highlights.mp4")
    concatenate_highlights(generated_highlights, master_output)


def main():
    """
    Find every session subfolder under `raw_footage/`, and process each one
    into its `<session>_<detector version>` subfolder under `pipeline_outputs/`. Both roots are
    resolved relative to this file's own location, not hardcoded to one
    machine/user - so this runs correctly regardless of whose computer or
    which clone of the repo it's launched from. Drop a new session folder
    (a full match day, an hour of footage, whatever) under raw_footage/ and
    it gets picked up automatically on the next run; no code change needed.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_footage_root = os.path.join(project_root, "raw_footage")
    outputs_root = os.path.join(project_root, "pipeline_outputs")

    session_dirs = sorted(d for d in glob.glob(os.path.join(raw_footage_root, "*")) if os.path.isdir(d))
    if not session_dirs:
        print(f"No session folders found under {raw_footage_root}")
        return

    for session_dir in session_dirs:
        session_name = os.path.basename(session_dir)
        print(f"\n{'#' * 60}\nSESSION: {session_name}\n{'#' * 60}")
        process_session(session_dir, os.path.join(outputs_root, f"{session_name}_{DETECTOR_VERSION}"))


if __name__ == "__main__":
    main()
