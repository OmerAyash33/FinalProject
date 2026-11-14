"""STAGE 2 of the pipeline: turns the raw per-frame CSV into team assignments and match events (see main.py)."""
import os
import time

# torch must be imported before pandas on this Windows setup (see main.py), whichever module a script imports first.
import torch  # noqa: F401
import numpy as np
import pandas as pd

import event_params as P
from audio_utils import AudioTrack, audio_path_for
from ball_track import BallTrack
from event_detection import detect_events
from event_scoring import score_events
from logging_utils import log
from team_colors import load_reference
from team_utils import build_team_model, our_shade, session_of

# Only tracks at or above this length vote on where the two team colors
# actually are. ByteTrack churns IDs hard enough that median track length
# across a full session is only ~5 frames, and half of all tracks run under
# 5 frames - a 1-3 frame "median" color is really just raw per-frame noise.
# Letting every track_id into the K-means fit equally let those noisy points
# drag the cluster centers themselves off target; now short tracks are only
# classified against the fit (via predict()), never allowed to shape it.
MIN_FRAMES_FOR_TEAM_FIT = 8

EVENT_COLUMNS = ["event_type", "category", "start_frame", "end_frame", "score", "duration_frames", "speed", "length_px",
                 "team_margin", "pressure", "visibility", "overlap_frames", "overlap_fraction", "closeness_goal_widths", "kick_z",
                 "shot_origin", "shot_confidence", "reasoning"]


class DataAnalyzer:
    """
    Pure data-analysis stage - never touches video files. Reads the raw per-frame detections CSV and produces two things:
      * a `team_cluster` column written back into that same CSV: 0 = Team A = OUR team, 1 = Team B = the opponents
        (KMeans on jersey colour; which cluster is ours comes from the per-session light/dark setting in event_params);
      * `events_csv`: dribbles (3 length classes), passes (success / incomplete) and shots (on / off target), each with a
        0-100 score. The logic is in event_detection.py, every threshold in event_params.py.
    """

    def __init__(self, raw_csv, events_csv, fps=30.0):
        self.raw_csv = raw_csv
        self.events_csv = events_csv
        self.fps = fps

    def run_advanced_analytics(self):
        """Assign teams via KMeans, detect and score the events, and export both results."""
        if not os.path.exists(self.raw_csv):
            return
        log("Starting AI Coach Data Analysis...")
        ana_start_time = time.time()

        df = pd.read_csv(self.raw_csv, low_memory=False)
        df.drop_duplicates(inplace=True)
        # A re-run of just this stage (events_csv deleted, raw_csv left in place)
        # would otherwise read back a raw_csv that already has team_cluster from
        # the previous run, and the merge below would collide into
        # team_cluster_x/team_cluster_y instead of overwriting it.
        df.drop(columns=["team_cluster"], errors="ignore", inplace=True)
        players = df[df["type"] == "person"].copy()

        if players.empty:
            self._write_events([])
            return

        shade = our_shade(session_of(self.raw_csv))
        model = None

        # Only players ByteTrack actually assigned a stable ID can be clustered by track (their median colour denoises
        # single-frame lighting/motion blur), and only players big enough to show a jersey (see TEAM_MIN_BOX_HEIGHT_PX).
        color_cols = ["color_b", "color_g", "color_r"]
        tracked = players[(players["track_id"] != -1) & (players["height"] >= P.TEAM_MIN_BOX_HEIGHT_PX)
                          & (players["height"] >= P.PERSON_MIN_ASPECT * players["width"])]
        if len(tracked["track_id"].unique()) >= 2:
            track_lengths = tracked.groupby("track_id").size()
            stable_colors = tracked.groupby("track_id")[color_cols].median()

            reliable = stable_colors[track_lengths >= MIN_FRAMES_FOR_TEAM_FIT]
            # Fallback for a short/low-footage session where not even 2 tracks clear the reliability bar: fit on everything.
            fit_source = reliable if len(reliable) >= 2 else stable_colors

            log(f"DataAnalyzer: Starting KMeans clustering to detect teams "
                f"({len(fit_source)}/{len(stable_colors)} usable tracks used to fit)...")
            # Two jersey colours, no separate referee cluster. Short/noisy tracks never shape the fit, they only get a label.
            model = build_team_model(fit_source[color_cols].values, session_of(self.raw_csv))
            stable_colors["team_cluster"] = model.team(stable_colors[color_cols].values)
            df = df.merge(stable_colors[["team_cluster"]], on="track_id", how="left")
            if load_reference(session_of(self.raw_csv)) is not None:
                log("DataAnalyzer: our team colour = the shirt colour you picked for this session (team_colors.json): Team A (ours, yellow) = players "
                    "closest to it, Team B (magenta) = the others. Players too small or without a stable track stay unassigned (grey).")
            else:
                log(f"DataAnalyzer: our shade is '{shade}': Team A (ours, yellow) = the {shade} cluster, Team B (magenta) = the other. "
                    f"Players too small or without a stable track stay unassigned (grey).")
        else:
            df["team_cluster"] = np.nan  # fewer than 2 usable tracked players -> nothing to cluster

        df.to_csv(self.raw_csv, index=False)  # persist team_cluster back onto the raw dataset for later stages
        log("DataAnalyzer: Clustering complete. Detecting dribbles, passes and shots...")

        track = BallTrack(df)
        log(f"DataAnalyzer: ball seen in {track.raw_sightings} frames ({track.dropped_jumps} jumps dropped), "
            f"camera motion {'available' if track.has_camera else 'MISSING (old CSV): speeds include camera swings'}.")
        if model is None:
            log("DataAnalyzer: no team clustering possible -> only dribbles and shots can be judged, not passes.")
        audio = AudioTrack.load(audio_path_for(self.raw_csv)) if P.AUDIO_ENABLED else None
        log("DataAnalyzer: kick-sound cue " + ("available." if audio is not None else "NOT available (no <video>_audio.npz: shots seen only by ball speed)."))
        events = detect_events(df, track, model, self.fps, audio=audio)
        events = score_events(events)
        self._write_events(events)
        counts = pd.Series([e["event_type"] for e in events]).value_counts().to_dict()
        log(f"Analysis completed in {time.time() - ana_start_time:.1f} seconds. Exported {len(events)} events: {counts}")

    def _write_events(self, events):
        out = pd.DataFrame(events, columns=EVENT_COLUMNS)
        out.to_csv(self.events_csv, index=False)
