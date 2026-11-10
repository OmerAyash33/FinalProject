"""
Measures, on real Stage 1 output, the quantities the event thresholds in src/event_params.py are chosen from, and
prints them. Run it after any detector change (or on a new kind of footage) and compare with the constants there.

What it reports, per session folder given (default: the two May sessions, footage the detector never trained on):
  1. ball visibility: share of frames with a ball, run lengths, gap lengths
  2. holds (ball in the self zone): how many per video, duration quantiles -> suggested short/medium/long dribble cut-offs
  3. what happens when a hold ends: scene-relative ball speed, whether the ball is seen again, whether a receiver is
     found, how clear his team is -> how many flights survive each threshold
  4. kicks that look like shots: speed above the shot threshold with a goal in view

Usage:  .venv/Scripts/python.exe tools/analyze_event_params.py [<session_dir_name> ...]   (folders under pipeline_outputs/)
"""
import glob
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch  # noqa: F401  (before pandas, see main.py)
import numpy as np
import pandas as pd

import event_params as P
from ball_track import BallTrack
from event_detection import detect_events, Scene, find_holds, _end_of_flight, _receiver, _team_vote
from team_utils import TeamModel, our_shade, session_of

DEFAULT = ["2026-05-01_morning_v2", "2026-05-05_evening_v2"]


def q(x, qs=(.1, .25, .5, .75, .9, .99)):
    x = np.asarray([v for v in x if np.isfinite(v)])
    return np.round(np.quantile(x, qs), 2).tolist() if len(x) else []


def count_events(cache, **overrides):
    """Total events by type over all cached videos with the given event_params overrides applied temporarily."""
    old = {k: getattr(P, k) for k in overrides}
    for k, v in overrides.items():
        setattr(P, k, v)
    try:
        counts = {}
        for df, tr, model, scene in cache:
            for e in detect_events(df, tr, model, scene=scene):
                counts[e["event_type"]] = counts.get(e["event_type"], 0) + 1
        return counts
    finally:
        for k, v in old.items():
            setattr(P, k, v)


def sensitivity(cache):
    """How the event counts react when one global is changed at a time (the others stay at their current value)."""
    n = len(cache)
    short = {"DRIBBLE_SHORT": "dS", "DRIBBLE_MEDIUM": "dM", "DRIBBLE_LONG": "dL", "PASS_SUCCESS": "pass+", "PASS_INCOMPLETE": "pass-",
             "SHOT_ON_TARGET": "shot-on", "SHOT_OFF_TARGET": "shot-off"}
    print(f"\n5. SENSITIVITY (events per video, {n} videos; one global changed at a time; current value in brackets)")
    grid = [("SELF_ZONE_ENTER_Y", (0.5, 0.6, 0.7)), ("SELF_ZONE_MIN_BALL_W", (0.02, 0.03, 0.05)), ("HOLD_GRACE_FRAMES", (8, 15, 30)),
            ("DRIBBLE_MIN_FRAMES", (8, 12, 20)), ("PASS_TRIGGER_SPEED_BW", (0.5, 0.75, 1.0, 1.5)), ("PASS_END_OFFSET_FRAMES", (8, 12, 20)),
            ("TEAM_MIN_MARGIN", (0.0, 0.15, 0.3)), ("SHOT_SPEED_BW", (1.5, 2.5, 4.0)), ("SHOT_ON_TARGET_FRAMES", (2, 5, 8))]
    for name, values in grid:
        print(f"   {name} [{getattr(P, name)}]")
        for v in values:
            c = count_events(cache, **{name: v})
            print(f"      = {v:<5}: " + "  ".join(f"{short[k]} {c.get(k, 0) / n:.1f}" for k in short))


def main():
    sessions = sys.argv[1:] or DEFAULT
    files = [f for s in sessions for f in sorted(glob.glob(os.path.join(PROJECT_ROOT, "pipeline_outputs", s, "*_data.csv")))]
    print(f"{len(files)} videos\n")
    vis, runs, gaps, hold_rows, shot_rows = [], [], [], [], []
    cache = []   # (df, track, model, scene) per video, reused by the sensitivity runs
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        tr = BallTrack(df)
        seen = tr.t.index[tr.t["seen"]].values
        vis.append(len(seen) / tr.n_frames)
        if len(seen):
            br = np.split(seen, np.where(np.diff(seen) > 1)[0] + 1)
            runs += [len(r) for r in br]
            gaps += [int(br[i + 1][0] - br[i][-1] - 1) for i in range(len(br) - 1)]

        # teams for the receiver lookup (same recipe as Stage 2)
        pl = df[(df["type"] == "person") & (df["track_id"] != -1) & (df["height"] >= P.TEAM_MIN_BOX_HEIGHT_PX) & (df["height"] >= P.PERSON_MIN_ASPECT * df["width"])]
        cols = ["color_b", "color_g", "color_r"]
        sc, ln = pl.groupby("track_id")[cols].median(), pl.groupby("track_id").size()
        fit = sc[ln >= 8] if (ln >= 8).sum() >= 2 else sc
        model = TeamModel(fit[cols].values, our_shade(session_of(f)))
        scene = Scene(df.assign(team_cluster=0), tr)

        cache.append((df, tr, model, scene))
        for h in find_holds(tr, scene):
            end = h["end"]
            sp = tr.speed(end, end + P.SPEED_WINDOW_FRAMES)
            speed = sp["mean"]
            if not np.isfinite(speed):   # same fallback as detect_events: judge the kick by how fast the ball was leaving
                speed = tr.speed(end - P.PASS_VECTOR_LOOKBACK_FRAMES, end, reach_past=False)["mean"]
            ref = tr.sightings(end + 1, end + P.PASS_END_OFFSET_FRAMES)
            rec_dist = margin = np.nan
            kind = None
            eo = _end_of_flight(tr, end, speed) if np.isfinite(speed) else None
            if eo is not None:
                kind, fr, bx, by, _len = eo
                r = _receiver(scene, fr, bx, by, P.PASS_RECEIVER_MAX_DIST_HEIGHTS if kind == "seen" else P.PASS_VECTOR_RECEIVER_DIST_HEIGHTS)
                if r is not None:
                    rec_dist = r[0] / max(1.0, r[2]["height"])
                    margin = _team_vote(scene, r[2], r[1], model)[1]
            gs = scene.goal_frames_between(end - P.SHOT_GOAL_LOOKBACK_FRAMES, end + P.SHOT_WINDOW_FRAMES)
            hold_rows.append({"video": os.path.basename(f)[:20], "dur": h["end"] - h["start"] + 1, "hits": h["hits"], "exit": h["exit_kind"],
                              "speed": speed, "peak": sp["peak"], "n_after": len(ref), "end_kind": kind, "rec_dist_heights": rec_dist, "margin": margin, "goal_frames": gs})

    print("1. BALL VISIBILITY")
    print(f"   frames with a ball per video: {q(vis, (0, .5, 1))} (min/median/max, fraction)")
    print(f"   consecutive-sighting run length  q10/25/50/75/90/99: {q(runs)}")
    print(f"   gap between runs (frames)        q10/25/50/75/90/99: {q(gaps)}\n")

    H = pd.DataFrame(hold_rows)
    print(f"2. HOLDS: {len(H)} in {len(files)} videos ({len(H) / max(1, len(files)):.1f} per video); exit kinds {H['exit'].value_counts().to_dict()}")
    print(f"   duration (frames) q10/25/50/75/90/99: {q(H['dur'])}")
    d = H[H["dur"] >= P.DRIBBLE_MIN_FRAMES]["dur"]
    print(f"   holds >= DRIBBLE_MIN_FRAMES={P.DRIBBLE_MIN_FRAMES}: {len(d)}; terciles of their length -> medium from {np.quantile(d, 1 / 3):.0f}f, long from {np.quantile(d, 2 / 3):.0f}f "
          f"(current: {P.DRIBBLE_MEDIUM_MIN_FRAMES} / {P.DRIBBLE_LONG_MIN_FRAMES})\n")

    print("3. WHAT HAPPENS WHEN A HOLD ENDS")
    print(f"   ball seen again within {P.PASS_END_OFFSET_FRAMES}f: {(H['n_after'] > 0).mean():.0%}")
    print(f"   scene-relative mean speed (bw/frame) q10/25/50/75/90/99: {q(H['speed'])}")
    print(f"   peak pair speed                      q10/25/50/75/90/99: {q(H['peak'])}")
    print(f"   end of flight: {H['end_kind'].value_counts(dropna=False).to_dict()} (seen = last sighting, vector = extrapolated)")
    print(f"   receiver found (feet within {P.PASS_RECEIVER_MAX_DIST_HEIGHTS} heights of a seen end, {P.PASS_VECTOR_RECEIVER_DIST_HEIGHTS} of an extrapolated one): {H['rec_dist_heights'].notna().mean():.0%}; team margin q10/25/50/75/90: {q(H['margin'], (.1, .25, .5, .75, .9))}")
    for thr in (0.3, 0.5, 0.75, 1.0, 1.5):
        ok = H[(H["speed"] >= thr) & H["rec_dist_heights"].notna()]
        print(f"   speed >= {thr:4.2f} bw/f and receiver found: {len(ok):4d} flights ({len(ok) / max(1, len(files)):.1f}/video), team clear (margin >= {P.TEAM_MIN_MARGIN}): {(ok['margin'] >= P.TEAM_MIN_MARGIN).sum()}")
    print()
    print("4. SHOT-LIKE KICKS (speed above threshold, goal frames in view)")
    for thr in (1.5, 2.0, 2.5, 3.0, 4.0):
        k = H[(H["speed"] >= thr)]
        print(f"   speed >= {thr:3.1f}: {len(k):3d} kicks, with >= {P.SHOT_MIN_GOAL_FRAMES} goal frames in view: {(k['goal_frames'] >= P.SHOT_MIN_GOAL_FRAMES).sum()}")
    sensitivity(cache)
    out_dir = os.path.join(PROJECT_ROOT, "pipeline_outputs", "_analysis")
    os.makedirs(out_dir, exist_ok=True)
    H.to_csv(os.path.join(out_dir, "hold_diagnostics.csv"), index=False)
    print("\nPer-hold table saved to pipeline_outputs/_analysis/hold_diagnostics.csv")


if __name__ == "__main__":
    main()
