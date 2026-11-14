"""
Progress scoreboard: how well does Stage 2 (event detection) agree with the user's own labels?

Runs Stage 2 in memory on every video that has labels in tools/review_tool/labels.csv (nothing in pipeline_outputs is touched),
matches events to labels and appends one row per split to pipeline_outputs/_analysis/benchmark_history.csv, so every change to the
algorithm or event_params.py can be compared with the earlier ones. If the numbers stop improving after sensible changes, the approach
itself is the limit (that is what this file is for).

Splits: "calib" = labelled videos the parameters were tuned on; "heldout" = labelled videos listed in HELDOUT_VIDEOS below. NEVER tune
on held-out videos - they are the honest number. (Until the user labels them, held-out has no rows.)

Matching (kick-window rule, because the user presses Space 1-3 s AFTER the event): a label matches an event of the same family when
event.start - 15 <= label <= event.end + 90 (frames). Shot labels within 150 frames of each other (SHOT + GOAL of one kick) form one shot.
  recall      = share of labels matched by an event of their family
  class_acc   = of the matched ones, share whose class is right (pass completed/incomplete, dribble short/long, shot on/off target)
  precision*  = share of the events (in the labelled videos) that match a label. Only a LOWER bound: it assumes every pass/shot/dribble
                the user saw was labelled, and the user admits some were not.
Usage:  .venv/Scripts/python.exe tools/benchmark.py "what changed in this run" [--baseline | --no-shot-check]
        --baseline = run with the parameters from BEFORE the 2026-11-07 label calibration (pass trigger 0.5, receiver 1.5 / 2.5, no audio)
"""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch  # noqa: F401  (must be imported before pandas, see main.py)
import numpy as np
import pandas as pd

import event_params as P
from audio_utils import AudioTrack, audio_path_for
from ball_track import BallTrack
from config import DETECTOR_VERSION
from event_detection import Scene, detect_events
from team_utils import build_team_model, session_of

LABELS = os.path.join(PROJECT_ROOT, "tools", "review_tool", "labels.csv")
HISTORY = os.path.join(PROJECT_ROOT, "pipeline_outputs", "_analysis", "benchmark_history.csv")
HELDOUT_VIDEOS = {   # video base names that must never be used for tuning (labelled by the user later)
    "20260130080730_000002", "20260130082530_000004", "20260130084330_000006", "20260130090130_000008",   # January (another day, 4K) videos 2, 4, 6, 8; chosen 2026-11-08
}
FAMILY_OF_LABEL = {"PASS_SUCCESS": "PASS", "PASS_INCOMPLETE": "PASS", "SHOT_ON_TARGET": "SHOT", "SHOT_OFF_TARGET": "SHOT", "GOAL": "SHOT",
                   "DRIBBLE_SHORT": "DRIB", "DRIBBLE_LONG": "DRIB"}
FAMILY_OF_EVENT = {"PASS_SUCCESS": "PASS", "PASS_INCOMPLETE": "PASS", "SHOT_ON_TARGET": "SHOT", "SHOT_OFF_TARGET": "SHOT",
                   "DRIBBLE_SHORT": "DRIB", "DRIBBLE_MEDIUM": "DRIB", "DRIBBLE_LONG": "DRIB"}
KEY_PARAMS = ["PASS_TRIGGER_SPEED_BW", "PASS_RECEIVER_MAX_DIST_HEIGHTS", "PASS_VECTOR_RECEIVER_DIST_HEIGHTS", "UNRESOLVED_PASS", "OUR_SHADE_DEFAULT",
              "HOLD_GRACE_FRAMES", "DRIBBLE_MEDIUM_MIN_FRAMES", "DRIBBLE_LONG_MIN_FRAMES", "AUDIO_ENABLED", "AUDIO_KICK_MIN_Z", "AUDIO_SHOT_GOAL_FRAMES",
              "AUDIO_SHOT_MAX_ZONE_SIGHTINGS", "AUDIO_SHOT_OUTCOME_DEFAULT", "SHOT_CHECK_ENABLED", "SHOT_CHECK_MIN_SCORE"]
BEFORE_CALIBRATION = {"PASS_TRIGGER_SPEED_BW": 0.5, "PASS_RECEIVER_MAX_DIST_HEIGHTS": 1.5, "PASS_VECTOR_RECEIVER_DIST_HEIGHTS": 2.5, "AUDIO_ENABLED": False, "SHOT_CHECK_ENABLED": False}


def stage2_events(session, video):
    """Events of one video exactly as DataAnalyzer would compute them (teams, track, scene, audio), without writing anything."""
    out = os.path.join(PROJECT_ROOT, "pipeline_outputs", f"{session}_{DETECTOR_VERSION}")
    raw = os.path.join(out, f"{video}_data.csv")
    df = pd.read_csv(raw, low_memory=False)
    df.drop_duplicates(inplace=True)
    df.drop(columns=["team_cluster"], errors="ignore", inplace=True)
    cols = ["color_b", "color_g", "color_r"]
    pl = df[(df["type"] == "person") & (df["track_id"] != -1) & (df["height"] >= P.TEAM_MIN_BOX_HEIGHT_PX) & (df["height"] >= P.PERSON_MIN_ASPECT * df["width"])]
    model = None
    if pl["track_id"].nunique() >= 2:
        sc, ln = pl.groupby("track_id")[cols].median(), pl.groupby("track_id").size()
        fit = sc[ln >= 8] if (ln >= 8).sum() >= 2 else sc
        model = build_team_model(fit[cols].values, session_of(raw))
        sc["team_cluster"] = model.team(sc[cols].values)
        df = df.merge(sc[["team_cluster"]], on="track_id", how="left")
    else:
        df["team_cluster"] = np.nan
    track = BallTrack(df)
    audio = AudioTrack.load(audio_path_for(raw)) if P.AUDIO_ENABLED else None
    return pd.DataFrame(detect_events(df, track, model, audio=audio, scene=Scene(df, track))), df["frame"].max()


def match_family(labels, events, fam):
    """Per-label and per-event hits for one family -> dict of counts. labels: DataFrame(frame,label); events: DataFrame(event_type,start_frame,end_frame)."""
    E = events[events["event_type"].map(FAMILY_OF_EVENT) == fam] if len(events) else events
    L = labels[labels["label"].map(FAMILY_OF_LABEL) == fam].sort_values("frame")
    if fam == "SHOT":   # a kick = SHOT + GOAL labels close together
        groups, cur = [], []
        for _, r in L.iterrows():
            if cur and r["frame"] - cur[-1][0] > 150:
                groups.append(cur); cur = []
            cur.append((int(r["frame"]), r["label"]))
        if cur:
            groups.append(cur)
    else:
        groups = [[(int(r["frame"]), r["label"])] for _, r in L.iterrows()]
    res = {"n_labels": len(groups), "matched": 0, "class_ok": 0, "events_matched": 0, "n_events": len(E)}
    ev_hit = np.zeros(len(E), dtype=bool)
    for g in groups:
        hit = None
        for i, (_, e) in enumerate(E.iterrows() if len(E) else []):
            if any(e["start_frame"] - 15 <= f <= e["end_frame"] + 90 for f, _ in g):
                ev_hit[i] = True
                hit = e if hit is None else hit
        if hit is None:
            continue
        res["matched"] += 1
        names = [lab for _, lab in g]
        if fam == "PASS":
            res["class_ok"] += int(hit["event_type"] == names[0])   # PASS_SUCCESS / PASS_INCOMPLETE are both label and event names
        elif fam == "DRIB":
            lab_cls = "DRIBBLE_SHORT" if names[0] == "DRIBBLE_SHORT" else "DRIBBLE_LONG"
            res["class_ok"] += int(hit["event_type"] == lab_cls)
            res["medium_hit"] = res.get("medium_hit", 0) + int(hit["event_type"] == "DRIBBLE_MEDIUM")
        else:
            res["class_ok"] += int(hit["event_type"] == ("SHOT_OFF_TARGET" if names == ["SHOT_OFF_TARGET"] else "SHOT_ON_TARGET"))
    res["events_matched"] = int(ev_hit.sum())
    return res


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    note = args[0] if args else ""
    if "--baseline" in sys.argv:
        for k, v in BEFORE_CALIBRATION.items():
            setattr(P, k, v)
        note = (note + " " if note else "") + "[baseline: pre-calibration params]"
    elif "--no-shot-check" in sys.argv:      # everything as now, but without stage B of the shot logic (event_params 4c)
        P.SHOT_CHECK_ENABLED = False
        note = (note + " " if note else "") + "[shot self-check OFF]"
    lab = pd.read_csv(LABELS, encoding="utf-8-sig")
    t0 = time.time()
    tot = {}
    minutes = {}
    for (session, video), g in lab.groupby(["session", "video"], sort=False):
        split = "heldout" if video in HELDOUT_VIDEOS else "calib"
        events, n_frames = stage2_events(session, video)
        minutes[split] = minutes.get(split, 0.0) + n_frames / 30.0 / 60.0
        for fam in ("PASS", "SHOT", "DRIB"):
            r = match_family(g, events, fam)
            d = tot.setdefault((split, fam), {})
            for k, v in r.items():
                d[k] = d.get(k, 0) + v
    rows = []
    snapshot = "; ".join(f"{k}={getattr(P, k)}" for k in KEY_PARAMS if hasattr(P, k))
    for (split, fam), d in sorted(tot.items()):
        n, m = d["n_labels"], d["matched"]
        rows.append({"time": time.strftime("%Y-%m-%d %H:%M"), "note": note, "split": split, "family": fam, "labels": n, "recall": round(m / n, 3) if n else None,
                     "class_acc": round(d["class_ok"] / m, 3) if m else None, "events": d["n_events"],
                     "precision_lower_bound": round(d["events_matched"] / d["n_events"], 3) if d["n_events"] else None,
                     "events_per_min": round(d["n_events"] / max(minutes[split], 1e-9), 2), "minutes": round(minutes[split], 1), "params": snapshot})
    R = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
    R.to_csv(HISTORY, mode="a", header=not os.path.exists(HISTORY), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(R.drop(columns=["params", "time"]).to_string(index=False))
    print(f"\nappended {len(R)} rows to {HISTORY} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
