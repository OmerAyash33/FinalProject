"""
Event scores (0-100) and the choice of which events enter the final video.

"Best" has to be defined per kind of event, so each score is a weighted sum of a few components, every one
first scaled to 0..1 (event_params.SCORE_* give the value that earns the full component, SCORE_WEIGHTS the
weights). What each component measures:
  duration         how long the possession lasted                               (dribble)
  length           how far the ball travelled in the pass, scene pixels         (passes)
  speed            mean ball speed relative to the scene, bw/frame              (passes, shots)
  pressure         opponents around the receiver / the dribbler                 (pass success, dribble)
  team_confidence  how clearly the receiver's colour belongs to one team        (passes)
  overlap          frames the ball overlapped the goal                          (shot on target)
  closeness        how near to the goal an off-target shot got                  (shot off target)
  visibility       share of the event's frames in which the ball was actually seen - a clip where the ball is
                   visible and trackable is a clip worth watching and a decision worth trusting
"""
# torch must be imported before pandas on this Windows setup (see main.py), whichever module a script imports first.
import torch  # noqa: F401
import numpy as np
import pandas as pd

import event_params as P


def _clip(x):
    return float(min(1.0, max(0.0, x)))


def _components(e):
    c = {"visibility": _clip(e.get("visibility", 0.0) / P.SCORE_VISIBILITY_FULL),
         "pressure": _clip(e.get("pressure", 0.0) / P.SCORE_PRESSURE_FULL)}
    if e["category"] == "DRIBBLE":
        c["duration"] = _clip(e.get("duration_frames", 0) / P.SCORE_DRIBBLE_FULL_FRAMES)
    if e["category"] in ("PASS_SUCCESS", "PASS_INCOMPLETE"):
        c["length"] = _clip(e.get("length_px", 0.0) / P.SCORE_LENGTH_FULL_PX)
        c["team_confidence"] = _clip(e.get("team_margin", 0.0))
    if e["category"] in ("PASS_SUCCESS", "PASS_INCOMPLETE", "SHOT_ON_TARGET", "SHOT_OFF_TARGET"):
        c["speed"] = _clip(e.get("speed", 0.0) / P.SCORE_SPEED_FULL_BW)
    if e["category"] in ("SHOT_ON_TARGET", "SHOT_OFF_TARGET") and e.get("kick_z") is not None and np.isfinite(e.get("kick_z", np.nan)):
        c["sound"] = _clip(e["kick_z"] / P.AUDIO_SCORE_FULL_Z)   # only present when the video has an audio track
    if e["category"] in ("SHOT_ON_TARGET", "SHOT_OFF_TARGET") and e.get("shot_confidence") is not None and np.isfinite(e.get("shot_confidence", np.nan)):
        c["check"] = _clip(e["shot_confidence"])   # the self-check's evidence score (event_params 4c): confirmed shots with strong evidence rank first
    if e["event_type"] == "SHOT_ON_TARGET":
        c["overlap"] = _clip(e.get("overlap_frames", 0) / (3.0 * P.SHOT_ON_TARGET_FRAMES))
    if e["event_type"] == "SHOT_OFF_TARGET":
        c["closeness"] = 1.0 - _clip(e.get("closeness_goal_widths", P.SCORE_CLOSENESS_FULL_GOAL_WIDTHS) / P.SCORE_CLOSENESS_FULL_GOAL_WIDTHS)
    return c


def score_event(e):
    """0-100 score of one event dict: weighted mean of its components (weights = SCORE_WEIGHTS[category])."""
    comp = _components(e)
    weights = {k: w for k, w in P.SCORE_WEIGHTS[e["category"]].items() if k not in ("sound", "check") or k in comp}   # no audio -> the sound weight drops out; no self-check -> the check weight drops out
    total = sum(weights.values())
    return 100.0 * sum(w * comp.get(name, 0.0) for name, w in weights.items()) / total


def score_events(events):
    for e in events:
        e["score"] = round(score_event(e), 1)
    return events


def select_events(all_events):
    """
    Keep at most TOP_N_PER_EVENT events of each kind, best score first (not first come first served).
    `all_events`: DataFrame with columns video, event_type, category, start_frame, end_frame, score (+ group when
    SELECTION_SCOPE == "session": one value per session/video pool). Two events of the same kind whose clips would
    overlap in the same video count as one: only the better one can be chosen. Returns the chosen rows, input columns kept.
    """
    if all_events.empty:
        return all_events
    clip_before, clip_after = 60, 300   # 2 s lead-in + 10 s window at 30 fps, as rendered by Stage 3
    chosen = []
    key = "event_type" if P.DRIBBLE_SELECTION == "per_class" else "category"
    for (_, kind), grp in all_events.groupby(["pool", key]):
        picked = []
        for _, e in grp.sort_values("score", ascending=False).iterrows():
            clash = any(p["video"] == e["video"] and e["start_frame"] - clip_before < p["start_frame"] + clip_after
                        and p["start_frame"] - clip_before < e["start_frame"] + clip_after for p in picked)
            if not clash:
                picked.append(e)
            if len(picked) >= P.TOP_N_PER_EVENT:
                break
        chosen += picked
    return pd.DataFrame(chosen).sort_values(["video", "start_frame"]).reset_index(drop=True)
