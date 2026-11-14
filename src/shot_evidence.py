"""
Shot self-check, stage B of the shot logic (event_params section 4c). Soft evidence only: five weak cues are added up, none is required.

  candidates()  every place a kick could have happened: each hold end, plus the last sighting of a big ball before a long gap
                (a shot often has no hold end at all: the ball was never "held" in the wearer's zone, it was just there and then gone).
  evidence()    the evidence score (0..1) of one candidate and the five cues it is made of.

The decision (add a shot / turn a pass into a shot / rank the shot) is made in event_detection._shot_self_check. Nothing here looks at labels.
"""
import numpy as np

import event_params as P


def ramp(x, lo, hi):
    """0 at `lo`, 1 at `hi`, linear in between (works when hi < lo too)."""
    return float(min(1.0, max(0.0, (float(x) - lo) / (hi - lo))))


def candidates(track, holds):
    """[(frame, kind)] sorted by frame; kind is "hold_end" or "vanish"."""
    out = [(int(h["end"]), "hold_end") for h in holds]
    t = track.t
    seen = t.index[t["seen"]].values
    gaps = [int(a) for a, b in zip(seen[:-1], seen[1:]) if b - a > P.SHOT_CHECK_VANISH_GAP_FRAMES]
    if len(seen):
        gaps.append(int(seen[-1]))
    for a in gaps:
        if any(abs(a - f) <= 25 for f, _ in out):
            continue
        last = track.sightings(a - 30, a)
        if len(last) >= 3 and float(last["w"].median()) >= P.SHOT_CHECK_BIG_BALL_W * track.screen_w:
            out.append((a, "vanish"))
    return sorted(out)


def evidence(scene, track, audio, f0, fps):
    """(score, cues): the weighted mean of the cues that exist (no audio track -> the click cue drops out), each cue 0..1."""
    r = P.SHOT_CHECK_RAMPS
    post = track.sightings(f0 + 5, f0 + 60)
    cues = {
        "goal_size": ramp(scene.goal_width_max(f0 - 30, f0 + 90) / track.screen_w, *r["goal_size"]),
        "goal_frames": ramp(scene.goal_frames_between(f0, f0 + 90), *r["goal_frames"]),
        "whip": ramp(scene.pan_max(f0 + 1, f0 + 26), *r["whip"]),
        "ball_gone": ramp(len(post), *r["ball_gone"]),
    }
    if audio is not None:
        cues["click"] = ramp(audio.kick_z(f0, fps, 0.25, 0.25), *r["click"])
    w = {k: v for k, v in P.SHOT_CHECK_WEIGHTS.items() if k in cues}
    return sum(w[k] * cues[k] for k in w) / sum(w.values()), cues


def describe(cues):
    names = {"goal_size": "goal size", "goal_frames": "goal in view", "click": "kick sound", "whip": "camera whip", "ball_gone": "ball gone"}
    return ", ".join(f"{names[k]} {cues[k]:.1f}" for k in ("goal_size", "goal_frames", "click", "whip", "ball_gone") if k in cues)
