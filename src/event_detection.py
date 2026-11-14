"""
Event detection on top of the cleaned ball track (see ball_track.py): holds/dribbles, passes, shots.

Pipeline per video:
  1. find_holds()          - stretches where the ball sits in the wearer's self zone (event_params section 2)
  2. dribble events        - every hold of at least DRIBBLE_MIN_FRAMES, in three length classes
  3. flights               - a flight starts where a hold ends, or where a fast ball leaves the lower picture
       - SHOT   if the ball is fast enough (SHOT_SPEED_BW) and a goal is in view; outcome from the ball/goal overlap
       - PASS   otherwise, if the ball really travels (PASS_TRIGGER_SPEED_BW); the receiver is the player closest
                to the ball at the END of the flight, his team (voted over several frame windows) decides success
Every number used lives in event_params.py. Every event carries the measurements the scoring needs.
"""
# torch must be imported before pandas on this Windows setup (see main.py), whichever module a script imports first.
import torch  # noqa: F401
import numpy as np
import pandas as pd

import event_params as P
import shot_evidence as SE
from ball_track import MIN_BW_PX


# ----------------------------------------------------------------------------------------------- holds
def find_holds(track, scene=None):
    """
    Holds of the ball in the self zone as dicts(start, end, hits, seen_frac, exit_kind).
    A hold opens at the first in-zone sighting. It stays open while further in-zone sightings arrive at most
    HOLD_GRACE_FRAMES apart. It closes when the ball is seen outside the zone ("seen_leaving") or when no in-zone
    sighting arrives for longer than the grace period ("vanished"). Holds with fewer than HOLD_MIN_HITS in-zone
    sightings, or whose ball was seen in too small a share of its frames (HOLD_MIN_SEEN_FRAC), are discarded as false detections.
    """
    t = track.t
    seen = t.index[t["seen"]].values
    xn, yn, wn = t.loc[seen, "x"].values / track.screen_w, t.loc[seen, "y"].values / track.screen_h, t.loc[seen, "w"].values / track.screen_w
    x_ok = (xn >= P.SELF_ZONE_X_MIN) & (xn <= P.SELF_ZONE_X_MAX) & (wn >= P.SELF_ZONE_MIN_BALL_W)
    enter = x_ok & (yn >= P.SELF_ZONE_ENTER_Y)
    stay = x_ok & (yn >= P.SELF_ZONE_EXIT_Y)
    if scene is not None:   # a ball at another player's feet is HIS possession, whatever part of the picture it sits in
        xs, ys = t.loc[seen, "x"].values, t.loc[seen, "y"].values
        contested = np.array([scene.player_has_ball(int(f), xs[i], ys[i]) for i, f in enumerate(seen)], dtype=bool)
        enter &= ~contested
        stay &= ~contested

    holds, cur = [], None

    def close(kind):
        nonlocal cur
        if cur is not None and cur["hits"] >= P.HOLD_MIN_HITS:
            cur["exit_kind"] = kind
            cur["seen_frac"] = cur["hits"] / float(cur["end"] - cur["start"] + 1)
            if cur["seen_frac"] >= P.HOLD_MIN_SEEN_FRAC:
                holds.append(cur)
        cur = None

    for i, f in enumerate(seen):
        if cur is None:
            if enter[i]:
                cur = {"start": int(f), "end": int(f), "hits": 1}
            continue
        if f - cur["end"] > P.HOLD_GRACE_FRAMES:            # nothing in the zone for too long: the ball vanished
            close("vanished")
            if enter[i]:
                cur = {"start": int(f), "end": int(f), "hits": 1}
        elif stay[i]:
            cur["end"], cur["hits"] = int(f), cur["hits"] + 1
        else:                                                # seen, but above the exit line: it left
            close("seen_leaving")
    close("vanished")
    return holds


def dribble_class(duration):
    """"DRIBBLE_SHORT" / "DRIBBLE_MEDIUM" / "DRIBBLE_LONG", or None for a hold too short to be a dribble."""
    if duration < P.DRIBBLE_MIN_FRAMES:
        return None
    if duration >= P.DRIBBLE_LONG_MIN_FRAMES:
        return "DRIBBLE_LONG"
    if duration >= P.DRIBBLE_MEDIUM_MIN_FRAMES:
        return "DRIBBLE_MEDIUM"
    return "DRIBBLE_SHORT"


# ----------------------------------------------------------------------------------------------- scene lookups
class Scene:
    """Fast per-frame access to the persons and goals of one video, with camera-motion alignment between frames."""

    def __init__(self, df, track):
        self.track = track
        pers = df[(df["type"] == "person") & (df["height"] >= P.PERSON_MIN_ASPECT * df["width"])]   # drops the wearer's own feet/knees/arms
        self.persons = {f: g for f, g in pers.groupby("frame")}
        goals = df[(df["type"] == "goal")].sort_values(["frame", "confidence"], ascending=[True, False]).drop_duplicates("frame")
        self.goal_frames = goals["frame"].values
        self.goals = goals.set_index("frame")
        cam = df[df["type"] == "camera"].drop_duplicates("frame").set_index("frame").sort_index()
        self.pan = (cam["cam_dx"].abs() + cam["cam_dy"].abs()).fillna(0.0)   # camera pan per frame in scene pixels (the whip after a kick)

    def pan_max(self, f0, f1):
        s = self.pan.loc[f0:f1]
        return float(s.max()) if len(s) else 0.0

    def goal_width_max(self, f0, f1):
        """Widest goal detection (pixels) in [f0, f1]; 0 if the goal was not seen. A shot is taken near the goal, so it looks big."""
        m = (self.goal_frames >= f0) & (self.goal_frames <= f1)
        return float(self.goals.loc[self.goal_frames[m], "width"].max()) if m.any() else 0.0

    def camera(self, f):
        f = min(max(int(f), 1), self.track.n_frames)
        return self.track.t.at[f, "cx"], self.track.t.at[f, "cy"]

    def player_has_ball(self, f, bx, by):
        """True if some player's feet are within HOLD_CONTESTED_DIST_HEIGHTS of their own height from the ball (frames f-2..f+2)."""
        for ff in range(f - 2, f + 3):
            g = self.persons.get(ff)
            if g is None:
                continue
            d = np.hypot(g["x_center"].values - bx, (g["y_center"] + g["height"] / 2).values - by)
            if (d <= P.HOLD_CONTESTED_DIST_HEIGHTS * g["height"].values).any():
                return True
        return False

    def goal_box_at(self, f):
        """Goal box (x1, y1, x2, y2) in the image of frame f, from the time-nearest goal detection within GOAL_CARRY_FRAMES,
        moved by how far the camera has turned since then. None if the goal has not been seen recently."""
        if len(self.goal_frames) == 0:
            return None
        i = int(np.searchsorted(self.goal_frames, f))
        cands = [self.goal_frames[j] for j in (i - 1, i) if 0 <= j < len(self.goal_frames)]
        g = min(cands, key=lambda c: abs(c - f))
        if abs(g - f) > P.GOAL_CARRY_FRAMES:
            return None
        r = self.goals.loc[g]
        cxf, cyf = self.camera(f)
        cxg, cyg = self.camera(g)
        sx, sy = cxf - cxg, cyf - cyg
        return (r["x_center"] - r["width"] / 2 + sx, r["y_center"] - r["height"] / 2 + sy,
                r["x_center"] + r["width"] / 2 + sx, r["y_center"] + r["height"] / 2 + sy)

    def goal_frames_between(self, f0, f1):
        return int(((self.goal_frames >= f0) & (self.goal_frames <= f1)).sum())


# ----------------------------------------------------------------------------------------------- helpers
def _pressure_around(scene, f, ball_xy, team_of_us=0, radius_heights=None):
    """Opponent (team 1) persons within `radius_heights` of their own height from a point, at frame f."""
    radius_heights = P.PRESSURE_RADIUS_HEIGHTS if radius_heights is None else radius_heights
    g = scene.persons.get(f)
    if g is None or "team_cluster" not in g:
        return 0
    opp = g[g["team_cluster"] == 1]
    if opp.empty:
        return 0
    d = np.hypot(opp["x_center"] - ball_xy[0], opp["y_center"] + opp["height"] / 2 - ball_xy[1])
    return int((d <= radius_heights * opp["height"]).sum())


def _receiver(scene, f_ref, ball_x, ball_y, max_dist_heights=None):
    """Person closest (feet to ball) at f_ref +- RECEIVER_FRAME_TOLERANCE frames, aligned for camera motion. None if nobody is close enough."""
    max_dist_heights = P.PASS_RECEIVER_MAX_DIST_HEIGHTS if max_dist_heights is None else max_dist_heights
    best = None
    cx_ref, cy_ref = scene.camera(f_ref)
    for f in range(f_ref - P.RECEIVER_FRAME_TOLERANCE, f_ref + P.RECEIVER_FRAME_TOLERANCE + 1):
        g = scene.persons.get(f)
        if g is None:
            continue
        cx, cy = scene.camera(f)
        foot_x = g["x_center"].values + (cx_ref - cx)
        foot_y = (g["y_center"] + g["height"] / 2).values + (cy_ref - cy)
        d = np.hypot(foot_x - ball_x, foot_y - ball_y)
        j = int(np.argmin(d))
        if d[j] <= max_dist_heights * g["height"].values[j] and (best is None or d[j] < best[0]):
            best = (float(d[j]), f, g.iloc[j])
    return best


def _team_vote(scene, receiver_row, receiver_frame, model):
    """
    Probability that the receiver is on OUR team, from his colour over windows of TEAM_VOTE_WINDOWS frames ending at the
    reference frame. A window's colour is the median chest colour of the receiver's track inside it (his own single
    detection when he has no track ID). Combined by TEAM_VOTE_MODE. Returns (p_ours, margin) or (None, 0).
    """
    tid = int(receiver_row["track_id"])
    probs = []
    for n in P.TEAM_VOTE_WINDOWS:
        cols = []
        if tid != -1:
            for f in range(receiver_frame - n + 1, receiver_frame + 1):
                g = scene.persons.get(f)
                if g is not None:
                    r = g[g["track_id"] == tid]
                    if not r.empty:
                        cols.append(r.iloc[0][["color_b", "color_g", "color_r"]].values.astype(float))
        if not cols:
            cols = [receiver_row[["color_b", "color_g", "color_r"]].values.astype(float)]
        probs.append(float(model.prob_ours(np.median(np.array(cols), axis=0))[0]))
    if P.TEAM_VOTE_MODE == "best":
        p = max(probs, key=lambda q: abs(q - 0.5))
    else:
        p = float(np.mean(probs))
    return p, abs(2 * p - 1)


def _overlap_fraction(ball, goal):
    """Share of the ball box that lies inside the goal box (ball: x,y,w,h centre format; goal: x1,y1,x2,y2)."""
    bx1, by1, bx2, by2 = ball[0] - ball[2] / 2, ball[1] - ball[3] / 2, ball[0] + ball[2] / 2, ball[1] + ball[3] / 2
    ix = max(0.0, min(bx2, goal[2]) - max(bx1, goal[0]))
    iy = max(0.0, min(by2, goal[3]) - max(by1, goal[1]))
    area = max((bx2 - bx1) * (by2 - by1), 1e-6)
    return ix * iy / area


# ----------------------------------------------------------------------------------------------- main entry
def detect_events(df, track, team_model, fps=30.0, scene=None, audio=None):
    """All dribble / pass / shot events of one video as a list of dicts (unscored; see event_scoring.score_events).
    `audio` (audio_utils.AudioTrack or None) adds the kick-sound cue for shots the ball track cannot see (event_params section 4b)."""
    scene = scene or Scene(df, track)
    events = []
    holds = find_holds(track, scene)
    t = track.t

    # ---- dribbles ----
    for h in holds:
        duration = h["end"] - h["start"] + 1
        cls = dribble_class(duration)
        if cls is None:
            continue
        pressure, n = 0.0, 0
        for f in t.index[(t.index >= h["start"]) & (t.index <= h["end"]) & t["seen"]].values:
            pressure += _pressure_around(scene, int(f), (t.at[f, "x"], t.at[f, "y"]))
            n += 1
        events.append({"event_type": cls, "category": "DRIBBLE", "start_frame": h["start"], "end_frame": h["end"],
                       "duration_frames": duration, "visibility": h["seen_frac"],
                       "pressure": pressure / n if n else 0.0, "reasoning": f"Hold {duration}f ({duration / fps:.2f}s), seen {h['hits']}x."})

    # ---- flights: triggered by hold ends ----
    triggers = [(h["end"], "hold_end") for h in holds]
    seen = t.index[t["seen"]].values
    for a, b in zip(seen[:-1], seen[1:]):   # fast kick without a hold: the ball's velocity vector is the trigger
        if b - a > 3 or t.at[a, "y"] / track.screen_h < P.SELF_ZONE_EXIT_Y:
            continue
        bw = max(6.0, (t.at[a, "w"] + t.at[b, "w"]) / 2)
        v = np.hypot(t.at[b, "sx"] - t.at[a, "sx"], t.at[b, "sy"] - t.at[a, "sy"]) / bw / (b - a)
        if P.FAST_KICK_TRIGGER_SPEED_BW <= v <= P.FAST_KICK_MAX_SPEED_BW and not scene.player_has_ball(int(a), t.at[a, "x"], t.at[a, "y"])                 and not any(h["start"] - 5 <= a <= h["end"] + 15 for h in holds):
            triggers.append((int(a), "fast_kick"))
    last_trigger = -10 ** 9
    use_audio = audio is not None and P.AUDIO_ENABLED
    for f0, kind in sorted(triggers):
        if f0 - last_trigger < 3:
            continue
        last_trigger = f0
        sp = track.speed(f0, f0 + P.SPEED_WINDOW_FRAMES)
        mean_speed = sp["mean"]
        if not np.isfinite(mean_speed):   # not seen again soon after: judge the kick by how fast the ball was leaving
            mean_speed = track.speed(f0 - P.PASS_VECTOR_LOOKBACK_FRAMES, f0, reach_past=False)["mean"]
        travelling = np.isfinite(mean_speed) and mean_speed >= P.PASS_TRIGGER_SPEED_BW
        # kick sound + goal in view: a shot the ball track cannot see (judged before the speed gate - the ball often vanishes)
        audio_shot = _judge_audio_shot(scene, track, audio, f0, mean_speed, fps) if use_audio and kind == "hold_end" else None
        if audio_shot is not None and (P.AUDIO_SHOT_PRIORITY == "over_pass" or not travelling):
            events.append(audio_shot)
            continue
        if not travelling:
            continue   # the ball did not really travel (or was not seen again): nothing to judge

        # --- shot? ---
        goal_seen = scene.goal_frames_between(f0 - P.SHOT_GOAL_LOOKBACK_FRAMES, f0 + P.SHOT_WINDOW_FRAMES)
        if mean_speed >= P.SHOT_SPEED_BW and goal_seen >= P.SHOT_MIN_GOAL_FRAMES:
            shot = _judge_shot(scene, track, f0, mean_speed, fps)
            if shot is not None:
                if use_audio:
                    shot["kick_z"] = audio.kick_z(f0, fps)
                events.append(shot)
                continue

        # --- pass ---
        pass_ev = _judge_pass(scene, track, f0, mean_speed, team_model, fps) if team_model is not None else None
        if pass_ev is not None:
            events.append(pass_ev)
        elif audio_shot is not None:   # AUDIO_SHOT_PRIORITY == "after_pass": no pass could be resolved, so the loud kick towards the goal is a shot
            events.append(audio_shot)

    if P.SHOT_CHECK_ENABLED:
        events = _shot_self_check(events, scene, track, audio if use_audio else None, holds, fps)
    events.sort(key=lambda e: e["start_frame"])
    return events


def _judge_shot(scene, track, f0, mean_speed, fps):
    t = track.t
    f_end = min(track.n_frames, f0 + P.SHOT_WINDOW_FRAMES)
    overlap_frames, best_overlap, last_goal, closest = 0, 0.0, None, None
    for f in range(f0, f_end + 1):
        if not track.present(f):
            continue
        goal = scene.goal_box_at(f)
        if goal is None:
            continue
        last_goal = goal
        row = t.loc[f]
        frac = _overlap_fraction((row["x"], row["y"], row["w"], row["h"]), goal)
        best_overlap = max(best_overlap, frac)
        if frac >= P.SHOT_OVERLAP_MIN_FRACTION:
            overlap_frames += 1
        gcx, gcy = (goal[0] + goal[2]) / 2, (goal[1] + goal[3]) / 2
        d = np.hypot(row["x"] - gcx, row["y"] - gcy) / max(goal[2] - goal[0], 1.0)
        closest = d if closest is None else min(closest, d)
    seen_in_window = track.sightings(f0, f_end)
    # Evidence gates: a kick is only a shot if the ball was seen often enough, started near the camera (at the wearer's feet)
    # and moved away from it (its image shrank). Camera whips make a far resting ball look fast; these gates reject that.
    w0 = float(t.at[f0, "w"])
    if len(seen_in_window) < P.SHOT_MIN_SIGHTINGS or w0 < P.SELF_ZONE_MIN_BALL_W * track.screen_w             or float(seen_in_window["w"].iloc[-1]) > P.SHOT_MAX_WIDTH_RATIO * w0:
        return None
    # The ball at the wearer's feet is big and sharp, so a real shot starts with a confident sighting (a hand or a shoe at the corner of the picture is not).
    if float(seen_in_window["conf"].iloc[0]) < P.SHOT_MIN_START_CONF:
        return None
    # Coherent flight: the sightings must contain SHOT_MIN_FAST_PAIRS pairs, close in time, that move at a plausible shot speed
    # (not a detector jump) and in the same direction. Camera whips and a ball jumping between two false detections fail this.
    fr = seen_in_window.index.values
    moves = []
    for a, b in zip(fr[:-1], fr[1:]):
        if b - a > P.SHOT_PAIR_MAX_DT:
            continue
        ra, rb = t.loc[a], t.loc[b]
        d = np.array([rb["sx"] - ra["sx"], rb["sy"] - ra["sy"]])
        n = float(np.hypot(*d))
        sp = n / max(MIN_BW_PX, (ra["w"] + rb["w"]) / 2) / (b - a)
        if P.SHOT_PAIR_MIN_SPEED_BW <= sp <= P.FAST_KICK_MAX_SPEED_BW and n > 1e-6:
            moves.append(d / n)
    coherent = max((sum(float(np.dot(u, v)) >= P.SHOT_PAIR_MIN_COS for v in moves) for u in moves), default=0)
    if coherent < P.SHOT_MIN_FAST_PAIRS:
        return None
    vis = len(seen_in_window) / float(f_end - f0 + 1)
    base = {"start_frame": int(f0), "end_frame": int(f_end), "speed": mean_speed, "visibility": vis, "shot_origin": "ball_track"}
    if overlap_frames >= P.SHOT_ON_TARGET_FRAMES:
        return {**base, "event_type": "SHOT_ON_TARGET", "category": "SHOT_ON_TARGET", "overlap_frames": overlap_frames, "overlap_fraction": best_overlap,
                "reasoning": f"Speed {mean_speed:.1f} bw/f, ball overlapped the goal for {overlap_frames} frames."}
    if P.SHOT_UNRESOLVED_AS != "off_target" or last_goal is None:
        return None
    # direction check: the ball must head towards the goal, otherwise this is a pass (or a clearance), not a shot
    d = track.direction(f0, f_end)
    gcx, gcy = (last_goal[0] + last_goal[2]) / 2, (last_goal[1] + last_goal[3]) / 2
    cx0, cy0 = track.t.at[f0, "cx"], track.t.at[f0, "cy"]
    x0, y0 = track.t.at[f0, "sx"], track.t.at[f0, "sy"]
    cxe, cye = scene.camera(f_end)
    to_goal = np.array([gcx - cxe - x0, gcy - cye - y0])
    n = np.hypot(*to_goal)
    if d is None or n < 1e-6 or float(np.dot(d, to_goal / n)) < P.SHOT_MIN_DIRECTION_COS:
        return None
    return {**base, "event_type": "SHOT_OFF_TARGET", "category": "SHOT_OFF_TARGET", "overlap_frames": overlap_frames, "overlap_fraction": best_overlap,
            "closeness_goal_widths": closest if closest is not None else P.SCORE_CLOSENESS_FULL_GOAL_WIDTHS,
            "reasoning": f"Speed {mean_speed:.1f} bw/f towards the goal, never overlapped it (closest {closest:.1f} goal-widths)." if closest is not None else
                         f"Speed {mean_speed:.1f} bw/f towards the goal, never overlapped it."}


def _ball_stays_with_wearer(track, f0, f_end):
    """True if the ball did not leave the wearer after f0, which a shot must do.
    A shot leaves him: if the ball is seen again shortly after the kick and is still nearly as big as it was (or bigger), it stayed at his feet / in his
    hands (a dribble, a hand throw). Same width gate as the visual shot rule (SHOT_MAX_WIDTH_RATIO), applied only when the ball is seen again at all.
    Removed 3 of 7 false audio shots on the labelled videos and none of the 4 real ones. And if the ball keeps being seen in the wearer's own zone
    after the kick, he still has it (a dribble towards the goal, a water bottle held in the hand that the detector took for a ball). The real shots have
    0-4 such sightings, the false ones 18-42."""
    t = track.t
    w0 = float(t.at[f0, "w"]) if f0 in t.index else float("nan")
    after = track.sightings(f0 + 3, f_end)
    if np.isfinite(w0) and w0 > 0 and len(after) and float(after["w"].iloc[-1]) > P.SHOT_MAX_WIDTH_RATIO * w0:
        return True
    return bool(len(after) and int((after["y"] / track.screen_h >= P.SELF_ZONE_EXIT_Y).sum()) >= P.AUDIO_SHOT_MAX_ZONE_SIGHTINGS)


def _unseen_shot_event(scene, track, f0, f_end, mean_speed, kick_z, origin, lead):
    """A shot whose flight the ball track did not follow (stage A "audio+goal", stage B "self_check"). On target if the ball was seen overlapping
    the goal like a visual shot (SHOT_ON_TARGET_FRAMES frames), otherwise the outcome is unknown and AUDIO_SHOT_OUTCOME_DEFAULT is used."""
    t = track.t
    overlap_frames, best_overlap, closest = 0, 0.0, None
    for f in range(f0, f_end + 1):
        if not track.present(f):
            continue
        goal = scene.goal_box_at(f)
        if goal is None:
            continue
        row = t.loc[f]
        frac = _overlap_fraction((row["x"], row["y"], row["w"], row["h"]), goal)
        best_overlap = max(best_overlap, frac)
        if frac >= P.SHOT_OVERLAP_MIN_FRACTION:
            overlap_frames += 1
        gcx, gcy = (goal[0] + goal[2]) / 2, (goal[1] + goal[3]) / 2
        d = np.hypot(row["x"] - gcx, row["y"] - gcy) / max(goal[2] - goal[0], 1.0)
        closest = d if closest is None else min(closest, d)
    speed = float(mean_speed) if np.isfinite(mean_speed) else 0.0
    vis = len(track.sightings(f0, f_end)) / float(f_end - f0 + 1)
    base = {"start_frame": int(f0), "end_frame": int(f_end), "speed": speed, "visibility": vis, "kick_z": kick_z, "shot_origin": origin}
    if overlap_frames >= P.SHOT_ON_TARGET_FRAMES:
        outcome, why = "SHOT_ON_TARGET", f"ball overlapped the goal for {overlap_frames} frames"
    else:   # the ball was not tracked into (or past) the goal: the outcome is unknown, a global default decides
        outcome = "SHOT_OFF_TARGET" if P.AUDIO_SHOT_OUTCOME_DEFAULT == "off_target" else "SHOT_ON_TARGET"
        why = f"ball not tracked in flight, outcome unknown -> {P.AUDIO_SHOT_OUTCOME_DEFAULT.replace('_', ' ')} by default (AUDIO_SHOT_OUTCOME_DEFAULT)"
    ev = {**base, "event_type": outcome, "category": outcome, "overlap_frames": overlap_frames, "overlap_fraction": best_overlap,
          "reasoning": f"{lead}; {why}."}
    if outcome == "SHOT_OFF_TARGET":
        ev["closeness_goal_widths"] = closest if closest is not None else P.SCORE_CLOSENESS_FULL_GOAL_WIDTHS
    return ev


def _judge_audio_shot(scene, track, audio, f0, mean_speed, fps):
    """Stage A, second rule: a hold end with a loud click (AUDIO_KICK_MIN_Z) and the goal in view for AUDIO_SHOT_GOAL_FRAMES of the next
    AUDIO_SHOT_GOAL_WINDOW_FRAMES frames is a shot (unless the ball stayed with the wearer)."""
    kick_z = audio.kick_z(f0, fps)
    if kick_z < P.AUDIO_KICK_MIN_Z:
        return None
    if scene.goal_frames_between(f0, f0 + P.AUDIO_SHOT_GOAL_WINDOW_FRAMES) < P.AUDIO_SHOT_GOAL_FRAMES:
        return None
    f_end = min(track.n_frames, f0 + P.SHOT_WINDOW_FRAMES)
    if _ball_stays_with_wearer(track, f0, f_end):
        return None
    return _unseen_shot_event(scene, track, f0, f_end, mean_speed, kick_z, "audio+goal",
                              f"Kick sound (z {kick_z:.0f}) at the hold end, goal in view {scene.goal_frames_between(f0, f0 + P.AUDIO_SHOT_GOAL_WINDOW_FRAMES)} of "
                              f"{P.AUDIO_SHOT_GOAL_WINDOW_FRAMES} frames after")


def _shot_self_check(events, scene, track, audio, holds, fps):
    """Stage B (event_params 4c): soft evidence for shots. (1) every shot already found carries its evidence score as `shot_confidence`;
    (2) every kick candidate that is not near such a shot and whose evidence reaches SHOT_CHECK_MIN_SCORE becomes a "possible shot" - it is ADDED, or
    it REPLACES the pass the same hold end produced. Nothing here is required for a shot to exist: stage A decides on its own."""
    is_shot = lambda e: e["category"] in ("SHOT_ON_TARGET", "SHOT_OFF_TARGET")
    shots = [e for e in events if is_shot(e)]
    for e in shots:
        e["shot_confidence"] = round(SE.evidence(scene, track, audio, int(e["start_frame"]), fps)[0], 3)
    taken = [int(e["start_frame"]) for e in shots]
    added = []
    for f0, _kind in SE.candidates(track, holds):
        if any(abs(f0 - s) <= P.SHOT_CHECK_DEDUP_FRAMES for s in taken):
            continue
        score, cues = SE.evidence(scene, track, audio, f0, fps)
        if score < P.SHOT_CHECK_MIN_SCORE:
            continue
        f_end = min(track.n_frames, f0 + P.SHOT_WINDOW_FRAMES)
        if _ball_stays_with_wearer(track, f0, f_end):
            continue
        ev = _unseen_shot_event(scene, track, f0, f_end, track.speed(f0, f0 + P.SPEED_WINDOW_FRAMES)["mean"],
                                audio.kick_z(f0, fps) if audio is not None else None, "self_check",
                                f"Possible shot (self-check, evidence {score:.2f}: {SE.describe(cues)})")
        ev["shot_confidence"] = round(score, 3)
        added.append((score, ev))
    keep = []
    for score, ev in sorted(added, key=lambda x: -x[0]):     # at most one possible shot per SHOT_CHECK_DEDUP_FRAMES, the best wins
        if all(abs(ev["start_frame"] - k["start_frame"]) > P.SHOT_CHECK_DEDUP_FRAMES for k in keep):
            keep.append(ev)
    for ev in keep:
        events = [e for e in events if not (e["category"] in ("PASS_SUCCESS", "PASS_INCOMPLETE") and abs(int(e["start_frame"]) - ev["start_frame"]) <= 5)]
        events.append(ev)
    return events


def _end_of_flight(track, f0, mean_speed):
    """
    Where the ball is at the END of its flight, as (kind, frame, x, y, length_px) in the image of that frame:
      "seen"   - the last sighting within PASS_END_OFFSET_FRAMES of the kick (or, if there is none, the first one within
                 PASS_SEARCH_MAX_FRAMES): the last place the ball was actually detected before it vanished;
      "vector" - the ball is not seen again, so its motion vector is extended: scene velocity measured just before the
                 hold ended, carried PASS_END_OFFSET_FRAMES forward (at most PASS_VECTOR_MAX_PX), moved into the image of
                 that frame with the camera's own motion.
    None when neither is possible.
    """
    t = track.t
    ref = track.sightings(f0 + 1, f0 + P.PASS_END_OFFSET_FRAMES)
    if ref.empty:
        ref = track.sightings(f0 + P.PASS_END_OFFSET_FRAMES + 1, f0 + P.PASS_SEARCH_MAX_FRAMES).iloc[:1]
        f_ref = int(ref.index[0]) if len(ref) else None
    else:
        f_ref = int(ref.index[-1])
    if f_ref is not None:
        length = float(np.hypot(t.at[f_ref, "sx"] - t.at[f0, "sx"], t.at[f_ref, "sy"] - t.at[f0, "sy"]))
        return "seen", f_ref, float(t.at[f_ref, "x"]), float(t.at[f_ref, "y"]), length
    vel = track.velocity(f0 - P.PASS_VECTOR_LOOKBACK_FRAMES, f0)
    if vel is None:
        return None
    v, _ = vel
    f_end = min(track.n_frames, f0 + P.PASS_END_OFFSET_FRAMES)
    travel = v * (f_end - f0)
    n = float(np.hypot(*travel))
    if n > P.PASS_VECTOR_MAX_PX:
        travel = travel * (P.PASS_VECTOR_MAX_PX / n)
        n = P.PASS_VECTOR_MAX_PX
    sx, sy = t.at[f0, "sx"] + travel[0], t.at[f0, "sy"] + travel[1]
    return "vector", int(f_end), float(sx + t.at[f_end, "cx"]), float(sy + t.at[f_end, "cy"]), n


def _judge_pass(scene, track, f0, mean_speed, team_model, fps):
    t = track.t
    end = _end_of_flight(track, f0, mean_speed)
    if end is None:
        return _unresolved(f0, mean_speed, "Ball not seen again after the kick and no direction to follow.", fps)
    kind, f_ref, bx, by, length = end
    rec = _receiver(scene, f_ref, bx, by, P.PASS_RECEIVER_MAX_DIST_HEIGHTS if kind == "seen" else P.PASS_VECTOR_RECEIVER_DIST_HEIGHTS)
    if rec is None:
        return _unresolved(f0, mean_speed, f"No player near the ball's end point ({kind}) at f{f_ref}.", fps)
    _, f_rec, row = rec
    if float(row["height"]) < P.TEAM_MIN_BOX_HEIGHT_PX:
        return _unresolved(f0, mean_speed, "Receiver too small to read a jersey colour.", fps)
    p_ours, margin = _team_vote(scene, row, f_rec, team_model)
    if p_ours is None or margin < P.TEAM_MIN_MARGIN:
        return _unresolved(f0, mean_speed, f"Receiver's team unclear (margin {margin:.2f}).", fps)
    ours = p_ours >= 0.5
    pressure = _pressure_around(scene, f_rec, (row["x_center"], row["y_center"] + row["height"] / 2)) if ours else 0
    vis = len(track.sightings(f0, f_ref)) / float(max(1, f_ref - f0 + 1))
    return {"event_type": "PASS_SUCCESS" if ours else "PASS_INCOMPLETE", "category": "PASS_SUCCESS" if ours else "PASS_INCOMPLETE",
            "start_frame": int(f0), "end_frame": int(f_ref), "speed": mean_speed, "length_px": length,
            "team_margin": margin, "pressure": pressure, "visibility": vis,
            "reasoning": f"Speed {mean_speed:.1f} bw/f, length {length:.0f}px ({kind} end), receiver {'ours' if ours else 'opponent'} (margin {margin:.2f})."}


def _unresolved(f0, mean_speed, why, fps):
    if P.UNRESOLVED_PASS != "incomplete":
        return None
    return {"event_type": "PASS_INCOMPLETE", "category": "PASS_INCOMPLETE", "start_frame": int(f0), "end_frame": int(f0 + P.PASS_END_OFFSET_FRAMES),
            "speed": mean_speed, "length_px": 0.0, "team_margin": 0.0, "pressure": 0, "visibility": 0.0, "reasoning": why}
