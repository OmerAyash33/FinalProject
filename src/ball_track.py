"""
A clean, camera-compensated ball trajectory built from Stage 1's sparse ball sightings, plus the
physics function that gives the ball's mean speed even when it is not seen in every frame.

Why this exists: the ball is detected in only 7-15% of frames, in runs of about 2 frames, and the
camera is worn on the body, so the picture itself swings around. Three things follow:
  * several detections can exist in one frame (a real ball + a false one): keep the most confident;
  * a single detection can jump across the picture (detector error): drop it when BOTH neighbours
    say the ball would have had to move faster than physically possible;
  * ball speed must be measured relative to the SCENE, not the lens. Stage 1 stores how far the
    scene slid in the image every frame (camera rows); the running sum of that is the camera's
    position, and ball position minus camera position is the ball's "scene coordinate". All speeds
    below are speeds of the scene coordinate, in ball-widths per frame (scale-free: a ball that is
    close looks big and moves many pixels, a far one the opposite, and the ratio is what stays put).
"""
# torch must be imported before pandas on this Windows setup (see main.py), whichever module a script imports first.
import torch  # noqa: F401
import numpy as np
import pandas as pd

import event_params as P

MIN_BW_PX = 6.0   # ball width used in speed normalisation never goes below this (tiny far balls would explode the ratio)


class BallTrack:
    """Per-frame ball table `t` (index = frame number, 1..N) and speed/geometry helpers."""

    def __init__(self, df):
        self.n_frames = int(df["frame"].max())
        self.screen_w = float(df["screen_w"].iloc[0])
        self.screen_h = float(df["screen_h"].iloc[0])
        idx = pd.RangeIndex(1, self.n_frames + 1, name="frame")
        t = pd.DataFrame(index=idx, columns=["seen", "interp", "x", "y", "w", "h", "conf", "sx", "sy", "cx", "cy"], dtype=float)
        t["seen"] = False
        t["interp"] = False

        # --- camera position: running sum of per-frame scene shifts (0 when the CSV has no camera rows) ---
        cam = df[df["type"] == "camera"]
        dx = pd.Series(0.0, index=idx)
        dy = pd.Series(0.0, index=idx)
        if not cam.empty:
            cam = cam.drop_duplicates("frame").set_index("frame")
            ok = cam["cam_resp"].fillna(0) >= P.CAMERA_MIN_RESP
            dx.loc[cam.index[ok]] = cam.loc[ok, "cam_dx"].values
            dy.loc[cam.index[ok]] = cam.loc[ok, "cam_dy"].values
        t["cx"] = dx.cumsum()
        t["cy"] = dy.cumsum()
        self.has_camera = not cam.empty

        # --- best ball detection per frame ---
        ball = df[(df["type"] == "sports ball") & (df["confidence"] >= P.BALL_MIN_CONF)]
        ball = ball.sort_values(["frame", "confidence"], ascending=[True, False]).drop_duplicates("frame")
        ball = ball[ball["frame"].between(1, self.n_frames)]
        f = ball["frame"].values
        t.loc[f, "seen"] = True
        for col, src in (("x", "x_center"), ("y", "y_center"), ("w", "width"), ("h", "height"), ("conf", "confidence")):
            t.loc[f, col] = ball[src].values
        t["sx"] = t["x"] - t["cx"]
        t["sy"] = t["y"] - t["cy"]
        self.t = t
        self.raw_sightings = int(t["seen"].sum())
        self._drop_jumps()
        self._interpolate()

    # ------------------------------------------------------------------ cleaning
    def _pair_speed(self, f1, f2):
        a, b = self.t.loc[f1], self.t.loc[f2]
        bw = max(MIN_BW_PX, (a["w"] + b["w"]) / 2)
        return float(np.hypot(b["sx"] - a["sx"], b["sy"] - a["sy"]) / bw / (f2 - f1))

    def _drop_jumps(self):
        """Remove a sighting when both neighbours (within BALL_JUMP_NEIGHBOUR_FRAMES) would need an impossible ball speed."""
        seen = self.t.index[self.t["seen"]].values
        drop = []
        for i, fr in enumerate(seen):
            prev_f = seen[i - 1] if i > 0 and fr - seen[i - 1] <= P.BALL_JUMP_NEIGHBOUR_FRAMES else None
            next_f = seen[i + 1] if i + 1 < len(seen) and seen[i + 1] - fr <= P.BALL_JUMP_NEIGHBOUR_FRAMES else None
            if prev_f is None or next_f is None:
                continue
            if self._pair_speed(prev_f, fr) > P.BALL_MAX_SPEED_BW and self._pair_speed(fr, next_f) > P.BALL_MAX_SPEED_BW:
                drop.append(fr)
        if drop:
            self.t.loc[drop, "seen"] = False
            self.t.loc[drop, ["x", "y", "w", "h", "conf", "sx", "sy"]] = np.nan
        self.dropped_jumps = len(drop)

    def _interpolate(self):
        """Straight-line fill (in scene coordinates, then back into image coordinates) of gaps up to BALL_INTERP_MAX_GAP."""
        seen = self.t.index[self.t["seen"]].values
        for a, b in zip(seen[:-1], seen[1:]):
            gap = b - a - 1
            if gap < 1 or gap > P.BALL_INTERP_MAX_GAP:
                continue
            fr = np.arange(a + 1, b)
            k = (fr - a) / float(b - a)
            for col in ("sx", "sy", "w", "h"):
                self.t.loc[fr, col] = self.t.at[a, col] + k * (self.t.at[b, col] - self.t.at[a, col])
            self.t.loc[fr, "x"] = self.t.loc[fr, "sx"] + self.t.loc[fr, "cx"]
            self.t.loc[fr, "y"] = self.t.loc[fr, "sy"] + self.t.loc[fr, "cy"]
            self.t.loc[fr, "interp"] = True

    # ------------------------------------------------------------------ queries
    def present(self, f):
        """True if the ball has a position at frame f (seen or interpolated)."""
        return 1 <= f <= self.n_frames and (bool(self.t.at[f, "seen"]) or bool(self.t.at[f, "interp"]))

    def sightings(self, f0, f1):
        """Rows of real (seen) detections in [f0, f1]."""
        f0, f1 = max(1, int(f0)), min(self.n_frames, int(f1))
        sub = self.t.loc[f0:f1]
        return sub[sub["seen"]]

    def in_self_zone(self, row, entered):
        """Is this sighting inside the wearer's possession zone? (`entered`: already inside -> use the lower, exit threshold)"""
        y_limit = P.SELF_ZONE_EXIT_Y if entered else P.SELF_ZONE_ENTER_Y
        xn, yn, wn = row["x"] / self.screen_w, row["y"] / self.screen_h, row["w"] / self.screen_w
        return (yn >= y_limit) and (wn >= P.SELF_ZONE_MIN_BALL_W) and (P.SELF_ZONE_X_MIN <= xn <= P.SELF_ZONE_X_MAX)

    def speed(self, f0, f1, reach_past=True):
        """
        Mean ball speed over [f0, f1] relative to the scene, in ball-widths per frame, valid when the ball is
        seen only now and then.

        The ball's scene coordinate is known only at its sightings. Walking through the sightings in order, every
        pair of consecutive ones that are at most SPEED_MAX_PAIR_DT frames apart gives a displacement; the speed is
        the total path divided by the total time those pairs cover (so a pair spanning a 6-frame gap counts for
        6 frames, not 1, and a pair of adjacent frames cannot be over-weighted). The last pair may reach past f1:
        a kick that leaves the picture right after f1 is still measured by the first sighting after it (switch that
        off with reach_past=False to measure only what happened up to f1). Pairs further apart than SPEED_MAX_PAIR_DT
        are ignored, because the ball may have changed direction in between.

        Returns dict(mean, peak, pairs, covered): `mean` and `peak` are NaN when no usable pair exists; `peak` is the
        fastest single pair of at most 3 frames; `covered` is the number of frames the pairs span.
        """
        seen = self.t.index[self.t["seen"]].values
        lo = np.searchsorted(seen, f0, side="left")
        hi = np.searchsorted(seen, f1, side="right")
        i0 = max(0, lo - 1)                      # the sighting just before f0 anchors the first pair
        sel = seen[i0:min(len(seen), hi + (1 if reach_past else 0))]    # and the first one after f1 closes the last pair
        path, span, peak, pairs = 0.0, 0, 0.0, 0
        for a, b in zip(sel[:-1], sel[1:]):
            dt = b - a
            if dt > P.SPEED_MAX_PAIR_DT or b < f0 or a > f1:
                continue
            ra, rb = self.t.loc[a], self.t.loc[b]
            bw = max(MIN_BW_PX, (ra["w"] + rb["w"]) / 2)
            disp = float(np.hypot(rb["sx"] - ra["sx"], rb["sy"] - ra["sy"])) / bw
            path += disp
            span += dt
            pairs += 1
            if dt <= 3:
                peak = max(peak, disp / dt)
        if not pairs:
            return {"mean": float("nan"), "peak": float("nan"), "pairs": 0, "covered": 0}
        return {"mean": min(path / span, P.BALL_MAX_SPEED_BW), "peak": min(peak, P.BALL_MAX_SPEED_BW), "pairs": pairs, "covered": span}

    def velocity(self, f0, f1):
        """Scene velocity (px/frame, vx, vy) from the first to the last sighting in [f0, f1] and the ball width in px, or None."""
        s = self.sightings(f0, f1)
        if len(s) < 2:
            return None
        dt = float(s.index[-1] - s.index[0])
        v = np.array([(s["sx"].iloc[-1] - s["sx"].iloc[0]) / dt, (s["sy"].iloc[-1] - s["sy"].iloc[0]) / dt])
        return v, max(MIN_BW_PX, float(s["w"].iloc[-1]))

    def direction(self, f0, f1):
        """Unit vector (scene coordinates) from the first to the last sighting in [f0, f1], or None."""
        s = self.sightings(f0, f1)
        if len(s) < 2:
            return None
        v = np.array([s["sx"].iloc[-1] - s["sx"].iloc[0], s["sy"].iloc[-1] - s["sy"].iloc[0]])
        n = np.hypot(*v)
        return v / n if n > 1e-6 else None
