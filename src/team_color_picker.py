"""
Pick OUR team's shirt colour by clicking on players (replaces guessing it from "light / dark").

Shows one cropped player at a time, cut out of the footage with the box Stage 1 found. The small rectangle on the player is exactly the
patch whose colour the pipeline measures. For each player:
    left click   (or Y)   this is a player of OUR team
    right click  (or O)   this is an opponent (optional, helps when the two shirts are similar)
    N / "next"            skip, show the next player
    B back, Z undo the last click, Enter = save and close
At most event_params.PICKER_MAX_PER_COLOR (5) players can be clicked per colour, and once PICKER_AUTO_FINISH_TOTAL (7) players in total
are clicked the window saves and closes by itself (the app then goes on to the processing stage). A new pick REPLACES the session's old one.
The median of the clicked shirts becomes the reference colour of the session (team_colors.json); Stage 2 then calls every player whose shirt
is closer to it than to the other colour "ours" - yellow, blue, red, anything.

Needs Stage 1 (the `<video>_data.csv`) of the videos, because the crops come from the detections.
Usage from source:  .venv/Scripts/python.exe src/team_color_picker.py <session folder name | video file(s)>
The packaged app (src/app.py) opens the same window after extraction.
"""
import os
import queue
import random
import sys
import threading
import tkinter as tk
from tkinter import messagebox

import torch  # noqa: F401  (must be imported before pandas on this Windows setup, see main.py)
import cv2
import numpy as np
from PIL import Image, ImageTk

import event_params as P
from team_colors import save_reference

BG, FG, PANEL, ACCENT = "#1b1b1b", "#eeeeee", "#262626", "#4be04b"
FONT = ("Segoe UI", 12)
FONT_BIG = ("Segoe UI", 15, "bold")
CANVAS_W, CANVAS_H = 440, 600
CHEST = (0.4, 0.3, 0.6, 0.5)   # x1, y1, x2, y2 of the measured patch, as fractions of the box (video_processor.py)
USE_COLS = ["frame", "type", "track_id", "x_center", "y_center", "width", "height", "screen_w", "screen_h", "color_b", "color_g", "color_r", "confidence"]


# ------------------------------------------------------------------------------------------------ candidates
def gather_candidates(video_csvs, max_candidates=300, seed=1, min_height=90):
    """
    One candidate per stable tracked player: his tallest sighting that is fully inside the picture (best to look at) and his median
    chest colour (what Stage 2 compares). Players that are too small to see clearly (< min_height px at the 960 px working size) are
    left out unless that leaves fewer than 30 candidates.
    """
    import pandas as pd
    frames = []
    for video, csv in video_csvs:
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv, usecols=lambda c: c in USE_COLS, low_memory=False)
        pl = df[(df["type"] == "person") & (df["track_id"] != -1) & (df["height"] >= P.px(P.TEAM_MIN_BOX_HEIGHT_PX, df["screen_w"]))
                & (df["height"] >= P.PERSON_MIN_ASPECT * df["width"]) & (df["confidence"] >= 0.4)].copy()
        if pl.empty:
            continue
        sizes = pl.groupby("track_id").size()
        pl = pl[pl["track_id"].isin(sizes[sizes >= 8].index)]
        pl["video"] = video
        frames.append(pl)
    if not frames:
        return []
    pl = pd.concat(frames, ignore_index=True)
    pl["key"] = pl["video"] + "#" + pl["track_id"].astype(str)
    med = pl.groupby("key")[["color_b", "color_g", "color_r"]].median()
    x1, x2 = pl["x_center"] - pl["width"] / 2, pl["x_center"] + pl["width"] / 2
    y1, y2 = pl["y_center"] - pl["height"] / 2, pl["y_center"] + pl["height"] / 2
    inside = (x1 > 3) & (y1 > 3) & (x2 < pl["screen_w"] - 3) & (y2 < pl["screen_h"] - 3) & (pl["height"] <= 4.5 * pl["width"])
    best = pl[inside]
    best = best.loc[best.groupby("key")["height"].idxmax()]
    big = best[best["height"] >= min_height]
    best = big if len(big) >= 30 else best
    out = []
    for _, r in best.iterrows():
        m = med.loc[r["key"]]
        out.append({"video": r["video"], "frame": int(r["frame"]), "track": int(r["track_id"]), "xc": float(r["x_center"]), "yc": float(r["y_center"]),
                    "w": float(r["width"]), "h": float(r["height"]), "sw": float(r["screen_w"]), "sh": float(r["screen_h"]),
                    "color": (float(m["color_b"]), float(m["color_g"]), float(m["color_r"]))})
    random.Random(seed).shuffle(out)
    return out[:max_candidates]


class CropReader:
    """Cuts a player out of a video frame. Runs in ONE worker thread (cv2 captures are not shared across threads)."""

    def __init__(self):
        self.caps = {}

    def read(self, c):
        cap = self.caps.get(c["video"])
        if cap is None:
            cap = self.caps[c["video"]] = cv2.VideoCapture(c["video"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, c["frame"] - 1))
        ok, img = cap.read()
        if not ok or img is None:
            return None
        H, W = img.shape[:2]
        sx, sy = W / c["sw"], H / c["sh"]
        x1, y1, x2, y2 = (c["xc"] - c["w"] / 2) * sx, (c["yc"] - c["h"] / 2) * sy, (c["xc"] + c["w"] / 2) * sx, (c["yc"] + c["h"] / 2) * sy
        mx, my = (x2 - x1) * 0.15, (y2 - y1) * 0.08
        cx1, cy1, cx2, cy2 = max(0, int(x1 - mx)), max(0, int(y1 - my)), min(W, int(x2 + mx)), min(H, int(y2 + my))
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        # the measured patch, in crop coordinates
        bw, bh = x2 - x1, y2 - y1
        patch = (x1 + CHEST[0] * bw - cx1, y1 + CHEST[1] * bh - cy1, x1 + CHEST[2] * bw - cx1, y1 + CHEST[3] * bh - cy1)
        scale = min(CANVAS_W / crop.shape[1], CANVAS_H / crop.shape[0])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), tuple(v * scale for v in patch)

    def close(self):
        for cap in self.caps.values():
            cap.release()


def _hex(bgr):
    b, g, r = (int(round(min(255, max(0, v)))) for v in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


# ------------------------------------------------------------------------------------------------ the window
class PickerWindow:
    def __init__(self, master, session, video_csvs):
        self.master = master
        self.session = session
        self.cands = gather_candidates(video_csvs)
        self.all_colors = np.array([c["color"] for c in self.cands]) if self.cands else np.zeros((0, 3))
        self.ours, self.theirs = [], []   # a new pick replaces the saved one of the session (hard limits, see event_params.PICKER_*)
        self.locked = False        # True once the auto-finish has been triggered: no more clicks
        self.history = []          # ("ours"/"theirs", colour) of the clicks of THIS run, for undo
        self.idx = 0
        self.saved = False
        self.reader = CropReader()
        self.jobs = queue.Queue()
        self.cache = {}
        self.pending = set()
        self.stop = False
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()
        self._build()
        self.show()

    # ---- background crop loading
    def _work(self):
        while not self.stop:
            try:
                i = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if i in self.cache:
                continue
            try:
                self.cache[i] = self.reader.read(self.cands[i])
            except Exception:
                self.cache[i] = None
            self.pending.discard(i)
        self.reader.close()   # released by the thread that used the captures: closing them from another thread while a read is running crashes libavcodec

    def _want(self, i):
        for j in range(i, min(i + 4, len(self.cands))):
            if j not in self.cache and j not in self.pending:
                self.pending.add(j)
                self.jobs.put(j)

    # ---- layout
    def _build(self):
        m = self.master
        m.title("Team colour")
        m.configure(bg=BG)
        left = tk.Frame(m, bg=BG)
        left.pack(side="left", padx=10, pady=10)
        self.canvas = tk.Canvas(left, width=CANVAS_W, height=CANVAS_H, bg="#000000", highlightthickness=0, cursor="hand2")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", lambda e: self.pick("ours"))
        self.canvas.bind("<Button-3>", lambda e: self.pick("theirs"))
        self.info = tk.Label(left, text="", bg=BG, fg="#9a9a9a", font=("Segoe UI", 10))
        self.info.pack(pady=(4, 0))

        right = tk.Frame(m, bg=PANEL, padx=14, pady=12)
        right.pack(side="left", fill="y", padx=(0, 10), pady=10)
        tk.Label(right, text="Pick our team's shirt colour", bg=PANEL, fg=FG, font=FONT_BIG).pack(anchor="w")
        for t in ("1. Left-click a player wearing OUR shirt colour", "2. Right-click a player of the opponents (optional)",
                  "3. Press Next (N) to skip a player you cannot judge"):
            tk.Label(right, text=t, bg=PANEL, fg="#c8c8c8", font=FONT, justify="left").pack(anchor="w", pady=1)
        tk.Label(right, text=f"At most {P.PICKER_MAX_PER_COLOR} players per colour. After {P.PICKER_AUTO_FINISH_TOTAL} picks in total the window saves "
                             f"and the processing continues by itself.", bg=PANEL, fg=ACCENT, font=FONT, justify="left", wraplength=330).pack(anchor="w", pady=(6, 0))
        tk.Label(right, text="Keys: Y ours | O opponent | N next | B back | Z undo | Enter save", bg=PANEL, fg="#8a8a8a", font=("Segoe UI", 9)).pack(anchor="w", pady=(6, 10))

        self.sw_ours = self._row(right, "Our colour:", ACCENT)
        self.sw_theirs = self._row(right, "Opponent colour:", "#ff60ff")
        self.captured = self._row(right, "Captured as ours:", FG, swatch=False)

        bt = dict(font=FONT, bd=0, padx=12, pady=7, cursor="hand2")
        self.btn_ours = tk.Button(right, text="Our team", bg="#2f7d2f", fg="white", command=lambda: self.pick("ours"), **bt)
        self.btn_ours.pack(fill="x", pady=(14, 3))
        self.btn_theirs = tk.Button(right, text="Opponents", bg="#8a3a8a", fg="white", command=lambda: self.pick("theirs"), **bt)
        self.btn_theirs.pack(fill="x", pady=3)
        tk.Button(right, text="Next player", bg="#3a3a3a", fg="white", command=self.next, **bt).pack(fill="x", pady=3)
        tk.Button(right, text="Back", bg="#3a3a3a", fg="white", command=self.back, **bt).pack(fill="x", pady=3)
        tk.Button(right, text="Undo last click", bg="#3a3a3a", fg="white", command=self.undo, **bt).pack(fill="x", pady=3)
        tk.Button(right, text="Reset picks", bg="#3a3a3a", fg="white", command=self.reset, **bt).pack(fill="x", pady=3)
        tk.Button(right, text="Finish and save", bg="#1f5fbf", fg="white", command=self.finish, **bt).pack(fill="x", pady=(14, 0))
        self.status = tk.Label(right, text="", bg=PANEL, fg="#ffd060", font=FONT, justify="left", wraplength=330)
        self.status.pack(anchor="w", pady=(10, 0))

        for key, fn in (("y", lambda: self.pick("ours")), ("o", lambda: self.pick("theirs")), ("n", self.next), ("b", self.back), ("z", self.undo)):
            m.bind(key, lambda e, fn=fn: fn())
            m.bind(key.upper(), lambda e, fn=fn: fn())
        m.bind("<Return>", lambda e: self.finish())
        m.bind("<Escape>", lambda e: self.close())
        m.protocol("WM_DELETE_WINDOW", self.close)

    def _row(self, parent, text, colour, swatch=True):
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill="x", pady=3)
        tk.Label(f, text=text, bg=PANEL, fg=FG, font=FONT).pack(side="left")
        value = tk.Label(f, text="", bg=PANEL, fg=colour, font=FONT)
        value.pack(side="right")
        if swatch:
            sw = tk.Canvas(f, width=54, height=22, bg=PANEL, highlightthickness=1, highlightbackground="#555555")
            sw.pack(side="right", padx=8)
            return sw, value
        return None, value

    # ---- state -> screen
    def show(self):
        self.canvas.delete("all")
        if not self.cands:
            self.canvas.create_text(CANVAS_W // 2, CANVAS_H // 2, text="No players found.\nRun extraction first.", fill="#ffffff", font=FONT, justify="center")
            self.refresh()
            return
        if self.idx >= len(self.cands):
            self.canvas.create_text(CANVAS_W // 2, CANVAS_H // 2, text="That was the last player.\nPress Save.", fill="#ffffff", font=FONT_BIG, justify="center")
            self.refresh()
            return
        self._want(self.idx)
        for _ in range(300):   # wait (briefly) for the background read
            if self.idx in self.cache:
                break
            self.master.update()
            self.master.after(10)
        got = self.cache.get(self.idx)
        c = self.cands[self.idx]
        if got is None:
            self.canvas.create_text(CANVAS_W // 2, CANVAS_H // 2, text="(frame could not be read)", fill="#ff8080", font=FONT)
            self.photo = None
        else:
            img, patch = got
            self.photo = ImageTk.PhotoImage(img)
            ox, oy = (CANVAS_W - img.width) // 2, (CANVAS_H - img.height) // 2
            self.canvas.create_image(ox, oy, image=self.photo, anchor="nw")
            self.canvas.create_rectangle(ox + patch[0], oy + patch[1], ox + patch[2], oy + patch[3], outline="#ffffff", width=2)
        self.info.config(text=f"{os.path.basename(c['video'])[:-4]}  frame {c['frame']}   player {self.idx + 1}/{len(self.cands)}")
        self.refresh()

    def refresh(self):
        for sw, value, lst in ((self.sw_ours, None, self.ours), (self.sw_theirs, None, self.theirs)):
            canvas, label = sw
            canvas.delete("all")
            if lst:
                med = np.median(np.array(lst), axis=0)
                canvas.create_rectangle(0, 0, 54, 22, fill=_hex(med), outline="")
            label.config(text=f"{len(lst)}/{P.PICKER_MAX_PER_COLOR}")
        for btn, lst, bg in ((self.btn_ours, self.ours, "#2f7d2f"), (self.btn_theirs, self.theirs, "#8a3a8a")):
            full = len(lst) >= P.PICKER_MAX_PER_COLOR or self.locked
            btn.config(state="disabled" if full else "normal", bg="#4a4a4a" if full else bg)
        _, cap = self.captured
        if self.ours and len(self.all_colors):
            from team_utils import TeamModel
            model = TeamModel.from_reference(self.all_colors, np.array(self.ours), np.array(self.theirs) if self.theirs else None)
            share = float((model.team(self.all_colors) == 0).mean())
            cap.config(text=f"{share * 100:.0f}%")
        else:
            cap.config(text="-")

    # ---- actions
    def pick(self, who):
        if self.locked or not self.cands or self.idx >= len(self.cands):
            return
        lst = self.ours if who == "ours" else self.theirs
        if len(lst) >= P.PICKER_MAX_PER_COLOR:   # HARD limit per colour: the click is refused and the player stays on screen
            self.status.config(text=f"{P.PICKER_MAX_PER_COLOR} players are already picked for this colour. Skip to the next player or finish.")
            return
        colour = self.cands[self.idx]["color"]
        lst.append(colour)
        self.history.append((who, colour))
        self.status.config(text="")
        self.idx += 1
        if len(self.ours) + len(self.theirs) >= P.PICKER_AUTO_FINISH_TOTAL:   # enough: save and go on to the processing stage by itself
            self.locked = True
            self.status.config(text=f"{len(self.ours) + len(self.theirs)} players picked. Saving and continuing with the processing...")
            self.show()
            self.master.after(900, self.finish)
            return
        if len(self.ours) >= P.PICKER_MAX_PER_COLOR and not self.theirs:
            self.status.config(text=f"{P.PICKER_MAX_PER_COLOR} of our players picked. You can also mark opponents (up to {P.PICKER_AUTO_FINISH_TOTAL - P.PICKER_MAX_PER_COLOR}) or press Finish.")
        self.show()

    def next(self):
        if self.cands and self.idx < len(self.cands):
            self.idx += 1
            self.show()

    def back(self):
        if self.idx > 0:
            self.idx -= 1
            self.show()

    def undo(self):
        if self.locked or not self.history:
            return
        who, colour = self.history.pop()
        lst = self.ours if who == "ours" else self.theirs
        lst.remove(colour)
        self.idx = max(0, self.idx - 1)
        self.show()

    def reset(self):
        if messagebox.askyesno("Team colour", "Clear all picks made in this window?", parent=self.master):
            self.ours.clear()
            self.theirs.clear()
            self.history.clear()
            self.locked = False
            self.show()

    def finish(self):
        if self.saved:   # the auto-finish timer and a click on "save" must not both run
            return
        if not self.ours:
            self.status.config(text="You have not picked a player of our team yet. Left-click a player wearing our shirt colour.")
            return
        save_reference(self.session, self.ours, self.theirs)
        self.saved = True
        self.close()

    def close(self):
        self.stop = True
        self.worker.join(timeout=15)   # the worker finishes its current frame, then releases the video captures itself
        self.master.destroy()


def pick_team_colors(session, video_csvs, parent=None):
    """Open the picker for `session`; `video_csvs` = [(video file, its _data.csv)]. Blocks until the window is closed. True if a colour was saved."""
    if parent is None:
        root = tk.Tk()
        win = PickerWindow(root, session, video_csvs)
        root.mainloop()
    else:
        root = tk.Toplevel(parent)
        win = PickerWindow(root, session, video_csvs)
        root.transient(parent)
        root.grab_set()
        parent.wait_window(root)
    return win.saved


def _csv_for(video_path, out_root):
    from app_paths import DETECTOR_VERSION
    session = os.path.basename(os.path.dirname(os.path.abspath(video_path)))
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(out_root, f"{session}_{DETECTOR_VERSION}", f"{base}_data.csv")


if __name__ == "__main__":
    import glob
    from app_paths import APP_ROOT
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if os.path.isfile(args[0]):
        vids = args
    else:
        vids = sorted(glob.glob(os.path.join(APP_ROOT, "raw_footage", args[0], "*.MP4")))
    if not vids:
        sys.exit("no videos found")
    out_root = os.path.join(APP_ROOT, "pipeline_outputs")
    ses = os.path.basename(os.path.dirname(os.path.abspath(vids[0])))
    # a few videos are enough: seven players come out of one
    step = max(1, len(vids) // 4)
    print("session:", ses, "| videos used:", len(vids[::step]))
    print("saved" if pick_team_colors(ses, [(v, _csv_for(v, out_root)) for v in vids[::step]]) else "nothing saved")
