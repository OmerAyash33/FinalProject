"""
TEMPORARY hand-labelling tool - delete the whole folder tools/review_tool when the labelling is done.

Plays the raw footage one video after another. Press SPACE to stop and mark what just happened
(pass completed / failed, shot on / off target, goal, short / long dribble); the mark is saved with the
video, frame and clock time to tools/review_tool/labels.csv after every single mark, and the tool
reopens where you stopped.

Keys
  Space            playing: stop and open the menu   |   menu open: close it and carry on   |   stopped: carry on
  Up / Down        move in the menu            Enter or 1-7: choose            Esc: close the menu, stay stopped
  Left / Right     5 s back / forward (also while the menu is open: go back to the exact moment, then choose)
  Shift+Left/Right 1 s back / forward         ,  and  .  one frame back / forward (while stopped)
  + / -            playback speed             N / P: next / previous video      M: open the menu while stopped
  Z                undo the last mark         double-click a mark in the right-hand list: jump to 2 s before it
  Delete           remove the selected mark of the right-hand list

Usage:  .venv/Scripts/python.exe tools/review_tool/review_tool.py [--start-video N]
"""
import argparse
import collections
import csv
import datetime
import glob
import json
import os
import sys
import time
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
# Order in which the sessions are offered (the two May sessions first, the 4K January one last: it decodes slowest).
SESSION_ORDER = ["2026-05-05_evening", "2026-05-01_morning", "2026-07-28", "2026-01-30"]

# code, text in the menu, colour of the menu line
LABELS = [
    ("PASS_SUCCESS", "מסירה שהושלמה", "#4be04b"),
    ("PASS_INCOMPLETE", "מסירה שלא הושלמה", "#ff6060"),
    ("SHOT_ON_TARGET", "בעיטה למסגרת", "#c8ffc8"),
    ("SHOT_OFF_TARGET", "בעיטה מחוץ למסגרת", "#ffc0cb"),
    ("GOAL", "גול", "#ffd700"),
    ("DRIBBLE_SHORT", "נגיעה / דריבל קצר", "#ffc878"),
    ("DRIBBLE_LONG", "דריבל ארוך", "#f0801e"),
]
HE = {c: t for c, t, _ in LABELS}
SPEEDS = [0.5, 1.0, 2.0, 3.0, 4.0]
CSV_FIELDS = ["session", "video", "frame", "time_sec", "clock", "label", "label_he"]
BG, FG, PANEL = "#1b1b1b", "#eeeeee", "#262626"
FONT = ("Segoe UI", 12)


def list_videos():
    out = []
    for s in SESSION_ORDER:
        for p in sorted(glob.glob(os.path.join(PROJECT_ROOT, "raw_footage", s, "*.MP4"))):
            out.append({"session": s, "path": p, "name": os.path.splitext(os.path.basename(p))[0]})
    return out


def start_time(name):
    """Clock time at the first frame, read from the camera's file name (YYYYMMDDHHMMSS_...)."""
    try:
        return datetime.datetime.strptime(name[:14], "%Y%m%d%H%M%S")
    except ValueError:
        return datetime.datetime(2000, 1, 1)


def mmss(sec):
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


class App:
    def __init__(self, root, videos, labels_path, state_path, start_index=None):
        self.root, self.videos = root, videos
        self.labels_path, self.state_path = labels_path, state_path
        self.cap, self.n, self.fps, self.pos = None, 0, 30.0, -1
        self.playing, self.speed = False, 1.0
        self.anchor_pos, self.anchor_t = 0, time.perf_counter()
        self.cur = 0
        self.menu_open = False
        self.dragging = False
        self.last_ui = 0.0
        self.labels = self.read_labels()
        self.panel_rows = []        # index into self.labels for each row of the right-hand list

        root.title("כלי סימון זמני")
        root.configure(bg=BG)
        root.geometry("1500x900")
        try:
            root.state("zoomed")
        except tk.TclError:
            pass

        # --- top bar
        top = tk.Frame(root, bg=BG)
        top.pack(fill="x", padx=8, pady=(6, 2))
        self.combo = ttk.Combobox(top, state="readonly", width=60, takefocus=0,
                                  values=[f"{i + 1:02d}   {v['session']}   {v['name']}" for i, v in enumerate(videos)])
        self.combo.pack(side="left")
        self.combo.bind("<<ComboboxSelected>>", self.on_combo)
        self.info = tk.Label(top, text="", bg=BG, fg=FG, font=FONT)
        self.info.pack(side="left", padx=14)
        self.speed_lbl = tk.Label(top, text="", bg=BG, fg="#ffd24a", font=("Segoe UI", 13, "bold"))
        self.speed_lbl.pack(side="right")

        # --- middle: video + right panel
        mid = tk.Frame(root, bg=BG)
        mid.pack(fill="both", expand=True, padx=8)
        self.video_box = tk.Frame(mid, bg="black")
        self.video_box.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(self.video_box, bg="black", highlightthickness=0, takefocus=0)
        self.canvas.pack(fill="both", expand=True)
        self.img_id = self.canvas.create_image(0, 0, anchor="center")
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.big = self.canvas.create_text(0, 0, text="", fill="#ffd24a", font=("Segoe UI", 28, "bold"))

        side = tk.Frame(mid, bg=PANEL, width=310)
        side.pack(side="right", fill="y", padx=(8, 0))
        side.pack_propagate(False)
        tk.Label(side, text="הסימונים בסרטון הזה", bg=PANEL, fg=FG, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=8, pady=(8, 2))
        self.panel = tk.Listbox(side, bg=PANEL, fg=FG, font=FONT, height=14, activestyle="none", takefocus=0,
                                selectbackground="#444444", highlightthickness=0, exportselection=False)
        self.panel.pack(fill="x", padx=8)
        self.panel.bind("<Double-Button-1>", self.on_panel_double)
        tk.Label(side, text="סך הכול בכל הסרטונים", bg=PANEL, fg=FG, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=8, pady=(14, 2))
        self.totals = tk.Label(side, text="", bg=PANEL, fg=FG, font=FONT, justify="left", anchor="w")
        self.totals.pack(fill="x", padx=8)

        # --- slider
        bar = tk.Frame(root, bg=BG)
        bar.pack(fill="x", padx=8, pady=(4, 0))
        self.time_lbl = tk.Label(bar, text="00:00 / 00:00", bg=BG, fg=FG, font=FONT, width=14)
        self.time_lbl.pack(side="left")
        self.slider = ttk.Scale(bar, from_=0, to=1, orient="horizontal", takefocus=0)
        self.slider.pack(side="left", fill="x", expand=True)
        self.slider.bind("<ButtonPress-1>", lambda e: setattr(self, "dragging", True))
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)
        self.status = tk.Label(root, text="", bg=BG, fg="#9fe39f", font=FONT, anchor="w")
        self.status.pack(fill="x", padx=8)
        tk.Label(root, text="Space = stop and mark   |   Left/Right = 5 s back/forward   |   Shift+Left/Right = 1 s   |   + / - = speed   |   "
                            "N / P = next / previous video   |   Z = undo last mark", bg=BG, fg="#9a9a9a", font=("Segoe UI", 10),
                 anchor="w").pack(fill="x", padx=8, pady=(0, 6))

        # --- menu (placed over the video while it is open)
        self.menu = tk.Frame(self.video_box, bg="#2b2b2b", bd=3, relief="ridge")
        tk.Label(self.menu, text="מה קרה?", bg="#2b2b2b", fg=FG, font=("Segoe UI", 12, "bold")).pack(padx=10, pady=(8, 4))
        tk.Label(self.menu, text="Up/Down + Enter, or press 1-7        Esc = cancel", bg="#2b2b2b", fg="#aaaaaa", font=("Segoe UI", 10)).pack()
        lf = tk.Frame(self.menu, bg="#2b2b2b")
        lf.pack(padx=10, pady=(4, 10))
        self.lb = tk.Listbox(lf, bg="#202020", fg=FG, font=("Segoe UI", 15), height=len(LABELS), width=26, activestyle="none",
                             exportselection=False, takefocus=0, selectbackground="#555555", highlightthickness=0)
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.lb.yview)
        self.lb.configure(yscrollcommand=sb.set)
        self.lb.pack(side="left")
        sb.pack(side="left", fill="y")
        for i, (code, he, color) in enumerate(LABELS):
            self.lb.insert("end", f" {i + 1}    {he}")
            self.lb.itemconfig(i, fg=color)
        self.lb.bind("<ButtonRelease-1>", self.on_menu_click)

        root.bind_all("<Key>", self.on_key)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.refresh_totals()
        self.update_speed_label()

        idx = start_index
        if idx is None:
            idx, frame = self.read_state()
        else:
            frame = 0
        self.load_video(idx, frame)
        self.root.after(5, self.tick)

    # ------------------------------------------------------------------------------------------- data
    def read_labels(self):
        if not os.path.exists(self.labels_path):
            return []
        with open(self.labels_path, newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def write_labels(self):
        with open(self.labels_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(self.labels)

    def read_state(self):
        try:
            with open(self.state_path, encoding="utf-8") as f:
                s = json.load(f)
            return int(s.get("video_index", 0)), int(s.get("frame", 0))
        except (OSError, ValueError):
            return 0, 0

    def save_state(self):
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump({"video_index": self.cur, "frame": max(0, self.pos)}, f)
        except OSError:
            pass

    def add_label(self, code):
        v = self.videos[self.cur]
        t = self.pos / self.fps
        clock = (start_time(v["name"]) + datetime.timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S")
        self.labels.append({"session": v["session"], "video": v["name"], "frame": self.pos, "time_sec": f"{t:.2f}",
                            "clock": clock, "label": code, "label_he": HE[code]})
        self.write_labels()
        self.save_state()
        self.refresh_panel()
        self.refresh_totals()
        self.say(f"נשמר: {HE[code]}   {mmss(t)}   (שעון {clock[11:]})")

    def video_rows(self):
        name = self.videos[self.cur]["name"]
        rows = [(i, r) for i, r in enumerate(self.labels) if r["video"] == name]
        return sorted(rows, key=lambda x: int(x[1]["frame"]))

    # ------------------------------------------------------------------------------------------- video
    def load_video(self, idx, frame=0):
        idx = max(0, min(len(self.videos) - 1, idx))
        if self.cap is not None:
            self.cap.release()
        self.cur = idx
        v = self.videos[idx]
        self.cap = cv2.VideoCapture(v["path"])
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.pos = -1
        self.slider.configure(to=max(1, self.n - 1))
        self.combo.current(idx)
        self.refresh_panel()
        self.seek(frame)
        self.root.focus_set()

    def seek(self, frame):
        if self.cap is None:
            return
        frame = max(0, min(self.n - 1, int(frame)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, im = self.cap.read()
        if ok:
            self.pos = frame
            self.show(im)
        self.anchor_pos, self.anchor_t = self.pos, time.perf_counter()
        self.update_bar(force=True)

    def show(self, bgr):
        self.last_bgr = bgr
        self.redraw()

    def redraw(self):
        bgr = getattr(self, "last_bgr", None)
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas.coords(self.img_id, cw // 2, ch // 2)
        self.canvas.coords(self.big, cw // 2, 36)
        if bgr is None or cw < 20 or ch < 20:
            return
        h, w = bgr.shape[:2]
        s = min(cw / w, ch / h)
        small = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_LINEAR)
        self._img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        self.canvas.itemconfig(self.img_id, image=self._img)

    def tick(self):
        t0 = time.perf_counter()
        try:
            if self.playing and self.cap is not None and not self.dragging:
                target = min(self.n - 1, self.anchor_pos + int((t0 - self.anchor_t) * self.fps * self.speed))
                if target > self.pos:
                    ok = True
                    while self.pos < target - 1:
                        if not self.cap.grab():
                            ok = False
                            break
                        self.pos += 1
                        if time.perf_counter() - t0 > 0.08:      # cannot keep up: show what we have
                            break
                    if ok:
                        ok, im = self.cap.read()
                        if ok:
                            self.pos += 1
                            self.show(im)
                    if not ok or self.pos >= self.n - 1:
                        self.on_video_end()
                    elif time.perf_counter() - t0 > 0.05:        # slower than real time: re-anchor so lag never piles up
                        self.anchor_pos, self.anchor_t = self.pos, time.perf_counter()
                self.update_bar()
        finally:
            self.root.after(4, self.tick)

    def on_video_end(self):
        if self.cur + 1 < len(self.videos):
            self.say("הסרטון הסתיים - עוברים לבא")
            self.save_state()
            self.load_video(self.cur + 1, 0)
            self.anchor_pos, self.anchor_t = 0, time.perf_counter()
        else:
            self.playing = False
            self.say("זה היה הסרטון האחרון")

    def update_bar(self, force=False):
        now = time.perf_counter()
        if not force and now - self.last_ui < 0.12:
            return
        self.last_ui = now
        t = max(0, self.pos) / self.fps
        v = self.videos[self.cur]
        clock = (start_time(v["name"]) + datetime.timedelta(seconds=t)).strftime("%H:%M:%S")
        self.time_lbl.configure(text=f"{mmss(t)} / {mmss(self.n / self.fps)}")
        self.info.configure(text=f"Video {self.cur + 1}/{len(self.videos)}     {v['session']}     {v['name']}     clock {clock}")
        if not self.dragging:
            self.slider.set(max(0, self.pos))
        self.canvas.itemconfig(self.big, text="" if self.playing else "עצור")

    # ------------------------------------------------------------------------------------------- UI helpers
    def say(self, text):
        self.status.configure(text=text)

    def update_speed_label(self):
        self.speed_lbl.configure(text=f"מהירות x{self.speed:g}")

    def refresh_panel(self):
        self.panel.delete(0, "end")
        self.panel_rows = []
        for i, r in self.video_rows():
            self.panel.insert("end", f"{mmss(float(r['time_sec']))}   {r['label_he']}")
            self.panel_rows.append(i)
            color = next((c for code, _, c in LABELS if code == r["label"]), FG)
            self.panel.itemconfig("end", fg=color)

    def refresh_totals(self):
        cnt = collections.Counter(r["label"] for r in self.labels)
        self.totals.configure(text="\n".join(f"{cnt.get(code, 0)}      {he}" for code, he, _ in LABELS) + f"\n\n{len(self.labels)}      total")

    def open_menu(self):
        self.menu_open = True
        self.menu.place(relx=0.5, rely=0.5, anchor="center")
        self.lb.selection_clear(0, "end")
        self.lb.selection_set(0)
        self.lb.activate(0)
        self.lb.see(0)

    def close_menu(self):
        self.menu_open = False
        self.menu.place_forget()
        self.root.focus_set()

    def resume(self):
        self.playing = True
        self.anchor_pos, self.anchor_t = self.pos, time.perf_counter()
        self.canvas.itemconfig(self.big, text="")

    def pause(self):
        self.playing = False
        self.update_bar(force=True)

    def choose(self, i):
        self.add_label(LABELS[i][0])
        self.close_menu()
        self.resume()

    # ------------------------------------------------------------------------------------------- events
    def on_combo(self, _e=None):
        self.save_state()
        self.load_video(self.combo.current(), 0)
        if self.playing:
            self.resume()

    def on_slider_release(self, _e=None):
        self.dragging = False
        self.seek(int(float(self.slider.get())))
        if self.playing:
            self.resume()

    def on_panel_double(self, _e=None):
        sel = self.panel.curselection()
        if sel:
            r = self.labels[self.panel_rows[sel[0]]]
            self.seek(int(r["frame"]) - 60)
            self.resume()
        self.root.focus_set()

    def on_menu_click(self, _e=None):
        sel = self.lb.curselection()
        if sel:
            self.choose(sel[0])

    def on_key(self, e):
        k = e.keysym
        step = 1.0 if (e.state & 0x1) else 5.0           # Shift held
        if k in ("Left", "Right"):
            self.seek(self.pos + (1 if k == "Right" else -1) * step * self.fps)
            return
        if self.menu_open:
            if k in ("Up", "Down"):
                cur = self.lb.curselection()
                i = (cur[0] if cur else 0) + (1 if k == "Down" else -1)
                i = max(0, min(len(LABELS) - 1, i))
                self.lb.selection_clear(0, "end")
                self.lb.selection_set(i)
                self.lb.activate(i)
                self.lb.see(i)
            elif k in ("Return", "KP_Enter"):
                cur = self.lb.curselection()
                self.choose(cur[0] if cur else 0)
            elif k == "Escape":
                self.close_menu()
            elif k == "space":
                self.close_menu()
                self.resume()
            elif k.isdigit() and 1 <= int(k) <= len(LABELS):
                self.choose(int(k) - 1)
            return
        if k == "space":
            if self.playing:
                self.pause()
                self.open_menu()
            else:
                self.resume()
        elif k in ("m", "M") and not self.playing:
            self.open_menu()
        elif k in ("n", "N"):
            self.save_state()
            self.load_video(self.cur + 1, 0)
            if self.playing:
                self.resume()
        elif k in ("p", "P"):
            self.save_state()
            self.load_video(self.cur - 1, 0)
            if self.playing:
                self.resume()
        elif k in ("plus", "equal", "KP_Add"):
            self.speed = SPEEDS[min(len(SPEEDS) - 1, SPEEDS.index(self.speed) + 1)]
            self.update_speed_label()
            self.resume() if self.playing else None
        elif k in ("minus", "KP_Subtract"):
            self.speed = SPEEDS[max(0, SPEEDS.index(self.speed) - 1)]
            self.update_speed_label()
            self.resume() if self.playing else None
        elif k == "period" and not self.playing:
            self.seek(self.pos + 1)
        elif k == "comma" and not self.playing:
            self.seek(self.pos - 1)
        elif k in ("z", "Z"):
            if self.labels:
                last = self.labels.pop()
                self.write_labels()
                self.refresh_panel()
                self.refresh_totals()
                self.say(f"בוטל: {last['label_he']}  {mmss(float(last['time_sec']))}  ({last['video'][-22:]})")
        elif k == "Delete":
            sel = self.panel.curselection()
            if sel:
                r = self.labels.pop(self.panel_rows[sel[0]])
                self.write_labels()
                self.refresh_panel()
                self.refresh_totals()
                self.say(f"נמחק: {r['label_he']}  {mmss(float(r['time_sec']))}")

    def on_close(self):
        self.save_state()
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def selftest(app, shot_dir):
    """Scripted run used once to check the tool: play, stop, mark, screenshot, quit. Writes only to the labels path it was given."""
    from PIL import ImageGrab

    def grab(name):
        x, y, w, h = app.root.winfo_rootx(), app.root.winfo_rooty(), app.root.winfo_width(), app.root.winfo_height()
        ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(os.path.join(shot_dir, name))

    class E:
        def __init__(self, keysym, state=0):
            self.keysym, self.state = keysym, state

    app.root.attributes("-topmost", True)       # on top of the other windows so that the screenshots show this window
    app.root.lift()
    steps = [
        (300, lambda: app.resume()),
        (2500, lambda: app.on_key(E("space"))),                  # stop + menu
        (700, lambda: grab("selftest_menu.png")),
        (200, lambda: app.on_key(E("Down"))),
        (200, lambda: app.on_key(E("Return"))),                  # choose 2nd label, carry on
        (600, lambda: app.on_key(E("plus"))),
        (800, lambda: grab("selftest_playing.png")),
        (200, lambda: app.on_key(E("Left", 0))),                 # 5 s back
        (600, lambda: app.on_key(E("space"))),
        (200, lambda: app.on_key(E("1"))),                       # label 1 by number
        (500, lambda: app.on_key(E("n"))),                       # next video
        (1500, lambda: app.on_key(E("z"))),                      # undo
        (500, lambda: print("SELFTEST rows:", [(r["video"][-8:], r["frame"], r["label"]) for r in app.labels], "pos", app.pos, "video", app.cur)),
        (200, lambda: app.on_close()),
    ]
    t = 0
    for delay, fn in steps:
        t += delay
        app.root.after(t, fn)


def main():
    try:      # crisp text on high-DPI screens, and screenshot coordinates that match the window
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-video", type=int, default=None, help="1-based number of the video to start with (default: where you stopped)")
    ap.add_argument("--labels", default=os.path.join(HERE, "labels.csv"))
    ap.add_argument("--state", default=os.path.join(HERE, "state.json"))
    ap.add_argument("--selftest", default=None, help="folder for the screenshots of a scripted test run")
    a = ap.parse_args()
    videos = list_videos()
    if not videos:
        print("no videos found under raw_footage/")
        sys.exit(1)
    root = tk.Tk()
    app = App(root, videos, a.labels, a.state, None if a.start_video is None else a.start_video - 1)
    if a.selftest:
        selftest(app, a.selftest)
    root.mainloop()


if __name__ == "__main__":
    main()
