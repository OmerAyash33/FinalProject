"""
Football highlights - the desktop app (this is the program that tools/build_exe.py packages into FootballHighlights.exe).

Put videos in (drag them onto the list, or "add videos"), press start. The app
  1. extracts the detections of every video (the slow part, a few minutes per video on a GPU),
  2. asks you to click a few players of YOUR team so it knows your shirt colour (skipped when the session already has one),
  3. finds dribbles / passes / shots, picks the best 3 of each kind and renders the highlight videos.
Results: <output folder>/<session>_v2/  (All_Matches_Master_Highlights.mp4 = everything in one reel).

The heavy work runs in a second process of this same program ("--worker"), so the window never freezes and "cancel" really stops it.
Run from source:  .venv/Scripts/python.exe src/app.py
"""
import ctypes
import json
import multiprocessing
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts")


# ------------------------------------------------------------------------------------------------ worker process
def run_worker(job_path):
    """Second process: Stage 1/audio ("extract") or Stage 2 + selection + Stage 3 ("analyze") for the videos in the job file."""
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)
    os.environ["FOOTBALL_LIVE_PREVIEW"] = "0"      # no preview window inside the app
    # Everything the pipeline prints goes into a log file that the window follows (a windowed exe has no usable stdout, and the file stays
    # as a record of the run).
    os.makedirs(os.path.dirname(job["log"]), exist_ok=True)
    sys.stdout = sys.stderr = open(job["log"], "w", encoding="utf-8", errors="replace", buffering=1)
    import torch  # noqa: F401  (must come before pandas, see main.py)
    import main
    from app_paths import DETECTOR_VERSION
    out_dir = os.path.join(job["out_root"], f"{job['session']}_{DETECTOR_VERSION}")
    main.process_session(os.path.dirname(job["videos"][0]), out_dir, video_files=job["videos"], stages=job["stage"])


# ------------------------------------------------------------------------------------------------ the window
BG, FG, PANEL = "#1b1b1b", "#eeeeee", "#262626"
FONT = ("Segoe UI", 11)


class App:
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        try:
            from tkinterdnd2 import TkinterDnD, DND_FILES
            self.root = TkinterDnD.Tk()
            self.dnd = DND_FILES
        except Exception:   # drag & drop is a convenience: the buttons work without it
            self.root = tk.Tk()
            self.dnd = None
        r = self.root
        r.title("Football highlights")
        r.configure(bg=BG)
        r.geometry("860x700")
        r.minsize(760, 620)
        from app_paths import APP_ROOT
        self.app_root = APP_ROOT
        self.videos = []
        self.proc = None
        self.worker = None
        self.cancelled = False
        self.msgs = queue.Queue()
        self.session_var = tk.StringVar()
        self.out_var = tk.StringVar(value=os.path.join(APP_ROOT, "output"))
        self.pick_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="")
        self._build()
        r.after(150, self._pump)

    # ---- layout
    def _build(self):
        tk, ttk, r = self.tk, self.ttk, self.root
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TProgressbar", troughcolor=PANEL, background="#4be04b", bordercolor=BG, lightcolor="#4be04b", darkcolor="#4be04b")
        top = tk.Frame(r, bg=BG)
        top.pack(fill="x", padx=14, pady=(12, 4))
        tk.Label(top, text="סרטוני כדורגל מצלמת חזה: הוספת סרטונים וקבלת קטעים מוקדשים", bg=BG, fg=FG, font=("Segoe UI", 14, "bold")).pack(anchor="e")

        box = tk.Frame(r, bg=PANEL)
        box.pack(fill="both", padx=14, pady=6)
        row = tk.Frame(box, bg=PANEL)
        row.pack(fill="both", padx=10, pady=10)
        side = tk.Frame(row, bg=PANEL)
        side.pack(side="right", padx=(10, 0), fill="y")
        for text, cmd in (("הוסף סרטונים", self.add_files), ("הוסף תיקייה", self.add_folder), ("הסר מסומנים", self.remove_selected), ("נקה הכל", self.clear)):
            tk.Button(side, text=text, command=cmd, bg="#3a3a3a", fg="white", bd=0, font=FONT, padx=12, pady=5, cursor="hand2").pack(fill="x", pady=2)
        listf = tk.Frame(row, bg=PANEL)
        listf.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(listf)
        sb.pack(side="right", fill="y")
        self.listbox = tk.Listbox(listf, height=8, bg="#111111", fg=FG, selectmode="extended", font=("Segoe UI", 10), yscrollcommand=sb.set,
                                  highlightthickness=0, bd=0, activestyle="none")
        self.listbox.pack(side="left", fill="both", expand=True)
        sb.config(command=self.listbox.yview)
        hint = "אפשר גם לגרור סרטונים לרשימה" if self.dnd else ""
        tk.Label(box, text=hint, bg=PANEL, fg="#9a9a9a", font=("Segoe UI", 10)).pack(anchor="e", padx=10, pady=(0, 6))
        if self.dnd:
            self.listbox.drop_target_register(self.dnd)
            self.listbox.dnd_bind("<<Drop>>", self._on_drop)

        opt = tk.Frame(r, bg=BG)
        opt.pack(fill="x", padx=14, pady=4)
        self._field(opt, "שם הסשן:", self.session_var)
        out = self._field(opt, "תיקיית תוצאות:", self.out_var)
        tk.Button(out, text="בחר", command=self.browse_out, bg="#3a3a3a", fg="white", bd=0, font=FONT, padx=10, cursor="hand2").pack(side="left", padx=(0, 6))
        tk.Checkbutton(opt, text="לבחור צבע חולצה של הקבוצה שלנו (מומלץ)", variable=self.pick_var, bg=BG, fg=FG, selectcolor=PANEL, activebackground=BG,
                       activeforeground=FG, font=FONT).pack(anchor="e", pady=(4, 0))

        btns = tk.Frame(r, bg=BG)
        btns.pack(fill="x", padx=14, pady=8)
        self.start_btn = tk.Button(btns, text="התחל עיבוד", command=self.start, bg="#2f7d2f", fg="white", bd=0, font=("Segoe UI", 12, "bold"), padx=24, pady=8, cursor="hand2")
        self.start_btn.pack(side="right")
        self.cancel_btn = tk.Button(btns, text="ביטול", command=self.cancel, bg="#8a3a3a", fg="white", bd=0, font=FONT, padx=14, pady=8, state="disabled", cursor="hand2")
        self.cancel_btn.pack(side="right", padx=8)
        tk.Button(btns, text="בחר צבע חולצה", command=self.pick_colors_now, bg="#3a3a3a", fg="white", bd=0, font=FONT, padx=14, pady=8, cursor="hand2").pack(side="right")
        tk.Button(btns, text="פתח תיקיית תוצאות", command=self.open_results, bg="#1f5fbf", fg="white", bd=0, font=FONT, padx=14, pady=8, cursor="hand2").pack(side="left")

        self.progress = ttk.Progressbar(r, mode="indeterminate")
        self.progress.pack(fill="x", padx=14, pady=(2, 2))
        tk.Label(r, textvariable=self.status_var, bg=BG, fg="#ffd060", font=FONT).pack(anchor="e", padx=14)
        logf = tk.Frame(r, bg=BG)
        logf.pack(fill="both", expand=True, padx=14, pady=(4, 12))
        lsb = tk.Scrollbar(logf)
        lsb.pack(side="right", fill="y")
        self.log_text = tk.Text(logf, height=10, bg="#0f0f0f", fg="#c8c8c8", font=("Consolas", 9), wrap="none", yscrollcommand=lsb.set, bd=0, state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True)
        lsb.config(command=self.log_text.yview)

    def _field(self, parent, label, var):
        tk = self.tk
        f = tk.Frame(parent, bg=BG)
        f.pack(fill="x", pady=2)
        tk.Label(f, text=label, bg=BG, fg=FG, font=FONT, width=14, anchor="e").pack(side="right")
        e = tk.Entry(f, textvariable=var, bg="#111111", fg=FG, insertbackground=FG, bd=0, font=("Segoe UI", 10))
        e.pack(side="right", fill="x", expand=True, padx=8, ipady=4)
        return f

    # ---- video list
    def _add(self, paths):
        for p in paths:
            if os.path.isdir(p):
                for name in sorted(os.listdir(p)):
                    if name.lower().endswith(VIDEO_EXT):
                        self._add([os.path.join(p, name)])
            elif p.lower().endswith(VIDEO_EXT) and p not in self.videos:
                self.videos.append(p)
                self.listbox.insert("end", p)
        if self.videos and not self.session_var.get():
            self.session_var.set(os.path.basename(os.path.dirname(os.path.abspath(self.videos[0]))) or "session")

    def _on_drop(self, event):
        self._add(self.root.tk.splitlist(event.data))

    def add_files(self):
        from tkinter import filedialog
        self._add(filedialog.askopenfilenames(parent=self.root, title="Videos", filetypes=[("Video", "*.mp4 *.MP4 *.mov *.avi *.mkv *.m4v *.mts"), ("All", "*.*")]))

    def add_folder(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(parent=self.root, title="Folder with videos")
        if d:
            self._add([d])

    def remove_selected(self):
        for i in reversed(self.listbox.curselection()):
            self.listbox.delete(i)
            del self.videos[i]

    def clear(self):
        self.listbox.delete(0, "end")
        self.videos.clear()

    def browse_out(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(parent=self.root, title="Results folder")
        if d:
            self.out_var.set(d)

    # ---- helpers
    def session(self):
        return (self.session_var.get().strip() or "session").replace("/", "_").replace("\\", "_")

    def out_dir(self, cfg=None):
        from app_paths import DETECTOR_VERSION
        cfg = cfg or self.snapshot()
        return os.path.join(cfg["out_root"], f"{cfg['session']}_{DETECTOR_VERSION}")

    def snapshot(self):
        """The window's settings as plain values. Taken on the main thread: the worker thread must not touch Tk variables."""
        return {"session": self.session(), "out_root": self.out_var.get(), "pick": bool(self.pick_var.get()), "videos": list(self.videos)}

    def open_results(self):
        d = self.out_dir()
        os.makedirs(d, exist_ok=True)
        os.startfile(d)

    def log(self, text):
        self.msgs.put(("log", text))

    def set_status(self, text):
        self.msgs.put(("status", text))

    def _pump(self):
        try:
            while True:
                kind, val = self.msgs.get_nowait()
                if kind == "log":
                    self.log_text.config(state="normal")
                    self.log_text.insert("end", val.rstrip("\n") + "\n")
                    self.log_text.see("end")
                    self.log_text.config(state="disabled")
                elif kind == "status":
                    self.status_var.set(val)
                elif kind == "busy":
                    if val:
                        self.progress.start(12)
                    else:
                        self.progress.stop()
                    self.start_btn.config(state="disabled" if val else "normal")
                    self.cancel_btn.config(state="normal" if val else "disabled")
                elif kind == "call":
                    val()
        except queue.Empty:
            pass
        self.root.after(150, self._pump)

    # ---- running
    def _spawn(self, stage, cfg):
        """Run one worker stage; stream its output into the log. Returns True when it finished cleanly."""
        log_path = os.path.join(cfg["out_root"], "logs", f"{cfg['session']}_{stage}.log")
        job = {"stage": stage, "session": cfg["session"], "out_root": cfg["out_root"], "videos": cfg["videos"], "log": log_path}
        fd, job_path = tempfile.mkstemp(suffix=".json", prefix="football_job_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(job, f)
        cmd = [sys.executable, "--worker", job_path] if getattr(sys, "frozen", False) else [sys.executable, os.path.abspath(__file__), "--worker", job_path]
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", FOOTBALL_LIVE_PREVIEW="0")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        open(log_path, "w").close()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, creationflags=flags, cwd=self.app_root)
        n_done = 0
        with open(log_path, "r", encoding="utf-8", errors="replace") as logf:
            while True:
                alive = self.proc.poll() is None
                for line in logf.read().splitlines():
                    self.log(line)
                    if "Processing Video:" in line:
                        n_done += 1
                        self.set_status(f"{'חילוץ נתונים' if stage == 'extract' else 'ניתוח'}: סרטון {n_done} מתוך {len(cfg['videos'])}")
                    elif "Creating Master Highlight Reel" in line:
                        self.set_status("מרכיב את הסרטון המסכם")
                if not alive:
                    break
                time.sleep(0.3)
        code = self.proc.returncode
        try:
            os.remove(job_path)
        except OSError:
            pass
        return code == 0 and not self.cancelled

    def start(self):
        if not self.videos:
            self.set_status("הוסף קודם סרטונים")
            return
        self.cancelled = False
        self.msgs.put(("busy", True))
        self.worker = threading.Thread(target=self._run_all, args=(self.snapshot(),), daemon=True)
        self.worker.start()

    def _run_all(self, cfg):
        t0 = time.time()
        try:
            self.set_status("חילוץ נתונים מהסרטונים (השלב הארוך)")
            if not self._spawn("extract", cfg):
                self.set_status("בוטל" if self.cancelled else "החילוץ נכשל, ראה את היומן")
                return
            from team_colors import load_reference
            if cfg["pick"] and load_reference(cfg["session"]) is None:
                self.set_status("בחר את צבע החולצה של הקבוצה שלך")
                self._ask_colors_blocking(cfg)
            self.set_status("מנתח אירועים ומרכיב סרטונים")
            if not self._spawn("analyze", cfg):
                self.set_status("בוטל" if self.cancelled else "הניתוח נכשל, ראה את היומן")
                return
            self.set_status(f"הסתיים ב-{(time.time() - t0) / 60:.1f} דקות. התוצאות בתיקייה למטה")
            self.log(f"DONE. Results: {self.out_dir(cfg)}")
        except Exception as e:   # never let the thread die silently
            self.log(f"ERROR: {e!r}")
            self.set_status("שגיאה, ראה את היומן")
        finally:
            self.msgs.put(("busy", False))

    def cancel(self):
        self.cancelled = True
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        self.set_status("מבטל...")

    # ---- team colour
    def _csv_pairs(self, cfg):
        out = self.out_dir(cfg)
        return [(v, os.path.join(out, f"{os.path.splitext(os.path.basename(v))[0]}_data.csv")) for v in cfg["videos"]]

    def _ask_colors_blocking(self, cfg):
        done = threading.Event()

        def open_window():
            try:
                self._open_picker(cfg)
            finally:
                done.set()
        self.msgs.put(("call", open_window))
        done.wait()

    def _open_picker(self, cfg):
        import team_color_picker
        pairs = [(v, c) for v, c in self._csv_pairs(cfg) if os.path.exists(c)]
        if not pairs:
            self.set_status("אין עדיין נתונים שחולצו: התחל עיבוד קודם")
            return
        step = max(1, len(pairs) // 4)   # a few videos are plenty: ten players come out of one
        team_color_picker.pick_team_colors(cfg["session"], pairs[::step], parent=self.root)

    def pick_colors_now(self):
        if not self.videos:
            self.set_status("הוסף קודם סרטונים")
            return
        self._open_picker(self.snapshot())


def main():
    multiprocessing.freeze_support()
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        run_worker(sys.argv[2])
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)   # sharp text on high-DPI screens
    except Exception:
        pass
    App().root.mainloop()


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
