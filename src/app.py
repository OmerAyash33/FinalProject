"""
Football highlights - the desktop app (this is the program that tools/build_exe.py packages into FootballHighlights.exe).

The window is five numbered steps:
  1. Add your videos (drag them onto the list, or "Add videos").
  2. Name the session and choose the output folder.
  3. Choose whether to pick your team's shirt colour (recommended).
  4. Press "Start processing". The app then runs, in order:
       stage 1  extract the detections of every video (the slow part, a few minutes per video on a GPU),
       stage 2  ask you to click a few players of YOUR team so it knows your shirt colour (skipped when not wanted / already known),
       stage 3  find dribbles / passes / shots, pick the best 3 of each kind and render the highlight videos.
  5. Results: clickable links to the highlight reel and the results folder appear in the window when the run is finished.
Results are written to <output folder>/<session>_v2/  (All_Matches_Master_Highlights.mp4 = everything in one reel).

The heavy work runs in a second process of this same program ("--worker"), so the window never freezes and "Cancel" really stops it.
Run from source:  .venv/Scripts/python.exe src/app.py
"""
import ctypes
import glob
import json
import multiprocessing
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts")
MASTER_NAME = "All_Matches_Master_Highlights.mp4"
PATH_IN_LINE = re.compile(r"[A-Za-z]:[\\/].*$")   # an absolute Windows path at the end of a log line -> shown as a clickable link


# ------------------------------------------------------------------------------------------------ worker process
def run_worker(job_path):
    """Second process: Stage 1/audio ("extract") or Stage 2 + selection + Stage 3 ("analyze") for the videos in the job file."""
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)
    if job.get("preview_file"):                    # the window of the app shows this JPEG (the worker has no window of its own)
        os.environ["FOOTBALL_LIVE_PREVIEW"] = "1"
        os.environ["FOOTBALL_PREVIEW_FILE"] = job["preview_file"]
    else:
        os.environ["FOOTBALL_LIVE_PREVIEW"] = "0"
    os.environ["FOOTBALL_IN_APP"] = "1"            # the app makes the links itself (links.print_result_links prints plain paths only)
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
ACCENT, LINK, MUTED, WARN = "#4be04b", "#5ab0ff", "#9a9a9a", "#ffd060"
FONT = ("Segoe UI", 11)
FONT_SMALL = ("Segoe UI", 10)


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
        r.title("Football chest-cam highlights")
        r.configure(bg=BG)
        r.geometry(f"900x{min(1000, r.winfo_screenheight() - 80)}")
        r.minsize(820, 640)
        from app_paths import APP_ROOT
        self.app_root = APP_ROOT
        self.videos = []
        self.proc = None
        self.worker = None
        self.cancelled = False
        self.msgs = queue.Queue()
        self.link_count = 0
        self.session_var = tk.StringVar()
        self.out_var = tk.StringVar(value=os.path.join(APP_ROOT, "output"))
        self.pick_var = tk.BooleanVar(value=True)
        self.live_var = tk.BooleanVar(value=True)
        self._live_win = self._live_frame = self._live_label = self._live_img = None
        self._live_path, self._live_mtime = None, 0
        self.status_var = tk.StringVar(value="")
        self._build()
        r.update_idletasks()   # the screen's text scale decides how tall the five steps are: size the window to its content (as far as the screen allows) so Step 4's progress bar and log are never cut off
        r.geometry(f"900x{max(640, min(r.winfo_reqheight() + 10, r.winfo_screenheight() - 80))}")
        r.after(150, self._pump)

    # ---- layout
    def _step(self, number, title, side="top", expand=False):
        """A numbered panel: 'Step N - title' header, returns (panel, the empty body frame)."""
        tk = self.tk
        outer = tk.Frame(self.root, bg=PANEL)
        outer.pack(side=side, fill="both" if expand else "x", expand=expand, padx=14, pady=5)
        head = tk.Frame(outer, bg=PANEL)
        head.pack(fill="x", padx=10, pady=(8, 2))
        tk.Label(head, text=f"Step {number}", bg=ACCENT, fg="#0b2a0b", font=("Segoe UI", 10, "bold"), padx=8, pady=1).pack(side="left")
        tk.Label(head, text=title, bg=PANEL, fg=FG, font=("Segoe UI", 12, "bold")).pack(side="left", padx=10)
        body = tk.Frame(outer, bg=PANEL)
        body.pack(fill="both" if expand else "x", expand=expand, padx=10, pady=(2, 10))
        return outer, body

    def _button(self, parent, text, command, bg="#3a3a3a", **kw):
        opts = dict(bg=bg, fg="white", bd=0, font=FONT, padx=12, pady=5, cursor="hand2", activebackground=bg, activeforeground="white")
        opts.update(kw)
        return self.tk.Button(parent, text=text, command=command, **opts)

    def _build(self):
        tk, ttk, r = self.tk, self.ttk, self.root
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TProgressbar", troughcolor=PANEL, background=ACCENT, bordercolor=BG, lightcolor=ACCENT, darkcolor=ACCENT)
        head = tk.Frame(r, bg=BG)
        head.pack(fill="x", padx=14, pady=(12, 4))
        tk.Label(head, text="Football chest-cam highlights", bg=BG, fg=FG, font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(head, text="Add your match videos and get the best dribbles, passes and shots as highlight clips.", bg=BG, fg=MUTED,
                 font=FONT_SMALL).pack(anchor="w")

        # Step 1 - videos
        _, body = self._step(1, "Add your videos")
        row = tk.Frame(body, bg=PANEL)
        row.pack(fill="x")
        listf = tk.Frame(row, bg=PANEL)
        listf.pack(fill="x", expand=True)
        sb = tk.Scrollbar(listf)
        sb.pack(side="right", fill="y")
        self.listbox = tk.Listbox(listf, height=3, bg="#111111", fg=FG, selectmode="extended", font=FONT_SMALL, yscrollcommand=sb.set,
                                  highlightthickness=0, bd=0, activestyle="none")
        self.listbox.pack(side="left", fill="both", expand=True)
        sb.config(command=self.listbox.yview)
        side = tk.Frame(body, bg=PANEL)
        side.pack(fill="x", pady=(6, 0))
        for text, cmd in (("Add videos", self.add_files), ("Add folder", self.add_folder), ("Remove selected", self.remove_selected), ("Clear all", self.clear)):
            self._button(side, text, cmd, pady=3).pack(side="left", padx=(0, 6))
        if self.dnd:
            tk.Label(side, text="You can also drag videos into the list.", bg=PANEL, fg=MUTED, font=FONT_SMALL).pack(side="left", padx=8)
            self.listbox.drop_target_register(self.dnd)
            self.listbox.dnd_bind("<<Drop>>", self._on_drop)

        # Step 2 - session + output
        _, body = self._step(2, "Name the session and choose where the results go")
        self._field(body, "Session name:", self.session_var)
        out = self._field(body, "Output folder:", self.out_var)
        self._button(out, "Browse...", self.browse_out, padx=10, pady=2).pack(side="left", padx=(8, 0))

        # Step 3 - shirt colour
        _, body = self._step(3, "Tell the app which shirt colour is yours")
        line = tk.Frame(body, bg=PANEL)
        line.pack(fill="x")
        tk.Checkbutton(line, text="After the extraction, pick our team's shirt colour (recommended)", variable=self.pick_var, bg=PANEL, fg=FG,
                       selectcolor="#111111", activebackground=PANEL, activeforeground=FG, font=FONT).pack(side="left")
        self._button(line, "Pick shirt colour now", self.pick_colors_now).pack(side="right")
        tk.Label(body, text="You click a few players of your team and the app learns the colour. Without it, the app guesses from light / dark shirts.", bg=PANEL, fg=MUTED, font=FONT_SMALL, justify="left", wraplength=800
                 ).pack(anchor="w", pady=(4, 0))

        # Step 5 - results (links)
        _, body = self._step(5, "Open your results", side="bottom")
        self.results_box = tk.Frame(body, bg=PANEL)
        self.results_box.pack(fill="x")
        self._results_placeholder()
        self._button(body, "Open results folder", self.open_results, bg="#1f5fbf").pack(anchor="w", pady=(8, 0))

        # Step 4 - start + progress + log
        _, body = self._step(4, "Start processing", expand=True)
        btns = tk.Frame(body, bg=PANEL)
        btns.pack(fill="x")
        self.start_btn = self._button(btns, "Start processing", self.start, bg="#2f7d2f", font=("Segoe UI", 12, "bold"), padx=24, pady=8)
        self.start_btn.pack(side="left")
        self.cancel_btn = self._button(btns, "Cancel", self.cancel, bg="#8a3a3a", pady=8, state="disabled")
        self.cancel_btn.pack(side="left", padx=8)
        tk.Label(btns, textvariable=self.status_var, bg=PANEL, fg=WARN, font=FONT, anchor="w", justify="left", wraplength=470).pack(side="left", padx=12, fill="x", expand=True)
        tk.Checkbutton(body, text="Show a live view of the detections while they are extracted (recommended)", variable=self.live_var, bg=PANEL, fg=FG,
                       selectcolor="#111111", activebackground=PANEL, activeforeground=FG, font=FONT_SMALL).pack(anchor="w", pady=(6, 0))
        self.progress = ttk.Progressbar(body, mode="indeterminate")
        self.progress.pack(fill="x", pady=(8, 4))
        logf = tk.Frame(body, bg=PANEL)
        logf.pack(fill="both", expand=True)
        lsb = tk.Scrollbar(logf)
        lsb.pack(side="right", fill="y")
        self.log_text = tk.Text(logf, height=6, bg="#0f0f0f", fg="#c8c8c8", font=("Consolas", 9), wrap="none", yscrollcommand=lsb.set, bd=0, state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True)
        lsb.config(command=self.log_text.yview)
        self.log_text.tag_config("link", foreground=LINK, underline=True)
        self.log_text.tag_bind("link", "<Enter>", lambda e: self.log_text.config(cursor="hand2"))
        self.log_text.tag_bind("link", "<Leave>", lambda e: self.log_text.config(cursor=""))

    def _field(self, parent, label, var):
        tk = self.tk
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill="x", pady=2)
        tk.Label(f, text=label, bg=PANEL, fg=FG, font=FONT, width=14, anchor="w").pack(side="left")
        e = tk.Entry(f, textvariable=var, bg="#111111", fg=FG, insertbackground=FG, bd=0, font=FONT_SMALL)
        e.pack(side="left", fill="x", expand=True, ipady=4)
        return f

    # ---- results: clickable links
    def _results_placeholder(self):
        for w in self.results_box.winfo_children():
            w.destroy()
        self.tk.Label(self.results_box, text="When the run is finished, the links to your highlight videos appear here.", bg=PANEL, fg=MUTED,
                      font=FONT_SMALL).pack(anchor="w")

    def _link_row(self, parent, label, path, text=None):
        """`label` and below it a clickable, underlined link (its text is the full path unless `text` is given)."""
        tk = self.tk
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill="x", pady=2)
        tk.Label(f, text=label, bg=PANEL, fg=FG, font=("Segoe UI", 10, "bold"), anchor="w").pack(anchor="w")
        link = tk.Label(f, text=text or path, bg=PANEL, fg=LINK, font=("Segoe UI", 10, "underline"), cursor="hand2", anchor="w", justify="left", wraplength=820)
        link.pack(anchor="w")
        link.bind("<Button-1>", lambda e, p=path: self.open_link(p))
        return link

    def show_results(self, out):
        """Step 5: one clickable row per result that exists."""
        for w in self.results_box.winfo_children():
            w.destroy()
        master = os.path.join(out, MASTER_NAME)
        if os.path.exists(master):
            self._link_row(self.results_box, "Highlight reel (all videos in one):", master)
        self._link_row(self.results_box, "Results folder (every file of this run):", out)
        per_video = [p for p in sorted(glob.glob(os.path.join(out, "*_highlights.mp4"))) if os.path.basename(p).lower() != MASTER_NAME.lower()]
        if len(per_video) > 1:   # with one video the reel above IS that video's highlights
            self.tk.Label(self.results_box, text="Highlights of each video:", bg=PANEL, fg=FG, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 0))
            for p in per_video[:6]:
                link = self.tk.Label(self.results_box, text=os.path.basename(p), bg=PANEL, fg=LINK, font=("Segoe UI", 10, "underline"), cursor="hand2")
                link.pack(anchor="w")
                link.bind("<Button-1>", lambda e, path=p: self.open_link(path))

    def open_link(self, path):
        from links import open_path
        try:
            open_path(path)
        except OSError as e:
            self.set_status(f"Could not open {path}: {e}")

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
        return {"session": self.session(), "out_root": self.out_var.get(), "pick": bool(self.pick_var.get()), "live": bool(self.live_var.get()), "videos": list(self.videos)}

    def open_results(self):
        d = self.out_dir()
        os.makedirs(d, exist_ok=True)
        self.open_link(d)

    def log(self, text):
        self.msgs.put(("log", text))

    def set_status(self, text):
        self.msgs.put(("status", text))

    def _insert_log(self, line):
        """Append one line to the log; an existing absolute path at the end of the line becomes a clickable link."""
        t = self.log_text
        line = line.rstrip("\n")
        t.config(state="normal")
        m = PATH_IN_LINE.search(line)
        path = m.group(0).strip() if m else None
        if path and os.path.exists(path):
            self.link_count += 1
            tag = f"link{self.link_count}"
            t.insert("end", line[:m.start()])
            t.insert("end", path, ("link", tag))
            t.tag_bind(tag, "<Button-1>", lambda e, p=path: self.open_link(p))
            t.insert("end", "\n")
        else:
            t.insert("end", line + "\n")
        t.see("end")
        t.config(state="disabled")

    def _pump(self):
        try:
            while True:
                kind, val = self.msgs.get_nowait()
                if kind == "log":
                    self._insert_log(val)
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
        self._live_tick()
        self.root.after(150, self._pump)

    # ---- live view (the worker writes the latest annotated frame to a JPEG about 6 times a second; this window shows it)
    def _live_open(self, path):
        tk = self.tk
        try:
            import PIL.ImageTk  # noqa: F401
        except ImportError:
            return
        self._live_close()
        self._live_path, self._live_mtime = path, 0
        w = tk.Toplevel(self.root)
        w.title("Live view - what the detector sees right now")
        w.configure(bg=BG)
        x = self.root.winfo_x() + self.root.winfo_width() + 8
        if x + 820 > self.root.winfo_screenwidth():
            x = max(0, self.root.winfo_screenwidth() - 820)
        w.geometry(f"800x500+{x}+{self.root.winfo_y()}")
        w.minsize(320, 220)
        tk.Label(w, text="Players are drawn in their team's colour (yellow = ours, magenta = the others, grey = not decided yet), the ball is white, the goal blue. "
                         "The extraction itself is not slowed down.", bg=BG, fg=MUTED, font=FONT_SMALL, wraplength=780, justify="left").pack(anchor="w", padx=8, pady=(6, 2))
        frame = tk.Frame(w, bg="#000000")
        frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        frame.pack_propagate(False)
        label = tk.Label(frame, text="Waiting for the first frame...", bg="#000000", fg=MUTED, font=FONT)
        label.place(relx=0.5, rely=0.5, anchor="center")
        w.protocol("WM_DELETE_WINDOW", self._live_close)
        self._live_win, self._live_frame, self._live_label = w, frame, label

    def _live_close(self):
        if self._live_win is not None:
            try:
                self._live_win.destroy()
            except Exception:
                pass
        self._live_win = self._live_frame = self._live_label = self._live_img = None
        if self._live_path:
            for p in (self._live_path, self._live_path + ".tmp"):
                try:
                    os.remove(p)
                except OSError:
                    pass
        self._live_path = None

    def _live_tick(self):
        if self._live_win is None or not self._live_path:
            return
        try:
            m = os.stat(self._live_path).st_mtime_ns
            if m == self._live_mtime:
                return
            from PIL import Image, ImageTk
            im = Image.open(self._live_path)
            im.load()
            self._live_mtime = m
            fw, fh = max(self._live_frame.winfo_width(), 50), max(self._live_frame.winfo_height(), 50)
            s = min(fw / im.width, fh / im.height)
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BILINEAR)
            self._live_img = ImageTk.PhotoImage(im)      # keep a reference, Tk drops the picture otherwise
            self._live_label.config(image=self._live_img, text="")
        except Exception:
            return    # file missing / being replaced right now: the next tick (150 ms) picks it up

    # ---- running
    def _spawn(self, stage, cfg, stage_no, stage_total):
        """Run one worker stage; stream its output into the log. Returns True when it finished cleanly."""
        log_path = os.path.join(cfg["out_root"], "logs", f"{cfg['session']}_{stage}.log")
        job = {"stage": stage, "session": cfg["session"], "out_root": cfg["out_root"], "videos": cfg["videos"], "log": log_path}
        live_path = None
        if stage == "extract" and cfg.get("live"):
            live_path = os.path.join(tempfile.gettempdir(), f"football_live_{os.getpid()}.jpg")
            job["preview_file"] = live_path
            self.msgs.put(("call", lambda p=live_path: self._live_open(p)))
        fd, job_path = tempfile.mkstemp(suffix=".json", prefix="football_job_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(job, f)
        cmd = [sys.executable, "--worker", job_path] if getattr(sys, "frozen", False) else [sys.executable, os.path.abspath(__file__), "--worker", job_path]
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", FOOTBALL_LIVE_PREVIEW="0", FOOTBALL_IN_APP="1")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        open(log_path, "w").close()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, creationflags=flags, cwd=self.app_root)
        n_done = 0
        what = "extracting the detections" if stage == "extract" else "finding events and rendering the highlights"
        with open(log_path, "r", encoding="utf-8", errors="replace") as logf:
            while True:
                alive = self.proc.poll() is None
                for line in logf.read().splitlines():
                    self.log(line)
                    if "Processing Video:" in line:
                        n_done += 1
                        self.set_status(f"Stage {stage_no} of {stage_total}: {what} - video {n_done} of {len(cfg['videos'])}")
                    elif "Creating Master Highlight Reel" in line:
                        self.set_status(f"Stage {stage_no} of {stage_total}: putting the highlight reel together")
                if not alive:
                    break
                time.sleep(0.3)
        code = self.proc.returncode
        if live_path:
            self.msgs.put(("call", self._live_close))
            for q in (live_path, live_path + ".tmp"):    # also when the user had already closed the live window and the worker kept writing
                try:
                    os.remove(q)
                except OSError:
                    pass
        try:
            os.remove(job_path)
        except OSError:
            pass
        return code == 0 and not self.cancelled

    def start(self):
        if not self.videos:
            self.set_status("Step 1 is missing: add at least one video first")
            return
        self.cancelled = False
        self.msgs.put(("call", self._results_placeholder))
        self.msgs.put(("busy", True))
        self.worker = threading.Thread(target=self._run_all, args=(self.snapshot(),), daemon=True)
        self.worker.start()

    def _run_all(self, cfg):
        t0 = time.time()
        out = self.out_dir(cfg)
        try:
            from team_colors import load_reference
            will_pick = cfg["pick"] and load_reference(cfg["session"]) is None
            total = 3 if will_pick else 2
            self.set_status(f"Stage 1 of {total}: extracting the detections (the long part)")
            if not self._spawn("extract", cfg, 1, total):
                self.set_status("Cancelled" if self.cancelled else "The extraction failed - see the log above")
                return
            if will_pick:
                self.set_status(f"Stage 2 of {total}: pick your team's shirt colour in the window that opened")
                self._ask_colors_blocking(cfg)
            self.set_status(f"Stage {total} of {total}: finding events and rendering the highlights")
            if not self._spawn("analyze", cfg, total, total):
                self.set_status("Cancelled" if self.cancelled else "The analysis failed - see the log above")
                return
            self.set_status(f"Finished in {(time.time() - t0) / 60:.1f} minutes - click a link in Step 5 to open your results")
            self.log(f"Finished in {(time.time() - t0) / 60:.1f} minutes. The links above (and in Step 5) open your results.")
            self.msgs.put(("call", lambda: self.show_results(out)))
        except Exception as e:   # never let the thread die silently
            self.log(f"ERROR: {e!r}")
            self.set_status("Something went wrong - see the log above")
        finally:
            self.msgs.put(("busy", False))

    def cancel(self):
        self.cancelled = True
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        self.set_status("Cancelling...")

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
            self.set_status("No extracted data yet - press Start processing first")
            return
        step = max(1, len(pairs) // 4)   # a few videos are plenty: seven players come out of one
        team_color_picker.pick_team_colors(cfg["session"], pairs[::step], parent=self.root)

    def pick_colors_now(self):
        if not self.videos:
            self.set_status("Step 1 is missing: add at least one video first")
            return
        self._open_picker(self.snapshot())


def run_selftest(out_path):
    """`FootballHighlights.exe --selftest out.txt`: imports everything the window and the worker need (for checking a packaged build)."""
    lines = []
    for name in ("torch", "cv2", "numpy", "pandas", "sklearn.cluster", "scipy.signal", "av", "PIL.ImageTk", "tkinter", "tkinterdnd2", "ultralytics",
                 "team_color_picker", "main", "tactical_analyzer", "highlight_generator", "links", "shot_evidence"):
        try:
            __import__(name)
            lines.append(f"ok    {name}")
        except Exception as e:   # report, never crash: the point is to list every missing piece
            lines.append(f"FAIL  {name}: {e!r}")
    from app_paths import APP_ROOT, BUNDLE_ROOT, DETECTOR_VERSION
    lines.append(f"APP_ROOT={APP_ROOT} BUNDLE_ROOT={BUNDLE_ROOT} model_exists={os.path.exists(os.path.join(BUNDLE_ROOT, 'models', f'match_detector_{DETECTOR_VERSION}.pt'))}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    multiprocessing.freeze_support()
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        run_worker(sys.argv[2])
        return
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        run_selftest(sys.argv[2])
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)   # sharp text on high-DPI screens
    except Exception:
        pass
    App().root.mainloop()


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
