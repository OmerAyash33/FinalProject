"""
A rough progress meter for the desktop app (src/app.py): how many videos are done and about how long is left.

It is deliberately casual - it only has to be roughly right, not exact:
  * Stage 1 (extraction) is almost all of the time. Its size is the video's frame count; its speed is measured from the worker's own log
    ("Extraction: Processed 1200 frames..." every 1200 frames, "Video X: extraction done in N seconds."), and between two log lines the
    position is interpolated from that speed, so the meter moves smoothly although the log lines come only every ~30 s.
  * Choosing the shirt colour is the user's own time: the clock for the bar stops while the window is open.
  * Stage 3 (events + highlights) is short; it is a fixed guess per video and per chosen move (the real run is ~5 s per move), and uses the
    real number of moves once the log says how many were chosen.
The app calls feed() with every log line and snapshot() every ~5 seconds. No Tk in here, so it can be tested on a recorded log.
"""
import re

DEFAULT_FPS = 35.0              # until the speed has been measured (this machine: ~40 frames/s on a 1080p video with the live view on)
FALLBACK_FRAMES = 9000          # a 5-minute video at 30 fps, when the frame count of a file cannot be read
MODEL_LOAD_S = 2.0              # the detector is loaded before the first frame
LOG_EVERY_FRAMES = 1200         # video_processor logs "Extraction: Processed N frames..." every this many frames
ANALYZE_PER_VIDEO_S = 4.0       # Stage 2 (events) per video, plus loading
ANALYZE_PER_MOVE_S = 5.0        # Stage 3: rendering one chosen move
ANALYZE_FIXED_S = 8.0           # selection + joining the master reel
DEFAULT_MOVES = 13              # chosen moves before the log has said how many

_PROCESSED = re.compile(r"Extraction: Processed (\d+) frames")
_EXTRACT_DONE = re.compile(r"Video .*: extraction done in ([\d.]+) seconds")
_MOVES = re.compile(r"(\d+) events chosen")


def count_frames(paths):
    """Frames per video file (cv2), FALLBACK_FRAMES for a file that cannot be read."""
    try:
        import cv2
    except Exception:   # noqa: BLE001
        return [FALLBACK_FRAMES] * len(paths)
    out = []
    for p in paths:
        n = 0
        try:
            cap = cv2.VideoCapture(p)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        except Exception:   # noqa: BLE001
            n = 0
        out.append(n if n > 0 else FALLBACK_FRAMES)
    return out


def fmt_eta(seconds):
    """'about 11 min left' - rounded so the text does not jitter every update."""
    s = max(0.0, float(seconds))
    if s < 15:
        return "almost done"
    if s < 90:
        return f"about {int(round(s / 10.0) * 10)} s left"
    if s < 3600:
        return f"about {int(round(s / 60.0))} min left"
    h, m = int(s // 3600), int(round((s % 3600) / 60.0))
    if m == 60:
        h, m = h + 1, 0
    return f"about {h} h {m:02d} min left"


class RunProgress:
    def __init__(self):
        self.running = False
        self.fraction = 0.0

    # ---- set-up and stage changes (called by the app)
    def start(self, frames, skip, now):
        """frames: frame count of every video, in processing order; skip: True where the video's data CSV exists already (nothing to extract)."""
        self.frames = list(frames)
        self.skip = list(skip)
        self.n = len(self.frames)
        self.t0 = now
        self.running = True
        self.fraction = 0.0
        self.stage = "extract"
        self.pick_wait, self._pick_t0 = 0.0, None
        self.idx = -1                                  # the video the worker is on (counts its "Processing Video:" lines)
        self.done = 0                                  # videos whose extraction has finished (skipped ones too)
        self.to_do = float(sum(f for f, s in zip(self.frames, self.skip) if not s))
        self.fin_frames = self.fin_time = 0.0          # finished videos that really were extracted: frames and seconds they took
        self.cur_t0 = self.last_n = self.last_t = None
        self.analyze_t0 = None
        self.moves = None

    def set_stage(self, stage, now):
        """'extract' | 'pick' | 'analyze'"""
        if not self.running:
            return
        if self.stage == "pick" and self._pick_t0 is not None:
            self.pick_wait += now - self._pick_t0
            self._pick_t0 = None
        self.stage = stage
        if stage == "pick":
            self._pick_t0 = now
        elif stage == "analyze":
            self.analyze_t0 = now
            self.idx = -1

    def finish(self, ok):
        self.running = False
        if ok:
            self.fraction = 1.0

    # ---- the worker's log lines
    def feed(self, line, now):
        if not self.running:
            return
        if self.stage == "extract":
            if "Processing Video:" in line:
                self.idx += 1
                self.cur_t0, self.last_n, self.last_t = now, None, None
                return
            m = _PROCESSED.search(line)
            if m:
                self.last_n, self.last_t = int(m.group(1)), now
                return
            m = _EXTRACT_DONE.search(line)
            if m and 0 <= self.idx < self.n:
                self.done += 1
                if not self.skip[self.idx]:
                    self.fin_frames += self.frames[self.idx]
                    self.fin_time += float(m.group(1))
                self.cur_t0 = self.last_n = self.last_t = None
        elif self.stage == "analyze":
            m = _MOVES.search(line)
            if m and "Selection" in line:
                self.moves = int(m.group(1))

    # ---- estimates
    def _next_line(self):
        """(frames, time) of the last log line of the current video (its start before the first one), and whether another line is still due."""
        if self.last_n:
            base_n, base_t = self.last_n, self.last_t
        else:
            base_n, base_t = 0, self.cur_t0 + MODEL_LOAD_S
        return base_n, base_t, self.frames[self.idx] - base_n > LOG_EVERY_FRAMES

    def _rate(self, now):
        """Frames per second: measured so far (finished videos + the part of the current one the log has reported), and never faster than the
        silence since the last log line allows - when the next line is overdue the machine is slower than the number says (a slow PC)."""
        frames, secs = self.fin_frames, self.fin_time
        if self.last_n and self.cur_t0 is not None and self.last_t is not None:
            frames += self.last_n
            secs += self.last_t - self.cur_t0
        rate = frames / secs if frames >= LOG_EVERY_FRAMES and secs > 5.0 else DEFAULT_FPS
        if self.cur_t0 is not None and 0 <= self.idx < self.n and not self.skip[self.idx]:
            _, base_t, due = self._next_line()
            quiet = now - base_t
            if due and quiet > LOG_EVERY_FRAMES / rate:
                rate = max(1.0, LOG_EVERY_FRAMES / quiet)
        return rate

    def _current_frames(self, now, rate):
        """Frames of the current video done by now: the last logged count plus what the speed says happened since."""
        if self.cur_t0 is None or not (0 <= self.idx < self.n) or self.skip[self.idx]:
            return 0.0
        base_n, base_t, due = self._next_line()
        est = base_n + rate * max(0.0, now - base_t)
        if due:
            est = min(est, base_n + LOG_EVERY_FRAMES - 1.0)     # the next line has not come yet, so fewer than that many frames can have passed
        return min(est, max(0.0, self.frames[self.idx] - 1.0))

    def analyze_estimate(self):
        moves = self.moves if self.moves is not None else DEFAULT_MOVES
        return ANALYZE_PER_VIDEO_S * self.n + ANALYZE_PER_MOVE_S * moves + ANALYZE_FIXED_S

    def snapshot(self, now):
        """{'fraction': 0..1 for the bar, 'text': one line for the window}"""
        if not self.running:
            return {"fraction": self.fraction, "text": ""}
        waiting = self._pick_t0 is not None
        active = now - self.t0 - self.pick_wait - ((now - self._pick_t0) if waiting else 0.0)
        if self.stage == "extract":
            rate = self._rate(now)
            left =max(0.0, self.to_do - self.fin_frames - self._current_frames(now, rate))
            eta = left / rate + self.analyze_estimate()
            text = f"Extracting: {self.done} of {self.n} video{'' if self.n == 1 else 's'} done  -  {fmt_eta(eta)}"
        elif self.stage == "pick":
            eta = self.analyze_estimate()
            text = "Extraction done  -  waiting for your shirt colour choice"
        else:
            eta = max(5.0, self.analyze_estimate() - (now - self.analyze_t0))
            text = f"Building the highlights  -  {fmt_eta(eta)}"
        if not waiting:
            frac = active / (active + eta) if active + eta > 0 else 0.0
            self.fraction = max(self.fraction, min(0.99, frac))   # never backwards, never "100%" before it is really over
        return {"fraction": self.fraction, "text": text}
