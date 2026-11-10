"""Background-thread frame producer used by Stage 2 so disk I/O never stalls the YOLO consumer."""
import sys
import time
import threading
import queue
import cv2


class FastVideoReader:
    """
    Reads a video on a background daemon thread, downscales each frame to
    `width`, and pushes it into a bounded queue for the main thread to consume.
    This keeps the AI (GPU/CPU) fed continuously instead of stalling on disk
    reads between frames.
    """

    def __init__(self, path, queue_size, width):
        self.path = path
        self.width = width
        self.Q = queue.Queue(maxsize=queue_size)  # bounded: caps RAM usage if the consumer falls behind
        self.running = True
        self.cap = cv2.VideoCapture(self.path)

        if not self.cap.isOpened():
            self.running = False
            print(f"\n[CRITICAL ERROR] OpenCV cannot open video file: {self.path}")
            sys.exit(1)

        self.thread = threading.Thread(target=self.update, daemon=True)

    def start(self):
        """Launch the background read loop and return self, so `.start()` can be chained onto the constructor."""
        if self.running:
            self.thread.start()
        return self

    def update(self):
        """Background-thread loop: read -> downscale -> enqueue, until the video ends."""
        while self.running:
            if not self.Q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.running = False
                    break

                # Nearest-neighbor: the fastest resize available, since this runs on
                # every single frame and speed here directly caps overall throughput.
                h, w = frame.shape[:2]
                ratio = self.width / float(w)
                new_h = int(h * ratio)
                frame = cv2.resize(frame, (self.width, new_h), interpolation=cv2.INTER_NEAREST)
                self.Q.put(frame)
            else:
                time.sleep(0.03)  # queue is full: back off briefly instead of busy-spinning

    def more(self):
        """True while the video is still being read, or while queued frames remain unconsumed."""
        return self.running or not self.Q.empty()

    def stop(self):
        """Signal the background thread to stop and release the underlying video handle."""
        self.running = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
