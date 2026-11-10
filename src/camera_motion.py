"""
Per-frame global camera motion, used as a SIDE SIGNAL for ball speed - never to warp the video.

The camera is worn on the body, so every head/torso turn slides the whole picture sideways. A ball
lying still on the grass then looks as fast in the image as a ball that was kicked. Stage 2 needs
the camera's own motion so it can subtract it from the ball's image velocity (ball speed relative
to the scene instead of relative to the lens).

Method: phase correlation (cv2.phaseCorrelate) between consecutive frames on a small grayscale
copy of the UPPER part of the frame only. The lower part is excluded on purpose: that is where the
wearer's own arm, shadow and the ball at their feet are, and none of them move with the scene.
Phase correlation returns one dominant translation, which is exactly the "everything slid by
(dx, dy)" quantity wanted here; it costs about a millisecond per frame.

Sign convention (checked against a synthetic shift in the unit test of this module's author):
(dx, dy) is how far the static scene MOVED in the image between the previous and the current
frame, in pixels of the frame handed to update(). A world-stationary object therefore moves by
exactly (dx, dy), so   ball image velocity - (dx, dy)   is the ball's velocity relative to the scene.
"""
import cv2
import numpy as np

CAMERA_MOTION_WIDTH = 240          # width of the grayscale copy phase correlation runs on
CAMERA_MOTION_ROI_BOTTOM = 0.60    # only the top 60% of the frame is used (see module docstring)


class CameraMotionEstimator:
    """Feed frames in order; update() returns (dx, dy, response) for each one."""

    def __init__(self):
        self._prev = None
        self._window = None
        self._scale = None

    def update(self, frame_bgr):
        """(dx, dy) in pixels of `frame_bgr`, and a 0..1 confidence (phase-correlation peak height)."""
        h, w = frame_bgr.shape[:2]
        roi = frame_bgr[: int(h * CAMERA_MOTION_ROI_BOTTOM)]
        small_h = max(8, int(round(roi.shape[0] * CAMERA_MOTION_WIDTH / w)))
        small = cv2.resize(roi, (CAMERA_MOTION_WIDTH, small_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._window is None or self._window.shape != gray.shape:
            self._window = cv2.createHanningWindow((gray.shape[1], gray.shape[0]), cv2.CV_32F)
            self._scale = w / float(CAMERA_MOTION_WIDTH)
            self._prev = None
        if self._prev is None:
            self._prev = gray
            return 0.0, 0.0, 0.0
        (sx, sy), response = cv2.phaseCorrelate(self._prev, gray, self._window)
        self._prev = gray
        return float(sx) * self._scale, float(sy) * self._scale, float(response)
