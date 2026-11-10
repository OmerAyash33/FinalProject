"""Classic (non-AI) grass detector used by Stage 2 to skip YOLO entirely on frames pointing off the pitch."""
import numpy as np
import cv2


class PitchDetector:
    """
    Finds the playable grass area via HSV color thresholding + morphology, and
    reports what fraction of the frame it covers. When that fraction is too
    low (camera looking at sky/crowd/bench), Stage 2 skips the expensive YOLO
    pass entirely for that frame.
    """

    def __init__(self):
        # HSV bounds for "grass green", tolerant enough to survive shadows and lighting shifts.
        self.lower_green = np.array([25, 20, 20])
        self.upper_green = np.array([90, 255, 255])

    def detect(self, frame):
        """Return the pitch mask, grass coverage ratio, and whether that ratio clears the usable-frame threshold."""
        h, w = frame.shape[:2]

        # Morphology below is O(pixels), so working on a small copy keeps this cheap
        # even though it runs on every single frame regardless of YOLO skip decisions.
        small_w, small_h = 320, int(320 * (h / w))
        small_frame = cv2.resize(frame, (small_w, small_h))

        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)  # hue isolates "green" independent of shadow/brightness
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))  # fill gaps from lines/shadows
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((10, 10), np.uint8))  # drop isolated noise pixels

        pitch_mask_small = self._build_pitch_mask(mask, small_h, small_w)
        pitch_mask = cv2.resize(pitch_mask_small, (w, h))

        grass_ratio = np.sum(pitch_mask > 0) / (h * w)

        return {
            "pitch_mask": pitch_mask,
            "penalty_mask": np.zeros((h, w), dtype=np.uint8),
            "grass_ratio": grass_ratio,
            "valid": grass_ratio >= 0.10,  # below 10% grass: not worth running YOLO on this frame
        }

    def _build_pitch_mask(self, mask, h, w):
        """Collapse the (possibly patchy) grass mask into a single solid polygon covering the actual pitch."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pitch_mask = np.zeros_like(mask)
        if not contours:
            return pitch_mask

        # Discard grass-colored blobs sitting in the top of the frame (e.g. sky/advertising banners):
        # a real pitch blob's centroid (m01/m00) should sit in the lower part of the image.
        bottom_contours = []
        for c in contours:
            m = cv2.moments(c)
            if m["m00"] > 0 and m["m01"] / m["m00"] > h * 0.4:
                bottom_contours.append(c)
        if not bottom_contours:
            return pitch_mask

        largest = max(bottom_contours, key=cv2.contourArea)
        hull_pts = cv2.convexHull(largest).reshape(-1, 2)
        if np.max(hull_pts[:, 1]) > h * 0.7:  # blob already reaches near the bottom edge
            hull_pts = np.vstack([hull_pts, [[0, h], [w, h]]])  # anchor the polygon to the true bottom corners

        row_sums = np.sum(mask, axis=1)
        sig_rows = np.where(row_sums > w * 0.1)[0]  # rows with enough grass pixels to be "real" pitch, not noise
        top_limit = sig_rows[0] if len(sig_rows) > 0 else int(h * 0.3)
        hull_pts[:, 1] = np.maximum(hull_pts[:, 1], top_limit)  # clip the polygon so it never bleeds into the sky

        hull = cv2.convexHull(hull_pts.reshape(-1, 1, 2))
        cv2.drawContours(pitch_mask, [hull], -1, 255, -1)
        return pitch_mask

    def is_point_in_pitch(self, point, result):
        """Look up whether `point` (x, y) falls inside the pitch mask from a prior `detect()` call."""
        mask = result.get("pitch_mask")
        if mask is None:
            return False
        x, y = int(point[0]), int(point[1])
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            return mask[y, x] > 0
        return False
