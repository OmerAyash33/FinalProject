"""STAGE 1 of the pipeline: optical-flow video stabilization (see main.py for the full stage list)."""
import numpy as np
import cv2
import pandas as pd

from logging_utils import log


class VideoStabilizer:
    """
    Smooths out camera shake using classic point-feature optical flow (NOT a
    Kalman filter — see the exponential-smoothing note on SMOOTHING_RADIUS
    below). Writes a stabilized copy of the video, and separately logs the
    per-frame correction it applied (dx, dy, da) to `transforms_csv` so later
    stages can mathematically undo it when they need to draw on the original,
    un-stabilized footage. Expected cost: ~10-15 ms/frame.
    """

    def __init__(self, input_path, output_path, transforms_csv):
        self.input_path = input_path
        self.output_path = output_path
        self.transforms_csv = transforms_csv
        # Draws the live optical-flow tracking points on the preview window when True.
        self.DEBUG_MODE = True

    def process(self):
        """Read `input_path` frame by frame, stabilize each one, and write the result + transform log."""
        log("Starting Optical Flow Stabilization...")
        cap = cv2.VideoCapture(self.input_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        ret, prev_frame = cap.read()
        if not ret:
            return False

        # Optical flow runs on a heavily downscaled copy for speed; SCALE_X/Y map
        # the resulting motion back up to the native HD frame for the actual warp.
        PROC_W, PROC_H = 480, 270
        SCALE_X = width / PROC_W
        SCALE_Y = height / PROC_H

        prev_small = cv2.resize(prev_frame, (PROC_W, PROC_H))
        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)

        out.write(prev_frame)  # first frame has no prior frame to compare against -> written as-is

        # Cumulative raw camera trajectory (where the camera actually is) vs. the
        # exponentially-smoothed trajectory (where it "should" be if it moved
        # smoothly). The gap between the two is exactly the shake we correct for.
        x_traj, y_traj, a_traj = 0.0, 0.0, 0.0
        sm_x_traj, sm_y_traj, sm_a_traj = 0.0, 0.0, 0.0

        prev_pts = None
        frame_count = 0
        transforms_data = [{"frame": frame_count, "dx": 0.0, "dy": 0.0, "da": 0.0}]  # frame 0: zero motion

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 1200 == 0:
                log(f"Stabilizer: Processed {frame_count} frames...")

            curr_small = cv2.resize(curr_frame, (PROC_W, PROC_H))
            curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)

            # Re-seed tracking points once too many have been lost (occlusion, frame cuts, etc.).
            if prev_pts is None or len(prev_pts) < 20:
                prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.02, minDistance=25)

            stabilized_frame = curr_frame.copy()  # default: no correction applied this frame
            dx_out, dy_out, da_out = 0.0, 0.0, 0.0
            curr_pts = None

            if prev_pts is not None and len(prev_pts) > 10:
                # Lucas-Kanade optical flow: where did each tracked point move to?
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
                idx = np.where(status == 1)[0]  # keep only points flow actually found in the new frame
                prev_pts = prev_pts[idx]
                curr_pts = curr_pts[idx]

                if len(prev_pts) >= 4:  # cv2.estimateAffinePartial2D needs >= 4 point pairs
                    matrix, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
                    if matrix is not None:
                        dx, dy = matrix[0, 2], matrix[1, 2]
                        da = np.arctan2(matrix[1, 0], matrix[0, 0])  # rotation implied by the affine matrix

                        x_traj += dx
                        y_traj += dy
                        a_traj += da

                        # Exponential moving average, NOT a Kalman filter: each new
                        # smoothed point is 95% the old smoothed trajectory + 5% the
                        # new raw one. Higher SMOOTHING_RADIUS = steadier but slower
                        # to follow intentional camera motion (e.g. genuinely running).
                        SMOOTHING_RADIUS = 0.95
                        sm_x_traj = sm_x_traj * SMOOTHING_RADIUS + x_traj * (1 - SMOOTHING_RADIUS)
                        sm_y_traj = sm_y_traj * SMOOTHING_RADIUS + y_traj * (1 - SMOOTHING_RADIUS)
                        sm_a_traj = sm_a_traj * SMOOTHING_RADIUS + a_traj * (1 - SMOOTHING_RADIUS)

                        # The correction is the gap between where the camera actually
                        # is (x_traj) and where the smoothed path says it should be.
                        diff_x = sm_x_traj - x_traj
                        diff_y = sm_y_traj - y_traj
                        diff_a = sm_a_traj - a_traj

                        hd_dx = diff_x * SCALE_X  # scale the correction back up to native HD pixels
                        hd_dy = diff_y * SCALE_Y
                        dx_out, dy_out, da_out = hd_dx, hd_dy, diff_a

                        correction_matrix = np.zeros((2, 3), np.float32)
                        correction_matrix[0, 0] = np.cos(diff_a)
                        correction_matrix[0, 1] = -np.sin(diff_a)
                        correction_matrix[1, 0] = np.sin(diff_a)
                        correction_matrix[1, 1] = np.cos(diff_a)
                        correction_matrix[0, 2] = hd_dx
                        correction_matrix[1, 2] = hd_dy

                        # Warp only, no crop/stretch: keeps 1:1 native resolution and
                        # geometry, at the cost of black borders appearing where the
                        # correction shifted the frame content off-canvas.
                        stabilized_frame = cv2.warpAffine(curr_frame, correction_matrix, (width, height))

            transforms_data.append({"frame": frame_count, "dx": dx_out, "dy": dy_out, "da": da_out})

            prev_pts = curr_pts.reshape(-1, 1, 2) if curr_pts is not None and len(curr_pts) > 0 else None
            prev_gray = curr_gray.copy()

            out.write(stabilized_frame)

            if frame_count % 3 == 0:  # sub-sampled preview: full-rate rendering isn't needed for a debug view
                view_w, view_h = 640, 360
                view_stab = cv2.resize(stabilized_frame, (view_w, view_h))

                if self.DEBUG_MODE and prev_pts is not None:
                    for pt in prev_pts:
                        px = int(pt[0][0] * (view_w / PROC_W))
                        py = int(pt[0][1] * (view_h / PROC_H))
                        cv2.circle(view_stab, (px, py), 3, (0, 255, 0), -1)

                cv2.putText(view_stab, "STABILIZING...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Processing Pipeline", view_stab)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        out.release()

        # Saved so later stages can invert this exact motion when re-drawing on the raw footage.
        pd.DataFrame(transforms_data).to_csv(self.transforms_csv, index=False)
        log("Stabilization Complete.")
        return True
