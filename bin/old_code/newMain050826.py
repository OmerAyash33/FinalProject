import os  # Imports OS module to handle file paths, directories, and system environment variables safely.
import sys  # Imports system module to allow clean program exits during critical failures.
import cv2  # Imports OpenCV library for all video reading, writing, image processing, and drawing operations.
import time  # Imports time module to track execution duration and manage thread sleeping.
import threading  # Imports threading module to run background tasks (like video reading) asynchronously.
import queue  # Imports queue module to create safe memory buffers between different processing threads.
import glob  # Imports glob module to easily search and find all video files within a target directory.
import warnings  # Imports warnings module to suppress unnecessary library alerts that clutter the console.
import math  # Imports math module for trigonometric calculations like angles and distances.
import numpy as np  # Imports NumPy module for highly optimized, large-scale mathematical array operations.
import pandas as pd  # Imports Pandas module to structure, manipulate, and save the extracted data as CSV files.
from collections import deque  # Imports deque to create an efficient rolling window for calculating average FPS.
from sklearn.cluster import KMeans  # Imports KMeans machine learning algorithm to group players into teams by color.
from ultralytics import YOLO  # Imports the YOLO neural network architecture for state-of-the-art object detection.
import torch  # Imports PyTorch to interface with the hardware and check if GPU acceleration is available.
from datetime import datetime  # Imports datetime module to attach real-world timestamps to console logging.

warnings.filterwarnings("ignore",
                        category=FutureWarning)  # Specifically suppresses FutureWarnings from Pandas to keep logs clean.
os.environ[
    'KMP_DUPLICATE_LIB_OK'] = 'True'  # Sets environment variable to prevent Intel OpenMP library from crashing during multithreading.

TARGET_WIDTH = 480  # Defines the fixed internal resolution width for AI tracking to maximize processing speed.
CSV_WRITE_CHUNK = 1800  # Defines how many frames to accumulate in RAM before writing to the hard drive, minimizing slow I/O operations.
PERSON_CONF_THRESHOLD = 0.3  # Minimum confidence required to keep a "person" detection.
BALL_CONF_THRESHOLD = 0.15  # Minimum confidence required to keep a "sports ball" detection - lower than persons, since the ball is small/fast and often scores lower even when clearly visible. Raised from 0.1 after visually confirming detections below ~0.15 are frequently not the ball at all (e.g. a player's foot, empty grass).
GOAL_CONF_THRESHOLD = 0.15  # Minimum confidence required to keep a "goal" detection - the goal mouth is often partially occluded/angled, so it rarely scores as high as a clear frontal view.

# --- Advanced tactical event thresholds ---
SHOT_VELOCITY_WINDOW = 5  # Frames after zone-exit checked for the high-exit-velocity shot criterion.
SHOT_TRAJECTORY_WINDOW = 100  # Frames after zone-exit checked for the bottom->upper trajectory shot criterion - wide enough to survive a multi-second detection gap mid-flight.
SHOT_VELOCITY_MULT = 1.6  # Ball speed must exceed (ball width * this) to count as a high-velocity exit.
SHOT_SHRINK_RATE = 0.08  # Minimum frame-to-frame ball width shrink ratio indicating depth (moving away from camera).
SHOT_LOWER_Y_RATIO = 0.7  # Ball must start below this fraction of screen height (near the wearer's feet).
SHOT_UPPER_Y_RATIO = 0.45  # Ball must rise above this fraction of screen height to confirm a forward/upward drive.
TURNOVER_TIME_LIMIT_SEC = 1.0  # A turnover can only be called within this many seconds of leaving the control zone.
TURNOVER_PROXIMITY_MULT = 3  # Opponent must be within (ball width * this) to be considered a tackle/turnover.
RECOVERY_MIN_HOLD_SEC = 0.2  # Minimum possession time required to count a reclaimed ball as a BALL_RECOVERY.
AWARENESS_WINDOW_SEC = 3.0  # How far back to look at camera pan history before a touch, for the awareness score.
SCAN_VELOCITY_THRESHOLD = 1.5  # Raw per-frame pan (px, in the 480x270 optical-flow space) that counts as a "fast" look.
MAX_BALL_GAP_FRAMES = 15  # Longest gap (~0.5s @30fps) between two real ball detections that we'll bridge with linear interpolation for general use (possession/pass/turnover).
SHOT_BALL_GAP_FRAMES = 60  # Longest gap (~2s @30fps) bridged specifically for shot classification - a fast shot can vanish from detection for over a second mid-flight.
SELF_ZONE_BOTTOM_Y_REF = 0.95  # Y-ratio treated as "true bottom" (right at the wearer's feet) - a ball here needs only the baseline width to count.
SELF_ZONE_BASE_WIDTH_RATIO = 0.02  # Minimum ball width (as a fraction of screen width) required right at the true bottom.
SELF_ZONE_Y_SLOPE = 0.15  # Extra required width-ratio per unit of y-deficit from the true bottom - a ball higher up in frame must look proportionally bigger to still count as close, instead of a flat width cutoff that can't tell "big and close" apart from "medium and far but coincidentally low in a wide shot".
SELF_ZONE_MIN_Y_RATIO = 0.40  # Loose sanity floor - below this the required width would already be implausibly large anyway, this just short-circuits obviously-too-high positions (sky/crossbar).
ZONE_CONFIRM_FRAMES = 1  # Consecutive in-zone frames required before confirming a new possession. Kept at 1 (effectively off) because ball detection is often too sparse near a real touch to ever see several consecutive confirmations - the size/height formula above already does the discrimination work.

UI_COLORS = {  # Defines a dictionary holding BGR color tuples for rendering graphics using OpenCV.
    "blue": (0, 191, 255),  # Defines standard color for Team 0.
    "red": (255, 0, 100),  # Defines standard color for Team 1.
    "white": (255, 255, 255),  # Defines standard color for the ball and general text elements.
    "dark_green": (0, 180, 45),  # Defines color indicating a successful AI Coach event.
    "dark_red": (180, 0, 60),  # Defines color indicating a failed or incomplete AI Coach event.
    "orange": (255, 160, 0),  # Defines color indicating an active dribbling phase.
    "cyan": (0, 255, 255),  # Defines color indicating strong ball control.
    "light_gray": (170, 170, 170),  # Defines color indicating a momentary ball touch.
    "yellow": (255, 255, 90),  # Defines highly visible color for special highlights or goals.
    "black": (0, 0, 0),  # Defines pure black used primarily for text shadows and backgrounds.
    "radar_bg": (0, 40, 0),  # Defines dark green background color specifically for the minimap radar.
    "radar_border": (180, 180, 180),  # Defines light gray color for the minimap radar border.
    "stats_card_bg": (20, 20, 20),  # Defines dark charcoal color for the background of the final stats card.
    "magenta": (255, 0, 255),  # Defines color indicating a turnover (ball lost to an opponent).
    "teal": (0, 200, 170),  # Defines color indicating a ball recovery (regained possession).
    "gold": (0, 215, 255)  # Defines color indicating a confirmed goal.
}

UI_SETTINGS = {  # Defines a dictionary holding numeric settings for typography and visual element scaling.
    "box_thickness": 2,  # Sets the default stroke thickness for bounding boxes drawn around players and the ball.
    "font_style": cv2.FONT_HERSHEY_DUPLEX,  # Sets the globally used OpenCV font family for all text overlays.
    "font_scale_large": 1.2,  # Sets the scaling multiplier for primary event titles (e.g., 'PASS SUCCESS').
    "font_thickness": 2,  # Sets the default stroke thickness for text elements.
    "font_scale_small": 0.9,  # Sets the scaling multiplier for secondary text like AI Coach reasoning.
}


def chromaticity(color_df):
    """
    Converts raw BGR samples into lighting-invariant chromaticity ratios (each channel
    divided by the sum of all three). Two jersey photos of the same color taken in different
    light/shadow still land near the same point in chromaticity space, whereas raw BGR
    magnitudes can drift enough to cross a KMeans cluster boundary.
    """
    total = color_df.sum(axis=1).replace(0, 1)  # Avoids divide-by-zero for pure-black samples.
    return color_df.div(total, axis=0)


def export_and_load_openvino(pt_path="yolo11n.pt", ov_dir="yolo11n_openvino_model",
                             task="detect"):  # Defines a utility function to ensure any given AI model is optimized for CPU inference.
    print(f"[INFO] Checking AI engine ({pt_path}) for CPU optimization...")  # Logs the start of the optimization check.

    if not os.path.exists(ov_dir):  # Checks if the compiled model directory does not exist on disk.
        print(
            f"[INFO] Compiling {pt_path} to OpenVINO. This happens only once...")  # Logs that a one-time compilation is starting.
        base_model = YOLO(pt_path)  # Loads the standard PyTorch model into RAM.
        base_model.export(format="openvino", imgsz=TARGET_WIDTH,
                          dynamic=False)  # Compiles the model to OpenVINO format, locking the input size for maximum speed.

    print(f"[INFO] Model {pt_path} loaded successfully.")  # Logs that the optimized model is ready for use.
    return YOLO(ov_dir,
                task=task)  # Initializes and returns the OpenVINO model object for the requested task type.


# =====================================================================
# STAGE 1: VideoStabilizer
# Analyzes frame-to-frame movement to counteract camera shake.
# Outputs a high-quality, stabilized version of the original video.
# =====================================================================
class VideoStabilizer:  # Defines the class responsible for pre-processing and smoothing the video footage.
    def __init__(self, input_path, output_path, pan_csv_path):  # Initializes the stabilizer with source and destination file paths.
        self.input_path = input_path  # Stores the absolute path of the raw, shaky source video.
        self.output_path = output_path  # Stores the absolute path where the stabilized video will be saved.
        self.pan_csv_path = pan_csv_path  # Stores the path where the raw per-frame camera pan (dx) log is saved.

    def process(self):  # Defines the main stabilization execution method.
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] STAGE 1: Stabilizing Video...")  # Logs the beginning of Stage 1.

        if os.path.exists(self.output_path) and os.path.exists(
                self.pan_csv_path):  # Checks if both the stabilized video and pan log already exist from a previous run.
            print(f"[INFO] Stabilized video already exists. Skipping Stage 1.")  # Logs skipping to save time.
            return True  # Returns True to signal successful completion (via cache).

        if os.path.exists(self.pan_csv_path):  # Clears any partial pan log left from an incomplete previous run.
            os.remove(self.pan_csv_path)  # Removes it to ensure a clean slate.
        pan_data = []  # Initializes an in-memory buffer for per-frame pan rows before flushing to disk.

        cap = cv2.VideoCapture(self.input_path)  # Opens the raw video file using OpenCV.
        if not cap.isOpened():  # Validates that the video file was opened successfully.
            print(
                f"[ERROR] Could not open video for stabilization: {self.input_path}")  # Logs an error if file is missing or corrupted.
            return False  # Returns False to halt the pipeline.

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Extracts the native width of the video.
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Extracts the native height of the video.
        fps = cap.get(cv2.CAP_PROP_FPS)  # Extracts the native frames per second.
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Extracts the total number of frames to track progress.

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Defines the MP4 codec used for writing the new video.
        out = cv2.VideoWriter(self.output_path, fourcc, fps,
                              (width, height))  # Initializes the video writer object with native parameters.

        ret, prev_frame = cap.read()  # Reads the very first frame of the video to serve as the initial reference point.
        if not ret: return False  # Exits if the video is completely empty.

        PROC_W, PROC_H = 480, 270  # Defines low-resolution dimensions specifically for fast optical flow math.
        SCALE_X = width / PROC_W  # Calculates the scalar required to map horizontal math back to the original HD width.
        SCALE_Y = height / PROC_H  # Calculates the scalar required to map vertical math back to the original HD height.

        prev_small = cv2.resize(prev_frame, (PROC_W, PROC_H))  # Aggressively downscales the first frame for speed.
        prev_gray = cv2.cvtColor(prev_small,
                                 cv2.COLOR_BGR2GRAY)  # Converts the first downscaled frame to grayscale (color is irrelevant for flow).

        margin_x = int(
            width * 0.05)  # Calculates a 5% horizontal margin to crop out black borders caused by stabilization.
        margin_y = int(height * 0.05)  # Calculates a 5% vertical margin to crop out black borders.
        cropped_prev = prev_frame[margin_y:height - margin_y,
                       margin_x:width - margin_x]  # Physically crops the first raw frame.
        stabilized_prev = cv2.resize(cropped_prev,
                                     (width, height))  # Stretches the cropped first frame back to native HD resolution.
        out.write(stabilized_prev)  # Writes the modified first frame to the output file to maintain sync.

        x_traj, y_traj, a_traj = 0.0, 0.0, 0.0  # Initializes cumulative trajectory variables for X, Y, and Angle.
        sm_x_traj, sm_y_traj, sm_a_traj = 0.0, 0.0, 0.0  # Initializes exponentially smoothed trajectory variables.

        prev_pts = None  # Initializes the array that will hold tracking points.
        frame_count = 1  # Initializes the frame counter, starting at 1 since the first frame is done.

        frame_timestamps = deque()  # Initializes a queue to hold recent processing timestamps for the live FPS counter.
        last_fps_update_time = time.time()  # Records the initial time for FPS tracking.
        displayed_fps = 0  # Initializes the displayed FPS metric.

        while True:  # Starts the sequential frame processing loop.
            ret, curr_frame = cap.read()  # Reads the next consecutive frame from the video.
            if not ret: break  # Breaks the loop when the video finishes.

            frame_count += 1  # Increments the global frame counter.

            curr_small = cv2.resize(curr_frame, (PROC_W, PROC_H))  # Downscales the current frame for optical flow math.
            curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)  # Converts the current frame to grayscale.

            if prev_pts is None or len(
                    prev_pts) < 20:  # Checks if tracking points need to be initialized or replenished.
                prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                                   minDistance=30)  # Detects strong geometric corners in the image.

            if prev_pts is not None and len(
                    prev_pts) > 10:  # Checks if enough valid points exist to calculate movement.
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts,
                                                                 None)  # Calculates where the tracking points moved.

                idx = np.where(status == 1)[0]  # Filters out points that were lost or occluded.
                prev_pts = prev_pts[idx]  # Keeps only the valid historical points.
                curr_pts = curr_pts[idx]  # Keeps only the valid new points.

                if len(prev_pts) >= 4:  # Ensures at least 4 points exist to build a transformation matrix.
                    matrix, _ = cv2.estimateAffinePartial2D(prev_pts,
                                                            curr_pts)  # Calculates the mathematical pan, tilt, and rotation between frames.
                    if matrix is not None:  # Verifies matrix success.
                        dx, dy = matrix[0, 2], matrix[1, 2]  # Extracts horizontal and vertical shift.
                        da = np.arctan2(matrix[1, 0], matrix[0, 0])  # Extracts rotational shift using trigonometry.

                        pan_data.append({"frame": frame_count,
                                         "dx": float(dx)})  # Logs the raw per-frame horizontal pan for the awareness metric.

                        x_traj += dx  # Accumulates raw horizontal movement.
                        y_traj += dy  # Accumulates raw vertical movement.
                        a_traj += da  # Accumulates raw rotational movement.

                        SMOOTHING_RADIUS = 0.95  # Defines how aggressively to smooth the movement (closer to 1.0 = more stable).
                        sm_x_traj = sm_x_traj * SMOOTHING_RADIUS + x_traj * (
                                    1 - SMOOTHING_RADIUS)  # Applies exponential smoothing to X.
                        sm_y_traj = sm_y_traj * SMOOTHING_RADIUS + y_traj * (
                                    1 - SMOOTHING_RADIUS)  # Applies exponential smoothing to Y.
                        sm_a_traj = sm_a_traj * SMOOTHING_RADIUS + a_traj * (
                                    1 - SMOOTHING_RADIUS)  # Applies exponential smoothing to Angle.

                        diff_x = sm_x_traj - x_traj  # Calculates the stabilizing difference in X.
                        diff_y = sm_y_traj - y_traj  # Calculates the stabilizing difference in Y.
                        diff_a = sm_a_traj - a_traj  # Calculates the stabilizing difference in Angle.

                        correction_matrix = np.zeros((2, 3), np.float32)  # Initializes an empty transformation matrix.
                        correction_matrix[0, 0] = np.cos(diff_a)  # Sets rotation cosine.
                        correction_matrix[0, 1] = -np.sin(diff_a)  # Sets rotation sine.
                        correction_matrix[1, 0] = np.sin(diff_a)  # Sets rotation sine.
                        correction_matrix[1, 1] = np.cos(diff_a)  # Sets rotation cosine.
                        correction_matrix[
                            0, 2] = diff_x * SCALE_X  # Sets X translation, scaling it back to HD proportions!
                        correction_matrix[
                            1, 2] = diff_y * SCALE_Y  # Sets Y translation, scaling it back to HD proportions!

                        stabilized_frame = cv2.warpAffine(curr_frame, correction_matrix, (width,
                                                                                          height))  # Mathematically warps the HD pixels to counter camera shake.
                        cropped = stabilized_frame[margin_y:height - margin_y,
                                  margin_x:width - margin_x]  # Crops out the black borders caused by the warp.
                        stabilized_frame = cv2.resize(cropped, (width,
                                                                height))  # Stretches the cropped HD frame back to native HD size.
                    else:  # Fallback if matrix failed.
                        stabilized_frame = curr_frame.copy()  # Uses raw frame.
                else:  # Fallback if point tracking failed.
                    stabilized_frame = curr_frame.copy()  # Uses raw frame.
            else:  # Fallback if initial feature detection failed.
                stabilized_frame = curr_frame.copy()  # Uses raw frame.

            prev_pts = curr_pts.reshape(-1, 1, 2) if 'curr_pts' in locals() and len(
                curr_pts) > 0 else None  # Sets the current points as history for the next iteration.
            prev_gray = curr_gray.copy()  # Sets current grayscale image as history for the next iteration.

            out.write(stabilized_frame)  # Writes the final, high-resolution stabilized frame to the MP4 file.

            if frame_count % CSV_WRITE_CHUNK == 0 and len(
                    pan_data) > 0:  # Checks if it's time to dump the pan RAM buffer to the hard drive.
                pan_chunk_df = pd.DataFrame(data=pan_data)  # Converts the buffered pan rows to a DataFrame.
                write_header = not os.path.exists(self.pan_csv_path)  # Decides if column headers are needed.
                pan_chunk_df.to_csv(self.pan_csv_path, mode="a", index=False,
                                    header=write_header)  # Appends the pan data to disk.
                pan_data.clear()  # Flushes the RAM buffer completely.

            current_time = time.time()  # Grabs current system time for FPS tracking.
            frame_timestamps.append(current_time)  # Appends timestamp to the rolling window queue.

            while frame_timestamps and frame_timestamps[
                0] < current_time - 3.0:  # Removes timestamps older than 3 seconds.
                frame_timestamps.popleft()  # Drops old timestamps.

            if current_time - last_fps_update_time >= 0.5:  # Updates the displayed UI string strictly twice a second.
                displayed_fps = len(frame_timestamps) // 3  # Averages processed frames over the 3-second window.
                last_fps_update_time = current_time  # Resets update timer.

            if frame_count % 3 == 0:  # Triggers the UI refresh only every 3rd frame to save CPU processing power.
                view_w, view_h = 640, 360  # Defines a small UI resolution so the window doesn't take over the user's screen.
                view_orig = cv2.resize(curr_frame, (view_w, view_h))  # Resizes original purely for the UI.
                view_stab = cv2.resize(stabilized_frame, (view_w, view_h))  # Resizes stabilized purely for the UI.

                cv2.putText(view_orig, "ORIGINAL", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2)  # Labels left screen.
                cv2.putText(view_stab, f"STAGE 1: STABILIZING ({displayed_fps} FPS) - {frame_count}/{total_frames}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Labels right screen.

                combined_view = cv2.hconcat(
                    [view_orig, view_stab])  # Splices the two frames horizontally into one wide window.
                cv2.imshow("Pipeline LIVE: Stage 1", combined_view)  # Renders the live window to the OS.

                if cv2.waitKey(1) & 0xFF == ord('q'):  # Allows manual user termination by pressing 'q'.
                    print("\n[WARNING] Process aborted by user.")  # Logs abortion.
                    break  # Exits loop.

        if len(pan_data) > 0:  # Handles any remaining unwritten pan rows after the video finishes.
            pan_final_df = pd.DataFrame(data=pan_data)  # Converts the leftover buffer to a DataFrame.
            pan_final_df.to_csv(self.pan_csv_path, mode="a", index=False,
                                header=not os.path.exists(self.pan_csv_path))  # Appends the final chunk.

        cap.release()  # Closes raw video reader.
        out.release()  # Finalizes stabilized HD video writer.
        cv2.destroyAllWindows()  # Closes UI windows.
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] STAGE 1 Complete. Saved: {os.path.basename(self.output_path)}")  # Logs success.
        return True  # Returns True to signal pipeline manager to proceed.


# =====================================================================
# STAGE 2 DEPENDENCIES
# Includes the multithreaded fast reader and classic color-based pitch detection.
# =====================================================================
class FastVideoReader:  # Defines a class that uses multithreading to read and resize frames in the background.
    def __init__(self, path, queue_size,
                 width):  # Initializes the reader with file path, RAM buffer size, and target resize width.
        self.path = path  # Stores the video file path.
        self.width = width  # Stores the target width for aggressive downscaling.
        self.Q = queue.Queue(maxsize=queue_size)  # Initializes a thread-safe Queue to act as the RAM buffer.
        self.running = True  # Sets the control flag for the background thread loop.
        self.cap = cv2.VideoCapture(self.path)  # Initializes the OpenCV video capture object.

        if not self.cap.isOpened():  # Checks if OpenCV successfully accessed the file.
            self.running = False  # Immediately shuts down the thread flag.
            print(f"\n[CRITICAL ERROR] OpenCV cannot open video file: {self.path}")  # Logs error to console.
            sys.exit(1)  # Aborts the entire Python process to prevent cascading errors.

        self.thread = threading.Thread(target=self.update,
                                       daemon=True)  # Instantiates the background thread pointing to the 'update' method.

    def start(self):  # Defines the method to launch the background thread.
        if self.running:  # Verifies initialization was successful.
            self.thread.start()  # Starts background execution.
        return self  # Returns the instance to allow method chaining upon creation.

    def update(self):  # Defines the core loop running silently on the background thread.
        while self.running:  # Loops as long as the video has not ended.
            if not self.Q.full():  # Checks if there is free space in the RAM buffer.
                ret, frame = self.cap.read()  # Reads the raw, high-res frame from the hard drive.
                if not ret:  # Checks if reading failed (end of file).
                    self.running = False  # Triggers loop exit.
                    break  # Breaks the loop immediately.

                # ARCHITECTURAL FIX: Downscale inside the thread! This makes the queue ultra-light and fast.
                h, w = frame.shape[:2]  # Extracts native dimensions.
                ratio = self.width / float(w)  # Calculates mathematical ratio to maintain aspect ratio during resize.
                new_h = int(h * ratio)  # Calculates the new proportional height.
                frame = cv2.resize(frame, (self.width, new_h),
                                   interpolation=cv2.INTER_LINEAR)  # Downscales the frame with linear interpolation - nearest-neighbor was faster but devastated detection confidence (e.g. the goal-segmentation model dropped from ~0.25 to ~0.03 on identical frames).

                self.Q.put(
                    frame)  # Pushes the tiny, lightweight frame into the high-speed RAM queue for the AI to consume.
            else:  # Executes if the AI is processing slower than the reader.
                time.sleep(
                    0.01)  # Pauses the reader thread for 10 milliseconds to prevent 100% CPU lockup while waiting.

    def more(self):  # Defines a method for the main AI thread to ask if frames are still available.
        return self.running or not self.Q.empty()  # Returns True if video is playing or leftover frames exist in RAM.

    def stop(self):  # Defines a safe shutdown method.
        self.running = False  # Kills the background loop.
        if hasattr(self, 'cap') and self.cap.isOpened():  # Verifies the capture object exists and is open.
            self.cap.release()  # Releases the file handle back to the OS.


class PitchDetectorClassic:  # Defines a lightweight class dedicated solely to finding the green grass using math, not heavy AI.
    def __init__(self):  # Constructor method.
        self.lower_green = np.array([25, 20, 20])  # Defines the minimum HSV color values that qualify as grass.
        self.upper_green = np.array([90, 255, 255])  # Defines the maximum HSV color values that qualify as grass.

    def detect(self, frame):  # Defines the main execution method to process a frame.
        h, w = frame.shape[:2]  # Extracts the downscaled frame dimensions.

        hsv = cv2.cvtColor(frame,
                           cv2.COLOR_BGR2HSV)  # Converts the BGR image to Hue-Saturation-Value space for robust color isolation.
        mask = cv2.inRange(hsv, self.lower_green,
                           self.upper_green)  # Creates a pure black and white mask isolating only the green pixels.

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10, 10),
                                                               np.uint8))  # Applies a mathematical filter to close tiny gaps inside the grass mask.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5),
                                                              np.uint8))  # Applies a filter to erase tiny isolated green noise outside the main pitch.

        pitch_mask = self._build_pitch_polygon(mask, h,
                                               w)  # Calls the helper function to build a unified geometric shape around the grass.
        grass_ratio = np.sum(pitch_mask > 0) / (
                    h * w)  # Calculates what percentage of the screen is covered by the unified pitch mask.

        return {  # Returns a structured dictionary with results.
            "pitch_mask": pitch_mask,  # The final binary image mask.
            "valid": grass_ratio >= 0.10,
            # Boolean flag evaluating to True only if at least 10% of the screen is grass (filters out sky/crowd shots).
        }

    def _build_pitch_polygon(self, mask, h, w):  # Defines a helper method to turn patchy grass into one solid shape.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)  # Mathematically traces the outlines of all green blobs.
        pitch_mask = np.zeros_like(mask)  # Creates a blank black canvas matching the image size.
        if not contours: return pitch_mask  # Returns immediately if zero grass was found.

        bottom_contours = [  # List comprehension to filter out green blobs near the top of the screen (like trees).
            c for c in contours
            if cv2.moments(c)["m00"] > 0 and cv2.moments(c)["m01"] / cv2.moments(c)["m00"] > h * 0.3
            # Keeps blobs whose center of mass is in the bottom 70% of the screen.
        ]
        if not bottom_contours: return pitch_mask  # Returns if all green was located in the sky area.

        largest = max(bottom_contours,
                      key=cv2.contourArea)  # Selects the single largest green blob surviving the filter.
        hull_pts = cv2.convexHull(largest).reshape(-1, 2)  # Wraps the largest blob in a "rubber band" convex polygon.

        if np.max(hull_pts[:, 1]) > h * 0.7:  # Checks if the bottom of this polygon reaches deep into the screen.
            hull_pts = np.vstack([hull_pts, [[0, h], [w,
                                                      h]]])  # Forces the polygon to artificially anchor to the absolute bottom left and right corners.

        hull = cv2.convexHull(
            hull_pts.reshape(-1, 1, 2))  # Recalculates the polygon boundary with the new artificial anchors.
        cv2.drawContours(pitch_mask, [hull], -1, 255,
                         -1)  # Fills the interior of the polygon with solid white on the blank canvas.
        return pitch_mask  # Returns the finalized shape.

    def is_point_in_pitch(self, point, result):  # Defines a fast spatial query method.
        mask = result.get("pitch_mask")  # Retrieves the active mask.
        if mask is None: return False  # Defensive fallback.
        x, y = int(point[0]), int(point[1])  # Casts query coordinates to integers for array indexing.
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[
            1]:  # Verifies the point is inside the image bounds to prevent crashes.
            return mask[y, x] > 0  # Returns True if the pixel at that coordinate is white (inside pitch).
        return False  # Fallback for out of bounds.


# =====================================================================
# STAGE 2: DataExtractor
# Reads the stabilized video, tracks players/ball using YOLO,
# isolates jersey colors, and logs everything to a structured CSV.
# =====================================================================
class DataExtractor:  # Defines the class responsible for turning pixels into hard data.
    def __init__(self, video_path, csv_filename):  # Initializes with the stabilized video path and the output CSV path.
        self.video_path = video_path  # Stores the input path.
        self.csv_filename = csv_filename  # Stores the output path.
        self.pitch_detector = PitchDetectorClassic()  # Instantiates the fast color-based pitch detector.
        self.model = export_and_load_openvino("yolo11n.pt", "yolo11n_openvino_model",
                                              task="detect")  # Loads the CPU-optimized YOLO tracking model, used exclusively for persistent person tracking.
        self.ball_model = export_and_load_openvino("ball_detection.pt", "ball_detection_openvino_model",
                                                    task="detect")  # Purpose-trained detector, "ball" is class 0 - finds the ball on nearly every frame (vs. sparse hits from generic yolo11n). Its own instance - never shares state with the person tracker.
        self.goal_model = export_and_load_openvino("goal_detection.pt", "goal_detection_openvino_model",
                                                    task="segment")  # goal_detection.pt's segmentation "goal" class was measured (real footage) to detect the goal far more often and more confidently than ball_detection.pt's own "goal" class (17% vs 1% of frames on one test clip) - used instead, as its own instance.

    def extract(self):  # Defines the main extraction loop.
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] STAGE 2: Extracting Data to CSV...")  # Logs the start of extraction.

        if os.path.exists(self.csv_filename):  # Checks if a raw data file from a previous incomplete run exists.
            os.remove(self.csv_filename)  # Deletes it to ensure a clean slate.

        reader = FastVideoReader(self.video_path, queue_size=128,
                                 width=TARGET_WIDTH).start()  # Spins up the background thread to start pumping frames into RAM.
        time.sleep(1.0)  # Pauses the main thread for 1 second to let the RAM buffer fill up with frames.

        cap = cv2.VideoCapture(self.video_path)  # Briefly opens the video on the main thread just to extract metadata.
        fps = cap.get(cv2.CAP_PROP_FPS)  # Extracts the native framerate.
        cap.release()  # Closes the metadata reader.

        match_data = []  # Initializes an empty list to act as a memory buffer for CSV rows before writing to disk.
        frame_counter = 0  # Initializes an integer to track current frame position.

        frame_times = deque()  # Initializes a queue to hold recent frame processing timestamps for calculating live UI speed.
        last_fps_update = time.time()  # Records the initial time marker for UI updates.
        display_fps = 0  # Initializes the UI speed metric.

        while reader.more():  # Starts the main data processing loop, running as long as the RAM queue has frames.
            try:  # Begins a try block to handle queue timeouts.
                frame = reader.Q.get(
                    timeout=3.0)  # Attempts to pop a frame from RAM, timing out if the reader thread crashes.
            except queue.Empty:  # Catches the specific timeout exception.
                if not reader.running: break  # Breaks loop if video is genuinely over.
                continue  # Skips iteration if reader is just temporarily stalled.

            frame_counter += 1  # Ticks the master frame counter.
            elapsed_seconds = int(frame_counter / fps)  # Calculates absolute video timeline seconds.
            frame_height, frame_width = frame.shape[:2]  # Extracts dimensions of the currently analyzed frame.

            current_time = time.time()  # Grabs current system time.
            frame_times.append(current_time)  # Logs the time this frame was processed.
            while frame_times and (current_time - frame_times[0]) > 3.0:  # Removes logs older than 3 seconds.
                frame_times.popleft()  # Discards old logs.
            if current_time - last_fps_update > 0.5:  # Updates the UI metric strictly twice a second to avoid UI flickering.
                display_fps = int(len(frame_times) / 3.0)  # Averages processed frames over the 3-second window.
                last_fps_update = current_time  # Resets the UI update timer.

            pitch_res = self.pitch_detector.detect(frame)  # Asks the pitch detector to analyze the current frame.

            if not pitch_res["valid"]:  # Checks if the frame is pointing at the sky/crowd.
                if frame_counter % 30 == 0:  # Limits console spam to once per second.
                    print(f"  -> Skipping frame {frame_counter} (No pitch detected)")  # Logs the skip.
                continue  # Immediately skips to the next frame, bypassing the heavy YOLO tracking entirely!

            person_results = self.model.track(frame, classes=[0], conf=PERSON_CONF_THRESHOLD, imgsz=TARGET_WIDTH,
                                              persist=True, tracker="bytetrack_custom.yaml", verbose=False,
                                              max_det=25)  # Tracks persons only, preserving stable track IDs for team clustering.
            ball_results = self.ball_model.predict(frame, classes=[0], conf=BALL_CONF_THRESHOLD,
                                                   imgsz=TARGET_WIDTH, verbose=False)  # ball_detection.pt: class 0="ball". Untracked, on its own model instance - mixing it into the person ByteTrack call (or even sharing the same model object) was suppressing/corrupting person tracking.
            goal_results = self.goal_model.predict(frame, classes=[1], conf=GOAL_CONF_THRESHOLD,
                                                   imgsz=TARGET_WIDTH, verbose=False)  # goal_detection.pt: class 1="goal". Separate model instance from the ball.

            person_boxes = person_results[0].boxes if len(
                person_results) > 0 else None  # Extracts person bounding boxes, or None if nothing came back.
            ball_result_boxes = ball_results[0].boxes if len(
                ball_results) > 0 else None  # Extracts the ball detections, or None if nothing came back.
            goal_result_boxes = goal_results[0].boxes if len(
                goal_results) > 0 else None  # Extracts the goal detections, or None if nothing came back.

            ball_box = None  # Highest-confidence "ball" detection clearing BALL_CONF_THRESHOLD.
            goal_boxes = []  # All "goal" detections clearing GOAL_CONF_THRESHOLD - a small pitch can show both goals at once.
            if ball_result_boxes is not None:
                for b in ball_result_boxes:
                    if ball_box is None or float(b.conf[0]) > float(ball_box.conf[0]):
                        ball_box = b
            if goal_result_boxes is not None:
                for b in goal_result_boxes:
                        goal_boxes.append(b)

            if (person_boxes is None or len(person_boxes) == 0) and (
                    ball_box is None):  # Validates that at least one object was found.
                continue  # Skips to next frame if both calls came back empty.

            boxes = list(person_boxes) if person_boxes is not None else []  # Starts the unified per-object list with all tracked persons.
            player_count = len(boxes)  # Persons are already filtered to PERSON_CONF_THRESHOLD by the call above.

            for box in boxes:  # Iterates through every single detected object in the frame.
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Extracts the four corners of the bounding box.
                cls_id = int(box.cls[0])  # Extracts the numerical class ID (0 or 32).
                label = self.model.names[cls_id]  # Maps ID to string ("person" or "sports ball").
                confidence = float(box.conf[0])  # Extracts model certainty probability.
                track_id = int(box.id[
                                   0]) if box.id is not None else -1  # Extracts the unique ByteTrack ID, or -1 if tracking failed.

                x_center = int((x1 + x2) / 2)  # Calculates geometric center X.
                y_center = int((y1 + y2) / 2)  # Calculates geometric center Y.
                width, height = x2 - x1, y2 - y1  # Calculates box dimensions.

                in_pitch = self.pitch_detector.is_point_in_pitch((x_center, y_center),
                                                                 pitch_res)  # Queries if the object is standing on the grass.

                color_b, color_g, color_r = 0, 0, 0  # Initializes default BGR color values.

                if label == "person":  # Triggers specific logic to extract jersey colors for team assignment later.
                    chest_y1 = int(y1 + height * 0.3)  # Defines crop top (below head).
                    chest_y2 = int(y1 + height * 0.5)  # Defines crop bottom (mid torso).
                    chest_x1 = int(x1 + width * 0.4)  # Defines crop left (squeezing inwards).
                    chest_x2 = int(x2 - width * 0.4)  # Defines crop right.

                    if chest_y2 > chest_y1 and chest_x2 > chest_x1:  # Ensures the mathematical crop box is positive and valid.
                        chest_crop = frame[chest_y1:chest_y2,
                                     chest_x1:chest_x2]  # Slices the numpy array to isolate the chest pixels.
                        avg_color = np.average(np.average(chest_crop, axis=0),
                                               axis=0)  # Mathematically calculates the average BGR color of the chest region.
                        color_b, color_g, color_r = avg_color[0], avg_color[1], avg_color[2]  # Assigns the channels.

                match_data.append({  # Appends a massive dictionary representing one single object on one single frame.
                    "frame": frame_counter, "time_sec": elapsed_seconds,  # Temporal data.
                    "screen_w": frame_width, "screen_h": frame_height,  # Spatial reference data.
                    "total_players": player_count, "type": label,  # Contextual data.
                    "track_id": track_id,  # Crucial linking data for trajectories.
                    "x_center": x_center, "y_center": y_center,  # Spatial coordinate data.
                    "width": width, "height": height,  # Scale data.
                    "in_pitch": in_pitch,  # Boolean spatial flag.
                    "color_b": color_b, "color_g": color_g, "color_r": color_r,  # Visual feature data.
                    "confidence": confidence  # Machine learning metric.
                })

            if ball_box is not None:  # Logs the ball's own row - it comes from a different model instance than the person boxes above, so it can't share the person-labeling loop.
                bx1, by1, bx2, by2 = map(int, ball_box.xyxy[0].tolist())
                match_data.append({
                    "frame": frame_counter, "time_sec": elapsed_seconds,
                    "screen_w": frame_width, "screen_h": frame_height,
                    "total_players": player_count, "type": "sports ball",
                    "track_id": -1,
                    "x_center": int((bx1 + bx2) / 2), "y_center": int((by1 + by2) / 2),
                    "width": bx2 - bx1, "height": by2 - by1,
                    "in_pitch": self.pitch_detector.is_point_in_pitch(
                        (int((bx1 + bx2) / 2), int((by1 + by2) / 2)), pitch_res),
                    "color_b": 0.0, "color_g": 0.0, "color_r": 0.0, "confidence": float(ball_box.conf[0])
                })

            for goal_box in goal_boxes:  # Logs every detected goal as its own row - a small pitch can show both goals in the same frame.
                gx1, gy1, gx2, gy2 = map(int, goal_box.xyxy[0].tolist())  # Extracts the goal box's corners.
                match_data.append({
                    "frame": frame_counter, "time_sec": elapsed_seconds,
                    "screen_w": frame_width, "screen_h": frame_height,
                    "total_players": player_count, "type": "goal",
                    "track_id": -1,
                    "x_center": int((gx1 + gx2) / 2), "y_center": int((gy1 + gy2) / 2),
                    "width": gx2 - gx1, "height": gy2 - gy1,
                    "in_pitch": False, "color_b": 0.0, "color_g": 0.0, "color_r": 0.0,
                    "confidence": float(goal_box.conf[0])
                })

            if frame_counter % CSV_WRITE_CHUNK == 0 and len(
                    match_data) > 0:  # Checks if it's time to dump the RAM buffer to the hard drive.
                chunk_df = pd.DataFrame(data=match_data)  # Converts the list of dictionaries to a Pandas DataFrame.
                write_header = not os.path.exists(self.csv_filename)  # Decides if column headers are needed.
                chunk_df.to_csv(self.csv_filename, mode="a", index=False,
                                header=write_header)  # Appends the data to the CSV on disk.
                match_data.clear()  # Flushes the RAM buffer completely.

            # --- LIVE UI RENDERER (THROTTLED) ---
            if frame_counter % 3 == 0:  # Renders the UI only every 3 frames to save CPU power.
                display_frame = frame.copy()  # Copies the raw frame to draw on.
                for box in boxes:  # Re-iterates the person-only box list just for drawing.
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), UI_COLORS["dark_red"],
                                  UI_SETTINGS["box_thickness"])
                if ball_box is not None:  # Draws the ball separately - it comes from a different model than the boxes list above.
                    bx1, by1, bx2, by2 = map(int, ball_box.xyxy[0].tolist())
                    cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), UI_COLORS["white"],
                                  UI_SETTINGS["box_thickness"])

                minutes, seconds = elapsed_seconds // 60, elapsed_seconds % 60  # Formats time safely.
                cv2.putText(display_frame, f"STAGE 2: Extracting... {minutes:02d}:{seconds:02d}", (20, 40),
                            UI_SETTINGS["font_style"], 0.8, UI_COLORS["white"], 2)  # Renders progress clock.
                cv2.putText(display_frame, f"FPS: {display_fps}", (20, 70), UI_SETTINGS["font_style"], 0.8,
                            UI_COLORS["dark_green"], 2)  # Renders speed metric.
                cv2.imshow("Pipeline LIVE: Stage 2", display_frame)  # Renders the final composited frame window.
                if cv2.waitKey(1) & 0xFF == ord('q'): break  # Refreshes window.

        reader.stop()  # Gracefully signals the background thread to terminate.
        cv2.destroyAllWindows()  # Closes the live preview window safely.

        if len(match_data) > 0:  # Handles any remaining unwritten data after the video finishes.
            final_df = pd.DataFrame(data=match_data)  # Converts to DataFrame.
            final_df.to_csv(self.csv_filename, mode="a", index=False,
                            header=not os.path.exists(self.csv_filename))  # Appends final chunk.

        print(f"[{datetime.now().strftime('%H:%M:%S')}] STAGE 2 Complete. CSV generated.")  # Logs completion.
        return True  # Returns True to signal successful pipeline progression.


# =====================================================================
# STAGE 3: TacticalAnalyzer
# Purely logical class that reads the CSV, assigns teams via KMeans,
# and tracks complex states (possessions, passes, interceptions).
# =====================================================================
class TacticalAnalyzer:  # Defines the class that acts as the "brain" or AI Coach.
    def __init__(self, raw_csv, events_csv, pan_csv=None):  # Initializes with paths to input raw data, output tactical events, and the optional camera-pan log.
        self.raw_csv = raw_csv  # Stores input path.
        self.events_csv = events_csv  # Stores output path.
        self.pan_csv = pan_csv  # Stores the camera pan (dx-per-frame) log path used for the awareness metric.
        self.fps = 30  # Standardizes expected framerate for time-based mathematical constraints.

    def _compute_scan_count(self, pan_df, touch_frame):  # Scores pre-touch situational awareness from camera pan history.
        if pan_df is None: return 0  # No pan data available (e.g. missing/legacy run) - returns a neutral score.

        window_start = touch_frame - int(self.fps * AWARENESS_WINDOW_SEC)  # Defines the lookback window start frame.
        window = pan_df[(pan_df["frame"] >= window_start) & (
                    pan_df["frame"] < touch_frame)]  # Slices the pan history strictly preceding this touch.
        if window.empty: return 0  # No pan history yet (e.g. start of video).

        scan_count = 0  # Initializes the oscillation counter.
        direction = None  # Tracks the last "fast" pan direction observed.
        for dx in window["dx"]:  # Iterates the raw per-frame pan values chronologically.
            if abs(dx) > SCAN_VELOCITY_THRESHOLD:  # Only counts frames where the camera is panning quickly (an active look).
                cur_dir = 1 if dx > 0 else -1  # Classifies the current fast pan as left or right.
                if direction is not None and cur_dir != direction:  # Detects a reversal - a head/shoulder turn.
                    scan_count += 1  # Registers one distinct scan.
                direction = cur_dir  # Updates the tracked direction.
        return scan_count  # Returns the total number of rapid direction reversals in the window.

    def analyze(self):  # Defines the master analysis method.
        print(f"[{datetime.now().strftime('%H:%M:%S')}] STAGE 3: Tactical AI Analysis...")  # Logs start.
        if not os.path.exists(self.raw_csv): return False  # Aborts safely if raw data is missing.

        pan_df = None  # Initializes the optional camera-pan dataset.
        if self.pan_csv and os.path.exists(self.pan_csv):  # Checks if a pan log was produced by Stage 1.
            pan_df = pd.read_csv(self.pan_csv)  # Loads the per-frame pan history into RAM.

        df = pd.read_csv(self.raw_csv, low_memory=False)  # Loads the entire raw dataset into RAM.
        players = df[df["type"] == "person"].copy()  # Creates an isolated subset of only human entities.

        if players.empty: return False  # Aborts if no players exist.

        df = df.drop(columns=["team_cluster"],
                     errors="ignore")  # Drops any team_cluster column left over from a previous analysis run, so re-running Stage 3 stays idempotent.

        tracked_players = players[
            players["track_id"] != -1]  # Filters out players who weren't assigned a stable tracking ID.
        unique_tracked_ids = tracked_players["track_id"].nunique()  # Counts how many distinct players actually have stable tracking.
        if unique_tracked_ids >= 3:  # KMeans requires at least as many samples as clusters - fewer than 3 tracked players means clustering can't run.
            stable_colors = tracked_players.groupby("track_id")[["color_b", "color_g",
                                                                 "color_r"]].median()  # Groups by ID and computes the median color to eliminate lighting shadows.
            kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto").fit(
                chromaticity(stable_colors))  # Fits cluster centroids from denoised per-track medians, in lighting-invariant chromaticity space (fitting quality benefits from track continuity)...

            df["team_cluster"] = -1  # Default for non-person rows (ball/goal) and any person row missing color data.
            person_mask = df["type"] == "person"
            df.loc[person_mask, "team_cluster"] = kmeans.predict(
                chromaticity(df.loc[person_mask, ["color_b", "color_g", "color_r"]]))  # ...but classifies every detection by its OWN color, not by a track_id lookup - ByteTrack frequently reassigns new IDs to the same physical player (occlusion, fast egocentric camera motion), and a join on track_id would flip that player's team color every time a new ID starts.

            stable_mask = person_mask & (df["track_id"] != -1)
            df.loc[stable_mask, "team_cluster"] = df.loc[stable_mask].groupby("track_id")["team_cluster"].transform(
                lambda s: s.mode().iloc[0])  # Smooths out single-frame color noise (motion blur/lighting/angle) within a track's own short lifespan, without reintroducing cross-track_id lookup dependence.
        else:  # Fallback if tracking failed globally or too few players were ever stably tracked.
            df["team_cluster"] = -1  # Assigns failure code.

        df.to_csv(self.raw_csv,
                  index=False)  # Overwrites the raw CSV to inject the newly calculated team clusters for future reference.

        df["is_interpolated"] = False  # Flags every real (non-synthetic) row; kept purely in-memory, never written back to the raw CSV.

        ball_rows = df[df["type"] == "sports ball"].sort_values("frame").drop_duplicates(
            subset="frame", keep="first")  # Isolates one ball detection per frame, chronologically ordered.
        if len(ball_rows) >= 2:  # Interpolation needs at least two real points to bridge between.
            full_range = pd.RangeIndex(int(ball_rows["frame"].min()),
                                       int(ball_rows["frame"].max()) + 1)  # Builds the complete frame timeline spanning the ball's first-to-last sighting.
            indexed = ball_rows.set_index("frame").reindex(full_range)  # Introduces NaN rows for every frame where the ball wasn't detected.
            indexed.index.name = "frame"
            was_missing = indexed["x_center"].isna()  # Remembers which frames had no real detection, before filling anything in.

            numeric_cols = ["x_center", "y_center", "width", "height"]  # Only the ball's geometry needs true interpolation.
            indexed[numeric_cols] = indexed[numeric_cols].interpolate(
                method="linear", limit=MAX_BALL_GAP_FRAMES,
                limit_area="inside")  # Bridges gaps up to MAX_BALL_GAP_FRAMES; never extrapolates past the first/last real sighting.

            indexed["total_players"] = indexed["total_players"].ffill().bfill()  # Fills gaps using the full contiguous timeline, so real neighbors are actually visible.
            indexed["time_sec"] = indexed["time_sec"].interpolate(limit_direction="both")  # Smoothly estimates timeline seconds across the full timeline before slicing.

            filled = was_missing & indexed["x_center"].notna()  # Frames that were missing and got successfully bridged.
            interpolated_rows = indexed[filled].copy()  # Isolates only the newly-synthesized rows (not the original real ones, not the un-bridged long gaps).

            if not interpolated_rows.empty:  # Only merge back if interpolation actually filled something.
                ref_row = ball_rows.iloc[0]  # Borrows constant/reference fields from a real ball row.
                interpolated_rows["screen_w"] = ref_row["screen_w"]  # Screen dimensions are constant per video.
                interpolated_rows["screen_h"] = ref_row["screen_h"]
                interpolated_rows["type"] = "sports ball"  # Restores the type label lost during reindex.
                interpolated_rows["track_id"] = -1  # Synthetic rows carry no real tracker identity.
                interpolated_rows["confidence"] = 0.0  # Marks these as non-detections for any confidence-sensitive logic.
                interpolated_rows["color_b"] = 0.0
                interpolated_rows["color_g"] = 0.0
                interpolated_rows["color_r"] = 0.0
                interpolated_rows["in_pitch"] = True  # Assumes the pitch was valid, since it's flanked by two real on-pitch detections.
                interpolated_rows["team_cluster"] = -1  # The ball never carries a team.
                interpolated_rows["is_interpolated"] = True  # Flags these rows as synthetic.
                interpolated_rows = interpolated_rows.reset_index()  # Restores "frame" as a regular column for concatenation.

                df = pd.concat([df, interpolated_rows], ignore_index=True,
                               sort=False)  # Merges the bridged ball positions back into the main dataset.
                df.sort_values("frame", inplace=True)  # Restores chronological order after the merge.
                print(
                    f"[INFO] Bridged {len(interpolated_rows)} missing ball frames via interpolation (gaps <= {MAX_BALL_GAP_FRAMES} frames).")  # Logs how much the ball trajectory was densified.

        shot_ball_positions = {}  # Frame -> (x_center, y_center, width) lookup used only inside shot classification, tolerating much longer gaps than the general-purpose interpolation above.
        if len(ball_rows) >= 2:  # Same real detections as above; a separate, more permissive pass over them.
            shot_indexed = ball_rows.set_index("frame").reindex(full_range)  # Reuses the same full timeline built earlier.
            shot_indexed.index.name = "frame"
            shot_numeric_cols = ["x_center", "y_center", "width"]  # A fast-moving shot only needs position + width, not height.
            shot_indexed[shot_numeric_cols] = shot_indexed[shot_numeric_cols].interpolate(
                method="linear", limit=SHOT_BALL_GAP_FRAMES,
                limit_area="inside")  # Bridges much longer gaps (~2s) - a real shot can vanish from detection for over a second mid-flight.
            for frame_num, row in shot_indexed[shot_numeric_cols].dropna().iterrows():  # Builds a plain dict for O(1) lookups during the main loop.
                shot_ball_positions[int(frame_num)] = (row["x_center"], row["y_center"], row["width"])

        active_possession = False  # Initializes boolean tracking if the camera wearer controls the ball.
        possession_start_frame = 0  # Initializes marker for possession start.
        frames_held = 0  # Initializes counter for uninterrupted control.
        frames_out = 0  # Initializes grace period counter for momentary ball loss.
        last_zone_by, last_zone_bw = None, None  # Tracks the last known in-zone ball position/width, used as the shot-detection baseline.
        pending_zone_frames = 0  # Counts consecutive in-zone frames not yet confirmed as real possession.
        possession_is_recovery = False  # Flags whether the current possession reclaims a ball that wasn't already ours.
        possession_scan_count = 0  # Stores the pre-touch awareness score for the current possession.

        pass_in_progress = False  # Initializes boolean tracking an airborne pass.
        pass_start_frame = 0  # Initializes marker for pass launch.
        pass_timeout = 0  # Initializes counter to abort tracking.
        pass_hold_time_sec = 0.0  # Initializes float storing possession duration before launch.

        match_events = []  # Initializes list to store official event dictionaries.

        evaluating_pass = False  # Initializes boolean flag for the 15-frame landing evaluation window.
        eval_frame_count = 0  # Initializes counter for the evaluation window.
        flight_history = []  # Initializes list to track distance metrics during flight.
        eval_closest_teams = []  # Initializes list to track proximity dominant teams.
        eval_zonal_teams = []  # Initializes list to track all teams in the landing zone.
        eval_min_dist = float('inf')  # Tracks the minimum ball-to-player distance seen during the landing window, for turnover detection.
        eval_last_bw = 0  # Tracks the last known ball width during the landing window, for turnover proximity scaling.
        prev_bx, prev_by = None, None  # Initializes variables to track ball velocity mathematically.

        self_team = None  # Stores the inferred team_cluster of the camera wearer, learned from successful passes.
        self_team_votes = []  # Accumulates dom_team results from every PASS_SUCCESS to infer self_team by majority vote.
        last_ball_owner = "unknown"  # Tracks who last had the ball ("self", "opponent", or "unknown"/loose).

        has_shot_baseline = False  # Whether we have a valid launch position/width to track shot kinematics from - tracked continuously alongside (not gating) pass-reception evaluation, and only consulted once reception is ruled out.
        shot_frame_count = 0  # Counts frames elapsed since the classification window opened.
        shot_speeds = []  # Collects qualifying high-velocity speed samples during the window.
        shot_max_shrink = 0.0  # Tracks the largest observed frame-to-frame ball width shrink ratio.
        shot_prev_bw = None  # Tracks the previous frame's ball width for shrink-rate calculation.
        shot_min_y = None  # Tracks the highest point (smallest y) the ball reached during the window.
        shot_y_start = None  # Stores the ball's y position at launch, as the trajectory baseline.
        shot_ball_in_goal = False  # Tracks whether the goal-segmentation model confirmed the ball entered the goal during the window.
        shot_prev_x, shot_prev_y = None, None  # Tracks the previous frame's shot-specific ball position, for the shot-local velocity calculation.

        grouped_frames = df.groupby(
            "frame")  # Pre-groups the massive DataFrame by frame number, shifting lookup from O(N) to O(1) for extreme speed.

        for f, frame_data in grouped_frames:  # Iterates sequentially through the grouped data.
            ball_data = frame_data[frame_data["type"] == "sports ball"]  # Isolates the row containing the ball.

            ball_in_self_control_zone = False  # Resets local flag for camera wearer control.
            bx, by, bw = 0, 0, 0  # Initializes local ball coordinates.

            if not ball_data.empty:  # Validates ball presence.
                bx = ball_data.iloc[0]["x_center"]  # Extracts X.
                by = ball_data.iloc[0]["y_center"]  # Extracts Y.
                bw = ball_data.iloc[0]["width"]  # Extracts Width (used as a depth proxy).
                screen_w = ball_data.iloc[0]["screen_w"]  # Extracts native screen width.
                screen_h = ball_data.iloc[0]["screen_h"]  # Extracts native screen height.

                # Define the POV camera possessive control zone: a ball higher in frame must look
                # proportionally bigger to count as close - distinguishes "big ball, actually near me"
                # from "medium ball, actually far away but coincidentally low in a wide/distant shot".
                zone_x_min = screen_w * 0.17  # Mathematically defines the left boundary.
                zone_x_max = screen_w * 0.83  # Mathematically defines the right boundary.
                y_ratio = by / screen_h
                y_deficit = max(0.0, SELF_ZONE_BOTTOM_Y_REF - y_ratio)
                required_width_ratio = SELF_ZONE_BASE_WIDTH_RATIO + SELF_ZONE_Y_SLOPE * y_deficit

                if y_ratio > SELF_ZONE_MIN_Y_RATIO and zone_x_min < bx < zone_x_max and (
                        bw / screen_w) > required_width_ratio:  # Evaluates if ball is spatially inside the zone AND large enough (for its height) to be close.
                    ball_in_self_control_zone = True  # Activates the state confirming ball control.

            if ball_in_self_control_zone:  # Branch handling active ball control.
                last_zone_by, last_zone_bw = by, bw  # Remembers the last in-zone ball position/width as the shot-detection baseline.
                if not active_possession:  # Triggers if brand new possession.
                    pending_zone_frames += 1  # Requires a few consecutive in-zone frames before trusting this as real possession, not a ball that just flashed large as it passed close by.
                    if pending_zone_frames >= ZONE_CONFIRM_FRAMES:
                        active_possession = True  # Shifts state machine to active.
                        possession_start_frame = f - ZONE_CONFIRM_FRAMES + 1  # Backdates the start to when the zone entry actually began.
                        frames_held = ZONE_CONFIRM_FRAMES  # Initializes tick to include the confirmation frames already elapsed.
                        frames_out = 0  # Resets grace period.
                        possession_is_recovery = (last_ball_owner != "self")  # Flags this touch as reclaiming a ball we didn't already have.
                        last_ball_owner = "self"  # The wearer now controls the ball.
                        possession_scan_count = self._compute_scan_count(pan_df, f)  # Scores pre-touch scanning awareness.
                        pending_zone_frames = 0  # Resets the confirmation counter for the next cycle.
                else:  # Continuing possession.
                    frames_held += 1  # Increments tick.
                    frames_out = 0  # Affirms control by resetting grace period.
            else:  # Branch handling ball outside the control zone.
                pending_zone_frames = 0  # Any break in zone-presence resets the confirmation counter - only a genuinely sustained run counts.
                if active_possession:  # Checks if control was just lost.
                    frames_out += 1  # Increments grace period.

                    if frames_out >= int(self.fps * 0.25):  # Checks if grace period (~8 frames) has expired.
                        if frames_held >= int(self.fps * 0.4):  # Hierarchical classification: Dribble.
                            simple_class = "DRIBBLE"
                        elif frames_held >= int(self.fps * 0.15):  # Classification: Control.
                            simple_class = "CONTROL"
                        else:  # Classification: Touch.
                            simple_class = "TOUCH"

                        is_recovery = possession_is_recovery and frames_held >= int(
                            self.fps * RECOVERY_MIN_HOLD_SEC)  # Confirms a reclaimed ball was actually controlled, not just brushed.
                        event_type = f"BALL_RECOVERY -> {simple_class}" if is_recovery else f"POSSESSION -> {simple_class}"  # Labels recoveries distinctly from routine possessions.

                        pass_hold_time_sec = frames_held / self.fps  # Calculates how long the ball was held.
                        reasoning = f"Held for: {pass_hold_time_sec:.2f}s."  # Builds the base reasoning string.
                        if is_recovery:  # Enriches the reasoning with the awareness score for recoveries.
                            reasoning += f" Recovered possession. Pre-touch scans: {possession_scan_count}."

                        match_events.append({  # Appends the concluded possession event.
                            "event_type": event_type, "start_frame": possession_start_frame,
                            "end_frame": f - int(self.fps * 0.2),
                            "pre_touch_scans": possession_scan_count,
                            "reasoning": reasoning
                        })

                        active_possession = False  # Resets state.
                        frames_held = 0  # Clears counter.
                        possession_is_recovery = False  # Resets the recovery flag for the next possession.

                        pass_in_progress = True  # Shifts state to airborne pass tracking.
                        pass_start_frame = f - int(self.fps * 0.25)  # Retrospectively marks the launch frame.
                        pass_timeout = 0  # Resets timeout clock.
                        evaluating_pass = False  # Disables evaluation window.
                        eval_min_dist = float('inf')  # Resets the turnover proximity tracker for the new flight.
                        eval_last_bw = last_zone_bw if last_zone_bw is not None else 0  # Seeds the proximity scale from the last known ball width.

                        has_shot_baseline = last_zone_by is not None and last_zone_bw is not None  # Only track kinematics if we actually saw the ball leaving the zone.
                        shot_frame_count = 0  # Resets the classification window clock.
                        shot_speeds = []  # Clears prior speed samples.
                        shot_max_shrink = 0.0  # Clears prior shrink tracking.
                        shot_prev_bw = last_zone_bw  # Seeds shrink-rate calculation from the last in-zone width.
                        shot_min_y = last_zone_by  # Seeds the trajectory apex from the last in-zone position.
                        shot_y_start = last_zone_by  # Records the launch y-position as the trajectory baseline.
                        shot_ball_in_goal = False  # Resets the goal-mask confirmation flag for the new classification window.
                        shot_prev_x, shot_prev_y = None, None  # Resets the shot-local velocity tracker for the new classification window.

            if pass_in_progress:  # Branch evaluating passes in flight.
                pass_timeout += 1  # Ticks timeout clock.
                speed = 0  # Initializes speed variable.

                if not ball_data.empty:  # Validates ball presence for kinematics.
                    if prev_bx is not None and prev_by is not None:  # Verifies historical data exists.
                        speed = math.hypot(bx - prev_bx,
                                           by - prev_by)  # Calculates pixel velocity via Pythagorean theorem.
                    prev_bx, prev_by = bx, by  # Updates history.

                if has_shot_baseline:  # Tracks shot-like kinematics continuously through the whole flight, in PARALLEL with (never gating) the pass-reception evaluation below - a ball that reaches a teammate is always a pass regardless of how it looked kinematically, so these stats are only consulted once reception is ruled out.
                    shot_frame_count += 1  # Ticks the tracking clock (used only to bound the velocity-sample window below).

                    shot_pos = shot_ball_positions.get(
                        f)  # Looks up this frame's ball position from the shot-specific, long-gap-tolerant trajectory (up to SHOT_BALL_GAP_FRAMES), independent of the general-purpose ball_data for this frame.
                    if shot_pos is not None:  # Validates a position (real or bridged) exists for this frame.
                        shot_bx, shot_by, shot_bw = shot_pos

                        shot_speed = 0  # Initializes this frame's shot-local velocity.
                        if shot_prev_x is not None and shot_prev_y is not None:  # Verifies historical data exists.
                            shot_speed = math.hypot(shot_bx - shot_prev_x,
                                                    shot_by - shot_prev_y)  # Calculates pixel velocity via Pythagorean theorem, using the shot-specific trajectory.
                        shot_prev_x, shot_prev_y = shot_bx, shot_by  # Updates history for the next frame.

                        if shot_frame_count <= SHOT_VELOCITY_WINDOW and shot_bw > 0 and shot_speed > (
                                shot_bw * SHOT_VELOCITY_MULT):  # High Exit Velocity criterion.
                            shot_speeds.append(shot_speed)  # Logs the qualifying high-velocity sample.

                        if shot_prev_bw is not None and shot_prev_bw > 0:  # Z-Axis Acceleration (Shrink Rate) criterion.
                            shrink = (shot_prev_bw - shot_bw) / shot_prev_bw  # Computes the frame-to-frame width shrink ratio.
                            if shrink > shot_max_shrink: shot_max_shrink = shrink  # Tracks the steepest shrink observed.
                        shot_prev_bw = shot_bw  # Updates history for the next frame's shrink calculation.

                        if shot_min_y is None or shot_by < shot_min_y: shot_min_y = shot_by  # Tracks the highest point (smallest y) reached.

                        if not shot_ball_in_goal:  # Once confirmed, no need to keep checking.
                            goal_rows = frame_data[frame_data["type"] == "goal"]  # Only checks THIS frame's goal detection(s), against THIS frame's ball position - a small pitch can show a goal ~40% of the time, so "a goal was visible somewhere in the window" barely filters anything on its own.
                            if not goal_rows.empty:
                                for _, g in goal_rows.iterrows():
                                    margin = g["width"] * 0.5  # Generous tolerance - the ball is small and box edges are approximate.
                                    if (g["x_center"] - g["width"] / 2 - margin) <= shot_bx <= (
                                            g["x_center"] + g["width"] / 2 + margin) and (
                                            g["y_center"] - g["height"] / 2 - margin) <= shot_by <= (
                                            g["y_center"] + g["height"] / 2 + margin):  # Requires the ball to actually be inside (or just at the edge of) THIS goal's box, not merely co-occurring in time.
                                        shot_ball_in_goal = True
                                        break

                if not evaluating_pass:  # Phase 1: Airborne tracking.
                    if not ball_data.empty:  # Validates ball.
                        min_dist = float('inf')  # Initializes minimum distance.
                        for _, player in frame_data[frame_data["type"] == "person"].iterrows():  # Iterates players.
                            d = math.hypot(player["x_center"] - bx,
                                           player["y_center"] - by)  # Calculates spatial distance to ball.
                            if d < min_dist: min_dist = d  # Tracks absolute minimum.
                        flight_history.append(min_dist)  # Logs history for convergence checks.

                        if speed < (bw * 0.8) and pass_timeout > int(
                                self.fps * 0.15):  # Triggers evaluation if velocity drops significantly, implying interception or reception.
                            evaluating_pass = True  # Shifts to Phase 2.
                            eval_frame_count = 0  # Resets evaluation window.

                    if pass_timeout > int(self.fps * 2.0) and (
                            not has_shot_baseline or shot_frame_count >= SHOT_TRAJECTORY_WINDOW):  # Waits for the full kinematic window before giving up empty-handed - a real shot/goal can take longer than the 2s "lost in flight" cutoff to reveal its shrink/trajectory signature.
                        # Nobody clearly received it within the timeout - only now do the kinematics get consulted,
                        # to tell a real powerful kick/shot apart from a pass that was simply never controlled by anyone.
                        is_high_velocity = len(shot_speeds) > 0
                        is_shrinking = shot_max_shrink > SHOT_SHRINK_RATE
                        is_upward = (shot_y_start is not None and shot_y_start > screen_h * SHOT_LOWER_Y_RATIO and
                                     shot_min_y is not None and shot_min_y < screen_h * SHOT_UPPER_Y_RATIO)
                        shot_votes = int(is_high_velocity) + int(is_shrinking) + int(is_upward)
                        speed_note = f" Est. speed index: {max(shot_speeds) * self.fps:.0f} px/s (relative, uncalibrated)." if shot_speeds else " (velocity unconfirmed - inferred from trajectory/shrink only)."

                        if shot_votes >= 2 and shot_ball_in_goal:
                            match_events.append({
                                "event_type": "GOAL_SCORED", "start_frame": pass_start_frame, "end_frame": f,
                                "reasoning": f"GOAL! Ball crossed the goal line.{speed_note}"
                            })
                        elif shot_votes >= 2:
                            match_events.append({
                                "event_type": "POWERFUL_KICK", "start_frame": pass_start_frame, "end_frame": f,
                                "reasoning": f"Powerful Kick: High velocity, moving away fast.{speed_note}"
                            })
                        else:
                            match_events.append({
                                "event_type": "PASS_INCOMPLETE", "start_frame": pass_start_frame, "end_frame": f,
                                "reasoning": "Timeout. Ball lost in flight."
                            })
                        pass_in_progress = False  # Kills state machine.
                        last_ball_owner = "unknown"  # The ball's final owner couldn't be determined.
                        prev_bx, prev_by = None, None  # Clears velocity history for the next possession cycle.

                else:  # Phase 2: Landing Evaluation Window.
                    eval_frame_count += 1  # Ticks evaluation clock.

                    if not ball_data.empty:  # Validates ball.
                        min_dist = float('inf')  # Initializes trackers.
                        closest_team = None
                        for _, player in frame_data[frame_data["type"] == "person"].iterrows():  # Iterates players.
                            d = math.hypot(player["x_center"] - bx, player["y_center"] - by)  # Calculates distance.
                            if d < min_dist:  # Isolates closest entity.
                                min_dist = d
                                closest_team = player["team_cluster"]
                            if d < (bw * 6):  # Logs zonal presence.
                                eval_zonal_teams.append(player["team_cluster"])

                        if min_dist < eval_min_dist: eval_min_dist = min_dist  # Tracks the closest anyone got to the ball during the window, for turnover detection.
                        eval_last_bw = bw  # Remembers the last known ball width, used to scale the turnover proximity check.
                        if closest_team is not None:  # Appends to proximity history.
                            eval_closest_teams.append(closest_team)

                    if eval_frame_count >= int(
                            self.fps * 0.5):  # Phase 3: Ensemble Consensus Voting (after 0.5 sec window).
                        algo1, algo2, algo3 = False, False, False  # Initializes three distinct algorithmic votes.
                        dom_team = None  # Initializes dominant team tracker.

                        if eval_closest_teams:  # Algorithm 1: Proximity Dominance.
                            dom_team = max(set(eval_closest_teams),
                                           key=eval_closest_teams.count)  # Finds statistical mode.
                            if eval_closest_teams.count(dom_team) >= int(
                                self.fps * 0.15): algo1 = True  # Passes if dominant team held proximity long enough.

                        if len(flight_history) >= 3:  # Algorithm 2: Distance Convergence.
                            last_dists = flight_history[-5:]  # Slices recent history.
                            if last_dists[0] > last_dists[-1] and last_dists[-1] < (
                                    bw * 4): algo2 = True  # Passes if distance is actively closing.

                        if eval_zonal_teams and dom_team is not None:  # Algorithm 3: Zonal Dominance.
                            zonal_dom = max(set(eval_zonal_teams),
                                            key=eval_zonal_teams.count)  # Finds mode of zonal presence.
                            if zonal_dom == dom_team: algo3 = True  # Passes if zonal dominance matches proximity dominance.

                        total_votes = algo1 + algo2 + algo3  # Sums the boolean votes.
                        is_self_reception = dom_team is not None and (
                                    self_team is None or int(dom_team) == int(
                                self_team))  # True if the receiving team is unknown-yet (bootstrap) or matches our own inferred team.
                        is_opponent_reception = dom_team is not None and self_team is not None and int(
                            dom_team) != int(self_team)  # True only once self_team is known and the receiver is clearly the other side.

                        time_since_exit = (f - pass_start_frame) / self.fps  # Calculates how long ago the ball left the control zone.
                        is_turnover = (dom_team is not None and is_opponent_reception and
                                      time_since_exit < TURNOVER_TIME_LIMIT_SEC and
                                      eval_min_dist < (eval_last_bw * TURNOVER_PROXIMITY_MULT))  # Opponent closed in fast and won it.

                        if total_votes >= 2 and dom_team is not None and is_self_reception:  # Executes Success Branch if majority consensus is reached AND the ball stayed with our own team - always wins over kinematics, no matter how hard/fast the ball was struck.
                            self_team_votes.append(dom_team)  # Reinforces (or bootstraps) the inferred self_team from this confirmed reception.
                            self_team = max(set(self_team_votes),
                                            key=self_team_votes.count)  # Recomputes self_team as the majority vote so far.

                            pressure_count = 0  # Initializes AI Coach metrics.
                            open_teammates = 0
                            target_player = None

                            teammates = frame_data[(frame_data["type"] == "person") & (
                                        frame_data["team_cluster"] == dom_team)]  # Isolates teammates.
                            enemies = frame_data[
                                (frame_data["type"] == "person") & (frame_data["team_cluster"] != dom_team) & (
                                    frame_data["team_cluster"].notna())]  # Isolates enemies.

                            min_t_dist = float('inf')  # Locates the specific receiving entity.
                            for _, t in teammates.iterrows():
                                d = math.hypot(t["x_center"] - bx, t["y_center"] - by)
                                if d < min_t_dist:
                                    min_t_dist = d
                                    target_player = t

                            if target_player is not None:  # Calculates advanced pressure metrics.
                                tx, ty = target_player["x_center"], target_player["y_center"]
                                for _, e in enemies.iterrows():
                                    if math.hypot(e["x_center"] - tx,
                                                  e["y_center"] - ty) < 150:  # 150px pressure radius.
                                        pressure_count += 1

                                for _, t in teammates.iterrows():  # Scans for better options.
                                    if t["track_id"] == target_player["track_id"]: continue
                                    e_count = sum(1 for _, e in enemies.iterrows() if
                                                  math.hypot(e["x_center"] - t["x_center"],
                                                             e["y_center"] - t["y_center"]) < 150)
                                    if e_count == 0: open_teammates += 1

                            match_events.append({  # Appends the completed successful pass.
                                "event_type": "PASS_SUCCESS", "start_frame": pass_start_frame, "end_frame": f,
                                "hold_time": pass_hold_time_sec, "pressure": pressure_count,
                                "open_options": open_teammates,
                                "reasoning": f"Hold: {pass_hold_time_sec:.1f}s. Pressure: {pressure_count}. Open Options: {open_teammates}"
                            })
                            last_ball_owner = "self"  # The ball stayed within our own team - a confirmed reception always wins over kinematics, no matter how hard/fast the ball was struck.
                            pass_in_progress = False
                            evaluating_pass = False
                            prev_bx, prev_by = None, None
                        elif has_shot_baseline and shot_frame_count < SHOT_TRAJECTORY_WINDOW:
                            # Not received by our team (yet) - but the shot-kinematics window hasn't fully matured, so it's too early to
                            # write this off as incomplete/turnover. A real shot/goal's shrink+trajectory signature can take longer to
                            # develop than this 0.5s reception check. Keep tracking (re-checking reception each frame) until it matures.
                            eval_frame_count = 0
                        else:
                            # The ball did NOT reach a teammate, and the shot-kinematics window (if any) has run its course - only now
                            # do the kinematics get consulted, to tell a goal-directed/uncontrolled powerful kick apart from a routine
                            # turnover or lost pass.
                            is_high_velocity = len(shot_speeds) > 0
                            is_shrinking = shot_max_shrink > SHOT_SHRINK_RATE
                            is_upward = (shot_y_start is not None and shot_y_start > screen_h * SHOT_LOWER_Y_RATIO and
                                         shot_min_y is not None and shot_min_y < screen_h * SHOT_UPPER_Y_RATIO)
                            shot_votes = int(is_high_velocity) + int(is_shrinking) + int(is_upward)
                            speed_note = f" Est. speed index: {max(shot_speeds) * self.fps:.0f} px/s (relative, uncalibrated)." if shot_speeds else " (velocity unconfirmed - inferred from trajectory/shrink only)."

                            if shot_votes >= 2 and shot_ball_in_goal:
                                match_events.append({
                                    "event_type": "GOAL_SCORED", "start_frame": pass_start_frame, "end_frame": f,
                                    "reasoning": f"GOAL! Ball crossed the goal line.{speed_note}"
                                })
                                last_ball_owner = "unknown"
                            elif shot_votes >= 2:
                                match_events.append({
                                    "event_type": "POWERFUL_KICK", "start_frame": pass_start_frame, "end_frame": f,
                                    "reasoning": f"Powerful Kick: High velocity, moving away fast.{speed_note}"
                                })
                                last_ball_owner = "unknown"
                            elif is_turnover:  # Executes Turnover Branch if an opponent closed in fast and took it within the time limit, without a strong shot signature.
                                match_events.append({
                                    "event_type": "TURNOVER_LOST", "start_frame": pass_start_frame, "end_frame": f,
                                    "reasoning": f"Dispossessed by opponent {time_since_exit:.2f}s after losing control."
                                })
                                last_ball_owner = "opponent"  # The opposing team now has the ball.
                            else:  # Executes Failure Branch if consensus rejected the pass and it wasn't a clear tackle either.
                                match_events.append({
                                    "event_type": "PASS_INCOMPLETE", "start_frame": pass_start_frame, "end_frame": f,
                                    "reasoning": "Incomplete. Ensemble rejected."
                                })
                                last_ball_owner = "unknown"  # No team clearly gained control - treat the next touch as a potential recovery.

                            pass_in_progress = False  # Resets state machine.
                            evaluating_pass = False
                            prev_bx, prev_by = None, None

        if len(match_events) > 0:  # Executes final export logic.
            events_df = pd.DataFrame(data=match_events)  # Converts to DataFrame.
            events_df.drop_duplicates(subset=["start_frame", "event_type"],
                                      inplace=True)  # Purges same-frame duplicates.
            events_df.to_csv(self.events_csv, index=False)  # Saves tactical events to disk.
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] STAGE 3 Complete. Exported {len(events_df)} tactical events.")  # Logs completion.
        else:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] STAGE 3 Complete. No significant events found.")  # Logs empty completion.

        return True  # Returns True to signal successful pipeline progression.


# =====================================================================
# STAGE 4: HighlightRenderer
# Reads the STABILIZED high-res video and the events CSV.
# Renders broadcast-style graphics (radar, spotlights, AI Coach cards).
# =====================================================================
class HighlightRenderer:  # Defines the class that produces the final visual output.
    def __init__(self, video_path, raw_csv, events_csv, output_video):  # Initializes with all necessary file paths.
        self.video_path = video_path  # Stores the path to the high-res stabilized video.
        self.raw_csv = raw_csv  # Stores raw data path.
        self.events_csv = events_csv  # Stores tactical events path.
        self.output_video = output_video  # Stores destination path.

    def render(self):  # Defines the master rendering method.
        print(f"[{datetime.now().strftime('%H:%M:%S')}] STAGE 4: Rendering High-Res Highlights...")  # Logs start.
        if not os.path.exists(self.events_csv) or not os.path.exists(self.video_path) or not os.path.exists(
                self.raw_csv):  # Validates dependencies.
            print("[ERROR] Missing files for Stage 4. Aborting render.")  # Logs failure.
            return False  # Aborts cleanly.

        events_df = pd.read_csv(self.events_csv)  # Loads events table.
        raw_df = pd.read_csv(self.raw_csv)  # Loads tracking table.

        if events_df.empty or raw_df.empty: return False  # Aborts if datasets are empty.

        events_df.sort_values(by="start_frame", inplace=True)  # Ensures chronological processing.

        cap = cv2.VideoCapture(self.video_path)  # Opens the high-res stabilized video.
        fps = cap.get(cv2.CAP_PROP_FPS)  # Extracts framerate.
        target_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Extracts original pristine width.
        target_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Extracts original pristine height.

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Defines standard codec.
        out = cv2.VideoWriter(self.output_video, fourcc, fps,
                              (target_width, target_height))  # Initializes the writer matching native resolution.

        # Calculate coordinate mapping ratio (since tracking was done on 480p, but rendering is on 1080p/720p).
        orig_w = raw_df["screen_w"].iloc[0]  # Extracts the width recorded during tracking (e.g., 480).
        orig_h = raw_df["screen_h"].iloc[0]  # Extracts the height recorded during tracking.
        scale_x = target_width / orig_w  # Mathematical scalar mapping X coordinates back to HD.
        scale_y = target_height / orig_h  # Mathematical scalar mapping Y coordinates back to HD.

        frames_grouped = raw_df.groupby("frame")  # Creates an O(1) lookup hash map for lightning-fast data retrieval.

        raw_clips = []  # Initializes buffer for discrete clips.
        for index, row in events_df.iterrows():  # Converts point events into 10-second intervals.
            e_start = int(row["start_frame"])  # Extracts event trigger frame.
            e_end = int(row["end_frame"])  # Extracts event's own real end frame.
            c_start = max(0, e_start - int(fps * 2))  # Pads start with 2 seconds of visual context.
            # The window must reach past the event's own real end (not just a flat 10s from its start) - GOAL/POWERFUL_KICK
            # events especially can run past 8s once the shot-classification window and pass_start_frame backdating are
            # included, and a card whose trigger frame falls outside the rendered window never fires at all.
            c_end = max(c_start + int(fps * 10), e_end + int(fps * 1.5))
            raw_clips.append({"type": row["event_type"], "c_start": c_start, "c_end": c_end, "e_start": e_start,
                              "e_end": e_end,
                              "reasoning": str(row.get("reasoning", "")),
                              "hold_time": row.get("hold_time", 0), "pressure": row.get("pressure", 0),
                              "open_options": row.get("open_options", 0),
                              "pre_touch_scans": row.get("pre_touch_scans", 0)})

        merged_clips = []  # Initializes buffer for merged, overlapping intervals.
        for clip in raw_clips:  # Iterates to deduplicate overlapping clips.
            if not merged_clips:  # Seeds the list.
                merged_clips.append({"start": clip["c_start"], "end": clip["c_end"], "events": [clip]})
            else:
                last = merged_clips[-1]  # Checks the most recent boundary.
                if clip["c_start"] <= last["end"]:  # Overlap detected.
                    last["end"] = max(last["end"], clip["c_end"])  # Extends the continuous clip.
                    last["events"].append(clip)  # Injects secondary event data.
                else:  # Gap detected.
                    merged_clips.append({"start": clip["c_start"], "end": clip["c_end"],
                                         "events": [clip]})  # Appends as new discrete clip.

        def label_to_color(label):  # Maps any event label string to its UI color, in one place shared by the running banner and every per-event coach card.
            if "GOAL" in label:
                return UI_COLORS["gold"]  # Maps confirmed goal color (checked before SUCCESS/SHOT).
            elif "SUCCESS" in label:
                return UI_COLORS["dark_green"]  # Maps success color.
            elif "KICK" in label:
                return UI_COLORS["yellow"]  # Maps powerful kicks to a highly visible highlight color.
            elif "TURNOVER" in label:
                return UI_COLORS["magenta"]  # Maps turnover (dispossessed) color.
            elif "RECOVERY" in label:
                return UI_COLORS["teal"]  # Maps ball recovery color (checked before DRIBBLE/CONTROL/TOUCH since it shares those suffixes).
            elif "INCOMPLETE" in label:
                return UI_COLORS["dark_red"]  # Maps failure color.
            elif "DRIBBLE" in label:
                return UI_COLORS["orange"]  # Maps dribble color.
            elif "CONTROL" in label:
                return UI_COLORS["cyan"]  # Maps control color.
            elif "TOUCH" in label:
                return UI_COLORS["light_gray"]  # Maps touch color.
            else:
                return UI_COLORS["black"]  # Fallback color.

        for merged in merged_clips:  # Master rendering loop iterating over defined continuous intervals.
            cap.set(cv2.CAP_PROP_POS_FRAMES, merged["start"])  # Seeks the video reader perfectly to the interval start.
            current_frame = merged["start"]  # Local frame tracker.
            ball_trail = []  # Initializes list to render visual ball trail.
            cards_shown = set()  # Tracks which of this merged clip's individual events have already had their own coach card rendered.

            while current_frame <= merged["end"]:  # Inner loop processing frame-by-frame.
                ret, frame = cap.read()  # Extracts the physical high-res image.
                if not ret: break  # Breaks if file ends prematurely.

                active_label = merged["events"][0]["type"]  # Sets default UI text.
                active_reasoning = merged["events"][0]["reasoning"]  # Sets default UI logic.

                for e in merged["events"]:  # Dynamically shifts the displayed text based on timeline progression.
                    if current_frame >= (e["e_start"] - int(fps * 2)):
                        active_label = e["type"]
                        active_reasoning = e["reasoning"]

                radar_w, radar_h = 240, 150  # Hardcodes the pixel dimensions of the minimap radar graphic (scaled up for HD).
                radar_x, radar_y = target_width - radar_w - 30, 30  # Calculates dynamic top-right positioning.
                radar_overlay = frame.copy()  # Creates a duplicate layer for alpha-transparent blending.

                cv2.rectangle(radar_overlay, (radar_x, radar_y), (radar_x + radar_w, radar_y + radar_h),
                              UI_COLORS["radar_bg"], -1)  # Draws the radar background.
                cv2.rectangle(radar_overlay, (radar_x, radar_y), (radar_x + radar_w, radar_y + radar_h),
                              UI_COLORS["radar_border"], 2)  # Draws the radar border.

                if current_frame in frames_grouped.groups:  # O(1) hash map lookup verifying data exists.
                    frame_objects = frames_grouped.get_group(current_frame)  # Retrieves all object rows.

                    for _, obj in frame_objects[
                        frame_objects["type"] == "sports ball"].iterrows():  # Iterates over balls.
                        cx, cy = int(obj["x_center"] * scale_x), int(
                            obj["y_center"] * scale_y)  # Computes HD coordinates natively!

                        ball_trail.append((cx, cy))  # Updates trail history.
                        if len(ball_trail) > 15: ball_trail.pop(0)  # Caps trail length to 0.5 seconds.

                        w, h = obj["width"] * scale_x, obj["height"] * scale_y  # Computes HD dimensions.
                        x1, y1 = int(cx - w / 2), int(cy - h / 2)  # Computes box corners.
                        x2, y2 = int(cx + w / 2), int(cy + h / 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), UI_COLORS["white"],
                                      2)  # Renders box directly onto the HD frame.

                        for i in range(1, len(ball_trail)):  # Renders trailing lines.
                            cv2.line(frame, ball_trail[i - 1], ball_trail[i], UI_COLORS["white"], 2)

                        rx = radar_x + int((cx / target_width) * radar_w)  # Maps X to radar.
                        ry = radar_y + int((cy / target_height) * radar_h)  # Maps Y to radar.
                        cv2.circle(radar_overlay, (rx, ry), 5, UI_COLORS["white"], -1)  # Renders radar dot.

                    for _, obj in frame_objects[frame_objects["type"] == "person"].iterrows():  # Iterates over players.
                        cx, cy = obj["x_center"] * scale_x, obj[
                            "y_center"] * scale_y  # Computes HD spatial center natively!
                        w, h = obj["width"] * scale_x, obj["height"] * scale_y  # Computes HD dimensions.

                        team = obj["team_cluster"]  # Extracts team ID.
                        if pd.notna(team):  # Validates ID.
                            if int(team) == 0:
                                color = UI_COLORS["blue"]  # Assigns blue.
                            elif int(team) == 1:
                                color = UI_COLORS["red"]  # Assigns red.
                            else:
                                color = UI_COLORS["yellow"]  # Assigns yellow (referee).

                            bottom_y = int(cy + h / 2)  # Locates feet coordinate.
                            cv2.ellipse(frame, (int(cx), bottom_y), (int(w / 1.5), int(h / 8)), 0, 0, 360, color,
                                        3)  # Renders 3D base spotlight.

                            x1, y1 = int(cx - w / 2), int(cy - h / 2)  # Computes box corners.
                            x2, y2 = int(cx + w / 2), int(cy + h / 2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                                          2)  # Renders thin box directly onto HD frame.

                            rx = radar_x + int((cx / target_width) * radar_w)  # Maps X to radar.
                            ry = radar_y + int((cy / target_height) * radar_h)  # Maps Y to radar.
                            cv2.circle(radar_overlay, (rx, ry), 4, color, -1)  # Renders radar dot.

                cv2.addWeighted(radar_overlay, 0.75, frame, 0.25, 0,
                                frame)  # Fuses the radar overlay transparently onto the HD frame.

                label_color = label_to_color(active_label)  # Looks up the running banner's color from the shared helper.

                cv2.putText(frame, active_label, (40, 80), UI_SETTINGS["font_style"], 2.0, UI_COLORS["black"],
                            4)  # Renders HD text shadow.
                cv2.putText(frame, active_label, (40, 80), UI_SETTINGS["font_style"], 2.0, label_color,
                            2)  # Renders HD primary text.

                reasoning_color = label_color  # Mirrors the title color for the reasoning text.
                cv2.putText(frame, active_reasoning, (20, target_height - 40), UI_SETTINGS["font_style"], 1.2,
                            UI_COLORS["black"], 4)  # Renders HD reasoning shadow.
                cv2.putText(frame, active_reasoning, (20, target_height - 40), UI_SETTINGS["font_style"], 1.2,
                            reasoning_color, 2)  # Renders HD reasoning text.

                out.write(frame)  # Submits the fully composited HD frame into the MP4 file encoder.
                current_frame += 1  # Ticks the loop counter.

                for event_idx, e in enumerate(
                        merged["events"]):  # Checks every individual event in this merged clip, not just the group as a whole - so none of them get skipped.
                    if current_frame != e["e_end"] + int(fps * 0.5) or event_idx in cards_shown:  # Fires shortly after THIS event's own real action concludes, not a fixed 8s-from-start offset that drifts onto unrelated later action once clips are merged.
                        continue
                    cards_shown.add(event_idx)  # Marks this event as having received its own coach card.

                    event_label = e["type"]  # Uses this event's own label, not whichever label happens to be active right now.
                    event_reasoning = e["reasoning"]
                    card_color = label_to_color(event_label)  # Derives the card's color from this specific event.

                    card_overlay = frame.copy()  # Creates an HD overlay copy to render the AI Coach stats card.
                    card_w, card_h = 600, 350  # Defines HD dimensions of the central stats card.
                    card_x, card_y = (target_width - card_w) // 2, (
                                target_height - card_h) // 2  # Mathematically calculates dead center.

                    cv2.rectangle(card_overlay, (card_x, card_y), (card_x + card_w, card_y + card_h),
                                  UI_COLORS["stats_card_bg"], -1)  # Draws the solid background on the overlay.
                    cv2.addWeighted(card_overlay, 0.9, frame, 0.1, 0, frame)  # Fuses the card layer with 90% opacity.
                    cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), card_color,
                                  4)  # Draws a sharp HD border around the card.

                    hold_time = e.get('hold_time', 0)  # Safely extracts hold_time.
                    pressure = e.get('pressure', 0)  # Safely extracts pressure.
                    open_opts = e.get('open_options', 0)  # Safely extracts open options.
                    pre_touch_scans = e.get('pre_touch_scans', 0)  # Safely extracts the pre-touch awareness score.
                    if pd.isna(hold_time): hold_time = 0  # Normalizes missing values from events without a hold time.
                    if pd.isna(pressure): pressure = 0  # Normalizes missing values from events without a pressure count.
                    if pd.isna(open_opts): open_opts = 0  # Normalizes missing values from events without open options.
                    if pd.isna(pre_touch_scans): pre_touch_scans = 0  # Normalizes missing values from events without a scan score.

                    if "GOAL" in event_label:  # Evaluates AI Coach logic for confirmed goals.
                        tip, tip_color = "GOAL! Incredible finish.", UI_COLORS["gold"]
                        timing_tip = event_reasoning if "px/s" in event_reasoning else ""
                    elif "SUCCESS" in event_label:  # Evaluates AI Coach logic for successful pass branches.
                        if pressure >= 2:
                            tip, tip_color = "Great skill! Completed under heavy pressure.", UI_COLORS["orange"]
                        elif open_opts > 0 and pressure == 0:
                            tip, tip_color = "Perfect vision! Found the open man.", UI_COLORS["dark_green"]
                        else:
                            tip, tip_color = "Good solid pass.", UI_COLORS["white"]
                        timing_tip = f"Time to release: {hold_time:.1f}s" if hold_time > 0 else ""
                    elif "KICK" in event_label:  # Evaluates powerful-kick branches (not confirmed as aimed at goal).
                        tip, tip_color = "Powerful Kick: High velocity, moving away fast.", UI_COLORS["yellow"]
                        timing_tip = event_reasoning if "px/s" in event_reasoning else ""
                    elif "TURNOVER" in event_label:  # Evaluates turnover branches.
                        tip, tip_color = "Turnover - dispossessed under pressure. Protect the ball.", UI_COLORS["magenta"]
                        timing_tip = "Shield the ball and check your surroundings earlier."
                    elif "RECOVERY" in event_label:  # Evaluates ball recovery branches.
                        if pre_touch_scans >= 4:
                            awareness_tip = "Excellent scanning."
                        elif pre_touch_scans >= 2:
                            awareness_tip = "Good awareness."
                        else:
                            awareness_tip = "Check your surroundings more."
                        tip, tip_color = "Good recovery! Won the ball back.", UI_COLORS["teal"]
                        timing_tip = f"Pre-Touch Awareness: {int(pre_touch_scans)} scans ({awareness_tip})"
                    elif "INCOMPLETE" in event_label:  # Evaluates failed pass branches.
                        tip, tip_color = "Pass intercepted. Look for space.", UI_COLORS["dark_red"]
                        timing_tip = "Try a shorter, safer pass next time."
                    else:  # Evaluates general touches.
                        tip, tip_color = "Keep ball close in tight spaces.", UI_COLORS["cyan"]
                        timing_tip = ""

                    cv2.putText(frame, "--- AI COACH ANALYSIS ---", (card_x + 100, card_y + 50),
                                UI_SETTINGS["font_style"], 1.0, UI_COLORS["white"], 2)  # Renders HD header.
                    cv2.putText(frame, f"Play Result: {event_label}", (card_x + 30, card_y + 110),
                                UI_SETTINGS["font_style"], 1.0, card_color, 2)  # Renders HD result.

                    if hold_time > 0:  # Renders HD metrics if available.
                        cv2.putText(frame, f"Hold Time: {hold_time:.1f} sec", (card_x + 30, card_y + 160),
                                    UI_SETTINGS["font_style"], 0.8, UI_COLORS["light_gray"], 1)
                        cv2.putText(frame, f"Pressure: {pressure} defenders", (card_x + 30, card_y + 200),
                                    UI_SETTINGS["font_style"], 0.8, UI_COLORS["light_gray"], 1)
                        cv2.putText(frame, f"Missed Options: {open_opts}", (card_x + 30, card_y + 240),
                                    UI_SETTINGS["font_style"], 0.8, UI_COLORS["light_gray"], 1)

                    cv2.putText(frame, tip, (card_x + 30, card_y + 300), UI_SETTINGS["font_style"], 0.8, tip_color,
                                1)  # Renders HD primary tip.
                    if timing_tip: cv2.putText(frame, timing_tip, (card_x + 30, card_y + 330),
                                               UI_SETTINGS["font_style"], 0.8, UI_COLORS["yellow"],
                                               1)  # Renders HD secondary tip.

                    freeze_duration_frames = int(fps * 2.0)  # Calculates exactly 2 seconds of freeze frames.
                    for _ in range(freeze_duration_frames): out.write(
                        frame)  # Dumps the exact same frame repeatedly into the HD file to simulate a pause.

        cap.release()  # Closes source reader safely.
        out.release()  # Finalizes HD MP4 encoding securely.
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] STAGE 4 Complete. Highlights rendered to: {os.path.basename(self.output_video)}")  # Logs success.
        return True  # Returns True to signal successful pipeline progression.


# =====================================================================
# STAGE 5: PipelineManager
# Orchestrates the entire flow: Stabilize -> Extract -> Analyze -> Render.
# Skips stages automatically if output files already exist.
# =====================================================================
class PipelineManager:  # Defines the master class that coordinates the directory processing.
    def __init__(self, folder_path):  # Initializes with the target directory.
        self.folder_path = folder_path  # Stores the absolute directory path.

    def run(self):  # Defines the main execution method.
        print(
            f"\n{'=' * 60}\nINITIALIZING FOOTBALL PIPELINE\nTarget: {self.folder_path}\n{'=' * 60}")  # Prints formatted header.

        search_pattern = os.path.join(self.folder_path, "*.MP4")  # Constructs search wildcard for uppercase extensions.
        video_files = glob.glob(search_pattern)  # Executes disk search.
        video_files.extend(glob.glob(os.path.join(self.folder_path, "*.mp4")))  # Appends lowercase extensions.
        video_files = list(set(video_files))  # Removes any duplicate paths via Set casting.
        video_files.sort()  # Sorts alphabetically to process in chronological order.

        if not video_files:  # Validates directory is not empty.
            print(f"[ERROR] No videos found in {self.folder_path}")  # Warns user.
            return  # Exits safely.

        stabilized_folder = os.path.join(self.folder_path,
                                         "Stabilized_Videos")  # Defines the sterile sub-directory for proxies.
        os.makedirs(stabilized_folder, exist_ok=True)  # Creates the sub-directory if it doesn't exist.

        generated_highlights = []  # Initializes buffer to store paths for the final master stitch.

        for video_path in video_files:  # Master batch loop over all physical files.
            if "highlights" in video_path.lower() or "master" in video_path.lower(): continue  # Protects against recursive loop ingestion.

            base_name = os.path.splitext(os.path.basename(video_path))[0]  # Extracts clean filename without extension.
            print(f"\n{'*' * 50}\nPROCESSING FILE: {base_name}\n{'*' * 50}")  # Prints file header.

            # Defines strict architectural paths.
            stabilized_vid = os.path.join(stabilized_folder, f"{base_name}_stabilized.mp4")  # Path for high-res proxy.
            pan_csv = os.path.join(stabilized_folder, f"{base_name}_pan.csv")  # Path for the per-frame camera pan log (awareness metric).
            raw_csv = os.path.join(self.folder_path, f"{base_name}_data.csv")  # Path for extraction data.
            events_csv = os.path.join(self.folder_path, f"{base_name}_events.csv")  # Path for tactical data.
            highlights_vid = os.path.join(self.folder_path, f"{base_name}_highlights.mp4")  # Path for final HD render.

            # EXECUTE STAGE 1
            if not VideoStabilizer(video_path, stabilized_vid,
                                   pan_csv).process(): continue  # Executes stabilization; skips to next video on failure.

            # EXECUTE STAGE 2
            if not os.path.exists(raw_csv):  # Evaluates cache dependency.
                if not DataExtractor(stabilized_vid,
                                     raw_csv).extract(): continue  # Executes extraction; skips on failure.
            else:
                print(f"[INFO] Raw CSV already exists. Skipping Stage 2.")  # Logs cache hit.

            # EXECUTE STAGE 3
            if not os.path.exists(events_csv):  # Evaluates cache dependency.
                if not TacticalAnalyzer(raw_csv, events_csv,
                                        pan_csv).analyze(): continue  # Executes analysis; skips on failure.
            else:
                print(f"[INFO] Events CSV already exists. Skipping Stage 3.")  # Logs cache hit.

            # EXECUTE STAGE 4
            if not os.path.exists(highlights_vid):  # Evaluates cache dependency.
                if HighlightRenderer(stabilized_vid, raw_csv, events_csv,
                                     highlights_vid).render():  # Executes rendering.
                    generated_highlights.append(highlights_vid)  # Queues success for master merge.
            else:
                print(f"[INFO] Highlights video already exists. Skipping Stage 4.")  # Logs cache hit.
                generated_highlights.append(highlights_vid)  # Queues existing file.

        if generated_highlights:  # Executes final stitching utility if files exist.
            master_output = os.path.join(self.folder_path,
                                         "All_Matches_Master_Highlights.mp4")  # Defines master output path.
            self._concatenate_highlights(generated_highlights, master_output)  # Calls private helper function to merge.

    def _concatenate_highlights(self, video_list, output_path):  # Defines private helper method for video stitching.
        print(f"\n{'=' * 50}\nSTITCHING MASTER HIGHLIGHT REEL...\n{'=' * 50}")  # Prints header.
        cap = cv2.VideoCapture(video_list[0])  # Opens first file as reference.
        fps = cap.get(cv2.CAP_PROP_FPS)  # Extracts reference FPS.
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Extracts reference width.
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Extracts reference height.
        cap.release()  # Closes reference.

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                              (width, height))  # Initializes master writer.

        for vid_path in video_list:  # Iterates files linearly.
            print(f"Adding {os.path.basename(vid_path)}...")  # Logs progress.
            cap = cv2.VideoCapture(vid_path)  # Opens individual clip.
            while True:  # Sequential frame dump loop.
                ret, frame = cap.read()  # Extracts image matrix.
                if not ret: break  # Validates end.
                out.write(frame)  # Commits to master encoder.
            cap.release()  # Closes clip.

        out.release()  # Finalizes master file.
        print(f"SUCCESS! Master highlights saved to: {output_path}")  # Logs absolute completion.


if __name__ == "__main__":  # Python idiom preventing unintended execution if imported as module.
    # Defines the absolute target directory containing raw footage.
    FOLDER_PATH = r"C:\Users\benbe\OneDrive\שולחן העבודה\final project 4.5\video"

    # Kicks off the entire orchestrated pipeline.
    PipelineManager(FOLDER_PATH).run()