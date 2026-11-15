"""STAGE 1 of the pipeline: YOLO detection/tracking + jersey-color extraction (see main.py for the full stage list)."""
import os
import threading
import time
from collections import deque
import queue

import cv2
import numpy as np

# config (and yolo_utils below) import torch - must happen before pandas is ever
# imported anywhere in the process, or torch's native DLL init fails on Windows
# (reproducible: `import pandas; import torch` fails, `import torch; import pandas`
# doesn't - some DLL pandas loads shadows one torch's c10.dll depends on).
from config import (TARGET_WIDTH, PREVIEW_WIDTH, CSV_WRITE_CHUNK, UI_COLORS, UI_SETTINGS,
                     DETECTOR_MODEL_PATH, DETECTOR_MODEL_OPENVINO_DIR, DETECTOR_CONF_THRESHOLD, DETECTOR_VERSION,
                     DETECTION_COLORS, LIVE_PREVIEW_FPS, LIVE_PREVIEW_FILE,
                     PERSON_CLASS_ID, BALL_CLASS_ID, GOAL_CLASS_ID)
from logging_utils import log
from camera_motion import CameraMotionEstimator
from event_params import TEAM_MIN_BOX_HEIGHT_PX
from team_colors import load_reference
from team_utils import LiveTeamColorizer, our_shade, session_of, team_color
from video_reader import FastVideoReader
from yolo_utils import load_yolo_model

import pandas as pd

# CSV "type" column values kept as-is (e.g. "sports ball", not this model's own "ball" class
# name) so tactical_analyzer.py and highlight_generator.py don't need to change at all.
CLASS_ID_TO_LABEL = {PERSON_CLASS_ID: "person", BALL_CLASS_ID: "sports ball", GOAL_CLASS_ID: "goal"}
# Fixed column order for every chunk appended to the CSV (detection rows leave the cam_* columns empty, camera rows the box columns).
CSV_COLUMNS = ["frame", "time_sec", "screen_w", "screen_h", "total_players", "type", "track_id",
               "x_center", "y_center", "width", "height", "color_b", "color_g", "color_r", "confidence",
               "cam_dx", "cam_dy", "cam_resp"]


def _put_text(img, text, org, scale, color, thickness=1):
    """Text with a black outline underneath so it stays readable on grass, sky and floodlit night footage alike."""
    cv2.putText(img, text, org, UI_SETTINGS["font_style"], scale, UI_COLORS["black"], thickness + 2)
    cv2.putText(img, text, org, UI_SETTINGS["font_style"], scale, color, thickness)


def _write_preview(img, path):
    """Latest live view as a JPEG for the desktop app: written to a temp name and swapped in, so the reader never sees a half-written file."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return
    tmp = path + ".tmp"
    for attempt in range(3):
        try:
            with open(tmp, "wb") as f:
                f.write(buf.tobytes())
            os.replace(tmp, path)
            return
        except OSError:
            time.sleep(0.003)   # the reader had the file open at that very moment (Windows refuses the swap): try again in a moment, then give up on this picture


class _PreviewWriter:
    """Encodes and writes the live-view JPEGs on a thread of its own (cv2 and file I/O release the GIL), so the extraction loop only pays for
    drawing the picture. Latest picture wins: if the writer is momentarily behind, the older one is simply replaced, never queued."""

    def __init__(self, path):
        self.path = path
        self._img = None
        self._stop = False
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, img):
        with self._cond:
            self._img = img
            self._cond.notify()

    def close(self):
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._thread.join(timeout=3.0)

    def _run(self):
        while True:
            with self._cond:
                while self._img is None and not self._stop:
                    self._cond.wait()
                img, self._img = self._img, None
            if img is None:
                return
            _write_preview(img, self.path)


class VideoProcessor:
    """
    Reads the source video and runs ONE unified YOLO model (fine-tuned on our
    own chest-cam footage) to detect/track persons, the ball, and the goal
    mouth in a single pass per frame, sampling each player's chest color for
    later team clustering. Every detection becomes one row in `csv_filename`
    — the raw dataset Stage 2 (tactical analysis) is built on. Every single
    frame is processed and logged; nothing is skipped, so the state machine
    downstream gets a complete, gap-free timeline to reason about.

    This replaced an earlier two-model design (a generic COCO person/ball
    model + a separately-trained, domain-mismatched goal model) once the
    unified model's own accuracy caught up - one weight file, one YOLO call
    per frame instead of two.

    ByteTrack itself uses a Kalman filter internally to predict each track's
    next position frame-to-frame — that's the pipeline's actual Kalman
    filter (there is no separate camera-stabilization stage before this one).
    """

    def __init__(self, video_path, csv_filename, show_preview=True):
        self.video_path = video_path
        self.csv_filename = csv_filename
        # False = headless: no window, no preview canvas, no drawing. The CSV is identical either way;
        # batch re-extraction (tools/extract_data_only.py) uses it because drawing + imshow cost time.
        self.show_preview = show_preview
        # Only one trained weight exists (no bigger "GPU-sized" variant), so both
        # hardware tiers load the same file; only the backend (raw PyTorch on GPU
        # vs OpenVINO on CPU) adapts.
        self.model, self.use_fp16 = load_yolo_model(cpu_model_path=DETECTOR_MODEL_PATH,
                                                     gpu_model_path=DETECTOR_MODEL_PATH,
                                                     ov_dir=DETECTOR_MODEL_OPENVINO_DIR, task='detect')

    def process_video(self, chunk_seconds=5, max_frames=None):
        """Run the full extraction loop over `video_path`, writing detections to `csv_filename` as it goes (`max_frames`: stop early, for tests)."""
        if not os.path.exists(self.video_path):
            return
        log("Starting Video Extraction (YOLO & Tracking)...")
        ext_start_time = time.time()
        if os.path.exists(self.csv_filename):
            os.remove(self.csv_filename)  # start clean: a partial CSV from a crashed prior run would corrupt this one

        reader = FastVideoReader(self.video_path, queue_size=128, width=TARGET_WIDTH).start()
        time.sleep(1.0)  # give the background reader a head start so the RAM buffer isn't empty at frame 1

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Preview canvas: fixed width, height derived to preserve the source
        # video's own aspect ratio (never squished/stretched). The app's live-view window is smaller than 1280 px anyway, so the picture that
        # goes to it is drawn at the AI resolution itself (no upscale, a smaller JPEG): that is what keeps a smooth view cheap.
        canvas_w = TARGET_WIDTH if LIVE_PREVIEW_FILE else PREVIEW_WIDTH
        preview_height = int(round(native_height * canvas_w / native_width))
        view_on = self.show_preview            # the live view is drawn at most LIVE_PREVIEW_FPS times a second (see config.py); 'q' / closing the window switches it off, the extraction goes on
        view_every = 1.0 / max(LIVE_PREVIEW_FPS, 0.5)
        last_view = 0.0
        video_name = os.path.basename(self.video_path)
        writer = _PreviewWriter(LIVE_PREVIEW_FILE) if (view_on and LIVE_PREVIEW_FILE) else None
        if view_on and not LIVE_PREVIEW_FILE:
            cv2.namedWindow("Football Video", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Football Video", PREVIEW_WIDTH, preview_height)

        match_data = []
        camera = CameraMotionEstimator()
        # Live team colours for the preview only (Stage 2 does the real clustering from the whole video).
        colorizer = (LiveTeamColorizer(our_shade(session_of(self.video_path)), reference=load_reference(session_of(self.video_path)))
                     if self.show_preview else None)
        frame_counter = 0
        frame_times = deque()  # rolling 3-second window of timestamps, used only for the live FPS readout
        last_fps_update_time = time.time()
        last_fps_text = "FPS: Calc..."

        while reader.more():
            try:
                frame = reader.Q.get(timeout=3.0)
            except queue.Empty:
                if not reader.running:
                    break
                continue  # reader is just momentarily behind, not actually finished

            frame_counter += 1
            if max_frames is not None and frame_counter > max_frames:
                break
            if frame_counter % 1200 == 0:
                log(f"Extraction: Processed {frame_counter} frames...")
            current_time = time.time()
            view = view_on and (current_time - last_view) >= view_every      # draw this frame's view?
            if view:
                last_view = current_time

            frame_times.append(current_time)
            while frame_times and (current_time - frame_times[0]) > 3.0:
                frame_times.popleft()

            if current_time - last_fps_update_time > 0.5:  # throttle the on-screen FPS text to 2 updates/sec
                calc_fps = len(frame_times) / 3.0 if len(frame_times) > 0 else 0
                last_fps_text = f"FPS: {int(calc_fps)}"
                last_fps_update_time = current_time

            elapsed_seconds = int(frame_counter / fps)
            frame_height, frame_width, _ = frame.shape

            # One "camera" row per frame (even when nothing is detected): how far the scene slid in the
            # image since the previous frame. Stage 2 subtracts it from the ball's image velocity so a
            # head turn is not mistaken for a kick. The columns cam_dx/cam_dy/cam_resp exist only on these rows.
            cam_dx, cam_dy, cam_resp = camera.update(frame)
            match_data.append({"frame": frame_counter, "time_sec": elapsed_seconds,
                               "screen_w": frame_width, "screen_h": frame_height,
                               "total_players": -1, "type": "camera", "track_id": -1,
                               "cam_dx": round(cam_dx, 2), "cam_dy": round(cam_dy, 2), "cam_resp": round(cam_resp, 3)})

            # Build the preview canvas by upscaling the small AI-resolution frame
            # WITH interpolation first, then draw boxes/text on top of that. Drawing
            # directly on the small frame and letting the OS window stretch the
            # result afterward is what caused the old blocky/pixelated look - the
            # text itself was rasterized small and then blown up.
            if not view:
                display_frame = None
            elif frame_width == canvas_w:
                display_frame = frame.copy()
            else:
                display_frame = cv2.resize(frame, (canvas_w, preview_height), interpolation=cv2.INTER_CUBIC)
            scale_x = canvas_w / frame_width
            scale_y = preview_height / frame_height

            # ONE call covers all three classes now - conf/max_det tuned for a
            # max-22-players-+1-ref-+1-ball-+2-goals scene: max_det=25 keeps NMS cheap.
            results = self.model.track(frame, classes=[PERSON_CLASS_ID, BALL_CLASS_ID, GOAL_CLASS_ID],
                                       conf=DETECTOR_CONF_THRESHOLD, imgsz=TARGET_WIDTH,
                                       persist=True, tracker="bytetrack.yaml", verbose=False, max_det=25,
                                       half=self.use_fp16)

            if len(results) == 0 or results[0].boxes is None:
                continue
            results = results[0]

            player_count = sum(1 for box in results.boxes if int(box.cls[0]) == PERSON_CLASS_ID)
            ball_count = sum(1 for box in results.boxes if int(box.cls[0]) == BALL_CLASS_ID)
            goal_count = sum(1 for box in results.boxes if int(box.cls[0]) == GOAL_CLASS_ID)

            frame_person_colors = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                label = CLASS_ID_TO_LABEL[cls_id]
                confidence = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1  # -1: ByteTrack couldn't assign a stable ID

                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                width, height = x2 - x1, y2 - y1

                color_b, color_g, color_r = 0, 0, 0

                if label == "person":
                    # Crop a small patch of the torso (avoids head/hair and legs/shorts)
                    # and average its color -> a cheap per-player "jersey color" fingerprint
                    # that Stage 2's KMeans clustering uses to sort players into teams.
                    chest_y1 = int(y1 + height * 0.3)
                    chest_y2 = int(y1 + height * 0.5)
                    chest_x1 = int(x1 + width * 0.4)
                    chest_x2 = int(x2 - width * 0.4)

                    if chest_y2 > chest_y1 and chest_x2 > chest_x1:
                        chest_crop = frame[chest_y1:chest_y2, chest_x1:chest_x2]
                        avg_color = np.average(np.average(chest_crop, axis=0), axis=0)
                        color_b, color_g, color_r = avg_color[0], avg_color[1], avg_color[2]
                        if view and colorizer is not None and confidence >= 0.5 and height >= TEAM_MIN_BOX_HEIGHT_PX:
                            frame_person_colors.append((color_b, color_g, color_r))

                match_data.append({
                    "frame": frame_counter, "time_sec": elapsed_seconds,
                    "screen_w": frame_width, "screen_h": frame_height,
                    "total_players": player_count, "type": label,
                    "track_id": track_id,
                    "x_center": x_center, "y_center": y_center,
                    "width": width, "height": height,
                    "color_b": color_b, "color_g": color_g,
                    "color_r": color_r, "confidence": confidence
                })

                # Box coordinates are in AI-resolution space; scale up to match the preview canvas.
                if display_frame is not None:
                    px1, py1, px2, py2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                    if label == "person":
                        box_color = team_color(colorizer.team_for(track_id, (color_b, color_g, color_r), height))
                    elif label == "goal":
                        box_color = DETECTION_COLORS["goal"]
                    else:
                        box_color = DETECTION_COLORS["ball"]
                    cv2.rectangle(display_frame, (px1, py1), (px2, py2), box_color, 2)
                    if label == "goal":
                        _put_text(display_frame, f"GOAL {confidence:.2f}", (px1, max(15, py1 - 6)), 0.5, box_color)

            if colorizer is not None and frame_person_colors:
                colorizer.observe(frame_counter, frame_person_colors)

            if frame_counter % CSV_WRITE_CHUNK == 0 and len(match_data) > 0:  # flush the RAM buffer periodically
                chunk_df = pd.DataFrame(data=match_data, columns=CSV_COLUMNS)
                chunk_df.drop_duplicates(inplace=True)
                write_header = not os.path.exists(self.csv_filename)
                chunk_df.to_csv(self.csv_filename, mode="a", index=False, header=write_header)
                match_data.clear()

            if display_frame is None:
                continue

            display_frame[0:122, 0:470] = (display_frame[0:122, 0:470] * 0.35).astype(np.uint8)  # dim the corner behind the read-out so dark-blue text stays legible on sky
            minutes, seconds = elapsed_seconds // 60, elapsed_seconds % 60
            total = f"/{n_frames}" if n_frames > 0 else ""
            _put_text(display_frame, f"Extracting Data: {minutes:02d}:{seconds:02d} |Frame:{frame_counter}{total}",
                      (40, 25), 0.6, UI_COLORS["white"])
            _put_text(display_frame, video_name, (40, preview_height - 18), 0.6, UI_COLORS["white"])
            _put_text(display_frame, f"{last_fps_text}   Model: {DETECTOR_VERSION}", (40, 50), 0.6, UI_COLORS["light_gray"])
            # Live detection read-out: what the model found in THIS frame, in the same colours the boxes are drawn in.
            _put_text(display_frame, f"Players: {player_count}", (40, 80), 0.6, DETECTION_COLORS["player_unassigned"])
            _put_text(display_frame, f"Ball: {'YES' if ball_count else '-'}", (200, 80), 0.6, DETECTION_COLORS["ball"])
            _put_text(display_frame, f"Goal: {'YES' if goal_count else '-'}", (330, 80), 0.6, DETECTION_COLORS["goal"])
            _put_text(display_frame, "Team A (ours)", (40, 110), 0.6, DETECTION_COLORS["team_a"])
            _put_text(display_frame, "Team B", (230, 110), 0.6, DETECTION_COLORS["team_b"])

            if writer is not None:
                writer.submit(display_frame)
            else:
                cv2.imshow("Football Video", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty("Football Video", cv2.WND_PROP_VISIBLE) < 1:
                    view_on = False        # only the live view stops - an accidental 'q' must not cut a long extraction short
                    cv2.destroyAllWindows()

        reader.stop()
        if writer is not None:
            writer.close()
        if self.show_preview and not LIVE_PREVIEW_FILE:
            cv2.destroyAllWindows()

        if len(match_data) > 0:  # flush whatever's left in the buffer once the video ends
            final_df = pd.DataFrame(data=match_data, columns=CSV_COLUMNS)
            final_df.drop_duplicates(inplace=True)
            final_df.to_csv(self.csv_filename, mode="a", index=False, header=not os.path.exists(self.csv_filename))
            log(f"Extraction completed in {time.time() - ext_start_time:.1f} sec.")
