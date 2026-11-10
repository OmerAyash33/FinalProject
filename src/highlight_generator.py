"""STAGE 3 of the pipeline: renders the final highlight reel onto the original footage."""
import os
import time

import cv2

# config imports torch - must happen before pandas is ever imported anywhere in
# the process, or torch's native DLL init fails on Windows (reproducible:
# `import pandas; import torch` fails, `import torch; import pandas` doesn't -
# some DLL pandas loads shadows one torch's c10.dll depends on).
from config import UI_COLORS, UI_SETTINGS, DETECTION_COLORS, EVENT_COLORS
from logging_utils import log
from team_utils import team_color

import pandas as pd

# On-screen wording of every event type (colours come from config.EVENT_COLORS).
EVENT_LABELS = {
    "PASS_SUCCESS": "PASS COMPLETED",
    "PASS_INCOMPLETE": "PASS INCOMPLETE",
    "SHOT_ON_TARGET": "SHOT ON TARGET",
    "SHOT_OFF_TARGET": "SHOT OFF TARGET",
    "DRIBBLE_SHORT": "DRIBBLE (short)",
    "DRIBBLE_MEDIUM": "DRIBBLE (medium)",
    "DRIBBLE_LONG": "DRIBBLE (long)",
}


def _num(row, key, default=None):
    v = row.get(key, default)
    return default if v is None or pd.isna(v) else v


class HighlightGenerator:
    """
    Renders bounding boxes and event graphics onto the original video at
    full native resolution/quality, scaling Stage 1's TARGET_WIDTH-sized
    tracking coordinates back up to HD. (There is no stabilization stage to
    reverse: Stage 1 tracks the same, unmodified video this stage draws on,
    so the coordinate mapping is a pure scale — no rotation/translation
    correction needed.)

    Everything the detector found is drawn, in the colours defined once in config.py (the same ones the Stage 1 live
    preview uses): goal dark blue, ball white, our team (A) yellow, the opponents (B) magenta, players without a team grey.
    The banner is coloured by event type (config.EVENT_COLORS).

    `events_df`: the events to render (main.py passes the top-N selection). When None, every event of `events_csv` is rendered.
    """

    def __init__(self, original_video, raw_csv, events_csv, output_video, events_df=None):
        self.original_video = original_video
        self.raw_csv = raw_csv
        self.events_csv = events_csv
        self.output_video = output_video
        self.events_df = events_df

    @staticmethod
    def _hd_box(obj, scale_x, scale_y):
        """One CSV row (centre/size in Stage 1's AI-resolution pixels) -> (x1, y1, x2, y2) on the full-size frame."""
        cx, cy = obj["x_center"] * scale_x, obj["y_center"] * scale_y
        w, h = obj["width"] * scale_x, obj["height"] * scale_y
        return int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)

    @staticmethod
    def _card_lines(row):
        """The measurements shown on the summary card, per event type."""
        et = row["event_type"]
        lines = []
        score = _num(row, "score")
        if score is not None:
            lines.append(f"Score: {score:.0f} / 100")
        if et.startswith("DRIBBLE"):
            d = _num(row, "duration_frames", 0)
            lines += [f"Possession: {d:.0f} frames ({d / 30:.1f} s)", f"Opponents close (avg): {_num(row, 'pressure', 0):.1f}"]
        elif et.startswith("PASS"):
            lines += [f"Ball speed: {_num(row, 'speed', 0):.1f} ball-widths/frame", f"Pass length: {_num(row, 'length_px', 0):.0f} px"]
            if et == "PASS_SUCCESS":
                lines.append(f"Opponents around receiver: {_num(row, 'pressure', 0):.0f}")
        else:
            lines += [f"Ball speed: {_num(row, 'speed', 0):.1f} ball-widths/frame", f"Frames overlapping the goal: {_num(row, 'overlap_frames', 0):.0f}"]
        return lines

    def create_highlights(self):
        """Render every selected event onto the original video, with a freeze-frame summary card after each clip."""
        if not os.path.exists(self.original_video) or not os.path.exists(self.raw_csv):
            return
        if self.events_df is None:
            if not os.path.exists(self.events_csv):
                return
            self.events_df = pd.read_csv(self.events_csv)
        events_df = self.events_df.copy()
        raw_df = pd.read_csv(self.raw_csv, low_memory=False)

        if events_df.empty or raw_df.empty:
            return
        log("Starting Highlights Rendering on original video...")
        render_start_time = time.time()

        events_df.sort_values(by="start_frame", inplace=True)
        cap = cv2.VideoCapture(self.original_video)
        fps = cap.get(cv2.CAP_PROP_FPS)

        hd_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        hd_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.output_video, fourcc, fps, (hd_width, hd_height))

        # Stage 1 tracked objects at TARGET_WIDTH resolution; scale those coordinates
        # up to the original HD frame this stage actually draws on.
        ai_w = raw_df["screen_w"].iloc[0]
        ai_h = raw_df["screen_h"].iloc[0]
        scale_to_hd_x = hd_width / ai_w
        scale_to_hd_y = hd_height / ai_h

        frames_grouped = raw_df[raw_df["type"] != "camera"].groupby("frame")
        label_scale = max(0.6, hd_width / 1920 * 0.8)  # "GOAL" tag size follows the video resolution
        label_px = int(40 * label_scale)

        raw_clips = []
        for _, row in events_df.iterrows():
            e_start = int(row["start_frame"])
            c_start = max(0, e_start - int(fps * 2))  # 2s of lead-in context before the event itself
            c_end = c_start + int(fps * 10)  # fixed 10s window per event
            raw_clips.append({"type": row["event_type"], "c_start": c_start, "c_end": c_end, "e_start": e_start, "row": row})

        # Two events close together shouldn't produce two overlapping renders of the
        # same footage — merge any clips whose windows touch into one continuous pass.
        merged_clips = []
        for clip in raw_clips:
            if not merged_clips:
                merged_clips.append({"start": clip["c_start"], "end": clip["c_end"], "events": [clip]})
            else:
                last = merged_clips[-1]
                if clip["c_start"] <= last["end"]:
                    last["end"] = max(last["end"], clip["c_end"])
                    last["events"].append(clip)
                else:
                    merged_clips.append({"start": clip["c_start"], "end": clip["c_end"], "events": [clip]})

        for merged in merged_clips:
            cap.set(cv2.CAP_PROP_POS_FRAMES, merged["start"])
            current_frame = merged["start"]
            log(f"HighlightGenerator: Rendering clip from frame {merged['start']} to {merged['end']}...")
            while current_frame <= merged["end"]:
                ret, frame = cap.read()
                if not ret:
                    break

                active = merged["events"][0]
                for e in merged["events"]:  # later events in the same merged window override the on-screen label
                    if current_frame >= (e["e_start"] - int(fps * 2)):
                        active = e
                active_label = active["type"]
                label_color = EVENT_COLORS.get(active_label, UI_COLORS["white"])

                # CSV frame numbers are 1-based (Stage 1 counts frames from 1), the video position 0-based.
                if (current_frame + 1) in frames_grouped.groups:
                    frame_objects = frames_grouped.get_group(current_frame + 1)

                    # Everything the detector found in this frame is drawn: goal first (so the
                    # players and ball sit on top of it), then the ball, then the players.
                    for _, obj in frame_objects[frame_objects["type"] == "goal"].iterrows():
                        x1, y1, x2, y2 = self._hd_box(obj, scale_to_hd_x, scale_to_hd_y)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), DETECTION_COLORS["goal"], 3)
                        cv2.putText(frame, "GOAL", (x1, max(label_px, y1 - 8)), UI_SETTINGS["font_style"],
                                    label_scale, UI_COLORS["black"], 4)
                        cv2.putText(frame, "GOAL", (x1, max(label_px, y1 - 8)), UI_SETTINGS["font_style"],
                                    label_scale, DETECTION_COLORS["goal"], 2)

                    for _, obj in frame_objects[frame_objects["type"] == "sports ball"].iterrows():
                        x1, y1, x2, y2 = self._hd_box(obj, scale_to_hd_x, scale_to_hd_y)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), DETECTION_COLORS["ball"], 3)

                    for _, obj in frame_objects[frame_objects["type"] == "person"].iterrows():
                        x1, y1, x2, y2 = self._hd_box(obj, scale_to_hd_x, scale_to_hd_y)
                        # team 0 = A (ours, yellow), 1 = B (magenta); NaN / -1 = detected but no team -> grey
                        cv2.rectangle(frame, (x1, y1), (x2, y2), team_color(obj.get("team_cluster")), 2)

                # Black shadow pass underneath, then the colored text on top -> readable on any background.
                banner = EVENT_LABELS.get(active_label, active_label)
                score = _num(active["row"], "score")
                if score is not None:
                    banner += f"  ({score:.0f})"
                cv2.putText(frame, banner, (40, 60), UI_SETTINGS["font_style"],
                            UI_SETTINGS["font_scale_large"] + 0.5, UI_COLORS["black"], 5)
                cv2.putText(frame, banner, (40, 60), UI_SETTINGS["font_style"],
                            UI_SETTINGS["font_scale_large"] + 0.5, label_color, 2)

                out.write(frame)
                current_frame += 1

                if current_frame == merged["end"] + 1:  # clip just ended -> hold on a summary card
                    best = max(merged["events"], key=lambda e: _num(e["row"], "score", 0))
                    card_label = best["type"]
                    card_color = EVENT_COLORS.get(card_label, UI_COLORS["white"])
                    lines = self._card_lines(best["row"])
                    card_overlay = frame.copy()
                    card_w, card_h = 560, 80 + 36 * len(lines)
                    card_x, card_y = (hd_width - card_w) // 2, (hd_height - card_h) // 2

                    cv2.rectangle(card_overlay, (card_x, card_y), (card_x + card_w, card_y + card_h),
                                  UI_COLORS["stats_card_bg"], -1)
                    cv2.addWeighted(card_overlay, 0.9, frame, 0.1, 0, frame)  # 90% opaque card over the frozen frame
                    cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), card_color, 3)
                    cv2.putText(frame, EVENT_LABELS.get(card_label, card_label), (card_x + 20, card_y + 45),
                                UI_SETTINGS["font_style"], 1.0, card_color, 2)
                    for i, text in enumerate(lines):
                        cv2.putText(frame, text, (card_x + 20, card_y + 90 + 36 * i), UI_SETTINGS["font_style"], 0.7,
                                    UI_COLORS["light_gray"], 1)

                    # "Pause": just re-encode the identical frame for ~2 seconds' worth of frames.
                    freeze_duration_frames = int(fps * 2.0)
                    for _ in range(freeze_duration_frames):
                        out.write(frame)

        cap.release()
        out.release()
        log(f"Rendering completed in {time.time() - render_start_time:.1f} sec. Saved to {os.path.basename(self.output_video)}")


def concatenate_highlights(video_list, output_path):
    """Stitch every per-video highlight reel into a single master MP4, in the given order."""
    if not video_list:
        print("No highlight videos generated to merge.")
        return
    print(f"\n{'=' * 50}\nCreating Master Highlight Reel...\n{'=' * 50}")

    cap = cv2.VideoCapture(video_list[0])  # first clip's dimensions/fps define the master file's format
    fps, width, height = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for vid_path in video_list:
        print(f"Adding {os.path.basename(vid_path)} to master reel...")
        cap = cv2.VideoCapture(vid_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()
    print(f"\nSUCCESS! Master highlights saved to: {output_path}")
