"""
Contact sheet of consecutive frames around a moment of a raw video, for judging by eye exactly when an event happens
(kick, touch, end of a dribble) - the manual labels are 1-3 s late, so a label only says "somewhere shortly before".

Each tile is stamped with its frame number (0-based, the same numbering as tools/review_tool/labels.csv) and its distance to the
label in seconds; with --boxes the detections of Stage 1 (pipeline_outputs/<session>_<detector>/<video>_data.csv, 960x540 pixels,
data frame = label frame + 1) are drawn: person green, ball white (type "sports ball"), goal blue, plus the camera pan as text.

Usage:  .venv/Scripts/python.exe tools/frame_sheet.py <session> <video name> <frame> [--before 150] [--after 15] [--step 10]
                                                       [--cols 4] [--tile 480] [--boxes] [--out sheet.png]
        e.g.  tools/frame_sheet.py 2026-07-28 20260728200142_000007 8387 --boxes --out sheet.png
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

COLORS = {"person": (60, 220, 60), "sports ball": (255, 255, 255), "goal": (200, 60, 0)}   # BGR; "sports ball" is the type name in the data CSV


def data_csv(session, video, detector):
    return os.path.join(ROOT, "pipeline_outputs", f"{session}_{detector}", f"{video}_data.csv")


def build_sheet(session, video, frame, before=150, after=15, step=10, cols=4, tile=480, boxes=False, detector="v2", video_path=None, data_path=None):
    path = video_path or os.path.join(ROOT, "raw_footage", session, f"{video}.MP4")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first = max(0, frame - before)
    last = min(n - 1, frame + after)
    want = list(range(first, last + 1, step))
    if frame not in want and first <= frame <= last:
        want.append(frame)
    want = sorted(set(want))
    df = None
    if boxes:
        df = pd.read_csv(data_path or data_csv(session, video, detector))
        df = df[df["frame"].between(first + 1, last + 1)]
    th = int(round(tile * h / w))
    tiles = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    pos = first - 1
    for target in want:
        while pos < target - 1:
            if not cap.grab():
                break
            pos += 1
        ok, im = cap.read()
        if not ok:
            break
        pos += 1
        im = cv2.resize(im, (tile, th), interpolation=cv2.INTER_AREA)
        if df is not None:
            sub = df[df["frame"] == target + 1]
            sx, sy = tile / 960.0, th / 540.0
            for _, r in sub.iterrows():
                if r["type"] in COLORS and not np.isnan(r["x_center"]):
                    x0, y0 = int((r["x_center"] - r["width"] / 2) * sx), int((r["y_center"] - r["height"] / 2) * sy)
                    x1, y1 = int((r["x_center"] + r["width"] / 2) * sx), int((r["y_center"] + r["height"] / 2) * sy)
                    cv2.rectangle(im, (x0, y0), (x1, y1), COLORS[r["type"]], 2 if r["type"] == "sports ball" else 1)
            cam = sub[sub["type"] == "camera"]
            if len(cam):
                c = cam.iloc[0]
                cv2.putText(im, f"pan {c['cam_dx']:.0f},{c['cam_dy']:.0f}", (6, th - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        d = (target - frame) / fps
        text = f"f{target}  {d:+.1f}s"
        cv2.rectangle(im, (0, 0), (150, 24), (0, 0, 0), -1)
        cv2.putText(im, text, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255) if target == frame else (255, 255, 255), 1, cv2.LINE_AA)
        if target == frame:
            cv2.rectangle(im, (0, 0), (tile - 1, th - 1), (0, 255, 255), 3)
        tiles.append(im)
    cap.release()
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        row += [np.zeros_like(tiles[0])] * (cols - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("video")
    ap.add_argument("frame", type=int)
    ap.add_argument("--before", type=int, default=150)
    ap.add_argument("--after", type=int, default=15)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--tile", type=int, default=480)
    ap.add_argument("--boxes", action="store_true")
    ap.add_argument("--detector", default="v2")
    ap.add_argument("--video-path", default=None, help="video file outside raw_footage/ (then <session> only names the output folder)")
    ap.add_argument("--data-path", default=None, help="the _data.csv of that video if it is not in pipeline_outputs/<session>_<detector>/")
    ap.add_argument("--out", default="sheet.png")
    a = ap.parse_args()
    sheet = build_sheet(a.session, a.video, a.frame, a.before, a.after, a.step, a.cols, a.tile, a.boxes, a.detector, a.video_path, a.data_path)
    cv2.imwrite(a.out, sheet)
    print(a.out, sheet.shape[1], "x", sheet.shape[0])


if __name__ == "__main__":
    main()
