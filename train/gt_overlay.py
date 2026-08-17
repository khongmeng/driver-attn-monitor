"""Ground-truth overlay demo: play a raw DMD video with the DMD annotation
(not a trained model's prediction) burned in, next to the per-frame cascade
feature values already extracted for that session.

Purpose: sanity-check what the label the classifier is trained against
actually looks like on video, side by side with the signals ("features")
that are supposed to predict it — independent of any classifier's output.

Reads the already-extracted ``train/output/features/<session>_features.csv``
(has both the per-frame cascade output and the raw ``gt_*`` DMD annotation
columns) and re-derives the same state label as `train/build_dataset.py`
(`_label_row` / `_gt_perclos`, identical thresholds/window), so the label
shown here matches exactly what `train_state.py` trains against.

The feature CSV is stride-2 (one row per 2 video frames); frames without a
row reuse ("forward-fill") the nearest earlier row's values so the overlay
stays in sync frame-for-frame with the source video.

Usage:
    python -m train.gt_overlay --source path\to\clip_rgb_face.mp4
        # auto-locates train/output/features/<session>_features.csv by name
    python -m train.gt_overlay --source clip.mp4 --features path\to.csv
    python -m train.gt_overlay --source clip.mp4 --save out.mp4 --no-window
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd

from .build_dataset import _gt_perclos, _label_row

COLORS = {
    "FOCUSED": (80, 220, 80),
    "DISTRACTED": (0, 165, 255),
    "DROWSY": (60, 60, 235),
    "TIRED": (0, 230, 230),
    "NO_FACE": (150, 150, 150),
    "UNLABELED": (100, 100, 100),
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _panel(frame, x, y, w, h, alpha=0.55):
    sub = frame[y:y + h, x:x + w]
    if sub.size:
        frame[y:y + h, x:x + w] = (sub.astype("float32") * (1 - alpha)).astype("uint8")


def _find_features_csv(source: str) -> str:
    stem = os.path.basename(source)
    stem = stem[:-len("_rgb_face.mp4")] if stem.endswith("_rgb_face.mp4") else os.path.splitext(stem)[0]
    hits = glob.glob(os.path.join("train", "output", "features", f"{stem}_features.csv"))
    if not hits:
        raise SystemExit(
            f"could not auto-locate a features CSV for '{source}' (looked for "
            f"train/output/features/{stem}_features.csv) — pass --features explicitly."
        )
    return hits[0]


def load_gt(csv_path: str, fps: float, drowsy_thresh: float, drowsy_window: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values("frame").reset_index(drop=True)
    df["gt_perclos"] = _gt_perclos(df, fps, drowsy_window)
    df["_drowsy_thresh"] = drowsy_thresh
    raw_state = df.apply(_label_row, axis=1)
    df = df.drop(columns="_drowsy_thresh")
    df["state"] = np.where(df["has_face"] == 1, raw_state.fillna("UNLABELED"), "NO_FACE")
    return df


def draw(frame, row, fps_disp):
    h, w = frame.shape[:2]
    state = row["state"] if row is not None else "NO_FACE"
    col = COLORS.get(state, (200, 200, 200))

    _panel(frame, 0, 0, w, 60, 0.5)
    cv2.putText(frame, f"GT: {state}", (16, 44), _FONT, 1.2, col, 3, cv2.LINE_AA)
    if row is not None:
        cv2.putText(frame, str(row.get("task", "")), (w - 220, 44), _FONT, 0.7,
                    (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps_disp:4.1f} fps", (w - 130, h - 16), _FONT, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA)

    if row is None or row.get("has_face") != 1:
        return

    x0, y0, x1, y1 = (int(row["bbox_x0"]), int(row["bbox_y0"]),
                      int(row["bbox_x1"]), int(row["bbox_y1"]))
    cv2.rectangle(frame, (x0, y0), (x1, y1), col, 2)

    def g(k, fmt="{:.2f}"):
        v = row.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        return fmt.format(v) if isinstance(v, (int, float)) else str(v)

    # left panel: cascade features (same signals the classifier consumes)
    px, py = 12, 70
    _panel(frame, px, py, 300, 150)
    cv2.putText(frame, "cascade features", (px + 10, py + 20), _FONT, 0.5,
                (150, 200, 255), 1, cv2.LINE_AA)
    feat_rows = [
        ("eye", f"{g('eye_open_prob')}  {'closed' if row.get('eye_closed') else 'open'}"),
        ("PERCLOS", f"{g('perclos', '{:.0%}')}  blinks {g('blink_count', '{:.0f}')}"),
        ("head y/p", f"{g('yaw', '{:+.0f}')} / {g('pitch', '{:+.0f}')}"),
        ("gaze y/p", f"{g('gaze_yaw', '{:+.0f}')} / {g('gaze_pitch', '{:+.0f}')}"),
    ]
    for i, (k, v) in enumerate(feat_rows):
        yy = py + 46 + i * 24
        cv2.putText(frame, k, (px + 10, yy), _FONT, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.putText(frame, v, (px + 120, yy), _FONT, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

    # right panel: raw DMD ground-truth annotation columns that state derives from
    qx, qy = w - 300, 70
    _panel(frame, qx, qy, 288, 170)
    cv2.putText(frame, "DMD ground truth", (qx + 10, qy + 20), _FONT, 0.5,
                (150, 255, 180), 1, cv2.LINE_AA)
    gt_rows = [
        ("eyes_state", g("gt_eyes_state", "{}")),
        ("eye_closed", g("gt_eye_closed", "{:.0f}")),
        ("yawn", g("gt_yawn", "{:.0f}")),
        ("gt_perclos15s", g("gt_perclos", "{:.0%}")),
        ("looking_road", g("gt_looking_road", "{:.0f}")),
        ("gaze_off_road", g("gt_gaze_off_road", "{:.0f}")),
        ("driver_action", g("gt_driver_action", "{}")),
    ]
    for i, (k, v) in enumerate(gt_rows):
        yy = qy + 40 + i * 20
        cv2.putText(frame, k, (qx + 10, yy), _FONT, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.putText(frame, str(v)[:16], (qx + 140, yy), _FONT, 0.45, (240, 240, 240), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="GT-overlay demo (DMD annotation -> video, no model).")
    ap.add_argument("--source", required=True, help="path to a DMD *_rgb_face.mp4")
    ap.add_argument("--features", default=None,
                    help="features CSV (default: auto-located in train/output/features/)")
    ap.add_argument("--drowsy-perclos", type=float, default=0.20)
    ap.add_argument("--drowsy-window", type=float, default=15.0)
    ap.add_argument("--save", default=None, help="write the annotated video to this path (.mp4)")
    ap.add_argument("--no-window", action="store_true", help="no display window")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    csv_path = args.features or _find_features_csv(args.source)
    print(f"loading ground truth from {csv_path}")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"could not open source: {args.source}")
    fps_val = cap.get(cv2.CAP_PROP_FPS) or 29.76
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    df = load_gt(csv_path, fps_val, args.drowsy_perclos, args.drowsy_window)
    by_frame = {int(r["frame"]): r for _, r in df.iterrows()}
    print(f"state counts in GT: {df['state'].value_counts().to_dict()}")

    writer = None
    if args.save:
        out_dir = os.path.dirname(args.save)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    frame_idx = 0
    last_row = None
    print("running — press Esc or q to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in by_frame:
                last_row = by_frame[frame_idx]
            draw(frame, last_row, fps_val)

            if args.save:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps_val, (w, h))
                writer.write(frame)

            if args.no_window:
                if total and frame_idx % 500 == 0:
                    print(f"  frame {frame_idx:,}/{total:,} ({frame_idx/total:.0%})")
            else:
                cv2.imshow("DMS ground truth", frame)
                wait_ms = max(1, int(1000 / fps_val))
                k = cv2.waitKey(wait_ms) & 0xFF
                if k in (27, ord("q")):
                    break
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"saved annotated video -> {args.save}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
