"""Offline feature extraction over DMD videos (PC / training prep).

Runs the production cascade (SCRFD -> 6DRepNet -> eye-state -> temporal) on each
DMD ``*_rgb_face.mp4`` and writes one CSV per session: a per-frame table of
extracted features joined with the DMD ground-truth labels. That table is both:
  * the "does the cascade work on DMD?" check (see ``train/validate.py``), and
  * the training set for the Stage-④ state classifier.

Examples
--------
    # one session, first 1500 frames, quick smoke test
    python -m train.extract_features --task drowsiness --limit-sessions 1 \
        --max-frames 1500

    # everything (drowsiness), every 2nd frame, with overlay videos
    python -m train.extract_features --task drowsiness --stride 2 --annotate

    # a single explicit video
    python -m train.extract_features --video datasets/DMD/.../foo_rgb_face.mp4
"""
from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional

import yaml

from .cascade import build_pipeline
from .cascade.base import FrameFeatures
from .dmd.annotations import load_frame_labels
from .dmd.dataset import DmdSession, find_sessions


def load_config(path: Optional[str] = None) -> dict:
    path = path or os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_label(row: dict, label) -> dict:
    """Attach DMD ground-truth columns (gt_*) to a feature row."""
    # drowsiness set (None = this session doesn't annotate that level)
    def _flag(v):
        return int(v) if v is not None else None
    row["gt_eyes_state"] = label.eyes_state if label else None
    row["gt_eye_closed"] = label.eye_closed if label else None   # 0/1/None
    row["gt_blink"] = _flag(label.blink) if label else None
    row["gt_yawn"] = _flag(label.yawn) if label else None
    row["gt_occluded"] = _flag(label.occluded) if label else None
    # distraction set
    row["gt_looking_road"] = (int(label.looking_road) if label and label.looking_road is not None else None)
    row["gt_gaze_off_road"] = label.gaze_off_road if label else None   # 0/1/None
    row["gt_driver_action"] = label.driver_action if label else None
    return row


def process_session(
    pipeline,
    session: DmdSession,
    out_dir: str,
    stride: int = 1,
    max_frames: Optional[int] = None,
    annotate: bool = False,
    start_frame: int = 0,
    mirror: bool = False,
) -> Optional[str]:
    import cv2
    import pandas as pd

    cap = cv2.VideoCapture(session.face_video)
    if not cap.isOpened():
        print(f"  !! could not open {session.face_video}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.76
    if fps <= 1:
        fps = 29.76
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    labels = load_frame_labels(session.annotation, n_frames=total) if session.annotation else []

    writer = None
    if annotate:
        ann_path = os.path.join(out_dir, f"{session.name}_annotated.mp4")
        writer = cv2.VideoWriter(ann_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    pipeline.reset()
    rows: List[dict] = []
    frame_idx = 0
    processed = 0
    started = time.time()
    print(f"  {session.name}: {w}x{h} @ {fps:.1f}fps, {total or '?'} frames"
          f"{f', {len(labels)} labeled' if labels else ' (no GT)'}")

    # Exact skip with grab() (no decode) so labels stay frame-aligned even on
    # H.264 where POS_FRAMES seeking can snap to a keyframe.
    if start_frame > 0:
        for _ in range(start_frame):
            if not cap.grab():
                break
        frame_idx = start_frame
        print(f"    starting at frame {start_frame} (~{start_frame / fps:.1f}s)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            ts = frame_idx / fps
            feat: FrameFeatures = pipeline.process_frame(frame, frame_idx, ts)
            row = feat.to_dict()
            label = labels[frame_idx] if frame_idx < len(labels) else None
            _merge_label(row, label)
            row["session"] = session.name
            row["task"] = session.task
            rows.append(row)

            if writer is not None:
                _draw_overlay(frame, feat, row, cv2, mirror=mirror)
                writer.write(frame)

            processed += 1
            frame_idx += 1
            if max_frames and processed >= max_frames:
                break
            if processed % 100 == 0:
                rate = processed / max(time.time() - started, 1e-6)
                print(f"    {processed} frames ({rate:.1f} fps)…", end="\r")
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if not rows:
        return None
    df = pd.DataFrame(rows)
    # tidy column order: ids first, then features, then GT
    lead = ["session", "task", "frame"]
    gt = [c for c in df.columns if c.startswith("gt_")]
    mid = [c for c in df.columns if c not in lead + gt]
    df = df[lead + mid + gt]

    out_csv = os.path.join(out_dir, f"{session.name}_features.csv")
    df.to_csv(out_csv, index=False)
    dur = time.time() - started
    face_pct = df["has_face"].mean() * 100
    print(f"\n    -> {out_csv}  ({len(df)} rows, {processed/max(dur,1e-6):.1f} fps, "
          f"face {face_pct:.0f}%)")
    return out_csv


_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
_GREEN = (80, 220, 80)
_RED = (60, 60, 235)
_CYAN = (230, 220, 90)
_GRAY = (190, 190, 190)
_WHITE = (245, 245, 245)


def _panel(frame, cv2, x, y, w, h, alpha=0.55):
    """Semi-transparent dark panel for legible text over video."""
    sub = frame[y:y + h, x:x + w]
    if sub.size:
        dark = (sub.astype("float32") * (1 - alpha)).astype("uint8")
        frame[y:y + h, x:x + w] = dark


def _draw_overlay(frame, feat: FrameFeatures, row: dict, cv2, mirror: bool = False):
    """Draw the overlay. `mirror` flips the *displayed* frame to driver view (so
    the gaze arrow reads intuitively); it never touches the stored feature values
    — those are computed on the original frame in the true camera frame."""
    h, w = frame.shape[:2]
    if mirror:
        frame[:] = cv2.flip(frame, 1)   # horizontal flip -> driver view

    def mx(x):
        return (w - 1 - int(x)) if mirror else int(x)

    # title strip (UI — drawn after the flip so text stays readable)
    _panel(frame, cv2, 0, 0, w, 34, alpha=0.5)
    view = "driver view" if mirror else "camera view"
    cv2.putText(frame, f"DMS cascade: SCRFD + 6DRepNet + open-closed-eye + gaze  ({view})",
                (12, 23), _FONT, 0.6, _WHITE, 1, cv2.LINE_AA)

    if not feat.has_face:
        cv2.putText(frame, "NO_FACE", (16, 80), _FONT, 1.1, _RED, 3, cv2.LINE_AA)
        return

    x0, y0, x1, y1 = int(feat.bbox_x0), int(feat.bbox_y0), int(feat.bbox_x1), int(feat.bbox_y1)
    eye_open = not feat.eye_closed
    box_col = _GREEN if eye_open else _RED
    bx0, bx1 = (mx(x1), mx(x0)) if mirror else (x0, x1)
    cv2.rectangle(frame, (bx0, y0), (bx1, y1), box_col, 2)
    tag = f"EYE {'OPEN' if eye_open else 'CLOSED'}  {feat.eye_open_prob:.2f}"
    cv2.putText(frame, tag, (bx0, max(50, y0 - 8)), _FONT, 0.6, box_col, 2, cv2.LINE_AA)

    # gaze arrow from the eye region (image dir = (gx, -gy)); mirroring the
    # endpoints keeps it aligned with the flipped eyeballs.
    has_gaze = feat.gaze_x == feat.gaze_x  # not NaN
    if has_gaze:
        ex, ey = (x0 + x1) // 2, y0 + int(0.42 * (y1 - y0))
        L = int(0.7 * (x1 - x0))
        ox, oy = mx(ex), ey
        tx, ty = mx(ex + feat.gaze_x * L), int(ey - feat.gaze_y * L)
        cv2.arrowedLine(frame, (ox, oy), (tx, ty), (40, 160, 255), 3,
                        cv2.LINE_AA, tipLength=0.3)

    # info panel (left)
    px, py, pw = 12, 44, 340
    _panel(frame, cv2, px, py, pw, 200)
    def line(i, label, value, col=_WHITE):
        yy = py + 26 + i * 26
        cv2.putText(frame, label, (px + 10, yy), _FONT, 0.55, _GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, value, (px + 160, yy), _FONT, 0.6, col, 2, cv2.LINE_AA)

    line(0, "Eye state", "OPEN" if eye_open else "CLOSED", _GREEN if eye_open else _RED)
    line(1, "PERCLOS / blinks", f"{feat.perclos:.0%}  /  {feat.blink_count}", _CYAN)
    line(2, "Head yaw / pitch", f"{feat.yaw:+.0f} / {feat.pitch:+.0f}", _CYAN)
    line(3, "Gaze yaw / pitch", f"{feat.gaze_yaw:+.0f} / {feat.gaze_pitch:+.0f}"
         if has_gaze else "-", (40, 160, 255))
    # Bottom rows show whichever GT this DMD subset provides.
    gt_look = row.get("gt_looking_road")
    if gt_look is not None and gt_look == gt_look:   # distraction set
        on_road = int(gt_look) == 1
        line(4, "DMD GT gaze", "ON ROAD" if on_road else "OFF ROAD",
             _GREEN if on_road else _RED)
        line(5, "DMD GT action", str(row.get("gt_driver_action")), _WHITE)
    else:                                            # drowsiness set
        line(4, "DMD GT eyes", f"{row.get('gt_eyes_state')}", _WHITE)
        line(5, "DMD GT blink", f"{row.get('gt_blink')}", _WHITE)


def main():
    ap = argparse.ArgumentParser(description="Extract cascade features over DMD videos.")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--dmd-root", default="datasets/DMD", help="DMD dataset root")
    ap.add_argument("--task", default="drowsiness",
                    help="DMD task subset (drowsiness/distraction/gaze); '' for all")
    ap.add_argument("--video", default=None, help="single explicit *_rgb_face.mp4 to run")
    ap.add_argument("--out-dir", default="train/output/features", help="where CSVs go")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None, help="cap processed frames/session")
    ap.add_argument("--start-frame", type=int, default=0, help="skip to this frame before processing")
    ap.add_argument("--limit-sessions", type=int, default=None, help="process at most N sessions")
    ap.add_argument("--annotate", action="store_true", help="also write overlay mp4s")
    ap.add_argument("--mirror", action="store_true",
                    help="flip the annotated video to driver view (display only; "
                         "feature values are unchanged)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.video:
        from .dmd.dataset import _find_annotation, _task_from_path
        v = os.path.abspath(args.video)
        sessions = [DmdSession(face_video=v, annotation=_find_annotation(v),
                               task=_task_from_path(v))]
    else:
        sessions = find_sessions(args.dmd_root, task=args.task or None,
                                 require_annotation=False)
    if args.limit_sessions:
        sessions = sessions[: args.limit_sessions]
    if not sessions:
        raise SystemExit(f"No sessions found under {args.dmd_root} (task={args.task!r}).")

    print(f"Building cascade…")
    pipeline = build_pipeline(cfg.get("cascade", {}))
    print(f"Found {len(sessions)} session(s). Output -> {args.out_dir}\n")

    written = []
    for s in sessions:
        out = process_session(pipeline, s, args.out_dir,
                              stride=args.stride, max_frames=args.max_frames,
                              annotate=args.annotate, start_frame=args.start_frame,
                              mirror=args.mirror)
        if out:
            written.append(out)

    print(f"\nDone. {len(written)} CSV(s) written to {args.out_dir}.")
    if written:
        print("Validate with:")
        print(f"  python -m train.validate {args.out_dir}")


if __name__ == "__main__":
    main()
