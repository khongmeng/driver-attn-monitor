"""Extract face crops + 3-class labels directly from raw DMD video, for the
raw-pixel end-to-end pipeline. No hand-engineered features anywhere in this
script — reuses only:
  * ``train.cascade.face_detector.ScrfdFaceDetector`` — face bbox, for
    cropping only (never any of its downstream pose/eye/gaze outputs)
  * ``train.dmd.dataset`` / ``train.dmd.annotations`` — session discovery and
    raw ground-truth parsing (reading the GT, not a derived feature)
See docs/METHODOLOGY.md §14 for why this pipeline is kept isolated from the
cascade/feature-based one (§6-§13), and ``train/rawpixel/README.md`` for how
to run the full raw-pixel pipeline end to end.

Usage:
    python -m train.rawpixel.extract_crops
    python -m train.rawpixel.extract_crops --limit-sessions 2 --max-frames 500
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
import yaml

from ..cascade.face_detector import ScrfdFaceDetector
from ..dmd.annotations import load_frame_labels
from ..dmd.dataset import DmdSession, find_sessions
from .labels import label_session

FPS_DEFAULT = 29.76


def load_config(path: str = None) -> dict:
    path = path or os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("rawpixel", {})


def crop_face(frame_bgr: np.ndarray, box, size: int, margin: float):
    """Square crop centered on the SCRFD box, expanded by `margin`, resized to
    `size`x`size`. Pads with edge-replication if the expanded box spills past
    the frame edge, so the face stays centered rather than off-crop."""
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = box.x0, box.y0, box.x1, box.y1
    bw, bh = x1 - x0, y1 - y0
    cx, cy = x0 + bw / 2, y0 + bh / 2
    half = max(bw, bh) * (1 + margin) / 2
    x0i, x1i = int(round(cx - half)), int(round(cx + half))
    y0i, y1i = int(round(cy - half)), int(round(cy + half))
    x0c, y0c = max(0, x0i), max(0, y0i)
    x1c, y1c = min(w, x1i), min(h, y1i)
    if x1c <= x0c or y1c <= y0c:
        return None
    crop = frame_bgr[y0c:y1c, x0c:x1c]
    pad_l, pad_t = x0c - x0i, y0c - y0i
    pad_r, pad_b = x1i - x1c, y1i - y1c
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(crop, max(pad_t, 0), max(pad_b, 0),
                                  max(pad_l, 0), max(pad_r, 0), cv2.BORDER_REPLICATE)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def process_session(detector, session: DmdSession, out_dir: str, size: int, margin: float,
                    stride: int, max_frames: int, det_thresh: float) -> str:
    cap = cv2.VideoCapture(session.face_video)
    if not cap.isOpened():
        print(f"  !! could not open {session.face_video}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
    if fps <= 1:
        fps = FPS_DEFAULT
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    gt_labels = load_frame_labels(session.annotation, n_frames=total) if session.annotation else []

    crops, frame_idx = [], []
    frame_i, processed = 0, 0
    started = time.time()
    print(f"  {session.name} [{session.task}]: {total or '?'} frames @ {fps:.1f}fps")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_i % stride != 0:
                frame_i += 1
                continue
            box = detector.detect_largest(frame)
            if box is not None and box.score >= det_thresh:
                crop = crop_face(frame, box, size, margin)
                if crop is not None:
                    crops.append(crop)
                    frame_idx.append(frame_i)
            # frames with no detected face are simply skipped — same net
            # effect as the cascade pipeline's has_face==0 rows getting
            # dropped in build_dataset.py, just done at extraction time here.
            processed += 1
            frame_i += 1
            if max_frames and processed >= max_frames:
                break
            if processed % 200 == 0:
                rate = processed / max(time.time() - started, 1e-6)
                print(f"    {processed} frames ({rate:.1f} fps)…", end="\r")
    finally:
        cap.release()

    if not crops:
        print(f"\n    !! no faces detected in {session.name}, skipping")
        return None

    frame_idx = np.array(frame_idx, dtype=np.int64)
    crops = np.stack(crops).astype(np.uint8)
    state = (label_session(gt_labels, session.task, frame_idx, fps) if gt_labels
            else np.array([None] * len(frame_idx), dtype=object))
    state_str = np.array([s if s else "" for s in state], dtype="U10")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{session.name}.npz")
    np.savez_compressed(out_path, crops=crops, frame_idx=frame_idx, state=state_str,
                        driver=f"{session.group}_{session.subject}", task=session.task)
    dur = time.time() - started
    labeled = int((state_str != "").sum())
    print(f"\n    -> {out_path}  ({len(crops)} crops, {labeled} labeled, "
          f"{len(crops)/max(dur,1e-6):.1f} fps)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Extract face crops + labels for the raw-pixel pipeline.")
    cfg = load_config()
    ap.add_argument("--dmd-root", default="datasets/DMD")
    ap.add_argument("--out-dir", default="train/output/rawpixel/crops")
    ap.add_argument("--size", type=int, default=cfg.get("crop", {}).get("size", 112))
    ap.add_argument("--margin", type=float, default=cfg.get("crop", {}).get("margin", 0.25))
    ap.add_argument("--stride", type=int, default=cfg.get("stride", 2),
                    help="match the cascade pipeline's extraction stride for a fair comparison")
    ap.add_argument("--det-thresh", type=float, default=cfg.get("face", {}).get("det_thresh", 0.5))
    ap.add_argument("--det-size", type=int, default=cfg.get("face", {}).get("det_size", 640))
    ap.add_argument("--limit-sessions", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    sessions = []
    for task in ("drowsiness", "distraction"):
        sessions += find_sessions(args.dmd_root, task=task, require_annotation=True)
    if args.limit_sessions:
        sessions = sessions[: args.limit_sessions]
    if not sessions:
        raise SystemExit(f"No sessions found under {args.dmd_root}")

    print(f"Found {len(sessions)} sessions. Output -> {args.out_dir}")
    detector = ScrfdFaceDetector(det_size=args.det_size, det_thresh=args.det_thresh,
                                 use_gpu=cfg.get("use_gpu", True))

    written = []
    for s in sessions:
        out = process_session(detector, s, args.out_dir, args.size, args.margin,
                              args.stride, args.max_frames, args.det_thresh)
        if out:
            written.append(out)
    print(f"\nDone. {len(written)}/{len(sessions)} sessions written to {args.out_dir}.")


if __name__ == "__main__":
    main()
