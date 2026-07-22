"""Real-time driver-state demo: camera -> cascade -> classifier -> overlay.

Runs the whole pipeline live so you can test the trained model end to end:
the pretrained feature cascade (SCRFD -> 6DRepNet -> eye-state -> gaze) extracts
features per frame (feature list read from the checkpoint), the trained
Stage-④ MLP classifies the driver state, a short
temporal smoother de-flickers it, and everything is drawn on the video. This is
the runtime that will eventually be ported to the Jetson.

Usage:
    python -m train.run_live                      # default webcam (index 0)
    python -m train.run_live --source 1           # camera index 1
    python -m train.run_live --source clip.mp4    # a video file instead of camera
    python -m train.run_live --mirror             # driver-view (flip display)
    python -m train.run_live --smooth 15          # temporal smoothing window (frames)
    python -m train.run_live --classifier models/state_classifier/binary/state_mlp.pt
                                                  # binary ATTENTIVE/INATTENTIVE model
    python -m train.run_live --source clip.mp4 --save out.mp4 --no-window
                                                  # batch: annotate a video to disk

Press Esc or q to quit.
"""
from __future__ import annotations

import argparse
import os
import time
from collections import deque

import cv2
import numpy as np
import torch
import yaml

from .cascade import build_pipeline
from .train_state import StateMLP, FEATURES, STATES

# state -> BGR colour
COLORS = {
    "FOCUSED": (80, 220, 80),
    "DISTRACTED": (0, 165, 255),
    "DROWSY": (60, 60, 235),
    "TIRED": (0, 230, 230),
    "ATTENTIVE": (80, 220, 80),      # binary classifier (--classes binary)
    "INATTENTIVE": (0, 100, 255),
    "NO_FACE": (150, 150, 150),
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_config(path=None) -> dict:
    path = path or os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class StateClassifier:
    """Loads the trained Stage-④ MLP and maps a FrameFeatures row -> state.

    The class set (4-state or binary) comes from the checkpoint, so the same
    runtime works with any model trained by ``train_state.py``.
    """
    def __init__(self, pt_path: str):
        if not os.path.exists(pt_path):
            raise FileNotFoundError(
                f"Classifier not found: {pt_path}. Train it first: "
                f"`python -m train.train_state`."
            )
        ckpt = torch.load(pt_path, map_location="cpu")
        sd = ckpt["state_dict"]
        self.states = ckpt.get("states", STATES)
        # feature list comes from the checkpoint, not the current FEATURES
        # constant, so older checkpoints (fewer features) still load; any
        # checkpoint feature the live cascade doesn't compute (e.g. the
        # rolling temporal features, not yet wired into the live pipeline)
        # falls back to NaN -> neutralised to the training mean below.
        self.features = ckpt.get("features", FEATURES)
        self.mean = sd["mean"]
        self.std = sd["std"]
        self.model = StateMLP(len(self.features), len(self.states), self.mean, self.std)
        self.model.load_state_dict(sd)
        self.model.eval()

    def probs(self, feat) -> np.ndarray:
        vals = [getattr(feat, c, float("nan")) for c in self.features]
        x = torch.tensor([vals], dtype=torch.float32)
        # neutralise any missing feature (e.g. gaze dropped this frame, or a
        # checkpoint feature the live cascade doesn't compute yet)
        x = torch.where(torch.isnan(x), self.mean, x)
        with torch.no_grad():
            p = torch.softmax(self.model(x), dim=1)[0].numpy()
        return p


def _panel(frame, x, y, w, h, alpha=0.55):
    sub = frame[y:y + h, x:x + w]
    if sub.size:
        frame[y:y + h, x:x + w] = (sub.astype("float32") * (1 - alpha)).astype("uint8")


def draw(frame, feat, state, conf, probs, states, fps, mirror):
    h, w = frame.shape[:2]
    if mirror:
        frame[:] = cv2.flip(frame, 1)

    def mx(x):
        return (w - 1 - int(x)) if mirror else int(x)

    col = COLORS.get(state, (200, 200, 200))

    # top banner: the predicted state, big
    _panel(frame, 0, 0, w, 60, 0.5)
    cv2.putText(frame, state, (16, 44), _FONT, 1.3, col, 3, cv2.LINE_AA)
    if state != "NO_FACE":
        cv2.putText(frame, f"{conf:.0%}", (w - 120, 44), _FONT, 1.1, col, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:4.1f} fps", (w - 130, h - 16), _FONT, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA)

    if feat is None or not feat.has_face:
        return

    x0, y0, x1, y1 = int(feat.bbox_x0), int(feat.bbox_y0), int(feat.bbox_x1), int(feat.bbox_y1)
    bx0, bx1 = (mx(x1), mx(x0)) if mirror else (x0, x1)
    cv2.rectangle(frame, (bx0, y0), (bx1, y1), col, 2)

    # feature panel
    px, py = 12, 70
    _panel(frame, px, py, 300, 170)
    rows = [
        ("eye", f"{feat.eye_open_prob:.2f} {'closed' if feat.eye_closed else 'open'}"),
        ("PERCLOS", f"{feat.perclos:.0%}   blinks {feat.blink_count}"),
        ("head y/p", f"{feat.yaw:+.0f} / {feat.pitch:+.0f}"),
        ("gaze y/p", f"{feat.gaze_yaw:+.0f} / {feat.gaze_pitch:+.0f}"),
    ]
    for i, (k, v) in enumerate(rows):
        yy = py + 26 + i * 24
        cv2.putText(frame, k, (px + 10, yy), _FONT, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.putText(frame, v, (px + 120, yy), _FONT, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    # per-class prob bars
    by = py + 26 + 4 * 24
    for i, s in enumerate(states):
        yy = by + i * 14
        bw = int(probs[i] * 150)
        cv2.rectangle(frame, (px + 120, yy - 8), (px + 120 + bw, yy),
                      COLORS.get(s, (200, 200, 200)), -1)
        cv2.putText(frame, s[:4], (px + 10, yy), _FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)


def open_source(source: str):
    """Open a camera index (trying several Windows backends) or a video file."""
    if not source.isdigit():
        return cv2.VideoCapture(source), False
    idx = int(source)
    for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened() and cap.read()[0]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return cap, True
        cap.release()
    return cv2.VideoCapture(idx), True   # last resort (main() reports if unopened)


def main():
    ap = argparse.ArgumentParser(description="Live driver-state demo (camera -> state).")
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--config", default=None)
    ap.add_argument("--classifier", default="models/state_classifier/state_mlp.pt")
    ap.add_argument("--smooth", type=int, default=15, help="temporal smoothing window (frames)")
    ap.add_argument("--mirror", action="store_true", help="driver-view (flip display)")
    ap.add_argument("--save", default=None, help="write the annotated video to this path (.mp4)")
    ap.add_argument("--no-window", action="store_true",
                    help="no display window; with --save, processes a file at full speed")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print("loading cascade + classifier …")
    pipeline = build_pipeline(cfg.get("cascade", {}))
    clf = StateClassifier(args.classifier)
    print(f"classifier states: {clf.states}")

    cap, is_cam = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(
            f"could not open source: {args.source}\n"
            f"  - if this is a camera index, try --source 1 (or 2), close apps using the webcam,\n"
            f"    and check Windows camera privacy settings.\n"
            f"  - no webcam? test on a video instead, e.g.:\n"
            f"      run_live.bat --source datasets\\DMD\\distraction\\...\\<name>_rgb_face.mp4"
        )
    fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps_val <= 1:
        fps_val = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam else 0

    writer = None
    if args.save:
        out_dir = os.path.dirname(args.save)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    prob_hist = deque(maxlen=max(1, args.smooth))
    frame_idx = 0
    t_last = time.time()
    fps_disp = 0.0
    print("running — press Esc or q to quit.")
    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            ts = time.time() if is_cam else frame_idx / fps_val
            feat = pipeline.process_frame(frame, frame_idx, ts)

            if feat.has_face:
                p = clf.probs(feat)
                prob_hist.append(p)
                avg = np.mean(prob_hist, axis=0)
                idx = int(avg.argmax())
                state, conf = clf.states[idx], float(avg[idx])
            else:
                prob_hist.clear()
                p = np.zeros(len(clf.states))
                state, conf = "NO_FACE", 0.0

            now = time.time()
            fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / max(now - t_last, 1e-6))
            t_last = now

            draw(frame, feat, state, conf, p, clf.states, fps_disp, args.mirror)

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
                cv2.imshow("DMS live — driver state", frame)
                # live camera paces itself; a video file must be throttled to its
                # fps so it plays at real-time instead of as fast as we can process.
                if is_cam:
                    wait_ms = 1
                else:
                    wait_ms = max(1, int((1.0 / fps_val - (time.time() - loop_start)) * 1000))
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
