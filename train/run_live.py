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
    python -m train.run_live --classifier models/state_classifier/three_gru_gated/state_gru.pt \
                              models/state_classifier/three_gru_yawn/state_gru.pt
                                                  # ensemble: averages both models' probabilities
                                                  # every frame (must share the same class order)

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
    """Loads the trained Stage-④ classifier and maps a FrameFeatures row -> state.

    The class set (4-state/binary/three) and feature list come from the
    checkpoint, so the same runtime works with any model trained by either
    ``train_state.py`` (a per-frame MLP) or ``train_sequence.py`` (a GRU over
    the frame sequence) — checkpoints from the latter carry a ``hidden`` /
    ``layers`` key, used here to tell them apart. The GRU path carries its
    hidden state across calls (one `StateClassifier` instance = one
    continuous stream), so frame order matters and a fresh instance is needed
    per video/session — matches how `train_sequence.py` evaluates it offline
    (one continuous forward pass per session, not independent per-frame
    windows).
    """
    def __init__(self, pt_path: str):
        if not os.path.exists(pt_path):
            raise FileNotFoundError(
                f"Classifier not found: {pt_path}. Train it first: "
                f"`python -m train.train_state` (MLP) or "
                f"`python -m train.train_sequence` (GRU)."
            )
        ckpt = torch.load(pt_path, map_location="cpu")
        sd = ckpt["state_dict"]
        self.states = ckpt.get("states", STATES)
        # feature list comes from the checkpoint, not the current FEATURES
        # constant, so older checkpoints (fewer features) still load; any
        # checkpoint feature the live cascade doesn't compute falls back to
        # NaN -> neutralised to the training mean below.
        self.features = ckpt.get("features", FEATURES)
        self.mean = sd["mean"]
        self.std = sd["std"]
        self.is_sequence = "hidden" in ckpt and "layers" in ckpt
        if self.is_sequence:
            if ckpt.get("gated_three"):
                from .train_sequence import GatedThreeGRU
                self.model = GatedThreeGRU(self.features, ckpt["fatigue_features"],
                                          ckpt["distract_features"], self.mean, self.std,
                                          hidden=ckpt["hidden"], layers=ckpt["layers"])
            elif ckpt.get("gated_fatigue"):
                from .train_sequence import GatedFatigueGRU
                self.model = GatedFatigueGRU(self.features, ckpt["fatigue_features"],
                                            self.mean, self.std,
                                            hidden=ckpt["hidden"], layers=ckpt["layers"])
            else:
                from .train_sequence import SequenceGRU
                self.model = SequenceGRU(len(self.features), len(self.states), self.mean, self.std,
                                         hidden=ckpt["hidden"], layers=ckpt["layers"])
            self._h = None   # GRU hidden state, carried across frames (a plain tensor for
                             # SequenceGRU, a (h_main, h_fat) tuple for GatedFatigueGRU, a
                             # (h_foc, h_dis, h_fat) tuple for GatedThreeGRU — probs() below
                             # just stores/replays whatever the model returns)
            # the GRU's recurrence has no notion of wall-clock time — it just
            # applies one update per call. Training only ever calls it once
            # every `extraction_stride` native frames (extract_features.py
            # skips the rest entirely, not just downsamples). Advancing it on
            # every native frame at inference runs the recurrence at
            # `extraction_stride`x the trained rate, distorting its learned
            # integration/decay timescale — feed it at the same stride here.
            self.gru_stride = ckpt.get("extraction_stride", 2)
            self._last_p = np.zeros(len(self.states))   # held prediction between stride steps
        else:
            self.model = StateMLP(len(self.features), len(self.states), self.mean, self.std)
        self.model.load_state_dict(sd)
        self.model.eval()

    def probs(self, feat, frame_idx: int = 0) -> np.ndarray:
        """`frame_idx` is the absolute native-video frame number — needed so a
        GRU checkpoint only advances its hidden state every `gru_stride`
        frames (matching training), holding its last prediction in between.
        Kept as an explicit parameter (not an internal counter) so an
        `EnsembleClassifier` of several members stays correctly in sync with
        the same absolute frame position, regardless of how many frames any
        one member has actually advanced on."""
        if self.is_sequence and frame_idx % self.gru_stride != 0:
            return self._last_p   # this frame was never seen in training either
        vals = [getattr(feat, c, float("nan")) for c in self.features]
        if self.is_sequence:
            x = torch.tensor([[vals]], dtype=torch.float32)   # (batch=1, seq=1, features)
            x = torch.where(torch.isnan(x), self.mean, x)
            with torch.no_grad():
                logits, self._h = self.model(x, self._h)
                p = torch.softmax(logits[0, 0], dim=0).numpy()
            self._last_p = p
            return p
        x = torch.tensor([vals], dtype=torch.float32)
        # neutralise any missing feature (e.g. gaze dropped this frame, or a
        # checkpoint feature the live cascade doesn't compute yet)
        x = torch.where(torch.isnan(x), self.mean, x)
        with torch.no_grad():
            p = torch.softmax(self.model(x), dim=1)[0].numpy()
        return p


class EnsembleClassifier:
    """Averages the softmax probabilities of several independently-trained
    `StateClassifier`s on every frame — the same mechanism validated offline
    (see docs/METHODOLOGY.md — the ensemble experiment): no new training, no
    shared state between members, just a per-frame average of each member's
    own opinion. Each member keeps its own GRU hidden state and its own
    stride-hold logic internally (via `StateClassifier.probs`); this class
    only combines their outputs.
    """
    def __init__(self, pt_paths: list):
        self.members = [StateClassifier(p) for p in pt_paths]
        self.states = self.members[0].states
        for p, m in zip(pt_paths[1:], self.members[1:]):
            if m.states != self.states:
                raise SystemExit(f"ensemble members must share the same class order — "
                                 f"{pt_paths[0]} has {self.states}, {p} has {m.states}")

    def probs(self, feat, frame_idx: int = 0) -> np.ndarray:
        ps = [m.probs(feat, frame_idx) for m in self.members]
        return np.mean(ps, axis=0)


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
    ap.add_argument("--classifier", nargs="+",
                    default=["models/gru_gated/state_gru.pt", "models/gru_single/state_gru.pt"],
                    help="one checkpoint path, or several for an ensemble (averages their "
                         "softmax probabilities every frame — see docs/METHODOLOGY.md's "
                         "ensemble experiment). All must share the same class order. Defaults "
                         "to the best result in the project (macro-F1 0.810, see models/README.md).")
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
    clf = EnsembleClassifier(args.classifier) if len(args.classifier) > 1 else StateClassifier(args.classifier[0])
    print(f"classifier states: {clf.states}"
         + (f"  (ensemble of {len(args.classifier)}: {args.classifier})" if len(args.classifier) > 1 else ""))

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
    p = np.zeros(len(clf.states))
    state, conf = "NO_FACE", 0.0
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
                # each classifier (or ensemble member) internally decides
                # whether to advance its GRU on this frame or hold its last
                # prediction, based on its own trained stride — see
                # StateClassifier.probs()
                p = clf.probs(feat, frame_idx)
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
