"""Run the full analysis stack on a recorded video (or live camera) — a quick
way to eyeball gaze / head-pose / blink-rate and phone-object detection on a
captured cabin clip.

Usage:
  python -m inference.run_video recordings/clip.mp4            # annotate + save
  python -m inference.run_video recordings/clip.mp4 --show     # also live window
  python -m inference.run_video 0 --show                       # live camera (index)

Output is written next to the input as <name>_annotated.mp4 unless --out is given.
"""
import argparse
import os
import time

import cv2
import yaml

from .face_analyzer import FaceAnalyzer
from .object_detector import ObjectDetector
from .state_detector import StateDetector
from .overlay import draw


def load_config() -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(description="Run DMS analysis on a video file or camera.")
    ap.add_argument('source', help="video file path, or integer camera index")
    ap.add_argument('--out', default=None, help="annotated output path (default: <name>_annotated.mp4)")
    ap.add_argument('--show', action='store_true', help="show a live preview window")
    ap.add_argument('--no-save', action='store_true', help="don't write an output video")
    args = ap.parse_args()

    cfg = load_config()

    is_camera = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_camera else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 1:   # 0, NaN, or bogus
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = args.out
    if out_path is None and not is_camera:
        base, _ = os.path.splitext(args.source)
        out_path = f"{base}_annotated.mp4"
    writer = None
    if not args.no_save and out_path:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    analyzer = FaceAnalyzer(cfg['face_mesh'])
    objdet   = ObjectDetector(cfg['object_detector'])
    detector = StateDetector(cfg)

    frame_idx    = 0
    phone_frames = 0
    face_frames  = 0
    state_counts = {}
    yaws, pitches = [], []
    started      = time.time()

    print(f"Source: {args.source}  ({w}x{h} @ {fps:.1f}fps, {total or '?'} frames)")
    if writer:
        print(f"Writing: {out_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Video clock drives blink-rate / PERCLOS; live camera uses wall clock.
            vts_ms = int(frame_idx / fps * 1000) if not is_camera else int(time.time() * 1000)
            vts_s  = vts_ms / 1000.0

            features = analyzer.process(frame)
            objects  = objdet.process(frame, vts_ms)
            result   = detector.update(features, now=None if is_camera else vts_s)

            if any(o.is_distraction for o in objects):
                phone_frames += 1
            if features:
                face_frames += 1
                yaws.append(result.yaw)
                pitches.append(result.pitch)
            state_counts[result.state.value] = state_counts.get(result.state.value, 0) + 1

            frame = draw(frame, features, result, fps, objects=objects)

            if writer:
                writer.write(frame)
            if args.show:
                cv2.imshow(cfg['display']['window_title'], frame)
                if cv2.waitKey(1) & 0xFF == 27:   # Esc
                    break

            frame_idx += 1
            if total and frame_idx % 30 == 0:
                print(f"  {frame_idx}/{total} frames...", end='\r')
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    dur = time.time() - started
    n   = max(frame_idx, 1)
    print(f"\n=== Summary ({frame_idx} frames, {dur:.1f}s, {frame_idx/max(dur,1e-6):.1f} fps processing) ===")
    print(f"Face detected:   {face_frames}/{frame_idx} ({face_frames/n:.0%})")
    print(f"Total blinks:    {result.blink_count}")
    print(f"Phone visible:   {phone_frames}/{frame_idx} ({phone_frames/n:.0%})")
    if yaws:
        print(f"Yaw range:       {min(yaws):+.0f}..{max(yaws):+.0f}  "
              f"Pitch range: {min(pitches):+.0f}..{max(pitches):+.0f}")
    if state_counts:
        dist = "  ".join(f"{k} {v/n:.0%}" for k, v in
                         sorted(state_counts.items(), key=lambda kv: -kv[1]))
        print(f"State breakdown: {dist}")
    if writer:
        print(f"Annotated video: {out_path}")


if __name__ == '__main__':
    main()
