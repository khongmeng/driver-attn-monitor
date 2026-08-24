"""Extract a handful of candidate frames from an existing annotated demo
video, to pick one as the paper's qualitative "system in action" figure.

Usage:
    python docs/paper/extract_frame.py <video> <out_prefix> <t1> <t2> ...
"""
import sys

import cv2

video, out_prefix = sys.argv[1], sys.argv[2]
times = [float(t) for t in sys.argv[3:]]

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
for t in times:
    frame_idx = int(t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"failed at t={t}s")
        continue
    out = f"{out_prefix}_{t:.0f}s.png"
    cv2.imwrite(out, frame)
    print(f"saved {out}")
cap.release()
