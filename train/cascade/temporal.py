"""Temporal layer — PERCLOS, blink detection, blink rate.

Fed one frame at a time (eye_closed decision from the stage-③ model, plus the
video timestamp), it maintains:
  * a rolling window of eye-closed flags -> PERCLOS (% of last N seconds closed)
  * a closed-run counter -> blink events (closed for >= `blink_min_frames`,
    then reopened)

This is the same logic the Jetson `state_detector.py` uses, but driven by the
eye-state *model* output instead of an EAR threshold, and clock-driven by the
video timestamp so offline extraction is deterministic.
"""
from __future__ import annotations

from collections import deque


class TemporalAggregator:
    def __init__(
        self,
        eye_closed_thresh: float = 0.5,   # eye_open_prob below this = closed
        blink_min_frames: int = 1,        # consecutive closed frames to count a blink
        perclos_window_sec: float = 60.0,
    ):
        self.eye_thresh = eye_closed_thresh
        self.blink_min = blink_min_frames
        self.window = perclos_window_sec

        self._closed_run = 0
        self.blink_count = 0
        self._start_ts = None
        self._log: deque = deque()   # (timestamp, closed_bool)

    def is_closed(self, eye_open_prob: float) -> bool:
        # NaN (no reading) is treated as "not closed" so it can't fake DROWSY.
        return eye_open_prob == eye_open_prob and eye_open_prob < self.eye_thresh

    def update(self, eye_open_prob: float, ts: float):
        """Returns (eye_closed:int, blink_this_frame:int, perclos:float, blink_rate:float)."""
        if self._start_ts is None:
            self._start_ts = ts

        closed = self.is_closed(eye_open_prob)
        blink_now = 0
        if closed:
            self._closed_run += 1
        else:
            if self._closed_run >= self.blink_min:
                self.blink_count += 1
                blink_now = 1
            self._closed_run = 0

        self._log.append((ts, closed))
        cutoff = ts - self.window
        while self._log and self._log[0][0] < cutoff:
            self._log.popleft()
        perclos = sum(1 for _, c in self._log if c) / max(len(self._log), 1)

        elapsed_min = max((ts - self._start_ts) / 60.0, 1e-6)
        blink_rate = self.blink_count / elapsed_min
        return int(closed), blink_now, perclos, blink_rate
