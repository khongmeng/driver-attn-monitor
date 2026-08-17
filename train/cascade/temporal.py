"""Temporal layer — PERCLOS, blink detection, blink rate, yawn detection.

Fed one frame at a time (eye_closed decision from the stage-③ model, plus the
video timestamp), it maintains:
  * a rolling window of eye-closed flags -> PERCLOS (% of last N seconds closed)
  * a closed-run counter -> blink events (closed for >= `blink_min_frames`,
    then reopened)
  * a mouth-open run timer -> yawn events (mouth open for >= `yawn_min_duration_sec`
    continuously, the same run-length idea as blink but duration- rather than
    frame-count-gated, since yawns last ~1-2s vs. a blink's few hundred ms)

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
        blink_min_frames: int = 1,        # consecutive closed frames to count a blink.
                                           # §6.1 found -85%/+41% per-session over/under-
                                           # counting at 1 and hypothesized a single noisy
                                           # dip was the cause; §8.7 tested blink_min_frames=2
                                           # and found it makes the AGGREGATE worse (0.61x vs
                                           # 1.01x, a systematic undercount, not a fix) — real
                                           # blinks are often only 1-2 samples wide at this
                                           # stride-2 extraction rate, so requiring 2
                                           # consecutive samples filters out genuine blinks
                                           # more often than it filters flicker. Reverted to 1;
                                           # the per-session swings are a real unresolved
                                           # limitation, not fixed by this debounce.
        perclos_window_sec: float = 60.0,
        blink_rate_min_elapsed_sec: float = 5.0,   # suppress blink_rate before this much
                                                    # session time has elapsed — with under
                                                    # a few seconds on the clock, one early
                                                    # blink divided by near-zero elapsed time
                                                    # spikes to hundreds/min (e.g. 1 blink at
                                                    # t=0.067s -> 892/min); NaN until then
                                                    # instead of a misleadingly huge number.
        yawn_min_duration_sec: float = 1.0,        # continuous mouth-open duration to
                                                    # count as a yawn (vs. a blink's few
                                                    # frames — yawns are much longer)
    ):
        self.eye_thresh = eye_closed_thresh
        self.blink_min = blink_min_frames
        self.window = perclos_window_sec
        self.blink_rate_min_elapsed = blink_rate_min_elapsed_sec
        self.yawn_min_duration = yawn_min_duration_sec

        self._closed_run = 0
        self.blink_count = 0
        self._start_ts = None
        self._log: deque = deque()   # (timestamp, closed_bool)

        self._yawn_open_since = None
        self.yawn_count = 0

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

        elapsed_sec = ts - self._start_ts
        if elapsed_sec < self.blink_rate_min_elapsed:
            blink_rate = float("nan")
        else:
            blink_rate = self.blink_count / (elapsed_sec / 60.0)
        return int(closed), blink_now, perclos, blink_rate

    def update_mouth(self, mouth_open: bool, ts: float):
        """Returns (yawn_active:int, yawn_count:int, yawn_rate:float). Call once
        per frame alongside `update()` — relies on `update()` having already
        set `_start_ts` this session (true from frame 0 since `update()` is
        called unconditionally in the pipeline, even with a NaN eye reading)."""
        yawn_active = 0
        if mouth_open:
            if self._yawn_open_since is None:
                self._yawn_open_since = ts
            if ts - self._yawn_open_since >= self.yawn_min_duration:
                yawn_active = 1
        else:
            if self._yawn_open_since is not None:
                if ts - self._yawn_open_since >= self.yawn_min_duration:
                    self.yawn_count += 1
                self._yawn_open_since = None

        start = self._start_ts if self._start_ts is not None else ts
        elapsed_sec = ts - start
        if elapsed_sec < self.blink_rate_min_elapsed:
            yawn_rate = float("nan")
        else:
            yawn_rate = self.yawn_count / (elapsed_sec / 60.0)
        return yawn_active, self.yawn_count, yawn_rate
