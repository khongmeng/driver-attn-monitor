import time
from collections import deque
from dataclasses import dataclass
from enum import Enum


class DriverState(Enum):
    FOCUSED    = "FOCUSED"
    DROWSY     = "DROWSY"
    DISTRACTED = "DISTRACTED"
    TIRED      = "TIRED"      # planned: yawn / low-vigilance — not yet wired up
    NO_FACE    = "NO_FACE"


@dataclass
class StateResult:
    state:       DriverState
    ear:         float = 0.0
    yaw:         float = 0.0
    pitch:       float = 0.0
    perclos:     float = 0.0
    blink_count: int   = 0
    blink_rate:  float = 0.0   # blinks per minute


class StateDetector:
    def __init__(self, cfg: dict):
        self._ear_thresh     = cfg['ear']['threshold']
        self._ear_consec     = cfg['ear']['consecutive_frames']
        self._yaw_thresh     = cfg['head_pose']['yaw_threshold']
        self._pitch_thresh   = cfg['head_pose']['pitch_threshold']
        self._perclos_window = cfg['state']['perclos_window_sec']
        self._drowsy_perclos = cfg['state']['drowsy_perclos']

        self._closed_frames = 0
        self._blink_count   = 0
        self._start_time    = None   # set on first update (wall-clock or video clock)
        self._eye_log: deque = deque()   # (timestamp, eye_closed)

    def update(self, features: list, now: float = None) -> StateResult:
        # `now` lets video-file playback drive metrics off the video clock
        # instead of wall-clock time; live capture leaves it None.
        if now is None:
            now = time.time()
        if self._start_time is None:
            self._start_time = now

        if not features:
            return StateResult(state=DriverState.NO_FACE)

        f = features[0]

        # Blink detection
        eye_closed = f.ear < self._ear_thresh
        if eye_closed:
            self._closed_frames += 1
        else:
            if self._closed_frames >= self._ear_consec:
                self._blink_count += 1
            self._closed_frames = 0

        # PERCLOS over rolling window
        self._eye_log.append((now, eye_closed))
        cutoff = now - self._perclos_window
        while self._eye_log and self._eye_log[0][0] < cutoff:
            self._eye_log.popleft()
        perclos = sum(1 for _, c in self._eye_log if c) / max(len(self._eye_log), 1)

        elapsed_min = (now - self._start_time) / 60.0
        blink_rate  = self._blink_count / max(elapsed_min, 1e-6)

        # State decision — drowsy takes priority over distracted
        looking_away = abs(f.yaw) > self._yaw_thresh or abs(f.pitch) > self._pitch_thresh
        if perclos >= self._drowsy_perclos:
            state = DriverState.DROWSY
        elif looking_away:
            state = DriverState.DISTRACTED
        else:
            state = DriverState.FOCUSED

        return StateResult(
            state=state, ear=f.ear,
            yaw=f.yaw, pitch=f.pitch,
            perclos=perclos,
            blink_count=self._blink_count,
            blink_rate=blink_rate,
        )
