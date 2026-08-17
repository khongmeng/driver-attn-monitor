"""Build the combined Stage-④ training table from per-session feature CSVs.

Concatenates ``train/output/features/*.csv``, derives a 4-state label from the
DMD ground truth (see ``train/FEATURES.md``), writes one training table, and
prints a class-balance report.

State derivation (per face frame):
  drowsiness sessions (eye/blink/yawn GT):
    TIRED    if gt_yawn == 1
    DROWSY   elif GT-PERCLOS (rolling `drowsy_window` s of gt_eye_closed) >= drowsy_perclos
             (default 20% over 15 s — matches DMD's short acted microsleeps;
              the 60 s clinical window is too long for them, see docs/METHODOLOGY.md)
    FOCUSED  else (alert driving)
  distraction sessions (gaze/action GT):
    DISTRACTED if gt_gaze_off_road == 1 or gt_driver_action is a distraction
    FOCUSED    elif gt_looking_road == 1 and gt_driver_action == 'safe_drive'
    (else -> unlabeled, dropped: unclassified / standstill / ambiguous)

NO_FACE frames (has_face == 0) are excluded — that's the detector gate, not a
classifier target.

Also adds rolling-window features on top of the per-frame cascade output
(perclos_15s, yaw_std_5s, gaze_yaw_std_5s, eye_open_prob_mean_5s) — see
`_add_temporal_features`. These are computed from the model's own signals
(not GT), so they're available at runtime too.

Usage:
    python -m train.build_dataset
    python -m train.build_dataset --in train/output/features --out train/output/train_table.csv
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

# driver_actions that mean the driver is distracted (not safe driving)
DISTRACTION_ACTIONS = {
    "texting_left", "texting_right", "phonecall_left", "phonecall_right",
    "radio", "drinking", "hair_and_makeup", "reach_side", "reach_backseat",
    "talking_to_passenger", "change_gear",
}
# actions that don't cleanly map to a state -> drop
SKIP_ACTIONS = {"unclassified", "standstill_or_waiting"}

# features the Stage-④ classifier consumes
FEATURE_COLS = [
    "yaw", "pitch", "eye_open_prob", "perclos", "blink_rate",
    "gaze_yaw", "gaze_pitch",
    "perclos_15s", "yaw_std_5s", "gaze_yaw_std_5s", "eye_open_prob_mean_5s",
    "mar", "yawn_rate",   # §8.7: mouth aspect ratio + yawn rate (new cascade stage,
                          # gives TIRED an actual live-computable signal instead of
                          # being inferred indirectly from eye/head cues)
]


def _gt_perclos(df: pd.DataFrame, fps: float, window_s: float) -> pd.Series:
    """Rolling fraction of GT eyes-closed over the last `window_s`, per session."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby("session", sort=False):
        g = g.sort_values("frame")
        closed = (g["gt_eye_closed"] == 1).astype(float)
        ts = pd.to_timedelta(g["frame"].to_numpy() / fps, unit="s")
        closed.index = ts
        roll = closed.rolling(f"{int(window_s)}s").mean()
        out.loc[g.index] = roll.to_numpy()
    return out


def _rolling(df: pd.DataFrame, col: str, fps: float, window_s: float, stat: str) -> pd.Series:
    """Time-indexed rolling stat ('mean'|'std') of `col` over `window_s`, per
    session. Same pattern as `_gt_perclos` but generalized to any model-output
    column (not GT), so it doubles as a runtime-computable feature."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby("session", sort=False):
        g = g.sort_values("frame")
        s = g[col].copy()
        s.index = pd.to_timedelta(g["frame"].to_numpy() / fps, unit="s")
        roll = getattr(s.rolling(f"{int(window_s)}s"), stat)()
        out.loc[g.index] = roll.to_numpy()
    return out


def _add_temporal_features(df: pd.DataFrame, fps: float, std_window: float,
                            perclos_window: float) -> pd.DataFrame:
    """Add rolling-window features on top of the per-frame cascade output.
    Faster-reacting complements to the cascade's fixed 60s PERCLOS: a short
    PERCLOS at the same timescale used to label DROWSY from GT (see §7 in
    docs/METHODOLOGY.md), plus head/gaze volatility over a short window to
    distinguish sustained movement from single-frame noise."""
    df["perclos_15s"] = _rolling(df, "eye_closed", fps, perclos_window, "mean")
    df["yaw_std_5s"] = _rolling(df, "yaw", fps, std_window, "std")
    df["gaze_yaw_std_5s"] = _rolling(df, "gaze_yaw", fps, std_window, "std")
    df["eye_open_prob_mean_5s"] = _rolling(df, "eye_open_prob", fps, std_window, "mean")

    # early rows in each session have too few samples for a full window:
    # std is NaN for a single point (fill 0, i.e. "no observed variation yet"),
    # the mean should fall back to the instantaneous value rather than drop.
    for c in ("yaw_std_5s", "gaze_yaw_std_5s"):
        df[c] = df[c].fillna(0.0)
    df["eye_open_prob_mean_5s"] = df["eye_open_prob_mean_5s"].fillna(df["eye_open_prob"])
    return df


def _label_row(r) -> str | None:
    if r["task"] == "drowsiness":
        if r.get("gt_yawn") == 1:
            return "TIRED"
        if r.get("gt_perclos", 0) >= r["_drowsy_thresh"]:
            return "DROWSY"
        return "FOCUSED"
    if r["task"] == "distraction":
        act = r.get("gt_driver_action")
        if r.get("gt_gaze_off_road") == 1 or (isinstance(act, str) and act in DISTRACTION_ACTIONS):
            return "DISTRACTED"
        if r.get("gt_looking_road") == 1 and act == "safe_drive":
            return "FOCUSED"
        return None
    return None


def build(in_dir: str, fps: float, drowsy_thresh: float, drowsy_window: float,
          std_window: float, perclos_window: float) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(in_dir, "*_features.csv")))
    if not files:
        raise SystemExit(f"No *_features.csv in {in_dir}. Run extract_features first.")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    print(f"loaded {len(files)} session CSVs -> {len(df):,} rows")

    df = df[df["has_face"] == 1].copy()          # drop NO_FACE (detector gate)
    df["gt_perclos"] = _gt_perclos(df, fps, drowsy_window)
    df["_drowsy_thresh"] = drowsy_thresh
    df["state"] = df.apply(_label_row, axis=1)
    df = df.drop(columns="_drowsy_thresh")
    df = _add_temporal_features(df, fps, std_window, perclos_window)

    labeled = df.dropna(subset=["state"]).copy()
    # drop frames missing any classifier feature
    labeled = labeled.dropna(subset=FEATURE_COLS)
    return labeled


def main():
    ap = argparse.ArgumentParser(description="Build the Stage-④ training table.")
    ap.add_argument("--in", dest="in_dir", default="train/output/features")
    ap.add_argument("--out", default="train/output/train_table.csv")
    ap.add_argument("--fps", type=float, default=29.76, help="DMD face-camera fps")
    ap.add_argument("--drowsy-perclos", type=float, default=0.20,
                    help="PERCLOS threshold for DROWSY")
    ap.add_argument("--drowsy-window", type=float, default=15.0,
                    help="PERCLOS window (s) for DROWSY; 15s captures DMD's short "
                         "acted microsleeps better than the 60s clinical default")
    ap.add_argument("--std-window", type=float, default=5.0,
                    help="window (s) for yaw/gaze volatility + short eye-prob features")
    ap.add_argument("--perclos-window", type=float, default=15.0,
                    help="window (s) for the faster-reacting perclos_15s feature")
    args = ap.parse_args()

    df = build(args.in_dir, args.fps, args.drowsy_perclos, args.drowsy_window,
              args.std_window, args.perclos_window)

    keep = (["session", "task", "frame", "state"] + FEATURE_COLS +
            ["roll", "eye_closed", "blink", "blink_count", "det_score",
             "mouth_open", "yawn_active", "yawn_count"])
    keep = [c for c in keep if c in df.columns]
    df[keep].to_csv(args.out, index=False)

    print(f"\nwrote {len(df):,} labeled face frames -> {args.out}")
    print("\n=== class balance ===")
    total = len(df)
    for state, n in df["state"].value_counts().items():
        print(f"  {state:11s} {n:8,d}  ({n/total:5.1%})")
    print("\n=== by source ===")
    print(pd.crosstab(df["task"], df["state"]).to_string())
    print(f"\nsessions: {df['session'].nunique()}")


if __name__ == "__main__":
    main()
