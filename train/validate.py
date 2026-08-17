"""Validate extracted cascade features against DMD ground truth.

Answers "does the cascade work on DMD?" before we invest in training. Reads the
per-session feature CSVs produced by ``extract_features.py`` and reports, per the
labels DMD actually provides (eyes-state / blinks / yawn):

  * face-detection coverage (has_face %)
  * eye-state agreement: model open/closed vs GT eyes_state, with the best
    decision threshold on eye_open_prob
  * blink counting: cascade blink events vs GT blink intervals
  * head-pose sanity: yaw/pitch ranges (no GT in the drowsiness set)

Usage:
    python -m train.validate train/output/features
    python -m train.validate train/output/features/foo_features.csv
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import List

import numpy as np
import pandas as pd


def _load(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if os.path.isdir(p):
            frames += [pd.read_csv(f) for f in glob.glob(os.path.join(p, "*_features.csv"))]
        elif p.endswith(".csv"):
            frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit("No *_features.csv found.")
    return pd.concat(frames, ignore_index=True)


def _count_intervals(series: pd.Series) -> int:
    """Number of 0->1 rising edges in a 0/1 series (i.e. distinct events)."""
    s = series.fillna(0).astype(int).to_numpy()
    return int(np.sum((s[1:] == 1) & (s[:-1] == 0)) + (s[:1] == 1).sum())


def _best_threshold(open_prob: np.ndarray, gt_closed: np.ndarray):
    """Pick the eye_open_prob threshold that maximizes closed-vs-open accuracy."""
    best_t, best_acc = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 19):
        pred_closed = (open_prob < t).astype(int)
        acc = float(np.mean(pred_closed == gt_closed))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


def _best_high_threshold(score: np.ndarray, gt_pos: np.ndarray):
    """Pick threshold where score>t best predicts gt_pos==1 (e.g. |yaw| -> off-road)."""
    best_t, best_acc = 0.0, 0.0
    lo, hi = float(np.percentile(score, 5)), float(np.percentile(score, 95))
    for t in np.linspace(lo, hi, 25):
        acc = float(np.mean((score > t).astype(int) == gt_pos))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


def _best_combined(a: np.ndarray, b: np.ndarray, gt_pos: np.ndarray):
    """Best (ta, tb) for the rule (a>ta OR b>tb) predicting gt_pos==1."""
    best, best_acc = (0.0, 0.0), 0.0
    a_grid = np.linspace(float(np.percentile(a, 50)), float(np.percentile(a, 97)), 12)
    b_grid = np.linspace(float(np.percentile(b, 50)), float(np.percentile(b, 97)), 12)
    for ta in a_grid:
        for tb in b_grid:
            pred = ((a > ta) | (b > tb)).astype(int)
            acc = float(np.mean(pred == gt_pos))
            if acc > best_acc:
                best_acc, best = acc, (float(ta), float(tb))
    return best, best_acc


def report(df: pd.DataFrame):
    n = len(df)
    print("=" * 64)
    print(f"VALIDATION REPORT — {n} frames across {df['session'].nunique()} session(s)")
    print("=" * 64)

    # --- face coverage ---
    face = df["has_face"].mean()
    print(f"\n[① face]  detected in {face:.1%} of frames")
    occ = df.get("gt_occluded")
    if occ is not None and occ.notna().any():
        print(f"          (GT occlusion present in {occ.fillna(0).mean():.1%} of frames)")

    faces = df[df["has_face"] == 1]

    # --- head pose sanity (no GT in drowsiness set) ---
    if "yaw" in faces and faces["yaw"].notna().any():
        y, p = faces["yaw"].dropna(), faces["pitch"].dropna()
        print(f"\n[② head pose]  yaw {y.min():+.0f}..{y.max():+.0f} (mean {y.mean():+.1f}), "
              f"pitch {p.min():+.0f}..{p.max():+.0f} (mean {p.mean():+.1f})")
    else:
        print("\n[② head pose]  no values (stage disabled?)")

    # --- eye state vs GT ---
    print("\n[③ eye state]")
    eye = faces.dropna(subset=["eye_open_prob"])
    eye = eye[eye["gt_eye_closed"].isin([0, 1])]
    if len(eye) == 0:
        print("    no overlapping frames with eye model output + GT open/close")
    else:
        gt_closed = eye["gt_eye_closed"].astype(int).to_numpy()
        pred_closed = eye["eye_closed"].astype(int).to_numpy()
        acc = float(np.mean(pred_closed == gt_closed))
        # confusion
        tp = int(np.sum((pred_closed == 1) & (gt_closed == 1)))
        tn = int(np.sum((pred_closed == 0) & (gt_closed == 0)))
        fp = int(np.sum((pred_closed == 1) & (gt_closed == 0)))
        fn = int(np.sum((pred_closed == 0) & (gt_closed == 1)))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        bt, bacc = _best_threshold(eye["eye_open_prob"].to_numpy(), gt_closed)
        print(f"    frames compared: {len(eye)}  (GT closed={gt_closed.sum()}, open={(gt_closed==0).sum()})")
        print(f"    accuracy @ current thresh : {acc:.1%}")
        print(f"    closed precision / recall : {prec:.1%} / {rec:.1%}")
        print(f"    confusion  TP={tp} TN={tn} FP={fp} FN={fn}")
        print(f"    best threshold on open_prob: {bt:.2f}  -> acc {bacc:.1%}")

    # --- yawn (MAR) vs GT ---
    print("\n[mouth / yawn]")
    mouth = faces.dropna(subset=["mar"])
    mouth = mouth[mouth["gt_yawn"].isin([0, 1])]
    if len(mouth) == 0:
        print("    no overlapping frames with mouth model output + GT yawn")
    else:
        gt_yawn = mouth["gt_yawn"].astype(int).to_numpy()
        mar = mouth["mar"].to_numpy()
        pred_open = mouth["mouth_open"].astype(int).to_numpy()
        acc = float(np.mean(pred_open == gt_yawn))
        tp = int(np.sum((pred_open == 1) & (gt_yawn == 1)))
        tn = int(np.sum((pred_open == 0) & (gt_yawn == 0)))
        fp = int(np.sum((pred_open == 1) & (gt_yawn == 0)))
        fn = int(np.sum((pred_open == 0) & (gt_yawn == 1)))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        bt, bacc = _best_high_threshold(mar, gt_yawn)
        print(f"    frames compared: {len(mouth)}  (GT yawn={gt_yawn.sum()}, not={(gt_yawn==0).sum()})")
        print(f"    MAR yawn>open_thresh accuracy: {acc:.1%}")
        print(f"    yawn precision / recall      : {prec:.1%} / {rec:.1%}")
        print(f"    confusion  TP={tp} TN={tn} FP={fp} FN={fn}")
        not_yawn_mar = f"{mar[gt_yawn==0].mean():.2f}" if (gt_yawn == 0).any() else "n/a"
        yawn_mar = f"{mar[gt_yawn==1].mean():.2f}" if (gt_yawn == 1).any() else "n/a"
        print(f"    best threshold on MAR: {bt:.2f}  -> acc {bacc:.1%}  "
              f"(not-yawning {not_yawn_mar} vs yawning {yawn_mar})")

    print("\n[yawn count vs GT]")
    per_sess = []
    for name, g in df.groupby("session"):
        if "gt_yawn" not in g or g["gt_yawn"].isna().all():
            continue
        pred_yawns = int(g["yawn_count"].max()) if "yawn_count" in g else 0
        gt_yawns = _count_intervals(g["gt_yawn"])
        per_sess.append((name, pred_yawns, gt_yawns))
    if per_sess:
        tot_pred = sum(p for _, p, _ in per_sess)
        tot_gt = sum(gt for _, _, gt in per_sess)
        print(f"    cascade yawn events: {tot_pred}   GT yawn intervals: {tot_gt}")
        for name, p, gt in per_sess[:10]:
            print(f"      {name[:40]:40s}  cascade={p:4d}  GT={gt:4d}")
    else:
        print("    no drowsiness-set sessions with GT yawn labels found")

    # --- looking-away vs GT (distraction set: gaze_on_road) ---
    if "gt_gaze_off_road" in df.columns and df["gt_gaze_off_road"].notna().any():
        print("\n[DISTRACTED — head pose / gaze vs gaze_on_road GT]")
        g = faces.dropna(subset=["yaw", "pitch"])
        g = g[g["gt_gaze_off_road"].isin([0, 1])]
        if len(g):
            off = g["gt_gaze_off_road"].astype(int).to_numpy()
            ay = g["yaw"].abs().to_numpy()
            ap = g["pitch"].abs().to_numpy()
            look_m = (off == 0)
            away_m = (off == 1)
            print(f"    frames: looking_road={look_m.sum()}  not_looking_road={away_m.sum()}")
            print(f"    |yaw|   looking {ay[look_m].mean():5.1f}  vs not-looking {ay[away_m].mean():5.1f}")
            print(f"    |pitch| looking {ap[look_m].mean():5.1f}  vs not-looking {ap[away_m].mean():5.1f}")
            # best single-threshold accuracy using |yaw|, then |yaw| OR |pitch|
            ty, acc_y = _best_high_threshold(ay, off)
            (tyc, tpc), acc_c = _best_combined(ay, ap, off)
            print(f"    rule |yaw|>{ty:.0f}            -> acc {acc_y:.1%}")
            print(f"    rule |yaw|>{tyc:.0f} OR |pitch|>{tpc:.0f} -> acc {acc_c:.1%}")
            if "gaze_yaw" in g.columns and g["gaze_yaw"].notna().any():
                gg = g.dropna(subset=["gaze_yaw", "gaze_pitch"])
                go = gg["gt_gaze_off_road"].astype(int).to_numpy()
                gmag = np.sqrt(gg["gaze_yaw"].to_numpy() ** 2 + gg["gaze_pitch"].to_numpy() ** 2)
                tg, acc_g = _best_high_threshold(gmag, go)
                print(f"    gaze model |gaze|>{tg:.0f}     -> acc {acc_g:.1%}  "
                      f"(looking {gmag[go == 0].mean():.1f} vs away {gmag[go == 1].mean():.1f})")
                # head pose + gaze combined — the payoff (gaze catches eyes-only glances)
                hy = gg["yaw"].abs().to_numpy()
                (th, tgc), acc_hg = _best_combined(hy, gmag, go)
                print(f"    head+gaze |yaw|>{th:.0f} OR |gaze|>{tgc:.0f} -> acc {acc_hg:.1%}")
        else:
            print("    no frames with head pose + gaze_on_road GT")

    # --- blink counting vs GT ---
    print("\n[blink]")
    per_sess = []
    for name, g in df.groupby("session"):
        pred_blinks = int(g["blink"].fillna(0).sum())
        gt_blinks = _count_intervals(g["gt_blink"]) if "gt_blink" in g else 0
        per_sess.append((name, pred_blinks, gt_blinks))
    tot_pred = sum(p for _, p, _ in per_sess)
    tot_gt = sum(gt for _, _, gt in per_sess)
    print(f"    cascade blink events: {tot_pred}   GT blink intervals: {tot_gt}")
    for name, p, gt in per_sess[:10]:
        print(f"      {name[:40]:40s}  cascade={p:4d}  GT={gt:4d}")

    print("\n" + "=" * 64)


def main():
    ap = argparse.ArgumentParser(description="Validate cascade features vs DMD GT.")
    ap.add_argument("paths", nargs="+", help="feature CSV(s) or a directory of them")
    args = ap.parse_args()
    df = _load(args.paths)
    report(df)


if __name__ == "__main__":
    main()
