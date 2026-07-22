"""Train the Stage-④ driver-state classifier on the DMD feature table.

Reads ``train/output/train_table.csv``, splits **by driver** (no frame leakage),
trains a small MLP (or a gradient-boosted-tree baseline) on the feature set
with tunable class weighting, reports per-class precision/recall + confusion
matrix on held-out drivers (raw and temporally-smoothed), and exports to ONNX
(normalization baked in) for the Jetson runtime.

Three class modes (``--classes``):
  four    FOCUSED / DISTRACTED / DROWSY / TIRED   (default, the full taxonomy)
  three   FOCUSED / DISTRACTED / FATIGUED — collapses DROWSY+TIRED into one
          "FATIGUED" class (kept separate from DISTRACTED, unlike binary).
  binary  ATTENTIVE / INATTENTIVE — collapses DISTRACTED+DROWSY+TIRED into one
          "inattention" class (the standard umbrella term in the DMS literature).
          Turns the 46/48/2.8/2.7 imbalance into ~46/54 and gives a strong,
          nearly-balanced baseline to compare the 4-state model against.
          All modes use the identical driver split (same seed, same held-out
          drivers), so results are directly comparable.

Usage:
    python -m train.train_state                     # 4-state (unchanged)
    python -m train.train_state --classes three     # FOCUSED/DISTRACTED/FATIGUED
    python -m train.train_state --classes binary    # ATTENTIVE vs INATTENTIVE
    python -m train.train_state --epochs 40 --val-frac 0.2
    python -m train.train_state --smooth-window 15  # + temporal smoothing at eval
    python -m train.train_state --weight-power 0.5  # dampen inverse-freq weights
    python -m train.train_state --loss focal        # focal loss instead of weighted CE
    python -m train.train_state --model gbt         # gradient-boosted trees, not MLP
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             roc_auc_score)

from .build_dataset import FEATURE_COLS as FEATURES

STATES = ["FOCUSED", "DISTRACTED", "DROWSY", "TIRED"]   # 4-state class index order

# three-class regrouping: fatigue (drowsy + tired) kept apart from distraction
THREE_STATES = ["FOCUSED", "DISTRACTED", "FATIGUED"]
TO_THREE = {"FOCUSED": "FOCUSED", "DISTRACTED": "DISTRACTED",
            "DROWSY": "FATIGUED", "TIRED": "FATIGUED"}

# binary regrouping: everything that is not focused driving is "inattention"
BINARY_STATES = ["ATTENTIVE", "INATTENTIVE"]            # class index order
TO_BINARY = {"FOCUSED": "ATTENTIVE", "DISTRACTED": "INATTENTIVE",
             "DROWSY": "INATTENTIVE", "TIRED": "INATTENTIVE"}

CLASS_SETS = {"four": STATES, "three": THREE_STATES, "binary": BINARY_STATES}
TARGET_MAPS = {"four": None, "three": TO_THREE, "binary": TO_BINARY}


def driver_of(session: str) -> str:
    """DMD driver id = group_subject (e.g. gA_1) — same driver spans sessions."""
    p = str(session).split("_")
    return f"{p[0]}_{p[1]}" if len(p) >= 2 else str(session)


def split_by_driver(df: pd.DataFrame, val_frac: float, seed: int):
    """Hold out whole drivers. Fatigue drivers (DROWSY/TIRED come only from the
    drowsiness sessions) are split separately so val always contains them."""
    rng = np.random.default_rng(seed)
    fatigue = set(df.loc[df.state.isin(["DROWSY", "TIRED"]), "driver"].unique())
    others = set(df.driver.unique()) - fatigue
    val_drivers = set()
    for grp in (sorted(fatigue), sorted(others)):
        g = list(grp)
        rng.shuffle(g)
        k = max(1, round(len(g) * val_frac))
        val_drivers |= set(g[:k])
    val_mask = df.driver.isin(val_drivers)
    return ~val_mask, val_mask, sorted(val_drivers)


class StateMLP(nn.Module):
    def __init__(self, n_feat, n_class, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.net = nn.Sequential(
            nn.Linear(n_feat, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_class),
        )

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


class FocalLoss(nn.Module):
    """Multiclass focal loss on top of per-class weights: down-weights
    already-easy (high-confidence-correct) examples so the rare classes don't
    need as aggressive a class weight to get attention during training."""
    def __init__(self, weight, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        w_t = self.weight[target]
        loss = -w_t * (1 - p_t).pow(self.gamma) * logp_t
        return loss.mean()


def _smooth_eval(va: pd.DataFrame, proba: np.ndarray, n_classes: int, window: int):
    """Rolling-mean the per-class probabilities over `window` frames within
    each session (sorted by frame), then argmax — same scheme as run_live.py's
    live smoothing (softmax-prob deque -> mean -> argmax), applied offline to
    the held-out val set to measure the effect before touching the live demo."""
    prob_cols = [f"_p{i}" for i in range(n_classes)]
    d = va[["session", "frame", "y"]].reset_index(drop=True).copy()
    for i, c in enumerate(prob_cols):
        d[c] = proba[:, i]
    d = d.sort_values(["session", "frame"]).reset_index(drop=True)
    roll = d.groupby("session")[prob_cols].rolling(window, min_periods=1).mean()
    roll = roll.reset_index(level=0, drop=True)
    pred_smooth = roll.to_numpy().argmax(1)
    return d["y"].to_numpy(), pred_smooth


def _report(y_true, pred, classes, header):
    print(f"\n=== {header} ===")
    print(classification_report(y_true, pred, labels=list(range(len(classes))),
                                target_names=classes, digits=3, zero_division=0))
    print("confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, pred, labels=list(range(len(classes))))
    print("            " + "  ".join(f"{s[:5]:>6s}" for s in classes))
    for i, s in enumerate(classes):
        print(f"  {s[:10]:10s} " + "  ".join(f"{cm[i][j]:6d}" for j in range(len(classes))))
    print(f"\nmacro-F1: {f1_score(y_true, pred, average='macro'):.3f} | "
          f"accuracy: {(pred == y_true).mean():.3f}")


def main():
    ap = argparse.ArgumentParser(description="Train the Stage-④ state classifier.")
    ap.add_argument("--table", default="train/output/train_table.csv")
    ap.add_argument("--classes", choices=sorted(CLASS_SETS), default="four",
                    help="four = FOCUSED/DISTRACTED/DROWSY/TIRED; "
                         "three = FOCUSED/DISTRACTED/FATIGUED (DROWSY+TIRED merged); "
                         "binary = ATTENTIVE/INATTENTIVE (non-FOCUSED collapsed)")
    ap.add_argument("--out-dir", default=None,
                    help="default: models/state_classifier (four), "
                         "models/state_classifier/three, or .../binary")
    ap.add_argument("--model", choices=["mlp", "gbt"], default="mlp",
                    help="mlp = the small torch MLP (default, exports ONNX); "
                         "gbt = sklearn HistGradientBoostingClassifier, a model-family "
                         "comparison baseline (not exported, no ONNX path yet)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight-power", type=float, default=1.0,
                    help="exponent on the inverse-frequency class weights; "
                         "1.0 = current behavior, 0.0 = no weighting (uniform)")
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce",
                    help="ce = weighted cross-entropy (default, mlp only); "
                         "focal = focal loss with the same class weights (mlp only)")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--smooth-window", type=int, default=0,
                    help="if >0, also report temporally-smoothed eval: rolling-mean "
                         "class probabilities over this many frames per session, "
                         "then argmax (mirrors run_live.py's --smooth)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    classes = CLASS_SETS[args.classes]
    out_dir = args.out_dir or (
        "models/state_classifier" if args.classes == "four"
        else f"models/state_classifier/{args.classes}")

    df = pd.read_csv(args.table)
    df["driver"] = df.session.map(driver_of)
    df = df.dropna(subset=FEATURES + ["state"])
    df = df[df.state.isin(STATES)].copy()
    # target column: 4-state as-is, or collapsed per TARGET_MAPS. The split
    # below still uses the original 4-state labels, so the same drivers are
    # held out in every mode and results stay comparable.
    target_map = TARGET_MAPS[args.classes]
    df["target"] = df.state if target_map is None else df.state.map(target_map)
    lab = {s: i for i, s in enumerate(classes)}
    df["y"] = df.target.map(lab)

    tr_mask, va_mask, val_drivers = split_by_driver(df, args.val_frac, args.seed)
    tr, va = df[tr_mask], df[va_mask]
    print(f"classes: {classes}")
    print(f"drivers: {df.driver.nunique()} total | "
          f"train {tr.driver.nunique()} ({len(tr):,} frames) | "
          f"val {va.driver.nunique()} ({len(va):,} frames)")
    print(f"held-out val drivers: {', '.join(val_drivers)}")
    print("\ntrain class counts:", tr.target.value_counts().to_dict())
    print("val   class counts:", va.target.value_counts().to_dict())

    Xtr_np = tr[FEATURES].to_numpy(np.float32)
    ytr_np = tr.y.to_numpy()
    Xva_np = va[FEATURES].to_numpy(np.float32)
    yva = va.y.to_numpy()

    # class weights: inverse frequency, dampened by --weight-power
    counts = np.bincount(ytr_np, minlength=len(classes)).astype(np.float64)
    w = (counts.sum() / (len(classes) * np.clip(counts, 1, None))) ** args.weight_power
    print("\nclass weights:", {classes[i]: round(float(w[i]), 2) for i in range(len(classes))})

    if args.model == "gbt":
        sample_weight = w[ytr_np]
        gbt = HistGradientBoostingClassifier(max_iter=300, random_state=args.seed)
        print(f"\ntraining HistGradientBoostingClassifier …")
        gbt.fit(Xtr_np, ytr_np, sample_weight=sample_weight)
        proba = gbt.predict_proba(Xva_np)
        pred = proba.argmax(1)
    else:
        Xtr = torch.tensor(Xtr_np)
        ytr = torch.tensor(ytr_np, dtype=torch.long)
        Xva = torch.tensor(Xva_np).to(dev)
        mean = Xtr.mean(0)
        std = Xtr.std(0).clamp_min(1e-6)
        weights = torch.tensor(w, dtype=torch.float32).to(dev)

        model = StateMLP(len(FEATURES), len(classes), mean, std).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        lossfn = (nn.CrossEntropyLoss(weight=weights) if args.loss == "ce"
                  else FocalLoss(weights, gamma=args.focal_gamma))
        ds = torch.utils.data.TensorDataset(Xtr, ytr)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True)

        print(f"\ntraining on {dev} ({args.loss} loss) …")
        for ep in range(args.epochs):
            model.train()
            tot = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                loss = lossfn(model(xb), yb)
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
            if (ep + 1) % 10 == 0 or ep == 0:
                model.eval()
                with torch.no_grad():
                    ep_pred = model(Xva).argmax(1).cpu().numpy()
                mf1 = f1_score(yva, ep_pred, average="macro")
                print(f"  epoch {ep+1:2d}  train_loss {tot/len(tr):.4f}  val_macroF1 {mf1:.3f}")

        model.eval()
        with torch.no_grad():
            proba = torch.softmax(model(Xva), 1).cpu().numpy()
        pred = proba.argmax(1)

    _report(yva, pred, classes, "validation (held-out drivers, raw per-frame)")

    if args.smooth_window > 0:
        y_smooth, pred_smooth = _smooth_eval(va, proba, len(classes), args.smooth_window)
        _report(y_smooth, pred_smooth, classes,
                f"validation, temporally smoothed (window={args.smooth_window} frames)")

    if args.classes == "binary":
        # threshold-free separability + which original states get caught
        prob = proba[:, 1]
        print(f"\nROC-AUC (INATTENTIVE): {roc_auc_score(yva, prob):.3f}")
        print("INATTENTIVE recall by original 4-state label:")
        va_state = va.state.to_numpy()
        for s in ("DISTRACTED", "DROWSY", "TIRED"):
            m = va_state == s
            if m.any():
                print(f"  {s:10s} {(pred[m] == 1).mean():6.1%}  of {m.sum():,} frames")
        m = va_state == "FOCUSED"
        if m.any():
            print(f"  FOCUSED kept ATTENTIVE: {(pred[m] == 0).mean():.1%} of {m.sum():,} frames")
    elif args.classes == "three":
        # which original DROWSY/TIRED frames actually get caught as FATIGUED
        fatigued_idx = classes.index("FATIGUED")
        va_state = va.state.to_numpy()
        print("\nFATIGUED recall by original label:")
        for s in ("DROWSY", "TIRED"):
            m = va_state == s
            if m.any():
                print(f"  {s:10s} {(pred[m] == fatigued_idx).mean():6.1%}  of {m.sum():,} frames")

    if args.model != "mlp":
        print(f"\n{args.model} model not exported (comparison baseline only; "
              f"use --model mlp for a deployable ONNX checkpoint).")
        return

    # export
    os.makedirs(out_dir, exist_ok=True)
    pt = os.path.join(out_dir, "state_mlp.pt")
    onnx = os.path.join(out_dir, "state_mlp.onnx")
    torch.save({"state_dict": model.state_dict(), "features": FEATURES,
                "states": classes}, pt)
    model_cpu = model.to("cpu").eval()
    dummy = torch.zeros(1, len(FEATURES))
    torch.onnx.export(model_cpu, dummy, onnx, input_names=["features"],
                      output_names=["logits"],
                      dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=13)
    print(f"\nsaved {pt}\nsaved {onnx}  (input: {len(FEATURES)} features, order = {FEATURES})")


if __name__ == "__main__":
    main()
