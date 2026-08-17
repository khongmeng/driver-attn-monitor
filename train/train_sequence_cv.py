"""5-fold driver cross-validation ensemble of the bidirectional GatedFatigueGRU
(docs/METHODOLOGY.md §16) -- the final push before the paper deadline, with
the Jetson/live-streaming constraint explicitly dropped so the architecture
is free to be non-causal.

Every experiment before this one used ONE fixed 10-train/4-held-out driver
split (seed=42) -- the 4 held-out drivers (gA_1, gC_14, gE_28, gZ_37) were
used both to pick each model's best checkpoint AND to report its final
number. That's the standard setup, but with only 10 training drivers it
means 8/10 of the training pool never gets to be "the val set" for any
model, and no single model's training ever sees more than 8 effective
drivers' worth of signal after further internal splitting is needed for
early stopping.

This script instead:
  1. Splits off the SAME true held-out 4 drivers as every other experiment
     (same split_by_driver call, same seed) -- untouched during all training,
     used exactly once at the end to score the final ensemble.
  2. Splits the remaining 10 training drivers into K folds (default 5, 2
     drivers/fold). For each fold, trains an independent bidirectional
     GatedFatigueGRU on the other 8 drivers, using the fold's 2 held-back
     drivers as ITS OWN internal validation set for checkpoint selection only.
  3. Ensembles all K folds' best checkpoints (softmax-probability averaging,
     the same mechanism as eval_ensemble.py / run_live.py's
     EnsembleClassifier) and reports ONE final number on the true held-out 4.

Net effect: every one of the 10 training drivers contributes to some fold's
training set AND acts as another fold's validation set, and the final
ensemble combines 5 models trained on different 8-driver subsets -- more
effective use of a fixed 14-driver pool than a single split, via the
variance-reduction the project's existing ensemble (§8.14/§8.16, macro-F1
0.810) already demonstrated is real for this exact family of models.

Usage:
    python -m train.train_sequence_cv
    python -m train.train_sequence_cv --k-folds 5 --epochs 200
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .train_sequence import (EYE_YAWN_FEATURES, GatedFatigueGRU, build_chunks,
                             full_sequence_eval)
from .train_state import (CLASS_SETS, FEATURES, STATES, TARGET_MAPS, FocalLoss,
                          _report, _smooth_eval, driver_of, split_by_driver)


def make_folds(drivers: list, k: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    d = list(drivers)
    rng.shuffle(d)
    return [d[i::k] for i in range(k)]


def windowed_sequence_eval(model, df: pd.DataFrame, feature_cols, dev: str, window_len: int) -> np.ndarray:
    """Fix for the §16 attempt-1/2 diagnosis: full_sequence_eval scores each
    held-out session in ONE forward pass (thousands of frames), but a
    bidirectional model was only ever trained on window_len-frame chunks
    (600, ~40s) -- sessions average ~3,300 frames (up to 7,508), so the
    backward-direction hidden state at eval time has to summarize a sequence
    length it never saw in training. This instead scores each session in
    the SAME non-overlapping window_len windows used for training -- exposure
    length now matches between train and eval. Each window is an
    independent forward pass (h=None); there's no clean "carry state between
    windows" semantic for a bidirectional GRU anyway (the backward pass
    would need to know the future beyond the window regardless)."""
    model.eval()
    proba_by_idx = {}
    with torch.no_grad():
        for session, g in df.groupby("session", sort=False):
            g = g.sort_values("frame")
            X_full = g[feature_cols].to_numpy(np.float32)
            idx_arr = g.index.to_numpy()
            T = len(X_full)
            for start in range(0, T, window_len):
                end = min(start + window_len, T)
                Xw = torch.tensor(X_full[start:end]).unsqueeze(0).to(dev)
                logits, _ = model(Xw)
                p = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                for idx, row in zip(idx_arr[start:end], p):
                    proba_by_idx[idx] = row
    return np.stack([proba_by_idx[idx] for idx in df.index])


def train_one_fold(fold_idx, fold_tr, fold_va, feat_cols, classes, args, dev, eval_fn):
    Xc, yc = build_chunks(fold_tr, feat_cols, args.chunk_len, args.chunk_stride)
    Xc_t = torch.tensor(Xc)
    yc_t = torch.tensor(yc, dtype=torch.long)
    mean = Xc_t.reshape(-1, Xc_t.shape[-1]).mean(0)
    std = Xc_t.reshape(-1, Xc_t.shape[-1]).std(0).clamp_min(1e-6)

    counts = np.bincount(yc.reshape(-1), minlength=len(classes)).astype(np.float64)
    w = (counts.sum() / (len(classes) * np.clip(counts, 1, None))) ** args.weight_power

    sampler = None
    if args.oversample > 0 and "FATIGUED" in classes:
        fatigue_idx = classes.index("FATIGUED")
        frac = (yc == fatigue_idx).mean(axis=1)
        sample_w = 1.0 + args.oversample * frac
        sampler = WeightedRandomSampler(torch.tensor(sample_w), num_samples=len(sample_w), replacement=True)

    ds = TensorDataset(Xc_t, yc_t)
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, shuffle=(sampler is None))

    weights = torch.tensor(w, dtype=torch.float32).to(dev)
    model = GatedFatigueGRU(feat_cols, EYE_YAWN_FEATURES, mean, std,
                            hidden=args.hidden, layers=args.layers,
                            bidirectional=not args.no_bidirectional).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossfn = (nn.CrossEntropyLoss(weight=weights) if args.loss == "ce"
             else FocalLoss(weights, gamma=args.focal_gamma))

    print(f"\n--- fold {fold_idx+1}/{args.k_folds}: "
          f"{fold_tr.driver.nunique()} train drivers ({len(fold_tr):,} frames) | "
          f"{fold_va.driver.nunique()} internal-val drivers "
          f"({sorted(fold_va.driver.unique())}, {len(fold_va):,} frames) ---")
    print(f"  {len(Xc):,} chunks | class weights: "
          f"{ {classes[i]: round(float(w[i]), 2) for i in range(len(classes))} }")

    best_mf1, best_state, best_epoch = -1.0, None, -1
    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = lossfn(logits.reshape(-1, len(classes)), yb.reshape(-1))
            loss.backward()
            opt.step()
            tot += loss.item() * xb.numel()
            n += xb.numel()
        if (ep + 1) % args.eval_every == 0 or ep == 0 or ep == args.epochs - 1:
            proba = eval_fn(model, fold_va, feat_cols, dev)
            yva_ep = fold_va.y.to_numpy()
            pred_ep = proba.argmax(1)
            mf1 = f1_score(yva_ep, pred_ep, average="macro")
            flag = ""
            if mf1 > best_mf1:
                best_mf1, best_epoch = mf1, ep + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                flag = "  (best)"
            print(f"    epoch {ep+1:3d}  train_loss {tot/n:.4f}  internal_val_macroF1 {mf1:.3f}{flag}")

    print(f"  fold {fold_idx+1} restoring best checkpoint: epoch {best_epoch} "
          f"(internal_val_macroF1 {best_mf1:.3f})")
    model.load_state_dict(best_state)
    return model, mean, std


def main():
    ap = argparse.ArgumentParser(description="K-fold driver-CV ensemble of the bidirectional GatedFatigueGRU.")
    ap.add_argument("--table", default="train/output/train_table.csv")
    ap.add_argument("--out-dir", default="models/state_classifier/three_gru_gated_bidir_cv5")
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--cv-seed", type=int, default=7, help="seed for assigning the 10 training drivers to folds")
    ap.add_argument("--chunk-len", type=int, default=600)
    ap.add_argument("--chunk-stride", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2, help="the TRUE held-out split (same as every other experiment)")
    ap.add_argument("--seed", type=int, default=42, help="the TRUE held-out split's seed (same as every other experiment)")
    ap.add_argument("--weight-power", type=float, default=0.3)
    ap.add_argument("--loss", choices=["ce", "focal"], default="focal")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--oversample", type=float, default=5.0)
    ap.add_argument("--smooth-window", type=int, default=30)
    ap.add_argument("--no-bidirectional", action="store_true",
                    help="ablation: use the causal (forward-only) GRU instead -- lets you isolate "
                         "how much of the result is the CV ensembling vs. bidirectionality")
    ap.add_argument("--windowed-eval", action="store_true",
                    help="score held-out sessions in non-overlapping chunk-len windows instead of "
                         "one whole-session forward pass -- fixes the train/eval sequence-length "
                         "mismatch diagnosed in docs/METHODOLOGY.md sect 16 attempts 1-2 (bidirectional "
                         "models trained on 600-frame chunks but evaluated on full ~3,300-frame-average "
                         "sessions in one pass). Used for BOTH internal-val checkpoint selection and "
                         "the final true-test score, so selection and reporting stay consistent.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    classes = CLASS_SETS["three"]
    feat_cols = FEATURES

    df = pd.read_csv(args.table)
    df["driver"] = df.session.map(driver_of)
    df = df.dropna(subset=feat_cols + ["state"])
    df = df[df.state.isin(STATES)].copy()
    target_map = TARGET_MAPS["three"]
    df["target"] = df.state if target_map is None else df.state.map(target_map)
    lab = {s: i for i, s in enumerate(classes)}
    df["y"] = df.target.map(lab)

    tr_mask, va_mask, held_out_drivers = split_by_driver(df, args.val_frac, args.seed)
    tr_pool = df[tr_mask].copy()
    va_true = df[va_mask].copy()
    train_drivers = sorted(tr_pool.driver.unique())
    print(f"classes: {classes}")
    print(f"TRUE held-out test drivers (never used for training or checkpoint selection): "
          f"{held_out_drivers}")
    print(f"training-pool drivers ({len(train_drivers)}): {train_drivers}")

    if args.windowed_eval:
        eval_fn = lambda model, df, feat_cols, dev: windowed_sequence_eval(
            model, df, feat_cols, dev, args.chunk_len)
        print(f"\nusing WINDOWED eval (window={args.chunk_len} frames, matches training chunk "
              f"length) for both internal-val checkpoint selection and the final test score")
    else:
        eval_fn = full_sequence_eval

    folds = make_folds(train_drivers, args.k_folds, args.cv_seed)
    print(f"\n{args.k_folds}-fold split of the training pool (cv-seed={args.cv_seed}):")
    for i, f in enumerate(folds):
        print(f"  fold {i+1}: {f}")

    fold_models = []
    fold_probas_on_test = []
    for i, fold_drivers in enumerate(folds):
        fold_tr = tr_pool[~tr_pool.driver.isin(fold_drivers)]
        fold_va = tr_pool[tr_pool.driver.isin(fold_drivers)]
        model, mean, std = train_one_fold(i, fold_tr, fold_va, feat_cols, classes, args, dev, eval_fn)

        fold_out_dir = os.path.join(args.out_dir, f"fold{i}")
        os.makedirs(fold_out_dir, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "features": feat_cols, "states": classes,
                   "model": "gru", "hidden": args.hidden, "layers": args.layers,
                   "gated_fatigue": True, "fatigue_features": EYE_YAWN_FEATURES,
                   "bidirectional": not args.no_bidirectional,
                   "fold": i, "fold_val_drivers": fold_drivers},
                  os.path.join(fold_out_dir, "state_gru.pt"))

        proba_test = eval_fn(model, va_true, feat_cols, dev)
        mf1_test = f1_score(va_true.y.to_numpy(), proba_test.argmax(1), average="macro")
        print(f"  fold {i+1} solo performance on the TRUE held-out test drivers: macro-F1 {mf1_test:.3f}")
        fold_models.append(model)
        fold_probas_on_test.append(proba_test)

    ensemble_proba = np.mean(fold_probas_on_test, axis=0)
    yva = va_true.y.to_numpy()
    pred = ensemble_proba.argmax(1)

    print(f"\n{'='*70}\nFINAL: {args.k_folds}-fold ensemble on the TRUE held-out test drivers "
          f"({held_out_drivers})\n{'='*70}")
    _report(yva, pred, classes, f"{args.k_folds}-fold ensemble (raw per-frame)")

    if args.smooth_window > 0:
        y_smooth, pred_smooth = _smooth_eval(va_true, ensemble_proba, len(classes), args.smooth_window)
        _report(y_smooth, pred_smooth, classes,
                f"{args.k_folds}-fold ensemble, temporally smoothed (window={args.smooth_window} frames)")
        pred_report = pred_smooth
    else:
        pred_report = pred

    fatigued_idx = classes.index("FATIGUED")
    va_state = va_true.state.to_numpy()
    print("\nFATIGUED recall by original label:")
    for s in ("DROWSY", "TIRED"):
        m = va_state == s
        if m.any():
            print(f"  {s:10s} {(pred_report[m] == fatigued_idx).mean():6.1%}  of {m.sum():,} frames")

    np.save(os.path.join(args.out_dir, "ensemble_test_proba.npy"), ensemble_proba)
    print(f"\nsaved fold checkpoints + ensemble proba under {args.out_dir}/")


if __name__ == "__main__":
    main()
