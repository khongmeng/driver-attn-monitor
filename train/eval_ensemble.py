"""Offline evaluation of a multi-checkpoint GRU ensemble (softmax-probability
averaging, same mechanism as run_live.py's EnsembleClassifier, but batched
over the full held-out validation set instead of frame-by-frame — for
measuring macro-F1 without running the live pipeline).

Written to verify the §8.14 ensemble result after the §14.8-style dense-eval
correction found better checkpoints for its two members
(three_gru_yawn_dense, three_gru_gated_dense) than the originally-documented
ones — see docs/METHODOLOGY.md.

Usage:
    python -m train.eval_ensemble models/state_classifier/three_gru_gated_dense/state_gru.pt models/state_classifier/three_gru_yawn_dense/state_gru.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from .train_sequence import GatedFatigueGRU, GatedThreeGRU, SequenceGRU, full_sequence_eval
from .train_state import STATES, _report, _smooth_eval, driver_of, split_by_driver


def load_model(pt_path: str):
    ckpt = torch.load(pt_path, map_location="cpu")
    sd = ckpt["state_dict"]
    states = ckpt.get("states", STATES)
    features = ckpt["features"]
    mean, std = sd["mean"], sd["std"]
    if ckpt.get("gated_three"):
        model = GatedThreeGRU(features, ckpt["fatigue_features"], ckpt["distract_features"],
                              mean, std, hidden=ckpt["hidden"], layers=ckpt["layers"])
    elif ckpt.get("gated_fatigue"):
        model = GatedFatigueGRU(features, ckpt["fatigue_features"], mean, std,
                                hidden=ckpt["hidden"], layers=ckpt["layers"])
    else:
        model = SequenceGRU(len(features), len(states), mean, std,
                            hidden=ckpt["hidden"], layers=ckpt["layers"])
    model.load_state_dict(sd)
    model.eval()
    return model, features, states


def main():
    ap = argparse.ArgumentParser(description="Evaluate a softmax-averaged ensemble of GRU checkpoints.")
    ap.add_argument("checkpoints", nargs="+", help="one or more state_gru.pt paths")
    ap.add_argument("--table", default="train/output/train_table.csv")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smooth-window", type=int, default=30)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    states = None
    for p in args.checkpoints:
        model, features, st = load_model(p)
        model.to(dev)
        if states is None:
            states = st
        elif st != states:
            raise SystemExit(f"checkpoint class order mismatch: {p} has {st}, expected {states}")
        models.append((model, features))
        print(f"loaded {p}  ({len(features)} features, hidden={model.gru.hidden_size if hasattr(model, 'gru') else '?'})")

    df = pd.read_csv(args.table)
    df["driver"] = df.session.map(driver_of)
    df = df.dropna(subset=list({f for _, feats in models for f in feats}) + ["state"])
    df = df[df.state.isin(STATES)].copy()
    lab = {s: i for i, s in enumerate(states)}
    from .train_state import TO_THREE
    target_map = TO_THREE if len(states) == 3 else None
    df["target"] = df.state if target_map is None else df.state.map(target_map)
    df["y"] = df.target.map(lab)
    df = df.dropna(subset=["y"])
    df["y"] = df["y"].astype(int)

    _, va_mask, val_drivers = split_by_driver(df, args.val_frac, args.seed)
    va = df[va_mask]
    print(f"\nheld-out val drivers: {', '.join(val_drivers)} ({len(va):,} frames)")

    proba_sum = None
    for model, features in models:
        proba = full_sequence_eval(model, va, features, dev)
        proba_sum = proba if proba_sum is None else proba_sum + proba
    proba = proba_sum / len(models)
    pred = proba.argmax(1)
    yva = va.y.to_numpy()

    _report(yva, pred, states, f"ensemble of {len(models)} checkpoint(s) — held-out drivers, raw per-frame")

    if args.smooth_window > 0:
        y_smooth, pred_smooth = _smooth_eval(va, proba, len(states), args.smooth_window)
        _report(y_smooth, pred_smooth, states,
                f"ensemble, temporally smoothed (window={args.smooth_window} frames)")


if __name__ == "__main__":
    main()
