"""Train the raw-pixel end-to-end classifier: cached frozen-CNN embeddings ->
EmbeddingGRU -> 3-class driver state. Self-contained counterpart to
train/train_sequence.py — see model.py's docstring and
docs/METHODOLOGY.md §14 for why this pipeline is kept decoupled from the
feature-based one, even though the training loop shape mirrors it closely
on purpose (for a clean side-by-side comparison).

Usage:
    python -m train.rawpixel.train_cnn_gru
    python -m train.rawpixel.train_cnn_gru --smooth-window 1
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from .model import (EmbeddingGRU, FocalLoss, THREE_STATES, build_chunks,
                    driver_of, full_sequence_eval, split_by_driver)


def _report(y_true, pred, classes, header):
    print(f"\n=== {header} ===")
    print(classification_report(y_true, pred, labels=list(range(len(classes))),
                                target_names=classes, digits=3, zero_division=0))
    cm = confusion_matrix(y_true, pred, labels=list(range(len(classes))))
    print("confusion matrix (rows=true, cols=pred):")
    print("            " + "  ".join(f"{s[:5]:>6s}" for s in classes))
    for i, s in enumerate(classes):
        print(f"  {s[:10]:10s} " + "  ".join(f"{cm[i][j]:6d}" for j in range(len(classes))))
    print(f"\nmacro-F1: {f1_score(y_true, pred, average='macro'):.3f} | "
          f"accuracy: {(pred == y_true).mean():.3f}")


def load_sessions(emb_dir: str) -> list:
    lab = {s: i for i, s in enumerate(THREE_STATES)}
    sessions = []
    for f in sorted(glob.glob(os.path.join(emb_dir, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        y = np.array([lab.get(s, -1) for s in d["state"]], dtype=np.int64)
        mask = y >= 0
        if not mask.any():
            continue
        name = os.path.splitext(os.path.basename(f))[0]
        sessions.append({
            "name": name,
            "driver": driver_of(name),
            "task": str(d["task"]),
            "embeddings": d["embeddings"][mask].astype(np.float32),
            "frame_idx": d["frame_idx"][mask],
            "y": y[mask],
        })
    return sessions


def main():
    ap = argparse.ArgumentParser(description="Train the raw-pixel CNN-embedding + GRU classifier.")
    ap.add_argument("--embeddings-dir", default="train/output/rawpixel/embeddings")
    ap.add_argument("--out-dir", default="models/rawpixel_classifier/three_cnn_gru")
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--chunk-len", type=int, default=600)
    ap.add_argument("--chunk-stride", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=5,
                    help="compute val macro-F1/accuracy every N epochs (1 = every epoch, "
                         "for a full loss/accuracy curve)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight-power", type=float, default=1.0)
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--smooth-window", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    classes = THREE_STATES

    sessions = load_sessions(args.embeddings_dir)
    if not sessions:
        raise SystemExit(f"No labeled sessions in {args.embeddings_dir}. "
                         f"Run extract_crops.py then embed_backbone.py first.")
    embed_dim = sessions[0]["embeddings"].shape[1]

    session_drivers = {s["name"]: s["driver"] for s in sessions}
    fatigue_idx = classes.index("FATIGUED")
    fatigue_drivers = {s["driver"] for s in sessions if (s["y"] == fatigue_idx).any()}
    val_drivers = split_by_driver(session_drivers, fatigue_drivers, args.val_frac, args.seed)

    tr_sessions = [s for s in sessions if s["driver"] not in val_drivers]
    va_sessions = [s for s in sessions if s["driver"] in val_drivers]
    n_tr = sum(len(s["y"]) for s in tr_sessions)
    n_va = sum(len(s["y"]) for s in va_sessions)
    all_drivers = sorted({s["driver"] for s in sessions})
    print(f"embed_dim: {embed_dim}")
    print(f"classes: {classes}")
    print(f"drivers: {len(all_drivers)} total | "
          f"train {len(all_drivers) - len(val_drivers)} ({n_tr:,} frames) | "
          f"val {len(val_drivers)} ({n_va:,} frames)")
    print(f"held-out val drivers: {', '.join(sorted(val_drivers))}")

    print(f"\nbuilding training chunks (len={args.chunk_len}, stride={args.chunk_stride}) ...")
    Xc, yc = build_chunks(tr_sessions, args.chunk_len, args.chunk_stride)
    print(f"  {len(Xc):,} chunks")

    Xc_t = torch.tensor(Xc)
    yc_t = torch.tensor(yc, dtype=torch.long)
    mean = Xc_t.reshape(-1, Xc_t.shape[-1]).mean(0)
    std = Xc_t.reshape(-1, Xc_t.shape[-1]).std(0).clamp_min(1e-6)

    counts = np.bincount(yc.reshape(-1), minlength=len(classes)).astype(np.float64)
    w = (counts.sum() / (len(classes) * np.clip(counts, 1, None))) ** args.weight_power
    print("\nclass weights:", {classes[i]: round(float(w[i]), 2) for i in range(len(classes))})

    ds = TensorDataset(Xc_t, yc_t)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)

    weights = torch.tensor(w, dtype=torch.float32).to(dev)
    model = EmbeddingGRU(embed_dim, len(classes), mean, std, proj_dim=args.proj_dim,
                         hidden=args.hidden, layers=args.layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossfn = (nn.CrossEntropyLoss(weight=weights) if args.loss == "ce"
             else FocalLoss(weights, gamma=args.focal_gamma))

    print(f"\ntraining on {dev} ({args.loss} loss) ...")
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
            proba_by_name = full_sequence_eval(model, va_sessions, dev)
            y_all = np.concatenate([s["y"] for s in va_sessions])
            p_all = np.concatenate([proba_by_name[s["name"]] for s in va_sessions])
            pred_all = p_all.argmax(1)
            mf1 = f1_score(y_all, pred_all, average="macro")
            acc = (pred_all == y_all).mean()
            flag = ""
            if mf1 > best_mf1:
                best_mf1, best_epoch = mf1, ep + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                flag = "  (best)"
            print(f"  epoch {ep+1:2d}  train_loss {tot/n:.4f}  val_macroF1 {mf1:.3f}  val_acc {acc:.3f}{flag}")

    print(f"\nrestoring best checkpoint: epoch {best_epoch} (val_macroF1 {best_mf1:.3f})")
    model.load_state_dict(best_state)

    proba_by_name = full_sequence_eval(model, va_sessions, dev)
    y_all = np.concatenate([s["y"] for s in va_sessions])
    p_all = np.concatenate([proba_by_name[s["name"]] for s in va_sessions])
    pred = p_all.argmax(1)

    _report(y_all, pred, classes, "validation (held-out drivers, raw per-frame)")

    if args.smooth_window > 0:
        y_parts, pred_parts = [], []
        for s in va_sessions:
            p = proba_by_name[s["name"]]
            d = pd.DataFrame(p, columns=[f"_p{i}" for i in range(len(classes))])
            roll = d.rolling(args.smooth_window, min_periods=1).mean()
            y_parts.append(s["y"])
            pred_parts.append(roll.to_numpy().argmax(1))
        y_smooth = np.concatenate(y_parts)
        pred_smooth = np.concatenate(pred_parts)
        _report(y_smooth, pred_smooth, classes,
                f"validation, temporally smoothed (window={args.smooth_window} frames)")

    if (y_all == fatigue_idx).any():
        recall = (pred[y_all == fatigue_idx] == fatigue_idx).mean()
        print(f"\nFATIGUED recall: {recall:.1%}  of {(y_all == fatigue_idx).sum():,} frames")

    os.makedirs(args.out_dir, exist_ok=True)
    pt = os.path.join(args.out_dir, "state_gru.pt")
    torch.save({"state_dict": model.state_dict(), "states": classes,
               "embed_dim": embed_dim, "proj_dim": args.proj_dim,
               "hidden": args.hidden, "layers": args.layers}, pt)
    print(f"\nsaved {pt}")


if __name__ == "__main__":
    main()
