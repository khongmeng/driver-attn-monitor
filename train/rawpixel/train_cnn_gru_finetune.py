"""Fine-tuning variant of train_cnn_gru.py: trains the CNN backbone's last
residual block (ResNet18 layer4) jointly with the projection+GRU+head,
instead of using a fully-frozen, pre-cached embedding. See
docs/METHODOLOGY.md §14.5 and FineTuneCNNGRU's docstring in model.py for why
only the last block (not the whole network) is unfrozen.

Reads raw face crops directly from extract_crops.py's output — there is no
embed_backbone.py-style caching step for this variant, since the backbone's
weights change during training so its output can't be precomputed once.

Compute note: backprop through a CNN is far more expensive than through a
tiny GRU alone, so this uses a shorter chunk length and smaller batch than
train_cnn_gru.py to fit comfortably in 16GB VRAM (see --chunk-len/--batch
defaults) — a compute-driven simplification, documented as a limitation
(shorter temporal context per training step than the frozen-embedding runs).

Usage:
    python -m train.rawpixel.train_cnn_gru_finetune
    python -m train.rawpixel.train_cnn_gru_finetune --epochs 15
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
from torch.utils.data import DataLoader, Dataset

from .model import (FineTuneCNNGRU, FocalLoss, THREE_STATES, driver_of,
                    full_sequence_eval_finetune, split_by_driver)


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


def load_sessions(crops_dir: str) -> list:
    lab = {s: i for i, s in enumerate(THREE_STATES)}
    sessions = []
    for f in sorted(glob.glob(os.path.join(crops_dir, "*.npz"))):
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
            "crops": d["crops"][mask],           # (T,H,W,3) uint8, kept in system RAM
            "frame_idx": d["frame_idx"][mask],
            "y": y[mask],
        })
    return sessions


class ChunkCropDataset(Dataset):
    """Lazily slices fixed-length raw-crop chunks from the already-loaded
    per-session arrays (no upfront duplication — full 65-session crop data
    is ~8.5GB, materializing every overlapping chunk separately would
    multiply that many times over)."""
    def __init__(self, sessions: list, chunk_len: int, stride: int):
        self.sessions = sessions
        self.chunk_len = chunk_len
        self.index = []
        for si, s in enumerate(sessions):
            T = len(s["y"])
            if T < chunk_len:
                continue
            for start in range(0, T - chunk_len + 1, stride):
                self.index.append((si, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        si, start = self.index[idx]
        s = self.sessions[si]
        crops = s["crops"][start:start + self.chunk_len]
        y = s["y"][start:start + self.chunk_len]
        return torch.from_numpy(crops.copy()), torch.from_numpy(y.copy())


def main():
    ap = argparse.ArgumentParser(description="Fine-tune ResNet18's last block jointly with the GRU.")
    ap.add_argument("--crops-dir", default="train/output/rawpixel/crops")
    ap.add_argument("--out-dir", default="models/rawpixel_classifier/three_cnn_gru_finetune")
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--chunk-len", type=int, default=256,
                    help="shorter than train_cnn_gru.py's 600 — CNN backprop is far more "
                         "expensive per frame than a GRU-only step (~17s of context at the "
                         "stride-2 rate, still covers the 15s PERCLOS label window)")
    ap.add_argument("--chunk-stride", type=int, default=256, help="no overlap by default, unlike "
                    "train_cnn_gru.py's 50%%, to keep the chunk count (and training time) down")
    ap.add_argument("--batch", type=int, default=2, help="chunks per step; batch*chunk_len images "
                    "go through the CNN with gradients each step — kept small for 16GB VRAM")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3, help="LR for the proj/GRU/head (new params)")
    ap.add_argument("--backbone-lr", type=float, default=1e-4,
                    help="LR for the unfrozen ResNet18 layer4 params — lower than --lr so "
                         "fine-tuning doesn't destroy the pretrained features in a few steps")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight-power", type=float, default=1.0)
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--smooth-window", type=int, default=0)
    ap.add_argument("--freeze-all", action="store_true",
                    help="ablation: keep layer4 frozen too (sanity check that this script "
                         "reproduces train_cnn_gru.py-like behavior when nothing is unfrozen)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    classes = THREE_STATES

    print("loading raw crops into memory (all 65 sessions, ~8.5GB) ...")
    sessions = load_sessions(args.crops_dir)
    if not sessions:
        raise SystemExit(f"No labeled sessions in {args.crops_dir}. Run extract_crops.py first.")

    session_drivers = {s["name"]: s["driver"] for s in sessions}
    fatigue_idx = classes.index("FATIGUED")
    fatigue_drivers = {s["driver"] for s in sessions if (s["y"] == fatigue_idx).any()}
    val_drivers = split_by_driver(session_drivers, fatigue_drivers, args.val_frac, args.seed)

    tr_sessions = [s for s in sessions if s["driver"] not in val_drivers]
    va_sessions = [s for s in sessions if s["driver"] in val_drivers]
    n_tr = sum(len(s["y"]) for s in tr_sessions)
    n_va = sum(len(s["y"]) for s in va_sessions)
    all_drivers = sorted({s["driver"] for s in sessions})
    print(f"classes: {classes}")
    print(f"drivers: {len(all_drivers)} total | "
          f"train {len(all_drivers) - len(val_drivers)} ({n_tr:,} frames) | "
          f"val {len(val_drivers)} ({n_va:,} frames)")
    print(f"held-out val drivers: {', '.join(sorted(val_drivers))}")

    ds = ChunkCropDataset(tr_sessions, args.chunk_len, args.chunk_stride)
    print(f"\n{len(ds):,} training chunks (len={args.chunk_len}, stride={args.chunk_stride})")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    counts = np.zeros(len(classes))
    for s in tr_sessions:
        counts += np.bincount(s["y"], minlength=len(classes))
    w = (counts.sum() / (len(classes) * np.clip(counts, 1, None))) ** args.weight_power
    print("class weights:", {classes[i]: round(float(w[i]), 2) for i in range(len(classes))})

    model = FineTuneCNNGRU(len(classes), proj_dim=args.proj_dim, hidden=args.hidden,
                           layers=args.layers, unfreeze_last_block=not args.freeze_all).to(dev)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable:,} / {n_total:,} "
          f"(layer4 {'unfrozen' if not args.freeze_all else 'frozen — ablation mode'})")

    backbone_params = list(model.backbone.layer4.parameters()) if not args.freeze_all else []
    other_params = list(model.proj.parameters()) + list(model.gru.parameters()) + list(model.head.parameters())
    param_groups = [{"params": other_params, "lr": args.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.backbone_lr})
    opt = torch.optim.Adam(param_groups)

    weights = torch.tensor(w, dtype=torch.float32).to(dev)
    lossfn = (nn.CrossEntropyLoss(weight=weights) if args.loss == "ce"
             else FocalLoss(weights, gamma=args.focal_gamma))

    print(f"\ntraining on {dev} ({args.loss} loss, lr={args.lr}/backbone_lr={args.backbone_lr}) ...")
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
        do_eval = (ep + 1) % args.eval_every == 0 or ep == 0 or ep == args.epochs - 1
        msg = f"  epoch {ep+1:2d}  train_loss {tot/n:.4f}"
        if do_eval:
            proba_by_name = full_sequence_eval_finetune(model, va_sessions, dev)
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
            msg += f"  val_macroF1 {mf1:.3f}  val_acc {acc:.3f}{flag}"
        print(msg)

    print(f"\nrestoring best checkpoint: epoch {best_epoch} (val_macroF1 {best_mf1:.3f})")
    model.load_state_dict(best_state)

    proba_by_name = full_sequence_eval_finetune(model, va_sessions, dev)
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
               "proj_dim": args.proj_dim, "hidden": args.hidden, "layers": args.layers,
               "unfroze_last_block": not args.freeze_all}, pt)
    print(f"\nsaved {pt}")


if __name__ == "__main__":
    main()
