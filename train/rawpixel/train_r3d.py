"""Train the 3D-CNN comparison baseline: R3D-18 (torchvision, Kinetics-400
pretrained) as a clip classifier over raw face crops. See
docs/METHODOLOGY.md §14.7 and R3DClipClassifier's docstring in model.py for
the full reasoning — this is the literature's recommended second comparison
point after the 2D-CNN+GRU variants (14.3/14.5), included to test whether a
genuinely spatiotemporal (3D-conv) architecture does better than a 2D-CNN
representation + GRU at our subject-count/compute scale.

Reads raw face crops directly from extract_crops.py's output, same as
train_cnn_gru_finetune.py — no separate caching step (the backbone trains).

Usage:
    python -m train.rawpixel.train_r3d
    python -m train.rawpixel.train_r3d --epochs 10
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from .model import FocalLoss, R3DClipClassifier, THREE_STATES, driver_of, split_by_driver


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
            "crops": d["crops"][mask],
            "frame_idx": d["frame_idx"][mask],
            "y": y[mask],
        })
    return sessions


class ClipDataset(Dataset):
    """Non-overlapping fixed-length clips, one label per clip (the mode of
    that clip's per-frame labels — clips are short, default 16 frames
    (~1.1s at the stride-2 rate), so a label change mid-clip is rare but not
    impossible near a state transition; mode is the simplest, most standard
    choice for collapsing a short window to one training target."""
    def __init__(self, sessions: list, clip_len: int):
        self.sessions = sessions
        self.clip_len = clip_len
        self.index = []
        for si, s in enumerate(sessions):
            T = len(s["y"])
            for start in range(0, T - clip_len + 1, clip_len):
                self.index.append((si, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        si, start = self.index[idx]
        s = self.sessions[si]
        clip = s["crops"][start:start + self.clip_len]
        y_window = s["y"][start:start + self.clip_len]
        y = int(np.bincount(y_window, minlength=len(THREE_STATES)).argmax())
        return torch.from_numpy(clip.copy()), y


def full_clip_eval(model: R3DClipClassifier, sessions: list, clip_len: int, dev: str,
                   batch: int = 32):
    """Non-overlapping clips over each full held-out session, in order;
    each clip's single prediction is broadcast across every frame in that
    clip so results are comparable (same frame-level macro-F1 metric) to
    every other experiment — a documented approximation, not a true
    per-frame temporal resolution match (see R3DClipClassifier's
    docstring). Trailing frames shorter than one clip are dropped (same
    convention as training)."""
    model.eval()
    y_true_parts, pred_parts = [], []
    with torch.no_grad():
        for s in sessions:
            T = len(s["y"])
            n_win = T // clip_len
            if n_win == 0:
                continue
            clips = np.stack([s["crops"][w * clip_len:(w + 1) * clip_len] for w in range(n_win)])
            clips_t = torch.from_numpy(clips).to(dev)
            logits_parts = [model(clips_t[i:i + batch]) for i in range(0, len(clips_t), batch)]
            pred_per_window = torch.cat(logits_parts, 0).argmax(1).cpu().numpy()
            pred_frames = np.repeat(pred_per_window, clip_len)
            y_true_parts.append(s["y"][:n_win * clip_len])
            pred_parts.append(pred_frames)
    return np.concatenate(y_true_parts), np.concatenate(pred_parts)


def main():
    ap = argparse.ArgumentParser(description="Train the R3D-18 3D-CNN clip-classifier baseline.")
    ap.add_argument("--crops-dir", default="train/output/rawpixel/crops")
    ap.add_argument("--out-dir", default="models/rawpixel_classifier/three_r3d")
    ap.add_argument("--clip-len", type=int, default=16,
                    help="R3D-18's Kinetics pretraining convention (~1.1s at the stride-2 rate — "
                         "much shorter context than the GRU variants get, a real limitation)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3, help="LR for the new fc head")
    ap.add_argument("--backbone-lr", type=float, default=1e-4, help="LR for unfrozen layer4")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight-power", type=float, default=1.0)
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--freeze-all", action="store_true",
                    help="ablation: keep layer4 frozen too (only the new fc head trains)")
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
    all_drivers = sorted({s["driver"] for s in sessions})
    print(f"classes: {classes}")
    print(f"drivers: {len(all_drivers)} total | "
          f"train {len(all_drivers) - len(val_drivers)} | val {len(val_drivers)}")
    print(f"held-out val drivers: {', '.join(sorted(val_drivers))}")

    ds = ClipDataset(tr_sessions, args.clip_len)
    print(f"\n{len(ds):,} training clips (len={args.clip_len}, non-overlapping)")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    clip_labels = np.array([ds[i][1] for i in range(len(ds))]) if len(ds) < 20000 else None
    if clip_labels is None:
        # avoid materializing every clip just to count labels on large datasets;
        # approximate via each session's raw frame counts instead (close enough
        # for class-weight purposes, since clips rarely straddle a label change)
        counts = np.zeros(len(classes))
        for s in tr_sessions:
            counts += np.bincount(s["y"], minlength=len(classes))
    else:
        counts = np.bincount(clip_labels, minlength=len(classes)).astype(np.float64)
    w = (counts.sum() / (len(classes) * np.clip(counts, 1, None))) ** args.weight_power
    print("class weights:", {classes[i]: round(float(w[i]), 2) for i in range(len(classes))})

    model = R3DClipClassifier(len(classes), unfreeze_last_block=not args.freeze_all).to(dev)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable:,} / {n_total:,} "
          f"(layer4 {'unfrozen' if not args.freeze_all else 'frozen — ablation mode'})")

    backbone_params = list(model.net.layer4.parameters()) if not args.freeze_all else []
    head_params = list(model.net.fc.parameters())
    param_groups = [{"params": head_params, "lr": args.lr}]
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
            logits = model(xb)
            loss = lossfn(logits, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        do_eval = (ep + 1) % args.eval_every == 0 or ep == 0 or ep == args.epochs - 1
        msg = f"  epoch {ep+1:2d}  train_loss {tot/n:.4f}"
        if do_eval:
            y_all, pred = full_clip_eval(model, va_sessions, args.clip_len, dev)
            mf1 = f1_score(y_all, pred, average="macro")
            acc = (pred == y_all).mean()
            flag = ""
            if mf1 > best_mf1:
                best_mf1, best_epoch = mf1, ep + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                flag = "  (best)"
            msg += f"  val_macroF1 {mf1:.3f}  val_acc {acc:.3f}{flag}"
        print(msg)

    print(f"\nrestoring best checkpoint: epoch {best_epoch} (val_macroF1 {best_mf1:.3f})")
    model.load_state_dict(best_state)

    y_all, pred = full_clip_eval(model, va_sessions, args.clip_len, dev)
    _report(y_all, pred, classes, "validation (held-out drivers, clip-broadcast-to-frame)")

    if (y_all == fatigue_idx).any():
        recall = (pred[y_all == fatigue_idx] == fatigue_idx).mean()
        print(f"\nFATIGUED recall: {recall:.1%}  of {(y_all == fatigue_idx).sum():,} frames")

    os.makedirs(args.out_dir, exist_ok=True)
    pt = os.path.join(args.out_dir, "state_r3d.pt")
    torch.save({"state_dict": model.state_dict(), "states": classes,
               "clip_len": args.clip_len, "unfroze_last_block": not args.freeze_all}, pt)
    print(f"\nsaved {pt}")


if __name__ == "__main__":
    main()
