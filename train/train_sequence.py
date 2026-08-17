"""Stage-④ classifier v2: a GRU over per-session feature sequences, instead of
the per-frame MLP in ``train_state.py``.

Motivation (docs/METHODOLOGY.md §8.3-§8.5): every state is inherently temporal
(PERCLOS *is* a window), but the MLP only sees one frame plus 4 hand-picked
rolling stats. This trains a small ``nn.GRU`` directly on the per-frame feature
sequence so the model can learn its own temporal integration instead of relying
on hand-chosen window lengths.

Training vs. evaluation use different granularities, on purpose:
  * **Train** on fixed-length chunks (default 600 frames, ~40s at the stride-2
    extraction rate) so gradient steps are batched and stable (variable-length
    full-session batches of 1 would mean very few, very long-BPTT steps).
  * **Evaluate** on the FULL, unchunked per-session sequence (one continuous
    forward pass per held-out session) — this matches how the model would
    actually run live (a continuous stream, not fixed windows) and keeps
    macro-F1 directly comparable to ``train_state.py``'s per-frame report.

Class imbalance for sequence data can't use ``train_state.py``'s row-level
undersampling/oversampling (§8.4/§8.5) — shuffling individual frames destroys
the temporal continuity a GRU depends on. Instead:
  * per-timestep class-weighted loss (weighted CE or focal — reuses
    ``train_state.FocalLoss`` unchanged, just flattened over (batch, time)),
  * chunk-level oversampling: chunks containing more FATIGUED frames are
    sampled more often (``--oversample``), which preserves each chunk's
    internal temporal order.

Reuses ``train_state.py``'s driver split, class-set definitions, FocalLoss,
and reporting functions so results land in the same format and are directly
comparable to §8.3-§8.5.

Usage:
    python -m train.train_sequence --classes three
    python -m train.train_sequence --classes three --loss focal --weight-power 0.3 --oversample 5.0
    python -m train.train_sequence --classes three --smooth-window 30
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

from .train_state import (CLASS_SETS, FEATURES, STATES, TARGET_MAPS, FocalLoss,
                          _report, _smooth_eval, driver_of, split_by_driver)

# eye/PERCLOS + yawn only — drops every head-pose/gaze feature to test whether
# the model's reliance on blink_rate/gaze_pitch (see docs/METHODOLOGY.md §8.9
# feature-ablation study) is a real physiological signal or an artifact of
# only having 14 drivers to learn from. blink_rate is included here since
# it's derived from the same eye-state model output as perclos/eye_open_prob
# (a different aggregation of eye closure, not a separate signal).
EYE_YAWN_FEATURES = ["eye_open_prob", "perclos", "perclos_15s", "eye_open_prob_mean_5s",
                     "blink_rate", "mar", "yawn_rate"]
SEQ_FEATURE_SETS = {"full": FEATURES, "eye_yawn": EYE_YAWN_FEATURES}

# gaze/head-pose only — the complement of EYE_YAWN_FEATURES within FEATURES.
# DISTRACTED's GT (build_dataset.py::_label_row) comes only from
# gt_gaze_off_road/gt_driver_action, never eye/yawn, so this is the DISTRACTED
# analog of EYE_YAWN_FEATURES for GatedThreeGRU.
DISTRACT_FEATURES = [f for f in FEATURES if f not in EYE_YAWN_FEATURES]


class _CausalTemporalBlock(nn.Module):
    """One dilated-causal-conv residual block (Bai et al. 2018 TCN design):
    symmetric padding + chomping the trailing `pad` timesteps off the output
    is what makes a normal Conv1d causal (no look-ahead) without a custom
    causal-conv op."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):  # x: (B, C, T)
        out = self.dropout(self.relu(self.conv1(x)[..., :-self.pad]))
        out = self.dropout(self.relu(self.conv2(out)[..., :-self.pad]))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class SequenceTCN(nn.Module):
    """CNN counterpart to SequenceGRU: a stack of dilated causal 1D-conv
    residual blocks over the feature sequence (dilation doubles each level,
    so receptive field grows as ~2^levels instead of the GRU's effectively
    unbounded recurrent memory). Default levels=6, kernel=3 gives a receptive
    field of ~253 frames (~17s at the stride-2 extraction rate) to cover the
    PERCLOS window. Stateless across calls (no hidden state to carry between
    chunks) — full_sequence_eval's one-forward-pass-per-session already gives
    it the full session as context, so this needs no eval-path changes."""
    def __init__(self, n_feat, n_class, mean, std, hidden=64, levels=6, kernel_size=3, dropout=0.2):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        blocks = []
        in_ch = n_feat
        for i in range(levels):
            blocks.append(_CausalTemporalBlock(in_ch, hidden, kernel_size, 2 ** i, dropout))
            in_ch = hidden
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, n_class)

    def forward(self, x, h0=None):
        xn = (x - self.mean) / self.std
        out = self.tcn(xn.transpose(1, 2)).transpose(1, 2)
        return self.head(out), None


class SequenceGRU(nn.Module):
    def __init__(self, n_feat, n_class, mean, std, hidden=64, layers=2, dropout=0.2,
                bidirectional=False):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.gru = nn.GRU(n_feat, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0, bidirectional=bidirectional)
        self.head = nn.Linear(hidden * (2 if bidirectional else 1), n_class)

    def forward(self, x, h0=None):
        x = (x - self.mean) / self.std
        out, h = self.gru(x, h0)
        return self.head(out), h


class GatedFatigueGRU(nn.Module):
    """Two-branch architecture, only valid for the three-class task (class
    order [FOCUSED, DISTRACTED, FATIGUED] is hardcoded to route outputs to
    the right branch). Unlike restricting the whole model to eye/PERCLOS+yawn
    features (§8.9 — a big regression, since DISTRACTED needs gaze/head-pose),
    this keeps the full feature set for FOCUSED/DISTRACTED via a 'main' GRU,
    and adds a completely separate 'fatigue' GRU fed ONLY the eye/PERCLOS+yawn
    subset. There is no computational path from gaze/head-pose into the
    FATIGUED logit — this is an architectural guarantee, not a learned
    preference, which is what "enforce" requires (a shared hidden state can't
    give that guarantee, since a GRU doesn't compartmentalize which input
    feature influenced which part of its memory)."""
    def __init__(self, all_feat_names, fatigue_feat_names, mean, std,
                hidden=64, layers=2, hidden_fatigue=32, layers_fatigue=1, dropout=0.2,
                bidirectional=False):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("fatigue_idx",
                             torch.tensor([all_feat_names.index(f) for f in fatigue_feat_names],
                                         dtype=torch.long))
        self.main_gru = nn.GRU(len(all_feat_names), hidden, num_layers=layers, batch_first=True,
                               dropout=dropout if layers > 1 else 0.0, bidirectional=bidirectional)
        self.fatigue_gru = nn.GRU(len(fatigue_feat_names), hidden_fatigue, num_layers=layers_fatigue,
                                  batch_first=True, dropout=dropout if layers_fatigue > 1 else 0.0,
                                  bidirectional=bidirectional)
        mult = 2 if bidirectional else 1
        self.main_head = nn.Linear(hidden * mult, 2)              # FOCUSED, DISTRACTED
        self.fatigue_head = nn.Linear(hidden_fatigue * mult, 1)   # FATIGUED

    def forward(self, x, h=None):
        h_main, h_fat = h if h is not None else (None, None)
        xn = (x - self.mean) / self.std
        x_fat = xn.index_select(-1, self.fatigue_idx)
        out_main, h_main = self.main_gru(xn, h_main)
        out_fat, h_fat = self.fatigue_gru(x_fat, h_fat)
        logits = torch.cat([self.main_head(out_main), self.fatigue_head(out_fat)], dim=-1)
        return logits, (h_main, h_fat)


class GatedThreeGRU(nn.Module):
    """Extends GatedFatigueGRU's idea to all three classes, per how their GT
    is actually derived in build_dataset.py::_label_row:
      * FATIGUED  <- gt_perclos / gt_yawn only              -> eye/PERCLOS+yawn branch
      * DISTRACTED <- gt_gaze_off_road / gt_driver_action    -> gaze/head-pose branch
      * FOCUSED   <- EITHER "absence of drowsy/tired" (drowsiness-protocol
                     frames) OR "gaze-on-road + safe_drive" (distraction-
                     protocol frames) — heterogeneous across DMD's two
                     protocols. A live model gets no protocol label telling
                     it which rule applies to the current frame, so unlike
                     FATIGUED/DISTRACTED, FOCUSED can't be narrowed to one
                     feature family — it keeps the full feature set.
    Three independent GRUs, no shared computation between any of them; class
    order [FOCUSED, DISTRACTED, FATIGUED] is hardcoded."""
    def __init__(self, all_feat_names, fatigue_feat_names, distract_feat_names, mean, std,
                hidden=64, layers=2, hidden_small=32, layers_small=1, dropout=0.2):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("fatigue_idx",
                             torch.tensor([all_feat_names.index(f) for f in fatigue_feat_names],
                                         dtype=torch.long))
        self.register_buffer("distract_idx",
                             torch.tensor([all_feat_names.index(f) for f in distract_feat_names],
                                         dtype=torch.long))
        self.focused_gru = nn.GRU(len(all_feat_names), hidden, num_layers=layers, batch_first=True,
                                  dropout=dropout if layers > 1 else 0.0)
        self.distract_gru = nn.GRU(len(distract_feat_names), hidden_small, num_layers=layers_small,
                                   batch_first=True, dropout=dropout if layers_small > 1 else 0.0)
        self.fatigue_gru = nn.GRU(len(fatigue_feat_names), hidden_small, num_layers=layers_small,
                                  batch_first=True, dropout=dropout if layers_small > 1 else 0.0)
        self.focused_head = nn.Linear(hidden, 1)
        self.distract_head = nn.Linear(hidden_small, 1)
        self.fatigue_head = nn.Linear(hidden_small, 1)

    def forward(self, x, h=None):
        h_foc, h_dis, h_fat = h if h is not None else (None, None, None)
        xn = (x - self.mean) / self.std
        x_dis = xn.index_select(-1, self.distract_idx)
        x_fat = xn.index_select(-1, self.fatigue_idx)
        out_foc, h_foc = self.focused_gru(xn, h_foc)
        out_dis, h_dis = self.distract_gru(x_dis, h_dis)
        out_fat, h_fat = self.fatigue_gru(x_fat, h_fat)
        logits = torch.cat([self.focused_head(out_foc), self.distract_head(out_dis),
                           self.fatigue_head(out_fat)], dim=-1)
        return logits, (h_foc, h_dis, h_fat)


def build_chunks(df: pd.DataFrame, feature_cols, chunk_len: int, stride: int):
    """Fixed-length, non-padded chunks per session (sessions shorter than
    chunk_len are skipped — none are in this dataset, but defensively handled
    rather than assumed)."""
    xs, ys = [], []
    skipped = []
    for session, g in df.groupby("session", sort=False):
        g = g.sort_values("frame")
        X = g[feature_cols].to_numpy(np.float32)
        y = g["y"].to_numpy(np.int64)
        T = len(X)
        if T < chunk_len:
            skipped.append((session, T))
            continue
        for start in range(0, T - chunk_len + 1, stride):
            xs.append(X[start:start + chunk_len])
            ys.append(y[start:start + chunk_len])
    if skipped:
        print(f"  (skipped {len(skipped)} session(s) shorter than chunk_len={chunk_len}: "
              + ", ".join(f"{s}({t})" for s, t in skipped[:5])
              + (" ..." if len(skipped) > 5 else ""))
    return np.stack(xs), np.stack(ys)


def full_sequence_eval(model: SequenceGRU, df: pd.DataFrame, feature_cols, dev: str) -> np.ndarray:
    """One continuous forward pass per session (matches live/streaming use),
    returned in df's original row order regardless of session grouping."""
    model.eval()
    proba_by_idx = {}
    with torch.no_grad():
        for session, g in df.groupby("session", sort=False):
            g = g.sort_values("frame")
            X = torch.tensor(g[feature_cols].to_numpy(np.float32)).unsqueeze(0).to(dev)
            logits, _ = model(X)
            p = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            for idx, row in zip(g.index, p):
                proba_by_idx[idx] = row
    return np.stack([proba_by_idx[idx] for idx in df.index])


def main():
    ap = argparse.ArgumentParser(description="Train the Stage-④ GRU sequence classifier.")
    ap.add_argument("--table", default="train/output/train_table.csv")
    ap.add_argument("--classes", choices=sorted(CLASS_SETS), default="three")
    ap.add_argument("--out-dir", default=None, help="default: models/state_classifier/<classes>_gru")
    ap.add_argument("--chunk-len", type=int, default=600, help="training chunk length in frames (~40s)")
    ap.add_argument("--chunk-stride", type=int, default=300, help="stride between chunks (50% overlap by default)")
    ap.add_argument("--model", choices=["gru", "tcn"], default="gru",
                    help="gru = SequenceGRU (default); tcn = SequenceTCN, a dilated-causal-"
                         "conv counterpart, comparison baseline only (not yet wired into "
                         "run_live.py or exported to ONNX)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--levels", type=int, default=6,
                    help="--model tcn only: number of dilated conv blocks "
                         "(receptive field grows as ~2^levels)")
    ap.add_argument("--kernel-size", type=int, default=3, help="--model tcn only: conv kernel size")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=5,
                    help="compute val macro-F1/accuracy every N epochs (always includes epoch 1 "
                         "and the final epoch); 1 = every epoch, for a full loss/accuracy curve")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight-power", type=float, default=1.0)
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--oversample", type=float, default=0.0,
                    help="chunk-level oversampling factor toward FATIGUED-containing chunks "
                         "(sample weight = 1 + factor * fraction_of_chunk_that_is_FATIGUED); "
                         "0 = off (uniform chunk sampling)")
    ap.add_argument("--balance", choices=["none", "undersample"], default="none",
                    help="none = train on all chunks (default). undersample = the sequence-model "
                         "analog of train_state.py's row-level undersampling (§8.4/8.5): keep every "
                         "chunk that contains at least one FATIGUED frame, and randomly drop "
                         "non-fatigue chunks down to that same count. Drops whole chunks, never "
                         "individual frames, so no surviving chunk's internal temporal order is "
                         "ever broken (row-level undersampling would fragment sequences).")
    ap.add_argument("--smooth-window", type=int, default=0)
    ap.add_argument("--feature-set", choices=sorted(SEQ_FEATURE_SETS), default="full",
                    help="full = all 13 features (default); eye_yawn = only "
                         "eye/PERCLOS + yawn features, drops every head-pose/gaze "
                         "feature entirely (tests whether the model's reliance on "
                         "blink_rate/gaze_pitch is real signal or a 14-driver artifact)")
    ap.add_argument("--gated-fatigue", action="store_true",
                    help="two-branch architecture, --classes three only: FATIGUED comes from "
                         "a separate GRU fed ONLY eye/PERCLOS+yawn features (architecturally "
                         "zero influence from gaze/head-pose); FOCUSED/DISTRACTED still see the "
                         "full feature set via a separate main branch. Use with --feature-set "
                         "full (the point is keeping everything for FOCUSED/DISTRACTED).")
    ap.add_argument("--bidirectional", action="store_true",
                    help="use a bidirectional GRU (both branches, for --gated-fatigue) instead of "
                         "causal/forward-only. Only valid when the model does NOT need to run as a "
                         "live incremental stream (full_sequence_eval already does one whole-session "
                         "forward pass, so this needs no eval-path changes, but run_live.py's "
                         "frame-by-frame hidden-state carrying does not support it). --model tcn "
                         "does not support this flag.")
    ap.add_argument("--gated-three", action="store_true",
                    help="three-branch architecture, --classes three only: FATIGUED sees only "
                         "eye/PERCLOS+yawn, DISTRACTED sees only gaze/head-pose (matches how "
                         "each is actually derived in build_dataset.py's GT), FOCUSED sees the "
                         "full feature set (its GT derivation is heterogeneous across DMD's two "
                         "protocols, so it can't be narrowed the way the other two can).")
    ap.add_argument("--extraction-stride", type=int, default=2,
                    help="the --stride used by extract_features.py to build this table (all "
                         "tables so far used 2 — every 2nd frame). Recorded in the checkpoint so "
                         "run_live.py can feed the GRU at the same effective rate it was trained "
                         "on instead of guessing; a GRU fed every native frame at inference after "
                         "being trained on stride-2 steps runs its recurrence at 2x the intended "
                         "rate, distorting its learned integration/decay timescale.")
    args = ap.parse_args()
    if (args.gated_fatigue or args.gated_three) and args.classes != "three":
        raise SystemExit("--gated-fatigue/--gated-three only support --classes three (hardcodes "
                         "the FOCUSED/DISTRACTED/FATIGUED output order)")
    if args.gated_fatigue and args.gated_three:
        raise SystemExit("--gated-fatigue and --gated-three are mutually exclusive")
    if args.model == "tcn" and (args.gated_fatigue or args.gated_three):
        raise SystemExit("--model tcn doesn't support --gated-fatigue/--gated-three "
                         "(those architectures are GRU-specific)")
    if args.model == "tcn" and args.bidirectional:
        raise SystemExit("--model tcn doesn't support --bidirectional")
    if args.gated_three and args.bidirectional:
        raise SystemExit("--gated-three + --bidirectional not implemented (not needed for the "
                         "§16 experiment this flag was added for)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    classes = CLASS_SETS[args.classes]
    feat_cols = SEQ_FEATURE_SETS[args.feature_set]
    arch_suffix = "_gated_three" if args.gated_three else ("_gated" if args.gated_fatigue else "")
    model_tag = "tcn" if args.model == "tcn" else "gru"
    out_dir = args.out_dir or (
        f"models/state_classifier/{args.classes}_{model_tag}"
        + ("" if args.feature_set == "full" else f"_{args.feature_set}")
        + arch_suffix)

    df = pd.read_csv(args.table)
    df["driver"] = df.session.map(driver_of)
    df = df.dropna(subset=feat_cols + ["state"])
    df = df[df.state.isin(STATES)].copy()
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

    print(f"\nbuilding training chunks (len={args.chunk_len}, stride={args.chunk_stride}) ...")
    Xc, yc = build_chunks(tr, feat_cols, args.chunk_len, args.chunk_stride)
    print(f"  {len(Xc):,} chunks")

    if args.balance == "undersample" and "FATIGUED" in classes:
        fatigue_idx = classes.index("FATIGUED")
        has_fatigue = (yc == fatigue_idx).any(axis=1)
        n_fat = int(has_fatigue.sum())
        rng = np.random.default_rng(args.seed)
        no_fat_idx = np.where(~has_fatigue)[0]
        keep_no_fat = rng.choice(no_fat_idx, size=min(n_fat, len(no_fat_idx)), replace=False)
        keep_idx = np.sort(np.concatenate([np.where(has_fatigue)[0], keep_no_fat]))
        print(f"chunk undersampling: kept {len(keep_idx)}/{len(Xc)} chunks "
              f"({n_fat} fatigue-containing + {len(keep_no_fat)} non-fatigue, dropped {len(Xc) - len(keep_idx)})")
        Xc, yc = Xc[keep_idx], yc[keep_idx]

    Xc_t = torch.tensor(Xc)
    yc_t = torch.tensor(yc, dtype=torch.long)
    mean = Xc_t.reshape(-1, Xc_t.shape[-1]).mean(0)
    std = Xc_t.reshape(-1, Xc_t.shape[-1]).std(0).clamp_min(1e-6)

    counts = np.bincount(yc.reshape(-1), minlength=len(classes)).astype(np.float64)
    w = (counts.sum() / (len(classes) * np.clip(counts, 1, None))) ** args.weight_power
    print("\nclass weights:", {classes[i]: round(float(w[i]), 2) for i in range(len(classes))})

    sampler = None
    if args.oversample > 0 and "FATIGUED" in classes:
        fatigue_idx = classes.index("FATIGUED")
        frac = (yc == fatigue_idx).mean(axis=1)
        sample_w = 1.0 + args.oversample * frac
        sampler = WeightedRandomSampler(torch.tensor(sample_w), num_samples=len(sample_w), replacement=True)
        print(f"chunk oversampling: factor={args.oversample}, "
              f"{int((frac > 0).sum())}/{len(frac)} chunks contain a FATIGUED frame")

    ds = TensorDataset(Xc_t, yc_t)
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, shuffle=(sampler is None))

    weights = torch.tensor(w, dtype=torch.float32).to(dev)
    if args.gated_three:
        model = GatedThreeGRU(feat_cols, EYE_YAWN_FEATURES, DISTRACT_FEATURES, mean, std,
                              hidden=args.hidden, layers=args.layers).to(dev)
        print(f"\ngated-three architecture: FATIGUED sees only {EYE_YAWN_FEATURES}\n"
              f"                          DISTRACTED sees only {DISTRACT_FEATURES}\n"
              f"                          FOCUSED sees all {feat_cols}")
    elif args.gated_fatigue:
        model = GatedFatigueGRU(feat_cols, EYE_YAWN_FEATURES, mean, std,
                                hidden=args.hidden, layers=args.layers,
                                bidirectional=args.bidirectional).to(dev)
        print(f"\ngated-fatigue architecture ({'bidirectional' if args.bidirectional else 'causal'}): "
              f"FATIGUED branch sees only {EYE_YAWN_FEATURES}")
    elif args.model == "tcn":
        model = SequenceTCN(len(feat_cols), len(classes), mean, std,
                            hidden=args.hidden, levels=args.levels,
                            kernel_size=args.kernel_size).to(dev)
        print(f"\nTCN architecture: {args.levels} dilated-causal-conv blocks, "
              f"kernel={args.kernel_size}, hidden={args.hidden}")
    else:
        model = SequenceGRU(len(feat_cols), len(classes), mean, std,
                            hidden=args.hidden, layers=args.layers,
                            bidirectional=args.bidirectional).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossfn = (nn.CrossEntropyLoss(weight=weights) if args.loss == "ce"
             else FocalLoss(weights, gamma=args.focal_gamma))

    print(f"\ntraining on {dev} ({args.loss} loss) ...")
    best_mf1 = -1.0
    best_state = None
    best_epoch = -1
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        n = 0
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
            proba = full_sequence_eval(model, va, feat_cols, dev)
            yva_ep = va.y.to_numpy()
            pred_ep = proba.argmax(1)
            mf1 = f1_score(yva_ep, pred_ep, average="macro")
            acc = (pred_ep == yva_ep).mean()
            flag = ""
            if mf1 > best_mf1:
                best_mf1, best_epoch = mf1, ep + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                flag = "  (best)"
            print(f"  epoch {ep+1:2d}  train_loss {tot/n:.4f}  val_macroF1 {mf1:.3f}  val_acc {acc:.3f}{flag}")

    print(f"\nrestoring best checkpoint: epoch {best_epoch} (val_macroF1 {best_mf1:.3f})")
    model.load_state_dict(best_state)

    proba = full_sequence_eval(model, va, feat_cols, dev)
    pred = proba.argmax(1)
    yva = va.y.to_numpy()

    _report(yva, pred, classes, "validation (held-out drivers, raw per-frame)")

    if args.smooth_window > 0:
        y_smooth, pred_smooth = _smooth_eval(va, proba, len(classes), args.smooth_window)
        _report(y_smooth, pred_smooth, classes,
                f"validation, temporally smoothed (window={args.smooth_window} frames)")
        pred_report = pred_smooth
    else:
        pred_report = pred

    if args.classes == "three":
        fatigued_idx = classes.index("FATIGUED")
        va_state = va.state.to_numpy()
        print("\nFATIGUED recall by original label:")
        for s in ("DROWSY", "TIRED"):
            m = va_state == s
            if m.any():
                print(f"  {s:10s} {(pred_report[m] == fatigued_idx).mean():6.1%}  of {m.sum():,} frames")
    elif args.classes == "binary":
        prob = proba[:, 1]
        from sklearn.metrics import roc_auc_score
        print(f"\nROC-AUC (INATTENTIVE): {roc_auc_score(yva, prob):.3f}")

    os.makedirs(out_dir, exist_ok=True)
    pt = os.path.join(out_dir, f"state_{model_tag}.pt")
    ckpt_out = {"state_dict": model.state_dict(), "features": feat_cols, "states": classes,
               "model": args.model, "hidden": args.hidden,
               "extraction_stride": args.extraction_stride, "bidirectional": args.bidirectional}
    if args.model == "tcn":
        ckpt_out["levels"] = args.levels
        ckpt_out["kernel_size"] = args.kernel_size
    else:
        ckpt_out["layers"] = args.layers
    if args.gated_three:
        ckpt_out["gated_three"] = True
        ckpt_out["fatigue_features"] = EYE_YAWN_FEATURES
        ckpt_out["distract_features"] = DISTRACT_FEATURES
    elif args.gated_fatigue:
        ckpt_out["gated_fatigue"] = True
        ckpt_out["fatigue_features"] = EYE_YAWN_FEATURES
    torch.save(ckpt_out, pt)
    print(f"\nsaved {pt}  ({len(feat_cols)} features, order = {feat_cols})")

    if args.gated_fatigue or args.gated_three:
        print("(skipping ONNX export for the gated architecture — tuple hidden "
              "state doesn't trace cleanly; the .pt checkpoint is what run_live.py uses)")
    elif args.bidirectional:
        print("(skipping ONNX export for the bidirectional model — not live-streamable, "
              "not wired into run_live.py; offline-only by design for this experiment)")
    elif args.model == "tcn":
        print("(skipping ONNX export for the TCN — comparison baseline only, not yet "
              "wired into run_live.py; the .pt checkpoint has everything needed to add "
              "that later)")
    else:
        onnx_path = os.path.join(out_dir, "state_gru.onnx")
        model_cpu = model.to("cpu").eval()
        dummy = torch.zeros(1, 10, len(feat_cols))
        torch.onnx.export(model_cpu, (dummy,), onnx_path, input_names=["features"],
                          output_names=["logits", "hidden"],
                          dynamic_axes={"features": {0: "batch", 1: "seq"},
                                       "logits": {0: "batch", 1: "seq"}},
                          opset_version=13)
        print(f"saved {onnx_path}")


if __name__ == "__main__":
    main()
