"""Self-contained GRU classifier + driver-split + chunking utilities for the
raw-pixel pipeline.

Deliberately duplicated from ``train/train_state.py`` and
``train/train_sequence.py`` (same algorithms: driver-stratified split,
fixed-length chunk training, full-session eval, focal loss) rather than
imported — see docs/METHODOLOGY.md §14 for why this pipeline is kept fully
decoupled from the feature-based one. Using the same seed/val_frac and the
identical split algorithm reproduces the SAME held-out driver set
(gA_1, gC_14, gE_28, gZ_37) as every experiment in train_state.py/
train_sequence.py, so results stay directly comparable despite the code
being independent.
"""
from __future__ import annotations

from typing import Dict, List, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THREE_STATES = ["FOCUSED", "DISTRACTED", "FATIGUED"]


def driver_of(session: str) -> str:
    """DMD driver id = group_subject (e.g. gA_1) — same driver spans sessions."""
    p = str(session).split("_")
    return f"{p[0]}_{p[1]}" if len(p) >= 2 else str(session)


def split_by_driver(session_drivers: Dict[str, str], fatigue_drivers: Set[str],
                    val_frac: float, seed: int) -> Set[str]:
    """Hold out whole drivers, stratifying fatigue-containing drivers
    separately so val always contains some — mirrors
    train_state.py::split_by_driver's algorithm exactly (same rng calls in
    the same order), so it reproduces the identical held-out driver set."""
    rng = np.random.default_rng(seed)
    all_drivers = set(session_drivers.values())
    others = all_drivers - fatigue_drivers
    val_drivers: Set[str] = set()
    for grp in (sorted(fatigue_drivers), sorted(others)):
        g = list(grp)
        rng.shuffle(g)
        k = max(1, round(len(g) * val_frac))
        val_drivers |= set(g[:k])
    return val_drivers


class FocalLoss(nn.Module):
    """Identical to train_state.py::FocalLoss — duplicated, not imported."""
    def __init__(self, weight, gamma: float = 2.0):
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


class EmbeddingGRU(nn.Module):
    """CNN-embedding counterpart to train_sequence.py's SequenceGRU: a small
    trainable linear bottleneck (frozen-embedding-dim -> proj_dim) — a
    lightweight 'linear probe' adaptation on top of the frozen backbone, not
    full fine-tuning — followed by a GRU over the projected sequence. The
    GRU/head sizing (hidden=64, layers=2) matches the feature-based
    SequenceGRU exactly so the *temporal* half of the architecture is held
    constant and only the *representation* (13 hand-features vs. a learned
    CNN embedding) differs between the two pipelines."""
    def __init__(self, embed_dim: int, n_class: int, mean, std, proj_dim: int = 128,
                hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.proj = nn.Sequential(nn.Linear(embed_dim, proj_dim), nn.ReLU())
        self.gru = nn.GRU(proj_dim, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_class)

    def forward(self, x, h0=None):
        x = (x - self.mean) / self.std
        x = self.proj(x)
        out, h = self.gru(x, h0)
        return self.head(out), h


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FineTuneCNNGRU(nn.Module):
    """End-to-end-trainable counterpart to EmbeddingGRU: same proj+GRU+head,
    but the CNN backbone lives INSIDE the autograd graph — specifically only
    ResNet18's last residual block (``layer4``) is unfrozen, everything
    before it (stem, layer1-3) stays frozen, the standard "fine-tune the
    last block" transfer-learning recipe (full end-to-end fine-tuning of an
    11M-param backbone on 10 drivers' faces was judged too high an
    overfitting risk to try first — see docs/METHODOLOGY.md §14.5). Input is
    raw face crops (uint8, BGR, HWC — straight from extract_crops.py's
    cached .npz), not pre-computed embeddings; there's no separate caching
    step for this variant since the backbone's weights change during
    training.
    """
    def __init__(self, n_class: int, proj_dim: int = 128, hidden: int = 64,
                layers: int = 2, dropout: float = 0.2, unfreeze_last_block: bool = True):
        super().__init__()
        import torchvision
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        for p in backbone.parameters():
            p.requires_grad_(False)
        self._frozen_modules = [backbone.conv1, backbone.bn1, backbone.layer1,
                                backbone.layer2, backbone.layer3]
        if unfreeze_last_block:
            for p in backbone.layer4.parameters():
                p.requires_grad_(True)
        self.backbone = backbone
        self.embed_dim = 512
        self.register_buffer("img_mean", IMAGENET_MEAN)
        self.register_buffer("img_std", IMAGENET_STD)
        self.proj = nn.Sequential(nn.Linear(self.embed_dim, proj_dim), nn.ReLU())
        self.gru = nn.GRU(proj_dim, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_class)

    def train(self, mode: bool = True):
        """Keep the frozen prefix (conv1/bn1/layer1-3) in eval mode even
        when the rest of the model is in train mode, so those layers'
        BatchNorm running stats don't drift on our small, non-IID training
        batches — only layer4's (unfrozen) BN should track batch statistics
        during training."""
        super().train(mode)
        if mode:
            for m in self._frozen_modules:
                m.eval()
        return self

    def encode(self, crops_bgr_uint8: torch.Tensor) -> torch.Tensor:
        """crops_bgr_uint8: (N,H,W,3) uint8 tensor. Returns (N, proj_dim)."""
        x = crops_bgr_uint8.float().flip(dims=[-1]) / 255.0   # BGR -> RGB
        x = x.permute(0, 3, 1, 2)                              # NHWC -> NCHW
        x = (x - self.img_mean) / self.img_std
        return self.proj(self.backbone(x))

    def forward(self, crops_bgr_uint8: torch.Tensor, h0=None):
        """crops_bgr_uint8: (B,T,H,W,3) uint8. Returns (logits (B,T,n_class), h)."""
        B, T = crops_bgr_uint8.shape[:2]
        flat = crops_bgr_uint8.reshape(B * T, *crops_bgr_uint8.shape[2:])
        z = self.encode(flat).reshape(B, T, -1)
        out, h = self.gru(z, h0)
        return self.head(out), h


KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
KINETICS_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)


class R3DClipClassifier(nn.Module):
    """Lightweight 3D-CNN comparison baseline: torchvision's R3D-18
    (ResNet-3D, 18 layers, Kinetics-400-pretrained) — the literature's
    second-choice recommendation for our subject-count/compute scale
    (arXiv:2210.09441 found lightweight 3D-CNNs realistic on comparable
    hardware, though generally underperforming a 2D-per-frame+temporal
    approach; docs/METHODOLOGY.md §14.7 for the write-up).

    Unlike EmbeddingGRU/FineTuneCNNGRU, this is a CLIP classifier, not a
    per-frame sequence model: it consumes one short, fixed-length window of
    raw crops (default 16 frames, matching R3D-18's Kinetics pretraining
    convention) and predicts ONE label for that whole window — the standard
    way 3D-CNNs are used for action recognition, and structurally different
    from the GRU variants' per-timestep output. train_r3d.py broadcasts
    each window's single prediction across all frames in that window to get
    a frame-level prediction stream comparable to every other experiment's
    macro-F1 — a documented approximation, not a like-for-like temporal
    resolution match.

    Same "fine-tune the last block" recipe as FineTuneCNNGRU (only `layer4`
    unfrozen) for methodological consistency with the 2D experiment — R3D-18
    was pretrained on Kinetics-400 (varied outdoor/action video), a bigger
    domain gap to a static in-cabin camera than ImageNet-to-face-crops, so
    some adaptation is expected to matter here too.
    """
    def __init__(self, n_class: int, unfreeze_last_block: bool = True):
        super().__init__()
        import torchvision
        net = torchvision.models.video.r3d_18(
            weights=torchvision.models.video.R3D_18_Weights.KINETICS400_V1)
        net.fc = nn.Linear(net.fc.in_features, n_class)
        for p in net.parameters():
            p.requires_grad_(False)
        self._frozen_modules = [net.stem, net.layer1, net.layer2, net.layer3]
        if unfreeze_last_block:
            for p in net.layer4.parameters():
                p.requires_grad_(True)
        for p in net.fc.parameters():
            p.requires_grad_(True)
        self.net = net
        self.register_buffer("mean", KINETICS_MEAN)
        self.register_buffer("std", KINETICS_STD)

    def train(self, mode: bool = True):
        """Same rationale as FineTuneCNNGRU.train(): keep the frozen prefix
        in eval mode so its BatchNorm running stats don't drift."""
        super().train(mode)
        if mode:
            for m in self._frozen_modules:
                m.eval()
        return self

    def forward(self, clips_bgr_uint8: torch.Tensor) -> torch.Tensor:
        """clips_bgr_uint8: (N,T,H,W,3) uint8. Returns (N, n_class) logits."""
        x = clips_bgr_uint8.float().flip(dims=[-1]) / 255.0   # BGR -> RGB
        x = x.permute(0, 4, 1, 2, 3)                           # N,T,H,W,C -> N,C,T,H,W
        x = (x - self.mean) / self.std
        return self.net(x)


def build_chunks(sessions: List[dict], chunk_len: int, stride: int):
    """sessions: list of dicts with 'embeddings' (T,D) float32 and 'y' (T,)
    int64. Same fixed-length, non-overlapping-beyond-stride chunking scheme
    as train_sequence.py::build_chunks."""
    xs, ys = [], []
    for s in sessions:
        X, y = s["embeddings"], s["y"]
        T = len(X)
        if T < chunk_len:
            continue
        for start in range(0, T - chunk_len + 1, stride):
            xs.append(X[start:start + chunk_len])
            ys.append(y[start:start + chunk_len])
    return np.stack(xs), np.stack(ys)


def full_sequence_eval(model: EmbeddingGRU, sessions: List[dict], dev: str) -> Dict[str, np.ndarray]:
    """One continuous forward pass per session (matches live/streaming use
    and train_sequence.py::full_sequence_eval), keyed by session name."""
    model.eval()
    out = {}
    with torch.no_grad():
        for s in sessions:
            X = torch.tensor(s["embeddings"], dtype=torch.float32).unsqueeze(0).to(dev)
            logits, _ = model(X)
            out[s["name"]] = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    return out


def full_sequence_eval_finetune(model: FineTuneCNNGRU, sessions: List[dict], dev: str,
                                encode_batch: int = 512) -> Dict[str, np.ndarray]:
    """Same idea as full_sequence_eval but for FineTuneCNNGRU: the CNN
    encode step is run batched (no_grad, so memory-cheap regardless of
    batch size — no activations need to be kept for backprop) over a whole
    session's raw crops to build its embedding sequence, then the GRU runs
    as a single full-session forward pass exactly like full_sequence_eval.
    Uses the model's CURRENT (fine-tuned) backbone weights, not a cached
    frozen embedding, so this reflects true post-training performance."""
    model.eval()
    out = {}
    with torch.no_grad():
        for s in sessions:
            crops = torch.from_numpy(s["crops"]).to(dev)
            zs = [model.encode(crops[i:i + encode_batch]) for i in range(0, len(crops), encode_batch)]
            z = torch.cat(zs, dim=0).unsqueeze(0)
            logits, _ = model.gru(z)
            logits = model.head(logits)
            out[s["name"]] = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    return out
