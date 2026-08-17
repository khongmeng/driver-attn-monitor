"""Cache frozen 2D-CNN embeddings for every extracted face crop.

Deliberately frozen (no gradient, no fine-tuning) — with only 14 drivers to
generalize across, and per the "In Defense of Image Pre-Training for
Spatiotemporal Recognition" finding that pretrained-2D-image features
transfer better than fine-tuning when video data is limited (see
docs/METHODOLOGY.md §14), full backbone fine-tuning is more likely to
overfit than help here. Downstream training (train_cnn_gru.py) only ever
sees these cached embeddings — the CNN forward pass happens exactly once.

Usage:
    python -m train.rawpixel.embed_backbone
    python -m train.rawpixel.embed_backbone --backbone resnet18
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_config(path: str = None) -> dict:
    path = path or os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("rawpixel", {})


def build_backbone(name: str):
    if name == "resnet18":
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        dim = 512
    elif name == "mobilenet_v3_small":
        m = torchvision.models.mobilenet_v3_small(
            weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier = nn.Identity()
        dim = 576
    else:
        raise ValueError(f"unknown backbone {name!r}")
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, dim


@torch.no_grad()
def embed_crops(model, crops_bgr: np.ndarray, dev: str, batch: int = 256) -> np.ndarray:
    """crops_bgr: (N,H,W,3) uint8, BGR (cv2 order). Returns (N,dim) float32."""
    mean, std = IMAGENET_MEAN.to(dev), IMAGENET_STD.to(dev)
    out = []
    for i in range(0, len(crops_bgr), batch):
        chunk = crops_bgr[i:i + batch]
        x = torch.from_numpy(chunk[..., ::-1].copy()).to(dev).float() / 255.0  # BGR->RGB
        x = x.permute(0, 3, 1, 2)
        x = (x - mean) / std
        emb = model(x)
        out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Cache frozen CNN embeddings for raw-pixel crops.")
    cfg = load_config()
    ap.add_argument("--crops-dir", default="train/output/rawpixel/crops")
    ap.add_argument("--out-dir", default="train/output/rawpixel/embeddings")
    ap.add_argument("--backbone", choices=["resnet18", "mobilenet_v3_small"],
                    default=cfg.get("backbone", {}).get("name", "resnet18"))
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, dim = build_backbone(args.backbone)
    model = model.to(dev)
    print(f"backbone: {args.backbone} (frozen, {dim}-dim embedding), device={dev}")

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.crops_dir, "*.npz")))
    if not files:
        raise SystemExit(f"No crop .npz files in {args.crops_dir}. Run extract_crops.py first.")

    started = time.time()
    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        emb = embed_crops(model, d["crops"], dev, args.batch)
        name = os.path.splitext(os.path.basename(f))[0]
        np.savez_compressed(
            os.path.join(args.out_dir, f"{name}.npz"),
            embeddings=emb, frame_idx=d["frame_idx"], state=d["state"],
            driver=str(d["driver"]), task=str(d["task"]),
        )
        print(f"  [{i+1}/{len(files)}] {name}: {len(d['crops'])} crops -> {emb.shape}")
    print(f"\nDone in {time.time()-started:.1f}s. Embeddings -> {args.out_dir}")


if __name__ == "__main__":
    main()
