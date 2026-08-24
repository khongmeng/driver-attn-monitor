"""Prints exact trainable/total parameter counts for every kept checkpoint,
for the paper's model-comparison table (params column).

Usage:
    python docs/paper/count_params.py
"""
import torch

CHECKPOINTS = {
    "mlp": "models/mlp/state_mlp.pt",
    "tcn": "models/tcn/state_tcn.pt",
    "r3d18": "models/r3d18/state_r3d.pt",
    "cnn_gru_finetuned": "models/cnn_gru_finetuned/state_gru.pt",
    "gru_single": "models/gru_single/state_gru.pt",
    "gru_gated": "models/gru_gated/state_gru.pt",
}

for name, path in CHECKPOINTS.items():
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    n = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    print(f"{name:20s} {n:>12,} params  ({path})")
