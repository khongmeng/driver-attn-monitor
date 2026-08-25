"""Accuracy (macro-F1) vs. measured on-device per-frame latency, log-x, for
every architecture. Companion to gen_accuracy_vs_params.py but on the axis
that actually matters for deployment: real latency measured on the Jetson
Orin Nano Super (MAXN). Latency is the stage-4 classifier only (the shared
cascade front-end is separate, see paper Section on embedded deployment).

Neural models timed in PyTorch (CUDA), classical in scikit-learn (CPU); the
very small feature heads (MLP, GRU) sit on a ~0.5 ms PyTorch kernel-launch
floor, so their spread understates the true compute gap -- see the paper's
Table II footnotes.

Usage:
    python docs/paper/gen_accuracy_vs_latency.py
"""
import os

import matplotlib.pyplot as plt

DATA = [
    # name, latency_ms, macro-F1, family
    ("SVM", 0.70, 0.531, "classical"),
    ("Random Forest", 88.0, 0.599, "classical"),
    ("GBT", 1.84, 0.620, "classical"),
    ("MLP", 0.51, 0.599, "hand-features"),
    ("TCN", 4.59, 0.720, "hand-features"),
    ("GRU (single)", 0.60, 0.790, "hand-features"),
    ("Gated-Fatigue GRU", 1.17, 0.798, "hand-features"),
    ("Ensemble", 1.77, 0.810, "hand-features"),
    ("ResNet18+GRU", 8.54, 0.742, "raw-pixel"),
    ("R3D-18", 18.4, 0.712, "raw-pixel"),
]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figure", "accuracy_vs_latency.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

fig, ax = plt.subplots(figsize=(5.2, 4.0))
colors = {"hand-features": "#2a78d6", "raw-pixel": "#c0392b", "classical": "#7f8c8d"}
markers = {"hand-features": "o", "raw-pixel": "s", "classical": "^"}
seen = set()
for name, lat, f1, family in DATA:
    label = family if family not in seen else None
    seen.add(family)
    ax.scatter(lat, f1, color=colors[family], marker=markers[family],
               s=70, zorder=3, label=label)
    # nudge labels that would otherwise collide with each other or the legend
    nudge = {
        "MLP": (6, -12),
        "GRU (single)": (-8, 7),
        "Gated-Fatigue GRU": (-6, -14),
        "Ensemble": (6, 6),
        "SVM": (8, 3),
        "GBT": (6, 5),
    }
    dx, dy = nudge.get(name, (6, 5))
    ax.annotate(name, (lat, f1), textcoords="offset points", xytext=(dx, dy), fontsize=8)

ax.set_xscale("log")
ax.set_xlabel("on-device latency per frame (ms, log scale)")
ax.set_ylabel("macro-F1")
ax.set_title("Accuracy vs. measured latency (Jetson Orin Nano, MAXN)")
ax.grid(alpha=0.3, which="both")
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"saved {OUT}")
