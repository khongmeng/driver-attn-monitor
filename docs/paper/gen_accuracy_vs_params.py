"""Accuracy (macro-F1) vs. classifier parameter count, log-x, for all
architectures with a real parameter count (classical ML baselines have no
comparable "params" so they are excluded from this specific plot -- they
are still in the main results table).

Usage:
    python docs/paper/gen_accuracy_vs_params.py
"""
import os

import matplotlib.pyplot as plt

DATA = [
    # name, params, macro-F1, family
    ("MLP", 3_101, 0.599, "hand-features"),
    ("TCN", 139_549, 0.720, "hand-features"),
    ("GRU (single)", 40_349, 0.790, "hand-features"),
    ("Gated-Fatigue GRU", 44_260, 0.798, "hand-features"),
    ("Ensemble", 84_609, 0.810, "hand-features"),
    ("ResNet18+GRU", 11_314_205, 0.742, "raw-pixel"),
    ("R3D-18", 33_177_437, 0.712, "raw-pixel"),
]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figure", "accuracy_vs_params.png")

fig, ax = plt.subplots(figsize=(5.2, 4.0))
colors = {"hand-features": "#2a78d6", "raw-pixel": "#c0392b"}
seen = set()
for name, params, f1, family in DATA:
    label = family if family not in seen else None
    seen.add(family)
    ax.scatter(params, f1, color=colors[family], s=70, zorder=3, label=label)
    ax.annotate(name, (params, f1), textcoords="offset points", xytext=(6, 5), fontsize=8)

ax.set_xscale("log")
ax.set_xlabel("classifier parameters (log scale)")
ax.set_ylabel("macro-F1")
ax.set_title("Accuracy vs. classifier size")
ax.grid(alpha=0.3, which="both")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"saved {OUT}")
