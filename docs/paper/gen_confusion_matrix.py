"""Generates the final-ensemble confusion-matrix figure for the paper, from
the real numbers in train/output/log_ensemble_dense_v1.txt (raw per-frame,
not smoothed -- the number the paper reports, macro-F1 0.810).

Usage:
    python docs/paper/gen_confusion_matrix.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np

STATES = ["FOCUSED", "DISTRACTED", "FATIGUED"]
CM = np.array([
    [16873, 2985, 544],
    [3138, 22169, 359],
    [376, 0, 1720],
])
CM_NORM = CM / CM.sum(axis=1, keepdims=True)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figure", "confusion_matrix.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

fig, ax = plt.subplots(figsize=(4.6, 4.0))
im = ax.imshow(CM_NORM, cmap="Blues", vmin=0, vmax=1)

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(STATES)
ax.set_yticklabels(STATES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")

for i in range(3):
    for j in range(3):
        color = "white" if CM_NORM[i, j] > 0.5 else "black"
        ax.text(j, i, f"{CM[i, j]:,}\n({CM_NORM[i, j]:.0%})",
                ha="center", va="center", color=color, fontsize=9)

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized fraction")
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"saved {OUT}")
