"""Class distribution chart for the dataset, showing why DROWSY/TIRED had
to be merged into FATIGUED (each is too small a slice on its own) and why
macro-F1 (not accuracy) is the right metric here. Real counts from
train/output/train_table.csv.

Usage:
    python docs/paper/gen_class_distribution.py
"""
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figure", "class_distribution.png")

STATES = ["DISTRACTED", "FOCUSED", "TIRED", "DROWSY"]
COUNTS = [105191, 97438, 6214, 5837]
TOTAL = sum(COUNTS)
COLORS = ["#2a78d6", "#2a78d6", "#c0392b", "#c0392b"]

fig, ax = plt.subplots(figsize=(5.2, 3.6))
bars = ax.bar(STATES, COUNTS, color=COLORS)
for bar, count in zip(bars, COUNTS):
    pct = 100 * count / TOTAL
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1500,
            f"{pct:.1f}%", ha="center", fontsize=9)

ax.set_ylabel("frame count")
ax.set_title("Label distribution before merging\nDROWSY + TIRED into FATIGUED", fontsize=11)
ax.text(2.5, 30000, "merged into\nFATIGUED\n(5.6% combined)",
        ha="center", fontsize=8.5, color="#c0392b")
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"saved {OUT}")
