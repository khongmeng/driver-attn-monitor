# Model checkpoints

This directory holds two kinds of things:

1. **Cascade weights** (`eye_state/`, `gaze/`, `landmark/`) — pretrained models
   used unchanged by the feature-extraction cascade (see `train/README.md`).
   These are fetched automatically by `python -m train.download_models`, so
   they're **not** committed to git — just re-run that command after cloning.
   (SCRFD and 6DRepNet, the other two cascade stages, auto-download through
   their own packages the first time the cascade runs — nothing to do for
   those either.)

2. **Stage-④ classifiers** — the driver-state classifiers *we* trained, one
   folder per architecture. This is the part covered below.

## Which one should I actually use?

**`gru_ensemble/`** — macro-F1 **0.810**, the best result in the project.
If you only need one model, use this one. It combines two checkpoints
(`gru_gated/` + `gru_single/`) by averaging their predicted probabilities —
see `gru_ensemble/README.md` for exact usage.

## The full comparison set

We trained and compared **10 architectures**, all on the identical
driver-independent split (10 training drivers / 4 held-out test drivers —
`gA_1, gC_14, gE_28, gZ_37` — never seen during training or checkpoint
selection). This is the accuracy-vs-compute sweep the paper's model
comparison is built on. Every entry below was trained for the model's full
epoch budget (200 epochs for every iterative/neural model, so results are
comparable — see the note on each row); where the same architecture was
tried with multiple hyperparameter tweaks, only the best-performing
checkpoint was kept.

| folder | architecture | macro-F1 | input | source |
|---|---|---|---|---|
| *(none — see note)* | SVM (RBF kernel) | 0.531 | 13 hand-engineered features | METHODOLOGY.md §8.15, §15 |
| *(none — see note)* | Random Forest | 0.599 | 13 hand-engineered features | METHODOLOGY.md §8.15, §15 |
| *(none — see note)* | Gradient Boosted Trees | 0.620 | 13 hand-engineered features | METHODOLOGY.md §8.15, §15 |
| `mlp/` | MLP (per-frame) | 0.599 | 13 hand-engineered features | METHODOLOGY.md §8.15, §15 |
| `tcn/` | TCN (temporal conv. net) | 0.720 | 13 hand-engineered features, sequence | METHODOLOGY.md §8.15, §14.9 |
| `r3d18/` | R3D-18 (3D-CNN) | 0.712 | raw face-crop video clips | METHODOLOGY.md §14.7, §14.9 |
| `cnn_gru_finetuned/` | ResNet18 (fine-tuned) + GRU | 0.742 | raw face-crop frames, sequence | METHODOLOGY.md §14.5, §14.8, §14.10 |
| `gru_single/` | GRU (single-branch) | 0.790 | 13 hand-engineered features, sequence | METHODOLOGY.md §8.7, §8.16 |
| `gru_gated/` | Gated-Fatigue GRU (dual-branch) | 0.798 | 13 hand-engineered features, sequence | METHODOLOGY.md §8.10, §8.16 |
| `gru_ensemble/` | Ensemble of `gru_gated` + `gru_single` | **0.810** | (combines the two above) | METHODOLOGY.md §8.14, §8.16 |

*(numbers above are each model's best reported macro-F1 — raw for most,
smoothed where smoothing measurably helped that model; see METHODOLOGY.md
for the raw/smoothed breakdown per model)*

**SVM / Random Forest / GBT have no saved checkpoint** — by design, these
three were trained purely as fast (seconds, not minutes) comparison
baselines and were never exported (`train_state.py` prints "model not
exported (comparison baseline only)" for these). Their results are fully
reproducible via the one-line commands below (deterministic given the
fixed seed); there's just no `.pt`/`.pkl` file sitting in this folder for
them.

**`cnn_gru_finetuned/` is a real, slightly counterintuitive result worth
knowing about:** at its original 15-epoch budget this architecture scored
0.753 — extending it to 200 epochs (to match every other model's budget)
actually landed *lower*, at 0.742 (best epoch 81/200). Unlike TCN/R3D-18
(which were stable or improved with more epochs), this one just gets
noisier with more training — consistent with it having by far the largest
trainable parameter count of any kept model (8.5M, ResNet18's last block)
against only 638 training chunks. Kept at 0.742 anyway, deliberately, for
apples-to-apples budget comparison with every other entry — see
METHODOLOGY.md §14.10 for the full curve and discussion.

## Two families, same comparison

- **Hand-engineered features** (SVM, Random Forest, GBT, `mlp/`,
  `tcn/`, `gru_single/`, `gru_gated/`, `gru_ensemble/`) — input is the
  13-column feature table from the cascade (head pose, PERCLOS, blink rate,
  gaze, MAR/yawn — see `train/FEATURES.md`). This is the main line of the
  project.
- **Raw pixels** (`r3d18/`, `cnn_gru_finetuned/`) — input is face-crop video
  directly, no hand-engineered features at all. Built specifically to test
  whether a learned CNN representation beats hand-engineered features at
  this data scale (14 drivers). **It doesn't** — every raw-pixel result sits
  below the hand-feature GRU family. Kept in the comparison because that's
  itself a real, documented finding (METHODOLOGY.md §14), not because it's
  the recommended path.

## Reproducing a checkpoint

Every checkpoint here was produced by one of the training scripts in
`train/` — see `train/README.md` for the full walkthrough. The short
version, per architecture:

| model | reproduce with |
|---|---|
| SVM | `python -m train.train_state --classes three --model svm` |
| Random Forest | `python -m train.train_state --classes three --model rf` |
| GBT | `python -m train.train_state --classes three --model gbt` |
| `mlp/` | `python -m train.train_state --classes three` |
| `tcn/` | `python -m train.train_sequence --classes three --model tcn --epochs 200 --eval-every 1` |
| `r3d18/` | `python -m train.rawpixel.train_r3d --epochs 200 --eval-every 1` |
| `cnn_gru_finetuned/` | `python -m train.rawpixel.train_cnn_gru_finetune --epochs 200 --eval-every 1` |
| `gru_single/` | `python -m train.train_sequence --classes three --loss focal --weight-power 0.3 --oversample 5.0 --smooth-window 30 --epochs 200 --eval-every 1` |
| `gru_gated/` | same as `gru_single/` + `--gated-fatigue` |
| `gru_ensemble/` | `python -m train.eval_ensemble models/gru_gated/state_gru.pt models/gru_single/state_gru.pt` |

## Committed vs. gitignored

Only `gru_gated/` and `gru_single/` have their `.pt` weight files committed
to git (needed for `gru_ensemble/`, the recommended model) — that's so a
fresh clone (e.g. on the Jetson) can run the best model immediately without
retraining. Every other checkpoint here is gitignored and reproducible via
the commands above; re-run the matching command if you need one of them.
