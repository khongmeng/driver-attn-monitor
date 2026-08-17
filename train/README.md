# Training pipeline (PC)

This is where the project's actual results come from: a **feature-extraction
cascade** (pretrained models, no training) turns DMD driver videos into a
per-frame feature table, then a **Stage-④ classifier** (the part we train)
turns that feature table into a driver state.

```
DMD *_rgb_face.mp4
  │
  ▼
 ① SCRFD-500M            face box + 5 keypoints
  │
  ├─▶ ② 6DRepNet          yaw / pitch / roll
  ├─▶ ③ open-closed-eye   eye open/closed
  ├─▶ ⑤ gaze-adas-0002    gaze yaw / pitch     (uses ② + eye crops)
  └─▶ mouth (2d106det)    Mouth Aspect Ratio → yawn
  │
  ▼
 temporal: PERCLOS, blink rate, yawn rate, rolling-window stats
  │
  ▼
 train/output/features/<session>_features.csv   (features + DMD ground truth)
  │
  ▼
 train/output/train_table.csv                    (all sessions, labeled)
  │
  ▼
 Stage-④ classifier (10 architectures compared — see models/README.md)
```

Every stage above ①–⑤ is a real pretrained model — this mirrors the
production target described in the root `CLAUDE.md`, not the geometric
heuristics `inference/` still uses on the Jetson today.

## Layout

```
train/
  cascade/                 the pretrained-model cascade (pluggable stages)
    face_detector.py       ① SCRFD-500M (InsightFace buffalo_sc)
    head_pose.py           ② 6DRepNet
    eye_state.py           ③ open-closed-eye-0001 (ONNX)
    gaze.py                ⑤ gaze-estimation-adas-0002 (OpenVINO IR)
    mouth.py                  2d106det landmarks → Mouth Aspect Ratio
    temporal.py             PERCLOS / blink / yawn aggregation
    pipeline.py             orchestrator → FrameFeatures
    base.py                 shared dataclasses
  dmd/                      DMD dataset utilities
    annotations.py          OpenLABEL JSON → per-frame ground-truth labels
    dataset.py               discover sessions (face video + annotation pairs)
  rawpixel/                 SEPARATE pipeline: raw pixels → CNN → GRU,
                             no hand-engineered features. Deliberately
                             isolated — see rawpixel/README.md.

  extract_features.py       CLI: run the cascade over DMD → labeled feature CSVs
  validate.py                compare extracted features vs DMD ground truth
  build_dataset.py           concat CSVs + derive labels → train_table.csv
  download_models.py         fetch cascade weights that don't auto-download
  gt_overlay.py               play DMD ground truth burned into video (sanity check)

  train_state.py             Stage-④: per-frame classifiers (MLP / SVM / RF / GBT)
  train_sequence.py          Stage-④: sequence classifiers (GRU / TCN / gated-GRU)
  train_sequence_cv.py        driver cross-validation ensemble variant
  eval_ensemble.py            offline eval of a multi-checkpoint ensemble
  run_live.py                 real-time demo: camera → cascade → classifier → overlay

  output/                    extracted CSVs / logs / demo videos  (gitignored)
```

Cascade parameters live in the `cascade:` section of the root `config.yaml`.

## 1. Set up the environment

```bash
conda env create -f train/environment.yml
conda activate dms-train
python -c "import torch; print(torch.cuda.get_device_name(0))"   # verify GPU
```

## 2. Fetch cascade weights

SCRFD and 6DRepNet auto-download the first time the cascade runs. Everything
else (eye-state, gaze, mouth-landmark) needs one command:

```bash
python -m train.download_models
```

## 3. Get the DMD dataset

Extract the DMD `.tar.gz` archives in place under `datasets/DMD/<task>/`
(gitignored — not part of this repo). The pipeline auto-discovers any
extracted session:

```
datasets/DMD/drowsiness/dmd-dataset-drowsiness-gA-1/dmd/gA/1/s5/
    gA_1_s5_<ts>_rgb_face.mp4              <- inferred on
    gA_1_s5_<ts>_rgb_ann_drowsiness.json   <- ground truth (eyes/blink/yawn)
```

The **drowsiness** subset carries eye-state/blink/yawn labels; the
**distraction** subset carries gaze/driver-action labels. Both are used —
see `docs/METHODOLOGY.md` for how each maps to the 4 states.

## 4. Extract features

```bash
# smoke test — one session, first 1500 frames
python -m train.extract_features --task drowsiness --limit-sessions 1 --max-frames 1500

# check the cascade against DMD ground truth
python -m train.validate train/output/features

# full extraction, all sessions, every 2nd frame
python -m train.extract_features --task drowsiness  --stride 2
python -m train.extract_features --task distraction --stride 2
```

`validate.py` reports (against whichever labels that DMD subset provides):
face-detection coverage, eye-state agreement (+ best decision threshold),
blink-event counting, head-pose range sanity. Run this before trusting a
fresh extraction — see `docs/METHODOLOGY.md` §6.1 for the kind of bugs this
step has actually caught in the past (not hypothetical).

## 5. Build the training table

```bash
python -m train.build_dataset
# -> train/output/train_table.csv + a class-balance report
```

Concatenates every session's feature CSV, derives the state label from DMD
ground truth (drowsiness sessions: yawn/PERCLOS-based; distraction sessions:
gaze/driver-action-based — see the docstring in `build_dataset.py` for the
exact rule), and adds a few rolling-window features on top of the raw
per-frame cascade output (so they're available live, not just offline).

## 6. Train a classifier

Two scripts, depending on architecture family:

```bash
# per-frame classifiers (MLP / SVM / Random Forest / GBT)
python -m train.train_state --classes three --model mlp

# sequence classifiers (GRU / TCN / gated-GRU) — the stronger family
python -m train.train_sequence --classes three --loss focal --weight-power 0.3 \
    --oversample 5.0 --smooth-window 30 --epochs 200 --eval-every 1
```

We compared **10 architectures** this way — see **`models/README.md`** for
the full comparison table, exact reproduce-command per architecture, and
which checkpoint to actually use (short answer: `models/gru_ensemble/`,
macro-F1 0.810). Full experiment writeups, every hyperparameter, and every
per-class result: `docs/METHODOLOGY.md`.

Both scripts share: `--classes {four,three,binary}` (which state grouping to
train), `--weight-power` (dampens inverse-frequency class weighting —
`1.0` over-corrects, `0.3` works much better), `--loss {ce,focal}`,
`--smooth-window N` (report a temporally-smoothed eval alongside the raw
one). `train_sequence.py` additionally supports `--model {gru,tcn}`,
`--gated-fatigue` (the dual-branch architecture that's the best single
model), and `--oversample` (chunk-level oversampling for sequence data —
row-level resampling would break temporal continuity, don't use
`train_state.py`'s `--balance undersample` idea here).

## 7. Try the trained pipeline live

```bash
python -m train.run_live                                    # webcam
python -m train.run_live --source clip.mp4                  # a video file
python -m train.run_live --mirror                            # driver-view display
python -m train.run_live --classifier models/gru_gated/state_gru.pt \
                          models/gru_single/state_gru.pt      # the ensemble
```

The class set and architecture (MLP vs. GRU vs. gated-GRU) are both read
from the checkpoint automatically — no extra flags needed to switch models.
Pass multiple `--classifier` paths to run an ensemble (softmax-probability
averaging across all of them, every frame).

## The raw-pixel comparison line

`train/rawpixel/` is a second, deliberately **isolated** Stage-④ line: raw
face-crop video → CNN embedding → GRU, no hand-engineered features at all.
Built to test whether a learned representation beats hand-engineered
features at this data scale — **it doesn't** (see `docs/METHODOLOGY.md`
§14), but it's kept as a real, documented negative result and a genuine
architecture in the comparison table. See `train/rawpixel/README.md` for
why it's kept separate and how to run it.

## Notes

- `onnxruntime-gpu` + InsightFace use the CUDA execution provider by
  default; set `cascade.use_gpu: false` in `config.yaml` to force CPU.
- GPU setup (RTX 4070 Ti Super): torch cu124 + `onnxruntime-gpu==1.22.0`
  (CUDA 12) gets the full cascade to ~59 fps (vs. ~15 fps CPU). Version
  match matters — `onnxruntime-gpu` 1.27+ needs CUDA 13 and won't load its
  CUDA provider against a cu124 torch install. Gaze (OpenVINO) stays on
  CPU regardless — the model is tiny, not worth the extra GPU plumbing.
- Feature extraction is **offline** — it doesn't need to hit real-time.
  Only the eventual deployed Jetson cascade does.
- Gaze convention (verified against known-direction frames,
  `cascade/gaze.py`): feed `head_pose_angles=[yaw, -pitch, roll]` (6DRepNet's
  pitch sign is inverted vs. the Intel head-pose model the gaze net expects);
  flip the output x-component to image coordinates (`gv[0] = -gv[0]`);
  forward axis is −z. Resulting convention: `gaze_yaw` + = image-right,
  `gaze_pitch` − = down — head pose is stored in this same frame so the two
  signals agree.
