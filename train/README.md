# Training pipeline (PC, RTX 4070 Ti Super)

PC-side work: extract features from the DMD dataset with a **pretrained model
cascade**, validate that the cascade actually works on DMD, then train the
Stage-④ driver-state classifier on the extracted features.

This mirrors the production target in the root `CLAUDE.md` — each stage is a real
pretrained model, replacing a fragile geometric heuristic from the MediaPipe
prototype.

```
DMD *_rgb_face.mp4
      │
      ▼
 ① SCRFD-500M (InsightFace)  ── face box + 5 keypoints ─┐
      │                                                  │
      ├─▶ ② 6DRepNet            ── yaw / pitch / roll     │  per-frame
      └─▶ ③ open-closed-eye-0001 ── eye open/closed ──────┤  FrameFeatures
                                       │                  │
                                       ▼                  │
                              temporal: PERCLOS, blink ───┘
      │
      ▼
  train/output/features/<session>_features.csv   (features + DMD ground truth)
```

## Layout

```
train/
  cascade/              the pretrained-model cascade (pluggable wrappers)
    face_detector.py    ① SCRFD via InsightFace (buffalo_sc = SCRFD-500M)
    head_pose.py        ② 6DRepNet
    eye_state.py        ③ open-closed-eye-0001 (ONNX)
    gaze.py             ⑤ gaze-estimation-adas-0002 (OpenVINO IR) — eye gaze
    temporal.py         PERCLOS / blink aggregation
    pipeline.py         orchestrator -> FrameFeatures
    base.py             shared dataclasses
  dmd/                  DMD dataset utilities
    annotations.py      OpenLABEL JSON -> per-frame ground-truth labels
    dataset.py          discover sessions (face video + annotation pairs)
  extract_features.py   CLI: run cascade over DMD -> labeled feature CSVs
  validate.py           compare extracted features vs DMD ground truth
  build_dataset.py      concat CSVs + derive 4-state labels -> train_table.csv
  train_state.py        train the Stage-④ MLP (features -> state) + export ONNX
  run_live.py           real-time demo: camera -> cascade -> classifier -> overlay
  download_models.py    fetch the eye-state ONNX
  output/               extracted CSVs / annotated videos  (gitignored)
```

Cascade parameters live in the `cascade:` section of the root `config.yaml`.

## Setup

```bash
conda env create -f train/environment.yml
conda activate dms-train
python -c "import torch; print(torch.cuda.get_device_name(0))"   # verify GPU
```

Fetch the eye-state weights (SCRFD + 6DRepNet auto-download on first run):

```bash
python -m train.download_models
```

## DMD data

Extract the DMD `.tar.gz` archives in place under `datasets/DMD/<task>/`. The
pipeline auto-discovers any extracted session:

```
datasets/DMD/drowsiness/dmd-dataset-drowsiness-gA-1/dmd/gA/1/s5/
    gA_1_s5_<ts>_rgb_face.mp4            <- inferred on
    gA_1_s5_<ts>_rgb_ann_drowsiness.json <- ground truth (eyes / blink / yawn)
```

The **drowsiness** subset carries the eye-state / blink / yawn labels this first
pass validates against. Distraction and gaze subsets are discovered too but have
different label sets.

## Run

```bash
# 1) Smoke test — one session, first 1500 frames
python -m train.extract_features --task drowsiness --limit-sessions 1 --max-frames 1500

# 2) Check the cascade against DMD ground truth
python -m train.validate train/output/features

# 3) Full extraction, all sessions, every 2nd frame (~59 fps on GPU)
python -m train.extract_features --task drowsiness  --stride 2
python -m train.extract_features --task distraction --stride 2

# 4) Build the combined Stage-④ training table (4-state labels from GT)
python -m train.build_dataset          # -> train/output/train_table.csv + class balance

# 5) Train the Stage-④ classifier (features -> state), export ONNX
python -m train.train_state                   # 4-state -> models/state_classifier/state_mlp.{pt,onnx}
python -m train.train_state --classes binary  # binary  -> models/state_classifier/binary/state_mlp.{pt,onnx}
python -m train.train_state --classes three --loss focal --weight-power 0.3 --smooth-window 30
                                                # 3-state (best config so far) -> models/state_classifier/three/state_mlp.{pt,onnx}

# 6) Test the whole pipeline live on a webcam
python -m train.run_live               # camera -> cascade -> classifier -> live overlay
```

### Class modes (`train_state.py --classes`)

| Mode | Classes | Balance | Why |
|---|---|---|---|
| `four` (default) | FOCUSED / DISTRACTED / DROWSY / TIRED | 46 / 48 / 2.8 / 2.7 % | full taxonomy (the project goal) |
| `three` | FOCUSED / DISTRACTED / FATIGUED | 46 / 48 / 5.5 % | DROWSY+TIRED merged into one fatigue class ("FATIGUED", not "TIRED" — TIRED already names a specific state in the 4-class taxonomy); keeps distraction and fatigue separate, unlike `binary` |
| `binary` | ATTENTIVE / INATTENTIVE | ~46 / 54 % | collapses all non-FOCUSED states into one "inattention" class — sidesteps the DROWSY/TIRED imbalance and gives a nearly-balanced comparison baseline |

All three modes use the **same driver split** (same seed, same held-out
drivers), so their held-out results are directly comparable. The binary run
also reports ROC-AUC and INATTENTIVE recall broken down by the original
4-state label; the `three` run reports FATIGUED recall broken down by
DROWSY/TIRED — so you can see which behaviours the collapsed class actually
catches. Results + discussion: `docs/METHODOLOGY.md` §8.

### Tuning flags (`train_state.py`)

| Flag | Effect |
|---|---|
| `--smooth-window N` | also reports a temporally-smoothed eval: rolling-mean the predicted class probabilities over N frames per session (sorted by frame), then argmax — same scheme as `run_live.py`'s `--smooth`, run offline first to measure the effect before touching the live demo |
| `--weight-power P` | exponent on the inverse-frequency class weights (`1.0` = old behavior, `0.0` = uniform/no weighting) — the default `1.0` turned out to over-fire the rare classes; `0.3` worked much better, see §8.3 |
| `--loss {ce,focal}` | `focal` down-weights already-easy examples on top of the class weights (`--focal-gamma`, default 2.0) |
| `--model {mlp,gbt}` | `gbt` trains a `HistGradientBoostingClassifier` (sklearn) on the same split/features as a model-family comparison baseline; not exported (no ONNX path yet) |

### Live demo (`run_live.py`)

Real-time end-to-end test of the trained model + full cascade:

```bash
python -m train.run_live                    # default webcam (index 0)
python -m train.run_live --source 1         # a different camera
python -m train.run_live --source clip.mp4  # a video file instead
python -m train.run_live --mirror           # driver-view (flip display)
python -m train.run_live --smooth 20        # longer temporal smoothing (steadier)
python -m train.run_live --classifier models/state_classifier/binary/state_mlp.pt
                                            # binary ATTENTIVE/INATTENTIVE model
```

The class set is read from the checkpoint, so the demo works with either the
4-state or the binary model without extra flags.

Shows the predicted **state** (big, colour-coded) + confidence, per-class
probability bars, the live features (eye/PERCLOS/head/gaze) and a gaze arrow.
Press **Esc**/**q** to quit. Run it in your own terminal (it needs the webcam +
a window). The default 4-state classifier is the original per-frame baseline
(macro-F1 0.35, expect flicker / false fatigue alarms); the offline
experiments in `docs/METHODOLOGY.md` §8.3 (temporal features, tuned weighting,
`three`-class FATIGUED) reach macro-F1 0.60 on held-out drivers but are not
yet wired into this live overlay — `--classifier` still loads the checkpoint's
own feature list (see `StateClassifier`), but the 4 new rolling features
(`perclos_15s`, `yaw_std_5s`, `gaze_yaw_std_5s`, `eye_open_prob_mean_5s`)
aren't computed by the live cascade yet, so they fall back to a neutral value
rather than actually helping here.

Each session writes `train/output/features/<session>_features.csv` — one row per
processed frame with the extracted features plus `gt_*` ground-truth columns.

### Useful flags (`extract_features.py`)

| Flag | Effect |
|---|---|
| `--task drowsiness` | DMD subset to walk (`''` for all) |
| `--video <path>` | run a single explicit `*_rgb_face.mp4` |
| `--stride N` | process every Nth frame |
| `--max-frames N` | cap frames per session (quick tests) |
| `--limit-sessions N` | process at most N sessions |
| `--annotate` | also write overlay `*_annotated.mp4` for visual QA |
| `--mirror` | flip the annotated video to **driver view** (display only — feature values in the CSV are unchanged; the gaze arrow then reads intuitively with the eyes) |

## What "works well" means

`validate.py` reports, against the labels DMD provides:

- **face coverage** — SCRFD should detect the driver in ~all non-occluded frames
- **eye-state agreement** — model open/closed vs GT `eyes_state`, plus the best
  decision threshold on `eye_open_prob`
- **blink counting** — cascade blink events vs GT blink intervals
- **head-pose sanity** — yaw/pitch ranges (the drowsiness set has no pose GT)

If eye-state agreement and blink counts line up with DMD, the cascade is sound →
next step is training the Stage-④ classifier on the feature CSVs to emit the four
driver states (FOCUSED / DISTRACTED / DROWSY / TIRED).

### First validation run (session gA_1_s5, 1500 frames, CPU)

| Signal | Result |
|---|---|
| Face detection (SCRFD) | 100% of frames |
| Eye state vs GT | 94.9% acc @ thresh 0.5; **98.2% @ 0.15**; closed recall 98.6% |
| Blink count | cascade 34 vs GT 25 (same ballpark) |
| Head pose (6DRepNet) | yaw −12..+68°, pitch −6..+18° (no GT in drowsiness set) |

Takeaway: the eye-state model tracks DMD ground truth closely. The default
`temporal.eye_closed_thresh: 0.5` over-counts closed frames a bit (lower
precision, slight blink over-count) — `validate.py` reports the best threshold on
`eye_open_prob` (~0.15 here). Calibrate `eye_closed_thresh` across several
sessions (and later on our own rig clips) rather than hard-coding one session's
optimum.

> **GPU is set up** (RTX 4070 Ti Super): torch cu124 + `onnxruntime-gpu==1.22.0`
> (CUDA 12) → the full cascade runs at **~59 fps** (4× the ~15 fps CPU baseline);
> GPU vs CPU features are identical. `train/cascade/__init__.py` adds torch's
> `lib/` to the DLL path so onnxruntime finds cuDNN 9 / cuBLAS 12. Gaze
> (OpenVINO) stays on CPU — it's tiny. Pin note: `onnxruntime-gpu==1.22.0`
> matches CUDA 12; 1.27 needs CUDA 13 and won't load its CUDA provider here.

### Distraction set — head pose + gaze vs `gaze_on_road` GT (session gA_1_s1, 900 frames)

The distraction subset labels `gaze_on_road/{looking_road,not_looking_road}` and
driver activities — so it validates the DISTRACTED signal directly.

| Predictor | Accuracy (off-road vs on-road) |
|---|---|
| Head pose `|yaw|` alone | 80.4% |
| **Gaze model `|gaze|`** | **94.4%** (on-road 11° vs off-road 29° gaze deviation) |
| Head pose + gaze combined | ~93% |

Takeaway: **gaze adds ~14 points over head pose** because it catches eyes-only
glances (radio, reach-side) where the head stays roughly forward — exactly the
case head pose alone misses. Gaze (stage ⑤) is therefore worth keeping as a
DISTRACTED feature, despite being "optional" in `CLAUDE.md`.

> Gaze convention (verified against known-direction frames, handled in
> `cascade/gaze.py`):
> - feed `head_pose_angles = [yaw, -pitch, roll]` (6DRepNet's pitch sign is
>   inverted vs the Intel head-pose model the gaze net expects);
> - the model emits gaze in an anatomical frame (subject's right = +x), so flip
>   the x-component to image coordinates (`gv[0] = -gv[0]`);
> - forward axis is −z, so deviation angles use `atan2(component, -gz)`.
>
> Resulting feature signs: `gaze_yaw` + = image-right / − = image-left;
> `gaze_pitch` − = down / + = up. Sign flips don't change `|gaze|`, so the 94.4%
> validation is unaffected.
>
> Head pose is stored in the **same image frame** so head and gaze read
> consistently: 6DRepNet's yaw is mirrored vs the image (its +yaw = image-left),
> so the stored `yaw` is negated; its pitch already matches (−pitch = down). The
> gaze stage still receives the raw 6DRepNet angles internally. Net: for both
> head and gaze, **+yaw = image-right, −pitch = down**, so head and eyes share one
> convention (verified against known-direction frames).

## Notes

- `onnxruntime-gpu` + InsightFace use the CUDA EP; set `cascade.use_gpu: false`
  in `config.yaml` to force CPU.
- Feature extraction is **offline** — it does not need to hit real-time. Only the
  deployed Jetson cascade does (and is built for it; see `CLAUDE.md`).
- Gaze (L2CS-Net) is intentionally **not** in this cascade — head pose covers the
  "looking away" signal for DISTRACTED; gaze is deferred per `CLAUDE.md`.
