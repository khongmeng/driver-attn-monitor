# Driver Monitoring System — Methodology & Paper Reference

Durable record of the full pipeline, parameters, models, datasets, conventions,
design decisions, and results — the source material for the paper. Update this as
the system evolves. Companion docs: `train/FEATURES.md` (feature-table schema),
`CLAUDE.md` (project overview), `train/README.md` (how to run).

_Last updated: 2026-07-22 (added temporal features + weight/loss tuning +
the three-class FATIGUED experiment, §8.3; previously: the binary
ATTENTIVE/INATTENTIVE experiment, §8.2; full feature extraction, training
table, Stage-④ 4-state baseline, live runtime demo)._

---

## 1. Objective

Real-time driver-state monitoring from a **single in-cabin RGB camera**,
classifying five outcomes: **FOCUSED, DISTRACTED, DROWSY, TIRED** (+ **NO_FACE**
fallback). Approach: **transfer learning / model composition** — a cascade of
pretrained models extracts interpretable features (head pose, eye state, gaze),
which feed a lightweight state classifier. No training from scratch.

**Target platform:** NVIDIA Jetson Orin Nano Super + ArduCam IMX477, real-time
(TensorRT FP16). **Development/training platform:** PC, RTX 4070 Ti Super.

---

## 2. Pipeline architecture (feature-extraction cascade)

```
RGB frame (1280×720)
  ① Face detection   → face box + 5 keypoints, NO_FACE gate
  ② Head pose        → yaw / pitch / roll        (from face crop)
  ③ Eye state        → open/closed prob per eye  (from eye crops)
  ⑤ Eye gaze         → gaze yaw / pitch          (eye crops + head pose)
  Temporal layer     → PERCLOS, blink rate       (over time)
  ④ State classifier → FOCUSED / DISTRACTED / DROWSY / TIRED   [baseline trained, §8.1]
```

Each stage is a swappable module (`train/cascade/`). Rationale for a **modular
cascade** over a single end-to-end network: debuggable per stage, needs far less
labeled data (only stage ④ is trained), interpretable features, and each stage
maps to an established pretrained model.

### Models (pretrained)

| Stage | Model | Source | Input | Output |
|---|---|---|---|---|
| ① Face | **SCRFD-500M** | InsightFace `buffalo_sc` (auto-download) | 640×640 | bbox + 5 kps |
| ② Head pose | **6DRepNet** | `pip install sixdrepnet` (0.1.6) | 224×224 crop | yaw, pitch, roll |
| ③ Eye state | **open-closed-eye-0001** | OpenVINO OMZ (ONNX) | 32×32×2 eyes | P(open) |
| ⑤ Gaze | **gaze-estimation-adas-0002** | OpenVINO OMZ (IR) | 2×60×60 eyes + head angles | 3D gaze vector |
| ④ State | MLP on features (planned) | trained here | 7-D feature vector | 4 states |

Model download URLs (public):
- eye: `storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/open-closed-eye-0001/open_closed_eye.onnx`
- gaze: `storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.{xml,bin}`
- SCRFD + 6DRepNet auto-download via their packages.

---

## 3. Feature extraction — per-frame + temporal

Per frame (`train/cascade/pipeline.py` → `FrameFeatures`): face box, det score,
head yaw/pitch/roll, per-eye + mean open-prob, eye-closed flag, gaze yaw/pitch +
3D vector. Temporal (`cascade/temporal.py`): PERCLOS (rolling window of the
eye-state *model* output), blink events/count/rate. Full schema in
`train/FEATURES.md`.

### Stage parameters

| Stage | Parameter | Value |
|---|---|---|
| ① SCRFD | det_size / det_thresh | 640 / 0.5 |
| ② 6DRepNet | crop margin around box | 0.25 |
| ③ eye | input / preprocessing | 32×32, RGB, `(x−127.5)/255` |
| ③ eye | eye-crop half-size | 0.30 × inter-ocular dist |
| ③ eye | closed threshold | eye_open_prob < 0.5 |
| ⑤ gaze | eye-crop half-size | 0.25 × inter-ocular dist |
| temporal | PERCLOS window / DROWSY thr (runtime) | 60 s / 0.20 |
| temporal | blink min consecutive closed frames | 1 |

**Preprocessing gotchas (verified, important for reproduction):**
- Eye model outputs a **softmax** already; feed RGB `(x−127.5)/255` — raw BGR 0–255
  makes the in-model softmax overflow to `[0, NaN]`.
- Gaze model: forward axis is **−z**; feed `head_pose_angles = [yaw, −pitch, roll]`
  (6DRepNet pitch sign is inverted vs the Intel head-pose convention it expects);
  flip output x (`gv[0] = −gv[0]`) — the model emits an anatomical vector
  (subject-right = +x), which is image-left on a driver-facing camera.

---

## 4. Coordinate conventions (verified against known-direction frames)

All head **and** gaze angles are in a single **image/camera frame**, degrees:

| | Negative | Positive |
|---|---|---|
| yaw | looking image-left | looking image-right |
| pitch | looking down | looking up |

- 6DRepNet's raw yaw is mirrored vs the image, so the stored `yaw = −pose.yaw`
  (`pipeline.py`); pitch already matches. The gaze model still receives the raw
  6DRepNet angles internally.
- Verified: whole-session head-vs-gaze yaw agreement **99.5%** (same sign when
  both |·|>10°); confirmed direction on zoomed eye crops (eyes-left→gaze<0, etc.).
- Note the mirror effect for demos: a driver-facing camera image is mirror-like,
  so overlays can be shown in "driver view" (`--mirror`, display-only) without
  changing feature values.

---

## 5. Dataset — DMD (Driver Monitoring Dataset, Vicomtech)

RGB in-cabin video, 1280×720. On disk: **65 sessions** = 16 **drowsiness** + 49
**distraction** (the gaze/hands subset was not used). CC BY-NC-ND, academic use.

### Ground-truth labels (temporal action annotations, OpenLABEL JSON)

**Drowsiness set:** `eyes_state` {open, close, opening, closing, undefined},
`blinks/blinking`, `yawning` {with hand, without hand}. + driver bbox.

**Distraction set:** `gaze_on_road` {looking_road, not_looking_road},
`driver_actions` {safe_drive, texting_left/right, phonecall_left/right, radio,
drinking, hair_and_makeup, reach_side, reach_backseat, talking_to_passenger,
change_gear, standstill_or_waiting, unclassified}, `hands_using_wheel`
{both, only_left, only_right, none}, `hand_on_gear`, `talking`. + object bboxes
{driver, cellphone, bottle, hair_comb}.

The two sets are **disjoint** — no session has both label families. The 4-state
set is assembled by combining them (§7).

### Extraction run

- Cascade run over all 65 face videos at **stride 2** (every 2nd frame).
- **227,056 frames** processed; per-session CSVs in `train/output/features/`.
- Throughput ≈ **59 fps** on the RTX 4070 Ti Super (see §9).

---

## 6. Feature/stage validation vs DMD ground truth

| Feature | Metric | Result |
|---|---|---|
| ① Face detection | coverage | **100%** of frames |
| ③ Eye state | acc vs `eyes_state` (open/close) | **~98%** (closed recall ~96%, best-thr 99%) |
| temporal | blink count vs GT intervals | same ballpark (e.g. 20 vs 16 on demo clip) |
| ② Head pose | `\|yaw\|` predicts `not_looking_road` | **~80%** |
| ⑤ Gaze | `\|gaze\|` predicts `not_looking_road` | **~94%** (on-road ~11° vs off-road ~29°) |

Head pose has no continuous GT in DMD (validated by range only); gaze/head are
validated via the distraction set's `gaze_on_road` labels. Gaze adds ~14 points
over head pose alone — it catches eyes-only glances the head misses.

---

## 7. State-label derivation (Stage-④ target)

Per face frame (`train/build_dataset.py`); NO_FACE excluded (detector gate).

**Drowsiness sessions:**
- `TIRED` if `gt_yawn == 1`
- else `DROWSY` if GT-PERCLOS ≥ **0.20** over a **15 s** window
- else `FOCUSED`

**Distraction sessions:**
- `DISTRACTED` if `gt_gaze_off_road == 1` OR `gt_driver_action` ∈ {texting_*,
  phonecall_*, radio, drinking, hair_and_makeup, reach_*, talking_to_passenger,
  change_gear}
- else `FOCUSED` if `gt_looking_road == 1` AND `gt_driver_action == safe_drive`
- else dropped (unclassified / standstill / ambiguous)

**DROWSY-window decision (important for the paper):** the clinical PERCLOS
definition (≥20% over **60 s**) yields only 2,505 DROWSY frames (5.7% of
drowsiness) because DMD's acted microsleeps are short — a 2 s microsleep is only
~3% PERCLOS in a 60 s window. Shortening to a **15 s** window (≥20%) captures the
microsleep timescale → **5,975** DROWSY frames without over-labeling normal
blinks (a 10% threshold would balloon to 41% and pull in blinking). Runtime
inference still uses the 60 s clinical window; the 15 s window is only for
labeling DMD's acted data.

### DMD drowsiness composition (why DROWSY is inherently a minority)
Of 44,114 drowsiness frames: eyes **open 64%**, opening 14%, closing 11%,
**close 10%**, undefined <1%. The set is mostly alert driving with brief drowsy
episodes — DROWSY being small is a property of the data, not missing data.

---

## 8. Training table — statistics

`train/output/train_table.csv`: **218,795 labeled face frames**, 65 drivers.

| State | Frames | Share | Source |
|---|---|---|---|
| DISTRACTED | 105,493 | 48.2% | distraction |
| FOCUSED | 101,113 | 46.2% | both |
| TIRED | 6,214 | 2.8% | drowsiness |
| DROWSY | 5,975 | 2.7% | drowsiness |

Classifier input features: `yaw, pitch, eye_open_prob, perclos, blink_rate,
gaze_yaw, gaze_pitch`. **Class imbalance** → use inverse-frequency class weights.
**Split must be grouped by driver** (session id) to avoid near-duplicate-frame
leakage between train/val. Note: the 65 sessions come from only **14 drivers**.

### 8.1 First Stage-④ 4-state baseline (`train/train_state.py`)
- Model: MLP 7→64→32→4, standardization baked in, inverse-freq class weights,
  Adam lr 1e-3, 40 epochs. Split by driver (10 train / 4 val), fatigue drivers
  split separately so val holds DROWSY/TIRED. Held-out val drivers: gA_1, gC_14,
  gE_28, gZ_37. Exported to ONNX (`models/state_classifier/state_mlp.onnx`).
- **Per-frame, cross-driver result (val):** accuracy 0.50, **macro-F1 0.35**.

  | class | precision | recall | f1 |
  |---|---|---|---|
  | FOCUSED | 0.62 | 0.43 | 0.51 |
  | DISTRACTED | 0.72 | 0.55 | 0.63 |
  | DROWSY | 0.10 | 0.57 | 0.17 |
  | TIRED | 0.05 | 0.42 | 0.09 |

- **Findings:** (1) DISTRACTED/FOCUSED are the strongest but still cross-confused
  across unseen drivers (domain shift; the 94% gaze number was single-clip,
  threshold-only). (2) DROWSY/TIRED have high recall but very low precision (class
  weighting over-predicts them) **and confuse with each other** — empirical
  evidence that the two fatigue stages are not cleanly separable from DMD alone.
  (3) Per-frame classification is noisy — no temporal smoothing yet.
- **Next levers (ranked):** temporal smoothing / hysteresis on the output
  (biggest expected win); add features (roll, eye_closed, rolling gaze/head std,
  deviation magnitudes); milder weighting or focal loss; bring in UTA-RLDD for a
  clean TIRED-vs-DROWSY split; more drivers (only 14 here limits generalization).

### 8.2 Binary experiment — ATTENTIVE vs INATTENTIVE (2026-07-16)

**Motivation:** the 4-state label distribution is heavily skewed (46/48/2.8/2.7%),
and the class-weighted 4-state baseline over-fires the two fatigue classes.
Collapsing all non-FOCUSED states into a single **INATTENTIVE** class ("driver
inattention" is the standard umbrella term covering distraction + fatigue)
yields a nearly balanced ~46/54 problem. This does not *solve* the imbalance —
DROWSY/TIRED frames are still rare inside INATTENTIVE — but it removes the
extreme class weights and gives a clean upper-bound-style baseline for "does the
system detect that something is wrong at all".

**Setup:** `python -m train.train_state --classes binary`. Identical to the
4-state run in every other respect — same table, features, MLP (7→64→32→2),
epochs/lr/seed, and crucially the **same driver split** (held-out: gA_1, gC_14,
gE_28, gZ_37), so results are directly comparable. Exported to
`models/state_classifier/binary/state_mlp.{pt,onnx}`; `run_live.py` reads the
class set from the checkpoint.

**Result (per-frame, held-out drivers):** accuracy **0.620**, macro-F1 **0.620**
(4-state: acc 0.497, macro-F1 0.347), ROC-AUC (INATTENTIVE) **0.682**.

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| ATTENTIVE | 0.550 | 0.677 | 0.607 | 21,262 |
| INATTENTIVE | 0.701 | 0.578 | 0.633 | 27,859 |

INATTENTIVE recall by original state: DISTRACTED **59.8%**, TIRED 42.3%,
DROWSY **23.8%**; FOCUSED kept ATTENTIVE 67.7%.

**Findings:**
1. Balancing the classes lifts overall accuracy 0.50 → 0.62 and removes the
   fatigue over-prediction (no more 8× class weights), but the per-frame
   ceiling stays modest — confirming §8.1's diagnosis that the limits are
   **method (no temporal context) and driver diversity (14 people)**, not the
   label taxonomy alone.
2. The per-state recall breakdown shows the binary model is effectively a
   *distraction* detector: DROWSY frames are mostly missed (23.8%) because with
   ~5% fatigue share inside INATTENTIVE, the head/gaze features dominate the
   decision. A production system still needs the fatigue signal handled
   separately (temporal PERCLOS path) or a hierarchical design
   (attention → which kind of inattention).
3. Useful paper framing: binary = "alert vs needs-attention" alarm baseline;
   the 4-state model refines *why*. The two share one training pipeline and one
   comparable evaluation protocol.

### 8.3 Temporal features + weight/loss tuning + three-class FATIGUED (2026-07-22)

**Motivation:** §8.1/§8.2 diagnosed the poor macro-F1 as method-limited, not
data-limited — per-frame classification with no temporal context, and
inverse-frequency class weights so aggressive they over-fire the rare fatigue
classes (DROWSY precision 0.10, TIRED precision 0.05 despite recall >0.4).
This experiment tests that diagnosis directly: add temporal features, tune the
weighting/loss, and also try a **three-class** taxonomy — FOCUSED /
DISTRACTED / **FATIGUED** (DROWSY+TIRED merged; named FATIGUED rather than
TIRED because TIRED already names a specific state in the 4-class taxonomy,
and reusing it for the merged class would be confusing — FATIGUED is the
umbrella term, parallel to INATTENTIVE in §8.2).

**New temporal features** (`train/build_dataset.py::_add_temporal_features`,
computed post-hoc from the existing per-frame cascade output, no re-extraction
needed): `perclos_15s` (faster-reacting PERCLOS at the same 15s timescale used
to label DROWSY from GT), `yaw_std_5s` / `gaze_yaw_std_5s` (head/gaze
volatility — sustained movement vs. single-frame noise), `eye_open_prob_mean_5s`
(smoothed eye-state signal). Classifier input grows from 7 to 11 features (see
`train/FEATURES.md`). All four are computed from the model's own output, not
GT, so they're runtime-computable in principle — but `run_live.py` does not
compute them yet (deferred; see below).

**New tuning options** (`train/train_state.py`): `--smooth-window N`
(post-hoc eval only — rolling-mean the predicted probabilities over N frames
per session, then argmax, mirroring `run_live.py`'s existing live smoothing);
`--weight-power P` (dampens the inverse-frequency class weights,
`w = (total/(n·count))^P`); `--loss focal` (focal loss on top of the class
weights); `--model gbt` (sklearn `HistGradientBoostingClassifier`, a
model-family comparison baseline, not exported).

**Results** (three-class, held-out drivers gA_1/gC_14/gE_28/gZ_37, same split
as §8.1/§8.2), macro-F1 progression:

| Config | macro-F1 (raw) | macro-F1 (smoothed) |
|---|---|---|
| new features, default weighting (`weight-power 1.0`, `ce`) | 0.510 | 0.525 (window=30) |
| `weight-power 0.5` | — | 0.578 (window=15) |
| `weight-power 0.3` | — | 0.583 (window=15) |
| `loss focal` (weight-power 1.0) | — | 0.532 (window=15) |
| **`loss focal` + `weight-power 0.3`** | **0.577** | **0.599 (window=30)** |
| `loss focal` + `weight-power 0.3` (window=60) | — | 0.596 (slightly worse — 30 is the sweet spot) |
| `model gbt` + `weight-power 0.3` | 0.518 | 0.515 (smoothing didn't help GBT) |

Winning config: `python -m train.train_state --classes three --loss focal
--weight-power 0.3 --smooth-window 30` → **macro-F1 0.599**, accuracy 0.644
(vs. four-state 0.35, binary 0.62 — three-class closes most of the gap to
binary while keeping distraction and fatigue separate).

| class | precision | recall | f1 |
|---|---|---|---|
| FOCUSED | 0.589 | 0.676 | 0.630 |
| DISTRACTED | 0.728 | 0.625 | 0.672 |
| FATIGUED | 0.446 | 0.553 | 0.494 |

FATIGUED recall by original label: DROWSY 62.4%, TIRED 40.7%.

**Findings:**
1. **The method diagnosis was right.** Temporal features + smoothing alone
   (default weighting) only moved macro-F1 0.510→0.525. **Weight-power
   dampening was the single biggest lever** (0.525→0.583 at `weight-power
   0.3`, same smoothing) — the original inverse-frequency weighting was
   overcorrecting for the imbalance, not undercorrecting. Stacking focal loss
   on top of the dampened weights added a further point (0.583→0.599).
2. FATIGUED precision improved dramatically over the old TIRED/DROWSY numbers
   (0.10–0.05 → 0.446) at the cost of some recall (62%/41% vs. the old
   run's >90% recall from over-firing) — a much more usable operating point:
   the old model cried fatigue constantly; this one is right when it does.
3. **GBT underperformed the MLP** (0.515–0.518 vs. 0.599) on this feature set
   and did not benefit from post-hoc smoothing the way the MLP did — sticking
   with the MLP for now.
4. Smoothing window sweep (15/30/60 frames, i.e. roughly 1s/2s/4s at the
   stride-2 extraction rate): gains plateau and slightly reverse past 30
   frames — window=30 is the current default recommendation, not window=60.
5. **The new features alone lift the other two modes too**, retrained with
   the same 11-feature table (default weighting, no smoothing — these were
   not the focus of this pass): four-state macro-F1 **0.35 → 0.435**, binary
   macro-F1 **0.62 → 0.643** (ROC-AUC 0.68 → 0.703). Neither has had the
   weight/loss tuning applied yet — that's the obvious next quick win for
   both if the four-state and binary variants are still wanted alongside
   `three`. Checkpoints in `models/state_classifier/` and `.../binary/` were
   overwritten with these 11-feature versions.
6. **Not yet ported to the live runtime.** `run_live.py`'s `StateClassifier`
   was fixed to read its feature list from the checkpoint (previously
   hardcoded to the module-level `FEATURES` constant, which would have broken
   both old 7-feature checkpoints and any new 11-feature one) and to
   gracefully neutralize features it can't compute — but the live cascade
   still only produces the original 7 instantaneous features. A checkpoint
   trained with the 4 new rolling features will run live, but those 4 inputs
   fall back to the training mean (neutral) until the live pipeline computes
   them, so the live demo won't see the full benefit measured here yet.

**Next steps (ranked):** port the rolling-feature computation into the live
runtime (turns this from an offline result into an actual live improvement);
a real sequence model (LSTM/GRU/temporal-conv) if there's still a gap once
that's done; then — per §8.1's original ordering — more drivers / UTA-RLDD
/ NTHU-DDD for the fatigue classes specifically, now that method fixes have
been tried first.

### Data-sufficiency assessment (the baseline is method-limited, not size-limited)
Separating "enough data" into three axes:
- **Frames — ample.** 218k labeled frames for a 7-feature MLP is far more than
  needed; frame count is *not* the bottleneck.
- **Drivers — thin.** Only **14** people (10 train / 4 val). Frames from one
  driver are highly correlated, so this is effectively ~14 independent samples —
  the real limit on *generalization to new people*.
- **DROWSY/TIRED — genuinely weak.** ~6k frames each, acted, DMD-sourced; a poor
  fatigue source. Needs **UTA-RLDD / NTHU-DDD** (not more DMD).

**Diagnosis:** the poor macro-F1 (0.35) is primarily **method, not data quantity**:
(1) per-frame classification discards temporal structure (the states are inherently
temporal — PERCLOS *is* a window, distraction is an episode); (2) aggressive class
weighting over-fires DROWSY/TIRED (high recall, low precision). The **features
themselves validate well** (eye 98%, gaze 94%), so the input is sound.

**Planned order (cheap → expensive):** (1) temporal modeling; (2) class-weight /
focal-loss tuning, or gradient-boosted trees; **then re-measure**; (3) add drivers
only if generalization is still capped; (4) UTA-RLDD / NTHU-DDD specifically for
the fatigue classes. Rationale: fixing method first means the *remaining* gap
tells us exactly how much (and what kind of) extra data is actually needed — a
cleaner paper narrative than collecting data blind.

**Update (2026-07-22):** steps (1) and (2) are done — see §8.3. Weight/loss
tuning turned out to matter far more than the temporal features/smoothing
alone (macro-F1 0.35 → 0.60 on the three-class FATIGUED variant), confirming
the method-not-data diagnosis. Remaining gap before (3)/(4): port the rolling
features into the live runtime, and consider a real sequence model.

## 13. Runtime / live demo

`train/run_live.py` (+ `run_live.bat`) runs the whole pipeline in real time:
`camera → cascade → 7 features → trained MLP → temporal smoothing → overlay`
(colour-coded state, confidence, per-class bars, live features). Works on a webcam
or a video file (file playback throttled to source fps). This is the same runtime
that ports to the Jetson (as TensorRT FP16 engines). Held-out vs trained driver
clips (`train/output/test_gC14_*.mp4`, `trained_gA5_*.mp4`) let the generalization
gap be *seen*, not just measured.

---

## 9. Reproducibility / environment

- Env: conda `dms-train`, **Python 3.11**. Deps: `train/environment.yml` /
  `requirements.txt`.
- Key versions: torch **2.6.0+cu124**, **onnxruntime-gpu 1.22.0** (CUDA 12),
  insightface 1.0.1, sixdrepnet 0.1.6, openvino 2026.2, opencv-python, pandas.
- **GPU setup (RTX 4070 Ti Super, driver 591.86):** install torch from the cu124
  wheel index; `onnxruntime-gpu` **must** match CUDA 12 (1.27 needs CUDA 13 and
  fails). `train/cascade/__init__.py` adds `torch/lib` to the DLL path so
  onnxruntime/InsightFace find cuDNN 9 / cuBLAS 12. Gaze (OpenVINO) runs on CPU.
- Throughput: ~59 fps GPU (4× the ~15 fps CPU baseline); GPU and CPU features
  identical (gaze 94.3 vs 94.4).
- Runtime target (Jetson): TensorRT FP16; INT8 avoided on TRT 10.x / JetPack 6.

---

## 10. Key design decisions (with rationale)

1. **Modular cascade, not end-to-end** — debuggable, low data need, interpretable.
2. **Head pose + gaze both for DISTRACTED** — gaze (94%) > head pose (80%) alone
   because it catches eyes-only glances; gaze kept despite being "optional".
3. **Eye-state model, not EAR heuristic** — EAR caused ~89% false DROWSY in the
   prototype; the model reaches ~98% vs GT.
4. **15 s PERCLOS window for DMD labeling** — matches acted-microsleep timescale
   (§7); runtime keeps the 60 s clinical window.
5. **One image-frame convention for all angles** — head and gaze consistent,
   verified; simplifies the classifier and the paper's reporting.
6. **Stride-2 extraction** — halves compute; consecutive frames are near-duplicates.

---

## 11. Known limitations / future work
- DROWSY/TIRED under-represented (acted data); calibrate on our own rig + augment
  with UTA-RLDD / NTHU-DDD; DROWSY window/threshold is a tunable choice.
- Head-pose sign/gaze conventions calibrated on limited frames — spot-check more
  drivers.
- Gaze in camera frame (fine for driver-facing rigs); revisit if rig geometry
  changes.
- Stage-④ classifier not yet trained; NO_FACE handled by the detector gate.
- Port the cascade to Jetson as TensorRT FP16 engines (per-device build).

---

## 12. Parameter quick-reference

| Group | Parameter | Value |
|---|---|---|
| Camera/DMD | resolution / fps | 1280×720 / 29.76 |
| Extraction | stride | 2 |
| SCRFD | det_size / thresh | 640 / 0.5 |
| 6DRepNet | crop margin | 0.25 |
| Eye | size / norm / crop / thr | 32² / (x−127.5)/255 RGB / 0.30 / 0.5 |
| Gaze | eye-crop frac | 0.25 |
| PERCLOS (runtime) | window / thr | 60 s / 0.20 |
| DROWSY (DMD labeling) | window / thr | 15 s / 0.20 |
| DISTRACTED (runtime) | yaw / pitch thr | 30° / 20° |
| Classifier | features | yaw, pitch, eye_open_prob, perclos, blink_rate, gaze_yaw, gaze_pitch |
