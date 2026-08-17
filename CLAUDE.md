# Driver Monitoring System — Claude Context

Real-time driver state detection on **Jetson Orin Nano Super** + **ArduCam IMX477** (single cabin camera).
University of St. Thomas research project — Khongmeng Kormoua, supervised by Dr. Cheol-Hong Min.

## Project goal

Detect 4 driver states in real time from a single cabin camera, plus a no-driver fallback:
- **FOCUSED** — looking at road, eyes open
- **DISTRACTED** — head yaw > 30° or pitch > 20°
- **DROWSY** — PERCLOS ≥ 20% over last 60s
- **TIRED** — early fatigue (yawning, low vigilance) — *yawn signal exists in the offline training pipeline (§8.7, MAR-based) and is part of the final recommended model (macro-F1 0.810 ensemble, see models/README.md), not yet wired into the live Jetson runtime*
- **NO_FACE** — driver not detected (fallback, not a driver state)

**Approach:** Transfer learning — compose pretrained models for face detection, head pose, and eye state; fine-tune for the 4-class output. Avoid training from scratch.

**Paper framing (in progress, 2026-08):** rescoping the paper around
**RGB-only input + embedded resource constraints** (target: Jetson Orin
Nano Super) rather than macro-F1 alone — the core question is which of the
10 compared architectures (`models/README.md`) is the right accuracy-vs-
compute operating point under a real embedded power/latency/memory budget,
not just which scores highest. See `docs/related_work_embedded.md` for the
supporting hardware-constraint and related-work research.

All suggestions should be Jetson-compatible (ONNX Runtime TRT EP or TensorRT), and respect the 4-state taxonomy (+ NO_FACE).

## Inference pipeline architecture

```
Camera → [Face Detection] → [Head Pose + Eye State] → [State Classifier] → Alert/Overlay
```

**Current state:** `inference/` runs a MediaPipe FaceMesh prototype on Jetson — single model gives landmarks, EAR comes from eye landmarks, head pose from `cv2.solvePnP` on 6 face points. This is the cross-platform development baseline, not the production target.

**Planned:** swap FaceMesh for the ONNX/TRT cascade below (SCRFD → 6DRepNet → open-closed-eye-0001 → MobileNetV3 classifier). DeepStream pipeline is a later hardened production build.

## Target production setup (the setup that actually works)

The FaceMesh prototype is fragile because **every signal is a geometric heuristic on landmarks, not a model**: EAR (from eye landmarks) decides DROWSY, `solvePnP` decides DISTRACTED, iris-offset draws the gaze arrow. There is *no* learned gaze, eye-state, or drowsiness model. These heuristics break on camera angle/lighting — confirmed in [[camera_view_findings]]. The production target replaces each heuristic with a model trained on in-cabin data, and adds two things that matter more than model choice:

**Core principle:** replace geometric heuristics with models → fix camera geometry → fine-tune on in-cabin data → run on TensorRT FP16.

| Stage | Model | Replaces (current heuristic) | Feeds state |
|---|---|---|---|
| ① Face detect | SCRFD-500M | FaceMesh detect | crop + NO_FACE gate |
| ② Head pose | 6DRepNet | `cv2.solvePnP` (yaw is broken, spans ±180°) | DISTRACTED |
| ③ Eye state | open-closed-eye-0001 | EAR threshold (causes ~89% false DROWSY) | PERCLOS → DROWSY; blink/yawn → TIRED |
| ④ State | MobileNetV3-Small (rules first, learned later) | the `if` ladder in `state_detector.py` | all 4 states |

**Two non-model root causes (higher leverage than swapping models):**
1. **Camera geometry** — mount near-frontal, ~eye level (steering column / A-pillar), NOT looking down from the headliner; add IR illuminator + IR-pass for night/glare invariance. A bad angle re-introduces every failure even with perfect models.
2. **Domain fine-tuning** — fine-tune ③ on MRL Eye (IR crops), validate ④ on DMD + NTHU-DDD, split DROWSY↔TIRED with UTA-RLDD, and calibrate thresholds (PERCLOS, yaw/pitch) on clips from *our own rig*, not paper defaults. See [[ref_datasets]].

**Temporal layer (don't skip):** PERCLOS over 60s rolling window fed by the eye-state *model* (not EAR), state hysteresis/smoothing to stop flicker, blink-rate + yawn detection to finally wire up the reserved TIRED state.

**Build order (leverage-ranked, don't need all of it to get unstuck):**
1. Fix the mount (near-frontal + IR) — free, root cause of worst numbers.
2. Swap in ③ open-closed-eye-0001 → kills false DROWSY (biggest single win).
3. Swap in ② 6DRepNet → fixes yaw, makes DISTRACTED real.
4. Swap in ① SCRFD → solid crops + NO_FACE.
5. ④ state: rules on the now-reliable signals first; upgrade to fine-tuned MobileNetV3 once rig clips are labeled.
6. Gaze (gaze-estimation-adas-0002) is **optional** for the minimal Jetson path. Now implemented as stage ⑤. **Correction (2026-07-29):** the original claim here ("lifts off-road detection 80%→92/94%") was measured on one session with a self-tuned threshold and does not reproduce on the full 65-session validation — see the corrected numbers below. Gaze's value as a DISTRACTED feature needs re-assessment (proper held-out threshold tuning, not same-session), not assumed; still worth keeping for now, but don't cite the old lift figure.

**Modular cascade vs. single end-to-end model:** start modular (above) — debuggable, fixable one stage at a time, matches the transfer-learning approach. A single end-to-end temporal net (e.g. trained on DMD video) can be more accurate but is a black box needing far more labeled data.

## Feature-extraction cascade (IMPLEMENTED, PC-side offline)

Stages ①②③⑤ + the temporal layer now exist as a **pluggable offline cascade** in
`train/cascade/`, run over DMD videos to build the training set for stage ④. This
is the PC-side training-prep counterpart to the (still-future) Jetson runtime
cascade. See `train/README.md` (how to run), `train/FEATURES.md` (feature schema),
and **`docs/METHODOLOGY.md`** (full pipeline / parameters / decisions / results —
the paper reference; keep it updated).

```
DMD *_rgb_face.mp4
  → ① SCRFD-500M (InsightFace buffalo_sc) → face box + 5 keypoints
  → ② 6DRepNet (pip sixdrepnet)           → yaw/pitch/roll
  → ③ open-closed-eye-0001 (ONNX)         → eye open/closed
  → ⑤ gaze-estimation-adas-0002 (OpenVINO)→ gaze yaw/pitch (uses ② + eye crops)
  → mouth: 2d106det (InsightFace buffalo_s)→ Mouth Aspect Ratio (MAR) → yawn (§8.7)
  → temporal: PERCLOS + blink + yawn      → train/output/features/<session>.csv (+ DMD GT)
```

- **Run:** `python -m train.extract_features --task drowsiness|distraction` then
  `python -m train.validate train/output/...`. Models in `train/cascade/`, DMD
  parsing (OpenLABEL→per-frame labels, drowsiness + distraction) in `train/dmd/`,
  config under `cascade:` in `config.yaml`, weights via `python -m train.download_models`.
- **Drowsiness validation, single-session spot-check (gA_1_s5, 1500 frames):**
  face 100%; eye-state vs GT 94.9% acc (98.2% tuned, closed recall 98.6%);
  blink 34 vs GT 25.
- **Distraction validation, single-session spot-check (gA_1_s1, 900 frames)**
  against `gaze_on_road` GT: head-pose `|yaw|` alone predicts off-road at
  **80%**; the gaze model appeared to lift it to 94% (on-road 11° vs off-road
  29° gaze deviation).
- **Full-dataset validation, all 65 sessions / 227k frames (2026-07-29,
  `train/validate.py`, see `docs/METHODOLOGY.md` §6.1):** does **not**
  reproduce the single-session numbers above. Eye-state: 96.8% acc but closed
  **recall drops to 82.6%** (TP=3657 TN=27964 FP=278 FN=769). Distraction:
  `|yaw|>48` → 82.8% vs `gaze>38` → 83.0% — **gaze and head-pose are
  statistically tied**, not gaze-beats-head-pose; per-session accuracy ranges
  49%–99%. Re-running the *same* self-tuning validation on gA_1_s1 alone today
  gives 87.8%/87.5%/88.6% (yaw/gaze/combined) — not 80%/94% — so the original
  single-session numbers were almost certainly a threshold tuned and evaluated
  on the same small slice, not a generalizing result. **Treat the 80%→94%
  gaze-lift claim as retracted** until re-validated with a proper held-out
  threshold-tuning split. Eye-state and blink-count also found to vary a lot
  per-session (blink count aggregate looks right, 1650 vs GT 1633, but that's
  cancellation — per-session swings from −85% to +41%).
  (Gaze conventions, verified vs known-direction frames — still valid,
  independent of the accuracy-lift claim above: feed
  `head_pose_angles=[yaw, -pitch, roll]` (6DRepNet pitch sign inverted vs Intel);
  flip output x to image frame (`gv[0]=-gv[0]`, model is anatomical); forward
  axis −z. Feature signs: gaze_yaw + = image-right, gaze_pitch − = down.)
  Stored head pose is put in the **same image frame**: 6DRepNet yaw is mirrored
  vs the image so the stored `yaw` is negated (pitch already matches); head and
  gaze then share +yaw=image-right / −pitch=down. (The gaze stage still gets raw
  6DRepNet angles internally.) Sign-only changes, so |yaw|/|gaze| validation is
  unaffected.
- DMD has **no continuous head-pose GT** (validated by range only); gaze is
  validated via the distraction set's `gaze_on_road` labels.
- **Stage ④ status — MLP baselines (§8.1–§8.5), superseded for the 3-class
  task by a GRU (§8.6–§8.7):** original MLP baselines on the pre-§6.1 table:
  4-state macro-F1 0.435, binary (`--classes binary`) 0.643/ROC-AUC 0.70,
  three-class (`--classes three`, DROWSY+TIRED merged into FATIGUED) tuned to
  macro-F1 0.60 (`--loss focal --weight-power 0.3 --smooth-window 30`).
  §8.4/§8.5 explored class-balanced training (undersampling) as an
  alternative to weighting — better FATIGUED recall (75–85%) but worse
  macro-F1 (0.45–0.53), a precision/recall tradeoff not a strict win.
  **§8.6 replaced the per-frame MLP with a small `nn.GRU`
  (`train/train_sequence.py`)** — real temporal modeling instead of 4
  hand-picked rolling-window stats, trained on fixed-length chunks but
  evaluated on full per-session sequences (matches live streaming). Result:
  **macro-F1 0.773**, the single biggest jump in the project, with FATIGUED
  precision and recall improving *together* for the first time (0.615/0.74)
  instead of trading off. Checkpoint (superseded — see `models/README.md` for the current final result, macro-F1 0.810).
  **§8.7 added the yawn/mouth feature (see below) and retrained the same
  GRU** — macro-F1 held flat (0.772) but **TIRED recall jumped 55.9%→80.4%**,
  a targeted, mechanistically-explained win (TIRED is yawn-defined in the
  labels) at the cost of some DISTRACTED recall. Checkpoint:
  a checkpoint superseded by later work — see `models/README.md` for the
  current model set. Both GRU checkpoints and every
  MLP checkpoint (§8.1–§8.5) coexisted in separate directories at the time.
  `train/run_live.py`'s `StateClassifier` now auto-detects GRU checkpoints
  (via the `hidden`/`layers` keys `train_sequence.py` saves) and carries the
  GRU's hidden state across frames — one `StateClassifier` instance per
  continuous video/session, same as how it's evaluated offline. Demo videos
  for all three three-class checkpoints (distraction + drowsiness, held-out
  driver gC_14) in `train/output/demo_gC14_*_three*.mp4`. Full discussion:
  `docs/METHODOLOGY.md` §8.3–§8.7.
- **Feature-extraction validated end-to-end against DMD GT on the full
  65-session/227k-frame dataset, twice (2026-07-29, `docs/METHODOLOGY.md`
  §6.1 and §8.7):** §6.1 found and fixed `blink_rate` divide-by-near-zero at
  session start, and retracted the "gaze lifts distraction accuracy 80%→94%"
  claim (didn't reproduce outside the single session it was measured on).
  §8.7 added a new **yawn/mouth stage** (InsightFace 2d106det landmarks →
  Mouth Aspect Ratio; no pretrained mouth-state model exists like
  open-closed-eye-0001 does for eyes) — landmark indices verified empirically
  on real frames, not assumed. Testing §6.1's own recommended blink debounce
  fix (`blink_min_frames` 1→2) **made blink counting worse** (aggregate ratio
  0.61x vs the original 1.01x) — reverted. Tuning the yawn MAR threshold to
  its "best accuracy" value (0.64) similarly **tanked yawn-event recall**
  (44 vs GT 76, was 87 vs GT 76 at 0.5) — reverted. Also found and fixed a
  `mar` divide-by-near-zero bug (same shape as `blink_rate`'s). Both reverts
  + the mar fix were applied **without re-running the ~87-minute extraction**
  a third time — blink/PERCLOS/mouth-derived columns are pure functions of
  the already-saved `eye_open_prob`/`mar`, recomputed post-hoc in seconds.
- **Not yet done:** re-assessing gaze's value as a DISTRACTED feature with a
  proper held-out threshold split instead of the retracted single-session
  claim. **Resolved since this section was written:** which three-class
  checkpoint is "the" model — the ensemble of the gated-fatigue GRU +
  single-branch GRU, macro-F1 **0.810**, is the final result (see
  `models/README.md` and §8.16). Retraining 4-state/binary on the
  corrected+enriched table remains open; the
  per-session blink-count variance (−85%/+41% swings) remains unresolved
  (debounce didn't fix it); porting the cascade to the Jetson runtime as TRT
  FP16 engines (the GRU checkpoints only run live via PyTorch so far, not
  yet exported/optimized as a TensorRT engine).

## Recommended models

### Stage 1 — Face Detection
- **SCRFD_500M** (InsightFace) — best accuracy/compute ratio, ONNX export built-in, ~2–5ms FP16 on Orin Nano
  - GitHub: https://github.com/deepinsight/insightface/tree/master/detection/scrfd
  - TRT: https://github.com/namdvt/SCRFD_FaceDetection_TensorRT
- **NVIDIA TAO FaceDetectIR** — use if running IR mode at night
  - NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/facedetectir

### Stage 2 — Head Pose Estimation
- **6DRepNet** — SOTA, full ±90° yaw/pitch/roll, IEEE T-IP 2024, clean ONNX export
  - GitHub: https://github.com/thohemp/6DRepNet | `pip install sixdrepnet`
- **DMHead** — fused 6DRepNet+WHENet, pre-converted ONNX/TRT/TFLite, handles 90°+ sideways turns
  - GitHub: https://github.com/PINTO0309/DMHead

### Stage 3 — Eye State (PERCLOS)
- **open-closed-eye-0001** (OpenVINO OMZ) — 32x32 input, 0.0014 GFLOPs, 95.84% accuracy, essentially free
  - GitHub: https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/open-closed-eye-0001/README.md
  - **Direct ONNX** (old `download.01.org` is dead): `storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/open-closed-eye-0001/open_closed_eye.onnx` — fetched by `train/download_models.py`.
  - **Preprocessing (critical):** RGB, `(x − 127.5)/255`. Raw 0–255 BGR makes the in-model softmax overflow to `[0, nan]`. Output is already softmaxed. Run per-eye, accumulate over sliding window for PERCLOS.

### Stage 4 — State Classifier
- Fine-tune **MobileNetV3-Small** or **EfficientNet-B0** on features (head pose angles + PERCLOS + yaw deviation)
- Or restructure **YOLOv8n** with auxiliary classification head for single end-to-end model
  - Docs: https://docs.ultralytics.com/guides/nvidia-jetson/ | TRT export: `yolo export model=yolov8n.pt format=engine`

### Reference pipeline (complete working example)
- **OpenVINO Driver Behaviour Demo**: https://github.com/incluit/OpenVino-Driver-Behaviour
  - Full cascade: face → head pose → landmarks → eye state → gaze. Computes EAR, PERCLOS, yaw/pitch thresholds.

## Recommended runtime

| Context | Runtime |
|---|---|
| Python development | ONNX Runtime GPU with TensorRT EP (`onnxruntime-gpu`, `TensorrtExecutionProvider`) |
| Production / max throughput | TensorRT direct (`trtexec --onnx=model.onnx --fp16`) |
| Hardened production build | NVIDIA DeepStream SDK |
| Fine-tuning NVIDIA ADAS models | NVIDIA TAO Toolkit |

- Use **FP16** on TRT — INT8 has known issues on TRT 10.x / JetPack 6.
- TRT engines are device-specific; must rebuild per device.
- MediaPipe and OpenCV DNN are prototyping/fallback only — not suitable for production on Jetson.

## Datasets for training / fine-tuning

| Priority | Dataset | Best for | URL |
|---|---|---|---|
| 1 | **DMD** (Vicomtech) | All 4 states, in-car, 41h, CC BY-NC-ND | https://dmd.vicomtech.org/ |
| 2 | **State Farm** (Kaggle) | DISTRACTED, 102K images | https://www.kaggle.com/c/state-farm-distracted-driver-detection |
| 3 | **NTHU-DDD** | DROWSY benchmark, day+night video | http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/ |
| 4 | **MRL Eye** | Eye state / PERCLOS fine-tuning, 85K IR crops | https://mrl.cs.vsb.cz/eyedataset |
| 5 | **UTA-RLDD** | TIRED vs DROWSY distinction, temporal | https://sites.google.com/view/utarldd/home |
| 6 | **YawDD** | Yawning (TIRED state), 322 clips | https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset |
| 7 | **CEW** | In-the-wild eye open/closed variation | https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html |
| 8 | **100-Driver** | Cross-condition generalization, 470K images | https://100-driver.github.io/ |
| 9 | **ETH-XGaze** | Gaze estimation pretraining, 1.1M images | https://github.com/xucong-zhang/ETH-XGaze |
| 10 | **UL-DD** (2025) | Newest open fatigue dataset | https://www.nature.com/articles/s41597-025-06540-1 |

## Project layout

```
inference/      Jetson runtime (camera + face analysis + state + overlay)
tools/          Operator-facing utilities (recorder + camera preview)
train/          PC training (RTX 4070 Ti Super): cascade/ (impl. feature-extraction
                cascade), dmd/ (DMD parsing), extract_features.py, validate.py
models/         Local model weights (gitignored; eye_state/ ONNX fetched by script)
recordings/     Captured session videos (gitignored)
datasets/       DMD dataset (gitignored; extract the per-group .tar.gz in place)
docs/           Dev log, reference links, recording guide, manual
hardware/       Hardware notes and setup photos
```

The old `archive/` (v1–v8 exploration scripts, CSI-Camera, DeepStream reference
apps), `assets/` test videos, and `imgs/` were removed in cleanup (commit
`4b40798`, 2026-06-23). The DeepStream production pipeline is still future work
(see "Target production setup"); there is no `deepstream/` dir yet.

## Data collection

Driver-attention data is captured in the lab by pairing the Jetson rig with a PC running **Assetto Corsa** as the driving simulator. The Jetson records the driver-facing IMX477 feed; the PC runs the sim in parallel on its own display. See `docs/recording_guide.md` for the operator runbook and `tools/record.py` for the recorder app.

## Environment

- Jetson inference: conda env `dms-infer` (`inference/environment.yml`). System OpenCV is symlinked in — do not pip-install opencv. See README.
- PC training: conda env `dms-train` (`train/environment.yml`). **Torch comes from the PyTorch pip wheel index** — the old `pytorch-cuda=12.8` conda pin does not solve (conda metapackage stopped at 12.4). `insightface` now installs as a prebuilt wheel (no compiler). Keep `config.yaml` ASCII — Windows gbk locale breaks `yaml.safe_load` on non-ASCII. See [[dms-train-env-gotchas]].
- Legacy `mp` env from early MediaPipe experiments still appears in `docs/log.txt`; superseded by `dms-infer`.
- All tunable parameters in `config.yaml` — no need to edit source files.
- See `docs/log.txt` for setup history and troubleshooting.
