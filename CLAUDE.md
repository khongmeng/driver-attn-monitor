# Driver Monitoring System — Claude Context

Real-time driver state detection on **Jetson Orin Nano Super** + **ArduCam IMX477** (single cabin camera).
University of St. Thomas research project — Khongmeng Kormoua, supervised by Dr. Cheol-Hong Min.

## Project goal

Detect 4 driver states in real time from a single cabin camera, plus a no-driver fallback:
- **FOCUSED** — looking at road, eyes open
- **DISTRACTED** — head yaw > 30° or pitch > 20°
- **DROWSY** — PERCLOS ≥ 20% over last 60s
- **TIRED** — early fatigue (yawning, low vigilance) — *enum reserved, detection not yet wired up*
- **NO_FACE** — driver not detected (fallback, not a driver state)

**Approach:** Transfer learning — compose pretrained models for face detection, head pose, and eye state; fine-tune for the 4-class output. Avoid training from scratch.

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
6. Gaze (gaze-estimation-adas-0002) is **optional** for the minimal Jetson path, but the offline cascade shows it lifts off-road detection 80%→92% over head pose alone (catches eyes-only glances). Now implemented as stage ⑤; keep it as a DISTRACTED feature, port to Jetson after ①–④.

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
  → temporal: PERCLOS + blink             → train/output/features/<session>.csv (+ DMD GT)
```

- **Run:** `python -m train.extract_features --task drowsiness|distraction` then
  `python -m train.validate train/output/...`. Models in `train/cascade/`, DMD
  parsing (OpenLABEL→per-frame labels, drowsiness + distraction) in `train/dmd/`,
  config under `cascade:` in `config.yaml`, weights via `python -m train.download_models`.
- **Drowsiness validation (gA_1_s5, 1500 frames):** face 100%; eye-state vs GT
  94.9% acc (98.2% tuned, closed recall 98.6%); blink 34 vs GT 25.
- **Distraction validation (gA_1_s1, 900 frames)** against `gaze_on_road` GT:
  head-pose `|yaw|` alone predicts off-road at **80%**; the **gaze model lifts it
  to 94%** (on-road 11° vs off-road 29° gaze deviation) — gaze catches eyes-only
  glances head pose misses, so it earns its place as a DISTRACTED feature.
  (Gaze conventions, verified vs known-direction frames: feed
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
- **Stage ④ status:** MLP baselines trained on the 218k-frame table + 4 added
  rolling-window temporal features (11 total; `train/train_state.py`, same
  driver split throughout): 4-state (default, macro-F1 **0.435**, was 0.35),
  **binary ATTENTIVE/INATTENTIVE** (`--classes binary`, macro-F1 **0.643**,
  ROC-AUC 0.70, was 0.62), and **three-class FOCUSED/DISTRACTED/FATIGUED**
  (`--classes three`, DROWSY+TIRED merged) additionally tuned with dampened
  class weighting + focal loss + post-hoc smoothing (`--loss focal
  --weight-power 0.3 --smooth-window 30`) reaching **macro-F1 0.60** —
  confirms the original 0.35 baseline was method-limited (aggressive
  weighting + no temporal context), not data-limited; the same weight/loss
  tuning is the obvious next step for the four-state and binary checkpoints
  too. Models in `models/state_classifier/` (`binary/`, `three/`);
  `train/run_live.py` auto-detects the class set from the checkpoint. Results
  + discussion: `docs/METHODOLOGY.md` §8 (§8.3 for the latest).
- **Not yet done:** porting the 4 new rolling temporal features into the live
  runtime (currently offline-only; `run_live.py` neutralizes them to the
  training mean if a checkpoint uses them), a real sequence model if a gap
  remains after that, TIRED/yawn signal, multi-session threshold calibration,
  porting the cascade to the Jetson runtime as TRT FP16 engines.

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
