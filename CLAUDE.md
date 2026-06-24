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
6. Gaze (L2CS-Net / gaze-estimation-adas-0002) is **optional** — not in the 4-state taxonomy, defer it.

**Modular cascade vs. single end-to-end model:** start modular (above) — debuggable, fixable one stage at a time, matches the transfer-learning approach. A single end-to-end temporal net (e.g. trained on DMD video) can be more accurate but is a black box needing far more labeled data.

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
  - Convert: `omz_converter` → ONNX → TRT. Run per-eye, accumulate over sliding window for PERCLOS.

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
tools/          Operator-facing utilities (data-collection recorder, etc.)
train/          PC training scaffold (RTX 4070 Ti Super)
models/         Local model weights (gitignored)
recordings/     Captured session videos (gitignored)
deepstream/     DeepStream configs + TAO models (future TensorRT pipeline)
assets/         Test videos and sample images
docs/           Dev log, reference links, recording guide
hardware/       Hardware notes and setup photos
archive/        Old exploration scripts (v1–v8)
```

## Data collection

Driver-attention data is captured in the lab by pairing the Jetson rig with a PC running **Assetto Corsa** as the driving simulator. The Jetson records the driver-facing IMX477 feed; the PC runs the sim in parallel on its own display. See `docs/recording_guide.md` for the operator runbook and `tools/record.py` for the recorder app.

## Environment

- Jetson inference: conda env `dms-infer` (`inference/environment.yml`). System OpenCV is symlinked in — do not pip-install opencv. See README.
- PC training: conda env `dms-train` (`train/environment.yml`).
- Legacy `mp` env from early MediaPipe experiments still appears in `docs/log.txt`; superseded by `dms-infer`.
- All tunable parameters in `config.yaml` — no need to edit source files.
- See `docs/log.txt` for setup history and troubleshooting.
