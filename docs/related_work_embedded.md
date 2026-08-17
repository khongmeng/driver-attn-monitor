# Related work — embedded / edge driver monitoring

Research notes for the paper's related-work section and the embedded-deployment
reframe (see `docs/METHODOLOGY.md` for the accuracy-side results this
complements). Compiled 2026-08-14. Confidence level is noted per source —
some details come from the primary paper (HTML fetch), some only from
search-engine summaries because the primary source blocked direct access.

## Jetson Orin Nano Super — hardware ceiling (vendor-given, high confidence)

- Power modes: **7W, 15W, 25W, MAXN Super**.
- Up to **67 INT8 TOPS** (MAXN Super mode).
- **1024 CUDA cores, 32 Tensor Cores** (Ampere GPU), 6-core Arm Cortex-A78AE
  CPU up to 1.7GHz (Super mode).
- **8GB LPDDR5** shared (CPU+GPU unified) memory, **102 GB/s** bandwidth.
- 7W mode: fanless-viable, battery/passive-cooling oriented, trades
  ~30-50% throughput for minimal cooling. 15W: sustained multi-model
  workloads. MAXN Super: uncapped, highest clocks across CPU/GPU/DLA/PVA.
- JetPack 6.2 made "Super Mode" production-supported, ~50-70% more
  performance than the original (pre-Super) power-mode numbers on the same
  hardware.

Sources:
- [Jetson Orin Nano Power: 7W–15W Modes, 8–12W Typical, 25W Peak](https://edgeaistack.ai/blog/jetson-orin-nano-power-consumption/)
- [NVIDIA Jetson Orin Nano Super Developer Kit - 67 TOPS AI board](https://www.kiwi-electronics.com/en/nvidia-jetson-orin-nano-super-developer-kit-11461)
- [JetPack 6.2 Brings Super Mode to NVIDIA Jetson Orin Nano and Jetson Orin NX Modules](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules)

## This project's own real-time constraint (derived from our own config/data, not literature)

- Camera (`config.yaml`): captures **1280x720 @ 30fps** (ArduCam IMX477).
- Every trained classifier/feature so far was built at **effective 15fps**
  (`extract_features.py --stride 2`, i.e. every 2nd native frame).
- Already found empirically (`docs/METHODOLOGY.md` §6.1 blink-debounce
  investigation): real blinks are **already only 1-2 samples wide** at that
  15fps rate — going any slower would further degrade blink/PERCLOS signal
  quality, a real accuracy cost specific to this system, not just a "feels
  slow" concern.
- **Proposed target for the paper:** minimum viable = sustain **≥15 FPS
  end-to-end** (≤66ms/frame for the *whole* pipeline: face detect + head
  pose + eye state + gaze + mouth landmarks + classifier, combined).
  Stretch target = native 30fps for better blink fidelity (~2x the compute
  budget of the 15fps target).
- **Not yet measured**: no stage of this cascade has been run on the actual
  Jetson yet (still MediaPipe prototype in `inference/`, per CLAUDE.md). The
  per-model latency figures already in CLAUDE.md (e.g. SCRFD-500M
  "~2-5ms FP16 on Orin Nano") are the *model authors'* own published
  numbers on their own benchmarking setup, not verified on this rig/camera/
  JetPack version.

## Paper 1 — "Low-Latency Embedded Driver Monitoring System with a Multi-Task Neural Network"

arXiv:2605.02563v1. **High confidence** — fetched full HTML.

**Architecture:** MobileNetV2-style backbone (inverted-residual blocks,
depthwise-separable convs). Single-frame, multi-task head predicting 6
outputs in one forward pass: 98-point facial landmarks (regression),
per-eye opening level (regression), eye visibility (binary), mouth opening
(3-class), head pose (yaw/pitch/roll), distraction (3-class: normal/
phone/smoking). Feature maps from 2 intermediate blocks + the last block
are concatenated after global-average-pooling. Three size variants:

| variant | params |
|---|---|
| Tiny | 124,566 |
| Small | 705,422 |
| Large | 2,330,318 |

**Dataset:** LaPa (22,176 images, 106 facial landmarks) + a small
pseudo-labeled distraction set the authors generated themselves (860
phone-use + 715 smoking images). **Static images — no video, no temporal
modeling of any kind.**

**Accuracy (Table III, Small model):** NME 2.350 (landmark error), eye-state
accuracy 97.8%, mouth accuracy 93.5%, head-pose error (2.62°, 2.79°, 3.19°)
yaw/pitch/roll.

**Latency/hardware — tested on Jetson Nano and Xavier NX, NOT Orin Nano:**
- Small model, FP16: **1.73ms on Xavier NX**, 13.64ms on Jetson Nano.
- Full pipeline incl. face detector, Xavier NX: **23.73ms end-to-end**
  (detecting every frame, N=1) down to 16.46ms (detecting every 8th frame,
  N=8).
- Face detector alone (RFB variant + TensorRT NMS) on Xavier NX: 1.729ms
  FP32 / 1.12ms FP16.
- Computational load: Small model ≈ 447.23 GMACs.
- Power: Jetson Nano 5-10W, Xavier NX 10-15W.

**Code public:** [github.com/cscribano/MtDMS](https://github.com/cscribano/MtDMS) (CC BY-NC-SA 4.0).

Source: [arXiv:2605.02563v1](https://arxiv.org/abs/2605.02563v1)

## Paper 2 — "Real-Time Distracted Driver Detection on Embedded Edge Devices Using Multi-Region CNN Embeddings"

TechRxiv, DOI 10.36227/techrxiv.177274113.38312616/v1. **Lower confidence —
primary source returned HTTP 403 both times; everything below is from
search-engine-summarized snippets, not a direct read of the paper. Verify
against the actual PDF before citing numbers.**

**Architecture:** shared MobileNetV3-Small backbone run over 3 regions
(full-frame, face crop, hand-region crop) per frame; region embeddings
concatenated → lightweight linear classification head. Temporal smoothing
via **EMA (exponential moving average)** on the output probabilities — a
fixed-decay filter, not a learned recurrent/temporal model.

**Dataset:** not confirmed by name in what was retrievable. 9
distraction-related categories (texting, drinking, phone use, interacting
with passengers, etc. — the category count/shape resembles the classic
State Farm distracted-driver taxonomy, but this is **not confirmed**, don't
cite as fact).

**Hardware:** **Jetson Orin Nano** specifically — matches our target chip.
ONNX export, TensorRT FP16.

**Reported performance:** "high accuracy" (no exact number retrieved).
Latency described as **"estimated/projected" 13-20 FPS, 50-75ms total
latency** — the hedging language in the retrievable summary suggests this
may be partly simulated rather than fully hardware-measured; confirm before
citing as a hard number.

Source: [TechRxiv full text](https://www.techrxiv.org/doi/full/10.36227/techrxiv.177274113.38312616/v1) (blocked — retry access before citing directly)

## How these position our paper

Both papers operate at the level of our cascade stages ②③ (head pose, eye
state) plus a mouth/landmark stage — **not** at the level of our stage ④
(temporal state classifier). Neither does anything resembling PERCLOS-style
temporal integration; Paper 1 is single-frame with zero memory across
frames, Paper 2's EMA smoothing is a fixed decay filter, not a learned
temporal model. **Neither attempts the DROWSY-vs-TIRED distinction** —
structurally impossible from a single frame or a fixed-decay filter, since
it requires integrating a real time window (PERCLOS, blink rate, yawn
pattern over ~15-60s).

**Implication for scoping our own embedded-benchmarking work:** we don't
need to re-prove "can face/eye/mouth detection run in real time on a
Jetson" — these two papers (plus the per-model citations already in
CLAUDE.md) establish that's feasible, and can simply be cited. Our own
measurement effort is better spent on: (1) confirming our *specific* chosen
cascade models (SCRFD-500M, 6DRepNet, open-closed-eye-0001,
gaze-estimation-adas-0002, 2d106det) land in a similar latency/power
ballpark on our actual Orin Nano, and (2) the part neither paper touches at
all — the accuracy-vs-compute tradeoff across our temporal classifier
family (MLP → TCN → GRU → gated-GRU → ensemble), which is where our
genuine, currently-uncharted contribution sits.

## Open items

- Get real access to the TechRxiv PDF (Paper 2) before citing its numbers
  directly — everything here is secondhand via search summaries.
- Confirm Paper 2's actual dataset name.
- Once we have Jetson bench time: compare our SCRFD/6DRepNet/eye-state
  numbers directly against Paper 1's face-detector (1.12-1.73ms FP16 on
  Xavier NX, weaker chip than our Orin Nano) as a sanity check/lower bound.
