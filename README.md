# Driver Monitoring System

Real-time driver state detection from a single in-cabin RGB camera.

Research project — University of St. Thomas.
Conducted by Khongmeng Kormoua, supervised by Dr. Cheol-Hong Min.

> **New to this repo?** Read this page top to bottom once, then jump to
> whichever quick-start section matches the machine in front of you.

## What this project does

Classifies the driver into one of four states, plus a no-driver fallback:

| State | Meaning |
|---|---|
| **FOCUSED** | Eyes open, looking at the road |
| **DISTRACTED** | Head turned away / not looking at the road |
| **DROWSY** | Eyes closed for a large fraction of the last ~15–60s (PERCLOS) |
| **TIRED** | Early fatigue signs — yawning, low vigilance |
| **NO_FACE** | Driver not detected (fallback, not a driver state) |

**Target hardware:** NVIDIA Jetson Orin Nano Super + ArduCam IMX477 camera,
real-time (TensorRT FP16). **Training hardware:** a PC with an NVIDIA GPU
(developed on an RTX 4070 Ti Super).

## Two runtimes — know which one you're looking at

This repo currently has **two separate inference paths**, at different
levels of maturity. Mixing them up is the single most confusing thing about
this repo for a new reader, so read this part carefully:

| | `inference/` (this repo's Jetson runtime today) | `train/run_live.py` (PC demo of the production target) |
|---|---|---|
| Runs on | Jetson (also runs on a PC for development) | PC only, for now |
| How it decides state | Geometric heuristics (EAR threshold, `solvePnP` head pose) | Trained models (see `models/README.md`) — the actual cascade + classifier this project's results are about |
| States it produces | FOCUSED / DROWSY / DISTRACTED / NO_FACE (TIRED reserved, not implemented) | All 5, including TIRED |
| Status | Working prototype, cross-platform baseline | Where the real accuracy numbers come from — **not yet ported to the Jetson** |

**In short:** if you're trying to reproduce this project's *results*
(the numbers in `docs/METHODOLOGY.md`), you want `train/`. If you're trying
to see *something* run live on the Jetson today, you want `inference/`.
Porting `train/run_live.py`'s cascade + classifier onto the Jetson (ONNX
Runtime + TensorRT) is the next major piece of work — see
`docs/METHODOLOGY.md` for where that stands.

## Repo layout

```
inference/      [Jetson] today's live runtime — camera, heuristic face
                 analysis, state detector, on-screen overlay
tools/          [Jetson] operator utilities — data-collection recorder GUI,
                 camera preview
train/          [PC] everything about training: the feature-extraction
                 cascade, the 10-architecture classifier comparison,
                 the PC-side demo of the full trained pipeline
                 (see train/README.md — start there for training work)
models/         Trained model weights (see models/README.md for what's
                 in here and which one to use)
docs/           Paper-facing docs — full methodology/results
                 (docs/METHODOLOGY.md), embedded-hardware related work
                 (docs/related_work_embedded.md), data-collection runbook
                 (docs/recording_guide.md)
recordings/     Captured session videos (gitignored — created by the
                 recorder tool)
datasets/       Training datasets, e.g. DMD (gitignored — see
                 train/README.md for how to get it)
hardware/       Hardware notes and setup photos
```

## Quick start — [Jetson] run the live prototype

**Prerequisites:** JetPack 6, miniforge3, system OpenCV with GStreamer
support.

1. **Create the conda environment**
   ```bash
   source ~/miniforge3/bin/activate
   conda env create -f inference/environment.yml
   conda activate dms-infer
   ```

2. **Link system OpenCV into the conda env** — OpenCV must come from the
   system build (GStreamer + Argus support). Do **not** `pip install
   opencv-python` here.
   ```bash
   SITE=$(python -c "import site; print(site.getpackages()[0])")
   ln -s /usr/lib/python3/dist-packages/cv2 $SITE/cv2
   ```

3. **Run it**
   ```bash
   conda activate dms-infer
   python run_inference.py
   # Quit: press Esc
   ```

If a camera window opens and you see a state label (FOCUSED/DROWSY/etc.)
update as you move, it's working. Troubleshooting history:
`docs/log.txt`.

## Quick start — [Jetson] collect a training clip

1. Launch the recorder — three ways, pick whichever's convenient:
   - **From Nautilus:** right-click `Start-Recorder.sh` → **Run as a
     Program** (GNOME 42+ blocks double-click-to-run by default).
   - **Desktop icon:** `./tools/install_launcher.sh` once, then double-click
     the new **DMS Recorder** icon.
   - **From a terminal:** `./Start-Recorder.sh`
2. Record, then stop. The clip lands in `recordings/<name>.mp4`.
3. See `docs/recording_guide.md` for the full lab-session runbook
   (pairing with the Assetto Corsa driving simulator, session naming, etc.)

## Quick start — [Jetson] review a recorded clip offline

Runs the same analysis stack as the live prototype, but on a saved file, and
writes an annotated copy back out.

1. **One-time:** fetch the phone/object-detector weights (gitignored):
   ```bash
   mkdir -p models
   curl -sL -o models/efficientdet_lite0.tflite \
     https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
   ```
2. **Run it:**
   ```bash
   conda activate dms-infer
   python -m inference.run_video recordings/<name>.mp4
   ```
   Writes `recordings/<name>_annotated.mp4` and prints a summary
   (face-detection coverage, blink count, phone-visible %, yaw/pitch range,
   state breakdown).

   | Flag | Effect |
   |---|---|
   | `--show` | Also open a live preview window while processing |
   | `--no-save` | Skip the output file, just print the summary |
   | `0` instead of a path | Run on the live camera instead of a file |

> Run as a module (`python -m inference.run_video`), not
> `python inference/run_video.py` — it uses relative imports. Expect
> ~6–7 fps on the Jetson CPU (this offline path uses MediaPipe, not the
> TensorRT cascade).

## Quick start — [PC] train / reproduce the results

**Prerequisites:** miniforge3 or Anaconda, an NVIDIA GPU + driver.

1. **Create the conda environment**
   ```bash
   conda env create -f train/environment.yml
   conda activate dms-train
   ```
2. **Verify the GPU is visible**
   ```bash
   python -c "import torch; print(torch.cuda.get_device_name(0))"
   ```
3. Continue with **`train/README.md`** — the full pipeline walkthrough
   (get the dataset, fetch cascade weights, extract features, train a
   classifier) lives there, not here.

## Config

All tunable parameters live in `config.yaml` — no need to edit source
files for a threshold/resolution change.

| Section | Key parameters |
|---|---|
| `camera` | sensor_id, resolution (1280×720), framerate (30), flip_method |
| `face_mesh` | max_faces, refine_landmarks |
| `ear` | threshold (0.21), consecutive_frames (3) |
| `head_pose` | yaw_threshold (30°), pitch_threshold (20°) |
| `state` | perclos_window_sec (60), drowsy_perclos (0.20) |
| `object_detector` | model_path, score_threshold (0.35), max_results (5) |
| `cascade` | the `train/` pipeline's cascade settings — see `train/README.md` |
| `display` | window_title, show_fps |

## Where to go next

| I want to... | Go to |
|---|---|
| Understand the full methodology, every experiment, and the results | `docs/METHODOLOGY.md` |
| See the embedded-hardware / related-work research | `docs/related_work_embedded.md` |
| Train or reproduce a classifier | `train/README.md` |
| Understand what's in `models/` and which checkpoint to use | `models/README.md` |
| Run a data-collection session | `docs/recording_guide.md` |
| See setup history / troubleshooting notes | `docs/log.txt` |
