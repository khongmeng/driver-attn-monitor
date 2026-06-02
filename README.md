# Driver Monitoring System

Real-time driver state detection on Jetson Orin Nano Super + ArduCam IMX477.

Research project — University of St. Thomas.
Conducted by Khongmeng Kormoua, supervised by Dr. Cheol-Hong Min.

## Detected states

| State | Condition |
|---|---|
| **FOCUSED** | Eyes open, head facing road |
| **DROWSY** | PERCLOS ≥ 20% (eyes closed > 20% of last 60 s) |
| **DISTRACTED** | Head yaw > 30° or pitch > 20° |
| **TIRED** | Early fatigue (yawning, low vigilance) |
| **NO_FACE** | Driver not detected |

## Project layout

```
inference/      Jetson runtime (camera + face analysis + state + overlay)
train/          PC training scripts (RTX 4070 Ti Super)
models/         Local model weights (gitignored — download or export separately)
deepstream/     DeepStream configs + TAO models (future TensorRT pipeline)
assets/         Test videos and sample images
docs/           Dev log and reference links
hardware/       Hardware notes and setup photos
archive/        Old exploration scripts (v1–v8) — kept for reference
```

---

## Inference setup (Jetson Orin Nano Super)

**Prerequisites:** JetPack 6, miniforge3, system OpenCV with GStreamer support.

### 1. Create the conda environment

```bash
source ~/miniforge3/bin/activate
conda env create -f inference/environment.yml
conda activate dms-infer
```

### 2. Link system OpenCV into the conda env

OpenCV must come from the system (built with GStreamer + Argus support) — do not pip-install it.

```bash
SITE=$(python -c "import site; print(site.getpackages()[0])")
ln -s /usr/lib/python3/dist-packages/cv2 $SITE/cv2
```

### 3. Run inference

```bash
conda activate dms-infer
python run_inference.py

# Quit: press Esc
```

### 4. (Optional) Launch the data-collection recorder

Three ways to open the app, pick whichever fits:

**A. From the repo folder** — in GNOME Files (Nautilus), right-click
   `Start-Recorder.sh` → **Run as a Program**. (GNOME 42+ blocks
   double-clicking executable scripts by default; the right-click menu
   bypasses that.)

**B. Put an icon on the Desktop** for one-click access from anywhere:
   ```bash
   ./tools/install_launcher.sh
   ```
   Double-click the new **DMS Recorder** icon on your Desktop.

**C. From a terminal**:
   ```bash
   ./Start-Recorder.sh        # or ./tools/start_recording.sh
   ```

See `docs/recording_guide.md` for the session runbook.

> See `docs/log.txt` for full setup history and troubleshooting.

---

## Training setup (Windows PC, RTX 4070 Ti Super)

**Prerequisites:** miniforge3 or Anaconda, NVIDIA driver ≥ 591.86.

### 1. Create the conda environment

```bash
conda env create -f train/environment.yml
conda activate dms-train
```

### 2. Verify GPU is available

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Config

All tunable parameters live in `config.yaml` — no need to edit source files.

| Section | Key parameters |
|---|---|
| `camera` | sensor_id, resolution (1280×720), framerate (30), flip_method |
| `face_mesh` | max_faces, refine_landmarks |
| `ear` | threshold (0.21), consecutive_frames (3) |
| `head_pose` | yaw_threshold (30°), pitch_threshold (20°) |
| `state` | perclos_window_sec (60), drowsy_perclos (0.20) |
| `display` | window_title, show_fps |

> Note: `TIRED` is reserved in the state enum but detection (yawning / low vigilance) is not yet implemented. The current pipeline labels FOCUSED / DROWSY / DISTRACTED / NO_FACE.
