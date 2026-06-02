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

## Record a clip and run inference on it

End-to-end workflow: capture a driver-facing clip, then run the analysis stack
(gaze, head pose, blink rate, and phone/object detection) on the saved file and
review an annotated video.

### 1. Record a clip

Launch the recorder (see options A–C above), capture your session, and stop.
Clips are saved to `recordings/`.

```bash
./Start-Recorder.sh
# ...record, then stop. Output lands in recordings/<name>.mp4
```

### 2. Download the object-detection model (one time only)

The phone/object detector uses EfficientDet-Lite (80 COCO classes). The weights
are gitignored, so fetch them once into `models/`:

```bash
mkdir -p models
curl -sL -o models/efficientdet_lite0.tflite \
  https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
```

### 3. Run inference on the saved video

```bash
conda activate dms-infer
python -m inference.run_video recordings/<name>.mp4
```

This writes an annotated copy to `recordings/<name>_annotated.mp4` and prints a
summary (face-detection coverage, blink count, phone-visible %, yaw/pitch range,
state breakdown).

**Options:**

| Flag | Effect |
|---|---|
| `--show` | Also open a live preview window while processing |
| `--no-save` | Skip the output file, just print the summary |
| *(camera index)* | Pass `0` instead of a path to run on the live camera: `python -m inference.run_video 0 --show` |

> Run it as a module (`python -m inference.run_video`), not `python inference/run_video.py` — the package uses relative imports.
>
> Expect ~6–7 fps on the Orin Nano CPU (MediaPipe FaceMesh + EfficientDet). This
> offline pass is for reviewing a recorded view; real-time live capture needs the
> ONNX/TRT cascade (see `CLAUDE.md`).

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
| `object_detector` | model_path, score_threshold (0.35), max_results (5) |
| `display` | window_title, show_fps |

> Note: `TIRED` is reserved in the state enum but detection (yawning / low vigilance) is not yet implemented. The current pipeline labels FOCUSED / DROWSY / DISTRACTED / NO_FACE.
