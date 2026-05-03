# Driver Monitoring System

Real-time driver state detection on Jetson Orin Nano Super + ArduCam IMX477.

Research project — University of St. Thomas.
Conducted by Khongmeng Kormoua, supervised by Dr. Cheol-Hong Min.

## Detected states
- **FOCUSED** — driver looking at road, eyes open
- **DROWSY** — PERCLOS ≥ 20% (eyes closed > 20% of last 60 s)
- **DISTRACTED** — head yaw > 30° or pitch > 20°
- **NO_FACE** — driver not detected

## Running on Jetson

```bash
# From project root
python run_inference.py

# Quit: press Esc
```

## Project layout

```
inference/      Clean Jetson runtime (camera + face analysis + state + overlay)
train/          PC training scaffold (RTX 4070 Ti Super)
models/         Local model weights (gitignored — generate or download)
deepstream/     DeepStream configs + TAO models (future TensorRT pipeline)
assets/         Test videos and sample images
docs/           Dev log, reference links, LaTeX manual
hardware/       Hardware notes and setup photos
archive/        Old exploration scripts (v1–v8) — kept for reference
```

## Environment setup (Jetson, conda `mp` env)

```bash
source ~/miniforge3/bin/activate
conda activate mp
python run_inference.py
```

See `docs/log.txt` for full setup history and troubleshooting.

## Config

All tunable parameters are in `config.yaml` — no need to edit source files.
