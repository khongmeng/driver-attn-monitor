# Raw-pixel end-to-end pipeline

A second, **independent** Stage-④ line: raw video frame → face crop → frozen
pretrained 2D-CNN embedding → GRU → 3-class state (FOCUSED/DISTRACTED/
FATIGUED). No hand-engineered features (head pose, PERCLOS, gaze, MAR)
anywhere in this pipeline — that's the whole point of it (see
`docs/METHODOLOGY.md` §14 for the motivation and full write-up).

## Why a separate directory

Kept deliberately decoupled from `train/cascade/` and
`train/train_state.py`/`train/train_sequence.py` (the feature-based
pipeline), so the two approaches can be trained, evaluated, and iterated on
independently — neither can accidentally break or silently change the
other. The only things reused across the boundary are genuine
**utilities**, not features or results:
- `train.cascade.face_detector.ScrfdFaceDetector` — face bbox, for cropping
  only (never its downstream pose/eye/gaze outputs).
- `train.dmd.dataset` / `train.dmd.annotations` — DMD session discovery and
  raw ground-truth parsing (reading the GT file, not a derived feature).

Everything else — label derivation (`labels.py`), the driver split, the
GRU architecture, chunk-based training, focal loss (`model.py`) — is a
deliberate, documented **duplicate** of the equivalent logic in
`train_state.py`/`train_sequence.py`, not an import, so a future change to
one pipeline's internals can never silently move the other's numbers. The
duplicated algorithms (same seed, same split logic) reproduce the *same*
held-out driver set (`gA_1, gC_14, gE_28, gZ_37`) so results stay directly,
fairly comparable despite the code being independent.

## Pipeline

```
DMD *_rgb_face.mp4
  │
  ▼ extract_crops.py   (SCRFD crop only, + GT label derivation)
train/output/rawpixel/crops/<session>.npz      {crops, frame_idx, state, driver, task}
  │
  ▼ embed_backbone.py  (frozen ImageNet-pretrained 2D-CNN, no fine-tuning)
train/output/rawpixel/embeddings/<session>.npz {embeddings, frame_idx, state, driver, task}
  │
  ▼ train_cnn_gru.py   (trainable linear bottleneck + GRU, chunk-trained,
  │                      full-session evaluated — mirrors train_sequence.py)
models/rawpixel_classifier/three_cnn_gru/state_gru.pt
```

## Usage

```bash
# 1. face crops + labels (~65 sessions; use --limit-sessions/--max-frames for a smoke test)
python -m train.rawpixel.extract_crops

# 2. frozen CNN embeddings (fast — one forward pass per crop, no backward)
python -m train.rawpixel.embed_backbone

# 3. train the GRU on the cached embeddings
python -m train.rawpixel.train_cnn_gru --smooth-window 1
```

Parameters (crop size/margin, face-detector settings, backbone choice, GRU
sizing) live under the `rawpixel:` key in the root `config.yaml`, separate
from the `cascade:` key used by the feature pipeline.

## Three variants

- `embed_backbone.py` + `train_cnn_gru.py` — **2D-CNN, frozen backbone.**
  Caches each frame's embedding once (ResNet18, ImageNet-pretrained), then
  only trains a small linear bottleneck + GRU on top. Result: macro-F1 0.576.
- `train_cnn_gru_finetune.py` — **2D-CNN, fine-tuned backbone.** Unfreezes
  just ResNet18's last residual block (`layer4`) and trains it jointly with
  the bottleneck + GRU (two learning rates: lower for the backbone). No
  caching step — reads raw crops directly. Result: macro-F1 0.695, a real
  improvement over frozen, still below the feature-based GRU's 0.773. Use
  `--freeze-all` to run this script in a frozen-equivalent ablation mode.
- `train_r3d.py` — **3D-CNN clip classifier** (R3D-18, Kinetics-400-
  pretrained, `layer4` fine-tuned). Structurally different from the other
  two: predicts one label per 16-frame clip (broadcast to every frame in
  it for a comparable frame-level macro-F1), not a true per-frame sequence
  output. **Best raw-pixel result: macro-F1 0.712** — beats both 2D-CNN+GRU
  variants, still below the feature-based GRU.

All three are documented in full, including a shared parameter reference
table, in `docs/METHODOLOGY.md` §14.

## Design decisions (see docs/METHODOLOGY.md §14 for the full reasoning)

- **Frozen backbone was tried first, then fine-tuning the last block was
  tried and tested directly** (not just assumed) after the user asked why
  it was frozen given ResNet18 is obviously fine-tunable. 14 subjects is too
  few to safely fine-tune the *whole* 2D-CNN without overfitting to those 14
  faces, and the literature (`arXiv:2205.01721`) found pretrained 2D-image
  features transfer *better* than fully fine-tuned/video-native ones when
  video training data is limited — but empirically, partially unfreezing
  (just the last block) still recovered most of the gap (0.576→0.695), so
  the frozen choice traded away real accuracy for training speed and
  overfitting safety. A small trainable linear bottleneck (`proj_dim`,
  default 128) sits between the CNN embedding and the GRU either way.
- **2D-CNN-per-frame + GRU, not a 3D-CNN or video transformer.** The one
  directly comparable published result found (DAD dataset, 31 subjects) had
  a 2D-per-frame approach match/beat 3D-CNN baselines at this subject-count
  scale, while running ~10x faster. Full video transformers were ruled out
  earlier as needing a GPU cluster we don't have.
- **112×112 crops, ResNet18 backbone (default).** Face-recognition-standard
  crop resolution; ResNet18 is the most common transfer-learning backbone in
  the comparable literature. `--backbone mobilenet_v3_small` is available
  for a lighter-weight comparison.
- **Same GRU sizing (hidden=64, layers=2) as the feature-based
  `SequenceGRU`.** Holds the temporal half of the architecture constant so
  the experiment isolates one variable: hand-engineered features vs. a
  learned CNN embedding as the per-frame representation.
