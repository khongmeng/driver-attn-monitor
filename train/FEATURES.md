# DMD feature table — schema & conventions

Authoritative reference for the per-frame CSVs produced by
`python -m train.extract_features` (`train/output/features/<session>_features.csv`).
One row = one processed video frame. **Read this before training** — especially
the angle conventions and the ground-truth provenance.

## Columns

### Identity
| Column | Meaning |
|---|---|
| `session` | DMD session name (e.g. `gA_1_s1_...`) |
| `task` | `drowsiness` \| `distraction` (which annotation set) |
| `frame` | 0-based frame index (aligns to the annotation JSON) |

### Face detection (① SCRFD)
| Column | Meaning |
|---|---|
| `has_face` | 1 if a face was detected, else 0 (→ NO_FACE gate) |
| `det_score` | detector confidence |
| `bbox_x0,y0,x1,y1` | face box in pixels |

### Head pose (② 6DRepNet) — degrees, **image frame**
| Column | Meaning |
|---|---|
| `yaw` | **+ = looking image-right, − = image-left** |
| `pitch` | **+ = up, − = down** |
| `roll` | head tilt (6DRepNet native; not used for the 4 states) |

### Eye state (③ open-closed-eye-0001)
| Column | Meaning |
|---|---|
| `left_open_prob`, `right_open_prob` | per-eye open probability (0–1) |
| `eye_open_prob` | mean of the two |
| `eye_closed` | 1 if `eye_open_prob < eye_closed_thresh` (default 0.5) |
| `eye_source` | `model` (ONNX) — provenance flag |

### Eye gaze (⑤ gaze-estimation-adas-0002) — degrees, **image frame**
| Column | Meaning |
|---|---|
| `gaze_yaw` | **+ = image-right, − = image-left** (same frame as head `yaw`) |
| `gaze_pitch` | **+ = up, − = down** (same frame as head `pitch`) |
| `gaze_x,y,z` | normalized gaze unit vector (image frame; forward = −z) |

### Temporal (driven by the eye-state model)
| Column | Meaning |
|---|---|
| `perclos` | fraction of last 60 s with eyes closed (→ DROWSY) |
| `blink` | 1 on the frame a blink completes |
| `blink_count` | cumulative blinks this session |
| `blink_rate` | blinks per minute |

### Temporal — rolling features added by `build_dataset.py` (Stage-④ only, for now)
Computed post-hoc from the columns above (`_add_temporal_features`), not by the
cascade itself — but derived from the model's own signals (not GT), so they
are runtime-computable. `run_live.py` does not compute them yet (see
`docs/METHODOLOGY.md` §8.3); a checkpoint that uses them falls back to the
training mean (neutral) for these columns when run live.

| Column | Meaning |
|---|---|
| `perclos_15s` | fraction of last 15 s with eyes closed (model, not GT) — faster-reacting companion to the 60 s `perclos`, same window as the DMD DROWSY-labeling timescale (§7) |
| `yaw_std_5s` | rolling std of `yaw` over 5 s — sustained head turn vs. single-frame noise |
| `gaze_yaw_std_5s` | rolling std of `gaze_yaw` over 5 s — same idea for gaze |
| `eye_open_prob_mean_5s` | rolling mean of `eye_open_prob` over 5 s — smoothed eye-state signal |

### Ground truth (`gt_*`) — provenance depends on `task`
`None`/blank = **the session does not annotate that level** (unlabeled), *not*
"negative". Only compare/train on non-null GT.

| Column | From | Values |
|---|---|---|
| `gt_eyes_state` | drowsiness | open / close / opening / closing / undefined |
| `gt_eye_closed` | drowsiness | 1 closed, 0 open, None on transition/unlabeled |
| `gt_blink` | drowsiness | 1 / 0 |
| `gt_yawn` | drowsiness | 1 / 0 |
| `gt_occluded` | drowsiness | 1 / 0 |
| `gt_looking_road` | distraction | 1 looking road, 0 not |
| `gt_gaze_off_road` | distraction | 1 not-looking, 0 looking (inverse of above) |
| `gt_driver_action` | distraction | safe_drive / texting_* / phonecall_* / radio / drinking / hair_and_makeup / reach_* / talking_to_passenger / change_gear / unclassified |

> **Drowsiness sessions fill only the eye/blink/yawn GT; distraction sessions
> fill only the gaze/action GT.** No single clip has both — assemble the 4-class
> set by combining sessions (see below).

## Angle conventions (important)

- All head + gaze angles are in the **camera/image frame**, degrees, and share
  one convention: **+yaw = image-right, −yaw = image-left, +pitch = up,
  −pitch = down.** Head and gaze therefore have the **same sign** for the same
  physical direction (verified on known-direction frames; whole-session
  head-vs-gaze yaw agreement 99.5%).
- Implementation notes (in `cascade/`): 6DRepNet's yaw is mirrored vs the image,
  so `pipeline.py` stores `yaw = -pose.yaw`; the gaze model emits an anatomical
  vector so `gaze.py` flips its x (`gv[0] = -gv[0]`) and is fed
  `head_pose_angles = [yaw, -pitch, roll]`. These are sign-only alignments and do
  not change any magnitude, so `|yaw|`/`|gaze|` validation is unaffected.
- **Gaze magnitude** `sqrt(gaze_yaw² + gaze_pitch²)`: ~11° on-road vs ~29°
  off-road on the validated clip.

## Validated signal quality (vs DMD GT)
| Feature | Result |
|---|---|
| Face detection | 100% of frames |
| Eye state vs `eyes_state` | ~98% acc, closed recall ~96% |
| Blink count | cascade ≈ GT (same ballpark) |
| Head `|yaw|` vs `gaze_on_road` | ~80% off-road |
| **Gaze `|gaze|` vs `gaze_on_road`** | **~94% off-road** |

## Suggested Stage-④ target mapping (4 states + NO_FACE)
| State | Rule from GT (to build labels) |
|---|---|
| NO_FACE | `has_face == 0` |
| DROWSY | drowsiness set, GT-PERCLOS ≥ 0.2 over **15 s** (matches DMD's short acted microsleeps; see `docs/METHODOLOGY.md` §7) |
| TIRED | drowsiness set, `gt_yawn == 1` (and/or blink patterns) |
| DISTRACTED | distraction set, `gt_gaze_off_road == 1` or non-safe `gt_driver_action` |
| FOCUSED | `gt_looking_road == 1` + `safe_drive` (distraction) / eyes-open alert (drowsiness) |

Features fed to the classifier: `yaw, pitch, eye_open_prob, perclos, blink_rate,
gaze_yaw, gaze_pitch, perclos_15s, yaw_std_5s, gaze_yaw_std_5s,
eye_open_prob_mean_5s` (all consistent, image-frame; last 4 are the rolling
features above).

## Not yet done (before/round training)
- Full-dataset extraction (16 drowsiness + 49 distraction sessions) — only demo
  clips exist so far. Run on GPU for speed.
- Finalize the label mapping above and materialize a single training table.
- Calibrate `eye_closed_thresh` / gaze / yaw thresholds across multiple drivers.
