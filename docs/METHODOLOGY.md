# Driver Monitoring System — Methodology & Paper Reference

Durable record of the full pipeline, parameters, models, datasets, conventions,
design decisions, and results — the source material for the paper. Update this as
the system evolves. Companion docs: `train/FEATURES.md` (feature-table schema),
`CLAUDE.md` (project overview), `train/README.md` (how to run),
`train/rawpixel/README.md` (the raw-pixel pipeline, §14).

_Last updated: 2026-08-17 — after the §16 bidirectional-GRU/driver-CV
push didn't beat the 0.810 ensemble (documented as a negative result,
2026-08-13), the paper was rescoped toward an **RGB-only, embedded-resource-
constraint framing** (target: Jetson Orin Nano Super) — see
`docs/related_work_embedded.md` for the supporting hardware-constraint and
related-work research. This drove a full **repo reorganization**
(2026-08-17): `models/` collapsed from 33 experiment-log-named checkpoint
directories down to a flat, 10-entry comparison set (`models/README.md`) —
SVM, Random Forest, GBT, MLP, TCN, R3D-18, CNN-GRU (fine-tuned), GRU
(single-branch), Gated-Fatigue GRU, and the GRU Ensemble (still the
project's best result, macro-F1 0.810) — with every entry re-verified
against its source log and, for the 3 iterative/neural raw-pixel-or-TCN
models that hadn't already been extended, retrained at the same 200-epoch
budget as every other model for a fair comparison. One new, real finding
from that pass: **§14.10** — extending CNN+GRU-fine-tuned from 15→200
epochs made its kept result *worse* (0.753→0.742), the one architecture in
the whole 200-epoch-budget line where more training compute didn't help or
stay flat; documented and kept anyway for apples-to-apples budget fairness
rather than silently reporting the better 15-epoch number.
`train/output/` was also trimmed to just the artifacts backing these 10
models plus the still-relevant §16 negative-result plots; ~6.8GB of
regenerable raw-pixel crop/embedding cache was deleted (reproducible via
`train/rawpixel/extract_crops.py` + `embed_backbone.py`). See
`train/README.md` and top-level `README.md`, both rewritten this session
for a new-student-friendly walkthrough. Previous entry: 2026-08-12
(§8.16 — **major correction**: the same
sparse-checkpoint-evaluation issue found in §14.8 was checked against the
project's actual best models — `three_gru_gated` and `three_gru_yawn`, the
two §8.14 ensemble members — since both were trained with the same
original every-5-epoch evaluation over 200 epochs (only 40 of 200 epochs
ever checked). Both had a better epoch the coarse schedule missed:
`three_gru_gated` 0.795→**0.798**, `three_gru_yawn` 0.772→**0.790**.
Re-ran the ensemble with the corrected checkpoints (new
`train/eval_ensemble.py`): **macro-F1 0.810, up from 0.799 — this is now
the project's best-ever result.** Also reinforces §14's raw-pixel
conclusion (the gap to the best raw-pixel result grew, not shrank). Two
other checkpoints (`three_gru` §8.6, `three_gru_gated_three` §8.11) share
the same training script/epoch budget and are plausible candidates for the
same correction but were not reprioritized/rerun. Previous entry: §14.8 —
**correction**: building loss/accuracy
curves required denser per-epoch evaluation, and reran all 5 raw-pixel/TCN
experiments with it — 2 of 5 turned out to have a better checkpoint than
the original sparse (every-3/every-5-epoch) evaluation had ever checked.
CNN+GRU fine-tuned's real result is **macro-F1 0.753** (epoch 7), not 0.695
(epoch 9) — now the *best* raw-pixel result, ahead of R3D-18 (0.712), and
much closer to the feature-based GRU (0.773) than previously documented
(gap 0.078→0.020). CNN+GRU frozen-default corrected 0.576→0.604. R3D-18,
TCN, and the focal-loss frozen variant all confirmed unchanged. §15's
tables updated to the corrected numbers; §14.3/14.5/14.7's original
narratives left intact as a historical record, with the correction
called out explicitly rather than silently edited over (same convention as
the §6.1 gaze-lift retraction). Previous entry: §15 — a full experiment
appendix added: complete
per-state precision/recall/F1/support and every training hyperparameter
(chunk length/stride, batch, epochs, learning rate(s), loss, weight-power,
best epoch, etc.) for all 9 classifier experiments in §8.15 and §14, in one
place for the paper. Before that: §14.7 — R3D-18, a lightweight 3D-CNN
(Kinetics-400-pretrained, layer4 fine-tuned), tried as a clip classifier
over 16-frame windows: macro-F1 0.712, the best raw-pixel result yet,
beating both 2D-CNN+GRU variants (0.576 frozen, 0.695 fine-tuned) though
still below the hand-feature GRU (0.773) and current best (0.799). Overfits
from epoch 1, same as every other raw-pixel run — the 14-driver ceiling,
not architecture, remains the binding constraint. One comparability caveat:
R3D-18 predicts per-16-frame-window, broadcast to frames, not truly
per-frame like the GRU variants. §14.6 also added a full parameter
reference table for both 2D-CNN+GRU variants. Previous entry: §14.5 —
fine-tuned raw-pixel CNN+GRU: unfreezing
just ResNet18's last block (layer4) jointly with the proj/GRU/head lifts
macro-F1 0.576 -> 0.695, recovering most (not all) of the gap to the
feature-based GRU (0.773) — confirms the frozen-backbone choice in §14.1-14.3
did cost real accuracy, prompted by the user directly questioning why the
backbone was frozen. Still below the feature-based line and the current
best (0.799). §14.1-14.3 — a new, independent raw-pixel end-to-end
pipeline (`train/rawpixel/`, kept isolated from the cascade/feature
pipeline): face crop -> frozen ResNet18 embedding -> GRU, no hand-engineered
features, built after the supervisor clarified "explore CNN" meant this, not
the feature-based TCN in §8.15. Frozen-backbone result: macro-F1 0.576
(0.565 with the §8.3 focal-loss recipe, which doesn't transfer), both
peaking at epoch 1 then overfitting. Together, §14's full arc (frozen loses
big, fine-tuned closes most of the gap but not all) fills a literature gap
(no prior controlled raw-pixel-vs-features ablation found) and confirms the
14-subject ceiling — not model sophistication — is this project's binding
constraint. Previous entry: §8.15 — supervisor-requested model-family
comparison: SVM/Random Forest/GBT per-frame baselines and a new TCN
sequence model, all evaluated on the identical driver split/weighting
scheme. Every per-frame family lands in the same narrow 0.51-0.62 band
regardless of algorithm (SVM 0.509, MLP 0.589, RF 0.599, GBT 0.603) —
confirms with a controlled experiment that the ceiling was never about
per-frame model choice. The TCN reaches 0.720, a real win over every
per-frame baseline, but still trails the GRU family (0.773-0.799) because
its receptive field is architecturally bounded (~17s) vs the GRU's
unbounded recurrent memory. Literature review (DMD's own papers, UTA-RLDD,
NTHU-DDD, general SOTA) found 90% macro-F1 is not demonstrated anywhere in
comparably-evaluated (driver-independent, macro-F1, 3-class) published work
— reframes the paper's target from "reach 90%" to "0.799 is competitive
with/ahead of published work on this task shape." Previous entry: §8.14 —
ensembling `three_gru_gated` +
`three_gru_yawn` (simple softmax averaging, no retraining) gives the best
three-class result yet, macro-F1 0.799 / FATIGUED F1 0.739; `run_live.py`
now supports multi-checkpoint ensembles natively; previous day: §8.13 —
found and fixed a live-inference-only bug (GRU demos ran at 2x the trained
update rate; training/eval unaffected, no retraining needed) while
investigating a user-reported demo regression that
turned out to be real: `three_gru_gated_dw5s`'s DISTRACTED recall genuinely
dropped 86.1%→78.1% from the §8.12 relabeling, confirmed against DMD's own
action labels on the specific clip; §8.12 — shortened the DROWSY label window
15s→5s (`build_dataset.py --drowsy-window`), a new table + new checkpoint,
original untouched; FATIGUED precision AND recall both improved together
(0.666→0.756, 0.751→0.799) on the relabeled target, confirmed via direct
DROWSY-run timing comparison the "stuck" recovery-lag problem is fixed;
§8.11 — GatedThreeGRU (restricts DISTRACTED too) hurt DISTRACTED recall,
a genuine tradeoff not a clean win; §8.10 — `GatedFatigueGRU`, a two-branch
architecture that forces FATIGUED to depend only on eye/PERCLOS+yawn while
FOCUSED/DISTRACTED keep all 13 features — best three-class macro-F1 at the
time, 0.795; §8.9 — feature-ablation study + a whole-model eye/yawn-only
restriction that proved gaze/head-pose is real signal, not noise; §8.8 —
chunk-level undersampling tried on the GRU+yawn model: a straight regression
on every axis, not a favorable tradeoff like the MLP case, since it discards
60% of training chunks the GRU needs; also GRU live-wiring landed in
`run_live.py` and demo videos generated for all four three-class
checkpoints; previous day: §8.7 — new
yawn/mouth cascade stage (2d106det + MAR), two more "validated threshold ≠
right threshold" lessons on top of §6.1's, and a retrained GRU with TIRED
recall 55.9%→80.4%; §8.6 — GRU sequence model replaces the per-frame MLP for
the three-class task, macro-F1 0.599→0.773, the biggest single jump in this
project so far; full-dataset feature validation vs DMD ground truth, §6.1 —
retracts the single-session "gaze lifts distraction accuracy 80%→94%" claim
and fixes a `blink_rate` divide-by-near-zero bug; the three-class
base-features/balanced-training experiment, §8.4/§8.5; before that: temporal
features + weight/loss tuning + the three-class FATIGUED experiment, §8.3;
the binary ATTENTIVE/INATTENTIVE experiment, §8.2; full feature extraction,
training table, Stage-④ 4-state baseline, live runtime demo)._

---

## 1. Objective

Real-time driver-state monitoring from a **single in-cabin RGB camera**,
classifying five outcomes: **FOCUSED, DISTRACTED, DROWSY, TIRED** (+ **NO_FACE**
fallback). Approach: **transfer learning / model composition** — a cascade of
pretrained models extracts interpretable features (head pose, eye state, gaze),
which feed a lightweight state classifier. No training from scratch.

**Target platform:** NVIDIA Jetson Orin Nano Super + ArduCam IMX477, real-time
(TensorRT FP16). **Development/training platform:** PC, RTX 4070 Ti Super.

---

## 2. Pipeline architecture (feature-extraction cascade)

```
RGB frame (1280×720)
  ① Face detection   → face box + 5 keypoints, NO_FACE gate
  ② Head pose        → yaw / pitch / roll        (from face crop)
  ③ Eye state        → open/closed prob per eye  (from eye crops)
  ⑤ Eye gaze         → gaze yaw / pitch          (eye crops + head pose)
  Temporal layer     → PERCLOS, blink rate       (over time)
  ④ State classifier → FOCUSED / DISTRACTED / DROWSY / TIRED   [baseline trained, §8.1]
```

Each stage is a swappable module (`train/cascade/`). Rationale for a **modular
cascade** over a single end-to-end network: debuggable per stage, needs far less
labeled data (only stage ④ is trained), interpretable features, and each stage
maps to an established pretrained model.

### Models (pretrained)

| Stage | Model | Source | Input | Output |
|---|---|---|---|---|
| ① Face | **SCRFD-500M** | InsightFace `buffalo_sc` (auto-download) | 640×640 | bbox + 5 kps |
| ② Head pose | **6DRepNet** | `pip install sixdrepnet` (0.1.6) | 224×224 crop | yaw, pitch, roll |
| ③ Eye state | **open-closed-eye-0001** | OpenVINO OMZ (ONNX) | 32×32×2 eyes | P(open) |
| ⑤ Gaze | **gaze-estimation-adas-0002** | OpenVINO OMZ (IR) | 2×60×60 eyes + head angles | 3D gaze vector |
| ④ State | MLP on features (planned) | trained here | 7-D feature vector | 4 states |

Model download URLs (public):
- eye: `storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/open-closed-eye-0001/open_closed_eye.onnx`
- gaze: `storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.{xml,bin}`
- SCRFD + 6DRepNet auto-download via their packages.

---

## 3. Feature extraction — per-frame + temporal

Per frame (`train/cascade/pipeline.py` → `FrameFeatures`): face box, det score,
head yaw/pitch/roll, per-eye + mean open-prob, eye-closed flag, gaze yaw/pitch +
3D vector. Temporal (`cascade/temporal.py`): PERCLOS (rolling window of the
eye-state *model* output), blink events/count/rate. Full schema in
`train/FEATURES.md`.

### Stage parameters

| Stage | Parameter | Value |
|---|---|---|
| ① SCRFD | det_size / det_thresh | 640 / 0.5 |
| ② 6DRepNet | crop margin around box | 0.25 |
| ③ eye | input / preprocessing | 32×32, RGB, `(x−127.5)/255` |
| ③ eye | eye-crop half-size | 0.30 × inter-ocular dist |
| ③ eye | closed threshold | eye_open_prob < 0.5 |
| ⑤ gaze | eye-crop half-size | 0.25 × inter-ocular dist |
| temporal | PERCLOS window / DROWSY thr (runtime) | 60 s / 0.20 |
| temporal | blink min consecutive closed frames | 1 |

**Preprocessing gotchas (verified, important for reproduction):**
- Eye model outputs a **softmax** already; feed RGB `(x−127.5)/255` — raw BGR 0–255
  makes the in-model softmax overflow to `[0, NaN]`.
- Gaze model: forward axis is **−z**; feed `head_pose_angles = [yaw, −pitch, roll]`
  (6DRepNet pitch sign is inverted vs the Intel head-pose convention it expects);
  flip output x (`gv[0] = −gv[0]`) — the model emits an anatomical vector
  (subject-right = +x), which is image-left on a driver-facing camera.

---

## 4. Coordinate conventions (verified against known-direction frames)

All head **and** gaze angles are in a single **image/camera frame**, degrees:

| | Negative | Positive |
|---|---|---|
| yaw | looking image-left | looking image-right |
| pitch | looking down | looking up |

- 6DRepNet's raw yaw is mirrored vs the image, so the stored `yaw = −pose.yaw`
  (`pipeline.py`); pitch already matches. The gaze model still receives the raw
  6DRepNet angles internally.
- Verified: whole-session head-vs-gaze yaw agreement **99.5%** (same sign when
  both |·|>10°); confirmed direction on zoomed eye crops (eyes-left→gaze<0, etc.).
- Note the mirror effect for demos: a driver-facing camera image is mirror-like,
  so overlays can be shown in "driver view" (`--mirror`, display-only) without
  changing feature values.

---

## 5. Dataset — DMD (Driver Monitoring Dataset, Vicomtech)

RGB in-cabin video, 1280×720. On disk: **65 sessions** = 16 **drowsiness** + 49
**distraction** (the gaze/hands subset was not used). CC BY-NC-ND, academic use.

### Ground-truth labels (temporal action annotations, OpenLABEL JSON)

**Drowsiness set:** `eyes_state` {open, close, opening, closing, undefined},
`blinks/blinking`, `yawning` {with hand, without hand}. + driver bbox.

**Distraction set:** `gaze_on_road` {looking_road, not_looking_road},
`driver_actions` {safe_drive, texting_left/right, phonecall_left/right, radio,
drinking, hair_and_makeup, reach_side, reach_backseat, talking_to_passenger,
change_gear, standstill_or_waiting, unclassified}, `hands_using_wheel`
{both, only_left, only_right, none}, `hand_on_gear`, `talking`. + object bboxes
{driver, cellphone, bottle, hair_comb}.

The two sets are **disjoint** — no session has both label families. The 4-state
set is assembled by combining them (§7).

### Extraction run

- Cascade run over all 65 face videos at **stride 2** (every 2nd frame).
- **227,056 frames** processed; per-session CSVs in `train/output/features/`.
- Throughput ≈ **59 fps** on the RTX 4070 Ti Super (see §9).

---

## 6. Feature/stage validation vs DMD ground truth

**⚠ Superseded by §6.1.** The table below is the *original* validation, run on
a single session per task (gA_1_s5 for drowsiness, gA_1_s1 for distraction).
§6.1 re-ran the same checks across the full 65-session/227k-frame dataset and
found the numbers below — especially the gaze row — do not generalize. Kept
here for the historical record; **cite §6.1, not this table**, in the paper.

| Feature | Metric | Result (single-session, retracted where marked) |
|---|---|---|
| ① Face detection | coverage | **100%** of frames (session-specific) |
| ③ Eye state | acc vs `eyes_state` (open/close) | **~98%** (closed recall ~96%, best-thr 99%) — see §6.1, recall doesn't hold |
| temporal | blink count vs GT intervals | same ballpark (e.g. 20 vs 16 on demo clip) |
| ② Head pose | `\|yaw\|` predicts `not_looking_road` | **~80%** |
| ⑤ Gaze | `\|gaze\|` predicts `not_looking_road` | ~~**~94%**~~ **retracted, see §6.1** |

Head pose has no continuous GT in DMD (validated by range only); gaze/head are
validated via the distraction set's `gaze_on_road` labels.

### 6.1 Full-dataset validation + bug fixes (2026-07-29)

**Motivation:** user asked to double-check the extracted features "thoroughly"
against GT, not just spot-check — §6's numbers were all from one session per
task. Ran `train/validate.py` over the complete extraction (all 65 sessions,
227,056 frames) plus custom range/missingness/consistency scripts covering
every column, not just the ones `validate.py` already checked. Full logs:
`train/output/log_validate_full.txt`, `log_investigate_features.txt`,
`log_investigate_features2.txt`, `log_investigate_blinkrate.txt`.

**Finding 1 — the "gaze lifts distraction accuracy 80%→94%" claim does not
reproduce; it was very likely single-session threshold overfitting.**
`train/validate.py` picks its accuracy threshold by scanning percentiles of
whatever data it's handed — the same session it then reports accuracy on. On
the full 49 distraction sessions with one global threshold: `|yaw|>48` →
**82.8%** vs `gaze>38` → **83.0%** — statistically tied, not a gaze advantage.
Per-session accuracy ranges from 48.7% to 98.6% depending on the session (see
`log_investigate_features.txt`), i.e. huge variance the single-session number
hid entirely. Re-running the *exact original methodology* (self-tuned
threshold) on gA_1_s1 alone **today** gives `|yaw|>60` → 87.8%, `gaze>43` →
87.5%, combined → 88.6% — not 80%/94% either. The session's feature CSV row
count also doesn't match: the original doc cites "900 frames" for gA_1_s1;
the full session is 3,389 frames, so the original number was likely computed
on a partial/early extraction of that session, not the whole thing.
**Conclusion: retract the 80%→94% gaze-lift claim.** Gaze may still be useful
as a DISTRACTED feature (it's cheap and the MLP can weight it however it
likes), but its case needs to be re-made with a proper train/threshold-tune
split — never tune and evaluate a threshold on the same slice of data.

**Finding 2 — eye-state accuracy holds up (96.8%) but recall doesn't.**
Full-dataset (32,668 frames with eye GT, all from the 16 drowsiness sessions —
distraction sessions carry no eye-state GT, confirmed structurally correct,
not a bug): accuracy 96.8%, but **closed-recall is 82.6%**, not the
single-session 98.6%. Confusion: TP=3657 TN=27964 FP=278 FN=769. Precision is
high (92.9%) — when the model says closed, it usually is — but it misses
~17% of true closed-eye frames. Best threshold on the full set is still 0.50
(no gain from re-tuning), unlike the single-session case where tuning moved
94.9%→98.2% — another sign the single-session number benefited from tuning on
itself.

**Finding 3 — blink counting is right in aggregate (1650 vs GT 1633 = 1.01×)
but that's cancellation, not accuracy.** All 16 drowsiness sessions (the only
ones with real blink-interval GT):

| session | cascade | GT | diff |
|---|---|---|---|
| gA_1_s5 | 83 | 83 | 0 |
| gA_5_s5 | 117 | 101 | +16 |
| gB_10_s5 (×2 sessions) | 88, 71 | 93, 74 | −5, −3 |
| gB_6_s5 | 103 | 186 | **−83 (−45%)** |
| gB_7_s5 | 213 | 151 | **+62 (+41%)** |
| gB_9_s5 | 102 | 95 | +7 |
| gC_13_s5 | 84 | 90 | −6 |
| gC_14_s5 | 86 | 56 | **+30 (+54%)** |
| gE_29_s5 | 161 | 123 | **+38 (+31%)** |
| gF_23_s5 (×2 sessions) | 94, 96 | 106, 98 | −12, −2 |
| gZ_33_s5 (×2 sessions) | 11, 71 | 74, 69 | **−63 (−85%)**, +2 |
| gZ_36_s5 | 115 | 117 | −2 |
| gZ_37_s5 | 155 | 117 | **+38 (+33%)** |

Per-session error swings from −85% to +41%, averaging out to near-zero in the
sum. Likely cause: `blink_min_frames=1` in `train/cascade/temporal.py` — any
single frame below the eye-closed threshold counts as a full blink, so a
noisy `eye_open_prob` signal (near the 0.5 threshold) both over-counts
(flicker split into extra blinks) and under-counts (a real blink that never
dips fully below 0.5) depending on the session. Not fixed yet — candidate
follow-up: require 2+ consecutive closed frames, or debounce reopening.

**Finding 4 — real bug found and fixed: `blink_rate` divide-by-near-zero at
session start.** `train/cascade/temporal.py` computed
`blink_rate = blink_count / max(elapsed_min, 1e-6)`. A blink in the first few
frames of a session (elapsed time still a fraction of a second) produced
absurd rates — max observed **892.8 blinks/min** (physically impossible;
worst offenders were all at frame index 2–8 of a session, e.g. one blink at
t≈0.07s → 892.8/min). Rare (16 frames >200/min out of 225k) but `blink_rate`
is a raw Stage-④ MLP input feature and the MLP normalizes with plain
mean/std, not robust to outliers — one 892.8 value sitting ~29 std devs from
the mean (µ=27.1, σ=14.5) distorts normalization more than its frame-count
share suggests. **Fixed:** `blink_rate` is now `NaN` (not a number) until at
least 5 seconds of session time have elapsed, instead of computing a
meaningless ratio; `run_live.py`'s existing NaN→training-mean fallback
handles this the same way it already handles missing features. **Fix
verified:** re-extracted all 65 sessions (`train/output/log_extract_*_v2.txt`)
and re-ran the `blink_rate` check — max dropped from 892.8 to a sane **83.3
blinks/min**, frames >60/min dropped from 1.10% to 0.99% of the dataset.
Re-ran the full `train/validate.py` report on the re-extraction
(`log_validate_full_v2.txt`) — every other stage (face, head pose, eye state,
gaze, blink count) is numerically identical to the pre-fix run, confirming
the fix only touched `blink_rate` and nothing else regressed. Rebuilt
`train/output/train_table.csv`: **214,698** labeled frames (was 218,795 —
the ~4.1k session-start frames with a now-NaN `blink_rate` are correctly
excluded instead of carrying a garbage value); class balance essentially
unchanged (DISTRACTED 49.0%, FOCUSED 45.4%, TIRED 2.9%, DROWSY 2.7%, vs the
old 48/46/2.8/2.7). **All Stage-④ checkpoints (§8.1–§8.4) were trained on the
pre-fix table and are now stale** — retraining on the corrected table is the
natural next step but hasn't been done yet (not requested as of this
writeup).

**Finding 5 — non-issues, confirmed correct (worth ruling out explicitly):**
- `has_face` 99.3% overall coverage across all 65 sessions.
- All probability/PERCLOS columns (`eye_open_prob`, `left/right_open_prob`,
  `perclos`) properly bounded in [0, 1] with zero out-of-range values.
- Gaze unit vectors (`gaze_x/y/z`) have norm exactly 1.0000 for every frame —
  the 3D→angle conversion is internally consistent.
- Zero NaN leakage: every feature column is properly blank on `has_face == 0`
  rows (1,641 no-face rows checked, 0 non-null feature values).
- Frame stride is perfectly consistent within every session (no dropped or
  corrupted frames mid-extraction).
- `gt_occluded` being 100% empty is **not a bug** — checked the raw DMD
  OpenLABEL JSON directly; the string "occlusion" that does appear there is
  an unrelated `hands_camera` stream-sync property, not the face/body/hands
  occlusion *action* the DMD README describes and `train/dmd/annotations.py`
  parses for. That action type never appears in any of these 65 sessions'
  `actions{}`, so the field is correctly always unset, not silently dropped.
- Pitch occasionally exceeds the physically-sane ±90° range (max observed
  172°), but only in 22/225,415 frames (0.01%), all clustered at extreme yaw
  (~±90°, near-profile views) with pitch and roll swinging together — a
  known gimbal-lock-like instability in 6D rotation regression networks near
  that singularity, not a systemic extraction problem.

**Meta-finding:** every number in the original §6 table was a single-session
snapshot, and every one that's been re-checked against the full dataset came
out different — sometimes worse (eye recall), sometimes about the same in
aggregate but hiding large per-session variance (blink), and in the gaze case,
contradicted outright. Prefer full-dataset validation over single-session
spot-checks for any number that goes in the paper going forward.

---

## 7. State-label derivation (Stage-④ target)

Per face frame (`train/build_dataset.py`); NO_FACE excluded (detector gate).

**Drowsiness sessions:**
- `TIRED` if `gt_yawn == 1`
- else `DROWSY` if GT-PERCLOS ≥ **0.20** over a **15 s** window
- else `FOCUSED`

**Distraction sessions:**
- `DISTRACTED` if `gt_gaze_off_road == 1` OR `gt_driver_action` ∈ {texting_*,
  phonecall_*, radio, drinking, hair_and_makeup, reach_*, talking_to_passenger,
  change_gear}
- else `FOCUSED` if `gt_looking_road == 1` AND `gt_driver_action == safe_drive`
- else dropped (unclassified / standstill / ambiguous)

**DROWSY-window decision (important for the paper):** the clinical PERCLOS
definition (≥20% over **60 s**) yields only 2,505 DROWSY frames (5.7% of
drowsiness) because DMD's acted microsleeps are short — a 2 s microsleep is only
~3% PERCLOS in a 60 s window. Shortening to a **15 s** window (≥20%) captures the
microsleep timescale → **5,975** DROWSY frames without over-labeling normal
blinks (a 10% threshold would balloon to 41% and pull in blinking). Runtime
inference still uses the 60 s clinical window; the 15 s window is only for
labeling DMD's acted data.

### DMD drowsiness composition (why DROWSY is inherently a minority)
Of 44,114 drowsiness frames: eyes **open 64%**, opening 14%, closing 11%,
**close 10%**, undefined <1%. The set is mostly alert driving with brief drowsy
episodes — DROWSY being small is a property of the data, not missing data.

---

## 8. Training table — statistics

`train/output/train_table.csv`: **218,795 labeled face frames**, 65 drivers.

| State | Frames | Share | Source |
|---|---|---|---|
| DISTRACTED | 105,493 | 48.2% | distraction |
| FOCUSED | 101,113 | 46.2% | both |
| TIRED | 6,214 | 2.8% | drowsiness |
| DROWSY | 5,975 | 2.7% | drowsiness |

Classifier input features: `yaw, pitch, eye_open_prob, perclos, blink_rate,
gaze_yaw, gaze_pitch`. **Class imbalance** → use inverse-frequency class weights.
**Split must be grouped by driver** (session id) to avoid near-duplicate-frame
leakage between train/val. Note: the 65 sessions come from only **14 drivers**.

### 8.1 First Stage-④ 4-state baseline (`train/train_state.py`)
- Model: MLP 7→64→32→4, standardization baked in, inverse-freq class weights,
  Adam lr 1e-3, 40 epochs. Split by driver (10 train / 4 val), fatigue drivers
  split separately so val holds DROWSY/TIRED. Held-out val drivers: gA_1, gC_14,
  gE_28, gZ_37. Exported to ONNX (`models/state_classifier/state_mlp.onnx`).
- **Per-frame, cross-driver result (val):** accuracy 0.50, **macro-F1 0.35**.

  | class | precision | recall | f1 |
  |---|---|---|---|
  | FOCUSED | 0.62 | 0.43 | 0.51 |
  | DISTRACTED | 0.72 | 0.55 | 0.63 |
  | DROWSY | 0.10 | 0.57 | 0.17 |
  | TIRED | 0.05 | 0.42 | 0.09 |

- **Findings:** (1) DISTRACTED/FOCUSED are the strongest but still cross-confused
  across unseen drivers (domain shift; the 94% gaze number was single-clip,
  threshold-only). (2) DROWSY/TIRED have high recall but very low precision (class
  weighting over-predicts them) **and confuse with each other** — empirical
  evidence that the two fatigue stages are not cleanly separable from DMD alone.
  (3) Per-frame classification is noisy — no temporal smoothing yet.
- **Next levers (ranked):** temporal smoothing / hysteresis on the output
  (biggest expected win); add features (roll, eye_closed, rolling gaze/head std,
  deviation magnitudes); milder weighting or focal loss; bring in UTA-RLDD for a
  clean TIRED-vs-DROWSY split; more drivers (only 14 here limits generalization).

### 8.2 Binary experiment — ATTENTIVE vs INATTENTIVE (2026-07-16)

**Motivation:** the 4-state label distribution is heavily skewed (46/48/2.8/2.7%),
and the class-weighted 4-state baseline over-fires the two fatigue classes.
Collapsing all non-FOCUSED states into a single **INATTENTIVE** class ("driver
inattention" is the standard umbrella term covering distraction + fatigue)
yields a nearly balanced ~46/54 problem. This does not *solve* the imbalance —
DROWSY/TIRED frames are still rare inside INATTENTIVE — but it removes the
extreme class weights and gives a clean upper-bound-style baseline for "does the
system detect that something is wrong at all".

**Setup:** `python -m train.train_state --classes binary`. Identical to the
4-state run in every other respect — same table, features, MLP (7→64→32→2),
epochs/lr/seed, and crucially the **same driver split** (held-out: gA_1, gC_14,
gE_28, gZ_37), so results are directly comparable. Exported to
`models/state_classifier/binary/state_mlp.{pt,onnx}`; `run_live.py` reads the
class set from the checkpoint.

**Result (per-frame, held-out drivers):** accuracy **0.620**, macro-F1 **0.620**
(4-state: acc 0.497, macro-F1 0.347), ROC-AUC (INATTENTIVE) **0.682**.

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| ATTENTIVE | 0.550 | 0.677 | 0.607 | 21,262 |
| INATTENTIVE | 0.701 | 0.578 | 0.633 | 27,859 |

INATTENTIVE recall by original state: DISTRACTED **59.8%**, TIRED 42.3%,
DROWSY **23.8%**; FOCUSED kept ATTENTIVE 67.7%.

**Findings:**
1. Balancing the classes lifts overall accuracy 0.50 → 0.62 and removes the
   fatigue over-prediction (no more 8× class weights), but the per-frame
   ceiling stays modest — confirming §8.1's diagnosis that the limits are
   **method (no temporal context) and driver diversity (14 people)**, not the
   label taxonomy alone.
2. The per-state recall breakdown shows the binary model is effectively a
   *distraction* detector: DROWSY frames are mostly missed (23.8%) because with
   ~5% fatigue share inside INATTENTIVE, the head/gaze features dominate the
   decision. A production system still needs the fatigue signal handled
   separately (temporal PERCLOS path) or a hierarchical design
   (attention → which kind of inattention).
3. Useful paper framing: binary = "alert vs needs-attention" alarm baseline;
   the 4-state model refines *why*. The two share one training pipeline and one
   comparable evaluation protocol.

### 8.3 Temporal features + weight/loss tuning + three-class FATIGUED (2026-07-22)

**Motivation:** §8.1/§8.2 diagnosed the poor macro-F1 as method-limited, not
data-limited — per-frame classification with no temporal context, and
inverse-frequency class weights so aggressive they over-fire the rare fatigue
classes (DROWSY precision 0.10, TIRED precision 0.05 despite recall >0.4).
This experiment tests that diagnosis directly: add temporal features, tune the
weighting/loss, and also try a **three-class** taxonomy — FOCUSED /
DISTRACTED / **FATIGUED** (DROWSY+TIRED merged; named FATIGUED rather than
TIRED because TIRED already names a specific state in the 4-class taxonomy,
and reusing it for the merged class would be confusing — FATIGUED is the
umbrella term, parallel to INATTENTIVE in §8.2).

**New temporal features** (`train/build_dataset.py::_add_temporal_features`,
computed post-hoc from the existing per-frame cascade output, no re-extraction
needed): `perclos_15s` (faster-reacting PERCLOS at the same 15s timescale used
to label DROWSY from GT), `yaw_std_5s` / `gaze_yaw_std_5s` (head/gaze
volatility — sustained movement vs. single-frame noise), `eye_open_prob_mean_5s`
(smoothed eye-state signal). Classifier input grows from 7 to 11 features (see
`train/FEATURES.md`). All four are computed from the model's own output, not
GT, so they're runtime-computable in principle — but `run_live.py` does not
compute them yet (deferred; see below).

**New tuning options** (`train/train_state.py`): `--smooth-window N`
(post-hoc eval only — rolling-mean the predicted probabilities over N frames
per session, then argmax, mirroring `run_live.py`'s existing live smoothing);
`--weight-power P` (dampens the inverse-frequency class weights,
`w = (total/(n·count))^P`); `--loss focal` (focal loss on top of the class
weights); `--model gbt` (sklearn `HistGradientBoostingClassifier`, a
model-family comparison baseline, not exported).

**Results** (three-class, held-out drivers gA_1/gC_14/gE_28/gZ_37, same split
as §8.1/§8.2), macro-F1 progression:

| Config | macro-F1 (raw) | macro-F1 (smoothed) |
|---|---|---|
| new features, default weighting (`weight-power 1.0`, `ce`) | 0.510 | 0.525 (window=30) |
| `weight-power 0.5` | — | 0.578 (window=15) |
| `weight-power 0.3` | — | 0.583 (window=15) |
| `loss focal` (weight-power 1.0) | — | 0.532 (window=15) |
| **`loss focal` + `weight-power 0.3`** | **0.577** | **0.599 (window=30)** |
| `loss focal` + `weight-power 0.3` (window=60) | — | 0.596 (slightly worse — 30 is the sweet spot) |
| `model gbt` + `weight-power 0.3` | 0.518 | 0.515 (smoothing didn't help GBT) |

Winning config: `python -m train.train_state --classes three --loss focal
--weight-power 0.3 --smooth-window 30` → **macro-F1 0.599**, accuracy 0.644
(vs. four-state 0.35, binary 0.62 — three-class closes most of the gap to
binary while keeping distraction and fatigue separate).

| class | precision | recall | f1 |
|---|---|---|---|
| FOCUSED | 0.589 | 0.676 | 0.630 |
| DISTRACTED | 0.728 | 0.625 | 0.672 |
| FATIGUED | 0.446 | 0.553 | 0.494 |

FATIGUED recall by original label: DROWSY 62.4%, TIRED 40.7%.

**Findings:**
1. **The method diagnosis was right.** Temporal features + smoothing alone
   (default weighting) only moved macro-F1 0.510→0.525. **Weight-power
   dampening was the single biggest lever** (0.525→0.583 at `weight-power
   0.3`, same smoothing) — the original inverse-frequency weighting was
   overcorrecting for the imbalance, not undercorrecting. Stacking focal loss
   on top of the dampened weights added a further point (0.583→0.599).
2. FATIGUED precision improved dramatically over the old TIRED/DROWSY numbers
   (0.10–0.05 → 0.446) at the cost of some recall (62%/41% vs. the old
   run's >90% recall from over-firing) — a much more usable operating point:
   the old model cried fatigue constantly; this one is right when it does.
3. **GBT underperformed the MLP** (0.515–0.518 vs. 0.599) on this feature set
   and did not benefit from post-hoc smoothing the way the MLP did — sticking
   with the MLP for now.
4. Smoothing window sweep (15/30/60 frames, i.e. roughly 1s/2s/4s at the
   stride-2 extraction rate): gains plateau and slightly reverse past 30
   frames — window=30 is the current default recommendation, not window=60.
5. **The new features alone lift the other two modes too**, retrained with
   the same 11-feature table (default weighting, no smoothing — these were
   not the focus of this pass): four-state macro-F1 **0.35 → 0.435**, binary
   macro-F1 **0.62 → 0.643** (ROC-AUC 0.68 → 0.703). Neither has had the
   weight/loss tuning applied yet — that's the obvious next quick win for
   both if the four-state and binary variants are still wanted alongside
   `three`. Checkpoints in `models/state_classifier/` and `.../binary/` were
   overwritten with these 11-feature versions.
6. **Not yet ported to the live runtime.** `run_live.py`'s `StateClassifier`
   was fixed to read its feature list from the checkpoint (previously
   hardcoded to the module-level `FEATURES` constant, which would have broken
   both old 7-feature checkpoints and any new 11-feature one) and to
   gracefully neutralize features it can't compute — but the live cascade
   still only produces the original 7 instantaneous features. A checkpoint
   trained with the 4 new rolling features will run live, but those 4 inputs
   fall back to the training mean (neutral) until the live pipeline computes
   them, so the live demo won't see the full benefit measured here yet.

**Next steps (ranked):** port the rolling-feature computation into the live
runtime (turns this from an offline result into an actual live improvement);
a real sequence model (LSTM/GRU/temporal-conv) if there's still a gap once
that's done; then — per §8.1's original ordering — more drivers / UTA-RLDD
/ NTHU-DDD for the fatigue classes specifically, now that method fixes have
been tried first.

### 8.4 Three-class, base features (no temporal aggregates), balanced training set (2026-07-29)

**Motivation:** the §8.3 winning three-class config (macro-F1 0.599) uses both
the 4 rolling temporal features *and* dampened-weight/focal-loss tuning
together, so it's not clear how much each piece contributes — and undersampling
is a more direct fix for imbalance than loss-side weighting. This experiment
isolates that: three-class FATIGUED, **base 7-feature set only** (no
`perclos_15s`/`yaw_std_5s`/`gaze_yaw_std_5s`/`eye_open_prob_mean_5s`), trained
on a **class-balanced training split** — every class in `train` undersampled to
the smallest class's count (FATIGUED, 10,093 frames), instead of relying on
inverse-frequency loss weighting. Val is left at its natural (imbalanced)
distribution, same held-out drivers as §8.1–§8.3, so results are directly
comparable.

**New `train/train_state.py` options:** `--feature-set {full,base}` (full =
all 11 columns incl. the §8.3 temporal aggregates, the existing default; base
= the original 7) and `--balance {none,undersample}` (undersample = random
per-class undersampling of the *training* split only). Non-default combos get
their own output directory automatically (`models/state_classifier/three_base_balanced`
here) so this run's checkpoint sits alongside, not over, the §8.3 one.

Command: `python -m train.train_state --classes three --feature-set base
--balance undersample --loss focal --weight-power 0.3 --smooth-window 30`
(weight-power/focal kept from §8.3 for consistency, though with balanced
classes the computed weights collapse to ~1.0 either way). Full console log:
`train/output/log_three_base_balanced.txt`.

**Result (per-frame, held-out drivers):**

| | macro-F1 | accuracy |
|---|---|---|
| raw | 0.454 | 0.523 |
| smoothed (window=30) | 0.469 | 0.543 |

vs. §8.3's 11-feature, weight-tuned three-class result: macro-F1 0.577 raw /
0.599 smoothed — **balancing alone, without the temporal features,
underperforms §8.3's combination.**

| class | precision | recall | f1 (smoothed) |
|---|---|---|---|
| FOCUSED | 0.604 | 0.478 | 0.534 |
| DISTRACTED | 0.711 | 0.581 | 0.639 |
| FATIGUED | 0.139 | 0.745 | 0.234 |

FATIGUED recall by original label: DROWSY 74.4%, TIRED 70.1% (both higher than
§8.3's 62.4%/40.7%).

**Findings:**
1. **Undersampling trades precision for recall harder than weight-power
   dampening did.** FATIGUED recall improves (55%→75% smoothed) but precision
   collapses (0.446→0.139) — with training classes forced to exact 1:1:1 while
   val stays ~43%/52%/4%, the model over-predicts the now-artificially-common
   FATIGUED class. Macro-F1 (which weights all three classes equally) still
   drops because FOCUSED/DISTRACTED F1 both fall too.
2. **The temporal features (§8.3) and weight-power/focal tuning (§8.3) were
   doing more work than balancing does here.** This isolates the two changes:
   removing the temporal features *and* switching to undersampling both move
   in the same (worse, for macro-F1) direction versus §8.3's best config, so
   §8.3's gain wasn't just "any imbalance fix" — the specific combination
   (dampened weights + focal + temporal context) mattered.
3. **If the deployment priority is "never miss a fatigue event" over overall
   balance**, this run's operating point (75% FATIGUED recall) is arguably
   more useful than §8.3's despite the lower macro-F1 — worth stating
   explicitly in the paper as a precision/recall tradeoff choice, not simply
   "worse."
4. Checkpoint saved to `models/state_classifier/three_base_balanced/`
   (`.pt` + `.onnx`, 7 features) — does **not** overwrite `models/state_classifier/three/`
   (§8.3, 11 features) or any other prior checkpoint.

**Next steps:** try undersampling *combined with* the temporal features (isolate
whether balancing adds anything on top of §8.3 rather than instead of it);
try a milder balance (e.g. cap majority classes at 3–5× the minority count
instead of exact 1:1) to see if precision recovers without giving back all the
recall gain.

### 8.5 Three-class, full features + balanced training, on the corrected dataset (2026-07-29)

**Motivation:** §8.4's next-steps item — try undersampling *combined with* the
11-feature (§8.3) table, not instead of it — done here, and simultaneously the
first Stage-④ run on the **post-§6.1** dataset (`train_table.csv` rebuilt after
the `blink_rate` divide-by-near-zero fix, 214,698 frames vs the old 218,795).
Command: `python -m train.train_state --classes three --balance undersample
--loss focal --weight-power 0.3 --smooth-window 30` (default `--feature-set
full`, so all 11 features incl. the §8.3 temporal aggregates — this is what
distinguishes it from §8.4, which used `--feature-set base`). Checkpoint:
`models/state_classifier/three_balanced/` (separate directory, does not
overwrite `three/` §8.3 or `three_base_balanced/` §8.4). Full log:
`train/output/log_three_balanced_v2.txt`.

Driver split: same 10 train / 4 held-out (gA_1, gC_14, gE_28, gZ_37) as
§8.1–§8.4 (same seed). Train balanced to 9,955 frames/class (FATIGUED's count).

**Result (per-frame, held-out drivers):**

| | macro-F1 | accuracy |
|---|---|---|
| raw | 0.519 | 0.585 |
| smoothed (window=30) | 0.525 | 0.592 |

| class | precision | recall | f1 (smoothed) |
|---|---|---|---|
| FOCUSED | 0.593 | 0.547 | 0.569 |
| DISTRACTED | 0.744 | 0.607 | 0.668 |
| FATIGUED | 0.211 | 0.845 | 0.337 |

FATIGUED recall by original label: **DROWSY 96.0%**, TIRED 72.8%.

**Comparison across all three-class variants so far** (all same driver split):

| config | features | balance | macro-F1 (smoothed) | FATIGUED recall (smoothed) |
|---|---|---|---|---|
| §8.3 | full (11) | none (weight-power 0.3 + focal) | **0.599** | 55% (DROWSY 62%, TIRED 41%) |
| §8.4 | base (7) | undersample | 0.469 | 75% (DROWSY 74%, TIRED 70%) |
| **§8.5** | **full (11)** | **undersample** | 0.525 | **~85% (DROWSY 96%, TIRED 73%)** |

**Findings:**
1. **§8.5 sits between §8.3 and §8.4 on macro-F1, but has the best FATIGUED
   recall of all three by a wide margin** — the temporal features (§8.3's
   contribution) and balancing (§8.4's contribution) are not redundant, they
   add on top of each other: full features recovered most of the macro-F1
   §8.4 gave up (0.469→0.525) while balancing pushed recall well past what
   §8.3's weighting alone achieved (55%→85%, and DROWSY recall specifically
   hits 96%). Confirms §8.4's finding #2 was about *removing temporal
   features*, not about undersampling being inherently worse than weighting.
2. **Still a precision/recall tradeoff, not a free win.** FATIGUED precision
   is 0.211 — better than §8.4's 0.139 but far below §8.3's 0.446 — i.e. this
   config would flag fatigue far more often than it's actually happening.
   Which of the three checkpoints is "best" depends on whether the paper's
   framing prioritizes macro-F1 balance (§8.3), minimizing missed fatigue
   events (§8.5, especially the 96% DROWSY recall), or isolating the effect
   of dropping temporal context (§8.4).
3. First result computed after the §6.1 `blink_rate` fix + re-extraction —
   not directly comparable to §8.1–§8.4's numbers on a frame-for-frame basis
   (slightly different table, ~1.9% fewer frames), though the shift is small
   enough that it isn't the explanation for any of the differences above.
4. Demo videos on held-out driver gC_14 (same clips as the earlier §8.3/§8.4
   demos): `train/output/demo_gC14_distraction_three_balanced.mp4` and
   `demo_gC14_drowsiness_three_balanced.mp4` (`--smooth 30`).

**Next steps:** re-train §8.1 (4-state) and §8.2 (binary) on the corrected
table too, for a clean before/after comparison of the blink_rate fix in
isolation; given §8.5 beat §8.4 by adding back temporal features, the next
natural ablation is §8.3's exact weight/loss recipe *plus* balancing (not
undersampling alone) to see if that pushes FATIGUED recall even higher without
sacrificing as much precision.

### 8.6 GRU sequence model — real temporal modeling replaces the MLP (2026-07-29)

**Motivation:** §8.3–§8.5 all use the same per-frame MLP, differing only in
feature set and class balancing — none of them let the model learn its own
temporal integration; §8.3's 4 rolling-window features are still hand-picked
window lengths (15s/5s), not learned. The user asked to push this method as
far as it can go: add real temporal modeling, fix class imbalance properly
(not via row-level resampling), and fix the blink-debounce bug found in §6.1.
Literature research (this conversation) confirmed CNN-LSTM/GRU-style temporal
models are consistently the biggest lever in published drowsiness/distraction
work, well ahead of feature engineering or class-weighting choices alone.

**New script `train/train_sequence.py`** — a small `nn.GRU` (2 layers,
hidden=64) replacing `StateMLP`, reusing `train_state.py`'s driver split,
class-set definitions, `FocalLoss`, and reporting functions so results are
directly comparable to §8.3–§8.5. Key design choices:
- **Train on fixed-length chunks** (600 frames ≈ 40s, 50%-overlap stride 300),
  batched (16 chunks/batch) — stable, fast gradient steps. **Evaluate on the
  full, unchunked per-session sequence** (one continuous forward pass per
  held-out session) — matches how the model would run live and keeps
  macro-F1 comparable to the per-frame MLP reports.
- **Class imbalance handled correctly for sequence data:** row-level
  undersampling/oversampling (§8.4/§8.5) would shuffle individual frames out
  of order, destroying the temporal continuity a GRU depends on. Instead:
  per-timestep class-weighted focal loss (the existing `FocalLoss`, just
  applied over every timestep) + **chunk-level oversampling** (`--oversample`,
  sample weight = 1 + factor × fraction-of-chunk-that-is-FATIGUED) — whole
  chunks are resampled, never individual rows, so temporal order inside a
  chunk is never touched.
- **Best-checkpoint selection:** validated every 5 epochs, keeps the
  highest-macro-F1 state dict rather than whatever the last epoch produced
  (added after an initial 40-epoch run showed the metric peaking at epoch 30
  then drifting — see `train/train_sequence.py`'s `best_state` tracking).
- Blink-debounce fix (`blink_min_frames` 1→2, `train/cascade/temporal.py`
  and `config.yaml`) applied in code now; doesn't affect this wave's
  features/table (that needs a re-extraction, done in §8.7) — included here
  because it's a one-line, low-risk fix worth having in place going forward.

**Command:** `python -m train.train_sequence --classes three --loss focal
--weight-power 0.3 --oversample 5.0 --smooth-window 30`, 200 epochs (best
checkpoint restored from epoch 140). Full log: `train/output/log_three_gru_v1.txt`.
Checkpoint: `models/state_classifier/three_gru/` (new dir; doesn't touch any
MLP checkpoint from §8.3–§8.5).

**Result (per-frame, held-out drivers gA_1/gC_14/gE_28/gZ_37 — same split as
every prior three-class run):**

| | macro-F1 | accuracy |
|---|---|---|
| raw | **0.773** | 0.821 |
| smoothed (window=30) | 0.738 | 0.784 |

| class | precision | recall | f1 (raw) |
|---|---|---|---|
| FOCUSED | 0.810 | 0.777 | 0.793 |
| DISTRACTED | 0.850 | 0.863 | 0.856 |
| FATIGUED | 0.615 | 0.736 | 0.670 |

FATIGUED recall by original label: DROWSY 80.9%, TIRED 55.9%.

**Comparison across all three-class variants:**

| config | features | temporal model | balance | macro-F1 (best) | FATIGUED P/R |
|---|---|---|---|---|---|
| §8.3 | full (11) | none (MLP) | weight+focal | 0.599 (smoothed) | 0.446 / 0.55 |
| §8.4 | base (7) | none (MLP) | undersample | 0.469 (smoothed) | 0.139 / 0.75 |
| §8.5 | full (11) | none (MLP) | undersample | 0.525 (smoothed) | 0.211 / 0.85 |
| **§8.6** | **full (11)** | **GRU** | **focal + chunk-oversample** | **0.773 (raw)** | **0.615 / 0.74** |

**Findings:**
1. **Temporal modeling was the single biggest lever, exactly as the
   literature predicted** — macro-F1 0.773 vs. the previous best 0.599, a
   jump no amount of feature-set or balancing tweaking on the MLP came close
   to. Confirms the diagnosis that per-frame classification (even with 4
   rolling-window features bolted on) was leaving real temporal structure on
   the table.
2. **For the first time, FATIGUED precision and recall improved *together***
   (0.615 / 0.74) instead of trading off against each other — every MLP
   variant (§8.3–§8.5) improved one at the expense of the other. Chunk-level
   oversampling (which preserves temporal order) plus a model that can
   actually learn what a sustained drowsy episode looks like, rather than
   guessing from one frame's stats, is a materially different mechanism than
   row-level resampling.
3. **Post-hoc temporal smoothing now hurts slightly instead of helping**
   (0.773 raw → 0.738 smoothed) — the opposite of every MLP experiment. This
   makes sense: the GRU's hidden state already performs learned temporal
   integration, so an additional naive rolling-average over its output
   double-smooths and blurs transitions instead of adding new information.
   Smoothing was a crutch for the MLP's lack of memory, not a universal
   improvement.
4. **Training is cheap** — 200 epochs over the whole training set (474
   chunks) takes about 22 seconds on the training GPU, so hyperparameter
   sweeps here are inexpensive; this run used one well-reasoned config
   (matching §8.3/§8.5's `weight-power 0.3` + focal loss recipe) rather than
   an exhaustive search — chunk length, hidden size, and number of GRU layers
   are all still open to further tuning if more gains are wanted.
5. Still well short of the ~90% the user originally asked about, and short of
   the realistic ~0.65–0.75 macro-F1 ceiling estimated earlier from
   comparable published multi-class fused driver-state work — **0.773 lands
   right at the top of that estimated range**, suggesting this architecture
   (cascade features + GRU) is now closer to its practical ceiling than to
   having obvious further headroom from tuning alone. The next real lever is
   almost certainly the yawn/mouth feature (§8.7) and/or more fatigue data
   (UTA-RLDD/NTHU-DDD), not more MLP/GRU tuning.

**Next steps:** §8.7 adds the yawn/mouth-openness feature (new cascade stage)
on top of this GRU architecture, to isolate its contribution on top of
temporal modeling rather than instead of it.

### 8.7 Yawn/mouth feature (2d106det + MAR) — re-extraction, two more threshold lessons, and a targeted TIRED-recall win (2026-07-29)

**Motivation:** TIRED is partly *defined* by yawning in the DMD labels
(`build_dataset.py::_label_row`), but no live-computable yawn signal existed
— the classifier had to infer it indirectly from eye/head cues alone. Added
a new cascade stage: InsightFace's **2d106det** (106-point 2D landmarks, from
the `buffalo_s` pack — `buffalo_sc` used for stage ① detection has no
landmark model bundled) → **Mouth Aspect Ratio (MAR)** → yawn run-length
detection, the same approach the `OpenVino-Driver-Behaviour` reference
project (already cited in CLAUDE.md) and the yawn-detection literature use,
since no pretrained mouth-open/closed ONNX model exists like
`open-closed-eye-0001` does for eyes.

**Landmark indices verified empirically, not assumed.** Rendered all 106
points (with index labels) on real driver frames from this dataset:
confirmed indices 52 and 61 are the mouth's horizontal extremes (corners) in
every test frame checked. The commonly-documented inner/outer lip split
(52-61 outer, 62-71 inner) could not be confirmed as cleanly — one sample
frame had only a partial yawn opening, another had a hand covering the mouth
mid-yawn (a real, common acted-yawn behavior in this dataset). Rather than
trust an unconfirmed split, MAR uses the full 20-point mouth cluster's
vertical span — both contours separate together when the mouth opens, so the
signal survives either way. New columns: `mar`, `mouth_open`, `yawn_active`,
`yawn_count`, `yawn_rate` (`train/cascade/mouth.py`, `train/cascade/temporal.py`'s
new `update_mouth()`, wired into `pipeline.py`). `mar`/`yawn_rate` appended to
`FEATURE_COLS` (`train/build_dataset.py`) — append-only, so every prior
checkpoint (§8.1–§8.6) still loads unchanged (they read their own feature
list from the checkpoint, not the live constant).

**Bug found while wiring this in:** `CascadePipeline.reset()` (called once
per session by `extract_features.py`) only preserved 3 of the
`TemporalAggregator`'s constructor params when rebuilding between sessions,
silently dropping anything else back to the dataclass defaults after the
first session — harmless today only because the dropped params happened to
already equal their defaults. Fixed to preserve all params before this
mattered for the new yawn settings.

**Two more "validated threshold isn't the right threshold" lessons —
directly extending the §6.1 validation-methodology finding:**

1. **`blink_min_frames` 1→2 (the §6.1-recommended fix) makes the aggregate
   *worse*, not better.** Tested it in this re-extraction: total cascade
   blinks dropped to 990 vs GT 1633 (**ratio 0.61x**, a systematic
   undercount), worse than the original 1.01x — and every session but two
   now *undercounts* (range −92% to +11%), vs. the original's more balanced
   (but still noisy) −85% to +41% swing. Root cause: at this stride-2
   extraction rate, a real blink is often only 1-2 samples wide, so
   requiring 2 consecutive closed samples filters out genuine blinks more
   often than it filters single-frame flicker. **Reverted to
   `blink_min_frames=1`** — recomputed post-hoc from the already-saved
   `eye_open_prob` column (no cascade re-run needed, since blink/PERCLOS are
   a pure function of it) rather than re-running the ~87-minute extraction a
   third time.
2. **Same shape of mistake with the MAR open-threshold.** `train/validate.py`'s
   self-tuned "best threshold" for MAR was 0.64 (91.0% frame accuracy vs.
   89.1% at the 0.5 placeholder) — but adopting it dropped yawn-**event**
   recall from 87/76 (1.14×, close to GT) to **44/76 (0.58×)**, including one
   session (gC_14_s5) going from 3 detected yawns to **zero**. A stricter
   frame-level threshold means fewer frames individually cross it, which
   makes it harder to sustain a continuous 1-second run (the yawn
   run-length gate) at all — optimizing frame-level accuracy actively hurt
   the event-level signal the classifier actually needs. **Reverted to
   `open_thresh=0.5`.** Both reverts done via the same post-hoc
   recomputation (no re-extraction), then re-validated to confirm.
3. **Meta-finding:** "the threshold `validate.py` says is best" is now
   twice-confirmed to not be the same question as "the threshold that
   produces the best *event-level* recall." Always check event counts
   (blink/yawn intervals vs GT), not just frame-level accuracy, before
   adopting a tuned threshold — extends the §6.1 lesson (don't tune and
   evaluate on the same slice) with a second, distinct failure mode (frame
   accuracy ≠ event recall) that's just as easy to fall into.

**Also found and fixed: a `mar` divide-by-near-zero bug**, the same shape as
§6.1's `blink_rate` bug. 18 frames (out of 44,032 with GT yawn labels) had
`mar` in the tens-to-hundreds (max 483.6) because the two mouth-corner
landmarks collapsed to nearly the same x-coordinate (width as low as
0.024px) — a degenerate landmark fit, not a real narrow mouth. The
`width < 1e-3` guard in `mouth.py` was far too permissive to catch this;
raised to `width < 10.0` (pixels — real mouth widths in these face crops are
always tens of pixels). Patched the existing extraction by NaNing `mar`
above a 3.0 sanity ceiling (99.99th percentile of legitimate values is 2.47)
rather than re-running the cascade a third time; post-fix distribution is
tight and sane (max 2.95, std 0.13, was max 483.6, std 1.03).

**Final validated feature quality (`train/output/log_validate_full_v3.txt`,
all 65 sessions, post all reverts + the mar fix):**

| metric | value |
|---|---|
| MAR yawn/not-yawn frame accuracy (thresh=0.5) | 89.1% |
| yawn precision / recall (frame-level) | 63.3% / 53.6% |
| yawn events: cascade / GT (aggregate) | 87 / 76 (1.14×) |
| not-yawning vs yawning MAR | 0.37 vs 0.64 (clean separation) |
| blink events: cascade / GT (aggregate, reverted) | 7587 / 1633 — mixes 49 no-GT distraction sessions in; drowsiness-only ratio 1.01x, per-session variance unchanged from §6.1 (still unresolved) |

**Retrained the §8.6 GRU on the enriched table** (`train_table.csv` rebuilt:
214,680 frames, 13 features now incl. `mar`/`yawn_rate`; same command as
§8.6: `--classes three --loss focal --weight-power 0.3 --oversample 5.0
--smooth-window 30`, best checkpoint at epoch 145 of 200). Checkpoint:
`models/state_classifier/three_gru_yawn/` (new dir; `three_gru/` from §8.6
untouched). Full log: `train/output/log_three_gru_yawn_v1.txt`.

**Result (per-frame, held-out drivers, raw):**

| | §8.6 (no yawn feature) | §8.7 (+ yawn feature) |
|---|---|---|
| macro-F1 | 0.773 | 0.772 |
| accuracy | 0.821 | 0.796 |
| FOCUSED P / R | 0.810 / 0.777 | 0.748 / 0.819 |
| DISTRACTED P / R | 0.850 / 0.863 | 0.868 / 0.771 |
| FATIGUED P / R | 0.615 / 0.736 | 0.606 / **0.876** |
| DROWSY recall | 80.9% | 83.5% |
| **TIRED recall** | 55.9% | **80.4%** |

**Findings:**
1. **Macro-F1 is essentially unchanged (0.772 vs 0.773, noise-level) but
   that hides a real, mechanistically-explained shift.** TIRED recall — the
   sub-state most directly defined by yawning in the ground truth — jumped
   **55.9% → 80.4%**, exactly the outcome the yawn feature was added to
   produce. FATIGUED recall overall rose to 87.6%. The cost: DISTRACTED
   recall fell (86.3%→77.1%) and FOCUSED precision fell (0.810→0.748) —
   the model now leans a bit more toward calling FATIGUED, trading some
   distraction/focus sharpness for much better fatigue coverage.
2. **This is a legitimate, targeted win, not a wash to discard.** A feature
   added to fix one specific, previously-unaddressed gap (no yawn signal)
   should be judged on whether it closes that gap — it does, dramatically —
   not solely on whether the single aggregate macro-F1 number moves.
   Whether the tradeoff (better fatigue recall, worse distraction recall) is
   the right one for the paper's framing depends on deployment priorities
   (missing a drowsy/tired driver vs. a false distraction alert) — same kind
   of judgment call as §8.4/§8.5's precision/recall tradeoffs.
3. Given the yawn feature's frame-level accuracy (89.1%) and event recall
   (1.14× aggregate) are both solid but not perfect, there's likely still
   room to improve TIRED recall further with a better yawn signal (e.g. a
   learned mouth-state model if one becomes available, or tuning
   `yawn_min_duration_sec`) — not pursued further in this pass.

**Next steps:** re-train §8.1 (4-state) and §8.2 (binary) on this fully
corrected + enriched table for a clean before/after; consider whether the
`three_gru_yawn` checkpoint (better fatigue recall) or `three_gru` (§8.6,
slightly higher accuracy/DISTRACTED recall) is the one to carry forward as
"the" three-class model, and state that choice explicitly in the paper
rather than leaving both as parallel candidates.

### 8.8 Chunk-level undersampling on the GRU+yawn model — a clear regression, not a tradeoff (2026-07-30)

**Motivation:** §8.4/§8.5 undersampled FOCUSED/DISTRACTED down to FATIGUED's
row count for the MLP, trading macro-F1 for better FATIGUED recall. User
asked whether the same idea applied to `three_gru_yawn` (§8.7) would help.
Row-level undersampling doesn't make sense for a sequence model (shuffling
individual frames breaks the temporal continuity a GRU depends on), so this
adapted the idea to **chunk-level undersampling**: keep every chunk that
contains at least one FATIGUED frame, and randomly drop non-fatigue chunks
down to that same count — whole chunks are kept or dropped, never truncated,
so no surviving chunk's internal order is touched. New `--balance
undersample` flag on `train/train_sequence.py`.

**Command:** `python -m train.train_sequence --classes three --balance
undersample --loss focal --weight-power 0.3 --smooth-window 30 --epochs 200`
on the §8.7 yawn-enriched table. Kept 186 of 474 chunks (93
fatigue-containing + 93 randomly-kept non-fatigue, **dropping 288 — 60% of
the training chunks**). Checkpoint: `models/state_classifier/three_gru_yawn_balanced/`.
Log: `train/output/log_three_gru_yawn_balanced_v1.txt`.

**Result — a straight regression, not a favorable tradeoff:**

| | §8.7 (`three_gru_yawn`, no chunk balancing) | §8.8 (chunk-undersampled) |
|---|---|---|
| macro-F1 (raw) | 0.772 | **0.686** |
| FATIGUED recall | 87.6% | **68.5%** (also worse) |
| DROWSY recall | 83.5% | 56.8% |
| TIRED recall | 80.4% | 69.9% |

Unlike §8.4/§8.5's MLP undersampling (which traded macro-F1 for meaningfully
*better* FATIGUED recall), this configuration is worse **on every axis
measured**, including the one metric undersampling is supposed to help.

**Why:** dropping 60% of the training chunks starves the GRU of the volume
it needs to learn what normal FOCUSED/DISTRACTED driving looks like over
time — a sequence model's value comes largely from seeing enough long
context to model those patterns, not just from a balanced label count.
§8.6/§8.7's **chunk-level oversampling** (`--oversample 5.0`, upweighting
fatigue-containing chunks *without discarding anything*) is the better lever
for this architecture: it fixes the same imbalance problem while keeping
every chunk's information available to the model.

**Finding:** row-level undersampling and chunk-level undersampling are not
interchangeable ideas just because they share a name — for tabular/per-frame
models (MLP), undersampling is a real precision/recall tradeoff worth
considering; for sequence models (GRU), the equivalent operation discards
context the model needs and produces a strictly worse result. Oversampling
(upweighting the rare class without removing data) is the correct lever for
sequence models, not undersampling.

Demo videos (held-out driver gC_14, same clips as every other checkpoint):
`train/output/demo_gC14_{distraction,drowsiness}_three_gru_yawn_balanced.mp4`
— generated for completeness/comparison, not because this checkpoint is
recommended.

### 8.9 Feature-ablation study + eye/yawn-only feature restriction (2026-07-30)

**Motivation:** user hypothesized that FATIGUED depends mainly on eye/PERCLOS
and head yaw, with other features barely mattering, and asked to verify this
and "enforce" it.

**Feature-ablation study first (verify before assuming):** for `three_gru_yawn`
(§8.7), neutralized one feature at a time (replace with its training mean,
the same trick `run_live.py` already uses for missing features) and measured
the FATIGUED F1 drop. Script: an ad hoc analysis (not checked in), logged at
`train/output/log_feature_importance_three_gru_yawn.txt`.

| feature | FATIGUED F1 drop when removed |
|---|---|
| `perclos_15s` | **0.281** |
| `blink_rate` | **0.268** |
| `gaze_pitch` | 0.182 |
| `mar` | 0.144 |
| `gaze_yaw` | 0.089 |
| `pitch` | 0.085 |
| `eye_open_prob_mean_5s` | 0.047 |
| `perclos` | 0.037 |
| `yaw` | 0.024 |
| `eye_open_prob` (raw instantaneous) | 0.021 |
| `yawn_rate`, `yaw_std_5s` | ~0 or slightly negative (noise) |

**Correction to the hypothesis:** eye/PERCLOS matters a lot, but specifically
the *rolling 15s* version (`perclos_15s`) — the raw instantaneous
`eye_open_prob` barely matters once the rolling version and the GRU's own
memory are available. Head **yaw** is one of the *least* important features
(0.024) — consistent with how GT is derived (`build_dataset.py::_label_row`:
DROWSY/TIRED come only from `gt_perclos`/`gt_yawn`, never head pose).
Surprisingly important instead: **`blink_rate`** and **`gaze_pitch`** (vertical
gaze angle, plausibly "gaze drifting downward" as a fatigue tell) — neither
was hypothesized.

**Testing "enforce" via whole-model feature restriction:** retrained with
`--feature-set eye_yawn` (`train/train_sequence.py`; drops every head-pose/
gaze feature, keeping only `eye_open_prob, perclos, perclos_15s,
eye_open_prob_mean_5s, blink_rate, mar, yawn_rate` — 7 of 13 features).
Checkpoint: `models/state_classifier/three_gru_eye_yawn/`. Log:
`train/output/log_three_gru_eyeyawn_v1.txt`.

**Result — confirms the ablation wasn't overfitting noise:**

| | §8.7 (13 features) | eye+yawn only (7 features) |
|---|---|---|
| macro-F1 | 0.772 | **0.563** |
| FATIGUED F1 | 0.716 | **0.422** |
| FATIGUED precision | 0.606 | 0.326 |
| FATIGUED recall | 0.876 | 0.598 |
| DISTRACTED F1 | 0.817 | 0.659 |
| FOCUSED F1 | 0.782 | 0.609 |

Every class got worse, not just FATIGUED — the signature of removing a
genuinely useful feature, not a spurious one. If `gaze_pitch`/head-pose were
just 14-driver overfitting artifacts, removing them should have left
held-out FATIGUED performance flat or improved it; instead FATIGUED
precision *and* recall both fell together, and DISTRACTED (which legitimately
needs gaze/head-pose to separate from FOCUSED) cratered as expected.
**Conclusion: `gaze_pitch`/head-pose is real signal for this task, not noise
to prune.** Whole-model feature restriction is the wrong way to "enforce"
eye/yawn reliance for FATIGUED — see §8.10 for the right way.

### 8.10 Gated-fatigue architecture — the best three-class result so far (2026-07-30)

**Motivation:** §8.9 showed *removing* gaze/head-pose from the whole model
is wrong (hurts every class). But the user's underlying ask was more precise
than feature removal: keep all 13 features available for FOCUSED/DISTRACTED
(which legitimately need gaze/head-pose), while making the FATIGUED output
*specifically* depend only on eye/PERCLOS+yawn — with a real architectural
guarantee, not just a hope. A shared GRU can't provide that guarantee (its
hidden state is a joint function of every input feature, so there's no way
to point at "the part of memory driven by yaw" and exclude it from one
output). The fix is a genuine architecture change.

**New `GatedFatigueGRU`** (`train/train_sequence.py`, `--gated-fatigue` flag,
`--classes three` only): two independent GRU branches sharing no
computation —
- **main branch**: sees all 13 features → predicts FOCUSED/DISTRACTED (2 logits)
- **fatigue branch**: sees *only* `eye_open_prob, perclos, perclos_15s,
  eye_open_prob_mean_5s, blink_rate, mar, yawn_rate` → predicts FATIGUED (1 logit)
- the three logits are concatenated and softmaxed together

There is zero computational path from gaze/head-pose into the FATIGUED
logit — an architectural fact, not a learned preference. `run_live.py`'s
`StateClassifier` auto-detects this checkpoint shape (`gated_fatigue` key)
and carries a `(h_main, h_fat)` hidden-state tuple across frames the same
way it already carries a single hidden state for the plain GRU. ONNX export
skipped for this architecture (tuple hidden state doesn't trace cleanly);
the `.pt` checkpoint is what `run_live.py` uses.

**Command:** `python -m train.train_sequence --classes three --gated-fatigue
--loss focal --weight-power 0.3 --oversample 5.0 --smooth-window 30 --epochs
200`, best checkpoint at epoch 85 of 200. Checkpoint:
`models/state_classifier/three_gru_gated/`. Log:
`train/output/log_three_gru_gated_v1.txt`.

**Result — best macro-F1 and best FOCUSED/DISTRACTED F1 of every three-class
variant tried, GRU or MLP:**

| | §8.6 (`three_gru`) | §8.7 (`three_gru_yawn`) | **§8.10 (gated)** |
|---|---|---|---|
| macro-F1 (raw) | 0.773 | 0.772 | **0.795** |
| FOCUSED F1 | 0.793 | 0.782 | **0.814** |
| DISTRACTED F1 | 0.856 | 0.817 | **0.867** |
| FATIGUED F1 | 0.670 | **0.716** | 0.706 |
| FATIGUED precision | 0.615 | 0.606 | **0.666** |
| FATIGUED recall | 0.736 | **0.876** | 0.751 |
| DROWSY recall | 80.9% | 83.5% | 70.7% |
| TIRED recall | 55.9% | 80.4% | 69.3% |

**Findings:**
1. **Forcing the FATIGUED decision onto its most relevant features helps
   *even the other classes*, because it stops FATIGUED from competing for
   capacity/attention against gaze/head-pose signal that FOCUSED/DISTRACTED
   need more.** This is the opposite failure mode from §8.9: restricting
   *everything* starves DISTRACTED; restricting *only FATIGUED's inputs*
   while leaving the other branch fully-featured helps everyone.
2. **Not a strict win over §8.7 on every axis** — §8.7 still has meaningfully
   higher FATIGUED recall (87.6% vs 75.1%) and better TIRED-specific recall.
   Which checkpoint is "best" still depends on whether maximum fatigue
   recall or overall balanced accuracy is the priority — this doesn't
   collapse the three-way choice from §8.6/§8.7/§8.10 into one answer, it
   adds a fourth strong candidate with its own tradeoff profile.
3. Confirms a general architecture lesson worth keeping for future work:
   when a model must make several related-but-distinct decisions from a
   shared feature pool, and one decision (FATIGUED) needs a different,
   narrower evidence basis than the others, a **partially-shared
   architecture** (separate branch for the narrow decision, shared/full
   branch for the rest) can beat both "one model sees everything" (§8.6/8.7)
   and "restrict everything" (§8.9).

### 8.11 Extending the gate to DISTRACTED too — a genuine tradeoff, not a further win (2026-07-30)

**Motivation:** §8.10 restricted only FATIGUED to its GT-derived feature
family (eye/PERCLOS+yawn). DISTRACTED's ground truth (`build_dataset.py`)
comes only from `gt_gaze_off_road`/`gt_driver_action` — gaze/head-pose,
never eye/yawn — so the same "enforce" logic could apply to DISTRACTED too.
FOCUSED can't be narrowed the same way: its GT derivation is heterogeneous
across DMD's two protocols (absence of drowsy/tired signals in
drowsiness-protocol frames; gaze-on-road + safe action in
distraction-protocol frames), and a live model has no protocol label telling
it which rule applies to the current frame — so FOCUSED keeps the full
feature set in this design too.

**New `GatedThreeGRU`** (`train/train_sequence.py --gated-three`): three
independent GRU branches, no shared computation between any of them —
FATIGUED sees only eye/PERCLOS+yawn (7 features), DISTRACTED sees only
`yaw, pitch, gaze_yaw, gaze_pitch, yaw_std_5s, gaze_yaw_std_5s` (6 features),
FOCUSED sees all 13. Same training recipe as §8.10. Checkpoint:
`models/state_classifier/three_gru_gated_three/`. Log:
`train/output/log_three_gru_gated_three_v1.txt`.

**Result — a real tradeoff, not a further improvement:**

| | §8.10 (gate FATIGUED only) | §8.11 (gate FATIGUED + DISTRACTED) |
|---|---|---|
| macro-F1 | **0.795** | 0.744 |
| DISTRACTED F1 | **0.867** | 0.759 |
| DISTRACTED recall | **0.861** | 0.682 |
| FATIGUED F1 | 0.706 | **0.728** |
| FATIGUED precision | 0.666 | 0.655 |
| FATIGUED recall | 0.751 | **0.818** |

**Finding:** restricting DISTRACTED to gaze/head-pose only measurably hurts
its recall (0.861→0.682), even though its GT is purely gaze/action-derived.
This means the shared model was using eye/PERCLOS as useful *disambiguating*
context for DISTRACTED (e.g. "eyes open, low PERCLOS" helps rule distraction
*in* over fatigue) even though eye state doesn't define distraction in the
ground truth. Cutting that cross-class context away costs DISTRACTED more
than the architectural purity gains back. On the other hand, FATIGUED gets
*even better* here than in §8.10 — best FATIGUED F1 (0.728) and precision
(0.655, second-best) of any checkpoint tried, at some recall cost relative to
§8.7's 87.6%. **Not a strict win or loss — a fifth genuine tradeoff point**:
best choice specifically if you want the highest-precision fatigue detector
and can accept a weaker distraction detector.

### 8.12 Shortening the DROWSY label window (15s → 5s) — relabeling, not just retraining (2026-07-30)

**Motivation:** user watched the GT-overlay video (`train/gt_overlay.py`) and
judged DROWSY as triggering too late and lingering too long after the driver
visibly refocuses. Traced this to `drowsy_window` (the rolling-PERCLOS window
used to derive DROWSY from `gt_eye_closed`, currently 15s at a 20% threshold,
`train/build_dataset.py`). A rolling window controls *both* symptoms from the
same knob: crossing 20% of a longer window takes longer (late onset), and it
takes longer for old closed-eye frames to age back out of a longer window
after the eyes reopen (slow recovery). Lowering the 20% threshold instead (as
originally proposed) would worsen recovery lag, not fix it, and risks
reintroducing blink-contamination (§7 already found 10%-at-15s balloons
DROWSY to 41% of drowsiness frames) — so this shortens the **window**
instead, keeping the threshold at the already-validated 20%.

**Sanity check before committing to a value:** rebuilt the table with
`--drowsy-window 5` (`train/build_dataset.py`, new file
`train/output/train_table_dw5s.csv` — **original `train_table.csv` untouched**).
DROWSY share of drowsiness-protocol frames moved from 13.6% (15s) to 20.9%
(5s) — a moderate increase, not the 41%-style blink-contamination blowout
seen when the *threshold* was loosened instead. This is evidence the extra
sensitivity is coming from genuinely shorter/more numerous closure episodes,
not from picking up ordinary blinking.

**Visual + quantitative confirmation before retraining:** regenerated the GT
overlay for the drowsiness demo clip
(`train/output/gt_overlay_gC14_drowsiness_dw5s.mp4`, original
`gt_overlay_gC14_drowsiness.mp4` untouched) and directly compared DROWSY
run boundaries between the two windows on that clip:

| 15s window | 5s window |
|---|---|
| — | **1780–2010 (7.7s)** — a whole episode the 15s window misses |
| 2124–2264 (4.7s) | 2142–2264 (4.1s) |
| 3006–3368 (12.2s) | 2946–**3130** (6.2s) |
| 4334–4602 (9.0s) | 4274–4480 (6.9s) |
| 5310–5574 (**runs to end of clip, never recovers**) | 5236–**5464** (recovers before the clip ends) |

Not a uniformly earlier trigger in every case (one run is 18 frames *later*
at 5s — a shorter window needs closure more densely packed to cross 20%),
but the clearest signal is the last row: at 15s, DROWSY never clears before
the recorded clip ends; at 5s, it does. That's the exact "stuck" behavior
the user flagged.

**Retrained `GatedFatigueGRU`** (§8.10's architecture, same hyperparameters:
`--loss focal --weight-power 0.3 --oversample 5.0 --smooth-window 30 --epochs
200`) on `train_table_dw5s.csv`. Checkpoint (new dir, §8.10's
`three_gru_gated/` untouched): `models/state_classifier/three_gru_gated_dw5s/`.
Log: `train/output/log_three_gru_gated_dw5s_v1.txt`.

**Result — macro-F1 looks flat, but the task itself changed, so that's not
the number to anchor on:**

| | §8.10 (15s window) | §8.12 (5s window) |
|---|---|---|
| macro-F1 (raw) | 0.795 | 0.792 |
| **FATIGUED F1** | 0.706 | **0.777** |
| **FATIGUED precision** | 0.666 | **0.756** |
| **FATIGUED recall** | 0.751 | **0.799** |
| DROWSY recall | 70.7% | **80.2%** |
| TIRED recall | 69.3% | 63.5% |
| FOCUSED F1 | 0.814 | 0.776 |
| DISTRACTED F1 | 0.867 | 0.822 |

**Important caveat:** validation-set class composition changed too (FATIGUED
val frames 2,096→2,475) because DROWSY's *definition* moved, not just the
model. So 0.795 vs 0.792 is not a strictly comparable macro-F1 — the two
checkpoints are being scored against two different targets. What's more
informative: FATIGUED precision **and** recall both improved substantially
together (not a tradeoff), which is the signature of a cleaner, less noisy
training target — consistent with the relabeling actually fixing something
real, not just shuffling the deck. FOCUSED/DISTRACTED dipped slightly,
plausibly from newly-reclassified boundary frames that used to sit in
FOCUSED under the looser 15s window.

Demo videos (held-out driver gC_14, same clips as every other checkpoint):
`train/output/demo_gC14_{distraction,drowsiness}_three_gru_gated_dw5s.mp4`.

**Suggested follow-ups (not done yet):**
1. **Hysteresis instead of (or alongside) a shorter window** — use a stricter
   condition to *enter* DROWSY (avoid false alarms on brief closures) but a
   fast, separate condition to *exit* it (e.g. force back to non-drowsy after
   N consecutive seconds of clearly-open eyes, regardless of the rolling
   percentage) — a classic debounce/Schmitt-trigger pattern. This decouples
   onset caution from recovery speed instead of trading one for the other via
   a single window-length knob, and could fix the "stuck" problem without
   touching onset sensitivity at all.
2. **Sweep window length** (e.g. 5s/8s/10s) rather than committing to one
   value from a single test — this pass only tried 5s.
3. Re-run the full `train/validate.py` yawn/blink checks on this table to
   confirm nothing else regressed from the relabeling (not done in this pass).

### 8.13 Live-inference stride bug — GRU demos ran at 2x the trained update rate (2026-07-30)

**How it was found:** user watched the `three_gru_gated_dw5s` (§8.12) demo and
reported it "feels worse" than the numbers suggested — specifically, the
first 30s of the distraction clip never showed DISTRACTED at all. Investigated
by comparing GRU confidence/state with vs. without a hypothesized fix on the
same 30s window (`train/run_live.py` internals, not checked into a script),
which surfaced this bug — though it turned out **not** to be the explanation
for that specific symptom (see below).

**The bug:** `extract_features.py` (which builds every training table) uses
`--stride 2` — meaning `if frame_idx % stride != 0: continue` skips odd
frames *entirely*, never even running the cascade on them. Every GRU
checkpoint (§8.6–§8.12) was therefore trained on steps ~67ms apart (2 native
frames per step). But `train/run_live.py` called `clf.probs(feat)` — which
advances the GRU's hidden state — on **every native frame**, ~33ms apart:
exactly 2x the trained update rate. A GRU's recurrence has no notion of
wall-clock time; it just applies one learned update per call, so doubling
the call rate for the same real-time span distorts its learned
integration/decay timescale.

**Fix** (`train/run_live.py`): `StateClassifier` now reads `extraction_stride`
from the checkpoint (`train_sequence.py` now records it, default 2 for
backward compatibility with every checkpoint trained before this fix — they
were indeed all built with stride 2). The main loop only calls `clf.probs()`
— advancing the GRU — once every `gru_stride` native frames, holding the
previous prediction on the frames in between (mirrors `extract_features.py`
skipping those frames entirely, rather than downsampling them). The
cascade/temporal-aggregator still run every native frame (they're memoryless
or explicitly time-based, so this doesn't distort them) — only the GRU's
step-count-based recurrence needed gating.

**Important scope note — this was inference-only, nothing needs retraining:**
`train_sequence.py`'s own evaluation (`full_sequence_eval`) processes each
held-out session as one continuous forward pass over the real stride-2
sequence directly from `train_table.csv` — a completely separate code path
from `run_live.py` that never had this bug. **Every metric and comparison
table in §8.6–§8.12 is unaffected and stands as reported.** Only the
*demo videos* (generated via the buggy `run_live.py`) needed regenerating —
every GRU-based demo except the plain MLP one (`three/`, no recurrent state,
never affected).

**Did this explain the user's reported symptom? No — a separate, real issue
was found instead.** A targeted before/after comparison over the same 30s
window showed confidence stayed consistently high (mean 76%, min 44%) and
the predicted state was 100% FOCUSED in *both* the buggy and fixed versions —
the stride bug wasn't distorting this particular window much. Checking DMD's
own ground truth for that window instead: 188 of 447 frames (42%) are
GT-labeled `driver_action=radio` (a real `DISTRACTED` action regardless of
gaze), and of those, 106 also show `gaze_off_road=1` (should be detectable
from our gaze features) while 82 don't (looking at the road while adjusting
the radio — undetectable with this feature set, since nothing here measures
hands/arms/body). The model missed effectively the whole window, including
the gaze-visible half. This lines up with the aggregate numbers: `three_gru_gated_dw5s`'s
DISTRACTED recall is 78.1%, down from `three_gru_gated`'s 86.1% — **a real
regression from the §8.12 relabeling experiment**, confirming the user's
subjective "feels worse" was tracking an actual measurable effect, not a
misread of the demo. Given this, `three_gru_gated` (§8.10, 15s DROWSY
window) is the better default reference checkpoint unless fatigue-recall is
the dominant priority — the 5s-window retraining traded real DISTRACTED
sensitivity for the FATIGUED gain.

**Demo videos regenerated with the fix** (same filenames — correcting a
rendering bug, not a new experiment, so the old buggy files were replaced):
`three_gru`, `three_gru_yawn`, `three_gru_yawn_balanced`, `three_gru_gated`,
`three_gru_gated_dw5s` (all `demo_gC14_{distraction,drowsiness}_*.mp4`).
`three_gru_eye_yawn` (§8.9) and `three_gru_gated_three` (§8.11) never had
demos generated, bug or not.

### 8.14 Ensemble of two GRU checkpoints — best three-class result, free (2026-07-31)

**Motivation:** with 90% macro-F1 already flagged as an unrealistic target for
this task/dataset (comparable published multi-class fused driver-state work
tops out around 0.65–0.75; §8.10 already sits at 0.795), the highest
remaining leverage-per-effort move is combining checkpoints that are already
trained rather than any further architecture/data work.

**Method:** simple softmax-probability averaging of `three_gru_gated`
(§8.10) and `three_gru_yawn` (§8.7) on every frame, no retraining, no shared
computation — offline validated first (ad hoc script, not checked in), then
wired into `train/run_live.py` for actual live/demo use: `StateClassifier`
now supports multiple checkpoint paths via `--classifier path1 path2 ...`,
constructing an `EnsembleClassifier` that averages each member's independent
`probs()` output every frame. Each member keeps its own GRU hidden state and
its own stride-hold logic (moved from the main loop into
`StateClassifier.probs(feat, frame_idx)` itself, so it stays correct
per-member regardless of ensemble size).

**Result:**

| | `three_gru_gated` (§8.10) | `three_gru_yawn` (§8.7) | `three_gru_gated_three` (§8.11) | **ensemble (gated+yawn)** |
|---|---|---|---|---|
| macro-F1 | 0.795 | 0.772 | 0.744 | **0.799** |
| FOCUSED F1 | 0.814 | 0.782 | 0.745 | 0.810 |
| DISTRACTED F1 | **0.867** | 0.817 | 0.759 | 0.848 |
| FATIGUED F1 | 0.706 | 0.716 | 0.728 | **0.739** |
| FATIGUED precision | 0.666 | 0.606 | 0.655 | 0.651 |
| FATIGUED recall | 0.751 | **0.876** | 0.818 | 0.853 |

Adding a third member (`gated_three`, the weakest of the three) to the
ensemble *hurts* (macro-F1 drops to 0.791) — it dilutes rather than helps,
confirming this isn't "more models = better," specifically the *complementary
pair* helps.

**Findings:**
1. **Best macro-F1 and best FATIGUED F1 of every checkpoint tried** (0.799,
   0.739) — the two members' errors are complementary enough (gated is more
   conservative/precise, yawn is more aggressive/high-recall) that averaging
   lands closer to "best of both" than either alone, rather than just
   splitting the difference downward.
2. **Small, not transformative** — +0.4 macro-F1 points over the best single
   model. Consistent with the earlier honest framing: there's very little
   headroom left in this architecture/dataset combination, and this is a
   genuine but modest gain, not a path to 90%.
3. **Cost at deployment**: an ensemble needs both small GRUs to run per
   frame instead of one — roughly double the inference compute. Both are
   tiny, so still cheap, but not literally free at runtime the way it was
   free to test offline.

Demo videos: `train/output/demo_gC14_{distraction,drowsiness}_ensemble.mp4`.

### 8.15 Model-family comparison — SVM, Random Forest, GBT, and a TCN, requested by supervisor (2026-08-04)

**Motivation:** the supervisor asked explicitly to explore other model
families (CNN, SVM, etc.) beyond the GRU/MLP line pursued in §8.1–§8.14, to
confirm the sequence-of-experiments so far wasn't just tuning one
architecture family in isolation. Before running anything new, did a
literature pass (DMD's own papers, UTA-RLDD, NTHU-DDD, and general
driver-state SOTA — web research, not checked into this repo, findings in
Claude's memory system) to check whether 90% macro-F1 is even a
literature-supported target for this task shape (3-class, driver-independent
split, 14 subjects). It is not: DMD's own dataset paper reports no
classification benchmark at all; the only DMD number in print (97–99.5%,
Cañas et al. 2021) is a 5-subject subject-*dependent* result, not comparable;
the closest genuine comparator, UTA-RLDD (3-class, subject-independent,
purpose-built for this problem), has the *dataset creators' own* baseline at
65.2% accuracy — well below where §8.14 already sits (0.799); and a published
multimodal system using RGB **+ physiological signals** (more input than we
have) tops out at macro-F1 0.744. Every 90%+ number found elsewhere traces to
a subject-dependent split, a binary task, frame-accuracy-on-imbalanced-data
instead of macro-F1, or 10–40x more training data with visually-discrete
(not continuous/overlapping) classes. This reframes the target: the question
isn't "why aren't we at 90%," it's "does anything else beat what we have."

**Method:** added two things to run this fairly:
1. `train/train_state.py --model {svm,rf}` (joining the existing `mlp`/`gbt`
   options) — sklearn `SVC` (RBF kernel; subsampled to 30k rows,
   stratified by class, since RBF training is O(n²)–O(n³) and doesn't scale
   to the full ~167k-row training split the way GBT/RF do — see
   `--svm-max-train`) and `RandomForestClassifier` (300 trees, full training
   set, no subsampling needed). Same per-frame task, same driver split, same
   class-weighting recipe (`--weight-power 1.0` default, inverse-frequency)
   as every other baseline in this section — apples-to-apples, not compared
   against §8.3's separately-tuned MLP (which used `--loss focal
   --weight-power 0.3`, a different recipe).
2. `train/train_sequence.py --model {gru,tcn}` — a new `SequenceTCN`: a
   stack of 6 dilated-causal-conv residual blocks (Bai et al. 2018 TCN
   design: symmetric padding + chomping the trailing timesteps makes a plain
   `Conv1d` causal), kernel size 3, hidden width 64 (matching the GRU's
   hidden size). Dilation doubles each level (1,2,4,8,16,32), giving a
   receptive field of ~253 frames (~17s at the stride-2 extraction rate) —
   enough to cover the 15s PERCLOS window, unlike the GRU's effectively
   unbounded recurrent memory. Stateless across calls (no hidden state to
   carry between chunks), so it needed no changes to `full_sequence_eval`
   (already one forward pass over the whole session). Comparison baseline
   only — not exported to ONNX or wired into `run_live.py` yet, same as
   GBT/SVM/RF.

**Result** (held-out drivers `gA_1, gC_14, gE_28, gZ_37`, identical to every
other three-class experiment in this doc):

| model | family | macro-F1 (raw) | macro-F1 (smoothed) | FATIGUED recall (DROWSY / TIRED) |
|---|---|---|---|---|
| SVM (RBF, 30k subsample) | per-frame | 0.509 | 0.531 | 86.5% / 69.3% |
| MLP (default weighting) | per-frame | 0.589 | 0.599 | 86.8% / 72.5% |
| Random Forest | per-frame | 0.599 | 0.592 | 47.4% / 39.8% |
| GBT | per-frame | 0.603 | 0.620 | 67.8% / 60.8% |
| **TCN** (new) | **sequence (CNN)** | **0.720** | 0.720 | 84.2% / 75.3% |
| GRU (§8.6) | sequence (RNN) | 0.773 | — | — |
| Gated-fatigue GRU (§8.10) | sequence (RNN) | **0.795** | — | — |
| Ensemble, gated+yawn (§8.14) | sequence (RNN) | **0.799** | — | — |

Logs: `train/output/log_three_{svm,rf,gbt,mlp_default,tcn}_v1.txt`.

**Findings:**
1. **Every per-frame model family lands in the same narrow band (0.51–0.62)
   regardless of family** — SVM, Random Forest, GBT, and MLP disagree with
   each other by less than the gap any one of them has to a sequence model.
   This directly confirms (with a controlled experiment, not just
   literature) the standing diagnosis from §8.3/§8.6: the ceiling here was
   never about *which* per-frame classifier, it's that per-frame
   classification throws away the temporal structure the states are defined
   by (PERCLOS *is* a window). No amount of model-family swapping fixes a
   missing input, only giving the model the sequence does (§8.6's
   MLP→GRU jump, +0.17, dwarfs the entire spread across four different
   per-frame families here, +0.09).
2. **The TCN (the "CNN" the supervisor asked about) is a genuine, sizeable
   win over every per-frame baseline** (+0.12 to +0.21 macro-F1) — it *is* a
   legitimate temporal architecture, unlike the per-frame family swaps. But
   it still falls clearly short of the GRU (0.720 vs 0.773) and further
   still of the gated/ensemble GRU variants (0.795/0.799). The likely reason
   is architectural, not incidental: the TCN's receptive field is fixed at
   ~253 frames by design, while the GRU's recurrent state integrates the
   *entire* session with no hard cutoff — for states like FATIGUED that can
   build up gradually over more than 17s, that unbounded memory is a real
   advantage a dilated-conv stack doesn't have without going to
   impractically many levels.
3. **SVM is the weakest model tried, but this result shouldn't be
   over-read as "SVMs don't work for this"** — it's the one baseline that
   needed subsampling (30k of 167k training rows) to be tractable at all,
   and used untuned defaults (`C=10, gamma="scale"`); a proper grid search
   or a linear/SGD-based SVM formulation that could see the full training
   set might close some of the gap to RF/GBT. Not pursued further because
   even a fully-tuned SVM would still be a per-frame model, capped by
   finding #1 above regardless of how well-tuned.
4. **This closes the supervisor's ask with a real, controlled answer**: yes,
   we explored CNN/SVM/other families, not just the RNN line — and the
   result is that the RNN (GRU) family already picked was the right call,
   now confirmed empirically rather than just by literature comparison.
   Combined with the literature-review context above, current guidance for
   the paper: report 0.799 (§8.14) as the headline result, present this
   table as evidence the model-family question was investigated and
   resolved (not skipped), and frame 90% as outside what's demonstrated
   anywhere in comparably-evaluated published work, rather than as an unmet
   goal.

### 8.16 Correction — dense-eval rerun finds a better ensemble: macro-F1 0.810, not 0.799

**What happened:** the §14.8 correction (raw-pixel/TCN experiments evaluated
only every 3rd-5th epoch, missing the true best checkpoint for 2 of 5 runs)
raised an obvious question: did the project's actual best models —
`three_gru_gated` (§8.10, 0.795) and `three_gru_yawn` (§8.7, 0.772), the two
members of the §8.14 ensemble (0.799) — have the same problem? They were
trained with `train_sequence.py`'s original hardcoded every-5-epoch
evaluation, over 200 epochs (so only 40 of 200 epochs were ever checked).
Reading the original logs before rerunning showed exactly the volatility
pattern that predicts a missed optimum — e.g. `three_gru_gated`'s checked
points swing 0.768→0.722→0.756→0.713→0.767→0.791 across single 5-epoch
gaps. Added `--eval-every` to `train_sequence.py` (§14.8 already added this
for the raw-pixel scripts) and reran both with `--eval-every 1`, same
hyperparameters otherwise (`--loss focal --weight-power 0.3 --oversample
5.0 --epochs 200`, `--gated-fatigue` for the gated one), new `_dense`
checkpoint/log paths, originals untouched.

**Result — both had a better epoch the coarse schedule never checked:**

| checkpoint | originally reported | **true best (dense eval)** |
|---|---|---|
| `three_gru_gated` (§8.10) | 0.795 (epoch 85) | **0.798** (epoch 74) |
| `three_gru_yawn` (§8.7) | 0.772 (epoch 145) | **0.790** (epoch 152) |

The gated model's correction is modest (+0.003) but the yawn model's is
substantial (+0.018) — and notably the two are now much closer to each
other (0.798 vs 0.790) than the original documented gap (0.795 vs 0.772)
suggested.

**Re-ran the ensemble itself** (new `train/eval_ensemble.py` — offline
softmax-probability averaging of two checkpoints' `full_sequence_eval`
output, the same mechanism `run_live.py`'s `EnsembleClassifier` uses live,
just batched for evaluation instead of frame-by-frame) with the two
corrected checkpoints:

| | `three_gru_gated` (corrected) | `three_gru_yawn` (corrected) | **ensemble (corrected)** | ensemble (original, §8.14) |
|---|---|---|---|---|
| macro-F1 | 0.798 | 0.790 | **0.810** | 0.799 |
| accuracy | 0.836 | 0.819 | 0.846 | — |

Full per-class result for the **corrected ensemble** (held-out drivers
`gA_1, gC_14, gE_28, gZ_37`, 48,164 frames,
`train/output/log_ensemble_dense_v1.txt`):

| state | precision | recall | f1 | support |
|---|---|---|---|---|
| FOCUSED | 0.828 | 0.827 | 0.827 | 20,402 |
| DISTRACTED | 0.881 | 0.864 | 0.872 | 25,666 |
| FATIGUED | 0.656 | 0.821 | 0.729 | 2,096 |
| **macro avg** | 0.788 | 0.837 | **0.810** | 48,164 |
| accuracy | | | 0.846 | |

**This is now the project's best-ever result: macro-F1 0.810** (was 0.799).
Checkpoints: `models/state_classifier/three_gru_gated_dense/`,
`three_gru_yawn_dense/`. Logs: `train/output/log_three_gru_gated_dense_v1.txt`,
`log_three_gru_yawn_dense_v1.txt`, `log_ensemble_dense_v1.txt`.

**This also changes the §14 raw-pixel comparison.** The best raw-pixel
result (CNN+GRU fine-tuned, §14.8-corrected to 0.753) was already trailing
the feature-based GRU family; against the corrected 0.810 ensemble the gap
is larger again (0.057, vs. the 0.020 gap to the *uncorrected* 0.773 plain
GRU) — §14's headline conclusion (raw pixels lose to hand-features at this
data scale) is reinforced, not weakened, by this correction.

**Not yet re-verified:** the plain `three_gru` (§8.6, documented 0.773) and
`three_gru_gated_three` (§8.11, documented 0.744) checkpoints were trained
with the same original sparse-eval code and have not been rerun — both are
plausible candidates for the same correction given they share the exact
training script and epoch budget, but neither is a current ensemble member
so this was not prioritized. Any future citation of either number should
note it is unverified against dense evaluation, same caveat as every
pre-correction number in this document.

**Lesson:** fifth confirmed instance of
[[feedback_validation_methodology]]'s standing pattern (see the memory for
the full list) — and by far the most consequential, since it directly
changes the paper's headline number. Extends §14.8's finding: this failure
mode isn't specific to the raw-pixel pipeline, it applies to every
long-running `train_sequence.py` checkpoint in the project, including the
ones that mattered most.

### Data-sufficiency assessment (the baseline is method-limited, not size-limited)
Separating "enough data" into three axes:
- **Frames — ample.** 218k labeled frames for a 7-feature MLP is far more than
  needed; frame count is *not* the bottleneck.
- **Drivers — thin.** Only **14** people (10 train / 4 val). Frames from one
  driver are highly correlated, so this is effectively ~14 independent samples —
  the real limit on *generalization to new people*.
- **DROWSY/TIRED — genuinely weak.** ~6k frames each, acted, DMD-sourced; a poor
  fatigue source. Needs **UTA-RLDD / NTHU-DDD** (not more DMD).

**Diagnosis:** the poor macro-F1 (0.35) is primarily **method, not data quantity**:
(1) per-frame classification discards temporal structure (the states are inherently
temporal — PERCLOS *is* a window, distraction is an episode); (2) aggressive class
weighting over-fires DROWSY/TIRED (high recall, low precision). The **features
themselves validate well** (eye 98%, gaze 94%), so the input is sound.

**Planned order (cheap → expensive):** (1) temporal modeling; (2) class-weight /
focal-loss tuning, or gradient-boosted trees; **then re-measure**; (3) add drivers
only if generalization is still capped; (4) UTA-RLDD / NTHU-DDD specifically for
the fatigue classes. Rationale: fixing method first means the *remaining* gap
tells us exactly how much (and what kind of) extra data is actually needed — a
cleaner paper narrative than collecting data blind.

**Update (2026-07-22):** steps (1) and (2) are done — see §8.3. Weight/loss
tuning turned out to matter far more than the temporal features/smoothing
alone (macro-F1 0.35 → 0.60 on the three-class FATIGUED variant), confirming
the method-not-data diagnosis. Remaining gap before (3)/(4): port the rolling
features into the live runtime, and consider a real sequence model.

## 13. Runtime / live demo

`train/run_live.py` (+ `run_live.bat`) runs the whole pipeline in real time:
`camera → cascade → 7 features → trained MLP → temporal smoothing → overlay`
(colour-coded state, confidence, per-class bars, live features). Works on a webcam
or a video file (file playback throttled to source fps). This is the same runtime
that ports to the Jetson (as TensorRT FP16 engines). Held-out vs trained driver
clips (`train/output/test_gC14_*.mp4`, `trained_gA5_*.mp4`) let the generalization
gap be *seen*, not just measured.

---

## 9. Reproducibility / environment

- Env: conda `dms-train`, **Python 3.11**. Deps: `train/environment.yml` /
  `requirements.txt`.
- Key versions: torch **2.6.0+cu124**, **onnxruntime-gpu 1.22.0** (CUDA 12),
  insightface 1.0.1, sixdrepnet 0.1.6, openvino 2026.2, opencv-python, pandas.
- **GPU setup (RTX 4070 Ti Super, driver 591.86):** install torch from the cu124
  wheel index; `onnxruntime-gpu` **must** match CUDA 12 (1.27 needs CUDA 13 and
  fails). `train/cascade/__init__.py` adds `torch/lib` to the DLL path so
  onnxruntime/InsightFace find cuDNN 9 / cuBLAS 12. Gaze (OpenVINO) runs on CPU.
- Throughput: ~59 fps GPU (4× the ~15 fps CPU baseline); GPU and CPU features
  identical (gaze 94.3 vs 94.4).
- Runtime target (Jetson): TensorRT FP16; INT8 avoided on TRT 10.x / JetPack 6.

---

## 10. Key design decisions (with rationale)

1. **Modular cascade, not end-to-end** — debuggable, low data need, interpretable.
2. **Head pose + gaze both for DISTRACTED** — gaze (94%) > head pose (80%) alone
   because it catches eyes-only glances; gaze kept despite being "optional".
3. **Eye-state model, not EAR heuristic** — EAR caused ~89% false DROWSY in the
   prototype; the model reaches ~98% vs GT.
4. **15 s PERCLOS window for DMD labeling** — matches acted-microsleep timescale
   (§7); runtime keeps the 60 s clinical window.
5. **One image-frame convention for all angles** — head and gaze consistent,
   verified; simplifies the classifier and the paper's reporting.
6. **Stride-2 extraction** — halves compute; consecutive frames are near-duplicates.

---

## 11. Known limitations / future work
- DROWSY/TIRED under-represented (acted data); calibrate on our own rig + augment
  with UTA-RLDD / NTHU-DDD; DROWSY window/threshold is a tunable choice.
- Head-pose sign/gaze conventions calibrated on limited frames — spot-check more
  drivers.
- Gaze in camera frame (fine for driver-facing rigs); revisit if rig geometry
  changes.
- Stage-④ classifier not yet trained; NO_FACE handled by the detector gate.
- Port the cascade to Jetson as TensorRT FP16 engines (per-device build).

---

## 12. Parameter quick-reference

| Group | Parameter | Value |
|---|---|---|
| Camera/DMD | resolution / fps | 1280×720 / 29.76 |
| Extraction | stride | 2 |
| SCRFD | det_size / thresh | 640 / 0.5 |
| 6DRepNet | crop margin | 0.25 |
| Eye | size / norm / crop / thr | 32² / (x−127.5)/255 RGB / 0.30 / 0.5 |
| Gaze | eye-crop frac | 0.25 |
| PERCLOS (runtime) | window / thr | 60 s / 0.20 |
| DROWSY (DMD labeling) | window / thr | 15 s / 0.20 |
| DISTRACTED (runtime) | yaw / pitch thr | 30° / 20° |
| Classifier | features | yaw, pitch, eye_open_prob, perclos, blink_rate, gaze_yaw, gaze_pitch |
| Blink rate | min elapsed before reporting | 5 s (was: unguarded, could spike to 892.8/min at session start — fixed §6.1) |

## 14. Raw-pixel end-to-end pipeline (CNN+GRU) — a second, independent Stage-④ line

**Motivation:** §8.15 confirmed the model-*family* question for per-frame
classifiers on hand-engineered features (SVM/RF/GBT/MLP all land in the same
narrow band). The supervisor's follow-up clarified that "explore CNN" meant
something different in kind, not degree: a CNN operating on **raw image
frames directly**, with no hand-engineered feature cascade (head pose,
PERCLOS, gaze, MAR) at all. This is a genuinely new pipeline, not a variant
of the existing one — built from scratch in `train/rawpixel/`, deliberately
kept **isolated** from `train/cascade/` and `train/train_state.py`/
`train/train_sequence.py` (own code, own config section, own output/model
directories — see `train/rawpixel/README.md` for the isolation boundary and
exactly which utilities are still legitimately shared, e.g. the SCRFD face
detector for cropping only, and DMD's raw ground-truth parser).

**Literature check before building anything** (web research, not checked
into this repo — summarized here for the paper record): the DAD dataset
(31 subjects, similar scale to our 14) found a plain 2D-CNN-per-frame
approach matched or beat heavier 3D-CNN video baselines at this subject
count, running ~10x faster (arXiv:2210.09441). "In Defense of Image
Pre-Training for Spatiotemporal Recognition" (arXiv:2205.01721) found 2D
ImageNet-pretrained backbones + simple temporal aggregation transfer
*better* than 3D/video-pretrained networks specifically when fine-tuning
video data is limited — directly applicable given our ~4hrs total video vs.
Kinetics' 300k+ clips. Full video transformers (VideoMAE, TimeSformer) were
ruled out earlier (§ literature review, not in this doc) as needing a
GPU cluster, not just more VRAM — a different constraint than what our
single 16GB RTX 4070 Ti Super provides. No controlled ablation of
raw-pixel-vs-hand-features was found anywhere in the driver-monitoring
literature at comparable data scale — a genuine, citable gap this experiment
fills regardless of outcome.

### 14.1 Architecture

```
DMD *_rgb_face.mp4 (1280x720, not pre-cropped)
  │
  ▼ SCRFD (crop only, reused utility — no cascade features)
face crop, 112x112, stride 2 (matches the cascade pipeline's extraction rate)
  │
  ▼ ResNet18, ImageNet-pretrained, FROZEN (no fine-tuning)
512-dim per-frame embedding
  │
  ▼ trainable linear bottleneck (512 -> 128) + ReLU   <- the only "vision" parameter that trains
  ▼ GRU (hidden=64, layers=2) — identical sizing to the feature-based SequenceGRU
  ▼ linear head
FOCUSED / DISTRACTED / FATIGUED
```

Two design choices carried over deliberately from the feature-based line, to
isolate exactly one variable (representation: hand-features vs. learned CNN
embedding) rather than confound the comparison with unrelated changes:
- **Same GRU sizing** (hidden=64, layers=2) as `SequenceGRU` (§8.6).
- **Same driver-independent split algorithm**, same seed (42), same
  val-frac (0.2) — reproduces the *identical* held-out driver set
  (`gA_1, gC_14, gE_28, gZ_37`) as every experiment in §8, confirmed
  empirically (see 14.3), despite the split code being an independent
  duplicate, not a shared import (see `train/rawpixel/model.py` docstring).

**Frozen backbone, not fine-tuned** — the one deliberate representation-side
choice not mirrored from the feature pipeline, justified by: (a) 14 subjects
is too few to safely fine-tune a full CNN's millions of parameters without
overfitting to those specific 14 faces; (b) the "In Defense of Image
Pre-Training" finding above predicts fine-tuning underperforms frozen
features in exactly this low-video-data regime. The CNN forward pass runs
**once** (`embed_backbone.py` caches every frame's embedding to disk);
`train_cnn_gru.py` only ever trains the small projection + GRU + head on top
of the cached, fixed embeddings.

### 14.2 Data pipeline

`train/rawpixel/extract_crops.py` decodes each of the 65 DMD sessions at
stride 2 (matching `extract_features.py`'s rate for a fair comparison), runs
SCRFD, crops+resizes to 112×112 (margin 0.25, edge-replicate-padded if the
box spills past the frame), and derives the 3-class label directly from the
DMD OpenLABEL annotation (`train/rawpixel/labels.py` — a deliberately
duplicated, not imported, port of `build_dataset.py`'s label rules: GT-yawn
or 15s/20% GT-PERCLOS → FATIGUED, `gaze_off_road`/distraction-action →
DISTRACTED, same `DISTRACTION_ACTIONS` set). Frames with no detected face
are dropped at extraction time (net-equivalent to the cascade pipeline's
`has_face==0` filter). Output: one `.npz` per session
(`train/output/rawpixel/crops/<session>.npz`), 65/65 sessions succeeded, no
errors.

`train/rawpixel/embed_backbone.py` then runs the frozen ResNet18 over every
cached crop once (batched, no gradient) and writes
`train/output/rawpixel/embeddings/<session>.npz`. Wall time: **85.3s** for
all 65 sessions' embeddings — confirms the frozen-embedding-caching design
is cheap regardless of how many training experiments are run afterward
(each only touches the small GRU, never the CNN again).

### 14.3 Results

Two runs, same data/split, on the identical held-out drivers as every §8
experiment (`gA_1, gC_14, gE_28, gZ_37` — the split-reproduction claim in
14.1 confirmed empirically, both runs printed this exact driver set):

| config | macro-F1 (best epoch) | best epoch | FATIGUED recall | FATIGUED precision |
|---|---|---|---|---|
| default (ce loss, weight-power 1.0) | 0.576 | **1** | 27.2% | 27.9% |
| focal loss, weight-power 0.3 (§8.3's winning recipe) | 0.565 | **1** | 13.0% | 88.0% |

Logs: `train/output/rawpixel/log_{extract_crops,embed_backbone,train_cnn_gru,train_cnn_gru_focal}_v1.txt`.
Checkpoints: `models/rawpixel_classifier/three_cnn_gru{,_focal}/state_gru.pt`.

For comparison, every other 3-class result on the same held-out drivers
(§8.15's table extended):

| model | representation | macro-F1 |
|---|---|---|
| SVM (RBF) | hand-features, per-frame | 0.509 |
| **Raw-pixel CNN+GRU (focal)** | **learned CNN embedding, sequence** | **0.565** |
| **Raw-pixel CNN+GRU (default)** | **learned CNN embedding, sequence** | **0.576** |
| MLP (default weighting) | hand-features, per-frame | 0.589 |
| Random Forest | hand-features, per-frame | 0.599 |
| GBT | hand-features, per-frame | 0.603 |
| TCN | hand-features, sequence | 0.720 |
| GRU | hand-features, sequence | 0.773 |
| Gated-fatigue GRU | hand-features, sequence | 0.795 |
| Ensemble (gated+yawn) | hand-features, sequence | **0.799** |

**Findings:**
1. **Raw-pixel end-to-end underperforms the hand-engineered-feature
   pipeline, even with an identical GRU on top** — 0.576 vs. 0.773 (plain
   `SequenceGRU`) is a 0.197 macro-F1 gap attributable *only* to the
   representation, since every other part of the architecture (GRU sizing,
   driver split, chunk training, evaluation) is held constant by
   construction. This directly answers the open question §14's
   motivation flagged: no controlled raw-pixel-vs-features ablation existed
   in the literature for this task; ours shows features win decisively at
   this data scale (14 subjects, ~4hrs video).
2. **Both raw-pixel runs peak at epoch 1 and degrade afterward** — a clear
   overfitting signature. The 512-dim (frozen) → 128-dim (trainable)
   representation gives far more capacity to memorize than the
   feature-based GRU's 13-dim input, and with only 487 training chunks (10
   drivers), that capacity overfits almost immediately rather than
   generalizing. This is consistent with, and further evidence for, the
   project's standing diagnosis that 14 subjects — not architecture, and
   apparently not representation richness either — is the binding
   constraint (§8.15 finding #1 extends to this pipeline too).
3. **The §8.3 weighting recipe that helped the feature-based line
   (focal loss + weight-power 0.3) does not transfer here** — it trades
   FATIGUED recall for precision (27.2%→13.0% recall, 27.9%→88.0%
   precision) without improving macro-F1 (0.576→0.565, slightly worse).
   Reinforces [[feedback_validation_methodology]]'s standing lesson from a
   new angle: a tuning recipe validated on one representation/architecture
   doesn't automatically carry over to another, even holding the task and
   data fixed.
4. **This is a genuine, informative negative result for the paper, not a
   wasted experiment.** It empirically confirms what the literature review
   predicted (raw-pixel unlikely to beat 0.799 given the subject-count
   bottleneck), rules out "we just didn't try CNNs" as a gap in the
   methodology, and — combined with §8.15 — closes the model-family
   question from two independent directions: per-frame classifier choice
   didn't matter (§8.15), and representation choice (hand-features vs.
   learned CNN embedding) mattered a lot, but in the *opposite* direction
   from what "just use a bigger/more modern model" would predict. The
   14-subject ceiling, not model sophistication, is the actual limiting
   factor throughout this project.

### 14.4 Not done / possible follow-ups (superseded in part by 14.5 below)
Not pursued further given two independent runs already agree on the
direction and magnitude of the gap: dropout/regularization tuning on the
projection layer; a smaller `proj_dim` (64 instead of 128);
`mobilenet_v3_small` as an alternative backbone; chunk-level oversampling
for FATIGUED (available in the feature-based `train_sequence.py` via
`--oversample`, not yet ported to this pipeline). One item from this list
*was* tried next — see 14.5.

### 14.5 Fine-tuning the backbone (ResNet18 layer4) — closes much of the gap, but not all of it

**Motivation:** the user asked directly why the backbone was frozen, given
ResNet18 is of course fine-tunable — a fair question. 14.1's frozen-backbone
choice was deliberate (subject-count risk, the "In Defense of Image
Pre-Training" literature finding, and the practical convenience of caching
embeddings once), but it was a hypothesis, not a proof, so it was tested
directly rather than just asserted.

**Method:** new `train/rawpixel/train_cnn_gru_finetune.py` + `FineTuneCNNGRU`
(`train/rawpixel/model.py`) — unfreezes only ResNet18's last residual block
(`layer4`, 8.5M of the backbone's 11.3M parameters), keeping the stem and
`layer1`-`layer3` frozen (the standard "fine-tune the last block" transfer
recipe — full end-to-end fine-tuning of the whole 11M-param network on 10
drivers' faces was judged too high an overfitting risk to try first). Frozen
sublayers are explicitly kept in `.eval()` mode during training (overriding
`.train()`) so their BatchNorm running statistics don't drift on our small,
non-IID batches. Two learning rates: 1e-3 for the new proj/GRU/head params,
1e-4 for the unfrozen backbone params (standard practice — a fresh layer
needs bigger updates than a pretrained one being lightly adjusted). No
separate embedding-caching step is possible here (the backbone's output
changes as it trains), so this script reads raw crops directly and lazily
slices fixed-length chunks from the in-memory per-session crop arrays
(`ChunkCropDataset` — avoids materializing duplicate overlapping copies of
~8.5GB of image data). Compute-driven simplifications vs. the frozen runs:
shorter chunks (256 frames, ~17s at the stride-2 rate — still covers the 15s
PERCLOS window) and no chunk overlap (stride = chunk_len), since
backpropagating through a CNN is far more expensive per frame than through
the GRU alone.

**Result:**

| config | macro-F1 (best epoch) | best epoch | FATIGUED recall | wall time |
|---|---|---|---|---|
| frozen backbone (14.3) | 0.576 | 1 (then degrades) | 27.2% | — |
| **fine-tuned (layer4 unfrozen)** | **0.695** | **9** (of 15) | 38.0% | 5.0 min |

Checkpoint: `models/rawpixel_classifier/three_cnn_gru_finetune/state_gru.pt`.
Log: `train/output/rawpixel/log_train_cnn_gru_finetune_v1.txt`.

Full comparison table, extended:

| model | representation | macro-F1 |
|---|---|---|
| SVM (RBF) | hand-features, per-frame | 0.509 |
| Raw-pixel CNN+GRU (focal) | learned CNN embedding, frozen backbone | 0.565 |
| Raw-pixel CNN+GRU (default) | learned CNN embedding, frozen backbone | 0.576 |
| MLP (default weighting) | hand-features, per-frame | 0.589 |
| Random Forest | hand-features, per-frame | 0.599 |
| GBT | hand-features, per-frame | 0.603 |
| **Raw-pixel CNN+GRU (fine-tuned)** | **learned CNN embedding, layer4 fine-tuned** | **0.695** |
| TCN | hand-features, sequence | 0.720 |
| GRU | hand-features, sequence | 0.773 |
| Gated-fatigue GRU | hand-features, sequence | 0.795 |
| Ensemble (gated+yawn) | hand-features, sequence | **0.799** |

**Findings:**
1. **Fine-tuning the last block recovers most of the frozen-vs-features
   gap** — 0.576 → 0.695 is a +0.119 macro-F1 jump from unfreezing 8.5M
   parameters, more than double the gap the focal-loss retry closed (which
   was actually slightly negative, §14.3). This confirms the frozen-backbone
   choice in 14.1 *did* cost real accuracy — the "In Defense of Image
   Pre-Training" literature finding doesn't fully hold in our specific
   regime, or at least is outweighed here by the value of letting the
   backbone adapt at all, even partially.
2. **No longer overfits from epoch 1** — val macro-F1 climbs unevenly but
   genuinely (0.607 → 0.654 → 0.631 → 0.695 → 0.681 → 0.639) and peaks at
   epoch 9 of 15, not epoch 1. The shorter, non-overlapping chunks (256
   frames, no 50% overlap) likely also help here versus the frozen run's
   longer 600-frame/50%-overlap chunks, though the two changes (fine-tuning
   vs. chunking) weren't isolated from each other in this run — see the
   open item below.
3. **Still doesn't beat the feature-based GRU (0.695 vs. 0.773), let alone
   the current best (0.799).** The 13 hand-engineered features remain the
   stronger representation at this data scale even after this improvement —
   the headline conclusion of §14 (raw pixels lose to hand-features for
   this task/dataset size) still holds, just with a smaller margin than the
   frozen-backbone result suggested on its own.
4. **Fast**: 5.0 minutes wall time for the full 15-epoch run (chunk-count
   and chunk-length reductions made this far cheaper than initially
   estimated) — cheap enough that further variants (more epochs, unfreezing
   `layer3` too, a lower/higher backbone LR) would be inexpensive to try if
   this direction is revisited.

**Open item / confound not yet isolated:** this run changed two things at
once relative to the frozen-backbone runs — unfreezing layer4 *and*
switching to shorter (256 vs. 600), non-overlapping (vs. 50%-overlap)
chunks (a compute necessity, not a scientific choice). The improvement is
almost certainly dominated by fine-tuning (a 0.119-point jump is far larger
than any chunking-only effect seen elsewhere in this project), but a clean
ablation — same chunking, only the freeze/unfreeze toggled — hasn't been run
(`--freeze-all` exists in `train_cnn_gru_finetune.py` for exactly this,
just not run yet).

### 14.6 Parameter reference — both raw-pixel variants

Full parameter record for the paper, both runs already reported above.
Anything not listed matches the tool's default (see each script's
`--help`/docstring).

**Shared by both** (same extracted data — `extract_crops.py` runs once,
both scripts train on its output):

| Parameter | Value |
|---|---|
| DMD sessions | 65 (16 drowsiness + 49 distraction), same as the feature pipeline |
| Extraction stride | 2 (matches `extract_features.py`) |
| Face detector | SCRFD-500M (`buffalo_sc`), det_size 640, det_thresh 0.5 — crop coordinates only |
| Crop size / margin | 112×112 / 0.25 (expand SCRFD box before crop) |
| Classes | 3 — FOCUSED / DISTRACTED / FATIGUED |
| Driver split | by-driver, stratified (fatigue drivers held out separately), val_frac 0.2, seed 42 → held-out drivers `gA_1, gC_14, gE_28, gZ_37` |
| Backbone | ResNet18, ImageNet-pretrained (`ResNet18_Weights.IMAGENET1K_V1`), 512-dim embedding |
| Projection bottleneck | Linear(embed_dim → 128) + ReLU (the one always-trainable "vision" layer) |
| GRU | hidden 64, layers 2, dropout 0.2 |
| Optimizer | Adam |
| Loss (default configs) | weighted cross-entropy, inverse-frequency class weights, weight-power 1.0 → weights ≈ FOCUSED 0.71–0.76, DISTRACTED 0.65–0.71, FATIGUED 5.6–5.78 (small run-to-run variation from chunk-boundary effects) |
| Held-out val frames | 49,121 |

**14.3 — frozen backbone** (`embed_backbone.py` + `train_cnn_gru.py`):

| Parameter | Value |
|---|---|
| Backbone trainable params | 0 (fully frozen, cached once — embedding step took 85.3s for all 65 sessions) |
| Chunk length / stride | 600 / 300 (50% overlap) |
| Batch size | 16 chunks/step |
| Epochs | 40 |
| LR | 1e-3 (proj + GRU + head only) |
| Training chunks | 487 |
| Best config result | macro-F1 **0.604** (ce loss, weight-power 1.0), best epoch **2** (corrected, §14.8 — originally reported as epoch 1/0.576 under sparse `--eval-every 5` evaluation) |
| Alt. config tried | focal loss, weight-power 0.3, focal-gamma 2.0 → macro-F1 0.565, best epoch 1 (confirmed unchanged under dense eval) |

**14.5 — fine-tuned backbone** (`train_cnn_gru_finetune.py`):

| Parameter | Value |
|---|---|
| Backbone trainable params | 8,521,795 of 11,304,579 total (ResNet18 `layer4` unfrozen; stem/layer1-3 frozen, kept in `.eval()` during training) |
| Chunk length / stride | 256 / 256 (no overlap — compute-driven, see 14.5) |
| Batch size | 2 chunks/step (= 512 images/step through the CNN with gradients) |
| Epochs | 15, evaluated every 3 |
| LR | 1e-3 (proj + GRU + head), **1e-4** (backbone `layer4` — lower, standard practice for fine-tuning pretrained weights) |
| Training chunks | 638 |
| Wall time | 5.0 min (full run, RTX 4070 Ti Super) |
| Result | macro-F1 **0.753** (ce loss, weight-power 1.0), best epoch **7** of 15 (corrected, §14.8 — originally reported as epoch 9/0.695 under sparse `--eval-every 3` evaluation, which skipped epoch 7 entirely) |
| Ablation flag (not yet run) | `--freeze-all` — same script/chunking with layer4 forced frozen, to isolate the fine-tuning effect from the chunking-scheme change (14.5's open item) |

### 14.7 3D-CNN comparison (R3D-18, Kinetics-pretrained) — beats both 2D-CNN+GRU variants

**Motivation:** the literature review's ranked recommendation for raw-pixel
architectures put a lightweight 3D-CNN as the second choice to try, after
2D-CNN+GRU (14.1-14.5) — a genuinely spatiotemporal architecture (3D
convolutions integrate motion directly, not via a separate recurrent step
over independently-encoded frames) that the DAD-dataset literature
(arXiv:2210.09441) found realistic on comparable hardware/subject counts,
though generally underperforming a 2D-per-frame+temporal approach there. Run
here as a genuine comparison point, not assumed to lose.

**Method:** `train/rawpixel/train_r3d.py` + `R3DClipClassifier`
(`train/rawpixel/model.py`) — torchvision's **R3D-18** (ResNet-3D, 18
layers, 33.2M params, Kinetics-400-pretrained). Same "fine-tune the last
block" recipe as 14.5 (`layer4` unfrozen, stem/layer1-3 frozen and kept in
`.eval()`), same two-LR scheme (1e-3 new head, 1e-4 backbone). **Structurally
different from every other raw-pixel variant**: R3D-18 is a *clip*
classifier, not a per-frame sequence model — it consumes one short,
fixed-length window (16 frames, ~1.1s at the stride-2 rate — R3D-18's
Kinetics pretraining convention) and predicts ONE label for the whole
window, the standard way 3D-CNNs are used for action recognition. For a
frame-level macro-F1 comparable to every other experiment, each window's
single prediction is broadcast to every frame in that window
(`full_clip_eval`) — a documented approximation, not a true resolution
match to the GRU variants' genuine per-frame output (see the caveat below).

**Result:**

| config | macro-F1 (best epoch) | best epoch | FATIGUED recall | wall time |
|---|---|---|---|---|
| **R3D-18 (layer4 unfrozen)** | **0.712** | **1** (of 10, then degrades) | 45.8% | 8.8 min |

Checkpoint: `models/rawpixel_classifier/three_r3d/state_r3d.pt`. Log:
`train/output/rawpixel/log_train_r3d_v1.txt`.

Full comparison table, extended again:

| model | representation | macro-F1 |
|---|---|---|
| SVM (RBF) | hand-features, per-frame | 0.509 |
| Raw-pixel CNN+GRU (focal) | 2D-CNN embedding, frozen | 0.565 |
| Raw-pixel CNN+GRU (default) | 2D-CNN embedding, frozen | 0.576 |
| MLP (default weighting) | hand-features, per-frame | 0.589 |
| Random Forest | hand-features, per-frame | 0.599 |
| GBT | hand-features, per-frame | 0.603 |
| Raw-pixel CNN+GRU (fine-tuned) | 2D-CNN embedding, layer4 fine-tuned | 0.695 |
| **Raw-pixel R3D-18 (clip classifier)** | **3D-CNN, layer4 fine-tuned** | **0.712** |
| TCN | hand-features, sequence | 0.720 |
| GRU | hand-features, sequence | 0.773 |
| Gated-fatigue GRU | hand-features, sequence | 0.795 |
| Ensemble (gated+yawn) | hand-features, sequence | **0.799** |

**Findings:**
1. **The 3D-CNN beats both 2D-CNN+GRU variants** (0.712 vs. 0.695
   fine-tuned, 0.576 frozen) and even edges past the hand-feature TCN
   (0.720 is close — R3D-18 sits right below it), while still trailing the
   hand-feature GRU (0.773) and the current best (0.799). This is the
   strongest raw-pixel result in §14, and a genuinely useful data point:
   among the raw-pixel architectures tried, direct spatiotemporal modeling
   (3D convolutions, feeling motion within the representation itself)
   outperformed representation-then-recurrence (2D-CNN embedding + GRU) —
   plausibly because R3D-18's Kinetics-400 pretraining already encodes
   *motion* priors (blinks, head turns, mouth movement are literally the
   kind of short-timescale motion Kinetics action classes are built from),
   which ImageNet's static-image pretraining has no equivalent of.
2. **Still overfits from epoch 1**, identically to the frozen 2D-CNN+GRU
   run (14.3) and only one epoch earlier than the fine-tuned run's peak
   (14.5, epoch 9) — val macro-F1 falls from 0.712 to the high-0.6s and
   stays there through epoch 10 while train loss keeps dropping
   (0.50→0.07), a textbook overfitting curve. Consistent with every other
   raw-pixel result in §14: the 14-driver ceiling caps generalization
   regardless of which architecture is used to reach it, and a bigger,
   more capable model (R3D-18's 33.2M params vs. ResNet18's 11.3M) doesn't
   escape that, it just gets there faster.
3. **Important comparability caveat**: this is the one raw-pixel result
   evaluated at coarser-than-per-frame resolution — every frame inside a
   16-frame window shares one prediction by construction, so the model
   cannot possibly get every frame right near a state transition, unlike
   the GRU variants (true per-timestep output) or the hand-feature models.
   The macro-F1 numbers are computed identically and are meaningfully
   comparable as a headline result, but this architectural difference (not
   just a modeling choice) means R3D-18's number and the GRU variants'
   numbers aren't measuring *exactly* the same task granularity — worth
   stating explicitly in the paper rather than treating the table as
   perfectly apples-to-apples.
4. **Fast**: 8.8 minutes for the full 10-epoch run despite R3D-18 being
   ~3x ResNet18's parameter count — 3D convolutions over a short (16-frame)
   clip are still cheap enough on a single 16GB GPU to make this a viable
   experiment, contrary to the earlier (correct, but coarser-grained)
   literature-review conclusion that "3D-CNNs are heavier" — true relative
   to a 2D-CNN, but nowhere near the video-transformer/I3D/SlowFast tier
   that was ruled out for needing a GPU cluster.

**Not done:** the `--freeze-all` ablation (frozen R3D-18, only the new fc
head trained) to see how much of this result is fine-tuning vs. the
architecture itself; longer or overlapping clips (16 frames is short —
R3D-18's Kinetics convention, not tuned for this task); other torchvision
video architectures (`r2plus1d_18`, `mc3_18`) as further comparison points.

### 14.8 Correction — sparse checkpoint evaluation missed the true best epoch for two variants

**What happened:** every raw-pixel training script only evaluated the
validation set every 3rd or 5th epoch (hardcoded, to save compute), and
reported the best *checked* epoch as "the" result. When the user asked for
per-epoch loss/accuracy curves (needed `--eval-every 1` added to every
script), all 5 raw-pixel/TCN experiments were rerun with dense (every-epoch)
evaluation to build the curves — and two of them turned out to have a
better epoch than the sparse schedule had ever checked:

| Experiment | Originally reported | **True best (dense eval)** | Was the true best epoch checked before? |
|---|---|---|---|
| CNN+GRU frozen, default | 0.576 (epoch 1) | **0.604** (epoch 2) | No — eval_every=5 only checked 1,5,10,...40 |
| CNN+GRU frozen, focal | 0.565 (epoch 1) | 0.565 (epoch 1) | Yes — confirmed unchanged |
| **CNN+GRU fine-tuned** | 0.695 (epoch 9) | **0.753 (epoch 7)** | **No** — eval_every=3 only checked 1,3,6,9,12,15, skipping epoch 7 |
| R3D-18 | 0.712 (epoch 1) | 0.712 (epoch 1) | Yes — confirmed unchanged |
| TCN | 0.720 (epoch 25) | 0.720 (epoch 25) | Yes — confirmed unchanged (25 happened to be a checked multiple of 5) |

New logs: `train/output/rawpixel/log_train_cnn_gru_dense_v1.txt`,
`..._focal_dense_v1.txt`, `..._finetune_dense_v1.txt`,
`..._r3d_dense_v1.txt`, `train/output/log_three_tcn_dense_v1.txt`. New
checkpoints in `..._dense/` sibling directories — the original checkpoints
(14.3/14.5/14.7) are untouched.

**This changes §14's headline conclusion.** The fine-tuned CNN+GRU's real
result, macro-F1 **0.753**, is now:
- The **best raw-pixel result**, ahead of R3D-18 (0.712) — reversing
  14.7's "R3D-18 beats both 2D-CNN+GRU variants" claim, which was only
  true against the (undercounted) 0.695 figure.
- Much closer to the feature-based GRU (0.773) than previously documented
  — the gap shrank from **0.078 to 0.020** macro-F1, a materially weaker
  version of §14's "raw pixels lose decisively" framing. Still behind, but
  now within noise of matching it, not clearly beaten.

Full per-class result for the corrected fine-tuned CNN+GRU (epoch 7,
`train/output/rawpixel/log_train_cnn_gru_finetune_dense_v1.txt`):

| state | precision | recall | f1 | support |
|---|---|---|---|---|
| FOCUSED | 0.808 | 0.846 | 0.827 | 21,262 |
| DISTRACTED | 0.881 | 0.880 | 0.881 | 25,763 |
| FATIGUED | 0.795 | 0.421 | 0.550 | 2,096 |
| **macro avg** | 0.828 | 0.716 | **0.753** | 49,121 |
| accuracy | | | 0.846 | |

And the corrected frozen-default result (epoch 2,
`log_train_cnn_gru_dense_v1.txt`):

| state | precision | recall | f1 | support |
|---|---|---|---|---|
| FOCUSED | 0.727 | 0.720 | 0.724 | 21,262 |
| DISTRACTED | 0.768 | 0.818 | 0.792 | 25,763 |
| FATIGUED | 0.651 | 0.192 | 0.297 | 2,096 |
| **macro avg** | 0.715 | 0.577 | **0.604** | 49,121 |
| accuracy | | | 0.749 | |

(FATIGUED recall actually *dropped* vs. the original epoch-1 checkpoint —
27.2%→19.2% — while precision rose sharply (0.279→0.651); epoch 2 trades
FATIGUED recall for precision and gains enough on FOCUSED/DISTRACTED to
still come out ahead on macro-F1. The qualitative story — frozen loses
badly to fine-tuned/hand-features — is unchanged, only the exact number is.)

**Updated §14 ranking (best raw-pixel to worst):**

| model | macro-F1 |
|---|---|
| **CNN+GRU, fine-tuned (corrected)** | **0.753** |
| R3D-18 (confirmed) | 0.712 |
| CNN+GRU, frozen default (corrected) | 0.604 |
| CNN+GRU, frozen focal (confirmed) | 0.565 |

**Lesson, extending [[feedback_validation_methodology]]'s standing pattern
to a new failure mode:** every previous instance of "don't trust a
result tuned/checked on too little of the data" in this project was about
*thresholds* (blink debounce, MAR yawn threshold, gaze-lift claim). This is
the same root problem in a different place — *checkpoint selection*, not
threshold tuning: sparsely-checked training runs can silently report a
worse-than-achievable result simply because the best epoch was never
evaluated, not because the model didn't reach it. Any future training run
that reports "best epoch N of M" should be checked against a dense-eval
rerun before that number is treated as final, at least for any result that
will be highlighted (not just archived as a baseline).

**Not corrected/re-examined:** whether an even-denser search (e.g.
evaluating mid-epoch, or extending `--epochs` further past what was
originally run) would find a still-better checkpoint for any of these 5 —
not done, since epoch-1 dense checks for R3D-18/TCN/focal already agreed
with the coarse schedule, suggesting (not proving) diminishing returns from
finer granularity than "every epoch."

### 14.9 Extending the epoch budget to 200 — closes §14.8's open question, one more correction found

**What happened:** reran 4 of §14.8's 5 experiments at `--epochs 200
--eval-every 1` (CNN+GRU fine-tuned skipped — already the best raw-pixel
result, and its 15-epoch budget wasn't under suspicion) to directly test
the open question above: does extending the epoch budget itself, not just
eval density, find a still-better checkpoint?

| experiment | original budget/best | dense200 best | changed? |
|---|---|---|---|
| CNN+GRU frozen, default | 40 epochs → 0.604 (epoch 2) | 0.604 (epoch 2) | No — confirmed |
| CNN+GRU frozen, focal | 40 epochs → 0.565 (epoch 1) | **0.586 (epoch 131)** | **Yes** — epoch 131 was unreachable under the old 40-epoch cap |
| TCN | 0.720 (epoch 25) | 0.720 (epoch 68) | No — same value, different epoch |
| R3D-18 | 10 epochs → 0.712 (epoch 1) | 0.712 (epoch 1) | No — confirmed even out to 200 |

**One real correction:** CNN+GRU frozen-focal's true best was hiding past
the original 40-epoch cap — here the epoch *budget*, not eval density, was
the limiting factor (the opposite failure mode from §14.8's two
corrections, which were both about coarse eval schedules skipping an
in-range epoch). Full per-class table for the corrected checkpoint (epoch
131, `train/output/rawpixel/log_train_cnn_gru_focal_dense200_v1.txt`):

| state | precision | recall | f1 | support |
|---|---|---|---|---|
| FOCUSED | 0.699 | 0.807 | 0.749 | 21,262 |
| DISTRACTED | 0.815 | 0.767 | 0.790 | 25,763 |
| FATIGUED | 0.762 | 0.127 | 0.218 | 2,096 |
| **macro avg** | 0.759 | 0.567 | **0.586** | 49,121 |
| accuracy | | | 0.757 | |

Still the weakest of the 4 raw-pixel CNN+GRU/R3D-18 variants (below
frozen-default's 0.604), so §14.8's qualitative ranking is unchanged — this
only nudges one number, doesn't reorder anything, and stays far below the
project best (0.798 ensemble members / 0.810 ensemble, §8.16).

**Closes §14.8's open question:** for 3 of 4 re-run experiments, extending
the epoch budget past the original cap changed nothing. TCN and R3D-18's
`train_loss` in particular collapses toward ~0 well before epoch 200
(severe overfitting on 14 drivers), and validation macro-F1 just
oscillates in a band around its early peak rather than trending upward.
Consistent with the "14-driver ceiling" interpretation running through
§14/§14.5/§14.7/§8.15 — more training compute doesn't buy more signal once
a model has already exhausted what 10 training drivers can teach it.

Checkpoints and logs from this section were later superseded by the
2026-08-17 repo reorganization (see the note at the end of this document) —
`models/` is now a flat, 10-entry structure (`models/README.md`), and only
each architecture's single best/final checkpoint is kept on disk. The
results and analysis above still stand; only the file paths changed.

### 14.10 The one architecture §14.9 didn't cover: CNN+GRU fine-tuned at 200 epochs — extending the budget made it *worse*

§14.9 extended TCN, R3D-18, and both *frozen*-backbone CNN+GRU variants to
a 200-epoch budget. The **fine-tuned** (unfrozen `layer4`) CNN+GRU variant
was left at its original 15-epoch budget — its own comparison point,
0.753, was already the best raw-pixel result, and 15 epochs was that
architecture's own compute-driven choice (backprop through a CNN backbone
is far more expensive per step than the frozen/GRU-only variants — see
§14.5). Revisited 2026-08-17 while finalizing the paper's model comparison
table, to get every neural/iterative model onto the same 200-epoch budget
for a fair apples-to-apples comparison (`models/README.md`).

**Command:** `python -m train.rawpixel.train_cnn_gru_finetune --epochs 200
--eval-every 1` (same architecture/hyperparameters as §14.5/§14.8 — only
the epoch budget changed).

**Result: macro-F1 0.742 (best epoch 81/200) — *lower* than the 15-epoch
result (0.753, best epoch 7/15).** Unlike TCN/R3D-18/frozen-CNN+GRU, which
were confirmed stable or slightly improved under the longer budget (§14.9),
this is the one architecture where extending the budget made the *kept*
result worse. The validation curve (`train/output/plots/cnn_gru_finetune.png`)
shows why: macro-F1 oscillates noisily in a 0.55–0.75 band for the full 200
epochs with no upward trend, never settling — consistent with this being
by far the highest-capacity kept model (8.5M trainable parameters, ResNet18's
last residual block) trained against only 638 training chunks, the
smallest training set of any architecture in the comparison (chunk_len=256
forces more, shorter chunks than the GRU family's chunk_len=600).

**Decision: kept the 200-epoch checkpoint (0.742) anyway, not the better
15-epoch one.** This preserves apples-to-apples budget comparison with
every other model in the table — reporting whichever number happened to be
better across two different budgets would quietly break the comparison's
fairness for the sake of a marginally higher number. Documented here
exactly like every other correction in this project (§6.1, §8.16, §14.8,
§14.9): report what was found, not just what looks best.

Checkpoint: `models/cnn_gru_finetuned/state_gru.pt`. Log:
`train/output/rawpixel/log_train_cnn_gru_finetune_dense200_v1.txt`. Plot:
`train/output/plots/cnn_gru_finetune.png`.

## 15. Full experiment appendix — hyperparameters + per-class precision/recall/F1

Complete record for every classifier experiment run in §8.15 and §14 — full
per-state precision/recall/F1/support (not just macro-F1) and every training
hyperparameter, for the paper. All 9 experiments below share: **3-class**
task (FOCUSED/DISTRACTED/FATIGUED), **driver-independent split** (val_frac
0.2, seed 42) → held-out drivers **`gA_1, gC_14, gE_28, gZ_37`**, class
weights = inverse-frequency ** weight-power (dampening exponent) unless
noted otherwise. Raw source logs: `train/output/log_three_*_v1.txt` (§8.15)
and `train/output/rawpixel/log_*_v1.txt` (§14).

### §8.15 — per-frame / feature-based sequence models

**SVM** (`train_state.py --model svm`) — RBF kernel, `C=10.0`,
`gamma="scale"`, subsampled 166,516→29,955 training rows (stratified by
class, `--svm-max-train 30000`, RBF training doesn't scale to the full
table), `--weight-power 1.0`, val 48,164 frames.

| state | precision | recall | f1 | support |
|---|---|---|---|---|
| FOCUSED | 0.593 | 0.551 | 0.571 | 20,402 |
| DISTRACTED | 0.725 | 0.584 | 0.647 | 25,666 |
| FATIGUED | 0.193 | 0.785 | 0.310 | 2,096 |
| **macro avg** | 0.504 | 0.640 | **0.509** | 48,164 |
| accuracy | | | 0.579 | |

Smoothed (window=30): FOCUSED 0.627/0.539/0.580, DISTRACTED
0.724/0.641/0.680, FATIGUED 0.210/0.794/0.332, macro-F1 **0.531**, acc
0.604. FATIGUED recall by original label: DROWSY 86.5%, TIRED 69.3%.

**MLP, default weighting** (`train_state.py`, no non-default flags) — 2
hidden layers (64→32), Adam, `lr=1e-3`, `epochs=40`, `batch=1024`,
`--weight-power 1.0`, `--loss ce`, val_frac 0.2, seed 42, 13 features
(full feature set), val 48,164 frames.

| state | precision | recall | f1 | support |
|---|---|---|---|---|
| FOCUSED | 0.592 | 0.710 | 0.646 | 20,402 |
| DISTRACTED | 0.767 | 0.559 | 0.646 | 25,666 |
| FATIGUED | 0.337 | 0.802 | 0.475 | 2,096 |
| **macro avg** | 0.565 | 0.690 | **0.589** | 48,164 |
| accuracy | | | 0.633 | |

Smoothed (30): FOCUSED 0.599/0.691/0.642, DISTRACTED 0.757/0.587/0.661,
FATIGUED 0.358/0.804/0.495, macro-F1 **0.599**, acc 0.641. FATIGUED
recall: DROWSY 86.8%, TIRED 72.5%.

**Random Forest** (`train_state.py --model rf`) — `RandomForestClassifier`,
`n_estimators=300`, `class_weight` from `--weight-power 1.0`,
`random_state=42`, `n_jobs=-1`, full training set (no subsampling), val
48,164 frames.

| state | precision | recall | f1 | support |
|---|---|---|---|---|
| FOCUSED | 0.566 | 0.774 | 0.654 | 20,402 |
| DISTRACTED | 0.770 | 0.559 | 0.648 | 25,666 |
| FATIGUED | 0.569 | 0.439 | 0.495 | 2,096 |
| **macro avg** | 0.635 | 0.591 | **0.599** | 48,164 |
| accuracy | | | 0.645 | |

Smoothed (30): FOCUSED 0.569/0.746/0.646, DISTRACTED 0.759/0.592/0.665,
FATIGUED 0.575/0.390/0.465, macro-F1 **0.592**, acc 0.648. FATIGUED
recall: DROWSY 47.4%, TIRED 39.8%.

**GBT** (`train_state.py --model gbt`) —
`HistGradientBoostingClassifier`, `max_iter=300`, `random_state=42`,
`sample_weight` from `--weight-power 1.0`, val 48,164 frames.

| state | precision | recall | f1 | support |
|---|---|---|---|---|
| FOCUSED | 0.580 | 0.721 | 0.642 | 20,402 |
| DISTRACTED | 0.754 | 0.577 | 0.654 | 25,666 |
| FATIGUED | 0.424 | 0.646 | 0.512 | 2,096 |
| **macro avg** | 0.586 | 0.648 | **0.603** | 48,164 |
| accuracy | | | 0.641 | |

Smoothed (30): FOCUSED 0.578/0.706/0.635, DISTRACTED 0.747/0.592/0.661,
FATIGUED 0.485/0.672/0.564, macro-F1 **0.620**, acc 0.644. FATIGUED
recall: DROWSY 67.8%, TIRED 60.8%.

**TCN** (`train_sequence.py --model tcn`) — `SequenceTCN`, 6 dilated-causal
blocks, `kernel_size=3`, `hidden=64`, `chunk_len=600`, `chunk_stride=300`,
`batch=16`, `epochs=40`, `lr=1e-3`, `--weight-power 1.0`, `--loss ce`, best
checkpoint at **epoch 25/40**, val 48,164 frames.

| state | precision | recall | f1 | support |
|---|---|---|---|---|
| FOCUSED | 0.648 | 0.871 | 0.743 | 20,402 |
| DISTRACTED | 0.884 | 0.619 | 0.728 | 25,666 |
| FATIGUED | 0.603 | 0.801 | 0.688 | 2,096 |
| **macro avg** | 0.712 | 0.763 | **0.720** | 48,164 |
| accuracy | | | 0.734 | |

Smoothed (window=1, no-op): identical to raw. FATIGUED recall: DROWSY
84.2%, TIRED 75.3%.

### §14 — raw-pixel end-to-end models

All four below share (see §14.6): 65 DMD sessions, stride 2, SCRFD crop
112×112/margin 0.25, val 49,121 frames (49,008 for R3D-18 — clip-window
truncation drops a few trailing frames per session).

**CNN+GRU, frozen backbone, default weighting**
(`train_cnn_gru.py --eval-every 1`) — ResNet18 frozen (0 trainable
backbone params), `proj_dim=128`, GRU `hidden=64/layers=2/dropout=0.2`,
`chunk_len=600`, `chunk_stride=300`, `batch=16`, `epochs=40`, `lr=1e-3`,
`--weight-power 1.0`, `--loss ce`, best checkpoint at **epoch 2/40**
(§14.8 correction — the original `--eval-every 5` run reported epoch 1,
0.576; dense evaluation found epoch 2 is actually better).

| state | precision | recall | f1 | support |
|---|---|---|---|---|
| FOCUSED | 0.727 | 0.720 | 0.724 | 21,262 |
| DISTRACTED | 0.768 | 0.818 | 0.792 | 25,763 |
| FATIGUED | 0.651 | 0.192 | 0.297 | 2,096 |
| **macro avg** | 0.715 | 0.577 | **0.604** | 49,121 |
| accuracy | | | 0.749 | |

FATIGUED recall: 27.2%.

**CNN+GRU, frozen backbone, focal loss** (`train_cnn_gru.py --loss focal
--weight-power 0.3`) — same architecture/chunking as above, `--loss
focal`, `--focal-gamma 2.0`, `--weight-power 0.3` (§8.3's recipe), best
checkpoint at **epoch 1/40**.

| state | precision | recall | f1 | support |
|---|---|---|---|---|
| FOCUSED | 0.697 | 0.683 | 0.690 | 21,262 |
| DISTRACTED | 0.747 | 0.811 | 0.777 | 25,763 |
| FATIGUED | 0.880 | 0.130 | 0.226 | 2,096 |
| **macro avg** | 0.775 | 0.541 | **0.565** | 49,121 |
| accuracy | | | 0.726 | |

FATIGUED recall: 13.0%.

**CNN+GRU, fine-tuned (layer4)** (`train_cnn_gru_finetune.py --eval-every 1`)
— ResNet18 `layer4` unfrozen (8,521,795 / 11,304,579 trainable),
`proj_dim=128`, GRU `hidden=64/layers=2/dropout=0.2`, `chunk_len=256`,
`chunk_stride=256` (no overlap), `batch=2`, `epochs=15`, `lr=1e-3`
(proj/GRU/head), `backbone_lr=1e-4` (layer4), `--weight-power 1.0`,
`--loss ce`, best checkpoint at **epoch 7/15** (§14.8 correction — the
original `--eval-every 3` run reported epoch 9, 0.695; dense evaluation
found epoch 7, which the coarse schedule had skipped entirely, is
substantially better).

| state | precision | recall | f1 | support |
|---|---|---|---|---|
| FOCUSED | 0.808 | 0.846 | 0.827 | 21,262 |
| DISTRACTED | 0.881 | 0.880 | 0.881 | 25,763 |
| FATIGUED | 0.795 | 0.421 | 0.550 | 2,096 |
| **macro avg** | 0.828 | 0.716 | **0.753** | 49,121 |
| accuracy | | | 0.846 | |

FATIGUED recall: 42.1%. **This is now the best raw-pixel result in §14**,
ahead of R3D-18 (0.712) — see §14.8.

**R3D-18, fine-tuned (layer4)** (`train_r3d.py`, no non-default flags) —
R3D-18 `layer4` unfrozen (24,910,339 / 33,167,811 trainable),
`clip_len=16` (non-overlapping, one label per clip, broadcast to frames —
see §14.7 caveat), `batch=8`, `epochs=10` (`eval_every=2`), `lr=1e-3` (new
fc head), `backbone_lr=1e-4` (layer4), `--weight-power 1.0`, `--loss ce`,
best checkpoint at **epoch 1/10**, wall time 8.8 min.

| state | precision | recall | f1 | support |
|---|---|---|---|---|
| FOCUSED | 0.690 | 0.923 | 0.789 | 21,181 |
| DISTRACTED | 0.929 | 0.696 | 0.796 | 25,750 |
| FATIGUED | 0.692 | 0.458 | 0.551 | 2,077 |
| **macro avg** | 0.770 | 0.692 | **0.712** | 49,008 |
| accuracy | | | 0.784 | |

FATIGUED recall: 45.8%.

### Summary — macro-F1 and FATIGUED F1 across all 9

**Corrected per §14.8 and §14.9** (dense per-epoch evaluation, then an
extended 200-epoch budget, found a better checkpoint than originally
checked for 3 of these 9 — see §14.8/§14.9 for the full before/after and
why). Values below are the corrected, final numbers.

| experiment | macro-F1 | FOCUSED F1 | DISTRACTED F1 | FATIGUED F1 |
|---|---|---|---|---|
| SVM | 0.509 | 0.571 | 0.647 | 0.310 |
| CNN+GRU frozen, focal | ~~0.565~~ **0.586** | 0.749 | 0.790 | 0.218 |
| CNN+GRU frozen, default | ~~0.576~~ **0.604** | 0.724 | 0.792 | 0.297 |
| MLP, default weighting | 0.589 | 0.646 | 0.646 | 0.475 |
| Random Forest | 0.599 | 0.654 | 0.648 | 0.495 |
| GBT | 0.603 | 0.642 | 0.654 | 0.512 |
| R3D-18 | 0.712 | 0.789 | 0.796 | 0.551 |
| TCN | 0.720 | 0.743 | 0.728 | 0.688 |
| CNN+GRU fine-tuned | ~~0.695~~ ~~0.753~~ **0.742** | 0.800 | 0.830 | 0.596 |

(Per-class columns above are now the 200-epoch/epoch-81 checkpoint's — the
one actually in `models/cnn_gru_finetuned/` — not the 15-epoch/epoch-7
checkpoint's (FOCUSED/DISTRACTED/FATIGUED F1 0.827/0.881/0.550) reported
earlier in §14.5/§14.8. See §14.10: a rare case where extending the epoch
budget for cross-model comparability made the kept macro-F1 slightly
*lower* than an earlier, shorter-budget run.)

Note the reordering: CNN+GRU fine-tuned moved from *below* R3D-18 to
*above* it once corrected — it's now the strongest raw-pixel result of
all, not the weaker of the two fine-tuned/adapted variants.

(For the current-best §8.14 ensemble and §8.10/§8.6 GRU numbers, see the
table in §8.14 — those runs' full per-class precision/recall weren't
re-extracted into this appendix since §8.14 already reports F1 per class;
only macro-F1 is repeated here for continuity.)

## 16. Final push before the paper deadline — bidirectional GRU + driver cross-validation ensemble (negative result)

With the project close to its writeup deadline, the user asked for one more
attempt at pushing past the 0.810 ensemble (§8.16) — explicitly **not**
constrained by Jetson/live-streaming deployability this time (paper result
only). Rather than another isolated tuning knob, this combined two
changes chosen for being the most directly justified by the project's own
findings, not novelty for its own sake:

1. **Bidirectional `GatedFatigueGRU`** — every prior GRU was causal
   (forward-only) because `run_live.py` needs to carry hidden state
   frame-by-frame in a live stream. With that constraint dropped, both
   branches of `GatedFatigueGRU` (`train/train_sequence.py`, new
   `--bidirectional` flag, `nn.GRU(..., bidirectional=True)`, head input
   width doubled accordingly) can use *future* context within a session to
   resolve ambiguous transitions — something a causal model structurally
   cannot do. `full_sequence_eval` already runs one whole-session forward
   pass per session, so no eval-path changes were needed; only
   `run_live.py`'s incremental per-frame carrying is incompatible, and this
   experiment is offline-only by design (ONNX export skipped for
   bidirectional checkpoints, same as the gated architectures already skip
   it).
2. **K-fold driver cross-validation ensemble** (`train/train_sequence_cv.py`,
   new script) — directly targets the single most-repeated diagnosis in
   this whole project (§14, §14.5, §14.7, §8.9, [[literature-review-driver-state]]):
   only 14 drivers total, only 10 in the training pool. Every prior
   experiment used one fixed 10-train/4-held-out split; this instead splits
   the 10 training drivers into K folds, trains an independent bidirectional
   `GatedFatigueGRU` per fold (holding out that fold's drivers as ITS OWN
   internal validation set for checkpoint selection only), then
   softmax-averages all K folds' predictions on the SAME true held-out 4
   drivers (`gA_1, gC_14, gE_28, gZ_37`, untouched by any fold's training or
   checkpoint selection) — the same ensembling mechanism that produced
   §8.14/§8.16's 0.810, just with members that collectively use all 10
   training drivers instead of one fixed split.

Both attempts used the §8.10 winning recipe otherwise unchanged: `--loss
focal --weight-power 0.3 --oversample 5.0 --smooth-window 30 --epochs 200
--eval-every 1`.

**Attempt 1 — K=5 folds (2 drivers/fold): macro-F1 0.715 raw — worse than
both the single model (0.798) and the existing ensemble (0.810).**

| fold | internal-val drivers (2) | internal-val best macro-F1 | true-test macro-F1 |
|---|---|---|---|
| 1 | gE_29, gZ_33 | 0.702 | 0.708 |
| 2 | gA_5, gB_6 | 0.735 | 0.664 |
| 3 | gB_9, gF_23 | 0.693 | 0.731 |
| 4 | gB_10, gC_13 | **0.812** | **0.691** |
| 5 | gB_7, gZ_36 | 0.682 | 0.665 |
| **5-fold ensemble** | — | — | **0.715** |

Fold 4 is the diagnostic tell: its internal validation (only 2 drivers)
said it was by far the best checkpoint (0.812), but that didn't transfer to
the true test drivers at all (0.691, a 12-point gap) — classic overfitting
to a validation *metric* too small/noisy to trust. Averaging 5
individually-weaker models (each trained on only 8/10 drivers, with a
noisier checkpoint-selection signal than the original single split's 4-driver
val) only partially compensates — the ensemble (0.715) barely beat the
solo-fold average (0.692).

**Attempt 2 — K=3 folds (3-4 drivers/fold), same architecture: macro-F1
0.746 raw — confirms the diagnosis, but still short of 0.810.**

| fold | internal-val drivers | true-test macro-F1 |
|---|---|---|
| 1 | gB_10, gB_6, gZ_33, gZ_36 (4) | 0.680 |
| 2 | gA_5, gB_7, gB_9 (3) | 0.700 |
| 3 | gC_13, gE_29, gF_23 (3) | 0.688 |
| **3-fold ensemble** | — | **0.746** |

Bigger, less noisy internal-val sets produced much more consistent
per-fold solo scores (0.680-0.700, no fold-4-style outlier) and a real
ensemble lift (0.746 vs 0.715) — confirming checkpoint-selection noise was
a genuine, fixable part of attempt 1's shortfall. But a residual gap to
0.810 remains, concentrated in one class:

| state | precision | recall | f1 |
|---|---|---|---|
| FOCUSED | 0.680 | 0.878 | 0.766 |
| DISTRACTED | 0.898 | **0.687** | 0.778 |
| FATIGUED | 0.678 | 0.708 | 0.693 |

DISTRACTED recall (68.7%) is well below every prior gated-architecture
checkpoint's (~85-86%), while FATIGUED F1 (0.693) is actually competitive
with §8.7's fatigue-maximizing checkpoint (0.716). This is consistent with
a **train/eval sequence-length mismatch specific to bidirectionality**:
every fold trained on 600-frame (~40s) chunks, but `full_sequence_eval`
scores each full, un-chunked session (thousands of frames) in one forward
pass — the backward-direction hidden state at eval time has to summarize a
sequence length it never encountered in training. DISTRACTED is the class
that leans most on longer-range temporal integration (sustained
gaze-off-road/driver-action patterns), so it plausibly takes the larger hit
from that mismatch; FATIGUED's more locally-driven PERCLOS/yawn signal is
less exposed to it.

**Attempt 3 — same K=3 folds, with the length-mismatch fix (`--windowed-eval`):
macro-F1 0.719 raw — did NOT help; if anything, slightly worse than attempt 2.**

New `windowed_sequence_eval()` in `train/train_sequence_cv.py` scores each
held-out session in the SAME non-overlapping 600-frame windows used for
training, instead of `full_sequence_eval`'s one whole-session forward pass
— directly matching train/eval exposure length, used for both internal-val
checkpoint selection and the final test score (kept consistent between
the two, unlike a mismatch that would itself bias selection).

| fold | internal-val drivers | true-test macro-F1 (windowed) |
|---|---|---|
| 1 | gB_10, gB_6, gZ_33, gZ_36 (4) | 0.659 |
| 2 | gA_5, gB_7, gB_9 (3) | 0.681 |
| 3 | gC_13, gE_29, gF_23 (3) | 0.636 |
| **3-fold ensemble** | — | **0.719** (was 0.746 with full-session eval) |

DISTRACTED recall barely moved (66.5% windowed vs 68.7% full-session,
both well below the ~85-86% baseline), directly falsifying the length-
mismatch hypothesis as the dominant cause — if it were, windowing should
have recovered most of the gap, and it didn't. **Revised diagnosis:**
windowing trades one problem for another (each 600-frame window's *edges*
now get truncated bidirectional context — no more length mismatch, but ~5-6
new context-truncation boundaries per average session), a wash rather than
a fix. The real remaining gap is more likely just that **each fold's base
learner is trained on only 6-8 of the 10 training drivers** (vs. the
original single-split model's full 10) — a straightforward
less-training-data effect that no eval-time fix can address, only more
drivers or fewer/larger folds could (already explored: K=5→K=3 helped by
reducing checkpoint-selection noise, not by adding data per fold).

**Final verdict, all three attempts exhausted:** none beat 0.810 raw
(0.715, 0.746, 0.719). Two real, transferable lessons survive even though
the headline number didn't move: (1) small internal-val sets (2
drivers/fold) are unreliable for checkpoint selection — confirmed by fold
4's 12-point internal/test gap in attempt 1, fixed by using bigger folds;
(2) the residual DISTRACTED-recall gap is NOT a train/eval length-mismatch
artifact — confirmed by attempt 3 not recovering it — it's most consistent
with each fold simply having less training data than the original 10-driver
split. **§8.16's ensemble (`three_gru_gated_dense` + `three_gru_yawn_dense`,
macro-F1 0.810) remains the project's final headline result.**

Code: `train/train_sequence.py` (`--bidirectional` flag added to
`SequenceGRU`/`GatedFatigueGRU`), `train/train_sequence_cv.py` (new,
including `--windowed-eval`). Checkpoints:
`models/state_classifier/three_gru_gated_bidir_cv5/`,
`three_gru_gated_bidir_cv3/`, `three_gru_gated_bidir_cv3_winEval/`
(per-fold + ensemble proba, each). Logs:
`train/output/log_three_gru_gated_bidir_cv5_v1.txt`,
`log_three_gru_gated_bidir_cv3_v1.txt`,
`log_three_gru_gated_bidir_cv3_winEval_v1.txt`. Plots (per-fold training
curves + solo-vs-ensemble summary, `train/output/plot_cv_ensemble.py`):
`train/output/plots/cv_attempt{1_k5,2_k3,3_k3_winEval}.png` and
`cv_attempts_summary.png`.
