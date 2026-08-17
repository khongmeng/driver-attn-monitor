# GRU Ensemble — the recommended model (macro-F1 0.810)

This is **not a separate trained checkpoint** — there's no weight file in this
folder. It's the softmax-probability average of two other checkpoints, run
together every frame:

- `models/gru_gated/state_gru.pt` (Gated-Fatigue GRU, macro-F1 0.798 alone)
- `models/gru_single/state_gru.pt` (single-branch GRU, macro-F1 0.790 alone)

Averaging the two beats either one alone (0.810) because they make
different kinds of mistakes — `gru_gated` is precise/conservative,
`gru_single` is higher-recall/more aggressive on the rare FATIGUED class.
See `docs/METHODOLOGY.md` §8.14/§8.16 for the full result and why this
combination specifically (adding a third, weaker member was tried and made
things worse — not "more models = better").

## Use it

**Offline evaluation** (macro-F1 on the held-out test drivers):
```bash
python -m train.eval_ensemble models/gru_gated/state_gru.pt models/gru_single/state_gru.pt
```

**Live demo** (camera or video file):
```bash
python -m train.run_live --classifier models/gru_gated/state_gru.pt models/gru_single/state_gru.pt
```
This is also `run_live.py`'s **default** — running it with no `--classifier`
flag at all uses this ensemble.
