"""Raw-pixel end-to-end driver-state pipeline: face crop -> frozen 2D-CNN
embedding -> GRU, no hand-engineered features. See docs/METHODOLOGY.md §14.

Deliberately independent of ``train/cascade/`` and ``train/train_state.py``/
``train/train_sequence.py`` (the feature-based pipeline) beyond a few shared
*utilities* (face detector, DMD dataset/annotation readers) — see each
module's docstring for exactly what is/isn't reused and why.
"""
