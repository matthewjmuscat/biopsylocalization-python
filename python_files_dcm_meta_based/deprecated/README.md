# Deprecated Runtime Modules

These modules were removed from the main biopsy localization runtime while the
project moves toward a stricter patient-runner architecture. They are preserved
for archaeology and possible downstream reimplementation, but new main-facing
code should not import them.

Moved here on 2026-05-19:

- `fanova.py` - deprecated FANOVA/Sobol analysis pathway.
- `csv_writers.py` - deprecated direct containment/dosimetry CSV writers.
- `machina_learning.py` - random forest exploratory analysis, not core runtime.
