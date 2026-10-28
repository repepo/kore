# Forcing 11: the random Reynolds-stress forcing

- How to run it and read its output, with a worked example: `docs/forcing.md` (User guide, Convective forcing).
- Where the force comes from, its statistics and how kore builds it: `docs/reynolds-stress.md` (Mathematical
  framework, Reynolds-stress forcing).
- The code: `bin/reynolds_stress.py` (its docstring has the formulas and conventions), the `rsf_*` parameters in
  `bin/params_default.py` (each commented), the frequency scan in `solve.py` (`forced_scan`).

This folder holds the checks of the implementation (`README.md`, `run_all.sh`) and, in `example_m2/`, the scripts of
an m = 2 scan of a solar model: a full sphere at N 896, six forcing configurations in one scan, frequencies queued
over several workers. Its `README.md` says what to edit before reusing them.
