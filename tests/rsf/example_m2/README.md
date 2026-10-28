# Forcing 11 example: m = 2 near the solar Rossby mode

The scripts of a larger forcing-11 scan than the example of `docs/forcing.md`: m = 2, symm −1, a full sphere at N 896
and lmax 33, around the sectoral Rossby mode of m = 2 near ω = −332 nHz. The background is a solar model with a smoothed
entropy gradient, a viscosity of 10¹² cm²/s in the convection zone over 10⁹ cm²/s in the radiative zone, and a
helioseismic rotation profile, in the frame rotating at 456 nHz. The scan covers 91 frequencies over ω = −336 … −329.5
nHz, with 16 realisations of each of six forcings: `rsf_theta` `'nu'` and `'mlt'`, each without a cut, with full forcing
above 0.74 R☉, and with full forcing above 0.76 R☉.

The scripts show how to

- put several forcing configurations side by side in one scan (`mkrhs.py`);
- choose the frequencies: a uniform grid, points around a mode, and one point on each resonance found by an eigenvalue
  solve (`mkfreqs.py`);
- share the frequencies among several workers through a queue, and queue the failed ones again (`mkq.py`, `runq.sh`);
- collect the spectra of all scans by configuration (`anarsf.py`) and plot them (`plotfinal.py`).

| script | what it does |
|---|---|
| `run_example.sh [STEP]` | all steps by default, or one of `build`, `eig`, `rhs`, `grid`, `res`; run it from its own directory |
| `mkrhs.py RUN K` | builds the six forcings side by side in `RUN/B_forced.npz` and writes their column ranges to `RUN/groups.txt` |
| `mkfreqs.py grid OUT1 OUT2` | the uniform grid plus points around the Rossby mode, split into two scans |
| `mkfreqs.py res EIGRUN OUT [GRID...]` | one frequency on each resonance in the window with a damping rate below 2 nHz |
| `mkq.py res FILE [CHUNK]`, `mkq.py nan SCANDIR` | queue chunks of frequencies; `nan` queues again the failed frequencies of a scan |
| `runq.sh NAME [NPS]` | a worker that solves queued chunks; start one per 16 free cores |
| `anarsf.py RESULTS OUT` | collects every `s*/rsf_ke.dat` into `OUT/spectrum.csv` and `OUT/modes.csv` |
| `plotfinal.py spectrum.csv out.png` | the plot of the spectrum |

**Before reusing the scripts:**

- **Tools.** The build step uses `mkgs.py`, a builder of the solar background, and the eigenvalue step `anafs.py`, an
  analysis of the modes. Neither is part of this repository: `TOOLS` in `run_example.sh` points to them, and the
  environment variable `ROT_FILE` to the rotation table that `mkgs.py` reads.
- **Paths.** `KB` (this code's `bin/`), `TOOLS` and `SHIM` at the top of `run_example.sh`, and the lines that activate
  the Python environment and put PETSc's `mpiexec` on `PATH` in `run_example.sh` and `runq.sh`, are those of the
  machine the example ran on: set them for yours. `SHIM` is a directory with a `sitecustomize.py` that sets
  `SLEPc.COMM_WORLD = PETSc.COMM_WORLD`, for versions of slepc4py without `SLEPc.COMM_WORLD`, which the eigensolver of
  `solve.py` uses.
- **MUMPS workspace.** `run_example.sh` sets no `mat_mumps_icntl_14`, and without it many frequencies fail. Add
  `par.petsc_opts['mat_mumps_icntl_14'] = 100` to the `extra=` of its build line; `runq.sh` copies `base/bin`, so its
  workers get it too.
- **Work directory.** The scripts write into their own directory: `base/` (the matrices, a few GB), `eig/`, `s1`–`s4`,
  `q/` and `s_*`. Copy this folder out of the checkout first, and run `mkq.py` from the copy.
- **Concurrency.** With no argument, `run_example.sh` runs the eigenvalue solve with two scans, then two more: up to
  four scans of 16 ranks and 40–90 GB each. On a smaller machine run the steps one at a time.
- **Hard-coded values.** The window, the Rossby mode (ω = −331.932 nHz, σ = −0.264 nHz) and the six configurations are
  set in `mkfreqs.py`, `mkq.py` and `mkrhs.py`.
