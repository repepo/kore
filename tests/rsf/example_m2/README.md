# Forcing 11 example: m = 2 near the Rossby mode (chandler, 2026-10-21)

The scripts of the first forcing-11 example, as they ran on chandler. How to run forcing 11 in general is in
`../HOWTO.md`; this folder is a worked case of it.

**Case.** m = 2, symm −1, full sphere (ricb 0), N 896, lmax 33. Background: the σ 0.004 smoothed entropy gradient
(cut 0.708), Dey ν, JSOC rotation, frame 456 nHz. 91 frequencies over ω −336 … −329.5 nHz, 16 realisations for each of
six forcings ('nu' and 'mlt', each with no cut, full above 0.74 R☉ and full above 0.76 R☉).

**Before reusing the scripts:**

- **Paths.** The scripts use cha2's layout. For another machine, change in `run_example.sh` `KB` (this code's `bin`),
  `TOOLS` (where `mkgs.py` and `anafs.py` are) and `SHIM` (the slepc4py shim); in both `run_example.sh` and `runq.sh`
  the `source ~/z2/bin/activate` line and the PETSc `mpiexec` added to `PATH`. Also set `ROT_FILE` (the rotation table
  `mkgs.py` reads; default `~/lab/dey_hl1/data/diff_rot.npz`).
- **MUMPS workspace.** `run_example.sh` sets no `mat_mumps_icntl_14`. Without it 24 of 91 frequencies failed in the first
  pass. Add `par.petsc_opts['mat_mumps_icntl_14'] = 100` to the `extra=` of its build line (`runq.sh` copies
  `base/bin`, so its workers get it too).
- **Work directory.** The scripts write into their own directory: `base/` (the matrices, a few GB), `eig/`, `s1`–`s4`,
  `q/`, `s_*`. Copy this folder out of the checkout first, and run `mkq.py` from the copy.
- **Concurrency.** With no argument, `run_example.sh` runs the eigen solve with two scans, then two more: up to four
  scans of 16 ranks and 40–90 GB each. On a smaller machine run the steps one at a time.

`mkgs.py` and `anafs.py` belong to the smooth-entropy-gradient work and are not copied here (cha2
`~/lab/fs_dsdr/tools`, and `kore/fs-dsdr/tools/` in the project folder).

| script | what it does |
|---|---|
| `run_example.sh [STEP]` | all steps by default, or one of `build`, `eig`, `rhs`, `grid`, `res`; run it from its own directory |
| `mkrhs.py RUN K` | builds the six forcings side by side in `RUN/B_forced.npz` and writes `RUN/groups.txt` |
| `mkfreqs.py grid OUT1 OUT2` | the uniform grid plus points around the Rossby mode, split into two scans |
| `mkfreqs.py res EIGRUN OUT [GRID...]` | one frequency on each resonance with \|σ\| < 2 nHz in the window |
| `mkq.py res FILE [CHUNK]`, `mkq.py nan SCANDIR` | queue frequency chunks; `nan` requeues the failed frequencies of a scan |
| `runq.sh NAME [NPS]` | a worker that solves queued chunks; start one per 16 free cores |
| `anarsf.py RESULTS OUT` | collects every `s*/rsf_ke.dat` into `OUT/spectrum.csv` and `OUT/modes.csv` |
| `plotfinal.py spectrum.csv out.png` | the spectrum plot |

The window, the Rossby mode (ω −331.932, σ −0.264 nHz) and the configurations are hard-coded in `mkfreqs.py`,
`mkq.py` and `mkrhs.py`.

**Results.** The run's data are in cha2 `~/lab/rsf11/example` and on the Mac in `~/lab/rsf_example/results/`. The
write-up is `kore/convective-forcing/example/RESULTS.md` in the project folder.
