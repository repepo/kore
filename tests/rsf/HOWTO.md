# Forcing 11: how to run the random Reynolds-stress (convective) forcing

Forcing 11 drives kore's linear model with a random stress, the way convection does, and returns the power spectrum of
the kinetic energy of the response. You choose m, symm and a list of frequencies. Each frequency costs one LU
factorisation; each realisation of the random stress is then a cheap back-substitution.

**Symbols and units.**

- ω and σ are the real and imaginary parts of an eigenfrequency in nHz, in the 456 nHz frame.
- w is a frequency in kore units, as the code uses it: w = −ω/456, so positive w is retrograde.
- An eigenvalue λ in `eigenvalues0.dat` has σ = 456 Re λ and ω = −456 Im λ. Its resonance is at w = Im λ.
- Radii are in kore units. In the solar runs the outer wall is r = 1 = 0.985 R☉ (`r_out`), so 0.70 R☉ is r = 0.7107.

## 1. Files

| file | what it does |
|---|---|
| `bin/reynolds_stress.py` | builds the random right-hand sides; the formulas and conventions are in its docstring |
| `bin/assemble.py` | for forcing 11 writes `A.npz`, `B.npz` and `B_forced.npz` (one column per realisation) |
| `bin/solve.py` | the frequency scan (`forced_scan`); writes `rsf_spectrum.dat` and `rsf_ke.dat` |
| `bin/params_default.py` | the `rsf_*` parameters, each documented |
| `tests/rsf/` | the checks; see `tests/rsf/README.md` |
| `tests/rsf/example_m2/` | the scripts of the m = 2 example (section 7) |

The copy made on 2026-10-21 adds the inner cut-off `rsf_rcut`, `rsf_rcut_width` (in `reynolds_stress.py` and
`params_default.py`) and the `sun_cut` test (`tests/rsf/`). An older `reynolds_stress.py` ignores `par.rsf_rcut`
without a message, and `log_asm` does not print it, so check a run's code with `grep -c rsf_rcut bin/reynolds_stress.py`
(0 means the old file).

## 2. Steps

### 2.1 Set up the run

Build the run directory as for any kore run (`parameters.py` and the background profiles), then add to `parameters.py`:

```python
par.forcing = 11
par.forcing_frequency = 0.0      # keep 0: A.npz and B.npz are then also the eigenvalue problem's pencil (2.3)
par.rsf_nsamp = 16               # realisations, one column of B_forced.npz each
par.rsf_seed = 0
par.rsf_theta = 'nu'             # or 'mlt' with par.rsf_lum = 7.0e-7 (the Sun in mkgs.py's units, see 4; needs visc_pow = 0)
par.rsf_nu_floor = 1e-3          # RZ nu over CZ nu: 1e-3 for the Dey viscosity (1e9 / 1e12 cm^2/s)
par.rsf_rcut = 0.671             # no forcing below 0.66 R_sun (1e-8 there) ...
par.rsf_rcut_width = 0.08        # ... full forcing above 0.74 R_sun; this edge needs N of about 900
par.rsf_split = 0.70/0.985       # also report the kinetic energy below 0.70 R_sun
par.petsc_opts['mat_mumps_icntl_14'] = 100   # extra MUMPS workspace for the forced LUs (section 2.7)
```

- Forcing 11 needs ViscosD > 0: the noise is the partner of the viscous force.
- A full sphere (ricb = 0) also needs N even, rpower_u ≥ 5 and rpower_v ≥ 3.
- `rsf_rcut` must lie in [ricb, 1 − `rsf_rcut_width`). In a shell whose ricb is above it, leave `rsf_rcut` at None.
- `reynolds_stress.py` stops with a message if any of these is not met.
- `forcing_amplitude_cmb` is ignored. The power scales with `rsf_scale` instead.
- To convert a cut-off given in R☉: `rsf_rcut = x_full/0.985 − rsf_rcut_width`. For full forcing above 0.74 R☉ with
  width 0.08 this gives 0.74/0.985 − 0.08 = 0.671.
- To use `mkrhs.py` (section 3), build the run with `rsf_theta = 'nu'` and without `rsf_rcut`, as the example did. It
  sets the cut-offs itself and stops otherwise.

### 2.2 Assemble

```bash
python bin/submatrices.py NP
mpiexec -n NP python bin/assemble.py > log_asm
```

- lmax − m + 1 must be a multiple of 2·NP. This is an existing kore rule.
- Check the block "Random Reynolds-stress forcing (forcing = 11)" in `log_asm`. It gives the number of noise centres,
  the l_c range, the maximum of Θ ν_t and the Chebyshev tail of the rows.
- A tail above 1e-6 prints "WARNING: under-resolved, raise N or l_c". Then raise N, or widen the sharpest edge
  (`rsf_rcut_width`, `rsf_taper`). A larger l_c (`rsf_lc_max`, `rsf_alpha`) also helps, but it changes the forcing.

### 2.3 Find the modes (optional, usually wanted)

The eigenvalue problem can reuse the same matrices.

```bash
mkdir eig && cp -r RUN/bin eig/ && ln -s $PWD/RUN/A.npz $PWD/RUN/B.npz eig/
echo "par.forcing = 0" >> eig/bin/parameters.py
echo "par.rtau = 0.0; par.itau = 333/456; par.nev = 80" >> eig/bin/parameters.py
echo "par.petsc_opts['st_mat_mumps_icntl_14'] = 200" >> eig/bin/parameters.py
cd eig && mpiexec -n NP python bin/solve.py > log_solve && python bin/spin_doctor.py NP > log_sd
```

- `eig/` (and each scan directory in 2.5) gets only `bin/` and links to the matrices. If `parameters.py` names a data
  file by a relative path (e.g. `diff_rot_file`), give its absolute path instead, as `mkgs.py` does, or link it in too.
- Set the target as for any eigen solve: `par.itau` is the w of the target and `par.nev` the number of modes. The
  defaults (itau 1.0, nev 5) look elsewhere. The example used ω −333 nHz (itau = 333/456), rtau 0 and nev 80.
- The eigen solve takes only the `eps_` and `st_` options, so the `mat_mumps_icntl_14` of 2.1 does not reach it. Set
  `st_mat_mumps_icntl_14` (200 in the example). Without it a large full sphere can stop with INFOG(1) = −9.
- On chandler, eigenvalue solves of this checkout need the slepc4py shim: `PYTHONPATH=~/lab/rsf11/shim:$PYTHONPATH`.
  chandler's slepc4py has no `SLEPc.COMM_WORLD`, which `solve.py`'s eigensolver uses. The shim is one file,
  `sitecustomize.py`, that sets `SLEPc.COMM_WORLD = PETSc.COMM_WORLD`. Using `PETSc.COMM_WORLD` in `solve.py` itself
  would make it unnecessary. Forced scans do not need the shim.

### 2.4 Choose the frequencies

- Around a mode λ: about 20 points across w = Im λ ± 5 |Re λ|, plus w = Im λ itself.
- Between modes: a uniform grid.
- The m = 2 example used a 0.25 nHz grid over ω −336 … −329.5 nHz, 13 extra points around the Rossby mode, and one point
  on each resonance with |σ| < 2 nHz (51 of them). That makes 91 frequencies.
- `tests/rsf/example_m2/mkfreqs.py grid g1.txt g2.txt` writes the grid, split into two lists. After the eigen solve
  (2.3), `mkfreqs.py res EIGDIR res.txt g1.txt g2.txt` writes the resonances that are not already on the grid to
  `res.txt`, and their modes to `res.txt.modes`. Both write w in kore units. `anarsf.py` reads `res.txt.modes`,
  `g1.txt` and `g2.txt` under these names from `run_example.sh`'s directory.
- It hard-codes the example's window, its Rossby mode and the |σ| < 2 nHz cut, so edit those lines for another case.

### 2.5 Run the scan

```bash
mkdir s1 && cp -r RUN/bin s1/ && ln -s $PWD/RUN/A.npz $PWD/RUN/B.npz $PWD/RUN/B_forced.npz s1/
cp freqs.txt s1/                       # one w per line, kore units
echo "par.rsf_frequencies = np.loadtxt('freqs.txt')" >> s1/bin/parameters.py
cd s1 && mpiexec -n 16 python bin/solve.py > log_scan
```

- Several scan directories can share one set of matrices and run at the same time. Split the list between them.
- Each frequency's row is appended as soon as it is solved. So a scan can be read while it runs, and a stopped scan
  only needs the remaining frequencies.
- Rerun in the same directory only with the same `B_forced.npz`. With another number of columns the rows differ in
  length and `np.loadtxt` (and so `anarsf.py` and `mkq.py`) fails on the file. After `mkrhs.py` (section 3), start new
  scan directories.
- A scan writes no fields.

### 2.6 Read the output

| file | columns |
|---|---|
| `rsf_spectrum.dat` | w, mean KE, its standard error; with `rsf_split` also the mean KE below `rsf_split` and its standard error |
| `rsf_ke.dat` | w, the KE of each realisation; with `rsf_split` then the KE below `rsf_split` of each realisation |

Each scan writes a `#` header line first, which `np.loadtxt` skips. KE is spin_doctor's kinetic energy.

The mean and standard error in `rsf_spectrum.dat` are over every column of `B_forced.npz`. With several configurations
side by side (section 3) they mix them, so use `rsf_ke.dat` with the column ranges in `groups.txt`; `anarsf.py` does this.

- **The mean KE at w is a two-sided power spectral density** in kore units. For m ≠ 0, ∫ (mean KE) dw/2π is the mean
  kinetic energy of the forced flow. For m = 0 the spectrum is even in w and that integral is twice the energy, so
  integrate over w > 0 only.
- **Only the solved symm class.** The noise drives both classes independently, so the energy at one m is the sum of a
  symm = +1 and a symm = −1 run.
- **Noise.** The standard error goes as 1/√`rsf_nsamp`. With 30 realisations it is about 18 % per point; with 16 it was
  13–28 % in the example.
- **Same draws at every w.** Column j always uses `numpy.random.default_rng([rsf_seed, j])`, at every w, in every scan
  and for any `rsf_nsamp`. So the shape of a spectrum is much less noisy than its absolute level.
- **Between configurations** the draws are the same only if m, symm, lmax and the noise centres are the same. The draws
  are laid out by l and by noise centre, and a centre is dropped where the stress is below 1e-7 of its maximum, which
  depends on `rsf_theta` and `rsf_rcut`. Compare the number of centres of each configuration: `log_asm` prints it for
  the assembled one ("noise centres"), and `reynolds_stress.centres()` gives it for any other (`mkrhs.py` prints only
  the tail line). In the m = 2 example, 'mlt' without a cut has 35 centres and the other five configurations 34, so its
  draws are independent of theirs.
- **Near an isolated damped mode** the spectrum is a Lorentzian with half-width |σ|. In the example the fitted
  half-width matched |σ| to 0.1 %. The mode's energy is then the Lorentzian's height × |Re λ|/2 (kore units). Fit a
  Lorentzian plus a background, as the example did, and use its height: the other modes add a background to the raw
  peak. `modes.csv` from `anarsf.py` uses the raw on-resonance PSD.
- **Absolute amplitudes.** `rsf_scale` multiplies Θ. Observed Rossby-mode amplitudes need about
  10 (1e12 cm²/s / ν_t)² × the 'nu' law, or about 1/20 × the 'mlt' law. Both are white-noise estimates. With the
  default l_c and `rsf_angular` = 1 the same modes are forced several times less, so `rsf_scale` goes up as much (see
  the comments of `rsf_theta` in `params_default.py`).

### 2.7 Failed frequencies

- **Symptom.** Every realisation of a frequency is NaN or Inf, and `log_scan` says "Solver crashed at w = …, got
  nan's!". The scan carries on with the next frequency.
- **Cause.** Most likely a MUMPS factorisation that ran out of workspace (INFOG(1) = −9). This is inferred: the scans
  printed no MUMPS code, but every retry with more workspace came back finite. PETSc does not stop on a failed
  factorisation and fills the solution with Inf. It is not the formulation: the same matrix solves fine on another try.
- **In the example** 24 of 91 frequencies failed without `mat_mumps_icntl_14`. All 24 came back finite when rerun with
  `mat_mumps_icntl_14 = 100`.
- **What to do.** Set `mat_mumps_icntl_14` from the start (2.1). Rerun any frequency that still fails, with 200 if
  needed. In the example, `mkq.py nan SCANDIR` (run in the example's directory) queues the failed frequencies and
  `runq.sh NAME` solves the queue. `runq.sh` copies `base/bin`, and `run_example.sh` does not set the option, so add it
  first: append `par.petsc_opts['mat_mumps_icntl_14'] = 100` to `base/bin/parameters.py`, or start each worker with
  `SOLVE="mpiexec -n 16 python bin/solve.py -mat_mumps_icntl_14 100" ./runq.sh NAME`.
- **A permanent fix** is drafted in the project folder as `kore/convective-forcing/solver-failures/proposed-solve.diff`.
  It checks every LU, prints the MUMPS code and, after a workspace error, retries with ICNTL(14) doubled, up to 3 times.
  It starts from `mat_mumps_icntl_14`, else `st_mat_mumps_icntl_14` (200 from mkgs.py), else 50. It was tested only with
  a scipy stand-in for petsc4py, not with real MUMPS. It is for your review and is not applied.

## 3. Several forcings with one LU

A and B do not depend on the forcing, so one scan can solve several forcing configurations: put their right-hand sides
side by side in `B_forced.npz`. Each extra configuration costs back-substitutions, not LUs.

`tests/rsf/example_m2/mkrhs.py RUN K [SEED]` does this for six configurations ('nu' and 'mlt' with `rsf_lum` 7.0e-7,
each with no cut, full above 0.74 R☉ and full above 0.76 R☉, edge width 0.08).

- RUN must be assembled for forcing 11 with `rsf_theta = 'nu'`, no `rsf_rcut`, `rsf_nsamp = K` and `rsf_seed = SEED`
  (default 0). The script stops otherwise.
- It replaces `RUN/B_forced.npz` with the 6K columns, keeps the original as `B_forced_base.npz` and writes `groups.txt`
  with each configuration's column range.
- It prints each configuration's Chebyshev tail: check it as in 2.2.
- `rsf_spectrum.dat` then averages over all six, so read `rsf_ke.dat`. `anarsf.py` splits its columns by `groups.txt`.

## 4. Parameters

| parameter | default | meaning and advice |
|---|---|---|
| `rsf_nsamp` | 1 | realisations; 16–30 is usual |
| `rsf_seed` | 0 | column j uses `default_rng([rsf_seed, j])` |
| `rsf_theta` | 'nu' | Θ law. 'nu': 3 ρ0 ν_t² l_c, grows as ν_t². 'mlt': ρ0 u_c² l_c³/3 with the mixing-length u_c, independent of ν; needs `rsf_lum` and visc_pow = 0. A number: constant Θ (tests) |
| `rsf_scale` | 1.0 | multiplies Θ |
| `rsf_lum` | None | luminosity in kore units, for 'mlt': 7.0e-7 for the Sun (ρ_c, r0 = 0.985 R☉, Ω = 2π × 456 nHz) |
| `rsf_alpha`, `rsf_lc_max` | 2.0, 0.25 | eddy size l_c, a smooth minimum of `rsf_alpha` H_p and `rsf_lc_max` |
| `rsf_lc` | None | a constant l_c instead |
| `rsf_nu_floor` | None | ν_t = ViscosD (ν/ν0 − floor), clipped at 0. None: visc0 for diffusivity 'step' or 'solar_rz' with visc0 < 1, else 0. With the Dey viscosity (mkgs.py `nu=dey`) set 1e-3, or the 1e9 RZ viscosity is forced too; with `nu=mk` the default is right |
| `rsf_angular` | 1 | 1: the variance at degree l times exp(−L l_c²/(2r²)), L = l(l+1) (eddies of finite horizontal size); 0: white in angle |
| `rsf_taper` | 0.04 | the stress goes to zero (erfc) over this width below the outer wall, and above ricb > 0 |
| `rsf_taper_in` | 0.2 | the same above r = 0 in a full sphere |
| `rsf_rcut` | None | no stress below this radius (1e-8 there); full stress above `rsf_rcut + rsf_rcut_width` |
| `rsf_rcut_width` | 0.04 | width of that edge. In a full sphere near r = 0.7, 0.04 needs N of about 1280 and 0.08 about 900. Use the edge, not a higher `rsf_nu_floor`: clipping ν_t leaves a corner the rows cannot resolve |
| `rsf_frequencies` | None | the scan's frequencies, kore units |
| `rsf_split` | None | a radius: also report the KE below it |

## 5. Cost

**m = 2 example on chandler.** symm −1, full sphere, N 896, lmax 33, 96 columns.

- 16 ranks per scan and four scans at once (64 ranks on chandler's 48 cores): 440–740 s per frequency, and 40–90 GB
  per scan.
- With fewer jobs running, a frequency took about 390 s.
- The times are from the scans' per-frequency log lines and the memory from `ps` during the run; neither is in a saved
  file.

**Factorisation versus back-substitution.** A timing test at N 384 (8 ranks) gave 0.3 s for the LU and 0.08 s for each
extra realisation, so there 30 realisations cost about eight LUs. At production N the LU should weigh more (inferred,
not measured).

## 6. Tests

`tests/rsf/run_all.sh [WORKDIR]` builds small runs and runs every check; `tests/rsf/README.md` explains each of them.

- It needs numpy, scipy, mpi4py, and petsc4py and slepc4py with MUMPS. Its header says it takes about half an hour on
  one core.
- `MPIEXEC` replaces `mpiexec`. It must start ranks of the MPI that mpi4py was built with, or the script stops. On cha2
  that is the `mpiexec` in `/home/repepo/petsc/arch-linux-c-opt/bin`.
- It solves only forced problems, so it does not need the chandler shim.
- `tests/rsf/regress_solve.sh OLD_KORE [WORKDIR]` checks that an eigenvalue run and a forcing-10 run are bitwise
  unchanged against OLD_KORE, a checkout from before forcing 11. On chandler its eigenvalue run needs the shim (2.3).
  `run_all.sh` does not run it.

## 7. The m = 2 example (`tests/rsf/example_m2/`)

These are the scripts as they ran on chandler on 2026-10-21. Their paths are cha2's:

- `~/lab/rsf11/kore_rcut/bin`: this code;
- `~/lab/fs_dsdr/tools`: `mkgs.py` (the σ 0.004 full-sphere background) and `anafs.py`. They are not copied here, but
  the project folder has both in `kore/fs-dsdr/tools/`. `anafs.py` needs SciPy ≥ 1.15 (`sph_harm_y`);
- `~/lab/rsf11/shim`: the slepc4py shim;
- `~/z2`: the Python environment;
- `/home/repepo/petsc/arch-linux-c-opt/bin`: the `mpiexec` of mpi4py's MPI;
- `~/lab/dey_hl1/data/diff_rot.npz`: the rotation table, which `mkgs.py` reads through `ROT_FILE` (not set by
  `run_example.sh`).

Before reusing the scripts:

- For another machine, edit `KB`, `TOOLS` and `SHIM` in `run_example.sh`, the `source` and `PATH` lines in both
  `run_example.sh` and `runq.sh`, and set `ROT_FILE`.
- `run_example.sh` sets no `mat_mumps_icntl_14`. Its first pass ran without it, and 24 of 91 frequencies failed. Add
  `par.petsc_opts['mat_mumps_icntl_14'] = 100` to the `extra=` of its build line, as in 2.1.
- The scripts work in their own directory: `run_example.sh` writes `base/` (the matrices, a few GB), `eig/` and
  `s1`–`s4` there, and `runq.sh` writes `q/` and `s_*`. Copy the folder out of the checkout first, and run `mkq.py`
  from the copy.
- `run_example.sh` with no argument runs jobs at once: the eigen solve (8 ranks) with the grid scans s1 and s2, then s3
  and s4 as soon as the eigen solve ends. That can be four scans of `NPS` ranks (16) and 40–90 GB each. On a smaller
  machine run the steps one at a time (`build`, `eig`, `rhs`, `grid`, `res`).

| script | what it does |
|---|---|
| `run_example.sh [build\|eig\|rhs\|grid\|res]` | builds the run (`mkgs.py`), eigen solve, six forcing configurations, two grid scans, then scans on each resonance |
| `mkrhs.py RUN K` | the six forcing configurations side by side (section 3) |
| `mkfreqs.py grid OUT1 OUT2`, `mkfreqs.py res EIGRUN OUT [GRID...]` | the grid and resonance frequency lists (section 2.4) |
| `mkq.py res FILE [CHUNK]`, `mkq.py nan SCANDIR` | queue frequency chunks in `q/todo`; `nan` queues each failed frequency of a scan |
| `runq.sh NAME [NPS]` | a worker that solves queued chunks; start one per 16 free cores |
| `anarsf.py RESULTS OUT` | RESULTS is `run_example.sh`'s directory. Reads every `s*/rsf_ke.dat`, splits the columns by `base/groups.txt` and writes `OUT/spectrum.csv` (per frequency and configuration: mean KE, its standard error, the same below `rsf_split`, and the mean share below). With `res.txt.modes` it also writes `OUT/modes.csv`. It skips failed rows and keeps a frequency solved twice once |
| `plotfinal.py spectrum.csv out.png` | the spectrum plot |

**Results.** `kore/convective-forcing/example/RESULTS.md` in the project folder has the write-up. The data are in cha2
`~/lab/rsf11/example` and on your Mac in `~/lab/rsf_example/results/`.

**Modal sums.** The project folder also has the scripts of the modal check (`mkadj.py`, `modal_v3.py`, `anamodal.py`).
They are not copied: the RZ modes are too non-normal (κ 1e10–1e14) for modal sums, so the solved spectrum is the result.
