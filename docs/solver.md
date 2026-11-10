# Solver: bad solutions and memory

`solve.py` finds the eigenvalues nearest a target τ = `rtau` + i `itau`. At each iteration it solves a linear system
with the matrix $A - \tau B$, which the MUMPS library factors once (a sparse LU factorization). This factorization uses
almost all the memory, and causes most numerical trouble. This page lists what to try when a solution looks wrong or a
run runs out of memory.

Solver options go in `par.petsc_opts` in `parameters.py`, for example

```python
par.petsc_opts['st_mat_mumps_cntl_1'] = 0.1
```

Eigenvalue runs read the options that start with `eps_` and `st_`, forced runs (`forcing` > 0) those that start with
`ksp_`, `pc_` and `mat_`. So the same MUMPS option is `st_mat_mumps_...` in an eigenvalue run and `mat_mumps_...` in a
forced run. An option given on the command line (`mpiexec -n 4 ./bin/solve.py -st_mat_mumps_icntl_14 100`) overrides
`parameters.py`. The defaults, with comments, are in `bin/params_default.py`.

## First, check the solution

Judge each solution with `spin_doctor.py`, not with the residual that `solve.py` prints (which refers to a rescaled
problem). Accept a mode when:

- its energy residual residσ (and the entropy residual resid𝑠, with `thermal = 1`) is below 1e-5;
- its ℓ-convergence is below 1e-4;
- in a full sphere, irreg₀ is below 1e-6 ([Regularity at the centre](numerical-method.md#regularity-at-the-centre)).

The same checks apply to forced runs. A forced flow that is a rigid rotation has no budget to judge: its residuals
compare round-off errors and can be ignored ([Spherical Couette flow](couette-flow.md#reading-the-output)).

A failure is one of two kinds:

- **Resolution.** It fails every time, and the eigenvalue moves when `N` or `lmax` is raised. Raise `N` for the
  residuals and `lmax` for the ℓ-convergence. A full sphere needs about twice the `N` of a shell. `N` must be even.
- **Solver.** It comes and goes between identical runs, or between runs with nearly the same parameters. See below.

## Choosing N and lmax

The resolution a mode needs depends on the problem: on its boundary and internal shear layers, the background field,
the stratification, and on the mode itself. `par.Ncheb(Ek, rule)` gives a starting `N` from the Ekman number,
N = a Ek<sup>−p</sup>, rounded up to a multiple of 8; each rule comes from a resolution study:

| rule | a, p | from | N at Ek 1e-5, 1e-7, 1e-9 |
| --- | --- | --- | --- |
| `'default'` | 17, 0.2 | the long-standing rule, shells | 170, 428, 1072 |
| `'magnetic'` | 4.55, 0.242 | torsional modes in a shell (Pm 0.1, Elsasser 10, thin conducting mantle): every residual below 1e-4, plus 15 % | 80, 232, 688 |
| `'ekman'` | 10.12, 0.25 | the Ekman layer of a viscous full sphere (spin-over, inertial modes): damping rate to 2e-8 | 184, 576, 1800 |
| `(a, p)` | your own | a fit of your own study | |

```python
par.N = par.Ncheb(par.Ek, 'magnetic')        # or par.Ncheb(par.Ek, (4.0, 0.25))
```

Notes from the studies behind them:

- A full sphere uses N/2 polynomials of one parity: double `N` for the shell rules.
- Below Ek 3e-8 the induction budget resid𝐛 of torsional modes needed up to 1.44 times the `'magnetic'` N; their
  eigenvalues had converged to 1e-6 already at it.
- The residuals do not see a too small `lmax`: judge `lmax` by the change of the eigenvalue and by the ℓ-convergence.
  With a field, `lmax` near `N` is safe; the torsional modes kept their eigenvalues to 1e-6 with lmax ≈ 425
  (Ek/1e-9)<sup>−0.145</sup>, far below `N` at low Ek. Solar inertial modes in a full sphere needed lmax 30–80 at N
  600–900, set by their ℓ-convergence.
- Thermal runs: the entropy can be unresolved while P and T look converged; check residσ and resid𝑠 against a run at
  higher `N`.

## Bad solutions

| symptom | what to try |
| --- | --- |
| the mode you want is not among the solutions | put the target on it (in nHz: `units.set_target(par, omega=..., sigma=...)`) and raise `nev`; to look for growing modes, put the target at a positive growth rate |
| along a series of runs, the mode drifts away from a fixed target, or the solver jumps to another mode | let each run aim at the eigenvalue the previous run found: `track_target = 1` ([Following a mode](output.md#following-a-mode)) |
| in a full sphere without viscosity or without thermal diffusion, modes that move when `N` or `lmax` is raised, many with irreg₀ near 1 | spurious modes ([Regularity at the centre](numerical-method.md#regularity-at-the-centre)): target the mode you want and keep only the modes that stay put |
| the run stops with `INFOG(1)=-9` or `-20` | MUMPS's workspace is too small: set `st_mat_mumps_icntl_14` to 50, then 100–200 or more (extra workspace, in %). A run near the limit can fail once and pass the next time |
| a forced run returns NaN | the same problem: set `mat_mumps_icntl_14` = 100, or turn on BLR (below); `mat_mumps_icntl_4` = 1 prints MUMPS's error codes |
| the run stops with `INFOG(1)=-13`, or is killed | out of memory: see below |
| in a cluster of nearly equal modes, some fail `spin_doctor` or move slightly between identical runs | round-off in MUMPS on several processes is amplified for nearly parallel eigenvectors. `st_mat_mumps_cntl_1` = 0.1 (default 0.01) makes this less frequent. Repeat the run and compare each mode with the repeat or with nearby parameters |
| two modes at an avoided crossing split differently from run to run | their sum is stable: quote the sum, or a trend over several runs |

Runs on a single process are exactly reproducible, so a small case can be debugged on one process.

Leave MUMPS's own scaling on, and do not lower the pivot threshold `cntl_1` below its default. `par.prescale = None`
(the default) chooses the matrix rescaling automatically.

## Out of memory

The memory grows roughly as $N^2$ times `lmax` to a power somewhat above 1, and with the width of the couplings
(differential rotation, the thermal wind, the magnetic field). As a reference, the smooth Sun example
(`examples/smooth_sun`) needs about 21 GB.

What to try, cheapest first:

1. **Resolution.** Use the smallest `N` and `lmax` that pass the checks above. The eigenvalue usually converges at a
   lower `N` than the residuals need.
2. **`drop_tol = 1e-14`** removes negligible matrix entries: about 30 % less memory, with essentially unchanged
   eigenvalues.
3. **Block low-rank factors (BLR):** `st_mat_mumps_icntl_35 = 2` (for forced runs `mat_mumps_icntl_35 = 2`), with the
   tolerance `st_mat_mumps_cntl_7 = 1e-14` (the default). Somewhat less memory, and often faster. Check the residuals:
   a larger tolerance can spoil the eigenvectors.
4. **Out of core:** `st_mat_mumps_icntl_22 = 1` writes the factors to disk, for about half the memory and a slower run.
   Point `st_mat_mumps_ooc_tmpdir` to a real disk, not to a `/tmp` held in memory.
5. **More processes** spread the factors: less memory per process, a little more in total. `lmax − m + 1` must be a
   multiple of twice `ncpus`.

Set `icntl_14` only as high as `INFOG(1)=-9` requires, since it adds memory.

## The cost of a differential rotation

A rotation profile with Legendre terms up to degree n couples each harmonic degree of the solution to its neighbours up
to n away. With the heat equation the thermal wind reaches further (about twice as far with `thermal_wind = 2`). The
matrices get wider, and the factorization needs more memory and time. The memory grows roughly linearly with n: at
degree 32 it can be ten times that of the same run without differential rotation.

To plan a large run, measure the cost at your resolution with

```bash
python3 tests/diff_rot/dr_cost.py WORKDIR N LMAX NCPUS [LIMIT_GB]
```

It runs a test problem for increasing degrees n, stops any step above `LIMIT_GB` (default 30) and skips the larger
degrees.
