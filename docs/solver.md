# Solver: bad solutions and memory

`solve.py` finds the eigenvalues λ of $A x = \lambda B x$ nearest a target τ = `rtau` + i `itau`
(shift-and-invert): each iteration solves a system with $A - \tau B$, which MUMPS factors once (a sparse LU). That
factorization takes almost all the memory and causes most of the numerical trouble. This page lists what to try when a
solution looks wrong or a run runs out of memory.

Solver options go in `par.petsc_opts` in `parameters.py`, for example

```python
par.petsc_opts['st_mat_mumps_cntl_1'] = 0.1
```

Eigenvalue runs read the `eps_` and `st_` keys, forced runs (`forcing` > 0) the `ksp_`, `pc_` and `mat_` keys, so the
same MUMPS option is `st_mat_mumps_...` in one and `mat_mumps_...` in the other. An option given on the command line
(`mpiexec -n 4 ./bin/solve.py -st_mat_mumps_icntl_14 100`) wins over `parameters.py`. The defaults, with notes, are in
`bin/params_default.py`.

## First, check the solution

Judge each solution with `spin_doctor.py`, not with the residual that `solve.py` prints: with the Ruiz prescaling (on by
default) that one is for the rescaled problem. Accept a mode when its energy and entropy residuals (residσ, resid𝑠)
are below 1e-5 and its ℓ-convergence is below 1e-4.

A failure is one of two kinds:

- **Resolution.** It fails every time, and the eigenvalue moves when N or lmax is raised. Raise N for the residuals and
  lmax for the ℓ-convergence. In a full sphere (ricb = 0) N counts both parities, so it needs about twice the N of a
  shell.
- **Solver.** It comes and goes between identical runs, or between runs with nearly the same parameters. See below.

## Bad solutions

| symptom | what to try |
| --- | --- |
| the mode you want is not among the solutions | put the target on it (`units.set_target(par, omega=..., sigma=...)`, in nHz) and raise `nev`; to look for growing modes, put the target at σ > 0 |
| the run stops with `INFOG(1)=-9` | MUMPS's workspace is too small: `st_mat_mumps_icntl_14` = 50, then 100–200 (extra workspace, in %) |
| a forced run returns NaN | the same failure, but PETSc does not stop a forced solve: it returns Inf. Set `mat_mumps_icntl_14` = 100; `mat_mumps_icntl_4` = 1 prints MUMPS's error codes |
| the run stops with `INFOG(1)=-13`, or is killed | out of memory: see below |
| some modes of a cluster of nearly equal ones, such as the r-mode ladders of a radiative interior, fail `spin_doctor` or move by 0.05–0.5 nHz between identical runs | MUMPS on several ranks is not bitwise reproducible, and nearly parallel eigenvectors amplify its round-off. `st_mat_mumps_cntl_1` = 0.1 (the pivot threshold, default 0.01) makes such failures about half as frequent, at the same cost. Repeat the run and compare each mode with the repeat or with runs at nearby parameters: `spin_doctor` misses some of these failures |
| two modes at an avoided crossing split differently from run to run (by up to 0.03 nHz) | their eigenvectors are nearly parallel, but the sum of the pair is stable (to 0.002 nHz): quote the sum, or a trend over several runs |

A larger `maxit`, `eps_balance twoside` or a higher N or lmax do not help with such clusters. Repeated runs on one rank
are bitwise identical, so a small case can be debugged on one rank.

Leave MUMPS's own scaling on (`st_mat_mumps_icntl_8` = 0 fails), and do not lower the pivot threshold (`cntl_1` = 1e-8
cost accuracy in thermal runs). `par.prescale` = None (the default) turns the Ruiz prescaling on, except in
thermal runs with `ThermaD` = 0, where it hurts; 0 or 1 force it.

## Out of memory

As a reference, the smooth Sun example (`examples/smooth_sun`: m 8, N 640, lmax 79, rotation to degree 4 with
`thermal_wind` 2, `drop_tol` 1e-14) peaks at 21 GB on 4 ranks.

For the smooth Sun the memory grows roughly as $N^2 (l_{\max} - m + 1)^{1.35}$ (within about 15 %, more above
N ≈ 1800). Wider angular couplings add to it: rotation to degree 12 needed 2.2 times the memory of
degree 4 at the same N, and `thermal_wind` 2 about 1.7 times `thermal_wind` 1.

What to try, cheapest first:

1. **Resolution.** Use the smallest N and lmax that pass the checks above. The eigenvalue usually converges at a lower
   N than the residuals need.
2. **`drop_tol` = 1e-14** removes negligible matrix entries: about 30 % less memory, with the eigenvalues unchanged to
   about 1e-3 nHz. Part of the drop is done after the Ruiz prescaling, and only when that is on.
3. **Block low-rank factors (BLR)**: `st_mat_mumps_icntl_35` = 2, with the tolerance `st_mat_mumps_cntl_7` = 1e-14 (set
   by default): about 13 % less memory and up to twice as fast. Check the residuals: a tolerance of 1e-12 broke the
   eigenvectors at N ≥ 640, and with `ThermaD` = 0 BLR and Ruiz together degraded them.
4. **Out of core**: `st_mat_mumps_icntl_22` = 1 writes the factors to disk, for about half the memory and a slower run.
   Put them on a real disk with `st_mat_mumps_ooc_tmpdir`, not on a `/tmp` held in memory. A full-sphere run at
   N 1024 needed about 90 GB on 8 ranks this way.
5. **More ranks** spread the factors: less memory per rank, a little more in total. `lmax − m + 1` must be a multiple
   of 2 `ncpus`.

`icntl_14` adds workspace, so set it only as high as `INFOG(1)=-9` requires.
