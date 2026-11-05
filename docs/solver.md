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
are below 1e-5 and its ℓ-convergence is below 1e-4. In a full sphere, also check irreg₀: above 1e-6 the mode is not
smooth at the centre ([Regularity at the centre](numerical-method.md#regularity-at-the-centre)). The same residuals judge
a forced run, where σ = 0. When the walls
move radially (forcing 14), residσ also tests the pressure at the walls, which needs more radial resolution than the rest
of the budget ([Tidal forcing](tidal-forcing.md#interpreting-the-response)).

A failure is one of two kinds:

- **Resolution.** It fails every time, and the eigenvalue moves when N or lmax is raised. Raise N for the residuals and
  lmax for the ℓ-convergence. In a full sphere (ricb = 0) N counts both parities, so it needs about twice the N of a
  shell. N must be even.
- **Solver.** It comes and goes between identical runs, or between runs with nearly the same parameters. See below.

## Bad solutions

| symptom | what to try |
| --- | --- |
| the mode you want is not among the solutions | put the target on it (`units.set_target(par, omega=..., sigma=...)`, in nHz) and raise `nev`; to look for growing modes, put the target at σ > 0 |
| in a full sphere without viscosity or with `ThermaD` = 0, modes that move when N or lmax is raised, some of them growing, many with irreg₀ near 1 | spurious modes ([Regularity at the centre](numerical-method.md#regularity-at-the-centre)): target the mode you want and keep the modes that stay put in N and lmax |
| the run stops with `INFOG(1)=-9` | MUMPS's workspace is too small: `st_mat_mumps_icntl_14` = 50, then 100–200, or more (extra workspace, in %; the free core nutation of `tests/centrifugal/fcn.sh`, with a field and a conducting inner core, runs with 500) |
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
degree 4 at the same N, and `thermal_wind` 2 about 1.7 times `thermal_wind` 1 ([below](#the-cost-of-a-differential-rotation),
how this grows with the degree).

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

## The cost of a differential rotation

A rotation profile with Legendre terms up to degree n couples each degree l of the solution to the degrees l − n to
l + n (to l ± (n + 1) between its poloidal and toroidal parts). With the heat equation the thermal wind reaches
further: to l ± (n + 2) with `thermal_wind` 1, and to l ± (2n + 2) with `thermal_wind` 2. The matrices get wider, and
the factorization in `solve.py` needs more memory and time. `tests/diff_rot/dr_cost.py` measures this on one problem:
N 96, lmax 97, m 2, structure 'Wu 2005' with stratification 'top hat', ricb 0.35, and a 'user' rotation of degree n,
on 4 ranks of a laptop (Apple M4 Pro). The peak memory and the time of `solve.py`:

| degree n | momentum only | with the heat equation, `thermal_wind` 1 | `thermal_wind` 2 |
| --- | --- | --- | --- |
| no rotation | 0.9 GB, 1 s | 1.0 GB, 2 s | 1.0 GB, 2 s |
| 0 | 0.9 GB, 1 s | 1.4 GB, 2 s | 1.4 GB, 2 s |
| 2 | 1.3 GB, 2 s | 2.3 GB, 4 s | 2.8 GB, 6 s |
| 4 | 1.8 GB, 3 s | 3.1 GB, 7 s | 4.3 GB, 12 s |
| 8 | 2.5 GB, 5 s | 5.0 GB, 17 s | 6.9 GB, 29 s |
| 12 | 3.5 GB, 9 s | 7.1 GB, 31 s | 8.5 GB, 35 s |
| 16 | 4.6 GB, 13 s | 8.4 GB, 49 s | 11.9 GB, 67 s |
| 24 | 6.1 GB, 43 s | 10.4 GB, 63 s | |
| 32 | 8.1 GB, 29 s | 15.3 GB, 95 s | |

How to read it:

- The memory grows about linearly with n, here by 0.2 GB per degree with the momentum equation alone and 0.4 GB per
  degree with the heat equation. At n 32 the solve needs 9 times the memory of the run without rotation, and 15 times
  with the heat equation.
- `thermal_wind` 2 costs a little less than `thermal_wind` 1 at twice the degree: its heat rows reach about twice as far.
- `submatrices.py` (2 s) and `assemble.py` (7 s at most) stay small: the factorization is what grows.
- The time is noisier than the memory, since it also depends on how many iterations the eigensolver needs: in the
  momentum case n 24 took longer than n 32.

The memory also grows with N and lmax (above). To plan a large run, run the script at its resolution:

```bash
python3 tests/diff_rot/dr_cost.py WORKDIR N LMAX NCPUS [LIMIT_GB]
```

It kills any step that goes above `LIMIT_GB` (default 30) and then skips the larger degrees of that case.
