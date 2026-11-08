# Checks of the composition

These scripts check the composition (`compositional = 1`), a second scalar that the flow carries across a background
gradient, that diffuses and that has its own buoyancy ([Composition](../../docs/background.md#composition)), and the
`'two zone'` gradient of the entropy and of the composition.

## Running them

```shell
tests/composition/run_all.sh [WORKDIR]
```

builds the test runs (in `tests/composition/runs/` by default), runs every check and exits with 1 if one failed. It
needs numpy, scipy, mpi4py, and petsc4py and slepc4py with MUMPS; `NCPUS` sets the number of ranks of each run (default
2, and at most 2 are used, as the reference runs), `NPAR` the number of runs built at a time (default half the
processors, at most 4), `MPIEXEC` replaces `mpiexec`. The runs are built with `tests/tidal/mk_run.py` (an
incompressible fluid in a rotating shell, ricb 0.35, Ek $10^{-3}$, no-slip walls, $m$ = 2, N 96, `lmax` 49 unless said
otherwise). About 3.5 min on 4 cores.

## The checks

**`composition_check.py`.** Five groups:

- **conductive_IC.** Three runs whose eigenvalues the `conductive_IC` branch (2d84351) gave for the same parameters,
  where its Boussinesq composition $\xi$ is $-c$, `BV2_comp` is `BeyonceC` and `Ecomp` is `CompoD`: its `'internal'`
  gradient (`'linear'` here) between walls at a fixed composition; its `'differential'` gradient (`'inverse square'`
  here, with `BeyonceC` = `BV2_comp` `ricb`/(1 − `ricb`)) between walls without flux; and its `'two zone'` entropy
  under an `'internal'` composition with another diffusivity, no flux at the inner wall and a fixed composition at the
  outer one. Expected: the first three eigenvalues to $10^{-12}$ (in practice $5\cdot10^{-14}$), and residσ, residω,
  resid𝑠 and resid𝑐 below $10^{-11}$.
- **A second entropy.** With `'Boussinesq'` ($\rho_0 = p_0 = 1$) the composition alone and the entropy alone, with
  the same numbers (`BeyonceC` = `Beyonce`, `CompoD` = `ThermaD`, the same gradient and walls), solve the same
  equations. Pairs of runs: a full sphere (also $m$ = 1, `symm` −1 with no flux at the wall, and without diffusion), no
  flux at the inner wall, a differential rotation (also without diffusion), a deformed body (a `'user'` flattening,
  whose $J - 1$ is not 0, and the exact spheroid), a field over a conducting inner core under a resolved mantle, the
  rotating solids (also with a field and a thin conducting mantle), the drag of the topography at a stress-free outer
  wall (whose buoyancy frequency is then $N_c$), the libration (forcing 7), diffusivities that vary (`'user'`, with
  $\kappa$ and $D$ swapped between the two runs, and `'step'` with `comp0` for `therm0`), the free decay
  (`hydro = 0`) and the `'two zone'` gradient. Expected: the eigenvalues (as sets, leaving out $\lambda = 0$ of the
  rotating solids), or the kinetic energy of the forced run, to $10^{-12}$, and `composition.dat` against
  `thermal.dat` to $10^{-12}$ (in practice $10^{-13}$ or better). Also: the two scalars together, equal in everything,
  are one scalar with twice the gradient, so the run has the eigenvalues of the entropy alone with `Beyonce` doubled;
  and with `'Wu 2005'` and no diffusion, where the rows of the composition are divided by $\rho_0$ and those of the
  entropy by $p_0$, the two runs have the same eigenvalues (their budgets are weighted by $\rho_0$ and by $p_0$).
- **Anelastic budgets.** `'Wu 2005'` (`r_out` 0.9) with the composition alone, and with both scalars, a `'two zone'`
  composition without flux at the walls, a diffusivity $D$ that follows the density (`comp_pow`) and a differential
  rotation (`'Y20'`). Expected: residσ, residω, resid𝑠 and resid𝑐 below $10^{-9}$ (in practice $4\cdot10^{-11}$ or
  better).
- **`ru_insane.py`.** The symmetry and quadrature tests of the operators, which cover the composition's blocks, on the
  runs of the anelastic budgets, the composition without diffusion in a full sphere and with `'Wu 2005'`, with a
  differential rotation and in the deformed body. Expected: every test below $10^{-6}$ (in practice $6\cdot10^{-11}$
  or better).
- **Stops.** The settings `background.check_composition` and `check_forcing` refuse stop in `submatrices.py` with their
  message: `compositional` other than 0 or 1, no `comp_stratification`, `'inverse square'` in a full sphere, a negative
  `CompoD`, a `BeyonceC` that is not finite, a wall condition other than 0 or 1, a `'two zone'` step of width 0, a
  `'user'` gradient that does not evaluate, `diffusivity = 'user'` without `'D'`, the wave-like tide (forcing 1) with
  the composition, and the steady rotation of the walls (forcing 12) without its diffusion.

Runs with `compositional = 0` are unchanged by the composition: `tests/diff_rot/regress.sh` and
`tests/rsf/regress_solve.sh` check that their matrices and solutions are bitwise those of the commit before it.
