# Checks of the thin conducting walls

These scripts check the thin conducting walls (`mantle = 'TWA'`, `innercore = 'TWA'`): their conditions on the field,
the terms of a fluid slipping along them, and the energy budget with them.

## Running them

```shell
tests/twa/run_all.sh [WORKDIR]
```

builds the test runs (in `tests/twa/runs/` by default), runs every check and exits with 1 if one failed. It needs numpy,
scipy, mpi4py, and petsc4py and slepc4py with MUMPS; `NCPUS` sets the number of ranks (default 2), `MPIEXEC` replaces
`mpiexec`. The runs are built with `tests/tidal/mk_run.py` (an incompressible fluid in a shell, ricb 0.35, N 48).

## The checks

**`twa_check.py`.** Five groups:

- **Free decay** (`hydro = 0`). Each mode of a field of uniform conductivity decays at a rate fixed by the wall conditions
  alone, a root of combinations of spherical Bessel functions that the script finds with scipy. A thin wall at the outer
  boundary (with `muw_cmb` 1, `np.inf`), at the inner one, at both with `mu` 1.5, and a full sphere with `muw_cmb` 3.
  Expected: the rates to $10^{-10}$ (in practice $10^{-15}$), resid𝐛 to $10^{-10}$, and resid𝐄 to $10^{-10}$ where
  `c` is 0 (with `c` other than 0 the thin-wall budget holds only to the order of the approximation).
- **conductive_IC.** Five runs with the flow whose eigenvalues the `conductive_IC` branch (2d84351) gave for the same
  parameters: thin walls at both boundaries with no-slip walls, with a fluid slipping along them and a field of degree 2,
  with `mu` 1.3, and a full sphere with a slipping fluid, of either equatorial symmetry. Expected: $10^{-10}$ (in
  practice $10^{-14}$). These runs
  use `lmax` 20 or 19 on 2 ranks (1 with `NCPUS = 1`), as the reference runs.
- **Insulating limit.** Thin walls with `c` and `c1` 0 and `mu = 1` give the eigenvalues of insulating walls
  ($10^{-12}$).
- **Rigid rotation offset.** With `diff_rot_type = 'rigid'` the walls turn with the fluid, and every eigenvalue is that
  of a frame turning with them, shifted by the rotation; resid𝐄 and εTWA, which do not depend on the frame, must not
  change ($10^{-6}$).
- **Sheared walls.** resid𝐄 closes to $10^{-8}$ with thin walls at both boundaries turning at different rates
  (`'shellular_boussinesq'`), with the fluid slipping along them.

About 1 min on 2 ranks.
