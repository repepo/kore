# Checks of the resolved conducting mantle

These scripts check the resolved conducting mantle (`mantle = 'resolved'`): the field in the layer, its matching at the
outer wall and at `r_surface`, the terms of a fluid slipping along it, its rotation with the fluid, and the energy budget
with it.

## Running them

```shell
tests/mantle/run_all.sh [WORKDIR]
```

builds the test runs (in `tests/mantle/runs/` by default), runs every check and exits with 1 if one failed. It needs
numpy, scipy, mpi4py, and petsc4py and slepc4py with MUMPS; `NCPUS` sets the number of ranks (default 2), `MPIEXEC`
replaces `mpiexec`. The runs are built with `tests/tidal/mk_run.py` (an incompressible fluid in a shell, ricb 0.35,
N 48).

## The checks

**`mantle_check.py`.** Seven groups:

- **Free decay** (`hydro = 0`). Each mode of the field decays at a rate fixed by the conductivities and the walls alone:
  a root of spherical Bessel functions in the fluid matched to the layer's solution, which is a Bessel function too for a
  uniform layer and is integrated with scipy otherwise. A uniform layer and an exponential one (the cases of
  `conductive_IC`'s own test), a full sphere, and a conducting inner core under a layer whose conductivity falls as a
  power of the radius. Expected: the rates to $10^{-10}$ (in practice $10^{-14}$), resid𝐛 and resid𝐄 to $10^{-10}$.
- **conductive_IC.** Five runs with the flow whose eigenvalues the `conductive_IC` branch (2d84351) gave for the same
  parameters: no-slip walls, a fluid slipping along both walls with a field of degree 2 under an exponential layer, a full
  sphere with a slipping fluid, a conducting inner core, and `m` 2 under a thin layer. Expected: $10^{-10}$ (in practice
  $10^{-14}$). These runs use `lmax` 19 to 21 on 2 ranks (1 with `NCPUS = 1`), as the reference runs.
- **Thin layer.** A layer of thickness h and conductivity σ tends to the thin wall (`mantle = 'TWA'`) with `c_cmb` = h and
  `c1_cmb` = σh, with a fluid slipping along it: halving h (0.01, 0.005, 0.0025) halves the difference of the eigenvalues
  (ratios between 0.4 and 0.6).
- **Vanishing conductivity.** A layer with σ = $10^{-9}$ gives the eigenvalues of an insulating mantle ($10^{-7}$).
- **Rigid rotation offset.** With `diff_rot_type = 'rigid'` the layer and a conducting inner core turn with the fluid,
  and every eigenvalue is that of a frame turning with them, shifted by the rotation ($10^{-12}$); resid𝐄 closes in both
  frames ($10^{-8}$). `'Wu 2005'` (`r_out` 0.9, N 64), with a fluid slipping along an exponential layer.
- **Sheared rotation.** resid𝐄 and resid𝐛 close to $10^{-8}$ with the layer, the inner core and the fluid turning at
  different rates (`diff_rot_type = 'user'`), on the same run.
- **Kinematic induction.** `just_induction.py` and `solve_ind.py` give the field of a mode without Lorentz force
  (`Hendrik` $10^{-30}$), in the fluid and in the layer, from its flow and eigenvalue ($10^{-10}$).

About 1.5 min on 2 ranks.
