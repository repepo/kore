# Checks of the runs without the flow

These scripts check the runs with `hydro = 0`, which leave the flow out and solve the field (or the entropy) alone, and
the kinematic induction of `bin/just_induction.py` and `bin/solve_ind.py`.

## Running them

```shell
tests/decay/run_all.sh [WORKDIR]
```

builds the test runs (in `tests/decay/runs/` by default), runs every check and exits with 1 if one failed. It needs
numpy, scipy, mpi4py, and petsc4py and slepc4py with MUMPS; `NCPUS` sets the number of ranks (default 2), `MPIEXEC`
replaces `mpiexec`. The runs are built with `tests/tidal/mk_run.py` (an incompressible fluid in a shell, ricb 0.35).

## The checks

**`decay_check.py`.** Without the flow a field in a fluid of uniform conductivity only diffuses, and each of its modes
decays at a rate fixed by its boundary conditions alone: a root of spherical Bessel functions, which the script finds with
scipy. The cases: a shell between insulators, a conducting inner core (`sigma_i2o` 0.3), a full sphere, the entropy of
`'Boussinesq'` with a fixed entropy and with a fixed flux at the outer wall. Two more cases have no closed form: with a
differential rotation the field's modes without the flow must equal those of the same run with the flow and a vanishing
B0 (the two then decouple), and `just_induction.py` with `solve_ind.py` must give back the field of a mode whose Lorentz
force vanishes, from its flow and its eigenvalue. Expected: every eigenvalue and field to $10^{-10}$ (in practice
$10^{-13}$ or better), and the budgets of `spin_doctor` (resid𝐛, resid𝐄, resid𝑠) to $10^{-9}$ (in practice
$5\cdot10^{-12}$ or better, but resid𝐛 with the conducting inner core, $2\cdot10^{-10}$: it takes the magnetic
diffusion from the integral of $\mathbf b\cdot\nabla^2\mathbf b$, whose round-off grows with N). About 30 s on 2 ranks.
