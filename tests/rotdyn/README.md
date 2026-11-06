# Checks of the rotating solids and of the angular momentum

These scripts check the rotating solids (`rotdyn = 1`): the mantle and the inner core turning about the rotation axis,
coupled to the fluid by the viscous and magnetic torques and to each other by gravity. They also check the angular
momentum balance of the fluid that `spin_doctor` reports (resid𝐋), with or without the solids.

## Running them

```shell
tests/rotdyn/run_all.sh [WORKDIR]
```

builds the test runs (in `tests/rotdyn/runs/` by default), runs every check and exits with 1 if one failed. It needs
numpy, scipy, mpi4py, and petsc4py and slepc4py with MUMPS; `NCPUS` sets the number of ranks (default 2, and at most 2
are used, as the reference runs), `MPIEXEC` replaces `mpiexec`. The runs are built with `tests/tidal/mk_run.py` (an
incompressible fluid in a shell, ricb 0.35, Ek $10^{-3}$, N 48 or 64, `lmax` 19 or 20).

## The checks

**`rotdyn_check.py`.** Six groups:

- **Decoupled solids.** With the fluid's torques switched off (`vtrq = 0`, no field) the solids feel only each other: two
  modes are the roots of $\lambda^2 + c\lambda + K(1/I_M + 1/I_{IC}) = 0$, with $c$ = `Gaspard/OmgtauIC` and $K$ =
  `gTorque Gaspard²`, and the others are the modes of fixed walls (`rotdyn = 0`). Expected: $10^{-12}$ (in practice
  $10^{-14}$).
- **Heavy solids.** Moments of inertia of $10^{12}$ give the eigenvalues of fixed walls. Expected: $10^{-10}$ (in
  practice $10^{-12}$).
- **conductive_IC.** Five runs whose eigenvalues and `rotdyn.dat` the `conductive_IC` branch (2d84351) gave for the same
  parameters: without a field, with two sets of solids (one of them Earth-like); thin walls with a dipole; a conducting
  inner core and a resolved mantle with a field of degree 2; an axial field with an exponential resolved mantle and a thin
  wall on the inner core. Expected: the first two eigenvalues to $10^{-10}$ (in practice $10^{-14}$), and for the first
  the rotation of the solids and the torques on the mantle, relative to $\omega_M$, to $10^{-8}$ (in practice
  $10^{-10}$, the precision of `rotdyn.dat`).
- **Conservation.** With a variable density (`'Wu 2005'`, `r_out` 0.9, N 64) every mode but the rigid rotation of
  everything ($\lambda = 0$) carries no total angular momentum (fluid and solids), closes resid𝐋, and closes the energy
  budget of the fluid and the solids: computed from `flow.dat` and `rotdyn.dat` without a field, resid𝐄 with a dipole, a
  conducting inner core and a resolved mantle. Expected: $10^{-8}$ (in practice $10^{-10}$ to $10^{-9}$).
- **Equatorial angular momentum.** For `m = 1`, `symm = -1` (fixed walls) resid𝐋 checks the balance of the equatorial
  angular momentum, Coriolis torque included: `'Wu 2005'` without and with a field (a conducting inner core and a
  resolved mantle), thin walls with a dipole, a toroidal field that does not vanish at the walls, and a full sphere with
  a field. Expected: $10^{-8}$ (in practice $10^{-15}$ to $10^{-9}$, the full sphere last).
- **Stops.** `rotdyn = 1` with `m = 1`, with a stress-free inner wall, with a differential rotation, with a vanishing
  moment of inertia, with `vtrq = 2`, and with a toroidal field that does not vanish at the walls stops in
  `submatrices.py` with its message.

About 1 min on 2 ranks.
