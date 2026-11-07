# Checks of the drag at the outer wall

These scripts check the drag at a stress-free outer wall (`kappa_cmb`, `topo_drag = 'rekier'`: `bin/topo.py`): the wall
pulls on the slipping fluid in proportion to the slip, with a constant coefficient or with the drag of the wall's
topography. They also check the power the drag dissipates and the damping rate it implies, which `spin_doctor` reports
(Dtopo, δσtopo).

## Running them

```shell
tests/drag/run_all.sh [WORKDIR]
```

builds the test runs (in `tests/drag/runs/` by default), runs every check and exits with 1 if one failed. It needs
numpy, scipy, mpi4py, and petsc4py and slepc4py with MUMPS; `NCPUS` sets the number of ranks (default 2, and at most 2
are used, as the reference runs), `MPIEXEC` replaces `mpiexec`. The runs are built with `tests/tidal/mk_run.py` (an
incompressible fluid in a shell, ricb 0.35, Ek $10^{-3}$, N 48 or 64, `lmax` 19 to 21).

## The checks

**`drag_check.py`.** Seven groups:

- **conductive_IC.** Eleven runs whose eigenvalues and `rotdyn.dat` the `conductive_IC` branch (2d84351) gave for the
  same parameters: a constant drag on the spin-over mode in a shell and in a full sphere; the drag of the topography on
  the spin-over mode, on an `m = 2` mode and in a full sphere; the rotating solids with a stress-free outer wall,
  without a drag, with each drag, with thin conducting layers and a dipole, and with a conducting inner core, a resolved
  mantle and a field of degree 2; the drag of the topography with a thin layer and a dipole. Expected: the first two eigenvalues to
  $10^{-10}$ (in practice $10^{-14}$), and with the solids the rotation of the solids and the torques on the mantle,
  relative to $\omega_M$, to $10^{-8}$ (in practice $10^{-10}$, the precision of `rotdyn.dat`).
- **Conservation.** With a variable density (`'Wu 2005'`, `r_out` 0.9, N 64) and the rotating solids, `kappa_cmb`
  without a field and the drag of the topography with a dipole, a conducting inner core and a resolved mantle: every
  mode but the rigid rotation of everything ($\lambda = 0$) carries no total angular momentum, closes resid𝐋, and closes
  the energy budget (from `flow.dat` and `rotdyn.dat` without a field, resid𝐄 with). Expected: $10^{-8}$ (in practice
  $10^{-10}$ to $10^{-9}$).
- **Equivalences.** A constant tensor given to the drag of the topography (the test hook `topo_K`) gives the eigenvalues
  of `kappa_cmb`; a rigid differential rotation $a$ with `kappa_cmb` gives those of a frame turning $1 + a$ times faster
  with `kappa_cmb`$/(1 + a)$, shifted by $-i\,m\,\mathrm{Gaspard}\,a$. Expected: $10^{-12}$ (in practice $10^{-14}$ and
  $10^{-16}$).
- **Limits.** `kappa_cmb` = $10^8$ gives the eigenvalues of a no-slip wall, without and with the rotating solids (and
  their `rotdyn.dat`). Expected: $10^{-8}$ (in practice $10^{-11}$ to $10^{-10}$). A weak drag (`kappa_cmb` =
  $10^{-6}$) dissipates `kappa_cmb` times the Dtopo/κ of the stress-free run, without and with the solids; expected
  $10^{-3}$ (in practice $10^{-5}$). With the mantle at rest it changes the damping rate of the spin-over mode by
  `kappa_cmb` times δσtopo/κ; expected $5\cdot10^{-3}$ (in practice $5\cdot10^{-4}$, the change of the other losses
  that the drag causes). With the solids that change is printed, not checked: about half, as the drag also turns the
  mantle.
- **Frequencies.** The buoyancy frequency and the frequency of the flow that the drag of the topography used
  (`params.dat`): `topo_N` and `topo_omega` when set, the stratification `'linear'` with Beyonce 10 ($\sqrt{10}$),
  `itau` and the forcing frequency of a forced run (forcing 6) otherwise; expected $10^{-9}$. The same run with a time
  unit twice as long has its eigenvalues doubled; expected $10^{-10}$ (in practice $10^{-15}$). The forced run closes
  Dtopo = −Wv_cmb; expected $10^{-8}$ (in practice to the printed digits).
- **The power through the wall.** In every run with a drag but the stiff ones (whose slip is round-off), Dtopo is minus
  the power of the viscous stress at the wall on the fluid and on the turning mantle. Expected: $10^{-8}$ (in practice
  $10^{-9}$ or better).
- **Stops.** The settings `background.check_drag` does not cover stop in `submatrices.py` with their message.

About 2 min on 2 ranks.
