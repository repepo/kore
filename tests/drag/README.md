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
are used, as the reference runs), `MPIEXEC` replaces `mpiexec`. The runs of `drag_check.py` are built with
`tests/tidal/mk_run.py` (an incompressible fluid in a shell, ricb 0.35, Ek $10^{-3}$, N 48 or 64, `lmax` 19 to 21),
those of `deformed_check.py` with `tests/centrifugal/mk_run.py` (one rank each, `NPAR` at a time: by default the
processors, at most one per 2 GB of available memory). The two scripts run side by side, each with its log in
`WORKDIR/logs`, printed once both have ended.

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

**`deformed_check.py`.** The drag on a body flattened by its rotation (`centrifugal = 1`), where it acts on the slip
along the spheroidal wall, per unit of its area, at the latitude of the wall's normal. Eight groups:

- **Axial.** The rigid rotation of a stress-free fluid decays, per unit `kappa_cmb`, at the rate of its integral over
  the wall of the body: the exact spheroid (`'spheroid'`, $f$ = 0.3 and −0.2, a full body and a shell) and the
  first-order `'uniform'` body ($f = \pm0.01$, against the first-order rate). Expected $10^{-6}$ (in practice
  $2\cdot10^{-8}$, the extrapolation in `kappa_cmb`).
- **Spin-over.** The power the drag would dissipate on the inviscid spin-over of the exact spheroid, whose flow is
  known, against its integral over the wall: `kappa_cmb`, each direction alone and a tensor that varies with latitude
  (the test hook `topo_K`), and the drag of the topography, with its buoyancy frequency along the wall. Expected
  $10^{-10}$ (in practice $5\cdot10^{-11}$), and $3\cdot10^{-6}$ for the drag of the topography, whose kinks limit
  kore's quadrature (in practice $4\cdot10^{-8}$).
- **First order.** The same in the `'uniform'` body ($f = \pm0.02, \pm0.01$) against the exact spheroid of the same
  flattening: the difference is of second order in $f$, for `kappa_cmb` and for the drag of the topography. Expected:
  an order between 1.7 and 2.3 (in practice 1.98 to 2.02).
- **Relabelled.** A sphere labelled as deformed inside, with a flattening $c(1 - r^2)$ that vanishes at the wall but
  not its derivative: the drag on its spin-over must stay that of the sphere. Its part odd in $c$, over $c$ and the
  sphere's value, is $6\cdot10^{-7}$ at most (expected $10^{-4}$); without the terms of the wall's metric in the
  derivative of the shape it is 0.27.
- **Weak.** In a viscous spheroidal shell ($f$ = 0.3, Ek $10^{-3}$) a weak drag changes the damping rate of the
  spin-over mode by the δσtopo the run reports: `kappa_cmb`, each direction and the drag of the topography. Expected
  $5\cdot10^{-3}$ (in practice $4\cdot10^{-4}$).
- **Stiff.** `kappa_cmb` = $10^8$ on that shell gives the eigenvalues of a no-slip wall. Expected $10^{-8}$ (in
  practice $4\cdot10^{-12}$).
- **Sphere.** $f = 0$ with the drag of the topography builds the matrices of `centrifugal = 0`, bitwise.
- **Stops.** The rotating solids still stop with a deformed body.

About 1 min on 4 processors.
