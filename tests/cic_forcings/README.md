# Checks of the tides and the libration of `conductive_IC`

These scripts check forcings 1, 2 and 8, which came from the `conductive_IC` branch: the wave-like tide driven by the
Coriolis force on the equilibrium tide, and the eccentricity tide of Enceladus
([Tidal forcing](../../docs/tidal-forcing.md)), and the libration in the frame of the mantle
([Libration](../../docs/libration.md)). They also check the stops of the settings these forcings do not cover, and of
forcings 3, 4 and 5 of that branch, which were not ported.

## Running them

```shell
tests/cic_forcings/run_all.sh [WORKDIR]
```

builds the test runs (in `tests/cic_forcings/runs/` by default), runs every check and exits with 1 if one failed. It
needs numpy, scipy, mpi4py, and petsc4py and slepc4py with MUMPS; `NCPUS` sets the number of ranks (default 2, and at
most 2 are used, as the reference runs), `MPIEXEC` replaces `mpiexec`. The runs are built with `tests/tidal/mk_run.py`
(an incompressible fluid in a rotating shell, ricb 0.35, Ek $10^{-3}$, no-slip walls, N 48 unless said otherwise). About
2.5 min on 2 ranks.

## The checks

**`cic_forcings_check.py`.** Six groups:

- **conductive_IC.** Four runs whose kinetic and magnetic energies the `conductive_IC` branch (2d84351) gave for the
  same parameters: the wave-like tide of $m$ = 2 with no-slip and with stress-free walls, and the libration in the frame
  of the mantle without a field and with B0 `'axial'` over a conducting inner core (`'conducting, Chebys'`, Le² = Em =
  $10^{-3}$). `conductive_IC`'s libration force was twice the Poincaré force of its angle, so these runs take twice its
  angle. And the fits of the eccentricity tide (`utils.eccen_tide`) against `conductive_IC`'s, at ricb 0.8 and 0.95, for
  both walls and the three components. Expected: the energies to $10^{-9}$ (the digits of `flow.dat`; in practice
  $10^{-10}$, their rounding), residσ, residω, resid𝐋 and resid𝐄 below $10^{-9}$ (in practice $5\cdot10^{-14}$), and the
  fits to the last bit.
- **The frame of the mantle.** Forcing 8 against forcing 7 with the same libration: the flow seen from the mantle is the
  flow of the mean frame minus the rigid libration of the mantle, and the field is the same in both frames. Cases: an
  incompressible fluid; `'Wu 2005'` (N 96) with buoyancy and a differential rotation (`'Y20-wall-bounded'`); B0
  `'axial'` over the Bessel inner core under a thin conducting mantle; a full sphere; an inner core that librates
  relative to the mantle; a stress-free inner wall. Where nothing couples the fluid to the walls (stress-free walls with
  no drag and no field, an inviscid full sphere) the fluid stays at rest in the mean frame, and the flow is minus the
  libration of the mantle. Expected: $10^{-12}$ of the largest coefficient (in practice $6\cdot10^{-15}$).
- **The wave-like tide beyond `conductive_IC`.** $m$ = 0 and $m$ = 1 (`symm` −1) in a shell and in a full sphere (N 96),
  with no-slip and stress-free walls, $m$ = 2 with a differential rotation and with a viscosity that varies
  (`diffusivity = 'user'`). Expected: residσ, residω, residens, and resid𝐋 for $m$ = 0 and 1, below $10^{-9}$ (in
  practice $2\cdot10^{-14}$, and residens $10^{-10}$). In a full sphere with a stress-free wall, or without viscosity,
  the Coriolis force on the equilibrium tide has a uniform curl, and the wave-like tide is a rigid rotation, about the
  axis for $m$ = 0 and about an equatorial axis for $m$ = 1, whose rate follows from the curl of the momentum equation.
  Expected: its rate to $10^{-10}$ and every other coefficient below $10^{-10}$ of it (in practice $10^{-14}$). The
  budgets of these runs are not checked: the rigid rotation does no work, and every term is round-off. And the run of
  the first group in viscous units (`Gaspard` $10^3$, `ViscosD` 1, the frequency and the amplitude $10^3$ times larger)
  has $10^6$ times its kinetic energy, to $10^{-9}$: the Coriolis force on the equilibrium tide carries `Gaspard`.
- **The eccentricity tide.** $m$ = 2, retrograde and prograde, and $m$ = 0, at ricb 37/42; with `'Wu 2005'` (N 96); with
  a differential rotation (`'Y20'`, which does not vanish at the walls); with B0 `'axial'` and insulating walls.
  Expected: residσ and residω, and resid𝐄 with the field, below $10^{-9}$ (in practice $8\cdot10^{-15}$). Its wall
  values are checked by `tests/forcing/rows_check.py`.
- **The libration where forcing 7 cannot follow.** A stress-free outer wall with a drag (`kappa_cmb`), with a thin
  conducting layer and with a resolved mantle, and a stress-free inner wall along a conducting inner core at rest in the
  frame of the mantle. Expected: residσ, residω, resid𝐋, and resid𝐄 with a field, below $10^{-9}$ (in practice
  $6\cdot10^{-14}$).
- **Stops.** The settings that `background.check_tides` does not cover stop in `submatrices.py` with their message:
  forcings 3, 4 and 5 of `conductive_IC`; the wave-like tide with $m$ = 3, with $m$ = 2 in a full sphere, with
  `'Wu 2005'`, with buoyancy, with a field, with a wall of zero radial stress, with a drag and at frequency 0; the
  eccentricity tide with $m$ = 1, with ricb 0.5, with a stress-free wall, with a conducting thin layer and with a
  conducting inner core; the libration in the frame of the mantle with $m$ = 2, with a libration of the inner core in a
  full sphere, at frequency 0, with an inviscid fluid in a shell, and with a wall of zero radial stress.
