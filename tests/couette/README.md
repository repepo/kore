# Checks of the steady rotation of the walls

These scripts check forcing 12, which turns the walls steadily about the rotation axis and solves for the steady flow
they drive, the spherical Couette flow ([Spherical Couette flow](../../docs/couette-flow.md)), and the budgets of the
forced flows that the walls drive along themselves.

## Running them

```shell
tests/couette/run_all.sh [WORKDIR]
```

builds the test runs (in `tests/couette/runs/` by default), runs every check and exits with 1 if one failed. It needs
numpy, scipy, mpi4py, and petsc4py and slepc4py with MUMPS; `NCPUS` sets the number of ranks (default 2, and at most 2
are used, as the reference runs), `MPIEXEC` replaces `mpiexec`. The runs are built with `tests/tidal/mk_run.py` (an
incompressible fluid in a rotating shell, ricb 0.35, Ek $10^{-3}$, no-slip walls, N 48, `lmax` 31 unless said
otherwise). About 1.5 min on 2 ranks.

## The checks

**`couette_check.py`.** Six groups:

- **conductive_IC.** Eleven runs whose kinetic and magnetic energies the `conductive_IC` branch (2d84351) gave for the
  same parameters: the inner core turning, both walls turning (at 0.4 and −0.3) and the mantle alone, without a field;
  with B0 `'axial'` (Le² = Em = $10^{-3}$) over an insulating, a conducting (`'conducting, Chebys'` and
  `'conducting, Bessel'`) and a thin-layer (`'TWA'`) inner core, under a thin-layer and a resolved mantle, and with the
  permeability jumping at a turning wall (a conducting inner core of permeability 1.5 and conductivity 0.3, a thin outer
  layer with a thickness and permeability 2); and B0 `'Luo_S2'` over the Bessel core. `conductive_IC` writes no magnetic
  energy of the inner core for forced runs, so that reference comes from this branch's `spin_doctor` on its fields.
  Expected: the energies to $10^{-9}$ (the digits of `flow.dat`; in practice $2\cdot10^{-10}$, their rounding), and
  residσ, resid𝐋 and resid𝐄 below $10^{-9}$ (in practice $6\cdot10^{-14}$, and $2\cdot10^{-10}$ with `'Luo_S2'`).
- **Stokes flow.** Without rotation (`Gaspard = 0`) the walls drive a flow $u_\phi = s\thinspace\omega(r)$ alone, whose
  viscous stress carries the same torque through every sphere: $r^4\rho_0\nu\thinspace\omega'$ is constant, and $\omega$
  takes the rates of the walls there. For `'Boussinesq'` $\omega = A + B/r^3$; for `'Wu 2005'` (`r_out` 0.9, the density
  falls 22 times across the shell), and for it with a viscosity that varies (`diffusivity = 'user'`), the check
  integrates the run's own profiles by quadrature. Expected: $T_1 = r\omega/2$ to $10^{-10}$ of its largest value, and
  every other coefficient below $10^{-10}$ of it (in practice $10^{-14}$ and 0).
- **Rigid rotation.** A full sphere whose wall turns, a shell whose walls turn at one rate, and an inner core that turns
  under a stress-free outer wall at rest: the fluid turns rigidly ($T_1 = \Omega r/2$ and nothing else), and an
  axisymmetric field is left as it is ($\mathbf b = 0$), also with conducting solids that turn with the fluid (a
  conducting inner core, thin layers, a resolved mantle, solids whose permeability differs from the fluid's). Expected:
  $10^{-10}$ (in practice $10^{-13}$ in the full sphere, $3\cdot10^{-15}$ in the shells). Every term of the budgets is
  then zero, so the residuals compare round-off with round-off and are not checked.
- **Budgets.** Where `conductive_IC` cannot follow: `'Wu 2005'` (`r_out` 0.9, N 96) with buoyancy, without a field and
  with one over the Chebyshev core and a thin outer layer; a differential rotation (`'Y20-wall-bounded'`) with a field
  over the Chebyshev core; the toroidal B0 `'Malkus'` and `'wall'` over the Chebyshev core; a stress-free outer wall at
  rest, along which the fluid slips, with a drag (`kappa_cmb`), with a thin conducting layer and with a resolved mantle.
  Expected: residσ, residω, resid𝐋, resid𝑠 and resid𝐄 below $10^{-9}$ (in practice $2\cdot10^{-13}$ or better).
  `'Malkus'` ($\mathbf B_0 \propto s\thinspace\hat{\boldsymbol\phi}$) is left as it is by an axisymmetric flow, which
  neither shears it nor carries $B_0/s$, uniform, around: ME, MEic and resid𝐛 must be exactly 0.
- **Other flows that the walls drive.** The libration (forcing 7, $m$ = 0), without a field and with one (a stress-free
  inner wall along a conducting inner core at rest), and the steady thermal forcing (forcing 13). Expected: residσ,
  residω, resid𝐋, resid𝑠 and resid𝐄 below $10^{-9}$ (in practice $3\cdot10^{-13}$ or better), and residω exactly 0 for
  the steady flow.
- **Stops.** The settings `background.check_forcing` does not cover stop in `submatrices.py` with their message: a
  forcing frequency, `m` 1, `symm` −1, no viscosity, a turning inner core in a full sphere, `ThermaD` 0 with buoyancy, a
  wall of zero radial stress, a turning stress-free wall, no turning wall, and a forcing number that does not exist. A
  stress-free wall at rest and a full sphere run.

Before 2026-11-08 resid𝐋 was not computed for forced runs, residσ was about 1 for the libration (the round-off of the
small power of the viscous force relative to itself; it is now relative to the viscous dissipation too where the walls
drive the flow, and so is resid𝐄), and residω, resid𝐛 and resid𝑠 were 0/0 where all their terms vanish (now 0).
