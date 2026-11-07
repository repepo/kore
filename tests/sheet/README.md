# Checks of the current sheets at slipping walls

Where the background field does not meet the field beyond an insulating wall, it ends there in a sheet of current,
which pushes the fluid along the wall. Along a wall that the fluid slips along (`bco` or `bci` 0 or 2), the stress
rows hold that push (`assemble.bc_u_sheet`, `background.sheet_jump`). These scripts check the push, and the stops that
came with it.

## Running them

```shell
tests/sheet/run_all.sh [WORKDIR]
```

builds the test runs (in `tests/sheet/runs/` by default), runs every check and exits with 1 if one failed. It needs
numpy, scipy, mpi4py, and petsc4py and slepc4py with MUMPS; `NCPUS` caps the ranks of each run (at most 2; default 2),
`MPIEXEC` replaces `mpiexec`. The runs are built with `tests/tidal/mk_run.py` (an incompressible fluid in a shell, ricb
0.35), with Ek = Em = $10^{-2}$, `Hendrik` 0.1 and N 160 unless stated. A run without the push is the same run with
the line `sheet_on = ...` of its copy of `assemble.py` switched off.

## The checks

**`sheet_check.py`.** Five groups, 44 checks:

- **Smoothed sheets.** A sheet is the limit of a field that falls to the field beyond across a layer of width $\delta$
  in the fluid, whose force kore takes in the volume. With $\delta$ = $2\cdot10^{-3}$, $10^{-3}$ and $5\cdot10^{-4}$ the
  first four eigenvalues ($m$ = 1 unless stated) of the run with the sheet and its push are approached as $\delta$.
  Expected: the error halves from $10^{-3}$ to $5\cdot10^{-4}$ (ratio between 0.35 and 0.65; in practice 0.49 to 0.53),
  the extrapolation that removes the terms in $\delta$ and $\delta^2$ is ten times closer than the narrowest layer (in
  practice $8\cdot10^{-8}$ to $5\cdot10^{-5}$), and the run without the push is ten times farther ($9\cdot10^{-3}$ to
  1.7). Nine cases: a toroidal sheet (`B0_tor = 'user'`, $k = r$) at the outer wall and at an inner wall with zero
  radial stress (`bci = 2`, with `cnorm_tor` 2 and `Hendrik` 0.025: the same field as with 1 and 0.1, so that a push
  that missed the normalisation would fail); a poloidal one (`'general'`) at the outer wall (degrees 1 and 3 together,
  `cnorm` 2 and `Hendrik` 0.025) and at the inner one; at an outer thin layer that does not conduct, with a thickness
  and a permeability (`c_cmb` 0.05, `muw_cmb` 3, `mu` 1.5), and at an inner one with degree 3 (`mu` 0.8): the push for
  the jump that their steady conditions set (the last group checks those conditions); the toroidal sheet at the outer
  wall of `'Wu 2005'` with a viscosity $1 + r$, where $\rho_0\nu$ at the wall is 0.07; the outer toroidal sheet with a
  drag at the same wall (`kappa_cmb` 0.01), whose terms share its rows, which must move the eigenvalues ten times more
  than the error of the narrowest layer (in practice $2\cdot10^{-2}$); and a poloidal and a toroidal sheet together at
  the outer wall with $m$ = 2 and `symm = -1` (`lmax` 25), whose rows and columns are the other sets of degrees.
- **Axial angular momentum.** With $m$ = 0 between insulating stress-free walls nothing outside the fluid exerts a
  torque on it, so every mode with $\lambda \neq 0$ carries no axial angular momentum. A toroidal sheet of degree 2
  under `'axial'` (N 64; 96 for `'Wu 2005'`) in a shell (`cnorm_tor = 'mag_energy'`), a full sphere and `'Wu 2005'`.
  Expected: $\lvert L_z\rvert$ and resid𝐋 below $10^{-9}$ (in practice below $10^{-13}$), and $\lvert L_z\rvert$ above
  $10^{-3}$ without the push (0.6 to 1.1).
- **Budgets.** residσ and resid𝐄 below $10^{-8}$ in the runs with the push of the two groups above (in practice
  $2\cdot10^{-10}$ and below; resid𝐄 is nan with `mu` other than 1, in the thin-layer cases).
- **Stops.** `submatrices.py` stops with its message on an inviscid full sphere with a poloidal sheet, and with one
  behind a layer where the permeability jumps (the message then asks for a field that meets the layer's condition: with
  `B0_cmb = 'wall'` the same run goes on), on `centrifugal = 1` with a sheet at a slipping wall, and on fields that go
  on beyond a slipping wall where the permeability jumps: `'axial'` with `mu` 1.5 or a thin layer of permeability 2, at
  the outer wall, at the inner one (the field imposed from beyond the outer wall, with no tag of `B0_ic`, which does not
  act on it) and in an inviscid fluid (whose remedy is then a viscous one), and `'dipole'` and `'G21 dipole'` with
  `B0_ic = 'frozen currents'` at a slipping inner wall with a thin layer of permeability 2. The same fields go on with a
  no-slip wall or with one permeability, and a poloidal sheet at a wall with zero radial stress (`bco = 2`) goes on.
- **The walls' conditions.** A poloidal sheet's jump comes from `background._pol_wall_row`, which the smoothed sheets
  cannot check: a wrong condition leaves a sheet in the smoothed runs too, which the push then moves with them. Its rows
  must equal the poloidal rows of the solver (`assemble.bc_b_spherical`: an insulator's, or `bc_b_twa`'s with $c_1 = 0$)
  at both walls, degrees 1 to 6, to $10^{-13}$ (in practice $2\cdot10^{-16}$), in the runs of the outer poloidal sheet
  (insulating walls) and of the two thin layers. And `'FDM'` and `'G21 dipole'`, which `B0_cmb = 'wall'` adjusts to an
  outer layer where the permeability jumps (`B0_alpha_cmb`, a third copy of the condition), have no sheet there, while
  `'G21 dipole'` with `B0_cmb = 'plain'` has one.

The toroidal sheet's stops in an inviscid run and with the rotating solids, and its runs at each kind of slipping wall,
are checked in `tests/inner_core`.

About 11 min on 2 ranks.
