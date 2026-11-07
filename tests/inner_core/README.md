# Checks of the inner core's field

These scripts check the field of a dipole at the centre (`B0 = 'dipole'`) and a fluid slipping along a conducting inner
core (`bci = 0` with `innercore = 'conducting, Chebys'` or `'conducting, Bessel'`), whose flow across the background
field enters the core's electric-field row, and the stops that came with them.

## Running them

```shell
tests/inner_core/run_all.sh [WORKDIR]
```

builds the test runs (in `tests/inner_core/runs/` by default), runs every check and exits with 1 if one failed. It needs
numpy, scipy, mpi4py, and petsc4py and slepc4py with MUMPS; `NCPUS` caps the ranks of each run (at most 2; default 2),
`MPIEXEC` replaces `mpiexec`. The runs are built with `tests/tidal/mk_run.py` (an incompressible fluid in a shell, ricb
0.35, Ek $10^{-3}$, N 48).

## The checks

**`core_check.py`.** Five groups, 50 checks:

- **The dipole against `conductive_IC`.** Seven eigenvalue runs whose eigenvalues the `conductive_IC` branch (2d84351)
  gave for the same parameters (Em = Le = $10^{-2}$, `cnorm` 1): insulating walls, `cnorm = 'mag_energy'`, the
  antisymmetric modes (and resid𝐋), a thin conducting layer at the outer wall, a resolved mantle, a thin layer on the
  inner core, and the rotating solids with a conducting inner core; and the libration of the mantle (forcing 7) with
  `'conducting, Bessel'`. Expected: the eigenvalues to $10^{-10}$ (in practice $4\cdot10^{-12}$, as `conductive_IC` cuts
  the series of the dipole's profile), the kinetic energy to $10^{-9}$ (`flow.dat` has 10 digits).
- **A slipping inner wall against `conductive_IC`.** Five eigenvalue runs with `bci = 0` and `'conducting, Chebys'`:
  `'axial'` with the antisymmetric modes of `m` 1 and with `m` 0 (and resid𝐋), the dipole, `'G21 dipole'` with `mu_i2o`
  2 and `sigma_i2o` 0.3, and `'Luo_S2'` with both walls stress-free. Expected: the eigenvalues to $10^{-10}$ (in practice
  $6\cdot10^{-13}$, $4\cdot10^{-15}$ without the dipole) and resid𝐄 to $10^{-8}$. resid𝐛, the budget of the fluid alone,
  does not see the slip's term in the core's row; resid𝐄 does ($7\cdot10^{-5}$ to $4\cdot10^{-2}$ without it).
- **A slipping inner wall, without a reference.** resid𝐄 with `'Wu 2005'` and buoyancy, the budget with a variable
  density ($10^{-8}$); a rigid rotation offset (`diff_rot_type = 'rigid'`, both walls stress-free) against the frame
  turning with the solids, whose eigenvalues it must give shifted by the rotation ($10^{-12}$, the modes found in both
  runs, at least `nev`) with the same resid𝐄; and `just_induction.py` with `solve_ind.py`, which must give back the field
  of a mode without Lorentz force (`Hendrik` $10^{-30}$) in the fluid and in the core from its flow ($10^{-10}$, in
  practice $6\cdot10^{-15}$).
- **Forced runs, Bessel against Chebyshev.** The libration of the mantle (forcing 7, `m` 0) with a slipping inner wall,
  with `'axial'` and with the dipole, also against `conductive_IC`'s kinetic energy; forcing 10 with `'G21 dipole'`,
  `sigma_i2o` 0.3 and `mu_i2o` 1.5, where a wrong factor of either would show; forcing 6, which moves the inner wall past
  the core at rest; and forcing 10 with a differential rotation. Expected: the flow and the field of the two cores to
  $10^{-11}$ (in practice $3\cdot10^{-14}$), resid𝐄 to $10^{-9}$. resid𝐄 to $10^{-9}$ also under forcing 6 with the core
  deforming (`forced_walls = 'deforming'`), with `'axial'` and with the dipole, and in `'Wu 2005'`, where the flow through
  the wall brings in the slip's term in $(\ln\rho_0)'$ (in practice $6\cdot10^{-12}$; $2\cdot10^{-3}$ without it).
- **Stops.** `submatrices.py` stops with its message on the dipole in a full sphere, `'general'` and `'user'` profiles
  that are not finite at the centre of a full sphere (and goes on with the `where` form its message suggests), a zero
  toroidal field with `cnorm_tor = 'mag_energy'`, a zero `'ferraro'` field with `cnorm = 'mag_energy'` and one that is not finite, a conducting
  inner core with `bci = 2`, `centrifugal = 1` with a fluid slipping along a conducting inner core, the libration of the
  inner core behind a stress-free wall along a conducting core, and a toroidal field that is not zero at the wall of an
  inviscid full sphere with an insulator beyond, whose current sheet there would push the fluid: also a sheet of
  $4\cdot10^{-6}$ of the toroidal field under a strong `'axial'` field (whose $h'$ hides it from `b0_at_wall`'s cut), and the
  small sheet that a profile which is not smooth at the centre leaves at the wall (the message says why); with
  `rotdyn = 1` it stops with the reason of the rotating solids. The same sheet at a viscous slipping wall goes on, its push
  in the stress rows (`tests/sheet` checks it): a stress-free outer or inner wall (`bco` or `bci` 0 or 2), and a thin layer
  that does not conduct at either wall. It goes on too with no-slip walls, a conducting inner core, a conducting thin layer
  at either wall or the resolved mantle beyond, and with `B0_tor = 'wall'`.

About 1.5 min on 2 ranks.
