# Checks of the forced problems with a field

These scripts check the forced problems with a field (`magnetic = 1`, `forcing` other than 0): the forced rows of the
field, the inner core `'conducting, Bessel'`, the conducting solids behind a wall that the forcing moves
(`forced_walls`), and the energy budget where the forcing moves the walls.

## Running them

```shell
tests/forced_field/run_all.sh [WORKDIR]
```

builds the test runs (in `tests/forced_field/runs/` by default), runs every check and exits with 1 if one failed. It
needs numpy, scipy, mpi4py, and petsc4py and slepc4py with MUMPS, and mpmath for the last check (skipped without it);
`NCPUS` sets the number of ranks (default 2, at most 2 are used), `MPIEXEC` replaces `mpiexec`. The runs are built with
`tests/tidal/mk_run.py` (an incompressible fluid in a shell, ricb 0.35, N 48).

## The checks

**`forced_check.py`.** Nine groups:

- **conductive_IC.** Eight forced runs against the solutions of the `conductive_IC` branch (2d84351) for the same
  parameters: libration (`forcing = 7`, m 0) with insulating, conducting (Chebyshev and Bessel) and thin-layer
  (`'TWA'`) inner cores, thin-layer and resolved mantles, and B0 `'axial'` and `'Luo_S2'`; forcing 9 with insulating
  walls, and with thin layers and `forced_walls = 'deforming'`. The kinetic energy and the magnetic energies in the fluid
  and in the inner core must match the reference to $10^{-8}$ (the digits they were written with), and resid𝐄 must close
  to $10^{-8}$. These runs use `lmax` 19 to 21 on 2 ranks (1 with `NCPUS = 1`), as the reference runs. A conducting inner
  core behind a wall that moves radially is not compared: `conductive_IC` follows neither `forced_walls` picture there.
- **Bessel against Chebyshev.** With the core at rest (`forced_walls = 'rigid'`), the inner core
  `'conducting, Bessel'` must give the solution of `'conducting, Chebys'` under forcing 9, also with the core turning
  with a differential rotation (`'shellular_boussinesq'`, so that the frequency in its frame is not the forcing's), and
  under forcing 6, under forcing 9 with `sigma_i2o` 0.3, and with `mu_i2o` 1.5 where forcing 9 moves the outer wall
  only (a core of another permeability needs an inner wall that stays in place). Expected: the flow and the field to
  $10^{-10}$, the energy and the Ohmic loss of the core (closed forms for Bessel) to $10^{-8}$, and resid𝐄 to
  $10^{-9}$.
- **The forced matrix.** A forced run solves $A - i\thinspace w\thinspace B$ in every row: two forcing frequencies must
  give matrices that differ by $i\thinspace\Delta w\thinspace B$, and frequency 0 must give the eigenproblem's $A$
  ($10^{-13}$), with a conducting inner core under the resolved mantle, and with a thin layer at the outer wall and
  `mu_i2o` 2. A frequency scan of forcing 11 with a conducting inner core must not depend on the frequency $A$ was
  assembled at ($10^{-10}$).
- **Moving walls.** resid𝐄 must close to $10^{-9}$ where the forcing moves the walls. With the solids at rest
  (`'rigid'`): a variable density (`'Wu 2005'`, N 96) with forcing 9 and the Bessel core, and forcing 14 with a thin
  outer layer; a toroidal B0 (`'Malkus'`) with forcing 9 and thin layers, and under the resolved mantle; forcing 10 with a
  thin outer layer; a full sphere (N 96) with forcing 9 and a thin outer layer; forcing 9 under the resolved mantle,
  also with a slipping fluid; forcing 7 with m 2 under the resolved mantle; forcing 6 through a slipping thin inner
  layer, and through a no-slip one while a permeable outer layer that does not conduct (`muw_cmb` 2, `c_cmb` 0.001,
  `c1_cmb` 0) stays in place; free walls (`bco = 2`, `bci = 2`) in eigenvalue problems. Where no wall moves radially,
  at a variable density: forcing 13 with the Chebyshev core, libration under the resolved mantle. With the solids
  deforming (`'deforming'`, the default of forcings 9, 10 and 14): forcing 14 with thin layers; the resolved mantle and
  the Chebyshev core at a variable density (forcings 9 and 14), with `'G21 dipole'` (also with a slipping fluid, a
  conductivity that varies across the mantle, and a full sphere), `'Malkus'` (also with thin layers), `'Luo_S2'`, a
  `'general'` field of several poloidal and toroidal degrees (also with thin layers), forcings 6, 10 and 7 with m 2, a
  differential rotation, and a Chebyshev core of `sigma_i2o` 0.3.
- **Thin resolved mantle.** A resolved mantle of thickness h and conductivity 0.02/h must tend to the thin layer of
  conductance 0.02 (`mantle = 'TWA'`, `c_cmb = 0`) as h decreases, under forcing 9 at the outer wall, with B0 `'axial'`
  and `'Malkus'`, for `'rigid'` and `'deforming'`: the difference of the fields must halve from h = 0.01 to 0.005 (1.8 to
  2.2 times smaller) and be below $3\cdot10^{-2}$ at 0.01.
- **The algebra of the radial flow.** The couplings of $u_r\hat{\mathbf r}\times\mathbf B_0$ (`assemble.bc_b_radial`)
  must agree with a projection on a Gauss grid ($10^{-13}$), at both walls, for a poloidal and a toroidal B0 of two
  degrees each, and with half the toroidal part alone (the thin layer under `'deforming'`).
- **Kinematic induction.** `just_induction.py` and `solve_ind.py` in a forced run must give back the field of the
  coupled run from its flow ($10^{-10}$), with the Bessel core, with the Chebyshev core under the resolved mantle, and
  with both solids deforming.
- **Stops.** With `forced_walls = 'deforming'` the Bessel core behind an inner wall that the forcing moves radially
  must stop, also with `forced_walls` left at its default under forcings 9 and 14, which checks that default. A
  permeability that jumps at a wall that the forcing moves, where B0 reaches it, must stop in both pictures: the
  Chebyshev core with `mu_i2o` 2 (`'deforming'` and `'rigid'`), the Bessel core with `mu_i2o` 2 (forcing 6), a
  conducting thin layer (`c_cmb` 0, `c1_cmb` 0.02) with `muw_cmb` 2 under a toroidal B0 (both pictures) and at rest
  under a poloidal one, a layer with a thickness (`c_cmb` 0.01) that does not conduct with `muw_cmb` 2 under a poloidal
  B0, `mu` 2, `mu` = `muw` = 2, and `muw_icb` 0.5 at an inner wall that moves alone. A permeability other than 1 must
  run at a wall that stays in place (`muw_cmb` under forcing 6, `mu_i2o` where only the outer wall moves), as `muw` of an
  insulating layer (`c` = `c1` = 0, where it enters nothing), of conducting thin layers that deform with the boundary
  under a poloidal B0, and of a layer with a thickness that does not conduct under a toroidal B0 alone, and where B0 is
  zero at the walls (the toroidal `'wall'`; a `'general'` field with h and h' zero at the outer wall, whose interpolated
  slope there is not exactly zero). A `forced_walls` other
  than None, `'rigid'` and `'deforming'` must stop, also in an eigenvalue problem. `'deforming'` must run with the
  Chebyshev core and the resolved mantle, and with the Bessel core where only the outer wall moves; forcing 6 must run
  with the Bessel core (`'rigid'` by default). Forcing 10 with a thin outer layer must assemble the matrix of
  `'deforming'` exactly by default, and not that of `'rigid'`. The Bessel core must stop in an eigenvalue problem, in a
  frequency scan, and in `just_induction.py` with a $\lambda$ other than $i$ `forcing_frequency`; so must
  `just_induction.py` with such a $\lambda$ in a run of forcing 9, which writes no `B.npz`.
- **The Bessel ratio.** `utils.xdlogjl`, $x\thinspace j_l'(x)/j_l(x)$, against mpmath for l up to 300 and $\lvert x\rvert$
  up to $1.4\times 10^4$ on the two lines of the core's $k\thinspace r$ ($10^{-12}$).

About 3 min on 2 ranks.
