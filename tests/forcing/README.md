# Checks of the boundary and body forcings

These scripts check the forcings of `bin/assemble.py` that are not covered by their own suites (forcing 11 has
`tests/rsf`, forcing 14 `tests/tidal`).

## Running them

```shell
tests/forcing/run_all.sh [WORKDIR]
```

builds the test runs (in `tests/forcing/runs/` by default), runs every check and exits with 1 if one failed. It needs
numpy, scipy, mpi4py, and petsc4py and slepc4py with MUMPS; `NCPUS` sets the number of ranks (default 2), `MPIEXEC`
replaces `mpiexec`. The runs are built with `tests/tidal/mk_run.py` (an incompressible fluid in a rotating shell,
no-slip walls, Ek $10^{-3}$).

## The checks

**`rows_check.py`.** A boundary forcing sets the flow at a wall through the right-hand side of a boundary-condition row
of `bc_u_spherical`: per degree, the rows of the outer wall come first, then those of the inner wall. The check solves
one small forced run per case and reads the wall values back from the solution: the imposed P or T, and the tangential
condition, which stays homogeneous (zero tangential velocity at a no-slip wall, zero tangential stress at a stress-free
one, written for any density) except for forcing 2, whose tide moves the walls along themselves too. Cases: forcing 6 at
a no-slip and at a stress-free inner wall, forcing 7 with m = 0 and m = 2, forcing 9 in an incompressible shell, with
the density of 'Wu 2005' and in a full sphere, forcing 10, forcing 12 with both walls turning, with the inner one under
a stress-free outer wall at rest, and in a full sphere, and forcing 2 with m = 2 for both signs of the frequency and
with m = 0 (ricb 37/42): $P = iwA\thinspace P_{\rm fit}$ at the outer wall and $iwA\thinspace r_i P_{\rm fit}$ at the
inner one, $(rP)'/r = iwA\thinspace(P_{\rm fit} + dP_{\rm fit})$ at both, with the fits of `utils.eccen_tide`. Expected:
every value to $10^{-10}$ of the amplitude (in practice $10^{-12}$ or better). Before 2026-11-06 forcings 6, 7 (m = 2)
and 9 put the inner wall's value on a row of the outer wall, which this check shows as errors of order 1, and before
2026-11-08 forcing 2 (in `conductive_IC`) left out the frequency $w$, and the factor $r_i$ at the inner wall.
Since 2026-11-09 the cases also include the settings that did not run before and run now: forcing 14 in a full sphere
with `bci = 0` and with an inviscid fluid at `bco = 0`, and forcing 9 with the drag `kappa_cmb` at an outer wall that
stays in place (`forcing_amplitude_cmb = 0`), which stopped; forcing 9 in a full sphere with `bci = 0` and forcing 7
(m = 2) with an inviscid fluid at `bco = 0`, whose one wall row is the radial flow, for which `assemble.py` wrote no
right-hand side; and two settings that ran before without a check: forcing 9 at a stress-free outer wall, and forcing
10 with an inviscid fluid at `bco = 0`.

**`stops_check.py`.** The settings of the forced runs that kore stops at startup (`background.check_forcing`, and
`background.check_transport` in `submatrices.py`), each with its message, and the nearby settings that run, each from
`submatrices.py` alone (1 rank, N 24): an inviscid fluid in a shell; the settings of forcings 6, 7, 9, 10, 11, 13 and
14, which `assemble.py` used to test only as it wrote the right-hand side, after every operator was built (6, 7, 9 and
10 then wrote none, and `solve.py` solved whatever `B_forced.npz` an earlier run had left); forcing 7 with m = 0 and an
inviscid fluid, whose flow is 0; a forced run of an entropy or a composition without diffusion where the fluid does not
carry it round: $w + m\Omega = 0$ everywhere (also at a frequency of the scan of forcing 11), $w + m\Omega$ of both
signs in the fluid (a critical layer), or a background flow with a meridional circulation; any forcing at frequency 0
with a wall of zero radial stress (`bco` or `bci` = 2); forcing 9 with the drag where it moves the outer wall. Among
the cases that run: a forced run without diffusion where the fluid carries the scalar round at a rate of one sign.
Also three settings outside the forcings that the same review relaxed: a thin layer and the resolved mantle around an
inviscid fluid with `bco = 2` (its one wall row keeps the fluid from crossing the wall whatever `bco`), and a uniform
differential rotation with a field under `ferraro_policy = 'refuse'`. Expected: every stop with its message, every
other case runs.
