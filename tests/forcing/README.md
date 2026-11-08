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
