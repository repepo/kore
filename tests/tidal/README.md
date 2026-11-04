# Checks of the tidal forcing (forcing = 14) and of the energy budget

These scripts check forcing 14, a radial velocity $A\thinspace Y_2^m$ imposed at the walls (`bin/assemble.py`), and the
kinetic energy budget of `spin_doctor.py` with the work of the wall stresses and the viscous dissipation
(`utils4pp.wall_powers`, `viscous_dissipation`, `pressure_force`). The equations are in `docs/anelastic.md` (Energy
balance), how the budget is computed in `docs/numerical-method.md` (Checks), and how to run the forcing in
`docs/tidal-forcing.md`.

## Running them

```shell
tests/tidal/run_all.sh [WORKDIR]
```

builds the test runs (in `tests/tidal/runs/` by default), runs both checks and exits with 1 if one failed. It needs
numpy, scipy, mpi4py, and petsc4py and slepc4py with MUMPS. `NCPUS` sets the number of ranks (default 2), `MPIEXEC`
replaces `mpiexec`. It takes about 2 minutes on 4 ranks. Each script also runs alone, with the directory of its runs as
argument.

`mk_run.py NAME 'par.x = ..; par.y = ..'` builds one run directory (a copy of `bin/` and a `parameters.py`) and runs
`submatrices.py`, `assemble.py`, `solve.py` and `spin_doctor.py` there. Its default is an incompressible fluid
(`'Boussinesq'`) in a rotating shell, ricb 0.35, no-slip walls, m 2, Ek $10^{-3}$, N 96 and lmax about N/2 (set
`g` = lmax/N to change it).

## The checks

**`stokes_check.py`.** Without rotation (`Gaspard = 0`) and in an incompressible fluid, the flow that forcing 14 drives
solves the unsteady Stokes equations $\lambda\mathbf u = -\nabla\Pi + \nu\nabla^2\mathbf u$, $\lambda = i w$, in the
degree $l = 2$ alone. With $\mathbf u = \nabla\times\nabla\times(P\thinspace Y\thinspace\mathbf r)$,

$$
P = a\thinspace r^2 + b\thinspace r^{-3} + c\thinspace h^{(1)}_2(kr) + d\thinspace h^{(2)}_2(kr), \qquad k^2 = -\lambda/\nu,
$$

with $P = A r/6$ and $P' + P/r = 0$ at each wall ($a$ and $j_2(kr)$ only in a full sphere). The potential part carries
the pressure, $\Pi = -\lambda\thinspace(r P_{\rm pot})'$. The check compares the run's $P_2(r)$ with $P$, and
`spin_doctor`'s kinetic energy, the pressure and viscous work at each wall and the viscous dissipation with their exact
values, in four cases: $m$ = 0, 1 (`symm` −1), 2 in a shell with both walls forced, and $m$ = 2 in a full sphere.
Expected: $P_2$ to $10^{-15}$, the other degrees exactly 0, the energies and powers to about $10^{-11}$ (the 10 digits
of `flow.dat`), residσ and residω below $10^{-12}$.

The spherical Hankel functions are written in closed form, with their exponentials taken at the wall where each one is
largest: with thin Stokes layers $j_l(kr)$ and $y_l(kr)$ both grow as $e^{\lvert\mathrm{Im}\thinspace k\rvert r}$, and the
solution in that basis loses all its digits (at $w = 2$, $\nu = 10^{-3}$).

**`budget_check.py`.** Seven runs with a variable density (`'Wu 2005'`, `r_out` 0.99), buoyancy, differential rotation
(`'Y20-wall-bounded'`, and `'conical'` with its thermal wind) and the three velocity wall conditions, forced and free:

| run | what it tests |
| --- | --- |
| `eig_noslip_wu` | eigenproblem, no-slip walls: the wall work is zero |
| `eig_stressfree_wu` | stress-free walls, N 192: zero wall work, also of the viscous stress, which tests the wall traction |
| `eig_bc2_wu` | `bco = 2` (zero radial stress, $u_r \neq 0$), N 192: the pressure does work at the outer wall in an eigenproblem |
| `f14_m2_thermal` | forcing 14, incompressible, with buoyancy |
| `f14_m1_diffrot` | forcing 14, the obliquity tide ($m$ = 1, `symm` −1), with a differential rotation |
| `f14_m2_wu_conical` | forcing 14, variable density, buoyancy, conical rotation and thermal wind, N 192, lmax about N/4 |
| `f14_sphere_wu` | forcing 14 in a full sphere with a variable density, N 384 |

For each solution it requires residσ and residω below $10^{-7}$, residD below $10^{-10}$ and a positive dissipation;
zero wall work where $u_r = 0$ at the walls; and, in forced runs, a positive power input $W_p + W_\nu^{\rm wall}$ equal to
$D_\nu - W_b - W_\Omega$. Last, it checks that `assemble.py` stops on every rank, with its message, when forcing 14
is given a stress-free wall. Expected: residσ $10^{-14}$ to $10^{-10}$, up to $2\times10^{-8}$ for `f14_m2_wu_conical`,
where it varies between repeated runs (MUMPS on several ranks is not reproducible); residD $10^{-15}$ to $10^{-12}$.

**Resolution of the wall pressure.** The pressure at a wall needs more radial resolution than the volume terms, with a
variable density: for `'Wu 2005'` at Ek $10^{-3}$, residσ of a forced run is $5\times10^{-2}$ at N 96 and
$2\times10^{-10}$ at N 192 in a shell, and 0.36, $6\times10^{-6}$, $10^{-8}$ at N 192, 288, 384 in a full sphere, while
the kinetic energy and the dissipation have converged at the lowest N. Hence the values of N above.
