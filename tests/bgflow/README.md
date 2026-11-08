# Checks of the background flow

These scripts check the background flow of `bgflow = 1` ([Background flow](../../docs/bgflow.md)): the steady,
axisymmetric flow that kore writes from a run of turning walls (forcing 12) or reads as expressions, the terms it adds
to the equations ([Background flow equations](../../docs/bgflow-equations.md)), and the powers that `spin_doctor`
reports.

## Running them

```shell
tests/bgflow/run_all.sh [WORKDIR]
```

builds the test runs (in `tests/bgflow/runs/` by default), runs every check and exits with 1 if one failed. It needs
numpy, scipy, mpi4py, and petsc4py and slepc4py with MUMPS; `NCPUS` sets the number of ranks (default 2, and at most 2
are used), `MPIEXEC` replaces `mpiexec`. The runs are built with `tests/tidal/mk_run.py` (an incompressible fluid in a
rotating shell, ricb 0.35, Ek $10^{-3}$, no-slip walls, `m` 2, N 48 unless said otherwise). About 8.5 minutes on 2
ranks.

## The checks

**`bgflow_check.py`.** Four groups:

- **conductive_IC.** The `conductive_IC` branch (2d84351) wrote `cic_bgflow_lin.npz` and `cic_bgflow_nl.npz` (this
  folder) from its run of an inner core turning at 1 (`m` 0, N 48, `lmax` 31): the degrees $n \le 12$ of the linear flow
  per unit Ro, and of the nonlinear flow at Ro 1 by its Newton's method. kore's run of the same parameters and
  `bin/bgflow.py 12 [nonlinear]` give both files again, and leave the run's own matrix A in the folder. Under each file
  (Ro 0.3 linear, Ro 1 nonlinear; `lmax` 29) the eigenvalues near $1i$, $0.5i$, $-0.5i$ and $1.5i$ are those of
  `conductive_IC`'s own matrices. Expected: the linear file to $10^{-12}$ of the flow's largest coefficient (in practice
  $2.5\times10^{-15}$), the nonlinear one to $10^{-10}$, since its Newton's method solves about 25 systems that the two
  branches assemble with different round-off (in practice $2.7\times10^{-12}$), A unchanged, the eigenvalues to
  $10^{-9}$ (in practice $4.5\times10^{-12}$), all dense and balanced.
- **Operators.** A toroidal background $T_1 = 0.34\thinspace r^3$, $T_3 = -0.04\thinspace r^3$ is the differential
  rotation $\Delta\Omega = 0.3\thinspace r^2 - 0.2\thinspace r^2 Y_2$ of `diff_rot_type = 'user'`: the matrices A and B
  agree to $10^{-14}$ of their largest entry (in practice $4\times10^{-20}$), in a shell with entropy, composition, a
  field and the density of `'Wu 2005'`, and in a full sphere with entropy and a field. A rigid rotation $T_1 = r$ at Ro
  0.1 is the frame turning at 1 + Ro: the eigenvalues $\lvert\lambda\rvert < 3$ of the run at `Gaspard` 1.1 (and
  `bgflow_Ro` 0, so that both have the same rows and bases), shifted by $-i\thinspace m$ Ro, are those of the run with
  the background, to $10^{-10}$ or ten times their condition number times the machine epsilon, in a viscous full sphere,
  an inviscid one with entropy, composition (without diffusion) and a field, a shell without diffusion of the entropy
  and the composition, a field over a conducting inner core under a resolved mantle (which turn with the fluid), and
  thin conducting walls (in practice $4\times10^{-12}$, a sixth of the bound). `rows_check.py` checks the rows of
  `bgflow.build` against a grid (below).
- **Budgets.** spin_doctor's budgets with the background's powers Wbg, Wbg_thm, Wbg_cmp, and those powers against the
  independent computation of `power_check.py` (below). Eigenvalue problems: `'Wu 2005'` (`r_out` 0.9, N 96) with
  entropy, composition and a field over a conducting inner core and a resolved mantle; a full sphere with entropy and a
  field (N 96); a shell without diffusion of the entropy and the composition; an inviscid full sphere (its r-mode, `m`
  2, `symm` −1; the spurious modes that spin_doctor flags as not regular at the centre are left out); thin conducting
  walls; `m` 1, `symm` −1 (the equatorial angular momentum). Forced problems: forcing 1 (`m` 2), forcing 8 and its frame
  (the flow of forcing 8 plus the libration of the mantle is that of forcing 7, to $10^{-12}$), forcing 8 with a thin
  conducting wall, forcing 12 with and without a field. Expected: residσ, residω, resid𝐋, resid𝑠, resid𝑐, resid𝐛 and
  resid𝐄 below $10^{-9}$ (in practice $3\times10^{-12}$) and residens below $10^{-6}$ without a field (in practice
  $3\times10^{-11}$); in the inviscid run, where σ is about $10^{-4}$ and its budget is round-off, residσ below
  $10^{-6}$ (in practice $8\times10^{-9}$, residens $1.4\times10^{-7}$). The powers agree to $10^{-8}$, about the digits
  that `flow.dat` prints (in practice $6\times10^{-11}$).
- **Stops and warnings.** The settings `bgflow.check` does not cover stop in `submatrices.py` with their message:
  `bgflow` 2, no file, a key or a degree that does not exist, an entropy or a composition without `thermal` or
  `compositional`, a background that is not symmetric about the equator, that crosses a wall, that does not turn rigidly
  at a conducting solid, an inviscid shell, `centrifugal = 1`, `rotdyn = 1`, `topo_drag = 'rekier'`, and a file of
  another `ricb`, another density or (nonlinear) another Ro. `bgflow.py` stops for a run with a background field and for
  a run that is not of forcing 12. A $P_2$ that does not vanish at the centre of a full sphere gives the warning of
  `background.centre_check` and runs.

**`rows_check.py RUNDIR`.** For a background of polynomials ($P_2$, $P_4$, $T_1$, $T_3$, $\sigma_0$, $\sigma_2$,
$\gamma_2$) and the density and $p_0$ of the run (`'Wu 2005'`), the value of each row of kore at a few radii and
degrees, from the angular coefficients and radial monomials of `bgflow.build`, against the same row computed on a grid
of $\theta$ from the vector fields: $\boldsymbol\Omega_U\times\mathbf u - \mathbf U\times\boldsymbol\omega$ in the rows
`u`, `v` (the radial derivative by Richardson extrapolation), the advection in the rows `h`, `c` (with and without the
factor $p_0$, $\rho_0$), $\mathbf U\times\mathbf b$ in the rows `f`, `g`. Expected: $10^{-8}$ of the largest value (in
practice $2.5\times10^{-10}$).

**`power_check.py RUNDIR`.** Wbg, Wbg_im, Wbg_thm and Wbg_cmp of the first solution of a run with `bgflow_profiles`,
from the textbook components of $(\mathbf A\cdot\nabla)\mathbf B$ in spherical coordinates on a grid of $(r, \theta)$,
with the background differentiated from its expressions (not from kore's profile codes), against those of `spin_doctor`.
Prints the relative differences, and the kinetic energy over that of `flow.dat` (1).
