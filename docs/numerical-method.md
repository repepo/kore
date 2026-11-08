# Numerical method

This page shows how kore turns the [anelastic equations](anelastic.md) into the sparse matrices of an eigenvalue
problem: the unknowns, the radial basis, the rows, the boundary conditions and the checks of a solution. The terms of a
differential rotation are built in the same way ([Differential rotation](differential_rotation_details.md), §3 to §7).

## Unknowns

At fixed $m$ and equatorial symmetry, the velocity is written with a poloidal scalar $P$ and a toroidal
scalar $T$,

$$
\mathbf u = \frac{1}{\rho_0}\nabla\times\nabla\times\left(\rho_0 P\thinspace\mathbf r\right) + \nabla\times\left(T\thinspace\mathbf r\right),
$$

so that $\nabla\cdot(\rho_0\mathbf u) = 0$ holds exactly and the mass equation is never solved. For one harmonic
$Y = Y_l^m(\theta,\phi)$, with $P = P_l(r)\thinspace Y$, $T = T_l(r)\thinspace Y$ and $\mathcal L = l(l+1)$,

$$
u_r = \frac{\mathcal L\thinspace P_l}{r}\thinspace Y, \qquad
\mathbf u_h = \left(P_l' + \frac{P_l}{r} + \frac{d\ln\rho_0}{dr}P_l\right) r\nabla Y + T_l\thinspace\nabla Y\times\mathbf r ,
$$

with a prime for $d/dr$ (`utils4pp.cheb2space_pol`, `cheb2space_tor`). With `thermal = 1` the third unknown is
$s = s_l(r)\thinspace Y$. With `compositional = 1` the composition $c = c_l(r)\thinspace Y$, of the degrees and parity
of $s$, comes after the unknowns of the field and before those of the rotating solids, so that a run without it keeps
the layout of its matrices.

## Radial basis

Each radial function $P_l$, $T_l$, $s_l$, $c_l$ is expanded in Chebyshev polynomials: N of them on the shell
$[$`ricb`$, 1]$, or, in a full sphere (`ricb = 0`), the N/2 polynomials on $[-1, 1]$ that have the parity of the field.
A radial derivative of order $k$ maps Chebyshev coefficients to coefficients in the Gegenbauer (ultraspherical) basis
$C^{(k)}$, where it is a sparse matrix, and so is the product with a profile expanded in Chebyshev polynomials. The rows
of each section are written in the Gegenbauer basis of its highest derivative: $C^{(4)}$ for `u`, $C^{(2)}$ for `v`,
`h` and `c` ($C^{(0)}$ for `h` with `ThermaD = 0`, $C^{(1)}$ if a thermal wind is included; $C^{(0)}$ for `c` with
`CompoD = 0`). This makes $A$ and $B$ sparse.

## Rows

The pressure is removed by taking the radial components of the curl and of the double curl of the momentum
equation; $\nabla\Pi$ drops out of both because it is a gradient. Each term becomes a radial operator acting on $P_l$,
$T_l$ or $s_l$:

| section | rows | equation used |
| --- | --- | --- |
| `u` | of $P_l$ | $r^5\rho_0\thinspace\hat{\mathbf r}\cdot\nabla\times\nabla\times(\text{momentum})/\mathcal L^3$ |
| `v` | of $T_l$ | $r^3\rho_0\thinspace\hat{\mathbf r}\cdot\nabla\times(\text{momentum})/\mathcal L^3$ |
| `h` | of $s_l$ | $r^2\thinspace(\text{heat equation})/\mathcal L^3$ |
| `c` | of $c_l$ | $r^2\thinspace(\text{equation of the composition})/\mathcal L^3$ |

All rows are projected on $Y_l^m$ and divided by $\mathcal L^3$, with $l$ the degree of the row. The powers of $r$ are
`rpower_u` (5) and `rpower_v` (3), the smallest that keep every operator a polynomial in $r$. The
factor $\rho_0$ (`rhopower_u = rhopower_v = 1`) improves the accuracy of the viscous rows near a surface where
$\rho_0$ is small. Without viscosity (`ViscosD = 0`) the momentum rows are $r^3\rho_0^2$ and $r^2\rho_0$ times the
equations. With `ThermaD = 0` the heat rows are divided by $p_0$, $r\thinspace(\partial_t s + u_r\thinspace dS/dr)/\mathcal L^3$,
which is better conditioned where $p_0$ is small (and $r^2/p_0$ times the equation with the thermal wind); with
`CompoD = 0` those of the composition are divided by $\rho_0$ in the same way. The Coriolis force couples $l$ to
$l\pm1$; the other terms keep $l$.

| term | function in `bin/operators.py` | profiles |
| --- | --- | --- |
| $\partial_t\mathbf u$ | `inertia` | $\ln\rho_0$ |
| $2\thinspace\mathrm{Gaspard}\thinspace\hat{\mathbf z}\times\mathbf u$ | `coriolis` | $\ln\rho_0$ |
| $\mathrm{ViscosD}\thinspace\rho_0^{-1}\nabla\cdot(\rho_0\nu\mathsf S)$ | `viscous_diffusion` | $\ln\rho_0$ (up to its 4th derivative), $\nu$ (up to its 2nd) |
| $\mathrm{Beyonce}\thinspace g\thinspace s\thinspace\hat{\mathbf r}$ | `buoyancy` | $g$ |
| $p_0\thinspace\partial_t s$ | `entropy` | $p_0$ |
| $p_0\thinspace u_r\thinspace dS/dr$ | `thermal_advection` | $p_0\thinspace dS/dr$ ($dS/dr$ if `ThermaD = 0`) |
| $\mathrm{ThermaD}\thinspace\nabla\cdot(\kappa p_0\nabla s)$ | `thermal_diffusion` | $\kappa p_0$ |
| $\mathrm{BeyonceC}\thinspace g\thinspace c\thinspace\hat{\mathbf r}$ | `comp_buoyancy` | $g$ |
| $\rho_0\thinspace\partial_t c$ | `composition` | $\rho_0$ |
| $\rho_0\thinspace u_r\thinspace dC/dr$ | `comp_advection` | $\rho_0\thinspace dC/dr$ ($dC/dr$ if `CompoD = 0`) |
| $\mathrm{CompoD}\thinspace\nabla\cdot(D\rho_0\nabla c)$ | `comp_diffusion` | $D\rho_0$ |

`submatrices.py` writes one sparse matrix per radial operator, for example `u1lho2_D0` for $r^4\rho_0\thinspace(\ln\rho_0)''$
times the unknown (the label grammar and the profile codes are in `utils.decode_label` and `radial_profiles.proffdir`).
For the section `h` row with `ThermaD > 0`, for example, the diffusion term is

$$
r^2\nabla\cdot(\kappa p_0\nabla s) = \left[r^2\kappa p_0\thinspace s_l'' + \left(2r\kappa p_0 + r^2(\kappa p_0)'\right)s_l' - \mathcal L\thinspace\kappa p_0\thinspace s_l\right] Y.
$$

## Eigenproblem

`assemble.py` puts the blocks into $A\mathbf x = \lambda B\mathbf x$. $B$ holds $-$`inertia` in the
momentum rows and `entropy` in the heat rows. $A$ holds `coriolis` $-$ `viscous_diffusion` $-$ `buoyancy` in the
momentum rows and $-$`thermal_advection` $+$ `thermal_diffusion` in the heat rows, so each row reads

$$
\lambda\thinspace\mathbf u + 2\thinspace\mathrm{Gaspard}\thinspace\hat{\mathbf z}\times\mathbf u = \mathrm{ViscosD}\thinspace\rho_0^{-1}\nabla\cdot(\rho_0\nu\mathsf S) + \mathrm{Beyonce}\thinspace g\thinspace s\thinspace\hat{\mathbf r},
$$

$$
\lambda\thinspace p_0 s = -p_0 u_r\frac{dS}{dr} + \mathrm{ThermaD}\thinspace\nabla\cdot(\kappa p_0\nabla s),
$$

projected as in the table above. The composition enters in the same way: `composition` in $B$, $-$`comp_buoyancy` in
the momentum rows of $A$, and $-$`comp_advection` $+$ `comp_diffusion` in its own rows. `solve.py` finds the eigenvalues nearest a target $\tau$ by shift-and-invert with SLEPc,
factoring $A - \tau B$ once with MUMPS ([Solver](solver.md)). Forced runs solve
$(A - i\thinspace\mathrm{wf}\thinspace B)\mathbf x = \mathbf b$, with wf the forcing frequency; for the random
force of forcing 11, $\mathbf b$ is derived on [Reynolds-stress forcing](reynolds-stress.md).

## Boundary conditions

The conditions (`assemble.bc_u_spherical`, `bc_theta_spherical`) are applied at the outer wall (`bco`) and at the inner
one (`bci`, when `ricb > 0`); the full sphere has no condition at the centre:

| setting | condition | rows |
| --- | --- | --- |
| `bco = 0` stress-free | $u_r = 0$, and zero tangential stress $\partial_r(u_\theta/r) = \partial_r(u_\phi/r) = 0$ | $P_l = 0$, $\thinspace r^2P_l'' + r^2(\ln\rho_0)'P_l' + \left[r^2(\ln\rho_0)'' + \mathcal L - 2 - r(\ln\rho_0)'\right]P_l = 0$, $\thinspace rT_l' - T_l = 0$ |
| `bco = 0` with a drag (`kappa_cmb`, `topo_drag`) | $u_r = 0$, and the tangential stress balances the drag of the wall on the slip | the stress-free rows plus the terms of the drag ([Drag at the outer wall equations](drag-equations.md#the-rows)) |
| `bco` 0 or 2 where the background field ends in a current sheet | the tangential stress balances the push of the sheet on the fluid | the stress-free rows plus the terms of the push ([Magnetic field equations](magnetic-equations.md#current-sheets-at-slipping-walls)) |
| `bco = 1` no-slip | $\mathbf u = 0$ | $P_l = 0$, $\thinspace P_l' + \left[(\ln\rho_0)' + 1/r\right]P_l = 0$, $\thinspace T_l = 0$ |
| `bco_thermal = 0` | fixed entropy | $s_l = 0$ |
| `bco_thermal = 1` | zero diffusive entropy flux | $s_l' = 0$ |
| `bco_compositional = 0` | fixed composition | $c_l = 0$ |
| `bco_compositional = 1` | zero diffusive flux of the composition | $c_l' = 0$ |

The density enters the stress-free and no-slip rows through $u_\theta$ and $u_\phi$. The thermal conditions are used
only when `ThermaD > 0`, those of the composition only when `CompoD > 0`. Without viscosity only $P_l = 0$ is imposed. `bco = 2` (and `bci = 2`) replaces $u_r = 0$ by a
zero viscous radial stress, $\left[r(\ln\rho_0)' - 3\right]P_l + 3rP_l' = 0$, which does not contain the pressure.

## Regularity at the centre

The centre of a full sphere is not a boundary, but the fields must be smooth there, and kore imposes no condition to
make them so. This section says why none is needed, when a solution is still not smooth at the centre, and how to
check it.

A field that is smooth at the centre has radial functions that go as $r^l$ there, for a component of degree $l$, times
a series in $r^2$. The basis gives $P_l$, $T_l$, $s_l$ (and the $f_l$, $g_l$ of a magnetic field) the parity of $l$, and
the equations exclude the lower powers of that parity, as long as the background is smooth at the centre too. Each
background profile follows the same rule with its own degree $n$: the $f_n$ of a differential rotation, the $h_n$,
$k_n$ of a magnetic field and the $\sigma_n$ of a thermal wind go as $r^n$, $dS/dr$ and $g$ as $r$, and $\ln\rho_0$,
$p_0$, $\nu$, $\kappa$ are series in $r^2$. kore evaluates every profile at $\lvert r\rvert$ with the parity of its
degree, which gives the parity but neither the power nor the smoothness. A profile can fail in two ways:

- **It lacks the power.** The $P_2$ term of `'Y20'` and `'conical'`, and the entropy of `thermal_wind` 3, keep their
  value at the centre. The modes converge in N, but to solutions that are not smooth there either.
- **It has a kink or a jump.** Its even or odd continuation to $r < 0$ is not smooth: `'Y20-wall-bounded'`,
  `'shellular'` unless $\sigma$ is even and positive, a `'user'` expression in $\lvert r\rvert$, a $dS/dr$ that changes
  abruptly near the centre. The modes then converge slowly in N.

With `ricb = 0`, `submatrices.py` checks every profile near the centre and prints a WARNING for each failure
(`background.centre_check`).

**Spurious modes.** Without viscosity (`ViscosD = 0`) or with `ThermaD = 0`, the spectrum of a full sphere also holds
modes that do not converge when N or lmax is raised: without viscosity some of them grow, with `ThermaD = 0` they
crowd near zero frequency. Many of them sit at the centre, where they are not smooth; others sit at the wall.
Target the frequency of the mode you want, and keep the modes that stay put in N and lmax.

**irreg₀.** In a full sphere `spin_doctor` prints irreg₀ for each solution (column 28 of `flow.dat`): the largest term
$a_p r^p$ with $p < l$ of the flow and of the field, at $r = \pi/\mathrm N$ (about the spacing of the Chebyshev points
at the centre), relative to their largest value (`utils4pp.centre_irregularity`). It is at round-off for a resolved
smooth mode and near 1 for a spurious mode at the centre. Values in between come from a background that is not smooth
at the centre, or from a mode that is not resolved there. After the table `spin_doctor` counts the solutions above
$10^{-6}$.

## Checks

`spin_doctor.py` evaluates the budgets of [Energy balance](anelastic.md#energy-balance) for each solution, from the
fields and independently of the matrices. The radial integrals use Gauss–Legendre quadrature with $3N/2$ nodes, the
angular ones the orthogonality of the harmonics ($\int\lvert Y_l\rvert^2\thinspace d\Omega = 4\pi/(2l+1)$). The force of a
differential rotation and the dissipation $D_\nu$ are computed on an $(r, \cos\theta)$ grid, with Gauss–Legendre
quadrature in $\cos\theta$, exact for these polynomials. Each residual is relative to the largest of its terms:

| residual | budget | in `spin_doctor`'s names |
| --- | --- | --- |
| residσ | $2\sigma K - W_\nu - W_b - W_c - W_\Omega - W_p - W_f = 0$ (and the Lorentz work) | `2σ KE - Dkin - Wthm - Wcmp - Wdr - Wp - Wf` |
| residω | its imaginary part, with the Coriolis force | `2w KE + Wcor_im - Dkin_im - Wthm_im - Wcmp_im - Wdr_im - Wp_im - Wf_im`, $w = \mathrm{Im}\thinspace\lambda$ |
| resid$s$ | $2\sigma T - W_{\rm adv} + D_{\rm th} - W_{\Omega,\rm th} = 0$ | `2σ TE - Dthm - Wadv - Wdr_thm` (`Dthm` $= -D_{\rm th}$) |
| resid$c$ | $2\sigma C - W_{\rm adv,c} + D_{\rm c} = 0$ | `2σ CE - Dcmp - Wadv_cmp - Wdr_cmp` (`Dcmp` $= -D_{\rm c}$) |
| residD | $W_\nu - W_\nu^{\rm wall} + D_\nu = 0$ | `Dkin - (Wv_icb + Wv_cmb) + Dvisc` |
| resid𝐋 | $\lambda L$ = the torques on the fluid ([Rotating solids equations](rotating-solids-equations.md#the-angular-momentum-balance-residl)) | `lam*angmom - (-vtorq - vtorq_ic - mtorq - mtorq_ic [+ i Gaspard angmom])` |

residσ, resid$s$ and resid$c$ judge whether a solution is resolved ([Solver](solver.md) gives the thresholds), also for forced
runs, where $\sigma = 0$. Where the walls drive the flow along themselves (the libration, forcing 7 with $m$ = 0, and
the steady rotation of the walls, forcing 12), $W_\nu = W_\nu^{\rm wall} - D_\nu$ is a small difference of two large
terms, so residσ, and resid𝐄 with a field, are relative to $D_\nu$ too. A residual whose terms all vanish, such as
residω of a steady flow with $m$ = 0, is 0. $W_f = 2\thinspace\mathrm{Re}\int\rho_0\thinspace\mathbf u^*\cdot\mathbf f\thinspace dV$
is the power of a body force that the right-hand side puts in the fluid: that of a known flow $\mathbf x$ whose
equations make it (forcings 1 and 8),

$$
\mathbf f = -\left[\lambda\mathbf x + 2\thinspace\mathrm{Gaspard}\thinspace\hat{\mathbf z}\times\mathbf x -
\frac{\nabla\cdot\boldsymbol\sigma(\mathbf x)}{\rho_0} - \mathbf F_\Omega(\mathbf x)\right] ,
$$

with $\mathbf F_\Omega$ the force of a differential rotation (`utils4pp.body_force_terms`). Its enstrophy term enters
residens, and $W_f$ is 0 for the other forcings. residD compares two independent computations of the viscous terms:
$W_\nu$ from the viscous force of each degree (`utils4pp.visforce`), and $D_\nu$ from the strain rate on the grid
(`utils4pp.viscous_dissipation`). It is $10^{-15}$ to $10^{-12}$.

**The work at the walls** (`utils4pp.wall_powers`). kore does not solve for the pressure. `spin_doctor` recovers it
from the momentum equation: $\nabla\Pi = \mathbf F - \lambda\mathbf u$, with $\mathbf F$ every force but the pressure, and
the consoidal component of a gradient is $\Pi_l/r$, so that for each degree $l$ of $P$

$$
\Pi_l = r\left[\mathbf F - \lambda\mathbf u\right]_{s,l} ,
$$

the consoidal component of the Coriolis, viscous and differential-rotation forces and of $\lambda\mathbf u$
(`utils4pp.pressure_force`). The buoyancy is radial and does not enter. With $q_l = \mathcal L P_l/r$ the radial
velocity, $s_l$ and $t_l$ the consoidal and toroidal components of $\mathbf u$,

$$
W_p = -2\thinspace\mathrm{Re}\sum_l \frac{4\pi}{2l+1}\Big[\rho_0\thinspace r^2\thinspace q_l^*\thinspace\Pi_l\Big]_{\rm inner}^{\rm outer},
$$

and the stress on a sphere is

$$
\boldsymbol\sigma\cdot\hat{\mathbf r} = \mathrm{ViscosD}\thinspace\rho_0\nu\left[\left(2q_l' + \tfrac23(\ln\rho_0)'q_l\right)Y\hat{\mathbf r} + \left(s_l' - \frac{s_l}{r} + \frac{q_l}{r}\right)r\nabla Y + \left(t_l' - \frac{t_l}{r}\right)r\nabla Y\times\hat{\mathbf r}\right],
$$

dotted with $\mathbf u^*$ and integrated over each wall for $W_\nu^{\rm wall}$. `flow.dat` has these terms for each wall
([Output files](output.md)).

$\Pi_l$ needs the radial derivatives of $P_l$ up to the third at the walls. That is where the truncation error of the
tau method is largest, and near a surface of low density the $(\ln\rho_0)'$ terms of the viscous force amplify it. The
wall pressure therefore converges with N more slowly than the volume integrals. For `'Wu 2005'` (`r_out` 0.99) at
Ek $10^{-3}$, residσ of a forced run is $5\times10^{-2}$ at N 96 and $2\times10^{-10}$ at N 192 in a shell, and 0.36,
$6\times10^{-6}$ and $10^{-8}$ at N 192, 288 and 384 in a full sphere, while the kinetic energy and the dissipation have
converged at the lowest N. Where $u_r = 0$ at the walls, $W_p = 0$ whatever the resolution.

`ru_insane.py` checks the same pairings on the operators themselves: inertia Hermitian and positive, Coriolis
anti-Hermitian, viscous Hermitian and negative, the entropy and diffusion blocks, the buoyancy and the advection against
quadratures, and the differential-rotation blocks against an $(r, \theta)$ grid. `tests/tidal` checks the budgets with
moving walls, against an exact solution of the forced flow and on runs with a variable density, buoyancy and a
differential rotation.
