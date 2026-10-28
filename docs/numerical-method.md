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
$s = s_l(r)\thinspace Y$.

## Radial basis

Each radial function $P_l$, $T_l$, $s_l$ is expanded in Chebyshev polynomials: N of them on the shell
$[$`ricb`$, 1]$, or, in a full sphere (`ricb = 0`), the N/2 polynomials on $[-1, 1]$ that have the parity of the field.
A radial derivative of order $k$ maps Chebyshev coefficients to coefficients in the Gegenbauer (ultraspherical) basis
$C^{(k)}$, where it is a sparse matrix, and so is the product with a profile expanded in Chebyshev polynomials. The rows
of each section are written in the Gegenbauer basis of its highest derivative: $C^{(4)}$ for `u`, $C^{(2)}$ for `v` and
`h` ($C^{(0)}$ for `h` with `ThermaD = 0`, $C^{(1)}$ if a thermal wind is included). This makes $A$ and $B$ sparse.

## Rows

The pressure is removed by taking the radial components of the curl and of the double curl of the momentum
equation; $\nabla\Pi$ drops out of both because it is a gradient. Each term becomes a radial operator acting on $P_l$,
$T_l$ or $s_l$:

| section | rows | equation used |
| --- | --- | --- |
| `u` | of $P_l$ | $r^5\rho_0\thinspace\hat{\mathbf r}\cdot\nabla\times\nabla\times(\text{momentum})/\mathcal L^3$ |
| `v` | of $T_l$ | $r^3\rho_0\thinspace\hat{\mathbf r}\cdot\nabla\times(\text{momentum})/\mathcal L^3$ |
| `h` | of $s_l$ | $r^2\thinspace(\text{heat equation})/\mathcal L^3$ |

All rows are projected on $Y_l^m$ and divided by $\mathcal L^3$, with $l$ the degree of the row. The powers of $r$ are
`rpower_u` (5) and `rpower_v` (3), the smallest that keep every operator a polynomial in $r$. The
factor $\rho_0$ (`rhopower_u = rhopower_v = 1`) improves the accuracy of the viscous rows near a surface where
$\rho_0$ is small. Without viscosity (`ViscosD = 0`) the momentum rows are $r^3\rho_0^2$ and $r^2\rho_0$ times the
equations. With `ThermaD = 0` the heat rows are divided by $p_0$, $r\thinspace(\partial_t s + u_r\thinspace dS/dr)/\mathcal L^3$,
which is better conditioned where $p_0$ is small (and $r^2/p_0$ times the equation with the thermal wind). The Coriolis
force couples $l$ to $l\pm1$; the other terms keep $l$.

| term | function in `bin/operators.py` | profiles |
| --- | --- | --- |
| $\partial_t\mathbf u$ | `inertia` | $\ln\rho_0$ |
| $2\thinspace\mathrm{Gaspard}\thinspace\hat{\mathbf z}\times\mathbf u$ | `coriolis` | $\ln\rho_0$ |
| $\mathrm{ViscosD}\thinspace\rho_0^{-1}\nabla\cdot(\rho_0\nu\mathsf S)$ | `viscous_diffusion` | $\ln\rho_0$ (up to its 4th derivative), $\nu$ (up to its 2nd) |
| $\mathrm{Beyonce}\thinspace g\thinspace s\thinspace\hat{\mathbf r}$ | `buoyancy` | $g$ |
| $p_0\thinspace\partial_t s$ | `entropy` | $p_0$ |
| $p_0\thinspace u_r\thinspace dS/dr$ | `thermal_advection` | $p_0\thinspace dS/dr$ ($dS/dr$ if `ThermaD = 0`) |
| $\mathrm{ThermaD}\thinspace\nabla\cdot(\kappa p_0\nabla s)$ | `thermal_diffusion` | $\kappa p_0$ |

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

projected as in the table above. `solve.py` finds the eigenvalues nearest a target $\tau$ by shift-and-invert with SLEPc,
factoring $A - \tau B$ once with MUMPS ([Solver](solver.md)). Forced runs solve
$(A - i\thinspace\mathrm{wf}\thinspace B)\mathbf x = \mathbf b$, with wf the forcing frequency; for the random
force of forcing 11, $\mathbf b$ is derived on [Reynolds-stress forcing](reynolds-stress.md).

## Boundary conditions

The conditions (`assemble.bc_u_spherical`, `bc_theta_spherical`) are applied at the outer wall (`bco`) and at the inner
one (`bci`, when `ricb > 0`); the full sphere has no condition at the centre:

| setting | condition | rows |
| --- | --- | --- |
| `bco = 0` stress-free | $u_r = 0$, and zero tangential stress $\partial_r(u_\theta/r) = \partial_r(u_\phi/r) = 0$ | $P_l = 0$, $\thinspace r^2P_l'' + r^2(\ln\rho_0)'P_l' + \left[r^2(\ln\rho_0)'' + \mathcal L - 2 - r(\ln\rho_0)'\right]P_l = 0$, $\thinspace rT_l' - T_l = 0$ |
| `bco = 1` no-slip | $\mathbf u = 0$ | $P_l = 0$, $\thinspace P_l' + \left[(\ln\rho_0)' + 1/r\right]P_l = 0$, $\thinspace T_l = 0$ |
| `bco_thermal = 0` | fixed entropy | $s_l = 0$ |
| `bco_thermal = 1` | zero diffusive entropy flux | $s_l' = 0$ |

The density enters the stress-free and no-slip rows through $u_\theta$ and $u_\phi$. The thermal conditions are used
only when `ThermaD > 0`. Without viscosity only $P_l = 0$ is imposed. `bco = 2` (and `bci = 2`) replaces $u_r = 0$ by a
zero viscous radial stress, $\left[r(\ln\rho_0)' - 3\right]P_l + 3rP_l' = 0$, which does not contain the pressure.

## Checks

`spin_doctor.py` evaluates the energy balance of each solution, built from the fields independently of the
matrices: with $K = \int\rho_0\lvert\mathbf u\rvert^2\thinspace dV$,

$$
\sigma K = -\frac{\mathrm{ViscosD}}{2}\int\rho_0\nu\thinspace\lvert\mathsf S\rvert^2\thinspace dV + \mathrm{Beyonce}\thinspace\mathrm{Re}\int\rho_0\thinspace g\thinspace u_r^* s\thinspace dV ,
$$

and the entropy balance, the heat equation times $s^*$ integrated:

$$
\sigma\int p_0\lvert s\rvert^2\thinspace dV = -\mathrm{Re}\int p_0\frac{dS}{dr}u_r s^*\thinspace dV - \mathrm{ThermaD}\int\kappa p_0\lvert\nabla s\rvert^2\thinspace dV .
$$

Their relative residuals are residσ and resid$s$ ([Solver](solver.md) gives the thresholds). `ru_insane.py` checks the
same pairings on the operators themselves: inertia Hermitian and positive, Coriolis anti-Hermitian, viscous Hermitian and
