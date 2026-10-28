# Anelastic equations

This branch solves the linear anelastic equations in the form of Lantz (1992) and Braginsky & Roberts (1995), "LBR",
about a background whose density varies with radius. This page gives those equations, says what the approximation
drops, and shows how the code builds them. The Boussinesq equations are the special case `structure = 'Boussinesq'`.
The scales and parameters are on the page [Non-dimensionalization](nondimensionalization.md).

## Background

The background is at rest in the frame rotating at $\Omega$, spherically symmetric and hydrostatic: density
$\rho_0(r)$, pressure $p_0(r)$, temperature $T_0(r)$, gravity $-g(r)\thinspace\hat{\mathbf r}$ with
$dp_0/dr = -\rho_0 g$, and specific entropy $s_0(r)$. Its entropy gradient sets the buoyancy frequency,

$$
N^2 = \frac{g\thinspace \alpha T_0}{c_p}\frac{ds_0}{dr},
$$

with $\alpha$ the thermal expansion coefficient and $c_p$ the specific heat at constant pressure. The kinematic viscosity
$\nu(r)$ and the thermal diffusivity $\kappa(r)$ are radial profiles. [Background model](background.md) explains how
to choose all of these.

## The equations

For a perturbation with velocity $\mathbf u$ and specific entropy $s'$, the linear LBR equations are

$$
\nabla\cdot(\rho_0\mathbf u) = 0,
$$

$$
\partial_t\mathbf u + 2\thinspace\mathbf\Omega\times\mathbf u = -\nabla\Pi + g\thinspace\frac{\alpha T_0 s'}{c_p}\thinspace\hat{\mathbf r} + \frac{1}{\rho_0}\nabla\cdot\left(\rho_0\nu\thinspace\mathsf S\right),
$$

$$
\rho_0 T_0\left(\partial_t s' + u_r\frac{ds_0}{dr}\right) = \nabla\cdot\left(\kappa\thinspace\rho_0 T_0\nabla s'\right),
$$

with the reduced pressure $\Pi = p'/\rho_0 + \Phi'$ ($\Phi'$ the perturbation of the gravitational potential) and the
rate-of-strain tensor without its trace,

$$
\mathsf S = \nabla\mathbf u + (\nabla\mathbf u)^{\mathsf T} - \tfrac{2}{3}(\nabla\cdot\mathbf u)\thinspace\mathsf I .
$$

The momentum equation is per unit mass. The flow is not solenoidal: $\nabla\cdot\mathbf u = -u_r\thinspace d\ln\rho_0/dr$.
The viscous force is the divergence of the full stress $\rho_0\nu\thinspace\mathsf S$, with the terms that come from
$d\ln\rho_0/dr$, from $d\nu/dr$ and from $\nabla\cdot\mathbf u$. Any $\nu(r)$ is allowed. The bulk viscosity is zero.
The heat flux is a diffusion of entropy.

**What is dropped.** Starting from the linearised compressible equations about the same background, LBR drops two
things (Brown et al. 2012; Vasil et al. 2013):

1. The acoustic part of $\partial_t\rho'$ in the mass equation. This removes sound.
2. The entropy part of $\partial_t\rho'$ in the mass equation, together with the force $(N^2/g)(p'/\rho_0)\thinspace\hat{\mathbf r}$
   in the momentum equation. That force is the buoyancy of the density $p'/c^2$ that the pressure perturbation induces,
   in the part of the stratification that is not adiabatic. These two terms must be dropped together.

The second step treats the background as adiabatic wherever the entropy gradient only corrects a term that is already
there. The entropy gradient is kept in the heat equation, where it is the source of $s'$ and gives gravity waves their
restoring force. The Coriolis force, the self-gravity (absorbed into $\Pi$) and the viscous force are not
approximated.

**Energy.** Multiply the momentum equation by $\rho_0\mathbf u^*$ and integrate. The pressure does no work, although
$\nabla\cdot\mathbf u\neq 0$, because $\nabla\Pi$ is a gradient acting on the solenoidal $\rho_0\mathbf u$ and
$u_r = 0$ on the walls. The Coriolis force does no work, and the viscous force only dissipates,
$-\tfrac12\int\rho_0\nu\thinspace\lvert\mathsf S\rvert^2\thinspace dV \le 0$, for any $\nu(r)$. This holds only because the two
terms of item 2 are dropped together: an anelastic set that keeps the force but drops the mass term makes the pressure
do work (Brown et al. 2012).

## The dimensionless form kore solves

kore uses the radius of the outer wall as the unit of length ($r = 1$; it is `r_out` star radii) and $\tau$ as the
unit of time, and writes $s = s'/c_p$ ([Non-dimensionalization](nondimensionalization.md) gives all the scales). It
assumes $\alpha T_0 = 1$ and uses $p_0$ in place of $\rho_0 T_0$, both exact
for an ideal gas of uniform composition. The equations are then

$$
\nabla\cdot(\rho_0\mathbf u) = 0,
$$

$$
\partial_t\mathbf u + 2\thinspace\mathrm{Gaspard}\thinspace\hat{\mathbf z}\times\mathbf u = -\nabla\Pi + \mathrm{Beyonce}\thinspace g\thinspace s\thinspace\hat{\mathbf r} + \mathrm{ViscosD}\thinspace\frac{1}{\rho_0}\nabla\cdot\left(\rho_0\nu\thinspace\mathsf S\right),
$$

$$
p_0\left(\partial_t s + u_r\frac{dS}{dr}\right) = \mathrm{ThermaD}\thinspace\nabla\cdot\left(\kappa\thinspace p_0\nabla s\right),
$$

where $dS/dr$ is the background entropy gradient over $c_p$, and $g$, $\nu$ and $\kappa$ are now dimensionless profiles
(their scales are in the parameters). The buoyancy frequency is

$$
N^2 = \mathrm{Beyonce}\thinspace g\thinspace\frac{dS}{dr}
$$

in units of $1/\tau^2$. Only $d\ln\rho_0/dr$ and its derivatives enter the momentum equation, and only the shape of
$p_0$ enters the heat equation, so the normalisation of $\rho_0$ and $p_0$ does not matter. With the differential
rotation on, the background flow adds three terms to these equations ([Differential rotation](differential_rotation_details.md),
§2.1).

| parameter | symbol | meaning |
| --- | --- | --- |
| `Gaspard` | $\Omega\tau$ | rotation rate of the frame |
| `Beyonce` | $g_{\mathrm{ref}}\thinspace\tau^2/L$ | gravity, with $g$ the dimensionless profile of the structure ($g = 1$ at the surface for the solar structures) |
| `ViscosD` | $\nu_{\mathrm{ref}}\thinspace\tau/L^2$ | viscosity; the Ekman number when $\tau = 1/\Omega$ |
| `ThermaD` | $\kappa_{\mathrm{ref}}\thinspace\tau/L^2$ | thermal diffusivity |
| `structure` | $\rho_0$, $p_0$, $g$ | `bin/radial_profiles.py` |
| `stratification` | $dS/dr$ | `bin/background.py` (`gradS`); only with `thermal = 1` |
| `diffusivity` | $\nu(r)$, $\kappa(r)$ | `bin/background.py` (`diffusivity`), the profiles $\nu/$`ViscosD` and $\kappa/$`ThermaD` |

Here $L$ is `r_out` star radii, and $g_{\mathrm{ref}}$, $\nu_{\mathrm{ref}}$, $\kappa_{\mathrm{ref}}$ are the values where the profiles
are 1. With `structure = 'Boussinesq'` ($\rho_0 = p_0 = 1$, $g = r$) and uniform diffusivities, the equations are the
Boussinesq equations, with $s$ in place of the temperature perturbation.

**Eigenvalues.** Fields go as $e^{im\phi + \lambda t}$, with $\lambda = \sigma - i\omega$: $\sigma$ is the growth rate and
$\omega$ the frequency, in units of $1/\tau$ in the rotating frame; $\omega > 0$ is prograde. `units.nHz` and
`python3 bin/units.py` give $\omega$ and $\sigma$ in nHz. Forced runs (`forcing > 0`) solve at
$\lambda = i\thinspace$`forcing_frequency`.

## How the code builds them

**Unknowns.** At fixed $m$ and equatorial symmetry, the velocity is written with a poloidal scalar $P$ and a toroidal
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

**Rows.** The pressure is removed by taking the radial components of the curl and of the double curl of the momentum
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

**Eigenproblem.** `assemble.py` puts the blocks into $A\mathbf x = \lambda B\mathbf x$. $B$ holds $-$`inertia` in the
momentum rows and `entropy` in the heat rows. $A$ holds `coriolis` $-$ `viscous_diffusion` $-$ `buoyancy` in the
momentum rows and $-$`thermal_advection` $+$ `thermal_diffusion` in the heat rows, so each row reads

$$
\lambda\thinspace\mathbf u + 2\thinspace\mathrm{Gaspard}\thinspace\hat{\mathbf z}\times\mathbf u = \mathrm{ViscosD}\thinspace\rho_0^{-1}\nabla\cdot(\rho_0\nu\mathsf S) + \mathrm{Beyonce}\thinspace g\thinspace s\thinspace\hat{\mathbf r},
$$

$$
\lambda\thinspace p_0 s = -p_0 u_r\frac{dS}{dr} + \mathrm{ThermaD}\thinspace\nabla\cdot(\kappa p_0\nabla s),
$$

projected as in the table above. Forced runs solve $(A - i\thinspace\mathrm{wf}\thinspace B)\mathbf x = \mathbf b$, with wf the
forcing frequency.

**Boundary conditions** (`assemble.bc_u_spherical`, `bc_theta_spherical`), at the outer wall (`bco`) and at the inner
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

**Checks.** `spin_doctor.py` evaluates the energy balance of each solution, built from the fields independently of the
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
negative.

## Where the approximation holds

- **Sound-proof**: $\max(\lvert\lambda\rvert^2, 4\Omega^2) \ll c^2k^2$ for a mode of wavenumber $k$, with $c$ the sound speed.
  It fails for p modes and near a surface where $c\to 0$.
- **Nearly adiabatic over the scale $\ell$ of the motion**: $N^2\ell/g \ll 1$, the relative size of the dropped terms
  of item 2. In kore's units this is $\lvert dS/dr\rvert\thinspace\ell \ll 1$. Their effect on frequencies is smaller,
  and for $N \gg \omega$ it does not grow with $N$, so a strongly stratified radiative zone is not a problem in itself.
  The case where it fails is that of low-order g modes with frequencies near $N$.
- **A possible background**: hydrostatic, and with $N^2 < g/H_\rho$ ($H_\rho$ the density scale height), since
  $g/c^2 = -d\ln\rho_0/dr - N^2/g$ must be positive. In kore's units this is $dS/dr < -d\ln\rho_0/dr$. kore never uses
  $c$, so it cannot detect a background that breaks this: check it when choosing `structure` and `stratification`
  separately.

## References

- Braginsky, S. I. & Roberts, P. H. 1995, Geophys. Astrophys. Fluid Dyn. 79, 1.
- Brown, B. P., Vasil, G. M. & Zweibel, E. G. 2012, ApJ 756, 109.
- Jones, C. A. et al. 2011, Icarus 216, 120.
- Lantz, S. R. 1992, PhD thesis, Cornell University.
- Lantz, S. R. & Fan, Y. 1999, ApJS 121, 247.
- Vasil, G. M., Lecoanet, D., Brown, B. P., Wood, T. S. & Zweibel, E. G. 2013, ApJ 773, 169.
