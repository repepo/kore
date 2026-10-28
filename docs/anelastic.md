# Anelastic equations

kore solves the linear anelastic equations in the form of Lantz (1992) and Braginsky & Roberts (1995), "LBR", for small
perturbations of a background whose density varies with radius. This page gives those equations, what the
approximation drops, how kore makes them dimensionless, and where they hold. The Boussinesq equations are a special
case. How the code discretizes the equations is on [Numerical method](numerical-method.md), and the terms that a
differential rotation adds are derived on [Differential rotation](differential_rotation_details.md).

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

## Non-dimensionalization

kore assumes $\alpha T_0 = 1$ and uses $p_0$ in place of $\rho_0 T_0$, both exact for an ideal gas of uniform
composition. Each quantity is then a scale times a dimensionless variable:

| quantity | scale | dimensionless variable |
| --- | --- | --- |
| length | $L$, the radius of the outer wall (`r_out` star radii) | $r$, with $r = 1$ at the outer wall |
| time | $\tau$ | $t$ |
| velocity | $L/\tau$ | $\mathbf u$ |
| reduced pressure | $L^2/\tau^2$ | $\Pi$ |
| entropy perturbation and background entropy | $c_p$ | $s$, $S$ |
| gravity | $g_{\mathrm{ref}}$ | the profile $g(r)$ of the structure |
| viscosity and thermal diffusivity | $\nu_{\mathrm{ref}}$, $\kappa_{\mathrm{ref}}$ | the profiles $\nu(r)$, $\kappa(r)$ of `diffusivity` |
| density and pressure | any | $\rho_0$, $p_0$ |

The equations are linear in $\mathbf u$ and $s$, and contain $\rho_0$ only through $d\ln\rho_0/dr$ and $p_0$ only through
its shape, so the scales of the velocity, of $\rho_0$ and of $p_0$ do not matter. The solar structures have $g = 1$ at
the surface, so that $g_{\mathrm{ref}}$ is the surface gravity; $\nu_{\mathrm{ref}}$ and $\kappa_{\mathrm{ref}}$ are the
values where the diffusivity profiles are 1 (the convection zone for `'solar_rz'` and `'step'`).

Insert these in the equations, multiply the momentum equation by $\tau^2/L$ and the heat equation by $\tau/c_p$:

$$
\nabla\cdot(\rho_0\mathbf u) = 0,
$$

$$
\partial_t\mathbf u + 2\thinspace\mathrm{Gaspard}\thinspace\hat{\mathbf z}\times\mathbf u = -\nabla\Pi + \mathrm{Beyonce}\thinspace g\thinspace s\thinspace\hat{\mathbf r} + \mathrm{ViscosD}\thinspace\frac{1}{\rho_0}\nabla\cdot\left(\rho_0\nu\thinspace\mathsf S\right),
$$

$$
p_0\left(\partial_t s + u_r\frac{dS}{dr}\right) = \mathrm{ThermaD}\thinspace\nabla\cdot\left(\kappa\thinspace p_0\nabla s\right).
$$

These are the equations kore solves. $dS/dr$ is the background entropy gradient over $c_p$, and $g$, $\nu$ and $\kappa$
are dimensionless profiles, set by the parameters `structure`, `stratification` and `diffusivity`
([Background model](background.md)). Four numbers carry the scales:

| parameter | definition | meaning |
| --- | --- | --- |
| `Gaspard` | $\Omega\tau$ | rotation rate of the frame |
| `Beyonce` | $g_{\mathrm{ref}}\thinspace\tau^2/L$ | gravity |
| `ViscosD` | $\nu_{\mathrm{ref}}\thinspace\tau/L^2$ | viscosity; the Ekman number when $\tau = 1/\Omega$ |
| `ThermaD` | $\kappa_{\mathrm{ref}}\thinspace\tau/L^2$ | thermal diffusivity |

The buoyancy frequency is

$$
N^2 = \mathrm{Beyonce}\thinspace g\thinspace\frac{dS}{dr}
$$

in units of $1/\tau^2$.

**Eigenvalues.** Fields go as $e^{im\phi + \lambda t}$, with $\lambda = \sigma - i\omega$: $\sigma$ is the growth rate and
$\omega$ the frequency, in units of $1/\tau$ in the rotating frame; $\omega > 0$ is prograde. Forced runs
(`forcing > 0`) solve at $\lambda = i\thinspace$`forcing_frequency`.

### The time scale

`par.timescale` and `par.set_scales()` set the four parameters from the Ekman number `Ek` $= \nu_{\mathrm{ref}}/(\Omega L^2)$
and `Etherm` $= \kappa_{\mathrm{ref}}/(\Omega L^2)$:

| `timescale` | $\tau$ | `Gaspard` | `ViscosD` | `ThermaD` | `Beyonce` |
| --- | --- | --- | --- | --- | --- |
| `'rotation'` | $1/\Omega$ | 1 | `Ek` | `Etherm` | 0 |
| `'viscous'` | $L^2/\nu_{\mathrm{ref}}$ | 1/`Ek` | 1 | `Etherm`/`Ek` | 0 |
| `'free fall'` | $\sqrt{L/g_{\mathrm{ref}}}$ | 0 | 0 | 0 | 1 |

`set_scales()` resets all four, so set the ones it leaves at 0 after calling it (`Beyonce` with `'rotation'`). The
parameter files of this branch use `'rotation'`, and `bin/units.py` sets the parameters from physical values for that
choice ([Background model](background.md#units)).

Ratios of the parameters do not depend on $\tau$:

- the Ekman number $E = \mathrm{ViscosD}/\mathrm{Gaspard} = \nu_{\mathrm{ref}}/(\Omega L^2)$;
- the Prandtl number $Pr = \mathrm{ViscosD}/\mathrm{ThermaD} = \nu_{\mathrm{ref}}/\kappa_{\mathrm{ref}}$;
- the buoyancy frequency over the rotation rate, $N/\Omega = \sqrt{\mathrm{Beyonce}\thinspace g\thinspace dS/dr}\thinspace/\mathrm{Gaspard}$;
- for convection, the Rayleigh number of an entropy drop $\Delta S$ across the shell (inner minus outer wall),
  $Ra = \mathrm{Beyonce}\thinspace\Delta S/(\mathrm{ViscosD}\thinspace\mathrm{ThermaD}) = g_{\mathrm{ref}}L^3\Delta S/(\nu_{\mathrm{ref}}\kappa_{\mathrm{ref}})$;
- the frequency over the rotation rate, $\omega/\Omega = -\mathrm{Im}\thinspace\lambda/\mathrm{Gaspard}$.

### The Boussinesq case

With `structure = 'Boussinesq'` ($\rho_0 = p_0 = 1$, $g = r$) and `diffusivity = 'uniform'`, the equations are the
Boussinesq equations: $\nabla\cdot\mathbf u = 0$, the viscous force is $\mathrm{ViscosD}\thinspace\nabla^2\mathbf u$, the
diffusion is $\mathrm{ThermaD}\thinspace\nabla^2 s$, and the buoyancy is $\mathrm{Beyonce}\thinspace r\thinspace s\thinspace\hat{\mathbf r}$,
with $s$ in place of the temperature perturbation. With stratification `'linear'` ($dS/dr = r$), $N^2 = \mathrm{Beyonce}\thinspace r^2$.

## Without thermal diffusion

With `ThermaD = 0` the heat equation loses its diffusion term,

$$
p_0\left(\partial_t s + u_r\frac{dS}{dr}\right) = 0 :
$$

the entropy perturbation is only carried by the flow, and the motion is adiabatic. This is the limit for modes whose
periods are short compared with the time of thermal diffusion. The equation is then first order in $r$, so it takes no
thermal boundary conditions: `bci_thermal` and `bco_thermal` are not used. The rows kore builds in that case are on
[Numerical method](numerical-method.md), and what to watch for in such runs is in
[Background model](background.md#thermad-0).

## Differential rotation

With a differential rotation $\Omega(r,\theta) = \Omega_{\rm ref}\left[1 + \Delta\Omega(r,\theta)\right]$, kore solves in
the frame rotating at $\Omega_{\rm ref}$, where the background is a steady azimuthal flow,

$$
\mathbf{U}_0 = \mathrm{Gaspard}\thinspace\Delta\Omega\thinspace r\sin\theta\thinspace\hat{\mathbf{e}}_\phi .
$$

The perturbation is advected and sheared by this flow. In the momentum equation this adds

$$
\mathrm{Gaspard}\left[\Delta\Omega\thinspace (i\thinspace m\thinspace \mathbf{u} + 2\hat{\mathbf{z}}\times\mathbf{u}) + r\sin\theta\thinspace (\mathbf{u}\cdot\nabla\Delta\Omega)\thinspace \hat{\mathbf{e}}_\phi\right]
$$

to the left side: the advection by the local rotation and its Coriolis force, and the transport of angular momentum by
the perturbation across the shear. The heat equation gains, inside the bracket multiplied by $p_0$,

$$
i\thinspace m\thinspace\mathrm{Gaspard}\thinspace\Delta\Omega\thinspace s + u_r\thinspace \frac{\partial S_{tw}}{\partial r} + \frac{u_\theta}{r}\thinspace \frac{\partial S_{tw}}{\partial\theta}.
$$

The first term is the advection of $s$ by $\mathbf U_0$. The other two are the advection, by the perturbation, of a
latitudinal background entropy $S_{tw}(r,\theta)$: either the thermal wind that balances $\Omega(r,\theta)$ (the
latitudinal entropy a differentially rotating star needs to be in equilibrium; Bekki, Cameron & Gizon 2022) or
prescribed. In a convection zone these terms act at the same order as the momentum terms. The background is fixed:
nothing maintains or evolves $\Omega$ and $S_{tw}$ against diffusion, and $\rho_0$, $T_0$ and $g$ stay spherical (no
centrifugal distortion). [Differential rotation](differential_rotation_details.md) derives these terms and their
spectral form.

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
