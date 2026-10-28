# Non-dimensionalization

kore solves dimensionless equations. This page gives the scales and the four parameters that carry them. The
equations, and what the anelastic approximation drops, are on the page [Anelastic equations](anelastic.md).

## The dimensional equations

In a frame rotating at the rate $\Omega$ about $\hat{\mathbf z}$, a perturbation with velocity $\mathbf u$ and specific
entropy $s'$ of the background $\rho_0(r)$, $p_0(r)$, $g(r)$, $s_0(r)$, $\nu(r)$, $\kappa(r)$ obeys

$$
\nabla\cdot(\rho_0\mathbf u) = 0,
$$

$$
\partial_t\mathbf u + 2\Omega\thinspace\hat{\mathbf z}\times\mathbf u = -\nabla\Pi + g\thinspace\frac{s'}{c_p}\thinspace\hat{\mathbf r} + \frac{1}{\rho_0}\nabla\cdot\left(\rho_0\nu\thinspace\mathsf S\right),
$$

$$
p_0\left(\partial_t s' + u_r\frac{ds_0}{dr}\right) = \nabla\cdot\left(\kappa\thinspace p_0\nabla s'\right),
$$

with the reduced pressure $\Pi$, the traceless rate of strain $\mathsf S$, and $c_p$ the specific heat at constant
pressure. These are the equations of the [Anelastic equations](anelastic.md) page for an ideal gas of uniform
composition, for which $\alpha T_0 = 1$ and $\rho_0 T_0$ is proportional to $p_0$.

## Scales

Each quantity is a scale times a dimensionless variable:

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
p_0\left(\partial_t s + u_r\frac{dS}{dr}\right) = \mathrm{ThermaD}\thinspace\nabla\cdot\left(\kappa\thinspace p_0\nabla s\right),
$$

with

$$
\mathrm{Gaspard} = \Omega\tau, \qquad \mathrm{Beyonce} = \frac{g_{\mathrm{ref}}\thinspace\tau^2}{L},
$$

$$
\mathrm{ViscosD} = \frac{\nu_{\mathrm{ref}}\thinspace\tau}{L^2}, \qquad \mathrm{ThermaD} = \frac{\kappa_{\mathrm{ref}}\thinspace\tau}{L^2}.
$$

The buoyancy frequency is $N^2 = \mathrm{Beyonce}\thinspace g\thinspace dS/dr$ in units of $1/\tau^2$. Eigenvalues are
in units of $1/\tau$, in the rotating frame: $\lambda = \sigma - i\omega$, with $\sigma$ the growth rate and $\omega$ the
frequency.

## The time scale

`par.timescale` and `par.set_scales()` set the four parameters from the Ekman number `Ek` $= \nu_{\mathrm{ref}}/(\Omega L^2)$
and `Etherm` $= \kappa_{\mathrm{ref}}/(\Omega L^2)$:

| `timescale` | $\tau$ | `Gaspard` | `ViscosD` | `ThermaD` | `Beyonce` |
| --- | --- | --- | --- | --- | --- |
| `'rotation'` | $1/\Omega$ | 1 | `Ek` | `Etherm` | 0 |
| `'viscous'` | $L^2/\nu_{\mathrm{ref}}$ | 1/`Ek` | 1 | `Etherm`/`Ek` | 0 |
| `'free fall'` | $\sqrt{L/g_{\mathrm{ref}}}$ | 0 | 0 | 0 | 1 |

`set_scales()` resets all four, so set the ones it leaves at 0 after calling it (`Beyonce` with `'rotation'`). The
parameter files of this branch use `'rotation'`.

Ratios of the parameters do not depend on $\tau$:

- the Ekman number $E = \mathrm{ViscosD}/\mathrm{Gaspard} = \nu_{\mathrm{ref}}/(\Omega L^2)$;
- the Prandtl number $Pr = \mathrm{ViscosD}/\mathrm{ThermaD} = \nu_{\mathrm{ref}}/\kappa_{\mathrm{ref}}$;
- the buoyancy frequency over the rotation rate, $N/\Omega = \sqrt{\mathrm{Beyonce}\thinspace g\thinspace dS/dr}\thinspace/\mathrm{Gaspard}$;
- for convection, the Rayleigh number of an entropy drop $\Delta S$ across the shell (inner minus outer wall),
  $Ra = \mathrm{Beyonce}\thinspace\Delta S/(\mathrm{ViscosD}\thinspace\mathrm{ThermaD}) = g_{\mathrm{ref}}L^3\Delta S/(\nu_{\mathrm{ref}}\kappa_{\mathrm{ref}})$;
- the frequency over the rotation rate, $\omega/\Omega = -\mathrm{Im}\thinspace\lambda/\mathrm{Gaspard}$.

## Physical values

`bin/units.py` turns physical values into the parameters, with $\tau = 1/\Omega$; kore itself never imports it. In a
parameter file, after `r_out` is set:

```python
import units
units.set_physical(par, frame_nHz=456.0, nu=1e12, kappa=1e12)   # Gaspard, Beyonce, ViscosD, ThermaD, diff_rot_frame
units.set_target(par, omega=-271.2, sigma=-33.2)                 # itau, rtau from ω, σ in nHz
```

`set_physical` takes $\nu$ and $\kappa$ in cm²/s and uses the solar radius and surface gravity unless `radius` and
`gsurf` (cgs) are given. For the Sun with `r_out` = 0.985 in a frame rotating at 456 nHz it gives
`Beyonce` $= 4.86\times10^4$ and, for $\nu = 10^{12}$ cm²/s, `ViscosD` $= 7.4\times10^{-5}$. In a run directory,
`python3 bin/units.py` prints the eigenvalues as $\omega$ and $\sigma$ in nHz, $\omega = -\mathrm{Im}\thinspace\lambda\cdot F/\mathrm{Gaspard}$
and $\sigma = \mathrm{Re}\thinspace\lambda\cdot F/\mathrm{Gaspard}$, with $F$ = `diff_rot_frame` the frame rate in nHz.

## The Boussinesq case

With `structure = 'Boussinesq'` ($\rho_0 = p_0 = 1$, $g = r$) and `diffusivity = 'uniform'`, the equations are the
Boussinesq equations: $\nabla\cdot\mathbf u = 0$, the viscous force is $\mathrm{ViscosD}\thinspace\nabla^2\mathbf u$, the
diffusion is $\mathrm{ThermaD}\thinspace\nabla^2 s$, and the buoyancy is $\mathrm{Beyonce}\thinspace r\thinspace s\thinspace\hat{\mathbf r}$,
with $s$ in place of the temperature perturbation. With stratification `'linear'` ($dS/dr = r$), $N^2 = \mathrm{Beyonce}\thinspace r^2$.
