# Background model

kore solves for small perturbations of a background. The background has four parts, each chosen by name in the
parameter file:

| part | parameter | what it sets |
| --- | --- | --- |
| structure | `structure` | the density $\rho_0$, $p_0 = \rho_0 T_0$ and gravity $g$, functions of $r$ |
| stratification | `stratification` | the entropy gradient $dS/dr$, with $S = \delta s/c_p$ (only with `thermal = 1`) |
| diffusivity | `diffusivity` | the radial profiles of the viscosity $\nu$ and the thermal diffusivity $\kappa$ |
| rotation | `diff_rot`, `diff_rot_type` | a uniform or a differential rotation $\Omega(r,\theta)$, and the latitudinal entropy that balances it |

The equations in which they enter are on the page [Anelastic equations](anelastic.md). The code is in
`bin/background.py` (the choice, the checks, the stratification and diffusivity types), `bin/radial_profiles.py` (one
class per structure) and `bin/diff_rot.py` (the rotation).

## Units

kore is dimensionless ([Non-dimensionalization](anelastic.md#non-dimensionalization)). The unit of length is the radius
of the outer wall, $r = 1$. `r_out` is that radius in star radii (1 by default), so $x = r\cdot$`r_out` is the radius
in star radii, the coordinate of the solar profiles below. The only physical unit kore reads is that of the solar
rotation data, in nHz.

`bin/units.py` turns physical values into the parameters, with the time unit $1/\Omega$; kore itself never imports it. In
a parameter file, after `r_out` is set:

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

## Quick start

The background of the smooth Sun example (`examples/smooth_sun/parameters.py`), a full sphere with the observed solar
rotation:

```python
par.thermal        = 1
par.r_out          = 0.985                 # outer wall in solar radii
par.structure      = 'model S hydrostatic'
par.stratification = 'solar_rz'            # a radiative interior below a slightly subadiabatic convection zone
par.solar_rz_delta_cz = -4e-7
par.diffusivity    = 'solar_rz'            # ν, κ drop to visc0, therm0 below 0.70 R☉
par.visc0, par.therm0 = 0.01, 0.04
par.diff_rot       = 1
par.diff_rot_type  = 'tabulated'           # Ω(r, θ) from a table
par.diff_rot_file  = 'bin/jsoc_hmi_2010_2026.npz'
par.diff_rot_nmax  = 4
par.thermal_wind   = 2                     # with its exact thermal wind
par.thermal_wind_radial = 0
units.set_physical(par, frame_nHz=456.0, nu=1e12, kappa=1e12)
```

The checked-in `bin/parameters.py`, a gravito-inertial mode, uses `structure = 'Wu 2005'` with `r_out = 0.99` and
`stratification = 'top hat'`, with uniform rotation. [Getting started](getting-started.md) uses an incompressible fluid
(`'Boussinesq'`, `thermal = 0`) with a shellular rotation.

## Structure

`structure` has no default.

| `structure` | $\rho_0$, $p_0$, $g$ | parameters |
| --- | --- | --- |
| `'model S hydrostatic'` | the Sun, for new runs: an even polynomial fit of $\ln\rho$ of Model S (`bin/model_s.at`) over the whole Sun, $g$ from its mass, $p_0$ from hydrostatic balance | `structure_deg` (30) |
| `'Wu 2005'` | $\rho_0 = (1 - x^2)^2$ (Wu 2005, $\beta = 2$) with its $g$ and $p_0$ | `r_out` < 1, so that $\rho_0 > 0$ at the wall |
| `'Boussinesq'` | $\rho_0 = p_0 = 1$, $g = r$: the Boussinesq equations | – |
| `'8-term fit'` (legacy) | the smooth fit of `bin/smoothed_sun.py`, up to 48 % off Model S in $\rho$ near the surface, which moves solar inertial modes by 1.5–8 nHz | – |
| `'old model S prototype'` (legacy) | Chebyshev fits of $\ln\rho$ and $\ln p$ of Model S, with the $g$ of the 8-term fit (not hydrostatic) | `solar_density_deg`, `solar_density_window`, `solar_density_sun_deg` |

- Only ratios of $\rho_0$ and of $p_0$ enter the equations, so their scale does not matter. $g$ is 1 at the surface
  ($x = 1$) in the solar structures and in `'Wu 2005'`.
- The three solar structures (the first and the last two rows) also make $x = r\cdot$`r_out` the radius of the solar
  rotation profiles (`diff_rot.solar_x`).
- `'model S hydrostatic'` is within 1.0 %, 0.5 % and 0.75 % of Model S in $\rho$, $p$ and $g$ at degree 30, and within
  0.14 %, 0.06 % and 0.10 % at degree 50. Use 50 to compare with papers that use Model S itself; degrees of 40 and above
  need about 16 more `lmax` for the same $\ell$-convergence of the HFR modes.
- The two legacy structures are kept so that runs made before `'model S hydrostatic'` give the same results. For
  `'old model S prototype'`, `solar_density_window` None fits the shell $[$`ricb`$\cdot$`r_out` $- 0.01$,
  `r_out` $+ 0.005]$ when `ricb` > 0, and is `'sun'` when `ricb` = 0: one even fit over the whole Sun of degree
  `solar_density_sun_deg` (60).

## Stratification

With `thermal = 1`, `stratification` sets $dS/dr$ for $r \ge 0$, in units of $1/L$; it has no default then. In a full
sphere (`ricb = 0`) kore uses it as an odd function of $r$, so it should go to zero smoothly at the centre.

| `stratification` | $dS/dr$ | parameters |
| --- | --- | --- |
| `'top hat'` | $\tfrac{A}{2}\left[\mathrm{erf}\frac{r - x_1}{w_1} - \mathrm{erf}\frac{r - x_2}{w_2}\right]$ | `aux1`…`aux5` = $x_1$, $w_1$, $x_2$, $w_2$, $A$ |
| `'odd top hat'` | the same made odd in $r$: with $x_1 = 0$ it rises as $A\thinspace\mathrm{erf}(r/w_1)$ from the centre | `aux1`…`aux5` |
| `'solar_rz'` | the radiative interior of Mukhopadhyay et al. 2025b, below | `solar_rz_dSdr`, `solar_rz_steps`, `solar_rz_core`, `solar_rz_delta_cz`, `solar_rz_delta_sf`, `solar_rz_os` |
| `'linear'` | $r$; with `'Boussinesq'`, $N^2$ = `Beyonce` $r^2$ (Rieutord 2006, Mirouh et al. 2016) | – |
| `'inverse square'` | $1/r^2$, conduction between the walls | needs `ricb` > 0 |
| `'user'` | `stratification_profile`, a numpy expression ([below](#user-profiles)) | `stratification_param` ($p$) |

- `'top hat'` in a full sphere needs $x_1 \gtrsim 6 w_1$, so that $dS/dr \to 0$ at the centre; otherwise use
  `'odd top hat'`. With `ricb` > 0 and `ThermaD = 0`, a $dS/dr \neq 0$ at the inner wall gives spurious modes trapped
  there ([below](#thermad-0)).
- `'odd top hat'` with `aux1`…`aux5` = 0, 0.03, 0.68, 0.06, 2.6 fits the $dS/dr$ of Model S to 10 % rms.

**`'solar_rz'`.** With $x = r\cdot$`r_out` in solar radii (their eq. 10, plus the convection zone of Bekki et al.
2024, eq. 2):

$$
\begin{aligned}
\frac{dS}{dr} &= \frac{A}{4}\left[1 - \mathrm{erf}\frac{x - x_1}{d}\right]\left[1 - \mathrm{erf}\frac{x - x_2}{d}\right]\mathrm{erf}\frac{x}{w_0}\cr
&\quad + \delta(x)\thinspace \frac{d\ln p_0}{dr}\thinspace \frac{1}{2}\left[1 + \mathrm{erf}\frac{x - x_{OS}}{d_{OS}}\right],
\end{aligned}
$$

$$
\delta(x) = \delta_{cz} + (\delta_{sf} - \delta_{cz})\thinspace e^{-((x - 0.985)/0.015)^2},
$$

with $A$ = `solar_rz_dSdr` (2.615), $[x_1, x_2, d]$ = `solar_rz_steps` ([0.61, 0.68, 0.04]), $w_0$ = `solar_rz_core`
(0.05: makes $dS/dr$ odd and smooth at $r = 0$, no effect above $x \approx 3w_0$), $[x_{OS}, d_{OS}]$ = `solar_rz_os`
([0.70, 0.04]) and $\delta_{cz}$, $\delta_{sf}$ = `solar_rz_delta_cz`, `solar_rz_delta_sf` (0). $A$ is their
$(\partial s_0/\partial r)_{RZ}$ = 1.32e-2 erg g⁻¹ K⁻¹ cm⁻¹ times $L/c_p$, with $c_p$ = 3.46e8 and `r_out` = 0.985, so
it scales with `r_out`. $\delta < 0$ is subadiabatic; kore stops if $\delta$ is set with another stratification. In the
smooth Sun example, $\delta_{cz} = -4\times10^{-7}$ makes the HFR $m = 8$ mode 34 nHz more retrograde than
$\delta_{cz} = 0$.

## Diffusivity

$\nu$ = `ViscosD` $\hat\nu(r)$ and $\kappa$ = `ThermaD` $\hat\kappa(r)$; `diffusivity` sets $\hat\nu$ and $\hat\kappa$.

| `diffusivity` | $\hat\nu$, $\hat\kappa$ | parameters |
| --- | --- | --- |
| `'uniform'` (default) | 1 | – |
| `'step'` | `visc0`, `therm0` below $r$ = `aux3`, 1 above, joined by an erf of width `aux4` | `visc0`, `therm0`, `aux3`, `aux4` |
| `'solar_rz'` | `visc0`, `therm0` below $x_{OS}$, 1 above, as $\tfrac12[1 + \mathrm{erf}((x - x_{OS})/d_{OS})]$ (Mukhopadhyay et al. 2025b, eqs. 8–9) | `visc0`, `therm0`, `solar_rz_os` |
| `'user'` | `diffusivity_profiles = {'nu': '...', 'kappa': '...'}`, numpy expressions ([below](#user-profiles)) | `diffusivity_param` ($p$) |

- With `'step'` and `'solar_rz'`, `ViscosD` and `ThermaD` are the values above the step (the convection zone of the
  Sun). Mukhopadhyay et al. 2025b use `visc0` = 0.01 and `therm0` = 0.04.
- Any type is multiplied by $(\rho_0(x_2)/\rho_0)^p$, $x_2$ = `aux3`, with $p$ = `visc_pow` for $\hat\nu$ and `therm_pow`
  for $\hat\kappa$: 0 (the default) keeps the kinematic diffusivities as set, 1 makes $\rho_0\nu$ and $\rho_0\kappa$
  follow the profile instead.
- [Forcing 11](forcing.md) (random Reynolds stresses) uses the turbulent part of the viscosity, `ViscosD` $(\hat\nu - \nu_f)$: the
  floor $\nu_f$ is `visc0` with `'step'` and `'solar_rz'` when `visc0` < 1 and 0 otherwise, unless `rsf_nu_floor` is
  set.

### ThermaD = 0

`ThermaD = 0` removes the thermal diffusion: the motion is adiabatic, and `bci_thermal`, `bco_thermal` are not used
([Without thermal diffusion](anelastic.md#without-thermal-diffusion)). It works with every structure, stratification and
rotation option. What to watch for:

- **Prefer `ThermaD = 0` to a very small `ThermaD`.** As `ThermaD` → 0 the eigenvalues of a resolved diffusive run
  approach the `ThermaD = 0` ones linearly. With the thermal wind and walls that are both stress-free and at fixed
  entropy (`bci_thermal = bco_thermal = 0`), a small `ThermaD` forms a thermal boundary layer of width about
  $(\mathrm{ThermaD}/\lvert\lambda\rvert)^{1/2}$ at the walls, which below `ThermaD` ≈ $10^{-7}$ needs far more radial
  resolution than the bulk. The `ThermaD = 0` run has no such layer, and neither have zero-flux or no-slip walls.
- **Let $dS/dr$ vanish at the inner wall.** In a shell, a $dS/dr \neq 0$ at the inner wall gives spurious modes trapped
  there. Their eigenvalues move with N, and `spin_doctor` flags them (residσ ≈ 1). For an erf top hat, take
  $x_1 \ge$ `ricb` $+ 3w_1$. If the gradient must reach the wall, raise `nev` by one or two, because the spurious modes
  take some of the slots.
- **With a differential rotation, expect a band of eigenvalues near $\sigma = 0$.** Without diffusion, the advection of
  $s$ by the rotation has a continuous spectrum, $\lambda = -i\thinspace m\thinspace\mathrm{Gaspard}\thinspace\Delta\Omega(r,\theta)$,
  which the discretization shows as many eigenvalues near $\sigma = 0$ with frequencies in the range of
  $m\thinspace\mathrm{Gaspard}\thinspace\Delta\Omega$. With the thermal wind some of them grow slowly. They peak at
  $l \approx$ `lmax` and change with `lmax`: discard any mode that fails the ℓ-convergence check of `spin_doctor`. They
  disappear with `thermal_wind = 0` or with a moderate diffusivity (`ThermaD` ≈ $10^{-3}$). Put the target away from
  the band, where the solver converges slowly.
- **Check the convergence in N.** In a full sphere, eigenfunctions with `ThermaD = 0` need more radial resolution than
  diffusive ones (N = 200 instead of 120 for a smooth solar model). Where $\rho_0$ nearly vanishes at the outer wall
  (structure `'Wu 2005'` with `r_out` close to 1), the thermal-wind rows converge slowly in N.

## Rotation

With `diff_rot = 0` (the default) the background rotates uniformly at the rate of the frame. With `diff_rot = 1` it
rotates at

$$
\Omega(r,\theta) = \Omega_{\rm ref}\left[1 + \Delta\Omega(r,\theta)\right], \qquad \Delta\Omega = \sum_n f_n(r)\thinspace P_n(\cos\theta), \quad n = 0, 2, 4, \dots
$$

where $\Omega_{\rm ref}$ is the rotation rate of the frame in which kore solves, and $\Delta\Omega$ the relative
departure from it, a sum of Legendre polynomials in $\cos\theta$ with radial coefficients $f_n(r)$. Only even $n$
appear, so the rotation is symmetric about the equator. The profiles do not depend on the structure. With
`thermal = 1`, the latitudinal entropy that balances the rotation, the thermal wind, is part of the background too
(`thermal_wind`, below). The terms this adds to the equations are in
[Anelastic equations](anelastic.md#differential-rotation).

### Frame and units

$\Delta\Omega$ is relative to the frame, and `Gaspard` converts it to the time unit, so any `timescale` works. Two
things follow:

- `ViscosD`, `ThermaD` and `Beyonce` must be computed for $\Omega_{\rm ref}$. For `'solar'` the frame must rotate at
  435 nHz; for `'solar_P4'` and `'tabulated'` at `diff_rot_frame` (456 nHz by default). The other types are
  dimensionless.
- With `thermal_wind = 2` the eigenvalues do not depend on the choice of frame. With `thermal_wind = 1` (linear in
  $\Delta\Omega$) they do, at second order in $\Delta\Omega$: by 0.1–0.9 nHz for solar inertial modes.

Eigenvalues are in the frame rotating at $\Omega_{\rm ref}$, as for uniform rotation.

### Parameters

| parameter | default | meaning |
| --- | --- | --- |
| `diff_rot` | 0 | 1 adds the differential rotation |
| `diff_rot_type` | `'Y20'` | the profile, see below |
| `diff_rot_ampl` | 1.0 | the factor of $\Delta\Omega$, for every type (table below) |
| `diff_rot_param` | 0.0 | the shape parameter: $\sigma$ of `'shellular'`, `p` in `'user'` expressions |
| `diff_rot_term_amp` | `[1.0]*9` | $a_0, a_2, \dots, a_{16}$: factors of the $P_0, P_2, \dots, P_{16}$ terms of $\Delta\Omega$, any type, after `diff_rot_ampl` (missing entries are 1); e.g. `[1, 1, 0]` drops the $P_4$ part of `'solar_P4'` |
| `diff_rot_profiles` | `{}` | `'user'`: `{n: 'numpy expression'}`, the shape $f_n$ of each even degree (below) |
| `diff_rot_frame` | 456.0 | `'solar_P4'`, `'tabulated'`: $\Omega_{\rm ref}/2\pi$ in nHz |
| `diff_rot_solar` | `[464, -76, -50, 433.5, 0.71, 0.045]` | `'solar_P4'`: $c_0$, $c_2$, $c_4$, $\Omega_{RZ}$ (nHz), $r_{tc}$, $d_{tc}$ |
| `diff_rot_file` | `''` | `'tabulated'`: the .npz file, absolute or relative to the run directory |
| `diff_rot_nmax` | 8 | `'tabulated'`: highest degree kept, even (up to 16 with `thermal_wind = 2`) |
| `diff_rot_rdeg` | 24 | `'tabulated'`: degree of the Chebyshev fit of each $f_n$ in $r$ |
| `diff_rot_window` | None | `'tabulated'`: None fits over the shell if ricb > 0 and is `'sun'` if ricb = 0; `'sun'`: one fit over the whole Sun, even in $r$, the same for any ricb |
| `diff_rot_sun_deg` | 60 | `'sun'`: degree of that fit |
| `thermal_wind` | 1 | the latitudinal background entropy, see below |
| `thermal_wind_radial` | 1 | 0 drops $u_r\thinspace \partial_r S_{tw}$ |
| `thermal_wind_amplitude` | 1.0 | factor on $S_{tw}$ (`thermal_wind` 1, 2) |
| `dSdtheta` | 0.0 | `thermal_wind = 3`: $\partial_\theta S_{tw}$ = `dSdtheta` $\sin 2\theta$ |

### Profiles

$a$ = `diff_rot_ampl`, $p$ = `diff_rot_param`. For the solar types $x$ is the fractional solar radius: $x = r\times$`r_out`
with a solar structure, $x = r$ otherwise.

| `diff_rot_type` | $\Delta\Omega(r,\theta)$ | degrees | notes |
| --- | --- | --- | --- |
| `'rigid'` | $a$ | $P_0$ | test profile: a uniform offset of the rotation rate, with an exact answer (below); $a = 0$ gives uniform rotation with the diff-rot machinery on (needed for `thermal_wind = 3`) |
| `'Y20'` | $a\thinspace P_2(\cos\theta)$ | $P_2$ | $P_2$ unnormalised, despite the name |
| `'Y20-wall-bounded'` | $a\thinspace (1 - r)(r - r_i)\thinspace P_2(\cos\theta)$, $r_i$ = `ricb` | $P_2$ | not smooth at $r = 0$: use ricb > 0 |
| `'shellular'` | $a\thinspace (r^\sigma - 1)$, $\sigma = p$ | $P_0$ | Baruteau & Rieutord 2013 ($a = 1$); not smooth at $r = 0$ unless $\sigma$ is even and positive |
| `'shellular_boussinesq'` | $\tfrac12 N_0^2\thinspace (1 - r^2)$, $N_0^2 = a$ | $P_0$ | Mirouh et al. 2016; use `thermal_wind = 0` (their background has no thermal wind), structure `'Boussinesq'` and stratification `'linear'` |
| `'cylindrical'` | $\varepsilon\thinspace r^2\sin^2\theta$, $\varepsilon = a$ | $P_0$, $P_2$ | Baruteau & Rieutord 2013; barotropic, $S_{tw} = 0$ |
| `'conical'` | $\varepsilon\sin^2\theta$, $\varepsilon = a$ | $P_0$, $P_2$ | Guenel et al. 2016 |
| `'solar'` | $a\thinspace \times$ helioseismic fit, $P_0 + P_2$, tanh step at $x = 0.71$ of width 0.1 | $P_0$, $P_2$ | relative to 435 nHz; 4.5 nHz too fast at the equator and 12 nHz at the pole compared with the observations |
| `'solar_P4'` | $a\thinspace (\Omega/\Omega_{\rm ref} - 1)$, $\Omega$ the fit below | $P_0$, $P_2$, $P_4$ | `diff_rot_solar`, `diff_rot_frame`; defaults from Mukhopadhyay et al. 2026 |
| `'tabulated'` | $a\thinspace (\Omega/\Omega_{\rm ref} - 1)$, $\Omega$ from `diff_rot_file` | $P_0 \dots P_{n_{\max}}$ | below |
| `'user'` | $a\sum_n f_n(r) P_n(\cos\theta)$, $f_n$ from `diff_rot_profiles` | the keys of `diff_rot_profiles` | below |

**`'rigid'`.** The eigenvalues are known exactly: with $\lambda_0$ the `diff_rot = 0` eigenvalue,

$$
\lambda(a, G) = \lambda_0\big(G(1 + a)\big) - i\thinspace m G a ,
$$

with $G$ = `Gaspard`.

**`'solar_P4'`.** A rigid radiative zone joined by an erf step at the tachocline $r_{tc}$ (width $d_{tc}$) to a
convection zone whose rotation depends on latitude only:

$$
\frac{\Omega}{2\pi} = \Omega_{RZ} + \tfrac12\left[\Omega_{CZ}(\theta) - \Omega_{RZ}\right]\left[1 + \mathrm{erf}\thinspace \frac{x - r_{tc}}{d_{tc}}\right],
$$

$$
\Omega_{CZ} = c_0 + c_2\cos^2\theta + c_4\cos^4\theta ,
$$

with `diff_rot_solar` = [$c_0$, $c_2$, $c_4$, $\Omega_{RZ}$, $r_{tc}$, $d_{tc}$] (nHz, and fractions of R☉).

**`'user'`.** `diff_rot_profiles` gives the shape of each even degree as a numpy expression ([below](#user-profiles)).
For example, the cylindrical rotation as a `'user'` profile (it gives the same matrices as `'cylindrical'`):

```python
par.diff_rot_type     = 'user'
par.diff_rot_profiles = {0: '(2/3)*r**2', 2: '-(2/3)*r**2'}
par.diff_rot_ampl     = -0.25
```

The degrees must be even. Any degree up to 16 works with `thermal_wind = 2` (32 with `thermal_wind = 1`).

**`'tabulated'`.** The file holds `ome[θ, r]` (nHz), `r` (R☉) and `theta` (rad), the layout of `data/diff_rot.npz`
of the Mukhopadhyay et al. 2025 data set. `tests/diff_rot/ls_to_tabulated.py` converts the Larson &
Schou 2018 table to it; `bin/jsoc_hmi_2010_2026.npz` is the mean of the 83 JSOC HMI 72-day inversions
(`hmi.V_sht_2drls`, NACOEFF 36) from 2010-04-30 to 2026-06-29 in this layout, on the 151 table radii. kore projects
$\Omega$ on $P_0 \dots P_{n_{\max}}$ at each table radius, fits each $f_n$ in $r$ over the table radii within 0.01 of
the shell, and prints the fit residuals and the truncation error in nHz (the largest difference between the table and
its expansion up to $P_{n_{\max}}$); check that line. For the Mukhopadhyay et al. 2025 profile the truncation error near
the poles at the surface is up to 17 nHz at `diff_rot_nmax = 8`; 12 is the working choice.

### Thermal wind

With `thermal = 1`, `thermal_wind` chooses the latitudinal background entropy $S_{tw}(r,\theta)$ and the terms the
differential rotation adds to the heat equation. The advection of $s$ by the rotation is always included.

| `thermal_wind` | heat-equation terms | $S_{tw}$ |
| --- | --- | --- |
| 0 | the advection of $s$ only | none (e.g. Mirouh et al. 2016) |
| 1 (default) | and the advection of $S_{tw}$ by the perturbation | thermal wind of $\Omega$ to first order in $\Delta\Omega$ ($S_{tw} \propto G^2/$`Beyonce`) |
| 2 | the same | thermal wind of the full $(1 + \Delta\Omega)^2$ balance; frame independent |
| 3 | the same | prescribed: $\partial_\theta S_{tw}$ = `dSdtheta` $\sin 2\theta$, uniform in $r$ (Bekki et al. 2024) |

- `thermal_wind` 1 and 2 need `Beyonce > 0`. With `timescale = 'rotation'`, `set_scales()` sets `Beyonce = 0`, so set
  it after calling `set_scales()`.
- `dSdtheta` is in the units of $s$ ($\delta s/c_p$): `dSdtheta` $= \Delta_\theta s/c_p$, with $\Delta_\theta s$ in erg g⁻¹ K⁻¹. Bekki et al. use
  $c_p$ = 4.17e8, Model S has about 3.46e8. Negative values give warm poles. The thermal-wind amplitude does not apply.
- `thermal_wind_radial = 0` drops $u_r\thinspace \partial_r S_{tw}$, as Bekki et al. 2022 and Mukhopadhyay et al. 2025 do. Keeping it is
  the consistent choice, but it moves the solar columnar and high-frequency retrograde modes by 5–15 nHz.
- With the thermal wind $S_{tw}$ is the only latitudinal entropy gradient; in a convection zone with $dS_0/dr = 0$ it is
  also the only background entropy gradient, and its effect on the modes is of the same order as the momentum terms.
- With `diff_rot = 1` the default `thermal_wind = 1` includes the thermal wind. For the advection of $s$ by the
  rotation alone, set `thermal_wind = 0`.

### Resolution and cost

- A differential rotation couples harmonic degrees $l$ that uniform rotation keeps apart, so the matrices get wider.
  Let $n_{\max}$ be the highest degree of $\Delta\Omega$. The momentum rows then couple degrees that differ by up to
  $n_{\max}$ ($P \to P$, $T \to T$) or $n_{\max} + 1$ ($P \leftrightarrow T$).
- $S_{tw}$ has degrees up to $n_{\max} + 2$ (`thermal_wind` 1) or $2n_{\max} + 2$ (`thermal_wind` 2), and the heat rows
  couple degrees that far apart. Matrices are therefore widest with `'tabulated'` and a large `diff_rot_nmax`.
- With `ricb = 0` (a full sphere) the profiles are evaluated at $\lvert r\rvert$, to keep them smooth at the centre.
  `'Y20-wall-bounded'` and `'shellular'` (except even $\sigma > 0$) have a kink at $r = 0$ and `submatrices.py` prints a
  WARNING. The `'solar'` tachocline step needs N ≈ 400 at `ricb = 0`.
- Modes that reach into a radiative interior have short radial wavelengths. In a full sphere with the radiative
  interior of stratification `'solar_rz'`, solar inertial modes needed N = 768 to 1024 for the `spin_doctor`
  residuals, although their eigenvalues converge at a lower N. [Solver](solver.md) has the memory this takes.
- Check convergence with `spin_doctor` as for any run; its residuals include the terms of the differential rotation.

## `'user'` profiles

`stratification_profile`, the two `diffusivity_profiles` and the `diff_rot_profiles` are numpy expressions of `r` (the
kore radius), `x` ($r\cdot$`r_out` with a solar structure, $r$ otherwise) and `p` (`stratification_param`,
`diffusivity_param` or `diff_rot_param`), with `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`,
`tanh`, `exp`, `log`, `log10`, `sqrt`, `abs`, `sign`, `pi`, `where`, `minimum`, `maximum`, `erf`, `erfc`. For example:

```python
par.stratification = 'user'
par.stratification_profile = '2.5*exp(-((r - 0.75)/0.06)**2)'      # dS/dr in 1/L for r >= 0
par.diffusivity = 'user'
par.diffusivity_profiles = {'nu': '0.01 + 0.99*0.5*(1 + tanh((x - 0.70)/0.015))', 'kappa': '1'}
```

kore evaluates them at $\lvert r\rvert$, so in a full sphere $dS/dr$ is odd and $\hat\nu$, $\hat\kappa$ and the $f_n$ are
even whatever the expression; for a profile that is smooth at the centre, $dS/dr$ should still vanish there. The
expressions must not contain `#`: `spin_doctor` writes them at the end of each `params.dat` row, after a `#`.

## Checks

Before anything is built, kore stops on a background that cannot work: an unknown type, a parameter name of an older
version of kore (the message gives the replacement), `'Wu 2005'` without $0 <$ `r_out` $< 1$, `'inverse square'` with
`ricb = 0`, $\delta$ without `'solar_rz'`, or a `'user'` expression that does not evaluate. `submatrices.py` also warns
about any name in the parameter file that kore does not define, so a misspelt name is not ignored without notice.
`spin_doctor` records the background of every solution in `params.dat` ([Output files](output.md)).
