# Background model

kore solves for small perturbations of a background in which the density $\rho_0$, the pressure (or $p_0 = \rho_0 T_0$)
and gravity $g$ depend on $r$ only, the entropy has a radial gradient $dS/dr$, and the viscosity $\nu$ and the thermal
diffusivity $\kappa$ are radial profiles. The background has three parts, each chosen by name in the parameter file:

| part | parameter | what it sets |
| --- | --- | --- |
| structure | `structure` | $\rho_0$, $p_0$ and $g$ |
| stratification | `stratification` | $dS/dr$, with $S = \delta s/c_p$ (only with `thermal = 1`) |
| diffusivity | `diffusivity` | the radial profiles of $\nu$ and $\kappa$ |

The rotation of the background, uniform or differential, is chosen separately ([Differential rotation](differential-rotation.md)).
The code is in `bin/background.py` (the choice, the checks, the stratification and diffusivity types) and
`bin/radial_profiles.py` (one class per structure).

## Units

kore is dimensionless; [Non-dimensionalization](nondimensionalization.md) gives the scales and the parameters that carry
them. The unit of length is the radius of the outer wall, $r = 1$. `r_out` is that radius in star radii (1 by
default), so $x = r\cdot$`r_out` is the radius in star radii, the coordinate of the solar profiles below. The only
physical unit kore reads is that of the solar rotation data, in nHz. `bin/units.py` sets the parameters from physical
values ([Physical values](nondimensionalization.md#physical-values)).

## Quick start

The background of the smooth Sun example (`examples/smooth_sun/parameters.py`):

```python
par.thermal        = 1
par.r_out          = 0.985                 # outer wall in solar radii
par.structure      = 'model S hydrostatic'
par.stratification = 'solar_rz'            # a radiative interior below a slightly subadiabatic convection zone
par.solar_rz_delta_cz = -4e-7
par.diffusivity    = 'solar_rz'            # ν, κ drop to visc0, therm0 below 0.70 R☉
par.visc0, par.therm0 = 0.01, 0.04
units.set_physical(par, frame_nHz=456.0, nu=1e12, kappa=1e12)
```

The checked-in `bin/parameters.py`, a gravito-inertial mode, uses `structure = 'Wu 2005'` with `r_out = 0.99` and
`stratification = 'top hat'`.

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
| `'user'` | `stratification_profile`, a numpy expression (below) | `stratification_param` ($p$) |

- `'top hat'` in a full sphere needs $x_1 \gtrsim 6 w_1$, so that $dS/dr \to 0$ at the centre; otherwise use
  `'odd top hat'`. With `ricb` > 0 and `ThermaD = 0`, a $dS/dr \neq 0$ at the inner wall gives spurious modes trapped
  there.
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
| `'user'` | `diffusivity_profiles = {'nu': '...', 'kappa': '...'}`, numpy expressions (below) | `diffusivity_param` ($p$) |

- With `'step'` and `'solar_rz'`, `ViscosD` and `ThermaD` are the values above the step (the convection zone of the
  Sun). Mukhopadhyay et al. 2025b use `visc0` = 0.01 and `therm0` = 0.04.
- Any type is multiplied by $(\rho_0(x_2)/\rho_0)^p$, $x_2$ = `aux3`, with $p$ = `visc_pow` for $\hat\nu$ and `therm_pow`
  for $\hat\kappa$: 0 (the default) keeps the kinematic diffusivities as set, 1 makes $\rho_0\nu$ and $\rho_0\kappa$
  follow the profile instead.
- Forcing 11 (random Reynolds stresses) uses the turbulent part of the viscosity, `ViscosD` $(\hat\nu - \nu_f)$: the
  floor $\nu_f$ is `visc0` with `'step'` and `'solar_rz'` when `visc0` < 1 and 0 otherwise, unless `rsf_nu_floor` is
  set.

## `'user'` profiles

`stratification_profile` and the two `diffusivity_profiles` are numpy expressions of `r` (the kore radius), `x` (as in
the rotation profiles: $r\cdot$`r_out` with a solar structure, $r$ otherwise) and `p`, with the functions listed for
`diff_rot_profiles` in [differential-rotation.md](differential-rotation.md) (`exp`, `tanh`, `erf`, `abs`, `where`, …).
For example:

```python
par.stratification = 'user'
par.stratification_profile = '2.5*exp(-((r - 0.75)/0.06)**2)'      # dS/dr in 1/L for r >= 0
par.diffusivity = 'user'
par.diffusivity_profiles = {'nu': '0.01 + 0.99*0.5*(1 + tanh((x - 0.70)/0.015))', 'kappa': '1'}
```

kore evaluates them at $\lvert r\rvert$, so in a full sphere $dS/dr$ is odd and $\hat\nu$, $\hat\kappa$ are even
whatever the expression; for a profile that is smooth at the centre, $dS/dr$ should still vanish there. The expressions
must not contain `#`: `spin_doctor` writes them at the end of each `params.dat` row, after a `#`.

## Checks and output

Before anything is built, kore stops on a background that cannot work: an unknown type, a parameter name of an older
version of kore (the message gives the replacement), `'Wu 2005'` without $0 <$ `r_out` $< 1$, `'inverse square'` with
`ricb = 0`, $\delta$ without `'solar_rz'`, or a `'user'` expression that does not evaluate. `submatrices.py` also warns about any name in the parameter file that kore
does not define, so a misspelt name is not ignored without notice.

`spin_doctor` writes the background to `params.dat` (the full list of columns is in
[Differential rotation, Output](differential-rotation.md#output)). The columns, counted from 1:

- 19 `r_out`, 20–24 `aux1`…`aux5`, 25 `visc0`, 34 `therm0`, 35 `visc_pow`, 36 `therm_pow`;
- 63 `structure`, as its index in `background.STRUCTURES`: `'8-term fit'` 0, `'old model S prototype'` 1,
  `'model S hydrostatic'` 2, `'Wu 2005'` 3, `'Boussinesq'` 4;
- 64 the degree of the Model S fit (`structure_deg` for `'model S hydrostatic'`), 65–66 the window of the
  `'old model S prototype'` fits (nan: the default; `'sun'`: ∓(`r_out` + 0.005)), 67 `solar_rz` (1 when the
  stratification is `'solar_rz'`), 68–74 its parameters, 77–78 $\delta_{cz}$, $\delta_{sf}$;
- 81 `stratification` and 82 `diffusivity`, as their index in `background.STRATIFICATIONS` (`'top hat'` 0,
  `'odd top hat'` 1, `'solar_rz'` 2, `'linear'` 3, `'inverse square'` 4, `'user'` 5; −1 without a stratification)
  and `background.DIFFUSIVITIES` (`'uniform'` 0, `'step'` 1, `'solar_rz'` 2, `'user'` 3);
- the `'user'` expressions, at the end of the row after a `#` that `np.loadtxt` skips (a row with nothing to note has no
  `#`).
