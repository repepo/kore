# Background model

kore computes small perturbations of a steady background state. You describe that background in the parameter file,
in seven independent parts:

| part | parameter | what it sets |
| --- | --- | --- |
| structure | `structure` | density, pressure and gravity as functions of radius |
| stratification | `stratification` | the radial entropy gradient (only with `thermal = 1`) |
| diffusivity | `diffusivity` | how viscosity and thermal diffusivity vary with radius |
| composition | `comp_stratification` | the gradient of a second scalar, the composition (only with `compositional = 1`) |
| rotation | `diff_rot`, `diff_rot_type` | uniform or differential rotation |
| shape | `centrifugal`, `centrifugal_shape` | a sphere, or a body flattened by its rotation |
| flow | `bgflow` | a steady axisymmetric flow, see [Background flow](bgflow.md) |

The equations where these enter are on [Anelastic equations](anelastic.md).

## Units

kore works with dimensionless quantities ([Non-dimensionalization](anelastic.md#non-dimensionalization)). The unit of
length is the radius of the outer wall. The parameter `r_out` (default 1) gives that radius as a fraction of the stellar
radius; the solar profiles below are written in terms of this fractional stellar radius, called $x$.

The helper module `bin/units.py` converts physical values into kore's parameters, using $1/\Omega$ as the time unit.
In a parameter file, after setting `r_out`:

```python
import units
units.set_physical(par, frame_nHz=456.0, nu=1e12, kappa=1e12)   # sets Gaspard, Beyonce, ViscosD, ThermaD, diff_rot_frame
units.set_target(par, omega=-271.2, sigma=-33.2)                 # sets the solver target from a frequency and a damping rate in nHz
```

`set_physical` takes the viscosity and thermal diffusivity in cm²/s, and uses the solar radius and surface gravity
unless you pass `radius` and `gsurf` (in cgs units). Running `python3 bin/units.py` in a run directory prints the
computed eigenvalues as frequency and damping rate in nHz.

## Quick start

The background of the smooth Sun example (`examples/smooth_sun/parameters.py`), a full sphere with the observed solar
rotation:

```python
par.thermal        = 1
par.r_out          = 0.985                 # outer wall in solar radii
par.structure      = 'model S hydrostatic'
par.stratification = 'solar_rz'            # a radiative interior below a slightly subadiabatic convection zone
par.solar_rz_delta_cz = -4e-7
par.diffusivity    = 'solar_rz'            # viscosity and diffusivity drop to visc0, therm0 below 0.70 R☉
par.visc0, par.therm0 = 0.01, 0.04
par.diff_rot       = 1
par.diff_rot_type  = 'tabulated'           # rotation rate read from a table
par.diff_rot_file  = 'bin/jsoc_hmi_2010_2026.npz'
par.diff_rot_nmax  = 4
par.thermal_wind   = 2                     # with its exact thermal wind
par.thermal_wind_radial = 0
units.set_physical(par, frame_nHz=456.0, nu=1e12, kappa=1e12)
```

The default `bin/parameters.py` uses `structure = 'Wu 2005'` and `stratification = 'top hat'` with uniform rotation.
[Getting started](getting-started.md) uses an incompressible fluid (`'Boussinesq'`, `thermal = 0`).

## Structure

`structure` sets the density, pressure and gravity. It has no default.

| `structure` | description | parameters |
| --- | --- | --- |
| `'model S hydrostatic'` | the Sun: a smooth fit to the density of solar Model S, with gravity and pressure consistent with it. Recommended for solar work | `structure_deg` (30) |
| `'Wu 2005'` | a polytrope-like density $\rho_0 = (1 - x^2)^2$ (Wu 2005) | needs `r_out` < 1, so that the density is not zero at the wall |
| `'Boussinesq'` | uniform density: the Boussinesq (incompressible) equations | – |
| `'8-term fit'` | legacy solar fit, kept to reproduce old runs | – |
| `'old model S prototype'` | legacy solar fit, kept to reproduce old runs | `solar_density_deg`, `solar_density_window`, `solar_density_sun_deg` |

- Only the shape of the density and pressure profiles matters, not their overall scale.
- `'model S hydrostatic'` matches Model S to within about 1 % at the default degree 30. Use `structure_deg = 50` to
  compare closely with work that uses Model S itself; this needs a slightly higher `lmax`.

## Stratification

With `thermal = 1`, `stratification` sets the background entropy gradient $dS/dr$, which controls buoyancy. A positive
gradient is stable (radiative), zero is neutral (convective). There is no default.

| `stratification` | profile | parameters |
| --- | --- | --- |
| `'top hat'` | a stable layer between two smooth steps | `aux1`…`aux5`: position and width of each step, then the amplitude |
| `'odd top hat'` | the same, but well behaved at the centre of a full sphere | `aux1`…`aux5` |
| `'solar_rz'` | the solar radiative interior of Mukhopadhyay et al. 2025b (below) | `solar_rz_*` |
| `'linear'` | proportional to $r$ | – |
| `'inverse square'` | proportional to $1/r^2$ (conduction between the walls) | needs `ricb` > 0 |
| `'two zone'` | a nearly neutral inner zone below a stable outer zone | `two_zone` = [transition radius, width] ([0.7, 0.1]) |
| `'user'` | your own expression ([below](#user-profiles)) | `stratification_profile`, `stratification_param` |

- In a full sphere the gradient must go to zero smoothly at the centre. Use `'odd top hat'` rather than `'top hat'`
  unless the first step is far from the centre.
- In a shell without thermal diffusion, keep the gradient zero at the inner wall (see [ThermaD = 0](#thermad-0)).
- `'odd top hat'` with `aux1`…`aux5` = 0, 0.03, 0.68, 0.06, 2.6 approximates the entropy gradient of solar Model S.

**`'solar_rz'`.** A stable radiative interior joined to a convection zone at about 0.7 solar radii, following
Mukhopadhyay et al. (2025b) and Bekki et al. (2024):

$$
\frac{dS}{dr} = \frac{A}{4}\left[1 - \mathrm{erf}\frac{x - x_1}{d}\right]\left[1 - \mathrm{erf}\frac{x - x_2}{d}\right]\mathrm{erf}\frac{x}{w_0}
+ \delta(x)\thinspace \frac{d\ln p_0}{dr}\thinspace \frac{1}{2}\left[1 + \mathrm{erf}\frac{x - x_{OS}}{d_{OS}}\right].
$$

The first term is the radiative interior, the second the convection zone. The parameters and their defaults:

| parameter | meaning | default |
| --- | --- | --- |
| `solar_rz_dSdr` | amplitude $A$ of the radiative gradient | 2.615 |
| `solar_rz_steps` | $[x_1, x_2, d]$, the steps that shape it | [0.61, 0.68, 0.04] |
| `solar_rz_core` | $w_0$, makes the profile smooth at the centre | 0.05 |
| `solar_rz_os` | $[x_{OS}, d_{OS}]$, base and width of the convection zone | [0.70, 0.04] |
| `solar_rz_delta_cz` | superadiabaticity $\delta$ of the convection zone (negative is stable) | 0 |
| `solar_rz_delta_sf` | $\delta$ just below the surface | 0 |

A slightly negative `solar_rz_delta_cz`, such as the $-4\times10^{-7}$ of the smooth Sun example, noticeably changes
the frequencies of the high-frequency retrograde modes.

## Diffusivity

The viscosity is `ViscosD` times a radial profile, and the thermal diffusivity is `ThermaD` times another.
`diffusivity` chooses those profiles (and the composition's, with `comp0` in place of `therm0`).

| `diffusivity` | profile | parameters |
| --- | --- | --- |
| `'uniform'` (default) | constant | – |
| `'step'` | `visc0`, `therm0` below radius `aux3`, 1 above, joined by a smooth step of width `aux4` | `visc0`, `therm0`, `aux3`, `aux4` |
| `'solar_rz'` | `visc0`, `therm0` in the radiative interior, 1 in the convection zone (Mukhopadhyay et al. 2025b) | `visc0`, `therm0`, `solar_rz_os` |
| `'user'` | your own expressions ([below](#user-profiles)) | `diffusivity_profiles`, `diffusivity_param` |

- With `'step'` and `'solar_rz'`, `ViscosD` and `ThermaD` are the values in the outer (convective) layer.
- `visc_pow`, `therm_pow` and `comp_pow` (default 0) multiply the profile by a power of the density. With 1, the
  dynamic diffusivities (density times diffusivity) follow the profile instead of the kinematic ones.
- [Convective forcing](forcing.md) uses only the turbulent part of the viscosity, above a floor `rsf_nu_floor`.

### ThermaD = 0

`ThermaD = 0` turns thermal diffusion off: the motion is adiabatic and the thermal boundary conditions are not used.
It works with every structure, stratification and rotation profile. Things to know:

- **Prefer `ThermaD = 0` to a very small `ThermaD`.** A very small diffusivity creates thin thermal boundary layers at
  the walls that need much more radial resolution. The eigenvalues approach the `ThermaD = 0` ones smoothly anyway.
- **Keep the entropy gradient zero at the inner wall.** Otherwise spurious modes appear there. They move when you change
  `N`, and `spin_doctor` flags them. If the gradient must reach the wall, ask for one or two more eigenvalues (`nev`),
  since the spurious modes take some of the slots.
- **With differential rotation, expect many spurious eigenvalues with near-zero damping.** They come from the entropy
  being carried by the rotation without diffusion. They change with `lmax` and fail the ℓ-convergence check of
  `spin_doctor`; discard them. Keep the solver target away from that band.
- **Check convergence in `N`.** Runs without diffusion usually need more radial resolution.
- **Forced runs.** Without diffusion, the entropy is only carried by the flow. If the forcing frequency, seen from the
  rotating fluid, is zero somewhere in the domain, the problem has no well-behaved solution and kore stops
  ([Combining features](compatibility.md#forced-runs)).

## Composition

A fluid can be stratified by composition as well as by entropy: helium below the convection zone of a star, or light
elements in a planet's core. With `compositional = 1` kore adds a second scalar, the composition, that is carried by
the flow, diffuses, and adds its own buoyancy. It can be used alone (`thermal = 0`) or together with the entropy, for a
double-diffusive fluid. Its equation is on [Anelastic equations](anelastic.md#composition).

```python
par.compositional = 1
par.comp_stratification = 'linear'      # background gradient proportional to r
par.BeyonceC = 10.                      # strength of its buoyancy; set after set_scales()
par.CompoD = 1e-3                       # its diffusivity
par.bci_compositional = 0               # at each wall, 0: fixed composition, 1: no flux
par.bco_compositional = 0
```

`comp_stratification` takes the values `'linear'`, `'inverse square'`, `'two zone'` (with `comp_two_zone`) and `'user'`
(with `comp_stratification_profile`), as for the entropy. It has no default. `BeyonceC` sets the strength of the
compositional buoyancy.

- A higher composition is lighter, as for the entropy: the layer is stable where `BeyonceC` times the gradient is positive.
- `CompoD = 0` turns its diffusion off; the advice of [ThermaD = 0](#thermad-0) then applies to it.
- `units.set_physical(par, ..., D=...)`, with $D$ in cm²/s, also sets `CompoD` and `BeyonceC`.
- The composition works with nearly every other feature. The tidal forcing 1 needs `compositional = 0`
  ([Combining features](compatibility.md)).
- Its results are in `composition.dat` ([Output files](output.md#compositiondat)).

## Rotation

With `diff_rot = 0` (the default) the fluid rotates uniformly with the frame. With `diff_rot = 1` it rotates
differentially, at

$$
\Omega(r,\theta) = \Omega_{\rm ref}\left[1 + \Delta\Omega(r,\theta)\right],
$$

where $\Omega_{\rm ref}$ is the rotation rate of the frame and $\Delta\Omega$ the relative departure from it. kore
expands $\Delta\Omega$ in even Legendre polynomials in latitude, so the rotation is always symmetric about the equator.
Eigenvalues are given in the frame rotating at $\Omega_{\rm ref}$.

With `thermal = 1`, the background also includes the latitudinal entropy variation that balances the differential
rotation, the *thermal wind* (see [below](#thermal-wind)).

### Parameters

| parameter | default | meaning |
| --- | --- | --- |
| `diff_rot` | 0 | 1 turns on differential rotation |
| `diff_rot_type` | `'Y20'` | the profile, see below |
| `diff_rot_ampl` | 1.0 | a factor that multiplies $\Delta\Omega$ |
| `diff_rot_param` | 0.0 | shape parameter of some profiles |
| `diff_rot_term_amp` | `[1.0]*9` | factors on each Legendre term; e.g. `[1, 1, 0]` drops the third term |
| `diff_rot_profiles` | `{}` | `'user'`: one expression per Legendre degree (below) |
| `diff_rot_frame` | 456.0 | `'solar_P4'`, `'tabulated'`: frame rotation rate in nHz |
| `diff_rot_solar` | `[464, -76, -50, 433.5, 0.71, 0.045]` | `'solar_P4'`: coefficients of the fit below |
| `diff_rot_file` | `''` | `'tabulated'`: the table file, absolute or relative to the run directory |
| `diff_rot_nmax` | 8 | `'tabulated'`, `'ferraro'`: highest Legendre degree kept (even) |
| `diff_rot_F` | `'psi'` | `'ferraro'`: the rotation as a function of the magnetic flux function |
| `diff_rot_rdeg`, `diff_rot_window`, `diff_rot_sun_deg` | 24, None, 60 | `'tabulated'`: settings of the radial fit |
| `thermal_wind` | 1 | the thermal wind, see below |
| `thermal_wind_radial` | 1 | 0 drops the radial advection of the thermal-wind entropy |
| `thermal_wind_amplitude` | 1.0 | a factor on the thermal-wind entropy |
| `dSdtheta` | 0.0 | `thermal_wind = 3`: the prescribed latitudinal entropy gradient |

If `diff_rot` is used with physical units (`'solar'`, `'solar_P4'`, `'tabulated'`), compute `ViscosD`, `ThermaD` and
`Beyonce` for the frame rate: 435 nHz for `'solar'`, `diff_rot_frame` for the other two. `units.set_physical` does this.

### Profiles

Below, $a$ is `diff_rot_ampl` and $p$ is `diff_rot_param`.

| `diff_rot_type` | rotation | notes |
| --- | --- | --- |
| `'rigid'` | uniform offset $a$ | for tests: the exact answer is known |
| `'Y20'` | $a\thinspace P_2(\cos\theta)$, latitudinal only | use `ricb` > 0 |
| `'Y20-wall-bounded'` | the same, vanishing at both walls | use `ricb` > 0 |
| `'shellular'` | $a\thinspace (r^p - 1)$, radial only | Baruteau & Rieutord 2013 |
| `'shellular_boussinesq'` | radial only, proportional to $1 - r^2$ | Mirouh et al. 2016; use with `thermal_wind = 0`, `'Boussinesq'` and `'linear'` |
| `'cylindrical'` | $a\thinspace r^2\sin^2\theta$, constant on cylinders | Baruteau & Rieutord 2013 |
| `'conical'` | $a\sin^2\theta$, constant on cones | Guenel et al. 2016; use `ricb` > 0 |
| `'solar'` | an older helioseismic fit, relative to 435 nHz | less accurate than `'solar_P4'` or `'tabulated'` |
| `'solar_P4'` | a fit of the solar rotation, see below | Mukhopadhyay et al. 2026 |
| `'tabulated'` | read from a table of observations | see below |
| `'user'` | your own expressions | see below |
| `'ferraro'` | constant along the field lines of a poloidal magnetic field | needs `magnetic = 1` ([Magnetic field](magnetic.md#with-a-differential-rotation)) |

**`'solar_P4'`.** A uniformly rotating radiative zone, joined at the tachocline (radius $r_{tc}$, width $d_{tc}$) to a
convection zone whose rotation depends only on latitude:

$$
\frac{\Omega}{2\pi} = \Omega_{RZ} + \tfrac12\left[\Omega_{CZ}(\theta) - \Omega_{RZ}\right]\left[1 + \mathrm{erf}\thinspace \frac{x - r_{tc}}{d_{tc}}\right],
\qquad
\Omega_{CZ} = c_0 + c_2\cos^2\theta + c_4\cos^4\theta ,
$$

with `diff_rot_solar` = [$c_0$, $c_2$, $c_4$, $\Omega_{RZ}$, $r_{tc}$, $d_{tc}$], rates in nHz and radii in solar radii.

**`'tabulated'`.** The rotation rate comes from an `.npz` file with the arrays `ome[θ, r]` (nHz), `r` (solar radii)
and `theta` (radians). `bin/jsoc_hmi_2010_2026.npz` is the mean solar rotation measured by HMI from 2010 to 2026, and
`tests/diff_rot/ls_to_tabulated.py` converts the Larson & Schou (2018) table to this format. kore expands the table up
to Legendre degree `diff_rot_nmax` and prints how well the expansion fits the table: check that line. Near the
surface at high latitude the solar profile needs a degree of about 12 for good accuracy.

**`'user'`.** `diff_rot_profiles` gives the radial shape of each even Legendre degree as an expression
([below](#user-profiles)). For example, the cylindrical profile written by hand:

```python
par.diff_rot_type     = 'user'
par.diff_rot_profiles = {0: '(2/3)*r**2', 2: '-(2/3)*r**2'}
par.diff_rot_ampl     = -0.25
```

### Thermal wind

In a stratified fluid, differential rotation must be balanced by a latitudinal variation of the background entropy, the
thermal wind. With `thermal = 1`, `thermal_wind` chooses how kore treats it:

| `thermal_wind` | background entropy |
| --- | --- |
| 0 | no thermal wind; the rotation only carries the entropy perturbation (as in Mirouh et al. 2016) |
| 1 (default) | the thermal wind to first order in the differential rotation |
| 2 | the exact thermal wind; the eigenvalues then do not depend on the choice of frame |
| 3 | a prescribed latitudinal entropy gradient, `dSdtheta` (as in Bekki et al. 2024) |

- Options 1 and 2 need a non-zero `Beyonce`. `set_scales()` with `timescale = 'rotation'` sets `Beyonce = 0`, so set it
  after calling `set_scales()`.
- `thermal_wind_radial = 0` drops one term of the thermal wind, as Bekki et al. 2022 and Mukhopadhyay et al. 2025 do.
  Keeping it is more consistent, but use 0 to compare with those papers.
- With a magnetic field, the thermal wind ignores the force of the background field ([Magnetic field](magnetic.md)).

### Resolution and cost

- Differential rotation couples spherical-harmonic degrees that uniform rotation keeps separate, so the matrices are
  larger and the runs slower. The cost grows with the highest Legendre degree of the rotation, so keep
  `diff_rot_nmax` as low as the problem allows.
- In a full sphere, some profiles are not smooth at the centre (`'Y20'`, `'conical'`, most `'shellular'`).
  `submatrices.py` prints a warning for each.
- Modes that reach into a radiative interior have short radial wavelengths and can need `N` of 700 or more. See
  [Solver](solver.md) for the memory this takes.
- Check every solution with `spin_doctor`, as for any run.

## Centrifugal deformation

A rotating star or planet is flattened at the poles. With `centrifugal = 1`, kore gives the background this shape: the
walls and the surfaces of constant density become oblate spheroids, and every term of the equations feels it. kore
keeps its spherical grid and solves in coordinates that follow the flattened surfaces. The equations are on
[Centrifugal deformation](centrifugal-equations.md).

The deformation is accurate to first order in the flattening, which is enough for most stars and planets.
`'spheroid'` treats a uniform-density body exactly, at any flattening.

### Flattening

| parameter | default | meaning |
| --- | --- | --- |
| `centrifugal` | 0 | 1 flattens the body |
| `centrifugal_shape` | `'clairaut'` | how the flattening varies with radius (below) |
| `centrifugal_q` | 0.0 | the ratio of centrifugal to gravitational acceleration at the equator |
| `centrifugal_f` | None | the flattening of the outer wall, (equatorial − polar radius) / equatorial radius |
| `centrifugal_profile`, `centrifugal_param` | | `'user'`: the flattening as an expression ([below](#user-profiles)) |

| `centrifugal_shape` | flattening | set by |
| --- | --- | --- |
| `'clairaut'` | the hydrostatic shape of the chosen structure | `centrifugal_q` or `centrifugal_f` |
| `'uniform'` | the same at every radius | `centrifugal_f` |
| `'user'` | any profile | `centrifugal_profile` |
| `'spheroid'` | the same at every radius, exact at any flattening | `centrifugal_f` |

- `'spheroid'` suits a uniform-density fluid in a fast-rotating or rigid container, such as a planetary core inside a
  rigid mantle. It needs `'Boussinesq'`, uniform diffusivities and no differential rotation. A negative flattening
  gives an elongated body.
- `submatrices.py` prints the flattening at the walls, and warns above 0.1, where the first-order treatment becomes
  inaccurate.

Typical values:

| body | `centrifugal_q` | surface flattening |
| --- | --- | --- |
| Earth | 3.5e-3 | 3.4e-3 (core-mantle boundary: 2.5e-3) |
| Jupiter | 0.089 | 0.065 |
| Saturn | 0.155 | 0.098 |
| Sun | 2e-5 | 8e-6 |

A Jupiter-like body:

```python
par.structure         = 'Wu 2005'
par.r_out             = 0.99
par.centrifugal       = 1
par.centrifugal_shape = 'clairaut'
par.centrifugal_q     = 0.089
```

The Earth's core, whose shape is set by the whole planet, so give the flattening of the core-mantle boundary directly:

```python
par.structure         = 'Boussinesq'
par.centrifugal       = 1
par.centrifugal_shape = 'uniform'
par.centrifugal_f     = 2.5e-3
```

### Limitations

- The deformation works with buoyancy, viscosity, thermal and compositional diffusion, differential rotation and a
  magnetic field (insulating walls or a conducting inner core), in shells and full spheres.
- It does not work with forced runs, walls of zero radial stress (`bci` or `bco` = 2), the rotating solids, the drag at
  the outer wall, thin conducting layers, a resolved mantle or a background flow. kore stops and says why
  ([Combining features](compatibility.md)).
- A viscous flow in a full sphere with `m` ≠ 0 needs `rpower_v` of at least 4; kore stops and says so.
- With the thermal wind, the background entropy ignores the flattening. `submatrices.py` warns about this.
- In `spin_doctor`, the total energy budget closes only approximately (to second order in the flattening), except with
  `'spheroid'`.

## User profiles

The `'user'` options take a numpy expression as a string. It may use `r` (kore's radius), `x` (the stellar radius with
a solar structure, otherwise equal to `r`), `p` (the matching `..._param` parameter), and the usual functions: `sin`,
`cos`, `tan`, `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`, `tanh`, `exp`, `log`, `log10`, `sqrt`, `abs`, `sign`,
`pi`, `where`, `minimum`, `maximum`, `erf`, `erfc`. For example:

```python
par.stratification = 'user'
par.stratification_profile = '2.5*exp(-((r - 0.75)/0.06)**2)'
par.diffusivity = 'user'
par.diffusivity_profiles = {'nu': '0.01 + 0.99*0.5*(1 + tanh((x - 0.70)/0.015))', 'kappa': '1'}
```

In a full sphere, profiles must be smooth at the centre; `submatrices.py` warns if they are not
([Regularity at the centre](numerical-method.md#regularity-at-the-centre)). Expressions must not contain `#`.

## Checks

kore checks the background before building anything, and stops with an explanation if something cannot work: an
unknown type, a parameter name from an older version of kore (the message gives the new name), or a combination that is
not supported. `submatrices.py` also warns about any parameter name it does not recognize, so a misspelled name is not
silently ignored.
