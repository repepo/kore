# Magnetic field

With `magnetic = 1`, kore adds an imposed background magnetic field to the problem. It solves for a small perturbation
of the field together with the flow: the induction equation for the perturbation, and the Lorentz force in the
momentum equation. The background field is steady and axisymmetric. It is given, not computed: there is no dynamo, and
the background flow does not react to the field. The equations and their derivation are on
[Magnetic field equations](magnetic-equations.md).

This page shows how to set up a run and how to choose the background field.

## Parameters and a first run

Two numbers set the strength of the field and of its diffusion:

| parameter | meaning |
| --- | --- |
| `Le` | Lehnert number: the Alfvén speed of the field over the rotation speed of the outer wall |
| `Em` | magnetic Ekman number: the magnetic diffusivity over the rotation rate times the radius squared |

Both use the rotation rate, as `Ek` does. Two ratios are common: the magnetic Prandtl number `Ek/Em` and the Elsasser
number `Le**2/Em`. The exact definitions are on [Magnetic field equations](magnetic-equations.md#equations).

An example: a uniform field along the rotation axis, in a shell with insulating walls, with Elsasser and magnetic
Prandtl numbers both equal to one. Add these lines to a `bin/parameters.py` like the checked-in one (which already
imports numpy and sets `m`, `symm`, `N`, `lmax` and the target), after its call to `set_scales()`:

```python
par.Ek = 1e-3                # Ekman number
par.Em = 1e-3                # magnetic Ekman number: Pm = Ek/Em = 1
par.Le = np.sqrt(1e-3)       # Lehnert number: Elsasser number Le**2/Em = 1
par.set_scales()             # call it again: it sets Hendrik and MagnetD from Le and Em

par.structure = 'Boussinesq' # any structure works (Background model)
par.magnetic = 1
par.B0 = 'axial'             # the poloidal background field (tables below)
par.cnorm = 'mag_energy'     # its normalisation: unit magnetic energy in the fluid
par.B0_tor = None            # no toroidal background field
par.innercore = 'insulator'
par.mantle = 'insulator'
par.ricb = 0.35; par.bci = 1; par.bco = 1   # no-slip walls
```

`set_scales()` turns `Le` and `Em` into the factors `Hendrik` (Lorentz force) and `MagnetD` (magnetic diffusion), so it
must run after they are set. Besides `'rotation'`, the time scales `'Alfven'` (the Alfvén time) and `'magnetic'` (the
magnetic diffusion time) are available. A run stops if `Le` or `Em` is zero.

## Choosing the background field

The background field has a poloidal part (`B0`, with field lines in meridional planes) and a toroidal part (`B0_tor`,
azimuthal). `B0` defaults to `'axial'` and `B0_tor` to `None`: for a toroidal field alone, set `B0 = None`.

### Equatorial symmetry

A poloidal field of odd degree is *dipolar* (antisymmetric about the equator); one of even degree is *quadrupolar*
(symmetric). For a toroidal field it is the other way round: a toroidal field of degree 1 is symmetric.

- The two parts can be combined only if they have the same symmetry; the run stops otherwise. So the dipole combines
  with a toroidal field of degree 2, not of degree 1.
- `symm` still selects the symmetry of the flow. With a symmetric background field the perturbation field has the flow's
  symmetry; with an antisymmetric one, the opposite.

### Poloidal field: `B0`

Each poloidal option is one radial shape (or several, for `'general'`), times `cnorm`. The formulas are on
[Magnetic field equations](magnetic-equations.md#poloidal-fields).

| `B0` | the field | extra parameters (default) | geometry | degree |
| --- | --- | --- | --- | --- |
| `'axial'` (default) | uniform field along the rotation axis | none | shell or full sphere | 1 |
| `'FDM'` | the slowest free decay mode of the given degree, for insulating walls | `B0_l` (1); `beta`, only the first guess of the root finder | shell or full sphere | `B0_l` = 1 or 2 |
| `'G21 dipole'` | a full-sphere dipole (Gerick et al. 2021) | `B0_ic` | shell or full sphere | 1 |
| `'Luo_S1'` | the shape of `'G21 dipole'` (Luo & Jackson 2022), at a larger scale | `B0_ic` | shell or full sphere | 1 |
| `'Luo_S2'` | a full-sphere quadrupole (Luo & Jackson 2022) | `B0_ic` | shell or full sphere | 2 |
| `'confined'` | zero outside a spherical layer, going smoothly to zero at its edges | `B0_l` (1), `B0_conf_r1` (0.0), `B0_conf_r2` (0.6), `B0_conf_p` (12) | the layer must lie in the fluid | `B0_l` = 1 or 2 |
| `'general'` | any sum of degrees, one expression per degree | `B0_profiles` = {degree: 'expression of r, p'}, `B0_param` (p) | shell or full sphere | any, of one parity |
| `'ferraro'` | field lines along the surfaces of constant rotation ([below](#with-a-differential-rotation)) | `B0_ferraro_G` ('w'), `B0_ferraro_nmax` (15, odd), `B0_param` | needs `diff_rot = 1` | odd degrees up to `B0_ferraro_nmax` |
| `None` | no poloidal field | | | |

Notes:

- `'G21 dipole'`, `'Luo_S1'` and `'Luo_S2'` are defined for a full sphere. With an inner core, `B0_ic = 'potential'`
  (default) adds a potential field so that the field continues into the core; `B0_ic = 'full sphere'` keeps the
  full-sphere formula.
- `'Luo_S1'` and `'G21 dipole'` give the same run when `cnorm` is a name (`'mag_energy'`, `'Schmitt2012'`, `'rms_cmb'`). With a
  numerical `cnorm` the `'Luo_S1'` field is 30 times stronger.
- `'confined'`: the layer runs from `B0_conf_r1` to `B0_conf_r2`, inside the outer wall. The defaults reach the centre,
  which suits a full sphere; with an inner core, set `B0_conf_r1` at or above `ricb`. `B0_conf_p` (at least 3) is how
  smoothly the field vanishes at the edges, so that no current sheet forms there.
- `'general'`: the expressions are numpy expressions of `r` and `p` (`sin`, `cos`, `exp`, `sqrt`, `where`, `erf`, `pi`, ...).
  In a full sphere the profile of degree n should go as `r**n` at the centre. Example:

    ```python
    par.B0_profiles = {1: 'r/6 - r**3/10', 3: '0.3*r**3*(1 - r**2)'}
    ```

- `'ferraro'`: `B0_ferraro_G` is a numpy expression of `w` (the differential rotation) and `p`.

### Toroidal field: `B0_tor`

Each toroidal option is a purely azimuthal field: one radial shape times `cnorm_tor`. Formulas:
[Magnetic field equations](magnetic-equations.md#toroidal-fields).

| `B0_tor` | the field | extra parameters (default) | geometry | degree |
| --- | --- | --- | --- | --- |
| `None` (default) | no toroidal field | | | |
| `'Malkus'` | grows linearly with the distance to the axis (degree 1), the field of Malkus (1967) | `B0_tor_l` (1) | shell or full sphere; degree 2 shell only | `B0_tor_l` = 1 or 2 |
| `'wall'` | zero at both walls | `B0_tor_l` (1) | shell or full sphere | `B0_tor_l` = 1 or 2 |
| `'user'` | a radial shape given as an expression of `r` and `p` | `B0_tor_profile`, `B0_tor_param` (p), `B0_tor_l` (1) | shell or full sphere | `B0_tor_l` = 1 or 2 |
| `'general'` | any sum of degrees | `B0_tor_profiles` = {degree: 'expression of r, p'}, `B0_tor_param` (p) | shell or full sphere | any, of one parity |

A `'user'` profile must be smooth, since its second derivative enters the equations. It is written to `params.dat`
after `#`; the `'general'` and `'ferraro'` expressions are not.

### Amplitude

The field in the equations is `Le` times a dimensionless field, and the Lorentz force goes as `Le` squared. The size
of the dimensionless field is set by `cnorm` (poloidal) and `cnorm_tor` (toroidal):

| value | meaning | for |
| --- | --- | --- |
| a number (default 1) | multiplies the formula of the shape as it is | all types |
| `'mag_energy'` | unit magnetic energy in the fluid (the integral of the squared field is 1) | all types |
| `'Schmitt2012'` | the integral of the squared field is 4 | all types |
| `'rms_cmb'` | the rms radial field over the outer wall is 1 | `cnorm` with `'axial'`, `'FDM'`, `'G21 dipole'`, `'Luo_S1'`, `'Luo_S2'` only |

With a number, the field's size depends on the arbitrary scale of the formula, so `Le` alone does not say how strong
the field is. Use `'mag_energy'` (as in the example) so that `Le` and the Elsasser number refer to a field of unit
energy. `'confined'`, `'general'` and `'ferraro'` stop with `'rms_cmb'`. Formulas:
[Magnetic field equations](magnetic-equations.md#normalisation).

## With a differential rotation

With `diff_rot = 1` the fluid rotates differentially, with a profile set by `diff_rot_type`
([Background model](background.md#rotation)). The background flow then shears the perturbation field, and kore adds that
induction term for any rotation profile.

**The Ferraro condition.** The rotation also shears the background field itself: it stretches poloidal field lines into
an azimuthal field, a process called *winding*. The background is steady only if the rotation rate is constant along
each poloidal field line (Ferraro's law of isorotation). A toroidal field always satisfies this; a poloidal one usually
does not. kore does not include the winding of the background field in the linear problem: if the condition fails, the
background is assumed to be held steady by something outside the model.

With a poloidal field (and `B0_wound = 0`), kore prints at the start of the run how far the field and the rotation are
from this condition, as a relative number ([definition](magnetic-equations.md#the-ferraro-condition)). Above `1e-7` it
prints a warning. To make it stop instead, set `ferraro_policy = 'refuse'` (default `'warn'`).

**Steady backgrounds.** Four ways to have a poloidal field and a differential rotation that are consistent:

| route | set | use it when |
| --- | --- | --- |
| field from the rotation | `B0 = 'ferraro'`, `B0_ferraro_G` | the rotation is given, and any field that follows it will do |
| rotation from the field | `diff_rot_type = 'ferraro'`, `diff_rot_F` ('psi'), `diff_rot_param`, `diff_rot_nmax` (8, even) | the field is given (any poloidal type, e.g. a model of an observed or a dynamo field), and the rotation only has to fit it |
| wound toroidal field | `B0_wound = 1`, `B0_wound_eta` (1.0) | both are given and do not match (e.g. a tachocline) |
| confined field | `B0 = 'confined'` inside a layer where the rotation is uniform | the field lives where there is no shear |

- `B0 = 'ferraro'` builds the poloidal field from the rotation: its flux function is a function `G` of the rotation, given
  by `B0_ferraro_G`. `G` should vanish on the values the rotation takes on the axis and, for a field kept inside the
  fluid, at the outer wall; kore prints how well these hold. It accepts a toroidal part of even degree. `cnorm` must be a
  number, `'mag_energy'` or `'Schmitt2012'`.
- `diff_rot_type = 'ferraro'` builds the rotation from the field: the rotation is a function `F` of the field's flux
  function `psi`, given by `diff_rot_F` (an expression of `psi` and `p` = `diff_rot_param`). It needs a poloidal field and
  cannot be combined with `B0 = 'ferraro'`. kore's rotation is symmetric about the equator, so with a quadrupolar field
  `F` must be even (e.g. a function of `psi**2`); the run stops otherwise.
- `B0_wound = 1` keeps both and adds the *wound field*: the steady toroidal field at which the winding is balanced by
  magnetic diffusion. It needs a poloidal `B0` and `diff_rot = 1`. `B0_wound_eta` (positive) is the diffusivity of that
  balance in units of `MagnetD`; a turbulent value above 1 gives a weaker wound field, while the perturbation still
  diffuses with `MagnetD`. The wound field is added to any `B0_tor`. Its strength is set by the physics: it grows with
  the shear and falls with the diffusivity, and can exceed the poloidal field many times. `cnorm` normalises the poloidal
  part alone. The run prints the residual of the steady balance and the degrees used, instead of the Ferraro check. Read
  the [caveats](#validity-and-caveats) before using it.

A conducting inner core must co-rotate rigidly with the fluid at its surface: the rotation at `ricb` may not depend on
latitude, or the run stops. Details of each route: [Magnetic field
equations](magnetic-equations.md#with-a-differential-rotation).

## Inner core and walls

| setting | values (default) | notes |
| --- | --- | --- |
| `mantle` | `'insulator'` | the only option: the field matches a potential field outside |
| `innercore` | `'insulator'` (default) or `'conducting, Chebys'` | conducting: needs `ricb > 0` and a no-slip inner wall (`bci = 1`) |
| `N_cic` | 64 | Chebyshev polynomials of the inner-core field (even, at least 4) |
| `mu_i2o` | 1.0 | magnetic permeability, inner core over fluid |
| `sigma_i2o` | 1.0 | electrical conductivity, inner core over fluid |
| `B0_ic` | `'potential'` | for `'G21 dipole'`, `'Luo_S1'`, `'Luo_S2'` with an inner core (above) |

The magnetic diffusivity of the inner core is `MagnetD` divided by `mu_i2o` and `sigma_i2o`. Matching conditions:
[Magnetic field equations](magnetic-equations.md#boundary-conditions).

## What is supported

Supported: a shell or a full sphere (`ricb = 0`); any `structure`; eigenvalue problems; with or without a differential
rotation; `ViscosD > 0`, or `ViscosD = 0` in a full sphere. The magnetic diffusivity is uniform.

Not supported (the run stops at startup):

- no background field at all (`B0 = None` and `B0_tor = None`), or `Le` or `Em` zero;
- forced runs (`forcing > 0`);
- `ViscosD = 0` with an inner core (`ricb > 0`);
- a `mantle` other than `'insulator'`;
- a conducting inner core in a full sphere, or without a no-slip inner wall (`bci = 1`), or with a rotation that varies
  with latitude at its surface;
- a poloidal and a toroidal part of opposite equatorial symmetry;
- the per-option limits of the tables above: degrees other than 1 or 2 for `'FDM'`, `'confined'` and `B0_tor_l`, `'Malkus'`
  of degree 2 in a full sphere, `'ferraro'` without `diff_rot = 1`, `B0 = 'ferraro'` with `diff_rot_type = 'ferraro'`,
  `B0_wound = 1` without a poloidal field or without `diff_rot = 1`, `B0_conf_p` below 3, and `cnorm = 'rms_cmb'` with
  `'confined'`, `'general'` or `'ferraro'`;
- a value of `mag_couplings`, `B0_ic` or `ferraro_policy` other than those listed on this page.

## Output

`solve.py` writes `real_magnetic.field`, `imag_magnetic.field` (and `real_magnetic_ic.field`, `imag_magnetic_ic.field` for
a conducting inner core). Each eigenvector is normalised so that the kinetic energy plus `Hendrik` times the magnetic
energy is 1 in the fluid.

`spin_doctor` adds the Lorentz work to residσ and residω and prints two more residuals: resid𝐛, the energy budget of the
induction equation in the fluid, and resid𝐄, the full energy budget including the field outside the fluid (no-slip walls
only: `bco = 1`, and `bci = 1` with an inner core). Like residσ ([Solver](solver.md)), they should be small for a resolved mode. It also prints
Mag/Kin, the ratio of magnetic to kinetic energy (without the factor `Hendrik`), and writes `magnetic.dat` and 25
magnetic columns of `params.dat` ([Output files](output.md#magneticdat)).

## Practical tips

- **Resolution.** Where a poloidal field crosses a no-slip wall, a thin magnetic boundary layer (a *Hartmann layer*)
  forms; it, not the bulk flow, usually sets N. It gets thinner at lower Ekman numbers, so N must grow. A toroidal field
  alone has no Hartmann layer and converges faster. A full sphere needs about twice the N of a shell. Always compare the
  residuals at two values of N; examples are in the [checks](magnetic-equations.md#checks).
- **`'confined'`** converges more slowly than the other fields, since it is smooth only to a finite order at its edges,
  and most slowly when its layer reaches the centre of a full sphere.
- **`tol_B0`** (default `1e-8`): the relative cut of the radial series of the background field. It sets the band width of
  the magnetic blocks and the fill-in of the LU factors. Lower it for more accurate operators.
- **`mag_couplings`**: `'closed'` (default) uses the closed formulas, which
  exist for fields of degree 1 and 2. `'algebra'` builds the couplings for any degree (`bin/mag_algebra.py`); it is used
  automatically for `'general'`, `'ferraro'` and `B0_wound = 1`. Both give the same eigenvalues to round-off.
- **Memory.** Fields of high degree, and wound fields with a high-degree rotation, couple more degrees and need more LU
  memory. If MUMPS stops with INFOG(1) = −9, raise `st_mat_mumps_icntl_14` in `petsc_opts` ([Solver](solver.md)).

## Validity and caveats

The background field is held fixed. Two effects that would change the background are left out:

- **Ohmic decay of the poloidal field.** A real field would decay by diffusion; here it does not, with or without
  rotation.
- **The Lorentz force of the background field on the background flow.** The rotation is imposed and is not modified by
  the field, and the thermal-wind entropy is computed from the rotation alone. With `B0_wound = 1` this force is
  dominated by the wound field, which can be large. For the tachocline this is a significant approximation: a wound
  field that strong would brake the shear that makes it, as self-consistent tachocline models include.

### When the field leaves the thermal wind unchanged

With `thermal_wind` 1 or 2, the latitudinal entropy of the background balances the differential rotation alone
([Background model](background.md#thermal-wind)). The force of the background field would enter the same balance. The
derivations and the estimates are in [Magnetic field equations](magnetic-equations.md#when-the-ignored-lorentz-force-matters).

**Exact: a field without current.** A poloidal field that carries no electric current in the fluid, with no toroidal
part, exerts no force, so the thermal wind is exact. If the rotation also satisfies the Ferraro condition, the field
does not wind up or decay, and the whole background is steady. `B0 = 'axial'` is such a field, and so is a `'general'`
field whose profile of each degree n combines `r**n` and, in a shell, `r**-(n+1)`.

**Approximate: any other field.** All other fields carry current: the other poloidal types, any `B0_tor`, and the wound
field. The background is then a good approximation under three conditions.

1. *The meridional force is weak.* The field's force in the meridional planes (magnetic pressure, the hoop stress of
   the azimuthal field, the tension of curved poloidal lines) adds to the thermal-wind balance. It is negligible when

    $$
    \frac{\mathrm{Le}^2\,\lvert\mathbf B_0\rvert^2}{\rho_0} \ll 2\,\lvert\Delta\Omega\rvert ,
    $$

    with $\rho_0$ the background density and $\mathbf B_0$ the whole dimensionless field, wound part included. A field
    that varies on a shorter scale than the rotation makes this stricter. Where it fails, the thermal-wind entropy is
    wrong by about the ratio of the two sides.

2. *The torque is slow.* A poloidal field crossed by the shear brakes it (Ferraro's mechanism), at a rate set by the
   Elsasser number of the poloidal field built on the background diffusivity:

    $$
    \Lambda_p = \frac{\mathrm{Le}^2}{\rho_0\,\mathrm{Em}\,\eta_w},
    $$

    with $\eta_w$ = `B0_wound_eta`. The thermal wind follows the imposed rotation, so the torque does not change it
    directly, but the rotation must be maintained against it. It is negligible when this rate is small compared with
    the mode frequencies, their damping rates, and the rate of whatever maintains the rotation. If viscous or turbulent
    stresses hold the rotation, it is tenable only for a Hartmann number of order one or less:

    $$
    Ha = \frac{\mathrm{Le}}{\sqrt{\rho_0\,\mathrm{Ek}\,\mathrm{Em}\,\eta_w}} \lesssim 1 .
    $$

3. *The field is steady.* The Ferraro condition holds, or `B0_wound = 1` balances the winding. The wound field can be
   much stronger than the poloidal field that makes it, so it must meet condition 1 too. The Ohmic decay of the field
   must be slow compared with the modes.

Where the rotation is uniform there is no thermal wind. In a stably stratified region the field's force is then
balanced by a small change of the stratification, which kore leaves out. It is negligible when the Alfvén frequency is
far below the buoyancy frequency.

**What condition 1 means for the modes.** The error it leaves in the frequencies is about half the left side of
condition 1, in units of the rotation rate, for modes that feel the baroclinicity (more for slower modes). The field's
direct effect on a mode, which kore includes in full, is larger by about the square of the number of the mode's
wavelengths along the field lines. So a field can change the modes while leaving the thermal wind unchanged.

A `B0_wound` background is therefore a snapshot, valid for modes faster than these effects. This is the tachocline
confinement problem (Gough & McIntyre 1998; MacGregor & Charbonneau 1999): with molecular diffusivities a weak poloidal
field in the radiative interior has a large Hartmann number, and its torque would impose isorotation along the field
lines and spread the shear of the convection zone inward. Below the tachocline a strong field is better represented by
a Ferraro-consistent background (`B0 = 'ferraro'`, `diff_rot_type = 'ferraro'`) or a `'confined'` one. The checks of this
code, with a Hartmann number near 30 and a braking time of about one rotation, test the mathematics (closed form,
residuals, budgets), not the physical validity of such a background.

### The Sun, Jupiter and Saturn

Orders of magnitude with published values, where the field and the shear meet. The numbers and references are on
[Magnetic field equations](magnetic-equations.md#the-sun-jupiter-and-saturn).

| region | condition 1: meridional force | condition 2: torque | the background |
| --- | --- | --- | --- |
| Sun, convection zone | holds: large-scale fields of a few thousand G give a ratio of 10⁻³ or less | the weak mean poloidal field (about 10 G) brakes the rotation over a century or more; Hartmann number of order one with turbulent values | consistent; the rotation does not follow poloidal field lines, so use a toroidal `B0_tor`, or `B0_wound` from a weak poloidal field |
| Sun, tachocline | holds for 10⁴ G (ratio about 0.005); fails near 10⁵ G (ratio about 0.5) | set by the poloidal field and the diffusivity (above) | consistent up to about 2×10⁴ G |
| Sun, radiative interior | no thermal wind: the rotation is uniform | the Ferraro condition holds | consistent with `'confined'` or `'ferraro'` fields |
| Jupiter, winds (above 0.96 R) | holds: ratio about 10⁻⁹ | about a century | consistent |
| Saturn, winds (above 0.86 R) | holds: ratio about 10⁻¹² | about 10⁴ years | consistent |

The Sun is the case for magnetic effects on inertial modes. At the base of its convection zone a 10⁴ G field has a
Lehnert number of a few hundredths: enough to matter for modes with several wavelengths along the field, while
condition 1 still holds. In Jupiter and Saturn the Lehnert number is 10⁻⁵ to 10⁻⁷. Their fields change neither the
thermal wind nor the inertial modes measurably: kore's background is consistent there, and the field's effect on the
modes is negligible.

### Checklist for solar inertial modes

Their periods are longer than a rotation and their damping is weak (Löptien et al. 2018; Gizon et al. 2021), which
makes every criterion above stricter.

- **Time scales.** The braking rate and the Ohmic decay rate of the poloidal field must be small compared with the
  damping rate of the modes, not only with their frequency. The wound field needs a diffusion time across its layer to
  build up (years for a thin tachocline with a turbulent diffusivity): on solar-cycle time scales it is a snapshot.
- **Sensitivity.** The effect of the field grows roughly as the square of the Alfvén frequency over the mode frequency,
  so low-frequency modes feel it most. Scan `B0_wound_eta` and `Le` rather than trusting one value.
- **Critical layers and continua.** Where the mode frequency matches the local advection by the rotation (the critical
  latitudes) and at Alfvén resonances, the spectrum has continuous parts and the eigenfunctions thin layers. Check
  convergence in N and lmax, and tell discrete modes from the continuum.
- **Thermal wind.** Keep the thermal-wind heat rows on (`thermal_wind`) and check condition 1 above: the background
  baroclinicity does not include the forces of the field.
- **Consistent diffusivities.** The observed line widths set the turbulent viscosity (`ViscosD`). Take a turbulent
  magnetic Prandtl number near one (`Em` close to `Ek`) and set `B0_wound_eta` to the background's turbulent
  diffusivity. A wound field built with a molecular diffusivity under modes damped by turbulence is inconsistent.
- **Radiative interior.** With molecular values the Hartmann number is huge: use `'confined'` or `'ferraro'` fields below
  the tachocline, not `B0_wound`.
- **Where the modes live.** Most observed modes are concentrated in the convection zone (many near the surface or the
  equator), so a tachocline field reaches them only through their tails. Test the high-latitude and deep modes first.
- **Measurability.** A frequency shift matters only above about 1 nHz, a few thousandths of the rotation rate.
- **Symmetry.** A dipolar poloidal field with kore's equatorially symmetric rotation gives a wound field of the same
  family, so the two mode classes stay separate, as without the field.
- **Practical.** The wound field has degrees up to the sum of those of the rotation and of the poloidal field, plus one,
  so a high-degree `'tabulated'` rotation means wide couplings and more memory (see [Practical tips](#practical-tips)). In
  each run read the printed residual of the induction balance and the ratio of the wound to the poloidal field.
- **Bracketing.** Compute the same modes with `B0_wound` and with a `'ferraro'` background (the field the torque would
  relax to). The spread measures what the steady-background assumption costs.

## Further reading

- [Magnetic field equations](magnetic-equations.md): the equations, every field profile, the rows, the boundary
  conditions, the Ferraro constructions, the wound field, the energy budgets, and the checks against `conductive_IC` and
  published results.
- [Derivation tools](derivations.md), part A: how the magnetic rows are derived (`tests/magnetic/derive.py`), fitted
  (`tools/derive/`) and built for any degree (`bin/mag_algebra.py`).
- [Output files](output.md#magneticdat): the columns of `magnetic.dat` and the magnetic columns of `params.dat`.
