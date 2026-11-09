# Magnetic field

With `magnetic = 1`, kore adds a background magnetic field of your choice and solves for small perturbations of the
flow and of the field together: the Lorentz force acts on the flow, and the flow induces a perturbation of the field.
The background field is fixed, steady and axisymmetric. It is imposed, not computed: there is no dynamo, and the
background flow does not react to it.

**The field does not enter the thermal-wind balance.** The [thermal-wind entropy](background.md#thermal-wind) is
computed from the rotation alone, as without a field, and the Lorentz force acts only on the perturbations. This is
exact for a current-free field, such as `B0 = 'axial'` or `'dipole'`, and a good approximation when the magnetic force
is weak compared with the Coriolis force of the differential rotation ([Validity and caveats](#validity-and-caveats)).

This page shows how to set up a run and choose the background field. The equations and the formula of every field are
on [Magnetic field equations](magnetic-equations.md).

## Parameters and a first run

Two dimensionless numbers set how strong the field is and how fast it diffuses, both measured against the rotation:

| parameter | meaning |
| --- | --- |
| `Le` | Lehnert number: the Alfvén speed of the field over the rotation speed of the outer wall |
| `Em` | magnetic Ekman number: the magnetic diffusivity over the rotation rate times the radius squared |

Two combinations are common: the magnetic Prandtl number `Ek/Em` and the Elsasser number `Le**2/Em`. The exact
definitions are on [Magnetic field equations](magnetic-equations.md#equations).

**A first run.** A uniform field along the rotation axis, in a shell with insulating walls, with Elsasser and magnetic
Prandtl numbers both equal to one. Add these lines at the end of a `bin/parameters.py` like the default one (which
already imports numpy and sets `m`, `symm`, `N`, `lmax` and the target):

```python
par.Ek = 1e-3                # Ekman number
par.Em = 1e-3                # magnetic Ekman number: Pm = Ek/Em = 1
par.Le = np.sqrt(1e-3)       # Lehnert number: Elsasser number Le**2/Em = 1
par.set_scales()             # call it again: it sets Hendrik and MagnetD from Le and Em

par.structure = 'Boussinesq' # any structure works (Background model)
par.magnetic = 1
par.thermal = 0              # no entropy equation
par.B0 = 'axial'             # the poloidal background field (tables below)
par.cnorm = 'mag_energy'     # its amplitude: the integral of B0**2 over the fluid is 1
par.B0_tor = None            # no toroidal background field
par.innercore = 'insulator'
par.mantle = 'insulator'
par.ricb = 0.35; par.bci = 1; par.bco = 1   # no-slip walls
```

`set_scales()` converts `Le` and `Em` into the factors that multiply the Lorentz force (`Hendrik`) and the magnetic
diffusion (`MagnetD`), so call it after setting them. Two values of `timescale` are magnetic: `'Alfven'` (the Alfvén
time) and `'magnetic'` (the magnetic diffusion time).

## Choosing the background field

The background field is the sum of two parts: a *poloidal* part, whose field lines lie in meridional planes like those
of a dipole, and a *toroidal* part, whose field lines are circles around the rotation axis. Choose a shape for each
with `B0` and `B0_tor`, then set their amplitude. `B0` defaults to `'axial'` and `B0_tor` to `None`; for a toroidal
field alone, set `B0 = None`.

### Equatorial symmetry

- A poloidal field of odd degree is *dipolar* (antisymmetric about the equator), one of even degree *quadrupolar*
  (symmetric). For a toroidal field it is the other way round: degree 1 is symmetric, degree 2 antisymmetric.
- The two parts must have the same symmetry. A dipole therefore combines with a toroidal field of degree 2, not 1.
- `symm` still selects the symmetry of the flow.

### Poloidal field: `B0`

The formulas are on [Magnetic field equations](magnetic-equations.md#poloidal-fields).

| `B0` | the field | extra parameters (default) | geometry |
| --- | --- | --- | --- |
| `'axial'` (default) | uniform, along the rotation axis | – | shell or full sphere |
| `'dipole'` | the field of a dipole at the centre | – | shell only |
| `'FDM'` | the slowest free-decay mode of degree `B0_l` (1 or 2) | `B0_l` (1) | shell or full sphere |
| `'G21 dipole'` | a full-sphere dipole (Gerick et al. 2021) | `B0_ic` | shell or full sphere |
| `'Luo_S1'` | the same shape (Luo & Jackson 2022) | `B0_ic` | shell or full sphere |
| `'Luo_S2'` | a full-sphere quadrupole (Luo & Jackson 2022) | `B0_ic` | shell or full sphere |
| `'confined'` | a field that is zero outside a spherical layer | `B0_l` (1), `B0_conf_r1` (0.0), `B0_conf_r2` (0.6), `B0_conf_p` (12) | the layer must lie in the fluid |
| `'general'` | your own radial profile for each degree | `B0_profiles`, `B0_param` | shell or full sphere |
| `'ferraro'` | field lines along the surfaces of constant rotation ([below](#four-ways-to-a-steady-background)) | `B0_ferraro_G` ('w'), `B0_ferraro_nmax` (15) | needs `diff_rot = 1` |
| `None` | no poloidal field | | |

- **Full-sphere shapes with an inner core.** `'G21 dipole'`, `'Luo_S1'` and `'Luo_S2'` are defined for a full sphere.
  With an inner core, `B0_ic = 'potential'` (default) continues the field into the core as a current-free field;
  `B0_ic = 'frozen currents'` keeps the full-sphere formula.
- **`'confined'`** fills the layer from `B0_conf_r1` to `B0_conf_r2`. The defaults reach the centre, which suits a full
  sphere; with an inner core, set `B0_conf_r1` at or above `ricb`. `B0_conf_p` (at least 3) sets how smoothly the field
  vanishes at the edges.
- **`'general'`** takes a numpy expression of `r` and `p` for each degree. In a full sphere the profile of degree n
  should behave as `r**n` near the centre. Example:

    ```python
    par.B0_profiles = {1: 'r/6 - r**3/10', 3: '0.3*r**3*(1 - r**2)'}
    ```

- **At the walls.** `'general'` and `'ferraro'` are treated as fields produced by currents inside the fluid. If they
  do not match a current-free field at an insulating wall, a current sheet forms there
  ([Current sheets at slipping walls](#current-sheets-at-slipping-walls)). Use `'axial'` for a uniform field and
  `'dipole'` for a dipole at the centre, rather than writing them as `'general'`.

### Toroidal field: `B0_tor`

A toroidal field is purely azimuthal, so differential rotation does not wind it up; it suits any rotation. The
formulas are on [Magnetic field equations](magnetic-equations.md#toroidal-fields).

| `B0_tor` | the field | extra parameters (default) |
| --- | --- | --- |
| `None` (default) | no toroidal field | |
| `'Malkus'` | grows linearly with distance from the axis (Malkus 1967) | `B0_tor_l` (1) |
| `'wall'` | zero at both walls | `B0_tor_l` (1) |
| `'user'` | a radial shape given as an expression of `r` and `p` | `B0_tor_profile`, `B0_tor_param`, `B0_tor_l` (1) |
| `'general'` | your own radial profile for each degree | `B0_tor_profiles`, `B0_tor_param` |

`B0_tor_l` is 1 or 2. A `'user'` profile must be smooth, since its second derivative enters the equations. Expressions
must not contain `#`.

A toroidal field that is not zero at a wall ends there in a current sheet, unless a conductor beyond the wall carries
it on ([Current sheets at slipping walls](#current-sheets-at-slipping-walls)). With the
[rotating solids](rotating-solids.md) it must vanish at both walls.

### Amplitude

The shapes above have arbitrary scales, so the size of the field is set separately: the field is `Le` times a
dimensionless field whose size is set by `cnorm` (poloidal part) and `cnorm_tor` (toroidal part).

| value | meaning |
| --- | --- |
| a number (default 1) | multiplies the formula as it is |
| `'mag_energy'` | the integral of the squared field over the fluid is 1 (Luo & Jackson 2022). Recommended |
| `'Schmitt2012'` | the integral of the squared field over the fluid is 4 |
| `'rms_cmb'` | the rms radial field at the outer wall is 1 (`cnorm` only, not for `'confined'`, `'general'`, `'ferraro'`) |

With `'mag_energy'`, `Le` and the Elsasser number refer to a field of fixed energy, whatever its shape. With a number,
the strength of the field depends on the arbitrary scale of the formula.

## With a differential rotation

With `diff_rot = 1` ([Background model](background.md#rotation)), the shear acts on the field in two ways:

- It shears the **perturbation** field. kore includes this for any rotation profile.
- It **winds up the background** poloidal field into an azimuthal field, which grows until magnetic diffusion balances
  it. kore keeps the background fixed and does not follow this growth. The background is therefore steady only if
  nothing is wound up, or if it already contains the balanced azimuthal field.

### The Ferraro condition

Differential rotation does not wind up a poloidal field line if the rotation rate is the same all along it (Ferraro's
law of isorotation). A toroidal field always satisfies it; a poloidal field usually does not.

At the start of a run with a poloidal field and `diff_rot = 1`, kore prints how far the field and the rotation are from
this condition ([definition](magnetic-equations.md#the-ferraro-condition)) and warns if the mismatch is significant.
Set `ferraro_policy = 'refuse'` to make it stop instead.

### Four ways to a steady background

| set-up | what kore does | other parameters (default) | use it when |
| --- | --- | --- | --- |
| `B0 = 'ferraro'` | keeps the rotation, and uses a poloidal field whose lines lie along the surfaces of constant rotation | `B0_ferraro_G` ('w'), `B0_ferraro_nmax` (15, odd) | the rotation is given, and any field that it does not wind up will do |
| `diff_rot_type = 'ferraro'` | keeps the field, and uses a rotation that is constant along each field line | `diff_rot_F` ('psi'), `diff_rot_nmax` (8) | the field is given, and the rotation only has to fit it |
| `B0 = 'confined'` | nothing: you put the field in a layer where the rotation is uniform | `B0_conf_r1`, `B0_conf_r2` | the field lives away from the shear (e.g. below the tachocline) |
| `B0_wound = 1` | keeps both, and adds the azimuthal field that the shear winds up, balanced by diffusion | `B0_wound_eta` (1.0) | both are given and do not match |

**`B0 = 'ferraro'`.** The field lines follow the surfaces of constant rotation, so nothing is wound up. `B0_ferraro_G`,
a function of the rotation rate `w`, sets how much magnetic flux passes between the axis and each surface. Where the
rotation is uniform the field vanishes. kore prints how well the field stays inside the fluid. `cnorm` must be a number,
`'mag_energy'` or `'Schmitt2012'`.

**`diff_rot_type = 'ferraro'`.** kore keeps the poloidal field and replaces the rotation by one that is constant along
each field line. `diff_rot_F` gives the rotation rate of each field line as a function of the flux function `psi`. With
a quadrupolar field, `diff_rot_F` must be even in `psi`.

**`B0 = 'confined'`.** If the rotation is uniform in the layer of the field, as below the solar tachocline, nothing is
wound up.

**`B0_wound = 1`.** kore keeps the poloidal field and the rotation as given, and adds the azimuthal field that the
shear winds up, at the strength where diffusion balances the winding. Its strength follows from the physics, not from
`cnorm`: it grows with the shear and with the inverse of the diffusivity, and can be many times stronger than the
poloidal field. `B0_wound_eta` is the diffusivity used for this balance, in units of `MagnetD`. This background holds
only while something maintains the rotation against the magnetic braking. Read [Validity and caveats](#validity-and-caveats)
before using it.

`B0 = 'ferraro'` and `diff_rot_type = 'ferraro'` describe a field and a rotation that have reached isorotation.
`B0_wound = 1` describes a shear that something maintains across the field lines.

A conducting inner core must rotate rigidly with the fluid at its surface, so the rotation at `ricb` must not depend on
latitude.

## Inner core and walls

The perturbation field is matched at each wall to the field beyond it. The mantle is an insulator, possibly with a
conducting layer at its base: either a thin one, described by boundary conditions alone, or one of any thickness whose
field kore solves for. The inner core is an insulator (again possibly with a thin conducting layer), or a conductor
whose field kore solves for.

| setting | values (default) | notes |
| --- | --- | --- |
| `mantle` | `'insulator'` (default), `'TWA'` or `'resolved'` | `'TWA'`: a thin conducting layer; `'resolved'`: a conducting layer of any thickness (both below) |
| `innercore` | `'insulator'` (default), `'conducting, Chebys'`, `'conducting, Bessel'` or `'TWA'` | need `ricb > 0`; `'conducting, Bessel'` in forced runs only ([Forced runs](#forced-runs)) |
| `N_cic` | 64 | number of Chebyshev polynomials for the inner-core field with `'conducting, Chebys'` (even, at least 4) |
| `mu_i2o` | 1.0 | magnetic permeability, inner core over fluid |
| `sigma_i2o` | 1.0 | electrical conductivity, inner core over fluid |
| `B0_ic` | `'potential'` | for `'G21 dipole'`, `'Luo_S1'`, `'Luo_S2'` with an inner core (above) |

Matching conditions: [Magnetic field equations](magnetic-equations.md#boundary-conditions).

### Thin conducting layer

A thin conducting layer, such as the base of the Earth's mantle, carries currents that couple the field across it.
`'TWA'` (the thin-wall approximation of Roberts, Glatzmaier & Clune 2010) models such a layer, at the base of the mantle
or on an insulating inner core, through boundary conditions alone, without extra unknowns.

| setting | default | meaning |
| --- | --- | --- |
| `c_cmb`, `c_icb` | `1e-5`, 0 | thickness times permeability of the layer, relative to the wall radius and to the fluid's permeability |
| `c1_cmb`, `c1_icb` | `1e-5`, 0 | conductance of the layer: thickness times conductivity, relative to the wall radius and to the fluid's conductivity |
| `muw_cmb`, `muw_icb` | 1.0 | permeability of the layer relative to vacuum |
| `mu` | 1.0 | permeability of the fluid relative to vacuum (used only by the thin-wall conditions) |
| `B0_cmb` | `'wall'` | how the background field meets a thin layer at the outer wall: `'wall'` adjusts it to the layer, `'plain'` keeps it as given |

- With the default permeabilities the conductance `c1` sets nearly all of the effect of the layer.
- **Validity.** The layer must be thin compared with its magnetic skin depth at the frequency of the mode. `spin_doctor`
  prints the estimated relative error, εTWA; keep it below about 1 %.
- **With differential rotation**, a conducting layer turns with the fluid next to it, so the rotation must be uniform in
  latitude at that wall.
- **On the exact spheroid** (`centrifugal_shape = 'spheroid'`) the layer at the outer wall works in its thin limit,
  `c_cmb = 0` and `mu = 1`, with a no-slip or a slipping fluid at the wall. The layer lies between two
  level surfaces, so in an oblate body it is a little thicker at the equator than at the poles
  ([The exact spheroid](centrifugal-equations.md#the-exact-spheroid)).

The conditions: [Magnetic field equations](magnetic-equations.md#thin-conducting-layer).

### Resolved conducting mantle

When the conducting layer is too thick, too conducting, or the mode too slow for the thin-wall approximation, use
`mantle = 'resolved'`. kore then puts a conducting layer between the outer wall and `r_surface`, with an insulator
beyond, and solves for the field inside it. The layer is solid and at rest, and its conductivity may vary with radius.

| setting | default | meaning |
| --- | --- | --- |
| `r_surface` | 1.83 | outer radius of the layer, the outer wall being at 1 (1.83 is the whole mantle of the Earth) |
| `N_mantle` | 64 | number of Chebyshev polynomials for the field in the layer (at least 4) |
| `sigma_mantle` | 1.0 | conductivity of the layer relative to the fluid's: a number, or an expression of `r` and `p` such as `'1e-2*exp(-p*(r - 1))'` |
| `sigma_mantle_param` | 0.0 | the `p` of that expression |

- **Resolution.** The conductivity profile must be smooth; kore stops if `N_mantle` does not resolve it. Check the
  field in the layer with a second run at a larger `N_mantle`.
- **With differential rotation** the rotation must be uniform in latitude at the outer wall.
- The layer and the fluid have the permeability of vacuum (`mu = 1`).
- **On the exact spheroid** (`centrifugal_shape = 'spheroid'`) the layer ends at the level surface `r_surface`, a
  spheroid of the same flattening, and with the flow the outer wall must be no-slip (`bco = 1`)
  ([The exact spheroid](centrifugal-equations.md#the-exact-spheroid)).

The equations: [Magnetic field equations](magnetic-equations.md#resolved-conducting-mantle).

### Current sheets at slipping walls

A background field that does not match the field beyond an insulating wall ends there in a sheet of electric current.
This happens for a toroidal field that is not zero at the wall, or a poloidal field that does not match the
current-free field outside. A conductor beyond the wall (a conducting inner core, a resolved mantle or a conducting thin
layer) carries the field on, with no sheet. The sheet pushes the fluid along the wall. A no-slip wall takes that push;
along a stress-free wall, kore includes it in the boundary conditions
([Magnetic field equations](magnetic-equations.md#current-sheets-at-slipping-walls)).

An inviscid fluid cannot take this push, so a current sheet stops an inviscid run. Use a field that matches at the wall
(such as `B0_tor = 'wall'`), a viscous fluid, or a conductor beyond the wall.

## The field without the flow

**Free decay: `hydro = 0`.** Without the momentum equation the field only diffuses (and is carried by the differential
rotation, if any). Each mode decays at a rate set by the conductivity, the walls and the inner core, which for uniform
conductivity is known exactly: a useful check of the boundary conditions. `B0` still sets the equatorial symmetry of the
perturbation. With `thermal = 1` the same switch gives the free decay of the entropy. This works for eigenvalue problems
only.

**Kinematic induction.** `bin/just_induction.py` and `bin/solve_ind.py` give the field that a given flow induces,
without the force of that field on the flow. Build a magnetic eigenvalue problem (`submatrices.py`, `assemble.py`), put
the flow in `flow.npy` ([Output files](output.md#solution-files)) from a run with the same `N`, `lmax`, `m`, `symm` and
`ricb`, then run

```shell
./bin/just_induction.py [lambda]
mpiexec -n 4 ./bin/solve_ind.py
```

The optional `lambda` is the time dependence of the flow, a complex number such as the eigenvalue of a mode; by default
it is $i$ times `forcing_frequency`. The field is written to `magnetic.npy`.

## Forced runs

With a field, every forcing except the wave-like tide (forcing 1) works as without one: the field is solved together
with the flow at the forcing frequency, and `spin_doctor` checks the budgets ([Tidal forcing](tidal-forcing.md),
[Libration](libration.md), [Thermal forcing](thermal-forcing.md), [Spherical Couette flow](couette-flow.md),
[Convective forcing](forcing.md)).

- **A conducting inner core at one frequency.** `innercore = 'conducting, Bessel'` uses the exact solution inside a
  uniform conducting core at the forcing frequency, so it adds no unknowns. It gives the same result as
  `'conducting, Chebys'`, but only for forced runs at `forcing_frequency` (not for frequency scans).
- **Walls that turn.** In the libration (forcing 7 with `m = 0`) and the rotating walls (forcing 12), a conducting
  solid behind a turning wall turns with it.

### A boundary that moves

Forcings 6, 9, 10 and 14, and forcing 7 with `m = 2`, represent a boundary that moves or is not spherical. The walls
stay spherical and the forcing prescribes the radial flow at the wall. A conducting solid behind the wall moves too, and
`forced_walls` sets how:

| `forced_walls` | the solid behind the wall | suits |
| --- | --- | --- |
| `'rigid'` | stays at rest; only its shape moves past the sphere | a triaxial mantle that librates (7 with `m = 2`), the flattened inner core of forcing 6 |
| `'deforming'` | deforms with the boundary | a boundary that moves (9, 10), a tide (14) |
| `None` (default) | `'rigid'` for forcings 6 and 7, `'deforming'` for 9, 10 and 14 | |

With insulating solids `forced_walls` changes nothing. A moving wall that the background field reaches needs the same
magnetic permeability on both sides (`mu`, `mu_i2o` and the thin layer's `muw` equal to 1); kore stops otherwise. The
eccentricity tide (forcing 2) needs insulating walls.

The equations: [Magnetic field equations](magnetic-equations.md#forced-runs).

## What is supported

kore checks the combination of options at startup and stops, with a message, on any it cannot handle.

Supported: a shell or a full sphere; any `structure`; eigenvalue problems and forced runs; with or without
differential rotation; a viscous fluid, or an inviscid one in a full sphere; with or without the flow (`hydro = 0`);
insulating walls, a thin conducting layer, a resolved conducting mantle or a conducting inner core; rotating solids
([Rotating solids](rotating-solids.md)). The magnetic diffusivity of the fluid is uniform.

The main restrictions:

- a background field is required, and `Le` and `Em` must not be zero;
- a conducting inner core, or a conducting layer at a wall, needs the rotation to be uniform in latitude at that wall;
- walls with zero radial stress (`bci` or `bco` = 2) do not work with a conducting solid behind them;
- the poloidal and toroidal parts must have the same equatorial symmetry;
- `'conducting, Bessel'` works only in forced runs at a single frequency;
- a current sheet at the wall of an inviscid fluid stops the run;
- the wave-like tide (forcing 1) does not work with a field.

[Combining features](compatibility.md) shows how the field combines with the other features.

## Output

`solve.py` writes the perturbation field to `magnetic.npy`, plus `magnetic_ic.npy` for a conducting inner core and
`magnetic_mantle.npy` for a resolved mantle ([Output files](output.md#solution-files)). Each eigenvector is normalised so
that its total (kinetic plus magnetic) energy is 1.

`spin_doctor` includes the Lorentz force in its budgets and prints two more residuals: resid𝐛, the energy budget of the
induction equation, and resid𝐄, the full energy budget including the field outside the fluid. Like residσ
([Solver](solver.md)), they should be small for a resolved mode; with a thin conducting layer of non-zero thickness,
resid𝐄 is only as small as the approximation error εTWA. It also prints Mag/Kin, the ratio of magnetic to kinetic
energy, and writes `magnetic.dat` ([Output files](output.md#magneticdat)).

## Practical tips

The field, more than the flow, usually sets the resolution and memory a run needs.

- **Resolution.** Where a poloidal field crosses a no-slip wall, a thin magnetic boundary layer (a *Hartmann layer*)
  forms. It usually sets `N`, and it gets thinner at lower Ekman numbers. A toroidal field alone has no Hartmann layer
  and converges faster. A full sphere needs about twice the `N` of a shell. Always compare the residuals at two values
  of `N`.
- **`'confined'`** converges more slowly than the other fields.
- **`tol_B0`** (default `1e-8`) is the relative truncation of the background field's radial series. Lower it for more
  accurate operators, at some cost in memory.
- **`mag_couplings`.** `'closed'` (default) uses closed formulas, which exist for fields of degree 1 and 2.
  `'algebra'` works for any degree and is used automatically for `'general'`, `'ferraro'` and `B0_wound = 1`. Both give
  the same eigenvalues.
- **Memory.** Fields of high degree couple more harmonics and need more memory. If MUMPS stops with
  INFOG(1) = −9, raise `st_mat_mumps_icntl_14` in `petsc_opts` ([Solver](solver.md)).

## Validity and caveats

kore holds the background field and the background flow fixed. This leaves out two effects:

- **Ohmic decay.** A field that carries electric current would slowly decay. Only the wound field is steady by itself.
- **The Lorentz force of the background field on the background flow.** The rotation is imposed, and the thermal wind
  is computed from the rotation alone.

### When the field leaves the thermal wind unchanged

**Exactly, for a current-free field.** A poloidal field that carries no current in the fluid exerts no force, so the
thermal wind is exact. `B0 = 'axial'` and `'dipole'` are such fields.

**Approximately, for any other field**, under three conditions (the derivations are in
[Magnetic field equations](magnetic-equations.md#when-the-ignored-lorentz-force-matters)):

1. *The magnetic force in the meridional plane is weak* compared with the Coriolis force of the differential rotation:

    $$
    \frac{\mathrm{Le}^2\,\lvert\mathbf B_0\rvert^2}{\rho_0} \ll 2\,\lvert\Delta\Omega\rvert ,
    $$

    with $\rho_0$ the background density, $\mathbf B_0$ the whole dimensionless field (wound part included) and
    $\Delta\Omega$ the relative differential rotation.

2. *The magnetic braking is slow.* A poloidal field crossed by the shear brakes it. The braking must be slow compared
   with the mode frequencies and damping rates, and with whatever maintains the rotation.

3. *The field is steady*: the Ferraro condition holds, or `B0_wound = 1` balances the winding.

The direct effect of the field on a mode, which kore includes in full, can be much larger than its effect on the
thermal wind. So a field can change the modes noticeably while leaving the thermal wind unchanged.

### Wound fields and the tachocline

A `B0_wound` background is a snapshot: it holds only for modes faster than the magnetic braking and the Ohmic decay.
With molecular diffusivities, even a weak poloidal field in the solar radiative interior would quickly enforce
isorotation along its field lines (the tachocline confinement problem: Gough & McIntyre 1998). A field below the
tachocline is better represented by a `'confined'` field whose layer lies there, where the rotation is uniform. A
toroidal `B0_tor` can be added in the tachocline and the convection zone, since a toroidal field is never wound up.

### The Sun, Jupiter and Saturn

Estimates with published values ([details](magnetic-equations.md#the-sun-jupiter-and-saturn)):

- **Sun, convection zone and tachocline.** The approximation holds for fields up to about $2\times10^4$ G. At the
  base of the convection zone a $10^4$ G field can noticeably affect modes with several wavelengths along the field.
- **Sun, radiative interior.** The rotation is uniform; use a `'confined'` field there.
- **Jupiter and Saturn.** The approximation holds easily, and the fields have a negligible effect on the large-scale
  inertial modes.

### Checklist for solar inertial modes

Solar inertial modes have periods longer than a rotation, so the field's effect on them is relatively large and the
background must stay steady over their lifetime.

- **Time scales.** The magnetic braking and decay must be slow compared with the damping of the modes.
- **Sensitivity.** Low-frequency modes feel the field most. Scan `Le` and `B0_wound_eta` rather than trusting one value.
- **Critical layers and continua.** Check convergence in `N` and `lmax`, and distinguish discrete modes from continuous
  parts of the spectrum.
- **Thermal wind.** Keep it on (`thermal_wind` 1 or 2) and check condition 1 above.
- **Consistent diffusivities.** Take a turbulent magnetic Prandtl number near one (`Em` close to `Ek`), and build the
  wound field with the same diffusivity (`B0_wound_eta = 1`).
- **Radiative interior.** Keep the field where the rotation is uniform: use `'confined'`, not `B0_wound`.
- **Measurability.** A frequency shift matters only above about 1 nHz.
- **Field geometry.** The field inside the Sun is not known. Compute the same modes with more than one geometry at a
  similar strength; the spread shows how much the results depend on the assumed field.

## Further reading

- [Magnetic field equations](magnetic-equations.md): the equations, every field profile, the boundary conditions and
  the energy budgets.
- [Derivation tools](derivations.md): how the magnetic terms are derived and checked.
- [Output files](output.md#magneticdat): the columns of `magnetic.dat`.
