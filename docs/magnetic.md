# Magnetic field

With `magnetic = 1`, kore adds a background magnetic field that you choose, and solves for small perturbations of the
flow and of the field together: the field's Lorentz force acts on the flow, and the flow induces a perturbation of the
field. The background field is fixed, steady and axisymmetric. It is imposed, not computed: there is no dynamo, and the
background flow does not react to it.

**kore does not include the field in the thermal-wind balance.** The [thermal-wind entropy](background.md#thermal-wind)
is computed from the imposed rotation alone, as without a field, and the field's force acts only on the perturbations.
This is exact for a field that carries no current, such as `B0 = 'axial'` or `'dipole'`, and a good approximation when
the field's force is weak compared with the Coriolis force of the differential rotation
([Validity and caveats](#validity-and-caveats)).

This page shows how to set up a run and how to choose the background field. The equations, the formula of every field
and the checks of the code are on [Magnetic field equations](magnetic-equations.md).

## Parameters and a first run

Two dimensionless numbers set how strong the field is and how fast it diffuses. Both are measured against the rotation,
as the Ekman number `Ek` is:

| parameter | meaning |
| --- | --- |
| `Le` | Lehnert number: the Alfvén speed of the field over the rotation speed of the outer wall |
| `Em` | magnetic Ekman number: the magnetic diffusivity over the rotation rate times the radius squared |

Two combinations are common: the magnetic Prandtl number `Ek/Em` and the Elsasser number `Le**2/Em`. The exact
definitions are on [Magnetic field equations](magnetic-equations.md#equations).

**A first run.** A uniform field along the rotation axis, in a shell with insulating walls, with Elsasser and magnetic
Prandtl numbers both equal to one. Add these lines at the end of a `bin/parameters.py` like the checked-in one (which
already imports numpy and sets `m`, `symm`, `N`, `lmax` and the target), so that no later line overrides them:

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

`set_scales()` turns `Le` and `Em` into the factors of the equations, `Hendrik` (Lorentz force) and `MagnetD` (magnetic
diffusion), so call it after setting them. Two values of `timescale` are magnetic: `'Alfven'` (the Alfvén time) and
`'magnetic'` (the magnetic diffusion time). A run stops if `Le` or `Em` is zero.

## Choosing the background field

The background field is one axisymmetric field for the whole domain. It is the sum of two parts: a *poloidal* part,
whose field lines lie in meridional planes like those of a dipole, and a *toroidal* part, whose field lines are circles
around the rotation axis. Choose a shape for each, with `B0` and `B0_tor`, then set their amplitude. `B0` defaults to
`'axial'` and `B0_tor` to `None`; for a toroidal field alone, set `B0 = None`.

### Equatorial symmetry

The symmetry of the field about the equator decides which parts can be combined, and how the perturbations are paired.

- A poloidal field of odd degree is *dipolar* (antisymmetric about the equator), one of even degree *quadrupolar*
  (symmetric). For a toroidal field it is the other way round: degree 1 is symmetric, degree 2 antisymmetric.
- The two parts must have the same symmetry, or the run stops. A dipole therefore combines with a toroidal field of
  degree 2, not of degree 1.
- `symm` still selects the symmetry of the flow. A symmetric background field gives the perturbation field the flow's
  symmetry; an antisymmetric one gives it the opposite.

### Poloidal field: `B0`

Each option is a fixed radial shape (several for `'general'`), multiplied by `cnorm` ([Amplitude](#amplitude)). The
formulas are on [Magnetic field equations](magnetic-equations.md#poloidal-fields).

| `B0` | the field | extra parameters (default) | geometry | degree |
| --- | --- | --- | --- | --- |
| `'axial'` (default) | uniform, along the rotation axis | none | shell or full sphere | 1 |
| `'dipole'` | the field of a dipole at the centre | none | shell only | 1 |
| `'FDM'` | the slowest free decay mode of the given degree, for insulating walls | `B0_l` (1); `beta`, the first guess of the root finder | shell or full sphere | `B0_l` = 1 or 2 |
| `'G21 dipole'` | a full-sphere dipole (Gerick et al. 2021) | `B0_ic` | shell or full sphere | 1 |
| `'Luo_S1'` | the shape of `'G21 dipole'` (Luo & Jackson 2022), at a larger scale | `B0_ic` | shell or full sphere | 1 |
| `'Luo_S2'` | a full-sphere quadrupole (Luo & Jackson 2022) | `B0_ic` | shell or full sphere | 2 |
| `'confined'` | zero outside a spherical layer, going smoothly to zero at its edges | `B0_l` (1), `B0_conf_r1` (0.0), `B0_conf_r2` (0.6), `B0_conf_p` (12) | the layer must lie in the fluid | `B0_l` = 1 or 2 |
| `'general'` | your own radial profile for each degree | `B0_profiles` = {degree: 'expression of r, p'}, `B0_param` (p) | shell or full sphere | any, of one parity |
| `'ferraro'` | field lines on the surfaces of constant rotation, so that the rotation does not wind it up ([below](#four-ways-to-a-steady-background)) | `B0_ferraro_G` ('w'), `B0_ferraro_nmax` (15, odd), `B0_param` | needs `diff_rot = 1` | odd, up to `B0_ferraro_nmax` |
| `None` | no poloidal field | | | |

Notes:

- **Full-sphere shapes with an inner core.** `'G21 dipole'`, `'Luo_S1'` and `'Luo_S2'` are defined for a full sphere.
  With an inner core, `B0_ic = 'potential'` (default) continues the field into the core as a potential field;
  `B0_ic = 'frozen currents'` keeps the full-sphere formula, whose currents below the inner boundary then stay in the
  core (fitting only when the core's magnetic decay time is much longer than the periods sought; the name was
  `'full sphere'` before 2026-11-06).
- **`'dipole'`** is the field of a dipole at the centre, seen from outside its source: it carries no current in the
  fluid and meets an insulating mantle as it is. Its source is in the inner core, so it needs one, and `B0_ic` does not
  apply to it. A conducting inner core that moves with a forced boundary holds it as a regular field instead
  ([A boundary that moves](#a-boundary-that-moves)).
- **`'Luo_S1'` and `'G21 dipole'`** give the same run when `cnorm` is a name (`'mag_energy'`, `'Schmitt2012'`,
  `'rms_cmb'`). With a numerical `cnorm` the `'Luo_S1'` field is 30 times stronger.
- **`'confined'`** puts the field in the layer from `B0_conf_r1` to `B0_conf_r2` and nowhere else. The defaults reach
  the centre, which suits a full sphere; with an inner core, set `B0_conf_r1` at or above `ricb`. `B0_conf_p` (at least
  3) sets how smoothly the field vanishes at the edges, so that no current sheet forms there.
- **`'general'`** takes numpy expressions of `r` and `p` (`sin`, `cos`, `exp`, `sqrt`, `where`, `erf`, `pi`, ...). In a
  full sphere the profile of degree n should go as `r**n` at the centre. Example:

    ```python
    par.B0_profiles = {1: 'r/6 - r**3/10', 3: '0.3*r**3*(1 - r**2)'}
    ```

- **`'ferraro'`** takes `B0_ferraro_G`, a numpy expression of `w` (the differential rotation) and `p`: see
  [Four ways to a steady background](#four-ways-to-a-steady-background).
- **At the walls.** kore takes `'general'` and `'ferraro'` as fields of currents in the fluid. Where they do not meet
  the potential field of those currents beyond an insulating wall, they end there in a sheet of current, which pushes
  the fluid along the wall ([Current sheets at slipping walls](#current-sheets-at-slipping-walls)). So does a part
  written into them whose sources lie beyond the wall, such as the uniform field `{1: 'r/2'}`: use `'axial'` for a
  uniform field and `'dipole'` for a dipole at the centre.

### Toroidal field: `B0_tor`

The toroidal part is purely azimuthal. A differential rotation does not wind it up, so it suits any rotation. Each
option is one radial shape times `cnorm_tor`; the formulas are on
[Magnetic field equations](magnetic-equations.md#toroidal-fields).

| `B0_tor` | the field | extra parameters (default) | geometry | degree |
| --- | --- | --- | --- | --- |
| `None` (default) | no toroidal field | | | |
| `'Malkus'` | grows linearly with the distance to the axis (degree 1), the field of Malkus (1967) | `B0_tor_l` (1) | shell or full sphere; degree 2 shell only | `B0_tor_l` = 1 or 2 |
| `'wall'` | zero at both walls | `B0_tor_l` (1) | shell or full sphere | `B0_tor_l` = 1 or 2 |
| `'user'` | a radial shape given as an expression of `r` and `p` | `B0_tor_profile`, `B0_tor_param` (p), `B0_tor_l` (1) | shell or full sphere | `B0_tor_l` = 1 or 2 |
| `'general'` | your own radial profile for each degree | `B0_tor_profiles` = {degree: 'expression of r, p'}, `B0_tor_param` (p) | shell or full sphere | any, of one parity |

A `'user'` profile must be smooth, since its second derivative enters the equations. The expressions of `'user'`,
`'general'` (poloidal or toroidal) and `'ferraro'` are written to `params.dat` after `#`, so they must not contain `#`.

**At the walls.** A toroidal field that is not zero at a wall ends there in a sheet of current, unless a conductor
beyond the wall carries the field on: a conducting inner core, the resolved mantle or a conducting thin layer. The sheet
pushes the fluid along the wall ([Current sheets at slipping walls](#current-sheets-at-slipping-walls)). With the
rotating solids the field must vanish at both walls, whatever they are ([Rotating solids](rotating-solids.md)).

### Amplitude

The shapes above have arbitrary scales, so the size of the field is set separately. The field in the equations is `Le`
times a dimensionless field, whose size is fixed by `cnorm` (poloidal part) and `cnorm_tor` (toroidal part). The Lorentz
force goes as `Le` squared.

| value | meaning | for |
| --- | --- | --- |
| a number (default 1) | multiplies the formula of the shape as it is | all types |
| `'mag_energy'` | the integral of the squared field over the fluid is 1 (Luo & Jackson 2022) | all types |
| `'Schmitt2012'` | the integral of the squared field over the fluid is 4 | all types |
| `'rms_cmb'` | the rms radial field over the outer wall is 1 | `cnorm` with `'axial'`, `'dipole'`, `'FDM'`, `'G21 dipole'`, `'Luo_S1'`, `'Luo_S2'` only |

Use `'mag_energy'`, as in the example: `Le` and the Elsasser number then refer to a field of fixed energy, whatever its
shape. With a number, the field's size depends on the arbitrary scale of the formula, and `Le` alone does not say how
strong the field is. `'confined'`, `'general'` and `'ferraro'` stop with `'rms_cmb'`. Formulas:
[Magnetic field equations](magnetic-equations.md#normalisation).

## With a differential rotation

With `diff_rot = 1` the fluid rotates differentially ([Background model](background.md#rotation)), and the shear acts on
magnetic field lines in two ways:

- It shears the **perturbation** field. kore includes this in the linear problem, for any rotation profile.
- It **winds up the background** poloidal field: it stretches poloidal field lines around the axis into an azimuthal
  field, which grows until magnetic diffusion balances the winding. kore holds the background fixed and does not follow
  this growth. So the background is steady only if nothing is wound up, or if it already holds the azimuthal field at
  which the winding is balanced; otherwise kore can only assume that something outside the model keeps it steady.

This section says when the background is steady, and gives four set-ups that make it so.

### The Ferraro condition

A differential rotation winds up a poloidal field line unless the rotation rate is the same all along it. So nothing is
wound up when the rotation rate is constant along every poloidal field line: this is Ferraro's law of isorotation. A
toroidal field always satisfies it; a poloidal field usually does not.

At the start of a run with a poloidal field and `diff_rot = 1`, kore prints how far the field and the rotation are from
this condition, as a relative number ([definition](magnetic-equations.md#the-ferraro-condition)). Above `1e-7` it prints
a warning and carries on; to make it stop instead, set `ferraro_policy = 'refuse'` (default `'warn'`). With
`B0_wound = 1` it prints the residual of the wound field's balance instead.

### Four ways to a steady background

Three set-ups avoid the winding by making the rotation rate the same along every field line: kore fits the field to the
rotation (`B0 = 'ferraro'`) or the rotation to the field (`diff_rot_type = 'ferraro'`), or you place the field where the
rotation is uniform (`B0 = 'confined'`). The fourth, `B0_wound = 1`, accepts the winding: it keeps a field that the
shear crosses, and adds the azimuthal field that the shear winds up from it.

| set-up | the winding | what kore does | other parameters (default) | use it when |
| --- | --- | --- | --- | --- |
| `B0 = 'ferraro'` | avoided | uses a poloidal field whose field lines lie on the surfaces of constant rotation, and keeps the rotation | `B0_ferraro_G` ('w'), `B0_ferraro_nmax` (15, odd) | the rotation is given, and any field that it does not wind up will do |
| `diff_rot_type = 'ferraro'` | avoided | keeps the field, and replaces the rotation by one that is constant along each field line | `diff_rot_F` ('psi'), `diff_rot_param`, `diff_rot_nmax` (8, even) | the field is given (e.g. a model of an observed or a dynamo field), and the rotation only has to fit it |
| `B0 = 'confined'` | avoided | nothing: the field sits in a layer where the rotation is uniform | `B0_conf_r1` (0.0), `B0_conf_r2` (0.6) | the field lives away from the shear (e.g. below the tachocline) |
| `B0_wound = 1` | balanced by diffusion | keeps the field and the rotation, and adds the azimuthal field that the shear winds up from the poloidal field | `B0_wound_eta` (1.0) | both are given and do not match (e.g. a tachocline) |

**`B0 = 'ferraro'`: field lines on the surfaces of constant rotation.** This set-up avoids the winding. The rotation
winds up any poloidal field line that crosses its surfaces of constant rotation, so kore uses a poloidal field whose
field lines lie on these surfaces. Nothing is wound up, no azimuthal field is created, and by itself the field puts no
torque on the rotation. The rotation is kept as given. The field lines are set by the rotation, and `B0_ferraro_G`, a
function of the rotation rate `w`, sets the magnetic flux that passes between the axis and each of these surfaces. Where
the rotation is uniform the field vanishes, so this field lives only where the rotation varies. `B0_ferraro_G` should
vanish on the values the rotation takes on the axis and, for a field that stays inside the fluid, at the outer wall;
kore prints how well these hold. A toroidal part of even degree can be added. `cnorm` must be a number, `'mag_energy'`
or `'Schmitt2012'`.

**`diff_rot_type = 'ferraro'`: rotation constant along the field lines.** This set-up also avoids the winding, the other
way round: kore keeps the poloidal field (any `B0` type) and replaces the rotation profile by one that is constant along
each of its field lines. `diff_rot_F` gives the rotation rate of each field line (times `diff_rot_ampl`, as for every
profile), as a function of the field's flux function `psi`, which labels the field lines (and of `p` =
`diff_rot_param`); the default `'psi'` is the simplest. It needs a poloidal field and cannot be combined with
`B0 = 'ferraro'`. kore's rotation is symmetric about the equator, so with a quadrupolar field `diff_rot_F` must be even
(e.g. a function of `psi**2`); the run stops otherwise.

**`B0 = 'confined'`: a field where the rotation is uniform.** The poloidal field fills a spherical layer and nothing
else. If the rotation is uniform in that layer, as below the solar tachocline, nothing is wound up.

**`B0_wound = 1`: the field wound up by the shear.** This set-up accepts the winding. kore keeps the poloidal field (any
type) and the rotation as given, although the shear crosses the field lines. It adds to the background the azimuthal
field that the shear winds up from the poloidal field, at the strength where magnetic diffusion balances the winding.
This *wound field* is computed at the start of the run and added to any `B0_tor`. Its strength follows from the physics,
not from `cnorm`, which normalises the poloidal part alone: it grows with the shear, is inversely proportional to the
diffusivity, and can exceed the poloidal field many times. `B0_wound_eta` (positive, default 1) is that diffusivity in
units of `MagnetD`; the perturbation still diffuses with `MagnetD`. The wound field's torque would brake the shear; kore
leaves it out, so this background holds only while something maintains the rotation. It needs a poloidal `B0` and
`diff_rot = 1`. Read [Validity and caveats](#validity-and-caveats) before using it.

**`B0 = 'ferraro'` or `B0_wound = 1`?** They describe different states. `B0 = 'ferraro'`, like
`diff_rot_type = 'ferraro'`, describes a field and a rotation that have reached isorotation: the rotation rate is the
same along every field line, and nothing is wound up. `B0_wound = 1` describes a shear that something maintains across
the field lines: the shear keeps winding them up, and diffusion keeps the wound field steady.

**A conducting inner core** must rotate rigidly with the fluid at its surface: the rotation at `ricb` must not depend on
latitude, or the run stops. The method of each set-up is on
[Magnetic field equations](magnetic-equations.md#with-a-differential-rotation).

## Inner core and walls

The perturbation field is matched at each wall to the field beyond it. The mantle is an electrical insulator, possibly
with a conducting layer at its base: a thin one, described by conditions at the wall alone, or one of any thickness whose
field kore solves for. Beyond the conducting parts the field is a potential field. The inner core is an insulator (again
with or without a thin conducting layer), or a conductor whose field kore then solves for together with the rest.

| setting | values (default) | notes |
| --- | --- | --- |
| `mantle` | `'insulator'` (default), `'TWA'` or `'resolved'` | `'TWA'`: a thin conducting layer at the base of the mantle; `'resolved'`: a conducting layer of any thickness (both below) |
| `innercore` | `'insulator'` (default), `'conducting, Chebys'`, `'conducting, Bessel'` or `'TWA'` | need `ricb > 0`; `'conducting, Bessel'` in forced runs only ([Forced runs](#forced-runs)); `'TWA'`: a thin conducting layer on an insulating inner core |
| `N_cic` | 64 | Chebyshev polynomials of the inner-core field with `'conducting, Chebys'` (even, at least 4) |
| `mu_i2o` | 1.0 | magnetic permeability, inner core over fluid |
| `sigma_i2o` | 1.0 | electrical conductivity, inner core over fluid |
| `B0_ic` | `'potential'` | for `'G21 dipole'`, `'Luo_S1'`, `'Luo_S2'` with an inner core (above) |

The magnetic diffusivity of the inner core is `MagnetD` divided by `mu_i2o` and `sigma_i2o`. Where the fluid slips
along a conducting inner core (a stress-free inner wall, `bci = 0`), its tangential electric field, that of its flow
across the background field included, is continued into the core, as along a
[thin conducting layer](#thin-conducting-layer). Matching conditions:
[Magnetic field equations](magnetic-equations.md#boundary-conditions).

### Thin conducting layer

A thin layer that conducts, such as the base of the Earth's mantle, carries currents that couple the field across it.
`'TWA'` (the thin-wall approximation of Roberts, Glatzmaier & Clune 2010, as in the `conductive_IC` branch) models such
a layer, at the base of an insulating mantle or on an insulating inner core, through the boundary conditions of the field
alone: the layer is not resolved and adds no unknowns. It is described by its thickness, conductivity and permeability,
through three numbers per wall:

| setting | default | meaning |
| --- | --- | --- |
| `c_cmb`, `c_icb` | `1e-5`, 0 | thickness times permeability of the layer, relative to the radius of the wall and to the fluid's permeability |
| `c1_cmb`, `c1_icb` | `1e-5`, 0 | conductance of the layer: thickness times conductivity, relative to the radius and to the fluid's conductivity |
| `muw_cmb`, `muw_icb` | 1.0 | permeability of the layer relative to vacuum; `np.inf` drops the terms of first order in the thickness |
| `mu` | 1.0 | permeability of the fluid relative to vacuum; it enters the thin-wall conditions only |
| `B0_cmb` | `'wall'` | how the background field meets a thin layer at the outer wall: `'wall'` or `'plain'` (below) |

With `c` and `c1` zero and `mu = 1` the layer is an insulator. A layer with the permeability of vacuum (`muw` 1, the
default, as at the Earth's mantle) lets a steady field through unchanged, and its conductance `c1` then sets nearly all
of its effect.
`mu` other than 1 needs a thin layer or a conducting inner core at every wall of the fluid.

- **Validity.** The approximation holds while the layer is thin against its magnetic skin depth at the frequency of the
  mode. spin_doctor prints the estimated relative error of the layer's effect, εTWA (the larger of the two walls), and
  writes both to `magnetic.dat`; keep it small, say below 1%.
- **A fluid slipping along the wall** (`bco = 0`, `bci = 0` or `ViscosD = 0`). The fluid's tangential electric field,
  including that of the flow across the background field, is continued into the layer in the frame of the solid
  (condition C.I of Rekier et al. 2023), which turns with the mantle with the [rotating solids](rotating-solids.md). The slip stands
  for a thin viscous layer that is not resolved, which must be thin against the magnetic skin depth: `assemble.py`
  prints Pm times the frequency in rotation units (that of the target, or `forcing_frequency` in a forced run), which
  should be small. A [drag](drag.md) at the outer wall changes nothing here.
- **With a differential rotation** the solid turns with the fluid next to it, so the rotation must be uniform in
  latitude at the wall: the run stops otherwise.
- **`B0_cmb`.** A steady background field drives no current in the layer, but it must meet the layer's condition at the
  outer wall. `'wall'` (default) adds a potential field to the fields of internal currents (`'dipole'`, `'G21 dipole'`,
  `'Luo_S1'`, `'Luo_S2'`, `'FDM'`) so that they do; it changes nothing for `muw_cmb` 1 and `mu` 1. `'plain'` keeps them as
  they are. The other fields are used as given.

The conditions and their checks: [Magnetic field equations](magnetic-equations.md#thin-conducting-layer).

### Resolved conducting mantle

`'TWA'` stands for a layer much thinner than the depth to which the field of a mode enters it, its magnetic skin
depth. A thicker or better conducting layer, or a slower mode, needs the field inside the layer itself:
`mantle = 'resolved'` puts a conducting layer between the outer wall and `r_surface`, with an insulator beyond, and solves
for its field together with the rest. The layer is solid and at rest (unless a forcing moves the outer boundary:
[A boundary that moves](#a-boundary-that-moves)), and its conductivity may vary with radius.

| setting | default | meaning |
| --- | --- | --- |
| `r_surface` | 1.83 | outer radius of the layer, the outer wall being at 1 (1.83 is the whole mantle of the Earth) |
| `N_mantle` | 64 | Chebyshev polynomials of the layer's field (at least 4) |
| `sigma_mantle` | 1.0 | conductivity of the layer relative to the fluid's: a number, or a numpy expression of `r` and `p` such as `'1e-2*exp(-p*(r - 1))'` |
| `sigma_mantle_param` | 0.0 | the `p` of that expression |

- **Resolution.** The run stops if `N_mantle` polynomials do not resolve the conductivity profile; a step or a kink is
  never resolved, so smooth it. The field in the layer needs enough of them too, which a second run with a larger
  `N_mantle` shows.
- **A fluid slipping along the layer** (`bco = 0` or `ViscosD = 0`) gets the same terms as along a thin layer, with the
  layer's conductivity at the wall.
- **With a differential rotation** the mantle turns with the fluid at the outer wall, so the rotation must be uniform in
  latitude there: the run stops otherwise.
- **A thin layer** of thickness h and conductivity σ tends to `mantle = 'TWA'` with `c_cmb` = h and `c1_cmb` = σh as h
  decreases, also under a forcing that moves the outer boundary, in both pictures of
  [A boundary that moves](#a-boundary-that-moves).
- The layer and the fluid have the permeability of vacuum (`mu = 1`).

The equations and their checks: [Magnetic field equations](magnetic-equations.md#resolved-conducting-mantle).

### Current sheets at slipping walls

A background field that does not meet the field beyond an insulating wall ends there in a sheet of current: a toroidal
field that is not zero at the wall, or a poloidal one that does not meet the field beyond. `'general'` and `'ferraro'`
need not; where the permeability jumps at the wall (`mu` other than 1, or a thin layer with `c` other than 0 and `muw`
other than 1), the other fields of sources inside the wall meet it only at the outer wall, where `B0_cmb = 'wall'`
adjusts them. A conducting inner core, the resolved mantle or a conducting thin layer carries the field on,
with no sheet in the fluid. The sheet pushes the fluid along the wall wherever the perturbation's field crosses the
wall. A no-slip wall takes that push. Along a stress-free wall (`bco` or `bci` 0 or 2) the fluid's viscous stress takes
it, and kore includes it in the conditions there
([Magnetic field equations](magnetic-equations.md#current-sheets-at-slipping-walls)).

- **Inviscid fluid.** It has no stress to take the push, so a sheet stops an inviscid run. Use a field that meets the
  wall (`B0_tor = 'wall'`, a poloidal field that matches the potential field or, where the permeability jumps, one that
  `B0_cmb = 'wall'` adjusts), a viscous fluid, or a conductor beyond.
- **`centrifugal = 1`.** A sheet at a slipping wall stops the run.
- **Fields that go on through the wall.** `'axial'`, imposed from beyond the outer wall, and at the inner wall
  `'dipole'` and the fields of `B0_ic = 'frozen currents'`, whose sources lie in the core, go on through the wall with
  no sheet. Where the permeability jumps at a slipping wall they stop: the field beyond then depends on how the jump
  bends the imposed field or meets the sources, which kore does not model.

## The field without the flow

Two tools leave the flow out: one checks the walls and the inner core against exact results, the other gives the field
that a known flow induces.

**Free decay: `hydro = 0`.** Without the momentum equation the field only diffuses, and is carried by the differential
rotation with `diff_rot = 1`. Each mode then decays at a rate that the conductivity, the walls and the inner core alone
set, which for a uniform conductivity is known exactly ([Checks](magnetic-equations.md#free-decay-without-the-flow-2026-11-06)).
The background field does no work, but `B0` still sets the equatorial symmetry of the perturbation, as with the flow.
With `thermal = 1` the same switch gives the free decay of the entropy. Only eigenvalue problems can leave the flow out.
`solve.py` writes no flow files and normalises each eigenvector to a magnetic energy of 1 (plus the entropy's with
`thermal = 1`), without `Hendrik`;
`spin_doctor` prints resid𝐛 and resid𝐄, for any walls, and none of the flow's columns.

**Kinematic induction.** `bin/just_induction.py` and `bin/solve_ind.py` give the field that a given flow induces in B0,
without the force of that field on the flow. Assemble a magnetic eigenvalue problem (`submatrices.py`, `assemble.py`),
put the flow in `flow.npy` or its text files ([Output files](output.md#solution-files)), written by a run with the
same `N`, `lmax`, `m`, `symm` and `ricb` (a forced run of the flow alone, for instance), then run

```shell
./bin/just_induction.py [lambda]
mpiexec -n 4 ./bin/solve_ind.py
```

The flow varies as $e^{\lambda t}$: `lambda` is the optional argument, a complex number such as the eigenvalue of a mode,
and its default is $i\,$`forcing_frequency`, the time dependence of the forced runs (a steady flow when it is 0). The
field is written as `magnetic.npy` (and the files of a conducting inner core or of a resolved mantle), one column per
column of the flow.

## Forced runs

A forced run gives the response of the flow and the field to a forcing at one frequency, such as a tide, a libration of
the walls or their steady rotation (`forcing` other than 0; [Tidal forcing](tidal-forcing.md), [Libration](libration.md),
[Thermal forcing](thermal-forcing.md), [Spherical Couette flow](couette-flow.md), [Convective forcing](forcing.md)).
With a field every forcing but the wave-like tide (forcing 1, whose equilibrium tide is that of a fluid without a field)
runs as without one: the field is solved with the flow at the forcing frequency, with any of the walls and inner cores
above (insulating walls for the eccentricity tide, forcing 2), and spin_doctor checks the budgets of the solution. The
solution is written as solved, with the amplitude that the forcing sets.

- **A conducting inner core at one frequency.** At a single frequency the field inside a uniform conducting inner core
  is a spherical Bessel function of the radius. `innercore = 'conducting, Bessel'` uses it, so the core adds conditions
  at the inner wall but no unknowns, and gives the solution of `'conducting, Chebys'` without `N_cic`. Since that field
  depends on the frequency, it serves the forced runs at `forcing_frequency` only. With a differential rotation the core
  turns with the fluid at the inner wall, and the frequency is that in its frame.
- **Budgets.** resid𝐄 includes the work of the pressure, with its magnetic part, where the forcing moves a boundary,
  and the work of a conducting solid that the boundary moves on the field
  ([Energy budgets](magnetic-equations.md#energy-budgets)).
- **Walls that turn.** Where the forcing turns a no-slip wall about the axis, in the libration (forcing 7, `m` 0) and the
  steady rotation of the walls (forcing 12), a conducting solid behind the wall turns with it. A rotation about the axis
  leaves the axisymmetric B0 as it is, so the solid induces no field itself: the field comes from the shear and the
  circulation of the fluid, and diffuses into the solid. In the frame of the mantle (forcing 8) the mantle is at rest,
  and the field is that of forcing 7.
- **Kinematic induction** works in the directory of a forced run too: `just_induction.py` without `lambda` takes the
  forced matrix as it is, at `forcing_frequency`. Another `lambda` needs `B.npz`, which forcing 11 alone writes.

The checks: [Magnetic field equations](magnetic-equations.md#forced-runs-2026-11-06).

### A boundary that moves

Forcings 6, 9, 10 and 14, and 7 with `m` 2, stand for a boundary that moves or is not spherical. The walls stay spheres,
and the forcing prescribes the radial flow at the wall instead. A conducting solid behind such a wall moves too, and the
currents in it depend on how it moves across B0. `forced_walls` sets that motion:

| `forced_walls` | the solid behind the wall | suits |
| --- | --- | --- |
| `'rigid'` | is at rest, and only its shape moves past the sphere: the flow passes through the wall | a triaxial mantle that librates (7 with `m` 2), the flattened inner core of forcing 6 |
| `'deforming'` | deforms with the boundary: its surface moves with the fluid across the wall, and its inside moves with it | a boundary that moves (9, 10), a tide (14) |
| `None` (default) | `'rigid'` for forcings 6 and 7, `'deforming'` for 9, 10 and 14 | |

A deforming solid moves in the simplest way that matches the wall:

- a resolved mantle moves radially only, with a speed that falls as $1/r^2$ from that of the wall, which keeps the
  volume of every shell;
- a conducting inner core (`'conducting, Chebys'`) moves with the simplest flow that is regular at its centre and has no
  motion along the wall;
- with `'TWA'` the solid moves with the fluid across the wall (and along it where the wall is no-slip), and so does the
  thin conducting layer at its surface, the only part of it that conducts.

Its motion across the background field, continued into the solid from the wall, induces a field there, which kore
solves for with the rest, and spin_doctor adds its work to resid𝐄. In a conducting inner core the background is continued
as the simplest field that is regular at the centre, so `'dipole'`, whose source is at the centre, becomes there a field
with currents. With a poloidal field and `'TWA'`, `'deforming'` gives the solutions of the `conductive_IC` branch. Where
a no-slip wall also moves along itself, as in a libration, the solid turns with it in both pictures. With insulating
solids `forced_walls` changes nothing. `'deforming'` does not take `'conducting, Bessel'` behind a wall that moves, since
the core's field is then no longer a Bessel function. The eccentricity tide (forcing 2) also moves its walls along
themselves, which a conducting solid behind them would follow by deforming: it needs insulating walls.

In both pictures a wall that moves, where B0 reaches it, needs the same permeability on both sides: a surface where the
permeability jumps would carry magnetized matter through B0 as it moves. So `mu` must be 1, and `mu_i2o` 1 with a
conducting inner core. A thin layer's `muw_cmb` or `muw_icb` must be 1 where the poloidal part of B0 reaches the wall if
the layer has a thickness (`c` not 0) or conducts at rest (`'rigid'`), and where the toroidal part reaches it if the
layer conducts (`c1` not 0). Otherwise it takes any value, as in an insulating layer (`c` and `c1` both 0) or a
conducting layer without a thickness (`c` 0) that deforms with the boundary under a poloidal B0. Walls that stay in
place take any permeability.

The equations: [Magnetic field equations](magnetic-equations.md#forced-runs).

## What is supported

kore checks the combination of options at startup and stops on any it cannot handle.

Supported: a shell or a full sphere (`ricb = 0`); any `structure`; eigenvalue problems and forced runs; with or without a differential
rotation; `ViscosD > 0`, or `ViscosD = 0` in a full sphere; with or without the flow (`hydro = 0`, above); insulating
walls, a thin conducting layer, a resolved conducting mantle or a conducting inner core; with rotating solids
(`rotdyn = 1`, [Rotating solids](rotating-solids.md)). The magnetic diffusivity of the fluid is uniform.

Not supported (the run stops at startup):

- no background field at all (`B0 = None` and `B0_tor = None`), or `Le` or `Em` zero;
- `ViscosD = 0` with an inner core (`ricb > 0`);
- a conducting inner core in a full sphere, with zero radial stress at the inner wall (`bci = 2`), with an `mu_i2o` or
  a `sigma_i2o` that is not a positive number, with an odd `N_cic` or one below 4 (`'conducting, Chebys'`), with a
  rotation that varies with latitude at its surface, or with `centrifugal = 1` and an `mu_i2o` other than 1 or a fluid
  slipping along it (`bci = 0`) across a poloidal field;
- `'conducting, Bessel'` in an eigenvalue problem, in a frequency scan of forcing 11 (`rsf_frequencies`), or in
  `just_induction.py` with a `lambda` other than $i\,$`forcing_frequency`; `just_induction.py` with such a `lambda` in a
  forced run other than forcing 11;
- a thin conducting layer on the inner core in a full sphere, with `centrifugal = 1`, with zero radial stress at the wall
  (`bco = 2` or `bci = 2`), or with a rotation that varies with latitude at the wall;
- negative `c_cmb`, `c1_cmb`, `c_icb` or `c1_icb`, a `muw_cmb`, `muw_icb` or `mu` that is not positive, or `mu` other
  than 1 without a thin layer or a conducting inner core at every wall;
- a resolved mantle with `mu` other than 1, with `centrifugal = 1`, with zero radial stress at the outer wall
  (`bco = 2`), or with a rotation that varies with latitude at the outer wall; an `r_surface` that is not above 1, an
  `N_mantle` that is not an integer of at least 4, or a `sigma_mantle` that is not positive everywhere in the layer or
  that `N_mantle` polynomials do not resolve;
- with `forced_walls = 'deforming'` (the default of forcings 9, 10 and 14), `'conducting, Bessel'` behind an inner wall
  that the forcing moves radially (forcing 6, and 9, 14 and 7 with `m` 2 where `forcing_amplitude_icb` is not 0);
- the libration of the inner core (forcing 7 with `m` 0, or forcing 8, and `forcing_amplitude_icb` not 0) behind a
  stress-free inner wall (`bci = 0`) along a conducting inner core or a conducting thin layer on it, across a poloidal
  field;
- the eccentricity tide (forcing 2) with a conducting solid behind a wall: a resolved mantle, a thin layer that conducts
  (`c1_cmb` or `c1_icb` not 0), a conducting inner core; and the wave-like tide (forcing 1) with a field;
- a current sheet of the field (a toroidal field that is not zero at the wall, or a poloidal one that does not meet the
  field beyond) at the wall of an inviscid full sphere, or at a slipping wall with `centrifugal = 1`, with an insulator
  or a thin layer that does not conduct (`c1_cmb` or `c1_icb` 0) beyond
  ([Current sheets at slipping walls](#current-sheets-at-slipping-walls));
- `'axial'` at a slipping wall, or `'dipole'` or `B0_ic = 'frozen currents'` at a slipping inner wall, where the
  permeability jumps (`mu` other than 1, or a thin layer whose `c_cmb` or `c_icb` is not 0 and whose `muw_cmb` or
  `muw_icb` is not 1), with an insulator or a thin layer that does not conduct beyond;
- a permeability other than 1 at a wall that the forcing moves radially and that B0 reaches: `mu`, `mu_i2o` with a
  conducting inner core, and a thin layer's `muw_cmb` or `muw_icb` where the part of B0 that it acts on reaches the wall
  ([A boundary that moves](#a-boundary-that-moves)); the outer wall moves under forcing 10, and under 9, 14 and 7 with
  `m` 2 where `forcing_amplitude_cmb` is not 0, the inner wall under 6, 9, 14 and 7 with `m` 2 where
  `forcing_amplitude_icb` is not 0;
- a poloidal and a toroidal part of opposite equatorial symmetry;
- the limits of the options above:
    - `'dipole'` in a full sphere;
    - degrees other than 1 or 2 for `'FDM'`, `'confined'` and `B0_tor_l`, and `'Malkus'` of degree 2 in a full sphere;
    - a `'confined'` layer that is empty or does not lie in the fluid, or `B0_conf_p` below 3;
    - `B0_tor = 'user'` without `B0_tor_profile`, or `'general'` without profiles or with degrees that are not positive
      integers of one parity; a `'user'` or `'general'` profile that is not finite at a point where it is interpolated
      (`r` = 0 among them in a full sphere, where an expression that is 0/0 needs another form);
    - an expression (`B0_tor_profile`, `B0_profiles`, `B0_tor_profiles`, `B0_ferraro_G`, `diff_rot_F`,
      `sigma_mantle`) that is empty or contains `#`;
    - a `cnorm` other than a number, `'rms_cmb'`, `'mag_energy'` or `'Schmitt2012'`, or `'rms_cmb'` with `'confined'`,
      `'general'` or `'ferraro'`; a `cnorm_tor` other than a number, `'mag_energy'` or `'Schmitt2012'`; a field that is
      zero with `'mag_energy'` or `'Schmitt2012'`;
    - `B0 = 'ferraro'` without `diff_rot = 1`, with `diff_rot_type = 'ferraro'`, with a `B0_ferraro_G` that is not
      finite on the values the rotation takes, or with `B0_ferraro_nmax` even or below 1;
    - `diff_rot_type = 'ferraro'` without a poloidal field, with a `diff_rot_F` of the wrong equatorial symmetry, or with
      a negative or odd `diff_rot_nmax`;
    - `B0_wound = 1` without a poloidal field or without `diff_rot = 1`, with `B0_wound_eta` not positive, or with a
      mantle other than `'insulator'` or a thin layer on the inner core;
- with `ferraro_policy = 'refuse'`, a field and a rotation further than `1e-7` from the Ferraro condition;
- a value of `B0`, `B0_tor`, `mantle`, `innercore`, `mag_couplings`, `B0_ic`, `B0_cmb`, `forced_walls` or
  `ferraro_policy` other than those listed on this page.

## Output

A magnetic run writes the perturbation field next to the flow, and spin_doctor checks each mode with the magnetic energy
budgets.

`solve.py` writes `magnetic.npy` (and `magnetic_ic.npy` for a conducting inner core, `magnetic_mantle.npy` for a
resolved mantle; `'conducting, Bessel'` writes no inner-core field; [Output files](output.md#solution-files)). Each eigenvector is normalised so that the kinetic
energy plus `Hendrik` times the magnetic energy is 1 in the fluid (the magnetic energy alone without the flow,
`hydro = 0`); a forced solution is written as solved.

`spin_doctor` adds the Lorentz work to residσ and residω, and prints two more residuals: resid𝐛, the energy budget of
the induction equation in the fluid, and resid𝐄, the full energy budget including the field outside the fluid, in thin
layers, in a resolved mantle and in rotating solids (not with thin layers and `mu` other than 1, with stress-free walls
and `centrifugal = 1`, or with rotating solids unless `vtrq` and `mtrq` are 1). Like residσ ([Solver](solver.md)), they
should be small for a resolved mode; with a thin layer whose `c` is not zero resid𝐄 closes only to the order of the
approximation, about εTWA.
It also prints Mag/Kin, the ratio of magnetic to kinetic energy (`Hendrik` ME/KE, both in the units of the kinetic
energy), εTWA with a thin layer, and writes `magnetic.dat` and the magnetic columns of `params.dat`
([Output files](output.md#magneticdat)).

## Practical tips

The field, more than the flow, usually sets the resolution and the memory a run needs.

- **Resolution.** Where a poloidal field crosses a no-slip wall, a thin magnetic boundary layer (a *Hartmann layer*)
  forms. It, not the bulk flow, usually sets N, and it gets thinner at lower Ekman numbers. A toroidal field alone has
  no Hartmann layer and converges faster. A full sphere needs about twice the N of a shell. Always compare the residuals
  at two values of N; examples are in the [checks](magnetic-equations.md#checks).
- **`'confined'`** converges more slowly than the other fields, since it is smooth only to a finite order at its edges,
  and most slowly when its layer reaches the centre of a full sphere.
- **`tol_B0`** (default `1e-8`) is the relative cut of the radial series of the background field. It sets the band width
  of the magnetic blocks and the fill-in of the LU factors. Lower it for more accurate operators.
- **`mag_couplings`** chooses how the couplings are built. `'closed'` (default) uses closed formulas, which exist for
  fields of degree 1 and 2. `'algebra'` builds them for any degree (`bin/mag_algebra.py`); it is used automatically for
  `'general'` (poloidal or toroidal), `'ferraro'` and `B0_wound = 1`. Both give the same eigenvalues to round-off.
- **Memory.** Fields of high degree, and wound fields with a high-degree rotation, couple more degrees and need more LU
  memory. If MUMPS stops with INFOG(1) = −9, raise `st_mat_mumps_icntl_14` in `petsc_opts` ([Solver](solver.md)).

## Validity and caveats

kore holds the background field and the background flow fixed. This section says what that leaves out and when it
matters. Two effects that would change the background are left out:

- **Ohmic decay.** A field that carries electric current would decay by diffusion; kore holds it fixed, with or without
  rotation. Only the wound field is steady by itself, since the winding balances its diffusion.
- **The Lorentz force of the background field on the background flow.** The rotation is imposed and is not changed by
  the field, and the thermal-wind entropy is computed from the rotation alone. With `B0_wound = 1` this force is
  dominated by the wound field, which can be large.

### When the field leaves the thermal wind unchanged

With `thermal_wind` 1 or 2, the latitudinal entropy of the background balances the differential rotation alone
([Background model](background.md#thermal-wind)). The force of the background field would enter the same balance. The
derivations and the estimates are in
[Magnetic field equations](magnetic-equations.md#when-the-ignored-lorentz-force-matters).

**Exact: a field without current.** A poloidal field that carries no electric current in the fluid, with no toroidal
part, exerts no force, so the thermal wind is exact. If the rotation also satisfies the Ferraro condition, the field
does not wind up or decay, and the whole background is steady. `B0 = 'axial'` and `'dipole'` are such fields, and so is
a `'general'` field whose profile of each degree n combines `r**n` and, in a shell, `r**-(n+1)`.

**Approximate: any other field.** All other fields carry current: the other poloidal types, any `B0_tor`, and the wound
field. The background is then a good approximation under three conditions.

1. *The meridional force is weak.* The field's force in the meridional planes (magnetic pressure, the hoop stress of the
   azimuthal field, the tension of curved poloidal lines) adds to the thermal-wind balance. It is negligible when

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
    directly, but the rotation must be maintained against it. The torque is negligible when this rate is small compared
    with the mode frequencies, their damping rates, and the rate of whatever maintains the rotation. If viscous or
    turbulent stresses hold the rotation, it is tenable only for a Hartmann number of order one or less:

    $$
    Ha = \frac{\mathrm{Le}}{\sqrt{\rho_0\,\mathrm{Ek}\,\mathrm{Em}\,\eta_w}} \lesssim 1 .
    $$

3. *The field is steady.* The Ferraro condition holds, or `B0_wound = 1` balances the winding. The wound field can be
   much stronger than the poloidal field that makes it, so it must meet condition 1 too. The Ohmic decay of the field
   must be slow compared with the modes.

Where the rotation is uniform there is no thermal wind. In a stably stratified region the field's force is then balanced
by a small change of the stratification, which kore leaves out. It is negligible when the Alfvén frequency is far below
the buoyancy frequency.

**What condition 1 means for the modes.** The error it leaves in the frequencies is about half the left side of
condition 1, in units of the rotation rate, for modes that feel the baroclinicity (more for slower modes). The field's
direct effect on a mode, which kore includes in full, is larger by a factor that grows as the square of the number of
the mode's wavelengths along the field lines. So a field can change the modes while leaving the thermal wind unchanged.

### Wound fields and the tachocline

A `B0_wound` background is a snapshot: it holds only for modes faster than the braking and the decay above. This is the
tachocline confinement problem (Gough & McIntyre 1998; MacGregor & Charbonneau 1999). With molecular diffusivities, even
a weak poloidal field in the radiative interior has a large Hartmann number, and its torque would impose isorotation
along its field lines and spread the shear of the convection zone inward. A wound field that strong would brake the
shear that makes it, as self-consistent tachocline models include.

A field below the tachocline is better represented by a `'confined'` field whose layer lies there (`B0_conf_r2` below
the tachocline): the rotation is uniform there, so the Ferraro condition holds. A `B0 = 'ferraro'` field cannot be put
there, since it follows the surfaces of constant rotation and vanishes where the rotation is uniform. A toroidal
`B0_tor` can be added for the tachocline and the convection zone, since a toroidal field is never wound up.

The checks of this code, with a Hartmann number near 30 and a braking time shorter than a rotation period, test the
mathematics (closed form, residuals, budgets), not the physical validity of such a background.

### The Sun, Jupiter and Saturn

Orders of magnitude with published values, where the field and the shear meet. The numbers and references are on
[Magnetic field equations](magnetic-equations.md#the-sun-jupiter-and-saturn).

| region | condition 1: meridional force | condition 2: torque | the background |
| --- | --- | --- | --- |
| Sun, convection zone | holds: a mean field of a few thousand G in the lower half gives a ratio of 10⁻³ or less, and the mean field is weaker higher up | the weak mean poloidal field (about 10 G) brakes the rotation over a century or more; Hartmann number of order one with turbulent values | consistent; the rotation does not follow poloidal field lines, so use a toroidal `B0_tor`, or `B0_wound` from a weak poloidal field |
| Sun, tachocline | holds for 10⁴ G (ratio about 0.005); fails near 10⁵ G (ratio about 0.5) | set by the poloidal field and the diffusivity ([above](#wound-fields-and-the-tachocline)) | consistent up to about 2×10⁴ G |
| Sun, radiative interior | no thermal wind: the rotation is uniform | the Ferraro condition holds | consistent with a `'confined'` field whose layer lies below the tachocline |
| Jupiter, winds (above 0.96 R) | holds: ratio about 10⁻⁹ | about a century | consistent |
| Saturn, winds (above 0.86 R) | holds: ratio about 10⁻¹² | about 10⁴ years | consistent |

The Sun is the case for magnetic effects on inertial modes. At the base of its convection zone a 10⁴ G field has a
Lehnert number of a few hundredths: enough to matter for modes with several wavelengths along the field, while condition
1 still holds. In Jupiter and Saturn the Lehnert number is 10⁻⁵ to 10⁻⁷. Their fields change neither the thermal wind
nor the large-scale inertial modes measurably: kore's background is consistent there, and the field's effect on the
modes is negligible.

### Checklist for solar inertial modes

Solar inertial modes have periods longer than a rotation (Löptien et al. 2018; Gizon et al. 2021). This makes the
criteria above stricter: the effect of the field on a mode grows as its frequency falls, and the background must stay
steady over the lifetime of the modes.

- **Time scales.** The braking rate and the Ohmic decay rate of the poloidal field must be small compared with the
  damping rate of the modes, not only with their frequency. The wound field needs a diffusion time across its layer to
  build up (months to years for a thin tachocline with a turbulent diffusivity): on solar-cycle time scales it is a
  snapshot.
- **Sensitivity.** The effect of the field grows roughly as the square of the Alfvén frequency over the mode frequency,
  so low-frequency modes feel it most. Scan `B0_wound_eta` and `Le` rather than trusting one value.
- **Critical layers and continua.** Where the mode frequency matches the local advection by the rotation (the critical
  latitudes) and at Alfvén resonances, the spectrum has continuous parts and the eigenfunctions thin layers. Check
  convergence in N and lmax, and tell discrete modes from the continuum.
- **Thermal wind.** Keep the thermal-wind entropy on (`thermal_wind` 1 or 2) and check condition 1 above: the background
  baroclinicity does not include the forces of the field.
- **Consistent diffusivities.** The observed line widths constrain the turbulent viscosity (`Ek`). Take a turbulent
  magnetic Prandtl number near one (`Em` close to `Ek`), and build the wound field with the same turbulent diffusivity
  (`B0_wound_eta = 1`). A wound field built with a molecular diffusivity under modes damped by turbulence is
  inconsistent.
- **Radiative interior.** With molecular values the Hartmann number there is huge, so a field there must not cross a
  shear: its torque would remove the shear (Ferraro's mechanism). Keep the field below the tachocline, where the
  rotation is uniform: use `'confined'` with its layer there, not `B0_wound`.
- **Where the modes live.** Most observed modes are concentrated in the convection zone (many near the surface or the
  equator), so a tachocline field reaches them only through their tails. Test the high-latitude and deep modes first.
- **Measurability.** A frequency shift matters only above about 1 nHz, a few thousandths of the rotation rate.
- **Symmetry.** A dipolar poloidal field with kore's equatorially symmetric rotation gives a wound field of the same
  family, so the two mode classes stay separate, as without the field.
- **Practical.** The wound field has degrees up to the sum of those of the rotation and of the poloidal field, plus one,
  so a high-degree `'tabulated'` rotation means wide couplings and more memory (see [Practical tips](#practical-tips)).
  In each run read the printed residual of the induction balance.
- **Field geometry.** The field inside the Sun is not known, and each set-up imposes its own geometry. Compute the same
  modes with more than one (for example a toroidal `B0_tor`, and `B0_wound` from a weak poloidal field) at a similar
  field strength: the spread shows how much the results depend on the field assumed.

## Further reading

- [Magnetic field equations](magnetic-equations.md): the equations, every field profile, the rows, the boundary
  conditions, the four steady set-ups with a differential rotation, the energy budgets, and the checks against
  `conductive_IC` and published results.
- [Derivation tools](derivations.md), part A: how the magnetic rows are derived (`tests/magnetic/derive.py`), fitted
  (`tools/derive/`) and built for any degree (`bin/mag_algebra.py`).
- [Output files](output.md#magneticdat): the columns of `magnetic.dat` and the magnetic columns of `params.dat`.
