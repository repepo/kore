# Drag at the outer wall

A stress-free outer wall lets the fluid slip along it freely, and a no-slip wall holds it. A drag lies between the two:
the wall pulls back on the slipping fluid in proportion to the slip, and so damps the flow without a viscous layer. kore
has two drags, as the `conductive_IC` branch: a constant drag coefficient, and the drag of the wall's topography, which
depends on latitude and on the direction of the flow (Rekier et al. 2025). Either one changes the condition of a
stress-free outer wall:

```python
par.bco = 0
par.kappa_cmb = 4.5e-6
```

| setting | default | meaning |
| --- | --- | --- |
| `kappa_cmb` | 0.0 | the constant drag coefficient, in units of the fluid's density at the wall times the rotation rate times the outer radius; 0: no drag |
| `topo_drag` | `None` | `'rekier'`: the drag of the topography, instead of `kappa_cmb` |
| `topo_hrms` | 0.0 | the rms height of the topography, over the outer radius |
| `topo_lambda0` | 0.431 | the wavelength below which the spectrum of the topography falls off, over the outer radius |
| `topo_H` | 0.8 | the Hurst exponent of the topography, which sets how fast its spectrum falls off (above 1/2) |
| `topo_N` | `None` | the buoyancy frequency at the wall, over the rotation rate; `None`: that of the stratification |
| `topo_omega` | `None` | the frequency of the flow relative to the mantle, over the rotation rate; `None`: from the run |

## Constant drag: `kappa_cmb`

`kappa_cmb` sets a drag that is the same everywhere on the wall and in every direction: the wall pulls on the fluid with
a stress of `kappa_cmb` times the slip, in the units above. The slip is the flow along the wall relative to the mantle,
which is at rest, or turns with the rotating solids ([Rotating solids](rotating-solids.md)).

For the Earth's core-mantle boundary, Rekier et al. (2025) give a drag coefficient of 11.3 kg m⁻² s⁻¹ at the diurnal
frequency: `kappa_cmb` = 4.5e-6, with a density of 9.9e3 kg m⁻³ at the wall, the Earth's rotation rate and a radius of
3480 km.

## Drag of the topography: `topo_drag = 'rekier'`

A flow along a wall with topography radiates internal waves into a rotating or stratified fluid, and the waves carry
momentum away from the flow. The drag this exerts depends on the latitude, through the Coriolis force, on the direction
of the flow, zonal or meridional, on its frequency, and on the buoyancy frequency at the wall. `topo_drag = 'rekier'`
uses the drag that Rekier et al. (2025) derive for a topography with a power-law spectrum, described by its rms height,
the wavelength where the spectrum rolls off, and its Hurst exponent. The Earth's core-mantle boundary (an rms height of
1.51 km, a roll-off at 1500 km and a Hurst exponent of 0.8, at a radius of 3480 km):

```python
par.bco = 0
par.topo_drag = 'rekier'
par.topo_hrms = 4.35e-4
par.topo_lambda0 = 0.431
par.topo_H = 0.8
```

Without buoyancy these values damp the spin-over mode as `kappa_cmb` = 4.5e-6 does, to 0.3 %.

- **The buoyancy frequency** at the wall is that of the stratification (`thermal = 1`), or 0 without buoyancy or where
  the stratification is unstable. `topo_N` sets it instead.
- **The frequency of the flow.** The drag depends on the frequency of the flow relative to the mantle, which an
  eigenvalue problem does not know in advance: kore takes the frequency of the solver's target (`itau`), or the forcing
  frequency in a forced run, unless `topo_omega` sets it. For a mode found away from the target, set `itau` to its
  frequency and solve again, until the two agree. The run stops if this frequency is 0.
- **Coupled degrees.** The drag varies with latitude, so the wall couples each degree of the flow to the others of the
  same symmetry, and for `m` other than 0 the poloidal flow to the toroidal one. Check the convergence in `lmax` as
  usual.

The formulas: [Drag at the outer wall equations](drag-equations.md).

## Weak and strong drag

kore applies the drag at a stress-free wall, where the fluid slips. As long as `kappa_cmb` is small against the square
root of the Ekman number (`ViscosD/Gaspard`), the fluid slips nearly freely, and the drag adds a damping of its own that
does not depend on the Ekman number: the regime of a drag on the flow outside the thin viscous layer of a real wall, the
one to run in. As `kappa_cmb` approaches the square root of the Ekman number, the drag holds the fluid at the wall, a
viscous layer forms, and the wall acts more and more as a no-slip wall. The Earth's 4.5e-6 is a weak drag down to an
Ekman number of about 1e-8. The same holds for each direction of the drag of the topography.

## With the rotating solids

With `rotdyn = 1` the outer wall may be stress-free, with or without a drag. The drag then acts on the slip relative to
the turning mantle, and pulls the mantle in return: its torque is the viscous torque on the mantle in `rotdyn.dat`.
Without a drag the mantle feels only the magnetic torque and gravity. A fluid slipping along a conducting outer wall
(a thin conducting layer or a resolved mantle, [Magnetic field](magnetic.md#inner-core-and-walls)) induces its field through the
slip relative to the turning mantle too.

## Output

`spin_doctor` prints two columns for a viscous run with a stress-free outer wall (not with `centrifugal = 1`):

- **Dtopo**, the power the drag dissipates, in the units of the energies;
- **δσtopo**, minus Dtopo over twice the energy of the mode (its kinetic energy, its magnetic energy inside and outside
  the fluid, and the energy of the rotating solids): the drag's share of the damping rate.

Without a drag both are per unit `kappa_cmb` (the columns Dtopo/κ and δσtopo/κ): the drag a constant `kappa_cmb` of 1
would exert on the free slip. For a weak drag this estimates the damping it would add, without a run: with the mantle at
rest, `kappa_cmb` times δσtopo/κ is the change of the damping rate of the spin-over mode to 5e-4 of it (Ek 1e-3). A drag
also changes the flow, and with it the other losses: with the rotating solids it turns the mantle too, and the damping
rate of a mode of the solids changed by about half of the estimate. Run with the drag for its effect.

`flow.dat` has both (columns 40–41), and `params.dat` the settings, with the buoyancy frequency and the frequency of the
flow that the drag of the topography used (columns 135–141, [Output files](output.md)).

## What is supported

Supported: any `structure` (the density and the viscosity at the wall enter); a full sphere; any `m`; with or without
buoyancy; with or without a field, with any magnetic walls; the rotating solids; a constant drag with a differential
rotation; eigenvalue problems, and the forced runs whose outer wall does not move (forcings 6, 11 and 13).

Not supported (the run stops at startup):

- a drag without the flow (`hydro = 0`), without viscosity (`ViscosD = 0`), or at an outer wall that is not stress-free
  (`bco` other than 0);
- `Gaspard = 0`, `centrifugal = 1`, or forcings 9 and 10, which move the outer wall;
- `kappa_cmb` and `topo_drag` together;
- the drag of the topography with a differential rotation, with a frequency of the flow of 0, or in a frequency scan of
  forcing 11 (`rsf_frequencies`) without `topo_omega`, since it is built for one frequency;
- `kappa_cmb` negative or not finite, `topo_drag` other than `None` and `'rekier'`, `topo_hrms` or `topo_lambda0` not
  positive, `topo_H` not above 1/2, `topo_N` or `topo_omega` negative.

## Reference

- Rekier, J., Triana, S. A., Barik, A., Abdulah, A. & Kang, S. 2025, arXiv:2507.01671 (the drag of the topography,
  their eqs. 20, 21, B1 and B2).
