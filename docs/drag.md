# Drag at the outer wall

A stress-free outer wall lets the fluid slip freely, and a no-slip wall holds it. A drag lies between the two: the wall
pulls back on the slipping fluid in proportion to the slip, which damps the flow without a viscous boundary layer. kore
has two drags: a constant drag coefficient, and the drag of the wall's topography, which depends on latitude and on the
direction of the flow (Rekier et al. 2025). Either one applies to a stress-free outer wall:

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

`kappa_cmb` sets a drag that is the same everywhere on the wall and in every direction. The slip is the flow along the
wall relative to the mantle, which is at rest or, with the [rotating solids](rotating-solids.md), turns.

For the Earth's core-mantle boundary, Rekier et al. (2025) give a drag coefficient that corresponds to
`kappa_cmb` = 4.5e-6.

## Drag of the topography: `topo_drag = 'rekier'`

A flow along a wall with topography radiates internal waves into a rotating or stratified fluid, and these waves carry
momentum away. The resulting drag depends on latitude, on the direction of the flow (zonal or meridional), on its
frequency, and on the buoyancy frequency at the wall. `topo_drag = 'rekier'` uses the drag that Rekier et al. (2025)
derive for a topography described by its rms height, the wavelength where its spectrum rolls off, and its Hurst
exponent. For the Earth's core-mantle boundary:

```python
par.bco = 0
par.topo_drag = 'rekier'
par.topo_hrms = 4.35e-4
par.topo_lambda0 = 0.431
par.topo_H = 0.8
```

Without buoyancy these values damp the spin-over mode about as much as `kappa_cmb` = 4.5e-6.

- **Buoyancy frequency.** By default, that of the stratification at the wall (0 without buoyancy or where the
  stratification is unstable). `topo_N` sets it instead.
- **Frequency of the flow.** The drag depends on the frequency of the flow, which an eigenvalue problem does not know in
  advance. kore uses the frequency of the solver's target (`itau`), or the forcing frequency in a forced run, unless
  `topo_omega` sets it. For a mode, iterate until the two agree ([below](#the-frequency-of-the-drag)).
- **Coupled degrees.** The drag varies with latitude, so it couples spherical-harmonic degrees. Check the convergence in
  `lmax` as usual.

### The frequency of the drag

`tools/topo_iterate.sh` repeats the solve until the frequency used by the drag matches that of the mode. Each pass
builds the drag at the solver's target, solves, and moves the target to the nearest eigenvalue through the file
`track_target` ([Following a mode](output.md#following-a-mode)). A few passes are usually enough. It needs
`track_target = 1`, `topo_omega = None`, and a first guess in the file. From kore's top folder:

```bash
echo "-0.02 0.99" > track_target    # the guess: Re λ and Im λ
tools/topo_iterate.sh 1e-10         # stops when λ changes by less than 1e-10 (relative); at most 20 passes
```

Each pass appends its results to the `.dat` files, so their last rows belong to the converged mode. The iteration
follows one mode at a time.

The formulas: [Drag at the outer wall equations](drag-equations.md).

## Weak and strong drag

As long as `kappa_cmb` is small compared with the square root of the Ekman number (`ViscosD/Gaspard`), the fluid slips
nearly freely and the drag adds a damping of its own that does not depend on the Ekman number. This is the regime the
drag is meant for. As `kappa_cmb` approaches the square root of the Ekman number, the drag holds the fluid at the wall,
a viscous layer forms, and the wall behaves more and more like a no-slip wall. The Earth value of 4.5e-6 is a weak drag
down to Ekman numbers of about 1e-8.

## With the rotating solids

With `rotdyn = 1` the drag acts on the slip relative to the turning mantle, and pulls the mantle in return; its torque
appears as the viscous torque on the mantle in `rotdyn.dat`.

## Output

For a viscous run with a stress-free outer wall, `spin_doctor` prints two columns:

- **Dtopo**, the power the drag dissipates;
- **δσtopo**, the drag's contribution to the damping rate of the mode.

Without a drag, both are given per unit `kappa_cmb` (columns Dtopo/κ and δσtopo/κ). Multiplied by a small `kappa_cmb`,
δσtopo/κ estimates the extra damping a weak drag would add, without a new run. To get the full effect of the drag, run
with it.

`flow.dat` holds both (columns 40–41), and `params.dat` the drag settings ([Output files](output.md)).

## What is supported

Supported: any structure; a full sphere; any `m`; with or without buoyancy; with or without a magnetic field; the
rotating solids; a constant drag together with differential rotation or a background flow; eigenvalue problems, and
the forced runs whose outer wall stays in place and may slip (forcings 6, 8, 11, 13, and 9 and 12 with the outer wall
at rest).

Not supported (kore stops with a message):

- a drag without the flow (`hydro = 0`), without viscosity, or at an outer wall that is not stress-free;
- a non-rotating frame, the centrifugal deformation, or a forcing that moves the outer wall radially;
- the wave-like tide (forcing 1);
- `kappa_cmb` and `topo_drag` together;
- the drag of the topography with differential rotation or a background flow, with a flow frequency of 0, or in a
  frequency scan of forcing 11 without `topo_omega`.

## Reference

- Rekier, J., Triana, S. A., Barik, A., Abdulah, A. & Kang, S. 2025, arXiv:2507.01671.
