# Rotating solids

An axisymmetric mode of the fluid pulls on the mantle and on the inner core, and their rotation changes in return.
`rotdyn = 1` lets both solids turn as rigid bodies about the rotation axis. They are coupled to the fluid by the viscous
and magnetic torques at the walls, and to each other by gravity.

| setting | default | meaning |
| --- | --- | --- |
| `rotdyn` | 0 | 1: the mantle and the inner core turn under the torques |
| `MoIZ_M`, `MoIZ_IC` | 11.16, 9.2e-3 | axial moments of inertia of the mantle and of the inner core, in units of the density unit times the outer radius of the fluid to the fifth power |
| `gTorque` | 5e-8 | gravitational coupling: the torque between inner core and mantle per radian of misalignment (dimensionless) |
| `OmgtauIC` | 2.3e3 | the time over which the inner core relaxes viscously back into alignment, times the rotation rate; `np.inf` for no relaxation |
| `vtrq`, `mtrq` | 1, 1 | 0 switches off the viscous or the magnetic torque on the solids |

The defaults are Earth-like (2.3e3 corresponds to a relaxation time of one year). A run needs an axisymmetric,
equatorially symmetric eigenvalue problem with an inner core and a no-slip inner wall. The outer wall is no-slip, or
stress-free with or without a drag ([Drag at the outer wall](drag.md)):

```python
par.m = 0
par.symm = 1
par.bci = 1
par.bco = 1
par.rotdyn = 1
```

## How the solids are coupled

Three unknowns join the fields: the rotation rates of the mantle and of the inner core relative to the frame, and the
angle of the inner core relative to the mantle.

- **The walls move with the solids.** The fluid at each no-slip wall turns with the solid beyond it. At a stress-free
  outer wall the fluid slips along the mantle, and a drag (if any) pulls it towards the mantle's rotation.
- **Viscous torque.** The stress of the fluid on each wall (at a stress-free outer wall, only that of the drag).
- **Magnetic torque.** The magnetic stress on whatever conducts beyond each wall: a thin conducting layer, a resolved
  mantle or a conducting inner core. An insulating solid feels none.
- **Gravitational torque.** It pulls the inner core and the mantle back into alignment, while the inner core relaxes
  viscously towards alignment over `OmgtauIC`.

No torque comes from outside, so fluid and solids together conserve their angular momentum. The eigenvalue 0 is the
rigid rotation of everything; ignore its residuals.

The equations: [Rotating solids equations](rotating-solids-equations.md).

## Output

- `rotdyn.npy`: the two rotation rates and the angle, one column per solution, normalized as the eigenvector.
- `rotdyn.dat`: one row per solution with the same three quantities, the gravitational torque, and the viscous and
  magnetic torques on the mantle and on the inner core ([Output files](output.md#rotdyndat)).
- With a field, resid𝐄 also counts the energy of the solids ([Magnetic field](magnetic.md#output)).

## The angular momentum check: resid𝐋

The angular momentum of the fluid changes only through the torques on it. `spin_doctor` computes both from the
solution and prints resid𝐋, the relative mismatch of this balance. Like residσ, it should be small for a resolved
mode. It is computed:

- for eigenvalue problems with `m = 0`, `symm = 1` (axial angular momentum) or `m = 1`, `symm = -1` (equatorial angular
  momentum, which includes the spin-over mode), with or without the rotating solids;
- for forced runs with `m = 0`, `symm = 1` whose force on the fluid is known: libration (forcings 7 and 8), rotating
  walls (forcing 12), thermal forcing (forcing 13), and the wave-like tide (forcing 1).

It is not computed with `centrifugal = 1`, for `m = 1` with differential rotation, or with walls of zero radial stress.
`flow.dat` holds the angular momentum, the torques and resid𝐋 ([Output files](output.md#flowdat)).

## What is supported

Supported: any `structure`; with or without buoyancy; a no-slip or a stress-free outer wall, with or without a drag;
with or without a magnetic field (with any poloidal `B0`, and a toroidal `B0_tor` that vanishes at both walls);
insulating walls, a thin conducting layer, a resolved mantle or a conducting inner core (`'conducting, Chebys'`).

Not supported (kore stops with a message):

- `m` other than 0, `symm` other than 1, a full sphere, forced runs, or no flow (`hydro = 0`);
- an inviscid fluid, an inner wall that is not no-slip, or an outer wall with zero radial stress (`bco = 2`);
- a non-rotating frame (`Gaspard = 0`);
- differential rotation, a background flow or the centrifugal deformation;
- a toroidal `B0_tor` that does not vanish at both walls.
