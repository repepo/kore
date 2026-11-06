# Rotating solids

An axisymmetric mode of the fluid pulls on the mantle and on the inner core, and their rotation changes in return.
`rotdyn = 1` lets both solids turn as rigid bodies about the rotation axis, coupled to the fluid by the viscous and
magnetic torques at the walls and to each other by gravity, as in the `conductive_IC` branch.

| setting | default | meaning |
| --- | --- | --- |
| `rotdyn` | 0 | 1: the mantle and the inner core turn under the torques |
| `MoIZ_M`, `MoIZ_IC` | 11.16, 9.2e-3 | axial moments of inertia of the mantle and of the inner core, in units of the run's density unit times the outer radius of the fluid to the fifth power |
| `gTorque` | 5e-8 | gravitational coupling: the torque between the inner core and the mantle per radian of misalignment, divided by that unit and by the rotation rate squared |
| `OmgtauIC` | 2.3e3 | the time over which the inner core relaxes viscously back into alignment, times the rotation rate; `np.inf` for no relaxation |
| `vtrq`, `mtrq` | 1, 1 | 0 switches off the viscous or the magnetic torque on the solids; the fluid still follows the walls |

The defaults are those of the `conductive_IC` branch for the Earth (2.3e3 is a relaxation time of one year). A run needs
an axisymmetric, equatorially symmetric eigenvalue problem with a no-slip inner wall; the outer wall is no-slip, as
here, or stress-free, with or without a drag ([Drag at the outer wall](drag.md)):

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
  outer wall the fluid slips along the mantle; a drag pulls it towards the mantle's rotation.
- **Viscous torque.** The stress of the fluid on each wall: at a stress-free outer wall, that of the drag, or none.
- **Magnetic torque.** The magnetic stress through each wall, which acts on whatever conducts beyond it: a thin
  conducting wall, a resolved mantle or a conducting inner core. An insulating solid feels none.
- **Gravitational torque.** It pulls the inner core and the mantle back into alignment, in proportion to the angle
  between them, while the inner core relaxes viscously towards alignment over `OmgtauIC`.

No torque comes from outside, so the fluid and the solids together conserve their angular momentum: every mode with a
nonzero eigenvalue carries none in total. The eigenvalue 0 is the rigid rotation of everything, whose residuals mean
nothing.

The equations: [Rotating solids equations](rotating-solids-equations.md).

## Output

- `real_rotdyn.field`, `imag_rotdyn.field`: the two rotation rates and the angle, one column per solution, in the
  normalisation of the eigenvector (the kinetic energy of the fluid, plus `Hendrik` times its magnetic energy, is 1).
- `rotdyn.dat`: one row per solution with the same three, the gravitational torque, and the viscous and magnetic torques
  on the mantle and on the inner core ([Output files](output.md#rotdyndat)).
- With a field, resid𝐄 also counts the rotational energy of the solids, their gravitational energy and the power of
  the torques on them ([Magnetic field](magnetic.md#output)), when both torques act (`vtrq` and `mtrq` 1).

## The angular momentum check: resid𝐋

The angular momentum of the fluid changes only through the torques on it, which `spin_doctor` computes from the fields,
apart from the matrices. It prints resid𝐋, the relative residual of that balance, for eigenvalue problems with the flow
and either `m = 0`, `symm = 1` (the axial angular momentum) or `m = 1`, `symm = -1` (the equatorial one, which the
Coriolis force turns; the spin-over mode is one of these), with or without the rotating solids. It is not computed with
`centrifugal = 1`, nor for `m = 1` with a differential rotation. Like residσ, it should be small for a resolved mode.
`flow.dat` has the angular momentum, the torques and resid𝐋 ([Output files](output.md#flowdat)).

## What is supported

Supported: any `structure`; with or without buoyancy; with or without a field, with any poloidal `B0` and a toroidal
`B0_tor` that vanishes at both walls; insulating walls, thin conducting walls, a resolved mantle or a conducting inner
core.

Not supported (the run stops at startup):

- `m` other than 0, `symm` other than 1, a full sphere (`ricb = 0`), forced runs, or no flow (`hydro = 0`);
- `ViscosD = 0`, an inner wall that is not no-slip (`bci` other than 1), or an outer wall that is neither no-slip nor
  stress-free (`bco = 2`);
- `Gaspard = 0`;
- a differential rotation (`diff_rot = 1`) or `centrifugal = 1`;
- a toroidal `B0_tor` that does not vanish at a wall;
- `MoIZ_M` or `MoIZ_IC` not positive and finite, `gTorque` negative or infinite, `OmgtauIC` not positive, `vtrq` or
  `mtrq` other than 0 or 1, `rotdyn` other than 0 or 1.
