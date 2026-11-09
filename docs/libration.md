# Libration

A moon on an eccentric orbit, or a body whose shape is not symmetric about its axis, does not spin at a steady rate:
its mantle turns back and forth around the mean rotation, at the frequency of the orbit. This is libration in
longitude. The fluid inside does not follow the mantle at once: the walls drag it through viscosity (and through the
magnetic field, if there is one), and a mantle whose equator is not round pushes it as it turns. kore returns the flow
that this oscillation drives, and its dissipation.

kore offers two equivalent descriptions:

- **Forcing 7** works in the frame that turns at the mean rate, where the walls move.
- **Forcing 8** works in the frame of the mantle, where the walls are at rest and the fluid feels the acceleration of the
  frame instead.

## Forcing 7: the walls librate

With `m = 0`, the mantle turns back and forth by the angle $A\cos wt$ about the rotation axis, and the inner core by
$A_{IC}\cos wt$, with $A$ = `forcing_amplitude_cmb`, $A_{IC}$ = `forcing_amplitude_icb` (in radians) and
$w$ = `forcing_frequency`. A no-slip wall carries the fluid with it.

Requirements:

- a no-slip outer wall (`bco = 1`) and a viscous fluid;
- the inner wall no-slip or stress-free (`bci` = 1 or 0). A stress-free inner wall does not pass the libration of the
  inner core to the fluid, so $A_{IC}$ then has no effect.

In a full sphere (`ricb = 0`) only the mantle librates.

With `m = 2`, forcing 7 represents a mantle with an elongated equator, which pushes the fluid as its shape librates.
The walls stay spherical, and the forcing imposes the corresponding radial velocity at each wall. This needs no-slip
walls (`bco = 1`, and `bci = 1` with an inner core), or an inviscid fluid in a full sphere. With a magnetic field, see
[A boundary that moves](magnetic.md#a-boundary-that-moves).

## Forcing 8: the frame of the mantle

The frame turns with the mantle, so the outer wall is at rest. The inner core turns with the mantle, or librates
relative to it by the angle $A_{IC}\cos wt$. The oscillation of the frame puts a body force in the fluid, the Poincaré
force,

$$
\mathbf f = w^2 A\cos wt\thinspace r\sin\theta\thinspace\hat{\boldsymbol\phi} .
$$

The flow is that of forcing 7 seen from the mantle. Because the walls are at rest, forcing 8 accepts walls that
forcing 7 cannot move: a stress-free outer wall, possibly with a drag ([Drag at the outer wall](drag.md)), and an
inviscid fluid in a full sphere.

Requirements:

- `m = 0`, `symm = 1` and a non-zero frequency;
- `bco` and `bci` 0 or 1 (an inviscid fluid takes any `bco`);
- in a full sphere, $A_{IC} = 0$.

Use forcing 8 for what an observer on the mantle sees, or for walls that forcing 7 does not accept. Both forcings work
with any structure, buoyancy, differential rotation and a magnetic field ([Magnetic field](magnetic.md#forced-runs)).

## Example: a librating shell in both frames

Write in `bin/parameters.py`

```python
from params_default import default_params

par = default_params()

par.timescale = 'rotation'
par.Ek = 1e-4
par.set_scales()                  # Gaspard = 1, ViscosD = Ek

# incompressible fluid in a rotating shell, no-slip walls
par.structure = 'Boussinesq'
par.ricb = 0.35
par.bci = 1
par.bco = 1

# axisymmetric and equatorially symmetric
par.m = 0
par.symm = 1

par.N = 96
par.lmax = 95
par.ncpus = 2

# the mantle librates by the angle cos(w t) at w = 1, the inner core with it
par.forcing = 7
par.forcing_frequency = 1.0
par.forcing_amplitude_cmb = 1.0
par.forcing_amplitude_icb = 1.0
```

and run

```shell
./bin/submatrices.py 2
mpiexec -n 2 ./bin/assemble.py
mpiexec -n 2 ./bin/solve.py
./bin/spin_doctor.py 2
```

It takes a few seconds. Then set `par.forcing = 8` and `par.forcing_amplitude_icb = 0.0` (the inner core turns with
the mantle) and run the same four steps. Both runs give the same viscous dissipation and the same torques on the walls,
since adding a rigid rotation does not change the shear. The kinetic energies differ: in the mean frame most of the
fluid stays at rest, while in the frame of the mantle it moves back and forth past the walls. The flow is
proportional to $A$, and every power to $A^2$.

## Reading the output

The flow is in `flow.npy`, relative to the frame of the run: the mean frame for forcing 7, the mantle for forcing 8.
For `m = 0` the angular velocity of the fluid relative to that frame is

$$
\omega = \frac{u_\phi}{r\sin\theta} = \frac{2}{r}\thinspace\mathrm{Re}\sum_l T_l(r)\thinspace P_l'(\cos\theta)\thinspace e^{iwt} ,
$$

with the toroidal coefficients $T_l(r)$, as on [Thermal forcing](thermal-forcing.md#reading-the-output).

`spin_doctor` checks the energy and angular momentum budgets. The input power P_in comes from the viscous stress of the
walls (forcing 7, `flow.dat` columns 24 and 25) or from the Poincaré force (forcing 8, column 42). Columns 31 and 33 of
`flow.dat` are the viscous torques on the mantle and on the inner core, and 35 and 37 the magnetic ones
([Output files](output.md#flowdat)).

## Parameters

| parameter | meaning |
| --- | --- |
| `forcing` | 7 (the walls librate) or 8 (the frame of the mantle) |
| `forcing_frequency` | $w$; not 0 |
| `forcing_amplitude_cmb` | $A$, the libration angle of the mantle, in radians |
| `forcing_amplitude_icb` | $A_{IC}$, that of the inner core: in the mean frame for forcing 7, relative to the mantle for forcing 8 (0 in a full sphere) |
| `m`, `symm` | 0, 1 (forcing 7 also 2, 1) |
| `bci`, `bco` | forcing 7: `bco = 1`, `bci` 0 or 1 (with `m = 2`: `bci = 1`); forcing 8: 0 or 1 |

[Combining features](compatibility.md#forced-runs) lists what else these forcings combine with.
