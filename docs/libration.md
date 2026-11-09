# Libration

A moon on an eccentric orbit, or a body whose shape is not symmetric about its axis, does not spin at a steady rate: its
mantle turns back and forth about the rotation axis around the mean rotation, at the frequency of the orbit. This is the
libration in longitude. The fluid inside does not follow the mantle at once: the walls drag it through the viscosity
and the field, and a mantle whose equator is not round pushes it as it turns. kore returns the flow that this
oscillation drives, with its dissipation. Forcing 7 describes the libration in the frame that turns at the mean rate,
where the walls move. Forcing 8 describes it in the frame of the mantle, where the walls are at rest and the fluid feels
the acceleration of the frame instead.

## Forcing 7: the walls librate

The frame turns at the mean rate, and the walls turn back and forth in it. With `m = 0` the mantle turns by the angle
$A\cos wt$ about the $z$ axis and the inner core by $A_{IC}\cos wt$, with $A$ = `forcing_amplitude_cmb`, $A_{IC}$ =
`forcing_amplitude_icb` (in radians) and $w$ = `forcing_frequency`. A no-slip wall carries the fluid with it,
$u_\phi = -wA\thinspace r\sin\theta\sin wt$ at the outer wall. The outer wall must be no-slip (`bco = 1`), and the
fluid viscous: walls that slide along themselves do not move an inviscid fluid, and kore stops. The inner wall may be
no-slip or stress-free (`bci` = 1 or 0); a stress-free inner wall does not pass the libration of the inner core to the
fluid, so $A_{IC}$ then has no effect. With a field it would pass it through the field where the inner core, or a thin
layer on it, conducts: kore stops on that case, $A_{IC}$ not 0 behind a stress-free inner wall under a poloidal B0. In a
full sphere (`ricb = 0`) only the mantle librates.

With `m = 2` forcing 7 stands for a mantle whose equator is elongated, which moves its shape past the sphere as it
librates. The walls stay spheres, and the forcing imposes the radial velocity of that motion at each wall,
$u_r = -wA\thinspace r_w\thinspace Y_2^2(\theta)\thinspace e^{i(2\phi + wt)}$ + c.c. at the wall of radius $r_w$, with
$A$ at the outer wall and $A_{IC}$ at the inner one, and no-slip walls (`bco = 1`, and `bci = 1` with an inner core),
or an inviscid fluid in a full sphere, whose one wall row keeps the fluid from crossing the wall whatever `bco`. With a field, how a
conducting solid behind such a wall moves is set by `forced_walls` ([Magnetic field](magnetic.md#a-boundary-that-moves)).

## Forcing 8: the frame of the mantle

The frame turns with the mantle, so the outer wall is at rest. The inner core turns with the mantle, or librates
relative to it by the angle $A_{IC}\cos wt$ ($A_{IC}$ = `forcing_amplitude_icb`). The frame's own oscillation puts a body
force in the fluid, the Poincaré force of its angular acceleration,

$$
\mathbf f = w^2 A\cos wt\thinspace r\sin\theta\thinspace\hat{\boldsymbol\phi} ,
$$

and, with a differential rotation, the Coriolis force of the libration on the background flow. The flow is that of
forcing 7 seen from the mantle: forcing 7's flow, with the inner core's angle $A + A_{IC}$, minus the rigid libration
of the mantle, $u_\phi = -wA\thinspace r\sin\theta\sin wt$. The field is the same in both frames, since a rotation
about the axis leaves the axisymmetric B0 as it is.

The walls are at rest in this frame, so forcing 8 takes the walls that forcing 7 cannot move: a stress-free outer wall,
along which the fluid slips relative to the mantle and a drag acts on that slip ([Drag at the outer wall](drag.md)),
with or without a conducting mantle, and an inviscid fluid in a full sphere. It needs `m = 0`, `symm = 1`, a frequency
other than 0, and walls that stay in place (`bco` and `bci` 0 or 1; an inviscid fluid takes any `bco`). In a full
sphere there is no inner core to librate: $A_{IC}$ must be 0. Any `structure`, buoyancy, a differential rotation and a
field work as in the other runs ([Magnetic field](magnetic.md#forced-runs)), but for the case that stops forcing 7 too:
an inner core that librates relative to the mantle ($A_{IC}$ not 0) behind a stress-free inner wall, where the core or
a thin layer on it conducts, under a poloidal B0. Use forcing 8 for what an observer on the mantle sees, or where
forcing 7 does not take the walls.

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
the mantle) and run the same four steps. Both runs give the same viscous dissipation, Dν = 0.02848, and the same
torques on the walls, since a rigid rotation is not sheared. The kinetic energy differs: 0.0283 in the mean frame,
where most of the fluid stays at rest and only the layers at the walls and the inertial waves they launch move, and
0.420 in the frame of the mantle, which turns back and forth past that fluid. The problem is linear: the flow is
proportional to $A$, and every power to $A^2$.

## Reading the output

The flow is in `flow.npy`, relative to the frame of the run: the mean frame for forcing 7,
the mantle for forcing 8. For `m = 0` the angular velocity of the fluid relative to that frame is

$$
\omega = \frac{u_\phi}{r\sin\theta} = \frac{2}{r}\thinspace\mathrm{Re}\sum_l T_l(r)\thinspace P_l'(\cos\theta)\thinspace e^{iwt} ,
$$

with the toroidal scalars $T_l(r)$ of odd $l$, as on [Thermal forcing](thermal-forcing.md#reading-the-output); the
rigid libration of the mantle is $T_1 = iwA\thinspace r/2$.

`spin_doctor` checks the budgets. With forcing 7 the walls put in P_in, the power of their viscous stress on the fluid
(`flow.dat` columns 24 and 25). With forcing 8 it is the power of the Poincaré force (`flow.dat` column 42), plus that
of the inner wall when the inner core librates relative to the mantle. Either way the viscous dissipation Dν uses it up,
with the Ohmic losses when there is a field and the work against the buoyancy of a stratified fluid (residσ, and
resid𝐄 with a field). In `flow.dat`, columns 31 and 33 are the viscous torques on the mantle and on the inner core, and
35 and 37 the magnetic ones ([Output files](output.md#flowdat)); resid𝐋 checks that they and, for forcing 8, the torque
of the Poincaré force change the angular momentum of the fluid as they should. Where nothing couples the fluid to the
walls (stress-free walls without a drag or a field, an inviscid fluid), the fluid of forcing 8 stays at rest in the mean
frame, and its flow is minus the libration of the mantle.

## Parameters

| parameter | meaning |
| --- | --- |
| `forcing` | 7 (the walls librate) or 8 (the frame of the mantle) |
| `forcing_frequency` | $w$, in units of $1/\tau$; not 0 |
| `forcing_amplitude_cmb` | $A$, the angle of the mantle's libration (forcing 7 with `m` 2: the amplitude of the radial velocity at the outer wall over $w$) |
| `forcing_amplitude_icb` | $A_{IC}$, that of the inner core: in the mean frame for forcing 7, relative to the mantle for forcing 8 (0 in a full sphere) |
| `m`, `symm` | 0, 1 (forcing 7 also 2, 1) |
| `bci`, `bco` | forcing 7: `bco = 1`, `bci` 0 or 1 (`m` 2: `bci = 1`); forcing 8: `bco` 0 or 1, `bci` 0 or 1. `bci` only with an inner core; an inviscid fluid (forcing 7 with `m` 2, forcing 8) takes any `bco` |

Like every forced run, forcings 7 and 8 need a spherical body (`centrifugal = 0`); [Combining features](compatibility.md#forced-runs)
shows what else they combine with. The solution is written as for any forced run ([Output files](output.md)).
`tests/forcing` checks the wall values of
forcing 7, `tests/couette` its budgets, and `tests/cic_forcings` forcing 8 against forcing 7 and against the
`conductive_IC` branch, its budgets where forcing 7 cannot follow, and its stops.
