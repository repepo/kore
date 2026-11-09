# Spherical Couette flow

Forcing 12 turns the walls steadily about the rotation axis, each at its own rate, and kore returns the steady flow
they drive through the viscosity of the fluid: the spherical Couette flow. An inner core that turns slightly faster or
slower than the mantle drives such a flow, as do the spheres of a laboratory experiment.

## What it imposes

The outer wall turns at the rate `forcing_amplitude_cmb` and the inner wall at `forcing_amplitude_icb`, relative to the
rotating frame. A wall that turns must be no-slip (`bco` or `bci` = 1), so that the fluid next to it turns with it. A
wall at rest may also be stress-free (`bco` or `bci` = 0); the fluid then slips along it, held back only by a drag if
there is one ([Drag at the outer wall](drag.md)).

The equations are linear, so the flow is proportional to the rates. This is the limit of small differences in rotation.

Requirements:

- `forcing_frequency = 0`, `m = 0` and `symm = 1` (for walls that turn back and forth, see [Libration](libration.md));
- a viscous fluid (`ViscosD > 0`);
- with buoyancy, its diffusion (`ThermaD > 0` with `thermal = 1`, `CompoD > 0` with `compositional = 1`).

In a full sphere (`ricb = 0`) only the outer wall can turn, and the fluid simply turns with it. Any structure,
differential rotation, a background flow and a magnetic field work; with a field, a conducting inner core or mantle
turns with its wall ([Magnetic field](magnetic.md#forced-runs)).

## Example: an inner core that turns

Write in `bin/parameters.py`

```python
from params_default import default_params

par = default_params()

par.timescale = 'rotation'
par.Ek = 1e-4
par.set_scales()                  # Gaspard = 1, ViscosD = Ek

# incompressible fluid between an inner core and the mantle, no-slip walls
par.structure = 'Boussinesq'
par.ricb = 0.35
par.bci = 1
par.bco = 1

# axisymmetric and equatorially symmetric
par.m = 0
par.symm = 1

par.N = 96
par.lmax = 191
par.ncpus = 2

# the inner core turns steadily, the mantle with the frame
par.forcing = 12
par.forcing_frequency = 0.0
par.forcing_amplitude_icb = 1.0
par.forcing_amplitude_cmb = 0.0
```

and run

```shell
./bin/submatrices.py 2
mpiexec -n 2 ./bin/assemble.py
mpiexec -n 2 ./bin/solve.py
./bin/spin_doctor.py 2
```

It takes a few seconds.

![Angular velocity of the example](assets/couette-flow.png)

The figure shows the angular velocity of the fluid relative to that of the inner core, in the northern hemisphere. Fast
rotation makes the flow nearly uniform along the rotation axis, so away from the walls the fluid turns in coaxial
cylinders. Outside the *tangent cylinder* (the cylinder that touches the equator of the inner core) each cylinder of
fluid meets only the mantle, and turns with it: here, not at all. Inside, each cylinder is held between the thin
boundary layers on the inner core and on the mantle, and turns at a rate in between: about half the rate of the inner
core near the axis, faster towards the tangent cylinder. At very small Ekman numbers this approaches the solution of
Proudman (1956),

$$
\frac{\omega}{\Omega_{IC}} = \frac{(1 - s^2)^{1/4}}{(1 - s^2)^{1/4} + (1 - s^2/r_i^2)^{1/4}}, \qquad s < r_i ,
$$

with $s$ the distance from the axis and $r_i$ the inner radius, and zero outside the tangent cylinder. A thin shear
layer on the tangent cylinder, the Stewartson layer, smooths the jump between the two regions.

## Reading the output

The flow is in `flow.npy` ([Output files](output.md#solution-files)); its imaginary part is zero, since the flow is
steady. For `m = 0` the angular velocity is

$$
\omega = \frac{u_\phi}{r\sin\theta} = \frac{2}{r}\sum_l T_l(r)\thinspace P_l'(\cos\theta) ,
$$

with the toroidal coefficients $T_l(r)$, as on [Thermal forcing](thermal-forcing.md#reading-the-output).

`spin_doctor` checks the energy and angular momentum budgets. The power that the turning walls put in (P_in) is
dissipated by viscosity, and by Ohmic losses with a field. Columns 31 and 33 of `flow.dat` are the viscous torques on
the mantle and on the inner core, and 35 and 37 the magnetic ones ([Output files](output.md#flowdat)). In the steady
state the torques on the two solids cancel. When the fluid turns rigidly (both walls at the same rate, or a full
sphere), nothing is sheared and the budget residuals compare round-off errors; they can then be close to 1 and should
be ignored.

## Parameters

| parameter | meaning |
| --- | --- |
| `forcing` | 12 |
| `forcing_frequency` | 0 |
| `forcing_amplitude_cmb` | the rotation rate of the outer wall, relative to the frame |
| `forcing_amplitude_icb` | the rotation rate of the inner wall (0 in a full sphere) |
| `m`, `symm` | 0, 1 |
| `bci`, `bco` | 1 for a wall that turns; 0 or 1 for a wall at rest |

The flow can also be used as the background of other runs, to study its waves and instabilities
([Background flow](bgflow.md#a-background-from-turning-walls)).
