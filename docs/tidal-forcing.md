# Tidal forcing

Forcing 14 moves the walls radially in the pattern of a degree-2 tide, the way the tidal bulge raised by a companion
moves the surface of a star or a planet. kore returns the flow this drives inside: tidally forced inertial and
gravito-inertial waves, their dissipation, and the power that the tide puts in. Repeating the run over a range of
`forcing_frequency` gives the frequency spectrum of the response.

Two related tides are described at the end of this page:

- Forcing 1 solves for the *wave-like tide* of an incompressible fluid, the part of the tide excited by rotation
  ([The wave-like tide](#the-wave-like-tide-forcing-1)).
- Forcing 2 moves the walls of the ocean of Enceladus with its eccentricity tide
  ([The eccentricity tide of Enceladus](#the-eccentricity-tide-of-enceladus-forcing-2)).

## What it imposes

At each wall the radial velocity is

$$
u_r = A\thinspace Y_2^m(\theta)\thinspace e^{i(m\phi + wt)} + \text{c.c.},
$$

with amplitude $A$ = `forcing_amplitude_cmb` at the outer wall and `forcing_amplitude_icb` at the inner wall, and
frequency $w$ = `forcing_frequency`. Three tides are possible:

| tide | `m` | `symm` |
| --- | --- | --- |
| eccentricity tide, $(l, m) = (2, 0)$ | 0 | 1 |
| obliquity tide, $(2, 1)$ | 1 | −1 |
| eccentricity and asynchronous tide, $(2, 2)$ | 2 | 1 |

The pattern rotates at the angular velocity $-w/m$ in the rotating frame, so a negative $w$ is prograde, as everywhere
in kore. For a body spinning at the frame rate $\Omega$ with a companion on a circular orbit of angular velocity $n$,
the (2, 2) tide has $w = 2(\Omega - n)/\Omega$ with `timescale = 'rotation'`. If you know the tidal displacement of the wall rather
than its velocity, set the amplitude to $w$ times the displacement. The flow is proportional to the amplitude, and every
power to its square.

Requirements:

- no-slip walls (`bco = 1`, and `bci = 1` with an inner core), or an inviscid fluid in a full sphere;
- one of the three (`m`, `symm`) pairs above.

In a full sphere (`ricb = 0`) only the outer wall moves. The forcing works with any `structure`, with buoyancy,
differential rotation, a background flow and a magnetic field ([Magnetic field](magnetic.md#forced-runs)).
[Combining features](compatibility.md#forced-runs) lists what else it combines with.

## Example: the (2, 2) tide in a rotating shell

Write in `bin/parameters.py`

```python
from params_default import default_params

par = default_params()

par.timescale = 'rotation'
par.Ek = 1e-3
par.set_scales()                  # Gaspard = 1, ViscosD = Ek

# incompressible fluid in a rotating shell, no-slip walls
par.structure = 'Boussinesq'
par.thermal = 0
par.ricb = 0.35
par.bci = 1
par.bco = 1

# the (2, 2) tide: m = 2, equatorially symmetric
par.m = 2
par.symm = 1

par.N = 96
par.lmax = 49
par.ncpus = 2

# radial velocity Y_2^2 at the outer wall, at the frequency w = 1 (retrograde in the frame)
par.forcing = 14
par.forcing_frequency = 1.0
par.forcing_amplitude_cmb = 1.0
par.forcing_amplitude_icb = 0.0
```

and run

```shell
./bin/submatrices.py 2
mpiexec -n 2 ./bin/assemble.py
mpiexec -n 2 ./bin/solve.py
./bin/spin_doctor.py 2
```

It takes a few seconds. `spin_doctor` prints the power the walls put in (P_in) and the viscous dissipation (Dν), and
writes the whole energy budget to `flow.dat` ([Output files](output.md)):

```python
import numpy as np
f = np.loadtxt('flow.dat', ndmin=2)[-1]
P_in  = f[20] + f[21] + f[23] + f[24]   # work of the pressure and of the viscous stress at both walls
D_nu  = f[25]                           # viscous dissipation
print(P_in, D_nu)                       # 0.1416912 0.1416912
```

With uniform rotation and no buoyancy, all the power that the tide puts in is dissipated by viscosity, so the two
numbers agree.

To compute a spectrum, loop over `forcing_frequency`. `submatrices.py` does not depend on it, but `assemble.py` and
`solve.py` must be run again for each frequency.

## The energy of the response

For a forced, periodic response, `spin_doctor` evaluates the energy budget of
[Energy balance](anelastic.md#energy-balance):

$$
P_{\rm in} = W_p + W_\nu^{\rm wall} = D_\nu - W_b - W_\Omega ,
$$

| quantity | columns | meaning |
| --- | --- | --- |
| $P_{\rm in}$ | `flow.dat` 21 + 22 + 24 + 25 | the power the moving walls put into the fluid: the tidal power |
| $W_p$ | `flow.dat` 21 (inner wall), 22 (outer wall) | the part done by pressure |
| $W_\nu^{\rm wall}$ | `flow.dat` 24, 25 | the part done by viscous stress |
| $D_\nu$ | `flow.dat` 26 | the viscous dissipation, the heat released |
| $-W_b$ | `thermal.dat` 5 (with the opposite sign) | the power given to the entropy perturbation, released by thermal diffusion |
| $W_\Omega$ | `flow.dat` 18 | the power taken from the differential rotation |

Which one to report depends on the question:

- **Orbital and spin evolution** depend on the tidal power $P_{\rm in}$. It equals the viscous dissipation $D_\nu$
  only with uniform rotation and no buoyancy.
- **Tidal heating** is $D_\nu$, plus the thermal diffusion of the entropy perturbation with `thermal = 1`.
- **Exchange with the differential rotation**, $W_\Omega$, can have either sign. The waves can draw energy from the
  shear, so that the fluid dissipates more than the tide puts in, or give energy to it. kore keeps the background
  rotation fixed, so report $W_\Omega$ together with the others.

All powers are averages over one period of the oscillation.

## Interpreting the response

- **Unstable modes.** The forced solve gives an answer even when the background has growing modes. The answer is the
  part of the flow at the forcing frequency, which is meaningful only if those instabilities are saturated by something
  else. It is not meaningful near a growing mode: run an eigenvalue solve (`forcing = 0`, `itau` = `forcing_frequency`)
  and check that no growing mode has a frequency close to the forcing.
- **Critical layers.** With differential rotation, waves can meet critical layers, where the forcing frequency matches
  the local rotation. Only viscosity resolves them; check that the response converges with `N` and `lmax`.
- **No-slip walls.** Forcing 14 imposes zero tangential velocity at the moving walls, which creates viscous boundary
  layers there. On the solid surface of a core this is physical. At the free surface of a star or a giant planet it is
  not. Compare runs at several Ekman numbers to tell the dissipation in the wall layers from that of the waves inside.
- **Resolution.** The pressure work $W_p$ converges more slowly with `N` than the kinetic energy, especially when the
  density varies strongly. Accept a response only when `spin_doctor`'s residσ is below $10^{-5}$.

## Parameters

| parameter | meaning |
| --- | --- |
| `forcing` | 14 |
| `forcing_frequency` | $w$, in the rotating frame; negative is prograde |
| `forcing_amplitude_cmb` | amplitude of the radial velocity at the outer wall |
| `forcing_amplitude_icb` | amplitude at the inner wall (not used in a full sphere) |
| `m`, `symm` | (0, 1), (1, −1) or (2, 1) |
| `bci`, `bco` | `bco = 1`, and `bci = 1` with an inner core |

## The wave-like tide (forcing 1)

In a fluid body the tide has two parts. The *equilibrium tide* is the large-scale flow that follows the tidal bulge as
it moves. The *wave-like tide* is the response that the Coriolis force on that flow excites: inertial waves, whose
dissipation depends strongly on frequency. Forcing 1 solves for the wave-like part alone, as Lin & Ogilvie (2018) do.

The equilibrium tide of degree 2 in a shell is

$$
\mathbf u_e = \nabla\big[X(r)\thinspace Y_2^m(\theta)\thinspace e^{i(m\phi + wt)}\big] + \text{c.c.}, \qquad
X(r) = -\frac{3a}{4}\thinspace\frac{r^2 + \tfrac23 r_i^5 r^{-3}}{1 - r_i^5} ,
$$

with $a$ = `forcing_amplitude_cmb`, $w$ = `forcing_frequency` and $r_i$ = `ricb`. The Coriolis force on this flow acts
as a body force on the wave-like tide, which kore solves for with walls at rest. The solution in `flow.npy` is the
wave-like tide; add $\mathbf u_e$ to get the whole tide.

Requirements:

- a uniform fluid without stratification or field: `structure = 'Boussinesq'`, `thermal = 0`, `compositional = 0`,
  `magnetic = 0`;
- (`m`, `symm`) = (0, 1), (1, −1) or (2, 1);
- `bco` 0 or 1, and `bci` 0 or 1 with an inner core;
- no drag at the outer wall;
- with `m = 2`, an inner core (in a full sphere this tide drives nothing). With `m` = 0 or 1 a full sphere works, and
  the fluid may then be inviscid.

Differential rotation and a background flow work.

**Example.** In the example of forcing 14 above, set

```python
par.forcing = 1
par.forcing_frequency = 1.0
par.forcing_amplitude_cmb = 1.0
```

and run the same four steps. The walls now do no work: the power comes from the body force (`flow.dat` column 42),
and the viscosity dissipates it.

## The eccentricity tide of Enceladus (forcing 2)

Enceladus has an ocean between an icy shell and a rocky core. On its eccentric orbit the tide stretches and squeezes
both, so the walls of the ocean move, radially and along themselves. Forcing 2 moves the walls with the fits of
Rekier et al. (2019, appendix A.1), which depend on the radius ratio `ricb`, and kore returns the flow in the ocean,
with its inertial waves and their dissipation.

`forcing_amplitude_cmb` scales the motion at both walls (1: the fits as they are). The tide of a moon in synchronous
rotation has three parts, each a separate run whose powers add: `m = 0`, and `m = 2` with `forcing_frequency` equal to
−`Gaspard` (prograde) and to `Gaspard` (retrograde).

Requirements:

- `ricb` between 16/21 and 1, where the fits hold (Enceladus has about 0.88);
- (`m`, `symm`) = (0, 1) or (2, 1);
- no-slip walls (`bci = bco = 1`) and a viscous fluid.

Any structure, buoyancy, differential rotation and a background flow work, as does a magnetic field with insulating
walls. kore stops with a conducting solid behind a wall ([Magnetic field](magnetic.md#a-boundary-that-moves)). The
power is read as for forcing 14 ([The energy of the response](#the-energy-of-the-response)).
