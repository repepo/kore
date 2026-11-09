# Tidal forcing

Forcing 14 moves the walls radially in the pattern of a degree-2 tide, as the tidal bulge raised by a companion moves
the surface of a star or a planet. kore returns the flow this drives in the fluid: the tidally forced inertial and
gravito-inertial waves, with their dissipation and the power that the tide puts in. A scan of `forcing_frequency` gives
the frequency spectrum of the response.

Two more tides are at the end of this page. Forcing 1 splits the tide of an incompressible fluid into the equilibrium
tide, the flow that follows the bulge, and the wave-like tide that the rotation excites, and solves for the wave-like
part ([The wave-like tide](#the-wave-like-tide-forcing-1)). Forcing 2 moves the walls of the ocean of Enceladus with its
eccentricity tide ([The eccentricity tide of Enceladus](#the-eccentricity-tide-of-enceladus-forcing-2)).

## What it imposes

At each wall $r_w$ the radial velocity is

$$
u_r(r_w, \theta, \phi, t) = A_w\thinspace Y_2^m(\theta)\thinspace e^{i(m\phi + wt)} + \text{c.c.},
$$

with $A_w$ = `forcing_amplitude_cmb` at the outer wall and `forcing_amplitude_icb` at the inner wall, $w$ =
`forcing_frequency`, and $Y_2^m$ kore's Schmidt harmonics. The physical radial velocity at the wall is therefore

| tide | `m`, `symm` | $u_r$ at the wall |
| --- | --- | --- |
| eccentricity, $(l, m) = (2, 0)$ | 0, 1 | $2A\thinspace P_2(\cos\theta)\cos wt$, with $P_2(\cos\theta) = \tfrac12(3\cos^2\theta - 1)$ |
| obliquity, $(2, 1)$ | 1, −1 | $-\sqrt6\thinspace A\sin\theta\cos\theta\cos(\phi + wt)$ |
| eccentricity and asynchronism, $(2, 2)$ | 2, 1 | $\tfrac{\sqrt6}{2}\thinspace A\sin^2\theta\cos(2\phi + wt)$ |

The tangential velocity at the walls is zero: forcing 14 keeps the no-slip conditions and replaces only $u_r = 0$ by
the imposed value. The pattern turns at the angular velocity $-w/m$ in the frame, so a negative $w$ is prograde, as
everywhere in kore. For a body spinning at the frame rate $\Omega$ and a companion on a circular orbit of angular velocity
$n$, the $(2, 2)$ tide has $w = 2(\Omega - n)\tau$ in kore's units ([Anelastic equations](anelastic.md#the-time-scale)).
If the wall follows a tidal displacement $\xi\thinspace Y_2^m e^{i(m\phi + wt)}$ + c.c., its velocity is $iw\xi$: set the
amplitude to $w\xi$ (the factor $i$ shifts the phase and changes no power). The problem is linear: the flow is
proportional to the amplitudes, and every power to their square.

Requirements: `bco = 1` and `bci = 1` (also in a full sphere, where `bci` is otherwise ignored), and one of the three
pairs (`m`, `symm`) above. In a full sphere (`ricb = 0`) only the outer wall moves. `assemble.py` stops with a message
when a requirement is not met. The forcing works with any `structure`, with `thermal = 1`, with a differential
rotation and with a field ([Magnetic field](magnetic.md#forced-runs)).

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

It takes a few seconds. `spin_doctor` prints residσ $1\times10^{-14}$, the power the walls put in (P_in) and the
viscous dissipation (Dν), and its `flow.dat` holds the whole energy budget ([Output files](output.md)):

```python
import numpy as np
f = np.loadtxt('flow.dat', ndmin=2)[-1]
P_in  = f[20] + f[21] + f[23] + f[24]   # work of the pressure and of the viscous stress at both walls
D_nu  = f[25]                           # viscous dissipation
print(P_in, D_nu)                       # 0.1416912 0.1416912
```

The outer wall puts in $W_p = 0.1618$ through the pressure and $W_\nu^{\rm wall} = -0.0201$ through the viscous stress,
and the viscosity dissipates the sum, $D_\nu = 0.1417$: with a uniform rotation and no buoyancy nothing else can take it.
The viscous wall work does not depend on the flow inside. On a no-slip wall the tangential velocity is zero, so
$\nabla\cdot(\rho_0\mathbf u) = 0$ fixes $\partial_r u_r$ there, and $W_\nu^{\rm wall} = -\tfrac{32\pi}{5}\thinspace\mathrm{ViscosD}\thinspace\lvert A\rvert^2$
at the outer wall of an incompressible fluid. All the dynamics of the response is in $W_p$.

A spectrum is a loop over `forcing_frequency`: `submatrices.py` does not depend on it, but `assemble.py` and `solve.py`
must be run again for each frequency.

## The energy of the response

`spin_doctor` evaluates the kinetic energy budget of [Energy balance](anelastic.md#energy-balance). For a forced,
periodic response ($\sigma = 0$) it reads

$$
P_{\rm in} = W_p + W_\nu^{\rm wall} = D_\nu - W_b - W_\Omega ,
$$

with, in `flow.dat` and `thermal.dat`:

| quantity | columns | meaning |
| --- | --- | --- |
| $P_{\rm in}$ | `flow.dat` 21 + 22 + 24 + 25 | the power the moving walls put into the fluid: the tidal power |
| $W_p$ | `flow.dat` 21 (inner wall), 22 (outer wall) | its part done by the pressure |
| $W_\nu^{\rm wall}$ | `flow.dat` 24, 25 | its part done by the viscous stress |
| $D_\nu$ | `flow.dat` 26 | the viscous dissipation, the heat released |
| $-W_b$ | `thermal.dat` 5 (with the opposite sign) | the power handed to the entropy perturbation, released by thermal diffusion |
| $W_\Omega$ | `flow.dat` 18 | the power taken from the differential rotation |

Which one to report depends on the question:

- **Orbital and spin evolution** depend on the energy the tide gives to the fluid, $P_{\rm in}$. With a uniform rotation
  and no buoyancy it equals $D_\nu$, which is why the viscous dissipation is often called the tidal dissipation. It is
  not in general.
- **Tidal heating** is $D_\nu$, plus the thermal diffusion of the entropy perturbation with `thermal = 1`.
- **The exchange with the differential rotation**, $W_\Omega$, has either sign. Where the shear feeds the waves
  ($W_\Omega > 0$) the fluid dissipates more than the tide puts in, and $P_{\rm in}$ can even be negative. Where the waves
  give energy and angular momentum to the mean flow, mostly at critical layers, it dissipates less. kore keeps the
  background fixed: whatever maintains it (convection, a body force) is assumed to absorb or supply $W_\Omega$. Report
  $W_\Omega$ with the others.

The amplitudes are those of the complex coefficients: every power above is the mean, over a period, of the power of the
physical flow.

## Interpreting the response

- **Unstable modes.** The forced solve only needs $A - iwB$ to be invertible, so it gives a response even when the
  background has growing modes elsewhere. The result is the part of the flow at the forcing frequency, in phase with
  the tide; in a linear problem the growing modes add to it at their own frequencies. It is meaningful when the
  instabilities are saturated, for example by the convection that the background represents, and their effect on the
  tide is in the viscosity. It is not meaningful near a growing mode: check with an eigenvalue solve
  (`forcing = 0`, `itau` = `forcing_frequency`) that no eigenvalue with $\mathrm{Re}\thinspace\lambda \geq 0$ is close to
  $i w$. A spectrum computed this way is not causal: do not transform it back into a response in time.
- **Critical layers.** With a differential rotation, where $w + m\thinspace\mathrm{Gaspard}\thinspace\Delta\Omega(r,\theta)$
  vanishes the waves meet a critical layer, which only the viscosity resolves. Check that the response converges with N
  and lmax there.
- **No-slip walls.** Forcing 14 imposes zero tangential velocity on the moving walls. Each wall then has an Ekman layer,
  whose dissipation scales as $\mathrm{Ek}^{1/2}$, more than a smooth flow inside ($\propto\mathrm{Ek}$). On the solid
  surface of a core this is physical. At the free surface of a star or a giant planet, or at the boundary of a
  convective envelope over a radiative zone, it is not: there the tangential flow of the tide does not vanish. Compare
  runs at several Ekman numbers to tell the wall layers from the waves inside.
- **Resolution of the wall pressure.** $W_p$ needs the pressure at the walls, which converges with N more slowly than
  the volume terms when the density varies ([Numerical method](numerical-method.md#checks)). For `'Wu 2005'`
  (`r_out` 0.99) at Ek $10^{-3}$, N 96 gives a 5 % error on $W_p$ in a shell, and N 192 in a full sphere a 40 % error,
  while the kinetic energy has converged. residσ shows it: accept a response when it is below $10^{-5}$.

## Parameters

| parameter | meaning |
| --- | --- |
| `forcing` | 14 |
| `forcing_frequency` | $w$, in units of $1/\tau$ in the rotating frame; negative is prograde |
| `forcing_amplitude_cmb` | $A$ at the outer wall: the amplitude of the radial velocity |
| `forcing_amplitude_icb` | $A$ at the inner wall (not used in a full sphere) |
| `m`, `symm` | (0, 1), (1, −1) or (2, 1) |
| `bci`, `bco` | 1 |

The fields are written as for any forced run ([Output files](output.md)), and `tests/tidal` checks the forcing against an
exact solution of the forced flow.

## The wave-like tide (forcing 1)

In a fluid body the tide has two parts. The equilibrium tide is the large-scale flow that follows the tidal bulge as it
moves, a potential flow in an incompressible fluid. The wave-like tide is the response that the Coriolis force on that
flow excites: inertial waves, whose dissipation depends strongly on the frequency. Forcing 1 solves for the wave-like
part alone, as Lin & Ogilvie (2018) do, so that its response and its dissipation are seen on their own.

The equilibrium tide of degree 2 in a shell is

$$
\mathbf u_e = \nabla\big[X(r)\thinspace Y_2^m(\theta)\thinspace e^{i(m\phi + wt)}\big] + \text{c.c.}, \qquad
X(r) = -\frac{3a}{4}\thinspace\frac{r^2 + \tfrac23 r_i^5 r^{-3}}{1 - r_i^5} ,
$$

with $a$ = `forcing_amplitude_cmb` and $w$ = `forcing_frequency`. It moves the outer wall radially,
$u_r = -\tfrac32 a\thinspace Y_2^m e^{i(m\phi + wt)}$ + c.c. there, and leaves the inner wall in place. In a fluid of
uniform density it is solenoidal and irrotational, so its inertia and its viscous force are gradients, which the
pressure takes up. What remains is the Coriolis force on it, $-2\thinspace\mathrm{Gaspard}\thinspace\hat{\mathbf z}\times\mathbf u_e$,
a body force on the wave-like tide $\mathbf u'$. kore solves for $\mathbf u'$, whose walls are at rest: impenetrable, and
no-slip or stress-free as `bci` and `bco` say. The whole tide is $\mathbf u_e + \mathbf u'$. The conditions at the walls
hold for $\mathbf u'$ alone, so at a no-slip wall the whole tide slips with the tangential velocity of $\mathbf u_e$.

Requirements: `structure = 'Boussinesq'`, `thermal = 0` and `magnetic = 0`, since the equilibrium tide above is that of
a uniform fluid, with no stratification to displace and no field to induce; (`m`, `symm`) = (0, 1), (1, −1) or (2, 1);
`bci` and `bco` 0 or 1, and no drag at the outer wall (`kappa_cmb` 0, `topo_drag` `None`), which would act on the slip
of the whole tide. With `m = 2` the run needs an inner core: in a full sphere the Coriolis force on the equilibrium tide
is a gradient and drives nothing. With `m` = 0 or 1 a full sphere works, and there alone the fluid may be inviscid
(`ViscosD = 0`).

**Example.** In the example of forcing 14 above, set

```python
par.forcing = 1
par.forcing_frequency = 1.0
par.forcing_amplitude_cmb = 1.0   # a
```

and run the same four steps. `spin_doctor` prints residσ $8\times10^{-15}$, and P_in = Dν = 0.019314: the power of the
body force on the wave-like tide, `flow.dat` column 42, which the viscosity dissipates. The walls do no work, since
$\mathbf u'$ is at rest there, and $D_\nu$ is the dissipation of $\mathbf u'$ alone. The flow in `flow.npy` is
$\mathbf u'$; add $\mathbf u_e$ for the whole tide.

## The eccentricity tide of Enceladus (forcing 2)

Enceladus keeps its ocean between an icy shell and a rocky core. On its eccentric orbit the tide stretches and squeezes
both, so the walls of the ocean move, radially and along themselves. Rekier et al. (2019, appendix A.1) give these
motions as fits in the ratio of the ocean's radii, `ricb` (`utils.eccen_tide`). Forcing 2 moves the walls with those
fits, and kore returns the flow in the ocean, with its inertial waves and their dissipation.

Each wall moves with the velocity $iw\boldsymbol\xi$ + c.c., with $\boldsymbol\xi$ the displacement of the tide at that wall,
radial and tangential, of degree 2, and $w$ = `forcing_frequency`. One amplitude, `forcing_amplitude_cmb`, scales the
fits at both walls (1: the fits as they are). The tide of a moon that turns once per orbit has three parts, each a run:
`m = 0`, and `m = 2` with $w$ = −`Gaspard` (prograde) and $w$ = `Gaspard` (retrograde); their powers add. The fits are
the tide at the orbital frequency, $|w|$ = `Gaspard`; at another frequency `assemble.py` prints a note and keeps the
same displacement.

Requirements: `ricb` from 16/21 to 1, where the fits hold (Enceladus has about 0.88); (`m`, `symm`) = (0, 1) or (2, 1);
no-slip walls (`bci = bco = 1`, `ViscosD > 0`), which the tide moves along themselves. Any `structure`, buoyancy and a
differential rotation work, and a field with insulating walls (`'insulator'`, or `'TWA'` with `c1` 0): a conducting
solid behind a wall would deform along it, which is not covered ([Magnetic field](magnetic.md#a-boundary-that-moves)).
The power is read as for forcing 14 ([The energy of the response](#the-energy-of-the-response)): P_in is the work of the
moving walls, `flow.dat` columns 21, 22, 24 and 25. For example, the prograde part with `ricb = 37/42`, Ek $10^{-3}$, N
48 and `lmax` 61 gives P_in = Dν = $2.17\times10^{-11}$ at `forcing_amplitude_cmb = 1`.

| parameter | forcing 1 | forcing 2 |
| --- | --- | --- |
| `forcing_frequency` | $w$, not 0 | $w$, ± `Gaspard` |
| `forcing_amplitude_cmb` | $a$: $u_r = -\tfrac32 a\thinspace Y_2^m$ at the outer wall | the scale of the fits, at both walls |
| `forcing_amplitude_icb` | not used | not used |
| `m`, `symm` | (0, 1), (1, −1), (2, 1) | (0, 1), (2, 1) |
| `bci`, `bco` | 0 or 1 | 1 |

`tests/cic_forcings` checks both against the `conductive_IC` branch, the wave-like tide against rigid rotations that
solve it exactly in a full sphere, and their budgets and stops; `tests/forcing` checks the wall values of forcing 2.
