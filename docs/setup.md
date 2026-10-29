# Setting up a problem

## How parameters are set

All parameters are attributes of one object, `par`. Their defaults, with comments listing the available options, are in `bin/params_default.py`. `bin/parameters.py` is where a problem is set up:

```python
from params_default import default_params, pmak, ssak
import numpy as np

par = default_params()      # every parameter at its default value

par.Ek       = 1e-6         # change only what differs from the defaults
par.magnetic = 1
...

par.set_scales()            # computes the derived parameters; call it last
```

`par.set_scales()` fills in the parameters that depend on others, unless they were set explicitly: the magnetic numbers `Em`, `Le` and `Le2`, the diffusion numbers `Etherm` and `Ecomp`, the buoyancy factors `BV2` and `BV2_comp`, the time scale factor `OmgTau`, the resolution `N` and `lmax`, and the solver target `tau`. Set every parameter before this call: a change made after it does not reach the derived parameters.

All the scripts in `bin/` read the parameters with `from parameters import par`, so the same `parameters.py` must be in place for every step of a run. Parameter studies set one parameter per run folder (see [Many runs](output.md#many-runs)); the spare parameters `aux1` and `aux2` are recorded in `params.dat` and can be used, for example, to define a ramp.

The sections below follow the order of `params_default.py`. The magnetic boundary conditions, the background field, the rotational dynamics and the solver options have pages of their own.


## Physics switches

| Parameter | Default | Meaning |
|---|---|---|
| `hydro` | 1 | Include the momentum (Navier-Stokes) equation for the flow $\mathbf{u}$. |
| `magnetic` | 0 | Include the induction equation for the magnetic field perturbation $\mathbf{b}$ and the Lorentz force. Needs a background field, see [Background magnetic field](background-field.md). |
| `thermal` | 0 | Include the heat equation and thermal buoyancy (Boussinesq). |
| `compositional` | 0 | Include the composition equation and compositional buoyancy (Boussinesq). |
| `rotdyn` | 0 | Let the mantle and the inner core rotate in response to axial torques, see [Rotational dynamics](rotdyn.md). |
| `rotation` | 1 | Rotating frame. With 0 the Coriolis force is left out; then use a time scale other than the rotation one (e.g. `timescale = 'viscous'`) and set `N` explicitly. |


## Geometry and symmetry

| Parameter | Default | Meaning |
|---|---|---|
| `ricb` | 0.35 | Inner core radius. The CMB radius is 1 and is the unit of length. `ricb = 0` is a full sphere. |
| `m` | 0 | Azimuthal wave number ($m \geq 0$). All fields vary as $e^{im\phi}$. |
| `symm` | 1 | Equatorial symmetry of the flow: 1 symmetric, -1 antisymmetric. The magnetic field perturbation has symmetry `symm` times that of the background field, and temperature and composition follow the flow. |


## Flow boundary conditions and Ekman number

| Parameter | Default | Meaning |
|---|---|---|
| `bci` | 1 | Flow condition at the inner core boundary (ICB): 0 stress-free, 1 no-slip (or boundary forcing). Ignored if `ricb = 0`. |
| `bco` | 1 | Flow condition at the core-mantle boundary (CMB): 0 stress-free, 1 no-slip (or boundary forcing). |
| `Ek` | $10^{-6}$ | Ekman number $E = \nu/\Omega L^2$. |

`Ek = 0` gives an inviscid problem, with only the no-penetration condition $u_r = 0$ at the boundary. It works only for a full sphere (`ricb = 0`), and `N` must then be set by hand. A conducting inner core, a thin conducting wall with `c1` $\neq 0$, and `rotdyn` all need a viscous fluid and no-slip boundaries.


## Eigenvalue or forced problem

With `forcing = 0` (the default) Kore solves the eigenvalue problem $A\,x = \lambda B\,x$ for the free modes. Any other value selects a forced problem, solved as one linear system at the frequency `forcing_frequency`:

| `forcing` | Forcing | Requirements |
|---|---|---|
| 1 | Tidal body force of Lin & Ogilvie (2018) | $m = 2$, symmetric |
| 2 | Boundary flow forcing | `bci = 1`, `bco = 1` |
| 3 | Tidal body force of Rovira-Navarro et al. (2018) | $m$ = 0 and 2 symmetric, $m = 1$ antisymmetric. Leaks power. |
| 4 | Test case: Lin body force with $X(r) = (A r^2 + B/r^3)\,C$ | $m = 2$, symmetric |
| 5 | Test case: Lin body force with $X(r) = 1/r$ | $m = 0$, symmetric |
| 6 | Radial velocity forcing at the ICB (Buffett 2010) | $m = 1$, antisymmetric |
| 7 | Longitudinal libration as boundary flow forcing | $m$ = 0 or 2, symmetric, no-slip |
| 8 | Longitudinal libration as a Poincaré body force, in the mantle frame | $m = 0$, symmetric, no-slip |
| 9 | Radial boundary flow forcing | $m = 2$, symmetric |
| 12 | Steady differential rotation of the inner core and/or the mantle: rigid rotation at the rates `forcing_amplitude_icb`, `forcing_amplitude_cmb` relative to the frame (units set by `timescale`) | $m = 0$, symmetric, no-slip, `ricb > 0`; `forcing_frequency = 0` for the steady flow. Tested without magnetic field. |

| Parameter | Default | Meaning |
|---|---|---|
| `forcing_frequency` | 0 | Frequency of the forcing, in the units set by `timescale`. Negative is prograde. |
| `forcing_amplitude_cmb` | 0 | Forcing amplitude at the CMB. Body forces use this value. |
| `forcing_amplitude_icb` | 0 | Forcing amplitude at the ICB. |
| `projection` | 1 | Recorded in `params.dat`; not used by the current solver. |

Forced problems use a direct solver and keep the amplitude set by the forcing (see [Solver settings](solver.md#forced-problems)).


## Time scale

The equations can be written with different units of time (see [Equations and scaling](equations.md)). The choice enters through the dimensionless rotation rate $\Omega\tau$, `OmgTau`, which `set_scales()` sets from `timescale` unless `OmgTau` is given directly:

| `timescale` | Unit of time $\tau$ | `OmgTau` |
|---|---|---|
| `'rotation'` (default) | $1/\Omega$ | 1 |
| `'viscous'` | $L^2/\nu$ | $1/E$ |
| `'Alfven'` | $L\sqrt{\mu_0\rho}/B_0$ | $1/Le$ |
| `'magnetic'` | $L^2/\eta$ | $1/E_\eta$ |

Eigenvalues, frequencies and forcing frequencies are all in these units.


## Magnetic field strength and diffusivity

The field strength and the magnetic diffusivity can be given in either of two ways:

- the Elsasser number `Lambda` ($\Lambda$, default 0.1) and the magnetic Prandtl number `Pm` (default 0.01). Then `set_scales()` computes the magnetic Ekman number $E_\eta$ = `Em` = `Ek/Pm`, `Le2` $= \Lambda E_\eta$ and the Lehnert number `Le` $= \sqrt{\Lambda E_\eta}$;
- or the Lehnert number `Le` and the magnetic Ekman number `Em` directly, in which case `Lambda` and `Pm` are not used.

The Lundquist number is `Le/Em`. The helper functions `pmak(Pm0, E0, Ek)` and `ssak(Lambda0, E0, Ek)` in `params_default.py` return `Pm` and `Lambda` as power laws of `Ek` through the point (`E0`, `Pm0`) or (`E0`, `Lambda0`), useful to keep a path towards Earth-like values when lowering `Ek`.

The background field itself is chosen with `B0`, `cnorm` and related parameters ([Background magnetic field](background-field.md)), and the magnetic boundary conditions with `innercore` and `mantle` ([Inner core and mantle](boundaries.md)).


## Thermal parameters

| Parameter | Default | Meaning |
|---|---|---|
| `Prandtl` | 1 | Prandtl number $Pr = \nu/\kappa$. |
| `Etherm` | `Ek/Prandtl` | Thermal Ekman number. |
| `heating` | `'internal'` | Background temperature gradient, see below. |
| `Ra` | $2.5168 \times 10^8$ | Rayleigh number $Ra = \alpha g_0 L^3 \Delta T/\nu\kappa$, with $L$ the CMB radius and $\Delta T$ the temperature scale below. |
| `BV2` | `-Ra*Ek**2/Prandtl` | Squared ratio of a reference Brunt-Väisälä frequency to the rotation rate, $N_0^2/\Omega^2$. Set it directly for problems without diffusion, where `Ra` is undefined. Negative values are unstably stratified. |
| `rc`, `h`, `rsy` | 0.7, 0.1, -1 | Transition radius, width and radial parity for `'two zone'` and `'user defined'`. |
| `bci_thermal`, `bco_thermal` | 1, 1 | At the ICB and the CMB: 0 isothermal ($\theta = 0$), 1 fixed heat flux ($\partial_r\theta = 0$). |

| `heating` | Background gradient $dT/dr$ | Temperature scale |
|---|---|---|
| `'internal'` | $-\beta r$ (uniform internal heating) | $\beta r_o^2$ |
| `'differential'` | $-\beta r^{-2}$, with $\beta = (T_i - T_o)\, r_i r_o / (r_o - r_i)$ | $T_i - T_o$ |
| `'two zone'` | $K\,$`ut.twozone(r)`: a neutral inner zone and a stratified outer zone joined by a tanh step at `rc` (Vidal et al. 2015) | $-r_o K$ |
| `'user defined'` | $K\,$`ut.BVprof(r)`, a function you edit in `bin/utils.py` (by default $\propto r$) | $-r_o K$ |

The profiles are expanded in Chebyshev polynomials to the tolerance `tol_tc` (default $10^{-6}$).


## Compositional parameters

These mirror the thermal ones: `Schmidt` (Schmidt number $Sc = \nu/D$, default 1), `Ecomp` (default `Ek/Schmidt`), `comp_background` (`'internal'` or `'differential'`), `Ra_comp` (compositional Rayleigh number, default 0), `BV2_comp` (default `-Ra_comp*Ek**2/Schmidt`), `rcc`, `hc`, `rsyc`, and `bci_compositional`, `bco_compositional` (0 fixed composition, 1 fixed flux).


## Resolution

| Parameter | Default | Meaning |
|---|---|---|
| `ncpus` | 4 | Number of MPI processes for `submatrices.py` and `assemble.py`. |
| `N` | `Ncheb(Ek)` | Number of Chebyshev polynomials in radius, in the fluid. Must be even for a full sphere. |
| `g` | 1 | Approximate ratio $l_{max}/N$. |
| `lmax` | from `N`, `g`, `ncpus`, `m` | Largest spherical-harmonic degree, $2\,n_{cpu} \lfloor gN/2n_{cpu} \rfloor + m - 1$. Set by hand, it must be even if $m$ is odd, and $l_{max} - m + 1$ must be divisible by $2\,n_{cpu}$. |
| `N_cic` | 64 | Chebyshev polynomials for the field inside a conducting inner core. |
| `inviscid_basis` | `'low'` | Basis of the flow equations in an inviscid full sphere with a magnetic field: `'low'` (more accurate at a given `N`, denser matrix) or `'high'` (banded, converges more slowly in `N`). See [Solver settings](solver.md#inviscid-runs-with-a-magnetic-field). |

How to choose `N` and `lmax`, and how to check that they are large enough, is explained in [Solver settings](solver.md#resolution) and [Output and postprocessing](output.md#checking-the-results).


## Target and eigensolver

| Parameter | Default | Meaning |
|---|---|---|
| `rtau`, `itau` | 0, 1 | Real (damping) and imaginary (frequency) parts of the target $\tau$ = `rtau + 1j*itau`. The eigensolver returns the modes closest to it. A positive frequency is retrograde. |
| `which_eigenpairs` | `'TM'` | Which eigenvalues SLEPc returns. First letter: `L` largest, `S` smallest, `T` closest to the target. Second letter: `M` magnitude, `R` real part, `I` imaginary part. Use `'TM'` with shift-and-invert. |
| `nev` | 3 | Number of eigenvalues to compute. |
| `maxit` | 50 | Maximum number of restarts of the eigensolver (iterations for forced problems). |
| `tol` | $10^{-16}$ | Convergence tolerance of the solver. |
| `track_target` | 0 | With 1, `set_scales()` reads `rtau` and `itau` from the first two numbers in a file `track_target`, e.g. to follow a mode along a parameter path. |

The PETSc, SLEPc and MUMPS options (`petsc_opts`) and the pre-scaling switch (`prescale`) are described in [Solver settings](solver.md).
