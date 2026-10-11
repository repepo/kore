# Inner core and mantle

The magnetic boundary conditions describe what lies on the other side of the fluid's boundaries: an electrical insulator, a conductor, or a thin conducting layer. They are set with `innercore` at the inner core boundary (ICB) and `mantle` at the core-mantle boundary (CMB). They only matter with `magnetic = 1`, and `innercore` only with an inner core (`ricb > 0`).

On this page the magnetic field perturbation is written, for each spherical harmonic $Y_l^m$, as

$$
\mathbf{b} = \nabla\times\nabla\times\left( f(r)\, Y_l^m\, \mathbf{r} \right) + \nabla\times\left( g(r)\, Y_l^m\, \mathbf{r} \right),
$$

with poloidal scalar $f$ and toroidal scalar $g$, and $F = r f$, $G = r g$. Primes are radial derivatives, $r_i$ is the ICB radius `ricb`, and the CMB radius is 1.


## The options

| Boundary | Setting | What it models | Parameters |
|---|---|---|---|
| ICB | `innercore = 'insulator'` | Electrically insulating inner core (default). | none |
| ICB | `innercore = 'conducting, Chebys'` | Solid inner core with its own conductivity and permeability; the induction equation is solved inside it. For eigenvalue problems. | `sigma_i2o`, `mu_i2o`, `N_cic`, `lmax_cic` |
| ICB | `innercore = 'conducting, Bessel'` | The same, with an analytical solution inside the inner core. For forced problems only. | `sigma_i2o`, `mu_i2o` |
| ICB | `innercore = 'perfect conductor, material'` | Infinitely conducting inner core; tangential electric field continuous in the frame of the inner core. Not usable at present, see below. | none |
| ICB | `innercore = 'perfect conductor, spatial'` | The same, with the tangential electric field continuous in the frame of the fluid. Not usable at present, see below. | none |
| ICB | `innercore = 'TWA'` | Thin conducting layer on top of an insulating inner core. | `c_icb`, `c1_icb` |
| CMB | `mantle = 'insulator'` | Electrically insulating mantle (default). | none |
| CMB | `mantle = 'TWA'` | Thin conducting layer at the base of an insulating mantle. | `c_cmb`, `c1_cmb`, `B0_cmb` |

Material properties are given relative to the fluid:

| Parameter | Default | Meaning |
|---|---|---|
| `sigma_i2o` | 1 | Electrical conductivity of the inner core / fluid. |
| `mu_i2o` | 1 | Magnetic permeability of the inner core / fluid. |
| `mu` | 1 | Magnetic permeability of the fluid / vacuum. It enters the thin-wall conditions. |

The background field has to be compatible with the boundaries as well. That is handled by `B0_ic` and `B0_cmb`, see [Background magnetic field](background-field.md#the-field-inside-the-inner-core-b0_ic).


## Insulating boundaries

Outside an insulator the field is a potential field, which decays away from the boundary. Matching $\mathbf{b}$ to it gives

$$
\text{CMB:}\quad (l+1)\,f + r f' = 0, \quad g = 0; \qquad\qquad \text{ICB:}\quad l\,f - r f' = 0, \quad g = 0 .
$$

No electric current crosses an insulating boundary, so there is no electromagnetic torque on it.


## Conducting inner core

With `innercore = 'conducting, Chebys'` Kore adds the magnetic field inside the inner core, $\mathbf{b}_i$, to the unknowns, expanded in `N_cic` Chebyshev polynomials and spherical harmonics up to `lmax_cic`. The inner core is solid and at rest in the rotating frame, so its field only diffuses:

$$
\partial_t \mathbf{b}_i = (\Omega\tau)\, E_\eta\, \eta_{i}\, \nabla^2 \mathbf{b}_i, \qquad \eta_i = \frac{1}{\mu_{i2o}\,\sigma_{i2o}},
$$

where $\eta_i$ is the inner core's magnetic diffusivity relative to the fluid's. At the ICB, the normal field, the tangential $\mathbf{H} = \mathbf{b}/\mu$ and the tangential electric field are continuous. With subscripts $o$ (fluid) and $i$ (inner core), and the permeabilities relative to the fluid's:

$$
f_o = f_i, \qquad (r f_o)' = \frac{(r f_i)'}{\mu_{i2o}}, \qquad g_o = \frac{g_i}{\mu_{i2o}},
$$

$$
(r g_o)' = \eta_i\,(r g_i)', \qquad r^2 \nabla_l^2 f_o = \eta_i\, r^2 \nabla_l^2 f_i, \qquad \nabla_l^2 = \frac{d^2}{dr^2} + \frac{2}{r}\frac{d}{dr} - \frac{l(l+1)}{r^2} .
$$

The electric field condition has no $\mathbf{u}\times\mathbf{B}_0$ term, so it assumes the fluid at the ICB moves with the inner core: use a no-slip ICB (`bci = 1`) and `Ek > 0`.

**Resolution.** `N_cic = 64` (the default) is safe for conductivity ratios from $10^{-4}$ to 1. In the tests behind this default, 16 polynomials put the eigenvalue off by $1.6\times10^{-3}$ at `sigma_i2o = 1`, and 48 gave broken eigenvectors for the modes away from the target at `sigma_i2o` = $10^{-2}$. The inner core adds $2\,n_{ic}$ unknowns, which is cheap: at $N = 232$ a torsional-mode run took 40 s and 7.8 GB with a conducting inner core, against 22 s and 7.3 GB with a thin wall.

**Limits.** As `sigma_i2o` goes to zero the solution converges to the insulating one (linearly in `sigma_i2o`).

`'conducting, Bessel'` is the forced-problem version: the inner-core field is taken as $j_l(k r)$, with $k = (1-i)\sqrt{\omega/2E_\eta}$ at the forcing frequency $\omega$, and only the matching conditions at the ICB are added to the matrix. Its ICB condition is flagged in the CHANGELOG as still to be checked.


## Perfectly conducting inner core

Inside a perfect conductor the field perturbation vanishes. The conditions are $f = 0$ and a zero tangential electric field at the ICB. With `'perfect conductor, material'` the electric field is taken in the frame of the inner core, which gives $\nabla_l^2 f = 0$ and $(r g)' = 0$. With `'perfect conductor, spatial'` it is taken in the rotating frame, which adds the $\mathbf{u}\times\mathbf{B}_0$ terms. The two are identical with a no-slip ICB.

These two options are not usable at present: they are missing from the list of inner-core conditions in `bin/utils.py` (`ic_bc_list`), so every script stops with `ValueError: 'perfect conductor, material' is not in list`. The boundary conditions themselves are still in `bin/assemble.py`, with comments saying they need fixing.


## Thin conducting wall

`'TWA'` (thin-wall approximation) represents a thin layer of thickness $h$, conductivity $\sigma_w$ and permeability $\mu_w$ between the fluid and an insulator (Roberts, Glatzmaier & Clune 2010). The layer is described by two numbers:

$$
c = \frac{h\,\mu_w}{r_j\,\mu_f}, \qquad c_1 = \frac{h\,\sigma_w}{r_j\,\sigma_f},
$$

where $r_j$ is the boundary radius and $\mu_f$, $\sigma_f$ are the fluid's values. $c_1$ is the wall-to-fluid conductance ratio and sets the eddy currents in the wall; $c$ is a magnetic-permeability term. Both are free parameters (`c_cmb`, `c1_cmb` at the CMB and `c_icb`, `c1_icb` at the ICB). The conditions, which follow Guervilly, Wood & Brummell (2013, appendix A), are

$$
\text{CMB:}\quad \left(\frac{1}{\mu} + l\,c\right) F' + \frac{l}{r}\, F + \frac{c_1}{\mu}\, r\, \mathcal{D}_l F = 0, \qquad G + c_1\, r\, G' = 0,
$$

$$
\text{ICB:}\quad \left(\frac{1}{\mu} + (l+1)\,c\right) F' - \frac{l+1}{r}\, F - \frac{c_1}{\mu}\, r\, \mathcal{D}_l F = 0, \qquad G - c_1\, r\, G' = 0,
$$

with $\mathcal{D}_l F = F'' - l(l+1)F/r^2 = r\,\nabla_l^2 f$. With $c = c_1 = 0$ and $\mu = 1$ they reduce to the insulating conditions. They were checked against the exact free-decay rates of a sphere and a shell bounded by thin walls, to $10^{-14}$.

The term $\mathcal{D}_l F$ stands for the wall's $\partial_t F/\eta$, taken from the fluid next to it. That substitution needs the wall to move with the fluid, so a wall with $c_1 \neq 0$ needs a no-slip boundary and `Ek > 0`; `assemble.py` prints a warning otherwise. With a wall at the CMB, keep `B0_cmb = 'wall'` (the default) so that the background field satisfies the wall condition too (see [Background magnetic field](background-field.md#a-thin-conducting-wall-at-the-cmb-b0_cmb)).


## Electromagnetic torques

A conducting layer or inner core lets currents close outside the fluid, and the Lorentz force then exerts an axial torque on the boundary. `spin_doctor.py` reports it for $m = 0$, symmetric flows: $|\Gamma|_{mag}$ for the mantle (with `mantle = 'TWA'`) and $|\Gamma|_{magIC}$ for the inner core (with `'conducting, Chebys'` or `'TWA'`). The complex torques are written to `magnetic.dat` (see [Output and postprocessing](output.md)). With `rotdyn = 1` the same torques make the mantle and inner core rotate (see [Rotational dynamics](rotdyn.md)).


## Thin wall or conducting inner core?

The thin wall is cheaper to set up, but it is not a substitute for a conducting inner core. When the inner core's field is a skin layer of depth $\delta = \sqrt{2\eta_i/\omega}$, the conducting inner core acts on the fluid exactly like a thin wall, but with a complex, frequency-dependent conductance, $c_1 \approx (1-i)\,\delta/2r_i$ for $\delta \ll r_i$. Kore's thin wall takes a real $c_1$. In tests with the torsional modes at $E = 10^{-7}$ and $10^{-8}$, the best real $c_1$ still left the eigenvalues 1 to 2% off, with errors of 5 to 15 degrees in the phase of the inner-core torque. The mode in which the inner core and the fluid oscillate together was off by about 20%. Use a conducting inner core when the inner core's electromagnetic coupling matters, and the thin wall for a thin conducting layer such as the base of the mantle.
