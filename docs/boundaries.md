# Inner core and mantle

The magnetic boundary conditions describe what lies on the other side of the fluid's boundaries: an electrical insulator, a conductor, or a conducting layer, thin or resolved. They are set with `innercore` at the inner core boundary (ICB) and `mantle` at the core-mantle boundary (CMB). They only matter with `magnetic = 1`, and `innercore` only with an inner core (`ricb > 0`).

On this page the magnetic field perturbation is written, for each spherical harmonic $Y_l^m$, as

$$
\mathbf{b} = \nabla\times\nabla\times\left( f(r)\, Y_l^m\, \mathbf{r} \right) + \nabla\times\left( g(r)\, Y_l^m\, \mathbf{r} \right),
$$

with poloidal scalar $f$ and toroidal scalar $g$, and $F = r f$, $G = r g$. Primes are radial derivatives, $r_i$ is the ICB radius `ricb`, and the CMB radius is 1.


## The options

| Boundary | Setting | What it models | Parameters |
|---|---|---|---|
| ICB | `innercore = 'insulator'` | Electrically insulating inner core (default). | none |
| ICB | `innercore = 'conducting, Chebys'` | Solid inner core with its own conductivity and permeability; the induction equation is solved inside it. For eigenvalue and forced problems. | `sigma_i2o`, `mu_i2o`, `N_cic` |
| ICB | `innercore = 'conducting, Bessel'` | The same, with an analytical solution inside the inner core. For forced problems only (`forcing` $\neq$ 0). | `sigma_i2o`, `mu_i2o` |
| ICB | `innercore = 'TWA'` | Thin conducting layer on top of an insulating inner core. | `c_icb`, `c1_icb`, `muw_icb` |
| CMB | `mantle = 'insulator'` | Electrically insulating mantle (default). | none |
| CMB | `mantle = 'TWA'` | Thin conducting layer at the base of an insulating mantle. | `c_cmb`, `c1_cmb`, `muw_cmb`, `B0_cmb` |
| CMB | `mantle = 'resolved'` | Conducting layer of any thickness and smooth conductivity profile at the base of an insulating mantle; the induction equation is solved inside it. | `r_surface`, `N_mantle`, `sigma_mantle`, `sigma_mantle_param` |

Material properties are given relative to the fluid:

| Parameter | Default | Meaning |
|---|---|---|
| `sigma_i2o` | 1 | Electrical conductivity of the inner core / fluid. |
| `mu_i2o` | 1 | Magnetic permeability of the inner core / fluid. |
| `mu` | 1 | Magnetic permeability of the fluid / vacuum. It enters the thin-wall conditions. |
| `muw_cmb`, `muw_icb` | 1 | Magnetic permeability of a thin wall / vacuum (see [Thin conducting wall](#thin-conducting-wall)). |

The background field has to be compatible with the boundaries as well. That is handled by `B0_ic` and `B0_cmb`, see [Background magnetic field](background-field.md#the-field-inside-the-inner-core-b0_ic).


## Insulating boundaries

Outside an insulator the field is a potential field, which decays away from the boundary. Matching $\mathbf{b}$ to it gives

$$
\text{CMB:}\quad (l+1)\,f + r f' = 0, \quad g = 0; \qquad\qquad \text{ICB:}\quad l\,f - r f' = 0, \quad g = 0 .
$$

No electric current crosses an insulating boundary, so there is no electromagnetic torque on it.


## Conducting inner core

With `innercore = 'conducting, Chebys'` Kore adds the magnetic field inside the inner core, $\mathbf{b}_i$, to the unknowns, expanded in `N_cic` Chebyshev polynomials and spherical harmonics up to `lmax`, as in the fluid. The inner core is solid and at rest in the rotating frame, so its field only diffuses:

$$
\partial_t \mathbf{b}_i = (\Omega\tau)\, E_\eta\, \eta_{i}\, \nabla^2 \mathbf{b}_i, \qquad \eta_i = \frac{1}{\mu_{i2o}\,\sigma_{i2o}},
$$

where $\eta_i$ is the inner core's magnetic diffusivity relative to the fluid's. In a forced problem the field oscillates at the forcing frequency $\omega$, $\partial_t = i\omega$ (before 2026-11-02 a forced run with `'conducting, Chebys'` stopped, or read the inner-core matrices of an earlier run, and had a steady field in the inner core). At the ICB, the normal field, the tangential $\mathbf{H} = \mathbf{b}/\mu$ and the tangential electric field are continuous. With subscripts $o$ (fluid) and $i$ (inner core), and the permeabilities relative to the fluid's:

$$
f_o = f_i, \qquad (r f_o)' = \frac{(r f_i)'}{\mu_{i2o}}, \qquad g_o = \frac{g_i}{\mu_{i2o}},
$$

$$
(r g_o)' = \eta_i\,(r g_i)', \qquad r^2 \nabla_l^2 f_o = \eta_i\, r^2 \nabla_l^2 f_i, \qquad \nabla_l^2 = \frac{d^2}{dr^2} + \frac{2}{r}\frac{d}{dr} - \frac{l(l+1)}{r^2} .
$$

In the fluid the electric field is $\mathbf{E} = (\Omega\tau)E_\eta\nabla\times\mathbf{b} - \mathbf{u}\times\mathbf{B}_0$. At a no-slip ICB the fluid moves with the inner core, $\mathbf{u} = 0$, and the condition on $g$ above holds as it stands. At a stress-free ICB (`bci = 0`) the fluid slides over the inner core, and the tangential part of $\mathbf{u}\times\mathbf{B}_0$, which is $B_{0r}\,(\mathbf{u}\times\hat{\mathbf{r}})$ since $u_r = 0$, enters the condition on $g$:

$$
(r g_o)' - \eta_i\,(r g_i)' = \frac{r\,S_l}{(\Omega\tau)E_\eta}, \qquad
S_l = \frac{\beta}{l(l+1)} \sum_{l'} \left[ i m\, D_{ll'}\, \frac{(r P_{l'})'}{r} + \frac{l_0(l_0+1) - l(l+1) - l'(l'+1)}{2}\, G_{ll'}\, T_{l'} \right],
$$

where $S_l$ is the $\nabla Y_l$ part of $B_{0r}\,(\mathbf{u}\times\hat{\mathbf{r}})$, $B_{0r} = \beta\, Y_{l_0}^0$ is the radial background field at the ICB (only a poloidal $\mathbf{B}_0$ has one), $P$ and $T$ are the poloidal and toroidal scalars of the flow, and $G_{ll'}$ and $D_{ll'}$ are the $Y_l$ components of $Y_{l_0}^0 Y_{l'}$ and of $(\mathrm{d}Y_{l_0}^0/\mathrm{d}\cos\theta)\, Y_{l'}$. A degree-1 field couples $g$ of degree $l$ to $P_l$ and $T_{l\pm1}$, a degree-2 field to $P_{l\pm1}$ and $T_l$, $T_{l\pm2}$. Before 2026-10-30 this term was missing, so a stress-free ICB with a conducting inner core gave wrong modes (eigenvalues off by up to 2% at `Ek` = `Em` = $10^{-3}$ in the tests); `spin_doctor`'s `resid𝐄` catches it.

The poloidal field is solved with the radial component of the induction equation, as with an insulating inner core, and only the first two conditions are imposed on $f$. With the radial induction equation satisfied on both sides of the ICB, continuity of $f$ makes the radial component of $\nabla\times\mathbf{E}$ continuous, and that is the condition on $\nabla_l^2 f$. Before October 2026 Kore used the consoidal component instead, which is third order in $f$, and imposed the $\nabla_l^2 f$ condition as an extra row. Its tau error was integrated across the shell into a spurious term $c/r$ of the radial equation. For a torsional mode at Ek = $10^{-9}$ and `sigma_i2o` = $10^{-6}$ with $N = 688$, this gave resid b = $10^{-3}$ and a low-$l$ poloidal field wrong by factors of up to 30. The radial component gives $10^{-5}$ and the insulating-core field to four digits at the same $N$.

`spin_doctor.py`'s residuals cover the fluid volume only. They do not test the inner-core field or the ICB conditions, so check `N_cic` by the change of the eigenvalue.

**Resolution.** `N_cic = 64` (the default) is safe for conductivity ratios from $10^{-4}$ to 1. In the tests behind this default, 16 polynomials put the eigenvalue off by $1.6\times10^{-3}$ at `sigma_i2o = 1`, and 48 gave broken eigenvectors for the modes away from the target at `sigma_i2o` = $10^{-2}$. The inner core adds $2\,n_{ic}$ unknowns, which is cheap: at $N = 232$ a torsional-mode run took 40 s and 7.8 GB with a conducting inner core, against 22 s and 7.3 GB with a thin wall.

**Limits.** As `sigma_i2o` goes to zero the solution converges to the insulating one (linearly in `sigma_i2o`).

In a forced problem the inner-core field is confined to a skin layer of thickness $\delta = \sqrt{2(\Omega\tau)E_\eta\eta_i/|\omega|}$ below the ICB, which `N_cic` must resolve. In tests ($\omega = 0.8$, $r_i = 0.35$) `N_cic = 64` gave the field to $3\times10^{-7}$ at $\delta = 0.014\,r_i$ ($E_\eta = 10^{-5}$), but only to $3\times10^{-3}$ at $\delta = 0.0045\,r_i$ ($E_\eta = 10^{-6}$), where it took 128 ($4\times10^{-9}$).

`'conducting, Bessel'` is the forced-problem version, without `N_cic`. At the forcing frequency the equation above, $i\omega\,\mathbf{b}_i = (\Omega\tau)E_\eta\eta_i\nabla^2\mathbf{b}_i$, is solved by $j_l(k r)$ in each degree, with $k^2 = -i\omega/((\Omega\tau)E_\eta\eta_i)$. With $x = k r_i$ and $x_d = x\, j_l'(x)/j_l(x)$, the conditions above become

$$
(r f_o)' = \frac{1 + x_d}{\mu_{i2o}}\, f_o, \qquad (r g_o)' - \frac{1 + x_d}{\sigma_{i2o}}\, g_o = \frac{r\,S_l}{(\Omega\tau)E_\eta} \qquad \text{at } r = r_i,
$$

with $S_l = 0$ at a no-slip ICB, and $x_d = l$ at $\omega = 0$ (`forcing = 12`). Without a forcing frequency there is no $k$, so a run with `forcing = 0` stops. In tests (forcings 1, 7, 9 and 12; `B0` = `'axial'`, `'dipole'` and `'Luo_S2'`; no-slip and stress-free ICB; `mu_i2o`, `sigma_i2o` $\neq 1$; the Alfvén time scale) the two inner cores give the same solution to $10^{-10}$. Before 2026-11-02 the two conditions had $f_o$ and $f_o'$ swapped, $k$ lacked $\Omega\tau$ and $\eta_i$, and the flow entered the condition on $g$ through terms valid for `B0 = 'axial'` only, so the solutions were wrong (the field by 9% to 600% in those tests), and `forcing = 12` divided by zero.


## Perfectly conducting inner core

Not available at present. The former options `'perfect conductor, material'` and `'perfect conductor, spatial'` were unreachable (missing from `ic_bc_list` in `bin/utils.py`) and were removed in October 2026, together with the consoidal projection they relied on. A rewrite is pending. With the radial projection, $f = 0$ at the ICB is enough for the poloidal field: with $\mathbf{u} = 0$ there, the radial induction equation then gives $\nabla_l^2 f = 0$, the zero tangential electric field.


## Thin conducting wall

`'TWA'` (thin-wall approximation) represents a thin layer of thickness $h$, conductivity $\sigma_w$ and permeability $\mu_w$ between the fluid and an insulator (Roberts, Glatzmaier & Clune 2010). The layer is described by two numbers:

$$
c = \frac{h\,\mu_w}{r_j\,\mu_f}, \qquad c_1 = \frac{h\,\sigma_w}{r_j\,\sigma_f},
$$

where $r_j$ is the boundary radius and $\mu_f$, $\sigma_f$ are the fluid's values. $c_1$ is the wall-to-fluid conductance ratio and sets the eddy currents in the wall; $c$ is a magnetic-permeability term. Both are free parameters (`c_cmb`, `c1_cmb` at the CMB and `c_icb`, `c1_icb` at the ICB). A third parameter, the wall's permeability relative to vacuum, $p = \mu_w/\mu_0$ (`muw_cmb`, `muw_icb`, default 1), fixes the wall's thickness, $h/r_j = c\,\mu/p$. The conditions are

$$
\text{CMB:}\quad \left(\frac{1}{\mu} + l\,c\right) F' + \frac{l}{r}\, F + \frac{c_1}{\mu}\, r\, \mathcal{D}_l F + \frac{c\,\mu\,l}{p\,r}\left(\frac{l+1}{p} - 1\right) F = 0, \qquad G + c_1\, r\, G' = 0,
$$

$$
\text{ICB:}\quad \left(\frac{1}{\mu} + (l+1)\,c\right) F' - \frac{l+1}{r}\, F - \frac{c_1}{\mu}\, r\, \mathcal{D}_l F - \frac{c\,\mu\,(l+1)}{p\,r}\left(\frac{l}{p} + 1\right) F = 0, \qquad G - c_1\, r\, G' = 0,
$$

with $\mathcal{D}_l F = F'' - l(l+1)F/r^2 = r\,\nabla_l^2 f$. With $c = c_1 = 0$ and $\mu = 1$ they reduce to the insulating conditions. They were checked against the exact free-decay rates of a sphere and a shell bounded by thin walls, to $10^{-14}$.

The last term of each poloidal condition is first order in $h$. It comes from the curvature of the wall, from matching the potential field at $r_j \pm h$ instead of $r_j$, and from the $l(l+1)/r^2$ part of $\nabla_l^2$ inside the wall. Guervilly, Wood & Brummell (2013, appendix A) take the limit $h \to 0$ with $h\mu_w$ and $h\sigma_w$ fixed, which needs $\mu_w \gg \mu_0$ for $c$ to stay finite. Their conditions are the ones above without this term, recovered with `muw_cmb = np.inf` (or `muw_icb`); for $p \gg 1$ the term is a correction of relative size about $1/(p\,l)$ to the $c$ term. For a wall with the permeability of vacuum ($p = 1$, the default, as at the Earth's CMB) the term cancels the $c$ term at leading order: a static field crosses such a wall unchanged, and only its conductance $c_1$ acts. With $c_1 = 0$ and $p = \mu = 1$ the wall conditions are then exactly the insulating ones, which the former conditions missed by $O(l\,c)$. Against an exact conducting layer (the layer's induction equation integrated across the wall, CMB and ICB, $p$ = 1 to 100, steady and oscillating fields), the conditions with the term are off by $O(h)$ relative to the wall's own effect; without it the error is $O(1)$ for small $p$ (8% at $l = 5$ at the ICB for $p = 1$).

The thin-wall approximation itself needs the wall to be much thinner than its magnetic skin depth, $h \ll \delta_w = \sqrt{2\eta_w/\omega}$: the error on the wall's contribution to an eigenvalue grows as $(h/\delta_w)^2$.

The term $\mathcal{D}_l F$ stands for the wall's $\partial_t F/\eta$, taken from the fluid next to it. That substitution needs the wall to move with the fluid, so a wall with $c_1 \neq 0$ needs a no-slip boundary and `Ek > 0`; `assemble.py` prints a warning otherwise. With a wall at the CMB, keep `B0_cmb = 'wall'` (the default) so that the background field satisfies the wall condition too (see [Background magnetic field](background-field.md#a-thin-conducting-wall-at-the-cmb-b0_cmb)).


## Conducting mantle layer

With `mantle = 'resolved'` Kore adds the magnetic field in a conducting layer at the base of the mantle, $1 \le r \le r_s$ (`r_surface`), to the unknowns: `N_mantle` Chebyshev polynomials on $[1, r_s]$ and spherical harmonics up to `lmax`, as in the fluid. Beyond $r_s$ the mantle is insulating. The layer is at rest (or rotates rigidly about $z$ with `rotdyn = 1`), so its field only diffuses, with the conductivity $\sigma_m(r)$ relative to the fluid's. Divided by the layer's diffusivity $1/\sigma_m$, the equations for its scalars $f_m$, $g_m$ are

$$
\sigma_m\, \partial_t f_m = (\Omega\tau)\, E_\eta\, \nabla_l^2 f_m, \qquad \sigma_m\, \partial_t g_m = (\Omega\tau)\, E_\eta \left[ \nabla_l^2 g_m - (\ln\sigma_m)' \,\frac{(r g_m)'}{r} \right],
$$

so the profile only enters the time derivative and the last term, and the equations stay well scaled when $\sigma_m$ is small or varies by orders of magnitude. With the permeability of vacuum everywhere, $\mathbf{b}$ and the tangential electric field are continuous at the CMB, and the field matches a potential field at $r_s$:

$$
f = f_m, \qquad (r f)' = (r f_m)', \qquad g = g_m, \qquad \sigma_m(1)\, (r g)' = (r g_m)' \qquad (r = 1),
$$

$$
r f_m' + (l+1)\, f_m = 0, \qquad g_m = 0 \qquad (r = r_s).
$$

As at a conducting inner core, the continuity of $\nabla_l^2 f/\sigma$ follows from these. The electric field condition has no $\mathbf{u}\times\mathbf{B}_0$ term, so the fluid at the CMB must move with the layer: with flow (`hydro = 1`) the run stops unless the CMB is no-slip (`bco = 1`, `Ek > 0`), and in any case unless `mu = 1` and no forcing moves the CMB other than by a rotation about $z$ (`forcing = 2`, `9`, and `7` with $m \neq 0$ stop). The background field needs no change: it continues into the layer as a potential field.

| Parameter | Default | Meaning |
|---|---|---|
| `r_surface` | 1.83 | Outer radius of the layer (1.83: the whole mantle of the Earth). |
| `N_mantle` | 64 | Chebyshev polynomials in the layer. |
| `sigma_mantle` | 1 | Conductivity of the layer / fluid: a number (uniform layer), or a numpy expression of `r` and `p`. |
| `sigma_mantle_param` | 0 | The value of `p` in `sigma_mantle`. |

An expression may use the functions allowed in `B0_tor_profile` (`sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `tanh`, `sinh`, `cosh`, `arctan`, `where`, `pi`, `erf`). For example `'1e-2*exp(-p*(r - 1))'` is the exponential profile of PR #17 (A. Nuyt), and `'1e-2*(1 - tanh((r - 1.05)/0.02))/2 + 1e-8'` a layer about 0.05 thick with a smooth top (it needs `N_mantle = 128`). The profile must be smooth, positive on $[1, r_s]$ and resolved by `N_mantle` polynomials: B holds the `N_mantle`-term Chebyshev series of $r^2\sigma_m$, and the toroidal rows that of $r^2(\ln\sigma_m)'$, with $(\ln\sigma_m)'$ from an interpolant of $\ln\sigma_m$. Where they do not resolve the profile they oscillate: $\sigma_m$ in B can dip below 0, which gives spurious growing modes, and the toroidal modes come out wrong. The run therefore stops when the Chebyshev coefficients beyond `N_mantle` exceed $10^{-6}$ of the largest for $r^2\sigma_m$, or $10^{-6}$ for $\ln\sigma_m$ (the relative error of $\sigma_m$). A step (`where`) or a kink (`abs`) is never resolved: use a smooth transition, e.g. with `tanh` or `erf`. In free-decay tests the profiles that pass give decay rates within $10^{-6}$ of the exact ones. The solution in the layer is written to `real_magnetic_mantle.field` and `imag_magnetic_mantle.field`. `just_induction.py` (the induction equation alone) does not include the layer and stops.

Unlike the thin wall, the layer holds at any frequency, for any thickness and any smooth profile that `N_mantle` resolves. For a thin uniform layer, $r_s = 1 + h$ and `sigma_mantle` $= c_1/h$, it tends to the thin wall with `c1_cmb` $= c_1$ (and `c_cmb = 0`) as $h \to 0$, with a difference of order $h$; as `sigma_mantle` goes to 0 it tends to the insulating mantle. Free-decay rates of a conducting sphere or shell inside a uniform layer agree with the exact ones (spherical Bessel functions) to $10^{-11}$.


## Electromagnetic torques

A conducting layer or inner core lets currents close outside the fluid, and the Lorentz force then exerts a torque on the boundary. `spin_doctor.py` reports $|\Gamma|_{mag}$ for the mantle (with `mantle = 'TWA'` or `'resolved'`) for $m = 0$, `symm = 1` (axial) and $m = 1$, `symm = -1` (equatorial), and $|\Gamma|_{magIC}$ for the inner core (with `'conducting, Chebys'`, `'conducting, Bessel'` or `'TWA'`) for $m = 0$, `symm = 1`. The complex torques are written to `magnetic.dat` (see [Output and postprocessing](output.md)). With `rotdyn = 1` the same torques make the mantle and inner core rotate (see [Rotational dynamics](rotdyn.md)).


## Thin wall or conducting inner core?

The thin wall is cheaper to set up, but it is not a substitute for a conducting inner core. When the inner core's field is a skin layer of depth $\delta = \sqrt{2\eta_i/\omega}$, the conducting inner core acts on the fluid exactly like a thin wall, but with a complex, frequency-dependent conductance, $c_1 \approx (1-i)\,\delta/2r_i$ for $\delta \ll r_i$. Kore's thin wall takes a real $c_1$. In tests with the torsional modes at $E = 10^{-7}$ and $10^{-8}$, the best real $c_1$ still left the eigenvalues 1 to 2% off, with errors of 5 to 15 degrees in the phase of the inner-core torque. The mode in which the inner core and the fluid oscillate together was off by about 20%. Use a conducting inner core when the inner core's electromagnetic coupling matters, and the thin wall for a thin conducting layer such as the base of the mantle. The same holds at the CMB: use `mantle = 'resolved'` when the layer is not thin compared with its skin depth, or when its conductivity varies across it.
