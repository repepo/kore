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

where $\eta_i$ is the inner core's magnetic diffusivity relative to the fluid's. In a forced problem the field oscillates at the forcing frequency $\omega$, $\partial_t = i\omega$. At the ICB, the normal field, the tangential $\mathbf{H} = \mathbf{b}/\mu$ and the tangential electric field are continuous. With subscripts $o$ (fluid) and $i$ (inner core), and the permeabilities relative to the fluid's:

$$
f_o = f_i, \qquad (r f_o)' = \frac{(r f_i)'}{\mu_{i2o}}, \qquad g_o = \frac{g_i}{\mu_{i2o}},
$$

$$
(r g_o)' = \eta_i\,(r g_i)', \qquad r^2 \nabla_l^2 f_o = \eta_i\, r^2 \nabla_l^2 f_i, \qquad \nabla_l^2 = \frac{d^2}{dr^2} + \frac{2}{r}\frac{d}{dr} - \frac{l(l+1)}{r^2} .
$$

The last condition, on $\nabla_l^2 f$, is not imposed: it follows from the others (see below).

In the fluid the electric field is $\mathbf{E} = (\Omega\tau)E_\eta\nabla\times\mathbf{b} - \mathbf{u}\times\mathbf{B}_0$. At a no-slip ICB the fluid moves with the inner core, $\mathbf{u} = 0$, and the condition on $g$ above holds as it stands. At a stress-free ICB (`bci = 0`) the fluid slides over the inner core, and the tangential part of $\mathbf{u}\times\mathbf{B}_0$, which is $B_{0r}\,(\mathbf{u}\times\hat{\mathbf{r}})$ since $u_r = 0$, enters the condition on $g$:

$$
(r g_o)' - \eta_i\,(r g_i)' = \frac{r\,S_l}{(\Omega\tau)E_\eta}, \qquad
S_l = \frac{\beta}{l(l+1)} \sum_{l'} \left[ i m\, D_{ll'}\, \frac{(r P_{l'})'}{r} + \frac{l_0(l_0+1) - l(l+1) - l'(l'+1)}{2}\, G_{ll'}\, T_{l'} \right],
$$

where $S_l$ is the $\nabla Y_l$ part of $B_{0r}\,(\mathbf{u}\times\hat{\mathbf{r}})$, $B_{0r} = \beta\, Y_{l_0}^0$ is the radial background field at the ICB (only a poloidal $\mathbf{B}_0$ has one), $P$ and $T$ are the poloidal and toroidal scalars of the flow, and $G_{ll'}$ and $D_{ll'}$ are the $Y_l$ components of $Y_{l_0}^0 Y_{l'}$ and of $(\mathrm{d}Y_{l_0}^0/\mathrm{d}\cos\theta)\, Y_{l'}$. A degree-1 field couples $g$ of degree $l$ to $P_l$ and $T_{l\pm1}$, a degree-2 field to $P_{l\pm1}$ and $T_l$, $T_{l\pm2}$.

The poloidal field is solved with the radial component of the induction equation, as with an insulating inner core, and only the first two conditions are imposed on $f$. With the radial induction equation satisfied on both sides of the ICB, continuity of $f$ makes the radial component of $\nabla\times\mathbf{E}$ continuous, and that is the condition on $\nabla_l^2 f$.

`spin_doctor.py`'s `resid𝐄` tests the inner-core field and the ICB conditions; the other residuals cover the fluid. It is `nan` in forced problems, so check `N_cic` there by the change of the solution (see [Output and postprocessing](output.md#checking-the-results)).

**Resolution.** `N_cic = 64` (the default) is safe for conductivity ratios from $10^{-4}$ to 1. In the tests behind this default, 16 polynomials put the eigenvalue off by $1.6\times10^{-3}$ at `sigma_i2o = 1`, and 48 gave broken eigenvectors for the modes away from the target at `sigma_i2o` = $10^{-2}$. The inner core adds $2\,n_{ic}$ unknowns, which is cheap: at $N = 232$ a torsional-mode run took 40 s and 7.8 GB with a conducting inner core, against 22 s and 7.3 GB with a thin wall.

**Limits.** As `sigma_i2o` goes to zero the solution converges to the insulating one (linearly in `sigma_i2o`).

In a forced problem the inner-core field is confined to a skin layer of thickness $\delta = \sqrt{2(\Omega\tau)E_\eta\eta_i/|\omega|}$ below the ICB, which `N_cic` must resolve. In tests ($\omega = 0.8$, $r_i = 0.35$) `N_cic = 64` gave the field to $3\times10^{-7}$ at $\delta = 0.014\,r_i$ ($E_\eta = 10^{-5}$), but only to $3\times10^{-3}$ at $\delta = 0.0045\,r_i$ ($E_\eta = 10^{-6}$), where it took 128 ($4\times10^{-9}$).

`'conducting, Bessel'` is the forced-problem version, without `N_cic`. At the forcing frequency the equation above, $i\omega\,\mathbf{b}_i = (\Omega\tau)E_\eta\eta_i\nabla^2\mathbf{b}_i$, is solved by $j_l(k r)$ in each degree, with $k^2 = -i\omega/((\Omega\tau)E_\eta\eta_i)$. With $x = k r_i$ and $x_d = x\, j_l'(x)/j_l(x)$, the conditions above become

$$
(r f_o)' = \frac{1 + x_d}{\mu_{i2o}}\, f_o, \qquad (r g_o)' - \frac{1 + x_d}{\sigma_{i2o}}\, g_o = \frac{r\,S_l}{(\Omega\tau)E_\eta} \qquad \text{at } r = r_i,
$$

with $S_l = 0$ at a no-slip ICB, and $x_d = l$ at $\omega = 0$ (`forcing = 12`). Without a forcing frequency there is no $k$, so a run with `forcing = 0` stops. In tests (forcings 1, 7, 9 and 12; `B0` = `'axial'`, `'dipole'` and `'Luo_S2'`; no-slip and stress-free ICB; `mu_i2o`, `sigma_i2o` $\neq 1$; the Alfvén time scale) the two inner cores give the same solution to $10^{-10}$.


## Perfectly conducting inner core

Not available at present; a rewrite is pending. With the radial projection, $f = 0$ at the ICB is enough for the poloidal field: with $\mathbf{u} = 0$ there, the radial induction equation then gives $\nabla_l^2 f = 0$, the zero tangential electric field.


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

The last term of each poloidal condition is first order in $h$. It comes from the curvature of the wall, from matching the potential field at $r_j \pm h$ instead of $r_j$, and from the $l(l+1)/r^2$ part of $\nabla_l^2$ inside the wall.

Guervilly, Wood & Brummell (2013, appendix A) take the limit $h \to 0$ with $h\mu_w$ and $h\sigma_w$ fixed, which needs $\mu_w \gg \mu_0$ for $c$ to stay finite. Their conditions are the ones above without this term, recovered with `muw_cmb = np.inf` (or `muw_icb`); for $p \gg 1$ the term is a correction of relative size about $1/(p\,l)$ to the $c$ term.

For a wall with the permeability of vacuum ($p = 1$, the default, as at the Earth's CMB) the term cancels the $c$ term at leading order: a static field crosses such a wall unchanged, and only its conductance $c_1$ acts. With $c_1 = 0$ and $p = \mu = 1$ the wall conditions are then exactly the insulating ones; without the term they differ from them by $O(l\,c)$.

Against an exact conducting layer (the layer's induction equation integrated across the wall, CMB and ICB, $p$ = 1 to 100, steady and oscillating fields), the conditions with the term are off by $O(h)$ relative to the wall's own effect; without it the error is $O(1)$ for small $p$ (8% at $l = 5$ at the ICB for $p = 1$).

The thin-wall approximation itself needs the wall to be much thinner than its magnetic skin depth, $h \ll \delta_w = \sqrt{2\eta_w/\omega}$: the error on the wall's contribution to an eigenvalue grows as $(h/\delta_w)^2$. Inside the wall $b'' = k^2 b$ with $k^2 = \lambda/\eta_w$, and the conditions keep only the linear profile across it; the first neglected term, the wall's own induction, is

$$
\varepsilon_\text{TWA} = \frac{|k h|^2}{3} = \frac{2}{3}\left(\frac{h}{\delta_w}\right)^2, \qquad \left(\frac{h}{\delta_w}\right)^2 = \frac{c\,c_1\, r_j^2\, |\lambda|}{2\,(\Omega\tau)E_\eta},
$$

of the wall's contribution (since $c\,c_1 = h^2\eta_f/(r_j^2\eta_w)$, for any `muw`), with $\lambda$ the eigenvalue (or the forcing frequency) in the run's time unit. The thin wall is valid while $(h/\delta_w)^2 \ll 1$; $\varepsilon_\text{TWA} \le 1\%$ means $h/\delta_w \lesssim 0.12$. Exact-layer comparisons give errors at or below $\varepsilon_\text{TWA}$. The weaker conditions $h/r_j \ll 1$ and $l\,h/r_j \ll 1$ (the O($h$) terms are in the conditions, so what is left is O($h^2$)) are not checked. `spin_doctor.py` computes it for each solution from its eigenvalue and writes it to `magnetic.dat` (CMB and ICB), and, when a thin wall is used, prints the larger of the two (`εTWA`) in its summary table. With $\sigma_w = \sigma_f$ and `muw = 1` ($c = c_1 = h/r_j$), $(h/\delta_w)^2 = c^2 r_j^2 |\lambda|/(2(\Omega\tau)E_\eta)$; for torsional modes ($\omega/\Omega = a\,L_e$) the 1% limit is about $c_1 \lesssim 0.17\,a^{-1/2}\Lambda^{-1/4}E_\eta^{1/4}$ in rotation units.

The term $\mathcal{D}_l F$ stands for the wall's $\partial_t F/\eta$, taken from the fluid next to it, and $G'$ for the wall's tangential electric field. Both assume the fluid moves with the wall; with a slipping fluid (stress-free wall or `Ek = 0`) the conditions get the $\mathbf{u}\times\mathbf{B}_0$ terms of [Slipping fluid at a conducting wall](#slipping-fluid-at-a-conducting-wall). With a wall at the CMB, keep `B0_cmb = 'wall'` (the default) so that the background field satisfies the wall condition too (see [Background magnetic field](background-field.md#a-thin-conducting-wall-at-the-cmb-b0_cmb)).


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

As at a conducting inner core, the continuity of $\nabla_l^2 f/\sigma$ follows from these. These conditions hold for a fluid that moves with the layer (a no-slip CMB); with a slipping fluid (`bco = 0` or `Ek = 0`) the electric-field condition gets the $\mathbf{u}\times\mathbf{B}_0$ term of the [next section](#slipping-fluid-at-a-conducting-wall). The run stops unless `mu = 1` and no forcing moves the CMB other than by a rotation about $z$ (`forcing = 2`, `9`, and `7` with $m \neq 0$ stop). The background field needs no change: it continues into the layer as a potential field.

| Parameter | Default | Meaning |
|---|---|---|
| `r_surface` | 1.83 | Outer radius of the layer (1.83: the whole mantle of the Earth). |
| `N_mantle` | 64 | Chebyshev polynomials in the layer. |
| `sigma_mantle` | 1 | Conductivity of the layer / fluid: a number (uniform layer), or a numpy expression of `r` and `p`. |
| `sigma_mantle_param` | 0 | The value of `p` in `sigma_mantle`. |

An expression may use the functions allowed in `B0_tor_profile` (`sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `tanh`, `sinh`, `cosh`, `arctan`, `where`, `pi`, `erf`). For example `'1e-2*exp(-p*(r - 1))'` is the exponential profile of PR #17 (A. Nuyt), and `'1e-2*(1 - tanh((r - 1.05)/0.02))/2 + 1e-8'` a layer about 0.05 thick with a smooth top (it needs `N_mantle = 128`).

The profile must be smooth, positive on $[1, r_s]$ and resolved by `N_mantle` polynomials: B holds the `N_mantle`-term Chebyshev series of $r^2\sigma_m$, and the toroidal rows that of $r^2(\ln\sigma_m)'$, with $(\ln\sigma_m)'$ from an interpolant of $\ln\sigma_m$. Where they do not resolve the profile they oscillate: $\sigma_m$ in B can dip below 0, which gives spurious growing modes, and the toroidal modes come out wrong. The run therefore stops when the Chebyshev coefficients beyond `N_mantle` exceed $10^{-6}$ of the largest for $r^2\sigma_m$, or $10^{-6}$ for $\ln\sigma_m$ (the relative error of $\sigma_m$). A step (`where`) or a kink (`abs`) is never resolved: use a smooth transition, e.g. with `tanh` or `erf`. In free-decay tests the profiles that pass give decay rates within $10^{-6}$ of the exact ones.

The solution in the layer is written to `magnetic_mantle.npy`. `just_induction.py` (the induction equation alone, see [Solver settings](solver.md#the-induced-field-of-a-given-flow)) does not include the layer and stops.

Unlike the thin wall, the layer holds at any frequency, for any thickness and any smooth profile that `N_mantle` resolves. For a thin uniform layer, $r_s = 1 + h$ and `sigma_mantle` $= c_1/h$, it tends to the thin wall with `c1_cmb` $= c_1$ (and `c_cmb = 0`) as $h \to 0$, with a difference of order $h$; as `sigma_mantle` goes to 0 it tends to the insulating mantle. Free-decay rates of a conducting sphere or shell inside a uniform layer agree with the exact ones (spherical Bessel functions) to $10^{-11}$.


## Slipping fluid at a conducting wall

At a stress-free wall (`bci = 0`, `bco = 0`), or with `Ek = 0`, the fluid slides over the solid, and the tangential electric field at the interface depends on the frame in which it is taken. Kore continues the fluid's tangential electric field in the frame of the solid,

$$
\hat{\mathbf{r}}\times\left[\mathbf{E}\right] = 0, \qquad \mathbf{E} = (\Omega\tau)E_\eta\nabla\times\mathbf{b} - \mathbf{u}\times\mathbf{B}_0 \ \text{in the fluid}, \qquad (\mathbf{u}\times\mathbf{B}_0)_t = B_{0r}\,(\mathbf{u}_t - \mathbf{v}_s)\times\hat{\mathbf{r}},
$$

with $\mathbf{v}_s$ the velocity of the solid ($\Omega_m\hat{\mathbf{z}}\times\mathbf{r}$ for a mantle that rotates with `rotdyn = 1`, else 0). This is condition C.I of Rekier, Triana, Trinh & Buffett (2023, arXiv:2303.00718). For a true velocity discontinuity first principles give instead the continuity of the field in each medium's rest frame, $\hat{\mathbf{r}}\times[\mathbf{E} + \mathbf{v}\times\mathbf{B}] = 0$ (C.II), which forces a current sheet at the interface. C.I is therefore a modelling choice: the slip stands in for a thin viscous (Ekman-Hartmann) layer that is not resolved, across which the electric field barely changes while the current density does. It holds while that layer is thin against the magnetic skin depth $\sqrt{2(\Omega\tau)E_\eta/|\omega|}$, i.e. for $E_k/E_\eta\cdot|\omega| \ll 1$ (Ekman layer) and a Hartmann layer $\sqrt{E_k E_\eta}/(L_e |B_{0r}|)$ much thinner than the skin depth. For the Earth at diurnal frequencies the skin depth is about 100 m and the Hartmann layer about 16 cm. `assemble.py` prints $E_k/E_\eta\cdot|$`itau`$|$ when a slipping fluid meets a thin wall. With equal conductivities on both sides ($\sigma_m = \sigma_f$) the jump in the tangential current density at the interface, $\sigma B_{0r}(\mathbf{u}_t - \mathbf{v}_s)$, comes from the slip alone.

The slip enters as follows. $S_l$ is the $\nabla Y_l$ part of $B_{0r}(\mathbf{u}_t\times\hat{\mathbf{r}})$ given above for a conducting inner core, and

$$
X_l = -\frac{\beta}{r} \sum_{l'} \left[ \frac{l_0(l_0+1) - l(l+1) - l'(l'+1)}{2}\, G_{ll'}\, \frac{(r P_{l'})'}{r} - i m\, D_{ll'}\, T_{l'} \right]
$$

the $Y_l$ part of $\hat{\mathbf{r}}\cdot\nabla\times(\mathbf{u}\times\mathbf{B}_0) = -\nabla\cdot(B_{0r}\mathbf{u}_t)$, the radial induction of the slip:

- Conducting inner core (`'conducting, Chebys'`, `'conducting, Bessel'`): $(r g_o)'$ in the condition on $g$ becomes $(r g_o)' - r S_l/((\Omega\tau)E_\eta)$ (above).
- Thin wall: in $G \pm c_1 r G'$, $G'$ becomes $G' - r^2 S_l/((\Omega\tau)E_\eta)$, and in the poloidal condition $\mathcal{D}_l F$ becomes $\mathcal{D}_l F + r^2 X_l/(l(l+1)(\Omega\tau)E_\eta)$, since the wall's $\partial_t F$ now includes the radial induction of the slip.
- Mantle layer: $\sigma_m(1)\,(r g)'$ becomes $\sigma_m(1)\left[(r g)' - r^2 S_l/((\Omega\tau)E_\eta)\right]$. The condition on $\nabla_l^2 f$ still follows from $f = f_m$.

`resid𝐄` closes with these terms: with slipping walls and $c = 0$ to $10^{-10}$ or better for thin walls at either boundary and for mantle layers of any thickness, against $10^{-3}$ to 1 without them. With $c \neq 0$ the thin-wall budget is exact only to $O(c\,c_1)$, and the slip makes this error larger (about $4\cdot10^{-3}$ at $c = c_1 = 10^{-2}$, $E_\eta = 10^{-3}$, against $4\cdot10^{-4}$ without slip). A thin uniform layer with $\sigma_m = 1$ and thickness $h$ tends to the thin wall with $c = c_1 = h$ at the same rate with and without slip; without the slip terms the difference is as large as the wall's whole effect. As $E_k$ decreases at fixed $E_\eta = 10^{-3}$ (mantle layer, $\sigma_m = 1$), the eigenvalue with a stress-free CMB and these terms approaches the no-slip one, as $E_k^{0.56}$ (from $8\cdot10^{-2}$ at $E_k = 10^{-3}$ to $6\cdot10^{-3}$ at $10^{-5}$), while without them (C.II) the difference levels off near $1.5\cdot10^{-2}$. With a thick layer ($r_s = 1.83$, $\sigma_m = 1$) and a weak field the tangential field at the CMB is $\mathbf{b}_t = C\,B_{0r}\mathbf{u}_t$ with $|C| = \sqrt{2}\,\delta/(4\eta)$ and $\arg C = 135°$, $\delta = \sqrt{2\eta/|\omega|}$, the local skin-layer solution of Rekier et al. (their eq. 16), to 0.1% at $E_\eta = 10^{-6}$.

## Topographic drag at the CMB

`kappa_cmb` ($\hat\kappa$, default 0) adds a linear drag to a stress-free CMB (`bco = 0`, `Ek > 0`): the tangential traction on the fluid is

$$
E_k\,\sigma_{rt} = -\hat\kappa\,(\mathbf{u}_t - \mathbf{v}_m), \qquad \hat\kappa = \frac{\kappa}{\rho\,\Omega R},
$$

for a stress $\kappa\,\mathbf{u}$ in SI units, with $\mathbf{v}_m$ the mantle's velocity ($\mathbf{v}_m = \Omega_m\hat{\mathbf{z}}\times\mathbf{r}$ with `rotdyn = 1`). With $\mathbf{u} = \nabla\times\nabla\times(P\,\hat{\mathbf{r}}) + \nabla\times(T\,\mathbf{r})$ per degree the boundary rows become

$$
P = 0, \qquad E_k P'' + \hat\kappa P' = 0, \qquad E_k\left(T' - \frac{T}{r}\right) + \hat\kappa\,(T - r\Omega_m\,\delta_{l1}) = 0 \qquad (r = 1),
$$

stress-free for $\hat\kappa = 0$ and no-slip for $\hat\kappa \to \infty$ (the eigenvalues approach the no-slip ones as $1/\hat\kappa$). In a test at $E_k = 10^{-5}$ the damping added per unit $\hat\kappa$ is constant to 1% up to $\hat\kappa \approx 0.03\sqrt{E_k}$, 8% lower at $0.3\sqrt{E_k}$, and falls as $1/\hat\kappa$ beyond $\sqrt{E_k}$. The drag is meant for the form drag of CMB topography on the interior flow, e.g. the topographic dissipation of Rekier et al. calibrated at the diurnal frequency, $\kappa \approx 11.3$ kg m$^{-2}$ s$^{-1}$, $\hat\kappa \approx 4.5\cdot10^{-6}$. In the Earth $\kappa$ is far larger than $\rho\sqrt{\nu\Omega} \approx 0.085$ kg m$^{-2}$ s$^{-1}$, but the form drag acts on the flow outside the viscous layer. In Kore the wall row also sets the slip of a viscous layer, which takes the drag as it stands only while $\hat\kappa \ll \sqrt{E_k}$ (drag-limited); for $\hat\kappa \gtrsim \sqrt{E_k}$ the Ekman layer limits the coupling instead. The magnetic conditions at a conducting CMB get the slip terms of the previous section.

The axial torque on the mantle is $\Gamma = (8\pi/3)(\Omega\tau)\hat\kappa\,(T_1(1) - \Omega_m)$, the viscous torque of `gamma_visc` evaluated through the drag row, and with `rotdyn = 1` it drives the mantle's rotation. `spin_doctor.py` writes the drag's dissipation $D_\text{topo} = 2(\Omega\tau)\hat\kappa\oint|\mathbf{u}_t - \mathbf{v}_m|^2\,dS$ and $\delta\sigma = -D_\text{topo}/(2E)$ to `flow.dat` (columns 21-22) and the summary table (see [Output](output.md)); for a stress-free run with $\hat\kappa = 0$ they are given per unit $\hat\kappa$, an a-posteriori estimate of the drag's damping. It assumes the mode keeps its shape and its other losses, so it is a rough guide (3% off the eigenvalue shift in a test at $E_k = 10^{-3}$, 25% at $E_k = 10^{-5}$): for numbers, run with the drag. The run stops for $\hat\kappa < 0$, `Ek = 0` (no tangential condition) and `forcing = 9`.

## Electromagnetic torques

A conducting layer or inner core lets currents close outside the fluid, and the Lorentz force then exerts a torque on the boundary. `spin_doctor.py` reports $|\Gamma|_{mag}$ for the mantle (with `mantle = 'TWA'` or `'resolved'`) for $m = 0$, `symm = 1` (axial) and $m = 1$, `symm = -1` (equatorial), and $|\Gamma|_{magIC}$ for the inner core (with `'conducting, Chebys'`, `'conducting, Bessel'` or `'TWA'`) for $m = 0$, `symm = 1`. The complex torques are written to `magnetic.dat` (see [Output and postprocessing](output.md)). With `rotdyn = 1` the same torques make the mantle and inner core rotate (see [Rotational dynamics](rotdyn.md)).


## Thin wall or conducting inner core?

The thin wall is cheaper to set up, but it is not a substitute for a conducting inner core. When the inner core's field is a skin layer of depth $\delta = \sqrt{2\eta_i/\omega}$, the conducting inner core acts on the fluid exactly like a thin wall, but with a complex, frequency-dependent conductance, $c_1 \approx (1-i)\,\delta/2r_i$ for $\delta \ll r_i$. Kore's thin wall takes a real $c_1$.

In tests with the torsional modes at $E = 10^{-7}$ and $10^{-8}$, the best real $c_1$ still left the eigenvalues 1 to 2% off, with errors of 5 to 15 degrees in the phase of the inner-core torque. The mode in which the inner core and the fluid oscillate together was off by about 20%.

Use a conducting inner core when the inner core's electromagnetic coupling matters, and the thin wall for a thin conducting layer such as the base of the mantle. The same holds at the CMB: use `mantle = 'resolved'` when the layer is not thin compared with its skin depth, or when its conductivity varies across it.
