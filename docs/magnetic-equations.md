# Magnetic field equations

This page gives the mathematics behind `magnetic = 1`, for readers who want to check the code or extend it: the
equations, the formula of every background field, how the equations become matrix rows, the boundary conditions, what a
differential rotation adds, the energy budgets, and the checks. How to set up a run is on [Magnetic field](magnetic.md);
how the rows were derived and fitted is on [Derivation tools](derivations.md), part A.

## Equations

kore solves for a small perturbation $\mathbf b$ of the imposed field $\mathbf B_0$, together with the flow $\mathbf u$.
Two terms couple them: the flow induces $\mathbf b$ by moving the field lines of $\mathbf B_0$, and the Lorentz force of
the two fields acts on the flow. Both terms are linear in the perturbation, because $\mathbf B_0$ is fixed.

In units of the time scale $\tau$ (`timescale`), the induction equation and the Lorentz force per unit mass in the
momentum equation ([Anelastic equations](anelastic.md)) are

$$
\frac{\partial \mathbf b}{\partial t} = \nabla\times(\mathbf u\times\mathbf B_0) + \mathrm{MagnetD}\,\nabla^2\mathbf b,
$$

$$
\mathbf F_L = \frac{\mathrm{Hendrik}}{\rho_0}\left[(\nabla\times\mathbf b)\times\mathbf B_0 + (\nabla\times\mathbf B_0)\times\mathbf b\right].
$$

The magnetic diffusivity $\eta$ is uniform (`background.diffusivity(..., 'eta')` returns 1). The perturbation field is
written with two scalars, as the flow is but without $\rho_0$, since $\mathbf b$ is solenoidal by itself:

$$
\mathbf b = \nabla\times\nabla\times(f\,\mathbf r) + \nabla\times(g\,\mathbf r).
$$

**Parameters.** The factor `Hendrik` sets the strength of the Lorentz force and `MagnetD` that of the magnetic
diffusion. `set_scales()` computes both from the Lehnert number and the magnetic Ekman number:

| factor | definition | `set_scales()` |
| --- | --- | --- |
| `Hendrik` | $(\tau B_0/r_0)^2/(\rho_0\mu_0)$ | $(\mathrm{Gaspard}\cdot\mathrm{Le})^2$ |
| `MagnetD` | $\eta_0\tau/r_0^2$ | $\mathrm{Gaspard}\cdot\mathrm{Em}$ |

Both numbers are defined with the rotation rate, as `Ek` is:

$$
\mathrm{Le} = \frac{B_0}{\Omega r_0\sqrt{\rho_0\mu_0}}, \qquad \mathrm{Em} = \frac{\eta_0}{\Omega r_0^2} = \frac{\mathrm{Ek}}{Pm},
\qquad \Lambda = \frac{\mathrm{Le}^2}{\mathrm{Em}}
$$

($\Lambda$ the Elsasser number, $Pm$ the magnetic Prandtl number). `Le` and `Em` must be set before `set_scales()` is
called. Besides `'rotation'` and `'viscous'`, `set_scales()` knows two magnetic time scales: `'Alfven'` (Gaspard
$= 1/\mathrm{Le}$, so Hendrik $= 1$) and `'magnetic'` (Gaspard $= 1/\mathrm{Em}$, so MagnetD $= 1$).

**Equatorial symmetry.** The symmetry of $\mathbf B_0$ about the equator decides which degrees $\mathbf b$ takes for a
given flow. A poloidal field of degree $n$ has the symmetry $(-1)^n$ and a toroidal one $(-1)^{n+1}$: the dipole is
antisymmetric, a toroidal field of degree 1 ($B_\phi\propto\sin\theta$) is symmetric, one of degree 2 antisymmetric.
$\mathbf b$ has the symmetry of $\mathbf u$ times that of $\mathbf B_0$ (`ut.bsymm = symm * symmB0`): with an
antisymmetric $\mathbf B_0$, $f$ takes the degrees of the toroidal flow and $g$ those of the poloidal flow; with a
symmetric one, $f$ takes the degrees of the poloidal flow and $g$ those of the toroidal flow. A poloidal and a toroidal
part can be combined only if they have the same symmetry.

## Background fields

The background field is steady and axisymmetric. It is the sum of a poloidal part (`B0`) and a toroidal part (`B0_tor`),
each built from radial profiles times Legendre polynomials and scaled by `cnorm` or `cnorm_tor`. This section gives the
profile of every option (`bin/background_field.py`) and how the amplitude is fixed.

### Poloidal fields

Each poloidal option gives a radial profile $h(r)$ of one degree $l$ (several degrees for `'general'` and `'ferraro'`).
The field follows from it by two curls,

$$
\mathbf B_0 = c_\mathrm{norm}\nabla\times\nabla\times(h(r)\,Y_l^0\,\mathbf r),
$$

with $Y_l^0$ the Schmidt harmonic $P_l(\cos\theta)$. Its radial component and its current are

$$
B_{0r} = c_\mathrm{norm}\,\frac{l(l+1)\,h}{r}\,Y_l^0, \qquad
\nabla\times\mathbf B_0 = c_\mathrm{norm}(D_l h)\,\partial_\theta Y_l^0\,\hat{\boldsymbol\phi}, \qquad
D_l h = h'' + \frac{2h'}{r} - \frac{l(l+1)h}{r^2} .
$$

| `B0` | $h(r)$ | degree, symmetry |
| --- | --- | --- |
| `'axial'` | $r/2$: the uniform field $c_\mathrm{norm}\hat{\mathbf z}$, no current | $l = 1$, antisymmetric |
| `'dipole'` | $1/(2r^2)$: the field of a dipole at the centre, $c_\mathrm{norm}(\cos\theta\,\hat{\mathbf r} + \tfrac12\sin\theta\,\hat{\boldsymbol\theta})/r^3$, no current in the fluid; `ricb` > 0 only | $l = 1$, antisymmetric |
| `'FDM'` | the poloidal free decay mode (equation below) | $l$ = `B0_l` = 1 (antisymmetric) or 2 (symmetric) |
| `'G21 dipole'` | $r/6 - r^3/10$ (Gerick et al. 2021) | $l = 1$, antisymmetric |
| `'Luo_S1'` | $5r - 3r^3$ (Luo & Jackson 2022): 30 times the $h$ of `'G21 dipole'`, so the same field with `cnorm` `'mag_energy'`, `'Schmitt2012'` or `'rms_cmb'`, and 30 times stronger with a numerical `cnorm` | $l = 1$, antisymmetric |
| `'Luo_S2'` | $157r^2 - 296r^4 + 143r^6$ (Luo & Jackson 2022) | $l = 2$, symmetric |
| `'confined'` | $r^l(r_2^2 - r^2)^p$ ($r_1 = 0$) or $r^l(r - r_1)^p(r_2 - r)^p$ in $r_1 \le r \le r_2$, zero outside ($r_1$, $r_2$ = `B0_conf_r1`, `B0_conf_r2`, $p$ = `B0_conf_p`, default 12) | $l$ = `B0_l` = 1 or 2 |
| `'general'` | $\sum_n h_n(r)\,Y_n^0$, `B0_profiles` = {n: 'numpy expression of r, p'} ($p$ = `B0_param`) | any degrees of one parity |
| `'ferraro'` | field lines on the surfaces of constant rotation, so that the rotation does not wind it up: the flux function is $\psi = G(\Delta\Omega)$ ([below](#field-lines-on-the-surfaces-of-constant-rotation-b0-ferraro)) | odd degrees up to `B0_ferraro_nmax`, antisymmetric |

**Free decay mode.** `'FDM'` is the field of degree $l$ that decays most slowly when both walls are insulating. In a
shell (Zhang & Fearn 1995) and in a full sphere (Gubbins & Roberts 1987) it is, with $\beta$ the root that `findbeta`
finds from the first guess `beta` (from the default, 3, the smallest root, that of the slowest mode, for degrees 1 and
2),

$$
h = j_l(\beta r)\,y_{l-1}(\beta) - j_{l-1}(\beta)\,y_l(\beta r) \quad (r_i > 0), \qquad
h = j_l(\beta r) \ \text{with}\ j_{l-1}(\beta) = 0 \quad (r_i = 0).
$$

**At the walls.** A field that does not match a potential field at an insulating wall meets the insulator with a surface
current. The FDM, `'G21 dipole'`, `'Luo_S1'` and `'Luo_S2'` match one at the outer wall ($rh' + (l+1)h = 0$ at $r = 1$),
and the FDM matches one at both walls, so it suits insulating walls. `'axial'` is the uniform field, potential
everywhere. `'dipole'` is a potential field throughout the fluid and matches one at the outer wall; its source lies at
the centre, inside the inner core. `'confined'` is zero at the wall. `'general'` and `'ferraro'` are taken as fields of
currents in the fluid, which need not match the potential field of those currents at either wall (at the inner wall the
potential field that is regular in the core has $rh' - lh = 0$). The induction does
not see the surface current, since the walls' conditions on $\mathbf b$ do not involve $\mathbf B_0$, but it pushes on
the fluid along the wall with a force that goes as $b_r$ there, as that of a toroidal field does. A no-slip wall takes
that push, and the stress rows of a slipping wall hold it
([Current sheets at slipping walls](#current-sheets-at-slipping-walls)).

**Full-sphere fields with an inner core.** `'G21 dipole'`, `'Luo_S1'` and `'Luo_S2'` are defined for a full sphere, and
`B0_ic` says how to continue them into an inner core. `'potential'` (default) adds $\beta_i r^{-(l+1)}$,
$\beta_i = r_i^{l+1}(r_i h' - l h)/(2l+1)$ at $r_i$, so that $rh' - lh = 0$ at the inner boundary: the field of the
currents in the fluid, continued as a potential field into the core.
`'frozen currents'` keeps $h$ as it is, with the
currents below $r_i$ in the core. `B0_ic` has no effect on the other types or with `ricb = 0`.

**Confined field.** `'confined'` fills the layer $r_1 \le r \le r_2$ and is zero outside it. Its purpose: a poloidal
field in a layer where the rotation is uniform is not wound up, so it is a steady background with a differential
rotation ([The Ferraro condition](#the-ferraro-condition)). With an inner core the layer must lie in the fluid
($r_1 \ge$ `ricb`). In a full sphere with $r_1 > 0$, $h$ is evaluated at $\lvert r\rvert$ with the parity $(-1)^l$.

The field must vanish smoothly at the edges of the layer. $h$ and its first $p - 1$ derivatives vanish there, so there
are no surface currents ($B_r \propto h$ and $B_\theta \propto h' + h/r$ are continuous for $p \ge 2$). The volume
current, which involves $h''$, is continuous for $p \ge 3$, the smallest value accepted: the rows of the momentum
equation hold $h'''$, which would carry a delta at the edges for $p = 2$. $p = 4$ makes $h'''$ continuous.

The price of a field smooth only to order $p$ is a slow convergence: its Chebyshev coefficients fall as $k^{-(p+1)}$.
With $p = 12$ the budgets close to $10^{-12}$ at N 96–128 in a shell, but a layer that reaches the centre of a full
sphere needs N 192 (degree 1) to 384 (degree 2) for $10^{-9}$. $h$ is evaluated in factored form: expanded into
monomials, a polynomial of degree $2p + l$ loses all its digits to cancellation near the edges.

**Any degrees.** `'general'` takes one profile $h_n$ per degree, given as expressions. Their derivatives come from
Chebyshev interpolants (degree max(64, 2N), their coefficients from a DCT, without those below $10^{-13}$ of the
largest, whose round-off the derivatives would amplify at the walls); in a full sphere they are evaluated at $\lvert r\rvert$ with the parity
$(-1)^n$, and $h_n$ should go as $r^n$ at the centre (`submatrices.py` warns otherwise). The degrees must have one
parity (one equatorial symmetry); the lowest sets it. `cnorm = 'mag_energy'` sums $\int B_0^2\,dV$ over the degrees.
The couplings come from the recurrence algebra ([Couplings for any background](#couplings-for-any-background)).

### Toroidal fields

A toroidal field is azimuthal: its field lines are circles around the axis. Each option gives a radial profile $k(r)$ of
degree $n$ = `B0_tor_l` (several degrees for `'general'`), scaled by $c_\mathrm{tor}$ = `cnorm_tor`:

$$
\mathbf B_0^{tor} = c_\mathrm{tor}\nabla\times(k(r)\,Y_n^0\,\mathbf r),
$$

$$
B_\phi = -c_\mathrm{tor}\,k\,\partial_\theta Y_n^0 =
\begin{cases} c_\mathrm{tor}\,k\sin\theta, & n = 1,\\ 3c_\mathrm{tor}\,k\sin\theta\cos\theta, & n = 2. \end{cases}
$$

| `B0_tor` | $k(r)$ | |
| --- | --- | --- |
| `'Malkus'` | $r$ | for $n = 1$, $B_\phi = c_\mathrm{tor}\,s$ (Malkus 1967), with a uniform axial current $2c_\mathrm{tor}$ |
| `'wall'` | $r(r - r_i)(1 - r)/(1 - r_i)^2$; with `ricb = 0`, $r^n(1 - r^2)$ | zero at both walls |
| `'user'` | `B0_tor_profile`, a numpy expression of $r$ and $p$ = `B0_tor_param` | its derivatives from a Chebyshev interpolant (degree max(64, 2N), cut as for the poloidal `'general'`); written to `params.dat` after `#` |
| `'general'` | $\sum_n k_n Y_n^0$, `B0_tor_profiles` = {n: 'numpy expression of r, p'}, with $p$ = `B0_tor_param` | any degrees of one parity; derivatives as for the poloidal `'general'` |

**At the walls.** A toroidal field that does not vanish at a wall (`'Malkus'`) ends there in a surface current,
$\mathbf K_0 = \pm B_{0\phi}\,\hat{\boldsymbol\theta}$ at the outer and the inner wall, unless a conductor beyond takes the
field on (a conducting inner core, the resolved mantle or a conducting thin layer, which hold its drop to zero
themselves). The induction does not see it: the walls' conditions on $\mathbf b$ do not involve $\mathbf B_0$, and where
no flow crosses a wall the toroidal $\mathbf u\times\mathbf B_0$ is normal to it, whether the fluid slips along it or not.
The sheet does push on the fluid, with $\mathrm{Hendrik}\,\mathbf K_0\times\mathbf b$, whose part along the wall is
$\mp\mathrm{Hendrik}\,B_{0\phi}\,b_r\,\hat{\boldsymbol\phi}$ per area. A no-slip wall takes that push, and the stress
rows of a slipping wall hold it ([Current sheets at slipping walls](#current-sheets-at-slipping-walls)); with
`rotdyn = 1` the field must vanish at both walls (`background.check_rotdyn`). Inside a conducting inner core $\mathbf B_0$ does not enter either: there
is no flow there, so the matching conditions do not involve it (for a wall that the forcing moves,
[Forced runs](#forced-runs)). `B0 = None` gives a toroidal field alone.

### Normalisation

The profiles have arbitrary scales, so the amplitude is set apart: `cnorm` multiplies the poloidal part and `cnorm_tor`
the toroidal part. Each is a number, or a name that kore turns into a number (`background_field.B0_norm`):

- `'rms_cmb'` (`cnorm` only): the rms of the radial field over the outer wall is 1,
  $c_\mathrm{norm} = \sqrt{2l+1}/(L\,h(1))$, $L = l(l+1)$. Single-degree types only: `'general'` and `'ferraro'` stop
  with a message, and so does `'confined'`, which is zero at the wall.
- `'mag_energy'`: $\int B_0^2\,dV = 1$ over the fluid (Luo & Jackson 2022).
- `'Schmitt2012'`: $\int B_0^2\,dV = 4$.
- `'rms_Bs'` (`cnorm` only): the volume rms of the cylindrical radial field over the fluid is 1,
  $\int B_s^2\,dV = V = \tfrac{4\pi}{3}(1 - r_i^3)$, with $B_s = B_r\sin\theta + B_\theta\cos\theta =
  \sin\theta\,\sum_n\left[L h_n P_n/r - \cos\theta\,(h_n' + h_n/r)P_n'\right]$, $L = n(n+1)$. The degrees are summed
  before squaring (they are not orthogonal in $B_s$), the field includes the `B0_ic` and `B0_cmb` terms, and the integral
  is by Gauss–Legendre quadrature in $r$ and $\cos\theta$ (`background_field.rms_Bs`). `'axial'` stops: its $B_s$ is zero.

For a poloidal field of one degree the integral is

$$
\int B_0^2\,dV = \frac{4\pi L}{2l+1}\int\left[(L + 1)h^2 + 2rhh' + r^2h'^2\right]dr ,
$$

summed over the degrees for `'general'` and `'ferraro'` (they are orthogonal), and taken over the layer only for
`'confined'`. For a toroidal field it is $n_l L\int k^2 r^2\,dr$, with $n_l = 4\pi/(2n+1)$ and $L = n(n+1)$, summed over
the degrees for `'general'`.

## Variable density

With a density that varies with radius, the flow is not solenoidal and the momentum equation carries $1/\rho_0$, so the
magnetic rows gain terms in the derivatives of $\ln\rho_0$. This section lists them. They vanish for `'Boussinesq'`.

The flow is written as

$$
\mathbf u = \rho_0^{-1}\nabla\times\nabla\times(\rho_0 P\,\mathbf r) + \nabla\times(T\,\mathbf r),
$$

so its consoidal part is $P' + P/r + \lambda_1 P$, with $\lambda_1 = (\ln\rho_0)'$, $\lambda_2 = (\ln\rho_0)''$, and
$\nabla\cdot\mathbf u = -\lambda_1 u_r$. The rows were derived from the definitions with `tests/magnetic/derive.py`
(sympy, exact projection on the harmonics). For $\mathbf B_0$ of degree 1:

- rows `f`, columns P: every $d/dr$ on $P$ becomes $d/dr + \lambda_1$ (label `f1bpo0lho1_D0`);
- rows `g`, columns P, diagonal in $l$ only (zero for m = 0), from
  $-\mathbf B_0\nabla\cdot\mathbf u$ and the $\lambda_1 P$ above:

    $$
    2im\left[r h\lambda_1 P' + r h\lambda_2 P + \left(1 - \tfrac{L}{2}\right) r h'\lambda_1 P - \tfrac{L}{2}\,h\lambda_1 P\right];
    $$

- rows `u`, with $\mathbf G$ the Lorentz force per volume:

    $$
    \rho_0\hat{\mathbf r}\cdot\nabla\times\nabla\times(\rho_0^{-1}\mathbf G) = \hat{\mathbf r}\cdot\nabla\times\nabla\times\mathbf G - \frac{\lambda_1}{r}\nabla_1\cdot\mathbf G_H .
    $$

    The extra term, $\lambda_1 r^4\,\nabla_1\cdot\mathbf G_H$ projected on $Y_l$, is, for the toroidal field of the same
    degree and for the poloidal field of degree $l_c = l\pm1$ (`operators.lorentz_rho`),

    $$
    -im\lambda_1\left[(L+2)r^2hg + Lr^3h'g + 2r^3hg'\right], \qquad
    \frac{y}{2}\lambda_1 r^3\left[2a\,h\,D_{l_c}f + L_c b\,(D_1 h) f\right],
    $$

    with $D_l$ as above and

    $$
    y = \langle Y_l\,\vert\,\cos\theta\,Y_{l_c}\rangle, \qquad a = 2 - L_c - L, \qquad b = L_c - 2 - L ;
    $$

- rows `v`, the T columns, the diffusion, the walls and the inner core: no $\rho_0$ terms.

The new labels pair `bpo` with `lho` (two profiles, plus `ohr1` in the rows `u`). They vanish exactly for
`'Boussinesq'`, whose matrices are bitwise those of the uniform-density code. `tol_B0` does not cut these products (they
converge slowly near the surface). `submatrices.py` counts odd profiles in both slots of a label for the parity at
`ricb = 0` (bitwise the same for every earlier label).

## Rows and implementation

This section says where the magnetic unknowns sit in the matrices and which code builds their rows. It matters only to
change or extend the code.

- **Sections.** The induction equation gives two sections of rows, `f` (rows $r^2\,\hat{\mathbf r}\cdot$ of the
  equation) and `g` ($r^2\,\hat{\mathbf r}\cdot\nabla\times$), both in the $C^{(2)}$ basis with two boundary-condition
  rows per block. They are placed after the entropy section:
  `ut.offset = {'u': 0, 'v': n, 'h': 2n, 'f': (2+thermal) n, 'g': (3+thermal) n}` and
  `sizmat = 2n + n*thermal + 2n*magnetic + 2*nic` (`nic` for a conducting inner core, [below](#boundary-conditions)).
  Non-magnetic matrices are unchanged.
- **Labels.** Each operator is a sum of labelled radial products. `decode_label` gives sections `f` and `g` the row
  power $2 - X$ and no $\rho_0$. The profile codes are `bpo` ($c_\mathrm{norm} h^{(k)}$), `bto`
  ($c_\mathrm{tor} k^{(j)}$) and `eta` ($\eta$). The Lorentz labels carry `ohr1` (one power of $\rho_0$ less), so the
  momentum rows hold $r^p$ times the force $\rho_0 \mathbf F_L$. `tol_B0` (default 1e-8) is the relative cut of the Chebyshev series of $r^p h^{(k)}$: it sets
  the band width of the magnetic blocks.
- **Operators** (`operators.lorentz`, `induction`, `magnetic_diffusion`, `b`). They are
  divided by $L^3$ of the row, as every operator of kore, with Hendrik and MagnetD for
  $(\Omega\tau\,\mathrm{Le})^2$ and $\Omega\tau\,\mathrm{Em}$. The sign convention is kore's: A holds the
  right-hand side, B is $-\mathbf b$ in the induction rows (as $-$inertia in the momentum rows).
- **Toroidal field** (`operators.induction_tor`, `lorentz_tor`). Closed forms for a toroidal field of degree 1 or 2. The
  rows were derived with `derive.py` (any $\rho_0$) and fitted in $l$: every coefficient is $im$ (same degree) or the
  Coriolis factors $C_- = \sqrt{l^2-m^2}/(2l-1)$, $C_+ = \sqrt{(l+1)^2-m^2}/(2l+3)$ (columns $l\mp1$) times a polynomial
  in $l$ (independent of $m$). Couplings: rows f from P at the same degree; rows g from P at $l\pm1$ (with $\lambda_1$
  terms) and from T at $l$; rows u from F at $l$ (with $\lambda_1$) and G at $l\pm1$ (with $\lambda_1$); rows v from F
  at $l\pm1$ and G at $l$. Degree 2 (`induction_tor2`, `lorentz_tor2`): the quadrupole angular factors
  (`operators.quad_factors`, the same-degree terms with $l(l+1) - 3m^2$) times polynomials in $l$, couplings up to
  $\Delta l = \pm2$: rows f from P at $l\pm1$; g from P at $l$, $l\pm2$ and T at $l\pm1$; u from F at $l\pm1$ and G at
  $l$, $l\pm2$; v from F at $l$, $l\pm2$ and G at $l\pm1$. `operators.lorentz` and `induction` add the poloidal and the
  toroidal parts (`assemble.py` loops over $\Delta l = -2..2$ and skips absent couplings). Column offsets:
  `operators.moff`.
- **Poloidal field of degree 2** (`operators.induction_pol2`, `lorentz_pol2`, for `'Luo_S2'` and `'FDM'` with
  `B0_l = 2`). Closed forms, derived with `derive.py` ($b_0$ = ('pol', 2), any $\rho_0$) and fitted in $l$, $m$ as for
  the toroidal field of degree 2: `quad_factors` times polynomials in $l$ (the same-degree terms carry $l(l+1) - 3m^2$),
  kept in two tables (row, column, $\Delta l$) of the label expressions. Couplings: rows f from P at $l$, $l\pm2$ and T
  at $l\pm1$; g from P at $l\pm1$ and T at $l$, $l\pm2$; u from F at $l$, $l\pm2$ and G at $l\pm1$; v from F at $l\pm1$
  and G at $l$, $l\pm2$. Two labels are added to those of the dipole (`u2bpo2ohr1_D1`, `u2bpo2ohr1_D0`). spin_doctor
  takes its Lorentz and induction powers from `magnetic_powers_grid` for these fields (`utils4pp.use_grid`).
- **Full sphere** (`ricb = 0`). As for the flow, each block holds the Chebyshev polynomials of one parity on $[-1, 1]$
  (`ut.remroco`): $f$ has the vector parity `vF = symmB0 * vP`, $g$ the opposite; $h$ has the parity $(-1)^l$ of its
  degree and $k$ that of $(-1)^n$ (`utils.odd_profiles`, both slots of a label). There is one boundary-condition row per
  block, the insulating condition at the outer wall (`assemble.bc_b_spherical`). The fields are regular at the centre:
  FDM $j_l(\beta r)$ (analytic derivatives, `background_field.jl_k`), the polynomial fields, 'wall' $= r^n(1 - r^2)$,
  'Malkus' for $n = 1$ only ($k = r$ is not regular for $n = 2$). 'general' and 'user' are evaluated as
  $f(\lvert r\rvert)\,\mathrm{sign}(r)^n$, regular only if $f$ goes as $r^n$ times a series in $r^2$; `submatrices.py`
  warns otherwise ([Regularity at the centre](numerical-method.md#regularity-at-the-centre)).
  `cnorm` and `cnorm_tor` are computed as with an inner core; spin_doctor's inner-wall terms are skipped.
- **Inviscid runs** (`ViscosD = 0`, full sphere only). The Lorentz labels have $D^3$ (rows u) and $D^2$ (rows v), so the
  rows are in $C^{(3)}$ (u) and $C^{(2)}$ (v), with one boundary-condition row ($u_r = 0$) in u and none in v; without a
  field they stay in $C^{(2)}$ and $C^{(1)}$. `decode_label` gives the u and v labels the row powers of the viscous
  rows.
- **Solver.** `solve.py` writes `magnetic.npy` and normalises the eigenvectors (over the
  fluid) to

    $$
    \mathrm{KE} + \mathrm{Hendrik}\int\tfrac12\lvert\mathbf b\rvert^2dV = 1 ,
    $$

    and without the flow (`hydro = 0`) to $\int\tfrac12\lvert\mathbf b\rvert^2dV = 1$.

### Couplings for any background

The closed forms above exist only for fields of degree 1 and 2. For a background of any degrees, `bin/mag_algebra.py`
builds the couplings of the Lorentz force and of the induction from the recurrence algebra that the differential
rotation uses, for

$$
\mathbf B_0 = \sum_n c_\mathrm{norm}\nabla\times\nabla\times(h_n Y_n^0\mathbf r) + \sum_n c_\mathrm{tor}\nabla\times(k_n Y_n^0\mathbf r).
$$

The background vectors are polynomials in $x = \cos\theta$ times $\hat{\mathbf r}$,
$\sin\theta\,\hat{\boldsymbol\theta}$ or $\sin\theta\,\hat{\boldsymbol\phi}$; the products with the perturbation are
sparse matrices on the degrees, times radial monomials; the rows take one radial derivative (product rule); each final
monomial becomes a label with the profile codes `hp` + degree ($c_\mathrm{norm}h_n$) and `kt` + degree
($c_\mathrm{tor}k_n$). [Derivation tools](derivations.md), part A.6, gives the method. The polynomials are kept as
Legendre series and applied to the harmonics through the Legendre recurrence, which keeps the couplings exact to round-off
at any degree (written in powers of $x$, $P_n$ has large coefficients of alternating signs, and their round-off grows by a
factor of about 2.4 per degree).

`mag_couplings = 'algebra'` uses it for every type; `'closed'` (default) keeps the closed forms. `'general'` (poloidal
or toroidal), `'ferraro'` and `B0_wound = 1` have no closed forms and switch to the algebra by themselves. For the types
that have closed forms both give the same eigenvalues to round-off.

## Boundary conditions

At each wall the perturbation field is matched to the field beyond it. The mantle is an insulator, possibly with a
conducting layer at its base, thin (conditions at the wall) or of any thickness (its field solved for), so beyond the
conducting parts the field is a potential field. The inner core is an insulator too (with or without a thin conducting
layer), or a conductor whose field is solved for with the rest.

**Insulating walls** (`assemble.bc_b_spherical`). Matching a potential field gives $(l+1)f + rf' = 0$ and $g = 0$ at the
outer wall, $lf - r f' = 0$ and $g = 0$ at the inner one.

**Conducting inner core.** `innercore = 'conducting, Chebys'` (eigenvalue problems and forced runs) adds the field
inside the inner core to the unknowns (`'conducting, Bessel'`, for forced runs, solves it in closed form instead: [Forced runs](#forced-runs)):

- **Unknowns.** Sections `fic`, `gic` after `f`, `g` (`ut.offset`), with the degrees of `f`, `g` (up to `lmax`). Each block is the
  Chebyshev series on $[-r_i, r_i]$ of one parity (that of the field at the centre), `N_cic`/2 coefficients per degree
  (`ut.Nic`, default `N_cic` = 64; fewer can spoil the eigenvectors of a weakly conducting core);
  `ut.nic` per section.
- **Equation.** The field diffuses (a core that a forcing moves also induces field: [Forced runs](#forced-runs)), with
  $\mu_{i2o}$, $\sigma_{i2o}$ (`mu_i2o`, `sigma_i2o`) the permeability and conductivity relative to the fluid:

    $$
    \frac{\partial\mathbf b}{\partial t} = \mathrm{MagnetD}\,\eta_i\nabla^2\mathbf b, \qquad \eta_i = \frac{1}{\mu_{i2o}\sigma_{i2o}},
    $$

    with rows $r^2\hat{\mathbf r}\cdot$ and $r^2\hat{\mathbf r}\cdot\nabla\times$ divided by $L^3$ as in the fluid, in the
    $C^{(2)}$ basis. Its matrices are built apart from the labels (`submatrices.py`: `r0_D0f_ic`, `r2_D0f_ic`,
    `r1_D1f_ic`, `r2_D2f_ic` and the same for g), with `ut.Dlam` and `bc_variables.Tk` on the inner core's domain (their
    optional domain arguments).

- **Matching at the ICB** (`assemble.bc_b_cic`), one row in each section, the inner one of `f`, `g` and the only one of
  `fic`, `gic`: $f_o = f_i$ ($[b_r]$), $(g + rg')_o = \eta_i (g + rg')_i$ (tangential $\mathbf E$),
  $\mu_{i2o}(f + rf')_o = (f + rf')_i$ and $\mu_{i2o}\,g_o = g_i$ (tangential $\mathbf H$). The other part of the
  tangential $\mathbf E$, which drives $b_r$, is not imposed: with $b_r$ continuous it follows from the radial induction
  equation on both sides, which holds that of the flow in the fluid (for a wall that the forcing moves,
  [Forced runs](#forced-runs)).
- **A fluid slipping along the core** (`bci = 0`; `utils.cic_slips`). The fluid's tangential electric field includes
  $-\mathbf u\times\mathbf B_0$, continued into the core as into a thin layer (condition C.I,
  [Thin conducting layer](#thin-conducting-layer)): $(rg)'_o - r\,S_l/\mathrm{MagnetD} = \eta_i\,(rg_i)'$, with the
  $S_l$ of a thin layer (`assemble.bc_b_cic` through `bc_b_slip`). The radial induction of the slip is in the fluid's own
  equation, and the poloidal matching needs nothing more. A toroidal $\mathbf B_0$ adds nothing: $\mathbf u\times\mathbf B_0$
  is then normal to the wall.
- **Output.** `magnetic_ic.npy`. The eigenvectors are normalised with the fluid's
  energies only (the inner-core field is scaled along).

### Thin conducting layer

A thin conducting layer between the fluid and an insulator carries currents that couple the field across it. The
thin-wall approximation (Roberts, Glatzmaier & Clune 2010; Guervilly, Wood & Brummell 2013, appendix A) replaces the layer
by conditions on the field at the wall, so the layer adds no unknowns. `mantle = 'TWA'` puts one at the base of the
mantle, at the outer wall, `innercore = 'TWA'` one on the inner core, at the inner wall; the conditions include the terms of
first order in the thickness (`assemble.bc_b_twa`).

A layer of thickness $h$, conductivity $\sigma_w$ and permeability $\mu_w$ at the radius $r$ of the wall is described by

$$
c = \frac{h\,\mu_w}{r\,\mu_f}, \qquad c_1 = \frac{h\,\sigma_w}{r\,\sigma_f}, \qquad p = \frac{\mu_w}{\mu_0},
$$

(`c_cmb`, `c1_cmb`, `muw_cmb` or `c_icb`, `c1_icb`, `muw_icb`), with $\mu_f = \mu\,\mu_0$ and $\sigma_f$ the fluid's
permeability and conductivity (`mu`); $h/r = c\,\mu/p$. With $F = rf$, $G = rg$ and
$\mathcal D_l F = F'' - l(l+1)F/r^2 = r\nabla^2_l f$ the conditions are

$$
\text{outer:}\quad \Big(\frac1\mu + l\,c\Big)F' + \frac lr F + \frac{c_1}{\mu}\,r\,\mathcal D_l F + \frac{c\,\mu\,l}{p\,r}\Big(\frac{l+1}{p} - 1\Big)F = 0, \qquad G + c_1\,r\,G' = 0,
$$

$$
\text{inner:}\quad \Big(\frac1\mu + (l+1)\,c\Big)F' - \frac{l+1}{r} F - \frac{c_1}{\mu}\,r\,\mathcal D_l F - \frac{c\,\mu\,(l+1)}{p\,r}\Big(\frac lp + 1\Big)F = 0, \qquad G - c_1\,r\,G' = 0.
$$

In kore's rows (on $f$, $g$) the outer conditions are written as they are, the inner poloidal one times $-1$ and the
toroidal ones divided by $r$, so that with $c = c_1 = 0$ and $\mu = 1$ they are the rows of the insulating walls.

- **The terms in $p$** are of first order in $h$: the curvature of the layer, the potential field matched at $r \pm h$,
  and the $l(l+1)/r^2$ of $\nabla^2_l$ in the layer. They vanish for $p = \infty$ (`np.inf`), the limit of Guervilly et
  al. ($h \to 0$ with $h\mu_w$ fixed). For $p = 1$, a layer with the permeability of vacuum, and $\mu = 1$ they cancel
  the $c$ term for a steady field, which then crosses the layer as it would an insulator.
- **$\mu$** enters only these conditions, so $\mu \neq 1$ needs a thin layer or a conducting inner core at every wall.
- **$\mathcal D_l F$** stands for the layer's $\partial_t F/\eta$, taken from the fluid next to it (uniform
  diffusivity), and $G'$ for the layer's tangential electric field. Both assume that the fluid moves with the solid at
  the wall: no slip, and no flow through it.
- **Validity.** The approximation keeps the linear profile of the field across the layer. Inside it $b'' = k^2 b$ with
  $k^2 = \lambda/\eta_w$, and the first term left out, the layer's own induction, is a relative error

    $$
    \varepsilon_\text{TWA} = \frac{\lvert kh\rvert^2}{3} = \frac23\Big(\frac{h}{\delta_w}\Big)^2, \qquad \Big(\frac{h}{\delta_w}\Big)^2 = \frac{c\,c_1\,r^2\,\lvert\lambda\rvert}{2\,\mathrm{MagnetD}}
    $$

    of the layer's effect ($c\,c_1 = h^2\eta_f/(r^2\eta_w)$, any $p$), with $\lambda$ the eigenvalue in the frame of the
    solid (`utils4pp.twa_error`; spin_doctor's εTWA). The conditions $h \ll r$ and $lh \ll r$ are not checked.

**A fluid slipping along a thin layer** (`bco = 0`, `bci = 0` or `ViscosD = 0`; `utils.twa_slips`). The conditions above
take the tangential electric field of the fluid, $\mathbf E = \mathrm{MagnetD}\,\nabla\times\mathbf b - \mathbf u\times\mathbf B_0$,
in the frame of the solid. kore continues it into the layer (condition C.I of Rekier, Triana, Trinh & Buffett 2023): a
modelling choice, for a slip that stands for a viscous layer that is not resolved, valid while that layer is thin
against the magnetic skin depth. With $u_r = 0$ at the wall, $(\mathbf u\times\mathbf B_0)_t = B_{0r}\,(\mathbf u_t\times\hat{\mathbf r})$,
which enters through two projections on the degree $l$: $S_l$, the $r\nabla Y_l$ part of $B_{0r}(\mathbf u_t\times\hat{\mathbf r})$,
and $X_l = [\hat{\mathbf r}\cdot\nabla\times(\mathbf u\times\mathbf B_0)]_l = -[\nabla\cdot(B_{0r}\mathbf u_t)]_l/r$, the radial
induction of the slip. In the conditions

$$
G' \to G' - \frac{r\,S_l}{\mathrm{MagnetD}}, \qquad \mathcal D_l F \to \mathcal D_l F + \frac{r^2 X_l}{l(l+1)\,\mathrm{MagnetD}},
$$

the second because the layer's $\partial_t F$ now includes the radial induction of the slip. `assemble.bc_b_slip` builds
both for any poloidal $\mathbf B_0$ with the recurrence algebra (`angular.Algebra`): $B_{0r} = \sum_n n(n+1)h_n P_n/r$
over the degrees of the background, times $\mathbf u_t = \sum_k s_k\,r\nabla Y_k + t_k\,r\nabla Y_k\times\hat{\mathbf r}$ with
$s_k = P_k' + P_k/r + (\ln\rho_0)'P_k$ and $t_k = T_k$ at the wall; a toroidal background has no $B_{0r}$. With the
rotating solids the outer wall's frame turns with the mantle, and $t_1 = T_1 - r\,\omega_M$ there, the slip relative
to it ([Rotating solids equations](rotating-solids-equations.md#equations)).

**The background field at a thin outer layer** (`B0_cmb`). A steady $\mathbf B_0$ drives no current in the layer, but
it must meet the steady condition, $(1/\mu + lc)(rh)' + (l + e)h = 0$ at $r = 1$ with $e = c\mu l((l+1)/p - 1)/p$
($e = 0$ for $p = \infty$). With `B0_cmb = 'wall'` (default) the fields of internal currents (`'dipole'`,
`'G21 dipole'`, `'Luo_S1'`, `'Luo_S2'`, `'FDM'`) gain $\alpha r^l$ in $h$ (`background_field.B0_alpha_cmb`), with
$\alpha$ set by that condition. The added field is a potential field: it carries no current and keeps the matching at
the inner wall. For $p = 1$ and
$\mu = 1$ the condition is the insulating one, which these fields already meet. `'plain'` leaves $h$ unchanged; the other
fields are used as given. At a thin layer on the inner core the background is not adjusted (for $p = 1$ and $\mu = 1$ a potential
field below meets its condition).

### Resolved conducting mantle

A layer that is not thin against its magnetic skin depth, or whose conductivity varies across it, cannot be reduced to
conditions at the wall. `mantle = 'resolved'` adds the field of a solid
conducting layer $1 \le r \le r_s$ (`r_surface`) to the unknowns, with an insulator beyond. Its conductivity relative to
the fluid's, $\sigma(r)$ (`sigma_mantle`, a number or an expression of $r$ and `sigma_mantle_param`), may vary with
radius.

- **Unknowns.** Sections `fm`, `gm`, last (after `fic`, `gic`; `ut.offset`), with the degrees of `f`, `g`. Each block is
  the Chebyshev series on $[1, r_s]$ (both parities), `N_mantle` coefficients per degree (`ut.nman` per section).
- **Equation.** The layer is at rest (unless a forcing moves the outer wall: [Forced runs](#forced-runs)) and its field
  diffuses, with the diffusivity $\mathrm{MagnetD}/\sigma$:

    $$
    \frac{\partial\mathbf b}{\partial t} = -\mathrm{MagnetD}\,\nabla\times\Big(\frac1\sigma\nabla\times\mathbf b\Big)
    \quad\Longrightarrow\quad
    \sigma\,\partial_t f = \mathrm{MagnetD}\,\nabla^2_l f, \qquad
    \sigma\,\partial_t g = \mathrm{MagnetD}\Big[\nabla^2_l g - \frac{(\ln\sigma)'}{r}(rg)'\Big].
    $$

    The rows are those of the fluid ($r^2\hat{\mathbf r}\cdot$ and $r^2\hat{\mathbf r}\cdot\nabla\times$, the
    $C^{(2)}$ basis) times $\sigma$, so that $\sigma$ enters only $B$, through $r^2\sigma$, and the toroidal rows through
    $(\ln\sigma)'$; $A$ stays well scaled for a small $\sigma$ or a large contrast. Their matrices are built apart from the
    labels (`submatrices.py`: `r2s_D0_man`, `r0_D0_man`, `r1_D1_man`, `r2_D2_man`, `r1d_D0_man`, `r2d_D1_man`), with
    $(\ln\sigma)'$ from a Chebyshev interpolant of $\ln\sigma$ (`background.dlnsigma_mantle`). `background.check_mantle`
    stops the run if `N_mantle` polynomials do not resolve $r^2\sigma$ and $\ln\sigma$ (coefficients beyond `N_mantle` above
    $10^{-6}$): an unresolved $\sigma$ in $B$ can dip below 0 and give growing modes.

- **Matching at the outer wall** (`assemble.bc_b_mantle`), the outer row of `f`, `g` and the first of `fm`, `gm`: $\mathbf b$
  continuous, $f = f_m$, $(rf)' = (rf_m)'$, $g = g_m$ (with $\mu = 1$ on both sides), and the tangential electric field
  $\eta\nabla\times\mathbf b$ continuous, $\sigma(1)\,(rg)' = (rg_m)'$. Its other part, which drives $b_r$, follows from
  the continuity of $b_r$ and the radial induction equation on both sides.
- **The surface** $r = r_s$, the second row of `fm`, `gm`: $r f_m' + (l+1) f_m = 0$ and $g_m = 0$ (a potential field
  beyond).
- **A fluid slipping along the layer** (`bco = 0` or `ViscosD = 0`; `utils.mantle_slips`). The fluid's tangential
  electric field includes $-\mathbf u\times\mathbf B_0$, continued into the layer as for a thin one (condition C.I):
  $\sigma(1)\,[(rg)' - r\,S_l/\mathrm{MagnetD}] = (rg_m)'$, with the $S_l$ of a thin layer (`assemble.bc_b_slip`, with
  $\sigma(1)$ for $c_1$). The radial induction of the slip is in the fluid's own equation, and the poloidal matching needs
  nothing more.
- **Output.** `magnetic_mantle.npy` (`fm` then `gm`), scaled with the fluid's
  field.

As $r_s \to 1$ with $\sigma\,(r_s - 1)$ fixed the layer tends to a thin layer (`'TWA'`) with $c = r_s - 1$ and $c_1 = \sigma(r_s - 1)$, at
first order in the thickness ([Checks](#checks)).

### Current sheets at slipping walls

Where $\mathbf B_0$ does not meet the field beyond an insulating wall, it ends there in a sheet of current, and the sheet
pushes the fluid along the wall wherever the perturbation's radial field crosses it. A no-slip wall takes that push. A
wall that the fluid slips along exerts no stress along itself, so the fluid's viscous stress must balance the push, and
the stress-free rows hold it (`assemble.bc_u_sheet`).

**The sheet** (`background.sheet_jump`). Beyond an insulator, or a thin layer that does not conduct ($c_1 = 0$), no
current flows, so the field there is the one that the wall's steady condition sets. In a sheet on the fluid's side of
the wall, the tangential $\mathbf B_0$ drops from its value in the fluid to the value that meets that condition with the
same $B_{0r}$. The drop $\Delta\mathbf B_{0t}$ carries the current $\mathbf K_0 = \mp\hat{\mathbf r}\times\Delta\mathbf B_{0t}$
at the outer and the inner wall. A conducting inner core, the resolved mantle and a conducting thin layer take the field
on and hold any drop of it themselves, with no sheet in the fluid. Degree by degree:

- **Toroidal.** Nothing beyond carries a toroidal field, so $\Delta\mathbf B_{0t} = B_{0\phi}\,\hat{\boldsymbol\phi}$
  at the wall.
- **Poloidal.** $\Delta\mathbf B_{0t} = c_\mathrm{norm}\,a_n\,r\nabla Y_n^0$, with $a_n$ from the steady condition of a
  thin layer with $c_1 = 0$ ([Thin conducting layer](#thin-conducting-layer); an insulator is $c = 0$, $\mu = 1$, which
  gives $(rh_n' + (n+1)h_n)/r$ at the outer wall and $(rh_n' - nh_n)/r$ at the inner one):

    $$
    a_n = \frac{D\,(rh_n)' + (n + e)\,h_n}{r\,D}, \qquad D = \frac1\mu + n\,c, \qquad e = \frac{c\,\mu\,n}{p}\Big(\frac{n+1}{p} - 1\Big) \qquad \text{(outer)},
    $$

    $$
    a_n = \frac{D\,(rh_n)' - (n + 1 + e)\,h_n}{r\,D}, \qquad D = \frac1\mu + (n+1)\,c, \qquad e = \frac{c\,\mu\,(n+1)}{p}\Big(\frac np + 1\Big) \qquad \text{(inner)},
    $$

    with $e = 0$ for $p = \infty$. A field that goes on through the wall has no sheet: `'axial'` at either wall (imposed
    from beyond the outer wall, it is a potential field in an insulating core too), and at the inner wall `'dipole'` and
    the fields of `B0_ic = 'frozen currents'`, whose sources lie in the core (`background.pol_continues`; `'confined'` is
    zero there). The other fields are those of sources inside the wall (currents in the fluid, as `'general'` and
    `'ferraro'` are taken whatever their profiles; at the outer wall also the source of `'dipole'` at the centre and the
    currents of `'frozen currents'` in the core), and end in a sheet where they do not meet the condition: the fields
    that `B0_cmb = 'wall'` adjusts meet it at the outer wall, and the FDM, `'G21 dipole'`, `'Luo_S1'` and `'Luo_S2'` meet
    it at the inner wall only where the permeability does not jump there.

A degree counts where its drop is above the round-off of its profile at the wall: $10^{-10}$ of the largest value of
that degree of $\mathbf B_0$ over the fluid, or for a poloidal degree the bound on $h_n'$ at a wall that moves
([Forced runs](#forced-runs)) where that is larger. So the fields that meet the condition by
construction have no sheet.

**The push.** The force on the sheet, $\mathrm{Hendrik}\,\mathbf K_0\times\mathbf b$, has a part along the wall,
$\mp\mathrm{Hendrik}\,b_r\,\Delta\mathbf B_{0t}$ at the outer and the inner wall, through $b_r$, which crosses the
sheet unchanged. The viscous stress of the fluid balances it, with one form at both walls, since the fluid lies on the
other side of the sheet at the inner wall:

$$
\mathrm{ViscosD}\,\rho_0\nu\left[r\,\partial_r\Big(\frac{\mathbf u_t}{r}\Big) + \nabla_H u_r\right] = -\mathrm{Hendrik}\,b_r\,\Delta\mathbf B_{0t}.
$$

The stress-free rows ([Boundary conditions](numerical-method.md#boundary-conditions)) are $r^2$ times the
$r\nabla Y_l$ part of the bracket (the row of $P_l$) and $r$ times its $r\nabla Y_l\times\hat{\mathbf r}$ part (that
of $T_l$). They gain $r^2 q$ and $r q$ times the same parts of $b_r\,\Delta\mathbf B_{0t}$, with
$q = \mathrm{Hendrik}/(\mathrm{ViscosD}\,\rho_0\nu)$ at the wall. With $b_r = \sum_k k(k+1)\,f_k\,Y_k/r$ these are entries on the columns of $f$ at the wall, from the
recurrence algebra (`angular.Algebra`, as in `assemble.bc_b_slip`). A drag at the same wall
([Drag at the outer wall equations](drag-equations.md)) adds its own terms to the same rows.

- **Zero radial stress** (`bco` or `bci` 2). The fluid crosses the wall, and the rows leave out the normal part of the
  push, $\mathrm{Hendrik}\,\Delta\mathbf B_{0t}\cdot\mathbf b_t$ along the fluid's outward normal (the drop of the
  magnetic pressure across the sheet), as their zero viscous radial stress leaves out the pressure itself.
- **The sheet's push on $\mathbf B_0$** itself, $\mp\mathrm{Hendrik}\,B_{0r}\,\Delta\mathbf B_{0t}$, is steady: it is part
  of the background's Lorentz force, which kore leaves out
  ([When the ignored Lorentz force matters](#when-the-ignored-lorentz-force-matters)).
- **Budgets.** The push works on the fluid through the viscous stress at the wall, so its power is in the work of that
  stress (`flow.dat` columns 24–25), which residσ and resid𝐄 count, and its torque in the viscous torque at the wall.
  The magnetic torque, taken with $\mathbf B_0$ on the fluid's side, holds that of the sheet's own currents, which the
  viscous one balances: only their sum (`flow.dat` columns 31–38) is the torque through the wall, which resid𝐋 counts.
  These budgets close with the push or without it, so they do not check it: the smoothed sheets and the angular
  momentum do ([Checks](#checks)).

**Stops** (`background.check_magnetic`):

- an inviscid fluid (`ViscosD = 0`, a full sphere), which has no stress at the wall to take the push: the push would
  drive a layer of no thickness along the wall, which its equations do not resolve;
- a deformed body (`centrifugal = 1`), where the push on the deformed wall is not derived;
- a field that goes on beyond a slipping wall (above) where the permeability jumps there: $\mu \neq 1$, or a thin layer
  with $c \neq 0$ and $p \neq 1$. The field beyond, and so the sheet, depends on how its sources meet the jump (for
  `'axial'` at the inner wall, on how the jump bends the imposed field), which kore does not model.

With `rotdyn = 1` a toroidal $\mathbf B_0$ must vanish at both walls (`background.check_rotdyn`).

## Forced runs

A forced run (`forcing` other than 0) solves the equations at the one frequency $w$ of the
forcing (`forcing_frequency`), with every perturbation going as $e^{iwt}$ and the forcing in the right-hand side:

$$
(A - i\,w\,B)\,\mathbf x = \mathbf b .
$$

- **Rows.** The field enters as the flow does: $A$ is the eigenproblem's, not normalised, and every row of the field, in
  the fluid, a conducting inner core and a resolved mantle, gets $-i\,w\,B$, with $B = -\mathbf b$ there
  (`assemble.py`). A frequency scan of forcing 11 shifts this matrix by $-i\,(w' - w)\,B$ (`B.npz`, written without the
  normalisation), so it may be assembled at any `forcing_frequency`.
- **Kinematic induction** (`just_induction.py`) in a forced run takes $A$ as it is for its default
  $\lambda = i\,w$. Another $\lambda$ takes $A - (\lambda - i\,w)B$, so it needs `B.npz`, which forcing 11 alone
  writes: in another forced run it stops.
- **Walls that turn about the axis** (forcing 7 with $m$ = 0, forcing 12). At a no-slip wall the fluid turns with the
  wall, and so does a conducting solid behind it. A rotation $\mathbf v = \Omega_w\hat{\mathbf z}\times\mathbf r$
  leaves the axisymmetric $\mathbf B_0$ as it is, $\nabla\times(\mathbf v\times\mathbf B_0) = 0$, so the solid's
  equations hold no motion; and at the wall the tangential part of $\mathbf u\times\mathbf B_0$, $u_\phi B_{0r}$, is
  the same on both sides, so the matching rows hold none either: they are those of walls at rest. The same holds with
  any permeability, since $B_{0r}$ is continuous and the turning surface moves into itself. Forcing 12 sets
  $T_1 = \Omega_w r_w/2$ at each no-slip wall ($\Omega_w$ = `forcing_amplitude_cmb`, `forcing_amplitude_icb`) at
  $w = 0$; a rigid rotation of the fluid and the solids together then induces no field
  ([Spherical Couette flow](couette-flow.md)).
- **Libration with a stress-free inner wall.** The libration of the walls (forcing 7, $m$ = 0) takes `bci = 0` too. The inner wall's row is then the stress-free one, which holds no libration of the inner core: the
  libration does not reach the fluid through the wall's stress. Across a poloidal field it would reach it through a
  conducting inner core or a conducting thin layer on it, by the $\mathbf u\times\mathbf B_0$ of the slip relative to the
  core, which is not built: that case stops (`background.check_forced_magnetic`). With an insulating core, or a toroidal
  $\mathbf B_0$ alone, `forcing_amplitude_icb` has no effect (`assemble.py` says so).
- **The libration in the frame of the mantle** (forcing 8, [Libration](libration.md)). Its right-hand side is
  $-A\mathbf x$ on the interior rows of every equation of the fluid, with $\mathbf x$ the rigid libration of the mantle
  ($T_1 = iwA\thinspace r/2$, `utils.lifted_flow`; `assemble.lifted_rhs`), the induction rows included. Since
  $\nabla\times(\mathbf x\times\mathbf B_0) = 0$ these get nothing, and the field is that of forcing 7 with the inner
  core's angle $A + A_{IC}$, while the flow is forcing 7's minus $\mathbf x$. The walls are at rest, so a fluid that slips
  along a stress-free wall slips relative to the solid behind it, with the rows of a slipping fluid at a wall at rest.
  The libration of the inner core relative to the mantle behind a stress-free inner wall stops as for forcing 7.
- **The eccentricity tide** (forcing 2). It moves the walls along themselves too, with a tangential velocity that is not
  a rotation. A conducting solid behind such a wall would deform along it and induce a field, which no row holds: the
  run needs insulating walls (`background.check_forced_magnetic`).

**The Bessel inner core** (`innercore = 'conducting, Bessel'`). At one frequency the field of a
uniform conducting inner core, regular at the centre, is known: each scalar goes as $j_l(kr)$, with

$$
k^2 = -\frac{i\,w}{\eta_i\,\mathrm{MagnetD}}, \qquad \eta_i = \frac{1}{\mu_{i2o}\,\sigma_{i2o}},
$$

and $w$ in the frame of the core: `forcing_frequency`, plus $m\,$Gaspard$\,\Delta\Omega(r_i)$ with a differential rotation
(`utils.bessel_ic`). The four matching conditions of `'conducting, Chebys'`, with the core's field eliminated, leave two
rows at the inner wall (`assemble.bc_b_bessel`), with $x_d = x\,j_l'(x)/j_l(x)$ at $x = k\,r_i$:

$$
\Big(\frac{1 + x_d}{\mu_{i2o}} - 1\Big)f - r_i f' = 0, \qquad g - \frac{\sigma_{i2o}}{1 + x_d}\,(g + r_i g') = 0,
$$

the rows of an insulating core for $\sigma_{i2o} \to 0$ with $\mu_{i2o} = 1$ (the poloidal one already for $w \to 0$). A
fluid slipping along the core (`bci = 0`) puts $g + r_i g' - r_i S_l/\mathrm{MagnetD}$ in the second, as for
`'conducting, Chebys'` (`assemble.bc_b_spherical`). `utils.xdlogjl` computes
$x_d = l - x\,j_{l+1}(x)/j_l(x)$, from scipy's exponentially scaled Bessel functions where $\lvert x\rvert > l$ and from the
continued fraction of $j_{l+1}/j_l$ (modified Lentz) elsewhere. These rows hold at that frequency only: the run stops in
an eigenvalue problem and in a frequency scan, and `just_induction.py` with another $\lambda$. spin_doctor takes the
core's energy and Ohmic loss in closed form (`utils4pp.bessel_budget`): since $k^2$ is imaginary,

$$
\int_0^{r_i} r^2\Big\lvert\frac{j_l(kr)}{j_l(kr_i)}\Big\rvert^2dr = -\frac{r_i\,\mathrm{Im}\,x_d}{\mathrm{Im}\,k^2}, \qquad
\int_0^{r_i}\big(l(l+1)\lvert f\rvert^2 + \lvert(rf)'\rvert^2\big)dr = r_i\,(1 + \mathrm{Re}\,x_d)\,\lvert f(r_i)\rvert^2 ,
$$

the first $r_i^3/(2l+3)$ at $k = 0$. `'conducting, Bessel'` writes no field files of the core.

**A boundary that moves** (`forced_walls`). Forcings 6, 9, 10 and 14, and 7 with $m$ = 2, stand for a boundary that
moves or is not spherical: the wall stays at the sphere, and the forcing imposes $u_r \neq 0$ there
(`background.moves_radially`). The tangential electric field is continuous at the wall, and in a conducting solid it is
$\mathrm{MagnetD}\,\eta_s\nabla\times\mathbf b - \mathbf v_s\times\mathbf B_0$, with $\mathbf v_s$ the velocity of the
solid's material, so the matching depends on how the solid moves. Where a no-slip wall moves along itself, as in a
libration, the solid turns with it, which induces nothing in the axisymmetric $\mathbf B_0$. For the radial motion
`forced_walls` chooses between two consistent pictures; `None`, the default, takes `'rigid'` for forcings 6 and 7 and
`'deforming'` for 9, 10 and 14 (`background.forced_walls`). The eccentricity tide (forcing 2) moves both walls, radially
and along themselves; it needs insulating walls, so of this section only the permeability below applies to it.

- **`'rigid'`**: the solid is at rest and only its shape moves past the sphere, so the flow passes through the
  wall. The fluid's tangential electric field, $\mathbf u\times\mathbf B_0$ included, is continued into the solid, as for a
  slipping fluid. With $u_r \neq 0$, $(\mathbf u\times\mathbf B_0)_t$ gains $u_r\,\hat{\mathbf r}\times\mathbf B_{0t}$, for a
  poloidal and a toroidal background. With $S_l$ its part on $r\nabla Y_l$ and
  $X_l = [\hat{\mathbf r}\cdot\nabla\times(u_r\hat{\mathbf r}\times\mathbf B_0)]_l = [\nabla_H\cdot(u_r\mathbf B_{0t})]_l$
  its radial induction,

    $$
    G' \to G' - \frac{r\,S_l}{\mathrm{MagnetD}}, \qquad \mathcal D_l F \to \mathcal D_l F + \frac{r^2 X_l}{l(l+1)\,\mathrm{MagnetD}}
    $$

    in the two conditions of a thin conducting layer, and the first in the electric-field row of a conducting inner core
    (both kinds) and of the resolved mantle (`assemble.bc_b_radial`, with $u_r = l(l+1)P/r$ on the columns of the flow's
    $P$ at the wall, from the recurrence algebra). There the toroidal part of the tangential electric field follows from
    $b_r$ continuous and the radial induction on both sides, with $u_r$ in the fluid and none in the solid: it matches a
    solid at rest too.

- **`'deforming'`**: the solid deforms with the boundary. Its surface moves with the fluid across the wall,
  $v_{s,r} = u_r$, so the $\mathbf u\times\mathbf B_0$ of the radial flow cancels in the matching, and the rows at the wall
  are those of a wall at rest without it. Along a slipping wall the fluid's motion relative to
  the solid keeps its $\mathbf u\times\mathbf B_0$ in the electric-field row (`assemble.bc_b_slip`), as at a wall at rest.
  Inside a thick conducting solid the motion induces a field too (below).

**A deforming solid.** With `'deforming'`, the resolved mantle and `'conducting, Chebys'` behind a wall that the forcing
moves (`background.deforms`) move inside as well, across the background field, and the field that this motion induces is
solved for with the rest:

- **Motion.** The simplest one that matches the wall, $\mathbf v_s = \nabla\times\nabla\times(P_s\,Y_l\,\mathbf r)$ with
  $P_s = P(r_w)\,\phi_l(r)$, $P(r_w)$ the flow's at the wall (`background.deform_profile`). In the mantle $\phi = 1/r$: a
  radial motion, $v_r = u_r(1)/r^2$, which keeps the volume of every shell. In the core
  $\phi_l = \tfrac12\big[(l+3)\,x^l - (l+1)\,x^{l+2}\big]$ with $x = r/r_i$: the polynomial of lowest degree that is
  regular at the centre with $\phi_l = 1$ and $(r\phi_l)' = 0$ at $r_i$. Both give the fluid's $u_r$ at the wall and no
  motion along it, so $\mathbf v_s = \mathbf u$ at a no-slip wall.
- **The background inside the solid** (`background_field.solid_profiles`). Each degree of $\mathbf B_0$ is continued from
  the wall so that it is continuous there, with $x = r/r_w$: a poloidal $h_n$, with $h_n'$, as the potential field
  $a\,x^n + b\,x^{-(n+1)}$ in the mantle and as $a\,x^n + b\,x^{n+2}$ in the core (regular, and a potential field where
  $\mathbf B_0$ meets the potential condition at the inner wall; for `'dipole'` it carries currents in place of the
  source at the centre); a toroidal $k_n$ as the steady field of a uniform
  conductor, $a\,x^n + b\,x^{-(n+1)}$ with $k_n = 0$ at $r_s$ in the mantle, $k_n(r_i)\,x^n$ in the core.
- **Rows** (`assemble.deform_rows`). With
  $\mathbf v_s\times\mathbf B_0 = \sum_l q_l\,Y_l\hat{\mathbf r} + s_l\,r\nabla Y_l + t_l\,r\nabla Y_l\times\hat{\mathbf r}$,
  from the recurrence algebra (`mag_algebra.induction_monomials`), the solid's equations gain

    $$
    \partial_t f \mathrel{+}= t_l, \qquad \partial_t g \mathrel{+}= \frac{q_l - (r s_l)'}{r},
    $$

    times $\sigma$ in the mantle's rows. For each degree $k$ of the flow this is a known function of $r$ times $P_k(r_w)$.
    Its Chebyshev series on the solid's interval, exact in the core where it is a polynomial, fills the solid's rows below
    the matching rows, on the columns of $P_k$ through its Chebyshev polynomials at the wall; in the core it has the parity
    of the rows, which is checked.

- **Stop** (`background.check_forced_magnetic`). `'conducting, Bessel'` behind an inner wall that the forcing moves,
  since the field that the core's motion induces is not a Bessel function.

**A thin conducting layer under `'deforming'`.** The solid deforms with the boundary, and the thin conducting layer at
its surface goes with it. The layer's conditions take $\partial_t F$ and the tangential electric field from the fluid
next to it, which holds that motion. One part is not there: a toroidal $\mathbf B_0$ that is not zero at the wall falls to
zero across the layer, toward the insulator beyond, so the layer's motion meets half of it on average
(`background.thin_drop`). Its conditions gain half the `'rigid'` terms of
the toroidal part alone ($S_l$ and $X_l$ with half the toroidal $\mathbf B_{0t}$ at the wall, `assemble.bc_b_radial`).
This is the limit of a thin resolved mantle that deforms ([Checks](#checks)). A poloidal $\mathbf B_0$
changes by $O(h)$ across the layer and needs nothing more.

**Permeability at a wall that moves.** A surface where the permeability jumps separates matter that $\mathbf B_0$
magnetizes differently, so moving it moves magnetization through $\mathbf B_0$, which changes the field. The rows above
hold with the same permeability on both sides of every surface that moves. A wall that the forcing moves, where
$\mathbf B_0$ reaches it, therefore needs $\mu_{i2o} = 1$ at an inner wall with a conducting inner core behind it, and
$\mu = 1$ at any other, in both pictures.
A thin layer holds $(p/\mu)\,\mathbf B_{0t}$ of the part of $\mathbf B_0$ that enters it ($\mathbf H_t$ is continuous),
and its $p$ must be 1 where that part reaches the wall:

- the poloidal part, in a layer with a thickness ($c \neq 0$), and in a conducting layer at rest (`'rigid'`), whose
  current is driven by $u_r\hat{\mathbf r}\times$ its own field, which its rows take as the fluid's;
- the toroidal part, in a conducting layer ($c_1 \neq 0$): at rest for the same reason, and deforming because the layer
  then meets $p/(2\mu)$ of it, not half.

Elsewhere $p$ takes any value: in an insulating layer ($c = c_1 = 0$), in a layer that does not conduct under a toroidal
$\mathbf B_0$ alone, which holds none of it, and in a conducting layer without a thickness ($c = 0$) that deforms with the
boundary under a poloidal $\mathbf B_0$, which carries the fluid's tangential electric field in its own frame whatever its
permeability. The run stops otherwise (`background.check_forced_magnetic`). $\mathbf B_0$ reaches the wall where $h_n$ or
$h_n'$ (its poloidal part) or $k_n$ (its toroidal part) of some degree is not zero there, to $10^{-6}$ of the largest of
them over the fluid; $h_n'$ to $10^{-12}\,n^2\,2/(1 - r_i)$ instead where that is larger, ten times a bound on the
round-off of the slope of a profile interpolated at degree $n = \max(64, 2N)$ (`background.b0_at_wall`: `'confined'` never reaches a wall,
the toroidal `'wall'` vanishes at both; a field meant to vanish there needs an `N` that resolves it). Walls that stay in
place take any permeability.

**Left out**, in both pictures: the matching holds at the sphere, not at the displaced boundary, so the displacement
$\xi_r$ times the jump of $\partial_r\mathbf B_0$ across the wall (the background's current there) is not in it, nor the
displacement of a current sheet of $\mathbf B_0$ at the wall
([Current sheets at slipping walls](#current-sheets-at-slipping-walls)).


## With a differential rotation

A differential rotation acts on magnetic field lines in two ways. It shears the perturbation field, which kore includes
in the linear problem: this section gives that term. It also winds up the background field, which kore holds fixed: the
subsections below give the condition under which nothing is wound up (the Ferraro condition), the set-ups that avoid the
winding, the one that balances it, and what the background leaves out.

With `diff_rot = 1` the background flow is $\mathbf U_0 = G\,\Delta\Omega\,r\sin\theta\,\hat{\boldsymbol\phi}$ ($G$ =
`Gaspard`), and the induction equation gains

$$
\nabla\times(\mathbf U_0\times\mathbf b) = G\left[-im\,\Delta\Omega\,\mathbf b + \sin\theta\,(r\,\partial_r\Delta\Omega\,b_r + \partial_\theta\Delta\Omega\,b_\theta)\,\hat{\boldsymbol\phi}\right]
$$

(the $\Omega$-effect, and the Doppler shift of the field). It is built for any $\Delta\Omega = \sum_n f_n P_n$ with the
recurrence algebra of the momentum terms (`diff_rot.induction_block`; [Derivation tools](derivations.md), part B): F→f
and G→g couple at even $\Delta l$ up to $n$, F→g (a poloidal field sheared into a toroidal one) at odd $\Delta l$ up to
$n + 1$. The Lorentz force needs nothing new: it does not involve $\mathbf U_0$.

**Conducting inner core.** A solid core cannot be sheared, so it must rotate rigidly with the fluid at its surface:
$\Delta\Omega(r_i, \theta) = c$ ($f_n(r_i) = 0$ for $n > 0$, `diff_rot.rigid_ricb`), or the run stops. Its field then
gains $-imGc\,\mathbf b$. The matching conditions are unchanged, because only $b_r$, which is continuous, enters the
tangential part of $\mathbf U_0\times\mathbf b$.

**A solid with a thin layer that conducts** ($c_1 \neq 0$) turns with the fluid next to it in the same way, so
$\Delta\Omega$ must be uniform in latitude at each such wall (`diff_rot.rigid_at`), or the run stops; a layer that
does not conduct ($c_1 = 0$) takes any rotation. In the frame of the solid the layer's
conditions are those above, and $\mathbf U_0\times\mathbf b$ cancels there with the solid's own rotation. The **resolved mantle** turns with
$\Delta\Omega(1) = c$, uniform in latitude, and its field gains $-imGc\,\mathbf b$ like a conducting inner core.

### The Ferraro condition

The rotation winds up a poloidal field line unless the rotation rate is the same all along it. So nothing is wound up
when the rotation rate is constant along every poloidal field line: Ferraro's law of isorotation. kore checks this at
the start of a run.

The winding is the induction of the background flow on the background field,

$$
\nabla\times(\mathbf U_0\times\mathbf B_0) = G\,r\sin\theta\,(\mathbf B_0\cdot\nabla\Delta\Omega)\,\hat{\boldsymbol\phi} .
$$

It vanishes always for a toroidal $\mathbf B_0$ ($\mathbf U_0\times\mathbf B_0 = 0$), and for a poloidal one only if
$\mathbf B_0\cdot\nabla\Delta\Omega = 0$. The linear problem does not contain this term: it belongs to the background.
`background.ferraro_mismatch` reports its relative size,

$$
\frac{\max\lvert r\,\mathbf B_0\cdot\nabla\Delta\Omega\rvert}{\max\lvert\mathbf B_0\rvert\,\max\lvert r\nabla\Delta\Omega\rvert} ,
$$

over the fluid at the start of every run with `diff_rot = 1` and a poloidal $\mathbf B_0$, except with `B0_wound = 1`
(computed from the shapes $f_n$, with a fourth-order difference in $r$, and the poloidal profiles, on a grid). Above
$10^{-7}$ the parameter `ferraro_policy` decides: `'warn'` (default) prints a warning and carries on, the background
being assumed to be maintained by something outside the model; `'refuse'` stops the run.

The background is steady in two ways. Nothing is wound up with:

- a toroidal field, of any shape;
- a `'confined'` field whose layer lies where $\Delta\Omega$ is uniform ([Poloidal fields](#poloidal-fields));
- `B0 = 'ferraro'`, a poloidal field whose field lines lie on the surfaces of constant rotation;
- `diff_rot_type = 'ferraro'`, a rotation that is constant along the field lines.

The winding goes on but is balanced by magnetic diffusion with any poloidal field and `B0_wound = 1`, which adds the
azimuthal field that the shear winds up (the run then reports that balance instead).

### Field lines on the surfaces of constant rotation: `B0 = 'ferraro'`

A differential rotation winds up a poloidal field whose field lines cross the surfaces of constant rotation.
`B0 = 'ferraro'` avoids the winding: kore uses a poloidal field whose field lines lie on the surfaces of constant
rotation, so nothing is wound up, no azimuthal field is created, and the background is steady. The rotation is kept as
given. Where the rotation is uniform the field vanishes, so this field lives only where the rotation varies. It is the
opposite of `B0_wound = 1` ([below](#the-wound-toroidal-field-b0_wound-1)), which keeps a field that the shear crosses
and adds the azimuthal field that the shear winds up from it.

**Method.** An axisymmetric poloidal field can be written with a flux function $\psi$, as
$\mathbf B_0 = \nabla\psi\times\nabla\phi$: its field lines are the lines of constant $\psi$ in a meridional plane. kore
takes $\psi = G(\Delta\Omega)$, a function of the rotation rate. Every surface of constant rotation is then a surface of
constant $\psi$, so $\mathbf B_0\cdot\nabla\Delta\Omega = 0$; and where $\Delta\Omega$ is uniform,
$\nabla\psi = G'\,\nabla\Delta\Omega = 0$, so the field vanishes. In kore's form $\psi = r\sum_n h_n(1 - x^2)P_n'(x)$,
$x = \cos\theta$, and the functions $(1 - x^2)P_n'$ are orthogonal with the weight $1/(1 - x^2)$, so the profiles are

$$
h_n(r) = \frac{2n+1}{2n(n+1)}\frac{1}{r}\int_{-1}^{1}\psi(r, x)\,P_n'(x)\,dx ,
$$

computed by Gauss–Legendre quadrature. $\Delta\Omega$ has only even $P_n$, so only odd $n$ appear: the field is dipolar.
$G$ = `B0_ferraro_G`, a numpy expression of `w` = $\Delta\Omega$ and `p` (`B0_param`); the derivatives of $h_n$ come
from Chebyshev interpolants of degree max(64, 2N).

**Conditions.** `background.ferraro_report` prints at the start of a run how well they hold (max $\lvert\psi\rvert$ on
the axis and at $r = 1$, and the error of $\psi$ rebuilt from the degrees):

- $\psi$ must vanish on the axis (it must be constant there for a regular field; for a rotation that varies along the
  axis, $G$ has to vanish on the values $\Delta\Omega$ takes there);
- and at the outer wall, if the field is to stay inside the fluid (wherever the field does not match a potential field
  at the wall, it meets the insulating mantle with a surface current: [Poloidal fields](#poloidal-fields));
- the series is cut at `B0_ferraro_nmax` (odd, $\ge 1$), which sets the residual $\mathbf B_0\cdot\nabla\Delta\Omega$.

A toroidal part (`B0_tor`, any type) can be added: it is consistent with any axisymmetric rotation, and it must have the
field's equatorial symmetry (an even degree). Example: for a cylindrical rotation $\Delta\Omega \propto s^2$, $G(w) = w$
gives $\psi \propto s^2$, the uniform axial field.

### Rotation constant along the field lines: `diff_rot_type = 'ferraro'`

This set-up also avoids the winding, the other way round, for when the poloidal field is given (any type, `'general'`
included; e.g. a model of an observed or a dynamo field). kore keeps the field and replaces the rotation profile by one
that is constant along every field line, $\Delta\Omega = F(\psi)$, with $\psi$ the field's own flux function. Then
$\mathbf B_0\cdot\nabla\Delta\Omega = 0$, nothing is wound up, and the background is steady. The force of the rotation
on the flow, its induction term and the thermal wind all use this profile.

**Method.** The flux function is built from the radial profiles of the field (`background_field.pol_profiles`),

$$
\psi(r, x) = r\sum_n c\,h_n(1 - x^2)P_n'(x).
$$

$F$ is evaluated on a Gauss–Legendre grid in $x$ and projected on the Legendre polynomials,

$$
f_n(r) = \frac{2n+1}{2}\int_{-1}^{1}F\big(\psi(r, x)\big)\,P_n(x)\,dx, \qquad n = 0, 2, \dots, \texttt{diff\_rot\_nmax},
$$

and $\Delta\Omega$ = `diff_rot_ampl` $\cdot\sum_n f_n P_n$, as for the other rotation profiles, so nothing else changes
(`diff_rot._ferraro_shapes`). $F$ = `diff_rot_F` (default `'psi'`), a numpy expression of `psi` and `p`
(`diff_rot_param`).

**Rules**, checked at the start of a run:

- `magnetic = 1` and a poloidal `B0` are needed (a toroidal field alone has no $\psi$);
- `B0 = 'ferraro'` is refused (the field would be set by the rotation and the rotation by the field);
- kore's $\Delta\Omega$ has only even $P_n$, so $F(\psi)$ must be symmetric about the equator: any $F$ for a dipolar
  field ($\psi$ even in $x$), an even $F$ (e.g. of `psi**2`) for a quadrupolar one ($\psi$ odd), else the run stops;
- with a conducting inner core $\Delta\Omega(r_i, \theta)$ must be uniform, which in general means $\psi$ constant on
  the inner core's surface;
- `diff_rot_nmax` must be even and not negative.

A toroidal part of `B0` is left as it is. The series is cut at `diff_rot_nmax`: `ferraro_mismatch` then reports only the
truncation, at round-off for fields of low degree with a polynomial $F$. Example: the uniform axial field
$\psi = r^2\sin^2\theta/2$ with $F(\psi) = \psi$ gives $\Delta\Omega \propto s^2$, the cylindrical profile.

### The wound toroidal field: `B0_wound = 1`

When both the rotation and a poloidal field are given and they do not satisfy the Ferraro condition (for example in a
model of the solar tachocline), the shear winds the poloidal field into an azimuthal one, which grows until magnetic
diffusion balances the winding. `B0_wound = 1` accepts the winding: it computes this steady azimuthal field and adds it
to the background, on top of any `B0_tor`. The poloidal field and the rotation are kept as given, so the shear keeps
crossing the field lines, and the winding is balanced by diffusion instead of avoided.

**Method.** The steady balance is the $\phi$-component of the induction equation for axisymmetric fields, with the
background flow $\mathbf U_0$ above ($r\sin\theta = s$) and a uniform diffusivity $\eta$ = `B0_wound_eta` times
`MagnetD`,

$$
\eta\left(\nabla^2 - \frac{1}{s^2}\right)B_\phi = -G\,s\,\mathbf B_p\cdot\nabla\Delta\Omega ,
$$

where $\mathbf B_p$ is the poloidal `B0` (any type, `cnorm` included). Written as kore's toroidal field, the operator
acts degree by degree:

$$
B_\phi = \sum_n k_n\sin\theta\,P_n'(\cos\theta), \qquad
\left(\nabla^2 - \frac{1}{s^2}\right)B_\phi = \sum_n (D_n k_n)\,\sin\theta\,P_n', \qquad
D_n k = k'' + \frac{2k'}{r} - \frac{n(n+1)k}{r^2} .
$$

The source is projected on $\sin\theta\,P_n'$ by Gauss–Legendre quadrature in $\cos\theta$, using

$$
\int_{-1}^{1}(1 - x^2)P_n'P_m'\,dx = \delta_{nm}\,\frac{2n(n+1)}{2n+1},
$$

which leaves one radial equation per degree, solved by Chebyshev collocation (`background_field.wound_profiles`).
Boundary conditions:

- $k_n = 0$ at an insulating wall ($B_\phi$ vanishes outside);
- regular at $r = 0$ in a full sphere (parity $(-1)^n$ on $[-1, 1]$);
- with a conducting inner core, which must rotate rigidly: $k = A r^n$ inside, and $B_\phi/\mu$ and $E_\theta$
  continuous, which gives at $r_i$ ($\sigma$ = `sigma_i2o`)

    $$
    r k_n' + \left(1 - \frac{n+1}{\sigma}\right)k_n = 0 .
    $$

The degrees run up to (that of $\mathbf B_p$) + (that of $\Delta\Omega$) + 1, with the field's symmetry (even for a
dipolar $\mathbf B_p$, odd for a quadrupolar one). The $k_n$ are added to any `B0_tor`, and the couplings come from
`mag_algebra.py` as for `'general'`.

**Size.** The physics, not `cnorm`, sets the size of the wound field: about $G\,\Delta\Omega/\eta$ (times the length
scale squared) relative to $\mathbf B_p$. `cnorm` normalises the poloidal field alone, so the Lehnert or Elsasser number
keeps its meaning for the field the user gives, and the wound part enters with its physical ratio. `B0_wound_eta`
(default 1) is the diffusivity of this balance in units of `MagnetD`, while the perturbation still diffuses with
`MagnetD`: a turbulent value above 1 gives a weaker wound field.

**Check.** At the start of a run, instead of the Ferraro check, `background.magnetic_diffrot` prints the degrees and the
residual of the steady balance on a grid off the collocation nodes,

$$
\frac{\max\lvert\eta(\nabla^2 - 1/s^2)B_\phi + G s\,\mathbf B_p\cdot\nabla\Delta\Omega\rvert}{\max\lvert G s\,\mathbf B_p\cdot\nabla\Delta\Omega\rvert} .
$$

**Left out.** Two effects make the background only approximately steady: the Ohmic decay of $\mathbf B_p$ (the poloidal
field is held fixed, as it is without rotation), and the Lorentz force of $\mathbf B_0$ on the background flow
($\Delta\Omega$ is imposed). This force is dominated by the wound $B_\phi$, which grows as $G\,\Delta\Omega/\eta$. The
next section says when it can be ignored.

### When the ignored Lorentz force matters

kore imposes the rotation and leaves out the force of the background field on it. This section estimates that force and
says when it is negligible. It has two parts: an azimuthal torque, which brakes the rotation, and a meridional part,
which enters the thermal-wind balance. The conditions that follow are summarised on
[Magnetic field](magnetic.md#when-the-field-leaves-the-thermal-wind-unchanged).

The estimates are orders of magnitude, in kore's units (time $1/\Omega$, length the outer radius;
`timescale = 'rotation'`, so `Gaspard` = 1, `Hendrik` = $\mathrm{Le}^2$, `MagnetD` = Em). With `cnorm = 'mag_energy'`
the maximum of the poloidal field is of order 1. With $\rho_0$ the background density (1 for `'Boussinesq'`) and
$\eta_w$ = `B0_wound_eta`, the Lorentz force per unit mass and the wound field are

$$
\mathbf F_0 = \frac{\mathrm{Le}^2}{\rho_0}(\nabla\times\mathbf B_0)\times\mathbf B_0, \qquad
B_\phi \approx \frac{\Delta\Omega}{\mathrm{Em}\,\eta_w}\,B_p .
$$

1. *The azimuthal torque*, $\propto \mathbf B_p\cdot\nabla(s B_\phi)$: Ferraro's mechanism. It brakes $\Delta\Omega$
   along the field lines in a time of about $1/(\Lambda_p\Omega)$, with $\Lambda_p$ the Elsasser number of the poloidal
   field built on the background diffusivity. In a real system the wound field first has to build up, so the braking
   takes at least one Alfvén crossing time of the poloidal field. If viscous (or turbulent) stresses hold
   $\Delta\Omega$, the torque compares with them through the Hartmann number $Ha$ of the background values:

    $$
    \Lambda_p = \frac{\mathrm{Le}^2}{\rho_0\,\mathrm{Em}\,\eta_w}, \qquad
    Ha^2 = \frac{\mathrm{Le}^2}{\rho_0\,\mathrm{Ek}\,\mathrm{Em}\,\eta_w}
    \quad \left(\frac{\mathrm{Le}^2}{\mathrm{Ek}\,\mathrm{Em}} \text{ for } \rho_0 = \eta_w = 1\right).
    $$

2. *The meridional part*, the hoop stress and magnetic pressure of $B_\phi$, $\propto B_\phi^2$: it changes the
   thermal-wind and entropy balance of the background and drives magnetic buoyancy. Relative to the rotation term of
   that balance it is about $\mathrm{Le}_\phi^2/(2\rho_0\lvert\Delta\Omega\rvert)$, with the Lehnert number of the wound
   field

    $$
    \mathrm{Le}_\phi = \mathrm{Le}\,\frac{\max\lvert B_\phi\rvert}{\max\lvert\mathbf B_p\rvert} .
    $$

**The thermal-wind balance with the field.** The thermal wind sets the latitudinal entropy that balances the
differential rotation ([Differential rotation §2.4](differential_rotation_details.md#24-thermal-wind-entropy)). With
$\mathbf F_0$ added to that balance, the $\phi$-component of its curl reads

$$
\frac{\mathrm{Beyonce}\,g}{r}\,\frac{\partial S}{\partial\theta} = G^2 s\,\frac{\partial(1 + \Delta\Omega)^2}{\partial z} + \big[\nabla\times\mathbf F_0\big]_\phi ,
$$

and kore keeps only the first term on the right. For a toroidal field alone the second is

$$
\big[\nabla\times\mathbf F_0\big]_\phi = -\frac{\mathrm{Le}^2}{\rho_0\,s}\,\frac{\partial B_\phi^2}{\partial z} + \frac{\mathrm{Le}^2\rho_0'}{2\rho_0^2\,r\,s^2}\,\frac{\partial(s^2B_\phi^2)}{\partial\theta} ,
$$

the hoop stress and, with a varying density, the magnetic pressure. It vanishes for a field without current in the
fluid: a poloidal field with $D_n h_n = 0$ for every degree, that is $h_n = a\,r^n + b\,r^{-(n+1)}$, and no toroidal
part. With a uniform density it also vanishes for the toroidal `'Malkus'` field with `B0_tor_l = 1`, $B_\phi \propto s$,
whose force is then a gradient that pressure balances. For other fields, with the field and the rotation varying on the
same scale, the ratio of the two terms is about

$$
\epsilon = \frac{\mathrm{Le}^2\lvert\mathbf B_0\rvert^2}{2\rho_0\lvert\Delta\Omega\rvert} ,
$$

condition 1 of [Magnetic field](magnetic.md#when-the-field-leaves-the-thermal-wind-unchanged); item 2 above is the same
estimate for the wound field. A field varying on a shorter scale $\ell$ raises it by the ratio of the scales, or by its
square for the tension of curved poloidal lines.

**In the mode frequencies.** The thermal wind enters the frequency squared of a mode at order $2\Delta\Omega$
([Differential rotation §1.3](differential_rotation_details.md#13-thermal-wind-balance)), so the error $\epsilon$ moves
it by about $\mathrm{Le}^2\lvert\mathbf B_0\rvert^2/\rho_0$, and the frequency $\omega$ of a mode by about
$\mathrm{Le}^2\lvert\mathbf B_0\rvert^2/2\rho_0\omega$, for modes that feel the baroclinicity and weighted by their
share of the magnetised region. The field's direct force on a mode, which the linear problem includes, enters the
frequency squared as $\mathrm{Le}^2\lvert\mathbf B_0\rvert^2k_\parallel^2/\rho_0$, with $k_\parallel$ the mode's
wavenumber along the field: larger by $(k_\parallel r)^2$.

**Uniform rotation.** Where $\Delta\Omega$ is uniform there is no thermal wind, and $\mathbf F_0$ is balanced by an
entropy change $\delta S$ with $(\mathrm{Beyonce}\,g/r)\,\partial_\theta\delta S = [\nabla\times\mathbf F_0]_\phi$.
Relative to the background stratification it is about $(v_A/N\ell)^2$, with $v_A$ the Alfvén speed, $N$ the buoyancy
frequency and $\ell$ the scale of the field: of order $10^{-4}$ for a $10^5$ G field on a tenth of the solar radius just
below the tachocline ($N \approx 10^{-3}\,\mathrm{s^{-1}}$), and larger nearer the convection zone, where $N$ falls to
zero.

#### The Sun, Jupiter and Saturn

This applies the estimates above to the three bodies, with published values, at the depth where the field and the shear
meet: the base of the solar convection zone, and the depth where the winds of Jupiter and Saturn end (about 3000 and
9000 km; Kaspi et al. 2018, Galanti et al. 2019), which is where their conductivity becomes large enough to limit them
(Liu et al. 2008). Here Le is the local Alfvén speed over $\Omega R$, the quantity
$\mathrm{Le}\lvert\mathbf B_0\rvert/\sqrt{\rho_0}$ of the conditions, and $\Delta\Omega$ is in units of $\Omega$.

| | Sun, base of the convection zone | Jupiter, 0.96 R | Saturn, 0.86 R |
| --- | --- | --- | --- |
| rotation period | 25 d | 9.9 h | 10.6 h |
| density (g cm⁻³) | 0.2 | 0.2 | 0.3 |
| field (G) | 10⁴ toroidal (10⁵ in flux-tube models; D'Silva & Choudhuri 1993) | 10 | 0.5 |
| Le | 0.03 (0.3) | 5×10⁻⁶ | 3×10⁻⁷ |
| $\lvert\Delta\Omega\rvert$ | 0.1 | 0.01 | 0.03 |
| $\epsilon$ | 0.005 (0.5) | 10⁻⁹ | 10⁻¹² |
| poloidal field (G) | 10 | 10 | 0.5 |
| magnetic diffusivity (cm² s⁻¹) | 10¹¹ to 10¹² (turbulent) | 10¹¹ | 3×10¹⁰ |
| braking time, $1/\Lambda_p\Omega$ | a century or more | about a century | about 10⁴ years |

For the Sun the frequency error of condition 1 is about 0.2 nHz for a 10⁴ G field and a mode of frequency near $\Omega$
(more for slower modes), and about 20 nHz for 10⁵ G, before weighting by the mode's share of the layer. It reaches 1 nHz
near 2×10⁴ G. For Jupiter and Saturn this error is below $10^{-10}$ of the rotation rate. The field's direct effect on a
mode of frequency near $\Omega$ stays below $10^{-6}$ of it for wavenumbers along the field up to about $300/R$
(Jupiter) and $5000/R$ (Saturn), which covers the large-scale inertial modes.

The resulting conditions, this summary, and the checklist for solar inertial modes are on
[Magnetic field](magnetic.md#validity-and-caveats).

## Energy budgets

For each mode, spin_doctor checks that the energy equations close: the rate of change of each energy must equal the
power that feeds it, so a large residual flags an unresolved or wrong mode. With a field, spin_doctor adds the power of
the Lorentz force to the kinetic budget (residσ, residω), checks the budget of the induction equation in the fluid
(resid𝐛), and checks the full budget including the field outside the fluid (resid𝐄). It prints resid𝐛, resid𝐄 and
Mag/Kin and writes `magnetic.dat`:

| column | name | meaning |
| --- | --- | --- |
| 1 | ME | magnetic energy in the fluid (without Hendrik; Mag/Kin is Hendrik ME/KE) |
| 2 | Mdfs | magnetic diffusion |
| 3 | Indu | induction work |
| 4 | residb | resid𝐛, the relative residual of the induction budget |
| 5, 6 | Wlor, Wlor_im | power of the Lorentz force, real and imaginary parts (in residσ, residω) |
| 7 | MEic | magnetic energy in the inner core (its field, or the potential field below an insulating one) |
| 8 | MEext | energy of the potential field outside the outer wall (beyond `r_surface` with a resolved mantle) |
| 9 | Ohm_f | Ohmic loss in the fluid |
| 10 | Ohm_ic | Ohmic loss in a conducting inner core |
| 11 | residE | resid𝐄, the full energy budget (nan for a viscous fluid along a wall that is not no-slip with `centrifugal = 1`, with thin layers and `mu` ≠ 1, and with rotating solids unless `vtrq` and `mtrq` are 1) |
| 12 | MEc | with `centrifugal = 1`, a thin layer or a resolved mantle: the energy that the deformation adds (0 without it) |
| 13, 14 | E_w, Ohm_w | with a thin layer or a resolved mantle: the magnetic energy stored in the thin layers or in the mantle, and their Ohmic loss |
| 15, 16 | eps_twa_o, eps_twa_i | with a thin layer or a resolved mantle: $\varepsilon_\text{TWA}$ of the outer and of the inner thin layer (0 where there is none) |

with the integrals over the fluid unless stated, and with the Lorentz force per volume without `Hendrik`
$\mathbf G = (\nabla\times\mathbf b)\times\mathbf B_0 + (\nabla\times\mathbf B_0)\times\mathbf b$,

$$
\mathrm{ME} = \int\tfrac12\lvert\mathbf b\rvert^2 dV, \qquad
\mathrm{Mdfs} = \mathrm{MagnetD}\int\mathbf b\cdot\nabla^2\mathbf b\,dV, \qquad
\mathrm{Indu} = \int\mathbf b\cdot\nabla\times(\mathbf u\times\mathbf B_0)\,dV,
$$

$$
\mathrm{resid}\mathbf b = \frac{\lvert 2\sigma\,\mathrm{ME} - \mathrm{Indu} - \mathrm{Mdfs}\rvert}{\max(\lvert 2\sigma\,\mathrm{ME}\rvert, \lvert\mathrm{Indu}\rvert, \lvert\mathrm{Mdfs}\rvert)}, \qquad
\mathrm{Wlor} + i\,\mathrm{Wlor\_im} = \mathrm{Hendrik}\cdot 2\int\mathbf u^*\cdot\mathbf G\,dV,
$$

$$
\mathrm{MEic} = \tfrac12\int_{\mathrm{ic}}\frac{\lvert\mathbf b\rvert^2}{\mu_{i2o}}\,dV, \qquad
\mathrm{MEext} = \tfrac12\int_{r > 1}\lvert\mathbf b\rvert^2dV, \qquad
\mathrm{Ohm\_f} = \mathrm{MagnetD}\int\lvert\nabla\times\mathbf b\rvert^2dV, \qquad
\mathrm{Ohm\_ic} = \mathrm{MagnetD}\,\eta_i\int_{\mathrm{ic}}\frac{\lvert\nabla\times\mathbf b\rvert^2}{\mu_{i2o}}\,dV .
$$

**The full budget.** resid𝐄 (`utils4pp.em_budget`) closes

$$
2\sigma[\mathrm{KE} + \mathrm{Hendrik}(\mathrm{ME} + \mathrm{MEic} + \mathrm{MEext})] = \mathrm{Dkin} + \mathrm{Wthm} + \mathrm{Wlor} + \mathrm{Hendrik}(\mathrm{Indu} - \mathrm{Ohm\_f} - \mathrm{Ohm\_ic}) :
$$

unlike resid𝐛 it sees the wall and ICB conditions. A body force that the right-hand side puts in the fluid, the Poincaré
force of the libration in the frame of the mantle (forcing 8), adds its power Wf (`flow.dat` column 42) on the right. Without the flow (`hydro = 0`) it is
$2\sigma(\mathrm{ME} + \mathrm{MEic} + \mathrm{MEext}) = \mathrm{Indu} - \mathrm{Ohm\_f} - \mathrm{Ohm\_ic}$, without
Hendrik and for any walls (Indu is then the induction by a differential rotation, or 0).

**Slipping walls.** Where the fluid slips along an impenetrable wall (`bco` or `bci` 0, or `ViscosD = 0`),
$\mathbf u\times\mathbf B_0$ carries magnetic energy through the wall, into the field beyond it (a conducting inner core
or mantle, a thin layer, or the potential field of an insulator).
resid𝐄 then uses the volume work of the induction alone, Indu minus its wall terms (`utils4pp.INDU_WALLS`, from the
$(r, \theta)$ grid), as with a differential rotation (below).

**Thin layers** (`utils4pp.em_budget`, `twa_budget`). The layers store
$E_w = \tfrac{cr}{2}\oint\lvert\mathbf b_t\rvert^2dS + \tfrac{h}{2p}\oint\lvert b_r\rvert^2dS$ (the poloidal tangential
field: the toroidal condition has no $c$ term; the second term only for finite $p$), and lose
$\mathrm{Ohm\_w} = \mathrm{MagnetD}\oint\lvert\mathbf K\rvert^2 dS/(c_1 r)$, with
$\mathbf K = \hat{\mathbf n}\times(\mathbf b_\text{beyond} - \mathbf b)$ the layer's current ($\hat{\mathbf n}$ out of the
fluid, $\mathbf b_\text{beyond}$ the potential field beyond the layer): $c_1\lvert r\nabla^2_l f\rvert^2$ from the poloidal
condition (with the radial induction of a slip, and that of a radial boundary flow, `utils4pp.radial_lapF`: all of it
where the solid is at rest, `'rigid'`, half its toroidal part where the layer deforms across a toroidal $\mathbf B_0$,
`background.thin_drop`) and $\lvert g\rvert^2/c_1$ from the toroidal one. The field beyond the layer, which MEext and
MEic count beyond $r \pm h$, follows from its conditions. The full budget gains $\mathrm{Hendrik}\,E_w$ on the left and
$-\mathrm{Hendrik}\,\mathrm{Ohm\_w}$ on the right. It closes exactly for $c = 0$, and to $O(c\,c_1)$ otherwise, the order of the approximation itself (resid𝐄 is
then about $\varepsilon_\text{TWA}$ or below); not with $\mu \neq 1$.

**Resolved mantle** (`utils4pp.mantle_budget`). The layer stores
$E_w = \tfrac12\int\lvert\mathbf b\rvert^2dV$ and loses $\mathrm{Ohm\_w} = \mathrm{MagnetD}\int\lvert\nabla\times\mathbf b\rvert^2/\sigma\,dV$
over $1 \le r \le r_s$ (Gauss-Legendre in $r$), and MEext is the potential field beyond $r_s$. The full budget gains them
as for thin layers, and closes exactly. A fluid slipping along the layer carries energy into it through
$\mathbf u\times\mathbf B_0$, as into a thin layer (`INDU_WALLS`, above).

**Moving walls** (forced runs, and free walls `bco` or `bci` 2). Where a wall moves radially the pressure works on the
fluid there. That work, Wp, enters residσ and the right-hand side of resid𝐄 with the magnetic part of the pressure, the
consoidal part of the Lorentz force at the wall (`utils4pp.wall_powers`, `lorentz_walls`). In a forced run the motion of
the boundary also works on the field, in a way that follows how the solid behind it moves
([Forced runs](#forced-runs)):

- **Where the solid moves with the fluid at the wall**: a no-slip wall that moves along itself (the libration), and a
  thin conducting layer under `'deforming'`, which moves with the fluid across the wall (and along it where the wall is
  no-slip). The wall's velocity $\mathbf V$ ($\mathbf u_t$ at a no-slip wall, plus
  $u_r\hat{\mathbf r}$ at a thin layer under `'deforming'`) carries the flux of $\mathbf V\times\mathbf B_0$ out of the fluid, which
  resid𝐄 keeps in the induction work (`INDU_ROT_SOLID`). At a thin layer it is the work against the Lorentz force on its
  current $\mathbf K$ (above), $2\,\mathrm{Re}\oint(\mathbf V\times\mathbf B_0)\cdot\mathbf K^*dS$, with $\mathbf B_0$ its
  mean across the layer (half the toroidal part, `background.thin_drop`). At a conducting inner core or the resolved
  mantle it is the flux of the jump of the tangential electric field, $2\,\mathrm{Re}\oint\big((\mathbf V\times\mathbf B_0)\times\mathbf b_T^*\big)\cdot
  \mathbf n\,dS$, with $\mathbf b_T$ the toroidal part of $\mathbf b$ (the only part with which a consoidal field has a
  flux).
- **A thick solid that deforms** (`'deforming'`, `background.deforms`). Its surface moves with the fluid across the
  wall, so the radial flow's part of the tangential electric field is continuous there and its fluxes cancel; along a
  slipping wall the flux of the slip stays in `INDU_WALLS`, as at rest. The work of its motion inside,
  $2\,\mathrm{Re}\int(\mathbf v_s\times\mathbf B_0)\cdot(\nabla\times\mathbf b)^*dV$ over the solid, joins the induction
  work (`utils4pp.deform_work`, on a Gauss grid in $r$ and $\cos\theta$, apart from the rows).
- **`'rigid'`**. The flow passes through the wall into the solid at rest, whose rows take its $\mathbf u\times\mathbf B_0$:
  its flux leaves the fluid for the solid's field, as at a slipping wall, and stays out of the induction work
  (`INDU_WALLS`).

With a thin layer whose $c$ is not zero this holds to the order of the approximation, as without the forcing.

**Rotating solids** (`rotdyn = 1`). The solids store their rotational and gravitational energy, and exchange energy with
the fluid through the torques: the full budget gains that energy on the left, and the power of the viscous torques and
the inner core's relaxation loss on the right, while the $\mathbf u\times\mathbf B_0$ flux through the moving no-slip
walls is minus the power of the magnetic torques and leaves the induction work (`INDU_WALLS`). Only with both torques on
the solids (`vtrq` and `mtrq` 1); [Rotating solids equations](rotating-solids-equations.md#energy).

**How the terms are computed.** spin_doctor's Lorentz force (`utils4pp.lorentz4pp`) acts per volume, and its power
$\int\mathbf u^*\cdot\mathbf G\,dV$ has no $\rho_0$. Its induction term (`induction4pp`) takes the components of
$\mathbf u$ with their $\lambda_1$ part, and holds for the non-solenoidal $\mathbf u$ of a variable density (resid𝐛
closes to $10^{-10}$ with 'Wu 2005', [Checks](#checks)). These spectral formulas are used
for a poloidal field of degree 1 alone, with the closed-form couplings and no differential rotation. Otherwise
(`utils4pp.use_grid`: a toroidal part, a poloidal field of degree other than 1, a differential rotation, the
recurrence-algebra couplings, or `KORE_MAG_GRID=1` in the environment) both come instead from
`utils4pp.magnetic_powers_grid`, computed on an $(r,\theta)$ grid from the vector fields $\mathbf u$, $\mathbf b$,
$\nabla\times\mathbf b$, $\mathbf B_0$, $\nabla\times\mathbf B_0$ (with the wall terms of the induction work); for a
poloidal field it agrees with the spectral formulas to the 10 printed digits.

**With a differential rotation.** spin_doctor's induction work includes $\mathbf U_0\times\mathbf b$ (on the
$(r, \theta)$ grid). Because $\mathbf U_0\times\mathbf b$ does not vanish at a wall where $\Delta\Omega \ne 0$, it
carries magnetic energy through the wall into the field outside: resid𝐛 (the fluid) uses the induction work with its
wall terms, resid𝐄 (the full budget, which counts the energy outside) the volume work alone (`utils4pp.INDU_WALLS`),
plus the power of the rotation's force on the flow. Where a conducting solid turns with the fluid (a conducting inner
core, a thin layer, a resolved mantle), the flux of $\mathbf U_0\times\mathbf b$ through its surface is the induction work of the solid's own
rotation, $\mathrm{Re}\int(\mathbf V\times\mathbf b)\cdot(\nabla\times\mathbf b)^*dV$ over it ($\mathbf V$ rigid, the potential
field beyond carries none), and resid𝐄 keeps it (`utils4pp.INDU_ROT_SOLID`). Without it resid𝐄 was about $10^{-4}$ for a
conducting inner core turning with $\Delta\Omega(r_i) \neq 0$; with it the budget is the same in every frame.

## Checks

The magnetic terms are checked in several independent ways, by the suites in `tests/` (each has a README that lists
its scripts):

- **Symbolic derivation.** Every coupling of the Lorentz force and the induction is compared, term by term, with a
  projection computed from the definitions with sympy (`tests/magnetic`, [Derivation tools](derivations.md)).
- **Exact solutions.** Free-decay rates of the field with insulating walls, a conducting inner core, a thin conducting
  layer and a resolved mantle (`tests/decay`, `tests/twa`, `tests/mantle`); a uniform offset of the rotation rate.
- **Published results.** The inertial modes of Wu (2005), the modes of the Malkus (1967) field, and the torsional modes
  of Luo & Jackson (2022).
- **Limits.** A thin resolved mantle against the thin-wall approximation; smoothed current sheets against the sheet
  conditions (`tests/sheet`); the Bessel inner core against the Chebyshev one (`tests/forced_field`); a weakly
  stratified `'Wu 2005'` against `'Boussinesq'`.
- **Budgets.** resid𝐛 and resid𝐄 close to round-off for resolved modes, with every combination of walls, inner core,
  density profile, differential rotation and forcing (`tests/magnetic`, `tests/forced_field`, `tests/inner_core`).

## References

- D'Silva, S. & Choudhuri, A. R. 1993, A&A 272, 621 (flux-tube fields at the base of the solar convection zone).
- Ferraro, V. C. A. 1937 (the law of isorotation).
- Galanti, E., Kaspi, Y., et al. 2019, GRL 46, 616 (the depth of Saturn's winds).
- Gerick, F. et al. 2021 ('G21 dipole').
- Gizon, L. et al. 2021; Löptien, B. et al. 2018 (solar inertial modes).
- Gough, D. O. & McIntyre, M. E. 1998; MacGregor, K. B. & Charbonneau, P. 1999 (tachocline confinement).
- Gubbins, D. & Roberts, P. H. 1987 (free decay modes of a sphere).
- Guervilly, C., Wood, T. S. & Brummell, N. H. 2013, arXiv:1307.3873 (the thin-wall conditions, appendix A).
- Kaspi, Y., Galanti, E., et al. 2018, Nature 555, 223 (the depth of Jupiter's winds).
- Liu, J., Goldreich, P. M. & Stevenson, D. J. 2008, Icarus 196, 653 (Ohmic limits on deep zonal winds).
- Luo, J. & Jackson, A. 2022 ('Luo_S1', 'Luo_S2', `'mag_energy'`, torsional modes).
- Malkus, W. V. R. 1967 (the field $s\,\hat{\boldsymbol\phi}$).
- Rekier, J., Triana, S. A., Trinh, A. & Buffett, B. 2023, arXiv:2303.00718 (the electric field at a slipping wall,
  condition C.I).
- Roberts, P. H., Glatzmaier, G. A. & Clune, T. L. 2010, GAFD (the thin-wall approximation).
- Schmitt, D. 2012 (`'Schmitt2012'`).
- Wu, Y. 2005 (inertial modes of a full sphere).
- Zhang, K. & Fearn, D. R. 1994, 1995 (free decay modes of a shell).
