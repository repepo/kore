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

**At the outer wall.** A field that does not match a potential field at the outer wall meets the insulating mantle with
a surface current, which the linear problem allows. The FDM, `'G21 dipole'`, `'Luo_S1'` and `'Luo_S2'` match one there
($rh' + (l+1)h = 0$ at $r = 1$), and the FDM matches one at both walls, so it suits insulating walls. `'axial'` is the
uniform field, potential everywhere. `'confined'` is zero at the wall. `'general'` and `'ferraro'` need not match.

**Full-sphere fields with an inner core.** `'G21 dipole'`, `'Luo_S1'` and `'Luo_S2'` are defined for a full sphere, and
`B0_ic` says how to continue them into an inner core. `'potential'` (default) adds $\beta_i r^{-(l+1)}$,
$\beta_i = r_i^{l+1}(r_i h' - l h)/(2l+1)$ at $r_i$, so that $rh' - lh = 0$ at the inner boundary: the field of the
currents in the fluid, continued as a potential field into the core (`conductive_IC`'s `B0_beta_ic`).
`'frozen currents'` (the name in `conductive_IC`; `'full sphere'` before 2026-11-06) keeps $h$ as it is, with the
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
Chebyshev interpolants (degree max(64, 2N)); in a full sphere they are evaluated at $\lvert r\rvert$ with the parity
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
| `'user'` | `B0_tor_profile`, a numpy expression of $r$ and $p$ = `B0_tor_param` | its derivatives from a Chebyshev interpolant (degree max(64, 2N)); written to `params.dat` after `#` |
| `'general'` | $\sum_n k_n Y_n^0$, `B0_tor_profiles` = {n: 'numpy expression of r, p'}, with $p$ = `B0_tor_param` | any degrees of one parity; derivatives as for the poloidal `'general'` |

**At the walls.** A toroidal field that does not vanish at a wall (`'Malkus'`) carries a surface current there. The
linear problem is unaffected: the walls' conditions on $\mathbf b$ do not involve $\mathbf B_0$, and
$\mathbf u\times\mathbf B_0 = 0$ on a no-slip wall. Inside a conducting inner core $\mathbf B_0$ does not enter either:
there is no flow there, and $\mathbf u\times\mathbf B_0 = 0$ at the no-slip inner boundary, so the matching conditions
do not involve it. `B0 = None` gives a toroidal field alone.

### Normalisation

The profiles have arbitrary scales, so the amplitude is set apart: `cnorm` multiplies the poloidal part and `cnorm_tor`
the toroidal part. Each is a number, or a name that kore turns into a number as `conductive_IC` does
(`background_field.B0_norm`, Gauss–Legendre quadrature with 240 nodes):

- `'rms_cmb'` (`cnorm` only): the rms of the radial field over the outer wall is 1,
  $c_\mathrm{norm} = \sqrt{2l+1}/(L\,h(1))$, $L = l(l+1)$. Single-degree types only: `'general'` and `'ferraro'` stop
  with a message, and so does `'confined'`, which is zero at the wall.
- `'mag_energy'`: $\int B_0^2\,dV = 1$ over the fluid (Luo & Jackson 2022).
- `'Schmitt2012'`: $\int B_0^2\,dV = 4$.

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
- rows `g`, columns P, diagonal in $l$ only (in the scale of `conductive_IC`'s rows; zero for m = 0), from
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
  momentum rows hold $r^p$ times the force $\rho_0 \mathbf F_L$. `submatrices.py` lists each label next to its
  `conductive_IC` name. `tol_B0` (default 1e-8) is the relative cut of the Chebyshev series of $r^p h^{(k)}$: it sets
  the band width of the magnetic blocks.
- **Operators** (`operators.lorentz`, `induction`, `magnetic_diffusion`, `b`). They are the `conductive_IC` operators
  divided by $L^3$ of the row, as every operator of this branch, with Hendrik and MagnetD for
  $(\Omega\tau\,\mathrm{Le})^2$ and $\Omega\tau\,\mathrm{Em}$. The sign convention is the branch's: A holds the
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
  rows. `conductive_IC` keeps $C^{(2)}$ and $C^{(1)}$ and so converts these terms with the wrong matrices; its torsional
  test is barely affected ([Checks](#full-sphere-inviscid-runs-and-degree-2-poloidal-fields)).
- **Solver.** `solve.py` writes `real_magnetic.field`, `imag_magnetic.field` and normalises the eigenvectors (over the
  shell, as `conductive_IC`) to

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
($c_\mathrm{tor}k_n$). [Derivation tools](derivations.md), part A.6, gives the method.

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

**Conducting inner core.** `innercore = 'conducting, Chebys'` (eigenvalue problems, no-slip inner boundary `bci = 1`)
adds the field inside the inner core to the unknowns, as `conductive_IC` does:

- **Unknowns.** Sections `fic`, `gic` after `f`, `g` (`ut.offset`), with the degrees of `f`, `g` (up to `lmax`;
  `conductive_IC`'s `lmax_cic` is not ported, as its ICB matching couples the degrees one by one). Each block is the
  Chebyshev series on $[-r_i, r_i]$ of one parity (that of the field at the centre), `N_cic`/2 coefficients per degree
  (`ut.Nic`, default `N_cic` = 64; `conductive_IC` found broken eigenvectors with 48 at `sigma_i2o` = $10^{-2}$);
  `ut.nic` per section.
- **Equation.** The field diffuses, with $\mu_{i2o}$, $\sigma_{i2o}$ (`mu_i2o`, `sigma_i2o`) the permeability and
  conductivity relative to the fluid:

    $$
    \frac{\partial\mathbf b}{\partial t} = \mathrm{MagnetD}\,\eta_i\nabla^2\mathbf b, \qquad \eta_i = \frac{1}{\mu_{i2o}\sigma_{i2o}},
    $$

    with rows $r^2\hat{\mathbf r}\cdot$ and $r^2\hat{\mathbf r}\cdot\nabla\times$ divided by $L^3$ as in the fluid, in the
    $C^{(2)}$ basis. Its matrices are built apart from the labels (`submatrices.py`: `r0_D0f_ic`, `r2_D0f_ic`,
    `r1_D1f_ic`, `r2_D2f_ic` and the same for g), with `ut.Dlam` and `bc_variables.Tk` on the inner core's domain (their
    optional domain arguments).
- **Matching at the ICB** (`assemble.bc_b_cic`), one row in each section, the inner one of `f`, `g` and the only one of
  `fic`, `gic`: $f_o = f_i$ ($[b_r]$), $(g + rg')_o = \eta_i (g + rg')_i$ (tangential $\mathbf E$),
  $\mu_{i2o}(f + rf')_o = (f + rf')_i$ and $\mu_{i2o}\,g_o = g_i$ (tangential $\mathbf H$). The continuity of
  $\eta\nabla^2_l f$ is not imposed: with $b_r$ continuous it follows from the radial induction equation on both sides,
  since $\mathbf u = 0$ at the no-slip ICB.
- **Output.** `real_magnetic_ic.field`, `imag_magnetic_ic.field`. The eigenvectors are normalised with the fluid's
  energies only (the inner-core field is scaled along), as in `conductive_IC`.

### Thin conducting walls

A thin conducting layer between the fluid and an insulator carries currents that couple the field across it. The
thin-wall approximation (Roberts, Glatzmaier & Clune 2010; Guervilly, Wood & Brummell 2013, appendix A) replaces the layer
by conditions on the field at the wall, so the layer adds no unknowns. `mantle = 'TWA'` puts one at the outer wall,
`innercore = 'TWA'` one at the inner wall; the conditions are those of `conductive_IC` (`bc_b_thinlayer`, with its terms of
first order in the thickness), in `assemble.bc_b_twa`.

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

- **The terms in $p$** are of first order in $h$: the curvature of the wall, the potential field matched at $r \pm h$,
  and the $l(l+1)/r^2$ of $\nabla^2_l$ in the wall. They vanish for $p = \infty$ (`np.inf`), the limit of Guervilly et
  al. ($h \to 0$ with $h\mu_w$ fixed). For $p = 1$, a wall with the permeability of vacuum, and $\mu = 1$ they cancel
  the $c$ term for a steady field, which then crosses the wall as it would an insulator.
- **$\mu$** enters only these conditions, so $\mu \neq 1$ needs a thin wall or a conducting inner core at every wall.
- **$\mathcal D_l F$** stands for the wall's $\partial_t F/\eta$, taken from the fluid next to it (uniform diffusivity),
  and $G'$ for the wall's tangential electric field. Both assume that the fluid moves with the wall.
- **Validity.** The approximation keeps the linear profile of the field across the wall. Inside it $b'' = k^2 b$ with
  $k^2 = \lambda/\eta_w$, and the first term left out, the wall's own induction, is a relative error

    $$
    \varepsilon_\text{TWA} = \frac{\lvert kh\rvert^2}{3} = \frac23\Big(\frac{h}{\delta_w}\Big)^2, \qquad \Big(\frac{h}{\delta_w}\Big)^2 = \frac{c\,c_1\,r^2\,\lvert\lambda\rvert}{2\,\mathrm{MagnetD}}
    $$

    of the wall's effect ($c\,c_1 = h^2\eta_f/(r^2\eta_w)$, any $p$), with $\lambda$ the eigenvalue in the frame of the
    wall (`utils4pp.twa_error`; spin_doctor's εTWA). The conditions $h \ll r$ and $lh \ll r$ are not checked.

**A fluid slipping along a thin wall** (`bco = 0`, `bci = 0` or `ViscosD = 0`; `utils.twa_slips`). The conditions above
take the tangential electric field of the fluid, $\mathbf E = \mathrm{MagnetD}\,\nabla\times\mathbf b - \mathbf u\times\mathbf B_0$,
in the frame of the wall. kore continues it into the wall (condition C.I of Rekier, Triana, Trinh & Buffett 2023): a
modelling choice, for a slip that stands for a viscous layer that is not resolved, valid while that layer is thin
against the magnetic skin depth. With $u_r = 0$ at the wall, $(\mathbf u\times\mathbf B_0)_t = B_{0r}\,(\mathbf u_t\times\hat{\mathbf r})$,
which enters through two projections on the degree $l$: $S_l$, the $r\nabla Y_l$ part of $B_{0r}(\mathbf u_t\times\hat{\mathbf r})$,
and $X_l = [\hat{\mathbf r}\cdot\nabla\times(\mathbf u\times\mathbf B_0)]_l = -[\nabla\cdot(B_{0r}\mathbf u_t)]_l/r$, the radial
induction of the slip. In the conditions

$$
G' \to G' - \frac{r\,S_l}{\mathrm{MagnetD}}, \qquad \mathcal D_l F \to \mathcal D_l F + \frac{r^2 X_l}{l(l+1)\,\mathrm{MagnetD}},
$$

the second because the wall's $\partial_t F$ now includes the radial induction of the slip. `assemble.bc_b_slip` builds
both for any poloidal $\mathbf B_0$ with the recurrence algebra (`angular.Algebra`): $B_{0r} = \sum_n n(n+1)h_n P_n/r$
over the degrees of the background, times $\mathbf u_t = \sum_k s_k\,r\nabla Y_k + t_k\,r\nabla Y_k\times\hat{\mathbf r}$ with
$s_k = P_k' + P_k/r + (\ln\rho_0)'P_k$ and $t_k = T_k$ at the wall; a toroidal background has no $B_{0r}$. With the
rotating solids the outer wall's frame turns with the mantle, and $t_1 = T_1 - r\,\omega_M$ there, the slip relative
to it ([Rotating solids equations](rotating-solids-equations.md#equations)).
`conductive_IC` has these terms for a single degree.

**The background field at a thin outer wall** (`B0_cmb`). A steady $\mathbf B_0$ drives no current in the wall, but
it must meet the steady condition, $(1/\mu + lc)(rh)' + (l + e)h = 0$ at $r = 1$ with $e = c\mu l((l+1)/p - 1)/p$
($e = 0$ for $p = \infty$). With `B0_cmb = 'wall'` (default) the fields of internal currents (`'G21 dipole'`, `'Luo_S1'`,
`'Luo_S2'`, `'FDM'`) gain $\alpha r^l$ in $h$ (`background_field.B0_alpha_cmb`), with $\alpha$ set by that condition. The
added field is a potential field: it carries no current and keeps the matching at the inner wall. For $p = 1$ and
$\mu = 1$ the condition is the insulating one, which these fields already meet. `'plain'` leaves $h$ unchanged; the other
fields are used as given. At a thin inner wall the background is not adjusted (for $p = 1$ and $\mu = 1$ a potential
field below meets its condition).

### Resolved conducting mantle

A layer that is not thin against its magnetic skin depth, or whose conductivity varies across it, cannot be reduced to
conditions at the wall. `mantle = 'resolved'` (as `conductive_IC`, after PR #17 by A. Nuyt) adds the field of a solid
conducting layer $1 \le r \le r_s$ (`r_surface`) to the unknowns, with an insulator beyond. Its conductivity relative to
the fluid's, $\sigma(r)$ (`sigma_mantle`, a number or an expression of $r$ and `sigma_mantle_param`), may vary with
radius.

- **Unknowns.** Sections `fm`, `gm`, last (after `fic`, `gic`; `ut.offset`), with the degrees of `f`, `g`. Each block is
  the Chebyshev series on $[1, r_s]$ (both parities), `N_mantle` coefficients per degree (`ut.nman` per section).
- **Equation.** The layer is at rest and its field diffuses, with the diffusivity $\mathrm{MagnetD}/\sigma$:

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
  electric field includes $-\mathbf u\times\mathbf B_0$, continued into the layer as for a thin wall (condition C.I):
  $\sigma(1)\,[(rg)' - r\,S_l/\mathrm{MagnetD}] = (rg_m)'$, with the $S_l$ of a thin wall (`assemble.bc_b_slip`, with
  $\sigma(1)$ for $c_1$). The radial induction of the slip is in the fluid's own equation, and the poloidal matching needs
  nothing more.
- **Output.** `real_magnetic_mantle.field`, `imag_magnetic_mantle.field` (`fm` then `gm`), scaled with the fluid's
  field.

As $r_s \to 1$ with $\sigma\,(r_s - 1)$ fixed the layer tends to a thin wall with $c = r_s - 1$ and $c_1 = \sigma(r_s - 1)$, at
first order in the thickness ([Checks](#resolved-conducting-mantle-2026-11-06)).

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

**Thin conducting walls** turn with the fluid next to them in the same way, so $\Delta\Omega$ must be uniform in
latitude at each thin wall (`diff_rot.rigid_at`), or the run stops. In the frame of the wall its conditions are those
above, and $\mathbf U_0\times\mathbf b$ cancels there with the wall's own motion. The **resolved mantle** turns with
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
  at the wall, it meets the insulating mantle with a surface current, which the linear problem allows);
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
| 1 | ME | magnetic energy in the fluid (without Hendrik, as `conductive_IC`; Mag/Kin is Hendrik ME/KE) |
| 2 | Mdfs | magnetic diffusion |
| 3 | Indu | induction work |
| 4 | residb | resid𝐛, the relative residual of the induction budget |
| 5, 6 | Wlor, Wlor_im | power of the Lorentz force, real and imaginary parts (in residσ, residω) |
| 7 | MEic | magnetic energy in the inner core (its field, or the potential field below an insulating one) |
| 8 | MEext | energy of the potential field outside the outer wall (beyond `r_surface` with a resolved mantle) |
| 9 | Ohm_f | Ohmic loss in the fluid |
| 10 | Ohm_ic | Ohmic loss in a conducting inner core |
| 11 | residE | resid𝐄, the full energy budget (nan with a flow through a wall, `bco = 2` or `bci = 2`, with stress-free walls and `centrifugal = 1`, with thin walls and `mu` ≠ 1, and with rotating solids unless `vtrq` and `mtrq` are 1) |
| 12 | MEc | with `centrifugal = 1`, a thin wall or a resolved mantle: the energy that the deformation adds (0 without it) |
| 13, 14 | E_w, Ohm_w | with a thin wall or a resolved mantle: the magnetic energy stored in the thin walls or in the layer, and their Ohmic loss |
| 15, 16 | eps_twa_o, eps_twa_i | with a thin wall or a resolved mantle: $\varepsilon_\text{TWA}$ of the outer and of the inner thin wall (0 where there is none) |

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

**The full budget.** resid𝐄 (`conductive_IC`'s, `utils4pp.em_budget`) closes

$$
2\sigma[\mathrm{KE} + \mathrm{Hendrik}(\mathrm{ME} + \mathrm{MEic} + \mathrm{MEext})] = \mathrm{Dkin} + \mathrm{Wthm} + \mathrm{Wlor} + \mathrm{Hendrik}(\mathrm{Indu} - \mathrm{Ohm\_f} - \mathrm{Ohm\_ic}) :
$$

unlike resid𝐛 it sees the wall and ICB conditions. The $1/\mu_{i2o}$ of the inner core's energy and Ohmic loss is not in
`conductive_IC`'s `em_budget` (there resid𝐄 is $10^{-2}$ at `mu_i2o` = 2). Without the flow (`hydro = 0`) it is
$2\sigma(\mathrm{ME} + \mathrm{MEic} + \mathrm{MEext}) = \mathrm{Indu} - \mathrm{Ohm\_f} - \mathrm{Ohm\_ic}$, without
Hendrik and for any walls (Indu is then the induction by a differential rotation, or 0).

**Slipping walls.** Where the fluid slips along an impenetrable wall (`bco` or `bci` 0, or `ViscosD = 0`),
$\mathbf u\times\mathbf B_0$ carries magnetic energy through the wall, into the field beyond it (and into a thin wall).
resid𝐄 then uses the volume work of the induction alone, Indu minus its wall terms (`utils4pp.INDU_WALLS`, from the
$(r, \theta)$ grid), as with a differential rotation (below).

**Thin walls** (`utils4pp.em_budget`, `twa_budget`, as `conductive_IC`). The walls store
$E_w = \tfrac{cr}{2}\oint\lvert\mathbf b_t\rvert^2dS + \tfrac{h}{2p}\oint\lvert b_r\rvert^2dS$ (the poloidal tangential
field: the toroidal condition has no $c$ term; the second term only for finite $p$), and lose
$\mathrm{Ohm\_w} = \mathrm{MagnetD}\oint\lvert\mathbf K\rvert^2 dS/(c_1 r)$, with $\mathbf K$ the jump of $\mathbf b_t$ across
the wall: $c_1\lvert r\nabla^2_l f\rvert^2$ from the poloidal condition (with the radial induction of a slip) and
$\lvert g\rvert^2/c_1$ from the toroidal one. The field beyond the wall, which MEext and MEic count beyond $r \pm h$, follows
from the wall conditions. The full budget gains $\mathrm{Hendrik}\,E_w$ on the left and $-\mathrm{Hendrik}\,\mathrm{Ohm\_w}$ on
the right. It closes exactly for $c = 0$, and to $O(c\,c_1)$ otherwise, the order of the approximation itself (resid𝐄 is
then about $\varepsilon_\text{TWA}$ or below); not with $\mu \neq 1$.

**Resolved mantle** (`utils4pp.mantle_budget`, as `conductive_IC`). The layer stores
$E_w = \tfrac12\int\lvert\mathbf b\rvert^2dV$ and loses $\mathrm{Ohm\_w} = \mathrm{MagnetD}\int\lvert\nabla\times\mathbf b\rvert^2/\sigma\,dV$
over $1 \le r \le r_s$ (Gauss-Legendre in $r$), and MEext is the potential field beyond $r_s$. The full budget gains them
as for thin walls, and closes exactly. A fluid slipping along the layer carries energy into it through
$\mathbf u\times\mathbf B_0$, as into a thin wall (`INDU_WALLS`, above).

**Rotating solids** (`rotdyn = 1`). The solids store their rotational and gravitational energy, and exchange energy with
the fluid through the torques: the full budget gains that energy on the left, and the power of the viscous torques and
the inner core's relaxation loss on the right, while the $\mathbf u\times\mathbf B_0$ flux through the moving no-slip
walls is minus the power of the magnetic torques and leaves the induction work (`INDU_WALLS`). Only with both torques on
the solids (`vtrq` and `mtrq` 1); [Rotating solids equations](rotating-solids-equations.md#energy).

**How the terms are computed.** spin_doctor's Lorentz force (`utils4pp.lorentz4pp`) acts per volume, and its power
$\int\mathbf u^*\cdot\mathbf G\,dV$ has no $\rho_0$. Its induction term (`induction4pp`) takes the components of
$\mathbf u$ with their $\lambda_1$ part, and holds for the non-solenoidal $\mathbf u$ of a variable density (resid𝐛
closes to $10^{-10}$ with 'Wu 2005', [Checks](#variable-density-checks-2026-10-28)). These spectral formulas are used
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
core, a thin wall, a resolved mantle), the flux of $\mathbf U_0\times\mathbf b$ through its surface is the induction work of the solid's own
rotation, $\mathrm{Re}\int(\mathbf V\times\mathbf b)\cdot(\nabla\times\mathbf b)^*dV$ over it ($\mathbf V$ rigid, the potential
field beyond carries none), and resid𝐄 keeps it (`utils4pp.INDU_ROT_SOLID`). Without it resid𝐄 was about $10^{-4}$ for a
conducting inner core turning with $\Delta\Omega(r_i) \neq 0$; with it the budget is the same in every frame.

## Checks

Each step of the port was checked when it was built: against `conductive_IC`, against exact and published solutions,
term by term against a symbolic derivation, and with the energy budgets. The parts were ported from the `conductive_IC`
branch (commit `39487bbf`), formula by formula, into this branch's conventions: on 2026-10-28 insulating walls (step 1),
a conducting inner core (step 2), any density profile (step 3), a toroidal background field (step 4), the full sphere,
inviscid runs and the poloidal fields of degree 2 (step 5); on 2026-10-30 the induction by a differential rotation, a
confined field, fields of any degrees and the Ferraro field, with couplings built from the recurrence algebra. The
scripts are in `tests/magnetic/` (`run_all.sh`).

### Against `conductive_IC`, insulating walls (2026-10-28)

Step 1 reproduces `conductive_IC` to round-off, for every eigenvalue and residual.

Spin-over geometry of the `conductive_IC` test: ricb 0.35, Ek 1e-3, no-slip walls (`bci = bco = 1`), N 68, lmax 64 (63
for m = 0), 2 ranks; $\Lambda = 1$, Pm = 1 (Le $= \sqrt{10^{-3}}$, Em $= 10^{-3}$), `cnorm = 'mag_energy'`, insulating
walls; `conductive_IC` at `39487bbf` with the same parameters. All five eigenvalues of each run:

| case | target | largest relative difference |
| --- | --- | --- |
| `'axial'`, m 1, symm −1 | 1.0i | 8e-15 |
| `'FDM'`, m 1, symm −1 | 1.0i | 7e-15 |
| `'axial'`, m 0, symm 1 | 0.3i | 1e-13 |
| `'FDM'`, m 0, symm 1 | 0.3i | 9e-14 |

The least damped m = 1 mode: λ = −0.104550975908 + 1.009269490234i (`'axial'`), −0.104198820746 + 1.008910499350i
(`'FDM'`). residσ and resid𝐛 agree with `conductive_IC`'s resid𝐮 and resid𝐛 mode by mode (1e-15 to 1e-10). Without the
field (`magnetic = 0`, structure `'Boussinesq'`, no-slip) the hydrodynamic spin-over is λ = −0.1019947097 +
1.0057336901i, the `conductive_IC` reference. Non-magnetic matrices are bitwise unchanged (`tests/diff_rot/regress.sh`).

### Conducting inner core (2026-10-28)

Step 2 reproduces `conductive_IC` with a conducting inner core to round-off, residuals included.

FDM ($l = 1$), `cnorm = 'Schmitt2012'`, ricb 0.35, Ek $10^{-5}$, $\Lambda = 10$, Pm 0.1, no-slip walls, insulating
mantle, conducting inner core, 4 ranks, the same parameters in `conductive_IC` (`39487bbf`). The m = 0 target is the
most torsional of the modes there at this resolution (ω 0.0747, σ −0.0233, 𝒯/𝒫 18, Geo 0.90).

| case | N, lmax | largest relative difference (5 eigenvalues) |
| --- | --- | --- |
| m 0, $\sigma_{i2o} = \mu_{i2o} = 1$, `N_cic` 64 | 80, 79 | 2e-13 |
| m 1 (target 1.0i) | 80, 80 | 4e-14 |
| m 0 | 128, 127 | 6e-13 |
| m 1 | 128, 128 | 1e-14 |
| m 0, $\sigma_{i2o} = 10^{-2}$ | 80, 79 | 3e-13 |
| m 0, `N_cic` 48 / 96 | 80, 79 | 3e-13 / 2e-13 |
| m 0, $\mu_{i2o} = 2$ | 80, 79 | 2e-13 |
| m 1, $\mu_{i2o} = 2$, $\sigma_{i2o} = 0.3$ | 80, 80 | 7e-14 |

The least damped m = 0 mode: λ = −0.023322278633 + 0.074686068661i (N 80), −0.023322202914 + 0.074686566522i (N 128).
residσ, resid𝐛 and resid𝐄 equal `conductive_IC`'s resid𝐮, resid𝐛, resid𝐄 to the printed digits (resid𝐄 with
$\mu_{i2o} \ne 1$ apart, see [Energy budgets](#energy-budgets)). N 80 is not enough at Ek $10^{-5}$: residuals $10^{-6}$
to $10^{-3}$, $10^{-14}$ to $10^{-6}$ at N 128. `N_cic` 48, 64 and 96 give the same eigenvalues to $2\cdot10^{-14}$. At
Ek $10^{-3}$ (the geometry above, $\Lambda$ = Pm = 1) 'axial' m 1 and 'FDM' m 0 agree to $3\cdot10^{-13}$, with all
residuals below $10^{-10}$.

### Variable density checks (2026-10-28)

No other code has the terms of step 3, so they were checked five ways: against their derivation, with operator
identities, row by row, with the energy budgets, and by continuity towards a uniform density. The scripts are in
`tests/magnetic/` (`run_all.sh`), with `bin/ru_insane.py`:

- **Derivation.** The rows derived from the definitions reproduce `conductive_IC`'s 44 uniform-density blocks exactly
  (factor −1, the A sign; l 3–6, m 0–3), and the $(\ln\rho_0)'$ terms equal the closed forms of
  [Variable density](#variable-density) to $\le 2\cdot10^{-14}$; the rows `v` and the T columns have none. The closed
  forms are those of the planning notes: none had to be corrected.
- **Uniform density unchanged.** The 4 insulating and 9 conducting-core cases above give A and B bitwise equal to the
  code before this step.
- **Operator identities** (`bin/ru_insane.py`, B0 'axial', no-slip):
  $\int\rho_0\mathbf u^*\cdot\mathbf F_L\,dV = -\mathrm{Hendrik}\,\overline{\int\mathbf b^*\cdot\nabla\times(\mathbf u\times\mathbf B_0)\,dV}$
  to $5\cdot10^{-15}$ ('Boussinesq') and $1\cdot10^{-10}$ ('Wu 2005', r_out 0.99, N 96; $7\cdot10^{-13}$ at N 160), m 0,
  1, 2; switching off the g-row or the u-row terms gives $2\cdot10^{-3}$ and 0.2. b block Hermitian positive and
  diffusion Hermitian negative to $10^{-14}$.
- **Row checks** (`row_check.py`, 'Wu 2005', N 192): every magnetic row block against the derivation, unweighted, to
  $10^{-14}$ ('axial' m 1, 'FDM' m 0 with `tol_B0` = 1e-15; $8\cdot10^{-9}$ with the default `tol_B0` = 1e-8, the cut of
  the B0 series). At N 96 the blocks with $\lambda_1$ are good to $10^{-7}$ only: the series of $(\ln\rho_0)'$ converge
  slowly, its pole sits at $r = 1/r_\mathrm{out}$.
- **Energy budgets** ('Wu 2005', r_out 0.99, ricb 0.35, FDM, conducting inner core, no-slip, $\Lambda$ = Pm = 1). At Ek
  $10^{-3}$, residσ, residω, resid𝐛 and resid𝐄 for m 0 and m 1 are $10^{-3}$ at N 68, $10^{-7}$ at N 128 and
  $\le 6\cdot10^{-10}$ at N 192. At Ek $10^{-4}$ (all residuals $\le 3\cdot10^{-9}$ at N 256):

    | mode | λ (N 256) | change |
    | --- | --- | --- |
    | least damped, m 1 | −0.005739718 + 1.003574886i | $3\cdot10^{-9}$ at N 192, $10^{-8}$ with lmax 96 instead of 64 |
    | m 0, near 0.0442i | −0.017420795 + 0.044242829i | |

    The budgets use spin_doctor's own Lorentz and induction formulas, so they check the rows independently; the f and g
    rows need no $\rho_0$ weight, although $\lambda_1 \approx -200$ at the outer wall.
- **Continuity** ('Wu 2005' with r_out = $\sqrt\epsilon$, so the stratification is $O(\epsilon)$, against 'Boussinesq';
  axial, m 1, Ek $10^{-3}$, N 96): the eigenvalue moves linearly with the stratification.

    | $\epsilon$ | $10^{-4}$ | $10^{-3}$ | $10^{-2}$ | $10^{-1}$ |
    | --- | --- | --- | --- | --- |
    | (λ − λ of 'Boussinesq')/ε | 0.05186 − 0.00053i | 0.05188 | 0.05215 | 0.05498 |

### Toroidal field (2026-10-28)

Step 4 was checked against its derivation, with the energy budgets, and against the exact frequencies of the Malkus
(1967) field.

- **Derivation.** The closed forms equal derive.py's rows ($b_0$ = ('tor', 1) and ('tor', 2), any $\rho_0$) for l 3–6, m
  0–3 (`derive_check.py`); every coefficient fitted from l 2–9, m 0–3 (n = 1) and l 3–9 (n = 2), exactly.
- **Earlier cases unchanged.** The 13 cases of steps 1 and 2 and three 'Wu 2005' cases of step 3 give A and B bitwise
  equal to the code before this step.
- **Row checks** ('Wu 2005', N 192): 'Malkus' m 1 to $10^{-14}$; 'wall' m 0, 2 to $2\cdot10^{-10}$, in the u-row blocks
  with $k\lambda_1$ only (they vanish at the wall, where $\lambda_1$ is large, so the relative error of the near-wall
  $\lambda_1$ series shows; it does not decrease with N).
- **Energy budgets.** 'Malkus' and 'wall', 'Boussinesq' (N 68, 96) and 'Wu 2005' (N 128, 192), insulating and conducting
  inner core, m 0 and 1, Ek = Em = $10^{-3}$, $\mathrm{Le}^2 = 10^{-3}$, `cnorm_tor = 'mag_energy'`: residσ, resid𝐛,
  resid𝐄 $\le 6\cdot10^{-13}$, eigenvalues the same at both N to the printed digits. Unlike a poloidal field, the
  toroidal one has no radial component, hence no Hartmann layer at the walls, and 'Wu 2005' converges fast.
- **Degree 2 and combined fields.** 'Malkus' and 'wall' with `B0_tor_l = 2`, 'Boussinesq' (N 68, 96) and 'Wu 2005' (N
  128, 192), conducting inner core, m 0 and 1: residuals $\le 10^{-11}$, eigenvalues the same at both N to the printed
  digits. FDM plus a degree-2 'wall' field ('Wu 2005'): residuals $2\cdot10^{-10}$ at N 192 (the poloidal field's
  Hartmann layer sets the resolution); row check of 'axial' plus a degree-2 'Malkus' field, m 2: $10^{-14}$. A 'user'
  profile `'r'` gives the 'Malkus' eigenvalues to $10^{-12}$.
- **Malkus (1967).** With $B_\phi = \mathrm{Le}\,s$ the induction is $im\,\mathrm{Le}\,\mathbf u$ and the Lorentz force,
  up to a gradient,

    $$
    \mathbf F_L = \frac{im\,\mathrm{Le}^2}{\lambda}\left(im\,\mathbf u + 2\hat{\mathbf z}\times\mathbf u\right),
    $$

    so the ideal problem reduces to that of the inertial waves (frequency $\lambda_\mathrm{in}$) in any container:

    $$
    \omega = \frac{\lambda_\mathrm{in}}{2}\left[1 \pm \sqrt{1 + \frac{4\mathrm{Le}^2 m(m - \lambda_\mathrm{in})}{\lambda_\mathrm{in}^2}}\right]
    $$

    (kore's Im of the eigenvalue; the spin-over, $\lambda_\mathrm{in} = m = 1$, does not move). For small Le the shift
    divided by $\mathrm{Le}^2$ tends to

    $$
    \frac{m(m-\lambda_\mathrm{in})}{\lambda_\mathrm{in}} .
    $$

    For six m = 1 modes near 0.75i ('Boussinesq', ricb 0.35, Pm 1, insulating walls) the frequency shift is exactly
    quadratic in Le (the same shift divided by $\mathrm{Le}^2$ for $\mathrm{Le}^2$ = 1e-6, 4e-6 and 1.6e-5), and its
    distance to this limit falls with the diffusivities: 2–8 % at Ek = Em = $10^{-5}$, 0.4–1 % at $10^{-6}$ (e.g. 0.3808
    against 0.3822 for $\lambda_\mathrm{in}$ = 0.7235). `tests/magnetic/malkus_check.py`.

### Full sphere, inviscid runs and degree-2 poloidal fields

Step 5 (2026-10-28) was checked against the inviscid modes of Wu (2005) and of Luo & Jackson (2022), against
`conductive_IC`, and with the energy budgets.

- **Hydrodynamics.** Inviscid Boussinesq full sphere (Ek = 0, `'Boussinesq'`): the inertial modes of Wu (2005, Table 1,
  β = 0) to the 10 digits given (e.g. 0.6666666667, 1.4964360838, 1.2319250547 for $\lvert m\rvert = 2$); 'Wu 2005'
  against its Table 2. No non-magnetic code changed (`regress.sh` bitwise).
- **Against `conductive_IC`** (`39487bbf`, the geometry of the first check, all eigenvalues of each run): full sphere,
  'axial', 'FDM' ($l$ = 1, 2), 'G21 dipole', 'Luo_S1', 'Luo_S2', m 0–2 and both symmetries, $\le 5\cdot10^{-13}$; with
  an inner core (insulating or conducting, `B0_ic = 'potential'`): 'G21 dipole', 'Luo_S1', 'Luo_S2', 'FDM' $l = 2$ (m 0,
  3), $\le 6\cdot10^{-13}$.
- **Derivation.** `derive_check.py` also checks the tables of `induction_pol2`, `lorentz_pol2`, read from
  `operators.py`, against derive.py's rows (l 3–6, m 0–3), to round-off.
- **Row checks** in a full sphere (`row_check.py` handles the parity-reduced bases and $\Delta l = \pm2$): 'Boussinesq'
  to $10^{-14}$; 'Wu 2005' needs N 288 (the pole of $(\ln\rho_0)'$ at $r = 1/r_\mathrm{out}$ is nearer to the domain
  $[-1, 1]$ than to a shell's; $3\cdot10^{-7}$ at N 192).
- **Energy budgets**, 'Wu 2005', Ek = Em = $10^{-3}$ ('Boussinesq': to $10^{-14}$):

    | case | N 128 | N 192 | N 256 |
    | --- | --- | --- | --- |
    | 'Luo_S2', full sphere, m 0 | $10^{-7}$ | $3\cdot10^{-11}$ | |
    | 'Luo_S2', full sphere, m 1 | $3\cdot10^{-6}$ | $10^{-9}$ | |
    | 'Luo_S2', insulating or conducting inner core | | $10^{-10}$ | |
    | FDM $l = 2$, full sphere, `'Schmitt2012'` (its Hartmann layer) | $5\cdot10^{-4}$ | $6\cdot10^{-6}$ | $3\cdot10^{-8}$ |

- **Malkus (1967), ideal.** Inviscid Boussinesq full sphere, $B_\phi = \mathrm{Le}\,s$, Em $10^{-9}$, m 2: for ten
  inertial modes $\lambda_\mathrm{in}$ of both symmetries and Le 0.1, 0.3, both branches of the Malkus frequency above
  to $10^{-5}$–$7\cdot10^{-3}$ relative at N 64 (the fast branch to $10^{-4}$ at N 96). The error does not change with
  Em ($10^{-8}$ to $10^{-10}$) or with Le: it is that of the magnetic boundary layer the insulating wall imposes
  (Malkus's ideal solution has a tangential $\mathbf b$ there), not resolved at these N.
- **Torsional modes of Luo & Jackson (2022)** (`conductive_IC`'s `tests/torsional_s2`: 'Luo_S2', inviscid, Lu
  $2\cdot10^4$, m 0, `timescale = 'Alfven'`). This code converges to their modes as $N^{-3}$:

    | mode | Luo & Jackson | relative difference |
    | --- | --- | --- |
    | 1 | −0.0065952461 + 1.0335959942i | $1.3\cdot10^{-5}$ (N 128), $4\cdot10^{-7}$ (N 320) |
    | 2 | −0.0141833345 + 1.9095123626i | $2\cdot10^{-4}$ (N 128), $7\cdot10^{-6}$ (N 384) |

    `conductive_IC` is within $10^{-6}$ already at N 128 because it mishandles the $D^3$ and $D^2$ Lorentz terms
    ([Rows and implementation](#rows-and-implementation), inviscid runs), whose effect there is only $10^{-7}$: with its
    bases set as here it gives this code's eigenvalues to $10^{-9}$. The slow convergence comes from the u rows in
    $C^{(3)}$, where the unresolved boundary layer of $\mathbf b$ (Em $5\cdot10^{-9}$) enters through $f'''$.

### Differential rotation, confined, general and Ferraro fields (2026-10-30)

The induction by a differential rotation and the fields of any degrees were checked against sympy, against exact
solutions (a rigid rotation, the uniform axial field, a closed-form wound field), and with the energy budgets; without
the new features nothing changed.

- **∇×(U0×b) against sympy** (`tests/magnetic/dr_induction_check.py`): the blocks of `diff_rot.induction_block` against
  `derive.rows_dr` for $n$ = 0–6, $m$ = 0–5, $l$ up to 25, term by term: $\le 10^{-13}$ at low $l$; at $l$ 20–25 the
  agreement is limited by derive.py's own quadrature ($10^{-10}$).
- **Rigid rotation, exact** (`rigid_check.py`): a uniform $\Delta\Omega = c$ must give the eigenvalue of Gaspard $1 + c$
  without rotation, minus $imc$. All eigenvalues agree to $\le 5\cdot10^{-14}$: hydrodynamics, 'axial', 'FDM', 'Luo_S2',
  'Malkus', 'wall' of degree 2, a conducting inner core, a full sphere.
- **Budgets with a solar-like $\Delta\Omega = f_0 + f_2P_2 + f_4P_4$** (Ek = Em = $10^{-3}$, $\Lambda$ = 1): 'Malkus',
  'axial' and 'Luo_S2', shell and full sphere, 'Boussinesq' and 'Wu 2005': residσ, resid𝐛, resid𝐄 $10^{-13}$–$10^{-9}$.
- **The algebra against the closed forms** (`mag_couplings` 'algebra' against 'closed', same runs): eigenvalues equal to
  $\le 6\cdot10^{-13}$ for 'axial' (m 0, 1), 'FDM' (degree 2), 'Luo_S2' (shell, full sphere, 'Wu 2005'), 'FDM' with 'Wu
  2005', 'Malkus' (and 'Wu 2005'), 'wall' of degree 2, 'axial' + 'wall', a conducting inner core, 'confined', and with a
  differential rotation. **Against sympy** (`algebra_check.py`): every block for poloidal and toroidal fields of degree
  1–4 (3 and 4 have no closed forms), several $m$, $l$ up to 12.
- **'general'**: $h_1 = r/2$ gives exactly the 'axial' eigenvalues, $k_1 = r$ exactly the 'Malkus' ones; fields of
  degrees 1 + 3 (full sphere), 2 + 4 (shell), toroidal 1 + 3, and a poloidal 1 + 3 with a toroidal 2 and a differential
  rotation (also 'Wu 2005'): budgets $\le 5\cdot10^{-12}$.
- **'confined'**: degree 1 and 2, shell layer [0.4, 0.8] and full-sphere layer [0, 0.6]: budgets $10^{-12}$–$10^{-14}$
  (shell), $4\cdot10^{-10}$ (degree 1, N 192) and $2\cdot10^{-9}$ (degree 2, N 384) in the full sphere; with
  $\Delta\Omega$ uniform inside the layer, `ferraro_mismatch` = $10^{-29}$ and the budgets close as without rotation.
- **'ferraro'**: with a cylindrical rotation and $G(w) = w$ the field is the uniform axial one, and the eigenvalues
  equal those of 'axial' to 10 digits; with $\Delta\Omega \propto s^2(1 - s^2)$ (degrees 0, 2, 4), $\psi$ is rebuilt
  from degrees 1–7 to $4\cdot10^{-15}$, `ferraro_mismatch` is $5\cdot10^{-14}$, and the budgets close to
  $\le 2\cdot10^{-11}$ in a shell, a full sphere and 'Wu 2005'.
- **`diff_rot_type = 'ferraro'`**: the uniform axial field with $F = \psi$ gives the eigenvalues of the cylindrical
  profile to 10 digits, and so does the round trip through `B0 = 'ferraro'` ($G = 5w$; budgets $4\cdot10^{-14}$).
  Budgets $\le 10^{-10}$ for all of the following, with `ferraro_mismatch`:

    | field | $F$ | `ferraro_mismatch` |
    | --- | --- | --- |
    | 'FDM' | $\psi$ | $3\cdot10^{-12}$ |
    | 'G21 dipole', 'Luo_S1' | $\psi$ | $6\cdot10^{-12}$ |
    | full sphere | $\psi$ | $2\cdot10^{-12}$ |
    | 'general', degrees 1 + 3 | $\psi + \psi^2$ | $2\cdot10^{-11}$ |
    | 'Luo_S2' | $\psi^2$ | $2\cdot10^{-10}$ |
    | 'confined', cut at `diff_rot_nmax` 4 | $\psi$ | $7\cdot10^{-9}$ |

    Refused as they should be: an odd $F$ with 'Luo_S2', `B0 = 'ferraro'`, a toroidal field alone, a conducting inner
    core with $\psi$ not constant on it. With a toroidal part of degree 2: `B0 = 'ferraro'` + 'Malkus' and 'Luo_S1' +
    'wall' with `diff_rot_type = 'ferraro'` close their budgets to $8\cdot10^{-12}$ and $10^{-9}$.
- **`B0_wound = 1`** (`tests/magnetic/wound_check.py`): a uniform axial $\mathbf B_p$ with $\Delta\Omega = a r^2$ gives
  one degree, $k_2 = A r^4/14 + C r^2 + D r^{-3}$: the solution matches it to $7\cdot10^{-14}$ (shell), $4\cdot10^{-13}$
  (conducting inner core) and $3\cdot10^{-12}$ (full sphere). The residual of the steady induction equation, by finite
  differences in $(r, \theta)$ with $\mathbf B_p$ from its flux function, is $\le 8\cdot10^{-10}$ ('general' 1 + 3,
  'Luo_S2' in a full sphere, 'FDM' with a conducting inner core), the floor of the differences; the $k_n$ change by
  $\le 2\cdot10^{-11}$ from 97 to 385 nodes. Budgets with the ratio of the maxima of the wound and the poloidal field
  from 0.5 to 4.4 ('general', 'FDM' with a conducting inner core, 'Luo_S2' in a full sphere, 'axial' plus a 'Malkus'
  part of degree 2): $\le 2.4\cdot10^{-10}$.
- **Unchanged without the new features**: `regress.sh` bitwise, heat_dr, ru_insane identical, and 26 magnetic cases
  build A and B bitwise equal to the code before.

### Free decay without the flow (2026-11-06)

Without the flow (`hydro = 0`, as `conductive_IC`) a field of uniform conductivity only diffuses, and each mode decays at
the rate MagnetD $k^2$, where $k$ makes the spherical Bessel functions $j_l(kr)$, $y_l(kr)$ meet the conditions at the
walls: $j_{l-1}(k)\,y_{l+1}(kr_i) = y_{l-1}(k)\,j_{l+1}(kr_i)$ for a poloidal field between insulators,
$j_l(k)\,y_l(kr_i) = y_l(k)\,j_l(kr_i)$ for a toroidal one, and a 3 × 3 determinant with $j_l(k\sqrt{\sigma_{i2o}}\,r)$
inside a conducting inner core. `tests/decay/decay_check.py` finds the roots with scipy and compares (`ricb` = 0.35, N 48,
`lmax` 15, `m` 0, MagnetD 1):

| case | modes | largest relative error | resid𝐛, resid𝐄 |
| --- | --- | --- | --- |
| insulating walls | poloidal $l$ = 1 (two), 3; toroidal $l$ = 2: rates 10.041029, 45.156235, 33.260470, 37.495006 | 4e-15 | ≤ 5e-12 |
| conducting inner core, `sigma_i2o` 0.3 | the same: 9.993017, 43.759526, 33.248628, 35.942816 | 6e-14 | ≤ 8e-11 |
| full sphere | $\pi^2$, $4\pi^2$, and 33.217462 (poloidal $l$ = 3 and toroidal $l$ = 2 together) | 9e-15 | ≤ 2e-13 |

With a differential rotation the modes of the field alone equal, to $10^{-15}$, those of the same run with the flow and a
B0 of $10^{-9}$, where the field and the flow decouple; the entropy of `'Boussinesq'` decays at the exact rates too, and
the field of `just_induction.py` with `solve_ind.py` is that of a mode without Lorentz force to $10^{-14}$.

### Thin conducting walls (2026-11-06)

The thin walls were ported from `conductive_IC` (2d84351) and checked against exact decay rates, against that branch, and
with the energy budgets (`tests/twa/twa_check.py`).

**Free decay.** Without the flow each mode decays at MagnetD $k^2$, where $k$ makes $j_l(kr)$, $y_l(kr)$ meet the
thin-wall conditions with $\mathcal D_l F = -k^2 F$ (`ricb` 0.35, N 48, `m` 0, MagnetD 1; poloidal $l$ = 1 (two modes),
3 and toroidal $l$ = 2):

| walls | largest relative error | resid𝐄 |
| --- | --- | --- |
| outer, $c = 0$, $c_1 = 0.1$ | 9e-16 | 8e-14 |
| outer, $c = 0.05$, $c_1 = 0.1$, $p$ = 1, 3, $\infty$ | 4e-15 | 2e-2 to 3e-2 |
| inner, $c = 0$, $c_1 = 0.2$ | 2e-15 | 4e-14 |
| both, $\mu$ = 1.5 | 2e-15 | nan |
| full sphere, $c = 0.05$, $c_1 = 0.1$, $p$ = 2, 3 | 3e-15 | 2e-2 |

With $c \neq 0$ resid𝐄 scales as $c\,c_1$ (at $c\,c_1 = 5\cdot10^{-5}$: $2\cdot10^{-5}$ to $5\cdot10^{-4}$ whether $c$ or
$c_1$ is the larger), at or below $\varepsilon_\text{TWA}$; the toroidal modes close to round-off (their condition has no
$c$).

**Against `conductive_IC`.** Thirteen runs with the flow (`'Boussinesq'`, Ek, Em and Le² $10^{-3}$, N 48, `lmax` 19 to
21): no-slip and stress-free walls, a thin wall at either or both boundaries, $p$ = 1, 2, 3, 4 and $\infty$, $\mu$ = 1.3,
the fields `'axial'`, `'G21 dipole'`, `'Luo_S2'` and `'FDM'`, $m$ = 0, 1, 2, both equatorial symmetries, and a full
sphere. The eigenvalues agree to $3\cdot10^{-14}$, resid𝐄 and εTWA to the printed digits, and the energies and losses of
the budget (MEic, E_w, MEext, Ohm_w, relative to Ohm_f) to about $10^{-10}$.

**Budgets.** With a slipping fluid and $c = 0$, resid𝐄 closes to about $10^{-10}$ (N 48), and with `'Wu 2005'`
(`r_out` 0.9, N 96, `lmax` 40) to $2.5\cdot10^{-10}$, as for insulating walls. A rigid rotation offset
(`diff_rot_type = 'rigid'`, $a$ = 0.3, thin walls with $c \neq 0$ and a slipping fluid) gives the eigenvalues of the frame
turning with the walls to $6\cdot10^{-15}$, with resid𝐄 and εTWA unchanged. With walls turning at different rates
(`'shellular_boussinesq'`) resid𝐄 closes to $3\cdot10^{-10}$, and for a conducting inner core turning with the fluid it
went from $1.5\cdot10^{-4}$ to $5\cdot10^{-10}$ (`INDU_ROT_SOLID`, above).

### Resolved conducting mantle (2026-11-06)

The resolved mantle was ported from `conductive_IC` (2d84351) and checked against exact decay rates, against that branch,
against the thin walls, and with the energy budgets (`tests/mantle/mantle_check.py`).

**Free decay.** Without the flow each mode decays at MagnetD $k^2$, with $j_l(kr)$, $y_l(kr)$ in the fluid and in the
layer $j_l(k\sqrt\sigma\,r)$, $y_l(k\sqrt\sigma\,r)$ for a uniform $\sigma$, or the solution of its equations integrated
from $r_s$ (scipy's DOP853 at rtol $10^{-13}$), matched as above (`ricb` 0.35, N 48, `m` 0, MagnetD 1; poloidal $l$ = 1
(two modes), 3 and toroidal $l$ = 2):

| layer | rates | largest relative error | resid𝐛, resid𝐄 |
| --- | --- | --- | --- |
| $\sigma$ = 0.3, $r_s$ = 1.4 (`conductive_IC`'s test) | 8.220477, 33.204363, 29.002791, 28.121940 | 1e-15 | ≤ 1e-13 |
| $\sigma = 0.3\,e^{-10(r-1)}$, $r_s$ = 1.4 (`conductive_IC`'s test) | 9.521791, 42.322149, 31.916490, 34.954516 | 7e-15 | ≤ 8e-13 |
| full sphere, $\sigma$ = 2, $r_s$ = 1.3 | 3.790859, 19.320734, 11.729515, 15.398290 | 1e-15 | ≤ 6e-13 |
| conducting inner core (`sigma_i2o` 0.3), $\sigma = 0.5\,r^{-6}$, $r_s$ = 1.5 | 8.623323, 36.247125, 29.781714, 29.597419 | 2e-14 | ≤ 1e-12 |

**Against `conductive_IC`.** Five runs with the flow (`'Boussinesq'`, Ek, Em and Le² $10^{-3}$, N 48, `lmax` 19 to 21):
no-slip walls under a uniform layer, a fluid slipping along both walls with a field of degree 2 under an exponential
layer, a full sphere with a slipping fluid, a conducting inner core, and $m$ = 2 under a thin layer ($r_s$ = 1.1,
$\sigma$ = 2) with a slipping fluid. The eigenvalues agree to $7\cdot10^{-15}$, resid𝐄 to the printed digits, and the
energies and losses of the budget (E_w, Ohm_w, MEext, relative to Ohm_f) to about $10^{-10}$.

**Limits.** A layer of thickness $h$ and conductance $\sigma h$ = 0.01, with a fluid slipping along it, differs from the
thin wall with $c = h$ and $c_1$ = 0.01 by $1.0\cdot10^{-4}$, $5.2\cdot10^{-5}$ and $2.6\cdot10^{-5}$ for $h$ = 0.01, 0.005
and 0.0025: at first order in $h$. A layer of $\sigma = 10^{-9}$ gives the eigenvalues of an insulating mantle to
$3\cdot10^{-12}$.

**Budgets and rotation.** With `'Wu 2005'` (`r_out` 0.9, N 64), a conducting inner core and a fluid slipping along an
exponential layer, a rigid rotation offset ($a$ = 0.3) gives the eigenvalues of the frame turning with the solids to
$3\cdot10^{-15}$, with resid𝐄 $2.5\cdot10^{-10}$ in both frames, and with the layer, the inner core and the fluid turning
at different rates ($\Delta\Omega = 0.4\,(0.5 + r^2)$) resid𝐄 closes to $3\cdot10^{-10}$. The field of `just_induction.py`
with `solve_ind.py` is that of a mode without Lorentz force to $4\cdot10^{-14}$, in the fluid and in the layer.

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
