# Magnetic field

With `magnetic = 1` kore adds an imposed background magnetic field $\mathbf{B}_0$, the induction equation for the
perturbation field $\mathbf{b}$ and the Lorentz force on the flow. The operators, the background fields and the boundary
conditions are ported from the `conductive_IC` branch (commit `39487bbf`), formula by formula, into this branch's
conventions (2026-10-28: step 1, insulating walls; step 2, a conducting inner core; step 3, any density profile; step 4, a
toroidal background field; step 5, the full sphere, inviscid runs and the poloidal fields of `conductive_IC` of degree 2;
2026-10-30: the induction by a differential rotation, a confined field, fields of any degrees and the Ferraro field, with
couplings built from the recurrence algebra).

**What works now:** a spherical shell (`ricb > 0`) with an insulating mantle and an insulating or conducting
(`'conducting, Chebys'`) inner core, or a full sphere (`ricb = 0`, insulating mantle), any `structure` (variable density,
the anelastic terms below), eigenvalue problems, a poloidal $\mathbf{B}_0$ (`'axial'`, `'G21 dipole'`, `'Luo_S1'`, `'FDM'` of
degree 1 or 2, `'Luo_S2'`, `'confined'`, `'general'` of any degrees, `'ferraro'`) and/or a toroidal one (`'Malkus'`, `'wall'`
or `'user'` of degree 1 or 2, `'general'` of any degrees), uniform magnetic diffusivity, with or without a differential
rotation (`diff_rot = 1`, any profile); `ViscosD > 0`, or `ViscosD = 0` (Ek = 0) in a full sphere. `background.check()` stops
on anything else.

## Equations

The field is written as $\mathbf{b} = \nabla\times\nabla\times(f\,\mathbf{r}) + \nabla\times(g\,\mathbf{r})$, with no
$\rho_0$ (it is solenoidal by itself). In units of the time scale $\tau$ (`timescale`),

$$
\frac{\partial \mathbf{b}}{\partial t} = \nabla\times(\mathbf{u}\times\mathbf{B}_0) + \mathrm{MagnetD}\,\nabla^2\mathbf{b},
\qquad
\mathbf{F}_L = \frac{\mathrm{Hendrik}}{\rho_0}\left[(\nabla\times\mathbf{b})\times\mathbf{B}_0 + (\nabla\times\mathbf{B}_0)\times\mathbf{b}\right],
$$

$\mathbf{F}_L$ per unit mass in the momentum equation, with

| factor | meaning | `set_scales()` |
| --- | --- | --- |
| `Hendrik` | $(\tau B_0/r_0)^2/(\rho_0\mu_0)$ | $(\mathrm{Gaspard}\cdot\mathrm{Le})^2$ |
| `MagnetD` | $\eta_0\tau/r_0^2$ | $\mathrm{Gaspard}\cdot\mathrm{Em}$ |

The inputs are the Lehnert number `Le` $= B_0/(\Omega r_0\sqrt{\rho_0\mu_0})$ and the magnetic Ekman number
`Em` $= \eta_0/(\Omega r_0^2)$, both defined with the rotation rate as `Ek` is: $\mathrm{Em} = \mathrm{Ek}/\mathrm{Pm}$ and
the Elsasser number is $\Lambda = \mathrm{Le}^2/\mathrm{Em}$. Besides `'rotation'` and `'viscous'`, `set_scales()` knows the
time scales `'Alfven'` (Gaspard $= 1/\mathrm{Le}$, so Hendrik $= 1$) and `'magnetic'` (Gaspard $= 1/\mathrm{Em}$, so
MagnetD $= 1$). Set `Le` and `Em` before calling `set_scales()`. The magnetic diffusivity $\eta$ is uniform for now
(`background.diffusivity(..., 'eta')` returns 1; the operators already have the $\eta'$ terms).

## Background field

$\mathbf{B}_0 = c_\mathrm{norm}\nabla\times\nabla\times(h(r)\,Y_l^0\,\mathbf{r})$, axisymmetric and poloidal
(`bin/background_field.py`):

| `B0` | $h(r)$ | degree, symmetry |
| --- | --- | --- |
| `'axial'` | $r/2$: the uniform field $c_\mathrm{norm}\hat{\mathbf{z}}$, no current | $l = 1$, antisymmetric |
| `'FDM'` | the poloidal free decay mode of the shell, $j_l(\beta r)\,y_{l-1}(\beta) - j_{l-1}(\beta)\,y_l(\beta r)$ (Zhang & Fearn 1995), $\beta$ the root of `findbeta` near `beta`; with `ricb = 0`, $j_l(\beta r)$ with $j_{l-1}(\beta) = 0$ (Gubbins & Roberts 1987) | $l$ = `B0_l` = 1 (antisymmetric) or 2 (symmetric) |
| `'G21 dipole'` | $r/6 - r^3/10$ (Gerick et al. 2021) | $l = 1$, antisymmetric |
| `'Luo_S1'` | $5r - 3r^3$ (Luo & Jackson 2022; the same field as `'G21 dipole'`) | $l = 1$, antisymmetric |
| `'Luo_S2'` | $157r^2 - 296r^4 + 143r^6$ (Luo & Jackson 2022) | $l = 2$, symmetric |
| `'confined'` | $r^l(r_2^2 - r^2)^p$ ($r_1 = 0$) or $r^l(r - r_1)^p(r_2 - r)^p$ in $r_1 \le r \le r_2$, zero outside ($r_1$, $r_2$ = `B0_conf_r1`, `B0_conf_r2`, $p$ = `B0_conf_p`, default 12) | $l$ = `B0_l` = 1 or 2 |
| `'general'` | $\sum_n h_n(r)\,Y_n^0$, `B0_profiles` = {n: 'numpy expression of r, p'} ($p$ = `B0_param`) | any degrees of one parity |
| `'ferraro'` | the field whose flux function is $\psi = G(\Delta\Omega)$ (below) | odd degrees up to `B0_ferraro_nmax`, antisymmetric |

All match a potential field at the outer wall. `'G21 dipole'`, `'Luo_S1'` and `'Luo_S2'` are full-sphere fields: with an inner
core, `B0_ic = 'potential'` (default) adds $\beta_i r^{-(l+1)}$ so that $rh' - lh = 0$ at the inner boundary (the field of
the currents in the fluid, continued as a potential field into the core), and `'full sphere'` keeps $h$ as it is
(`conductive_IC`'s `B0_beta_ic`). `cnorm` is a number, `'rms_cmb'` (rms of the radial field over the outer wall
1), `'mag_energy'` ($\int B_0^2\,dV = 1$ over the shell) or `'Schmitt2012'` ($= 4$), computed as in `conductive_IC`.
`tol_B0` (default 1e-8) is the relative cut of the Chebyshev series of $r^p h^{(k)}$: it sets the band width of the
magnetic blocks.

**Confined field** (2026-10-30). `'confined'` is zero outside $r_1 \le r \le r_2$, and $h$ and its first $p - 1$ derivatives
vanish at the edges, so there are no surface currents ($B_r \propto h$, $B_\theta \propto h' + h/r$ and the current are
continuous). Its purpose: a poloidal field inside a region where $\Delta\Omega$ is uniform is a steady background with the
differential rotation (Ferraro, below). With an inner core the region must lie in the fluid ($r_1 \ge$ `ricb`). The edges
are smooth only to order $p$, so the Chebyshev series converge algebraically: with $p = 12$ the budgets close to $10^{-12}$ at
N 96–128 in a shell, but a region that reaches the centre of a full sphere needs N 192 (degree 1) to 384 (degree 2) for
$10^{-9}$. $h$ is evaluated in factored form (expanded into monomials, a polynomial of degree $2p + l$ loses all its digits
to cancellation near the edges).

**Any degrees** (2026-10-30). `'general'` takes $h_n$ degree by degree as expressions; their derivatives come from Chebyshev
interpolants (degree max(64, 2N), at $\lvert r\rvert$ with the parity $(-1)^n$ in a full sphere, where $h_n$ should go as $r^n$ at the
centre). The degrees must have one parity (one equatorial symmetry). `cnorm = 'mag_energy'` sums $\int B_0^2 dV$ over the
degrees. The couplings come from the recurrence algebra (`mag_algebra.py`, below).

**Ferraro field** (2026-10-30, needs `diff_rot = 1`). An axisymmetric poloidal field is $\mathbf{B}_0 = \nabla\psi\times\nabla\phi$;
with $\psi = G(\Delta\Omega)$ its field lines lie on the surfaces of constant rotation, so $\mathbf{B}_0\cdot\nabla\Omega = 0$
(Ferraro's law of isorotation) and the rotation and the field are a steady background together. In kore's form
$\psi = r\sum_n h_n(1 - x^2)P_n'(x)$, and the functions $(1 - x^2)P_n'$ are orthogonal with the weight $1/(1 - x^2)$, so
$h_n(r) = \frac{2n+1}{2n(n+1)}\frac{1}{r}\int_{-1}^{1}\psi(r, x)\,P_n'(x)\,dx$, computed by Gauss–Legendre quadrature. $\Delta\Omega$
has only even $P_n$, so only odd $n$ appear. $G$ = `B0_ferraro_G`, a numpy expression of `w` = $\Delta\Omega$ and `p`. The
assumptions, which `background.ferraro_report` prints at the start of a run: $\psi$ must vanish on the axis (it must be
constant there for a regular field; for a rotation that varies along the axis, $G$ has to vanish on the values
$\Delta\Omega$ takes there), and at the outer wall if the field is to stay inside (otherwise it meets the insulating mantle
with a surface current, which the linear problem allows); the series is cut at `B0_ferraro_nmax`, which sets the residual
$\mathbf{B}_0\cdot\nabla\Omega$. A toroidal part (`B0_tor`, any type) can be added: it is consistent with any axisymmetric rotation, and it
must have the field's equatorial symmetry (an even degree). Example: for a cylindrical rotation $\Delta\Omega \propto s^2$, $G(w) = w$ gives $\psi \propto s^2$, the
uniform axial field.

**The rotation from the field: `diff_rot_type = 'ferraro'`** (2026-10-30). The reverse choice: the poloidal field is given (any
type, `'general'` included) and the rotation is deduced from it, $\Delta\Omega = F(\psi)$, with $\psi$ the field's own flux
function. Use it when the field is the starting point (a model of an observed or a dynamo field) and the rotation only has to
be compatible with it; use `B0 = 'ferraro'` when the rotation is. Since $\Delta\Omega$ is constant on every field line,
$\mathbf{B}_0\cdot\nabla\Delta\Omega = 0$ and the background is steady. Method: $\psi(r, x) = r\sum_n c\,h_n(1 - x^2)P_n'(x)$ from the
radial profiles of the field (`background_field.pol_profiles`), $F$ evaluated on a Gauss–Legendre grid in $x$ and projected on
the Legendre polynomials,

$$
f_n(r) = \frac{2n+1}{2}\int_{-1}^{1}F\big(\psi(r, x)\big)\,P_n(x)\,dx, \qquad n = 0, 2, \dots, \texttt{diff\_rot\_nmax},
$$

and $\Delta\Omega$ = `diff_rot_ampl` $\cdot\sum_n f_n P_n$, as for the other rotation profiles: the force on the flow, the induction
term and the thermal wind need nothing new. $F$ = `diff_rot_F` (default `'psi'`), a numpy expression of `psi` and `p`
(`diff_rot_param`). Rules, checked at the start of a run: a poloidal `B0` is needed (a toroidal field alone has no $\psi$);
`B0 = 'ferraro'` is refused (each would be built from the other); kore's $\Delta\Omega$ has only even $P_n$, so $F(\psi)$ must be
symmetric about the equator: any $F$ for a dipolar field ($\psi$ even in $x$), an even $F$ (e.g. of `psi**2`) for a quadrupolar one
($\psi$ odd), else the run stops; with a conducting inner core $\Delta\Omega(r_i, \theta)$ must be uniform, which in general means
$\psi$ constant on the inner core's surface. A toroidal part of `B0` is left as it is. The series is cut at `diff_rot_nmax` (even): `ferraro_mismatch` then reports only
the truncation, at round-off for fields of low degree with a polynomial $F$. Example: the uniform axial field $\psi = r^2\sin^2\theta/2$ with
$F(\psi) = \psi$ gives $\Delta\Omega \propto s^2$, the cylindrical profile.

**Toroidal field** (step 4). $\mathbf{B}_0^{tor} = c_\mathrm{tor}\nabla\times(k(r)\,Y_n^0\,\mathbf{r})$, purely azimuthal,
$B_\phi = -c_\mathrm{tor}\,k\,\partial_\theta Y_n^0$: $c_\mathrm{tor}\,k\sin\theta$ for $n$ = `B0_tor_l` = 1, $3c_\mathrm{tor}\,k\sin\theta\cos\theta$ for
$n = 2$; $c_\mathrm{tor}$ = `cnorm_tor` (a number, `'mag_energy'` or `'Schmitt2012'`; $\int B_0^2 dV = n_l L\int k^2 r^2 dr$):

| `B0_tor` | $k(r)$ | |
| --- | --- | --- |
| `'Malkus'` | $r$ | $\mathbf{B}_0 = c_\mathrm{tor}\,s\,\hat{\boldsymbol\phi}$ (Malkus 1967), current $\nabla\times\mathbf{B}_0 = 2c_\mathrm{tor}\hat{\mathbf{z}}$ |
| `'wall'` | $r(r - r_i)(1 - r)/(1 - r_i)^2$ | zero at both walls |
| `'user'` | `B0_tor_profile`, a numpy expression of $r$ and $p$ = `B0_tor_param` | its derivatives from a Chebyshev interpolant (degree max(64, 2N)); written to `params.dat` after `#` |
| `'general'` | $\sum_n k_n Y_n^0$, `B0_tor_profiles` = {n: 'expression'} | any degrees of one parity (2026-10-30) |

`B0 = None` gives a toroidal field alone; with both, they must have the same equatorial symmetry (a poloidal field of
degree $n$ has $(-1)^n$, a toroidal one $(-1)^{n+1}$): the dipole goes with a degree-2 toroidal field, not with a degree-1 one.
A toroidal field that does not vanish at a wall (`'Malkus'`) carries a surface current there; the linear problem is
unaffected (the walls' conditions on $\mathbf{b}$ do not involve $\mathbf{B}_0$, and $\mathbf{u}\times\mathbf{B}_0 = 0$ on a
no-slip wall). Inside a conducting inner core $\mathbf{B}_0$ does not enter: there is no flow there, and
$\mathbf{u}\times\mathbf{B}_0 = 0$ at the no-slip inner boundary, so the matching conditions do not involve it either.

With an equatorially antisymmetric $\mathbf{B}_0$ the field $\mathbf{b}$ has the symmetry opposite to $\mathbf{u}$
(`ut.bsymm = symm * symmB0`): $f$ takes the degrees of the toroidal flow and $g$ those of the poloidal flow. A degree-1
toroidal field is symmetric ($B_\phi\propto\sin\theta$), so $f$ then has the degrees of the poloidal flow; a degree-2 one is
antisymmetric, like the dipole.

## Implementation

- **Sections.** `f` (rows $r^2\,\hat{\mathbf{r}}\cdot$ of the induction equation) and `g` ($r^2\,\hat{\mathbf{r}}\cdot\nabla\times$),
  both in the $C^{(2)}$ basis with two boundary-condition rows per block, are placed after the entropy section:
  `ut.offset = {'u': 0, 'v': n, 'h': 2n, 'f': (2+thermal) n, 'g': (3+thermal) n}` and
  `sizmat = 2n + n*thermal + 2n*magnetic`. Non-magnetic matrices are unchanged.
- **Labels.** `decode_label` gives sections `f` and `g` the row power $2 - X$ and no $\rho_0$. The profile codes are
  `bpo` ($c_\mathrm{norm} h^{(k)}$) and `eta` ($\eta$). The Lorentz labels carry `ohr1` (one power of $\rho_0$ less), so the
  momentum rows hold $r^p$ times the force $\rho_0 \mathbf{F}_L$. `submatrices.py` lists each label next to its
  `conductive_IC` name.
- **Operators** (`operators.lorentz`, `induction`, `magnetic_diffusion`, `b`): the `conductive_IC` operators divided by
  $L^3$ of the row, as every operator of this branch, with Hendrik and MagnetD for $(\Omega\tau\,\mathrm{Le})^2$ and
  $\Omega\tau\,\mathrm{Em}$. The sign convention is the branch's: A holds the right-hand side, B is $-\mathbf{b}$ in the
  induction rows (as $-$inertia in the momentum rows).
- **Boundary conditions** (`assemble.bc_b_spherical`): insulating walls, $(l+1)f + rf' = 0$ and $g = 0$ at the outer wall,
  $lf - r f' = 0$ and $g = 0$ at the inner one.
- **Variable density** (step 3). With $u = \rho_0^{-1}\nabla\times\nabla\times(\rho_0 P\,\mathbf{r}) + \nabla\times(T\,\mathbf{r})$ the
  consoidal part of the flow is $P' + P/r + \lambda_1 P$, $\lambda_1 = (\ln\rho_0)'$, $\lambda_2 = (\ln\rho_0)''$, and
  $\nabla\cdot\mathbf{u} = -\lambda_1 u_r$. Derived from the definitions with `tests/magnetic/derive.py` (sympy, exact
  projection on the harmonics), for $\mathbf{B}_0$ of degree 1:
    - rows `f`, columns P: every $d/dr$ on $P$ becomes $d/dr + \lambda_1$ (label `f1bpo0lho1_D0`);
    - rows `g`, columns P, diagonal in $l$ only: $2im[\,r h\lambda_1 P' + r h\lambda_2 P + (1 - L/2)\,r h'\lambda_1 P - (L/2)\,h\lambda_1 P\,]$
      (in the scale of `conductive_IC`'s rows; zero for m = 0), from $-\mathbf{B}_0\nabla\cdot\mathbf{u}$ and the $\lambda_1 P$ above;
    - rows `u`: $\rho_0\hat{\mathbf{r}}\cdot\nabla\times\nabla\times(\rho_0^{-1}\mathbf{G}) = \hat{\mathbf{r}}\cdot\nabla\times\nabla\times\mathbf{G} -
      (\lambda_1/r)\nabla_1\cdot\mathbf{G}_H$, $\mathbf{G}$ the Lorentz force per volume: the extra
      $\lambda_1 r^4(\nabla_1\cdot\mathbf{G}_H)_l$, i.e. $-im\lambda_1[(L+2)r^2hg + Lr^3h'g + 2r^3hg']$ for the toroidal field of the same
      degree and $(y/2)\lambda_1 r^3[2a\,h\,D_{l_c}f + L_c b\,(D_1 h) f]$ for the poloidal field of degree $l_c = l\pm1$
      ($y = \langle Y_l\,\vert\,\cos\theta\,Y_{l_c}\rangle$, $a = 2 - L_c - L$, $b = L_c - 2 - L$, $D_l f = f'' + 2f'/r - l(l+1)f/r^2$;
      `operators.lorentz_rho`);
    - rows `v`, the T columns, the diffusion, the walls and the inner core: no $\rho_0$ terms.
  The new labels pair `bpo` with `lho` (two profiles, plus `ohr1` in the rows `u`); they vanish exactly for `'Boussinesq'`,
  whose matrices are those of steps 1 and 2 bitwise. `tol_B0` does not cut these products (they converge slowly near the
  surface). `submatrices.py` now counts odd profiles in both slots of a label for the parity at ricb = 0 (bitwise the same for
  every existing label).
- **Toroidal field** (step 4, `operators.induction_tor`, `lorentz_tor`; profile code `bto` = $c_\mathrm{tor} k^{(j)}$). The rows were
  derived with `tests/magnetic/derive.py` (any $\rho_0$) and fitted in $l$: every coefficient is $im$ (same degree) or the
  Coriolis factors $C_- = \sqrt{l^2-m^2}/(2l-1)$, $C_+ = \sqrt{(l+1)^2-m^2}/(2l+3)$ (columns $l\mp1$) times a polynomial in
  $l$ (independent of $m$). Couplings: rows f from P at the same degree; rows g from P at $l\pm1$ (with $\lambda_1$ terms) and
  from T at $l$; rows u from F at $l$ (with $\lambda_1$) and G at $l\pm1$ (with $\lambda_1$); rows v from F at $l\pm1$ and G
  at $l$. Degree 2 (`induction_tor2`, `lorentz_tor2`): the quadrupole angular factors (`operators.quad_factors`, the same-degree
  terms with $l(l+1) - 3m^2$) times polynomials in $l$, couplings up to $\Delta l = \pm2$: rows f from P at $l\pm1$; g from P at
  $l$, $l\pm2$ and T at $l\pm1$; u from F at $l\pm1$ and G at $l$, $l\pm2$; v from F at $l$, $l\pm2$ and G at $l\pm1$.
  `operators.lorentz` and `induction` add the poloidal and the toroidal parts (`assemble.py` loops over $\Delta l = -2..2$
  and skips absent couplings). Column offsets: `operators.moff`.
- **Poloidal field of degree 2** (step 5, `operators.induction_pol2`, `lorentz_pol2`, for `'Luo_S2'` and `'FDM'` with
  `B0_l = 2`). Derived with `derive.py` ($b_0$ = ('pol', 2), any $\rho_0$) and fitted in $l$, $m$ as for the toroidal field
  of degree 2: `quad_factors` times polynomials in $l$ (the same-degree terms carry $l(l+1) - 3m^2$), kept in two tables
  (row, column, $\Delta l$) of the label expressions. Couplings: rows f from P at $l$, $l\pm2$ and T at $l\pm1$; g from P at
  $l\pm1$ and T at $l$, $l\pm2$; u from F at $l$, $l\pm2$ and G at $l\pm1$; v from F at $l\pm1$ and G at $l$, $l\pm2$. Two labels
  are added to those of the dipole (`u2bpo2ohr1_D1`, `u2bpo2ohr1_D0`). spin_doctor takes its Lorentz and induction powers
  from `magnetic_powers_grid` for these fields (`utils4pp.use_grid`).
- **Full sphere** (step 5, `ricb = 0`). As for the flow, each block holds the Chebyshev polynomials of one parity on
  $[-1, 1]$ (`ut.remroco`): $f$ has the vector parity `vF = symmB0 * vP`, $g$ the opposite; $h$ has the parity $(-1)^l$ of
  its degree and $k$ that of $(-1)^n$ (`utils.odd_profiles`, both slots of a label). One boundary-condition row per block,
  the insulating condition at the outer wall (`assemble.bc_b_spherical`). The fields are regular at the centre: FDM
  $j_l(\beta r)$ (analytic derivatives, `background_field.jl_k`), the polynomial fields, 'wall' $= r^n(1 - r^2)$, 'Malkus'
  for $n = 1$ only (`k = r` is not regular for $n = 2$), 'user' as $f(\lvert r\rvert )\,\mathrm{sign}(r)^n$. `cnorm` and `cnorm_tor` are
  computed as with an inner core; spin_doctor's inner-wall terms are skipped.
- **Inviscid runs** (step 5, `ViscosD = 0`, full sphere only). The Lorentz labels have $D^3$ (rows u) and $D^2$ (rows v), so
  the rows are in $C^{(3)}$ (u) and $C^{(2)}$ (v), with one boundary-condition row ($u_r = 0$) in u and none in v; without
  a field they stay in $C^{(2)}$ and $C^{(1)}$. `decode_label` gives the u and v labels the row powers of the viscous rows.
  `conductive_IC` keeps $C^{(2)}$ and $C^{(1)}$ and so converts these terms with the wrong matrices; its torsional test is
  barely affected (below).
- **Solver and output.** `solve.py` writes `real_magnetic.field`, `imag_magnetic.field` and normalises the eigenvectors to
  $\mathrm{KE} + \mathrm{Hendrik}\int\tfrac12\lvert \mathbf{b}\rvert ^2dV = 1$ (over the shell, as `conductive_IC`).

## With a differential rotation (2026-10-30)

With `diff_rot = 1` the background flow $\mathbf{U}_0 = G\,\Delta\Omega\,r\sin\theta\,\hat{\boldsymbol\phi}$ also shears the field: the induction
equation gains

$$
\nabla\times(\mathbf{U}_0\times\mathbf{b}) = G\left[-im\,\Delta\Omega\,\mathbf{b} + \sin\theta\,(r\,\partial_r\Delta\Omega\,b_r + \partial_\theta\Delta\Omega\,b_\theta)\,\hat{\boldsymbol\phi}\right]
$$

(the $\Omega$-effect, and the Doppler shift of the field). It is built for any $\Delta\Omega = \sum_n f_n P_n$ with the recurrence
algebra of the momentum terms (`diff_rot.induction_block`; [Derivation tools](derivations.md), part B): F→f and G→g couple at
even $\Delta l$ up to $n$, F→g (a poloidal field sheared into a toroidal one) at odd $\Delta l$ up to $n + 1$. The Lorentz force
needs nothing new: it does not involve $\mathbf{U}_0$.

- **Conducting inner core.** It must co-rotate rigidly, $\Delta\Omega(r_i, \theta) = c$ ($f_n(r_i) = 0$ for $n > 0$, else the run
  stops: a latitude-dependent rotation would shear a solid core). Its field then gains $-imGc\,\mathbf{b}$; the matching
  conditions are unchanged, because only $b_r$, which is continuous, enters the tangential part of $\mathbf{U}_0\times\mathbf{b}$.
- **Steady background (Ferraro).** $\nabla\times(\mathbf{U}_0\times\mathbf{B}_0) = r\sin\theta\,(\mathbf{B}_0\cdot\nabla\Delta\Omega)\,\hat{\boldsymbol\phi}$ must
  vanish for the background to be steady: always for a toroidal $\mathbf{B}_0$, for a poloidal one only if $\Delta\Omega$ is
  constant along its field lines. The linear problem does not contain this term. `background.ferraro_mismatch` reports
  $\max\lvert \mathbf{B}_0\cdot\nabla\Delta\Omega\rvert \,r/(\max\lvert \mathbf{B}_0\rvert \max\lvert r\nabla\Delta\Omega\rvert )$ at the start of every run; above $10^{-7}$
  `background.FERRARO_POLICY` decides: `'warn'` (default: a warning, the background is assumed to be maintained) or
  `'refuse'`. Consistent backgrounds: a toroidal field, a `'confined'` field where $\Delta\Omega$ is uniform, `B0 = 'ferraro'`, or
  `diff_rot_type = 'ferraro'`.
- **Energy budgets.** spin_doctor's induction work includes $\mathbf{U}_0\times\mathbf{b}$ (on the $(r, \theta)$ grid). Because
  $\mathbf{U}_0\times\mathbf{b}$ does not vanish at a wall where $\Delta\Omega \ne 0$, it carries magnetic energy through the wall into the
  field outside: resid𝐛 (the fluid) uses the induction work with its wall terms, resid𝐄 (the full budget, which counts the
  energy outside) the volume work alone (`utils4pp.INDU_WALLS`), plus the power of the rotation's force on the flow.

## Couplings for any background (2026-10-30)

`bin/mag_algebra.py` builds the couplings of the Lorentz force and of the induction for
$\mathbf{B}_0 = \sum_n c_\mathrm{norm}\nabla\times\nabla\times(h_n Y_n^0\mathbf{r}) + \sum_n c_\mathrm{tor}\nabla\times(k_n Y_n^0\mathbf{r})$, any degrees,
from the recurrence algebra that the differential rotation uses, instead of the closed forms of `operators.py` (which exist for
degrees 1 and 2 only). The background vectors are polynomials in $x = \cos\theta$ times $\hat{\mathbf{r}}$, $\sin\theta\,\hat{\boldsymbol\theta}$ or
$\sin\theta\,\hat{\boldsymbol\phi}$; the products with the perturbation are sparse matrices on the degrees, times radial monomials; the rows
take one radial derivative (product rule); each final monomial becomes a label with the profile codes `hp` + degree
($c_\mathrm{norm}h_n$) and `kt` + degree ($c_\mathrm{tor}k_n$). [Derivation tools](derivations.md), part A, gives the method.
`mag_couplings = 'algebra'` uses it for every type; `'closed'` (default) keeps the closed forms, which `'general'` and `'ferraro'`
cannot use. For the types that have closed forms both give the same eigenvalues to round-off.

## Conducting inner core

`innercore = 'conducting, Chebys'` (eigenvalue problems, no-slip inner boundary `bci = 1`) adds the field inside the inner
core to the unknowns, as `conductive_IC` does:

- **Unknowns.** Sections `fic`, `gic` after `f`, `g` (`ut.offset`), with the degrees of `f`, `g` (up to `lmax`;
  `conductive_IC`'s `lmax_cic` is not ported, as its ICB matching couples the degrees one by one). Each block is the
  Chebyshev series on $[-r_i, r_i]$ of one parity (that of the field at the centre), `N_cic`/2
  coefficients per degree (`ut.Nic`, default `N_cic` = 64); `ut.nic` per section.
- **Equation.** $\partial\mathbf{b}/\partial t = \mathrm{MagnetD}\,\eta_i\nabla^2\mathbf{b}$, $\eta_i = 1/(\mu_{i2o}\sigma_{i2o})$
  (`mu_i2o`, `sigma_i2o`: permeability and conductivity relative to the fluid), rows $r^2\hat{\mathbf{r}}\cdot$ and
  $r^2\hat{\mathbf{r}}\cdot\nabla\times$ divided by $L^3$ as in the fluid, in the $C^{(2)}$ basis. Its matrices are built
  apart from the labels (`submatrices.py`: `r0_D0f_ic`, `r2_D0f_ic`, `r1_D1f_ic`, `r2_D2f_ic` and the same for g), with
  `ut.Dlam` and `bc_variables.Tk` on the inner core's domain (their optional domain arguments).
- **Matching at the ICB** (`assemble.bc_b_cic`), one row in each section, the inner one of `f`, `g` and the only one of
  `fic`, `gic`: $f_o = f_i$ ($[b_r]$), $(g + rg')_o = \eta_i (g + rg')_i$ (tangential $\mathbf{E}$),
  $\mu_{i2o}(f + rf')_o = (f + rf')_i$ and $\mu_{i2o}\,g_o = g_i$ (tangential $\mathbf{H}$). The continuity of
  $\eta\nabla^2_l f$ is not imposed: with $b_r$ continuous it follows from the radial induction equation on both sides,
  since $\mathbf{u} = 0$ at the no-slip ICB.
- **Output.** `real_magnetic_ic.field`, `imag_magnetic_ic.field`. The eigenvectors are normalised with the fluid's
  energies only (the inner-core field is scaled along), as in `conductive_IC`.

## Post-processing

`spin_doctor` adds the power of the Lorentz force to residσ and residω, prints resid𝐛 and Mag/Kin, and writes
`magnetic.dat`:

| column | name | |
| --- | --- | --- |
| 1 | ME | $\int\tfrac12\lvert \mathbf{b}\rvert ^2 dV$ (without Hendrik, as `conductive_IC`; Mag/Kin is ME/KE) |
| 2 | Mdfs | $\mathrm{MagnetD}\int\mathbf{b}\cdot\nabla^2\mathbf{b}\,dV$ |
| 3 | Indu | $\int\mathbf{b}\cdot\nabla\times(\mathbf{u}\times\mathbf{B}_0)\,dV$ |
| 4 | residb | resid𝐛 $= \lvert 2\sigma\,\mathrm{ME} - \mathrm{Indu} - \mathrm{Mdfs}\rvert /\max$ |
| 5, 6 | Wlor, Wlor_im | $\mathrm{Hendrik}\cdot 2\int\mathbf{u}^*\cdot\mathbf{F}\,dV$, real and imaginary parts (in residσ, residω) |
| 7 | MEic | $\tfrac12\int\lvert \mathbf{b}\rvert ^2/\mu_{i2o}\,dV$ in the inner core (its field, or the potential field below an insulating one) |
| 8 | MEext | $\tfrac12\int\lvert \mathbf{b}\rvert ^2dV$ of the potential field outside the outer wall |
| 9 | Ohm_f | $\mathrm{MagnetD}\int\lvert \nabla\times\mathbf{b}\rvert ^2dV$ over the fluid |
| 10 | Ohm_ic | $\mathrm{MagnetD}\,\eta_i\int\lvert \nabla\times\mathbf{b}\rvert ^2/\mu_{i2o}\,dV$ over a conducting inner core |
| 11 | residE | resid𝐄, the full energy budget (nan without no-slip walls) |

resid𝐄 (`conductive_IC`'s, `utils4pp.em_budget`) closes
$2\sigma[\mathrm{KE} + \mathrm{Hendrik}(\mathrm{ME} + \mathrm{MEic} + \mathrm{MEext})] = \mathrm{Dkin} + \mathrm{Wthm} + \mathrm{Wlor}
+ \mathrm{Hendrik}(\mathrm{Indu} - \mathrm{Ohm\_f} - \mathrm{Ohm\_ic})$: unlike resid𝐛 it sees the wall and ICB conditions.
The $1/\mu_{i2o}$ of the inner core's energy and Ohmic loss is not in `conductive_IC`'s `em_budget` (there resid𝐄 is
$10^{-2}$ at `mu_i2o` = 2).

`params.dat` gains thirteen columns at the end: `magnetic`, `B0` (index in `background_field.B0_TYPES`, −1 without a field),
`B0_l`, `cnorm` (as a number), `Hendrik`, `MagnetD`, `innercore` (index in `background.INNER_CORES`), `N_cic`, `mu_i2o`,
`sigma_i2o`, and (step 4) `B0_tor` (index in `background_field.B0_TOR_TYPES`, −1 without), `B0_tor_l`, `cnorm_tor`. spin_doctor's Lorentz force (`utils4pp.lorentz4pp`) acts per volume and its power $\int\mathbf{u}^*\cdot\mathbf{G}\,dV$
has no $\rho_0$; its induction term (`induction4pp`) takes the components of $\mathbf{u}$ with their $\lambda_1$ part, and holds
for the non-solenoidal $\mathbf{u}$ of a variable density (resid𝐛 closes to $10^{-10}$ with 'Wu 2005', below). With a
toroidal $\mathbf{B}_0$ (or `KORE_MAG_GRID=1` in the environment) both come instead from `utils4pp.magnetic_powers_grid`,
computed on an $(r,\theta)$ grid from the vector fields $\mathbf{u}$, $\mathbf{b}$, $\nabla\times\mathbf{b}$, $\mathbf{B}_0$,
$\nabla\times\mathbf{B}_0$ (with the wall terms of $\int\mathbf{b}^*\cdot\nabla\times(\mathbf{u}\times\mathbf{B}_0)\,dV$); for a poloidal
field it agrees with the spectral formulas to the 10 printed digits.

## Checks (2026-10-28)

Spin-over geometry of the `conductive_IC` test: ricb 0.35, Ek 1e-3, no-slip walls (`bci = bco = 1`), N 68, lmax 64
(63 for m = 0), 2 ranks; $\Lambda = 1$, Pm = 1 (Le $= \sqrt{10^{-3}}$, Em $= 10^{-3}$), `cnorm = 'mag_energy'`, insulating
walls; `conductive_IC` at `39487bbf` with the same parameters. All five eigenvalues of each run:

| case | target | largest relative difference |
| --- | --- | --- |
| `'axial'`, m 1, symm −1 | 1.0i | 8e-15 |
| `'FDM'`, m 1, symm −1 | 1.0i | 7e-15 |
| `'axial'`, m 0, symm 1 | 0.3i | 1e-13 |
| `'FDM'`, m 0, symm 1 | 0.3i | 9e-14 |

The least damped m = 1 mode: λ = −0.104550975908 + 1.009269490234i (`'axial'`), −0.104198820746 + 1.008910499350i
(`'FDM'`). residσ and resid𝐛 agree with `conductive_IC`'s resid𝐮 and resid𝐛 mode by mode (1e-15 to 1e-10). Without the field
(`magnetic = 0`, structure `'Boussinesq'`, no-slip) the hydrodynamic spin-over is λ = −0.1019947097 + 1.0057336901i, the
`conductive_IC` reference. Non-magnetic matrices are bitwise unchanged (`tests/diff_rot/regress.sh`).

### Conducting inner core (2026-10-28)

FDM ($l = 1$), `cnorm = 'Schmitt2012'`, ricb 0.35, Ek $10^{-5}$, $\Lambda = 10$, Pm 0.1, no-slip walls, insulating mantle,
conducting inner core, 4 ranks, the same parameters in `conductive_IC` (`39487bbf`). The m = 0 target is the most
torsional of the modes there at this resolution (ω 0.0747, σ −0.0233, 𝒯/𝒫 18, Geo 0.90).

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
$\mu_{i2o} \ne 1$ apart, see above). N 80 is not enough at Ek $10^{-5}$: residuals $10^{-6}$ to $10^{-3}$, $10^{-14}$ to
$10^{-6}$ at N 128. `N_cic` 48, 64 and 96 give the same eigenvalues to $2\cdot10^{-14}$. At Ek $10^{-3}$ (the geometry above,
$\Lambda$ = Pm = 1) 'axial' m 1 and 'FDM' m 0 agree to $3\cdot10^{-13}$, with all residuals below $10^{-10}$.

### Variable density (2026-10-28)

Everything in `tests/magnetic/` (`run_all.sh`):

- **Derivation.** The rows derived from the definitions reproduce `conductive_IC`'s 44 uniform-density blocks exactly
  (factor −1, the A sign; l 3–6, m 0–3), and the $(\ln\rho_0)'$ terms equal the closed forms above to $\le 2\cdot10^{-14}$;
  the rows `v` and the T columns have none. The closed forms are those of the planning notes: none had to be corrected.
- **Uniform density unchanged.** The 4 insulating and 9 conducting-core cases of the checks above give A and B bitwise
  equal to the code before this step.
- **Operator identities** (`ru_insane.py`, B0 'axial', no-slip): $\int\rho_0\mathbf{u}^*\cdot\mathbf{F}_L\,dV =
  -\mathrm{Hendrik}\,\overline{\int\mathbf{b}^*\cdot\nabla\times(\mathbf{u}\times\mathbf{B}_0)\,dV}$ to $5\cdot10^{-15}$ ('Boussinesq') and
  $1\cdot10^{-10}$ ('Wu 2005', r_out 0.99, N 96; $7\cdot10^{-13}$ at N 160), m 0, 1, 2; switching off the g-row or the u-row
  terms gives $2\cdot10^{-3}$ and 0.2. b block Hermitian positive and diffusion Hermitian negative to $10^{-14}$.
- **Row checks** (`row_check.py`, 'Wu 2005', N 192): every magnetic row block against the derivation, unweighted, to
  $10^{-14}$ ('axial' m 1, 'FDM' m 0 with `tol_B0` = 1e-15; $8\cdot10^{-9}$ with the default `tol_B0` = 1e-8, the cut of the B0
  series). At N 96 the blocks with $\lambda_1$ are good to $10^{-7}$ only: the series of $(\ln\rho_0)'$ converge slowly, its pole
  sits at $r = 1/r_\mathrm{out}$.
- **Energy budgets** ('Wu 2005', r_out 0.99, ricb 0.35, FDM, conducting inner core, no-slip, $\Lambda$ = Pm = 1). At Ek $10^{-3}$, N 192:
  residσ, residω, resid𝐛, resid𝐄 $\le 6\cdot10^{-10}$ for m 0 and m 1 (N 68: $10^{-3}$, N 128: $10^{-7}$). At Ek $10^{-4}$ the least
  damped m = 1 mode is λ = −0.005739718 + 1.003574886i (N 256; N 192 differs by $3\cdot10^{-9}$, lmax 96 instead of 64 by
  $10^{-8}$), the m = 0 mode at 0.0442i is λ = −0.017420795 + 0.044242829i, with all residuals $\le 3\cdot10^{-9}$ at N 256.
  The budgets use spin_doctor's own Lorentz and induction formulas, so they check the rows independently; the f and g rows
  need no $\rho_0$ weight, although $\lambda_1 \approx -200$ at the outer wall.
- **Continuity** ('Wu 2005' with r_out = $\sqrt\epsilon$, so the stratification is $O(\epsilon)$, against 'Boussinesq'; axial, m 1,
  Ek $10^{-3}$, N 96): $(\lambda - \lambda_\mathrm{uniform})/\epsilon$ = 0.05186 − 0.00053i, 0.05188, 0.05215, 0.05498 for
  $\epsilon = 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}$: the eigenvalue moves linearly with the stratification.

### Toroidal field (2026-10-28)

- **Derivation.** The closed forms equal derive.py's rows ($b_0$ = ('tor', 1) and ('tor', 2), any $\rho_0$) for l 3–6, m 0–3
  (`derive_check.py`); every coefficient fitted from l 2–9, m 0–3 (n = 1) and l 3–9 (n = 2), exactly.
- **Earlier cases unchanged.** The 13 cases of steps 1 and 2 and three 'Wu 2005' cases of step 3 give A and B bitwise
  equal to the code before this step.
- **Row checks** ('Wu 2005', N 192): 'Malkus' m 1 to $10^{-14}$; 'wall' m 0, 2 to $2\cdot10^{-10}$, in the u-row blocks with
  $k\lambda_1$ only (they vanish at the wall, where $\lambda_1$ is large, so the relative error of the near-wall $\lambda_1$
  series shows; it does not decrease with N).
- **Energy budgets.** 'Malkus' and 'wall', 'Boussinesq' (N 68, 96) and 'Wu 2005' (N 128, 192), insulating and conducting
  inner core, m 0 and 1, Ek = Em = $10^{-3}$, $\mathrm{Le}^2 = 10^{-3}$, `cnorm_tor = 'mag_energy'`: residσ, resid𝐛, resid𝐄
  $\le 6\cdot10^{-13}$, eigenvalues the same at both N to the printed digits. Unlike a poloidal field (Variable density,
  above), the toroidal one has no radial component, hence no Hartmann layer at the walls, and 'Wu 2005' converges fast.
- **Degree 2 and combined fields.** 'Malkus' and 'wall' with `B0_tor_l = 2`, 'Boussinesq' (N 68, 96) and 'Wu 2005' (N 128, 192),
  conducting inner core, m 0 and 1: residuals $\le 10^{-11}$, eigenvalues the same at both N to the printed digits. FDM plus
  a degree-2 'wall' field ('Wu 2005'): residuals $2\cdot10^{-10}$ at N 192 (the poloidal field's Hartmann layer sets the
  resolution); row check of 'axial' plus a degree-2 'Malkus' field, m 2: $10^{-14}$. A 'user' profile `'r'` gives the
  'Malkus' eigenvalues to $10^{-12}$.
- **Malkus (1967).** With $\mathbf{B}_0 = \mathrm{Le}\,s\hat{\boldsymbol\phi}$ the induction is $im\,\mathrm{Le}\,\mathbf{u}$ and the
  Lorentz force $(im\,\mathrm{Le}^2/\lambda)(im\mathbf{u} + 2\hat{\mathbf{z}}\times\mathbf{u})$ up to a gradient, so the ideal
  problem reduces to that of the inertial waves in any container: $\omega = (\lambda_\mathrm{in}/2)[1 \pm
  \sqrt{1 + 4\mathrm{Le}^2 m(m - \lambda_\mathrm{in})/\lambda_\mathrm{in}^2}]$ (kore's Im of the eigenvalue; the spin-over,
  $\lambda_\mathrm{in} = m = 1$, does not move). For six m = 1 modes near 0.75i ('Boussinesq', ricb 0.35, Pm 1, insulating walls)
  the frequency shift is exactly quadratic in Le ($\mathrm{Le}^2$ = 1e-6, 4e-6, 1.6e-5 give the same shift/$\mathrm{Le}^2$) and
  tends to $m(m-\lambda_\mathrm{in})/\lambda_\mathrm{in}$: 2–8 % off at Ek = Em = $10^{-5}$, 0.4–1 % at $10^{-6}$ (e.g. 0.3808
  against 0.3822 for $\lambda_\mathrm{in}$ = 0.7235). `tests/magnetic/malkus_check.py`.

### Full sphere, inviscid runs and degree-2 poloidal fields (2026-10-28, step 5)

- **Hydrodynamics.** inviscid Boussinesq full sphere (Ek = 0, `'Boussinesq'`): the inertial modes of Wu (2005, Table 1, β = 0)
  to the 10 digits given (e.g. 0.6666666667, 1.4964360838, 1.2319250547 for $\lvert m\rvert  = 2$); 'Wu 2005' against its Table 2.
  No non-magnetic code changed (`regress.sh` bitwise).
- **Against `conductive_IC`** (`39487bbf`, the geometry of Checks, all eigenvalues of each run): full sphere, 'axial', 'FDM'
  ($l$ = 1, 2), 'G21 dipole', 'Luo_S1', 'Luo_S2', m 0–2 and both symmetries, $\le 5\cdot10^{-13}$; with an inner core
  (insulating or conducting, `B0_ic = 'potential'`): 'G21 dipole', 'Luo_S1', 'Luo_S2', 'FDM' $l = 2$ (m 0, 3), $\le 6\cdot10^{-13}$.
- **Derivation.** `derive_check.py` also checks the tables of `induction_pol2`, `lorentz_pol2`, read from `operators.py`,
  against derive.py's rows (l 3–6, m 0–3), to round-off.
- **Row checks** in a full sphere (`row_check.py` now handles the parity-reduced bases and $\Delta l = \pm2$): 'Boussinesq' to
  $10^{-14}$; 'Wu 2005' needs N 288 (the pole of $(\ln\rho_0)'$ at $r = 1/r_\mathrm{out}$ is nearer to the domain $[-1, 1]$
  than to a shell's; $3\cdot10^{-7}$ at N 192).
- **Energy budgets**, 'Wu 2005', Ek = Em = $10^{-3}$: 'Luo_S2' full sphere m 0, 1 to $3\cdot10^{-11}$ and $10^{-9}$ at N 192
  ($10^{-7}$, $3\cdot10^{-6}$ at N 128); with an insulating or conducting inner core $10^{-10}$; FDM $l = 2$ full sphere
  (`'Schmitt2012'`) $5\cdot10^{-4}$, $6\cdot10^{-6}$, $3\cdot10^{-8}$ at N 128, 192, 256 (its Hartmann layer); 'Boussinesq' to $10^{-14}$.
- **Malkus (1967), ideal.** inviscid Boussinesq full sphere, $\mathbf{B}_0 = \mathrm{Le}\,s\hat{\boldsymbol\phi}$, Em $10^{-9}$,
  m 2: for ten inertial modes $\lambda_\mathrm{in}$ of both symmetries and Le 0.1, 0.3, both branches of
  $\omega = (\lambda_\mathrm{in}/2)[1 \pm \sqrt{1 + 4\mathrm{Le}^2 m(m - \lambda_\mathrm{in})/\lambda_\mathrm{in}^2}]$ to
  $10^{-5}$–$7\cdot10^{-3}$ relative at N 64 (the fast branch to $10^{-4}$ at N 96). The error does not change with Em
  ($10^{-8}$ to $10^{-10}$) or with Le: it is that of the magnetic boundary layer the insulating wall imposes (Malkus's ideal
  solution has a tangential $\mathbf{b}$ there), not resolved at these N.
- **Torsional modes of Luo & Jackson (2022)** (`conductive_IC`'s `tests/torsional_s2`: 'Luo_S2', inviscid, Lu $2\cdot10^4$,
  m 0, `timescale = 'Alfven'`). Their modes 1 and 2 are $-0.0065952461 + 1.0335959942i$ and $-0.0141833345 + 1.9095123626i$;
  this code converges to them as $N^{-3}$: mode 1 $1.3\cdot10^{-5}$ off at N 128, $4\cdot10^{-7}$ at N 320; mode 2 $2\cdot10^{-4}$
  at N 128, $7\cdot10^{-6}$ at N 384. `conductive_IC` is within $10^{-6}$ already at N 128 because it mishandles the $D^3$ and $D^2$
  Lorentz terms (above), whose effect there is only $10^{-7}$: with its bases set as here it gives this code's eigenvalues to
  $10^{-9}$. The slow convergence comes from the u rows in $C^{(3)}$, where the unresolved boundary layer of $\mathbf{b}$
  (Em $5\cdot10^{-9}$) enters through $f'''$.

### Differential rotation, confined, general and Ferraro fields (2026-10-30)

- **∇×(U0×b) against sympy** (`tests/magnetic/dr_induction_check.py`): the blocks of `diff_rot.induction_block` against
  `derive.rows_dr` for $n$ = 0–6, $m$ = 0–5, $l$ up to 25, term by term: $\le 10^{-13}$ at low $l$; at $l$ 20–25 the agreement is
  limited by derive.py's own quadrature ($10^{-10}$).
- **Rigid rotation, exact** (`rigid_check.py`): $\Delta\Omega = c$ must give $\lambda = \lambda(\text{Gaspard } 1 + c) - imc$. All eigenvalues
  agree to $\le 5\cdot10^{-14}$: hydrodynamics, 'axial', 'FDM', 'Luo_S2', 'Malkus', 'wall' of degree 2, a conducting inner core,
  a full sphere.
- **Budgets with a solar-like $\Delta\Omega = f_0 + f_2P_2 + f_4P_4$** (Ek = Em = $10^{-3}$, $\Lambda$ = 1): 'Malkus', 'axial' and
  'Luo_S2', shell and full sphere, 'Boussinesq' and 'Wu 2005': residσ, resid𝐛, resid𝐄 $10^{-13}$–$10^{-9}$.
- **The algebra against the closed forms** (`mag_couplings` 'algebra' against 'closed', same runs): eigenvalues equal to
  $\le 6\cdot10^{-13}$ for 'axial' (m 0, 1), 'FDM' (degree 2), 'Luo_S2' (shell, full sphere, 'Wu 2005'), 'FDM' with 'Wu 2005',
  'Malkus' (and 'Wu 2005'), 'wall' of degree 2, 'axial' + 'wall', a conducting inner core, 'confined', and with a differential
  rotation. **Against sympy** (`algebra_check.py`): every block for poloidal and toroidal fields of degree 1–4 (3 and 4 have no
  closed forms), several $m$, $l$ up to 12.
- **'general'**: $h_1 = r/2$ gives exactly the 'axial' eigenvalues, $k_1 = r$ exactly the 'Malkus' ones; fields of degrees 1 + 3
  (full sphere), 2 + 4 (shell), toroidal 1 + 3, and a poloidal 1 + 3 with a toroidal 2 and a differential rotation (also
  'Wu 2005'): budgets $\le 5\cdot10^{-12}$.
- **'confined'**: degree 1 and 2, shell region [0.4, 0.8] and full-sphere region [0, 0.6]: budgets $10^{-12}$–$10^{-14}$ (shell),
  $4\cdot10^{-10}$ (degree 1, N 192) and $2\cdot10^{-9}$ (degree 2, N 384) in the full sphere; with $\Delta\Omega$ uniform inside the
  region, `ferraro_mismatch` = $10^{-29}$ and the budgets close as without rotation.
- **'ferraro'**: with a cylindrical rotation and $G(w) = w$ the field is the uniform axial one, and the eigenvalues equal those
  of 'axial' to 10 digits; with $\Delta\Omega \propto s^2(1 - s^2)$ (degrees 0, 2, 4), $\psi$ is rebuilt from degrees 1–7 to
  $4\cdot10^{-15}$, `ferraro_mismatch` is $5\cdot10^{-14}$, and the budgets close to $\le 2\cdot10^{-11}$ in a shell, a full sphere and
  'Wu 2005'.
- **`diff_rot_type = 'ferraro'`**: the uniform axial field with $F = \psi$ gives the eigenvalues of the cylindrical profile to 10
  digits, and so does the round trip through `B0 = 'ferraro'` ($G = 5w$; budgets $4\cdot10^{-14}$). `ferraro_mismatch`: $3\cdot10^{-12}$
  ('FDM'), $6\cdot10^{-12}$ ('G21 dipole', 'Luo_S1'), $2\cdot10^{-12}$ (full sphere), $2\cdot10^{-11}$ ('general' of degrees 1 + 3,
  $F = \psi + \psi^2$), $2\cdot10^{-10}$ ('Luo_S2' with $F = \psi^2$), $7\cdot10^{-9}$ ('confined', cut at `diff_rot_nmax` 4); budgets
  $\le 10^{-10}$. Refused as they should be: an odd $F$ with 'Luo_S2', `B0 = 'ferraro'`, a toroidal field alone, a conducting
  inner core with $\psi$ not constant on it. With a toroidal part of degree 2: `B0 = 'ferraro'` + 'Malkus' and 'Luo_S1' + 'wall' with
  `diff_rot_type = 'ferraro'` close their budgets to $8\cdot10^{-12}$ and $10^{-9}$.
- **Unchanged without the new features**: `regress.sh` bitwise, heat_dr, ru_insane identical, and 26 magnetic cases build A and
  B bitwise equal to the code before.

## Not yet implemented

In the order planned: a magnetic diffusivity profile; also thin walls (`'TWA'`), rotdyn, a perfectly conducting inner core,
the point `'dipole'`, inviscid runs with an inner core, forced runs, a conducting inner core with a latitude-dependent
rotation at its surface, and the background term $\nabla\times(\mathbf{U}_0\times\mathbf{B}_0)$ as a forcing.
