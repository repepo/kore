# Derivation tools

kore builds its matrices from closed forms: every block of a row of degree $l$ coupled to a column of degree
$l + \Delta l$ is a sum of radial operators (Chebyshev/Gegenbauer matrices, `submatrices.py`) times coefficients in $l$
and $m$ (`operators.py`, `diff_rot.py`). Two kinds of background break the spherical symmetry and need couplings
between degrees: an imposed **magnetic field** $\mathbf{B}_0$ ([Magnetic field](magnetic.md)) and a **differential
rotation** $\Delta\Omega(r, \theta)$ ([Differential rotation](differential_rotation_details.md)). This page shows how
their coefficients were obtained and checked, with commands you can run.

| background | coefficients in kore | how they were obtained | how they are checked |
| --- | --- | --- | --- |
| magnetic field | closed forms in `bin/operators.py` (`lorentz*`, `induction*`) | derived from the definitions with sympy (`tests/magnetic/derive.py`), fitted in $l$, $m$ (`tools/derive/fit_rows.py`), turned into code (`tools/derive/gen_table.py`) | `tests/magnetic/derive_check.py`, `row_check.py`, `budget_check.py`, `ru_insane.py` |
| differential rotation | computed at run time by an exact recurrence algebra (`bin/angular.py`, `bin/diff_rot.py`) | no fitting: the algebra is exact for any degree of $\Delta\Omega$ | `tests/diff_rot/angular_check.py`, `gen_check.py` (closed forms of `diff_rot_coefficients.py`), `quad_reference.py`, `tests/heat_dr/tw_sympy.py` |

**Prerequisites.** Python 3 with numpy and scipy (kore's environment, e.g. `source ~/z2/bin/activate`), and sympy for the
magnetic derivations (`pip install sympy`). The differential-rotation tools need no sympy, except `tw_sympy.py`.
Commands are given from the repository root unless a run directory is named. The outputs below are from 2026-10-29.

## 1. Magnetic field

### 1.1 What is derived

With $\mathbf{u} = \rho_0^{-1}\nabla\times\nabla\times(\rho_0 P Y_l\mathbf{r}) + \nabla\times(T Y_l\mathbf{r})$,
$\mathbf{b} = \nabla\times\nabla\times(F Y_l\mathbf{r}) + \nabla\times(G Y_l\mathbf{r})$ and a background field of degree $n$,
poloidal $\mathbf{B}_0 = \nabla\times\nabla\times(h Y_n^0\mathbf{r})$ or toroidal $\mathbf{B}_0 = \nabla\times(k Y_n^0\mathbf{r})$,
kore needs four row projections ($Y_l$ are kore's Schmidt harmonics):

| row | equation | projection |
| --- | --- | --- |
| u | momentum, poloidal | $\rho_0\hat{\mathbf{r}}\cdot\nabla\times\nabla\times(\rho_0^{-1}\mathbf{G})$, $\mathbf{G} = (\nabla\times\mathbf{b})\times\mathbf{B}_0 + (\nabla\times\mathbf{B}_0)\times\mathbf{b}$ |
| v | momentum, toroidal | $\rho_0\hat{\mathbf{r}}\cdot\nabla\times(\rho_0^{-1}\mathbf{G})$ |
| f | induction, poloidal | $\hat{\mathbf{r}}\cdot\nabla\times(\mathbf{u}\times\mathbf{B}_0)$ |
| g | induction, toroidal | $\hat{\mathbf{r}}\cdot\nabla\times\nabla\times(\mathbf{u}\times\mathbf{B}_0)$ |

`tests/magnetic/derive.py` builds these vectors with sympy (any $\rho_0$, through $\ln\rho_0$), projects them on $Y_{l'}$
by Gauss–Legendre quadrature in $\theta$ (exact for these polynomials), and returns each projection as a sum of
monomials: a coefficient times $r^p$ times derivatives of the unknown (`P0`, `P1` = $P'$, ...), of the background (`h0`,
`h1`, ... or `k0`, ...) and of $\ln\rho_0$ (`lr1` = $\lambda_1 = (\ln\rho_0)'$, `lr2`, ...). Kore's operator is that row
times $-r^5/L'^3$ (u) or $-r^3/L'^3$ (v, f, g), times Hendrik for the Lorentz force ($L' = l'(l'+1)$ of the row).

### 1.2 Derive one block

Lorentz force of the dipole ($n = 1$, the default `--b0 pol1`) in the u row of degree 3, column $F$ of degree 2, $m = 1$
(`derive.py ROW COL LCOL LROW M`; ~20 s):

```
$ cd tests/magnetic && python3 derive.py u F 2 3 1
-271.529003975633 +0.000000000000i   r^-5  F0 h0
-81.458701192690 +0.000000000000i   r^-4  F0 h0 lr1
+54.305800795127 +0.000000000000i   r^-4  F0 h1
+99.560634791066 +0.000000000000i   r^-4  F1 h0
...
-9.050966799188 +0.000000000000i   r^-2  F3 h0
(14 terms)
```

Read the last line as $-9.0510\,r^{-2} h F'''$. In `operators.lorentz_pol` (u row, `bpol`, $\Delta l = -1$) the same term
is $C\cdot 2\,\mathtt{u2bpo0ohr1\_D3}$ with $C = \sqrt{l^2 - m^2}(l^2-1)/(2l-1) = 4.5255$ ($l = 3$, $m = 1$): the label
`u2bpo0ohr1_D3` is $r^{5-2} h F'''$, and $-r^5\times(-9.0510\,r^{-2}) = +2C\,r^3$. Only the terms with $\ln\rho_0$
(the anelastic ones, zero for structure `'Boussinesq'`):

```
$ python3 derive.py u F 2 3 1 --rho
-81.458701192690 +0.000000000000i   r^-4  F0 h0 lr1
+27.152900397563 +0.000000000000i   r^-3  F0 h1 lr1
+18.101933598376 +0.000000000000i   r^-3  F1 h0 lr1
+13.576450198782 +0.000000000000i   r^-2  F0 h2 lr1
+9.050966799188 +0.000000000000i   r^-2  F2 h0 lr1
(5 terms)
```

`--b0` chooses another field: `pol2` (degree-2 poloidal, `'Luo_S2'`, FDM with `B0_l = 2`), `tor1`, `tor2`. For example the
induction of the quadrupole in the f row of degree 4 from $P$ of degree 2 (~2 s):

```
$ python3 derive.py f P 2 4 1 --b0 pol2
+28.168588671694 +0.000000000000i   r^-2  P0 h0 lr1
-28.168588671694 +0.000000000000i   r^-2  P0 h1
+28.168588671694 +0.000000000000i   r^-2  P1 h0
(3 terms)
```

### 1.3 Fit closed forms in l and m

A coefficient of one monomial is, as a function of $l$ and $m$, an angular factor $A(l, m, \Delta l)$ of the field's
degree times a polynomial $a(l) + b(l)\,m^2$:

| | $\Delta l = 0$ | $\Delta l = \mp1$ | $\Delta l = \mp2$ |
| --- | --- | --- | --- |
| $n = 1$ | $im$ | $\sqrt{l^2-m^2}/(2l-1)$, $\sqrt{(l+1)^2-m^2}/(2l+3)$ | — |
| $n = 2$ | $1/((2l-1)(2l+3))$ | $im\sqrt{l^2-m^2}/(2l-1)$, $im\sqrt{(l+1)^2-m^2}/(2l+3)$ | $\frac{\sqrt{(l^2-m^2)((l-1)^2-m^2)}}{(2l-1)(2l-3)}$, $\frac{\sqrt{((l+1)^2-m^2)((l+2)^2-m^2)}}{(2l+3)(2l+5)}$ |

(`operators.quad_factors` for $n = 2$). `tools/derive/fit_rows.py KIND N` derives every block for a range of $l$, $m$ in
parallel, divides by $A$ and fits the polynomial of lowest degree that reproduces all points exactly (rational
coefficients, verified on every point). The toroidal $n = 1$ field, rows f and g only (25 s on 30 cores; the rows are
cached in `fit_tor1.pkl`, so a second run is instant):

```
$ python3 tools/derive/fit_rows.py tor 1 --rows f,g --l 2:8 --jobs 30 | tee tor1_fg.log
f P dl+0  P0*k0 r^-2             [a + b m^2] = l*(l + 1)
g P dl-1  P0*k0 r^-3             [a + b m^2] = l*(l - 1)*(l + 1)
g P dl-1  P0*k0*lr1 r^-2         [a + b m^2] = l*(l - 1)*(l + 1)
g P dl-1  P0*k1 r^-2             [a + b m^2] = -l*(l - 1)*(l + 1)
g P dl+1  P0*k0 r^-3             [a + b m^2] = -l*(l + 1)*(l + 2)
g P dl+1  P0*k0*lr1 r^-2         [a + b m^2] = -l*(l + 1)*(l + 2)
g P dl+1  P0*k1 r^-2             [a + b m^2] = l*(l + 1)*(l + 2)
g T dl+0  T0*k0 r^-2             [a + b m^2] = l*(l + 1)
```

A line ending in `None` means the polynomial needs more points than the $l$ range gives: widen it. For the degree-2
poloidal field, `python3 tools/derive/fit_rows.py pol 2 --jobs 30` (l 3–9, ~15 min) gives all blocks except four of the
u row at $\Delta l = 0$, whose $a(l)$ has degree 6; `--rows u --l 3:14` gives those too. These two runs reproduce every
coefficient of `operators.induction_pol2` and `lorentz_pol2` (step 5 of the port).

### 1.4 Turn the fit into code

`tools/derive/gen_table.py` writes the table entries in kore's labels and lists the labels `submatrices.py` must
build, with the vector parity of their column:

```
$ python3 tools/derive/gen_table.py tor1_fg.log
# induction (rows f, g)
        ('f', 'P', 0): lambda: (l*(l + 1))*f1bto0_D0,
        ('g', 'P', -1): lambda: (l*(l - 1)*(l + 1))*g2bto0_D0 + (l*(l - 1)*(l + 1))*g1bto0lho1_D0 + (-l*(l - 1)*(l + 1))*g1bto1_D0,
        ('g', 'P', 1): lambda: (-l*(l + 1)*(l + 2))*g2bto0_D0 + (-l*(l + 1)*(l + 2))*g1bto0lho1_D0 + (l*(l + 1)*(l + 2))*g1bto1_D0,
        ('g', 'T', 0): lambda: (l*(l + 1))*g1bto0_D0,
# Lorentz force (rows u, v)
# labels (submatrices.py: labl += [...], arg2 += [the parity] for each)
#  vP: ['f1bto0_D0', 'g2bto0_D0', 'g1bto0lho1_D0', 'g1bto1_D0']
#  vT: ['g1bto0_D0']
#  vF: []
#  vG: []
```

The operator returns `-A(l, m, Δl) * entry / L**3` (times `par.Hendrik` for rows u, v) with the column offset of
`operators.moff`; compare `operators.induction_tor`, where the f row is `-1j*m*L*f1bto0_D0` ($A = im$, entry $L$) and the
g row at $\Delta l = -1$ is `-Cm*l*(l-1)*(l+1)*(g2bto0_D0 + g1bto0lho1_D0 - g1bto1_D0)`. The label rule: a monomial
$X^{(d)} B^{(j)} \lambda_k\, r^p$ is `{row}{-p}{code}{j}[lho{k}]ohr1_D{d}` in rows u, v and `{row}{-(p+1)}{code}{j}[lho{k}]_D{d}`
in rows f, g, with `code` = `bpo` ($c_\mathrm{norm} h$) or `bto` ($c_\mathrm{tor} k$).

### 1.5 Check against kore

- **`tests/magnetic/derive_check.py`** (~40 min, no run directory): derive.py against conductive_IC's uniform-density
  operators, the $\lambda$ terms, and the closed forms of the toroidal fields and of `induction_pol2`, `lorentz_pol2` (read
  from the source of `bin/operators.py`), $l$ 3–6, $m$ 0–3. Its last lines:
  ```
  rows v and T columns: no (ln rho0) terms
  derive_check.py: ok
  ```
- **`tests/magnetic/row_check.py [lmax] [tol]`** (in a run directory, after `submatrices.py`): applies kore's operators
  to random fields, reads each row block in its Gegenbauer basis on an $r$ grid and compares it with derive.py evaluated
  with the run's own profiles. `tests/magnetic/mk_run.py NAME 'par.x = ..'` builds such a directory:
  ```
  $ python3 tests/magnetic/mk_run.py rc_s2 "par.structure = 'Boussinesq'; par.B0 = 'Luo_S2'; par.tol_B0 = 1e-15"
  rc_s2 ok
  $ cd rc_s2 && python3 ../tests/magnetic/row_check.py 7
  row f l= 2  column P l= 2: 1.17e-14
  row f l= 2  column P l= 4: 4.25e-15
  row f l= 2  column T l= 1: 3.34e-15
  ...
  row_check.py (Boussinesq, B0 Luo_S2, B0_tor None, m 1): worst 1.24e-14 -> ok
  ```
  (12 minutes: each block calls derive.py; the argument 7 limits the row degrees checked to 7).
  (use the absolute path of `row_check.py` if the run directory is elsewhere; with 'Wu 2005' use N 192 in a shell, N 288
  in a full sphere, for round-off).
- **Energy budgets** (`budget_check.py`, after a solve) and the identity
  $\int\rho_0\mathbf{u}^*\cdot\mathbf{F}_L\,dV = -\mathrm{Hendrik}\,\overline{\int\mathbf{b}^*\cdot\nabla\times(\mathbf{u}\times\mathbf{B}_0)dV}$
  (`bin/ru_insane.py`) check the rows through spin_doctor's independent formulas. `tests/magnetic/run_all.sh` runs everything.

### 1.6 Adding a new background field

1. **Its shape.** In `bin/background_field.py`: add the name to `B0_TYPES` (at the end: its index goes to `params.dat`),
   its $h(r)$ and derivatives to `h(r, k)` (a polynomial goes in `POLY`; mind the parity $(-1)^n$ in a full sphere), its
   degree and equatorial symmetry (`B0_l`, `symmB0`), and its normalisation in `B0_norm` if `cnorm` is not a number.
2. **Its couplings.** If its degree is 1 or 2, the existing operators apply unchanged (they depend on the degree, not on
   $h$). For another degree $n$: derive (§1.2 with `rows(..., b0=('pol', n))`), add $n$'s angular factors to
   `fit_rows.A`, fit (§1.3), generate the code (§1.4) and add a `lorentz_poln`/`induction_poln` pair with the new labels
   in `submatrices.py`; `assemble.py` loops over $\Delta l = -2..2$, so widen that loop for $n > 2$.
3. **Check.** Extend `derive_check.py` with the new tables (§5 of that file reads them from `operators.py`), run
   `row_check.py` and the budgets on a run with the new field, and compare with an independent code or an analytic
   result where you have one ([Magnetic field](magnetic.md), Checks).

## 2. Differential rotation

### 2.1 The recurrence algebra (`bin/angular.py`)

At fixed $m$, functions of $\theta$ are expanded on kore's Schmidt harmonics $Y_l$ (Condon–Shortley phase,
$\int|Y_l|^2d\Omega = 4\pi/(2l+1)$). Multiplying by a function of $x = \cos\theta$ is a matrix on the degree index (row
= target $l'$, column = source $l$), built from two exact recurrences:

$$
x Y_l = p_l Y_{l+1} + q_l Y_{l-1}\ \to\ X,\qquad (1 - x^2)\frac{dY_l}{dx} = (l+1)q_l Y_{l-1} - l p_l Y_{l+1}\ \to\ D = -\sin\theta\,\partial_\theta,
$$

$p_l = \sqrt{(l+1)^2-m^2}/(2l+1)$, $q_l = \sqrt{l^2-m^2}/(2l+1)$ (`angular.coefficients`). `angular.Algebra.get(m, lw)` holds,
on the degrees 0..lw: `X`, `D`, `Dd` $= D^\dagger = 2X - D$, `I`, `L` $= \mathrm{diag}(l(l+1))$, `Li` $= L^{-1}$ (0 at $l = 0$),
`N` $= \mathrm{diag}(4\pi/(2l+1))$, and `legendre(n)` gives $P_n(X)$ and $P_n'(X)$ by the Legendre recurrence. Vector
fields are `Field`s $(q, s, t)$ on $\mathbf{R}_l = Y_l\hat{\mathbf{r}}$, $\mathbf{S}_l = r\nabla Y_l$, $\mathbf{T}_l = r\nabla Y_l\times\hat{\mathbf{r}}$, with
`mult(F, g(X), g'(X))` (multiplication by $g(x)$), `zcross(F)` ($\hat{\mathbf{z}}\times$), `zcross_r(c)`
($\hat{\mathbf{z}}\times(c\,\hat{\mathbf{r}})$) and `sin_theta(F)` ($\sin\theta F_\theta$). Products of truncated
matrices are wrong in their last few degrees: build with lw a few degrees above what you use (kore uses lmax + n + 6).

Try it against the harmonics on a grid (`angular.ylm_grid(m, lmax, x)` returns $Y_l$, $\partial_\theta Y_l$,
$im Y_l/\sin\theta$):

```python
$ python3
>>> import sys; sys.path.insert(0, 'bin')
>>> import numpy as np, angular
>>> A = angular.Algebra.get(2, 30)                      # m = 2, degrees 0..30
>>> x = np.array([-0.7, 0.1, 0.55])
>>> Y, dY, Z = angular.ylm_grid(2, 30, x)
>>> x*Y[5], A.X.toarray()[:, 5] @ Y                     # x Y_5 = Σ_k X[k, 5] Y_k
(array([ 0.21275696, -0.01739509, -0.03535338]), array([ 0.21275696, -0.01739509, -0.03535338]))
>>> P4, dP4 = A.legendre(4)
>>> np.polynomial.legendre.Legendre.basis(4)(x)*Y[7], P4.toarray()[:, 7] @ Y
(array([-0.09618301,  0.06666419,  0.10236181]), array([-0.09618301,  0.06666419,  0.10236181]))
>>> np.nonzero(P4.toarray()[:, 7])[0]                   # P_4 couples l = 7 to l' = 3..11, every other degree
array([ 3,  5,  7,  9, 11])
>>> -np.sqrt(1 - x**2)*dY[5], A.D.toarray()[:, 5] @ Y   # D = -sinθ ∂θ
(array([ 1.1811106 , -1.58080089,  1.58869119]), array([ 1.1811106 , -1.58080089,  1.58869119]))
```

### 2.2 The couplings (`bin/diff_rot.py`)

$\Delta\Omega = \mathtt{diff\_rot\_ampl}\sum_n f_n(r)P_n(\cos\theta)$ ($n$ even, `shapes()`) acts on the flow through
$\mathbf{W} = G[\Delta\Omega(im\mathbf{u} + 2\hat{\mathbf{z}}\times\mathbf{u}) + \sin\theta(r\partial_r\Delta\Omega\,u_r + \partial_\theta\Delta\Omega\,u_\theta)\hat{\boldsymbol\phi}]$
and on the heat equation through $i m G\Delta\Omega s$ and $\mathbf{u}\cdot\nabla S_{tw}$. For each $n$:

1. `_primitives(n, m, lw)`: the 12 matrices $C[k][\mathrm{shape}]$, the projection $k$ = q, s, t of $\mathbf{W}$ for the radial
   shapes $f L P/r$, $f a$, $f T$ ($a = (rP)'/r + \lambda_1 P$) and $(r f')LP/r$, built with the algebra above;
2. `momentum_coefficients(n, m)`: their combinations into the coefficients of each radial operator of the 2-curl (u) and
   1-curl (v) rows, times $L'$ of the target — e.g. `f_2C_D0P` multiplies $f P$ in the u row — as matrices
   [target, source], stored by diagonal; `momentum_coef(name, n, l, m, Δl)` reads one;
3. `heat_coefficients(m)`, `heat_coef`: $P_n(X)$, $P_n'(X)$, $P_n'(X)D$ for the heat rows; `sigma_coefficients(r)`: the
   thermal-wind $\sigma_n$ by solving $D\sigma = (1-x^2)\partial_x S_{tw}$ in Legendre space;
4. `labels()` gives `submatrices.py` the radial operators (profile codes `zf`, `zd`, `zs` + degree for $f_n$, $rf_n'$,
   $r^2f_n''$; `tw`, `tp` for $p_0\sigma_n$, $p_0\sigma_n'$), `dl_range()` gives `assemble.py` the $\Delta l$ to place
   ($|\Delta l| \le n$, even, P→P and T→T; $\le n+1$, odd, P↔T), and `momentum_block`, `heat_block` give `operators.py`
   each block: $G\sum_n\sum$ coefficient × radial operator $/L'^3$.

Compare a few couplings with the closed forms of `tests/diff_rot/diff_rot_coefficients.py` (names `f_*` for $f_2$,
`f0_*` for $f_0$) and the primitives with the former Gauss–Legendre quadrature (`tests/diff_rot/quad_reference.py`).
`diff_rot` reads `parameters`; for a session outside a run directory give it a stand-in with `lmax`:

(numpy 2 prints the numbers as `np.float64(...)`.)

```python
$ python3
>>> import sys, types; sys.path[:0] = ['bin', 'tests/diff_rot']
>>> sys.modules['parameters'] = types.SimpleNamespace(par=types.SimpleNamespace(lmax=40))
>>> import numpy as np, diff_rot as dr, diff_rot_coefficients as dc, quad_reference as qr
>>> l, m = 6, 1
>>> [(dl, dr.momentum_coef('f_1C_D0P', 2, l, m, dl).real, dc.f_1C_D0P(l, m, dl)) for dl in (-3, -1, 1, 3)]
[(-3, 20.409040291494..., 20.409040291494...), (-1, -24.739970002053..., -24.739970002053...),
 (1, 50.935260967822..., 50.935260967822...), (3, -201.005201402848..., -201.005201402848...)]
>>> dr.momentum_coef('f_2C_D0P', 2, l, m, 0), dc.f_2C_D0P(l, m, 0)
(397.0909090909...j, 397.0909090909...j)
>>> C, A = dr._primitives(4, 3, 30); Q = qr.primitives(4, 3, 30)
>>> max(np.max(abs(C[k][s].toarray()[:26, :26] - Q[k][s][:26, :26])) for k in C for s in C[k])
6.128431095930864e-14
```

(`f_1C_D0P` is a P column in the v row, a P↔T coupling, so it reaches $|\Delta l| = n + 1 = 3$ for $P_2$.)

### 2.3 Checks

- `python3 tests/diff_rot/angular_check.py [--quick]` (no run directory; 10 s with `--quick`): the harmonics, $X$, $D$,
  the vector operations and the primitives against `quad_reference.py`. Ends with `angular_check: ok`.
- `tests/diff_rot/gen_check.py [mmax [nmax [lmax]]]`, in a run directory with `diff_rot = 1`: every coupling of $f_0$, $f_2$
  against the closed forms, and that no coupling of $n$ = 4..nmax falls outside the $\Delta l$ kore assembles;
  `tests/heat_dr/mk_run.py` builds a run.
- `python3 tests/heat_dr/tw_sympy.py` (sympy): the thermal-wind $\sigma_2$, $\sigma_4$, $\sigma_6$ derived from
  thermal-wind balance against the closed forms; ends with `0 failures`.
- `tests/diff_rot/regress.sh OLD_KORE` (A, B against an older checkout), `bin/ru_insane.py` (the rows against a grid
  evaluation) and `tests/heat_dr/run_all.sh` (rows, assembly, budgets).

### 2.4 Adding a new rotation profile

Most profiles need no code: `diff_rot_type = 'user'` with numpy expressions of $r$ (and $x$ = fractional solar radius,
$p$ = `diff_rot_param`) for each even degree:

```
$ python3 tests/heat_dr/mk_run.py dr_user "par.diff_rot_type = 'user'; par.diff_rot_profiles = {0: '0.1*r**2', 4: '-0.05*(1 - r)'}; par.thermal_wind = 2"
dr_user ok
$ cd dr_user && python3 ../tests/diff_rot/gen_check.py 4 8
closed forms vs diff_rot.py, lmax 20, m = 0..4 (relative to the largest coefficient of the row block)
  n=0 u,P rows: max |diff_rot - closed| / row scale = 3.0e-17  (m, l) = (1, 3)
  ...
Δl selection rules, n = 4..8
  n= 4  largest coefficient outside the assembled Δl range (relative) 0.0e+00
  ...
worst 8.4e-16  ->  ok
```

A degree of any even $n$ works: the algebra builds $P_n(X)$ and the couplings for it, so no closed form has to be
derived. To add a named type instead, add it to `TYPES` (at the end) and its $\{n: f_n(r)\}$ to `shapes()` in
`bin/diff_rot.py`, then run `gen_check.py`, `ru_insane.py` and `regress.sh` (an unchanged profile must give bitwise the
same A, B). A new *kind* of background flow (not $\Delta\Omega\,r\sin\theta\,\hat{\boldsymbol\phi}$) needs new primitives:
write its $\mathbf{W}$ with the `Field` operations as `_primitives` does, and check it against a quadrature reference
written like `quad_reference.py`.
