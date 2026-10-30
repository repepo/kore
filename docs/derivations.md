# Derivation tools

Kore builds its matrices block by block. A block couples a row of degree $l'$ to a column of degree $l = l' + \Delta l$,
and it is a sum of radial operators (Chebyshev/Gegenbauer matrices, built by `submatrices.py`) times coefficients that
depend on $l$ and $m$ (in `operators.py` and `diff_rot.py`). A spherically symmetric background couples each degree only
to itself. Two backgrounds in kore depend on latitude, so they couple different degrees:

- an imposed **magnetic field** $\mathbf{B}_0$ ([Magnetic field](magnetic.md)), part A of this page;
- a **differential rotation** $\Delta\Omega(r, \theta)$ ([Differential rotation](differential_rotation_details.md)), part B.

For each, this page explains why the coefficients are computed the way they are (motivation), the mathematics
(method), which file and function does each step (implementation), how the result is checked and how far the checks
reach (verification), and how to run the tools yourself (worked examples).

| background | where kore gets the coefficients | how they were obtained | how they are checked |
| --- | --- | --- | --- |
| magnetic field | closed forms in `bin/operators.py` (`lorentz*`, `induction*`) | derived once from the definitions with sympy (`tests/magnetic/derive.py`), turned into formulas in $l$, $m$ (`tools/derive/fit_rows.py`) and into code (`tools/derive/gen_table.py`) | `tests/magnetic/derive_check.py`, `row_check.py`, `budget_check.py`, `bin/ru_insane.py`, comparison with `conductive_IC` |
| differential rotation | computed while kore runs, by an exact recurrence algebra (`bin/angular.py`, `bin/diff_rot.py`) | nothing to fit: the algebra is exact for any degree of $\Delta\Omega$ | `tests/diff_rot/angular_check.py`, `gen_check.py` (closed forms of `diff_rot_coefficients.py`), `quad_reference.py`, `tests/heat_dr/tw_sympy.py` |

**Prerequisites.** Python 3 with numpy and scipy (kore's environment, e.g. `source ~/z2/bin/activate`), and sympy for the
magnetic derivations (`pip install sympy`). The differential-rotation tools need no sympy, except `tw_sympy.py`.
Commands are given from the repository root unless a run directory is named. The outputs shown were recorded on
2026-10-29.

**Notation.** At a fixed azimuthal order $m$ the angular functions are kore's Schmidt harmonics
$Y_l = \sqrt{4\pi/(2l+1)}\,Y_l^m$ (Condon–Shortley phase), with $\int|Y_l|^2d\Omega = 4\pi/(2l+1)$; $L = l(l+1)$, and
$L'$ is the $L$ of the row. The flow is $\mathbf{u} = \rho_0^{-1}\nabla\times\nabla\times(\rho_0 P Y_l\mathbf{r}) + \nabla\times(T Y_l\mathbf{r})$
and the magnetic field $\mathbf{b} = \nabla\times\nabla\times(F Y_l\mathbf{r}) + \nabla\times(G Y_l\mathbf{r})$. The rows
of the momentum equation are its projections $\hat{\mathbf{r}}\cdot\nabla\times\nabla\times$ (section u, the 2-curl) and
$\hat{\mathbf{r}}\cdot\nabla\times$ (section v, the 1-curl); those of the induction equation are $\hat{\mathbf{r}}\cdot$
(section f) and $\hat{\mathbf{r}}\cdot\nabla\times$ (section g).

## A. Operators involving the background magnetic field

### A.1 Motivation

The Lorentz force $(\nabla\times\mathbf{b})\times\mathbf{B}_0 + (\nabla\times\mathbf{B}_0)\times\mathbf{b}$ and the induction
$\nabla\times(\mathbf{u}\times\mathbf{B}_0)$ are products of a perturbation of degree $l$ with a background of degree $n$.
Kore needs, for every pair of degrees, every radial term of these products, with its coefficient. With a variable density
the factors $\rho_0$ and $\rho_0^{-1}$ add terms in $(\ln\rho_0)'$ and $(\ln\rho_0)''$. Done by hand, this algebra is long
and easy to get wrong; `conductive_IC` has hand-derived forms for a dipole and a quadrupole at uniform density only.

So the algebra is done by a program, once, from the definitions:

- **sympy derives each block exactly**, for a given $(l, l', m)$ (`tests/magnetic/derive.py`). This needs no trust in a
  hand calculation or in another code.
- **The results are turned into formulas in $l$ and $m$** and written into `bin/operators.py`. Kore then evaluates
  formulas while it assembles. Calling sympy at run time instead would take seconds per block (there are thousands of
  blocks) and would make sympy a requirement of kore.

### A.2 Method

**Step 1: derive one block.** For the chosen background, poloidal $\mathbf{B}_0 = \nabla\times\nabla\times(h Y_n^0\mathbf{r})$ or
toroidal $\mathbf{B}_0 = \nabla\times(k Y_n^0\mathbf{r})$ ($n$ = 1 or 2), and a column field of degree $l$, derive.py builds
the row expression:

| row | equation | expression |
| --- | --- | --- |
| u | momentum, poloidal | $\rho_0\hat{\mathbf{r}}\cdot\nabla\times\nabla\times(\rho_0^{-1}\mathbf{G})$, $\mathbf{G} = (\nabla\times\mathbf{b})\times\mathbf{B}_0 + (\nabla\times\mathbf{B}_0)\times\mathbf{b}$ |
| v | momentum, toroidal | $\rho_0\hat{\mathbf{r}}\cdot\nabla\times(\rho_0^{-1}\mathbf{G})$ |
| f | induction, poloidal | $\hat{\mathbf{r}}\cdot\nabla\times(\mathbf{u}\times\mathbf{B}_0)$ |
| g | induction, toroidal | $\hat{\mathbf{r}}\cdot\nabla\times\nabla\times(\mathbf{u}\times\mathbf{B}_0)$ |

The radial functions ($P, T, F, G, h, k, \ln\rho_0$) stay symbolic; only $\theta$ is explicit. The expression is expanded
into monomials, each a coefficient times $r^p$ times derivatives of the unknown, of the background and of $\ln\rho_0$, for
example $r^{-2}\,h\,F'''$. The coefficient of each monomial still depends on $\theta$. It is projected on $Y_{l'}$:

$$
c = \frac{\int(\ldots)\,Y_{l'}^*\,d\Omega}{\int|Y_{l'}|^2d\Omega}.
$$

The integrand is a trigonometric polynomial of low degree, so 80-point Gauss–Legendre quadrature in $\theta$ gives the
integral exactly, up to round-off. The result for one block is a list of numbers, one per monomial.

**Step 2: which blocks exist (selection rules).** A background of degree $n$ multiplies the perturbation by functions of
$\theta$ of degree $n$. Projected back on the harmonics, degree $l$ then reaches only $l' = l + \Delta l$ with
$|\Delta l| \le n$; the curl that turns a poloidal field into a toroidal one shifts the parity of $\Delta l$ by one. For
the dipole ($n = 1$), for example, the u row takes $F$ from $l \pm 1$ and $G$ from $l$; for $n = 2$, $F$ from $l$ and
$l \pm 2$ and $G$ from $l \pm 1$. derive.py finds these rules itself: a block that does not exist comes out with no terms.

**Step 3: the form of the coefficients in $l$ and $m$.** Multiplying by a function of $\theta$ of degree $n$ is
multiplying by $\cos\theta$ up to $n$ times (or differentiating in $\theta$, which the same recurrences express; see
part B). Each multiplication by $\cos\theta$ moves $l$ by one, with the coefficients $p_l = \sqrt{(l+1)^2-m^2}/(2l+1)$
and $q_l = \sqrt{l^2-m^2}/(2l+1)$. All the square roots of a coefficient come from these factors, and the $\phi$
derivatives bring $im$. A block of a given $\Delta l$ shares one product of them, the **angular factor**
$A(l, m, \Delta l)$:

| | $\Delta l = 0$ | $\Delta l = \mp1$ | $\Delta l = \mp2$ |
| --- | --- | --- | --- |
| $n = 1$ | $im$ | $\sqrt{l^2-m^2}/(2l-1)$, $\sqrt{(l+1)^2-m^2}/(2l+3)$ | — |
| $n = 2$ | $1/((2l-1)(2l+3))$ | $im\sqrt{l^2-m^2}/(2l-1)$, $im\sqrt{(l+1)^2-m^2}/(2l+3)$ | $\frac{\sqrt{(l^2-m^2)((l-1)^2-m^2)}}{(2l-1)(2l-3)}$, $\frac{\sqrt{((l+1)^2-m^2)((l+2)^2-m^2)}}{(2l+3)(2l+5)}$ |

What remains of each coefficient, $c/A$, is a polynomial in $l$. It depends on $m$ only through $m^2$: the background is
axisymmetric, so changing the sign of $m$ changes only the $im$ factors, which $A$ already holds. So every coefficient
has the form

$$
c(l, m) = A(l, m, \Delta l)\,\big[a(l) + b(l)\,m^2\big],\qquad a, b\ \text{polynomials with rational coefficients}.
$$

**Step 4: find $a$ and $b$ by exact interpolation.** derive.py gives $c$ for a range of $(l, m)$, for example $l$ = 3–9 and
$m$ = 0–3, some 20 to 28 points. For each monomial, polynomials $a$, $b$ of increasing degree are tried:

1. their coefficients are solved by least squares from all the points (each polynomial must have fewer coefficients
   than there are values of $l$, so there are more points than unknowns);
2. each coefficient is rounded to the nearest rational with denominator $\le 60$;
3. the rounded polynomial is kept only if it reproduces every point to $10^{-9}$ of the largest value (in practice
   $10^{-13}$); otherwise the next degree is tried.

This is exact interpolation, not an approximation. The data are exact up to round-off, and the result has rational
coefficients and passes through more points than it has unknowns. If the form of step 3 did not hold for some block (a
wrong angular factor, a coefficient that is not a polynomial), no polynomial would pass through all the points, and the
tool would print `None` rather than a formula that is slightly off. This is why the tool is called a *fit*: it finds the
closed form from exact samples.

**Step 5: kore's scaling and labels.** Kore's rows are derive.py's rows times $-r^5/L'^3$ (u) or $-r^3/L'^3$ (v, f, g), and
times Hendrik for the Lorentz force (the minus sign: kore's matrix A holds minus the right-hand side). Each monomial becomes
a label, the name of the radial matrix `submatrices.py` builds: $X^{(d)} B^{(j)} \lambda_k\, r^p$ is
`{row}{-p}{code}{j}[lho{k}]ohr1_D{d}` in rows u, v and `{row}{-(p+1)}{code}{j}[lho{k}]_D{d}` in rows f, g, with `code` =
`bpo` ($c_\mathrm{norm} h$) or `bto` ($c_\mathrm{tor} k$) and $\lambda_k$ the $k$-th derivative of $\ln\rho_0$. An operator
then returns $-A\cdot\sum(\text{polynomial}\times\text{label})/L'^3$ for each block.

### A.3 Implementation

| step | file | function |
| --- | --- | --- |
| background profiles $h$, $k$ and their derivatives | `bin/background_field.py` | `h`, `k`, `bpoX`, `btoX` |
| step 1, derive one block | `tests/magnetic/derive.py` | `rows` (`velocity`, `bfield`, `B0`, `lorentz`, `project`) |
| steps 3–4, closed forms in $l$, $m$ | `tools/derive/fit_rows.py` | `A` (angular factors), `fit` (exact interpolation), `main` |
| step 5, code | `tools/derive/gen_table.py` | `main` (labels, table entries, label parities) |
| the operators | `bin/operators.py` | `lorentz`, `induction` (poloidal + toroidal); degree-1 poloidal `lorentz_pol`, `induction_pol` (conductive_IC's forms, plus `lorentz_rho` and the `lho` labels for $(\ln\rho_0)'$); `lorentz_tor`, `induction_tor`, `*_tor2`, `*_pol2` (fitted); `quad_factors`, `moff` (column offsets) |
| the radial matrices of the labels | `bin/submatrices.py` | the magnetic label lists |
| placing the blocks, $\Delta l = -2..2$ | `bin/assemble.py` | the loops over `op.lorentz`, `op.induction` |

Where each closed form in `operators.py` comes from:

- degree-1 poloidal field ('axial', 'FDM' with `B0_l` = 1, 'G21 dipole', 'Luo_S1'): `conductive_IC`'s hand-derived forms
  (commit 39487bbf), with the $(\ln\rho_0)'$ terms derived for this branch;
- degree-2 poloidal field ('Luo_S2', 'FDM' with `B0_l` = 2): fitted (`fit_rows.py pol 2`) and generated (`gen_table.py`);
- toroidal fields of degree 1 and 2: fitted (`fit_rows.py tor 1`, `tor 2`).

### A.4 Verification

- **`tests/magnetic/derive_check.py`** (no run directory, about 40 minutes): derive.py against every closed form, term by
  term: `conductive_IC`'s uniform-density operators, the $(\ln\rho_0)'$ terms, the toroidal forms, and the degree-2 tables
  (read from the source of `operators.py`). Degrees $l$ 3–6, $m$ 0–3.
- **`tests/magnetic/row_check.py`** (in a run directory): what kore really assembles, i.e. the closed forms times this
  run's radial matrices, in their Gegenbauer bases, against derive.py evaluated with the run's own profiles. This catches a
  wrong label, a wrong power of $r$ or a wrong basis, even when the closed form is right. Degrees up to $m + 4$ by default.
- **Energy budgets** (`budget_check.py`) and the identity
  $\int\rho_0\mathbf{u}^*\cdot\mathbf{F}_L\,dV = -\mathrm{Hendrik}\,\overline{\int\mathbf{b}^*\cdot\nabla\times(\mathbf{u}\times\mathbf{B}_0)dV}$
  (`bin/ru_insane.py`): spin_doctor computes the powers of the Lorentz force and of the induction from the eigenvectors
  with its own formulas (on an $(r, \theta)$ grid for toroidal and degree-2 fields), independently of the operators.

**Which degrees these checks reach.** derive_check ($l$ 3–6) and row_check with its default limit look at degrees inside
the range the closed forms were fitted on. They show that the code equals the fit and the derivation there, not that the
formulas hold at large $l$. The evidence at large $l$ is:

- poloidal fields ('axial', 'FDM' of degree 1 and 2, 'G21 dipole', 'Luo_S1', 'Luo_S2'): the eigenvalues agree with
  `conductive_IC`, whose closed forms were derived independently, to $6\cdot10^{-13}$ or better with lmax 63–66 (full
  sphere and inner core) and 79–128 (conducting inner core), and to $10^{-9}$ for the inviscid torsional modes of 'Luo_S2'
  with lmax 95–191 (against `conductive_IC` with its inviscid bases corrected, [Magnetic field](magnetic.md));
- toroidal fields (no independent code): the energy budgets, which sum the powers over all degrees of the run
  (lmax 47–64, residuals $10^{-11}$ to $10^{-13}$), and the Malkus (1967) frequencies ([Magnetic field](magnetic.md),
  Checks). Both test the operators through whole eigenvectors, not coefficient by coefficient;
- a one-off comparison (2026-10-30) of every fitted coefficient of the degree-2 fields with derive.py at $l = 20$, $m = 3$
  and $l = 31$, $m = 7$, outside the fitting range: largest relative difference $4.3\cdot10^{-10}$ (poloidal, 40 blocks)
  and $2.0\cdot10^{-10}$ (toroidal, 34 blocks), no term missing (the round-off of derive.py's quadrature at these
  degrees). The degree-1 fields were not compared there.

### A.5 How to use it

**Derive one block.** The Lorentz force of the dipole (`--b0 pol1`, the default) in the u row of degree 3, column $F$
of degree 2, $m = 1$ (`derive.py ROW COL LCOL LROW M`; about 20 s):

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

`--b0` chooses another field: `pol2` (degree-2 poloidal: 'Luo_S2', FDM with `B0_l = 2`), `tor1`, `tor2`. The induction of
the quadrupole in the f row of degree 4 from $P$ of degree 2 (about 2 s):

```
$ python3 derive.py f P 2 4 1 --b0 pol2
+28.168588671694 +0.000000000000i   r^-2  P0 h0 lr1
-28.168588671694 +0.000000000000i   r^-2  P0 h1
+28.168588671694 +0.000000000000i   r^-2  P1 h0
(3 terms)
```

**Find the closed forms.** `tools/derive/fit_rows.py KIND N` does steps 1, 3 and 4 for every block. The toroidal $n = 1$
field, rows f and g only (25 s on 30 cores; the derived rows are cached in `fit_tor1.pkl`, so a second run is instant):

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

A line ending in `None` means the polynomial needs more points than the $l$ range gives: widen the range. For the degree-2
poloidal field, `python3 tools/derive/fit_rows.py pol 2 --jobs 30` ($l$ 3–9, about 15 min) gives all blocks except four of
the u row at $\Delta l = 0$, whose $a(l)$ has degree 6; `--rows u --l 3:14` gives those too. These two runs reproduce every
coefficient of `operators.induction_pol2` and `lorentz_pol2`.

**Turn them into code.** `tools/derive/gen_table.py` writes the table entries in kore's labels and lists the labels
`submatrices.py` must build, with the vector parity of their column:

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

The operator returns `-A(l, m, Δl) * entry / L**3` (times `par.Hendrik` for rows u, v), with the column offset of
`operators.moff`. Compare `operators.induction_tor`, where the f row is `-1j*m*L*f1bto0_D0` ($A = im$, entry $L$) and the
g row at $\Delta l = -1$ is `-Cm*l*(l-1)*(l+1)*(g2bto0_D0 + g1bto0lho1_D0 - g1bto1_D0)`.

**Check against kore.** derive_check needs no run directory; its last lines are

```
rows v and T columns: no (ln rho0) terms
derive_check.py: ok
```

row_check runs in a run directory after `submatrices.py`; `tests/magnetic/mk_run.py NAME 'par.x = ..'` builds one:

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

This takes 12 minutes: each block calls derive.py, and the argument 7 limits the row degrees checked to 7. Use the
absolute path of `row_check.py` if the run directory is elsewhere; with 'Wu 2005', use N 192 in a shell and N 288 in a
full sphere to reach round-off. `tests/magnetic/run_all.sh` runs all the magnetic checks.

**Add a new background field.**

1. *Its shape.* In `bin/background_field.py`, add the name to `B0_TYPES` (at the end: its index goes to `params.dat`), its
   $h(r)$ and derivatives to `h(r, k)` (a polynomial goes in `POLY`; mind the parity $(-1)^n$ in a full sphere), its degree
   and equatorial symmetry (`B0_l`, `symmB0`), and its normalisation in `B0_norm` if `cnorm` is not a number.
2. *Its couplings.* If its degree is 1 or 2, the existing operators apply unchanged: they depend on the degree, not on
   $h$. For another degree $n$: derive (`rows(..., b0=('pol', n))`), add $n$'s angular factors to `fit_rows.A`, find the
   closed forms, generate the code and add a `lorentz_poln`/`induction_poln` pair with the new labels in `submatrices.py`;
   `assemble.py` loops over $\Delta l = -2..2$, so widen that loop for $n > 2$.
3. *Check.* Extend `derive_check.py` with the new tables (its section 5 reads them from `operators.py`), run `row_check.py`
   and the budgets on a run with the new field, compare at a few large $l$ with derive.py, and compare with an independent
   code or an analytic result where you have one.

## B. Operators involving background differential rotation

### B.1 Motivation

A rotation that varies with latitude couples each degree $l$ of the perturbation to $l \pm 1$, $l \pm 2$, …, up to the
degree of $\Delta\Omega$. Kore needs those couplings for every $l$ up to lmax, which can be a few thousand for the Sun,
and for every even degree $n$ of $\Delta\Omega$; observed rotation profiles need $n$ up to 12 or more. Three ways to get
them were considered:

- **closed forms** (Gaunt-type integrals worked out by hand) exist for $n$ = 0 and 2 (`diff_rot_coefficients.py`). Each
  new $n$ means a new derivation;
- **quadrature on a $\theta$ grid** (kore's former method, now `quad_reference.py`) works for any $n$, but it costs
  $O(\mathrm{lmax}^2)$ time and memory (177 s and 2.5 GB per rank at lmax 1025) and fails above $l \approx 645$, where
  scipy's harmonics overflow;
- **an exact recurrence algebra** (`bin/angular.py`), the method kore uses. Every function of $\cos\theta$ that appears is
  a polynomial, and multiplication by $\cos\theta$ on the harmonics is exactly a three-term recurrence. So every coupling
  is a short product of sparse banded matrices: exact up to round-off, for any $n$, at any degree, in $O(\mathrm{lmax})$
  time and memory (0.014 s and 6.8 MB at lmax 1025). Nothing is fitted, and nothing needs deriving for a new profile.

### B.2 Method

**Step 1: the force.** The rotation is $\Omega = \Omega_\mathrm{ref}[1 + \Delta\Omega(r, \theta)]$ with

$$
\Delta\Omega = \mathtt{diff\_rot\_ampl}\sum_n f_n(r)\,P_n(\cos\theta),\qquad n\ \text{even}.
$$

In the frame of $\Omega_\mathrm{ref}$ it is the flow $\mathbf{U}_0 = G\,\Delta\Omega\,r\sin\theta\,\hat{\boldsymbol\phi}$
($G$ = Gaspard), which acts on the perturbation through

$$
\mathbf{W} = G\big[\Delta\Omega\,(im\,\mathbf{u} + 2\hat{\mathbf{z}}\times\mathbf{u}) + \sin\theta\,(r\,\partial_r\Delta\Omega\,u_r + \partial_\theta\Delta\Omega\,u_\theta)\,\hat{\boldsymbol\phi}\big],
$$

and on the heat equation through $imG\Delta\Omega\,s$ and $\mathbf{u}\cdot\nabla S_{tw}$, where $S_{tw}$ is the
thermal-wind entropy that balances the rotation.

**Step 2: two exact recurrences.** With $x = \cos\theta$,

$$
x\,Y_l = p_l\,Y_{l+1} + q_l\,Y_{l-1},\qquad (1 - x^2)\frac{dY_l}{dx} = (l+1)\,q_l\,Y_{l-1} - l\,p_l\,Y_{l+1},
$$

$p_l = \sqrt{(l+1)^2-m^2}/(2l+1)$, $q_l = \sqrt{l^2-m^2}/(2l+1)$, for kore's Schmidt harmonics. Written on the degree
index (row = target $l'$, column = source $l$), they are two tridiagonal matrices: $X$ (multiplication by $x$) and $D$
(multiplication by $-\sin\theta\,\partial_\theta$). Their adjoint under $\int d\Omega$ is $D^\dagger = 2X - D$.

**Step 3: functions of $\theta$ become polynomials in $X$.** Multiplying by $P_n(x)$ is the matrix $P_n(X)$, built with the
Legendre recurrence $(k+1)P_{k+1} = (2k+1)\,X P_k - k\,P_{k-1}$; $P_n'(X)$ follows in the same way. $P_n(X)$ has $n + 1$
non-zero diagonals, every other one: $P_n$ couples $l$ to $l' = l - n, l - n + 2, \ldots, l + n$. These are the selection
rules: P→P and T→T couplings have $|\Delta l| \le n$, even; P↔T couplings have $|\Delta l| \le n + 1$, odd (the
$\hat{\mathbf{z}}\times$ and the curl add one degree). The matrices are exact; only their last few rows are wrong, because
products of truncated matrices miss the degrees beyond the truncation. They are therefore built a few degrees beyond what
is used (lmax + n + 6).

**Step 4: vector fields.** A vector field is expanded as
$\mathbf{F} = \sum_l\,[\,q_l Y_l\hat{\mathbf{r}} + s_l\,r\nabla Y_l + t_l\,r\nabla Y_l\times\hat{\mathbf{r}}\,]$. The operations
$\mathbf{W}$ needs are exact sparse products of $X$, $D$ and diagonal matrices: multiplication by $g(x)$, $\hat{\mathbf{z}}\times$,
$\hat{\mathbf{z}}\times(c\,\hat{\mathbf{r}})$, and the scalar $\sin\theta\,F_\theta$. The formulas are in the docstring of
`bin/angular.py` and in [Differential rotation](differential_rotation_details.md), §3.

**Step 5: the 12 primitives.** For one degree $n$, with $\Delta\Omega = f(r)\,P_n(x)$, $\mathbf{W}$ is linear in four radial
shapes: $f\,LP/r$ (from $u_r$), $f\,a$ with $a = (rP)'/r + \lambda_1 P$ (the consoidal part), $f\,T$ (the toroidal part), and
$(rf')\,LP/r$ (the radial shear). Applying the operations of step 4 to each shape and taking the three components $q$, $s$,
$t$ of the result gives 4 × 3 = 12 sparse matrices $C[k][\mathrm{shape}]$ (rows: target degree, columns: source degree).

**Step 6: from primitives to radial operators.** The u row is the 2-curl $\hat{\mathbf{r}}\cdot\nabla\times\nabla\times$ and
the v row is the 1-curl $\hat{\mathbf{r}}\cdot\nabla\times$. Applied to a field with components $(q, s, t)$, they combine $q$
and $s$ (u) or $t$ (v) with radial derivatives. Combining the primitives in the same way gives, for each radial operator
of the row (for example $f\,P$, $f\,P'$, $f\,(\ln\rho_0)'P$, $(rf')P$), its coefficient for every pair $(l', l)$, multiplied
by $L'$ of the target. For example, `f_2C_D0P` multiplies $f\,P$ in the u row.

**Step 7: the heat equation and the thermal wind.** The heat terms need only $P_n(X)$, $P_n'(X)$ and $P_n'(X)\,D$. The
thermal-wind entropy $S_{tw} = \sum_n \sigma_n(r)P_n(\cos\theta)$ follows from thermal-wind balance: in Legendre space,
$D\sigma = (1-x^2)\,\partial_x S_{tw}$ is solved for the $\sigma_n$ by forward substitution, exactly.

**Step 8: into kore.** Each radial operator is a label with the profile codes `zf`, `zd`, `zs` + degree for $f_n$,
$rf_n'$, $r^2f_n''$ (and `tw`, `tp` for $p_0\sigma_n$, $p_0\sigma_n'$). `submatrices.py` builds its matrix. A block is then
$G\sum_n\sum(\text{coefficient}\times\text{radial operator})/L'^3$, placed by `assemble.py` at the $\Delta l$ of the
selection rules.

### B.3 Implementation

| step | file | function |
| --- | --- | --- |
| the profiles $f_n(r)$ and their fits | `bin/diff_rot.py` | `shapes`, `fX`, `profile_functions` |
| step 2, the recurrences | `bin/angular.py` | `coefficients`, `Algebra` (`X`, `D`, `Dd`, `I`, `L`, `Li`, `N`) |
| step 3, $P_n(X)$ | `bin/angular.py` | `Algebra.legendre` |
| step 4, vector fields | `bin/angular.py` | `Field`, `Algebra.field`, `mult`, `zcross`, `zcross_r`, `sin_theta` |
| step 5, the 12 primitives | `bin/diff_rot.py` | `_primitives` |
| step 6, coefficients of the radial operators | `bin/diff_rot.py` | `momentum_coefficients`, `momentum_coef` |
| step 7, heat terms and thermal wind | `bin/diff_rot.py` | `heat_coefficients`, `heat_coef`, `sigma_coefficients` |
| step 8, labels, $\Delta l$ ranges, blocks | `bin/diff_rot.py` | `labels`, `dl_range`, `momentum_block`, `heat_block` (called by `submatrices.py`, `assemble.py`, `operators.py`) |
| harmonics on a grid (for checks and post-processing) | `bin/angular.py` | `ylm_grid`, `ylm` |

### B.4 Verification

- **`tests/diff_rot/angular_check.py`** (no run directory): each piece of the algebra against something computed another
  way: the harmonics against exact rational Legendre values ($l \le 2000$) and scipy ($l \le 640$), their norms up to
  $l = 3000$ (also $m$ = 400 and 1425), $X$ and $D$ on a grid and their adjoints, the vector operations against
  Gauss–Legendre projections at low degree and around $l = 1200$, and the 12 primitives against the former quadrature
  builder (`quad_reference.py`) for $n \le 16$. Because it checks the algebra itself, it covers every $n$ and degrees far
  above what the closed forms reach.
- **`tests/diff_rot/gen_check.py`** (in a run directory): every coupling of $f_0$ and $f_2$ against the independent closed
  forms of `diff_rot_coefficients.py`, for all $l \le$ lmax and $m \le$ mmax (recorded runs: lmax 20 and 1025, agreement
  to $10^{-15}$ of the row scale), and, for $n$ = 4 … nmax, that no coupling falls outside the $\Delta l$ kore assembles (a
  coupling there would be dropped silently).
- **`tests/heat_dr/tw_sympy.py`**: the thermal-wind $\sigma_2$, $\sigma_4$, $\sigma_6$ derived symbolically from
  thermal-wind balance, against the closed forms that gen_check compares the run-time $\sigma_n$ with.
- **Whole-system checks**: `bin/ru_insane.py` (the assembled rows against a grid evaluation of the force),
  `tests/diff_rot/regress.sh` (A and B against an older checkout), `tests/heat_dr/run_all.sh` (rows, assembly, energy
  budgets), and the Mukhopadhyay et al. (2025) solar benchmark (`run_bekki.py`, [Benchmarks](benchmarks.md)).

### B.5 How to use it

**The recurrence algebra.** Check it against the harmonics on a grid. `angular.ylm_grid(m, lmax, x)` returns $Y_l$,
$\partial_\theta Y_l$ and $imY_l/\sin\theta$:

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

**The couplings.** Compare a few couplings with the closed forms of `tests/diff_rot/diff_rot_coefficients.py` (names
`f_*` for $f_2$, `f0_*` for $f_0$), and the primitives with the former quadrature (`tests/diff_rot/quad_reference.py`).
`diff_rot` reads `parameters`; in a session outside a run directory, give it a stand-in with `lmax`. (numpy 2 prints the
numbers as `np.float64(...)`.)

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

`f_1C_D0P` is a P column in the v row, a P↔T coupling, so for $P_2$ it reaches $|\Delta l| = n + 1 = 3$.

**The checks.** `python3 tests/diff_rot/angular_check.py --quick` takes 10 s and ends with `angular_check: ok`.
`python3 tests/heat_dr/tw_sympy.py` ends with `0 failures`. gen_check runs in a run directory (below).

**Add a new rotation profile.** Most profiles need no code: use `diff_rot_type = 'user'` with numpy expressions of $r$ (and
of $x$, the fractional solar radius, and $p$ = `diff_rot_param`) for each even degree:

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

Any even degree $n$ works: the algebra builds $P_n(X)$ and the couplings for it, so nothing has to be derived. To add a
named type instead, add it to `TYPES` (at the end) and its $\{n: f_n(r)\}$ to `shapes()` in `bin/diff_rot.py`; then run
`gen_check.py`, `ru_insane.py` and `regress.sh` (an unchanged profile must give bitwise the same A and B). A new *kind* of
background flow (not of the form $\Delta\Omega\,r\sin\theta\,\hat{\boldsymbol\phi}$) needs new primitives: write its
$\mathbf{W}$ with the `Field` operations as `_primitives` does, and check it against a quadrature reference written like
`quad_reference.py`.
