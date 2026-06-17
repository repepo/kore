# Mantle Magnetic Diffusion in Kore

`Ultrapherical_Spectral_Method.ipynb`, provides the ultraspherical spectral method and mantle-only magnetic diffusion model used as conceptual and numerical guidance.

The resolved mantle option adds magnetic diffusion in a solid shell outside the fluid core. It does not add mantle flow. It does not yet add a mantle Lorentz-force volume coupling or a new electromagnetic torque model. It also keeps the combined case

```python
mantle = 'resolved'
innercore = 'conducting, Chebys'
```

temporarily guarded until the simpler core plus resolved mantle configuration is fully validated.

---

## 1. Physical Objective

The original Kore magnetic boundary options at the core-mantle boundary were:

```python
mantle = 'insulator'
mantle = 'TWA'
```

The new option is:

```python
mantle = 'resolved'
```

The physical goal is to replace a purely boundary-condition representation of the mantle by an explicitly resolved conducting shell:

$$
r \in [r_{\rm cmb}, r_{\rm surface}].
$$

Kore nondimensionalizes the CMB radius as:

$$
r_{\rm cmb}=1.
$$

Therefore the resolved mantle domain is:

$$
1 \le r \le r_{\rm surface}.
$$

The mantle is assumed to be solid. Hence it has no velocity field:

$$
\mathbf u_m = 0.
$$

Its magnetic field evolves only by magnetic diffusion:

$$
\partial_t \mathbf B_m
=
\nabla \times
\left(
\eta_m(r)\nabla\times \mathbf B_m
\right),
$$

with the nondimensional scaling already used by Kore through `par.Em` and `par.OmgTau`.

---

## 2. Magnetic Field Representation

Kore uses a poloidal-toroidal representation of the magnetic field. For each spherical harmonic degree and order,

$$
\ell,m,
$$

the mantle magnetic field is represented by two radial scalar functions:

$$
F_m(r) \qquad \text{and} \qquad G_m(r).
$$

In the notebook these are written directly as:

$$
F_{\ell m}(r),
\qquad
G_{\ell m}(r).
$$

In the Kore implementation we use the code sections:

```text
p = mantle poloidal scalar F_m
q = mantle toroidal scalar G_m
```

The one-letter names `p` and `q` were chosen deliberately because the existing matrix-file parser in `operators.py` reads the first character of each `.mtx` file name as the section name.

The magnetic field components associated with the radial scalars are qualitatively:

$$
B_r
=
\frac{\ell(\ell+1)}{r^2}
F_{\ell m}(r)Y_{\ell m},
$$

$$
B_\theta
=
\frac{1}{r}F'_{\ell m}(r)\partial_\theta Y_{\ell m}
+
\frac{1}{r\sin\theta}G_{\ell m}(r)\partial_\phi Y_{\ell m},
$$

$$
B_\phi
=
\frac{1}{r\sin\theta}F'_{\ell m}(r)\partial_\phi Y_{\ell m}
-
\frac{1}{r}G_{\ell m}(r)\partial_\theta Y_{\ell m}.
$$

Thus:

- `p` or `F_m` controls the poloidal magnetic field, including the radial magnetic field.
- `q` or `G_m` controls the toroidal magnetic field, which cannot extend into an insulating exterior.

---

## 3. Mantle Conductivity and Diffusivity

The resolved mantle conductivity is parameterized in `parameters.py` by:

```python
r_surface = 1.8
N_mantle = 8
mu_mantle = 1.0
sigma_mantle_0 = 1e3
sigma_mantle_alpha = 1.0
```

The conductivity profile is:

$$
\sigma_m(r)
=
\sigma_{\rm mantle,0}
\exp
\left[-\alpha_\sigma(r-r_{\rm cmb})
\right].
$$

In code:

```python
def conductivity_mantle(r):
    return par.sigma_mantle_0 * np.exp(-par.sigma_mantle_alpha * (r - rcmb))
```

The magnetic diffusivity is:

$$
\eta_m(r)=\frac{1}{\sigma_m(r)}.
$$

In code:

```python
def mag_diffus_mantle(r):
    return 1/conductivity_mantle(r)
```

Its derivative is:

$$
\frac{d\eta_m}{dr}
= \alpha_\sigma \eta_m(r).
$$

In code:

```python
def dmag_diffus_mantle(r):
    return -par.sigma_mantle_alpha * mag_diffus_mantle(r)
```

This derivative appears only in the toroidal diffusion operator. Physically, this is because variable diffusivity modifies the curl-curl diffusion of the toroidal component through the gradient of diffusivity.

---

## 4. Mantle Magnetic Diffusion Equations

The notebook introduces the mantle-only equations:

$$
i\omega F_m
=
\eta_m(r)
\left[
F_m''
+
\frac{2}{r}F_m'
-
\frac{\ell(\ell+1)}{r^2}F_m
\right],
$$

and

$$
i\omega G_m
=
\eta_m(r)
\left[
G_m''
+
\frac{2}{r}G_m'
-
\frac{\ell(\ell+1)}{r^2}G_m
\right]
+
\eta_m'(r)
\left[
G_m'
+
\frac{1}{r}G_m
\right].
$$

Kore's magnetic operators are assembled in a form multiplied by:

$$
Lr^2,
\qquad
L=\ell(\ell+1).
$$

The implemented mantle poloidal diffusion operator is:

$$
\mathcal D_p
=
\Omega\tau E_m L
\left[
-L\eta_m I
+
2r\eta_m \frac{d}{dr}
+
r^2\eta_m \frac{d^2}{dr^2}
\right].
$$

The implemented mantle toroidal diffusion operator is:

$$
\mathcal D_q
=
\Omega\tau E_m L
\left[
-L\eta_m I
+
2r\eta_m \frac{d}{dr}
+
r^2\eta_m \frac{d^2}{dr^2}
+
r\eta_m' I
+
r^2\eta_m'\frac{d}{dr}
\right].
$$

In `operators.py` these are:

```python
def magnetic_diffusion_mantle(l, section, component, offdiag):
    out = 0
    L = l*(l+1)

    if offdiag == 0:
        if section == 'p' and component == 'bpol_mantle':
            out = L*( -L*etaIp + 2*retaD1p + r2etaD2p )

        elif section == 'q' and component == 'btor_mantle':
            out = L*( -L*etaIq + 2*retaD1q + r2etaD2q
                      + reta1Iq + r2eta1D1q )

    return par.OmgTau * par.Em * out
```

The mass or time-derivative operators are:

$$
\mathcal M_p = Lr^2 I,
\qquad
\mathcal M_q = Lr^2 I.
$$

In code:

```python
def b_mantle(l, section, component, offdiag):
    out = 0
    L = l*(l+1)

    if offdiag == 0:
        if section == 'p' and component == 'bpol_mantle':
            out = L * r2Ip

        elif section == 'q' and component == 'btor_mantle':
            out = L * r2Iq

    return out
```

Implementation note: the assembled mantle blocks currently follow the same positive-block convention as Kore's Chebyshev conducting-inner-core magnetic sections `fic/gic`. This convention should be included in the eigenvalue sign validation, because the fluid-core induction equation uses the familiar `iwb - diffusion` pattern.

---

## 5. Radial Mapping and Chebyshev Expansion

The ultraspherical method works on:

$$
x \in [-1,1].
$$

The physical mantle radius is mapped by:

$$
r(x)
=
\frac{r_{\rm surface}-1}{2}x
+
\frac{r_{\rm surface}+1}{2}.
$$

Let:

$$
L_r=r_{\rm surface}-1.
$$

Then:

$$
\frac{d}{dr}
=
\frac{2}{L_r}\frac{d}{dx},
$$

and:

$$
\frac{d^2}{dr^2}
=
\left(\frac{2}{L_r}\right)^2
\frac{d^2}{dx^2}.
$$

The unknown mantle scalars are expanded as Chebyshev series:

$$
F_m(r(x))
\approx
\sum_{k=0}^{N_{\rm mantle}-1} p_kT_k(x),
$$

$$
G_m(r(x))
\approx
\sum_{k=0}^{N_{\rm mantle}-1} q_kT_k(x).
$$

The unknowns in the matrix problem are the coefficient vectors:

$$
p = (p_0,p_1,\dots,p_{N_{\rm mantle}-1})^T,
$$

$$
q = (q_0,q_1,\dots,q_{N_{\rm mantle}-1})^T.
$$

Kore's existing derivative helper:

```python
ut.Dlam(lamb, N, R1, R2)
```

already includes the mapping factor:

$$
\left(\frac{2}{R_2-R_1}\right)^\lambda.
$$

Therefore the mantle derivative matrices are constructed on:

$$
R_1=1,
\qquad
R_2=r_{\rm surface}.
$$

In code:

```python
D1m = ut.Dlam(1, Nm, R1, R2)
D2m = ut.Dlam(2, Nm, R1, R2)
```

---

## 6. Ultraspherical Spectral Method

The notebook explains the ultraspherical method used to build well-conditioned spectral operators.

A scalar function is represented by Chebyshev coefficients:

$$
u(x)
\approx
\sum_{k=0}^{N-1}u_kT_k(x).
$$

Differentiation changes the polynomial basis. For example:

$$
\frac{d}{dx}T_k(x)
=
kC_{k-1}^{(1)}(x),
$$

so first derivatives naturally live in the ultraspherical basis:

$$
C^{(1)}.
$$

After two derivatives, the result lives in:

$$
C^{(2)}.
$$

The role of conversion matrices is to put all terms into the same basis:

$$
S_0: T \rightarrow C^{(1)},
$$

$$
S_1: C^{(1)} \rightarrow C^{(2)}.
$$

For a second-order differential operator:

$$
a_2(x)u''
+
a_1(x)u'
+
a_0(x)u,
$$

the coefficient-space operator has the schematic form:

$$
L_N
=
M_2[a_2]D_2
+
M_2[a_1]S_1D_1
+
M_2[a_0]S_1S_0.
$$

Here:

- `D1` and `D2` are sparse differentiation matrices.
- `S0` and `S1` are sparse or banded conversion matrices.
- `M_lambda[a]` is multiplication by a variable coefficient in coefficient space.

This produces almost-banded matrices: dense rows only appear where boundary conditions are inserted.

---

## 7. Mantle Submatrix Generation

The mantle matrices are generated in `submatrices.py` by:

```python
def write_mantle_magnetic_submatrices():
    Nm = par.N_mantle
    R1 = ut.rcmb
    R2 = par.r_surface
```

The implementation constructs:

```python
S0m = ut.Slam(0, Nm)
S1m = ut.Slam(1, Nm)
S10m = S1m * S0m

D1m = ut.Dlam(1, Nm, R1, R2)
D2m = ut.Dlam(2, Nm, R1, R2)
```

The radial coefficients are expanded spectrally:

```python
r2m = ut.chebco(2, Nm, tol, R1, R2)

eta0m = ut.chebco_f(ut.mag_diffus_mantle, 0, Nm, R1, R2, tol)
reta0m = ut.chebco_f(ut.mag_diffus_mantle, 1, Nm, R1, R2, tol)
r2eta0m = ut.chebco_f(ut.mag_diffus_mantle, 2, Nm, R1, R2, tol)
reta1m = ut.chebco_f(ut.dmag_diffus_mantle, 1, Nm, R1, R2, tol)
r2eta1m = ut.chebco_f(ut.dmag_diffus_mantle, 2, Nm, R1, R2, tol)
```

These correspond to the functions:

$$
r^2,
\qquad
\eta_m,
\qquad
r\eta_m,
\qquad
r^2\eta_m,
\qquad
r\eta_m',
\qquad
r^2\eta_m'.
$$

The files written for `p` are:

```text
p20.mtx
peta000.mtx
peta101.mtx
peta202.mtx
```

The files written for `q` are:

```text
q20.mtx
qeta000.mtx
qeta101.mtx
qeta202.mtx
qeta110.mtx
qeta211.mtx
```

The extra `qeta110` and `qeta211` matrices encode the diffusivity-gradient terms:

$$
r\eta_m'I,
\qquad
r^2\eta_m'\frac{d}{dr}.
$$

---

## 8. Reserved Boundary Rows

Each mantle scalar equation is second order. Therefore each mantle block needs two boundary rows.

This is implemented by shifting the differential operator down by two rows:

```python
def _shift_mantle_bc_rows(matrix, Nm):
    if np.isscalar(matrix):
        matrix = ss.csr_matrix((Nm,Nm))
    else:
        matrix = ss.csr_matrix(matrix)
    z2m = ss.csr_matrix((2,Nm))
    return ss.vstack([z2m, matrix[:-2,:]], format='csr')
```

Mathematically, each mantle matrix has the structure:

$$
\begin{pmatrix}
\text{boundary row 0} \\
\text{boundary row 1} \\
\text{interior diffusion rows}
\end{pmatrix}.
$$

Before boundary rows are inserted, the first two rows of the `.mtx` files are empty.

---

## 9. Boundary Evaluation Vectors

Boundary rows are constructed using Chebyshev evaluation vectors from `bc_variables.py`.

The helper:

```python
Tk(x, N, lamb_max, R1, R2)
```

returns Chebyshev values and radial derivatives at:

$$
x=-1
$$

or:

$$
x=+1.
$$

For the mantle:

```python
Tm_cmb  = Tk(-1, par.N_mantle-1, 4, ut.rcmb, par.r_surface)
Tm_surf = Tk( 1, par.N_mantle-1, 4, ut.rcmb, par.r_surface)
```

Thus:

$$
x=-1 \quad \Longleftrightarrow \quad r=1,
$$

and:

$$
x=+1 \quad \Longleftrightarrow \quad r=r_{\rm surface}.
$$

The code stores:

```python
Fm0_cmb = Tm_cmb[:,0]
Fm1_cmb = Tm_cmb[:,1]
Fm2_cmb = Tm_cmb[:,2]

Gm0_cmb = Tm_cmb[:,0]
Gm1_cmb = Tm_cmb[:,1]

Fm0_surf = Tm_surf[:,0]
Fm1_surf = Tm_surf[:,1]

Gm0_surf = Tm_surf[:,0]
```

These are row vectors acting on the mantle Chebyshev coefficients.

For example:

$$
F_m(1)
\quad \Longleftrightarrow \quad
\texttt{bv.Fm0\_cmb}\,p,
$$

and:

$$
F_m'(r_{\rm surface})
\quad \Longleftrightarrow \quad
\texttt{bv.Fm1\_surf}\,p.
$$

---

## 10. Global State Vector

Adding mantle unknowns requires explicit global offsets. The intended state vector is:

$$
x =
\left(
u,
v,
f,
g,
f_{\rm ic},
g_{\rm ic},
p,
q,
h,
i
\right)^T.
$$

Here:

- `u` is the fluid poloidal velocity.
- `v` is the fluid toroidal velocity.
- `f` is the fluid-core poloidal magnetic scalar.
- `g` is the fluid-core toroidal magnetic scalar.
- `fic` is the conducting-inner-core poloidal magnetic scalar.
- `gic` is the conducting-inner-core toroidal magnetic scalar.
- `p` is the mantle poloidal magnetic scalar.
- `q` is the mantle toroidal magnetic scalar.
- `h` is temperature.
- `i` is composition.

In `utils.py`, the resolved mantle sizes are:

```python
resolved_mantle = (par.magnetic == 1 and par.mantle == 'resolved')
Nman = par.N_mantle if resolved_mantle else 0
nman = int(Nman*(par.lmax-par.m+1)/2) if resolved_mantle else 0
```

The offsets are:

```python
offset_u = 0
offset_v = offset_u + size_u
offset_f = offset_v + size_v
offset_g = offset_f + size_f
offset_rotdyn = offset_g + size_g
offset_fic = offset_rotdyn + size_rotdyn
offset_gic = offset_fic + size_fic
offset_p = offset_gic + size_gic
offset_q = offset_p + size_p
offset_h = offset_q + size_q
offset_i = offset_h + size_h
```

This was necessary because hard-coded positions such as:

$$
2n,
\qquad
3n,
\qquad
4n
$$

would become unsafe once `p` and `q` were inserted.

---

## 11. Matrix Assembly of Mantle Blocks

The mantle blocks are inserted in `assemble.py`.

For the generalized eigenvalue problem:

$$
Ax=\lambda Bx,
$$

the mantle mass blocks are inserted into `B`:

```python
if ut.resolved_mantle:
    for k,l in enumerate(loc_mag_f):
        row = ut.offset_p + ( rank*bpp + k )*par.N_mantle
        col = row
        mtx = op.b_mantle(l, 'p', 'bpol_mantle', 0)
        loc_list = ut.packit(loc_list, mtx, row, col)

    for k,l in enumerate(loc_mag_g):
        row = ut.offset_q + ( rank*bpp + k )*par.N_mantle
        col = row
        mtx = op.b_mantle(l, 'q', 'btor_mantle', 0)
        loc_list = ut.packit(loc_list, mtx, row, col)
```

The mantle diffusion blocks are inserted into `A`:

```python
if ut.resolved_mantle:
    for k,l in enumerate(loc_mag_f):
        col0 = (rank*bpp + k )*par.N_mantle
        row  = ut.offset_p + col0
        col  = row

        mtx = op.magnetic_diffusion_mantle(l, 'p', 'bpol_mantle', 0)
        loc_list = ut.packit(loc_list, mtx, row, col)

    for k,l in enumerate(loc_mag_g):
        col0 = (rank*bpp + k )*par.N_mantle
        row  = ut.offset_q + col0
        col  = row

        mtx = op.magnetic_diffusion_mantle(l, 'q', 'btor_mantle', 0)
        loc_list = ut.packit(loc_list, mtx, row, col)
```

These blocks are diagonal in spherical harmonic degree. There is no direct `p` to `q` coupling in the mantle diffusion operator.

---

## 12. CMB Coupling Conditions

When:

```python
mantle = 'resolved'
```

Kore inserts core-mantle interface rows at:

$$
r=r_{\rm cmb}=1,
$$

we define:

$$
\mu_{m2o}=\mu_{\rm mantle},
$$

and:

$$
\eta_{m2o}
=
\frac{1}{\mu_{\rm mantle}\sigma_m(1)}.
$$

The first-pass poloidal coupling rows are:

$$
F_c - F_m = 0,
$$

and:

$$
\mu_{m2o}
\left(
F_c+rF_c'
\right)
-
\left(
F_m+rF_m'
\right)
=0.
$$

The first-pass toroidal coupling rows are:

$$
\mu_{m2o}G_c-G_m=0,
$$

and:

$$
G_c+rG_c'
-
\eta_{m2o}
\left(
G_m+rG_m'
\right)
=0.
$$

These are implemented in:

```python
def bc_b_mantle_cmb(l, loc):
```

with locations:

```python
'section_f'
'section_p'
'section_g'
'section_q'
```

The core rows use:

```python
bv.Tb
```

and the mantle rows use:

```python
bv.Fm0_cmb
bv.Fm1_cmb
bv.Gm0_cmb
bv.Gm1_cmb
```

The `section_f` and `section_g` rows occupy the existing core CMB boundary slots. If an inner core is present, these rows are shifted down by one row to avoid overwriting the ICB boundary rows.

The `section_p` and `section_q` rows occupy the first reserved row of each mantle block.

---

## 13. Outer Mantle Surface Conditions

At the mantle surface:

$$
r=r_{\rm surface},
$$

the exterior is assumed electrically insulating.

The toroidal field cannot exist in a current-free exterior, so:

$$
G_m(r_{\rm surface})=0.
$$

The poloidal field is matched to an external potential field. For the decaying external solution:

$$
F_{\rm ext}(r)\propto r^{-(\ell+1)}.
$$

Therefore:

$$
F_m'(r_{\rm surface})
+
\frac{\ell+1}{r_{\rm surface}}F_m(r_{\rm surface})
=0.
$$

Kore implements the equivalent form:

$$
r_{\rm surface}F_m'(r_{\rm surface})
+
(\ell+1)F_m(r_{\rm surface})
=0.
$$

The surface rows are implemented in:

```python
def bc_b_mantle_surface(l, loc):
```

with:

```python
out[0,:] = par.r_surface*bv.Fm1_surf + (l+1)*bv.Fm0_surf
```

for the poloidal mantle scalar, and:

```python
out[0,:] = bv.Gm0_surf
```

for the toroidal mantle scalar.

These rows occupy the second reserved row of each mantle block.

---

## 14. Boundary Row Layout

For every spherical harmonic degree in the mantle `p` block:

$$
\text{row 0}:
\quad
\mu_{m2o}(F_c+rF_c')-(F_m+rF_m')=0,
$$

$$
\text{row 1}:
\quad
r_{\rm surface}F_m'
+
(\ell+1)F_m=0.
$$

For every spherical harmonic degree in the mantle `q` block:

$$
\text{row 0}:
\quad
G_c+rG_c'
-
\eta_{m2o}(G_m+rG_m')=0,
$$

$$
\text{row 1}:
\quad
G_m(r_{\rm surface})=0.
$$

The corresponding rows in the mass matrix `B` must remain zero, because boundary and interface conditions are algebraic constraints, not time-evolution equations.

---

## 15. What Was Validated

The following structural validations were performed in the test repository after creating mantle submatrices with:

$$
N_{\rm mantle}=8.
$$

First, all mantle `.mtx` files had the correct shape:

```text
p20.mtx          rows=8 cols=8 nnz=41
peta000.mtx      rows=8 cols=8 nnz=45
peta101.mtx      rows=8 cols=8 nnz=38
peta202.mtx      rows=8 cols=8 nnz=24
q20.mtx          rows=8 cols=8 nnz=41
qeta000.mtx      rows=8 cols=8 nnz=45
qeta101.mtx      rows=8 cols=8 nnz=38
qeta110.mtx      rows=8 cols=8 nnz=45
qeta202.mtx      rows=8 cols=8 nnz=24
qeta211.mtx      rows=8 cols=8 nnz=33
```

Second, the first two rows of every mantle `.mtx` file were empty. This confirms that boundary rows were reserved correctly.

Third, the full matrices assembled successfully:

```text
A.npz (144672, 144672) nnz = 6463221
B.npz (144672, 144672) nnz = 1414116
```

Therefore:

$$
A,B \in \mathbb C^{144672\times 144672}.
$$

Fourth, 528 mantle boundary/interface rows were checked in `B`:

```text
checked boundary rows: 528
bad B boundary rows: []
```

This confirms:

$$
B_{\rm boundary}=0.
$$

The user also confirmed that the same boundary rows are nonzero in `A`, so the constraints are actually imposed.

---

## 16. Portability Fix for macOS NumPy

During assembly on macOS, NumPy raised:

```text
AttributeError: module 'numpy' has no attribute 'float128'
```

This came from older Kore induction formulas that used:

```python
np.float128
```

Some NumPy builds, especially on macOS, do not expose `float128`.

The fix in `operators.py` was:

```python
float_ext = getattr(np, 'float128', np.longdouble)
```

and:

```python
l = float_ext(l)
```

This preserves the intention of using the widest available real scalar, while remaining portable.

---

## 17. Code Map

### `bin/parameters.py`

Adds user-facing mantle parameters:

```python
# mantle = 'resolved'
r_surface = 1.8
N_mantle = 48
mu_mantle = 1.0
sigma_mantle_0 = 1e3
sigma_mantle_alpha = 1.0
```

### `bin/utils.py`

Adds:

```python
resolved_mantle
Nman
nman
offset_p
offset_q
conductivity_mantle
mag_diffus_mantle
dmag_diffus_mantle
```

Also expands:

```python
mantle_bc_list = ['insulator', 'TWA', 'resolved']
```

### `bin/bc_variables.py`

Adds CMB and surface Chebyshev evaluation rows for the mantle:

```python
Tm_cmb
Tm_surf
Fm0_cmb
Fm1_cmb
Gm0_cmb
Gm1_cmb
Fm0_surf
Fm1_surf
Gm0_surf
```

### `bin/submatrices.py`

Adds:

```python
write_mantle_magnetic_submatrices()
```

and writes `p*.mtx` and `q*.mtx` files.

### `bin/operators.py`

Adds:

```python
b_mantle()
magnetic_diffusion_mantle()
```

and expands the `.mtx` loader to recognize:

```python
p
q
```

### `bin/assemble.py`

Adds:

```python
bc_b_mantle_cmb()
bc_b_mantle_surface()
```

and inserts the mantle `p/q` blocks into `A` and `B`.

---

## 18. Scientific Interpretation

The resolved mantle implementation changes the CMB from a local magnetic boundary condition into an interface between two conducting domains.

For:

```python
mantle = 'insulator'
```

the core magnetic field is matched directly to a potential exterior at the CMB.

For:

```python
mantle = 'TWA'
```

the effect of a conducting layer is compressed into a thin-wall boundary condition.

For:

```python
mantle = 'resolved'
```

the mantle magnetic field has its own radial structure:

$$
F_m(r),
\qquad
G_m(r).
$$

The mantle can store and diffuse induced magnetic field. This should modify magnetic damping and phase lags relative to the insulating case. Qualitatively:

- Very low mantle conductivity should approach the insulating-CMB result.
- Larger mantle conductivity should increase the importance of magnetic diffusion in the mantle.
- Increasing `N_mantle` should lead to spectral convergence for smooth conductivity profiles.
- The toroidal field remains confined inside the conducting region and vanishes at the insulating surface.
- The poloidal field can penetrate into the insulating exterior through potential-field matching.

---

## Cell 19: Current Limitations

The current first-pass implementation is intentionally limited.

Not yet implemented:

- Mantle velocity.
- Mantle induction by mantle flow.
- Volume Lorentz-force feedback from mantle variables into the core momentum equation.
- A new resolved-mantle electromagnetic torque model.
- The additional diagnostic poloidal electric-field row that would make the interface treatment fully analogous to the conducting-inner-core formulation.

The combined state vector

$$
\text{conducting inner core}
+
\text{fluid core}
+
\text{resolved conducting mantle}
$$

is now structurally enabled in the matrix assembly. The previous temporary guard blocking

```python
resolved_mantle and cic
```

has been removed. The main row-placement change is that the resolved-mantle CMB poloidal row uses the third core `f` boundary slot when the Chebyshev conducting inner core is active, because the inner-core coupling already uses two `f` rows at the ICB.

This combined case should still be treated as a validation target rather than a finished physical production model until residual and convergence tests have been performed.

---

## Cell 20: Recommended Next Validation Steps

### 1. Backward compatibility

Run known cases with:

```python
mantle = 'insulator'
```

and:

```python
mantle = 'TWA'
```

and verify unchanged results relative to the original Kore behavior.

### 2. Small resolved-mantle eigenvalue test

Use:

```python
mantle = 'resolved'
innercore = 'insulator'
N_mantle = 8
thermal = 0
compositional = 0
rotdyn = 0
```

Check that eigenvalues can be computed and that eigenvectors have nontrivial `p/q` structure.

### 3. Conductivity limits

Vary:

```python
sigma_mantle_0
```

and compare against:

```python
mantle = 'insulator'
```

Expected qualitative behavior:

$$
\sigma_m \rightarrow 0
\quad
\Longrightarrow
\quad
\text{approach insulating behavior}.
$$

Large conductivity should alter damping and phase lags more strongly.

### 4. Mantle radial convergence

Run:

```python
N_mantle = 8, 12, 16, 24, ...
```

and check convergence of selected eigenvalues.

### 5. CMB residual diagnostics

For a computed eigenvector, reconstruct the residuals:

$$
F_c-F_m,
$$

$$
\mu_{m2o}(F_c+rF_c')-(F_m+rF_m'),
$$

$$
\mu_{m2o}G_c-G_m,
$$

$$
G_c+rG_c'-\eta_{m2o}(G_m+rG_m').
$$

These should be close to numerical precision.

### 6. Surface residual diagnostics

Check:

$$
r_{\rm surface}F_m'
+
(\ell+1)F_m,
$$

and:

$$
G_m.
$$

at:

$$
r=r_{\rm surface}.
$$

### 7. Full combined-system validation

Run the full:

$$
\text{conducting inner core}
+
\text{fluid core}
+
\text{resolved conducting mantle}
$$

case and check all ICB, CMB, and mantle-surface residuals.

---

## Cell 21: Summary

The resolved mantle implementation converts the mantle from a boundary approximation into an explicitly discretized magnetic diffusion domain. The mantle uses its own Chebyshev grid, its own conductivity profile, its own `p/q` magnetic scalars, and its own radial diffusion operators.

The key scientific change is:

$$
\text{CMB boundary condition}
\quad \rightarrow \quad
\text{CMB conducting interface}.
$$

The key numerical change is:

$$
\text{old global state vector}
\quad \rightarrow \quad
(u,v,f,g,f_{\rm ic},g_{\rm ic},p,q,h,i)^T.
$$

The key spectral change is:

$$
\text{core radial Chebyshev domain}
\quad +
\quad
\text{independent mantle Chebyshev shell}.
$$

The implementation now passes the first structural assembly tests:

- mantle submatrices have the correct shape,
- mantle operator boundary rows are reserved,
- full `A` and `B` assemble,
- `A` and `B` have matching global dimensions,
- mantle boundary rows are zero in `B`,
- mantle boundary rows are nonzero in `A`.

The next stage is physical validation through eigenvalue comparisons, conductivity-limit studies, residual diagnostics, and radial convergence tests.
