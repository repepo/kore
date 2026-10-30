# Conducting-Mantle Extension for Kore

## Complete Change Record, Matrix Specification, and Merge Guide

**Inventory date:** 29 October 2026.

**Scope:** the current `kore_mantle_diffusion` source tree, including the
initial resolved-mantle commit, subsequent working-tree corrections,
diagnostics, tests, plotting tools, and the research notebook.

This is an expanded English counterpart to
[the French merge guide](CONDUCTING_MANTLE_MERGE_GUIDE.md). It is intended for
the person reviewing or merging the extension into another Kore branch.
The French document has been retained.

The explanations distinguish four categories throughout:

- **Implemented:** directly visible in the current source.
- **Verified:** exercised by the tests or integration runs identified here.
- **Inherited:** already present in the selected pre-extension baseline.
- **Open:** not implemented generally, or not yet independently validated.

An implemented parameter is not, by itself, evidence that every physical
configuration it suggests is supported or validated.

## Contents

1. [Baseline, evidence, and historical scope](#1-baseline-evidence-and-historical-scope)
2. [Evolution from the initial code](#2-evolution-from-the-initial-code)
3. [Before-and-after architecture](#3-before-and-after-architecture)
4. [Physical model and nondimensionalization](#4-physical-model-and-nondimensionalization)
5. [Parameters and configuration contracts](#5-parameters-and-configuration-contracts)
6. [Conductivity, diffusivity, and conductance](#6-conductivity-diffusivity-and-conductance)
7. [Global unknown-vector layout](#7-global-unknown-vector-layout)
8. [Detailed row and column indexing](#8-detailed-row-and-column-indexing)
9. [Radial bases and submatrix generation](#9-radial-bases-and-submatrix-generation)
10. [Mantle differential operators](#10-mantle-differential-operators)
11. [Assembly call paths and matrix signs](#11-assembly-call-paths-and-matrix-signs)
12. [Exact interface and exterior rows](#12-exact-interface-and-exterior-rows)
13. [Worked numerical matrix example](#13-worked-numerical-matrix-example)
14. [Existing inner-core and rotation paths](#14-existing-inner-core-and-rotation-paths)
15. [Thin-wall corrections](#15-thin-wall-corrections)
16. [Eigenvalue and forced-solution extraction](#16-eigenvalue-and-forced-solution-extraction)
17. [Output-file contracts](#17-output-file-contracts)
18. [Complete electromagnetic torque calculation](#18-complete-electromagnetic-torque-calculation)
19. [G21 background-field derivative correction](#19-g21-background-field-derivative-correction)
20. [Energy diagnostics and radial quadrature](#20-energy-diagnostics-and-radial-quadrature)
21. [Independent audit programs](#21-independent-audit-programs)
22. [Portability and library-compatibility changes](#22-portability-and-library-compatibility-changes)
23. [Visualization and field reconstruction](#23-visualization-and-field-reconstruction)
24. [Research notebook and experiment isolation](#24-research-notebook-and-experiment-isolation)
25. [Experiments currently implemented](#25-experiments-currently-implemented)
26. [Tests and available validation evidence](#26-tests-and-available-validation-evidence)
27. [Limitations and unresolved issues](#27-limitations-and-unresolved-issues)
28. [File-by-file merge inventory](#28-file-by-file-merge-inventory)
29. [Recommended merge procedure](#29-recommended-merge-procedure)
30. [Post-merge verification and rebuild rules](#30-post-merge-verification-and-rebuild-rules)
31. [Function and call-site reference](#31-function-and-call-site-reference)
32. [Final acceptance checklist](#32-final-acceptance-checklist)

## 1. Baseline, Evidence, and Historical Scope

### 1.1 Exact local Git references

| Item | Reference |
| --- | --- |
| Configured remote | `https://github.com/repepo/kore.git` |
| Working branch | `alex_experiments` |
| Pre-extension local branch | `conductive_IC` |
| Pre-extension commit | `5fa15664c7ee517965ac9ff90357c3d8804ae627` |
| Initial resolved-mantle commit | `9ab88971c7942ed393c93d993548e706cd5d767b` |
| State documented | Initial extension commit **plus current uncommitted and untracked work** |

These are local references inspected for this document. The remote repository
was not fetched or updated as part of the documentation task.

The baseline already supports an insulating mantle boundary and a thin-wall
approximation, or TWA. Consequently, “the original version without a
conducting mantle” means **without an explicitly resolved conducting mantle
domain**, not without any representation of mantle electrical conductance.

The baseline also contains conducting-inner-core developments and axial
rotational dynamics. Those features must not be attributed to the new mantle
extension. Their indexing and output extraction have nevertheless been
updated to coexist with the added mantle blocks.

### 1.2 What “from the very beginning” can establish

The complete, reproducible comparison is:

```text
pre-extension baseline 5fa15664
    -> initial extension commit 9ab88971
    -> subsequent corrections and tools in the current working tree
```

The research workflow also passed through intermediate torque diagnostics,
scripts, notebook arrangements, and parameter experiments. Some of those
intermediate versions are no longer present as source files or individual
commits. They are described where their role is known, but this document does
not invent exact historical diffs, dates, or function signatures for missing
files.

In particular, old `mantle_wave_postprocessing/...` runners discussed during
development are not dependencies of the current notebook and are not present
in the current tree. Recovering an older script would require a separate
source audit before including it in a merge.

### 1.3 Why the initial commit is insufficient

The initial commit adds the mantle domain and several dependent changes, but
it does not include all corrections now used in the research calculations.
Important later changes include:

- the forced-frequency term in the mantle equations;
- the revised TWA poloidal condition;
- corrected G21 second and third derivatives;
- complete, rather than partial, Cartesian Maxwell torques;
- Gauss-Legendre energy integration;
- independent audit tools and analytical tests;
- the current standalone benchmark notebook.

**Cherry-picking `9ab88971` alone does not reproduce the current implementation.**

### 1.4 Commands that reproduce the source comparison

```bash
git status --short
git log --oneline --decorate -12
git diff --stat 5fa15664 -- bin koreviz koreviz_mod tools
git diff 5fa15664 -- bin/utils.py bin/submatrices.py bin/operators.py
git diff 5fa15664 -- bin/bc_variables.py bin/assemble.py bin/solve_nopp.py
git diff HEAD -- bin/utils.py bin/assemble.py bin/utils4pp.py
git ls-files --others --exclude-standard
```

Ordinary `git diff` does not include untracked file contents. Tests, tools,
documentation, and notebooks must therefore be inventoried separately.

## 2. Evolution from the Initial Code

### 2.1 Stage A: insulating or thin-wall outer boundary

In the baseline, the magnetic field is solved in the fluid core and, when
enabled, the solid inner core. The mantle is represented through a boundary
condition at the CMB:

```python
mantle = 'insulator'
# or
mantle = 'TWA'
```

There are no mantle radial coefficients, no mantle mass or diffusion blocks,
and no output files containing a mantle eigenfunction.

For the TWA, conductance and wall properties enter through `c_cmb` and
`c1_cmb`; changing these modifies the CMB condition without adding a domain.

### 2.2 Stage B: a separate conducting mantle domain

The initial extension introduces:

```python
mantle = 'resolved'
```

Two magnetic scalar fields are added in a shell outside the CMB. They have
their own Chebyshev basis, radial resolution, conductivity profile, interface
conditions, and insulating exterior condition.

The global vector is enlarged, and the assembly and solver readers are
updated together. This is the central architectural change: it is not a
post-processing extension or an extrapolation of the core field.

### 2.3 Stage C: coherent dimensions, outputs, and portability

The extension centralizes block sizes and offsets; updates both eigenmode
and forced-solution extraction; adds mantle field files; extends selected
visualizers; and includes several sparse-matrix, special-function, and
scalar-extraction compatibility fixes.

These changes are important even in configurations where the mantle is
disabled, because the same code paths now consume the centralized offsets.

### 2.4 Stage D: physical and numerical corrections

Subsequent checking identified distinct issues that must not be confused:

| Issue | Nature | What must be recomputed |
| --- | --- | --- |
| Missing harmonic mass term in mantle forced rows | Incorrect forced operator | Assemble and solve again |
| TWA radial denominator and curvature expression | Changed boundary model | Assemble and solve again |
| Wrong G21 derivative signs | Incorrect background-dependent operators and diagnostics | Regenerate submatrices, assemble, and solve again |
| Incomplete equatorial Maxwell expression | Incomplete torque diagnostic | Recompute torque diagnostics from compatible fields |
| Inaccurate energy quadrature for nearly rigid flows | Diagnostic integration error | Recompute diagnostics only |

Small TWA/resolved eigenvalue differences are not proof that their magnetic
responses are identical. Conversely, a large energy residual is not always
a spatial-resolution failure: in the audited nearly rigid flow, the
quadrature itself was responsible for the apparent mechanical discrepancy.

### 2.5 Stage E: research experiments and interpretation safeguards

The current notebook replaces ad hoc comparisons with isolated experiments:
matched thickness and conductance, fixed-conductance thickness sweeps,
exponential-profile sweeps, kinetic-overlap mode tracking, normalized complex
torques, and resolution checks.

An intermediate surface-torque calculation retained only $B_{0r}\mathbf b_t$.
It has been superseded by the complete linear expression. Intermediate plots
also showed local insulating Maxwell traction as though it were torque
deposited in mantle material. The current notebook excludes that profile
from torque plots while retaining the integrated cancellation as a check.

## 3. Before-and-After Architecture

### 3.1 Functional overview

| Component | Baseline | Current implementation |
| --- | --- | --- |
| Mantle model choices | `insulator`, `TWA` | Adds `resolved` |
| Mantle magnetic unknowns | None | Poloidal `p`, toroidal `q` |
| Mantle geometry | Reduced to CMB condition | Explicit shell $[1,r_s]$ |
| Mantle radial resolution | None | `N_mantle` |
| Conductivity distribution | Wall conductance parameters | Radial exponential profile, including uniform limit |
| Exterior boundary | At CMB for insulating core boundary | At `r_surface` for resolved mantle |
| Unknown indexing | Repeated arithmetic expressions | Shared sizes and offsets in `utils.py` |
| Solver | PETSc/SLEPc | Same solver, enlarged matrices and extraction |
| Core magnetic files | Core coefficients | Still core coefficients only |
| Mantle magnetic files | Absent | Separate real/imaginary mantle files |
| CMB torque | Primarily axial spectral diagnostic | Adds full Cartesian surface calculation and profiles |
| Energy quadrature in `utils4pp` | Corrected-weight Chebyshev sampling | Unit-weight Gauss-Legendre at `2*N` |
| Experiment management | Existing scripts/manual setup | Isolated snapshots, manifests, candidate tracking |

### 3.2 High-level execution graph

```text
parameters.py
    |
    +--> utils.py: global sizes, offsets, profiles, conventions
    |
    +--> submatrices.py
    |        |
    |        +--> existing core / inner-core submatrices
    |        +--> write_mantle_magnetic_submatrices() when resolved
    |                 |
    |                 +--> p*.mtx, q*.mtx
    |
    +--> assemble.py
    |        |
    |        +--> operators.py: load submatrices and combine operators
    |        +--> bc_variables.py: interface evaluation vectors
    |        +--> A.npz and B.npz, or forced operator / RHS
    |
    +--> solve_nopp.py
    |        |
    |        +--> SLEPc EPS or PETSc KSP
    |        +--> slice solution using utils offsets
    |        +--> core, inner-core, mantle, thermal, composition files
    |
    +--> spin_doctor.py / explicit notebook analyzer
             |
             +--> utils4pp.diagnose(): volume diagnostics
             +--> utils.gamma_magnetic_cmb_distribution(): surface torque
             +--> diagnostics, mode matching, figures, exports
```

The notebook calls these programs in subprocesses. It does not replace their
domain equations with a separate reduced solver.

## 4. Physical Model and Nondimensionalization

### 4.1 Domains and reference scales

Let $R=R_{\rm CMB}$. The code uses:

$$
r_{\rm CMB}=1,\qquad r_{\rm ICB}=\texttt{ricb},\qquad
1\le r\le r_s=\texttt{r_surface}.
$$

The fluid core remains the mechanical domain. The mantle is electrically
conducting but stationary in the reference frame used by this extension.
It has no added velocity, pressure, temperature, elastic, or convective
unknowns.

The exterior beyond $r_s$ is insulating. Its potential field is eliminated
analytically through a boundary condition, rather than discretized.

### 4.2 Linear perturbations

The fluid and mantle magnetic perturbations evolve around Kore's imposed
background field. For free modes:

$$
\mathbf b(\mathbf r,t)=\Re\left[\widehat{\mathbf b}(\mathbf r)e^{\lambda t}\right],
\qquad \lambda=\sigma+i\omega.
$$

A stable mode has $\sigma<0$. The positive decay rate used in comparisons is
$\gamma=-\sigma$.

The extension does not solve a nonlinear dynamo or construct a new
self-consistent magnetic background throughout the planet. It solves the
linear perturbation in an added conducting region.

### 4.3 Dimensionless controls

For `OmgTau = 1`:

$$
E=\frac{\nu}{\Omega R^2},\qquad
E_m=\frac{\eta_f}{\Omega R^2},\qquad
Le=\frac{B_*}{\Omega R\sqrt{\rho\mu_f}},
$$

$$
Pm=\frac E{E_m},\qquad \Lambda=\frac{Le^2}{E_m},\qquad Lu=\frac{Le}{E_m}.
$$

The source retains `OmgTau` explicitly. It must not be removed merely because
the present benchmarks set it to one. A comparison made with another time
scale must account for that choice consistently in operators, frequencies,
energies, and torques.

### 4.4 Magnetic scalar convention

For $L=\ell(\ell+1)$:

$$
b_r=\frac Lr f_\ell Y_\ell^m,
$$

$$
\mathbf b_t=\left(f_\ell'+\frac{f_\ell}{r}\right)\nabla_\Omega Y_\ell^m
+g_\ell\left(\nabla_\Omega Y_\ell^m\times\hat{\mathbf r}\right).
$$

The mantle uses exactly the same scalar convention, but its coefficients
belong to another radial interval. Conditions written for $rf_\ell$ must not
be copied directly onto $f_\ell$ without transformation.

### 4.5 Verified permeability scope

The controlled benchmarks use equal, unit relative permeabilities. Although
`mu_mantle` appears in interface expressions, the current bulk diffusivity
function returns $1/s$, without an explicit mantle-permeability division.
General permeability contrasts therefore require an additional consistency
audit and independent validation. They are not certified by the present tests.

## 5. Parameters and Configuration Contracts

### 5.1 New or clarified inputs

| Parameter | Meaning | Active path |
| --- | --- | --- |
| `mantle` | Model selector: `insulator`, `TWA`, `resolved` | Magnetic problem |
| `r_surface` | Outer radius of the mantle conductor in CMB-radius units | resolved |
| `N_mantle` | Chebyshev coefficients per mantle scalar and degree | resolved |
| `sigma_mantle_0` | CMB conductivity relative to reference fluid conductivity | resolved |
| `sigma_mantle_alpha` | Exponential radial decay coefficient | resolved |
| `mu_mantle` | Mantle/fluid permeability parameter at interfaces | resolved; general contrasts not validated |
| `muw_cmb` | TWA wall permeability relative to vacuum | CMB TWA |
| `muw_icb` | Corresponding inner-wall quantity | ICB TWA |
| `c_cmb`, `c1_cmb` | Existing TWA inputs with clarified interpretation | CMB TWA |

The mantle-model diagnostic index now supports `2 = resolved`, in addition
to `0 = insulator` and `1 = TWA`. External readers that assume a binary
choice must be updated.

### 5.2 Physics versus local defaults

The currently edited `parameters.py` also changes several values from the
baseline: `m`, `symm`, `Ek`, magnetic activation, field normalization,
mechanical conditions, resolutions, eigenvalue target, and solver tolerance.

These are **experiment settings**, not all required changes to the Kore API.
When merging, introduce the new parameter definitions while preserving or
deliberately choosing the destination branch's defaults.

### 5.3 Derived inputs must agree

Two equivalent input patterns are supported by the parameter conventions:

```python
# Elsasser / magnetic Prandtl formulation
Em = Ek / Pm
Le2 = Lambda * Em
Le = np.sqrt(Le2)
```

```python
# Lehnert / magnetic Ekman formulation
Le2 = Le**2
Lambda = Le2 / Em
Pm = Ek / Em
```

Only one set should actively define the final values. The extension does not
introduce a global configuration validator that prevents later assignments
from overwriting earlier ones.

### 5.4 Required parameter guards

For resolved calculations, require a positive shell thickness, a meaningful
radial resolution, finite conductivity, and strictly positive conductivity
throughout the conductor. An insulating mantle must be selected by its model
name, not by setting `sigma_mantle_0 = 0` in the resolved diffusion operator.

The notebook includes more parameter checks than the low-level routines.
Do not assume every direct script call rejects all invalid combinations.

## 6. Conductivity, Diffusivity, and Conductance

### 6.1 Profile functions added to `utils.py`

```python
conductivity_mantle(r)
mag_diffus_mantle(r)
dmag_diffus_mantle(r)
```

They implement:

$$
s(r)=s_0e^{-\alpha(r-1)},\qquad
\eta_m(r)=\frac1{s(r)},\qquad \eta_m'(r)=\alpha\eta_m(r).
$$

These are dimensionless conductivity and inverse-conductivity functions,
not SI conductivity or diffusivity values. The current fluid conductivity
function returns unity; a future nonuniform-core profile must also be
checked against interface assumptions.

### 6.2 Total conductance

With $D=r_s-1$:

$$
G^*=\int_1^{r_s}s(r)\,dr=\frac{G}{\sigma_fR},
$$

$$
G^*=\begin{cases}
s_0D,&\alpha=0,\\
s_0\dfrac{1-e^{-\alpha D}}\alpha,&\alpha>0.
\end{cases}
$$

At prescribed $G^*$, the notebook solves this relation for $s_0$ using
`-np.expm1(-alpha*D)` near the uniform limit and checks the resulting
integral numerically.

### 6.3 Matching a wall and a resolved profile

At unit relative permeability:

$$
c_1=G^*,\qquad c=h/R.
$$

The configuration comment more generally defines
$c=(h/R)(\mu_w/\mu_f)$. Conductance alone does not determine $h$ or $c$.

For an exponential distribution:

$$
\frac{d_{\rm eff}}R=\frac{G^*}{s_0}.
$$

This is a measure of basal concentration, not necessarily the geometric
shell thickness or a material interface.

### 6.4 No hidden automatic matching in the core solver

The example conductance-matching formula in `parameters.py` is enclosed in
a triple-quoted string. It is not executed. Core Kore does not automatically
make `c1_cmb` agree with a resolved profile.

The notebook performs the matching explicitly when creating each case.
Activating a similar formula in a standalone parameter file requires placing
it after the definitions it depends on.

## 7. Global Unknown-Vector Layout

### 7.1 Original size expression

The pre-extension code uses:

```python
sizmat = (2*n*par.hydro + 2*n*par.magnetic + 3*par.rotdyn
          + 2*nic*cic + n*par.thermal + n*par.compositional)
```

Many consumers separately reconstruct offsets with expressions such as
`2*n`, `3*n`, `4*n`, and sums of active-field sizes. This becomes fragile
when extra domains are inserted before later fields.

### 7.2 Current shared size definitions

```python
resolved_mantle = par.magnetic == 1 and par.mantle == 'resolved'
Nman = par.N_mantle if resolved_mantle else 0
nman = int(Nman * (par.lmax - par.m + 1) / 2) if resolved_mantle else 0

size_u = n * par.hydro
size_v = n * par.hydro
size_f = n * par.magnetic
size_g = n * par.magnetic
size_rotdyn = 3 * par.rotdyn
size_fic = nic * cic
size_gic = nic * cic
size_p = nman
size_q = nman
size_h = n * par.thermal
size_i = n * par.compositional
```

The existing definitions of `N1`, `Nic`, `n`, `nic`, and `n0` retain the
origin-parity conventions for the core and inner core.

### 7.3 Current order and recurrence

$$
\mathbf x=(\mathbf u,\mathbf v,\mathbf f,\mathbf g,
\mathbf r_{\rm axial},\mathbf f_{ic},\mathbf g_{ic},
\mathbf p,\mathbf q,\mathbf h,\mathbf i)^T.
$$

| Block | Start | End, exclusive |
| --- | --- | --- |
| Core poloidal velocity | `offset_u` | `offset_u + size_u` |
| Core toroidal velocity | `offset_v` | `offset_v + size_v` |
| Core poloidal magnetic field | `offset_f` | `offset_f + size_f` |
| Core toroidal magnetic field | `offset_g` | `offset_g + size_g` |
| Existing axial rotation variables | `offset_rotdyn` | `offset_rotdyn + size_rotdyn` |
| Inner-core poloidal magnetic field | `offset_fic` | `offset_fic + size_fic` |
| Inner-core toroidal magnetic field | `offset_gic` | `offset_gic + size_gic` |
| Mantle poloidal magnetic field | `offset_p` | `offset_p + size_p` |
| Mantle toroidal magnetic field | `offset_q` | `offset_q + size_q` |
| Temperature | `offset_h` | `offset_h + size_h` |
| Composition | `offset_i` | `offset_i + size_i` |

Starting with `offset_u = 0`, each subsequent offset is the previous offset
plus the previous size. The total size is `offset_i + size_i`.

The letters `p` and `q` in these new blocks denote mantle magnetic scalars;
they do not introduce a mantle pressure unknown.

### 7.4 Disabled domains have zero size

When the mantle is not resolved, `size_p = size_q = 0`. Later fields then
occupy their non-mantle positions without a separate indexing branch.

When magnetic equations are disabled, `resolved_mantle` is also false. A
mantle parameter name alone must not enlarge a hydrodynamic-only problem.

### 7.5 Import-time state

These globals are evaluated when `utils.py` is imported. Changing a parameter
module already loaded into a notebook does not reliably rebuild all related
globals in other imported modules. The experiment notebook uses fresh
processes and case-specific parameter copies to avoid that problem.

## 8. Detailed Row and Column Indexing

### 8.1 Block sizes per harmonic degree

Let:

$$
n_b=\frac{\ell_{\max}-m+1}{2},\qquad
n=N_1n_b,\qquad n_m=N_{\rm mantle}n_b.
$$

The angular degrees of the mantle follow the core magnetic field. However,
their radial coefficient count is `N_mantle`, not `N1`.

The assembly distributes angular blocks over MPI ranks. If `rank` is the
rank index and `bpp` the number of blocks per rank, define:

$$
j=\texttt{rank}\,\texttt{bpp}+k.
$$

The first global row of a local block is:

```python
core_f_row = ut.offset_f + j * ut.N1
core_g_row = ut.offset_g + j * ut.N1
mantle_p_row = ut.offset_p + j * par.N_mantle
mantle_q_row = ut.offset_q + j * par.N_mantle
```

For diagonal differential operators, the starting column equals the starting
row. Interface rows instead place coefficients in both the core and mantle
column blocks.

### 8.2 Before/after expressions used in assembly

Here `n = nb*N1` and option flags have their existing meanings.

| Purpose | Previous expression | Current expression |
| --- | --- | --- |
| Core magnetic poloidal block start | `hydro*(2*nb*N1)` | `ut.offset_f` |
| Core magnetic toroidal block start | `(2*hydro+1)*nb*N1` | `ut.offset_g` |
| MHD Lorentz poloidal-field column | `2*nb*N1` | `ut.offset_f` |
| MHD Lorentz toroidal-field column | `3*nb*N1` | `ut.offset_g` |
| Axial rotational block | `4*nb*N1` in several MHD paths | `ut.offset_rotdyn` |
| Inner-core poloidal block | `2*(hydro+magnetic)*nb*N1 + 3*rotdyn` | `ut.offset_fic` |
| Inner-core toroidal block | Previous start plus `nb_cic*Nic` | `ut.offset_gic` |
| Temperature block | Several expressions based on `2*(hydro+magnetic)*nb*N1` | `ut.offset_h` |
| Composition block | Previous start plus active thermal size | `ut.offset_i` |
| Mantle poloidal block | Absent | `ut.offset_p` |
| Mantle toroidal block | Absent | `ut.offset_q` |

The old expressions were not all identical across consumers. Centralization
removes that duplication; it does not mean every old expression was a fully
general formula for all flag combinations.

### 8.3 Rows updated in B

The mass-matrix assembly now uses centralized offsets for:

1. core magnetic poloidal rows;
2. core magnetic toroidal rows;
3. conducting-inner-core magnetic rows;
4. the three inherited axial rotational variables;
5. temperature rows;
6. composition rows;
7. the newly inserted mantle poloidal and toroidal rows.

The velocity mass rows are retained under the existing core convention.

### 8.4 Rows and columns updated in A

Affected call sites include:

- magnetic-field columns of both poloidal and toroidal momentum equations;
- temperature and composition columns in the buoyancy terms;
- core magnetic equation row starts and magnetic diffusion columns;
- CMB and ICB boundary-condition row locations;
- mantle diffusion and forced-frequency rows;
- conducting-inner-core differential equation rows;
- axial torque, gravitational torque, and relaxation rows;
- temperature and composition differential equations.

There are still some inherited local expressions for velocity columns and
other specialized paths. This extension centralizes the changed domain
layout; it is not a complete rewrite of every index expression in Kore.

### 8.5 Interface row ownership

The mantle equation supplies two boundary rows per scalar. Core magnetic
blocks also retain their pre-existing boundary rows. The interface is split
between them rather than adding an extra independent constraint block.

This arrangement is essential to preserve a square system with the correct
number of conditions. Adding the same interface condition twice, or adding
extra rows without extra unknowns, would alter the eigenproblem.

## 9. Radial Bases and Submatrix Generation

### 9.1 Functions introduced in `submatrices.py`

```python
_shift_mantle_bc_rows(matrix, Nm)
write_mantle_magnetic_submatrices()
```

`main(ncpus)` calls the second function only when `ut.resolved_mantle` is true.

### 9.2 Mapping of the mantle interval

The mantle Chebyshev coordinate is:

$$
x_m=\frac{2r-(1+r_s)}{r_s-1},\qquad
\partial_r=\frac{2}{r_s-1}\partial_{x_m}.
$$

Its coefficient vectors have length `N_mantle`. The core uses its own
interval and derivative scaling. Reusing the core derivative matrix for the
mantle would therefore be incorrect even if the two resolutions were equal.

### 9.3 Basis conversion and differentiation

The new construction uses:

```python
S0m = ut.Slam(0, Nm)
S1m = ut.Slam(1, Nm)
S10m = S1m * S0m
D1m = ut.Dlam(1, Nm, 1, par.r_surface)
D2m = ut.Dlam(2, Nm, 1, par.r_surface)
```

Products are represented in the appropriate ultraspherical bases before
conversion to the common $C^{(2)}$ equation basis.

### 9.4 Exact building blocks

Let $M_\nu[a]$ denote multiplication by a represented profile in basis
$C^{(\nu)}$. The implemented constructions are:

| Internal label | Implemented matrix expression |
| --- | --- |
| `20` | `S10m * Mlam(r2m, 0, 0)` |
| `eta000` | `S10m * Mlam(eta0m, 0, 0)` |
| `eta101` | `S1m * Mlam(S0m * reta0m, 1, 0) * D1m` |
| `eta202` | `Mlam(S10m * r2eta0m, 2, 0) * D2m` |
| `eta110` | `S10m * Mlam(reta1m, 0, 0)` |
| `eta211` | `S1m * Mlam(S0m * r2eta1m, 1, 0) * D1m` |

The profiles are generated through `chebco` or `chebco_f`, using `tol_tc`.
The scalar equation then combines these building blocks with angular factors.

### 9.5 Tau-row shift

For an unshifted $N_m\times N_m$ operator $K$:

$$
\widetilde K=
\begin{pmatrix}
\mathbf0\\
\mathbf0\\
K_{0,:}\\
\vdots\\
K_{N_m-3,:}
\end{pmatrix}.
$$

The implementation stacks two zero rows above `matrix[:-2, :]`. It does not
merely overwrite the first two original rows: the differential rows are
shifted downward, and the last two original rows are omitted.

Both mass and diffusion building blocks use the same row shift. Consequently,
the first two mantle rows carry constraints with zero temporal mass.

### 9.6 Generated files and operator variable names

| Generated file | Variable loaded by `operators.py` | Role |
| --- | --- | --- |
| `p20.mtx` | `r2Ip` | Mantle poloidal mass |
| `q20.mtx` | `r2Iq` | Mantle toroidal mass |
| `peta000.mtx` | `etaIp` | Poloidal multiplication by $\eta$ |
| `qeta000.mtx` | `etaIq` | Toroidal multiplication by $\eta$ |
| `peta101.mtx` | `retaD1p` | Poloidal $r\eta\partial_r$ |
| `qeta101.mtx` | `retaD1q` | Toroidal $r\eta\partial_r$ |
| `peta202.mtx` | `r2etaD2p` | Poloidal $r^2\eta\partial_r^2$ |
| `qeta202.mtx` | `r2etaD2q` | Toroidal $r^2\eta\partial_r^2$ |
| `qeta110.mtx` | `reta1Iq` | Toroidal $r\eta'$ |
| `qeta211.mtx` | `r2eta1D1q` | Toroidal $r^2\eta'\partial_r$ |

The profile-loader section list in `operators.py` now includes `i`, `p`, and
`q`, in addition to the previous sections. Stale `.mtx` files in a shared
working directory can therefore be significant; isolated case directories
are preferable.

## 10. Mantle Differential Operators

### 10.1 Added mass operator

```python
b_mantle(l, section, component, offdiag)
```

Supported diagonal calls are:

```python
op.b_mantle(l, 'p', 'bpol_mantle', 0)
op.b_mantle(l, 'q', 'btor_mantle', 0)
```

They return $Lr^2$ in the respective radial representations. No velocity
source is added in this function.

### 10.2 Added diffusion operator

```python
magnetic_diffusion_mantle(l, section, component, offdiag)
```

The two calls are:

```python
op.magnetic_diffusion_mantle(l, 'p', 'bpol_mantle', 0)
op.magnetic_diffusion_mantle(l, 'q', 'btor_mantle', 0)
```

They implement:

$$
D_p=\Omega\tau E_mL
\left[-L\eta+2r\eta\partial_r+r^2\eta\partial_r^2\right],
$$

$$
D_q=\Omega\tau E_mL
\left[-L\eta+2r\eta\partial_r+r^2\eta\partial_r^2
+r\eta'+r^2\eta'\partial_r\right].
$$

### 10.3 Scalar differential equations

After dividing by the mass factor:

$$
\lambda f_m=\Omega\tau E_m\eta
\left(f_m''+\frac2r f_m'-\frac L{r^2}f_m\right),
$$

$$
\lambda g_m=\Omega\tau E_m
\left[\eta\left(g_m''+\frac2r g_m'-\frac L{r^2}g_m\right)
+\eta'\left(g_m'+\frac{g_m}r\right)\right].
$$

The derivative-of-diffusivity terms are part of the toroidal equation. They
must not be removed by replacing the radial conductivity with an average
conductivity or by using the constant-diffusivity operator unchanged.

### 10.4 No direct volume coupling from mantle to velocity

The new mantle rows contain no bulk velocity-induction term. The existing
core momentum rows do not acquire a direct volume Lorentz block involving
mantle coefficients. The coupling is mediated by the interface conditions
and the resulting magnetic field in the core.

## 11. Assembly Call Paths and Matrix Signs

### 11.1 Model-selection path for core boundary rows

For the poloidal core equation, the outer boundary dispatch is now:

```python
if par.mantle == 'TWA':
    bc_b_list_outer = bc_b_thinlayer(
        l, 'nocurl', 1/par.mu, par.c_cmb, par.c1_cmb, 'cmb')
elif ut.resolved_mantle:
    bc_b_list_outer = bc_b_mantle_cmb(l, 'section_f')
else:
    bc_b_list_outer = bc_b_cmb(l, 'section_f', par.innercore)
```

For the toroidal core equation, the corresponding calls use `'1curl'`,
`'section_g'`, and the toroidal boundary expressions.

The old insulating or TWA CMB row is **replaced**, not retained in addition
to the resolved-mantle interface condition.

### 11.2 New mantle rows in B

For every mantle poloidal harmonic:

```python
row = ut.offset_p + j * par.N_mantle
col = row
mtx = op.b_mantle(l, 'p', 'bpol_mantle', 0)
loc_list = ut.packit(loc_list, mtx, row, col)
```

The toroidal path is identical with `offset_q`, section `'q'`, and component
`'btor_mantle'`.

### 11.3 New mantle rows in A

For the poloidal block:

```python
mtx = op.magnetic_diffusion_mantle(l, 'p', 'bpol_mantle', 0)
if par.forcing != 0:
    mtx = mtx - 1j*ut.wf*op.b_mantle(l, 'p', 'bpol_mantle', 0)
loc_list = ut.packit(loc_list, mtx, row, col)

bc_cmb = bc_b_mantle_cmb(l, 'section_p')
bc_surface = bc_b_mantle_surface(l, 'section_p')
```

The toroidal block follows the same pattern with `'q'` and `'section_q'`.
The returned sparse triplets are appended to the assembly lists.

### 11.4 Sign convention is row-dependent

The eigenproblem remains:

$$
A\mathbf x=\lambda B\mathbf x.
$$

| Interior equation | A contribution | B contribution |
| --- | --- | --- |
| Core magnetic diffusion | $-D_f$ | $-M_f$ |
| Mantle magnetic diffusion | $+D_m$ | $+M_m$ |

These two row orientations both describe positive-time diffusion with
$\lambda M\mathbf b=D\mathbf b$. A merge must not change one sign without
changing the corresponding equation consistently.

For the harmonic forced problem, the mantle operator is:

$$
D_m-i\omega_fM_m.
$$

The core induction rows use the opposite row orientation,
$i\omega_fM_f-D_f$ together with their induction terms. The mantle volume
RHS remains zero; excitation enters through coupling to the core.

### 11.5 Forced-frequency correction after the initial extension

The initial mantle assembly included diffusion but omitted the harmonic
mass term in a forced run. The current code adds the `-1j*ut.wf*...` term to
both mantle scalar blocks when `forcing != 0`.

Without it, a time-dependent core forcing would be coupled to a mantle
treated as stationary. This is a physical correction, not merely a solver
option or plotting change.

### 11.6 Boundary rows and singular B

The mantle boundary rows are algebraic constraints. Their B rows remain
zero because the shifted mass operator has zero leading rows.

A singular mass matrix is expected in this tau formulation. Filling boundary
mass rows to make B invertible would change the generalized eigenproblem.

### 11.7 Common scaling of A and B

The eigenvalue assembly retains the common normalization by `Bnorm`.
Comparing a physical-space operator directly to a saved A block without
removing this normalization can generate an apparent multiplicative error.

The induction audit recovers the shared scaling from a known magnetic mass
column of B before comparing projected physical-space induction with A.

### 11.8 Global block pattern in the basic MHD shell

For the illustrative case with no conducting inner core, no rotation
unknowns, and no thermal or compositional fields, the new vector is
$(\mathbf u,\mathbf v,\mathbf f,\mathbf g,\mathbf p,\mathbf q)$.
The structural pattern is:

```text
                    columns
             u      v      f      g      p      q
          +------+------+------+------+------+------+
  u rows  | Uu   | Uv   | Luf  | Lug  |  0   |  0   |
  v rows  | Vu   | Vv   | Lvf  | Lvg  |  0   |  0   |
  f rows  | Ifu  | Ifv  | Af   |  0   | Cfp  |  0   |
  g rows  | Igu  | Igv  |  0   | Ag   |  0   | Cgq  |
  p rows  |  0   |  0   | Cpf  |  0   | Ap   |  0   |
  q rows  |  0   |  0   |  0   | Cqg  |  0   | Aq   |
          +------+------+------+------+------+------+
```

Here:

- `Uu/Uv/Vu/Vv` are inherited momentum operators, including rotation and
  viscous contributions;
- `Luf/Lug/Lvf/Lvg` are the inherited core Lorentz couplings;
- `Ifu/Ifv/Igu/Igv` are the core induction couplings;
- `Af/Ag` contain core diffusion and their relevant boundary rows;
- `Ap/Aq` contain mantle diffusion and mantle-side boundary rows;
- `Cfp/Cpf/Cgq/Cqg` are **interface-row entries only**, not bulk source terms.

The corresponding B pattern is block diagonal, with inherited velocity
blocks, the existing negative core magnetic mass blocks, and positive mantle
mass blocks. All appropriate constraint rows in those mass blocks are zero.

The displayed A diagonal blocks are not pure differential matrices on every
row: their reserved rows contain the diagonal pieces of boundary conditions.
This distinction matters when inspecting the sparsity plot or extracting an
interior operator for a regression test.

The baseline is recovered by removing p/q rows and columns and restoring the
selected insulating or TWA condition in the core's outer boundary rows.

## 12. Exact Interface and Exterior Rows

### 12.1 Boundary-evaluation vectors

`bc_variables.py` adds:

```python
Tm_cmb = Tk(-1, par.N_mantle-1, 4, ut.rcmb, par.r_surface)
Tm_surf = Tk(1, par.N_mantle-1, 4, ut.rcmb, par.r_surface)
```

The exported vectors include:

```text
Fm0_cmb, Fm1_cmb, Fm2_cmb
Gm0_cmb, Gm1_cmb
Fm0_surf, Fm1_surf
Gm0_surf
```

The core-side vectors already exist as columns of `bv.Tb`. Mantle and core
derivatives use their own interval scaling.

### 12.2 General index definitions

For a poloidal degree $\ell$:

```python
core_f = ut.offset_f + ut.N1 * ((l - mf)//2)
mantle_p = ut.offset_p + par.N_mantle * ((l - mf)//2)
```

For a toroidal degree:

```python
core_g = ut.offset_g + ut.N1 * ((l - mg)//2)
mantle_q = ut.offset_q + par.N_mantle * ((l - mg)//2)
```

The actual implementation uses `int(...)`. The appropriate starting degrees
`mf` and `mg` depend on `ut.symmB0`, `ut.m_top`, and `ut.m_bot`.

In a shell:

```python
delta_row_g = 1
delta_row_f = 1 + int(bool(ut.cic))
```

Without an ICB, the corresponding base shift is zero. The extra poloidal
row for a conducting inner core must remain reserved.

### 12.3 CMB row specification

Define $r_j=1$, $\mu_r=\texttt{mu_mantle}$, and
$\eta_r=\texttt{mag_diffus_mantle}(r_j)/\mu_r$.

| Function branch | Global row | Core columns | Mantle columns | Equation |
| --- | --- | --- | --- | --- |
| `section_f` | `core_f + delta_row_f` | `Tb[:,0]` in the core f block | `-Fm0_cmb` in p | $f_f-f_m=0$ |
| `section_p` | `mantle_p` | $\mu_r(T_b+r_jT_b')$ in f | $-(F_{m,0}+r_jF_{m,1})$ in p | $\mu_r(f_f+r_jf_f')-(f_m+r_jf_m')=0$ |
| `section_g` | `core_g + delta_row_g` | $\mu_rT_b$ in g | $-G_{m,0}$ in q | $\mu_rg_f-g_m=0$ |
| `section_q` | `mantle_q` | $T_b+r_jT_b'$ in g | $-\eta_r(G_{m,0}+r_jG_{m,1})$ in q | $g_f+r_jg_f'-\eta_r(g_m+r_jg_m')=0$ |

The notation in the table describes row coefficient vectors, not new scalar
unknowns. Each returned row may therefore contain nonzeros in two distant
global column blocks.

The conditions represent normal magnetic continuity, tangential magnetic
matching with permeability, and a diffusive tangential electric-field
relation under the assumptions of the present model.

### 12.4 Exterior rows

| Function branch | Global row | Nonzero columns | Equation |
| --- | --- | --- | --- |
| `section_p` | `mantle_p + 1` | p block: `r_surface*Fm1_surf + (l+1)*Fm0_surf` | $r_sf_m'(r_s)+(\ell+1)f_m(r_s)=0$ |
| `section_q` | `mantle_q + 1` | q block: `Gm0_surf` | $g_m(r_s)=0$ |

These are potential-field conditions at the **outer mantle surface**, not
at the CMB. There is no additional discretized exterior shell.

### 12.5 Sparse-triplet contract

Both new boundary functions return:

```python
[data, global_row_indices, global_column_indices]
```

The core-side temporary matrix has `N1` columns; the mantle-side matrix has
`N_mantle` columns. Their COO data are concatenated after applying different
column offsets but the same global row offset.

Confusing row offsets with column offsets, or assuming both temporary
matrices have the same width, is a likely merge error.

### 12.6 Physical restriction at a slipping boundary

The current new CMB routine does not add a generic velocity column for the
motional electric-field jump associated with relative fluid/wall motion.
Because the electric field includes $-\mathbf u\times\mathbf B_0$, this
requires separate treatment for a slipping or moving interface.

The controlled physical benchmark therefore uses an immobile **no-slip CMB**.
A stress-free test can satisfy the discretized energy balances without
validating the electric boundary condition for an immobile wall with slip.

## 13. Worked Numerical Matrix Example

### 13.1 Example configuration

Consider:

```python
hydro = magnetic = 1
thermal = compositional = rotdyn = 0
innercore = 'insulator'
ricb = 0.35
m = 1
symm = -1
N = 64
lmax = 64
N_mantle = 64
mantle = 'resolved'
```

Then $n_b=32$, $N_1=64$, $n=2048$, and $n_m=2048$.

### 13.2 Global column ranges

All ranges below use zero-based indices; the upper bound is exclusive.

| Field | No resolved mantle | Resolved mantle |
| --- | --- | --- |
| u | `[0, 2048)` | `[0, 2048)` |
| v | `[2048, 4096)` | `[2048, 4096)` |
| f | `[4096, 6144)` | `[4096, 6144)` |
| g | `[6144, 8192)` | `[6144, 8192)` |
| p | Absent | `[8192, 10240)` |
| q | Absent | `[10240, 12288)` |
| Total size | 8192 | 12288 |

The extension adds 4096 magnetic unknowns in this example. If temperature
is enabled, its start moves from 8192 to 12288; it must not overlap p or q.

### 13.3 First magnetic harmonic blocks

For the G21 dipolar background and this velocity symmetry, the first
poloidal magnetic degree is $\ell=1$ and the first toroidal magnetic degree
is $\ell=2$. They are different parity sequences.

| Row | Meaning |
| --- | --- |
| 4096 | Existing ICB poloidal insulating condition for first f block |
| 4097 | CMB continuity between first f and p blocks |
| 4098 through 4159 | Interior rows of first core poloidal block |
| 6144 | Existing ICB toroidal insulating condition for first g block |
| 6145 | CMB matching between first g and q blocks |
| 6146 through 6207 | Interior rows of first core toroidal block |
| 8192 | Complementary poloidal CMB interface condition |
| 8193 | First mantle poloidal exterior condition |
| 8194 through 8255 | Interior mantle poloidal rows |
| 10240 | Complementary toroidal CMB interface condition |
| 10241 | First mantle toroidal exterior condition |
| 10242 through 10303 | Interior mantle toroidal rows |

The first f/p interface row spans columns `[4096,4159]` and `[8192,8255]`.
The first g/q interface row spans `[6144,6207]` and `[10240,10303]`.
Here these last ranges are inclusive coefficient-column ranges.

This example must not be generalized by copying the numbers: changing N,
degree truncation, symmetry, or active fields changes the offsets.

### 13.4 Adding the inherited conducting inner core

Keeping the same core truncation, take `N_cic = 16`, `lmax_cic = 64`, and
enable the Chebyshev conducting inner core. Then `Nic = 8` and `nic = 256`.
With rotation, temperature, and composition still disabled:

```text
offset_fic = 8192
offset_gic = 8448
offset_p   = 8704
offset_q   = 10752
total size = 12800
```

The core poloidal CMB condition now follows the two IC interface rows and
therefore uses the third reserved row of each corresponding core f block.
The mantle still reserves two rows per scalar, because its own equations
remain second order.

## 14. Existing Inner-Core and Rotation Paths

### 14.1 Inner-core coupling was not newly invented

`bc_b_cic(l, loc)` remains the inherited core/inner-core matching routine.
The extension changes the global positioning of its entries, not the
fundamental origin of the conducting-inner-core feature.

| Branch | Row block | Core column block | Inner-core column block |
| --- | --- | --- | --- |
| `section_f` | `offset_f` | `offset_f` | `offset_fic` |
| `section_g` | `offset_g` | `offset_g` | `offset_gic` |
| `section_fic` | `offset_fic` | `offset_f` | `offset_fic` |
| `section_gic` | `offset_gic` | `offset_g` | `offset_gic` |

The local degree index is multiplied by `N1` on the fluid side and by `Nic`
on the solid-inner-core side. The mantle uses neither of those widths.

### 14.2 Other ICB routines

`bc_b_icb` and the TWA and insulating outer-boundary routines also replace
hard-coded magnetic row starts by `offset_f` and `offset_g`. The Bessel
branch receives an additional complex-square-root fix described below.

### 14.3 Axial rotational dynamics

The three inherited rotational variables now start at `offset_rotdyn`.
The following consumers were updated accordingly:

- rotational mass entries;
- toroidal velocity boundary-condition columns for the rotating boundaries;
- axial viscous and magnetic torque rows;
- gravitational torque entries;
- mantle/inner-core misalignment and relaxation entries;
- eigenmode and forced-solution extraction.

This does **not** add free equatorial mantle rotation. Computing x/y torque
diagnostics does not create an FCN or a fully coupled nutation model.

## 15. Thin-Wall Corrections

### 15.1 Existing function, changed expression

The public call remains:

```python
bc_b_thinlayer(l, loc, mu_vf, c, c1, boundary)
```

Here `loc` selects `'nocurl'` or `'1curl'`, and `boundary` selects `'cmb'`
or `'icb'`.

Define the transformed radial quantities used in this function:

$$
F=rf,\quad F'=f+rf',\quad F''=2f'+rf'',\qquad
G=rg,\quad G'=g+rg'.
$$

Also define $\epsilon=+1$ at the CMB and $-1$ at the ICB, and:

$$
k=(\ell+1/2)\epsilon-1/2.
$$

### 15.2 Corrected radial denominator

Previous code:

```python
nabF = F2 - l*(l+1)*F/rj
```

Current code:

```python
nabF = F2 - l*(l+1)*F/rj**2
```

The difference is hidden at a unit-radius CMB but not at a non-unit ICB.

### 15.3 Previous and current poloidal conditions

The previous expression was:

$$
\mu_{vf}F'+\frac{kF}{r_j}+\epsilon kcF'
+\epsilon c_1r_j\left(\mu_{vf}+\frac12\epsilon kc\right)\mathcal LF=0.
$$

The current expression is:

$$
(\mu_{vf}+\epsilon kc)F'+\frac{kF}{r_j}
+\epsilon c_1r_j\mu_{vf}\mathcal LF+C_{\rm curv}F=0,
$$

where $\mathcal LF=F''-\ell(\ell+1)F/r_j^2$ and:

$$
C_{\rm curv}=\begin{cases}
\dfrac{c\ell}{\mu_{vf}\mu_w r_j}
\left(\dfrac{\ell+1}{\mu_w}-1\right),&\mathrm{CMB},\\
-\dfrac{c(\ell+1)}{\mu_{vf}\mu_w r_j}
\left(\dfrac\ell{\mu_w}+1\right),&\mathrm{ICB}.
\end{cases}
$$

`mu_vf` is vacuum/fluid permeability. `mu_w` is read from `muw_cmb` or
`muw_icb`, with a fallback of one; nonpositive values are rejected.
The old mixed $cc_1$ contribution is not retained in the same form.

### 15.4 Toroidal condition

Its algebraic form remains:

$$
G+\epsilon r_jc_1G'=0.
$$

The row and column locations nevertheless follow the updated shared offsets.

### 15.5 Consequence for regression comparisons

The TWA itself changed. Differences between a historical TWA calculation
and a current resolved calculation cannot all be attributed to the new
mantle domain. A controlled comparison must use the same corrected common
code and explicitly identify the wall approximation being tested.

Unit-permeability thin-shell comparisons have been exercised. General wall
permeability contrasts, the corrected ICB wall, and slipping walls need
dedicated physical benchmarks.

## 16. Eigenvalue and Forced-Solution Extraction

### 16.1 Existing numerical solvers retained

`solve_nopp.py` continues to use SLEPc EPS for the generalized eigenproblem
and PETSc KSP for harmonic forcing. There is no replacement eigensolver or
automatic global `--mantle` command-line parser.

The principal change is that both solver branches now slice the global
solution using the same `ut.offset_*` and `ut.size_*` definitions as assembly.

### 16.2 Updated slicing contract

| Output quantity | Current global slice |
| --- | --- |
| Fluid velocity | `offset_u : offset_v + size_v` |
| Fluid magnetic field | `offset_f : offset_g + size_g` |
| Axial rotation | `offset_rotdyn : offset_rotdyn + size_rotdyn` |
| Inner-core magnetic field | `offset_fic : offset_gic + size_gic` |
| Mantle magnetic field | `offset_p : offset_q + size_q` |
| Temperature | `offset_h : offset_h + size_h` |
| Composition | `offset_i : offset_i + size_i` |

The eigenvalue branch applies these slices to the real and imaginary
eigenvector matrices. The forced branch applies them to the collected
complex solution vector and reshapes outputs into a single solution column.

### 16.3 No automatic global cleanup after failure

The historical solver can leave old files in a shared directory after a
failed or differently configured calculation. It has not become a
transactional output manager. The notebook avoids this by isolating cases,
checking outputs, and writing a completion marker only after analysis.

## 17. Output-File Contracts

### 17.1 Mantle field files

```text
real_magnetic_mantle.field
imag_magnetic_mantle.field
```

They contain **mantle coefficients only**. The first half comprises all
poloidal mantle harmonic blocks, followed by all toroidal mantle blocks.
Each block has `N_mantle` coefficients. Each solution is a column, giving
`2*nman` rows.

The existing `real_magnetic.field` and `imag_magnetic.field` remain fluid-core
files. Appending mantle coefficients to them would break their readers and
is not the current implementation.

### 17.2 Existing fields and diagnostics

The inner-core field files remain separate. Velocity, thermal, composition,
and rotation outputs retain their roles, with extraction positions updated.

Field files are opened with `wb` when written successfully. Several `.dat`
diagnostic files use append mode. The output policy is therefore not uniform.

### 17.3 New Cartesian torque file

For the equatorial diagnostic path:

```text
magnetic_cmb_xyz_full.dat
```

Its columns are:

```text
sigma omega Gamma_x_real Gamma_x_imag Gamma_y_real Gamma_y_imag Gamma_z_real Gamma_z_imag
```

The header identifies the complete linear Maxwell expression and dynamic
dimensionless units. The file is appended to, not overwritten on each
`spin_doctor` call.

The separate name avoids mixing these values with the intermediate partial
proxy output `magnetic_cmb_xyz.dat`.

### 17.4 Model index and legacy readers

The diagnostic mantle index now admits value 2 for resolved calculations.
Legacy scripts that assume only values 0 and 1 must be checked. Additional
mantle parameters are not automatically encoded as a new complete schema
in every historical `.dat` table; case manifests remain important.

## 18. Complete Electromagnetic Torque Calculation

### 18.1 Existing axial spectral functions

`gamma_magnetic()` continues to provide the axial spectral torque operator
for the appropriate `m=0`, `symm=1` configurations. Its mantle check now
accepts both TWA and resolved models.

`gamma_magnetic_ic()` remains the inherited inner-core axial diagnostic.
Both return correctly shaped zero arrays when inactive instead of a scalar
zero, improving consistency with dot products.

`dot_scalar(a, b)` was added to selected consumers to extract a scalar from
the result of a NumPy dot product explicitly.

### 18.2 New shared surface API

```python
gamma_magnetic_cmb_distribution(b_sol, ntheta=181, nphi=361,
                                theta0=0.0, theta1=np.pi)
gamma_magnetic_cmb_xyz(b_sol, ntheta=181, nphi=361)
```

The input is the expanded **fluid-core** magnetic coefficient vector, not
the full global vector and not the mantle output vector.

Internal helpers are:

```python
_split_core_magnetic_solution(b_sol)
_core_magnetic_cmb(b_sol, theta, phi)
_core_tangential_magnetic_cmb(b_sol, theta, phi)
_background_br_cmb(theta)
```

The tangential-only helper remains a compatibility wrapper around the
three-component reconstruction.

### 18.3 Complete versus intermediate expression

The intermediate equatorial diagnostic retained only
$B_{0r}\mathbf b_t$. The current routine includes both linear Maxwell terms:

$$
\mathbf t_\Omega=-R^3\hat{\mathbf r}\times
\left(B_{0r}\mathbf b_t+b_r\mathbf B_{0t}\right).
$$

This is the raw code-unit torque density per solid angle. Physical units
require the field and permeability scales as well.

For the axisymmetric poloidal background, $B_{0\phi}=0$. Define:

$$
A_\theta=B_{0r}b_\theta+b_rB_{0\theta},\qquad
A_\phi=B_{0r}b_\phi.
$$

Then:

$$
\mathbf t_\Omega=-R^3
\begin{pmatrix}
-A_\theta\sin\phi-A_\phi\cos\theta\cos\phi\\
A_\theta\cos\phi-A_\phi\cos\theta\sin\phi\\
A_\phi\sin\theta
\end{pmatrix}.
$$

The negative sign is the current mantle-side convention, matching the axial
spectral diagnostic under its applicable assumptions.

### 18.4 Reconstruction and quadrature

The routine evaluates core radial Chebyshev functions and derivatives at
the outer core endpoint. It reconstructs $b_r$, $b_\theta$, and $b_\phi$.
The harmonics are evaluated at longitude zero and multiplied by $e^{im\phi}$
instead of recomputed for every longitude.

Angular integration uses Gauss-Legendre nodes in $\cos\theta$ and a periodic
uniform longitude grid without a duplicated endpoint. Coordinate poles are
avoided. Input angular bounds are in radians.

### 18.5 Returned dictionary

| Key | Shape | Meaning |
| --- | --- | --- |
| `theta` | `(ntheta,)` | Increasing colatitude |
| `phi` | `(nphi,)` | Periodic longitude samples |
| `density_xyz` | `(ntheta, nphi, 3)` | Complex density per solid angle |
| `profile_xyz` | `(ntheta, 3)` | Longitude-integrated contribution, including $\sin\theta$ |
| `theta_weights` | `(ntheta,)` | Weights for integrating profiles over colatitude |
| `solid_angle_weights` | `(ntheta,)` | Weights for the cosine-coordinate integration |
| `torque_xyz` | `(3,)` | Signed complex integrated x, y, z torque |

The profiles satisfy:

$$
\Gamma_j=\int_0^\pi\frac{d\Gamma_j}{d\theta}\,d\theta.
$$

Integrate the complex signed profile, not its absolute value. Displaying
latitude in degrees on a plot does not change the profile's units: it is
still defined per radian of colatitude.

### 18.6 Units and amplitude normalization

`spin_doctor` multiplies raw torque by:

$$
Le^2(\Omega\tau)^2.
$$

No automatic conversion to N m is performed. For the usual dynamic torque
scale with time unit $\tau$, dimensional conversion involves
$\rho R^5/\tau^2$ as well as a consistent physical mode-amplitude choice.

Eigenvectors have arbitrary scale and phase. Comparing two unnormalized
torque amplitudes is not a physically controlled comparison. The notebook
uses a common kinetic norm and references phases to the fluid's equatorial
rotation projection.

### 18.7 `spin_doctor` integration

When magnetism is active and `m != 0`, the terminal table adds x/y magnitudes
and labels the existing mantle column explicitly as the z component.
The full complex values are retained in the new output file.

For `m=0`, the historical applicable axial spectral path remains in use.
The new equatorial-file condition does not produce the Cartesian file for
every axisymmetric calculation.

### 18.8 Symmetry checks and insulating traction

For the present axisymmetric background, integrated axial torque belongs
to $m=0$ and integrated equatorial torque to $|m|=1$. Higher azimuthal
orders have zero global vector torque under full-sphere integration.

The present $m=1$ convention gives $\Gamma_y=i\Gamma_x$ for the tested
equatorial pattern. Equal x/y magnitudes are therefore expected and do not
mean the two components are in phase.

A local Maxwell traction can be nonzero in an insulating region. It must
not be interpreted as torque deposited locally in insulating mantle
material. In the insulating potential-field reference, the integrated
torque cancels numerically.

The low-level API preserves this raw surface calculation as an analytical
check; it does not force the result to zero based on the model name. The
notebook excludes the insulating local profile from mantle-torque plots.
Computing the volume distribution of Lorentz torque deposited throughout
the mantle is a different diagnostic and is not implemented by this surface
API.

## 19. G21 Background-Field Derivative Correction

### 19.1 Exact error

For the imposed G21 dipole:

$$
h(r)=\frac r6-\frac{r^3}{10},\qquad
h'(r)=\frac16-\frac{3r^2}{10},
$$

$$
h''(r)=-\frac35r,\qquad h'''(r)=-\frac35.
$$

The baseline and initial mantle commit used positive signs for the last
two expressions. The current `h2` and `h3` fix those signs.

The optional radial power multiplies the derivative. These functions do
not mean “differentiate the already radially weighted profile.”

### 19.2 Why this changed actual solutions

These derivatives enter submatrix generation, assembled induction and
Lorentz operators, and post-processing. The error was not confined to the
energy-residual display.

The reconstructed radial and consoidal induction components were also
inconsistent with solenoidality. An independent test now checks:

$$
s_\ell^{\rm ind}=
\frac{r\partial_rq_\ell^{\rm ind}+2q_\ell^{\rm ind}}{\ell(\ell+1)}.
$$

### 19.3 Audit example

For the audited stress-free uniform-mantle example:

| Quantity | Before correction | After correction |
| --- | ---: | ---: |
| SOM-like frequency | Approximately 1.0060970 | Approximately 1.0042572 |
| Real eigenvalue part | Approximately -0.0041631 | Approximately -0.0040836 |
| Magnetic balance residual | Approximately 0.0224 | Order $10^{-9}$ |

Projection of the corrected physical-space induction onto the corresponding
assembled blocks agreed at about $10^{-14}$ in the checked case.

These numbers document a debugging experiment, not an Earth model or a
validation of a slipping electrical interface.

### 19.4 Mandatory rebuild rule

Every affected G21 calculation must regenerate its submatrices, assemble
again, and solve again. This applies to insulating and TWA calculations too.
Mixing old fields with new derivative functions is not a valid comparison.

## 20. Energy Diagnostics and Radial Quadrature

### 20.1 Previous quadrature

`utils4pp.diagnose` sampled functions at Gauss-Chebyshev nodes and compensated
for their weight using $\sqrt{1-x^2}$. This converges toward an unweighted
integral, but is not an exact Gaussian polynomial quadrature for that
unweighted integral.

For nearly rigid rotation, the dissipation is small relative to individual
terms. Small integration errors could therefore generate a large relative
mechanical identity residual even after spatial convergence.

### 20.2 Current quadrature

```python
nquad = 2*par.N
xk, weights = np.polynomial.legendre.leggauss(nquad)
xk, weights = xk[::-1], weights[::-1]
sqx = weights*nquad/np.pi
```

The reversed order preserves the historical outer-to-inner ordering of
diagnostic samples. The radial transformation now uses `ut.rcmb` explicitly.

The historical integration function remains:

```python
cg_quad(f, Ra, Rb, N, sqx)
```

Its formula is unchanged:

$$
\int_{R_a}^{R_b}f(r)\,dr\approx
\frac{R_b-R_a}{2}\frac\pi N\sum_k\texttt{sqx}_k f_k.
$$

Passing the rescaled Legendre weights makes this the intended Gauss-Legendre
rule without changing every worker signature. `N` in these worker calls
is now the quadrature count, not the number of retained field coefficients.

### 20.3 Affected worker calls

`diagnose` now passes `nquad` and the corresponding weights to:

```python
flow_worker(..., Ra, Rb, nquad, sqx, eigval)
magnetic_worker(..., Ra, Rb, nquad, sqx)
thermal_worker(..., Ra, Rb, nquad, sqx, 'thermal')
thermal_worker(..., Ra, Rb, nquad, sqx, 'compositional')
```

The field reconstructions still use their actual coefficient arrays.
This change does not increase the matrix dimensions or alter eigenvalues.

The older `utils_pp` and `utils_pp_new` modules were not globally replaced
by this change. Their consumers must not automatically be assumed to use
the same new quadrature.

### 20.4 Magnetic balance: no extra diffusion flux should be added

The core diagnostic uses the volume product with the Laplacian:

$$
2\sigma M=I+D_m,\qquad D_m=-Q_{\rm Ohm}+S_D.
$$

The boundary contribution is already implicit in the volume Laplacian
identity. Adding an extra diffusion flux without replacing the existing
term would double-count it.

The identity can close over the fluid core without adding mantle energy.
That does not make it a global core-mantle-exterior energy balance. The
diffusion surface term is not the entire Poynting flux.

### 20.5 Mechanical identity and boundary work

The independent check verifies:

$$
D_\Delta+D_{\rm int}=P_{\rm ICB}+P_{\rm CMB}.
$$

Here $D_\Delta$ is the velocity/Laplacian volume product, $D_{\rm int}$ the
positive strain dissipation, and the P terms are viscous traction work with
the outward normal of the fluid domain.

Stationary no-slip and impermeable stress-free boundaries should have zero
surface work. A forced moving boundary need not. The historical forcing
power diagnostic has not been generalized to every possible forcing here.

### 20.6 Quantitative quadrature audit

On the checked nearly rigid stress-free example, changing quadrature alone
reduced `resid0` from about $3.23\times10^{-2}$ to about $2.3\times10^{-11}$,
and `resid_u` from about $4.37\times10^{-4}$ to about $6.9\times10^{-13}$.

The eigenvalue and eigenvector were unchanged. Only diagnostics needed to
be rerun for this correction.

### 20.7 Explicit radius arrays

`utils4pp.funcheb` now uses:

```python
if r is None:
    x00 = x0
```

instead of `r == None`. The old comparison raised an ambiguous-array truth
error when evaluating multiple explicit radii, such as both boundaries in
the surface-work audits.

## 21. Independent Audit Programs

### 21.1 Magnetic balance audit

`tools/check_magnetic_energy_balance.py` reads saved eigenvalues and core
fields and reintegrates them independently at configurable Gauss-Legendre
resolutions.

```bash
python3 tools/check_magnetic_energy_balance.py --solutions 0
python3 tools/check_magnetic_energy_balance.py --solutions 0 --check-induction
```

It compares storage, induction, volume diffusion, positive current-squared
dissipation, and the signed diffusion surface term.

The optional projected-induction check:

1. reads saved A and B;
2. recovers their common scaling from a magnetic mass column;
3. reconstructs physical-space induction on an oversampled grid;
4. converts it into the ultraspherical equation basis;
5. compares interior core induction rows, excluding the boundary rows.

That projected check is restricted to the implemented shell/radial-projection
case and excludes the singular classical dipole. It is not a generic
verifier for every projection or geometry in Kore.

### 21.2 Viscous balance audit

`tools/check_viscous_energy_balance.py` evaluates diffusion, strain
dissipation, explicit boundary traction work, and the kinetic balance with
Lorentz work:

```bash
python3 tools/check_viscous_energy_balance.py --solutions 0
```

The outward-normal sign at the ICB is opposite to that at the CMB. The
implemented audit requires an unforced, unstratified fluid shell with
nonzero ICB radius. It supports the current hydrodynamic/MHD diagnostic
scope, not every forced or stratified combination.

### 21.3 Audit contracts

These scripts do not assemble or solve again and do not overwrite the
simulation files. Parameters must match the saved fields and matrices.
Dimension checks alone cannot detect physically different configurations
with identical sizes; case manifests are still needed.

Additional derivations and measured terms are retained in:

- [MAGNETIC_ENERGY_AUDIT.md](MAGNETIC_ENERGY_AUDIT.md);
- [VISCOUS_ENERGY_AUDIT.md](VISCOUS_ENERGY_AUDIT.md).

## 22. Portability and Library-Compatibility Changes

### 22.1 Sparse diagonal types

Explicit `dtype=float` was added to selected sparse diagonals in:

```text
bin/utils.py: Dlam, Slam
koreviz/libkoreviz.py
koreviz_mod/kmode.py
tools/plot_poltor.py
tools/plot_rad_con_tor_profile.py
tools/shear_layer_profile.py
```

These are portability changes, not new physical terms.

### 22.2 Spherical harmonic API

`utils.Ylm` now selects the available SciPy API:

```python
if hasattr(scsp, 'sph_harm_y'):
    out = scsp.sph_harm_y(l, m, theta, phi)
else:
    out = scsp.sph_harm(m, l, phi, theta)
```

The seminormalization factor remains
$\sqrt{4\pi/(2\ell+1)}$. The two APIs have different argument orders;
renaming the function without reordering arguments would change the field.

### 22.3 Extended scalar type

`operators.py` now defines:

```python
float_ext = getattr(np, 'float128', np.longdouble)
```

The induction routines use this available wide scalar instead of assuming
`np.float128` exists on every NumPy/platform combination. The actual
precision of the fallback is platform-dependent.

### 22.4 Inner-core file extensions

Several `mmread` calls for inherited inner-core operators now include the
`.mtx` suffix explicitly, such as `r2_D0f_ic.mtx` and `r2_D2g_ic.mtx`.

### 22.5 Complex Bessel argument

The relevant ICB forced branches now evaluate:

```python
(1-1j)*np.sqrt(par.forcing_frequency/(2*par.Em) + 0j)
```

instead of taking a potentially invalid real square root. This avoids a
numerical domain error; it does not independently validate every signed
frequency and Bessel-boundary convention.

### 22.6 Matplotlib contours

`koreviz_mod/plotlib.py` adds a helper handling old contour collections and
the newer contour-set API. Selected standalone tools also avoid assuming
that `.collections` is always present.

These changes remove or avoid contour-edge artifacts and API failures.
They do not imply complete mantle-domain support in every `koreviz` class.

### 22.7 Scalar extraction

`dot_scalar` is used in `spin_doctor.py`, `postprocess.py`, and
`postprocess_new.py` where a matrix-vector product must become a single
diagnostic value. This avoids depending on implicit conversion of a
one-element array.

## 23. Visualization and Field Reconstruction

### 23.1 `tools/plot_field.py`

The script now supports:

```text
u   fluid velocity in [ricb, 1]
b   fluid magnetic field in [ricb, 1]
bm  resolved mantle magnetic field in [1, r_surface]
```

The selected domain controls radial mapping, differentiation, coefficient
count, and shell masking. Core-origin parity expansion is used only when
appropriate; mantle coefficients are not expanded as full-sphere core data.

Display options are `raw`, `abs`, and `phase`. Phase is masked where amplitude
is too small. Raw real-part plots depend on longitude and the arbitrary
phase of the eigenvector.

```bash
python3 tools/plot_field.py 0 100 181 0 180 bm abs
python3 tools/plot_field.py 0 100 181 0 180 bm phase
```

The script still enables LaTeX text rendering and may require a TeX
installation. Its phase plot is not automatically a phase relative to a
physical forcing or a selected velocity component.

### 23.2 `tools/plot_b_layers.py`

This added script reconstructs fields across the available inner core,
fluid core, and resolved mantle, using the proper radial basis for each.
Inner-core parity-reduced coefficients are restored before reconstruction.

```bash
python3 tools/plot_b_layers.py 0 100 181 0 180 abs --save b_layers.png
```

It reads the separate field files when available. It does not make missing
layers conducting or infer a mantle field from the core output alone.
Comparing independently autoscaled color maps can conceal amplitude changes;
normalization and color limits must be controlled for quantitative use.

### 23.3 `tools/plot_cmb_torque_latitude.py`

This tool now calls the shared complete Maxwell routine rather than
maintaining a separate axial-only reconstruction.

Its principal options are:

```text
solnum
--component x|y|z
--profile latitude|longitude
--quantity physical_abs|real|imag|complex_abs
--phase DEGREES
--theta0 DEGREES --theta1 DEGREES
--ntheta N --nphi N
--normalize --save IMAGE
```

```bash
python3 tools/plot_cmb_torque_latitude.py 0 --component x --quantity real
```

`physical_abs` denotes the absolute value of the selected real part, not a
phase-invariant physical mode amplitude. `complex_abs` is the modulus of
the complex amplitude. Plot normalization divides by the displayed maximum;
it does not establish an SI amplitude for an eigenmode.

This low-level tool does not automatically suppress an insulating surface
traction profile. The research notebook does suppress that misleading
material-torque interpretation in its plots.

### 23.4 `tools/plot_matrix_structure.py`

The new utility reads Kore's CSR `.npz` format and can display sparsity or
nonzero magnitudes, threshold small entries, crop the view, and label blocks
using the current shared offsets.

```bash
python3 tools/plot_matrix_structure.py A.npz B.npz --save matrix_structure.png
```

It also accepts some matrix objects and analysis dictionaries. It is a
structural visualization, not a condition-number calculation or proof of
spectral correctness.

### 23.5 Historical visualization modules

The `koreviz` and `koreviz_mod` changes primarily concern sparse types and
Matplotlib compatibility. General resolved-mantle support was not added
to every existing visualization class. The explicit `bm` and multilayer
tools remain the dedicated reconstruction paths.

## 24. Research Notebook and Experiment Isolation

Notebook:

[som_thin_wall_benchmark.ipynb](../experiments/som_thin_wall_benchmark/som_thin_wall_benchmark.ipynb).

### 24.1 Self-contained orchestration

The notebook contains the setup, launcher, analyzer source, mode selection,
plots, and exports. It does not depend on the removed historical sweep
helpers. It still calls the actual Kore source programs.

Each case contains:

```text
case/
  bin/parameters.py
  bin/*.py                     # source snapshot
  manifest.json
  analysis_settings.json
  analyze_case.py               # emitted from notebook ANALYZER cell
  run.log
  A.npz, B.npz, *.mtx, *.field
  candidates.json
  candidates.npz
  complete.json
```

### 24.2 Parameter rewriting

`parameter_text(overrides)` rewrites assignments in a parsed parameter-file
AST and appends final overrides. Function definitions are retained without
rewriting their bodies. This acts on the **case copy**, not the repository's
global parameter file.

Fresh subprocesses avoid stale `parameters`/`utils` globals in the kernel.
The kernel interpreter, solver interpreter, and MPI launcher are explicit.
A small MPI-size check catches a mismatched launcher/interpreter pairing.

### 24.3 Fingerprints and caching

Case fingerprints include:

- physical parameters and resolutions;
- hashes of Python source files in `bin`;
- the analyzer text;
- analysis-grid and torque-grid settings;
- Python interpreter and MPI launcher strings.

A matching completed case is reused. An incompatible or incomplete case
directory causes an explicit stop rather than an overwrite. Completion is
recorded only after the analyzer succeeds.

This does not lock binary versions of PETSc/SLEPc/MPI, the operating system,
or every dependency. A publishable archive should preserve the environment
and relevant version information separately.

### 24.4 Candidate diagnostics

The analyzer retains all converged candidates, not merely the mode nearest
the target. It constructs the full vector, including mantle fields, when
checking the matrix residual:

$$
\frac{\|A\mathbf x-\lambda B\mathbf x\|}
{\|A\mathbf x\|+|\lambda|\|B\mathbf x\|}.
$$

It also computes energy diagnostics, a rigid-rotation projection, normalized
torques, and latitude profiles.

### 24.5 Mode tracking

The insulating reference is identified by its rigid-rotation projection
within a frequency window. Conducting cases are matched by kinetic overlap
on a common radial grid and common padded harmonic-degree layout.

The notebook checks:

- rigid-rotation purity;
- overlap with the selected reference or previous profile;
- separation from the second-best candidate;
- matrix and energy residuals;
- stable eigenvalue sign;
- overlap between resolutions.

Manual indices can be supplied, but do not disable those checks. A rejected
case must not silently appear on a published curve.

### 24.6 Torque and phase observables

The kinetic norm is computed from reconstructed velocity components with
the same harmonic and radial metric. Torque amplitudes are divided by this
norm.

A volume projection of the degree-one toroidal velocity onto solid rotation
provides the phase reference $\Omega_x$, with $\Omega_y=i\Omega_x$ in the
current convention. The ratio $\Gamma_x/\Omega_x$ is invariant to an
arbitrary global eigenvector phase.

Near-zero torque phases and percentage errors with a near-zero denominator
are reported as undefined, rather than converted into misleading numbers.

## 25. Experiments Currently Implemented

### 25.1 Initial matched uniform-shell triplet

The initial comparison uses insulating, TWA, and resolved cases at two
resolutions. For the conducting pair:

$$
h/R=0.01,\qquad s=0.1,\qquad G^*=0.001.
$$

The common numerical controls are $E=10^{-3}$, $E_m=10^{-3}$, and
$Le=10^{-2}$, with unit permeabilities, an insulating inner core, stress-free
ICB, and stationary no-slip CMB.

These are tractable numerical benchmark parameters, not a realistic Earth
interior model. The isolated fluid's SOM-like branch is not automatically
the fully coupled terrestrial free-core-nutation mode.

### 25.2 Uniform thickness sweep at fixed conductance

For each thickness:

$$
s=\frac{G^*}{h/R},\qquad c=h/R,\qquad c_1=G^*.
$$

The default seven thicknesses range from 0.002 to 0.2. Two conducting models
and two resolutions give 28 cases; the insulating reference is reused.

The matched comparison tests how the TWA approaches the resolved shell as
the layer becomes thinner. The user has observed decreasing discrepancies
in that limit. Article claims must retain the corresponding manifests,
tables, and numerical-convergence evidence.

### 25.3 Exponential profiles at fixed conductance and domain

The resolved domain depth D is fixed; alpha varies and sigma0 is recalculated
to keep the integral conductance fixed.

The TWA is computed once per resolution. Its `c1` is the prescribed G*, and
its conventional thickness is explicitly held fixed. In the initial
configuration, `EXP_TWA_H = EXP_DEPTH`.

Defaults are D = 0.2 and alpha values 0, 5, 10, 20, and 50. Including one TWA
reference per resolution gives 12 cases. The alpha = 0 case is the uniform
resolved reference.

This experiment asks whether radial redistribution matters beyond total
conductance. It is not a sequence in which TWA secretly changes thickness
to follow the effective depth of every profile.

### 25.4 Plots and exports

Separate figures show frequency and damping corrections, direct model
differences, normalized torque amplitudes, phase differences, conductivity
profiles, and selected latitude profiles. TWA uses distinct dashed styling
so it remains visible when curves overlap.

Exports include PNG/PDF figures and section-specific CSV tables:

```text
selected_modes.csv
all_candidates.csv
model_differences.csv
profile_comparisons.csv
```

Numerical bands combine resolution changes and a working floor. They are
not confidence intervals or rigorous error bounds for a nonnormal
eigenproblem. Identical printed values at two resolutions do not imply
infinite numerical accuracy.

### 25.5 Studies not yet implemented in this notebook

Generic multi-Le, multi-Em, and multi-E campaigns, universal extrapolation
laws, and terrestrial K_CMB inference are not implemented by these cells.
Their appearance in planning documents does not make them existing code.

## 26. Tests and Available Validation Evidence

### 26.1 Local test command

```bash
python3 -m unittest discover -s tests -v
```

The current suite has 19 test methods across six files; several methods
cover multiple degrees, radial powers, or conductivity profiles.

| File | Methods | Main coverage |
| --- | ---: | --- |
| `test_background_derivatives.py` | 2 | G21 derivatives, weighted derivatives, full-sphere parity |
| `test_induction_solenoidality.py` | 1 | Independent radial/consoidal induction consistency |
| `test_cmb_torque.py` | 5 | Potential-field cancellation, axial analytical amplitude/sign, spectral references, disabled path, m-selection and equatorial relation |
| `test_magnetic_energy_audit.py` | 2 | Polynomial diffusion/current/surface identity, explicit-radius reconstruction |
| `test_viscous_energy_audit.py` | 3 | Rigid rotation, nonzero polynomial boundary work, actual diagnose quadrature path |
| `test_som_notebook_analysis.py` | 6 | Cell syntax and IDs, wrapped phase, zero-torque guards, eigenvector-phase invariance, exponential conductance, invalid inputs |

### 26.2 Integration runs

Development verification included actual MPI assembly/solution/analyzer
runs for:

- the uniform triplet at both resolutions;
- a two-thickness, two-resolution conducting subset;
- the complete initial exponential-profile campaign.

These checks used temporary directories to avoid replacing the research
outputs. Candidate matching and inter-resolution checks passed in the
specified tests. Generated plots were also inspected.

### 26.3 Limits of this evidence

The tests do not cover every combination of inner-core model, permeability,
forcing, stratification, geometry, and moving interface. There is no
exhaustive independent shell-diffusion eigenvalue benchmark or complete
global core-mantle energy validation in the current suite.

Small residuals establish consistency of the tested discretized equations;
they do not establish that every selected boundary model is physically
appropriate for a planet.

## 27. Limitations and Unresolved Issues

### 27.1 General permeability contrasts

`mu_mantle` appears at interfaces, but the bulk mantle inverse-conductivity
profile and diffusion operator do not explicitly divide by it. General
contrasts must not be advertised as validated until the entire convention
has been checked and independently benchmarked.

### 27.2 Slip, moving walls, and equatorial feedback

The new CMB interface does not generically include motional electric-field
terms for relative wall motion. The no-slip stationary-wall experiment is
the controlled protocol used for the comparison.

Inherited `rotdyn` remains axial. Cartesian torque diagnostics do not add
equatorial orientation unknowns or mechanical torque feedback for m = 1.

### 27.3 Profiles and exterior physics

The exposed conductivity profile is radial and exponential. Generic
piecewise, lateral, or anisotropic conductivities, topographic interfaces,
additional external conductors, and a radial permeability profile are not
implemented as new general options.

### 27.4 Mantle heat and global energy

The mantle magnetic field is available, but `spin_doctor` does not provide
a complete global mantle-energy budget, a thermal diffusion solution,
automatic mantle heating initial conditions, or a convection/plume model.
The surface torque routine is not a volumetric torque-deposition diagnostic.

### 27.5 Pure magnetic-decay path

The current `magnetic_worker` initializes reconstructed induction components
inside `if par.hydro`, then uses them in magnetic dot products. The general
`hydro = 0` magnetic-decay path therefore needs a separate initialization
and test audit; the present MHD energy work has not fixed that path globally.

### 27.6 Combination coverage

Conducting-inner-core compatibility is represented in the layout and
interface code, but does not have the same integration coverage as the
insulating-inner-core benchmark. Forced Bessel, thermal/compositional,
full-sphere, and rotational combinations require their own regression runs.

### 27.7 Source and reader consistency

Historical postprocessors and visualizers have not all been unified. Some
retain their own reconstruction or quadrature helpers and time-scale
assumptions. Do not mix their outputs without checking conventions.

### 27.8 Resolution and planetary extrapolation

Large alpha can produce a very large diffusivity contrast even where the
magnetic field looks smooth. Check profile representation, matrix
conditioning, N_mantle, angular truncation, algebraic error, and mode tracking.

No universal extrapolation from accessible E and Em to planetary values has
been implemented or validated by adding the mantle domain.

### 27.9 Multiprocessing remains inherited behavior

`utils4pp` still selects a `fork` multiprocessing start method at import and
uses module-level reconstructed radial grids in its workers. The current
notebook isolates this code in subprocesses rather than importing all solver
modules into an already-running kernel.

This extension has not redesigned the diagnostics for arbitrary `spawn`
contexts, all operating systems, or repeated start-method initialization in
another application's process. A destination branch that changes process
startup must review that inherited execution model explicitly.

## 28. File-by-File Merge Inventory

### 28.1 Tracked files changed relative to the baseline

| File | Main differences | Merge role |
| --- | --- | --- |
| `bin/utils.py` | Layout, profiles, model index, G21, harmonics, basis types, torque API | Essential shared contracts |
| `bin/parameters.py` | New parameter definitions and many local experiment values | Separate schema from defaults |
| `bin/submatrices.py` | Mantle radial operators and tau shifts | Essential |
| `bin/operators.py` | p/q/i loader, mantle mass/diffusion, portability and IC filenames | Essential |
| `bin/bc_variables.py` | Mantle endpoint values and derivatives | Essential |
| `bin/assemble.py` | Offsets, A/B mantle blocks, interfaces, harmonic term, TWA and Bessel corrections | Essential; highest conflict risk |
| `bin/solve_nopp.py` | Shared slicing and mantle outputs in both solution paths | Essential |
| `bin/spin_doctor.py` | Resolved axial support, Cartesian torques, scalar extraction, new file | Core diagnostic integration |
| `bin/utils4pp.py` | Explicit-radius fix and diagnostic quadrature | Numerical diagnostic correction |
| `bin/postprocess.py` | Resolved axial support, mantle index 2, scalar extraction | Legacy-reader compatibility |
| `bin/postprocess_new.py` | Corresponding legacy diagnostic updates | Legacy-reader compatibility |
| `koreviz/libkoreviz.py` | Sparse diagonal types | Portability |
| `koreviz_mod/kmode.py` | Sparse diagonal types | Portability |
| `koreviz_mod/plotlib.py` | Contour API compatibility | Portability |
| `tools/plot_field.py` | bm domain, phase, reconstruction and shell mask | Visualization |
| `tools/plot_poltor.py` | Sparse diagonal types | Portability |
| `tools/plot_rad_con_tor_profile.py` | Sparse diagonal types | Portability |
| `tools/shear_layer_profile.py` | Sparse diagonal types | Portability |
| `.gitignore` | Legacy experiment/output exclusions | Repository hygiene; incomplete for current results tree |

Some of these files are clean relative to HEAD because their changes are
already in `9ab88971`. They must still be included when comparing against
the pre-extension baseline.

### 28.2 Added current files

```text
tools/check_magnetic_energy_balance.py
tools/check_viscous_energy_balance.py
tools/plot_b_layers.py
tools/plot_cmb_torque_latitude.py
tools/plot_matrix_structure.py

tests/test_background_derivatives.py
tests/test_induction_solenoidality.py
tests/test_cmb_torque.py
tests/test_magnetic_energy_audit.py
tests/test_viscous_energy_audit.py
tests/test_som_notebook_analysis.py

experiments/som_thin_wall_benchmark/som_thin_wall_benchmark.ipynb

docs/ARTICLE_1_PLAN.md
docs/MAGNETIC_ENERGY_AUDIT.md
docs/VISCOUS_ENERGY_AUDIT.md
docs/CONDUCTING_MANTLE_MERGE_GUIDE.md
docs/CONDUCTING_MANTLE_MERGE_GUIDE_EN.md
```

Generated `analyze_case.py` files and copied `bin/*.py` files under result
directories are snapshots, not additional source modules to merge one by one.

### 28.3 Deletions and documentation cleanup

The worktree also reports deletion of the old
`Mantle_Resolution_Kore_Scientific_Documentation.md` and `mkdocs.yml`.
Those deletions are not required to execute the resolved mantle. Decide
whether to retain them as repository-cleanup changes separately from the
physics merge.

The old long scientific document exists in the initial extension commit;
the current guides additionally describe later corrections. Its historical
presence must not be mistaken for an active current source file.

## 29. Recommended Merge Procedure

### 29.1 Protect the current work first

Before merging, archive the complete local work, inspect tracked and
untracked files, and identify the exact destination commit. Do not use a
global reset or checkout to clean up work that has not been archived.

This document performs no commit, merge, reset, or source cleanup.

### 29.2 Exclude generated experiment snapshots

The present `.gitignore` excludes common `.mtx`, `.npz`, `.dat`, and `.field`
files and some historical result locations. It does not generically cover
the current `experiments/**/results/` tree.

Without an explicit exclusion, `git add .` can stage manifests, logs, images,
and hundreds of copied Python files. Consider a reviewed ignore rule:

```gitignore
experiments/**/results/
```

This recommendation is not automatically applied by writing the guide.
Archive scientific outputs separately with their manifests.

### 29.3 Suggested review batches

1. Portable sparse types, special-function APIs, contour APIs, IC filenames,
   and scalar extraction.
2. G21 derivative correction and its tests, independently of the mantle.
3. Shared global layout and all required consumers.
4. Mantle profile, radial bases, differential operators, interface rows,
   assembly, and outputs as a coherent feature batch.
5. Forced harmonic mantle term with dedicated forced-response validation.
6. Revised TWA as an explicit change to the reference wall model.
7. Complete Cartesian Maxwell torque and tests.
8. Energy quadrature and audit tools.
9. Visualization, notebook, documentation, and result-archiving policy.

This is a logical review order, not a claim that the historical commit can
be mechanically split into independent cherry-picks. Layout and domain
changes span several files and must stay mutually consistent.

### 29.4 Likely semantic conflicts

| Area | Required review |
| --- | --- |
| IC and rotation ordering | Verify every source and consumer uses the same layout |
| Poloidal projection | Preserve extra IC boundary rows and derivative basis |
| Toroidal CMB matching | Check diffusive and motional electric terms |
| TWA | Check meanings of c, c1, permeability, curvature, and approximation order |
| Matrix signs | Preserve corresponding A/B row orientations |
| Background derivatives | Apply common corrections to all model comparisons |
| Time/field normalization | Do not interpret rescaling as a physical mantle effect |
| Output slicing | Keep core and mantle field files separate |
| Diagnostics | Distinguish volume Laplacian, positive dissipation, and boundary work |

A conflict-free textual merge can still be physically wrong.

### 29.5 Explicit staging example

After reviewing parameter defaults and diffs, a core selection could be:

```bash
git add bin/utils.py bin/parameters.py bin/submatrices.py bin/operators.py
git add bin/bc_variables.py bin/assemble.py bin/solve_nopp.py
git add bin/spin_doctor.py bin/utils4pp.py
git add bin/postprocess.py bin/postprocess_new.py
git add tests/test_background_derivatives.py tests/test_induction_solenoidality.py
git add tests/test_cmb_torque.py tests/test_magnetic_energy_audit.py
git add tests/test_viscous_energy_audit.py tests/test_som_notebook_analysis.py
git diff --cached --stat
git diff --cached --check
```

Tools, visualization changes, and documentation can be reviewed in separate
batches. Add the notebook path explicitly, not its whole result directory.
Decide a policy for embedded notebook execution outputs without deleting
unarchived research results.

## 30. Post-Merge Verification and Rebuild Rules

### 30.1 Tests

```bash
python3 -m unittest discover -s tests -v
git diff --check
```

The test suite does not automatically launch all MPI notebook campaigns.
Identify whether any whitespace errors come from unrelated pre-existing
local changes before altering them.

### 30.2 Regression sequence

1. Check insulating and TWA cases without the resolved blocks.
2. Verify matrix sizes and original core output dimensions.
3. Use a common corrected G21 reference, or a field unaffected by that
   correction, when separating indexing changes from physical changes.
4. Run the matched thin uniform shell at two resolutions.
5. Compare selected eigenfunctions, not only column indices or frequencies.
6. Check energy identities, torque symmetries, and insulating cancellation.
7. Extend to the intended IC, forcing, stratification, and rotation options.

### 30.3 Standard isolated-case command sequence

```bash
python3 bin/submatrices.py 4 &&
mpiexec -n 4 python3 bin/assemble.py &&
mpiexec -n 4 python3 bin/solve_nopp.py \
  -st_type sinvert \
  -st_pc_factor_mat_solver_type mumps \
  -eps_error_relative &&
python3 bin/spin_doctor.py 4
```

This assumes a degree truncation compatible with four MPI ranks and an
environment providing the selected MUMPS solver. The configured notebook
kernel name alone is not proof that the interpreter has working
PETSc/SLEPc/MPI libraries.

The mantle shares the core magnetic angular distribution across MPI ranks;
it does not introduce an independent mantle MPI partition. Preserve the
existing divisibility requirement:

$$
(\ell_{\max}-m+1)\ \text{is divisible by}\ 2n_{\rm MPI}.
$$

Check the inner-core angular partition separately when it is enabled.

### 30.4 Additional integration tests required by a wider merge scope

- resolved mantle plus Chebyshev conducting inner core;
- simultaneous mantle, temperature, and composition blocks;
- inherited axial rotational dynamics and its three variable positions;
- harmonic forcing at nonzero and signed frequencies;
- full-sphere origin parity;
- low-conductivity approach to the insulating limit, avoiding exactly zero
  conductivity in an inverse-conductivity operator;
- independent validation of non-unit permeability after resolving its
  operator convention;
- motional electric matching for slipping or moving walls if required.

### 30.5 Rebuild decision table

| Change | Submatrices | Assembly | Solution | Diagnostics |
| --- | --- | --- | --- | --- |
| G21 derivatives or imposed radial field profile | Rebuild | Rebuild | Recompute | Recompute |
| Mantle conductivity, domain, or radial resolution | Rebuild | Rebuild | Recompute | Recompute |
| Diffusion operator | Rebuild if representation changes | Rebuild | Recompute | Recompute |
| TWA/interface expression | Rebuild if radial dependencies change | Rebuild | Recompute | Recompute |
| Forced mantle harmonic term | Existing compatible submatrices may be reused | Rebuild | Recompute | Recompute |
| Torque diagnostic only | No | No | No | Recompute |
| Diagnostic quadrature only | No | No | No | Recompute |
| Figure styling only | No | No | No | Replot only |

When provenance is ambiguous, rebuild the complete chain in a new isolated
directory rather than mixing old fields and new parameters.

## 31. Function and Call-Site Reference

### 31.1 Added mantle model functions

| Function | Caller or consumer | Contract |
| --- | --- | --- |
| `utils.conductivity_mantle(r)` | Profile evaluation, inverse diffusivity | Dimensionless exponential conductivity |
| `utils.mag_diffus_mantle(r)` | Mantle submatrix generation, CMB matching | Returns inverse conductivity |
| `utils.dmag_diffus_mantle(r)` | Toroidal diffusion submatrices | Exact radial derivative of that inverse profile |
| `submatrices._shift_mantle_bc_rows(matrix, Nm)` | Mantle submatrix writer | Two leading constraint rows |
| `submatrices.write_mantle_magnetic_submatrices()` | `submatrices.main` | Generates p/q radial files when resolved |
| `operators.b_mantle(l, section, component, offdiag)` | A forced rows and B eigenvalue rows | Mantle mass operator |
| `operators.magnetic_diffusion_mantle(...)` | A mantle rows | Variable-diffusivity poloidal/toroidal operators |
| `assemble.bc_b_mantle_cmb(l, loc)` | Core and mantle A assembly | Coupled global sparse triplets |
| `assemble.bc_b_mantle_surface(l, loc)` | Mantle A assembly | Insulating exterior sparse triplets |

### 31.2 Changed existing calls and contracts

| Function or path | Change |
| --- | --- |
| `utils` module initialization | Defines model flag, sizes, offsets, and enlarged `sizmat` |
| `utils.h2`, `utils.h3` | Correct G21 signs |
| `utils.Dlam`, `utils.Slam` | Explicit sparse floating dtype |
| `utils.Ylm` | New/old SciPy harmonic dispatch with correct argument order |
| `operators` module initialization | Loads i/p/q sections and explicit IC filenames |
| `operators.induction`, `induction_consoidal` | Portable wide scalar selection |
| `assemble.bc_b_thinlayer` | Corrected radial denominator and poloidal wall expression |
| `assemble.bc_b_cmb` | Centralized magnetic offsets |
| `assemble.bc_b_icb` | Centralized offsets and complex Bessel root |
| `assemble.bc_b_cic` | Centralized core/IC row and column positions |
| `solve_nopp.main` | New slicing in free/forced branches and mantle file output |
| `spin_doctor.main` | Resolved axial support, Cartesian diagnostic, new output file |
| `postprocess.main`, `postprocess_new.main` | Resolved model index and axial/scalar compatibility |
| `utils4pp.funcheb` | Explicit array-radius support through `is None` |
| `utils4pp.cg_quad` | Same signature/formula; documented scaled-weight interpretation |
| `utils4pp.diagnose` | Gauss-Legendre nodes/weights and updated worker quadrature count |

### 31.3 Added torque functions

| Function | Role |
| --- | --- |
| `_split_core_magnetic_solution` | Validate and split expanded core coefficients |
| `_core_magnetic_cmb` | Reconstruct all three perturbation components |
| `_core_tangential_magnetic_cmb` | Compatibility wrapper returning tangential components |
| `_background_br_cmb` | Axisymmetric imposed radial field at the CMB |
| `gamma_magnetic_cmb_distribution` | Complete linear surface density, profiles, and integrals |
| `gamma_magnetic_cmb_xyz` | Integrated-vector wrapper |
| `dot_scalar` in selected diagnostics | Explicit scalar product extraction |

### 31.4 Notebook orchestration and analysis functions

The current notebook contains functions for source hashing, parameter
rewriting, command execution, case creation, observables, phase wrapping,
plot export, and profile conductance normalization. Important names include:

```text
sha256
parameter_text
run_command
selected_torque
finish_plot
mode_observables
paired_observables
angle_difference
thickness_case
exponential_sigma0
exponential_case
```

The `ANALYZER` string defines the per-case program executed with the Kore
interpreter. Changing its text invalidates matching case fingerprints;
plot-only changes do not need to trigger new physical solves.

### 31.5 Source locations are not matrix indices

Line numbers in `assemble.py` move as code is edited. The formulas in
sections 8, 12, and 13 specify **matrix** rows and columns independently of
textual source line numbers. For a merge, use function names and current
diff hunks to locate code, and use the shared offset formulas to verify
the numerical matrix positions.

### 31.6 Existing low-level helpers reused by the extension

| Existing helper | New or expanded use |
| --- | --- |
| `utils.ell` | Provides identical angular degree/parity sequences for core and mantle |
| `utils.chebco`, `utils.chebco_f` | Represent radius powers and mantle diffusivity profiles on the mantle interval |
| `utils.Mlam` | Builds multiplication matrices in the required ultraspherical basis |
| `utils.Slam`, `utils.Dlam` | Builds mantle conversions and correctly scaled derivatives |
| `bc_variables.Tk` | Evaluates mantle basis derivatives at CMB and outer surface |
| `utils.packit` | Inserts mantle mass/diffusion blocks into global sparse triplets |
| `utils.load_csr` | Reads saved Kore-format A/B in solve and audit programs |
| `utils4pp.expand_sol`, `expand_reshape_sol` | Expands core origin parity and reorganizes core coefficients before diagnostics |
| `utils4pp.cheb2space_pol`, `cheb2space_tor` | Reconstructs candidate velocities for the common matching metric |
| `utils4pp.flow_worker`, `magnetic_worker`, `thermal_worker` | Receives the revised quadrature count and weights through `diagnose` |

These helpers were not all newly introduced. Keeping that distinction avoids
unnecessary reimplementation when merging with a branch that already has an
equivalent base helper.

## 32. Final Acceptance Checklist

- [ ] Record exact source and destination Git commits.
- [ ] Include the initial extension and subsequent working-tree corrections.
- [ ] Review untracked source, tests, tools, and documents explicitly.
- [ ] Keep generated experiment snapshots out of source staging.
- [ ] Separate parameter definitions from local experiment values.
- [ ] Verify identical vector layout in assembly, solution extraction, and readers.
- [ ] Preserve distinct core, inner-core, and mantle radial bases.
- [ ] Check tau-row shifts and additional conducting-IC poloidal rows.
- [ ] Verify interface row ownership and two-block column placement.
- [ ] Preserve the paired signs of A and B interior rows.
- [ ] Include the harmonic mantle mass term in forced calculations.
- [ ] Treat the revised TWA as a separately documented physical change.
- [ ] Regenerate every affected G21 calculation with corrected derivatives.
- [ ] Keep complete torque outputs separate from the old partial proxy.
- [ ] Do not present insulating local traction as mantle-material torque.
- [ ] Preserve energy normalization and new quadrature conventions.
- [ ] Run the analytical tests in the merged branch.
- [ ] Repeat the matched thin-shell benchmark at multiple resolutions.
- [ ] Exercise the IC, forcing, thermal, compositional, and rotation scope actually required.
- [ ] State permeability and wall-slip limitations explicitly.
- [ ] Decide documentation deletions and MkDocs changes separately.
- [ ] Archive the numerical environment and validation manifests.

**Acceptance criterion:** a coherent end-to-end implementation that preserves
the destination branch's required behavior and states its physical validity
scope. A clean textual merge is not sufficient scientific validation.
