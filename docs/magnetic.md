# Magnetic field

With `magnetic = 1` kore adds an imposed background magnetic field $\mathbf{B}_0$, the induction equation for the
perturbation field $\mathbf{b}$ and the Lorentz force on the flow. The operators, the background fields and the boundary
conditions are ported from the `conductive_IC` branch (commit `39487bbf`), formula by formula, into this branch's
conventions (2026-10-28: step 1, insulating walls; step 2, a conducting inner core).

**What works now:** a spherical shell (`ricb > 0`) with an insulating mantle and an insulating or conducting
(`'conducting, Chebys'`) inner core, `structure = 'uniform'`
(Boussinesq), `ViscosD > 0`, eigenvalue problems, $\mathbf{B}_0$ `'axial'` or `'FDM'` of degree 1. `background.check()`
stops on anything else, in particular on a variable density: the Lorentz force in the poloidal rows and the induction by
the poloidal flow need $(\ln\rho_0)'$ terms that are not implemented yet.

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
| `'FDM'` | the poloidal free decay mode of the shell, $j_l(\beta r)\,y_{l-1}(\beta) - j_{l-1}(\beta)\,y_l(\beta r)$ (Zhang & Fearn 1995), $\beta$ the root of `findbeta` near `beta` | $l$ = `B0_l` (1 only for now) |

Both match a potential field at the walls. `cnorm` is a number, `'rms_cmb'` (rms of the radial field over the outer wall
1), `'mag_energy'` ($\int B_0^2\,dV = 1$ over the shell) or `'Schmitt2012'` ($= 4$), computed as in `conductive_IC`.
`tol_B0` (default 1e-8) is the relative cut of the Chebyshev series of $r^p h^{(k)}$: it sets the band width of the
magnetic blocks.

With an equatorially antisymmetric $\mathbf{B}_0$ the field $\mathbf{b}$ has the symmetry opposite to $\mathbf{u}$
(`ut.bsymm = symm * symmB0`): $f$ takes the degrees of the toroidal flow and $g$ those of the poloidal flow.

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
- **Solver and output.** `solve.py` writes `real_magnetic.field`, `imag_magnetic.field` and normalises the eigenvectors to
  $\mathrm{KE} + \mathrm{Hendrik}\int\tfrac12|\mathbf{b}|^2dV = 1$ (over the shell, as `conductive_IC`).

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
| 1 | ME | $\int\tfrac12|\mathbf{b}|^2 dV$ (without Hendrik, as `conductive_IC`; Mag/Kin is ME/KE) |
| 2 | Mdfs | $\mathrm{MagnetD}\int\mathbf{b}\cdot\nabla^2\mathbf{b}\,dV$ |
| 3 | Indu | $\int\mathbf{b}\cdot\nabla\times(\mathbf{u}\times\mathbf{B}_0)\,dV$ |
| 4 | residb | resid𝐛 $= |2\sigma\,\mathrm{ME} - \mathrm{Indu} - \mathrm{Mdfs}|/\max$ |
| 5, 6 | Wlor, Wlor_im | $\mathrm{Hendrik}\cdot 2\int\mathbf{u}^*\cdot\mathbf{F}\,dV$, real and imaginary parts (in residσ, residω) |
| 7 | MEic | $\tfrac12\int|\mathbf{b}|^2/\mu_{i2o}\,dV$ in the inner core (its field, or the potential field below an insulating one) |
| 8 | MEext | $\tfrac12\int|\mathbf{b}|^2dV$ of the potential field outside the outer wall |
| 9 | Ohm_f | $\mathrm{MagnetD}\int|\nabla\times\mathbf{b}|^2dV$ over the fluid |
| 10 | Ohm_ic | $\mathrm{MagnetD}\,\eta_i\int|\nabla\times\mathbf{b}|^2/\mu_{i2o}\,dV$ over a conducting inner core |
| 11 | residE | resid𝐄, the full energy budget (nan without no-slip walls) |

resid𝐄 (`conductive_IC`'s, `utils4pp.em_budget`) closes
$2\sigma[\mathrm{KE} + \mathrm{Hendrik}(\mathrm{ME} + \mathrm{MEic} + \mathrm{MEext})] = \mathrm{Dkin} + \mathrm{Wthm} + \mathrm{Wlor}
+ \mathrm{Hendrik}(\mathrm{Indu} - \mathrm{Ohm\_f} - \mathrm{Ohm\_ic})$: unlike resid𝐛 it sees the wall and ICB conditions.
The $1/\mu_{i2o}$ of the inner core's energy and Ohmic loss is not in `conductive_IC`'s `em_budget` (there resid𝐄 is
$10^{-2}$ at `mu_i2o` = 2).

`params.dat` gains ten columns at the end: `magnetic`, `B0` (index in `background_field.B0_TYPES`, −1 without a field),
`B0_l`, `cnorm` (as a number), `Hendrik`, `MagnetD`, `innercore` (index in `background.INNER_CORES`), `N_cic`, `mu_i2o`,
`sigma_i2o`. The Lorentz and induction terms of spin_doctor are the Boussinesq
ones (`utils4pp.lorentz4pp`, `induction4pp`).

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
(`magnetic = 0`, structure `'uniform'`, no-slip) the hydrodynamic spin-over is λ = −0.1019947097 + 1.0057336901i, the
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

## Not yet implemented

In the order planned: the $(\ln\rho_0)'$ terms for any structure with
`ru_insane` and quadrature checks and the full energy budget of `conductive_IC` (resid𝐄), a toroidal $\mathbf{B}_0$; also
thin walls (`'TWA'`), rotdyn, the full sphere (the parity of $h$), the quadrupole couplings (`'Luo_S2'`, `'FDM'` with
`B0_l = 2`), the point `'dipole'`, inviscid and forced runs, and the induction by a differential rotation.
