# kore, `anelastic_log` branch: guide for Claude

kore computes linear modes (eigenvalue problems) and boundary-forced responses of a rotating, anelastic fluid in a
spherical shell or full sphere, with a spectral method: spherical harmonics in angle, Chebyshev/Gegenbauer in radius,
sparse matrices solved with SLEPc/PETSc (MUMPS). This branch adds variable density written with ln ρ₀ (hence "log"),
an entropy equation, a solar background, and differential rotation with its thermal wind.

Written 2026-10-13 against commit `b4598010` ("Radiative interior (Mukhopadhyay et al. 2025b) ..."), branch
`anelastic_log`, in sync with `origin` (github.com/repepo/kore). Line numbers drift; check the code before relying on a
detail here. The commit messages are long and accurate: `git log` is the best history. §15 lists what each thread of
the claude.ai project behind this branch found and changed.

## 1. Working here

- **Python**: `source ~/z2/bin/activate` (numpy, scipy ≥ 1.15.3 for `sph_harm_y`, mpi4py, petsc4py, slepc4py with
  complex scalars, sympy). `pygyre` and `astropy` are only needed for the file-based models, which are broken anyway (§12).
- **Memory**: keep numerical tests under 40 GB on the Mac laptop, 100 GB on the Linux server (mongo). MUMPS is the
  big consumer; see §9 for out-of-core options.
- **Commits**: leave changes uncommitted unless asked; the owner reviews and commits.
- **Where things run**: every script reads `bin/parameters.py` (imported as `from parameters import par`) and writes
  its output to the current directory; `operators.py` loads every `*.mtx` in the current directory. So run from the
  repo root (as `runKore.sh` does) or from a run directory holding its own `bin/` copy (as the test scripts build).
  `.gitignore` ignores `*.mtx *.npz *.field *.dat`.
- **Before trusting a change**: run `ru_insane.py` (operator symmetry, seconds), then the pipeline and `spin_doctor`
  residuals; for diff-rot or heat-equation changes, `tests/diff_rot/regress.sh` (bitwise A, B against an older checkout)
  and `tests/heat_dr/run_all.sh`. Existing runs are expected to stay bitwise identical unless a change says otherwise.
- The checked-in `bin/parameters.py` is the "gi1" gravito-inertial mode: ricb 0.35, m 2, thermal, 'user def',
  N 192, lmax 121, ncpus 10, target 2.04i; expected λ = −0.049653 + 2.037155i.

## 2. Pipeline (`runKore.sh`)

```
./bin/submatrices.py ncpus          # radial operator blocks -> <label>.mtx  (multiprocessing)
./bin/ru_insane.py                  # optional: operator sanity tests on the *.mtx, no MPI/SLEPc, exit 1 on failure
mpiexec -n ncpus ./bin/assemble.py  # A.npz, and B.npz (forcing = 0) or B_forced.npz (forcing > 0; both for 11)
mpiexec -n ncpus ./bin/solve.py [petsc opts]   # SLEPc shift-invert (forcing = 0) or KSP LU (forcing > 0)
./bin/spin_doctor.py ncpus | tee out2          # energy/entropy budgets, residuals, tables
```

`runKore.sh` then moves the outputs to `../run`. Constraint enforced by `assemble.py`: `lmax - m + 1` must be a
multiple of `2*ncpus` (number of MPI ranks); the default `lmax` formula in `params_default.py` satisfies it.

Outputs: `real_flow.field`, `imag_flow.field` (and `*_thermal.field`), one column per solution, eigenvectors
normalised to kinetic energy 1; `eigenvalues0.dat` (this solve, σ and ω columns); `timing.dat`. `spin_doctor` appends
to `eigenvalues.dat`, `params.dat` (76 columns), `flow.dat` (20), `thermal.dat` (7). A `params.dat` from an older version
has fewer columns: start a new one.

## 3. File map

`bin/` (≈8k lines):

| file | role |
| --- | --- |
| `parameters.py` | the run's parameters: `par = default_params()`, `par.timescale = 'rotation'; par.set_scales()`, then overrides |
| `params_default.py` | every parameter with long comments (the best parameter reference), `set_scales`, solver options `petsc_opts`, `prescale` |
| `utils.py` | globals (`N1`, `n`, `sizmat`, `symm1`, `adiab_tw`, `dr_high`, `tw_nmax`...), label decoding, Chebyshev transforms (`chebco_f`, `funcheb`, `fonzie`, `angine`), Gegenbauer matrices `Dlam`, `Slam`, `Mlam`, parity `remroco`, `ell`, erf profiles, `load_npz_mmap` |
| `radial_profiles.py` | background classes (`user_defined_profiles`, `solar_profiles`, `modelS_profiles`, `profiles_from_file`, `Boussinesq_profiles`), instance `prf`; diff-rot profiles `f_n`, thermal-wind σₙ; the `*X(r, Dorder)` derivative helpers; `proffdir` (label code → function); `burrito` (the integrand of one operator) |
| `smoothed_sun.py` | analytic even fit of the Model S ln ρ, with g, p, d ln ρ/dr, d ln p/dr as Chebyshev series |
| `model_s.at` | Model S table (columns As, c₁, ρ/ρ_c, p/p_c, x) for `solar_density = 'model S'` |
| `submatrices.py` | builds the list of labels, computes each unique operator in parallel, writes `<label>.mtx` |
| `operators.py` | loads every `*.mtx` as a global named by its label; physics functions `inertia`, `coriolis`, `viscous_diffusion`, `buoyancy`, `differential_rotation`, `entropy`, `thermal_advection`, `thermal_diffusion`, `thermal_diffrot` |
| `assemble.py` | MPI over l; places the blocks, boundary-condition rows (`bc_u_spherical`, `bc_theta_spherical`), forcings 6, 7, 9, 10, 11; normalises A, B by ‖A‖ (eigen); `gather_csr` |
| `reynolds_stress.py` | forcing 11: random Reynolds-stress body force f = ρ₀⁻¹∇·(ρ₀ s), the fluctuation-dissipation partner of the viscous force; `rhs(nsamp, seed)` gives one realisation per column of B_forced (docstring: physics, discretisation, normalisation) |
| `bc_variables.py` | Chebyshev values/derivatives at the walls (`Ta`, `Tb`) and ln ρ₀ and its first two derivatives there (`lha*`, `lhb*`) |
| `solve.py` | reads its own rows of A, B (memory-mapped), optional Ruiz scaling, SLEPc/KSP, KE normalisation, writes fields |
| `utils4pp.py` | post-processing: solution expansion, velocity/curl l-components, per-l workers (`flow_worker`, `thermal_worker`), `diff_rot_setup`, `diagnose`, `kinetic_energy`, `identify` |
| `spin_doctor.py` | runs the diagnostics per solution, prints the table, writes `params.dat`, `flow.dat`, `thermal.dat` |
| `ru_insane.py` | operator symmetry and diff-rot block tests (see its docstring) |
| `diff_rot_coefficients.py` | closed-form angular couplings of the P₀ and P₂ diff-rot force (44 functions, \|Δl\| ≤ 3) |
| `diff_rot_generic.py` | the same couplings for any Pₙ by Gauss–Legendre quadrature; `labels(n, section, component)`, `coef`, `max_dl` |
| `heat_dr_coefficients.py` | angular couplings of the heat-equation diff-rot terms from the X (cos θ) and D matrices: `coef('P2' / 'dP4' / 'dP6D' ...)` |
| `targets.py` | analytic targets (spin-over, Rieutord–Valdettaro attractor) |

Other directories: `tests/diff_rot/`, `tests/heat_dr/` and `tests/rsf/` (§11, each with a README); `docs/` (mkdocs; the branch's
physics notes are `differential-rotation.md` (user guide), `heat-equation.md` (derivation, implementation, benchmarks,
§9 radiative interior), `thermad0.md` (ThermaD = 0 with the thermal wind)); `tools/` (submission, plotting,
`kore2sprouts.py`, `m2petsc.py`), `koreviz/`, `koreviz_mod/` (visualisation); `examples/spinover/` (stale, §12);
`srunKore.sh` (SLURM); `poly.n1_gamma3.h5`, `theprofile.data` (model files).

## 4. Formulation

**Unknowns.** Fixed m and one equatorial symmetry `symm` (±1). Per degree l: poloidal P_l(r), toroidal T_l(r), and with
`thermal = 1` the entropy perturbation s_l(r), s = δs/c_p. Magnetic and compositional physics were removed.

**Velocity** (`utils4pp.cheb2space_pol`): u = ρ₀⁻¹∇×∇×(ρ₀ P **r**) + ∇×(T **r**), so ∇·(ρ₀u) = 0 exactly:
u_r = L P/r, consoidal part P′ + P(1/r + (ln ρ₀)′), L = l(l+1).

**Momentum** (anelastic, LBR form, per unit mass): rows are r̂·∇×∇× (section `u`, rows of P_l) and r̂·∇× (section
`v`, rows of T_l) of the equation, multiplied by r^rpower ρ₀^rhopower (`rpower_u` 5, `rpower_v` 3, `rhopower_u = rhopower_v`
1; the r powers are the smallest keeping every operator polynomial, larger ones only spoil residuals). Forces:
Coriolis 2 Gaspard ẑ×u, viscous ViscosD ρ₀⁻¹∇·(ρ₀ν S) (derivatives of ln ρ₀ up to 4th order), buoyancy
Beyonce g s r̂ (`u2gra0_D0`), differential rotation (§8).

**Heat equation** (section `h`), with p₀ standing for ρ₀T₀:
p₀(∂ₜs + u_r dS/dr) = ThermaD ∇·(κ p₀ ∇s) [+ diff-rot terms, §8]. Rows are r² × equation. With `ThermaD = 0` the rows
are divided by p₀ (r s and −r u_r dS/dr, profile `dSd`), better conditioned where p₀ ~ 1e-7 at the surface; with
`ThermaD = 0` plus diff rot and `thermal_wind ≥ 1` (`ut.adiab_tw`) they are r² × eq / p₀ in a C^(1) basis. No thermal
BC rows without diffusion.

**Generalised eigenproblem** A x = λ B x, fields ∝ e^{λt}, λ = σ + iω (negative ω is prograde). B is −inertia
(momentum) and +entropy (heat); in A the momentum rows are Coriolis − viscous − buoyancy + diff rot, the heat rows
−advection + diffusion + diff-rot terms (operators return the A sign). Forced runs (`forcing > 0`) use
A − i wf B as the matrix and a boundary right-hand side (forcing 11: random body forces, `reynolds_stress.py`).

**Scales.** `Gaspard` = Ωτ, `Beyonce` = (N₀τ)², `ViscosD` = ν₀τ/r₀² (the Ekman number when τ = 1/Ω),
`ThermaD` = κ₀τ/r₀². `set_scales()` resets them by `timescale` ('rotation': Gaspard 1, Beyonce 0, ViscosD = Ek,
ThermaD = Etherm), so set physical values **after** calling it. Eigenvalues are in units of 1/τ, in the frame
rotating at Gaspard/τ.

## 5. Numerics and matrix layout

- **Radial basis.** Chebyshev in x ∈ [−1, 1]: x ↦ r ∈ [ricb, 1] with an inner core (`N1 = N`), x = r ∈ [−1, 1] in a full
  sphere (`ricb = 0`), where parity halves the basis (`N1 = N/2`, N must be even, use about twice a shell's N).
  Each section is written in a Gegenbauer basis: `u` C^(4), `v` C^(2), `h` C^(2) (C^(0) if ThermaD = 0, C^(1) with
  `adiab_tw`); inviscid full sphere: `u` C^(2), `v` C^(1). Products are `ut.Mlam` (multiplication by a profile's
  coefficients), derivatives `ut.Dlam`, basis changes `ut.Slam`.
- **Sizes.** `n = N1*(lmax-m+1)/2`, `sizmat = 2n + n*thermal`. Unknown vector: [P blocks for the poloidal l's]
  [T blocks for the toroidal l's][s blocks, same l's as P], each block `N1` Chebyshev coefficients. `ut.ell(m, lmax, symm)`
  gives the poloidal l's, toroidal l's, all l's. Each row block starts with its BC rows (u: 4 with an inner core, 2
  without; v: 2/1; h: 2/1 when ThermaD > 0), made room for by dropping the last rows of the operator.
- **Full-sphere parity.** At ricb = 0, `submatrices` keeps one parity of rows and columns (`ut.remroco`) from the
  vector parity (vP = (−1)^m symm for P and s, −vP for T) and each operator's parity (r power, derivative orders,
  odd profiles in `ut.odd_profiles`). It prints a WARNING if a profile has wrong-parity coefficients: those parts are
  silently dropped. Profiles are therefore built even or odd: `pdS`, `dSd` as sign(r)·f(|r|), diff-rot profiles at |r|.
- **Operator labels** (`ut.decode_label`): `sX[fu1D][fu2D]_DY`, e.g. `u2lho1vsc0_D1`. `s` is the section; `X` lowers the r
  power (u: rpower_u − X, v: rpower_v − X, h: 2 − X, or 1 − X for ThermaD = 0 without `adiab_tw`; inviscid uses 3 − X,
  2 − X and ρ₀², ρ₀); each `fuD` is a 3-letter profile code and its derivative order; `_DY` is the derivative order on
  the unknown. `lh1`, `lh2` with order k mean (ln ρ₀)′·(ln ρ₀)^(k) and (ln ρ₀)″·(ln ρ₀)^(k). `rap.burrito` evaluates
  r^rpower ρ₀^rhopower f₁ f₂ on the Chebyshev nodes; `ut.chebco_f` transforms it (coefficients below 1e-15 of the
  largest are zeroed).
- **Profile codes** (`rap.proffdir`): `lho` ln ρ₀, `vsc` ν, `gra` g, `pss` p₀, `pdS` p₀ dS/dr, `dSd` dS/dr, `kps` κp₀;
  diff rot `abu`/`spv`/`psl` = f₀, r f₀′, r² f₀″ and `aub`/`svp`/`pls` = f₂, r f₂′, r² f₂″, `ab4`/`sv4`/`pl4`... for
  n ≥ 4 (`ut.dr_code`); thermal wind `tw2`/`tp2` ... = p σₙ, p σₙ′ (`ut.tw_code`, n up to 34 with letters). Codes not
  in `proffdir` are added on demand by `rap._add_generic_profiles`.
- **Row scaling.** Every operator in `operators.py` is divided by L² or L³ (L of the row) so that all rows are
  (equation)·r^power/L³; terms with an explicit L from u_r = LP/r use L². `differential_rotation` was missing this until
  `26d595eb` (every diff_rot run before that is wrong).
- **Assembly.** `assemble.py` scatters the l's over MPI ranks, collects COO pieces with `ut.packit` and gathers with
  `Gatherv`; eigenproblem A and B are divided by ‖A‖_F.

## 6. Background models (`par.model_type`)

- `'user def'` (`user_defined_profiles`): Wu 2005 polytrope-like β = 2, ln ρ₀ = ln (1 − r₁²)², r₁ = r·aux0, analytic g,
  p, d ln p/dr; ν = κ = 1. `aux0` is the "peel" so that ρ₀(1) > 0 (keep it ≥ 0.99, never below 0.985).
  dS/dr = erf top hat `ut.erf_top_hat(r, aux1..aux5 = x1, w1, x2, w2, A)`. Other density laws are commented in place.
- `'smoothed sun'` (`solar_profiles`): length unit aux0 R☉ (x = r·aux0), ρ, g, p from `smoothed_sun.py`
  (`solar_density = 'fit'`, up to 48 % off Model S near the surface) or `modelS_profiles` (`'model S'`: Chebyshev fits of
  ln ρ, ln p from `model_s.at`; shell fit of degree `solar_density_deg` on `solar_density_window`, or with `'sun'`, the
  default at ricb = 0, one even whole-Sun fit of degree `solar_density_sun_deg`). dS/dr is the odd top hat
  `ut.erf_top_hat_odd` (x1 = 0 is fine at ricb = 0), ν/ν_cz, κ/κ_cz step from `visc0`, `therm0` below `aux3` (width `aux4`)
  to 1, times (ρ(x2)/ρ)^`visc_pow`/`therm_pow`. Model S is fitted by aux1..5 = 0, 0.03, 0.68, 0.06, 2.6;
  Beyonce ≈ 4.9e4 for the solar rate.
- `solar_rz = 1` (with 'smoothed sun'): the radiative interior of Mukhopadhyay, Bekki, Zhu & Gizon 2025b (their eqs.
  8–10): dS/dr = A/4 [1 − erf((x−x1)/d)][1 − erf((x−x2)/d)] erf(x/w0) with `solar_rz_dSdr`, `solar_rz_steps`,
  `solar_rz_core`, and the diffusivity step at `solar_rz_os` instead of aux3/aux4. `tests/diff_rot/mk_rz.py` builds
  their set-up for any inner radius; results in `docs/heat-equation.md` §9.
- `'poly'`, `'mesa'`, `'gsm'`, `'astropy table'` (`profiles_from_file`) and `'Boussinesq'`: broken on this branch (§12).
  Note that the default in `params_default.py` is `'poly'`, so `parameters.py` must set `model_type`.

To add a model: subclass `user_defined_profiles` (or `solar_profiles`) with `logrho`, `gravity`, `pressure`, `dlog_p`,
`dlog_rho`, `gradS`, `viscosity`, `thermal_diffusivity` (each `f(r, rpower)` returning `r**rpower * value`, valid on
[−1, 1] with the right parity if ricb = 0) and select it at the bottom of the class section of `radial_profiles.py`.

## 7. Boundary conditions (`assemble.bc_u_spherical`, `bc_theta_spherical`)

- `bco`/`bci` = 1 no-slip: P = 0, P′ + P((ln ρ₀)′ + 1/r) = 0, T = 0.
- 0 stress-free, impenetrable: P = 0; r²P″ + r²(ln ρ₀)′P′ + (r²(ln ρ₀)″ + L − 2 − r(ln ρ₀)′)P = 0; rT′ − T = 0.
- 2: zero radial stress instead of P = 0: (r(ln ρ₀)′ − 3)P + 3rP′ = 0.
- `bci`, `bci_thermal` are ignored at ricb = 0 (regularity comes from the parity basis).
- Thermal (ThermaD > 0 only): `b*_thermal` 0 fixed s, 1 s′ = 0.

## 8. Differential rotation (`diff_rot = 1`)

User guide: `docs/differential-rotation.md`. Background Ω = Ω_ref [1 + ΔΩ], ΔΩ = Σₙ aₙ fₙ(r) Pₙ(cos θ), n even; flow
U₀ = Gaspard ΔΩ r sin θ e_φ in the frame Ω_ref. Only 'user def' and 'smoothed sun' define the profiles.

- **Types** (`diff_rot_type`, amplitude `diff_rot_amplitude`): `'Y20'`, `'rigid'` (test: exact Doppler identity),
  `'Y20-wall-bounded'`, `'shellular'`, `'cylindrical'`, `'conical'` (Guenel et al. 2016), `'shellular_boussinesq'`,
  `'solar'` (P₀ + P₂ tanh tachocline, frame 435 nHz), `'solar_P4'` (c₀ + c₂cos²θ + c₄cos⁴θ with erf tachocline,
  `diff_rot_solar`, frame `diff_rot_frame`), `'tabulated'` (Ω(r, θ) from `diff_rot_file`, .npz with `ome[θ, r]` nHz,
  `r` R☉, `theta` rad; projected on P₀..P_`diff_rot_nmax`; shell fit of degree `diff_rot_rdeg`, or `diff_rot_window =
  'sun'` (default at ricb = 0) one even fit of degree `diff_rot_sun_deg`, optionally held below `diff_rot_hold`).
  Per-term factors `diff_rot_term_amp` (a₀...a₁₆). f₀, f₂ are `prf.abu`, `prf.aub`; n ≥ 4 come from `rap.f_new`.
- **Momentum** (`operators.differential_rotation`): G[ΔΩ(i m u + 2ẑ×u) + r sin θ (u·∇ΔΩ) e_φ]; P₀, P₂ from the closed
  forms (`diff_rot_coefficients.py`), n ≥ 4 (`ut.dr_high`) from `diff_rot_generic.py` (`diff_rot_generic = 1` rebuilds
  P₀, P₂ that way as a self-check). Couplings |Δl| ≤ n (P→P, T→T) and n + 1 (P↔T); `ut.dr_offd` gives the column block.
- **Heat** (`operators.thermal_diffrot`, `thermal = 1`): i m G ΔΩ s always; `thermal_wind` 1 (linear) / 2 (exact
  (1+ΔΩ)², frame independent) add u_r ∂S_tw/∂r + (u_θ/r)∂S_tw/∂θ with the thermal-wind entropy S_tw = Σ σₙ Pₙ
  (`rap.sigma2/4/6`, `sigma_all` for high n; needs Beyonce > 0); 3 prescribes ∂S_tw/∂θ = `dSdtheta` sin 2θ (Bekki et
  al. 2024); 0 none. `thermal_wind_radial = 0` drops the u_r term (as Bekki et al. and Mukhopadhyay et al.);
  `thermal_wind_amplitude` scales S_tw. Couplings up to `ut.tw_nmax` (n_max + 2, or 2 n_max + 2 for thermal_wind 2).
- ViscosD, ThermaD, Beyonce must be computed for Ω_ref. At ricb = 0, 'Y20-wall-bounded' and 'shellular' (unless σ is even
  and > 0) have a kink at r = 0 (WARNING); 'solar' needs N ~ 400.
- Benchmarks: Guenel et al. 2016 conical mode ω = 1.8173; solar inertial modes of Mukhopadhyay et al. 2025
  (`tests/diff_rot/run_bekki.py`, needs their Edmond `diff_rot.npz`, not in the repo): five modes within 0.2 nHz in
  frequency and 0.4 nHz in growth rate of their compressible values; tables in `docs/heat-equation.md` §6, §8, §9.

## 9. Solver (`solve.py`)

- Options come from `par.petsc_opts` (only the eps_/st_ group for eigenproblems, ksp_/pc_/mat_ for forced); command-line
  options win. Defaults: shift-invert at τ = rtau + i·itau, MUMPS LU, `cntl_1` 0.01, BLR off (`icntl_35 = 0`; set 2
  with `cntl_7 = 1e-14` for large N). For big full-sphere runs: `st_mat_mumps_icntl_14` 50–200 (INFOG(1) = −9 otherwise)
  and out-of-core `st_mat_mumps_icntl_22 = 1`.
- `par.prescale`: None = Ruiz equilibration of A − τB, on except for thermal with ThermaD = 0 (where it hurts); 0/1
  force it. Eigenvectors are mapped back before writing. The residual SLEPc prints is for the scaled problem; judge
  accuracy with spin_doctor.
- `which_eigenpairs` 'TM', `nev`, `tol` 1e-16, `maxit` 50.
- Forced runs solve every column of B_forced.npz with one LU (forcing 11: `rsf_nsamp` realisations; the fields files
  get one column each). Forcing 11 with `rsf_frequencies` set: for each frequency w, one LU of A − i(w − wf)B
  (B.npz unnormalised) and back-substitutions for all realisations; appends to `rsf_spectrum.dat` (w, mean KE,
  standard error; mean KE = two-sided PSD, ∫ dw/2π gives the mean KE for m ≠ 0, twice it for m = 0; of the solved symm
  class) and `rsf_ke.dat`, no fields. The noise amplitude is applied at r, so nothing forces where ν_t = 0. To keep
  the forcing out of the tail of a viscosity step use `rsf_rcut` (a smooth erfc edge, `rsf_rcut_width`), not a higher
  `rsf_nu_floor`: clipping ν_t leaves a corner the rows cannot resolve.

## 10. Post-processing (`spin_doctor.py`, `utils4pp.py`)

Radial integrals use Gauss–Legendre with 3N/2 nodes (`upp.setup_grid`); energies are ρ₀-weighted. Per solution the
table prints σ, ω, 𝒯/𝒫, the residuals, peak ℓ, ℓ-width, ℓ-convergence, max of the curl of the viscous force
(cvf_r, cvf_l, cvfmax; r < 0.01 skipped at ricb = 0) and residens.

- residσ (`resid1`): 2σKE − Dkin − Wthm − Wdr = 0; residω (`resid2`): the imaginary part with Coriolis;
  resid𝑠 (`resid3`): 2σTE − Dthm − Wadv − Wdr_thm = 0 (TE = ½∫p₀|s|²); residens: enstrophy balance.
- These residuals, not the eigenvalue alone, judge resolution: compare with a higher-N run; for thermal runs the
  entropy can be unresolved while P, T look converged. The docs report 1e-7 to 1e-12 for converged runs.
- `params.dat` columns are `spin_doctor.param_table()` in order (append new ones at the end); `diff_rot_type` and
  `solar_density` are written as indices into `diff_rot_types`, `solar_densities`; None as nan. `flow.dat`,
  `thermal.dat` columns are the `flow_cols`, `thermal_cols` lists.
- With diff rot, `upp.diff_rot_setup` computes the force and heat terms on an (r, cos θ) grid and projects them back.

## 11. Tests

- `bin/ru_insane.py [nvec] [--tol 1e-6]` after `submatrices.py`: inertia Hermitian positive, Coriolis anti-Hermitian,
  viscous Hermitian negative, entropy/diffusion blocks, buoyancy/advection against quadratures, diff-rot blocks against
  an (r, θ) grid. Needs N ≥ 96 (fails on truncation below); does not cover the inviscid formulation.
- `tests/heat_dr/` (README): `mk_run.py NAME 'par.x = ..'` builds a run dir (defaults: user def, ricb 0.35, m 1, conical
  −0.25, ViscosD = ThermaD = 1e-3) and runs submatrices + assemble; `heat_check.py`, `asm_check.py`, `row_check.py`,
  `rigid_thermal.py`, `tw_sympy.py`, `csolve.py` (scipy solve, no equilibration), `sd_check.py`, `ebal.py`, `terms.py`,
  `mpi_check.sh` (bitwise A, B on 1–4 ranks), `forced_check.py`; `run_all.sh` runs them all (~10 min).
- `tests/diff_rot/` (README): `gen_check.py` (generic vs closed-form couplings), `regress.sh OLD_KORE` (13 configurations,
  A and B bitwise equal), `tw_edmond.py`, `mk_bekki.py` / `solve_bekki.py` / `run_bekki.py` (solar benchmark),
  `mk_rz.py` (radiative interior, any inner radius), `ls_to_edmond_layout.py` (HMI table converter).
- `tests/rsf/` (README): forcing 11. `builder_check.py` (f = u against the inertia rows), `noise_check.py` (the noise
  amplitude against the documented Θ laws, and the noise against the viscous operator, exactly; no forcing beyond the
  dissipation bound), `spectrum_check.py pick|fields|check` (solve.py's sampled spectrum against the exact E[KE(w)]
  and the single-mode energy Θ f_eff); `run_all.sh` (needs petsc4py for the solves, and an `MPIEXEC` that starts
  mpi4py's MPI); `regress_solve.sh OLD_KORE` (eigen and forcing-10 solves bitwise against an older checkout).
- `examples/spinover/unitTest.py` is the upstream test and is stale here (§12).

## 12. Known issues and gotchas

- **File-based models and 'Boussinesq' are broken**: `profiles_from_file` and `Boussinesq_profiles` have no `logrho`, and
  `bc_variables.py` calls `rap.logrhoX` at import, so they should fail at once (inferred from the code, acknowledged as
  untouched in `docs/heat-equation.md` §7). 'Boussinesq' also reads `par.heating`, which `params_default` only sets when
  `model_type` is already 'Boussinesq' at construction (never), and has no diff-rot profiles, so the Mirouh et al. 2016
  benchmark cannot run.
- **Inviscid** (`ViscosD = 0`): `submatrices` switches to the inviscid bases only when also ricb = 0, while
  `assemble.bc_u_spherical` uses one BC row whenever ViscosD = 0, so ViscosD = 0 with ricb > 0 looks inconsistent.
  Untested on this branch.
- **ThermaD = 0**: spurious modes trapped at the inner boundary unless dS/dr → 0 there (keep x1 ≥ ricb + 3 w1); a tiny
  ThermaD (1e-9) reproduces ThermaD = 0 eigenvalues except with the thermal wind and fixed-entropy walls (boundary
  layer, `docs/thermad0.md`). Raise `nev` by 1–2 if spurious modes take slots.
- **ricb = 0**: profiles must be smooth and of the right parity at r = 0 (dS/dr from 'user def' needs x1 ≥ ~6 w1).
  Model S profiles need ~100 Chebyshev polynomials per parity; ru_insane fails at N 96 there.
- **Open physics items** (§15, threads 11 and 13): the thermal wind omits the p₁S₀′ term, which matters where S₀′ ≠ 0
  (the `solar_rz` interior); with the diffusionless thermal wind there are small-scale growing modes peaking at
  l ≈ lmax that do not converge in lmax (physical or not is open); the σₙ (ln ρ₀)′ heat rows converge slowly in N near
  the surface for 'user def' with aux0 → 1 at ricb = 0 (only `tests/heat_dr/row_check.py` sees it); two growing modes
  in the m = 3 n0 window of the solar benchmark (−260.6 nHz, growth +18.0; −250.5, +5.0) have not been examined.
- `params_default` says `rpower_pp`, `rhopower_pp` are unused, but `utils4pp.setup_grid` uses them as the enstrophy
  weight `const0`.
- `utils4pp` calls `mp.set_start_method('fork')` at import (needed on macOS).
- Stale upstream leftovers: `tools/submit4.sh` calls the removed `postprocess.py`; `examples/spinover/` uses the old
  parameter file with magnetic options; `.github/workflows/main.yml` targets the `variable_density` branch, so no CI
  runs on this branch (`ci.yml` only deploys the mkdocs site from main/master); the top-level `README.md` is upstream's.
- `rvisc`, `hvisc` are only echoed to `params.dat` (the 'user def' viscosity step is commented out).

## 13. Recipes

- **New parameter**: default and comment in `params_default.py`; if it should be recorded, append a tuple at the end
  of `spin_doctor.param_table()` and update the list in `docs/differential-rotation.md` (Output).
- **New radial profile in an operator**: a function `f(r, rpower)` and its `fX(r, Dorder)` in `radial_profiles.py`, a
  3-letter code in `proffdir` (and in `ut.odd_profiles` if odd), labels in `submatrices.py` with the vector parity of the
  unknown's column (vP for P and s, vT for T, as the neighbouring labels), the term in `operators.py` with the L²/L³
  division, the block in `assemble.py`. Check with `ru_insane.py` and a balance in `spin_doctor`.
- **New diff-rot type**: `aub`/`abu` branches in `user_defined_profiles` (or `f_new` for n ≥ 4 and `ut.dr_high`), the
  index in `spin_doctor.diff_rot_types` (append), the table in `docs/differential-rotation.md`; run `regress.sh`.

## 14. History (this branch, newest first)

`b4598010` radiative interior, whole-Sun Model S and tabulated fits, ricb 0 for 'model S', `diff_rot_hold`;
`dc2790d3`, `bc6c7eed`, `e42b55df`, `16be479f` docs; `f9b733ce` diff-rot options in params.dat; `92a0d020` any Pₙ,
'solar_P4', 'tabulated', amplitude controls, Model S density; `18641206` Gaspard in diff rot, thermal_wind 2 and 3,
ThermaD = 0 with diff rot, spin_doctor diff-rot powers (fixed the old `flow_worker` crash); `be2a4db5` diff-rot terms
in the heat equation; `652e1490` 'smoothed sun'; `26d595eb` 1/L³ on the diff-rot operator; `6799c90e` solver defaults
(Ruiz auto, BLR off); `8d44f177` rpower/rhopower defaults; `dc140f06` gi1 parameters; `357f83b7` spin_doctor tables;
`42f4ccca` rhopower = 1; `102bccc8` relative tolerance in `chebco_f`; `d0da30dc`, `8eb16917`, `2e64687c` cleanup
(stale scripts, magnetic/compositional code, body forcings 1–5, 8 removed); `2cd8f2be` thermal path fixed, budgets;
`00bc67e3` ru_insane; `6490a1b3` conductive_IC solver port, KE normalisation; `ec94dd6a` ρ₀ weight in dissipation,
Gauss–Legendre quadrature. Earlier: the `differential_rotation` merge `5446485f`.

## 15. Project threads and their end results

The branch was built and checked in the claude.ai project "kore - anelastic_log". Its threads, oldest first, with what
each one left behind. `project-files/...` is the project's shared folder (`/mnt/project-files/` inside a project
session, not in this repo); run folders are on the Mac (`~/lab/...`) or on mongo.

1. **Gas giant tidal forcing literature** (Oct 3–8, left open). Literature notes
   (`project-files/literature/tidal-forcing-gas-giants.md`) and a derivation of the equilibrium/wavelike split for
   kore's per-unit-mass rows (`wavelike-split-derivation.md`, LaTeX in `latex/`, sympy checks in `checks/`;
   `anelastic-tidal-forcing-kore.md` is superseded). With a potential flow carrying the bulge, only three sources force
   the waves: toroidal l = 3, poloidal l = 2 (∝ ρ₀′/ρ₀) and entropy l = 2 (∝ s₀′); the ω² part drops out. Its §10 gives
   kore's harmonic normalisation and signs. Nothing was implemented: kore has no tidal body force.
2. **Solar eigenmodes with kore** (Oct 5). Feasibility notes, `project-files/sun/solar-eigenmodes-notes.md`, written
   before any code check: density contrast manageable, results in the unstable convection zone depend on the chosen
   superadiabaticity and diffusivities, radiative zone and tachocline stiff, differential rotation the main gap. Threads
   12, 13 and 15 did the solar work.
3. **kore anelastic_log code summary** (Oct 8). First summary at the merge `5446485f`
   (`project-files/kore/anelastic_log-code-summary.md`, refreshed in thread 10, superseded by this file) with a bug
   list, most of it fixed since (the file-based and Boussinesq models still lack `logrho`).
4. **Testing without forcing or differential rotation** (Oct 8–9). Same eigenvalues as the pre-merge code. `ec94dd6a`
   (ρ₀ weight in the viscous dissipation, Gauss–Legendre quadrature); `6490a1b3` (conductive_IC solver port:
   `petsc_opts`, Ruiz, memory-mapped rows, KE = 1, Gatherv assembly; runKore.sh runs spin_doctor into `out2`). Then the
   thermal-path review, `project-files/kore/thermal/thermal-entropy-review.md`: the shell entropy coupling matches an
   independent solver to 1e-9, and `2cd8f2be` fixed `kappressX` (any ThermaD > 0 crashed), the (κp₀)′s′ coefficient (2,
   should be 1), the parity of p₀ dS/dr and Mlam at ricb = 0 and the missing −iω_f entropy term of forced runs; it also
   added the buoyancy work to residσ and the entropy budget resid𝑠, divided the heat rows by p₀ when ThermaD = 0, and
   changed MUMPS `cntl_1` from 1e-8 to 0.01 (1e-8 made the residuals grow with N). anel10 with thermal = 1 then closes
   all three budgets to 1e-8 or better (modes 0–7) at N = 480, lmax 101.
5. **Balance checks for anelastic LBR** (Oct 8–9). In `project-files/kore/balance/`: `energy-balance-derivation.md`
   (energy, frequency and entropy budgets from the momentum equation; the pressure does no work because the compression
   work cancels the pressure part of the buoyancy; the budgets read off the matrix rows, `ebal.py`; a convective case,
   conv1, with every term O(1)); `anelastic-limitations.md` (sound filtering, of size ω max(ω, 2Ω, N)/(c²k²); the
   dropped pressure-buoyancy term, of size N²ℓ/g; and the background must be a possible hydrostatic state, c² > 0, which
   kore cannot check. Both truncations are below 1e-3 for inertial modes of the Sun and the Earth's core, 2–25 % in
   Jupiter, and too large for low-order g modes near N in a strongly stratified zone);
   `viscous/viscous-operator-accuracy.md` (the 1e-8 anti-Hermitian part of the poloidal viscous block came from
   cancelling ln ρ₀ derivatives near the wall; rows times ρ₀ cure it). Code: `00bc67e3` ru_insane.py; `d0da30dc`,
   `8eb16917`, `2e64687c` the cleanup (−5956 lines, A and B bitwise identical on 15 configurations;
   `project-files/kore/cleanup/NOTES.md`); `102bccc8` relative cut in `chebco_f` (the absolute 1e-9 cut became the error
   floor for small products); `42f4ccca` rhopower 1, residω. The peel rule (§6) was set here.
6. **Align spin_doctor with conductive_IC** (Oct 9). `357f83b7`: list-based output tables, residσ in flow.dat.
7. **Gravito-inertial mode with buoyancy and viscosity** (Oct 9). `dc140f06`: the checked-in gi1 `parameters.py`, a mode
   damped 44 % by viscosity and 56 % by buoyancy work (the split moves with ThermaD; see its header comment).
8. **Full-sphere ricb=0 check on mongo** (Oct 9–10). `project-files/ricb0_check-20261009T221302-fcab.md` (the fullest of
   three versions): ricb = 0 eigenvalues sit where ricb = 0.05, 0.02, 0.01 extrapolates (differences ∝ ricb³) for m =
   0–3 and both symmetries; residuals of 1e-7 at N = 192 are resolution (1e-14 at N = 384), hence ~2N for a full sphere;
   eigenfunctions go as r^l at the centre; rpower_u 5, rpower_v 3 is the right pair (lower breaks ricb = 0, higher costs
   residuals). `8d44f177`: rpower/rhopower defaults; cvfmax skips r < 0.01 (round-off at the innermost node). Runs in
   `mongo:/nvm/lab/anel_dev`.
9. **Ruiz and l-dependent row scaling** (Oct 9–10). `project-files/kore/scaling/scaling-study.md`: l-dependent row
   factors cannot fix the conditioning (the disparities are boundary rows against operator rows, and Chebyshev index
   within a block) and add nothing on top of Ruiz, which takes κ₂(A − τB) from beyond double precision to ~5e7 at N =
   128 and helps except at exactly ThermaD = 0 (5–1e4× worse). The ThermaD = 0 spurious modes come from dS/dr ≠ 0 at the
   inner boundary: make dS/dr vanish there (or at r = 0) and prefer a small ThermaD > 0 with prescale on (ThermaD 1e-9
   at ricb 0.02 needs N ≥ 560). `6799c90e`: automatic prescale, BLR off, the advice as comments in `params_default.py`.
10. **Refresh the code summary** (Oct 10–11). Thread 3's summary brought to `6799c90e`, correcting one claim: the
    diff-rot operator had its density terms; its real bug was the missing 1/L³ (fixed in thread 11).
11. **Differential rotation equations and benchmarks** (Oct 11–12). `26d595eb` (1/L³ on the diff-rot operator: every
    earlier diff-rot run is wrong), `be2a4db5` (diff-rot terms in the heat equation), `18641206` (Gaspard factors,
    thermal_wind 2 and 3, ThermaD = 0 with diff rot, spin_doctor `diff_rot_setup`, ru_insane diff-rot tests,
    `tests/heat_dr`, `docs/thermad0.md`). Benchmarks: rigid-rotation offset identity, Guenel et al. 2016 conical m = 0
    (ω ≈ 1.817). Notes in `project-files/kore/diffrot/README.md`. Its final summary, the project doc "kore
    anelastic_log: differential rotation, final summary", lists 15 open problems: thread 13 addressed the first four (no
    switch for the u_r ∂S_tw/∂r term, Ω limited to P₀ + P₂, two Bekki et al. results not reproduced, the 'smoothed sun'
    density); what remains is in §12.
12. **Sun-like background profile** (Oct 11). `652e1490`: model_type 'smoothed sun' (smoothed Model S ρ, g, p; odd erf
    top-hat dS/dr; ν, κ steps; length unit aux0 R☉, Beyonce 4.90e4 for the Carrington rate; N ≳ 11/w1), 'user def'
    matrices bit-identical. Notes in `project-files/sun/solar-profile.md`. 'Boussinesq' was found already broken and
    left for the owner.
13. **Bekki 2024 gap: plan for review** (Oct 12–13). Step 1
    (`project-files/kore/diffrot/checks/bekki/step1/STEP1-SUMMARY.md`): no growing modes were missed, the entropy
    coupling and the units are right except c_p (the papers use 4.17e8, kore runs used 3.46e8), and the gap comes from
    the radial shear of Ω in the lower convection zone, absent from a radially uniform P₀ + P₂ profile. Step 2:
    `92a0d020` (any even Pₙ by quadrature, 'solar_P4', 'tabulated' Ω(r, θ), amplitude controls, `thermal_wind_radial`,
    `solar_density = 'model S'`) and `f9b733ce` (diff-rot options in params.dat). With the observed Ω (Edmond data) the
    five Mukhopadhyay et al. 2025 modes agree within 0.2 nHz in frequency (P₂ only: off by 0.8–9 nHz, and the
    high-latitude mode damped instead of growing); the convective onset is |Δθ_s| 513 (Edmond Ω) and 658 (Larson &
    Schou) in the papers' c_p, against Bekki's ~500. Runs in `~/lab/kore_step2_runs`; tables in `docs/heat-equation.md`
    §8.
14. **Differential rotation docs review** (Oct 13). `16be479f` (user guide `docs/differential-rotation.md`,
    repo-relative references, solar benchmark scripts); `e42b55df`, `bc6c7eed`, `dc2790d3` (math that renders on GitHub:
    no `\tag`, plain-text equation numbers).
15. **Inner boundary scan for Mukhopadhyay modes** (Oct 13). `project-files/REPORT.md`, `ricb_scan.png` (runs in
    `~/lab/ricb_scan`): with the radiative interior of Mukhopadhyay et al. 2025b, the inner radius changes no mode by
    more than 0.01 nHz once it is below ~0.63 R☉, down to the full sphere; the dependence sits in 0.64–0.71 R☉. The
    radiative zone shifts the modes by 0.4–13 nHz and damps them; m = 2 Rossby lies near the ℓ = 2 RZ Rossby sequence
    and m = 1 high-latitude mixes with ℓ = 3 RZ modes. CZ-only frequencies differ from their Table 1 by 1–10 nHz
    (candidates: anelastic vs compressible, their thermal wind, their free surface). With an adiabatic interior instead,
    the modes kept moving as the boundary went deeper. `b4598010` ported `solar_rz`, the whole-Sun 'sun' fits,
    `diff_rot_hold`, `mk_rz.py` and docs §9. `background_fits.png` and `density_fit_vs_modelS.png` compare the fits with
    Model S: 'sun' within 1.7e-3 in ρ (worst at 0.70–0.72 R☉) and 2.2e-4 in p; the shell fit 2.1e-4 and 2.8e-5; 'fit'
    ~10 % wiggles and 48 % near the surface; `solar_rz` makes N² vanish ~0.04 R☉ below Model S's drop at 0.713 R☉, while
    the aux top hat leaks N into the convection zone up to ~0.8 R☉.
16. **Code guide for Claude** (Oct 13). This file; a copy is `project-files/kore/CLAUDE.md`.
