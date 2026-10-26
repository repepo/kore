# Checks of the differential rotation (bin/diff_rot.py: any Pₙ, 'solar_P4', 'tabulated', 'user')

Companion of `docs/heat-equation.md` §8 and of `tests/heat_dr/` (whose `csolve.py`, `sd_check.py` and `ebal.py`
handle any Δl range and are used here too). Every script runs from a kore run directory (`bin/` copy, `parameters.py`;
`ebal.py`/`sd_check.py` also need the `*.mtx` files of `submatrices.py`) and prints a verdict ending in `ok` or `FAIL`.
The `*_bekki.py` scripts are the exception: they run from anywhere and build their own run directories from the
repo's `bin/`.

| script | what it checks |
| --- | --- |
| `gen_check.py [mmax [nmax [lmax]]]` | the couplings of `bin/diff_rot.py` (the recurrence algebra of `bin/angular.py`) against the closed forms of `tests/diff_rot/diff_rot_coefficients.py` (f₀, f₂, every coefficient, m ≤ mmax, l ≤ lmax, \|Δl\| ≤ 3), relative to the largest coefficient of the same row block; for n = 4…nmax that no coupling falls outside the Δl ranges kore assembles (P→P, T→T: \|Δl\| ≤ n, even; P↔T: \|Δl\| ≤ n + 1, odd); and for runs with P₀, P₂ only and thermal_wind 1 or 2, the σₙ of `diff_rot.sigma_coefficients` against their closed forms. Tolerance 1e-12; `lmax` overrides the run's. Run it by its path from a run directory (it imports the closed forms from its own directory) |
| `angular_check.py [--quick]` | standalone (no run directory): `bin/angular.py` — the harmonics of `ylm_grid` against exact rational P_l (l ≤ 2000), scipy (l ≤ 640) and their norms (l ≤ 3000); X, D on a grid and their adjoints; the vector operations (g(x)·, ẑ×, sinθ F_θ) against grid projections at low l and around l = 1200; the 12 primitives of `diff_rot._primitives` against `quad_reference.py` (n ≤ 16) |
| `diff_rot_coefficients.py` | not a check: the closed forms of the P₀, P₂ couplings and of σ₂, σ₄, σ₆ (formerly in `bin/`), the reference of `gen_check.py` |
| `quad_reference.py` | not a check: the former Gauss–Legendre quadrature builder of the primitives (scipy harmonics, valid for l ≤ 645), kept as the reference of `angular_check.py` |
| `tw_edmond.py diff_rot.npz [c_p]` | a 'tabulated' run reading the Mukhopadhyay et al. Edmond file: kore's ∂θS_tw against their `dsdt`, at three latitudes and six radii, the pole-to-equator contrast, and the rms mismatch over the shell |
| `regress.sh OLD_KORE [WORKDIR]` | builds 15 diff-rot configurations (conical, Y20, cylindrical, solar, shellular, Y20-wall-bounded, solar_P4, a 'user' profile with degrees 0, 2, 4; thermal_wind 0–3; ricb 0; ThermaD 0; the '8-term fit' solar structure; Gaspard 3; m 2; diff_rot 0) with an older checkout and with this one and requires A, B bitwise equal, or equal to 1e-13 of each row's scale ('round-off'). Each configuration is written in the parameter names of three generations of checkouts (with `bin/background.py`; with `bin/diff_rot.py` but `model_type` and `aux0`; before `bin/diff_rot.py`: `diff_rot_amplitude`, σ of 'shellular' as the amplitude), and the 'user' one is built only against checkouts that have it |
| `run_bekki.py DIFF_ROT_NPZ [WORKDIR]` | the like-for-like solar benchmark of `docs/heat-equation.md` §8: builds four runs with `mk_bekki.py` (the Mukhopadhyay et al. 2025 Ω file, nmax 12, thermal_wind 2, `thermal_wind_radial` 0, N 64, lmax m + 63), solves near five published modes with `solve_bekki.py` and prints kore against the paper; the verdict compares kore with the §8 values to 2e-3 nHz (a regression check). DIFF_ROT_NPZ is `data/diff_rot.npz` of that paper's data set (Edmond, doi:10.17617/3.NP26AI) |
| `mk_bekki.py NAME [key=val ...]` | not a check: builds and assembles one run of that solar set-up (Model S density, shell 0.71–0.985 R☉, δ = 0, ν = κ = 1e12 cm²/s, stress-free, ∂ᵣs = 0); keys for m, symm, N, lmax, the rotation profile (`drtype`, `file`, `nmax`, `frame`, `amp`), `tw`, `rad`, `ds` (Δθs in erg/g/K for thermal_wind 3), `nu`, `kappa` and `extra` python lines; see its docstring |
| `mk_rz.py NAME [key=val ...]` | not a check: builds and assembles the radiative-interior set-up of Mukhopadhyay et al. 2025b (arXiv 2512.12691; `docs/heat-equation.md` §9) for any inner radius `rin` (R☉, 0 for the full sphere): structure 'old model S prototype' fitted over the whole Sun, stratification and diffusivity 'solar_rz', 'solar_P4' or a 'tabulated' Ω with `diff_rot_window = 'sun'` (`hold=` x_h sets `diff_rot_hold`, which is faulty: do not use it), stress-free, s′ = 0 at the bottom and s = 0 at the top; see its docstring |
| `solve_bekki.py RUN freq [growth [k [ref]]]` | not a check: scipy shift-invert solve (Ruiz-equilibrated, as `solve.py` with prescale 1) of a `mk_bekki.py` run near freq + growth (nHz), reported in a frame at ref = 456 nHz |
| `ls_to_edmond_layout.py DIR OUT.npz` | not a check: converts the Larson & Schou 2018 table `rot2d.hmiv72d.ave` (+ `rmesh.orig`) to the layout `diff_rot_type = 'tabulated'` reads |
| `profiles/` | not a check: `rot2d.hmiv72d.ave` (Ω in nHz, 151 radii × 49 latitudes 90°…0°) and `rmesh.orig` (601 radii, every 4th is a table row), the Larson & Schou 2018 HMI average (input of `ls_to_edmond_layout.py`) |
| `lit_bench.py WORKDIR CASE ... [N= lmax= ncpus= nev= key=value]` | the incompressible and Boussinesq rows of `docs/differential_rotation_details.md` §11: Baruteau & Rieutord 2013 (cases `br*`), Guenel et al. 2016 (`gu*`) and Mirouh et al. 2016 (`mi*`) with structure 'uniform' (and stratification 'linear' for Mirouh); builds each run, runs submatrices, assemble and solve (MUMPS) at the published eigenvalue, prints the nearest eigenvalue in the paper's convention, the peak memory and the time. `list` prints the cases with the papers' resolutions |
| `solar_checks.py EDMOND_NPZ [WORKDIR]` | the solar rows of `docs/differential_rotation_details.md` §11: Mukhopadhyay et al. 2025a and 2025b (rin 0.5), each with and without the thermal wind; scipy, one rank |
| `dr_cost.py WORKDIR N LMAX NCPUS [LIMIT_GB]` | not a check: time and peak memory of submatrices, assemble and solve against the highest degree n of a 'user' ΔΩ (no differential rotation, momentum only, thermal_wind 1 and 2), with nnz(A) and the operator count; kills a step above LIMIT_GB (30) and skips the larger n. Its results are to go in `docs/differential_rotation_details.md` §12, which is not written yet |
| `bin/ru_insane.py` | on any 'solar_P4' or 'tabulated' run: the momentum and heat diff-rot blocks against its (r, θ) grid, with ΔΩ summed over every Pₙ and S_tw built independently (∂θS pointwise, S by cumulative Gauss integration in cos θ, ∂ᵣS by Chebyshev differentiation); honours the term amplitudes and `thermal_wind_radial` |
| `tests/heat_dr/sd_check.py` | after `csolve.py`: spin_doctor's diff-rot powers (`utils4pp.diff_rot_setup`, generic in n) against the matrix pairings of `ebal.py` |

Results on the final code (2026-10-13):

| check | configuration | worst |
| --- | --- | --- |
| `gen_check.py 8 16` | lmax 48 | 4.4e-11 (closed forms), 5.9e-12 (Δl ranges, n 4–16) |
| `ru_insane.py` | tabulated nmax 8, thermal_wind 2; tabulated nmax 6, thermal_wind 1, `thermal_wind_radial` 0, `diff_rot_term_amp` ≠ 1, `thermal_wind_amplitude` 0.7; solar_P4 amplitude 1.3, thermal_wind 2 (N 96) | grid ~1e-14, all PASS; a 1e-4 change of one generic coefficient or of σₙ is caught (1.5e-4, 1.2e-3) |
| `sd_check.py` | tabulated nmax 8, thermal_wind 2; solar_P4 amplitude 1.3, thermal_wind 1, radial 0, term amplitudes (N 48, m 1, 6 modes each) | powers vs pairings ≤ 1.1e-8; resid1 ≤ 2e-7, resid2 ≤ 1e-7, resens ≤ 8e-6, resid3 ≤ 8e-5 with the terms, 1e-2 to 1 without |
| `tw_edmond.py` | tabulated nmax 12, thermal_wind 2, the Edmond file | rms 3 %; pole-to-equator contrast within 0.1–2 % at 0.72–0.95 R☉ |
| `run_bekki.py` | the five modes of §8 (m 1 high-latitude, m 3 Rossby n0 and n1, m 3 columnar, m 10 HFR); also `mk_bekki.py tw=3 ds=-1000` (Bekki et al. 2024 onset scan, m 1) | all five equal to the §8 values within 2e-3 nHz; the onset run gives growth +5.015 nHz (§8: +5.02) |
| `regress.sh` | against commit 18641206 (13 configurations), plus the Bekki benchmark runs (smoothed sun with Model S density, m 1, thermal_wind 2 and 3, N 64) | A, B bitwise equal |
| `ncpus` 1…4 (as `tests/heat_dr/mpi_check.sh`) | tabulated nmax 8, thermal_wind 2, radial 0, term amplitudes, m 1; solar_P4 m 2 symm −1 (smoothed sun, N 40, lmax m + 23) | A, B bitwise equal on 1–4 ranks |

Results after the angular algebra of `bin/angular.py` replaced the quadrature builder and scipy's harmonics (2026-10-23,
Mac, scipy 1.17):

| check | configuration | worst |
| --- | --- | --- |
| `angular_check.py` | full (harmonics to l = 3000, also m 400 and 1425; vector operations around l = 1200, primitives n ≤ 16) | harmonics vs exact P_l 1.1e-13, vs scipy 4.6e-12, norms at large m 2.5e-13 (without the power-of-two rescaling of `ylm_grid`: 5e-2 at m 400, 0.27 at m 1425); vector operations 9e-12 (l ≈ 1200, tolerance 2e-9); primitives vs `quad_reference.py` 2.3e-12 |
| `gen_check.py 8 16`, `gen_check.py 4 16 1025` | lmax 20 and 1025 | 1.3e-15 and 1.2e-15 of the row scale (closed forms); 0 outside the Δl ranges (n 4–16) |
| `ru_insane.py` | solar_P4 amplitude 1.3 thermal_wind 2; solar_P4 ThermaD 0; conical with `diff_rot_generic = 1` (m 1, thermal_wind 2; m 0) (N 96) | diff-rot rows vs grid ≤ 9.2e-12, all PASS |
| `regress.sh` | against commit b4598010 (13 configurations) | 2 bitwise, 11 round-off ≤ 5.6e-16 of the row scale (thermal rows) |
| `tests/heat_dr/run_all.sh` | all steps except `tw_sympy.py` (no sympy in that environment) | heat_check 43, asm_check 41, row_check 9 ok; rigid-offset identity ≤ 1.2e-9; sd_check ok |
| `mpi_check.sh` | as in the script, plus solar_P4 and `diff_rot_generic = 1` | bitwise on 1–4 ranks (its 5-rank stop test needs GNU `timeout`, absent on macOS; run by hand it stops with the message) |
| full pipeline | conical, thermal_wind 2, m 2, N 32, lmax 129 and 705, against b4598010 | eigenvalues 1.6e-13; spin_doctor columns equal to 9 digits; at lmax 705 the former diagnostics are NaN, the new ones finite and equal to lmax 129 |
| setup time and memory of the generic couplings | n 0, 2, lmax 1025 | 0.014 s, 6.8 MB (quadrature: 177 s, 2.5 GB per rank, and wrong from l = 646) |

Results after the differential rotation was unified in `bin/diff_rot.py` (2026-10-23; new parameter names, 'user'
profiles, every degree from the algebra, σₙ by the Legendre algebra):

| check | configuration | worst |
| --- | --- | --- |
| references | 15 configurations rebuilt with the new names and compared with the code before (closed forms and algebra for P₀, P₂) | A ≤ 5.5e-15 of the row scale, B bitwise; eigenvalues ≤ 2.6e-9 (csolve's own run-to-run spread is ~3e-13 and more on sensitive modes) |
| `regress.sh` | against the code before the unification (old parameter names), 14 configurations | 1 bitwise, 13 round-off ≤ 5.5e-15 |
| `gen_check.py` | conical, cylindrical as 'user', solar_P4; lmax 20 and 1025 | ≤ 1.3e-15 (couplings); σ₂, σ₄, σ₆ vs closed forms ≤ 2.1e-16 of r²G²\|∂ΔΩ\|/(Beyonce g) |
| 'user' profile | cylindrical rotation as strings vs `'cylindrical'` | A, B bitwise equal |
| `ru_insane.py` | 'user' (P₀, P₂, P₄), solar_P4 with thermal_wind 3, rigid, solar_P4 ThermaD 0, shellular σ = 2 at ricb 0 (N 192) | all PASS; diff-rot rows vs grid ≤ 1.8e-9 |
| `tests/heat_dr/run_all.sh` | complete, with sympy | tw_sympy 0 failures; heat_check 43, asm_check 41, row_check 9 ok; rigid identity ≤ 1.3e-9; sd_check ok |
| `mpi_check.sh` | as in the script | bitwise on 1–4 ranks (5-rank stop test: GNU `timeout` absent on macOS) |
| `run_bekki.py` | the five modes | eigenvalues identical to the code before to 1e-4 nHz; all within 2e-3 nHz of the §8 values |
| full pipeline | 'user' profile (P₀, P₂, P₄), N 96 | spin_doctor residuals ≤ 2e-6; `params.dat` rows end with `# user: zf0 = … ; p = 0.7`, `np.loadtxt` reads them |

Results after the background came to be chosen by `structure`, `stratification` and `diffusivity` (`bin/background.py`,
2026-10-26; Linux, scipy 1.15.3), each against 1aca07b with the former names:

| check | configuration | worst |
| --- | --- | --- |
| `regress.sh` | the 15 configurations | A, B bitwise equal |
| `mk_bekki.py`, `mk_rz.py` | 2 and 5 builds (`rz` 1 and 0, the full sphere, '8-term fit', a 'tabulated' rotation) | A, B bitwise equal |
| `tests/heat_dr/run_all.sh`, `tests/rsf/run_all.sh` | complete, on both trees | all 206 matrices bitwise equal; the outputs differ only in two reworded echo lines, a timing and two round-off terms (5e-18, 4e-19) that change between repeats of the same code |
| `tests/rsf/regress_solve.sh` | eigen and forced solves | bitwise equal |
| the smooth Sun example | N 64, lmax 23, 9 variants (with and without `diff_rot_hold`, a 0.6 R☉ shell, δ_sf, `visc_pow`, `therm_pow`, the legacy structures) | A, B bitwise equal; solved without the hold: eigenvalues and fields bitwise equal |
