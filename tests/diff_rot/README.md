# Checks of the generic differential rotation (any Pₙ, 'solar_P4', 'tabulated')

Companion of `docs/heat-equation.md` §8 and of `tests/heat_dr/` (whose `csolve.py`, `sd_check.py` and `ebal.py`
handle any Δl range and are used here too). Every script runs from a kore run directory (`bin/` copy, `parameters.py`;
`ebal.py`/`sd_check.py` also need the `*.mtx` files of `submatrices.py`) and prints a verdict ending in `ok` or `FAIL`.

| script | what it checks |
| --- | --- |
| `gen_check.py [mmax [nmax]]` | the quadrature builder `bin/diff_rot_generic.py` against the closed forms of `bin/diff_rot_coefficients.py` (f₀, f₂, every coefficient, m ≤ mmax, l ≤ lmax, \|Δl\| ≤ 3), and for n = 4…nmax that no coupling falls outside the Δl ranges kore assembles (P→P, T→T: \|Δl\| ≤ n, even; P↔T: \|Δl\| ≤ n + 1, odd); tolerance 1e-10 |
| `tw_edmond.py diff_rot.npz [c_p]` | a 'tabulated' run reading the Mukhopadhyay et al. Edmond file: kore's ∂θS_tw against their `dsdt`, at three latitudes and six radii, the pole-to-equator contrast, and the rms mismatch over the shell |
| `regress.sh OLD_KORE [WORKDIR]` | builds 13 existing diff-rot configurations (conical, Y20, cylindrical, solar, shellular, Y20-wall-bounded; thermal_wind 0–3; ricb 0; ThermaD 0; smoothed sun; Gaspard 3; m 2; diff_rot 0) with an older checkout and with this one and requires A, B bitwise equal |
| `ls_to_edmond_layout.py DIR OUT.npz` | not a check: converts the Larson & Schou 2018 table `rot2d.hmiv72d.ave` (+ `rmesh.orig`) to the layout `diff_rot_type = 'tabulated'` reads |
| `bin/ru_insane.py` | on any 'solar_P4' or 'tabulated' run: the momentum and heat diff-rot blocks against its (r, θ) grid, with ΔΩ summed over every Pₙ and S_tw built independently (∂θS pointwise, S by cumulative Gauss integration in cos θ, ∂ᵣS by Chebyshev differentiation); honours the term amplitudes and `thermal_wind_radial` |
| `tests/heat_dr/sd_check.py` | after `csolve.py`: spin_doctor's diff-rot powers (`utils4pp.diff_rot_setup`, generic in n) against the matrix pairings of `ebal.py` |

Results on the final code (cloud container, 2026-10-13):

| check | configuration | worst |
| --- | --- | --- |
| `gen_check.py 8 16` | lmax 48 | 4.4e-11 (closed forms), 5.9e-12 (Δl ranges, n 4–16) |
| `ru_insane.py` | tabulated nmax 8, thermal_wind 2; tabulated nmax 6, thermal_wind 1, `thermal_wind_radial` 0, `diff_rot_term_amp` ≠ 1, `thermal_wind_amplitude` 0.7; solar_P4 amplitude 1.3, thermal_wind 2 (N 96) | grid ~1e-14, all PASS; a 1e-4 change of one generic coefficient or of σₙ is caught (1.5e-4, 1.2e-3) |
| `sd_check.py` | tabulated nmax 8, thermal_wind 2; solar_P4 amplitude 1.3, thermal_wind 1, radial 0, term amplitudes (N 48, m 1, 6 modes each) | powers vs pairings ≤ 1.1e-8; resid1 ≤ 2e-7, resid2 ≤ 1e-7, resens ≤ 8e-6, resid3 ≤ 8e-5 with the terms, 1e-2 to 1 without |
| `tw_edmond.py` | tabulated nmax 12, thermal_wind 2, the Edmond file | rms 3 %; pole-to-equator contrast within 0.1–2 % at 0.72–0.95 R☉ |
| `regress.sh` | against commit 18641206 (13 configurations), plus the Bekki benchmark runs (smoothed sun with Model S density, m 1, thermal_wind 2 and 3, N 64) | A, B bitwise equal |
| `ncpus` 1…4 (as `tests/heat_dr/mpi_check.sh`) | tabulated nmax 8, thermal_wind 2, radial 0, term amplitudes, m 1; solar_P4 m 2 symm −1 (smoothed sun, N 40, lmax m + 23) | A, B bitwise equal on 1–4 ranks |
