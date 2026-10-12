# ThermaD = 0 with the thermal wind (thermal_wind = 1, 2)

Results of the change, cloud container, 2026-10-12. The analysis scripts named below (`track.py`, `coupling.py`,
`dense.py`) are in the project folder, `kore/diffrot/checks/thermad0/`; the checks themselves are in `tests/heat_dr/`.

## The issue, precisely

With `ThermaD = 0` kore writes the heat equation divided by p₀ (p₀ ≈ 1e-7 near the surface made the p₀-weighted rows
ill-conditioned) and multiplied by r, in a C⁽⁰⁾ Gegenbauer basis, with no thermal bc rows: `decode_label` gives section
h `rpower = 1 - rx`, and `submatrices.py` sets `gebasis[2] = 0`. Term (c) of `docs/heat-equation.md` §2, the advection of
the thermal-wind entropy by u_θ, is p₀ σₙ [P + r P′ + r (ln ρ)′ P] Pₙ′(X) D / r² before the row weight. Two things break:

1. **Basis.** Its P′ part (`h1tw20_D1`, `h1tw40_D1`, `h1tw60_D1`) is a first derivative, which lands in C⁽¹⁾. In a C⁽⁰⁾
   section `submatrices.py` builds `G[gbx][gbx-dx] = G[0][-1]`, and Python's negative index returns `G0[-1] = 1` (the
   identity) instead of failing: the D1 output, a C⁽¹⁾ series, would be read as Chebyshev coefficients. Silently wrong.
2. **Row weight.** With the r weight, the P part (`h2tw20_D0`) decodes to `rpower = 1 - 2 = -1`, a profile σₙ/r. In the
   full sphere σ₂(0) ≠ 0 for the conical, Y20 and solar profiles (σ₂ ∝ r f/g and g ∝ r), so σₙ/r is singular at r = 0
   and has the wrong parity for `remroco`. At ricb > 0 it would be regular, but item 1 remains.

The old code stopped in `submatrices.py` before reaching either. Dropping the /p₀ would fix item 2 but bring back the
conditioning problem; the fix keeps /p₀ and changes r → r² and C⁽⁰⁾ → C⁽¹⁾ for this case only.

## What changed (file by file)

The new mode is decided in one place, `utils.adiab_tw = thermal == 1 and ThermaD == 0 and diff_rot == 1 and
thermal_wind >= 1`. In it the section-h rows are r² (eq)/p₀ in C⁽¹⁾, still without bc rows, and they reuse the labels of
both paths.

| file | change |
| --- | --- |
| `bin/utils.py` | `adiab_tw` (the only place that decides the mode); `decode_label`: section h `rpower = 2 - rx` when `adiab_tw`. So `h0_D0` is r² s, `h1dSd0_D0` is r² uᵣ S′ (as r L P S′), `h0abu0_D0`/`h0aub0_D0` is the s advection with the r² weight, and the thermal-wind labels of the ThermaD > 0 path (`h1tp20_D0`, `h2tw20_D0`, `h1tw20_D1`, `h1tw20lho1_D0`, `h1tw20_D0`, and the 4, 6 ones) decode as they do there |
| `bin/submatrices.py` | `gebasis[2] = 1 if ut.adiab_tw else 0`; the `sys.exit` for ThermaD = 0 with the thermal wind is removed; `G[gbx][gbx-dx]` raises a `ValueError` (label, dx, basis) when dx > gbx; `chop = 0` for section h when ThermaD = 0 (at ricb > 0, `chop = gbx = 1` would otherwise put one zero row on top of every section-h block); comments |
| `bin/radial_profiles.py` | `ptw(r)`: p₀, or 1 when `adiab_tw`; `tw2/4/6`, `tp2/4/6` use it, so in the new mode they are σₙ and σₙ′ (bitwise the same as before when ThermaD > 0) |
| `bin/operators.py` | comments only: `thermal_diffrot` already used the same (b), (c), (c′) labels for both paths, and (a) uses `h0abu0_D0`/`h0aub0_D0` whenever ThermaD = 0 |
| `bin/params_default.py` | `thermal_wind` comment: ThermaD > 0 is no longer required |
| `bin/assemble.py` | unchanged: `bc_theta_spherical` returns nothing for ThermaD = 0, and section h is packed from the operators |
| `bin/ru_insane.py`, `tests/heat_dr/ebal.py` | `gb['h'] = 1` in the new mode, `chop['h'] = 0` for every ThermaD = 0 run, entropy pairing weight `wgt['h'] = p₀` (rows r² (eq)/p₀) instead of p₀ r. `heat_check.py`, `asm_check.py`, `terms.py` and `sd_check.py` take these from `ebal.py` |
| `tests/heat_dr/asm_check.py` | docstring: in the new mode the reference run is the same run with `diff_rot_amplitude = 0`, because diff_rot = 0 or thermal_wind = 0 gives the old r (eq)/p₀ rows in C⁽⁰⁾. A wrong reference trips the existing assertion, which now hints at that |
| `tests/heat_dr/run_all.sh` | new-mode runs `k1 km0 km2b ky20wb kshell kg3 ksun ksun_a` with thermal_wind 1 and 2, and their `z_*` references (amplitude 0), in the heat_check and asm_check lists; rigid configurations `K1` (G = 1) and `K13` (G = 3), whose baseline is rigid with amplitude 0 (the same heat rows); `terms.py` and `sd_check.py` on `k1_tw2` |
| `tests/heat_dr/README.md`, `docs/heat-equation.md` | the asm_check reference; §0 row weights, §3 the implementation, §5 the submatrices row, §7 the open item updated |

Deviations from the proposed design:
- `chop = 0` for section h (one line in `submatrices.py`). It was needed: at ricb > 0, C⁽¹⁾ would otherwise reserve one
  zero row per l for a bc that does not exist. Mutation M4 below shows what that does.
- asm_check and the rigid test need a reference with the same heat rows, so they use the `diff_rot_amplitude = 0` run
  instead of the `diff_rot = 0` one. No code beyond comments and the run_all lines.

Bitwise constraint: ThermaD > 0 runs, and ThermaD = 0 runs with thermal_wind = 0 or diff_rot = 0, never enter the new
branches (`adiab_tw` is False; `ptw` returns `prf.pressure(r, 0)` as before; `chop` was already 0 for gbx = 0).
Confirmed in test f.

## Results

Runs are the test defaults of `mk_run.py` (ricb 0.35, m 1, conical −0.25, Beyonce 10, ViscosD 1e-3, stress-free,
user-def profiles) plus `par.ThermaD = 0`, unless stated. "tw2" means thermal_wind = 2. `SUN` is the run_all.sh string
(ricb 0, smoothed sun, N 120, lmax 30) with ThermaD = 0.

### a. heat_check.py and asm_check.py (pass < 1e-8)

asm_check's reference is `z_*` (same run, amplitude 0).

| run | heat_check tw1 | heat_check tw2 | asm_check tw1 | asm_check tw2 |
| --- | --- | --- | --- | --- |
| k1: conical m 1, N 96 | 3.0e-15 | 2.6e-15 | 4.5e-15 | 3.0e-15 |
| km0: m 0, lmax 21 | 1.1e-14 | 9.6e-15 | 9.1e-15 | 8.8e-15 |
| km2b: m 2, symm −1, lmax 21 | 3.3e-15 | 3.1e-15 | 3.4e-15 | 3.7e-15 |
| ky20wb: Y20-wall-bounded | 7.9e-15 | 7.9e-15 | 1.7e-14 | 1.9e-14 |
| kshell: shellular | 2.8e-15 | 2.3e-15 | 2.9e-15 | 1.9e-15 |
| kg3: Gaspard 3, Beyonce 90 | 2.6e-15 | 2.2e-15 | 4.0e-15 | 2.6e-15 |
| ksun: SUN, ricb 0, symm 1 | 5.5e-15 | 6.3e-15 | 9.1e-13 | 5.7e-13 |
| ksun_a: SUN, symm −1 | 5.9e-15 | 7.4e-15 | 2.3e-12 | 8.7e-13 |

All 32 are ok, and run_all.sh reproduces them. These numbers are p₀-weighted (the rows r² (eq)/p₀ are
paired with p₀ s*, p₀ ≈ 2.5e-6 at r = 1 for the test profiles), so they do not see the rows near the wall. There the new
rows are less accurate than the ThermaD > 0 ones: term (c) carries σₙ (ln ρ)′ without p₀, and (ln ρ)′ = −4 aux0² r/(1 −
aux0² r²) is nearly singular at r = 1/aux0. Unweighted (`tests/heat_dr/row_check.py`, P = 0 at the walls), the rows are
right to 4e-11 at ricb 0.35 (N 96) and 6e-11 for the smoothed sun (N 120), but only to 5e-8 – 3e-7 for the 'user def'
profiles at ricb 0 (N 96; 3e-11 at N 160) and 3e-3 with aux0 = 0.999 (N 96), where heat_check still reports 5e-15. In the eigenvalues it is not the limiting error in the
runs tried (full sphere, user def, |λ(N 96) − λ(N 240)| 7e-5 to 5.5e-4 with thermal_wind 1 against 5e-5 to 3.6e-4
(N 96 vs 128) with thermal_wind 0; at aux0 = 0.999 both are 5e-2 to 1e-1 off at N 96), but the p₀-weighted checks are
blind to it.
A has no zero row and B no zero section-h row (k1: 0 of 2880, and 60 zero B rows, all momentum bc rows).

The adapted tools on the unchanged modes: heat_check s96 3.3e-13, k0 1.4e-15, sun 3.4e-11, g3_tw2 2.9e-13; asm_check
k0 1.4e-15; ru_insane PASS on all four.

### b. ru_insane.py (N ≥ 96, tol 1e-6): all 16 runs PASS

| run | heat advection vs grid | thermal wind vs grid | advection anti-Herm. | worst (all tests) |
| --- | --- | --- | --- | --- |
| k1, k1_tw2 | 2.6e-15 | 1.2e-14, 1.1e-14 | 2.0e-16 | 4.7e-08 |
| km0, km0_tw2 | 0 | 1.0e-14, 9.7e-15 | n/a (m 0) | 4.8e-09 |
| km2b, km2b_tw2 | 6.8e-15 | 1.1e-14, 1.1e-14 | 2.0e-16 | 3.7e-08 |
| ky20wb, ky20wb_tw2 | 1.5e-14 | 1.2e-14, 1.2e-14 | 1.2e-16 | 4.7e-08 |
| kshell, kshell_tw2 | 1.4e-14 | 6.0e-15, 4.9e-15 | 9.0e-17 | 4.7e-08 |
| kg3, kg3_tw2 | 2.7e-15 | 1.2e-14, 1.1e-14 | 2.2e-16 | 4.7e-08 |
| ksun, ksun_tw2 | 3.1e-15 | 1.7e-14, 1.1e-14 | 6.2e-16 | 6.9e-08 |
| ksun_a, ksun_a_tw2 | 8.0e-15 | 2.3e-14, 2.6e-14 | 8.0e-16 | 5.8e-08 |

The worst column is the viscous Hermitian test, which is the same with ThermaD > 0. The entropy rows match direct
quadratures: entropy block Hermitian ≤ 1.4e-15, and buoyancy, advection and entropy rows vs quadrature ≤ 4e-15 at
ricb 0.35 (buoyancy 2e-8 and 6e-8 for the sun, as with ThermaD > 0).

Mutations (each applied to a copy of the code):

| mutation | heat_check | ru_insane |
| --- | --- | --- |
| M1: keep p₀ in tw/tp | 2.8e-2 FAIL | thermal wind 0.23, FAIL |
| M3: gebasis[2] = 0 (old basis) | `submatrices.py` raises `ValueError: h1tw20_D1: derivative order 1 needs at least the C^(1) basis, section h has C^(0)` | n/a |
| M4: no `chop = 0` (zero row per l at ricb > 0) | 0.18 FAIL | entropy Hermitian 1.4, advection 2.4, FAIL |

### c. Rigid-offset identity (rigid_thermal.py, m 2, N 64, lmax 25, 16 modes, ThermaD 0, thermal_wind 1)

| baseline | G = 1 | G = 3 |
| --- | --- | --- |
| rigid, amplitude 0, Gaspard G(1 + 0.1) (same heat rows; run_all `K1`, `K13`) | 1.4e-9 (1.3e-9 in the run_all pieces) | 8.4e-12 (8.2e-12) |
| diff_rot = 0 (old r (eq)/p₀ rows in C⁽⁰⁾): a different discretization of the same problem | 8.0e-7 (N 64), 6.6e-9 (N 96), 7.8e-9 (N 128) | 2.5e-7 (N 64) |

For a rigid offset S_tw = 0, so this tests the new rows and basis with term (a), not the thermal wind. The 1.4e-9 at
G = 1 is set by the most damped modes (σ ≈ −0.97), where csolve's residuals are ~7e-9. The second row is the old and
new discretizations converging to each other with N.

### d. ThermaD → 0 continuity

The modes are lightly damped and thermally coupled. They come from k1 at N 128 (tw1), with their share of buoyancy in the
momentum balance and of the diff-rot heat terms in the entropy balance (`coupling.py`):

| mode | λ (ThermaD = 0, N 128) | buoyancy \|B\|/(\|λ\| K) | diff-rot heat \|H\|/(\|λ\| E) | of which u·∇S_tw |
| --- | --- | --- | --- | --- |
| T1 | −0.01684560+0.49088938i | 0.045 | 0.49 | 0.10 |
| T2 | −0.07611871+0.44729603i | 0.11 | 0.37 | 0.066 |
| T3 | −0.12446846+0.39326248i | 0.22 | 0.37 | 0.051 |
| T4 | −0.11370717+1.76151161i | 0.29 | 0.12 | 0.0062 |
| T5 | −0.04118984+2.54022727i | 0.55 | 0.075 | 0.0041 |

λ(ThermaD = 0) changes by 6e-9 to 4e-8 between N 96 and 128 (tw1 and tw2), so it is converged in N.

Each entry below is |λ(ThermaD) − λ(ThermaD = 0)|, from shift-invert at each target (`track.py`; residuals ≤ 3e-11).

Fixed entropy at the walls (bco_thermal = bci_thermal = 0, as mk_run sets them), stress-free, thermal_wind 1:

| mode | N | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | 96 | 4.9e-04 | 5.4e-05 | 5.4e-06 | 2.6e-07 | 8.8e-06 | 9.9e-06 |
| T1 | 128 | 4.9e-04 | 5.4e-05 | 5.4e-06 | 5.5e-07 | 2.4e-06 | 6.0e-06 |
| T1 | 192 | | | | | 5.7e-08 | |
| T1 | 256 | | | | | 5.4e-08 | 1.7e-06 |
| T2 | 96 | 1.3e-03 | 1.5e-04 | 1.5e-05 | 7.0e-06 | 3.2e-05 | 4.6e-05 |
| T2 | 128 | 1.3e-03 | 1.5e-04 | 1.5e-05 | 1.4e-06 | 1.1e-05 | 2.8e-05 |
| T3 | 96 | 3.4e-03 | 3.7e-04 | 3.6e-05 | 1.1e-05 | 1.1e-04 | 9.9e-05 |
| T3 | 128 | 3.4e-03 | 3.7e-04 | 3.6e-05 | 3.9e-06 | 2.1e-05 | 6.1e-05 |
| T4 | 96 | 4.2e-03 | 4.2e-04 | 4.2e-05 | 1.3e-05 | 1.2e-05 | 1.2e-05 |
| T4 | 128 | 4.2e-03 | 4.2e-04 | 4.2e-05 | 3.3e-06 | 7.5e-06 | 7.3e-06 |
| T5 | 96 | 1.9e-03 | 1.9e-04 | 1.9e-05 | 1.0e-05 | 1.0e-05 | 1.0e-05 |
| T5 | 128 | 1.9e-03 | 1.9e-04 | 1.9e-05 | 1.9e-06 | 6.2e-06 | 6.1e-06 |
| T5 | 192 | | | | | 2.2e-06 | |
| T5 | 256 | | | | | 1.6e-07 | 1.7e-06 |

The same, thermal_wind 2 (λ(0): T1 −0.01701638+0.49013119i, T2 −0.07696123+0.44671144i, T3 −0.12435151+0.39357407i,
T4 −0.11360552+1.76197107i, T5 −0.04089672+2.54083851i):

| mode | N | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | 96 / 128 | 4.9e-04 / 4.9e-04 | 5.4e-05 / 5.4e-05 | 5.5e-06 / 5.5e-06 | 2.4e-07 / 5.5e-07 | 6.6e-06 / 1.8e-06 | 7.6e-06 / 4.6e-06 |
| T2 | 96 / 128 | 1.3e-03 / 1.3e-03 | 1.5e-04 / 1.5e-04 | 1.5e-05 / 1.5e-05 | 6.4e-06 / 1.4e-06 | 2.1e-05 / 9.0e-06 | 3.9e-05 / 2.3e-05 |
| T3 | 96 / 128 | 3.5e-03 / 3.5e-03 | 3.7e-04 / 3.7e-04 | 3.7e-05 / 3.7e-05 | 1.0e-05 / 3.9e-06 | 9.4e-05 / 2.0e-05 | 8.8e-05 / 5.4e-05 |
| T4 | 96 / 128 | 4.1e-03 / 4.1e-03 | 4.2e-04 / 4.2e-04 | 4.2e-05 / 4.2e-05 | 1.2e-05 / 3.3e-06 | 1.0e-05 / 6.3e-06 | 9.8e-06 / 6.0e-06 |
| T5 | 96 / 128 | 1.9e-03 / 1.9e-03 | 1.9e-04 / 1.9e-04 | 1.9e-05 / 1.9e-05 | 8.7e-06 / 1.7e-06 | 8.4e-06 / 5.1e-06 | 8.3e-06 / 5.0e-06 |

Zero entropy flux at the walls (bci_thermal = bco_thermal = 1), stress-free, thermal_wind 1 (at ThermaD = 0 the thermal
bc parameters do not enter: A is byte-identical to the fixed-entropy run):

| mode | N | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | 96 / 128 | 5.6e-04 / 5.6e-04 | 6.0e-05 / 6.0e-05 | 6.0e-06 / 6.0e-06 | 6.0e-07 / 6.0e-07 | 6.3e-08 / 6.1e-08 | 4.2e-09 / 1.7e-09 |
| T2 | 96 / 128 | 1.6e-03 / 1.6e-03 | 1.9e-04 / 1.9e-04 | 1.8e-05 / 1.8e-05 | 1.8e-06 / 1.8e-06 | 2.1e-07 / 1.9e-07 | 2.5e-08 / 9.1e-09 |
| T3 | 96 / 128 | 3.3e-03 / 3.3e-03 | 3.7e-04 / 3.7e-04 | 3.6e-05 / 3.6e-05 | 3.6e-06 / 3.6e-06 | 3.6e-07 / 3.8e-07 | 1.8e-08 / 5.1e-09 |
| T4 | 96 / 128 | 4.1e-03 / 4.1e-03 | 4.2e-04 / 4.2e-04 | 4.2e-05 / 4.2e-05 | 4.2e-06 / 4.2e-06 | 4.2e-07 / 4.2e-07 | 5.9e-09 / 4.8e-09 |
| T5 | 96 / 128 | 1.9e-03 / 1.9e-03 | 1.9e-04 / 1.9e-04 | 1.9e-05 / 1.9e-05 | 1.9e-06 / 1.9e-06 | 1.9e-07 / 1.9e-07 | 2.7e-09 / 1.9e-09 |

Controls at N 96, fixed entropy. No-slip walls (bci = bco = 1, thermal_wind 1): T1 3.7e-6, 3.7e-8, 4.1e-10;
T4 4.2e-5, 4.2e-7, 4.2e-9; T5 2.1e-5, 2.1e-7, 2.1e-9 at ThermaD 1e-6, 1e-8, 1e-10. thermal_wind 0 (old rows, unchanged
code): T1 5.5e-6, 5.5e-8, 5.8e-10; T4 4.1e-5, 4.1e-7, 4.1e-9; T5 1.8e-5, 1.8e-7, 1.8e-9.

Reading. When N resolves the ThermaD > 0 run, the eigenvalues converge to the ThermaD = 0 ones linearly in ThermaD with
the same slope for both bcs: λ(K) − λ(0) ≈ c K, c = 5 to 40. That holds at every ThermaD with zero-flux or no-slip
walls, and down to 1e-7 (N 128) and 1e-8 (N 256) with fixed entropy and stress-free walls. With fixed entropy and
stress-free walls the ThermaD = 0 entropy is not zero at the wall: u_θ ≠ 0 there and u_θ ∂θS_tw/r ≠ 0. A ThermaD > 0
run then has a layer of width ~ (ThermaD/|λ|)^½, about 1e-4 at 1e-8 and 1e-5 at 1e-10. Below the resolved range its
eigenvalue stalls 1e-6 to 1e-4 away, and the gap falls with N (T1 at 1e-10: 9.9e-6, 6.0e-6, 1.7e-6 at N 96, 128, 256;
T5 at 1e-8: 1.0e-5, 6.2e-6, 2.2e-6, 1.6e-7 at N 96, 128, 192, 256). The ThermaD = 0 runs have no layer and are
converged at N 96. thermal_wind = 0 has no such layer: uᵣ = 0 and S′ = 0 at the walls make s = 0 there anyway.

Box counts (complete spectra at N 96 by dense QZ after Ruiz equilibration, accurate to ~5e-7 against shift-invert; the
unequilibrated QZ was off by up to 1e-1 and gave fake eigenvalues with σ up to 49). The numbers are eigenvalues within
±0.05, ±0.1 and ±0.2 in σ and ω of each mode, written ThermaD 0 / ThermaD 1e-10:

| mode | tw1 | tw2 |
| --- | --- | --- |
| T1 | 1/1, 2/2, 4/4 | 1/1, 2/2, 4/4 |
| T2 | 1/1, 3/3, 74/72 | 1/1, 3/3, 76/75 |
| T3 | 1/1, 3/3, 266/259 | 1/1, 3/3, 264/258 |
| T4 | 1/1, 1/1, 1/1 | 1/1, 1/1, 1/1 |
| T5 | 1/1, 2/2, 2/2 | 1/1, 2/2, 2/2 |

Whole spectrum, N 96 tw1 (tw2):

| | ThermaD 0 | ThermaD 1e-10 |
| --- | --- | --- |
| finite eigenvalues | 2820 (2820) | 2800 (2800) |
| in the band \|σ\| < 0.03, ω ∈ (−0.01, 0.26) | 790 (791) | 770 (771) |
| outside the band with σ > −0.3 | 130 (129) | 130 (129) |
| growing (σ > 1e-3, all inside the band) | 22 (19) | 22 (19) |

The 20 extra ThermaD = 0 eigenvalues are the 20 s coefficients that the ThermaD > 0 run spends on bc rows, and they all
fall in the band. Outside the band the 130 (129) modes pair one to one: median distance 4.7e-6 (3.8e-6), maximum 1.2e-4
(1.0e-4), the latter for modes close to the band. Only the ±0.2 boxes of T2 and T3 reach the band, which is where their
counts differ. No spurious mode outside the band.

### e. Balances

- k1 (tw1, N 96, 8 modes near −0.05 + 1.2i): sd_check `worst dWdr, dWdr_thm 1.9e-10 -> ok`; resid3 with the new terms
  3e-13 to 7e-15 (without them 7e-1 to 5e-3), resid1 ≤ 5.8e-7, resid2 ≤ 3.4e-8. eig2kore + spin_doctor: resid𝑠
  ≤ 2.9e-13, residσ ≤ 5.8e-7. terms.py entropy balance λE − A − D − H: res ≤ 3.6e-14.
- k1_tw2 (the same, and the run_all line): sd_check 1.9e-10 ok; terms.py res ≤ 1.0e-13; resid3 ≤ 2.8e-12.
- ksun_tw2 (ricb 0, N 120, 12 modes near −0.05 + 0.8i): terms.py res ≤ 1.7e-12, but sd_check FAIL (5.2). Modes 0–4
  and 7 (ω ≈ 0.2494–0.2520) sit at the edge of the advection continuum (−i m G ΔΩ, ω ∈ [0, 0.25]) and are not resolved
  at any N. The discrete modes near 0.51i are under-resolved at N 120: resid3 3–6e-2, dWdr_thm 2.7e-4 to 1.4e-3. At the
  same N the ThermaD = 1e-10 run gives dWdr_thm 4e-4 to 2e-3 and thermal_wind 0 gives 2e-5 to 3.5e-4. At N 200 the new
  mode gives resid3 1.6e-8 to 2.7e-7 and dWdr_thm 1.9e-10 to 3.4e-8 for those modes (λ = −0.023069 + 0.515537i,
  ThermaD = 1e-10 at N 120: −0.023070 + 0.515536i). So run_all's sd_check line uses k1_tw2.

### f. Bitwise regression against the code before this change (each built with its own mk_run.py)

A.npz and B.npz are byte-identical (`cmp`) for s96, k0, nodr_k0, sunk0_tw0 (SUN, ThermaD 0, thermal_wind 0),
nodr_sunk0 (SUN, ThermaD 0, diff_rot 0), sun, sun_a_tw2, g3_tw2, m0_tw2, bc1 (thermal bc 1), f10 and f10k (forcing 10,
ThermaD 1e-3 and 0, thermal_wind 0; A and B_forced).

### g. MPI (mpiexec -n 1, 2, 3, 4)

A.npz and B.npz are byte-identical to the single-rank files, with the same pattern and 0 difference, for mpi_k1
(lmax 24, tw1), mpi_ksun (smoothed sun, ricb 0, N 64, lmax 24, tw2) and mpi_km0 (m 0, lmax 23, tw2).

### h. Independent reviews

Discretisation: no defect. B matches only the claimed combination (C⁽¹⁾, r², first row) to 1e-15, every other basis,
row or weight is off by ≥ 0.19; 19 other configurations are byte-identical to the old code (every .mtx), MPI with 2
ranks too. Its one finding, that the p₀-weighted checks cannot see the rows near the wall, gave `row_check.py` (a).
Numerics: with S_tw = 0 the new rows reproduce the old-row spectrum (conical N 96: ≤ 4e-8; Y20-wall-bounded: 69
eigenvalues, median 1e-9, max 5e-7); the smoothed sun with thermal_wind 2 converges in N (λ(N 160) − λ(N 200) ≈ 1e-8 to 2e-8 for
the discrete modes) and ThermaD = 1e-10 at N 200 is within 8e-9 of ThermaD = 0 there; the growing modes of the caveats.

## Caveats

- With fixed-entropy, stress-free walls, ThermaD → 0 at fixed N is a singular limit (d). The ThermaD = 0 run is the
  well-resolved one. A "tiny ThermaD instead of 0" run, as the params comment suggests for the old ThermaD = 0, is not
  equivalent below ThermaD ~ 1e-7 at N 128.
- Term (a) without diffusion has a continuous spectrum, −i m G ΔΩ(r, θ). ThermaD = 0 with thermal_wind = 0 has it too.
  The discretization shows it as hundreds of eigenvalues near σ = 0. With the thermal wind some of them grow: 22 (19 with
  thermal_wind 2) with σ ≤ 0.011 for k1 (N 96, lmax 20), 6 with σ ≤ 0.013 for Y20-wall-bounded, and 26 up to σ = 0.10
  for the smoothed sun (ricb 0, N 160, lmax 30). They are converged in N but not in lmax: their eigenfunctions peak at
  l ≈ lmax, and the sun has 43 and 57 of them, σ up to 0.12, at lmax 40 and 50. The ThermaD = 1e-10 runs have the same
  ones; they are absent with thermal_wind = 0, with S_tw = 0 in the new rows, and with ThermaD = 1e-3. So they are
  small-scale instabilities of the diffusionless thermal wind, unresolved in l, not an artefact of the new rows. ARPACK converges badly near it: csolve at 0 + 0.6i got 11/12 after 5000
  iterations at ThermaD 0 and at 1e-10, and some shift-inverts at T2 took tens of minutes. Use shifts away from that band.
- In the full sphere, ThermaD = 0 eigenfunctions need more N than ThermaD = 1e-3 ones (e: N 120 → 200 for the smoothed
  sun).
- `solve.py` turns Ruiz prescaling off automatically for ThermaD = 0. That was tuned on the r (eq)/p₀ rows and is not
  tested here for the r² rows (no PETSc/SLEPc in this container).
- A run with thermal = 1, diff_rot = 1, ThermaD = 0 and the default thermal_wind = 1 used to stop and now runs. The old
  behaviour (advection of s only) needs thermal_wind = 0, as before.
- The rigid test (c) does not exercise the thermal wind, since S_tw = 0 for a rigid offset. The thermal-wind blocks are
  covered by a, b and d.
