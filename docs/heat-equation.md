# Differential rotation in the heat equation: derivation, implementation, benchmarks

The user guide (parameters, profiles, examples) is [differential-rotation.md](differential-rotation.md). This page
holds the derivation (§0–3), the implementation and its tests (§5, §8) and the benchmarks (§4, §6, §8). It was derived
on 2026-10-11 and committed in three steps on `anelastic_log`: the heat-equation terms (`be2a4db5`); the `Gaspard`
factors, the exact and prescribed thermal wind, `ThermaD = 0` (with [thermad0.md](thermad0.md)) and the `spin_doctor`
diagnostics (`18641206`); any Pₙ, 'solar_P4', 'tabulated' and the amplitude controls (`92a0d020`, §8). The checks
are in `tests/heat_dr/` and `tests/diff_rot/`. §7 lists what is still open.

## 0. Result in one paragraph

kore's heat equation (section h, `operators.entropy/thermal_advection/thermal_diffusion`) is

    p₀ ∂s/∂t = −p₀ uᵣ S₀′ + ThermaD ∇·(κ p₀ ∇s),      p₀ = ρ₀T₀, rows × r² (÷ p₀ if ThermaD = 0; × r instead of r² if ThermaD = 0 without the thermal wind).

With a background rotation Ω = Ω₀ [1 + ΔΩ(r,θ)], ΔΩ = f₀(r) + f₂(r) P₂(cosθ), the perturbation is advected by
U₀ = ΔΩ r sinθ e_φ, and the background entropy acquires a latitudinal part S_tw(r,θ) fixed by thermal-wind balance
(§1). Linearising ∂s/∂t + U₀·∇s + u·∇(S₀ + S_tw) = diffusion adds three terms to the left-hand side:

    p₀ [ i m ΔΩ s  +  uᵣ ∂S_tw/∂r  +  (u_θ/r) ∂S_tw/∂θ ].                                                 (1)

The first follows from the linearisation alone. The other two follow from the background entropy, which in kore's
framework (spherical ρ₀, T₀, g, no mean centrifugal term, LBR buoyancy) is the thermal wind of §1; they must be
switchable, because the Boussinesq benchmark of Mirouh et al. 2016 imposes Ω(r) with a purely radial T₀ (§4).
Nothing else changes: the momentum equation keeps its present diff-rot term (the background pressure is a gradient
and drops out of the curled equations, the buoyancy stays Beyonce g s e_r), the diffusion operator acts on s only
(S_tw is assumed maintained, like S₀), U₀·∇S_tw = ΔΩ ∂_φ S_tw = 0, and there is no viscous heating in kore. §2 gives
the terms in kore's harmonic form, §3 what they need in the code.

## 1. Thermal-wind entropy S_tw

Steady background in the rotating frame, kore units (time 1/Ω₀, length R, buoyancy Beyonce g(r) s e_r so that
N² = Beyonce g dS/dr): U₀·∇U₀ + 2ẑ×U₀ = −∇p_b + Beyonce g S_tw e_r. The left side is −[(1+ΔΩ)² − 1] r sinθ e_s
(centrifugal + Coriolis of the differential rotation, e_s = cylindrical radial). Its curl, φ-component, gives

    (1/r) ∂S_tw/∂θ = ( r sinθ / (Beyonce g(r)) ) ∂[(1+ΔΩ)²]/∂z,        ∂/∂z = cosθ ∂/∂r − (sinθ/r) ∂/∂θ.      (2)

This is the thermal-wind equation r sinθ ∂Ω²/∂z = (g/(c_p r)) ∂S/∂θ in kore's variables; it is eq. (27) of Bekki,
Cameron & Gizon 2022 (A&A 662, A16). The l = 0 part of S_tw is absorbed in S₀(r). To first order in ΔΩ,
(1+ΔΩ)² → 1 + 2ΔΩ and (2) integrates in closed form:

    S_tw = σ₂(r) P₂(cosθ) + σ₄(r) P₄(cosθ),
    σ₂ = −( 2r² / (Beyonce g) ) [ f₀′/3 + f₂′/21 + (4/7) f₂/r ],                                              (3)
    σ₄ = −( 6r² / (35 Beyonce g) ) [ f₂′ − 2 f₂/r ].

This linear form is `thermal_wind = 1`. Keeping the ΔΩ² term (`thermal_wind = 2`) changes σ₂, σ₄ by products of the
profiles and adds a P₆ component; with the frame rate `Gaspard` = G (time unit 1/Ω_ref) the right side of (2) gets a
factor G². With e = 0 for `thermal_wind = 1` and e = 1 for `thermal_wind = 2` (closed forms, checked by `tests/heat_dr/tw_sympy.py`):

    σ₂ = −( 2r²G² / (Beyonce g) ) [ (1 + e f₀)(f₀′/3 + f₂′/21 + (4/7) f₂/r) + e f₂ (f₀′ + f₂′)/21 ],
    σ₄ = −( 6r²G² / (35 Beyonce g) ) [ (1 + e f₀)(f₂′ − 2f₂/r) + e (f₂f₀′ + (4/11) f₂f₂′ + (14/11) f₂²/r) ],          (3′)
    σ₆ = −e ( 4r²G² / (77 Beyonce g) ) f₂ (f₂′ − 2f₂/r).

For the conical profile the exact form changes σ₂ and σ₄ by the factors 1 + 2ε/3 and 1 + 12ε/11, i.e. 17 % and 27 % for
ε = −0.25; σ₆ needs P₆(X) = (231X⁶ − 315X⁴ + 105X² − 5)/16 and Δl = ±6 (P) and ±5 (T) blocks in §2–3. Only the exact
form is independent of the frame: a run in a frame rotating at Ω_ref(1 + c) with f₀ shifted by −c gives the same modes.
Checks: the cylindrical profile ΔΩ = ε s² (f₀ = (2/3)εr², f₂ = −(2/3)εr²) gives σ₂ = σ₄ = 0 in both versions, as it
must (barotropic); the conical profile ΔΩ = ε sin²θ gives σ₂ = 16εr/(21 Beyonce g), σ₄ = −8εr/(35 Beyonce g), so for
ε > 0 (equator faster, as in the Sun's convection zone) S_tw(pole) − S_tw(equator) = εr/(Beyonce g) > 0: warmer
poles, the known solar sense. For the 'solar' fit (tanh step at rtc) f₂′, f₀′ are localised at the tachocline; the
(4/7) f₂/r and −(12/35) f₂/r parts of (3) fill the convection zone, where kore's S₀′ is zero, so there S_tw is the
only background entropy gradient.

Size. Solar case (Beyonce = 4.9e4, f₂ = −0.198, r ≈ 0.9, g ≈ 1): σ₂ ≈ 4e-6, σ₄ ≈ −1e-6, pole-minus-equator
S_tw ≈ 5e-6 = S/c_p, i.e. ≈ 1.9e3 erg g⁻¹ K⁻¹ with c_p ≈ 3.5e8, the thermal-wind value at the top of the range Bekki
et al. scan. The small number is misleading: S_tw ∝ 1/(Beyonce g) while the buoyancy is Beyonce g s, so in the
convection zone (S₀′ = 0) the entropy perturbation driven by (u_θ/r)∂_θS_tw feeds back on the momentum equation at
O(ΔΩ) relative to Coriolis, independent of Beyonce, i.e. at the same order as the momentum diff-rot term. The
thermal-wind terms are leading-order for convection-zone modes (this is the baroclinic coupling that destabilises the
high-latitude modes in Bekki et al.), not a correction.

Prescribed S_tw (`thermal_wind = 3`). Bekki et al. 2024 (and Bekki et al. 2022, eq. 26) impose a latitudinal entropy
gradient independent of Ω, ∂s/∂θ = Δθs sin 2θ, uniform in r. Since sin 2θ = −(2/3) dP₂/dθ, that is S_tw = σ₂ P₂ with
σ₂ = −(2/3) `dSdtheta` constant and σ₄ = σ₆ = 0, where `dSdtheta` = Δθs/c_p (kore's s is δs/c_p; Δθs < 0 gives warm
poles). Then (b) vanishes and only (c), (c′) act; Ω still enters through (a) and the momentum equation (for uniform
rotation, `diff_rot_type = 'rigid'` with `diff_rot_amplitude = 0`).

Caveat: S_tw is the LBR-consistent thermal wind. The full anelastic balance would add (S₀′/(ρ₀ g)) p_b to S_tw, the
same p₁ S₀′ term kore drops for the perturbations, so it differs only where S₀′ ≠ 0 (radiative zone). Neglected,
consistently with kore's spherical background: the perturbation density in the background centrifugal acceleration,
the θ-dependence of ρ₀T₀ and g induced by the thermal wind, and ρ_tw in the mass flux, all of relative size
ΔΩ/(Beyonce g) in the anelastic case, negligible for the Sun (Beyonce ≈ 5e4).

## 2. The three terms in kore's harmonic form

s = Σ s_l(r) S_l^m, and u from P, T as in kore (u = ρ₀⁻¹∇×∇×(ρ₀ P r) + ∇×(T r), S_l^m the Schmidt harmonics of
`ut.Ylm`, L = l(l+1), h = ln ρ₀):

    uᵣ = L P S_l / r,      u_θ = (1/r) [ ∂ᵣ(rP) + r h′ P ] ∂_θ S_l  +  i m T S_l / sinθ.

All angular couplings come from two matrices on the degree index at fixed m (row = target l′, column = source l):

    x S_l = p_l S_{l+1} + q_l S_{l−1},           p_l = √((l+1)² − m²)/(2l+1),  q_l = √(l² − m²)/(2l+1)      → X
    (1 − x²) dS_l/dx = (l+1) q_l S_{l−1} − l p_l S_{l+1}                                                   → D

with x = cosθ; then P₂(X) = (3X² − 1)/2, P₄(X) = (35X⁴ − 30X² + 3)/8, P₂′(X) = 3X, P₄′(X) = (35X³ − 15X)/2.
(For instance P₂(X)_{l,l} = (3(p_l q_{l+1} + q_l p_{l−1}) − 1)/2, P₂(X)_{l+2,l} = (3/2) p_{l+1} p_l,
P₂(X)_{l−2,l} = (3/2) q_{l−1} q_l; in general X_{l′l} = √((2l′+1)/(2l+1)) √(4π/3) (−1)^m gaunt(l′,1,l,−m,0,m) and
the same with n = 2, 4 for Pₙ(X).) The row l′ of the heat equation (before the r² p₀ row weight) gains

    (a) advection        i m G [ f₀ δ_{l′l} + f₂ P₂(X)_{l′l} ] s_l                                 s → s, Δl = 0, ±2
    (b) radial           ( L_l P_l / r ) [ σ₂′ P₂(X)_{l′l} + σ₄′ P₄(X)_{l′l} ]                    P → s, Δl = 0, ±2, ±4
    (c) latitudinal, P   ( [P_l + r P_l′ + r h′ P_l] / r² ) [ σ₂ (3X D)_{l′l} + σ₄ ((35X³ − 15X) D / 2)_{l′l} ]   P → s, Δl = 0, ±2, ±4
    (c′) latitudinal, T  −( i m T_l / r ) [ σ₂ (3X)_{l′l} + σ₄ ((35X³ − 15X)/2)_{l′l} ]              T → s, Δl = ±1, ±3

where L_l is that of the source degree; with `thermal_wind = 2` the sums run to σ₆ (P₆′(X) = (1386X⁵ − 1260X³ + 210X)/16),
Δl up to ±6 for P and ±5 for T. The even Δl for P and odd for T match the equatorial-symmetry classes of
`ut.ell` (section h uses the poloidal l set), so the block structure is that of the momentum diff-rot blocks with
Δl = ±4 added. For m = 0 the terms (a) and (c′) vanish (and the couplings into l = 0, absent from kore's basis, are
harmless: nothing feeds back from s₀). Build P₂(X), P₄(X), Pₙ′(X)D from X, D with about five degrees more than lmax
and cut afterwards: formed at lmax they are wrong in the top degrees (errors of order one in the entries). Note (a)
is anti-Hermitian: in the entropy budget it only shifts the frequency (∫ p₀ ΔΩ |s|² dV), while (b) and (c) exchange
entropy "energy" with the flow.

Verification (sympy, 0 failures; `tests/heat_dr/tw_sympy.py` and `heat_check.py` repeat the main checks): (3) against direct integration of (2); the curl of the background
balance vanishes with S_tw in place (linear and exact); barotropic check; the matrices X, D, P₂(X), P₄(X), P′(X)D
against quadrature projections for m = 0…3; and (b), (c), (c′) against the full 3-D u·∇S_tw built from P, T, ln ρ₀
with sympy and projected on S_{l′}, for m = 0, 1, 2 and every Δl. Independent readers redid the thermal wind with
full spherical vector calculus, the linearisation, the couplings with Wigner-3j, and the velocity sign conventions
against kore's assembled Coriolis operator; none found a discrepancy.

## 3. What kore needs

Profiles: f₀, f₂ exist (`abu`, `aub`); new are σ₂, σ₄ from (3), which need g, g′, f′ and f″ (`gra`, `spv`/`svp`,
`psl`/`pls` give r f′, r² f″). Following `pdS` = p₀ dS/dr and `kps` = κ p₀, define the products `tw2` = p₀ σ₂,
`tw4` = p₀ σ₄ and `tp2` = p₀ σ₂′, `tp4` = p₀ σ₄′ (a label `h1tw21_D0` would mean r (p₀σ₂)′ P, which is not (b)).
Parity for `ricb = 0`: σ₂, σ₄ are even in r (f even, f′ and g odd), so `tw2`, `tw4` are even like all other profiles,
but `tp2`, `tp4` are odd and must be added to the `adj` lists of `submatrices.py` (lines 286, 340; also
`tests/heat_dr/ebal.py` and `ru_insane.row_parity`) next to `gra, pdS, dSd`, otherwise term (b) is silently dropped
in full-sphere runs. (The 'solar' profile's r-parity defect is fixed by evaluating the profiles at |r|, §5.)
Rows × r² p₀ and ÷ L_{l′}³ like the other section-h operators:

    (a)  i m r² p₀ f₀ s,  i m r² p₀ f₂ P₂(X) s          labels  h0pss0abu0_D0,  h0pss0aub0_D0        (s column, Δl 0, ±2)
    (b)  L_l r p₀ σₙ′ Pₙ(X) P                           labels  h1tp20_D0,  h1tp40_D0                 (P column, Δl 0, ±2, ±4)
    (c)  p₀ σₙ [P + r P′ + r h′ P] Pₙ′(X) D              labels  h2tw20_D0, h1tw20_D1, h1tw20lho1_D0 (and tw4)   (P column)
    (c′) −i m r p₀ σₙ Pₙ′(X) T                           labels  h1tw20_D0,  h1tw40_D0                 (T column, Δl ±1, ±3)

All twelve labels parse with `utils.decode_label` (13-character two-function form for the first and last of (c)).
Sign: the section-h rows of A hold the right-hand side of the heat equation (`thermal_advection` enters as
−r p₀ S₀′ u_r, with B = +`entropy`), so the three left-hand-side terms of (1) go into A with a minus sign, opposite
to the momentum rows, where B = −`inertia` and the diff-rot block is added with +. (a) carries the factor `Gaspard`,
like the momentum diff-rot operator (both had none before the second round, so only `timescale = 'rotation'` was
right); (b), (c), (c′) carry G² through σₙ (not with `thermal_wind = 3`). `thermal_wind = 0/1/2/3` selects whether
(b), (c), (c′) are included and in which form; (a) always is.

Bookkeeping: in `submatrices.py` the T-column labels are registered with vector parity `vT`, the others with `vP`.
`assemble.py` section h today calls only diagonal operators; it needs the s-column loop over Δl ∈ {−2, 0, 2}, the
P-column over {−4, …, 4} and the T-column over {−3, −1, 1, 3}, with the `offd` of `operators.differential_rotation`
(P/s: offd = Δl/2; T: the symm-dependent offsets of its 'utor' branches). The thermal boundary rows and forced runs
need no change (every 'h' label carries the bc rows; `−i wf entropy` is already added). With ThermaD = 0 the
section-h rows are r × (eq)/p₀ in a C⁰ basis, so (a) becomes `h0abu0_D0`, `h0aub0_D0` (no p₀); (b), (c), (c′) do
not fit there: the r row weight makes the P/r part of (c) non-polynomial and the basis has no C¹ for the P′ term.
So with ThermaD = 0 and `thermal_wind ≥ 1` (`utils.adiab_tw`, decided there only) the section-h rows are r² × (eq)/p₀
in a C¹ basis (`gebasis[2] = 1`, still no bc rows, `chop = 0`): the labels are the same (`decode_label` gives
section h the r² weight), and `tw2/4/6`, `tp2/4/6` are σₙ, σₙ′ without p₀. With ThermaD = 0 and `thermal_wind = 0`
or `diff_rot = 0` nothing changes (r × (eq)/p₀ in C⁰). `submatrices.py` raises if a label's derivative order is
above its section's basis (`G[gbx][gbx-dx]` with dx > gbx indexed from the end of the list: a wrong basis, silently).

Diagnostics: the new powers couple different l, so `utils4pp` now builds the diff-rot force and heat terms on an
(r, cosθ) grid once per mode (`diff_rot_setup`) and projects them back on the harmonics, so each per-l worker can pair
them with its own l (§5). The matrix-level budget (`heat_blocks()` in `tests/heat_dr/ebal.py`, entropy balance
λ⟨E⟩ − ⟨A⟩ − ⟨D⟩ − ⟨H⟩ = 0) is the independent check of it.

## 4. Benchmarks and which terms they need

- Mirouh et al. 2016 (JFM 800, 213; kore `shellular_boussinesq`): Boussinesq, Ω(r) = 1 + (N²/2)(1 − r²) imposed
  from Rieutord 2006's baroclinic flow, radial T₀ only; their heat equation (2.9) is (λ + imΩ)T + r v_r = (E/P)ΔT.
  Needs (a) only, with S_tw = 0 (their background is balanced by centrifugal buoyancy, which kore does not carry).
  With (3) switched on, σ₂ = 2N²r²/(3 Beyonce) would be as large as S₀′ and the comparison would test the wrong
  model. Also needs `aub/abu/logrho` in `Boussinesq_profiles`.
- Bekki et al. 2022: fully compressible, solar ρ₀, helioseismic Ω, S_tw from (2) with the full Ω² and s₀ taken
  uniform in r. Needs (a) + (c), (c′); kore's S_tw also has (b), which they drop by construction. Their eq. (1)
  momentum terms −(Ω−Ω₀)∂_φv − 2Ω ẑ×v − r sinθ (v·∇Ω) are kore's existing diff-rot operator.
- Baruteau & Rieutord 2013, Guenel et al. 2016, Astoul et al. 2021: thermal = 0, unaffected.

## 5. Implementation (2026-10-12, commits `be2a4db5`, `18641206`)

Implemented in two rounds, committed as `be2a4db5` and `18641206`. Files in `bin/`, plus
`tests/heat_dr/` and this document under `docs/`:

| file | change |
| --- | --- |
| `params_default.py` | `thermal_wind = 0/1/2/3` (none / linear (3) / exact (3′) thermal wind / prescribed ∂θS_tw = `dSdtheta` sin 2θ, §1; (a) is always on with `diff_rot = 1`, `thermal = 1`); test profile `diff_rot_type = 'rigid'`; 'solar' comments (amplitude, 435 nHz frame, N at ricb 0) |
| `radial_profiles.py` | `sigma2`, `sigma4`, `sigma6` from (3′) with G²; profiles `tw2`, `tw4`, `tw6` = p₀σₙ and `tp2`, `tp4`, `tp6` = p₀σₙ′; `'rigid'` branch of `abu`/`aub`; at ricb = 0 `abuX`/`aubX` evaluate the profiles at \|r\| (even by construction); 'solar' times `diff_rot_amplitude`, tachocline at x = r·aux0 for 'smoothed sun'; `ptw`: σₙ, σₙ′ without p₀ when `utils.adiab_tw`; σ₂ = −(2/3) `dSdtheta`, σ₄ = 0 for `thermal_wind = 3` |
| `submatrices.py` | the labels of §3 (with σ₆ for `thermal_wind = 2`), `tp2/4/6` in both `adj` lists; ThermaD = 0 labels `h0abu0_D0`, `h0aub0_D0`; exits if `Beyonce = 0` with `thermal_wind = 1, 2`; ThermaD = 0 with `thermal_wind ≥ 1`: `gebasis[2] = 1` and `chop = 0` for section h (§3, `docs/thermad0.md`); raises if a label's derivative order exceeds its section's basis; warns for 'Y20-wall-bounded' and 'shellular' at ricb = 0; parity-warning floor so an identically zero S_tw ('cylindrical') stays quiet |
| `operators.py` | `thermal_diffrot(l, 'h', component, Δl)`: 'temp' (a), 'upol' (b) + (c), 'utor' (c′), with σ₆; `Gaspard` on (a) and on `differential_rotation` |
| `heat_dr_coefficients.py` (new) | X, D, Pₙ(X), Pₙ′(X), Pₙ′(X)D for n = 2, 4, 6, built well above lmax and cut (products of the truncated X are wrong near the top) |
| `assemble.py` | section h: s column Δl ∈ {−2, 0, 2}, P column −6…6, T column −5…5; stops with a message unless `lmax − m + 1` is a multiple of 2·ncpus (otherwise `Scatter` dropped degrees or hung) |
| `utils4pp.py`, `spin_doctor.py` | the broken `diff_rot_power` is replaced by `diff_rot_setup()`: the diff-rot force −G[(U₀′·∇)u + (u·∇)U₀′] and the heat terms p₀[i m G ΔΩ s + u·∇S_tw] on an (r, cosθ) grid, projected back per l, with ρ₀. New powers `Wdr`, `Wdr_im` (flow), `Ensdr` (enstrophy), `Wdr_thm` (entropy) enter `resid1`, `resid2`, `resens`, `resid3` |
| `utils.py` | `adiab_tw` (ThermaD = 0, `diff_rot = 1`, `thermal_wind ≥ 1`): `decode_label` gives section h the r² row weight |
| `ru_insane.py` | diff-rot tests (below) and `tp2/4/6` in `row_parity` |

Output columns appended (old columns unchanged, so files written before the change cannot be appended to):
`flow.dat` gets `Wdr`, `Wdr_im`, `Ensdr`; `thermal.dat` gets `Wdr_thm`; `params.dat` gets `diff_rot`,
`diff_rot_amplitude`, `thermal_wind`, the index of `diff_rot_type` in ('Y20', 'Y20-wall-bounded', 'shellular',
'shellular_boussinesq', 'cylindrical', 'conical', 'solar', 'rigid', 'solar_P4', 'tabulated'), −1 if not listed, and
`dSdtheta`; the §8 options follow them. `spin_doctor` is about
1.5 times slower with `diff_rot = 1`.

Changes visible to existing runs: at `Gaspard = 1` and ricb > 0 the matrices are bitwise unchanged. A run with
ThermaD = 0, `diff_rot = 1` and the default `thermal_wind = 1` used to stop and now runs (advection of s only is
`thermal_wind = 0`, as before). All ricb = 0
diff-rot matrices change (the \|r\| fix). 'solar' with an amplitude ≠ 1, or with 'smoothed sun', gives a different
profile than before.

Tests (`tests/heat_dr/run_all.sh`, `mpi_check.sh`, `ru_insane.py`; amplitude −0.25, user-def profiles, ricb 0.35 unless
stated). Every check prints a verdict, `ok` or `FAIL`, and `run_all.sh` stops if a check crashes:

| check | configurations | worst |
| --- | --- | --- |
| sympy: (3′) against the integrated thermal-wind balance (`tw_sympy.py`) | thermal_wind 1 and 2, any G | 0 failures |
| operator blocks vs an (r, θ) quadrature of p₀ s*[i m G ΔΩ s + u·∇S_tw] for random fields (`heat_check.py`) | 43 runs: conical m 0, 1, 2 (symm ±1), N 96 and 160; thermal_wind 0, 1, 2, 3; Y20-wall-bounded, shellular, uniform ('rigid', amplitude 0, thermal_wind 3); smoothed sun and 'solar' at ricb 0 (m 0, 1, 2); G = 3; ThermaD = 0 with thermal_wind 0, 1, 2, 3 (18 runs, also at ricb 0) | ≤ 9e-13 (ricb 0.35, N 96), 1e-15 (N 160), ≤ 1e-10 (smoothed sun, N 120), 6e-14 ('solar', N 400), ≤ 1.1e-14 (ThermaD = 0) |
| heat rows of the assembled A(diff_rot 1) − A(diff_rot 0) vs the same quadrature (`asm_check.py`) | the same runs (ThermaD = 0 with the thermal wind: reference amplitude 0) | ≤ 1e-10 (41 runs; ≤ 2.3e-12 with ThermaD = 0) |
| section-h rows of A, B unweighted vs (r, θ) physics, claimed basis/row/weight only (`row_check.py`) | ThermaD = 0: thermal_wind 0, 1, 2, 3; m 0, 1, 2; G 3; smoothed sun (9 runs) | all ok at 1e-8 |
| rigid offset, λ(ΔΩ, G) = λ(diff_rot 0, G(1+ΔΩ)) − i m G ΔΩ (`rigid_thermal.py`) | m 2, 16 modes; G = 1, 3, 10³ (viscous units); ThermaD = 0 (G 3), and ThermaD = 0 with thermal_wind 1 (G 1, 3) | 4e-12, 2e-12, 7e-10, 2e-11; 9e-10, 9e-12 |
| `thermal_wind = 2` → 1 as the amplitude → 0 (`tw_limit.py`) | conical, Y20-wall-bounded | difference ∝ amplitude² |
| `spin_doctor` powers vs the matrix pairings: \|Wdr + i Wdr_im + 2R\|, \|Wdr_thm − 2 Re H\| (`sd_check.py`) | sun, rfix1, g3_tw2 (G 3, thermal_wind 2), s96, tw0, y20wb, shell, k1_tw2 (ThermaD = 0), s96_tw3 | 1e-8 (sun), 6e-7 (rfix1, N 64), 1e-9 (g3_tw2), ≤ 7e-10 (the others) |
| `spin_doctor` residuals with the new terms (`sd_check.py`) | same | resid1 ≤ 1.5e-6, resid2 ≤ 1.5e-7, resens ≤ 4e-4, resid3 ≤ 2e-8 at N ≥ 96, the diff_rot = 0 level of the same N; without the new terms they are 10²–10⁹ times larger (rfix1 at N 64 is truncation-limited) |
| `ncpus` 1…4 (`mpi_check.sh`) | m 0, 1, 2 (symm −1), smoothed sun m 0, 1, thermal bc 1, forcing 10, thermal_wind 2 with G 3, diff_rot 0 | A, B, B_forced bitwise equal; 5 ranks stops with the message |
| forced run: A_forced = c (A − i ω_f B) (`forced_check.py`) | forcing 6, 7, 9, 10 | 1e-15 (0.04–0.5 with a block missing) |
| `ru_insane.py`: momentum and heat diff-rot blocks vs a grid (I, C, R, Ht, Hw), (a) anti-Hermitian, rigid R = ΔΩ(i m G I + C) | 27 runs: the runs above (22) plus cylindrical with thermal_wind 1 and 2, rigid m 2, conical m 0 symm −1, conical m 1; third round: 16 ThermaD = 0 runs (`docs/thermad0.md`) and 14 on the final code (5 with thermal_wind 3) | grid 2e-10, advection 1e-11, thermal wind 2e-13, anti-Hermitian 2e-12, rigid 2e-15; all PASS |

The ~1e-11 floor at N 96 is the Chebyshev truncation of the profile products (1e-13 to 1e-14 at N 160). 'solar' at
ricb = 0 needs N ≈ 400 for that floor (the tachocline step); at N 120 the checks fail on truncation, not on a bug.
`ru_insane` catches all 18 one-coefficient mutations of the operators tried (sign flips, 0.1–1 % scalings, the target
instead of the source L, a dropped P₄ term); the reviewers' mutations of the \|r\| evaluation and of the G² in
`utils4pp` are caught by `ru_insane` and `sd_check`.

Reviews: five or six independent reviewers plus a critic after each round. No defect in the physics, the signs or the Gaspard
factors; the defects found in the tests and comments are fixed above.

## 6. Benchmark: solar inertial modes (Bekki et al. 2022, Mukhopadhyay et al. 2025)

Best target: Mukhopadhyay, Bekki, Zhu & Gizon 2025 (A&A, arXiv 2501.16797), the setup of Bekki, Cameron & Gizon 2022
with Model S and tables to 1e-4 nHz for compressible, anelastic and Boussinesq models. Their anelastic model is kore's.
Set-up as in `tests/diff_rot/mk_bekki.py` with `drtype=Y20` (N 64, lmax m + 63, ν = κ = 10¹² cm² s⁻¹, δ = 0).

- Uniform rotation: kore matches their anelastic Tables C.1–C.4 to 0.006 nHz or better for 27 modes (n = 0 and 1
  Rossby, high-frequency retrograde, columnar; m = 1…16). This needs the Model S density: the stock 'smoothed sun'
  fit is up to 48 % off near the surface, which moves n = 1, HFR and columnar modes by 1.5–8 nHz (18 at m = 16).
- `Gaspard`: the same problem in time units 1/Ω₀ and 1/Ω_Carrington agrees to 2e-6 nHz; with `thermal_wind = 2` the
  frame (426.86 vs 456 nHz) does not matter (2e-6 nHz), with `thermal_wind = 1` it moves modes by 0.1–0.9 nHz.
- Differential rotation (their C.5–C.9, kore minus target, nHz, `thermal_wind = 2` without u_r ∂ᵣS_tw as in the
  papers): n = 0 within 0.8, n = 1 within 1.1–2.7, high-latitude m 1, 2 within 0.2–1.5, HFR 2.7–6.2, columnar 3.4–9.2,
  growing with m. The remaining difference is the rotation profile: kore's Ω has P₀ + P₂ only (4.5 nHz too fast at the
  equator, 12 at the pole), theirs has P₄ and radial shear.
- Keeping u_r ∂ᵣS_tw (kore always does) moves columnar modes by 5–15 nHz and HFR by 8–10 nHz, and at δ = 0 makes the
  low latitudes superadiabatic: m = 1 symm +1 then has a weakly growing mode (12.4 nHz, growth 1.8 nHz).
- Not reproduced: the growing m = 1 high-latitude mode of Bekki et al. 2022 (−90.9 nHz, growth 14.1 nHz; kore's
  nearest are damped by 3 nHz), and the baroclinic onset of Bekki et al. 2024 (kore needs a 4–8 times larger Δθs).
  Both were the P₂-only rotation profile: with the observed Ω(r, θ) kore has the growing high-latitude mode (+11.3 nHz, Mukhopadhyay et al. 2025: +11.7) and the onset near 500 (§8). The Δθs runs used a run-local edit of σ₂, σ₄; with
  `thermal_wind = 3`, `dSdtheta` = Δθs/c_p the warm-poles 4000 erg g⁻¹ K⁻¹ case gives A and B bitwise equal to that
  run (−101.42 nHz, growth +7.30 nHz).

## 7. Still open

- Done in §8: the switch `thermal_wind_radial` for u_r ∂ᵣS_tw (term (b)), P₄ ('solar_P4') and tabulated Ω(r, θ), the
  Model S density (`solar_density`), and amplitude controls.
- ThermaD = 0 with `thermal_wind ≥ 1`: implemented (r² rows in C¹, §3) and tested (`docs/thermad0.md`); the
  ThermaD → 0 limit with fixed-entropy walls has an unresolved boundary layer, and without diffusion the thermal wind
  has small-scale growing modes that are not converged in lmax, see there. Its rows carry σₙ (ln ρ)′ (term (c)) without
  p₀, which converges slowly in N when ρ nearly vanishes at the surface ('user def' with aux0 → 1, ricb = 0); the
  p₀-weighted checks do not see it, `tests/heat_dr/row_check.py` does.
- The 'smoothed sun' density near the surface (§6): `solar_density = 'model S'` (§8) avoids it; the default is unchanged.
- 'Y20-wall-bounded' and 'shellular' (except even σ > 0) at ricb = 0: not smooth at r = 0, now a WARNING.
- Boussinesq profiles (no `abu`, `aub`, `logrho`): left untouched. The same holds for the profiles read from a file (`'poly'`, `'mesa'`,
  `'gsm'`, `'astropy table'`): only `'user def'` and `'smoothed sun'` define a rotation profile.
- `tests/heat_dr/csolve.py` has no equilibration (kore's `solve.py` has Ruiz scaling); it now warns when every
  eigenvalue comes back equal to the shift.

## 8. Generic differential rotation: any Pₙ, 'solar_P4', 'tabulated' (2026-10-13)

Commit `92a0d020`. ΔΩ = Σₙ aₙ fₙ(r) Pₙ(cos θ), n even, any n.

**Couplings.** n = 0, 2 keep the closed forms of `diff_rot_coefficients.py`. For n ≥ 4 the new
`bin/diff_rot_generic.py` projects the force W = G[ΔΩ(i m u + 2ẑ×u) + sinθ(r∂ᵣΔΩ u_r + ∂θΔΩ u_θ)e_φ] of each source
degree on the target degrees by Gauss–Legendre quadrature in cos θ (exact for these polynomial integrands) and turns the
12 primitive projections into the 18 coefficients of the existing labels, in the naming and normalisation of
`diff_rot_coefficients.py` (cached per (n, m), built to lmax + n + 6). Radial profiles fₙ, r fₙ′, r² fₙ″ have the codes
`abN`, `svN`, `plN` (N = 4, 6, 8, then a, c, e, g for 10–16). Couplings: P→P and T→T |Δl| ≤ n, P↔T |Δl| ≤ n + 1, so
`assemble.py` loops over Δl up to `utils.dr_nmax` (+1). `diff_rot_generic = 1` builds n = 0, 2 with the quadrature
too, as a self-check (agrees to ~1e-12).

**Thermal-wind entropy.** For thermal_wind = 1, 2 with degrees above 2, `radial_profiles.sigma_all` takes the Legendre
coefficients of ∂S/∂x = −(r²G²/(Beyonce g))[x ∂ᵣF + (1 − x²)∂ₓF/r] (F = 2ΔΩ or (1 + ΔΩ)² − 1) by quadrature on the
Chebyshev nodes and integrates in x; σₙ′ by Chebyshev differentiation. S_tw has degrees 2…nmax + 2 (thermal_wind 1) or
2…2 nmax + 2 (thermal_wind 2): profiles `tw2`…`tw8`, `twa`, `twc`, …, `twy` (n = 34) and `tp*` for σₙ′.
`heat_dr_coefficients.py` builds Pₙ(X), Pₙ′(X), Pₙ′(X)D for any n by the Legendre recurrence. thermal_wind = 3 is
unchanged (σ₂ only).

**Profiles.**

| `diff_rot_type` | Ω(r, θ) |
| --- | --- |
| `'solar_P4'` | Ω_RZ + [Ω_CZ(θ) − Ω_RZ](1 + erf((x − rtc)/dtc))/2, Ω_CZ = c₀ + c₂cos²θ + c₄cos⁴θ, `diff_rot_solar` = [464, −76, −50, 433.5, 0.71, 0.045] (Mukhopadhyay et al. 2026); P₀, P₂, P₄ |
| `'tabulated'` | `diff_rot_file`: .npz with `ome[θ, r]` (nHz), `r` (R☉), `theta` (rad), the layout of the Mukhopadhyay et al. 2025 Edmond file `data/diff_rot.npz`; `tests/diff_rot/ls_to_edmond_layout.py` converts the Larson & Schou 2018 table. Projected on P₀…P_nmax (`diff_rot_nmax`, 4–16) at each table radius, then a Chebyshev fit of degree `diff_rot_rdeg` (24) over the shell; the fit and truncation errors are printed |

Both use ΔΩ = Ω/Ω_ref − 1 with Ω_ref/2π = `diff_rot_frame` (456 nHz) and x = r·aux0 for 'smoothed sun'. For the Edmond
profile the truncation error at nmax 8 is up to 17 nHz near the poles at the surface; nmax 12 is the working choice.

**Amplitude controls**, for every type: `diff_rot_term_amp` = [a₀, a₂, …, a₁₆] multiplies each Pₙ term of ΔΩ (missing
entries 1); `diff_rot_amplitude` multiplies the whole of 'solar_P4' and 'tabulated' (the other types keep its old
meaning); `thermal_wind_amplitude` multiplies S_tw, which is otherwise the thermal wind of the scaled ΔΩ.
`thermal_wind_radial` = 0 drops u_r ∂ᵣS_tw (term (b)), as the papers do; 1 (default) keeps it.

**Model S density.** `solar_density = 'model S'` replaces the 'smoothed sun' ln ρ and ln p by Chebyshev fits of
`bin/model_s.at` (degree `solar_density_deg` = 30 on `solar_density_window`, default the shell widened by 0.01 below and
0.005 above). It reproduces the run-local Model S fit of the §6 benchmark runs: A and B bitwise equal.

**Other files.** `utils.py`: `dr_high`, `dr_nmax`, `tw_nmax`, `tw_degrees`, `dr_code`, `tw_code`, `odd_profiles`,
`dr_offd`. `operators.py`: `differential_rotation` sums the closed forms and the generic terms; `thermal_diffrot` loops
over `tw_degrees` and Δl up to `tw_nmax`. `submatrices.py`: the labels per n. `utils4pp.py` (spin_doctor),
`ru_insane.py`, `tests/heat_dr/ebal.py`: generic Δl ranges, ΔΩ and S_tw summed over n, the switches and amplitudes.
`spin_doctor.py`: 'solar_P4' and 'tabulated' get the type indices 8 and 9, and `params.dat` gets 25 more columns after
`dSdtheta` (listed in `docs/differential-rotation.md`, Output), so files written before cannot be appended to.

**Changes visible to existing runs:** none. A and B are bitwise equal to 18641206 for 13 configurations
(`tests/diff_rot/regress.sh`) and for the benchmark runs of §6.

**Tests** (`tests/diff_rot/README.md`): generic vs closed-form coefficients 4e-11, no coupling outside the assembled Δl
ranges for n 4–16 (6e-12); `ru_insane` grid checks ~1e-14 on 'tabulated' (thermal_wind 1 and 2, radial 0 and 1, term
and thermal-wind amplitudes) and 'solar_P4', and it catches a 1e-4 change of one generic coefficient or of σₙ;
spin_doctor powers against the matrix pairings ≤ 1e-8, with the energy balances closing to 1e-7 (1e-2 to 1 without the
new terms); kore's ∂θS_tw against the Edmond `dsdt` 3 % rms (pole-to-equator contrast within 0.1–2 %).

**Like-for-like benchmark** (Mukhopadhyay et al. 2025 set-up: Model S density, shell 0.71–0.985 R☉, ν = κ = 1e12,
δ = 0, stress-free, ∂ᵣs = 0, thermal_wind = 2 without u_r ∂ᵣS_tw, their Ω from the Edmond file, N 64, lmax m + 63;
`tests/diff_rot/run_bekki.py` rebuilds and solves these five modes):

| mode (symm) | kore, nmax 12 (nHz) | Mukhopadhyay compressible (nHz) | kore − theirs | kore, P₀ + P₂ only (§6) |
| --- | --- | --- | --- | --- |
| m 1 high-latitude (−1) | −95.759, growth +11.301 | −95.70, +11.69 | −0.06, −0.39 | −95.67, −15.8 |
| m 3 Rossby n0 (−1) | −262.052, −0.877 | −262.07, −0.88 | +0.02, 0.00 | within 0.8 |
| m 3 Rossby n1 (−1) | −181.934, −9.700 | −182.05, −9.67 | +0.12, −0.03 | 1.1–2.7 off |
| m 3 columnar (+1) | 210.908, −11.090 | 210.84, −11.09 | +0.07, 0.00 | 3.4–9.2 off |
| m 10 HFR (+1) | −211.088, −19.963 | −211.28, −20.00 | +0.19, +0.04 | 2.7–6.2 off |

The m 1 mode is converged: nmax 8 / 12 / 16 give −95.805 / −95.759 / −95.749 (growth +11.48 / +11.30 / +11.29), and
N, lmax 96 reproduces N 64 to 1e-6 nHz. Their targets are their compressible values; the anelastic ones of the
same paper differ from them by amounts of this order, and their published eigenfunctions use 24 radial points. The m 3
symm −1 window around n0 also holds two growing modes (−260.59 nHz, growth +17.98; −250.45, +5.04) with residuals
3e-5 to 4e-5 (1e-7 for n0); not examined, and not reported in the paper's tables.

**Bekki et al. 2024 onset with the observed Ω** (thermal_wind = 3, Δθs prescribed, otherwise as above, m 1 symm −1;
`tests/diff_rot/mk_bekki.py tw=3 ds=-1000 file=...`, then `solve_bekki.py RUN -93 5`):
the m 1 high-latitude mode becomes unstable at
|Δθs| = 425 erg/g/K with the Edmond Ω and 546 with the Larson & Schou 2018 table (c_p = 3.46e8; 513 and 658 in the
papers' c_p = 4.17e8), against ~500 published and 2199 (2650) for the radially uniform P₀ + P₂ profile of §6. Growth
at |Δθs| = 0, 250, 500, 1000, 2000: −3.70, −1.60, +0.68, +5.02, +12.53 nHz (Edmond, mode at −91.1…−94.6 nHz);
−3.83, −2.30, −0.36, +3.56, +10.42 (Larson & Schou, −96.1…−98.9 nHz). So the gap of §6 was the radial shear of Ω in
the lower convection zone, as step 1 of the comparison concluded with kore's 'solar' step (onset 509).
