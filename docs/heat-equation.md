# Heat equation with differential rotation: the missing terms

Derived 2026-10-11 for kore `anelastic_log` (HEAD `652e1490`). Verified with sympy in `checks/check_heat.py`
(output `checks/check_heat.out`), by four independent re-derivations (thermal wind, linearisation, harmonic couplings,
code mapping) and against Bekki, Cameron & Gizon 2022. Implemented and tested on 2026-10-12 in the work copy
`~/lab/aneldiro` (§5), committed there by Andres as `09285a5e`. A second round, uncommitted on top of it, the same day added the `Gaspard` factors, the exact (1+ΔΩ)² thermal
wind (`thermal_wind = 2`), `ThermaD = 0`, the `spin_doctor` diagnostics and the `ru_insane` tests (§5), and ran the
solar benchmark (§6). A third round, also uncommitted, added ThermaD = 0 with the thermal wind (§3, `docs/thermad0.md`)
and a prescribed S_tw, `thermal_wind = 3` (§1). §7 lists what is still open.

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
factor G². With e = 0 for `thermal_wind = 1` and e = 1 for `thermal_wind = 2` (closed forms of `checks/check_heat.out`):

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

Verification (`checks/check_heat.py`, 0 failures): (3) against direct integration of (2); the curl of the background
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
`kore/balance/ebal.py` and `ru_insane.row_parity`) next to `gra, pdS, dSd`, otherwise term (b) is silently dropped
in full-sphere runs. (The 'solar' profile's r-parity defect, README §5.6, is fixed by evaluating the profiles at |r|, §5.)
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
  model. Also needs `aub/abu/logrho` in `Boussinesq_profiles` (README §3D).
- Bekki et al. 2022: fully compressible, solar ρ₀, helioseismic Ω, S_tw from (2) with the full Ω² and s₀ taken
  uniform in r. Needs (a) + (c), (c′); kore's S_tw also has (b), which they drop by construction. Their eq. (1)
  momentum terms −(Ω−Ω₀)∂_φv − 2Ω ẑ×v − r sinθ (v·∇Ω) are kore's existing diff-rot operator.
- Baruteau & Rieutord 2013, Guenel et al. 2016, Astoul et al. 2021: thermal = 0, unaffected.

## 5. Implementation (2026-10-12, work copy `~/lab/aneldiro`)

Implemented in the clone `/Users/triana/lab/aneldiro` (branch `anelastic_log`, origin = the Dropbox checkout) in two
rounds: the first is commit `09285a5e` on `652e1490`, the second and third are uncommitted on top of it. Files in `bin/`, plus
`tests/heat_dr/` and this document under `docs/`:

| file | change |
| --- | --- |
| `params_default.py` | `thermal_wind = 0/1/2/3` (none / linear (3) / exact (3′) thermal wind / prescribed ∂θS_tw = `dSdtheta` sin 2θ, §1; (a) is always on with `diff_rot = 1`, `thermal = 1`); test profile `diff_rot_type = 'rigid'`; 'solar' comments (amplitude, 435 nHz frame, N at ricb 0) |
| `radial_profiles.py` | `sigma2`, `sigma4`, `sigma6` from (3′) with G²; profiles `tw2`, `tw4`, `tw6` = p₀σₙ and `tp2`, `tp4`, `tp6` = p₀σₙ′; `'rigid'` branch of `abu`/`aub`; at ricb = 0 `abuX`/`aubX` evaluate the profiles at \|r\| (even by construction, README §5.6); 'solar' times `diff_rot_amplitude`, tachocline at x = r·aux0 for 'smoothed sun' (README §5.7); `ptw`: σₙ, σₙ′ without p₀ when `utils.adiab_tw`; σ₂ = −(2/3) `dSdtheta`, σ₄ = 0 for `thermal_wind = 3` |
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
'shellular_boussinesq', 'cylindrical', 'conical', 'solar', 'rigid'), −1 if not listed, and `dSdtheta`. `spin_doctor` is about
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
Runs, scripts and notes in `checks/bekki/` (N 64, lmax m + 63, ν = κ = 10¹² cm² s⁻¹, δ = 0).

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
  Both are likely the P₂-only rotation profile; not proven. The Δθs runs used a run-local edit of σ₂, σ₄; with
  `thermal_wind = 3`, `dSdtheta` = Δθs/c_p the warm-poles 4000 erg g⁻¹ K⁻¹ case gives A and B bitwise equal to that
  run (−101.42 nHz, growth +7.30 nHz).

## 7. Still open

- A switch to drop u_r ∂ᵣS_tw (term (b)) and a P₄ (or tabulated) Ω(r, θ): both are needed for a like-for-like
  comparison with §6. Not added, to keep the code as it is until wanted.
- ThermaD = 0 with `thermal_wind ≥ 1`: implemented (r² rows in C¹, §3) and tested (`docs/thermad0.md`); the
  ThermaD → 0 limit with fixed-entropy walls has an unresolved boundary layer, and without diffusion the thermal wind
  has small-scale growing modes that are not converged in lmax, see there. Its rows carry σₙ (ln ρ)′ (term (c)) without
  p₀, which converges slowly in N when ρ nearly vanishes at the surface ('user def' with aux0 → 1, ricb = 0); the
  p₀-weighted checks do not see it, `tests/heat_dr/row_check.py` does.
- The 'smoothed sun' density near the surface (§6).
- 'Y20-wall-bounded' and 'shellular' (except even σ > 0) at ricb = 0: not smooth at r = 0, now a WARNING.
- Boussinesq profiles (no `abu`, `aub`, `logrho`): left untouched.
- `tests/heat_dr/csolve.py` has no equilibration (kore's `solve.py` has Ruiz scaling); it now warns when every
  eigenvalue comes back equal to the shift.
