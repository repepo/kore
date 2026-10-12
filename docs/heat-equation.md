# Heat equation with differential rotation: the missing terms

Derived 2026-10-11 for kore `anelastic_log` (HEAD `652e1490`). Verified with sympy in `checks/check_heat.py`
(output `checks/check_heat.out`), by four independent re-derivations (thermal wind, linearisation, harmonic couplings,
code mapping) and against Bekki, Cameron & Gizon 2022. Implemented and tested on 2026-10-12 in the work copy
`~/lab/aneldiro` (§5), uncommitted; nothing in Andres's Dropbox folder was changed.

## 0. Result in one paragraph

kore's heat equation (section h, `operators.entropy/thermal_advection/thermal_diffusion`) is

    p₀ ∂s/∂t = −p₀ uᵣ S₀′ + ThermaD ∇·(κ p₀ ∇s),      p₀ = ρ₀T₀, rows × r² (× r and ÷ p₀ if ThermaD = 0).

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

This linear form is what §2–3 implement. Keeping the ΔΩ² term changes σ₂, σ₄ by products of the profiles and adds a
P₆ component (expressions in `checks/check_heat.out`): for the conical profile σ₂ and σ₄ change by the factors 1 + 2ε/3
and 1 + 12ε/11, i.e. 17 % and 27 % for ε = −0.25, and it would need P₆(X) = (231X⁶ − 315X⁴ + 105X² − 5)/16 and Δl = ±6 blocks in §2–3.
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

    (a) advection        i m [ f₀ δ_{l′l} + f₂ P₂(X)_{l′l} ] s_l                                   s → s, Δl = 0, ±2
    (b) radial           ( L_l P_l / r ) [ σ₂′ P₂(X)_{l′l} + σ₄′ P₄(X)_{l′l} ]                    P → s, Δl = 0, ±2, ±4
    (c) latitudinal, P   ( [P_l + r P_l′ + r h′ P_l] / r² ) [ σ₂ (3X D)_{l′l} + σ₄ ((35X³ − 15X) D / 2)_{l′l} ]   P → s, Δl = 0, ±2, ±4
    (c′) latitudinal, T  −( i m T_l / r ) [ σ₂ (3X)_{l′l} + σ₄ ((35X³ − 15X)/2)_{l′l} ]              T → s, Δl = ±1, ±3

where L_l is that of the source degree. The even Δl for P and odd for T match the equatorial-symmetry classes of
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
in full-sphere runs. The 'solar' profile's r-parity defect (README §5.6) passes into σ₂, σ₄ through f₂′ until fixed.
Rows × r² p₀ and ÷ L_{l′}³ like the other section-h operators:

    (a)  i m r² p₀ f₀ s,  i m r² p₀ f₂ P₂(X) s          labels  h0pss0abu0_D0,  h0pss0aub0_D0        (s column, Δl 0, ±2)
    (b)  L_l r p₀ σₙ′ Pₙ(X) P                           labels  h1tp20_D0,  h1tp40_D0                 (P column, Δl 0, ±2, ±4)
    (c)  p₀ σₙ [P + r P′ + r h′ P] Pₙ′(X) D              labels  h2tw20_D0, h1tw20_D1, h1tw20lho1_D0 (and tw4)   (P column)
    (c′) −i m r p₀ σₙ Pₙ′(X) T                           labels  h1tw20_D0,  h1tw40_D0                 (T column, Δl ±1, ±3)

All twelve labels parse with `utils.decode_label` (13-character two-function form for the first and last of (c)).
Sign: the section-h rows of A hold the right-hand side of the heat equation (`thermal_advection` enters as
−r p₀ S₀′ u_r, with B = +`entropy`), so the three left-hand-side terms of (1) go into A with a minus sign, opposite
to the momentum rows, where B = −`inertia` and the diff-rot block is added with +. Like the momentum diff-rot operator
these terms carry no `Gaspard` factor (timescale 'rotation' only). A switch (say `thermal_wind = 0/1`) selects
whether (b), (c), (c′) are included; (a) always is.

Bookkeeping: in `submatrices.py` the T-column labels are registered with vector parity `vT`, the others with `vP`.
`assemble.py` section h today calls only diagonal operators; it needs the s-column loop over Δl ∈ {−2, 0, 2}, the
P-column over {−4, …, 4} and the T-column over {−3, −1, 1, 3}, with the `offd` of `operators.differential_rotation`
(P/s: offd = Δl/2; T: the symm-dependent offsets of its 'utor' branches). The thermal boundary rows and forced runs
need no change (every 'h' label carries the bc rows; `−i wf entropy` is already added). ThermaD = 0 is better left
for later: its r row weight makes the P/r part of (c) non-polynomial and its section-h basis has no C¹ for the P′
term, so it would need the r² weight and `gebasis[2] = 1` for section h, besides p₀-free profile names.

Diagnostics: the new powers couple different l, so the per-l `utils4pp.thermal_worker` cannot host them and
`spin_doctor`'s thermal residual will measure the missing term until it is extended. The simple route is the
matrix-level budget: a `heat_blocks()` in `kore/balance/ebal.py` pairing the new blocks, like `dr_blocks` for the
momentum term, with the entropy balance λ⟨E⟩ − ⟨A⟩ − ⟨A_tw⟩ − ⟨D⟩ = 0. `ru_insane` can test (a) as anti-Hermitian.

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

## 5. Implementation (2026-10-12, work copy `~/lab/aneldiro`, uncommitted)

Implemented in the clone `/Users/triana/lab/aneldiro` (branch `anelastic_log`, HEAD `652e1490`, origin = the Dropbox
checkout), left uncommitted. Seven files in `bin/` changed or added, plus `tests/heat_dr/` and this document under `docs/`:

| file | change |
| --- | --- |
| `params_default.py` | `thermal_wind = 1` switch (terms (b), (c), (c′); (a) is always on with `diff_rot = 1`, `thermal = 1`); comment line for the test profile `diff_rot_type = 'rigid'` |
| `radial_profiles.py` | `sigma2`, `sigma4` from (3); profiles `tw2`, `tw4` = p₀σₙ and `tp2`, `tp4` = p₀σₙ′ with their `…X` fits in `proffdir`; `'rigid'` branch of `abu`/`aub` (f₀ = ΔΩ, f₂ = 0) |
| `submatrices.py` | the twelve labels of §3 (`vP` for the s and P columns, `vT` for the T column); `tp2`, `tp4` in both `adj` lists; exits if `ThermaD = 0`, or `Beyonce = 0` with `thermal_wind = 1` |
| `ru_insane.py` | `tp2`, `tp4` in `row_parity` |
| `heat_dr_coefficients.py` (new) | X, D and P₂(X), P₄(X), P₂′(X), P₄′(X), Pₙ′(X)D per m, built to lmax + 8 (products of the truncated X are wrong within a few degrees of the truncation); `coef(name, l, m, Δl)` |
| `operators.py` | `thermal_diffrot(l, 'h', component, Δl)`: 'temp' (a), 'upol' (b) + (c), 'utor' (c′), returned as −(…)/L³ with the `offd` of the source block |
| `assemble.py` | section h: the s-column loop over Δl ∈ {−2, 0, 2}, the P-column over {−4, …, 4} and the T-column over {−3, −1, 1, 3} |

Deviations from §3: none in the labels; the angular products are taken from matrices rather than closed forms, which
is what makes (c) and (b) exact at every l ≤ lmax. The thermal boundary rows, `forcing` and `ncpus > 1` paths are
untouched (the new blocks use the same `col0`, `row` and `offd` arithmetic as the Coriolis couplings), but only
single-rank runs were tested.

Tests (`tests/heat_dr/run_all.sh`; outputs in `kore/diffrot/checks/heat/`), amplitude −0.25, user-def profiles with ricb 0.35 unless stated:

| check | configurations | worst relative difference |
| --- | --- | --- |
| operator blocks vs (r, θ) quadrature of p₀ s*[i m ΔΩ s + u·∇S_tw], random fields (`heat_check.py`) | conical: m 1 (N 96, 160), m 0, m 2 symm −1, thermal_wind 0, smoothed sun ricb 0 (N 120, lmax 30, symm ±1); Y20-wall-bounded and shellular (f′, f″ ≠ 0), m 1 | 5e-12, 5e-14, 9e-12, 4e-12, 2e-15, 4e-11, 3e-11; 2e-11, 7e-12 |
| heat rows of the assembled A(diff_rot 1) − A(diff_rot 0) vs the same quadrature (`asm_check.py`) | the same runs except N 160 and thermal_wind 0 | the same numbers |
| the code's σ₂, σ₄, σ₂′, σ₄′ fits vs a direct evaluation of (3) at the nodes (inside `heat_check.py`) | all runs | 2e-12 (5e-10 at ricb 0) |
| rigid offset with the heat equation on, λ(ΔΩ) = (1+ΔΩ)λ(0; E, κ, Beyonce rescaled) − i m ΔΩ (`rigid_thermal.py`) | m 2, 16 thermally coupled modes (Im B ≈ 0.2–0.5 of the balance), ΔΩ 0.1 | 9e-12 |
| smoothed sun, ricb 0, 12 modes near −0.05 + 0.8i: entropy balance λE − A − D − H (`terms.py`) | Beyonce 10³, ViscosD = ThermaD = 10⁻³ | residual 3e-10 … 2e-9, Im H ≈ 0.2 Im A |

The 1.7e-6 seen at N = 48 is the Chebyshev truncation of the profile products, not a defect (it falls to 5e-14 at N = 160).

Review (six independent refuters on the label convention, the angular coefficients, the index logic and packing,
the profiles and parity, the tests, and the physics, then a critic; `checks/heat/`): no defect in the operator, the
coefficients (re-derived against Gauss–Legendre projections to 1e-12 and end to end against a direct projection of
the three terms in 182 (m, symm, Δl) cases to 3e-11), the packing (5572 couplings against the hard-coded offsets of
`operators.differential_rotation`), the parity lists or the signs (kore's inertia, Coriolis and buoyancy operators
were re-derived to fix the sign of the buoyancy term the thermal wind assumes). Two findings were folded in: with
`timescale = 'rotation'` `set_scales()` leaves `Beyonce = 0`, and σ₂, σ₄ ∝ 1/Beyonce then silently produce NaN
submatrices, so `submatrices.py` now exits unless `Beyonce > 0` when `thermal_wind = 1`; and the conical profile has
f′ = 0, so the f₀′, f₂′ terms of (3) were only covered by the sympy check until the wall-bounded and shellular runs
above were added. Not tested: `ncpus > 1` (the blocks use the same `col0`/`row` arithmetic as the Coriolis
couplings), `forcing ≠ 0` with `diff_rot = 1` and `thermal = 1`, `bci_thermal`/`bco_thermal = 1` (the bc rows are
untouched by construction), the 'solar' and 'cylindrical' profiles at ricb = 0 (pre-existing parity issues; for
'cylindrical' S_tw = 0 exactly and the parity warning of `submatrices.py` fires on round-off), and Boussinesq runs
(`Boussinesq_profiles` has no `abu`/`aub`/`logrho`, README §3D).

Still open: `spin_doctor`'s thermal residual does not know the new terms (README §5), so `resid-s` of a diff-rot thermal
run is meaningless until `utils4pp.thermal_worker` is extended; `ru_insane` has no anti-Hermitian test of (a) (the
thermal_wind = 0 run shows Re⟨H⟩ ≈ 1e-17); ThermaD = 0; the exact
(1+ΔΩ)² thermal wind (needs P₆ and σ₆); the 'solar' profile parity (README §5.6); a benchmark run against
Bekki, Cameron & Gizon 2022 (§4).
