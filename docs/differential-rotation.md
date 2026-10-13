# Differential rotation

kore can solve for linear perturbations of a background that rotates differentially,

    Ω(r, θ) = Ω_ref [1 + ΔΩ(r, θ)],      ΔΩ = Σₙ fₙ(r) Pₙ(cos θ),  n = 0, 2, 4, …

where Ω_ref is the rotation rate of the frame. This page is the user guide: what is modelled, the parameters, the
built-in profiles and their limits. The derivation and the implementation notes are in
[heat-equation.md](heat-equation.md) (thermal terms, generic Pₙ couplings, benchmarks) and [thermad0.md](thermad0.md)
(`ThermaD = 0`); the checks are described in `tests/diff_rot/README.md` and `tests/heat_dr/README.md`.

## What is modelled

The perturbation is advected by the background flow U₀ = G ΔΩ r sinθ e_φ, with G = `Gaspard` = Ω_ref τ. In the
momentum equation this adds

    G [ ΔΩ (i m u + 2 ẑ×u) + r sinθ (u·∇ΔΩ) e_φ ],

with the density stratification carried through u = ρ₀⁻¹∇×∇×(ρ₀ P r) + ∇×(T r). With `thermal = 1` the entropy
equation gains the advection of s by U₀ and, optionally, the advection of a latitudinal background entropy S_tw(r, θ),

    i m G ΔΩ s  +  u_r ∂S_tw/∂r  +  (u_θ/r) ∂S_tw/∂θ,

where S_tw is either the thermal wind that balances Ω(r, θ) (Bekki, Cameron & Gizon 2022, eq. 27) or prescribed.

Not modelled: the background is fixed (nothing maintains or evolves Ω and S_tw against diffusion), ρ₀, T₀ and g stay
spherical (no centrifugal distortion), and there is no magnetic field.

**Supported reference models.** Only `model_type = 'user def'` and `'smoothed sun'` define the rotation profiles
(`radial_profiles.user_defined_profiles.abu/aub`). With `'poly'`, `'mesa'`, `'gsm'`, `'astropy table'` or
`'Boussinesq'` a `diff_rot = 1` run stops with an `AttributeError` in `submatrices.py`.

## Frame and units

ΔΩ is relative to the frame, and G converts it to the time unit, so any `timescale` works. Two things follow:

- `ViscosD`, `ThermaD` and `Beyonce` must be computed for Ω_ref. For `'solar'` the frame must rotate at 435 nHz; for
  `'solar_P4'` and `'tabulated'` at `diff_rot_frame` (456 nHz by default). The other types are dimensionless.
- With `thermal_wind = 2` the eigenvalues do not depend on the choice of frame. With `thermal_wind = 1` (linear in
  ΔΩ) they do, at O(ΔΩ²): 0.1–0.9 nHz for the solar modes of heat-equation.md §6.

Eigenvalues are in the rotating frame of Ω_ref, as for uniform rotation.

## Quick start

```python
par.diff_rot           = 1
par.diff_rot_type      = 'conical'   # see the table of profiles below
par.diff_rot_amplitude = -0.25       # ε for 'conical'
par.thermal_wind       = 1           # only read with thermal = 1; needs Beyonce > 0
```

Then run `submatrices.py`, `assemble.py` and `solve.py` as usual. As for every run, `lmax - m + 1` must be a multiple
of `2*ncpus` (`assemble.py` stops otherwise).

## Parameters

| parameter | default | meaning |
| --- | --- | --- |
| `diff_rot` | 0 | 1 adds the differential-rotation terms |
| `diff_rot_type` | `'Y20'` | the profile, see below |
| `diff_rot_amplitude` | 1.0 | amplitude; its meaning depends on the type (table below) |
| `diff_rot_term_amp` | `[1.0]*9` | a₀, a₂, …, a₁₆: factors of the P₀, P₂, …, P₁₆ terms of ΔΩ, any type (missing entries are 1) |
| `diff_rot_frame` | 456.0 | `'solar_P4'`, `'tabulated'`: Ω_ref/2π in nHz |
| `diff_rot_solar` | `[464, -76, -50, 433.5, 0.71, 0.045]` | `'solar_P4'`: c₀, c₂, c₄, Ω_RZ (nHz), r_tc, d_tc |
| `diff_rot_file` | `''` | `'tabulated'`: the .npz file, absolute or relative to the run directory |
| `diff_rot_nmax` | 8 | `'tabulated'`: highest degree kept, even, 4 to 16 |
| `diff_rot_rdeg` | 24 | `'tabulated'`: degree of the Chebyshev fit of each fₙ in r |
| `diff_rot_generic` | 0 | 1 builds the P₀, P₂ couplings by quadrature too (self-check, agrees to ~1e-12) |
| `thermal_wind` | 1 | heat-equation terms, see below |
| `thermal_wind_radial` | 1 | 0 drops u_r ∂S_tw/∂r |
| `thermal_wind_amplitude` | 1.0 | factor on S_tw (`thermal_wind` 1, 2) |
| `dSdtheta` | 0.0 | `thermal_wind = 3`: ∂S_tw/∂θ = `dSdtheta` sin 2θ |
| `solar_density` | `'fit'` | `'smoothed sun'` only: `'model S'` uses Chebyshev fits of `bin/model_s.at` for ln ρ, ln p (needs ricb > 0) |
| `solar_density_deg`, `solar_density_window` | 30, None | degree and x-interval of those fits (None: the shell, widened by 0.01 below, 0.005 above) |

## Profiles

a = `diff_rot_amplitude`. For the solar types x is the fractional solar radius: x = r·`aux0` with
`model_type = 'smoothed sun'`, x = r otherwise.

| `diff_rot_type` | ΔΩ(r, θ) | degrees | notes |
| --- | --- | --- | --- |
| `'rigid'` | a | P₀ | test profile: λ(a, G) = λ(diff_rot 0, G(1 + a)) − i m G a exactly; a = 0 gives uniform rotation with the diff-rot machinery on (needed for `thermal_wind = 3`) |
| `'Y20'` | a P₂(cos θ) | P₂ | P₂ unnormalised, despite the name |
| `'Y20-wall-bounded'` | a (1 − r)(r − ricb) P₂(cos θ) | P₂ | not smooth at r = 0: use ricb > 0 |
| `'shellular'` | r^σ − 1, σ = a | P₀ | Baruteau & Rieutord 2013; not smooth at r = 0 unless σ is even and positive |
| `'shellular_boussinesq'` | (N₀²/2)(1 − r²), N₀² = a | P₀ | Mirouh et al. 2016; use `thermal_wind = 0` (their background has no thermal wind) and `'user def'` profiles |
| `'cylindrical'` | ε r² sin²θ, ε = a | P₀, P₂ | Baruteau & Rieutord 2013; barotropic, S_tw = 0 |
| `'conical'` | ε sin²θ, ε = a | P₀, P₂ | Guenel et al. 2016 |
| `'solar'` | a × helioseismic fit, P₀ + P₂, tanh step at x = 0.71 of width 0.1 | P₀, P₂ | relative to 435 nHz; 4.5 nHz too fast at the equator and 12 nHz at the pole compared with the observations (heat-equation.md §6) |
| `'solar_P4'` | a [Ω/Ω_ref − 1], Ω/2π = Ω_RZ + [Ω_CZ(θ) − Ω_RZ](1 + erf((x − r_tc)/d_tc))/2, Ω_CZ = c₀ + c₂cos²θ + c₄cos⁴θ | P₀, P₂, P₄ | `diff_rot_solar`, `diff_rot_frame`; defaults from Mukhopadhyay et al. 2026 |
| `'tabulated'` | a [Ω/Ω_ref − 1], Ω from `diff_rot_file` | P₀ … P_nmax | below |

`diff_rot_term_amp` multiplies each fₙ for every type, after the type's own amplitude. To switch off only the P₄ part
of `'solar_P4'`, for example, use `diff_rot_term_amp = [1, 1, 0]`.

**`'tabulated'`.** The file holds `ome[θ, r]` (nHz), `r` (R☉) and `theta` (rad), the layout of `data/diff_rot.npz`
of the Mukhopadhyay et al. 2025 data set (Edmond). `tests/diff_rot/ls_to_edmond_layout.py` converts the Larson &
Schou 2018 table to it. kore projects Ω on P₀ … P_nmax at each table radius, fits each fₙ in r over the table radii
within 0.01 of the shell, and prints the fit residuals and the truncation error max|Ω(P₀…P_nmax) − Ω| in nHz;
check that line. For the Edmond profile the truncation error near the poles at the surface is up to 17 nHz at
`diff_rot_nmax = 8`; 12 is the working choice.

## Heat equation (`thermal = 1`)

| `thermal_wind` | terms added | S_tw |
| --- | --- | --- |
| 0 | i m G ΔΩ s | none (e.g. Mirouh et al. 2016) |
| 1 (default) | the three terms | thermal wind of Ω to first order in ΔΩ (S_tw ∝ G²/`Beyonce`) |
| 2 | the three terms | thermal wind of the full (1 + ΔΩ)² balance; frame independent |
| 3 | the three terms | prescribed: ∂S_tw/∂θ = `dSdtheta` sin 2θ, uniform in r (Bekki et al. 2024) |

- `thermal_wind` 1 and 2 need `Beyonce > 0`. With `timescale = 'rotation'`, `set_scales()` sets `Beyonce = 0`, so set
  it after calling `set_scales()`.
- `dSdtheta` is in the units of s (δs/c_p): `dSdtheta` = Δθs/c_p, with Δθs in erg g⁻¹ K⁻¹. Bekki et al. use
  c_p = 4.17e8, Model S has about 3.46e8. Negative values give warm poles. The thermal-wind amplitude does not apply.
- `thermal_wind_radial = 0` drops u_r ∂S_tw/∂r, as Bekki et al. 2022 and Mukhopadhyay et al. 2025 do. Keeping it is
  the consistent choice, but it moves the solar columnar and high-frequency retrograde modes by 5–15 nHz.
- With the thermal wind S_tw is the only latitudinal entropy gradient; in a convection zone with dS₀/dr = 0 it is
  also the only background entropy gradient, and its effect on the modes is of the same order as the momentum terms.
- `ThermaD = 0` works with every option. With `thermal_wind ≥ 1` the heat rows are written as r²(eq)/p₀ in a C⁽¹⁾
  basis; see [thermad0.md](thermad0.md) for the caveats (unresolved boundary layer as `ThermaD` → 0, small-scale
  growing modes without diffusion).

## Resolution and cost

- Couplings span |Δl| ≤ n_max for P→P and T→T, n_max + 1 for P↔T, and up to n_max + 2 (`thermal_wind` 1) or
  2 n_max + 2 (`thermal_wind` 2) in the heat rows, so matrices get wider with `'tabulated'` and large `diff_rot_nmax`.
- With `ricb = 0` the profiles are evaluated at |r|. `'Y20-wall-bounded'` and `'shellular'` (except even σ > 0) have a
  kink at r = 0 and `submatrices.py` prints a WARNING. The `'solar'` tachocline step needs N ≈ 400 at `ricb = 0`.
- Check convergence with `spin_doctor` as for any run; its residuals include the diff-rot terms.

## Output

`spin_doctor` adds the diff-rot powers to `flow.dat` (`Wdr`, `Wdr_im`, `Ensdr`) and `thermal.dat` (`Wdr_thm`), and
writes the options to `params.dat`, after the general columns (the order is that of `spin_doctor.param_table`):

    diff_rot, diff_rot_amplitude, thermal_wind, diff_rot_type, dSdtheta,
    diff_rot_term_amp_0, _2, ..., _16, thermal_wind_amplitude, thermal_wind_radial,
    diff_rot_frame, diff_rot_nmax, diff_rot_rdeg, diff_rot_generic, diff_rot_solar_0 ... _5,
    solar_density, solar_density_deg, solar_density_window_lo, solar_density_window_hi

`diff_rot_type` and `solar_density` are written as their index in `spin_doctor.diff_rot_types` ('Y20' 0, …, 'rigid' 7,
'solar_P4' 8, 'tabulated' 9) and `spin_doctor.solar_densities` ('fit' 0, 'model S' 1); a default
`solar_density_window` (None) is written as nan. `diff_rot_file` is not recorded. A `params.dat` written by an older
version has fewer columns; start a new file rather than appending to it.

## Examples

**Conical rotation, Guenel et al. 2016.** `ricb = 0.71`, `diff_rot_type = 'conical'`, `diff_rot_amplitude = -0.25`,
E = 1e-5, uniform density (`'user def'`), stress-free, `thermal = 0`, m = 0: their DT mode near ω ≈ 1.82; kore gives
ω = 1.8173, σ = −0.021.

**Solar inertial modes, Mukhopadhyay et al. 2025.** The like-for-like set-up of heat-equation.md §8.
`tests/diff_rot/mk_bekki.py` builds it (with keys for m, the profile, the thermal-wind options and Δθs),
`solve_bekki.py` solves it near a target frequency in nHz, and `run_bekki.py` reproduces the five modes of §8. The
parameters it sets:

```python
par.timescale = 'rotation'; par.set_scales()
par.model_type = 'smoothed sun'; par.solar_density = 'model S'
par.aux0 = 0.985; par.ricb = 0.71/0.985                  # shell 0.71-0.985 R☉
par.aux1, par.aux2, par.aux3, par.aux4, par.aux5 = 0.0, 0.1, 0.68, 0.06, 0.0   # dS0/dr = 0 (δ = 0)
par.thermal = 1; par.bci = par.bco = 0; par.bci_thermal = par.bco_thermal = 1
par.diff_rot = 1; par.diff_rot_type = 'tabulated'; par.diff_rot_file = 'diff_rot.npz'
par.diff_rot_nmax = 12; par.diff_rot_frame = 456.0
par.thermal_wind = 2; par.thermal_wind_radial = 0
# L = aux0 R☉, Ω = 2π diff_rot_frame nHz, ν = κ = 1e12 cm²/s, g_surf = 2.7361e4 cm/s²:
par.Gaspard = 1; par.Beyonce = g_surf/(L*Ω**2); par.ViscosD = ν/(Ω*L**2); par.ThermaD = κ/(Ω*L**2)
par.N = 64; par.lmax = par.m + 63
```

The m = 1 high-latitude mode comes out at −95.76 nHz with growth rate +11.30 nHz (theirs: −95.70, +11.69); the
table in heat-equation.md §8 has more modes. With `thermal_wind = 3` and `dSdtheta` = Δθs/c_p the same set-up gives the
baroclinic onset of Bekki et al. 2024 (|Δθs| ≈ 425 erg g⁻¹ K⁻¹ with the Edmond Ω).

## References

- Baruteau & Rieutord 2013, JFM 719, 47 (shellular and cylindrical profiles)
- Guenel, Baruteau, Mathis & Rieutord 2016, A&A 589, A22 (conical profile)
- Mirouh, Baruteau, Rieutord & Ballot 2016, JFM 800, 213 (shellular, Boussinesq)
- Bekki, Cameron & Gizon 2022, A&A 662, A16 (solar inertial modes, thermal wind)
- Bekki et al. 2024 (baroclinic instability of the high-latitude mode, prescribed Δθs)
- Mukhopadhyay, Bekki, Zhu & Gizon 2025, A&A, arXiv 2501.16797 (benchmark tables, Edmond Ω data)
- Larson & Schou 2018 (HMI rotation table)
