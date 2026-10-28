# Differential rotation

kore can compute the linear oscillation modes of a star or planet whose rotation rate varies with radius and latitude,
not only of a rigidly rotating one. The background rotation is

$$
\Omega(r,\theta) = \Omega_{\rm ref}\left[1 + \Delta\Omega(r,\theta)\right], \qquad \Delta\Omega = \sum_n f_n(r)\thinspace P_n(\cos\theta), \quad n = 0, 2, 4, \dots
$$

Here $\Omega_{\rm ref}$ is the rotation rate of the frame in which kore solves, and $\Delta\Omega$ is the relative
departure from it, written as a sum of Legendre polynomials in $\cos\theta$ with radial coefficients $f_n(r)$. Only even
$n$ appear, so the rotation is symmetric about the equator.

This page is the user guide: what is modelled, the parameters, the built-in profiles and their limits, and two worked
examples. The derivation and the implementation notes are in [heat-equation.md](heat-equation.md) (thermal terms,
generic $P_n$ couplings, benchmarks, and a [table of notation and code names](heat-equation.md#notation-and-code-names))
and in [thermad0.md](thermad0.md) (runs without thermal diffusion, `ThermaD = 0`). The checks are described in
`tests/diff_rot/README.md` and `tests/heat_dr/README.md`.

## What is modelled

kore solves in a frame rotating at $\Omega_{\rm ref}$. In that frame the differential rotation is a steady azimuthal
flow,

$$
\mathbf{U}_ 0 = G\thinspace \Delta\Omega\thinspace r\sin\theta\thinspace \hat{\mathbf{e}}_ \phi ,
$$

where $G$ = `Gaspard` $= \Omega_{\rm ref}\tau$ is the frame's rotation rate in kore's time unit $\tau$ (1 with
`timescale = 'rotation'`). `Gaspard` only sets the time unit; the frame itself is fixed by the profile, since
$\Delta\Omega$ is measured relative to it (`diff_rot_frame` for the solar profiles). The perturbation $\mathbf u$ is advected and sheared by this flow. In the momentum equation
this adds

$$
G\left[\Delta\Omega\thinspace (i\thinspace m\thinspace \mathbf{u} + 2\hat{\mathbf{z}}\times\mathbf{u}) + r\sin\theta\thinspace (\mathbf{u}\cdot\nabla\Delta\Omega)\thinspace \hat{\mathbf{e}}_ \phi\right].
$$

The first part is the advection by the local rotation and its Coriolis force, the second the transport of angular
momentum by the perturbation across the shear. The density stratification is carried through the anelastic form of the
velocity,

$$
\mathbf{u} = \rho_0^{-1}\nabla\times\nabla\times(\rho_0 P\thinspace \mathbf{r}) + \nabla\times(T\thinspace \mathbf{r}),
$$

with $P$ and $T$ the poloidal and toroidal scalars that kore solves for.

With `thermal = 1` the entropy equation gains up to three terms,

$$
i\thinspace m G\thinspace \Delta\Omega\thinspace s + u_r\thinspace \frac{\partial S_{tw}}{\partial r} + \frac{u_\theta}{r}\thinspace \frac{\partial S_{tw}}{\partial\theta}.
$$

The first is the advection of the entropy perturbation $s$ by $\mathbf U_0$. The other two are the advection of a
latitudinal background entropy $S_{tw}(r,\theta)$ by the perturbation. $S_{tw}$ is either the thermal wind that balances
$\Omega(r,\theta)$ (the latitudinal entropy a differentially rotating star needs to be in equilibrium; Bekki, Cameron &
Gizon 2022, eq. 27) or prescribed. These thermal-wind terms are not small corrections: in a convection zone they act at
the same order as the momentum terms (heat-equation.md §1).

Not modelled:

- the background is fixed: nothing maintains or evolves $\Omega$ and $S_{tw}$ against diffusion;
- $\rho_0$, $T_0$ and $g$ stay spherical (no centrifugal distortion);
- there is no magnetic field.

**Where it lives.** Everything specific to the differential rotation (profiles, thermal wind, couplings, the terms of
each row) is in `bin/diff_rot.py`, built on the angular algebra of `bin/angular.py`; the details are in
[differential_rotation_details.md](differential_rotation_details.md). The rotation profiles do not depend on the
background model ([background.md](background.md)).

## Frame and units

$\Delta\Omega$ is relative to the frame, and $G$ converts it to the time unit, so any `timescale` works. Two things
follow:

- `ViscosD`, `ThermaD` and `Beyonce` must be computed for $\Omega_{\rm ref}$. For `'solar'` the frame must rotate at
  435 nHz; for `'solar_P4'` and `'tabulated'` at `diff_rot_frame` (456 nHz by default). The other types are
  dimensionless.
- With `thermal_wind = 2` the eigenvalues do not depend on the choice of frame. With `thermal_wind = 1` (linear in
  $\Delta\Omega$) they do, at second order in $\Delta\Omega$: 0.1–0.9 nHz for the solar modes of heat-equation.md §6.

Eigenvalues are in the rotating frame of $\Omega_{\rm ref}$, as for uniform rotation.

## Quick start

```python
par.diff_rot           = 1
par.diff_rot_type      = 'conical'   # see the table of profiles below
par.diff_rot_ampl = -0.25            # the factor of ΔΩ: ε for 'conical'
par.thermal_wind       = 1           # only read with thermal = 1; needs Beyonce > 0
```

Then run `submatrices.py`, `assemble.py` and `solve.py` as usual. As for every run, `lmax - m + 1` must be a multiple
of `2*ncpus` (`assemble.py` stops otherwise).

## Parameters

| parameter | default | meaning |
| --- | --- | --- |
| `diff_rot` | 0 | 1 adds the differential-rotation terms |
| `diff_rot_type` | `'Y20'` | the profile, see below |
| `diff_rot_ampl` | 1.0 | the factor of $\Delta\Omega$, for every type (table below) |
| `diff_rot_param` | 0.0 | the shape parameter: $\sigma$ of `'shellular'`, `p` in `'user'` expressions |
| `diff_rot_term_amp` | `[1.0]*9` | $a_0, a_2, \dots, a_{16}$: factors of the $P_0, P_2, \dots, P_{16}$ terms of $\Delta\Omega$, any type, after `diff_rot_ampl` (missing entries are 1); e.g. `[1, 1, 0]` drops the $P_4$ part of `'solar_P4'` |
| `diff_rot_profiles` | `{}` | `'user'`: `{n: 'numpy expression'}`, the shape $f_n$ of each even degree (below) |
| `diff_rot_frame` | 456.0 | `'solar_P4'`, `'tabulated'`: $\Omega_{\rm ref}/2\pi$ in nHz |
| `diff_rot_solar` | `[464, -76, -50, 433.5, 0.71, 0.045]` | `'solar_P4'`: $c_0$, $c_2$, $c_4$, $\Omega_{RZ}$ (nHz), $r_{tc}$, $d_{tc}$ |
| `diff_rot_file` | `''` | `'tabulated'`: the .npz file, absolute or relative to the run directory |
| `diff_rot_nmax` | 8 | `'tabulated'`: highest degree kept, even (up to 16 with `thermal_wind = 2`) |
| `diff_rot_rdeg` | 24 | `'tabulated'`: degree of the Chebyshev fit of each $f_n$ in $r$ |
| `diff_rot_window` | None | `'tabulated'`: None fits over the shell if ricb > 0 and is `'sun'` if ricb = 0; `'sun'`: one fit over the whole Sun, even in $r$, the same for any ricb ([heat-equation.md](heat-equation.md) §9) |
| `diff_rot_sun_deg` | 60 | `'sun'`: degree of that fit |
| `thermal_wind` | 1 | heat-equation terms, see below |
| `thermal_wind_radial` | 1 | 0 drops $u_r\thinspace \partial_r S_{tw}$ |
| `thermal_wind_amplitude` | 1.0 | factor on $S_{tw}$ (`thermal_wind` 1, 2) |
| `dSdtheta` | 0.0 | `thermal_wind = 3`: $\partial_\theta S_{tw}$ = `dSdtheta` $\sin 2\theta$ |
| `drop_tol` | 0 | numerics: drops matrix entries below `drop_tol` (relative) to save memory; 0 is off ([params_default.py](../bin/params_default.py)) |

The background (`structure`, `stratification`, `diffusivity`, `r_out`) has its own page, [background.md](background.md).

## Profiles

$a$ = `diff_rot_ampl`, $p$ = `diff_rot_param`. For the solar types $x$ is the fractional solar radius: $x = r\times$`r_out`
with a solar structure ([background.md](background.md)), $x = r$ otherwise.

| `diff_rot_type` | $\Delta\Omega(r,\theta)$ | degrees | notes |
| --- | --- | --- | --- |
| `'rigid'` | $a$ | $P_0$ | test profile: a uniform offset of the rotation rate, with an exact answer (below); $a = 0$ gives uniform rotation with the diff-rot machinery on (needed for `thermal_wind = 3`) |
| `'Y20'` | $a\thinspace P_2(\cos\theta)$ | $P_2$ | $P_2$ unnormalised, despite the name |
| `'Y20-wall-bounded'` | $a\thinspace (1 - r)(r - r_i)\thinspace P_2(\cos\theta)$, $r_i$ = `ricb` | $P_2$ | not smooth at $r = 0$: use ricb > 0 |
| `'shellular'` | $a\thinspace (r^\sigma - 1)$, $\sigma = p$ | $P_0$ | Baruteau & Rieutord 2013 ($a = 1$); not smooth at $r = 0$ unless $\sigma$ is even and positive |
| `'shellular_boussinesq'` | $\tfrac12 N_0^2\thinspace (1 - r^2)$, $N_0^2 = a$ | $P_0$ | Mirouh et al. 2016; use `thermal_wind = 0` (their background has no thermal wind), structure `'Boussinesq'` and stratification `'linear'` |
| `'cylindrical'` | $\varepsilon\thinspace r^2\sin^2\theta$, $\varepsilon = a$ | $P_0$, $P_2$ | Baruteau & Rieutord 2013; barotropic, $S_{tw} = 0$ |
| `'conical'` | $\varepsilon\sin^2\theta$, $\varepsilon = a$ | $P_0$, $P_2$ | Guenel et al. 2016 |
| `'solar'` | $a\thinspace \times$ helioseismic fit, $P_0 + P_2$, tanh step at $x = 0.71$ of width 0.1 | $P_0$, $P_2$ | relative to 435 nHz; 4.5 nHz too fast at the equator and 12 nHz at the pole compared with the observations (heat-equation.md §6) |
| `'solar_P4'` | $a\thinspace (\Omega/\Omega_{\rm ref} - 1)$, $\Omega$ the fit below | $P_0$, $P_2$, $P_4$ | `diff_rot_solar`, `diff_rot_frame`; defaults from Mukhopadhyay et al. 2026 |
| `'tabulated'` | $a\thinspace (\Omega/\Omega_{\rm ref} - 1)$, $\Omega$ from `diff_rot_file` | $P_0 \dots P_{n_{\max}}$ | below |
| `'user'` | $a\sum_n f_n(r) P_n(\cos\theta)$, $f_n$ from `diff_rot_profiles` | the keys of `diff_rot_profiles` | below |

**`'rigid'`.** The eigenvalues are known exactly: with $\lambda_0$ the `diff_rot = 0` eigenvalue,

$$
\lambda(a, G) = \lambda_0\big(G(1 + a)\big) - i\thinspace m G a .
$$

**`'solar_P4'`.** A rigid radiative zone joined by an erf step at the tachocline $r_{tc}$ (width $d_{tc}$) to a
convection zone whose rotation depends on latitude only:

$$
\frac{\Omega}{2\pi} = \Omega_{RZ} + \tfrac12\left[\Omega_{CZ}(\theta) - \Omega_{RZ}\right]\left[1 + \mathrm{erf}\thinspace \frac{x - r_{tc}}{d_{tc}}\right],
\qquad \Omega_{CZ} = c_0 + c_2\cos^2\theta + c_4\cos^4\theta ,
$$

with `diff_rot_solar` = [$c_0$, $c_2$, $c_4$, $\Omega_{RZ}$, $r_{tc}$, $d_{tc}$] (nHz, and fractions of R☉).

**`'user'`.** `diff_rot_profiles` gives the shape of each even degree as a numpy expression of `r` (kore radius), `x`
(fractional solar radius) and `p` (`diff_rot_param`), with `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `sinh`,
`cosh`, `tanh`, `exp`, `log`, `log10`, `sqrt`, `abs`, `sign`, `pi`, `where`, `minimum`, `maximum`, `erf`, `erfc`. For
example, the cylindrical rotation as a `'user'` profile (it gives the same matrices as `'cylindrical'`):

```python
par.diff_rot_type     = 'user'
par.diff_rot_profiles = {0: '(2/3)*r**2', 2: '-(2/3)*r**2'}
par.diff_rot_ampl     = -0.25
```

The degrees must be even, and the expressions must not contain `#`: `spin_doctor` writes them at the end of each
`params.dat` row, after a `#`. Any degree up to 16 works with `thermal_wind = 2` (32 with `thermal_wind = 1`).

**`'tabulated'`.** The file holds `ome[θ, r]` (nHz), `r` (R☉) and `theta` (rad), the layout of `data/diff_rot.npz`
of the Mukhopadhyay et al. 2025 data set. `tests/diff_rot/ls_to_tabulated.py` converts the Larson &
Schou 2018 table to it; `bin/jsoc_hmi_2010_2026.npz` is the mean of the 83 JSOC HMI 72-day inversions
(`hmi.V_sht_2drls`, NACOEFF 36) from 2010-04-30 to 2026-06-29 in this layout, on the 151 table radii. kore projects $\Omega$ on $P_0 \dots P_{n_{\max}}$ at each table radius, fits each $f_n$ in $r$ over the table radii
within 0.01 of the shell, and prints the fit residuals and the truncation error in nHz (the largest difference
between the table and its expansion up to $P_{n_{\max}}$); check that line. For the Mukhopadhyay et al. 2025 profile the truncation error near the poles at the surface is up to 17 nHz at
`diff_rot_nmax = 8`; 12 is the working choice.

## Heat equation (`thermal = 1`)

`thermal_wind` chooses which entropy terms are added and where the latitudinal background entropy $S_{tw}$ comes from.
The advection of $s$ by the differential rotation, the first term, is always included.

| `thermal_wind` | terms added | $S_{tw}$ |
| --- | --- | --- |
| 0 | $i\thinspace m G\thinspace \Delta\Omega\thinspace s$ | none (e.g. Mirouh et al. 2016) |
| 1 (default) | the three terms | thermal wind of $\Omega$ to first order in $\Delta\Omega$ ($S_{tw} \propto G^2/$`Beyonce`) |
| 2 | the three terms | thermal wind of the full $(1 + \Delta\Omega)^2$ balance; frame independent |
| 3 | the three terms | prescribed: $\partial_\theta S_{tw}$ = `dSdtheta` $\sin 2\theta$, uniform in $r$ (Bekki et al. 2024) |

- `thermal_wind` 1 and 2 need `Beyonce > 0`. With `timescale = 'rotation'`, `set_scales()` sets `Beyonce = 0`, so set
  it after calling `set_scales()`.
- `dSdtheta` is in the units of $s$ ($\delta s/c_p$): `dSdtheta` $= \Delta_\theta s/c_p$, with $\Delta_\theta s$ in erg g⁻¹ K⁻¹. Bekki et al. use
  $c_p$ = 4.17e8, Model S has about 3.46e8. Negative values give warm poles. The thermal-wind amplitude does not apply.
- `thermal_wind_radial = 0` drops $u_r\thinspace \partial_r S_{tw}$, as Bekki et al. 2022 and Mukhopadhyay et al. 2025 do. Keeping it is
  the consistent choice, but it moves the solar columnar and high-frequency retrograde modes by 5–15 nHz.
- With the thermal wind $S_{tw}$ is the only latitudinal entropy gradient; in a convection zone with $dS_0/dr = 0$ it is
  also the only background entropy gradient, and its effect on the modes is of the same order as the momentum terms.
- `ThermaD = 0` works with every option. With `thermal_wind ≥ 1` the heat rows are written as $r^2\thinspace (\mathrm{eq})/p_0$ in a $C^{(1)}$
  basis; see [thermad0.md](thermad0.md) for the caveats (unresolved boundary layer as `ThermaD` $\to 0$, small-scale
  growing modes without diffusion).

## Resolution and cost

- A differential rotation couples harmonic degrees $l$ that uniform rotation keeps apart, so the matrices get wider.
  Let $n_{\max}$ be the highest degree of $\Delta\Omega$. The momentum rows then couple degrees that differ by up to
  $n_{\max}$ ($P \to P$, $T \to T$) or $n_{\max} + 1$ ($P \leftrightarrow T$).
- $S_{tw}$ has degrees up to $n_{\max} + 2$ (`thermal_wind` 1) or $2n_{\max} + 2$ (`thermal_wind` 2), and the heat rows
  couple degrees that far apart. Matrices are therefore widest with `'tabulated'` and a large `diff_rot_nmax`.
- With `ricb = 0` (a full sphere) the profiles are evaluated at $\lvert r\rvert$, to keep them smooth at the centre. `'Y20-wall-bounded'` and `'shellular'` (except even $\sigma > 0$) have a
  kink at $r = 0$ and `submatrices.py` prints a WARNING. The `'solar'` tachocline step needs N ≈ 400 at `ricb = 0`.
- Check convergence with `spin_doctor` as for any run; its residuals include the diff-rot terms.

## Output

`spin_doctor` adds the diff-rot powers to `flow.dat` (`Wdr`, `Wdr_im`, `Ensdr`) and `thermal.dat` (`Wdr_thm`), and
writes the options to `params.dat`, after the general columns (the order is that of `spin_doctor.param_table`):

    diff_rot, diff_rot_ampl, thermal_wind, diff_rot_type, dSdtheta,
    diff_rot_term_amp_0, _2, ..., _16, thermal_wind_amplitude, thermal_wind_radial,
    diff_rot_frame, diff_rot_nmax, diff_rot_rdeg, diff_rot_generic, diff_rot_solar_0 ... _5,
    structure, solar_density_deg, solar_density_window_lo, solar_density_window_hi,
    solar_rz, solar_rz_dSdr, solar_rz_x1, solar_rz_x2, solar_rz_d, solar_rz_core, solar_rz_xos, solar_rz_dos,
    diff_rot_window, diff_rot_hold, solar_rz_delta_cz, solar_rz_delta_sf, drop_tol, diff_rot_param,
    stratification, diffusivity

`diff_rot_type` is written as its index in `diff_rot.TYPES` ('Y20' 0, …, 'rigid' 7, 'solar_P4' 8, 'tabulated' 9,
'user' 10); the background columns are described in [background.md](background.md). A row ends, after a `#` that
`np.loadtxt` skips, with the `'user'` profile definitions and, for `'tabulated'`, `diff_rot_file` with the first 12 hex
digits of its sha256 (a row with neither has no `#`). `diff_rot_window` is 0 for the shell fit and 1 for the whole-Sun
fit (with `diff_rot_rdeg` then the degree used, `diff_rot_sun_deg`). The column `diff_rot_ampl` is the former
`diff_rot_amplitude`, and the removed options `diff_rot_generic` and `diff_rot_hold` are always 0 and nan, so that the
other columns keep their positions; `diff_rot_param`, `stratification` and `diffusivity` are appended at the end. A
`params.dat` written by an older version may have fewer columns; start a new file rather than appending to it. The
former names `diff_rot_amplitude` and `diff_rot_generic`, and `diff_rot_hold` (removed on 2026-10-26,
[differential_rotation_details.md](differential_rotation_details.md) §5.1), stop a run (`diff_rot.check`).

## Examples

**Conical rotation, Guenel et al. 2016.** `ricb = 0.71`, `diff_rot_type = 'conical'`, `diff_rot_ampl = -0.25`,
$E = 10^{-5}$, uniform density (structure `'Boussinesq'`), stress-free, `thermal = 0`, $m = 0$: their DT mode near $\omega \approx 1.82$; kore gives
$\omega = 1.8173$, $\sigma = -0.021$.

**Solar inertial modes, Mukhopadhyay et al. 2025.** The like-for-like set-up of heat-equation.md §8.
`tests/diff_rot/mk_bekki.py` builds it (with keys for $m$, the profile, the thermal-wind options and $\Delta_\theta s$),
`solve_bekki.py` solves it near a target frequency in nHz, and `run_bekki.py` reproduces the five modes of §8. The
parameters it sets:

```python
import units
par.timescale = 'rotation'; par.set_scales()
par.structure = 'old model S prototype'                  # the Model S density (background.md)
par.r_out = 0.985; par.ricb = 0.71/0.985                 # shell 0.71-0.985 R☉
par.stratification = 'odd top hat'                       # with A = aux5 = 0: dS0/dr = 0 (δ = 0)
par.aux1, par.aux2, par.aux3, par.aux4, par.aux5 = 0.0, 0.1, 0.68, 0.06, 0.0
par.thermal = 1; par.bci = par.bco = 0; par.bci_thermal = par.bco_thermal = 1
par.diff_rot = 1; par.diff_rot_type = 'tabulated'; par.diff_rot_file = 'diff_rot.npz'
par.diff_rot_nmax = 12
par.thermal_wind = 2; par.thermal_wind_radial = 0
units.set_physical(par, frame_nHz=456.0, nu=1e12, kappa=1e12)   # Gaspard, Beyonce, ViscosD, ThermaD, diff_rot_frame
par.N = 64; par.lmax = par.m + 63
```

The $m = 1$ high-latitude mode comes out at −95.76 nHz with growth rate +11.30 nHz (theirs: −95.70, +11.69); the
table in heat-equation.md §8 has more modes. With `thermal_wind = 3` and `dSdtheta` $= \Delta_\theta s/c_p$ the same set-up gives the
baroclinic onset of Bekki et al. 2024 ($\lvert\Delta_\theta s\rvert \approx 425$ erg g⁻¹ K⁻¹ with the Mukhopadhyay et al. 2025 $\Omega$).

## References

- Baruteau & Rieutord 2013, JFM 719, 47 (shellular and cylindrical profiles)
- Guenel, Baruteau, Mathis & Rieutord 2016, A&A 589, A22 (conical profile)
- Mirouh, Baruteau, Rieutord & Ballot 2016, JFM 800, 213 (shellular, Boussinesq)
- Bekki, Cameron & Gizon 2022, A&A 662, A16 (solar inertial modes, thermal wind)
- Bekki et al. 2024 (baroclinic instability of the high-latitude mode, prescribed $\Delta_\theta s$)
- Mukhopadhyay, Bekki, Zhu & Gizon 2025, A&A, arXiv 2501.16797 (benchmark tables, their $\Omega$ data)
- Larson & Schou 2018 (HMI rotation table)
