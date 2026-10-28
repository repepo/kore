# Benchmarks

kore reproduces published eigenvalues of differentially rotating fluids, from an incompressible fluid to the solar
convection zone with a radiative interior. This page gives the set-up of each benchmark, the script that runs it, and
the result.

| benchmark | physics | result |
| --- | --- | --- |
| [Baruteau & Rieutord 2013](#baruteau-rieutord-2013) | incompressible, shellular and cylindrical rotation | reproduced |
| [Guenel et al. 2016](#guenel-et-al-2016) | incompressible, conical rotation | reproduced, including an unstable mode |
| [Mirouh et al. 2016](#mirouh-et-al-2016) | Boussinesq, stably stratified, shellular rotation | reproduced |
| [Mukhopadhyay et al. 2025a](#mukhopadhyay-et-al-2025a) | solar convection zone, observed rotation, thermal wind | reproduced, including an unstable mode |
| [Bekki et al. 2024](#bekki-et-al-2024) | onset of the high-latitude instability | reproduced |
| [Mukhopadhyay et al. 2025b](#mukhopadhyay-et-al-2025b) | solar convection zone with a radiative interior | reproduced, except two high-latitude modes |

The scripts are in `tests/diff_rot/`. Each one builds a run directory with a copy of `bin/` and its own
`parameters.py`, runs `submatrices.py` and `assemble.py`, and solves near the published eigenvalue.

**Conventions.** kore's eigenvalue is $\lambda = \sigma - i\omega$ for fields $\propto e^{\lambda t + im\phi}$ in the
frame rotating at $\Omega_{\rm ref}$ ([Anelastic equations](anelastic.md)). The three fluid-dynamics papers work in the
inertial frame, with fields $\propto e^{i\Omega_p t + im\phi}$ (Baruteau & Rieutord, Guenel et al.) or
$e^{(\tau + i\omega)t + im\phi}$ (Mirouh et al.), so their frequency is $\mathrm{Im}\thinspace\lambda - m$ and their
damping rate $\tau = \mathrm{Re}\thinspace\lambda$, in units of $\Omega_{\rm ref}$. The solar papers give frequencies
in nHz in the frame rotating at 456 nHz, negative for retrograde modes, and growth rates in nHz.

## Incompressible and Boussinesq fluids

These three papers use a viscous fluid in a spherical shell with stress-free walls. Their Ekman number is
$E = \nu/(\Omega_{\rm ref}R^2)$, which is `ViscosD` with `timescale = 'rotation'`. The common settings are

```python
par.timescale = 'rotation'; par.set_scales()
par.structure = 'Boussinesq'                 # uniform density
par.bci = par.bco = 0                        # stress-free
par.diff_rot = 1; par.thermal_wind = 0
par.ricb = 0.35                              # their η, unless stated
par.ViscosD = 1e-8                           # their E
```

`tests/diff_rot/lit_bench.py WORKDIR CASE` builds and solves one published mode; `lit_bench.py WORKDIR list` prints the
cases (`br*`, `gu*`, `mi*`). The runs take minutes on 8 ranks; the largest needs 26 GB.

### Baruteau & Rieutord 2013

JFM 719, 47. Shellular rotation $\Omega = \Omega_{\rm ref}(r/R)^\sigma$, with $\Omega_{\rm ref}$ the surface rate, and
cylindrical rotation $\Omega = \Omega_{\rm ref}(1 + \varepsilon s^2)$, with $\Omega_{\rm ref}$ the rate on the axis:

```python
par.thermal = 0
par.diff_rot_type = 'shellular';   par.diff_rot_param = -0.5   # σ: ΔΩ = r^σ - 1
par.diff_rot_type = 'cylindrical'; par.diff_rot_ampl = -0.4    # ε: ΔΩ = ε r² sin²θ
```

**Result.** For the 12 modes compared, the frequencies agree to $2\times10^{-4}\thinspace\Omega_{\rm ref}$ and the
damping rates to 1 %, once kore is resolved (some modes need more resolution than the paper used). Two modes differ in
the damping rate: an $m = 3$ mode in a cluster that does not settle with resolution, and the growing $m = -6$ mode,
whose growth rate is 6 % lower, a shift of 0.4 % in its critical Ekman number. Two modes at $E \le 5\times10^{-9}$ need
about 190 GB and were not computed.

### Guenel et al. 2016

A&A 589, A22. Conical rotation $\Omega = \Omega_{\rm ref}(1 + \varepsilon\sin^2\theta)$, in the frame rotating at the
pole:

```python
par.thermal = 0
par.diff_rot_type = 'conical'; par.diff_rot_ampl = -0.25      # ε
par.ricb = 0.71
```

**Result.** For the 7 modes compared, the frequencies agree to $10^{-4}\thinspace\Omega_{\rm ref}$
($5\times10^{-4}$ for one) and six damping or growth rates to 1 %, including the unstable mode with a corotation layer.
The seventh, at $\eta = 0.10$, sits in a cluster and is 6 % off in damping rate.

### Mirouh et al. 2016

JFM 800, 213. A stably stratified Boussinesq fluid with $N^2$ uniform, $g = r$, the temperature fixed at both walls,
and the rotation $\Omega = \Omega_{\rm ref}\left[1 + \tfrac{N^2}{2}(1 - r^2)\right]$:

```python
par.thermal = 1; par.stratification = 'linear'               # dS/dr = r, so the buoyancy frequency is N² r² with g = r
par.bci_thermal = par.bco_thermal = 0
par.Beyonce = 1.5                                            # their N²
par.ThermaD = 1e-7                                           # their E/P
par.diff_rot_type = 'shellular_boussinesq'; par.diff_rot_ampl = 1.5   # N² again
```

`thermal_wind = 0` is required: their background is balanced by centrifugal buoyancy, which kore does not carry.

**Result.** Of the 14 modes, 13 are identified. Their frequencies agree to $2\times10^{-4}\thinspace\Omega_{\rm ref}$,
and the damping or growth rates of 9 to 1 % and of 3 more to 2–4 %. One mode at $E = 10^{-10}$ needs more resolution
than fits in memory. A growing mode in a dense cluster of corotation modes is not identified.

## The Sun

These papers solve the fully compressible equations about Model S; kore is anelastic. The common set-up: the Model S
density, $\nu = \kappa = 10^{12}$ cm² s⁻¹, an adiabatic convection zone, stress-free walls, the frame rotating at
456 nHz and their rotation profile. "With the thermal wind" means `thermal_wind = 2` without
$u_r\thinspace\partial_r S_{tw}$ (`thermal_wind_radial = 0`), as in the papers.

### Mukhopadhyay et al. 2025a

A&A, arXiv 2501.16797. The convection zone, 0.71–0.985 R☉, with their tabulated rotation, the file `data/diff_rot.npz`
of their data set (doi:10.17617/3.NP26AI):

```python
import units
par.timescale = 'rotation'; par.set_scales()
par.structure = 'old model S prototype'                  # the Model S density
par.r_out = 0.985; par.ricb = 0.71/0.985
par.thermal = 1
par.stratification = 'odd top hat'                       # with aux5 = 0: dS/dr = 0
par.aux1, par.aux2, par.aux3, par.aux4, par.aux5 = 0.0, 0.1, 0.68, 0.06, 0.0
par.bci = par.bco = 0; par.bci_thermal = par.bco_thermal = 1
par.diff_rot = 1; par.diff_rot_type = 'tabulated'; par.diff_rot_file = 'diff_rot.npz'
par.diff_rot_nmax = 12
par.thermal_wind = 2; par.thermal_wind_radial = 0
units.set_physical(par, frame_nHz=456.0, nu=1e12, kappa=1e12)   # Gaspard, Beyonce, ViscosD, ThermaD, diff_rot_frame
par.N = 64; par.lmax = par.m + 63
```

`tests/diff_rot/mk_bekki.py NAME m=… symm=… file=…` builds this run for any $m$ and symmetry, and `solve_bekki.py NAME
FREQ GROWTH` solves it near a frequency and growth rate in nHz. `run_bekki.py DIFF_ROT_NPZ` runs the five modes below
(about a minute each, one core).

**Result.** For the $m = 3$ Rossby modes ($n = 0, 1$), the $m = 3$ columnar mode, the $m = 10$ high-frequency
retrograde mode and the unstable $m = 1$ high-latitude mode, kore agrees with their compressible values to 0.2 nHz in
frequency and 0.4 nHz in growth rate. The high-latitude mode is at −95.76 nHz with growth rate +11.30 nHz (theirs:
−95.70, +11.69). This needs the thermal wind: without it the frequencies are off by up to 11 nHz and the high-latitude
mode is stable.

### Bekki et al. 2024

The high-latitude mode of the previous set-up becomes baroclinically unstable when the poles are warm enough. With
`thermal_wind = 3` the latitudinal entropy gradient is prescribed, $\partial_\theta S_{tw}$ = `dSdtheta`
$\sin 2\theta$, with `dSdtheta` $= \Delta_\theta s/c_p$ (negative: warm poles):

```
tests/diff_rot/mk_bekki.py RUN tw=3 ds=-1000 file=diff_rot.npz m=1 symm=-1
tests/diff_rot/solve_bekki.py RUN -93 5
```

with `ds` = $\Delta_\theta s$ in erg g⁻¹ K⁻¹.

**Result.** The $m = 1$ high-latitude mode becomes unstable at $\lvert\Delta_\theta s\rvert$ = 425 erg g⁻¹ K⁻¹ with
the rotation of Mukhopadhyay et al. 2025, and at 546 with the Larson & Schou 2018 table, for $c_p = 3.46\times10^{8}$.
The published value is about 500. The radial shear of the rotation in the lower convection zone sets the onset: a
rotation profile uniform in radius needs about four times more.

### Mukhopadhyay et al. 2025b

A&A 708, A253. The convection zone extended down to 0.5 R☉ with their stably stratified, rigidly rotating radiative
interior:

```python
par.ricb = 0.5/0.985
par.solar_density_window = 'sun'                                # one Model S fit over the whole Sun
par.stratification = 'solar_rz'; par.diffusivity = 'solar_rz'   # their eqs. (8)-(10)
par.visc0, par.therm0 = 0.01, 0.04                              # ν and κ below 0.70 R☉, relative to the convection zone
par.diff_rot_type = 'solar_P4'                                  # their eqs. (5)-(6)
par.bci_thermal = 1; par.bco_thermal = 0                        # zero entropy flux at the bottom, s = 0 at the top
par.N = 128
```

the rest as above. `tests/diff_rot/mk_rz.py NAME rin=0.5 m=… symm=…` builds it for any inner radius `rin` (in R☉, 0 for
the full sphere). `tests/diff_rot/solar_checks.py DIFF_ROT_NPZ` runs the modes of both Mukhopadhyay papers, with and
without the thermal wind (about 40 minutes on one core).

**Result.** Without the thermal wind, every convection-zone and mixed mode whose counterpart can be identified (8 modes)
agrees with theirs to 0.13 nHz, and the two published growth rates to 0.006 nHz. With the thermal wind the
convection-zone modes are 2–10 nHz off, so their radiative-interior runs appear to have no latitudinal entropy gradient,
unlike those of 2025a. Their two high-latitude convection-zone modes ($m = 1$ and 2) are not matched: without the
thermal wind kore's nearest convection-zone modes are strongly damped, and the paper gives no growth rates to compare.
