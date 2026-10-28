# Output files

A run writes all its files into the run directory ([Getting started](getting-started.md)). `solve.py` overwrites its
files on every run, and `spin_doctor.py` appends one row per solution to its own, so that a series of runs builds up a
table.

| file | written by | content |
| --- | --- | --- |
| `*.mtx` | `submatrices.py` | one sparse matrix per radial operator |
| `A.npz`, `B.npz` | `assemble.py` | the matrices of the eigenproblem $A\mathbf x = \lambda B\mathbf x$ |
| `eigenvalues0.dat` | `solve.py` | the eigenvalues of the last solve, one per row: $\mathrm{Re}\thinspace\lambda$, $\mathrm{Im}\thinspace\lambda$ |
| `real_flow.field`, `imag_flow.field` | `solve.py` | the eigenvectors of the flow, real and imaginary parts of their spectral coefficients, one column per solution; with `thermal = 1` also `real_thermal.field`, `imag_thermal.field` for $s$ |
| `timing.dat` | `solve.py` | the run time of the solve, appended |
| `eigenvalues.dat` | `spin_doctor.py` | the eigenvalues, appended |
| `params.dat` | `spin_doctor.py` | the parameters of each solution (below) |
| `flow.dat` | `spin_doctor.py` | the kinetic energy and the checks of each solution (below) |
| `thermal.dat` | `spin_doctor.py`, with `thermal = 1` | the entropy budget of each solution (below) |

`python3 bin/units.py` prints the eigenvalues of `eigenvalues0.dat` as frequency and growth rate in nHz
([Background model](background.md#units)). Rows of `params.dat`, `flow.dat` and `thermal.dat` with the same index
belong to the same solution. Start a new `params.dat` rather than append to one written by another version of kore,
which may have other columns.

## `flow.dat`

Columns, counted from 1:

| columns | content |
| --- | --- |
| 1–3 | kinetic energy, and its poloidal and toroidal parts (the eigenvectors are normalised to kinetic energy 1) |
| 4 | the viscous dissipation |
| 5–7 | the degree $l$ of the largest kinetic energy, the width of the $l$ spectrum, and the ℓ-convergence (the columns Peak ℓ, ℓ-Width and ℓ-Convergence of the `spin_doctor` table) |
| 8–11 | the enstrophy, and the Coriolis, viscous and buoyancy terms of its balance |
| 12–14 | the radius, the degree and the value of the largest toroidal component of the curl of the viscous force (cvf_r, cvf_l, cvfmax) |
| 15–17 | the relative residuals of the enstrophy balance (residens), of the imaginary part of the power balance (residω) and of its real part, the energy residual (residσ) |
| 18–20 | the power of the differential rotation, its imaginary part, and its enstrophy term (0 without it) |

## `thermal.dat`

| columns | content |
| --- | --- |
| 1 | the entropy "energy" $\tfrac12\int p_0\lvert s\rvert^2\thinspace dV$ |
| 2 | the entropy advection by the background gradient |
| 3 | the thermal diffusion |
| 4 | the relative residual of the entropy balance (resid𝑠) |
| 5 | the work of the buoyancy force |
| 6 | the energy residual (residσ) |
| 7 | the terms of the differential rotation in the heat equation |

## `params.dat`

The columns follow the order of `spin_doctor.param_table`, counted from 1:

| columns | parameters |
| --- | --- |
| 1–10 | `thermal`, `m`, `symm`, `ricb`, `bci`, `bco`, `forcing`, `forcing_frequency`, `forcing_amplitude_cmb`, `forcing_amplitude_icb` |
| 11–18 | `Gaspard`, `Beyonce`, `ViscosD`, `ThermaD`, `ncpus`, `N`, `lmax`, the run time (solve and post-processing, in seconds) |
| 19–36 | `r_out`, `aux1`…`aux5`, `visc0`, `hvisc`, `rvisc`, `rpower_u`, `rhopower_u`, `rpower_v`, `rhopower_v`, `rpower_pp`, `rhopower_pp`, `therm0`, `visc_pow`, `therm_pow` |
| 37–41 | `diff_rot`, `diff_rot_ampl`, `thermal_wind`, `diff_rot_type`, `dSdtheta` |
| 42–50 | `diff_rot_term_amp` $a_0, a_2, \dots, a_{16}$ |
| 51–56 | `thermal_wind_amplitude`, `thermal_wind_radial`, `diff_rot_frame`, `diff_rot_nmax`, `diff_rot_rdeg`, `diff_rot_generic` |
| 57–62 | `diff_rot_solar` |
| 63–66 | `structure`, the degree of the Model S fit, the window of the `'old model S prototype'` fits |
| 67–74 | `solar_rz` (1 when the stratification is `'solar_rz'`), `solar_rz_dSdr`, `solar_rz_steps`, `solar_rz_core`, `solar_rz_os` |
| 75–80 | `diff_rot_window`, `diff_rot_hold`, `solar_rz_delta_cz`, `solar_rz_delta_sf`, `drop_tol`, `diff_rot_param` |
| 81–82 | `stratification`, `diffusivity` |

- Names are written as their index in the lists of the code: `structure` in `background.STRUCTURES` (`'8-term fit'` 0,
  `'old model S prototype'` 1, `'model S hydrostatic'` 2, `'Wu 2005'` 3, `'Boussinesq'` 4), `stratification` in
  `background.STRATIFICATIONS` (`'top hat'` 0, `'odd top hat'` 1, `'solar_rz'` 2, `'linear'` 3, `'inverse square'` 4,
  `'user'` 5; −1 without a stratification), `diffusivity` in `background.DIFFUSIVITIES` (`'uniform'` 0, `'step'` 1,
  `'solar_rz'` 2, `'user'` 3) and `diff_rot_type` in `diff_rot.TYPES` (`'Y20'` 0, …, `'rigid'` 7, `'solar_P4'` 8,
  `'tabulated'` 9, `'user'` 10).
- Column 64 is `structure_deg` for `'model S hydrostatic'`. Columns 65–66 are nan for the default window and
  ∓(`r_out` + 0.005) for `'sun'`.
- `diff_rot_window` is 0 for the shell fit and 1 for the whole-Sun fit, with `diff_rot_rdeg` then the degree used,
  `diff_rot_sun_deg`.
- `diff_rot_generic` and `diff_rot_hold` are not used (always 0 and nan); they keep the positions of the columns after
  them.
- A row ends, after a `#` that `np.loadtxt` skips, with the `'user'` expressions and, for `'tabulated'`,
  `diff_rot_file` with the first 12 hex digits of its sha256. A row with neither has no `#`.
