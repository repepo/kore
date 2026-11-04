# Output files

A run writes all its files into the run directory ([Getting started](getting-started.md)). `solve.py` overwrites its
files on every run, and `spin_doctor.py` appends one row per solution to its own, so that a series of runs builds up a
table.

| file | written by | content |
| --- | --- | --- |
| `submatrices.npz` | `submatrices.py` | every radial operator (one sparse matrix per label) and the parameters they were built from; `operators.py` stops if a run's parameters differ. With `write_mtx = 1` also one `<label>.mtx` (Matrix Market) per operator, for outside tools |
| `A.npz`, `B.npz` | `assemble.py` | the matrices of the eigenproblem $A\mathbf x = \lambda B\mathbf x$ |
| `B_forced.npz` | `assemble.py`, with `forcing` > 0 | the right-hand sides of a forced run, one per column |
| `eigenvalues0.dat` | `solve.py` | the eigenvalues of the last solve, one per row: $\mathrm{Re}\thinspace\lambda$, $\mathrm{Im}\thinspace\lambda$ |
| `real_flow.field`, `imag_flow.field` | `solve.py` | the eigenvectors of the flow (or the forced flows), real and imaginary parts of their spectral coefficients, one column per solution; with `thermal = 1` also `real_thermal.field`, `imag_thermal.field` for $s$; with `magnetic = 1` also `real_magnetic.field`, `imag_magnetic.field` for $\mathbf b$, and `real_magnetic_ic.field`, `imag_magnetic_ic.field` for a conducting inner core |
| `rsf_spectrum.dat`, `rsf_ke.dat` | `solve.py`, with `forcing = 11` and `rsf_frequencies` | the power spectrum of the kinetic energy of the forced flow, appended, one row per frequency ([Convective forcing](forcing.md#reading-the-output)) |
| `timing.dat` | `solve.py` | the run time of the solve, appended |
| `eigenvalues.dat` | `spin_doctor.py` | the eigenvalues, appended |
| `params.dat` | `spin_doctor.py` | the parameters of each solution (below) |
| `flow.dat` | `spin_doctor.py` | the kinetic energy and the checks of each solution (below) |
| `thermal.dat` | `spin_doctor.py`, with `thermal = 1` | the entropy budget of each solution (below) |
| `magnetic.dat` | `spin_doctor.py`, with `magnetic = 1` | the magnetic energies and budgets of each solution (below) |

`python3 bin/units.py` prints the eigenvalues of `eigenvalues0.dat` as frequency and growth rate in nHz
([Background model](background.md#units)). Rows of `params.dat`, `flow.dat`, `thermal.dat` and `magnetic.dat` with the same index
belong to the same solution. Start a new `params.dat` rather than append to one written by another version of kore,
which may have other columns.

## `flow.dat`

Columns, counted from 1:

| columns | content |
| --- | --- |
| 1–3 | kinetic energy, and its poloidal and toroidal parts (the eigenvectors are normalised to kinetic energy 1) |
| 4 | the power of the viscous force $\int\mathbf u\cdot(\nabla\cdot\boldsymbol\sigma)\thinspace dV$: minus the viscous dissipation when $\mathbf u = 0$ at the walls (column 26 otherwise) |
| 5–7 | the degree $l$ of the largest kinetic energy, the width of the $l$ spectrum, and the ℓ-convergence (the columns Peak ℓ, ℓ-Width and ℓ-Convergence of the `spin_doctor` table) |
| 8–11 | the enstrophy, and the Coriolis, viscous and buoyancy terms of its balance |
| 12–14 | the radius, the degree and the value of the largest toroidal component of the curl of the viscous force (cvf_r, cvf_l, cvfmax) |
| 15–17 | the relative residuals of the enstrophy balance (residens), of the imaginary part of the power balance (residω) and of its real part, the energy residual (residσ) |
| 18–20 | the power of the differential rotation, its imaginary part, and its enstrophy term (0 without it) |
| 21–23 | the power of the pressure on the fluid at the inner and at the outer wall, $-2\,\mathrm{Re}\oint\rho_0\Pi\thinspace u_r^*\thinspace dA$ with $\Pi = p'/\rho_0$, and the imaginary part of their sum; zero unless $u_r \neq 0$ at a wall (forced boundary flows, `bco = 2`); in residσ and residω |
| 24–25 | the power of the viscous stress on the fluid at the inner and at the outer wall, $2\,\mathrm{Re}\oint\mathbf u^*\cdot\boldsymbol\sigma\cdot\mathbf n\thinspace dA$ |
| 26 | the viscous dissipation $\int\boldsymbol\sigma:\nabla\mathbf u\thinspace dV = \int 2\rho_0\nu\thinspace\mathsf e_d:\mathsf e_d\thinspace dV \geq 0$ ($\mathsf e_d$ the deviatoric rate of strain), from an $(r, \theta)$ grid; column 4 is $\int\mathbf u\cdot(\nabla\cdot\boldsymbol\sigma)\thinspace dV$ = columns 24 + 25 − 26 |
| 27 | the relative residual of that identity |

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

## `magnetic.dat`

With `magnetic = 1` ([Magnetic field](magnetic.md)). The energies are without the factor Hendrik, as in `conductive_IC`;
the definitions are on [Magnetic field equations](magnetic-equations.md#energy-budgets).

| column | content |
| --- | --- |
| 1 | ME, the magnetic energy $\int\tfrac12\lvert\mathbf b\rvert^2 dV$ in the fluid (the column Mag/Kin of the `spin_doctor` table is ME/KE) |
| 2 | Mdfs, the magnetic diffusion term |
| 3 | Indu, the induction work |
| 4 | the relative residual of the induction budget in the fluid (resid𝐛) |
| 5, 6 | Wlor, Wlor_im: the power of the Lorentz force, real and imaginary parts (in residσ and residω) |
| 7 | MEic, the magnetic energy in the inner core |
| 8 | MEext, the energy of the potential field outside the outer wall |
| 9, 10 | Ohm_f, Ohm_ic: the Ohmic losses in the fluid and in a conducting inner core |
| 11 | the relative residual of the full energy budget (resid𝐄; nan unless the walls are no-slip: `bco = 1`, and `bci = 1` with an inner core) |

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
| 83–95 | `magnetic`, `B0`, `B0_l`, `cnorm`, `Hendrik`, `MagnetD`, `innercore`, `N_cic`, `mu_i2o`, `sigma_i2o`, `B0_tor`, `B0_tor_l`, `cnorm_tor` |
| 96–107 | `B0_wound` (1 if the wound field was added), `B0_wound_eta`, `B0_conf_r1`, `B0_conf_r2`, `B0_conf_p`, `B0_param`, `B0_tor_param`, `B0_ferraro_nmax`, `mag_couplings` (as used: 0 the closed forms, 1 `mag_algebra.py`), `tol_B0`, `B0_ic`, `ferraro_policy` |

- Names are written as their index in the lists of the code: `structure` in `background.STRUCTURES` (`'8-term fit'` 0,
  `'old model S prototype'` 1, `'model S hydrostatic'` 2, `'Wu 2005'` 3, `'Boussinesq'` 4), `stratification` in
  `background.STRATIFICATIONS` (`'top hat'` 0, `'odd top hat'` 1, `'solar_rz'` 2, `'linear'` 3, `'inverse square'` 4,
  `'user'` 5; −1 without a stratification), `diffusivity` in `background.DIFFUSIVITIES` (`'uniform'` 0, `'step'` 1,
  `'solar_rz'` 2, `'user'` 3) and `diff_rot_type` in `diff_rot.TYPES` (`'Y20'` 0, …, `'rigid'` 7, `'solar_P4'` 8,
  `'tabulated'` 9, `'user'` 10, `'ferraro'` 11).
- Column 64 is `structure_deg` for `'model S hydrostatic'`. Columns 65–66 are nan for the default window and
  ∓(`r_out` + 0.005) for `'sun'`.
- `diff_rot_window` is 0 for the shell fit and 1 for the whole-Sun fit, with `diff_rot_rdeg` then the degree used,
  `diff_rot_sun_deg`.
- The magnetic columns: `B0` is the index in `background_field.B0_TYPES` (`'axial'` 0, `'FDM'` 1, `'G21 dipole'` 2,
  `'Luo_S1'` 3, `'Luo_S2'` 4, `'confined'` 5, `'general'` 6, `'ferraro'` 7; −1 without a poloidal field), `B0_tor` in
  `background_field.B0_TOR_TYPES` (`'Malkus'` 0, `'wall'` 1, `'user'` 2, `'general'` 3; −1 without), `innercore` in
  `background.INNER_CORES` (`'insulator'` 0, `'conducting, Chebys'` 1), `B0_ic` in `background.B0_ICS` (`'potential'` 0,
  `'full sphere'` 1) and `ferraro_policy` in `background.FERRARO_POLICIES` (`'warn'` 0, `'refuse'` 1). `cnorm` and
  `cnorm_tor` are the numbers used. `B0_l` is the lowest degree for `'general'` and 1 for `'ferraro'`; `B0_tor_l` is the
  parameter, which `B0_tor = 'general'` ignores. The expressions of `'general'` (`B0_profiles`, `B0_tor_profiles`), of
  `'ferraro'` (`B0_ferraro_G`) and of `diff_rot_type = 'ferraro'` (`diff_rot_F`) go into the note after `#`.
- `diff_rot_generic` and `diff_rot_hold` are not used (always 0 and nan); they keep the positions of the columns after
  them.
- A row ends, after a `#` that `np.loadtxt` skips, with the `'user'`, `'general'` and `'ferraro'` expressions and, for
  `'tabulated'`, `diff_rot_file` with the first 12 hex digits of its sha256. A row with none of these has no `#`.
- A `params.dat` written before the columns 96–107 were added has fewer columns: start a new file.
