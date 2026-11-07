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
| `real_flow.field`, `imag_flow.field` | `solve.py` | the eigenvectors of the flow (or the forced flows), real and imaginary parts of their spectral coefficients, one column per solution; with `thermal = 1` also `real_thermal.field`, `imag_thermal.field` for $s$; with `magnetic = 1` also `real_magnetic.field`, `imag_magnetic.field` for $\mathbf b$, `real_magnetic_ic.field`, `imag_magnetic_ic.field` for the inner core `'conducting, Chebys'`, and `real_magnetic_mantle.field`, `imag_magnetic_mantle.field` for a resolved mantle; with `rotdyn = 1` also `real_rotdyn.field`, `imag_rotdyn.field` for the rotation rates of the mantle and of the inner core and the angle between them ([Rotating solids](rotating-solids.md)). Without the flow (`hydro = 0`) the flow's files are not written |
| `rsf_spectrum.dat`, `rsf_ke.dat` | `solve.py`, with `forcing = 11` and `rsf_frequencies` | the power spectrum of the kinetic energy of the forced flow, appended, one row per frequency ([Convective forcing](forcing.md#reading-the-output)) |
| `timing.dat` | `solve.py` | the run time of the solve, appended |
| `eigenvalues.dat` | `spin_doctor.py` | the eigenvalues, appended |
| `params.dat` | `spin_doctor.py` | the parameters of each solution (below) |
| `flow.dat` | `spin_doctor.py` | the kinetic energy and the checks of each solution (below) |
| `thermal.dat` | `spin_doctor.py`, with `thermal = 1` | the entropy budget of each solution (below) |
| `magnetic.dat` | `spin_doctor.py`, with `magnetic = 1` | the magnetic energies and budgets of each solution (below) |
| `rotdyn.dat` | `spin_doctor.py`, with `rotdyn = 1` | the rotation of the solids and the torques on them, for each solution (below) |

`python3 bin/units.py` prints the eigenvalues of `eigenvalues0.dat` as frequency and growth rate in nHz
([Background model](background.md#units)). Rows of `params.dat`, `flow.dat`, `thermal.dat`, `magnetic.dat` and `rotdyn.dat` with the same index
belong to the same solution. Start a new `params.dat` rather than append to one written by another version of kore,
which may have other columns.

## `flow.dat`

Columns, counted from 1:

| columns | content |
| --- | --- |
| 1–3 | kinetic energy, and its poloidal and toroidal parts (the eigenvectors are normalised to kinetic energy 1); with `centrifugal = 1` column 1 is the kinetic energy of the deformed body, and columns 2–3 are its parts without the terms of the deformation. Without the flow (`hydro = 0`) the flow's columns are 0 or nan, the eigenvectors are normalised to a magnetic energy (`magnetic.dat`, column 1) plus an entropy "energy" (`thermal.dat`, column 1) of 1, and columns 5–7 and 28 describe the field, or the entropy without a field |
| 4 | the power of the viscous force $\int\mathbf u\cdot(\nabla\cdot\boldsymbol\sigma)\thinspace dV$: minus the viscous dissipation when $\mathbf u = 0$ at the walls (column 26 otherwise); with `centrifugal = 1` that of the deformed body |
| 5–7 | the degree $l$ of the largest kinetic energy, the width of the $l$ spectrum, and the ℓ-convergence (the columns Peak ℓ, ℓ-Width and ℓ-Convergence of the `spin_doctor` table) |
| 8–11 | the enstrophy, and the Coriolis, viscous and buoyancy terms of its balance |
| 12–14 | the radius, the degree and the value of the largest toroidal component of the curl of the viscous force (cvf_r, cvf_l, cvfmax) |
| 15–17 | the relative residuals of the enstrophy balance (residens; NaN with `centrifugal = 1`), of the imaginary part of the power balance (residω) and of its real part, the energy residual (residσ) |
| 18–20 | the power of the differential rotation, its imaginary part, and its enstrophy term (0 without it); with `centrifugal = 1` the powers are those of the deformed body |
| 21–23 | the power of the pressure on the fluid at the inner and at the outer wall, $-2\,\mathrm{Re}\oint\rho_0\Pi\thinspace u_r^*\thinspace dA$ with $\Pi = p'/\rho_0$, and the imaginary part of their sum; zero unless $u_r \neq 0$ at a wall (forced boundary flows, `bco = 2`); in residσ and residω |
| 24–25 | the power of the viscous stress on the fluid at the inner and at the outer wall, $2\,\mathrm{Re}\oint\mathbf u^*\cdot\boldsymbol\sigma\cdot\mathbf n\thinspace dA$ |
| 26 | the viscous dissipation $\int\boldsymbol\sigma:\nabla\mathbf u\thinspace dV = \int 2\rho_0\nu\thinspace\mathsf e_d:\mathsf e_d\thinspace dV \geq 0$ ($\mathsf e_d$ the deviatoric rate of strain), from an $(r, \theta)$ grid; column 4 is $\int\mathbf u\cdot(\nabla\cdot\boldsymbol\sigma)\thinspace dV$ = columns 24 + 25 − 26 |
| 27 | the relative residual of that identity (NaN with `centrifugal = 1`, where columns 24–26 keep their spherical form) |
| 28 | in a full sphere, irreg₀: how far the solution is from smooth at the centre ([Regularity at the centre](numerical-method.md#regularity-at-the-centre)); NaN with an inner core |
| 29–38 | for `m = 0`, `symm = 1` and `m = 1`, `symm = -1` (0 otherwise, and with `centrifugal = 1`), real and imaginary parts of: the angular momentum of the fluid, axial or equatorial, and the viscous torques on the mantle and on the inner core, then the magnetic ones ([Rotating solids equations](rotating-solids-equations.md#torques)) |
| 39 | resid𝐋, the relative residual of the fluid's angular momentum balance (NaN where it is not computed, [Rotating solids](rotating-solids.md#the-angular-momentum-check-residl)) |
| 40–41 | with a viscous fluid at a stress-free outer wall (not with `centrifugal = 1`), the power the drag at the wall dissipates and the damping rate it implies, Dtopo and δσtopo, per unit `kappa_cmb` without a drag ([Drag at the outer wall](drag.md#output)); 0 otherwise |

## `thermal.dat`

| columns | content |
| --- | --- |
| 1 | the entropy "energy" $\tfrac12\int p_0\lvert s\rvert^2\thinspace dV$ (of the deformed body with `centrifugal = 1`) |
| 2 | the entropy advection by the background gradient |
| 3 | the thermal diffusion (of the deformed body with `centrifugal = 1`) |
| 4 | the relative residual of the entropy balance (resid𝑠) |
| 5 | the work of the buoyancy force |
| 6 | the energy residual (residσ) |
| 7 | the terms of the differential rotation in the heat equation |

## `magnetic.dat`

With `magnetic = 1` ([Magnetic field](magnetic.md)). The energies are without the factor Hendrik, as in `conductive_IC`;
the definitions are on [Magnetic field equations](magnetic-equations.md#energy-budgets).

| column | content |
| --- | --- |
| 1 | ME, the magnetic energy $\int\tfrac12\lvert\mathbf b\rvert^2 dV$ in the fluid (the column Mag/Kin of the `spin_doctor` table is Hendrik ME/KE, in the units of KE) |
| 2 | Mdfs, the magnetic diffusion term (of the deformed fluid with `centrifugal = 1`) |
| 3 | Indu, the induction work |
| 4 | the relative residual of the induction budget in the fluid (resid𝐛) |
| 5, 6 | Wlor, Wlor_im: the power of the Lorentz force, real and imaginary parts (in residσ and residω) |
| 7 | MEic, the magnetic energy in the inner core |
| 8 | MEext, the energy of the potential field outside the outer wall (beyond `r_surface` with a resolved mantle) |
| 9, 10 | Ohm_f, Ohm_ic: the Ohmic losses in the fluid and in a conducting inner core |
| 11 | the relative residual of the full energy budget (resid𝐄; nan with stress-free walls and `centrifugal = 1`, with thin conducting layers and `mu` other than 1, and with rotating solids unless `vtrq` and `mtrq` are 1) |
| 12 | with `centrifugal = 1`, a thin conducting layer or a resolved mantle: MEc, the part of the magnetic energy that the deformation adds, so that ME + MEc is the energy of the field in the deformed fluid (the column Mag/Kin of the `spin_doctor` table is Hendrik (ME + MEc)/KE; [Centrifugal deformation](centrifugal-equations.md#energy-budgets)); 0 without the deformation |
| 13, 14 | with a thin conducting layer (`mantle` or `innercore` `'TWA'`) or a resolved mantle (`mantle = 'resolved'`) only: E_w, the magnetic energy stored in the thin layers or in the mantle, and Ohm_w, their Ohmic loss |
| 15, 16 | with a thin conducting layer or a resolved mantle only: εTWA, the estimated relative error of the outer and of the inner thin layer (0 where there is none; the `spin_doctor` table shows the larger) |

## `rotdyn.dat`

With `rotdyn = 1` ([Rotating solids](rotating-solids.md)), the real and imaginary parts (columns 1–2, 3–4, ...) of

| columns | content |
| --- | --- |
| 1–6 | the rotation rates of the mantle and of the inner core relative to the frame, and the angle of the inner core relative to the mantle |
| 7–8 | the gravitational torque on the mantle (minus that on the inner core) |
| 9–12 | the viscous and the magnetic torques on the mantle |
| 13–16 | the viscous and the magnetic torques on the inner core |

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
| 108–112 | `centrifugal`, `centrifugal_shape`, `centrifugal_q`, `centrifugal_f` (the flattening of the outer wall, as used), `centrifugal_param` |
| 113 | `hydro` (0: no flow) |
| 114–122 | `mantle`, `c_cmb`, `c1_cmb`, `muw_cmb`, `c_icb`, `c1_icb`, `muw_icb`, `mu`, `B0_cmb` |
| 123–127 | `r_surface`, `N_mantle`, the conductivity of a resolved mantle at the outer wall (as used), `sigma_mantle_param`, its conductance $\int\sigma\,dr$ (the `c1_cmb` of a thin conducting layer of the same conductance) |
| 128–134 | `rotdyn`, `MoIZ_M`, `MoIZ_IC`, `gTorque`, `OmgtauIC`, `vtrq`, `mtrq` |
| 135–141 | `kappa_cmb`, `topo_drag` (1 for `'rekier'`, 0 for `None`), and with `'rekier'` (0 otherwise) `topo_hrms`, `topo_lambda0`, `topo_H`, and the buoyancy frequency and the frequency of the flow that the drag used, over the rotation rate ([Drag at the outer wall](drag.md)) |

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
  `background.INNER_CORES` (`'insulator'` 0, `'conducting, Chebys'` 1, `'TWA'` 2, `'conducting, Bessel'` 3), `mantle` in `background.MAGNETIC_WALLS`
  (`'insulator'` 0, `'TWA'` 1, `'resolved'` 2), `B0_ic` in `background.B0_ICS` (`'potential'` 0, `'frozen currents'` 1, formerly
  `'full sphere'`), `B0_cmb` in `background.B0_CMBS` (`'wall'` 0, `'plain'` 1) and `ferraro_policy` in
  `background.FERRARO_POLICIES` (`'warn'` 0, `'refuse'` 1). `muw_cmb` and `muw_icb` may be `inf`. `cnorm` and
  `cnorm_tor` are the numbers used. `B0_l` is the lowest degree for `'general'` and 1 for `'ferraro'`; `B0_tor_l` is the
  parameter, which `B0_tor = 'general'` ignores. The expressions of `'general'` (`B0_profiles`, `B0_tor_profiles`), of
  `'ferraro'` (`B0_ferraro_G`), of `diff_rot_type = 'ferraro'` (`diff_rot_F`) and of a resolved mantle (`sigma_mantle`)
  go into the note after `#`. Columns 125 and 127 are 0 without a resolved mantle.
- `centrifugal_shape` is the index in `centrifugal.SHAPES` (`'clairaut'` 0, `'uniform'` 1, `'user'` 2, `'spheroid'` 3).
  Column 111 is the flattening of the outer wall that the run used, with any shape (0 with `centrifugal = 0`).
- `diff_rot_generic` and `diff_rot_hold` are not used (always 0 and nan); they keep the positions of the columns after
  them.
- A row ends, after a `#` that `np.loadtxt` skips, with the `'user'`, `'general'` and `'ferraro'` expressions, that of
  `sigma_mantle` and, for `'tabulated'`, `diff_rot_file` with the first 12 hex digits of its sha256. A row with none of
  these has no `#`.
- A `params.dat` written before the columns 108–112, column 113, the columns 114–122, 123–127, 128–134 or 135–141 were
  added has fewer columns: start a new file. The same holds for `flow.dat` before its columns 29–39 or 40–41.
