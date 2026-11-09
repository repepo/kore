# kore, `anelastic_log` branch: guide for Claude

kore computes linear modes (eigenvalue problems) and boundary-forced responses of a rotating, anelastic fluid in a
spherical shell or full sphere, with a spectral method: spherical harmonics in angle, Chebyshev/Gegenbauer in radius,
sparse matrices solved with SLEPc/PETSc (MUMPS). This branch adds variable density written with ln ρ₀ (hence "log"),
an entropy equation, a solar background, and differential rotation with its thermal wind.

Written 2026-10-13 against commit `b4598010` ("Radiative interior (Mukhopadhyay et al. 2025b) ..."), branch
`anelastic_log` (github.com/repepo/kore), and checked against the code at `a2abc05` on 2026-10-28 (the differential
rotation in `bin/diff_rot.py`, the background chosen by `structure`, `stratification` and `diffusivity`, `diff_rot_hold`
removed, `docs/solver.md`, the GitHub test), and against `80c2f233` on 2026-11-03 (the review of the magnetic port). Line numbers
drift; check the code before relying on a detail here. The commit messages are long and accurate: `git log` is the best
history. §15 lists what the first threads of the claude.ai project behind this branch found and changed.

## 1. Working here

- **Python**: `source ~/z2/bin/activate` (numpy, scipy, mpi4py, petsc4py, slepc4py with complex scalars; sympy for
  `tests/heat_dr/tw_sympy.py`). `bin/` must run on the numpy 1.26 and scipy 1.11 of the GitHub test (§11): kore builds
  its spherical harmonics itself (`angular.py`), and only the tests that compare with scipy's `sph_harm_y`
  (`tests/diff_rot/angular_check.py`, `quad_reference.py`, `tests/heat_dr/row_check.py`) need scipy ≥ 1.15. `pygyre`
  and `astropy` are no longer needed (the file-based models were removed).
- **Memory**: keep numerical tests under 40 GB on the Mac laptop and 100 GB on the Linux server mongo; on chandler
  (48 cores) keep each run under 240 GB. MUMPS is the big consumer; see §9 for out-of-core options.
- **Commits**: leave changes uncommitted unless asked; the owner reviews and commits.
- **Where things run**: every script reads `bin/parameters.py` (imported as `from parameters import par`) and writes
  its output to the current directory; `operators.py` loads `submatrices.npz` from the current directory (2026-11-03:
  formerly every `*.mtx` there) and stops if it was built with other parameters. So run from the
  repo root (as `runKore.sh` does) or from a run directory holding its own `bin/` copy (as the test scripts build).
  `.gitignore` ignores `*.mtx *.npz *.field *.npy *.dat` (except `bin/jsoc_hmi_2010_2026.npz` and
  `tests/bgflow/cic_bgflow_*.npz`), `__pycache__` and `.pytest_cache`.
- **Before trusting a change**: run `ru_insane.py` (operator symmetry, seconds), then the pipeline and `spin_doctor`
  residuals; for changes to the background, the differential rotation or the heat equation, `tests/diff_rot/regress.sh`
  (bitwise A, B against an older checkout) and `tests/heat_dr/run_all.sh`; before a push, `cd tests/ci; pytest -v`
  (the GitHub test, §11). Existing runs are expected to stay bitwise identical unless a change says otherwise.
  `docs/checking.md` ("Modifying kore") is the user-facing guide to these checks. NEW 2026-11-09: `tests/run_suites.sh
  [WORKDIR [SUITE ...]]` runs every suite (or those named) side by side, the longest first, each in WORKDIR/SUITE with
  its log in WORKDIR/logs; with `OLD_KORE` also regress.sh and regress_solve.sh. To split the suites between two machines,
  give each its list (the magnetic suite on one, the others on the other); `MAG_NPAR`, `CF_NPAR`, `OTHER_NPAR` (sheet,
  bgflow, heat_dr: their runs side by side, NPAR in each suite's run_all.sh) and `JOBS` set the jobs; with jobs side
  by side the suites set OMP_NUM_THREADS (and OpenBLAS's, MKL's) to 1 unless set, and Open MPI's binding to none.
- The checked-in `bin/parameters.py` is the "gi1" gravito-inertial mode: ricb 0.35, m 2, thermal, structure 'Wu 2005'
  (r_out 0.99) with stratification 'top hat', N 192, lmax 121, ncpus 10, target 2.04i; expected
  λ = −0.049653 + 2.037155i. `examples/smooth_sun/parameters.py` is the solar example (§6).

## 2. Pipeline (`runKore.sh`)

```
./bin/submatrices.py ncpus          # radial operator blocks -> submatrices.npz  (multiprocessing)
./bin/ru_insane.py                  # optional (not in runKore.sh): operator tests on submatrices.npz, exit 1 on failure
mpiexec -n ncpus ./bin/assemble.py  # A.npz, and B.npz (forcing = 0) or B_forced.npz (forcing > 0; both for 11)
mpiexec -n ncpus ./bin/solve.py [petsc opts]   # SLEPc shift-invert (forcing = 0) or KSP LU (forcing > 0)
./bin/spin_doctor.py ncpus | tee out2          # energy/entropy budgets, residuals, tables
```

`runKore.sh` sets its own `ncpus` (10), which must match `par.ncpus`, and passes its `opts` to `solve.py`. It then
copies `params_default.py` and `parameters.py` to `../run` and moves the outputs there. Constraint enforced by
`assemble.py`: `lmax - m + 1` must be a multiple of `2*ncpus` (number of MPI ranks); the default `lmax` formula in
`params_default.py` satisfies it.

Outputs: `flow.npy` (and `thermal.npy`, `magnetic.npy`, `composition.npy`), complex, one column per solution (NEW
2026-11-08, from conductive_IC: `field_format = 'npy'`, the default; `'text'` writes the former `real_flow.field`,
`imag_flow.field`, ... with savetxt's %.18e, the same numbers to the last bit; writing one format deletes the other's
files; `bin/field_io.py`, read by the tests and tools, and `utils.save_field`, `load_field`, `field_exists`), eigenvectors
normalised to kinetic energy 1 (KE + Hendrik·ME = 1 with `magnetic = 1`; ME + TE + CE
= 1 without the flow, `hydro = 0`); `eigenvalues0.dat` (this solve: Re λ, Im λ); `timing.dat`. `spin_doctor` appends to
`eigenvalues.dat`, `params.dat` (156 columns, and a `#` note on rows that need one, §10), `flow.dat` (48), `thermal.dat`
(8), `composition.dat` (8, with `compositional = 1`), `magnetic.dat` (11; 12 with `centrifugal = 1`, 16 with a thin wall or a resolved mantle). A conducting inner core adds
`magnetic_ic`, a resolved mantle `magnetic_mantle`, `rotdyn = 1` `rotdyn` (ω_M, ω_IC, φ) and
`rotdyn.dat` (16). A `params.dat` from an older version has fewer columns: start a new one. `python3 bin/units.py` prints a run's eigenvalues as ω, σ in nHz.
`solve.py` writes `no_conv_solution` when no eigenpair converged or a forced solution has NaN (NEW 2026-11-08, for
`tools/ramp.sh`).

Parameter file (NEW 2026-11-08, from conductive_IC 08c004a, Ankit Barik): `par.export_json(file)` at the end of
`parameters.py` writes every attribute of `par` exactly (tags `__ndarray__` + dtype, `__numpy__` [dtype, value],
`__complex__`, `__tuple__`, `__dict__` pairs for non-str keys; NaN as JSON NaN), rank 0 only (mpi4py if loaded, else
the launcher's rank variables), through a temporary file and `os.replace`; `par.load_json(file)` sets every name, unknown
ones too (check_names warns), and does not call set_scales. docs/output.md#the-parameter-file.

## 3. File map

`bin/` (≈10.6k lines of Python):

| file | role |
| --- | --- |
| `parameters.py` | the run's parameters: `par = default_params()`, `par.timescale = 'rotation'; par.set_scales()`, then overrides |
| `params_default.py` | every parameter with long comments (the best parameter reference), `set_scales`, `check_names` (warns about names that kore does not define), solver options `petsc_opts`, `prescale`, `drop_tol` |
| `units.py` | physical units for parameter files and scripts (kore itself never imports it): `set_physical` (Gaspard, Beyonce, ViscosD, ThermaD, `diff_rot_frame` from cgs values; with `D` also CompoD, and BeyonceC = Beyonce), `set_target` (`itau`, `rtau` from ω, σ in nHz), `nHz`; as a script, prints a run's eigenvalues in nHz |
| `utils.py` | globals (`N1`, `n`, `sizmat`, `symm1`, `adiab_tw`, `odd_profiles`...), `dr_offd`, label decoding, Chebyshev transforms (`chebco_f`, `funcheb`, `fonzie`, `angine`), Gegenbauer matrices `Dlam`, `Slam`, `Mlam`, parity `remroco`, `ell`, erf profiles (`erf_top_hat`, `erf_top_hat_odd`, `erf_box`), `Ylm`, `load_npz_mmap`, `eccen_tide` (forcing 2's fits), `lifted_flow` (the known flow x of forcings 1, 8), `adiabatic_rows` (`adiab_tw`, `adiab_c`: the rows of s, c without diffusion under a thermal wind or a background flow) |
| `background.py` | the background besides the rotation (§6): the type names `STRUCTURES`, `STRATIFICATIONS`, `DIFFUSIVITIES` (their order is the index written to `params.dat`), `check()` (stops on unknown, removed (`REMOVED`) or impossible settings; `check_magnetic()` for `magnetic = 1`, with the names `MAG_COUPLINGS`, `B0_ICS`, `FERRARO_POLICIES`; `check_composition()` for `compositional = 1`, with `COMP_STRATIFICATIONS`; `check_centrifugal()` for `centrifugal = 1`; `check_twa()`, `check_mantle()` for the thin walls and the resolved mantle, `check_rotdyn()` for the rotating solids, `check_drag()` for the drag at the outer wall, `check_forcing()` for forced runs: a known forcing number (`FORCINGS`; `NOT_PORTED` 3, 4, 5), forcing 12's settings, and `check_tides()` for 1, 2, 8), the resolved mantle's σ (`sigma_mantle`, `dlnsigma_mantle`, `mantle_conductance`), the Ferraro check (`ferraro_mismatch`, `ferraro_report`, `magnetic_diffrot`), dS/dr (`gradS`), dC/dr (`gradC`), ν, κ, D and η (`diffusivity`), the 'user', 'general' and 'ferraro' expressions and their `description()` for `params.dat` |
| `background_field.py` | the imposed magnetic field B0 (`magnetic = 1`, §4): `B0_TYPES` ('axial', 'FDM', 'G21 dipole', 'Luo_S1', 'Luo_S2', 'confined', 'general', 'ferraro', 'dipole'; the polynomial ones in `POLY`, `B0_beta_ic`), `B0_TOR_TYPES`, `symmB0`, `B0_l`, the FDM root `findbeta`, h and its derivatives `h(r, k)` (`jl_k` at ricb = 0), k(r, d), the 'general' interpolants (`gen_profile`), the 'ferraro' projection (`ferraro_profile`, `ferraro_cnorm`), the wound field (`wound_profiles`, `wound_residual`), `cnorm` (`B0_norm`), `cnorm_tor` (with `cnorm_error`, `cnorm_tor_error` for check_magnetic), the profiles `bpoX`, `btoX` and, degree by degree, `pol_profiles`, `tor_profiles`; numpy and scipy only |
| `mag_algebra.py` | the Lorentz and induction couplings for any background degree from the recurrence algebra of `angular.py` (`mag_couplings = 'algebra'`, forced for 'general', 'ferraro' and `B0_wound`): `coefficients`, `labels` (profile codes `hp`/`kt` + degree), `block`, `dl_range` |
| `bgflow.py` | NEW 2026-11-08 (from conductive_IC): the background flow (`bgflow = 1`, §4): U, S, C from `bgflow_file` (an .npz of Chebyshev series) or `bgflow_profiles` (expressions; `load`, `source`, `raw`, `degrees`, `nmax`), the profile codes `wq`, `ws`, `wt`, `wh`, `wd`, `wc`, `we` + degree (`profile_functions`, `odd_codes`), the couplings from the algebra of `angular.py` and `mag_algebra` (`build`, `coefficients` cached per m, `labels`, `dl_range`, `block`), `check` (the stops), `rigid_rate` (the rate of a conducting solid at a wall), `steadiness` (B0 under U), `description` (params.dat); for spin_doctor `flow_grid`, `scalar_grid`; as a script (`bgflow.py nmax [nonlinear]`) the writer of `bgflow_file` from a forcing-12 run (`write`, `newton`) |
| `radial_profiles.py` | one class per structure (`wu2005_profiles`, `eight_term_profiles`, `modelS_profiles`, `modelS_hydrostatic_profiles`, `boussinesq_profiles`, on a base class `structure` that takes dS/dr, ν and κ from `background.py`), the dict `structures` and the instance `prf`; the `*X(r, Dorder)` derivative helpers; `proffdir` (label code → function, the background flow's `w` codes from `bgflow.profile_functions`); `burrito` (the integrand of one operator) |
| `smoothed_sun.py` | analytic even fit of the Model S ln ρ, with g, p, d ln ρ/dr, d ln p/dr (structure '8-term fit'; its g also serves 'old model S prototype') |
| `model_s.at` | Model S table (columns As, c₁, ρ/ρ_c, p/p_c, x) for 'model S hydrostatic' and 'old model S prototype' |
| `jsoc_hmi_2010_2026.npz` | the observed rotation Ω(r, θ), the mean of 83 HMI 72-day inversions (2010–2026), in the layout of `diff_rot_type = 'tabulated'` (`examples/smooth_sun`) |
| `submatrices.py` | builds the list of labels (with `diff_rot.labels()`), computes each unique operator in parallel, writes them all to `submatrices.npz` (`submatrix_io.write`, with the parameters; `<label>.mtx` too with `write_mtx = 1`); calls `par.check_names()` and `diff_rot.check()` |
| `submatrix_io.py` | the operator file `submatrices.npz`: `write`, `read`, `check` (stops when the run's parameters differ from those the operators were built with, all but `OPERATOR_INDEPENDENT` and `rsf_*`, and the sha256 of `bgflow_file`); numpy and scipy only |
| `operators.py` | loads `submatrices.npz` (after `submatrix_io.check`), each operator as a global named by its label; physics functions `inertia`, `coriolis`, `viscous_diffusion`, `buoyancy`, `differential_rotation`, `entropy`, `thermal_advection`, `thermal_diffusion`, `thermal_diffrot` (the diff-rot ones return `diff_rot.momentum_block`, `heat_block`), for `compositional = 1` `comp_buoyancy`, `composition`, `comp_advection`, `comp_diffusion`, `comp_diffrot` (`diff_rot.comp_block`), for `magnetic = 1` `lorentz`, `b`, `induction`, `magnetic_diffusion`, and for `centrifugal = 1` `centrifugal_inertia`, `centrifugal_coriolis`, `centrifugal_entropy`, `centrifugal_magnetic` (they return `centrifugal.momentum_block`, `entropy_block`, `magnetic_block`), and for `bgflow = 1` `background_flow` (`bgflow.block`) |
| `assemble.py` | MPI over l; places the blocks (the diff-rot Δl from `diff_rot.dl_range`), boundary-condition rows (`bc_u_spherical`, `bc_theta_spherical`, the push of B0's current sheet at a slipping wall `bc_u_sheet`; the field's `bc_b_spherical`, with thin walls `bc_b_twa` and `bc_b_slip`, the resolved mantle `bc_b_mantle`), the background flow's blocks `bgflow_blocks` (and the Doppler term of its rate at a conducting solid), forcings 1, 2, 6, 7, 8, 9, 10, 11, 12 (the walls' steady rotation: T_1 = Ω r/2 at each no-slip wall); `lifted_rhs` (forcings 1, 8: b = −A x on the interior rows); normalises A, B by ‖A‖ (eigen); `gather_csr` |
| `reynolds_stress.py` | forcing 11: random Reynolds-stress body force f = ρ₀⁻¹∇·(ρ₀ s), the fluctuation-dissipation partner of the viscous force; `rhs(nsamp, seed)` gives one realisation per column of B_forced (docstring: physics, discretisation, normalisation) |
| `bc_variables.py` | Chebyshev values/derivatives at the walls (`Ta`, `Tb`) and ln ρ₀ and its first two derivatives there (`lha*`, `lhb*`) |
| `solve.py` | reads its own rows of A, B (memory-mapped), optional Ruiz scaling, SLEPc/KSP, KE normalisation, writes fields (`utils.save_field`) |
| `field_io.py` | NEW 2026-11-08 (from conductive_IC): the solution files in either format, `FORMATS`, `files`, `exists`, `save` (deletes the other format's files), `load` (a column, a list of them, or all; None if absent); reads no parameters, so tests and tools use it on any folder |
| `just_induction.py`, `solve_ind.py` | NEW 2026-11-06 (from conductive_IC): kinematic induction, the field a given flow (flow.npy or the text files) induces: (A_bb − λ B_bb) b = −A_bu u from A.npz, B.npz of a magnetic eigen assembly (λ the argument, default i forcing_frequency; in a forced assembly A holds A − i wf B, so λ = i wf needs no B.npz) → A_ind.npz, u_ind.npz; solve_ind.py (MPI, KSP LU) writes real/imag_magnetic(_ic, _mantle).field |
| `rotdyn.py` | NEW 2026-11-06 (from conductive_IC): the rotating solids (`rotdyn = 1`, §4): `a_rows`, `b_rows` (the three rows of A, B and the wall entries), `coupling` (K), `relaxation`, `viscous_factor`, `magnetic_factors` (C_l); for spin_doctor `L_case`, `angular_momentum`, `viscous_torques`, `wall_torques` (resid𝐋, axial and equatorial) |
| `topo.py` | NEW 2026-11-06 (from conductive_IC): the drag at a stress-free outer wall (`kappa_cmb`, `topo_drag = 'rekier'`, §4): `on`, `wall_profiles` (ρ₀, ν at the wall), `ratio` (q = Gaspard/(ViscosD ν(1))), `N_cmb`, `omega0`, `Ik_R`, `kappa` (Rekier et al. 2025 B1 with the 4π², or the test hook `topo_K`), `grid` (the cached Gauss–Legendre wall grid), `couplings` (A_ss, A_st, A_ts, A_tt), `rows` (the drag's entries in the stress rows, the ω_M column with `rotdyn = 1`); for spin_doctor `dissipation` (Dtopo); NEW 2026-11-09 the deformed body: `deformed`, `wall_metric` (h_θ, h_φ, x_n, |N| of the wall), `wall_tensor` (K_eff, which `grid` caches) |
| `utils4pp.py` | post-processing: solution expansion, velocity/curl l-components, per-l workers (`flow_worker`, `thermal_worker`, `composition_worker`), `diff_rot_setup`, `diagnose_composition`, `composition_energy`, `centrifugal_composition_terms` (2026-11-08), `diagnose`, `kinetic_energy`, `identify`, `centrifugal_terms` (the metric parts of KE, the Coriolis power and TE with `centrifugal = 1`), `centrifugal_magnetic_energy`, `centrifugal_magnetic_terms` (with a field: MEc, the metric Lorentz and induction work, the potential field on the deformed walls), `centrifugal_dissipation_terms` (phase 2: the metric parts of the viscous power, the work of ΔΩ and the thermal dissipation); thin walls `twa_budget`, `slip_lapF`, `twa_error`, `slipping_walls`, the resolved mantle `mantle_budget` (2026-11-06); `body_force_terms` (forcings 1, 8: Wf, its enstrophy term and torque, 2026-11-08); `bgflow_background`, `bgflow_setup` (a background flow: Wbg, Wbg_im, Ensbg in the flow worker, Wbg_thm, Wbg_cmp in the thermal and composition workers, 2026-11-08) |
| `spin_doctor.py` | runs the diagnostics per solution, prints the table, writes `params.dat` (`param_table`, `row_note`), `flow.dat`, `thermal.dat`, `composition.dat` (the names of each table are `tools/get_data.py`'s, which `tests/tools` checks), and `track_target`; stops with the file's name when a solution file is missing |
| `ru_insane.py` | operator symmetry and diff-rot block tests (see its docstring) |
| `diff_rot.py` | the differential rotation (§8): the shapes fₙ of every type and their fits, the thermal-wind σₙ, the angular couplings of the momentum and heat terms, the labels, blocks and Δl ranges |
| `centrifugal.py` | the centrifugal deformation (`centrifugal = 1`, §4): the flattening f(r) of each shape (`SHAPES`, `radau`, `shape_functions`, `flattening`), the profile codes `ce0`–`ce2` (`profile_functions`), `g_monopole`, `to_physical` (plots), `report`, the angular couplings (`primitives`, `momentum_coefficients`, `heat_coefficients`) and the terms (`labels`, `dl_range`, `momentum_block`, `entropy_block`); with a field the metric Lorentz force and induction (`magnetic_build`, `magnetic_labels`, `magnetic_dl_range`, `magnetic_block`) and the insulating walls (`wall_coefficients`, `wall_rows`, `wall_potential`); phase 2 the dissipative terms (`diffusion_build`, `diffusion_labels`, `diffusion_orders`, `diffusion_block`, the walls `wall_dissipative`, the ICB `icb_electric`), the differential rotation (`rotation_build`, `rotation_block`) and `deformed` (whether any of it applies); the exact spheroid (section 8: `exact`, `spheroid_axes`, `spheroid_terms`, `spheroid_build`, `spheroid_walls`, `spheroid_block`, `spheroid_primitives`; with a field `spheroid_magnetic_build`, `spheroid_lambda`, `spheroid_field_walls`, `spheroid_potential`, `spheroid_icb`); only `parameters` and `angular` imported at the top |
| `angular.py` | the angular algebra at fixed m: the recurrence matrices X (cos θ) and D (−sin θ ∂θ), the vector operations (g(cos θ)·, ẑ×, sin θ F_θ), Schmidt harmonics at any l (`ylm_grid`, `ylm`); numpy and scipy only |
| `targets.py` | analytic targets (spin-over, Rieutord–Valdettaro attractor) |

Other directories: `tests/ci/` (the test GitHub runs on every push, §11); the other folders of `tests/`, one per
suite, each with a README and (but `tests/diff_rot/`) a `run_all.sh` (§11); `docs/` (mkdocs, public user documentation: `installation.md` (from conductive_IC), `getting-started.md` (a
step-by-step run of Baruteau & Rieutord 2013 fig. 10), the user guide `background.md` (structure, stratification,
diffusivity with ThermaD = 0, composition, rotation and thermal wind, centrifugal deformation), `forcing.md` (forcing 11: a worked m = 2 Rossby-mode scan, set-up,
output), `tidal-forcing.md` (forcing 14: the walls move as a degree-2 tide; the energy of the response; forcing 1, the
wave-like tide; forcing 2, the eccentricity tide of Enceladus), `libration.md` (forcings 7 and 8: the walls librate, or
the frame of the mantle with the Poincaré force; one example in both frames), `couette-flow.md`
(forcing 12: the walls turn steadily, spherical Couette flow; Proudman's solution as the example, figure
`docs/assets/couette-flow.png`), `bgflow.md` (a background flow with a meridional circulation: written from a forcing-12
run, linear or by Newton's method, or given by expressions; the example of waves on the flow of a turning inner core),
`magnetic.md` (choosing the background field B0, its options and caveats), `rotating-solids.md` (`rotdyn`, resid𝐋), `drag.md` (`kappa_cmb`, `topo_drag`: the drag at the outer wall, weak and strong drag, Dtopo), `compatibility.md` (which features combine, which combinations stop at startup, and which ones a suite checks), `solver.md` (bad solutions, memory) and `output.md` (the files and their columns), `benchmarks.md` (set-up and result of each published benchmark), `checking.md` (which check to run after a change,
and how: `regress.sh`, `regress_solve.sh`, the suites), and the mathematical framework
`anelastic.md` (the LBR equations, non-dimensionalization, ThermaD = 0, the composition, the terms of a differential
rotation, the energy balance with the work at the walls, validity),
`numerical-method.md` (unknowns, radial basis, rows, boundary conditions, checks), `differential_rotation_details.md`
(the equations and code of the differential rotation), `centrifugal-equations.md` (the map to the level surfaces, the
covariant equations and their rows, Clairaut's equation, the effective gravity, budgets, plots, accuracy, checks), `magnetic-equations.md` (the magnetic equations, every B0
profile, rows, boundary conditions, Ferraro constructions, wound field, budgets, checks), `rotating-solids-equations.md`
(the solids' equations, the torques, resid𝐋, their energy, checks), `drag-equations.md` (the drag's condition, the
tensor of the topography, the rows, Dtopo and the power through the wall, checks), `bgflow-equations.md` (the background
flow's terms, their rows and labels, the walls, the budgets Wbg, Newton's method, the file) and `reynolds-stress.md` (forcing 11: origin of the force,
Landau-Lifshitz stress, mode energy, rows, spectrum)); `examples/smooth_sun/` (the solar example: full sphere, m 8 HFR mode); `tools/` (submission, plotting (they read flow.npy or the
text files by `utils.load_field`; fixed 2026-11-08: `plot_field.py` draws 'u', 'mf', the forces, the buoyancy of the
entropy and the composition, their curls and 'b' (utils4pp.worker_4plot); `load_solution.py` the inner core with the
fluid's degrees and the field's parity; `plot_rad_con_tor_profile.py` ρ0 |u|² with the anelastic consoidal part, in a full
sphere too, checked against KE), `ell_spectrum.py`, `shear_layer_profile.py`, `plot_poltor.py`,
`kore2sprouts.py`, `m2petsc.py`; NEW 2026-11-08, from conductive_IC, series of runs, docs/output.md#many-runs: `ramp.sh`
(one folder, one step per value, `track_target`), `setpar.py` (writes `par.name = value`: in place of each one-target
assignment, by ast byte spans so comments and other statements stay, else a new last line; warns for Ek, Etherm,
Ecomp, Le, timescale after the last set_scales call), `dodirs.sh`, `dosubs.sh`, `subramp.sh` (PBS, runs ramp.sh),
`reap.sh`, `get_data.py` (pandas, named columns, read with float_precision 'round_trip'), `getresults.py`,
`topo_iterate.sh`; NEW 2026-11-09 `compat_tables.py`: the cells of the two tables of `docs/compatibility.md`, the grid of
feature pairs and the forcings, with their marks, notes and tests, which it writes between the page's markers;
`--check` exits 1 if the page differs), `koreviz/`, `koreviz_mod/` (visualisation); `srunKore.sh` (SLURM).

## 4. Formulation

**Unknowns.** Fixed m and one equatorial symmetry `symm` (±1). Per degree l: poloidal P_l(r), toroidal T_l(r), with
`thermal = 1` the entropy perturbation s_l(r), s = δs/c_p, and with `magnetic = 1` (2026-10-28) the field
b = ∇×∇×(f **r**) + ∇×(g **r**), and with `compositional = 1` (2026-11-08) the composition c_l(r), on the l's of s.

**Magnetic field** (`magnetic = 1`, `docs/magnetic.md`, `docs/magnetic-equations.md`, ported from conductive_IC `39487bbf`): imposed poloidal
B0 = cnorm ∇×∇×(h Y_l0 **r**) ('axial', 'G21 dipole', 'Luo_S1', 'FDM' with `B0_l` 1 or 2, 'Luo_S2', 'dipole'; `B0_ic` for the
full-sphere fields with an inner core; `background_field.py`); ∂b/∂t = ∇×(u×B0) + MagnetD ∇²b
(rows r² r̂· in section `f`, r² r̂·∇× in `g`) and the Lorentz force Hendrik ρ₀⁻¹[(∇×b)×B0 + (∇×B0)×b] in the momentum rows
(labels with `ohr1`). Hendrik = (Gaspard Le)², MagnetD = Gaspard Em (`set_scales`, also timescales 'Alfven' and
'magnetic'). b has the symmetry `ut.bsymm` = symm·symmB0: with an antisymmetric B0 f takes the l's of T and g those of P.
A conducting inner core (`innercore = 'conducting, Chebys'`, `N_cic`, `mu_i2o`, `sigma_i2o`; bci = 0 or 1, phase 6b) adds sections
`fic`, `gic` (Chebyshev on [−ricb, ricb], one parity, `ut.Nic` = N_cic/2 per l) matched at the ICB by `assemble.bc_b_cic`.
A toroidal B0 (step 4): `B0_tor` 'Malkus' (k = r, B0 = s φ̂), 'wall' or 'user' (`B0_tor_profile`), `B0_tor_l` 1 or 2,
`cnorm_tor`, profile `bto`, `operators.lorentz_tor`/`induction_tor` (n = 2: `*_tor2`, Δl up to ±2; closed forms fitted from
`tests/magnetic/derive.py`), alone (`B0 = None`) or with a poloidal B0 of the same symmetry (dipole + n = 2); spin_doctor's Lorentz and induction work then come from the (r, θ) grid
(`utils4pp.magnetic_powers_grid`). Any structure (step 3): the (ln ρ₀)′ terms are in the f, g rows (P columns) and the u rows (`operators.lorentz_rho`), labels
`bpo` × `lho` (with `ohr1` in u), derived in `tests/magnetic/derive.py`; they vanish exactly for 'Boussinesq'. Step 5: the
poloidal B0 of degree 2 (`operators.lorentz_pol2`/`induction_pol2`, tables of label expressions fitted from derive.py; spin_doctor
uses the grid for it), the full sphere (ricb = 0: parity of h (-1)^l, of k (-1)^n via `ut.odd_profiles`, f of parity
`vF = symmB0·vP`, one bc row at the CMB; FDM = j_l(βr) with analytic derivatives `bf.jl_k`) and inviscid runs (ViscosD = 0,
full sphere only: u rows in C^(3), v rows in C^(2) with no bc row, since the Lorentz labels have D3, D2). Uniform η, no
inviscid shell: `background.check_magnetic()` stops otherwise. Forced runs: see below (phase 6a).
Thin conducting walls (2026-11-06, conductive_IC port phase 3): `mantle`, `innercore` = 'TWA' (`c_cmb`, `c1_cmb`, `muw_cmb`,
`c_icb`, `c1_icb`, `muw_icb`, fluid `mu`, `B0_cmb`; `background.check_twa`). `assemble.bc_b_twa`: conductive_IC's
bc_b_thinlayer with its O(h) muw terms, outer rows as there, inner poloidal × −1, toroidal / r, so c = c1 = 0, mu = 1 is
the insulating row (to round-off); `bc_b_slip` (`ut.twa_slips`: stress-free or ViscosD = 0, c1 ≠ 0, a poloidal B0) adds
u×B0 by condition C.I (G′ → G′ − r S_l/η, D F → D F + r² X_l/(L η)) for any poloidal B0 through `angular.Algebra`
(conductive_IC: one degree). `background_field.B0_alpha_cmb`: B0_cmb 'wall' adds α r^l to h (`h` wraps `_h`). A thin wall
co-rotates: ΔΩ uniform in latitude there (`diff_rot.rigid_at`). mu ≠ 1 only with thin walls or a conducting IC at every wall.
Resolved conducting mantle (2026-11-06, conductive_IC port phase 4, after PR #17 by A. Nuyt): `mantle = 'resolved'`
(`r_surface`, `N_mantle`, `sigma_mantle` a number or an expression of r, p = `sigma_mantle_param`; `background.check_mantle`
stops on an unresolved profile, mu ≠ 1, centrifugal, bco = 2). Sections `fm`, `gm` last (after fic, gic), N_mantle
Chebyshev coefficients on [1, r_surface] per l (`ut.nman` per section); rows divided by η = 1/σ: B = L r²σ, A = −MagnetD L r²
∇²_l (g: − r²(ln σ)′ D1 − r (ln σ)′), operators `*_man` built in submatrices.py; `assemble.bc_b_mantle`: f = fm, (rf)′ =
(r fm)′, g = gm, σ(1)(rg)′ = (r gm)′ on the outer rows of f, g and row 0 of fm, gm, r fm′ + (l+1) fm = 0, gm = 0 at
r_surface (row 1); a slipping fluid (`ut.mantle_slips`) adds σ(1)×(the thin wall's G′ term) through `bc_b_slip(...,
factor=σ(1))`. With diff_rot the layer co-rotates with ΔΩ(1) (rigid_at; Doppler −i m G c b). Budget: `upp.mantle_budget`
(E_w, Ohm_w = ∫|∇×b|²/σ, MEext beyond r_surface), exact; INDU_ROT_SOLID includes it.
Rotating solids (2026-11-06, conductive_IC port phase 5a): `rotdyn = 1` (`MoIZ_M`, `MoIZ_IC`, `gTorque`, `OmgtauIC`, `vtrq`,
`mtrq`; `background.check_rotdyn`: hydro, m = 0, symm = 1, ricb > 0, eigen, ViscosD > 0, bci = 1, bco = 1 or 0 (phase
5b), Gaspard > 0, no diff_rot, no centrifugal, toroidal B0 zero at both walls). Three unknowns ω_M, ω_IC, φ after everything (`ut.offset['rot']`
= sizmat − 3): λ I_M ω_M = Γ_M + Kφ, λ I_IC ω_IC = Γ_IC − Kφ, λφ = ω_IC − ω_M − (Gaspard/OmgtauIC)φ, K = gTorque Gaspard²;
rows on rank 0 (`rotdyn.a_rows`, `b_rows`; A = −RHS, B = −(factor of λ) as everywhere). The no-slip rows of T of degree 1
(section v, outer row first) get −ω_M and −ri ω_IC (T = ωr); with bco = 0 the outer row stays stress-free, the drag acts
on the slip T_1 − rω_M (`topo.rows`) and a slipping conducting wall induces through it (`assemble.bc_b_slip`). Viscous torques from T at the walls with ρ0 ν ViscosD,
magnetic −Hendrik ∮ s B0_r b_φ = Σ C_l g_l for any poloidal B0 (`bf.pol_profiles`, cnorm included); the Maxwell stress on
the fluid's side, i.e. on whatever conducts beyond (thin wall, resolved mantle, conducting IC). A rigid rotation of an
axisymmetric B0 induces nothing, so the field rows are unchanged (just_induction leaves the three unknowns out).
Drag at the outer wall (2026-11-06, conductive_IC port phase 5b; `docs/drag.md`, `docs/drag-equations.md`, `bin/topo.py`):
at a stress-free outer wall the stress row becomes σ_t + qK(u_t − v_m) = 0, q = Gaspard/(ViscosD ν(1)), K in units of
ρ_w Ω r0, v_m the mantle's velocity (rotdyn = 1). `kappa_cmb`: a constant K; `topo_drag = 'rekier'` (`topo_hrms`,
`topo_lambda0`, `topo_H`, `topo_N`, `topo_omega`; test hook `topo_K`): K = diag(κ_θθ, κ_φφ)(θ) from Rekier et al. 2025 B1
with the 4π² the printed B1 lacks, and B2 (`topo.kappa`, midpoint rule in α with 2048 points; N = topo_N or that of the
stratification, 0 without buoyancy or unstable, `topo.N_cmb`; w0 = topo_omega or |itau| (forced: |forcing_frequency|) over
Gaspard, `topo.omega0`). `topo.rows` adds r² q Σ_l′(A_ss s + A_st t) to the stress row of section u and r q Σ_l′(A_ts s +
A_tt t) to that of section v (s = P′ + P/r + (ln ρ0)′P, t = T; `topo.couplings` by Gauss–Legendre on the cached
`topo.grid`, max(4(lmax + 2), 2000) nodes, scipy's roots_legendre): a constant K couples nothing, K(θ) couples l to l ± 2
within P and within T and, for m ≠ 0, P_l to T_(l±1). With rotdyn = 1 t_1 = T_1 − rω_M (the column `ut.offset['rot']`); the
drag's stress is then the viscous torque on the mantle (rotdyn's torque from T at the wall; none without a drag).
NEW 2026-11-09, a deformed body (centrifugal = 1): the stress rows hold [Fᵀσ_X N]_t (N = F⁻ᵀr̂ = ∇r), so the drag of the
body's wall enters them and Dtopo through K_eff = (|N|/J)FᵀKF = diag(h_θ κ_θθ(x_n), h_φ κ_φφ(x_n)) on (w_θ, w_φ), exact
for the axisymmetric map; x_n the cosine of the normal's colatitude, N_cmb|N| the buoyancy frequency there (N_cmb with
graviX's monopole when centrifugal_q ≠ 0); first order h_θ = 1 + ε − (3ε + 2η)P2, h_φ = 1 − ε − (ε + 2η)P2,
|N| = 1 − (ε + η)P2, x_n = x − 1.5εx(1 − x²) (ε, η = ce0, ce1 at the wall); spheroid Q = a²x² + c²(1 − x²),
h_θ = aQ^1.5, h_φ = a³Q^0.5, |N| = aQ^0.5, x_n = ax/√Q. κ at x_n per node, never expanded (kinks). A constant K then
couples degrees too (`topo.rows` takes the couplings when `deformed()`). rotdyn still stops with centrifugal.
`background.check_drag`: hydro, ViscosD > 0, bco = 0, Gaspard > 0, not forcings 9, 10, not kappa_cmb with
topo_drag; for 'rekier' no diff_rot, w0 ≠ 0, and topo_omega with a forcing 11 scan (`rsf_frequencies`). The drag and the
solids' parameters are in `submatrix_io.OPERATOR_INDEPENDENT` (rows of assemble.py only).
Forcing 12 (2026-11-08, conductive_IC 0780790; `docs/couette-flow.md`, `tests/couette`): the walls turn steadily about z
at forcing_amplitude_cmb, _icb (units 1/τ), T_1 = Ω r/2 on the no-slip wall's T row (field + c.c.: u_φ = Ω r sinθ), at
wf = 0. `background.check_forcing`: m = 0, symm = 1, wf = 0, ViscosD > 0, no IC rate at ricb = 0, bc 0 or 1 (since
2026-11-09 the rule of every steady forcing), a turning wall no-slip, some wall turning; ThermaD > 0 with thermal (CompoD
> 0 with the composition) from `background.check_transport` (w + mΩ = 0 at m = 0). Conducting solids turn with their walls: a rotation leaves
the axisymmetric B0 alone and the tangential u×B0 = u_φ B0r θ̂ matches, so the field rows are those of walls at rest
(any permeability). A forced rigid rotation (walls at one rate, full sphere, IC under a stress-free wall at rest) gives
b = 0 and O(1) residuals (round-off over round-off), but for resid𝐋 (round-off, 9e-15 in a shell, since its scale got
ViscosD ∫ρ0 r³|T| on 2026-11-09).
Forcings 1, 2, 8 (2026-11-08, conductive_IC port phase 7b; `background.check_tides`, `tests/cic_forcings`; 3, 4, 5 stop
with `background.NOT_PORTED`): 2 = the Enceladus eccentricity tide on the no-slip rows of P_2 (P = i wf A P_fit, ricb
i wf A P_fit at the ICB, (rP)'/r = i wf A (P_fit + dP_fit) at both; `utils.eccen_tide`, Padé fits in ricb − 37/42, keyed
m sign(wf); conductive_IC left out wf and the ricb; 16/21 ≤ ricb < 1, bci = bco = 1, insulating walls). 1 and 8 put a
body force by a known flow x (`utils.lifted_flow`): b = −A x on the interior rows of u, v, h, f, g (`assemble.lifted_rhs`;
boundary rows = the empty top rows of op.inertia/entropy/b, and every other row 0), so u' = u − x meets homogeneous
wall rows exactly. 8: x = the mantle's rigid libration T_1 = i wf A r/2, u8 = u7(A_icb7 = A + A_icb) − x, b8 = b7 (the
ICB's relative libration on its no-slip T row; conductive_IC's force was 2× the Poincaré force; any walls 0/1,
ViscosD = 0 only at ricb = 0). 1: x = ∇(X Y_2^m), P_2 = −(a/4)(r² − ricb⁵/r³)/(1 − ricb⁵) (u_r = −1.5 a at the CMB),
Ogilvie's homogeneous rows on u'; m = 0, 1, 2 ((2, 1) needs ricb > 0: the force is a gradient), Boussinesq, thermal =
magnetic = 0, bc 0/1, no drag; carries Gaspard (conductive_IC's did not). spin_doctor: `upp.body_force_terms` (Wf, its
enstrophy term, its torque Tf) in residσ, residω, residens, resid𝐄, resid𝐋 and P_in, flow.dat 42–43.
Forced runs with a field (2026-11-06, conductive_IC port phase 6a; `docs/magnetic-equations.md#forced-runs`,
`tests/forced_field`): every field row (f, g, fic, gic, fm, gm) gets +1j wf op.b (A − i wf B, B = −op.b there); A is not
normalised (forcing ≠ 0), forcing 11 writes B.npz unnormalised and scans with A − i(w − wf)B. `innercore = 'conducting,
Bessel'` (forced only): the core's field j_l(k r), k² = −i w/(η_i MagnetD), w = wf + m Gaspard ΔΩ(ricb) (`utils.bessel_ic`,
`xdlogjl` = x j_l′/j_l), two inner rows `assemble.bc_b_bessel` (bc_b_cic with the core eliminated), no fic/gic; energy and
Ohmic loss in closed form (`utils4pp.bessel_budget`); stops in eigenproblems, rsf scans and just_induction with λ ≠ i wf
(just_induction with λ ≠ i wf stops in any forced run but forcing 11, the only one that writes B.npz).
Walls the forcing moves radially (`background.moves_radially`: 10; 9, 14, 7 m ≠ 0 with their amplitude; 6 at the ICB;
2 at both walls, with forcing_amplitude_cmb):
`forced_walls` (`background.forced_walls`; None → 'rigid' for forcings 6, 7, 'deforming' for 9, 10, 14) says how the
conducting solid behind moves. 'rigid' (`flows_through`): the solid at rest, the flow's u×B0 continued into it,
`assemble.bc_b_radial` (S_l, X_l of u_r r̂×B0_t, recurrence algebra, on the P columns at the wall) in the TWA rows and the
E_t row of cIC/Bessel/mantle. 'deforming': rows at the wall without the radial flow's u×B0 (as conductive_IC; a slip
along the wall keeps its own, bc_b_slip, and its flux stays in INDU_WALLS); a thick solid
(`background.deforms`: resolved mantle, 'conducting, Chebys') moves inside with v = ∇×∇×(P(r_w) φ_l Y r⃗)
(`deform_profile`: mantle φ = 1/r, purely radial; core φ = ((l+3)x^l − (l+1)x^(l+2))/2, φ(1) = 1, (rφ)′(1) = 0), across B0
continued inside (`background_field.solid_profiles`: poloidal a x^n + b x^−(n+1) in the mantle, a x^n + b x^(n+2) in the
core, C¹; toroidal zero at r_surface, k(ricb) x^n in the core), and its induction fills the solid's rows
(`assemble.deform_rows`: ∂t f += t_l, ∂t g += (q_l − (r s_l)′)/r, × σ in the mantle; `mag_algebra.induction_monomials`;
Chebyshev projection on Nq nodes into C^(2), parity checked in the core). A TWA under 'deforming' with a toroidal B0 ≠ 0 at
the wall (`background.thin_drop`) gets half the toroidal bc_b_radial terms (`tor_half`): the drop of B0_tor to 0 across the
layer, the limit of a thin deforming resolved mantle (78% off without it). Stops (`check_forced_magnetic`): 'deforming'
with the inner wall moving and Bessel; a permeability jump at a wall that moves radially where B0 reaches it
(`b0_at_wall(wall, parts)`: h, h' (parts 'p') or k ('t') not 0 there, to 1e-6 of the largest of all three; h' to 10× the
round-off of a slope of an interpolant of degree n = max(64, 2N), ~1e-13 n² 2/(1 − ricb), where larger: 1e-7 to 3e-6 at
N 800 to 2000; an unresolved profile is off by more), in both pictures: mu = 1 and mu_i2o = 1 (either part); a TWA's
muw = 1 for the poloidal part if c ≠ 0 or the layer conducts at rest ('rigid': its E is driven by u_r r̂ × its own
B_t = (p/μ) B0_t, the rows take the fluid's), for the toroidal part if c1 ≠ 0 ('rigid': same; 'deforming': the layer meets
p/(2μ) of B0_t, not half); muw free otherwise (c = c1 = 0: enters nothing; a deforming sheet, c = 0, carries the fluid's
E' in its frame whatever p, and p enters the rows only through c; a thick layer with c1 = 0 holds no B0_tor) (Andres
2026-11-06, for simplicity: a moving surface where μ jumps moves the magnetization of B0; fixed walls take any μ;
general terms derived, not built); a forced_walls value is
checked in every magnetic run (`check_magnetic`); a TWA or the
resolved mantle with bco/bci = 2 stops (a radial flow nothing prescribes). conductive_IC's cIC under a radial forcing (rows without
u×B0, no motion inside) matches neither picture, so it is not compared; its resolved mantle stops there. Left out in both:
the interface displacement (ξ_r times the jump of ∂_r B0). spin_doctor: Wp includes the Lorentz pressure
(`utils4pp.lorentz_walls`) when `walls_move()` (forcing ≠ 0, bco or bci = 2), also in resid𝐄; the flux of V×B0 at a wall
whose solid moves with the fluid stays in the induction work (INDU_ROT_SOLID; V = u_t at a no-slip wall, plus u_r r̂ at a
TWA under 'deforming' with the mean B0, toroidal halved: TWA 2Re∮(V×B0)·K*, cIC/mantle 2Re∮((V×B0)×b_tor*)·n); a
deforming thick solid's work 2Re∫(v×B0)·(∇×b)* dV (`utils4pp.deform_work`, Gauss grid) joins it (Iwall −= it; without it
resid𝐄 ~1e-2); the 'rigid' flux stays in INDU_WALLS; twa_budget's DF gets radial_lapF ('rigid') or 0.5 radial_lapF(tor_only)
(thin_drop). B0_tor allowed in forced runs (conductive_IC stops it).
B0 'dipole' and a slipping inner core (2026-11-07, conductive_IC port phase 6b; `tests/inner_core`): `B0 = 'dipole'`
(B0_TYPES index 8) h = r⁻²/2 (`_h`: (0.5, −1, 3, −12)[k] r^(−2−k)), l = 1, symmB0 = −1, potential in the fluid; in
B0_CMB_TYPES; ricb > 0 only (check_magnetic); cnorm 'mag_energy' 0.1462474928615371 at ricb 0.35, 'rms_cmb' √3;
`solid_profiles` continues it into a deforming core as a x + b x³ (currents in place of the source); conductive_IC's
eigenvalues to 4e-12 (its r⁻² series is cut at tol_B0). `_check_general`, `_not_finite` (background.py): a 'general'
or B0_tor 'user' profile must be finite at the nodes of its interpolant (r = 0 among them at ricb = 0; only the nodes are
tested), and a cnorm or cnorm_tor that is not finite (also a zero field with 'mag_energy'; for 'ferraro', whose bf.cnorm is
1, `ferraro_cnorm` in magnetic_diffrot, which also stops on h_n interpolants that are not finite, with any cnorm) stops. A fluid slipping along a
conducting inner core (`ut.cic_slips`: Chebys or Bessel, hydro, a poloidal B0, bci = 0; a toroidal u×B0 is normal to the
wall): condition C.I in the core's E_t row through `bc_b_slip(l, 'g', 'inner', row, factor)`, factor −1 in bc_b_cic's g
row (G′ coefficient +1) and +σ_i2o/(1 + xd) in the Bessel g row (G′ coefficient −σ_i2o/(1 + xd)): bc_b_slip adds
sgn·factor·(−r S_l/MagnetD) with sgn = −1 at the inner wall, while bc_b_radial takes the signed G′ coefficient. No f-row
term (the slip's radial induction is in the fluid's equation). resid𝐄 needs nothing new (INDU_WALLS, the 'imperm'
gate; conductive_IC adds a slip_power); without the term it is 7e-5 to 4e-2 in the five slip runs, a wrong sign 5e-3 to
3e-2, and resid𝐛 sees neither. Stops: a conducting core with bci = 2 (hydro); centrifugal = 1 with a slipping Chebys core and a poloidal B0 (the
metric u×B0 is not derived); forcing 7 with m = 0, forcing_amplitude_icb ≠ 0 and bci = 0 along a conducting core or a TWA
with c1 ≠ 0, with a poloidal B0 (`check_forced_magnetic`: the frame term of the librating core, ct·C_icb in the g row, and its work in
resid𝐄, are not built). Forcing 7 with m = 0 takes bci = 0 (any bci at ricb = 0, two rows only with bci = 1 as before):
the stress-free row takes no C_icb (conductive_IC wrote C_icb on it), and assemble.py notes that forcing_amplitude_icb
then has no effect. The C.I note prints Pm |forcing_frequency|/Gaspard in forced runs (it printed itau). rotdyn still
needs bci = 1, as in conductive_IC. Phase 6b stopped a toroidal B0 ≠ 0 at any slipping wall with an insulator or a TWA
with c1 = 0 beyond (as conductive_IC, which also stops behind a conductor); phase 6c (below) puts the push in the stress
rows and keeps the stop for an inviscid fluid. `tor_at_wall`: B0_tor's k alone, 1e-10 of max |k| per degree on
check_rotdyn's 201 points (not b0_at_wall's cut, which the poloidal field sets, nor its try/except); the wound field is
left out, `tor_profiles(wound=False)`: it is 0 at insulating walls and thin layers by construction, to its fit's
round-off, 4e-9 at N 800 in a full sphere. With rotdyn = 1 check_rotdyn stops on k ≠ 0 at either wall whatever the wall.
regress.sh m_tor (both walls) and m_gen (bci) are no-slip there; docs/magnetic-equations.md's inviscid Malkus (1967)
check (insulating mantle) stops. ru_insane's current-free identity skips a toroidal part (B0_tor, the wound field).
Current sheets at slipping walls (2026-11-07, conductive_IC port phase 6c; `tests/sheet`): where B0 ends at a wall with
an insulator or a TWA with c1 = 0 beyond (`background.wall_bare`) in a current sheet on the fluid's side, ΔB0_t =
B0_t(fluid) − μ H0_t(beyond) (`background.sheet_jump`, with cnorm: toroidal {n: k_n(r_w)} where |k| is above 1e-10 of
max |k| per degree, the wound field left out; poloidal {n: a_n}, a_n = ±R/(r D) from `_pol_wall_row`, R the steady f row
of a TWA with c1 = 0 (c, muw, mu; an insulator c = 0, mu = 1), D = 1/μ + n c outside, 1/μ + (n+1) c inside, counted
above max(1e-10, 1e-12 n_i² 2/(1 − ricb)) of the degree's largest |B0| (n_i = max(64, 2N)); none for a field that goes
on beyond, `pol_continues`: 'axial' (imposed from beyond the outer wall, a potential field in an insulating core too)
and 'confined' at both walls, 'dipole' and the B0_ic 'frozen currents' fields at the inner wall; every other field is
one of sources inside the wall, 'general' and 'ferraro' whatever their profiles taken as fields of currents in the
fluid). The sheet K0 = ∓r̂×ΔB0_t (outer/inner) pushes a fluid slipping along the wall (`wall_slips`: bco/bci 0 or 2, or
ViscosD = 0; `sheet_push`) with ∓Hendrik b_r ΔB0_t per area; the tangential stress, viscous and Maxwell, is continuous
through the sheet, so the balance has one form at both walls: ViscosD ρ0 ν [r∂r(u_t/r) + ∇₁u_r/r] = −Hendrik b_r ΔB0_t.
`assemble.bc_u_sheet(l, sec)` adds r² q (section u: outer row 1, inner row 3) and r q (section v: rows 0, 1) times the
S_l, T_l parts of b_r ΔB0_t (`mag_algebra._vcross` on `angular.Algebra`, b_r = L_k f_k/r: the f columns of degrees |k −
l| ≤ nB + 1 at the wall, Tb/Ta, one parity at ricb = 0), q = Hendrik/(ViscosD ρ0 ν) at the wall; `sheet_on` in
assemble.main (tests/sheet patches that line for its no-push runs). bc 2 leaves out the normal part, +Hendrik ΔB0_t·b_t
along the fluid's outward normal (the drop of the magnetic pressure across the sheet, as it leaves out p). The push's
power is in Wv (flow.dat 24–25) and its torque in the viscous torque; the magnetic torque, B0 taken on the fluid's side
(rotdyn.wall_torques), holds the sheet's own, so only their sum is the torque through the wall. residσ, resid𝐄 and
resid𝐋 close with or without the push and do not check it: smoothed sheets and L_z = 0 (m 0, insulating stress-free
walls) do, and `_pol_wall_row` is checked against bc_b_spherical's rows (tests/sheet). Stops (check_magnetic, over the
walls that slip and are bare): a sheet in an inviscid fluid (a full sphere: ViscosD = 0 with ricb > 0 stops earlier; its
message names the likely causes below 1e-4 of max |k| for 'user' or 'general': an inexact zero, an interpolant that does
not resolve it, a profile not smooth at r = 0, e.g. 'r*(1 - r)**2' for degree 1); a sheet with centrifugal = 1; a field
that goes on beyond (not 'confined') where `wall_permeable` (mu ≠ 1, or a TWA with c ≠ 0 and muw ≠ 1) and
b0_at_wall(wall, 'p'). FIX: B0_tor 'user' (`k_user`) drops the Chebyshev coefficients below 1e-13 of the largest, as
'general' (their round-off moved the eigenvalues 1e-8 at N 240 and left residσ at 1e-4 at bci 2). FIX: the interpolants
of 'general', 'ferraro', B0_tor 'user', the wound field's ΔΩ and the resolved mantle's ln σ take their coefficients from
a DCT (`background_field.cheb_interp`): numpy's Chebyshev.interpolate (chebvander recurrence) leaves round-off of about
1e-16 j at degree j, above the 1e-13 cut from N about 450 (erratically), and the derivatives amplified it (h''' of
'general' r**2 off by 4e8 at N 800 in a shell; its eigenvalues (no-slip, Ek 1e-3) 3e-5 off at N 450, 2e-4 at N 800, the
operators dense in r; false sheets). The wound field's ΔΩ (`_wound_rot`, whose slope the source takes) is cut at 1e-13
too (the 'conical' ΔΩ, constant in r, had a slope of 2.5e-9 of itself at N 48, 2e-4 at N 800), and its k at 1e-13
instead of 1e-14 (k'', the highest derivative the operators take, 2e-7 off at the walls at N 800 for 'conical', below
1e-8 now; the cut also drops resolved content below 1e-13, k'' about 4e-6 off for a ΔΩ layer of width 0.02). regress.sh
m_gen (outer), m_fer and m_fdr (both walls; mk_run's bco = bci = 0) hold the poloidal push: against a checkout before 6c
they FAIL, as intended (the stress rows of those walls gain entries on f, and A, B, divided by ‖A‖, change by 1e-5
elsewhere; with the normalisation undone only those rows differ); m_wnd FAILs too, by 8e-13 of the row scale (its
eigenvalues by 4e-12), with the wound field's round-off cut. tests/centrifugal/run_all.sh runs its piped checks with
`bash -o pipefail` (a FAIL was hidden).
2026-10-30: with `diff_rot = 1` the induction gains ∇×(U0×b) (`diff_rot.induction_block`, any ΔΩ, same algebra as the force;
a conducting inner core must co-rotate rigidly, `diff_rot.rigid_ricb`; `background.ferraro_mismatch` reports B0⋅∇Ω and
`par.ferraro_policy` 'warn'/'refuse' (a parameter since 2026-11-03, formerly the constant `background.FERRARO_POLICY`);
spin_doctor's resid𝐄 uses the volume induction work, `utils4pp.INDU_WALLS`). New B0 types
'confined' (h = r^l (r2² - r²)^p or r^l (r - r1)^p (r2 - r)^p, factored evaluation, at |r| with the parity (-1)^(l+k) when
r1 > 0 in a full sphere (FIX 2026-11-03), `B0_conf_p` ≥ 3), 'general' (`B0_profiles` with p = `B0_param`, `B0_tor_profiles`
with p = `B0_tor_param`, any degrees) and 'ferraro' (ψ = G(ΔΩ), `B0_ferraro_G`, odd degrees to `B0_ferraro_nmax`; a toroidal part of even degree
may be added). The reverse, diff_rot_type 'ferraro' (NEW 2026-10-30): ΔΩ = F(ψ) of the given poloidal B0 (`diff_rot_F`, numpy in
psi and p; even degrees to `diff_rot_nmax`, `diff_rot._ferraro_shapes`); refused with B0 'ferraro' (circular), without a
poloidal B0, or with an F(ψ) odd in cos θ (quadrupolar field). B0_wound = 1 (NEW 2026-10-30): adds the steady toroidal field
that ΔΩ winds from the poloidal B0, B0_wound_eta MagnetD (∇² - 1/s²)B_φ = -G s B_p⋅∇ΔΩ, one Chebyshev-collocation ODE per degree
(`background_field.wound_profiles`, added to tor_profiles; forces mag_algebra); `wound_residual` replaces the Ferraro warning;
`tests/magnetic/wound_check.py`. `bin/mag_algebra.py` builds the
Lorentz/induction couplings for any degrees from the recurrence algebra (`mag_couplings = 'algebra'`, forced for
'general'/'ferraro'; `ut.mag_algebra`, `ut.mag_dl`; profile codes `hp`/`kt` + degree); same eigenvalues as the closed forms
to 1e-13. Checks: tests/magnetic `dr_induction_check.py`, `algebra_check.py`, `rigid_check.py`.
'Wu 2005' with r_out 0.99 needs N ≈ 192 at Ek 1e-3 for 1e-10 residuals (row checks in a full sphere: N 288).

**Centrifugal deformation** (`centrifugal = 1`, NEW 2026-11-05, `bin/centrifugal.py`; user guide `docs/background.md`
(Centrifugal deformation), equations and checks `docs/centrifugal-equations.md`): the body flattened at first order in
f(r), the flattening of the level surfaces. Map X = α + ξ, ξ = ε(r)∇(r²P₂/2), ε = −(2/3)f (smooth at r = 0): the walls
and every background profile live on the level surfaces r, and P, T describe w = J F⁻¹u (mass conservation, the
impenetrable and no-slip rows exact). Covariant momentum: M∂ₜw, M = FᵀF/J (B, `operators.centrifugal_inertia`),
2 Gaspard z̃×w, z̃ = F⁻¹ẑ (A, `centrifugal_coriolis`); heat J∂ₜs, J − 1 = ηP₂ (B, `centrifugal_entropy`, label
`h0pss0ce10_D0` or `h0ce10_D0`); the first-order closed forms are in the module docstring. Built with the recurrence
algebra of angular.py and projected like the diff-rot force: Δl = 0, ±2 (same kind), ±1, ±3 (P↔T) (`centrifugal.dl_range`,
`assemble.centrifugal_blocks`). B is complex with centrifugal = 1 (real otherwise, so existing runs stay bitwise equal).
Profile codes `ce0` = ε, `ce1` = rε′ = η, `ce2` = r(rε′)′, even in r. `centrifugal_shape`: 'clairaut' (Radau's form of
Clairaut's equation from the structure's g, `centrifugal.radau`; f_top = 5q_top/(2(η_top + 2)), q_top = q x_top³, x_top =
r_out for 'model S hydrostatic', else 1; `centrifugal_f` rescales it to f at the outer wall), 'uniform' (`centrifugal_f`),
'user' (`centrifugal_profile` in r, x, p, as the other 'user' expressions). `centrifugal_q` > 0 adds the monopole
−(2/3)Ω²r to g in `radial_profiles.graviX` (the buoyancy); `graviX(r, D, False)` is the structure's g, which the thermal
wind keeps (as `diff_rot.sigma_coefficients`). `background.check_centrifugal` stops with forcing ≠ 0, bco or bci = 2
(zero radial stress) with ViscosD > 0, a conducting inner core with mu_i2o ≠ 1, and a viscous full sphere with m ≠ 0 and
rpower_v < 4; `centrifugal.report` warns that thermal_wind 1, 2 keep the sphere's S_tw. Checks: `tests/centrifugal/`
(§11); spin-over of a Maclaurin spheroid Im λ = 1 + f + f²/3 (exact 1 + f + f²/2).
Dissipative terms (phase 2, 2026-11-06; `centrifugal.py` sections 6, 7): the covariant viscous force Fᵀ V(F w/J) = V(w) + δV
with no-slip and stress-free walls (the tangential traction on the level surface), the heat diffusion ∇⋅(κ M⁻¹∇s) with
fixed-entropy and fixed-flux walls (r̂⋅M⁻¹∇s), the magnetic diffusion E = η M∇×(M b̃) in the fluid and a conducting inner
core (with the ICB row of the tangential electric field), and a differential rotation given on the level surfaces
(`diffusion_build`, `wall_dissipative`, `icb_electric`, `rotation_build`; labels `diffusion_labels`, `rotation_labels`,
blocks `operators.centrifugal_diffusion`, `centrifugal_diffusion_ic`, `centrifugal_rotation`; wall rows in
`assemble.bc_u_centrifugal`, `bc_theta_centrifugal`, `bc_b_cic_centrifugal`). With m ≠ 0 the sections v (viscous, P''')
and g (field, F''') are built in the C^(3) basis (`diffusion_orders`), and the rows of g and gic at r³ in a full sphere and
a conducting inner core (`utils.rpower_g`, `rpower_gic`). Every phase-2 change is gated on `centrifugal.deformed()` (f ≠ 0
somewhere): f = 0 keeps the sphere's bases and rows, bitwise centrifugal = 0. deformed() runs before the structure exists
(background.check, utils), so 'clairaut' is decided from centrifugal_f or centrifugal_q. A shift taken from an f = 0 run
also holds the change of basis (1.5e-6 in λ for the nutation at N 160): take shifts between deformed runs (±f).
The exact spheroid ('spheroid', NEW 2026-11-06, `centrifugal.py` section 8): one flattening everywhere, the linear stretch
X = diag(a, a, c)α (c/a = 1 − f, a²c = 1) kept exactly at any f < 1: J = 1, M = diag(a², a², c²), z̃ = ẑ/c, the body's
Laplacian D = ∇²/a² + (1/c² − 1/a²)∂z²; exact inertia, Coriolis, viscous and heat terms and stress-free and fixed-flux rows
(`spheroid_terms`, `spheroid_build`, `spheroid_walls`; kore's own labels without profile, e.g. `u3_D2`, `u3vsc0_D4`,
`h0kps0_D2`; Δl up to ±4 same kind, ±3 P↔T; `spheroid_block` behind `momentum_block`, `diffusion_block`; `exact()` gates
it). Homogeneous only ('Boussinesq', diffusivity 'uniform', centrifugal_q = 0; no ΔΩ: check_centrifugal stops);
the buoyancy g = r is then normal to the level surfaces (Maclaurin). Budgets exact too (`spheroid_primitives` in
`utils4pp.centrifugal_terms`). With a field (step 2, NEW 2026-11-06): b̃ = F⁻¹b, the induction is the sphere's (J = 1);
the Lorentz force ∇×(M b̃)×B̃0 + ∇×(M B̃0)×b̃ less the sphere's (`spheroid_magnetic_build`, mag_algebra's labels, Δl up to
n + 3 through B0 of degree n, routed by `magnetic_coefficients`) and the diffusion's field M∇×(M b̃) (`spheroid_terms`
'mag', labels `f{rx}eta0_D{d}`, also in a conducting inner core, whose ICB row is `spheroid_icb`); insulating walls by the
exact potential field in the spheroidal coordinates of the wall, whose angle is the reference colatitude there:
q_l(b̃) = λ_l s_l(M b̃), t_l(M b̃) = 0, λ_l = (a²/k) d ln R_l/dν at ν0 = c/k, k = √|a² − c²| (R_l = Q_l^m(iν) oblate,
Q_l^m(ν) prolate, P_l^m in an insulating inner core; `spheroid_lambda` by recurrence, `spheroid_field_walls` through
`wall_rows`). Every budget closes to round-off, resid𝐄 too (`spheroid_potential`, the loss ∫ j*⋅M j with j = ∇×(M b̃)).
Checks `tests/centrifugal/spheroid_check.py` (exact Galerkin solvers for viscous, heat and field-decay modes).
With a field (magnetic = 1, phase 1b, 2026-11-05; `centrifugal.py` sections 4, 5): F, G describe the flux b̃ = J F⁻¹b, and
kore's B0 formula gives B̃0 (the body's field is F B̃0/J). Covariant Lorentz force (1/J)[∇×(M b̃)×B̃0 + ∇×(M B̃0)×b̃],
induction ∂ₜb̃ = ∇×(w×B̃0/J); first-order parts δ𝐆, δ𝐄 built with mag_algebra's recurrence algebra
(`centrifugal.magnetic_build`), radial factors the product codes amplitude × B0 profile (`product_code`: `E` = ε, `N` = η,
`C` = rη′, `Q` = r(rη′)′ (`ce3`), then `p`/`t` and the degree; `magnetic_profile_functions`), labels
`magnetic_labels` (submatrices.py), blocks `operators.centrifugal_magnetic`, Δl up to nmax + 3 (`magnetic_dl_range`).
Insulating walls matched on the deformed wall: extra entries in the bc rows on F, F′, G of l ± 1 … l ± 3
(`centrifugal.wall_coefficients`, `wall_rows`, `assemble.bc_b_centrifugal`); a conducting inner core is deformed with the
fluid and, with mu_i2o = 1, needs no new rows for the ideal terms (its magnetic diffusion adds the ICB row, phase 2).

**Velocity** (`utils4pp.cheb2space_pol`): u = ρ₀⁻¹∇×∇×(ρ₀ P **r**) + ∇×(T **r**), so ∇·(ρ₀u) = 0 exactly:
u_r = L P/r, consoidal part P′ + P(1/r + (ln ρ₀)′), L = l(l+1).

**Momentum** (anelastic, LBR form, per unit mass): rows are r̂·∇×∇× (section `u`, rows of P_l) and r̂·∇× (section
`v`, rows of T_l) of the equation, multiplied by r^rpower ρ₀^rhopower (`rpower_u` 5, `rpower_v` 3, `rhopower_u = rhopower_v`
1; the r powers are the smallest keeping every operator polynomial, larger ones only spoil residuals). Forces:
Coriolis 2 Gaspard ẑ×u, viscous ViscosD ρ₀⁻¹∇·(ρ₀ν S) (derivatives of ln ρ₀ up to 4th order), buoyancy
Beyonce g s r̂ (`u2gra0_D0`), differential rotation (§8).

**Heat equation** (section `h`), with p₀ standing for ρ₀T₀:
p₀(∂ₜs + u_r dS/dr) = ThermaD ∇·(κ p₀ ∇s) [+ diff-rot terms, §8]. Rows are r² × equation. With `ThermaD = 0` the rows
are divided by p₀ (r s and −r u_r dS/dr, profile `dSd`), better conditioned where p₀ ~ 1e-7 at the surface; with
`ThermaD = 0` plus diff rot and `thermal_wind ≥ 1` (`ut.adiab_tw`) they are r² × eq / p₀ in a C^(1) basis. No thermal
BC rows without diffusion.

**Composition** (`compositional = 1`, section `c`; NEW 2026-11-08, conductive_IC port phase 8; `docs/anelastic.md`,
`docs/background.md`, Composition): a second scalar carried as s is, ρ₀(∂ₜc + u_r dC/dr) = CompoD ∇·(D ρ₀ ∇c), with its own
buoyancy BeyonceC g c r̂ in the u rows (`operators.comp_buoyancy`), so N² = Beyonce g dS/dr + BeyonceC g dC/dr; a larger c
is lighter, as s. Its rows are weighted by ρ₀ where the entropy's are by p₀ (so 'Boussinesq' runs of c alone and s alone
with the same numbers are identical; anelastic ones only without diffusion). Rows r² × eq in C^(2), bc rows
`b*_compositional` (§7); with CompoD = 0 r × eq / ρ₀ in C^(0), no bc rows, prescale off (as ThermaD = 0). dC/dr is
`comp_stratification` of amplitude 1 (§6), D/CompoD from `diffusivity` (D, `comp0`, `comp_pow`); `set_scales` sets CompoD
from `Ecomp` and BeyonceC = 0, so set BeyonceC after it. It goes wherever s goes: the full sphere (vector parity that of
P), diff rot (the advection i m G ΔΩ c only, no thermal wind of the composition, `diff_rot.comp_block`), the deformed body
(`centrifugal.entropy_block(..., 'c')`, the diffusion's metric terms with the code `Drh`), fields, the solids, the drag
(`topo.N_cmb` adds BeyonceC g dC/dr), forced runs (with CompoD = 0 only where w + mΩ keeps one sign in the fluid, never
0, and no background flow has a meridional circulation: `background.check_transport`, so not forcing 12; forcing 1
needs compositional = 0) and `hydro = 0`. conductive_IC's Boussinesq composition ξ is −c, BV2_comp = BeyonceC, Ecomp =
CompoD; its 'internal' gradient
is 'linear', its 'differential' one 'inverse square' with BeyonceC = BV2_comp ricb/(1 − ricb), its 'two zone' heating the
entropy's stratification 'two zone'; it has no 'two zone' composition (kore's is `comp_stratification = 'two zone'`).

**Background flow** (`bgflow = 1`, `bin/bgflow.py`; NEW 2026-11-08, conductive_IC port phase 9; `docs/bgflow.md`,
`docs/bgflow-equations.md`): a steady axisymmetric flow U = Ro G Σₙ[ρ₀⁻¹∇×∇×(ρ₀ Pₙ Yₙ r) + ∇×(Tₙ Yₙ r)] (Ro =
`bgflow_Ro`, G = Gaspard or 1 when Gaspard = 0, Yₙ = Pₙ(cos θ); T₁ = r is a rigid rotation at rate Ro G), with the entropy
S = Ro Σ σₙ Yₙ and the composition C = Ro Σ γₙ Yₙ that came with it (optional). The scalars are numpy expressions of r
(`bgflow_profiles`, keys 'P2', 'T1', 'S2', 'C2', ...; interpolated at the Chebyshev–Lobatto points, so wall values are
exact) or the Chebyshev series of `bgflow_file`, which `python3 bin/bgflow.py nmax [nonlinear]` writes in the folder of a
forcing-12 run (m = 0): the linear flow per unit Ro, or the nonlinear one at Ro = amplitude/G by Newton's method
(continuation in steps dRo 0.15, the Jacobian is kore's own background matrix; it then holds at that Ro only).
conductive_IC's files are read as they are (uniform density). The terms: (U·∇)u + (u·∇)U = ∇(U·u) − U×ω − u×Ω_U in the
u, v rows, −p₀(U·∇s + u·∇S) in h, −ρ₀(U·∇c + u·∇C) in c, ∇×(U×b) in f, g; each is a radial monomial times an angular
coupling [l′, l] computed exactly with the recurrence algebra (`angular.py`, mag_algebra's vector products), a degree n
reaching |Δl| ≤ n + 1, placed by `assemble.bgflow_blocks`. Labels `{row}{q}w{B}{n}{j}[lho{k}|pss0|dns0]_D{d}`, profile
codes wq, ws, wt (U's components, and Ω_U from them), wh, wd (p₀σₙ, p₀σₙ′), wc, we (ρ₀γₙ, ρ₀γₙ′) + the degree character
(mag_algebra.DIGITS, n ≤ 35); lho{k} is (ln ρ₀)^(k), a derivative. P‴ enters the u rows, so gebasis 3, 2 as with a field
(inviscid r⁵ρ₀², r³ρ₀); without diffusion of s (c) the h (c) rows are r² eq/p₀ (ρ₀) in C^(1) (`ut.adiabatic_rows`). The
background must be symmetric about the equator, must not cross a wall, and must turn rigidly at a conducting solid (an
inner core, a thin wall, the resolved mantle), whose field then gets the Doppler term −i m Ω_w b (`bgflow.rigid_rate`);
with B0, `bgflow.steadiness` reports |∇×(U×B0)|. Stops (`bgflow.check`): `centrifugal = 1`, `rotdyn = 1`, `topo_drag =
'rekier'`, an inviscid shell, a file of another ricb, another density (its poloidal part is ρ₀-dependent) or (nonlinear)
another Ro; the writer stops for a run with a background field (B0, whose induced field the file would leave out) or not of
forcing 12 (m 0, steady). The force puts no torque on the fluid for any m
(r×[...] is a flux), so resid𝐋 holds, axial and equatorial. The operators depend on the file's content: `submatrix_io`
keys its sha256.

**Generalised eigenproblem** A x = λ B x, fields ∝ e^{imφ + λt}: Re λ is the growth rate σ, negative Im λ is prograde
(as `itau` and `forcing_frequency`). In nHz the frequency is ω = −Im λ·F/G and σ = Re λ·F/G (F = `diff_rot_frame`,
G = Gaspard; `units.nHz`, `units.set_target`), so prograde modes have ω > 0 and the retrograde HFR modes ω < 0.
B is −inertia (momentum) and +entropy (heat); in A the momentum rows are Coriolis − viscous − buoyancy + diff rot, the
heat rows −advection + diffusion + diff-rot terms (operators return the A sign). Forced runs (`forcing > 0`) use
A − i wf B as the matrix (wf = `forcing_frequency`) and a boundary right-hand side (forcing 11: random body forces,
`reynolds_stress.py`; forcings 1 and 8: −A x of a known flow, §4). The boundary forcings (2, 6, 7, 9, 10, 12,
14) put their wall values on the right-hand side of the wall rows of `bc_u_spherical`: per degree P at the outer wall is
row 0 and at the inner wall row 2 (T: rows 0 and 1). The tangential rows keep 0 (but forcing 2's, whose tide moves the
walls along themselves): they are the physical conditions for any P (no-slip (rP)'/r + P (ln ρ₀)' = 0, stress-free),
where conductive_IC's rows P' = 0, P'' = 0 need the values of P' and P'' (`tests/forcing/rows_check.py`).

**Scales.** `Gaspard` = Ωτ, `Beyonce` = (N₀τ)², `ViscosD` = ν₀τ/r₀² (the Ekman number when τ = 1/Ω),
`ThermaD` = κ₀τ/r₀²; N² = Beyonce g dS/dr. `set_scales()` resets them by `timescale` ('rotation': Gaspard 1,
Beyonce 0, ViscosD = Ek, ThermaD = Etherm), so set physical values **after** calling it; `units.set_physical` computes
them from cgs values. Eigenvalues are in units of 1/τ, in the frame rotating at Gaspard/τ. kore stays dimensionless:
the only physical unit it reads is the nHz of the solar rotation data.

## 5. Numerics and matrix layout

- **Radial basis.** Chebyshev in x ∈ [−1, 1]: x ↦ r ∈ [ricb, 1] with an inner core (`N1 = N`), x = r ∈ [−1, 1] in a full
  sphere (`ricb = 0`), where parity halves the basis (`N1 = N/2`, use about twice a shell's N; N must be even for any
  ricb, `background.check` stops otherwise).
  Each section is written in a Gegenbauer basis: `u` C^(4), `v` C^(2), `f`, `g` C^(2), `h` C^(2) (C^(0) if ThermaD = 0, C^(1) with
  `adiab_tw`), `c` C^(2) (C^(0) if CompoD = 0); inviscid full sphere: `u` C^(2), `v` C^(1). Products are `ut.Mlam` (multiplication by a profile's
  coefficients), derivatives `ut.Dlam`, basis changes `ut.Slam`.
- **Sizes.** `n = N1*(lmax-m+1)/2`, `sizmat = 2n*hydro + n*thermal + 2n*magnetic + 2 nic + 2 nman + n*compositional + 3 rotdyn`.
  Unknown vector: [P blocks for the poloidal l's][T blocks for the toroidal l's][s blocks, same l's as P][f blocks][g
  blocks][fic][gic][fm][gm][c blocks, same l's as P][ω_M, ω_IC, φ], each block `N1` Chebyshev coefficients (`Nic` for fic,
  gic, `N_mantle` for fm, gm; the last three with `rotdyn = 1`); `ut.offset` gives each section's first row (f, g after s,
  so non-magnetic matrices are unchanged, and c after all the fields, `ut.offset['c']`, so runs without it are unchanged;
  the u rows' buoyancy still writes 2n for section h). `hydro = 0` (NEW 2026-11-06, as conductive_IC: no flow, eigenvalue problems only) drops
  u, v: the other sections start 2n earlier, assemble.py skips the flow's rows and columns, solve.py writes no flow files
  and normalises to ME (+ TE) = 1 without Hendrik, spin_doctor runs the flow's diagnostics on `upp.zero_flow()` (zeros;
  residσ, residω, residens nan, resid𝐄 for any walls, the table without the flow's columns, `identify` on b or s);
  submatrices.py builds the same operators (`hydro` is in `submatrix_io.OPERATOR_INDEPENDENT`). `ut.ell(m, lmax, symm)`
  gives the poloidal l's, toroidal l's, all l's. Each row block starts with its BC rows (u: 4 with an inner core, 2
  without; v: 2/1; h: 2/1 when ThermaD > 0; c: 2/1 when CompoD > 0; f, g: 2, outer then inner wall), made room for by dropping the last rows of
  the operator.
- **Full-sphere parity.** At ricb = 0, `submatrices` keeps one parity of rows and columns (`ut.remroco`) from the
  vector parity (vP = (−1)^m symm for P and s, −vP for T) and each operator's parity (r power, derivative orders,
  odd profiles in `ut.odd_profiles`, counted in both profile slots of a label since 2026-10-28). It prints a WARNING if a profile has wrong-parity coefficients: those parts are
  silently dropped. Profiles are therefore built even or odd: `pdS`, `dSd` as sign(r)·f(|r|), diff-rot profiles and
  'user' diffusivities at |r|. Parity is not regularity (2026-11-04): a degree-n profile must also be r^n times a series
  in r². At ricb = 0 `background.centre_check` (called by `submatrices.py`) warns about missing powers (the modes then
  converge to irregular solutions: 'Y20', 'conical', thermal_wind 3, a 'tabulated' f_n(0) ~ 1e-5 with HMI) and about
  kinks (slow convergence in N); spin_doctor's irreg₀ (`upp.centre_irregularity`, flow.dat column 28) measures each mode,
  max |a_p| (π/N)^p over p < l relative to the field's max: round-off when regular, ~1 for spurious modes, > 1e-6
  counted after the table.
- **Operator labels** (`ut.decode_label`): `sX[fu1D][fu2D]_DY`, e.g. `u2lho1vsc0_D1`. `s` is the section; `X` lowers the r
  power (u: rpower_u − X, v: rpower_v − X, h: 2 − X, or 1 − X for ThermaD = 0 without `adiab_tw`; c: 2 − X, or 1 − X for
  CompoD = 0; f, g: 2 − X without ρ₀;
  inviscid uses 3 − X, 2 − X and ρ₀², ρ₀; `ohr1` as the second profile lowers the ρ₀ power by one); each `fuD` is a 3-letter profile code and its derivative order; `_DY` is the derivative order on
  the unknown. `lh1`, `lh2` with order k mean (ln ρ₀)′·(ln ρ₀)^(k) and (ln ρ₀)″·(ln ρ₀)^(k). `rap.burrito` evaluates
  r^rpower ρ₀^rhopower f₁ f₂ on the Chebyshev nodes; `ut.chebco_f` transforms it (`ut.gimmedachebs`: coefficients
  below max(1e-15, `drop_tol`) of the largest are zeroed).
- **Profile codes** (`rap.proffdir`): `lho` ln ρ₀, `vsc` ν, `gra` g, `pss` p₀, `pdS` p₀ dS/dr, `dSd` dS/dr, `kps` κp₀,
  `eta` η, `bpo` cnorm h (B0; its series are cut at `tol_B0` relative, `ut.gimmedachebs`); the composition's (2026-11-08)
  `dns` ρ₀, `rdC` ρ₀ dC/dr, `dCd` dC/dr, `Drh` Dρ₀ (as `pss`, `pdS`, `dSd`, `kps` for s; `rdC`, `dCd` odd);
  differential rotation `zf`/`zd`/`zs` + degree = aₙfₙ, r(aₙfₙ)′, r²(aₙfₙ)″ (with `diff_rot_ampl`) and thermal wind
  `tw`/`tp` + degree = p₀σₙ, p₀σₙ′, the degree n as one character of 0–9a–z (`diff_rot.codes`, so n ≤ 35). `burrito`
  adds these to `proffdir` from `diff_rot.profile_functions()` on first use.
- **Row scaling.** Every operator in `operators.py` is divided by L² or L³ (L of the row) so that all rows are
  (equation)·r^power/L³; terms with an explicit L from u_r = LP/r use L². `differential_rotation` was missing this until
  `26d595eb` (every diff_rot run before that is wrong).
- **Assembly.** `assemble.py` scatters the l's over MPI ranks, collects COO pieces with `ut.packit` and gathers with
  `Gatherv`; eigenproblem A and B are divided by ‖A‖_F.

## 6. Background model (`structure`, `stratification`, `diffusivity`)

User guide: `docs/background.md`. Three parts, each chosen by name; `background.check()` (called by `radial_profiles.py`
before it builds `prf`) stops on an unknown type and on settings that cannot work.

- **`r_out`** (formerly `aux0`): the outer wall in star radii, so x = r·r_out is the radius in star radii and the unit of
  length is r_out star radii. With a solar structure (`background.SOLAR`) the rotation profiles and the 'user'
  expressions use the same x (`diff_rot.solar_x`); otherwise their x is r. Beyonce = 4.86e4 at r_out 0.985 and 456 nHz.
- **`structure`** (no default): ρ₀, p₀ (= ρ₀T₀), g, d ln ρ₀/dr, d ln p₀/dr, one class of `radial_profiles.py` each.
  `'model S hydrostatic'` (for new solar runs: an even fit of ln ρ of `model_s.at` of degree `structure_deg` over the
  whole Sun, g from its mass, p from dp/dr = −ρg; degree 30 is within 1 % of Model S, 50 within 0.14 %); `'Wu 2005'`
  (ρ₀ = (1 − x²)², β = 2, analytic g and p; r_out < 1 is the "peel" that keeps ρ₀(1) > 0: keep it ≥ 0.99, never below
  0.985); `'Boussinesq'` (ρ₀ = p₀ = 1, g = r: the Boussinesq equations). Legacy, kept so that older parameter files give the same
  results: `'8-term fit'` (`smoothed_sun.py`, ρ up to 48 % off Model S near the surface) and `'old model S prototype'`
  (Chebyshev fits of ln ρ, ln p of `model_s.at` controlled by `solar_density_deg`, `solar_density_window`,
  `solar_density_sun_deg`, with the g of the 8-term fit, so not hydrostatic).
- **`stratification`** (dS/dr for r ≥ 0, used as an odd function of r; no default when `thermal = 1`): `'top hat'`
  (`ut.erf_top_hat`, aux1..aux5 = x1, w1, x2, w2, A in r; at ricb = 0 it needs x1 ≳ 6 w1), `'odd top hat'`
  (`ut.erf_top_hat_odd`, x1 = 0 is fine at ricb = 0; 0, 0.03, 0.68, 0.06, 2.6 fits Model S to 10 % rms), `'solar_rz'`
  (the radiative interior of Mukhopadhyay et al. 2025b, their eq. 10: `solar_rz_dSdr`, `solar_rz_steps`, `solar_rz_core`;
  plus the convection-zone superadiabaticity δ(x) d ln p₀/dr above the `solar_rz_os` step, δ from `solar_rz_delta_cz`,
  `solar_rz_delta_sf`, which kore accepts with this stratification only), `'linear'` (r: with 'Boussinesq',
  N² = Beyonce r²), `'inverse square'` (1/r², needs ricb > 0), `'user'` (`stratification_profile`,
  `stratification_param`), `'two zone'` (2026-11-08, conductive_IC's 'two zone' heating: (1 + tanh(2(r − rc)/h))/2,
  [rc, h] = `two_zone`, default [0.7, 0.1]).
- **`comp_stratification`** (dC/dr for r ≥ 0, odd in r, amplitude 1; `compositional = 1`, no default; 2026-11-08,
  `background.gradC`): `'linear'`, `'inverse square'` (needs ricb > 0), `'two zone'` ([rc, h] = `comp_two_zone`),
  `'user'` (`comp_stratification_profile`, `comp_stratification_param`); its index in `background.COMP_STRATIFICATIONS`.
- **`diffusivity`** (ν/ViscosD, κ/ThermaD and D/CompoD, default `'uniform'`): `'step'` (`visc0`, `therm0`, `comp0` below
  r = aux3, 1 above, an erf of width aux4), `'solar_rz'` (the same step at `solar_rz_os`, their eqs. 8–9), `'user'`
  (`diffusivity_profiles` = {'nu': ..., 'kappa': ...} and 'D': ... with `compositional = 1`, `diffusivity_param`). Any
  type is multiplied by (ρ₀(aux3)/ρ₀)^p, p = `visc_pow`, `therm_pow`, `comp_pow`. Forcing 11 takes the floor ν_f = visc0 for 'step' and 'solar_rz' when visc0 < 1, else 0
  (`background.nu_floor`), unless `rsf_nu_floor` is set.
- **'user' types**: numpy expressions of r, x and p with the names of `diff_rot._USER_NS` (sin … tanh, exp, log, sqrt,
  abs, where, erf, ...), evaluated at |r|, and without `#` (they are written to `params.dat` after a `#`).
- **Removed names** (`model_type`, `heating`, `solar_rz`, `solar_density`, `aux0`) stop a run with their replacement
  (`background.REMOVED`), and so does structure `'uniform'` (now `'Boussinesq'`, renamed 2026-10-28);
  `docs/background.md` translates old parameter files. The file-based models (`'poly'`, `'mesa'`, `'gsm'`,
  `'astropy table'`) were removed: they had not worked on this branch.
- **Smooth Sun** (`examples/smooth_sun/parameters.py`, full sphere): 'model S hydrostatic' (`structure_deg` 30),
  r_out 0.985, stratification and diffusivity 'solar_rz' (visc0 0.01, therm0 0.04) with δ_cz = −4e-7, ν = κ = 1e12 cm²/s,
  the HMI rotation table (P₀–P₄, `diff_rot_window` 'sun') with its exact thermal wind (`thermal_wind` 2,
  `thermal_wind_radial` 0), stress-free walls with ∂S/∂r = 0, N 640, lmax 79, `drop_tol` 1e-14; its m 8 HFR mode is at
  ω = −271.19, σ = −33.18 nHz in the 456 nHz frame (21 GB, about 5 minutes on 4 ranks).

To add a type, see §13.

## 7. Boundary conditions (`assemble.bc_u_spherical`, `bc_theta_spherical`)

- `bco`/`bci` = 1 no-slip: P = 0, P′ + P((ln ρ₀)′ + 1/r) = 0, T = 0.
- 0 stress-free, impenetrable: P = 0; r²P″ + r²(ln ρ₀)′P′ + (r²(ln ρ₀)″ + L − 2 − r(ln ρ₀)′)P = 0; rT′ − T = 0.
- 2: zero radial stress instead of P = 0: (r(ln ρ₀)′ − 3)P + 3rP′ = 0.
- 0 or 2 where B0 ends at the wall in a current sheet (an insulator or a TWA with c1 = 0 beyond): the tangential stress
  rows gain the sheet's push (`bc_u_sheet`, phase 6c, §4); a drag at bco = 0 adds `topo.rows`.
- `bci`, `bci_thermal`, `bci_compositional` are ignored at ricb = 0 (regularity comes from the parity basis).
- Thermal (ThermaD > 0 only): `b*_thermal` 0 fixed s, 1 s′ = 0.
- Composition (CompoD > 0 only, 2026-11-08): `b*_compositional` 0 fixed c, 1 c′ = 0 (`background.check_composition`
  stops on other values).

## 8. Differential rotation (`diff_rot = 1`, `bin/diff_rot.py`)

User guide: `docs/background.md` (Rotation); equations, code and checks: `docs/differential_rotation_details.md`.
Background Ω = Ω_ref [1 + ΔΩ], ΔΩ = `diff_rot_ampl` Σₙ aₙ fₙ(r) Pₙ(cos θ), n even, aₙ = `diff_rot_term_amp` (default 1);
flow U₀ = Gaspard ΔΩ r sin θ e_φ in the frame Ω_ref. It works with any background.

- **Types** (`diff_rot_type`, listed in `diff_rot.TYPES`, shapes fₙ in `diff_rot.shapes()`): `'Y20'`, `'rigid'` (test:
  exact Doppler identity), `'Y20-wall-bounded'`, `'shellular'` (σ = `diff_rot_param`), `'cylindrical'`, `'conical'`
  (Guenel et al. 2016), `'shellular_boussinesq'`, `'solar'` (P₀ + P₂ tanh tachocline, frame 435 nHz), `'solar_P4'`
  (c₀ + c₂cos²θ + c₄cos⁴θ with erf tachocline, `diff_rot_solar`, frame `diff_rot_frame`), `'tabulated'` (Ω(r, θ) from
  `diff_rot_file`, .npz with `ome[θ, r]` nHz, `r` R☉, `theta` rad; projected on P₀..P_`diff_rot_nmax`; shell fit of
  degree `diff_rot_rdeg`, or with `diff_rot_window = 'sun'` (default at ricb = 0) one even fit of degree
  `diff_rot_sun_deg`), `'user'` (`diff_rot_profiles` = {n: numpy expression of r, x, p}, p = `diff_rot_param`). `'ferraro'` (ΔΩ = F(ψ) of the poloidal B0, see the magnetic paragraph).
  `diff_rot.fX(n, r, Dorder)` is the Chebyshev fit of `diff_rot_ampl` aₙfₙ (at |r| when ricb = 0), `diff_rot.raw(n, r)`
  the same without the fit.
- **Momentum** (`operators.differential_rotation` → `diff_rot.momentum_block`):
  G[ΔΩ(i m u + 2ẑ×u) + r sin θ (u·∇ΔΩ) e_φ]. The angular couplings of every degree come from the recurrence algebra
  of `angular.py` (`diff_rot._primitives`, `momentum_coefficients`); `tests/diff_rot/gen_check.py` checks them against
  the closed forms of P₀, P₂ in `tests/diff_rot/diff_rot_coefficients.py`. Couplings |Δl| ≤ n (P→P, T→T) and n + 1
  (P↔T) (`diff_rot.dl_range`); `ut.dr_offd` gives the column block.
- **Heat** (`operators.thermal_diffrot` → `diff_rot.heat_block`, `thermal = 1`): i m G ΔΩ s always; `thermal_wind` 1
  (linear) / 2 (exact (1+ΔΩ)², frame independent) add u_r ∂S_tw/∂r + (u_θ/r)∂S_tw/∂θ with the thermal-wind entropy
  S_tw = Σ σₙ Pₙ (`diff_rot.sigma_coefficients`, by the Legendre algebra; needs Beyonce ≠ 0); 3 prescribes
  ∂S_tw/∂θ = `dSdtheta` sin 2θ (Bekki et al. 2024); 0 none. `thermal_wind_radial = 0` drops the u_r term (as Bekki et al.
  and Mukhopadhyay et al.); `thermal_wind_amplitude` scales S_tw. S_tw has degrees up to `diff_rot.tw_nmax()` (n_max + 2
  for thermal_wind 1, 2 n_max + 2 for 2, 2 for 3).
- **Composition** (`operators.comp_diffrot` → `diff_rot.comp_block`, `compositional = 1`, 2026-11-08): i m G ΔΩ c only;
  the thermal wind stays the entropy's (no background composition varies with θ).
- `diff_rot.check()` (called by `submatrices.py` when `diff_rot = 1`) stops on the former names `diff_rot_amplitude`,
  `diff_rot_generic` and `diff_rot_hold` (removed 2026-10-26, §12), on `thermal_wind` outside 0–3 and on
  `thermal_wind` 1 or 2 with Beyonce = 0. With `diff_rot = 0` these names only get the warning of `check_names`.
- ViscosD, ThermaD, Beyonce must be computed for Ω_ref (`units.set_physical` does). At ricb = 0, 'Y20' and 'conical' are
  irregular (their P2 term ≠ 0 at r = 0) and 'Y20-wall-bounded', 'shellular' (unless σ is even and > 0) have a kink
  (WARNINGs from `background.centre_check`); 'solar' needs N ~ 400.
- Benchmarks: Guenel et al. 2016 conical mode ω = 1.8173; solar inertial modes of Mukhopadhyay et al. 2025
  (`tests/diff_rot/run_bekki.py`, needs their `data/diff_rot.npz`, not in the repo): five modes within 0.2 nHz in
  frequency and 0.4 nHz in growth rate of their compressible values; set-up and results in `docs/benchmarks.md`
  (`tests/diff_rot/lit_bench.py`, `solar_checks.py`). The full tables are in `docs/heat-equation.md` §6, §8, §9 and
  `docs/differential_rotation_details.md` §11 of commit 1fbe5f08 (`git show 1fbe5f08:docs/heat-equation.md`). The cost
  against the highest degree (`tests/diff_rot/dr_cost.py`) is in `docs/solver.md` (N 96, lmax 97, 4 ranks: the
  solve's memory grows about linearly, 0.2 GB per degree with the momentum equation alone, 0.4 GB with the heat equation).

## 9. Solver (`solve.py`)

- Options come from `par.petsc_opts` (only the eps_/st_ group for eigenproblems, ksp_/pc_/mat_ for forced); command-line
  options win. Defaults: shift-invert at τ = rtau + i·itau (`units.set_target` sets them from ω, σ in nHz), MUMPS LU,
  `cntl_1` 0.01 (see §12), BLR off (`icntl_35 = 0`; set 2 with `cntl_7 = 1e-14` for large N). For big full-sphere runs:
  `st_mat_mumps_icntl_14` 50–200 (INFOG(1) = −9 otherwise) and out-of-core `st_mat_mumps_icntl_22 = 1`. What to try
  for bad solutions or when memory runs out: `docs/solver.md`.
- `par.prescale`: None = Ruiz equilibration of A − τB, on except for thermal with ThermaD = 0 (where it hurts); 0/1
  force it. Eigenvectors are mapped back before writing. The residual SLEPc prints is for the scaled problem; judge
  accuracy with spin_doctor.
- `which_eigenpairs` 'TM', `nev` (default 5), `tol` 1e-16, `maxit` 50.
- Forced runs: MUMPS LU without BLR, as the eigenproblems. conductive_IC turned on BLR with cntl_1 1e-8 for them
  (1c9ba86: N 992, Ek 1e-9, 316 s and 165 GB, where without them 622 s, 200 GB and NaN); here they are commented in
  `petsc_opts` (`mat_mumps_icntl_35`, `mat_mumps_cntl_7`), since BLR changes every forced result at round-off and
  cntl_1 1e-8 cost accuracy in thermal eigen runs (§12).
- `track_target` (NEW 2026-11-08, from conductive_IC): 1 aims at the eigenvalue in the file `track_target`, which
  `utils.py` reads into rtau, itau at import, before `radial_profiles` (whose import runs `background.check`; not in
  set_scales, which parameters.py calls before its values), and spin_doctor writes back the eigenvalue nearest the
  target; 2 writes that of the smallest KP/KT (needs hydro). Eigenproblems only (`background.check`). `topo.omega0`
  takes itau, so `tools/topo_iterate.sh` makes the drag's frequency the mode's own by fixed-point iteration.
- `par.drop_tol` (default 0, off; the smooth Sun example uses 1e-14): cuts the profiles' Chebyshev coefficients at
  max(1e-15, drop_tol) of the largest, removes entries below drop_tol times the largest of each operator submatrix
  (`submatrices.py`) and, after the Ruiz prescaling, entries of A and max(|τ|, 1) B below drop_tol (`solve.py`). About
  30–35 % less memory at eigenvalues unchanged to ~1e-3 nHz in thermal solar runs.
- Forced runs solve every column of B_forced.npz with one LU (forcing 11: `rsf_nsamp` realisations; the fields files
  get one column each). Forcing 11 with `rsf_frequencies` set: for each frequency w, one LU of A − i(w − wf)B
  (B.npz unnormalised) and back-substitutions for all realisations; appends to `rsf_spectrum.dat` (w, mean KE,
  standard error; mean KE = two-sided PSD, ∫ dw/2π gives the mean KE for m ≠ 0, twice it for m = 0; of the solved symm
  class; with `rsf_split` = a radius, also the KE below it) and `rsf_ke.dat`, no fields. The noise amplitude is
  applied at r, so nothing forces where ν_t = 0. To keep the forcing out of the tail of a viscosity step use `rsf_rcut`
  (a smooth erfc edge, `rsf_rcut_width`), not a higher `rsf_nu_floor`: clipping ν_t leaves a corner the rows cannot
  resolve.

## 10. Post-processing (`spin_doctor.py`, `utils4pp.py`)

Radial integrals use Gauss–Legendre with 3N/2 nodes (`upp.setup_grid`); energies are ρ₀-weighted. Per solution the
table prints Re λ, Im λ, 𝒯/𝒫, the residuals, peak ℓ, ℓ-width, ℓ-convergence, max of the curl of the viscous force
(cvf_r, cvf_l, cvfmax; r < 0.01 skipped at ricb = 0), residens and, at ricb = 0, irreg₀ (§5).

- residσ (`resid1`): 2σKE − Dkin − Wthm − Wcmp − Wdr − Wlor = 0; residω (`resid2`): the imaginary part with Coriolis;
  resid𝑠 (`resid3`): 2σTE − Dthm − Wadv − Wdr_thm = 0 (TE = ½∫p₀|s|²); with `compositional = 1` (2026-11-08) resid𝑐
  (`resid4`, `composition.dat`): 2σCE − Dcmp − Wadv_cmp − Wdr_cmp = 0 (CE = ½∫ρ₀|c|²; Wdr_cmp is round-off, the advection
  being imaginary), and Wcmp, Wcmp_im, the power of its buoyancy (flow.dat 44–45), in residσ, residω and resid𝐄; residens: enstrophy balance; with `magnetic = 1`
  resid𝐛: 2σME − Indu − Mdfs = 0 in the fluid, and resid𝐄 the full energy budget with the inner core, the outside
  potential field and the Ohmic losses (`upp.em_budget`; impenetrable walls: with a slipping fluid the u×B0 wall flux
  comes from the grid, `upp.INDU_WALLS`, 2026-11-06; thin walls add E_w, Ohm_w (`upp.twa_budget`, `slip_lapF`), exact for
  c = 0 and O(c c1) ≈ εTWA (`upp.twa_error`) otherwise; a resolved mantle adds its E_w, Ohm_w (`upp.mantle_budget`),
  exact; a conducting solid turning with ΔΩ keeps its U0×b flux, `upp.INDU_ROT_SOLID`; rotating solids add I|ω|² + K|φ|²
  and the power of their viscous torques, their u×B0 wall flux leaves Indu, only with vtrq = mtrq = 1) (`upp.diagnose_magnetic`, Boussinesq Lorentz and induction terms ported
  from conductive_IC; ME = ½∫|b|², no Hendrik, as conductive_IC; Mag/Kin = Hendrik ME/KE since 2026-11-06, in KE units as conductive_IC 7660861). residω is O(1) for real
  eigenvalues (m = 0, ω = 0): all its terms are round-off there. NEW 2026-11-08: where the walls drive the flow along
  themselves (`walls_drive`: forcing 7 with m = 0, forcing 12) residσ and resid𝐄 are relative to Dvisc too (Dkin =
  Wv − Dvisc is a small difference there; residσ was ~1 for the libration); residω, resid𝐛, resid𝑠 are 0 when all
  their terms are exactly 0 (a steady m = 0 flow; 'Malkus' under an axisymmetric flow, b ≡ 0). A forced rigid rotation
  (walls at one rate, a full sphere, an inner core under a stress-free wall at rest) has round-off terms only: its
  residuals are round-off over round-off, O(1), but for resid𝐋, whose scale has had ViscosD ∫ρ0 r³|T| since 2026-11-09.
  NEW 2026-11-08: forcings 1 and 8 add the power of their body force,
  Wf (and Wf_im) from `upp.body_force_terms` (the routines of flow_worker on x, then on u), to residσ, residω, resid𝐄 and
  P_in (flow.dat 42–43), its enstrophy term to residens, its torque to resid𝐋. NEW 2026-11-08: a background flow
  (`bgflow = 1`) adds Wbg = −2 Re ∫ρ₀u*·[(U·∇)u + (u·∇)U], Wbg_im and Ensbg (flow.dat 46–48) to residσ, residω, resid𝐄 and
  residens, Wbg_thm (thermal.dat 8) to resid𝑠, Wbg_cmp (composition.dat 8) to resid𝑐 (`upp.bgflow_setup`: the terms on
  an (r, cos θ) grid from `bgflow.flow_grid`, `scalar_grid`, as diff_rot_setup; tests/bgflow/power_check.py recomputes them
  from textbook components). Full-sphere budgets at N 48 are ~1e-8 (forcing 7 too), 1e-15 at N 96.
- resid𝐋 (NEW 2026-11-06, `rotdyn.L_case`: eigen, hydro, no centrifugal; m = 0, symm = 1, or m = 1, symm = −1 without
  diff_rot; since 2026-11-08 also forcings 7 (m = 0), 8, 12 and 13, axial, and 1 (m = 0 axial, m = 1 equatorial),
  with λ = i wf and the body force's torque, and with a background flow, which has no torque): λL = −(torques on the walls) (+ i Gaspard L for m = 1, L = (L_x − i L_y)/N, N = −1/√2), viscous from T of
  degree 1, magnetic from the Maxwell stress on a θ grid (`rotdyn.wall_torques`, any B0); relative to |λ|∫|T| and the
  torques. 1e-15 hydro, 1e-11 with fields, 1e-9 full sphere. flow.dat columns 29–39 (L, torques, resid𝐋); rotdyn.dat.
- Dtopo, δσtopo (NEW 2026-11-06, `topo.dissipation`; viscous runs with bco = 0, Gaspard > 0; with centrifugal = 1 since
  2026-11-09, on the body's wall through K_eff and with KE + KEc): the power
  the drag dissipates, Dtopo = 2 Gaspard ρ_w ∮(u_t − v_m)*·K·(u_t − v_m) dS on the wall grid, and δσtopo = −Dtopo/(2E), E =
  KE + Hendrik (ME + MEc + MEic + MEext + E_w) + I_M|ω_M|² + I_IC|ω_IC|² + K|φ|². Without a drag both are per unit κ
  (K = 1, columns 'Dtopo/κ', 'δσtopo/κ'): the first-order damping of a weak drag, which leaves out the change of the other
  losses (5e-4 of it for the spin-over mode at Ek 1e-3; with the solids the drag turns the mantle and the change is about
  half). Dtopo = −Wv_cmb − 2 Re(ω_M* Γ^ν_M) (tests/drag). flow.dat columns 40–41; params.dat 135–141 (kappa_cmb,
  topo_drag 1 for 'rekier', topo_hrms, topo_lambda0, topo_H and the N/Ω, w0/Ω used, 0 without 'rekier').
- These residuals, not the eigenvalue alone, judge resolution: compare with a higher-N run; for thermal runs the
  entropy can be unresolved while P, T look converged. The docs report 1e-7 to 1e-12 for converged runs.
  `docs/solver.md` accepts a mode when residσ and resid𝑠 are below 1e-5 and its ℓ-convergence below 1e-4; spin_doctor
  itself only prints the values (no pass/fail flag), and it misses some solver failures in clusters of nearly equal
  modes (§12).
- `params.dat` columns are `spin_doctor.param_table()` in order (append new ones at the end); `diff_rot_type`,
  `structure`, `stratification` and `diffusivity` are written as their index in `diff_rot.TYPES`,
  `background.STRUCTURES`, `STRATIFICATIONS`, `DIFFUSIVITIES` (−1 if not listed, or without a stratification); None as
  nan. A row with 'user' expressions or a 'tabulated' rotation ends with ` # ` and a note (`spin_doctor.row_note`: the
  expressions, the table file and the first 12 hex digits of its sha256), which `np.loadtxt` skips. `flow.dat`,
  `thermal.dat`, `magnetic.dat`, `composition.dat` columns are the `flow_cols`, `thermal_cols`, `magnetic_cols`,
  `composition_cols` lists. params.dat 142–154 (2026-11-08): compositional, comp_stratification (its index in
  `background.COMP_STRATIFICATIONS`, −1 without the composition), comp_stratification_param, BeyonceC, CompoD, comp0,
  comp_pow, bci/bco_compositional, comp_two_zone and two_zone (rc, h each); 155–156 (2026-11-08) bgflow, bgflow_Ro (its
  expressions, or its file and sha256, in the row's note, `bgflow.description`).
- With diff rot, `upp.diff_rot_setup` computes the force and heat terms on an (r, cos θ) grid and projects them back.
- With `centrifugal = 1` (2026-11-05) KE (also solve.py's normalisation), the Coriolis power and TE include the metric
  parts (`upp.centrifugal_terms`, from the fields on the grid, apart from the matrices; CE and Dcmp
  `upp.centrifugal_composition_terms`, 2026-11-08); KP and KT are without them;
  residens is NaN. `tools/plot_field.py` maps the fields to the body (`centrifugal.to_physical`). With a field (phase
  1b) the normalisation is KE + Hendrik (ME + MEc), MEc the metric part of the magnetic energy (`magnetic.dat` column 12,
  `upp.centrifugal_magnetic_energy`); `upp.centrifugal_magnetic_terms` adds the metric Lorentz work to residσ, residω and
  the metric induction work to resid𝐛 (round-off); resid𝐄 closes at O(f²) only (the products of first-order terms).
  Phase 2: `upp.centrifugal_dissipation_terms` adds the metric parts of the viscous power Dkin, of the work of ΔΩ and of
  the thermal dissipation (the budgets close to round-off); Dvisc and the wall powers stay spherical, so residD is NaN.
- Walls (2026-11-04): `upp.wall_powers` gives the work of the pressure (Π = r G_s from the consoidal momentum equation,
  `pressure_force`; in residσ, residω) and of the viscous stress at each wall, `upp.viscous_dissipation` the dissipation
  ∫𝛔:∇𝐮 on an (r, θ) grid; `Dkin` = Wv − Dvisc (residD). Π at the walls converges more slowly than the volume terms
  ('Wu 2005' at Ek 1e-3: N 96 gives residσ 5e-2 in a forced run, N 192 2e-10); residσ flags it.

## 11. Tests

- `bin/ru_insane.py [nvec] [--tol 1e-6]` after `submatrices.py`: inertia Hermitian positive, Coriolis anti-Hermitian,
  viscous Hermitian negative, entropy/diffusion blocks, buoyancy/advection against quadratures, diff-rot blocks against
  an (r, θ) grid; with magnetic = 1 the b and diffusion blocks and, for B0 'axial' with no-slip walls, the Lorentz/induction
  energy identity; with compositional = 1 (2026-11-08) the composition's blocks (c Hermitian positive, its diffusion
  Hermitian negative, its buoyancy, advection and c against quadratures, the diff-rot and centrifugal blocks against a
  grid). Needs N ≥ 96 (fails on truncation below); does not cover the inviscid formulation.
- `tests/heat_dr/` (README): `mk_run.py NAME 'par.x = ..'` builds a run dir (defaults: structure 'Wu 2005' with
  stratification 'top hat', ricb 0.35, m 1, conical −0.25, ViscosD = ThermaD = 1e-3) and runs submatrices + assemble;
  `heat_check.py`, `asm_check.py`, `row_check.py`, `rigid_thermal.py`, `tw_sympy.py`, `csolve.py` (scipy solve, no
  equilibration), `eig2kore.py` (writes csolve's eigenpairs as solve.py does, so spin_doctor runs without
  PETSc/SLEPc), `sd_check.py`, `ebal.py`, `terms.py`, `tw_limit.py` (thermal_wind 2 → 1 as the amplitude → 0),
  `mpi_check.sh` (bitwise A, B on 1–4 ranks), `forced_check.py`; `run_all.sh` runs all but `eig2kore.py`, `mpi_check.sh`
  and `forced_check.py` (~10 min serial; NEW 2026-11-09: it builds the runs side by side first, NPAR at a time, default
  the processors with at most one per 2 GB; each check it runs exits with 1 on a FAIL, as does its line on the WARNINGs
  of the 'solar' and 'cylindrical' runs, and it stops there, so its exit code is their verdict; tw_limit.py's ratios
  and terms.py's balances are read).
- `tests/diff_rot/` (README): `gen_check.py` (the couplings of `diff_rot.py` against the closed forms of
  `tests/diff_rot/diff_rot_coefficients.py`, and the Δl ranges), `angular_check.py` (`angular.py`, no run directory;
  its reference for the primitives is `quad_reference.py`, the former quadrature builder, valid for l ≤ 645),
  `regress.sh OLD_KORE` (26 small runs, 15 of the rotation and the heat equation, each written in the parameter names of
  three generations of checkouts, and 11 magnetic ones, skipped when the older copy lacks their code; A and B bitwise
  equal, or to 1e-13 of the row scale; how to make the older copy: `git archive <commit> | tar -x -C DIR`, docs/checking.md), `tw_mukhopadhyay.py`, `mk_bekki.py` / `solve_bekki.py` / `run_bekki.py` (solar benchmark),
  `mk_rz.py` (radiative interior, any inner radius), `lit_bench.py` and `solar_checks.py` (the benchmarks of
  `docs/benchmarks.md`), `dr_cost.py` (time and memory against the degree of ΔΩ),
  `ls_to_tabulated.py` (HMI table converter, input in `profiles/`: the Larson & Schou 2018 HMI average
  `rot2d.hmiv72d.ave` and `rmesh.orig`).
- `tests/magnetic/` (README, `run_all.sh`: NEW 2026-11-09 its checks are independent jobs, NPAR at a time (default the
  processors, at most one per 3 GB), the longest first (FIRST), logs in WORKDIR/logs, the sympy checks in parts
  (`--part k/K`: derive_check.py NPAR, algebra_check.py NPAR/2, dr_induction_check.py NPAR/8, rounded up); two hours
  of jobs, 31 min with NPAR = 4): `derive.py` (the magnetic rows from the definitions, sympy),
  `derive_check.py` (against conductive_IC's operators and the anelastic closed forms), `row_check.py` (each row block of a
  run against derive.py, unweighted), `budget_check.py` (spin_doctor's residσ, residω, resid𝐛, resid𝐄), `continuity.py`
  ('Wu 2005' with r_out = √ε → 'Boussinesq'), `malkus_check.py` (the Malkus field against Malkus 1967), `mk_run.py`.
- `tests/tools/` (NEW 2026-11-08, README, `run_all.sh`, 2 min on one rank, needs pandas and matplotlib): `tools_check.py`,
  the solution files in both formats bitwise (every field, rotdyn, kinematic induction, the bgflow writer), the JSON
  round trip and a run from it, track_target 1 and 2, ramp.sh (tracking, the stop), reap.sh + get_data.py,
  topo_iterate.sh (a fixed point), setpar.py, get_data's names against spin_doctor's tables (ast), the stops, and the
  plotting tools through `plot_check.py` (load_solution against expand_sol and the inner core's continuity at the ICB,
  plot_rad_con_tor_profile's energy against KE, plot_field's 'b' against its coefficients; shell, full sphere 'Wu 2005',
  forced, text format).
- `tools/derive/` (NEW 2026-10-29, README): `fit_rows.py` (fits the magnetic rows of derive.py in l, m: angular factor ×
  (a(l) + b(l) m²), exact rational coefficients; reproduces operators.*_tor, *_tor2, *_pol2) and `gen_table.py` (the fit
  as operator table entries in kore's labels). `docs/derivations.md` explains the couplings of both latitude-dependent
  backgrounds, each with motivation, method, implementation map, verification (which degrees each check reaches) and
  worked examples: part A the magnetic field (derive.py `--b0 pol1|pol2|tor1|tor2`, the exact-interpolation fit, the code
  generation), part B the differential rotation (the recurrence algebra of `bin/angular.py`, the 12 primitives and the
  couplings of `bin/diff_rot.py`). The scripts' docstrings give the same rationale (2026-10-30).
- `tests/centrifugal/` (README, `run_all.sh`: NEW 2026-11-09 its checks are jobs side by side, NPAR at a time, the longest
  first (FIRST), logs in WORKDIR/logs; 55 min of processor time, 25 min with NPAR = 4; `SKIP_SOLVE=1` without PETSc;
  NEW 2026-11-05): `derive.py`
  (sympy: every closed form of `centrifugal.py` from the map and w = J F⁻¹u), `algebra_check.py` (the couplings against
  quadratures with scipy's lpmv), `clairaut_check.py` (Maclaurin 5/4, polytrope 7.5/π², 'Wu 2005' against Clairaut's
  equation from ρ, the Sun's f/q 0.519), `eig_check.py` (f = 0 bitwise, the spin-over, a relabelled sphere moving at
  O(c²), homogeneous 'clairaut' = 'uniform' (bitwise since phase 6d), the monopole = Beyonce(1 − 2q/3), the ideal fluid's
  real frequencies; with a field: f = 0 bitwise, the relabelled sphere against the field written on the sphere, in a
  viscous fluid; the relabelled free decay), `budget_check.py [tol] [HALF | exact]`
  (with a field also resid𝐛, and resid𝐄 of second order from the run at half the flattening; `exact`: resid𝐄 below tol,
  the exact spheroid's `bms_*` runs), `mk_run.py` (defaults: the
  inviscid Boussinesq full sphere, m 1, N 24, lmax 20, 'uniform' f 0); phase 1b: `derive_mag.py [--quick]` (sympy: the
  covariant magnetic terms from b = F b̃/J, kore's couplings against them, the wall terms; part B in parallel) and
  `bc_check.py` (the wall rows against the exact matching of a potential field on the deformed wall: residual O(f²)),
  `fcn.sh [WORKDIR] [NP]` (the free core nutation with a field and a conducting inner core at Ek 1e-7, the shift taken
  between ±f: Im λ moves by f(1 + f/3) plus 6.1e-4 of f from the Ekman layers and the field, Greenspan's value 4.6e-4;
  15 ranks, `st_mat_mumps_icntl_14` = 1000, 5 min, 15 GB; not in run_all.sh); phase 2: `derive_diff.py [--quick]` (sympy:
  the viscous force, the traction on the level surface, the heat diffusion and flux, the magnetic diffusion and the ΔΩ
  force from the map, and kore's blocks and wall rows against them), `spheroid_modes.py [--ekman]` (the exact inertial
  modes of a spheroid by the polynomial method: dω/df of 73 modes against kore's), `shape_check.py [NPROC]` (damped modes
  on deformed walls at Ek 1e-3 against Hadamard's formula and a volume formula: no-slip, stress-free, heat);
  `ru_insane.py` checks the metric blocks against quadratures, and the viscous force Hermitian and negative; the exact
  spheroid: `spheroid_check.py` (its terms from the map, kore's couplings and walls against projections with
  derive_diff.py's Laurent polynomials, which it imports (derive_diff runs its checks in `main()`), and the inviscid,
  viscous and heat modes against the polynomial method and exact rational Galerkin solvers; with a field (step 2) the
  field's couplings, λ_l against mpmath and the wall rows on exact potential fields, the axisymmetric free decay against
  an exact Galerkin solver (the exterior through the closed form of Q_l^1), conducting and insulating inner cores, the
  parting from 'uniform' as f²; 3 min on 4 cores; decay runs need Le 1e-8, or a viscous flow mode nearby moves the rates
  by 1e-9).
- `tests/rsf/` (README; `HOWTO.md` points to `docs/forcing.md` and `docs/reynolds-stress.md`; `example_m2/`, the
  chandler m = 2 scan, has its own README): shared
  pieces in `rsf_common.py`; `builder_check.py` (f = u against the inertia rows), `noise_check.py` (the noise
  amplitude against the documented Θ laws, and the noise against the viscous operator, exactly; no forcing beyond the
  dissipation bound), `spectrum_check.py pick|fields|check` (solve.py's sampled spectrum against the exact E[KE(w)]
  and the single-mode energy Θ f_eff; in a shell `sp`, NEW 2026-11-09 also in a full sphere `sp_fs` and with a field
  and a conducting inner core `sp_mag`); `run_all.sh` (needs petsc4py for the solves, and an `MPIEXEC` that starts
  mpi4py's MPI); `regress_solve.sh OLD_KORE` (eigen and forcing-10 solves bitwise against an older checkout).
- `tests/tidal/` (README, `run_all.sh`, ~2 min on 4 ranks): `stokes_check.py` (forcing 14 without rotation against the
  exact unsteady Stokes solution: P_2(r), KE, Wp, Wv, Dvisc), `budget_check.py` (residσ, residω, residD and the wall
  work on seven runs: 'Wu 2005', buoyancy, diff rot, the three wall conditions, a full sphere at N 384), `mk_run.py`.
- `tests/decay/` (README, `run_all.sh`, ~30 s; NEW 2026-11-06): `decay_check.py` (hydro = 0: the free-decay rates of the
  field between insulators, with a conducting inner core (sigma_i2o 0.3), in a full sphere, and of the entropy, against
  the roots of spherical Bessel functions (scipy), to 1e-10, the budgets to 1e-9 (resid𝐛 with the conducting core 1e-10
  at round-off: Mdfs = ∫ b⋅∇²b, third derivatives, grows with N); with a differential rotation hydro = 0 against hydro = 1 with
  cnorm 1e-9; just_induction.py + solve_ind.py against a mode with Hendrik 1e-30).
- `tests/twa/` (README, `run_all.sh`, ~1 min; NEW 2026-11-06): `twa_check.py` (thin walls: hydro = 0 free decay against
  the exact thin-wall rates; with the flow against conductive_IC 2d84351 eigenvalues hard-coded for N 48, lmax 20/19 on 2
  ranks; c = c1 = 0 equals insulating; the rigid offset with turning walls; resid𝐄 with sheared walls).
- `tests/mantle/` (README, `run_all.sh`, ~1.5 min; NEW 2026-11-06): `mantle_check.py` (the resolved mantle: hydro = 0
  free decay against exact rates, Bessel functions or DOP853 in the layer, uniform, exponential, full sphere, conducting
  IC; with the flow against conductive_IC 2d84351 eigenvalues hard-coded for N 48, lmax 19-21 on 2 ranks; a thin layer
  tends to 'TWA' (c = h, c1 = σh) at O(h); σ = 1e-9 equals insulating; 'Wu 2005' rigid offset and sheared rotation
  resid𝐄; just_induction with the layer's field).
- `tests/rotdyn/` (README, `run_all.sh`, ~1 min; NEW 2026-11-06): `rotdyn_check.py` (rotdyn = 1: decoupled solids (vtrq = 0)
  against the roots of λ² + cλ + K(1/I_M + 1/I_IC), heavy solids (I = 1e12) against fixed walls; conductive_IC 2d84351
  eigenvalues and rotdyn.dat ratios hard-coded for N 48, lmax 19 on 2 ranks (hydro, Earth-like, TWA, cIC + resolved
  mantle, axial + exponential mantle); 'Wu 2005' total angular momentum, resid𝐋, energy and resid𝐄; equatorial resid𝐋
  (m = 1, symm = −1) with fields, TWA, toroidal B0 and a full sphere; the stops of check_rotdyn, whose wall case is now
  bci = 0).
- `tests/drag/` (README, `run_all.sh`, ~2 min; NEW 2026-11-06): `drag_check.py` (kappa_cmb, topo_drag 'rekier': eleven
  runs against conductive_IC 2d84351 eigenvalues and rotdyn.dat ratios hard-coded for N 48, lmax 19-21 on 2 ranks
  (spin-over, m = 2, full sphere, the solids with a stress-free wall, TWA, cIC + resolved mantle); 'Wu 2005' with the
  solids: total angular momentum, resid𝐋, energy and resid𝐄; topo_K constant = kappa_cmb; a rigid diff_rot a with
  kappa_cmb = a frame 1 + a times faster with kappa_cmb/(1 + a), shifted by −i m Gaspard a; kappa_cmb 1e8 = no-slip;
  a weak drag against Dtopo/κ and δσtopo/κ; the N/Ω and w0/Ω in params.dat, a doubled time unit, a forced run;
  Dtopo = −Wv_cmb − 2 Re(ω_M* Γ_M) in every run with a drag; the stops of check_drag). `deformed_check.py` (NEW
  2026-11-09, ~1 min on 4 processors, side by side with drag_check.py; runs by tests/centrifugal/mk_run.py, one rank,
  NPAR at a time): the rigid rotation of 'spheroid' f 0.3, −0.2 (full, shell) and 'uniform' ±0.01 decays per unit
  kappa_cmb as on the body's wall (2e-8); the inviscid spin-over of the exact spheroid: −Dtopo/(2E) of its eigenvector
  (the drag set in memory by a helper, after the imports) against the closed-form wall integral (kappa_cmb, topo_K hooks
  5e-11; 'rekier' with N|∇r| 4e-8); 'uniform' ±0.02, ±0.01 off the spheroid as f² (orders 1.98-2.02); a relabelled
  sphere 'user' f = c(1 − r²), c = ±1e-3: odd part/(c R) 6e-7 (0.27 without the η terms); weak drag vs dsig_topo
  (f 0.3, Ek 1e-3) 4e-4; kappa_cmb 1e8 = no-slip 4e-12 (by solve.py: scipy's splu fails on the stiff deformed rows);
  f = 0 bitwise; rotdyn + centrifugal stops.
- `tests/forced_field/` (README, `run_all.sh`, ~3 min; NEW 2026-11-06): `forced_check.py` (8 forced runs with a field
  against conductive_IC 2d84351, KE, ME, MEic hard-coded for N 48, lmax 19-21 on 2 ranks: libration, forcing 9 insulating
  and TWA 'deforming'; Bessel = Chebys ('rigid') under forcing 9, also with diff_rot and with sigma_i2o 0.3, and with
  mu_i2o 1.5 where forcing 9 moves the outer wall only, and forcing 6; A(w1) − A(w2) = −i(w1 − w2)B and A(0) = A_eig;
  forcing 11 scans independent of the assembly frequency; resid𝐄 at moving walls, 13 'rigid' (2 of them free walls,
  bco/bci = 2, in eigenproblems) and 17 'deforming' runs ('Wu 2005' N 96, Malkus, general, G21, Luo_S2, full sphere, slip,
  σ(r), sigma_i2o, forcings 6, 7 m 2, 9, 10, 14), and 2 with no wall moving radially (forcing 13, libration); a thin
  resolved mantle → TWA at O(h) in both pictures (axial, Malkus); bc_b_radial's algebra (and tor_half) against a grid;
  kinematic induction in forced runs; the stops, also of the 'deforming' defaults of forcings 9 and 14 (forcing 10's by
  its matrix, default = 'deforming' exactly), and of a permeability jump at a wall that moves where B0 reaches it, in both
  pictures (also muw of a sheet at rest under B0 'axial'; not at walls that stay in place, for muw of a TWA with
  c = c1 = 0, of sheets that deform under 'axial' or of a thick layer with c1 = 0 under 'Malkus', or where B0 is 0 at the
  walls: B0_tor 'wall', a 'general' h with h = h' = 0 at the wall); xdlogjl against mpmath). A thin layer's budget closes exactly at c = 0 or
  c1 = 0 (O(c c1) otherwise): f6_muw_fixed shows muw at a fixed wall with c_cmb 1e-3, c1_cmb 0.
- `tests/inner_core/` (README, `run_all.sh`, ~1.5 min; NEW 2026-11-07): `core_check.py` (B0 'dipole' against conductive_IC
  2d84351 eigenvalues hard-coded for N 48, lmax 19-20 on 2 ranks: insulating, mag_energy, antisymmetric + resid𝐋, TWA
  mantle, resolved mantle, TWA core, rotdyn; forcing 7 KE with Bessel; a slipping cIC (bci = 0) against conductive_IC:
  axial m 1 antisymmetric and m 0 + resid𝐋 (1 rank), dipole, G21 with μ 2, σ 0.3, Luo_S2 with both walls slipping, resid𝐄;
  'Wu 2005' with buoyancy resid𝐄 (N_cic 48: 24 leaves 2e-8 in the most damped mode); a rigid offset against the frame
  turning with the solids; just_induction + solve_ind = a mode with Hendrik 1e-30; Bessel = Chebys under forcing 7 (axial,
  dipole, + conductive_IC KE), 10 (G21, μ 1.5, σ 0.3; and with diff_rot) and 6 ('rigid'), and 6 'deforming' resid𝐄 (axial,
  dipole), 6 in 'Wu 2005' (the slip's (ln ρ0)' term, which only a flow through the wall sees: 2e-3 without it); 26 stops
  and runs that go on: dipole, a singular 'general' and a 0/0 'user' profile in a full sphere (and the where() form,
  which runs), a zero field with cnorm_tor 'mag_energy', a zero 'ferraro' field and a NaN one (cnorm 1), cIC with bci = 2, centrifugal with a
  slipping cIC, forcing 7 with the core librating behind a slipping conducting wall, a toroidal B0's sheet in an
  inviscid full sphere (also 4e-6 of the field under a strong 'axial' one, and an irregular full-sphere profile's),
  rotdyn's stop, and the runs that go on (since 6c a viscous slipping wall: bco 0 and 2, bci 0 and 2, TWA c1 = 0 at
  either wall; no-slip, cIC, TWA c1 ≠ 0 at either wall, resolved mantle, 'wall'); 50 checks).
- `tests/sheet/` (README, `run_all.sh`, ~11 min of runs on 2 ranks, NEW 2026-11-09 side by side, NPAR at a time: 8
  min with NPAR = 2 on four processors that two other suites shared; NEW 2026-11-07, phase 6c): `sheet_check.py` (the
  push of
  B0's current sheet at a slipping wall: nine smoothed sheets, B0 falling to the field beyond over δ = 2e-3, 1e-3, 5e-4,
  the first four eigenvalues against the sharp run with the push (error ratio 0.35–0.65, the δ, δ² extrapolation 10×
  closer, the no-push run 10× farther): toroidal k = r at the outer wall (bco 0) and the inner one (bci 2, cnorm_tor 2
  with Hendrik 0.025, the eigenvalues of 1 and 0.1), 'general' outer (n 1 and 3, cnorm 2 with Hendrik 0.025) and inner,
  a TWA with c1 = 0 outer (n 1, c 0.05, muw 3, mu 1.5) and inner (n 3, mu 0.8), Wu 2005 with ν = 1 + r (ρ0 ν = 0.07 at
  the wall), the outer toroidal one with kappa_cmb 0.01 (it must move the eigenvalues 10× the error at 5e-4), and
  'general' with a toroidal sheet at the outer wall with m 2, symm -1 (lmax 25: the other l-sets); L_z = 0 and resid𝐋
  for m 0 in a shell (cnorm_tor 'mag_energy'), a full sphere and Wu (|L_z| > 1e-3 without the push); residσ, resid𝐄 <
  1e-8; stops (inviscid poloidal, also behind a permeable layer with B0_cmb 'plain'; centrifugal; 'axial' with mu 1.5 or
  a muw 2 layer at the outer wall, at the inner one (no B0_ic tag) and inviscid; 'dipole' and G21 'frozen currents' at a
  muw 2 inner layer) and runs (no-slip, one permeability, a poloidal sheet at bco 2, the inviscid permeable layer with
  B0_cmb 'wall'); `_pol_wall_row` against bc_b_spherical's f rows at both walls, n 1–6, to 1e-13, and B0_cmb 'wall'
  ('FDM', G21) with no sheet at a permeable outer layer, 'plain' G21 with one; the no-push runs patch assemble.py's
  sheet_on line; 44 checks).
- `tests/forcing/` (README, `run_all.sh`, ~1.5 min; NEW 2026-11-06): `rows_check.py` (forcings 6 at a no-slip and a
  stress-free ICB, 7 with m = 0 and 2, 9 Boussinesq, 'Wu 2005' and full sphere, 10, and 12 with both walls turning,
  under a stress-free outer wall and in a full sphere, and (2026-11-08) 2 with m 2 for both signs of wf and m 0 at ricb
  37/42 (expected values strings evaluated in the run, from `utils.eccen_tide`), and (2026-11-09) the settings that did
  not run before phase 11 and run now (14 in a full sphere with bci 0 and inviscid at bco 0, 9 with kappa_cmb at an
  outer wall that stays in place, which stopped; 9 in a full sphere with bci 0 and 7 (m 2) inviscid at bco 0, which got
  no right-hand side) and two that ran before without a check (9 at a stress-free outer wall, 10 inviscid at bco 0): the wall values read back from the solution of small forced runs built with
  `tests/tidal/mk_run.py`, to 1e-10 of the amplitude); `stops_check.py` (NEW 2026-11-09: the stops phase 11 added or
  moved to startup, each with its message: forcings 6, 7, 9, 10, 11 (its own conditions included), 13 and 14 in
  `background.check_forcing`, the steady-wall rule, the transport rule of `background.check_transport`, the inviscid
  shell, 9 with the drag; and the nearby settings that run; from `submatrices.py` alone on 1 rank. Forcing 12's stops
  are in tests/couette, those of forcings 1, 2, 8 and 3-5 in tests/cic_forcings).
- `tests/couette/` (README, `run_all.sh`, ~1.5 min on 2 ranks; NEW 2026-11-08): `couette_check.py`, forcing 12: eleven
  runs against conductive_IC 2d84351 (KE, ME, MEic hard-coded, N 48, lmax 31; the IC turning, both walls, the mantle;
  'axial' over insulating, Chebys, Bessel, TWA cores, TWA and resolved mantles, mu_i2o 1.5 and muw_cmb 2 with c_cmb;
  'Luo_S2' Bessel; MEic from al's spin_doctor on its fields, as conductive_IC writes none for forced runs), the Stokes
  flow without rotation (r⁴ρ₀νω' const: 'Boussinesq', 'Wu 2005', variable ν), rigid rotations (b = 0 with conducting
  and permeable solids), the budgets ('Wu 2005' with buoyancy and a field, diff_rot, 'Malkus' (b ≡ 0) and 'wall', a
  slipping fluid with a drag, TWA, resolved mantle), the libration and forcing 13 (resid𝐋, residσ, residω), the stops of
  `background.check_forcing`.
- `tests/cic_forcings/` (README, `run_all.sh`, ~2.5 min on 2 ranks; NEW 2026-11-08): `cic_forcings_check.py`, forcings 1, 2,
  8: KE (ME) against conductive_IC 2d84351 (forcing 1 m 2 no-slip and stress-free, lmax 41; forcing 8 at twice cic's
  angle, without and with 'axial' + Chebys, lmax 23) and the 12 eccen_tide pairs bitwise; forcing 8 = forcing 7 − x to
  1e-12 (Boussinesq, 'Wu 2005' + diff_rot, Bessel + TWA field, full sphere, IC librating relative, slipping ICB; −x
  alone without coupling: stress-free, inviscid full sphere); forcing 1 budgets (m 0, 1 shell and full sphere at N 96,
  stress-free, diff_rot, 'user' ν), the exact rigid rotations of a stress-free or inviscid full sphere (T_1 = 3ia/(2w) r
  for m 0, (3√3/4) i a/(w − 1) r for m 1), Gaspard scaling (viscous units: KE × 1e6); forcing 2 budgets (m 0, 2 ±,
  'Wu 2005', 'Y20' diff_rot, a field); forcing 8 budgets where 7 cannot go (drag, TWA, resolved mantle, slipping ICB
  along a Chebys core); 21 stops.
- `tests/composition/` (README, `run_all.sh`, ~3.5 min on 4 cores; NEW 2026-11-08): `composition_check.py`, the
  composition: three runs against conductive_IC 2d84351 eigenvalues hard-coded (N 96, lmax 49, 2 ranks: its 'internal'
  and 'differential' compositions and its 'two zone' entropy under a composition), to 1e-12; with 'Boussinesq' the
  composition alone against the entropy alone with the same numbers (eigenvalues, or KE of the forced run, and
  composition.dat against thermal.dat to 1e-12) in 17 pairs (full sphere, flux walls, no diffusion, diff_rot, a 'user'
  deformation and the spheroid, a field with a conducting IC and a resolved mantle, the solids, the drag, libration,
  'user' and 'step' diffusivities, hydro = 0, 'two zone'), and 'Wu 2005' without diffusion (eigenvalues only: the
  budgets are weighted by ρ₀ and p₀); both scalars equal = the entropy with Beyonce doubled; 'Wu 2005' budgets below
  1e-9; ru_insane on six runs; 11 stops.
- `tests/bgflow/` (README, `run_all.sh`, ~8.5 min of runs on 2 ranks, NEW 2026-11-09 side by side, NPAR at a time: 9
  min with NPAR = 2 on four processors that two other suites shared; NEW 2026-11-08): `bgflow_check.py`, the
  background flow:
  `bin/bgflow.py 12 [nonlinear]` in the run of conductive_IC 2d84351's inner core turning at 1 (N 48, lmax 31) gives its
  files `cic_bgflow_lin.npz`, `cic_bgflow_nl.npz` again (1e-12 of the largest coefficient, the nonlinear one 1e-10) and
  leaves A as it was; eigenvalues under them against conductive_IC's (1e-9); a toroidal background = diff_rot 'user'
  (A, B to 1e-14: 'Wu 2005' shell with s, c, a field, and a full sphere); a rigid T_1 = r at Ro 0.1 = the frame at
  Gaspard 1.1 shifted by −i m Ro (viscous and inviscid full spheres, no diffusion, conducting IC and resolved mantle,
  TWA); budgets with Wbg, Wbg_thm, Wbg_cmp (below 1e-9) and the powers against `power_check.py` (textbook components on an
  (r, θ) grid, 1e-8) in eigen and forced runs (forcings 1, 8 (+ x = forcing 7), 12 with a field; m 1 symm −1 for the
  equatorial resid𝐋); the stops and the full-sphere centre warning. `rows_check.py` (each row of `bgflow.build` against
  the fields on a θ grid, 1e-8).
- `tests/ci/test_spinover.py` (`cd tests/ci; pytest -v`, about 6 s): the test GitHub runs on every push to main, the default branch
  (`.github/workflows/anelastic_log.yml`: Ubuntu 24.04 with its PETSc/SLEPc 3.19, numpy 1.26 and scipy 1.11 packages, so
  `bin/` must not need newer ones; pushes that change only docs or `.md` files are skipped). It runs the smooth Sun
  example with m 1, N 32, lmax 16 on 2 ranks, and a rigid rotation offset ('rigid', 0.1): its spin-over is exact, λ = i
  (found to 1e-16; a 1 % error in the rotation terms or in the ν′ or (ln ρ₀)′ terms of the toroidal viscous force moves
  it by 1e-6 to 1e-3). conductive_IC has its own tests and workflow: do not run one branch's tests on the other.

## 12. Known issues and gotchas

- **`diff_rot_hold`** (found faulty 2026-10-25, removed 2026-10-26): it held every degree below x_h, so the latitudinal
  shear of the table at x_h filled the whole core (with the HMI table at 0.6 R☉, a 4 nHz inversion artefact that gave
  the core spurious modes). Most smooth-Sun results from before 2026-10-26 used 0.6 (`docs/heat-equation.md` §8 of
  commit 1fbe5f08 says which); their parameter files stop until the line is deleted, and their `params.dat` rows have x_h in its column.
- **MUMPS on several ranks is not bitwise reproducible** (2026-10-25): nearly parallel eigenvectors, such as the r-mode
  ladders of a radiative interior, amplify the round-off, and about half of repeated runs came out "smeared" (ladder
  members 0.05–0.5 nHz off, not all of them flagged by `spin_doctor`). `par.petsc_opts['st_mat_mumps_cntl_1'] = 0.1`
  (default 0.01) cut that to about a quarter at the same cost; cross-check ladder members against a repeat or
  neighbouring runs (`docs/solver.md`). Multithreaded BLAS adds last-bit differences even for separated modes
  (2026-11-06: one of three identical runs differed at 1e-15), and so, more rarely, does a solve on 2 ranks with one
  thread (one of five identical runs, with the eigenvectors' arbitrary phase changed too): compare solves with
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, repeat one that differs, and judge a change by its matrices (`regress.sh`).
- **Inviscid** (`ViscosD = 0`): `submatrices` switches to the inviscid bases only when also ricb = 0, while
  `assemble.bc_u_spherical` uses one BC row whenever ViscosD = 0, so ViscosD = 0 with ricb > 0 looks inconsistent.
  Untested on this branch.
- **ThermaD = 0**: spurious modes trapped at the inner boundary unless dS/dr → 0 there (with 'top hat', keep
  x1 ≥ ricb + 3 w1); a tiny ThermaD (1e-9) reproduces ThermaD = 0 eigenvalues except with the thermal wind and
  fixed-entropy walls (boundary layer, `docs/background.md`, ThermaD = 0). Raise `nev` by 1–2 if spurious modes take slots.
- **ricb = 0**: profiles must be smooth and of the right parity at r = 0 (stratification 'top hat' needs x1 ≥ ~6 w1;
  'odd top hat' and 'solar_rz' are smooth there) and a degree-n profile must go as r^n (§5). Model S profiles need ~100
  Chebyshev polynomials per parity; ru_insane fails at N 96 there. Inviscid and ThermaD = 0 full-sphere spectra hold
  spurious modes that do not converge in N or lmax, many of them at the centre (irreg₀ 0.02–2).
- **Open physics items** (§15, threads 11 and 13): the thermal wind omits the p₁S₀′ term, which matters where S₀′ ≠ 0
  (the 'solar_rz' interior); with the diffusionless thermal wind there are small-scale growing modes peaking at
  l ≈ lmax that do not converge in lmax (physical or not is open); the σₙ (ln ρ₀)′ heat rows converge slowly in N near
  the surface for 'Wu 2005' with r_out → 1 at ricb = 0 (only `tests/heat_dr/row_check.py` sees it); two growing modes
  in the m = 3 n0 window of the solar benchmark (−260.6 nHz, growth +18.0; −250.5, +5.0) have not been examined.
- **numpy's Chebyshev interpolants and fits** (phases 6c, 6d): `Chebyshev.interpolate`, `chebinterpolate`, `chebfit`
  and `Chebyshev.fit` build their matrix by chebvander's recurrence, whose round-off grows as about 1e-16 j at degree
  j (above a 1e-13 cut from degree ~900) and which each derivative amplifies by up to j²; `chebfit` and `fit` also
  solve by an SVD (LAPACK gelsd) that fails to converge at some sizes, a LinAlgError (Gauss-Lobatto interpolation in a
  shell at M = 360 and 596 among others with the reference LAPACK, at other sizes with OpenBLAS). Interpolate by a DCT
  instead, `background_field.cheb_interp` at the first-kind points (`utils.chebco_f` on kore's grid) or a DCT-I at
  Gauss-Lobatto points (`wound_profiles`), and drop the coefficients below a fraction of the largest before
  differentiating: 1e-13 for the profiles, 1e-15 for Clairaut's g/x (the solar structures have content down to 1e-15).
- **Legendre polynomials in powers of x** (phase 6e): the coefficients of P_n in powers of x grow as about (1 + √2)^n
  with alternating signs, so a P_n converted to a numpy Polynomial (Legendre.convert, leg2poly) and evaluated by
  Horner on the matrix X has a round-off that grows by a factor of about 2.4 per degree (P_n(X) 6e-12 of its largest
  entry off at n 15, 4e-7 at n 29, 8e-5 at n 35). Keep angular factors as numpy Legendre series
  (`mag_algebra._poly_mats` sums the P_k(X), P_k'(X) of `angular.Algebra.legendre`); a Polynomial is fine only for low
  degrees (x, 1 - x²).
- `params_default` says `rpower_pp`, `rhopower_pp` are "not read by the current pipeline", but `utils4pp.setup_grid`
  uses them as the enstrophy weight `const0`.
- `utils4pp` calls `mp.set_start_method('fork')` at import (needed on macOS).
- Stale upstream leftovers: `tools/submit4.sh` calls the removed `postprocess.py`; the top-level `README.md` is
  upstream's, except its badges (tests on main, docs, license) and (NEW 2026-11-09) its Documentation section, which points to the online manual
  (repepo.github.io/kore, published by `.github/workflows/docs.yml` from the default branch) in place of the old setup
  steps.
- `rvisc`, `hvisc` are not used; they are only echoed to `params.dat`.
- `Ek`, `Etherm`, `Ecomp`, `Le` and `timescale` are read only by `set_scales`, which `parameters.py` calls before its own
  values: setting them after it changes nothing (ramp `ViscosD`, `ThermaD`, ...; `tools/setpar.py` warns).

## 13. Recipes

- **New parameter**: default and comment in `params_default.py` (a name without a default there gets the warning of
  `check_names`); if it should be recorded, append a tuple at the end of `spin_doctor.param_table()` and update the
  table in `docs/output.md` and the names of `tools/get_data.py` (`tests/tools` compares them; the same for a new column
  of `flow.dat` and the other tables). A parameter that changes no operator (solver, forcing, boundary conditions, output) goes
  into `submatrix_io.OPERATOR_INDEPENDENT`, or changing it between `submatrices.py` and `assemble.py` stops the run.
- **New radial profile in an operator**: a function `f(r, rpower)` and its `fX(r, Dorder)` in `radial_profiles.py`, a
  3-letter code in `proffdir` (and in `ut.odd_profiles` if odd), labels in `submatrices.py` with the vector parity of the
  unknown's column (vP for P and s, vT for T, as the neighbouring labels), the term in `operators.py` with the L²/L³
  division, the block in `assemble.py`. Check with `ru_insane.py` and a balance in `spin_doctor`.
- **New feature, or a new stop**: its row and column in `docs/compatibility.md` (the grid, the forcings table and the
  conditions under it): edit the cells in `tools/compat_tables.py` (and the codes, notes and requirements in the page
  itself), run it, and `--check` before a push. A test mark only where a suite checks the combination against theory, an
  independent computation, conductive_IC or a closing budget (§11), not where it only compares a run with an earlier
  one. The page's tables are HTML in markdown (`md_in_html`, `<div class="compat" markdown="1">`), styled by
  `docs/stylesheets/compat.css` (`extra_css`).
- **New diff-rot type**: its shapes {n: fₙ} in `diff_rot.shapes()`, its name appended to `diff_rot.TYPES` (the index is
  written to `params.dat`), the table in `docs/background.md` (Profiles); run `regress.sh` and `gen_check.py`.
- **New background type**: a structure is a subclass of `radial_profiles.structure` with `logrho`, `gravity`,
  `pressure` (each `f(r, rpower)` returning `r**rpower * value`) and `dlog_p`, `dlog_rho` (`f(r)`), valid on [−1, 1]
  with the right parity if ricb = 0, plus its entry in `structures`; a stratification or diffusivity is a branch in
  `background.gradS` or `background.diffusivity`. In each case append the name to its tuple in `background.py` (the
  index is written to `params.dat`), add the checks it needs to `background.check()`, and document it in
  `params_default.py` and `docs/background.md`; run `regress.sh`.

## 14. History (this branch, newest first)

Interrupts of tests/sheet and tests/bgflow (2026-11-09): every process of `sheet_check.py` and `bgflow_check.py` runs
through `step()`, in a session of its own with stdin from /dev/null; an INT, TERM or HUP gives each step that runs one
SIGINT (two make Open MPI's mpiexec quit without stopping its ranks; `OMPI_MCA_odls_base_sigkill_timeout` 0 unless
set), a SIGKILL to its process group 0.5 s later (3 s for mpiexec) if it has not ended and again once it has (the
workers of submatrices.py's Pool), starts no step after it, and ends the check with 128 + the signal; any other end of
the main thread (an error, the end of the check) stops the runs too (`_abort`, before concurrent.futures joins the
pool); bgflow's `eigs` looks for an interrupt between its long steps; the two `run_all.sh` (INT, TERM, HUP) wait for
the check to end and then end by the signal they received, or that the check alone received (its exit 129, 130, 143:
a run_all.sh started in the background by a script ignores INT); an interrupt takes up to about 3 s (mpiexec's grace:
on chandler, MPICH's hydra, the ranks ended at the SIGKILL);
`6f6f2ba` the drag at the outer wall on a deformed body (2026-11-09): `kappa_cmb` and `topo_drag = 'rekier'` with
`centrifugal = 1`, any shape (check_drag's stop removed): the drag acts on the slip along the body's wall through
K_eff (`topo.wall_metric`, `wall_tensor`; §4), Dtopo and δσtopo in spin_doctor too; `tests/drag/deformed_check.py`
(run_all.sh runs it beside drag_check.py), drag_check's stop case removed; docs: drag.md 'On a deformed body',
drag-equations.md 'On a deformed body' and its checks, centrifugal-equations.md, background.md, checking.md,
compatibility.md (Kd and Kt with Cf and Sph tested); the rotating solids still stop with a deformed body;
`9d2d3355` to `d45ac19b` (2026-11-09): the docs rewritten and simplified; `main` is the default branch (the GitHub test
runs on pushes to it);
`7af05852` the conductive_IC port, phase 11 (2026-11-09): `docs/compatibility.md` ('Combining features'), which features combine,
which stop and which a suite checks: a grid of feature pairs and a table of the forcings, with the conditions of each
forcing under it, written by `tools/compat_tables.py` (`--check`); the audit behind it: an inviscid fluid in a shell
stops in `background.check` whatever the features (its wall rows left A − λB singular), forcing 7 with m = 0 and an
inviscid fluid stops (its flow is 0), a forced run (at any frequency, also those of forcing 11's scan) of an entropy or
a composition without diffusion stops where the fluid does not carry it round (`background.check_transport`, from
submatrices.py; the rows hold i(w + mΩ)s): w + mΩ = 0 everywhere (A singular), w + mΩ of both signs in the fluid (a
critical layer) or a background flow with a meridional circulation (the scalar resonates on its closed streamlines);
any forcing at frequency 0 needs walls that stay in place, bco and bci 0 or 1, as forcing 12 did (13 at bci 2 gave
resid𝐋 1); the settings of forcings 6, 7, 9, 10, 11, 13 and 14 stop at startup (`check_forcing`; assemble.py wrote no
right-hand side for some and solve.py then solved an earlier run's `B_forced.npz`, which assemble.py now removes first;
forcing 11's own conditions, in `reynolds_stress.py`, came only after every operator was built), with new stops for 9
and 10 at bco 2 where they move the outer wall (the flow went on the stress row) and for 10 with m = 0 (an IndexError);
and the stops stricter than needed are relaxed: bci in a full sphere for forcings 7 (m 2), 9 and 14, bco with an
inviscid fluid for forcings 1, 7 (m 2), 8 and 14, a thin layer and the resolved mantle (7 and 9 wrote no right-hand
side, the others stopped; `utils4pp.walls_move` only with viscosity, so that an inviscid run at bco 2 gives bco 0's
files bitwise), forcing 9 with the drag where the outer wall stays in place (`moves_radially`), forcing 13 at frequency
0 between stress-free walls with `kappa_cmb`, the rigid rotation that a differential rotation or a background flow needs
at a thin layer only where it conducts (c1 ≠ 0; so too INDU_ROT_SOLID in `utils4pp.magnetic_powers_grid`: counted at a
layer that does not conduct as well, it would leave resid𝐄 4e-4 instead of 3e-8 with 'conical'), centrifugal's stops on
the flow's rows only
with hydro = 1, bgflow's density check of a file only where its flow has a poloidal part (a rotation's file, whose P is
round-off, stopped); the Ferraro check of a uniform ΔΩ is 0 (it was round-off over round-off, up to 1, a false warning
and a stop with 'refuse'); thermal_wind's message says Beyonce ≠ 0, as it tests; resid𝐋's scale with ViscosD ∫ρ0 r³|T|
(a steady run on which no torque acts), and no resid𝐋 at a wall of zero radial stress of a viscous fluid
(`rotdyn.L_case`: the fluid crosses it); resid𝐄 of an inviscid fluid in a deformed body (centrifugal = 1) whatever
bco (it was nan at bco 0);
`tests/forcing/stops_check.py`, the new cases of rows_check.py, couette's and drag's runs that stopped before (drag's
checks NaN-safe), cic_forcings comparing the entropy's file ('thermal'), rsf's spectrum check in a full sphere (sp_fs)
and with a field and a conducting inner core (sp_mag), the checks of heat_dr's run_all.sh exit with 1 on a FAIL (its
parity WARNINGs too); faster tests: the
magnetic, centrifugal, sheet, bgflow and heat_dr suites run their checks or build their runs side by side (NPAR, capped
by the memory; the magnetic sympy checks in parts, `--part k/K`), `tests/run_suites.sh` runs the suites side by side
(one BLAS thread per process, Open MPI unbound), the ci test without pytest; the stops and conditions on each docs page
with their exact scope; `theprofile.data` and `poly.n1_gamma3.h5`, which nothing read, removed; the README's old
setup steps replaced by links to the online manual;
`6349f974` the conductive_IC port, phase 10 (2026-11-08): the solution files as `.npy` by default (`field_format`, `bin/field_io.py`,
every reader: spin_doctor, bgflow.py, just_induction.py, the tests, the tools, koreviz), the JSON parameter file
(`export_json`, `load_json`, from conductive_IC 08c004a by Ankit Barik), eigenmode tracking (`track_target`), the run
tools (`ramp.sh`, `setpar.py`, `dodirs.sh`, `subramp.sh`, `reap.sh`, `get_data.py` with every column named,
`getresults.py`, `topo_iterate.sh`), the plotting tools that read solutions (`plot_field.py` 'u', 'b', the buoyancy and the
curls, `load_solution.py`'s inner core, `plot_rad_con_tor_profile.py` with the density), `no_conv_solution` for a forced NaN, the commented BLR options for forced runs,
`tests/tools/`; every run bitwise as before;
`1e9880f6` the conductive_IC port, phase 9 (2026-11-08): the background flow (`bgflow = 1`, `bin/bgflow.py`): an axisymmetric
flow with poloidal and toroidal parts, and the entropy and composition it carries, from expressions or from a file that
`bin/bgflow.py nmax [nonlinear]` writes from a forcing-12 run (linear, or nonlinear by Newton's method; conductive_IC's
files are read too), for any density, the full sphere, the entropy, the composition and the field (conducting solids
turning with it), with exact angular couplings (the recurrence algebra) instead of conductive_IC's quadratures; Wbg,
Wbg_im, Ensbg, Wbg_thm, Wbg_cmp in spin_doctor's budgets (flow.dat 46–48, thermal.dat and composition.dat 8, params.dat
155–156), resid𝐋 with it (axial and equatorial), the sha256 of `bgflow_file` in submatrix_io's check,
`background.centre_check` on its scalars, `tests/bgflow/`, `docs/bgflow.md`, `docs/bgflow-equations.md`; stops for
centrifugal, rotdyn, topo_drag 'rekier', an inviscid shell, and the writer for a run with a background field;
`93a8789` the conductive_IC port, phase 8 (2026-11-08): the composition (`compositional = 1`, section c after the fields, so runs
without it are unchanged: ρ₀(∂ₜc + u_r dC/dr) = CompoD ∇·(D ρ₀ ∇c) and the buoyancy BeyonceC g c, conductive_IC's
BV2_comp; `comp_stratification` 'linear', 'inverse square', 'two zone', 'user'; D in every diffusivity type; with the full
sphere, diff rot, the deformed body, fields, the solids, the drag, forced runs and hydro = 0), the stratification 'two
zone' (conductive_IC's 'two zone' heating), resid𝑐 and Wcmp in spin_doctor (`composition.dat`, flow.dat 44–45,
params.dat 142–154), the composition in ru_insane, `units.set_physical(..., D=)`, `tests/composition/`;
`cdda756e` the conductive_IC port, phase 7b (2026-11-08): forcings 1 (the wave-like tide, any m of degree 2, with Gaspard), 2 (the
Enceladus eccentricity tide, with w and ricb fixed) and 8 (the libration in the frame of the mantle, the Poincaré force
at its size, any walls) under conductive_IC's numbers, 3, 4, 5 stopped (`background.NOT_PORTED`), `check_tides`; the
lifted right-hand side −A x (`assemble.lifted_rhs`, `utils.lifted_flow`), the body force's power, enstrophy term and
torque in spin_doctor (flow.dat 42–43), `tests/cic_forcings/`, `docs/libration.md`, the √2 in tidal-forcing.md's table
of u_r (kore's Y_l^m have ∫|Y|² = 4π/(2l+1) for every m);
`dde6f94` the conductive_IC port, phase 7a (2026-11-08): forcing 12, the steady rotation of the walls (spherical Couette flow;
`background.check_forcing`, also stopping unknown forcing numbers), resid𝐋 for forcings 7 (m = 0), 12 and 13, residσ and
resid𝐄 relative to Dvisc where the walls drive the flow, residω, resid𝐛, resid𝑠 0 when all their terms are 0,
`tests/couette/`, `docs/couette-flow.md`;
`893bc76` the conductive_IC port, phase 6e (2026-11-08): the background's angular factors (mag_algebra, the sheet push, the
centrifugal rotation force) as Legendre series summed over the recurrence's P_k(X), not in powers of x, §12;
`52fc122` the conductive_IC port, phase 6d (2026-11-08): numpy's remaining Chebyshev interpolants and fits replaced by DCTs
(Clairaut's g/x, the amplitudes inside a conducting inner core, model S hydrostatic's ρ, g, ln p, row_check's fits, and
the wound field's k and ru_insane's S_tw, whose SVD failed at some N), §12;
`8b841ed` the conductive_IC port, phase 6c (2026-11-07): the push of B0's current sheets, toroidal and poloidal, at slipping
viscous walls in the stress rows (`assemble.bc_u_sheet`, `background.sheet_jump`), the stop kept for an inviscid fluid,
the round-off of B0_tor 'user' profiles, the interpolants of B0's profiles by a DCT, `tests/sheet/`;
`4b44eb0` the conductive_IC port, phase 6b (2026-11-07): B0 'dipole', a fluid slipping along a conducting inner core (both kinds,
condition C.I in the core's electric-field row), the libration of the mantle with a stress-free inner wall, finite
'general' and 'user' profiles, the stop for a toroidal B0's current sheet at a slipping wall, `tests/inner_core/`;
`ddd59a2` the conductive_IC port, phase 6a (2026-11-07): forced runs with a field (every field row; 'conducting, Bessel';
walls that the forcing moves, `forced_walls` 'rigid' or 'deforming', the deforming solids' induction; the magnetic
pressure, the solids' work and the Bessel core in resid𝐄; one permeability at a wall that moves), `tests/forced_field/`;
`ad16ce4` the conductive_IC port, phase 5b (2026-11-06): the drag at a stress-free outer wall (`kappa_cmb`, `topo_drag = 'rekier'`
with the 4π² of Rekier et al.'s B1; `bin/topo.py`), with the rotating solids too (rotdyn with bco = 0, the drag on the
slip relative to the mantle), Dtopo and δσtopo (per unit κ without a drag), `tests/drag/`, `docs/drag.md`,
`docs/drag-equations.md` (params.dat columns 135-141, flow.dat 40-41);
`829dd49` the conductive_IC port, phase 5a (2026-11-06): the rotating solids (`rotdyn = 1`: ω_M, ω_IC, φ; viscous and magnetic
torques for any poloidal B0 and any walls, gravitational coupling, relaxation; `bin/rotdyn.py`), resid𝐋 (axial and
equatorial, with or without the solids), the solids in resid𝐄, `rotdyn.dat` (`tests/rotdyn/`,
`docs/rotating-solids.md`, `docs/rotating-solids-equations.md`; params.dat columns 128-134, flow.dat 29-39);
`9d205ca` the conductive_IC port, phase 4 (2026-11-06): the resolved conducting mantle (`mantle = 'resolved'`, sections fm, gm,
any smooth σ(r), the slipping-fluid term, co-rotation with ΔΩ(1), its energy and Ohmic loss in resid𝐄; `tests/mantle/`;
params.dat columns 123-127, magnetic.dat 12-16 with it);
`e23afcb` The conductive_IC port, phase 3 (2026-11-06): thin conducting walls (mantle, innercore 'TWA') with the slipping-fluid
terms for any poloidal B0, B0_cmb, εTWA, the wall energies in resid𝐄; resid𝐄 also for stress-free impenetrable walls,
and fixed for a conducting inner core turning with ΔΩ (`tests/twa/`; params.dat columns 114-122, magnetic.dat 12-16);
`be668dd` the conductive_IC port, phase 2: `hydro = 0`, the field (or the entropy) without the flow, its free decay
checked against exact rates, and the kinematic induction `just_induction.py`, `solve_ind.py` (`tests/decay/`; params.dat
column 113); `342c18a` the conductive_IC port, phase 1: Mag/Kin = Hendrik ME/KE (as conductive_IC 7660861), the wall rows
of forcings 6 (also at a stress-free ICB), 7 with m = 2 and 9 (as conductive_IC 35fff71), B0_ic 'full sphere' renamed
'frozen currents' (the old name stops a run), `cnorm` as a numpy number, `tests/forcing/`;
`66a97b50` the exact spheroid with a field (step 2: the Lorentz force and the magnetic diffusion with the whole metric,
insulating walls matched to the exact potential field of the spheroid, a conducting inner core with mu_i2o = 1; budgets to
round-off); `06256694` the exact spheroid, `centrifugal_shape = 'spheroid'` (step 1: one flattening at any f < 1, a linear
stretch, exact for a homogeneous fluid; `tests/centrifugal/spheroid_check.py`);
`25fd581f` the dissipative terms of the deformed body (phase 2: the viscous force with no-slip and stress-free walls, the
heat and magnetic diffusion with their wall rows, a conducting inner core, the metric terms of a differential rotation);
`57970a41` the centrifugal deformation with a field (phase 1b: the metric Lorentz force and induction, the insulating
walls on the deformed body, a conducting inner core with mu_i2o = 1, the free core nutation check `fcn.sh`);
`9d245cd2` the centrifugal deformation, `centrifugal = 1` (`bin/centrifugal.py`, `tests/centrifugal/`,
`docs/centrifugal-equations.md`);
`80c2f233`, `03213082` docs/magnetic.md (when the ignored Lorentz force of B0 on ΔΩ matters, a checklist for solar
inertial modes with `B0_wound`); `ab2c9201` `B0_wound`, the toroidal field wound by ΔΩ and balanced by diffusion;
`71259d5c` diff_rot_type 'ferraro'; `b2d01e55` mag_algebra drops round-off coefficients, display math in the docs;
`5c9127f7` the induction by ΔΩ, B0 'confined', 'general' and 'ferraro', `bin/mag_algebra.py`, the Ferraro check, the full
energy budget with the wall flux; `dfb9c2fd`, `713b4c2c` `docs/derivations.md` and `tools/derive/`; `43113392`
`PETSc.COMM_WORLD` in solve.py (slepc4py 3.26); `512fb493` the merge of anelastic_mag (the magnetic field, steps 1–5)
into anelastic_log;
`71b0b0f` the magnetic field in a full sphere, inviscid runs, 'G21 dipole', 'Luo_S1', 'Luo_S2', FDM of degree 2
(step 5); `683efac` the toroidal B0 (step 4); `c55a8d9` the magnetic field with a variable density (step 3,
`tests/magnetic/`); `c457a28` the magnetic field, steps 1 and 2 (insulating walls, conducting inner core;
`docs/magnetic.md`); `a2abc05` the spin-over test GitHub runs on every push (`tests/ci/`, `.github/workflows/anelastic_log.yml`; the
stale `main.yml`, `examples/kore_test.py` and `examples/spinover/` removed); `f810fd9` `docs/solver.md`; `3402fbf`
`diff_rot_hold` removed; `b0dd868` the docs in the new background names, `docs/background.md`; `61506f8` 'user'
diffusivity evaluated at |r|;
`49a4575` the background chosen by `structure`, `stratification`, `diffusivity` (`bin/background.py`), `r_out` for
`aux0`, the file-based models removed, the smooth Sun example without `diff_rot_hold` at δ_cz −4e-7; `1e808d8`
`bin/units.py`, the unread-name warning, Re λ / Im λ headers, the rotation file in params.dat; `1aca07b`
`diff_rot_term_amp` and the params.dat layout restored, the old diff-rot names stop a run; `135b39b` README authors;
`4b8e3c8` differential-rotation tests; `6019425` the differential rotation in `bin/diff_rot.py` on the angular algebra
of `bin/angular.py` (`diff_rot_ampl`, 'user' profiles, every degree from the algebra); `73c64a1` forcing 11 (random
Reynolds stresses); `2e61b3e` smooth Sun: structure 'model S hydrostatic', δ_cz and δ_sf, `drop_tol`, the HMI
2010–2026 mean; `9aff8bf` faster spin_doctor (Chebyshev tables, serial diagnose);
`b4598010` radiative interior, whole-Sun Model S and tabulated fits, ricb 0 for 'model S', `diff_rot_hold`;
`dc2790d3`, `bc6c7eed`, `e42b55df`, `16be479f` docs; `f9b733ce` diff-rot options in params.dat; `92a0d020` any Pₙ,
'solar_P4', 'tabulated', amplitude controls, Model S density; `18641206` Gaspard in diff rot, thermal_wind 2 and 3,
ThermaD = 0 with diff rot, spin_doctor diff-rot powers (fixed the old `flow_worker` crash); `be2a4db5` diff-rot terms
in the heat equation; `652e1490` 'smoothed sun'; `26d595eb` 1/L³ on the diff-rot operator; `6799c90e` solver defaults
(Ruiz auto, BLR off); `8d44f177` rpower/rhopower defaults; `dc140f06` gi1 parameters; `357f83b7` spin_doctor tables;
`42f4ccca` rhopower = 1; `102bccc8` relative tolerance in `chebco_f`; `d0da30dc`, `8eb16917`, `2e64687c` cleanup
(stale scripts, magnetic/compositional code, body forcings 1–5, 8 removed); `2cd8f2be` thermal path fixed, budgets;
`00bc67e3` ru_insane; `6490a1b3` conductive_IC solver port, KE normalisation; `ec94dd6a` ρ₀ weight in dissipation,
Gauss–Legendre quadrature. Earlier: the `differential_rotation` merge `5446485f`.

## 15. Project threads and their end results

The branch was built and checked in the claude.ai project "kore - anelastic_log". Its first threads, oldest first, with
what each one left behind (in the parameter names of that time: `docs/background.md` translates them); the later work
is in the docs and the commit messages. `project-files/...` is the project's shared folder (`/mnt/project-files/`
inside a project session, not in this repo); run folders are on the Mac (`~/lab/...`) or on mongo.

1. **Gas giant tidal forcing literature** (Oct 3–8, left open). Literature notes
   (`project-files/literature/tidal-forcing-gas-giants.md`) and a derivation of the equilibrium/wavelike split for
   kore's per-unit-mass rows (`wavelike-split-derivation.md`, LaTeX in `latex/`, sympy checks in `checks/`;
   `anelastic-tidal-forcing-kore.md` is superseded). With a potential flow carrying the bulge, only three sources force
   the waves: toroidal l = 3, poloidal l = 2 (∝ ρ₀′/ρ₀) and entropy l = 2 (∝ s₀′); the ω² part drops out. Its §10 gives
   kore's harmonic normalisation and signs. Nothing was implemented: kore has no tidal body force.
2. **Solar eigenmodes with kore** (Oct 5). Feasibility notes, `project-files/sun/solar-eigenmodes-notes.md`, written
   before any code check: density contrast manageable, results in the unstable convection zone depend on the chosen
   superadiabaticity and diffusivities, radiative zone and tachocline stiff, differential rotation the main gap. Threads
   12, 13 and 15 did the solar work.
3. **kore anelastic_log code summary** (Oct 8). First summary at the merge `5446485f`
   (`project-files/kore/anelastic_log-code-summary.md`, refreshed in thread 10, superseded by this file) with a bug
   list, most of it fixed since (the file-based and Boussinesq models still lack `logrho`).
4. **Testing without forcing or differential rotation** (Oct 8–9). Same eigenvalues as the pre-merge code. `ec94dd6a`
   (ρ₀ weight in the viscous dissipation, Gauss–Legendre quadrature); `6490a1b3` (conductive_IC solver port:
   `petsc_opts`, Ruiz, memory-mapped rows, KE = 1, Gatherv assembly; runKore.sh runs spin_doctor into `out2`). Then the
   thermal-path review, `project-files/kore/thermal/thermal-entropy-review.md`: the shell entropy coupling matches an
   independent solver to 1e-9, and `2cd8f2be` fixed `kappressX` (any ThermaD > 0 crashed), the (κp₀)′s′ coefficient (2,
   should be 1), the parity of p₀ dS/dr and Mlam at ricb = 0 and the missing −iω_f entropy term of forced runs; it also
   added the buoyancy work to residσ and the entropy budget resid𝑠, divided the heat rows by p₀ when ThermaD = 0, and
   changed MUMPS `cntl_1` from 1e-8 to 0.01 (1e-8 made the residuals grow with N). anel10 with thermal = 1 then closes
   all three budgets to 1e-8 or better (modes 0–7) at N = 480, lmax 101.
5. **Balance checks for anelastic LBR** (Oct 8–9). In `project-files/kore/balance/`: `energy-balance-derivation.md`
   (energy, frequency and entropy budgets from the momentum equation; the pressure does no work because the compression
   work cancels the pressure part of the buoyancy; the budgets read off the matrix rows, `ebal.py`; a convective case,
   conv1, with every term O(1)); `anelastic-limitations.md` (sound filtering, of size ω max(ω, 2Ω, N)/(c²k²); the
   dropped pressure-buoyancy term, of size N²ℓ/g; and the background must be a possible hydrostatic state, c² > 0, which
   kore cannot check. Both truncations are below 1e-3 for inertial modes of the Sun and the Earth's core, 2–25 % in
   Jupiter, and too large for low-order g modes near N in a strongly stratified zone);
   `viscous/viscous-operator-accuracy.md` (the 1e-8 anti-Hermitian part of the poloidal viscous block came from
   cancelling ln ρ₀ derivatives near the wall; rows times ρ₀ cure it). Code: `00bc67e3` ru_insane.py; `d0da30dc`,
   `8eb16917`, `2e64687c` the cleanup (−5956 lines, A and B bitwise identical on 15 configurations;
   `project-files/kore/cleanup/NOTES.md`); `102bccc8` relative cut in `chebco_f` (the absolute 1e-9 cut became the error
   floor for small products); `42f4ccca` rhopower 1, residω. The peel rule (§6) was set here.
6. **Align spin_doctor with conductive_IC** (Oct 9). `357f83b7`: list-based output tables, residσ in flow.dat.
7. **Gravito-inertial mode with buoyancy and viscosity** (Oct 9). `dc140f06`: the checked-in gi1 `parameters.py`, a mode
   damped 44 % by viscosity and 56 % by buoyancy work (the split moves with ThermaD; see its header comment).
8. **Full-sphere ricb=0 check on mongo** (Oct 9–10). `project-files/ricb0_check-20261009T221302-fcab.md` (the fullest of
   three versions): ricb = 0 eigenvalues sit where ricb = 0.05, 0.02, 0.01 extrapolates (differences ∝ ricb³) for m =
   0–3 and both symmetries; residuals of 1e-7 at N = 192 are resolution (1e-14 at N = 384), hence ~2N for a full sphere;
   eigenfunctions go as r^l at the centre; rpower_u 5, rpower_v 3 is the right pair (lower breaks ricb = 0, higher costs
   residuals). `8d44f177`: rpower/rhopower defaults; cvfmax skips r < 0.01 (round-off at the innermost node). Runs in
   `mongo:/nvm/lab/anel_dev`.
9. **Ruiz and l-dependent row scaling** (Oct 9–10). `project-files/kore/scaling/scaling-study.md`: l-dependent row
   factors cannot fix the conditioning (the disparities are boundary rows against operator rows, and Chebyshev index
   within a block) and add nothing on top of Ruiz, which takes κ₂(A − τB) from beyond double precision to ~5e7 at N =
   128 and helps except at exactly ThermaD = 0 (5–1e4× worse). The ThermaD = 0 spurious modes come from dS/dr ≠ 0 at the
   inner boundary: make dS/dr vanish there (or at r = 0) and prefer a small ThermaD > 0 with prescale on (ThermaD 1e-9
   at ricb 0.02 needs N ≥ 560). `6799c90e`: automatic prescale, BLR off, the advice as comments in `params_default.py`.
10. **Refresh the code summary** (Oct 10–11). Thread 3's summary brought to `6799c90e`, correcting one claim: the
    diff-rot operator had its density terms; its real bug was the missing 1/L³ (fixed in thread 11).
11. **Differential rotation equations and benchmarks** (Oct 11–12). `26d595eb` (1/L³ on the diff-rot operator: every
    earlier diff-rot run is wrong), `be2a4db5` (diff-rot terms in the heat equation), `18641206` (Gaspard factors,
    thermal_wind 2 and 3, ThermaD = 0 with diff rot, spin_doctor `diff_rot_setup`, ru_insane diff-rot tests,
    `tests/heat_dr`, `docs/thermad0.md`). Benchmarks: rigid-rotation offset identity, Guenel et al. 2016 conical m = 0
    (ω ≈ 1.817). Notes in `project-files/kore/diffrot/README.md`. Its final summary, the project doc "kore
    anelastic_log: differential rotation, final summary", lists 15 open problems: thread 13 addressed the first four (no
    switch for the u_r ∂S_tw/∂r term, Ω limited to P₀ + P₂, two Bekki et al. results not reproduced, the 'smoothed sun'
    density); what remains is in §12.
12. **Sun-like background profile** (Oct 11). `652e1490`: model_type 'smoothed sun' (smoothed Model S ρ, g, p; odd erf
    top-hat dS/dr; ν, κ steps; length unit aux0 R☉, Beyonce 4.90e4 for the Carrington rate; N ≳ 11/w1), 'user def'
    matrices bit-identical. Notes in `project-files/sun/solar-profile.md`. 'Boussinesq' was found already broken and
    left for the owner.
13. **Bekki 2024 gap: plan for review** (Oct 12–13). Step 1
    (`project-files/kore/diffrot/checks/bekki/step1/STEP1-SUMMARY.md`): no growing modes were missed, the entropy
    coupling and the units are right except c_p (the papers use 4.17e8, kore runs used 3.46e8), and the gap comes from
    the radial shear of Ω in the lower convection zone, absent from a radially uniform P₀ + P₂ profile. Step 2:
    `92a0d020` (any even Pₙ by quadrature, 'solar_P4', 'tabulated' Ω(r, θ), amplitude controls, `thermal_wind_radial`,
    `solar_density = 'model S'`) and `f9b733ce` (diff-rot options in params.dat). With the observed Ω (Mukhopadhyay et
    al. 2025 data) the five Mukhopadhyay et al. 2025 modes agree within 0.2 nHz in frequency (P₂ only: off by 0.8–9 nHz,
    and the high-latitude mode damped instead of growing); the convective onset is |Δθ_s| 513 (Mukhopadhyay et al. 2025
    Ω) and 658 (Larson & Schou) in the papers' c_p, against Bekki's ~500. Runs in `~/lab/kore_step2_runs`; tables in
    `docs/heat-equation.md` §8 (commit 1fbe5f08).
14. **Differential rotation docs review** (Oct 13). `16be479f` (user guide `docs/differential-rotation.md`,
    repo-relative references, solar benchmark scripts); `e42b55df`, `bc6c7eed`, `dc2790d3` (math that renders on GitHub:
    no `\tag`, plain-text equation numbers).
15. **Inner boundary scan for Mukhopadhyay modes** (Oct 13). `project-files/REPORT.md`, `ricb_scan.png` (runs in
    `~/lab/ricb_scan`): with the radiative interior of Mukhopadhyay et al. 2025b, the inner radius changes no mode by
    more than 0.01 nHz once it is below ~0.63 R☉, down to the full sphere; the dependence sits in 0.64–0.71 R☉. The
    radiative zone shifts the modes by 0.4–13 nHz and damps them; m = 2 Rossby lies near the ℓ = 2 RZ Rossby sequence
    and m = 1 high-latitude mixes with ℓ = 3 RZ modes. CZ-only frequencies differ from their Table 1 by 1–10 nHz
    (candidates: anelastic vs compressible, their thermal wind, their free surface). With an adiabatic interior instead,
    the modes kept moving as the boundary went deeper. `b4598010` ported `solar_rz`, the whole-Sun 'sun' fits,
    `diff_rot_hold`, `mk_rz.py` and docs §9. `background_fits.png` and `density_fit_vs_modelS.png` compare the fits with
    Model S: 'sun' within 1.7e-3 in ρ (worst at 0.70–0.72 R☉) and 2.2e-4 in p; the shell fit 2.1e-4 and 2.8e-5; 'fit'
    ~10 % wiggles and 48 % near the surface; `solar_rz` makes N² vanish ~0.04 R☉ below Model S's drop at 0.713 R☉, while
    the aux top hat leaks N into the convection zone up to ~0.8 R☉.
16. **Code guide for Claude** (Oct 13). This file; a copy is `project-files/kore/CLAUDE.md`.
