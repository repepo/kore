# Output files

A run writes all its files into the run directory ([Getting started](getting-started.md)). `solve.py` overwrites its
files on every run, while `spin_doctor.py` appends one row per solution to its own, so that a series of runs builds up
a table.

| file | written by | content |
| --- | --- | --- |
| `submatrices.npz` | `submatrices.py` | the radial operators and the parameters they were built with; later steps stop if the parameters have changed. With `write_mtx = 1`, also one Matrix Market file per operator |
| `A.npz`, `B.npz` | `assemble.py` | the matrices of the eigenvalue problem $A\mathbf x = \lambda B\mathbf x$ |
| `B_forced.npz` | `assemble.py`, with `forcing` > 0 | the right-hand sides of a forced run, one per column |
| `eigenvalues0.dat` | `solve.py` | the eigenvalues of the last solve, one per row: real and imaginary parts |
| `flow.npy` | `solve.py` | the solutions for the flow, one column per solution ([below](#solution-files)); also `thermal.npy` (entropy), `composition.npy`, `magnetic.npy`, `magnetic_ic.npy`, `magnetic_mantle.npy` and `rotdyn.npy` when those features are on |
| `no_conv_solution` | `solve.py` | written when no eigenvalue converged, or when a forced solution contains NaN ([Solver](solver.md#bad-solutions)) |
| `rsf_spectrum.dat`, `rsf_ke.dat` | `solve.py`, with `forcing = 11` | the power spectrum of the forced kinetic energy ([Convective forcing](forcing.md#reading-the-output)) |
| `timing.dat` | `solve.py` | the run time of the solve, appended |
| `eigenvalues.dat` | `spin_doctor.py` | the eigenvalues, appended |
| `track_target` | `spin_doctor.py`, with `track_target` 1 or 2 | the eigenvalue that the next run aims at ([below](#following-a-mode)) |
| `params.dat` | `spin_doctor.py` | the parameters of each solution (below) |
| `flow.dat` | `spin_doctor.py` | the kinetic energy and the checks of each solution (below) |
| `thermal.dat` | `spin_doctor.py`, with `thermal = 1` | the entropy budget of each solution (below) |
| `composition.dat` | `spin_doctor.py`, with `compositional = 1` | the composition budget of each solution (below) |
| `magnetic.dat` | `spin_doctor.py`, with `magnetic = 1` | the magnetic energies and budgets of each solution (below) |
| `rotdyn.dat` | `spin_doctor.py`, with `rotdyn = 1` | the rotation of the solids and the torques on them (below) |
| `parameters.json` | `parameters.py`, when it calls `par.export_json` | every parameter of the run ([below](#the-parameter-file)) |

`python3 bin/units.py` prints the eigenvalues of `eigenvalues0.dat` as frequency and growth rate in nHz
([Background model](background.md#units)). Rows with the same index in `params.dat`, `flow.dat`, `thermal.dat`,
`composition.dat`, `magnetic.dat` and `rotdyn.dat` belong to the same solution. Start a new `params.dat` rather than
append to one written by another version of kore, which may have different columns.

## `flow.dat`

Columns, counted from 1:

| columns | content |
| --- | --- |
| 1–3 | kinetic energy, and its poloidal and toroidal parts (eigenvectors are normalised to kinetic energy 1). Without the flow (`hydro = 0`), the eigenvectors are normalised to the magnetic energy plus the entropy and composition "energies" instead |
| 4 | the power of the viscous force |
| 5–7 | Peak ℓ, ℓ-Width and ℓ-Convergence: the degree with the largest kinetic energy, the width of the spectrum, and how fast it falls off |
| 8–11 | the enstrophy, and the Coriolis, viscous and buoyancy terms of its balance |
| 12–14 | position and value of the largest toroidal component of the curl of the viscous force |
| 15–17 | residens, residω, residσ: the relative residuals of the enstrophy balance and of the imaginary and real parts of the energy balance |
| 18–20 | the power exchanged with the differential rotation, its imaginary part, and its enstrophy term |
| 21–23 | the power of the pressure at the inner and outer walls, and the imaginary part of their sum (non-zero only where the walls move) |
| 24–25 | the power of the viscous stress at the inner and outer walls |
| 26 | the viscous dissipation |
| 27 | the relative residual of the identity 4 = 24 + 25 − 26 |
| 28 | irreg₀, in a full sphere: how far the solution is from smooth at the centre ([Regularity at the centre](numerical-method.md#regularity-at-the-centre)) |
| 29–38 | for `m = 0`, `symm = 1` and `m = 1`, `symm = -1`: real and imaginary parts of the angular momentum of the fluid, of the viscous torques on the mantle and the inner core, and of the magnetic torques ([Rotating solids equations](rotating-solids-equations.md#torques)) |
| 39 | resid𝐋, the relative residual of the angular momentum balance ([Rotating solids](rotating-solids.md#the-angular-momentum-check-residl)) |
| 40–41 | Dtopo and δσtopo: the power dissipated by a drag at the outer wall and its share of the damping rate ([Drag at the outer wall](drag.md#output)) |
| 42–43 | the power of the body force of forcings 1 and 8, and its imaginary part |
| 44–45 | with `compositional = 1`, the power of the compositional buoyancy, and its imaginary part |
| 46–48 | with a background flow, the power the perturbation draws from it, its imaginary part, and its enstrophy term ([Background flow](bgflow.md#output)) |

Columns that do not apply to a run are 0 or NaN.

## `thermal.dat`

| columns | content |
| --- | --- |
| 1 | the entropy "energy" |
| 2 | the advection by the background entropy gradient |
| 3 | the thermal diffusion |
| 4 | resid𝑠, the relative residual of the entropy balance |
| 5 | the work of the buoyancy force |
| 6 | residσ, the energy residual |
| 7 | the terms of the differential rotation |
| 8 | the work of the background flow terms (0 without one) |

## `composition.dat`

With `compositional = 1` ([Composition](background.md#composition)), the same columns as `thermal.dat`, for the
composition; column 4 is resid𝑐.

## `magnetic.dat`

With `magnetic = 1` ([Magnetic field](magnetic.md)). The energies are defined on
[Magnetic field equations](magnetic-equations.md#energy-budgets).

| column | content |
| --- | --- |
| 1 | ME, the magnetic energy in the fluid |
| 2 | Mdfs, the magnetic diffusion term |
| 3 | Indu, the induction work |
| 4 | resid𝐛, the relative residual of the induction budget |
| 5, 6 | Wlor, Wlor_im: the power of the Lorentz force, real and imaginary parts |
| 7 | MEic, the magnetic energy in the inner core |
| 8 | MEext, the energy of the field outside |
| 9, 10 | Ohm_f, Ohm_ic: the Ohmic losses in the fluid and in a conducting inner core |
| 11 | resid𝐄, the relative residual of the full energy budget |
| 12 | MEc, the part of the magnetic energy added by the centrifugal deformation or a thin layer |
| 13, 14 | E_w, Ohm_w: the magnetic energy and Ohmic loss in a thin conducting layer or a resolved mantle |
| 15, 16 | εTWA, the estimated relative error of the outer and inner thin layers |

The column Mag/Kin of the `spin_doctor` table is the magnetic energy over the kinetic energy, in the same units.

## `rotdyn.dat`

With `rotdyn = 1` ([Rotating solids](rotating-solids.md)), real and imaginary parts (columns 1–2, 3–4, ...) of:

| columns | content |
| --- | --- |
| 1–6 | the rotation rates of the mantle and of the inner core relative to the frame, and the angle of the inner core relative to the mantle |
| 7–8 | the gravitational torque on the mantle |
| 9–12 | the viscous and magnetic torques on the mantle |
| 13–16 | the viscous and magnetic torques on the inner core |

## `params.dat`

Columns, counted from 1:

| columns | parameters |
| --- | --- |
| 1–10 | `thermal`, `m`, `symm`, `ricb`, `bci`, `bco`, `forcing`, `forcing_frequency`, `forcing_amplitude_cmb`, `forcing_amplitude_icb` |
| 11–18 | `Gaspard`, `Beyonce`, `ViscosD`, `ThermaD`, `ncpus`, `N`, `lmax`, the run time in seconds |
| 19–36 | `r_out`, `aux1`…`aux5`, `visc0`, `hvisc`, `rvisc`, `rpower_u`, `rhopower_u`, `rpower_v`, `rhopower_v`, `rpower_pp`, `rhopower_pp`, `therm0`, `visc_pow`, `therm_pow` |
| 37–41 | `diff_rot`, `diff_rot_ampl`, `thermal_wind`, `diff_rot_type`, `dSdtheta` |
| 42–50 | `diff_rot_term_amp` |
| 51–56 | `thermal_wind_amplitude`, `thermal_wind_radial`, `diff_rot_frame`, `diff_rot_nmax`, `diff_rot_rdeg`, (unused) |
| 57–62 | `diff_rot_solar` |
| 63–66 | `structure`, the degree of the Model S fit, the fit window of `'old model S prototype'` |
| 67–74 | `solar_rz` (1 with stratification `'solar_rz'`), `solar_rz_dSdr`, `solar_rz_steps`, `solar_rz_core`, `solar_rz_os` |
| 75–80 | `diff_rot_window`, (unused), `solar_rz_delta_cz`, `solar_rz_delta_sf`, `drop_tol`, `diff_rot_param` |
| 81–82 | `stratification`, `diffusivity` |
| 83–95 | `magnetic`, `B0`, `B0_l`, `cnorm`, `Hendrik`, `MagnetD`, `innercore`, `N_cic`, `mu_i2o`, `sigma_i2o`, `B0_tor`, `B0_tor_l`, `cnorm_tor` |
| 96–107 | `B0_wound`, `B0_wound_eta`, `B0_conf_r1`, `B0_conf_r2`, `B0_conf_p`, `B0_param`, `B0_tor_param`, `B0_ferraro_nmax`, `mag_couplings` (0 closed forms, 1 algebra), `tol_B0`, `B0_ic`, `ferraro_policy` |
| 108–112 | `centrifugal`, `centrifugal_shape`, `centrifugal_q`, `centrifugal_f` (as used), `centrifugal_param` |
| 113 | `hydro` |
| 114–122 | `mantle`, `c_cmb`, `c1_cmb`, `muw_cmb`, `c_icb`, `c1_icb`, `muw_icb`, `mu`, `B0_cmb` |
| 123–127 | `r_surface`, `N_mantle`, the conductivity of a resolved mantle at the outer wall, `sigma_mantle_param`, the conductance of the mantle |
| 128–134 | `rotdyn`, `MoIZ_M`, `MoIZ_IC`, `gTorque`, `OmgtauIC`, `vtrq`, `mtrq` |
| 135–141 | `kappa_cmb`, `topo_drag` (1 for `'rekier'`), `topo_hrms`, `topo_lambda0`, `topo_H`, and the buoyancy frequency and flow frequency that the topographic drag used |
| 142–150 | `compositional`, `comp_stratification`, `comp_stratification_param`, `BeyonceC`, `CompoD`, `comp0`, `comp_pow`, `bci_compositional`, `bco_compositional` |
| 151–154 | `comp_two_zone` and `two_zone` |
| 155–156 | `bgflow`, `bgflow_Ro` |

- Options given by name are written as their position in the list of choices, counting from 0, in the order of the
  tables on the corresponding pages; −1 means "none". For example `structure`: `'8-term fit'` 0,
  `'old model S prototype'` 1, `'model S hydrostatic'` 2, `'Wu 2005'` 3, `'Boussinesq'` 4. `tools/get_data.py`
  (below) reads the file with named columns.
- A row ends, after a `#` that `np.loadtxt` skips, with any expressions of the run (`'user'`, `'general'`,
  `'ferraro'`, `sigma_mantle`, `bgflow_profiles`) and the names of the data files it read, with a short checksum.

## Solution files

The solution files hold the spectral coefficients of each field, one column per solution
([Numerical method](numerical-method.md)). By default each field is one binary numpy file of complex numbers:
`flow.npy`, `thermal.npy`, and so on. With `field_format = 'text'` each field is instead two text files,
`real_flow.field` and `imag_flow.field`. Text files are much larger and slower to read and write.

`spin_doctor.py`, the tools and koreviz read either format. In python,

```python
import field_io                            # with kore's bin/ on the path
x = field_io.load('flow', folder='path/to/run')   # all solutions, one per column
x2 = field_io.load('flow', 2)                     # solution 2 alone
```

returns `None` when the field has no file. A `.npy` file can also be read directly with `np.load('flow.npy')`.

## The parameter file

`par.export_json('parameters.json')` at the end of `parameters.py` writes every parameter of the run to one file,
exactly. To run the same case again, here or elsewhere, use as parameter file

```python
from params_default import default_params
par = default_params()
par.load_json('parameters.json')
```

## Many runs

A parameter study runs one case many times with one parameter changed. The scripts in `tools/` run such a series in
one folder, one step after the other (`ramp.sh`), or in one folder per run on a cluster, and collect the results into
one table.

| script | what it does |
| --- | --- |
| `tools/ramp.sh name value1 value2 ...` | runs the case once per value of `par.name` (e.g. `tools/ramp.sh ViscosD 1e-4 7e-5 5e-5`), from kore's top folder. The results accumulate in the `.dat` files. It stops when a step finds no solution. The environment variables `ncpus`, `ranks` (MPI processes), `opts` (extra solver options) and `MPIEXEC` can be set |
| `tools/setpar.py name value [file]` | sets `par.name = value` in `bin/parameters.py`, so that it takes effect after everything else |
| `tools/dodirs.sh prefix name mode start step end` | creates one folder per value of `par.name`, each with a copy of kore. With `mode` `e` the values are powers of ten, with `d` they are linear. Edit the paths at the top of the script first |
| `tools/dosubs.sh` (same arguments) | submits a batch job in each folder that runs `ramp.sh` there. Edit the job header in `tools/subramp.sh` for your cluster first |
| `tools/reap.sh prefix` | collects the `.dat` files of the folders `prefix*` into `prefix.par`, `.flo`, `.thm`, `.cmp`, `.mag`, `.rot` and `.eig` |
| `tools/get_data.py [prefix]` | reads them (or the `.dat` files of the current folder) into one pandas table with named columns, and writes it to `prefix.csv` (or `results.csv`) |
| `tools/getresults.py [prefix]` | in IPython, `%run -i tools/getresults.py prefix` gives the same table as `df`, and its most used columns as arrays |

Ramp a parameter that the run actually reads, such as `ViscosD`, not `Ek`: `Ek`, `Etherm`, `Ecomp` and `Le` are only
used by `set_scales()`.

### Following a mode

Along a ramp the eigenvalues move, and a fixed target can lose the mode of interest. With `track_target = 1` each run
aims at the eigenvalue stored in the file `track_target` (instead of `rtau`, `itau`), and `spin_doctor.py` replaces it
with the eigenvalue it found nearest to it, so the next run aims there. Small steps keep the same mode. Before the first
run, write the starting eigenvalue to the file (`echo "-0.1 1.0" > track_target`: real and imaginary parts), or let a
first run with `track_target = 2` pick its solution with the smallest ratio of poloidal to toroidal kinetic energy,
such as the spin-over mode. Tracking works for eigenvalue problems only.
