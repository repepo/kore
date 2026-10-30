# Output and postprocessing

## Solutions

`solve_nopp.py` writes the solutions to the run folder:

| File | Content |
|---|---|
| `eigenvalues0.dat` | Eigenvalue problems: one row per solution, damping $\sigma$ and frequency $\omega$ ($\lambda = \sigma + i\omega$). |
| `real_flow.field`, `imag_flow.field` | Real and imaginary parts of the flow's spectral coefficients, one column per solution. |
| `real_magnetic.field`, `imag_magnetic.field` | The magnetic field perturbation, with `magnetic = 1`. |
| `real_magnetic_ic.field`, `imag_magnetic_ic.field` | The field inside a conducting inner core (`'conducting, Chebys'`). |
| `real_temperature.field`, `real_composition.field` (and `imag_`) | Temperature and composition, when included. |
| `rotdyn.field` | Mantle and inner-core rotation rates and their angle, as complex numbers, with `rotdyn = 1`. |
| `timing.dat` | Total run time of `solve_nopp.py`, in seconds. |
| `no_conv_solution` | Written instead when no eigenvalue converged. |

The coefficients are ordered by spherical-harmonic degree, poloidal part first and then toroidal, with the Chebyshev coefficients of each degree together (see [Numerical method](method.md#unknowns)). `tools/load_solution.py` loads a solution into an interactive python session.

Eigenvectors are only defined up to a complex factor. Kore scales each one so that its kinetic plus magnetic energy in the fluid equals 1 (see [Solver settings](solver.md#eigenvalue-problems)); its phase is arbitrary. Forced solutions keep the amplitude set by the forcing.


## spin_doctor

```Shell
./bin/spin_doctor.py ncpus
```

`spin_doctor.py` computes, for every solution in the folder, the energies, dissipation rates, work of the forces and torques, and residuals of the energy and angular-momentum balances that show how accurate the solution is. It prints a summary table and appends the results to the `.dat` files below. With `hydro = 0` (free decay of the field) there is no `flow.dat`, and the flow residuals (resid0, resid𝐮, resid𝐋) are not computed.


### The summary table

| Column | Meaning |
|---|---|
| ★ | Solution number. |
| Damping σ, Frequency ω | Real and imaginary parts of the eigenvalue (the forcing frequency for forced problems). |
| resid0, resid𝐮, resid𝐛, residθ, resid𝐋, resid𝐄 | Residuals of the balances, see below. |
| Tor/Pol | Toroidal over poloidal kinetic energy. |
| Mag/Kin | Magnetic over kinetic energy. |
| Geo | Geostrophic fraction: the share of the energy of $u_\phi$ held by its average along the rotation axis (see below). Only for $m = 0$, `symm = 1`; `nan` otherwise. |
| \|𝚪\|mag | Modulus of the electromagnetic torque on the mantle. |
| \|𝚪\|magIC | Modulus of the electromagnetic torque on the inner core. |


The geostrophic fraction measures how close the azimuthal flow is to being invariant along the rotation axis:

$$
\text{Geo} = \frac{\int |\langle u_\phi \rangle|^2\,dV}{\int |u_\phi|^2\,dV}, \qquad
\langle u_\phi \rangle(s) = \frac{1}{H}\int u_\phi\,dz ,
$$

where the average is taken along each fluid column of height $H$, from the inner core (or the equator, outside the tangent cylinder) to the core-mantle boundary. Since the column average and the rest of $u_\phi$ are orthogonal, $1 - \text{Geo}$ is the share of the energy of $u_\phi$ in its $z$-dependent part. A torsional mode has Geo close to 1 (with Tor/Pol $\gg 1$ and Mag/Kin $\approx 1$); in the inviscid full sphere it is 1 to round-off. It is computed by `geostrophic_fraction` in `bin/utils4pp.py`, on a grid of 240 cells in $s$ and 48 Gauss-Legendre points in $z$ over the northern half of the fluid ($u_\phi$ is equatorially symmetric for $m = 0$, `symm = 1`). These points do not resolve the Ekman layers, so Geo describes the interior flow.

### Checking the results

The residuals compare terms of balances that an exact solution satisfies. They are computed from the solution alone, independently of the solver, and are the best test of a solution's accuracy. Each is the size of the imbalance divided by the largest term:

| Residual | Balance |
|---|---|
| `resid0` | The viscous dissipation computed in two ways, from $\int \mathbf{u}\cdot\nabla^2\mathbf{u}\,dV$ and from the rate of strain, plus the work of a rotating mantle and inner core on the fluid with `rotdyn = 1`, or of the walls set rotating by `forcing = 12` (the rotation rate times the viscous torque on each wall). `nan` if `Ek = 0`. |
| `resid𝐮` | Kinetic energy: $2\sigma$ times the kinetic energy against the viscous dissipation and the work of the Lorentz and buoyancy forces. With `forcing = 12` (steady, no body force) $\int \mathbf{u}\cdot\nabla^2\mathbf{u}\,dV$ vanishes, and `resid𝐮` is its size relative to the viscous dissipation. With `bgflow = 1` the balance includes the work of the background flow (see [Background flow](bgflow.md)). |
| `resid𝐛` | Magnetic energy: $2\sigma$ times the magnetic energy against the induction work and magnetic diffusion. |
| `residθ` | Thermal "energy": $2\sigma$ times it against thermal diffusion and advection. |
| `resid𝐋` | Angular momentum of the fluid: $\lambda L$ ($L_z$ for $m = 0$; its equatorial components for $m = 1$, with the Coriolis torque) against the viscous and electromagnetic torques on the mantle and the inner core, and the Maxwell stress of the background field on the fluid side of a wall where those torques leave it out: a toroidal field with $k \neq 0$ at the wall, and for $m = 1$ an insulating CMB with `B0 = 'axial'` and the ICB of a conducting or thin-wall inner core, or of one that holds currents or sources of $\mathbf{B}_0$ (`B0_ic = 'full sphere'`, `B0 = 'dipole'`) (see [Angular momentum balance](equations.md#angular-momentum-balance)). Only for eigenvalue problems with $m = 0$, `symm = 1` and $m = 1$, `symm = -1`, and for `forcing = 12`, where the steady torques on the mantle and the inner core cancel; `nan` otherwise. It is 0 when no torque acts at the walls (`Ek = 0` or stress-free walls, or a full sphere with a stress-free CMB, and no magnetic torques): the balance is then not tested, so judge such modes by `resid𝐮` and `resid𝐛`. |
| `resid𝐄` | Total energy of fluid, inner core, thin walls and the potential field outside: $2\sigma$ times kinetic plus all magnetic energy against the viscous dissipation, the work of the background currents ($\mathbf{J}_0\times\mathbf{b}$, plus buoyancy) and the Ohmic losses in the fluid, a conducting inner core and thin walls, written as $\eta\int|\nabla\times\mathbf{b}|^2\,dV$. Unlike `resid𝐛` it tests the matching conditions at the ICB and CMB. With `rotdyn = 1` it includes the rotational energy of the mantle and the inner core, the gravitational energy and the inner core's viscous relaxation, so it also tests their equations. At a stress-free wall it includes the power $2\,\mathrm{Re}\oint B_{0r}\,\mathbf{u}_t\cdot\mathbf{b}_t^*\,dS$ that the induction term carries through the wall (zero at a no-slip wall); a thin wall still needs a no-slip boundary there, and `rotdyn = 1` no-slip walls. `nan` for forced problems (including `forcing = 12`, whose energy balance `resid0` and `resid𝐮` cover), `'conducting, Bessel'`, thin walls with `mu` $\neq 1$, or `rotdyn = 1` with `vtrq` or `mtrq` set to 0. |

As a rule of thumb they should all be below about $10^{-4}$. If they are not, increase the resolution. Exceptions and caveats:

- The eigenvectors of the modes found away from the target are usually less accurate than the one closest to $\tau$. To get a particular mode as accurately as possible, place the target on it.
- The residuals do not detect a too small `lmax`. Keep `lmax` close to `N` (see [Solver settings](solver.md#resolution)).
- `resid𝐮` and `resid𝐛` cover the fluid only and test its two equations separately. The magnetic diffusion in `magnetic.dat`, $\int \mathbf{b}\cdot\nabla^2\mathbf{b}\,dV$, is not the Ohmic loss: it also contains the flux of magnetic energy through the ICB and CMB, which goes to a conducting inner core, a thin wall or the field outside. `resid𝐄` and `budget.dat` give the separate terms. In `resid𝐄` the Lorentz work and the induction term cancel, so it is usually much smaller than `resid𝐛`. It is weaker on each equation separately, so judge a solution by all of the residuals.
- With a thin wall that has both $c \neq 0$ and $c_1 \neq 0$, `resid𝐄` closes only to about $c\,c_1$ times the wall's share of the energy budget, because the thin-wall conditions are first order in the wall thickness.
- `resid𝐋` is relative to the largest term of the balance, so it can be large for a mode that carries almost no angular momentum, even a resolved one: in an inviscid full sphere with `B0 = 'axial'` some $m = 1$ modes have $|L| \sim 10^{-12}$ and `resid𝐋` $\sim 10^{-3}$ while `resid𝐮` and `resid𝐛` are at round-off.
- `solve_nopp.py` also prints a residual for each eigenvalue, $\|Ax-\lambda Bx\|/\|\lambda x\|$. It only shows that SLEPc converged, and stays small even for inaccurate eigenvectors. Judge accuracy by `spin_doctor`'s residuals.


### Result files

Each run appends one row per solution to the files below, in the same order in every file. Several runs in the same folder therefore build up a table of results, which `tools/getresults.py` can load.

**`params.dat`**, the parameters of the run (63 columns): hydro, magnetic, thermal, compositional, Ek, m, symm, ricb, bci, bco, forcing, forcing_frequency, forcing_amplitude_cmb, forcing_amplitude_icb, projection, B0type, beta_actual, B0_l, innercore_mag_bc, c_icb, c1_icb, mantle_mag_bc, c_cmb, c1_cmb, mu, Em, Le2, B0_norm, Etherm, heating, BV2, rc, h, rsy, bci_thermal, bco_thermal, Ecomp, compositional_background, BV2_comp, rcc, hc, rsyc, bci_compositional, bco_compositional, OmgTau, ncpus, N, lmax, runtime, mu_i2o, sigma_i2o, aux1, aux2, rotdyn, MoIZ_M, MoIZ_IC, OmgtauIC, gTorque, bgflow, bgflow_Ro, B0_tor, B0_tor_l, cnorm_tor. Text options are stored as their position in the list of options: `B0type` in `['axial', 'dipole', 'G21 dipole', 'Luo_S1', 'Luo_S2', 'FDM']` (-1 for `B0 = None`), `B0_tor` in `['Malkus', 'wall', 'user']` (-1 without), `innercore_mag_bc` in `['insulator', 'TWA', 'conducting, Chebys', 'conducting, Bessel']`, `mantle_mag_bc` in `['insulator', 'TWA']`, `heating` in `['internal', 'differential', 'two zone', 'user defined']`. `runtime` is the total time of `solve_nopp.py` plus `spin_doctor.py`.

**`eigenvalues.dat`**: damping $\sigma$ and frequency $\omega$.

**`flow.dat`** (20 columns; 19 in files written before 2026-10-23, 18 before 2026-10-11):

| Columns | Quantity                                                              |
|---------|-----------------------------------------------------------------------|
| 1-3     | Kinetic energy: total, poloidal, toroidal                             |
| 4       | Viscous dissipation from $\int \mathbf{u}\cdot\nabla^2\mathbf{u}\,dV$ |
| 5       | Viscous (internal) dissipation from the rate of strain                |
| 6-8     | Work of the Lorentz force, the thermal and the compositional buoyancy |
| 9-10    | `resid0`, `resid𝐮`                                                   |
| 11-12   | Viscous torque on the mantle (real, imaginary)                        |
| 13-14   | Viscous torque on the inner core (real, imaginary)                    |
| 15      | Unused (0)                                                            |
| 16-17   | Angular momentum of the fluid $L$ (real, imaginary)                   |
| 18      | `residL`                                                              |
| 19      | Geostrophic fraction of $u_\phi$ (`nan` unless $m = 0$, `symm = 1`)   |
| 20      | Work of the background flow (`bgflow = 1`, see [Background flow](bgflow.md); 0 otherwise) |

**`magnetic.dat`** (9 columns): magnetic energy, magnetic diffusion, induction work, `resid𝐛`, electromagnetic torque on the mantle (real, imaginary), electromagnetic torque on the inner core (real, imaginary), rms of $b_r$ at the CMB.

**`budget.dat`** (8 columns, in kinetic-energy units): magnetic energy inside the inner core (its field if conducting, $\frac{1}{2}\int|\mathbf{b}|^2/\mu_{i2o}\,dV$ with $\mu_{i2o}$ = `mu_i2o`; the potential field below an insulating inner core or thin wall), magnetic energy stored in thin walls, energy of the potential field outside the CMB, Ohmic loss in the fluid, in a conducting inner core ($\eta_i\int|\nabla\times\mathbf{b}|^2/\mu_{i2o}\,dV$, $\eta_i = 1/(\mu_{i2o}\sigma_{i2o})$) and in thin walls, Lorentz work plus induction work (the work of the background currents), `resid𝐄`.

**`thermal.dat`** (4 columns): thermal "energy", thermal advection, thermal diffusion, `residθ`.

**`compositional.dat`** (3 columns): compositional "energy", advection, diffusion.

**`rotdyn.dat`** (8 complex columns): see [Rotational dynamics](rotdyn.md#output).

Energies, dissipation rates and work are in the units set by `timescale`, with the magnetic energy scaled so that kinetic and magnetic energies are comparable (an eigenvector has kinetic plus magnetic energy 1). Torques are the torques the fluid exerts on the mantle and the inner core, as complex amplitudes: axial for $m = 0$, `symm = 1`, equatorial for $m = 1$, `symm = -1`.


## Plotting and analysis

### Scripts in `tools/`

Run them from the run folder (they read `bin/parameters.py` and the `.field` files there):

| Script | Usage | What it does |
|---|---|---|
| `plot_field.py` | `python3 tools/plot_field.py nsol nR ntheta theta0 theta1 field opt [tex]` | Meridional sections of the three components of the flow (`field = u`) or magnetic field (`b`) of solution `nsol`, on `nR` × `ntheta` points between colatitudes `theta0` and `theta1`. `opt` is `raw` (real part) or `abs` (modulus). The optional last argument is `tex` (the default, LaTeX labels) or `notex` (matplotlib's mathtext, no LaTeX needed). Example: `python3 tools/plot_field.py 0 40 80 0 180 u raw notex`. |
| `plot_poltor.py` | `python3 tools/plot_poltor.py nsol nR ntheta theta0 theta1` | Meridional sections of the poloidal and toroidal flow. Needs LaTeX and the `cmasher` package. |
| `ell_spectrum.py` | `python3 tools/ell_spectrum.py nsol radius field` | Spectrum in spherical-harmonic degree $l$ at a given radius. |
| `shear_layer_profile.py` | `python3 tools/shear_layer_profile.py nsol z0 nl` | Flow profile across a shear layer, through the point `z0` on the axis, over $\pm 5E^{1/3}$. |
| `plot_rad_con_tor_profile.py` | `%run -i tools/plot_rad_con_tor_profile.py nsol nR R1 R2` (IPython) | Radial profiles of kinetic energy and dissipation between radii `R1` and `R2`. |
| `load_solution.py` | `%run -i tools/load_solution.py nsol field` (IPython) | Loads solution `nsol` (`u` or `b`) into the session, including the field inside the inner core. |


### koreviz

`koreviz` is a python package for plots and 3D output of a solution. It needs SHTns (see [Installation](installation.md#optional-shtns-and-pyevtk)). In python, from the run folder (which contains `koreviz/` and `bin/`):

```python
import koreviz as kv
import matplotlib.pyplot as plt

u = kv.kmode(field='u', solnum=0)      # flow of solution 0 on a grid
u.merid(field='uphi')                  # meridional section of u_phi
u.surf(field='ur', r=0.9)              # u_r on the sphere r = 0.9
u.equat(field='ur')                    # equatorial section

b = kv.kmode(field='b', solnum=0, ic=True)   # magnetic field, also inside the inner core
b.merid(field='bphi', ic=True)               # the section continues into the inner core

plt.show()
```

The field names are `ur`, `utheta`, `uphi`, `br`, `btheta`, `bphi`, `temperature`, `composition`, `energy` (kinetic energy density) and `vortz` (axial vorticity, with `vort=True`). With `ic=True` the field inside the inner core is the computed one for `'conducting, Chebys'`, and the matching potential field for `'insulator'` and `'TWA'`. `potextra(rout)` returns the potential field outside the CMB (or inside an insulating inner core) at the radii `rout`. `kv.writeVts(mode, scals=[...], vecs=[...])` writes `out.vts` for ParaView (needs pyevtk), optionally with the potential field outside the core (`potextra=True`).


## Many runs

The scripts in `tools/` help with parameter studies on a PBS-managed cluster. Edit the paths in them first (they assume the code in `~/kore` and runs in `~/data`).

| Script                                     | What it does                                                                                                                                                                                                                                            |
|--------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `dodirs.sh prefix var mode start step end` | Creates one run folder per value of `par.var`, named `prefix<value>`, each with its own copy of the code. With `mode` = `e` the values are powers of ten ($10^{start}$, ...), with `d` they are linear. Example: `./dodirs.sh run_Ek_ Ek e -5 -0.1 -6`. |
| `dosubs.sh` (same arguments)               | Submits the runs, using `subramp.sh` as the job script.                                                                                                                                                                                                 |
| `reap.sh prefix`                           | Collects the results of all the folders into `prefix.par`, `.flo`, `.eig`, `.mag` and `.thm`.                                                                                                                                                           |
| `getresults.py`                            | `%run -i tools/getresults.py prefix` loads the collected results (or the `.dat` files of the current folder) in IPython.                                                                                                                                |
| `get_data.py`                              | Loads the collected results (or the `.dat` files of the current folder) in a pandas DataFrame.                                                                                                                                                          |
| `join.sh label1 label2 out`                | Concatenates two collected data sets.                                                                                                                                                                                                                   |
