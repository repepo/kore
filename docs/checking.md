# Modifying kore

After changing kore, two questions matter: do the runs that worked before still give the same results, and is the new
part right? kore has a tool for each. Everything below runs from the repository, on one machine; most checks take
seconds to minutes.

## Which check, when

| check | what it tells you | when to run it | time (laptop) |
| --- | --- | --- | --- |
| residuals of `spin_doctor.py` | each solution satisfies its energy and entropy budgets | every run ([Solver](solver.md#first-check-the-solution)) | part of the run |
| `bin/ru_insane.py` | the operators have the symmetries of the equations (inertia Hermitian and positive, Coriolis anti-Hermitian, viscous force Hermitian and negative, ...) | after `submatrices.py`, when an operator changed | seconds |
| `tests/ci` (`pytest -v`) | the spin-over mode of the smooth Sun is exact | before every push; GitHub runs it on every push | 6 s |
| `tests/diff_rot/regress.sh` | the matrices A and B of 26 small runs are unchanged ([below](#are-existing-runs-unchanged)) | after any change that should not alter results | 3 min |
| `tests/heat_dr/run_all.sh` | the rows of the heat equation and of the differential rotation, their assembly and their budgets | after a change of the background, the rotation or the heat equation | 6 min |
| `tests/magnetic/run_all.sh` | the magnetic rows against a symbolic derivation, the budgets, published results | after a change of the magnetic field | 30 min with `SKIP_DERIVE=1`, 60 min in full |
| `tests/centrifugal/run_all.sh` | the terms of the centrifugal deformation, with and without a field, against a symbolic derivation and quadratures, the rows of the insulating walls against the exact matching, its eigenvalues against exact results (among them the modes of a spheroid, inviscid and damped), the budgets | after a change of the deformation, of the inertia, the Coriolis force, the viscous force, the heat equation, the Lorentz force, the induction, the diffusion, the differential rotation or the wall rows, or of `spin_doctor`'s budgets | 17 min on 16 cores, 13 min with `SKIP_SOLVE=1` (no PETSc) |
| `tests/rsf/run_all.sh` | the random Reynolds-stress forcing (forcing 11) | after a change of the forcing | 3 min |
| `tests/tidal/run_all.sh` | the tidal forcing (forcing 14) against an exact solution, and the energy budget with the work at the walls and the viscous dissipation | after a change of a boundary forcing, of `spin_doctor`'s budgets or of a force | 2 min on 4 ranks |
| `tests/decay/run_all.sh` | the free decay of the field and of the entropy without the flow (`hydro = 0`) against exact rates, and the kinematic induction (`just_induction.py`, `solve_ind.py`) | after a change of the magnetic diffusion, the magnetic walls, a conducting inner core, the heat diffusion or the layout of the unknowns | 30 s |
| `tests/forcing/run_all.sh` | the wall values that the boundary forcings 2, 6, 7, 9, 10 and 12 impose, read back from the solution | after a change of a boundary forcing or of the wall rows | 40 s |
| `tests/couette/run_all.sh` | the steady rotation of the walls (forcing 12): the solutions of the `conductive_IC` branch, the exact flow without rotation, the rigid rotations with conducting solids, the budgets of the flows that the walls drive (also the libration and the thermal forcing), the stops | after a change of a boundary forcing, of the magnetic walls or the inner core, or of `spin_doctor`'s budgets | 1.5 min on 2 ranks |
| `tests/cic_forcings/run_all.sh` | the tides and the libration of the `conductive_IC` branch (forcings 1, 2 and 8): their solutions there, the libration in the frame of the mantle against that of the walls (forcing 7), the wave-like tide against the rigid rotations that solve it in a full sphere, the budgets with the power of the body force, the stops | after a change of a boundary or body forcing, of the right-hand side of a forced run, or of `spin_doctor`'s budgets | 2.5 min on 2 ranks |
| `tests/twa/run_all.sh` | the thin conducting layer (`'TWA'`): exact free-decay rates, eigenvalues of the `conductive_IC` branch, the slipping fluid, solids that turn with the fluid, the budgets | after a change of the magnetic walls, of the induction by the flow or the rotation, or of `spin_doctor`'s magnetic budgets | 1 min |
| `tests/mantle/run_all.sh` | the resolved conducting mantle (`'resolved'`): exact free-decay rates, eigenvalues of the `conductive_IC` branch, the limits of a thin layer and of a vanishing conductivity, a layer that turns with the fluid, the budgets with a variable density, the kinematic induction | after a change of the magnetic walls, of the magnetic diffusion, of the induction by the flow or the rotation, of the layout of the unknowns, or of `spin_doctor`'s magnetic budgets | 1.5 min |
| `tests/forced_field/run_all.sh` | the forced runs with a field: the solutions of the `conductive_IC` branch, the Bessel inner core against the Chebyshev one, the forced matrix against the eigenproblem's, the budgets where the forcing moves the walls with the solids at rest and deforming, a thin resolved mantle against the thin layer, the kinematic induction, the stops | after a change of the forced assembly, a boundary forcing, the magnetic walls or the inner core, or `spin_doctor`'s budgets | 3 min |
| `tests/inner_core/run_all.sh` | the field of a dipole at the centre and a fluid slipping along a conducting inner core: eigenvalues and forced solutions of the `conductive_IC` branch, the Bessel inner core against the Chebyshev one, the budgets with a variable density, a rigid rotation offset, the stops (among them a toroidal field's current sheet in an inviscid run) | after a change of the magnetic walls or the inner core, of the induction by the flow, or of `spin_doctor`'s magnetic budgets | 1.5 min |
| `tests/sheet/run_all.sh` | the push of the background field's current sheet on a fluid slipping along a wall: smoothed sheets converge to it (toroidal and poloidal, either wall, zero radial stress, permeable thin layers, a variable density, a drag, other normalisations, $m$ = 2 with `symm = -1`), no axial angular momentum with $m$ = 0, the budgets, the stops, the walls' conditions against the solver's rows | after a change of the stress-free rows, of the magnetic walls or of the background field | 11 min on 2 ranks |
| `tests/rotdyn/run_all.sh` | the rotating solids (`rotdyn = 1`) and resid𝐋: exact decoupled and heavy solids, eigenvalues and torques of the `conductive_IC` branch, the conservation of angular momentum and energy with a variable density, the equatorial angular momentum, the stops | after a change of the walls, of the viscous or magnetic stresses, of the layout of the unknowns, or of `spin_doctor`'s budgets | 1 min |
| `tests/drag/run_all.sh` | the drag at the outer wall (`kappa_cmb`, `topo_drag`): eigenvalues and torques of the `conductive_IC` branch, the conservation of angular momentum and energy with the rotating solids, the limits of a weak and of a stiff drag, a rigid differential rotation, the frequencies of the drag of the topography, the power through the wall, the stops | after a change of the walls, of the viscous stress, of the rotating solids, or of `spin_doctor`'s budgets | 2 min |
| `tests/composition/run_all.sh` | the composition (`compositional = 1`) and the `'two zone'` gradient: eigenvalues of the `conductive_IC` branch, the composition alone against the entropy alone in every setup (a full sphere, a differential rotation, the deformed body, a field with conducting solids, the rotating solids, the drag, a forced run, diffusivities that vary, the free decay), both scalars against one, the budgets with a variable density, `ru_insane.py`, the stops | after a change of the heat equation or of the composition's, of the buoyancy, of the layout of the unknowns, or of `spin_doctor`'s budgets | 3.5 min on 4 cores |
| `tests/bgflow/run_all.sh` | the background flow (`bgflow = 1`): the files that the writer of the `conductive_IC` branch writes, linear and nonlinear, and its eigenvalues on them; a toroidal background against the differential rotation; a rigid rotation against the turning frame (a full sphere, an inviscid one, without diffusion, a field with conducting solids and thin walls); the rows against a grid; the budgets and the background's powers against an independent computation, eigenvalue and forced problems; the stops | after a change of the background flow, of an advection term or the induction, of the layout of the unknowns, or of `spin_doctor`'s budgets | 8.5 min on 2 ranks |
| `tests/rsf/regress_solve.sh` | eigenvalues, fields and forced solutions bitwise equal to an older copy | after a change of `solve.py` | 15 s |
| `tests/tools/run_all.sh` | the solution files in their two formats, the parameter file, the tracking of a mode, the tools that run and collect series of runs (`ramp.sh`, `setpar.py`, `topo_iterate.sh`, `reap.sh`, `get_data.py`), with their stops, and the plotting tools that read the solutions | after a change of the files a run writes or reads, of `spin_doctor`'s tables, or of the tools | 2 min on one rank |

`tests/diff_rot`, `tests/heat_dr`, `tests/magnetic`, `tests/centrifugal`, `tests/rsf`, `tests/tidal`, `tests/forcing`, `tests/couette`, `tests/cic_forcings`, `tests/decay`, `tests/twa`, `tests/mantle`, `tests/forced_field`, `tests/inner_core`, `tests/sheet`, `tests/rotdyn`, `tests/drag`, `tests/composition`, `tests/bgflow` and `tests/tools` each have a README
that lists their scripts and what each one checks; the CI test explains itself in `tests/ci/test_spinover.py`.

## Are existing runs unchanged?

`tests/diff_rot/regress.sh` builds the same small runs twice, once with an older copy of kore and once with yours, and
compares their matrices A and B entry by entry. Equal matrices give equal eigenvalues and eigenvectors, so this is the
check to run after a change that should leave every existing run alone: a cleanup, a speed-up, a new option that is
off by default. It needs numpy, scipy and mpi4py, not PETSc.

**1. Make an older copy.** Usually the last commit, which does not contain your uncommitted changes:

```bash
mkdir -p /tmp/kore_old && git archive HEAD | tar -x -C /tmp/kore_old
```

For another version, put its commit in place of `HEAD`.

**2. Run the comparison** from the repository (the second argument is where the runs are built):

```bash
tests/diff_rot/regress.sh /tmp/kore_old /tmp/regress_runs
```

Each run is built as `o_NAME` with the older copy and `n_NAME` with yours, each with its logs.

**3. Read the verdicts.** Each run prints one line:

| verdict | meaning |
| --- | --- |
| `ok` | A and B are bitwise equal |
| `round-off` | they differ only in the last bits: every entry by less than 1e-13 of the largest entry of its row |
| `skipped` | the older copy predates a feature the run uses; not a failure |
| `FAIL` | anything else, including a run that does not build (its logs say why) |

The last line is `regression ok` with the number of runs compared, or `regression FAIL`. A change that is meant to alter
results, such as a bug fix, fails in the runs it concerns; check that it fails only there.

**What it covers.** The runs are small (N 48, lmax 20 or 21, one rank, a few seconds each) and start from structure
'Wu 2005' with stratification 'top hat', the heat equation, an inner core at 0.35, m 1 and a 'conical' rotation. They
change one aspect each:

- 15 runs of the differential rotation and the heat equation: every rotation profile, `thermal_wind` 0 to 3, a full
  sphere, `ThermaD` = 0, a solar structure, `Gaspard` 3, m 2, and no rotation at all;
- 11 runs of the magnetic field: the poloidal fields 'axial', 'FDM', 'Luo_S2', 'confined', 'general' and 'ferraro', a
  toroidal field, a conducting inner core, a full sphere, the induction by the rotation, the wound field,
  `diff_rot_type = 'ferraro'`, and the couplings of `mag_couplings = 'algebra'`.

Older copies spell some parameters differently; the script gives each one the names it reads, and skips the magnetic
runs whose code an older copy does not have.

**What it does not cover.** It compares matrices, so a change in `solve.py` or `spin_doctor.py` passes unseen. For
`solve.py`, `tests/rsf/regress_solve.sh` compares the solutions of an eigenvalue run and a forced run with an older copy
in the same way.

**Adding a run.** Append a name and its parameter changes to the lists at the top of the script: `names` and `cfgs` for
a run of the rotation, `mnames`, `mcfgs` and `mneeds` (the code the run needs, so that older copies without it skip it)
for a magnetic run.

## Is the new part right?

There is no single tool for this, but the same few checks catch most mistakes:

- **Budgets.** Solve a few modes and look at the residuals of `spin_doctor.py`: a new term that is wrong usually breaks
  the energy or entropy budget, since `spin_doctor` computes each term from the fields, independently of the matrices.
- **Symmetries.** `bin/ru_insane.py` checks every block of the operators against its expected symmetry.
- **Limits and exact cases.** A new term should vanish, or reduce to a known one, in some limit. Two existing checks
  show the idea: a uniform rotation offset must give the eigenvalues of a faster uniform rotation, shifted in frequency
  (the CI test, `tests/magnetic/rigid_check.py`), and 'Wu 2005' must tend to 'Boussinesq' as its density contrast
  vanishes (`tests/magnetic/continuity.py`).
- **Published results.** [Benchmarks](benchmarks.md) lists the ones kore reproduces, with their set-ups.
