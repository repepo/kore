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
| `tests/centrifugal/run_all.sh` | the terms of the centrifugal deformation against a symbolic derivation and quadratures, its eigenvalues against exact results, the budgets | after a change of the deformation, of the inertia, the Coriolis force or the heat equation, or of `spin_doctor`'s budgets | 5 min, 2 min with `SKIP_SOLVE=1` (no PETSc) |
| `tests/rsf/run_all.sh` | the random Reynolds-stress forcing (forcing 11) | after a change of the forcing | 3 min |
| `tests/tidal/run_all.sh` | the tidal forcing (forcing 14) against an exact solution, and the energy budget with the work at the walls and the viscous dissipation | after a change of a boundary forcing, of `spin_doctor`'s budgets or of a force | 2 min on 4 ranks |
| `tests/rsf/regress_solve.sh` | eigenvalues, fields and forced solutions bitwise equal to an older copy | after a change of `solve.py` | 15 s |

`tests/diff_rot`, `tests/heat_dr`, `tests/magnetic`, `tests/centrifugal`, `tests/rsf` and `tests/tidal` each have a README
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
