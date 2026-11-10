# Modifying kore

After changing kore, two questions matter: do the runs that worked before still give the same results, and is the new
part right? kore has a tool for each. Everything below runs from the repository; most checks take seconds to minutes.

## Which check, when

| check | what it tells you | when to run it |
| --- | --- | --- |
| residuals of `spin_doctor.py` | each solution satisfies its energy and entropy budgets | every run ([Solver](solver.md#first-check-the-solution)) |
| `bin/ru_insane.py` | the operators have the symmetries of the equations | after `submatrices.py`, when an operator changed (seconds) |
| `tests/ci` (`pytest -v`) | quick end-to-end tests: the spin-over of the smooth Sun, the onset of convection and torsional modes against published values, the symmetries of the operators; GitHub runs them on every push, with the `decay`, `tidal`, `forcing`, `rotdyn`, `inner_core` and `composition` suites below | before every push (about a minute) |
| `tests/diff_rot/regress.sh` | the matrices of a set of small runs are unchanged ([below](#are-existing-runs-unchanged)) | after any change that should not alter results (minutes) |
| `tests/rsf/regress_solve.sh` | eigenvalues, fields and forced solutions are unchanged | after a change of `solve.py` |
| `tests/heat_dr/run_all.sh` | the heat equation and the differential rotation | after a change of the background, the rotation or the heat equation |
| `tests/magnetic/run_all.sh` | the magnetic field: the terms against a symbolic derivation, the budgets, published results | after a change of the magnetic field (the longest suite) |
| `tests/centrifugal/run_all.sh` | the centrifugal deformation, against exact results | after a change of the deformation or of any term it modifies (long; `SKIP_SOLVE=1` runs it without PETSc) |
| `tests/rsf/run_all.sh` | the convective forcing (forcing 11) | after a change of that forcing |
| `tests/tidal/run_all.sh` | the tidal forcing (forcing 14) against an exact solution, and its energy budget | after a change of a boundary forcing, of a force or of `spin_doctor`'s budgets |
| `tests/forcing/run_all.sh` | the wall values imposed by the boundary forcings | after a change of a boundary forcing or of the wall conditions |
| `tests/couette/run_all.sh` | the rotating walls (forcing 12), libration and thermal forcing | after a change of a boundary forcing or of the magnetic walls |
| `tests/cic_forcings/run_all.sh` | the wave-like tide, the Enceladus tide and libration in the mantle frame (forcings 1, 2, 8) | after a change of a forcing or of `spin_doctor`'s budgets |
| `tests/decay/run_all.sh` | free decay without the flow, against exact rates; kinematic induction | after a change of the magnetic or thermal diffusion or the magnetic walls |
| `tests/twa/run_all.sh` | the thin conducting layer | after a change of the magnetic walls |
| `tests/mantle/run_all.sh` | the resolved conducting mantle | after a change of the magnetic walls or diffusion |
| `tests/forced_field/run_all.sh` | forced runs with a magnetic field | after a change of the forced assembly or the magnetic walls |
| `tests/inner_core/run_all.sh` | the dipole field and a fluid slipping along a conducting inner core | after a change of the magnetic walls or the inner core |
| `tests/sheet/run_all.sh` | the current sheet of the background field at a slipping wall | after a change of the stress-free conditions, the magnetic walls or the background field |
| `tests/rotdyn/run_all.sh` | the rotating solids and the angular momentum check | after a change of the walls or of the stresses on them |
| `tests/drag/run_all.sh` | the drag at the outer wall, also on a deformed body | after a change of the walls, the viscous stress, the rotating solids or the centrifugal deformation |
| `tests/composition/run_all.sh` | the composition | after a change of the heat or composition equations or the buoyancy |
| `tests/bgflow/run_all.sh` | the background flow | after a change of the background flow or of an advection term |
| `tests/tools/run_all.sh` | the output files, the parameter file, mode tracking and the tools for series of runs | after a change of the files a run writes or reads, or of the tools |

Each folder in `tests/` has a README that lists its scripts and what each one checks.

**All the suites at once.** `tests/run_suites.sh` runs every suite side by side, the longest first, each in its own
folder with its own log, and ends with the list of suites that failed. With `OLD_KORE` set to an older copy it also
runs the two regression scripts:

```bash
OLD_KORE=/tmp/kore_old tests/run_suites.sh /tmp/suites                    # all of them, logs in /tmp/suites/logs
tests/run_suites.sh /tmp/suites decay forcing ci                         # only these
```

You can split the suites between machines by giving each machine its own list. `JOBS` sets how many suites run at a
time, and `MAG_NPAR`, `CF_NPAR` and `OTHER_NPAR` how many runs some suites start in parallel.

## Is it the mode you want?

Besides the residuals, `spin_doctor.py` prints what a mode looks like: 𝒯/𝒫, the ratio of toroidal to poloidal kinetic
energy, Peak ℓ and ℓ-Width, Mag/Kin with a field, and, for `m = 0`, `symm = 1`, Geo (NEW 2026-11-10, from conductive_IC):
the fraction of the energy of $u_\phi$ held by its average along the rotation axis, $\int\lvert\langle u_\phi\rangle\rvert^2dV
/\int\lvert u_\phi\rvert^2dV$, with $\langle u_\phi\rangle(s)$ the mean over each fluid column (`flow.dat` column 50). A torsional
mode has Geo close to 1. The column average does not resolve the Ekman layers, so Geo measures the interior.

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

**3. Read the verdicts.** Each run prints one line:

| verdict | meaning |
| --- | --- |
| `ok` | A and B are bitwise equal |
| `round-off` | they differ only in the last bits |
| `skipped` | the older copy predates a feature the run uses; not a failure |
| `FAIL` | anything else, including a run that does not build (its logs say why) |

The last line is `regression ok` or `regression FAIL`. A change that is meant to alter results, such as a bug fix,
fails in the runs it concerns; check that it fails only there.

**What it covers.** The runs are small and cover the differential rotation profiles, the thermal wind options, a full
sphere, the solar structure, and the main magnetic field options. It compares matrices, so changes in `solve.py` or
`spin_doctor.py` are not seen: use `tests/rsf/regress_solve.sh` for `solve.py`.

**Adding a run.** Append a name and its parameter changes to the lists at the top of the script: `names` and `cfgs` for
a run of the rotation, `mnames`, `mcfgs` and `mneeds` for a magnetic run.

## Is the new part right?

There is no single tool for this, but the same few checks catch most mistakes:

- **Budgets.** Solve a few modes and look at the residuals of `spin_doctor.py`. A wrong new term usually breaks the
  energy or entropy budget, since `spin_doctor` computes each term from the solution, independently of the matrices.
- **Symmetries.** `bin/ru_insane.py` checks every block of the operators against its expected symmetry.
- **Limits and exact cases.** A new term should vanish, or reduce to a known one, in some limit. For example, a uniform
  rotation offset must give the eigenvalues of a faster uniform rotation, shifted in frequency.
- **Published results.** [Benchmarks](benchmarks.md) lists those kore reproduces.
- **Combinations.** Run a new feature with each of the others, and make kore stop with a message where a combination
  is not supported. Then add its row and column to [Combining features](compatibility.md): the cells are in
  `tools/compat_tables.py`, which writes the tables of that page (`--check` says whether the page is up to date).
