# Checks of the run files and the run tools

These scripts check the files that a run writes and reads besides its matrices ([Output files](../../docs/output.md)):
the solution files in their two formats, the parameter file `parameters.json`, the file `track_target` of the eigenmode
tracking, the tools that run and collect series of runs (`tools/ramp.sh`, `tools/setpar.py`,
`tools/topo_iterate.sh`, `tools/reap.sh`, `tools/get_data.py`), and the plotting tools that read the solutions
(`tools/load_solution.py`, `tools/plot_rad_con_tor_profile.py`, `tools/plot_field.py`).

## Running them

```shell
tests/tools/run_all.sh [WORKDIR]
```

builds the test runs (in `tests/tools/runs/` by default), runs every check and exits with 1 if one failed. It needs
numpy, scipy, mpi4py, petsc4py and slepc4py with MUMPS, pandas for `tools/get_data.py` and matplotlib for the
plotting tools; `MPIEXEC` replaces `mpiexec`. The runs are built from the defaults of `tests/tidal/mk_run.py` (an incompressible fluid in a rotating shell,
ricb 0.35, Ek $10^{-3}$, no-slip walls) at N 32, and solve on one rank, where MUMPS repeats a solution to the last bit.
About 2 minutes.

## The checks

**`tools_check.py`.** Nine groups:

- **Solution files.** Three runs, each solved with `field_format` `'npy'` and `'text'`: an eigenvalue problem with every
  field (entropy, composition, a field over a conducting inner core and a resolved mantle), the rotating solids with a
  thin conducting wall, and a forced run (forcing 7) with a field followed by the kinematic induction
  (`just_induction.py`, `solve_ind.py`). Expected: the solutions of the two formats are the same numbers to the last
  bit, and spin_doctor writes the same files (`params.dat` but its run time). Solving again in the other format leaves
  only that format's files, with the same numbers; `field_io.load` reads one solution and a list of them; `bgflow.py`
  writes the same `bgflow.npz` from a forcing-12 run in either format.
- **Parameter file.** Parameters of every kind (numpy scalars and arrays, complex numbers, tuples, dictionaries with
  integer keys, None, nan, names that kore does not define) through `par.export_json` and `par.load_json`. Expected:
  every value back with its type; a run whose `parameters.py` reads the file (a `'user'` rotation, whose profiles are a
  dictionary with integer keys) gives the matrices, solutions and spin_doctor files of the run that wrote it.
- **Tracking.** `track_target` 2 writes the eigenvalue of the solution with the smallest KP/KT (the spin-over mode,
  `m` 1, `symm` −1). A run with `track_target` 1 and that file, with `rtau`, `itau` at 0, 3, finds it again (to
  $10^{-10}$, in practice $2.5\times10^{-16}$) and writes back the eigenvalue nearest it.
- **Ramp and collection.** `tools/ramp.sh ViscosD 1e-3 8e-4 6e-4` with `track_target` 1: one step per value,
  `parameters.py` ends with the last value, set in a line of several statements, and each step's tracked eigenvalue is
  the one nearest the step before's. `tools/ramp.sh` alone runs the case once; a ramp whose first step finds no
  eigenvalue (`maxit` 1, `tol` $10^{-30}$) stops with exit status 1. `tools/reap.sh` collects two of these folders, and
  `tools/get_data.py` reads them, and a `params.dat` whose rows end with the expressions of a `'user'` rotation.
  Expected: the named columns equal the `.dat` files, number for number.
- **Drag frequency.** `tools/topo_iterate.sh` with `topo_drag = 'rekier'` (entropy, a stress-free outer wall, `m` 1,
  `symm` −1, from $\lambda = 0.98i$): it converges, to a fixed point. Expected: the run with `topo_omega` set to the
  result gives the same eigenvalue (to $10^{-9}$, in practice $2\times10^{-14}$).
- **`tools/setpar.py`.** Replaces an assignment and keeps its comment, one over two lines, one of several statements on
  a line, and adds a new last line for a parameter that the file does not set; nothing else changes.
- **Column names.** The names of `tools/get_data.py` are those of spin_doctor's tables (`param_table`, `flow_cols`,
  `thermal_cols`, `composition_cols`, `magnetic_cols`, `rot_cols`), read from the source of `bin/spin_doctor.py`.
- **Stops.** A `field_format` other than `'npy'` and `'text'`; `track_target` 3, with a forcing, 2 without the flow
  (`hydro` 0), 1 without the file; spin_doctor without solution files; `tools/setpar.py` with a value that is not
  python (the file is left as it was); `tools/ramp.sh` outside kore; `tools/topo_iterate.sh` without the file
  `track_target` and without its parameters.
- **Plotting tools.** `plot_check.py` runs them on solution 0 (matplotlib's Agg backend, no LaTeX) of the run with every
  field, of a full sphere with the density of `'Wu 2005'`, the entropy and a field, of the forced run and of a run in the
  text format. Expected: `tools/load_solution.py` unpacks the coefficients that `utils4pp.expand_sol` gives, and its
  inner core continues the fluid's $f$, $\partial_r f$ and $g$ at the inner boundary (to $10^{-10}$, in practice
  $10^{-14}$); the kinetic energy of `tools/plot_rad_con_tor_profile.py` is KE of `flow.dat` (to $10^{-9}$, the digits
  of the file); `tools/plot_field.py` draws the flow, the viscous force, the buoyancy (with the composition), the curls
  and the field, whose components are those of its Chebyshev coefficients (to $10^{-12}$).
