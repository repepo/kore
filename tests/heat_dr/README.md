# Checks of the differential-rotation terms of the heat equation

Companion of `docs/heat-equation.md`. Everything here runs on one core with numpy, scipy and mpi4py, from a kore
run directory (a `bin/` copy, `parameters.py`, the `*.mtx` files of `submatrices.py`, and `A.npz`, `B.npz` of
`assemble.py`). `run_all.sh` builds the runs it needs under `tests/heat_dr/runs/` and prints one line per check.

| script | what it checks |
| --- | --- |
| `mk_run.py NAME 'par.x = ..'` | builds a run directory from the repo's `bin/` with the test defaults (user-def profiles, ricb 0.35, m 1, conical rotation with amplitude -0.25, thermal_wind 1, Beyonce 10, ViscosD = ThermaD = 1e-3) and runs submatrices + assemble |
| `heat_check.py [ntrials]` | operator level: for random P, T, s, the pairing of the new blocks of `operators.thermal_diffrot` with s* against an (r, theta) quadrature of p0 s* [i m dOmega s + u_r dS_tw/dr + (u_theta/r) dS_tw/dtheta] built independently (LBR velocity, Schmidt harmonics from scipy, thermal wind recomputed from f0, f2, g); passes below 1e-8, typically 1e-12 (1e-6 at N = 48 is profile truncation, not a bug) |
| `asm_check.py ../NODR/A.npz [ntrials]` | assembled level: the heat rows of A(diff_rot = 1) - A(diff_rot = 0), both runs otherwise identical, against the same quadrature (undoes the Frobenius normalisation of A) |
| `rigid_thermal.py BASE DR dO m` | the exact identity for a rigid offset Omega = Omega0 (1 + dO), with the heat equation on: lambda(dO) = (1 + dO) lambda(0; E/(1+dO), kappa/(1+dO), Beyonce/(1+dO)^2) - i m dO (`diff_rot_type = 'rigid'`) |
| `csolve.py sr si k` | scipy shift-invert solve of A x = lambda B x near sr + i si, writes `eig.npz` |
| `ebal.py`, `terms.py` | matrix-level energy and entropy budgets of the modes (`terms.py` prints lambda E - A - D - H = 0 per mode, H the new terms) |

Results of 2026-10-12 (cloud container, HEAD 652e1490 + this change): heat_check 5e-12 (N 96), 5e-14 (N 160), 9e-12 (m 0),
4e-12 (m 2, symm -1), 2e-15 (thermal_wind 0), 4e-11 (smoothed sun, ricb 0, N 120, lmax 30); asm_check the same numbers;
rigid identity worst 8.6e-12 over 16 modes; smoothed-sun entropy balance closes to 1e-9 with ImH ~ 0.2 ImA.
