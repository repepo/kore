# Checks of the differential-rotation terms of the heat equation

Companion of `docs/heat-equation.md`. Everything here runs on one core with numpy, scipy and mpi4py, from a kore
run directory (a `bin/` copy, `parameters.py`, the `*.mtx` files of `submatrices.py`, and `A.npz`, `B.npz` of
`assemble.py`). `run_all.sh` builds the runs it needs under `tests/heat_dr/runs/` and prints one line per check.

| script | what it checks |
| --- | --- |
| `mk_run.py NAME 'par.x = ..'` | builds a run directory from the repo's `bin/` with the test defaults (user-def profiles, ricb 0.35, m 1, conical rotation with amplitude -0.25, thermal_wind 1, Beyonce 10, ViscosD = ThermaD = 1e-3) and runs submatrices + assemble |
| `heat_check.py [ntrials]` | operator level: for random P, T, s, the pairing of the new blocks of `operators.thermal_diffrot` with s* against an (r, theta) quadrature of p0 s* [i m G dOmega s + u_r dS_tw/dr + (u_theta/r) dS_tw/dtheta] built independently (LBR velocity, Schmidt harmonics from scipy, S_tw integrated in cos(theta) from the thermal-wind balance with F = 2 dOmega or (1 + dOmega)^2 - 1, so thermal_wind = 1 and 2 and any Gaspard G; for thermal_wind = 3 the prescribed dS_tw/dtheta = dSdtheta sin 2 theta); also compares the code's sigma2, sigma4, sigma6 fits with that integral; passes below 1e-8, typically 1e-12 (1e-6 at N = 48 is profile truncation, not a bug) |
| `asm_check.py ../NODR/A.npz [ntrials]` | assembled level: the heat rows of A(diff_rot = 1) - A(diff_rot = 0), both runs otherwise identical, against the same quadrature (undoes the Frobenius normalisation of A); with ThermaD = 0 and thermal_wind >= 1 (heat rows r^2 (eq)/p0 in C^(1)) the reference is the same run with diff_rot_amplitude = 0 (`z_*` in `run_all.sh`), since diff_rot = 0 has other heat rows |
| `row_check.py [tol] [ntrials]` | ThermaD = 0 runs: every section-h row block of A x and B x for random fields, read as a series in the C^(1) or C^(0) basis, against u_r S0', i m G dOmega s and u.grad S_tw evaluated directly in (r, theta), without the p0 weight of the checks above (which hides the rows near the wall); passes only if the claimed basis, first row and r^2 (or r) weight match and every other combination is off by > 1e-3 |
| `rigid_thermal.py BASE DR dO m G` | the exact identity for a rigid offset Omega = G (1 + dO), with the heat equation on, for any Gaspard G: lambda(rigid dO, Gaspard G) = lambda(diff_rot = 0, Gaspard G (1 + dO)) - i m G dO, ViscosD, ThermaD, Beyonce unchanged (`diff_rot_type = 'rigid'`) |
| `tw_sympy.py` | sympy: sigma_n of the thermal wind for thermal_wind = 1 and 2 (any G) against the closed forms coded in `radial_profiles.sigma2/4/6` |
| `tw_limit.py a DIR1 DIR2 ...` | thermal_wind = 2 -> 1 as the amplitude a -> 0: max|H2 - H1|/max|H1| from the `heat_check_H.npy` of the two runs, divided by a |
| `csolve.py sr si k` | scipy shift-invert solve of A x = lambda B x near sr + i si, writes `eig.npz` |
| `sd_check.py` | spin_doctor's balances (`utils4pp.diagnose`) per mode of `eig.npz`, with and without the diff-rot terms Wdr, Wdr_im, Ensdr, Wdr_thm, and Wdr + i Wdr_im against -2R, Wdr_thm against 2 Re H of `ebal.py` |
| `mpi_check.sh [WORKDIR]` | `ncpus > 1`: assembles diff_rot = 1, thermal = 1 runs (m 1, m 0, m 2 symm -1, ricb 0 m 1 and m 0 symm -1, thermal bc 1, forcing 10) and a diff_rot = 0 control with `mpiexec -n 1 .. 4` and compares A, B (or B_forced) with the single-rank ones; needs lmax - m + 1 a multiple of 2 x ranks (lmax = m + 23) |
| `forced_check.py ../EIG [..]` | in a forced run (forcing 6, 7, 9 or 10): its A must be c (A_eig - i wf B_eig), c = Anorm, for the same run with forcing = 0, row by row in sections u, v, h; a diff_rot = 0 (or thermal_wind = 0) EIG shows what a missing block gives (~0.1-0.5) |
| `ebal.py`, `terms.py` | matrix-level energy and entropy budgets of the modes (`terms.py` prints lambda E - A - D - H = 0 per mode, H the new terms) |

`ru_insane.py` (in `bin/`) has the matching unit tests: the momentum and heat diff-rot blocks against its own grid,
the anti-Hermitian heat advection and the rigid-offset identity; it runs on any diff-rot run (N >= 96 for its 1e-6
tolerance; at N = 48 the pre-existing viscous and inertia Hermitian tests fail on truncation).

Every check prints a verdict line ending in `ok` or `FAIL`; `run_all.sh` stops at the first crash (`set -e -o pipefail`).
Results of the last full run (cloud container, 2026-10-12): see section 5 of `docs/heat-equation.md`.

At ricb = 0 the diff-rot profiles are taken at |r| (the physical, even profiles), in the code and in `heat_check.py`.
'Y20-wall-bounded' and 'shellular' (unless sigma is even and > 0) have a kink at r = 0 there, and `submatrices.py`
prints a WARNING; use ricb > 0 for them. 'solar' at ricb = 0 needs N ~ 400 for the 1e-8 tolerance of these checks.
