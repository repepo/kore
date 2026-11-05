# Checks of the centrifugal deformation

Companion of `docs/background.md` (Centrifugal deformation) and `docs/centrifugal-equations.md`, for `centrifugal = 1`
(`bin/centrifugal.py`, NEW 2026-11-05). `run_all.sh [WORKDIR]` builds the runs it needs under `tests/centrifugal/runs/` and
prints one verdict per check (about 5 minutes; the budgets need petsc4py and slepc4py, `SKIP_SOLVE=1 run_all.sh` skips them;
with `OLD_KORE` set to a copy of the commit before the change it also runs `tests/diff_rot/regress.sh`).

| script | what it checks |
| --- | --- |
| `derive.py` | the tool: from the definitions alone (the map X = α + ξ, ξ = ε(r)∇(r²P2/2), and w = J F⁻¹u), with sympy in Cartesian components, the closed forms that `centrifugal.py` implements: the level surfaces r(1 + εP2); the identities that hold for any map (mass conservation, the covariant Coriolis force (F⁻¹ẑ)×w, the advection w_r S0′/J, the buoyancy Φ′(r) r̂); the first-order (M − I)w, (z̃ − ẑ)×w and J − 1; the map of `to_physical`; the effective potential on the level surfaces, whose monopole gives g − (2/3)Ω²r (a few seconds) |
| `algebra_check.py [--quick]` | the angular couplings of `centrifugal.primitives` (the recurrence algebra of `bin/angular.py`) against a projection of the closed forms by Gauss-Legendre quadrature, with harmonics from `scipy.special.lpmv`: every term, m = 0–6, l up to 36, and the P2 coupling of the heat rows (5e-14) |
| `clairaut_check.py` | the flattening of `'clairaut'`: a homogeneous body (f = 5q/4, η = 0), the polytrope of index 1 (f/q = 7.5/π²), `'Wu 2005'` against Clairaut's second-order equation integrated from ρ itself (7e-10), the profile codes against finite differences and their Chebyshev fits, and the Sun (`'model S hydrostatic'`: f/q = 0.519) |
| `eig_check.py` | eigenvalues of small runs (scipy shift-invert on A.npz, B.npz): f = 0 gives the matrices of `centrifugal = 0` entry by entry; the spin-over of a Maclaurin spheroid (inviscid, uniform f) has (Im λ − 1)/f → 1 (exact 1 + f + f²/2, first order 1 + f + f²/3); the relabelling test (f = c(1 − r²)², a sphere labelled differently inside: 13 resolved modes move at O(c²), their odd part is O(c³)); `'clairaut'` of a homogeneous body equals `'uniform'` f = 5q/4; the monopole equals Beyonce(1 − 2q/3) for g = r; the stratified, deformed ideal fluid has imaginary eigenvalues (1e-16) |
| `bin/ru_insane.py` | with `centrifugal = 1` also: the metric blocks (inertia, Coriolis, entropy) packed as `assemble.py` packs them, against direct (r, cos θ) quadratures of the closed forms on random fields, the inertia and entropy blocks Hermitian with I + Ic and E + Ec positive, the Coriolis block anti-Hermitian, and KE of the deformed body against `utils4pp.centrifugal_terms` (round-off: 1e-14); `ru_wu_dr` adds a differential rotation with its thermal wind, which keeps the structure's g while the buoyancy takes the effective one |
| `budget_check.py [tol]` | after `spin_doctor.py`: residσ, residω and resid𝑠 of every solution below tol (1e-8), also with a differential rotation (`bud_wu_dr`). spin_doctor weights KE with M, the Coriolis power with z̃ and TE with J by quadrature, apart from the matrices; without these weights the residuals are 1e-3 to 3e-2 |
| `mk_run.py NAME 'par.x = ..' [solve]` | builds a run directory from the repo's `bin/` with the test defaults (the inviscid Boussinesq full sphere, m 1, symm −1, N 24, lmax 20, `'uniform'` f = 0, one rank) and runs submatrices and assemble (with `solve` also solve.py and spin_doctor.py) |

Results of 2026-11-05 are in `docs/centrifugal-equations.md` (Checks). What the checks do not reach: the metric terms that are
left out (the viscous force, the thermal diffusion, the stress-free rows, the differential rotation and the magnetic terms of
the deformed body), whose size is estimated there.
