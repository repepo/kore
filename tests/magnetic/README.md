# Checks of the magnetic field

Companion of `docs/magnetic.md`, for the magnetic terms with a variable density (steps 3 to 5 of the port from conductive_IC,
2026-10-28: any density, the toroidal B0, the full sphere and the poloidal B0 of degree 2). `run_all.sh [WORKDIR]` builds the runs it needs under `tests/magnetic/runs/` and prints one verdict per check
(about 60 minutes, two thirds of it derive_check.py, which `SKIP_DERIVE=1 run_all.sh` skips; the solves need petsc4py and slepc4py, see the note on slepc4py 3.26 below).

| script | what it checks |
| --- | --- |
| `derive.py` | the tool: the rows of the Lorentz force (u: ρ0 r̂⋅∇×∇×(ρ0⁻¹𝐆), v: ρ0 r̂⋅∇×(ρ0⁻¹𝐆)) and of the induction (f: r̂⋅∇×(u×B0), g: r̂⋅∇×∇×(u×B0)) derived with sympy from the definitions u = ρ0⁻¹∇×∇×(ρ0 P 𝐫) + ∇×(T 𝐫), b = ∇×∇×(f 𝐫) + ∇×(g 𝐫), B0 = ∇×∇×(h Y₁⁰ 𝐫), projected on kore's Schmidt harmonics by exact θ-quadrature; each row is a sum of monomials of P^(k), h^(k), (ln ρ0)^(k) with numeric coefficients. From the command line, `python3 derive.py ROW COL LCOL LROW M [--rho]` prints one row's terms, e.g. `python3 derive.py g P 3 3 1 --rho` the variable-density terms of the g rows (~10 s) |
| `derive_check.py` | derive.py against conductive_IC's closed forms for uniform density (factor -1, the A sign; 44 blocks), against the (ln ρ0)' terms implemented in `bin/operators.py` (rows f, g, u), and that rows v and T columns have none; the toroidal B0 of degrees 1, 2 and (NEW 2026-10-28, step 5) the poloidal B0 of degree 2, whose tables it reads from `bin/operators.py` (`induction_pol2`, `lorentz_pol2`); l = 3..6, m = 0..3 (~40 minutes) |
| `mk_run.py NAME 'par.x = ..' [solve]` | builds a run directory from the repo's `bin/` with the magnetic test defaults ('Wu 2005', r_out 0.99, ricb 0.35, no-slip, B0 'axial', insulating walls, Ek = Em = 1e-3, Λ = Pm = 1, m 1, N 96, lmax 24, 2 ranks) and runs submatrices (and with `solve` the whole pipeline) |
| `row_check.py [lmax] [tol]` | every magnetic row block of the run, applied to random fields and read on an r grid in its Gegenbauer basis, against derive.py evaluated with the run's own profiles; unweighted, so it sees the rows near the wall. 'Wu 2005' with r_out 0.99 needs N 192 for 1e-14 (1e-7 at N 96: the series of (ln ρ0)' converge slowly, its pole is at r = 1/r_out). Step 5: B0 of degree 2 (Δl ±2) and the full sphere (parity-reduced bases, r in [0.1, 1]; 'Wu 2005' needs N 288 there) |
| `bin/ru_insane.py` | with magnetic = 1 also: the b block Hermitian positive and the diffusion Hermitian negative under the radial pairing, and for B0 'axial' with no-slip walls the energy identity ∫ρ0 u*⋅𝐅_L dV = -Hendrik conj(∫b*⋅∇×(u×B0) dV), which ties the (ln ρ0)' terms of the Lorentz rows to those of the induction rows (switching off one of them gives 2e-3 to 0.2) |
| `budget_check.py [tol]` | after spin_doctor: residσ, residω, resid𝐛, resid𝐄 of every solution below tol (1e-8): the budgets use utils4pp's own formulas for the Lorentz force and the induction |
| `malkus_check.py [Ek] [N] [lmax]` | the Malkus field B0 = Le s φ̂ against Malkus (1967): the frequency shift of inertial modes is Le² m(m - λ)/λ in the ideal limit; checks that it is quadratic in Le and, for the least damped m = 1 mode near 0.75i, within 10 % at Ek = Em = 1e-5 (0.4-1 % at 1e-6) |
| `dr_induction_check.py [--quick]` | NEW (2026-10-30): the induction by a differential rotation, ∇×(U0×b), of `diff_rot.induction_block` against `derive.rows_dr` (sympy), every block for n = 0–6, m up to 5, l up to 25 (sympy's quadrature limits the agreement to ~1e-10 at l 20–25) |
| `algebra_check.py [--quick]` | NEW (2026-10-30): the couplings of `bin/mag_algebra.py` (any background degree) against derive.py, poloidal and toroidal fields of degree 1–4 |
| `rigid_check.py [c]` | NEW (2026-10-30): ΔΩ = c is a rigid rotation, so λ(ΔΩ = c) = λ(Gaspard 1 + c) - i m c exactly, for 'axial', 'Luo_S2', 'Malkus', a conducting inner core, a full sphere and a 'general' field (builds and solves runs in the current directory) |
| `continuity.py [eps ...]` | 'Wu 2005' with r_out = sqrt(eps) (stratification O(eps)) against 'Boussinesq': (λ - λ_uniform)/eps must tend to a constant |

`derive.py --b0 pol1|pol2|tor1|tor2` (NEW 2026-10-29) picks the background field from the command line. The closed forms were
fitted from derive.py with `tools/derive/fit_rows.py` and written as code with `tools/derive/gen_table.py`; `docs/derivations.md`
is the guide to all of these, with worked examples.

Results of 2026-10-28 are in `docs/magnetic.md` (Checks). `derive.py` was used to derive, not only to check, the anelastic
terms: the derivation, the row checks and the budgets are independent of each other.

On chandler, slepc4py 3.26 has no `SLEPc.COMM_WORLD`, which `bin/solve.py` uses: the solves fail there until that is fixed
(`PETSc.COMM_WORLD` is the same communicator); slepc4py 3.25 (mongo) works.
