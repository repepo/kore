# Checks of the random Reynolds-stress forcing (forcing = 11)

Companion of `bin/reynolds_stress.py` (its docstring has the physics, the discretisation and the normalisation; the
derivation is the project note `kore/convective-forcing/reynolds-stress-forcing.md`). The checks run from a kore run
directory (a `bin/` copy, `parameters.py`, the `*.mtx` files of `submatrices.py`, and the `A.npz`, `B.npz`,
`B_forced.npz` of `assemble.py`, which writes all three for forcing 11). `run_all.sh [WORKDIR]` builds the runs with
`tests/heat_dr/mk_run.py` under `tests/rsf/runs/`, prints one verdict line per check and exits 1 if any failed; the
solves need petsc4py/slepc4py with MUMPS. `MPIEXEC` overrides `mpiexec`; it must start ranks of the MPI that mpi4py was
built with (the script checks), otherwise 'two ranks' are two independent copies and the multi-rank checks test nothing.

The runs: `sh48`, `sh128` (shell, ricb 0.35, m 1), `fs160` (full sphere, m 1, symm −1), `fs192` (m 2, symm −1), `shpow`
and `fspow` (rpower_u 6, rpower_v 4, rhopower 2, shell N 192 and sphere N 320: the pairing weight ρ₀^(1−rhopower) r^(3−rpower) amplifies the truncation of the rows,
so B1 needs more N than with the default powers), all hydro, uniform rotation, Θ = 1, ν_t = ν, l_c 0.05,
taper 0.1; `sun` and `sun_mlt` (full sphere with the '8-term fit' structure, N 192, ν_t from a convection-zone step over a radiative
floor, l_c from H_p, rsf_theta 'nu' and 'mlt'); `sun_cut` (as `sun_mlt`, with the inner cut-off `rsf_rcut` 0.45,
width 0.2, across the tail of the viscosity step); `sp` (shell N 48, lmax 24, no-slip, l_c 0.1, 40 realisations) for the
solves.

| script | what it checks |
| --- | --- |
| `builder_check.py [tol]` | a force equal to a velocity field, f = u (fP = U, fB = V, fC = W), must give `op.inertia` applied to u, for every l and random smooth P, T: the projection b_u = r^(rpower_u−2)ρ₀^rhopower_u [fP − (r fB)′]/L², b_v = −r^(rpower_v−1)ρ₀^rhopower_v fC/L², the Chebyshev transform, the C^(4)/C^(2) basis change, the parity rows (ricb = 0) and the bc offset. Pass < 1e-9 (1e-8 for `sh48`, whose profiles are truncated at N 48) |
| `noise_check.py [tol]` | exactly, without sampling: every noise function (one centre, one c_k, one l) paired with random fields a that vanish near the walls; the targets are built in the script from kore's profiles and the documented laws, not from `reynolds_stress.py`. With the run's parameters: **B0** the noise amplitude τA_lk S^(−1/2) at the nodes against the documented formulas (Θ from `rsf_theta` times `rsf_scale`, ν_t with its floor, l_c, the angular factor, the taper, the normalisation; < 1e-12); **B1** kore's rows against the weak form −∫ρ₀ e(a)*:s dV from the stress itself, for smooth fields (< 1e-6); **B5** for fields localised at several radii (also where ν_t vanishes), E\|⟨a, f⟩\|² from the rows against the Cauchy–Schwarz bound 4∫ρ₀Θν_tτ²[e^(−Ll_c²/2r²)]\|d(a)\|² dV (every ratio < 1 + 1e-4): noise reaching below the convection zone would exceed the bound by orders of magnitude. With Θ = 1, `rsf_scale` 2.5, ν_t = ν, rsf_angular 0 and the run's `rsf_lc` (0.05 if None, so not the H_p-based l_c of the sun runs): **B2** the covariance in the white limit (l_c 0.003, fine quadrature) per channel c1, c2, c5 (P) and c3, c6 (T) and per l, Σ\|⟨a, f⟩\|² against the same channel and l of 4∫ρ₀ 2.5ν τ²\|d(a)\|² dV (every \|ratio − 1\| < 1e-2; the deviation is O(l_c²)); **B3** kore's viscous rows against the traceless strain, −2Re⟨a, Va⟩ = 4∫ρ₀ν\|d\|² dV (< 1e-6): the noise is the partner of kore's own damping; **B4** (printed) the same at that l_c |
| `spectrum_check.py pick` | dense eigensolve of a small run (sizmat ≲ 3000); picks the least damped mode with no other eigenvalue within 4\|σ\|; writes `rsf_freqs.npy` (21 frequencies over ω_n ± 5\|σ_n\|) and `mode.npz` |
| `spectrum_check.py fields` | after `solve.py` (no `rsf_frequencies`): every column of the written fields solves A x = b_j (< 1e-9), and column 0 of B_forced.npz is the noise map M times the draws of `default_rng([rsf_seed, 0])` (< 1e-12) |
| `spectrum_check.py check` | after `solve.py` with `rsf_frequencies = np.load('rsf_freqs.npy')`: the KE of every realisation in `rsf_ke.dat` against `utils4pp.kinetic_energy` of a scipy solve (< 1e-8), and `rsf_spectrum.dat` (mean KE over `rsf_nsamp` realisations, standard error) against the exact E[KE(w)] = Σ_cols KE(F(w)⁻¹M) (pass max\|z\| < 4.5; the mean z² is printed, not judged, since the same realisations are used at every w). Also the mode: its energy P_n from the left eigenvector, the normal-mode estimate E\|⟨a_n, f⟩\|²/(2\|σ_n\|) and the fluctuation-dissipation value Θ f_eff (printed) |
| `regress_solve.sh OLD_KORE [WORKDIR]` | existing runs unchanged: an eigenvalue run (thermal, conical diff rot) and a forced run (forcing 10) built and solved with an older checkout and with this one (one rank, one thread); A, B, B_forced, `eigenvalues0.dat` and the field files must be bitwise equal. `tests/diff_rot/regress.sh OLD_KORE` does the same for the matrices of 13 eigenvalue configurations |

The KE of `spectrum_check.py` is the energy pairing of `bin/ru_insane.py` applied to the inertia rows; it equals
`utils4pp.kinetic_energy` only if the integration-by-parts boundary terms vanish, so the spectrum run has no-slip walls
(with stress-free walls the tangential velocity at the wall spoils it; the script prints both on 3 solutions).

Results (2026-10-20, container with scipy and Open MPI, `solve.py` replaced by a scipy stand-in for the solves; the
same on chandler with PETSc/MUMPS, where all checks pass with solves to 5.6e-12, the 1-vs-2-rank scan equal to 1.2e-13,
and `regress_solve.sh` and `tests/diff_rot/regress.sh` bitwise equal; chandler's slepc4py 3.26 has no `SLEPc.COMM_WORLD`,
which `solve.py`'s eigensolver uses before and after this change, so the eigenvalue run of `regress_solve.sh` needed
`SLEPc.COMM_WORLD = PETSc.COMM_WORLD` from a `sitecustomize.py` on PYTHONPATH there). builder: 9.5e-10 (sh48), 1.5e-15 (sh128),
5.4e-15 (fs160), 2.4e-15 (fs192), 9.9e-15 (shpow), 4.3e-15 (fspow), 4.9e-15 (sun). noise (sh128 / fs160 / fs192 / shpow /
fspow / sun / sun_mlt): B0 ≤ 2.8e-15, \|S − 1\| ≤ 9e-4 for constant l_c; B1 8.3e-8 / 3.4e-8 / 5.3e-10 / 3.2e-12 / 9.2e-8 /
3.2e-9 / 1.4e-9; B5 ratios ≤ 0.03 / 0.11 / 0.11 / 0.03 / 0.11 / 0.15 / 0.16, and in the sun runs E\|⟨a, f⟩\|² for the
field at r = 0.45 is 4e-11 ('nu') and 1.5e-6 ('mlt') of that at 0.85, following the erf tail of ν_t; B2 per channel
0.9951–0.9993, per l within 0.0052 of 1; B3 ≤ 9e-12; B4 0.44–0.48. 2-rank assembly (real 2 ranks) identical. Solves (sp):
fields 1.1e-13, column 0 2.2e-16; KE of every realisation 7.2e-13; spectrum max\|z\| 0.72 (40 realisations); mode
−0.0487 − 0.3284i: P_n 0.477, normal-mode estimate 0.461, Θ f_eff 0.461, −⟨a, Va⟩/\|σ\| 1.0002 (all of its damping is
viscous); the exact peak is 1.44 × the single-mode Lorentzian (the other modes' broad background).

`rsf_rcut` (2026-10-21, container): `sun_cut` builder 4.9e-15, B0 1.3e-15, B1 1.4e-9, B5 ≤ 0.13, with the bound at
r = 0.3 and 0.45 at 7e-22 and 9e-19 of its convection-zone value (1.5e-6 at 0.45 without the cut in `sun_mlt`).
Mutants of the cut (derivative sign, missing second derivative, edge position, cut ignored) all fail B0, B1 or B5;
without `rsf_rcut` B_forced is bitwise unchanged.

Mutation test (2026-10-20): 18 plausible bugs planted one at a time in `reynolds_stress.py` (amplitude laws, floor,
`rsf_scale`, angular factor, χ_k at high l, normalisation, its exponent and derivative sign, taper position and
derivative, lattice stopping at the walls, signs in f^P and (f^B)′, l_c′, the full-sphere row parity). Each is caught by
`builder_check` or `noise_check` in at least one run that uses the code it changes; the shell runs (constant Θ and
l_c) do not use the 'nu' law, l_c′ or S′, and in the sun runs S^(−1/2) and the taper absorb a lattice that stops at
the walls.

The Chebyshev tail that `noise_check.py` prints is that of single noise functions, which N 160–192 in the full sphere
does not resolve at l_c 0.05 (1e-2). B1 uses smooth fields, which do not see it; the localised fields of B5 do (strong
and weak forms differ by 4e-2 for them in fs160), which B5's margin absorbs. For production, the tail printed by `assemble.py` is that of the realisations:
raise N or l_c when it warns (> 1e-6).
