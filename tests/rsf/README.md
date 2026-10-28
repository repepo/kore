# Checks of the random Reynolds-stress forcing (forcing = 11)

These scripts check `bin/reynolds_stress.py`, which builds the right-hand sides of forcing 11, and the frequency scan of
`bin/solve.py`. The physics and the formulas are in `docs/reynolds-stress.md`, and how to run the forcing is in
`docs/forcing.md`.

## Running them

```shell
tests/rsf/run_all.sh [WORKDIR]
```

builds the test runs with `tests/heat_dr/mk_run.py` (in `tests/rsf/runs/` by default), runs every check, prints one
verdict line per check and exits with 1 if any failed. It needs numpy, scipy and mpi4py, and petsc4py and slepc4py with
MUMPS for the solves. It takes about half an hour on one core. `MPIEXEC` replaces `mpiexec`; it must start ranks of
the MPI that mpi4py was built with, which the script checks, since otherwise two ranks would be two independent copies
and the multi-rank checks would test nothing.

Each check can also be run by hand, from a kore run directory built for forcing 11: a copy of `bin/` with its
`parameters.py`, the `*.mtx` files of `submatrices.py`, and the `A.npz`, `B.npz` and `B_forced.npz` of `assemble.py`.

## The runs

All are hydrodynamic and rigidly rotating.

- `sh48`, `sh128`: a shell (ricb 0.35, m 1) at N 48 and 128.
- `fs160`, `fs192`: a full sphere with symm −1, at m 1 and m 2, the two parities of $P$.
- `shpow`, `fspow`: other powers of $r$ and $\rho_0$ in the rows (`rpower_u` 6, `rpower_v` 4, `rhopower_u` =
  `rhopower_v` = 2), in a shell at N 192 and a full sphere at N 320.
- `sun`, `sun_mlt`: a solar-like full sphere (the `'8-term fit'` structure, N 192), with $\nu_t$ from a viscosity step
  between a convection zone and a radiative floor, $\ell_c$ from $H_p$, and `rsf_theta` `'nu'` and `'mlt'`.
- `sun_cut`: as `sun_mlt`, with the inner cut `rsf_rcut` 0.45 of width 0.2, across the tail of the viscosity step.
- `sp`: a shell at N 48 and lmax 24 with no-slip walls, $\ell_c$ = 0.1 and 40 realisations, for the solves.

The first six runs use $\Theta = 1$, $\nu_t = \nu$, $\ell_c = 0.05$ and `rsf_taper` 0.1.

## The checks

**`builder_check.py [tol]`.** A force equal to a velocity field, $\mathbf f = \mathbf u$, must give kore's inertia rows
(`operators.inertia`) applied to $\mathbf u$, for every $l$ and random smooth $P$ and $T$. This checks the projection
onto the rows, $b_u = r^{\mathrm{rpower\_u}-2}\rho_0^{\mathrm{rhopower\_u}}[f^P - (rf^B)']/\mathcal L^2$ and
$b_v = -r^{\mathrm{rpower\_v}-1}\rho_0^{\mathrm{rhopower\_v}}f^C/\mathcal L^2$, the Chebyshev transform, the change to
the Gegenbauer basis, the parity rows of a full sphere and the offset of the boundary rows. It passes below $10^{-9}$
($10^{-8}$ for `sh48`, whose profiles are truncated at N 48).

**`noise_check.py [tol]`.** An exact check, without sampling. Each noise function (one centre, one $c_k$, one $l$) is
paired with random smooth fields $\mathbf a$ that vanish near the walls, against targets built from kore's profiles and
the documented laws, not from `reynolds_stress.py`. With the run's own `rsf_*` parameters:

- **B0**: the amplitude of the noise against the documented formulas: $\Theta$ from `rsf_theta` times `rsf_scale`,
  $\nu_t$ with its floor, $\ell_c$, the angular factor, the taper and the normalisation (below $10^{-12}$).
- **B1**: the rows against the weak form $-\tfrac12\int\rho_0\thinspace\mathsf S(\mathbf a)^*:\mathsf s\thinspace dV$
  computed from the stress itself, for smooth fields (below `tol`, $10^{-6}$ by default).
- **B5**: for fields localised at several radii, also where $\nu_t = 0$, the variance
  $\mathbb E\lvert\langle\mathbf a, \mathbf f\rangle\rvert^2$ from the rows against its upper bound
  $\int\rho_0\Theta\nu_t\tau^2 e^{-\mathcal L\ell_c^2/(2r^2)}\lvert\mathsf S(\mathbf a)\rvert^2\thinspace dV$ (every
  ratio below $1 + 10^{-4}$). A noise that leaked below the convection zone would exceed it by orders of magnitude.

With $\Theta = 1$, `rsf_scale` 2.5, $\nu_t = \nu$, `rsf_angular` 0 and the run's `rsf_lc`:

- **B2**: the covariance in the white limit, per channel ($c_1$, $c_2$, $c_5$ in the rows of $P$; $c_3$, $c_6$ in those
  of $T$) and per $l$, against the same channel and $l$ of $2.5\int\rho_0\nu\tau^2\lvert\mathsf S(\mathbf a)\rvert^2\thinspace dV$
  (every ratio within $10^{-2}$ of 1).
- **B3**: kore's viscous rows against the strain rate,
  $-2\thinspace\mathrm{Re}\langle\mathbf a, V\mathbf a\rangle = \int\rho_0\nu\lvert\mathsf S(\mathbf a)\rvert^2\thinspace dV$
  (below `tol`): the noise is the partner of kore's own damping.
- **B4**: the same ratio at the run's $\ell_c$, printed but not judged.

**`spectrum_check.py pick | fields | check`.** The forced solves against the exact expected spectrum, which the script
computes with scipy from the map of the noise, for a run small enough for a dense eigensolve.

- `pick`: picks the least damped mode with no other eigenvalue within $4\lvert\mathrm{Re}\thinspace\lambda\rvert$, and
  writes `rsf_freqs.npy` (21 frequencies over $\mathrm{Im}\thinspace\lambda \pm 5\lvert\mathrm{Re}\thinspace\lambda\rvert$) and `mode.npz`.
- `fields`: after `solve.py` without `rsf_frequencies`, every column of the fields solves its forced problem (residual
  below $10^{-9}$), and column 0 of `B_forced.npz` is the noise map times the draws of `default_rng([rsf_seed, 0])`.
- `check`: after `solve.py` with `rsf_frequencies = np.load('rsf_freqs.npy')`, the kinetic energy of every realisation
  in `rsf_ke.dat` against a scipy solve (below $10^{-8}$), and `rsf_spectrum.dat` against the exact mean (every
  deviation below 4.5 standard errors). It also prints the energy of the mode, its normal-mode estimate and the
  fluctuation–dissipation value $\Theta_{\rm eff}f_t$.

**`regress_solve.sh OLD_KORE [WORKDIR]`.** An eigenvalue run (thermal, conical differential rotation) and a forced run
(forcing 10), built and solved with the `bin/` of another checkout `OLD_KORE` and with this one, on one rank and one
thread: `A`, `B`, `B_forced`, `eigenvalues0.dat` and the field files must be bitwise equal. `run_all.sh` does not run
it. `tests/diff_rot/regress.sh OLD_KORE` does the same for the matrices of 13 eigenvalue configurations.

`rsf_common.py` holds the pieces the checks share: the energy pairing of kore's rows with a field (as `bin/ru_insane.py`),
the test fields, and their strain rate in tensor harmonics.

## Notes

- The kinetic energy of `spectrum_check.py` is the energy pairing of `bin/ru_insane.py` applied to the inertia rows. It
  equals `utils4pp.kinetic_energy` only when the boundary terms of the integration by parts vanish, so the spectrum run
  has no-slip walls; the script prints both for three solutions.
- The Chebyshev tail that `noise_check.py` prints is that of single noise functions, which the full-sphere runs do not
  resolve at $\ell_c = 0.05$. B1 uses smooth fields, which do not see it, and the margin of B5 absorbs it. In a
  production run the tail that `assemble.py` prints is that of the realisations: raise N or $\ell_c$ when it warns.
