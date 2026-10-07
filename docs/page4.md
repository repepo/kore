# Using Kore

## Solver Settings

The solver options live in `bin/parameters.py`. Most of them are collected in the `petsc_opts` dictionary, which `solve_nopp.py` loads into PETSc's options database at start-up. Each key is a PETSc/SLEPc/MUMPS option name without the leading `-`. Only the group that matches the problem type is used: `eps_*` and `st_*` options for eigenvalue problems (`forcing = 0`), and `ksp_*`, `pc_*` and `mat_*` options for forced problems. Any option given on the command line takes precedence over `petsc_opts`, for example:

```Shell
mpiexec -n 4 ./bin/solve_nopp.py -st_mat_mumps_icntl_35 0
```

The defaults described below are the recommended settings for production runs.

### Eigenvalue problems

**Eigensolver.** `solve_nopp.py` solves the generalized eigenvalue problem $A\,x = \lambda B\,x$ with SLEPc's default Krylov-Schur method. It looks for `nev` eigenvalues (default 3) closest to the target $\tau$ = `rtau + 1j*itau`, as selected by `which_eigenpairs = 'TM'` (target magnitude). The real part of $\tau$ is the damping and the imaginary part the frequency of the modes sought. `tol` and `maxit` set SLEPc's convergence tolerance and maximum number of restarts. The size of the Krylov search space (`eps_ncv`) is left at SLEPc's default.

**Shift-and-invert** (`'st_type': 'sinvert'`). SLEPc does not work with $A$ and $B$ directly but with $(A-\tau B)^{-1}B$. The eigenvalues closest to $\tau$ become the largest ones of this operator, so the method converges in a few iterations even deep inside the spectrum. Each iteration needs one solve with $A-\tau B$, so that matrix is factorized once, at the start.

**Direct LU with MUMPS** (`'st_pc_factor_mat_solver_type': 'mumps'`). The factorization of $A-\tau B$ is done by the parallel sparse direct solver MUMPS. Factorization takes most of the run time and memory.

- `'st_mat_mumps_cntl_1': 1e-8` is MUMPS's relative pivoting threshold. MUMPS's default (0.01) makes it delay many pivots on Kore's matrices, so large problems run out of workspace (error `INFOG(1) = -9`) and the factorization is slow. A small threshold avoids this, while still guarding against an exactly zero pivot. Do not raise it above about 1e-8 when block low-rank factorization is on (see the next item): larger values can spoil the eigenvectors of large problems.
- `'st_mat_mumps_icntl_35': 2` switches on block low-rank (BLR) factorization. MUMPS compresses blocks of the factors that are numerically of low rank, which reduces both memory and factorization time considerably for large matrices.
- `'st_mat_mumps_cntl_7': 1e-14` is the BLR compression tolerance. It is set close to machine precision so that BLR does not reduce accuracy. Looser values (e.g. 1e-12 or 1e-10) save a little more memory but degrade the eigenvectors.
- MUMPS's own row and column scaling of the matrix (`icntl_8`) is left at its default and should stay on.

**Pre-scaling** (`prescale = 1`, in `parameters.py` after `petsc_opts`). The rows and columns of Kore's matrices differ in size by many orders of magnitude. With `prescale = 1`, `solve_nopp.py` equilibrates $A-\tau B$ before the factorization (Ruiz scaling: rows and columns are scaled repeatedly until their largest entries are all of order one). It then solves the scaled problem $(D_r A D_c)\,y = \lambda\,(D_r B D_c)\,y$. The eigenvalues are the same, and the eigenvectors are mapped back, $x = D_c\,y$, before they are written to disk. Pre-scaling reduces the condition number of the matrix by many orders of magnitude and gives more accurate eigenvectors, at the cost of a few seconds. It is used for eigenvalue problems only. Set `prescale = 0` to switch it off.

**Eigenvector normalisation.** Eigenvectors are only defined up to a complex factor. Before they are written, each eigenvector is scaled so that its total kinetic plus magnetic energy, integrated over the fluid outer core, equals 1, using the same definitions as `spin_doctor.py`. Energies, dissipations and torques reported by `spin_doctor.py` therefore refer to a mode of unit total energy. The complex phase is left as returned by SLEPc.

**Error report** (`'eps_error_relative': '::ascii_info_detail'`). After the solve, `solve_nopp.py` prints each eigenvalue with its relative residual $\|Ax-\lambda Bx\|/\|\lambda x\|$. With `prescale = 1` this residual refers to the scaled problem. A small value shows that SLEPc converged, but it does not guarantee an accurate eigenvector. Always check the residuals computed by `spin_doctor.py` (see below).

**Optional settings.** These are commented out in `petsc_opts` and can be switched on when needed:

- `'eps_balance': 'twoside'` balances the matrices before the solve. It gives cleaner eigenvalues for final runs, at roughly 50% more run time.
- `'st_mat_mumps_icntl_14'` increases MUMPS's workspace (in percent). It is only needed if `INFOG(1) = -9` still appears.
- `'st_mat_mumps_icntl_22': 1` stores the factors out of core, on disk, which roughly halves the memory needed. Set `'st_mat_mumps_ooc_tmpdir'` to a directory on a real disk, not on a memory-backed `/tmp`.
- `'st_mat_mumps_icntl_28': 2` with `'st_mat_mumps_icntl_29': 2` makes MUMPS compute its fill-reducing ordering in parallel, with ParMETIS.

### Forced problems

For forced problems (`forcing > 0`) Kore solves a single linear system. The defaults are a direct solve: `'ksp_type': 'preonly'`, `'pc_type': 'lu'` and `'pc_factor_mat_solver_type': 'mumps'`. Iterative solvers (PETSc's default GMRES with ILU) do not converge reliably on these matrices. If MUMPS reports `INFOG(1) = -9`, uncomment `'mat_mumps_cntl_1'`. Pre-scaling and the energy normalisation are not applied: the solution keeps the amplitude set by the forcing.

### Resolution

The Chebyshev truncation `N` is set by `Ncheb(Ek)` at the top of `parameters.py`, and the spherical-harmonic truncation `lmax` follows from `N` (with `g = 1`, `lmax = N - 1` for `m = 0`). `Ncheb` grows as $E_k^{-0.242}$; for example it gives `N` = 136, 232, 400 and 688 at $E_k$ = 1e-6, 1e-7, 1e-8 and 1e-9. It was obtained for a torsional-mode problem. Other problems (stronger fields, other boundary conditions, buoyancy) may need a higher resolution. Keep `lmax` close to `N`: a too small `lmax` changes the eigenvalues without showing up in the residuals below.

### Checking the results

`spin_doctor.py` computes, for each solution, residuals of the kinetic, magnetic and thermal energy balances (`resid0`, `resid𝐮`, `resid𝐛`, `residθ`) and, for $m=0$, of the axial angular momentum balance (`resid𝐋`). These are independent of the solver and are the best test of a solution's accuracy. As a rule of thumb they should all be below about 1e-4. If they are not, increase the resolution. Two exceptions:

- With `rotdyn = 1`, `resid0` stays at about 1e-4 to 1e-3, because the work done by the rotating mantle and inner core on the fluid is not included in that balance yet.
- The eigenvectors of the modes found away from the target are usually less accurate than the one closest to $\tau$. To get a particular mode as accurately as possible, place the target on it.
