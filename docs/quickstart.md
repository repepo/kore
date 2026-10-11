# Quick start

This page runs the case that comes with the code, a torsional Alfvén mode, from start to finish. It assumes PETSc and SLEPc are installed and their environment is loaded (see [Installation](installation.md)).


## Get the code

Clone the repository:
```Shell
git clone https://github.com/repepo/kore.git
```
For regular work, make a copy of the source directory and keep the original clean. For example:
```Shell
cp -r kore kwork1
cd kwork1
```
All the commands below are run from this top folder (`kwork1` here). The matrices, solutions and results are written there.


## Set the parameters

Every parameter, its default value and the available options are defined in `bin/params_default.py`. The file you edit is `bin/parameters.py`. It creates the parameter object `par`, changes only what differs from the defaults, and ends with `par.set_scales()`, which computes the derived parameters (the magnetic Ekman and Lehnert numbers, the time scale factor, the resolution, the solver target, ...). The version in the repository sets up a torsional-mode problem:

```python
par = default_params()

# Torsional-mode setup: FDM field with a thin conducting wall at the CMB
par.Ek       = 10**-7
par.magnetic = 1
par.B0       = 'FDM'     # torsional waves need B_s != 0, the axial field has none
par.mantle   = 'TWA'
par.c_cmb    = 1e-2
par.c1_cmb   = 1.5e-2    # magnetic torque ~ viscous torque
par.Lambda   = 10        # Le = sqrt(Lambda*Ek/Pm) = 3.2e-3
par.Pm       = 0.1       # Lundquist number Le/Em = 3.2e3

par.rtau     = -0.0046   # aims at the mode lambda = -0.00463 + 0.0115i
par.itau     = 0.0115

par.set_scales()
```

The other parameters keep their defaults: an inner core of radius 0.35 (the CMB radius is 1), no-slip boundaries, an insulating inner core, $m = 0$, the rotation time scale, and $N$ = 232 Chebyshev polynomials with $l_{max}$ = 231 at this Ekman number. Always set parameters before `par.set_scales()`: a change made after it does not reach the derived parameters. [Setting up a problem](setup.md) describes all the parameters.


## Run

A run has four steps. `ncpus` is the number of MPI processes. For the first two steps it must equal `par.ncpus` (4 by default), because the spherical-harmonic degrees are distributed in equal blocks among the processes.

1. Generate the submatrices, the building blocks of the operators, as `*.mtx` files:
    ```Shell
    ./bin/submatrices.py ncpus
    ```
2. Assemble the matrices $A$ and $B$ of the eigenvalue problem $A\,x = \lambda B\,x$ (or the matrix and right-hand side of a forced problem), written as `A.npz` and `B.npz`:
    ```Shell
    mpiexec -n ncpus ./bin/assemble.py
    ```
3. Solve. The number of processes can differ from `par.ncpus` here. Solver options can be appended to the command (see [Solver settings](solver.md)):
    ```Shell
    mpiexec -n ncpus ./bin/solve_nopp.py
    ```
4. Postprocess: compute energies, dissipation, torques and the residuals that tell whether the solutions are accurate:
    ```Shell
    ./bin/spin_doctor.py ncpus
    ```

The script `bin/runKore.sh` runs the four steps with 4 processes.


## What you should see

`solve_nopp.py` prints the eigenvalues it found and their residuals. `spin_doctor.py` then prints a summary table, one row per solution:

```
  ★      Damping σ       Frequency ω      resid0     resid𝐮     resid𝐛     residθ     resid𝐋    Tor/Pol    Mag/Kin     |𝚪|mag    |𝚪|magIC
 ‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾
  0     -0.00464013       0.01150398     8.46e-13   1.89e-07   1.55e-06   0.00e+00   5.30e-07   1.56e+02   1.01e+00   3.07e-05   0.00e+00
  1     -0.00435660       0.01030532     6.36e-11   1.89e-07   1.97e-06   0.00e+00   3.91e-07   1.80e+02   1.01e+00   4.06e-05   0.00e+00
  2     -0.00494936       0.01270521     1.54e-10   1.45e-07   1.18e-06   0.00e+00   4.37e-07   1.38e+02   1.01e+00   2.86e-05   0.00e+00
```

Solution 0 is the mode closest to the target. Its eigenvalue is $\lambda = \sigma + i\omega$, with damping rate $\sigma$ and frequency $\omega$ in units of the rotation rate. It is a torsional mode: the flow is almost entirely toroidal (Tor/Pol = 156), its kinetic and magnetic energies are equal (Mag/Kin = 1.01), and the magnetic torque on the conducting mantle ($|\Gamma|_{mag}$) is nonzero. All the residuals are far below $10^{-4}$, so the solution is accurate. The whole run takes about half a minute on 4 cores of a desktop machine.

[Output and postprocessing](output.md) explains every column, the files that are written and the plotting tools.


## Output files

The solutions are written as `real_*.field` and `imag_*.field` (one column per solution) and, for eigenvalue problems, `eigenvalues0.dat`. `spin_doctor.py` appends one line per solution to `params.dat`, `eigenvalues.dat`, `flow.dat`, `magnetic.dat` (with a magnetic field) and, when they are included, `thermal.dat`, `compositional.dat` and `rotdyn.dat`. Because these files are appended to, several runs in the same folder build up a table of results.


## Next steps

- Change the physics and the geometry: [Setting up a problem](setup.md).
- Add a conducting inner core or mantle: [Inner core and mantle](boundaries.md).
- Run the test cases, which reproduce published results: [Examples and tests](examples.md).
