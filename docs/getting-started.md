# Getting started

This page goes through one complete run of kore, from the parameter file to a checked eigenvalue. The example is the
simplest of the [Benchmarks](benchmarks.md): an inertial mode of an incompressible fluid in a differentially rotating
spherical shell, from Baruteau & Rieutord (2013, fig. 10). It takes about a minute on four cores and needs less than
2 GB of memory.

## 1. Prepare the environment

Install PETSc, SLEPc and their python bindings as described in [Installation](installation.md), then load the
environment in the terminal you will use:

```shell
source $HOME/kore_env.sh
```

## 2. Get the code and make a run directory

```shell
git clone -b anelastic_log https://github.com/repepo/kore.git
mkdir br10
cp -r kore/bin br10/
cd br10
```

kore runs from a directory that holds a copy of `bin/`, and writes all its files there. Keep the clone itself clean and
make one run directory per set of parameters.

## 3. Write the parameter file

Every script reads its parameters from `bin/parameters.py`. Replace the content of that file with

```python
from params_default import default_params

par = default_params()

# Time unit 1/Ω and the Ekman number E = ν/(Ω R²)
par.timescale = 'rotation'
par.Ek = 8e-7
par.set_scales()                  # Gaspard = 1, ViscosD = Ek

# Fluid and container: incompressible, a shell with inner radius 0.2, stress-free walls
par.structure = 'Boussinesq'      # ρ0 = 1
par.thermal = 0                   # no heat equation
par.ricb = 0.2
par.bci = 0
par.bco = 0

# Differential rotation Ω = Ω_ref r^σ, with σ = 0.5
par.diff_rot = 1
par.diff_rot_type = 'shellular'
par.diff_rot_param = 0.5
par.thermal_wind = 0

# Axisymmetric, equatorially symmetric modes
par.m = 0
par.symm = 1

# Resolution and cores: lmax - m + 1 must be a multiple of 2*ncpus
par.N = 132
par.lmax = 271
par.ncpus = 4

# Look for the 5 eigenvalues nearest the target rtau + i*itau
par.rtau = -0.008
par.itau = 1.23
par.nev = 5
```

`default_params()` sets every parameter to its default. `bin/params_default.py` lists them all, with comments. The
parameters used here:

- `timescale` and `set_scales()` choose the unit of time and set the four dimensionless numbers `Gaspard`, `Beyonce`,
  `ViscosD` and `ThermaD` from it ([Anelastic equations](anelastic.md#non-dimensionalization)). The unit of length is
  the radius of the outer wall.
- `structure`, `thermal`, `diff_rot` and their options set the background: density, entropy gradient, diffusivities and
  rotation ([Background model](background.md)).
- `ricb` is the inner radius (0 for a full sphere), and `bci`, `bco` the conditions at the inner and outer walls
  (0 stress-free, 1 no-slip).
- `m` is the azimuthal wavenumber, and `symm` the equatorial symmetry of the flow (1 symmetric, −1 antisymmetric).
- `N` is the number of Chebyshev polynomials in radius, and `lmax` the highest spherical-harmonic degree.
- `rtau` + i `itau` is the target: the solver returns the `nev` eigenvalues nearest to it.

kore's eigenvalue is $\lambda = \sigma - i\omega$, for perturbations $\propto e^{\lambda t + im\phi}$ in the frame
rotating at $\Omega_{\rm ref}$, with $\sigma$ the growth rate (negative when damped) and $\omega$ the frequency, here in
units of $\Omega_{\rm ref}$. The target above is the published mode.

## 4. Build the matrices

```shell
./bin/submatrices.py 4
mpiexec -n 4 ./bin/assemble.py
```

`submatrices.py` computes one sparse matrix per radial operator of the equations (the `*.mtx` files), using 4
processes. `assemble.py` combines them over all degrees $l$ into the matrices $A$ and $B$ of the eigenproblem
$A\mathbf x = \lambda B\mathbf x$ (`A.npz`, `B.npz`). The number of MPI ranks must equal `par.ncpus`.

## 5. Solve

```shell
mpiexec -n 4 ./bin/solve.py
```

`solve.py` factors $A - \tau B$ once and finds the eigenvalues nearest the target $\tau$. It prints them with their
residuals:

```
  -0.007855+1.227914i       5.32254e-12
  -0.019000+1.231247i       3.10986e-11
  -0.019022+1.223615i       5.29565e-11
  -0.019437+1.238668i       3.77176e-11
  -0.019738+1.216233i       1.01615e-10
```

The first one is the mode of the paper. It writes the eigenvalues to `eigenvalues0.dat` and the eigenvectors to
`real_flow.field` and `imag_flow.field`. The residuals printed here only say that the solver converged. They do not
say whether the resolution is sufficient.

## 6. Check the solution

```shell
./bin/spin_doctor.py 4
```

`spin_doctor.py` rebuilds each solution in physical space and checks the balances it must satisfy. The first columns of
its table are

```
  ★          Re λ              Im λ          𝒯/𝒫       residσ     residω    Peak ℓ   ℓ-Width   ℓ-Convergence
  0      -0.007855111       1.227914453    1.12e+00   1.75e-05   1.01e-08     3         44        4.68e-12
  1      -0.018999641       1.231246928    1.14e+00   3.12e-04   2.23e-06     5         74        2.71e-10
```

A mode is resolved when its energy residual residσ is below $10^{-5}$ and its ℓ-convergence below $10^{-4}$ (with
`thermal = 1`, the entropy residual resid𝑠 must also be below $10^{-5}$). The ℓ-convergence is fine, but residσ is
$1.8\times10^{-5}$ for the first mode and $3\times10^{-4}$ for the others: the radial resolution is not enough.
`spin_doctor.py` also appends one row per mode to `params.dat`, `flow.dat` and `eigenvalues.dat`
([Output files](output.md)).

## 7. Raise the resolution

Set `par.N = 160` in `bin/parameters.py` and repeat steps 4 to 6. Now

```
  ★          Re λ              Im λ          𝒯/𝒫       residσ     residω    Peak ℓ   ℓ-Width   ℓ-Convergence
  0      -0.007855595       1.227914636    1.12e+00   7.59e-08   2.45e-10     3         44        4.68e-12
```

residσ is $8\times10^{-8}$, and the eigenvalue moved by $5\times10^{-7}$ from the run at N = 132: the mode is resolved.
Its eigenvalue often converges at a lower N than its residual, so judge a run by the residuals. The other four
eigenvalues changed completely: at N = 132 they were not solutions of the equations.

## 8. Compare with the paper

Baruteau & Rieutord give $\lvert\omega\rvert = 1.2279$ and the damping rate $\tau = -7.86\times10^{-3}$, in units of
the surface rotation rate. kore gives $\lvert\omega\rvert = 1.22791$ and $\sigma = -7.856\times10^{-3}$. For $m = 0$ the
modes come in pairs $\lambda$ and $\lambda^*$, so the sign of the frequency carries no information.

## Running all steps at once

`runKore.sh`, at the top of the repository, runs steps 4 to 6 in one go and then moves the results, the matrices
and a copy of the parameter files to `../run`. Copy it to the run directory and set `ncpus` in its first lines to
`par.ncpus`:

```shell
cp ../kore/runKore.sh .
./runKore.sh
```

## Where to go next

- Change the problem: [Background model](background.md) explains the density, stratification, diffusivity and rotation
  options, and [Benchmarks](benchmarks.md) gives the parameters of the other published cases, including the Sun.
- A solution that fails the checks, or a run that runs out of memory: [Solver](solver.md).
- The equations kore solves and how it discretizes them: the pages under Mathematical framework.
