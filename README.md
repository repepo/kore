# Kore
[![Tests](https://github.com/repepo/kore/actions/workflows/main.yml/badge.svg?branch=conductive_IC)](https://github.com/repepo/kore/actions/workflows/main.yml)  [![Docs](https://img.shields.io/badge/docs-online-blue)](https://repepo.github.io/kore/)  [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

*KOR-ee*, from the greek **Κόρη**, the queen of the underworld, daughter of Zeus and Demeter. Kore is a numerical tool to study the core flow within rapidly rotating planets or other rotating fluids contained within near-spherical boundaries. The current version solves the *linear* Navier-Stokes equation and optionally the induction and thermal/compositional (Boussinesq) equations for a viscous, incompressible and conductive fluid with an externally imposed magnetic field, and enclosed within a near-spherical boundary. A solid inner core can be included optionally.

Kore assumes all dynamical variables to oscillate in a harmonic fashion. The oscillation frequency can be imposed externally, as is the case when doing forced motion studies (e.g. *tidal* forcing), or it can be obtained as part of the solution to an eigenvalue problem. In Kore's current implementation the eigenmodes comprise the inertial mode set (resoring force is Coriolis), the gravity modes or g mode set (restoring force is buoyancy), the torsional Alfvén mode set (restoring force is the Lorentz force), and the magneto-Archimedes-Coriolis (MAC) set (combination of Coriolis, Lorentz, and buoyancy as restoring forces).

Kore's distinctive feature is the use of a very efficient spectral method employing Gegenbauer (also known as ultraspherical) polynomials as a basis in the radial direction. This approach leads to *sparse* matrices representing the differential equations, as opposed to *dense* matrices, as in traditional Chebyshev colocation methods. Sparse matrices have smaller memory-footprint and are more suitable for systematic core flow studies at extremely low viscosities (or small Ekman numbers).     

Kore is free for everyone to use under the GPL v3 license. Too often in the scientific literature the numerical methods used are presented with enough detail to guarantee reproducibility, but only *in principle*. Without access to the actual implementation of those methods, which would require a significant amount of work and time to develop, readers are left effectively without the possibility to reproduce or verify the results presented. This leads to very slow scientific progress. We share our code to avoid this.

If this code is useful for your research, we invite you to cite the relevant papers (coming soon) and hope that you can also contribute to the project. 

## Getting Started

### Prerequisites

* python3 with numpy and scipy
* [PETSc](https://petsc.org/) with complex scalars, MUMPS and SuperLU_DIST, plus petsc4py and mpi4py
* [SLEPc](https://slepc.upv.es/) and slepc4py

Step-by-step installation instructions for MacOS and Linux are in [the online documentation](https://repepo.github.io/kore/page2/).


### Installing and running `Kore`
Clone the repository with
```sh
git clone https://github.com/repepo/kore.git
```
For regular work, make a copy of the source directory, keeping the original source clean. For example:
```sh
cp -r kore kwork1
cd kwork1
```

Modify `bin/parameters.py` as desired. All commands below are run from the top folder (`kwork1` here), where the matrices and results are written.

First generate the submatrices:
```sh
./bin/submatrices.py ncpus
```
where `ncpus` is the number of cpu's (cores) in your system.

To assemble the main matrices do:
```sh
mpiexec -n ncpus ./bin/assemble.py
```

To solve the problem do:
```sh
mpiexec -n ncpus ./bin/solve_nopp.py
```
The PETSc/SLEPc/MUMPS solver options are set in the `petsc_opts` dictionary at the end of `bin/parameters.py`, which has separate groups for eigenvalue and forced problems. Options given on the command line, e.g. `mpiexec -n ncpus ./bin/solve_nopp.py -st_mat_mumps_icntl_14 50`, take precedence over the ones in `parameters.py`.

The solutions are written to disk (`real_*.field` and `imag_*.field` files and, for eigenvalue problems, `eigenvalues0.dat`). To postprocess them do:
```sh
./bin/spin_doctor.py ncpus
```
This prints a summary table with energy-balance residuals for each solution, and writes/appends the results to the file `flow.dat`, and the parameters used to the file `params.dat`, one line for each solution. If solving an eigenvalue problem, the eigenvalues are written/appended to the file `eigenvalues.dat`. If solving with magnetic fields, an additional file `magnetic.dat` is created/appended, and similarly `thermal.dat`, `compositional.dat` and `rotdyn.dat` when those are included.

We include a set of scripts in the `tools` folder:
```
dodirs.sh
dosubs.sh
reap.sh
getresults.py
```
to aid in submitting/collecting results of a large number of runs to/from a PBS-managed cluster. The code itself can run however even on a single cpu machine (albeit slowly). 

## Authors

* **Santiago Andres Triana** - *This implementation*
* **Jeremy Rekier** - *Sparse spectral method*
* **Antony Trinh** - *Tensor calculus*
* **Ankit Barik** - *Convection branch, visualization*
* **Fleur Seuren** - *Buoyancy*

## License

GPLv3

