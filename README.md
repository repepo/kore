# Kore
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

*KOR-ee*, from the greek **Κόρη**, the queen of the underworld, daughter of Zeus and Demeter. Kore is a numerical tool to study the core flow within rapidly rotating planets or other rotating fluids contained within near-spherical boundaries. The current version solves the *linear* Navier-Stokes equation and optionally the induction and thermal/compositional equations for a viscous, incompressible and conductive fluid with an externally imposed magnetic field, and enclosed within a rotating spherical shell.

Kore assumes all dynamical variables to oscillate in a harmonic fashion. The oscillation frequency can be imposed externally, as is the case when doing forced motion studies (e.g. *tidal* forcing), or it can be obtained as part of the solution to an eigenvalue problem. In Kore's current implementation the eigenmodes comprise the inertial mode set (resoring force is Coriolis), the gravity modes or g mode set (restoring force is buoyancy), the torsional Alfvén mode set (restoring force is the Lorentz force), and the magneto-Archimedes-Coriolis (MAC) set (combination of Coriolis, Lorentz, and buoyancy as restoring forces).

Kore's distinctive feature is the use of a very efficient spectral method employing Gegenbauer (also known as ultraspherical) polynomials as a basis in the radial direction. This approach leads to *sparse* matrices representing the differential equations, as opposed to *dense* matrices, as in traditional Chebyshev colocation methods. Sparse matrices have smaller memory-footprint and are more suitable for systematic core flow studies at extremely low viscosities (or small Ekman numbers).     

Kore is free for everyone to use under the GPL v3 license. Too often in the scientific literature the numerical methods used are presented with enough detail to guarantee reproducibility, but only *in principle*. Without access to the actual implementation of those methods, which would require a significant amount of work and time to develop, readers are left effectively without the possibility to reproduce or verify the results presented. This leads to very slow scientific progress. We share our code to avoid this.

If this code is useful for your research, we invite you to cite the relevant papers (coming soon) and hope that you can also contribute to the project. 

## Getting Started

### Prerequisites

* python3
* [PETSc](https://www.mcs.anl.gov/petsc/) with complex scalars, mumps and superlu_dist.
* [SLEPc](http://slepc.upv.es/)
* [petsc4py](https://bitbucket.org/petsc/petsc4py/src/master/)
* [slepc4py](https://bitbucket.org/slepc/slepc4py/src/master/)
* [mpi4py](https://bitbucket.org/mpi4py/mpi4py/src/master/)
* [wigxjpf](http://fy.chalmers.se/subatom/wigxjpf/)

#### Installing PETSc

Download PETSc release 3.12.5. This release supports SuperLU_DIST version 5.4.0, which has much lower memory footprint than the newest version. We need to download SuperLU_DIST (no need to unpack it) and then download and unpack PETSc:
```
wget https://portal.nersc.gov/project/sparse/superlu/superlu_dist_5.4.0.tar.gz
wget http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-3.12.5.tar.gz
tar xvf petsc-lite-3.12.5.tar.gz
cd petsc-3.12.5
```
We need PETSc built with support for complex scalars. We need also the external packages `superlu_dist` (which we just downloaded) and `mumps`.
Therefore the configure command should include the options:
```
--with-scalar-type=complex --download-mumps=1 --download-superlu_dist=../superlu_dist_5.4.0.tar.gz
```
Additional options might be needed according to your specific system, please consult the PETSc installation documentation [here](https://www.mcs.anl.gov/petsc/documentation/installation.html). PETSc requires a working MPI installation, either `mpich` or `openmpi`. In our own experience, it saves a lot of headache if we include `mpich` as an external package to be installed along with PETSc. Therefore we include the option `--download-mpich=1`
Just to provide an example, the configure command needed in our own computing cluster is (get yourself some coffee, this step takes several minutes to complete):
```
./configure --download-mpich --with-scalar-type=complex --download-mumps=1 --download-parmetis --download-metis --download-scalapack=1 --download-fblaslapack=1 --with-debugging=0 --download-superlu_dist=../superlu_dist_5.4.0.tar.gz --download-ptscotch=1 CXXOPTFLAGS='-O3 -march=native' FOPTFLAGS='-O3 -march=native' COPTFLAGS='-O3 -march=native' --with-cxx-dialect=C++11
```
If everything goes well then you can build the libraries (modify `/path/to` as needed):
```
make PETSC_DIR=/path/to/petsc-3.12.5 PETSC_ARCH=slu540
```
then test the libraries:
```
make PETSC_DIR=/path/to/petsc-3.12.5 PETSC_ARCH=slu540 check
```
The MPI executables are now installed under `/path/to/petsc-3.12.5/slu540/bin/` so we need to prepend that directory to the `$PATH` variable. A  good place to do that could be in your `.profile`. Include the following lines:
```
export PETSC_DIR=/path/to/petsc-3.12.5
export PETSC_ARCH=slu540
export PATH=$PETSC_DIR/$PETSC_ARCH/bin:$PATH
```
PETSc and MPI are now ready!

#### Installing SLEPc
Download release 3.12.2. Unpack and cd to the installation directory:
```
cd
wget http://slepc.upv.es/download/distrib/slepc-3.12.2.tar.gz
tar xvf slepc-3.12.2.tar.gz
cd slepc-3.12.2
```
Make sure the environment variables `PETSC_DIR` and `PETSC_ARCH` are exported already: if you modified your `.profile` as suggested above then simply do
```
source ~/.profile
``` 
Then configure, build and test SLEPc (modify `/path/to` as needed):
```
./configure
make SLEPC_DIR=/path/to/slepc-3.12.2 PETSC_DIR=/path/to/petsc-3.12.5 PETSC_ARCH=slu540
make SLEPC_DIR=/path/to/slepc-3.12.2 PETSC_DIR=/path/to/petsc-3.12.5 check
```
Finally, export the variable `SLEPC_DIR`. (Adding this line to your `.profile` is a good idea)
```
export SLEPC_DIR=/path/to/slepc-3.12.2
```
SLEPc is now ready.

#### Installing petsc4py, slepc4py and mpi4py
Get the petsc4py tarball and unpack:
```
cd
wget https://bitbucket.org/petsc/petsc4py/downloads/petsc4py-3.12.0.tar.gz
tar xvf petsc4py-3.12.0.tar.gz
```
Then build and install to python3:
```
cd petsc4py-3.12.0
python3 setup.py build
python3 setup.py install --user
```
Follow a completely analogous procedure for slepc4py and mpi4py. Download the tarballs with:
```
cd
wget https://bitbucket.org/slepc/slepc4py/downloads/slepc4py-3.12.0.tar.gz
wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.0.3.tar.gz
```

#### Installing wigxjpf
This is a library to compute Wigner-3j and 6j symbols, useful to compute the products of spherical harmonics. Download and unpack:
```
cd
wget http://fy.chalmers.se/subatom/wigxjpf/wigxjpf-1.11.tar.gz
tar xvf wigxjpf-1.11.tar.gz
```
Then build and install:
```
cd wigxjpf-1.11
make
python3 setup.py install --user
```


### Installing and running `Kore`
Clone the repository with
```
git clone https://bitbucket.org/repepo/kore.git
```
Or download and unzip the tar file under the downloads section.
```sh
cd
wget https://bitbucket.org/repepo/Kore/downloads/kore-0.2.tar.gz
tar xvf kore-0.2.tar.gz
```
For regular work, make a copy of the source directory, keeping the original source clean. For example:

```sh
cp -r kore-0.2 kwork1
cd kwork1
```

Modify the `parameters.py` under `kwork1/bin/` file as desired.

Then generate the submatrices:
```sh
./bin/submatrices.py ncpus
```
where `ncpus` is the number of cpu's (cores) in your system.

To assemble the main matrices do:
```sh
mpiexec -n ncpus ./bin/assemble.py
```

If you are solving an *eigenvalue* problem, do the following export:
```sh
export opts="-st_type sinvert -eps_error_relative ::ascii_info_detail"
```

If you are solving a *forced* problem then do:
```sh
export opts='-ksp_type preonly -pc_type lu'
```
Others options might be required depending on the size of the matrices.


To solve the problem do
```sh
mpiexec -n ncpus ./bin/solve.py $opts
```

The result is written/appended to the file `flow.dat`, and the parameters used are written/appended to the file `params.dat`, one line for each solution. If solving an eigenvalue problem, the eigenvalues are written/appended to the file `eigenvalues.dat`. If solving with magnetic fields, an additional file `magnetic.dat` is created/appended. 

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

