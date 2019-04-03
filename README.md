# tintin

Numerical code to solve the linear Navier-Stokes and induction equations for a viscous, incompressible and conductive fluid enclosed within a rotating spherical shell. 

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

Download release 3.10.5. We use this version and not the latest because we need a matching petsc4py, which at the time of this writing is only version 3.10.1, therefore we are restricted to PETSc's 3.10 series. Unpack and cd to the installation directory:
```
wget http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-3.10.5.tar.gz
tar xvf petsc-lite-3.10.5.tar.gz
cd petsc-3.10.5
```
We need PETSc built with support for complex scalars. We need also the external packages `mumps` and `superlu_dist`.
Therefore the configure command should include the options:
```
--with-scalar-type=complex --download-mumps=1 --download-superlu_dist=1
```
Additional options might be needed according to your specific system, please consult the PETSc installation documentation [here](https://www.mcs.anl.gov/petsc/documentation/installation.html). PETSc requires a working MPI installation, either `mpich` or `openmpi`. In our own experience, it saves a lot of headache if we include `mpich` as an external package to be installed along with PETSc. Therefore we include the option `--download-mpich=1`
Just to provide an example, the configure command needed in our own computing cluster is (get yourself some coffee, this step takes several minutes to complete):
```
./configure --download-mpich --with-scalar-type=complex --download-mumps=1 --download-parmetis --download-metis --download-scalapack=1 --download-fblaslapack=1 --with-debugging=0 --download-superlu_dist=1 --download-ptscotch=1 CXXOPTFLAGS='-O3 -march=native' FOPTFLAGS='-O3 -march=native' COPTFLAGS='-O3 -march=native' --with-cxx-dialect=C++11
```
If everything goes well then you can build the libraries (modify `/path/to` as needed):
```
make PETSC_DIR=/path/to/petsc-3.10.5 PETSC_ARCH=arch-linux2-c-opt all
```
then test the libraries:
```
make PETSC_DIR=/path/to/petsc-3.10.5 PETSC_ARCH=arch-linux2-c-opt check
```
The MPI executables are now installed under `/path/to/petsc-3.10.5/arch-linux2-c-opt/bin/` so we need to prepend that directory to the `$PATH` variable. A  good place to do that could be in your `.profile`. Include the following lines:
```
export PETSC_DIR=/path/to/petsc-3.10.5
export PETSC_ARCH=arch-linux2-c-opt
export PATH=$PETSC_DIR/arch-linux2-c-opt/bin/:$HOME/.local/bin/:$PATH
```
PETSc and MPI are now ready!

#### Installing SLEPc
Download release 3.10.2. Unpack and cd to the installation directory:
```
http://slepc.upv.es/download/distrib/slepc-3.10.2.tar.gz
tar xvf slepc-3.10.2.tar.gz
cd slepc-3.10.2
```
Make sure the environment variables `PETSC_DIR` and `PETSC_ARCH` are exported already: if you modified your `.profile` as suggested above then simply do
```
source .profile
``` 
Then configure, build and test SLEPc (modify `/path/to` as needed):
```
./configure
make SLEPC_DIR=/path/to/slepc-3.10.2 PETSC_DIR=/path/to/petsc-3.10.5 PETSC_ARCH=arch-linux2-c-opt
make SLEPC_DIR=/path/to/slepc-3.10.2 PETSC_DIR=/path/to/petsc-3.10.5 check
```
Finally, export the variable `SLEPC_DIR`. (Adding this line to your `.profile` is a good idea)
```
export SLEPC_DIR=/path/to/slepc-3.10.2
```
SLEPc is now ready.

#### Installing petsc4py, slepc4py and mpi4py
Get the petsc4py tarball and unpack:
```
wget https://bitbucket.org/petsc/petsc4py/downloads/petsc4py-3.10.1.tar.gz
tar xvf petsc4py-3.10.1.tar.gz
```
Then build and install to python3:
```
cd petsc4py-3.10.1
python3 setup.py build
python3 setup.py install --user
```
Follow a completely analogous procedure for slepc4py and mpi4py.

#### Installing wigxjpf
This is a library to compute Wigner-3j and 6j symbols. Download and unpack:
```
wget http://fy.chalmers.se/subatom/wigxjpf/wigxjpf-1.9.tar.gz
tar xvf wigxjpf-1.9.tar.gz
```
Then build and install:
```
cd wigxjpf-1.9
make
python3 pywigxjpf/setup.py install --user
```


### Installing and running `tintin`

Simply download and unzip the tar file under the downloads section.

```sh
wget https://bitbucket.org/repepo/tintin/downloads/tintin-0.1.tar.gz
tar xvf tintin-0.1.tar.gz
```
For regular work, make a copy of the source directory, keeping the original source clean. For example:

```sh
cp -r tintin-0.1 tintin_work1
cd tintin_work1
```

Modify the `parameters.py` file as desired.

Then generate the submatrices:
```sh
./submatrices.py ncpus
```
where `ncpus` is the number of cpu's (cores) in your system.

To assemble the main matrices do:
```sh
mpiexec -n ncpus ./assemble.py
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
mpiexec -n ncpus ./solve.py $opts
```

The result is written/appended to the file `flow.dat`, and the parameters used are written/appended to the file `params.dat`, one line for each solution. If solving an eigenvalue problem, the eigenvalues are written/appended to the file `eigenvalues.dat`. If solving with magnetic fields, an additional file `magnetic.dat` is created/appended. 

We include a set of scripts:
```
dodirs.sh
dosubs.sh
reap.sh
getresults.py
```
to aid in submitting/collecting results of a large number of runs to/from a PBS-managed cluster. The code itself can run however even on a single cpu machine (albeit slowly). 

## Authors

* **Santiago Andres Triana** - *Python implementation*
* **Jeremy Rekier** - *Sparse spectral method*
* **Antony Trinh** - *Tensor calculus*

## License

GPLv3

