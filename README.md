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

Detailed instructions coming soon.

### Installing and running

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

