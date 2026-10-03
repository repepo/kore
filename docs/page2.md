# Installation Notes 

To run **`kore`** we need the PETSc/SLEPc packages and their python bindings petsc4py/slepc4py. PETSc needs to be compiled with support for complex scalars. MUMPS and SuperLU_dist are external packages that need to be installed together with PETSc.

If you are interested in problems that involve very large matrices, for instance when considering extremely small viscosities, we recommend doing so with a machine with at least 128 GB of memory. The number of processing cores is not critical, eight are fine, 24 are plenty. Small to moderate size matrices can be solved using a laptop, depending on available memory. 

From our experience the PETSc version that has allowed us to solve the largest problems is version 3.9.4. That version requires python 3.7, however. Newer PETSc versions seem to have much larger memory footprint, although they can still handle medium size problems without issues. So, our advice is to install PETSc version 3.9.3 if the problem involves very large matrices, otherwise it is better to stick to the most recent PETSc release.

## Installing PETSc/SLEPc on MacOS

It is convenient to have a dedicated python environment to use with **`kore`**. It is quite simple to create and activate it:

```Shell
python3 -m venv kore_env
source kore_env/bin/activate
```

Then we need to install scipy and cython in that environment:
```Shell
pip3 install scipy cython
```

Now we download the latest PETSc/SLEPc releases using `git`
```Shell
git clone -b release https://gitlab.com/petsc/petsc.git petsc
git clone -b release https://gitlab.com/slepc/slepc slepc
```

We go now into the newly created `petsc` folder and configure PETSc:
```Shell
cd petsc
./configure --with-petsc4py --download-mpi4py --download-mpich --with-scalar-type=complex --download-mumps --download-parmetis --download-metis --download-scalapack --download-openblas --download-openblas-make-options='USE_THREAD=0 USE_OPENMP=0 USE_LOCKING=1' --with-debugging=0 --download-superlu_dist --download-ptscotch CXXOPTFLAGS='-O3 -march=native' FOPTFLAGS='-O3 -march=native' COPTFLAGS='-O3 -march=native' --download-bison
```
We recommend OpenBLAS (`--download-openblas`) rather than the reference BLAS/LAPACK (`--download-fblaslapack`). MUMPS spends most of its factorization time in dense BLAS kernels, and with OpenBLAS the LU factorization was 5 to 8 times faster in our tests (Linux, 24-core Threadripper, MUMPS 5.8.2). For example, a torsional-mode run at N = 976 took 4.4 minutes instead of 20. The make options build a single-threaded OpenBLAS, so that it doesn't compete with the MPI processes for cores.

Then we build PETSc:
```Shell
make PETSC_DIR=/path/to/petsc PETSC_ARCH=arch-darwin-c-opt all
```
where you must replace `/path/to/petsc` with the actual path for your case. Before checking that everything works we must tell python where to find `petsc4py` and `mpi4py`, which were compiled along in the step above:
```Shell
export PYTHONPATH=/path/to/petsc/arch-darwin-c-opt/lib
```
where again we must replace `/path/to/petsc` with the actual path. Now we test the installation:
```Shell
make PETSC_DIR=/path/to/petsc PETSC_ARCH=arch-darwin-c-opt check
```
If no errors appear then you can proceed to install SLEPc. If an error is reported due to python not being able to find PETSc (even though we just updated the `PYTHONPATH`), but the other MPI tests were successful, then don't worry, it is safe to ignore the error.

We need to setup some environment variables before installing SLEPc:
```Shell
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=arch-darwin-c-opt
export SLEPC_DIR=/path/to/slepc
```

Now go to the slepc folder, configure and build SLEPc:
```Shell
cd $SLEPC_DIR
./configure
make
make check
```
The last step is to install slepc4py, which is distributed along with SLEPc:
```Shell
cd $SLEPC_DIR/src/binding/slepc4py
python3 setup.py build
python3 setup.py install
```
Almost there now. We need to update the `PATH` environment variable so that we can use the MPI library provided by PETSc instead of the one provided by your system, if any:
```Shell
export PATH=$PETSC_DIR/$PETSC_ARCH/bin:$PATH
```

It is a good idea to keep the necessary commands for initialization in a separate file. It should contain the following lines:
```Shell
source /path/to/kore_env/bin/activate
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=arch-darwin-c-opt
export SLEPC_DIR=/path/to/slepc
export PYTHONPATH=$PETSC_DIR/$PETSC_ARCH/lib
export PATH=$PETSC_DIR/$PETSC_ARCH/bin:$PATH
```
We can write that to a file named e.g. `kore_env.sh` in your home directory, so every time you need to prepare to run **`kore`** we do first:
```Shell
source $HOME/kore_env.sh
```


## Installing PETSc/SLEPc on Linux

The steps are the same as for MacOS, except for the name of the PETSc architecture (`arch-linux-c-opt`) and the way slepc4py is built. These instructions were tested on Debian 13 with gcc/gfortran 14 and PETSc/SLEPc 3.25.

First we need compilers and a few build tools. On Debian or Ubuntu they can be installed with:
```Shell
sudo apt install build-essential gfortran git cmake python3-dev python3-venv
```

Then we create and activate a dedicated python environment for **`kore`**, and install scipy and cython in it:
```Shell
python3 -m venv kore_env
source kore_env/bin/activate
pip3 install scipy cython
```

Now we download the latest PETSc/SLEPc releases using `git`
```Shell
git clone -b release https://gitlab.com/petsc/petsc.git petsc
git clone -b release https://gitlab.com/slepc/slepc slepc
```

We go now into the newly created `petsc` folder and configure PETSc, using OpenBLAS as explained in the MacOS section:
```Shell
cd petsc
./configure --with-petsc4py --download-mpi4py --download-mpich --with-scalar-type=complex --download-mumps --download-parmetis --download-metis --download-scalapack --download-openblas --download-openblas-make-options='USE_THREAD=0 USE_OPENMP=0 USE_LOCKING=1' --with-debugging=0 --download-superlu_dist --download-ptscotch CXXOPTFLAGS='-O3 -march=native' FOPTFLAGS='-O3 -march=native' COPTFLAGS='-O3 -march=native' --download-bison
```
Then we build PETSc:
```Shell
make PETSC_DIR=/path/to/petsc PETSC_ARCH=arch-linux-c-opt all
```
where you must replace `/path/to/petsc` with the actual path for your case. Before checking that everything works we must tell python where to find `petsc4py` and `mpi4py`, which were compiled along in the step above:
```Shell
export PYTHONPATH=/path/to/petsc/arch-linux-c-opt/lib
```
Now we test the installation:
```Shell
make PETSC_DIR=/path/to/petsc PETSC_ARCH=arch-linux-c-opt check
```

We need to setup some environment variables before installing SLEPc:
```Shell
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=arch-linux-c-opt
export SLEPC_DIR=/path/to/slepc
export PATH=$PETSC_DIR/$PETSC_ARCH/bin:$PATH
```

Now go to the slepc folder, configure and build SLEPc. The `--with-slepc4py` option builds slepc4py along with SLEPc, inside `$SLEPC_DIR/$PETSC_ARCH/lib`:
```Shell
cd $SLEPC_DIR
./configure --with-slepc4py
make
make check
```

As for MacOS, it is a good idea to keep the necessary commands for initialization in a separate file. Note that `PYTHONPATH` must now include the SLEPc folder too, so that python finds slepc4py:
```Shell
source /path/to/kore_env/bin/activate
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=arch-linux-c-opt
export SLEPC_DIR=/path/to/slepc
export PYTHONPATH=$PETSC_DIR/$PETSC_ARCH/lib:$SLEPC_DIR/$PETSC_ARCH/lib
export PATH=$PETSC_DIR/$PETSC_ARCH/bin:$PATH
```
We can write that to e.g. `$HOME/kore_env.sh` and do `source $HOME/kore_env.sh` before running **`kore`**.

Several PETSc builds can live side by side in the same `petsc` and `slepc` folders, each under its own `PETSC_ARCH` name (passed as `PETSC_ARCH=...` to PETSc's `./configure`). Switching between them only requires changing `PETSC_ARCH`.


## SHTns
Coming soon
