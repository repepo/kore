# Software implementation

Although the [physical problem](https://yorchmcm.github.io/kore_sandbox/2definition/) and [numerical method](https://yorchmcm.github.io/kore_sandbox/4numerical/) have been explained, using the code can - and will - be disorienting. Thus, an overall explanation of how the code is structured is presented below. This will cover most - **but not all** - of the files within the [`bin` folder](https://github.com/repepo/kore/tree/main/bin). The content below will be high level, aimed at relating the different parts of the code to what has been explained in the rest of this documentation. Low level details pertaining to e.g. the specific way in which matrices are built will not be given. 

## Running the software

It is to be noted that some pre-operatorions are required to run Kore, namely exporting all required paths and activating the environment (see [Installation](https://yorchmcm.github.io/kore_sandbox/1installation/) and the README file in the [Kore Github repo](https://github.com/repepo/kore)). All this will be assumed to have been done. Another important aspect of Kore is that it is parallelized. Parallelization-related commands and code will therefore be intertwined with Kore's code per se. Parallelization details will not be discussed here, though.

## General workflow

### Defining the problem

The first step to solve a problem with Kore is, well, to define a problem. This is done in the `parameters.py` file. It is here where the equations that need to be solved are selected, among those explained in [Physical definition](https://yorchmcm.github.io/kore_sandbox/2definition/). This file is generally divided in big sections pertaining to each of the equations, where the relevant non-dimensional quantities are also defined. For instance, the section devoted to the momentum equation contains the value of the Ekmann number, while the section devoted to induction contains the value of the Lenhert and magnetic Ekmann numbers. After all sections, the value of $\Omega\tau$ is selected (see [Non-dimensionalization](https://yorchmcm.github.io/kore_sandbox/3nondim/)), which sets the non-dimensionalization and the final form of the equations.

Another thing that is done in this file, under a section called `Resolution`, is fixing the truncation order for the Chebyshev expansions and the truncation degree for the spherical harmonic expansions. The former can be done either manually or according to the Ekmann number, while the latter is determined according to the number of CPUs used in the parallelization. In this regard, it was mentioned in [Numerical scheme](https://yorchmcm.github.io/kore_sandbox/4numerical/) that Kore solves the problem in a *per-order* basis, and only for a given symmetry of the solution. These two elements are also set in this file, under the `hydrodynamic parameters` section.

The last section in this file is devoted to SLEPc, the algorithm to solve for the eigenvalues of the problem. Here, the ''fixed point'' mentioned in [Numerical scheme](https://yorchmcm.github.io/kore_sandbox/4numerical/) is set, as well as the maximum number of iterations or the required tolerance in the obtained eigenvalues.

### Building the submatrices

The next step is just building the system of equations. It was explained in [Numerical scheme](https://yorchmcm.github.io/kore_sandbox/4numerical/) that the equations in this system - which results from the angular discretization - will be determined by the order $m$ and symmetry of the solution and that, when written in matrix form, each entry of the matrix will be a differential operator. In the radial discretization, however, each of these operators is turned into a sub-matrix itself, all coming from the Chebyshev discretization scheme (see [Numerical scheme](https://yorchmcm.github.io/kore_sandbox/4numerical/) or [Olver and Townsend, 2013](https://epubs.siam.org/doi/abs/10.1137/120865458)). Thus, in order to build the whole system, these submatrices are built first. This is done by running the `submatrices.py` file. Running this file requires one argument: the number of cores. Once the submatrices are built they are written to txt-like files with the `.mtx` extension. These files are stored one directory above the `submatrices.py` file itself.

### Assembling the system

The next step is assembling these submatrices into the bigger system, namely into matrices $\mathbf A$ and $\mathbf B$ of the generalized eigenvalue problem. This is done in the `assemble.py`, which should be run next. This task is parallelized with [MPI](https://mpitutorial.com/), and the command that should be run from the command line is

```console
mpiexec -n [NUMBER OF CORES] [PATH]/assemble.py
```

If all goes well, the matrices should be written in the `A.mtx` and `B.mtx` files together with all other submatrices.

### Solving the system

Once these two matrices are built, all that is left is to solve the system. Some options for the SLEPc solver are required first, so they should be specified with the command

```console
export opts='-st_type sinvert -eps_error_relative ::ascii_info_detail -eps_balance twoside -pc_factor_mat_solver_type mumps -mat_mumps_icntl_14 10000'

```

TODO: No estoy muy seguro de qué significa cada una de estas opciones. Solo he corrido este commando una vez bajo ordern directa y expresa de Andrés. Yo en realidad solo soy un mandao.

With these options, the command to run the `solve.py` can be run, which is also done in paraller with MPI:

```console
mpiexec -n [NUMBER OF CORES] [PATH]/solve.py $opts

```

If all goes well, the solution is written into txt files. If solved as an eigenvalue problem, the eigenvalues are written to an `eigenvalues0.dat` file. The eigenvectors are written to files with the `.field` extension. In a solved problem, these files do not contain the eigenvectors but rather the response vectors. These vectors are just segments of the massive vector containing all Chebyshev coefficients of all spherical harmonic coefficients of all scalar variables. The part pertaining to the hydrodynamic equations contains both the $\mathcal P$ and $\mathcal T$ potentials, while the part pertaining to the induced magnetic field contains both the $\mathcal F$ and $\mathcal G$ potentials. TODO: Eso último me lo he inventado jajajaja

### Post processing the solution

In an eigenvalue problem, the quality of the converged solution obtained with SLEPc can be checked by running the `spin_doctor.py` file, which provides TODO: Ni idea de qué provides.

On the other hand, the eigenvectors - or forced solution - can be visualized by running the `plot_field.py` file.
