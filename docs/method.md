# Numerical method

This page gives an overview of how Kore turns the equations into matrices. It is not needed to run the code, but it helps when reading the source or the solution files.


## Unknowns

The flow and the magnetic field are written in poloidal-toroidal form and expanded in spherical harmonics $Y_l^m$. For one azimuthal wave number $m$,

$$
\mathbf{u} = \sum_l \nabla\times\nabla\times\left( P_l(r)\, Y_l^m\, \mathbf{r} \right) + \nabla\times\left( T_l(r)\, Y_l^m\, \mathbf{r} \right),
\qquad
\mathbf{b} = \sum_l \nabla\times\nabla\times\left( f_l(r)\, Y_l^m\, \mathbf{r} \right) + \nabla\times\left( g_l(r)\, Y_l^m\, \mathbf{r} \right),
$$

and the temperature and composition are expanded directly in $Y_l^m$. Because the problem is linear and axisymmetric, different $m$ do not couple, and the two equatorial symmetries separate: each one keeps every other degree $l$ in the poloidal part and the remaining ones in the toroidal part. So each of the poloidal and toroidal parts has $(l_{max} - m + 1)/2$ degrees.

In radius, each scalar is expanded in Chebyshev polynomials on $[r_i, 1]$, `N` of them with an inner core. In a full sphere (`ricb = 0`) each scalar has a definite parity in $r$, so only the even or the odd polynomials are used: `N/2` coefficients.

The unknown vector holds, in this order: the flow ($P$, then $T$), the magnetic field ($f$, then $g$), the 3 rotational unknowns ($\omega_M$, $\omega_{IC}$, $\varphi$) with `rotdyn = 1`, the inner-core field ($f_i$, then $g_i$, `N_cic/2` coefficients per degree) with a conducting inner core, the temperature and the composition. Within each part the coefficients are grouped by degree $l$. This is also the layout of the `.field` files.


## Sparse spectral method

Each equation is projected so that it involves only the scalars: the radial component of the curl of the curl of the momentum equation for $P$, the radial component of its curl for $T$, and the corresponding projections of the induction equation for $f$ and $g$. The equations are multiplied by a suitable power of $r$ so that all their coefficients are polynomials in $r$.

Kore then uses the ultraspherical spectral method (Olver & Townsend 2013). A derivative of order $k$ maps Chebyshev polynomials to Gegenbauer (ultraspherical) polynomials $C^{(k)}$ with a sparse matrix, multiplication by a polynomial is banded in the Gegenbauer basis, and conversions between bases are banded too. An equation of order $k$ is therefore represented by a sparse, banded matrix in the $C^{(k)}$ basis, instead of the dense matrix of a collocation method. The boundary conditions replace the first rows of each block (a tau method).

The work is split in three programs:

1. `submatrices.py` builds the radial operators (each combination of a power of $r$, a radial profile and a derivative) and writes them as `*.mtx` files.
2. `assemble.py` combines them into the matrices $A$ and $B$, block by block in $l$, adding the coupling between neighbouring degrees (Coriolis, Lorentz and induction terms), the boundary conditions and, if needed, the inner-core and rotational blocks. The degrees are distributed over the MPI processes in equal blocks, which is why `lmax - m + 1` must be divisible by `2*ncpus`.
3. `solve_nopp.py` solves $A\,x = \lambda B\,x$ by shift-and-invert around the target, with a sparse LU factorisation of $A - \tau B$ by MUMPS (see [Solver settings](solver.md)), or $A\,x = b$ for a forced problem.
