# Numerical scheme

In order to solve [the non-dimensional problem](https://yorchmcm.github.io/kore_sandbox/3nondim/), the implementation of Kore is quite the opposite of straightforward, but the methods are traditional. Below, the translation between the non-dimensional equations and the [software implementation](https://yorchmcm.github.io/kore_sandbox/5implementation/) will be presented. An attempt has been made at dividing this section and its steps in a way in which the relationship with the different parts of Kore are as clear as possible. The momentum equation without the Lorentz force and the rotation time scale is used to illustrate the procedure, which is extendable to all other equations and terms.

In short, Kore writes $\mathbf u$ and $\mathbf b$ in their poloidal and toroidal components. Discretization of the problem is made in terms of frequency, with spherical harmonics serving for angular discretization and Chebyshev polynomials serving for radial discretization. The differential system in the unknowns then turns into a large algebraic linear system in their coefficients, which is solved as an eigenvalue problem if a forcing is not specified (or solved directly otherwise).

## Poloidal-toroidal decomposition and the $u$ and $v$ sections

Possibly the best place to start could be the way in which the vector equations are *dissected*. This decomposition is not motivated by software, but rather has been a traditional manner in which to study rotating flows (see e.g. [Tilgner 1999](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.59.1789)). It should be noted that, although it hasn't been explicitely mentioned in [Physical Definition](https://yorchmcm.github.io/kore_sandbox/2definition/), a (quasi-)incompressible fluid can only produce a divergence-less velocity field $\mathbf u$ by virtue of the continuity equation. Similarly, [Gauss' law for a magnetic field](https://en.wikipedia.org/wiki/Gauss%27s_law_for_magnetism) imposes a divergence-less induced magnetic field $\mathbf b$. These conditions impose further constraints on the relationship between the components of each vector field, reducing the number of actual free parameters from three per field to just two.

Rather than using the actual components of the (e.g. velocity) field, it has been a traditional approach to express the vector by mean of two scalar quantites, $\mathcal P$ and $\mathcal T$, called the *poloidal* and *toroidal* potentials respectively, as follows:

$$
\mathbf u = \nabla\times\nabla\times(\mathcal P\mathbf r) + \nabla\times(\mathcal T\mathbf r)
$$

The magnetic field $\mathbf b$ is written in a similar way, but the poloidal and toroidal potentials are denoted $\mathcal F$ and $\mathcal G$ respectively. Note that, because $\mathbf u$ is given as the *curls* of vectors, the condition of zero divergence is automatically satisfied. However, because we now have two unknowns rather than three, the original momentum (or induction) equation cannot be used. Instead, the $u$ and $v$ sections are used. The $u$ section is the radial projection of the second curl of the momentum equation ($\hat{\mathbf r}\cdot\nabla\times\nabla\times$), while the $v$ section is the radial projection of the first curl of the momentum equation ($\hat{\mathbf r}\cdot\nabla\times$). For simplicity reasons, however, the projection will be performed here by dot-multiplication with $\mathbf r$ rather than $\hat{\mathbf r}$. The same thing is done for the induction equation, where Kore now calls the sections $f$ and $g$ rather than $u$ and $v$ respectively. Each section is now one scalar equation, and the two sections together allow to solve for the two scalar unknowns.

Finally, it will be explicitely noted that the poloidal and toroidal potentials are time-independent complex functions, by virtue of $\mathbf u$ (or $\mathbf b$) having these same properties.

## Angular discretization, operators and symmetry

The introduction of the poloidal and toroidal potentials leaves us with a set of *explicitely* scalar equations and unknowns. Each of this unknowns is a complex number with a purely spatial dependance, which is advantegously expressed in terms of the radial $r$, colatitude $\theta$ and longitude $\varphi$ spherical coordinates given the (quasi-)spherical symmetry of the problem at hand. It is thus reasonable to expand the unknowns in [spherical harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics) $Y_l^m(\theta,\varphi) = e^{i\varphi}P_l^m(\cos\theta)$, with coefficients e.g. $\mathcal P_{l,m}$ that only depend on $r$:

$$
\mathcal P = \sum_{l=0}^{L}\sum_{m = -l}^lP_{l,m}(r)Y_l^m(\theta,\varphi)
$$

To be rigurous, the equal sign can only hold if the upper bound of the summation over $l$ is infinity, rather that some maximum degree $L$. However, it is assumed that the maximum degree is chosen such that the truncated series is a good enough approximation to the infinite one so that one can make no distinction between the two. In this manner, one unknown function dependent upon all three coordinates is transformed into $(L+1)^2$ unknowns dependent upon $r$. The velocity (or induced magnetic field) thus involves twice as many coefficients to account for both the poloida and toroidal potentials, each involving $(L+1)^2$ coefficients.

This greatly simplifies the expressions for all the terms in the $u$ and $v$ sections. These sections can be built term by term, in what Kore calls *operators*. Consider, for example, the first term in the momentum equation, $\lambda\mathbf u$ (see [Non-dimensionalization](https://yorchmcm.github.io/kore_sandbox/3nondim/)). When writing the $v$ section, this term will turn into $\mathbf r\cdot\nabla\times\mathbf u$. If one does the (very extensive) math, this is simply $-r^2\nabla^2_s\mathcal P$, where $\nabla^2_s$ represents the surface laplacian, i.e. the standard laplacian with all radial derivatives removed. By virtue of the properties of the spherical harmonics, such a quantiy can be written as:

$$
\mathbf r\cdot\nabla\times\mathbf u = -r^2\nabla^2_s\mathcal P = \sum_{l=0}^{L}\sum_{m = -l}^ll(l+1)\mathcal P_{l,m}(r)Y_l^m(\theta,\varphi)
$$

To simplify the notation, one may write just the $l,m$ coefficients of that series as below:

$$
(\mathbf r\cdot\nabla\times\mathbf u)_{l,m} = l(l+1)\mathcal P_{l,m}
$$

Kore calls this the *operator* $u$ in the $v$ section. A less trivial example - but a more informative one for later sections - is the *viscous diffusion* operator in the $u$ section, i.e. the projected second curl of the viscous force:

$$
(\mathbf r\cdot\nabla\times\nabla\times\nabla^2\mathbf u)_{l,m} = \frac{d^4\mathcal P_{l,m}}{dr^4} + \frac{4}{r}\frac{d^3\mathcal P_{l,m}}{dr^3} - \frac{2l(l+1)}{r^2}\frac{d^2\mathcal P_{l,m}}{dr^2} + \frac{l(l+1)(l^2+l-2)}{r^4}\mathcal P_{l,m}
$$

Such a process is followed for all the terms in the equation, through convenient use of the properties of spherical harmonics and their derivatives. Some need more algebra than others to be written in a useful manner. The Coriolis term results in a coupling between the poloidal and toroidal coefficients $\mathcal P_{l,m}$ and $\mathcal T_{l,m}$, so that each section $u$ and $v$ will *in principle* involve all $2(L+1)^2$ unknowns. When putting all operators together, each section represents an equatlity between two series, so their coefficients are made to match. This results in a total of $(L+1)^2$ equations per section. The two sections then allow to solve all $2(L+1)^2$ unknowns.

When this process is carried out explicitely, however, two things occur that allow to reduce the size of the system:

- The equations for the $u$ and $v$ sections do not mix coefficients with different order $m$, so that one could gather all equations with the same $m$ and solve them independently. For this reason, Kore works in a *per-order* basis, where a value of $m$ is specified and the appropriate system is built. Each order $m$ contains only $2(L-|m|+1)$ terms and unknowns.
- Within a fixed order $m$, each poloidal coefficient $\mathcal P_{l,m}$ is only related in the $u$ section with the toroidal coefficients of adjacent degree, i.e. $\mathcal T_{l+1,m}$ and $\mathcal T_{l-1,m}$. Similarly, the $v$ section only relates each toroidal coefficient $\mathcal T_{l,m}$ with poloidal coefficients $\mathcal P_{l+1,m}$ and $\mathcal P_{l-1,m}$. This results in couplings that skip every other degree, so that poloidal coefficients of even degree will **only** be coupled with toroidal coefficients of odd degree - and viceversa. Thus, one could effectively build a system with only poloidal terms of a single parity and toroidal terms of the other parity. These only represent half of the coefficients, and so the size of the system is further reduced to just $L-|m|+1$. Kore follows this simplification and only builds the system pertaining to the parity that is specified.

This second point has geometric significance. Note that the parity of the spherical harmonic functions with respect to the $\cos\theta$ variable is given by the parity of their degree: harmonics with even degree are even themselves. Thus, harmonics with even parity are symmetric across the equator, while harmonics with odd parity are antisymmetric. A solution yields a set of poloidal coefficients of one parity and a set of toroidal coefficients of the opposing parity. By writing the expressions out and reasoning what *equatorial symmetry* means for each of the three components of $\mathbf u$, it turns out that the parity of $\mathbf u$ is the same as that of $\mathcal P$ and opposite to that of $\mathcal T$. A system with even degrees for $\mathcal P_{l,m}$ produces a symmetric flow, and viceversa. It is this symmetry of the solution that Kore uses to build the appropriate system of equations.

Note that this set of equations has two very distinct halves: one coming from the $u$ section, the other coming from the $v$ section. Within each half, the equations are exactly the same, but they just involve different poloidal-toroidal coefficients. If one was to write them in matrix form, where each entry is an operator, they would find that the rows within each half just get displaced.


## Radial discretization, inner cores and symmetry considerations

The previous angular discretization results in a set of $L-|m|+1$ ODEs in $r$ to solve for all the involved poloidal-toroidal coefficients. To solve them numerically, the radial dependance of each coefficient on $r$ is expressed in terms of [Chebyshev functions](https://en.wikipedia.org/wiki/Chebyshev_polynomials) $T_k(r)$ as:

$$
\mathcal P_{l,m} = \sum_{k=0}^N\mathcal P_{l,m}^kT_k(r)\ \ \ \ \ \ \ \ \ \ \ \ \ \mathcal T_{l,m} = \sum_{k=0}^N\mathcal T_{l,m}^kT_k(r)
$$

Here, the coefficients $\mathcal P_{l,m}^k$ are constants and do not depend on any spatial coordinate. Again, the equalities should only hold for an infinite sum, but the truncated $N+1$ term expansion is assumed to be good enough and indistinguishable from the infinite series. By means of this expansion, each one of the $L-|m|+1$ equations obtained from the angular discretization can be written as an equality between two series, whose terms are made to match. Although the idea is the same as with the angular discretization, the process in this case is somewhat more complicated because the Chebyshev family does not offer as convenient differentiation properties as those of the spherical harmonics, but rather involve other families of polynomials jointly known as the *ultraspherical* or [*Gegenbauer* polynomials](https://en.wikipedia.org/wiki/Gegenbauer_polynomials). A detailed explanation of the full extent of the discretization in terms of Gegenbauer polynomials is provided by [Olver and Townsend (2013)](https://epubs.siam.org/doi/abs/10.1137/120865458). It is this algorithm that Kore implements, and a short keynote summary is provided here.

The algorithm is thought of as working with vectors and vector bases. Each family of Gegenbauer polynomials has a generating function whose behaviour is determined by a parameter $\lambda$ and thus the family is denoted as $C^{(\lambda)}$, with each member of the family being denoted as $C^{(\lambda)}_k$. A function can be written as an expansion series with any of those families, and a vector collecting the coefficients of the series in each family is said to be *in the basis* of that family. Differentiation of a function written in a $C^{(\lambda)}$ series can be performed by means of a matrix operation $\mathcal D_{\lambda}$, which acts on the vector of coefficients in the $C^{(\lambda)}$ basis and returns the coefficients of the derivative, now in the $C^{(\lambda+1)}$ basis. A second derivative is just the successive application of such a matrix, the result of which is provided in the $C^{(\lambda+2)}$ basis. One can imagine that an operator involving different derivatives of different orders, like the viscous diffusion operator above, will naturally involve vectors in different bases. Writing them in the same basis is achieved by pre-multiplication with a matrix $\mathcal S_\lambda$, which takes a vector in the $C^{(\lambda)}$ basis and returns the coefficients of the same function in the $C^{(\lambda+1)}$ basis. Products between two functions can be done in *Gegenbauer space* by using a matrix $\mathcal M_\lambda$, which is built with the coefficients of one of the functions and pre-multiplies the vector of coefficients of the other function. The result is the vector of coefficients of the product of the two functions. The inputs need to be in the same basis, and the output is given in that basis as well.

In the end, each of the operators obtained in the angular discretization can be written by an appropriate combination of these $\mathcal D_\lambda$, $\mathcal S_\lambda$ and $\mathcal M_\lambda$. This is exactly what Kore does. For each of the $L-|m|+1$ equations obtained in the angular discretization, the operators apply on the vector of coefficients of each of the involved $\mathcal P_{l,m}$ and $\mathcal T_{l,m}$. Each of these scalar equations then turn into $N+1$ scalar equations which, stacked together, constitue a system of $(N+1)(L-|m|+1)$ equations to solve for the same number of unknowns. This step effectively turns each entry of the matrix obtained with the angular discretization into a sub-matrix itself, while each entry in the unknown vector from before gets turned into a sub-vector itself. The structure that was mentioned earlier for the entries of the matrix in the angular discretization is maintained by the submatrices in this second discretization.

Something important to keep in mind when using Chebyshev or Gegenbauer polynomials is that their domain is limited to the $[-1,1]$ interval. Therefore, the radial domain of the problem needs to be mapped accordingly. Here, two situations need to be distinguished.

- In the presence of an inner core, the radial domain is $[r_{icb},r_{cmb}]$. This interval of $r$ is scaled and shifted to match the $[-1,1]$ interval of $x$ as:

$$
x = 2\frac{r - r_{icb}}{r_{cmb} - r_{icb}} - 1
$$

- In the absence of an inner core, mapping the interval $[0,r_{cmb}]$ to the interval $[-1,1]$ creates conflicts at the origin (see [Rekier et al., 2019](https://academic.oup.com/gji/article/216/2/777/5159470)) and it is more convenient to map the whole diameter $[-r_{cmb}, r_{cmb}]$ to the Chebyshev domain. This is done through the transformation:

$$
x = \frac{r}{r_{cmb}}
$$

In any case, the assumed symmetry that was used in the angular discretization is to be mantained, which imposes further restrictions on the shape of the radial functions. In the end, only Chebyshev polynomials with the same parity as the flow can be part of the solution, which is to say that all coefficients corresponding to the other parity need to vanish. Imposing this reduces the number of equations and unknowns by half, yielding a final system with $(N+1)(L-|m|+1)/2$ equations and unknowns.

## Boundary conditions, generalization and final notes

For a fully defined problem, the differential equation(s) discretized above need to be complemented with boundary conditions, all of which are discretized in exactly the same manner. TODO: Not sure about this. There is this part with generalized spherical harmonics and the canonical basis that I am not sure how it translates to the final matrix system.

Although the numerical scheme has been illustrated using the momentum equation, exactly the same type of procedure can be followed for the induction equation and its $f$ and $g$ sections, and the already scalar equations of heat and composition. However, the induction equation involves a product of the induced field $\mathbf b$ and the background field $\mathbf B_o$, both of which are written in spherical harmonics (see . The product of two spherical harmonics involves the so-called [Wigner 3-j symbols](https://en.wikipedia.org/wiki/3-j_symbol), which greatly complicates the analytical approach. In practice, in order to know what derivatives are needed in each operator, a Mathematica package called [TenGSHui](https://meetingorganizer.copernicus.org/EPSC2013/EPSC2013-919.pdf) had been previously used, and the operators it output were implemented in Kore.

Another final detail is that, in the implementation of Kore, factors of $1/r$ have been removed from the differential equations for the $r$-dependent coefficients of the spherical harmonic expansions. These equations have been multiplied by whatever power of $r$ is required to eliminate all $r$ from the denominators. This renders all differential operators in $r$ to be of the form $r^\alpha(d/dr)^\beta$ with some $\alpha,\beta\geq0$.

At the end of the whole discretization process, the whole system can be written in matrix form. Due to the shape of [the original equations](https://yorchmcm.github.io/kore_sandbox/3nondim/), this system will have two parts: one of them proportional to $\lambda$ - which has not really been fixed - and another part independent of it. This allows to write the system in the form $\mathbf A\mathbf x = \lambda\mathbf B\mathbf x$, with $\mathbf x$ being the vector containing all $(N+1)(L-|m|+1)/2$ unknowns. This is a (massive) generalized eigenvalue problem, which is not solved entirely but rather an algorithm is used to solve only for eigenvalues close to a specified ''fixed point''.
