# Background flow

Kore can solve for perturbations of an axisymmetric background flow $\mathbf{U}(r,\theta)$ in the rotating frame. An example is the steady flow driven by a differentially rotating inner core (`forcing = 12`), whose modes are then the instabilities of the Stewartson layer. This is for the flow only (`magnetic = thermal = compositional = 0`) and needs an inner core (`ricb > 0`). The code is in `bin/bgflow.py`.

## Equations

Linearising the advection term about $\mathbf{U}$ adds to the momentum equation, on the left-hand side as the Coriolis force,

$$
\mathbf{F} = (\mathbf{U}\cdot\nabla)\mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{U}.
$$

$\mathbf{U}$ is a real flow with $m = 0$, given by its poloidal and toroidal scalars as for $\mathbf{u}$,

$$
\mathbf{U} = Ro \sum_n \left[ \nabla\times\nabla\times\left(P_n Y_n\,\mathbf{r}\right) + \nabla\times\left(T_n Y_n\,\mathbf{r}\right) \right],
$$

with $Y_n$ the Legendre polynomial of degree $n$ in $\cos\theta$ and $Ro$ = `bgflow_Ro` (in units of $\Omega$; the code multiplies it by `OmgTau`). Since $\nabla\cdot\mathbf{U} = 0$ and $\mathbf{u} = 0$ on the walls, the term $(\mathbf{U}\cdot\nabla)\mathbf{u}$ does no work, and the background flow feeds a perturbation at the rate

$$
W = -2\,\mathrm{Re}\int \mathbf{u}^*\cdot(\mathbf{u}\cdot\nabla)\mathbf{U}\,dV,
$$

which `spin_doctor.py` adds to the kinetic energy balance (`resid𝐮`, and column 20 of `flow.dat`).

## Projection

A flow of degree $l$ (and $\mathbf{U}$, with $m = 0$) has, in $(r, \theta, \phi)$ components,

$$
\mathbf{u} = q\,(L Y, 0, 0) + s\,(0, \partial_\theta Y, i Z) + T\,(0, i Z, -\partial_\theta Y),
\qquad q = P/r, \quad s = (rP)'/r, \quad Z = \frac{m Y}{\sin\theta}, \quad L = l(l+1).
$$

With $(\mathbf{a}\cdot\nabla)\mathbf{b}$ written out in spherical components, $r\mathbf{F}$ is a sum of products $b\,\rho$, $b\,(r\rho')$ and $(rb')\,\rho$ of a radial factor $b$ ($q$, $s$ or $T$) of $\mathbf{U}$ and one $\rho$ of $\mathbf{u}$, each times an angular function. With $r\mathbf{F} = \sum_{l'} \left( F^r\,Y_{l'}\,\hat{\mathbf{r}} + F^s\,r\nabla Y_{l'} + F^t\,\hat{\mathbf{r}}\times r\nabla Y_{l'} \right)$, the rows of degree $l'$ of the momentum equation get

$$
r^5\,\hat{\mathbf{r}}\cdot\nabla\times\nabla\times\mathbf{F} = L'\left( r^2 F^r - r^3\,\partial_r F^s \right),
\qquad
r^3\,\hat{\mathbf{r}}\cdot\nabla\times\mathbf{F} = -L'\,r\,F^t.
$$

The angular functions are projected on $Y_{l'}$, $r\nabla Y_{l'}$ and $\hat{\mathbf{r}}\times r\nabla Y_{l'}$ by Gauss-Legendre quadrature in $\cos\theta$, which is exact for these polynomial integrands, and the radial operators are built as in `submatrices.py`. A term of $\mathbf{U}$ of degree $n$ couples $l$ to the degrees $l'$ with $|l - l'| \le n$.

## Nonlinear basic state

The steady flow at a finite $Ro$ is not $Ro$ times the linear one: the advection $(\mathbf{U}\cdot\nabla)\mathbf{U}$ changes it at order $Ro^2$, with the same sign for both signs of $Ro$. In kore's field $x$ (the flow is $\mathbf{U} = 2x$, field + c.c.) the steady problem is

$$
A\,x + \tfrac{1}{2}\,B(2x)\,x = b,
$$

where $A\,x = b$ is the linear `forcing = 12` problem and $B(\mathbf{U})$ is the matrix of $\mathbf{F}$ with background $\mathbf{U}$. $B(2x)$ is also the Jacobian of the nonlinear term, so a Newton step is one forced solve,

$$
(A + B_k)\,x_{k+1} = b + \tfrac{1}{2}\,B_k\,x_k, \qquad B_k = B(2x_k).
$$

`bgflow.py` takes $Ro$ from zero to `forcing_amplitude_icb` in steps of at most 0.15, with a few Newton steps each. All the degrees of $x_k$ enter $B_k$; with fewer, the convergence is only linear. The matrices are nearly dense, so the solves are dense (about 20 s per step at `N = 48`, `lmax = 95`).

## Usage

1. Set up the steady flow with `forcing = 12` (`m = 0`, `symm = 1`, `forcing_frequency = 0`) at the `Ek` and `ricb` of the stability problem.
2. Either, for the linear flow, solve it and run `python3 bin/bgflow.py nmax` in that folder. This writes `bgflow.npz` with $P_n$ and $T_n$ for $n \le$ `nmax`, per unit `forcing_amplitude_icb`, for any `bgflow_Ro`.
   Or, for the nonlinear flow at $Ro$, set `forcing_amplitude_icb` $= Ro$, run `submatrices.py`, then `python3 bin/bgflow.py nmax nonlinear`. The file then holds that $Ro$, and the stability runs must use `bgflow_Ro` $= Ro$ (checked).
3. In the stability run set `bgflow = 1`, `bgflow_file` (the path of that file) and `bgflow_Ro` $= \Delta\Omega_{\rm icb}/\Omega$.

The number of degrees needed grows as the boundary layers thin: at $E = 10^{-4}$ and `ricb = 1/3`, the onsets with `nmax = 40` differ from those with `nmax = 60` by less than $10^{-4}$ in $Ro$. Since each term couples $|l - l'| \le n$, the matrix $A$ is nearly dense when `nmax` is comparable to `lmax`.

For example, at `Ek = 1e-4` and `ricb = 1/3` (`nmax = 40`; `N = 48`, `lmax = m + 95`, `symm = 1`) the first unstable modes on the nonlinear basic state are $m = 1$ at $Ro = -0.647$ and $m = 3$ at $Ro = 0.419$, followed by $m = 4$ at $0.446$ (Hollerbach 2003: $-0.651$, $0.414$, $0.453$). On the linear one they are $m = 1$ at $-0.644$ and $m = 3$ at $0.452$, and $m = 4$ stays stable.

A rigid rotation, $\mathbf{U} = Ro\,\hat{\mathbf{z}}\times\mathbf{r}$ (one term, $T_1 = r$), is an exact test: with `OmgTau = 1` the eigenvalues are those without background flow at `OmgTau` $= 1+Ro$ and `Ek` $/(1+Ro)$, minus $i m Ro$.
