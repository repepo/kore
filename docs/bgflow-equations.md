# Background flow equations

This page gives the background flow of `bgflow = 1` ([Background flow](bgflow.md)), the terms it adds to the
perturbation equations, how they become rows, what happens at the walls, the powers that `spin_doctor` reports, and the
Newton's method that gives the nonlinear flow of turning walls. They are the terms of the `conductive_IC` branch's
`bgflow.py`, written in `bin/bgflow.py` for any density, in a shell or a full sphere, with the entropy, the composition
and the magnetic field.

## The background

The background is written as the perturbations are ([Numerical method](numerical-method.md#unknowns)), with a poloidal
and a toroidal scalar for each degree, so that it carries no mass across a sphere whatever its scalars:

$$
\mathbf U = \mathrm{Ro}\thinspace G\sum_n\left[\frac{1}{\rho_0}\nabla\times\nabla\times(\rho_0 P_n Y_n\thinspace\mathbf r)
+ \nabla\times(T_n Y_n\thinspace\mathbf r)\right],
\qquad S_U = \mathrm{Ro}\sum_n\sigma_n Y_n, \qquad C_U = \mathrm{Ro}\sum_n\gamma_n Y_n ,
$$

with $Y_n$ the Legendre polynomial of degree $n$ in $\cos\theta$, $\mathrm{Ro}$ = `bgflow_Ro`, $G$ = `Gaspard` (1 when
`Gaspard = 0`), and $\nabla\cdot(\rho_0\mathbf U) = 0$. $S_U$ and $C_U$ are the entropy and the composition that the
flow brought about in the run that made it. With $\lambda_1 = (\ln\rho_0)'$ and a prime for $d/dr$, the components are

$$
U_r = \mathrm{Ro}\thinspace G\sum_n\frac{n(n+1)P_n}{r}Y_n, \qquad
U_\theta = \mathrm{Ro}\thinspace G\sum_n\left(P_n' + \frac{P_n}{r} + \lambda_1 P_n\right)\partial_\theta Y_n, \qquad
U_\phi = -\mathrm{Ro}\thinspace G\sum_n T_n\thinspace\partial_\theta Y_n .
$$

$T_1 = r$ is a rotation at the rate $\mathrm{Ro}\thinspace G$, and at Ro = 1 $T_1 = (f_0 - f_2/5)\thinspace r$,
$T_3 = f_2\thinspace r/5$ is the differential rotation $\Delta\Omega = f_0 + f_2 Y_2$ of `diff_rot`. The poloidal
scalars make the meridional circulation. The vorticity $\boldsymbol\Omega_U = \nabla\times\mathbf U$ follows degree by
degree: $\Omega_\phi = \mathrm{Ro}\thinspace G\left[(rs_n)'/r - n(n+1)P_n/r^2\right]\partial_\theta Y_n$ from $P_n$, with
$s_n = P_n' + P_n/r + \lambda_1 P_n$, and $\Omega_r = \mathrm{Ro}\thinspace G\thinspace n(n+1)T_n Y_n/r$,
$\Omega_\theta = \mathrm{Ro}\thinspace G\thinspace(rT_n)'\thinspace\partial_\theta Y_n/r$ from $T_n$.

Like the perturbations, the background is symmetric about the equator: $P_n$, $\sigma_n$, $\gamma_n$ of even $n$ and
$T_n$ of odd $n$.

## The terms

The perturbation equations, linearised about $\mathbf U$, $S_U$ and $C_U$, gain the advection of the perturbation by
the background and of the background by the perturbation. The momentum equation gains

$$
(\mathbf U\cdot\nabla)\mathbf u + (\mathbf u\cdot\nabla)\mathbf U
= \nabla(\mathbf U\cdot\mathbf u) - \mathbf U\times\boldsymbol\omega - \mathbf u\times\boldsymbol\Omega_U ,
\qquad \boldsymbol\omega = \nabla\times\mathbf u ,
$$

on the side of the inertia; the gradient joins the pressure and leaves the rows, which take the curl and the double curl
of the equation. The heat and the composition equations gain

$$
p_0\left(\mathbf U\cdot\nabla s + \mathbf u\cdot\nabla S_U\right), \qquad
\rho_0\left(\mathbf U\cdot\nabla c + \mathbf u\cdot\nabla C_U\right)
$$

on the side of $\partial_t$, and the induction equation gains $\nabla\times(\mathbf U\times\mathbf b)$ on the side of
the induction. The background itself is held fixed: what keeps $\mathbf U$, $S_U$ and $C_U$ steady (the walls, the
buoyancy of $S_U$ and $C_U$, the advection of $\mathbf U$ by itself) belongs to the equations of the background, not to
those of the perturbation.

## Rows and implementation

This section says how the terms become radial operators and where they sit in the matrices. It matters only to change
or extend the code.

- **Angular couplings.** The background vectors are polynomials in $\cos\theta$ times $\hat{\mathbf r}$,
  $\sin\theta\thinspace\hat{\boldsymbol\theta}$ or $\sin\theta\thinspace\hat{\boldsymbol\phi}$, and their products
  with the perturbation are sparse matrices over the degrees, built exactly with the recurrence algebra of `angular.py`,
  as for a background field ([Magnetic field equations](magnetic-equations.md#couplings-for-any-background)): a degree
  $n$ of $\mathbf U$ couples the degree $l$ of a row to $l'$ with $\lvert l - l'\rvert \le n + 1$. `bgflow.build`
  writes each term as a sum of radial monomials times these matrices, in kore's rows: $r^5\rho_0$ and $r^3\rho_0$ times
  $\hat{\mathbf r}\cdot\nabla\times\nabla\times$ and $\hat{\mathbf r}\cdot\nabla\times$ of the momentum equation
  (sections `u`, `v`), $r^2$ times the heat and composition equations (`h`, `c`), and the rows `f`, `g` of the induction.
- **Labels and profile codes.** A monomial $r^p\thinspace w_B^{(j)}\thinspace X^{(d)}$ of a row is the label
  `{row}{q}w{B}{n}{j}[lho{k} | pss0 | dns0]_D{d}`, with $q$ the power of the row minus $p$, $w_B$ a profile code of
  degree $n$ and its $j$-th derivative, `lho{k}` a factor $(\ln\rho_0)^{(k)}$, `pss0` and `dns0` a factor $p_0$ or
  $\rho_0$, and $X^{(d)}$ the $d$-th derivative of the unknown. For example `v2wt10_D1` is
  $r\thinspace\mathrm{Ro}\thinspace G\thinspace T_1 P_l'$ in the rows of $T_l$, times its angular coefficient. The profile codes, per degree $n$ (one character, 0–9 then a–z: degrees
  up to 35), are

    | code | profile |
    | --- | --- |
    | `wq` | $\mathrm{Ro}\thinspace G\thinspace n(n+1)P_n/r$ |
    | `ws` | $\mathrm{Ro}\thinspace G\left(P_n' + P_n/r + \lambda_1 P_n\right)$ |
    | `wt` | $\mathrm{Ro}\thinspace G\thinspace T_n$ |
    | `wh`, `wd` | $\mathrm{Ro}\thinspace p_0\sigma_n$, $\mathrm{Ro}\thinspace p_0\sigma_n'$ (without $p_0$ where the heat rows are divided by it) |
    | `wc`, `we` | $\mathrm{Ro}\thinspace\rho_0\gamma_n$, $\mathrm{Ro}\thinspace\rho_0\gamma_n'$ (without $\rho_0$ where the rows of the composition are divided by it) |

- **Assembly.** `assemble.bgflow_blocks` places the blocks of each row degree for every $\Delta l$ up to the largest
  degree of the background plus one, in the rows `u`, `v` (columns $P$, $T$), `h` ($s$, $P$, $T$), `c` ($c$, $P$, $T$)
  and `f`, `g` (columns of the field). `operators.background_flow` returns each block as the sum of the angular
  coefficients times the radial operators, divided by $\mathcal L^3$ of the row as every operator.
- **Without diffusion.** With `ThermaD = 0` the heat rows are $r^2(\text{eq})/p_0$ in the $C^{(1)}$ basis, as with the
  thermal wind ([Differential rotation](differential_rotation_details.md)): $\mathbf u\cdot\nabla S_U$ holds $u_\theta$,
  which holds $P_l'$, and $\mathbf U\cdot\nabla s$ holds $U_\theta\partial_\theta s/r$. With `CompoD = 0` the rows of
  the composition are $r^2(\text{eq})/\rho_0$ in $C^{(1)}$ in the same way.
- **Inviscid runs** (`ViscosD = 0`, full sphere). The terms hold $P_l'''$ in the rows `u` and $P_l''$ in the rows `v`,
  so these rows are in $C^{(3)}$ and $C^{(2)}$ and are $r^5\rho_0^2$ and $r^3\rho_0$ times the equations, as with a
  magnetic field.
- **Full sphere.** Each scalar has the parity $(-1)^n$ of its degree, and the codes `wq`, `ws`, `wd`, `we` the opposite
  one. An expression is evaluated at $\lvert r\rvert$ with that parity and interpolated at the Chebyshev–Lobatto points
  of $[-1, 1]$ (of $[r_i, 1]$ in a shell), so that a scalar that vanishes at the wall vanishes there to round-off.
  A scalar of degree $n$ is smooth at the centre if it goes as $r^n$ times a series in $r^2$; `submatrices.py` warns
  for each one that does not ([Regularity at the centre](numerical-method.md#regularity-at-the-centre)).

## The walls

kore's walls are impenetrable, so the background must not cross them: $P_n = 0$ at each wall, which `submatrices.py`
checks to $10^{-10}$ of the largest $\lvert P_n\rvert$, $\lvert T_n\rvert$. A conducting solid at a wall (an inner core,
a thin conducting layer, a resolved mantle) carries its field along as a solid body, so the background must turn
rigidly there: $P_n = P_n' = 0$, and $T_n = 0$ for $n \neq 1$. The solid then turns with the fluid at the wall, at

$$
\Omega_w = \mathrm{Ro}\thinspace G\thinspace\frac{T_1(r_w)}{r_w}
$$

relative to the frame, and the equation of its field gains the advection by that rotation, $-i\thinspace m\thinspace
\Omega_w\thinspace\mathbf b$ (`bgflow.rigid_rate`, added as for a differential rotation). The Bessel inner core and
`spin_doctor`'s energies of the solids use the frequency in the frame of the solid.

With a background field $\mathbf B_0$ the linear problem also assumes that the field is steady under the background
flow. `bgflow.steadiness` reports

$$
\max\lvert\nabla\times(\mathbf U\times\mathbf B_0)\rvert\thinspace r \Big/ \left(\max\lvert\mathbf U\rvert\max\lvert\mathbf B_0\rvert\right)
$$

on a grid of the fluid (with the flow of a differential rotation added), and warns above $10^{-7}$, or stops with
`ferraro_policy = 'refuse'`. A rotation about the axis keeps an axisymmetric field steady when the rotation rate is
constant along the field lines ([Magnetic field equations](magnetic-equations.md#the-ferraro-condition)), so a rigid
rotation keeps any of them.

## Energy budgets

`spin_doctor` adds the background's terms to the budgets of [Energy balance](anelastic.md#energy-balance), computed on
a grid of Gauss–Legendre points in $\cos\theta$ that makes the angular products exact (`utils4pp.bgflow_setup`):

$$
W_\mathrm{bg} = -2\thinspace\mathrm{Re}\int\rho_0\thinspace\mathbf u^*\cdot\left[(\mathbf U\cdot\nabla)\mathbf u + (\mathbf u\cdot\nabla)\mathbf U\right] dV ,
$$

its imaginary part, and the enstrophy term of the same force. The part of $(\mathbf U\cdot\nabla)\mathbf u$ has no real
part, $\mathrm{Re}\int\rho_0\thinspace\mathbf u^*\cdot(\mathbf U\cdot\nabla)\mathbf u\thinspace dV = \tfrac12\int\rho_0\thinspace\mathbf U\cdot\nabla\lvert\mathbf u\rvert^2\thinspace dV = 0$,
since $\nabla\cdot(\rho_0\mathbf U) = 0$ and $\mathbf U$ runs along the walls: $W_\mathrm{bg}$ is the power that the
perturbation draws from the shear of the background through $(\mathbf u\cdot\nabla)\mathbf U$. In the same way
$\mathbf r\times[(\mathbf U\cdot\nabla)\mathbf u + (\mathbf u\cdot\nabla)\mathbf U] = (\mathbf U\cdot\nabla)(\mathbf r\times\mathbf u) + (\mathbf u\cdot\nabla)(\mathbf r\times\mathbf U)$,
whose integral with $\rho_0$ is a flux through the walls, zero when neither flow crosses them: the background puts no
torque in the fluid, and the budget of its angular momentum (resid𝐋, axial and equatorial) holds as without it. The heat and composition budgets gain

$$
W_\mathrm{bg}^{s} = -2\thinspace\mathrm{Re}\int p_0\thinspace s^*\left(\mathbf U\cdot\nabla s + \mathbf u\cdot\nabla S_U\right) dV ,
\qquad
W_\mathrm{bg}^{c} = -2\thinspace\mathrm{Re}\int\rho_0\thinspace c^*\left(\mathbf U\cdot\nabla c + \mathbf u\cdot\nabla C_U\right) dV .
$$

The force is computed with its gradient $\nabla(\mathbf U\cdot\mathbf u)$, so that the pressure that `spin_doctor`
derives from the momentum equation, and its power at the walls, are the physical ones. The columns are in
[Output files](output.md).

## Newton's method

The nonlinear flow of turning walls is the steady solution of the full axisymmetric equations, in which the flow also
carries itself and its entropy and composition along. `bgflow.py` finds it with kore's own matrices. For $m = 0$ the
fields of a forced run are $\mathbf x + \mathrm{c.c.} = 2\mathbf x$, so the steady problem reads

$$
A\mathbf x + \tfrac12 B(2\mathbf x)\thinspace\mathbf x = \mathbf b ,
$$

with $A\mathbf x = \mathbf b$ the linear problem of forcing 12 and $B(\mathbf V)$ the matrix of the terms of a
background $\mathbf V$. $B(\mathbf V)\mathbf w$ is linear and symmetric in its two flows, so $B(2\mathbf x)$ is also the
Jacobian of the quadratic term, and a Newton step is

$$
\left(A + B_k\right)\mathbf x_{k+1} = \mathbf b + \tfrac12 B_k\thinspace\mathbf x_k, \qquad B_k = B(2\mathbf x_k),
$$

with every degree of $\mathbf x_k$ in $B_k$. $B_k$ is the matrix of a run with `bgflow = 1` and the background
$2\mathbf x_k$ minus that of the same run at `bgflow_Ro = 0`, so that both have the same rows and bases. Ro rises from 0
in steps of at most 0.15, each starting from the flow of the last step with its part of order $\mathrm{Ro}^2$ scaled to
the new Ro, and the iterations of a step stop when one moves $\mathbf x$ by less than $10^{-9}$ of itself. The matrices of $m = 0$
couple every degree of the background with every other, so the solves are dense. At Ek $10^{-3}$ and Ro 1
([the example](bgflow.md#example-waves-on-the-flow-of-an-inner-core-that-turns)), each step takes three or four
iterations, the change squaring at each, and the flow differs from Ro times the linear flow by 6%.

## The file

`bgflow.py nmax [nonlinear]` writes `bgflow.npz`:

| key | content |
| --- | --- |
| `nP`, `P`; `nT`, `T` | the degrees $n \le$ nmax and the Chebyshev coefficients of $P_n$, $T_n$ per unit Ro, one row per degree, on $[r_i, 1]$ (on $[-1, 1]$, of parity $(-1)^n$, in a full sphere) |
| `nS`, `S`; `nC`, `C` | the same for $\sigma_n$ and $\gamma_n$, with `thermal = 1` and `compositional = 1` |
| `ricb` | the radius of the inner wall |
| `rho_r`, `rho` | the density at 8 radii: the poloidal flow is that density's |
| `wall`, `ratio` | the wall that turned (`'inner'` or `'outer'`), and the rate of the other relative to it |
| `Ro` | for a nonlinear flow, the Ro it holds at |

P and T are the scalars of the run's flow divided by the rate of the wall in units of $1/\tau$, and $\sigma_n$,
$\gamma_n$ those of its entropy and composition divided by that rate over $G$. The files of `conductive_IC` hold `nP`, `P`, `nT`, `T`, `ricb` (and `Ro`), and are read as they are,
for a uniform density.
