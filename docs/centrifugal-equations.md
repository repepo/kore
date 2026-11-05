# Centrifugal deformation equations

This page gives the mathematics behind `centrifugal = 1`, for readers who want to check the code or extend it: the
coordinates that follow the level surfaces, the equations in them and their rows, the flattening of each shape, the
effective gravity, the energy budgets, the plots, the accuracy, and the checks. How to set up a run is on
[Background model](background.md#centrifugal-deformation). The code is in `bin/centrifugal.py`, the checks in
`tests/centrifugal/`.

## Coordinates that follow the level surfaces

kore's flow conserves mass exactly only when the density depends on the radius alone. In a flattened body the density
is constant on spheroids, so rather than give every profile a part that depends on latitude, kore maps the body onto a
reference space in which the level surfaces are spheres.

A point $\boldsymbol\alpha$ of the reference space, at radius $r$ and colatitude $\theta$, is at
$\mathbf X = \boldsymbol\alpha + \boldsymbol\xi$ in the body, with

$$
\boldsymbol\xi = \varepsilon(r)\thinspace\nabla\left(\tfrac12 r^2P_2(\cos\theta)\right)
= \varepsilon(r)\thinspace r\left[P_2\thinspace\hat{\mathbf r} + \tfrac12\frac{dP_2}{d\theta}\thinspace\hat{\boldsymbol\theta}\right],
\qquad \varepsilon = -\tfrac23 f ,
$$

where $f(r)$ is the flattening of the level surface $r$, positive for an oblate body. At first order that surface is at
the distance $r(1 + \varepsilon P_2)$ from the centre: its flattening is $f$ and its volume that of the sphere of radius
$r$. In Cartesian components $\boldsymbol\xi = \varepsilon(r)\thinspace(-\alpha_x/2, -\alpha_y/2, \alpha_z)$, an
axisymmetric stretch at each radius, so the map is smooth at the centre when $\varepsilon$ is even and smooth in $r$; a
purely radial $\boldsymbol\xi$ would not be.
Every background profile is a function of $r$, and the walls are the level surfaces $r$ = `ricb` and $r = 1$.

With the deformation gradient $\mathsf F = \partial\mathbf X/\partial\boldsymbol\alpha = \mathsf I + \nabla\boldsymbol\xi$
and $J = \det\mathsf F$, kore's poloidal and toroidal scalars describe the flow carried back to the reference space,

$$
\mathbf w = J\thinspace\mathsf F^{-1}\mathbf u .
$$

This transform keeps what kore's formulation relies on. Mass: $\nabla_\alpha\cdot(\rho_0\mathbf w) = J\thinspace\nabla_X\cdot(\rho_0\mathbf u)$,
so the exact $\nabla\cdot(\rho_0\mathbf w) = 0$ of kore's scalars is exact mass conservation in the body. Walls:
$\mathbf u\cdot\mathbf n$ is proportional to $w_r$ on a level surface, and $\mathbf u = 0$ where $\mathbf w = 0$, so the
impenetrable and no-slip rows are unchanged.

## Equations

kore removes the pressure by taking curls of the momentum equation. In the reference space this works when the equation
is written with covariant components, $\mathsf F^{\mathsf T}$ times it: the pressure force stays a gradient,
$\mathsf F^{\mathsf T}\nabla_X\Pi = \nabla_\alpha\Pi$. So does the buoyancy, which acts along the gradient of the
effective potential, constant on the level surfaces. The dimensionless equations of
[Anelastic equations](anelastic.md#non-dimensionalization) become

$$
\mathsf M\thinspace\partial_t\mathbf w + 2\thinspace\mathrm{Gaspard}\thinspace\tilde{\mathbf z}\times\mathbf w = -\nabla\Pi + \mathrm{Beyonce}\thinspace g_{\rm eff}\thinspace s\thinspace\hat{\mathbf r} + \dots ,
\qquad \mathsf M = \frac{\mathsf F^{\mathsf T}\mathsf F}{J}, \quad \tilde{\mathbf z} = \mathsf F^{-1}\hat{\mathbf z},
$$

$$
p_0\left(J\thinspace\partial_t s + w_r\frac{dS}{dr}\right) = \dots ,
$$

with the dots the viscous force and the thermal diffusion. Three terms change: the inertia gains the metric
$\mathsf M$, the Coriolis force turns with the map ($\mathsf F^{\mathsf T}(\hat{\mathbf z}\times\mathsf F\mathbf w)/J = \tilde{\mathbf z}\times\mathbf w$),
and the rate of change of the entropy gains $J$. The advection keeps its form, $\mathbf u\cdot\nabla_X S = w_r S'/J$,
and the buoyancy takes the effective gravity $g_{\rm eff}$ ([below](#effective-gravity)).

**At first order.** With $\eta = r\varepsilon'$ and $w_z = \hat{\mathbf z}\cdot\mathbf w$,

$$
(\mathsf M - \mathsf I)\thinspace\mathbf w = \varepsilon\left(3w_z\hat{\mathbf z} - \mathbf w\right) + \eta\left[\tfrac32\cos\theta\left(w_z\hat{\mathbf r} + w_r\hat{\mathbf z}\right) - w_r\hat{\mathbf r} - P_2\thinspace\mathbf w\right],
$$

$$
(\tilde{\mathbf z} - \hat{\mathbf z})\times\mathbf w = -\varepsilon\thinspace\hat{\mathbf z}\times\mathbf w - \eta\left[\tfrac32\cos^2\theta\thinspace\hat{\mathbf z}\times\mathbf w - \tfrac12\cos\theta\thinspace\hat{\mathbf r}\times\mathbf w\right],
$$

$$
J - 1 = \eta\thinspace P_2 .
$$

A uniform flattening has $\eta = 0$: the map is then a linear stretch, and $J = 1$ at this order.

## Rows

The new terms have the form of the force of a differential rotation, so kore builds them with the same tools. Each one
multiplies the components of $\mathbf w$ by polynomials in $\cos\theta$ of degree 2 at most, with the radial factor
$\varepsilon$ or $\eta$; kore builds their couplings with the recurrence algebra of `bin/angular.py` and projects them on
its rows as it does the force ([Spectral form](differential_rotation_details.md#3-spectral-form)), without the terms
that come from the shear of the rotation.

- **Selection rules.** A degree $l$ couples to $l$ and $l \pm 2$ of the same kind (poloidal to poloidal, toroidal to
  toroidal, entropy to entropy), and to $l \pm 1$ and $l \pm 3$ of the other kind (poloidal to toroidal).
- **Matrices.** $A$ gains $2\thinspace\mathrm{Gaspard}\thinspace(\tilde{\mathbf z} - \hat{\mathbf z})\times\mathbf w$ in
  the momentum rows. $B$ gains $-(\mathsf M - \mathsf I)\thinspace\mathbf w$ in the momentum rows and
  $(J - 1)\thinspace s = \eta P_2\thinspace s$ in the heat rows, with the signs of the inertia and of the entropy. The
  poloidal–toroidal couplings of the inertia are imaginary, so $B$ is complex.
- **Radial operators.** Their labels are those of the differential rotation with the profile code of $\varepsilon$
  (`ce0`) or $\eta$ (`ce1`) in place of that of $\Delta\Omega$, and that of $r\varepsilon'$ (`ce1`) or $r\eta'$ (`ce2`)
  for the derivative terms (`centrifugal.MOMENTUM`). The heat rows use the operator of the entropy times `ce1`. The
  three codes are Chebyshev fits on kore's grid, even in $r$.

## The flattening

The flattening of each level surface follows from hydrostatic balance in the rotating body. `'clairaut'` solves for it,
`'uniform'` and `'user'` take it as given.

**`'clairaut'`.** Clairaut's equation, in Radau's form for $\eta_R = d\ln f/d\ln x$, with $x$ the radius in units of
the radius of the body:

$$
x\frac{d\eta_R}{dx} = 6 - 6\thinspace\frac{\rho}{\bar\rho}\thinspace(\eta_R + 1) - \eta_R^2 + \eta_R, \qquad \eta_R(0) = 0,
$$

with $\bar\rho$ the mean density inside $x$. kore takes the density ratio from the gravity alone, by Poisson's equation,

$$
\frac{\rho}{\bar\rho} = \frac{x g' + 2g}{3g},
$$

so the structure's $g$ must be its self-gravity. kore integrates from the centre and fixes the scale of $f$ by the
surface condition $x f' + 2f = \tfrac52 q$, that is

$$
f(x_t) = \frac{5\thinspace q_t}{2\left(\eta_R(x_t) + 2\right)}, \qquad q_t = q\thinspace x_t^3 ,
$$

at the top $x_t$ of the structure: the surface $x = 1$, or the outer wall for `'model S hydrostatic'`, whose gravity ends
there (the mass above it neglected). With `centrifugal_f` set, the scale instead makes $f$ = `centrifugal_f` at the
outer wall. A homogeneous body has $\eta_R = 0$ and $f = 5q/4$, the Maclaurin spheroid at first order; a polytrope of
index 1 has $f/q = 15/(2\pi^2)$ at the surface. Since $\eta = \varepsilon\thinspace\eta_R$, the codes `ce1` and `ce2`
follow from $\eta_R$ without differentiating:

$$
r\varepsilon' = \varepsilon\thinspace\eta_R, \qquad r\left(r\varepsilon'\right)' = \varepsilon\left[6 - 6\thinspace\frac{\rho}{\bar\rho}\thinspace(\eta_R + 1) + \eta_R\right].
$$

**`'uniform'` and `'user'`.** `'uniform'` has $f$ = `centrifugal_f` and $\eta = 0$. `'user'` evaluates
`centrifugal_profile` at $\lvert r\rvert$ and takes $r\varepsilon'$ and $r(r\varepsilon')'$ from the derivatives of the
Chebyshev fit of $\varepsilon$.

## Effective gravity

The buoyancy acts along the gradient of the effective potential, the gravitational plus the centrifugal one, which is
constant on each level surface. At first order its parts in $P_2$ cancel there, and what remains is

$$
\Phi_{\rm eff} = \Phi_0(r) - \tfrac13\Omega^2r^2, \qquad g_{\rm eff} = \frac{d\Phi_{\rm eff}}{dr} = g - \tfrac23\Omega^2 r .
$$

In kore's units the second term is $-\tfrac23\thinspace q_t\thinspace g(x_t)\thinspace x/x_t$
(`centrifugal.g_monopole`), which `radial_profiles.graviX` adds to $g$ when `centrifugal_q` > 0, with any shape.

## Energy budgets

The energies of the deformed body are integrals over the body. Written in the reference space they gain the metric, so
`spin_doctor` adds the metric parts to its budgets, computed from the fields on its own grid
(`utils4pp.centrifugal_terms`), independently of the matrices. In the conventions of
[Energy balance](anelastic.md#energy-balance):

$$
K = \int\rho_0\thinspace\mathbf w^*\cdot\mathsf M\thinspace\mathbf w\thinspace dV, \qquad
T = \int p_0\thinspace J\thinspace\lvert s\rvert^2\thinspace dV ,
$$

and the power of the Coriolis force is $2\int\rho_0\thinspace\mathbf w^*\cdot[2\thinspace\mathrm{Gaspard}\thinspace\tilde{\mathbf z}\times\mathbf w]\thinspace dV$,
imaginary: it still does no work. `solve.py` normalises the eigenvectors to this $K$. The viscous and diffusive terms are
those of the sphere, as in the matrices, so the budgets close as well as without the deformation. `spin_doctor` does
not compute the enstrophy residual (residens is NaN).

## Plots

`tools/plot_field.py` draws the fields where they are in the body (`centrifugal.to_physical`). The reference point
$(r, \theta)$ is at the distance $\varpi$ from the axis and the height $z$,

$$
\varpi = r\sin\theta\thinspace(1 - \varepsilon/2), \qquad z = r\cos\theta\thinspace(1 + \varepsilon),
$$

and the flow there is $\mathbf u = \mathsf F\mathbf w/J$, at first order

$$
\mathbf u = \mathbf w + \varepsilon\left(\tfrac32 w_z\hat{\mathbf z} - \tfrac12\mathbf w\right) + \eta\thinspace w_r\left(\tfrac32\cos\theta\thinspace\hat{\mathbf z} - \tfrac12\hat{\mathbf r}\right) - \eta P_2\thinspace\mathbf w ,
$$

given in the spherical basis of the body point, turned by $-\tfrac32\varepsilon\sin\theta\cos\theta$ in colatitude from
that of $(r, \theta)$. A scalar, such as $s$, only moves.

## Accuracy

The map, the transform of the flow and the covariant equations are exact. kore keeps them at first order in $f$, and
keeps some terms in the form they have in a sphere.

- **First order.** The error is of order $f^2$. For the spin-over of a Maclaurin spheroid it is $f^2/6$ in the frequency
  ([Checks](#checks)): 0.2 % at Saturn's flattening, where the deformation moves the frequency by 10 %.
  `submatrices.py` warns above $f = 0.1$.
- **Viscous force and thermal diffusion.** Their metric parts are of relative size $f$, so an eigenvalue is off by
  about $f$ times the part of it that the diffusion makes: for a weakly damped mode, a fraction of order $f$ of its
  damping rate, and the same amount in frequency.
- **Differential rotation.** Its terms keep their spherical form: the terms of order $f\thinspace\Delta\Omega$ are left
  out.
- **What stops the run.** The rows of a stress-free wall change on a deformed wall, so kore stops with stress-free walls
  in a viscous run. It also stops with `magnetic = 1` and with a forcing.

## Implementation

Where each piece is in the code:

| piece | code |
| --- | --- |
| the flattening and the profile codes | `centrifugal.radau`, `shape_functions`, `profile_functions` (codes `ce0`, `ce1`, `ce2` in `radial_profiles.proffdir`) |
| the angular couplings | `centrifugal.primitives`, `momentum_coefficients`, `heat_coefficients` |
| labels, blocks, assembly | `centrifugal.labels` (`submatrices.py`), `momentum_block`, `entropy_block` (`operators.centrifugal_inertia`, `centrifugal_coriolis`, `centrifugal_entropy`), `dl_range` (`assemble.centrifugal_blocks`) |
| effective gravity | `centrifugal.g_monopole` in `radial_profiles.graviX` |
| stops and warnings | `background.check_centrifugal`, `centrifugal.report` (`submatrices.py`), the centre check of the flattening |
| energies | `utils4pp.centrifugal_terms` (`solve.py`, `spin_doctor.py`) |
| plots | `centrifugal.to_physical` (`tools/plot_field.py`) |

## Checks

The deformation was checked in four ways: its closed forms against a symbolic derivation from the definitions, its
couplings and blocks against direct quadratures, its eigenvalues against exact results, and the energy budgets. The
scripts are in `tests/centrifugal/` (README; `run_all.sh`, about 5 minutes). Results of 2026-11-05:

| check | script | result |
| --- | --- | --- |
| every closed form above, from the map and $\mathbf w = J\mathsf F^{-1}\mathbf u$ in Cartesian components (sympy): the level surfaces, the exact identities of the covariant equations, the first-order terms, the plot map, the effective potential | `derive.py` | exact |
| the angular couplings against Gauss–Legendre projections of the closed forms, $m$ = 0–6, $l \le 36$ | `algebra_check.py` | 5e-14 |
| the metric blocks against $(r, \theta)$ quadratures, in shells and full spheres; inertia and entropy Hermitian and positive, Coriolis anti-Hermitian | `bin/ru_insane.py` | 6e-14 |
| $f = 0$ gives the matrices of `centrifugal = 0`; `centrifugal = 0` leaves the 26 runs of `tests/diff_rot/regress.sh` unchanged | `eig_check.py`, `regress.sh` | bitwise |
| the spin-over of a Maclaurin spheroid (inviscid, uniform $f$): $\mathrm{Im}\thinspace\lambda = 1 + f + f^2/3$, against the exact $2a^2/(a^2 + c^2) = 1 + f + f^2/2$ | `eig_check.py` | $f/3$ in $(\mathrm{Im}\thinspace\lambda - 1)/f - 1$ for $f = \pm10^{-3}, \pm10^{-2}$ |
| a sphere labelled as a deformed one ($f = c(1 - r^2)^2$, zero at the wall): 13 resolved modes move as $c^2$, with no part odd in $c$ | `eig_check.py` | odd part 1.6e-4 $c^2$ |
| `'clairaut'` of a homogeneous body against `'uniform'` with $f = 5q/4$ | `eig_check.py` | 2e-13 |
| the effective gravity with $g = r$: $q = 0.1$ against `Beyonce` $(1 - 2q/3)$ | `eig_check.py` | below 1e-14 |
| a deformed, stratified ideal fluid has real frequencies | `eig_check.py` | $\lvert\mathrm{Re}\thinspace\lambda\rvert$ 2e-16 |
| `'clairaut'`: a homogeneous body (5/4), the polytrope of index 1 ($15/(2\pi^2)$), `'Wu 2005'` against Clairaut's second-order equation integrated from $\rho$; `'model S hydrostatic'` gives $f/q$ = 0.519 at its wall | `clairaut_check.py` | 3e-12; below 1e-9 |
| the budgets residσ, residω, resid𝑠 of `spin_doctor` (`solve.py`, 2 ranks; `'Wu 2005'` with an inner core, with and without a differential rotation, and a `'Boussinesq'` full sphere) | `budget_check.py` | below 7e-10, as without the deformation; 1e-3 to 3e-2 without the metric parts |
