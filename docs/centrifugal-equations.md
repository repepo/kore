# Centrifugal deformation equations

This page gives the mathematics behind `centrifugal = 1`, for readers who want to check the code or extend it: the
coordinates that follow the level surfaces, the equations in them and their rows, the magnetic field, the flattening of
each shape, the effective gravity, the energy budgets, the plots, the accuracy, and the checks. How to set up a run is on
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

## Magnetic field

A field is carried back to the reference space in the same way as the flow, so that kore's poloidal and toroidal scalars
of the field keep their meaning and most of its rows keep the form they have in a sphere.

**The field as a flux.** kore's scalars describe

$$
\tilde{\mathbf b} = J\thinspace\mathsf F^{-1}\mathbf b, \qquad \tilde{\mathbf B}_0 = J\thinspace\mathsf F^{-1}\mathbf B_0 .
$$

The flux of $\mathbf b$ through a surface of the body is that of $\tilde{\mathbf b}$ through its image in the reference
space, so $\nabla\cdot\tilde{\mathbf b} = 0$ is exact and the field lines of $\tilde{\mathbf b}$ are those of
$\mathbf b$, moved with the level surfaces. kore's formula for `B0` and `B0_tor` gives $\tilde{\mathbf B}_0$: the
background field of the body is $\mathsf F\tilde{\mathbf B}_0/J$, the field of the sphere stretched with the level
surfaces.

**Equations.** The Lorentz force is written with covariant components, as the rest of the momentum equation, and the
induction equation for the flux:

$$
\mathsf F^{\mathsf T}\mathbf F_L = \frac{\mathrm{Hendrik}}{\rho_0 J}\left[\nabla\times(\mathsf M\tilde{\mathbf b})\times\tilde{\mathbf B}_0 + \nabla\times(\mathsf M\tilde{\mathbf B}_0)\times\tilde{\mathbf b}\right],
\qquad
\partial_t\tilde{\mathbf b} = \nabla\times\left(\frac{\mathbf w\times\tilde{\mathbf B}_0}{J}\right) + \mathrm{MagnetD}\thinspace\nabla^2\tilde{\mathbf b} .
$$

Two identities of the map give them. The current of the body, carried back as a flux, is $\nabla\times(\mathsf M\tilde{\mathbf b})$;
and the cross product of two fields carried back as fluxes, $\mathbf a = \mathsf F\tilde{\mathbf a}/J$ and
$\mathbf c = \mathsf F\tilde{\mathbf c}/J$, has the covariant components
$\mathsf F^{\mathsf T}(\mathbf a\times\mathbf c) = (\tilde{\mathbf a}\times\tilde{\mathbf c})/J$. The magnetic diffusion
keeps its spherical form ([Accuracy](#accuracy)).

**At first order.** With $\mathsf M - \mathsf I$ and $J - 1 = \eta P_2$ from above, and the currents
$\mathbf j = \nabla\times\tilde{\mathbf b}$ and $\mathbf J_0 = \nabla\times\tilde{\mathbf B}_0$ of the sphere, the force
and the electric field of the induction gain

$$
\delta\mathbf G = -\eta P_2\left(\mathbf J_0\times\tilde{\mathbf b} + \mathbf j\times\tilde{\mathbf B}_0\right)
+ \left[\nabla\times\left((\mathsf M - \mathsf I)\tilde{\mathbf B}_0\right)\right]\times\tilde{\mathbf b}
+ \left[\nabla\times\left((\mathsf M - \mathsf I)\tilde{\mathbf b}\right)\right]\times\tilde{\mathbf B}_0 ,
\qquad
\delta\mathbf E = -\eta P_2\thinspace\mathbf w\times\tilde{\mathbf B}_0 .
$$

A uniform flattening has $\eta = 0$, and only the two currents of the metric remain. A flattening that varies with the
radius gives the background field a current even where the spherical formula has none: the axial field, for one, is no
longer uniform in the body.

**Rows.** $\delta\mathbf G$ enters the momentum rows as the Lorentz force of the sphere does (times Hendrik), and
$\nabla\times\delta\mathbf E$ the rows of $f$ and $g$ as the induction does; both are built with the recurrence algebra
of `bin/mag_algebra.py`. A background of degrees up to $n$ couples a degree $l$ to $l \pm (n + 3)$ at most, two more
than in the sphere. The radial factors are products of the profile codes of the deformation and of the background: a
label carries the amplitude `E` ($\varepsilon$), `N` ($\eta$), `C` ($r\eta'$) or `Q` ($r(r\eta')'$), then `p` or `t`
and the degree of the poloidal or toroidal profile (`centrifugal.product_code`).

**Insulating walls.** Outside an insulating wall the field is a potential field $\nabla\phi$ of the body, and the wall
is a level surface, so the matching happens on the spheroid. In the reference space, at the wall $r = R$, the flux
through the wall and the tangential field are continuous:

$$
\tilde b_r = \left[J\thinspace\mathsf F^{-1}\nabla\phi(\mathbf X)\right]_r, \qquad
\left[\mathsf M\tilde{\mathbf b}\right]_{\rm tan} = \nabla_{\rm tan}\thinspace\phi(\mathbf X(\boldsymbol\alpha)) .
$$

At first order, with $m_s$ and $m_t$ the spheroidal and toroidal parts of $(\mathsf M - \mathsf I)\tilde{\mathbf b}$ at
the wall, $\delta s = (\boldsymbol\xi\cdot\nabla\phi)/R$ and $\delta q = -[\nabla\times(\boldsymbol\xi\times\nabla\phi)]_r$
(each of degree $l$), eliminating the potential gives the rows

$$
(l+1)F + RF' + R(m_s - \delta s) - \frac{R\thinspace\delta q}{l+1} = 0, \qquad G + m_t = 0 \qquad \text{(outer wall)},
$$

$$
lF - RF' - R(m_s - \delta s) - \frac{R\thinspace\delta q}{l} = 0, \qquad G + m_t = 0 \qquad \text{(insulating inner core)},
$$

which couple $F_l$ to $F_{l\pm2}$, $G_{l\pm1}$ and $G_{l\pm3}$, and $G_l$ to $G_{l\pm2}$, $F_{l\pm1}$ and $F_{l\pm3}$.

**Conducting inner core.** The inner core is deformed with the fluid, on the same map, and its field is a flux too. With
the permeability of the fluid (`mu_i2o` = 1) the matching needs no new terms: the flux through the inner boundary is
continuous, and with it the tangential $\mathsf M\tilde{\mathbf b}$ is continuous exactly when the tangential
$\tilde{\mathbf b}$ is; the rows of the electric field keep their form, since the flow vanishes at the no-slip boundary.

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

**With a field.** The magnetic energy of the deformed fluid is

$$
\tfrac12\int\tilde{\mathbf b}^*\cdot\mathsf M\thinspace\tilde{\mathbf b}\thinspace dV = \mathrm{ME} + \mathrm{MEc},
$$

with MEc the part of the metric (`magnetic.dat`, column 12), and `solve.py` normalises to $K$ plus Hendrik times it.
`spin_doctor` adds the metric parts of the Lorentz work ($\delta\mathbf G$) to residσ and residω and those of the
induction work ($\delta\mathbf E$) to resid𝐛 (`utils4pp.centrifugal_magnetic_terms`), so these close to round-off. The
full budget resid𝐄 also counts the Ohmic loss with the metric current $\nabla\times((\mathsf M - \mathsf I)\tilde{\mathbf b})$,
the energy of the potential field matched on each deformed wall, and the field of a conducting inner core with its
metric. It closes at second order in $f$ only: kore keeps the first order of $1/J$ and of $\mathsf M$ in the force, in
the induction and in the matching at the walls, and the energy does not cancel their products.

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
that of $(r, \theta)$. The field $\mathbf b = \mathsf F\tilde{\mathbf b}/J$ maps in the same way. A scalar, such as $s$,
only moves.

## Accuracy

The map, the transform of the flow and the covariant equations are exact. kore keeps them at first order in $f$, and
keeps some terms in the form they have in a sphere.

- **First order.** The error is of order $f^2$. For the spin-over of a Maclaurin spheroid it is $f^2/6$ in the frequency
  ([Checks](#checks)): 0.2 % at Saturn's flattening, where the deformation moves the frequency by 10 %.
  `submatrices.py` warns above $f = 0.1$.
- **Viscous force, thermal and magnetic diffusion.** Their metric parts are of relative size $f$, so an eigenvalue is
  off by about $f$ times the part of it that the diffusion makes: for a weakly damped mode, a fraction of order $f$ of
  its damping rate, and the same amount in frequency.
- **Differential rotation.** Its terms keep their spherical form: the terms of order $f\thinspace\Delta\Omega$ are left
  out.
- **What stops the run.** The rows of a stress-free wall change on a deformed wall, so kore stops with stress-free walls
  in a viscous run. The matching rows of a conducting inner core change too when its permeability differs from the
  fluid's, so kore stops unless `mu_i2o` = 1. It also stops with a forcing.

## Implementation

Where each piece is in the code:

| piece | code |
| --- | --- |
| the flattening and the profile codes | `centrifugal.radau`, `shape_functions`, `profile_functions` (codes `ce0`, `ce1`, `ce2` in `radial_profiles.proffdir`) |
| the angular couplings | `centrifugal.primitives`, `momentum_coefficients`, `heat_coefficients` |
| labels, blocks, assembly | `centrifugal.labels` (`submatrices.py`), `momentum_block`, `entropy_block` (`operators.centrifugal_inertia`, `centrifugal_coriolis`, `centrifugal_entropy`), `dl_range` (`assemble.centrifugal_blocks`) |
| the magnetic terms | `centrifugal.magnetic_build` with the product codes of `magnetic_profile_functions`, `magnetic_labels` (`submatrices.py`), `magnetic_block` (`operators.centrifugal_magnetic`), `magnetic_dl_range` (`assemble.py`) |
| the insulating walls | `centrifugal.wall_coefficients`, `wall_rows` (`assemble.bc_b_centrifugal`) |
| effective gravity | `centrifugal.g_monopole` in `radial_profiles.graviX` |
| stops and warnings | `background.check_centrifugal`, `centrifugal.report` (`submatrices.py`), the centre check of the flattening |
| energies | `utils4pp.centrifugal_terms` (`solve.py`, `spin_doctor.py`); with a field `centrifugal_magnetic_energy` (`solve.py`) and `centrifugal_magnetic_terms` (`spin_doctor.py`, with `centrifugal.wall_potential` and `amplitudes`) |
| plots | `centrifugal.to_physical` (`tools/plot_field.py`, the flow and the field) |

## Checks

The deformation was checked in four ways: its closed forms against a symbolic derivation from the definitions, its
couplings, blocks and wall rows against direct quadratures, its eigenvalues against exact results, and the energy
budgets. The scripts are in `tests/centrifugal/` (README; `run_all.sh`, about 7 minutes on 16 cores). Results of 2026-11-05:

| check | script | result |
| --- | --- | --- |
| every closed form above, from the map and $\mathbf w = J\mathsf F^{-1}\mathbf u$ in Cartesian components (sympy): the level surfaces, the exact identities of the covariant equations, the first-order terms, the plot map, the effective potential | `derive.py` | exact |
| the magnetic terms, from $\mathbf b = \mathsf F\tilde{\mathbf b}/J$ alone (sympy): the covariant Lorentz force and induction, their first-order parts $\delta\mathbf G$ and $\delta\mathbf E$, a potential field on the deformed wall; then kore's couplings against the projections of $\delta\mathbf G$ and $\delta\mathbf E$ on its rows, block by block, and the wall terms against closed forms | `derive_mag.py` | exact; blocks 1e-14 |
| the angular couplings against Gauss–Legendre projections of the closed forms, $m$ = 0–6, $l \le 36$ | `algebra_check.py` | 5e-14 |
| the rows of an insulating wall (outer wall and inner core, $m$ = 0, 1, 2, 5) against the exact matching of a potential field on the deformed wall, computed on the wall by quadrature | `bc_check.py` | of second order: the residual falls by 4.0 when $f$ is halved, by 2.0 with the rows of the sphere |
| the metric blocks against $(r, \theta)$ quadratures, in shells and full spheres; inertia and entropy Hermitian and positive, Coriolis anti-Hermitian | `bin/ru_insane.py` | 6e-14 |
| $f = 0$ gives the matrices of `centrifugal = 0`, also with a field (a conducting inner core; a poloidal and toroidal background); `centrifugal = 0` leaves the 26 runs of `tests/diff_rot/regress.sh` unchanged | `eig_check.py`, `regress.sh` | bitwise |
| the spin-over of a Maclaurin spheroid (inviscid, uniform $f$): $\mathrm{Im}\thinspace\lambda = 1 + f + f^2/3$, against the exact $2a^2/(a^2 + c^2) = 1 + f + f^2/2$ | `eig_check.py` | $f/3$ in $(\mathrm{Im}\thinspace\lambda - 1)/f - 1$ for $f = \pm10^{-3}, \pm10^{-2}$ |
| a sphere labelled as a deformed one ($f = c(1 - r^2)^2$, zero at the wall): 13 resolved modes move as $c^2$, with no part odd in $c$ | `eig_check.py` | odd part 1.6e-4 $c^2$ |
| the same with a field: the axial and the Malkus field carried by the relabelling, against the same physical field written on the sphere (`'general'` profiles); the shifts of the resolved modes (Em = 1e-4) | `eig_check.py` | agree to 1–2 % (axial) and 3–4 % (Malkus); the rest is the metric part of the magnetic diffusion, left out ([Accuracy](#accuracy)) |
| `'clairaut'` of a homogeneous body against `'uniform'` with $f = 5q/4$ | `eig_check.py` | 2e-13 |
| the effective gravity with $g = r$: $q = 0.1$ against `Beyonce` $(1 - 2q/3)$ | `eig_check.py` | below 1e-14 |
| a deformed, stratified ideal fluid has real frequencies | `eig_check.py` | $\lvert\mathrm{Re}\thinspace\lambda\rvert$ 2e-16 |
| `'clairaut'`: a homogeneous body (5/4), the polytrope of index 1 ($15/(2\pi^2)$), `'Wu 2005'` against Clairaut's second-order equation integrated from $\rho$; `'model S hydrostatic'` gives $f/q$ = 0.519 at its wall | `clairaut_check.py` | 3e-12; below 1e-9 |
| the budgets residσ, residω, resid𝑠 of `spin_doctor` (`solve.py`, 2 ranks; `'Wu 2005'` with an inner core, with and without a differential rotation, and a `'Boussinesq'` full sphere) | `budget_check.py` | below 7e-10, as without the deformation; 1e-3 to 3e-2 without the metric parts |
| the budgets with a field (`solve.py`, 2 ranks): the axial field, a conducting inner core, a full sphere, the Malkus field, a poloidal and toroidal background with a differential rotation, each at two flattenings $f$ and $f/2$; `'Wu 2005'` with an inner core | `budget_check.py` | residσ and residω below 1e-13, resid𝐛 below 3e-12 (2e-8 for `'Wu 2005'`, set by its resolution as in the sphere); resid𝐄 of second order (orders 1.7 to 2.0 from the two flattenings) |
| the free core nutation, the spin-over of a fixed mantle, with a field (Ek 1e-7, Em 1e-5, Le 3e-3, the axial field, a conducting inner core) at $f$ = 2.5e-3 and $f/2$, and without the field | `fcn.sh` | $\mathrm{Im}\thinspace\lambda$ moves by $f(1 + f/3)$, with the coefficient of $f^2$ 1/3 to 1e-3, up to 1e-4 of $f$ that the viscosity and the field add (in their spherical form, [Accuracy](#accuracy)); the field changes the shift by 3e-5 of $f$, N 200 against N 160 by 5e-6 |
