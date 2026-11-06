# Centrifugal deformation equations

This page gives the mathematics behind `centrifugal = 1`, for readers who want to check the code or extend it: the
coordinates that follow the level surfaces, the equations in them and their rows, the magnetic field, the dissipative
terms, the differential rotation, the exact spheroid, the flattening of each shape, the effective gravity, the energy
budgets, the plots, the accuracy, and the checks. How to set up a run is on
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

with the dots the viscous force and the thermal diffusion, which take covariant forms too
([Dissipative terms](#dissipative-terms)). Three terms of the ideal fluid change: the inertia gains the metric
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
\partial_t\tilde{\mathbf b} = \nabla\times\left(\frac{\mathbf w\times\tilde{\mathbf B}_0}{J} - \mathrm{MagnetD}\thinspace\mathsf M\thinspace\nabla\times(\mathsf M\tilde{\mathbf b})\right) .
$$

Two identities of the map give them. The current of the body, carried back as a flux, is $\nabla\times(\mathsf M\tilde{\mathbf b})$;
and the cross product of two fields carried back as fluxes, $\mathbf a = \mathsf F\tilde{\mathbf a}/J$ and
$\mathbf c = \mathsf F\tilde{\mathbf c}/J$, has the covariant components
$\mathsf F^{\mathsf T}(\mathbf a\times\mathbf c) = (\tilde{\mathbf a}\times\tilde{\mathbf c})/J$. The electric field
of the diffusion is MagnetD times the current of the body, whose covariant components are
$\mathsf M\thinspace\nabla\times(\mathsf M\tilde{\mathbf b})$ ([Dissipative terms](#dissipative-terms)).

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
the permeability of the fluid (`mu_i2o` = 1) the field $\tilde{\mathbf b}$ is continuous at the inner boundary, since
$\mathbf b$ is and the map is smooth across it, so the rows that match the field keep their form. The tangential electric
field is continuous too. The flow vanishes at the no-slip boundary, so on each side this field is that of the diffusion,
and its row gains the metric part of the diffusion of the fluid and of the core ([Dissipative terms](#dissipative-terms)).

## Dissipative terms

The viscous force and the thermal and magnetic diffusion act in the body, so a damped mode feels the deformed walls and
the deformed profiles through them. kore carries each back to the reference space as it does the inertia: it keeps its
spherical form and gains a part of first order in $f$.

**Viscous force.** With $\mathbf V(\mathbf u) = \rho_0^{-1}\nabla\cdot(\rho_0\nu\thinspace\mathsf S)$ the force per
unit mass ([Anelastic equations](anelastic.md)), its covariant form is, up to a gradient that the curl rows remove,

$$
\mathsf F^{\mathsf T}\mathbf V\left(\frac{\mathsf F\mathbf w}{J}\right) = \mathbf V(\mathbf w) - \boldsymbol\xi\times\nabla\times\mathbf V(\mathbf w)
+ \mathbf V\left(\nabla\times(\boldsymbol\xi\times\mathbf w) - \boldsymbol\xi\thinspace\nabla\cdot\mathbf w\right)
- \xi_r\frac{d\nu}{dr}\thinspace\nabla\cdot\mathsf S(\mathbf w) - \nabla\left(\frac{\xi_r}{\rho_0}\frac{d(\rho_0\nu)}{dr}\right)\cdot\mathsf S(\mathbf w) .
$$

The second and third terms carry the force of the sphere along the displacement $\boldsymbol\xi$; the last two come from
$\nu$ and $\rho_0$, which are constant on the level surfaces rather than on the spheres.

**Thermal diffusion.** The heat rows are the equation times $J$, and the diffusion becomes

$$
J\thinspace\nabla_X\cdot\left(\kappa p_0\nabla_X s\right) = \nabla\cdot\left(\kappa p_0\thinspace\mathsf M^{-1}\nabla s\right),
$$

whose first-order part is $-\nabla\cdot(\kappa p_0(\mathsf M - \mathsf I)\nabla s)$.

**Magnetic diffusion.** The current of the body, carried back as a flux, is $\nabla\times(\mathsf M\tilde{\mathbf b})$,
and the covariant components of the electric field of the diffusion are MagnetD times
$\mathsf M\thinspace\nabla\times(\mathsf M\tilde{\mathbf b})$. Its first-order part is MagnetD times
$(\mathsf M - \mathsf I)\nabla\times\tilde{\mathbf b} + \nabla\times((\mathsf M - \mathsf I)\tilde{\mathbf b})$. A
conducting inner core gains the same terms with its own diffusivity.

**Walls.** The no-slip, impenetrable and fixed-entropy rows keep their form: $\mathbf w = 0$, $w_r = 0$ and $s = 0$ on a
level surface are $\mathbf u = 0$, $\mathbf u\cdot\mathbf n = 0$ and $s = 0$ in the body. The two other walls change:

- a stress-free wall has no tangential stress on the deformed wall: the tangential part of
  $\mathsf F^{\mathsf T}\mathsf S_X\mathsf F^{-\mathsf T}\hat{\mathbf r}$ vanishes, with $\mathsf S_X$ the rate of strain
  of the body, $\mathsf F^{-\mathsf T}\hat{\mathbf r}$ the normal to the level surface and $\mathsf F^{\mathsf T}$ taking
  the components along its tangents;
- a fixed-flux wall has no flux through the deformed wall: $\hat{\mathbf r}\cdot\mathsf M^{-1}\nabla s = 0$.

**Rows.** The first-order parts multiply the fields and their derivatives by polynomials in $\cos\theta$ of degree 2 at
most, so they couple the degrees as the inertia does: $l$ to $l$ and $l \pm 2$ of the same kind, and to $l \pm 1$ and
$l \pm 3$ of the other. Their radial factors are products of the amplitudes of the map, of the diffusivity and its
derivatives, and of the derivatives of $\ln\rho_0$ (`centrifugal.diffusion_build`). When $m \neq 0$ the toroidal rows of
the viscous force and of the induction reach the third derivative of $P$ and of $F$, and kore builds those sections in the
basis that has it. In a full sphere these rows have terms in $P/r$ and $F/r$, polynomials only once the rows are
multiplied by $r^4$ and $r^3$: kore does it for the field (in a conducting inner core too), and asks for `rpower_v` = 4
for the flow.

## Differential rotation

A differential rotation is a profile like the others: $\Delta\Omega(r, \theta)$ is given in the reference space, so the
point of the body at $\mathbf X(r, \theta)$ turns at $\Delta\Omega(r, \theta)$, and a rotation that depends on $r$ alone
is constant on the level surfaces. The flow of the body is $\mathbf U_0 = \mathrm{Gaspard}\thinspace\Delta\Omega\thinspace\hat{\mathbf z}\times\mathbf X$.
The map commutes with the rotations about the axis, so the flux of this flow is $J$ times the spherical formula
$\mathbf U_{00} = \mathrm{Gaspard}\thinspace\Delta\Omega\thinspace\hat{\mathbf z}\times\boldsymbol\alpha$, and its terms
change as follows.

- **Momentum.** The force $(\mathbf U_0\cdot\nabla)\mathbf u + (\mathbf u\cdot\nabla)\mathbf U_0$ gains, up to a
  gradient, with $\boldsymbol\omega_0 = \nabla\times\mathbf U_{00}$,

  $$
  \delta\mathbf W = -\mathbf U_{00}\times\nabla\times\left((\mathsf M - \mathsf I)\thinspace\mathbf w\right)
  + (\varepsilon + \eta P_2)\thinspace\mathbf w\times\boldsymbol\omega_0
  - \eta\thinspace\mathrm{Gaspard}\thinspace\Delta\Omega\left(\hat{\mathbf z}\times\mathbf w - \cos\theta\thinspace\hat{\mathbf r}\times\mathbf w\right).
  $$

- **Heat.** The advection of the entropy times $J$ is $(1 + \eta P_2)\thinspace\mathbf U_{00}\cdot\nabla s$: the heat
  rows gain $i m\thinspace\mathrm{Gaspard}\thinspace\eta P_2\thinspace\Delta\Omega\thinspace s$.
- **Induction.** The electric field of the flow carries back exactly,
  $\mathsf F^{\mathsf T}(\mathbf U_0\times\mathbf b) = \mathbf U_{00}\times\tilde{\mathbf b}$: no new term.

A degree $n$ of $\Delta\Omega$ couples $l$ to $l \pm (n + 2)$ of the same kind and to $l \pm (n + 3)$ of the other, two
more than in the sphere (`centrifugal.rotation_build`).

## The exact spheroid

When every level surface has the same flattening, the body is a stretched sphere, and kore keeps the map exactly, at any
flattening: `centrifugal_shape = 'spheroid'`. It serves a homogeneous fluid in a strongly flattened body or container,
such as a fast-rotating planet with a rigid mantle (or, with $f < 0$, an elongated one), and it can be checked against
exact solutions at any $f$.

**The map.** The point $\boldsymbol\alpha$ of the reference space is at

$$
\mathbf X = \mathsf F\boldsymbol\alpha, \qquad \mathsf F = \mathrm{diag}(a, a, c), \qquad \frac{c}{a} = 1 - f, \qquad a^2c = 1,
$$

so the level surface $r$ is the spheroid of equatorial radius $ar$ and polar radius $cr$, with the volume of the sphere
of radius $r$. At first order in $f$ this is the map of `'uniform'`, $a = 1 + f/3$ and $c = 1 - 2f/3$. The metric is
constant, $J = 1$, $\mathsf M = \mathsf F^{\mathsf T}\mathsf F = \mathrm{diag}(a^2, a^2, c^2)$ and
$\tilde{\mathbf z} = \hat{\mathbf z}/c$, and the Laplacian of the body, written in the reference space, is

$$
D = \nabla\cdot\left(\mathsf M^{-1}\nabla\right) = \frac{1}{a^2}\nabla^2 + \left(\frac{1}{c^2} - \frac{1}{a^2}\right)\partial_z^2 .
$$

**The terms.** Each term that changes gains a part that kore keeps whole. With
$\mathsf Z\mathbf v = \hat{\mathbf z}(\hat{\mathbf z}\cdot\mathbf v)$, $\partial_z = \hat{\mathbf z}\cdot\nabla$ (on
each Cartesian component of a vector) and $x = \cos\theta$, the parts beyond the sphere's are

| term | its part beyond the sphere's |
| --- | --- |
| inertia | $(\mathsf M - \mathsf I)\thinspace\mathbf w = (a^2 - 1)\thinspace\mathbf w + (c^2 - a^2)\thinspace\mathsf Z\mathbf w$ |
| Coriolis force | $(\tilde{\mathbf z} - \hat{\mathbf z})\times\mathbf w = (1/c - 1)\thinspace\hat{\mathbf z}\times\mathbf w$ |
| viscous force | $\nu\left[(a^2/c^2 - 1)\thinspace\partial_z^2\mathbf w + (c^2 - a^2)\thinspace\mathsf Z D\mathbf w\right]$ |
| thermal diffusion | $\kappa p_0\left[(1/a^2 - 1)\thinspace\nabla^2 s + (1/c^2 - 1/a^2)\thinspace\partial_z^2 s\right]$ |
| traction on a level surface | $(a^2/c^2 - 1)\thinspace x\thinspace\partial_z\mathbf w + (c^2 - a^2)\thinspace\mathsf Z\left[\partial_r\mathbf w/a^2 + (1/c^2 - 1/a^2)\thinspace x\thinspace\partial_z\mathbf w\right]$ |
| flux through a level surface | $(1/a^2 - 1)\thinspace\partial_r s + (1/c^2 - 1/a^2)\thinspace x\thinspace\partial_z s$ |

The traction is that of the stress-free wall ([Dissipative terms](#dissipative-terms)),
$\mathsf F^{\mathsf T}\mathsf S_X\mathsf F^{-\mathsf T}\hat{\mathbf r} = \mathsf M(\nabla\mathbf w)\mathsf M^{-1}\hat{\mathbf r} + (\nabla\mathbf w)^{\mathsf T}\hat{\mathbf r}$,
with $(\nabla\mathbf w)_{ik} = \partial_k w_i$, less the sphere's. The rest keeps its spherical form: the time
derivative of the entropy and its advection ($J = 1$), the pressure, and the buoyancy. With the gravity $g = r$ of
`'Boussinesq'`, the buoyancy of the reference space acts in the body along $\mathsf F^{-\mathsf T}\boldsymbol\alpha = \nabla_X(r^2/2)$:
normal to the level surfaces, as the effective gravity of a homogeneous body in hydrostatic equilibrium (a Maclaurin
spheroid), whose effective potential depends on $r$ alone.

**Rows.** $\mathsf Z$ multiplies by polynomials in $\cos\theta$ of degree 2 and $\partial_z$ raises or lowers the degree by
one, so the inertia couples a degree $l$ to $l$ and $l \pm 2$ of the same kind and to $l \pm 1$ of the other, the viscous
force and the stress-free rows reach $l \pm 4$ and $l \pm 3$, and the heat rows $l \pm 2$. The radial operators are kore's
own, with no profile (`centrifugal.spheroid_build`, `spheroid_walls`), and the blocks reach $\lvert\Delta l\rvert$ = 4,
one more than at first order.

**Scope.** The fluid is homogeneous (`structure = 'Boussinesq'`, `diffusivity = 'uniform'`), with `centrifugal_q` = 0 (the
effective gravity is already kore's $g$), and any $f < 1$. kore stops with a magnetic field or a differential rotation.
The budgets of `spin_doctor` and the plots use the exact metric and map.

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
imaginary: it still does no work. `solve.py` normalises the eigenvectors to this $K$. `spin_doctor` adds the metric parts
of the viscous power, of the power of a differential rotation and of the entropy diffusion in the same way
(`utils4pp.centrifugal_dissipation_terms`), so the budgets close as well as without the deformation. It computes neither
the enstrophy residual nor the viscous one, whose wall terms keep their spherical form (residens and residD are NaN).

**With a field.** The magnetic energy of the deformed fluid is

$$
\tfrac12\int\tilde{\mathbf b}^*\cdot\mathsf M\thinspace\tilde{\mathbf b}\thinspace dV = \mathrm{ME} + \mathrm{MEc},
$$

with MEc the part of the metric (`magnetic.dat`, column 12), and `solve.py` normalises to $K$ plus Hendrik times it.
`spin_doctor` adds the metric parts of the Lorentz work ($\delta\mathbf G$) to residσ and residω, and those of the
induction work ($\delta\mathbf E$) and of the magnetic diffusion to resid𝐛 (`utils4pp.centrifugal_magnetic_terms`), so
these close to round-off. The
full budget resid𝐄 also counts the Ohmic loss with the metric current $\nabla\times((\mathsf M - \mathsf I)\tilde{\mathbf b})$,
the energy of the potential field matched on each deformed wall, and the field of a conducting inner core with its
metric. It closes at second order in $f$ only: kore keeps the first order of $1/J$ and of $\mathsf M$ in the force, in
the induction, in the diffusion and in the matching at the walls, and the energy does not cancel their products.

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

The map, the transform of the flow and the covariant equations are exact. kore keeps them at first order in $f$, except
for the exact spheroid ([The exact spheroid](#the-exact-spheroid)), where the only error is that of the discretization.

- **First order.** The error is of order $f^2$. For the spin-over of a Maclaurin spheroid it is $f^2/6$ in the frequency
  ([Checks](#checks)): 0.2 % at Saturn's flattening, where the deformation moves the frequency by 10 %.
  `submatrices.py` warns above $f = 0.1$.
- **Thermal wind.** With `thermal_wind` 1 or 2 the entropy $S_{tw}$ balances $\Delta\Omega$ as in the sphere: the terms
  of order $f\thinspace\Delta\Omega$ of that balance are missing, while its advection and the other terms of
  $\Delta\Omega$ are those of the deformed body. `submatrices.py` warns.
- **What stops the run.** The rows of a wall of zero radial stress (`bco` or `bci` = 2) would change on a deformed wall,
  so kore stops with them in a viscous run. The matching rows of a conducting inner core change too when its permeability
  differs from the fluid's, so kore stops unless `mu_i2o` = 1. It also stops with a forcing, and in a viscous full
  sphere with $m \neq 0$ unless `rpower_v` = 4 ([Dissipative terms](#dissipative-terms)).
- **Shifts measured from a sphere.** A body that is not deformed keeps the bases and rows of the sphere
  (`centrifugal.deformed`), so that its matrices are those of `centrifugal = 0`. A deformed body with $m \neq 0$ writes
  the rows of the toroidal flow in $C^{(3)}$ when it is viscous, and those of the toroidal field when the field diffuses
  ([Dissipative terms](#dissipative-terms)). A shift taken between a deformed run and one at $f = 0$ therefore also
  holds the difference of their errors of discretization, which near the resolution limit can rival the part of the
  shift that the Ekman layers make: for the nutation of [Checks](#checks) without its field, 1.5e-6 in $\lambda$ at
  N 160, 6e-4 of $f$, as large as that part. Taken between deformed runs, at $\pm f$, the shift does not carry it: it
  moves by 1e-6 of $f$ from N 160 to N 200, 5e-6 with the field.

## Implementation

Where each piece is in the code:

| piece | code |
| --- | --- |
| the flattening and the profile codes | `centrifugal.radau`, `shape_functions`, `profile_functions` (codes `ce0`, `ce1`, `ce2` in `radial_profiles.proffdir`) |
| the angular couplings | `centrifugal.primitives`, `momentum_coefficients`, `heat_coefficients` |
| labels, blocks, assembly | `centrifugal.labels` (`submatrices.py`), `momentum_block`, `entropy_block` (`operators.centrifugal_inertia`, `centrifugal_coriolis`, `centrifugal_entropy`), `dl_range` (`assemble.centrifugal_blocks`) |
| the magnetic terms | `centrifugal.magnetic_build` with the product codes of `magnetic_profile_functions`, `magnetic_labels` (`submatrices.py`), `magnetic_block` (`operators.centrifugal_magnetic`), `magnetic_dl_range` (`assemble.py`) |
| the insulating walls | `centrifugal.wall_coefficients`, `wall_rows` (`assemble.bc_b_centrifugal`) |
| the dissipative terms | `centrifugal.diffusion_build` (`viscous_force`, `thermal_diffusion_metric`, `magnetic_diffusion_metric`) with the product codes of `diffusion_profile_functions`, `diffusion_labels` (`submatrices.py`, with the bases of `diffusion_orders`), `diffusion_block` (`operators.centrifugal_diffusion`) |
| the stress-free and fixed-flux walls | `centrifugal.wall_dissipative`, `wall_dissipative_rows` (`assemble.bc_u_centrifugal`, `bc_theta_centrifugal`) |
| a conducting inner core | `centrifugal.ic_labels`, `diffusion_block_ic` (`operators.centrifugal_diffusion_ic`), `icb_electric`, `icb_rows` (`assemble.bc_b_cic_centrifugal`); the rows $r^3$ of `utils.rpower_g`, `rpower_gic` |
| the differential rotation | `centrifugal.rotation_force`, `rotation_build`, `rotation_heat_coefficients`, `rotation_labels`, `rotation_block` (`operators.centrifugal_rotation`) |
| effective gravity | `centrifugal.g_monopole` in `radial_profiles.graviX` |
| stops and warnings | `background.check_centrifugal`, `centrifugal.report` (`submatrices.py`), the centre check of the flattening; `centrifugal.deformed` (without a deformation kore keeps the bases and rows of the sphere) |
| energies | `utils4pp.centrifugal_terms` (`solve.py`, `spin_doctor.py`) and `centrifugal_dissipation_terms` (`spin_doctor.py`); with a field `centrifugal_magnetic_energy` (`solve.py`) and `centrifugal_magnetic_terms` (`spin_doctor.py`, with `centrifugal.wall_potential` and `amplitudes`) |
| plots | `centrifugal.to_physical` (`tools/plot_field.py`, the flow and the field) |
| the exact spheroid | `centrifugal.exact`, `spheroid_axes`, `spheroid_terms`, `spheroid_build`, `spheroid_walls`, `spheroid_block` (through `momentum_block` and `diffusion_block`), `spheroid_primitives` (`utils4pp.centrifugal_terms`) |

## Checks

The deformation was checked in four ways: its closed forms against a symbolic derivation from the definitions, its
couplings, blocks and wall rows against direct quadratures, its eigenvalues against exact results (among them the modes
of a spheroid, computed without the map), and the energy budgets. The scripts are in `tests/centrifugal/` (README;
`run_all.sh`, about 14 minutes on 16 cores). Results of 2026-11-06:

| check | script | result |
| --- | --- | --- |
| every closed form above, from the map and $\mathbf w = J\mathsf F^{-1}\mathbf u$ in Cartesian components (sympy): the level surfaces, the exact identities of the covariant equations, the first-order terms, the plot map, the effective potential | `derive.py` | exact |
| the magnetic terms, from $\mathbf b = \mathsf F\tilde{\mathbf b}/J$ alone (sympy): the covariant Lorentz force and induction, their first-order parts $\delta\mathbf G$ and $\delta\mathbf E$, a potential field on the deformed wall; then kore's couplings against the projections of $\delta\mathbf G$ and $\delta\mathbf E$ on its rows, block by block, and the wall terms against closed forms | `derive_mag.py` | exact; blocks 1e-14 |
| the dissipative terms and the differential rotation, from the map alone (sympy): the covariant viscous force, the stress on the level surface, the thermal diffusion and the flux through the level surface, the magnetic diffusion, the force of a differential rotation; then kore's couplings, the stress-free and fixed-flux rows and the row of the electric field at a conducting inner core against their projections on kore's rows | `derive_diff.py` | exact; 239 blocks to 1e-13, the wall rows to 3e-13 |
| the angular couplings against Gauss–Legendre projections of the closed forms, $m$ = 0–6, $l \le 36$ | `algebra_check.py` | 5e-14 |
| the rows of an insulating wall (outer wall and inner core, $m$ = 0, 1, 2, 5) against the exact matching of a potential field on the deformed wall, computed on the wall by quadrature | `bc_check.py` | of second order: the residual falls by 4.0 when $f$ is halved, by 2.0 with the rows of the sphere |
| the metric blocks against $(r, \theta)$ quadratures, in shells and full spheres; inertia and entropy Hermitian and positive, Coriolis anti-Hermitian, the viscous force Hermitian and negative | `bin/ru_insane.py` | 6e-14; the viscous force 6e-11 with the profiles of `'Wu 2005'`, as in the sphere |
| $f = 0$ gives the matrices of `centrifugal = 0`, also with a field (a conducting inner core; a poloidal and toroidal background); `centrifugal = 0` leaves the 26 runs of `tests/diff_rot/regress.sh` unchanged | `eig_check.py`, `regress.sh` | bitwise |
| the spin-over of a Maclaurin spheroid (inviscid, uniform $f$): $\mathrm{Im}\thinspace\lambda = 1 + f + f^2/3$, against the exact $2a^2/(a^2 + c^2) = 1 + f + f^2/2$ | `eig_check.py` | $f/3$ in $(\mathrm{Im}\thinspace\lambda - 1)/f - 1$ for $f = \pm10^{-3}, \pm10^{-2}$ |
| a sphere labelled as a deformed one ($f = c(1 - r^2)^2$, zero at the wall): 13 resolved modes move as $c^2$, with no part odd in $c$ | `eig_check.py` | odd part 1.6e-4 $c^2$ |
| the same with a field, in a viscous fluid (Ek = Em = 1e-3): the axial and the Malkus field carried by the relabelling, against the same physical field written on the sphere (`'general'` profiles); the parts of the shifts odd in $c$ | `eig_check.py` | 4e-6 (axial) and 3e-5 (Malkus); 3 % to 60 % without the metric part of the magnetic diffusion, 4 % up to 160 times the shift without that of the viscous force |
| the free decay of a field in a sphere labelled as deformed inside: a full sphere, and a conducting inner core of the fluid's conductivity (the relabelling moves its boundary, which a uniform conductor does not see) and of another conductivity | `eig_check.py` | their part odd in $c$ is 2e-13 to 4e-11 of $\lambda$ at $c$ = 1e-3, of order $c^3$ |
| the inertial modes of an inviscid homogeneous fluid in a spheroid, computed exactly without the map (the polynomial method, in Cartesian components): $d\omega/df$ of the 73 modes of degree $\le 7$ with $m$ = 0, 1, 2 against kore's | `spheroid_modes.py` | 1.4e-11 |
| damped modes when the walls are deformed (Ek = 1e-3): kore's $d\lambda/df$ against the exact first-order shift computed from the sphere's mode alone (Hadamard's formula and a volume formula), with no-slip and stress-free walls, fixed entropy and fixed flux, in full spheres and shells | `shape_check.py` | 1e-10 to 3e-8; kore is 5e-2 off without the metric viscous terms |
| the exact spheroid ($f$ = 0.3 and −0.2): its terms from the map (sympy); kore's couplings and wall rows against their projections on kore's rows; 176 inertial modes (degree $\le 7$, $m$ = 0–3) against the polynomial method; 20 viscous modes at Ek 0.1 (no-slip and stress-free) and 16 diffusion modes against exact Galerkin solutions (polynomial fields, rational integrals); the rigid rotation of a stress-free shell | `spheroid_check.py` | exact; 448 blocks to 4e-14, the wall rows to 1e-13; the modes to 2e-12, 6e-13 and 2e-13; $\lvert\lambda\rvert$ 2e-18 |
| `'clairaut'` of a homogeneous body against `'uniform'` with $f = 5q/4$ | `eig_check.py` | 2e-13 |
| the effective gravity with $g = r$: $q = 0.1$ against `Beyonce` $(1 - 2q/3)$ | `eig_check.py` | below 1e-14 |
| a deformed, stratified ideal fluid has real frequencies | `eig_check.py` | $\lvert\mathrm{Re}\thinspace\lambda\rvert$ 2e-16 |
| `'clairaut'`: a homogeneous body (5/4), the polytrope of index 1 ($15/(2\pi^2)$), `'Wu 2005'` against Clairaut's second-order equation integrated from $\rho$; `'model S hydrostatic'` gives $f/q$ = 0.519 at its wall | `clairaut_check.py` | 3e-12; below 1e-9 |
| the budgets residσ, residω, resid𝑠 of `spin_doctor` (`solve.py`, 2 ranks; `'Wu 2005'` with an inner core, with and without a differential rotation, and a `'Boussinesq'` full sphere) | `budget_check.py` | below 1e-9, as in the sphere (6e-10); 1e-3 to 3e-2 without the metric weights |
| the budgets with a field (`solve.py`, 2 ranks): the axial field, a conducting inner core, a full sphere, the Malkus field, a poloidal and toroidal background with a differential rotation, each at two flattenings $f$ and $f/2$; `'Wu 2005'` with an inner core | `budget_check.py` | residσ and residω below 2e-13, resid𝐛 below 1e-12 (2e-8 for `'Wu 2005'`, set by its resolution as in the sphere); resid𝐄 of second order (orders 1.7 to 2.2 from the two flattenings) |
| the free core nutation, the spin-over of a fixed mantle, with a field (Ek 1e-7, Em 1e-5, Le 3e-3, the axial field, a conducting inner core) at $f$ = ±2.5e-3 and $f/2$, and without the field; the shift is taken between the deformed runs ([Accuracy](#accuracy)) | `fcn.sh` | $\mathrm{Im}\thinspace\lambda$ moves by $f(1 + f/3)$, with the coefficient of $f^2$ 1/3 to 7e-4, plus 5.8e-4 of $f$ from the Ekman layers (Greenspan's boundary-layer theory gives 4.6e-4, which kore approaches slowly); the field changes the shift by 3.6e-5 of $f$, N 200 against N 160 by 5e-6 |
