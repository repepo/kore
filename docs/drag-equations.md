# Drag at the outer wall equations

This page gives the condition of a drag at a stress-free outer wall ([Drag at the outer wall](drag.md)), the drag of the
topography, how the condition enters the rows of the wall, and the power and damping rate that `spin_doctor` reports.
They hold for any density and viscosity profile (`bin/topo.py`).

## The condition

The wall pulls on the fluid along it with a stress in proportion to the slip, the flow along the wall relative to the
mantle. The tangential viscous stress of the fluid at the outer wall ($u_r = 0$ there) then meets

$$
\rho_0\,\nu\,\mathrm{ViscosD}\;r\,\partial_r\!\left(\frac{\mathbf u_t}{r}\right)
= -\,\mathrm{Gaspard}\;\rho_w\,\mathsf K\,(\mathbf u_t - \mathbf v_m) \qquad (r = 1),
$$

with $\mathsf K$ the drag coefficient in units of $\rho_w\Omega r_0$ ($\rho_w$ the fluid's density at the wall), $\nu$
the viscosity profile of the structure, and $\mathbf v_m = \omega_M\,\hat{\mathbf z}\times\mathbf r$ the velocity of the
mantle (`rotdyn = 1`, 0 otherwise). $\mathsf K = 0$ is the stress-free wall and $\mathsf K\to\infty$ the no-slip one.
$\rho_0 = \rho_w$ cancels, and the condition is

$$
r\,\partial_r\!\left(\frac{\mathbf u_t}{r}\right) + q\,\mathsf K\,(\mathbf u_t - \mathbf v_m) = 0, \qquad
q = \frac{\mathrm{Gaspard}}{\mathrm{ViscosD}\,\nu(1)} .
$$

`kappa_cmb` is $\mathsf K$ times the unit tensor. For a dimensional drag of stress $\kappa\,\mathbf u_t$,
`kappa_cmb` $= \kappa/(\rho_w\Omega r_0)$.

## The drag of the topography

A flow of amplitudes $U_x$ (zonal) and $U_y$ (meridional) and frequency $\omega_0$ along a topography of spectrum $S(k)$
radiates internal waves of mean power per unit area

$$
P = \frac{\rho\,I_k}{2\lvert\omega_0\rvert}\int_0^{2\pi}\left(U_x^2\cos^2\alpha + U_y^2\sin^2\alpha\right) g\;d\alpha,
\qquad
g = \sqrt{\max\left[0,\ (\omega_0^2 - N^2)(f^2 - \omega_0^2) + \omega_0^2\,\tilde f^2\sin^2\alpha\right]},
$$

$$
f = 2\Omega\cos\theta, \qquad \tilde f = 2\Omega\sin\theta, \qquad
I_k = \int_0^\infty S\,k^2\,dk = \frac{h_\text{rms}^2\,k_0\,\Gamma(H - 1/2)}{4\sqrt\pi\,\Gamma(H)}, \qquad
k_0 = \frac{2\pi}{\lambda_0},
$$

with $\alpha$ the direction of a wave vector of the topography from east, $\theta$ the colatitude and $N$ the buoyancy
frequency at the wall (Rekier et al. 2025, eqs. B1 and B2). Their eq. (B1) as printed (version 4 of the preprint) lacks
a factor $4\pi^2$, lost between their eqs. (A19) and (10); with it, a sinusoidal ridge gives the classical
$\rho U^2h^2k\sqrt{N^2 - \omega^2}/4$ of Bell (1975), and kore keeps it. The power has no term that mixes the zonal and
the meridional flow, so the linear drag of the same mean power is a diagonal tensor on $(u_\theta, u_\phi)$, in units of
$\rho\Omega r_0$:

$$
\mathsf K_{\theta\theta} = \frac{I_k/r_0}{\lvert\omega_0/\Omega\rvert}\int_0^{2\pi}\sin^2\alpha\,\frac{g}{\Omega^2}\,d\alpha,
\qquad
\mathsf K_{\phi\phi} = \frac{I_k/r_0}{\lvert\omega_0/\Omega\rvert}\int_0^{2\pi}\cos^2\alpha\,\frac{g}{\Omega^2}\,d\alpha .
$$

The inputs are `topo_hrms` $= h_\text{rms}/r_0$, `topo_lambda0` $= \lambda_0/r_0$ and `topo_H` $= H$, with
$N/\Omega$ from the stratification, $\sqrt{\mathrm{Beyonce}\;g\,dS/dr}/\mathrm{Gaspard}$ at the wall (0 without buoyancy
or where it is unstable), or `topo_N`, and $\omega_0/\Omega$ = `topo_omega`, or $\lvert$`itau`$\rvert$/Gaspard
(eigenvalue problems), or $\lvert$`forcing_frequency`$\rvert$/Gaspard (forced runs). The integral over $\alpha$ is a
midpoint rule of 2048 points (`topo.kappa`). Where $g$ vanishes for every $\alpha$ no wave can be radiated and the drag
vanishes: near the poles when $N$ exceeds $\omega_0$.

## The rows

At the wall a degree $l$ of the flow slips along it as

$$
\mathbf u_t = s_l\,\nabla_1 Y_l + t_l\,\nabla_1 Y_l\times\hat{\mathbf r}, \qquad
s_l = P_l' + \frac{P_l}{r} + (\ln\rho_0)'P_l, \qquad t_l = T_l ,
$$

with $\nabla_1$ the gradient on the unit sphere ([Numerical method](numerical-method.md#unknowns)). The stress-free rows
of `assemble.bc_u_spherical` are $r^2$ (section `u`, the second row of the outer wall) and $r$ (section `v`, its first
row) times the parts of $r\,\partial_r(\mathbf u_t/r)$ on $\nabla_1 Y_l$ and on $\nabla_1 Y_l\times\hat{\mathbf r}$. The
drag adds the same parts of $q\,\mathsf K\,\mathbf u_t$, which couple the degree $l$ to $s$ and $t$ of every degree
$l'$:

$$
(\text{row } \texttt u)_l + r^2 q\sum_{l'}\left(A_{ss}^{ll'}s_{l'} + A_{st}^{ll'}t_{l'}\right) = 0, \qquad
(\text{row } \texttt v)_l + r\,q\sum_{l'}\left(A_{ts}^{ll'}s_{l'} + A_{tt}^{ll'}t_{l'}\right) = 0,
$$

$$
A_{xy}^{ll'} = \frac{\oint x_l^*\cdot\mathsf K\,y_{l'}\,dS}{\oint\lvert x_l\rvert^2dS}, \qquad
x, y \in \{\,s: \nabla_1 Y,\ \ t: \nabla_1 Y\times\hat{\mathbf r}\,\} ,
$$

on the unit sphere (`topo.couplings`, `topo.rows`). On a sphere a constant drag gives $A_{ss} = A_{tt}$ equal to
`kappa_cmb` and no coupling between degrees. The drag of the topography is even about the equator, so it couples degrees of the same
symmetry: $l$ to $l\pm2, \dots$ within $P$ and within $T$, and for $m \neq 0$ $P_l$ to $T_{l\pm1}, \dots$. The integrals
are Gauss–Legendre quadratures in $\cos\theta$ with $\max(4(l_\text{max} + 2), 2000)$ nodes, many since $\mathsf K$ has
kinks where $g$ starts to vanish. With the rotating mantle the toroidal flow of degree 1 slips by
$T_1 - r\,\omega_M$, which gives the rows a column of $\omega_M$
([Rotating solids equations](rotating-solids-equations.md#equations)).

## Power and damping rate

`spin_doctor` computes the power the drag dissipates from the flow at the wall (`topo.dissipation`),

$$
D_\text{topo} = 2\,\mathrm{Gaspard}\;\rho_w\oint(\mathbf u_t - \mathbf v_m)^*\cdot\mathsf K\,(\mathbf u_t - \mathbf v_m)\,dS ,
$$

in the units of its energies (the integral of $\rho_0\lvert\mathbf u\rvert^2$ of the complex amplitude, whose rate of
change is $2\sigma$ times it), and the damping rate it implies, $\delta\sigma_\text{topo} = -D_\text{topo}/(2E)$, with
$E$ the energy of the mode: the kinetic energy, plus `Hendrik` times the magnetic energy inside and outside the fluid
and that stored in the conducting walls, plus the energy of the rotating solids. With a stress-free wall and no drag,
both use $\mathsf K = 1$: per unit `kappa_cmb`.

**The power through the wall.** The viscous stress at the wall works on the fluid at the rate $W_\nu$ (`flow.dat`
column 25), and on the mantle at the rate $2\,\mathrm{Re}(\omega_M^*\Gamma^\nu_M)$ of its viscous torque. The condition
above makes their sum minus the drag's dissipation:

$$
D_\text{topo} = -W_\nu - 2\,\mathrm{Re}\left(\omega_M^*\,\Gamma^\nu_M\right) .
$$

Where the background field also ends in a current sheet at the wall, its push joins the drag in the condition
([Magnetic field equations](magnetic-equations.md#current-sheets-at-slipping-walls)), and $W_\nu$ holds the push's
power too, which then adds to the right-hand side. So the budgets need no new term: residσ and resid𝐄 already count the
power of the stress at the walls, and the energy budget of the rotating solids the power of their torques. $\delta\sigma_\text{topo}$ is the drag's share of the damping
rate, one term of that budget.

## On a deformed body

On a body flattened by its rotation (`centrifugal = 1`) the outer wall is the level surface $r = 1$ of the map that
carries the reference sphere into the body, and kore solves for the flux $\mathbf w = J\mathsf F^{-1}\mathbf u$ of the
reference space ([Centrifugal deformation equations](centrifugal-equations.md)). The drag acts on the slip along the
body's wall, per unit of its area. Carried back to the reference sphere, it becomes a drag on $\mathbf w$ with an
effective tensor, which the rows and the power above take in place of $\mathsf K$.

**The condition.** With $\mathsf F$ the gradient of the map, $J$ its determinant and $\mathbf N = \mathsf F^{-\mathsf T}
\hat{\mathbf r}$ the normal to the level surface (not normalised), the rows of a stress-free wall hold the tangential
components of $\mathsf F^{\mathsf T}\mathsf S_X\mathbf N$, with $\mathsf S_X$ the rate of strain of the body
([Dissipative terms](centrifugal-equations.md#dissipative-terms)). The drag's condition on the body's wall,
$2\mathsf S_X\hat{\mathbf n} + q\,\mathsf K\,\mathbf u_t = 0$, times $\lvert\mathbf N\rvert\,\mathsf F^{\mathsf T}$, is

$$
2\left[\mathsf F^{\mathsf T}\mathsf S_X\mathbf N\right]_t + q\,\mathsf K_\text{eff}\,\mathbf w_t = 0, \qquad
\mathsf K_\text{eff} = \frac{\lvert\mathbf N\rvert}{J}\,\mathsf F^{\mathsf T}\mathsf K\,\mathsf F .
$$

The map is axisymmetric, so $\mathsf K_\text{eff}$ is diagonal on $(w_\theta, w_\phi)$, as $\mathsf K$ is on
$(u_\theta, u_\phi)$:

$$
\mathsf K_\text{eff} = \mathrm{diag}\left(h_\theta\,\mathsf K_{\theta\theta}(x_n),\ h_\phi\,\mathsf K_{\phi\phi}(x_n)\right),
\qquad
h_\theta = \frac{\lvert\mathbf N\rvert\,\lvert\mathsf F\hat{\boldsymbol\theta}\rvert^2}{J}, \qquad
h_\phi = \frac{\lvert\mathbf N\rvert\,\lvert\mathsf F\hat{\boldsymbol\phi}\rvert^2}{J},
$$

with $x_n$ the cosine of the colatitude of the wall's normal. The drag of the topography takes there the local vertical,
$f = 2\Omega x_n$ and $\tilde f = 2\Omega\sqrt{1 - x_n^2}$, and the buoyancy frequency $N\lvert\mathbf N\rvert$: the
effective gravity and the gradient of the entropy across the wall both grow as $\lvert\mathbf N\rvert =
\lvert\nabla r\rvert$ ($N$ that of the reference sphere, `topo_N` when set). With $x = \cos\theta$ and $P_2 = (3x^2 -
1)/2$:

- **first order** (`'clairaut'`, `'uniform'`, `'user'`; $\varepsilon$ and $\eta$ the amplitudes of the map at the wall):
  $h_\theta = 1 + \varepsilon - (3\varepsilon + 2\eta)P_2$, $h_\phi = 1 - \varepsilon - (\varepsilon + 2\eta)P_2$,
  $\lvert\mathbf N\rvert = 1 - (\varepsilon + \eta)P_2$, $x_n = x - \tfrac32\varepsilon x(1 - x^2)$;
- **the exact spheroid** (semi-axes $a$, $a$, $c$ with $a^2c = 1$, $Q = a^2x^2 + c^2(1 - x^2)$):
  $h_\theta = aQ^{3/2}$, $h_\phi = a^3Q^{1/2}$, $\lvert\mathbf N\rvert = aQ^{1/2}$, $x_n = ax/Q^{1/2}$.

kore takes $\mathsf K$ at $x_n$ node by node, without expanding it about $x$, since the drag of the topography has kinks
(`topo.wall_metric`, `topo.wall_tensor`). $\mathsf K_\text{eff}$ varies with latitude even for a constant `kappa_cmb`,
so the couplings $A_{xy}^{ll'}$ join the degrees: $l$ to $l \pm 2$ and, for $m \neq 0$, $P_l$ to $T_{l\pm1}$ at first
order, every degree of the same symmetry for the exact spheroid.

**Power.** The area of the body's wall is $dS_X = J\lvert\mathbf N\rvert\,dS$ (Nanson's formula), so the same tensor
gives the power the drag dissipates on the body's wall,

$$
D_\text{topo} = 2\,\mathrm{Gaspard}\;\rho_w\oint\mathbf w_t^*\cdot\mathsf K_\text{eff}\,\mathbf w_t\,dS
$$

on the reference sphere, and $\delta\sigma_\text{topo} = -D_\text{topo}/(2E)$ with $E$ the energy of the deformed body
([Energy budgets](centrifugal-equations.md#energy-budgets)). kore stops with the rotating solids on a deformed body, so
$\mathbf v_m = 0$ here.

## Checks

`tests/drag` checks the drag against exact limits (a very stiff drag gives a no-slip wall; a weak drag adds the damping
that Dtopo/κ and δσtopo/κ predict), against equivalences (a constant drag tensor gives the constant drag; a uniform
rotation offset gives a faster frame), the conservation of angular momentum and energy with the rotating solids, and
the power through the wall. With the Earth's topography and no stratification, the drag of the topography damps the
spin-over mode as much as `kappa_cmb` = 4.5e-6, to within 0.3 %.

On a deformed body, `tests/drag/deformed_check.py` checks the drag against the exact decay of a rigid rotation and the
exact spin-over flow of a spheroid, at flattenings up to 0.3 (to $2\cdot10^{-8}$), the first-order shapes against the
exact spheroid (they differ at second order in the flattening), a sphere relabelled as deformed (its drag is the
sphere's), and the weak and stiff limits in a viscous spheroidal shell.

## Reference

- Rekier, J., Triana, S. A., Barik, A., Abdulah, A. & Kang, S. 2025, arXiv:2507.01671 (eqs. 20, 21, B1 and B2).
