# Rotating solids equations

This page gives the equations of the rotating solids (`rotdyn = 1`, [Rotating solids](rotating-solids.md)), the torques
that couple them to the fluid, and the angular momentum balance that `spin_doctor` checks (resid𝐋). They hold for any density profile and any poloidal background field (`bin/rotdyn.py`).

## Equations

The mantle and the inner core turn about the rotation axis at the rates $\omega_M$ and $\omega_{IC}$ relative to the
frame, and $\varphi$ is the angle of the inner core relative to the mantle. Each solid's angular momentum changes by the
torques on it. For a mode $e^{\lambda t}$

$$
\lambda I_M\,\omega_M = \Gamma_M + K\varphi, \qquad
\lambda I_{IC}\,\omega_{IC} = \Gamma_{IC} - K\varphi, \qquad
\lambda\varphi = \omega_{IC} - \omega_M - c\,\varphi ,
$$

with $I_M$, $I_{IC}$ = `MoIZ_M`, `MoIZ_IC`, $K$ = `gTorque` Gaspard² and $c$ = Gaspard/`OmgtauIC` in the units of the
run. $\Gamma_M$ and $\Gamma_{IC}$ are the torques of the fluid on the mantle and on the inner core (below): the viscous
ones times `vtrq`, the magnetic ones times `mtrq`. These are the last three rows of $A$ and $B$, after all the fields
(`utils.offset['rot']`).

**The walls.** The fluid follows the solids through the no-slip walls. A rigid rotation $\omega$ about the axis is the
toroidal flow of degree 1 with $T = \omega r$ ($u_\phi = T\sin\theta$, kore's $Y_1^0$ being $\cos\theta$), so the
no-slip rows of that degree become

$$
T(1) = \omega_M, \qquad T(r_i) = r_i\,\omega_{IC} ,
$$

and every other wall condition is unchanged. At a stress-free outer wall (`bco = 0`) the row of $T_1$ stays the
stress-free one, and a drag acts there on the slip relative to the mantle, $T_1(1) - \omega_M$
([Drag at the outer wall equations](drag-equations.md#the-rows)); without a drag only the field couples the fluid to the
mantle. A rigid rotation carries an axisymmetric field without changing it, so the rotating solids leave the conditions
on the field, and the field of a conducting inner core or mantle, as they are for solids at rest, but for a fluid
slipping along a conducting outer wall, whose electric field enters the wall through the slip relative to the mantle
(`assemble.bc_b_slip`).

## Torques

The torques are the moments of the stresses on each wall. With $T$ the toroidal flow of degree 1, the axial angular
momentum of the fluid is

$$
L = \int\rho_0\,s\,u_\phi\,dV = \frac{8\pi}{3}\int_{r_i}^{1}\rho_0\,r^3\,T\,dr .
$$

**Viscous.** The shear stress on a sphere is $\sigma_{r\phi} = c_\nu\,(T' - T/r)\sin\theta$ with
$c_\nu = \rho_0\,\nu\,$`ViscosD`, so that

$$
\Gamma^\nu_M = -\frac{8\pi}{3}\,c_\nu\,r^2\left(rT' - T\right)\Big|_{r=1}, \qquad
\Gamma^\nu_{IC} = \frac{8\pi}{3}\,c_\nu\,r^2\left(rT' - T\right)\Big|_{r=r_i} .
$$

**Magnetic.** The magnetic stress on the fluid's side of each wall, with $B_{0\phi} = 0$ there:

$$
\Gamma^B_M = -\mathrm{Hendrik}\oint s\,B_{0r}\,b_\phi\,dS\,\Big|_{r=1}, \qquad
\Gamma^B_{IC} = \mathrm{Hendrik}\oint s\,B_{0r}\,b_\phi\,dS\,\Big|_{r=r_i} .
$$

It acts on everything beyond the wall: a thin conducting layer, a resolved mantle or a conducting inner core (at an insulating wall
$b_\phi = 0$). With $b_\phi = -\sum_l g_l\,\partial_\theta Y_l$ and $B_{0r} = \sum_n n(n+1)\,h_n Y_n/r$ (the degrees $n$
of the poloidal background, `cnorm` included),

$$
\oint s\,B_{0r}\,b_\phi\,dS = \sum_l C_l\,g_l, \qquad
C_l = 2\pi r^2\sum_n n(n+1)\,h_n\,\frac{2l(l+1)}{(2l+1)(2n+1)}\left(\delta_{n,l-1} - \delta_{n,l+1}\right),
$$

which is $\tfrac{16\pi}{5}\,r^2 h_1 g_2$ for a dipole.

**Gravitational.** $K\varphi$ on the mantle and $-K\varphi$ on the inner core, which also relaxes towards alignment at
the rate $c$.

## The angular momentum balance: resid𝐋

The fluid's angular momentum changes only through the torques on it. `spin_doctor` computes them from the fields, apart
from the matrices, and checks

$$
\lambda L = -\Gamma_M - \Gamma_{IC} ,
$$

with the viscous torques from $T$ at the walls and the magnetic ones from the stress
$\mathrm{Hendrik}\,(B_{0r}\,\mathbf b + b_r\,\mathbf B_0)$ on a grid in $\theta$ at each wall, for any poloidal and
toroidal background (`rotdyn.viscous_torques`, `wall_torques`), whatever `vtrq` and `mtrq`. Both are taken on the
fluid's side of the walls: where $\mathbf B_0$ ends at a wall in a current sheet
([Current sheets at slipping walls](magnetic-equations.md#current-sheets-at-slipping-walls)), the magnetic one holds
the torque on the sheet's currents, which the viscous one balances at a slipping wall. The other forces exert no
net axial torque: the pressure is normal to the walls, the buoyancy radial, and for $m = 0$ neither the Coriolis force
nor the advection by a differential rotation $\Delta\Omega$ has one, since
$\int\rho_0\,\mathbf u\cdot\nabla(s^2\Delta\Omega)\,dV = 0$ between impenetrable walls.

**Equatorial.** For $m = 1$, `symm = -1` the same balance holds for the equatorial angular momentum, with

$$
L = \frac{16\pi}{3}\int_{r_i}^{1}\rho_0\,r^3\,T\,dr = \frac{L_x - iL_y}{N} ,
$$

$N = -1/\sqrt2$ the normalisation of kore's $Y_1^1 = N\sin\theta\,e^{i\phi}$, the torques in the same combination (the
viscous ones with $16\pi/3$ for $8\pi/3$), and the Coriolis torque $-\boldsymbol\Omega\times\mathbf L$, which is
$i\,\mathrm{Gaspard}\,L$:

$$
\lambda L = -\Gamma_M - \Gamma_{IC} + i\,\mathrm{Gaspard}\,L .
$$

A differential rotation would add a torque here, so this case needs `diff_rot = 0`.

**Forced runs.** In a forced run $\lambda = i\,w$, $w$ the forcing frequency, and the axial balance holds as above where
the forcing moves the walls only along themselves and puts no force in the fluid: the libration (forcing 7 with $m$ =
0), the steady rotation of the walls (forcing 12, where $\lambda = 0$ and the torques on the two solids cancel) and the
thermal forcing (forcing 13). A flow through a wall would add the Coriolis torque of the mass it carries across, and a
body force its own torque. For the body forces $\mathbf f$ that `spin_doctor` knows, the Poincaré force of the libration
in the frame of the mantle (forcing 8) and the force of the wave-like tide (forcing 1, axial for $m$ = 0 and equatorial
for $m$ = 1), the balance includes that torque, the same integral as $L$ over the toroidal part of $\mathbf f$ of degree 1
(`utils4pp.body_force_terms`). resid𝐋 is not computed for the other forcings, nor with a wall of zero radial stress
(`bco` or `bci` = 2) of a viscous fluid: the fluid crosses it, carrying angular momentum through it.

resid𝐋 is the residual of the balance relative to the largest of $\lvert\lambda\rvert L_{\rm abs}$, the torques,
ViscosD $L_{\rm abs}$, and Gaspard $L_{\rm abs}$ for $m = 1$, with $L_{\rm abs}$ the same integral of $\lvert T\rvert$:
a mode on which no torque acts (stress-free walls) is then checked to carry no angular momentum, and a steady flow
($\lambda = 0$) on which none acts, such as the thermal forcing between stress-free walls with a drag, to feel no torque
on the scale of the viscous one.

## Energy

With a field, resid𝐄 ([Magnetic field equations](magnetic-equations.md#energy-budgets)) counts the solids too. In the
units of `spin_doctor`'s kinetic energy a rigid rotation $\omega$ of a body of moment of inertia $I$ has the energy
$I\lvert\omega\rvert^2$, and the equations above give

$$
2\sigma\left(I_M\lvert\omega_M\rvert^2 + I_{IC}\lvert\omega_{IC}\rvert^2 + K\lvert\varphi\rvert^2\right)
= 2\,\mathrm{Re}\left(\omega_M^*\,\Gamma_M + \omega_{IC}^*\,\Gamma_{IC}\right) - 2Kc\,\lvert\varphi\rvert^2 .
$$

The power of the viscous torques is minus the work of the moving walls on the fluid, which the viscous power of the
fluid includes; at a stress-free outer wall with a drag the two differ by the power the drag dissipates ([Drag at the
outer wall equations](drag-equations.md#power-and-damping-rate)). The power of the magnetic torques moves energy between
the solids and the field: it is minus the flux of $\mathbf v\times\mathbf B_0$ through the moving walls ($\mathbf v$
their velocity), which resid𝐄 leaves out of the induction work with the rest of the flux of $\mathbf u\times\mathbf B_0$
through the walls (`utils4pp.INDU_WALLS`). So resid𝐄 adds the energy of the solids to the left-hand side, and the power
of the viscous torques and the loss $-2Kc\lvert\varphi\rvert^2$ to the right. It does so when both torques act on the
solids (`vtrq` and `mtrq` 1).

## Checks

`tests/rotdyn` checks the rotating solids against exact results (solids coupled only to each other; very heavy solids,
which behave as fixed walls), the conservation of total angular momentum and energy with a variable density and a
magnetic field, and the equatorial angular momentum balance. `tests/couette` checks resid𝐋 in the forced runs with
turning walls, libration and thermal forcing.
