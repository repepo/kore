# Drag at the outer wall equations

This page gives the condition of a drag at a stress-free outer wall ([Drag at the outer wall](drag.md)), the drag of the
topography, how the condition enters the rows of the wall, and the power and damping rate that `spin_doctor` reports.
They are those of the `conductive_IC` branch (its `topo.py`, the drag rows of its `assemble.py` and
`utils4pp.cmb_slip_K`), written for any density and viscosity profile in `bin/topo.py`.

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

on the unit sphere (`topo.couplings`, `topo.rows`). A constant drag gives $A_{ss} = A_{tt} = $ `kappa_cmb` and no
coupling between degrees. The drag of the topography is even about the equator, so it couples degrees of the same
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

## Checks (2026-11-06)

`tests/drag/drag_check.py` (`run_all.sh`, about 2 min on 2 ranks):

- **Against `conductive_IC`** (2d84351), eleven runs: a constant drag on the spin-over mode in a shell and in a full
  sphere; the drag of the topography on the spin-over mode, on an $m = 2$ mode and in a full sphere; the rotating solids
  with a stress-free outer wall without a drag, with each drag, with thin conducting layers and a dipole, and with a
  conducting inner core, a resolved mantle and a field of degree 2; the drag of the topography with a thin layer and a
  dipole.
  Eigenvalues to $6\cdot10^{-14}$, and the rotation of the solids and the torques on the mantle in `rotdyn.dat` to
  $3\cdot10^{-10}$, its precision.
- **Conservation** with `'Wu 2005'` (`r_out` 0.9, N 64) and the rotating solids: every mode but $\lambda = 0$ carries
  no total angular momentum ($2\cdot10^{-9}$) and closes resid𝐋 ($10^{-9}$); with `kappa_cmb` the energy budget of
  the fluid and the solids closes ($10^{-10}$), with the drag of the topography, a dipole, a conducting inner core and a
  resolved mantle resid𝐄 does ($10^{-10}$).
- **Equivalences.** A constant tensor given to the drag of the topography (the test hook `topo_K`) gives the eigenvalues
  of `kappa_cmb` ($2\cdot10^{-14}$). A rigid differential rotation $a = 0.3$ with `kappa_cmb` gives the eigenvalues of
  a frame turning $1 + a$ times faster with `kappa_cmb`$/(1 + a)$, shifted by $-i\,m\,\mathrm{Gaspard}\,a$
  ($2\cdot10^{-16}$): the drag acts on the slip relative to the mantle, in units of the frame's rotation rate.
- **Limits.** `kappa_cmb` = $10^8$ gives the eigenvalues of a no-slip wall ($7\cdot10^{-12}$; with the rotating solids
  $10^{-10}$, and their `rotdyn.dat`). A weak drag (`kappa_cmb` = $10^{-6}$) dissipates `kappa_cmb` times the Dtopo/κ of
  the stress-free run ($6\cdot10^{-5}$; with the rotating solids $2\cdot10^{-5}$), and with the mantle at rest changes
  the damping rate of the spin-over mode by `kappa_cmb` times its δσtopo/κ ($5\cdot10^{-4}$). With the rotating solids
  the change is 0.49 times it for their axial mode: the drag turns the mantle, which changes the other losses at the
  same order.
- **Frequencies.** $N/\Omega$ and $\omega_0/\Omega$ in `params.dat`: `topo_N` and `topo_omega` when set,
  $\sqrt{10}$ for the stratification `'linear'` with Beyonce 10, $\lvert$`itau`$\rvert$ and the forcing frequency
  otherwise ($10^{-9}$). The same run with a time unit twice as long (Gaspard 2, Beyonce 40, ViscosD, ThermaD and `itau`
  doubled) has its eigenvalues doubled ($10^{-15}$).
- **The power through the wall** closes in every run with a drag but the stiff ones (whose slip is round-off), to
  $2\cdot10^{-9}$, and in a forced run (forcing 6) to the printed digits.
- **Stops** of `background.check_drag`.

Also checked once:

- **Calibration.** At Ek $10^{-4}$ (N 64) the Earth's topography (`topo_hrms` 4.35e-4, `topo_lambda0` 0.431, `topo_H`
  0.8, $N = 0$) adds to the damping of the spin-over mode ($\omega_0/\Omega$ = 0.995) $-2.3100\cdot10^{-5}$, and
  `kappa_cmb` = 4.5e-6 adds $-2.3038\cdot10^{-5}$.
- **Weak and strong drag** (the spin-over mode, Ek $10^{-3}$, $\sqrt{E} = 0.032$). The change of the damping rate over
  `kappa_cmb` times δσtopo/κ is 0.97 at `kappa_cmb` = $10^{-3}$, 0.76 at $10^{-2}$ and 0.47 at $3\cdot10^{-2}$; the
  share of the no-slip wall's added damping is 0.72 at $3\cdot10^{-2}$, 0.94 at 0.1 and 0.998 at 1. With
  `kappa_cmb` = $10^{-4}$ the added damping is $-5.042\cdot10^{-4}$ at Ek $10^{-3}$ and $-5.063\cdot10^{-4}$ at
  $10^{-4}$.
- **Quadrature.** For the spin-over test ($\omega_0/\Omega$ = 0.98, $N/\Omega$ = 0.5) the couplings change by
  $6\cdot10^{-7}$ of the largest from 2000 to 16000 nodes, the error of the kinks. The nodes are those of
  `conductive_IC` (computed by `scipy.special.roots_legendre` instead of `numpy`'s `leggauss`, ten times faster; the
  couplings agree to $5\cdot10^{-12}$).
- **Other forcings.** The random Reynolds stress (forcing 11) with `kappa_cmb` and with the drag of the topography, and
  the entropy at the inner wall (forcing 13) with `kappa_cmb`: Dtopo is minus the power of the stress at the wall, to
  the printed digits.
- **Operators.** The drag and the rotating solids change only rows of `assemble.py`: `submatrices.npz` is bitwise the
  same with and without them, so their parameters are not stored in it.

## Reference

- Rekier, J., Triana, S. A., Barik, A., Abdulah, A. & Kang, S. 2025, arXiv:2507.01671 (eqs. 20, 21, B1 and B2).
