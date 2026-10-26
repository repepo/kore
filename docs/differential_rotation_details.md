# Differential rotation in kore: physics, equations and implementation

This page is a self-contained reference for the differential rotation of kore's `anelastic_log` branch. It covers:

- the physics: what a differentially rotating background does to a linear perturbation (§1);
- the anelastic (LBR) equations with differential rotation, and the derivation of every term that the differential
  rotation adds to the momentum and heat equations (§2);
- the spectral form of these terms: how they couple spherical-harmonic degrees, and how the angular coupling
  coefficients are obtained (§3);
- the implementation, term by term: radial profiles, operator labels, the operators, the assembly of the matrices and
  the diagnostics (§4 to §8);
- the internal consistency checks built into the code, and the known limitations (§9, §10);
- benchmarks against published eigenvalues: Baruteau & Rieutord 2013, Guenel et al. 2016, Mirouh et al. 2016,
  Mukhopadhyay et al. 2025a and 2025b (§11);
- the cost in time and memory against the highest degree of $\Delta\Omega$ (§12).

It describes the code after the differential rotation was unified in `bin/diff_rot.py` on the angular algebra of
`bin/angular.py` (2026-10-23, on top of commit `b4598010`). The code references give a file and a function, and a line
number in that version (line numbers drift; the function names do not).

Contents

- [0. Notation and code names](#0-notation-and-code-names)
- [1. Physics](#1-physics)
- [2. Equations](#2-equations)
- [3. Spectral form](#3-spectral-form)
- [4. Implementation overview](#4-implementation-overview)
- [5. Radial profiles](#5-radial-profiles)
- [6. Radial operators and labels](#6-radial-operators-and-labels)
- [7. The operators and the assembly](#7-the-operators-and-the-assembly)
- [8. Diagnostics](#8-diagnostics)
- [9. Internal consistency checks](#9-internal-consistency-checks)
- [10. Limitations](#10-limitations)
- [11. Benchmarks against published results](#11-benchmarks-against-published-results)
- [12. Cost against the highest degree](#12-cost-against-the-highest-degree)

## 0. Notation and code names

Everything is dimensionless. The length unit is the outer radius $r_0$ (r = 1 at the outer boundary), and the time unit
is $\tau$, chosen by `par.timescale` (`'rotation'` gives $\tau = 1/\Omega_{\rm ref}$).

| symbol | meaning | in the code |
| --- | --- | --- |
| $r, \theta, \phi$; $x$ | spherical coordinates; $x = \cos\theta$ | |
| $\varpi$ | cylindrical radius $r\sin\theta$ | |
| $\hat{\mathbf z}$ | rotation axis | |
| $\Omega_{\rm ref}$ | rotation rate of the frame in which kore solves | `diff_rot_frame` (nHz) for the solar profiles |
| $G$ | $\Omega_{\rm ref}\tau$, the frame's rotation rate in time units | `par.Gaspard` |
| $\Delta\Omega(r,\theta)$ | relative differential rotation, $\Omega = \Omega_{\rm ref}(1 + \Delta\Omega)$ | `par.diff_rot_type` |
| $f_n(r)$ | Legendre coefficients of $\Delta\Omega$: $\Delta\Omega = \sum_n f_n(r) P_n(x)$, $n$ even (`diff_rot_ampl` times $a_n$ (`diff_rot_term_amp`) times the shape of the type) | `zf0`, `zf2`, `zf4`, …; `zd`, `zs` for $rf_n'$, $r^2f_n''$ |
| $\mathbf U_0$ | the background flow seen in the rotating frame | |
| $\rho_0, T_0$ | background density and temperature | `prf.density` |
| $h$ | $\ln\rho_0$ | profile code `lho` |
| $p_0$ | $\rho_0 T_0$ (not the pressure) | `pss` |
| $g(r)$ | gravity | `gra` |
| $S_0(r)$ | spherical background entropy, in units of $c_p$; $S_0'$ its gradient | `dSd` ($S_0'$), `pdS` ($p_0 S_0'$) |
| $S_{tw}(r,\theta)$ | latitudinal background entropy (thermal wind, or prescribed) | |
| $\sigma_n(r)$ | Legendre coefficients of $S_{tw}$ | `tw2` = $p_0\sigma_2$, `tp2` = $p_0\sigma_2'$, … |
| $\nu, \kappa$ | viscosity and thermal diffusivity profiles | `vsc`, `kps` = $\kappa p_0$ |
| Beyonce | $(N_0\tau)^2$: the buoyancy force is $\mathrm{Beyonce}\thinspace g s \hat{\mathbf r}$ | `par.Beyonce` |
| ViscosD, ThermaD | $\nu_0\tau/r_0^2$, $\kappa_0\tau/r_0^2$ | `par.ViscosD`, `par.ThermaD` |
| $\mathbf u$, $s$ | velocity and entropy perturbation ($s = \delta s/c_p$) | |
| $P, T$ | poloidal and toroidal scalars of $\mathbf u$ | sections `u` and `v` |
| $m$ | azimuthal order: fields go as $e^{i m\phi}$ | `par.m` |
| $l$, $L$ | harmonic degree, $L = l(l+1)$ | `l` |
| $l'$, $l$ | target degree (row) and source degree (column) of a coupling | `l`, `l + offdiag` in `operators.py` |
| $Y_l$ | Schmidt-normalised spherical harmonic $Y_l^m$, $\int\lvert Y_l\rvert^2 d\Omega = n_l = 4\pi/(2l+1)$ | `utils.Ylm` |
| $\lambda$ | eigenvalue, fields $\propto e^{\lambda t}$, $\lambda = \sigma + i\omega$ ($\omega < 0$ is prograde) | |

Two code conventions recur everywhere:

- **Sections.** The unknowns and the matrix rows are stacked in blocks: section `u` holds the rows of the radial
  component of the double curl of the momentum equation (one row block per poloidal degree), section `v` the rows of the
  radial component of its curl (one per toroidal degree), and section `h` the heat equation (one per degree of $s$,
  which follows the poloidal degrees).
- **Row weights.** Each row is multiplied by a power of $r$ and of $\rho_0$, chosen so that every radial coefficient
  is a polynomial, and divided by $L'^3$, where $l'$ is the row degree.

## 1. Physics

### 1.1 The background

kore solves in a frame rotating at the constant rate $\Omega_{\rm ref}$. The background rotates at

$$
\Omega(r,\theta) = \Omega_{\rm ref}\left[1 + \Delta\Omega(r,\theta)\right], \qquad \Delta\Omega = \sum_{n = 0, 2, 4, \dots} f_n(r) P_n(\cos\theta) .
$$

Only even $n$ appear, so the rotation is symmetric about the equator, and the equatorial symmetry of the modes
(`par.symm`) still separates the problem into two classes. In the rotating frame the differential rotation is a steady
azimuthal flow

$$
\mathbf U_0 = G\thinspace \Delta\Omega\thinspace r\sin\theta\thinspace \hat{\mathbf e}_ \phi = G\thinspace \Delta\Omega\thinspace \hat{\mathbf z}\times\mathbf r . \qquad (1.1)
$$

The factor $G$ = `Gaspard` converts the rotation rate $\Omega_{\rm ref}\Delta\Omega$ to the time unit $\tau$. It is 1
when $\tau = 1/\Omega_{\rm ref}$. $G$ does not choose the frame: $\Delta\Omega$ is defined relative to the frame, and
for the solar profiles the frame rate is set by `diff_rot_frame` (or fixed at 435 nHz for `'solar'`).

The spherical background ($\rho_0$, $T_0$, $g$, $S_0$, $\nu$, $\kappa$) is the same as without differential
rotation. The differential rotation adds only two things: the flow $\mathbf U_0$, and a latitudinal entropy
$S_{tw}(r,\theta)$ (§1.3).

### 1.2 What the differential rotation does to a perturbation

A perturbation $\mathbf u$ is carried by the background flow and sheared by it. Linearising the advection
$(\mathbf U_0 + \mathbf u)\cdot\nabla(\mathbf U_0 + \mathbf u)$ gives (§2.3) three effects:

1. **Advection at the local rotation rate.** A pattern $e^{i m\phi}$ is carried round at the local rate
   $G\Delta\Omega$, which shifts its frequency by $-m G\Delta\Omega$ locally. With uniform $\Delta\Omega$ this is just a
   Doppler shift.
2. **Coriolis force of the local rotation.** The Coriolis parameter becomes $2G(1 + \Delta\Omega)$ instead of $2G$.
   Together with (1), a uniform $\Delta\Omega$ is exactly a change of frame (§2.6).
3. **Transport across the shear.** A perturbation that moves across the gradient of $\Omega$ carries its angular
   momentum to a place where the background rotates at a different rate. This is the term
   $r\sin\theta\thinspace (\mathbf u\cdot\nabla\Delta\Omega)\thinspace \hat{\mathbf e}_ \phi$. It is the only term through
   which the perturbation can exchange kinetic energy with the background flow (§2.3), so it is what drives shear
   instabilities.

In the heat equation the entropy perturbation is advected by $\mathbf U_0$ (the same Doppler shift for $s$), and the
perturbation velocity advects the latitudinal background entropy $S_{tw}$.

### 1.3 Thermal-wind balance

A rotation rate that varies along the rotation axis cannot be balanced by pressure alone. The centrifugal force of
the differential rotation then has a curl, and in a stratified fluid only a buoyancy force that varies with latitude,
that is a latitudinal entropy gradient, can balance it. This is thermal-wind balance. A differentially rotating star
in equilibrium therefore has a background entropy $S_0(r) + S_{tw}(r,\theta)$. kore computes $S_{tw}$ from $\Delta\Omega$
(`thermal_wind` 1 or 2), prescribes it (`thermal_wind = 3`), or omits it (`thermal_wind = 0`).

$S_{tw}$ is small in amplitude, since it scales as $G^2/(\mathrm{Beyonce}\thinspace g)$. But the buoyancy it produces
through a perturbation scales as $\mathrm{Beyonce}\thinspace g\thinspace s$. The product does not depend on
Beyonce: the feedback of $S_{tw}$ on the motion is of order $\Delta\Omega$ relative to the Coriolis force, the same
order as the momentum terms of §1.2. In a convection zone, where $S_0' \approx 0$, $S_{tw}$ is the only background
entropy gradient, and the meridional flow of a perturbation across it (the baroclinic coupling) can destabilise
modes.

### 1.4 What is not modelled

- The background is imposed and fixed: nothing maintains $\Omega$ or $S_{tw}$ against viscosity or diffusion, and the
  perturbation equations have no source terms for them.
- $\rho_0$, $T_0$ and $g$ stay spherical: no centrifugal distortion, no latitudinal variation of $\rho_0 T_0$ or $g$
  induced by the thermal wind, and no thermal-wind density in the mass flux. These effects are of relative size
  $\Delta\Omega/(\mathrm{Beyonce}\thinspace g)$.
- The thermal wind is the one consistent with LBR buoyancy. The full anelastic balance would add a term
  $p_b S_0'/(\rho_0 g)$ to $S_{tw}$ ($p_b$ the background pressure perturbation), the analogue of the $p_1 S_0'$ term
  that LBR drops for the perturbations. It matters only where $S_0' \neq 0$.
- There is no magnetic field and no viscous heating.

## 2. Equations

### 2.1 The linear anelastic (LBR) equations

In the frame rotating at $\Omega_{\rm ref}$, with the background of §1.1, a perturbation $(\mathbf u, s)$ obeys,
per unit mass (Lantz–Braginsky–Roberts form, with $\varpi'$ the reduced pressure $p'/\rho_0$):

$$
\frac{\partial\mathbf u}{\partial t} + 2G\thinspace \hat{\mathbf z}\times\mathbf u + \mathbf W(\mathbf u) = -\nabla\varpi' + \mathrm{Beyonce}\thinspace g\thinspace s\thinspace \hat{\mathbf r} + \mathrm{ViscosD}\thinspace \mathbf F_\nu(\mathbf u), \qquad (2.1)
$$

$$
\nabla\cdot(\rho_0\mathbf u) = 0, \qquad (2.2)
$$

$$
p_0\left[\frac{\partial s}{\partial t} + \mathbf U_0\cdot\nabla s + u_r S_0' + \mathbf u\cdot\nabla S_{tw}\right] = \mathrm{ThermaD}\thinspace \nabla\cdot(\kappa\thinspace p_0\nabla s). \qquad (2.3)
$$

Here $\mathbf F_\nu = \rho_0^{-1}\nabla\cdot(\rho_0\nu\thinspace \mathsf S)$ is the viscous force of the anelastic stress
tensor $\mathsf S$, and

$$
\mathbf W(\mathbf u) = (\mathbf U_0\cdot\nabla)\mathbf u + (\mathbf u\cdot\nabla)\mathbf U_0 \qquad (2.4)
$$

is the linearised advection by the background flow. Without differential rotation, $\mathbf W = 0$,
$\mathbf U_0\cdot\nabla s = 0$ and $S_{tw} = 0$, and (2.1)–(2.3) are kore's standard equations. The differential
rotation adds exactly three terms: $\mathbf W$ in (2.1), and $p_0\mathbf U_0\cdot\nabla s$ and
$p_0\mathbf u\cdot\nabla S_{tw}$ in (2.3).

Remarks:

- In LBR form the buoyancy depends on $s$ only. The reduced pressure is a gradient, so it disappears when the curl of
  (2.1) is taken (§3.2): kore never solves for it.
- The viscous force, the buoyancy and the diffusion are not changed by the differential rotation.
- (2.2) is satisfied exactly by the poloidal–toroidal form of $\mathbf u$ (§3.1).

kore solves the eigenvalue problem for fields $\propto e^{\lambda t + i m\phi}$, so $\partial_t \to \lambda$ and
$\partial_\phi \to i m$.

### 2.2 The background flow as a local rotation

Write the background rotation in time units as $\Omega_d(r,\theta) = G\Delta\Omega$, so that
$\mathbf U_0 = \Omega_d\thinspace \hat{\mathbf z}\times\mathbf r$. Two identities are used below:

$$
\mathbf U_0\cdot\nabla = \Omega_d\thinspace \partial_\phi, \qquad (\mathbf v\cdot\nabla)(\hat{\mathbf z}\times\mathbf r) = \hat{\mathbf z}\times\mathbf v ,
$$

for any vector $\mathbf v$. In components, with $\hat{\mathbf z} = \cos\theta\thinspace \hat{\mathbf r} - \sin\theta\thinspace \hat{\boldsymbol\theta}$,

$$
\hat{\mathbf z}\times\mathbf u = \left(-\sin\theta\thinspace u_\phi,\ -\cos\theta\thinspace u_\phi,\ \sin\theta\thinspace u_r + \cos\theta\thinspace u_\theta\right) \qquad (2.5)
$$

in the $(\hat{\mathbf r}, \hat{\boldsymbol\theta}, \hat{\mathbf e}_ \phi)$ basis.

### 2.3 Derivation of the momentum term

**Advection of the perturbation by $\mathbf U_0$.** $(\mathbf U_0\cdot\nabla)\mathbf u = \Omega_d\thinspace \partial_\phi\mathbf u$, where
$\partial_\phi$ also acts on the unit vectors: $\partial_\phi\hat{\mathbf r} = \sin\theta\thinspace \hat{\mathbf e}_ \phi$,
$\partial_\phi\hat{\boldsymbol\theta} = \cos\theta\thinspace \hat{\mathbf e}_ \phi$,
$\partial_\phi\hat{\mathbf e}_ \phi = -\sin\theta\thinspace \hat{\mathbf r} - \cos\theta\thinspace \hat{\boldsymbol\theta}$. With (2.5) this is

$$
(\mathbf U_0\cdot\nabla)\mathbf u = \Omega_d\left[i m\thinspace \mathbf u + \hat{\mathbf z}\times\mathbf u\right],
$$

where $i m\thinspace \mathbf u$ means $i m$ times each component.

**Advection of $\mathbf U_0$ by the perturbation.**

$$
(\mathbf u\cdot\nabla)\mathbf U_0 = (\mathbf u\cdot\nabla\Omega_d)\thinspace \hat{\mathbf z}\times\mathbf r + \Omega_d\thinspace (\mathbf u\cdot\nabla)(\hat{\mathbf z}\times\mathbf r)
= r\sin\theta\thinspace (\mathbf u\cdot\nabla\Omega_d)\thinspace \hat{\mathbf e}_ \phi + \Omega_d\thinspace \hat{\mathbf z}\times\mathbf u .
$$

**Sum.** With $\mathbf u\cdot\nabla\Delta\Omega = u_r\partial_r\Delta\Omega + (u_\theta/r)\partial_\theta\Delta\Omega$,

$$
\mathbf W = G\left[\Delta\Omega\left(i m\thinspace \mathbf u + 2\hat{\mathbf z}\times\mathbf u\right) + \sin\theta\left(r\thinspace \partial_r\Delta\Omega\thinspace u_r + \partial_\theta\Delta\Omega\thinspace u_\theta\right)\hat{\mathbf e}_ \phi\right]. \qquad (2.6)
$$

This is the force that `operators.differential_rotation` discretises. The three parts are those of §1.2: the Doppler
term $i m\Delta\Omega\thinspace \mathbf u$; the Coriolis force of the local rotation $2\Delta\Omega\thinspace \hat{\mathbf z}\times\mathbf u$, which
adds to the frame's $2\hat{\mathbf z}\times\mathbf u$; and the shear term.

For $\Delta\Omega = f(r)P_n(x)$, using $\partial_\theta P_n(\cos\theta) = -\sin\theta\thinspace P_n'(x)$, the components are

$$
\begin{aligned}
W_r/G &= f P_n\left[i m\thinspace u_r - 2\sin\theta\thinspace u_\phi\right],\cr
W_\theta/G &= f P_n\left[i m\thinspace u_\theta - 2x\thinspace u_\phi\right],\cr
W_\phi/G &= f P_n\left[i m\thinspace u_\phi + 2\sin\theta\thinspace u_r + 2x\thinspace u_\theta\right] + \sin\theta\left[(r f')\thinspace P_n\thinspace u_r - f\sin\theta\thinspace P_n'\thinspace u_\theta\right].
\end{aligned} \qquad (2.7)
$$

The profile enters only as $f$ and $r f'$; the second derivative $r^2 f''$ appears later, when the double curl
differentiates $W$ in $r$ (§3.4).

**Energy.** For a mode, $2\thinspace\mathrm{Re}\int\rho_0\thinspace \mathbf u^\ast\cdot\mathbf W\thinspace dV$ is the rate at
which $\mathbf W$ removes kinetic energy. The Doppler part gives $i m\int\rho_0\Delta\Omega\lvert\mathbf u\rvert^2 dV$,
which is imaginary. The Coriolis part gives $\mathbf u^\ast\cdot(\hat{\mathbf z}\times\mathbf u) = \hat{\mathbf z}\cdot(\mathbf u\times\mathbf u^\ast)$, which is also
imaginary. Only the shear term,
$2G\thinspace\mathrm{Re}\int\rho_0\thinspace r\sin\theta\thinspace u_\phi^\ast\thinspace (\mathbf u\cdot\nabla\Delta\Omega)\thinspace dV$, exchanges energy
with the background. These parts shift the frequency (residω) or change the growth rate (residσ) in `spin_doctor`
(§8).

### 2.4 Thermal-wind entropy

**The balance.** The steady background satisfies, in the rotating frame and per unit mass,

$$
(\mathbf U_0\cdot\nabla)\mathbf U_0 + 2G\thinspace \hat{\mathbf z}\times\mathbf U_0 = -\nabla\varpi_b + \mathrm{Beyonce}\thinspace g\thinspace S_{tw}\thinspace \hat{\mathbf r},
$$

where $\varpi_b$ collects the pressure and gravity potential terms. With $\mathbf U_0 = \Omega_d\thinspace \hat{\mathbf z}\times\mathbf r$, the left side is
$-(\Omega_d^2 + 2G\Omega_d)\thinspace \varpi\thinspace \hat{\mathbf e}_ \varpi = -G^2\left[(1 + \Delta\Omega)^2 - 1\right]\varpi\thinspace \hat{\mathbf e}_ \varpi$, the
centrifugal and Coriolis forces of the differential rotation ($\hat{\mathbf e}_ \varpi$ the cylindrical radial unit vector).
The $\phi$-component of the curl removes $\varpi_b$. Using $(\nabla\times F\hat{\mathbf e}_ \varpi)_ \phi = \partial_z F$ and
$(\nabla\times F\hat{\mathbf r})_ \phi = -r^{-1}\partial_\theta F$,

$$
\frac{1}{r}\frac{\partial S_{tw}}{\partial\theta} = \frac{G^2\thinspace r\sin\theta}{\mathrm{Beyonce}\thinspace g}\thinspace \frac{\partial (1 + \Delta\Omega)^2}{\partial z}, \qquad \frac{\partial}{\partial z} = \cos\theta\thinspace \frac{\partial}{\partial r} - \frac{\sin\theta}{r}\frac{\partial}{\partial\theta}. \qquad (2.8)
$$

**In the variable $x$.** With $\partial_\theta = -\sin\theta\thinspace \partial_x$ and $\partial_z = x\partial_r + r^{-1}(1 - x^2)\partial_x$, (2.8) becomes

$$
\frac{\partial S_{tw}}{\partial x} = -\frac{r^2 G^2}{\mathrm{Beyonce}\thinspace g}\left[x\thinspace \frac{\partial F}{\partial r} + \frac{1 - x^2}{r}\frac{\partial F}{\partial x}\right],
\qquad F = \begin{cases} 2\Delta\Omega & \text{(linear)} \cr (1 + \Delta\Omega)^2 - 1 & \text{(exact)} \end{cases} \qquad (2.9)
$$

`thermal_wind = 1` uses the linear form, `thermal_wind = 2` the exact one. Only the exact form is independent of the
frame: $G^2(1 + \Delta\Omega)^2$ is the square of the physical rotation rate, the same whatever $\Omega_{\rm ref}$ the
profile is written against, whereas $G^2\cdot 2\Delta\Omega$ is $2\Omega_{\rm ref}(\Omega - \Omega_{\rm ref})\tau^2$
and changes with the frame. With `thermal_wind = 1` the modes therefore move with the choice of frame, at second order
in $\Delta\Omega$.

The right side of (2.9) is a polynomial in $x$ of degree $n_{\max} + 1$ (linear) or $2n_{\max} + 1$ (exact), so
$S_{tw} = \sum_n \sigma_n(r) P_n(x)$ has even degrees up to $n_{\max} + 2$ or $2n_{\max} + 2$. The $l = 0$ part
$\sigma_0$ is a spherical entropy, absorbed in $S_0$, and set to zero.

**Closed forms for $\Delta\Omega = f_0 + f_2 P_2$.** Writing the bracket of (2.9) on $P_1$ and $P_3$ (with
$x P_2 = \tfrac{3}{5}P_3 + \tfrac{2}{5}P_1$, $3x(1 - x^2) = \tfrac{6}{5}(P_1 - P_3)$, $P_2' = 3P_1$,
$P_4' = 7P_3 + 3P_1$) and matching coefficients gives, with $e = 0$ for `thermal_wind = 1` and $e = 1$ for
`thermal_wind = 2`,

$$
\begin{aligned}
\sigma_2 &= -\frac{2r^2G^2}{\mathrm{Beyonce}\thinspace g}\left[(1 + e f_0)\left(\frac{f_0'}{3} + \frac{f_2'}{21} + \frac{4}{7}\frac{f_2}{r}\right) + \frac{e\thinspace f_2 (f_0' + f_2')}{21}\right],\cr
\sigma_4 &= -\frac{6r^2G^2}{35\thinspace \mathrm{Beyonce}\thinspace g}\left[(1 + e f_0)\left(f_2' - \frac{2f_2}{r}\right) + e\left(f_2 f_0' + \frac{4}{11}f_2 f_2' + \frac{14}{11}\frac{f_2^2}{r}\right)\right],\cr
\sigma_6 &= -e\thinspace \frac{4r^2G^2}{77\thinspace \mathrm{Beyonce}\thinspace g}\thinspace f_2\left(f_2' - \frac{2f_2}{r}\right).
\end{aligned} \qquad (2.10)
$$

kore computes the $\sigma_n$ from (2.9) by the Legendre-series algebra of §5.4, for any degrees of $\Delta\Omega$;
(2.10) is kept as a validation (`tests/diff_rot/gen_check.py`, §9). Two analytic checks:

- **Cylindrical rotation**, $\Delta\Omega = \varepsilon\varpi^2$, i.e. $f_0 = \tfrac{2}{3}\varepsilon r^2$, $f_2 = -\tfrac{2}{3}\varepsilon r^2$:
  $\sigma_2 = \sigma_4 = \sigma_6 = 0$. A rotation constant on cylinders is barotropic and needs no thermal wind.
- **Conical rotation**, $\Delta\Omega = \varepsilon\sin^2\theta$, i.e. $f_0 = \tfrac{2}{3}\varepsilon$, $f_2 = -\tfrac{2}{3}\varepsilon$
  (linear form): $\sigma_2 = 16\varepsilon r/(21\thinspace\mathrm{Beyonce}\thinspace g)$, $\sigma_4 = -8\varepsilon r/(35\thinspace\mathrm{Beyonce}\thinspace g)$, and
  $S_{tw}(\text{pole}) - S_{tw}(\text{equator}) = \varepsilon r/(\mathrm{Beyonce}\thinspace g)$: an equator rotating faster
  ($\varepsilon > 0$) needs warmer poles.

**Prescribed latitudinal entropy (`thermal_wind = 3`).** $\partial_\theta S_{tw}$ = `dSdtheta` $\sin 2\theta$,
independent of $\Omega$ and of $r$. Since $\sin 2\theta = -\tfrac{2}{3}\thinspace dP_2/d\theta$,

$$
\sigma_2 = -\tfrac{2}{3}\thinspace \texttt{dSdtheta}, \qquad \sigma_n = 0 \quad (n \ge 4).
$$

`dSdtheta` is in the units of $s$ ($\delta s/c_p$); negative values give warm poles. $\Omega$ still enters through
$\mathbf W$ and the advection of $s$. For a uniformly rotating background with a prescribed $S_{tw}$, use
`diff_rot_type = 'rigid'` with `diff_rot_ampl = 0`.

### 2.5 The heat-equation terms

$S_{tw}$ does not depend on $\phi$, so $\mathbf U_0\cdot\nabla S_{tw} = 0$, and $\mathbf U_0\cdot\nabla s = G\Delta\Omega\thinspace \partial_\phi s$.
The left side of (2.3) gains

$$
p_0\left[\thinspace \underbrace{i m G\Delta\Omega\thinspace s}_ {\text{(a) advection}} + \underbrace{u_r\frac{\partial S_{tw}}{\partial r}}_ {\text{(b) radial}} + \underbrace{\frac{u_\theta}{r}\frac{\partial S_{tw}}{\partial\theta}}_ {\text{(c) latitudinal}}\thinspace \right]. \qquad (2.11)
$$

(a) is always included. (b) and (c) are included for `thermal_wind` 1, 2, 3; `thermal_wind_radial = 0` drops (b).
(a) is anti-Hermitian under the entropy pairing $\int s^\ast(\cdot)\thinspace dV$, so it only shifts the frequency; (b) and (c)
exchange entropy variance with the flow.

### 2.6 An exact result: the rigid offset

For a uniform $\Delta\Omega = a$ (`diff_rot_type = 'rigid'`), $\partial\Delta\Omega = 0$, $S_{tw} = 0$, and (2.6) and
(2.11) reduce to $G a(i m\mathbf u + 2\hat{\mathbf z}\times\mathbf u)$ and $i m G a\thinspace p_0 s$. Equations (2.1) and (2.3) become those
of uniform rotation at rate $G(1 + a)$, with $\lambda$ replaced by $\lambda + i m G a$. So

$$
\lambda(a, G) = \lambda_0\big(G(1 + a)\big) - i m G a , \qquad (2.12)
$$

where $\lambda_0(G)$ is the eigenvalue without differential rotation. This identity tests the Doppler and Coriolis
parts of the operators and their signs (§9).

## 3. Spectral form

### 3.1 Fields

At fixed $m$ and equatorial symmetry, each field is a sum over degrees,

$$
\mathbf u = \sum_l\left[\rho_0^{-1}\nabla\times\nabla\times(\rho_0 P_l Y_l\thinspace \mathbf r) + \nabla\times(T_l Y_l\thinspace \mathbf r)\right], \qquad s = \sum_l s_l Y_l ,
$$

where $P$ and $s$ use the poloidal degrees and $T$ the toroidal ones (`utils.ell(m, lmax, symm)` returns both
lists). This form satisfies (2.2) exactly. For one degree, with $Z_l = i m Y_l/\sin\theta$ and

$$
a_l = \frac{(r P_l)'}{r} + h' P_l = \frac{P_l}{r} + P_l' + h' P_l , \qquad (3.1)
$$

the components are

$$
u_r = \frac{L P_l}{r} Y_l, \qquad u_\theta = a_l\thinspace \partial_\theta Y_l + T_l Z_l, \qquad u_\phi = a_l Z_l - T_l\thinspace \partial_\theta Y_l . \qquad (3.2)
$$

These expressions underlie `diff_rot._primitives` (through the basis fields of §3.3), `utils4pp.diff_rot_setup` and
`ru_insane.py`.

### 3.2 Projection of a vector field, and kore's rows

Any vector field $\mathbf F$ is projected on vector spherical harmonics as

$$
\mathbf F = \sum_l\left[F^q_l\thinspace Y_l\thinspace \hat{\mathbf r} + F^s_l\thinspace r\nabla Y_l + F^t_l\thinspace r\nabla Y_l\times\hat{\mathbf r}\right], \qquad
r\nabla Y = \partial_\theta Y\thinspace \hat{\boldsymbol\theta} + Z\thinspace \hat{\mathbf e}_ \phi, \quad r\nabla Y\times\hat{\mathbf r} = Z\thinspace \hat{\boldsymbol\theta} - \partial_\theta Y\thinspace \hat{\mathbf e}_ \phi , \qquad (3.3)
$$

with, since $\int(\lvert\partial_\theta Y_l\rvert^2 + \lvert Z_l\rvert^2)\thinspace d\Omega = L n_l$,

$$
F^q_l = \frac{1}{n_l}\int F_r Y_l^\ast\thinspace d\Omega, \qquad
F^s_l = \frac{1}{L n_l}\int\left(F_\theta\thinspace \partial_\theta Y_l^\ast + F_\phi Z_l^\ast\right)d\Omega, \qquad
F^t_l = \frac{1}{L n_l}\int\left(F_\theta Z_l^\ast - F_\phi\thinspace \partial_\theta Y_l^\ast\right)d\Omega . \qquad (3.4)
$$

By (3.2) the velocity has $u^q = L P/r$, $u^s = a$, $u^t = T$.

Taking curls removes the pressure:

$$
\hat{\mathbf r}\cdot\nabla\times\mathbf F = \sum_l\frac{L}{r}F^t_l Y_l, \qquad
\hat{\mathbf r}\cdot\nabla\times\nabla\times\mathbf F = \sum_l\frac{L}{r^2}\left[F^q_l - \partial_r(r F^s_l)\right]Y_l . \qquad (3.5)
$$

(The toroidal part $F^t r\nabla Y\times\hat{\mathbf r} = \nabla\times(F^t Y\mathbf r)$ has no radial curl-curl, and the spheroidal part
$F^q Y\hat{\mathbf r} + F^s r\nabla Y$ has the curl $r^{-1}\left[F^q - \partial_r(r F^s)\right] r\nabla Y\times\hat{\mathbf r}$.)

kore's rows of degree $l'$ are (3.5) multiplied by $r^{\rm rpower}\rho_0^{\rm rhopower}$ and divided by $L'^3$. With the
defaults `rpower_u` = 5, `rpower_v` = 3, `rhopower_u` = `rhopower_v` = 1:

$$
\mathrm{row}_ u(\mathbf F) = \frac{r^3\rho_0}{L'^2}\left[F^q_{l'} - \partial_r(r F^s_{l'})\right], \qquad
\mathrm{row}_ v(\mathbf F) = \frac{r^2\rho_0}{L'^2}F^t_{l'} . \qquad (3.6)
$$

For example, $\mathbf F = \mathbf u$ gives
$\mathrm{row}_ u = \left[L r^2\rho_0 P - r^3\rho_0(2P' + rP'' + h'P + r h'' P + r h' P')\right]/L^2$ and $\mathrm{row}_ v = r^2\rho_0 T/L^2$,
which is `operators.inertia` (line 28). The diff-rot operator is (3.6) applied to $\mathbf W$.

### 3.3 Angular primitives

Insert one source degree $l$ of (3.2) into (2.7), with $\Delta\Omega = f(r)P_n(x)$. Each component of $\mathbf W$ is a
sum of four radial "shapes" times angular functions:

| shape | radial factor | from |
| --- | --- | --- |
| `q` | $f\thinspace L P_l/r$ | $u_r$ with $f$ |
| `a` | $f\thinspace a_l$ | the consoidal part of $u_\theta$, $u_\phi$ |
| `T` | $f\thinspace T_l$ | the toroidal part |
| `dq` | $(r f')\thinspace L P_l/r$ | $u_r$ in the shear term |

From (2.7) and (3.2), the angular functions, per unit $G$ and per unit shape, are ($Y = Y_l$, $Z = Z_l$):

| shape | $W_r$ | $W_\theta$ | $W_\phi$ |
| --- | --- | --- | --- |
| `q` | $i m P_n Y$ | 0 | $2\sin\theta\thinspace P_n Y$ |
| `a` | $-2 i m P_n Y$ | $P_n(i m\thinspace \partial_\theta Y - 2xZ)$ | $P_n(i m Z + 2x\thinspace \partial_\theta Y) - \sin^2\theta\thinspace P_n'\thinspace \partial_\theta Y$ |
| `T` | $2\sin\theta\thinspace P_n\thinspace \partial_\theta Y$ | $P_n(i m Z + 2x\thinspace \partial_\theta Y)$ | $P_n(-i m\thinspace \partial_\theta Y + 2xZ) - \sin^2\theta\thinspace P_n' Z$ |
| `dq` | 0 | 0 | $\sin\theta\thinspace P_n Y$ |

Projecting each entry on the target degree $l'$ with (3.4) defines twelve matrices $C^k_{\rm sh}[l', l]$, with
$k \in \lbrace q, s, t\rbrace$ the projection and sh the shape. Then, for the target degree $l'$,

$$
W^k_{l'} = G\sum_l\left[C^k_q\thinspace f\frac{L_l P_l}{r} + C^k_a\thinspace f a_l + C^k_T\thinspace f T_l + C^k_{dq}\thinspace (r f')\frac{L_l P_l}{r}\right]. \qquad (3.7)
$$

**The angular algebra.** The $C^k_{\rm sh}$ are not computed by integration but built exactly, as sparse products
of a few matrices on the degree index (`bin/angular.py`). Multiplying a field by a function of $x$ maps degree $l$
to a few neighbouring degrees, by two exact recurrences of the Schmidt harmonics ($p_l$, $q_l$ as in (3.8) below):

$$
x\thinspace Y_l = p_l Y_{l+1} + q_l Y_{l-1} \quad\to X, \qquad
(1 - x^2)\frac{dY_l}{dx} = -\sin\theta\thinspace \partial_\theta Y_l = (l+1)q_l Y_{l-1} - l p_l Y_{l+1} \quad\to D ,
$$

with rows the target degree and columns the source degree. Their adjoints under $\int d\Omega$ (with
$N = \mathrm{diag}(n_l)$) are $X^\dagger = N^{-1}X^{\rm T}N = X$ and $D^\dagger = N^{-1}D^{\rm T}N = 2X - D$ (integration by
parts). $P_n(X)$ and $P_n'(X)$ follow from the three-term recurrences
$(k+1)P_{k+1} = (2k+1)XP_k - kP_{k-1}$, $P_{k+1}' = (k+1)P_k + XP_k'$. Write $R_l = Y_l\hat{\mathbf r}$, $S_l = r\nabla Y_l$,
$T_l = r\nabla Y_l\times\hat{\mathbf r}$ for the basis fields of (3.3), $L$ for $\mathrm{diag}(l(l+1))$ acting on the source
and $1/L'$ for the division of each row by its $L'$ (0 for $l' = 0$). The operations needed by (2.7) are then:

| operation | result on $(q, s, t)$ | from |
| --- | --- | --- |
| $g(x)\thinspace R_l$ | $q$: $g(X)$ | |
| $g(x)\thinspace S_l$ | $s$: $(g(X)L - g'(X)D)/L'$, $t$: $-i m\thinspace g'(X)/L'$ | $\int g\nabla Y_l\cdot\nabla Y_{l'}^\ast = \int Y_{l'}^\ast(L_l g Y_l - g' DY_l)$ |
| $g(x)\thinspace T_l$ | $s$: $i m\thinspace g'(X)/L'$, $t$: $(g(X)L - g'(X)D)/L'$ | the same, rotated by $\hat{\mathbf r}\times$ |
| $\hat{\mathbf z}\times(c\thinspace Y\hat{\mathbf r})$ | $s$: $-i m\thinspace c/L'$, $t$: $D^\dagger c/L'$ | $\hat{\mathbf z}\times\hat{\mathbf r} = \sin\theta\thinspace \hat{\mathbf e}_ \phi$ |
| $\hat{\mathbf z}\times S_l$ | $-i m R_l - x T_l$ | (2.5) |
| $\hat{\mathbf z}\times T_l$ | $-(DY_l)\hat{\mathbf r} + x S_l$ | (2.5) |
| $\sin\theta\thinspace F_\theta$ (a scalar) | $-D s + i m\thinspace t$ | $\sin\theta\thinspace \partial_\theta Y = -DY$ |

`diff_rot._primitives` (line 327) composes them, per unit $G$ and per unit shape:

$$
\mathbf W_{\rm sh} = P_n(x)\left[i m\thinspace \mathbf u_{\rm sh} + 2\hat{\mathbf z}\times\mathbf u_{\rm sh}\right] + \hat{\mathbf z}\times\left[-P_n'(x)\sin\theta\thinspace u_{\theta,{\rm sh}}\thinspace \hat{\mathbf r}\right], \qquad
\mathbf W_{dq} = \hat{\mathbf z}\times\left[P_n(x)\thinspace Y\hat{\mathbf r}\right],
$$

with $\mathbf u_q = R_l$, $\mathbf u_a = S_l$, $\mathbf u_T = T_l$ (the second term, the $u_\theta$ part of the shear term,
vanishes for the shape `q`). Every product is a sparse product of banded matrices, so the cost is $O(l_{\max})$ and the
result is exact to round-off at any degree. The matrices are built up to $l_w$ = `lmax` + n + 6, because a product of
matrices truncated at $l_w$ is wrong only in its last few degrees.

The primitives were first computed by Gauss–Legendre quadrature of the table above; that builder is kept as an
independent reference in `tests/diff_rot/quad_reference.py`. It cost $O(l_{\max}^3)$ (177 s at `lmax` 1025) and failed
silently from $l = 646$ on, where `scipy.special.sph_harm_y` returns NaN; the two agree to $\sim 10^{-12}$ below that
(§9).

### 3.4 From the primitives to the radial operators

Put (3.7) into (3.6) and expand $a_l$ with (3.1). For the $v$ rows this is direct:

$$
\mathrm{row}_ v(\mathbf W) = \frac{G r^2\rho_0}{L'^2}\left[(L_l C^t_q + C^t_a)\frac{fP}{r} + C^t_a\thinspace f(P' + h'P) + C^t_T\thinspace f T + L_l C^t_{dq}\frac{(rf')P}{r}\right].
$$

For the $u$ rows, $\partial_r(r W^s)$ differentiates the profile and the unknowns: with
$r a = P + rP' + r h' P$,

$$
\partial_r(r f a) = f'(P + rP' + rh'P) + f(2P' + rP'' + h'P + rh''P + rh'P'), \qquad
\partial_r\left(r\thinspace C_{dq}^s (rf')\frac{L_lP}{r}\right) = L_l C^s_{dq}\left[(f' + r f'')P + (rf')P'\right],
$$

which is where $r^2 f''$ enters. Collecting the coefficient of each radial operator gives the table below. Each
coefficient is a matrix in $(l', l)$, and in the code it is stored multiplied by $L'$; the operator then divides by
$L'^3$ (§7.1), so the net factor is $1/L'^2$ as in (3.6). The radial operators include the row weights, and their
labels (§6) use the profile codes `{a}`, `{s}`, `{p}` of $f_n$, $r f_n'$, $r^2 f_n''$, that is `zf`, `zd`, `zs`
followed by the degree character (`zf0`, `zd0`, `zs0` for $n = 0$, `zf2`, … ; §5.3).

| row, column | coefficient name | coefficient / $L'$ | radial operator | label |
| --- | --- | --- | --- | --- |
| u, P | `f_2C_D0P` | $L_l C^q_q + C^q_a$ | $r^2\rho_0 f P$ | `u3{a}0_D0` |
| u, P | `f_2C_D1P` | $C^q_a - L_l C^s_q - 2C^s_a$ | $r^3\rho_0 f P'$ | `u2{a}0_D1` |
| u, P | `f_2C_lho1_D0P` | $C^q_a - C^s_a$ | $r^3\rho_0 f h' P$ | `u2{a}0lho1_D0` |
| u, P | `f_2C_lho2_D0P` | $-C^s_a$ | $r^4\rho_0 f h'' P$ | `u1{a}0lho2_D0` |
| u, P | `f_2C_lho1_D1P` | $-C^s_a$ | $r^4\rho_0 f h' P'$ | `u1{a}0lho1_D1` |
| u, P | `f_2C_D2P` | $-C^s_a$ | $r^4\rho_0 f P''$ | `u1{a}0_D2` |
| u, P | `df_2C_D0P` | $L_l C^q_{dq} - L_l C^s_q - C^s_a - L_l C^s_{dq}$ | $r^2\rho_0 (rf') P$ | `u3{s}0_D0` |
| u, P | `df_2C_D1P` | $-(C^s_a + L_l C^s_{dq})$ | $r^3\rho_0 (rf') P'$ | `u2{s}0_D1` |
| u, P | `df_2C_lho1_D0P` | $-C^s_a$ | $r^3\rho_0 (rf') h' P$ | `u2{s}0lho1_D0` |
| u, P | `d2f_2C_D0P` | $-L_l C^s_{dq}$ | $r^2\rho_0 (r^2 f'') P$ | `u3{p}0_D0` |
| u, T | `f_2C_D0T` | $C^q_T - C^s_T$ | $r^3\rho_0 f T$ | `u2{a}0_D0` |
| u, T | `f_2C_D1T` | $-C^s_T$ | $r^4\rho_0 f T'$ | `u1{a}0_D1` |
| u, T | `df_2C_D0T` | $-C^s_T$ | $r^3\rho_0 (rf') T$ | `u2{s}0_D0` |
| v, P | `f_1C_D0P` | $L_l C^t_q + C^t_a$ | $r\rho_0 f P$ | `v2{a}0_D0` |
| v, P | `f_1C_D1P` | $C^t_a$ | $r^2\rho_0 f P'$ | `v1{a}0_D1` |
| v, P | `f_1C_lho1_D0P` | $C^t_a$ | $r^2\rho_0 f h' P$ | `v1{a}0lho1_D0` |
| v, P | `df_1C_D0P` | $L_l C^t_{dq}$ | $r\rho_0 (rf') P$ | `v2{s}0_D0` |
| v, T | `f_1C_D0T` | $C^t_T$ | $r^2\rho_0 f T$ | `v1{a}0_D0` |

This table is `diff_rot.momentum_coefficients` (line 351, the dictionary `K`) and `diff_rot.MOMENTUM` (line 413). The
coefficient names are those of the closed forms, `tests/diff_rot/diff_rot_coefficients.py`: `f`, `df`, `d2f` for the profile factor ($f$, $rf'$,
$r^2f''$), `1C`/`2C` for the single/double curl rows, the radial factor (`D0`, `D1`, `D2`, `lho1`, `lho2`), and `P`/`T`
for the column. ($C^q_{dq} = 0$, since the shear term has no radial component, so `df_2C_D0P` keeps
$L_l C^q_{dq}$ only formally.)

### 3.5 Selection rules

- The angular factors of (2.7) are $P_n$, $xP_n$, $\sin\theta P_n$ and $\sin^2\theta\thinspace P_n'$, polynomials in $x$ (up to the
  $\sin\theta$ factors absorbed by $\partial_\theta Y$ and $Z$) of degree at most $n + 1$. So $W$ from degree $l$
  reaches the target degrees $\lvert l' - l\rvert \le n + 1$.
- $\Delta\Omega$ is even in $x$, so $\mathbf W$ keeps the equatorial symmetry of $\mathbf u$. In kore's ordering the
  poloidal and toroidal degrees of one symmetry class alternate, so same-type couplings ($P \to P$, $T \to T$) have
  even $\Delta l$ and mixed ones ($P \leftrightarrow T$) odd $\Delta l$.
- Together: $\lvert\Delta l\rvert \le n$ for $P \to P$ and $T \to T$, and $\le n + 1$ for $P \leftrightarrow T$
  (`diff_rot.max_dl`, line 428). For $\Delta\Omega = f_0 + f_2P_2$: $\Delta l \in \lbrace 0, \pm 2\rbrace$ and
  $\lbrace\pm 1, \pm 3\rbrace$; the $f_0$ terms alone have $\Delta l = 0$ and $\pm 1$, like the Coriolis force.
- For $m = 0$ the Doppler term vanishes, but the Coriolis and shear terms do not.

The coupled problem is therefore wider than with uniform rotation: the row block of degree $l'$ has entries in the
column blocks of up to $n_{\max} + 1$ neighbouring degrees of each type.

### 3.6 Closed forms for $P_0$ and $P_2$

For $n = 0$ and $n = 2$ the coefficients of the table of §3.4 were first derived in closed form. kore now builds every
degree with the algebra of §3.3, and the closed forms are kept as its validation in
`tests/diff_rot/diff_rot_coefficients.py` (44 functions, $\lvert\Delta l\rvert \le 3$, and the $\sigma_n$ of (2.10)): `f_*` and `df_*`, `d2f_*` for $f_2$
($P_2$, unnormalised, despite "Y20" in the comments), `f0_*`, `df0_*`, `d2f0_*` for $f_0$. Each function takes the
target degree `l`, `m` and `offdiag` = $l - l'$ (source minus target) and returns the coefficient, already multiplied
by $L'$, with explicit conditions where a degree would fall below $m$ or below 1.

Two worked examples show the structure.

- **$T \to T$, $P_0$, $\Delta l = 0$.** With $P_n = 1$, the `T` entries of the table of §3.3 give
  $C^t_T = \left[i m L n_l + 2\int x(\partial_\theta Y Z^\ast - Z\partial_\theta Y^\ast)d\Omega\right]/(L n_l)$.
  For real $Y$ and $\partial_\theta Y$ the integral is
  $-2im\int x Y\partial_\theta Y\thinspace d\theta\thinspace d\phi = -i m n_l$, so
  $L C^t_T = i m (L - 2)$, which is `f0_1C_D0T` = `1j*(-2 + l + l**2)*m`. The $i m L$ is the Doppler term,
  and $-2im$ is the local Coriolis force: compare `operators.coriolis`, whose $T \to T$ row is $-2 i m G\thinspace v_1/L^3$.
- **$P \to T$, $P_0$, $\Delta l = -1$.** For constant $f_0 = a$, `f0_1C_D0P` = $2(l-1)^2(l+1)\sqrt{l^2 - m^2}/(2l - 1)$ and
  `f0_1C_lho1_D0P` = `f0_1C_D1P` = $-2(l^2 - 1)\sqrt{l^2 - m^2}/(2l - 1)$ are $2L$ times the coefficients
  $C(l - 1)$ and $-C$ of the $\Delta l = -1$ Coriolis row, $C = (l-1)\sqrt{l^2 - m^2}/(l(2l - 1))$ (`operators.coriolis`,
  line 90). So a uniform $f_0 = a$ gives exactly $a$ times the Coriolis operator plus $i m G a$ times the inertia
  operator, which is (2.12) at the matrix level.

The algebra (§3.3) and the closed forms agree to round-off, $\le 1.3\times 10^{-15}$ of the largest coefficient of the
same row block, up to `lmax` 1025 (`tests/diff_rot/gen_check.py`, §9). Entry by entry, the closed forms are the more
accurate: a few coefficients are small differences of terms of size $O(L)$ in the algebra of §3.4 (for example
`df0_2C_D1P` = $2im$ exactly, $-L(C^s_a + L_l C^s_{dq})$ in the algebra), so their own relative error grows like
$L\epsilon$, about $10^{-10}$ at $l \approx 1000$, while staying at round-off on the scale of the row, which is the
scale at which they enter $A$. Moving the profile $f$ through the curls analytically before projecting would remove
these cancellations; it has not been needed.

### 3.7 The heat-equation terms in harmonic form

Multiplication by a function of $x$ is a matrix on the degree index. With Schmidt harmonics and
$p_l = \sqrt{(l+1)^2 - m^2}/(2l+1)$, $q_l = \sqrt{l^2 - m^2}/(2l+1)$:

$$
x\thinspace Y_l = p_l Y_{l+1} + q_l Y_{l-1} \quad\to X, \qquad
(1 - x^2)\frac{dY_l}{dx} = (l+1)q_l Y_{l-1} - l p_l Y_{l+1} \quad\to D , \qquad (3.8)
$$

with rows the target degree and columns the source degree. Multiplying by $P_n(x)$ is then the matrix $P_n(X)$, and
by $P_n'(x)$ the matrix $P_n'(X)$. These are the matrices of the angular algebra of §3.3 (`angular.Algebra`), shared
by the momentum and heat terms.

For one source degree, $\partial_\theta S_{tw} = -\sin\theta\sum_n\sigma_n P_n'(x)$, and
$\sin\theta\thinspace \partial_\theta Y = -(1 - x^2)\thinspace dY/dx = -DY$. With (3.2), the three terms of (2.11) projected on degree
$l'$ are, before the row weight and the $p_0$ factor,

$$
\begin{aligned}
\text{(a)}\quad & i m G\sum_n f_n\thinspace \left[P_n(X)\right]_ {l'l}\thinspace s_l ,\cr
\text{(b)}\quad & \frac{L_l P_l}{r}\sum_n\sigma_n'\thinspace \left[P_n(X)\right]_ {l'l} ,\cr
\text{(c)}\quad & \frac{a_l}{r}\sum_n\sigma_n\thinspace \left[P_n'(X)\thinspace D\right]_ {l'l} ,\cr
\text{(c}')\quad & -\frac{i m T_l}{r}\sum_n\sigma_n\thinspace \left[P_n'(X)\right]_ {l'l} .
\end{aligned} \qquad (3.9)
$$

(c) and (c′) are the poloidal and toroidal parts of the latitudinal term. The couplings are
$\Delta l \in \lbrace 0, \pm 2, \dots, \pm n_{\max}\rbrace$ for (a) ($s \to s$), $\lbrace 0, \pm 2, \dots, \pm n_{tw}\rbrace$
for (b), (c) ($P \to s$) and $\lbrace\pm 1, \dots, \pm(n_{tw} - 1)\rbrace$ for (c′) ($T \to s$), with $n_{tw}$ the
highest degree of $S_{tw}$ (`diff_rot.tw_nmax`).

## 4. Implementation overview

Everything specific to the differential rotation is in one module, [`bin/diff_rot.py`](../bin/diff_rot.py), built on
the angular algebra of [`bin/angular.py`](../bin/angular.py). The rest of kore calls it:

| stage | what it does for the differential rotation | where |
| --- | --- | --- |
| parameters | switches, profile choice, amplitude, `'user'` profiles | [`params_default.py`](../bin/params_default.py#L107), lines 107–162 |
| angular algebra | $X$, $D$, $P_n(X)$, the vector operations, harmonics on a grid | [`angular.py`](../bin/angular.py) |
| everything else | profiles, thermal wind, angular couplings, labels, blocks, Δl ranges | [`diff_rot.py`](../bin/diff_rot.py), sections 1–4 of the module |
| radial matrices | builds the `.mtx` of the labels it is given | [`submatrices.py`](../bin/submatrices.py#L189) (`dr.labels()`), [`radial_profiles.burrito`](../bin/radial_profiles.py) (`dr.profile_functions()`) |
| operators | thin wrappers | [`operators.differential_rotation`](../bin/operators.py#L113), [`thermal_diffrot`](../bin/operators.py#L190) |
| assembly | places the blocks, over the Δl of `dr.dl_range` | [`assemble.py`](../bin/assemble.py#L262), lines 262–461 |
| diagnostics | powers and budgets on an $(r, \theta)$ grid, independently of the algebra | [`utils4pp.diff_rot_setup`](../bin/utils4pp.py#L494), [`spin_doctor.py`](../bin/spin_doctor.py) |

`diff_rot.py` imports only `parameters` and `angular` at the top; `utils` and `radial_profiles` are imported inside the
functions that need them. `radial_profiles.burrito` imports `diff_rot` when it meets a differential-rotation profile
code, so there is no import cycle, and the angular part (`_primitives`, `momentum_coefficients`) can be imported and
tested without a run directory.

### 4.1 Parameters

| parameter | default | meaning |
| --- | --- | --- |
| `diff_rot` | 0 | 1 switches the differential rotation on |
| `diff_rot_type` | `'Y20'` | the profile (§5.1) |
| `diff_rot_ampl` | 1.0 | the factor of $\Delta\Omega$, for every type |
| `diff_rot_param` | 0.0 | the shape parameter: $\sigma$ of `'shellular'`, `p` in `'user'` expressions |
| `diff_rot_term_amp` | `[1.0]*9` | $a_0, a_2, \dots, a_{16}$: factors of the $P_0, P_2, \dots, P_{16}$ terms of $\Delta\Omega$, any type, after `diff_rot_ampl` (missing entries are 1) |
| `diff_rot_profiles` | `{}` | `'user'`: `{n: 'numpy expression'}` for each even degree (§5.1) |
| `diff_rot_frame` | 456.0 | `'solar_P4'`, `'tabulated'`: $\Omega_{\rm ref}/2\pi$ in nHz |
| `diff_rot_solar` | `[464, -76, -50, 433.5, 0.71, 0.045]` | `'solar_P4'`: $c_0$, $c_2$, $c_4$, $\Omega_{RZ}$ (nHz), $r_{tc}$, $d_{tc}$ |
| `diff_rot_file`, `diff_rot_nmax`, `diff_rot_rdeg` | `''`, 8, 24 | `'tabulated'`: the .npz file, the highest degree (even), the degree of the radial fit |
| `diff_rot_window`, `diff_rot_sun_deg`, `diff_rot_hold` | None, 60, None | `'tabulated'`: shell fit or one even whole-Sun fit, its degree, and an optional radius below which $f_n$ is held constant |
| `thermal_wind` | 1 | 0 none, 1 linear, 2 exact, 3 prescribed (§2.4) |
| `thermal_wind_radial` | 1 | 0 drops term (b) |
| `thermal_wind_amplitude` | 1.0 | factor on $S_{tw}$ (`thermal_wind` 1, 2) |
| `dSdtheta` | 0.0 | `thermal_wind = 3`: $\partial_\theta S_{tw}$ = `dSdtheta` $\sin 2\theta$ |

`Gaspard`, `Beyonce`, `ViscosD`, `ThermaD` must be computed for $\Omega_{\rm ref}$, and set after `par.set_scales()`,
which resets them (with `timescale = 'rotation'` it sets `Beyonce = 0`; `thermal_wind` 1 and 2 then stop
`submatrices.py` with an error through `diff_rot.check`, since $S_{tw} \propto 1/\mathrm{Beyonce}$).

Before 2026-10-23 the amplitude was `diff_rot_amplitude`, which was the exponent $\sigma$ for `'shellular'` and $N_0^2$
for `'shellular_boussinesq'`; the switch `diff_rot_generic` (closed forms or the algebra for $P_0$, $P_2$) no longer
exists. Both former names stop a run (`diff_rot.check`), so that an old parameter file does not run with the defaults.

### 4.2 The functions of `diff_rot.py`

| function | returns |
| --- | --- |
| `shapes()` (line 116) | `{n: f(r)}`, the shapes of the degrees of $\Delta\Omega$ for `diff_rot_type`, before `diff_rot_ampl` |
| `degrees()`, `n_max()` | the even degrees of $\Delta\Omega$ and the highest one |
| `tw_degrees()`, `tw_nmax()` | the degrees of $S_{tw}$: 2 to $n_{\max} + 2$ (`thermal_wind` 1), $2n_{\max} + 2$ (2), 2 (3), none (0) |
| `codes(n)` | the profile codes `zf`, `zd`, `zs`, `tw`, `tp` + the degree character |
| `fX(n, r, Dorder)` (line 227) | $f_n$ (`diff_rot_ampl` × $a_n$ × shape) and its derivatives, from the Chebyshev fit (§5.2) |
| `raw(n, r)` | $f_n$ evaluated directly |
| `sigma_coefficients(r)` (line 254) | the Legendre coefficients $\sigma_k(r)$ of $S_{tw}$ (§5.4) |
| `sigmaX(n, r, Dorder)` | $\sigma_n$ and its derivatives from its Chebyshev fit |
| `profile_functions()` (line 306) | `{code: f(r, Dorder)}` of every diff-rot profile code, for `burrito` |
| `momentum_coefficients(n, m)`, `momentum_coef(...)` | the coefficients of §3.4, and one of them for a target degree and Δl |
| `heat_coefficients(m)`, `heat_coef(...)` | $P_n(X)$, $P_n'(X)$, $P_n'(X)D$ by diagonal, and one coefficient |
| `labels()` (line 446) | every radial operator of the terms, with the parity class of its column |
| `dl_range(section, component)` (line 460) | the Δl of the blocks of a row section and column |
| `momentum_block(...)`, `heat_block(...)` (lines 473, 492) | one block of $A$ |
| `check()`, `description()` | stops on settings that cannot work; the `'user'` definitions as text for `params.dat` |
| `TYPES` | the types, in the order of their code in `params.dat` |

## 5. Radial profiles

### 5.1 The profiles $f_n$ of each type

`diff_rot.shapes` (line 116) defines the shape of each degree; $f_n$ = `diff_rot_ampl` × $a_n$ × shape, with $a_n$ =
`diff_rot_term_amp[n/2]` (1 in the table below). With $a$ =
`diff_rot_ampl`, $p$ = `diff_rot_param`, and $x$ the fractional solar radius (`diff_rot.solar_x`:
$x = r\cdot$`aux0` with `model_type = 'smoothed sun'`, $x = r$ otherwise):

| `diff_rot_type` | $f_0/a$ | $f_2/a$ | $f_n/a$, $n \ge 4$ |
| --- | --- | --- | --- |
| `'rigid'` | 1 | | |
| `'Y20'` | | 1 | |
| `'Y20-wall-bounded'` | | $(1 - r)(r - r_i)$, $r_i$ = `ricb` | |
| `'shellular'` | $r^p - 1$ | | |
| `'shellular_boussinesq'` | $\tfrac12(1 - r^2)$ | | |
| `'cylindrical'` ($a\varpi^2$) | $\tfrac{2}{3}r^2$ | $-\tfrac{2}{3}r^2$ | |
| `'conical'` ($a\sin^2\theta$) | $\tfrac{2}{3}$ | $-\tfrac{2}{3}$ | |
| `'solar'` | $(-7.695/435)\thinspace H(x)$ | $(-86.04/435)\thinspace H(x)$ | |
| `'solar_P4'` | $(\Omega_{RZ} + (O_0 - \Omega_{RZ})E)/\Omega_{\rm ref} - 1$ | $O_2 E/\Omega_{\rm ref}$ | $O_4 E/\Omega_{\rm ref}$ ($n = 4$) |
| `'tabulated'` | fit | fit | fit, up to `diff_rot_nmax` |
| `'user'` | `diff_rot_profiles[0]` | `diff_rot_profiles[2]` | `diff_rot_profiles[n]` |

Here $H(x) = \tfrac12\left[1 + \tanh(2(x - 0.71)/0.1)\right]$ (the 'solar' frame is 435 nHz), and for `'solar_P4'`
$E(x) = \tfrac12\left[1 + \mathrm{erf}((x - r_{tc})/d_{tc})\right]$ and $\Omega_{CZ} = c_0 + c_2\cos^2\theta + c_4\cos^4\theta$ projected on
Legendre polynomials ($\cos^2\theta = (1 + 2P_2)/3$, $\cos^4\theta = 1/5 + 4P_2/7 + 8P_4/35$):
$O_0 = c_0 + c_2/3 + c_4/5$, $O_2 = 2c_2/3 + 4c_4/7$, $O_4 = 8c_4/35$.

**`'user'`.** `diff_rot_profiles` maps each even degree to a string, a numpy expression of `r` (kore radius), `x`
(fractional solar radius) and `p` (`diff_rot_param`), evaluated in a namespace with only `sin`, `cos`, `tan`, `arcsin`,
`arccos`, `arctan`, `sinh`, `cosh`, `tanh`, `exp`, `log`, `log10`, `sqrt`, `abs`, `sign`, `pi`, `where`, `minimum`,
`maximum`, `erf`, `erfc` (no builtins). For example a cylindrical rotation:

```python
par.diff_rot = 1; par.diff_rot_type = 'user'
par.diff_rot_profiles = {0: '(2/3)*r**2', 2: '-(2/3)*r**2'}
par.diff_rot_ampl = -0.25
```

builds matrices bitwise identical to `diff_rot_type = 'cylindrical'` with the same amplitude. The degrees must be
even integers, and the strings must not contain `#`: `spin_doctor.py` ends each `params.dat` row of a `'user'` run
with them, after a `#` (`diff_rot.description`), so that every row carries its profile and `np.loadtxt` still reads the
numeric columns.

**`'tabulated'`** (`diff_rot._tabulated_fits`, line 42). The file holds `ome[θ, r]` (nHz), `r` (R☉) and `theta`
(rad). At each table radius, $\Omega/\Omega_{\rm ref} - 1$ is interpolated in $\cos\theta$ by a cubic spline and projected on
$P_n$ with 64 Gauss–Legendre nodes, $f_n = \tfrac{2n+1}{2}\int(\Omega/\Omega_{\rm ref} - 1)P_n\thinspace dx$. Each $f_n(r)$ is
then fitted by least squares with a Chebyshev series:

- by default (`diff_rot_window = None` with `ricb > 0`), on the table radii within 0.01 of the shell, of degree
  min(`diff_rot_rdeg`, half the number of radii);
- with `diff_rot_window = 'sun'` (the default at `ricb = 0`), once over the whole Sun, of degree `diff_rot_sun_deg`,
  even in $r$: the data are mirrored to $-r$, continued below the first table radius by its value, and the odd
  coefficients are set to zero. `diff_rot_hold` = $x_h$ holds each $f_n$ at $f_n(x_h)$ below $x_h$, blended by an erf of
  width 0.01.

The fit residuals and, for the shell fit, the truncation error of the expansion up to `diff_rot_nmax` are printed in
nHz.

### 5.2 Derivatives and the full sphere

The operators need $f_n$, $r f_n'$ and $r^2 f_n''$ on the Chebyshev nodes. `diff_rot.fX` (line 227) transforms the
shape to $N$ Chebyshev coefficients on the radial domain (`utils.chebco_f`, coefficients below $10^{-14}$ of the
largest dropped; cached per degree), evaluates the series or its derivatives (`utils.funcheb`) and multiplies by
`diff_rot_ampl` and $a_n$. So $f'$ and $f''$ are always consistent with the resolved $f$, also for tabulated, step-like or
`'user'` profiles.

With `ricb = 0` the radial coordinate is $r \in [-1, 1]$ and each operator keeps only one parity of Chebyshev
polynomials (§6.3). The shapes are therefore evaluated at $\lvert r\rvert$, which makes $f_n$ even, $f_n'$ odd, and
$rf_n'$, $r^2f_n''$ even. A shape with a kink at $r = 0$ (`'Y20-wall-bounded'`, `'shellular'` unless $p$ is even and
positive, or a `'user'` expression such as `r`) then converges slowly in $N$; `submatrices.py` prints a WARNING for the
predefined ones (line 242).

### 5.3 Profile codes

Each profile used in an operator has a three-letter code: two letters and the degree character `diff_rot.DIGITS[n]`
(0–9, then a–z for 10–35):

| code | profile | function (`diff_rot.profile_functions`) |
| --- | --- | --- |
| `zf0`, `zf2`, … | $f_n$ (zonal flow) | `fX(n, r, Dorder)` |
| `zd0`, `zd2`, … | $r f_n'$ | Chebyshev fit of $r\thinspace$`fX(n, r, 1)` |
| `zs0`, `zs2`, … | $r^2 f_n''$ | Chebyshev fit of $r^2\thinspace$`fX(n, r, 2)` |
| `tw2`, `tw4`, … | $p_0\sigma_n$ | Chebyshev fit of $p_0\thinspace$`sigmaX(n, r, 0)` |
| `tp2`, `tp4`, … | $p_0\sigma_n'$ (odd in $r$, in `utils.odd_profiles`) | Chebyshev fit of $p_0\thinspace$`sigmaX(n, r, 1)` |

They are not in `radial_profiles.proffdir` until `burrito` first meets one and adds `diff_rot.profile_functions()`.
With `adiab_tw` (`ThermaD = 0` with the thermal wind) the `tw`, `tp` profiles are $\sigma_n$, $\sigma_n'$ without $p_0$
(`diff_rot._ptw`), because those heat rows are divided by $p_0$ (§7.2). A separate `tp` code is needed: a label such as
`h1tw21_D0` would mean $r(p_0\sigma_2)'P$, the derivative of the product, not $r p_0\sigma_2' P$.

### 5.4 The thermal-wind coefficients $\sigma_n$

`diff_rot.sigma_coefficients(r)` (line 254) solves (2.9) in Legendre space at all the radii at once, with the $m = 0$
matrices of `angular.py` (the Schmidt harmonics of order 0 are the Legendre polynomials):

- the Legendre coefficients of $\Delta\Omega$ and $\partial_r\Delta\Omega$ are the $f_n$ and $f_n'$ (`fX`); a product of two
  series is $\sum_n f_n\thinspace P_n(X)$ applied to the other one, so $F = 2\Delta\Omega + \Delta\Omega^2$ and
  $\partial_r F = 2(1 + \Delta\Omega)\partial_r\Delta\Omega$ (`thermal_wind = 2`), or $2\Delta\Omega$, $2\partial_r\Delta\Omega$ (1);
- multiplication by $x$ is $X$ and $(1 - x^2)\thinspace d/dx$ is $D$, so the bracket of (2.9) is
  $X\thinspace\partial_rF + DF/r$, and $(1 - x^2)\thinspace\partial_x S_{tw} = -\tfrac{r^2G^2}{\mathrm{Beyonce}\thinspace g}(I - X^2)(X\partial_rF + DF/r)$;
- that is $D\sigma$, which is solved for $\sigma$ (with $\sigma_0 = 0$, the $l = 0$ part belonging to $S_0$) by forward
  substitution of $h_k = (k+2)q_{k+1}\sigma_{k+1} - (k-1)p_{k-1}\sigma_{k-1}$ (`_solve_D`).

This is exact (no quadrature), for any degrees of $\Delta\Omega$, and matches the closed forms (2.10) to $2\times 10^{-16}$
of the natural scale $r^2G^2\lvert\partial\Delta\Omega\rvert/(\mathrm{Beyonce}\thinspace g)$ (§9). The result is multiplied
by `thermal_wind_amplitude`; `thermal_wind = 3` gives $\sigma_2 = -\tfrac23$`dSdtheta`. `sigmaX` differentiates the
Chebyshev series of $\sigma_n(r)$ for $\sigma_n'$. At `ricb = 0`, $\sigma_n$ is even ($f$ even, $f'$ and $g$ odd) and
$\sigma_n'$ odd, hence `tp` in `odd_profiles`.

## 6. Radial operators and labels

### 6.1 Label grammar

A label names one radial operator: `sX[fu1D1][fu2D2]_DY` (`utils.decode_label`, line 117).

- `s`: the section, `u`, `v` or `h`.
- `X`: lowers the row's power of $r$: the operator carries $r^{p - X}$, with $p$ = `rpower_u` (5) in `u`,
  `rpower_v` (3) in `v`, and 2 in `h` (1 when `ThermaD = 0` without `adiab_tw`). Sections `u` and `v` also carry
  $\rho_0$ to the power `rhopower_u`, `rhopower_v` (1 by default), section `h` no $\rho_0$.
- `fu1D1`, `fu2D2`: up to two profiles (§5.3) and their derivative orders. `lho1`, `lho2` are $h'$ and $h''$.
- `_DY`: the radial derivative order applied to the unknown.

For example, with the defaults, `u2zd20lho1_D0` is $r^3\rho_0\thinspace (r f_2')\thinspace h'$ acting on $P$, and `h1tw20lho1_D0` is
$r\thinspace p_0\sigma_2\thinspace h'$ acting on $P$. `rap.burrito` evaluates $r^{\rm rpower}\rho_0^{\rm rhopower} f_1 f_2$ on the
Chebyshev nodes, `utils.chebco_f` transforms it, and `utils.Mlam` builds the multiplication matrix in the Gegenbauer
basis $C^{(\lambda)}$ of the section; `submatrices.py` then multiplies on the right by the derivative matrix and changes
the basis: `u` is $C^{(4)}$, `v` $C^{(2)}$, `h` $C^{(2)}$ ($C^{(0)}$ if `ThermaD = 0`, $C^{(1)}$ with `adiab_tw`). The
diff-rot labels need at most $P''$ in `u`, $P'$ in `v` and $P'$ in `h`, so they fit those bases. An operator that
several terms share is computed only once.

### 6.2 The labels

`diff_rot.labels()` (line 446) lists every radial operator of the terms, and `submatrices.py` (line 189) adds them with
the vector parity of their column (`vP` for $P$ and $s$, `vT` for $T$), after `diff_rot.check()`:

- for each degree $n$ of $\Delta\Omega$, the 18 momentum labels of the table of §3.4 (`diff_rot.MOMENTUM` with the codes
  of $n$);
- with `thermal = 1` (`_heat_labels`, line 432): (a) `h0pss0zf{n}0_D0` ($r^2 p_0 f_n$; `h0zf{n}0_D0` without `pss` if
  `ThermaD = 0`) for each degree of $\Delta\Omega$; and for each degree $n$ of $S_{tw}$: (b) `h1tp{n}0_D0`
  ($r p_0\sigma_n'$, only if `thermal_wind_radial = 1`), (c) `h2tw{n}0_D0`, `h1tw{n}0_D1`, `h1tw{n}0lho1_D0`
  ($p_0\sigma_n$ times $P$, $rP'$, $rh'P$), (c′) `h1tw{n}0_D0` ($r p_0\sigma_n$ acting on $T$, parity `vT`).

Like every other operator, each diff-rot matrix has its last `chop` rows dropped and that many zero rows added at the
top, to make room for the boundary-condition rows of its row block, so the boundary conditions are unchanged by the
differential rotation.

### 6.3 Parity in a full sphere

At `ricb = 0` the vector parity of the unknowns is $v_P = (-1)^m$`symm` for $P$ and $s$, and $v_T = -v_P$ for $T$. An
operator has the parity $(-1)^{\text{rpower} + \text{dorder}_ 1 + \text{dorder}_ 2 + \text{DY} + \text{adj}}$, with `adj` = 1 for a
first profile in `utils.odd_profiles`. `utils.remroco` keeps the rows of parity $v\times$(operator parity) and the
columns of parity $v$. This is correct only if every profile has the parity assumed: $f_n$, $rf_n'$, $r^2f_n''$, $\sigma_n$
even, $\sigma_n'$ odd, $h^{(k)}$ of parity $(-1)^k$; §5.2 explains how the profiles are made so. If a profile's Chebyshev
coefficients of the wrong parity exceed $10^{-8}$ of the largest, `submatrices.py` prints a WARNING: that part of the
profile is dropped.

## 7. The operators and the assembly

### 7.1 Momentum: `diff_rot.momentum_block`

`operators.differential_rotation(l, section, component, offdiag)` (line 113) calls
`diff_rot.momentum_block(l, section, component, offdiag, M)` (line 473) with `M` the operators' namespace, where the
`.mtx` matrices are loaded under their labels. For the row degree `l` (= $l'$), the section (`'u'` or `'v'`), the column
type (`'upol'` for $P$, `'utor'` for $T$) and `offdiag` = $l - l'$, it sums over the degrees $n$ of $\Delta\Omega$ that
couple this Δl (§3.5): $\lvert\Delta l\rvert \le$ `max_dl(n, …)` and the parity of Δl; for each, the coefficient
`momentum_coef(name, n, l, m, offdiag)` times the matrix of each label of `momentum_labels(n, …)`. The result is
multiplied by `Gaspard` and divided by $L'^3$. Since each stored coefficient carries a factor $L'$, the block is

$$
\frac{G}{L'^3}\sum(L'\times\text{coefficient})\times(\text{radial operator}) = \mathrm{row}(\mathbf W) \quad\text{of (3.6)} .
$$

The $1/L'^3$ makes these rows consistent with all the other momentum operators (§3.2). It returns `[matrix, offd]`,
`offd` the column block offset (`utils.dr_offd`, §7.3), or `[0, 0]` if no degree couples this Δl.

`momentum_coefficients(n, m)` (line 351) builds the coefficients once per $(n, m)$, for degrees up to `lmax` + n + 6
(building beyond `lmax` keeps the entries near `lmax` free of truncation errors), and stores them both as sparse
matrices and by diagonal (`angular.band`, $\lvert\Delta l\rvert \le n + 3$), so that `momentum_coef` is an array lookup. The
whole set takes milliseconds at `lmax` 1025.

### 7.2 Heat: `diff_rot.heat_block`

`operators.thermal_diffrot` (line 190) calls `diff_rot.heat_block(l, component, offdiag, M)` (line 492), which returns
the section-`h` blocks of (3.9) times the row weight. With `ThermaD > 0` the rows are $r^2 p_0\times$(eq)$/L'^3$:

| term | component, $\Delta l$ | expression | radial operator |
| --- | --- | --- | --- |
| (a) | `'temp'`, even, $\lvert\Delta l\rvert \le n$ | `1j*m*heat_coef('Pn')*M['h0pss0zf{n}0_D0']`, times `Gaspard` | $i m G\thinspace r^2 p_0 f_n\thinspace s$ |
| (b) | `'upol'`, even, $\le$ `tw_nmax` | `Ls*heat_coef('Pn')*M['h1tp{n}0_D0']` | $L_l\thinspace r p_0\sigma_n'\thinspace P$ |
| (c) | `'upol'`, even, $\le$ `tw_nmax` | `heat_coef('dPnD')*(M['h2tw{n}0_D0'] + M['h1tw{n}0_D1'] + M['h1tw{n}0lho1_D0'])` | $p_0\sigma_n(P + rP' + rh'P)$ |
| (c′) | `'utor'`, odd, $<$ `tw_nmax` | `-1j*m*heat_coef('dPn')*M['h1tw{n}0_D0']` | $-i m\thinspace r p_0\sigma_n\thinspace T$ |

with the sums over the degrees of $\Delta\Omega$ for (a) and of $S_{tw}$ for (b), (c), (c′). `Ls` is $L_l$ of the source
degree (from $u_r = L_l P/r$). The angular coefficients come from `heat_coefficients(m)` (line 386): $X$, $D$ from
`angular.Algebra` up to `lmax` + max(8, $n_{hi}$ + 4), and $P_n(X)$, $P_n'(X)$, $P_n'(X)D$ for every even $n$ up to
the highest degree of $\Delta\Omega$ and $S_{tw}$, stored by diagonal. The column offset is the index difference in
`utils.ell`.

**Sign.** The block is $-\text{out}/L'^3$. The section-`h` rows of $A$ hold the right-hand side of the heat equation
($B$ = +`entropy`, `thermal_advection` = $-r p_0 S_0' u_r$), so the left-hand-side terms (2.11) enter $A$ with a minus
sign. In the momentum rows it is the opposite: $B$ = −`inertia`, so the left-hand-side term $\mathbf W$ enters $A$ with
a plus sign, as the Coriolis force does.

**Factors of $G$.** (a) carries `Gaspard` explicitly; (b), (c), (c′) carry $G^2$ through $\sigma_n$, except with
`thermal_wind = 3`.

**Without thermal diffusion.** With `ThermaD = 0` the heat rows are divided by $p_0$, which is about $10^{-7}$ near the
surface of a solar model and spoils the conditioning otherwise.

- With `thermal_wind = 0` the rows are $r\times$(eq)$/p_0$ in $C^{(0)}$, and (a) is `h0zf{n}0_D0` ($r f_n s$).
- With `thermal_wind ≥ 1` (`adiab_tw`), (b) and (c) do not fit there: the weight $r$ leaves $P/r$ in (c), and
  $C^{(0)}$ cannot hold $P'$. The rows are then $r^2\times$(eq)$/p_0$ in $C^{(1)}$ (`decode_label`, line 139;
  `submatrices.py`, line 93), with the same labels, the `tw`, `tp` profiles without $p_0$ (§5.3), and still no
  thermal boundary rows.

### 7.3 Assembly: `assemble.py`

`assemble.py` loops over the row degrees of each MPI rank. The unknown vector is [$P$ blocks][$T$ blocks][$s$
blocks], each of `N1` Chebyshev coefficients, with `nb` blocks per type. For row block $l'$ (row `row`, diagonal
column `col0`) and each Δl of `diff_rot.dl_range(section, component)` (line 460) whose source degree exists, it calls the
operator and, unless it returned `[0, 0]`, adds the block at column `basecol + col0 + offd*N1`:

| rows | column | Δl (`dl_range`) | lines |
| --- | --- | --- | --- |
| `u` | $P$ (`basecol = 0`) | $-n_{\max}$ … $n_{\max}$, step 2 | 262–270 |
| `u` | $T$ (`basecol = nb*N1`) | $-n_{\max}-1$ … $n_{\max}+1$, step 2 | 288–296 |
| `v` | $P$ | $-n_{\max}-1$ … $n_{\max}+1$, step 2 | 350–358 |
| `v` | $T$ | $-n_{\max}$ … $n_{\max}$, step 2 | 376–384 |
| `h` | $s$ (`basecol = 2*nb*N1`), term (a) | $-n_{\max}$ … $n_{\max}$, step 2 | 438–443 |
| `h` | $P$, terms (b), (c) | $-n_{tw}$ … $n_{tw}$, step 2 | 448–453 |
| `h` | $T$, term (c′) | $1-n_{tw}$ … $n_{tw}-1$, step 2 | 456–461 |

with $n_{\max}$ the highest degree of $\Delta\Omega$ and $n_{tw}$ that of $S_{tw}$. The momentum blocks are added with a plus
sign to $A$, whose momentum rows are Coriolis − viscous − buoyancy + diff rot; the heat blocks carry their sign
already (§7.2). $B$ is unchanged: the differential rotation has no time derivative. Forced runs ($A - i\omega_f B$) need no
change either. $A$ and $B$ are then divided by $\lVert A\rVert_F$ as usual.

The offset `offd` counts blocks, not degrees: the poloidal degrees of a symmetry class step by 2, so for a $P$ (or
$s$) column `offd` = $\Delta l/2$; for a $T$ column it is the index difference between the toroidal list and the
poloidal one, which depends on the symmetry (`utils.dr_offd`, line 49).

## 8. Diagnostics

`spin_doctor.py` checks every solution against the energy, frequency and entropy budgets. With differential
rotation, the terms couple different degrees, so a per-degree worker cannot evaluate them from its own degree alone.
`utils4pp.diff_rot_setup` (line 494), called once per solution by `utils4pp.diagnose`, therefore:

1. builds $\mathbf u$ and $\partial_r\mathbf u$ on the $(r_k, \cos\theta)$ grid from the per-degree components (`velocity`), with
   `lmax` + 10 + max($n_{tw}$, $n_{\max}$) Gauss–Legendre nodes in $\cos\theta$. The harmonics
   $Y_l$, $\partial_\theta Y_l$ and $Z_l$ come from `angular.ylm_grid`: the recurrence of $X$ started at
   $Y_m \propto \sin^m\theta$, differentiated in $\theta$ for $\partial_\theta Y_l$, and started at $\sin^{m-1}\theta$ for
   $Y_l/\sin\theta$, so that nothing is divided by $\sin\theta$. The recurrence is linear, so it is carried as a
   mantissa times a power of two per node: for large $m$ the start $\sin^{m-1}\theta$ would underflow near the poles
   (from $m \approx 330$ at $l$ of a few thousand), and the mantissas are rescaled by $2^{-600}$ when they grow past
   $2^{600}$, which is exact. It is valid at any $l$ and $m$ (it matches exact rational values of $P_l$ to
   $\sim 10^{-13}$ up to $l = 2000$, and keeps the norms to $\sim 10^{-13}$ at $m = 1425$, $l = 3000$), where
   `scipy.special.sph_harm_y` is NaN from $l = 646$ on; `utils.Ylm` and `utils.Ylm_full` use it too;
2. evaluates $\Delta\Omega$, $\partial_r\Delta\Omega$, $\partial_r^2\Delta\Omega$, $\partial_\theta\Delta\Omega$ there, as numpy Legendre series of the $f_n$ (`diff_rot.fX`);
3. forms the force $\mathbf F = -\mathbf W$ of (2.6) and its radial derivative (the `force` closure, line 541), and projects
   them back on $(q, s, t)$ with (3.4): `drforce[l]`;
4. with `thermal = 1`, forms $p_0[i m G\Delta\Omega s + \mathbf u\cdot\nabla S_{tw}]$ and projects it on $Y_l$: `drheat[l]`.

Only the profiles $f_n$ come from `diff_rot.py`; the angular part here is independent of its algebra. The
$\sigma_n$ used here are computed by `utils4pp.thermal_wind_entropy` (line 451), which integrates (2.9) in $x$ with
numpy Legendre series at each radius and writes out its $r$ derivative, not with `diff_rot.sigma_coefficients`. So
the budgets test the assembled operators against a second implementation of the same equations.

The per-degree workers then add:

- `flow_worker` (lines 692–702): `Wdr` = $2\thinspace\mathrm{Re}\int\rho_0\mathbf u^\ast\cdot\mathbf F\thinspace dV$, `Wdr_im` its imaginary part, and
  `Ensdr` the enstrophy term $(\nabla\times\mathbf u)\cdot(\nabla\times\mathbf F)$;
- `thermal_worker` (lines 788–791): `Wdr_thm` = $-2\thinspace\mathrm{Re}\int p_0 s^\ast[i m G\Delta\Omega s + \mathbf u\cdot\nabla S_{tw}]\thinspace dV$.

`spin_doctor` (lines 360–377) includes them in the residuals:

$$
\begin{aligned}
\text{resid}\sigma &: \quad 2\sigma\thinspace KE - D_{kin} - W_{thm} - W_{dr} = 0,\cr
\text{resid}\omega &: \quad 2\omega\thinspace KE + W_{cor,im} - D_{kin,im} - W_{thm,im} - W_{dr,im} = 0,\cr
\text{resid}s &: \quad 2\sigma\thinspace TE - D_{thm} - W_{adv} - W_{dr,thm} = 0 ,
\end{aligned}
$$

and the enstrophy residual includes `Ensdr`. By §2.3, only the shear part of $\mathbf W$ contributes to `Wdr`; the
Doppler and local-Coriolis parts appear in `Wdr_im`. The diff-rot options are written to `params.dat`
(`spin_doctor.param_table`: `diff_rot_ampl`, the $a_n$, `diff_rot_param`, and `diff_rot_type` as its index in `diff_rot.TYPES`,
with a `'user'` run's profile definitions at the end of the row after `#`), and `Wdr`, `Wdr_im`, `Ensdr`, `Wdr_thm` to
`flow.dat` and `thermal.dat`.

## 9. Internal consistency checks

These checks test the implementation against itself and against exact results, not against published calculations.

- **`bin/ru_insane.py`**, after `submatrices.py` (no MPI, seconds; needs N ≥ 96). With `diff_rot = 1`,
  `dr_blocks` (line 200) packs the diff-rot blocks as `assemble.py` does, and the test pairs them with random smooth
  fields, comparing each $P$/$T$ part against a direct $(r, \cos\theta)$ quadrature of
  $\int\rho_0\mathbf x^\ast\cdot G[(\mathbf U_0'\cdot\nabla)\mathbf y + (\mathbf y\cdot\nabla)\mathbf U_0']\thinspace dV$; the heat blocks against
  $-\int p_0 x_s^\ast\thinspace i m G\Delta\Omega\thinspace y_s$ (and their anti-Hermitian symmetry) and against
  $-\int p_0 x_s^\ast\thinspace \mathbf u(y)\cdot\nabla S_{tw}$; and for `'rigid'`, the block identities of §3.6:
  $R = a(i m G I + C)$, (a) $= -i m G a E$, thermal-wind blocks zero. A 1 % error in one coefficient of one $\Delta l$
  shows up at $10^{-4}$ to 1. Its grid uses only the profiles of `diff_rot.py`: $\Delta\Omega$ as numpy Legendre series,
  $\partial_\theta S_{tw}$ pointwise from the balance, $S_{tw}$ by Gauss integration in $x$, $\partial_r S_{tw}$ by Chebyshev
  differentiation.
- **`tests/diff_rot/angular_check.py`** (standalone, 30 s; `--quick` 4 s): the harmonics of `angular.ylm_grid`
  against exact rational values of $P_l$ (up to $l = 2000$), against scipy ($l \le 640$), and their norms up to
  $l = 3000$; $X$ and $D$ applied on a grid, and $X^\dagger = X$, $D^\dagger = 2X - D$; every vector operation of §3.3
  against grid projections of the same fields, at low degree and around $l = 1200$; the primitives against the former
  quadrature builder (`quad_reference.py`, $n$ up to 16). References evaluated on a grid have their own floor of about
  $\epsilon l^2$ near the poles, so their tolerance is max($10^{-11}$, $10^{-15}l^2$).
- **`tests/diff_rot/gen_check.py [mmax [nmax [lmax]]]`**: the couplings of `diff_rot.py` against the closed forms of
  `tests/diff_rot/diff_rot_coefficients.py`, relative to the largest coefficient of the same row block
  ($\le 1.3\times 10^{-15}$ at `lmax` 20 and 1025); the selection rules of §3.5 for $n$ = 4 to 16 (the coefficients
  outside them are exactly zero); and for runs with $P_0$, $P_2$ only and `thermal_wind` 1 or 2, the $\sigma_n$ of
  `diff_rot.sigma_coefficients` against (2.10) ($\le 2\times 10^{-16}$ of $r^2G^2\lvert\partial\Delta\Omega\rvert/(\mathrm{Beyonce}\thinspace g)$).
- **`tests/diff_rot/regress.sh OLD_KORE`**: $A$ and $B$ against an older checkout, bitwise or at round-off
  ($< 10^{-13}$ of each row's scale), on 15 configurations (including `'solar_P4'` and a `'user'` profile with degrees
  0, 2, 4). Each is written with both parameter namings (§4.1), and the one of the checkout is used.
- **`tests/heat_dr/`** (`run_all.sh`): `tw_sympy.py` (the closed forms (2.10) against (2.9)), `heat_check.py` (the heat
  operators against a grid quadrature, and the $\sigma_n$ fits against an independent integration of (2.9), any degrees), `rigid_thermal.py` (the identity (2.12) with the heat equation), `asm_check.py`,
  `row_check.py`, `ebal.py` (budgets from the matrix rows), `mpi_check.sh` (bitwise on 1–4 ranks), and others.
- **`spin_doctor` residuals** (§8), for every run: they close to round-off for a resolved mode, and are the check of
  resolution in $N$ and `lmax`. With `lmax` ≥ 646 they were NaN before `angular.ylm_grid` replaced scipy's harmonics.

## 10. Limitations

- The profiles no longer depend on the background model (only `solar_x` reads `model_type`), but the file-based models
  are broken on this branch for other reasons. `'Boussinesq'` works since 2026-10-23 (§11.4).
- The degree character of the profile codes limits the degrees to 35: $n_{\max} \le 16$ with `thermal_wind = 2`
  ($S_{tw}$ up to $2n_{\max} + 2 = 34$), $n_{\max} \le 32$ with `thermal_wind = 1`; `diff_rot.check` stops otherwise.
- `'user'` expressions are evaluated with Python's `eval` in a namespace without builtins: enough to stop accidents,
  not a sandbox for untrusted input.
- The inviscid formulation (`ViscosD = 0`, rows times $\rho_0^2$) builds the same labels with different weights, but is
  untested with differential rotation, and `ru_insane.py` does not cover it.
- With `thermal_wind = 1` the eigenvalues depend on the choice of frame at second order in $\Delta\Omega$;
  `thermal_wind = 2` is frame independent.
- The $p_b S_0'$ term of the thermal wind is omitted (§1.4); it matters where $S_0' \neq 0$, e.g. in a radiative interior
  (`solar_rz = 1`).
- With `ThermaD = 0` and the thermal wind, the boundary layer that a small `ThermaD` would form at fixed-entropy walls
  is absent, and small-scale growing modes peaking at $l \approx$ `lmax` can appear that do not converge in `lmax`.
- The matrices widen with the highest degree: the momentum rows couple $n_{\max} + 1$ degrees on each side, the heat
  rows $n_{tw}$ = $2n_{\max} + 2$ with `thermal_wind = 2`. `'tabulated'` with a large `diff_rot_nmax` is the most
  expensive case (§12). Sharp profiles (the `'solar'` tachocline at `ricb = 0`) need large N (about 400).

## 11. Benchmarks against published results

kore's differential rotation against five published sets of eigenvalues, from an incompressible fluid to the solar
convection zone with a radiative interior. Computed on a Mac (48 GB) on 2026-10-23 (Guenel et al.: 2026-10-25).

| paper | fluid | rotation profile | kore model |
| --- | --- | --- | --- |
| Baruteau & Rieutord 2013, JFM 719, 47 | incompressible, viscous | shellular $(r/R)^\sigma$, cylindrical $1 + \varepsilon s^2$ | `'Boussinesq'`, `thermal = 0` |
| Guenel, Baruteau, Mathis & Rieutord 2016, A&A 589, A22 | incompressible, viscous | conical $1 + \varepsilon\sin^2\theta$ | `'Boussinesq'`, `thermal = 0` |
| Mirouh, Baruteau, Rieutord & Ballot 2016, JFM 800, 213 | Boussinesq, stably stratified | $1 + \frac{N^2}{2}(1 - r^2)$ | `'Boussinesq'`, `heating = 'internal'` |
| Mukhopadhyay, Bekki, Zhu & Gizon 2025a, A&A (arXiv 2501.16797) | fully compressible, solar convection zone | their Edmond file | `'smoothed sun'`, Model S density, 'tabulated' |
| Mukhopadhyay, Bekki, Zhu & Gizon 2025b, A&A 708, A253 | as above, with a radiative interior | 'solar_P4' (their eqs. 5–6) | as above, `solar_rz = 1`, rin 0.5 R☉ |

### 11.1 Set-ups and conventions

**kore.** Fields go as $\exp(\lambda t + im\phi)$ in the frame rotating at $\Omega_{\rm ref}$, with
$\lambda = \sigma + i\omega$, $\sigma < 0$ damped and $\omega < 0$ prograde.

**Baruteau & Rieutord 2013.** Fields go as $\exp(i\Omega_p t + im\phi)$ in the inertial frame. They quote
$\lvert\omega\rvert = \lvert\mathrm{Re}\,\Omega_p\rvert$ and $\tau$ ($< 0$ damped) in units of $\Omega_{\rm ref}$,
where $\Omega_{\rm ref}$ is $\Omega$ at the surface (shellular) or on the axis (cylindrical).
- Conversion: $\Omega_p = \omega - m$ and $\tau = \sigma$. Their $m = -6$ mode is kore's $m = 6$ with
  $\Omega_p \to -\overline{\Omega_p}$.
- $E = \nu/(\Omega_{\rm ref}R^2)$ = `ViscosD`. Stress-free walls, $\eta = 0.35$ unless stated, equatorially
  symmetric modes (`symm = 1`) unless stated.
- kore: `diff_rot_type = 'shellular'` with `diff_rot_param` $= \sigma$ (ΔΩ $= r^\sigma - 1$), or `'cylindrical'` with
  `diff_rot_ampl` $= \varepsilon$ (ΔΩ $= \varepsilon r^2\sin^2\theta$). Uniform density (`model_type = 'Boussinesq'`),
  `thermal = 0`.

**Guenel et al. 2016.** The same equations and conventions as Baruteau & Rieutord 2013 (their eqs. 4–6), with
$\Omega = 1 + \varepsilon\sin^2\theta$ in units of the pole rate $\Omega_{\rm ref}$: kore's `'conical'` with
`diff_rot_ampl` $= \varepsilon$, in the frame rotating at the pole. Their figure labels give $\lvert\omega\rvert$ only. For
$m = 2$ the sign of $\Omega_p$ is the one in their text (Figs. 14, 15) or the only one inside the inertial range
$\lvert\Omega_p + m\Omega(\theta)\rvert < 2\Omega(\theta)$ (Fig. 13).

**Mirouh et al. 2016.** Their equations (2.7)–(2.9) are kore's with:
- `model_type = 'Boussinesq'` and `heating = 'internal'`: $\rho = p_0 = 1$, $g = r$, $dS/dr = r$, so the advection term
  is $r v_r$;
- `Beyonce` $= N^2$ and `ThermaD` $= E/P$, with $T = 0$ at both walls;
- `'shellular_boussinesq'` with `diff_rot_ampl` $= N^2$, and `thermal_wind = 0`. Their background is balanced by
  centrifugal buoyancy, which kore does not carry.

They also use the inertial frame: $\lambda = \tau + i\omega$, so $\omega = \omega_{\rm kore} - m$ and $\tau = \sigma$.

**Solar papers.**
- Frequencies are in nHz in the frame rotating at 456 nHz, with negative values retrograde, followed by the growth rate.
- Model S density, $\nu = \kappa = 10^{12}$ cm² s⁻¹, adiabatic convection zone, stress-free walls (set-ups in
  `docs/heat-equation.md` §8 and §9).
- "With thermal wind" means `thermal_wind = 2` (the exact $\Omega^2$) without
  $u_r\,\partial_r S_{tw}$ (`thermal_wind_radial = 0`).
- 2025a: shell 0.71–0.985 R☉, N 64, lmax $m + 63$. 2025b: rin 0.5 R☉, N 128.

### 11.2 Results

The deviation is kore minus published. For Baruteau and Mirouh it is in units of $\Omega_{\rm ref}$ for the frequency
and relative for $\tau$; for the solar papers it is in nHz. The resolution is kore's (N × lmax), with the paper's in
brackets where they differ.

#### Baruteau & Rieutord 2013 (incompressible)

| fig. | set-up | published $\lvert\omega\rvert$, $\tau$ | kore | Δω, Δτ/τ | N × lmax (paper) |
| --- | --- | --- | --- | --- | --- |
| 9 | shellular σ = 0 (rigid), m 0, E 1e-7 | 1.1037, −4.27e-3 | 1.10374, −4.2715e-3 | +4e-5, 0.0 % | 250 × 607 (250 × 600) |
| 9 | shellular σ = −0.5, m 0, E 1e-7 | 1.1890, −3.96e-3 | 1.18922, −3.9611e-3 | +2e-4, 0.0 % | 250 × 607 |
| 9 | shellular σ = −1.3, m 0, E 1e-7 | 1.3101, −1.28e-3 | 1.30997, −1.2869e-3 | −1e-4, 0.5 % | 250 × 607 |
| 10 | shellular σ = 0.5, η = 0.2, m 0, E 8e-7 | 1.2279, −7.86e-3 | 1.22791, −7.8551e-3 | 0, 0.1 % | 132 × 271 (132 × 264) |
| 11 | shellular σ = −1, m 0, E 1e-8 | 1.1113, −2.15e-3 | 1.11133, −2.1533e-3 | +3e-5, 0.2 % | 500 × 1007 (500 × 1000) |
| 11 | shellular σ = −0.5, m 0, E 1e-8 | 1.8993, −2.00e-3 | 1.89928, −1.9991e-3 | 0, 0.0 % | 580 × 1007 (580 × 1000) |
| 12 | shellular σ = 0.5, m 0, E 1e-8 | 0.9235, −3.99e-3 | 0.92349, −3.9899e-3 | 0, 0.0 % | 520 × 1007 (400 × 800) |
| 12 | shellular σ = −0.05, m 3 antisymmetric, E 1e-8 | Ω_p −2.5132, −3.77e-3 | −2.51322, −3.60e-3 | 0, 4.5 % | 200 × 402 (200 × 400); not converged |
| 15 | shellular σ = −0.2, m −6, E 6e-9 | Ω_p 7.1086, **+1.82e-4** | 7.10862, **+1.713e-4** | 0, 6 % | 500 × 1413 (500 × 1401) |
| 6 | cylindrical ε = −0.3, m 0, E 1e-8 | 0.9109, −7.45e-3 | 0.91090, −7.5128e-3 | 0, 0.8 % | 640 × 1215 (450 × 800) |
| 7 | cylindrical ε = −0.4, m 0, E 1e-8 | 1.3291, −6.05e-3 | 1.32891, −6.0588e-3 | −2e-4, 0.1 % | 450 × 815 (450 × 800) |
| 8 | cylindrical ε = −0.4, η = 0.05, m 0, E 1e-8 | 1.0995, −2.26e-3 | 1.09949, −2.2532e-3 | 0, 0.3 % | 560 × 1007 (450 × 800) |
| 13 | cylindrical ε = 0.4, m 2, E 2e-9 | Ω_p −2.2991, −4.47e-3 | not run | | (880 × 2400): ~190 GB |
| 14 | cylindrical ε = 0.4, m 2, E 5e-9 | Ω_p −2.2003, −3.28e-3 | not run | | (880 × 2400): ~190 GB |

Notes:
- Every frequency agrees to 2e-4 or better. Every damping rate agrees to 1 % or better once kore is resolved, except
  Fig. 12 (m 3) and Fig. 15.
- kore sometimes needs more than the paper's resolution for $\tau$. At the paper's resolution, Fig. 12 (m 0) gives
  −3.73e-3 (6.5 % off), Fig. 6 −7.06e-3 and Fig. 8 −2.54e-3 (12 % off).
- **Fig. 6** sits in a dense cluster: at 640 × 1215 there are two modes at ω = 0.91090, with τ = −7.51e-3 and
  −7.37e-3. The published −7.45e-3 lies between them, and at 360 × 655 and 560 × 1007 the nearest mode is a different
  one. The figure is matched to about 1 % but not uniquely identified.
- **Fig. 12, m 3** is the toroidal mode $\tilde\Omega_p = 2\Omega/(m+1)$ followed to σ = −0.05. Its frequency is
  matched to 2e-5, but the cluster around it does not settle: the nearest mode has τ = −3.60, −3.50, −3.80 and
  −3.95e-3 at N = 200, 280, 360 and 440, and an almost neutral mode appears at ω = 0.48681.
- **Fig. 15** is the growing mode. Its growth rate is linear in $E - E_c$ with $E_c \approx 10^{-8}$ (their Fig. 16),
  so 6 % in τ is a 0.4 % shift in $E_c$. kore reproduces the instability and the frequency to 2e-5.

#### Guenel et al. 2016 (incompressible, conical rotation)

| fig. | set-up | published $\lvert\omega\rvert$, $\tau$ | kore | Δω, Δτ/τ | N × lmax (paper) |
| --- | --- | --- | --- | --- | --- |
| 6 | ε = 0.3, η = 0.71, m 0, E 5e-8 | 1.1447, −3.60e-3 | 1.14469, −3.6020e-3 | 0, 0.1 % | 200 × 607 (200 × 600) |
| 7 | ε = −0.25, η = 0.71, m 0, E 1e-8 | 1.3552, −1.75e-3 | 1.35523, −1.7508e-3 | 0, 0.0 % | 300 × 1007 (300 × 1000) |
| 8 | ε = 0.3, η = 0.10, m 0, E 1e-8 | 1.6590, −2.93e-3 | 1.65911, −3.104e-3 | +1e-4, 6 % | 460 × 1103 (400 × 1000) |
| 8 | ε = −0.25, η = 0.71, m 0, E 1e-8, quasi-periodic | 1.2337, −1.02e-4 | 1.23372, −1.0200e-4 | 0, 0.0 % | 250 × 815 (250 × 800) |
| 12 | DT mode, ε = −0.25, η = 0.71, m 0, E 2.2e-8 | 1.8681, −4.93e-3 | 1.86824, −4.9423e-3 | +1e-4, 0.3 % | 296 × 751 (296 × 740) |
| 13 | DT mode, ε = −0.43, η = 0.35, m 2, E 1e-8 | Ω_p −3.2223, −8.11e-3 | −3.22283, −8.026e-3 | −5e-4, 1 % | 350 × 1009 (350 × 1001) |
| 15 | ε = −0.5, η = 0.71, m 2, E 1e-8, corotation | Ω_p −1.0697, **+4.62e-3** | −1.06970, **+4.619e-3** | 0, 0.0 % | 302 × 1425 (302 × 1421) |
| 14 | ε = 0.3, η = 0.71, m 2, E 1e-9, corotation | Ω_p −2.2951, −1.03e-2 | not run | | (458 × 2529): ~50 GB |
| §4.3.2 | DT mode, ε = −0.25, η = 0.71, m 0, E 1e-5 | "≈ 1.82" | 1.81734, −2.101e-2 | | 60 × 159 (60 × 150) |

Notes:
- Six of the seven quoted modes that were run agree to 1e-4 in frequency and to 1 % or better in τ, among them the
  growing mode of Fig. 15 and both turning-surface (DT) modes. The m 2 Fig. 13 is 5e-4 off in $\Omega_p$.
- **Fig. 8, η = 0.10** is 6 % off in τ, the same at 400 × 1007 (−3.08e-3) and 460 × 1103 (−3.10e-3), so kore is
  converged. The mode sits in a cluster: a neighbour at 1.65931 has τ = −2.98e-3.
- **Fig. 15** is the unstable mode with a corotation layer. kore returns a cloud of eigenvalues within $10^{-5}$ of it
  (+4.616 to +4.625e-3, all at ω = 0.93030), at every resolution tried (250 × 1201, 302 × 1425, 340 × 1425). The cloud
  is typical of a strongly non-normal eigenvalue. Its centre, +4.619e-3, does not move with resolution.
- **Fig. 13**: a second mode lies at $\Omega_p$ = −3.22240 with τ = −9.13e-3, closer in frequency and 12 % off in τ.
- **Fig. 14** needs about 50 GB at the paper's resolution.
- The E = 1e-5 DT mode is the starting point of their Fig. 12 sequence; they give its frequency only as "about 1.82".

#### Mirouh et al. 2016 (Boussinesq, stress-free walls)

The figure labels give $\lvert\omega\rvert$, τ and the resolution (Nr × L) of each mode; the runs below use that
resolution, rounded up to kore's lmax. The labels of Figs. 5 and 10 drop the sign of τ.

| source | set-up | published ω, τ | kore | Δω, Δτ/τ | N × lmax (paper) | residual |
| --- | --- | --- | --- | --- | --- | --- |
| Table 1 | m 4, N² 1.5, P 1e-2, E 1e-9 | −2.67162, −2.623e-3 | −2.67172, −2.548e-3 | −1e-4, 3 % | 420 × 275 (–) | 3e-12 |
| Table 1 = fig. 6(a) | m 0, N² 0.45, P 1e-2, E 1e-9 | 1.75431, −1.777e-3 | 1.754118, −1.7653e-3 | −2e-4, 0.7 % | 498 × 1007 (498 × 996) | 3e-11 |
| Table 1 = fig. 6(c) | m 0, N² 2.6, P 1e-2, E 1e-9 | 0.80413, −1.572e-3 | 0.804121, −1.5775e-3 | 0, 0.3 % | 400 × 815 (400 × 800) | 4e-12 |
| Table 1 | m 0, N² 2.3, P 1e-2, E 1e-10 | 0.30613, −8.884e-4 | 0.30592, −1.303e-3 | −2e-4, 47 % | 500 × 1007 (–); not converged | 2e-9 |
| fig. 5(a) | m 0, N² 0, P 1e-2, E 1e-7 | 1.6739, −6.31e-3 | 1.673897, −6.3115e-3 | 0, 0.0 % | 300 × 607 (300 × 600) | 5e-13 |
| fig. 5(b) | N² 0.45, otherwise as 5(a) | 1.7658, −9.44e-3 | 1.765788, −9.4443e-3 | 0, 0.0 % | 300 × 607 | 3e-12 |
| fig. 5(c) | N² 0.75 | 1.8220, −1.24e-2 | 1.822043, −1.2387e-2 | 0, 0.1 % | 300 × 607 | 6e-12 |
| fig. 5(d) | N² 0.79 | 1.8286, −1.36e-2 | 1.828603, −1.3579e-2 | 0, 0.2 % | 300 × 607 | 3e-12 |
| fig. 6(b) | m 0, N² 1.1, P 1e-2, E 1e-8, quasi-periodic | 1.4902, −3.82e-3 | 1.490236, −3.8266e-3 | 0, 0.2 % | 300 × 607 (300 × 600) | 3e-12 |
| fig. 8 | m 0, N² 3, P 1, E 1e-9, HT mode | 1.3856, −4.81e-4 | 1.385579, −4.6354e-4 | 0, 3.6 % | 498 × 1007 (498 × 996) | 6e-11 |
| fig. 9 | m 0, N² 6.7, P 1e-2, E 1e-10, HT mode, **growing** | 0.3296, +3.29e-2 | 0.329814, +3.2948e-2 | +2e-4, 0.1 % | 496 × 1007 (496 × 992) | 5e-10 |
| fig. 10 | m 0, N² 1.5, P 1e-2, E 1e-10, HT mode | 0.2515, (−)8.13e-4 | 0.251271, −8.2843e-4 | −2e-4, 2 % | 500 × 1007 (500 × 1000) | 2e-10 |
| fig. 11 | m 3, N² 6, P 1, E 1e-7, corotation, **growing** | −3.8612, +7.35e-3 | not identified | | 200 × 402 (200 × 400) | ~1e-7 |
| fig. 12 | m 4, N² 1.5, P 1e-2, E 1e-9, regular mode | −2.6715, −4.42e-5 | −2.67145, −4.4217e-5 | 0, 0.0 % | 398 × 803 (398 × 797) | 3e-11 |

Notes:
- Figure 7 shows damping rates against E only, no mode. All other modes of Figs. 5–12 are listed.
- Thirteen of the fourteen modes are identified. Their frequencies agree to 2e-4 or better, and τ to 1 % or better
  except Fig. 10 (2 %), Table 1 m 4 (3 %), Fig. 8 (3.6 %) and the unconverged E = 1e-10 row.
- **Fig. 12 is not the Table 1 m 4 mode**: same parameters, but τ is 60 times smaller. kore finds both.
- **Table 1 rows 2–3** are Figs. 6(a) and 6(c). At the figures' resolutions they agree to 0.7 % and 0.3 % in τ; coarser
  runs (420 × 815, 300 × 607) gave 6 % and 0.6 %.
- **Fig. 10**: kore's mode at 0.251271 is damped. Taking the unsigned label as a damping rate, as in Fig. 5, it matches
  to 2 %; there is no growing mode near the label. It sits in a cluster of modes within 2e-3 in ω.
- **Fig. 11** lies in a dense cluster of growing modes (corotation resonance) that changes with resolution: the
  nearest mode is (−3.86179, +1.108e-2) at 200 × 402 and (−3.86068, +7.247e-3) at 260 × 530.
- **Eigenvectors of Figs. 6(c), 8** (and weakly 6(a), 10) carry a toroidal component at $l \approx$ lmax, concentrated
  at the walls, where the energy density reaches $10^8$ times the bulk. It follows the truncation: for Fig. 6(c) with
  lmax 1007 the toroidal coefficients drop to 2e-7 near $l = 400$ and rise to 1.4e-2 at lmax. The eigenvalue is
  unaffected (lmax 815 → 1007: Δω 8e-6, Δτ 0.3 %, residual 2e-12). Plots of these modes need 11–12 decades to show
  the bulk structure.
- **E = 1e-10, Table 1** is beyond the resolution that fits in memory with three fields.

#### Mukhopadhyay et al. 2025a (convection zone, Edmond Ω): their compressible values

| mode (symm) | published | kore with thermal wind | Δ | kore without thermal wind | Δ |
| --- | --- | --- | --- | --- | --- |
| m 1 high-latitude (−1) | −95.70, +11.69 | −95.759, +11.301 | −0.06, −0.39 | −91.099, −3.697 | +4.6, −15.4 |
| m 3 Rossby n0 (−1) | −262.07, −0.88 | −262.052, −0.877 | +0.02, 0.00 | −263.037, −0.580 | −1.0, +0.3 |
| m 3 Rossby n1 (−1) | −182.05, −9.67 | −181.934, −9.700 | +0.12, −0.03 | −182.439, −9.838 | −0.4, −0.2 |
| m 3 columnar (+1) | 210.84, −11.09 | 210.908, −11.090 | +0.07, 0.00 | 213.585, −12.248 | +2.7, −1.2 |
| m 10 high-frequency retrograde (+1) | −211.28, −20.00 | −211.088, −19.963 | +0.19, +0.04 | −200.414, −19.267 | +10.9, +0.7 |

The thermal wind is part of their model, and kore matches only with it (to 0.2 nHz). Without it the high-latitude mode
is no longer unstable.

#### Mukhopadhyay et al. 2025b (radiative interior, rin 0.5 R☉)

They publish growth rates only for the two modes of their Table 2. "KE" is the fraction of kore's kinetic energy below
0.71 R☉.

| their mode | published | kore without thermal wind (KE) | Δ | kore with thermal wind | Δ |
| --- | --- | --- | --- | --- | --- |
| 4(b) mixed, m 5 | −136.740, −11.067 | −136.740, −11.073 (0.93) | 0.000, −0.006 | −136.736, −11.091 | +0.004, −0.024 |
| 5(b) mixed, m 7 | −240.068, −25.593 | −240.090, −25.593 (0.99) | −0.02, 0.000 | −240.252, −25.595 | −0.18, −0.002 |
| 3(b) mixed, m 2 | −333.7 | −333.685, −3.159 (0.98) | +0.02 | −333.675, −3.160 | +0.03 |
| 3(a) convection-zone Rossby, m 2 | −331.1 | −331.162, −0.584 (0.33) | −0.06 | −331.031, −0.733 | +0.07 |
| 4(a) convection-zone Rossby, m 5 | −137.9 | −137.923, −13.726 (0.03) | −0.02 | −131.334, −11.894 | +6.6 |
| m 3 equatorial Rossby (their §3.2) | −263.3 | −263.282, −4.467 (0.29) | +0.02 | −261.469, −4.215 | +1.8 |
| 5(a) convection-zone high-frequency retrograde, m 7 | −248.5 | −248.372, −19.736 (0.01) | +0.13 | −258.904, −21.332 | −10.4 |
| 1(b) mixed, m 1 | −94.1 | −94.018, −6.488 (0.96) or −94.217, −4.670 (0.98) | +0.08 / −0.12 | −94.188, −5.812 | −0.09 |
| 1(a) convection-zone high-latitude, m 1 | −92.2 | only strongly damped convection-zone modes, e.g. −91.951, −33.9 (0.02) | ? | −91.981, −7.932 (0.41) | +0.22 |
| 2(a) convection-zone high-latitude, m 2 | −177.2 | −173.950, −47.1 (0.00) or −175.074, −63.1 | ? | −181.423, −13.429 (0.06) | −4.2 |
| 2(b) mixed, m 2 | −187.1 | no mixed mode nearby | | no mixed mode nearby | |

Without the thermal wind, every convection-zone and mixed mode whose counterpart can be identified agrees to 0.13 nHz
or better. With it, the convection-zone modes were 2–10 nHz off. So the 2025b runs appear to have no latitudinal entropy
gradient, unlike 2025a, where kore matches only with it. Their high-latitude modes 1(a) and 2(a) remain unmatched. Without the
baroclinic term, kore's convection-zone candidates near them are damped by 34–63 nHz, and the paper gives no growth
rates to choose between them.

### 11.3 Summary

| paper | modes compared | frequencies | damping or growth rates |
| --- | --- | --- | --- |
| Baruteau & Rieutord 2013 | 12 of 14 (2 beyond memory) | all within 2e-4 Ω_ref | 10 of 12 within 1 % once resolved; Fig. 12 (m 3) not converged; Fig. 15 growing, 6 % |
| Guenel et al. 2016 | 7 of 8 (1 beyond memory) | all within 1e-4 Ω_ref (Fig. 13: 5e-4) | 6 within 1 %, including the growing mode; Fig. 8 (η = 0.10) 6 % |
| Mirouh et al. 2016 | 14 (Table 1 and Figs. 5, 6, 8–12) | 13 identified, within 2e-4 | 9 within 1 %, 3 within 2–3.6 %; Table 1 E = 1e-10 not resolved; Fig. 11 not identified |
| Mukhopadhyay et al. 2025a | 5 | within 0.2 nHz (with thermal wind) | within 0.4 nHz |
| Mukhopadhyay et al. 2025b | 11 | 8 within 0.13 nHz (without thermal wind); 2 high-latitude unmatched; 1 with no counterpart | the two published, within 0.006 nHz |

### 11.4 Reproducing

- Baruteau, Guenel and Mirouh: `tests/diff_rot/lit_bench.py WORKDIR CASE [N=.. lmax=.. ncpus=8 nev=..]`. `list` prints the
  cases. The largest run here (Fig. 6 at 640 × 1215) took 26 GB and 5.5 minutes on 8 ranks.
- Solar: `tests/diff_rot/solar_checks.py EDMOND_NPZ [WORKDIR]`, about 40 minutes on one core.
- `model_type = 'Boussinesq'` is the uniform-density model these runs need ($\ln\rho_0 = 0$, $p_0 = 1$, $g = r$). Its
  $dS/dr$ comes from `heating`: `'internal'` gives $r$, `'differential'` gives $1/r^2$, `'top hat'` gives the aux1–aux5
  erf top hat.
