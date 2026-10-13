# Differential rotation in the heat equation: derivation, implementation, benchmarks

kore has long included a differential rotation of the background in the momentum equation. This page explains why it
must also enter the heat (entropy) equation, and records how that was derived, coded and tested. If you only want to
run the code, start with the user guide, [differential-rotation.md](differential-rotation.md), which lists the
parameters, profiles and examples. This page is the reference behind it.

The physics in two sentences. A perturbation is carried around by the mean flow, so its entropy is advected by the
differential rotation. And a star whose rotation rate varies with latitude and depth is only in equilibrium if its
entropy also varies with latitude (thermal-wind balance); the flow of a perturbation across that latitudinal entropy
gradient changes its entropy, which feeds back on the motion through buoyancy.

How the page is organised:

- [Notation and code names](#notation-and-code-names) defines the symbols and the kore jargon used below.
- §0 states the result: three new terms in the heat equation.
- §1 derives the background entropy $S_{tw}$ that thermal-wind balance requires.
- §2 writes the three terms in kore's spherical-harmonic form.
- §3 explains how they map onto kore's matrices and operator labels.
- §4 lists the published benchmarks and which terms each needs.
- §5 records the implementation and its tests.
- §6 compares with the solar inertial-mode calculations of Bekki et al. 2022 and Mukhopadhyay et al. 2025.
- §7 lists what is still open.
- §8 generalises the rotation profile to any Legendre degree ('solar_P4', 'tabulated') and repeats the solar benchmark
  like for like.

History. The terms were derived on 2026-10-11 and committed in three steps on `anelastic_log`:

1. the heat-equation terms (`be2a4db5`);
2. the `Gaspard` factors, the exact and prescribed thermal wind, `ThermaD = 0` (with [thermad0.md](thermad0.md)) and
   the `spin_doctor` diagnostics (`18641206`);
3. any $P_n$, 'solar_P4', 'tabulated' and the amplitude controls (`92a0d020`, §8).

The checks are in `tests/heat_dr/` and `tests/diff_rot/`.

## Notation and code names

Everything is dimensionless, in kore units: the length unit $R$ is the outer radius of the shell, and the time unit is
$1/\Omega_0$ unless stated otherwise (see `Gaspard` below).

| symbol | meaning | name in the code |
| --- | --- | --- |
| $r$, $\theta$, $\phi$ | spherical coordinates; $x = \cos\theta$ | |
| $\Omega_0$, $\Omega_{\rm ref}$ | rotation rate of the frame in which kore solves | `diff_rot_frame` (in nHz, solar profiles) |
| $G$ | $\Omega_{\rm ref}\tau$, the frame's rotation rate in kore's time unit $\tau$ (the Coriolis factor); 1 with `timescale = 'rotation'` | `Gaspard` |
| $\Delta\Omega(r, \theta)$ | relative differential rotation: $\Omega = \Omega_0\thinspace (1 + \Delta\Omega)$ | chosen by `diff_rot_type` |
| $f_n(r)$ | Legendre coefficients of $\Delta\Omega$ in $\cos\theta$ | `abu` ($f_0$), `aub` ($f_2$), `ab4`, … |
| $\mathbf U_0$ | the azimuthal mean flow of $\Delta\Omega$, seen in the rotating frame | |
| $s$ | entropy perturbation, $\delta s/c_p$ | the unknown of section h |
| $S_0(r)$ | background entropy (spherical part); $S_0'$ its radial gradient | `dSd`; `pdS` is $p_0 S_0'$ |
| $S_{tw}(r, \theta)$ | latitudinal background entropy required by thermal-wind balance (§1) | |
| $\sigma_n(r)$ | Legendre coefficients of $S_{tw}$ in $\cos\theta$ | `tw2` is $p_0\sigma_2$, `tp2` is $p_0\sigma_2'$, … |
| $\rho_0$, $T_0$, $p_0$ | background density, temperature, and $p_0 = \rho_0 T_0$ | `pss` is $p_0$ |
| $h$ | $\ln\rho_0$ | `lho` |
| $g(r)$ | gravity profile | `gra` |
| Beyonce | buoyancy scale: the buoyancy force is $\mathrm{Beyonce}\thinspace g\thinspace s\thinspace \hat{\mathbf e}_ r$ | `Beyonce` |
| ThermaD, $\kappa$ | thermal diffusion number and diffusivity profile | `ThermaD`; `kps` is $\kappa p_0$ |
| $P$, $T$ | poloidal and toroidal scalars of the velocity $\mathbf u$ | sections u and v |
| $l$, $m$, $L$ | harmonic degree, azimuthal order, $L = l(l+1)$ | `l`, `m` |
| $i\thinspace m$ | the imaginary unit times $m$: fields go as $e^{i m\phi}$, so $\partial_\phi \to i\thinspace m$ | |
| $l'$, $\Delta l$ | target degree (matrix row) and $\Delta l = l' - l$ | `offd` |
| $S_l^m$ | Schmidt-normalised spherical harmonics | `ut.Ylm` |

Code concepts used on this page:

- **Sections.** kore stacks its unknowns in blocks: u (poloidal scalar $P$, from the radial component of the curl of
  the curl of the momentum equation), v (toroidal scalar $T$, from the radial component of the curl) and h (the
  entropy $s$, from the heat equation). "Section h" means the matrix rows that hold the heat equation.
- **Eigenproblem.** kore solves $A\mathbf x = \lambda B\mathbf x$. The real part of $\lambda$ is the growth rate, and the
  frequency is minus its imaginary part.
- **Row weights.** Before projection each equation is multiplied by a power of $r$ (and in section h by $p_0$), so
  that its radial coefficients are polynomials and the matrices are banded. Every operator is also divided by $L^3$, a
  normalisation common to all of kore's operators.
- **Gegenbauer bases.** The rows are projected on Gegenbauer polynomials $C^{(k)}$ (`gebasis` in `submatrices.py`). A
  row with radial derivatives up to order $k$ needs at least the $C^{(k)}$ basis; section h uses $C^{(2)}$ by default.
- **Operator labels.** Each radial operator is named by a short string, read by `utils.decode_label`. For example
  `h1tw20_D0` means: section `h`; `1` lowers the row's power of $r$ by one; the profile `tw2` with derivative order
  `0`; and `_D0`, no radial derivative of the unknown. A second profile can follow the first (`h1tw20lho1_D0` is
  $r\thinspace p_0\sigma_2\thinspace h'$ acting on $P$).
- **Parity in a full sphere.** With no inner core (`ricb = 0`) kore expands each radial function in Chebyshev
  polynomials of one parity in $r$. Every profile must then be even or odd in $r$, and the odd ones are listed in the
  `adj` lists of `submatrices.py`.
- **Symmetry.** `symm` = ±1 selects the equatorial symmetry of the mode; it fixes which degrees $l$ appear for each
  unknown (`ut.ell`).
- **LBR.** The Lantz–Braginsky–Roberts form of the anelastic approximation, in which the buoyancy depends on the
  entropy perturbation only.

## 0. Result in brief

kore's heat equation is the set of rows of section h, built by `operators.entropy`, `thermal_advection` and
`thermal_diffusion`:

$$
p_0\thinspace \frac{\partial s}{\partial t} = -p_0\thinspace u_r S_0' + \mathrm{ThermaD}\thinspace \nabla\cdot(\kappa\thinspace p_0\nabla s), \qquad p_0 = \rho_0 T_0.
$$

In words: the entropy perturbation changes because the radial flow carries fluid across the background entropy
gradient, and it diffuses.

The rows are multiplied by $r^2$. Without thermal diffusion (ThermaD = 0) they are also divided by $p_0$, and the
weight is $r$ instead of $r^2$ unless the thermal wind is on (§3 and [thermad0.md](thermad0.md)).

Now let the background rotate differentially,

$$
\Omega = \Omega_0\thinspace [1 + \Delta\Omega(r, \theta)], \qquad \Delta\Omega = f_0(r) + f_2(r)\thinspace P_2(\cos\theta).
$$

(§0 to §7 use this $P_0 + P_2$ form; §8 extends it to any even degree.) Two things change.

- The perturbation is advected by the mean flow of the differential rotation,

$$
\mathbf U_0 = \Delta\Omega\thinspace r\sin\theta\thinspace \hat{\mathbf e}_ \phi .
$$

- The background entropy acquires a latitudinal part $S_{tw}(r, \theta)$, fixed by thermal-wind balance (§1).

Linearising

$$
\frac{\partial s}{\partial t} + \mathbf U_0\cdot\nabla s + \mathbf u\cdot\nabla(S_0 + S_{tw}) = \text{diffusion}
$$

adds three terms to the left-hand side of the heat equation:

$$
p_0\left[\thinspace i\thinspace m\thinspace \Delta\Omega\thinspace s + u_r\thinspace \frac{\partial S_{tw}}{\partial r} + \frac{u_\theta}{r}\thinspace \frac{\partial S_{tw}}{\partial\theta} \right]. \tag{1}
$$

They are, in order:

- **(a) advection.** The perturbation is carried around at the local rotation rate. On its own this only shifts
  frequencies (it is the Doppler shift of a rigid rotation, made local).
- **(b) radial.** The radial velocity moves fluid across the radial gradient of $S_{tw}$.
- **(c) latitudinal.** The meridional velocity moves fluid across the latitudinal gradient of $S_{tw}$. This is the
  baroclinic coupling. In §2 it splits into (c), from $P$, and (c′), from $T$.

The first term follows from the linearisation alone. The other two come from the background entropy, which in kore's
framework is the thermal wind of §1 (kore's background is spherical in $\rho_0$, $T_0$ and $g$, has no mean
centrifugal term, and uses LBR buoyancy). Terms (b) and (c) must be switchable (`thermal_wind`), because the
Boussinesq benchmark of Mirouh et al. 2016 imposes $\Omega(r)$ with a purely radial $T_0$ (§4).

Nothing else changes:

- The momentum equation keeps its present diff-rot term. The background pressure is a gradient and drops out of the
  curled equations, and the buoyancy stays $\mathrm{Beyonce}\thinspace g\thinspace s\thinspace \hat{\mathbf e}_ r$.
- The diffusion acts on $s$ only: $S_{tw}$ is assumed to be maintained, like $S_0$.
- The mean flow does not advect $S_{tw}$, because $S_{tw}$ does not depend on $\phi$: $\mathbf U_0\cdot\nabla S_{tw} = 0$.
- There is no viscous heating in kore.

§2 gives the terms in kore's harmonic form, §3 what they need in the code.

## 1. Thermal-wind entropy S_tw

A rotation rate that varies along the rotation axis cannot be balanced by pressure alone. The centrifugal and Coriolis
forces of the differential rotation then have a curl, and in a stratified fluid only a buoyancy force that varies with
latitude can balance it. This is thermal-wind balance. This section computes the latitudinal entropy $S_{tw}$ that
kore's background must therefore have, and estimates its size.

### The balance

Consider the steady background in the rotating frame, in kore units: time $1/\Omega_0$, length $R$, and buoyancy
$\mathrm{Beyonce}\thinspace g(r)\thinspace s\thinspace \hat{\mathbf e}_ r$, so that

$$
N^2 = \mathrm{Beyonce}\thinspace g\thinspace \frac{dS}{dr}.
$$

The momentum balance of the mean flow is

$$
\mathbf U_0\cdot\nabla\mathbf U_0 + 2\hat{\mathbf z}\times\mathbf U_0 = -\nabla p_b + \mathrm{Beyonce}\thinspace g\thinspace S_{tw}\thinspace \hat{\mathbf e}_ r .
$$

Its left side, the centrifugal and Coriolis forces of the differential rotation, is

$$
-\left[(1+\Delta\Omega)^2 - 1\right] r\sin\theta\thinspace \hat{\mathbf e}_ s ,
$$

where $\hat{\mathbf e}_ s$ is the cylindrical radial unit vector. Taking the curl removes the pressure, and its
$\phi$-component gives

$$
\frac{1}{r}\frac{\partial S_{tw}}{\partial\theta} = \frac{r\sin\theta}{\mathrm{Beyonce}\thinspace g(r)}\thinspace \frac{\partial (1+\Delta\Omega)^2}{\partial z},
\qquad \frac{\partial}{\partial z} = \cos\theta\thinspace \frac{\partial}{\partial r} - \frac{\sin\theta}{r}\frac{\partial}{\partial\theta}. \tag{2}
$$

This is the thermal-wind equation in kore's variables. In dimensional form it reads

$$
r\sin\theta\thinspace \frac{\partial\Omega^2}{\partial z} = \frac{g}{c_p\thinspace r}\thinspace \frac{\partial S}{\partial\theta},
$$

which is eq. (27) of Bekki, Cameron & Gizon 2022 (A&A 662, A16). The $l = 0$ part of $S_{tw}$ is absorbed in $S_0(r)$.

### Linear and exact forms

To first order in $\Delta\Omega$, replace $(1+\Delta\Omega)^2$ by $1 + 2\Delta\Omega$. Then (2) integrates in closed form:

$$
\begin{aligned}
S_{tw} &= \sigma_2(r)\thinspace P_2(\cos\theta) + \sigma_4(r)\thinspace P_4(\cos\theta),\cr
\sigma_2 &= -\frac{2r^2}{\mathrm{Beyonce}\thinspace g}\left[\frac{f_0'}{3} + \frac{f_2'}{21} + \frac{4}{7}\frac{f_2}{r}\right],\cr
\sigma_4 &= -\frac{6r^2}{35\thinspace \mathrm{Beyonce}\thinspace g}\left[f_2' - \frac{2f_2}{r}\right].
\end{aligned} \tag{3}
$$

This linear form is `thermal_wind = 1`.

Keeping the $\Delta\Omega^2$ term gives the exact form, `thermal_wind = 2`. It changes $\sigma_2$ and $\sigma_4$ by
products of the profiles and adds a $P_6$ component. In addition, when kore's time unit $\tau$ is not $1/\Omega_{\rm ref}$, so that
$G$ = `Gaspard` = $\Omega_{\rm ref}\tau \neq 1$, the right side of (2) gets a factor $G^2$. With $e = 0$ for `thermal_wind = 1` and $e = 1$ for
`thermal_wind = 2`, the closed forms are (checked by `tests/heat_dr/tw_sympy.py`):

$$
\begin{aligned}
\sigma_2 &= -\frac{2r^2G^2}{\mathrm{Beyonce}\thinspace g}\left[(1+e f_0)\left(\frac{f_0'}{3} + \frac{f_2'}{21} + \frac{4}{7}\frac{f_2}{r}\right) + \frac{e\thinspace f_2\thinspace (f_0' + f_2')}{21}\right],\cr
\sigma_4 &= -\frac{6r^2G^2}{35\thinspace \mathrm{Beyonce}\thinspace g}\left[(1+e f_0)\left(f_2' - \frac{2f_2}{r}\right) + e\left(f_2 f_0' + \frac{4}{11} f_2 f_2' + \frac{14}{11}\frac{f_2^2}{r}\right)\right],\cr
\sigma_6 &= -e\thinspace \frac{4r^2G^2}{77\thinspace \mathrm{Beyonce}\thinspace g}\thinspace f_2\left(f_2' - \frac{2f_2}{r}\right).
\end{aligned} \tag{3'}
$$

How much the exact form matters. For the conical profile below, it multiplies $\sigma_2$ by $1 + 2\varepsilon/3$ and
$\sigma_4$ by $1 + 12\varepsilon/11$: 17 % and 27 % for $\varepsilon = -0.25$. The $\sigma_6$ term needs

$$
P_6(X) = \frac{231X^6 - 315X^4 + 105X^2 - 5}{16}
$$

and couplings $\Delta l = \pm 6$ (for $P$) and $\pm 5$ (for $T$) in §2 and §3.

Frame independence. Only the exact form is independent of the frame: a run in a frame rotating at
$\Omega_{\rm ref}(1 + c)$, with $f_0$ shifted by $-c$, gives the same modes. With the linear form the choice of frame
moves the modes slightly (§6).

### Checks on simple profiles

- **Cylindrical rotation**, constant on cylinders: $\Delta\Omega = \varepsilon\thinspace r^2\sin^2\theta$, that is
  $f_0 = (2/3)\thinspace \varepsilon r^2$ and $f_2 = -(2/3)\thinspace \varepsilon r^2$. It gives $\sigma_2 = \sigma_4 = 0$ in both
  versions, as it must: a barotropic rotation needs no thermal wind.
- **Conical rotation**, constant on radial lines: $\Delta\Omega = \varepsilon\sin^2\theta$. It gives the values below.
  For $\varepsilon > 0$ (equator faster, as in the Sun's convection zone) the poles are warmer than the equator, the
  known solar sense.

$$
\sigma_2 = \frac{16\thinspace \varepsilon r}{21\thinspace \mathrm{Beyonce}\thinspace g}, \qquad
\sigma_4 = -\frac{8\thinspace \varepsilon r}{35\thinspace \mathrm{Beyonce}\thinspace g}, \qquad
S_{tw}(\text{pole}) - S_{tw}(\text{equator}) = \frac{\varepsilon r}{\mathrm{Beyonce}\thinspace g} > 0 .
$$

- **The 'solar' fit**, a tanh step at the tachocline radius `rtc`. Here $f_0'$ and $f_2'$ are localised at the
  tachocline. The $(4/7)\thinspace f_2/r$ and $-(12/35)\thinspace f_2/r$ parts of (3) fill the convection zone, where kore's $S_0'$ is
  zero, so there $S_{tw}$ is the only background entropy gradient.

### How large is the thermal wind

Take the solar case: Beyonce = 4.9e4, $f_2 = -0.198$, $r \approx 0.9$, $g \approx 1$. Then $\sigma_2 \approx 4\times10^{-6}$
and $\sigma_4 \approx -10^{-6}$. The pole-minus-equator contrast of $S_{tw}$ is about $5\times10^{-6}$ in units of $c_p$
(kore's $s$ is $\delta s/c_p$). With $c_p \approx 3.5\times10^{8}$ that is $\approx 1.9\times10^{3}$ erg g⁻¹ K⁻¹, the
thermal-wind value at the top of the range Bekki et al. scan.

The small number is misleading. $S_{tw}$ scales as $1/(\mathrm{Beyonce}\thinspace g)$, while the buoyancy it produces is
$\mathrm{Beyonce}\thinspace g\thinspace s$. So in the convection zone, where $S_0' = 0$, the entropy perturbation driven by term (c) feeds
back on the momentum equation at order $\Delta\Omega$ relative to the Coriolis force, independently of Beyonce. That is
the same order as the momentum diff-rot term. The thermal-wind terms are therefore leading order for convection-zone
modes, not a correction: this is the baroclinic coupling that destabilises the high-latitude modes in Bekki et al.

### Prescribed latitudinal entropy (thermal_wind = 3)

Bekki et al. 2024 (and Bekki et al. 2022, eq. 26) impose a latitudinal entropy gradient that is independent of
$\Omega$ and uniform in $r$:

$$
\frac{\partial s}{\partial\theta} = \Delta_\theta s\thinspace \sin 2\theta .
$$

Since $\sin 2\theta = -(2/3)\thinspace dP_2/d\theta$, this is $S_{tw} = \sigma_2 P_2$ with a constant $\sigma_2$ and nothing else:

$$
\sigma_2 = -\tfrac{2}{3}\thinspace \mathtt{dSdtheta}, \qquad \sigma_4 = \sigma_6 = 0, \qquad \mathtt{dSdtheta} = \Delta_\theta s/c_p .
$$

kore's $s$ is $\delta s/c_p$, and $\Delta_\theta s < 0$ gives warm poles. Because $\sigma_2$ is constant in $r$, term (b)
vanishes and only (c) and (c′) act. $\Omega$ still enters through (a) and the momentum equation. For uniform rotation
use `diff_rot_type = 'rigid'` with `diff_rot_amplitude = 0`.

### What is neglected

$S_{tw}$ is the thermal wind consistent with LBR. The full anelastic balance would add

$$
\frac{S_0'}{\rho_0\thinspace g}\thinspace p_b
$$

to $S_{tw}$. This is the same $p_1 S_0'$ term that kore drops for the perturbations, so it matters only where
$S_0' \neq 0$ (the radiative zone).

Also neglected, consistently with kore's spherical background, are three effects of relative size
$\Delta\Omega/(\mathrm{Beyonce}\thinspace g)$ in the anelastic case, negligible for the Sun (Beyonce ≈ 5e4):

- the perturbation density in the background centrifugal acceleration;
- the $\theta$-dependence of $\rho_0 T_0$ and $g$ induced by the thermal wind;
- the thermal-wind density $\rho_{tw}$ in the mass flux.

## 2. The three terms in kore's harmonic form

kore expands every field in spherical harmonics of one azimuthal order $m$ and solves for a radial function per degree
$l$. Multiplying by a function of latitude, such as $\Delta\Omega$ or $S_{tw}$, mixes neighbouring degrees. This section
writes the three terms as such couplings. All of them are built from two small matrices, $X$ and $D$.

### Fields

The entropy and the velocity are

$$
s = \sum_l s_l(r)\thinspace S_l^m, \qquad
\mathbf u = \rho_0^{-1}\nabla\times\nabla\times(\rho_0 P\thinspace \mathbf r) + \nabla\times(T\thinspace \mathbf r),
$$

with $S_l^m$ the Schmidt harmonics of `ut.Ylm`, $L = l(l+1)$ and $h = \ln\rho_0$. The terms need two velocity
components, for one degree:

$$
u_r = \frac{L P}{r}\thinspace S_l, \qquad
u_\theta = \frac{1}{r}\left[\partial_r(rP) + r h' P\right]\partial_\theta S_l + \frac{i\thinspace m T}{\sin\theta}\thinspace S_l .
$$

### Angular coupling matrices

With $x = \cos\theta$, two recurrences define matrices on the degree index at fixed $m$ (row: target degree $l'$;
column: source degree $l$):

$$
\begin{aligned}
x\thinspace S_l &= p_l\thinspace S_{l+1} + q_l\thinspace S_{l-1} &&\to X,\cr
(1 - x^2)\thinspace \frac{dS_l}{dx} &= (l+1)\thinspace q_l\thinspace S_{l-1} - l\thinspace p_l\thinspace S_{l+1} &&\to D,\cr
p_l &= \frac{\sqrt{(l+1)^2 - m^2}}{2l+1}, \qquad q_l = \frac{\sqrt{l^2 - m^2}}{2l+1}.
\end{aligned}
$$

Multiplying by a polynomial in $\cos\theta$ is then the same polynomial of the matrix $X$:

$$
P_2(X) = \frac{3X^2 - 1}{2}, \qquad P_4(X) = \frac{35X^4 - 30X^2 + 3}{8}, \qquad
P_2'(X) = 3X, \qquad P_4'(X) = \frac{35X^3 - 15X}{2}.
$$

For instance, the non-zero entries of $P_2(X)$ in column $l$ are

$$
P_2(X)_ {l,l} = \tfrac12\left[3(p_l\thinspace q_{l+1} + q_l\thinspace p_{l-1}) - 1\right], \qquad
P_2(X)_ {l+2,l} = \tfrac32\thinspace p_{l+1}\thinspace p_l, \qquad
P_2(X)_ {l-2,l} = \tfrac32\thinspace q_{l-1}\thinspace q_l .
$$

In general they are Gaunt coefficients,

$$
X_{l'l} = \sqrt{\frac{2l'+1}{2l+1}}\thinspace \sqrt{\frac{4\pi}{3}}\thinspace (-1)^m\thinspace \mathrm{gaunt}(l',1,l,-m,0,m),
$$

and the same with $n = 2, 4$ in place of 1 for $P_n(X)$.

### The terms

Row $l'$ of the heat equation gains the following terms, before the $r^2 p_0$ row weight. $L_l$ is that of the source
degree $l$.

$$
\begin{aligned}
\text{(a) advection:}\quad & i\thinspace m G\thinspace \left[f_0\thinspace \delta_{l'l} + f_2\thinspace P_2(X)_ {l'l}\right] s_l ,\cr
\text{(b) radial:}\quad & \frac{L_l P_l}{r}\left[\sigma_2'\thinspace P_2(X)_ {l'l} + \sigma_4'\thinspace P_4(X)_ {l'l}\right],\cr
\text{(c) latitudinal, from } P\text{:}\quad & \frac{P_l + r P_l' + r h' P_l}{r^2}\left[\sigma_2\thinspace (3XD)_ {l'l} + \sigma_4\left(\tfrac12(35X^3 - 15X)D\right)_ {l'l}\right],\cr
\text{(c}'\text{) latitudinal, from } T\text{:}\quad & -\frac{i\thinspace m T_l}{r}\left[\sigma_2\thinspace (3X)_ {l'l} + \sigma_4\left(\tfrac12(35X^3 - 15X)\right)_ {l'l}\right].
\end{aligned}
$$

| term | couples | $\Delta l$ |
| --- | --- | --- |
| (a) advection | $s \to s$ | $0, \pm 2$ |
| (b) radial | $P \to s$ | $0, \pm 2, \pm 4$ |
| (c) latitudinal, $P$ | $P \to s$ | $0, \pm 2, \pm 4$ |
| (c′) latitudinal, $T$ | $T \to s$ | $\pm 1, \pm 3$ |

Remarks:

- With `thermal_wind = 2` the sums run to $\sigma_6$, and $\Delta l$ goes up to $\pm 6$ for $P$ and $\pm 5$ for $T$. The
  extra polynomial is given below.
- The even $\Delta l$ for $P$ and odd $\Delta l$ for $T$ match the equatorial-symmetry classes of `ut.ell` (section h
  uses the poloidal set of degrees). So the block structure is that of the momentum diff-rot blocks, with
  $\Delta l = \pm 4$ added.
- For $m = 0$ the terms (a) and (c′) vanish. The couplings into $l = 0$, which is absent from kore's basis, are
  harmless: nothing feeds back from $s_0$.
- Build $P_2(X)$, $P_4(X)$ and $P_n'(X)D$ from $X$ and $D$ with about five degrees more than lmax, and cut afterwards.
  Formed at lmax they are wrong in the top degrees (errors of order one in the entries).
- Term (a) is anti-Hermitian. In the entropy budget it only shifts the frequency, by an amount set by
  $\int p_0\thinspace \Delta\Omega\thinspace \lvert s\rvert^2\thinspace dV$, while (b) and (c) exchange entropy "energy" with the flow.

The polynomial needed for $\sigma_6$ is

$$
P_6'(X) = \frac{1386X^5 - 1260X^3 + 210X}{16}.
$$

### Verification

sympy, 0 failures; `tests/heat_dr/tw_sympy.py` and `heat_check.py` repeat the main checks:

- (3) against direct integration of (2);
- the curl of the background balance vanishes with $S_{tw}$ in place (linear and exact);
- the barotropic check of §1;
- the matrices $X$, $D$, $P_2(X)$, $P_4(X)$, $P'(X)D$ against quadrature projections for $m = 0\ldots 3$;
- (b), (c), (c′) against the full 3-D $\mathbf u\cdot\nabla S_{tw}$ built from $P$, $T$ and $\ln\rho_0$ with sympy and
  projected on $S_{l'}$, for $m = 0, 1, 2$ and every $\Delta l$.

Independent readers redid the thermal wind with full spherical vector calculus, the linearisation, the couplings with
Wigner 3j symbols, and the velocity sign conventions against kore's assembled Coriolis operator. None found a
discrepancy.

## 3. What kore needs

This section maps the terms of §2 onto kore's building blocks: radial profiles (functions of $r$ known on the radial
grid), operator labels, and the assembly loops.

### Radial profiles

- $f_0$ and $f_2$ exist already (`abu`, `aub`).
- New are $\sigma_2$ and $\sigma_4$ from (3). They need $g$, $g'$, $f'$ and $f''$: `gra`, and `spv`/`svp` and
  `psl`/`pls`, which give $r f'$ and $r^2 f''$.
- Following `pdS` = $p_0\thinspace dS/dr$ and `kps` = $\kappa p_0$, the code uses the products `tw2` = $p_0\sigma_2$, `tw4` =
  $p_0\sigma_4$, `tp2` = $p_0\sigma_2'$ and `tp4` = $p_0\sigma_4'$. Separate `tp` profiles are needed because a label
  such as `h1tw21_D0` would mean $r\thinspace (p_0\sigma_2)'\thinspace P$, the derivative of the product, which is not (b).

### Parity in a full sphere

At `ricb = 0`, $\sigma_2$ and $\sigma_4$ are even in $r$ ($f$ is even, $f'$ and $g$ are odd). So `tw2` and `tw4` are
even, like all other profiles. But `tp2` and `tp4` are odd. They must be added to the `adj` lists of `submatrices.py`
(lines 286 and 340; also `tests/heat_dr/ebal.py` and `ru_insane.row_parity`), next to `gra`, `pdS` and `dSd`.
Otherwise term (b) is silently dropped in full-sphere runs. (The 'solar' profile's parity defect in $r$ is fixed by
evaluating the profiles at $\lvert r\rvert$, §5.)

### Operators and labels

Like the other section-h operators, the rows are multiplied by $r^2 p_0$ and divided by $L_{l'}^3$:

| term | operator | labels | column, $\Delta l$ |
| --- | --- | --- | --- |
| (a) | $i\thinspace m\thinspace r^2 p_0 f_0\thinspace s$ and $i\thinspace m\thinspace r^2 p_0 f_2\thinspace P_2(X)\thinspace s$ | `h0pss0abu0_D0`, `h0pss0aub0_D0` | $s$; $0, \pm 2$ |
| (b) | $L_l\thinspace r\thinspace p_0\thinspace \sigma_n'\thinspace P_n(X)\thinspace P$ | `h1tp20_D0`, `h1tp40_D0` | $P$; $0, \pm 2, \pm 4$ |
| (c) | $p_0\thinspace \sigma_n\left[P + r P' + r h' P\right] P_n'(X)\thinspace D$ | `h2tw20_D0`, `h1tw20_D1`, `h1tw20lho1_D0` (and tw4) | $P$ |
| (c′) | $-i\thinspace m\thinspace r\thinspace p_0\thinspace \sigma_n\thinspace P_n'(X)\thinspace T$ | `h1tw20_D0`, `h1tw40_D0` | $T$; $\pm 1, \pm 3$ |

All twelve labels parse with `utils.decode_label` (the 13-character two-function form for the first and last of (c)).

### Signs and factors

The section-h rows of $A$ hold the right-hand side of the heat equation: `thermal_advection` enters as
$-r p_0 S_0' u_r$, with $B$ = +`entropy`. So the three left-hand-side terms of (1) go into $A$ with a minus sign. This is
opposite to the momentum rows, where $B$ = −`inertia` and the diff-rot block is added with +.

- (a) carries the factor `Gaspard`, like the momentum diff-rot operator. Both had none before the second round, so only
  `timescale = 'rotation'` was right.
- (b), (c) and (c′) carry $G^2$ through $\sigma_n$ (but not with `thermal_wind = 3`).
- `thermal_wind = 0/1/2/3` selects whether (b), (c) and (c′) are included, and in which form; (a) always is.

### Assembly

- In `submatrices.py` the $T$-column labels are registered with vector parity `vT`, the others with `vP`.
- `assemble.py` section h used to call only diagonal operators. It needs an $s$-column loop over
  $\Delta l \in \lbrace -2, 0, 2\rbrace$, a $P$-column loop over $\lbrace -4, \ldots, 4\rbrace$ and a $T$-column loop over
  $\lbrace -3, -1, 1, 3\rbrace$. These use the `offd` of `operators.differential_rotation`: for $P$ and $s$, offd = $\Delta l/2$;
  for $T$, the symm-dependent offsets of its 'utor' branches.
- The thermal boundary rows and forced runs need no change: every 'h' label carries the bc rows, and `−i wf entropy`
  is already added.

### Without thermal diffusion (ThermaD = 0)

With ThermaD = 0 the section-h rows are $r \times (\mathrm{eq})/p_0$ in a $C^{(0)}$ basis. Term (a) then becomes
`h0abu0_D0` and `h0aub0_D0` (no $p_0$). Terms (b), (c) and (c′) do not fit there: the $r$ row weight makes the $P/r$
part of (c) non-polynomial, and the basis has no $C^{(1)}$ for the $P'$ term.

So with ThermaD = 0 and `thermal_wind ≥ 1` (decided only in `utils.adiab_tw`), the section-h rows are
$r^2 \times (\mathrm{eq})/p_0$ in a $C^{(1)}$ basis: `gebasis[2] = 1`, still no bc rows, `chop = 0`. The labels are the
same (`decode_label` gives section h the $r^2$ weight), and `tw2/4/6`, `tp2/4/6` are $\sigma_n$ and $\sigma_n'$ without
$p_0$. With ThermaD = 0 and `thermal_wind = 0` or `diff_rot = 0` nothing changes ($r \times (\mathrm{eq})/p_0$ in
$C^{(0)}$). Details and tests are in [thermad0.md](thermad0.md).

`submatrices.py` raises an error if a label's derivative order is above its section's basis. Before, `G[gbx][gbx-dx]`
with dx > gbx indexed from the end of the list, which silently picked a wrong basis.

### Diagnostics

The new powers couple different $l$. `utils4pp` (used by `spin_doctor`) therefore builds the diff-rot force and heat
terms on an $(r, \cos\theta)$ grid once per mode (`diff_rot_setup`) and projects them back on the harmonics, so that
each per-$l$ worker can pair them with its own $l$ (§5). The matrix-level budget, `heat_blocks()` in
`tests/heat_dr/ebal.py`, is the independent check. It verifies the entropy balance

$$
\lambda\langle E\rangle - \langle A\rangle - \langle D\rangle - \langle H\rangle = 0 .
$$

## 4. Benchmarks and which terms they need

**Mirouh et al. 2016** (JFM 800, 213; kore `shellular_boussinesq`). Boussinesq, with the rotation profile imposed from
Rieutord 2006's baroclinic flow and a radial $T_0$ only:

$$
\Omega(r) = 1 + \frac{N^2}{2}\thinspace (1 - r^2).
$$

Their heat equation (2.9) is

$$
(\lambda + i\thinspace m\Omega)\thinspace T + r v_r = \frac{E}{P}\thinspace \Delta T .
$$

It needs (a) only, with $S_{tw} = 0$: their background is balanced by centrifugal buoyancy, which kore does not carry.
With (3) switched on, $\sigma_2 = 2N^2r^2/(3\thinspace \mathrm{Beyonce})$ would be as large as $S_0'$, and the comparison would
test the wrong model. It also needs `aub`, `abu` and `logrho` in `Boussinesq_profiles`.

**Bekki et al. 2022.** Fully compressible, solar $\rho_0$, helioseismic $\Omega$, and $S_{tw}$ from (2) with the full
$\Omega^2$, with $s_0$ uniform in $r$. It needs (a), (c) and (c′). kore's $S_{tw}$ also has (b), which they drop by
construction. The momentum terms of their eq. (1),

$$
-(\Omega-\Omega_0)\thinspace \partial_\phi\mathbf v - 2\Omega\thinspace \hat{\mathbf z}\times\mathbf v - r\sin\theta\thinspace (\mathbf v\cdot\nabla\Omega),
$$

are kore's existing diff-rot operator.

**Baruteau & Rieutord 2013, Guenel et al. 2016, Astoul et al. 2021.** thermal = 0, so unaffected.

## 5. Implementation (2026-10-12, commits be2a4db5, 18641206)

The terms were implemented in two rounds, committed as `be2a4db5` and `18641206`. The files changed are in `bin/`,
plus `tests/heat_dr/` and this document under `docs/`.

| file | change |
| --- | --- |
| `params_default.py` | `thermal_wind = 0/1/2/3`: none, linear (3), exact (3′) thermal wind, or the prescribed $\partial_\theta S_{tw}$ = `dSdtheta` $\sin 2\theta$ (§1). Term (a) is always on with `diff_rot = 1` and `thermal = 1`. Test profile `diff_rot_type = 'rigid'`. 'solar' comments (amplitude, 435 nHz frame, N at ricb 0) |
| `radial_profiles.py` | `sigma2`, `sigma4`, `sigma6` from (3′) with $G^2$; profiles `tw2`, `tw4`, `tw6` = $p_0\sigma_n$ and `tp2`, `tp4`, `tp6` = $p_0\sigma_n'$. `'rigid'` branch of `abu`/`aub`. At ricb = 0 `abuX`/`aubX` evaluate the profiles at $\lvert r\rvert$ (even by construction). 'solar' times `diff_rot_amplitude`, tachocline at $x = r\cdot\mathrm{aux0}$ for 'smoothed sun'. `ptw`: $\sigma_n$, $\sigma_n'$ without $p_0$ when `utils.adiab_tw`. $\sigma_2 = -(2/3)$ `dSdtheta`, $\sigma_4 = 0$ for `thermal_wind = 3` |
| `submatrices.py` | the labels of §3 (with $\sigma_6$ for `thermal_wind = 2`); `tp2/4/6` in both `adj` lists; ThermaD = 0 labels `h0abu0_D0`, `h0aub0_D0`. Exits if `Beyonce = 0` with `thermal_wind = 1, 2`. ThermaD = 0 with `thermal_wind ≥ 1`: `gebasis[2] = 1` and `chop = 0` for section h (§3, `docs/thermad0.md`). Raises if a label's derivative order exceeds its section's basis. Warns for 'Y20-wall-bounded' and 'shellular' at ricb = 0. Parity-warning floor, so an identically zero $S_{tw}$ ('cylindrical') stays quiet |
| `operators.py` | `thermal_diffrot(l, 'h', component, Δl)`: 'temp' (a), 'upol' (b) + (c), 'utor' (c′), with $\sigma_6$. `Gaspard` on (a) and on `differential_rotation` |
| `heat_dr_coefficients.py` (new) | $X$, $D$, $P_n(X)$, $P_n'(X)$, $P_n'(X)D$ for $n = 2, 4, 6$, built well above lmax and cut (products of the truncated $X$ are wrong near the top) |
| `assemble.py` | section h: $s$ column $\Delta l \in \lbrace -2, 0, 2\rbrace$, $P$ column $-6\ldots 6$, $T$ column $-5\ldots 5$. Stops with a message unless `lmax − m + 1` is a multiple of $2\cdot\mathrm{ncpus}$ (otherwise `Scatter` dropped degrees or hung) |
| `utils4pp.py`, `spin_doctor.py` | the broken `diff_rot_power` is replaced by `diff_rot_setup()`, which evaluates the diff-rot force and heat terms below on an $(r, \cos\theta)$ grid, projected back per $l$, with $\rho_0$. New powers `Wdr`, `Wdr_im` (flow), `Ensdr` (enstrophy) and `Wdr_thm` (entropy) enter `resid1`, `resid2`, `resens`, `resid3` |
| `utils.py` | `adiab_tw` (ThermaD = 0, `diff_rot = 1`, `thermal_wind ≥ 1`): `decode_label` gives section h the $r^2$ row weight |
| `ru_insane.py` | diff-rot tests (below) and `tp2/4/6` in `row_parity` |

The diff-rot force and the heat terms evaluated by `diff_rot_setup()` are, with $\mathbf U_0'$ the mean flow of
$\Delta\Omega$:

$$
-G\left[(\mathbf U_0'\cdot\nabla)\mathbf u + (\mathbf u\cdot\nabla)\mathbf U_0'\right]
\qquad\text{and}\qquad
p_0\left[i\thinspace m G\thinspace \Delta\Omega\thinspace s + \mathbf u\cdot\nabla S_{tw}\right].
$$

Output columns appended (old columns unchanged, so files written before the change cannot be appended to):

- `flow.dat` gets `Wdr`, `Wdr_im`, `Ensdr`;
- `thermal.dat` gets `Wdr_thm`;
- `params.dat` gets `diff_rot`, `diff_rot_amplitude`, `thermal_wind`, the index of `diff_rot_type` in ('Y20',
  'Y20-wall-bounded', 'shellular', 'shellular_boussinesq', 'cylindrical', 'conical', 'solar', 'rigid', 'solar_P4',
  'tabulated'), −1 if not listed, and `dSdtheta`; the §8 options follow them.

`spin_doctor` is about 1.5 times slower with `diff_rot = 1`.

Changes visible to existing runs:

- At `Gaspard = 1` and ricb > 0 the matrices are bitwise unchanged.
- A run with ThermaD = 0, `diff_rot = 1` and the default `thermal_wind = 1` used to stop and now runs. Advection of $s$
  only is `thermal_wind = 0`, as before.
- All ricb = 0 diff-rot matrices change (the $\lvert r\rvert$ fix).
- 'solar' with an amplitude ≠ 1, or with 'smoothed sun', gives a different profile than before.

### Tests

The tests are run by `tests/heat_dr/run_all.sh`, `mpi_check.sh` and `ru_insane.py`. Unless stated, they use amplitude
−0.25, user-def profiles and ricb 0.35. Every check prints a verdict, `ok` or `FAIL`, and `run_all.sh` stops if a check
crashes.

Two formulas used in the table. The quadrature $Q$ is an $(r, \theta)$ integral of the new heat terms against a
random field,

$$
Q = \int p_0\thinspace s^\ast \left[i\thinspace m G\thinspace \Delta\Omega\thinspace s + \mathbf u\cdot\nabla S_{tw}\right] dV ,
$$

and the rigid offset is the exact relation obeyed by a uniform $\Delta\Omega$, with $\lambda_0$ computed with
`diff_rot = 0`:

$$
\lambda(\Delta\Omega, G) = \lambda_0\big(G(1+\Delta\Omega)\big) - i\thinspace m G\thinspace \Delta\Omega .
$$

| check | configurations | worst |
| --- | --- | --- |
| sympy: (3′) against the integrated thermal-wind balance (`tw_sympy.py`) | thermal_wind 1 and 2, any $G$ | 0 failures |
| operator blocks vs $Q$ for random fields (`heat_check.py`) | 43 runs: conical m 0, 1, 2 (symm ±1), N 96 and 160; thermal_wind 0, 1, 2, 3; Y20-wall-bounded, shellular, uniform ('rigid', amplitude 0, thermal_wind 3); smoothed sun and 'solar' at ricb 0 (m 0, 1, 2); $G = 3$; ThermaD = 0 with thermal_wind 0, 1, 2, 3 (18 runs, also at ricb 0) | ≤ 9e-13 (ricb 0.35, N 96), 1e-15 (N 160), ≤ 1e-10 (smoothed sun, N 120), 6e-14 ('solar', N 400), ≤ 1.1e-14 (ThermaD = 0) |
| heat rows of the assembled $A$(diff_rot 1) − $A$(diff_rot 0) vs $Q$ (`asm_check.py`) | the same runs (ThermaD = 0 with the thermal wind: reference amplitude 0) | ≤ 1e-10 (41 runs; ≤ 2.3e-12 with ThermaD = 0) |
| section-h rows of $A$, $B$ unweighted vs $(r, \theta)$ physics, claimed basis/row/weight only (`row_check.py`) | ThermaD = 0: thermal_wind 0, 1, 2, 3; m 0, 1, 2; G 3; smoothed sun (9 runs) | all ok at 1e-8 |
| rigid offset (`rigid_thermal.py`) | m 2, 16 modes; $G = 1, 3, 10^3$ (viscous units); ThermaD = 0 (G 3), and ThermaD = 0 with thermal_wind 1 (G 1, 3) | 4e-12, 2e-12, 7e-10, 2e-11; 9e-10, 9e-12 |
| `thermal_wind = 2` → 1 as the amplitude → 0 (`tw_limit.py`) | conical, Y20-wall-bounded | difference $\propto \text{amplitude}^2$ |
| `spin_doctor` powers vs the matrix pairings: $\lvert\mathrm{Wdr} + i\thinspace \mathrm{Wdr}_ {\rm im} + 2R\rvert$, $\lvert\mathrm{Wdr}_ {\rm thm} - 2\thinspace \mathrm{Re}\thinspace H\rvert$ (`sd_check.py`) | sun, rfix1, g3_tw2 (G 3, thermal_wind 2), s96, tw0, y20wb, shell, k1_tw2 (ThermaD = 0), s96_tw3 | 1e-8 (sun), 6e-7 (rfix1, N 64), 1e-9 (g3_tw2), ≤ 7e-10 (the others) |
| `spin_doctor` residuals with the new terms (`sd_check.py`) | same | resid1 ≤ 1.5e-6, resid2 ≤ 1.5e-7, resens ≤ 4e-4, resid3 ≤ 2e-8 at N ≥ 96, the diff_rot = 0 level of the same N; without the new terms they are $10^2$–$10^9$ times larger (rfix1 at N 64 is truncation-limited) |
| `ncpus` 1…4 (`mpi_check.sh`) | m 0, 1, 2 (symm −1), smoothed sun m 0, 1, thermal bc 1, forcing 10, thermal_wind 2 with G 3, diff_rot 0 | $A$, $B$, $B_{\rm forced}$ bitwise equal; 5 ranks stops with the message |
| forced run: $A_{\rm forced} = c\thinspace (A - i\thinspace \omega_f B)$ (`forced_check.py`) | forcing 6, 7, 9, 10 | 1e-15 (0.04–0.5 with a block missing) |
| `ru_insane.py`: momentum and heat diff-rot blocks vs a grid ($I$, $C$, $R$, $\mathrm{Ht}$, $\mathrm{Hw}$), (a) anti-Hermitian, rigid $R = \Delta\Omega\thinspace (i\thinspace m G I + C)$ | 27 runs: the runs above (22) plus cylindrical with thermal_wind 1 and 2, rigid m 2, conical m 0 symm −1, conical m 1; third round: 16 ThermaD = 0 runs (`docs/thermad0.md`) and 14 on the final code (5 with thermal_wind 3) | grid 2e-10, advection 1e-11, thermal wind 2e-13, anti-Hermitian 2e-12, rigid 2e-15; all PASS |

Notes on the results:

- The ~1e-11 floor at N 96 is the Chebyshev truncation of the profile products (1e-13 to 1e-14 at N 160).
- 'solar' at ricb = 0 needs N ≈ 400 for that floor, because of the tachocline step. At N 120 the checks fail on
  truncation, not on a bug.
- `ru_insane` catches all 18 one-coefficient mutations of the operators tried: sign flips, 0.1–1 % scalings, the
  target instead of the source $L$, a dropped $P_4$ term. The reviewers' mutations of the $\lvert r\rvert$ evaluation
  and of the $G^2$ in `utils4pp` are caught by `ru_insane` and `sd_check`.

Reviews: five or six independent reviewers plus a critic after each round. None found a defect in the physics, the
signs or the Gaspard factors; the defects found in the tests and comments are fixed above.

## 6. Benchmark: solar inertial modes (Bekki et al. 2022, Mukhopadhyay et al. 2025)

The best target is Mukhopadhyay, Bekki, Zhu & Gizon 2025 (A&A, arXiv 2501.16797). It uses the set-up of Bekki,
Cameron & Gizon 2022 with Model S, and tabulates modes to 1e-4 nHz for compressible, anelastic and Boussinesq models.
Their anelastic model is kore's.

The set-up is that of `tests/diff_rot/mk_bekki.py` with `drtype=Y20`: N 64, lmax $m + 63$,
$\nu = \kappa = 10^{12}$ cm² s⁻¹, $\delta = 0$ (adiabatic background, $S_0' = 0$). This section uses kore's
$P_0 + P_2$ rotation profile; §8 repeats the comparison with their full $\Omega(r, \theta)$.

**Uniform rotation.** kore matches their anelastic Tables C.1–C.4 to 0.006 nHz or better for 27 modes ($n = 0$ and 1
Rossby, high-frequency retrograde (HFR), columnar; $m = 1\ldots 16$). This needs the Model S density: the stock
'smoothed sun' fit is up to 48 % off near the surface, which moves the $n = 1$, HFR and columnar modes by 1.5–8 nHz
(18 at $m = 16$).

**Time unit (`Gaspard`).** The same problem solved with time unit $1/\Omega_0$ ($G = 1$) and with time unit
$1/\Omega_{\rm Carrington}$ ($G \neq 1$), in the same frame, agrees to 2e-6 nHz.

**Frame.** Changing the frame itself (426.86 vs 456 nHz, with $\Delta\Omega$ redefined relative to it) does not matter
with `thermal_wind = 2` (2e-6 nHz); with `thermal_wind = 1` it moves modes by 0.1–0.9 nHz.

**Differential rotation** (their Tables C.5–C.9, `thermal_wind = 2` without $u_r\thinspace \partial_r S_{tw}$, as in the papers).
kore minus target, in nHz:

- $n = 0$ Rossby: within 0.8;
- $n = 1$ Rossby: within 1.1–2.7;
- high-latitude, m 1 and 2: within 0.2–1.5;
- HFR: 2.7–6.2;
- columnar: 3.4–9.2, growing with $m$.

The remaining difference is the rotation profile. kore's $\Omega$ here has $P_0 + P_2$ only (4.5 nHz too fast at the
equator, 12 at the pole); theirs has $P_4$ and radial shear.

**Term (b).** Keeping $u_r\thinspace \partial_r S_{tw}$ (kore always did before §8 added the switch) moves columnar modes by 5–15
nHz and HFR modes by 8–10 nHz. At $\delta = 0$ it also makes the low latitudes superadiabatic: $m = 1$ symm +1 then has
a weakly growing mode (12.4 nHz, growth 1.8 nHz).

**Not reproduced with the $P_0 + P_2$ profile:**

- the growing $m = 1$ high-latitude mode of Bekki et al. 2022 (−90.9 nHz, growth 14.1 nHz); kore's nearest modes are
  damped by 3 nHz;
- the baroclinic onset of Bekki et al. 2024; kore needs a 4–8 times larger $\Delta_\theta s$.

Both were caused by the $P_2$-only rotation profile. With the observed $\Omega(r, \theta)$, kore has the growing
high-latitude mode (+11.3 nHz; Mukhopadhyay et al. 2025: +11.7) and the onset near 500 (§8).

The $\Delta_\theta s$ runs used a run-local edit of $\sigma_2$, $\sigma_4$. With `thermal_wind = 3` and `dSdtheta` =
$\Delta_\theta s/c_p$, the warm-poles case at 4000 erg g⁻¹ K⁻¹ gives $A$ and $B$ bitwise equal to that run (−101.42 nHz,
growth +7.30 nHz).

## 7. Still open

- Done in §8: the switch `thermal_wind_radial` for $u_r\thinspace \partial_r S_{tw}$ (term (b)), $P_4$ ('solar_P4') and
  tabulated $\Omega(r, \theta)$, the Model S density (`solar_density`), and the amplitude controls.
- ThermaD = 0 with `thermal_wind ≥ 1` is implemented ($r^2$ rows in $C^{(1)}$, §3) and tested
  ([thermad0.md](thermad0.md)). Open there:
    - the ThermaD → 0 limit with fixed-entropy walls has an unresolved boundary layer;
    - without diffusion the thermal wind has small-scale growing modes that are not converged in lmax;
    - its rows carry $\sigma_n\thinspace (\ln\rho)'$ (term (c)) without $p_0$, which converges slowly in N when $\rho$ nearly
      vanishes at the surface ('user def' with aux0 → 1, ricb = 0). The $p_0$-weighted checks do not see it;
      `tests/heat_dr/row_check.py` does.
- The 'smoothed sun' density near the surface (§6): `solar_density = 'model S'` (§8) avoids it; the default is
  unchanged.
- 'Y20-wall-bounded' and 'shellular' (except even $\sigma > 0$) at ricb = 0 are not smooth at $r = 0$; this is now a
  WARNING.
- Boussinesq profiles (no `abu`, `aub`, `logrho`) are left untouched. The same holds for the profiles read from a file
  (`'poly'`, `'mesa'`, `'gsm'`, `'astropy table'`): only `'user def'` and `'smoothed sun'` define a rotation profile.
- `tests/heat_dr/csolve.py` has no equilibration (kore's `solve.py` has Ruiz scaling); it now warns when every
  eigenvalue comes back equal to the shift.

## 8. Generic differential rotation: any Pₙ, 'solar_P4', 'tabulated' (2026-10-13)

Commit `92a0d020`. The rotation profile is no longer limited to $P_0 + P_2$. It can be any sum of even Legendre
polynomials, each with an amplitude factor $a_n$:

$$
\Delta\Omega = \sum_{n\ \text{even}} a_n\thinspace f_n(r)\thinspace P_n(\cos\theta).
$$

This makes the observed solar rotation usable, either as a fit ('solar_P4') or as a helioseismic table ('tabulated').

### Momentum couplings

$n = 0$ and 2 keep the closed forms of `diff_rot_coefficients.py`. For $n \ge 4$ the new `bin/diff_rot_generic.py`
projects the diff-rot force of each source degree on the target degrees:

$$
\mathbf W = G\left[\Delta\Omega\thinspace (i\thinspace m\thinspace \mathbf u + 2\hat{\mathbf z}\times\mathbf u) +
\sin\theta\thinspace \left(r\thinspace \partial_r\Delta\Omega\thinspace u_r + \partial_\theta\Delta\Omega\thinspace u_\theta\right)\hat{\mathbf e}_ \phi\right].
$$

- The projection uses Gauss–Legendre quadrature in $\cos\theta$, exact for these polynomial integrands.
- It turns the 12 primitive projections into the 18 coefficients of the existing labels, in the naming and
  normalisation of `diff_rot_coefficients.py`. They are cached per $(n, m)$ and built to $\mathrm{lmax} + n + 6$.
- The radial profiles $f_n$, $r f_n'$ and $r^2 f_n''$ have the codes `abN`, `svN`, `plN` (N = 4, 6, 8, then a, c, e, g
  for 10–16).
- Couplings: $P \to P$ and $T \to T$ have $\lvert\Delta l\rvert \le n$, and $P \leftrightarrow T$ have
  $\lvert\Delta l\rvert \le n + 1$, so `assemble.py` loops over $\Delta l$ up to `utils.dr_nmax` (+1).
- `diff_rot_generic = 1` builds $n = 0, 2$ with the quadrature too, as a self-check (agrees to ~1e-12).

### Thermal-wind entropy

For `thermal_wind` = 1 or 2 with degrees above 2, `radial_profiles.sigma_all` computes the Legendre coefficients of

$$
\frac{\partial S}{\partial x} = -\frac{r^2G^2}{\mathrm{Beyonce}\thinspace g}\left[x\thinspace \partial_r F + \frac{(1 - x^2)\thinspace \partial_x F}{r}\right],
\qquad F = 2\Delta\Omega \ \ \text{or}\ \ (1 + \Delta\Omega)^2 - 1,
$$

by quadrature on the Chebyshev nodes, and integrates in $x$. $\sigma_n'$ comes from Chebyshev differentiation.

- $S_{tw}$ has degrees $2\ldots\mathrm{nmax} + 2$ (thermal_wind 1) or $2\ldots 2\thinspace \mathrm{nmax} + 2$ (thermal_wind 2).
- Profiles: `tw2`…`tw8`, `twa`, `twc`, …, `twy` ($n = 34$) for $\sigma_n$, and `tp*` for $\sigma_n'$.
- `heat_dr_coefficients.py` builds $P_n(X)$, $P_n'(X)$ and $P_n'(X)D$ for any $n$ by the Legendre recurrence.
- `thermal_wind = 3` is unchanged ($\sigma_2$ only).

### Profiles

'solar_P4' is an analytic fit with $P_0$, $P_2$ and $P_4$, with an erf step at the tachocline:

$$
\Omega = \Omega_{\rm RZ} + \left[\Omega_{\rm CZ}(\theta) - \Omega_{\rm RZ}\right]\frac{1 + \mathrm{erf}\big((x - \mathrm{rtc})/\mathrm{dtc}\big)}{2},
\qquad \Omega_{\rm CZ} = c_0 + c_2\cos^2\theta + c_4\cos^4\theta .
$$

`diff_rot_solar` = [$c_0$, $c_2$, $c_4$, $\Omega_{\rm RZ}$, rtc, dtc] = [464, −76, −50, 433.5, 0.71, 0.045] (nHz and
$R_\odot$; Mukhopadhyay et al. 2026).

'tabulated' reads $\Omega(r, \theta)$ from `diff_rot_file`, an .npz with `ome[θ, r]` (nHz), `r` (R☉) and `theta`
(rad). This is the layout of the Mukhopadhyay et al. 2025 Edmond file `data/diff_rot.npz`;
`tests/diff_rot/ls_to_edmond_layout.py` converts the Larson & Schou 2018 table. At each table radius the profile is
projected on $P_0\ldots P_{\mathrm{nmax}}$ (`diff_rot_nmax`, 4–16), then fitted over the shell by a Chebyshev series of
degree `diff_rot_rdeg` (24). The fit and truncation errors are printed.

Both use $\Delta\Omega = \Omega/\Omega_{\rm ref} - 1$, with $\Omega_{\rm ref}/2\pi$ = `diff_rot_frame` (456 nHz), and
$x = r\cdot\mathrm{aux0}$ for 'smoothed sun'. For the Edmond profile the truncation error at nmax 8 is up to 17 nHz near
the poles at the surface; nmax 12 is the working choice.

### Amplitude controls

These apply to every profile type:

- `diff_rot_term_amp` = $[a_0, a_2, \ldots, a_{16}]$ multiplies each $P_n$ term of $\Delta\Omega$ (missing entries are 1).
- `diff_rot_amplitude` multiplies the whole of 'solar_P4' and 'tabulated' (the other types keep its old meaning).
- `thermal_wind_amplitude` multiplies $S_{tw}$, which is otherwise the thermal wind of the scaled $\Delta\Omega$.
- `thermal_wind_radial` = 0 drops $u_r\thinspace \partial_r S_{tw}$ (term (b)), as the papers do; 1 (default) keeps it.

### Model S density

`solar_density = 'model S'` replaces the 'smoothed sun' $\ln\rho$ and $\ln p$ by Chebyshev fits of `bin/model_s.at`.
The fit has degree `solar_density_deg` (30) on `solar_density_window`, by default the shell widened by 0.01 below and
0.005 above. It reproduces the run-local Model S fit of the §6 benchmark runs: $A$ and $B$ are bitwise equal.

### Other files

- `utils.py`: `dr_high`, `dr_nmax`, `tw_nmax`, `tw_degrees`, `dr_code`, `tw_code`, `odd_profiles`, `dr_offd`.
- `operators.py`: `differential_rotation` sums the closed forms and the generic terms; `thermal_diffrot` loops over
  `tw_degrees` and $\Delta l$ up to `tw_nmax`.
- `submatrices.py`: the labels per $n$.
- `utils4pp.py` (spin_doctor), `ru_insane.py`, `tests/heat_dr/ebal.py`: generic $\Delta l$ ranges, $\Delta\Omega$ and
  $S_{tw}$ summed over $n$, the switches and amplitudes.
- `spin_doctor.py`: 'solar_P4' and 'tabulated' get the type indices 8 and 9, and `params.dat` gets 25 more columns after
  `dSdtheta` (listed in `docs/differential-rotation.md`, Output), so files written before cannot be appended to.

Changes visible to existing runs: none. $A$ and $B$ are bitwise equal to `18641206` for 13 configurations
(`tests/diff_rot/regress.sh`) and for the benchmark runs of §6.

### Tests

See `tests/diff_rot/README.md`.

- Generic against closed-form coefficients: 4e-11.
- No coupling outside the assembled $\Delta l$ ranges for n 4–16 (6e-12).
- `ru_insane` grid checks: ~1e-14 on 'tabulated' (thermal_wind 1 and 2, radial 0 and 1, term and thermal-wind
  amplitudes) and on 'solar_P4'. It catches a 1e-4 change of one generic coefficient or of $\sigma_n$.
- spin_doctor powers against the matrix pairings: ≤ 1e-8, with the energy balances closing to 1e-7 (1e-2 to 1
  without the new terms).
- kore's $\partial_\theta S_{tw}$ against the Edmond `dsdt`: 3 % rms (pole-to-equator contrast within 0.1–2 %).

### Like-for-like solar benchmark

This repeats §6 with the Mukhopadhyay et al. 2025 set-up in full: Model S density, shell 0.71–0.985 R☉,
$\nu = \kappa = 10^{12}$, $\delta = 0$, stress-free, $\partial_r s = 0$, `thermal_wind = 2` without
$u_r\thinspace \partial_r S_{tw}$, their $\Omega$ from the Edmond file, N 64, lmax $m + 63$.
`tests/diff_rot/run_bekki.py` rebuilds and solves these five modes. Frequencies are in nHz in the 456 nHz frame,
negative is retrograde; the second number is the growth rate.

| mode (symm) | kore, nmax 12 (nHz) | Mukhopadhyay compressible (nHz) | kore − theirs | kore, $P_0 + P_2$ only (§6) |
| --- | --- | --- | --- | --- |
| m 1 high-latitude (−1) | −95.759, growth +11.301 | −95.70, +11.69 | −0.06, −0.39 | −95.67, −15.8 |
| m 3 Rossby n0 (−1) | −262.052, −0.877 | −262.07, −0.88 | +0.02, 0.00 | within 0.8 |
| m 3 Rossby n1 (−1) | −181.934, −9.700 | −182.05, −9.67 | +0.12, −0.03 | 1.1–2.7 off |
| m 3 columnar (+1) | 210.908, −11.090 | 210.84, −11.09 | +0.07, 0.00 | 3.4–9.2 off |
| m 10 HFR (+1) | −211.088, −19.963 | −211.28, −20.00 | +0.19, +0.04 | 2.7–6.2 off |

- The m 1 mode is converged: nmax 8 / 12 / 16 give −95.805 / −95.759 / −95.749 (growth +11.48 / +11.30 / +11.29), and
  N, lmax 96 reproduces N 64 to 1e-6 nHz.
- Their targets are their compressible values. The anelastic ones of the same paper differ from them by amounts of
  this order, and their published eigenfunctions use 24 radial points.
- The m 3 symm −1 window around n0 also holds two growing modes (−260.59 nHz, growth +17.98; −250.45, +5.04) with
  solver residuals 3e-5 to 4e-5 (1e-7 for n0). They were not examined, and are not reported in the paper's tables.

spin_doctor's balances for these five modes close to 5e-6 or better. The table gives them with the differential-rotation
terms (Wdr, Wdr_im, Ensdr, Wdr_thm) and, in brackets, without them, which shows how much those terms matter. resid1 and
resid2 are the real and imaginary parts of the kinetic-energy balance, resens the enstrophy balance and resid3 the
entropy balance. Computed by `tests/heat_dr/sd_check.py` (the balances of `utils4pp.diagnose`) on the
`run_bekki.py` eigenvectors, N 64, nmax 12:

| mode (symm) | resid1 | resid2 | resens | resid3 |
| --- | --- | --- | --- | --- |
| m 1 high-latitude (−1) | 1.4e-9 (3.7e-3) | 2.4e-10 (0.77) | 3.3e-8 (0.13) | 5.9e-12 (1.3) |
| m 3 Rossby n0 (−1) | 2.2e-9 (0.23) | 1.9e-11 (0.15) | 8.2e-8 (0.10) | 2.2e-11 (0.84) |
| m 3 Rossby n1 (−1) | 9.7e-9 (0.076) | 1.3e-9 (0.59) | 3.6e-8 (0.073) | 3.8e-11 (0.48) |
| m 3 columnar (+1) | 1.9e-7 (0.037) | 4.9e-9 (0.095) | 5.0e-6 (0.042) | 5.5e-11 (0.72) |
| m 10 HFR (+1) | 8.2e-8 (0.16) | 2.1e-9 (0.15) | 1.7e-6 (0.031) | 1.5e-11 (0.016) |

The diff-rot powers also agree with the matrix-level pairings of `tests/heat_dr/ebal.py` to 1.4e-9 or better for every
mode.

### Bekki et al. 2024 onset with the observed rotation

Set-up: `thermal_wind = 3` with $\Delta_\theta s$ prescribed, otherwise as above, m 1 symm −1. Run
`tests/diff_rot/mk_bekki.py tw=3 ds=-1000 file=...`, then `solve_bekki.py RUN -93 5`.

The m 1 high-latitude mode becomes unstable at $\lvert\Delta_\theta s\rvert$ = 425 erg/g/K with the Edmond $\Omega$, and
at 546 with the Larson & Schou 2018 table ($c_p = 3.46\times10^{8}$; 513 and 658 with the papers' $c_p = 4.17\times10^{8}$).
The published value is ~500; the radially uniform $P_0 + P_2$ profile of §6 needs 2199 (2650).

Growth rate (nHz) against $\lvert\Delta_\theta s\rvert$ (erg/g/K):

| $\lvert\Delta_\theta s\rvert$ | 0 | 250 | 500 | 1000 | 2000 |
| --- | --- | --- | --- | --- | --- |
| Edmond (mode at −91.1…−94.6 nHz) | −3.70 | −1.60 | +0.68 | +5.02 | +12.53 |
| Larson & Schou (mode at −96.1…−98.9 nHz) | −3.83 | −2.30 | −0.36 | +3.56 | +10.42 |

So the gap of §6 was the radial shear of $\Omega$ in the lower convection zone, as step 1 of the comparison had
concluded with kore's 'solar' step (onset 509).
