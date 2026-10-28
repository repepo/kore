# Reynolds-stress forcing

This page derives the random force of forcing 11: where it comes from, its statistics, why its strength follows from
the viscosity, how kore writes it in its variables, and what the resulting spectrum means. How to run it is on
[Convective forcing](forcing.md). The notation is that of [Anelastic equations](anelastic.md), in kore's units.

## Where the force comes from

In a convection zone the velocity is the sum of a large-scale flow $\mathbf u$, the modes that kore solves for, and
the convective eddies $\mathbf u'$, which are much smaller and faster. The eddies act on $\mathbf u$ through the
divergence of their flux of momentum, the Reynolds stress: the momentum equation of $\mathbf u$ contains the force

$$
-\frac{1}{\rho_0}\nabla\cdot\left(\rho_0\thinspace\mathbf u'\mathbf u'\right).
$$

This force has two parts.

1. **The response to the large-scale flow.** On average the eddies mix momentum down the gradients of $\mathbf u$, as a
   viscosity does: the trace-free part of the mean of $\mathbf u'\mathbf u'$ changes by $-\nu_t\thinspace\mathsf S(\mathbf u)$,
   with $\nu_t$ the turbulent (eddy) viscosity and $\mathsf S$ the trace-free rate of strain. In a convection zone this
   is kore's viscous force: kore's viscosity $\nu$ is the sum of $\nu_t$ and a part $\nu_f$ that is not turbulent, such
   as the viscosity of a radiative zone.
2. **The fluctuation.** What is left, $\mathbf u'\mathbf u'$ minus its mean, does not depend on $\mathbf u$. It is a
   random stress, and its divergence is a random force that drives the modes,

$$
\mathbf f = \frac{1}{\rho_0}\nabla\cdot\left(\rho_0\thinspace\mathsf s\right),
$$

with $\mathsf s$ a random symmetric tensor per unit mass. Its sign does not matter, since $\mathsf s$ and $-\mathsf s$
have the same statistics.

The two parts come from the same eddies, so they are not independent. The viscous part alone would damp every mode to
zero, and the random part alone would let the modes grow without bound. Together they keep each mode at a finite
energy, in a statistically steady state. Brownian motion is the same: the friction on a particle and the random kicks
it receives come from the same molecules, and their ratio is set by the temperature (Einstein's relation). For a fluid
in thermal equilibrium, Landau and Lifshitz add to the viscous stress a random stress whose variance is the temperature
times the viscosity, so that the fluctuations of the flow have the energy that equipartition requires. Forcing 11
applies this construction to the eddies: the viscosity is $\nu_t$, and the temperature is a *turbulent temperature*
$\Theta(r)$, about the kinetic energy of one eddy.

## The random stress

kore takes $\mathsf s$ trace-free and white, with the covariance

$$
\mathbb E\left[s_{ij}(\mathbf x, t)\thinspace s_{kl}^*(\mathbf x', t')\right] = \frac{4\thinspace\Theta\thinspace\nu_t}{\rho_0}\thinspace\Pi_{ijkl}\thinspace\delta(\mathbf x - \mathbf x')\thinspace\delta(t - t'),
$$

$$
\Pi_{ijkl} = \tfrac12\left(\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk}\right) - \tfrac13\delta_{ij}\delta_{kl},
$$

where $\mathbb E$ is the mean over realisations and $\Pi$ the projector onto symmetric trace-free tensors. With
$\Theta$ replaced by $k_B T$ and $\rho_0\nu_t$ by the dynamic viscosity, this is the random stress of Landau and
Lifshitz (divided by $\rho_0^2$, because $\mathsf s$ is per unit mass).

- **White.** The eddies are much smaller than the modes, and their turnover time is shorter than the periods of the
  modes, so that on the scales of the modes their stress is uncorrelated in space and time. kore can give the eddies a
  finite size ([below](#finite-eddies-and-walls)).
- **Trace-free.** The trace of $\mathbf u'\mathbf u'$ is the turbulent pressure. Its gradient drops out of the curls
  that kore takes of the momentum equation, and what is left has no dissipative partner, since kore has no bulk
  viscosity. Leaving it out keeps the relation of the next section exact.
- **Strength.** The factor $4\Theta\nu_t/\rho_0$ is what the next section requires. A direct estimate gives the same
  form. For Gaussian eddies with $\langle u'_i u'_k\rangle = \tfrac13 u_c^2\delta_{ik}$, the trace-free part of
  $\mathbf u'\mathbf u'$ has the covariance $2(u_c^2/3)^2\thinspace\Pi$. If it is correlated over a volume $\ell_c^3$
  and a time $\tau_c = \ell_c/u_c$, it acts on larger scales as a white noise of strength
  $2(u_c^2/3)^2\thinspace\ell_c^3\tau_c\thinspace\Pi$. Equating this with $4\Theta\nu_t\Pi/\rho_0$, with the
  mixing-length viscosity $\nu_t = u_c\ell_c/3$, gives $\Theta = \rho_0 u_c^2\ell_c^3/6$; a time correlation
  $e^{-\lvert t\rvert/\tau_c}$, whose integral over all lags is $2\tau_c$, gives

$$
\Theta = \tfrac13\thinspace\rho_0\thinspace u_c^2\thinspace\ell_c^3 ,
$$

the value kore uses. $\Theta$ is thus the kinetic energy of one eddy, up to a factor of order 1, which `rsf_scale`
absorbs.

## Energy of a mode

With the inner product $\langle\mathbf a, \mathbf b\rangle = \int\rho_0\thinspace\mathbf a^*\cdot\mathbf b\thinspace dV$,
kore normalizes its eigenvectors to $\langle\mathbf a, \mathbf a\rangle = 1$, a kinetic energy of 1. The random stress
vanishes at the walls ([below](#finite-eddies-and-walls)), so for any velocity field $\mathbf a$ an integration by parts
gives

$$
\langle\mathbf a, \mathbf f\rangle = -\int\rho_0\thinspace\nabla\mathbf a^* : \mathsf s\thinspace dV = -\tfrac12\int\rho_0\thinspace\mathsf S(\mathbf a)^* : \mathsf s\thinspace dV,
$$

because only the symmetric trace-free part of $\nabla\mathbf a$, which is $\tfrac12\mathsf S(\mathbf a)$, meets the
symmetric trace-free $\mathsf s$. With the covariance above,

$$
\mathbb E\left[\langle\mathbf a, \mathbf f\rangle(t)\thinspace\langle\mathbf a, \mathbf f\rangle^*(t')\right] = \int\rho_0\thinspace\Theta\thinspace\nu_t\thinspace\lvert\mathsf S(\mathbf a)\rvert^2\thinspace dV\;\delta(t - t'),
$$

with $\lvert\mathsf S\rvert^2 = \mathsf S^* : \mathsf S$. The viscous force does on the same field the work
$-\tfrac12\int\rho_0\nu\lvert\mathsf S(\mathbf a)\rvert^2\thinspace dV$ ([Anelastic equations](anelastic.md#the-equations)).
The force that a field receives and the energy that it dissipates are the same integral: with a uniform $\Theta$, the
variance of the force is $2\Theta$ times the dissipation by $\nu_t$. This is the fluctuation–dissipation relation.

For a mode $\mathbf a$ of eigenvalue $\lambda$, with modes that are orthogonal, the amplitude of the mode obeys
$\dot c = \lambda c + \langle\mathbf a, \mathbf f\rangle$, with
$-\mathrm{Re}\thinspace\lambda = \tfrac12\int\rho_0\nu\lvert\mathsf S(\mathbf a)\rvert^2\thinspace dV$. A white force of
strength $Q$ keeps $\mathbb E\lvert c\rvert^2 = Q/(2\lvert\mathrm{Re}\thinspace\lambda\rvert)$, so the kinetic energy of
the mode is

$$
\mathbb E\lvert c\rvert^2 = \frac{\int\rho_0\Theta\nu_t\lvert\mathsf S\rvert^2\thinspace dV}{\int\rho_0\nu\lvert\mathsf S\rvert^2\thinspace dV} = \Theta_{\rm eff}\thinspace f_t,
$$

$$
\Theta_{\rm eff} = \frac{\int\rho_0\Theta\nu_t\lvert\mathsf S\rvert^2\thinspace dV}{\int\rho_0\nu_t\lvert\mathsf S\rvert^2\thinspace dV},
\qquad
f_t = \frac{\int\rho_0\nu_t\lvert\mathsf S\rvert^2\thinspace dV}{\int\rho_0\nu\lvert\mathsf S\rvert^2\thinspace dV}.
$$

- A mode holds the energy of one eddy, $\Theta_{\rm eff}$, averaged with the mode's own dissipation as the weight,
  times the share $f_t$ of its damping that the turbulence provides. This is the result of Goldreich & Keeley (1977) for
  the solar p modes: the energy of a mode is about that of an eddy.
- For a given $\Theta$ the energy does not depend on the viscosity: $\nu$ sets the width of the resonance, not its
  area. (With the `'nu'` law of `rsf_theta`, $\Theta$ itself grows as $\nu_t^2$.)
- A mode that dissipates mostly where the viscosity is not turbulent, in a radiative zone for example, has
  $f_t \ll 1$ and receives little energy.
- The modes are close to orthogonal when they are weakly damped in a uniformly rotating fluid. A differential
  rotation makes them far from orthogonal: the response to the force then differs from this estimate, and only the
  forced solve, which kore does, gives it.

## The force in kore's variables

At degree $l$, with $Y = Y_l^m$, $n_l = \int\lvert Y\rvert^2\thinspace d\Omega = 4\pi/(2l+1)$, $\mathcal L = l(l+1)$
and the vector harmonics $\mathbf B = r\nabla Y$ and $\mathbf C = \hat{\mathbf r}\times\mathbf B$, the stress is
$\mathsf s = \sum_k c_k(r)\thinspace\mathsf T_k$ with five trace-free tensor harmonics:

| $k$ | $\mathsf T_k$ | $\int\lvert\mathsf T_k\rvert^2\thinspace d\Omega/n_l$ |
| --- | --- | --- |
| 1 | $Y\thinspace\hat{\mathbf r}\hat{\mathbf r} - \tfrac12 Y(\mathsf I - \hat{\mathbf r}\hat{\mathbf r})$ | $3/2$ |
| 2 | $\hat{\mathbf r}\mathbf B + \mathbf B\hat{\mathbf r}$ | $2\mathcal L$ |
| 3 | $\hat{\mathbf r}\mathbf C + \mathbf C\hat{\mathbf r}$ | $2\mathcal L$ |
| 5 | $\nabla_1\nabla_1 Y + \tfrac12\mathcal L\thinspace Y(\mathsf I - \hat{\mathbf r}\hat{\mathbf r})$, the trace-free Hessian of $Y$ on the unit sphere | $\mathcal L(\mathcal L - 2)/2$ |
| 6 | the tensor of components $(-b, a, b)$, where $\mathsf T_5$ has $(a, b, -a)$ | $\mathcal L(\mathcal L - 2)/2$ |

$\nabla_1$ is the gradient on the unit sphere, and the components are $(\theta\theta, \theta\phi, \phi\phi)$.
$\mathsf T_5$ and $\mathsf T_6$ exist for $l \geq 2$. The force $\mathbf f = f^P\thinspace Y\hat{\mathbf r} + f^B\thinspace\mathbf B + f^C\thinspace\mathbf C$
then has, with $h = d\ln\rho_0/dr$ and a prime for $d/dr$,

$$
f^P = c_1' + \left(\frac3r + h\right)c_1 - \frac{\mathcal L\thinspace c_2}{r},
$$

$$
f^B = -\frac{c_1}{2r} + c_2' + \left(\frac3r + h\right)c_2 - \frac{(\mathcal L - 2)\thinspace c_5}{2r},
$$

$$
f^C = c_3' + \left(\frac3r + h\right)c_3 - \frac{(\mathcal L - 2)\thinspace c_6}{2r}.
$$

kore's momentum rows are $r^5\rho_0\thinspace\hat{\mathbf r}\cdot\nabla\times\nabla\times(\cdot)/\mathcal L^3$ and
$r^3\rho_0\thinspace\hat{\mathbf r}\cdot\nabla\times(\cdot)/\mathcal L^3$ ([Numerical method](numerical-method.md#rows)),
and the curls of $\mathbf f$ at degree $l$ are $(\mathcal L/r^2)[f^P - (rf^B)']$ and $-(\mathcal L/r)f^C$. The
right-hand side in the rows of $P_l$ and $T_l$ is therefore

$$
b_u = \frac{r^3\rho_0}{\mathcal L^2}\left[f^P - (r f^B)'\right], \qquad b_v = -\frac{r^2\rho_0}{\mathcal L^2}\thinspace f^C,
$$

for the default powers of $r$ and $\rho_0$ (`rpower_u`, `rhopower_u`, `rpower_v`, `rhopower_v` in general), and zero in
the rows of the heat equation.

- **Only the shear stresses drive the radial vorticity.** $c_3$ is the $r\theta$, $r\phi$ part of the stress and $c_6$
  its horizontal shear: they force the toroidal rows, and with them the Rossby modes and the other vortical modes.
  $c_1$, $c_2$ and $c_5$ force the poloidal flow.
- **Statistics.** Since $\delta(\mathbf x - \mathbf x') = \delta(r - r')\thinspace\delta(\Omega - \Omega')/r^2$, the
  $c_k$ are independent across $k$, $l$, $m$ and frequency, with

$$
\mathbb E\left[c_k(r)\thinspace c_k^*(r')\right] = \frac{4\Theta\nu_t}{\rho_0}\thinspace\frac{\chi_k}{r^2 n_l}\thinspace\delta(r - r'),
$$

$$
\chi_k = \frac{n_l}{\int\lvert\mathsf T_k\rvert^2\thinspace d\Omega} = \tfrac23,\ \frac{1}{2\mathcal L},\ \frac{1}{2\mathcal L},\ \frac{2}{\mathcal L(\mathcal L - 2)},\ \frac{2}{\mathcal L(\mathcal L - 2)} .
$$

## Finite eddies and walls

A force that is white in radius cannot be resolved, and real eddies have a finite size. kore therefore smooths the
noise over the eddy size $\ell_c(r)$, and makes it vanish at the walls.

**Radial correlation.** The white $\delta(r - r')$ is replaced by a unit-area Gaussian of width $\ell_c$. kore draws
the noise at centres $r_j$ spaced $h_j = \sigma_j/2$, with $\sigma_j = \ell_c(r_j)/\sqrt2$, from below the inner wall
(or $r = 0$) to above the outer wall, and builds

$$
c_k(r) = \tau(r)\thinspace A_{lk}(r)\thinspace S(r)^{-1/2}\sum_j\sqrt{h_j}\thinspace\xi_j\thinspace g_j(r),
$$

$$
S(r) = \sum_j h_j\thinspace g_j(r), \qquad A_{lk} = \left(\frac{4\Theta\nu_t}{\rho_0}\thinspace\frac{\chi_k}{r^2 n_l}\right)^{1/2},
$$

with $g_j$ unit-area Gaussians of width $\sigma_j$ centred at $r_j$ and $\xi_j$ independent complex Gaussian numbers
of unit variance. For a uniform $\ell_c$, $\mathbb E[c_k(r)c_k^*(r')]$ is $\tau(r)\tau(r')A_{lk}(r)A_{lk}(r')$ times
a unit-area Gaussian of width $\ell_c$ in $r - r'$. The amplitude $A_{lk}$ is taken at $r$, not at the centres, so
that the stress vanishes wherever $\nu_t$ does: none of it leaks into a radiative zone.

**Horizontal size.** With `rsf_angular = 1` the variance at degree $l$ is multiplied by
$e^{-\mathcal L\ell_c^2/(2r^2)}$: eddies of size $\ell_c$ do not force degrees much above $r/\ell_c$.

**Walls.** At a stress-free wall the rows carry no surface traction, so the stress must vanish there. The taper $\tau$
goes from 1 to 0 over `rsf_taper` below the outer wall, as $\tfrac12\thinspace\mathrm{erfc}$ of width `rsf_taper`/8
centred `rsf_taper`/2 below it. The same holds above an inner core, over `rsf_taper_in` above $r = 0$ in a full
sphere, and above `rsf_rcut` over `rsf_rcut_width` when a cut is set.

These choices only lower the force: for every field $\mathbf a$,

$$
\mathbb E\lvert\langle\mathbf a, \mathbf f\rangle\rvert^2 \leq \int\rho_0\thinspace\Theta\nu_t\thinspace\tau^2\thinspace e^{-\mathcal L\ell_c^2/(2r^2)}\thinspace\lvert\mathsf S(\mathbf a)\rvert^2\thinspace dV
$$

per unit time, with equality in the white limit. A mode receives less energy than $\Theta_{\rm eff}f_t$ when its radial
or horizontal scale approaches $\ell_c$, and when it dissipates near a wall, inside the taper.

## Eddy size and turbulent temperature

- **Turbulent viscosity.** $\nu_t$ = `ViscosD` $(\hat\nu - \nu_f)$, clipped at 0, with $\hat\nu$ the viscosity profile
  and $\nu_f$ = `rsf_nu_floor` ([Convective forcing](forcing.md#where-the-force-acts)).
- **Eddy size.** $\ell_c = \left(\ell_{\max}^{-4} + (\alpha H_p)^{-4}\right)^{-1/4}$, a smooth minimum of $\alpha H_p$
  and $\ell_{\max}$, with $H_p = -p_0/p_0'$ the pressure scale height (infinite at $r = 0$), $\alpha$ = `rsf_alpha` and
  $\ell_{\max}$ = `rsf_lc_max`; or the constant `rsf_lc`.
- **Turbulent temperature**, `rsf_theta`, times `rsf_scale`:
    - `'nu'`: eddies of size $\ell_c$ whose mixing-length viscosity is $\nu_t$, $u_c = 3\nu_t/\ell_c$, so that
      $\Theta = \tfrac13\rho_0u_c^2\ell_c^3 = 3\rho_0\nu_t^2\ell_c$;
    - `'mlt'`: the mixing-length velocity of the convective flux, $u_c^3 = \alpha F_c/(10\rho_0)$, with
      $F_c = L_\star/(4\pi r^2)\thinspace\nu_t/\max\nu_t$ and the luminosity $L_\star$ = `rsf_lum`, and $\Theta = \tfrac13\rho_0u_c^2\ell_c^3$;
    - a number: a uniform $\Theta$.

$\Theta$ is an energy, in units of $\rho_u L^5/\tau^2$, with $\rho_u$ the unit of density and $L$ and $\tau$ the
units of length and time.

## The spectrum

Each realisation is the Fourier amplitude of the force at the frequency $w$, normalized so that, for any quantity $y$
that depends linearly on the flow, $\mathbb E\lvert y(w)\rvert^2$ is the two-sided power spectral density of $y$:
$\int\mathbb E\lvert y(w)\rvert^2\thinspace dw/2\pi$ is the variance of $y$. The kinetic energy that kore reports for
an azimuthal wavenumber $m$ is that of the real flow $\mathbf u\thinspace e^{im\phi}$ + c.c., so that for $m \neq 0$
the integral of the mean kinetic energy over $dw/2\pi$ is the mean kinetic energy of the forced flow. For $m = 0$ the
flow is real, its spectrum is even in $w$, and the halves $w > 0$ and $w < 0$ are the same motion: the integral over
all $w$ is twice the energy. A white noise is white in every rotating frame, so the frame of the run does not matter.

Near a mode, the forced flow is $c(w)\thinspace\mathbf a$ with $c(w) = \langle\mathbf a, \mathbf f(w)\rangle/(iw - \lambda)$,
and its mean kinetic energy is

$$
\mathbb E\lvert c(w)\rvert^2 = \frac{Q}{(\mathrm{Re}\thinspace\lambda)^2 + (w - \mathrm{Im}\thinspace\lambda)^2},
$$

a Lorentzian centred at $w = \mathrm{Im}\thinspace\lambda$, of half-width $\lvert\mathrm{Re}\thinspace\lambda\rvert$
and height $H = Q/(\mathrm{Re}\thinspace\lambda)^2$. Its area,
$\int\mathbb E\lvert c\rvert^2\thinspace dw/2\pi = Q/(2\lvert\mathrm{Re}\thinspace\lambda\rvert)$, is the energy of the
mode: $H\lvert\mathrm{Re}\thinspace\lambda\rvert/2$.

## Limits

- The model is linear: it gives the energy that each mode receives, not the nonlinear saturation or the coupling
  between modes.
- An unstable mode has no stationary response.
- The time correlation of the eddies is neglected. Deep in the solar convection zone the turnover time approaches the
  periods of the inertial modes. The fluctuation–dissipation relation still holds for a viscosity that depends on
  frequency: read $\nu_t$ as the eddy viscosity at the frequency of the modes.
- Left out: the turbulent pressure (the trace of $\mathbf u'\mathbf u'$), the fluctuations of the convective heat flux
  (the heat equation is not forced) and any anisotropy of the eddies.

Philidet & Gizon (2023) model the same source for the solar Rossby modes on a two-dimensional $\beta$ plane, with a
source taken from the observed surface turbulence and a damping set separately. Here the source and the damping are
tied by the fluctuation–dissipation relation, and the source has a depth profile.

## References

- Goldreich, P. & Keeley, D. A. 1977, ApJ 212, 243.
- Landau, L. D. & Lifshitz, E. M. 1987, *Fluid Mechanics*, 2nd ed., Pergamon, chapter XVII (Fluctuations in fluid
  dynamics).
- Löptien, B., Gizon, L., Birch, A. C. et al. 2018, Nature Astronomy 2, 568.
- Philidet, J. & Gizon, L. 2023, A&A 673, A124.
