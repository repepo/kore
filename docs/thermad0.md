# ThermaD = 0: runs without thermal diffusion

With `ThermaD = 0` the heat equation has no diffusion term. The entropy perturbation is only carried by the flow, so
the motion is adiabatic:

$$
p_0\left(\partial_t s + u_r\frac{dS}{dr}\right) = 0,
$$

plus, with a differential rotation, the advection terms of
[Differential rotation](differential-rotation.md#heat-equation-thermal-1). This is the limit to use when thermal
diffusion is negligible on the time scale of the modes. It works with every background and with every
differential-rotation option, including the thermal wind.

## What kore does differently

- **No thermal boundary conditions.** The heat equation is first order in $r$ without diffusion, so `bci_thermal` and
  `bco_thermal` are not used.
- **Heat rows divided by $p_0$.** $p_0$ is about $10^{-7}$ near the surface of a solar model, and rows weighted by $p_0$
  are badly conditioned there. The heat rows are $r/p_0$ times the equation, or $r^2/p_0$ times it with a differential
  rotation and `thermal_wind` ≥ 1 ([details](differential_rotation_details.md#72-heat-diff_rotheat_block)).
- **No Ruiz prescaling.** With the default `prescale = None`, `solve.py` turns the prescaling off for `ThermaD = 0`,
  where it degrades the solutions ([Solver](solver.md)).

## Recommendations

**Prefer `ThermaD = 0` to a very small `ThermaD`.** As `ThermaD` → 0 the eigenvalues of a diffusive run approach the
`ThermaD = 0` ones linearly in `ThermaD`, as long as the diffusive run is resolved. With the thermal wind and walls that
are both stress-free and at fixed entropy (`bci_thermal = bco_thermal = 0`), a small `ThermaD` forms a thermal boundary
layer of width about $(\mathrm{ThermaD}/\lvert\lambda\rvert)^{1/2}$ at the walls, because the adiabatic entropy is not
zero there. Below `ThermaD` ≈ $10^{-7}$ that layer needs far more radial resolution than the bulk. The
`ThermaD = 0` run has no layer. Zero-flux or no-slip walls have no such layer either.

**Let $dS/dr$ vanish at the inner wall.** In a shell, a $dS/dr \neq 0$ at the inner wall gives spurious modes trapped
there. Their eigenvalues move with N and `spin_doctor` flags them (residσ ≈ 1). For an erf top hat, take
$x_1 \ge$ `ricb` $+ 3w_1$. If the gradient must reach the wall, raise `nev` by one or two, because the spurious modes take
some of the slots.

**Expect a band of eigenvalues near σ = 0 with a differential rotation.** Without diffusion, the advection of $s$ by the
differential rotation has a continuous spectrum,

$$
\lambda = -i\thinspace m\thinspace\mathrm{Gaspard}\thinspace\Delta\Omega(r,\theta),
$$

which the discretization shows as many eigenvalues near $\sigma = 0$ whose frequencies lie in the range of
$m\thinspace\mathrm{Gaspard}\thinspace\Delta\Omega$. With the thermal wind some of them grow slowly. They peak at
$l \approx$ `lmax` and change with `lmax`: they are small-scale instabilities of the diffusionless thermal wind, not
modes at the resolved scales. Discard any mode that fails the ℓ-convergence check of `spin_doctor`. They disappear
with `thermal_wind = 0` or with a moderate diffusivity (`ThermaD` ≈ $10^{-3}$).

**Put the target away from that band.** The eigensolver converges slowly near it.

**Check the convergence in N.** In a full sphere, eigenfunctions with `ThermaD = 0` need more radial resolution than
diffusive ones (N = 200 instead of 120 for a smooth solar model). Where $\rho_0$ nearly vanishes at the outer wall
(structure `'Wu 2005'` with `r_out` close to 1), the thermal-wind rows converge slowly in N.

**Choose the thermal-wind option on purpose.** With `diff_rot = 1` the default `thermal_wind = 1` includes the
thermal-wind terms. For the advection of $s$ by the differential rotation alone, set `thermal_wind = 0`.
