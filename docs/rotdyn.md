# Rotational dynamics

By default the mantle and the inner core rotate rigidly with the reference frame. With `rotdyn = 1` they can instead respond to the axial torques that the fluid exerts on them, and to the gravitational torque between them. The inner core can also relax viscously towards gravitational alignment with the mantle. This couples the fluid's axisymmetric zonal flow (for example torsional modes) to changes in the rotation of the mantle and the inner core.


## Unknowns and equations

`rotdyn = 1` adds three unknowns to the problem: the mantle and inner-core rotation rates relative to the frame, $\omega_M$ and $\omega_{IC}$, and the angle between the inner core and the mantle, $\varphi$. Their equations, for a mode with eigenvalue $\lambda$, are

$$
\lambda\, I_M\, \omega_M = \Gamma^{\nu}_M + \Gamma^{B}_M + K \varphi,
$$

$$
\lambda\, I_{IC}\, \omega_{IC} = \Gamma^{\nu}_{IC} + \Gamma^{B}_{IC} - K \varphi,
$$

$$
\lambda\, \varphi = \omega_{IC} - \omega_M - \frac{\Omega\tau}{\Omega\tau_{IC}}\, \varphi .
$$

$I_M$ and $I_{IC}$ are the axial moments of inertia, $\Gamma^\nu$ and $\Gamma^B$ the viscous and electromagnetic torques of the fluid on each body, and $K$ the gravitational coupling constant. The last term in the third equation is the viscous relaxation of the inner core, with relaxation time $\tau_{IC}$. The fluid is coupled to the solid bodies through the no-slip condition on the axisymmetric toroidal flow of degree 1: its value at the CMB is $r_{cmb}\,\omega_M$ and at the ICB $r_i\,\omega_{IC}$, instead of zero.

The viscous torques scale with $E\,(\Omega\tau)$, the electromagnetic torques with $Le^2 (\Omega\tau)^2$, and the gravitational one with $(\Omega\tau)^2$, so all are in the units set by `timescale`. An electromagnetic torque exists only on a body that conducts: the mantle with `mantle = 'TWA'`, the inner core with `'conducting, Chebys'` or `'TWA'` (see [Inner core and mantle](boundaries.md#electromagnetic-torques)).


## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `rotdyn` | 0 | 1 lets the mantle and the inner core respond to axial torques. |
| `MoIZ_M` | 11.16 | Axial moment of inertia of the mantle, $I_M$, in units of $\rho L^5$. |
| `MoIZ_IC` | $9.2\times10^{-3}$ | Axial moment of inertia of the inner core, $I_{IC}$, in units of $\rho L^5$. |
| `gTorque` | $5\times10^{-8}$ | Gravitational torque constant, divided by $\rho L^5 \Omega^2$. $K$ = `gTorque * OmgTau**2`. |
| `OmgtauIC` | $2.3\times10^{3}$ | $\Omega\tau_{IC}$, the inner core's viscous relaxation time times the rotation rate. 2300 corresponds to $\tau_{IC}$ = 1 year for the Earth. |
| `vtrq` | 1 | 0 switches off the viscous torques on the mantle and the inner core. |
| `mtrq` | 1 | 0 switches off the electromagnetic torques. |

The default moments of inertia and coupling constant are Earth-like values.


## Requirements

- An axisymmetric, equatorially symmetric problem: `m = 0`, `symm = 1`.
- An inner core (`ricb > 0`) and no-slip boundaries (`bci = bco = 1`, `Ek > 0`), since the solid bodies drag the fluid through the no-slip condition.
- Eigenvalue problems. The rotation of the boundaries in forced problems (libration) is prescribed through `forcing` instead.


## Output

`solve_nopp.py` writes the three unknowns of each solution to `rotdyn.field` (complex numbers). `spin_doctor.py` appends to `rotdyn.dat`, one row per solution, the complex values of

| Column | Quantity |
|---|---|
| 1 | $\omega_M$, mantle rotation rate |
| 2 | $\omega_{IC}$, inner core rotation rate |
| 3 | $\varphi$, inner core - mantle angle |
| 4 | gravitational torque $K\varphi$ |
| 5 | viscous torque on the mantle |
| 6 | electromagnetic torque on the mantle |
| 7 | viscous torque on the inner core |
| 8 | electromagnetic torque on the inner core |

The total axial angular momentum (fluid, mantle and inner core) is conserved by these equations; the fluid's own balance and the torques are written out in [Angular momentum balance](equations.md#angular-momentum-balance). `spin_doctor`'s `resid𝐋` checks the fluid's part of the balance, $\lambda L_z$ against the torques on the boundaries. Its `resid0` stays at $10^{-4}$ to $10^{-3}$ with `rotdyn = 1`, because the work done by the moving boundaries on the fluid is not included in that balance yet (see [Checking the results](output.md#checking-the-results)).
