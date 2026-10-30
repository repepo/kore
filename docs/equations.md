# Equations and scaling

Kore solves the *linearised* equations for small perturbations of a state of rest in the rotating frame: the flow $\mathbf{u}$, the magnetic field $\mathbf{b}$ (on top of the imposed background field $\mathbf{B}_0$, see [Background magnetic field](background-field.md)), the temperature $\theta$ and the composition $\xi$ (on top of background profiles $T(r)$ and $C(r)$). The Lorentz force and the induction term are linearised about $\mathbf{B}_0$:

$$
(\nabla\times\mathbf{B})\times\mathbf{B} \;\longrightarrow\; (\nabla\times\mathbf{b})\times\mathbf{B}_0 + (\nabla\times\mathbf{B}_0)\times\mathbf{b},
\qquad
\nabla\times(\mathbf{u}\times\mathbf{B}) \;\longrightarrow\; \nabla\times(\mathbf{u}\times\mathbf{B}_0).
$$

Every perturbation varies as $e^{\lambda t + i m \phi}$. In an eigenvalue problem $\lambda = \sigma + i\omega$ is unknown, with damping rate $\sigma$ and frequency $\omega$. In a forced problem $\lambda = i\omega$ is the forcing frequency.


## Non-dimensionalization

For numerical convenience we use *dimensionless* variables when solving the Navier Stokes equation, the induction equation, the heat equation and the composition equation. A non-dimensionalization procedure is always possible thanks to the Buckingham-$\pi$ theorem. The unit of length $L$ is the CMB radius.



### The momentum equation

In a reference frame rotating with angular speed $\Omega$, the dimensional form of the *linear* momentum equation (i.e. the Navier-Stokes equation) describing the acceleration $\partial_t \mathbf{u}$ of a small fluid parcel with density $\rho$, including buoyancy and the Lorentz force, is:  

$$
\rho\partial_t \mathbf{u} +2\rho\,\mathbf{\Omega}\times\mathbf{u}=-\nabla p +\frac{1}{\mu_0}(\nabla \times \mathbf{B})\times \mathbf{B} + \rho\mathbf{g}+\rho\nu\nabla^2{\mathbf{u}},
$$

where $p$ is the reduced pressure, $\mathbf{B}$ the magnetic field, and $\nu$ is the kinematic viscosity of the fluid. Assume small density perturbations $\rho'$ following

$$
\rho=\rho_0+\rho'=\rho_0-\rho_0\alpha\theta,
$$

where $\alpha$ is the fluid's thermal expansion coefficient and $\theta$ is the temperature variation from the isentropic temperature profile $T(r)$.
Assume also a gravitational acceleration following

$$
\mathbf{g}=-g_0\frac{r}{R}\mathbf{\hat r}.
$$

Within the *Boussinesq approximation* the density variations enter only through the buoyancy force, so

$$
\rho\mathbf{g} \longrightarrow \rho' \mathbf{g} = \rho_0 \alpha g_0 \frac{r}{R} \theta \mathbf{\hat r}.
$$

Now we make the dimensional units explicit. With $L$ being the unit of length, $\tau$ the unit of time,  $\theta^*$ the unit of temperature, $P^*$ the unit of pressure, and $\mathbf{\hat z}$ the unit vector along $\mathbf{\Omega}$, then the momentum equation, after dividing by $\rho_0$, is

$$
\frac{L}{\tau^2}\,\partial_t \mathbf{u} +L\frac{\Omega}{\tau}\, 2\mathbf{\hat z}\times\mathbf{u}=-\frac{P^*}{\rho_0 L} \nabla p +  \frac{B_0^2}{L\rho_0 \mu_0}(\nabla \times \mathbf{B})\times \mathbf{B}          +  \frac{\alpha g_0 \theta^*}{R} L r \theta\mathbf{\hat r}+\frac{\nu }{\tau L}\nabla^2 \mathbf{u},
$$

where it is understood that the variables $r, t, \mathbf{u}, p, \theta, \mathbf{B}$ are now *dimensionless*. Multiply the equation by $\tau^2/L$ and get

$$
\partial_t \mathbf{u} + 2\,\Omega\tau\,\mathbf{\hat z}\times\mathbf{u}=- \nabla p + \frac{\tau^2 B_0^2}{L^2\rho_0 \mu_0}(\nabla \times \mathbf{B})\times \mathbf{B} + \tau^2 \frac{\alpha g_0 \theta^*}{R} r \theta \mathbf{\hat r}+\tau\frac{\nu}{L^2}\nabla^2 \mathbf{u}.
$$

Above we have chosen the pressure scale as $P^*=\rho_0 L^2/\tau^2$. For the time being we leave the temperature scale $\theta^*$ unspecified, and define the *Ekman number* $E$ as

$$
E \equiv \frac{\nu}{\Omega L^2},
$$

The *Lehnert number* $Le$ as

$$
Le \equiv \frac{B_0}{\Omega L \sqrt{\rho_0\mu_0}},
$$

the *Rayleigh number* $Ra$ as

$$
Ra \equiv \frac{\alpha g_0 \theta^* L^4}{\nu\kappa R},
$$

and the *Prandtl number* $Pr$ as

$$
Pr \equiv \frac{\nu}{\kappa},
$$

where $\kappa$ is the thermal diffusivity of the fluid So, the momentum equation becomes

$$
\partial_t \mathbf{u} + 2\,(\Omega\tau)\,\mathbf{\hat z}\times\mathbf{u}=-\nabla p + (\Omega\tau)^2 Le^2(\nabla \times \mathbf{B})\times \mathbf{B}  + (\Omega\tau)^2 E^2\frac{Ra}{Pr}\,\theta\, r\, \mathbf{\hat r}+(\Omega\tau)\,E\,\nabla^2 \mathbf{u}.
$$

Alternatively, if we deal with problems without viscosity or thermal diffusion where the Rayleigh number diverges, it is better to define a *reference Brunt-Väisälä frequency* $N_0$ such that

$$
N_0^2 \equiv -\frac{\alpha g_0 \theta^*}{R}.
$$

If the Rayleigh number is finite then we can write

$$
E^2\frac{Ra}{Pr} = -\frac{N_0^2}{\Omega^2}.
$$

The momentum equation, using the reference Brunt-Väisälä frequency reads

$$
\partial_t \mathbf{u} + 2\,(\Omega\tau)\,\mathbf{\hat z}\times\mathbf{u}=-\nabla p + (\Omega\tau)^2 Le^2(\nabla \times \mathbf{B})\times \mathbf{B}  - (\Omega\tau)^2 \frac{N_0^2}{\Omega^2}\,\theta\, r\, \mathbf{\hat r}+(\Omega\tau)\,E\,\nabla^2 \mathbf{u}.
$$


A common choice for the time scale is the rotation time scale, so $\tau=1/\Omega$ and the $(\Omega\tau)$ factors go away. Another choice is the viscous diffusion time scale, with $\tau=L^2/\nu$, in which case $\Omega\tau=1/E$. Yet another choice is the Alfvén wave time scale, with $\tau=L \sqrt{\mu_0\rho_0}/B_0$ so that $\Omega\tau=1/Le$.



### The induction equation

The induction equation in dimensional form is

$$
\partial_t \mathbf{B} = \nabla \times (\mathbf{u} \times \mathbf{B}) + \eta \nabla^2 \mathbf{B},
$$

where $\eta$ is the magnetic diffusivity. Making the dimensional scale factors explicit we get

$$
\frac{B_0}{\tau} \partial_t \mathbf{B} = \frac{B_0}{\tau} \nabla \times (\mathbf{u} \times \mathbf{B}) + \eta \frac{B_0}{L^2}  \nabla^2\mathbf{B},
$$

where $\mathbf{u}, \mathbf{B}, t$ are now dimensionless. Multiply now by $\tau/B_0$ and obtain

$$
\partial_t \mathbf{B} = \nabla \times (\mathbf{u} \times \mathbf{B}) + (\Omega\tau)E_\eta \nabla^2 \mathbf{B},
$$

where $E_\eta$ is the *magnetic Ekman number* defined as

$$
E_\eta \equiv \frac{\eta}{\Omega L^2}.
$$



### The heat equation

The heat equation in its dimensional form is

$$
\partial_t \theta=-\mathbf{u}\cdot\nabla T+\kappa \nabla^2 \theta.
$$

We assume that an isentropic temperature background $T(r)$ exists, which is only a function of radius. Its gradient is then $\nabla T=\mathbf{\hat r}\,\mathrm{d}T/\mathrm{d}r$. Now we write $\mathrm{d}T/\mathrm{d}r=C\,f(r)$, where $C$ is a scale factor for the gradient (can be negative) and $f(r)$ is a dimensionless function of $r$ (with $r$ also dimensionless). Then we can write the heat equation, using this time dimensionless variables exclusively as

$$
\partial_t \theta=-\frac{LC}{\theta^*}\,u_r\,f(r)+(\Omega\tau)\frac{E}{Pr} \nabla^2 \theta.
$$

In **`kore`** we choose always the temperature scale as $\theta^*=-LC$ so that the heat equation reads simply

$$
\partial_t \theta=u_r\,f(r)+(\Omega\tau)\frac{E}{Pr} \nabla^2 \theta.
$$

A temperature profile with a linear gradient is common in the literature. In that case $\partial_r T=-\beta r$ (dimensional). In dimensionless variables this is $-\beta L r$ ($r$ now dimensionless), so that the temperature scale is $\theta^*=\beta L^2$. And if the length scale is the CMB radius, i.e. $L=R$, then the Rayleigh number becomes

$$
Ra = \frac{\alpha g_0 \beta R^5}{\nu\kappa}.
$$


### The composition equation

The composition $\xi$ is treated like the temperature, with the compositional diffusivity $D$ in place of $\kappa$ and a background gradient $\mathrm{d}C/\mathrm{d}r$ (`comp_background`). With the composition scale chosen in the same way, the dimensionless equation is

$$
\partial_t \xi = u_r\, f_\xi(r) + (\Omega\tau)\frac{E}{Sc} \nabla^2 \xi,
$$

where $Sc = \nu/D$ is the Schmidt number. Composition adds a buoyancy term $-(\Omega\tau)^2 \frac{N_\xi^2}{\Omega^2}\,\xi\, r\, \mathbf{\hat r}$ to the momentum equation, with $N_\xi^2/\Omega^2 = -E^2 Ra_\xi/Sc$ (`BV2_comp`) and $Ra_\xi$ the compositional Rayleigh number (`Ra_comp`).


### Induction in a conducting inner core

In a solid, conducting inner core there is no flow, and the field there, $\mathbf{b}_i$, only diffuses:

$$
\partial_t \mathbf{b}_i = (\Omega\tau)\, \frac{E_\eta}{\mu_{i2o}\,\sigma_{i2o}}\, \nabla^2 \mathbf{b}_i,
$$

where $\sigma_{i2o}$ and $\mu_{i2o}$ are the conductivity and permeability of the inner core relative to the fluid. The conditions that join $\mathbf{b}_i$ to the field in the fluid are on [Inner core and mantle](boundaries.md#conducting-inner-core).


## The equations Kore solves

Collecting the above, the dimensionless equations are

$$
\begin{aligned}
\lambda\,\mathbf{u} = {}& - 2\,(\Omega\tau)\,\mathbf{\hat z}\times\mathbf{u} - \nabla p + (\Omega\tau)^2 Le^2 \left[ (\nabla\times\mathbf{b})\times\mathbf{B}_0 + (\nabla\times\mathbf{B}_0)\times\mathbf{b} \right] \\
& - (\Omega\tau)^2 \left( \frac{N_0^2}{\Omega^2}\,\theta + \frac{N_\xi^2}{\Omega^2}\,\xi \right) r\,\mathbf{\hat r} + (\Omega\tau)\,E\,\nabla^2\mathbf{u},
\end{aligned}
$$

$$
\lambda\,\mathbf{b} = \nabla\times(\mathbf{u}\times\mathbf{B}_0) + (\Omega\tau)\,E_\eta\,\nabla^2\mathbf{b},
$$

$$
\lambda\,\theta = u_r\, f(r) + (\Omega\tau)\,\frac{E}{Pr}\,\nabla^2\theta,
\qquad
\lambda\,\xi = u_r\, f_\xi(r) + (\Omega\tau)\,\frac{E}{Sc}\,\nabla^2\xi,
$$

with $\nabla\cdot\mathbf{u} = \nabla\cdot\mathbf{b} = 0$, plus forcing terms in forced problems. The Coriolis term is left out with `rotation = 0`. With `rotdyn = 1` three more equations describe the rotation of the mantle and the inner core (see [Rotational dynamics](rotdyn.md)). The parameters map to Kore's names as follows:

| Symbol | Parameter | Definition |
|---|---|---|
| $\Omega\tau$ | `OmgTau` | Set by `timescale` (see [Setting up a problem](setup.md#time-scale)) |
| $E$ | `Ek` | $\nu/\Omega L^2$ |
| $E_\eta$ | `Em` | $\eta/\Omega L^2$ |
| $Le$ | `Le` (`Le2` = $Le^2$) | $B_0/\Omega L\sqrt{\rho\mu_0}$ |
| $\Lambda$, $Pm$ | `Lambda`, `Pm` | $Le^2/E_\eta$ and $E/E_\eta$ |
| $N_0^2/\Omega^2$ | `BV2` | $-E^2 Ra/Pr$ |
| $N_\xi^2/\Omega^2$ | `BV2_comp` | $-E^2 Ra_\xi/Sc$ |
| $Pr$, $Sc$ | `Prandtl`, `Schmidt` | $\nu/\kappa$, $\nu/D$ |

The boundary conditions are described in [Setting up a problem](setup.md#flow-boundary-conditions-and-ekman-number) (flow, temperature, composition) and [Inner core and mantle](boundaries.md) (magnetic field).


## Angular momentum balance

For an axisymmetric, equatorially symmetric problem ($m = 0$, `symm = 1`) the fluid, the mantle and the inner core exchange axial angular momentum through the viscous and electromagnetic stresses on the boundaries. Everything below is dimensionless, with angular momentum in units of $\rho L^5/\tau$ and torques in units of $\rho L^5/\tau^2$, and everything is a complex amplitude of $e^{\lambda t}$: the balances are linear, so no complex conjugates are added.


### The fluid

Only the axisymmetric toroidal flow of degree 1, $u_\phi = t_1(r)\sin\theta$, carries axial angular momentum:

$$
L_z = \int s\, u_\phi\, dV = \frac{8\pi}{3} \int_{r_i}^{1} r^3\, t_1(r)\, dr ,
$$

with $s = r\sin\theta$ the distance to the axis. Its rate of change is the total axial torque on the fluid. The Coriolis force and the pressure exert no axial torque on the fluid as a whole, and the torque of the Lorentz force reduces to the Maxwell stress on the boundaries, so the fluid only exchanges angular momentum with the mantle and the inner core:

$$
\lambda\, L_z = - \left( \Gamma^{\nu}_M + \Gamma^{B}_M + \Gamma^{\nu}_{IC} + \Gamma^{B}_{IC} \right),
$$

where $\Gamma^{\nu}$ and $\Gamma^{B}$ are the viscous and electromagnetic torques that the fluid exerts on the mantle ($M$) and on the inner core ($IC$). This is the balance that `spin_doctor`'s `resid𝐋` checks, and it holds whether or not the boundaries are free to rotate.


### The torques

The viscous torque is the surface integral of the azimuthal viscous stress $\sigma_{r\phi} = r\,\partial_r (u_\phi/r)$, which for the degree-1 flow depends on $t_1$ at the boundary only:

$$
\Gamma^{\nu}_M = -\,E\,(\Omega\tau) \oint_{CMB} s\, \sigma_{r\phi}\, dS = -\frac{8\pi}{3}\, E\,(\Omega\tau)\, \left( r\, t_1' - t_1 \right)_{r = 1},
$$

$$
\Gamma^{\nu}_{IC} = +\,E\,(\Omega\tau) \oint_{ICB} s\, \sigma_{r\phi}\, dS = \frac{8\pi}{3}\, E\,(\Omega\tau)\, r_i^2 \left( r\, t_1' - t_1 \right)_{r = r_i} .
$$

The electromagnetic torque is the surface integral of the Maxwell stress $B_{0r}\, b_\phi$ of the poloidal background field:

$$
\Gamma^{B}_M = -\,Le^2 (\Omega\tau)^2 \oint_{CMB} s\, B_{0r}\, b_\phi\, dS, \qquad
\Gamma^{B}_{IC} = +\,Le^2 (\Omega\tau)^2 \oint_{ICB} s\, B_{0r}\, b_\phi\, dS .
$$

It is nonzero only on a body in which currents can flow: the mantle with a thin conducting wall, the inner core when it is conducting or has a thin wall. Since $B_{0r} \propto Y_l^0$ and $b_\phi$ comes from the toroidal scalars of $\mathbf{b}$, the integral selects a few toroidal degrees: $g_2$ at the boundary for a degree-1 background field (`'axial'`, the dipoles, FDM with `B0_l = 1`), and $g_1$ and $g_3$ for the quadrupole `'Luo_S2'`. `spin_doctor` writes these four torques, as complex amplitudes, to `flow.dat` and `magnetic.dat`.

A toroidal background field (`B0_tor`, see [Background magnetic field](background-field.md#toroidal-field-b0_tor)) adds the stress $b_r B_{0\phi}$ wherever $k \ne 0$ at a wall, conducting or not (for $m = 0$ it involves $f_2$ at the wall for `B0_tor_l = 1`, $f_1$ and $f_3$ for `B0_tor_l = 2`). Its torque on the fluid enters `resid𝐋`; the torques on the mantle and the inner core above leave it out, which is why `rotdyn = 1` needs $k = 0$ at the walls.

For $m = 1$, `symm = -1`, `resid𝐋` checks the equatorial components of the same balance. The angular momentum is then the complex amplitude $L = \frac{16\pi}{3}\int_{r_i}^{1} r^3\, t_1(r)\, dr$ of the degree-1 toroidal flow, and the Coriolis torque of the rotating frame enters:

$$
\lambda\, L = i\,(\Omega\tau)\, L - \left( \Gamma^{\nu}_M + \Gamma^{\nu}_{IC} + \Gamma^{B}_M \right) + \Gamma^{B}_{w} .
$$

$\Gamma^{B}_{w}$ is the Lorentz torque on the fluid that $\Gamma^{B}_M$ leaves out, the Maxwell stress on the fluid side of its walls (outward normal of the fluid):

$$
\Gamma^{B}_{w} = Le^2 (\Omega\tau)^2 \oint \mathbf{r}\times\left( b_r\, \mathbf{B}_0 + B_{0r}\, \mathbf{b} \right) dS .
$$

`spin_doctor` evaluates it at an insulating CMB for `B0 = 'axial'`, whose sources lie outside (it is the torque $\mathbf{m}\times\mathbf{B}_0$ on the induced dipole $\mathbf{m}$), and at the ICB of a conducting or thin-wall inner core, or of one that holds currents or sources of $\mathbf{B}_0$ (`B0_ic = 'full sphere'`, `B0 = 'dipole'`). At a thin-wall mantle its CMB part equals $-\Gamma^{B}_M$, already in the balance. For $m = 0$ these terms vanish ($b_\phi = 0$ at an insulating wall) or are $-\Gamma^{B}_{IC}$; the stress of a toroidal field above is part of $\Gamma^{B}_{w}$ for both $m$.


### The mantle and the inner core

With `rotdyn = 0` the mantle and the inner core rotate with the frame, and the torques above are simply what the fluid exerts on them. With `rotdyn = 1` they respond to those torques, to the gravitational torque between them, and to the viscous relaxation of the inner core (see [Rotational dynamics](rotdyn.md)):

$$
\lambda\, I_M\, \omega_M = \Gamma^{\nu}_M + \Gamma^{B}_M + K \varphi,
$$

$$
\lambda\, I_{IC}\, \omega_{IC} = \Gamma^{\nu}_{IC} + \Gamma^{B}_{IC} - K \varphi,
$$

$$
\lambda\, \varphi = \omega_{IC} - \omega_M - \frac{\Omega\tau}{\Omega\tau_{IC}}\, \varphi ,
$$

with $I_M$ and $I_{IC}$ the axial moments of inertia (`MoIZ_M`, `MoIZ_IC`, in units of $\rho L^5$), $\omega_M$ and $\omega_{IC}$ the rotation rates relative to the frame, $\varphi$ the longitude of the inner core relative to the mantle, $K$ = `gTorque`$\,(\Omega\tau)^2$ the gravitational coupling constant and $\Omega\tau_{IC}$ = `OmgtauIC`. The solid bodies enter the fluid's equations through the no-slip condition on $t_1$: $t_1(1) = \omega_M$ and $t_1(r_i) = r_i\,\omega_{IC}$ instead of zero.


### Conservation

Adding the three balances, the gravitational torque cancels (it is internal to the mantle-inner core system) and the boundary torques cancel against the fluid's, so the total axial angular momentum is conserved:

$$
\lambda \left( L_z + I_M\, \omega_M + I_{IC}\, \omega_{IC} \right) = 0 .
$$

For a mode with $\lambda \neq 0$ the total must therefore vanish, and Kore's eigenmodes satisfy this to discretisation error (about $10^{-4}$ at $N = 48$, improving with resolution). With `rotdyn = 0` the mantle and the inner core are infinitely heavy, and the fluid's angular momentum alone decays or oscillates at the rate set by the torques.
