# Non-dimensionalization


For numerical convenience we use *dimensionless* variables when solving the Navier Stokes equation, the induction equation, and the heat equation. A non-dimensionalization procedure is always possible thanks to the Buckingham-$\pi$ theorem.



## The momentum equation

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



## The induction equation

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



## The heat equation

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
