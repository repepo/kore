# Non-dimensionalization

For numerical convenience we render *dimensionless* the Navier Stokes equation, the induction equation, and the heat equation. Such procedure is possible thanks to the Buckingham-$\pi$ theorem.

In a reference frame rotating with angular speed $\Omega$ the dimensional form of the *linear* Navier-Stokes (NS) equation describing the acceleration $\partial_t \mathbf{u}$ of a small fluid parcel with density $\rho$ is 

$$
\rho\partial_t \mathbf{u} +2\rho\mathbf{\Omega}\times\mathbf{u}=-\nabla P+\rho\mathbf{g}+\rho\nu\nabla^2{\mathbf{u}},
$$

where $\nu$ is the kinematic viscosity of the fluid. Assume small density perturbations $\rho'$ following

$$
\rho=\rho_0+\rho'=\rho_0-\rho_0\alpha\theta,
$$

where $\alpha$ is the fluid's thermal expansion coefficient and $\theta$ is the temperature variation from the isentropic temperature profile $T(r)$.
Assume also a gravitational acceleration following

$$
\mathbf{g}=-g_0\frac{r}{R}\mathbf{\hat r}.
$$

Within the Boussinesq approximation the density variations enter only through the buoyancy force, so

$$
\rho\mathbf{g} \longrightarrow \rho' \mathbf{g} = \rho_0 \alpha g_0 \frac{r}{R} \theta \mathbf{\hat r}.
$$

Now we make the dimensional units explicit. With $R$ being the unit of length, $1/\Omega$ the unit of time,  $\theta^*$ the unit of temperature, $P^*$ the unit of pressure, and $\mathbf{\hat z}$ the unit vector along $\mathbf{\hat \Omega}$, then the NS equation, after dividing by $\rho_0$, is

$$
\Omega^2 R\,\partial_t \mathbf{u} +\Omega^2 R\,2\mathbf{\hat z}\times\mathbf{u}=-\frac{P^*}{\rho_0 R} \nabla P+\alpha g_0 \theta^* r \theta \mathbf{\hat r}+\frac{\nu \Omega}{R}\nabla^2 \mathbf{u},
$$

where now the variables $r, t, \mathbf{u}, P, \theta$ have been rendered \emph{dimensionless}. Divide now the equation by $\Omega^2 R$ and get

$$
\partial_t \mathbf{u} + 2\mathbf{\hat z}\times\mathbf{u}=-\frac{P^*}{\rho_0 \Omega^2 R^2} \nabla P+\frac{\alpha g_0 \theta^*}{\Omega^2 R} r \theta \mathbf{\hat r}+\frac{\nu}{\Omega R^2}\nabla^2 \mathbf{u}.
$$

We are free to choose the scales $P^*$ and $\theta^*$ as we wish. We choose then $P*=\rho_0\Omega^2 R^2$. For the time being we leave $\theta^*$ unspecified, and define the *Ekman number* as

$$
E=\frac{\nu}{\Omega R^2},
$$

The Rayleigh number as

$$
Ra \equiv \frac{\alpha g_0 \theta^* R^3}{\nu\kappa},
$$

and the *Prandtl number* as

$$
Pr = \frac{\nu}{\kappa},
$$

where $\kappa$ is the thermal diffusivity of the fluid. So, the NS equation becomes

$$
\partial_t \mathbf{u} + 2\mathbf{\hat z}\times\mathbf{u}=-\nabla P+ Ra\frac{E^2}{Pr}\,\theta\, r\, \mathbf{\hat r}+E\nabla^2 \mathbf{u}.
$$
