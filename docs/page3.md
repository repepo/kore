# Non-dimensionalization

For numerical convenience we render *dimensionless* The Navier Stokes equation, the induction equation, and the heat equation. Such procedure is possible thanks to the Buckingham-Pi theorem.

The dimensional form of the linear Navier-Stokes (NS) equation describing the acceleration $\partial_t \mathbf{u}$ of a small fluid parcel with density $\rho$ is

$$
\rho\partial_t \mathbf{u} +2\rho\mathbf{\Omega}\times\mathbf{u}=-\nabla P+\rho\mathbf{g}+\rho\nu\nabla^2{\mathbf{u}}.
$$

Assume small density perturbations $\rho'$ following

$$
\rho=\rho_0+\rho'=\rho_0-\rho_0\alpha\theta,
$$

where $\alpha$ is the thermal expansion coefficient and $\theta$ is the temperature variation from the isentropic temperature profile $T(r)$.
Assume also

$$
\mathbf{g}=-g_0\frac{r}{R}\mathbf{\hat r}.
$$

Within the Boussinesq approximation, the density variations enter only through the buoyancy force, so

$$
\rho\mathbf{g} \longrightarrow \rho' \mathbf{g} = \rho_0 \alpha g_0 \frac{r}{R} \theta \mathbf{\hat r}.
$$

Now we make the dimensional units explicit. With the unit of length being $R$, the unit of time $1/\Omega$, the unit of temperature $\theta^*$, the unit of pressure $P^*$ then the NS equation, after dividing by $\rho_0$, is

$$
\Omega^2 R\,\partial_t \mathbf{u} +\Omega^2 R\,2\mathbf{\hat z}\times\mathbf{u}=-\frac{P^*}{\rho_0 R} \nabla P+\alpha g_0 \theta^* r \theta \mathbf{\hat r}+\frac{\nu \Omega}{R}\nabla^2 \mathbf{u},
$$

where now the variables $r, t, \mathbf{u}, P, \theta$ have been rendered \emph{dimensionless}. Divide now the equation by $\Omega^2 R$ and get

$$
\partial_t \mathbf{u} + 2\mathbf{\hat z}\times\mathbf{u}=-\frac{P^*}{\rho_0 \Omega^2 R^2} \nabla P+\frac{\alpha g_0 \theta^*}{\Omega^2 R} r \theta \mathbf{\hat r}+\frac{\nu}{\Omega R^2}\nabla^2 \mathbf{u}.
$$

We are free to choose the scales $P^*$ and $\theta^*$ as we wish. We choose then $P*=\rho_0\Omega^2 R^2$. For the time being we leave $\theta^*$ unspecified and simply define the \emph{dimensionless} Brunt-V\"ais\"al\"a frequency $N$ such that

$$
N^2\equiv\frac{\alpha g_0 \theta^*}{\Omega^2 R},
$$

so that the NS equation becomes

$$
\partial_t \mathbf{u} + 2\mathbf{\hat z}\times\mathbf{u}=-\nabla P+ N^2\,\theta r\, \mathbf{\hat r}+E\nabla^2 \mathbf{u},
$$

where $E$ is the Ekman number.
