# Non-dimensionalization

In solving the [relevant equations](https://yorchmcm.github.io/kore_sandbox/2definition/), Kore uses *dimensionaless* variables. A non-dimensionalization procedure is always possible thanks to the [Buckingham-$\pi$ theorem](https://en.wikipedia.org/wiki/Buckingham_%CF%80_theorem). For this, the following will be used as units of length, mass, time, temperature respectively: $L$, $\tau$, $\rho L^3$, $\theta^*$. On the other hand, several vectors can be written as the product of a typical value of their magnitude times a non-dimensional vector to provide direction. Thus, one can write $\mathbf g \rightarrow g^*\mathbf g$, $\mathbf B_o \rightarrow B^*\mathbf B_o$ and $\mathbf b \rightarrow B^*\mathbf b$, where the vectors on the left hand sides are dimensional and those on the right hand side are non-dimensional, while the starred quantities are their characteristic values and provide the units. Similarly one can also write $\mathbf\Omega = \Omega\hat{\mathbf z}$ (see [Physical Definition]((https://yorchmcm.github.io/kore_sandbox/2definition/))). Finally, and because it represents an angular frequency, the non-dimensionalization $\lambda \rightarrow \lambda/\tau$ follows.

After dividing the momentum equation by $\rho$, applying all appropriate non-dimensionalizations to all equations and cancelling some common terms, the non-dimensional equations read:

$$ \displaylines{
\lambda\mathbf u + 2(\Omega\tau)\hat{\mathbf z}\times\mathbf u = -\nabla p-(\Omega\tau)^2\frac{\alpha g^*\theta^*}{\Omega^2L}\theta\mathbf g + (\Omega\tau)\frac{\nu}{\Omega L^2}\nabla^2\mathbf u + \frac{(\Omega\tau)^2}{\rho\mu_o}\bigg(\frac{B^*}{\Omega L}\bigg)^2(\nabla\times\mathbf b)\times\mathbf B_o \\
\lambda\mathbf b = \nabla\times(\mathbf u\times\mathbf B_o) + (\Omega\tau)\frac{\eta}{\Omega L^2}\nabla^2\mathbf b \\
\lambda\theta = -\mathbf u\cdot\nabla T_o + (\Omega\tau)\frac{\kappa}{\Omega L^2}\nabla^2\theta
} $$

Here, all vector quantites, derivatives, dependent variables as well as the background temperature profile $T_o$ and the eigenfrequency $\lambda$ are non-dimensional. These equations can be simpified by the introduction of several non-dimensional numbers:

$$ \displaylines{
\text{Ekmann number:}\ \ E = \frac{\nu}{\Omega L^2} \\
\text{Lenhert number:}\ \ Le = \frac{B^*}{\Omega L\sqrt{\rho\mu_o}} \\
\text{Rayleigh number:}\ \ Ra = \frac{\alpha g^*\theta^* L^3}{\nu\kappa} \\
\text{Prandtl number:}\ \ Pr = \frac{\nu}{\kappa} \\
\text{Magnetic Ekmann number:}\ \ E_\eta = \frac{\eta}{\Omega L^2}
} $$

Using these quantities, the equations can be rewritten as:

$$ \displaylines{
\lambda\mathbf u + 2(\Omega\tau)\hat{\mathbf z}\times\mathbf u = -\nabla p-(\Omega\tau E)^2\frac{Ra}{Pr}\theta\mathbf g + \Omega\tau E\nabla^2\mathbf u + (\Omega\tau Le)^2(\nabla\times\mathbf b)\times\mathbf B_o \\
\lambda\mathbf b = \nabla\times(\mathbf u\times\mathbf B_o) + \Omega\tau E_\eta\nabla^2\mathbf b \\
\lambda\theta = -\mathbf u\cdot\nabla T_o + \frac{\Omega\tau E}{Pr}\nabla^2\theta
} $$

Inviscid fluids ($\nu = 0$) don't allow for the use of the Rayleigh number, while perfectly insulating fluids ($\kappa = 0$) don't allow for the use of either the Rayleigh or Prandtl numbers. In this cases, the following non-dimensional number is better suited:

$$
\text{Brunt-Väisälä frequency:}\ \ N_o^2 = -\frac{\alpha g^*\theta^*}{L}
$$

The (non-dimensional) momentum equation is then written as:

$$
\lambda\mathbf u + 2(\Omega\tau)\hat{\mathbf z}\times\mathbf u = -\nabla p-(\Omega\tau)^2\frac{N_o^2}{\Omega^2}\theta\mathbf g + (\Omega\tau E)\nabla^2\mathbf u + (\Omega\tau Le)^2(\nabla\times\mathbf b)\times\mathbf B_o
$$

If both the Rayleigh and Prandtl numbers are defined, the following holds:

$$
E^2\frac{Ra}{Pr} = -\frac{N_o^2}{\Omega^2}
$$

Different non-dimensionalizations are now possible depending of the choice of the time scale $\tau$. Kore supports four different time scales, which are selected by prescribing the value of the $\Omega\tau$ factor in one of the following ways:


## Rotation time scale

This time scale has $\tau = 1/\Omega$, so that $\Omega\tau = 1$. The non-dimensional equations read:

$$ \displaylines{
\lambda\mathbf u + 2\hat{\mathbf z}\times\mathbf u = -\nabla p - E^2\frac{Ra}{Pr}\theta\mathbf g + E\nabla^2\mathbf u + Le^2(\nabla\times\mathbf b)\times\mathbf B_o \\
\lambda\mathbf b = \nabla\times(\mathbf u\times\mathbf B_o) + E_\eta\nabla^2\mathbf b \\
\lambda\theta = -\mathbf u\cdot\nabla T_o + \frac{E}{Pr}\nabla^2\theta
} $$

This time scale is not suitable for problems in which the domain does not rotate. This is not a problem because Kore does not consider this possibility.


## Viscous diffusion time scale

This time scale has

$$\tau = \frac{L^2}{\nu} \longrightarrow \Omega\tau = \frac{1}{E}$$

The non-dimensional equations read:

$$ \displaylines{
\lambda\mathbf u + \frac{2}{E}\hat{\mathbf z}\times\mathbf u = -\nabla p-\frac{Ra}{Pr}\theta\mathbf g + \nabla^2\mathbf u + \bigg(\frac{Le}{E}\bigg)^2(\nabla\times\mathbf b)\times\mathbf B_o \\
\lambda\mathbf b = \nabla\times(\mathbf u\times\mathbf B_o) + \frac{E_\eta}{E}\nabla^2\mathbf b \\
\lambda\theta = -\mathbf u\cdot\nabla T_o + \frac{1}{Pr}\nabla^2\theta
} $$

This time scale is not suitable for inviscid problems, where $\nu = E = 0$. Note also that, given the typical small values of $E$ (in the other of $10^{-15}$ for the Earth's liquid core), this is probably not the most suitable time scale in the general case.


## Magnetic diffusion time scale

This time scale has

$$\tau = \frac{L^2}{\eta} \longrightarrow \Omega\tau = \frac{1}{E_\eta}$$

The non-dimensional equations read:

$$ \displaylines{
\lambda\mathbf u + \frac{2}{E_\eta}\hat{\mathbf z}\times\mathbf u = -\nabla p - \bigg(\frac{E}{E_\eta}\bigg)^2\frac{Ra}{Pr}\theta\mathbf g + \frac{E}{E_\eta}\nabla^2\mathbf u + \bigg(\frac{Le}{E_\eta}\bigg)^2(\nabla\times\mathbf b)\times\mathbf B_o \\
\lambda\mathbf b = \nabla\times(\mathbf u\times\mathbf B_o) + \nabla^2\mathbf b \\
\lambda\theta = -\mathbf u\cdot\nabla T_o + \frac{E}{E_\eta Pr}\nabla^2\theta
} $$

## Alfvén wave time scale

This time scale has

$$\tau = \frac{L\sqrt{\rho\mu_o}}{B^*} \longrightarrow \Omega\tau = \frac{1}{Le}$$

The non-dimensional equations read:

$$ \displaylines{
\lambda\mathbf u + \frac{2}{Le}\hat{\mathbf z}\times\mathbf u = -\nabla p-\bigg(\frac{E}{Le}\bigg)^2\frac{Ra}{Pr}\theta\mathbf g + \frac{E}{Le}\nabla^2\mathbf u + (\nabla\times\mathbf b)\times\mathbf B_o \\
\lambda\mathbf b = \nabla\times(\mathbf u\times\mathbf B_o) + \frac{E_\eta}{Le}\nabla^2\mathbf b \\
\lambda\theta = -\mathbf u\cdot\nabla T_o + \frac{E}{LePr}\nabla^2\theta
} $$
