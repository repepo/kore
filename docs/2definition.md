# Physical definition

Kore solves the flow inside near-spherical rotating bodies in a wide range of scenarios pertaining to the context of planetary liquid cores. The basic physical problem considers the fluid body rotating to a large extent as a rigid body with angular velocity $\mathbf\Omega$, and small velocity perturbations $\mathbf u$ exist over the background field $\mathbf\Omega\times\mathbf r$. Kore assumes this perturbation to be oscillatory and very small compared to the maximum value of $\mathbf\Omega\times\mathbf r$ attained in the volume's domain. In the most basic of scenarios, Kore solves the linearized and incompressible Navier-Stokes momentum equation and finds $\mathbf u$. In the absence of external forcings, the equation can be solved as an eigenvalue problem, for which the frequencies $\omega$ of oscillation of $\mathbf u$ are obtained together with their respective eigenmodes - characteristic velocity profiles, see Numerical Scheme for more information.

Below, the basic equations defining each problem will be provided, together with their respective boundary conditions. All are given in their homogeneous version - i.e. for the resolution of the egienvalue problem. Forcings will be covered in Numerical Scheme.

## Momentum equation

The most basic of problems poses the Navier-Stokes momentum equation. Under the assumption of oscillatory velocity perturbations, one can write $\mathbf u \sim e^{\lambda t}$, where $\lambda\in\mathbb C$ encompasses both the damped and oscillatory characteristics of the motion. Time derivatives of $\mathbf u$ then turn into products by $\lambda$. With this notation, the momentum equation in a reference frame rotating with the container/planet at angular speed $\mathbf\Omega$ - the *mantle frame* - is written as:

$$
\lambda\rho\mathbf u + 2\rho\mathbf\Omega\times\mathbf u = -\nabla p + \rho\mathbf g + \rho\nu\nabla^2\mathbf u
$$

Here, $\rho$ and $\nu > 0$ are the fluid's density and kinematic viscosity respectively, while $p$ is the reduced pressure - sum of physical pressure and centrifugal potential - and $\mathbf g$ is the acceleration of gravity. Note that, in this equation, $\mathbf u$ is a complex vector that does **not** depend on time.

In the general case, the momentum equation is imposed in a (near-)spherical shell of inner radius $r_{icb}$ and outer radius $r_{cmb}$ - the *inner core boundary* (ICB) and the *core mantle boundary* (CMB). This scenario then requires the no-penetration and no-slip boundary condition at both surfaces. Under the assumption that the inner core and mantle rotate at the same angular speed $\mathbf\Omega$, this boundary condition is trivially written as $\mathbf u_{icb} = \mathbf u_{cmb} = \mathbf 0$. At the moment, Kore cannot handle differential rotation between the inner core and mantle. Alternatively, Kore allows to set stress-free boundary conditions, so that the radial derivative of $\mathbf u$ vanishes at the boundaries. The boundary conditions at the ICB and the CMB need not be the same. TODO: La stress-free boundary condition no es compatible con un flujo mínimamente viscoso, no?

If the physical model lacks an inner core ($r_{icb} = 0$), the inviscid momentum equation can be considered, i.e. $\nu = 0$. This problem only has one boundary, namely the CBM, where the only feasible boundary condition is now the stress-free BC. The second boundary condition required to complete the problem is that of regularity at the origin.

Currently, Kore can only work with constantly rotating bodies, so that one can write $\mathbf\Omega = \Omega\hat{\mathbf z}$. Here, $\Omega$ is just the angular rate of rotation while $\hat{\mathbf z}$ is the unit vector along the body's rotation axis.

## Induction equation

On Earth, the liquid core is conductive iron and its motion produces a magnetic field. In the presence of an imposed background magnetic field $\mathbf B_o$, the small oscilations $\mathbf u$ will induce a similarly small and oscillatiory magnetic field perturbation $\mathbf b \sim e^{\lambda t}$, with magnitude much smaller than $\mathbf B_o$. The evolution of this magnetic perturbation is given by the electromagnetic induction equation as (see Triana et al. 2021a):

$$
\lambda\mathbf b = \nabla\times(\mathbf u\times\mathbf B_o) + \eta\nabla^2\mathbf b
$$

Here, $\eta$ is the fluid's magnetic diffusivity, while $\mathbf b$ is again a complex vector with a purely spatial dependance.

The magnetic boundary conditions are many. TODO: I'm not sure how many there are, and I don't really understand them all...

In the presence of magnetic fields, the conductive iron will be subject to an electromagnetic force $\mathbf F_{em}$ (the Lorentz force), which should be added to the right hand side of the momentum equation. This force is written as:

$$
\mathbf F_{em} = \frac{1}{\mu_o}(\nabla\times\mathbf b)\times\mathbf B_o
$$

Here, $\mu_o$ is the magnetic permeability. The momentum equation - or its boundary conditions - are not affected any further.

## Heat equation

Heat distribution becomes important when either the eigenfrequencies or the forcings share the same time scales as, most notably, heat convection. Kore can solve the linearized heat equation:

$$
\lambda\theta = -\mathbf u \cdot \nabla T_o + \kappa\nabla^2\theta
$$

Here, $\theta$ is a small and oscillatory temperature deviation from a background isentropic temperature profile $T_o$, which is assumed by Kore to be dependent only on $r$. The resulting *adiabatic temperature gradient* then reduces to a radial derivative, while its dot product with $\mathbf u$ will only involve the radial velocity. Kore implements three different possibilities for this gradient. On the other hand, $\kappa$ is the thermal diffusivity of the medium.

Kore implements the constant-temperature and constant-flux boundary conditions - a prescribed value of $\theta$ or its derivative, respectively. Either can be chosen at the ICB or the CMB, but note that imposing a constant heat flux at both boundaries results in an ill-posed problem.

Mathematically speaking, this equation could be solved independently. Physically, temperature changes will trigger density changes in the fluid. In Kore, this coupling is implemented through the *Boussinesq approximation*, by which the only real impact of a density change is in the buoyancy term $\rho\mathbf g$ of the momentum equation. In the Boussinesq approximation, this term is written as:

$$
\rho\mathbf g \longrightarrow \rho'\mathbf g = -\rho\alpha\theta\mathbf g
$$

Here, $\rho' = -\rho\alpha\theta$ is the density perturbation caused by the respective temperature perturbation $\theta$, which is proportional to $\rho$ itself and the fluid's *expansion coefficient* $\alpha$.

## Compositional equation

TODO: Coming soon...
