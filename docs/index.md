# Introduction

*KOR-ee*, from the greek **Κόρη**, the queen of the underworld, daughter of Zeus and Demeter. Kore is a numerical tool to study the core flow within rapidly rotating planets or other rotating fluids contained within near-spherical boundaries.

Kore solves the *linear* Navier-Stokes equation and, optionally, the induction equation and the heat and composition equations (Boussinesq approximation) for a viscous, incompressible and electrically conducting fluid in a rotating spherical shell or full sphere. An externally imposed magnetic field can be included. So can a solid inner core, which can be electrically insulating, conducting, perfectly conducting, or covered by a thin conducting layer. The mantle can be insulating or have a thin conducting layer at its base, and both the mantle and the inner core can rotate in response to the torques the fluid exerts on them.

Kore assumes all dynamical variables to oscillate in a harmonic fashion. The oscillation frequency can be imposed externally, as is the case when doing forced motion studies (e.g. tidal forcing or libration), or it can be obtained as part of the solution to an eigenvalue problem. The eigenmodes Kore finds include the inertial modes (the restoring force is the Coriolis force), the gravity modes (buoyancy), the torsional Alfvén modes (the Lorentz force) and the magneto-Archimedes-Coriolis (MAC) modes (a combination of the three).

Kore's distinctive feature is the use of a very efficient spectral method employing Gegenbauer (also known as ultraspherical) polynomials as a basis in the radial direction. This approach leads to *sparse* matrices representing the differential equations, as opposed to *dense* matrices, as in traditional Chebyshev collocation methods. Sparse matrices have a smaller memory footprint and are more suitable for systematic core flow studies at extremely low viscosities (or small Ekman numbers). The matrices are solved in parallel with PETSc, SLEPc and MUMPS.

Kore is free for everyone to use under the GPL v3 license. Too often in the scientific literature the numerical methods used are presented with enough detail to guarantee reproducibility, but only *in principle*. Without access to the actual implementation of those methods, which would require a significant amount of work and time to develop, readers are left effectively without the possibility to reproduce or verify the results presented. This leads to very slow scientific progress. We share our code to avoid this.

If this code is useful for your research, we invite you to cite the relevant papers (coming soon) and hope that you can also contribute to the project.


## Where to start

| If you want to | Read |
|---|---|
| install PETSc, SLEPc and Kore | [Installation](installation.md) |
| run a first case and see what comes out | [Quick start](quickstart.md) |
| set up your own problem | [Setting up a problem](setup.md), then the other pages of the User guide |
| use an inner core or a conducting mantle | [Inner core and mantle](boundaries.md) |
| choose the imposed magnetic field | [Background magnetic field](background-field.md) |
| understand and check the results | [Output and postprocessing](output.md) |
| see the equations and their scaling | [Equations and scaling](equations.md) |
| reproduce published results | [Examples and tests](examples.md) |

The source code is on [GitHub](https://github.com/repepo/kore). The changes to the code are listed, with how each was checked, in [CHANGELOG.txt](https://github.com/repepo/kore/blob/conductive_IC/CHANGELOG.txt).


## Authors

* **Santiago Andres Triana**, *this implementation*
* **Jeremy Rekier**, *sparse spectral method*
* **Antony Trinh**, *tensor calculus*
* **Ankit Barik**, *convection branch, visualization*
* **Fleur Seuren**, *buoyancy*
