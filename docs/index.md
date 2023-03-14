# Introduction


KOR-ee, from the greek Κόρη, the queen of the underworld, daughter of Zeus and Demeter. Kore is a numerical tool to study the core flow within rapidly rotating planets or other rotating fluids contained within near-spherical boundaries. The current version solves the linear Navier-Stokes and induction equations for a viscous, incompressible and conductive fluid with an externally imposed magnetic field, and enclosed within a rotating spherical shell.

Kore assumes all dynamical variables to oscillate in a harmonic fashion. The oscillation frequency can be imposed externally, as is the case when doing forced motion studies (e.g. tidal forcing), or it can be obtained as part of the solution to an eigenvalue problem. In Kore's current implementation, the eigenmodes are the inertial modes of the rotating fluid. Inertial modes are the global modes of a rotating flow in which the Coriolis force participates prominently in the restoring force balance.

Kore's distinctive feature is the use of a very efficient spectral method employing Gegenbauer (also known as ultraspherical) polynomials as a basis in the radial direction. This approach leads to sparse matrices representing the differential equations, as opposed to dense matrices, as in traditional Chebyshev colocation methods. Sparse matrices have smaller memory-footprint and are more suitable for systematic core flow studies at extremely low viscosities (or small Ekman numbers).

Kore is free for everyone to use, with no restrictions. Too often in the scientific literature the numerical methods used are presented with enough detail to guarantee reproducibility, but only in principle. Without access to the actual implementation of those methods, which would require a significant amount of work and time to develop, readers are left effectively without the possibility to reproduce or verify the results presented. This leads to very slow scientific progress. We share our code to avoid this.

If this code is useful for your research, we invite you to cite the relevant papers (coming soon) and hope that you can also contribute to the project.
