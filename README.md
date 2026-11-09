# Kore

[![Tests](https://github.com/repepo/kore/actions/workflows/anelastic_log.yml/badge.svg?branch=anelastic_log)](https://github.com/repepo/kore/actions/workflows/anelastic_log.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

*KOR-ee*, from the greek **Κόρη**, the queen of the underworld, daughter of Zeus and Demeter. Kore is a numerical tool to study the core flow within rapidly rotating planets or other rotating fluids contained within near-spherical boundaries. The current version solves the *linear* Navier-Stokes equation and optionally the induction and thermal/compositional (Boussinesq) equations for a viscous, incompressible and conductive fluid with an externally imposed magnetic field, and enclosed within a near-spherical boundary. A solid inner core can be included optionally.

Kore assumes all dynamical variables to oscillate in a harmonic fashion. The oscillation frequency can be imposed externally, as is the case when doing forced motion studies (e.g. *tidal* forcing), or it can be obtained as part of the solution to an eigenvalue problem. In Kore's current implementation the eigenmodes comprise the inertial mode set (resoring force is Coriolis), the gravity modes or g mode set (restoring force is buoyancy), the torsional Alfvén mode set (restoring force is the Lorentz force), and the magneto-Archimedes-Coriolis (MAC) set (combination of Coriolis, Lorentz, and buoyancy as restoring forces).

Kore's distinctive feature is the use of a very efficient spectral method employing Gegenbauer (also known as ultraspherical) polynomials as a basis in the radial direction. This approach leads to *sparse* matrices representing the differential equations, as opposed to *dense* matrices, as in traditional Chebyshev colocation methods. Sparse matrices have smaller memory-footprint and are more suitable for systematic core flow studies at extremely low viscosities (or small Ekman numbers).     

Kore is free for everyone to use under the GPL v3 license. Too often in the scientific literature the numerical methods used are presented with enough detail to guarantee reproducibility, but only *in principle*. Without access to the actual implementation of those methods, which would require a significant amount of work and time to develop, readers are left effectively without the possibility to reproduce or verify the results presented. This leads to very slow scientific progress. We share our code to avoid this.

If this code is useful for your research, we invite you to cite the relevant papers (coming soon) and hope that you can also contribute to the project. 

## Documentation

The user manual is online at [repepo.github.io/kore](https://repepo.github.io/kore/). It is built from the `docs/` folder of the repository's default branch. Start with:

* [Installation](https://repepo.github.io/kore/installation/): PETSc, SLEPc and the python packages that kore needs.
* [Getting started](https://repepo.github.io/kore/getting-started/): one complete run, from the parameter file to a checked eigenvalue.
* [Combining features](https://repepo.github.io/kore/compatibility/): which features work together, which combinations kore stops on, and which ones the tests check.

To read the manual of another branch, run `mkdocs serve` in its folder (after `pip install mkdocs-material`).

## Authors

* **Santiago Andres Triana** - *This implementation*
* **Jeremy Rekier** - *Sparse spectral method*
* **Ankit Barik** - *Convection, visualization*
* **Fleur Seuren** - *Buoyancy*
* **Daoud Laariara** - *Differential rotation*
* **Antony Trinh** - *Tensor calculus (TenGSHui)*

## License

GPLv3

