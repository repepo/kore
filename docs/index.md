# Introduction


KOR-ee, from the greek Κόρη, the queen of the underworld, daughter of Zeus and Demeter. Kore is a numerical tool to study the flows inside rapidly rotating planets and stars, or any rotating fluid enclosed in a near-spherical container.

Kore solves the linearized equations of a rotating fluid whose density may vary with radius (the anelastic approximation), with an equation for the entropy and, optionally, an imposed magnetic field. The fluid fills a spherical shell or a full sphere. Kore assumes that every quantity oscillates in time at a single complex frequency. It can either find this frequency, as the eigenvalue of a free oscillation mode (for example the inertial modes of a rotating fluid, in which the Coriolis force is the main restoring force), or impose it, to compute the response to an external forcing such as a tide.

Kore uses a spectral method with Gegenbauer (ultraspherical) polynomials in radius. Unlike traditional Chebyshev collocation, this gives sparse matrices, which need much less memory. This makes it possible to study flows at extremely small viscosities (small Ekman numbers).

To start, see [Installation](installation.md) and [Getting started](getting-started.md). The user guide then takes the features one by one, and [Combining features](compatibility.md) shows which of them work together.

Kore is free for everyone to use, with no restrictions. Too often in the scientific literature the numerical methods used are presented with enough detail to guarantee reproducibility, but only in principle. Without access to the actual implementation of those methods, which would require a significant amount of work and time to develop, readers are left effectively without the possibility to reproduce or verify the results presented. This leads to very slow scientific progress. We share our code to avoid this.

If this code is useful for your research, we invite you to cite the relevant papers (coming soon) and hope that you can also contribute to the project.

## Authors

* **Santiago Andres Triana**, *this implementation*
* **Jeremy Rekier**, *sparse spectral method*
* **Ankit Barik**, *convection, visualization*
* **Fleur Seuren**, *buoyancy*
* **Daoud Laariara**, *differential rotation*
* **Antony Trinh**, *tensor calculus (TenGSHui)*
