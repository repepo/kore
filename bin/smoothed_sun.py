"""
smoothed_sun
=============

Smooth, dimensionless approximation to the density profile of the Sun,
based on standard solar **Model S** (Christensen-Dalsgaard et al. 1996).

The natural logarithm of the density, scaled by the central density
``RHO_C``, is represented as an expansion in the **even Chebyshev
polynomials** of the fractional radius ``x = r / R_sun``::

        ln( rho(x) / RHO_C ) = sum_{k=0}^{7} B[k] * T_{2k}(x)

Because only even-order Chebyshev polynomials appear, the expression is an
even function of ``x`` (so ``d rho / dr = 0`` at the centre, as required by
spherical symmetry).  The handy identity

        T_{2k}(x) = T_k(2 x^2 - 1)

lets every term be evaluated from a single ``arccos`` call.

The expansion was obtained by a (weighted) least-squares fit of
``ln(rho/rho_c)`` over ``0 <= x < 1``.  It reproduces the Model S density
to about **6.5 % RMS** in relative terms (worst case ~16 % in the thin
near-surface layers), is strictly positive, and is C-infinity on [0, 1].

Coordinates / units
-------------------
    x       = r / R_sun          dimensionless fractional radius, 0..1
    rho/RHO_C                     dimensionless density
    rho                          g / cm^3   (multiply the above by RHO_C)

Quick start
-----------
    >>> from smoothed_sun import log_density, density, RHO_C
    >>> log_density(0.0)          # ln(rho/rho_c) at the centre  (~ -0.05)
    >>> density(0.5)              # rho/rho_c at mid-radius
    >>> RHO_C * density(0.5)      # rho in g/cm^3 at mid-radius
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "RHO_C",
    "R_SUN",
    "G_CGS",
    "B",
    "log_density",
    "density",
    "density_cgs",
    "mass",
    "pressure",
    "central_pressure_cgs",
    "pressure_cgs",
    "gravity",
    "surface_gravity_cgs",
    "gravity_cgs",
]

# --------------------------------------------------------------------------
# Model S scale factors
# --------------------------------------------------------------------------
#: Central density of Model S, used as the density scale factor [g / cm^3].
RHO_C: float = 1.542365e2

#: Photospheric solar radius of Model S [cm] (handy for converting x <-> r).
R_SUN: float = 6.959906e10

#: Newton's gravitational constant [cgs], used for the pressure integration.
G_CGS: float = 6.674300e-8

# --------------------------------------------------------------------------
# Expansion coefficients
#
#   ln(rho/RHO_C) = sum_k B[k] * T_{2k}(x),   T_{2k}(x) = T_k(2 x^2 - 1)
#
#   B[k] is the coefficient of the even Chebyshev polynomial T_{2k}.
# --------------------------------------------------------------------------
B: np.ndarray = np.array(
    [
        -6.734049,   # b_0   (coeff of T_0)
        -5.685907,   # b_2   (coeff of T_2)
        -0.427676,   # b_4   (coeff of T_4)
        -1.435741,   # b_6   (coeff of T_6)
        -0.507593,   # b_8   (coeff of T_8)
        -0.508516,   # b_10  (coeff of T_10)
        -0.230134,   # b_12  (coeff of T_12)
        -0.218338,   # b_14  (coeff of T_14)
    ]
)


def log_density(x):
    """Dimensionless log density ``ln(rho / RHO_C)`` at fractional radius ``x``.

    Parameters
    ----------
    x : float or array_like
        Fractional radius ``r / R_sun`` in the range [0, 1].

    Returns
    -------
    numpy.ndarray (or float for scalar input)
        ``ln(rho(x) / RHO_C)`` — the dimensionless natural log of the density.

    Notes
    -----
    Evaluated as ``sum_k B[k] * cos(k * arccos(2 x**2 - 1))`` using the
    identity ``T_{2k}(x) = T_k(2 x**2 - 1)``.  The argument of ``arccos`` is
    clipped to [-1, 1] to tolerate tiny floating-point excursions for x
    slightly outside [0, 1].
    """
    x = np.asarray(x, dtype=float)
    t = np.clip(2.0 * x**2 - 1.0, -1.0, 1.0)
    theta = np.arccos(t)
    # sum_k B[k] * cos(k * theta)
    k = np.arange(B.size)
    # broadcast: theta[..., None] * k -> cos -> weighted sum over k
    result = np.cos(theta[..., None] * k) @ B
    return result[()] if result.ndim == 0 else result


def density(x):
    """Dimensionless density ``rho / RHO_C`` at fractional radius ``x``.

    Equivalent to ``exp(log_density(x))``.  Strictly positive.
    """
    return np.exp(log_density(x))


def density_cgs(x):
    """Physical density in g/cm^3 at fractional radius ``x`` (= ``RHO_C * density(x)``)."""
    return RHO_C * density(x)


# --------------------------------------------------------------------------
# Pressure from hydrostatic equilibrium
#
# Hydrostatic balance:           dP/dr = -rho * G m(r) / r^2 ,
#   m(r) = \int_0^r 4 pi r'^2 rho dr'.
#
# In dimensionless radius x = r/R_sun, with rho_tilde = rho/RHO_C and the
# dimensionless mass
#       m_tilde(x) = \int_0^x x'^2 rho_tilde(x') dx'      (so m = 4 pi R_sun^3 RHO_C m_tilde),
# the balance integrates to
#       P(x) = P_c - 4 pi G R_sun^2 RHO_C^2 * I(x),
#       I(x) = \int_0^x rho_tilde(x') m_tilde(x') / x'^2 dx'.
# Requiring the surface pressure to vanish, P(1)=0, fixes the normalisation:
#       P(x)/P_c = 1 - I(x)/I(1),
# which depends only on the (dimensionless) density profile.  The implied
# central pressure is  P_c = 4 pi G R_sun^2 RHO_C^2 * I(1).
# --------------------------------------------------------------------------
_GRID_N = 20001              # number of points in the internal fine grid
_MASS_TABLE = None           # cache: (x_grid, m_tilde_grid)
_PRESSURE_TABLE = None       # cache: (x_grid, P_over_Pc_grid, I_total)


def _cumtrapz(y, x):
    """Cumulative trapezoidal integral of y(x) with the same length as y (out[0]=0)."""
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def _mass_table():
    """Cached fine-grid table of the dimensionless enclosed mass m_tilde(x)."""
    global _MASS_TABLE
    if _MASS_TABLE is None:
        x = np.linspace(0.0, 1.0, _GRID_N)
        m_t = _cumtrapz(x**2 * density(x), x)          # m_tilde(x) = int_0^x x'^2 rho_t dx'
        _MASS_TABLE = (x, m_t)
    return _MASS_TABLE


def _pressure_table():
    """Cached fine-grid table of P/P_c (from hydrostatic equilibrium)."""
    global _PRESSURE_TABLE
    if _PRESSURE_TABLE is None:
        x, m_t = _mass_table()
        rho_t = density(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            integrand = np.where(x > 0.0, rho_t * m_t / x**2, 0.0)
        integrand[0] = 0.0                              # limit is 0 at the centre
        I = _cumtrapz(integrand, x)
        _PRESSURE_TABLE = (x, 1.0 - I / I[-1], I[-1])
    return _PRESSURE_TABLE


def mass(x):
    """Dimensionless enclosed mass m_tilde(x) = integral_0^x x'^2 (rho/RHO_C) dx'.

    The physical enclosed mass is ``4*pi*R_SUN**3*RHO_C * mass(x)``;
    ``mass(1.0)`` therefore equals ``M_sun / (4*pi*R_SUN**3*RHO_C)``.
    """
    gx, m_t = _mass_table()
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    res = np.interp(x, gx, m_t)
    return res[()] if res.ndim == 0 else res


def pressure(x):
    """Pressure normalised to the central value, ``P(x)/P_c``, vs fractional radius.

    Obtained by integrating hydrostatic equilibrium for the density expansion,
    normalised so that ``P(0)/P_c = 1`` and ``P(1)/P_c = 0``.

    Parameters
    ----------
    x : float or array_like
        Fractional radius ``r / R_sun`` in [0, 1].

    Returns
    -------
    numpy.ndarray (or float for scalar input)
        ``P(x) / P_c`` (dimensionless), monotonically decreasing from 1 to 0.
    """
    gx, gP, _ = _pressure_table()
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    res = np.interp(x, gx, gP)
    return res[()] if res.ndim == 0 else res


def central_pressure_cgs():
    """Central pressure implied by the density profile + hydrostatic balance [dyn/cm^2].

    P_c = 4*pi*G*R_SUN**2*RHO_C**2 * I(1).  (Model S value is ~2.3e17.)
    """
    _, _, I_total = _pressure_table()
    return 4.0 * np.pi * G_CGS * R_SUN**2 * RHO_C**2 * I_total


def pressure_cgs(x):
    """Physical pressure in dyn/cm^2 (= erg/cm^3): ``central_pressure_cgs() * pressure(x)``."""
    return central_pressure_cgs() * pressure(x)


# --------------------------------------------------------------------------
# Acceleration of gravity
#
#   g(r) = G m(r) / r^2 .
# With m = 4 pi R_sun^3 RHO_C m_tilde(x) and r = R_sun x,
#   g(x) = 4 pi G R_sun RHO_C * m_tilde(x) / x^2 ,
# so, normalised to the surface value g(R_sun),
#   g(x)/g_surf = m_tilde(x) / (x^2 * m_tilde(1)) .
# This is 0 at the centre, 1 at the surface, and peaks in the deep interior.
# --------------------------------------------------------------------------
def gravity(x):
    """Gravitational acceleration normalised to the surface value, ``g(x)/g_surf``.

    Parameters
    ----------
    x : float or array_like
        Fractional radius ``r / R_sun`` in [0, 1].

    Returns
    -------
    numpy.ndarray (or float for scalar input)
        ``g(x) / g(R_sun) = m_tilde(x) / (x**2 * m_tilde(1))`` (dimensionless),
        equal to 0 at the centre and 1 at the surface.
    """
    gx, m_t = _mass_table()
    with np.errstate(divide="ignore", invalid="ignore"):
        g_grid = np.where(gx > 0.0, m_t / (gx**2 * m_t[-1]), 0.0)
    g_grid[0] = 0.0                                     # g -> 0 at the centre
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    res = np.interp(x, gx, g_grid)
    return res[()] if res.ndim == 0 else res


def surface_gravity_cgs():
    """Surface gravity implied by the density profile [cm/s^2].

    g_surf = 4*pi*G*R_SUN*RHO_C * m_tilde(1) = G*M_sun/R_SUN**2  (~2.74e4).
    """
    _, m_t = _mass_table()
    return 4.0 * np.pi * G_CGS * R_SUN * RHO_C * m_t[-1]


def gravity_cgs(x):
    """Physical gravitational acceleration in cm/s^2: ``surface_gravity_cgs() * gravity(x)``."""
    return surface_gravity_cgs() * gravity(x)


# --------------------------------------------------------------------------
# Simple self-test / demonstration
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print(__doc__.splitlines()[2])  # title line
    print(f"\nRHO_C = {RHO_C:.6e} g/cm^3   R_SUN = {R_SUN:.6e} cm")
    print(f"{len(B)} even-Chebyshev coefficients: T_0 .. T_{2*(len(B)-1)}\n")

    print(f"Implied central pressure P_c = {central_pressure_cgs():.4e} dyn/cm^2 "
          f"(Model S ~ 2.34e17)")
    print(f"Implied surface gravity      = {surface_gravity_cgs():.4e} cm/s^2 "
          f"(Model S ~ 2.74e4)\n")

    print(f"{'x':>6} {'rho/rho_c':>13} {'P/P_c':>13} {'g/g_surf':>11}")
    for x0 in (0.0, 0.1, 0.3, 0.5, 0.713, 0.9, 0.99, 1.0):
        print(f"{x0:6.3f} {float(density(x0)):13.4e} {float(pressure(x0)):13.4e} "
              f"{float(gravity(x0)):11.4f}")

    # vectorised call check
    xs = np.linspace(0.0, 1.0, 5)
    print("\nvectorised gravity(linspace(0,1,5)):")
    print(gravity(xs))
