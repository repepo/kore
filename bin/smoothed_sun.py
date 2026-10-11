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
    "dlog_density",
    "dlog_pressure",
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
# Enclosed mass, gravity and pressure from hydrostatic equilibrium
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
#
# Everything is done with Chebyshev series on [-1, 1]: x^2 rho_tilde is even and
# entire, so ``_CHEB_DEG`` terms reach round-off; it is integrated term by term
# (m_tilde, odd in x), divided by x^2 = (T_0 + T_2)/2 (m_tilde = O(x^3), so the
# remainder is round-off) for g, multiplied by rho_tilde and integrated again
# for I.  The profiles and all their derivatives are then smooth to round-off,
# which kore's Chebyshev expansions need (a fine-grid trapezoid rule with linear
# interpolation was only accurate to ~1e-5 in g and ~1e-6 in P).
# Negative x is accepted, as kore uses when ricb = 0: rho, P are even in x and
# g, m are odd, exactly, because the wrong-parity coefficients the interpolation
# leaves (~1e-16 of the largest) are zeroed at each step (so g(0) = 0 exactly).
# P/P_c = 1 - I(x)/I(1) is summed as the single series -J(x)/I(1), J = I - I(1),
# which vanishes at x = 1 to round-off and keeps ~1e-17 absolute accuracy up to
# the surface, where P/P_c ~ 1e-8 at x = 0.99.
# --------------------------------------------------------------------------
_CHEB_DEG = 160              # degree of the Chebyshev series (x^2 rho tail ~1e-15 at 150, 5e-16 at 160; 200 changes nothing)
_CHEB = None                 # cache: dict of coefficient arrays


def _cheb():
    """Cached Chebyshev coefficients of rho_tilde, m_tilde, m_tilde/x^2 and I on [-1, 1]."""
    global _CHEB
    if _CHEB is None:
        ch = np.polynomial.chebyshev
        rho_c = ch.chebinterpolate(density, _CHEB_DEG)                       # rho_tilde(x), even
        f_c = ch.chebinterpolate(lambda t: t**2 * density(t), _CHEB_DEG)     # x^2 rho_tilde, even
        rho_c[1::2] = 0.0                                                    # zero the round-off odd parts (~1e-16)
        f_c[1::2] = 0.0                                                      # so that all parities below are exact
        m_c = ch.chebint(f_c, lbnd=0.0)                                      # m_tilde(x), odd
        m_c[0::2] = 0.0
        q_c, _ = ch.chebdiv(m_c, np.array([0.5, 0.0, 0.5]))                  # m_tilde/x^2, odd
        q_c[0::2] = 0.0
        I_c = ch.chebint(ch.chebmul(rho_c, q_c), lbnd=0.0)                   # I(x), even
        I_c[1::2] = 0.0
        I1 = ch.chebval(1.0, I_c)
        J_c = I_c.copy()
        J_c[0] -= I1                                                         # J(x) = I(x) - I(1): P/P_c = -J/I(1), ~1e-16 at x = 1
        _CHEB = dict(rho=rho_c, m=m_c, q=q_c, I=I_c, J=J_c,
                     m1=ch.chebval(1.0, m_c), I1=I1)
    return _CHEB


def _clip(x):
    return np.clip(np.asarray(x, dtype=float), -1.0, 1.0)


def mass(x):
    """Dimensionless enclosed mass m_tilde(x) = integral_0^x x'^2 (rho/RHO_C) dx' (odd in x).

    The physical enclosed mass is ``4*pi*R_SUN**3*RHO_C * mass(x)``;
    ``mass(1.0)`` therefore equals ``M_sun / (4*pi*R_SUN**3*RHO_C)``.
    """
    res = np.polynomial.chebyshev.chebval(_clip(x), _cheb()['m'])
    return res[()] if res.ndim == 0 else res


def pressure(x):
    """Pressure normalised to the central value, ``P(x)/P_c``, vs fractional radius (even in x).

    Obtained by integrating hydrostatic equilibrium for the density expansion,
    normalised so that ``P(0)/P_c = 1`` and ``P(1)/P_c = 0`` (summed as the single
    series ``-J(x)/I(1)``, ``J = I - I(1)``, so it stays accurate to ~1e-17 absolute
    up to the surface, where it vanishes to round-off).

    Parameters
    ----------
    x : float or array_like
        Fractional radius ``r / R_sun`` in [-1, 1].

    Returns
    -------
    numpy.ndarray (or float for scalar input)
        ``P(x) / P_c`` (dimensionless), monotonically decreasing from 1 to 0.
    """
    c = _cheb()
    res = -np.polynomial.chebyshev.chebval(_clip(x), c['J']) / c['I1']
    return res[()] if res.ndim == 0 else res


def central_pressure_cgs():
    """Central pressure implied by the density profile + hydrostatic balance [dyn/cm^2].

    P_c = 4*pi*G*R_SUN**2*RHO_C**2 * I(1).  (Model S value is ~2.3e17.)
    """
    return 4.0 * np.pi * G_CGS * R_SUN**2 * RHO_C**2 * _cheb()['I1']


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
    """Gravitational acceleration normalised to the surface value, ``g(x)/g_surf`` (odd in x).

    Parameters
    ----------
    x : float or array_like
        Fractional radius ``r / R_sun`` in [-1, 1].

    Returns
    -------
    numpy.ndarray (or float for scalar input)
        ``g(x) / g(R_sun) = m_tilde(x) / (x**2 * m_tilde(1))`` (dimensionless),
        equal to 0 at the centre and 1 at the surface.
    """
    c = _cheb()
    res = np.polynomial.chebyshev.chebval(_clip(x), c['q']) / c['m1']
    return res[()] if res.ndim == 0 else res


def surface_gravity_cgs():
    """Surface gravity implied by the density profile [cm/s^2].

    g_surf = 4*pi*G*R_SUN*RHO_C * m_tilde(1) = G*M_sun/R_SUN**2  (~2.74e4).
    """
    return 4.0 * np.pi * G_CGS * R_SUN * RHO_C * _cheb()['m1']


def gravity_cgs(x):
    """Physical gravitational acceleration in cm/s^2: ``surface_gravity_cgs() * gravity(x)``."""
    return surface_gravity_cgs() * gravity(x)


# --------------------------------------------------------------------------
# Logarithmic derivatives with respect to x
# --------------------------------------------------------------------------
def dlog_density(x):
    """``d ln(rho) / dx`` at fractional radius ``x`` (odd in x), from the Chebyshev expansion."""
    c = np.zeros(2 * B.size - 1)
    c[::2] = B                                           # T_{2k} coefficients as a full series
    res = np.polynomial.chebyshev.chebval(_clip(x), np.polynomial.chebyshev.chebder(c))
    return res[()] if res.ndim == 0 else res


def dlog_pressure(x):
    """``d ln(P) / dx`` (odd in x), from hydrostatic balance: ``-(m_tilde(1)/I(1)) rho_tilde g_tilde / (P/P_c)``.

    It diverges at |x| -> 1, where P -> 0.
    """
    c = _cheb()
    x = _clip(x)
    with np.errstate(divide='ignore', invalid='ignore'):                  # -inf at |x| = 1 (P = 0 there)
        res = -(c['m1'] / c['I1']) * density(x) * gravity(x) / pressure(x)
    return res[()] if res.ndim == 0 else res


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

    print(f"{'x':>6} {'rho/rho_c':>13} {'P/P_c':>13} {'g/g_surf':>11} {'dlnrho/dx':>11} {'dlnP/dx':>11}")
    for x0 in (0.0, 0.1, 0.3, 0.5, 0.713, 0.9, 0.99, 1.0):
        print(f"{x0:6.3f} {float(density(x0)):13.4e} {float(pressure(x0)):13.4e} "
              f"{float(gravity(x0)):11.4f} {float(dlog_density(x0)):11.4f} {float(dlog_pressure(x0)):11.4e}")

    # hydrostatic balance check: dP/dx + (m1/I1) rho g = 0, with a finite difference of P
    xs, h = np.linspace(0.05, 0.95, 7), 1e-5
    dP = (pressure(xs + h) - pressure(xs - h)) / (2 * h)
    print(f"\nmax |dP/dx - P dlnP/dx| = {np.max(np.abs(dP - pressure(xs) * dlog_pressure(xs))):.1e}")

    # vectorised call check
    xs = np.linspace(0.0, 1.0, 5)
    print("\nvectorised gravity(linspace(0,1,5)):")
    print(gravity(xs))
