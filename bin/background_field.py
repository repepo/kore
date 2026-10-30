'''
NEW (2026-10-28): the imposed background magnetic field B0 (magnetic = 1), ported from the conductive_IC branch
(bin/utils.py there: jl, nl, findbeta, h0..h3, B0_norm, symmB0) for the types of B0_TYPES.

B0 = cnorm ∇×∇×(h(r) Y_l0 r), axisymmetric and poloidal, with Y_l0 the Schmidt harmonic P_l(cos θ) of kore. So
B0_r = cnorm l(l+1) h Y_l0/r and ∇×B0 = cnorm (D_l h) ∂θY_l0 φ̂, D_l h = h'' + 2h'/r - l(l+1)h/r².
    'axial'  h = r/2 (l = 1): the uniform field B0 = cnorm ẑ, no current.
    'G21 dipole'  h = r/6 - r³/10 (Gerick et al. 2021), 'Luo_S1' h = (5 - 3r²) r, 'Luo_S2' h = (157 - 296r² + 143r⁴) r² (degree 2)
             (Luo & Jackson 2022): fields of a full sphere; with ricb > 0 and B0_ic = 'potential' (default, as conductive_IC)
             h + β r^-(l+1), β = ricb^(l+1)(ricb h' - l h)/(2l+1) at ricb, so that B0 continues as a potential field into the
             inner core (NEW 2026-10-28)
    'FDM'    the poloidal free decay mode of degree l = B0_l of the shell (Zhang & Fearn, GAFD 1995, eq. 2.7):
             h = j_l(βr) y_{l-1}(β) - j_{l-1}(β) y_l(βr), β the root of findbeta near par.beta. It matches a potential
             field at both walls, so it suits insulating walls.
NEW (2026-10-28): a toroidal field B0_tor = cnorm_tor ∇×(k(r) Y_n0 𝐫), B_φ = -cnorm_tor k ∂θY_n0, n = B0_tor_l, alone (B0 = None)
or added to the poloidal one (same equatorial symmetry only):
    'Malkus' k = r (n = 1): B0 = cnorm_tor s φ̂ (Malkus 1967), J0 = ∇×B0 = 2 cnorm_tor ẑ.
    'wall'   k = r (r - ricb)(1 - r)/(1 - ricb)² (n = 1): zero at both walls; with ricb = 0, k = r^n (1 - r²).
    'user'   k = B0_tor_profile, a numpy expression of r and p = B0_tor_param (its derivatives from a Chebyshev interpolant).
NEW (2026-10-30): 'confined' h = r^l (r2² - r²)^p (r1 = 0) or r^l (r - r1)^p (r2 - r)^p between r1 = B0_conf_r1 and r2 = B0_conf_r2,
    zero outside, l = B0_l (1 or 2), p = B0_conf_p: h and its first p - 1 derivatives vanish at the edges, so the field is zero
    outside the region with no surface currents (B_r ∝ h and B_θ ∝ h' + h/r continuous; the current ∝ h'' + ... continuous for
    p >= 3). The point: a poloidal field inside a region where ΔΩ is uniform is a steady background with the differential
    rotation (Ferraro's law, background.ferraro_mismatch). The kink at the edges is resolved by the Chebyshev series only
    algebraically (coefficients ~ k^-(p+1)): use p = 4 and check convergence in N.
Equatorial symmetry: a poloidal field of degree n has (-1)^n, a toroidal one (-1)^(n+1) (B_φ ∝ sinθ is symmetric for n = 1).
Only parameters, numpy and scipy are imported: utils.py and radial_profiles.py import this module.
'''
import numpy as np
import scipy.special as scsp
import scipy.optimize as so
from parameters import par

# the names the 'user' and 'general' expressions may use (NEW 2026-10-30: defined here, before cnorm needs them at import)
USER_NS = { name: getattr(np, name) for name in ('sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'tanh', 'sinh', 'cosh',
                                                  'arctan', 'where', 'pi') }
USER_NS['erf'] = scsp.erf

B0_TYPES = ('axial', 'FDM', 'G21 dipole', 'Luo_S1', 'Luo_S2', 'confined', 'general', 'ferraro')   # as written to params.dat: the index here (append new types at the end)
B0_TOR_TYPES = ('Malkus', 'wall', 'user', 'general')   # NEW (2026-10-28): toroidal B0, as written to params.dat

if par.B0 in ('FDM', 'confined'):   # NEW (2026-10-30): 'confined', of degree B0_l
    B0_l   = par.B0_l
    symmB0 = int((-1)**B0_l)  # equatorial symmetry of B0
elif par.B0 == 'ferraro':    # NEW (2026-10-30): odd degrees 1, 3, ..., B0_ferraro_nmax (ψ = G(ΔΩ) is even in cos θ)
    B0_l   = 1
    symmB0 = -1
elif par.B0 == 'general':    # NEW (2026-10-30): degrees of one parity, the lowest sets the symmetry (background.check_magnetic)
    B0_l   = min(par.B0_profiles) if par.B0_profiles else 1
    symmB0 = int((-1)**B0_l)
elif par.B0 == 'Luo_S2':      # NEW (2026-10-28): degree 2
    B0_l   = 2
    symmB0 = 1
else:  # 'axial' (and any name that background.check() stops on)
    B0_l   = 1
    symmB0 = -1
symm_pol = symmB0 if par.B0 is not None else None                                       # NEW (2026-10-28)
B0_tor_l = (min(par.B0_tor_profiles) if par.B0_tor_profiles else 1) if par.B0_tor == 'general' else par.B0_tor_l   # NEW (2026-10-30)
symm_tor = int((-1)**(B0_tor_l + 1)) if par.B0_tor is not None else None             # of the toroidal B0
if par.B0 is None and symm_tor is not None:
    symmB0 = symm_tor   # (with both fields background.check_magnetic() requires symm_pol = symm_tor)



def jl_smx(l, x, d):
    '''
    Spherical Bessel function of the first kind and derivatives, small argument (x << 1) only, 0 <= d <= 3
    '''
    c1 =  (2**l)*scsp.factorial(l)/scsp.factorial(2*l+1)
    c2 = -(2**l)*scsp.factorial(l+1)/scsp.factorial(2*l+3)
    c3 =  (2**l)*scsp.factorial(l+2)/(2*scsp.factorial(2*l+5))
    if d == 0:
        out = c1*x**l + c2*x**(l+2) + c3*x**(l+4)
    elif d == 1:
        out = c1*l*x**(l-1) + c2*(l+2)*x**(l+1) + c3*(l+4)*x**(l+3)
    elif d == 2:
        if l >= 2:
            out = c1*l*(l-1)*x**(l-2) + c2*(l+2)*(l+1)*x**l + c3*(l+4)*(l+3)*x**(l+2)
        else:
            out = c2*(l+2)*(l+1)*x**l + c3*(l+4)*(l+3)*x**(l+2)
    elif d == 3:
        if l >= 3:
            out = c1*l*(l-1)*(l-2)*x**(l-3) + c2*(l+2)*(l+1)*l*x**(l-1) + c3*(l+4)*(l+3)*(l+2)*x**(l+1)
        else:
            out = c2*(l+2)*(l+1)*l*x**(l-1) + c3*(l+4)*(l+3)*(l+2)*x**(l+1)
    return out


def jl(l, x, d):
    '''Spherical Bessel function of the first kind (d = 0) or its derivative (d = 1)'''
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.zeros_like(x)
    k = x < 1e-3
    out[k]  = jl_smx(l, x[k], d)
    out[~k] = scsp.spherical_jn(l, x[~k], derivative=d)
    return out


def nl(l, x, d):
    '''Spherical Bessel function of the second kind (d = 0) or its derivative (d = 1)'''
    return scsp.spherical_yn(l, x, derivative=d)


def jl_k(l, x, k):
    '''
    NEW (2026-10-28): the k-th derivative (any k) of the spherical Bessel function j_l at x >= 0, to round-off: the Taylor series
    Σ_s (-1)^s x^(2s+l) / (2^s s! (2l+2s+1)!!) for x < 1, else the sum of x^-p j_(l+i)(x) that j_n' = (n/x) j_n - j_(n+1) gives
    '''
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.zeros_like(x)
    sm = x < 1
    for s in range(20):
        e = 2*s + l
        if e >= k:
            c = (-1)**s/(2.**s*scsp.factorial(s)*scsp.factorial2(2*l + 2*s + 1))*np.prod([e - j for j in range(k)])
            out[sm] += c*x[sm]**(e - k)
    terms = {(0, 0): 1.0}   # (i, p): coefficient of x^-p j_(l+i)
    for _ in range(k):
        new = {}
        for (i, p), c in terms.items():
            new[(i, p + 1)] = new.get((i, p + 1), 0) + c*(l + i - p)
            new[(i + 1, p)] = new.get((i + 1, p), 0) - c
        terms = new
    xb = x[~sm]
    out[~sm] = sum(c*xb**(-p)*scsp.spherical_jn(l + i, xb) for (i, p), c in terms.items())
    return out


def findbeta(beta0, l, ricb):
    '''radial wavenumber β of the poloidal free decay mode of degree l, the root nearest beta0 (Zhang & Fearn 1995, p. 196)'''
    f0 = lambda x: jl(l+1, x*ricb, 0) * nl(l-1, x, 0) - jl(l-1, x, 0) * nl(l+1, x*ricb, 0)
    if ricb == 0:   # NEW (2026-10-28): full sphere, h = j_l(βr) with j_{l-1}(β) = 0 (Gubbins & Roberts 1987, p. 49)
        f0 = lambda x: jl(l-1, x, 0)
    return so.root(f0, beta0).x[0]


beta = findbeta(par.beta, B0_l, par.ricb) if (par.magnetic and par.B0 == 'FDM') else None



P_ =np.polynomial.Polynomial
POLY = { 'G21 dipole': P_([0, 1/6, 0, -1/10]), 'Luo_S1': P_([0, 5, 0, -3]), 'Luo_S2': P_([0, 0, 157, 0, -296, 0, 143]) }   # NEW (2026-10-28)


def B0_beta_ic():
    '''NEW (2026-10-28): conductive_IC's B0_beta_ic: β of the potential field β r^-(l+1) added to h (B0_ic = 'potential', ricb > 0,
    the full-sphere fields of POLY), so that r h' - l h = 0 at the inner boundary'''
    if par.magnetic and par.B0 in POLY and par.ricb > 0 and par.B0_ic == 'potential':
        ri = par.ricb; l = B0_l
        return ri**(l+1)*(ri*POLY[par.B0].deriv(1)(ri) - l*POLY[par.B0](ri))/(2*l + 1)
    return 0.0

def _confined():
    '''
    NEW (2026-10-30): 'confined' as l and its linear factors [(a, b)] with h = r^l Π (a + b r)^p inside [r1, r2], and (r1, r2).
    The factors are kept apart and scaled to O(1) (h is evaluated in factored form by h()): expanded into monomials, a
    polynomial of degree 2p + l with p ~ 10 loses all its digits to cancellation near the edges.
    '''
    l, r1, r2 = B0_l, par.B0_conf_r1, par.B0_conf_r2
    if r1 == 0:   # (r2² - r²)^p = (r2 - r)^p (r2 + r)^p, scaled by r2^-2p
        return l, [(1., -1/r2), (1., 1/r2)], r1, r2
    w = (r2 - r1)/2   # (r - r1)^p (r2 - r)^p scaled by w^-2p
    return l, [(-r1/w, 1/w), (r2/w, -1/w)], r1, r2

def _confined_h(r, k):
    '''NEW (2026-10-30): the k-th derivative of r^l Π (a + b r)^p (no cut), by the Leibniz rule on the factored form'''
    from math import comb
    l, fac, _, _ = _confined()
    p = par.B0_conf_p
    def dpow(base, n, j, slope):   # d^j/dr^j (base)^n for base linear in r with the given slope
        if j > n:
            return np.zeros_like(r)
        return np.prod([n - i for i in range(j)])*slope**j*base**(n - j)
    parts = [ [dpow(r, l, j, 1.) for j in range(k + 1)] ] + [ [dpow(a + b*r, p, j, b) for j in range(k + 1)] for a, b in fac ]
    out = parts[0]
    for q in parts[1:]:   # Leibniz: (fg)^(j) = Σ C(j, i) f^(i) g^(j-i)
        out = [ sum(comb(j, i)*out[i]*q[j - i] for i in range(j + 1)) for j in range(k + 1) ]
    return out[k]

def h(r, k):
    '''
    The k-th r-derivative (k = 0..3) of the poloidal scalar h(r) of B0, without cnorm, at r > 0 (ricb > 0).
    The FDM expressions are those of conductive_IC's h0..h3 (ricb > 0 branch), which also hold for B0_l > 1.
    NEW (2026-10-28): with ricb = 0 (r on [-1, 1]) the FDM is h = j_l(βr), of parity (-1)^l (jl_k).
    '''
    r = np.asarray(r, dtype=float)
    if par.B0 == 'FDM' and par.ricb == 0:   # FIX (2026-10-28): analytic derivatives (a Chebyshev interpolant lost 3e-5 in the third)
        return beta**k*jl_k(B0_l, beta*np.abs(r), k)*np.where(r < 0, (-1.)**(B0_l + k), 1.)
    if par.B0 == 'axial':
        return [ r/2, 0.5*np.ones_like(r), np.zeros_like(r), np.zeros_like(r) ][k]
    if par.B0 == 'confined':   # NEW (2026-10-30): the polynomial inside [r1, r2] (|r| in a full sphere, r^l carries the parity)
        _, _, r1, r2 = _confined()
        inside = (np.abs(r) >= r1) & (np.abs(r) <= r2)
        return np.where(inside, _confined_h(r, k), 0.)
    if par.B0 in POLY:   # NEW (2026-10-28): the polynomial fields, plus the potential part of B0_ic (beta_ic)
        out = POLY[par.B0].deriv(k)(r) if k > 0 else POLY[par.B0](r)
        if beta_ic != 0:
            out = out + beta_ic*np.prod([-(B0_l + 1) - j for j in range(k)])*r**(-(B0_l + 1) - k)   # k-th derivative of β r^-(l+1)
        return out
    l, b = B0_l, beta
    x = b*r
    if k == 0:
        return jl(l,x,0)*nl(l-1,b,0) - jl(l-1,b,0)*nl(l,x,0)
    if k == 1:
        return b*(jl(l,x,1)*nl(l-1,b,0) - jl(l-1,b,0)*nl(l,x,1))
    if k == 2:
        return ( (x**2*jl(l-1,x,1) + (1+l)*(jl(l,x,0) - x*jl(l,x,1)))*nl(l-1,b,0)
                 - jl(l-1,b,0)*(x**2*nl(l-1,x,1) + (1+l)*(nl(l,x,0) - b*r*nl(l,x,1))) )/r**2
    if l >= 2:
        return ( (x**3*jl(l-2,x,1) + l*x*jl(l-1,x,0) - x**2*jl(l-1,x,1) - 2*l*x**2*jl(l-1,x,1) - 3*jl(l,x,0) - 4*l*jl(l,x,0)
                  - l**2*jl(l,x,0) + 3*x*jl(l,x,1) + 4*l*x*jl(l,x,1) + l**2*x*jl(l,x,1))*nl(l-1,b,0)
                 + jl(l-1,b,0)*(-(x**3*nl(l-2,x,1)) - l*x*nl(l-1,x,0) + x**2*nl(l-1,x,1) + 2*l*x**2*nl(l-1,x,1) + 3*nl(l,x,0)
                  + 4*l*nl(l,x,0) + l**2*nl(l,x,0) - 3*x*nl(l,x,1) - 4*l*x*nl(l,x,1) - l**2*x*nl(l,x,1)) )/r**3
    return ( -2*x**2*jl(0,x,1)*nl(0,b,0) - 8*jl(1,x,0)*nl(0,b,0) + 8*x*jl(1,x,1)*nl(0,b,0) - x**3*jl(1,x,1)*nl(0,b,0)
             + 2*x**2*jl(0,b,0)*nl(0,x,1) + 8*jl(0,b,0)*nl(1,x,0) - 8*x*jl(0,b,0)*nl(1,x,1) + x**3*jl(0,b,0)*nl(1,x,1) )/r**3


_gen = {}
def _gen_interp(kind, n):
    '''
    NEW (2026-10-30): the Chebyshev interpolant of the 'general' radial expression of degree n, kind 'h' (B0_profiles) or 'k'
    (B0_tor_profiles): degree max(64, 2N) on [ricb, 1], or on [-1, 1] at |r| with the parity (-1)^n in a full sphere (the
    expression should go as r^n at r = 0); coefficients below 1e-15 of the largest are dropped, so that the derivatives (up
    to the third) do not amplify round-off.
    '''
    key = (kind, n)
    if key not in _gen:
        expr = (par.B0_profiles if kind == 'h' else par.B0_tor_profiles)[n]
        code = compile(expr, '<B0 %s_%d>' % (kind, n), 'eval')
        f = lambda r: np.broadcast_to(np.asarray(eval(code, {'__builtins__': {}}, dict(USER_NS, r=r, p=par.B0_param)), dtype=float), np.shape(r))
        deg = max(64, 2*par.N)
        if par.ricb == 0:
            c = np.polynomial.chebyshev.Chebyshev.interpolate(lambda r: f(np.abs(r))*np.sign(r)**n, deg, domain=[-1., 1.])
        else:
            c = np.polynomial.chebyshev.Chebyshev.interpolate(f, deg, domain=[par.ricb, 1.])
        c.coef[np.abs(c.coef) < 1e-13*np.abs(c.coef).max()] = 0.   # the round-off of the interpolation, which the derivatives amplify
        _gen[key] = c
    return _gen[key]

def gen_profile(kind, n, r, d):
    '''NEW (2026-10-30): the d-th r-derivative of the 'general' h_n (kind 'h') or k_n (kind 'k'), without cnorm'''
    c = _gen_interp(kind, n)
    r = np.asarray(r, dtype=float)
    return c.deriv(d)(r) if d > 0 else c(r)

# ------------------------------------------------- NEW (2026-10-30): 'ferraro', a poloidal field on the isorotation surfaces
# A poloidal axisymmetric field B0 = ∇ψ × ∇φ with ψ = G(ΔΩ) has B0⋅∇ΔΩ = 0 wherever ∇ΔΩ ≠ 0 (its field lines lie on the
# surfaces of constant rotation: Ferraro's law), so U0 and B0 are a steady background together. For kore's form
# B0 = Σ_n ∇×∇×(h_n Y_n⁰ 𝐫), the vector potential is A_φ = h_n sinθ P_n'(x), so ψ = r sinθ A_φ = r Σ_n h_n (1 - x²) P_n'(x).
# The functions (1 - x²) P_n' are orthogonal with the weight 1/(1 - x²) (∫ (1 - x²) P_n' P_m' dx = δ_nm 2n(n+1)/(2n+1)), so
#     h_n(r) = (2n+1)/(2n(n+1)) (1/r) ∫ ψ(r, x) P_n'(x) dx,
# by Gauss-Legendre quadrature in x. ΔΩ has only even P_n, so ψ is even in x and only odd n appear (an antisymmetric, dipolar
# field); the series is cut at B0_ferraro_nmax. The user chooses G (B0_ferraro_G, a numpy expression of w = ΔΩ and p =
# B0_param). Assumptions (background.ferraro_report prints how well they hold): ψ = 0 on the axis (it must be constant
# there for a regular field, and the functions (1 - x²) P_n' vanish there), and at the centre in a full sphere; at the outer
# wall ψ = 0 keeps the field inside (no radial field crosses it), otherwise the field meets the insulating mantle with a
# surface current (allowed: the perturbation's boundary conditions do not involve B0).
_fer = {}
def _ferraro_psi(r, x):
    '''ψ = G(ΔΩ(r, x)) on the grid (r, x), ΔΩ from diff_rot.raw (amplitude included)'''
    import diff_rot as dr
    P = np.polynomial.legendre.Legendre
    w = sum(np.outer(dr.raw(n, np.abs(r)), P.basis(n)(x)) for n in dr.degrees())
    code = compile(par.B0_ferraro_G, '<B0_ferraro_G>', 'eval')
    return np.broadcast_to(np.asarray(eval(code, {'__builtins__': {}}, dict(USER_NS, w=w, p=par.B0_param)), dtype=float), w.shape)

def _ferraro_raw(n, r):
    '''h_n(r) of ψ by the projection above, at the radii r (r > 0)'''
    xg, wg = np.polynomial.legendre.leggauss(4*par.B0_ferraro_nmax + 64)
    dPn = np.polynomial.legendre.Legendre.basis(n).deriv()(xg)
    return (2*n + 1)/(2*n*(n + 1.))*(_ferraro_psi(r, xg) @ (wg*dPn))/r

def ferraro_profile(n, r, d):
    '''the d-th r-derivative of h_n of 'ferraro' (without cnorm): a Chebyshev interpolant of degree max(64, 2N), on [-1, 1] at |r|
    with the parity (-1)^n in a full sphere'''
    if n not in _fer:
        deg = max(64, 2*par.N)
        if par.ricb == 0:
            g = lambda rr: _ferraro_raw(n, np.maximum(np.abs(rr), 1e-12))*np.sign(rr)**n
            c = np.polynomial.chebyshev.Chebyshev.interpolate(g, deg, domain=[-1., 1.])
        else:
            c = np.polynomial.chebyshev.Chebyshev.interpolate(lambda rr: _ferraro_raw(n, rr), deg, domain=[par.ricb, 1.])
        c.coef[np.abs(c.coef) < 1e-13*np.abs(c.coef).max()] = 0.
        _fer[n] = c
    c = _fer[n]
    r = np.asarray(r, dtype=float)
    return c.deriv(d)(r) if d > 0 else c(r)

def ferraro_degrees():
    return list(range(1, par.B0_ferraro_nmax + 1, 2))

_cnf = []
def ferraro_cnorm():
    '''cnorm of 'ferraro' (a number, or from ∫ B0² dV summed over the degrees), computed on first use (it needs diff_rot)'''
    if not _cnf:
        if par.cnorm not in ['mag_energy', 'Schmitt2012']:
            _cnf.append(float(par.cnorm))
        else:
            xk, wk = np.polynomial.legendre.leggauss(240)
            rk = 0.5*(1 - par.ricb)*(xk + 1) + max(par.ricb, 0.)
            integ = 0.
            for n in ferraro_degrees():
                L = n*(n + 1.); y0, y1 = ferraro_profile(n, rk, 0), ferraro_profile(n, rk, 1)
                integ += 0.5*(1 - par.ricb)*np.sum(wk*4*np.pi*L/(2*n + 1)*((L + 1)*y0**2 + 2*rk*y0*y1 + rk**2*y1**2))
            _cnf.append((1 if par.cnorm == 'mag_energy' else 2)/np.sqrt(integ))
    return _cnf[0]

def B0_norm():
    '''
    cnorm as a number (par.cnorm: a number, 'rms_cmb', 'mag_energy' or 'Schmitt2012'), as conductive_IC's B0_norm:
    ∫ B0² dV over the shell from Gauss-Legendre quadrature with 240 nodes.
    '''
    if par.B0 == 'ferraro':   # NEW (2026-10-30): computed on first use (ferraro_cnorm)
        return 1.0
    if par.B0 == 'general':   # NEW (2026-10-30): ∫ B0² dV is the sum over the degrees (they are orthogonal)
        if par.cnorm not in ['mag_energy', 'Schmitt2012']:
            return float(par.cnorm)
        xk, wk = np.polynomial.legendre.leggauss(240)
        rk = 0.5*(1 - par.ricb)*(xk + 1) + par.ricb
        integ = 0.
        for n in par.B0_profiles:
            L = n*(n + 1.); y0, y1 = gen_profile('h', n, rk, 0), gen_profile('h', n, rk, 1)
            integ += 0.5*(1 - par.ricb)*np.sum(wk*4*np.pi*L/(2*n + 1)*((L + 1)*y0**2 + 2*rk*y0*y1 + rk**2*y1**2))
        return (1 if par.cnorm == 'mag_energy' else 2)/np.sqrt(integ)
    l = B0_l
    L = l*(l+1)
    if par.cnorm == 'rms_cmb':  # rms of the radial B0 over the CMB is 1
        return np.sqrt(2*l+1) / ( L * h(np.array([1.0]), 0)[0] )
    if par.cnorm in ['mag_energy', 'Schmitt2012']:  # ∫ B0² dV = 1 or 4
        ricb = par.ricb
        N = 240
        xk, wk = np.polynomial.legendre.leggauss(N)
        sqx = wk*N/np.pi
        if par.B0 == 'confined':   # NEW (2026-10-30): integrate over the region only, where h is a polynomial (exact)
            ricb, rtop = max(par.B0_conf_r1, ricb), par.B0_conf_r2
            rk = 0.5*(rtop-ricb)*( xk + 1 ) + ricb
            y0, y1 = h(rk, 0), h(rk, 1)
            f0 = 4*np.pi*L/(2*l+1)
            integ = (np.pi/N) * ( (rtop-ricb)/2 ) * np.sum( sqx*f0*( (L+1)*y0**2 + 2*rk*y0*y1 + rk**2*y1**2 ) )
            return (1 if par.cnorm == 'mag_energy' else 2)/np.sqrt(integ)
        rk = 0.5*(1-ricb)*( xk + 1 ) + ricb
        y0 = h(rk, 0)
        y1 = h(rk, 1)
        f0 = 4*np.pi*L/(2*l+1)
        integ = (np.pi/N) * ( (1-ricb)/2 ) * np.sum( sqx*f0*( (L+1)*y0**2 + 2*rk*y0*y1 + rk**2*y1**2 ) )
        return (1 if par.cnorm == 'mag_energy' else 2)/np.sqrt(integ)
    return float(par.cnorm)


beta_ic = B0_beta_ic()   # NEW (2026-10-28)
# (only for the settings that background.check_magnetic() accepts, which imports this module before it checks them)
cnorm = B0_norm() if (par.magnetic and par.B0 in B0_TYPES) else 0.0   # FIX (2026-10-28): also with ricb = 0


def bpoX(r, Dorder):
    '''profile code 'bpo' (radial_profiles.proffdir): cnorm h^(Dorder)(r), Dorder = 0..3 (zero without a poloidal B0)'''
    if par.B0 is None:   # NEW (2026-10-28): a toroidal B0 alone
        return np.zeros_like(np.asarray(r, dtype=float))
    return cnorm*h(r, Dorder)



# ---------------------------------------------------------------------------- NEW (2026-10-28): toroidal background field

_kuser = None

def k_user():
    '''the Chebyshev interpolant (degree max(64, 2N) on [ricb, 1]) of the 'user' expression B0_tor_profile'''
    global _kuser
    if _kuser is None:
        code = compile(par.B0_tor_profile, '<B0_tor_profile>', 'eval')
        f = lambda r: np.broadcast_to(np.asarray(eval(code, {'__builtins__': {}}, dict(USER_NS, r=r, p=par.B0_tor_param)), dtype=float), np.shape(r))
        if par.ricb == 0:   # NEW (2026-10-28): full sphere, at |r| with the parity (-1)^n (the expression should go as r^n at r = 0)
            g = lambda r: f(np.abs(r))*np.sign(r)**par.B0_tor_l
            _kuser = np.polynomial.chebyshev.Chebyshev.interpolate(g, max(64, 2*par.N), domain=[-1., 1.])
        else:
            _kuser = np.polynomial.chebyshev.Chebyshev.interpolate(f, max(64, 2*par.N), domain=[par.ricb, 1.])
    return _kuser


def k(r, d):
    '''the d-th r-derivative (d = 0..3) of the radial function k(r) of the toroidal B0 (without cnorm_tor)'''
    r = np.asarray(r, dtype=float)
    if par.B0_tor == 'user':
        return k_user().deriv(d)(r) if d > 0 else k_user()(r)
    if par.B0_tor == 'Malkus':
        return [ r, np.ones_like(r), np.zeros_like(r), np.zeros_like(r) ][d]
    if par.ricb == 0:   # NEW (2026-10-28): 'wall' in a full sphere, regular at r = 0 with the parity (-1)^n: r^n (1 - r²)
        n = par.B0_tor_l
        p_ = np.polynomial.Polynomial([0]*n + [1, 0, -1])
        return p_.deriv(d)(r) if d > 0 else p_(r)
    # 'wall': r (r - ri)(1 - r)/(1 - ri)² = (-r³ + (1 + ri) r² - ri r)/(1 - ri)²
    ri = par.ricb; c = 1/(1 - ri)**2
    return c*[ -r**3 + (1 + ri)*r**2 - ri*r, -3*r**2 + 2*(1 + ri)*r - ri, -6*r + 2*(1 + ri), -6*np.ones_like(r) ][d]


def B0_tor_norm():
    '''cnorm_tor as a number: ∫ B0_tor² dV over the shell = n_l L ∫ k² r² dr, n_l = 4π/(2l+1), L = l(l+1)'''
    if par.B0_tor == 'general' and par.cnorm_tor in ('mag_energy', 'Schmitt2012'):   # NEW (2026-10-30): the sum over the degrees
        xk, wk = np.polynomial.legendre.leggauss(240)
        rk = 0.5*(1 - par.ricb)*(xk + 1) + par.ricb
        integ = sum(4*np.pi/(2*n + 1)*n*(n + 1)*0.5*(1 - par.ricb)*np.sum(wk*gen_profile('k', n, rk, 0)**2*rk**2) for n in par.B0_tor_profiles)
        return (1 if par.cnorm_tor == 'mag_energy' else 2)/np.sqrt(integ)
    if par.cnorm_tor in ('mag_energy', 'Schmitt2012'):
        n = par.B0_tor_l
        xk, wk = np.polynomial.legendre.leggauss(240)
        rk = 0.5*(1 - par.ricb)*(xk + 1) + par.ricb
        integ = 4*np.pi/(2*n + 1)*n*(n + 1)*0.5*(1 - par.ricb)*np.sum(wk*k(rk, 0)**2*rk**2)
        return (1 if par.cnorm_tor == 'mag_energy' else 2)/np.sqrt(integ)
    return float(par.cnorm_tor)


cnorm_tor = B0_tor_norm() if (par.magnetic and par.B0_tor in B0_TOR_TYPES) else 0.0   # FIX (2026-10-28): also with ricb = 0


def btoX(r, Dorder):
    '''profile code 'bto' (radial_profiles.proffdir): cnorm_tor k^(Dorder)(r), Dorder = 0..3'''
    return cnorm_tor*k(r, Dorder)



# ------------------------------------------------------------- NEW (2026-10-30): the background degree by degree
def pol_profiles():
    '''{n: f(r, Dorder)}: the poloidal background degree by degree, cnorm h_n^(Dorder) (mag_algebra.py); {} without one'''
    if not par.magnetic or par.B0 is None:
        return {}
    if par.B0 == 'general':
        return {n: (lambda r, d, n=n: cnorm*gen_profile('h', n, r, d)) for n in sorted(par.B0_profiles)}
    if par.B0 == 'ferraro':
        return {n: (lambda r, d, n=n: ferraro_cnorm()*ferraro_profile(n, r, d)) for n in ferraro_degrees()}
    return {B0_l: bpoX}

def tor_profiles():
    '''{n: f(r, Dorder)}: the toroidal background degree by degree, cnorm_tor k_n^(Dorder) (mag_algebra.py)'''
    if not par.magnetic or par.B0_tor is None:
        return {}
    if par.B0_tor == 'general':
        return {n: (lambda r, d, n=n: cnorm_tor*gen_profile('k', n, r, d)) for n in sorted(par.B0_tor_profiles)}
    return {par.B0_tor_l: btoX}
