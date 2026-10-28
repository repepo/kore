'''
NEW (2026-10-28): the imposed background magnetic field B0 (magnetic = 1), ported from the conductive_IC branch
(bin/utils.py there: jl, nl, findbeta, h0..h3, B0_norm, symmB0) for the types of B0_TYPES.

B0 = cnorm ∇×∇×(h(r) Y_l0 r), axisymmetric and poloidal, with Y_l0 the Schmidt harmonic P_l(cos θ) of kore. So
B0_r = cnorm l(l+1) h Y_l0/r and ∇×B0 = cnorm (D_l h) ∂θY_l0 φ̂, D_l h = h'' + 2h'/r - l(l+1)h/r².
    'axial'  h = r/2 (l = 1): the uniform field B0 = cnorm ẑ, no current.
    'FDM'    the poloidal free decay mode of degree l = B0_l of the shell (Zhang & Fearn, GAFD 1995, eq. 2.7):
             h = j_l(βr) y_{l-1}(β) - j_{l-1}(β) y_l(βr), β the root of findbeta near par.beta. It matches a potential
             field at both walls, so it suits insulating walls.
NEW (2026-10-28): a toroidal field B0_tor = cnorm_tor ∇×(k(r) Y_n0 𝐫), B_φ = -cnorm_tor k ∂θY_n0, n = B0_tor_l, alone (B0 = None)
or added to the poloidal one (same equatorial symmetry only):
    'Malkus' k = r (n = 1): B0 = cnorm_tor s φ̂ (Malkus 1967), J0 = ∇×B0 = 2 cnorm_tor ẑ.
    'wall'   k = r (r - ricb)(1 - r)/(1 - ricb)² (n = 1): zero at both walls.
    'user'   k = B0_tor_profile, a numpy expression of r and p = B0_tor_param (its derivatives from a Chebyshev interpolant).
Equatorial symmetry: a poloidal field of degree n has (-1)^n, a toroidal one (-1)^(n+1) (B_φ ∝ sinθ is symmetric for n = 1).
Only parameters, numpy and scipy are imported: utils.py and radial_profiles.py import this module.
'''
import numpy as np
import scipy.special as scsp
import scipy.optimize as so
from parameters import par

B0_TYPES = ('axial', 'FDM')   # as written to params.dat: the index here (append new types at the end)
B0_TOR_TYPES = ('Malkus', 'wall', 'user')   # NEW (2026-10-28): toroidal B0, as written to params.dat

if par.B0 == 'FDM':
    B0_l   = par.B0_l
    symmB0 = int((-1)**B0_l)  # equatorial symmetry of B0
else:  # 'axial' (and any name that background.check() stops on)
    B0_l   = 1
    symmB0 = -1
symm_pol = symmB0 if par.B0 is not None else None                                       # NEW (2026-10-28)
symm_tor = int((-1)**(par.B0_tor_l + 1)) if par.B0_tor is not None else None             # of the toroidal B0
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


def findbeta(beta0, l, ricb):
    '''radial wavenumber β of the poloidal free decay mode of degree l, the root nearest beta0 (Zhang & Fearn 1995, p. 196)'''
    f0 = lambda x: jl(l+1, x*ricb, 0) * nl(l-1, x, 0) - jl(l-1, x, 0) * nl(l+1, x*ricb, 0)
    return so.root(f0, beta0).x[0]


beta = findbeta(par.beta, B0_l, par.ricb) if (par.magnetic and par.B0 == 'FDM' and par.ricb > 0) else None



def h(r, k):
    '''
    The k-th r-derivative (k = 0..3) of the poloidal scalar h(r) of B0, without cnorm, at r > 0 (ricb > 0).
    The FDM expressions are those of conductive_IC's h0..h3 (ricb > 0 branch), which also hold for B0_l > 1.
    '''
    r = np.asarray(r, dtype=float)
    if par.B0 == 'axial':
        return [ r/2, 0.5*np.ones_like(r), np.zeros_like(r), np.zeros_like(r) ][k]
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


def B0_norm():
    '''
    cnorm as a number (par.cnorm: a number, 'rms_cmb', 'mag_energy' or 'Schmitt2012'), as conductive_IC's B0_norm:
    ∫ B0² dV over the shell from Gauss-Legendre quadrature with 240 nodes.
    '''
    l = B0_l
    L = l*(l+1)
    if par.cnorm == 'rms_cmb':  # rms of the radial B0 over the CMB is 1
        return np.sqrt(2*l+1) / ( L * h(np.array([1.0]), 0)[0] )
    if par.cnorm in ['mag_energy', 'Schmitt2012']:  # ∫ B0² dV = 1 or 4
        ricb = par.ricb
        N = 240
        xk, wk = np.polynomial.legendre.leggauss(N)
        sqx = wk*N/np.pi
        rk = 0.5*(1-ricb)*( xk + 1 ) + ricb
        y0 = h(rk, 0)
        y1 = h(rk, 1)
        f0 = 4*np.pi*L/(2*l+1)
        integ = (np.pi/N) * ( (1-ricb)/2 ) * np.sum( sqx*f0*( (L+1)*y0**2 + 2*rk*y0*y1 + rk**2*y1**2 ) )
        return (1 if par.cnorm == 'mag_energy' else 2)/np.sqrt(integ)
    return float(par.cnorm)


# (only for the settings that background.check_magnetic() accepts, which imports this module before it checks them)
cnorm = B0_norm() if (par.magnetic and par.B0 in B0_TYPES and par.ricb > 0) else 0.0


def bpoX(r, Dorder):
    '''profile code 'bpo' (radial_profiles.proffdir): cnorm h^(Dorder)(r), Dorder = 0..3 (zero without a poloidal B0)'''
    if par.B0 is None:   # NEW (2026-10-28): a toroidal B0 alone
        return np.zeros_like(np.asarray(r, dtype=float))
    return cnorm*h(r, Dorder)



# ---------------------------------------------------------------------------- NEW (2026-10-28): toroidal background field

USER_NS = { name: getattr(np, name) for name in ('sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'tanh', 'sinh', 'cosh',
                                                  'arctan', 'where', 'pi') }
USER_NS['erf'] = scsp.erf
_kuser = None

def k_user():
    '''the Chebyshev interpolant (degree max(64, 2N) on [ricb, 1]) of the 'user' expression B0_tor_profile'''
    global _kuser
    if _kuser is None:
        code = compile(par.B0_tor_profile, '<B0_tor_profile>', 'eval')
        f = lambda r: np.broadcast_to(np.asarray(eval(code, {'__builtins__': {}}, dict(USER_NS, r=r, p=par.B0_tor_param)), dtype=float), np.shape(r))
        _kuser = np.polynomial.chebyshev.Chebyshev.interpolate(f, max(64, 2*par.N), domain=[par.ricb, 1.])
    return _kuser


def k(r, d):
    '''the d-th r-derivative (d = 0..3) of the radial function k(r) of the toroidal B0 (without cnorm_tor)'''
    r = np.asarray(r, dtype=float)
    if par.B0_tor == 'user':
        return k_user().deriv(d)(r) if d > 0 else k_user()(r)
    if par.B0_tor == 'Malkus':
        return [ r, np.ones_like(r), np.zeros_like(r), np.zeros_like(r) ][d]
    # 'wall': r (r - ri)(1 - r)/(1 - ri)² = (-r³ + (1 + ri) r² - ri r)/(1 - ri)²
    ri = par.ricb; c = 1/(1 - ri)**2
    return c*[ -r**3 + (1 + ri)*r**2 - ri*r, -3*r**2 + 2*(1 + ri)*r - ri, -6*r + 2*(1 + ri), -6*np.ones_like(r) ][d]


def B0_tor_norm():
    '''cnorm_tor as a number: ∫ B0_tor² dV over the shell = n_l L ∫ k² r² dr, n_l = 4π/(2l+1), L = l(l+1)'''
    if par.cnorm_tor in ('mag_energy', 'Schmitt2012'):
        n = par.B0_tor_l
        xk, wk = np.polynomial.legendre.leggauss(240)
        rk = 0.5*(1 - par.ricb)*(xk + 1) + par.ricb
        integ = 4*np.pi/(2*n + 1)*n*(n + 1)*0.5*(1 - par.ricb)*np.sum(wk*k(rk, 0)**2*rk**2)
        return (1 if par.cnorm_tor == 'mag_energy' else 2)/np.sqrt(integ)
    return float(par.cnorm_tor)


cnorm_tor = B0_tor_norm() if (par.magnetic and par.B0_tor in B0_TOR_TYPES and par.ricb > 0) else 0.0


def btoX(r, Dorder):
    '''profile code 'bto' (radial_profiles.proffdir): cnorm_tor k^(Dorder)(r), Dorder = 0..3'''
    return cnorm_tor*k(r, Dorder)
