'''
NEW (2026-11-04): latitude-dependent topographic drag at a stress-free CMB (topo_drag = 'rekier'), from the internal-wave
power of Rekier, Triana, Barik, Abdulah & Kang (arXiv:2507.01671), eqs. (20), (21), (B1), (B2), with the factor 4 pi^2
that the printed (B1) of v4 lacks (it is lost between (A19) and (10): with the (A2) convention, eq. (12) makes
|h^|^2/A = 4 pi^2 S). With it a sinusoidal ridge gives Bell's (1975) rho U^2 h^2 k sqrt(N^2 - w^2)/4.

For a tangential flow of amplitudes U_x (zonal) and U_y (meridional) over topography of spectrum S(k), eq. (B1) with
that factor gives the mean power per unit area
    P = rho I_k/(2 |w0|) int_0^2pi (U_x^2 cos^2 a + U_y^2 sin^2 a) g(a) da,
    g = sqrt(max(0, (w0^2 - N^2)(f^2 - w0^2) + w0^2 ft^2 sin^2 a)),  f = 2 Omega cos(theta), ft = 2 Omega sin(theta),
    I_k = int S k^2 dk = h_rms^2 k0 Gamma(H - 1/2)/(4 sqrt(pi) Gamma(H)),  k0 = 2 pi/lambda0          (eq. B2)
(theta the colatitude). There is no zonal-meridional cross term, so a linear drag tau = -K u with mean power K <u^2>
needs the diagonal tensor K = diag(kappa_tt, kappa_pp) acting on (u_theta, u_phi), in units of rho Omega R:
    kappa_pp = (I_k/R)/|w0/Omega| int cos^2 a g/Omega^2 da,  kappa_tt = the same with sin^2 a.
rho and R cancel; the inputs are topo_hrms = h_rms/R, topo_lambda0 = lambda0/R and topo_H. N is the Brunt-Vaisala
frequency at the CMB (N_cmb), w0 the frequency of the flow relative to the mantle (omega0).

The boundary condition Ek sigma_rt = -K (u_t - v_m), projected on grad(Y_l) and grad(Y_l) x r_hat (u_t = s grad Y +
t grad Y x r_hat per degree, s = P' + P/r, t = T), couples the CMB rows of P_l and T_l to s and t of all l' through
    A_ss = oint conj(a_l).K.a_l' / oint |a_l|^2,  A_st = ... K.b_l',  A_ts = conj(b_l).K.a_l',  A_tt = conj(b_l).K.b_l',
a = grad Y = (dY/dtheta, i m Y/sin), b = grad Y x r_hat = (i m Y/sin, -dY/dtheta), oint |a_l|^2 = L 4 pi/(2l+1).
A latitude-dependent K couples l to l +- 2, ... within P and within T, and P_l to T_(l+-1), ... for m != 0 (also when
K is isotropic but depends on theta). A constant isotropic K gives A_ss = A_tt = kappa, A_st = A_ts = 0.
'''

import numpy as np
from scipy.special import gamma
from parameters import par
import utils as ut

_cache = {}


def N_cmb():
    '''
    N/Omega at the CMB from the background stratification (buoyancy OmgTau^2 BV2 r times the gradient shape, gravity
    proportional to r): thermal BV2 x (1 'internal', ricb/(1-ricb) 'differential', twozone(1) 'two zone', BVprof(1)
    'user defined'), plus compositional BV2_comp x (1 'internal', ricb/(1-ricb) 'differential'). topo_N overrides it.
    An unstable stratification (N^2 < 0) is taken as N = 0. 0 without buoyancy.
    '''
    if par.topo_N is not None:
        return float(par.topo_N)
    n2 = 0.0
    if par.thermal:
        if par.heating == 'internal':
            s = 1.0
        elif par.heating == 'differential':
            s = par.ricb/(1 - par.ricb)
        elif par.heating == 'two zone':
            s = ut.twozone(np.array([1.0]), par.args)[0]
        else:  # 'user defined'
            s = ut.BVprof(np.array([1.0]), par.args)[0]
        n2 += par.BV2*s
    if par.compositional:
        s = par.ricb/(1 - par.ricb) if par.comp_background == 'differential' else 1.0
        n2 += par.BV2_comp*s
    return float(np.sqrt(max(n2, 0.0)))


def omega0():
    '''
    |w0|/Omega: topo_omega if set, else the forcing frequency (forced problems) or the solver target's frequency
    |Im tau| (eigenvalue problems; with track_target = 1 the target follows the mode, tools/topo_iterate.sh), divided by
    OmgTau (frequencies are in units of 1/tau, Omega tau = OmgTau)
    '''
    if par.topo_omega is not None:
        return abs(float(par.topo_omega))
    w = par.forcing_frequency if par.forcing != 0 else complex(par.tau).imag
    return abs(w)/par.OmgTau


def Ik_R():
    '''I_k/R of eq. (B2) from topo_hrms = h_rms/R, topo_lambda0 = lambda0/R and topo_H'''
    k0R = 2*np.pi/par.topo_lambda0
    return par.topo_hrms**2*k0R*gamma(par.topo_H - 0.5)/(4*np.sqrt(np.pi)*gamma(par.topo_H))


def kappa(x, na=2048):
    '''
    [kappa_tt, kappa_pp] (meridional, zonal; units rho Omega R) at x = cos(colatitude), eq. (B1) with the factor 4 pi^2
    '''
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if par.topo_K is not None:  # test hook: a function of x giving (kappa_tt, kappa_pp)
        kt, kp = par.topo_K(x)
        return np.broadcast_to(kt, x.shape).astype(float), np.broadcast_to(kp, x.shape).astype(float)
    w0, n = omega0(), N_cmb()
    a = (np.arange(na) + 0.5)*2*np.pi/na
    f2, ft2 = 4*x**2, 4*(1 - x**2)
    g = np.sqrt(np.maximum(0.0, (w0**2 - n**2)*(f2[:, None] - w0**2) + w0**2*ft2[:, None]*np.sin(a)**2))
    c = Ik_R()/w0*(2*np.pi/na)  # FIX (2026-11-04): no 1/(4 pi^2), the factor missing from the printed eq. (B1) is restored
    return c*np.sum(np.sin(a)**2*g, axis=1), c*np.sum(np.cos(a)**2*g, axis=1)


def couplings(nx=None):
    '''
    The coupling matrices A_ss, A_st, A_ts, A_tt[l, l'] (l, l' = 0..lmax+1) at the CMB for par.m, cached. Gauss-Legendre
    quadrature in x = cos(theta) with nx points (K has kinks where the root of eq. B1 vanishes, so many points).
    '''
    from utils4pp import ylm_grid
    key = (par.m, par.lmax, nx)
    if key in _cache:
        return _cache[key]
    lm = par.lmax + 1
    if nx is None:
        nx = max(4*lm, 2000)
    x, w = np.polynomial.legendre.leggauss(nx)
    Y, dY, Z = ylm_grid(par.m, lm, x)        # dY = dY/dtheta (real), Z = i m Y/sin(theta)
    kt, kp = kappa(x)
    ls = np.arange(lm + 1)
    nrm = 4*np.pi/(2*ls + 1)*np.maximum(ls*(ls + 1), 1)
    W = 2*np.pi*w
    # a = (dY, Z), b = (Z, -dY): conj(a_l).K.a_l' = kt dY_l dY_l' + kp conj(Z_l) Z_l', etc.
    Ass = ((W*kt*dY) @ dY.T + (W*kp*np.conj(Z)) @ Z.T)/nrm[:, None]
    Ast = ((W*kt*dY) @ Z.T - (W*kp*np.conj(Z)) @ dY.T)/nrm[:, None]
    Ats = ((W*kt*np.conj(Z)) @ dY.T - (W*kp*dY) @ Z.T)/nrm[:, None]
    Att = ((W*kt*np.conj(Z)) @ Z.T + (W*kp*dY) @ dY.T)/nrm[:, None]
    # prune the round-off of couplings that vanish by symmetry
    tol = 1e-13*max(abs(Ass).max(), abs(Att).max(), 1e-300)
    for M in (Ass, Ast, Ats, Att):
        M[abs(M) < tol] = 0
    _cache[key] = (Ass, Ast, Ats, Att)
    return _cache[key]
