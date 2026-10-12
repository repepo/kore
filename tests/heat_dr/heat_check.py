#!/usr/bin/env python3
'''
heat_check.py: check the differential-rotation blocks of the heat equation (operators.thermal_diffrot) against a
direct (r, theta) quadrature of the same terms built from the 3-D fields, for random P, T, s.

Matrix side: H = -<s, blocks(P, T, s)> with the pairing of ebal.py (the blocks carry the sign of the A right-hand side).
Grid side:   G = int p0 conj(s) [ i m dOmega s + u_r dS_tw/dr + (u_theta/r) dS_tw/dtheta ] dV, with
             u_r = L P Y/r, u_theta = (1/r)[(rP)' + r h' P] dY/dtheta + i m T Y/sin(theta), dOmega = f0 + f2 P2,
             S_tw = sigma2 P2 + sigma4 P4 from eq. (3) of heat-equation.md evaluated at the nodes (own derivative).
Run in a kore run directory with the *.mtx files (after submatrices.py): python3 heat_check.py [ntrials]
'''
import sys, os, numpy as np
sys.argv, argv0 = ['ebal.py', 'none'], sys.argv
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ebal.py')).read().split("if __name__")[0])
import scipy.special as scsp
from math import factorial
ntr = int(argv0[1]) if len(argv0) > 1 else 3
rng = np.random.default_rng(1)

def random_fields(kmax, decay):
    def one(kind, nlk):
        out = np.zeros((nlk, N1), dtype=complex)
        for k in range(nlk):
            c = np.zeros(N, dtype=complex)
            c[:kmax] = (rng.standard_normal(kmax) + 1j*rng.standard_normal(kmax))*decay**np.arange(kmax)
            if ricb > 0:
                out[k] = c[:N1]
            else:
                out[k] = c[istart[kind]::2][:N1]
        return out
    return one('P', len(lp)), one('T', len(lt)), one('S', len(lp))

# ------------------------------------------------------------------ grid side
Nx = 2*par.lmax + 12
xg, wx = np.polynomial.legendre.leggauss(Nx)
sg = np.sqrt(1 - xg**2)
def Sch(l, x):   # Schmidt harmonic without exp(i m phi): sqrt((l-m)!/(l+m)!) P_l^m(x), same normalisation as ut.Ylm
    if l < m: return np.zeros_like(x)
    return np.sqrt(factorial(l - m)/factorial(l + m))*scsp.lpmv(m, l, x)
def dSch_dth(l, x):   # dY/dtheta = -sin(th) dY/dx, with (1-x^2) dP_l^m/dx = [(l+1)(l+m) P_{l-1}^m - l(l-m+1) P_{l+1}^m]/(2l+1)
    nrm = np.sqrt(factorial(l - m)/factorial(l + m))
    dPdx_1mx2 = ((l + 1)*(l + m)*scsp.lpmv(m, l - 1, x) - l*(l - m + 1)*scsp.lpmv(m, l + 1, x))/(2*l + 1) if l - 1 >= m else (-l*(l - m + 1)*scsp.lpmv(m, l + 1, x))/(2*l + 1)
    return -nrm*dPdx_1mx2/np.sqrt(1 - x**2)
P2x = (3*xg**2 - 1)/2; P4x = (35*xg**4 - 30*xg**2 + 3)/8
dP2x = 3*xg; dP4x = (35*xg**3 - 15*xg)/2
# radial profiles at the nodes rk (ebal), thermal wind from eq. (3) with its own derivative
g0 = rap.graviX(rk, 0); g1 = rap.graviX(rk, 1)
f0, f0p, f0pp = rap.abuX(rk, 0), rap.abuX(rk, 1), rap.abuX(rk, 2)
f2, f2p, f2pp = rap.aubX(rk, 0), rap.aubX(rk, 1), rap.aubX(rk, 2)
Bq = par.Beyonce
b2 = f0p/3 + f2p/21 + (4/7)*f2/rk;            b2p = f0pp/3 + f2pp/21 + (4/7)*(f2p/rk - f2/rk**2)
b4 = f2p - 2*f2/rk;                           b4p = f2pp - 2*(f2p/rk - f2/rk**2)
pref = rk**2/g0; prefp = 2*rk/g0 - rk**2*g1/g0**2
sig2 = -(2/Bq)*pref*b2;      sig2p = -(2/Bq)*(prefp*b2 + pref*b2p)
sig4 = -(6/(35*Bq))*pref*b4; sig4p = -(6/(35*Bq))*(prefp*b4 + pref*b4p)
h1 = rap.logrhoX(rk, 1)
if not par.thermal_wind:
    sig2 = sig4 = sig2p = sig4p = np.zeros_like(rk)
else:   # the code's Chebyshev fits of sigma2, sigma4 and their derivatives against the direct evaluation above
    prof_err = max(np.max(abs(a - b))/max(np.max(abs(b)), 1e-300) for a, b in
                   ((rap.sigma2X(rk, 0), sig2), (rap.sigma4X(rk, 0), sig4), (rap.sigma2X(rk, 1), sig2p), (rap.sigma4X(rk, 1), sig4p)))
    print('profiles: worst relative difference of rap.sigma2X/sigma4X (orders 0, 1) from the direct eq. (3) evaluation %.1e' % prof_err)
dxdr = 2/(rcmb - ricb) if ricb > 0 else 1/rcmb
def vals_and_der(vec, kind):
    c = full_coeffs(vec, istart[kind])
    return Tmat @ c, (ch.chebvander(xk, N - 2) @ ch.chebder(c))*dxdr

def grid_side(P, T, S):
    # fields on the (r, x) grid, shape (Nq, Nx)
    s = np.zeros((Nq, Nx), dtype=complex); ur = np.zeros_like(s); uth = np.zeros_like(s)
    for k, l in enumerate(lp):
        Y = Sch(l, xg); dY = dSch_dth(l, xg)
        sv = vals_unknown(S[k], 'S'); Pv, Pd = vals_and_der(P[k], 'P')
        s += np.outer(sv, Y)
        ur += np.outer(l*(l + 1)*Pv/rk, Y)
        uth += np.outer((Pv + rk*Pd + rk*h1*Pv)/rk, dY)
    for k, l in enumerate(lt):
        Y = Sch(l, xg); Tv = vals_unknown(T[k], 'T')
        uth += np.outer(1j*m*Tv, Y/sg)
    dOm = np.outer(f0, np.ones(Nx)) + np.outer(f2, P2x)
    dStw_dr = np.outer(sig2p, P2x) + np.outer(sig4p, P4x)
    dStw_dth = -(np.outer(sig2, dP2x) + np.outer(sig4, dP4x))*sg[None, :]
    F = 1j*m*dOm*s + ur*dStw_dr + uth*dStw_dth/rk[:, None]
    integrand = pss0[:, None]*np.conj(s)*F*rk[:, None]**2
    return 2*np.pi*jac*np.sum(wk[:, None]*wx[None, :]*integrand)

# ---- run (asm_check.py executes everything above this line)
print('m = %d, symm = %d, ricb = %g, lmax = %d, N = %d, diff_rot_type = %s, thermal_wind = %d' % (m, par.symm, ricb, par.lmax, N, par.diff_rot_type, par.thermal_wind))
worst = 0
for t in range(ntr):
    P, T, S = random_fields(max(4, N1//3), 0.7)
    X = cross_pairings([(P, T, S)], [(P, T, S)])
    H = -X[0][0]['H']
    G = grid_side(P, T, S)
    err = abs(H - G)/max(abs(G), 1e-300); worst = max(worst, err)
    print(' trial %d: matrix %s  grid %s  rel diff %.1e' % (t, H, G, err))
print('worst relative difference %.1e  ->  %s' % (worst, 'ok' if worst < 1e-8 else 'FAIL'))
