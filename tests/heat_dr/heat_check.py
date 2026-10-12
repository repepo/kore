#!/usr/bin/env python3
'''
heat_check.py: check the differential-rotation blocks of the heat equation (operators.thermal_diffrot) against a
direct (r, theta) quadrature of the same terms built from the 3-D fields, for random P, T, s.

Matrix side: H = -<s, blocks(P, T, s)> with the pairing of ebal.py (the blocks carry the sign of the A right-hand side).
Grid side:   G = int p0 conj(s) [ i m Gaspard dOmega s + u_r dS_tw/dr + (u_theta/r) dS_tw/dtheta ] dV, with
             u_r = L P Y/r, u_theta = (1/r)[(rP)' + r h' P] dY/dtheta + i m T Y/sin(theta), dOmega = f0 + f2 P2, and
             S_tw from the thermal-wind balance itself, not from the closed forms: with x = cos(theta),
             dS_tw/dx = -(Gaspard^2 r^2/(Beyonce g)) [x F_r + (1 - x^2) F_x/r],  F = 2 dOmega (thermal_wind = 1)
             or (1 + dOmega)^2 - 1 (thermal_wind = 2), integrated in x (Legendre series, l = 0 part dropped); dS_tw/dr from
             the same integral of the r derivative of dS_tw/dx. Its P2, P4, P6 coefficients are compared with the code's
             fits rap.sigma2X/4X/6X. thermal_wind = 3 (prescribed): dS_tw/dtheta = dSdtheta sin(2 theta), dS_tw/dx = -2 dSdtheta x.
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
# radial profiles at the nodes rk (ebal); thermal wind integrated from the balance on the (r, x) grid
G2B = par.Gaspard**2/par.Beyonce if par.thermal_wind in (1, 2) else 0.
g0 = rap.graviX(rk, 0); g1 = rap.graviX(rk, 1)
# ricb = 0: the physical f0, f2 are even in r, so take them at |r| (at r itself, the odd part of a mixed-parity profile
# would integrate to zero over -1 < r < 1 here exactly as submatrices.py drops it, and the check could not see it)
ra, sr = (np.abs(rk), np.sign(rk)) if ricb == 0 else (rk, 1)
f0, f0p, f0pp = rap.abuX(ra, 0), sr*rap.abuX(ra, 1), rap.abuX(ra, 2)
f2, f2p, f2pp = rap.aubX(ra, 0), sr*rap.aubX(ra, 1), rap.aubX(ra, 2)
R, X = rk[:, None], xg[None, :]
P2x = (3*X**2 - 1)/2
dO, dO_r, dO_rr = f0[:, None] + f2[:, None]*P2x, f0p[:, None] + f2p[:, None]*P2x, f0pp[:, None] + f2pp[:, None]*P2x
dO_x, dO_rx = 3*X*f2[:, None], 3*X*f2p[:, None]
if par.thermal_wind == 2:   # F = (1 + dO)^2 - 1
    F_r, F_x = 2*(1 + dO)*dO_r, 2*(1 + dO)*dO_x
    F_rr, F_rx = 2*(dO_r**2 + (1 + dO)*dO_rr), 2*(dO_r*dO_x + (1 + dO)*dO_rx)
else:                       # F = 2 dO
    F_r, F_x, F_rr, F_rx = 2*dO_r, 2*dO_x, 2*dO_rr, 2*dO_rx
pre, pre_r = -G2B*R**2/g0[:, None], -G2B*(2*R/g0[:, None] - R**2*g1[:, None]/g0[:, None]**2)
dS_dx  = pre*(X*F_r + (1 - X**2)*F_x/R)
dS_dxr = pre_r*(X*F_r + (1 - X**2)*F_x/R) + pre*(X*F_rr + (1 - X**2)*(F_rx/R - F_x/R**2))
aS = abs(pre)*(abs(F_r) + abs(F_x/R))   # size of grad S_tw before the cancellations (S_tw = 0 for 'cylindrical')
if par.thermal_wind == 3:   # prescribed, uniform in r: dS_tw/dtheta = dSdtheta sin(2 theta) = -sin(theta) dS_tw/dx
    dS_dx, dS_dxr, aS = -2*par.dSdtheta*X + 0*R, 0*R*X, abs(par.dSdtheta) + 0*R*X
def x_integral(h):   # Legendre coefficients (rows) of the x-antiderivative of h(r, x) without its l = 0 part
    nl = 10   # h is a polynomial of degree <= 7 in x
    Pn = np.array([np.polynomial.legendre.legval(xg, np.eye(nl)[n]) for n in range(nl)])
    a = (h*wx[None, :]) @ Pn.T*((2*np.arange(nl) + 1)/2)[None, :]
    S = np.polynomial.legendre.legint(a, lbnd=-1, axis=1); S[:, 0] = 0
    return S
Sc, Sc_r = x_integral(dS_dx), x_integral(dS_dxr)
Stw_r = np.polynomial.legendre.legval(xg, Sc_r.T)        # (Nq, Nx)
Stw_th = -sg[None, :]*dS_dx                             # dS/dtheta = -sin(theta) dS/dx
h1 = rap.logrhoX(rk, 1)
print('thermal wind: largest odd-n / even-n Legendre coefficient of S_tw %.1e (must be round-off)' % (np.max(abs(Sc[:, 1::2]))/max(np.max(abs(Sc)), 1e-300)))
if par.thermal_wind:   # the code's Chebyshev fits of sigma_n and their derivatives against the integral above
    fits = [(rap.sigma2X(rk, 0), Sc[:, 2]), (rap.sigma4X(rk, 0), Sc[:, 4]), (rap.sigma2X(rk, 1), Sc_r[:, 2]), (rap.sigma4X(rk, 1), Sc_r[:, 4])]
    if par.thermal_wind == 2:
        fits += [(rap.sigma6X(rk, 0), Sc[:, 6]), (rap.sigma6X(rk, 1), Sc_r[:, 6])]
    else:
        print('thermal_wind = %d: max |sigma6| / max |sigma2| from the integral = %.1e (must be round-off)' % (par.thermal_wind, np.max(abs(Sc[:, 6]))/np.max(abs(Sc[:, 2]))))
    prof_err = max(np.max(abs(a - b))/max(np.max(abs(b)), 1e-300) for a, b in fits) if par.thermal_wind != 3 else \
        max(np.max(abs(a - b)) for a, b in fits)/np.max(abs(Sc[:, 2]))   # sigma4 and sigma2' are 0: relative to sigma2
    print('profiles: worst relative difference of rap.sigmaNX (n = 2, 4%s; orders 0, 1) from the thermal-wind integral %.1e' % (', 6' if par.thermal_wind == 2 else '', prof_err))
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
    F = 1j*m*par.Gaspard*dO*s + ur*Stw_r + uth*Stw_th/rk[:, None]
    Fa = max(m, 1)*par.Gaspard*abs(dO*s) + aS*(abs(ur) + abs(uth/rk[:, None]))   # scale of |F|, for a G that vanishes
    w = 2*np.pi*jac*wk[:, None]*wx[None, :]*pss0[:, None]*rk[:, None]**2
    return np.sum(w*np.conj(s)*F), np.sum(w*abs(s)*Fa)

# ---- run (asm_check.py executes everything above this line)
print('m = %d, symm = %d, ricb = %g, lmax = %d, N = %d, diff_rot_type = %s, amplitude = %g, thermal_wind = %d, Gaspard = %g, ThermaD = %g' % (m, par.symm, ricb, par.lmax, N, par.diff_rot_type, par.diff_rot_amplitude, par.thermal_wind, par.Gaspard, par.ThermaD))
worst = 0; Hs = []
for t in range(ntr):
    P, T, S = random_fields(max(4, N1//3), 0.7)
    X = cross_pairings([(P, T, S)], [(P, T, S)])
    H = -X[0][0]['H']; Hs.append(H)
    G, Ga = grid_side(P, T, S)
    err = abs(H - G)/max(abs(G), Ga, 1e-300); worst = max(worst, err if np.isfinite(err) else np.inf)
    print(' trial %d: matrix %s  grid %s  rel diff %.1e' % (t, H, G, err))
np.save('heat_check_H.npy', np.array(Hs))   # for tw_limit.py
print('worst relative difference %.1e  ->  %s' % (worst, 'ok' if worst < 1e-8 else 'FAIL'))
