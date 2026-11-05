#!/usr/bin/env python3
'''
algebra_check.py: the angular couplings of bin/centrifugal.py against a direct projection of its closed forms (NEW 2026-11-05).

centrifugal.primitives builds the metric terms with the recurrence algebra of bin/angular.py, as coupling matrices
C[k][shape][l', l]: the component k (q, s, t) of degree l' of the term applied to the source w = Y_l r̂ (shape q),
r∇Y_l (a) or r∇Y_l × r̂ (T). Here the same numbers come from the closed forms (derive.py) evaluated on a Gauss-Legendre
grid in cos θ, with harmonics from scipy.special.lpmv (not from angular.py), and projected:
    q_l' = ∫ W_r Y_l'* dΩ / N_l',   s_l' = ∫ W⋅(r∇Y_l')* dΩ / (L' N_l'),   t_l' = ∫ W⋅(r∇Y_l' × r̂)* dΩ / (L' N_l'),
N_l = ∫ |Y_l|² dΩ, L = l(l + 1), for every term (inertia and Coriolis, factors ε and η), m = 0..6 and l up to 36 (sources
l <= lw - 4, so that no target is cut by the truncation of the algebra). Also the P2 coupling of the heat rows
(heat_coefficients and _lookup, as entropy_block reads them). Errors are relative to the largest coefficient of the term.

Usage: python3 algebra_check.py [--quick]   (--quick: m = 0..3, l up to 12; a few seconds either way)
'''
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bin'))
import numpy as np
from scipy.special import lpmv, gammaln
import centrifugal as cf

quick = '--quick' in sys.argv
ms = range(0, 4) if quick else range(0, 7)
lw = 16 if quick else 40          # degrees 0..lw of the algebra
xg, wx = np.polynomial.legendre.leggauss(lw + 24)
sg = np.sqrt(1 - xg**2)
P2 = 1.5*xg**2 - 0.5

def norm(l, m):
    return np.exp(0.5*(gammaln(l - m + 1) - gammaln(l + m + 1)))

def Y(l, m):
    '''the θ-part of the Schmidt harmonic of utils.Ylm: sqrt((l-m)!/(l+m)!) P_l^m (Condon-Shortley phase, as lpmv)'''
    return norm(l, m)*lpmv(m, l, xg) if l >= m else 0*xg

def dY(l, m):
    '''∂θ Y_l^m = [(l - m + 1) P_{l+1}^m - (l + 1) x P_l^m]/sinθ, times the normalisation of Y_l^m'''
    return norm(l, m)*((l - m + 1)*lpmv(m, l + 1, xg) - (l + 1)*xg*lpmv(m, l, xg))/sg if l >= m else 0*xg

def source(sh, l, m):
    '''(w_r, w_θ, w_φ) of the unit source of degree l (factor e^{imφ} dropped)'''
    y, d = Y(l, m), dY(l, m)
    return {'q': (y, 0*y, 0*y), 'a': (0*y, d, 1j*m*y/sg), 'T': (0*y, 1j*m*y/sg, -d)}[sh]

def term(kind, amp, w):
    '''the closed form of the metric term (unit factor ε or η) applied to w on the grid'''
    wr, wt, wp = w
    wz = xg*wr - sg*wt
    zc = (-sg*wp, -xg*wp, xg*wt + sg*wr)               # ẑ × w
    rc = (0*wr, -wp, wt)                               # r̂ × w
    zh = (xg, -sg, 0*xg)
    if (kind, amp) == ('inertia', 'e'):
        return tuple(3*wz*zz - ww for zz, ww in zip(zh, w))
    if (kind, amp) == ('inertia', 'h'):
        rh = (1 + 0*xg, 0*xg, 0*xg)
        return tuple(1.5*xg*(wz*rr + wr*zz) - wr*rr - P2*ww for rr, zz, ww in zip(rh, zh, w))
    if (kind, amp) == ('coriolis', 'e'):
        return tuple(-a for a in zc)
    return tuple(-1.5*xg**2*a + 0.5*xg*b for a, b in zip(zc, rc))

def project(W, lt, m):
    '''(q, s, t) of degree lt of the field W'''
    y, d = Y(lt, m), dY(lt, m)
    N = np.sum(wx*y*y)
    L = lt*(lt + 1.)
    q = np.sum(wx*W[0]*y)/N
    if lt == 0:
        return q, 0., 0.
    s = np.sum(wx*(W[1]*d + W[2]*(-1j*m*y/sg)))/(L*N)
    t = np.sum(wx*(W[1]*(-1j*m*y/sg) - W[2]*d))/(L*N)
    return q, s, t

# the conventions: lpmv's harmonics are those of angular.ylm_grid (only the normalisation is shared)
import angular
dev = max(np.abs(angular.ylm_grid(m, lw, xg)[0][l] - Y(l, m)).max() for m in ms for l in range(m, lw + 1))
print('harmonics: lpmv against angular.ylm_grid %.1e (conventions)' % dev)

worst = {}
for m in ms:
    for kind in ('inertia', 'coriolis'):
        for amp in ('e', 'h'):
            C = cf.primitives(kind, amp, m, lw)[0]
            scale = max(abs(C[k][sh]).max() for k in 'qst' for sh in ('q', 'a', 'T'))
            err = 0.
            for sh in ('q', 'a', 'T'):
                for l in range(max(m, 1 if sh != 'q' else 0), lw - 3):
                    W = term(kind, amp, source(sh, l, m))
                    for lt in range(max(m, l - 4, 0), l + 5):
                        proj = project(W, lt, m)
                        for k, v in zip('qst', proj):
                            err = max(err, abs(C[k][sh][lt, l] - v))
            key = '%s %s' % (kind, amp)
            worst[key] = max(worst.get(key, 0.), err/scale)
    # the heat rows: P2 coupling, target l from source l + offdiag
    bands = cf.heat_coefficients(m)   # (on the degrees of the run's parameters.py: lmax + 8 >= lw here)
    err = 0.
    for l in range(m, lw - 3):
        for off in (-2, 0, 2):
            ls = l + off
            if ls < m:
                continue
            num = np.sum(wx*P2*Y(ls, m)*Y(l, m))/np.sum(wx*Y(l, m)**2)
            err = max(err, abs(cf._lookup(bands, l, m, off) - num))
    worst['heat P2'] = max(worst.get('heat P2', 0.), err)

tol = 1e-12
for k, v in worst.items():
    print('%-14s worst %.2e %s' % (k, v, 'ok' if v < tol else 'FAIL'))
bad = [k for k, v in worst.items() if not v < tol]
print('algebra_check.py: %s (m = %d..%d, l <= %d, tol %g)' % ('FAIL ' + ', '.join(bad) if bad else 'ok', ms[0], ms[-1], lw - 4, tol))
sys.exit(1 if bad else 0)
