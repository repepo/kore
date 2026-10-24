#!/usr/bin/env python3
'''
builder_check.py [tol]: the row builder of bin/reynolds_stress.py against kore's inertia operator.

Run in a kore run directory after submatrices.py (any parameters with ViscosD > 0). A force equal to a velocity field,
f = u (fP = U = L P/r, fB = V = P' + P (1/r + (ln rho0)'), fC = W = -T), must give the rows of the inertia operator
applied to u: rows_u(l) = op.inertia(l, 'u', 'upol', 0) P_l and rows_v(l) = op.inertia(l, 'v', 'utor', 0) T_l. This
checks the projection b_u = r^(rpower_u - 2) rho0^rhopower_u [fP - (r fB)']/L^2,
b_v = -r^(rpower_v - 1) rho0^rhopower_v fC/L^2 (the run's powers, as in its inertia rows), the Chebyshev transform, the
basis change, the parity rows (ricb = 0) and the bc offset, for every l, with random smooth P and T. Prints the worst
relative difference per section and 'ok' or 'FAIL' (default tol 1e-9).
'''
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'bin'))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import numpy.polynomial.chebyshev as ch
from parameters import par
import utils as ut
import operators as op
import reynolds_stress as rs
import rsf_common as rc

tol = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-9
rng = np.random.default_rng(1)
bd = rs.Builder()
r, h1, h2 = bd.r, bd.bg['h1'], bd.bg['h2']
x = r if par.ricb == 0 else (r - par.ricb)*rc.dxdr - 1   # Chebyshev variable at the builder's nodes
kmax = max(8, ut.N1//4)
win = np.exp(-np.arange(kmax)/(kmax/8))

def random_coef():
    c = np.zeros(ut.N1, dtype=complex)
    c[:kmax] = win*(rng.standard_normal(kmax) + 1j*rng.standard_normal(kmax))
    return c

def values(coef, kind, nd):
    c = np.zeros(par.N, dtype=complex)
    if par.ricb > 0:
        c[:] = coef
    else:
        c[rc.istart[kind]::2] = coef
    return [ch.chebval(x, ch.chebder(c, k) if k else c)*rc.dxdr**k for k in range(nd)]

worst = {'u': 0., 'v': 0.}
for l in rc.lp:
    L = l*(l + 1.)
    p = random_coef()
    P, dP, d2P = values(p, 'P', 3)
    fP = L*P/r
    fB = dP + P*(1/r + h1)
    dfB = d2P + dP*(1/r + h1) + P*(h2 - 1/r**2)
    b = bd.rows_u(l, fP, fB, dfB)[:, 0]
    ref = op.inertia(l, 'u', 'upol', 0) @ p
    worst['u'] = max(worst['u'], np.abs(b - ref).max()/np.abs(ref).max())
for l in rc.lt:
    t = random_coef()
    T, = values(t, 'T', 1)
    b = bd.rows_v(l, -T)[:, 0]
    ref = op.inertia(l, 'v', 'utor', 0) @ t
    worst['v'] = max(worst['v'], np.abs(b - ref).max()/np.abs(ref).max())

ok = max(worst.values()) < tol
print('builder_check (N %d, lmax %d, m %d, symm %d, ricb %g): f = u against the inertia rows, max rel. diff  u %.1e  v %.1e   %s'
      % (par.N, par.lmax, par.m, par.symm, par.ricb, worst['u'], worst['v'], 'ok' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
