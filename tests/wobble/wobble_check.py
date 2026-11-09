#!/usr/bin/env python3
'''
usage: wobble_check.py [WORKDIR] [NCPUS]   (default tests/wobble/runs, 4 ranks)

The wobble of the mantle (wobble = 1, bin/wobble.py) against exact limits and the conservation of angular momentum.
Exit code 1 if a check fails. NEW (2026-11-09).

1. A heavy mantle (MoIX_M = 1e12): the modes of the fixed wall (wobble = 0) to 1e-9, and the mantle's free wobble
   λ = -i dynflat_M (prograde) to 1e-9, in a sphere and in a spheroid.
2. The torques: for every computed mode of a mantle of the paper's q = 1 (and 10) in a spheroid of flattening 0.02 and 0.1 and
   in a sphere, the angular momentum budget of the fluid in the frame of the mantle,
       λ(A_f Mc + L) + iG(C_f - A_f) Mc - iG L = -(Γ_P' + Γ_cent + Γ_ν),
   closes (wobble.budget, relative 1e-6), with the torque of the reduced pressure Γ_P' a sizeable part of the total in the
   spheroids, so that it is checked on its own; in the sphere Γ_P' = 0.
3. The Chandler wobble (E 1e-4 on the equatorial radius, q 1, flattening 1e-3) against the fit of Triana et al. (2019,
   eq. 32), ω_CW ≈ -α(1 + q), to 2 %.
'''
import os, sys
import numpy as np
T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, T)
import kw

W = sys.argv[1] if len(sys.argv) > 1 else os.path.join(T, 'runs')
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 4
os.makedirs(W, exist_ok=True)
fails = []

def check(name, err, tol):
    ok = err < tol
    print('  %-70s %.1e (%s %.0e)' % (name, err, 'ok, <' if ok else 'FAIL, >=', tol), flush=True)
    if not ok:
        fails.append(name)

def lmax_for(N):
    return 2*NP*max(1, int(0.5*N/(2*NP)))

print('1. heavy mantle')
for f in (0.0, 0.02):
    d = kw.setup(os.path.join(W, 'heavy_%g' % f), Ek=1e-4, f=f, N=48, lmax=lmax_for(48), ncpus=NP)
    w0 = kw.spin_over(f)
    fixed = kw.solve(d, wobble=0, itau=w0, nev=6)
    heavy = kw.solve(d, wobble=1, MoIX_M=1e12, dynflat_M=3e-3, itau=w0, nev=6)
    lf = np.array([r['lam'] for r in fixed]); lh = np.array([r['lam'] for r in heavy])
    err = max(min(abs(lh - l)) for l in lf)/abs(w0)
    check('f = %g: the fixed wall\'s modes near the spin-over' % f, err, 1e-9)
    cw = kw.solve(d, wobble=1, MoIX_M=1e12, dynflat_M=3e-3, itau=-3e-3, nev=2)
    err = min(abs(r['lam'] + 3e-3j) for r in cw)/3e-3
    check('f = %g: the free wobble of the mantle alone, -i e' % f, err, 1e-9)

print('2. torques and the angular momentum budget')
for f in (0.0, 0.02, 0.1):
    d = kw.setup(os.path.join(W, 'budget_%g' % f), Ek=1e-4, f=f, N=128, lmax=lmax_for(128), ncpus=NP)
    for q in (1.0, 10.0):
        A, e = kw.paper(q, f) if f else kw.paper(q, 0.0)
        for itau in (kw.spin_over(f) + f*q, -f*(1 + q) if f else -1e-3):
            res = kw.solve(d, wobble=1, MoIX_M=A, dynflat_M=e if f else 1e-3, itau=itau, nev=4)
            for r in res:
                tot = abs(r['total'])
                print('     f %-5g q %-4g λ = %+.8f %+.8fi  |Γ_P\'| %.2e |Γ_cent| %.2e |Γ_ν| %.2e  resid %.1e'
                      % (f, q, r['lam'].real, r['lam'].imag, abs(r['pressure']), abs(r['centrifugal']),
                         abs(r['viscous']), r['resid']))
            check('f = %g, q = %g, target %+.4f: budget residual (max over modes)' % (f, q, itau),
                  max(r['resid'] for r in res), 1e-6)
            if f == 0:
                check('f = 0: no pressure torque', max(abs(r['pressure']) + abs(r['centrifugal']) for r in res), 1e-14)

print('3. Chandler wobble of a nearly rigid body')
alpha, q, E = 1e-3, 1.0, 1e-4
d = kw.setup(os.path.join(W, 'cw'), Ek=kw.ek_kore(E, alpha), f=alpha, N=128, lmax=lmax_for(128), ncpus=NP)
A, e = kw.paper(q, alpha)
res = kw.solve(d, wobble=1, MoIX_M=A, dynflat_M=e, itau=-alpha*(1 + q), nev=2)
lam = min((r['lam'] for r in res), key=lambda l: abs(l + 1j*alpha*(1 + q)))
fit = alpha*(-1 - 1.02e-5 - 4.75e-3*E**0.5 + 0.237*E + q*(-1 + 1.21e-5 + 3.292*E**0.5 + 12.53*E))
print('     CW λ = %+.6e %+.6ei, paper fit ω %.6e' % (lam.real, lam.imag, fit))
check('CW frequency against the paper\'s fit (relative)', abs(lam.imag - fit)/abs(fit), 2e-2)

print('\nwobble_check: ' + ('all checks passed' if not fails else 'FAILED: ' + '; '.join(fails)))
sys.exit(1 if fails else 0)
