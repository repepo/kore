#!/usr/bin/env python3
'''
usage: spheroid_mantle_check.py [WORKDIR] [NCPUS]   (default tests/mantle/runs_spheroid, 4 ranks)

NEW (2026-11-09): the conducting mantles on the exact spheroid (centrifugal_shape 'spheroid'; bin/centrifugal.py, section 8,
the end): the resolved mantle (mantle = 'resolved') and the thin conducting layer (mantle = 'TWA', its thin limit c = 0), in
runs (m 1, the run helper of tests/wobble). Exit code 1 if a check fails. derive_spheroid.py checks the rows of the thin
layer without a run.

1. A resolved mantle of the fluid's conductivity (σ = 1) on [1, r_surface] without the flow (hydro = 0): the field decays as in
   a uniform spheroid of the size r_surface with an insulating exterior, whose rates are those of the unit spheroid divided
   by r_surface² (a run with an insulating wall): the mantle's metric diffusion, its electric-field row at r = 1 and the
   insulating rows at r_surface; f = 0.1 and -0.1 (prolate), 1e-9.
2. The thin layer of conductance c1 against a resolved layer of thickness h (r_surface = 1 + h, ∫ σ dr = c1): the decay rates
   differ at first order in h (halving h halves the difference), for a uniform layer (σ = c1/h) and one whose σ falls by
   e³ across it (the (ln σ)' part of the mantle's rows gm), without the flow, and with the flow (Ek 1e-3, Em 1e-3, B0
   'G21 dipole', a no-slip wall); f = 0.1.
3. A flattening of 1e-7 against the sphere: the eigenvalues of the thin layer and of a resolved mantle whose σ varies (with
   the flow, B0 'G21 dipole') move by O(f) only, below 1e-6.
'''
import os, sys
import numpy as np
T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(T, '..', 'wobble'))
import kw   # noqa: E402

W = sys.argv[1] if len(sys.argv) > 1 else os.path.join(T, 'runs_spheroid')
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 4
os.makedirs(W, exist_ok=True)
fails = []

def check(name, err, tol):
    ok = err < tol
    print('  %-80s %.1e (%s %.0e)' % (name, err, 'ok, <' if ok else 'FAIL, >=', tol), flush=True)
    if not ok:
        fails.append(name)

def lmax_for(N):
    return 2*NP*max(1, int(0.5*N/(2*NP)))

def closest(lam, ref):
    '''the largest relative distance from a value of ref to the nearest of lam'''
    lam = np.asarray(lam)
    return max(np.min(abs(lam - z))/abs(z) for z in ref)

def eig(name, solve=None, **p):
    d = kw.setup(os.path.join(W, name), ncpus=NP, **p)
    return np.array([r['lam'] for r in kw.solve(d, **(solve or {}))])

DEC = dict(hydro=0, magnetic=1, B0='axial', cnorm=1, Le=1.0, Em=1.0, Ek=1e-3, N=48, lmax=lmax_for(48))
FLOW = dict(magnetic=1, B0='G21 dipole', Le=0.03, Em=1e-3, Ek=1e-3, N=64, lmax=lmax_for(64))
def resolved(rs, sigma, nm=16):
    return dict(mantle='resolved', r_surface=rs, sigma_mantle=sigma, N_mantle=nm)
def twa(c1):
    return dict(mantle='TWA', c_cmb=0.0, c1_cmb=c1)

print('1. a mantle of the fluid\'s conductivity: the decay of a larger spheroid')
RS = 1.25
for f in (0.1, -0.1):
    ins = eig('ins_%g' % f, dict(rtau=-10.0*RS**2, itau=0.0, nev=6), f=f, **DEC)
    res = eig('res1_%g' % f, dict(rtau=-10.0, itau=0.0, nev=6), f=f, **DEC, **resolved(RS, 1.0, 32))
    print('     insulator/r_s² %s\n     resolved      %s' % (np.sort_complex(ins/RS**2), np.sort_complex(res)), flush=True)
    ref = sorted(ins/RS**2, key=lambda z: abs(z + 10.0))[:3]   # the three nearest the target
    check('f = %+.1f: the rates of the resolved mantle (σ = 1) against the insulating spheroid / r_surface²' % f,
          closest(res, ref), 1e-9)

print('2. the thin layer against resolved layers of thickness h')
C1, K = 0.05, 3.0
for label, base, slv in (('decay', DEC, dict(rtau=-10.0, itau=0.0, nev=4)),
                         ('flow', FLOW, dict(rtau=0.0, itau=1.0, nev=4))):
    lt = eig('twa_%s' % label, slv, f=0.1, **base, **twa(C1))
    for prof in ('uniform', 'exp'):
        dif = []
        for h in (0.02, 0.01, 0.005):
            sig = C1/h if prof == 'uniform' else '%r*exp(-%r*(r - 1))' % (float(C1*K/(h*(1 - np.exp(-K)))), float(K/h))
            lr = eig('thin_%s_%s_%g' % (label, prof, h), slv, f=0.1, **base, **resolved(1 + h, sig, 12))
            dif.append(closest(lr, lt[:2]))
        q = [dif[1]/dif[0], dif[2]/dif[1]]
        print('     %s, %s σ: differences %s' % (label, prof, ' '.join('%.2e' % x for x in dif)), flush=True)
        check('%s, %s σ: halving h halves the difference from the thin layer (ratios %.2f, %.2f)' % (label, prof, *q),
              max(abs(x - 0.5) for x in q), 0.1)
        check('%s, %s σ: the difference at h = 0.005' % (label, prof), dif[2], 2e-3)

print('3. a flattening of 1e-7 against the sphere')
for name, extra in (('twa', twa(C1)), ('res', resolved(1.2, '0.5*exp(-2*(r - 1))', 24))):
    l0 = eig('%s_sphere' % name, dict(rtau=0.0, itau=1.0, nev=4), f=0.0, **FLOW, **extra)
    l1 = eig('%s_f1e-7' % name, dict(rtau=0.0, itau=1.0, nev=4), f=1e-7, **FLOW, **extra)
    print('     sphere %s\n     f 1e-7 %s' % (l0, l1), flush=True)
    check('%s: the eigenvalues at f = 1e-7 against the sphere\'s' % name, closest(l1, l0[:3]), 1e-6)

print('\nspheroid_mantle_check: ' + ('all checks passed' if not fails else 'FAILED: ' + '; '.join(fails)))
sys.exit(1 if fails else 0)
