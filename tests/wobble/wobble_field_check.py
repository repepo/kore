#!/usr/bin/env python3
'''
usage: wobble_field_check.py [WORKDIR] [NCPUS]   (default tests/wobble/runs_field, 4 ranks)

The torques of the field on the wobbling mantle (wobble = 1 with magnetic = 1, bin/wobble.py, NEW 2026-11-09), in small runs
(Ek 1e-4, Em 1e-3, Le 0.03: Elsasser number 0.9). Exit code 1 if a check fails. tests/wobble/derive_field.py checks the
field terms themselves without a run.

1. A heavy mantle (MoIX_M = 1e12) with a field: the modes of the fixed wall (wobble = 0) to 1e-9.
2. The angular momentum budget of the fluid with the field,
       λ(A_f Mc + L) + iG(C_f - A_f) Mc - iG L = -(Γ_P' + Γ_cent + Γ_ν) + (the torque of the field on the fluid),
   for every computed mode near the free core nutation and the Chandler wobble of a mantle of the paper's q = 1, closes
   (wobble.budget, relative 1e-6). It checks the Lorentz part of the pressure on the wall (in Γ_P') against the Maxwell
   stress through the wall (Γ_B = -(the torque on the fluid)):
     a. spheroid of flattening 0.1, insulating mantle, B0 'axial' (imposed from outside: Γ_B is the torque on its sources)
        and 'G21 dipole' (internal currents; Γ_B of the order of the flattening, from its wall currents on the spheroid);
        NEW (2026-11-09) with 'G21 dipole' also a thin conducting layer (c1_cmb 1e-2, its thin limit c_cmb = 0) and a
        resolved conducting mantle (r_surface 1.1, sigma_mantle 0.1) on the spheroid, where Γ_B is the torque on their currents;
     b. sphere, B0 'G21 dipole': an insulating mantle (Γ_B = 0 to round-off: the stress through the wall is that at infinity),
        a thin conducting layer ('TWA', c1_cmb 1e-2) and a resolved conducting mantle (r_surface 1.1, sigma_mantle 0.1), where
        Γ_B is the torque on the mantle's currents.
'''
import os, sys
import numpy as np
T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, T)
import kw

W = sys.argv[1] if len(sys.argv) > 1 else os.path.join(T, 'runs_field')
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 4
os.makedirs(W, exist_ok=True)
fails = []
MAG = dict(magnetic=1, Le=0.03, Em=1e-3)

def check(name, err, tol):
    ok = err < tol
    print('  %-78s %.1e (%s %.0e)' % (name, err, 'ok, <' if ok else 'FAIL, >=', tol), flush=True)
    if not ok:
        fails.append(name)

def lmax_for(N):
    return 2*NP*max(1, int(0.5*N/(2*NP)))

def show(tag, r):
    print('     %-28s λ = %+.8f %+.8fi  |Γ_P\'| %.2e (Lorentz %.2e) |Γ_ν| %.2e |Γ_B| %.2e  resid %.1e'
          % (tag, r['lam'].real, r['lam'].imag, abs(r['pressure']), abs(r['pressure_lorentz']), abs(r['viscous']),
             abs(r['magnetic']), r['resid']), flush=True)

print('1. heavy mantle with a field')
f = 0.02
d = kw.setup(os.path.join(W, 'heavy'), Ek=1e-4, f=f, N=96, lmax=lmax_for(96), ncpus=NP, B0='axial', **MAG)
w0 = kw.spin_over(f)
fixed = kw.solve(d, wobble=0, itau=w0, nev=6)
heavy = kw.solve(d, wobble=1, MoIX_M=1e12, dynflat_M=3e-3, itau=w0, nev=6)
lf = np.array([r['lam'] for r in fixed]); lh = np.array([r['lam'] for r in heavy])
check('f = %g, B0 axial: the fixed wall\'s modes near the spin-over' % f, max(min(abs(lh - l)) for l in lf)/abs(w0), 1e-9)

print('2. the angular momentum budget with a field')
cases = [('a', 0.1, 'axial', {}), ('a', 0.1, 'G21 dipole', {}),
         ('a', 0.1, 'G21 dipole', {'mantle': 'TWA', 'c_cmb': 0.0, 'c1_cmb': 1e-2}),          # (NEW 2026-11-09)
         ('a', 0.1, 'G21 dipole', {'mantle': 'resolved', 'r_surface': 1.1, 'N_mantle': 24, 'sigma_mantle': 0.1}),
         ('b', 0.0, 'G21 dipole', {'mantle': 'insulator'}),
         ('b', 0.0, 'G21 dipole', {'mantle': 'TWA', 'c_cmb': 1e-5, 'c1_cmb': 1e-2}),
         ('b', 0.0, 'G21 dipole', {'mantle': 'resolved', 'r_surface': 1.1, 'N_mantle': 24, 'sigma_mantle': 0.1})]
q = 1.0
for part, f, B0, extra in cases:
    name = '%s_%g_%s_%s' % (part, f, B0.replace(' ', ''), extra.get('mantle', 'insulator'))
    d = kw.setup(os.path.join(W, name), Ek=1e-4, f=f, N=128, lmax=lmax_for(128), ncpus=NP, B0=B0, **MAG, **extra)
    A, e = kw.paper(q, f) if f else (kw.paper(q, 0.0)[0], 1e-3)
    tag = '%s, f %g, %s, %s' % (part, f, B0, extra.get('mantle', 'insulator'))
    gb, gnu = [], []
    for itau in (kw.spin_over(f) + f*q, -f*(1 + q) if f else -2e-3):
        res = kw.solve(d, wobble=1, MoIX_M=A, dynflat_M=e, itau=itau, nev=4)
        for r in res:
            show('%s %+.3f' % (name, itau), r)
            gb.append(abs(r['magnetic'])); gnu.append(abs(r['viscous']) + abs(r['pressure']))
        check('%s, target %+.4f: budget residual (max over modes)' % (tag, itau), max(r['resid'] for r in res), 1e-6)
        if f:
            check('%s, target %+.4f: the Lorentz part of Γ_P\' is checked (> 1e-3 of Γ_P\')' % (tag, itau),
                  1e-3/max(abs(r['pressure_lorentz'])/abs(r['pressure']) for r in res), 1)
    if f == 0 and extra.get('mantle') == 'insulator':
        check('%s: no torque of the field on an insulating mantle, |Γ_B|/(|Γ_ν| + |Γ_P\'|)' % tag,
              max(np.array(gb)/np.array(gnu)), 1e-10)
    elif extra.get('mantle') in ('TWA', 'resolved'):   # (NEW 2026-11-09: on the spheroid too)
        check('%s: the conducting mantle feels the field, (|Γ_ν| + |Γ_P\'|)/|Γ_B| < 1e3' % tag,
              min(np.array(gnu)/np.array(gb))/1e3, 1)

print('\nwobble_field_check: ' + ('all checks passed' if not fails else 'FAILED: ' + '; '.join(fails)))
sys.exit(1 if fails else 0)
