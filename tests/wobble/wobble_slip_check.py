#!/usr/bin/env python3
'''
usage: wobble_slip_check.py [WORKDIR] [NCPUS]   (default tests/wobble/runs_slip, 4 ranks)

The wobble of the mantle (wobble = 1, bin/wobble.py) with a fluid that slips along the wall: a stress-free wall (bco = 0) or an
inviscid fluid (ViscosD = 0), whose torques on the mantle gain the inertia and the Coriolis force of the flow along the wall in
the reduced pressure and the wall terms in w of the viscous torque (wobble.slip_weights, NEW 2026-11-09). Exit code 1 if a
check fails. tests/wobble/derive.py checks the tables of those terms without a run.

1. Inviscid, against the exact solution: in the exact spheroid the uniform-vorticity flow w = Ω×α (T₁ alone) is an exact
   inviscid mode, and with the mantle (Hough's and Poincaré's model) the two modes, the free core nutation (FCN) and the
   Chandler wobble (CW), are the eigenvalues of the 2×2 pencil of
       λ(a² + c²) Ω - (2iG/c) Ω + (2λ/a) Mc = 0                           (the vorticity of the fluid, curl of the
                                                                         covariant momentum equation)
       λ(A_M + A_f) Mc + iG(e A_M + C_f - A_f) Mc + (λ - iG) L = 0,  L = (8π/15) Ω/a   (the angular momentum of the whole)
   (Ω, Mc in the conv. of wobble.py). Flattening 0.1 (q 1 and 10), 0.02 (q 1) and -0.1 (prolate, q 1; q 10, where the light
   oblate mantle on the prolate core is unstable: a pair λ = ±σ + iω), 1e-10.
2. Stress-free (Ek 1e-4): the angular momentum budget of the fluid (as in wobble_check.py) for the FCN and the CW of q 1 and 10
   in spheroids of flattening 0.02 and 0.1 (1e-8); in a sphere the stress-free wall has no torque on the mantle (|Γ_ν| below
   1e-12 of |λ L|: the wall terms in w cancel those of ∂w/∂r) and the CW is -i e (1e-10).
3. Stress-free towards inviscid: at flattening 0.1, q 1, the FCN and the CW of Ek 1e-4 and 1e-5 differ from the inviscid ones
   by O(Ek) (the ratio of the differences 0.1 within 0.02).
4. With a field (Le 0.03, Em 1e-3, flattening 0.1, q 1): the budget for the FCN and the CW (the modes with the largest pressure
   torque near the inviscid ones), stress-free and inviscid, with an insulating mantle and B0 'axial', a thin conducting layer
   and a resolved conducting mantle with B0 'G21 dipole' (1e-8).
'''
import os, sys
import numpy as np
import scipy.linalg as sl
T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, T)
import kw

W = sys.argv[1] if len(sys.argv) > 1 else os.path.join(T, 'runs_slip')
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 4
os.makedirs(W, exist_ok=True)
fails = []
INV = dict(Ek=0.0, ViscosD=0.0, rpower_v=3)   # the inviscid full sphere's rows
MAG = dict(magnetic=1, Le=0.03, Em=1e-3)

def check(name, err, tol):
    ok = err < tol
    print('  %-80s %.1e (%s %.0e)' % (name, err, 'ok, <' if ok else 'FAIL, >=', tol), flush=True)
    if not ok:
        fails.append(name)

def lmax_for(N):
    return 2*NP*max(1, int(0.5*N/(2*NP)))

def hough(f, A_M, e, G=1.0):
    '''the FCN and the CW of the inviscid fluid in the wobbling spheroid: the eigenvalues of the pencil of the header'''
    a, c = kw.axes(f)
    Af, Cf = (4*np.pi/15)*(a*a + c*c), (8*np.pi/15)*a*a
    k = 8*np.pi/(15*a)
    A = np.array([[2j*G/c, 0], [1j*G*k, -1j*G*(e*A_M + Cf - Af)]])
    B = np.array([[a*a + c*c, 2/a], [k, A_M + Af]])
    return sl.eigvals(A, B)

def near(d, z, nev=2, **p):
    '''the computed mode nearest z (the target 1e-4 off z, away from the exact eigenvalue)'''
    res = kw.solve(d, itau=z.imag + 1e-4, rtau=z.real, nev=nev, **p)
    return min(res, key=lambda r: abs(r['lam'] - z))

def mantle(q, f):
    A, e = kw.paper(q, abs(f))
    return dict(wobble=1, MoIX_M=A, dynflat_M=e)

print('1. inviscid: the FCN and the CW against the exact Poincaré-Hough modes')
for f, qs in ((0.1, (1.0, 10.0)), (0.02, (1.0,)), (-0.1, (1.0, 10.0))):
    d = kw.setup(os.path.join(W, 'inv_%g' % f), f=f, N=32, lmax=lmax_for(32), ncpus=NP, **INV)
    for q in qs:
        mp = mantle(q, f)
        for z in hough(f, mp['MoIX_M'], mp['dynflat_M']):
            r = near(d, z, **mp)
            print('     f %+g q %-4g exact %+.12f %+.12fi  kore %+.12f %+.12fi' % (f, q, z.real, z.imag, r['lam'].real,
                                                                                 r['lam'].imag), flush=True)
            check('f = %+g, q = %g, λ %+.4f%+.4fi: kore against the exact mode' % (f, q, z.real, z.imag),
                  abs(r['lam'] - z)/abs(z), 1e-10)

print('2. stress-free: the angular momentum budget')
for f in (0.02, 0.1, 0.0):
    d = kw.setup(os.path.join(W, 'sf_%g' % f), Ek=1e-4, bco=0, f=f, N=96, lmax=lmax_for(96), ncpus=NP)
    for q in (1.0, 10.0):
        mp = mantle(q, f) if f else dict(mantle(q, 0.0), dynflat_M=1e-3)
        for z in hough(f, mp['MoIX_M'], mp['dynflat_M']):
            r = near(d, z, nev=4, **mp)
            print('     f %-4g q %-4g λ = %+.8f %+.8fi  |Γ_P\'| %.2e |Γ_ν| %.2e  resid %.1e'
                  % (f, q, r['lam'].real, r['lam'].imag, abs(r['pressure']), abs(r['viscous']), r['resid']), flush=True)
            if f:
                check('f = %g, q = %g, λ near %+.4fi: budget residual' % (f, q, z.imag), r['resid'], 1e-8)
            else:
                scale = abs(r['lam'])*(abs(r['L']) + mp['MoIX_M']*abs(r['Mc']))
                check('sphere, q = %g, λ near %+.4fi: no torque of the stress-free wall, |Γ_ν|/|λ|(|L| + A_M|Mc|)'
                      % (q, z.imag), abs(r['viscous'])/scale, 1e-12)
                if z.imag < 0:
                    check('sphere, q = %g: the CW, -i e' % q, abs(r['lam'] + 1e-3j)/1e-3, 1e-10)

print('3. stress-free towards inviscid')
f, q = 0.1, 1.0
mp = mantle(q, f)
zs = sorted(hough(f, mp['MoIX_M'], mp['dynflat_M']), key=lambda z: -z.imag)   # the FCN (Im > 0), then the CW
dif = []
for Ek, N in ((1e-4, 96), (1e-5, 192)):
    d = kw.setup(os.path.join(W, 'ek_%g' % Ek), Ek=Ek, bco=0, f=f, N=N, lmax=lmax_for(N), ncpus=NP)
    dif.append([abs(near(d, z, **mp)['lam'] - z) for z in zs])
    print('     Ek %g: |λ - λ_inviscid| FCN %.3e, CW %.3e' % (Ek, *dif[-1]), flush=True)
for k, name in enumerate(('FCN', 'CW')):
    check('%s: the difference from the inviscid mode falls as Ek (ratio %.3f)' % (name, dif[1][k]/dif[0][k]),
          abs(dif[1][k]/dif[0][k] - 0.1), 0.02)

print('4. with a field')
f, q = 0.1, 1.0
mp = mantle(q, f)
cases = [('axial', {}), ('G21 dipole', {'mantle': 'TWA', 'c_cmb': 0.0, 'c1_cmb': 1e-2}),
         ('G21 dipole', {'mantle': 'resolved', 'r_surface': 1.1, 'N_mantle': 24, 'sigma_mantle': 0.1})]
for visc, vp in (('stress-free', dict(Ek=1e-4, bco=0)), ('inviscid', INV)):
    for B0, extra in cases:
        mt = extra.get('mantle', 'insulator')
        d = kw.setup(os.path.join(W, 'mag_%s_%s_%s' % (visc, B0.replace(' ', ''), mt)), f=f, N=128, lmax=lmax_for(128),
                     ncpus=NP, B0=B0, **vp, **MAG, **extra)
        for z in hough(f, mp['MoIX_M'], mp['dynflat_M']):
            res = kw.solve(d, itau=z.imag + 1e-4, rtau=0.0, nev=4, **mp)
            r = max(res, key=lambda r: abs(r['pressure']))
            print('     %-11s %-10s %-9s λ = %+.8f %+.8fi  |Γ_P\'| %.2e (Lorentz %.2e) |Γ_B| %.2e  resid %.1e'
                  % (visc, B0, mt, r['lam'].real, r['lam'].imag, abs(r['pressure']), abs(r['pressure_lorentz']),
                     abs(r['magnetic']), r['resid']), flush=True)
            check('%s, %s, %s, λ near %+.4fi: budget residual' % (visc, B0, mt, z.imag), r['resid'], 1e-8)

print('\nwobble_slip_check: ' + ('all checks passed' if not fails else 'FAILED: ' + '; '.join(fails)))
sys.exit(1 if fails else 0)
