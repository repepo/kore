#!/usr/bin/env python3
'''
usage: triana2019.py WORKDIR STAGE [NJOBS] [NCPUS]      STAGE: fig2, fig3, fits, fig5, fig9 or all (= fig2, fig3, fits)
       (environment: RES scales the resolution of every run, default 1; MPIEXEC overrides 'mpiexec')

kore with the wobble of the mantle (wobble = 1, bin/wobble.py) against Triana, Rekier, Trinh & Dehant (2019, GJI 218, 1071;
arXiv:1904.11837): a homogeneous fluid in a flattened mantle (flattening alpha of the wall and of the mantle's outer surface),
m = 1, equatorially antisymmetric, no-slip wall, in the paper's parameters: E = nu/(Omega a²) on the equatorial radius a
(kore's Ek = E a_k², kw.ek_kore), q = rho_f/(rho_m (r0⁵ - 1)) (kw.paper). NJOBS runs side by side (default 4), each on
NCPUS ranks (default 8). Each stage writes WORKDIR/STAGE.json (figs.py plots them). NEW (2026-11-09).

  fig2   q = 0.1, alpha = 1e-4, E = 1e-6: the Chandler wobble (CW), the free core nutation (FCN) and their neighbouring
         modes, with the ratio of the kinetic energies of the mantle and of the fluid in the steadily rotating frame
  fig3   the modes near the FCN at E = 1e-6, alpha = 1e-4 for 11 values of q from 0 to 30 (their Figs. 3, 4 and 10):
         frequency, damping, energies, the share of T11, the Poincaré power over the viscous dissipation, the pressure torque
         over the viscous torque
  fits   the FCN at q = 0, 0.05, 0.1 and the CW at q = 0.05, 0.1 for E = 1e-4..1e-8 (one per decade) and alpha = 1e-6 and
         10^-4.6: against their approximations eqs. 28-29 (FCN) and 32-33 (CW)
  fig5   (optional) the FCN at alpha = 1e-5, q = 5 for E from 1e-4 to 1e-8 (their Fig. 5)
  fig9   (optional) the inviscid modes near the spin-over (Ek = 0, no wobble) for alpha from 0 to 0.05 (their Fig. 9)
'''
import os, sys, json
import numpy as np
from multiprocessing import Pool
T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, T)
import kw

W = os.path.abspath(sys.argv[1]); STAGE = sys.argv[2]
NJOBS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
NP = int(sys.argv[4]) if len(sys.argv) > 4 else 8
RES = float(os.environ.get('RES', 1.0))
os.makedirs(W, exist_ok=True)


def resolution(E):
    '''(N, lmax) for the paper's E: the FCN converged to ~1e-7 in its damping at E = 1e-4, 1e-6, 1e-8 (alpha 1e-4 and
    10^-4.6, q 0.1) with N 96, 256, 832 and lmax 48, 96, 256; this rule (N 128, 320, 1024; lmax 64, 112, 320) adds a margin'''
    N = int(RES*max(128, 320*(E/1e-6)**-0.25))
    N += N % 2
    L = max(2*NP, int(RES*max(64, 112*(E/1e-6)**-0.23)))
    return N, 2*NP*int(np.ceil(L/(2*NP)))


def dedupe(recs, tol=1e-9):
    out = []
    for r in sorted(recs, key=lambda r: (r['lam'].imag, r['lam'].real)):
        if not out or min(abs(r['lam'] - o['lam']) for o in out) > tol*max(1, abs(r['lam'])):
            out.append(r)
    return out


def clean(r):
    '''a JSON-able record: complex numbers as [re, im]; the ratios the figures need'''
    o = {k: ([v.real, v.imag] if isinstance(v, complex) else v) for k, v in r.items()}
    if 'KE' in r:
        o['ratio_srf'] = r['KM_srf']/r['KE_srf']
        o['T11'] = r['KE_T11']/r['KE']
        o['DP_Dkin'] = r['DP']/r['Dkin']
        o['torque_ratio'] = abs(r['pressure'] + r['centrifugal'])/abs(r['viscous'])
        o['torque_ratio_reduced'] = abs(r['pressure'])/abs(r['viscous'])
    return o


def case(name, E, alpha, solves):
    '''one run directory (one E, alpha) and its solves: solves is a list of (tag, dict of kw.solve parameters)'''
    N, lmax = resolution(E) if E > 0 else (int(RES*80), 2*NP*int(np.ceil(RES*60/(2*NP))))
    extra = {} if E > 0 else {'rpower_v': 3}   # the inviscid full sphere's rows
    d = kw.setup(os.path.join(W, STAGE, name), Ek=kw.ek_kore(E, alpha), f=alpha, N=N, lmax=lmax, ncpus=NP, **extra)
    out = []
    for tag, p in solves:
        recs = kw.solve(d, **p)
        out.append({'tag': tag, 'p': p, 'modes': [clean(r) for r in recs]})
    for f in os.listdir(d):   # keep the logs and the results, not the matrices
        if f.endswith('.npz') or f.endswith('.npy'):
            os.remove(os.path.join(d, f))
    return {'name': name, 'E': E, 'alpha': alpha, 'N': N, 'lmax': lmax, 'solves': out}


def wob(q, alpha, **p):
    if q == 0:
        return dict(wobble=0, **p)
    A, e = kw.paper(q, alpha)
    return dict(wobble=1, MoIX_M=A, dynflat_M=e, **p)


def jobs():
    E6 = 1e-6
    if STAGE == 'fig2':
        q, a = 0.1, 1e-4
        fcn = kw.spin_over(a) + a*q
        sv = [('fcn%+.2f' % dw, wob(q, a, itau=fcn + dw, rtau=-0.002, nev=12)) for dw in (-0.05, 0.0, 0.05)]
        sv.append(('cw', wob(q, a, itau=-a*(1 + q), rtau=0.0, nev=12)))
        return [('q0.1', E6, a, sv)]
    if STAGE == 'fig3':
        a = 1e-4
        qs = np.array([0, 1, 2, 3, 5, 7, 10, 14, 18, 24, 30.])
        # (omega, sigma, nev): a 3x3 grid that follows the FCN from sigma/sqrt(E) = -2.7 to about -30, and a row at
        # sigma = -0.03 for the inertial modes down to sigma/sqrt(E) of about -38 (the paper's window is omega 1 to 1.018)
        targets = [(w, s, 8) for w in (1.0, 1.005, 1.01) for s in (-0.003, -0.01, -0.02)] + \
                  [(w, -0.03, 16) for w in (0.99, 1.0, 1.01, 1.02)]
        # one run directory per job: the q's split in NJOBS interleaved groups
        groups = [qs[i::NJOBS] for i in range(NJOBS)]
        return [('g%d' % i, E6, a, [('q%g_%g_%g' % (q, w, s), wob(q, a, itau=w, rtau=s, nev=n))
                                    for q in g for (w, s, n) in targets]) for i, g in enumerate(groups)]
    if STAGE == 'fig5':
        a, q = 1e-5, 5.0
        out = []
        for E in 10**np.arange(-4, -8.01, -0.25):
            s = -3*E**0.5
            out.append(('E%.2f' % np.log10(E), E, a, [('w%g' % w, wob(q, a, itau=w, rtau=s, nev=12))
                                                      for w in (0.995, 1.0, 1.005, 1.01)]))
        return out
    if STAGE == 'fits':
        out = []
        for E in 10**np.arange(-4, -8.01, -1.0):
            for a in (1e-6, 10**-4.6):
                sv = [('fcn_q%g' % q, wob(q, a, itau=kw.spin_over(a) + a*q, rtau=-2.63*E**0.5, nev=3)) for q in (0, 0.05, 0.1)]
                sv += [('cw_q%g' % q, wob(q, a, itau=-a*(1 + q), rtau=0.0, nev=3)) for q in (0.05, 0.1)]
                out.append(('E%.1f_a%.1f' % (np.log10(E), np.log10(a)), E, a, sv))
        return out
    if STAGE == 'fig9':
        return [('a%.4f' % a, 0.0, a, [('so', dict(wobble=0, itau=kw.spin_over(a), rtau=0.0, nev=40))])
                for a in np.round(np.linspace(0.0, 0.05, 26), 4)]
    sys.exit('unknown stage %r' % STAGE)


def run(job):
    try:
        return case(*job)
    except SystemExit as e:
        return {'name': job[0], 'E': job[1], 'alpha': job[2], 'error': str(e)[-2000:]}


if __name__ == '__main__':
    if STAGE == 'all':
        for s in ('fig2', 'fig3', 'fits'):
            os.system('%s %s %s %s %d %d' % (sys.executable, os.path.abspath(__file__), W, s, NJOBS, NP))
        sys.exit(0)
    js = jobs()
    with Pool(min(NJOBS, len(js))) as pool:
        res = []
        for r in pool.imap_unordered(run, js):
            res.append(r)
            print('%s: %s done (%d of %d)%s' % (STAGE, r['name'], len(res), len(js), ' ERROR' if 'error' in r else ''),
                  flush=True)
            json.dump(res, open(os.path.join(W, STAGE + '.json'), 'w'))
