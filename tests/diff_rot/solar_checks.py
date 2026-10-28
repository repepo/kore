#!/usr/bin/env python3
'''
solar_checks.py DIFF_ROT_NPZ [WORKDIR]: the solar rows of docs/differential_rotation_details.md §11. Mukhopadhyay et al.
2025a (their Ω, mk_bekki.py) and 2025b (radiative interior, rin 0.5 R☉, mk_rz.py), each with the thermal wind (thermal_wind 2 without
u_r ∂S_tw/∂r) and without it (thermal_wind 0). Prints, per mode, the eigenvalues nearest the published one (nHz, frame
456 nHz, negative retrograde; growth rate second) with the solver residual and, for 2025b, the fraction of the kinetic
energy below 0.71 R☉ (CZ modes ~0, mixed modes large). One rank, scipy only (solve_bekki.py), < 1 GB per run.
DIFF_ROT_NPZ is data/diff_rot.npz of the Mukhopadhyay et al. 2025 data set (doi:10.17617/3.NP26AI).
'''
import os, sys, subprocess, re
import numpy as np
if len(sys.argv) < 2:
    sys.exit(__doc__)
T = os.path.dirname(os.path.abspath(__file__))
EDM = os.path.abspath(sys.argv[1])
W = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else 'solar_runs')
os.makedirs(W, exist_ok=True)

KE = r'''
import sys, numpy as np
sys.path.insert(0, 'bin')
from parameters import par
import utils as ut, radial_profiles as rap
import numpy.polynomial.chebyshev as C
z = np.load('eig.npz'); V = z['vecs']; f = z['f']; g = z['g']; res = z['res']
lp, lt, _ = ut.ell(par.m, par.lmax, par.symm); N1 = ut.N1; nb = len(lp)
a, b = par.ricb, 1.0
r = np.linspace(a, b, 4001)[1:-1]; x = 2*(r - a)/(b - a) - 1
rho = np.exp(rap.prf.logrho(r, 0)); h1 = rap.logrhoX(r, 1)
cut = 0.71/par.r_out
for j in range(V.shape[1]):
    e = np.zeros_like(r)
    for k, l in enumerate(lp):
        c = V[k*N1:(k+1)*N1, j]; L = l*(l + 1.); nl = 4*np.pi/(2*l + 1)
        P = C.chebval(x, c); dP = C.chebval(x, C.chebder(c))*2/(b - a)
        e += nl*(np.abs(L*P/r)**2 + L*np.abs(P/r + dP + h1*P)**2)
    for k, l in enumerate(lt):
        c = V[(nb + k)*N1:(nb + k + 1)*N1, j]; L = l*(l + 1.); nl = 4*np.pi/(2*l + 1)
        e += nl*L*np.abs(C.chebval(x, c))**2
    w = rho*e*r**2
    print('%d %.4f %.4f %.3f %.1e' % (j, f[j], g[j], np.trapezoid(w*(r < cut), r)/np.trapezoid(w, r), res[j]))
'''

def solve(run, tf, tg, k=8, ke=False):
    subprocess.run([sys.executable, T + '/solve_bekki.py', run, str(tf), str(tg), str(k)], capture_output=True, text=True, check=True)
    if ke:
        q = subprocess.run([sys.executable, '-c', KE], cwd=run, capture_output=True, text=True)
        rows = [l.split() for l in q.stdout.splitlines() if re.fullmatch(r'\d+ \S+ \S+ \S+ \S+', l.strip())]
        return [(float(a), float(b), float(c), float(d)) for _, a, b, c, d in rows]
    z = np.load(run + '/eig.npz')
    return [(z['f'][j], z['g'][j], np.nan, z['res'][j]) for j in range(len(z['f']))]

def report(label, tf, tg, rows, n=3):
    rows = sorted(rows, key=lambda t: abs(t[0] - tf) + (abs(t[1] - tg) if tg is not None else 0))
    print('%-44s published %9.3f %s' % (label, tf, '' if tg is None else '%+8.3f' % tg))
    for f, g, fr, res in rows[:n]:
        print('      kore %9.3f %+8.3f   %s res %.0e' % (f, g, '' if np.isnan(fr) else 'KE<0.71 %.2f ' % fr, res))
    sys.stdout.flush()

def build(script, run, args):
    if not os.path.isdir(run):
        p = subprocess.run([sys.executable, T + '/' + script, run] + args, capture_output=True, text=True)
        if p.returncode:
            print(run, 'BUILD FAILED', p.stdout[-600:], p.stderr[-600:]); return False
    return True

# Mukhopadhyay et al. 2025a: their Ω
A = [(1, -1, 'm 1 high-latitude', -95.70, 11.69), (3, -1, 'm 3 Rossby n0', -262.07, -0.88),
     (3, -1, 'm 3 Rossby n1', -182.05, -9.67), (3, 1, 'm 3 columnar', 210.84, -11.09), (10, 1, 'm 10 HFR', -211.28, -20.00)]
for tw in (2, 0):
    print('\n== Mukhopadhyay 2025a, their Ω, tw %d' % tw)
    for m, symm, name, tf, tg in A:
        run = '%s/mk25a_m%d%s_tw%d' % (W, m, 'a' if symm < 0 else 's', tw)
        if build('mk_bekki.py', run, ['m=%d' % m, 'symm=%d' % symm, 'file=' + EDM, 'tw=%d' % tw]):
            report(name, tf, tg, solve(run, tf, tg))

# Mukhopadhyay et al. 2025b: radiative interior, rin 0.5, 'solar_P4'
B = [('1(a) CZ high-lat. m=1', 1, -1, -92.2, None), ('1(b) mixed m=1', 1, -1, -94.1, None),
     ('2(a) CZ high-lat. m=2', 2, -1, -177.2, None), ('2(b) mixed m=2', 2, -1, -187.1, None),
     ('3(a) CZ Rossby m=2', 2, -1, -331.1, None), ('3(b) mixed m=2', 2, -1, -333.7, None),
     ('4(a) CZ Rossby m=5', 5, -1, -137.9, None), ('4(b) mixed m=5 (Table 2)', 5, -1, -136.740, -11.067),
     ('5(a) CZ HFR m=7', 7, 1, -248.5, None), ('5(b) mixed m=7 (Table 2)', 7, 1, -240.068, -25.593),
     ('m=3 equatorial Rossby (text)', 3, -1, -263.3, None)]
for tw in (0, 2):
    print('\n== Mukhopadhyay 2025b, rin 0.5, tw %d' % tw)
    for label, m, symm, tf, tg in B:
        run = '%s/mk25b_m%d%s_tw%d' % (W, m, 'a' if symm < 0 else 's', tw)
        if build('mk_rz.py', run, ['m=%d' % m, 'symm=%d' % symm, 'rin=0.5', 'tw=%d' % tw]):
            report(label, tf, tg, solve(run, tf, tg if tg is not None else -8.0, 10, ke=True), 4)
