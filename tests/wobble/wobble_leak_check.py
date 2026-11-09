#!/usr/bin/env python3
"""
usage: wobble_leak_check.py [WORKDIR] [NCPUS]   (default tests/wobble/runs_leak, 4 ranks)

The wobble of the mantle (wobble = 1, bin/wobble.py) with a toroidal B0 that does not vanish at the wall and 'leaks' into a
conducting mantle (NEW 2026-11-09): B0 'G21 dipole' plus B0_tor 'user' k = r² of degree 2 (cnorm_tor 1), Ek 1e-4, Le 0.03,
Em 1e-3, the paper's q = 1. Exit code 1 if a check fails.

1. The torque on a resolved mantle (r_surface 1.1, sigma_mantle 0.1) computed inside it: the volume integral of the
   linearised Lorentz force J0 × b + j × B0 over the mantle, with B0 continued into it (background_field.solid_profiles:
   the toroidal part the steady field of a uniform conductor, zero at r_surface) and b the mantle's own solution, plus the
   Maxwell stress through r_surface, against wobble.py's Γ_B (minus the Maxwell stress through the wall on the fluid's side),
   for every computed mode near the free core nutation and the Chandler wobble, in a sphere and in a spheroid of flattening
   0.1, no-slip and stress-free (1e-9). In the sphere the stress through r_surface is 0 (1e-10 of Γ_B), and the leaked
   field and its currents give a sizeable part of the torque (more than 10 % of it for the modes that turn the mantle).
2. The angular momentum budget of the fluid (wobble.budget) for the mode that turns the mantle most in each solve, with
   the resolved mantle and the thin layer (c1 1e-2), the same four cases (1e-8).
3. Thin resolved layers (h = 0.02, 0.01, 0.005, sigma = c1/h) against the thin layer (c_cmb 0, c1_cmb 1e-2): the FCN and the
   CW and their torques Γ_B/Mc differ at first order in h (ratios 0.5 within 0.1), sphere and spheroid 0.1, no-slip.
4. With an insulating mantle the leaking B0_tor still stops (its current sheet at the wall).
"""
import os, sys, json, shutil, subprocess
import numpy as np
T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, T)
import kw

W = sys.argv[1] if len(sys.argv) > 1 else os.path.join(T, 'runs_leak')
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 4
os.makedirs(W, exist_ok=True)
fails = []
MAG = dict(magnetic=1, Le=0.03, Em=1e-3, B0='G21 dipole', B0_tor='user', B0_tor_l=2, B0_tor_profile='r**2', cnorm_tor=1.0)
RES = dict(mantle='resolved', r_surface=1.1, N_mantle=24, sigma_mantle=0.1)
C1 = 1e-2
TWA = dict(mantle='TWA', c_cmb=0.0, c1_cmb=C1)
q = 1.0

def check(name, err, tol):
    ok = err < tol
    print('  %-84s %.1e (%s %.0e)' % (name, err, 'ok, <' if ok else 'FAIL, >=', tol), flush=True)
    if not ok:
        fails.append(name)

def lmax_for(N):
    return 2*NP*max(1, int(0.5*N/(2*NP)))

def mantle(f):
    A, e = kw.paper(q, f) if f else (kw.paper(q, 0.0)[0], 1e-3)
    return dict(wobble=1, MoIX_M=A, dynflat_M=e)

def targets(f):
    return (kw.spin_over(f) + f*q + 0.003, -f*(1 + q) - 0.003 if f else -2e-3)

ANALYSE = """#!/usr/bin/env python3
'''
Run in a kore run directory after solve.py (wobble = 1, magnetic = 1, mantle = 'resolved'): for each eigenpair, the torque
of the field on the mantle three ways, conv. as wobble.budget:
  GB      wobble.budget's Γ_B (minus the Maxwell stress through the wall on the fluid's side), with Hendrik
  Tvol    ∫ X × F_body dV over the mantle, F = J × B linearised (J0 × b + j × B0), B0 continued into the mantle by
          background_field.solid_profiles (poloidal: current-free; toroidal: the steady field of a uniform conductor, zero
          at r_surface), b the mantle's own solution; on the spheroid in reference variables, X × F_body = D⁻¹(α × D⁻² F_cov)
  Ms      the Maxwell stress through r_surface (mantle side): torque on everything inside r_surface
  Tleak   Tvol with the toroidal B0 alone (the leaked field and its currents)
Identity: Tvol = Ms - M(1) = Ms + GB; in a sphere with internal sources Ms = 0, so Tvol = GB. Writes leak.json.
'''
import sys, json
sys.path.insert(0, 'bin')
import numpy as np
import numpy.polynomial.chebyshev as ch
from parameters import par
import utils as ut
import wobble as wb
import background_field as bf

ev = np.loadtxt('eigenvalues0.dat', ndmin=2)
lam = ev[:, 0] + 1j*ev[:, 1]
flow, M = ut.load_field('flow'), ut.load_field('wobble')
field, man = ut.load_field('magnetic'), ut.load_field('magnetic_mantle')
a, c = wb.axes()
beta = 1/c**2 - 1/a**2
R1, R2, Nm = ut.rcmb, par.r_surface, par.N_mantle
lpm, ltm = [np.asarray(x, dtype=int) for x in ut.ell(par.m, par.lmax, ut.bsymm)[:2]]
nb = len(lpm)
pol, tor = bf.solid_profiles('outer')
nB = max([1] + list(pol) + list(tor))
gr = wb._field_grid(par.lmax + 1, nB, par.lmax + 2*nB + 24)
x, s, w = gr['x'], gr['s'], gr['w']

def B0_at(R, parts='pt'):
    rr = np.array([R])
    return {'h': {n: tuple(float(np.ravel(f(rr, d))[0]) for d in range(3)) for n, f in pol.items()} if 'p' in parts else {},
            'k': {n: tuple(float(np.ravel(f(rr, d))[0]) for d in range(2)) for n, f in tor.items()} if 't' in parts else {}}

def jets(bman, R):
    xm = 2*(R - R1)/(R2 - R1) - 1
    def mfun(cc, nd):
        cc = np.array(cc, dtype=complex); out = []
        for _ in range(nd + 1):
            out.append(ch.chebval(xm, cc)); cc = ch.chebder(cc)*2/(R2 - R1)
        return out
    Fj = np.array([mfun(bman[k*Nm:(k + 1)*Nm], 2) for k in range(nb)])[None]
    Gj = np.array([mfun(bman[(nb + k)*Nm:(nb + k + 1)*Nm], 1) for k in range(len(ltm))])[None]
    return Fj, Gj

def torque_density(fl):
    Fr, Ft, Fp = wb._lorentz(fl)
    dens = (-x*Fp - 1j*Ft)/a**2 + beta*(x*Fr - s*Ft)*(1j*s)
    return (2*np.pi/wb._N1/a)*(dens @ w)

xr, wr = np.polynomial.legendre.leggauss(max(3*Nm, 64))
rq = R1 + 0.5*(R2 - R1)*(xr + 1); wq = 0.5*(R2 - R1)*wr
out = []
for i, l in enumerate(lam):
    b = wb.budget(flow[:, i], M[0, i], l, field[:, i])
    bman = man[:, i]
    tv = {'pt': 0j, 't': 0j}
    for R, wR in zip(rq, wq):
        Fj, Gj = jets(bman, R)
        for parts in tv:
            fl = wb._wall_fields(gr, R, a, c, lpm, Fj, ltm, Gj, B0_at(R, parts))
            tv[parts] += wR*R**3*np.ravel(torque_density(fl))[0]
    Fj, Gj = jets(bman, R2)
    Ms = np.ravel(wb._maxwell(gr, wb._wall_fields(gr, R2, a, c, lpm, Fj, ltm, Gj, B0_at(R2)), R2, a, c))[0]
    H = par.Hendrik
    rec = {'lam': [l.real, l.imag], 'resid': b['resid'], 'GB': b['magnetic'], 'Tvol': H*tv['pt'], 'Tleak': H*tv['t'],
           'Ms': H*Ms, 'pressure': b['pressure'], 'viscous': b['viscous']}
    out.append({k: ([v.real, v.imag] if isinstance(v, complex) else float(v) if not isinstance(v, list) else v)
                for k, v in rec.items()})
json.dump(out, open('leak.json', 'w'))
"""

def analyse(d):
    open(os.path.join(d, 'leak_analyse.py'), 'w').write(ANALYSE)
    r = subprocess.run([sys.executable, 'leak_analyse.py'], cwd=d, capture_output=True, text=True)
    if r.returncode:
        sys.exit('%s: leak_analyse.py failed\n%s' % (d, r.stderr[-3000:]))
    out = json.load(open(os.path.join(d, 'leak.json')))
    return [{k: (complex(*v) if isinstance(v, list) else v) for k, v in o.items()} for o in out]

def turns(r):
    return abs(r['Mc'])*abs(r['lam'])

print('1, 2. the torque on the resolved mantle from inside it, and the budgets')
for f in (0.0, 0.1):
    for bco in (1, 0):
        for name, extra in (('resolved', RES), ('TWA', TWA)):
            d = kw.setup(os.path.join(W, '%s_%g_b%d' % (name, f, bco)), Ek=1e-4, bco=bco, f=f, N=128, lmax=lmax_for(128),
                         ncpus=NP, **MAG, **extra)
            for itau in targets(f):
                res = kw.solve(d, itau=itau, nev=4, **mantle(f))
                tag = 'f %g, bco %d, %s, target %+.3f' % (f, bco, name, itau)
                top = max(res, key=turns)
                check('%s: budget of the mode that turns the mantle most' % tag, top['resid'], 1e-8)
                if name != 'resolved':
                    continue
                lk = analyse(d)
                for r, k in zip(res, lk):
                    print('     λ = %+.6f %+.6fi  Γ_B %+.3e%+.3ei  inside %+.3e%+.3ei  through r_surface %.1e  leak %.2f'
                          % (r['lam'].real, r['lam'].imag, k['GB'].real, k['GB'].imag, k['Tvol'].real, k['Tvol'].imag,
                             abs(k['Ms']), abs(k['Tleak'])/abs(k['GB'])), flush=True)
                check('%s: Γ_B against the torque inside the mantle (max over modes)' % tag,
                      max(abs(k['GB'] - (k['Tvol'] - k['Ms']))/abs(k['GB']) for k in lk), 1e-9)
                if f == 0:
                    check('%s: no stress through r_surface (max over modes)' % tag,
                          max(abs(k['Ms'])/abs(k['GB']) for k in lk), 1e-10)
                kt = lk[res.index(top)]
                check('%s: the leaked field\'s share of the torque > 0.1' % tag, 0.1/(abs(kt['Tleak'])/abs(kt['GB'])), 1)

print('3. thin resolved layers against the thin layer')
for f in (0.0, 0.1):
    def modes(d):
        return [max(kw.solve(d, itau=itau, nev=4, **mantle(f)), key=turns) for itau in targets(f)]
    ref = modes(kw.setup(os.path.join(W, 'thin_twa_%g' % f), Ek=1e-4, f=f, N=128, lmax=lmax_for(128), ncpus=NP, **MAG, **TWA))
    dl, dg = [], []
    for h in (0.02, 0.01, 0.005):
        rs = modes(kw.setup(os.path.join(W, 'thin_res_%g_%g' % (f, h)), Ek=1e-4, f=f, N=128, lmax=lmax_for(128), ncpus=NP,
                            **MAG, mantle='resolved', r_surface=1 + h, N_mantle=16, sigma_mantle=C1/h))
        dl.append([abs(r['lam'] - p['lam'])/abs(p['lam']) for r, p in zip(rs, ref)])
        dg.append([abs(r['magnetic']/r['Mc'] - p['magnetic']/p['Mc'])/abs(p['magnetic']/p['Mc']) for r, p in zip(rs, ref)])
    for k, mode in enumerate(('FCN', 'CW')):
        for what, dd in (('λ', dl), ('Γ_B/Mc', dg)):
            v = [x[k] for x in dd]
            rat = [v[1]/v[0], v[2]/v[1]]
            print('     f %g %s %s: differences %s' % (f, mode, what, ' '.join('%.2e' % x for x in v)), flush=True)
            check('f %g, %s, %s: halving h halves the difference (ratios %.2f, %.2f)' % (f, mode, what, *rat),
                  max(abs(x - 0.5) for x in rat), 0.1)

print('4. an insulating mantle')
try:
    kw.setup(os.path.join(W, 'insulator'), Ek=1e-4, f=0.1, N=32, lmax=lmax_for(32), ncpus=NP, wobble=1, **MAG)
    check('the leaking B0_tor with an insulating mantle stops', 1.0, 0.5)
except SystemExit as ex:
    check('the leaking B0_tor with an insulating mantle stops', 0.0 if 'insulating mantle' in str(ex) else 1.0, 0.5)

print('\nwobble_leak_check: ' + ('all checks passed' if not fails else 'FAILED: ' + '; '.join(fails)))
sys.exit(1 if fails else 0)
