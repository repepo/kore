#!/usr/bin/env python3
'''
NEW (2026-11-10): cnorm = 'rms_Bs', the volume rms of B_s = B_r sinθ + B_θ cosθ over the fluid is 1 (background_field.rms_Bs).
usage: rms_bs_check.py [WORKDIR]   (default tests/magnetic/runs/rms_bs; seconds, no solves)
For each case below it builds WORKDIR/NAME/bin (the repo's bin/ and a parameters.py) and, in a fresh python, takes cnorm from
background_field and integrates B_s² of the field kore uses (cnorm h, B0_ic and B0_cmb terms included) on a grid of its own
(Gauss-Legendre in θ itself, weight sin θ, and in r, other node counts than rms_Bs): rms must be 1 to 1e-12. The full-sphere
Luo & Jackson fields must give conductive_IC's constants (params_default.py there, 'rms_Bs=1 (volume rms of B_s)') to 1e-13.
With an inner core it also checks that h is the field kore uses: r h' = l h at ricb (B0_ic 'potential') and the thin-wall
condition (1/mu + l c)(r h)' + (l + e) h = 0 at r = 1 (B0_cmb 'wall'), and that 'axial' is refused (B_s = 0).
'''
import os, sys, shutil, subprocess, textwrap
BIN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bin'))
WORK = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'rms_bs'))
T = textwrap.dedent('''
from params_default import default_params
import numpy as np
par = default_params()
par.timescale = 'rotation'
par.Ek = 1e-3; par.Em = 1e-3; par.Le = np.sqrt(1e-3)
par.set_scales()
par.structure = 'Boussinesq'
par.ricb = 0.35; par.bci = 1; par.bco = 1; par.m = 1; par.symm = 1
par.magnetic = 1; par.cnorm = 'rms_Bs'; par.innercore = 'insulator'; par.mantle = 'insulator'
par.N = 96; par.lmax = 24; par.ncpus = 2
{extra}
''')
TWA = "par.mantle = 'TWA'; par.c_cmb = 3e-5; par.c1_cmb = 3e-5; par.muw_cmb = 1.0; par.B0_cmb = 'wall'"
CASES = [   # name, parameters, the cnorm expected (None: not checked)
    ('s2_fs',      "par.ricb = 0; par.B0 = 'Luo_S2'", 0.0158567582314039),
    ('s1_fs',      "par.ricb = 0; par.B0 = 'Luo_S1'; par.symm = -1", 0.6972166887783963),
    ('s2_twa',     "par.B0 = 'Luo_S2'; par.B0_ic = 'potential'; par.innercore = 'conducting, Chebys'; " + TWA, None),
    ('s2_twa_inf', "par.B0 = 'Luo_S2'; par.innercore = 'conducting, Chebys'; " + TWA + "; par.c_cmb = 0.05; par.muw_cmb = np.inf", None),
    ('s1_frozen',  "par.B0 = 'Luo_S1'; par.B0_ic = 'frozen currents'; par.symm = -1", None),
    ('fdm2',       "par.B0 = 'FDM'; par.B0_l = 2", None),
    ('dipole',     "par.B0 = 'dipole'; par.symm = -1", None),
    ('confined',   "par.B0 = 'confined'; par.B0_l = 1; par.B0_conf_r1 = 0.5; par.B0_conf_r2 = 0.9; par.symm = -1", None),
    ('ferraro',    "par.diff_rot = 1; par.diff_rot_type = 'user'; par.diff_rot_profiles = {0: '0.05*r**2', 2: '-0.1*r**2', 4: '-0.03*r**4'}; par.B0 = 'ferraro'; par.B0_ferraro_G = 'w'; par.B0_ferraro_nmax = 7; par.symm = -1", None),
    ('general13',  "par.B0 = 'general'; par.B0_profiles = {1: 'r*(1 - r**2)', 3: '0.3*r**3*(1 - r)'}; par.symm = -1", None),
]
CHILD = textwrap.dedent('''
import sys, numpy as np
sys.path.insert(0, 'bin')
import background_field as bf
from parameters import par
c = bf.cnorm
if par.B0 == 'ferraro':   # its cnorm is computed on first use
    c = bf.ferraro_cnorm(); degs = bf.ferraro_degrees(); hf = bf.ferraro_profile
elif par.B0 == 'general':
    degs = [n for n in par.B0_profiles if isinstance(n, int)]; hf = lambda n, r, d: bf.gen_profile('h', n, r, d)
else:
    degs = [bf.B0_l]; hf = lambda n, r, d: bf.h(r, d)
r_lo, r_hi = (max(par.B0_conf_r1, par.ricb), par.B0_conf_r2) if par.B0 == 'confined' else (par.ricb, 1.0)
xr, wr = np.polynomial.legendre.leggauss(301); r = 0.5*(r_hi - r_lo)*(xr + 1) + r_lo; wr = 0.5*(r_hi - r_lo)*wr
xt, wt = np.polynomial.legendre.leggauss(64); th = 0.5*np.pi*(xt + 1); wt = 0.5*np.pi*wt
R, TH = np.meshgrid(r, th, indexing='ij')
Br = 0.; Bt = 0.
for n in degs:   # B_r = c L h P_n/r, B_θ = c (h' + h/r) dP_n/dθ
    P = np.polynomial.legendre.Legendre.basis(n)
    Br = Br + c*n*(n + 1)*hf(n, R, 0)*P(np.cos(TH))/R
    Bt = Bt - c*(hf(n, R, 1) + hf(n, R, 0)/R)*P.deriv()(np.cos(TH))*np.sin(TH)
Bs = Br*np.sin(TH) + Bt*np.cos(TH)
rms = np.sqrt(2*np.pi*np.sum(np.outer(wr*r**2, wt)*np.sin(TH)*Bs**2)/(4*np.pi/3*(1 - par.ricb**3)))
out = [c, rms - 1]
if par.ricb > 0 and par.B0 in ('Luo_S1', 'Luo_S2', 'G21 dipole') and par.B0_ic == 'potential':
    ri = np.array([par.ricb]); l = bf.B0_l
    out.append(abs(ri[0]*bf.h(ri, 1)[0] - l*bf.h(ri, 0)[0])/abs(bf.h(ri, 0)[0]))
if par.mantle == 'TWA' and par.B0_cmb == 'wall':
    one = np.array([1.0]); l = bf.B0_l; k = 1/par.mu + l*par.c_cmb; p = par.muw_cmb
    e = 0.0 if np.isinf(p) else par.c_cmb*par.mu*l*((l + 1)/p - 1)/p
    h0, h1 = bf.h(one, 0)[0], bf.h(one, 1)[0]
    out.append(abs(k*(h0 + h1) + (l + e)*h0)/abs(h0))
    out.append(bf.alpha_cmb)
print(' '.join('%.17g' % v for v in out))
''')

def build(name, extra):
    d = os.path.join(WORK, name)
    if os.path.isdir(d): shutil.rmtree(d)
    os.makedirs(d + '/bin')
    for f in os.listdir(BIN):
        if os.path.isfile(BIN + '/' + f) and f != 'parameters.py': shutil.copy(BIN + '/' + f, d + '/bin/' + f)
    open(d + '/bin/parameters.py', 'w').write(T.format(extra=extra))
    return d

fails = 0
for name, extra, want in CASES:
    d = build(name, extra)
    p = subprocess.run([sys.executable, '-c', CHILD], cwd=d, capture_output=True, text=True)
    if p.returncode:
        print('%-11s FAIL (error)\n%s' % (name, p.stderr[-2000:])); fails += 1; continue
    v = [float(x) for x in p.stdout.split()]
    c, drms, more = v[0], v[1], v[2:]
    ok = abs(drms) < 1e-12
    msg = 'cnorm %.16e  rms-1 %+.1e' % (c, drms)
    if 'TWA' in extra:
        ok = ok and more[-2] < 1e-12; msg += '  wall cond %.1e  alpha_cmb %+.3e' % (more[-2], more[-1])
        more = more[:-2]
    if more:   # B0_ic 'potential' condition at ricb
        ok = ok and more[0] < 1e-12; msg += '  icb cond %.1e' % more[0]
    if want is not None:
        ok = ok and abs(c/want - 1) < 1e-13; msg += '  vs conductive_IC %+.1e' % (c/want - 1)
    print('%-11s %s %s' % (name, 'PASS' if ok else 'FAIL', msg)); fails += not ok

# fields with B_s = 0 are refused by background.check_magnetic: 'axial', and 'ferraro' of a rotation on cylinders (ψ(s), a
# uniform axial field here)
for name, extra, key in (('axial', "par.B0 = 'axial'; par.symm = -1", 'B_s = 0'),
                         ('ferraro_cyl', "par.diff_rot = 1; par.diff_rot_type = 'user'; par.diff_rot_ampl = 0.2; par.diff_rot_profiles = {0: '(2/3)*r**2 - (8/15)*r**4', 2: '-(2/3)*r**2 + (16/21)*r**4', 4: '-(8/35)*r**4'}; par.B0 = 'ferraro'; par.B0_ferraro_G = 'w'; par.B0_ferraro_nmax = 7; par.symm = -1", 'B_s is zero')):
    d = build(name, extra)
    p = subprocess.run([sys.executable, '-c', "import sys; sys.path.insert(0, 'bin'); import background; background.check_magnetic()"],
                       cwd=d, capture_output=True, text=True)
    ok = p.returncode != 0 and key in p.stderr
    print('%-11s %s %s' % (name, 'PASS' if ok else 'FAIL', (p.stderr.strip().splitlines() or [''])[-1]))
    fails += not ok
print('rms_bs_check: %s (%d failed)' % ('PASS' if fails == 0 else 'FAIL', fails))
sys.exit(1 if fails else 0)
