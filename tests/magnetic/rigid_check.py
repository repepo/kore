#!/usr/bin/env python3
'''
rigid_check.py [c]: the exact rigid-rotation test of the magnetic field with a differential rotation (NEW 2026-10-30).

WHY. A uniform ΔΩ = c is a rigid rotation: the problem is the one without ΔΩ in a frame that rotates c faster. Kore's
operators for ΔΩ (the force on the flow, diff_rot.momentum_block, and the induction ∇×(U0×b), diff_rot.induction_block, with
the co-rotating conducting inner core) then have to give exactly
    λ(ΔΩ = c, Gaspard 1) = λ(no ΔΩ, Gaspard 1 + c) - i m c
for every eigenvalue, with the same ViscosD, MagnetD, Hendrik (set_scales runs with Gaspard 1 in both). It tests the new
induction term exactly, its boundary terms included, for each kind of background field.
HOW. For each case, two runs with tests/magnetic/mk_run.py ('Wu 2005' defaults, ricb 0.35, Ek = Em = 1e-3): A with
diff_rot_type 'rigid' and diff_rot_ampl = c, B with Gaspard = 1 + c; all eigenvalues of A against those of B shifted by
-i m c (nearest), relative difference below 1e-11.
usage: python3 tests/magnetic/rigid_check.py [c]    (c defaults to 0.13; run in a scratch directory, it builds runs there)
'''
import sys, os, subprocess
import numpy as np
T = os.path.dirname(os.path.abspath(__file__))
c = float(sys.argv[1]) if len(sys.argv) > 1 else 0.13
CASES = {   # extra parameters of mk_run.py
    'rg_axial':  "par.N = 64",
    'rg_luo2':   "par.N = 64; par.B0 = 'Luo_S2'; par.symm = 1; par.itau = 0.6",
    'rg_malkus': "par.N = 64; par.B0 = None; par.B0_tor = 'Malkus'",
    'rg_cic':    "par.N = 64; par.innercore = 'conducting, Chebys'",
    'rg_fs':     "par.N = 96; par.ricb = 0; par.B0 = 'Luo_S2'; par.symm = 1; par.itau = 0.6",
    'rg_alg':    "par.N = 64; par.B0 = 'general'; par.B0_profiles = {1: 'r/6 - r**3/10', 3: '0.3*r**3*(1 - r)'}",
}
worst = 0.
for name, extra in CASES.items():
    m = 1
    a = extra + "; par.diff_rot = 1; par.diff_rot_type = 'rigid'; par.diff_rot_ampl = %r; par.itau = par.itau - %r" % (c, m*c)
    b = extra + "; par.Gaspard = %r" % (1 + c)
    lam = []
    for tag, ex in (('A', a), ('B', b)):
        d = '%s_%s' % (name, tag)
        r = subprocess.run([sys.executable, T + '/mk_run.py', d, ex, 'solve'], capture_output=True, text=True)
        if r.returncode:
            print(name, 'FAILED', r.stdout[-500:]); sys.exit(1)
        e = np.loadtxt(d + '/eigenvalues0.dat').reshape(-1, 2)
        lam.append(e[:, 0] + 1j*e[:, 1])
    la, lb = lam[0], lam[1] - 1j*m*c
    err = max(min(abs(x - lb))/abs(x) for x in la)
    worst = max(worst, err)
    print('%-10s worst |λ_A - (λ_B - i m c)|/|λ| = %.1e over %d eigenvalues' % (name, err, len(la)))
print('rigid_check.py: worst %.1e -> %s' % (worst, 'ok' if worst < 1e-11 else 'FAIL'))
sys.exit(0 if worst < 1e-11 else 1)
