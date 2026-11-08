#!/usr/bin/env python3
'''
usage: rows_check.py [WORKDIR]   (default: tests/forcing/runs)

The boundary forcings set the values of the flow at the walls through the right-hand side of the boundary-condition rows
of bc_u_spherical (bin/assemble.py: per degree, the rows P = 0 and the tangential condition at the outer wall, then the
same at the inner wall; for T the outer row, then the inner one). This check solves one small forced run per case and
reads the wall values back from the solution: the imposed P or T, and the tangential condition, which must stay
homogeneous (zero tangential velocity at a no-slip wall, (rP)'/r + P (ln ρ0)' = 0, or zero tangential stress at a
stress-free one). The cases: forcing 6 (no-slip and stress-free inner wall), 7 (m = 0 and m = 2), 9 (in an
incompressible shell, with a variable density, and in a full sphere), 10, and NEW (2026-11-08) 12 (both walls turning,
the inner one under a stress-free outer wall at rest, whose toroidal row stays rT' - T = 0, and a full sphere). Before
2026-11-06 forcings 6, 7 (m = 2) and 9 put the inner wall's value on the outer wall's rows.

Exit code 1 if a case is off by more than 1e-10 of its amplitude.
'''
import os, sys
import numpy as np
import numpy.polynomial.chebyshev as ch

T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(T, '..', 'tidal'))
import mk_run   # noqa: E402  (the run builder of tests/tidal)

W = sys.argv[1] if len(sys.argv) > 1 else os.path.join(T, 'runs')
os.makedirs(W, exist_ok=True)

wf = 0.37
hft = 0.0025   # the inner core's flattening in forcing 6
WU = "par.structure = 'Wu 2005'; par.r_out = 0.99"
cases = [
    # name, changes, the walls' expected values: list of (kind, l, wall, value) with kind 'P' (P), 'T' (T),
    # 'ns' (no-slip: (rP)'/r + P (ln ρ0)' = 0), 'sf' (stress-free consoidal row), each at wall 'o' (outer) or 'i' (inner)
    ('f6_noslip', 'par.forcing = 6; par.m = 1; par.symm = -1; par.bci = 1; par.forcing_amplitude_cmb = 0.; '
                  'par.forcing_amplitude_icb = 1.',
     [('P', 2, 'i', 0.35/6*1j*(2/3)*np.sqrt(6)*hft*0.35), ('ns', 2, 'i', 0), ('P', 2, 'o', 0), ('ns', 2, 'o', 0)]),
    ('f6_stressfree', 'par.forcing = 6; par.m = 1; par.symm = -1; par.bci = 0; par.forcing_amplitude_cmb = 0.; '
                      'par.forcing_amplitude_icb = 1.',
     [('P', 2, 'i', 0.35/6*1j*(2/3)*np.sqrt(6)*hft*0.35), ('sf', 2, 'i', 0), ('P', 2, 'o', 0)]),
    ('f7_m0', 'par.forcing = 7; par.m = 0; par.symm = 1; par.forcing_amplitude_cmb = 0.5; par.forcing_amplitude_icb = 1.',
     [('T', 1, 'o', 1j*wf*0.5/2), ('T', 1, 'i', 1j*wf*1.0*0.35/2)]),
    ('f7_m2', 'par.forcing = 7; par.m = 2; par.symm = 1; par.forcing_amplitude_cmb = 0.5; par.forcing_amplitude_icb = 1.',
     [('P', 2, 'o', -wf/6*0.5), ('ns', 2, 'o', 0), ('P', 2, 'i', -wf/6*1.0*0.35**2), ('ns', 2, 'i', 0)]),
    ('f9', 'par.forcing = 9; par.m = 2; par.symm = 1; par.forcing_amplitude_cmb = 0.5; par.forcing_amplitude_icb = 1.',
     [('P', 2, 'o', 0.5/6), ('ns', 2, 'o', 0), ('P', 2, 'i', 0.35/6), ('ns', 2, 'i', 0)]),
    ('f9_wu', 'par.forcing = 9; par.m = 2; par.symm = 1; par.forcing_amplitude_cmb = 0.5; par.forcing_amplitude_icb = 1.; '
              + WU,
     [('P', 2, 'o', 0.5/6), ('ns', 2, 'o', 0), ('P', 2, 'i', 0.35/6), ('ns', 2, 'i', 0)]),
    ('f9_sphere', 'par.forcing = 9; par.m = 2; par.symm = 1; par.ricb = 0; par.forcing_amplitude_cmb = 0.5',
     [('P', 2, 'o', 0.5/6), ('ns', 2, 'o', 0)]),
    ('f10', 'par.forcing = 10; par.m = 3; par.symm = 1',
     [('P', 3, 'o', 1/12), ('ns', 3, 'o', 0), ('P', 3, 'i', 0), ('ns', 3, 'i', 0)]),
    # NEW (2026-11-08): forcing 12, T_1 = ΔΩ r/2 at a turning wall ('tsf': the toroidal stress-free row rT' - T = 0)
    ('f12', 'par.forcing = 12; par.m = 0; par.symm = 1; par.forcing_frequency = 0.; par.forcing_amplitude_cmb = 0.5; '
            'par.forcing_amplitude_icb = 1.',
     [('T', 1, 'o', 0.5/2), ('T', 1, 'i', 1.0*0.35/2)]),
    ('f12_slip', 'par.forcing = 12; par.m = 0; par.symm = 1; par.forcing_frequency = 0.; par.bco = 0; '
                 'par.forcing_amplitude_cmb = 0.; par.forcing_amplitude_icb = 1.',
     [('T', 1, 'i', 1.0*0.35/2), ('tsf', 1, 'o', 0)]),
    ('f12_sphere', 'par.forcing = 12; par.m = 0; par.symm = 1; par.forcing_frequency = 0.; par.ricb = 0; '
                   'par.forcing_amplitude_cmb = 0.5',
     [('T', 1, 'o', 0.5/2)]),
]

check = r'''
import sys; sys.path.insert(0, 'bin')
import numpy as np
import numpy.polynomial.chebyshev as ch
from parameters import par
import utils as ut
import radial_profiles as rap
import utils4pp as upp
try:
    x = np.load('flow.npy')
except OSError:
    x = np.loadtxt('real_flow.field') + 1j*np.loadtxt('imag_flow.field')
x = x.reshape(x.shape[0], -1)[:, 0]
Plj, Tlj = upp.expand_reshape_sol(x, par.symm)
lp, lt = ut.ell(par.m, par.lmax, par.symm)[:2]
dom = [par.ricb, 1.] if par.ricb > 0 else [-1., 1.]
def series(kind, l):
    c = Plj[list(lp).index(l)] if kind not in ('T', 'tsf') else Tlj[list(lt).index(l)]
    return ch.Chebyshev(c, domain=dom)
for kind, l, wall, value in eval(sys.argv[1]):
    r = 1. if wall == 'o' else par.ricb
    F = series(kind, l)
    F0, F1, F2 = F(r), F.deriv()(r), F.deriv(2)(r)
    h1, h2 = rap.logrhoX(np.array([r]), 1)[0], rap.logrhoX(np.array([r]), 2)[0]
    L = l*(l + 1)
    got = {'P': F0, 'T': F0, 'ns': F1 + F0*(1/r + h1),
           'sf': r**2*F2 + r**2*h1*F1 + (r**2*h2 + L - 2 - r*h1)*F0, 'tsf': r*F1 - F0}[kind]
    print(kind, l, wall, repr(complex(got)), repr(complex(value)))
'''

fail = 0
for name, extra, expect in cases:
    d = os.path.join(W, name)
    mk_run.build(d, 'par.N = 48; par.forcing_frequency = %r; %s' % (wf, extra))
    import subprocess
    p = subprocess.run([sys.executable, '-c', check, repr(expect)], cwd=d, capture_output=True, text=True)
    if p.returncode:
        print(name, 'FAILED to read the solution:', p.stderr[-2000:]); fail = 1; continue
    scale = max(abs(v) for _, _, _, v in expect)
    worst = 0
    for line in p.stdout.split('\n'):
        if not line.strip(): continue
        kind, l, wall, got, value = line.split(' ', 4)
        got, value = complex(got), complex(value)
        worst = max(worst, abs(got - value)/scale)
    ok = worst < 1e-10
    fail |= not ok
    print('%-14s %s  (largest error %.1e of the amplitude)' % (name, 'ok' if ok else 'FAIL', worst))
print('rows_check: ' + ('all cases ok' if not fail else 'FAILED'))
sys.exit(fail)
