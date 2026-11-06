#!/usr/bin/env python3
'''
usage: twa_check.py [WORKDIR]   (default: tests/twa/runs)

Thin conducting walls (mantle, innercore = 'TWA'; assemble.bc_b_twa, bc_b_slip) against exact results, against the
conductive_IC branch and against the energy budgets. Exit code 1 if a check fails.

1. Free decay without the flow (hydro = 0). In a fluid of uniform diffusivity each mode is j_l(kr), y_l(kr) and decays at
   the rate MagnetD k², with k set by the wall conditions. With F = r f, G = r g and D F = -k² F there, a thin wall gives
       outer:  (1/mu + l c) F' + l F/r - (c1/mu) r k² F + (c mu l/(p r)) ((l+1)/p - 1) F = 0,          G + c1 r G' = 0
       inner:  (1/mu + (l+1) c) F' - (l+1) F/r + (c1/mu) r k² F - (c mu (l+1)/(p r)) (l/p + 1) F = 0,   G - c1 r G' = 0
   (the p terms vanish for p = muw = np.inf), and an insulating wall the same with c = c1 = 0, mu = 1. Every rate must
   agree to 1e-10, resid𝐛 must close to 1e-10, and resid𝐄 too where c = 0 (with c != 0 the thin-wall budget holds only
   to the order of the approximation).
2. With the flow, against the eigenvalues of conductive_IC 2d84351 for the same runs ('Boussinesq', N 48, lmax 20 or 19,
   2 ranks): no-slip walls, fluid slipping along both walls with a field of degree 2, mu != 1, a full sphere; 1e-10.
3. A thin wall with c = c1 = 0 and mu = 1 gives the eigenvalues of an insulating wall (1e-12).
4. A rigid rotation offset: with diff_rot_type 'rigid' (ΔΩ = a) the walls turn with the fluid, and every eigenvalue is
   that of Gaspard (1 + a) shifted by -i m a, while resid𝐄 and εTWA do not change (the same mode in another frame).
5. resid𝐄 closes (1e-8) with walls that turn at different rates (diff_rot_type 'shellular_boussinesq', thin walls at both
   boundaries) and with a fluid slipping along them (c = 0).
'''
import os, sys
import numpy as np
from scipy.special import spherical_jn as J, spherical_yn as Y
from scipy.optimize import brentq

T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(T, '..', 'tidal'))
import mk_run   # noqa: E402  (the run builder of tests/tidal: 'Boussinesq', ricb 0.35, NCPUS ranks)

W = sys.argv[1] if len(sys.argv) > 1 else os.path.join(T, 'runs')
os.makedirs(W, exist_ok=True)
RI = 0.35
LMAX_LINE = 'par.lmax = par.ellmax(par.ncpus, max(g, 4.*par.ncpus/par.N), par.m, par.N)'
assert LMAX_LINE in mk_run.T


def build(name, extra, lmax=None):
    '''mk_run.build in W/name; with lmax given, that lmax on 2 ranks (1 if NCPUS = 1), as the conductive_IC runs'''
    d = os.path.join(W, name)
    if lmax is None:
        mk_run.build(d, extra)
        return d
    T0, n0 = mk_run.T, mk_run.NCPUS
    mk_run.T, mk_run.NCPUS = T0.replace(LMAX_LINE, 'par.lmax = %d' % lmax), min(n0, 2)
    try:
        mk_run.build(d, extra)
    finally:
        mk_run.T, mk_run.NCPUS = T0, n0
    return d

def eig(d):
    e = np.loadtxt(os.path.join(d, 'eigenvalues0.dat')).reshape(-1, 2)
    return e[:, 0] + 1j*e[:, 1]

def mag(d):
    '''magnetic.dat: resid𝐛 (column 4), resid𝐄 (11), εTWA of the outer and the inner wall (15, 16)'''
    a = np.loadtxt(os.path.join(d, 'magnetic.dat'), ndmin=2)
    return a[:, 3], a[:, 10], np.maximum(a[:, 14], a[:, 15])

def top(x):
    '''the largest value of x, nan if all are nan (resid𝐄 where it does not apply)'''
    x = np.asarray(x, dtype=float)
    return np.nan if np.all(np.isnan(x)) else np.nanmax(x)

def closest(lam, ref):
    '''the largest relative distance from a value of ref to the nearest of lam'''
    return max(np.min(abs(lam - z))/abs(z) for z in ref)


# 1. free decay: the exact rates
def roots(f, n, kmax=40.):
    '''the first n positive roots of f(k), bracketed on a fine grid'''
    ks = np.linspace(0.05, kmax, 40000)
    v = np.array([f(k) for k in ks])
    out = []
    for a, b, fa, fb in zip(ks[:-1], ks[1:], v[:-1], v[1:]):
        if np.sign(fa) != np.sign(fb):
            r = brentq(f, a, b, xtol=1e-15, rtol=1e-15)
            if abs(f(r)) < 1e-8*max(abs(fa), abs(fb)):   # a root, not a pole of y_l
                out.append(r)
        if len(out) == n:
            break
    return np.array(out)

def wall_row(Z, kind, side, l, k, w):
    '''the condition of a wall w = (c, c1, p, mu) at r = 1 (side 'o') or ri ('i') on the function Z(l, k r)'''
    c, c1, p, mu = w
    r = 1.0 if side == 'o' else RI
    z, dz = Z(l, k*r), k*Z(l, k*r, True)
    F, F1 = r*z, z + r*dz                     # F = r f and F' (G and G' for the toroidal field)
    if kind == 't':
        return (F + (1 if side == 'o' else -1)*c1*r*F1)/r
    if side == 'o':
        e = 0. if np.isinf(p) else c*mu*l/(p*r)*((l + 1)/p - 1)
        return (1/mu + l*c)*F1 + l*F/r - (c1/mu)*r*k**2*F + e*F
    e = 0. if np.isinf(p) else c*mu*(l + 1)/(p*r)*(l/p + 1)
    return (1/mu + (l + 1)*c)*F1 - (l + 1)*F/r + (c1/mu)*r*k**2*F - e*F

def rates(kind, l, n, o, i=None):
    '''the first n decay rates k² of the degree l, with the walls o (outer) and i (inner; None: a full sphere)'''
    if i is None:
        return roots(lambda k: wall_row(J, kind, 'o', l, k, o), n)**2
    return roots(lambda k: wall_row(J, kind, 'o', l, k, o)*wall_row(Y, kind, 'i', l, k, i)
                           - wall_row(Y, kind, 'o', l, k, o)*wall_row(J, kind, 'i', l, k, i), n)**2

INS = (0., 0., 1., 1.)
def twa(prefix, w):
    c, c1, p, _ = w
    return ("; par.%s = 'TWA'; par.c_%s = %r; par.c1_%s = %r; par.muw_%s = %s"
            % ('mantle' if prefix == 'cmb' else 'innercore', prefix, c, prefix, c1, prefix, 'np.inf' if np.isinf(p) else repr(p)))

DEC = ("par.hydro = 0; par.magnetic = 1; par.B0 = 'axial'; par.cnorm = 1; par.Hendrik = 1.0; par.MagnetD = 1.0; par.N = 48; "
       "g = 0.34; par.m = 0; par.symm = 1; par.rtau = -10.0; par.itau = 0.0; par.nev = 6")
# name, outer wall, inner wall (None: insulating, or a full sphere with ricb = 0), mu; m = 0, symm = 1: the poloidal l = 1
# (two modes) and 3, the toroidal l = 2
decay = [
    ('decay_cmb_c1',      (0.0, 0.1, 1., 1.),      None,                  1.0, True),
    ('decay_cmb_p1',      (0.05, 0.1, 1., 1.),     None,                  1.0, True),
    ('decay_cmb_pinf',    (0.05, 0.1, np.inf, 1.), None,                  1.0, True),
    ('decay_icb_c1',      None,                    (0.0, 0.2, 1., 1.),    1.0, True),
    ('decay_both_mu',     (0.05, 0.1, 2., 1.5),    (0.03, 0.2, 1., 1.5),  1.5, True),
    ('decay_sphere_p3',   (0.05, 0.1, 3., 1.),     None,                  1.0, False),
]

fail = 0
def report(name, ok, text):
    global fail
    fail |= not ok
    print('%-22s %s  (%s)' % (name, 'ok' if ok else 'FAIL', text))

for name, o, i, mu, shell in decay:
    oo = INS[:3] + (mu,) if o is None else o
    ii = INS[:3] + (mu,) if i is None else i
    extra = DEC + ('' if shell else '; par.ricb = 0') + ('' if o is None else twa('cmb', o)) + ('' if i is None else twa('icb', i))
    if mu != 1:
        extra += '; par.mu = %r' % mu
    d = build(name, extra)
    ins = ii if shell else None
    ref = -np.r_[rates('p', 1, 2, oo, ins), rates('p', 3, 1, oo, ins), rates('t', 2, 1, oo, ins)]
    rb, rE, _ = mag(d)
    err = closest(eig(d), ref)
    exactE = (o is None or o[0] == 0) and (i is None or i[0] == 0) and mu == 1
    ok = err < 1e-10 and top(rb) < 1e-10 and (not exactE or top(rE) < 1e-10)
    report(name, ok, 'rates %.1e, resid𝐛 %.1e, resid𝐄 %.1e%s' % (err, top(rb), top(rE),
                                                             '' if exactE else ' (not tested: c != 0 or mu != 1)'))

# 2. against conductive_IC 2d84351 (the same runs: Ek 1e-3, Em 1e-3, Le² 1e-3, cnorm 'mag_energy'; first two eigenvalues)
FLOW = ("par.magnetic = 1; par.cnorm = 'mag_energy'; par.Hendrik = 1e-3; par.MagnetD = 1e-3; par.ViscosD = 1e-3; par.N = 48; "
        "par.rtau = 0.0; par.itau = 1.0; par.nev = 4")
cic = [
    ('cic_noslip_both', 20, "par.B0 = 'G21 dipole'; par.m = 1; par.symm = 1" + twa('cmb', (0.02, 0.03, np.inf, 1))
                            + twa('icb', (0.01, 0.05, 2.0, 1)),
     [-2.187282981218494349e-01 + 1.190482032963313674e+00j, -2.849285615204567024e-01 + 8.889125534713646637e-01j]),
    ('cic_slip_degree2', 19, "par.B0 = 'Luo_S2'; par.B0_l = 2; par.m = 0; par.symm = -1; par.bco = 0; par.bci = 0"
                             + twa('cmb', (0.02, 0.03, 4.0, 1)) + twa('icb', (0.01, 0.05, 1.0, 1)),
     [-1.943155510068610703e-01 + 9.491992619479689974e-01j, -1.546971875160249033e-01 + 1.137944626318003527e+00j]),
    ('cic_mu', 20, "par.B0 = 'G21 dipole'; par.m = 1; par.symm = -1; par.mu = 1.3" + twa('cmb', (0.02, 0.01, 1.0, 1))
                   + twa('icb', (0.01, 0.02, 1.0, 1)),
     [-1.037140936137195824e-01 + 1.010304652619480104e+00j, -3.263570452534672595e-01 + 1.050456853209116748e+00j]),
    ('cic_sphere_slip', 20, "par.B0 = 'G21 dipole'; par.ricb = 0; par.m = 1; par.symm = 1; par.bco = 0"
                            + twa('cmb', (0.0, 0.05, 1.0, 1)),
     [-7.210833710972244792e-02 + 1.050987281389437866e+00j, -1.395775162764246458e-01 + 7.812180598135879395e-01j]),
    ('cic_sphere_slip_anti', 20, "par.B0 = 'G21 dipole'; par.ricb = 0; par.m = 1; par.symm = -1; par.bco = 0"
                                 + twa('cmb', (0.0, 0.05, 1.0, 1)),
     [-5.257002578067904511e-03 + 1.003040199676247690e+00j, -1.861329373942501864e-01 + 1.037317971452621501e+00j]),
]
for name, lmax, extra, ref in cic:
    d = build(name, FLOW + '; ' + extra, lmax)
    err = closest(eig(d), np.array(ref))
    rb, rE, eps = mag(d)
    report(name, err < 1e-10, 'eigenvalues %.1e from conductive_IC; resid𝐛 %.1e, resid𝐄 %.1e, εTWA %.1e'
           % (err, top(rb), top(rE), eps.max()))

# 3. c = c1 = 0, mu = 1: the insulating walls (with a fluid slipping along them)
SLIP = FLOW + "; g = 0.42; par.B0 = 'Luo_S2'; par.B0_l = 2; par.m = 1; par.symm = 1; par.bco = 0; par.bci = 0; par.nev = 3"
l0 = eig(build('insulating', SLIP))
l1 = eig(build('twa_zero', SLIP + twa('cmb', (0., 0., 1., 1)) + twa('icb', (0., 0., 1., 1))))
report('twa_zero', closest(l1, l0[:2]) < 1e-12, 'eigenvalues %.1e from the insulating walls' % closest(l1, l0[:2]))

# 4. a rigid rotation offset a = 0.3 (thin walls with c != 0 at both boundaries, the fluid slipping along them)
RIG = (FLOW + "; g = 0.42; par.B0 = 'G21 dipole'; par.m = 1; par.symm = 1; par.bco = 0; par.bci = 0; par.nev = 3"
       + twa('cmb', (0.01, 0.03, 2.0, 1)) + twa('icb', (0.02, 0.05, 1.0, 1)))
da = build('rigid_offset', RIG + "; par.diff_rot = 1; par.diff_rot_type = 'rigid'; par.diff_rot_ampl = 0.3; par.itau = 0.7")
db = build('rigid_frame', RIG + "; par.Gaspard = 1.3")
la, lb = eig(da), eig(db) - 0.3j
err = closest(la, lb[:2])
_, ea, ta = mag(da)
_, eb, tb = mag(db)
k = [int(np.argmin(abs(la - z))) for z in lb[:2]]
dE = max(abs(ea[k] - eb[:2])/eb[:2]); dT = max(abs(ta[k] - tb[:2])/tb[:2])
report('rigid_offset', err < 1e-12 and dE < 1e-6 and dT < 1e-6,
       'eigenvalues %.1e, resid𝐄 %.1e and εTWA %.1e from those of the turning frame' % (err, dE, dT))

# 5. walls that turn at different rates (ΔΩ = a (1 - r²)/2), the fluid slipping along them, c = 0
d = build('sheared_walls', FLOW + "; g = 0.42; par.B0 = 'G21 dipole'; par.m = 1; par.symm = 1; par.bco = 0; par.bci = 0; "
          "par.nev = 3; par.diff_rot = 1; par.diff_rot_type = 'shellular_boussinesq'; par.diff_rot_ampl = 0.4"
          + twa('cmb', (0.0, 0.03, 1.0, 1)) + twa('icb', (0.0, 0.05, 1.0, 1)))
rb, rE, _ = mag(d)
report('sheared_walls', top(rE) < 1e-8, 'resid𝐄 %.1e, resid𝐛 %.1e' % (top(rE), top(rb)))

print('twa_check: ' + ('all cases ok' if not fail else 'FAILED'))
sys.exit(fail)
