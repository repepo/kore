#!/usr/bin/env python3
'''
usage: mantle_check.py [WORKDIR]   (default: tests/mantle/runs)

The resolved conducting mantle (mantle = 'resolved': operators.magnetic_diffusion of the sections fm, gm,
assemble.bc_b_mantle) against exact results, against the conductive_IC branch, against the thin conducting wall and
against the energy budgets. Exit code 1 if a check fails.

1. Free decay without the flow (hydro = 0). With MagnetD = 1 the field of degree l decays as exp(-k² t): in the fluid it
   is j_l(kr), y_l(kr) (or the solution that meets the inner core), in the layer the solution of
       f'' + 2 f'/r - L f/r² = -k² σ f,      g'' + 2 g'/r - L g/r² - (ln σ)' (r g)'/r = -k² σ g      (L = l(l+1))
   (j_l(k √σ r), y_l(k √σ r) for a uniform σ, otherwise integrated from r_surface with scipy's DOP853 at rtol 1e-13),
   with f, f', g and σ (r g)' continuous at r = 1 (σ is the layer's conductivity relative to the fluid's), and
   r f' + (l+1) f = 0, g = 0 at r_surface (a potential field beyond). Every rate to 1e-10, resid𝐛 and resid𝐄 to 1e-10:
   a uniform layer and an exponential one (the cases of conductive_IC's tests/test_conducting_mantle.py), a full
   sphere, and a conducting inner core under a layer whose σ falls as a power of r.
2. With the flow, against the eigenvalues of conductive_IC 2d84351 for the same runs ('Boussinesq', N 48, lmax 19 to 21,
   2 ranks): no-slip walls, the fluid slipping along both walls with a field of degree 2 and an exponential layer, a
   full sphere with a slipping fluid, a conducting inner core, m = 2 with a thin layer and a slipping fluid; 1e-10.
3. A thin layer (r_surface = 1 + h, σ = c1/h) tends to the thin wall of the same conductance (mantle = 'TWA', c = h, c1)
   at first order in h, with the fluid slipping along it (the u×B0 terms of both): halving h halves the difference.
4. A layer of vanishing conductance (σ = 1e-9) gives the eigenvalues of an insulating mantle (1e-7), with the fluid
   slipping along it.
5. A rigid rotation offset with a variable density ('Wu 2005'): with diff_rot_type 'rigid' (ΔΩ = a) the layer and a
   conducting inner core turn with the fluid, and every eigenvalue is that of Gaspard (1 + a) shifted by -i m a (1e-12),
   while resid𝐄 closes in both frames (1e-8: the induction work of the turning solids is in the budget).
6. resid𝐄 and resid𝐛 close (1e-8) with the layer, a conducting inner core and the fluid turning at different rates
   (diff_rot_type 'user', ΔΩ = a (0.5 + r²)), the fluid slipping along the layer, 'Wu 2005'.
7. just_induction.py and solve_ind.py give the field, in the fluid and in the layer, of a mode whose Lorentz force
   vanishes (Hendrik 1e-30), from its flow and eigenvalue (1e-10).
'''
import os, sys, subprocess
import numpy as np
from scipy.special import spherical_jn as J, spherical_yn as Y
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(T, '..', 'tidal'))
import mk_run   # noqa: E402  (the run builder of tests/tidal: 'Boussinesq', ricb 0.35, NCPUS ranks)
sys.path.append(os.path.join(T, '..', '..', 'bin'))
import field_io   # noqa: E402  (the solution files, in either format)

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
    '''magnetic.dat: resid𝐛 (column 4) and resid𝐄 (11)'''
    a = np.loadtxt(os.path.join(d, 'magnetic.dat'), ndmin=2)
    return a[:, 3], a[:, 10]

def top(x):
    '''the largest value of x, nan if all are nan (resid𝐄 where it does not apply)'''
    x = np.asarray(x, dtype=float)
    return np.nan if np.all(np.isnan(x)) else np.nanmax(x)

def closest(lam, ref):
    '''the largest relative distance from a value of ref to the nearest of lam'''
    return max(np.min(abs(lam - z))/abs(z) for z in ref)

def field(d, name):   # the solutions, from name.npy or the text files
    return field_io.load(name, folder=d)

fail = 0
def report(name, ok, text):
    global fail
    fail |= not ok
    print('%-16s %s  (%s)' % (name, 'ok' if ok else 'FAIL', text))


# 1. free decay: the exact rates
def fluid(kind, l, k, inner):
    '''f and f' at r = 1 (g and g' for kind 't') of the fluid's solution that meets the inner condition: inner None (a full
    sphere), 'ins' (an insulating inner core at RI), or the conductivity s of a conducting inner core (mu = 1)'''
    if inner is None:
        return J(l, k), k*J(l, k, True)
    if inner == 'ins':
        if kind == 'p':   # r f' - l f = 0 at RI (a potential field inside)
            row = lambda Z: k*RI*Z(l, k*RI, True) - l*Z(l, k*RI)
        else:             # g = 0
            row = lambda Z: Z(l, k*RI)
        a, b = row(Y), -row(J)
    else:                 # the inner core's j_l(k √s r): f and f' continuous at RI, g and (r g)'/s continuous
        s = inner
        q = k*np.sqrt(s)
        fi, dfi = J(l, q*RI), q*J(l, q*RI, True)
        M = [[J(l, k*RI), Y(l, k*RI)], [k*J(l, k*RI, True), k*Y(l, k*RI, True)]]
        a, b = np.linalg.solve(M, [fi, dfi] if kind == 'p' else [fi, ((fi + RI*dfi)/s - fi)/RI])
    return a*J(l, k) + b*Y(l, k), k*(a*J(l, k, True) + b*Y(l, k, True))

def layer(kind, l, k, lay):
    '''f and f' at r = 1 (g and g') of the layer's solution that meets the surface condition, lay = (r_surface, σ: a number,
    or functions σ(r) and (ln σ)'(r))'''
    rs, sig, dls = lay
    if not callable(sig):   # uniform: j_l and y_l of k √σ r
        q = k*np.sqrt(sig)
        if kind == 'p':     # r f' + (l + 1) f = 0 at r_surface
            row = lambda Z: q*rs*Z(l, q*rs, True) + (l + 1)*Z(l, q*rs)
        else:               # g = 0
            row = lambda Z: Z(l, q*rs)
        a, b = row(Y), -row(J)
        return a*J(l, q) + b*Y(l, q), q*(a*J(l, q, True) + b*Y(l, q, True))
    L = l*(l + 1)
    def rhs(r, y):
        f, df = y
        ddf = -2*df/r + L*f/r**2 - k**2*sig(r)*f
        if kind == 't':
            ddf += dls(r)*(df + f/r)
        return [df, ddf]
    y = solve_ivp(rhs, (rs, 1.0), [1.0, -(l + 1)/rs] if kind == 'p' else [0.0, 1.0], method='DOP853', rtol=1e-13,
                  atol=1e-16).y
    return y[0, -1], y[1, -1]

def matching(kind, l, k, inner, lay):
    '''the determinant of the conditions at r = 1: f, f' continuous; g and σ(1) (r g)' continuous, (r g)' = g + g' there'''
    f, df = fluid(kind, l, k, inner)
    fm, dfm = layer(kind, l, k, lay)
    if kind == 'p':
        return f*dfm - fm*df
    s1 = lay[1](1.0) if callable(lay[1]) else lay[1]
    return f*(fm + dfm) - s1*fm*(f + df)

def rates(kind, l, n, inner, lay, dk=0.01):
    '''the first n decay rates k² of the degree l, the roots bracketed on steps dk'''
    out, k = [], 0.05
    f0 = matching(kind, l, k, inner, lay)
    while len(out) < n:
        f1 = matching(kind, l, k + dk, inner, lay)
        if np.sign(f1) != np.sign(f0):
            out.append(brentq(lambda x: matching(kind, l, x, inner, lay), k, k + dk, xtol=1e-15, rtol=1e-15))
        k, f0 = k + dk, f1
    return np.array(out)**2

def resolved(rs, sigma, nm, p=None):
    out = "; par.mantle = 'resolved'; par.r_surface = %r; par.N_mantle = %d; par.sigma_mantle = %r" % (rs, nm, sigma)
    return out + ('' if p is None else '; par.sigma_mantle_param = %r' % p)

DEC = ("par.hydro = 0; par.magnetic = 1; par.B0 = 'axial'; par.cnorm = 1; par.Hendrik = 1.0; par.MagnetD = 1.0; par.N = 48; "
       "g = 0.34; par.m = 0; par.symm = 1; par.itau = 0.0; par.nev = 6")
# name, changes, inner (None: a full sphere, 'ins', or the inner core's s), the layer (r_surface, σ, (ln σ)'); m = 0,
# symm = 1: the poloidal l = 1 (two modes) and 3, the toroidal l = 2
decay = [
    ('decay_uniform', "par.rtau = -10.0" + resolved(1.4, 0.3, 32), 'ins', (1.4, 0.3, None)),
    ('decay_exp', "par.rtau = -10.0" + resolved(1.4, '0.3*exp(-p*(r - 1))', 48, 10.0), 'ins',
     (1.4, lambda r: 0.3*np.exp(-10*(r - 1)), lambda r: -10.0)),
    ('decay_sphere', "par.ricb = 0; par.rtau = -10.0" + resolved(1.3, 2.0, 24), None, (1.3, 2.0, None)),
    ('decay_cic', "par.innercore = 'conducting, Chebys'; par.sigma_i2o = 0.3; par.N_cic = 32; par.rtau = -8.0"
                  + resolved(1.5, '0.5*r**(-p)', 32, 6.0), 0.3,
     (1.5, lambda r: 0.5*r**-6, lambda r: -6/r)),
]
for name, extra, inner, lay in decay:
    d = build(name, DEC + '; ' + extra)
    ref = -np.r_[rates('p', 1, 2, inner, lay), rates('p', 3, 1, inner, lay), rates('t', 2, 1, inner, lay)]
    rb, rE = mag(d)
    err = closest(eig(d), ref)
    report(name, err < 1e-10 and top(rb) < 1e-10 and top(rE) < 1e-10,
           'rates %.1e, resid𝐛 %.1e, resid𝐄 %.1e' % (err, top(rb), top(rE)))

# 2. against conductive_IC 2d84351 (the same runs: Ek 1e-3, Em 1e-3, Le² 1e-3, cnorm 'mag_energy'; first two eigenvalues)
FLOW = ("par.magnetic = 1; par.cnorm = 'mag_energy'; par.Hendrik = 1e-3; par.MagnetD = 1e-3; par.ViscosD = 1e-3; par.N = 48; "
        "par.rtau = 0.0; par.itau = 1.0; par.nev = 4")
cic = [
    ('cic_noslip', 20, "par.B0 = 'G21 dipole'; par.m = 1; par.symm = 1" + resolved(1.3, 0.5, 24),
     [-2.189257542658584188e-01 + 1.186705147077108613e+00j, -2.829635554045004109e-01 + 8.851812486937911117e-01j]),
    ('cic_slip_exp', 20, "par.B0 = 'Luo_S2'; par.B0_l = 2; par.m = 1; par.symm = 1; par.bco = 0; par.bci = 0"
                         + resolved(1.3, '0.5*exp(-p*(r - 1))', 32, 5.0),
     [-1.348911330885377846e-01 + 8.784959438985906477e-01j, -1.052897289342807213e-01 + 1.171828507682453280e+00j]),
    ('cic_sphere_slip', 20, "par.B0 = 'G21 dipole'; par.ricb = 0; par.m = 1; par.symm = -1; par.bco = 0"
                            + resolved(1.2, 1.0, 24),
     [-3.135203734838539445e-03 + 1.003250866586339995e+00j, -1.837322970440588843e-01 + 1.037571733808358321e+00j]),
    ('cic_innercore', 19, "par.B0 = 'axial'; par.m = 0; par.symm = 1; par.innercore = 'conducting, Chebys'; par.N_cic = 24"
                          + resolved(1.3, 0.2, 24),
     [-2.484116626944381789e-01 + 1.009483775046214671e+00j, -1.491071239782212621e-01 + 1.338150048687188232e+00j]),
    ('cic_m2_slip', 21, "par.B0 = 'axial'; par.m = 2; par.symm = -1; par.bco = 0" + resolved(1.1, 2.0, 24),
     [-1.542370003186011029e-01 + 1.173514718393598777e+00j, -2.448650470406585344e-01 + 9.477205394187935283e-01j]),
]
for name, lmax, extra, ref in cic:
    d = build(name, FLOW + '; ' + extra, lmax)
    err = closest(eig(d), np.array(ref))
    rb, rE = mag(d)
    report(name, err < 1e-10, 'eigenvalues %.1e from conductive_IC; resid𝐛 %.1e, resid𝐄 %.1e' % (err, top(rb), top(rE)))

# 3. a thin layer of thickness h and conductance c1 = σ h against the thin wall (c = h, c1), the fluid slipping along it
C1 = 0.01
SLIP = FLOW + "; g = 0.42; par.B0 = 'G21 dipole'; par.m = 1; par.symm = 1; par.bco = 0; par.nev = 3"
dif = []
for h in (0.01, 0.005, 0.0025):
    lr = eig(build('thin_%g' % h, SLIP + resolved(1 + h, C1/h, 12)))
    lt = eig(build('wall_%g' % h, SLIP + "; par.mantle = 'TWA'; par.c_cmb = %r; par.c1_cmb = %r; par.muw_cmb = 1.0" % (h, C1)))
    dif.append(closest(lr, lt[:2]))
q = [dif[1]/dif[0], dif[2]/dif[1]]
report('thin_layer', all(0.4 < x < 0.6 for x in q) and dif[2] < 1e-3,
       'differences from the thin wall %s for h = 0.01, 0.005, 0.0025 (ratios %.2f, %.2f)'
       % (', '.join('%.1e' % x for x in dif), q[0], q[1]))

# 4. a layer of vanishing conductance: an insulating mantle (the fluid slipping along it, a field of degree 2)
INS = FLOW + "; g = 0.42; par.B0 = 'Luo_S2'; par.B0_l = 2; par.m = 1; par.symm = 1; par.bco = 0; par.bci = 0; par.nev = 3"
l0 = eig(build('insulating', INS))
l1 = eig(build('sigma_zero', INS + resolved(1.3, 1e-9, 16)))
report('sigma_zero', closest(l1, l0[:2]) < 1e-7, 'eigenvalues %.1e from the insulating mantle' % closest(l1, l0[:2]))

# 5. a rigid rotation offset a = 0.3 ('Wu 2005' with r_out 0.9, N 64, which resolves it: the budgets close to 1e-9; a
#    conducting inner core, the fluid slipping along the layer)
WU = (FLOW + "; par.structure = 'Wu 2005'; par.r_out = 0.9; par.N = 64; g = 0.42; par.B0 = 'G21 dipole'; par.m = 1; "
      "par.symm = 1; par.bco = 0; par.nev = 3; par.innercore = 'conducting, Chebys'; par.sigma_i2o = 0.5; par.N_cic = 24"
      + resolved(1.3, '0.5*exp(-p*(r - 1))', 24, 5.0))
da = build('rigid_offset', WU + "; par.diff_rot = 1; par.diff_rot_type = 'rigid'; par.diff_rot_ampl = 0.3; par.itau = 0.7")
db = build('rigid_frame', WU + "; par.Gaspard = 1.3")
la, lb = eig(da), eig(db) - 0.3j
err = closest(la, lb[:2])
ea, eb = top(mag(da)[1]), top(mag(db)[1])
report('rigid_offset', err < 1e-12 and ea < 1e-8 and eb < 1e-8,
       'eigenvalues %.1e from those of the turning frame; resid𝐄 %.1e, %.1e in the turning frame' % (err, ea, eb))

# 6. the layer, the inner core and the fluid turning at different rates (ΔΩ = 0.4 (0.5 + r²))
d = build('sheared', WU + "; par.diff_rot = 1; par.diff_rot_type = 'user'; par.diff_rot_profiles = {0: '0.5 + r**2'}; "
                          "par.diff_rot_ampl = 0.4")
rb, rE = mag(d)
report('sheared', top(rE) < 1e-8 and top(rb) < 1e-8, 'resid𝐄 %.1e, resid𝐛 %.1e' % (top(rE), top(rb)))

# 7. kinematic induction: just_induction.py and solve_ind.py against the field of a mode without Lorentz force
d = build('induction', "par.magnetic = 1; par.B0 = 'axial'; par.cnorm = 1; par.Hendrik = 1e-30; par.MagnetD = 1e-2; "
                       "par.m = 1; par.symm = 1; par.rtau = 0.0; par.itau = 0.5; par.nev = 2; par.N = 48; g = 0.34"
                       + resolved(1.3, 0.5, 24))
beig, meig, z = field(d, 'magnetic'), field(d, 'magnetic_mantle'), eig(d)[0]
py, n = sys.executable, str(mk_run.NCPUS)
for cmd in ([py, 'bin/just_induction.py', repr(complex(z))], mk_run.MPIEXEC.split() + ['-n', n, py, 'bin/solve_ind.py']):
    p = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
    if p.returncode:
        sys.exit('%s: %s failed\n%s' % (d, cmd[-1], (p.stdout + p.stderr)[-3000:]))
eb_ = abs(field(d, 'magnetic')[:, 0] - beig[:, 0]).max()/abs(beig[:, 0]).max()
em_ = abs(field(d, 'magnetic_mantle')[:, 0] - meig[:, 0]).max()/abs(meig[:, 0]).max()
report('induction', eb_ < 1e-10 and em_ < 1e-10, 'field %.1e in the fluid, %.1e in the layer' % (eb_, em_))

print('mantle_check: ' + ('all cases ok' if not fail else 'FAILED'))
sys.exit(fail)
