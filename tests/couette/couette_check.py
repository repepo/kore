#!/usr/bin/env python3
'''
usage: couette_check.py [WORKDIR]   (default: tests/couette/runs)

Forcing 12, the steady rotation of the walls (spherical Couette flow; NEW 2026-11-08, from conductive_IC 0780790), and the
budgets of the forced flows that the walls drive. An incompressible fluid in a rotating shell (ricb 0.35, Ek 1e-3, no-slip
walls, N 48) unless said otherwise, built with tests/tidal/mk_run.py's template. Six groups:

1. Against conductive_IC 2d84351: eleven runs (lmax 31, 2 ranks), the inner core turning, both walls turning (0.4 and
   -0.3) or the mantle alone, without a field and with B0 'axial' over an insulating, conducting (Chebyshev and Bessel)
   and thin-layer ('TWA') inner core, a thin-layer and a resolved mantle, permeability jumps at the turning walls (a
   conducting inner core of permeability 1.5 and conductivity 0.3, a thin outer layer with a thickness and permeability
   2), and B0 'Luo_S2' over the Bessel core (Le² = Em = 1e-3). The kinetic and magnetic energies of conductive_IC's
   solutions (its spin_doctor), and the magnetic energy in its inner core (al's spin_doctor on its fields; conductive_IC
   writes none for forced runs), must match to 1e-9 (the digits of flow.dat), and residσ, resid𝐋 and resid𝐄 must be
   below 1e-9.
2. Stokes flow: without rotation (Gaspard = 0) the walls drive u_phi = s ω(r) alone, where the viscous stress transmits a
   torque that does not depend on r, r⁴ ρ0 ν ω' constant, and ω takes the rates of the walls there: T_1 = r ω/2 and
   nothing else. 'Boussinesq' (ω = A + B/r³), 'Wu 2005' (r_out 0.9, ρ0 falls 22 times across the shell), and 'Wu 2005'
   with a viscosity that varies (diffusivity 'user'), ω by quadrature of the run's own profiles. To 1e-10 of the largest
   T_1, and every other coefficient below 1e-10 of it.
3. Rigid rotation: a full sphere whose wall turns, a shell whose walls turn at one rate, and an inner core that turns
   under a stress-free outer wall at rest. The fluid turns rigidly (T_1 = ΔΩ r/2, nothing else), and the axisymmetric
   field is not changed (b = 0), also in conducting solids that turn with the fluid: a conducting inner core, a thin
   layer, a resolved mantle, and solids whose permeability differs from the fluid's. To 1e-10. (Every term of the
   budgets is then zero, and the residuals compare round-off with round-off: they are not checked.)
4. Budgets (residσ, residω, resid𝐋, resid𝑠, resid𝐄 below 1e-9) where conductive_IC cannot follow: 'Wu 2005' (N 96)
   with buoyancy, without and with a field over the Chebyshev core and a thin outer layer; a differential rotation
   ('Y20-wall-bounded') with a field over the Chebyshev core; the toroidal B0 'Malkus' and
   'wall' over the Chebyshev core ('Malkus', B ∝ s, is not changed by an axisymmetric flow: b = 0, and resid𝐛 is 0, not
   0/0); a stress-free outer wall at rest, along which the fluid slips, with a drag (kappa_cmb), with a thin conducting
   layer and with a resolved mantle. resid𝐄 is relative to the viscous dissipation too (as residσ, below).
5. The budgets of the other forced flows that the walls drive: the libration (forcing 7, m 0), with and without a field
   (a stress-free inner wall along a conducting inner core at rest), and the thermal forcing (13, steady). resid𝐋 is
   computed for them since 2026-11-08 (it was nan), and residσ is relative to the viscous dissipation too (it was about
   1 for the libration, the round-off of the small power of the viscous force relative to itself). residω is 0 for the
   steady runs (it was 0/0).
6. The stops: forcing 12 with a forcing frequency, m 1, symm -1, no viscosity, a turning inner core in a full sphere,
   ThermaD 0, a wall of zero radial stress, a turning stress-free wall, no turning wall; a forcing number that does not
   exist. And the cases that run: a stress-free wall at rest, a full sphere.

Exit code 1 if a check failed.
'''
import os, sys, shutil, subprocess, json
import numpy as np

T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(T, '..', 'tidal'))
import mk_run   # noqa: E402  (the run builder of tests/tidal: 'Boussinesq', ricb 0.35, NCPUS ranks)

W = sys.argv[1] if len(sys.argv) > 1 else os.path.join(T, 'runs')
os.makedirs(W, exist_ok=True)
LMAX_LINE = 'par.lmax = par.ellmax(par.ncpus, max(g, 4.*par.ncpus/par.N), par.m, par.N)'
assert LMAX_LINE in mk_run.T
py = sys.executable


def setup(d, extra, lmax):
    '''the run directory d (bin and parameters.py, as mk_run.build) with that lmax, on 2 ranks (1 with NCPUS = 1)'''
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d + '/bin')
    for f in os.listdir(mk_run.BIN):
        if os.path.isfile(mk_run.BIN + '/' + f) and f != 'parameters.py':
            shutil.copy(mk_run.BIN + '/' + f, d + '/bin/' + f)
    n = min(mk_run.NCPUS, 2)
    t = mk_run.T.replace(LMAX_LINE, 'par.lmax = %d' % lmax)
    open(d + '/bin/parameters.py', 'w').write(t.format(ncpus=n, extra=extra.replace('; ', '\n')))
    return str(n)

def run(d, n, steps='savd', check=True):
    '''the steps of a run in d: s(ubmatrices), a(ssemble), v (solve), d (spin_doctor); (returncode, output) of the last'''
    cmds = {'s': [py, 'bin/submatrices.py', n], 'a': mk_run.MPIEXEC.split() + ['-n', n, py, 'bin/assemble.py'],
            'v': mk_run.MPIEXEC.split() + ['-n', n, py, 'bin/solve.py'], 'd': [py, 'bin/spin_doctor.py', n]}
    for st in steps:
        p = subprocess.run(cmds[st], cwd=d, capture_output=True, text=True)
        out = p.stdout + p.stderr
        open(os.path.join(d, 'log_' + st), 'w').write(out)
        if p.returncode:
            if check:
                sys.exit('%s: step %s failed\n%s' % (d, st, out[-3000:]))
            return p.returncode, out
    return 0, out

def build(name, extra, lmax=31, steps='savd'):
    d = os.path.join(W, name)
    run(d, setup(d, extra, lmax), steps)
    return d

def dat(d, f):
    return np.loadtxt(os.path.join(d, f), ndmin=2)[0]

fail = 0
def report(name, ok, text):
    global fail
    fail |= not ok
    print('%-14s %s  (%s)' % (name, 'ok' if ok else 'FAIL', text))

def budgets(d):
    '''the residuals of spin_doctor (nan where not computed): residσ, residω, resid𝐋, and resid𝑠, resid𝐄 where they exist'''
    f = dat(d, 'flow.dat')
    out = {'σ': f[16], 'ω': f[15], '𝐋': f[38]}
    if os.path.exists(os.path.join(d, 'thermal.dat')):
        out['𝑠'] = dat(d, 'thermal.dat')[3]
    if os.path.exists(os.path.join(d, 'magnetic.dat')):
        out['𝐄'] = dat(d, 'magnetic.dat')[10]
    return out

def budgets_ok(name, d, tol=1e-9, need=('σ', 'ω', '𝐋')):
    b = budgets(d)
    bad = [k for k in b if not b[k] <= tol] + [k for k in need if k not in b]
    report(name, not bad, ', '.join('resid%s %.1e' % (k, v) for k, v in b.items()))


# the flow (and the field) of the solution in its run directory, against the expected T_1(r) = r ω(r)/2: 'stokes' (ω from
# r⁴ ρ0 ν ω' = const and the walls' rates, by Gauss-Legendre quadrature of the run's profiles) or 'rigid' (ω = the rate)
READ = r'''
import os, sys, json; sys.path.insert(0, 'bin')
import numpy as np
import numpy.polynomial.chebyshev as ch
from parameters import par
import utils as ut
import radial_profiles as rap
import utils4pp as upp
kind = sys.argv[1]
def load(name):   # the first solution, from name.npy or the text files
    return ut.load_field(name, 0)
Plj, Tlj = upp.expand_reshape_sol(load('flow'), par.symm)
lp, lt = ut.ell(par.m, par.lmax, par.symm)[:2]
ri, wo, wi = par.ricb, par.forcing_amplitude_cmb, par.forcing_amplitude_icb
dom = [ri, 1.] if ri > 0 else [-1., 1.]
r = np.linspace(max(ri, 1e-3), 1., 201)
if kind == 'rigid':
    w = np.full_like(r, wo if par.bco == 1 else wi)
else:
    xg, wg = np.polynomial.legendre.leggauss(200)
    def I(b):   # the integral of 1/(r⁴ ρ0 ν) from ri to b
        rr = ri + (b - ri)*(xg + 1)/2
        return (b - ri)/2*np.sum(wg/(rr**4*rap.prf.density(rr, 0)*rap.prf.viscosity(rr, 0)))
    w = wi + (wo - wi)*np.array([I(b) for b in r])/I(1.)
T1 = ch.Chebyshev(Tlj[list(lt).index(1)], domain=dom)(r)
exact = r*w/2
scale = abs(exact).max()
others = max(abs(Plj).max(), abs(np.delete(Tlj, list(lt).index(1), axis=0)).max() if len(lt) > 1 else 0.)
out = {'T1': abs(T1 - exact).max()/scale, 'others': others/scale}
if par.magnetic:
    b = [abs(load(n)).max() for n in ('magnetic', 'magnetic_ic', 'magnetic_mantle') if ut.field_exists(n)]
    out['b'] = max(b)/scale
print(json.dumps(out))
'''

def profile(name, d, kind, tol=1e-10):
    p = subprocess.run([py, '-c', READ, kind], cwd=d, capture_output=True, text=True)
    if p.returncode:
        report(name, False, 'could not read the solution: ' + p.stderr.strip()[-300:])
        return
    e = json.loads(p.stdout.strip().splitlines()[-1])
    report(name, max(e.values()) < tol, ', '.join('%s %.1e' % (k, v) for k, v in e.items()) + ' of the largest T_1')


F12 = "par.m = 0; par.symm = 1; par.forcing = 12; par.forcing_frequency = 0.0; par.N = 48"
IC1 = "par.forcing_amplitude_cmb = 0.0; par.forcing_amplitude_icb = 1.0"
BOTH = "par.forcing_amplitude_cmb = 0.4; par.forcing_amplitude_icb = -0.3"
MAG = "par.magnetic = 1; par.cnorm = 'mag_energy'; par.Hendrik = 1e-3; par.MagnetD = 1e-3; par.B0 = 'axial'"
CHEB, BES = "par.innercore = 'conducting, Chebys'; par.N_cic = 32", "par.innercore = 'conducting, Bessel'"
TWA2 = "par.mantle = 'TWA'; par.c_cmb = 0.0; par.c1_cmb = 0.02; par.innercore = 'TWA'; par.c_icb = 0.0; par.c1_icb = 0.05"
MAN = "par.mantle = 'resolved'; par.r_surface = 1.3; par.N_mantle = 24; par.sigma_mantle = 0.5"
WU = ("par.thermal = 1; par.structure = 'Wu 2005'; par.stratification = 'top hat'; par.r_out = 0.9; par.Beyonce = 10.0; "
      "par.ThermaD = 1e-3; par.rpower_u = 5; par.rhopower_u = 1; par.rpower_v = 3; par.rhopower_v = 1; par.rpower_pp = 4; "
      "par.rhopower_pp = 4; par.N = 96")

MU_IC = "par.mu_i2o = 1.5; par.sigma_i2o = 0.3"
MUW = "par.mantle = 'TWA'; par.c_cmb = 0.01; par.c1_cmb = 0.02; par.muw_cmb = 2.0"

# 1. against conductive_IC 2d84351: (changes, (KE, ME, MEic) of its solution)
CIC = {
    'c_ic':      (IC1, (2.7382913011e-03,)),
    'c_both':    (BOTH, (6.0728367282e-02,)),
    'c_cmb':     ("par.forcing_amplitude_cmb = 1.0; par.forcing_amplitude_icb = 0.0", (3.9341001724e-01,)),
    'c_ins':     (MAG + "; " + IC1, (2.7107112456e-03, 9.2818757303e-02, 1.3257110790e-02)),
    'c_cheb':    (MAG + "; " + IC1 + "; " + CHEB, (3.1154573566e-03, 6.3118169690e-01, 1.6893389650e-01)),
    'c_bes':     (MAG + "; " + IC1 + "; " + BES, (3.1154573566e-03, 6.3118169690e-01, 1.6893389650e-01)),
    'c_twa':     (MAG + "; " + BOTH + "; " + TWA2, (6.1155860242e-02, 1.1069818412e-01, 6.5596258430e-03)),
    'c_man':     (MAG + "; " + BOTH + "; " + MAN + "; " + CHEB, (6.2273459938e-02, 5.6644882057e-01, 9.6123757910e-02)),
    'c_s2_bes':  (MAG.replace("'axial'", "'Luo_S2'; par.B0_l = 2") + "; " + IC1 + "; " + BES,
                  (1.9078254509e-03, 1.1299716761e+01, 2.4658329070e+00)),
    'c_mu':      (MAG + "; " + IC1 + "; " + CHEB + "; " + MU_IC, (2.8825706594e-03, 2.5806390545e-01, 5.9359477760e-02)),
    'c_muw':     (MAG + "; " + BOTH + "; " + MUW, (6.1217147232e-02, 6.8848824988e-02, 6.6963774050e-03)),
}
for name, (extra, ref) in CIC.items():
    d = build(name, F12 + "; " + extra)
    got = (dat(d, 'flow.dat')[0],) + ((dat(d, 'magnetic.dat')[0], dat(d, 'magnetic.dat')[6]) if len(ref) > 1 else ())
    err = max(abs(g/r - 1) for g, r in zip(got, ref))
    b = budgets(d)
    worst = max(v for k, v in b.items() if k in ('σ', '𝐋', '𝐄'))
    report(name, err < 1e-9 and worst < 1e-9, '%s %.1e from conductive_IC; residσ, resid𝐋%s at most %.1e'
           % ('KE' if len(ref) == 1 else 'KE, ME, MEic', err, ', resid𝐄' if '𝐄' in b else '', worst))

# 2. Stokes flow: no rotation
NOROT = "par.Gaspard = 0.0"
profile('stokes', build('stokes', F12 + "; " + BOTH + "; " + NOROT), 'stokes')
WU0 = "par.structure = 'Wu 2005'; par.r_out = 0.9"
profile('stokes_wu', build('stokes_wu', F12 + "; " + BOTH + "; " + NOROT + "; " + WU0), 'stokes')
NU = "par.diffusivity = 'user'; par.diffusivity_profiles = {'nu': '1 + 2*r**2', 'kappa': '1'}"
profile('stokes_wu_nu', build('stokes_wu_nu', F12 + "; " + BOTH + "; " + NOROT + "; " + WU0 + "; " + NU), 'stokes')

# 3. rigid rotation
FS = "par.ricb = 0; par.innercore = 'insulator'; par.forcing_amplitude_cmb = 0.7; par.forcing_amplitude_icb = 0.0"
profile('rigid_sphere', build('rigid_sphere', F12 + "; " + MAG + "; " + FS, lmax=31), 'rigid')
SAME = "par.forcing_amplitude_cmb = 0.6; par.forcing_amplitude_icb = 0.6"
profile('rigid_shell', build('rigid_shell', F12 + "; " + MAG + "; " + SAME + "; " + MAN + "; " + CHEB), 'rigid')
profile('rigid_twa', build('rigid_twa', F12 + "; " + MAG + "; " + SAME + "; " + TWA2), 'rigid')
SFO = "par.bco = 0; par.forcing_amplitude_cmb = 0.0; par.forcing_amplitude_icb = 0.5"
profile('rigid_slip', build('rigid_slip', F12 + "; " + MAG + "; " + SFO + "; " + CHEB), 'rigid')
profile('rigid_mu', build('rigid_mu', F12 + "; " + MAG + "; " + SAME + "; " + CHEB + "; " + MU_IC + "; " + MUW), 'rigid')

# 4. budgets beyond conductive_IC
budgets_ok('wu_thermal', build('wu_thermal', F12 + "; " + IC1 + "; " + WU, lmax=47), need=('σ', 'ω', '𝐋', '𝑠'))
budgets_ok('wu_field', build('wu_field', F12 + "; " + BOTH + "; " + WU + "; " + MAG + "; " + CHEB
                             + "; par.mantle = 'TWA'; par.c_cmb = 0.0; par.c1_cmb = 0.02", lmax=47), need=('σ', 'ω', '𝐋', '𝑠', '𝐄'))
DR = "par.diff_rot = 1; par.diff_rot_type = 'Y20-wall-bounded'; par.diff_rot_ampl = 0.1"
budgets_ok('diffrot', build('diffrot', F12 + "; " + BOTH + "; " + DR + "; " + MAG + "; " + CHEB), need=('σ', 'ω', '𝐋', '𝐄'))
TOR = MAG.replace("'axial'", "None; par.B0_tor = '%s'; par.cnorm_tor = 'mag_energy'")
d = build('malkus', F12 + "; " + BOTH + "; " + TOR % 'Malkus' + "; " + CHEB)
budgets_ok('malkus', d, need=('σ', 'ω', '𝐋', '𝐄'))
m = dat(d, 'magnetic.dat')
report('malkus_b0', m[0] == 0 and m[6] == 0 and m[3] == 0, 'ME %r, MEic %r, resid𝐛 %r: an axisymmetric flow neither shears '
       'the field ∝ s φ̂ nor carries its B/s, uniform, around' % (m[0], m[6], m[3]))
budgets_ok('tor_wall', build('tor_wall', F12 + "; " + BOTH + "; " + TOR % 'wall' + "; " + CHEB), need=('σ', 'ω', '𝐋', '𝐄'))
SF = "par.bco = 0; " + IC1
budgets_ok('slip_drag', build('slip_drag', F12 + "; " + SF + "; par.kappa_cmb = 0.05"))
budgets_ok('slip_twa', build('slip_twa', F12 + "; " + SF + "; " + MAG + "; par.mantle = 'TWA'; par.c_cmb = 0.0; "
                             "par.c1_cmb = 0.02"), need=('σ', 'ω', '𝐋', '𝐄'))
budgets_ok('slip_man', build('slip_man', F12 + "; " + SF + "; " + MAG + "; " + MAN), need=('σ', 'ω', '𝐋', '𝐄'))

# 5. the libration and the thermal forcing
LIB = "par.m = 0; par.symm = 1; par.forcing = 7; par.forcing_frequency = 0.37; par.forcing_amplitude_cmb = 1.0; par.N = 48"
budgets_ok('lib', build('lib', LIB, lmax=19))
budgets_ok('lib_slip', build('lib_slip', LIB + "; par.bci = 0; " + MAG + "; " + CHEB, lmax=19), need=('σ', 'ω', '𝐋', '𝐄'))
F13 = ("par.m = 0; par.symm = 1; par.forcing = 13; par.forcing_frequency = 0.0; par.forcing_amplitude_icb = -1.0; "
       "par.thermal = 1; par.stratification = 'inverse square'; par.Beyonce = 10.0; par.ThermaD = 1e-3; par.N = 48")
d = build('f13', F13, lmax=31)
b = budgets(d)
budgets_ok('f13', d, need=('σ', 'ω', '𝐋', '𝑠'))
report('f13_resid_w', b['ω'] == 0, 'residω %r: every term of the imaginary balance is 0 in a steady flow of m = 0' % b['ω'])

# 6. the stops (at the start of submatrices.py) and the cases that run
BASE = F12 + "; " + IC1
STOPS = [
    ('stop_freq', "par.forcing_frequency = 0.5", 'it needs forcing_frequency = 0'),
    ('stop_m1', "par.m = 1", 'needs m = 0 and symm = 1'),
    ('stop_symm', "par.symm = -1", 'needs m = 0 and symm = 1'),
    ('stop_inviscid', "par.ViscosD = 0.0", 'needs a viscous fluid'),
    ('stop_sphere_ic', "par.ricb = 0", 'has no inner core to turn'),
    ('stop_thermal', "par.thermal = 1; par.stratification = 'top hat'; par.ThermaD = 0.0", 'needs ThermaD > 0'),
    ('stop_bc2', "par.bco = 2", 'walls that stay in place'),
    ('stop_slip_turn', "par.bci = 0", 'the inner wall turns'),
    ('stop_no_turn', "par.forcing_amplitude_icb = 0.0", 'needs a wall that turns'),
    ('stop_forcing', "par.forcing = 15", 'forcing 15: 0 (an eigenvalue problem)'),
    ('slip_at_rest', "par.bco = 0", None),
    ('sphere', "par.ricb = 0; par.forcing_amplitude_icb = 0.0; par.forcing_amplitude_cmb = 1.0", None),
]
for name, extra, msg in STOPS:
    d = os.path.join(W, name)
    rc, out = run(d, setup(d, BASE + "; " + extra, 31), 's', check=False)
    ok = (rc == 0) if msg is None else (rc != 0 and msg in out)
    report(name, ok, 'runs' if msg is None and rc == 0 else (out.strip().splitlines() or ['(no output)'])[-1][:90])

print('couette_check: ' + ('all cases ok' if not fail else 'FAILED'))
sys.exit(fail)
