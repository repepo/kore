#!/usr/bin/env python3
'''
usage: cic_forcings_check.py [WORKDIR]   (default: tests/cic_forcings/runs)

Forcings 1, 2 and 8, ported from conductive_IC (NEW 2026-11-08): the wave-like tide (Lin & Ogilvie 2018), the
eccentricity tide of Enceladus (Rekier et al. 2019) and the libration in the frame of the mantle. An incompressible
fluid in a rotating shell (ricb 0.35, Ek 1e-3, no-slip walls, N 48) unless said otherwise, built with
tests/tidal/mk_run.py's template on 2 ranks. Six groups:

1. Against conductive_IC 2d84351: forcing 1 (m 2, w 0.37, a 1, lmax 41) with no-slip and with stress-free walls, and
   forcing 8 (w 0.7, lmax 23) without and with a field (B0 'axial' over a conducting inner core, Chebyshev, Le² = Em =
   1e-3): the kinetic and magnetic energies of conductive_IC's solutions must match to 1e-9 (the digits of flow.dat).
   conductive_IC's forcing 8 is twice the Poincaré force of its angle: the runs here take twice its angle (0.3 for
   0.15). And the fits of forcing 2 (utils.eccen_tide) against conductive_IC's, at ricb 0.8 and 0.95, every wall and m:
   to the last bit.
2. Forcing 8 against forcing 7 minus the rigid libration of the mantle, T_1 = i w A r/2 (utils.lifted_flow): the flow
   of the mantle's frame is that of the mean frame minus the mantle's own motion, with the inner core librating with the
   mantle (forcing 7's forcing_amplitude_icb = A), or relative to it by forcing_amplitude_icb (forcing 7's A plus that);
   the field is the same in both frames. Cases: incompressible; 'Wu 2005' (N 96) with buoyancy and a differential
   rotation ('Y20-wall-bounded'); B0 'axial' over a Bessel inner core and a thin conducting mantle; a full sphere; the
   inner core librating relative to the mantle; a stress-free inner wall. To 1e-12 of the largest coefficient. Where
   nothing couples the fluid to the walls (stress-free walls, no drag, insulating; an inviscid fluid, in a full sphere) the
   fluid stays at rest in the mean frame: the flow is -T_1 of the mantle, to 1e-12.
3. Forcing 1 beyond conductive_IC: m 0 and m 1 (symm -1) in a shell and in a full sphere (N 96), with no-slip and
   stress-free walls, with a differential rotation, with a viscosity that varies (diffusivity 'user'): residσ, residω,
   resid𝐋 (m 0 and 1) and residens below 1e-9. In a full sphere with a stress-free wall, or without viscosity, the force
   -2 ẑ×𝐮_e has a uniform curl and the response is a rigid rotation, T_1 = g r with g = 3 i a/(2 w) (m 0, about z) and
   g = (3√3/4) i a/(w - 1) (m 1, about an equatorial axis; Gaspard 1): to 1e-10, and every other coefficient below 1e-10
   of g (the budgets are then round-off over round-off). In viscous units (Gaspard 1/Ek, ViscosD 1, w and a times 1/Ek)
   the run of group 1 has 1/Ek² times its kinetic energy, to 1e-9: the Coriolis force carries Gaspard (conductive_IC's
   did not).
4. Forcing 2: the budgets (residσ, residω below 1e-9, resid𝐄 with a field) for m 0 and m 2, prograde and retrograde,
   'Wu 2005', a differential rotation ('Y20', not 0 at the walls), and B0 'axial' with insulating walls. Its wall rows
   are checked by tests/forcing/rows_check.py.
5. Forcing 8 where forcing 7 cannot follow: a stress-free outer wall with a drag (kappa_cmb), with a thin conducting
   layer and with the resolved mantle; a stress-free inner wall along a conducting inner core that librates with the
   mantle (at rest in its frame). residσ, residω, resid𝐋 (and resid𝐄) below 1e-9.
6. The stops: conductive_IC's forcings 3, 4 and 5; forcing 1 with m 3, with m 2 in a full sphere, a density that varies,
   buoyancy, a field, a wall of zero radial stress, a drag, at frequency 0; forcing 2 with m 1, ricb 0.5, a stress-free
   wall, a conducting thin layer, a conducting inner core; forcing 8 with m 2, an inner core's libration in a full sphere,
   at frequency 0, an inviscid fluid in a shell, a wall of zero radial stress.

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

def build(name, extra, lmax, steps='savd'):
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
    '''the residuals of spin_doctor (nan where not computed): residσ, residω, resid𝐋, residens, and resid𝐄 with a field'''
    f = dat(d, 'flow.dat')
    out = {'σ': f[16], 'ω': f[15], '𝐋': f[38], 'ens': f[14]}
    if os.path.exists(os.path.join(d, 'magnetic.dat')):
        out['𝐄'] = dat(d, 'magnetic.dat')[10]
        del out['ens']   # the enstrophy budget leaves out the Lorentz force
    return out

def budgets_ok(name, d, tol=1e-9, need=('σ', 'ω')):
    b = budgets(d)
    bad = [k for k in need if not b.get(k, np.nan) <= tol] + [k for k, v in b.items() if not (np.isnan(v) or v <= tol)]
    report(name, not bad, ', '.join('resid%s %.1e' % (k, v) for k, v in b.items() if not np.isnan(v)))


# the flow (and the fields) of a forcing 8 run against those of a forcing 7 run, or against -x alone (OTHER = '-')
FRAME = r'''
import os, sys, json; sys.path.insert(0, 'bin')
import numpy as np
from parameters import par
import utils as ut
def load(d, name):
    f = os.path.join(d, 'real_%s.field' % name)
    if not os.path.exists(f):
        return None
    return (np.loadtxt(f) + 1j*np.loadtxt(os.path.join(d, 'imag_%s.field' % name))).reshape(-1)
x = ut.lifted_flow()[:2*ut.n]
u8 = load('.', 'flow')
other = sys.argv[1]
if other == '-':
    out = {'flow': abs(u8 + x).max()/abs(x).max()}
else:
    u7 = load(other, 'flow')
    out = {'flow': abs(u8 + x - u7).max()/abs(u7).max()}
    for n in ('magnetic', 'magnetic_ic', 'magnetic_mantle', 'temperature'):
        a, b = load('.', n), load(other, n)
        if a is not None and b is not None:
            out[n] = abs(a - b).max()/max(abs(b).max(), 1e-300)
print(json.dumps(out))
'''

def frame(name, d8, d7, tol=1e-12):
    p = subprocess.run([py, '-c', FRAME, d7 if d7 == '-' else os.path.abspath(d7)], cwd=d8, capture_output=True, text=True)
    if p.returncode:
        report(name, False, 'could not read the solutions: ' + p.stderr.strip()[-300:])
        return
    e = json.loads(p.stdout.strip().splitlines()[-1])
    report(name, max(e.values()) < tol, ', '.join('%s %.1e' % (k, v) for k, v in e.items()))


MAG = "par.magnetic = 1; par.cnorm = 'mag_energy'; par.Hendrik = 1e-3; par.MagnetD = 1e-3; par.B0 = 'axial'"
CHEB, BES = "par.innercore = 'conducting, Chebys'; par.N_cic = 32", "par.innercore = 'conducting, Bessel'"
TWAO = "par.mantle = 'TWA'; par.c_cmb = 0.0; par.c1_cmb = 0.02"
MAN = "par.mantle = 'resolved'; par.r_surface = 1.3; par.N_mantle = 24; par.sigma_mantle = 0.5"
WU = ("par.thermal = 1; par.structure = 'Wu 2005'; par.stratification = 'top hat'; par.r_out = 0.9; par.Beyonce = 10.0; "
      "par.ThermaD = 1e-3; par.rpower_u = 5; par.rhopower_u = 1; par.rpower_v = 3; par.rhopower_v = 1; par.rpower_pp = 4; "
      "par.rhopower_pp = 4; par.N = 96")
DR = "par.diff_rot = 1; par.diff_rot_type = 'Y20-wall-bounded'; par.diff_rot_ampl = 0.1"
F1 = "par.forcing = 1; par.forcing_frequency = 0.37; par.forcing_amplitude_cmb = 1.0; par.N = 48"
F2 = "par.forcing = 2; par.ricb = 37/42; par.forcing_amplitude_cmb = 1.0; par.N = 48"
LIB = "par.m = 0; par.symm = 1; par.forcing_frequency = 0.7; par.forcing_amplitude_cmb = 0.3; par.N = 48"
F8, F7 = LIB + "; par.forcing = 8", LIB + "; par.forcing = 7; par.forcing_amplitude_icb = 0.3"

# 1. against conductive_IC 2d84351: (changes, lmax, (KE, ME) of its solution)
CIC = {
    'c_f1':       (F1 + "; par.m = 2", 41, (5.0350759843e-02,)),
    'c_f1_slip':  (F1 + "; par.m = 2; par.bci = 0; par.bco = 0", 41, (8.7999031604e-02,)),
    'c_f8':       (F8, 23, (1.6973093202e-02,)),
    'c_f8_field': (F8 + "; " + MAG + "; " + CHEB, 23, (1.7019592232e-02, 3.7256211901e-02)),
}
for name, (extra, lmax, ref) in CIC.items():
    d = build(name, extra, lmax)
    got = (dat(d, 'flow.dat')[0],) + ((dat(d, 'magnetic.dat')[0],) if len(ref) > 1 else ())
    err = max(abs(g/r - 1) for g, r in zip(got, ref))
    b = budgets(d)
    worst = max(v for k, v in b.items() if k in ('σ', 'ω', '𝐋', '𝐄') and not np.isnan(v))
    report(name, err < 1e-9 and worst < 1e-9, '%s %.1e from conductive_IC; budgets at most %.1e'
           % ('KE' if len(ref) == 1 else 'KE, ME', err, worst))

# conductive_IC's ut.eccen_tide (2d84351): (eta, wall, m): (P, dP)
FITS = {
    (0.8, 'cmb', -2): (8.246467346278637e-07, 3.9310213917936784e-09),
    (0.8, 'cmb', 0): (-2.8856624541294215e-07, -1.3755709396941786e-09),
    (0.8, 'cmb', 2): (-1.1780667637540912e-07, -5.615744845419032e-10),
    (0.8, 'icb', -2): (1.2901565456021017e-09, 5.061346641813474e-11),
    (0.8, 'icb', 0): (-4.5146074643384673e-10, -1.7711023833988863e-11),
    (0.8, 'icb', 2): (-1.8430807794315726e-10, -7.2304952025906315e-12),
    (0.95, 'cmb', -2): (2.1038494183494752e-07, 4.357704561645377e-08),
    (0.95, 'cmb', 0): (-7.361939386581997e-08, -1.524878946547115e-08),
    (0.95, 'cmb', 2): (-3.0054991690706804e-08, -6.225292230921966e-09),
    (0.95, 'icb', -2): (1.585162852354347e-09, 6.218670677477261e-11),
    (0.95, 'icb', 0): (-5.54691449640414e-10, -2.1760814340324427e-11),
    (0.95, 'icb', 2): (-2.2645183605062098e-10, -8.883815253538888e-12),
}
code = ('import sys, json; sys.path.insert(0, "bin"); import utils as ut; '
        'print(json.dumps([list(ut.eccen_tide(m, e, w)) for e, w, m in %r]))' % list(FITS))
p = subprocess.run([py, '-c', code], cwd=os.path.join(W, 'c_f1'), capture_output=True, text=True)
if p.returncode:
    report('fits', False, p.stderr.strip()[-300:])
else:
    got = json.loads(p.stdout.strip().splitlines()[-1])
    nbad = sum(tuple(g) != v for g, v in zip(got, FITS.values()))
    report('fits', nbad == 0, '%d of %d pairs differ from conductive_IC' % (nbad, len(FITS)))

# 2. forcing 8 against forcing 7 minus the mantle's libration
frame('lib', build('lib', F8, 23), build('lib7', F7, 23))
frame('lib_wu_dr', build('lib_wu_dr', F8 + "; " + WU + "; " + DR, 47), build('lib7_wu_dr', F7 + "; " + WU + "; " + DR, 47))
FLD = MAG + "; " + BES + "; " + TWAO
frame('lib_field', build('lib_field', F8 + "; " + FLD, 23), build('lib7_field', F7 + "; " + FLD, 23))
SPH = "par.ricb = 0; par.forcing_amplitude_icb = 0.0"
frame('lib_sphere', build('lib_sphere', F8 + "; " + SPH, 23), build('lib7_sphere', F7 + "; " + SPH, 23))
frame('lib_icb', build('lib_icb', F8 + "; par.forcing_amplitude_icb = 0.2", 23),
      build('lib7_icb', F7.replace('par.forcing_amplitude_icb = 0.3', 'par.forcing_amplitude_icb = 0.5'), 23))
frame('lib_slip_icb', build('lib_slip_icb', F8 + "; par.bci = 0", 23), build('lib7_slip_icb', F7 + "; par.bci = 0", 23))
frame('lib_free', build('lib_free', F8 + "; par.bci = 0; par.bco = 0", 23), '-')
frame('lib_inviscid', build('lib_inviscid', F8 + "; par.ViscosD = 0.0; par.ricb = 0", 23), '-')

# 3. forcing 1 beyond conductive_IC
for name, extra, lmax in [
        ('f1_m0', F1 + "; par.m = 0; par.symm = 1", 23),
        ('f1_m0_slip', F1 + "; par.m = 0; par.symm = 1; par.bci = 0; par.bco = 0", 23),
        ('f1_m1', F1 + "; par.m = 1; par.symm = -1", 24),
        ('f1_m0_sphere', F1 + "; par.m = 0; par.symm = 1; par.ricb = 0; par.N = 96", 23),
        ('f1_m1_sphere', F1 + "; par.m = 1; par.symm = -1; par.ricb = 0; par.N = 96", 24),
        ('f1_m2_dr', F1 + "; par.m = 2; " + DR, 41),
        ('f1_m2_nu', F1 + "; par.m = 2; par.diffusivity = 'user'; "
                          "par.diffusivity_profiles = {'nu': '1 + 2*r**2', 'kappa': '1'}", 41)]:
    budgets_ok(name, build(name, extra, lmax), need=('σ', 'ω', 'ens') + (('𝐋',) if ('m = 2' not in extra) else ()))
# a full sphere with a stress-free wall (or no viscosity): the force has a uniform curl, the response is a rigid rotation
RIGID = r'''
import sys, json; sys.path.insert(0, 'bin')
import numpy as np
from parameters import par
import utils as ut
import utils4pp as upp
x = (np.loadtxt('real_flow.field') + 1j*np.loadtxt('imag_flow.field')).reshape(-1)
P, T = upp.expand_reshape_sol(x, par.symm)
k = list(ut.ell(par.m, par.lmax, par.symm)[1]).index(1)
a, w, G = par.forcing_amplitude_cmb, par.forcing_frequency, par.Gaspard
g = 1.5j*a*G/w if par.m == 0 else 0.75*np.sqrt(3)*1j*a*G/(w - G)   # T_1 = g r
want = np.zeros_like(T); want[k, 1] = g
print(json.dumps({'T1': abs(T[k, 1]/g - 1), 'rest': max(abs(T - want).max(), abs(P).max())/abs(g)}))
'''
for name, extra in [('f1_m0_rigid', F1 + "; par.m = 0; par.symm = 1; par.ricb = 0; par.bco = 0"),
                    ('f1_m1_rigid', F1 + "; par.m = 1; par.symm = -1; par.ricb = 0; par.bco = 0"),
                    ('f1_m1_inviscid', F1 + "; par.m = 1; par.symm = -1; par.ricb = 0; par.ViscosD = 0.0")]:
    d = build(name, extra, 23 if 'm = 0' in extra else 24)
    p = subprocess.run([py, '-c', RIGID], cwd=d, capture_output=True, text=True)
    if p.returncode:
        report(name, False, 'could not read the solution: ' + p.stderr.strip()[-300:])
        continue
    e = json.loads(p.stdout.strip().splitlines()[-1])
    report(name, max(e.values()) < 1e-10, 'T_1 %.1e from the rigid rotation, the rest %.1e of it' % (e['T1'], e['rest']))
d = build('f1_viscous', F1.replace('0.37', '370.0').replace('amplitude_cmb = 1.0', 'amplitude_cmb = 1000.0')
          + "; par.m = 2; par.Gaspard = 1000.0; par.ViscosD = 1.0", 41)
ke = dat(d, 'flow.dat')[0]
report('f1_viscous', abs(ke/(5.0350759843e-02*1e6) - 1) < 1e-9, 'KE %.10e, 1e6 times conductive_IC\'s in rotation units to '
       '%.1e' % (ke, abs(ke/(5.0350759843e-02*1e6) - 1)))

# 4. forcing 2
for name, extra, lmax, need in [
        ('f2_m2_retro', F2 + "; par.m = 2; par.forcing_frequency = 1.0", 41, ('σ', 'ω')),
        ('f2_m2_pro', F2 + "; par.m = 2; par.forcing_frequency = -1.0", 41, ('σ', 'ω')),
        ('f2_m0', F2 + "; par.m = 0; par.forcing_frequency = 1.0", 23, ('σ', 'ω')),
        ('f2_wu', F2 + "; par.m = 2; par.forcing_frequency = -1.0; " + WU, 45, ('σ', 'ω')),
        ('f2_dr', F2 + "; par.m = 2; par.forcing_frequency = -1.0; par.diff_rot = 1; par.diff_rot_type = 'Y20'; "
                       "par.diff_rot_ampl = 0.1", 41, ('σ', 'ω')),
        ('f2_field', F2 + "; par.m = 2; par.forcing_frequency = -1.0; " + MAG, 41, ('σ', 'ω', '𝐄'))]:
    budgets_ok(name, build(name, extra, lmax), need=need)

# 5. forcing 8 where forcing 7 cannot follow
for name, extra, need in [
        ('lib_drag', F8 + "; par.bco = 0; par.kappa_cmb = 0.05", ('σ', 'ω', '𝐋')),
        ('lib_twa', F8 + "; par.bco = 0; " + MAG + "; " + TWAO, ('σ', 'ω', '𝐋', '𝐄')),
        ('lib_man', F8 + "; par.bco = 0; " + MAG + "; " + MAN, ('σ', 'ω', '𝐋', '𝐄')),
        ('lib_cheb_slip', F8 + "; par.bci = 0; " + MAG + "; " + CHEB, ('σ', 'ω', '𝐋', '𝐄'))]:
    budgets_ok(name, build(name, extra, 23), need=need)

# 6. the stops (at the start of submatrices.py)
STOPS = [
    ('stop_f3', LIB + "; par.forcing = 3", 'forcing 3: conductive_IC'),
    ('stop_f4', LIB + "; par.forcing = 4", 'forcing 4: conductive_IC'),
    ('stop_f5', LIB + "; par.forcing = 5", 'forcing 5: conductive_IC'),
    ('stop_f1_m3', F1 + "; par.m = 3", 'needs (m, symm) = (0, 1), (1, -1) or (2, 1)'),
    ('stop_f1_sphere', F1 + "; par.m = 2; par.ricb = 0", 'needs an inner core'),
    ('stop_f1_wu', F1 + "; par.m = 2; par.structure = 'Wu 2005'; par.r_out = 0.9", "needs structure 'Boussinesq'"),
    ('stop_f1_thermal', F1 + "; par.m = 2; par.thermal = 1; par.stratification = 'top hat'", "thermal = 0"),
    ('stop_f1_field', F1 + "; par.m = 2; " + MAG, "magnetic = 0"),
    ('stop_f1_bc2', F1 + "; par.m = 2; par.bco = 2", 'needs impenetrable walls'),
    ('stop_f1_drag', F1 + "; par.m = 2; par.bco = 0; par.kappa_cmb = 0.05", 'with a drag at the outer wall'),
    ('stop_f1_w0', F1 + "; par.m = 2; par.forcing_frequency = 0.0", 'needs forcing_frequency != 0'),
    ('stop_f2_m1', F2 + "; par.m = 1; par.symm = -1", 'needs m = 0 or 2 and symm = 1'),
    ('stop_f2_ricb', F2 + "; par.m = 2; par.ricb = 0.5", 'the fits of the tide hold for ricb'),
    ('stop_f2_slip', F2 + "; par.m = 2; par.bco = 0", 'needs no-slip walls'),
    ('stop_f2_twa', F2 + "; par.m = 2; " + MAG + "; " + TWAO, 'with a conducting solid behind a wall'),
    ('stop_f2_cheb', F2 + "; par.m = 2; " + MAG + "; " + CHEB, 'with a conducting solid behind a wall'),
    ('stop_f8_m2', F8 + "; par.m = 2", 'needs m = 0 and symm = 1'),
    ('stop_f8_sphere', F8 + "; par.ricb = 0; par.forcing_amplitude_icb = 0.1", 'has no inner core to librate'),
    ('stop_f8_w0', F8 + "; par.forcing_frequency = 0.0", 'needs forcing_frequency != 0'),
    ('stop_f8_inviscid', F8 + "; par.ViscosD = 0.0", 'with an inviscid fluid (ViscosD = 0) needs ricb = 0'),
    ('stop_f8_bc2', F8 + "; par.bci = 2", 'needs walls that stay in place'),
]
for name, extra, msg in STOPS:
    d = os.path.join(W, name)
    rc, out = run(d, setup(d, extra, 23 if 'm = 0' in extra or 'par.m' not in extra else 41), 's', check=False)
    ok = rc != 0 and msg in out
    report(name, ok, (out.strip().splitlines() or ['(no output)'])[-1][:90])

print('cic_forcings_check: ' + ('all cases ok' if not fail else 'FAILED'))
sys.exit(fail)
