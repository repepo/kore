#!/usr/bin/env python3
'''
usage: composition_check.py [WORKDIR]   (default: tests/composition/runs)

The composition (compositional = 1, NEW 2026-11-08, from conductive_IC) and the 'two zone' gradient. An incompressible fluid
('Boussinesq') in a rotating shell (ricb 0.35, Ek 1e-3, no-slip walls, m 2, N 96, lmax 49) unless said otherwise, built with
tests/tidal/mk_run.py's template on at most 2 ranks; the runs go NPAR at a time (default: half the processors, at most 4).
Five groups:

1. Against conductive_IC 2d84351, its Boussinesq composition ξ = -c: 'linear' (its 'internal', BeyonceC = BV2_comp 10,
   CompoD = Ecomp 1e-3, c = 0 at the walls); 'inverse square' (its 'differential', BeyonceC = BV2_comp ricb/(1 - ricb),
   zero flux at the walls); the entropy's 'two zone' (Beyonce = BV2 10, ThermaD 1e-3) with the composition 'linear'
   (BeyonceC 5, CompoD 1e-4, zero flux at the inner wall, c = 0 at the outer one). The first three eigenvalues to 1e-12
   (in practice 6e-14), and the budgets residσ, residω, resid𝑠, resid𝑐 below 1e-11.
2. The composition is a second entropy. With 'Boussinesq' (ρ0 = p0 = 1) a run with the composition alone and a run with
   the entropy alone, with the same numbers (BeyonceC = Beyonce, CompoD = ThermaD, comp_stratification = stratification,
   the same walls), solve the same equations: their eigenvalues (as sets, λ = 0 of the rotating solids left out), or the
   kinetic energy of a forced run, must agree to 1e-12, and composition.dat to thermal.dat (CE, Wadv, D, W, Wdr) to 1e-12
   of their largest value. Cases: a full sphere (also m 1, symm -1 with zero flux at the wall, and without diffusion),
   zero flux at the inner wall, a differential rotation (also without diffusion), a deformed body (a 'user' flattening,
   so that J - 1 is not 0, and the exact spheroid), a field over a conducting inner core under a resolved mantle, the
   rotating solids (also with a field and a thin conducting mantle), the drag of the topography at a stress-free outer
   wall (its N from N_c²), the libration (forcing 7), diffusivities that vary ('user', with κ and D swapped between the
   two runs, and 'step' with comp0 for therm0), hydro = 0 (the free decay) and the 'two zone' gradient. Also: with both
   scalars, equal in everything, s + c is one scalar with twice the gradient, so the run has the eigenvalues of the
   entropy alone with Beyonce doubled; and with 'Wu 2005' and no diffusion (where the rows are divided by ρ0 and by p0)
   the two runs have the same eigenvalues (not the same budgets, weighted by p0 and by ρ0).
3. Anelastic budgets: 'Wu 2005' (r_out 0.9, N 96) with the composition alone (c = 0 at the walls), and with both
   scalars, a 'two zone' composition with zero flux at the walls, D varying with the density (comp_pow) and a
   differential rotation ('Y20'): residσ, residω, resid𝑠, resid𝑐 below 1e-9.
4. ru_insane.py (exit 0: every symmetry and quadrature test below 1e-6) on the runs of 3, on the composition without
   diffusion, with a differential rotation and in the deformed body of 2.
5. The stops at the start of submatrices.py: compositional 2, no comp_stratification, 'inverse square' in a full
   sphere, a negative CompoD, a BeyonceC that is not finite, a wall condition 2, a 'two zone' step of width 0, a 'user'
   gradient that does not evaluate, diffusivity 'user' without 'D', forcing 1 with the composition, forcing 12 without
   its diffusion.

Exit code 1 if a check failed.
'''
import os, sys, shutil, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor

T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(T, '..', 'tidal'))
import mk_run   # noqa: E402  (the run builder of tests/tidal: 'Boussinesq', ricb 0.35, m 2, N 96)

W = sys.argv[1] if len(sys.argv) > 1 else os.path.join(T, 'runs')
os.makedirs(W, exist_ok=True)
LMAX_LINE = 'par.lmax = par.ellmax(par.ncpus, max(g, 4.*par.ncpus/par.N), par.m, par.N)'
assert LMAX_LINE in mk_run.T
py = sys.executable
NR = str(min(mk_run.NCPUS, 2))
NPAR = int(os.environ.get('NPAR', max(1, min(4, (os.cpu_count() or 2)//2))))


def setup(name, extra, lmax=None):
    '''the run directory (bin and parameters.py, as mk_run.build) on NR ranks, lmax from mk_run's g unless given'''
    d = os.path.join(W, name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d + '/bin')
    for f in os.listdir(mk_run.BIN):
        if os.path.isfile(mk_run.BIN + '/' + f) and f != 'parameters.py':
            shutil.copy(mk_run.BIN + '/' + f, d + '/bin/' + f)
    t = mk_run.T if lmax is None else mk_run.T.replace(LMAX_LINE, 'par.lmax = %d' % lmax)
    open(d + '/bin/parameters.py', 'w').write(t.format(ncpus=NR, extra=extra.replace('; ', '\n')))
    return d

def run(d, steps='savd'):
    '''the steps of a run in d: s(ubmatrices), a(ssemble), v (solve), d (spin_doctor), i (ru_insane); (returncode, output)
    of the first step that fails, or of the last'''
    cmds = {'s': [py, 'bin/submatrices.py', NR], 'a': mk_run.MPIEXEC.split() + ['-n', NR, py, 'bin/assemble.py'],
            'v': mk_run.MPIEXEC.split() + ['-n', NR, py, 'bin/solve.py'], 'd': [py, 'bin/spin_doctor.py', NR],
            'i': [py, 'bin/ru_insane.py']}
    out = ''
    for st in steps:
        p = subprocess.run(cmds[st], cwd=d, capture_output=True, text=True)
        out = p.stdout + p.stderr
        open(os.path.join(d, 'log_' + st), 'w').write(out)
        if p.returncode:
            return p.returncode, out
    for f in os.listdir(d):
        if f.endswith('.mtx'):
            os.remove(os.path.join(d, f))
    return 0, out

def build_all(jobs):
    '''jobs: {name: (extra, lmax)}; builds and runs them NPAR at a time; {name: (directory, error or None)}'''
    def one(item):
        name, (extra, lmax) = item
        d = setup(name, extra, lmax)
        rc, out = run(d)
        return name, (d, None if rc == 0 else out.strip()[-600:])
    with ThreadPoolExecutor(NPAR) as ex:
        return dict(ex.map(one, jobs.items()))

def eig(d):
    a = np.loadtxt(os.path.join(d, 'eigenvalues.dat'), ndmin=2)
    return a[:, 0] + 1j*a[:, 1]

def dat(d, f):
    return np.loadtxt(os.path.join(d, f), ndmin=2)

def resid(d):
    '''residσ, residω (flow.dat), resid𝑠 (thermal.dat) and resid𝑐 (composition.dat) of every solution, where written'''
    f = dat(d, 'flow.dat')
    out = {'σ': f[:, 16], 'ω': f[:, 15]}
    if os.path.exists(os.path.join(d, 'thermal.dat')):
        out['𝑠'] = dat(d, 'thermal.dat')[:, 3]
    if os.path.exists(os.path.join(d, 'composition.dat')):
        out['𝑐'] = dat(d, 'composition.dat')[:, 3]
    return {k: float(np.max(v)) for k, v in out.items()}

fail = 0
def report(name, ok, text):
    global fail
    fail |= not ok
    print('%-18s %s  (%s)' % (name, 'ok' if ok else 'FAIL', text))
    sys.stdout.flush()

def failed(name, res):
    d, err = res[name]
    if err:
        report(name, False, 'the run failed: ' + err.replace('\n', ' | ')[-300:])
    return bool(err)


TH = "par.thermal = 1; par.stratification = 'linear'; par.Beyonce = 10.; par.ThermaD = 1e-3; par.bci_thermal = 0; par.bco_thermal = 0"
CO = ("par.compositional = 1; par.comp_stratification = 'linear'; par.BeyonceC = 10.; par.CompoD = 1e-3; "
      "par.bci_compositional = 0; par.bco_compositional = 0")
MAG = "par.magnetic = 1; par.cnorm = 'mag_energy'; par.Hendrik = 1e-3; par.MagnetD = 1e-3"
SOL = ("par.N = 48; par.m = 0; par.symm = 1; par.rotdyn = 1; par.gTorque = 1e-3; par.MoIZ_M = 1.0; par.MoIZ_IC = 0.05; "
       "par.OmgtauIC = 50.0; par.rtau = -0.04; par.itau = 0.13")
REK = "par.topo_drag = 'rekier'; par.topo_hrms = 0.02; par.topo_lambda0 = 0.431; par.topo_H = 0.8"
U1, U2 = "'1 + 0.5*r**2'", "'2 - r'"
WU = "par.structure = 'Wu 2005'; par.r_out = 0.9"

# 1. against conductive_IC 2d84351 (ricb 0.35, m 2, Ek 1e-3, N 96, lmax 49): its first three eigenvalues
CIC = {
    'cic_linear': (CO, [-0.353479897948153+1.0917965319842928j, -0.23196706174876022+1.3109667450746054j,
                        -0.5407678012277615+0.9613829492392014j]),
    'cic_inverse_square': ("par.compositional = 1; par.comp_stratification = 'inverse square'; par.BeyonceC = 10.*0.35/0.65; "
                           "par.CompoD = 1e-3; par.bci_compositional = 1; par.bco_compositional = 1",
                           [-0.3605396316430414+1.2911203278967895j, -0.5220203745483062+1.1169160212642348j,
                            -0.24853202495938637+1.6036038992764252j]),
    'cic_two_zone': ("par.thermal = 1; par.stratification = 'two zone'; par.Beyonce = 10.; par.ThermaD = 1e-3; "
                     "par.bci_thermal = 0; par.bco_thermal = 0; par.compositional = 1; par.comp_stratification = 'linear'; "
                     "par.BeyonceC = 5.; par.CompoD = 1e-4; par.bci_compositional = 1; par.bco_compositional = 0",
                     [-0.35154202854658245+1.0702899892547828j, -0.22014972126985713+1.2969702666008978j,
                      -0.50006738703808+0.9258260155571156j]),
}

# 2. the entropy alone against the composition alone: (both runs, the entropy's run, the composition's run)
SAME = {
    'sphere':        ("par.ricb = 0.0", "", ""),
    'sphere_flux':   ("par.ricb = 0.0; par.m = 1; par.symm = -1", "par.bco_thermal = 1", "par.bco_compositional = 1"),
    'sphere_nodiff': ("par.ricb = 0.0; par.N = 48", "par.ThermaD = 0", "par.CompoD = 0"),
    'flux_inner':    ("", "par.bci_thermal = 1", "par.bci_compositional = 1"),
    'diff_rot':      ("par.diff_rot = 1", "", ""),
    'diff_rot_nodiff': ("par.diff_rot = 1; par.diff_rot_type = 'Y20'", "par.ThermaD = 0", "par.CompoD = 0"),
    'deformed':      ("par.centrifugal = 1; par.centrifugal_shape = 'user'; par.centrifugal_profile = '0.05*(1 + r**2)'; "
                      "par.N = 48", "", ""),
    'spheroid':      ("par.centrifugal = 1; par.centrifugal_shape = 'spheroid'; par.centrifugal_f = 0.3; par.N = 40; g = 0.8",
                      "", ""),
    'field':         (MAG + "; par.B0 = 'axial'; par.innercore = 'conducting, Chebys'; par.mu_i2o = 1.; par.N_cic = 24; "
                      "par.mantle = 'resolved'; par.r_surface = 1.3; par.N_mantle = 24; par.sigma_mantle = 0.5; par.N = 48",
                      "", ""),
    'solids':        (SOL, "", ""),
    'solids_field':  (SOL + "; " + MAG + "; par.B0 = 'G21 dipole'; par.mantle = 'TWA'; par.c_cmb = 0.0; par.c1_cmb = 0.02",
                      "", ""),
    'drag':          ("par.N = 48; par.m = 1; par.symm = -1; par.bco = 0; par.itau = 0.98; " + REK, "", ""),
    'libration':     ("par.N = 48; par.m = 0; par.symm = 1; par.forcing = 7; par.forcing_frequency = 0.7; "
                      "par.forcing_amplitude_cmb = 0.3; par.forcing_amplitude_icb = 0.3", "", ""),
    'user':          ("par.diffusivity = 'user'",
                      "par.diffusivity_profiles = {'nu': '1 + 0*r', 'kappa': %s, 'D': %s}" % (U1, U2),
                      "par.diffusivity_profiles = {'nu': '1 + 0*r', 'kappa': %s, 'D': %s}" % (U2, U1)),
    'step':          ("par.diffusivity = 'step'", "par.therm0 = 0.3", "par.comp0 = 0.3"),
    'decay':         ("par.hydro = 0; par.rtau = -0.02; par.itau = 0.0; par.nev = 4", "", ""),
    'two_zone':      ("par.bci = 0", "par.stratification = 'two zone'; par.two_zone = [0.6, 0.15]; par.bci_thermal = 1",
                      "par.comp_stratification = 'two zone'; par.comp_two_zone = [0.6, 0.15]; par.bci_compositional = 1"),
    'anelastic_nodiff': (WU, "par.ThermaD = 0", "par.CompoD = 0"),
}

# 3. anelastic budgets
BUD = {
    'wu_comp': WU + "; " + CO,
    'wu_both': (WU + "; " + TH + "; par.compositional = 1; par.comp_stratification = 'two zone'; par.BeyonceC = 5.; "
                "par.CompoD = 3e-4; par.bci_compositional = 1; par.bco_compositional = 1; par.comp_pow = 0.5; "
                "par.diff_rot = 1; par.diff_rot_type = 'Y20'; par.diff_rot_ampl = 0.1"),
}

jobs = {k: (v[0], 49) for k, v in CIC.items()}
for k, (common, th, co) in SAME.items():
    jobs['th_' + k] = ('; '.join(x for x in (TH, th, common) if x), None)
    jobs['co_' + k] = ('; '.join(x for x in (CO, co, common) if x), None)
jobs['both'] = (TH + "; " + CO, None)
jobs['th_doubled'] = (TH + "; par.Beyonce = 20.", None)
jobs.update({k: (v, None) for k, v in BUD.items()})
res = build_all(jobs)

print('1. against conductive_IC 2d84351')
for name, (extra, ref) in CIC.items():
    if failed(name, res):
        continue
    d = res[name][0]
    lam, ref = eig(d), np.array(ref)
    err = max(abs(lam - x).min() for x in ref)/abs(ref).max()
    r = resid(d)
    report(name, err < 1e-12 and max(r.values()) < 1e-11,
           'eigenvalues %.1e from conductive_IC; %s' % (err, ', '.join('resid%s %.1e' % kv for kv in r.items())))

print('2. the composition is a second entropy')
pairs = [(k, 'th_' + k, 'co_' + k) for k in SAME] + [('both scalars', 'th_doubled', 'both')]
for name, a, b in pairs:
    if failed(a, res) or failed(b, res):
        continue
    da, db = res[a][0], res[b][0]
    if 'forcing = 7' in jobs[b][0]:
        x, y = dat(da, 'flow.dat')[:, 0], dat(db, 'flow.dat')[:, 0]
        err, what = abs(x - y).max()/abs(x).max(), 'KE'
    else:
        x, y = eig(da), eig(db)
        x = x[abs(x) > 1e-8]
        err, what = max(abs(y - v).min() for v in x)/abs(x).max(), 'eigenvalues'
    text = '%s %.1e' % (what, err)
    ok = err < 1e-12
    if name != 'both scalars' and 'Wu 2005' not in jobs[b][0]:   # (anelastic: the budgets are weighted by p0 and by ρ0)
        t, c = dat(da, 'thermal.dat')[:, [0, 1, 2, 4, 6]], dat(db, 'composition.dat')[:, [0, 1, 2, 4, 6]]
        if t.shape == c.shape:
            e2 = abs(t - c).max()/abs(t).max()
            ok &= e2 < 1e-12
            text += ', composition.dat vs thermal.dat %.1e' % e2
        else:
            text += ' (composition.dat not compared: %d and %d solutions)' % (len(c), len(t))
    report(name, ok, text)

print('3. anelastic budgets')
for name in BUD:
    if failed(name, res):
        continue
    r = resid(res[name][0])
    report(name, max(r.values()) < 1e-9, ', '.join('resid%s %.1e' % kv for kv in r.items()))

print('4. ru_insane.py')
for name in ('wu_comp', 'wu_both', 'co_sphere_nodiff', 'co_diff_rot', 'co_deformed', 'co_anelastic_nodiff'):
    if res[name][1]:
        report('ru_insane ' + name, False, 'no run')
        continue
    rc, out = run(res[name][0], 'i')
    last = [ln for ln in out.splitlines() if ln.startswith('ru_insane.py:')]
    report('ru_insane ' + name, rc == 0 and 'PASS' in out, last[-1][14:110] if last else out.strip()[-100:])

print('5. stops')
STOPS = [
    ('stop_flag',       "par.compositional = 2", 'compositional 2: 1 (the composition is solved) or 0'),
    ('stop_none',       CO + "; par.comp_stratification = None", 'comp_stratification None: set one of'),
    ('stop_inv_sphere', CO + "; par.comp_stratification = 'inverse square'; par.ricb = 0.0",
                        "comp_stratification 'inverse square' needs ricb > 0"),
    ('stop_CompoD',     CO + "; par.CompoD = -1e-3", 'the compositional diffusion factor is >= 0'),
    ('stop_BeyonceC',   CO + "; par.BeyonceC = float('nan')", 'BeyonceC = nan: a finite number'),
    ('stop_wall',       CO + "; par.bco_compositional = 2", '0 (fixed composition) or 1 (zero flux)'),
    ('stop_two_zone',   CO + "; par.comp_stratification = 'two zone'; par.comp_two_zone = [0.7, 0.0]", 'with a width h > 0'),
    ('stop_user',       CO + "; par.comp_stratification = 'user'; par.comp_stratification_profile = 'r**'",
                        'comp_stratification_profile'),
    ('stop_user_D',     CO + "; par.diffusivity = 'user'; par.diffusivity_profiles = {'nu': '1 + 0*r', 'kappa': '1 + 0*r'}",
                        "'D': '...'"),
    ('stop_forcing1',   CO + "; par.forcing = 1; par.forcing_frequency = 0.37", 'compositional = 0'),
    ('stop_forcing12',  CO + "; par.CompoD = 0; par.m = 0; par.forcing = 12; par.forcing_frequency = 0.0; "
                        "par.forcing_amplitude_cmb = 1.0", 'forcing 12 with compositional = 1 needs CompoD > 0'),
]
for name, extra, msg in STOPS:
    rc, out = run(setup(name, extra, 49 if 'par.m = 0' not in extra else 31), 's')
    ok = rc != 0 and msg in out
    report(name, ok, (out.strip().splitlines() or ['(no output)'])[-1][:100])

print('composition_check: ' + ('all cases ok' if not fail else 'FAILED'))
sys.exit(fail)
