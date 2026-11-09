#!/usr/bin/env python3
'''
usage: tools_check.py [WORKDIR]   (default: tests/tools/runs)

Checks of the files of a run and of the run tools (docs/output.md, docs/running.md):

1. The solution files (par.field_format, bin/field_io.py). Three runs, each solved with field_format 'npy' and 'text':
   an eigenvalue problem with every field (entropy, composition, a field over a conducting inner core and a resolved
   mantle), one of the rotating solids (rotdyn) with a thin wall, and a forced run with a field followed by the kinematic
   induction (just_induction.py, solve_ind.py). The two formats hold the same numbers to the last bit, and spin_doctor's
   files are the same (params.dat but its run time). Solving again in the other format leaves only its files.
   field_io.load reads one solution or a list of them. bgflow.py writes the same background from a forcing-12 run in
   either format.
2. The parameter file (par.export_json, par.load_json): parameters of every kind (numpy scalars and arrays, complex
   numbers, tuples, dictionaries with integer keys, None, nan, names that kore does not define) come back with their
   values and types, and a run whose parameters.py reads the file gives the matrices, solutions and spin_doctor files of
   the run that wrote it.
3. Eigenmode tracking (track_target): 2 writes the eigenvalue of the solution with the smallest KP/KT; 1 aims the solver
   at the eigenvalue in the file, far from rtau, itau, and writes back the nearest one.
4. tools/ramp.sh: a ramp in ViscosD with track_target = 1 runs one step per value; parameters.py ends with the last value
   (set by tools/setpar.py in a line of several statements); each step's tracked eigenvalue is the one nearest the step
   before's. A ramp whose step finds no eigenvalue stops. tools/reap.sh collects two folders, and tools/get_data.py
   reads them, and a params.dat with the expressions of a 'user' rotation, with named columns equal to the .dat files.
5. tools/topo_iterate.sh: the frequency of the drag (topo_drag = 'rekier', topo_omega = None) converges to the mode's
   own, a fixed point: the run with topo_omega set to it gives the same eigenvalue.
6. tools/setpar.py: in place of an assignment (its comment kept), of one over two lines, of one among others on a line,
   and as a new last line; nothing else changes. It warns when it sets Ek where par.set_scales(), which alone reads it,
   is not called after it.
7. The column names of tools/get_data.py are those of spin_doctor.py's tables (read from its source).
8. The stops: field_format, track_target (its values, with a forcing, 2 without the flow, 1 without the file),
   spin_doctor without solution files, tools/setpar.py with a value that is not python, tools/ramp.sh outside kore,
   tools/topo_iterate.sh without the file track_target or without its parameters.
9. The plotting tools (tests/tools/plot_check.py, with matplotlib), on the run with every field, on a full sphere with
   the density of 'Wu 2005', the entropy and a field, on the forced run and on a run in the text format:
   tools/load_solution.py unpacks the coefficients of utils4pp.expand_sol, and its inner core continues the fluid's f,
   df/dr and g at the inner boundary; tools/plot_rad_con_tor_profile.py's kinetic energy is KE; tools/plot_field.py
   draws the flow, the forces, the buoyancy, the curls and the field, whose components are those of its coefficients.

The runs solve on one rank, where the solutions are reproducible to the last bit. Exit code 1 if a check failed.
'''
import ast, json, os, shutil, subprocess, sys
import numpy as np

T = os.path.dirname(os.path.abspath(__file__))
KORE = os.path.normpath(os.path.join(T, '..', '..'))
sys.path.insert(0, os.path.join(T, '..', 'tidal'))
import mk_run   # noqa: E402  (the run builder of tests/tidal: 'Boussinesq', ricb 0.35)
sys.path.insert(0, os.path.join(KORE, 'bin'))
import field_io   # noqa: E402
from params_default import default_params   # noqa: E402
sys.path.insert(0, os.path.join(KORE, 'tools'))

W = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(T, 'runs')
os.makedirs(W, exist_ok=True)
py = sys.executable
MPI = mk_run.MPIEXEC.split()
FIELDS = ('flow', 'thermal', 'composition', 'magnetic', 'magnetic_ic', 'magnetic_mantle', 'rotdyn')
ENV = dict(os.environ, ncpus='1', ranks='1')   # for the tools' scripts


def setup(d, extra, tools=False, tail=''):
    '''the run folder d: kore's bin (and the tools' scripts) and a parameters.py, mk_run's defaults at N 32, then extra'''
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d + '/bin')
    for f in os.listdir(mk_run.BIN):
        if os.path.isfile(mk_run.BIN + '/' + f) and f != 'parameters.py':
            shutil.copy(mk_run.BIN + '/' + f, d + '/bin/' + f)
    if tools:
        os.makedirs(d + '/tools')
        for f in ('ramp.sh', 'setpar.py', 'topo_iterate.sh'):
            shutil.copy(os.path.join(KORE, 'tools', f), d + '/tools/' + f)
    t = mk_run.T.format(ncpus=1, extra=('par.N = 32; ' + extra).replace('; ', '\n'))
    open(d + '/bin/parameters.py', 'w').write(t + tail)
    return d


def run(d, steps='savd', check=True):
    '''the steps of a run in d: s(ubmatrices), a(ssemble), v (solve), d (spin_doctor), j (just_induction), i (solve_ind);
    (returncode, output) of the last'''
    cmds = {'s': [py, 'bin/submatrices.py', '1'], 'a': MPI + ['-n', '1', py, 'bin/assemble.py'],
            'v': MPI + ['-n', '1', py, 'bin/solve.py'], 'd': [py, 'bin/spin_doctor.py', '1'],
            'j': [py, 'bin/just_induction.py'], 'i': MPI + ['-n', '1', py, 'bin/solve_ind.py']}
    out = ''
    for st in steps:
        p = subprocess.run(cmds[st], cwd=d, capture_output=True, text=True)
        out = p.stdout + p.stderr
        open(os.path.join(d, 'log_' + st), 'a').write(out)
        if p.returncode:
            if check:
                sys.exit('%s: step %s failed\n%s' % (d, st, out[-3000:]))
            return p.returncode, out
    return 0, out


def build(name, extra, steps='savd', **kw):
    d = setup(os.path.join(W, name), extra, **kw)
    run(d, steps)
    return d


def sh(d, args, check=False):
    '''a command of the tools in the folder d: (returncode, output)'''
    p = subprocess.run(args, cwd=d, capture_output=True, text=True, env=ENV)
    out = p.stdout + p.stderr
    if check and p.returncode:
        sys.exit('%s: %s failed\n%s' % (d, ' '.join(args), out[-3000:]))
    return p.returncode, out


fail = 0
def report(name, ok, text):
    global fail
    fail |= not ok
    print('%-18s %s  (%s)' % (name, 'ok' if ok else 'FAIL', text), flush=True)


def dat_rows(f):
    '''the rows of a .dat file of spin_doctor as lists of words, without the # note that ends some rows of params.dat'''
    return [ln.partition('#')[0].split() for ln in open(f)]


def same_dat(a, b):
    '''the .dat files of spin_doctor in folders a and b that differ (params.dat but its column 18, the run time)'''
    bad = []
    for f in sorted(set(x for x in os.listdir(a) + os.listdir(b) if x.endswith('.dat') and x != 'timing.dat')):
        fa, fb = os.path.join(a, f), os.path.join(b, f)
        if not (os.path.exists(fa) and os.path.exists(fb)):
            bad.append(f + ' (missing)')
        elif f == 'params.dat':
            ra, rb = dat_rows(fa), dat_rows(fb)
            if [r[:17] + r[18:] for r in ra] != [r[:17] + r[18:] for r in rb]:
                bad.append(f)
        elif open(fa).read() != open(fb).read():
            bad.append(f)
    return bad


def same_fields(a, b):
    '''(the fields written in folder a, those that differ from folder b's)'''
    names = [n for n in FIELDS if field_io.exists(n, a) or field_io.exists(n, b)]
    bad = []
    for n in names:
        x, y = field_io.load(n, folder=a), field_io.load(n, folder=b)
        if x is None or y is None or x.shape != y.shape or not np.array_equal(x, y):
            bad.append(n)
    return names, bad


def eigs(d):
    e = np.loadtxt(os.path.join(d, 'eigenvalues.dat'), ndmin=2)
    return e[:, 0] + 1j*e[:, 1]


def target(d):
    t = np.loadtxt(os.path.join(d, 'track_target')).ravel()
    return complex(t[0], t[1])


TH = "par.thermal = 1; par.stratification = 'linear'; par.Beyonce = 10.; par.ThermaD = 1e-3; par.bci_thermal = 0; par.bco_thermal = 0"
CO = ("par.compositional = 1; par.comp_stratification = 'linear'; par.BeyonceC = 5.; par.CompoD = 3e-4; "
      "par.bci_compositional = 0; par.bco_compositional = 0")
MAG = "par.magnetic = 1; par.cnorm = 'mag_energy'; par.Hendrik = 1e-3; par.MagnetD = 1e-3; par.B0 = 'axial'"
CHEB = "par.innercore = 'conducting, Chebys'; par.N_cic = 16"
MAN = "par.mantle = 'resolved'; par.r_surface = 1.3; par.N_mantle = 16; par.sigma_mantle = 0.5"
SOL = ("par.m = 0; par.symm = 1; par.rotdyn = 1; par.gTorque = 1e-3; par.MoIZ_M = 1.0; par.MoIZ_IC = 0.05; "
       "par.OmgtauIC = 50.0; par.rtau = -0.04; par.itau = 0.13")
TWA = "par.mantle = 'TWA'; par.c_cmb = 0.0; par.c1_cmb = 0.02"
LIB = "par.m = 0; par.symm = 1; par.forcing = 7; par.forcing_frequency = 0.7; par.forcing_amplitude_cmb = 0.3"
REK = "par.topo_drag = 'rekier'; par.topo_hrms = 0.02; par.topo_lambda0 = 0.431; par.topo_H = 0.8"
SPIN = "par.m = 1; par.symm = -1; par.nev = 4"   # the spin-over mode and its neighbours
TEXT = "par.field_format = 'text'"


# ----------------------------------------------------------------------------------------------------------------------
# 1. the solution files in both formats
for name, extra, steps in (('every', '; '.join((TH, CO, MAG, CHEB, MAN)), 'savd'),
                           ('rotdyn', SOL + '; ' + MAG + '; ' + TWA, 'savd'),
                           ('induction', LIB + '; ' + MAG, 'savjid')):
    a = build('fmt_%s_npy' % name, extra, steps)
    b = build('fmt_%s_text' % name, extra + '; ' + TEXT, steps)
    names, bad = same_fields(a, b)
    files = (all(field_io.exists(n, a) and not os.path.exists(field_io.files(n, 'text', a)[0]) for n in names) and
             all(not os.path.exists(field_io.files(n, 'npy', b)[0]) for n in names))
    report('format_' + name, names and not bad and files and not same_dat(a, b),
           '%s; %s; spin_doctor: %s' % (', '.join(names), 'differ: ' + ', '.join(bad) if bad else 'bitwise equal',
                                        ', '.join(same_dat(a, b)) or 'the same files'))

# solving again in the other format leaves only its files, with the same numbers; one solution, or a list of them
a, b = os.path.join(W, 'fmt_every_npy'), os.path.join(W, 'fmt_every_text')
sh(b, [py, os.path.join(KORE, 'tools', 'setpar.py'), 'field_format', "'npy'", 'bin/parameters.py'], check=True)
run(b, 'v')
names, bad = same_fields(a, b)
left = [f for f in os.listdir(b) if f.endswith('.field')]
x = field_io.load('flow', folder=a)
cols = (np.array_equal(field_io.load('flow', 1, folder=a), x[:, 1]) and
        np.array_equal(field_io.load('flow', [0, 2], folder=a), x[:, [0, 2]]) and
        field_io.load('flow', 1, folder=a).ndim == 1 and field_io.load('nothing', folder=a) is None)
report('format_switch', not bad and not left and cols,
       'text then npy in one folder: %d text files left, %s; load of one and of two solutions %s' %
       (len(left), 'the same numbers' if not bad else 'differ: ' + ', '.join(bad), 'right' if cols else 'WRONG'))

# the writer of the background flow reads the solution of a forcing-12 run in either format
F12 = "par.m = 0; par.symm = 1; par.forcing = 12; par.forcing_frequency = 0.0; par.forcing_amplitude_icb = 1.0; " \
      "par.forcing_amplitude_cmb = 0.0"
bg = []
for fmt, extra in (('npy', F12), ('text', F12 + '; ' + TEXT)):
    d = build('bgw_' + fmt, extra, 'sav')
    sh(d, [py, 'bin/bgflow.py', '8'], check=True)
    bg.append(np.load(d + '/bgflow.npz'))
same = sorted(bg[0].files) == sorted(bg[1].files) and all(np.array_equal(bg[0][k], bg[1][k]) for k in bg[0].files)
report('format_bgflow', same, 'bgflow.npz of the two formats %s' % ('bitwise equal' if same else 'DIFFER'))


# ----------------------------------------------------------------------------------------------------------------------
# 2. the parameter file
def same_value(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc')
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return list(a) == list(b) and all(same_value(a[k], b[k]) for k in a)
    if isinstance(a, (float, np.floating)) and np.isnan(a):
        return bool(np.isnan(b))
    return a == b


p = default_params()
p.set_scales()
for k, v in {'ricb': np.float64(0.35), 'N': np.int64(32), 'aux1': np.float32(0.5), 'rtau': complex(-0.1, 2.0),
             'diff_rot_profiles': {0: '0.3*r**2', 2: '-0.2*r**2'}, 'solar_rz_steps': (0.7, 0.71, np.float64(0.02)),
             'B0_param': np.array([[1.0, 2.5], [np.nan, -4.0]]), 'lvals': np.arange(3, dtype=np.int32),
             'topo_omega': None, 'aux5': float('nan'), 'my_flag': np.bool_(True), 'my_c': np.complex128(1 - 2j),
             'my_list': [1, (2, 'three'), {'a': np.int16(4), 5: None}], 'bgflow_profiles': {'T1': '0.5*r'}}.items():
    setattr(p, k, v)
p.export_json(os.path.join(W, 'roundtrip.json'))
q = default_params()
q.load_json(os.path.join(W, 'roundtrip.json'))
bad = [k for k in vars(p) if not same_value(getattr(p, k), getattr(q, k, '<absent>'))]
report('json_roundtrip', not bad and set(vars(p)) == set(vars(q)),
       '%d parameters, %s' % (len(vars(p)), 'every value and type back' if not bad else 'differ: ' + ', '.join(bad)))

# a run from the file, against the run that wrote it (a 'user' rotation: a dictionary with integer keys)
DRU = ("par.diff_rot = 1; par.diff_rot_type = 'user'; par.diff_rot_ampl = 1.0; par.thermal_wind = 1; "
       "par.diff_rot_profiles = {0: '0.3*r**2', 2: '-0.2*r**2'}")
a = build('json_write', TH + '; ' + DRU + '; par.m = 1', tail="par.export_json('parameters.json')\n")
b = setup(os.path.join(W, 'json_read'), '')
open(b + '/bin/parameters.py', 'w').write("from params_default import default_params\npar = default_params()\n"
                                          "par.load_json(%r)\n" % os.path.join(a, 'parameters.json'))
run(b)
mats = all(np.array_equal(np.load(a + '/%s.npz' % M)[k], np.load(b + '/%s.npz' % M)[k])
           for M in ('A', 'B') for k in ('data', 'indices', 'indptr', 'shape'))
names, bad = same_fields(a, b)
report('json_run', mats and names and not bad and not same_dat(a, b),
       'matrices %s, solutions %s, spin_doctor %s' % ('equal' if mats else 'DIFFER', 'equal' if not bad else 'DIFFER',
                                                     ', '.join(same_dat(a, b)) or 'the same files'))


# ----------------------------------------------------------------------------------------------------------------------
# 3. eigenmode tracking
d2 = build('track2', SPIN + '; par.track_target = 2')
flo = np.loadtxt(d2 + '/flow.dat', ndmin=2)
lam, j = eigs(d2), int(np.argmin(flo[:, 1]/flo[:, 2]))
report('track_2', target(d2) == lam[j], 'track_target holds solution %d of %d, the smallest KP/KT %.3e: %s' %
       (j, len(lam), flo[j, 1]/flo[j, 2], target(d2)))

d1 = setup(os.path.join(W, 'track1'), SPIN + '; par.track_target = 1; par.rtau = 0.0; par.itau = 3.0')
shutil.copy(d2 + '/track_target', d1)
run(d1)
z0, lam = target(d2), eigs(d1)
k = int(np.argmin(abs(lam - z0)))
report('track_1', abs(lam[k] - z0) < 1e-10*abs(z0) and target(d1) == lam[k],
       'target %s from the file (rtau, itau 0, 3): eigenvalue %s found again (%.1e), written back' %
       (z0, lam[k], abs(lam[k] - z0)/abs(z0)))


# ----------------------------------------------------------------------------------------------------------------------
# 4. ramp, reap, get_data
dr = setup(os.path.join(W, 'ramp_a'), SPIN + '; par.track_target = 1', tools=True)
shutil.copy(d2 + '/track_target', dr)
visc = ['1e-3', '8e-4', '6e-4']
rc, out = sh(dr, ['tools/ramp.sh', 'ViscosD'] + visc)
open(dr + '/ramp.log', 'w').write(out)
ok, text = rc == 0, 'exit %d' % rc
if ok:
    tree = ast.parse(open(dr + '/bin/parameters.py').read())
    last = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and
            ast.unparse(n.targets[0]) == 'par.ViscosD']
    line = [ln for ln in open(dr + '/bin/parameters.py') if 'par.ViscosD' in ln]
    pv = np.array([float(r[12]) for r in dat_rows(dr + '/params.dat')])
    lam, z = eigs(dr), target(d2)
    steps = []
    for v in visc:
        rows = np.flatnonzero(pv == float(v))
        z = lam[rows[np.argmin(abs(lam[rows] - z))]]
        steps.append(z)
    ok = (last == [6e-4] and len(line) == 1 and line[0].startswith('par.Gaspard') and target(dr) == z and
          all(np.isin(pv, [float(v) for v in visc])) and len(set(pv)) == 3)
    text = 'steps %s; parameters.py: %s; tracked %s' % (
        ', '.join(str(np.count_nonzero(pv == float(v))) for v in visc) + ' rows', line[0].strip(),
        ' -> '.join('%.6f%+.6fi' % (s.real, s.imag) for s in steps))
report('ramp', ok, text)

db = setup(os.path.join(W, 'ramp_b'), SPIN, tools=True)
rc, out = sh(db, ['tools/ramp.sh'])
report('ramp_once', rc == 0 and os.path.exists(db + '/params.dat') and not os.path.exists(db + '/track_target'),
       'no parameter: the case as it is, exit %d' % rc)

dc = setup(os.path.join(W, 'ramp_c'), SPIN + '; par.maxit = 1; par.tol = 1e-30', tools=True)
rc, out = sh(dc, ['tools/ramp.sh', 'ViscosD', '1e-3', '8e-4'])
report('ramp_stop', rc == 1 and 'ramp stopped' in out and 'par.ViscosD = 1e-3' in out and
       not os.path.exists(dc + '/params.dat'), 'no converged eigenvalue at the first step: exit %d, %s' %
       (rc, out.strip().splitlines()[-1] if out.strip() else ''))

try:
    import pandas   # noqa: F401
    import get_data
except ImportError as e:
    report('get_data', False, 'tools/get_data.py needs pandas: %s' % e)
else:
    rc, out = sh(W, [os.path.join(KORE, 'tools', 'reap.sh'), 'ramp_'])
    df = get_data.load_results(os.path.join(W, 'ramp_'))
    par_ = [r for d in (dr, db) for r in dat_rows(d + '/params.dat')]
    flo = np.vstack([np.loadtxt(d + '/flow.dat', ndmin=2) for d in (dr, db)])
    eig = np.vstack([np.loadtxt(d + '/eigenvalues.dat', ndmin=2) for d in (dr, db)])
    cols = get_data.PARAMS + get_data.FLOW + get_data.EIGENVALUES
    ok = (rc == 0 and list(df.columns) == cols and len(df) == len(par_) == len(flo) == len(eig) and
          np.array_equal(df['ViscosD'].values, [float(r[12]) for r in par_]) and
          np.array_equal(df['N'].values, [int(r[15]) for r in par_]) and
          np.array_equal(df[get_data.FLOW].values, flo, equal_nan=True) and   # (irreg0 and residL are nan here)
          np.array_equal(df[get_data.EIGENVALUES].values, eig))
    rc2, out2 = sh(W, [py, os.path.join(KORE, 'tools', 'get_data.py'), 'ramp_'])
    csv = pandas.read_csv(os.path.join(W, 'ramp_.csv'), index_col=0)
    ok = ok and rc2 == 0 and csv.shape == df.shape
    # a params.dat whose rows end with the expressions of a 'user' rotation
    one = get_data.load_result(os.path.join(W, 'json_write'))
    ok_user = (list(one.columns[:len(get_data.PARAMS)]) == get_data.PARAMS and one['diff_rot'].iloc[0] == 1 and
               len(one) == len(dat_rows(os.path.join(W, 'json_write', 'params.dat'))))
    report('reap_get_data', ok and ok_user, '%d rows of 2 folders, %d named columns equal to the .dat files; %s; '
           "a 'user' rotation's params.dat %s" % (len(df), df.shape[1], out2.strip(), 'read' if ok_user else 'WRONG'))


# ----------------------------------------------------------------------------------------------------------------------
# 5. the frequency of the drag made the mode's own
DRAG = TH + "; par.m = 1; par.symm = -1; par.bco = 0; par.nev = 3; " + REK
dt = setup(os.path.join(W, 'topo_iter'), DRAG + '; par.track_target = 1', tools=True)
np.savetxt(dt + '/track_target', [[0.0, 0.98]])
rc, out = sh(dt, ['tools/topo_iterate.sh', '1e-10', '20'])
open(dt + '/topo_iterate.log', 'w').write(out)
if rc == 0:
    z = target(dt)
    df_ = setup(os.path.join(W, 'topo_fixed'), DRAG + '; par.topo_omega = %r; par.rtau = %r; par.itau = %r' %
                (abs(z.imag), z.real, z.imag))
    run(df_)
    lam = eigs(df_)
    e = np.min(abs(lam - z))/abs(z)
    n = sum(ln.startswith('step') for ln in out.splitlines())
    report('topo_iterate', e < 1e-9, 'converged in %d steps to %s; with topo_omega = %.10f fixed: %.1e' %
           (n, z, abs(z.imag), e))
else:
    report('topo_iterate', False, 'exit %d: %s' % (rc, out[-500:]))


# ----------------------------------------------------------------------------------------------------------------------
# 6. tools/setpar.py
f = os.path.join(W, 'setpar_parameters.py')
src = ("from params_default import default_params\npar = default_params()\npar.set_scales()\n"
       "par.N = 32   # radial (ρ, Ω)\n"
       "par.Gaspard = 1.; par.ViscosD = 1e-3; par.ThermaD = 1e-3\n"
       "par.diff_rot_profiles = {0: '0.3*r**2',\n                         2: '-0.2*r**2'}   # two lines\n"
       "s = 'par.N = 1'\n#par.lmax = 10\n")
want = ("from params_default import default_params\npar = default_params()\npar.set_scales()\n"
        "par.N = 48   # radial (ρ, Ω)\n"
        "par.Gaspard = 1.; par.ViscosD = 5e-4; par.ThermaD = 1e-3\n"
        "par.diff_rot_profiles = {0: 'r'}   # two lines\n"
        "s = 'par.N = 1'\n#par.lmax = 10\npar.lmax = 20\n")
open(f, 'w').write(src)
for name, value in (('N', '48'), ('ViscosD', '5e-4'), ('diff_rot_profiles', "{0: 'r'}"), ('lmax', '20')):
    sh(W, [py, os.path.join(KORE, 'tools', 'setpar.py'), name, value, f], check=True)
got = open(f).read()
report('setpar', got == want, 'one line with its comment, one of several statements, two lines, a new last line%s' %
       ('' if got == want else ':\n' + got))
g = os.path.join(W, 'setpar_scales.py')   # Ek, which only set_scales() reads: it holds before the call, not after
open(g, 'w').write("par.Ek = 1e-3\npar.set_scales()\npar.ViscosD = 1e-3\n")
_, out1 = sh(W, [py, os.path.join(KORE, 'tools', 'setpar.py'), 'Ek', '1e-4', g], check=True)
open(g, 'w').write("par.set_scales()\npar.ViscosD = 1e-3\n")
_, out2 = sh(W, [py, os.path.join(KORE, 'tools', 'setpar.py'), 'Ek', '1e-4', g], check=True)
report('setpar_scales', 'warning' not in out1 and 'only par.set_scales() reads par.Ek' in out2,
       'Ek before set_scales(): no warning; after it: %s' % (out2.strip()[:60] + '...' if out2 else 'NO WARNING'))


# ----------------------------------------------------------------------------------------------------------------------
# 7. the column names of get_data.py and spin_doctor.py's tables
def names_of(node):
    '''the names, the first elements of the tuples, of a table of spin_doctor.py'''
    if isinstance(node, ast.List):
        return [ast.literal_eval(e.elts[0]) for e in node.elts]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return names_of(node.left) + names_of(node.right)
    if isinstance(node, ast.IfExp):   # columns written only in some runs
        return names_of(node.body) + names_of(node.orelse)
    if isinstance(node, ast.ListComp):   # [(name % expr, ...) for k in range(n)]
        g = node.generators[0]
        assert isinstance(g.iter, ast.Call) and g.iter.func.id == 'range' and len(g.iter.args) == 1
        code = compile(ast.Expression(node.elt.elts[0]), 'spin_doctor.py', 'eval')
        return [eval(code, {}, {g.target.id: k}) for k in range(ast.literal_eval(g.iter.args[0]))]
    raise ValueError('a table of spin_doctor.py that this check does not read: ' + ast.dump(node)[:200])


sd = ast.parse(open(os.path.join(KORE, 'bin', 'spin_doctor.py')).read())
tables = {}
for node in ast.walk(sd):
    if isinstance(node, ast.FunctionDef) and node.name == 'param_table':
        tables['param_table'] = names_of([s for s in node.body if isinstance(s, ast.Return)][0].value)
    elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in \
            ('flow_cols', 'thermal_cols', 'composition_cols', 'magnetic_cols', 'rot_cols'):
        tables[node.targets[0].id] = names_of(node.value)
try:
    import get_data
    pairs = (('param_table', get_data.PARAMS), ('flow_cols', get_data.FLOW), ('thermal_cols', get_data.THERMAL),
             ('composition_cols', get_data.COMPOSITION), ('magnetic_cols', get_data.MAGNETIC),
             ('rot_cols', get_data.ROTDYN))
    expect = {k: [n + p for n in tables[k] for p in ('_re', '_im')] if k == 'rot_cols' else tables[k] for k, _ in pairs}
    bad = [k for k, names in pairs if names != expect[k]]
    report('get_data_names', set(tables) == set(k for k, _ in pairs) and not bad,
           ', '.join('%s %d' % (k, len(expect[k])) for k, _ in pairs) + ('' if not bad else '; differ: ' + ', '.join(bad)))
except ImportError as e:
    report('get_data_names', False, 'tools/get_data.py needs pandas: %s' % e)


# ----------------------------------------------------------------------------------------------------------------------
# 8. the stops
def stop(name, extra, steps, msg, files=()):
    d = setup(os.path.join(W, name), extra)
    for src in files:
        shutil.copy(src, d)
    rc, out = run(d, steps, check=False)
    report(name, rc != 0 and msg in out, msg if msg in out else 'exit %d: %s' % (rc, out.strip()[-300:]))


stop('stop_format', "par.field_format = 'hdf5'", 's', "field_format 'hdf5'")
stop('stop_track_3', 'par.track_target = 3', 's', 'track_target 3: 0 (no tracking)')
stop('stop_track_forced', LIB + '; par.track_target = 1', 's', 'track_target follows an eigenvalue')
stop('stop_track_hydro0', MAG + '; par.hydro = 0; par.track_target = 2', 's', 'it needs the flow (hydro = 1)')
stop('stop_track_file', 'par.track_target = 1', 's', 'track_target = 1 needs the file track_target')
stop('stop_no_solution', '', 'sad', 'no solution file of the flow')

rc, out = sh(W, [py, os.path.join(KORE, 'tools', 'setpar.py'), 'N', 'par.(', f])
report('stop_setpar', rc != 0 and 'tools/setpar.py' in out and open(f).read() == want,
       out.strip().splitlines()[-1] if out.strip() else 'exit %d' % rc)
os.makedirs(os.path.join(W, 'empty'), exist_ok=True)
rc, out = sh(os.path.join(W, 'empty'), [os.path.join(KORE, 'tools', 'ramp.sh')])
report('stop_ramp', rc != 0 and 'run it from the top folder' in out, out.strip())
d = setup(os.path.join(W, 'stop_topo'), DRAG + '; par.track_target = 1', tools=True)
rc, out = sh(d, ['tools/topo_iterate.sh'])
report('stop_topo_file', rc != 0 and 'needs a file track_target' in out, out.strip())
d = setup(os.path.join(W, 'stop_topo_par'), DRAG, tools=True)
np.savetxt(d + '/track_target', [[0.0, 0.98]])
rc, out = sh(d, ['tools/topo_iterate.sh'])
report('stop_topo_par', rc != 0 and 'needs par.track_target = 1' in out, out.strip())

# ----------------------------------------------------------------------------------------------------------------------
# 9. the plotting tools (tests/tools/plot_check.py runs them)
def plot(d, kind, arg=None):
    '''plot_check.py's dictionary for the run folder d, or {'error': its last line}'''
    rc, out = sh(d, [py, os.path.join(T, 'plot_check.py'), kind] + ([arg] if arg else []))
    lines = out.strip().splitlines()
    if rc == 0 and lines and lines[-1].startswith('PLOTCHECK '):
        return json.loads(lines[-1][len('PLOTCHECK '):])
    return {'error': lines[-1] if lines else 'exit %d' % rc}


try:
    import matplotlib   # noqa: F401
    mpl = True
except ImportError as e:
    mpl = False
    report('plots', False, 'the plotting tools need matplotlib: %s' % e)
if mpl:
    build('plots_fs', TH + '; ' + MAG + "; par.structure = 'Wu 2005'; par.r_out = 0.9; par.stratification = 'top hat'; "
          "par.ricb = 0.0")
    for name, folder, loads, fields in (('every', 'fmt_every_npy', ['u', 'b'], ['b', 'buo', 'vis', 'curl_u']),
                                        ('fullsphere', 'plots_fs', ['u', 'b'], ['b', 'u', 'buo', 'curl_buo']),
                                        ('forced', 'fmt_induction_npy', ['u'], ['u']),
                                        ('text', 'fmt_rotdyn_text', ['u'], ['u', 'b'])):
        d = os.path.join(W, folder)
        ok, res = True, []
        for a in loads:
            r = plot(d, 'load_solution', a)
            ok &= 'error' not in r and r['diff'] == 0 and r.get('jump', 0) < 1e-10
            res.append('load_solution %s %s' % (a, r.get('error') or
                                                ('jumps %.0e' % r['jump'] if 'jump' in r else 'its coefficients')))
        r = plot(d, 'profile')
        ok &= 'error' not in r and abs(r['rel']) < 1e-9
        res.append('profile %s' % (r.get('error') or 'KE to %.0e' % abs(r['rel'])))
        for a in fields:
            r = plot(d, 'field', a)
            ok &= 'error' not in r and r['finite'] and r.get('diff', 0) < 1e-12
            res.append('%s %s' % (a, r.get('error') or ('to %.0e' % r['diff'] if 'diff' in r else 'drawn')))
        report('plots_' + name, ok, '; '.join(res))

print('tools_check: ' + ('all checks ok' if not fail else 'FAILED'))
sys.exit(fail)
