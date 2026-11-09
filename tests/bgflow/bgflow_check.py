#!/usr/bin/env python3
'''
usage: bgflow_check.py [WORKDIR]   (default: tests/bgflow/runs)

Checks of the background flow (bgflow = 1, bin/bgflow.py; docs/bgflow.md):

1. Against conductive_IC 2d84351. Its bgflow.py wrote cic_bgflow_lin.npz and cic_bgflow_nl.npz (this folder) from its
   forcing-12 run: an inner core turning at 1 in a shell (ricb 0.35, 'Boussinesq', Ek 1e-3, no-slip, N 48, lmax 31), the
   degrees n <= 12, the nonlinear flow at Ro 1 by its Newton's method. kore's run with the same parameters and its writer
   (python3 bin/bgflow.py 12 [nonlinear]) give both files again (to 1e-12 of the flow's largest coefficient, the
   nonlinear one to 1e-10: its Newton's method solves about 25 systems, which the two branches assemble with different
   round-off), and leave the run's own matrix A in the folder. Under each background (Ro 0.3 linear, Ro 1 nonlinear;
   m 2, N 48, lmax 29) the eigenvalues near 1i, 0.5i, -0.5i and 1.5i are conductive_IC's, from its own matrices (to
   1e-9; dense and balanced, as below).
2. The operators.
   - A toroidal background (T_1, T_3) is a differential rotation (diff_rot 'user'): the same matrices A and B, to 1e-14 of
     the largest entry, with entropy, composition, a field and the density of 'Wu 2005' in a shell, and in a full sphere.
   - A rigid rotation T_1 = r at Ro = 0.1 is the frame turning at 1 + Ro: the eigenvalues of the run at Gaspard 1 + Ro (and
     bgflow_Ro = 0, the same rows and bases), shifted by -i m Ro. Each eigenvalue |λ| < 3 of a dense, balanced
     eigensolution within 1e-10 or 10 times its condition number times the machine epsilon: a viscous full sphere, an
     inviscid one with entropy and composition without diffusion and a field, a shell without diffusion of s and c, a
     field over a conducting inner core and a resolved mantle (which turn with the background), thin conducting walls.
   - rows_check.py: the angular coefficients and radial monomials of bgflow.build against the terms computed on a grid
     of θ from the fields, with the density and p0 of 'Wu 2005' (to 1e-8).
3. spin_doctor's budgets with the background's powers Wbg, Wbg_thm, Wbg_cmp, and those powers against the independent
   textbook computation of power_check.py (to 1e-8, about the digits flow.dat prints). Eigenvalue problems: 'Wu 2005' with
   entropy, composition, a field over a conducting inner core and a resolved mantle; a full sphere with entropy and a field;
   a shell without diffusion of s and c; an inviscid full sphere (its solutions that spin_doctor flags as not
   regular at the centre, irreg₀ > 1e-6, are spurious and left out); thin conducting walls; m 1, symm -1 (the equatorial
   angular momentum, resid𝐋: the background puts no torque in the fluid). Forced problems: forcing 1 (m 2),
   forcing 8 (and its frame: u_8 + x = u_7, the flow of forcing 7, to 1e-12, x the libration of the mantle), forcing 12
   with and without a field, forcing 8 with a thin conducting wall. The residuals residσ, residω, resid𝐋, resid𝑠, resid𝑐,
   resid𝐛, resid𝐄 are below 1e-9 (residσ 1e-6 in the inviscid run, where σ is about 1e-4 and the budget its round-off),
   residens below 1e-6 without a field (the enstrophy budget leaves out the Lorentz force).
4. The stops (bgflow.check, the writer), and the warning on a background that is not regular at the centre of a full
   sphere (background.centre_check).

Exit code 1 if a check failed.
'''
import os, sys, shutil, subprocess, json, signal, threading
from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np
import scipy.linalg as sl
import scipy.sparse as ss

T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(T, '..', 'tidal'))
import mk_run   # noqa: E402  (the run builder of tests/tidal: 'Boussinesq', ricb 0.35, NCPUS ranks)

W = sys.argv[1] if len(sys.argv) > 1 else os.path.join(T, 'runs')
os.makedirs(W, exist_ok=True)
LMAX_LINE = 'par.lmax = par.ellmax(par.ncpus, max(g, 4.*par.ncpus/par.N), par.m, par.N)'
assert LMAX_LINE in mk_run.T
py = sys.executable
NC = str(min(mk_run.NCPUS, 2))   # (every lmax - m + 1 below is a multiple of 4)


def setup(d, extra, lmax, files=()):
    '''the run directory d (bin and parameters.py, as mk_run.build) with that lmax; files: [(source, name)] copied into it'''
    halt_if_stopped(d)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d + '/bin')
    for f in os.listdir(mk_run.BIN):
        if os.path.isfile(mk_run.BIN + '/' + f) and f != 'parameters.py':
            shutil.copy(mk_run.BIN + '/' + f, d + '/bin/' + f)
    t = mk_run.T.replace(LMAX_LINE, 'par.lmax = %d' % lmax)
    open(d + '/bin/parameters.py', 'w').write(t.format(ncpus=NC, extra=extra.replace('; ', '\n')))
    for src, name in files:
        shutil.copy(src, os.path.join(d, name))


def run(d, steps='savd', check=True):
    '''the steps of a run in d: s(ubmatrices), a(ssemble), v (solve), d (spin_doctor); (returncode, output) of the last'''
    cmds = {'s': [py, 'bin/submatrices.py', NC], 'a': mk_run.MPIEXEC.split() + ['-n', NC, py, 'bin/assemble.py'],
            'v': mk_run.MPIEXEC.split() + ['-n', NC, py, 'bin/solve.py'], 'd': [py, 'bin/spin_doctor.py', NC]}
    out = ''
    for st in steps:
        p = step(cmds[st], d, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out = p.stdout + p.stderr
        open(os.path.join(d, 'log_' + st), 'w').write(out)
        if p.returncode:
            if check:
                sys.exit('%s: step %s failed\n%s' % (d, st, out[-3000:]))
            return p.returncode, out
    return 0, out


def _build(name, extra, lmax, steps='savd', files=()):
    d = os.path.join(W, name)
    setup(d, extra, lmax, files)
    run(d, steps)
    return d


# NEW (2026-11-09): the runs are built side by side, NPAR at a time (environment variable; by default the processors over
# the ranks of a run, and at most one run per 2 GB of available memory), each by a thread that waits on its steps: start()
# queues a run, build() waits for it. Every run is queued before the checks (below the settings), and the checks read
# them in their own order, so the output is that of a serial run.
def npar():
    if os.environ.get('NPAR'):
        return max(1, int(os.environ['NPAR']))
    n = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    n //= int(NC)
    try:
        mem = [int(x.split()[1]) for x in open('/proc/meminfo') if x.startswith('MemAvailable:')][0] // (2*1024**2)
        n = min(n, mem)
    except (OSError, IndexError, ValueError):
        pass
    return max(1, n)

POOL = ThreadPoolExecutor(npar())
RUNS = {}   # name: (settings, future)

# NEW (2026-11-09): an interrupt (INT, or TERM) passes a Ctrl-C to the steps that run, no step starts after it, and the
# check ends, with 128 + the signal: the pool's threads would otherwise go on with their runs, and then with the runs
# not yet started, before the process ends. Every process of the check is a step (step()): it starts, and joins LIVE,
# only under LOCK and while STOP is empty, and the handler, under LOCK, fills STOP and interrupts the steps of LIVE, so
# that none escapes it (a thread whose step ended while the main thread was in eigs, which holds the interpreter, may
# start its next one first: LIVE has it). The handler raises nothing, and takes no lock but LOCK (reentrant: the main
# thread, which runs it, may hold it), with os.kill, not Popen.send_signal, which takes a lock of the process: an
# exception raised at any point of the main thread can leave a lock of the pool, of a future or of a process taken, on
# which a thread then waits forever. Each thread ends at its next look at STOP instead (halt_if_stopped()): the main
# thread waits in steps of 0.2 s (step(), result()), and a thread of the pool ends its run at its next step.
STOP, LIVE, LOCK = [], set(), threading.RLock()   # STOP: the signals received

def _interrupted(signum, frame):
    with LOCK:
        STOP.append(signum)
        for p in LIVE:
            _ctrl_c(p)

signal.signal(signal.SIGINT, _interrupted)
signal.signal(signal.SIGTERM, _interrupted)

def _ctrl_c(p):
    if p.returncode is None:   # (not yet waited for)
        try:
            os.kill(p.pid, signal.SIGINT)
        except ProcessLookupError:
            pass

def halt_if_stopped(d):
    '''after an interrupt, ends the thread: the main thread (the check) with 128 + the signal, another (its run in d)
    as a run that failed'''
    if STOP:
        if threading.current_thread() is threading.main_thread():
            print('%s: interrupted' % os.path.basename(sys.argv[0]), file=sys.stderr, flush=True)
            raise SystemExit(128 + STOP[0])
        sys.exit('%s: interrupted' % d)

def step(cmd, d, **kw):
    '''subprocess.run(cmd, cwd=d, **kw) for every process of the check: none starts after an interrupt, which ends the
    one that runs'''
    with LOCK:
        halt_if_stopped(d)
        p = subprocess.Popen(cmd, cwd=d, **kw)
        LIVE.add(p)
    hit = False
    try:
        while True:
            try:
                out, err = p.communicate(timeout=0.2)   # (a timeout loses no output)
                break
            except subprocess.TimeoutExpired:
                if STOP and not hit:   # (started by the main thread while it ran the handler: not yet in LIVE then)
                    _ctrl_c(p)
                    hit = True
    finally:
        with LOCK:
            LIVE.discard(p)
    halt_if_stopped(d)
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)

def result(f):
    '''f.result() of a future of the pool, waited for in steps of 0.2 s'''
    while not wait((f,), timeout=0.2).done:
        halt_if_stopped('')
    halt_if_stopped('')
    return f.result()

def start(name, extra, lmax, steps='savd', files=()):
    '''queues the run W/name (once; the same name must come with the same settings)'''
    key = (extra, lmax, steps, list(files))
    if name not in RUNS:
        RUNS[name] = (key, POOL.submit(_build, name, extra, lmax, steps, files))
    assert RUNS[name][0] == key, 'two runs named %s' % name
    return RUNS[name][1]

def build(name, extra, lmax, steps='savd', files=()):
    '''the folder of the run W/name, once built; a run that fails stops the check (and the runs not yet started)'''
    try:
        return result(start(name, extra, lmax, steps, files))
    except BaseException:
        for key, f in RUNS.values():
            f.cancel()
        raise


def script(d, args, check=True):
    '''python3 args in the run directory d: (returncode, output)'''
    p = step([py] + args, d, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = p.stdout + p.stderr
    if check and p.returncode:
        sys.exit('%s: %s failed\n%s' % (d, ' '.join(args), out[-3000:]))
    return p.returncode, out


fail = 0
def report(name, ok, text):
    global fail
    fail |= not ok
    print('%-18s %s  (%s)' % (name, 'ok' if ok else 'FAIL', text), flush=True)


def load_csr(f):
    with np.load(f) as z:
        return ss.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))


def balance(A, B, it=8):
    '''the pencil (A, B) with rows and columns scaled to unit largest entries: kore's unscaled pencils have condition
    estimates 1e3 times larger than the eigenvalues', and dense QZ moves their eigenvalues by that much'''
    for _ in range(it):
        r = np.maximum(abs(A).max(1), abs(B).max(1)); r[r == 0] = 1
        A, B = A/r[:, None], B/r[:, None]
        c = np.maximum(abs(A).max(0), abs(B).max(0)); c[c == 0] = 1
        A, B = A/c[None, :], B/c[None, :]
    return A, B


def eigs(d, vectors=False):
    '''the finite eigenvalues of the run's (A, B), and with vectors their condition numbers times the machine epsilon'''
    A, B = balance(load_csr(d + '/A.npz').toarray(), load_csr(d + '/B.npz').toarray())
    if not vectors:
        w = sl.eigvals(A, B)
        return w[np.isfinite(w)]
    w, vl, vr = sl.eig(A, B, left=True, right=True)
    ok = np.isfinite(w)
    w, vl, vr = w[ok], vl[:, ok], vr[:, ok]
    den = abs(np.einsum('ij,ik,kj->j', vl.conj(), B, vr))
    kap = np.linalg.norm(vl, axis=0)*np.linalg.norm(vr, axis=0)*(np.linalg.norm(A, 2) + abs(w)*np.linalg.norm(B, 2))/den
    return w, kap*np.finfo(float).eps


def table(d, f):
    return np.loadtxt(os.path.join(d, f), ndmin=2)


def budgets(d):
    '''the largest residual of each budget of spin_doctor over the solutions of a run (nan where not computed), but those
    that spin_doctor flags as not regular at the centre of a full sphere (irreg₀ > 1e-6, flow.dat column 28: the spurious
    modes of an inviscid sphere); None if no solution is left'''
    f = table(d, 'flow.dat')
    keep = ~(f[:, 27] > 1e-6)
    if not keep.any():
        return None
    out = {'σ': f[:, 16], 'ω': f[:, 15], '𝐋': f[:, 38], 'ens': f[:, 14]}
    for name, key, col in (('thermal.dat', '𝑠', 3), ('composition.dat', '𝑐', 3), ('magnetic.dat', '𝐛', 3),
                           ('magnetic.dat', '𝐄', 10)):
        if os.path.exists(os.path.join(d, name)):
            out[key] = table(d, name)[:, col]
    if '𝐄' in out:
        del out['ens']   # (the enstrophy budget leaves out the Lorentz force)
    return {k: (np.nanmax(v[keep]) if np.any(np.isfinite(v[keep])) else np.nan) for k, v in out.items()}


def budgets_ok(name, d, need, tol=1e-9, loose=()):
    '''the budgets of need present and below tol (those of loose below 1e-6, as residens), every other one below tol too'''
    b = budgets(d)
    if b is None:
        report(name, False, 'no solution is regular at the centre (irreg₀ > 1e-6)')
        return
    lim = lambda k: 1e-6 if (k in loose or k == 'ens') else tol
    bad = [k for k in need if not b.get(k, np.nan) <= lim(k)] + \
          [k for k, v in b.items() if not (np.isnan(v) or v <= lim(k))]
    report(name, not bad, ', '.join('resid%s %.1e' % (k, v) for k, v in b.items() if not np.isnan(v)))


def powers_ok(name, d, tol=1e-8):
    '''the background's powers of spin_doctor against power_check.py'''
    rc, out = script(d, [os.path.join(T, 'power_check.py'), '.'], check=False)
    if rc:
        report(name + ' powers', False, 'power_check.py failed: ' + out.strip()[-300:])
        return
    e = json.loads(out.strip().splitlines()[-1])
    errs = {k: e[k] for k in ('Wbg', 'Wbg_im', 'Wbg_thm', 'Wbg_cmp') if k in e}
    ok = abs(e['KE/KE_sd'] - 1) < 1e-10 and max(errs.values()) < tol
    report(name + ' powers', ok, 'KE %.1e, ' % abs(e['KE/KE_sd'] - 1) + ', '.join('%s %.1e' % kv for kv in errs.items()))


TH = "par.thermal = 1; par.stratification = 'linear'; par.Beyonce = 10.; par.ThermaD = 1e-3; par.bci_thermal = 0; par.bco_thermal = 0"
CO = ("par.compositional = 1; par.comp_stratification = 'linear'; par.BeyonceC = 5.; par.CompoD = 3e-4; "
      "par.bci_compositional = 0; par.bco_compositional = 0")
MAG = "par.magnetic = 1; par.cnorm = 'mag_energy'; par.Hendrik = 1e-3; par.MagnetD = 1e-3"
WU = "par.structure = 'Wu 2005'; par.r_out = 0.9"
CHEB = "par.innercore = 'conducting, Chebys'; par.N_cic = 32"
MAN = "par.mantle = 'resolved'; par.r_surface = 1.3; par.N_mantle = 32; par.sigma_mantle = 0.5"
TWA2 = "par.mantle = 'TWA'; par.c_cmb = 0.0; par.c1_cmb = 0.02; par.innercore = 'TWA'; par.c_icb = 0.0; par.c1_icb = 0.01"
BG = lambda prof, ro=0.3: "par.bgflow = 1; par.bgflow_Ro = %r; par.bgflow_profiles = %r" % (ro, prof)
# a background in the shell: rigid at both walls (P_n = P_n' = 0 and T_3 = 0 there), and one in the full sphere (r^n times a
# series in r²), with entropy and composition
SHELL = {'P2': '0.2*(r-ricb)**2*(1-r)**2', 'T1': '0.5*r', 'T3': '(r-ricb)*(1-r)*r', 'S2': '0.05*(r-ricb)*(1-r)',
         'C2': '0.02*r*(1-r)'}
FLOW = {k: v for k, v in SHELL.items() if k[0] in 'PT'}
SPHERE = {'P2': '0.1*r**2*(1-r**2)**2', 'T1': '0.5*r', 'T3': 'r**3*(1-r**2)', 'S2': '0.05*r**2*(1-r**2)'}
LIN, NL = os.path.join(T, 'cic_bgflow_lin.npz'), os.path.join(T, 'cic_bgflow_nl.npz')
F12 = "par.m = 0; par.symm = 1; par.forcing = 12; par.forcing_frequency = 0.0; par.N = 48"   # (1. and 3.)
SMALL = 'par.N = 24; par.m = 2; par.ViscosD = 1e-2; par.thermal_wind = 0'   # (2.)
DR = "par.diff_rot = 1; par.diff_rot_ampl = 1.0; par.diff_rot_type = 'user'; par.diff_rot_profiles = {0: '0.3*r**2', 2: '-0.2*r**2'}"
TOR = BG({'T1': '0.34*r**3', 'T3': '-0.04*r**3'}, 1.0)   # the same flow: T_1 = (f_0 - f_2/5) r, T_3 = f_2 r/5
OPS = (('shell', '; '.join([SMALL, TH, CO, MAG, WU, "par.B0 = 'axial'"])),
       ('sphere', '; '.join([SMALL, TH, MAG, "par.ricb = 0.0; par.B0 = 'G21 dipole'"])))
Ro = 0.1
FRAME = {
    'sphere': "par.ricb = 0.0",
    'sphere_inviscid': "par.ricb = 0.0; par.ViscosD = 0.0; " + WU + "; " + TH + "; par.ThermaD = 0.0; " + CO
                       + "; par.CompoD = 0.0; " + MAG + "; par.B0 = 'axial'",
    'adiabatic': WU + "; " + TH + "; par.ThermaD = 0.0; " + CO + "; par.CompoD = 0.0",
    'field_solids': MAG + "; par.B0 = 'axial'; par.innercore = 'conducting, Chebys'; par.N_cic = 24; par.mantle = 'resolved'; "
                    "par.r_surface = 1.3; par.N_mantle = 24; par.sigma_mantle = 0.5",
    'field_twa': MAG + "; par.B0 = 'axial'; " + TWA2,
}
WU_ALL = '; '.join([WU, TH, CO, MAG, "par.B0 = 'axial'", CHEB, MAN, 'par.N = 96', BG(SHELL)])
F1 = "par.forcing = 1; par.forcing_frequency = 0.37; par.forcing_amplitude_cmb = 1.0; par.N = 48"   # (3.)
LIB = "par.m = 0; par.symm = 1; par.forcing_frequency = 0.7; par.forcing_amplitude_cmb = 0.3; par.N = 48"
F8, F7 = LIB + "; par.forcing = 8", LIB + "; par.forcing = 7; par.forcing_amplitude_icb = 0.3"
BOTH = "par.forcing_amplitude_cmb = 0.4; par.forcing_amplitude_icb = -0.3"
BUDGETS = [   # (3.) name, settings, lmax
    ('sphere', '; '.join(["par.ricb = 0.0", TH, MAG, "par.B0 = 'G21 dipole'", 'par.N = 96', BG(SPHERE)]), 45),
    ('adiabatic', '; '.join([WU, TH, CO, "par.Beyonce = 0.1; par.BeyonceC = 0.05; par.ThermaD = 0; par.CompoD = 0",
                             'par.N = 64', BG(SHELL, 0.1)]), 29),
    ('inviscid', "par.ricb = 0.0; par.ViscosD = 0; par.symm = -1; par.itau = 0.6667; par.N = 64; "
                 + BG({k: v for k, v in SPHERE.items() if k[0] in 'PT'}, 0.01), 29),
    ('twa', '; '.join([MAG, "par.B0 = 'axial'", TWA2, 'par.N = 64', BG(FLOW)]), 29),
    ('equatorial', "par.m = 1; par.symm = -1; par.itau = -1.0; par.N = 48; " + BG(FLOW), 32),
    ('f1', F1 + "; par.m = 2; " + BG(FLOW), 41),
    ('f8', F8 + "; " + BG(FLOW), 23),
    ('f7', F7 + "; " + BG(FLOW), 23),
    ('f8_twa', F8 + "; par.bco = 0; " + MAG + "; par.B0 = 'axial'; par.mantle = 'TWA'; par.c_cmb = 0.0; "
               "par.c1_cmb = 0.02; " + BG(FLOW), 23),
    ('f12', F12 + "; " + BOTH + "; " + BG(FLOW), 23),
    ('f12_field', F12 + "; " + BOTH + "; " + MAG + "; par.B0 = 'axial'; " + CHEB + "; " + BG(FLOW), 23),
]
RUN = {name: (extra, lmax) for name, extra, lmax in BUDGETS}

# every run of 1. to 3., queued now (the longest first)
start('wu_all', WU_ALL, 41)
for name, extra, lmax in BUDGETS:
    start(name, extra, lmax)
start('cic_f12', F12 + "; par.forcing_amplitude_icb = 1.0; par.forcing_amplitude_cmb = 0.0", 31, steps='sav')
for tag, f, ro in (('lin', LIN, 0.3), ('nl', NL, 1.0)):
    start('cic_eig_' + tag, "par.m = 2; par.N = 48; par.bgflow = 1; par.bgflow_Ro = %r" % ro, 29, steps='sa',
          files=[(f, 'bgflow.npz')])
for tag, ex in OPS:
    start('dr_' + tag, ex + '; ' + DR, 13, steps='sa'), start('tor_' + tag, ex + '; ' + TOR, 13, steps='sa')
for tag, ex in FRAME.items():
    ex = SMALL + '; ' + ex
    start('frame_%s_bg' % tag, ex + '; ' + BG({'T1': 'r'}, Ro), 13, steps='sa')
    start('frame_%s_ref' % tag, ex + '; par.Gaspard = %r; ' % (1 + Ro) + BG({'T1': 'r'}, 0.0), 13, steps='sa')


# ----------------------------------------------------------------------------------------------------------------------
# 1. against conductive_IC 2d84351
d = build('cic_f12', F12 + "; par.forcing_amplitude_icb = 1.0; par.forcing_amplitude_cmb = 0.0", 31, steps='sav')
A0 = load_csr(d + '/A.npz')
for args, ref, tag, tol in ((['12'], LIN, 'writer_lin', 1e-12), (['12', 'nonlinear'], NL, 'writer_nl', 1e-10)):
    script(d, ['bin/bgflow.py'] + args)
    with np.load(d + '/bgflow.npz') as a, np.load(ref) as c:
        same = all(list(a[n]) == list(c[n]) for n in ('nP', 'nT')) and ('Ro' in a) == ('Ro' in c) and \
               ('Ro' not in c or abs(float(a['Ro']) - float(c['Ro'])) < 1e-14)
        big = max(abs(c[k]).max() for k in ('P', 'T'))   # (the poloidal flow is 1e-2 of the toroidal one here)
        e = {k: abs(a[k] - c[k]).max()/big for k in ('P', 'T')} if same else {}
    report(tag, same and max(e.values()) < tol, ', '.join('%s %.1e' % kv for kv in e.items()) if same else
           'the degrees or the Ro of the file differ from conductive_IC\'s')
dA = abs(load_csr(d + '/A.npz') - A0).max()/abs(A0).max()
report('writer_restores', dA < 1e-14, 'max |A after - A before| = %.1e of the largest entry' % dA)

REF = {   # conductive_IC's eigenvalues near 1i, 0.5i, -0.5i, 1.5i (its matrices, dense and balanced)
    'lin': [complex(-0.1667533005673, 1.2258193669855), complex(-0.2795147978667, 0.918941463754),
            complex(-0.3240131490617, 0.7166882420793), complex(-0.3902380788711, 0.546804948397),
            complex(-0.4889498423463, 0.4726086050307), complex(-0.1540634199934, -0.2075484805445),
            complex(-0.3719454019109, -0.7527721338384), complex(-0.2902199977725, -0.8559603758909),
            complex(-0.1985266002144, 1.6117562976973), complex(-0.2217131003323, 1.7579294793084)],
    'nl': [complex(-0.1817673887761, 1.1436563470974), complex(-0.303627968112, 0.7982984471795),
           complex(-0.3960169639093, 1.1830091878408), complex(-0.3761732225943, 0.5873611169318),
           complex(-0.4593226688486, 0.429064991105), complex(-0.0937271885561, -0.2252021208495),
           complex(-0.2067178627567, -0.2005470298454), complex(-0.4395107058544, -0.4635039597469),
           complex(-0.2312882080646, 1.4983714376737), complex(-0.2456383387399, 1.6533969578523),
           complex(-0.2922168920758, 1.7153608356238)]}
for tag, f, ro in (('lin', LIN, 0.3), ('nl', NL, 1.0)):
    d = build('cic_eig_' + tag, "par.m = 2; par.N = 48; par.bgflow = 1; par.bgflow_Ro = %r" % ro, 29, steps='sa',
              files=[(f, 'bgflow.npz')])
    w = eigs(d)
    err = max(min(abs(w - x)) for x in REF[tag])
    near = sorted(set(int(i) for t in (1j, 0.5j, -0.5j, 1.5j) for i in np.argsort(abs(w - t))[:3]))
    extra = max(min(abs(np.array(REF[tag]) - w[i])) for i in near)   # (and no eigenvalue there that conductive_IC lacks)
    report('cic_eig_' + tag, max(err, extra) < 1e-9, 'max |λ - λ_cic| %.1e over %d eigenvalues' % (max(err, extra), len(REF[tag])))


# ----------------------------------------------------------------------------------------------------------------------
# 2. the operators (SMALL, DR, TOR, OPS, Ro and FRAME above)
for tag, ex in OPS:
    d1, d2 = build('dr_' + tag, ex + '; ' + DR, 13, steps='sa'), build('tor_' + tag, ex + '; ' + TOR, 13, steps='sa')
    e = {}
    for M in ('A', 'B'):
        M1, M2 = load_csr(d1 + '/%s.npz' % M), load_csr(d2 + '/%s.npz' % M)
        e[M] = abs(M1 - M2).max()/abs(M1).max()
    report('tor=diffrot_' + tag, max(e.values()) < 1e-14, 'A %.1e, B %.1e of the largest entry' % (e['A'], e['B']))

for tag, ex in FRAME.items():
    ex = SMALL + '; ' + ex
    d1 = build('frame_%s_bg' % tag, ex + '; ' + BG({'T1': 'r'}, Ro), 13, steps='sa')
    d2 = build('frame_%s_ref' % tag, ex + '; par.Gaspard = %r; ' % (1 + Ro) + BG({'T1': 'r'}, 0.0), 13, steps='sa')
    (w1, e1), (w2, e2) = eigs(d1, True), eigs(d2, True)
    w2 = w2 - 1j*2*Ro
    sel = abs(w1) < 3
    j = np.array([np.argmin(abs(w2 - x)) for x in w1[sel]])
    dl = abs(w1[sel] - w2[j])
    lim = np.maximum(1e-10, 10*np.maximum(e1[sel], e2[j]))
    report('frame_' + tag, bool(np.all(dl <= lim)) and sel.sum() > 0,
           '%d eigenvalues, max |Δλ| %.1e, max |Δλ|/(κ eps) %.1e' % (sel.sum(), dl.max(), (dl/(lim/10)).max()))

d_wu = build('wu_all', WU_ALL, 41)
rc, out = script(d_wu, [os.path.join(T, 'rows_check.py'), '.'], check=False)
if rc:
    report('rows', False, 'rows_check.py failed: ' + out.strip()[-300:])
else:
    e = json.loads(out.strip().splitlines()[-1])
    worst = max(e, key=e.get)
    report('rows', e[worst] < 1e-8, '%d rows and columns, worst %s %.1e' % (len(e), worst, e[worst]))


# ----------------------------------------------------------------------------------------------------------------------
# 3. the budgets (the runs in BUDGETS above)
run_of = lambda name: build(name, *RUN[name])
budgets_ok('wu_all', d_wu, ('σ', 'ω', '𝑠', '𝑐', '𝐛', '𝐄'))
powers_ok('wu_all', d_wu)
d = run_of('sphere')
budgets_ok('sphere', d, ('σ', 'ω', '𝑠', '𝐛', '𝐄'))
powers_ok('sphere', d)
d = run_of('adiabatic')
budgets_ok('adiabatic', d, ('σ', 'ω', '𝑠', '𝑐'))
powers_ok('adiabatic', d)
d = run_of('inviscid')
budgets_ok('inviscid', d, ('σ', 'ω'), loose=('σ',))
d = run_of('twa')
budgets_ok('twa', d, ('σ', 'ω', '𝐛', '𝐄'))
d = run_of('equatorial')
budgets_ok('equatorial', d, ('σ', 'ω', '𝐋'))

d = run_of('f1')
budgets_ok('f1', d, ('σ', 'ω'))
powers_ok('f1', d)
d8 = run_of('f8')
budgets_ok('f8', d8, ('σ', 'ω', '𝐋'))
d7 = run_of('f7')
FRAME8 = r'''
import os, sys, json; sys.path.insert(0, 'bin')
import numpy as np
from parameters import par
import utils as ut
import field_io
load = lambda d: field_io.load('flow', folder=d).reshape(-1)   # flow.npy or the text files
x = ut.lifted_flow()[:2*ut.n]
u8, u7 = load('.'), load(sys.argv[1])
print(json.dumps({'flow': abs(u8 + x - u7).max()/abs(u7).max()}))
'''
rc, out = script(d8, ['-c', FRAME8, os.path.abspath(d7)], check=False)
e = json.loads(out.strip().splitlines()[-1])['flow'] if rc == 0 else np.nan
report('f8+x=f7', e < 1e-12, '%.1e of the largest coefficient' % e)
d = run_of('f8_twa')
budgets_ok('f8_twa', d, ('σ', 'ω', '𝐋', '𝐄'))
d = run_of('f12')
budgets_ok('f12', d, ('σ', '𝐋'))
d = run_of('f12_field')
budgets_ok('f12_field', d, ('σ', '𝐋', '𝐄'))


# ----------------------------------------------------------------------------------------------------------------------
# 4. the stops
ST = 'par.N = 24; '
STOPS = [
    ('stop_bgflow2', ST + "par.bgflow = 2", '0 (no background flow) or 1', ()),
    ('stop_nofile', ST + "par.bgflow = 1; par.bgflow_Ro = 0.3", 'no background file', ()),
    ('stop_key', ST + BG({'X2': 'r'}), 'the keys are the kind', ()),
    ('stop_degree', ST + BG({'T37': 'r**37*(1-r)'}), 'above the largest the profile codes allow', ()),
    ('stop_S', ST + BG({'T1': 'r', 'S2': 'r'}), 'needs thermal = 1', ()),
    ('stop_C', ST + BG({'T1': 'r', 'C2': 'r'}), 'needs compositional = 1', ()),
    ('stop_antisym', ST + BG({'T2': '(r-ricb)*(1-r)'}), 'must be symmetric about the equator', ()),
    ('stop_through', ST + BG({'P2': 'r**2'}), 'flows through the wall', ()),
    ('stop_nonrigid', ST + MAG + "; par.B0 = 'axial'; " + CHEB + "; " + BG({'T3': 'r**3*(1-r)'}),
     'needs a rigid rotation of the background there', ()),
    ('stop_inviscid', ST + "par.ViscosD = 0.0; " + BG({'T1': 'r'}), 'needs ricb = 0', ()),
    ('stop_centrifugal', ST + "par.centrifugal = 1; par.centrifugal_shape = 'uniform'; par.centrifugal_f = 0.0; "
     "par.centrifugal_q = 0.0; " + BG({'T1': 'r'}), 'with centrifugal = 1 is not covered', ()),
    ('stop_rotdyn', ST + "par.m = 0; par.symm = 1; par.rotdyn = 1; par.gTorque = 1e-3; par.MoIZ_M = 1.0; "
     "par.MoIZ_IC = 0.05; par.OmgtauIC = 50.0; " + BG({'T1': 'r'}), 'with rotdyn = 1 is not covered', ()),
    ('stop_rekier', ST + "par.bco = 0; par.topo_drag = 'rekier'; par.topo_hrms = 0.02; par.topo_lambda0 = 0.431; "
     "par.topo_H = 0.8; " + BG({'T1': 'r'}), "topo_drag 'rekier' is not covered", ()),
    ('stop_ricb', ST + "par.ricb = 0.4; par.bgflow = 1; par.bgflow_Ro = 0.3", 'is for ricb', [(LIN, 'bgflow.npz')]),
    ('stop_Ro', ST + "par.bgflow = 1; par.bgflow_Ro = 0.5", 'holds the nonlinear flow at Ro', [(NL, 'bgflow.npz')]),
    ('stop_density', ST + WU + "; par.bgflow = 1; par.bgflow_Ro = 0.3", 'was written for another density',
     [(LIN, 'bgflow.npz')]),
]
def stop_run(name, extra, files):
    d = os.path.join(W, name)
    setup(d, extra, 13, files)
    return run(d, 's', check=False)
stopped = [POOL.submit(stop_run, name, extra, files) for name, extra, msg, files in STOPS]   # (side by side too)
for (name, extra, msg, files), f in zip(STOPS, stopped):
    rc, out = result(f)
    report(name, rc != 0 and msg in out, (out.strip().splitlines() or ['(no output)'])[-1][:110])
d = os.path.join(W, 'warn_centre')   # (a warning, not a stop: P_2 must go as r² at the centre of a full sphere)
setup(d, ST + "par.ricb = 0.0; " + BG({'P2': '0.1*(1-r**2)', 'T1': 'r'}), 13)
rc, out = run(d, 's', check=False)
report('warn_centre', rc == 0 and 'WARNING: P_2 of the background flow is not regular at r = 0' in out,
       ([l for l in out.splitlines() if 'WARNING' in l] or ['(no warning)'])[0][:110])
for name, extra, msg in (
        ('stop_writer_field', F12 + "; par.forcing_amplitude_icb = 1.0; " + MAG + "; par.B0 = 'axial'",
         'the run has a background field'),
        ('stop_writer_f12', "par.N = 24", 'writes the steady flow of a forcing-12 run')):
    d = os.path.join(W, name)
    setup(d, extra, 23 if 'm = 0' in extra else 13)
    rc, out = script(d, ['bin/bgflow.py', '12'], check=False)
    report(name, rc != 0 and msg in out, (out.strip().splitlines() or ['(no output)'])[-1][:110])

print('bgflow_check: ' + ('all cases ok' if not fail else 'FAILED'))
halt_if_stopped('')   # (an interrupt, while the main thread did not look)
sys.exit(1 if fail else 0)
