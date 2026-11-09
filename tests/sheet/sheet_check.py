#!/usr/bin/env python3
'''
usage: sheet_check.py [WORKDIR]   (default: tests/sheet/runs)

The push of B0's current sheet on a fluid slipping along a wall (background.sheet_jump, assemble.bc_u_sheet). Where B0
ends at a wall with an insulator, or a thin layer that does not conduct, beyond it and does not meet the field there, it
ends in a current sheet on the fluid's side, whose Lorentz force pushes the fluid along the wall, ∓Hendrik b_r ΔB0_t per
area at the outer and the inner wall; the stress rows of a slipping viscous wall (bco or bci 0 or 2) take that push, in
one form at both walls. Exit code 1 if a check fails.

1. Smoothed sheets. A sheet is the limit of a B0 that falls to the field beyond across a layer of width δ in the fluid,
   whose Lorentz force kore takes in the volume. Such a B0, at δ = 2e-3, 1e-3 and 5e-4, must approach the sharp run with
   the push as δ (Ek = Em = 1e-2, Hendrik 0.1, N 160, m 1 and lmax 24 unless stated, the first four eigenvalues of the
   sharp run against the nearest of each smoothed run): the error halves from 1e-3 to 5e-4 (ratio 0.35 to 0.65), the
   extrapolation that removes the terms in δ and δ², (8λ(δ/4) - 6λ(δ/2) + λ(δ))/3, is ten times closer than λ(5e-4), and
   the sharp run without the push (sheet_on False in the run's copy of assemble.py) is ten times farther. Toroidal
   (B0_tor 'user', k = r) at the outer wall (bco 0) and at the inner one with zero radial stress (bci 2, cnorm_tor 2
   with Hendrik 0.025: the push must carry the normalisation); poloidal ('general') at the outer wall (degrees 1 and 3
   together, cnorm 2 with Hendrik 0.025) and at the inner one; at an outer thin layer with a thickness and permeability
   3 (c 0.05, mu 1.5) and at an inner one of degree 3 (c 0.05, muw 3, mu 0.8), the push for the jump that the permeable
   conditions set (5. checks those conditions); the toroidal sheet at the outer wall of 'Wu 2005' with a viscosity that
   grows outward (the anelastic stress, ν(1) ρ0(1) in the push's factor); the outer toroidal sheet with a drag at the
   wall (kappa_cmb 0.01, in the same stress rows), which must move the eigenvalues ten times more than the error at
   5e-4; and a poloidal and a toroidal sheet together at the outer wall with m 2 and symm -1 (lmax 25: the other l-sets
   of the rows and of f).
2. Axial angular momentum. With m 0 and insulating stress-free walls nothing holds the fluid's axial angular momentum
   L_z, so L_z = 0 in every mode with λ != 0. The toroidal sheet's push has a torque, so the check needs it ('axial'
   with a toroidal B0 of degree 2 that is not 0 at the walls: a shell, with cnorm_tor 'mag_energy', a full sphere and
   'Wu 2005'). Expected: |L_z| < 1e-9 and resid𝐋 < 1e-9 (|L_z| > 1e-3 without the push).
3. Budgets: residσ and resid𝐄 < 1e-8 in the sharp runs of 1. and 2. (resid𝐄 without the thin layers, where it is nan
   with mu != 1), the modes with λ != 0.
4. Stops (submatrices.py): an inviscid full sphere with a poloidal sheet, and with one behind a layer where the
   permeability jumps (its message then asks for the layer's condition, which B0_cmb = 'wall' meets: that run goes on);
   centrifugal = 1 with a sheet at a slipping wall; 'axial' at a slipping wall where the permeability jumps (mu 1.5, or
   a thin layer of permeability 2; at the inner wall, the imposed field; in an inviscid fluid, whose remedy is a viscous
   one), 'dipole' and 'G21 dipole' with B0_ic 'frozen currents' at a slipping inner wall with a thin layer of
   permeability 2 (fields that go on beyond the wall, whose field there kore does not know). The same go on with a
   no-slip wall, or with permeability 1 in the layer and the fluid, and a poloidal sheet at a wall with zero radial
   stress (bco 2) goes on.
5. The wall conditions. sheet_jump takes the jump from background._pol_wall_row, which the smoothed sheets cannot check
   (a wrong condition leaves a residual sheet in the smoothed runs too, which the push then moves with them): its R must
   be the f row of the wall that the solver uses (assemble.bc_b_spherical: an insulator's, or bc_b_twa's with c1 = 0),
   at both walls, degrees 1 to 6, for N1 random values and slopes at once, to 1e-13 (insulating walls, and the thin
   layers of 1. with c 0.05, muw 3 and mu 1.5 or 0.8). And 'FDM' and 'G21 dipole', which B0_cmb = 'wall' adjusts to the
   outer layer's condition (background_field.B0_alpha_cmb, a third copy), have no sheet there (c 0.05, muw 3, mu 1.5),
   while 'G21 dipole' with B0_cmb = 'plain' has one.
'''
import os, sys, shutil, subprocess, signal, threading, time
from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np

T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(T, '..', 'tidal'))
import mk_run   # noqa: E402  (the run builder of tests/tidal: 'Boussinesq', ricb 0.35, no-slip walls)

W = sys.argv[1] if len(sys.argv) > 1 else os.path.join(T, 'runs')
os.makedirs(W, exist_ok=True)
LMAX_LINE = 'par.lmax = par.ellmax(par.ncpus, max(g, 4.*par.ncpus/par.N), par.m, par.N)'
assert LMAX_LINE in mk_run.T
PUSH_LINE = '    sheet_on = bool(par.magnetic'   # assemble.main: whether the stress rows take the push
NCPUS = min(mk_run.NCPUS, 2)


# NEW (2026-11-09): the runs are built side by side, NPAR at a time (environment variable; by default the processors over
# NCPUS, and at most one run per 2 GB of available memory), each by a thread that waits on its steps; start() queues a run,
# build() waits for it. The checks read the runs in their own order, so the output is that of a serial run.
def npar():
    if os.environ.get('NPAR'):
        return max(1, int(os.environ['NPAR']))
    n = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    n //= NCPUS
    try:
        mem = [int(x.split()[1]) for x in open('/proc/meminfo') if x.startswith('MemAvailable:')][0] // (2*1024**2)
        n = min(n, mem)
    except (OSError, IndexError, ValueError):
        pass
    return max(1, n)

POOL = ThreadPoolExecutor(npar())
RUNS = {}   # name: (settings, future)

# NEW (2026-11-09): an interrupt (INT, TERM, or HUP) passes one Ctrl-C to each step that runs, no step starts after it,
# and the check ends, with 128 + the signal: the pool's threads would otherwise go on with their runs, and then with the
# runs not yet started, before the process ends. Every process of the check is a step (step()): its launch begins only
# under LOCK and while STOP is empty, it joins LIVE under LOCK, and the handler, under LOCK, fills STOP and interrupts
# the steps of LIVE; a step whose launch had begun when the signal came (the main thread may run the handler inside
# Popen) is interrupted as soon as it has started. The handler raises nothing, and takes no lock but LOCK (reentrant:
# the main thread, which runs it, may hold it), with os.kill, not Popen.send_signal, which takes a lock of the process:
# an exception raised at any point of the main thread can leave a lock of the pool, of a future or of a process taken,
# on which a thread then waits forever. Each thread ends at its next look at STOP instead (halt_if_stopped()): the main
# thread waits in steps of 0.2 s (step(), result()), and a thread of the pool ends its run at its next step.
# FIX (2026-11-09, review of the above): each step runs in a session of its own, so that a Ctrl-C or a hangup at the
# terminal reaches it only through the handler, and gets one SIGINT, never two (HIT): a second one makes Open MPI's
# mpiexec quit at once without stopping its ranks, and can leave the workers of submatrices.py's Pool running for good.
# With odls_base_sigkill_timeout 0 mpiexec stops its ranks at once after the one. A step that has not ended GRACE s
# after its SIGINT (MPI_GRACE for mpiexec) gets a SIGKILL to its process group: a Pool can hang in its own clean-up,
# and its orphaned workers hold the step's output open; so does the group of an interrupted step once the step has
# ended, for the workers it may have left. Any other end of the main thread (an error, a run that failed, the end of
# the check) stops the runs too (_abort), instead of building every run still queued before the process ends.
os.environ.setdefault('OMPI_MCA_odls_base_sigkill_timeout', '0')
STOP, LIVE, LOCK = [], set(), threading.RLock()   # STOP: the signals received
HIT, ABORT = {}, []   # HIT: the steps interrupted, with the time; ABORT: the main thread ended otherwise
GRACE, MPI_GRACE = 0.5, 3.0

def _interrupted(signum, frame):
    with LOCK:
        STOP.append(signum)
        for p in LIVE:
            _ctrl_c(p)

for _s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(_s, _interrupted)

def _ctrl_c(p):
    '''one SIGINT to the step p, the first time only (setdefault is one call, which the handler cannot split)'''
    t = [time.monotonic()]
    if HIT.setdefault(p, t) is t and p.returncode is None:   # (not yet waited for)
        try:
            os.kill(p.pid, signal.SIGINT)
        except ProcessLookupError:
            pass

def _kill_group(p):
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass

def _abort():
    '''when the main thread ends but by an interrupt (before the threads of the pool are joined): the runs stop at their
    next step, the steps that run get their Ctrl-C, and the runs not yet started are dropped'''
    with LOCK:
        if STOP:
            return
        ABORT.append(True)
        for p in LIVE:
            _ctrl_c(p)
    POOL.shutdown(wait=False, cancel_futures=True)

if hasattr(threading, '_register_atexit'):   # (CPython: its hooks run in reverse order, so before concurrent.futures')
    threading._register_atexit(_abort)

def halt_if_stopped(d):
    '''after an interrupt, ends the thread: the main thread (the check) with 128 + the signal, another (its run in d)
    as a run that failed; so too once the main thread has ended (ABORT)'''
    if STOP or ABORT:
        if threading.current_thread() is threading.main_thread():
            print('%s: interrupted' % os.path.basename(sys.argv[0]), file=sys.stderr, flush=True)
            raise SystemExit(128 + STOP[0] if STOP else 1)
        sys.exit('%s: %s' % (d, 'interrupted' if STOP else 'stopped'))

def step(cmd, d, **kw):
    '''subprocess.run(cmd, cwd=d, **kw) for every process of the check, in a session of its own: none starts after an
    interrupt, which ends the one that runs'''
    kw.setdefault('stdin', subprocess.DEVNULL)   # (outside the terminal's session a step must not read it)
    grace = MPI_GRACE if os.path.basename(cmd[0]) in ('mpiexec', 'mpirun', 'orterun') else GRACE
    with LOCK:
        halt_if_stopped(d)
        p = subprocess.Popen(cmd, cwd=d, start_new_session=True, **kw)
        LIVE.add(p)
        if STOP or ABORT:   # (the handler ran inside Popen, in the main thread)
            _ctrl_c(p)
    killed = False
    try:
        while True:
            try:
                out, err = p.communicate(timeout=0.2)   # (a timeout loses no output)
                break
            except subprocess.TimeoutExpired:   # (keyed on the output, not on p: its workers may hold it open)
                t = HIT.get(p)
                if t and not killed and time.monotonic() - t[0] > grace:
                    _kill_group(p)
                    killed = True
    finally:
        with LOCK:
            LIVE.discard(p)
        if p in HIT:
            _kill_group(p)
    halt_if_stopped(d)
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)

def result(f):
    '''f.result() of a future of the pool, waited for in steps of 0.2 s'''
    while not wait((f,), timeout=0.2).done:
        halt_if_stopped('')
    halt_if_stopped('')
    return f.result()

def start(name, extra, lmax, nopush=False):
    '''queues the run W/name (once; the same name must come with the same settings)'''
    key = (extra, lmax, nopush)
    if name not in RUNS:
        RUNS[name] = (key, POOL.submit(_build, name, extra, lmax, nopush))
    assert RUNS[name][0] == key, 'two runs named %s' % name
    return RUNS[name][1]

def build(name, extra, lmax, nopush=False):
    '''the folder of the run W/name, once built; a run that fails stops the check (and the runs not yet started)'''
    try:
        return result(start(name, extra, lmax, nopush))
    except BaseException:
        for key, f in RUNS.values():
            f.cancel()
        raise


def _build(name, extra, lmax, nopush=False):
    '''a run in W/name (mk_run's defaults, then extra) with that lmax on NCPUS ranks; nopush: without the push'''
    d = os.path.join(W, name)
    halt_if_stopped(d)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d + '/bin')
    for f in os.listdir(mk_run.BIN):
        if os.path.isfile(mk_run.BIN + '/' + f) and f != 'parameters.py':
            shutil.copy(mk_run.BIN + '/' + f, d + '/bin/' + f)
    if nopush:
        a = open(d + '/bin/assemble.py').read()
        assert a.count(PUSH_LINE) == 1, 'assemble.py has no line %r' % PUSH_LINE
        open(d + '/bin/assemble.py', 'w').write(a.replace(PUSH_LINE, PUSH_LINE.replace('bool(', 'False and bool(')))
    T0 = mk_run.T.replace(LMAX_LINE, 'par.lmax = %d' % lmax)
    # NEW (2026-11-09): more MUMPS workspace (%), as in tests/magnetic: with many runs side by side, lz_wu_nopush's
    # factorization ran short of it once (INFOG(1) = -9, on chandler with some 70 jobs at once), never alone
    T0 = T0.replace('{extra}', "par.petsc_opts['st_mat_mumps_icntl_14'] = 50\n{extra}")
    open(d + '/bin/parameters.py', 'w').write(T0.format(ncpus=NCPUS, extra=extra.replace('; ', '\n')))
    py, n = sys.executable, str(NCPUS)
    for cmd, log in (([py, 'bin/submatrices.py', n], 'log_sub'),
                     (mk_run.MPIEXEC.split() + ['-n', n, py, 'bin/assemble.py'], 'log_asm'),
                     (mk_run.MPIEXEC.split() + ['-n', n, py, 'bin/solve.py'], 'log_solve'),
                     ([py, 'bin/spin_doctor.py', n], 'out2')):
        with open(d + '/' + log, 'w') as out:
            p = step(cmd, d, stdout=out, stderr=subprocess.STDOUT)
        if p.returncode:
            sys.exit('%s: %s failed\n%s' % (d, cmd[-2] if cmd[-1] == n else cmd[-1], open(d + '/' + log).read()[-3000:]))
    for f in os.listdir(d):
        if f.endswith('.mtx'):
            os.remove(d + '/' + f)
    return d

def eig(d):
    e = np.loadtxt(os.path.join(d, 'eigenvalues0.dat')).reshape(-1, 2)
    return e[:, 0] + 1j*e[:, 1]

def col(d, name, c):
    '''column c (from 1) of flow.dat or magnetic.dat'''
    return np.loadtxt(os.path.join(d, name), ndmin=2)[:, c - 1]

def nearest(lam, ref):
    return np.array([lam[np.argmin(abs(lam - z))] for z in ref])

fail = 0
def report(name, ok, text):
    global fail
    fail |= not ok
    print('%-22s %s  (%s)' % (name, 'ok' if ok else 'FAIL', text))
    sys.stdout.flush()

budgets = []   # (name, run, whether resid𝐄 is checked)


# 1. smoothed sheets
RI = 0.35
MAG = ("par.magnetic = 1; par.Ek = 1e-2; par.ViscosD = 1e-2; par.Hendrik = 1e-1; par.MagnetD = 1e-2; par.nev = 6; "
       "par.itau = 0.5; par.N = 160; par.m = 1; par.symm = 1")
TOR = "par.B0 = None; par.B0_tor = 'user'; par.B0_tor_l = 1; par.cnorm_tor = 1"
GEN = "par.B0 = 'general'; par.cnorm = 1"
WU = ("par.thermal = 1; par.ThermaD = 1e-2; par.structure = 'Wu 2005'; par.stratification = 'top hat'; par.r_out = 0.9; "
      "par.rpower_u = 5; par.rhopower_u = 1; par.rpower_v = 3; par.rhopower_v = 1; par.rpower_pp = 4; par.rhopower_pp = 4; "
      "par.diffusivity = 'user'; par.diffusivity_profiles = {'nu': '1 + r', 'kappa': '1'}")

def tor_smooth(g, wall, d):
    '''k = g falling to 0 at the wall over a width d'''
    return '(%s)*(1 - exp(-(%s)/%r))' % (g, '1 - r' if wall == 'outer' else 'r - %r' % RI, d)

def pol_smooth(a, k, n, wall, d, c=0.0, p=1.0, mu=1.0):
    '''h = a r^k plus A d exp(-|r - r_w|/d), A such that h meets the steady condition of the wall (background._pol_wall_row:
    (1/mu + n c)(r h)' + (n + e) h = 0 outside, (n + 1 + e) h - (1/mu + (n+1) c)(r h)' = 0 inside), so that B0 meets the
    field beyond and falls to the profile a r^k over the width d'''
    if wall == 'outer':
        D = 1/mu + n*c
        e = (c*mu*n/p)*((n + 1)/p - 1) if c else 0.0
        A = -a*(D*(k + 1) + n + e)/(D*(1 + d) + (n + e)*d)
        return '%r*r**%d + (%r)*%r*exp(-(1 - r)/%r)' % (a, k, A, d, d)
    D = 1/mu + (n + 1)*c
    e = (c*mu*(n + 1)/p)*(n/p + 1) if c else 0.0
    A = -a*RI**k*((n + 1 + e) - D*(k + 1))/((n + 1 + e)*d + D*(RI - d))
    return '%r*r**%d + (%r)*%r*exp(-(r - %r)/%r)' % (a, k, A, d, RI, d)

def gen(prof):
    return 'par.B0_profiles = {%s}' % ', '.join("%d: '%s'" % (n, s) for n, s in prof.items())

TWA_OUT = (1.5, "par.mantle = 'TWA'; par.c_cmb = 0.05; par.c1_cmb = 0; par.muw_cmb = 3.0; par.mu = 1.5; "
                "par.innercore = 'TWA'; par.c_icb = 0; par.c1_icb = 0")
TWA_IN = (0.8, "par.innercore = 'TWA'; par.c_icb = 0.05; par.c1_icb = 0; par.muw_icb = 3.0; par.mu = 0.8; "
               "par.mantle = 'TWA'; par.c_cmb = 0; par.c1_cmb = 0")
cases = [   # name, settings, the B0 of a smoothing width (None: sharp), whether resid𝐄 is checked
    ('tor_out', MAG + '; ' + TOR + '; par.bco = 0; par.bci = 1',
     lambda d: "par.B0_tor_profile = '%s'" % ('r' if d is None else tor_smooth('r', 'outer', d)), True),
    ('tor_out_drag', MAG + '; ' + TOR + '; par.bco = 0; par.bci = 1; par.kappa_cmb = 0.01',   # the drag in the same rows
     lambda d: "par.B0_tor_profile = '%s'" % ('r' if d is None else tor_smooth('r', 'outer', d)), True),
    ('tor_in_bc2', MAG + '; ' + TOR + '; par.bco = 1; par.bci = 2; par.cnorm_tor = 2.0; par.Hendrik = 0.025',
     lambda d: "par.B0_tor_profile = '%s'" % ('r' if d is None else tor_smooth('r', 'inner', d)), True),
    ('pol_out', MAG + '; ' + GEN + '; par.bco = 0; par.bci = 1; par.cnorm = 2.0; par.Hendrik = 0.025',
     lambda d: gen({1: 'r**2', 3: '0.5*r**4'} if d is None else
                   {1: pol_smooth(1.0, 2, 1, 'outer', d), 3: pol_smooth(0.5, 4, 3, 'outer', d)}), True),
    ('pol_in', MAG + '; ' + GEN + '; par.bco = 1; par.bci = 0',
     lambda d: gen({1: 'r**2'} if d is None else {1: pol_smooth(1.0, 2, 1, 'inner', d)}), True),
    ('pol_layer_out', MAG + '; ' + GEN + '; ' + TWA_OUT[1] + '; par.bco = 0; par.bci = 1',
     lambda d: gen({1: 'r**2'} if d is None else {1: pol_smooth(1.0, 2, 1, 'outer', d, 0.05, 3.0, TWA_OUT[0])}), False),
    ('pol_layer_in', MAG + '; ' + GEN + '; ' + TWA_IN[1] + '; par.bco = 1; par.bci = 0',
     lambda d: gen({3: 'r**4'} if d is None else {3: pol_smooth(1.0, 4, 3, 'inner', d, 0.05, 3.0, TWA_IN[0])}), False),
    ('wu_tor_out', MAG + '; ' + WU + '; ' + TOR + '; par.bco = 0; par.bci = 1',
     lambda d: "par.B0_tor_profile = '%s'" % ('r' if d is None else tor_smooth('r', 'outer', d)), True),
    ('both_out_m2', MAG.replace('par.m = 1; par.symm = 1', 'par.m = 2; par.symm = -1') + '; ' + GEN +
     "; par.B0_tor = 'user'; par.B0_tor_l = 2; par.cnorm_tor = 1; par.bco = 0; par.bci = 1",   # both sheets, other l-sets
     lambda d: (gen({1: 'r**2'}) + "; par.B0_tor_profile = 'r'") if d is None else
               (gen({1: pol_smooth(1.0, 2, 1, 'outer', d)}) + "; par.B0_tor_profile = '%s'" % tor_smooth('r', 'outer', d)),
     True),
]
assert 'par.m = 2; par.symm = -1' in cases[-1][1]
lmax_of = lambda base: 23 + int(base.split('par.m = ')[1].split(';')[0])   # lmax - m + 1 = 24, a multiple of 2 NCPUS
M0 = ("par.magnetic = 1; par.Ek = 1e-2; par.ViscosD = 1e-2; par.Hendrik = 1e-1; par.MagnetD = 1e-2; par.nev = 4; "
      "par.itau = 0.5; par.N = 64; par.B0 = 'axial'; par.cnorm = 1; par.B0_tor = 'user'; par.B0_tor_l = 2; par.cnorm_tor = 1; "
      "par.m = 0; par.symm = 1; par.bco = 0; par.bci = 0")   # (the runs of 2.)
LZ = (('lz_shell', "par.B0_tor_profile = 'r'; par.cnorm_tor = 'mag_energy'"),
      ('lz_sphere', "par.ricb = 0; par.B0_tor_profile = 'r**2'"),
      ('lz_wu', "par.N = 96; " + WU + "; par.B0_tor_profile = 'r'"))
for name, base, b0, check_E in cases:   # every run of 1. and 2., queued now
    L = lmax_of(base)
    start(name, base + '; ' + b0(None), L)
    start(name + '_nopush', base + '; ' + b0(None), L, nopush=True)
    for dl in (2e-3, 1e-3, 5e-4):
        start('%s_d%g' % (name, dl), base + '; ' + b0(dl), L)
for name, extra in LZ:
    for nopush in (False, True):
        start(name + ('_nopush' if nopush else ''), M0 + '; ' + extra, 23, nopush=nopush)

refs = {}
for name, base, b0, check_E in cases:
    L = lmax_of(base)
    d0 = build(name, base + '; ' + b0(None), L)
    ref = refs[name] = eig(d0)[:4]
    budgets.append((name, d0, check_E))
    e0 = np.max(abs(nearest(eig(build(name + '_nopush', base + '; ' + b0(None), L, nopush=True)), ref) - ref))
    lam = [nearest(eig(build('%s_d%g' % (name, dl), base + '; ' + b0(dl), L)), ref) for dl in (2e-3, 1e-3, 5e-4)]
    e1, e2, e3 = [np.max(abs(x - ref)) for x in lam]
    eR = np.max(abs((8*lam[2] - 6*lam[1] + lam[0])/3 - ref))   # the terms in δ and δ² removed
    ed = np.max(abs(nearest(refs['tor_out'], ref) - ref)) if name.endswith('_drag') else np.inf   # what the drag moves
    report(name, 0.35 < e3/e2 < 0.65 and eR < 0.1*e3 and e0 > 10*e3 and ed > 10*e3,
           '|Δλ| %.1e, %.1e, %.1e at δ 2e-3, 1e-3, 5e-4 (ratio %.2f), extrapolated %.1e; %.1e without the push%s'
           % (e1, e2, e3, e3/e2, eR, e0, '' if np.isinf(ed) else ', %.1e without the drag' % ed))


# 2. axial angular momentum (m 0, insulating stress-free walls; M0 and LZ above)
for name, extra in LZ:
    out = []
    for nopush in (False, True):
        d = build(name + ('_nopush' if nopush else ''), M0 + '; ' + extra, 23, nopush=nopush)
        live = abs(eig(d)) > 1e-6   # (the mode λ = 0, a rotation of the whole fluid, carries L_z)
        out.append((np.max(abs(col(d, 'flow.dat', 29) + 1j*col(d, 'flow.dat', 30))[live]),
                    np.max(col(d, 'flow.dat', 39)[live]), d))
    budgets.append((name, out[0][2], True))
    report(name, out[0][0] < 1e-9 and out[0][1] < 1e-9 and out[1][0] > 1e-3,
           '|L_z| %.1e, resid𝐋 %.1e; |L_z| %.1e without the push' % (out[0][0], out[0][1], out[1][0]))


# 3. budgets of the sharp runs
for name, d, check_E in budgets:
    live = abs(eig(d)) > 1e-6
    rs = np.max(col(d, 'flow.dat', 17)[live])
    rE = np.max(col(d, 'magnetic.dat', 11)[live]) if check_E else np.nan
    report('budget_' + name, rs < 1e-8 and (not check_E or rE < 1e-8),
           'residσ %.1e%s' % (rs, ', resid𝐄 %.1e' % rE if check_E else ''))


# 4. stops
def stops(name, extra, words):
    '''whether submatrices.py stops with a message that contains words (a string, or a tuple of strings that must all be
    there); words None: whether it goes on'''
    d = os.path.join(W, name)
    halt_if_stopped(d)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d + '/bin')
    for f in os.listdir(mk_run.BIN):
        if os.path.isfile(mk_run.BIN + '/' + f) and f != 'parameters.py':
            shutil.copy(mk_run.BIN + '/' + f, d + '/bin/' + f)
    open(d + '/bin/parameters.py', 'w').write(mk_run.T.format(ncpus=1, extra=extra.replace('; ', '\n')))
    p = step([sys.executable, 'bin/submatrices.py', '1'], d, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if words is None:
        return p.returncode == 0
    return p.returncode != 0 and all(w in p.stdout + p.stderr for w in ((words,) if isinstance(words, str) else words))

SMALL = "par.N = 24; par.magnetic = 1; par.cnorm = 1; par.Hendrik = 1e-3; par.MagnetD = 1e-3; par.m = 1"
LAYER_O = "par.mantle = 'TWA'; par.c_cmb = 0.05; par.c1_cmb = 0"
LAYER_I = "par.innercore = 'TWA'; par.c_icb = 0.05; par.c1_icb = 0"
stop_cases = [
    ('stop_inviscid_pol', SMALL + "; par.ricb = 0; par.ViscosD = 0; par.m = 2; par.B0 = 'general'; "
                          "par.B0_profiles = {1: 'r**3'}",
     ('B0 ends at the outer wall in a current sheet (the poloidal B0 does not meet the field beyond (degree 1: B0_θ jumps '
      'by 5.0e+00 ∂θP_1))', 'and the fluid is inviscid (ViscosD = 0)')),
    ('stop_centrifugal', SMALL + "; par.B0 = None; par.B0_tor = 'Malkus'; par.bco = 0; par.centrifugal = 1; "
                         "par.centrifugal_shape = 'uniform'; par.centrifugal_f = 0.05",
     'in a deformed body (centrifugal = 1): the push of the sheet on the deformed wall'),
    ('stop_axial_mu', SMALL + "; par.B0 = 'axial'; par.mu = 1.5; " + LAYER_O + "; par.innercore = 'TWA'; par.c_icb = 0; "
                      "par.c1_icb = 0; par.bco = 0",
     ("B0 'axial' goes on beyond the outer wall", '(mu = 1.5, c_cmb = 0.05, muw_cmb = 1.0)')),
    ('stop_axial_muw', SMALL + "; par.B0 = 'axial'; " + LAYER_O + "; par.muw_cmb = 2.0; par.bco = 0",
     ("B0 'axial' goes on beyond the outer wall", '(mu = 1.0, c_cmb = 0.05, muw_cmb = 2.0)')),
    ('stop_dipole_inner', SMALL + "; par.B0 = 'dipole'; " + LAYER_I + "; par.muw_icb = 2.0; par.bci = 0",
     "B0 'dipole' goes on beyond the inner wall"),
    ('stop_frozen_inner', SMALL + "; par.B0 = 'G21 dipole'; par.B0_ic = 'frozen currents'; " + LAYER_I +
                          "; par.muw_icb = 2.0; par.bci = 0",
     "B0 'G21 dipole' (B0_ic 'frozen currents') goes on beyond the inner wall"),
    ('stop_axial_inner', SMALL + "; par.B0 = 'axial'; par.B0_ic = 'frozen currents'; par.mu = 1.5; par.mantle = 'TWA'; "
                         "par.c_cmb = 0; par.c1_cmb = 0; par.innercore = 'TWA'; par.c_icb = 0; par.c1_icb = 0; par.bco = 1; "
                         "par.bci = 0",   # (B0_ic does not act on 'axial': no tag)
     ("B0 'axial' goes on beyond the inner wall (the field imposed from beyond the outer wall, through the core)",
      'depends on how the jump bends the imposed field')),
    ('stop_axial_inviscid', SMALL + "; par.ricb = 0; par.ViscosD = 0; par.B0 = 'axial'; " + LAYER_O + "; par.muw_cmb = 2.0",
     ("B0 'axial' goes on beyond the outer wall", 'a viscous fluid with a no-slip wall (ViscosD > 0, bco = 1)')),
    ('stop_inviscid_layer', SMALL + "; par.ricb = 0; par.ViscosD = 0; par.B0 = 'G21 dipole'; par.mu = 1.5; "
                            "par.mantle = 'TWA'; par.c_cmb = 0; par.c1_cmb = 0; par.B0_cmb = 'plain'",
     ('B0 ends at the outer wall in a current sheet (the poloidal B0 does not meet the field beyond',
      "a poloidal one that meets the steady condition of the wall, where the permeability jumps (B0_cmb = 'wall'")),
    ('runs_inviscid_layer', SMALL + "; par.ricb = 0; par.ViscosD = 0; par.B0 = 'G21 dipole'; par.mu = 1.5; "
                            "par.mantle = 'TWA'; par.c_cmb = 0; par.c1_cmb = 0", None),   # B0_cmb 'wall': no sheet
    ('runs_axial_noslip', SMALL + "; par.B0 = 'axial'; par.mu = 1.5; " + LAYER_O + "; par.innercore = 'TWA'; "
                          "par.c_icb = 0; par.c1_icb = 0", None),
    ('runs_axial_layer', SMALL + "; par.B0 = 'axial'; " + LAYER_O + "; par.bco = 0", None),   # muw = mu = 1
    ('runs_dipole_layer', SMALL + "; par.B0 = 'dipole'; " + LAYER_I + "; par.bci = 0", None),
    ('runs_pol_bco2', SMALL + "; par.B0 = 'general'; par.B0_profiles = {1: 'r**2'}; par.bco = 2", None),
]
stopped = [POOL.submit(stops, name, extra, words) for name, extra, words in stop_cases]   # (side by side too)
for (name, extra, words), f in zip(stop_cases, stopped):
    report(name, result(f), 'stops with its message' if words else 'goes on')


# 5. the wall conditions: background._pol_wall_row against the f rows of assemble.bc_b_spherical, and B0_cmb = 'wall'
ROWS = r'''
import sys, numpy as np
sys.path.insert(0, 'bin')
import bc_variables as bv, utils as ut, assemble as asm, background as bg
from parameters import par
rng = np.random.default_rng(7)
worst = 0.0
for k, wall in enumerate(('outer', 'inner')):
    T = bv.Tb if wall == 'outer' else bv.Ta
    T0 = T.copy()
    for n in range(1, 7):
        H = rng.standard_normal((ut.N1, 2))   # N1 profiles at once: the value and the slope at the wall of each
        T[:, 0], T[:, 1] = H[:, 0], H[:, 1]
        data, rows, cols = asm.bc_b_spherical(n, 'f')
        T[:] = T0
        lfirst = ut.m_bot if ut.symmB0 == -1 else ut.m_top
        row0 = ut.offset['f'] + int(ut.N1*(n - lfirst)/2)
        sel = (rows == row0 + k) & (cols >= row0) & (cols < row0 + ut.N1)   # (not the slip's entries on u, v)
        got = np.zeros(ut.N1, dtype=complex)
        got[cols[sel] - row0] = data[sel]
        want = np.array([bg._pol_wall_row(wall, n, a, b)[0] for a, b in H])
        worst = np.maximum(worst, np.max(np.abs(got - want))/np.max(np.abs(want)))   # (a NaN stays, and fails)
print('WORST %.3e' % worst)
'''
JUMP = r'''
import sys
sys.path.insert(0, 'bin')
import background as bg
print('JUMP', bg.sheet_jump('outer')[1])
'''

def probe(d, code, key):
    '''what the python code run in the run directory d prints after key, on the one line of its stdout that starts with
    key (None if the code fails or prints no such line), and the tail of its output'''
    p = step([sys.executable, '-c', code], d, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    hit = [s[len(key):].strip() for s in p.stdout.splitlines() if s.startswith(key)]
    return (hit[0] if p.returncode == 0 and len(hit) == 1 else None), (p.stdout + p.stderr)[-500:]

for name in ('pol_out', 'pol_layer_out', 'pol_layer_in'):
    got, out = probe(os.path.join(W, name), ROWS, 'WORST ')
    w = np.inf if got is None else float(got)
    report('rows_' + name, w < 1e-13, "_pol_wall_row against bc_b_spherical's f rows, both walls, n 1-6: %.1e%s"
           % (w, '' if got is not None else ': ' + out))

LAYER_P = ("par.mantle = 'TWA'; par.c_cmb = 0.05; par.c1_cmb = 0; par.muw_cmb = 3.0; par.mu = 1.5; par.innercore = 'TWA'; "
           "par.c_icb = 0; par.c1_icb = 0; par.bco = 0")
for name, extra, has in (('cmb_wall_fdm', "par.B0 = 'FDM'", False), ('cmb_wall_g21', "par.B0 = 'G21 dipole'", False),
                         ('cmb_plain_g21', "par.B0 = 'G21 dipole'; par.B0_cmb = 'plain'", True)):
    d = os.path.join(W, name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d + '/bin')
    for f in os.listdir(mk_run.BIN):
        if os.path.isfile(mk_run.BIN + '/' + f) and f != 'parameters.py':
            shutil.copy(mk_run.BIN + '/' + f, d + '/bin/' + f)
    open(d + '/bin/parameters.py', 'w').write(mk_run.T.format(ncpus=1, extra=(SMALL + '; ' + LAYER_P + '; ' + extra)
                                                              .replace('; ', '\n')))
    jump, out = probe(d, JUMP, 'JUMP ')
    report(name, jump is not None and (jump != '{}') == has, 'the outer sheet %s' % (jump if jump is not None else out))

print('sheet_check: ' + ('all cases ok' if not fail else 'FAILED'))
halt_if_stopped('')   # (an interrupt, while the main thread did not look)
sys.exit(fail)
