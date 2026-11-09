#!/usr/bin/env python3
'''
NEW (2026-11-09): the drag at the outer wall on a deformed body (centrifugal = 1; bin/topo.py, wall_tensor).
usage: deformed_check.py [WORKDIR]   (default tests/drag/runs; runs built with tests/centrifugal/mk_run.py, one rank each,
       NPAR at a time: environment variable, by default the processors, at most one per 2 GB of available memory)

On the level surface of a deformed body the drag acts on the slip along the surface. kore's stress rows hold the traction of
that surface, so the drag enters them, and the power it dissipates, through K_eff = (|N|/J) F^T K F, taken at the latitude of
the wall's normal (topo.py). The checks:

- axial      the rigid rotation (m = 0) of a stress-free fluid in the exact spheroid ('spheroid', f = 0.3 and -0.2), in a full
             body and in a shell (ricb 0.35): its decay rate per unit kappa_cmb (Richardson in kappa_cmb = 1e-5, 2e-5) against
             -R_ax/(1 - ricb^5), R_ax = D/(kappa E) of the flow z x X on the body's wall, at any Ek (the rigid rotation has
             no strain); and in the first-order 'uniform' body (f = +-0.01) against the first-order value
             -5 (1 + 8f/15)/((1 + 2f/3)(1 - ricb^5)). Expected 1e-6 (the Richardson residual).
- spin-over  the inviscid spin-over of the exact spheroid (f = 0.3, -0.2), the linear flow Q(w x X) tangent to the wall: the
             power the drag would dissipate, -Dtopo/(2 E) of its eigenvector (topo.dissipation, the drag set in memory, since an
             inviscid run has no drag rows), against the same integral over the body's wall: R_so for kappa_cmb, its meridional
             and zonal parts (the hook topo_K (1, 0) and (0, 1)), the hook (x^2, x^2), and the drag of the topography
             ('rekier', with its buoyancy frequency varying along the wall), whose kappa is taken at the latitude of the
             wall's normal. Expected 1e-10 (3e-6 for 'rekier', whose kinks limit kore's quadrature to about 1e-6).
- first      the same in the first-order 'uniform' body (f = +-0.02, +-0.01) against the integral over the exact spheroid of
             the same f: the difference is of second order in f (its order from 0.02 to 0.01 between 1.7 and 2.3), for
             kappa_cmb and for 'rekier' (its kappa at the normal's latitude and its buoyancy frequency to first order).
- relabelled the spin-over of a sphere labelled as deformed, f = c (1 - r^2) ('user', c = +-1e-3), which vanishes at the wall
             but not its derivative: the level surfaces inside are spheroids, the body is still the sphere. The terms of
             K_eff in the derivative of the shape (eta, through the wall's |N| and J) must leave the drag's power that of the
             sphere: its part odd in c over c, relative to the sphere's value, below 1e-4 (kappa_cmb, each direction, 'rekier'
             without buoyancy: topo_N is the buoyancy frequency of the level surfaces, which the relabelling moves); it is of
             order 1 without them.
- weak       a viscous spin-over in a spheroidal shell (f = 0.3, Ek 1e-3): a weak drag (kappa_cmb, each direction alone through
             topo_K, and 'rekier') changes the damping rate by the run's own dsig_topo (spin_doctor, flow.dat), to first
             order: this ties the rows to the dissipation. Expected 5e-3 (the drag also changes the other losses a little).
- stiff      kappa_cmb = 1e8 on the same spheroidal shell gives the eigenvalues of a no-slip wall (solve.py, three nearest
             1.34i). Expected 1e-8.
- sphere     f = 0 ('uniform', centrifugal = 1) with a drag builds the matrices of centrifugal = 0 bitwise.
- stops      the rotating solids with a deformed body still stop (their torques are those of spherical walls).
'''
import json, os, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl

HERE = os.path.dirname(os.path.abspath(__file__))
MK = os.path.join(HERE, '..', 'centrifugal', 'mk_run.py')
W = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'runs'))
os.makedirs(W, exist_ok=True)
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):   # runs side by side: one thread each
    os.environ.setdefault(v, '1')
fail = 0


def report(name, ok, text):
    global fail
    print('%-10s %s  %s' % (name, 'ok    ' if ok else 'FAILED', text), flush=True)
    fail |= not ok


def npar():
    if os.environ.get('NPAR'):
        return max(1, int(os.environ['NPAR']))
    n = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    try:
        mem = [int(x.split()[1]) for x in open('/proc/meminfo') if x.startswith('MemAvailable:')][0] // (2*1024**2)
        n = min(n, mem)
    except (OSError, IndexError, ValueError):
        pass
    return max(1, n)


RUNS = {}   # name: (settings, solve)


def run(name, extra, solve=False):
    '''registers the run name, built below with the others, side by side'''
    RUNS[name] = (extra, solve)


def build(name):
    extra, solve = RUNS[name]
    p = subprocess.run([sys.executable, MK, D(name), extra.replace('; ', '\n')] + (['solve'] if solve else []),
                       capture_output=True, text=True)
    return name, p.returncode, p.stdout[-3000:] + p.stderr[-3000:]


def D(name):
    return os.path.join(W, 'def_' + name)


def matrices(d):
    def load(M):
        z = np.load(os.path.join(d, M + '.npz'))
        return ss.csr_matrix((z['data'], z['indices'], z['indptr']), shape=z['shape']).astype(complex)
    return load('A'), load('B')


def eigs(d, sigma, k=4):
    '''the k eigenvalues of A x = lambda B x nearest sigma (shift-invert, scipy)'''
    A, B = matrices(d)
    lu = ssl.splu((A - sigma*B).tocsc())
    op = ssl.LinearOperator(A.shape, matvec=lambda x: lu.solve(B @ x), dtype=complex)
    nu = ssl.eigs(op, k=k, which='LM', tol=1e-14, ncv=40, return_eigenvectors=False)
    lam = sigma + 1/nu
    return lam[np.argsort(abs(lam - sigma))]


def axes(f):
    '''the semi-axes a, c of the exact spheroid's outer wall (c/a = 1 - f, a^2 c = 1)'''
    return (1 - f)**(-1/3), (1 - f)**(2/3)


# ---------------------------------------------------------------------------------------------------- the body's integrals
# On the wall X = (a sin t cos p, a sin t sin p, c cos t) (x = cos t), dS = a^2 s dx dp with s = sqrt(b2 + e2 x^2),
# b2 = (c/a)^2, e2 = 1 - b2, and the normal's cosine x_n = x/s. Per unit Gaspard and kappa, D/(2 E) (u_1 = Re u):
#   the rigid rotation z x X, E = (8 pi/15) a^4 c (1 - ricb^5):  R_ax = (15/4) a^2 int kappa_pp (1 - x^2) s dx
#   the spin-over Q (w x X), w = (1, i, 0), E = (8 pi/15) a^2 c^2 (a^2 + c^2):
#       R_so = (15/4) a^2 int [kappa_tt s^2 + kappa_pp x^2] s dx/(2 - e2)
# by the midpoint rule on NMID cells (kappa smooth: exact to 1e-11; with the kinks of 'rekier' to about 1e-9).
NMID = 200000


def body_rates(f, kt, kp):
    '''(R_ax, R_so) for the tensor kt, kp: functions of the normal's cosine, or their values at the midpoints'''
    a, c = axes(f)
    e2 = 1 - (c/a)**2
    x = -1 + (np.arange(NMID) + 0.5)*2/NMID
    s = np.sqrt(1 - e2 + e2*x*x)
    t, p = (kt(x/s), kp(x/s)) if callable(kt) else (kt, kp)
    w = 2/NMID
    return 3.75*a*a*w*np.sum(p*(1 - x*x)*s), 3.75*a*a*w*np.sum((t*s*s + p*x*x)*s)/(2 - e2)


ONE = lambda x: np.ones_like(x)
ZERO = lambda x: np.zeros_like(x)

# The helper, run in an inviscid run's folder: -Dtopo/(2 E Gaspard) of the eigenvector nearest the target for the drag
# given (set after the imports, whose checks would stop an inviscid run with a drag); for 'rekier' also kappa at the
# normals of the midpoints of the exact spheroid fref, with its buoyancy frequency N |grad r| = N a^2 s (kref.npy)
HELPER = r'''
import sys, json
sys.path.insert(0, 'bin')
import numpy as np
from parameters import par
target, fref, nmid = complex(sys.argv[1]), float(sys.argv[3]), int(sys.argv[4])
import utils as ut, utils4pp as upp, topo
exec(sys.argv[2])
ev = np.loadtxt('eigenvalues0.dat').reshape((-1, 2))
lam = ev[:, 0] + 1j*ev[:, 1]
i = int(np.argmin(abs(lam - target)))
u = ut.load_field('flow').reshape((2*ut.n, -1))[:, i]
usol2 = upp.expand_reshape_sol(u, par.symm)
KE = upp.kinetic_energy(usol2, par.ricb, ut.rcmb, ut.ell(par.m, par.lmax, par.symm)[2])
KE += upp.centrifugal_terms(par.ricb, ut.rcmb)[0]
out = dict(lam=[lam[i].real, lam[i].imag], rate=-topo.dissipation(usol2)/(2*KE*par.Gaspard))
if par.topo_drag is not None and not par.topo_K:
    a, c = (1 - fref)**(-1/3), (1 - fref)**(2/3)
    x = -1 + (np.arange(nmid) + 0.5)*2/nmid
    s = np.sqrt((c/a)**2 + (1 - (c/a)**2)*x*x)
    k = np.zeros((2, nmid))
    for j in range(0, nmid, 5000):
        k[:, j:j + 5000] = topo.kappa((x/s)[j:j + 5000], n=topo.N_cmb()*a*a*s[j:j + 5000])
    np.save('kref.npy', k)
print(json.dumps(out))
'''
RK = ("par.topo_drag = 'rekier'; par.topo_hrms = 1e-3; par.topo_lambda0 = 0.3; par.topo_H = 0.8; par.topo_N = 0.5; "
      "par.topo_omega = 0.98")
DRAGS = (('kappa_cmb', '', ONE, ONE),
         ('meridional', "par.topo_drag = 'rekier'; par.topo_K = '(1 + 0*x, 0*x)'", ONE, ZERO),
         ('zonal', "par.topo_drag = 'rekier'; par.topo_K = '(0*x, 1 + 0*x)'", ZERO, ONE),
         ('x^2', "par.topo_drag = 'rekier'; par.topo_K = '(x**2, x**2)'", lambda x: x*x, lambda x: x*x),
         ('rekier', RK, None, None))


def power(name, target, setting, fref, kt, kp):
    '''(the run's -Dtopo/(2 E), the integral over the wall of the spheroid fref, the eigenvalue)'''
    p = subprocess.run([sys.executable, '-c', HELPER, repr(target), setting, repr(fref), str(NMID)], cwd=D(name),
                       capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-2000:], p.stderr[-2000:])
        sys.exit('deformed_check: the helper failed in %s' % D(name))
    o = json.loads(p.stdout)
    if kt is None:   # 'rekier'
        kt, kp = np.load(os.path.join(D(name), 'kref.npy'))
    return o['rate'], -body_rates(fref, kt, kp)[1], complex(*o['lam'])


# ------------------------------------------------------------------------------------------------------------- the runs
AX = ("par.ViscosD = 1e-2; par.bco = 0; par.bci = 0; par.m = 0; par.symm = 1; par.N = 24; par.lmax = 21; par.ricb = %r; "
      "par.kappa_cmb = %r; ")
for f in (0.3, -0.2):
    for ri in (0.0, 0.35):
        for k in (1e-5, 2e-5):
            run('ax_%g_%g_%g' % (f, ri, k), AX % (ri, k) + "par.centrifugal_shape = 'spheroid'; par.centrifugal_f = %r" % f)
for f in (0.01, -0.01):
    for k in (1e-5, 2e-5):
        run('ax_u_%g_%g' % (f, k), AX % (0.35, k) + "par.centrifugal_f = %r" % f)
SO = {}   # the spin-over runs, name: (f or c, the flattening of the reference, the target, the shape)
for f in (0.3, -0.2):
    SO['so_%g' % f] = (f, f, 2j/(1 + (1 - f)**2), "par.centrifugal_shape = 'spheroid'; par.centrifugal_f = %r" % f)
for f in (0.02, 0.01, -0.01, -0.02):
    SO['so_u_%g' % f] = (f, f, 1j*(1 + f + f*f/3), "par.centrifugal_f = %r" % f)
for c in (1e-3, -1e-3):
    SO['so_rl_%g' % c] = (c, 0.0, 1j, "par.centrifugal_shape = 'user'; par.centrifugal_f = None; "
                          "par.centrifugal_profile = 'p*(1 - r**2)'; par.centrifugal_param = %r" % c)
for name, (f, fref, t, shape) in SO.items():
    run(name, "par.bco = 0; par.itau = %r; " % (abs(t) + 1e-3) + shape, solve=True)
VIS = ("par.centrifugal_shape = 'spheroid'; par.centrifugal_f = 0.3; par.ViscosD = 1e-3; par.ricb = 0.35; par.bci = 1; "
       "par.bco = %d; par.N = 32; par.lmax = 32; par.itau = 1.34")   # (the spin-over near 1.342i)
WEAK = (('kappa_cmb', 'par.kappa_cmb = 1e-6'),
        ('meridional', "par.topo_drag = 'rekier'; par.topo_K = '(1e-6 + 0*x, 0*x)'"),
        ('zonal', "par.topo_drag = 'rekier'; par.topo_K = '(0*x, 1e-6 + 0*x)'"),
        ('rekier', RK.replace('1e-3', '2e-4') + '; par.topo_omega = None'))
run('weak_0', VIS % 0, solve=True)
for name, setting in WEAK:
    run('weak_' + name, VIS % 0 + '; ' + setting, solve=True)
run('stiff', VIS % 0 + '; par.kappa_cmb = 1e8', solve=True)   # (solve.py: scipy's splu loses the stiff rows)
run('noslip', VIS % 1, solve=True)
SPH = "par.ViscosD = 1e-3; par.ricb = 0.35; par.bci = 1; par.bco = 0; par.lmax = 20; par.topo_drag = 'rekier'; " + \
      "par.topo_hrms = 1e-3; par.topo_N = 0.5; par.topo_omega = 0.98"
run('sph_f0', SPH + "; par.centrifugal_shape = 'uniform'; par.centrifugal_f = 0.0")
run('sph_c0', SPH + '; par.centrifugal = 0')
with ThreadPoolExecutor(npar()) as pool:   # the solves first: they take longest
    done = list(pool.map(build, sorted(RUNS, key=lambda n: not RUNS[n][1])))
bad = [(n, out) for n, rc, out in done if rc]
for n, out in bad:
    print(out)
if bad:
    sys.exit('deformed_check: the runs %s failed' % ', '.join(n for n, _ in bad))

# ------------------------------------------------------------------------------------------------------------------- axial
rows = []
for f in (0.3, -0.2):
    R = body_rates(f, ONE, ONE)[0]
    for ri in (0.0, 0.35):
        ref = -R/(1 - ri**5)
        lam = [eigs(D('ax_%g_%g_%g' % (f, ri, k)), ref*k*(1 + 1e-3))[0] for k in (1e-5, 2e-5)]
        rate = (4*lam[0] - lam[1]).real/2e-5
        rows.append((abs(rate/ref - 1), "'spheroid' f %+.1f ricb %.2f: %.9f against %.9f" % (f, ri, rate, ref)))
for f in (0.01, -0.01):
    ref = -5*(1 + 8*f/15)/((1 + 2*f/3)*(1 - 0.35**5))
    lam = [eigs(D('ax_u_%g_%g' % (f, k)), ref*k*(1 + 1e-3))[0] for k in (1e-5, 2e-5)]
    rate = (4*lam[0] - lam[1]).real/2e-5
    rows.append((abs(rate/ref - 1), "'uniform' f %+.2f ricb 0.35: %.9f against %.9f" % (f, rate, ref)))
err = max(r[0] for r in rows)
report('axial', err < 1e-6, 'the rigid rotation decays per unit kappa_cmb as on the body to %.1e' % err)
for e, t in rows:
    print('           %s (%.1e)' % (t, e))

# --------------------------------------------------------------------------------------------------------------- spin-over
with ThreadPoolExecutor(npar()) as pool:   # (run, drag): (rate, the body's integral, lambda)
    jobs = {(name, drag): pool.submit(power, name, t, setting.replace('topo_N = 0.5', 'topo_N = 0.0') if '_rl_' in name
                                      else setting, fref, kt, kp)
            for name, (f, fref, t, _) in SO.items() for drag, setting, kt, kp in DRAGS}
    P = {k: j.result() for k, j in jobs.items()}
rows = []
for f in (0.3, -0.2):
    for drag, *_ in DRAGS:
        rate, ref, lam = P['so_%g' % f, drag]
        e, tol = abs(rate/ref - 1), (3e-6 if drag == 'rekier' else 1e-10)
        rows.append((e, tol, "f %+.1f %-10s: %.12f against %.12f (lambda %.9fi)" % (f, drag, rate, ref, lam.imag)))
bad = [r for r in rows if r[0] > r[1]]
report('spin-over', not bad, 'the power of the drag on the inviscid spin-over is the body\'s integral (worst %.1e)'
       % max(r[0] for r in rows))
for e, tol, t in rows:
    print('           %s (%.1e)' % (t, e))

rows = []
for drag in ('kappa_cmb', 'rekier'):
    for sg in (1, -1):
        e = {f: P['so_u_%g' % (sg*f), drag][0]/P['so_u_%g' % (sg*f), drag][1] - 1 for f in (0.02, 0.01)}
        o = np.log2(abs(e[0.02]/e[0.01]))
        rows.append((o, "%-9s f %+.2f, %+.2f: off the spheroid by %+.2e, %+.2e: order %.2f" % (drag, sg*0.02, sg*0.01,
                                                                                              e[0.02], e[0.01], o)))
ok = all(1.7 < o < 2.3 for o, _ in rows)
report('first', ok, "the first-order body's drag is the exact spheroid's to second order in f (orders %.2f to %.2f)"
       % (min(o for o, _ in rows), max(o for o, _ in rows)))
for o, t in rows:
    print('           %s' % t)

rows = []
for drag, *_ in DRAGS:
    if drag == 'x^2':
        continue
    rp, rm, r0 = P['so_rl_0.001', drag][0], P['so_rl_-0.001', drag][0], P['so_rl_0.001', drag][1]
    odd = abs(rp - rm)/2/1e-3/abs(r0)
    rows.append((odd, "%-10s: %.12f, %.12f, the sphere's %.12f: odd part/(c R) %.1e" % (drag, rp, rm, r0, odd)))
err = max(r[0] for r in rows)
report('relabelled', err < 1e-4, "a sphere labelled as deformed keeps the sphere's drag: the part odd in c over c %.1e"
       % err)
for e, t in rows:
    print('           %s' % t)

# -------------------------------------------------------------------------------------------------------------------- weak
lam0 = eigs(D('weak_0'), 1.34j)[0]
rows = []
for name, setting in WEAK:
    d = D('weak_' + name)
    lam = eigs(d, lam0)[0]
    ev = np.loadtxt(os.path.join(d, 'eigenvalues0.dat')).reshape((-1, 2))
    i = int(np.argmin(abs(ev[:, 0] + 1j*ev[:, 1] - lam)))
    dsig = np.atleast_2d(np.loadtxt(os.path.join(d, 'flow.dat')))[i, 40]   # column 41: dsig_topo
    e = abs((lam - lam0).real/dsig - 1)
    rows.append((e, '%-10s: the damping rate changes by %.6e, dsig_topo %.6e' % (name, (lam - lam0).real, dsig)))
err = max(r[0] for r in rows)
report('weak', err < 5e-3, 'a weak drag changes the damping rate by the run\'s dsig_topo to %.1e' % err)
for e, t in rows:
    print('           %s (%.1e)' % (t, e))

# ------------------------------------------------------------------------------------------------------------------- stiff
ev = {n: np.loadtxt(os.path.join(D(n), 'eigenvalues0.dat')).reshape((-1, 2)) @ [1, 1j] for n in ('stiff', 'noslip')}
ls = ev['stiff'][np.argsort(abs(ev['stiff'] - 1.34j))][:3]
err = max(min(abs(l - ev['noslip']))/abs(l) for l in ls)
report('stiff', err < 1e-8, 'kappa_cmb = 1e8 gives the eigenvalues of a no-slip wall to %.1e' % err)

# ------------------------------------------------------------------------------------------------------------------ sphere
a, b = matrices(D('sph_f0')), matrices(D('sph_c0'))
same = all(x.shape == y.shape and (x != y).nnz == 0 for x, y in zip(a, b))
report('sphere', same, 'f = 0 with the drag of the topography builds the matrices of centrifugal = 0 %s'
       % ('bitwise' if same else 'NOT bitwise'))

# ------------------------------------------------------------------------------------------------------------------- stops
d = D('stop_rotdyn')
if os.path.isdir(d):
    shutil.rmtree(d)
p = subprocess.run([sys.executable, MK, d, "par.centrifugal_f = 0.1; par.ViscosD = 1e-3; par.ricb = 0.35; par.bci = 1; "
                    "par.bco = 0; par.m = 0; par.symm = 1; par.kappa_cmb = 0.01; par.rotdyn = 1".replace('; ', '\n')],
                   capture_output=True, text=True)
words = 'rotdyn = 1 needs a spherical body (centrifugal = 0)'
ok = p.returncode != 0 and words in p.stdout + p.stderr
report('stops', ok, 'the rotating solids with a deformed body and a drag stop%s' % ('' if ok else ': they did not'))

print('deformed_check: ' + ('all cases ok' if not fail else 'FAILED'))
sys.exit(fail)
