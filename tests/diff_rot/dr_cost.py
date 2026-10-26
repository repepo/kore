#!/usr/bin/env python3
'''
dr_cost.py WORKDIR N LMAX NCPUS [LIMIT_GB]: the cost of the differential rotation against its highest degree n.

For each case and n, builds a run in WORKDIR ('Wu 2005' with 'top hat', ricb 0.35, m 2, symm 1, diff_rot_type 'user' with
ΔΩ = Σ_{k even <= n} 0.02 r² P_k), runs submatrices (NCPUS processes), assemble and solve (mpiexec -n NCPUS, MUMPS
shift-invert, nev 4), and records the wall time and the peak memory of each step (the RSS summed over every process the
step started, sampled every 0.1 s), the number and size of the *.mtx files and nnz of A and B. Cases: no differential
rotation; momentum only (thermal 0); thermal with thermal_wind 1 (heat rows coupled up to n + 2) and 2 (2n + 2).

Memory watchdog: when a step's processes exceed LIMIT_GB (default 30) together, the whole process tree is killed, the
run is recorded as 'killed', and the larger n of the same case are skipped. Results are appended to
WORKDIR/results.json (a rerun skips what is there) and printed; the matrices are deleted after each run.
'''
import os, sys, shutil, subprocess, threading, time, json, signal, textwrap
import numpy as np

if len(sys.argv) < 5:
    sys.exit(__doc__)
W = os.path.abspath(sys.argv[1]); N, LMAX, NC = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
LIMIT = float(sys.argv[5]) if len(sys.argv) > 5 else 30.0
BIN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bin'))
if (LMAX - 2 + 1) % (2*NC):
    sys.exit('LMAX - m + 1 (m = 2) must be a multiple of 2 NCPUS (assemble.py)')
os.makedirs(W, exist_ok=True)

T =textwrap.dedent('''
from params_default import default_params
import numpy as np
par = default_params()
par.timescale = 'rotation'
par.set_scales()
par.ricb=0.35; par.m=2; par.symm=1; par.structure='Wu 2005'; par.stratification='top hat'
par.r_out=0.99; par.aux1=0.5; par.aux2=0.03; par.aux3=0.85; par.aux4=0.03; par.aux5=2.5
par.Gaspard=1.; par.Beyonce=10.; par.ViscosD=1e-4; par.ThermaD=1e-4
par.bci=0; par.bco=0; par.bci_thermal=0; par.bco_thermal=0
par.N={N}; par.lmax={L}; par.ncpus={NC}
par.rtau=0; par.itau=1.0; par.nev=4; par.which_eigenpairs='TM'
par.thermal={thermal}
par.diff_rot={dr}; par.diff_rot_type='user'; par.diff_rot_ampl=1.0
par.diff_rot_profiles={prof!r}
par.thermal_wind={tw}; par.thermal_wind_radial=1
''')


def tree(root):
    '''{pid: rss kB} of root and all its descendants'''
    out = subprocess.run(['ps', '-ax', '-o', 'pid=,ppid=,rss='], capture_output=True, text=True).stdout
    kids, rss = {}, {}
    for l in out.splitlines():
        p, pp, r = (int(x) for x in l.split())
        kids.setdefault(pp, []).append(p); rss[p] = r
    seen, todo = {}, [root]
    while todo:
        p = todo.pop()
        if p in rss and p not in seen:
            seen[p] = rss[p]; todo += kids.get(p, [])
    return seen


def run(cmd, d, log):
    '''runs cmd in d; returns (status, seconds, peak GB); status 'ok', 'failed' or 'killed' (memory above LIMIT)'''
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=d, stdout=open(d + '/' + log, 'w'), stderr=subprocess.STDOUT, start_new_session=True)
    peak, killed = 0.0, False
    while p.poll() is None:
        procs = tree(p.pid)
        gb = sum(procs.values())/1024**2
        peak = max(peak, gb)
        if gb > LIMIT:
            for q in procs:
                try:
                    os.kill(q, signal.SIGKILL)
                except OSError:            # already gone (macOS reports EPERM for a dead group)
                    pass
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:            # already gone (macOS reports EPERM for a dead group)
                pass
            killed = True
            p.wait()
            break
        time.sleep(0.1)
    dt = time.time() - t0
    return ('killed' if killed else 'ok' if p.returncode == 0 else 'failed'), dt, peak


CASES = [('none', 0, 0, [None]),
         ('mom', 0, 0, [0, 2, 4, 8, 12, 16, 24, 32]),
         ('none_th', 1, 0, [None]),
         ('tw1', 1, 1, [0, 2, 4, 8, 12, 16, 24, 32]),
         ('tw2', 1, 2, [0, 2, 4, 8, 12, 16])]
RF = W + '/results.json'
res = json.load(open(RF)) if os.path.isfile(RF) else {}
print('N %d lmax %d ncpus %d, memory limit %.1f GB' % (N, LMAX, NC, LIMIT), flush=True)
for case, thermal, tw, ns in CASES:
    stop = False
    for n in ns:
        key = '%s_n%s_N%d_L%d' % (case, n, N, LMAX)
        if stop:
            print('%-8s n %-4s skipped (a smaller n exceeded the memory limit)' % (case, n), flush=True)
            res[key] = dict(case=case, n=n, N=N, lmax=LMAX, status='skipped'); json.dump(res, open(RF, 'w'), indent=1)
            continue
        if key in res and res[key].get('status') in ('ok', 'killed'):
            stop = res[key]['status'] == 'killed'
            continue
        d = W + '/' + key
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d + '/bin')
        for f in os.listdir(BIN):
            if os.path.isfile(BIN + '/' + f) and f != 'parameters.py':
                shutil.copy(BIN + '/' + f, d + '/bin/' + f)
        prof = {} if n is None else {k: '0.02*r**2' for k in range(0, n + 1, 2)}
        open(d + '/bin/parameters.py', 'w').write(T.format(N=N, L=LMAX, NC=NC, thermal=thermal, dr=int(n is not None),
                                                            prof=prof, tw=tw))
        r = dict(case=case, n=n, N=N, lmax=LMAX, status='ok')
        for step, cmd in (('sub', [sys.executable, 'bin/submatrices.py', str(NC)]),
                          ('asm', ['mpiexec', '-n', str(NC), sys.executable, 'bin/assemble.py']),
                          ('solve', ['mpiexec', '-n', str(NC), sys.executable, 'bin/solve.py'])):
            st, dt, gb = run(cmd, d, 'log_' + step)
            if step == 'solve' and st == 'failed' and 'INFOG(1)=-9' in open(d + '/log_solve').read():
                # MUMPS workspace too small for the pivoting: once more with 100 % extra workspace (icntl_14)
                st, dt, gb = run(cmd + ['-st_mat_mumps_icntl_14', '100'], d, 'log_solve')
                r['icntl_14'] = 100
            r['t_' + step], r['gb_' + step] = dt, gb
            if step == 'sub':
                mtx = [f for f in os.listdir(d) if f.endswith('.mtx')]
                r['n_mtx'], r['mb_mtx'] = len(mtx), sum(os.path.getsize(d + '/' + f) for f in mtx)/1e6
            if step == 'asm' and st == 'ok':
                z = np.load(d + '/A.npz'); r['nnzA'] = int(len(z['data'])); r['size'] = int(z['shape'][0])
                r['mb_AB'] = (os.path.getsize(d + '/A.npz') + os.path.getsize(d + '/B.npz'))/1e6
            if st != 'ok':
                r['status'] = '%s in %s' % (st, step) if st == 'failed' else 'killed'
                r['killed_step'] = step
                if st == 'failed':
                    print(open(d + '/log_' + step).read()[-1500:])
                break
        res[key] = r
        json.dump(res, open(RF, 'w'), indent=1)
        f = lambda k, fmt: (fmt % r[k]) if k in r else '-'
        print('%-8s n %-4s %-8s sub %s s %s GB | asm %s s %s GB | solve %s s %s GB | %s mtx %s MB | size %s nnz A %s' % (
            case, n, (r['status'] + ('*' if 'icntl_14' in r else '')) if r['status'] == 'ok' else r['status'] + '(' + r.get('killed_step', '') + ')',
            f('t_sub', '%5.1f'), f('gb_sub', '%5.2f'), f('t_asm', '%5.1f'), f('gb_asm', '%5.2f'),
            f('t_solve', '%6.1f'), f('gb_solve', '%5.2f'), f('n_mtx', '%d'), f('mb_mtx', '%.1f'),
            f('size', '%d'), f('nnzA', '%.3e')), flush=True)
        stop = r['status'] == 'killed'
        for fn in os.listdir(d):
            if fn.endswith(('.mtx', '.npz', '.field')):
                os.remove(d + '/' + fn)
