#!/usr/bin/env python3
'''
lit_bench.py WORKDIR CASE [CASE ...] [N=.. lmax=.. ncpus=.. nev=.. key=value ...]

The incompressible and Boussinesq literature benchmarks of docs/differential_rotation_details.md §11: Baruteau & Rieutord 2013
(JFM 719, 47; cases 'br*'), Guenel, Baruteau, Mathis & Rieutord 2016 (A&A 589, A22; cases 'gu*') and Mirouh, Baruteau,
Rieutord & Ballot 2016 (JFM 800, 213; cases 'mi*'). Each case builds
WORKDIR/CASE_N<N>_L<lmax> (a copy of the repo's bin and a parameters.py), runs submatrices, assemble and solve (MUMPS,
shift-invert at the published eigenvalue) and prints the eigenvalue nearest the published one, the peak memory of the
solve (the RSS summed over every process a step started, sampled every 0.2 s) and the wall time. A step whose processes
exceed limit= GB (default 30) together is killed. 'list' as CASE prints the cases.
Extra key=value pairs override the case's N, lmax, ncpus, nev, limit or any par attribute (e.g. ViscosD=1e-7).

Conventions. kore: fields ∝ exp(λt + imφ) in the frame rotating at Ω_ref, λ = σ + iω. Baruteau & Rieutord:
exp(iΩ_p t + imφ) in the inertial frame, |ω| = |Re Ω_p| and τ (< 0 damped) in units of Ω_ref, so Ω_p = ω - m, τ = σ;
their m < 0 mode is kore's -m with Ω_p -> -conj(Ω_p). Guenel et al.: the same convention as Baruteau & Rieutord, with
Ω = 1 + ε sin²θ in units of the pole rate (kore 'conical', diff_rot_ampl = ε). Mirouh et al.: exp(λt + imφ) in the inertial frame, λ = τ + iω,
so ω = ω_kore - m, τ = σ. Both: Ω_ref is Ω at the surface (shellular) or on the axis (cylindrical), E = ν/(Ω_ref R²)
= ViscosD; stress-free walls; Mirouh: g = r, ∇T0 = r, P = ν/κ, ThermaD = E/P, Beyonce = N², T = 0 at the walls.
'''
import os, sys, shutil, subprocess, time, signal, textwrap
import numpy as np

BIN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bin'))

# name: (description, published (|ω| or ω inertial, τ), m, symm, settings, N, L) -- N, L the paper's resolution
BR = "par.structure='Boussinesq'; par.thermal=0; par.diff_rot=1; par.thermal_wind=0; par.diff_rot_ampl=1.0; "
GU = BR + "par.diff_rot_type='conical'; "
MI = ("par.structure='Boussinesq'; par.stratification='linear'; par.thermal=1; par.diff_rot=1; par.thermal_wind=0; "
      "par.diff_rot_type='shellular_boussinesq'; par.bci_thermal=0; par.bco_thermal=0; ")
CASES = {
    'br6':   ('BR13 fig. 6, cylindrical eps -0.3, m 0, E 1e-8', (0.9109, -7.45e-3), 0, 1,
              BR + "par.diff_rot_type='cylindrical'; par.diff_rot_ampl=-0.3; par.ricb=0.35; par.ViscosD=1e-8", 450, 800),
    'br7':   ('BR13 fig. 7, cylindrical eps -0.4, m 0, E 1e-8', (1.3291, -6.05e-3), 0, 1,
              BR + "par.diff_rot_type='cylindrical'; par.diff_rot_ampl=-0.4; par.ricb=0.35; par.ViscosD=1e-8", 450, 800),
    'br8':   ('BR13 fig. 8, cylindrical eps -0.4, eta 0.05, m 0, E 1e-8', (1.0995, -2.26e-3), 0, 1,
              BR + "par.diff_rot_type='cylindrical'; par.diff_rot_ampl=-0.4; par.ricb=0.05; par.ViscosD=1e-8", 450, 800),
    'br9a':  ('BR13 fig. 9, shellular sigma 0 (rigid), m 0, E 1e-7', (1.1037, -4.27e-3), 0, 1,
              BR + "par.diff_rot_type='shellular'; par.diff_rot_param=0.0; par.ricb=0.35; par.ViscosD=1e-7", 250, 600),
    'br9b':  ('BR13 fig. 9, shellular sigma -0.5, m 0, E 1e-7', (1.1890, -3.96e-3), 0, 1,
              BR + "par.diff_rot_type='shellular'; par.diff_rot_param=-0.5; par.ricb=0.35; par.ViscosD=1e-7", 250, 600),
    'br9c':  ('BR13 fig. 9, shellular sigma -1.3, m 0, E 1e-7', (1.3101, -1.28e-3), 0, 1,
              BR + "par.diff_rot_type='shellular'; par.diff_rot_param=-1.3; par.ricb=0.35; par.ViscosD=1e-7", 250, 600),
    'br10':  ('BR13 fig. 10, shellular sigma 0.5, eta 0.2, m 0, E 8e-7', (1.2279, -7.86e-3), 0, 1,
              BR + "par.diff_rot_type='shellular'; par.diff_rot_param=0.5; par.ricb=0.2; par.ViscosD=8e-7", 132, 264),
    'br11a': ('BR13 fig. 11, shellular sigma -1, m 0, E 1e-8', (1.1113, -2.15e-3), 0, 1,
              BR + "par.diff_rot_type='shellular'; par.diff_rot_param=-1.0; par.ricb=0.35; par.ViscosD=1e-8", 500, 1000),
    'br11b': ('BR13 fig. 11, shellular sigma -0.5, m 0, E 1e-8', (1.8993, -2.00e-3), 0, 1,
              BR + "par.diff_rot_type='shellular'; par.diff_rot_param=-0.5; par.ricb=0.35; par.ViscosD=1e-8", 580, 1000),
    'br12a': ('BR13 fig. 12, shellular sigma 0.5, m 0, E 1e-8', (0.9235, -3.99e-3), 0, 1,
              BR + "par.diff_rot_type='shellular'; par.diff_rot_param=0.5; par.ricb=0.35; par.ViscosD=1e-8", 400, 800),
    'br12b': ('BR13 fig. 12, shellular sigma -0.05, m 3 antisymmetric, E 1e-8', (-2.5132, -3.77e-3), 3, -1,
              BR + "par.diff_rot_type='shellular'; par.diff_rot_param=-0.05; par.ricb=0.35; par.ViscosD=1e-8", 200, 400),
    'br13':  ('BR13 fig. 13, cylindrical eps 0.4, m 2, E 2e-9', (-2.2991, -4.47e-3), 2, 1,
              BR + "par.diff_rot_type='cylindrical'; par.diff_rot_ampl=0.4; par.ricb=0.35; par.ViscosD=2e-9", 880, 2400),
    'br14':  ('BR13 fig. 14, cylindrical eps 0.4, m 2, E 5e-9', (-2.2003, -3.28e-3), 2, 1,
              BR + "par.diff_rot_type='cylindrical'; par.diff_rot_ampl=0.4; par.ricb=0.35; par.ViscosD=5e-9", 880, 2400),
    'br15':  ('BR13 fig. 15, shellular sigma -0.2, m -6 (kore m 6), E 6e-9', (-7.1086, 1.82e-4), 6, 1,
              BR + "par.diff_rot_type='shellular'; par.diff_rot_param=-0.2; par.ricb=0.35; par.ViscosD=6e-9", 500, 1401),
    'gu4':   ('GBMR16 sec. 4.3.2, conical eps -0.25, eta 0.71, m 0, E 1e-5 (DT mode, "about 1.82")', (1.82, -2e-2), 0, 1,
              GU + "par.diff_rot_ampl=-0.25; par.ricb=0.71; par.ViscosD=1e-5", 60, 150),
    'gu6':   ('GBMR16 fig. 6, conical eps 0.3, eta 0.71, m 0, E 5e-8', (1.1447, -3.60e-3), 0, 1,
              GU + "par.diff_rot_ampl=0.3; par.ricb=0.71; par.ViscosD=5e-8", 200, 600),
    'gu7':   ('GBMR16 fig. 7, conical eps -0.25, eta 0.71, m 0, E 1e-8', (1.3552, -1.75e-3), 0, 1,
              GU + "par.diff_rot_ampl=-0.25; par.ricb=0.71; par.ViscosD=1e-8", 300, 1000),
    'gu8a':  ('GBMR16 fig. 8, conical eps 0.3, eta 0.10, m 0, E 1e-8', (1.6590, -2.93e-3), 0, 1,
              GU + "par.diff_rot_ampl=0.3; par.ricb=0.10; par.ViscosD=1e-8", 400, 1000),
    'gu8b':  ('GBMR16 fig. 8, conical eps -0.25, eta 0.71, m 0, E 1e-8 (quasi-periodic)', (1.2337, -1.02e-4), 0, 1,
              GU + "par.diff_rot_ampl=-0.25; par.ricb=0.71; par.ViscosD=1e-8", 250, 800),
    'gu12':  ('GBMR16 fig. 12, conical eps -0.25, eta 0.71, m 0, E 2.2e-8 (DT mode)', (1.8681, -4.93e-3), 0, 1,
              GU + "par.diff_rot_ampl=-0.25; par.ricb=0.71; par.ViscosD=2.2e-8", 296, 740),
    'gu13':  ('GBMR16 fig. 13, conical eps -0.43, eta 0.35, m 2, E 1e-8 (DT mode)', (-3.2223, -8.11e-3), 2, 1,
              GU + "par.diff_rot_ampl=-0.43; par.ricb=0.35; par.ViscosD=1e-8", 350, 1001),
    'gu14':  ('GBMR16 fig. 14, conical eps 0.3, eta 0.71, m 2, E 1e-9 (corotation)', (-2.2951, -1.03e-2), 2, 1,
              GU + "par.diff_rot_ampl=0.3; par.ricb=0.71; par.ViscosD=1e-9", 458, 2529),
    'gu15':  ('GBMR16 fig. 15, conical eps -0.5, eta 0.71, m 2, E 1e-8 (unstable, corotation)', (-1.0697, 4.62e-3), 2, 1,
              GU + "par.diff_rot_ampl=-0.5; par.ricb=0.71; par.ViscosD=1e-8", 302, 1421),
    'mi1a':  ('MBRB16 table 1, m 4, N2 1.5, P 1e-2, E 1e-9', (-2.67162, -2.623e-3), 4, 1,
              MI + "par.Beyonce=1.5; par.diff_rot_ampl=1.5; par.ricb=0.35; par.ViscosD=1e-9; par.ThermaD=1e-7", None, None),
    'mi1b':  ('MBRB16 table 1, m 0, N2 0.45, P 1e-2, E 1e-9', (1.75431, -1.777e-3), 0, 1,
              MI + "par.Beyonce=0.45; par.diff_rot_ampl=0.45; par.ricb=0.35; par.ViscosD=1e-9; par.ThermaD=1e-7", None, None),
    'mi1c':  ('MBRB16 table 1, m 0, N2 2.6, P 1e-2, E 1e-9', (0.80413, -1.572e-3), 0, 1,
              MI + "par.Beyonce=2.6; par.diff_rot_ampl=2.6; par.ricb=0.35; par.ViscosD=1e-9; par.ThermaD=1e-7", None, None),
    'mi1d':  ('MBRB16 table 1, m 0, N2 2.3, P 1e-2, E 1e-10', (0.30613, -8.884e-4), 0, 1,
              MI + "par.Beyonce=2.3; par.diff_rot_ampl=2.3; par.ricb=0.35; par.ViscosD=1e-10; par.ThermaD=1e-8", None, None),
    # figures 5, 6, 8-12: |omega| and tau from the figure labels, resolution (N, L) of the labels; fig. 5 is damped (the
    # labels drop the sign of tau); figs. 6(a), 6(c) are table 1 rows 2, 3; fig. 12 is NOT table 1 row 1 (tau 60 times smaller)
    'mi5a':  ('MBRB16 fig. 5(a), m 0, N2 0, P 1e-2, E 1e-7', (1.6739, -6.31e-3), 0, 1,
              MI + "par.Beyonce=0.0; par.diff_rot_ampl=0.0; par.ricb=0.35; par.ViscosD=1e-7; par.ThermaD=1e-5", 300, 600),
    'mi5b':  ('MBRB16 fig. 5(b), m 0, N2 0.45, P 1e-2, E 1e-7', (1.7658, -9.44e-3), 0, 1,
              MI + "par.Beyonce=0.45; par.diff_rot_ampl=0.45; par.ricb=0.35; par.ViscosD=1e-7; par.ThermaD=1e-5", 300, 600),
    'mi5c':  ('MBRB16 fig. 5(c), m 0, N2 0.75, P 1e-2, E 1e-7', (1.8220, -1.24e-2), 0, 1,
              MI + "par.Beyonce=0.75; par.diff_rot_ampl=0.75; par.ricb=0.35; par.ViscosD=1e-7; par.ThermaD=1e-5", 300, 600),
    'mi5d':  ('MBRB16 fig. 5(d), m 0, N2 0.79, P 1e-2, E 1e-7', (1.8286, -1.36e-2), 0, 1,
              MI + "par.Beyonce=0.79; par.diff_rot_ampl=0.79; par.ricb=0.35; par.ViscosD=1e-7; par.ThermaD=1e-5", 300, 600),
    'mi6a':  ('MBRB16 fig. 6(a) = table 1, m 0, N2 0.45, P 1e-2, E 1e-9', (1.7541, -1.78e-3), 0, 1,
              MI + "par.Beyonce=0.45; par.diff_rot_ampl=0.45; par.ricb=0.35; par.ViscosD=1e-9; par.ThermaD=1e-7", 498, 996),
    'mi6b':  ('MBRB16 fig. 6(b), m 0, N2 1.1, P 1e-2, E 1e-8 (quasi-periodic)', (1.4902, -3.82e-3), 0, 1,
              MI + "par.Beyonce=1.1; par.diff_rot_ampl=1.1; par.ricb=0.35; par.ViscosD=1e-8; par.ThermaD=1e-6", 300, 600),
    'mi6c':  ('MBRB16 fig. 6(c) = table 1, m 0, N2 2.6, P 1e-2, E 1e-9', (0.8041, -1.57e-3), 0, 1,
              MI + "par.Beyonce=2.6; par.diff_rot_ampl=2.6; par.ricb=0.35; par.ViscosD=1e-9; par.ThermaD=1e-7", 400, 800),
    'mi8':   ('MBRB16 fig. 8, m 0, N2 3, P 1, E 1e-9 (HT mode)', (1.3856, -4.81e-4), 0, 1,
              MI + "par.Beyonce=3.0; par.diff_rot_ampl=3.0; par.ricb=0.35; par.ViscosD=1e-9; par.ThermaD=1e-9", 498, 996),
    'mi9':   ('MBRB16 fig. 9, m 0, N2 6.7, P 1e-2, E 1e-10 (HT mode, wedge, growing)', (0.3296, 3.29e-2), 0, 1,
              MI + "par.Beyonce=6.7; par.diff_rot_ampl=6.7; par.ricb=0.35; par.ViscosD=1e-10; par.ThermaD=1e-8", 496, 992),
    'mi10':  ('MBRB16 fig. 10, m 0, N2 1.5, P 1e-2, E 1e-10 (HT mode; label tau 8.13e-4, sign not given)', (0.2515, 8.13e-4), 0, 1,
              MI + "par.Beyonce=1.5; par.diff_rot_ampl=1.5; par.ricb=0.35; par.ViscosD=1e-10; par.ThermaD=1e-8", 500, 1000),
    'mi11':  ('MBRB16 fig. 11, m 3, N2 6, P 1, E 1e-7 (HT mode, corotation, growing)', (-3.8612, 7.35e-3), 3, 1,
              MI + "par.Beyonce=6.0; par.diff_rot_ampl=6.0; par.ricb=0.35; par.ViscosD=1e-7; par.ThermaD=1e-7", 200, 400),
    'mi12':  ('MBRB16 fig. 12, m 4, N2 1.5, P 1e-2, E 1e-9 (regular mode)', (-2.6715, -4.42e-5), 4, 1,
              MI + "par.Beyonce=1.5; par.diff_rot_ampl=1.5; par.ricb=0.35; par.ViscosD=1e-9; par.ThermaD=1e-7", 398, 797),
}

T = textwrap.dedent('''
from params_default import default_params
import numpy as np
par = default_params()
par.timescale = 'rotation'
par.set_scales()
par.Gaspard = 1.; par.Beyonce = 0.
par.bci = 0; par.bco = 0
par.m = {m}; par.symm = {symm}
{settings}
par.N = {N}; par.lmax = {lmax}; par.ncpus = {ncpus}
par.rtau = {rtau!r}; par.itau = {itau!r}; par.nev = {nev}; par.which_eigenpairs = 'TM'
{extra}
''')


def kore_target(name):
    '''the published eigenvalue as kore's (σ, ω) in the frame rotating at Ω_ref'''
    _, (w, tau), m, _, _, _, _ = CASES[name]
    if name.startswith(('br', 'gu')) and m == 0:
        return tau, abs(w)
    return tau, w + m      # Ω_p = ω_kore - m (Baruteau, with m -> -m, Ω_p -> -conj for fig. 15); ω = ω_kore - m (Mirouh)


def lmax_for(L, m, ncpus):
    '''the smallest lmax >= L with lmax - m + 1 a multiple of 2 ncpus (assemble.py)'''
    lmax = max(L, m + 1)
    while (lmax - m + 1) % (2*ncpus):
        lmax += 1
    return lmax


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


def run(cmd, cwd, log, limit):
    '''runs cmd in cwd; returns (return code, seconds, peak GB); kills the whole process tree above limit GB (code -9)'''
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, stdout=open(cwd + '/' + log, 'w'), stderr=subprocess.STDOUT, start_new_session=True)
    peak = 0.0
    while p.poll() is None:
        procs = tree(p.pid)
        gb = sum(procs.values())/1024**2
        peak = max(peak, gb)
        if gb > limit:
            for q in procs:
                try:
                    os.kill(q, signal.SIGKILL)
                except OSError:
                    pass
            p.wait()
            open(cwd + '/' + log, 'a').write('\nKILLED: %.1f GB > limit %.1f GB\n' % (gb, limit))
            return -9, time.time() - t0, peak
        time.sleep(0.2)
    return p.returncode, time.time() - t0, peak


def main():
    if len(sys.argv) < 3 or sys.argv[2] == 'list':
        print(__doc__)
        for k, v in CASES.items():
            print('%-6s %-62s published %s  paper Nr %s L %s' % (k, v[0], v[1], v[5], v[6]))
        return
    W = os.path.abspath(sys.argv[1])
    names = [a for a in sys.argv[2:] if '=' not in a]
    kv = dict(a.split('=', 1) for a in sys.argv[2:] if '=' in a)
    for name in names:
        desc, pub, m, symm, settings, N0, L0 = CASES[name]
        N = int(kv.get('N', N0 or 0)); L = int(kv.get('lmax', L0 or 0))
        ncpus = int(kv.get('ncpus', 8)); nev = int(kv.get('nev', 6)); limit = float(kv.get('limit', 30))
        lmax = lmax_for(L, m, ncpus)
        extra = '\n'.join('par.%s = %s' % (k, v) for k, v in kv.items() if k not in ('N', 'lmax', 'ncpus', 'nev', 'tag', 'limit'))
        rtau, itau = kore_target(name)
        d = '%s/%s_N%d_L%d%s' % (W, name, N, lmax, kv.get('tag', ''))
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d + '/bin')
        for f in os.listdir(BIN):
            if os.path.isfile(BIN + '/' + f) and f != 'parameters.py':
                shutil.copy(BIN + '/' + f, d + '/bin/' + f)
        open(d + '/bin/parameters.py', 'w').write(T.format(m=m, symm=symm, settings=settings.replace('; ', '\n'), N=N,
                                                            lmax=lmax, ncpus=ncpus, rtau=rtau, itau=itau, nev=nev, extra=extra))
        print('%s: %s\n  N %d lmax %d ncpus %d, target kore (σ, ω) = (%.4e, %.5f)' % (name, desc, N, lmax, ncpus, rtau, itau), flush=True)
        times = {}
        for step, cmd in (('sub', [sys.executable, 'bin/submatrices.py', str(ncpus)]),
                          ('asm', ['mpiexec', '-n', str(ncpus), sys.executable, 'bin/assemble.py']),
                          ('solve', ['mpiexec', '-n', str(ncpus), sys.executable, 'bin/solve.py'])):
            rc, dt, mem = run(cmd, d, 'log_' + step, limit)
            times[step] = (dt, mem)
            if rc:
                print('  %s %s (see %s/log_%s)' % (step, 'KILLED above %.0f GB' % limit if rc == -9 else 'FAILED', d, step))
                if rc != -9:
                    print(open(d + '/log_' + step).read()[-1500:])
                break
        else:
            ev = np.loadtxt(d + '/eigenvalues0.dat', ndmin=2)
            lam = ev[:, 0] + 1j*ev[:, 1]
            j = np.argmin(np.abs(lam - (rtau + 1j*itau)))
            s, w = lam[j].real, lam[j].imag
            wp = (abs(w) if (name.startswith(('br', 'gu')) and m == 0) else w - m)
            print('  kore (σ, ω) = (%+.5e, %+.6f) -> paper convention (%.5f, %+.4e), published (%.5f, %+.4e)' %
                  (s, w, wp, s, pub[0], pub[1]))
            print('  all: ' + ', '.join('(%+.4e, %+.5f)' % (l.real, l.imag) for l in lam))
            open(d + '/result.txt', 'w').write('%s %d %d %.10e %.10e %.10e %.10e %.1f %.1f\n' % (
                name, N, lmax, s, w, pub[0], pub[1], times['solve'][1], sum(t[0] for t in times.values())))
        print('  time sub %.0f s, asm %.0f s, solve %.0f s; peak RSS solve %.1f GB' % (
            times['sub'][0], times.get('asm', (0, 0))[0], times.get('solve', (0, 0))[0], times.get('solve', (0, 0))[1]), flush=True)


if __name__ == '__main__':
    main()
