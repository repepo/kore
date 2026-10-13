#!/usr/bin/env python3
'''
gen_check.py: the generic differential-rotation couplings (bin/diff_rot_generic.py) against the closed forms of
bin/diff_rot_coefficients.py for f_0 and f_2, every coefficient, m = 0..mmax, all l <= lmax and |Δl| <= 3; and, for
n = 4..nmax, that the generic coefficients vanish outside the Δl ranges assembled by kore (|Δl| <= n, even, for P→P
and T→T; |Δl| <= n + 1, odd, for P↔T). Run in a run directory (uses bin/ and its lmax):

    python3 gen_check.py [mmax [nmax]]          defaults 8, 16
'''
import sys, os
sys.path.insert(0, os.path.abspath('bin'))
import numpy as np
import diff_rot_coefficients as dr
import diff_rot_generic as drg
from parameters import par

mmax = int(sys.argv[1]) if len(sys.argv) > 1 else 8
nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 16
lmax = par.lmax
tol = 1e-10

def closed_name(name, n):
    pre = 'f0' if n == 0 else 'f'
    for a in ('d2f_', 'df_', 'f_'):
        if name.startswith(a):
            return a.replace('f', pre, 1) + name[len(a):]

worst = 0.
print('closed forms vs generic, lmax %d, m = 0..%d' % (lmax, mmax))
for n in (0, 2):
    for name in drg._matrices(n, 0):
        fn = closed_name(name, n)
        if not hasattr(dr, fn):
            continue
        err, sc = 0., 0.
        for m in range(mmax + 1):
            ours, theirs = [], []
            for l in range(max(m, 1), lmax + 1):
                for dl in range(-3, 4):
                    if l + dl < max(m, 1):
                        continue
                    ours.append(drg.coef(name, n, l, m, dl)); theirs.append(complex(getattr(dr, fn)(l, m, dl)))
            ours, theirs = np.array(ours), np.array(theirs)
            sc = max(sc, np.abs(theirs).max())
            err = max(err, np.abs(ours - theirs).max())
        rel = err/sc if sc > 0 else err
        worst = max(worst, rel)
        print('  n=%d %-16s max|generic - closed|/max|closed| = %.1e' % (n, fn, rel))

print('Δl selection rules, n = 4..%d' % nmax)
for n in range(4, nmax + 1, 2):
    bad = 0.
    for m in (0, 1, 3, mmax):
        K = drg._matrices(n, m)
        for name, M in K.items():
            sec = 'v' if '_1C_' in name else 'u'
            comp = 'upol' if name.endswith('P') else 'utor'
            dmax = drg.max_dl(n, sec, comp)
            par_ok = dmax % 2
            sc = np.abs(M).max()
            for l in range(max(m, 1), min(lmax, M.shape[0]) + 1):
                for ls in range(max(m, 1), M.shape[1]):
                    dl = ls - l
                    if abs(dl) > dmax or abs(dl) % 2 != par_ok:
                        bad = max(bad, abs(M[l, ls])/(sc if sc > 0 else 1))
    worst = max(worst, bad)
    print('  n=%2d  largest coefficient outside the assembled Δl range (relative) %.1e' % (n, bad))

print('worst %.1e  ->  %s' % (worst, 'ok' if worst < tol else 'FAIL'))
sys.exit(0 if worst < tol else 1)
