#!/usr/bin/env python3
'''
fit_rows.py: closed forms in l and m of the magnetic rows, fitted to the exact rows of tests/magnetic/derive.py.
NEW (2026-10-29): the fitting scripts of steps 4 and 5 of the magnetic port (toroidal B0 of degrees 1, 2; poloidal B0 of
degree 2), in one script. docs/derivations.md is the guide.

For a background field B0 of kind KIND (pol: ∇×∇×(h Y_n⁰ 𝐫), tor: ∇×(k Y_n⁰ 𝐫)) and degree n, every row block (row u, v, f or
g; column P, T, F or G; column degree l + Δl) is a sum of monomials (h^(j) or k^(j), (ln ρ0)^(k), the unknown's derivative,
a power of r). Each monomial's coefficient c(l, m) is the angular factor A(l, m, Δl) of the field's degree times a
polynomial a(l) + b(l) m²:
    n = 1:  A = i m (Δl = 0), √(l² - m²)/(2l - 1) (Δl = -1), √((l+1)² - m²)/(2l + 3) (Δl = +1)
    n = 2:  A = 1/((2l - 1)(2l + 3)) (Δl = 0), i m √(l² - m²)/(2l - 1) (Δl = -1), i m √((l+1)² - m²)/(2l + 3) (Δl = +1),
            √((l² - m²)((l-1)² - m²))/((2l - 1)(2l - 3)) (Δl = -2), √(((l+1)² - m²)((l+2)² - m²))/((2l + 3)(2l + 5)) (Δl = +2)
(operators.quad_factors for n = 2). The script derives the rows for l in LMIN..LMAX, m in MMIN..MMAX on JOBS processes
(cached in CACHE), divides by A, and fits a(l) + b(l) m² of the lowest total degree that reproduces every point to 1e-9
relative with rational coefficients (denominators up to 60). It prints one line per (row, column, Δl, monomial):
    ROW COL dl±D  MONOMIAL r^P   [a + b m^2] = <sympy expression in l, m>
'None' means no such polynomial was found: widen the l range (more points) and run again. These lines are the input of
gen_table.py, which writes the operator code.

usage: python3 tools/derive/fit_rows.py KIND N [--rows u,v,f,g] [--l LMIN:LMAX] [--m MMIN:MMAX] [--jobs J] [--cache FILE]
    KIND pol or tor, N 1 or 2; defaults: all rows, l 3:9, m 0:3, J = number of cores, CACHE fit_KINDN.pkl in the current directory
example (~15 min on 30 cores):  python3 tools/derive/fit_rows.py pol 2 --jobs 30 > fit_pol2.log
'''
import os, sys, pickle, argparse
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import sympy as sy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tests', 'magnetic'))
import derive as d

COLS = {'u': 'FG', 'v': 'FG', 'f': 'PT', 'g': 'PT'}


def A(n, l, m, dl):
    '''the angular factor of a coupling Δl for a background field of degree n (0 where there is no coupling)'''
    if n == 1:
        return {0: 1j*m, -1: np.sqrt(l**2 - m**2)/(2*l - 1), 1: np.sqrt((l+1)**2 - m**2)/(2*l + 3)}.get(dl, 0)
    return {0: 1/((2*l - 1)*(2*l + 3)), -1: 1j*m*np.sqrt(l**2 - m**2)/(2*l - 1), 1: 1j*m*np.sqrt((l+1)**2 - m**2)/(2*l + 3),
            -2: np.sqrt(max((l**2 - m**2)*((l-1)**2 - m**2), 0))/((2*l - 1)*(2*l - 3)),
            2: np.sqrt(((l+1)**2 - m**2)*((l+2)**2 - m**2))/((2*l + 3)*(2*l + 5))}.get(dl, 0)


def job(a):
    b0, row, col, dl, l, m = a
    return a[1:], d.rows(row, col, l + dl, l, m, b0=b0)


def fit(pts, y):
    '''a(l) + b(l) m² of the lowest total degree through the points (l, m) -> y, rational coefficients; None if none'''
    Ls, Ms = sy.symbols('l m')
    nl = len(set(l for l, m in pts))
    for tot in range(0, 14):
        for da in range(0, tot + 1):
            db = tot - da - 1
            if da + 1 > nl - 1 or db + 1 > nl - 1:   # keep a spare l: an exact fit must be over-determined
                continue
            M = np.array([[l**i for i in range(da + 1)] + [m**2*l**i for i in range(db + 1)] for l, m in pts], dtype=float)
            c, *_ = np.linalg.lstsq(M, y, rcond=None)
            cr = [Fraction(ci).limit_denominator(60) for ci in c]
            yy = M @ np.array([float(q) for q in cr])
            if np.max(abs(yy - y)) < 1e-9*max(1, np.max(abs(y))):
                cs = [sy.Rational(q.numerator, q.denominator) for q in cr]
                return sy.factor(sum(cs[i]*Ls**i for i in range(da + 1)) + Ms**2*sum(cs[da + 1 + i]*Ls**i for i in range(db + 1)))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('kind', choices=('pol', 'tor')); ap.add_argument('n', type=int, choices=(1, 2))
    ap.add_argument('--rows', default='u,v,f,g'); ap.add_argument('--l', default='3:9'); ap.add_argument('--m', default='0:3')
    ap.add_argument('--jobs', type=int, default=os.cpu_count()); ap.add_argument('--cache', default=None)
    a = ap.parse_args()
    b0 = (a.kind, a.n)
    l0, l1 = map(int, a.l.split(':')); m0, m1 = map(int, a.m.split(':'))
    dls = (-1, 0, 1) if a.n == 1 else (-2, -1, 0, 1, 2)
    cases = [(row, col, dl) for row in a.rows.split(',') for col in COLS[row] for dl in dls]
    cache = a.cache or 'fit_%s%d.pkl' % b0
    res = pickle.load(open(cache, 'rb')) if os.path.exists(cache) else {}
    jobs = [(b0, row, col, dl, l, m) for (row, col, dl) in cases for l in range(l0, l1 + 1) for m in range(m0, m1 + 1)
            if l >= m and l + dl >= max(m, 1) and (row, col, dl, l, m) not in res]
    if jobs:
        with ProcessPoolExecutor(a.jobs) as ex:
            res.update(dict(ex.map(job, jobs)))
        pickle.dump(res, open(cache, 'wb'))
    for (row, col, dl) in cases:
        sel = {(l, m): out for (rw, c, dd, l, m), out in res.items() if (rw, c, dd) == (row, col, dl) and l0 <= l <= l1 and m0 <= m <= m1}
        keys = set(k for out in sel.values() for k in out)
        for key in sorted(keys, key=str):
            pts, y = [], []
            for (l, m), out in sel.items():
                an = A(a.n, l, m, dl)
                if abs(an) < 1e-14:
                    continue
                v = out.get(key, 0)/an
                assert abs(v.imag) < 1e-8*max(1, abs(v)), (row, col, dl, key, l, m)   # A carries all the i's
                pts.append((l, m)); y.append(v.real)
            name = '*'.join('%s%d' % n for n, p in key[0]) + ' r^%d' % key[1]
            print('%s %s dl%+d  %-22s [a + b m^2] = %s' % (row, col, dl, name, fit(pts, np.array(y))), flush=True)


if __name__ == '__main__':
    main()
