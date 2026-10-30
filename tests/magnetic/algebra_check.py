#!/usr/bin/env python3
'''
algebra_check.py: the magnetic couplings of bin/mag_algebra.py against derive.py, for background fields of any degree
(NEW 2026-10-30; needs sympy; no run directory).

WHY. mag_algebra.py builds the Lorentz force and the induction for an axisymmetric B0 of any degrees from the recurrence
algebra of angular.py, with no closed forms. For degrees 1 and 2 the runs agree with the closed forms of operators.py to
round-off (eigenvalues, tests in docs/magnetic.md); for higher degrees this script is the reference: every block, term by
term, against the sympy derivation of derive.rows (b0 = ('pol', n) or ('tor', n)), and that blocks exist exactly where the
derivation has terms. Degrees n = 1..4 (n = 3 and 4 have no closed forms), several m, row degrees up to 12.
HOW. A block is Σ coefficient × label (in kore's A sign, without Hendrik and 1/L'³). The label r^(rpow - X) B_n^(j)
[λ_k] X^(d) (rpow 5, 3, 2, 2 for u, v, f, g) and kore's row = derive's row × (-r^5/L'³) (u) or × (-r³/L'³) (v, f, g) give
the candidate -Σ coefficient × r^(-X) (u, v) or r^(-1-X) (f, g) × B_n^(j) λ_k X^(d) in derive's monomials, which
derive.compare() must find equal (factor 1, mismatch below 1e-10).
usage: python3 tests/magnetic/algebra_check.py [--quick]
'''
import sys, os, types, re
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.normpath(os.path.join(HERE, '..', '..', 'bin')), HERE]
sys.modules.setdefault('parameters', types.SimpleNamespace(par=types.SimpleNamespace(lmax=40, m=0)))
import sympy as sy
import derive as d
import mag_algebra as ma
from derive import r

quick = '--quick' in sys.argv
S = {}
def sym(name):
    return S.setdefault(name, sy.Symbol(name))

def candidate(blocks, row, X, l, dl):
    '''the block (row, column X, row degree l, source l + dl) in derive's monomials'''
    out = 0
    for lab, M in blocks.get((row, X), {}).items():
        c = complex(M[l, l + dl]) if 0 <= l + dl < M.shape[1] else 0
        if c == 0:
            continue
        rr, rx, B, n, j, lam, dd = re.match(r'([uvfg])(\d)(hp|kt)(\w)(\d)(?:lho(\d))?(?:ohr1)?_D(\d)$', lab).groups()
        p = -int(rx) if rr in 'uv' else -1 - int(rx)
        mono = r**p*sym(('h' if B == 'hp' else 'k') + j)*(sym('lr' + lam) if lam else 1)*sym('%s%s' % (X, dd))
        out += -c*mono
    return sy.expand(out)

_scale = {}
def rowscale(row, X, n, m, l, kind):
    '''the largest derived coefficient of the row over all Δl (cached): the scale of the round-off'''
    key = (row, X, n, m, l, kind)
    if key not in _scale:
        vals = [abs(v) for dl in range(-n - 1, n + 2) if l + dl >= max(m, 1) for v in d.rows(row, X, l + dl, l, m, b0=(kind, n)).values()]
        _scale[key] = max(vals + [1e-300])
    return _scale[key]

def tol(l):
    '''the agreement required: 1e-10, looser at large l, where derive.py's own quadrature loses digits (1e-9 at l = 25)'''
    return 1e-10 if l < 15 else 1e-8

bad, n_ok = [], 0
cases = [(1, 1, 3), (3, 2, 5)] if quick else [(1, 1, 3), (2, 0, 4), (3, 2, 5), (3, 1, 6), (4, 3, 6), (4, 0, 8), (3, 2, 12)]
for kind in ('pol', 'tor'):
    for n, m, l in cases:
        blocks = ma.build(m, {'h' if kind == 'pol' else 'k': [n]}, l + n + 12)
        for row, cols in (('u', 'FG'), ('v', 'FG'), ('f', 'PT'), ('g', 'PT')):
            for X in cols:
                for dl in range(-n - 1, n + 2):
                    lc = l + dl
                    if lc < max(m, 1):
                        continue
                    der = d.rows(row, X, lc, l, m, b0=(kind, n))
                    cand = candidate(blocks, row, X, l, dl)
                    label = '%s n=%d m=%d: %s row l=%d, %s col l%+d' % (kind, n, m, row, l, X, dl)
                    if cand == 0:
                        # a block kore does not have: the derivation must vanish there too, up to the round-off of its
                        # quadrature in θ, which grows with l (relative to the row's largest coefficient, 1e-9)
                        if der and max(abs(v) for v in der.values()) > 1e-9*rowscale(row, X, n, m, l, kind):
                            bad.append(label); print('%-50s derived terms, no block  <-- FAIL' % label)
                        continue
                    if not der:
                        bad.append(label); print('%-50s a block, no derived terms  <-- FAIL' % label); continue
                    c, err = d.compare(der, d.as_dict(cand), label)
                    if err > tol(l) or abs(c - 1) > tol(l):
                        bad.append(label)
                    else:
                        n_ok += 1
print('algebra_check.py: %d blocks ok, %s' % (n_ok, 'ok' if not bad else 'FAIL: %s' % bad))
sys.exit(1 if bad else 0)
