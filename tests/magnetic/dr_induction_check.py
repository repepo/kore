#!/usr/bin/env python3
'''
dr_induction_check.py: the induction by a differential rotation, ∇×(U0×b), of bin/diff_rot.py against derive.py
(NEW 2026-10-30; needs sympy; no run directory).

WHY. diff_rot.induction_block builds the coefficients of ∇×(U0×b) at run time from the recurrence algebra of angular.py,
for any degree n of ΔΩ = Σ f_n(r) P_n(cos θ). This script checks them against the independent sympy derivation of
derive.rows_dr (the same vectors built from the definitions and projected by quadrature), term by term, for several n, m
and degrees l, including degrees far above the small ones (the algebra is exact at any l, so must the agreement be), and
that kore has a block exactly where the derivation has terms (the selection rules of induction_dl_range).
HOW. kore's block is Gaspard Σ coefficient × label / L'³, a label being r^(2-X) times f_n (code zf) or r f_n' (code zd)
times F^(d) or G^(d); derive.py's row times -r³/L'³ must equal it. With Gaspard = 1 the candidate in derive's monomials is
    -Σ coefficient × r^(-1-X) × (w or r w') × X^(d)        (w = f_n)
and derive.compare() must find the factor 1 and a mismatch below 1e-10.
usage: python3 tests/magnetic/dr_induction_check.py [--quick]
'''
import sys, os, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.normpath(os.path.join(HERE, '..', '..', 'bin')), HERE]
sys.modules.setdefault('parameters', types.SimpleNamespace(par=types.SimpleNamespace(lmax=60)))
import re
import sympy as sy
import derive as d
import diff_rot as dr
from derive import r

quick = '--quick' in sys.argv
w0, w1 = sy.symbols('w0 w1')
X = {k: sy.Symbol(k) for k in ('F0', 'F1', 'F2', 'G0', 'G1')}


def candidate(row, col, n, l, m, dl):
    '''kore's block (row 'f'/'g', column 'bpol'/'btor', source l + dl) as derive.py's row: a sympy expression, 0 if none'''
    out = 0
    for name, lab in dr.induction_labels(n, row, col):
        c = complex(dr.induction_coef(name, n, l, m, dl))
        if c == 0:
            continue
        sec, x, code, dd = re.match(r'([fg])(\d)(z[fd])\w0_D(\d)$', lab).groups()
        prof = w0 if code == 'zf' else r*w1
        unk = X['%s%d' % ('F' if col == 'bpol' else 'G', int(dd))]
        out += -c*r**(-1 - int(x))*prof*unk
    return sy.expand(out)


_scale = {}
def rowscale(row, part, n, m, l):
    '''the largest derived coefficient of the row over the allowed Δl (cached): the scale of the round-off'''
    key = (row, part, n, m, l)
    if key not in _scale:
        vals = [abs(v) for dl in range(-n - 1, n + 2) if l + dl >= max(m, 1) for v in d.rows_dr(row, part, l + dl, l, m, n).values()]
        _scale[key] = max(vals + [1e-300])
    return _scale[key]

def tol(l):
    '''the agreement required: 1e-10, looser at large l, where derive.py's own quadrature loses digits (1e-9 at l = 25)'''
    return 1e-10 if l < 15 else 1e-8

bad, n_ok = [], 0
cases = [(0, 1, 3), (2, 2, 4), (2, 0, 5), (4, 3, 6)] if quick else \
        [(0, 0, 3), (0, 1, 3), (2, 0, 5), (2, 1, 4), (2, 2, 4), (4, 1, 6), (4, 3, 6), (6, 2, 8), (2, 3, 20), (4, 5, 25)]
for n, m, l in cases:
    for row, col, part in (('f', 'bpol', 'F'), ('f', 'btor', 'G'), ('g', 'bpol', 'F'), ('g', 'btor', 'G')):
        for dl in range(-n - 1, n + 2):
            lc = l + dl
            if lc < max(m, 1):
                continue
            der = d.rows_dr(row, part, lc, l, m, n)
            cand = candidate(row, col, n, l, m, dl)
            label = 'n=%d m=%d: %s row l=%d, %s col l%+d' % (n, m, row, l, part, dl)
            if cand == 0:
                # a block kore does not have: the derivation must vanish there too, up to the round-off of its quadrature
                # in θ, which grows with l (relative to the row's largest coefficient, 1e-9)
                if der and max(abs(v) for v in der.values()) > 1e-9*rowscale(row, part, n, m, l):
                    bad.append(label); print('%-50s derived terms, no kore block  <-- FAIL' % label)
                continue
            if not der:
                bad.append(label); print('%-50s a kore block, no derived terms  <-- FAIL' % label); continue
            c, err = d.compare(der, d.as_dict(cand), label)
            if err > tol(l) or abs(c - 1) > tol(l):
                bad.append(label)
            else:
                n_ok += 1
print('dr_induction_check.py: %d blocks ok, %s' % (n_ok, 'ok' if not bad else 'FAIL: %s' % bad))
sys.exit(1 if bad else 0)
