#!/usr/bin/env python3
'''
row_check.py: the magnetic rows of a run against the independent derivation of derive.py (run in a run directory, after
submatrices.py; ricb > 0, or a full sphere: NEW 2026-10-28; poloidal B0 of degree 1 or 2, toroidal B0).

For random smooth P, T, f, g of low degree it applies kore's operators (operators.lorentz, operators.induction) block by
block, reads each row block as a series in its Gegenbauer basis (C^(4) for u, C^(2) for v, f, g) on an r grid, and compares
it with derive.rows(): the same physics projected on the harmonics by exact θ-quadrature, in monomials of P^(k), h^(k),
(ln ρ0)^(k), evaluated here with this run's profiles (radial_profiles.logrhoX, background_field.bpoX). The comparison is
unweighted, max over r of |kore - derived| / max |derived| for each (row l, column l) block, so it sees the near-surface
part of the rows. Expected row scales (A sign): induction -r³ (.)/L³, Lorentz -Hendrik r⁵ (.)/L³ (u) and r³ (v).
In a full sphere (ricb = 0) the columns hold only the Chebyshev polynomials of their vector parity and the rows the Gegenbauer
ones of the row's parity (after the BC rows, as submatrices.py's remroco and chop leave them; the row parity is taken as the
one of the two that matches), r in (0, 1]; with Ek = 0 the u rows are in C^(3).
usage: python3 row_check.py [lmax_check] [tol]     (lmax_check defaults to m + 4; needs sympy; tol defaults to 1e-8)
'''
import sys, os
sys.path.insert(0, os.path.abspath('bin'))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import numpy.polynomial.chebyshev as ch
from parameters import par
import utils as ut
import radial_profiles as rap
import background_field as bf
import operators as op
import derive

m = par.m
lmc = int(sys.argv[1]) if len(sys.argv) > 1 else m + 4
tol = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-8
assert par.magnetic
N, N1, ri = par.N, ut.N1, par.ricb
FS = (ri == 0)   # NEW (2026-10-28): full sphere
lp, lt, _ = [list(a) for a in ut.ell(m, par.lmax, par.symm)]
lf, lg = (lt, lp) if ut.symmB0 == -1 else (lp, lt)
rng = np.random.default_rng(7)

r  = 0.5*(1 - ri)*(np.cos(np.pi*(np.arange(200) + 0.5)/200)[::-1] + 1) + ri
x  = 2*(r - ri)/(1 - ri) - 1
dx = 2/(1 - ri)
if FS:   # x = r on [-1, 1]
    r = 0.1 + 0.45*(np.cos(np.pi*(np.arange(200) + 0.5)/200)[::-1] + 1); x = r; dx = 1.   # (away from r = 0, where the r^-p of the monomials cancel)

def gegen(lamb):   # C^(lamb) polynomials at x, (len(x), N)
    E = np.zeros((len(x), N)); E[:, 0] = 1; E[:, 1] = 2*lamb*x
    for k in range(2, N):
        E[:, k] = (2*(k + lamb - 1)*x*E[:, k-1] - (k + 2*lamb - 2)*E[:, k-2])/k
    return E
gu = 3 if (FS and par.ViscosD == 0) else 4   # as submatrices.py
GB = {'u': (gu, gegen(gu)), 'v': (2, gegen(2)), 'f': (2, gegen(2)), 'g': (2, gegen(2))}

def row_vals(y, sec, q0=0):
    gb, E = GB[sec]
    c = np.zeros(N, dtype=complex)
    if FS:   # rows: chop BC rows, then the C^(gb) coefficients q0, q0 + 2, ...
        chop = 0 if (sec == 'v' and par.ViscosD == 0) else gb//2
        c[q0::2][:N1 - chop] = y[chop:]
    else:
        c[:N - gb] = y[gb:]
    return E @ c

vP = int((1 - 2*(m % 2))*par.symm); vF = ut.symmB0*vP
VPAR = {'P': vP, 'T': -vP, 'F': vF, 'G': -vF}
def rand_coef(part):
    k = N1//6
    c = np.zeros(N, dtype=complex); c[:k] = np.exp(-np.arange(k)/3.)*(rng.standard_normal(k) + 1j*rng.standard_normal(k))
    if FS:   # only the Chebyshev polynomials of the column's parity, k of them
        c = np.zeros(N, dtype=complex); c[(0 if VPAR[part] == 1 else 1)::2][:k] = np.exp(-np.arange(k)/3.)*(rng.standard_normal(k) + 1j*rng.standard_normal(k))
    return c

def to_kore(c, part):   # the column vector as kore stores it
    return c[(0 if VPAR[part] == 1 else 1)::2][:N1] if FS else c[:N1]

def derivs(c, n=5):
    out, d = [], c.copy()
    for j in range(n):
        out.append(ch.chebval(x, d)); d = ch.chebder(d)*dx
    return out

prof = {'h': [bf.bpoX(r, k) for k in range(4)], 'lr': [rap.logrhoX(r, k) for k in range(4)],
        'k': [bf.btoX(r, k) for k in range(4)]}   # NEW (2026-10-28): the toroidal B0 (zero without it)
B0S = ([('pol', bf.B0_l)] if par.B0 is not None else []) + ([('tor', par.B0_tor_l)] if par.B0_tor is not None else [])

def evaluate(dct, fun):
    tot = np.zeros_like(r, dtype=complex)
    for (mono, rp), c in dct.items():
        v = c*r**rp
        for (name, k), p in mono:
            v = v*(fun[k] if name in 'PTFG' else prof[name][k])**p
        tot += v
    return tot

worst = 0.
D3 = (-2, -1, 0, 1, 2)   # NEW (2026-10-28): the degree-2 fields couple l to l ± 2
for sec, kind_cols in (('f', (('upol', 'P', lp, D3), ('utor', 'T', lt, D3))),
                       ('g', (('upol', 'P', lp, D3), ('utor', 'T', lt, D3))),
                       ('u', (('bpol', 'F', lf, D3), ('btor', 'G', lg, D3))),
                       ('v', (('bpol', 'F', lf, D3), ('btor', 'G', lg, D3)))):
    lrows = {'f': lf, 'g': lg, 'u': lp, 'v': lt}[sec]
    for l in [l for l in lrows if l <= lmc]:
        L = l*(l + 1.)
        for comp, part, lcols, dls in kind_cols:
            for i in dls:
                lc = l + i
                if lc not in lcols:
                    continue
                fn = op.lorentz if sec in 'uv' else op.induction
                mtx, offd = fn(l, sec, comp, i)
                dct = {}
                for b0 in B0S:   # the poloidal and the toroidal B0 (their monomials, in h or k, are distinct)
                    dct.update(derive.rows(sec, part, lc, l, m, b0=b0))
                if np.isscalar(mtx):   # no coupling in kore: there must be none in the derivation either
                    if dct:
                        worst = 1.; print('row %s l=%2d  column %s l=%2d: derived terms but no kore block  <-- FAIL' % (sec, l, part, lc))
                    continue
                c = rand_coef(part)
                y = (mtx @ to_kore(c, part)).astype(complex)
                scale = -(par.Hendrik if sec in 'uv' else 1.)*r**(5 if sec == 'u' else 3)/L**3
                ref = scale*evaluate(dct, derivs(c))
                kore = row_vals(y, sec)
                if FS:   # the row parity: the one that matches
                    k1 = row_vals(y, sec, 1)
                    if np.max(np.abs(k1 - ref)) < np.max(np.abs(kore - ref)): kore = k1
                big = np.max(np.abs(ref))
                if big == 0:   # a block that vanishes (e.g. the g rows from P at m = 0): kore's must vanish too
                    err = np.max(np.abs(kore))/max(np.max(np.abs(mtx.toarray())), 1e-300) if np.max(np.abs(kore)) > 0 else 0.
                else:
                    err = np.max(np.abs(kore - ref))/big
                worst = max(worst, err)
                print('row %s l=%2d  column %s l=%2d: %.2e %s' % (sec, l, part, lc, err, '' if err < tol else '  <-- FAIL'))
print('row_check.py (%s, B0 %s, B0_tor %s, m %d): worst %.2e -> %s' % (par.structure, par.B0, par.B0_tor, m, worst, 'ok' if worst < tol else 'FAIL'))
sys.exit(0 if worst < tol else 1)
