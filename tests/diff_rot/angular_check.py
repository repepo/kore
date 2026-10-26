#!/usr/bin/env python3
'''
angular_check.py: tests of bin/angular.py (the recurrence algebra of the angular couplings) and of the primitives of
the differential-rotation force in bin/diff_rot.py built on it. Standalone: needs no run directory (a stub parameters module is used).

    python3 tests/diff_rot/angular_check.py [--quick]

 1. harmonics: angular.ylm_grid against exact Legendre values (m = 0, the recurrence in rational arithmetic at the
    float x, up to l = 2000), against scipy.special.sph_harm_y (m = 0, 1, 2, 7, 30, l <= 640: scipy is NaN from
    l = 646 on), and their norms ∫|Y_l|² dΩ = 4π/(2l+1), ∫(|∂θY_l|² + |Z_l|²) dΩ = L 4π/(2l+1) up to l = 3000;
 2. the matrices on the grid: Σ_k X[k,l] Y_k = x Y_l, Σ_k D[k,l] Y_k = -sinθ ∂θY_l, Z = i m Y/sinθ, and the
    adjoints X† = X, D† = 2X - D;
 3. the vector operations (multiplication by a polynomial g(x), ẑ×, ẑ×(c r̂), sinθ F_θ) on R_l, S_l, T_l against
    projections of the same fields built on a Gauss-Legendre grid, at low degree and in a window at high degree;
 4. the 12 primitives of diff_rot._primitives against the former quadrature builder (quad_reference.py), n = 0 ... 16.

Tolerances. Exact and algebraic references: 1e-12 (relative to the envelope of P_l, or to the largest entry). References
evaluated on a grid (scipy, quadrature projections, norms): their own floor is about eps l² near the poles (an error
eps in a node moves a degree-l harmonic by eps l²), so the tolerance is max(1e-11, 1e-15 l²) at the largest degree
used. Exit code 1 on failure.
'''
import sys, os, types
from fractions import Fraction
import numpy as np
import scipy.sparse as sp
import scipy.special as scsp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.normpath(os.path.join(HERE, '..', '..', 'bin')), HERE]
sys.modules.setdefault('parameters', types.SimpleNamespace(par=types.SimpleNamespace(lmax=60)))
import angular
import diff_rot as dr
import quad_reference as qr

quick = '--quick' in sys.argv
res = {}   # name: [worst error, tolerance]

def report(name, err, tol):
    e, t = res.get(name, [0., tol])
    res[name] = [max(e, err), min(t, tol)] if name in res else [err, tol]

def gtol(l):   # tolerance of a grid-evaluated reference at degree l
    return max(1e-11, 1e-15*l**2)

def dense(M):
    return np.asarray(M.todense() if sp.issparse(M) else M)

def rel(a, b, scale=None):
    s = np.abs(b).max() if scale is None else scale
    return np.abs(np.asarray(a) - np.asarray(b)).max()/s if s > 0 else np.abs(a).max()

# --- 1. harmonics ----------------------------------------------------------------------------------------------------
LX = 2000 if not quick else 700
for xf in (1/3, 0.999, -0.7):                       # exact P_l at the float x, by the recurrence in rational arithmetic
    xs = Fraction(xf); P0, P1 = Fraction(1), xs; exact = {}
    keep = set(range(100, LX + 1, 100))
    for l in range(1, LX):
        P0, P1 = P1, ((2*l + 1)*xs*P1 - l*P0)/(l + 1)
        if l + 1 in keep:
            exact[l + 1] = float(P1)
    Y = angular.ylm_grid(0, LX, [xf])[0][:, 0]
    for l, v in exact.items():
        report('harmonics vs exact P_l (l <= %d)' % LX, abs(Y[l] - v)/np.sqrt(2/(np.pi*l*np.sqrt(1 - xf**2))), 1e-12)

x, w = np.polynomial.legendre.leggauss(700)
th = np.arccos(x)
for m in (0, 1, 2, 7, 30):
    Y, dY, Z = angular.ylm_grid(m, 640, x)
    for l in range(m, 641, 3 if not quick else 29):
        y, dy = scsp.sph_harm_y(l, m, th, 0.0, diff_n=1)
        c = np.sqrt(4*np.pi/(2*l + 1))
        report('harmonics vs scipy (l <= 640)', rel(Y[l], y.real*c), gtol(640))
        if l > 0:
            report('harmonics vs scipy (l <= 640)', rel(dY[l], dy[:, 0].real*c), gtol(640))

LH = 3000 if not quick else 1000
x, w = np.polynomial.legendre.leggauss(LH + 20)
sth = np.sqrt(1 - x**2)
for m in (0, 2, 50):
    Y, dY, Z = angular.ylm_grid(m, LH, x)
    l = np.arange(LH + 1)
    nl = 4*np.pi/(2*l + 1)
    k = l >= max(m, 1)
    n0 = 2*np.pi*(Y**2 @ w)
    n1 = 2*np.pi*((dY**2 + np.abs(Z)**2) @ w)
    report('harmonics: norms (l <= %d)' % LH, np.abs(n0[k]/nl[k] - 1).max(), gtol(LH))
    report('harmonics: norms (l <= %d)' % LH, np.abs(n1[k]/(l[k]*(l[k] + 1)*nl[k]) - 1).max(), gtol(LH))
    sub = np.arange(max(m, 1), LH + 1, 97)
    G = 2*np.pi*((Y[sub]*w) @ Y[sub].T)/np.sqrt(np.outer(nl[sub], nl[sub]))
    report('harmonics: norms (l <= %d)' % LH, np.abs(G - np.diag(np.diag(G))).max(), gtol(LH))
    # --- 2. the matrices on the grid ---------------------------------------------------------------------------------
    A = angular.Algebra.get(m, LH)
    j = slice(m, LH)                                 # the last degree has no Y_{l+1} on the grid
    for a, b in ((A.X.T @ Y, x*Y), (A.D.T @ Y, -sth*dY)):
        report('X, D on the grid (l <= %d)' % LH, max(rel(a[i], b[i]) for i in range(m, LH)), gtol(LH))
    if m:
        report('X, D on the grid (l <= %d)' % LH, max(rel(Z[i], 1j*m*Y[i]/sth) for i in range(m, LH + 1)), gtol(LH))
    if LH <= 1200 or m == 2:
        N = A.N.toarray(); Ni = np.diag(1/np.diag(N))
        report('adjoints X† = X, D† = 2X - D', rel(Ni @ A.X.toarray().T @ N, A.X.toarray()), 1e-12)
        report('adjoints X† = X, D† = 2X - D', rel(Ni @ A.D.toarray().T @ N, A.Dd.toarray()), 1e-12)

# large m at large l: sin^m θ underflows near the poles where Y_l is not small (ylm_grid rescales by powers of two)
for m, L in (((400, 3000), (1425, 3000)) if not quick else ((1425, 3000),)):
    x, w = np.polynomial.legendre.leggauss(L + 20)
    Y, dY, Z = angular.ylm_grid(m, L, x)
    l = np.arange(L + 1); nl = 4*np.pi/(2*l + 1); k = l >= m
    report('harmonics: norms at large m (l <= 3000)', np.abs((2*np.pi*(Y**2 @ w))[k]/nl[k] - 1).max(), gtol(L))
    report('harmonics: norms at large m (l <= 3000)',
           np.abs((2*np.pi*((dY**2 + np.abs(Z)**2) @ w))[k]/(l[k]*(l[k] + 1)*nl[k]) - 1).max(), gtol(L))

# --- 3. vector operations against grid projections -------------------------------------------------------------------
def ops_check(m, lw, sources, gc):
    '''sources: degrees of the R, S, T fields; gc: Legendre coefficients of g(x); targets below lw - 8 (truncation)'''
    A = angular.Algebra.get(m, lw)
    g = sum(c*A.legendre(k)[0] for k, c in enumerate(gc))
    dg = sum(c*A.legendre(k)[1] for k, c in enumerate(gc))
    G = np.polynomial.legendre.Legendre(gc)
    xq, wq = np.polynomial.legendre.leggauss(lw + len(gc) + 8)
    sq = np.sqrt(1 - xq**2)
    Y, dY, Z = angular.ylm_grid(m, lw, xq)
    l = np.arange(lw + 1); L = l*(l + 1.); nl = 4*np.pi/(2*l + 1)
    tgt = np.arange(max(m, 1), lw - 8)
    def proj(Fr, Ft, Fp):   # (q, s, t) on the target degrees, eq. (3.4) of docs/differential_rotation_details.md
        return (2*np.pi*(Y[tgt]*wq) @ Fr/nl[tgt],
                2*np.pi*((dY[tgt]*wq) @ Ft + (np.conj(Z[tgt])*wq) @ Fp)/(L*nl)[tgt],
                2*np.pi*((np.conj(Z[tgt])*wq) @ Ft - (dY[tgt]*wq) @ Fp)/(L*nl)[tgt])
    zc = lambda Fr, Ft, Fp: (-sq*Fp, -xq*Fp, sq*Fr + xq*Ft)   # ẑ × F, eq. (2.5)
    E = np.eye(lw + 1)
    tol = gtol(lw)
    for src in sources:
        e = E[:, [src]]
        fields = {'R': ((Y[src], 0*xq, 0*xq), A.field(q=e)), 'S': ((0*xq, dY[src], Z[src]), A.field(s=e)),
                  'T': ((0*xq, Z[src], -dY[src]), A.field(t=e))}
        gx = G(xq)
        for kind, (F, Fa) in fields.items():
            for name, ref, alg in (('g F', proj(gx*F[0], gx*F[1], gx*F[2]), A.mult(Fa, g, dg)),
                                   ('z x F', proj(*zc(*F)), A.zcross(Fa))):
                scale = max(np.abs(r).max() for r in ref)   # the field's scale: largest of q, s, t
                for k, r in zip('qst', ref):
                    report('vector operations vs grid (lw %d): %s' % (lw, name), rel(dense(alg[k])[tgt, 0], r, scale), tol)
            st = 2*np.pi*(Y[tgt]*wq) @ (sq*F[1])/nl[tgt]
            report('vector operations vs grid (lw %d): sin F_theta' % lw, rel(dense(A.sin_theta(Fa))[tgt, 0], st), tol)
        ref = proj(*zc(Y[src], 0*xq, 0*xq)); alg = A.zcross_r(e)
        scale = max(np.abs(r).max() for r in ref)
        for k, r in zip('qst', ref):
            report('vector operations vs grid (lw %d): z x (c r)' % lw, rel(dense(alg[k])[tgt, 0], r, scale), tol)

rng = np.random.default_rng(1)
for m in (0, 1, 3):
    ops_check(m, 60, range(max(m, 1), 45, 1 if not quick else 7), rng.standard_normal(5))
if quick:
    ops_check(2, 700, (600, 601), rng.standard_normal(4))
else:
    ops_check(2, 1300, (1200, 1201, 1255), rng.standard_normal(4))
    ops_check(17, 1300, (1100, 1250), rng.standard_normal(6))

# --- 4. primitives against the former quadrature ---------------------------------------------------------------------
for n in ((0, 2, 4, 8, 16) if not quick else (0, 2, 6)):
    for m in (0, 1, 2, 5):
        lw = 120
        Cq = qr.primitives(n, m, lw)
        Ca, _ = dr._primitives(n, m, lw)
        j, k0 = max(m, 1), lw - n - 4
        for k in 'qst':
            for sh in ('q', 'a', 'T', 'dq'):
                a, b = Ca[k][sh].toarray()[j:k0, j:k0], Cq[k][sh][j:k0, j:k0]
                report('primitives vs former quadrature (lw 120)', rel(a, b) if np.abs(b).max() > 0 else np.abs(a).max(), gtol(lw))

ok = True
for name, (e, t) in res.items():
    flag = e < t
    ok &= flag
    print('  %-52s %.1e  (tol %.0e)  %s' % (name, e, t, 'ok' if flag else 'FAIL'))
print('angular_check:', 'ok' if ok else 'FAIL')
sys.exit(0 if ok else 1)
