#!/usr/bin/env python3
'''
derive_mag.py: the magnetic terms of the deformed body (centrifugal = 1 with magnetic = 1, bin/centrifugal.py sections 4 and
5) derived with sympy from their definitions, and the couplings kore builds checked against them (NEW 2026-11-05).

A. From the map alone (Cartesian components, first order in a bookkeeping factor δ: ε -> δ ε), with b = F b̃/J and
   B0 = F B̃0/J (b̃, B̃0 the fluxes of the reference space, any fields):
   1. Fᵀ(a × c) = (ã × c̃)/J for a = F ã/J, c = F c̃/J (any invertible F): the covariant Lorentz force and the electric
      field u × B0 of the induction become (j̃ × B̃0 + J̃0 × b̃)/J and (w × B̃0)/J;
   2. J F⁻¹ (∇_X × b) = ∇_α × (M b̃), M = FᵀF/J: the current is j̃ = ∇×(M b̃), and with Faraday's law (the same identity for
      the electric field) ∂ₜb̃ = -∇×(Fᵀ E);
   3. the first-order parts: (1/J)[∇×(M b̃) × B̃0 + ∇×(M B̃0) × b̃] - (J0×b - B0×J) = δ𝐆 with
      δ𝐆 = -η P2 (J0×b - B0×J) + [∇×((M - I)B0)]×b - B0×[∇×((M - I)b)]   (b, B0, J, J0 of the reference space),
      and w × B̃0/J - w × B̃0 = δ𝐄 = -η P2 (w × B0);
   4. the walls: J F⁻¹ ∇φ(X) = ∇φ + ∇×(∇φ × ξ) + ξ ∇²φ at first order (Nanson: the flux b̃_r of a potential field) and
      ∇_α(φ(X)) = Fᵀ ∇φ(X) (the tangential field), and (ξ⋅∇φ) = ε ∇⋅((r²P2/2)∇φ)-type identity used by the check of C.
B. The couplings of centrifugal.magnetic_build (the recurrence algebra) against the projection of δ𝐆 and δ𝐄 on kore's rows,
   as tests/magnetic/algebra_check.py does for the sphere: δ𝐆 and δ𝐄 built from the closed form of (M - I) (checked by
   derive.py) with ε a symbolic function of r, the rows ρ0 r̂⋅∇×∇×(δ𝐆/ρ0) (u), ρ0 r̂⋅∇×(δ𝐆/ρ0) (v), r̂⋅∇×δ𝐄 (f) and
   r̂⋅∇×∇×δ𝐄 (g) projected on Y_l' by tests/magnetic/derive.py, every block term by term (the amplitudes of the labels are
   E = ε, N = r ε', C = r N', Q = r C').
C. The wall rows of centrifugal.wall_coefficients: their parts δs = (ξ⋅∇φ)_l/R and δq = -[∇×(ξ×∇φ)]_r against closed forms
   (ξ⋅∇φ = ε ∇(r²P2/2)⋅∇φ = (ε/2) ∇²((r²P2/2) φ) for a harmonic φ); tests/centrifugal/bc_check.py checks the rows themselves
   against the exact conditions on the deformed wall.

Usage: python3 derive_mag.py [--quick]   (part B derives its blocks in parallel on up to 16 cores: two and a half minutes
on 16, eight on four, a quarter of an hour on one; --quick checks two of the seven cases of part B; prints each check and
'derive_mag.py: ok', or the failures)
'''
import sys, os, types, re
import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.normpath(os.path.join(HERE, '..', '..', 'bin')), os.path.normpath(os.path.join(HERE, '..', 'magnetic'))]
sys.modules.setdefault('parameters', types.SimpleNamespace(par=types.SimpleNamespace(lmax=40, m=0)))
quick = '--quick' in sys.argv
bad = []

def zero(expr, what):
    '''prints ok if expr (a scalar or a matrix) simplifies to zero'''
    f = lambda v: sp.simplify(sp.expand(v))
    expr = expr.applyfunc(f) if isinstance(expr, sp.MatrixBase) else f(expr)
    ok = all(v == 0 for v in expr) if isinstance(expr, sp.MatrixBase) else expr == 0
    print(('  ok    ' if ok else '  FAIL  ') + what)
    if not ok:
        bad.append(what)

# ======================================================================================================================
def part_a():
    print('A. the covariant forms, from the map (Cartesian, first order in δ)')
    x, y, z, d = sp.symbols('x y z delta', real=True)
    V = (x, y, z)
    Rr = sp.sqrt(x**2 + y**2 + z**2)
    e = sp.Function('e')
    er = e(Rr)
    cth = z/Rr
    P2 = (3*cth**2 - 1)/2
    eta = sp.simplify(Rr*sp.diff(er, x)*Rr/x)              # r ε'
    Phi2 = (z**2 - (x**2 + y**2)/2)/2                      # r² P2/2
    xi = d*er*sp.Matrix([sp.diff(Phi2, v) for v in V])
    F = (sp.Matrix(V) + xi).jacobian(V)
    J = F.det()
    Finv = sp.eye(3) - (F - sp.eye(3))                     # F⁻¹ at first order (F - I is O(δ))
    zh = sp.Matrix([0, 0, 1]); rh = sp.Matrix(V)/Rr
    first = lambda ex: ex.applyfunc(lambda v: sp.diff(v, d).subs(d, 0)) if isinstance(ex, sp.MatrixBase) else sp.diff(ex, d).subs(d, 0)
    at0 = lambda ex: ex.applyfunc(lambda v: v.subs(d, 0)) if isinstance(ex, sp.MatrixBase) else ex.subs(d, 0)
    curl = lambda W: sp.Matrix([sp.diff(W[2], y) - sp.diff(W[1], z), sp.diff(W[0], z) - sp.diff(W[2], x), sp.diff(W[1], x) - sp.diff(W[0], y)])

    Fg = sp.Matrix(3, 3, lambda i, j: sp.Symbol('a%d%d' % (i, j)))
    at = sp.Matrix(sp.symbols('p1 p2 p3')); ct = sp.Matrix(sp.symbols('q1 q2 q3'))
    Jg = Fg.det()
    zero(Fg.T*((Fg*at/Jg).cross(Fg*ct/Jg)) - at.cross(ct)/Jg, '1. Fᵀ(a × c) = (ã × c̃)/J for a = F ã/J, c = F c̃/J (any F)')

    bt = sp.Matrix([sp.Function('b%d' % i)(x, y, z) for i in range(3)])     # b̃(α), any field
    bX = F*bt/J                                                                # b at X(α)
    dX = lambda f, k: sum(sp.diff(f, V[a])*Finv[a, k] for a in range(3))    # ∂/∂X_k by the chain rule (first order)
    jX = sp.Matrix([dX(bX[2], 1) - dX(bX[1], 2), dX(bX[0], 2) - dX(bX[2], 0), dX(bX[1], 0) - dX(bX[0], 1)])
    M = F.T*F/J
    lhs, rhs = J*Finv*jX, curl(M*bt)
    zero(first(lhs - rhs), '2. J F⁻¹ (∇_X × b) = ∇×(M b̃) (first order; with Faraday: ∂ₜb̃ = -∇×(Fᵀ E))')
    zero(at0(lhs - rhs), '   (and at zeroth order)')

    B0t = sp.Matrix([sp.Function('B%d' % i)(x, y, z) for i in range(3)])   # B̃0(α), any field
    wt = sp.Matrix([sp.Function('w%d' % i)(x, y, z) for i in range(3)])
    M1 = lambda v: er*(3*(zh.T*v)[0]*zh - v) + eta*(sp.Rational(3, 2)*cth*((zh.T*v)[0]*rh + (rh.T*v)[0]*zh) - (rh.T*v)[0]*rh - P2*v)
    Gfull = (curl(M*bt).cross(B0t) + curl(M*B0t).cross(bt))/J
    G0 = curl(bt).cross(B0t) + curl(B0t).cross(bt)
    dG = -eta*P2*G0 + curl(M1(B0t)).cross(bt) - B0t.cross(curl(M1(bt)))
    zero(first(Gfull) - dG, '3. δ𝐆 = -η P2 (J0×b - B0×J) + [∇×((M - I)B0)]×b - B0×[∇×((M - I)b)]')
    zero(first(wt.cross(B0t)/J) + eta*P2*wt.cross(B0t), '   δ𝐄 = -η P2 (w × B0)')

    ph = sp.Function('phi')
    # ∇φ at X, written with the derivatives of φ at α expanded to first order: ∂_k φ(X) = ∂_k φ + ξ⋅∇(∂_k φ)
    g0 = sp.Matrix([sp.diff(ph(x, y, z), v) for v in V])
    gX = g0 + sp.Matrix([sum(xi[a]*sp.diff(g0[k], V[a]) for a in range(3)) for k in range(3)])
    lap = sum(sp.diff(ph(x, y, z), v, 2) for v in V)
    zero(first(J*Finv*gX) - (curl(g0.cross(first(xi))) + first(xi)*lap), '4. J F⁻¹ ∇φ(X) = ∇φ + ∇×(∇φ × ξ) + ξ ∇²φ (Nanson, first order)')
    PhiA = ph(x, y, z) + (xi.T*g0)[0]                                          # φ(X(α)) at first order
    zero(first(sp.Matrix([sp.diff(PhiA, v) for v in V]) - F.T*gX), '   ∇_α φ(X(α)) = Fᵀ ∇φ(X) (the tangential field)')
    f2 = Phi2
    zero(first((xi.T*g0)[0]) - er*(sp.Rational(1, 2)*sum(sp.diff(f2*ph(x, y, z), v, 2) for v in V)
                                     - sp.Rational(1, 2)*f2*lap), '   ξ⋅∇φ = (ε/2)[∇²((r²P2/2) φ) - (r²P2/2) ∇²φ] (r²P2 harmonic)')


if '--skip-a' not in sys.argv:
    part_a()

# ======================================================================================================================
print('B. the couplings of centrifugal.magnetic_build against the projected rows of δ𝐆 and δ𝐄')
# With generic radial functions sympy needs minutes per block (the metric doubles the size of every expression); the rows are
# linear in each radial function, so explicit ones with random coefficients, checked at a few radii, test the same identity:
# ε, h, k, ln ρ0 and the unknowns F, G, P, T are polynomials, the rows are built and projected on Y_l' (Gauss-Legendre in θ)
# by tests/magnetic/derive.py's vector calculus, and kore's blocks are evaluated with the same functions.
import derive as dv
import centrifugal as cf
r, th = dv.r, dv.th
EXPL = {'e': sp.Rational(7, 10) + sp.Rational(3, 10)*r**2 - sp.Rational(9, 20)*r**4,
        'h': r/2 + r**3/5 - r**5/10, 'k': sp.Rational(3, 10)*r + r**3/4,
        'F': r**2 + r**3/5 - sp.Rational(3, 10)*r**5, 'G': sp.Rational(2, 5)*r**3 + r**4 - r/10,
        'P': 1 - r**2/5 + sp.Rational(3, 5)*r**4 - sp.Rational(1, 3)*r**5, 'T': sp.Rational(1, 5) + r**3 - r**6/2,
        'lr': -sp.Rational(11, 10)*r**2 + sp.Rational(7, 20)*r**3}
dv.P, dv.T, dv.F, dv.G, dv.h, dv.kt, dv.lr = (EXPL[k] for k in ('P', 'T', 'F', 'G', 'h', 'k', 'lr'))
eps = EXPL['e']
RADII = np.array([0.45, 0.7, 0.93])
lam = lambda ex: (lambda f: (lambda rr: np.broadcast_to(np.asarray(f(rr), dtype=float), np.shape(rr))))(sp.lambdify(r, ex, 'numpy'))
FUNS = {k: [lam(sp.diff(EXPL[k], r, j)) for j in range(6)] for k in ('h', 'k', 'F', 'G', 'P', 'T', 'lr')}
a_ = [eps]
for _ in range(3):
    a_.append(sp.expand(r*sp.diff(a_[-1], r)))
AMPF = {a: lam(ex) for a, ex in zip('ENCQ', a_)}   # E = ε, N = r ε', C = r N', Q = r C'

def metric(v):
    '''(M - I) v of a vector v = (v_r, v_θ, v_φ), the closed form of centrifugal.py (section 1)'''
    c, s_ = sp.cos(th), sp.sin(th)
    et = r*sp.diff(eps, r)
    p2 = (3*c**2 - 1)/2
    vr, vt, vp = v
    vz = vr*c - vt*s_
    zs = (c, -s_, 0)
    return tuple(eps*(3*vz*zs[i] - v[i]) + et*(sp.Rational(3, 2)*c*(vz*int(i == 0) + vr*zs[i]) - vr*int(i == 0) - p2*v[i])
                 for i in range(3))

def project_num(ex, lrow, m):
    '''the Y_lrow coefficient of ex(r, θ) at RADII (quadrature in θ, derive.py's nodes)'''
    f = sp.lambdify((r, th), ex, 'numpy')
    yr = sp.lambdify(th, dv.Y(lrow, m), 'numpy')(dv.THQ)
    norm = 4*np.pi/(2*lrow + 1)
    return np.array([2*np.pi*np.sum(dv.WQ*np.broadcast_to(np.asarray(f(ri, dv.THQ), dtype=complex), dv.THQ.shape)
                                    *np.conj(yr)*np.sin(dv.THQ))/norm for ri in RADII])

def rows_metric(kind, part, lcol, lrow, m, b0):
    '''the projected row of δ𝐆 (kind u, v; b of part F, G) or δ𝐄 (kind f, g; u of part P, T) at RADII'''
    dv.B0_KIND = b0
    B0 = dv.B0()
    et = r*sp.diff(eps, r)
    p2 = (3*sp.cos(th)**2 - 1)/2
    if kind in ('u', 'v'):
        b = dv.bfield(lcol, m, part)
        G0 = dv.add(dv.cross(dv.curl(B0, 0), b), dv.scal(-1, dv.cross(B0, dv.curl(b, m))))   # J0×b - B0×J
        dG = dv.add(dv.add(dv.scal(-et*p2, G0), dv.cross(dv.curl(metric(B0), 0), b)),
                    dv.scal(-1, dv.cross(B0, dv.curl(metric(b), m))))
        rho = sp.exp(dv.lr)
        Gr = dv.scal(1/rho, dG)
        ex = dv.curl(dv.curl(Gr, m), m)[0] if kind == 'u' else dv.curl(Gr, m)[0]
        return project_num(rho*ex, lrow, m)
    E = dv.scal(-et*p2, dv.cross(dv.velocity(lcol, m, part), B0))
    ex = dv.curl(E, m)[0] if kind == 'f' else dv.curl(dv.curl(E, m), m)[0]
    return project_num(ex, lrow, m)

def candidate(blocks, row, X, l, dl):
    '''kore's block at RADII: -Σ coefficient r^(-rx) (u, v) or r^(-1-rx) (f, g) a B_n^(j) λ_k X^(d) (as algebra_check.py)'''
    out = np.zeros(len(RADII), dtype=complex)
    for lab, Mx in blocks.get((row, X), {}).items():
        c = complex(Mx[l, l + dl]) if 0 <= l + dl < Mx.shape[1] else 0
        if c == 0:
            continue
        rr, rx, a, B, n, j, lm, dd = re.match(r'([uvfg])(\d)([ENCQ])([pt])(\w)(\d)(?:lho(\d))?(?:ohr1)?_D(\d)$', lab).groups()
        p = -int(rx) if rr in 'uv' else -1 - int(rx)
        val = RADII**p*AMPF[a](RADII)*FUNS['h' if B == 'p' else 'k'][int(j)](RADII)*FUNS[X][int(dd)](RADII)
        if lm:
            val = val*FUNS['lr'][int(lm)](RADII)
        out -= c*val
    return out

n_ok = 0
cases = [('pol', 1, 1, 3), ('tor', 1, 0, 3)] if quick else \
        [('pol', 1, 1, 3), ('pol', 1, 0, 2), ('pol', 2, 2, 4), ('pol', 3, 1, 4), ('tor', 1, 1, 3), ('tor', 2, 0, 4), ('tor', 2, 2, 3)]
ROWS = (('u', 'F'), ('u', 'G'), ('v', 'F'), ('v', 'G'), ('f', 'P'), ('f', 'T'), ('g', 'P'), ('g', 'T'))
BLOCKS = {}
for kind, n, m, l in cases:
    blocks = cf.magnetic_build(m, {'h' if kind == 'pol' else 'k': [n]}, l + n + 16)
    BLOCKS[(kind, n, m, l)] = (blocks, cf._bandwidth(blocks))

def one_block(task):
    '''one block: the projected row (row, column X) of the column degree l + dl, for one case'''
    (kind, n, m, l), row, X, dl = task
    return task, rows_metric(row, X, l + dl, l, m, (kind, n))

# NEW (2026-11-05): the blocks are independent, so they are derived in parallel (about a quarter of an hour on one core);
# the u rows, the longest, first
TASKS = [(c, row, X, dl) for row, X in ROWS for c in cases
         for dl in range(-BLOCKS[c][1] - 1, BLOCKS[c][1] + 2) if c[3] + dl >= max(c[2], 1)]
try:
    import multiprocessing
    ncpu = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    with multiprocessing.get_context('fork').Pool(max(1, min(ncpu, 16))) as pool:
        DERIVED = dict(pool.imap_unordered(one_block, TASKS))
except ValueError:          # no fork on this platform: one block after the other
    DERIVED = dict(map(one_block, TASKS))
for kind, n, m, l in cases:
    B = BLOCKS[(kind, n, m, l)][1]
    worst = 0
    for row, X in ROWS:
        der = {t[3]: v for t, v in DERIVED.items() if t[:3] == ((kind, n, m, l), row, X)}
        scale = max(np.abs(v).max() for v in der.values())
        for dl, dvals in sorted(der.items()):
            cvals = candidate(BLOCKS[(kind, n, m, l)][0], row, X, l, dl)
            diff = np.abs(dvals - cvals).max()
            err = diff/scale if scale > 0 else (0. if diff == 0 else np.inf)   # (a row block that vanishes: m = 0, toroidal B0)
            worst = max(worst, err)
            if err > 1e-11:
                bad.append('%s n=%d m=%d %s%s l=%d dl=%+d' % (kind, n, m, row, X, l, dl))
                print('  FAIL  %s n=%d m=%d: %s row l=%d, %s col l%+d: derived %s, kore %s (rel. %.1e)'
                      % (kind, n, m, row, l, X, dl, np.round(dvals, 8), np.round(cvals, 8), err))
            else:
                n_ok += 1
    print('  %s  %s n=%d m=%d l=%d: every block (u, v, f, g rows, |Δl| <= %d) against the projected rows, worst %.1e'
          % ('ok  ' if worst <= 1e-11 else 'FAIL', kind, n, m, l, B + 1, worst))
print('  %d blocks agree' % n_ok)

# ======================================================================================================================
print('C. the parts δs and δq of the wall rows (centrifugal.wall_coefficients) against closed forms')
import scipy.sparse as sps
import angular
import mag_algebra as ma
from scipy.special import lpmv
from math import factorial
def Ylm(l, m, xx):
    '''kore's Schmidt harmonic, θ part, from scipy (lpmv has the Condon-Shortley phase)'''
    return np.sqrt(factorial(l - m)/factorial(l + m))*lpmv(m, l, xx)
def dYlm(l, m, th_, h=1e-4):
    '''∂θ of Ylm by a fourth-order difference'''
    f = lambda t: Ylm(l, m, np.cos(t))
    return (-f(th_ + 2*h) + 8*f(th_ + h) - 8*f(th_ - h) + f(th_ - 2*h))/(12*h)
xq, wq = np.polynomial.legendre.leggauss(64)
thq, snq = np.arccos(xq), np.sqrt(1 - xq**2)
lw = 30
lv = np.arange(lw + 1)
for m in (0, 1, 3):
    A = angular.Algebra.get(m, lw)
    P2m = A.legendre(2)[0].toarray()
    for wall in ('outer', 'inner'):
        # the matrices of wall_coefficients: ∇φ per unit c (q = -(l+1) outside, l inside; s = 1), Ds and Dq
        vq = -(lv + 1.) if wall == 'outer' else lv*1.
        V = A.field(q=sps.diags(vq).tocsr() @ A.I, s=A.I)
        Ds = (A.legendre(2)[0] @ V.q - 1.5*(A.X @ A.sin_theta(V))).toarray()
        Dq = (ma._vcross(A, 'r', cf._P2p, V) + ma._vcross(A, 'st', -1.5*cf._xp, V)).t.toarray()
        # δs/ε = (ξ⋅∇φ)/(ε R) from ξ⋅∇φ = (ε/2)∇²((r²P2/2)φ): (c_l/4) [P2]_l'l [(1 - l)(2 - l) - l'(l'+1)] outside (φ ∝ r^-(l+1)),
        # (c_l/4) [P2]_l'l [(l + 2)(l + 3) - l'(l'+1)] inside (φ ∝ r^l)
        k = (1 - lv)*(2 - lv) if wall == 'outer' else (lv + 2)*(lv + 3)
        Ds_closed = P2m*(k[None, :] - (lv*(lv + 1))[:, None])/4
        rg = slice(m, lw - 6)
        err = np.abs(Ds - Ds_closed)[rg, rg].max()
        print('  %s  m=%d %-5s δs = (ξ⋅∇φ)/R against (ε/2)∇²((r²P2/2)φ): max diff %.1e' % ('ok  ' if err < 1e-12 else 'FAIL', m, wall, err))
        if err > 1e-12:
            bad.append('δs %s m=%d' % (wall, m))
        # δq/(-ε L) = the T part of (ξ/(ε R)) × ∇φ, ξ/(ε R) = P2 r̂ - (3/2) x sinθ θ̂, by quadrature: T_l = (i m Y/sinθ) θ̂ - ∂θY φ̂,
        # t_l' = ∫ A⋅T_l'* dΩ/(L' n_l')
        errq = 0
        for ls in range(max(m, 1), 12):
            Y = Ylm(ls, m, xq)
            gr, gt, gp = vq[ls]*Y, dYlm(ls, m, thq), 1j*m*Y/snq
            xr, xt = cf._P2p(xq), -1.5*xq*snq
            At, Ap = -xr*gp, xr*gt - xt*gr          # (ξ×∇φ)_θ = -ξ_r g_φ, (ξ×∇φ)_φ = ξ_r g_θ - ξ_θ g_r (ξ_φ = 0)
            for lt in range(max(m, 1), 12):
                Yt, dYt = Ylm(lt, m, xq), dYlm(lt, m, thq)
                num = 2*np.pi*np.sum(wq*(At*np.conj(1j*m*Yt/snq) - Ap*dYt))
                errq = max(errq, abs(num/(lt*(lt + 1)*4*np.pi/(2*lt + 1)) - Dq[lt, ls]))
        print('  %s  m=%d %-5s δq: the T part of (P2 r̂ - (3/2) x sinθ θ̂)×∇φ against a quadrature: max diff %.1e' % ('ok  ' if errq < 1e-9 else 'FAIL', m, wall, errq))
        if errq > 1e-9:
            bad.append('δq %s m=%d' % (wall, m))

print('derive_mag.py: %s' % ('ok' if not bad else 'FAIL: %s' % bad))
sys.exit(1 if bad else 0)
