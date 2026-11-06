#!/usr/bin/env python3
'''
spheroid_check.py: the exact spheroid (centrifugal_shape 'spheroid', bin/centrifugal.py section 8) checked from its
definitions and against solutions computed without kore (NEW 2026-11-06).

The body is the image of the unit ball under the linear map X = F α, F = diag(a, a, c), c/a = 1 - f, a² c = 1, filled with
a homogeneous fluid; the field of the reference space is w = J F⁻¹ u = F⁻¹ u (J = 1). Each check prints one verdict line:

  formulas   (sympy, exact rationals, an oblate and a prolate (a, c)) the terms of section 8's header from the map, for
             polynomial fields (w solenoidal): the covariant inertia Fᵀ u, Coriolis force Fᵀ(ẑ×u) and viscous force
             Fᵀ∇_X²u minus the sphere's, the traction Fᵀσ_X F⁻ᵀ𝐫 (σ_X the strain rate of the body) and the flux through the
             level surfaces, the heat diffusion; the pressure gradient, buoyancy Fᵀ∇_X(r²/2) = 𝐫, advection and mass
             conservation keep their spherical form. Also: the effective potential of a homogeneous body in hydrostatic
             equilibrium (Maclaurin) is a function of r, so its gravity is normal to the level surfaces; to_physical
             against the velocity F w in the spherical basis of the physical point.
  couplings  kore's couplings (centrifugal.spheroid_build: the inertia, Coriolis, viscous and heat terms in the rows u, v,
             h) and wall rows (centrifugal.spheroid_walls: stress-free and fixed flux) against the projections of those
             formulas on kore's rows, with explicit radial functions (derive_diff.py's Laurent polynomials in r, cos θ,
             sin θ). Here (ẑ⋅∇)v is taken from the velocity gradient in spherical components and ∇²w = ∇⋅σ(w), not with
             the identities that centrifugal.py uses. m = 0..3, f = 0.3 and -0.2, the walls at r = 1 and 0.35.
  inviscid   kore's inertial modes against the exact ones (spheroid_modes.Space, the polynomial method, degree <= 7),
             m = 0..3, both symmetries, f = 0.3 and -0.2; the spin-over 2/(1 + (1 - f)²)
  viscous    the four least-damped modes at Ek = 0.1 with no-slip and with stress-free walls, against an exact Galerkin
             solver: solenoidal polynomial fields of one m and one symmetry (poloidal and toroidal potentials times powers
             of 1 - |α|², in the components u_x ± i u_y and u_z), integrals as exact rationals (moments of the ball),
             Cholesky at 50 digits; the stress-free rows enter only through the weak form ∫ σ:σ (a natural condition)
  heat       the first four diffusion modes, fixed temperature and fixed flux, against the same Galerkin for scalars
  rotation   stress-free walls of a shell: the rigid rotation about the axis has the eigenvalue 0

Usage: python3 spheroid_check.py   (about two minutes; builds its runs with mk_run.py in the current directory)
'''
import os, sys, re, time
from fractions import Fraction as Fr
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import derive_diff as dd                       # its Laurent polynomials (and a parameters module for centrifugal.py)
import spheroid_modes as sm
cf = dd.cf
Q = Fr
fails = []
T0 = time.time()

def verdict(name, ok, msg):
    print('%-10s %s  %s' % (name + ':', 'ok  ' if ok else 'FAIL', msg), flush=True)
    if not ok:
        fails.append(name)

def axes(f):
    return (1 - f)**(-1/3), (1 - f)**(2/3)

# ======================================================================================================================
# formulas
def part_formulas():
    import sympy as sp
    x, y, z = sp.symbols('x y z', real=True)
    V = (x, y, z)
    al, zh = sp.Matrix(V), sp.Matrix([0, 0, 1])
    grad = lambda g: sp.Matrix([sp.diff(g, v) for v in V])
    div = lambda A: sum(sp.diff(A[i], V[i]) for i in range(3))
    curl = lambda A: sp.Matrix([sp.diff(A[2], y) - sp.diff(A[1], z), sp.diff(A[0], z) - sp.diff(A[2], x),
                                sp.diff(A[1], x) - sp.diff(A[0], y)])
    jac = lambda A: sp.Matrix(3, 3, lambda i, k: sp.diff(A[i], V[k]))           # (∇A)_ik = ∂_k A_i
    lap = lambda g: sum(sp.diff(g, v, 2) for v in V)
    zgrad = lambda A: grad(A[2]) - zh.cross(curl(A))                             # (ẑ⋅∇)A, as centrifugal.py
    psi = sp.Matrix([Q(1, 3)*x*y*z + y**3 - Q(2, 5)*z**4 + x**2*z, Q(3, 7)*x**3*z - y*z**2 + Q(1, 2)*x**4,
                     x*y**2 - Q(5, 3)*z**3*y + Q(2, 9)*x**2*y**2])
    w = curl(psi)                                                               # solenoidal
    S = Q(3, 4) - x*z + Q(2, 3)*y**2*z**2 + x**3*y - Q(1, 5)*z**4 + Q(4, 7)*x**2*y*z
    Pr = x**2*y - Q(1, 3)*z**3 + y*z
    worst = 0
    for a, c in ((Q(10, 9), Q(81, 100)), (Q(4, 5), Q(25, 16))):                # oblate, prolate; a² c = 1
        assert a*a*c == 1
        F = sp.diag(a, a, c); Fi = F.inv(); Qm = F.T*F
        s_ = (a, a, c)
        DX = lambda g, k: sp.diff(g, V[k])/s_[k]                                 # ∂/∂X_k of a function of α
        gradX = lambda g: sp.Matrix([DX(g, k) for k in range(3)])
        u = F*w                                                                 # J = 1
        Zw = lambda A: zh*zh.dot(A)
        dz2 = zgrad(zgrad(w))
        Dw = -curl(curl(w))/a**2 + (1/c**2 - 1/a**2)*dz2
        GX = sp.Matrix(3, 3, lambda i, k: DX(u[i], k))
        Gw = jac(w)
        rdw = Gw*al                                                             # (𝐫⋅∇)w = r ∂ᵣw
        checks = [
            F.T*u - w - ((a**2 - 1)*w + (c**2 - a**2)*Zw(w)),                    # inertia
            F.T*zh.cross(u) - zh.cross(w) - (1/c - 1)*zh.cross(w),                # Coriolis
            F.T*u.applyfunc(lambda g: sum(DX(DX(g, k), k) for k in range(3))) - w.applyfunc(lap)
            - ((a**2/c**2 - 1)*dz2 + (c**2 - a**2)*Zw(Dw)),                       # viscous force (∇²w = -∇×∇×w)
            F.T*(GX + GX.T)*Fi.T*al - (Qm*Gw*Qm.inv()*al + Gw.T*al),              # traction (times r)
            Qm*Gw*Qm.inv()*al - rdw - ((a**2/c**2 - 1)*z*zgrad(w) + (c**2 - a**2)*Zw(rdw/a**2 + (1/c**2 - 1/a**2)*z*zgrad(w))),
            sp.Matrix([sum(DX(DX(S, k), k) for k in range(3)) - lap(S) - ((1/a**2 - 1)*lap(S) + (1/c**2 - 1/a**2)*sp.diff(S, z, 2))]),
            sp.Matrix([(Fi.T*al).dot(gradX(S)) - al.dot(grad(S)) - ((1/a**2 - 1)*al.dot(grad(S)) + (1/c**2 - 1/a**2)*z*sp.diff(S, z))]),
            F.T*gradX(Pr) - grad(Pr),                                             # pressure gradient
            F.T*gradX((x**2 + y**2 + z**2)/2) - al,                               # buoyancy: Fᵀ∇_X(r²/2) = 𝐫
            sp.Matrix([u.dot(gradX(S)) - w.dot(grad(S))]),                        # advection
            sp.Matrix([sum(DX(u[k], k) for k in range(3)) - div(w)])]             # mass
        for ex in checks:
            for e in ex:
                e = sp.expand(e)
                worst = max(worst, max([abs(float(t)) for t in sp.Poly(e, *V).coeffs()] or [0.]))
    # Maclaurin: with πGρ = 1, Φ = A1 (X² + Y²) + A3 Z² - Ω² (X² + Y²)/2 is a function of r when (A1 - Ω²/2) a² = A3 c²
    mac = 0
    for f in (0.05, 0.3, 0.6):
        a, c = axes(f)
        e = np.sqrt(1 - (c/a)**2)
        A1 = np.sqrt(1 - e*e)*np.arcsin(e)/e**3 - (1 - e*e)/e**2
        A3 = 2/e**2 - 2*np.sqrt(1 - e*e)*np.arcsin(e)/e**3
        Om2 = 2*np.sqrt(1 - e*e)*(3 - 2*e*e)*np.arcsin(e)/e**3 - 6*(1 - e*e)/e**2
        mac = max(mac, abs((A1 - Om2/2)*a*a - A3*c*c)/(A3*c*c))
    # to_physical: the velocity F w at the point F α in the spherical basis of that point
    cf.par.centrifugal_shape, cf.par.centrifugal_f = 'spheroid', 0.3
    a, c = axes(0.3)
    rng = np.random.default_rng(5)
    r, th = rng.uniform(0.1, 1, 6), rng.uniform(0.05, 3.1, 6)
    wr, wt, wp = rng.normal(size=(3, 6))
    s, zz, ur, ut, up = cf.to_physical(r, th, wr, wt, wp)
    tp = 0
    for i in range(6):
        er = np.array([np.sin(th[i]), 0, np.cos(th[i])]); et = np.array([np.cos(th[i]), 0, -np.sin(th[i])])
        wv = wr[i]*er + wt[i]*et + wp[i]*np.array([0, 1, 0])
        Xp = np.diag([a, a, c]) @ (r[i]*er)
        uv = np.diag([a, a, c]) @ wv
        R_ = np.linalg.norm(Xp); n_r = Xp/R_
        n_t = np.array([n_r[2], 0, -n_r[0]])/np.hypot(n_r[0], n_r[2])
        tp = max(tp, abs(s[i] - Xp[0]), abs(zz[i] - Xp[2]), abs(ur[i] - uv @ n_r), abs(ut[i] - uv @ n_t), abs(up[i] - uv[1]))
    del cf.par.centrifugal_shape, cf.par.centrifugal_f
    verdict('formulas', worst == 0 and mac < 1e-12 and tp < 1e-14,
            'the terms of section 8 from the map, exact (oblate and prolate); Maclaurin level surfaces %.0e; to_physical %.0e'
            % (mac, tp))

# ======================================================================================================================
# couplings: the projections of the exact terms (derive_diff.py's Laurent polynomials; homogeneous fluid: λ1 = 0)
LP, CC, SS, IR, IS = dd.LP, dd.CC, dd.SS, dd.IR, dd.IS
ZH = (CC, -1*SS, LP())

def flow0(l, m, X):
    '''kore's flow of a homogeneous fluid: P, ∇×∇×(P Y 𝐫); T, ∇×(T Y 𝐫)'''
    if X == 'P':
        P = dd.RAD['P']
        return dd.qst(l, m, l*(l + 1)*P*IR, P.dr() + P*IR, LP())
    return dd.qst(l, m, LP(), LP(), dd.RAD['T'])

def zgrad(v, m):
    '''(ẑ⋅∇)v = Σ_k ẑ_k (ê_k⋅∇)v from the velocity gradient (ẑ = (cos θ, -sin θ, 0))'''
    G = dd.gradv(v, m)
    return tuple(CC*G[i][0] - SS*G[i][1] for i in range(3))

def Zv(v):
    vz = CC*v[0] - SS*v[1]
    return (CC*vz, -1*SS*vz, LP())

def exact_terms(kind, m, X, l, f):
    a, c = axes(f)
    a2, c2 = a*a, c*c
    if X == 'S':
        g = dd.grad(dd.RAD['S']*dd.Ylp(l, m), m)
        sz = CC*g[0] - SS*g[1]
        if kind == 'heat':
            gz = dd.grad(sz, m)
            return (1/a2 - 1)*dd.div(g, m) + (1/c2 - 1/a2)*(CC*gz[0] - SS*gz[1])
        return (1/a2 - 1)*g[0] + (1/c2 - 1/a2)*(CC*sz)                         # 'flux'
    w = flow0(l, m, X)
    if kind == 'inertia':
        return dd.vadd(dd.vsc(a2 - 1, w), dd.vsc(c2 - a2, Zv(w)))
    if kind == 'coriolis':
        return dd.vsc(1/c - 1, dd.cross(ZH, w))
    if kind == 'visc':
        z2 = zgrad(zgrad(w, m), m)
        Dw = dd.vadd(dd.vsc(1/a2, dd.tdiv(dd.sigma(w, m), m)), dd.vsc(1/c2 - 1/a2, z2))   # ∇²w = ∇⋅σ(w) (∇⋅w = 0)
        return dd.vadd(dd.vsc(a2/c2 - 1, z2), dd.vsc(c2 - a2, Zv(Dw)))
    G = dd.gradv(w, m)                                                          # 'traction'
    xz = dd.vsc(CC, zgrad(w, m))
    return dd.vadd(dd.vsc(a2/c2 - 1, xz), dd.vsc(c2 - a2, Zv(dd.vadd(dd.vsc(1/a2, tuple(G[i][0] for i in range(3))),
                                                                   dd.vsc(1/c2 - 1/a2, xz)))))

def coupling_column(task):
    kind, m, X, l, f = task
    t0 = time.time()
    E = exact_terms(kind, m, X, l, f)
    rows = {'h': E} if X == 'S' else {'u': dd.curl(dd.curl(E, m), m)[0], 'v': dd.curl(E, m)[0]}
    out = {}
    for row, ex in rows.items():
        for lr in range(max(0 if (row == 'h' and m == 0) else max(m, 1), l - 4), l + 5):
            out[(row, lr)] = dd.project_num(ex, lr, m)
    return task, out, time.time() - t0

def wall_column(task):
    R, m, X, l, f = task
    t0 = time.time()
    out = {}
    if X in 'PT':
        Tr = exact_terms('traction', m, X, l, f)
        vt, vp = Tr[1].ev([R], dd.XQ)[0], Tr[2].ev([R], dd.XQ)[0]
        for lr in range(max(max(m, 1), l - 4), l + 5):
            Y = dd.Ylp(lr, m)
            dY, mY = Y.dth().ev([1.], dd.XQ)[0], (1j*m*Y*IS).ev([1.], dd.XQ)[0]
            nrm = lr*(lr + 1)*4*np.pi/(2*lr + 1)
            integ = lambda A, B: 2*np.pi*np.sum(dd.WXQ*(vt*np.conj(A) + vp*np.conj(B)))/nrm
            out[('u', lr)] = R**2*integ(dY, mY)                                  # S_l = (0, ∂θY, i m Y/sinθ)
            out[('v', lr)] = R*integ(mY, -dY)                                    # T_l = (0, i m Y/sinθ, -∂θY)
    else:
        flux = exact_terms('flux', m, 'S', l, f)
        for lr in range(max(0 if m == 0 else m, l - 4), l + 5):
            out[('h', lr)] = dd.project_num(flux, lr, m, radii=np.array([R]))[0]
    return task, out, time.time() - t0

def run_task(t):
    return (t[0],) + (coupling_column if t[0] == 'B' else wall_column)(t[1])

LABEL = re.compile(r'^([uvh])(\d)(?:vsc0|kps0)?_D(\d)$')

def part_couplings():
    LW = 24
    cases = []
    for f in (0.3, -0.2):
        cases += [('inertia', 1, 'P', 3, f), ('inertia', 2, 'T', 4, f), ('coriolis', 1, 'T', 2, f), ('coriolis', 0, 'P', 3, f),
                  ('visc', 1, 'P', 3, f), ('visc', 1, 'T', 2, f), ('visc', 0, 'T', 3, f), ('visc', 2, 'P', 4, f),
                  ('visc', 3, 'T', 5, f), ('heat', 1, 'S', 3, f), ('heat', 0, 'S', 2, f), ('heat', 2, 'S', 5, f)]
    walls = []
    for f in (0.3, -0.2):
        walls += [(1.0, 1, 'P', 3, f), (1.0, 1, 'T', 2, f), (1.0, 0, 'T', 3, f), (0.35, 2, 'P', 4, f), (0.35, 3, 'T', 4, f),
                  (1.0, 0, 'S', 2, f), (0.35, 1, 'S', 3, f)]
    blocks = {(m, f): cf.spheroid_build(m, LW, f=f) for m in range(4) for f in (0.3, -0.2)}
    tasks = [('B', c) for c in cases] + [('W', w) for w in walls]
    try:
        import multiprocessing
        ncpu = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
        with multiprocessing.get_context('fork').Pool(max(1, min(ncpu, 16))) as pool:
            res = list(pool.imap_unordered(run_task, tasks))
    except ValueError:
        res = list(map(run_task, tasks))
    worst_b = worst_w = 0
    nb = 0
    for part, task, out, dt in res:
        scale = max(np.abs(v).max() for v in out.values())
        if part == 'B':
            kind, m, X, l, f = task
            B = blocks[(m, f)][kind]
            for (row, lr), der in out.items():
                can = np.zeros(len(dd.RADII), dtype=complex)
                for lab, M in B.get((row, X), {}).items():
                    cv = complex(M[lr, l])
                    if cv != 0:
                        rw, rx, d = LABEL.match(lab).groups()
                        can += cv*dd.RADII**(-int(rx))*dd.FUNS[X][int(d)](dd.RADII)
                err = np.abs(der - can).max()/scale
                worst_b = max(worst_b, err)
                nb += 1
                if err > 1e-11:
                    print('  FAIL  %s f=%g m=%d: row %s l=%d, column %s l=%d: derived %s, kore %s'
                          % (kind, f, m, row, lr, X, l, np.round(der, 8), np.round(can, 8)))
        else:
            R, m, X, l, f = task
            W = cf.spheroid_walls(m, LW, R, f)
            for (row, lr), der in out.items():
                can = sum(complex(M[lr, l])*float(dd.FUNS[X][int(src[1])](np.array([R]))[0])
                          for (rw, src), M in W.items() if rw == row and src[0] == X)
                err = abs(der - can)/scale
                worst_w = max(worst_w, err)
                nb += 1
                if err > 1e-11:
                    print('  FAIL  wall R=%g f=%g m=%d: row %s l=%d, column %s l=%d: derived %s, kore %s'
                          % (R, f, m, row, lr, X, l, der, can))
    verdict('couplings', worst_b < 1e-11 and worst_w < 1e-11,
            '%d blocks (inertia, Coriolis, viscous, heat; |Δl| <= 4) worst %.1e, the stress-free and flux wall rows worst %.1e'
            % (nb, worst_b, worst_w))

# ======================================================================================================================
# exact Galerkin solvers in the spheroid (no kore). A scalar of azimuthal order μ is w^μ g(ρ², z) (w = α1 + i α2, w̄^|μ| for
# μ < 0), (μ, g) with g = {(j, k): rational} for ρ^{2j} z^k; vectors in the components (A_+, A_-, A_z), A_± = A_x ± i A_y
# (orders m + 1, m - 1, m for a field of order m). With real g, the operators below keep the coefficients real.
def _g(*terms):
    o = {}
    for s, g in terms:
        for e, v in g.items():
            o[e] = o.get(e, 0) + s*v
    return {e: v for e, v in o.items() if v != 0}
_r2 = lambda g: {(j + 1, k): v for (j, k), v in g.items()}
_dp = lambda g: {(j - 1, k): v*j for (j, k), v in g.items() if j > 0}           # ∂/∂(ρ²)
def _gmul(g, h):
    o = {}
    for (j1, k1), v1 in g.items():
        for (j2, k2), v2 in h.items():
            o[(j1 + j2, k1 + k2)] = o.get((j1 + j2, k1 + k2), 0) + v1*v2
    return {e: v for e, v in o.items() if v != 0}
def _s(*terms):
    mu = {S[0] for _, S in terms if S[1]}
    assert len(mu) <= 1
    return (mu.pop() if mu else terms[0][1][0], _g(*[(s, S[1]) for s, S in terms]))
def d_w(S):
    mu, g = S
    return (mu - 1, _g((mu, g), (1, _r2(_dp(g))))) if mu >= 1 else (mu - 1, _dp(g))
def d_wb(S):
    mu, g = S
    return (mu + 1, _g((-mu, g), (1, _r2(_dp(g))))) if mu <= -1 else (mu + 1, _dp(g))
def d_z(S):
    return (S[0], {(j, k - 1): v*k for (j, k), v in S[1].items() if k > 0})
def rcurl(A):
    '''-i ∇×A'''
    Ap, Am, Az = A
    return (_s((-2, d_wb(Az)), (1, d_z(Ap))), _s((2, d_w(Az)), (-1, d_z(Am))), _s((1, d_wb(Am)), (-1, d_w(Ap))))
def times_alpha(S):
    mu, g = S
    return ((mu + 1, g) if mu >= 0 else (mu + 1, _r2(g)), (mu - 1, _r2(g)) if mu > 0 else (mu - 1, g),
            (mu, {(j, k + 1): v for (j, k), v in g.items()}))
FB = {(0, 0): Q(1), (1, 0): Q(-1), (0, 2): Q(-1)}                               # 1 - |α|²

def moment(p, q):
    '''∫ ρ^{2p} z^q dV / 2π over the unit ball'''
    if q % 2:
        return Q(0)
    B = Q(1)
    for i in range(1, p + 1):
        B *= i
    for i in range(p + 1):
        B /= Q(q + 1, 2) + i
    return B/(2*p + q + 3)

class Gal:
    '''bilinear forms ∫ F conj(H) dV of lists of scalars of one order, at 50 digits'''
    def __init__(self):
        import mpmath as mp
        self.mp = mp
        mp.mp.dps = 50
        self.mom = {}
    def num(self, v):
        return self.mp.mpf(v.numerator)/v.denominator
    def form(self, Fs, Hs):
        '''[p, q] = ∫ Fs[q] conj(Hs[p])'''
        mus = {S[0] for S in Fs + Hs if S[1]}
        n1, n2 = len(Hs), len(Fs)
        if not mus:
            return np.zeros((n1, n2), dtype=object)
        assert len(mus) == 1, mus
        mu = abs(mus.pop())
        tf, th = sorted({e for S in Fs for e in S[1]}), sorted({e for S in Hs for e in S[1]})
        def mat(Ss, ts):
            ix = {e: i for i, e in enumerate(ts)}
            M = np.empty((len(ts), len(Ss)), dtype=object); M[:] = self.mp.mpf(0)
            for i, S in enumerate(Ss):
                for e, v in S[1].items():
                    M[ix[e], i] = self.num(v)
            return M
        W = np.empty((len(th), len(tf)), dtype=object)
        for i, (j1, k1) in enumerate(th):
            for j, (j2, k2) in enumerate(tf):
                key = (mu + j1 + j2, k1 + k2)
                if key not in self.mom:
                    self.mom[key] = self.num(moment(*key))
                W[i, j] = self.mom[key]
        return mat(Hs, th).T.dot(W).dot(mat(Fs, tf))
    def eig(self, G, Op, hermitian=False):
        '''eigenvalues of Op x = λ G x (G symmetric positive), Cholesky at 50 digits, then double precision'''
        mp = self.mp
        L = mp.cholesky(mp.matrix(G.tolist()))
        Li = mp.inverse(L)
        A = Li*mp.matrix(Op.tolist())*Li.T
        n = G.shape[0]
        Af = np.array([[complex(A[i, j]) for j in range(n)] for i in range(n)])
        ev = np.linalg.eigvalsh((Af.real + Af.real.T)/2) if hermitian else np.linalg.eigvals(Af)
        return ev[np.argsort(-ev.real)]

def galerkin_flow(D, m, symm, kind, f, Ek):
    '''λ u = -2ẑ×u - ∇p + Ek ∇²u in the spheroid (a, a, c), u = F w, degree <= D: poloidal -i∇×(-i∇×(Fb^q p α)), toroidal
    -i∇×(Fb^q' t α), q = 2, q' = 1 (no-slip) or q = 1, q' = 0 (stress-free: tangent fields, the stress-free condition is
    natural in the weak form -(1/2)∫σ:σ*); p = w^m ρ^{2j} z^k even in z for symm 1 (t odd), and the reverse for symm -1'''
    gal = Gal(); mp = gal.mp
    qp, qt = (2, 1) if kind == 'noslip' else (1, 0)
    basis = []
    for X, q, zpar in (('P', qp, 0 if symm == 1 else 1), ('T', qt, 1 if symm == 1 else 0)):
        Fq = {(0, 0): Q(1)}
        for _ in range(q):
            Fq = _gmul(Fq, FB)
        top = D - (2*q + m + 1 - (2 if X == 'P' else 1))
        for k in range(zpar, top + 1, 2):
            for j in range((top - k)//2 + 1):
                if m == 0 and j == 0 and k % 2 == 0:     # radial potentials give no field (m = 0): left out
                    continue
                A = times_alpha((m, _gmul(Fq, {(j, k): Q(1)})))
                basis.append(rcurl(rcurl(A)) if X == 'P' else rcurl(A))
    fm = mp.mpf(f.numerator)/f.denominator
    a, c = (1 - fm)**(-mp.mpf(1)/3), (1 - fm)**(mp.mpf(2)/3)
    sig, tau = (a, a, c), (2/a, 2/a, 1/c)                       # u_i = sig_i w_i; D_+ = (2/a)∂w̄, D_- = (2/a)∂w, D_z = ∂z/c
    nw = (mp.mpf(1)/2, mp.mpf(1)/2, mp.mpf(1))                  # u⋅v* = (u_+v_+* + u_-v_-*)/2 + u_z v_z*
    dop = (d_wb, d_w, d_z)
    comp = [[b[i] for b in basis] for i in range(3)]
    Bpp, Bmm, Bzz = gal.form(comp[0], comp[0]), gal.form(comp[1], comp[1]), gal.form(comp[2], comp[2])
    G = a*a*(Bpp + Bmm)/2 + c*c*Bzz
    S = a*a*(Bpp - Bmm)                                          # -2∫(ẑ×u)⋅v* = -i S: (ẑ×u)_± = ±i u_±
    der = {(i, j): [dop[j](s) for s in comp[i]] for i in range(3) for j in range(3)}
    Vv = 0*G
    for i in range(3):
        for j in range(3):
            Vv = Vv - nw[i]*nw[j]*sig[i]**2*tau[j]**2*gal.form(der[(i, j)], der[(i, j)])
            if kind == 'stressfree':
                Vv = Vv - nw[i]*nw[j]*sig[i]*tau[j]*sig[j]*tau[i]*gal.form(der[(i, j)], der[(j, i)])
    Ekm = mp.mpf(Ek.numerator)/Ek.denominator
    return gal.eig(G, S*mp.mpc(0, -1) + Vv*Ekm)

def galerkin_heat(D, m, symm, kind, f):
    '''λ s = ∇²s in the spheroid, s = 0 (kind 'temperature': s = Fb w^m ρ^{2j} z^k) or ∂n s = 0 ('flux': natural)'''
    gal = Gal(); mp = gal.mp
    top = D - m - (2 if kind == 'temperature' else 0)
    basis = [(m, _gmul(FB, {(j, k): Q(1)}) if kind == 'temperature' else {(j, k): Q(1)})
             for k in range(0 if symm == 1 else 1, top + 1, 2) for j in range((top - k)//2 + 1)]
    fm = mp.mpf(f.numerator)/f.denominator
    a, c = (1 - fm)**(-mp.mpf(1)/3), (1 - fm)**(mp.mpf(2)/3)
    G = gal.form(basis, basis)
    K = 0*G
    for dop, t, nw in ((d_wb, 2/a, mp.mpf(1)/2), (d_w, 2/a, mp.mpf(1)/2), (d_z, 1/c, mp.mpf(1))):
        ds = [dop(s) for s in basis]
        K = K - nw*t**2*gal.form(ds, ds)
    return gal.eig(G, K, hermitian=True)

# ======================================================================================================================
SPH = "par.centrifugal_shape = 'spheroid'; par.centrifugal_f = %r; "

VISC = ((1, -1, Q(3, 10), 'noslip'), (1, -1, Q(3, 10), 'stressfree'), (2, 1, Q(-1, 5), 'noslip'), (0, 1, Q(3, 10), 'stressfree'),
        (3, -1, Q(-1, 5), 'stressfree'))
HEAT = ((1, -1, Q(3, 10), 'temperature'), (1, 1, Q(3, 10), 'flux'), (2, 1, Q(-1, 5), 'temperature'), (0, -1, Q(3, 10), 'flux'))
DV, DH = 22, 24        # the degrees of the Galerkin fields (at 18 and 20 the fourth modes are still 1e-8 off)

def galerkin_job(job):
    kind, (m, sy, f, wall) = job
    return galerkin_flow(DV, m, sy, wall, f, Q(1, 10))[:4] if kind == 'v' else galerkin_heat(DH, m, sy, wall, f)[:4]

def part_modes():
    from concurrent.futures import ThreadPoolExecutor
    ncpu = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    runs = {}
    for f in (0.3, -0.2):
        for m in range(4):
            for sy in (1, -1):
                runs['sc_i_%g_%d_%+d' % (f, m, sy)] = SPH % f + 'par.m = %d; par.symm = %d; par.lmax = %d' % (m, sy, 21 - m % 2)
        runs['sc_r_%g' % f] = SPH % f + ('par.ViscosD = 1e-3; par.bco = 0; par.bci = 0; par.ricb = 0.35; par.m = 0; par.symm = 1; '
                                         'par.N = 32; par.lmax = 21')
    for (m, sy, f, wall) in VISC:
        runs['sc_v_%d_%+d_%g_%s' % (m, sy, float(f), wall)] = SPH % float(f) + (
            'par.m = %d; par.symm = %d; par.ViscosD = 0.1; par.bco = %d; par.N = 40; par.lmax = %d; par.rpower_v = 4'
            % (m, sy, 1 if wall == 'noslip' else 0, 31 - m % 2))
    for (m, sy, f, wall) in HEAT:
        runs['sc_h_%d_%+d_%g_%s' % (m, sy, float(f), wall)] = SPH % float(f) + (
            'par.m = %d; par.symm = %d; par.ViscosD = 0.1; par.N = 40; par.lmax = %d; par.rpower_v = 4; par.thermal = 1; '
            'par.ThermaD = 1.0; par.Beyonce = 0.0; par.stratification = "linear"; par.bco_thermal = %d'
            % (m, sy, 31 - m % 2, 0 if wall == 'temperature' else 1))
    with ThreadPoolExecutor(max(1, min(ncpu, 8))) as ex:            # mk_run.py runs (subprocesses)
        list(ex.map(lambda kv: sm.build(*kv), runs.items()))
    jobs = [('v', c) for c in VISC] + [('h', c) for c in HEAT]
    try:
        import multiprocessing
        with multiprocessing.get_context('fork').Pool(max(1, min(ncpu, len(jobs)))) as pool:
            refs = pool.map(galerkin_job, jobs)
    except ValueError:
        refs = list(map(galerkin_job, jobs))
    # inviscid: the exact polynomial modes (Space)
    worst, n, so = 0, 0, 0
    sp_ = sm.Space(7)
    for f in (0.3, -0.2):
        a, c = axes(f)
        modes = sp_.modes(a, c)
        so = max(so, abs(sm.find(modes, 1, -1, 2/(1 + (1 - f)**2))[2] - 2j/(1 + (1 - f)**2)))
        for m in range(4):
            for sy in (1, -1):
                for (mm, ss, lam, _) in modes:
                    if (mm, ss) == (m, sy) and abs(lam.imag) > 1e-8:
                        worst = max(worst, abs(sm.kore_eigs('sc_i_%g_%d_%+d' % (f, m, sy), lam + 1e-7j) - lam))
                        n += 1
    verdict('inviscid', worst < 1e-10 and so < 1e-13 and n > 150,
            '%d modes of degree <= 7 (m 0..3, both symmetries, f = 0.3 and -0.2): max |λ (kore) - exact| = %.1e; '
            'spin-over against 2/(1 + (1 - f)²) %.0e' % (n, worst, so))
    # viscous (Ek = 0.1) and heat: the four least-damped modes of each case against the Galerkin solver
    worst = {'v': 0, 'h': 0}
    for (kind, (m, sy, f, wall)), ref in zip(jobs, refs):
        name = 'sc_%s_%d_%+d_%g_%s' % (kind, m, sy, float(f), wall)
        for lam in ref:
            worst[kind] = max(worst[kind], abs(sm.kore_eigs(name, lam + 1e-6) - lam)/max(1, abs(lam)))
    verdict('viscous', worst['v'] < 1e-10, '%d modes at Ek 0.1 (no-slip and stress-free, m 0..3, f = 0.3 and -0.2) against the '
            'exact Galerkin solver: max |λ (kore) - Galerkin|/max(1, |λ|) = %.1e' % (4*len(VISC), worst['v']))
    verdict('heat', worst['h'] < 1e-10, '%d diffusion modes (fixed temperature and fixed flux) against the exact Galerkin '
            'solver: max |λ (kore) - Galerkin|/max(1, |λ|) = %.1e' % (4*len(HEAT), worst['h']))
    # rigid rotation in a shell with stress-free walls
    worst = max(abs(sm.kore_eigs('sc_r_%g' % f, 1e-4 + 1e-4j)) for f in (0.3, -0.2))
    verdict('rotation', worst < 1e-11, 'stress-free shell (ricb 0.35, f = 0.3 and -0.2): the rigid rotation, |λ| = %.0e' % worst)

if __name__ == '__main__':
    part_formulas()
    part_couplings()
    part_modes()
    print('spheroid_check.py: %s (%.0f s)' % ('FAIL ' + ', '.join(fails) if fails else 'ok', time.time() - T0))
    sys.exit(1 if fails else 0)
