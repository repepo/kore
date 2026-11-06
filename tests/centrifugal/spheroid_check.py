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
             against the velocity F w in the spherical basis of the physical point. NEW (2026-11-06, step 2) the field,
             b = F b̃ and B0 = F B̃0 (fluxes, J = 1): the current F⁻¹∇_X×b = ∇×(Q b̃) and F⁻¹∇_X×E = ∇×(Fᵀ E), the
             induction Fᵀ(u×B0) = w×B̃0, the covariant Lorentz force Fᵀ[(∇_X×b)×B0 + (∇_X×B0)×b] and its part beyond the
             sphere's, the diffusion's field Fᵀ∇_X×b = Q∇×(Q b̃) and its expansion in Z.
  couplings  kore's couplings (centrifugal.spheroid_build: the inertia, Coriolis, viscous and heat terms in the rows u, v,
             h) and wall rows (centrifugal.spheroid_walls: stress-free and fixed flux) against the projections of those
             formulas on kore's rows, with explicit radial functions (derive_diff.py's Laurent polynomials in r, cos θ,
             sin θ). Here (ẑ⋅∇)v is taken from the velocity gradient in spherical components and ∇²w = ∇⋅σ(w), not with
             the identities that centrifugal.py uses. m = 0..3, f = 0.3 and -0.2, the walls at r = 1 and 0.35.
  field      (NEW 2026-11-06, step 2) the same for the field: the diffusion in the rows f, g (spheroid_build 'mag') and
             the Lorentz force in u, v (spheroid_magnetic_build) through poloidal and toroidal backgrounds of degree 1 to 3
             with explicit profiles, and the ICB row of a conducting inner core (spheroid_icb), against Q∇×(Q b̃) and the
             force written with Q as a matrix in spherical components (not with Z).
  walls      (NEW 2026-11-06, step 2) the insulating walls: λ_l (spheroid_lambda) against mpmath's legenq and legenp in
             the spheroidal coordinates of the wall; potential fields of the spheroid, their flux and tangential field in
             the reference space from the Jacobian of those coordinates, pointwise against the formulas of section 8; then
             kore's wall rows (spheroid_field_walls) on the l-components of these fields (and the sphere's rows, which
             must not close).
  decay      (NEW 2026-11-06, step 2) the free decay of the axisymmetric field in a full spheroid with an insulating
             exterior (Em 1e-2, Le 1e-8 so that no flow mode comes near), against an exact Galerkin solver: the toroidal
             field as a scalar of order 1 that vanishes at the wall, the poloidal ψ = A_φ of order 1 with ∇²ψ = λψ and the
             exterior through the natural boundary term Σ λ_l ⟨ψ|P_l^1⟩⟨v|P_l^1⟩/n_l (λ_l from the closed form of Q_l^1).
  cores      (NEW 2026-11-06, step 2) a conducting inner core of the fluid's conductivity (mu_i2o 1) gives the decay of
             the full spheroid (m = 0 against the Galerkin solver, m = 1 and 2 against kore's full spheroid); an insulating
             inner core is the limit σ_i -> 0 of a conducting one (σ_i = 1e-3 and 1e-5). Field modes are told from flow
             modes by their eigenvectors.
  first order (NEW 2026-11-06, step 2) the exact spheroid and 'uniform' part as f²: magnetic modes of a viscous shell
             with a conducting inner core (every term of the field), f = 0.01 and 0.005.
  inviscid   kore's inertial modes against the exact ones (spheroid_modes.Space, the polynomial method, degree <= 7),
             m = 0..3, both symmetries, f = 0.3 and -0.2; the spin-over 2/(1 + (1 - f)²)
  viscous    the four least-damped modes at Ek = 0.1 with no-slip and with stress-free walls, against an exact Galerkin
             solver: solenoidal polynomial fields of one m and one symmetry (poloidal and toroidal potentials times powers
             of 1 - |α|², in the components u_x ± i u_y and u_z), integrals as exact rationals (moments of the ball),
             Cholesky at 50 digits; the stress-free rows enter only through the weak form ∫ σ:σ (a natural condition)
  heat       the first four diffusion modes, fixed temperature and fixed flux, against the same Galerkin for scalars
  rotation   stress-free walls of a shell: the rigid rotation about the axis has the eigenvalue 0

Usage: python3 spheroid_check.py   (about three minutes on four cores; builds its runs with mk_run.py in the current
directory)
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
    # NEW (2026-11-06, step 2) the field: b̃, B̃0 (solenoidal) and an electric field, polynomials of α
    bt = curl(sp.Matrix([x*y**2 - Q(2, 3)*z**3 + x*z, Q(1, 4)*x**2*z**2 + y**3, Q(5, 6)*x*y*z - y**2*x**2 + z]))
    B0t = curl(sp.Matrix([Q(3, 5)*y*z**2 - x**3, x*z + Q(2, 7)*y**2*z, Q(1, 3)*x**2*y + z**3*x]))
    Ef = sp.Matrix([x*z**2 - Q(1, 2)*y, y**2*z + Q(3, 4)*x**3, x*y - Q(2, 5)*z**3])
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
        # NEW (2026-11-06, step 2) the field of the body b = F b̃, B0 = F B̃0 (J = 1): the current, Faraday's law, the
        # covariant Lorentz force and its part beyond the sphere's, the electric field of the diffusion and the induction
        curlX = lambda A: sp.Matrix([DX(A[2], 1) - DX(A[1], 2), DX(A[0], 2) - DX(A[2], 0), DX(A[1], 0) - DX(A[0], 1)])
        b, B0 = F*bt, F*B0t
        jt, J0t = curl(Qm*bt), curl(Qm*B0t)                                     # j̃ = ∇×(Q b̃)
        dl = c**2 - a**2
        cz = lambda A: curl(Zw(A))
        checks += [
            Fi*curlX(b) - jt,                                                     # F⁻¹ ∇_X×b = ∇×(Q b̃)
            Fi*curlX(Ef) - curl(F.T*Ef),                                          # ∂ₜb̃ = F⁻¹∂ₜb = -∇×(Fᵀ𝐄)
            F.T*(curlX(b).cross(B0) + curlX(B0).cross(b)) - (jt.cross(B0t) + J0t.cross(bt)),   # covariant Lorentz force
            jt.cross(B0t) + J0t.cross(bt) - (curl(bt).cross(B0t) + curl(B0t).cross(bt))
            - ((a**2 - 1)*(curl(B0t).cross(bt) - B0t.cross(curl(bt))) + dl*(cz(B0t).cross(bt) - B0t.cross(cz(bt)))),
            F.T*curlX(b) - Qm*jt,                                                 # Fᵀ(η∇_X×b) = η Q ∇×(Q b̃)
            Qm*jt - curl(bt) - ((a**4 - 1)*curl(bt) + a**2*dl*(Zw(curl(bt)) + cz(bt)) + dl**2*Zw(cz(bt))),
            F.T*u.cross(B0) - w.cross(B0t)]                                       # Fᵀ(u × B0) = w × B̃0
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

# NEW (2026-11-06, step 2) the field: Q as the matrix diag(a², a², c²) turned to the basis r̂, θ̂, φ̂ (not with Z), the
# background B̃0 of degree n as mag_algebra's (without cnorm), its profiles h, k explicit polynomials
BEXPL = {'h': dd.r/2 + dd.r**3/5 - dd.r**5/10, 'k': dd.Q(3, 10)*dd.r + dd.r**3/4}
BRAD = {k: dd.radial(v) for k, v in BEXPL.items()}
BFUNS = {k: [dd.lam(dd.sp.diff(v, dd.r, j)) for j in range(7)] for k, v in BEXPL.items()}

def qmat(v, f):
    '''Q v in spherical components: Q_rr = a² sin²θ + c² cos²θ, Q_rθ = (a² - c²) sinθ cosθ, Q_θθ = a² cos²θ + c² sin²θ,
    Q_φφ = a²'''
    a, c = axes(f)
    a2, c2 = a*a, c*c
    vr, vt, vp = v
    return ((a2*SS*SS + c2*CC*CC)*vr + (a2 - c2)*SS*CC*vt, (a2 - c2)*SS*CC*vr + (a2*CC*CC + c2*SS*SS)*vt, a2*vp)

def background(B, n):
    '''B̃0 of degree n: 'h', ∇×∇×(h Y_n 𝐫); 'k', ∇×(k Y_n 𝐫)'''
    if B == 'h':
        H = BRAD['h']
        return dd.qst(n, 0, n*(n + 1)*H*IR, H.dr() + H*IR, LP())
    return dd.qst(n, 0, LP(), LP(), BRAD['k'])

def diffusion_field(b, m, f):
    '''the electric field of the diffusion per unit η beyond the sphere's, Q ∇×(Q b̃) - ∇×b̃'''
    return dd.vadd(qmat(dd.curl(qmat(b, f), m), f), dd.vsc(-1, dd.curl(b, m)))

def field_column(task):
    '''the rows of the exact field terms for the column X (F, G) of degree l: kind 'mag', the rows f = r̂⋅∇×E', g = r̂⋅∇×∇×E'
    of the diffusion's term E' = -[Q ∇×(Q b̃) - ∇×b̃]; kind 'lorentz', the rows u = r̂⋅∇×∇×G, v = r̂⋅∇×G of the force beyond
    the sphere's, G = ∇×(Q B̃0)×b̃ - B̃0×∇×(Q b̃) - [(∇×B̃0)×b̃ - B̃0×(∇×b̃)], B̃0 of the part B and degree n'''
    kind, m, X, l, f, B, n = task
    t0 = time.time()
    b = dd.field(l, m, X)
    if kind == 'mag':
        c1 = dd.curl(dd.vsc(-1, diffusion_field(b, m, f)), m)
        rows, dl = {'f': c1[0], 'g': dd.curl(c1, m)[0]}, 4
    else:
        B0 = background(B, n)
        G = dd.vadd(dd.cross(dd.curl(qmat(B0, f), 0), b), dd.vsc(-1, dd.cross(B0, dd.curl(qmat(b, f), m))),
                    dd.vsc(-1, dd.cross(dd.curl(B0, 0), b)), dd.cross(B0, dd.curl(b, m)))
        c1 = dd.curl(G, m)
        rows, dl = {'u': dd.curl(c1, m)[0], 'v': c1[0]}, n + 3
    out = {}
    for row, ex in rows.items():
        for lr in range(max(max(m, 1), l - dl), l + dl + 1):
            out[(row, lr)] = dd.project_num(ex, lr, m)
    return task, out, time.time() - t0

def icb_column(task):
    '''R times the S part of Q ∇×(Q b̃) - ∇×b̃ at r = R (the ICB row of a conducting inner core, per unit η)'''
    R, m, X, l, f = task
    t0 = time.time()
    E = diffusion_field(dd.field(l, m, X), m, f)
    vt, vp = E[1].ev([R], dd.XQ)[0], E[2].ev([R], dd.XQ)[0]
    out = {}
    for lr in range(max(max(m, 1), l - 4), l + 5):
        Y = dd.Ylp(lr, m)
        dY, mY = Y.dth().ev([1.], dd.XQ)[0], (1j*m*Y*IS).ev([1.], dd.XQ)[0]
        nrm = lr*(lr + 1)*4*np.pi/(2*lr + 1)
        out[('E', lr)] = R*2*np.pi*np.sum(dd.WXQ*(vt*np.conj(dY) + vp*np.conj(mY)))/nrm   # S_l = (0, ∂θY, i m Y/sinθ)
    return task, out, time.time() - t0

def run_task(t):
    return (t[0],) + {'B': coupling_column, 'W': wall_column, 'F': field_column, 'I': icb_column}[t[0]](t[1])

LABEL = re.compile(r'^([uvhfg])(\d)(?:vsc0|kps0|eta0)?_D(\d)$')
LABEL_MA = re.compile(r'^([uv])(\d)(hp|kt)(\w)(\d)ohr1_D(\d)$')   # NEW (2026-11-06, step 2): mag_algebra's (B̃0's profile)

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
    # NEW (2026-11-06, step 2) the field: the diffusion (rows f, g), the Lorentz force (u, v) and the ICB row
    fields = [('mag', m, X, l, f, None, None) for f in (0.3, -0.2)
              for (m, X, l) in ((1, 'F', 3), (1, 'G', 2), (0, 'F', 3), (0, 'G', 2), (2, 'F', 4), (3, 'G', 5))]
    fields += [('lorentz', m, X, l, 0.3, B, n) for (B, n, m, X, l) in (('h', 1, 1, 'F', 3), ('h', 1, 0, 'G', 2),
               ('h', 2, 2, 'F', 4), ('h', 3, 1, 'G', 4), ('k', 1, 1, 'F', 3), ('k', 2, 0, 'G', 4), ('k', 2, 2, 'F', 3))]
    fields += [('lorentz', m, X, l, -0.2, B, n) for (B, n, m, X, l) in (('h', 1, 1, 'G', 3), ('k', 1, 2, 'G', 4),
                                                                         ('h', 2, 0, 'F', 3))]
    icbs = [(0.35, 1, 'F', 3, 0.3), (0.35, 0, 'G', 2, 0.3), (0.35, 2, 'F', 4, -0.2), (0.35, 1, 'G', 3, -0.2)]
    blocks = {(m, f): cf.spheroid_build(m, LW, f=f) for m in range(4) for f in (0.3, -0.2)}
    mblocks = {(B, n, m, f): cf.spheroid_magnetic_build(m, {B: [n]}, LW, f) for (kind, m, X, l, f, B, n) in fields
               if kind == 'lorentz'}
    tasks = [('B', c) for c in cases] + [('W', w) for w in walls] + [('F', c) for c in fields] + [('I', c) for c in icbs]
    try:
        import multiprocessing
        ncpu = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
        with multiprocessing.get_context('fork').Pool(max(1, min(ncpu, 16))) as pool:
            res = list(pool.imap_unordered(run_task, tasks))
    except ValueError:
        res = list(map(run_task, tasks))
    worst_b = worst_w = worst_f = worst_i = 0
    nb = nf = 0
    for part, task, out, dt in res:
        scale = max(np.abs(v).max() for v in out.values())
        if part == 'F':   # NEW (2026-11-06, step 2): kore's sign for f, g (the A sign) and for u, v (mag_algebra's): minus
            kind, m, X, l, f, B, n = task
            for (row, lr), der in out.items():
                can = np.zeros(len(dd.RADII), dtype=complex)
                if kind == 'mag':
                    for lab, M in blocks[(m, f)]['mag'].get((row, X), {}).items():
                        cv = complex(M[lr, l])
                        if cv != 0:
                            rw, rx, d = LABEL.match(lab).groups()
                            can -= cv*dd.RADII**(-1 - int(rx))*dd.FUNS[X][int(d)](dd.RADII)
                else:
                    for lab, M in mblocks[(B, n, m, f)].get((row, X), {}).items():
                        cv = complex(M[lr, l])
                        if cv != 0:
                            rw, rx, code, nd, j, d = LABEL_MA.match(lab).groups()
                            assert code[0] == B and int(nd, 36) == n, lab
                            can -= cv*dd.RADII**(-int(rx))*BFUNS[B][int(j)](dd.RADII)*dd.FUNS[X][int(d)](dd.RADII)
                diff = np.abs(der - can).max()
                err = diff/scale if scale > 0 else diff     # (a row that vanishes: m = 0 and a toroidal B̃0)
                worst_f = max(worst_f, err)
                nf += 1
                if err > 1e-11:
                    print('  FAIL  %s f=%g m=%d B0 %s%s: row %s l=%d, column %s l=%d: derived %s, kore %s'
                          % (kind, f, m, B, n, row, lr, X, l, np.round(der, 8), np.round(can, 8)))
            continue
        if part == 'I':
            R, m, X, l, f = task
            W = cf.spheroid_icb(m, LW, R, f)
            for (row, lr), der in out.items():
                can = sum(complex(M[lr, l])*float(dd.FUNS[X][d](np.array([R]))[0]) for (Xn, d), M in W.items() if Xn == X)
                err = abs(der - can)/scale
                worst_i = max(worst_i, err)
                nf += 1
                if err > 1e-11:
                    print('  FAIL  ICB R=%g f=%g m=%d: l=%d, column %s l=%d: derived %s, kore %s' % (R, f, m, lr, X, l, der, can))
            continue
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
    verdict('field', worst_f < 1e-11 and worst_i < 1e-11 and nf > 300,
            '%d blocks of the field (the diffusion in f, g, |Δl| <= 4; the Lorentz force in u, v with poloidal and toroidal B̃0 '
            'of degree 1 to 3, |Δl| <= n + 3) worst %.1e, the ICB row of a conducting inner core worst %.1e' % (nf, worst_f, worst_i))

# ======================================================================================================================
# walls (NEW 2026-11-06, step 2): the insulating wall rows against exact potential fields of the spheroid
def radial_mp(wall, l, m, f):
    '''R_l(ν0), dR_l/dν(ν0) by mpmath (legenq outside the wall, legenp inside an inner core; of iν oblate, ν prolate) and
    ν0, k, a, c, oblate'''
    import mpmath as mp
    mp.mp.dps = 30
    fm = mp.mpf(f)
    a, c = (1 - fm)**(-mp.mpf(1)/3), (1 - fm)**(mp.mpf(2)/3)
    k = mp.sqrt(abs(a*a - c*c))
    ob = a > c
    fn = mp.legenq if wall == 'outer' else mp.legenp
    R = (lambda nu: fn(l, m, 1j*nu, type=3)) if ob else (lambda nu: fn(l, m, nu, type=3))
    nu0 = c/k
    return complex(R(nu0)), complex(mp.diff(R, nu0)), float(nu0), float(k), float(a), float(c), ob

def ferrers(l, m, x):
    '''P_l^m(x) (Condon-Shortley phase, as mpmath's legenp of type 2) and its x-derivative'''
    Pl = np.polynomial.Legendre.basis(l).convert(kind=np.polynomial.Polynomial)
    pm = Pl.deriv(m) if m else Pl
    w = 1 - x*x
    P = (-1)**m*w**(m/2)*pm(x)
    dP = (-1)**m*(-m*x*w**(m/2 - 1)*pm(x) + w**(m/2)*pm.deriv()(x)) if m else Pl.deriv()(x)
    return P, dP

def wall_lambda_task(task):
    '''the largest relative difference of centrifugal.spheroid_lambda from (a²/k) R_l'/R_l of mpmath, l <= 12'''
    wall, m, f = task
    lam = cf.spheroid_lambda(wall, m, 12, f)
    worst = 0
    for l in range(max(m, 1), 13):
        R0, R1, nu0, k, a, c, _ = radial_mp(wall, l, m, f)
        ref = (a*a/k)*R1/R0
        worst = max(worst, abs(ref - lam[l])/abs(ref), abs(ref.imag)/abs(ref))
    return task, worst

def wall_field_task(task):
    '''
    The potential field φ = R_l(ν) P_l^m(x) e^{imφ} of the spheroidal coordinates of the wall r = R (ν0 = c/k): its gradient
    from the Jacobian of (ν, x, φ) -> X, at the wall points α = R (sinθ, 0, cosθ), x = cos θ, then b̃ = F⁻¹∇φ in spherical
    components of the reference space; checks (1) the tangential field Q b̃ = Fᵀ∇φ is the gradient of its trace Φ = R_l(ν0)
    P_l^m(x) on the wall (the spheroidal angle is the reference colatitude) and b̃_r R = λ_l Φ (the flux), pointwise; (2) the
    rows of centrifugal.spheroid_field_walls applied to the l-components of b̃ there (F, R F', G from q, s, t) vanish
    '''
    import angular
    wall, l0, m, f, R = task
    R0, R1, nu0, k, a, c, ob = radial_mp(wall, l0, m, f)
    nx = 48
    xs, wx = np.polynomial.legendre.leggauss(nx)
    P, dP = ferrers(l0, m, xs)
    st = np.sqrt(1 - xs*xs)
    kR = k*R
    g = np.sqrt(1 + nu0**2) if ob else np.sqrt(nu0**2 - 1)
    S, dSn, dSx = kR*g*st, kR*nu0/g*st, -kR*g*xs/st                  # s = k R g(ν) sinθ: its ν and x derivatives
    dZn, dZx = kR*xs, kR*nu0*np.ones_like(xs)                         # z = k R ν x
    V = []
    for i in range(nx):
        Jm = np.array([[dSn[i], dSx[i], 0.], [0., 0., S[i]], [dZn[i], dZx[i], 0.]])   # columns ∂X/∂ν, ∂X/∂x, ∂X/∂φ (at φ = 0)
        gq = np.array([R1*P[i], R0*dP[i], 1j*m*R0*P[i]])                              # ∂φ/∂(ν, x, φ)
        gX = np.linalg.solve(Jm.T, gq)                                                # ∇_X φ = J⁻ᵀ g
        bt, Qb = gX/np.array([a, a, c]), gX*np.array([a, a, c])                       # b̃ = F⁻¹∇φ, Q b̃ = Fᵀ∇φ
        x, s_ = xs[i], st[i]
        V.append((bt[0]*s_ + bt[2]*x, bt[0]*x - bt[2]*s_, bt[1], R0*P[i], -s_*R0*dP[i], Qb[0]*x - Qb[2]*s_, Qb[1]))
    br, bth, bph, Phi, dPhi, Qth, Qph = np.array(V).T
    lam0 = ((a*a/k)*R1/R0).real
    sc_ = np.max(np.abs(br)) + np.max(np.abs(Qth))
    ept = max(np.max(np.abs(Qth - dPhi/R))/sc_, np.max(np.abs(Qph - 1j*m*Phi/(R*st)))/sc_,
              np.max(np.abs(br*R - lam0*Phi))/np.max(np.abs(br*R)))
    lw = 24
    Y, dY, Z = angular.ylm_grid(m, lw, xs)
    lv = np.arange(lw + 1); nl = 4*np.pi/(2*lv + 1); L = lv*(lv + 1.)
    Ls = np.where(L > 0, L, 1.)
    q = 2*np.pi*(Y*wx) @ br/nl
    s = np.where(L > 0, 2*np.pi*((dY*wx) @ bth + (np.conj(Z)*wx) @ bph)/(Ls*nl), 0)
    t = np.where(L > 0, 2*np.pi*((np.conj(Z)*wx) @ bth - (dY*wx) @ bph)/(Ls*nl), 0)
    F = np.where(L > 0, R*q/Ls, 0)                                    # q = L F/R, s = F' + F/R, t = G
    F1, G = s - F/R, t
    W = cf.spheroid_field_walls(wall, m, lw, R, f)
    rf = ((lv + 1)*F + R*F1) if wall == 'outer' else (lv*F - R*F1)    # the sphere's rows (assemble.bc_b_spherical)
    rf = rf + W[('f', 'F0')] @ F + W[('f', 'F1')] @ F1 + W[('f', 'G0')] @ G
    rg = G + W[('g', 'F0')] @ F + W[('g', 'F1')] @ F1 + W[('g', 'G0')] @ G
    scale = max(np.max(np.abs(F)), np.max(np.abs(R*F1)), np.max(np.abs(G)))
    lo = max(m, 1)
    erow = max(np.max(np.abs(rf[lo:lw - 4])), np.max(np.abs(rg[lo:lw - 4])))/scale
    esph = np.max(np.abs(((lv + 1)*F + R*F1)[lo:lw - 4] if wall == 'outer' else (lv*F - R*F1)[lo:lw - 4]))/scale
    return task, (ept, erow, esph)

def part_walls():
    lt = [(wall, m, f) for f in (0.3, -0.2, 0.9, -3.0, 1e-3) for m in (0, 2) for wall in ('outer', 'inner')]
    ft = [(wall, l0, m, f, R) for (f, R) in ((0.3, 1.0), (-0.2, 1.0), (0.3, 0.35), (0.6, 1.0), (-1.5, 0.5))
          for m in (0, 1, 2) for wall in ('outer', 'inner') for l0 in (max(m, 1), max(m, 1) + 1, max(m, 1) + 4)]
    try:
        import multiprocessing
        ncpu = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
        with multiprocessing.get_context('fork').Pool(max(1, min(ncpu, 16))) as pool:
            rl = pool.map(wall_lambda_task, lt)
            rf = pool.map(wall_field_task, ft)
    except ValueError:
        rl, rf = list(map(wall_lambda_task, lt)), list(map(wall_field_task, ft))
    wl = max(v for _, v in rl)
    wp, wr, ws = (max(v[i] for _, v in rf) for i in range(3))
    sph = sorted(v[2] for _, v in rf)[len(rf)//2]
    verdict('walls', wl < 1e-13 and wp < 1e-13 and wr < 1e-12 and sph > 1e-3,
            'λ_l (insulating walls) against mpmath %.0e (f = 1e-3 to -3, m 0 and 2, l <= 12); the potential fields of the '
            'spheroid: their flux and tangential field pointwise %.0e, and kore\'s wall rows on them %.0e (the sphere\'s rows '
            'alone leave %.0e, the median), outer wall and inner core, m 0..2, f 0.3, -0.2, 0.6, -1.5' % (wl, wp, wr, sph))

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

# NEW (2026-11-06, step 2) the free decay of an axisymmetric field in the spheroid with an insulating exterior. The poloidal
# field is ∇×(A φ̂) and ψ = A e^{iφ} a scalar of order 1 with λψ = ∇²ψ inside (η = 1), continuous with its normal derivative
# to the potential field ψ_e outside (∇²ψ_e = 0, decaying): the weak form λ ∫ψ v* = -∫∇ψ⋅∇v* + ∮ v* ∂ₙψ_e dS, natural at
# the wall, with ∮ v* ∂ₙψ_e dS from the spheroidal coordinates of the wall: ψ_e = Σ c_l P_l^1(x) e^{iφ} Q_l^1(z)/Q_l^1(z0)
# (x = cos θ the reference colatitude, z = iν oblate, ν prolate), whose flux through the wall per dx dφ is (a²/k) ∂ν ψ_e, so
# ∮ v* ∂ₙψ_e dS = 2π Σ_l λ_l c_l d_l* ∫|P_l^1|² dx, λ_l = (a²/k) d ln Q_l^1/dν at ν0 = c/k, here from the closed form of Q_l
# (not kore's recurrence, nor mpmath). The toroidal field b_φ φ̂ gives b_+ = b_x + i b_y = i b_φ e^{iφ}, a scalar of order 1
# with λ b_+ = ∇²b_+ and b_+ = 0 on the wall: galerkin_heat's problem of order 1 with a fixed temperature.
def _leg(l):
    '''P_l as exact coefficients of 1, x, x², ... (Bonnet's recurrence)'''
    P0, P1 = [Q(1)], [Q(0), Q(1)]
    if l == 0:
        return P0
    for n in range(1, l):
        P2 = [Q(0)]*(n + 2)
        for k, v in enumerate(P1):
            P2[k + 1] += Q(2*n + 1, n + 1)*v
        for k, v in enumerate(P0):
            P2[k] -= Q(n, n + 1)*v
        P0, P1 = P1, P2
    return P1
_pder = lambda A: [i*v for i, v in enumerate(A)][1:] or [Q(0)]
def _pmul(A, B):
    o = [Q(0)]*(len(A) + len(B) - 1)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            o[i + j] += a*b
    return o

def lambda_q1(l, f, mp):
    '''λ_l of m = 1 outside the wall at 70 digits: Q_l = P_l L/2 - W_(l-1), L = ln((z + 1)/(z - 1)), W_(l-1) =
    Σ_k P_(k-1) P_(l-k)/k (Christoffel), Q_l^1 ∝ (z² - 1)^½ Q_l', so d ln Q_l^1/dz = z/(z² - 1) + Q_l''/Q_l'
    '''
    with mp.workdps(70):
        ev = lambda A, z: sum((mp.mpf(v.numerator)/v.denominator*z**i for i, v in enumerate(A)), mp.mpc(0))
        fm = mp.mpf(f.numerator)/f.denominator
        a, c = (1 - fm)**(-mp.mpf(1)/3), (1 - fm)**(mp.mpf(2)/3)
        k = mp.sqrt(abs(a*a - c*c))
        z, dz = (mp.mpc(0, c/k), mp.mpc(0, 1)) if a > c else (mp.mpc(c/k), mp.mpc(1))
        P = _leg(l)
        W = [Q(0)]*l
        for kk in range(1, l + 1):
            for i, v in enumerate(_pmul(_leg(kk - 1), _leg(l - kk))):
                W[i] += v/kk
        L, L1, L2 = mp.log((z + 1)/(z - 1)), -2/(z*z - 1), 4*z/(z*z - 1)**2
        P1, P2 = _pder(P), _pder(_pder(P))
        Q1 = ev(P1, z)*L/2 + ev(P, z)*L1/2 - ev(_pder(W), z)
        Q2 = ev(P2, z)*L/2 + ev(P1, z)*L1 + ev(P, z)*L2/2 - ev(_pder(_pder(W)), z)
        return +mp.re((a*a/k)*dz*(z/(z*z - 1) + Q2/Q1))

def galerkin_decay(D, symm, kind, f):
    '''the decay rates (η = 1) of the axisymmetric field of degree <= D in the spheroid (a, a, c) with an insulating exterior:
    kind 'P' poloidal (ψ = w ρ^2j z^k, even in z for symm 1, the wall natural, the exterior's energy added), 'T' toroidal'''
    if kind == 'T':
        return galerkin_heat(D, 1, symm, 'temperature', f)
    gal = Gal(); mp = gal.mp
    fm = mp.mpf(f.numerator)/f.denominator
    a, c = (1 - fm)**(-mp.mpf(1)/3), (1 - fm)**(mp.mpf(2)/3)
    mons = [(j, k) for k in range(0 if symm == 1 else 1, D, 2) for j in range((D - 1 - k)//2 + 1)]
    basis = [(1, {e: Q(1)}) for e in mons]
    G = gal.form(basis, basis)
    K = 0*G
    for dop, t, nw in ((d_wb, 2/a, mp.mpf(1)/2), (d_w, 2/a, mp.mpf(1)/2), (d_z, 1/c, mp.mpf(1))):
        ds = [dop(s) for s in basis]
        K = K - nw*t**2*gal.form(ds, ds)
    # the wall: ψ = sinθ (1 - x²)^j x^k on it; its coefficient on P_l^1 = -sinθ P_l'(x) is -∫ (1 - x²)^(j+1) x^k P_l' dx/n_l,
    # n_l = ∫ |P_l^1|² dx = 2 l (l + 1)/(2 l + 1)
    for l in range(1, D + 2):
        dP = _pder(_leg(l))
        pr = []
        for (j, k) in mons:
            B = [Q(1)]
            for _ in range(j + 1):
                B = _pmul(B, [Q(1), Q(0), Q(-1)])
            pr.append(sum(v*w*Q(2, i + i2 + k + 1) for i, v in enumerate(B) for i2, w in enumerate(dP) if (i + i2 + k) % 2 == 0))
        if not any(pr):
            continue
        lam = lambda_q1(l, f, mp)
        nl = Q(2*l*(l + 1), 2*l + 1)
        for p in range(len(mons)):
            for q in range(len(mons)):
                if pr[p] and pr[q]:
                    v = pr[p]*pr[q]/nl
                    K[p, q] += lam*mp.mpf(v.numerator)/v.denominator
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

# ======================================================================================================================
# NEW (2026-11-06, step 2) the free decay of a field: Le = 1e-8, so that the field does not act back on the flow (with 1e-6
# a viscous mode of the flow 1e-2 away moves a decay rate of the shell by 1e-9), and the decay rates are Em times those of η = 1
DEC = "par.magnetic = 1; par.B0 = 'axial'; par.Em = 1e-2; par.Le = 1e-8; par.m = %d; par.symm = %d; par.lmax = %d; "
FULL = "par.Ek = 0; par.set_scales(); par.ViscosD = 0; par.ricb = 0; par.N = 32"
SHELL = "par.Ek = 1e-3; par.set_scales(); par.ricb = 0.35; par.N = 48"
CIC = "; par.innercore = 'conducting, Chebys'; par.mu_i2o = 1.; par.Nic = 24; par.sigma_i2o = %r"
EM, DD = 1e-2, 24       # the degree of the Galerkin fields (at 20 the fourth toroidal mode is 1e-10 off)

def kore_pairs(d, sigma, k=8):
    '''the k eigenpairs of the run d nearest sigma (scipy shift-invert on A.npz, B.npz)'''
    import scipy.sparse as ss, scipy.sparse.linalg as ssl
    def load(M):
        z = np.load(os.path.join(d, M + '.npz'))
        return ss.csr_matrix((z['data'], z['indices'], z['indptr']), shape=z['shape']).astype(complex)
    A, B = load('A'), load('B')
    lu = ssl.splu((A - sigma*B).tocsc())
    op = ssl.LinearOperator(A.shape, matvec=lambda x: lu.solve(B @ x), dtype=complex)
    nu, V = ssl.eigs(op, k=k, which='LM', tol=1e-14, ncv=40)
    return sigma + 1/nu, V

def field_modes(d, shifts):
    '''the decay modes of the field in the run d (no inner-core field): real eigenvalues whose flow part (the first half of
    the eigenvector) is below 1e-6 of the field part'''
    out = []
    for s in shifts:
        lam, V = kore_pairs(d, s + 1e-7j)
        n = V.shape[0]//2
        for i, l in enumerate(lam):
            if abs(l.imag) < 1e-8 and np.linalg.norm(V[:n, i]) < 1e-6*np.linalg.norm(V[n:, i]) and \
                    min([abs(l - y) for y in out] + [1.]) > 1e-9:
                out.append(l)
    return sorted(out, key=lambda z: -z.real)

def decay_job(job):
    kind, sy, f = job
    return [EM*float(e.real) for e in galerkin_decay(DD, sy, kind, f)[:4]]

def part_field_modes():
    from concurrent.futures import ThreadPoolExecutor
    ncpu = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    F3, F2 = Q(3, 10), Q(-1, 5)
    runs = {}
    for f in (F3, F2):
        for sy in (1, -1):
            runs['sc_d_%g_%+d' % (f, sy)] = SPH % float(f) + DEC % (0, sy, 21) + FULL
    for sy in (1, -1):
        runs['sc_c_%+d' % sy] = SPH % 0.3 + DEC % (0, sy, 21) + SHELL + CIC % 1.
    for m, sy, f in ((1, 1, 0.3), (2, -1, -0.2)):
        runs['sc_dm%d' % m] = SPH % f + DEC % (m, sy, 21 - m % 2) + FULL
        runs['sc_cm%d' % m] = SPH % f + DEC % (m, sy, 21 - m % 2) + SHELL + CIC % 1.
    for m, f in ((0, 0.3), (1, -0.2)):
        runs['sc_in%d' % m] = SPH % f + DEC % (m, 1, 21 - m % 2) + SHELL
        for k, s in (('3', 1e-3), ('5', 1e-5)):
            runs['sc_s%s%d' % (k, m)] = SPH % f + DEC % (m, 1, 21 - m % 2) + SHELL + CIC % s
    with ThreadPoolExecutor(max(1, min(ncpu, 8))) as ex:
        list(ex.map(lambda kv: sm.build(*kv), runs.items()))
    jobs = [(kind, sy if kind == 'P' else -sy, f) for f in (F3, F2) for sy in (1, -1) for kind in 'PT']   # kore's symm sy
    try:
        import multiprocessing
        with multiprocessing.get_context('fork').Pool(max(1, min(ncpu, len(jobs)))) as pool:
            refs = dict(zip(jobs, pool.map(decay_job, jobs)))
    except ValueError:
        refs = dict(zip(jobs, map(decay_job, jobs)))
    near = lambda d, x: (lambda l: l[np.argmin(abs(l - x))])(kore_pairs(d, x + 1e-9j, 3)[0])
    gal = lambda f, sy: refs[('P', sy, f)] + refs[('T', -sy, f)]      # the field of kore's symm sy: P of ψ's parity sy
    # the full spheroid against the Galerkin solver
    worst, n = 0, 0
    for f in (F3, F2):
        for sy in (1, -1):
            for x in gal(f, sy):
                worst = max(worst, abs(near('sc_d_%g_%+d' % (f, sy), x) - x)/abs(x))
                n += 1
    verdict('decay', worst < 1e-10 and n == 32, 'the free decay of the axisymmetric field in a spheroid with an insulating '
            'exterior (f = 0.3 and -0.2, poloidal and toroidal, both symmetries): %d decay rates against the exact Galerkin '
            'solver (the potential field outside from the closed form of Q_l^1), max |λ (kore) - Galerkin|/|λ| = %.1e'
            % (n, worst))
    # a conducting inner core of the fluid's conductivity: the full spheroid's decay; an insulating one: the limit σ_i -> 0
    wc, nc = 0, 0
    for sy in (1, -1):
        for x in gal(F3, sy):
            wc = max(wc, abs(near('sc_c_%+d' % sy, x) - x)/abs(x))
            nc += 1
    for m in (1, 2):
        for x in field_modes('sc_dm%d' % m, (-0.15, -0.4, -0.7))[:8]:
            wc = max(wc, abs(near('sc_cm%d' % m, x) - x)/abs(x))
            nc += 1
    wr, we, ni = 0, 0, 0
    for m in (0, 1):
        for x in field_modes('sc_in%d' % m, (-0.15, -0.4, -0.7))[:6]:
            l3, l5 = near('sc_s3%d' % m, x), near('sc_s5%d' % m, x)
            wr = max(wr, abs((l3 - x)/(100*(l5 - x)) - 1))                # the limit is approached as σ_i
            we = max(we, abs(l5 - (l3 - l5)/99 - x)/abs(x))                 # and the linear extrapolation reaches it
            ni += 1
    verdict('cores', wc < 1e-10 and nc >= 28 and wr < 0.02 and we < 1e-8 and ni >= 10,
            'a conducting inner core of the fluid\'s conductivity (ricb 0.35) gives the decay of the full spheroid: %d rates '
            '(m = 0 against the Galerkin solver, m = 1, 2 against kore\'s full spheroid) to %.1e; an insulating inner core is '
            'the limit of a conducting one: %d decay rates approached as σ_i (σ_i = 1e-3, 1e-5: |ratio/100 - 1| <= %.1e), '
            'the extrapolation to σ_i = 0 within %.1e' % (nc, wc, ni, wr, we))

# NEW (2026-11-06, step 2) the exact spheroid and the first-order equations ('uniform', whose map is the spheroid's to first
# order) must part at second order in f: the magnetic modes of a viscous shell with a conducting inner core and an insulating
# exterior (every term of the field: the Lorentz force, the induction, the diffusion, the walls and the ICB)
MF = ("par.magnetic = 1; par.B0 = 'axial'; par.Ek = 1e-3; par.Em = 1e-3; par.Le = 0.1; par.set_scales(); par.ricb = 0.35; "
      "par.innercore = 'conducting, Chebys'; par.Nic = 24; par.N = 48; par.lmax = 24; par.m = 1; par.symm = 1")

def part_first_order():
    from concurrent.futures import ThreadPoolExecutor
    runs = {'sc_fo_0': MF + '; par.centrifugal = 0'}
    for f in (0.005, 0.01):
        runs['sc_fo_s_%g' % f] = SPH % f + MF
        runs['sc_fo_u_%g' % f] = "par.centrifugal_shape = 'uniform'; par.centrifugal_f = %r; " % f + MF
    with ThreadPoolExecutor(5) as ex:
        list(ex.map(lambda kv: sm.build(*kv), runs.items()))
    near = lambda d, x: (lambda l: l[np.argmin(abs(l - x))])(kore_pairs(d, x + 1e-9j, 3)[0])
    orders, rel = [], []
    for x in kore_pairs('sc_fo_0', -0.02 + 1.0j, 10)[0]:
        d = {f: near('sc_fo_s_%g' % f, x) - near('sc_fo_u_%g' % f, x) for f in (0.005, 0.01)}
        orders.append(np.log2(abs(d[0.01])/abs(d[0.005])))
        rel.append(abs(d[0.01])/abs(near('sc_fo_u_0.01', x) - x))
    verdict('first order', len(orders) == 10 and max(abs(np.array(orders) - 2)) < 0.1 and max(rel) < 0.05,
            'magnetic modes of a viscous shell (Le 0.1, a conducting inner core, an insulating exterior): the exact spheroid '
            'and the first-order equations part as f² (orders %.2f to %.2f from f = 0.01 and 0.005, %d modes), at f = 0.01 '
            'by %.0e to %.0e of the first-order shift' % (min(orders), max(orders), len(orders), min(rel), max(rel)))

if __name__ == '__main__':
    part_formulas()
    part_couplings()
    part_walls()
    part_modes()
    part_field_modes()
    part_first_order()
    print('spheroid_check.py: %s (%.0f s)' % ('FAIL ' + ', '.join(fails) if fails else 'ok', time.time() - T0))
    sys.exit(1 if fails else 0)
