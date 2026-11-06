#!/usr/bin/env python3
'''
derive_diff.py: the dissipative terms of the deformed body (centrifugal = 1, bin/centrifugal.py section 6: the viscous force,
the thermal and the magnetic diffusion and the stress-free and fixed-flux walls) derived with sympy from their definitions,
and the couplings kore builds checked against them (NEW 2026-11-05, phase 2).

A. From the map alone (Cartesian components, first order in a bookkeeping factor δ: ξ -> δ ξ; explicit smooth radial
   profiles ε, ρ0, ν, κ, η and polynomial fields w, s, b̃, so that every identity is exact rational arithmetic, checked at
   three points to 50 digits; the profiles are functions of the label r = |α| of the level surface):
   1. the viscous force per unit mass V(u) = (1/ρ0) ∇⋅(ρ0 ν σ(u)), σ(u) = ∇u + (∇u)ᵀ - (2/3)(∇⋅u) I: its covariant form is
      Fᵀ V_X(F w/J) = V(w) + ∇(ξ⋅V w) - ξ×(∇×V w) + V(∇×(ξ×w) - ξ ∇⋅w) + δ_cV(w),
      δ_cV(w) = -(ξ⋅∇ν) ∇⋅σ(w) - ∇(ξ⋅∇μ/ρ0)⋅σ(w), μ = ρ0 ν (the change of the profiles), and V(w) = ν ∇⋅σ(w) + σ(w)∇μ/ρ0
      with ∇⋅σ(w) = (4/3)∇(∇⋅w) - ∇×∇×w;
   2. the traction on the level surface, Fᵀ σ_X F⁻ᵀ r̂ (its tangential part is the stress-free condition), at first order,
      and the identities that write it with vector spherical harmonics (centrifugal.wall_traction);
   3. the heat: J ∇_X⋅(k ∇_X s) = ∇⋅(k M⁻¹ ∇s), and the flux through the level surface, (F⁻ᵀ r̂)⋅∇_X s = r̂⋅M⁻¹∇s/J;
   4. the magnetic diffusion: the electric field Fᵀ(η ∇_X×b) of b = F b̃/J is η M ∇×(M b̃) at first order;
   5. NEW (2026-11-06) the differential rotation, ΔΩ(α) given in the reference space (of r and cos θ): the force
      W(u) = (U0⋅∇)u + (u⋅∇)U0 of U0 = ΔΩ ẑ×X, covariant, is W(w) + δW up to a gradient (centrifugal.rotation_force),
      its advection of s and its induction (no metric part).
B. The rows in spherical components, with explicit radial functions (polynomials with rational coefficients; the rows are
   linear in each of them, so this checks the same identities as symbolic functions would), the stress tensor from the
   velocity gradient in spherical components, every field kept expanded as a Laurent polynomial in r, cos θ and sin θ
   (class LP: exact products and derivatives, float coefficients), and the projection on Y_l' by Gauss-Legendre quadrature
   in cos θ, at three radii:
   0. the spherical rows against kore's own operators.viscous_diffusion, thermal_diffusion and magnetic_diffusion (their
      formulas read from bin/operators.py), which checks this script's vector calculus;
   1. the couplings of centrifugal.diffusion_build against the projection of the first-order terms of A on kore's rows:
      r̂⋅∇×∇×δV (u) and r̂⋅∇×δV (v), δV the formula of A1 (its gradient dropped: the rows are curls); -∇⋅(κp (M - I)∇s)
      (h); r̂⋅∇×δ𝐄 (f) and r̂⋅∇×∇×δ𝐄 (g), δ𝐄 = -η [(M - I)∇×b̃ + ∇×((M - I)b̃)] (with the closed form of M - I, checked by
      derive.py); every block of a column degree against the rows of its neighbours, |Δl| <= 4; NEW (2026-11-06) the
      couplings of centrifugal.rotation_build against r̂⋅∇×∇×δW (u) and r̂⋅∇×δW (v) for ΔΩ = f(r) P_n, |Δl| <= n + 3.
C. The wall rows of centrifugal.wall_dissipative: the S and T parts of the first-order traction of A2 (the formula with the
   tensors, built here in spherical components) at the outer and an inner wall, and the first-order flux, by quadrature.
D. NEW (2026-11-06) the ICB row of the electric field at a conducting inner core, centrifugal.icb_electric: the S part of
   the first-order field of A4 per unit η, (M - I)∇×b̃ + ∇×((M - I) b̃), at r = R, by quadrature.

Usage: python3 derive_diff.py [--quick] [--skip-a]   (B, C and D derive their columns in parallel, on up to 16 cores; --quick
checks fewer cases; prints each check and 'derive_diff.py: ok', or the failures)
'''
import sys, os, types, re, ast, time
import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.normpath(os.path.join(HERE, '..', '..', 'bin'))
sys.path[:0] = [BIN, os.path.normpath(os.path.join(HERE, '..', 'magnetic'))]
sys.modules.setdefault('parameters', types.SimpleNamespace(par=types.SimpleNamespace(lmax=40, m=0)))
quick = '--quick' in sys.argv
Q = sp.Rational
bad = []
T0 = time.time()

def verdict(ok, what):
    print(('  ok    ' if ok else '  FAIL  ') + what, flush=True)
    if not ok:
        bad.append(what)

# ======================================================================================================================
def part_a():
    print('A. the covariant forms, from the map (Cartesian, first order in δ, exact)', flush=True)
    import random
    x, y, z, d = sp.symbols('x y z delta', real=True)
    V = (x, y, z)
    s2 = x**2 + y**2 + z**2
    r_ = sp.sqrt(s2)
    X0 = sp.Matrix(V)
    zh = sp.Matrix([0, 0, 1])
    rh = X0/r_
    # explicit profiles of the label r (functions of r², smooth and even) and polynomial fields
    e = Q(3, 10) + Q(1, 5)*s2 - Q(1, 7)*s2**2           # ε
    rho = 1 + Q(1, 3)*s2 + Q(2, 5)*s2**2                 # ρ0
    nu = Q(2, 3) + Q(1, 4)*s2 - Q(1, 9)*s2**2           # ν
    random.seed(3)
    def rpoly(deg=2):
        mons = [x**i*y**j*z**k for i in range(deg + 1) for j in range(deg + 1) for k in range(deg + 1) if i + j + k <= deg]
        return sum(Q(random.randint(-9, 9), random.randint(1, 7))*mm for mm in mons)
    w = sp.Matrix([rpoly(), rpoly(), rpoly()])
    s_ = rpoly(3)                                        # an entropy

    grad = lambda f: sp.Matrix([sp.diff(f, v) for v in V])
    div = lambda W: sum(sp.diff(W[i], V[i]) for i in range(3))
    curl = lambda W: sp.Matrix([sp.diff(W[2], y) - sp.diff(W[1], z), sp.diff(W[0], z) - sp.diff(W[2], x),
                                sp.diff(W[1], x) - sp.diff(W[0], y)])
    jac = lambda W: sp.Matrix(3, 3, lambda i, k: sp.diff(W[i], V[k]))      # (∇W)_ik = ∂_k W_i
    def sigma(W):
        G = jac(W)
        return G + G.T - Q(2, 3)*G.trace()*sp.eye(3)
    def visc(W):
        '''V(W) = (1/ρ0) ∇⋅(ρ0 ν σ(W)), from the definition'''
        S = rho*nu*sigma(W)
        return sp.Matrix([sum(sp.diff(S[i, k], V[k]) for k in range(3)) for i in range(3)])/rho
    divsig = lambda W: sp.Matrix([sum(sp.diff(sigma(W)[i, k], V[k]) for k in range(3)) for i in range(3)])

    Phi2 = (z**2 - (x**2 + y**2)/2)/2                 # r² P2/2
    xi1 = e*grad(Phi2)                                # ξ (without δ)
    G1 = jac(xi1)                                     # ∇ξ
    F = sp.eye(3) + d*G1
    first = lambda ex: ex.applyfunc(lambda v: sp.diff(v, d).subs(d, 0)) if isinstance(ex, sp.MatrixBase) else sp.diff(ex, d).subs(d, 0)
    def DX(f, k):
        '''∂/∂X_k of a function of α, at first order: Σ_a ∂_a f (δ_ak - δ G1[a, k])'''
        return sum(sp.diff(f, V[a])*((1 if a == k else 0) - d*G1[a, k]) for a in range(3))

    pts = [(Q(1, 3), Q(-1, 5), Q(2, 7)), (Q(-2, 5), Q(1, 4), Q(-1, 3)), (Q(1, 7), Q(3, 8), Q(1, 2))]
    def zero_at(expr, what):
        '''zero at the three points, to 50 digits'''
        vals = []
        for p in pts:
            sub = dict(zip(V, p))
            vals += [abs(sp.N(v.subs(sub), 50)) for v in (expr if isinstance(expr, sp.MatrixBase) else [expr])]
        mx = max(vals)
        verdict(mx < 1e-35, what + '   (%.0e)' % mx)

    # 1. the viscous force
    mu = rho*nu
    zero_at(visc(w) - (nu*(Q(4, 3)*grad(div(w)) - curl(curl(w))) + sigma(w)*grad(mu)/rho),
            '1. V(w) = ν [(4/3)∇(∇⋅w) - ∇×∇×w] + σ(w)∇μ/ρ0,  μ = ρ0 ν')
    u = w + d*(G1*w - G1.trace()*w)                    # F w/J at first order
    GuX = sp.Matrix(3, 3, lambda i, k: DX(u[i], k))     # ∇_X u
    sigX = GuX + GuX.T - Q(2, 3)*GuX.trace()*sp.eye(3)
    S = mu*sigX
    VX = sp.Matrix([sum(DX(S[i, k], k) for k in range(3)) for i in range(3)])/rho
    Vt1 = first((sp.eye(3) + d*G1.T)*VX)              # the first-order part of Fᵀ V_X(u)
    Vw = visc(w)
    du = curl(xi1.cross(w)) - xi1*div(w)               # ∇×(ξ×w) - ξ ∇⋅w
    dcV = -xi1.dot(grad(nu))*divsig(w) - sigma(w)*grad(xi1.dot(grad(mu))/rho)
    zero_at(Vt1 - (grad(xi1.dot(Vw)) - xi1.cross(curl(Vw)) + visc(du) + dcV),
            '   Fᵀ V_X(F w/J) - V(w) = ∇(ξ⋅V w) - ξ×(∇×V w) + V(∇×(ξ×w) - ξ ∇⋅w) + δ_cV(w),  δ_cV(w) = -(ξ⋅∇ν)∇⋅σ(w) - ∇(ξ⋅∇μ/ρ0)⋅σ(w)')

    # 2. the traction on the level surface and its pieces
    Gw = jac(w)
    T1 = first((sp.eye(3) + d*G1.T)*sigX*(sp.eye(3) - d*G1.T)*rh)
    K = Gw*G1
    du0 = G1*w - G1.trace()*w                          # u - w at the same α
    zero_at(T1 - (sigma(du0)*rh - (K + K.T - Q(2, 3)*K.trace()*sp.eye(3))*rh + G1.T*sigma(w)*rh - sigma(w)*G1.T*rh),
            '2. Fᵀσ_X F⁻ᵀ r̂ at first order: σ(δu0) r̂ - [∇w∇ξ + (∇w∇ξ)ᵀ - (2/3) tr] r̂ + (∇ξ)ᵀσ(w) r̂ - σ(w)(∇ξ)ᵀ r̂,  δu0 = (∇ξ)w - (∇⋅ξ)w')
    H = sp.diag(-Q(1, 2), -Q(1, 2), 1)
    eta = sp.simplify(r_*sp.diff(e, x)*r_/x)          # η = r ε'
    zero_at(G1 - (e*H + eta*(H*rh)*rh.T), '   ∇ξ = ε H + η (H r̂) r̂ᵀ,  H = ∇∇(r²P2/2) (so (∇ξ)ᵀ r̂ = ε H r̂ + η P2 r̂, ∇⋅ξ = η P2)')
    zero_at(Gw*G1*rh - Gw*((e + eta)*H*rh), '   (∇w)(∇ξ) r̂ = (a⋅∇) w,  a = (ε + η) H r̂')
    wr = w.dot(rh)
    zero_at(Gw.T*rh - (grad(wr) - (w - wr*rh)/r_), '   (∇w)ᵀ r̂ = ∇w_r - (w - w_r r̂)/r')
    zero_at(zh.T*sigma(w) - (2*grad(w[2]) - zh.cross(curl(w)) - Q(2, 3)*div(w)*zh).T, '   ẑ⋅σ(w) = 2∇w_z - ẑ×(∇×w) - (2/3)(∇⋅w) ẑ')
    zero_at(Gw*zh - (grad(w[2]) - zh.cross(curl(w))), '   (ẑ⋅∇)w = ∇w_z - ẑ×(∇×w)')

    # 3. the heat
    kap = Q(1, 2) + Q(1, 3)*s2                          # κ p
    gX = sp.Matrix([DX(s_, k) for k in range(3)])
    lhs = first(F.det()*sum(DX(kap*gX[k], k) for k in range(3)))
    M1 = first(F.T*F/F.det())
    zero_at(lhs + div(kap*M1*grad(s_)), '3. J ∇_X⋅(k ∇_X s) = ∇⋅(k M⁻¹∇s): first order -∇⋅(k (M - I)∇s)')
    nrm = (sp.eye(3) - d*G1.T)*rh                     # F⁻ᵀ r̂, normal to the level surface
    zero_at(first(nrm.dot(gX)) + (M1*grad(s_)).dot(rh) + G1.trace()*grad(s_).dot(rh),
            '   (F⁻ᵀr̂)⋅∇_X s = r̂⋅M⁻¹∇s/J: the flux row ∂ᵣs - r̂⋅(M - I)∇s = 0 at first order')

    # 4. the magnetic diffusion
    bt = sp.Matrix([rpoly(), rpoly(), rpoly()])        # b̃
    eta_m = Q(3, 4) + Q(1, 6)*s2
    bX = F*bt/F.det()
    jX = sp.Matrix([DX(bX[2], 1) - DX(bX[1], 2), DX(bX[0], 2) - DX(bX[2], 0), DX(bX[1], 0) - DX(bX[0], 1)])
    zero_at(first(F.T*(eta_m*jX)) - eta_m*(M1*curl(bt) + curl(M1*bt)),
            '4. Fᵀ(η ∇_X×b), b = F b̃/J: first order η [(M - I)∇×b̃ + ∇×((M - I) b̃)]')

    # 5. NEW (2026-11-06): the differential rotation, ΔΩ a function of the reference point α (of r and z)
    Om = Q(1, 3) + Q(1, 5)*s2 + Q(2, 7)*z**2 - Q(1, 9)*z**4/s2
    U0 = Om*zh.cross(X0 + d*xi1)                      # U0 = ΔΩ ẑ×X in the body
    U00 = Om*zh.cross(X0)                             # its spherical formula
    WX = sp.Matrix([sum(U0[k]*DX(u[i], k) for k in range(3)) + sum(u[k]*DX(U0[i], k) for k in range(3)) for i in range(3)])
    om0 = curl(U00)
    P2c = (3*z**2/s2 - 1)/2
    dW = -U00.cross(curl(M1*w)) + (e + eta*P2c)*w.cross(om0) - eta*Om*(zh.cross(w) - (z/r_)*rh.cross(w))
    zero_at(curl(first((sp.eye(3) + d*G1.T)*WX) - dW),
            '5. Fᵀ W_X(F w/J) - W(w) = δW + a gradient, W(u) = (U0⋅∇)u + (u⋅∇)U0: δW = -U00×∇×((M - I)w) + (ε + η P2) w×ω0 '
            '- η ΔΩ (ẑ×w - x r̂×w)')
    zero_at(first((sp.eye(3) - d*G1)*zh.cross(X0 + d*xi1)), '   F⁻¹(ẑ×X) = ẑ×α: the map commutes with the rotations about the axis')
    zero_at(first(F.det()*sum(U0[k]*DX(s_, k) for k in range(3))) - eta*P2c*U00.dot(grad(s_)),
            '   J U0⋅∇_X s = (1 + η P2) U00⋅∇s at first order (the heat rows)')
    zero_at(first(F.T*U0.cross(bX)), '   Fᵀ(U0×b) = U00×b̃: the induction by the background flow has no metric part')
    print('  (%.0f s)' % (time.time() - T0), flush=True)

if __name__ == '__main__' and '--skip-a' not in sys.argv:   # (NEW 2026-11-06: not when imported, by spheroid_check.py)
    part_a()

# ======================================================================================================================
# B and C: spherical components (r, θ, φ), ∂φ -> i m, kore's Schmidt harmonics (tests/magnetic/derive.py's Y). With explicit
# radial functions every component of every field here is a Laurent polynomial in r, c = cos θ and s = sin θ. sympy's
# expression trees for these rows grow to 1e7 operations (an hour per column), so the fields are kept expanded instead, as
# {(power of r, of c, of s): coefficient} in a canonical basis (s² = 1 - c²): products and derivatives are exact, the
# coefficients floats, and a column takes seconds.
import derive as dv
import centrifugal as cf
from functools import lru_cache

@lru_cache(maxsize=None)
def _canon(a, b):
    '''c^a s^b in the basis c^a, c^a s (a >= 0) and s^b, c s^b (b < 0): ((a', b'), integer coefficient), ...'''
    if b >= 2:              # s² = 1 - c²
        parts = [(k, v) for k, v in _canon(a, b - 2)] + [(k, -v) for k, v in _canon(a + 2, b - 2)]
    elif b < 0 and a >= 2:  # c² = 1 - s²
        parts = [(k, v) for k, v in _canon(a - 2, b)] + [(k, -v) for k, v in _canon(a - 2, b + 2)]
    else:
        return (((a, b), 1),)
    out = {}
    for k, v in parts:
        out[k] = out.get(k, 0) + v
    return tuple((k, v) for k, v in out.items() if v != 0)

class LP:
    '''a Laurent polynomial in r, c = cos θ and s = sin θ, {(p, a, b): coefficient} for r^p c^a s^b (canonical in c, s)'''
    __slots__ = ('t',)
    def __init__(self, t=None):
        self.t = t if t is not None else {}
    @staticmethod
    def of(terms):
        out = {}
        for (p, a, b), v in terms:
            for (a2, b2), w in _canon(a, b):
                k = (p, a2, b2)
                out[k] = out.get(k, 0) + w*v
        return LP({k: v for k, v in out.items() if v != 0})
    @staticmethod
    def lift(o):
        return o if isinstance(o, LP) else LP({(0, 0, 0): complex(o)} if o != 0 else {})
    def __add__(self, o):
        t = dict(self.t)
        for k, v in LP.lift(o).t.items():
            t[k] = t.get(k, 0) + v
        return LP({k: v for k, v in t.items() if v != 0})
    __radd__ = __add__
    def __neg__(self):
        return LP({k: -v for k, v in self.t.items()})
    def __sub__(self, o):
        return self + (-LP.lift(o))
    def __rsub__(self, o):
        return LP.lift(o) - self
    def __mul__(self, o):
        if not isinstance(o, LP):
            return LP({k: o*v for k, v in self.t.items()}) if o != 0 else LP()
        A, B = (self, o) if len(self.t) <= len(o.t) else (o, self)
        return LP.of([((p1 + p2, a1 + a2, b1 + b2), v1*v2) for (p1, a1, b1), v1 in A.t.items() for (p2, a2, b2), v2 in B.t.items()])
    __rmul__ = __mul__
    def dr(self):
        return LP({(p - 1, a, b): p*v for (p, a, b), v in self.t.items() if p != 0})
    def dth(self):
        '''∂θ (c^a s^b) = -a c^(a-1) s^(b+1) + b c^(a+1) s^(b-1)'''
        return LP.of([((p, a - 1, b + 1), -a*v) for (p, a, b), v in self.t.items() if a != 0] +
                     [((p, a + 1, b - 1), b*v) for (p, a, b), v in self.t.items() if b != 0])
    def at_r(self, R):
        '''the θ part at r = R, as an LP in c, s'''
        out = {}
        for (p, a, b), v in self.t.items():
            out[(0, a, b)] = out.get((0, a, b), 0) + v*R**p
        return LP(out)
    def ev(self, radii, x):
        '''values [radius, node] at the radii and the nodes x = cos θ'''
        if not self.t:
            return np.zeros((len(radii), len(x)), dtype=complex)
        k = np.array(list(self.t.keys()), dtype=float)
        v = np.array(list(self.t.values()), dtype=complex)
        rr = np.asarray(radii, dtype=float)[:, None, None]
        xs, ss = np.asarray(x)[None, None, :], np.sqrt(1 - np.asarray(x)**2)[None, None, :]
        return np.sum(v[None, :, None]*rr**k[None, :, 0, None]*xs**k[None, :, 1, None]*ss**k[None, :, 2, None], axis=1)

def mono(p=0, a=0, b=0, v=1.):
    return LP({(p, a, b): complex(v)})
RR, CC, SS = mono(1), mono(a=1), mono(b=1)
IR, IS = mono(-1), mono(b=-1)                # 1/r, 1/sin θ
COT = mono(a=1, b=-1)                        # cot θ

def radial(ex):
    '''an LP from a sympy polynomial in r (dv.r)'''
    return LP({(k[0], 0, 0): complex(v) for k, v in sp.Poly(sp.expand(ex), dv.r).terms()})

C_, S_ = sp.symbols('C_ S_')
@lru_cache(maxsize=None)
def Ylp(l, m):
    '''kore's Schmidt harmonic Y_l^m(θ) (derive.Y) as an LP'''
    ex = sp.expand(dv.Y(l, m).subs({sp.sin(dv.th): S_, sp.cos(dv.th): C_}))
    return LP.of([((0, k[0], k[1]), complex(v)) for k, v in sp.Poly(ex, C_, S_).terms()])

# --- vector calculus, vectors = tuples of 3 LPs (r, θ, φ components), ∂φ -> i m
vadd = lambda *vs: tuple(sum((v[i] for v in vs), LP()) for i in range(3))
vsc = lambda c, v: tuple(c*x for x in v)
cross = lambda A, B: (A[1]*B[2] - A[2]*B[1], A[2]*B[0] - A[0]*B[2], A[0]*B[1] - A[1]*B[0])
def curl(A, m):
    Ar, At, Ap = A
    return ((CC*Ap + SS*Ap.dth() - 1j*m*At)*IR*IS, (1j*m*Ar*IS - (RR*Ap).dr())*IR, ((RR*At).dr() - Ar.dth())*IR)
def div(A, m):
    return (RR*RR*A[0]).dr()*IR*IR + (CC*A[1] + SS*A[1].dth())*IR*IS + 1j*m*A[2]*IR*IS
def grad(g, m):
    return (g.dr(), g.dth()*IR, 1j*m*g*IR*IS)
def gradv(v, m):
    '''the velocity gradient, (∇v)_ik = ê_i⋅(ê_k⋅∇)v (i: component, k: direction), with the derivatives of the unit vectors'''
    vr, vt, vp = v
    return [[vr.dr(), (vr.dth() - vt)*IR, (1j*m*vr*IS - vp)*IR],
            [vt.dr(), (vt.dth() + vr)*IR, (1j*m*vt*IS - vp*COT)*IR],
            [vp.dr(), vp.dth()*IR, (1j*m*vp*IS + vr + vt*COT)*IR]]
T_ = lambda M: [[M[k][i] for k in range(3)] for i in range(3)]
mmul = lambda A, B: [[sum((A[i][j]*B[j][k] for j in range(3)), LP()) for k in range(3)] for i in range(3)]
mvec = lambda M, v: tuple(sum((M[i][k]*v[k] for k in range(3)), LP()) for i in range(3))
trace = lambda M: M[0][0] + M[1][1] + M[2][2]
def sym_dev(G):
    '''G + Gᵀ - (2/3) tr G I'''
    t = trace(G)
    return [[G[i][k] + G[k][i] - ((2/3)*t if i == k else 0) for k in range(3)] for i in range(3)]
sigma = lambda v, m: sym_dev(gradv(v, m))
def tdiv(T, m):
    '''the divergence of a symmetric tensor'''
    d = lambda i, k: T[i][k]
    return (d(0, 0).dr() + d(0, 1).dth()*IR + 1j*m*d(0, 2)*IR*IS + (2*d(0, 0) - d(1, 1) - d(2, 2) + COT*d(0, 1))*IR,
            d(1, 0).dr() + d(1, 1).dth()*IR + 1j*m*d(1, 2)*IR*IS + ((d(1, 1) - d(2, 2))*COT + 3*d(0, 1))*IR,
            d(2, 0).dr() + d(2, 1).dth()*IR + 1j*m*d(2, 2)*IR*IS + (3*d(0, 2) + 2*COT*d(1, 2))*IR)

# --- explicit radial functions (polynomials with rational coefficients; ln ρ0 a polynomial, so λ1 is one too)
r = dv.r
EXPL = {'e': Q(7, 10) + Q(3, 10)*r**2 - Q(9, 20)*r**4,            # ε
        'nu': Q(2, 3) + Q(1, 4)*r**2 - Q(1, 9)*r**4 + r**5/7,     # ν, κ p or η: the profile c of the labels
        'lr': -Q(11, 10)*r**2 + Q(7, 20)*r**3,                     # ln ρ0
        'P': 1 - r**2/5 + Q(3, 5)*r**4 - Q(1, 3)*r**5, 'T': Q(1, 5) + r**3 - r**6/2,
        'S': Q(1, 2) - r**2 + Q(2, 7)*r**3 + r**4/3,
        'F': r**2 + r**3/5 - Q(3, 10)*r**5, 'G': Q(2, 5)*r**3 + r**4 - r/10,
        'om': Q(1, 4) - Q(2, 5)*r**2 + Q(1, 3)*r**3 + Q(1, 6)*r**4}     # NEW (2026-11-06): the shape f_n of ΔΩ = f_n P_n
RAD = {k: radial(v) for k, v in EXPL.items()}
EPS, NU, LAM1 = RAD['e'], RAD['nu'], RAD['lr'].dr()
KAP = NU.dr() + NU*LAM1                       # μ'/ρ0, μ = ρ0 ν
RADII = np.array([0.45, 0.7, 0.93])
lam = lambda ex: (lambda f: (lambda rr: np.broadcast_to(np.asarray(f(rr), dtype=float), np.shape(rr))))(sp.lambdify(r, ex, 'numpy'))
FUNS = {k: [lam(sp.diff(EXPL[k], r, j)) for j in range(7)] for k in EXPL}
a_ = [EXPL['e']]
for _ in range(4):
    a_.append(sp.expand(r*sp.diff(a_[-1], r)))
AMPF = {a: lam(ex) for a, ex in zip('ENCQK', a_)}   # E = ε, N = r ε', C = r N', Q = r C', K = r Q'
P2 = 1.5*CC*CC - 0.5
XI = (EPS*RR*P2, -1.5*EPS*RR*CC*SS, LP())     # ξ = ε ∇(r²P2/2) = ε r (P2, -(3/2) cos θ sin θ, 0)

def qst(l, m, q, s, t):
    '''q R_l + s S_l + t T_l (S_l = r∇Y_l, T_l = ∇×(𝐫 Y_l)), q, s, t radial LPs'''
    Y = Ylp(l, m)
    dY, mY = Y.dth(), 1j*m*Y*IS
    return (q*Y, s*dY + t*mY, s*mY - t*dY)

def flow(l, m, X):
    '''kore's flow: P, ρ0⁻¹ ∇×∇×(ρ0 P Y 𝐫) (q = L P/r, s = P' + P/r + λ1 P); T, ∇×(T Y 𝐫)'''
    if X == 'P':
        P = RAD['P']
        return qst(l, m, l*(l + 1)*P*IR, P.dr() + P*IR + LAM1*P, LP())
    return qst(l, m, LP(), LP(), RAD['T'])

def field(l, m, X):
    '''kore's field: F, ∇×∇×(F Y 𝐫); G, ∇×(G Y 𝐫)'''
    if X == 'F':
        F = RAD['F']
        return qst(l, m, l*(l + 1)*F*IR, F.dr() + F*IR, LP())
    return qst(l, m, LP(), LP(), RAD['G'])

def visc(v, m):
    '''V(v) = (1/ρ0) ∇⋅(ρ0 ν σ(v)) = ν ∇⋅σ(v) + (μ'/ρ0) σ(v) r̂ (μ = ρ0 ν radial; A1)'''
    S = sigma(v, m)
    return vadd(vsc(NU, tdiv(S, m)), (KAP*S[0][0], KAP*S[1][0], KAP*S[2][0]))

def metric(v):
    '''(M - I) v, the closed form of centrifugal.py (section 1; checked by derive.py)'''
    et = RR*EPS.dr()
    vr, vt, vp = v
    vz = vr*CC - vt*SS
    zs = (CC, -SS, LP())
    return tuple(EPS*(3*vz*zs[i] - v[i]) + et*(1.5*CC*(vz*(1 if i == 0 else 0) + vr*zs[i]) - (vr if i == 0 else 0) - P2*v[i])
                 for i in range(3))

def delta_V(w, m):
    '''δV of A1 without its gradient: -ξ×(∇×V w) + V(∇×(ξ×w) - ξ ∇⋅w) - (ξ⋅∇ν)∇⋅σ(w) - σ(w)∇(ξ⋅∇μ/ρ0)'''
    out = vsc(-1, cross(XI, curl(visc(w, m), m)))
    du = vadd(curl(cross(XI, w), m), vsc(-div(w, m), XI))
    S = sigma(w, m)
    return vadd(out, visc(du, m), vsc(-XI[0]*NU.dr(), tdiv(S, m)), vsc(-1, mvec(S, grad(XI[0]*KAP, 0))))   # (ξ_r κ is axisymmetric)

# The projections integrate in x = cos θ (Gauss-Legendre, exact: a field of order m times Y_l'^m is a polynomial in x),
# not in θ as derive.py: the terms of a row have factors 1/sinᵏθ that cancel in their sum, and nodes in θ, which crowd the
# poles (θ ~ 1/N²), would amplify the round-off of that cancellation; nodes in x stay at θ > 2.4/N.
XQ, WXQ = np.polynomial.legendre.leggauss(40)

def project_num(ex, lrow, m, radii=RADII):
    '''the Y_lrow coefficient of the LP ex at the radii'''
    yr = Ylp(lrow, m).ev([1.], XQ)[0]
    return 2*np.pi*np.sum(WXQ[None, :]*ex.ev(radii, XQ)*np.conj(yr)[None, :], axis=1)/(4*np.pi/(2*lrow + 1))

# ---------------------------------------------------------------------------------- B0: the spherical rows against kore's
def kore_spherical():
    '''operators.viscous_diffusion, thermal_diffusion and magnetic_diffusion read from bin/operators.py, and the names of
    their labels'''
    tree = ast.parse(open(os.path.join(BIN, 'operators.py')).read())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('viscous_diffusion', 'thermal_diffusion', 'magnetic_diffusion')]
    names = {n.id for f in fns for n in ast.walk(f) if isinstance(n, ast.Name) and re.match(r'^[uvhfg]\d\w*_D\d$', n.id)}
    return fns, names

def label_value(lab, X):
    '''the value at RADII of a label of kore's spherical operators applied to the radial function X: r^rpower ρ0^rhopower
    f1^(d1) f2^(d2) X^(dx), as utils.decode_label and radial_profiles.burrito (lh1 k: λ1 λ_k)'''
    sec, rx, dx = lab[0], int(lab[1]), int(lab[-1])
    rpow, rhop = {'u': (5, 1), 'v': (3, 1), 'h': (2, 0), 'f': (2, 0), 'g': (2, 0)}[sec]
    val = RADII**(rpow - rx)*np.exp(FUNS['lr'][0](RADII))**rhop*FUNS[X][dx](RADII)
    for code, k in re.findall(r'([a-z]{2}[a-z0-9])(\d)', lab[2:-3]):
        if code in ('vsc', 'kps', 'eta'):
            val = val*FUNS['nu'][int(k)](RADII)
        elif code == 'lho':
            val = val*FUNS['lr'][int(k)](RADII)
        elif code in ('lh1', 'lh2'):
            val = val*FUNS['lr'][int(code[2])](RADII)*FUNS['lr'][int(k)](RADII)
        else:
            raise ValueError(lab)
    return val

def part_b0():
    print('B0. the spherical rows of this script against kore\'s operators (bin/operators.py)', flush=True)
    fns, names = kore_spherical()
    # the fields of this script against derive.py's definitions (ρ0⁻¹∇×∇×(ρ0 P Y 𝐫), ∇×∇×(F Y 𝐫), ...), at a few points
    dv.P, dv.T, dv.F, dv.G, dv.lr = (EXPL[k] for k in ('P', 'T', 'F', 'G', 'lr'))
    th0 = np.array([0.4, 1.3, 2.6])
    err = 0
    for m, l in ((1, 3), (2, 2)):
        for X, mk in (('P', flow), ('T', flow), ('F', field), ('G', field)):
            ref = dv.velocity(l, m, X) if X in 'PT' else dv.bfield(l, m, X)
            mine = mk(l, m, X)
            for i in range(3):
                f = sp.lambdify((dv.r, dv.th), ref[i], 'numpy')
                a = np.array([[complex(f(ri, t)) for t in th0] for ri in RADII])
                err = max(err, np.abs(a - mine[i].ev(RADII, np.cos(th0))).max()/max(1, np.abs(a).max()))
    verdict(err < 1e-12, 'the flows and fields of P, T, F, G against derive.py\'s definitions (rel. %.0e)' % err)
    for m, l in ((1, 3), (0, 2), (2, 4)):
        L = l*(l + 1)
        rows = {}
        Vw = visc(flow(l, m, 'P'), m)
        rows[('u', 'P')] = project_num(curl(curl(Vw, m), m)[0], l, m)
        rows[('v', 'T')] = project_num(curl(visc(flow(l, m, 'T'), m), m)[0], l, m)
        rows[('h', 'S')] = project_num(div(vsc(NU, grad(RAD['S']*Ylp(l, m), m)), m), l, m)
        for X, sec in (('F', 'f'), ('G', 'g')):
            E = vsc(-1*NU, curl(field(l, m, X), m))
            rows[(sec, X)] = project_num(curl(E, m)[0] if sec == 'f' else curl(curl(E, m), m)[0], l, m)
        for (sec, X), der in rows.items():
            ns = {'par': types.SimpleNamespace(ViscosD=1., ThermaD=1., MagnetD=1., mu_i2o=1., sigma_i2o=1.), 'np': np}
            ns.update({n: label_value(n, X) for n in names if n[0] == sec})
            exec(compile(ast.Module(body=fns, type_ignores=[]), 'operators.py', 'exec'), ns)
            fn, comp = {'u': ('viscous_diffusion', 'upol'), 'v': ('viscous_diffusion', 'utor'), 'h': ('thermal_diffusion', ''),
                        'f': ('magnetic_diffusion', 'bpol'), 'g': ('magnetic_diffusion', 'btor')}[sec]
            k = ns[fn](l, sec, comp, 0)*L**3/RADII**{'u': 5, 'v': 3, 'h': 2, 'f': 3, 'g': 3}[sec]
            if sec in 'uv':
                k = k/np.exp(FUNS['lr'][0](RADII))
            err = np.abs(der - k).max()/np.abs(der).max()
            verdict(err < 1e-12, 'm=%d l=%d %s row (%s): projected %s, kore %s (rel. %.0e)'
                    % (m, l, sec, fn, np.round(der.real, 6), np.round(k.real, 6), err))

# ------------------------------------------------------------------------------------------ B1: the first-order couplings
LABEL = re.compile(r'([uvhfg])(\d)(?:lh(\d)(\d))?([ENCQK])(vs|kp|et|z[0-9a-z])(\d)(?:lho(\d))?_D(\d)$')

def candidate(blocks, row, X, lrow, lcol):
    '''kore's block at RADII: Σ coefficient r^(-rx) a c^(j) Λ X^(d) (rows u, v, h, the RHS sign), minus Σ coefficient
    r^(-1-rx) ... (rows f, g, the A sign)'''
    out = np.zeros(len(RADII), dtype=complex)
    for lab, Mx in blocks.get((row, X), {}).items():
        c = complex(Mx[lrow, lcol])
        if c == 0:
            continue
        rr, rx, i, k, a, sfx, j, lm, dd = LABEL.match(lab).groups()
        p = -int(rx) if rr in 'uvh' else -1 - int(rx)
        val = RADII**p*AMPF[a](RADII)*FUNS['om' if sfx[0] == 'z' else 'nu'][int(j)](RADII)*FUNS[X][int(dd)](RADII)
        for kk in [x for x in (i, k, lm) if x]:
            val = val*FUNS['lr'][int(kk)](RADII)
        out += c*val if rr in 'uvh' else -c*val
    return out

def legendre_lp(n):
    '''P_n(cos θ) as an LP'''
    c = np.polynomial.legendre.leg2poly([0]*n + [1])
    return sum((float(ck)*mono(a=k) for k, ck in enumerate(c) if ck != 0), LP())

def delta_W(w, m, n):
    '''NEW (2026-11-06): δW of A5 for ΔΩ = f(r) P_n(cos θ) (f = EXPL['om']), U00 = ΔΩ r sinθ e_φ:
    -U00×∇×((M - I)w) + (ε + η P2) w×ω0 - η ΔΩ (ẑ×w - x r̂×w), ω0 = ∇×U00 (axisymmetric)'''
    Om = RAD['om']*legendre_lp(n)
    U = (LP(), LP(), Om*RR*SS)
    om0 = curl(U, 0)
    et = RR*EPS.dr()
    zh, rh = (CC, -1*SS, LP()), (LP.lift(1), LP(), LP())
    return vadd(vsc(-1, cross(U, curl(metric(w), m))), vsc(EPS + et*P2, cross(w, om0)),
                vsc(-1*et*Om, vadd(cross(zh, w), vsc(-1*CC, cross(rh, w)))))

def derived_rows(kind, m, X, lcol):
    '''{row: LP} of the first-order terms for the column X of degree lcol'''
    if kind.startswith('rot'):   # NEW (2026-11-06)
        c1 = curl(delta_W(flow(lcol, m, X), m, int(kind[3:])), m)
        return {'u': curl(c1, m)[0], 'v': c1[0]}
    if kind == 'visc':
        c1 = curl(delta_V(flow(lcol, m, X), m), m)
        return {'u': curl(c1, m)[0], 'v': c1[0]}
    if kind == 'heat':
        return {'h': -1*div(vsc(NU, metric(grad(RAD['S']*Ylp(lcol, m), m))), m)}
    b = field(lcol, m, X)
    c1 = curl(vsc(-1*NU, vadd(metric(curl(b, m)), curl(metric(b), m))), m)
    return {'f': c1[0], 'g': curl(c1, m)[0]}

def one_column(task):
    '''one column of part B: the projected rows of its neighbours'''
    kind, m, X, lcol = task
    t0 = time.time()
    out = {}
    dl = int(kind[3:]) + 3 if kind.startswith('rot') else 4   # ΔΩ of degree n couples l to l ± (n + 3)
    for row, ex in derived_rows(kind, m, X, lcol).items():
        lmin = 0 if row == 'h' and m == 0 else max(m, 1)
        for lrow in range(max(lmin, lcol - dl), lcol + dl + 1):
            out[(row, lrow)] = project_num(ex, lrow, m)
    return task, out, time.time() - t0

# ------------------------------------------------------------------------------------------------------ C: the wall rows
def wall_column(task):
    '''one column of part C: the S and T parts of the first-order traction (A2, with the tensors), or the first-order
    flux, at the wall R'''
    R, m, X, lcol = task
    t0 = time.time()
    out = {}
    if X in 'PT':
        w = flow(lcol, m, X)
        G1 = gradv(XI, 0)                              # ∇ξ (axisymmetric)
        Gw, S0 = gradv(w, m), sigma(w, m)
        du0 = vadd(mvec(G1, w), vsc(-1*trace(G1), w))
        K = mmul(Gw, G1)
        rh = (LP.lift(1), LP(), LP())
        T1 = vadd(mvec(sigma(du0, m), rh), vsc(-1, mvec(sym_dev(K), rh)), mvec(T_(G1), mvec(S0, rh)),
                  vsc(-1, mvec(S0, mvec(T_(G1), rh))))
        vt, vp = T1[1].ev([R], XQ)[0], T1[2].ev([R], XQ)[0]
        for lrow in range(max(max(m, 1), lcol - 4), lcol + 5):
            Y = Ylp(lrow, m)
            dY, mY = Y.dth().ev([1.], XQ)[0], (1j*m*Y*IS).ev([1.], XQ)[0]
            nrm = lrow*(lrow + 1)*4*np.pi/(2*lrow + 1)
            integ = lambda a, b: 2*np.pi*np.sum(WXQ*(vt*np.conj(a) + vp*np.conj(b)))/nrm
            out[('u', lrow)] = R**2*integ(dY, mY)         # S_l = (0, ∂θY, i m Y/sinθ)
            out[('v', lrow)] = R*integ(mY, -dY)           # T_l = (0, i m Y/sinθ, -∂θY)
    else:
        flux = -1*metric(grad(RAD['S']*Ylp(lcol, m), m))[0]
        for lrow in range(max(0 if m == 0 else m, lcol - 4), lcol + 5):
            out[('h', lrow)] = project_num(flux, lrow, m, radii=np.array([R]))[0]
    return task, out, time.time() - t0

# ------------------------------------------------------------------- D: the ICB row of the electric field (conducting core)
def icb_column(task):
    '''one column of part D: R times the S part of δ𝐄 = (M - I)∇×b̃ + ∇×((M - I) b̃) (per unit η) at r = R'''
    R, m, X, lcol = task
    t0 = time.time()
    b = field(lcol, m, X)
    E = vadd(metric(curl(b, m)), curl(metric(b), m))
    vt, vp = E[1].ev([R], XQ)[0], E[2].ev([R], XQ)[0]
    out = {}
    for lrow in range(max(max(m, 1), lcol - 4), lcol + 5):
        Y = Ylp(lrow, m)
        dY, mY = Y.dth().ev([1.], XQ)[0], (1j*m*Y*IS).ev([1.], XQ)[0]
        nrm = lrow*(lrow + 1)*4*np.pi/(2*lrow + 1)
        out[('E', lrow)] = R*2*np.pi*np.sum(WXQ*(vt*np.conj(dY) + vp*np.conj(mY)))/nrm   # S_l = (0, ∂θY, i m Y/sinθ)
    return task, out, time.time() - t0

# ======================================================================================================================
def run(task):
    return (task[0],) + {'B': one_column, 'C': wall_column, 'D': icb_column}[task[0]](task[1])

def main():   # (NEW 2026-11-06: a function, so that spheroid_check.py can import the Laurent polynomials above)
    part_b0()

    print('B1. the couplings of centrifugal.diffusion_build against the projected first-order rows', flush=True)
    CASES = [('visc', 1, 'P', 3), ('visc', 1, 'T', 2), ('heat', 1, 'S', 3), ('mag', 1, 'F', 3), ('mag', 1, 'G', 2)] if quick else \
            [('visc', 1, 'P', 3), ('visc', 1, 'T', 2), ('visc', 0, 'P', 2), ('visc', 0, 'T', 3), ('visc', 2, 'P', 4), ('visc', 3, 'T', 4),
             ('heat', 1, 'S', 3), ('heat', 0, 'S', 2), ('heat', 2, 'S', 4),
             ('mag', 1, 'F', 3), ('mag', 1, 'G', 2), ('mag', 0, 'F', 2), ('mag', 2, 'G', 4), ('mag', 0, 'G', 3),
             ('rot0', 1, 'P', 3), ('rot2', 1, 'T', 2), ('rot2', 0, 'P', 2), ('rot4', 2, 'P', 4), ('rot2', 3, 'T', 4)]
    CASES += [('rot2', 1, 'P', 3)] if quick else []
    WALLS = [(1.0, 1, 'P', 3), (1.0, 1, 'T', 2), (0.35, 2, 'P', 4), (1.0, 0, 'S', 2), (0.35, 1, 'S', 3)] if quick else \
            [(1.0, 1, 'P', 3), (1.0, 1, 'T', 2), (1.0, 0, 'P', 2), (1.0, 0, 'T', 3), (0.35, 2, 'P', 4), (0.35, 3, 'T', 4),
             (1.0, 0, 'S', 2), (0.35, 1, 'S', 3), (1.0, 2, 'S', 4)]
    ICB = [(0.35, 1, 'F', 3), (0.35, 1, 'G', 2)] if quick else \
          [(0.35, 1, 'F', 3), (0.35, 1, 'G', 2), (0.35, 0, 'F', 2), (0.35, 0, 'G', 3), (0.6, 2, 'F', 4), (0.6, 3, 'G', 4)]
    LW = 24
    BLOCKS = {}
    for m in sorted({c[1] for c in CASES}):
        BLOCKS[m] = cf.diffusion_build(m, LW)
    ROTB = {(c[1], int(c[0][3:])): cf.rotation_build(c[1], LW, [int(c[0][3:])]) for c in CASES if c[0].startswith('rot')}
    TASKS = [('B', c) for c in CASES] + [('C', w) for w in WALLS] + [('D', e) for e in ICB]
    TASKS.sort(key=lambda t: 0 if t[1][0] == 'visc' else (1 if t[0] == 'C' else 2))    # the longest first
    try:
        import multiprocessing
        ncpu = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
        with multiprocessing.get_context('fork').Pool(max(1, min(ncpu, 16))) as pool:
            RESULTS = list(pool.imap_unordered(run, TASKS))
    except ValueError:          # no fork on this platform
        RESULTS = list(map(run, TASKS))

    n_ok = 0
    for part, task, out, dt in sorted([x for x in RESULTS if x[0] == 'B'], key=lambda x: str(x[1])):
        kind, m, X, lcol = task
        worst = 0
        scale = max(np.abs(v).max() for v in out.values())
        for (row, lrow), der in sorted(out.items()):
            can = candidate(ROTB[(m, int(kind[3:]))] if kind.startswith('rot') else BLOCKS[m], row, X, lrow, lcol)
            err = np.abs(der - can).max()/scale
            worst = max(worst, err)
            if err > 1e-11:
                bad.append('%s m=%d %s%d row %s%d' % (kind, m, X, lcol, row, lrow))
                print('  FAIL  %s m=%d: %s row l=%d, column %s l=%d: derived %s, kore %s (rel. %.1e)'
                      % (kind, m, row, lrow, X, lcol, np.round(der, 8), np.round(can, 8), err))
            else:
                n_ok += 1
        dl = int(kind[3:]) + 3 if kind.startswith('rot') else 4
        print("  %s  %-4s m=%d column %s l=%d: the rows %s of l' = l-%d..l+%d against kore's blocks, worst %.1e (%.0f s)"
              % ('ok  ' if worst <= 1e-11 else 'FAIL', kind, m, X, lcol, '/'.join(sorted({k[0] for k in out})), dl, dl, worst, dt),
              flush=True)
    print('  %d blocks agree' % n_ok)

    print('C. the wall rows of centrifugal.wall_dissipative against the projected first-order traction and flux', flush=True)
    for part, task, out, dt in sorted([x for x in RESULTS if x[0] == 'C'], key=lambda x: str(x[1])):
        R, m, X, lcol = task
        vals = {a: float(AMPF[a](np.array([R]))[0]) for a in 'ENCQK'}
        vals.update({('lam', k): float(FUNS['lr'][k](np.array([R]))[0]) for k in range(1, 4)})
        W = cf.wall_dissipative(m, LW, R, vals)
        worst = 0
        scale = max(abs(v) for v in out.values())
        for (row, lrow), der in sorted(out.items()):
            can = sum(complex(M[lrow, lcol])*float(FUNS[src[0]][int(src[1])](np.array([R]))[0])
                      for (rw, src), M in W.items() if rw == row and src[0] == X)
            err = abs(der - can)/scale
            worst = max(worst, err)
            if err > 1e-11:
                bad.append('wall R=%g m=%d %s%d row %s%d' % (R, m, X, lcol, row, lrow))
                print('  FAIL  R=%g m=%d: %s row l=%d, column %s l=%d: derived %s, kore %s' % (R, m, row, lrow, X, lcol, der, can))
        print("  %s  R=%g m=%d column %s l=%d: the %s rows of l' = l-4..l+4, worst %.1e (%.0f s)"
              % ('ok  ' if worst <= 1e-11 else 'FAIL', R, m, X, lcol, 'stress-free' if X in 'PT' else 'flux', worst, dt), flush=True)

    print('D. the ICB row of the electric field (centrifugal.icb_electric) against the projected first-order field', flush=True)
    for part, task, out, dt in sorted([x for x in RESULTS if x[0] == 'D'], key=lambda x: str(x[1])):
        R, m, X, lcol = task
        vals = {a: float(AMPF[a](np.array([R]))[0]) for a in 'ENCQK'}
        vals.update({('lam', k): 0. for k in range(1, 4)})
        W = cf.icb_electric(m, LW, R, vals)
        worst = 0
        scale = max(abs(v) for v in out.values()) or 1.   # (zero for m = 0 and F: δ𝐄 is azimuthal, toroidal, there)
        for (row, lrow), der in sorted(out.items()):
            can = sum(complex(M[lrow, lcol])*float(FUNS[X][d](np.array([R]))[0]) for (Xn, d), M in W.items() if Xn == X)
            err = abs(der - can)/scale
            worst = max(worst, err)
            if err > 1e-11:
                bad.append('icb R=%g m=%d %s%d row %d' % (R, m, X, lcol, lrow))
                print('  FAIL  R=%g m=%d: row l=%d, column %s l=%d: derived %s, kore %s' % (R, m, lrow, X, lcol, der, can))
        print("  %s  R=%g m=%d column %s l=%d: the ICB rows of l' = l-4..l+4, worst %.1e (%.0f s)"
              % ('ok  ' if worst <= 1e-11 else 'FAIL', R, m, X, lcol, worst, dt), flush=True)

    print('derive_diff.py: %s (%.0f s)' % ('ok' if not bad else 'FAIL: %s' % bad, time.time() - T0))
    sys.exit(1 if bad else 0)

if __name__ == '__main__':
    main()
