#!/usr/bin/env python3
'''
derive.py: the metric terms of bin/centrifugal.py derived with sympy from their definitions (NEW 2026-11-05).

The map of the reference space to the body is X = α + ξ(α), ξ = ε(r) ∇(r² P2(cos θ)/2), with r = |α|; the field of the
reference space is w = J F⁻¹ u (F = I + ∇ξ, J = det F). From these definitions alone (Cartesian components, ε an arbitrary
function of r, first order in a bookkeeping factor δ: ε -> δ ε) the script checks, and prints, the closed forms that
centrifugal.py implements (in its primitives, entropy_block, g_monopole and to_physical):

  1. the level surfaces: |X| = r (1 + ε P2) + O(ε²), so f = -(3/2) ε, ε = -(2/3) f
  2. the identities that hold for any map (no expansion): F w/J has the divergence ∇_X⋅(ρ0 F w/J) = ∇_α⋅(ρ0 w)/J
     (mass conservation is ∇⋅(ρ0 w) = 0), the covariant Coriolis force Fᵀ(ẑ × F w)/J = (F⁻¹ẑ) × w, the advection
     (F w/J)⋅∇_X S0 = w_r S0'(r)/J, and the covariant buoyancy Fᵀ∇_X Φ(r) = Φ'(r) r̂
  3. the first-order terms, with η = r ε', x = cos θ, w_z = ẑ⋅w, w_r = r̂⋅w:
       M = FᵀF/J:  (M - I) w   = ε (3 w_z ẑ - w) + η [ (3/2) x (w_z r̂ + w_r ẑ) - w_r r̂ - P2 w ]
       z̃ = F⁻¹ẑ:  (z̃ - ẑ) × w = -ε ẑ×w - η [ (3/2) x² ẑ×w - (x/2) r̂×w ]
       J - 1 = η P2
  4. the map of to_physical: X = (s (1 - ε/2), z (1 + ε)) in cylindrical components, and
       F w/J = w + ε ((3/2) w_z ẑ - w/2) + η w_r ((3/2) x ẑ - r̂/2) - η P2 w
  5. the effective potential on the level surfaces: with Φ = Φ0(|X|) + Φ2(|X|) P2 - Ω² |X|² (1 - P2)/3 (gravity and
     centrifugal potential, first order) its P2 part on |X| = r (1 + ε P2) vanishes when Φ0'(r) r ε + Φ2(r) + Ω² r²/3 = 0
     (Clairaut's condition, which fixes ε), and what is left is Φ0(r) - Ω² r²/3: the buoyancy has g - (2/3) Ω² r.

Usage: python3 derive.py   (a few seconds; prints each closed form and 'derive.py: ok', or stops at the first one that fails)
'''
import sys
import sympy as sp

x, y, z, d = sp.symbols('x y z delta', real=True)
R = sp.sqrt(x**2 + y**2 + z**2)
e = sp.Function('e')                                   # ε(r)
w = sp.Matrix(sp.symbols('w1 w2 w3'))                  # a field of the reference space at α = (x, y, z)
X0 = sp.Matrix([x, y, z])
zh = sp.Matrix([0, 0, 1])
rh = X0/R
cth = z/R
P2 = (3*cth**2 - 1)/2
er = e(R)
eta = R*sp.diff(er, x)*R/x                             # r ε'(r) (∂ₓε = ε' x/r)
eta = sp.simplify(eta)

def first_order(expr):
    '''the coefficient of δ in expr (a matrix or a scalar)'''
    f = lambda v: sp.diff(v, d).subs(d, 0)
    return expr.applyfunc(f) if isinstance(expr, sp.MatrixBase) else f(expr)

def zero(expr, what):
    expr = sp.simplify(sp.expand(expr)) if not isinstance(expr, sp.MatrixBase) else expr.applyfunc(lambda v: sp.simplify(sp.expand(v)))
    ok = (expr == 0) if not isinstance(expr, sp.MatrixBase) else all(v == 0 for v in expr)
    print(('  ok    ' if ok else '  FAIL  ') + what)
    if not ok:
        print('        residual:', expr)
        sys.exit('derive.py: FAIL')

Phi2 = (z**2 - (x**2 + y**2)/2)/2                      # r² P2/2
xi = d*er*sp.Matrix([sp.diff(Phi2, v) for v in (x, y, z)])
X = X0 + xi
F = X.jacobian([x, y, z])
G1 = first_order(F)                                    # F = I + δ G1 exactly (ξ is linear in δ)
J = F.det()
Finv = sp.eye(3) - d*G1                                # F⁻¹ at first order in δ (enough for every first-order term)

print('1. the level surfaces')
zero(first_order(sp.sqrt((X.T*X)[0])) - R*er*P2, '|X| = r (1 + ε P2) at first order')

print('2. identities for any map (exact)')
A = sp.Matrix(3, 3, lambda i, j: sp.Symbol('a%d%d' % (i, j)))
zero(A.T*(zh.cross(A*w))/A.det() - (A.inv()*zh).cross(w), 'Fᵀ(ẑ × F w)/J = (F⁻¹ẑ) × w (any invertible F)')
gS = sp.Matrix(sp.symbols('g1 g2 g3'))               # ∇_α S0 (S0 a function of the reference point)
zero(((A*w/A.det()).T*(A.inv().T*gS))[0] - (w.T*gS)[0]/A.det(), '(F w/J)⋅∇_X S0 = w⋅∇_α S0/J, so w_r S0\'/J for S0(r)')
zero(A.T*(A.inv().T*gS) - gS, 'Fᵀ∇_X Φ = ∇_α Φ, so Φ\'(r) r̂ for Φ(r): the buoyancy keeps its form')
# divergence: for the map X(α), ∇_X⋅(F w/J) = (∇_α⋅w)/J (Piola identity); checked on the first-order map with a generic w(α)
W = sp.Matrix([sp.Function('W%d' % i)(x, y, z) for i in range(3)])
U = F*W/J                                              # u as a function of α
divX = sum(sum(sp.diff(U[i], a)*Finv[a_i, i] for a_i, a in enumerate((x, y, z))) for i in range(3))   # ∂u_i/∂X_i
divA = sum(sp.diff(W[i], v) for i, v in enumerate((x, y, z)))
zero(first_order(divX - divA/J), '∇_X⋅(F w/J) = ∇_α⋅w/J (Piola), first order (with ρ0(r): ∇_X⋅(ρ0 u) = ∇_α⋅(ρ0 w)/J)')

print('3. the first-order metric terms')
wz = (zh.T*w)[0]; wr = (rh.T*w)[0]
M1 = first_order(F.T*F/J)
cM = er*(3*wz*zh - w) + eta*(sp.Rational(3, 2)*cth*(wz*rh + wr*zh) - wr*rh - P2*w)
zero(M1*w - cM, '(M - I) w = ε (3 w_z ẑ - w) + η [(3/2) x (w_z r̂ + w_r ẑ) - w_r r̂ - P2 w]')
zt1 = first_order(Finv*zh)
cC = -er*zh.cross(w) - eta*(sp.Rational(3, 2)*cth**2*zh.cross(w) - cth/2*rh.cross(w))
zero(zt1.cross(w) - cC, '(z̃ - ẑ) × w = -ε ẑ×w - η [(3/2) x² ẑ×w - (x/2) r̂×w]')
zero(first_order(J) - eta*P2, 'J - 1 = η P2')

print('4. the map of to_physical')
zero(first_order(X) - er*sp.Matrix([-x/2, -y/2, z]), 'X = α + ε (-x/2, -y/2, z): s (1 - ε/2), z (1 + ε)')
cu = er*(sp.Rational(3, 2)*wz*zh - w/2) + eta*wr*(sp.Rational(3, 2)*cth*zh - rh/2) - eta*P2*w
zero(first_order(F*w/J) - cu, 'F w/J = w + ε ((3/2) w_z ẑ - w/2) + η w_r ((3/2) x ẑ - r̂/2) - η P2 w')

print('5. the effective potential on the level surfaces')
r_, Om, eps = sp.symbols('r Omega epsilon', positive=True)
mu = sp.Symbol('mu')                                   # cos θ
P2m = (3*mu**2 - 1)/2
Phi0, Phi2f = sp.Function('Phi0'), sp.Function('Phi2')
Xr = r_*(1 + d*eps*P2m)                                # |X| on the level surface r
Phi = Phi0(Xr) + d*Phi2f(Xr)*P2m - Om**2*Xr**2*(1 - P2m)/3   # Φ2 and the Ω² terms are first order (δ)
Phi = Phi.subs(Om, sp.sqrt(d)*Om)
P1 = sp.expand(first_order(Phi))
cond = sp.expand(sp.diff(Phi0(r_), r_)*r_*eps + Phi2f(r_) + Om**2*r_**2/3)
zero(sp.expand(P1 - (cond*P2m - Om**2*r_**2/3)), 'P2 part = [Φ0\' r ε + Φ2 + Ω² r²/3] P2, the rest -Ω² r²/3: g_eff = g - (2/3) Ω² r')

print('derive.py: ok (every closed form of bin/centrifugal.py follows from X = α + ξ, w = J F⁻¹ u at first order)')
