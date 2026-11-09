#!/usr/bin/env python3
'''
usage: derive_spheroid.py   (sympy and numpy, about a minute, no run)

NEW (2026-11-09): the rows of the thin conducting layer on the exact spheroid (mantle = 'TWA', bin/centrifugal.py, section 8:
spheroid_twa, twa_weight) on their own, for polynomial fields b̃ = Σ ∇×∇×(f_l Y_l r) + ∇×(g_l Y_l r) (m = 1 and 2, kore's
Schmidt Y_l, random coefficients, f_l and g_l r^l times polynomials in r²) in spheroids of flattening 0.1 (oblate) and -0.1
(prolate), at R = 1 and 0.8:
  1. the weight V(x) of the sheet (twa_weight) against (a² - c²)/(c² + (a² - c²) x²), 1e-14;
  2. the rows: pointwise on the sphere r = R, the covariant electric field E = Q ∇×(Q b̃) (sympy, Cartesian, Q = diag(a², a²,
     c²)), its radial part replaced by the one that makes the radial current of the layer vanish, (Q⁻¹E)_r = 0, the sheet
     K̃ = c1 R (Q⁻¹E)_H and (Q - I) b̃, projected on r∇Y_l and r∇Y_l × r̂ (Gauss-Legendre in cos θ): the rows
         f:  (l - μ) F - μ R (F' + m_s - k_t),     g:  G + m_t + k_s
     (μ of the insulating spheroid, spheroid_lambda) against kore's: the sphere's thin-wall rows (assemble.bc_b_twa with
     c = 0, mu = 1: (l + 1) F + R F' + c1 R² ∇²_l F and G + c1 (G + R G')) plus spheroid_twa's parts on F, F', F'', G, G',
     1e-11 of the largest row. The construction of K̃ here (a 3×3 solve per point) is independent of the formula of
     centrifugal.py (the weight V); with c1 = 0 the rows are those of the insulating spheroid (spheroid_field_walls).
Exit code 1 if a check fails.
'''
import os, sys, types
import numpy as np
import sympy as sp

BIN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bin'))
sys.path.insert(0, BIN)
sys.modules.setdefault('parameters', types.SimpleNamespace(par=types.SimpleNamespace(lmax=40, m=1)))
import centrifugal as cf
import angular

fails = []
def check(name, err, tol):
    ok = err < tol
    print('  %-72s %.1e (%s %.0e)' % (name, err, 'ok, <' if ok else 'FAIL, >=', tol), flush=True)
    if not ok:
        fails.append(name)

rng = np.random.default_rng(11)
X, Y, Z, rs = sp.symbols('x y z r', real=True)
R2 = X**2 + Y**2 + Z**2

def solid(l, m):
    '''r^l Y_l e^{imφ} (kore's Schmidt harmonic, Condon-Shortley phase) as a polynomial in x, y, z'''
    t = sp.symbols('t')
    p = sp.Poly(sp.diff(sp.legendre(l, t), t, m), t)
    pre = (-1)**m*sp.sqrt(sp.factorial(l - m)/sp.factorial(l + m))*(X + sp.I*Y)**m
    return sp.expand(pre*sum(c*Z**k*R2**((l - m - k)//2) for (k,), c in p.terms()))

def curl(v):
    return [sp.diff(v[2], Y) - sp.diff(v[1], Z), sp.diff(v[0], Z) - sp.diff(v[2], X), sp.diff(v[1], X) - sp.diff(v[0], Y)]

def radial(l, cs):
    return rs**l*sum(c*rs**(2*k) for k, c in enumerate(cs))

def field(pol, tor, m):
    '''Σ ∇×∇×(f_l Y_l r) + ∇×(g_l Y_l r), {l: coefficients of the polynomial in r² times r^l}'''
    rv = [X, Y, Z]
    out = [0, 0, 0]
    for terms, twice in ((pol, True), (tor, False)):
        for l, cs in terms.items():
            phi = sum(c*R2**k for k, c in enumerate(cs))*solid(l, m)
            v = curl([phi*x for x in rv])
            if twice:
                v = curl(v)
            out = [o + w for o, w in zip(out, v)]
    return [sp.expand(o) for o in out]

def spherical(v, R, x, s):
    '''(r, θ, φ) components at φ = 0 on the sphere R, at x = cos θ, of a Cartesian sympy vector'''
    fx = [sp.lambdify((X, Y, Z), c, 'numpy') for c in v]
    px, pz = R*s, R*x
    vx, vy, vz = (np.broadcast_to(f(px, 0*px, pz), px.shape).astype(complex) for f in fx)
    return np.array([s*vx + x*vz, x*vx - s*vz, vy])

print('1. the weight of the sheet')
xx = np.linspace(-1, 1, 2001)
for f in (0.1, -0.1, 1/300):
    a, c = cf.spheroid_axes(f)
    V = cf.twa_weight(f)
    ref = (a*a - c*c)/(c*c + (a*a - c*c)*xx**2)
    check('f = %.4g: V(x), degree %d' % (f, V.degree()), np.abs(V(xx) - ref).max()/np.abs(ref).max(), 1e-14)

print('2. the rows of the thin layer against the sheet built pointwise')
LW, LT = 70, 40                       # the algebra's degrees, and the rows compared (l <= LT)
xg, wg = np.polynomial.legendre.leggauss(120)
sg = np.sqrt(1 - xg**2)
c1 = 0.37
for m in (1, 2):
    lf = list(range(m, m + 4))
    pol = {l: list(rng.normal(size=2) + 1j*rng.normal(size=2)) for l in lf}
    tor = {l: list(rng.normal(size=2) + 1j*rng.normal(size=2)) for l in lf}
    b_sym = field(pol, tor, m)
    Yg, dY, Zg = angular.ylm_grid(m, LW, xg)
    lv = np.arange(LW + 1)
    nl, Lv = 4*np.pi/(2*lv + 1), lv*(lv + 1.)
    def proj(Vt, Vp):
        '''s_l, t_l of the tangential field (Vt, Vp) e^{imφ}: on r∇Y_l = (∂θY, Z) and r∇Y_l × r̂ = (Z, -∂θY)'''
        s_ = 2*np.pi*((Vt[None, :]*dY + Vp[None, :]*np.conj(Zg)) @ wg)
        t_ = 2*np.pi*((Vt[None, :]*np.conj(Zg) - Vp[None, :]*dY) @ wg)
        with np.errstate(all='ignore'):
            return np.where(Lv > 0, s_/(Lv*nl), 0), np.where(Lv > 0, t_/(Lv*nl), 0)
    for f in (0.1, -0.1):
        a, c = cf.spheroid_axes(f)
        Qd = [a*a, a*a, c*c]
        E_sym = [q*e for q, e in zip(Qd, curl([q*v for q, v in zip(Qd, b_sym)]))]       # Q ∇×(Q b̃)
        dQb = [(q - 1)*v for q, v in zip(Qd, b_sym)]
        mu = cf.spheroid_lambda('outer', m, LW, f)/(lv + 1)
        for R in (1.0, 0.8):
            # F, F', F'' and G, G' of each degree at R
            jet = {}
            for l in lv:
                jet[('F', l)] = [complex(sp.diff(radial(l, pol[l]), rs, d).subs(rs, R)) if l in pol else 0j for d in range(3)]
                jet[('G', l)] = [complex(sp.diff(radial(l, tor[l]), rs, d).subs(rs, R)) if l in tor else 0j for d in range(2)]
            # the sheet, pointwise: E with the radial part that makes (Q⁻¹E)_r = 0
            Er, Et, Ep = spherical(E_sym, R, xg, sg)
            # Cartesian unit vectors at φ = 0: r̂ = (s, 0, x), θ̂ = (x, 0, -s), φ̂ = (0, 1, 0)
            K = np.zeros((3, xg.size), dtype=complex)
            for i in range(xg.size):
                rh, th = np.array([sg[i], 0, xg[i]]), np.array([xg[i], 0, -sg[i]])
                ph = np.array([0, 1., 0])
                Pinv = np.diag([1/a**2, 1/a**2, 1/c**2])
                EH = Et[i]*th + Ep[i]*ph
                er = -(rh @ Pinv @ EH)/(rh @ Pinv @ rh)                  # (Q⁻¹(E_H + er r̂))_r = 0
                Jv = c1*R*(Pinv @ (EH + er*rh))
                K[:, i] = [Jv @ rh, Jv @ th, Jv @ ph]
            check('m %d, f %+.1f, R %.1f: the sheet has no radial part' % (m, f, R),
                  np.abs(K[0]).max()/np.abs(K).max(), 1e-13)
            ks, kt = proj(K[1], K[2])
            _, mt_ = proj(*spherical(dQb, R, xg, sg)[1:])
            ms, _ = proj(*spherical(dQb, R, xg, sg)[1:])
            F = np.array([jet[('F', l)] for l in lv]); G = np.array([jet[('G', l)] for l in lv])
            rowf = (lv - mu)*F[:, 0] - mu*R*(F[:, 1] + ms - kt)
            rowg = G[:, 0] + mt_ + ks
            # kore: the sphere's thin-wall rows plus spheroid_twa
            W = cf.spheroid_twa(m, LW, R, c1, f)
            lap = F[:, 2] + 2*F[:, 1]/R - Lv*F[:, 0]/R**2
            kf = (lv + 1)*F[:, 0] + R*F[:, 1] + c1*R**2*lap
            kg = G[:, 0] + c1*(G[:, 0] + R*G[:, 1])
            src = {'F0': F[:, 0], 'F1': F[:, 1], 'F2': F[:, 2], 'G0': G[:, 0], 'G1': G[:, 1]}
            for (row, sname), M in W.items():
                if row == 'f':
                    kf = kf + M @ src[sname]
                else:
                    kg = kg + M @ src[sname]
            sel = (lv >= m) & (lv <= LT)
            scale = max(np.abs(rowf[sel]).max(), np.abs(rowg[sel]).max())
            err = max(np.abs(kf - rowf)[sel].max(), np.abs(kg - rowg)[sel].max())/scale
            check('m %d, f %+.1f, R %.1f: kore\'s rows f, g on the field (l <= %d)' % (m, f, R, LT), err, 1e-11)
            # c1 = 0: the insulating spheroid's rows
            W0 = cf.spheroid_twa(m, LW, R, 0.0, f)
            Wi = cf.spheroid_field_walls('outer', m, LW, R, f)
            d0 = max((abs(W0[k] - Wi[k]).max() if k in Wi else abs(W0[k]).max()) for k in W0)
            check('m %d, f %+.1f, R %.1f: c1 = 0 gives the insulating spheroid\'s rows' % (m, f, R), d0, 1e-14)

print('\nderive_spheroid: ' + ('all checks passed' if not fails else 'FAILED: ' + '; '.join(fails)))
sys.exit(1 if fails else 0)
