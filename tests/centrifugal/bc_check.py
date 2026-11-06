#!/usr/bin/env python3
'''
bc_check.py: the wall rows of an insulating wall on the deformed body (centrifugal.wall_coefficients) against the exact
conditions on the deformed wall (NEW 2026-11-05; numpy and scipy only, no run directory).

WHY. Outside an insulating wall the field is a potential field ∇φ in the body, and the wall is the level surface r = R of the
map X = α + ξ(α). The rows kore uses are the first-order form of the matching (centrifugal.py, section 5); an error in them
would leave a residual of order f in the eigenvalues that no relabelling test sees (the relabelled sphere keeps its walls).
HOW. For a potential field with random coefficients a_l (outside the CMB: r^-(l+1) Y_l; inside an insulating inner core:
r^l Y_l) the field b̃ of the reference space at the wall is computed exactly, pointwise on the sphere |α| = R (Gauss-Legendre
in cos θ, the plane φ = 0, the map's F = I + ∇ξ exact): its flux b̃_r = [J F⁻¹ ∇φ(X)]_r, and its tangential part from
[M b̃]_tan = [Fᵀ ∇φ(X)]_tan (M = FᵀF/J). Its l-components give F_l(R), F_l'(R), G_l(R), which must satisfy the rows:
    outer: (l+1) F + R F' + (metric part) = 0,   G + (metric part) = 0;   inner: l F - R F' + (...) = 0,   G + (...) = 0.
The residuals must be of second order in ε (they fall by 4 when ε is halved), while the rows of the sphere alone leave
residuals of first order (they fall by 2). The map: ε(r) = ε0 (1 + κ (r² - R²)), so η = r ε' = 2 κ ε0 R² at the wall.
usage: python3 bc_check.py   (a few seconds; prints the residuals and 'bc_check.py: ok' or the failures)
'''
import sys, os, types
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', 'bin')))
sys.modules.setdefault('parameters', types.SimpleNamespace(par=types.SimpleNamespace(lmax=40, m=0)))
from scipy.special import lpmv
from math import factorial
import centrifugal as cf

LW = 40          # degrees kept in the projections and the rows
LPHI = 8         # the potential field has the degrees m..LPHI
xq, wq = np.polynomial.legendre.leggauss(160)
th = np.arccos(xq); sn = np.sqrt(1 - xq**2)

def Y(l, m, x):
    '''kore's Schmidt harmonic (θ part, Condon-Shortley phase) and its θ-derivative'''
    if l < m:
        return 0*x, 0*x
    c = np.sqrt(factorial(l - m)/factorial(l + m))
    y = c*lpmv(m, l, x)
    # ∂θ P_l^m = -sinθ dP/dx; dP/dx from (x² - 1) dP_l^m/dx = l x P_l^m - (l + m) P_(l-1)^m
    dPdx = (l*x*lpmv(m, l, x) - (l + m)*lpmv(m, l - 1, x))/(x*x - 1)
    return y, -np.sqrt(1 - x*x)*c*dPdx

def components(m, wall, a, R, e0, kap):
    '''the l-components (q, s, t) at r = R of the field b̃ of the reference space that matches the potential field with the
    coefficients a (dict l: a_l), on the map ε(r) = e0 (1 + kap (r² - R²)); also the values F, F', G'''
    eps, deps = e0, 2*kap*e0*R                       # ε(R), ε'(R)
    # the point α = R (sinθ, 0, cosθ) and the Cartesian basis of its spherical one (φ = 0)
    al = R*np.stack([sn, 0*sn, xq], 1)
    rh = np.stack([sn, 0*sn, xq], 1); thh = np.stack([xq, 0*sn, -sn], 1); ph = np.tile([0., 1., 0.], (len(xq), 1))
    gf = np.stack([-al[:, 0]/2, -al[:, 1]/2, al[:, 2]], 1)            # ∇(r² P2/2) = (-x/2, -y/2, z)
    X = al + eps*gf                                                   # ξ = ε ∇(r² P2/2)
    Fm = np.eye(3)[None] + eps*np.diag([-0.5, -0.5, 1.])[None] + (deps/R)*gf[:, :, None]*al[:, None, :]
    # ∇φ at X, from the spherical coordinates of X (same azimuth φ = 0: the map is axisymmetric)
    rX = np.linalg.norm(X, axis=1); cX = X[:, 2]/rX; sX = np.sqrt(1 - cX**2)
    rhX = np.stack([sX, 0*sX, cX], 1); thX = np.stack([cX, 0*sX, -sX], 1)
    g = np.zeros((len(xq), 3), dtype=complex)
    for l, al_ in a.items():
        y, dy = Y(l, m, cX)
        if wall == 'outer':
            gr, gt, gp = -(l + 1)*al_*rX**(-l - 2)*y, al_*rX**(-l - 2)*dy, al_*rX**(-l - 2)*1j*m*y/sX
        else:
            gr, gt, gp = l*al_*rX**(l - 1)*y, al_*rX**(l - 1)*dy, al_*rX**(l - 1)*1j*m*y/sX
        g += gr[:, None]*rhX + gt[:, None]*thX + gp[:, None]*ph
    J = np.linalg.det(Fm)
    Fi = np.linalg.inv(Fm)
    flux = J[:, None]*np.einsum('nij,nj->ni', Fi, g)                  # J F⁻¹ ∇φ(X)
    cov = np.einsum('nji,nj->ni', Fm, g)                              # Fᵀ ∇φ(X)
    M = np.einsum('nki,nkj->nij', Fm, Fm)/J[:, None, None]
    br = np.einsum('ni,ni->n', flux, rh)                              # b̃_r
    # tangential b̃ from [M b̃]_tan = cov_tan with b̃ = br r̂ + bt θ̂ + bp φ̂: a 2×2 system at each point
    E = np.stack([thh, ph], 2)                                        # tangential basis, (n, 3, 2)
    Mtt = np.einsum('nki,nkl,nlj->nij', E, M, E)
    rhs = np.einsum('nki,nk->ni', E, cov) - br[:, None]*np.einsum('nki,nkl,nl->ni', E, M, rh)
    btan = np.linalg.solve(Mtt, rhs[:, :, None])[:, :, 0]
    bt, bp = btan[:, 0], btan[:, 1]
    # l-components: q_l = ∫ b_r Y* / n_l; s_l, t_l from b_θ = s ∂θY + t (i m Y/sinθ), b_φ = s (i m Y/sinθ) - t ∂θY
    out = {}
    for l in range(max(m, 1), LW + 1):
        y, dy = Y(l, m, xq)
        nl, L = 4*np.pi/(2*l + 1), l*(l + 1.)
        z = 1j*m*y/sn
        q = 2*np.pi*np.sum(wq*br*y)/nl
        s = 2*np.pi*np.sum(wq*(bt*dy + bp*np.conj(z)))/(nl*L)
        t = 2*np.pi*np.sum(wq*(bt*np.conj(z) - bp*dy))/(nl*L)
        Fl = R*q/L
        out[l] = (Fl, s - Fl/R, t)                                     # F, F', G
    return out, eps, 2*kap*e0*R**2

def residuals(m, wall, a, R, e0, kap, metric=True):
    '''max over l <= LPHI + 2 of the f and g rows, relative to |(l+1) F| + |R F'| (outer) or |l F| + |R F'| (inner)'''
    vals, eps, eta = components(m, wall, a, R, e0, kap)
    W = cf.wall_coefficients(wall, m, LW + 8, eps, eta, R)
    W = {k: v.toarray() for k, v in W.items()}
    rf = rg = 0
    for l in range(max(m, 1), LPHI + 3):
        F, F1, G = vals[l]
        rowf = (l + 1)*F + R*F1 if wall == 'outer' else l*F - R*F1
        scale = abs((l + 1)*F) + abs(R*F1) if wall == 'outer' else abs(l*F) + abs(R*F1)
        rowg = G
        if metric:
            for k, (Fk, F1k, Gk) in vals.items():
                rowf += W[('f', 'F0')][l, k]*Fk + W[('f', 'F1')][l, k]*F1k + W[('f', 'G0')][l, k]*Gk
                rowg += W[('g', 'F0')][l, k]*Fk + W[('g', 'F1')][l, k]*F1k + W[('g', 'G0')][l, k]*Gk
        norm = max(abs(vals[k][0]) + abs(R*vals[k][1]) for k in vals)
        rf = max(rf, abs(rowf)/norm)
        rg = max(rg, abs(rowg)/norm)
    return rf, rg

rng = np.random.default_rng(3)
bad = []
for wall, R in (('outer', 1.0), ('inner', 0.35)):
    for m in (0, 1, 2, 5):
        a = {l: complex(*rng.normal(size=2)) for l in range(max(m, 1), LPHI + 1)}
        e0, kap = -3e-3, 0.8
        r1 = residuals(m, wall, a, R, e0, kap)
        r2 = residuals(m, wall, a, R, e0/2, kap)
        s1 = residuals(m, wall, a, R, e0, kap, metric=False)
        s2 = residuals(m, wall, a, R, e0/2, kap, metric=False)
        order = lambda x1, x2: np.log2(x1/x2) if x2 > 1e-14 else np.nan
        o_f, o_g, so_f, so_g = order(r1[0], r2[0]), order(r1[1], r2[1]), order(s1[0], s2[0]), order(s1[1], s2[1])
        # second order with the metric parts (g rows: also exactly 0 at m = 0, no toroidal part at all), first order without,
        # and well below the sphere's residual
        ok = (o_f > 1.9 and (o_g > 1.9 or r1[1] < 1e-13) and abs(so_f - 1) < 0.1 and (abs(so_g - 1) < 0.1 or s1[1] < 1e-13)
              and r1[0] < 0.05*s1[0] and r1[1] <= 0.05*s1[1])
        print('%s  %-5s m=%d  ε0 = %.0e: f rows %.2e (order %.2f), g rows %.2e (order %.2f);  the sphere\'s rows: %.2e (order %.2f), %.2e (order %.2f)'
              % ('ok  ' if ok else 'FAIL', wall, m, e0, r1[0], o_f, r1[1], o_g, s1[0], so_f, s1[1], so_g))
        if not ok:
            bad.append('%s m=%d' % (wall, m))
print('bc_check.py: %s' % ('ok' if not bad else 'FAIL: %s' % bad))
sys.exit(1 if bad else 0)
