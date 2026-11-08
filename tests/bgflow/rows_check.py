#!/usr/bin/env python3
'''
usage: python3 rows_check.py RUNDIR   (a run folder of kore, for its structure: ρ0, p0 and their logarithmic derivatives)

An independent check of bgflow.build, the angular coefficients and radial monomials of the background's terms: for a
background U (P_n, T_n of degrees 2, 4 and 1, 3), S and C given here as polynomials, and the density of the run, the
value of each row of kore at a few radii and degrees l -> l',
    Σ_labels M[l', l] r^p code^(j)(r) [λ_k | p0 | ρ0] X^(d)(r)
for a perturbation X = x(r) of each kind (P, T of the flow, S, C, F, G of the field), against the same row computed here
on a grid of θ from the vector fields and projected on the degree l':
    u, v: F' = Ω_U × u - U × ω (ω = ∇×u, Ω_U = ∇×U),  rows r⁵ r̂⋅∇×∇×F' and r³ r̂⋅∇×F' (the r-derivative of r s by central
          differences of steps h and h/2, combined by Richardson extrapolation);
    h, c: -r² w (u⋅∇S + U⋅∇s) (w = p0 or ρ0, or 1 where the rows are divided by it: both are checked);
    f, g: E = U × b, rows -r² L' t and -r L'[q - (r s)'] of E (the induction ∇×E in kore's rows and signs),
with (q, s, t) the components of a vector on the degree l' (angular.py) and L' = l'(l' + 1). Prints one JSON line: the
largest difference of each row and column, relative to the largest value, over l, l' and the radii.
'''
import os, sys, json
import numpy as np
os.chdir(sys.argv[1])
sys.path.insert(0, 'bin')
from parameters import par   # noqa: E402
import angular               # noqa: E402
import bgflow as bgf         # noqa: E402
import radial_profiles as rap  # noqa: E402

m = par.m
Pl = np.polynomial.Polynomial
BG = {'P': {2: Pl([0, 0, 0.2, -0.3, 0.1]), 4: Pl([0, 0, 0, 0, 0.05, -0.05])},
      'T': {1: Pl([0, 0, 0, 0.3]), 3: Pl([0, 0, 0, -0.1, 0.12])},
      'S': {0: Pl([0.1, 0, -0.2]), 2: Pl([0, 0, 0.2, 0.1])},
      'C': {2: Pl([0, 0, 0, 0, -0.15])}}
degs = {k: sorted(v) for k, v in BG.items()}
nm = max(n for v in degs.values() for n in v)
lw = max(par.lmax, m + 8) + nm + 8
rho = lambda r: rap.prf.density(np.atleast_1d(r), 0)[0]
p0 = lambda r: rap.prf.pressure(np.atleast_1d(r), 0)[0]
lam = lambda r, k: rap.logrhoX(np.atleast_1d(np.asarray(r, float)), k)[0]
dom = [max(par.ricb, 0.05), 1.0]

poly = Pl([0.1, 0.3, -0.7, 0.4, 0.2])   # the perturbation's radial scalar x(r)
pr = lambda r, k=0: poly.deriv(k)(r) if k else poly(r)
x, w = np.polynomial.legendre.leggauss(160)
sth = np.sqrt(1 - x**2)
Yg, dYg, Zg = angular.ylm_grid(m, lw, x)
Leg = np.polynomial.Legendre
cross = lambda a, b: [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
dot = lambda a, b: a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def check(heatw):
    compw = 'dns0' if heatw else ''
    wS = (lambda r: p0(r)) if heatw else (lambda r: 1.0)
    wC = (lambda r: rho(r)) if compw else (lambda r: 1.0)

    def fn(B, n):   # the radial functions of the profile codes (bgflow.py, section 2), per unit Ro G
        if B == 'q': return lambda r: n*(n + 1)*BG['P'][n](r)/r
        if B == 's': return lambda r: BG['P'][n].deriv()(r) + BG['P'][n](r)/r + lam(r, 1)*BG['P'][n](r)
        if B == 't': return lambda r: BG['T'][n](r)
        if B == 'h': return lambda r: wS(r)*BG['S'][n](r)
        if B == 'd': return lambda r: wS(r)*BG['S'][n].deriv()(r)
        if B == 'c': return lambda r: wC(r)*BG['C'][n](r)
        if B == 'e': return lambda r: wC(r)*BG['C'][n].deriv()(r)
    CH = {}
    def code(B, n, j, r):   # (and their derivatives, from a Chebyshev interpolant)
        if (B, n) not in CH:
            f = fn(B, n)
            CH[(B, n)] = np.polynomial.Chebyshev.interpolate(lambda rr: np.array([f(t) for t in rr]), 60, domain=dom)
        c = CH[(B, n)]
        return (c.deriv(j) if j else c)(r)

    phys = {'hydro': True, 'thermal': True, 'compositional': True, 'magnetic': True, 'heat': heatw, 'comp': compw}
    out = bgf.build(m, degs, lw, phys)

    def label_value(row, X, l, lp, r):
        v = 0
        for lab, M in out.get((row, X), {}).items():
            rx, B, n, j = int(lab[1]), lab[3], bgf.DIGITS.index(lab[4]), int(lab[5])
            rest, d = lab[6:].split('_D')
            ex = {'pss0': p0(r), 'dns0': rho(r), '': 1.0}[rest] if not rest.startswith('lho') else lam(r, int(rest[3:]))
            v += M[lp, l]*r**(bgf.ROWPOW[row] - rx)*code(B, n, j, r)*ex*pr(r, int(d))
        return v

    def bg_fields(r):
        z = 0*x
        U, Wv = [z, z, z], [z, z, z]
        for n in degs['P']:
            Pn, dPn = Leg.basis(n)(x), Leg.basis(n).deriv()(x)
            q, s, ds = code('q', n, 0, r), code('s', n, 0, r), code('s', n, 1, r)
            U = [U[0] + q*Pn, U[1] - s*sth*dPn, U[2]]
            Wv = [Wv[0], Wv[1], Wv[2] + (q/r - ds - s/r)*sth*dPn]
        for n in degs['T']:
            Pn, dPn = Leg.basis(n)(x), Leg.basis(n).deriv()(x)
            t, dt = code('t', n, 0, r), code('t', n, 1, r)
            U = [U[0], U[1], U[2] + t*sth*dPn]
            Wv = [Wv[0] + n*(n + 1)*t/r*Pn, Wv[1] - (dt + t/r)*sth*dPn, Wv[2]]
        grads = {}
        for K, Bv, Bd in (('S', 'h', 'd'), ('C', 'c', 'e')):
            acc = [z, z, z]
            for n in degs[K]:
                Pn, dPn = Leg.basis(n)(x), Leg.basis(n).deriv()(x)
                acc = [acc[0] + code(Bd, n, 0, r)*Pn, acc[1] - code(Bv, n, 0, r)/r*sth*dPn, acc[2]]
            grads[K] = acc
        return U, Wv, grads['S'], grads['C']

    def pert(X, kind, l, r):
        Y, dY, Z = Yg[l], dYg[l], Zg[l]
        L = l*(l + 1.)
        p, dp, d2p = pr(r), pr(r, 1), pr(r, 2)
        l1, l2 = lam(r, 1), lam(r, 2)
        if kind == 'u':
            if X == 'P':
                s = dp + p/r + l1*p
                return [L*p/r*Y, s*dY, s*Z]
            return [0*Y, p*Z, -p*dY]
        if kind == 'w':
            if X == 'P':
                s = dp + p/r + l1*p
                ds = d2p + dp/r - p/r**2 + l2*p + l1*dp
                t = L*p/r**2 - ds - s/r
                return [0*Y, t*Z, -t*dY]
            s = dp + p/r
            return [L*p/r*Y, s*dY, s*Z]
        if kind == 'b':
            if X == 'F':
                s = dp + p/r
                return [L*p/r*Y, s*dY, s*Z]
            return [0*Y, p*Z, -p*dY]
        return [dp*Y, p/r*dY, p/r*Z]   # 'g': the gradient of the scalar x Y

    def proj(F, lp):
        Y, dY, Z = Yg[lp], dYg[lp], Zg[lp]
        n = np.sum(w*Y**2); Lp = lp*(lp + 1.)
        return (np.sum(w*F[0]*Y)/n, np.sum(w*(F[1]*dY + F[2]*np.conj(Z)))/(Lp*n),
                np.sum(w*(F[1]*np.conj(Z) - F[2]*dY))/(Lp*n))

    def vec(row, X, l, r):
        U, Wv, GS, GC = bg_fields(r)
        if row in 'uv':
            return [a - b for a, b in zip(cross(Wv, pert(X, 'u', l, r)), cross(U, pert(X, 'w', l, r)))]
        return cross(U, pert(X, 'b', l, r))

    def drs(row, X, l, lp, r, h):   # (r s)' by fourth-order central differences of step h
        rs = [(r + k*h)*proj(vec(row, X, l, r + k*h), lp)[1] for k in (-2, -1, 1, 2)]
        return (rs[0] - 8*rs[1] + 8*rs[2] - rs[3])/(12*h)

    def exact(row, X, l, lp, r, h=4e-3):
        Lp = lp*(lp + 1.)
        if row in 'hc':
            U, Wv, GS, GC = bg_fields(r)
            Y = Yg[lp]; n = np.sum(w*Y**2)
            if X in 'PT':
                f = dot(pert(X, 'u', l, r), GS if row == 'h' else GC)
            else:
                f = (wS(r) if row == 'h' else wC(r))*dot(U, pert(X, 'g', l, r))
            return -r**2*np.sum(w*f*Y)/n
        q, s, t = proj(vec(row, X, l, r), lp)
        if row in 'vf':
            return (1 if row == 'v' else -1)*r**2*Lp*t
        d = (16*drs(row, X, l, lp, r, h/2) - drs(row, X, l, lp, r, h))/15   # Richardson: O(h^6)
        return r**3*Lp*(q - d) if row == 'u' else -r*Lp*(q - d)

    res = {}
    cases = [('h', 'P'), ('h', 'T'), ('h', 'S'), ('c', 'P'), ('c', 'T'), ('c', 'C')]
    if heatw:   # (the rows u, v, f, g do not depend on the weights: once)
        cases += [('u', 'P'), ('u', 'T'), ('v', 'P'), ('v', 'T'), ('f', 'F'), ('f', 'G'), ('g', 'F'), ('g', 'G')]
    for row, X in cases:
        big, err = 0, 0
        for l in range(max(m, 1), m + 6):
            for lp in range(max(m, 1, l - nm - 1), l + nm + 2):
                for r in (0.45, 0.62, 0.91):
                    e, b = exact(row, X, l, lp, r), label_value(row, X, l, lp, r)
                    big, err = max(big, abs(e)), max(err, abs(e - b))
        res['%s%s%s' % (row, X, '' if row not in 'hc' else ' ' + ({'h': heatw, 'c': compw}[row] or 'w = 1'))] = err/big
    return res

res = check('pss0')
res.update(check(''))
print(json.dumps(res))
