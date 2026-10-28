#!/usr/bin/env python3
'''
derive.py: the magnetic rows of anelastic_log derived from the definitions, independently of kore's operators.

Fields at fixed m (factor e^{imφ} implicit), on kore's Schmidt harmonics Y_l = sqrt(4π/(2l+1)) Y_l^m (Condon-Shortley):
    u  = ρ0⁻¹ ∇×∇×(ρ0 P Y_l 𝐫) + ∇×(T Y_l 𝐫)          (the velocity of anelastic_log)
    b  = ∇×∇×(F Y_l 𝐫) + ∇×(G Y_l 𝐫)
    B0 = ∇×∇×(h Y_1^0 𝐫)                              (degree 1, any h)
and the projections (coefficient of Y_l' of)
    Lorentz, rows u: ρ0 r̂⋅∇×∇×(ρ0⁻¹ 𝐆),  rows v: ρ0 r̂⋅∇×(ρ0⁻¹ 𝐆),   𝐆 = (∇×b)×B0 + (∇×B0)×b
    induction, rows f: r̂⋅∇×(u×B0),          rows g: r̂⋅∇×∇×(u×B0)
Each projection is a sum of monomials (P, P', ..., h, h', ..., λ1 = (ln ρ0)', λ2, ...) times powers of r, with numeric
coefficients: the θ-integrals are done by Gauss-Legendre quadrature in θ (exact to round-off for these trigonometric
polynomials). compare() checks a candidate closed form term by term, up to one overall factor (the row weight r^p/L'^3
and the sign convention of the A matrix), which it fits and returns.
usage (sympy needed; derive_check.py has the checks against the closed forms):
    python3 derive.py ROW COL LCOL LROW M [--rho]
prints the projection of row ROW (u, v: Lorentz force; f, g: induction) on Y_LROW of the column field COL (u, v rows: F or G,
the poloidal or toroidal b; f, g rows: P or T, the poloidal or toroidal u) of degree LCOL, at order M, as a sum of terms
coefficient * r^p * monomial (P1 = P', h2 = h'', lr1 = (ln ρ0)' = λ1, ...); with --rho only the terms with (ln ρ0) derivatives.
Example: python3 derive.py g P 3 3 1 --rho   (the variable-density terms of the g rows, P column, l = 3, m = 1).
Kore's operators are these rows times -r⁵/L³ (u), -r³/L³ (v, f, g), and Hendrik for the Lorentz force (the A sign).
'''
import itertools
import numpy as np
import sympy as sy

r, th = sy.symbols('r theta', positive=True)
FUN = {k: sy.Function(k)(r) for k in ('P', 'T', 'F', 'G', 'h', 'lr')}   # lr = ln ρ0
P, T, F, G, h, lr = (FUN[k] for k in ('P', 'T', 'F', 'G', 'h', 'lr'))


def Y(l, m):
    '''kore's Schmidt harmonic, θ part'''
    return sy.sqrt(4*sy.pi/(2*l+1)) * sy.Ynm(l, m, th, 0).expand(func=True)


# ---------------------------------------------------------------- spherical vector calculus, ∂φ -> i m
def curl(A, m):
    Ar, At, Ap = A
    s = sy.sin(th)
    return ( (sy.diff(s*Ap, th) - sy.I*m*At)/(r*s),
             (sy.I*m*Ar/s - sy.diff(r*Ap, r))/r,
             (sy.diff(r*At, r) - sy.diff(Ar, th))/r )

def cross(A, B):
    return ( A[1]*B[2] - A[2]*B[1], A[2]*B[0] - A[0]*B[2], A[0]*B[1] - A[1]*B[0] )

def div(A, m):
    s = sy.sin(th)
    return sy.diff(r**2*A[0], r)/r**2 + sy.diff(s*A[1], th)/(r*s) + sy.I*m*A[2]/(r*s)

def radial(psi, y):            # psi(r) y(θ) 𝐫
    return (r*psi*y, 0, 0)

def scal(c, A):
    return tuple(c*a for a in A)

def add(A, B):
    return tuple(a + b for a, b in zip(A, B))


def velocity(l, m, part):
    y = Y(l, m)
    if part == 'P':
        rho = sy.exp(lr)
        return scal(1/rho, curl(curl(radial(rho*P, y), m), m))
    return curl(radial(T, y), m)

def bfield(l, m, part):
    y = Y(l, m)
    if part == 'F':
        return curl(curl(radial(F, y), m), m)
    return curl(radial(G, y), m)

def B0():
    return curl(curl(radial(h, sy.cos(th)), 0), 0)


def lorentz(b, m):
    b0 = B0()
    return add(cross(curl(b, m), b0), cross(curl(b0, 0), b))


# ---------------------------------------------------------------- projection onto Y_l' and the monomials
DERIV = 6
def symbols_for():
    '''sympy derivatives -> plain symbols P0, P1, ... (lr1 = λ1, lr2 = λ2, ...)'''
    sub, names = {}, {}
    for k, f in FUN.items():
        for d in range(DERIV, -1, -1):
            s = sy.Symbol('%s%d' % (k, d))
            sub[sy.diff(f, r, d) if d else f] = s
            names[s] = (k, d)
    return sub, names
SUB, NAMES = symbols_for()

xq, wq = np.polynomial.legendre.leggauss(80)
THQ = (xq + 1)*np.pi/2; WQ = wq*np.pi/2   # Gauss-Legendre in θ on [0, π]

def project(expr, lrow, m):
    '''{(monomial, r power): coefficient} of the Y_lrow component of expr(r, θ)'''
    yr = sy.lambdify(th, Y(lrow, m), 'numpy')(THQ)
    norm = 4*np.pi/(2*lrow + 1)
    e = sy.expand(expr.subs(SUB))
    gens = [s for s in NAMES if e.has(s)]
    out = {}
    poly = sy.Poly(e, *gens) if gens else None
    for monom, coeff in (poly.terms() if poly else []):
        if all(d == 0 for d in monom):
            continue
        c = sy.expand(coeff*r**12)            # a polynomial in r
        cp = sy.Poly(c, r)
        for (k,), ck in cp.terms():
            f = sy.lambdify(th, ck, 'numpy')
            vals = np.broadcast_to(np.asarray(f(THQ), dtype=complex), THQ.shape)
            val = 2*np.pi*np.sum(WQ*vals*np.conj(yr)*np.sin(THQ))/norm  # ∫dΩ (φ part: 2π) / ∫|Y|²dΩ
            if abs(val) > 1e-12:
                key = (tuple(sorted((NAMES[g], p) for g, p in zip(gens, monom) if p)), k - 12)
                out[key] = out.get(key, 0) + val
    return out


def rows(kind, part, lcol, lrow, m):
    '''the projected row (dict) of: kind 'lorentz' rows 'u'/'v' with b part 'F'/'G'; kind 'induction' rows 'f'/'g' with u part'''
    if kind in ('u', 'v'):
        Gv = lorentz(bfield(lcol, m, part), m)
        rho = sy.exp(lr)
        Gr = scal(1/rho, Gv)
        e = curl(curl(Gr, m), m)[0] if kind == 'u' else curl(Gr, m)[0]
        return project(rho*e, lrow, m)
    E = cross(velocity(lcol, m, part), B0())
    e = curl(E, m)[0] if kind == 'f' else curl(curl(E, m), m)[0]
    return project(e, lrow, m)


def compare(a, b, label):
    '''a = c*b term by term? prints the fitted c and the worst relative mismatch'''
    keys = set(a) | set(b)
    big = max(abs(v) for v in a.values())
    ratios = [a[k]/b[k] for k in keys if k in a and k in b and abs(a[k]) > 1e-8*big]
    c = ratios[0] if ratios else 0
    err = max(abs(a.get(k, 0) - c*b.get(k, 0)) for k in keys)/big
    print('%-58s factor %-24s mismatch %.1e %s' % (label, np.round(c, 12), err, 'ok' if err < 1e-10 else 'DIFFERENT'))
    return c, err


def as_dict(expr):
    '''a closed form (sympy, in the symbols P0, P1, h0, lr1, ... and r) as the dict of project()'''
    e = sy.expand(expr)
    gens = [s for s in NAMES if e.has(s)]
    out = {}
    for monom, coeff in sy.Poly(e, *gens).terms():
        cp = sy.Poly(sy.expand(coeff*r**12), r)
        for (k,), ck in cp.terms():
            key = (tuple(sorted((NAMES[g], p) for g, p in zip(gens, monom) if p)), k - 12)
            out[key] = out.get(key, 0) + complex(ck)
    return out


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    row, col, lc, lr_, mm = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    out = rows(row, col, lc, lr_, mm)
    if '--rho' in sys.argv:
        out = {k: v for k, v in out.items() if any(n[0] == 'lr' for n, p in k[0])}
    for (mono, rp), c in sorted(out.items(), key=lambda kv: (kv[0][1], str(kv[0][0]))):
        name = ' '.join('%s%d%s' % (n, d, '' if p == 1 else '^%d' % p) for (n, d), p in mono)
        print('%+.12f %+.12fi   r^%-3d %s' % (c.real, c.imag, rp, name))
    print('(%d terms)' % len(out))
