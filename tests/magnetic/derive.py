#!/usr/bin/env python3
'''
derive.py: the magnetic rows of anelastic_log derived from the definitions, independently of kore's operators.

WHY. Kore never forms the Lorentz force or the induction term on a grid: it assembles, for every pair of degrees
(row l', column l), radial operators (Chebyshev/Gegenbauer matrices) multiplied by coefficients in l and m. Those
coefficients come from projecting vector identities on spherical harmonics, a long and error-prone algebra, more so with a
variable density (the factors ρ0, ρ0⁻¹ produce the (ln ρ0)' terms). This script does that algebra mechanically, from the
definitions, so that every coefficient kore uses can be derived or checked without trusting a hand calculation or another
code. docs/derivations.md (part A) explains how its output becomes kore's closed forms (tools/derive/fit_rows.py).

WHAT. Fields at fixed m (factor e^{imφ} implicit), on kore's Schmidt harmonics Y_l = sqrt(4π/(2l+1)) Y_l^m (Condon-Shortley):
    u  = ρ0⁻¹ ∇×∇×(ρ0 P Y_l 𝐫) + ∇×(T Y_l 𝐫)          (the velocity of anelastic_log)
    b  = ∇×∇×(F Y_l 𝐫) + ∇×(G Y_l 𝐫)
    B0 = ∇×∇×(h Y_n^0 𝐫) or ∇×(k Y_n^0 𝐫)             (poloidal or toroidal background field of degree n, any h, k)
and the projections (coefficient of Y_l' of)
    Lorentz, rows u: ρ0 r̂⋅∇×∇×(ρ0⁻¹ 𝐆),  rows v: ρ0 r̂⋅∇×(ρ0⁻¹ 𝐆),   𝐆 = (∇×b)×B0 + (∇×B0)×b
    induction, rows f: r̂⋅∇×(u×B0),          rows g: r̂⋅∇×∇×(u×B0)

HOW. The radial functions P, T, F, G, h, k and ln ρ0 stay symbolic (sympy functions of r); only θ is made explicit through
Y_l(θ). sympy builds the vectors and their curls exactly; each result is then expanded into monomials such as
r^p · F‴ · h · (ln ρ0)′, and the θ-part of each monomial's coefficient is projected on Y_l' by Gauss-Legendre quadrature
in θ (80 nodes): the integrands are trigonometric polynomials of low degree, so the quadrature is exact up to round-off.
The result is a dictionary {(monomial, power of r): complex coefficient} for one (row, column, l, l', m): numbers, not
formulas in l and m (turning them into formulas is the job of tools/derive/fit_rows.py).

usage (sympy needed; derive_check.py has the checks against the closed forms):
    python3 derive.py ROW COL LCOL LROW M [--rho] [--b0 KIND]
prints the projection of row ROW (u, v: Lorentz force; f, g: induction) on Y_LROW of the column field COL (u, v rows: F or G,
the poloidal or toroidal b; f, g rows: P or T, the poloidal or toroidal u) of degree LCOL, at order M, as a sum of terms
coefficient * r^p * monomial (P1 = P', h2 = h'', lr1 = (ln ρ0)' = λ1, ...); with --rho only the terms with (ln ρ0) derivatives.
Example: python3 derive.py g P 3 3 1 --rho   (the variable-density terms of the g rows, P column, l = 3, m = 1).
--b0 (NEW 2026-10-29) picks the background field: pol1 (default, the dipole ∇×∇×(h Y_1⁰ 𝐫)), pol2, tor1, tor2 (∇×(k Y_n⁰ 𝐫)).
A guide to this and the other derivation tools: docs/derivations.md.
Kore's operators are these rows times -r⁵/L³ (u), -r³/L³ (v, f, g), and Hendrik for the Lorentz force (the A sign).
compare() checks a candidate closed form term by term, up to one overall factor (the row weight r^p/L'^3 and the sign
convention of the A matrix), which it fits and returns.
'''
import itertools
import numpy as np
import sympy as sy

# The independent variables and the unknown radial functions. Everything radial stays symbolic, so a result holds for any
# radial profile; θ is explicit, and φ enters only through ∂φ -> i m.
r, th = sy.symbols('r theta', positive=True)
FUN = {k: sy.Function(k)(r) for k in ('P', 'T', 'F', 'G', 'h', 'lr', 'k')}   # lr = ln ρ0; k: toroidal B0
P, T, F, G, h, lr, kt = (FUN[k] for k in ('P', 'T', 'F', 'G', 'h', 'lr', 'k'))


def Y(l, m):
    '''kore's Schmidt harmonic, θ part: sqrt(4π/(2l+1)) Y_l^m(θ, 0), as an explicit trigonometric expression'''
    return sy.sqrt(4*sy.pi/(2*l+1)) * sy.Ynm(l, m, th, 0).expand(func=True)


# ---------------------------------------------------------------- spherical vector calculus, ∂φ -> i m
# A vector is a tuple of its (r, θ, φ) components. These are the textbook formulas in spherical coordinates, with every
# ∂/∂φ replaced by i m (the background field is axisymmetric, so for B0 they are called with m = 0).

def curl(A, m):
    '''∇×A'''
    Ar, At, Ap = A
    s = sy.sin(th)
    return ( (sy.diff(s*Ap, th) - sy.I*m*At)/(r*s),
             (sy.I*m*Ar/s - sy.diff(r*Ap, r))/r,
             (sy.diff(r*At, r) - sy.diff(Ar, th))/r )

def cross(A, B):
    '''A×B'''
    return ( A[1]*B[2] - A[2]*B[1], A[2]*B[0] - A[0]*B[2], A[0]*B[1] - A[1]*B[0] )

def div(A, m):
    '''∇⋅A'''
    s = sy.sin(th)
    return sy.diff(r**2*A[0], r)/r**2 + sy.diff(s*A[1], th)/(r*s) + sy.I*m*A[2]/(r*s)

def radial(psi, y):
    '''the vector psi(r) y(θ) 𝐫 (𝐫 = r r̂), the potential of the poloidal/toroidal decompositions'''
    return (r*psi*y, 0, 0)

def scal(c, A):
    '''c A for a scalar c'''
    return tuple(c*a for a in A)

def add(A, B):
    '''A + B'''
    return tuple(a + b for a, b in zip(A, B))


# ---------------------------------------------------------------- the fields of one degree
def velocity(l, m, part):
    '''the velocity of degree l made of P (part 'P': ρ0⁻¹∇×∇×(ρ0 P Y 𝐫), so that ∇⋅(ρ0 u) = 0) or of T (∇×(T Y 𝐫))'''
    y = Y(l, m)
    if part == 'P':
        rho = sy.exp(lr)
        return scal(1/rho, curl(curl(radial(rho*P, y), m), m))
    return curl(radial(T, y), m)

def bfield(l, m, part):
    '''the magnetic field of degree l made of F (∇×∇×(F Y 𝐫)) or of G (∇×(G Y 𝐫)); no ρ0: b is solenoidal by itself'''
    y = Y(l, m)
    if part == 'F':
        return curl(curl(radial(F, y), m), m)
    return curl(radial(G, y), m)

B0_KIND = ('pol', 1)   # NEW (2026-10-28): ('pol', n): ∇×∇×(h Y_n⁰ 𝐫); ('tor', n): ∇×(k Y_n⁰ 𝐫); set by rows(..., b0=)

def B0():
    '''the axisymmetric background field selected by B0_KIND'''
    kind, n = B0_KIND
    if kind == 'pol':
        return curl(curl(radial(h, Y(n, 0)), 0), 0)
    return curl(radial(kt, Y(n, 0)), 0)


def lorentz(b, m):
    '''the linearised Lorentz force per unit volume, (∇×b)×B0 + (∇×B0)×b (the magnetic pressure is part of it, not split off)'''
    b0 = B0()
    return add(cross(curl(b, m), b0), cross(curl(b0, 0), b))


# ---------------------------------------------------------------- projection onto Y_l' and the monomials
# sympy keeps derivatives as Derivative(F(r), (r, 3)); to collect terms they are replaced by plain symbols F3, h1, lr2, ...,
# so that an expression becomes a polynomial in those symbols and in r, with coefficients that depend on θ only.
DERIV = 6   # the highest radial derivative that can appear
def symbols_for():
    '''sympy derivatives -> plain symbols P0, P1, ... (lr1 = λ1, lr2 = λ2, ...); also the inverse map symbol -> (name, order)'''
    sub, names = {}, {}
    for k, f in FUN.items():
        for d in range(DERIV, -1, -1):
            s = sy.Symbol('%s%d' % (k, d))
            sub[sy.diff(f, r, d) if d else f] = s
            names[s] = (k, d)
    return sub, names
SUB, NAMES = symbols_for()

# Gauss-Legendre nodes in θ on [0, π]. The θ-integrands are trigonometric polynomials of degree well below 160, which this
# rule integrates exactly up to round-off (the sinθ of dΩ is included explicitly below).
xq, wq = np.polynomial.legendre.leggauss(80)
THQ = (xq + 1)*np.pi/2; WQ = wq*np.pi/2

def project(expr, lrow, m):
    '''
    {(monomial, r power): coefficient} of the Y_lrow component of expr(r, θ): coefficient = ∫ expr Y* dΩ / ∫ |Y|² dΩ.
    A monomial is a tuple of ((name, derivative order), power), e.g. ((('F', 3), 1), (('h', 0), 1)) for F‴ h.
    '''
    yr = sy.lambdify(th, Y(lrow, m), 'numpy')(THQ)
    norm = 4*np.pi/(2*lrow + 1)
    e = sy.expand(expr.subs(SUB))
    gens = [s for s in NAMES if e.has(s)]
    out = {}
    poly = sy.Poly(e, *gens) if gens else None
    for monom, coeff in (poly.terms() if poly else []):     # one monomial in the radial functions at a time
        if all(d == 0 for d in monom):
            continue
        c = sy.expand(coeff*r**12)            # its coefficient is a Laurent polynomial in r: shift it to a polynomial
        cp = sy.Poly(c, r)
        for (k,), ck in cp.terms():           # one power of r at a time: ck depends on θ only
            f = sy.lambdify(th, ck, 'numpy')
            vals = np.broadcast_to(np.asarray(f(THQ), dtype=complex), THQ.shape)
            val = 2*np.pi*np.sum(WQ*vals*np.conj(yr)*np.sin(THQ))/norm  # ∫dΩ (φ part: 2π) / ∫|Y|²dΩ
            mag = 2*np.pi*np.sum(WQ*np.abs(vals*yr)*np.sin(THQ))/norm  # the same of the absolute value: the cancellation scale
            key = (tuple(sorted((NAMES[g], p) for g, p in zip(gens, monom) if p)), k - 12)
            v0, m0 = out.get(key, (0, 0))
            out[key] = (v0 + val, m0 + mag)
    # a projection that is zero in exact arithmetic comes out as round-off relative to the size of what cancelled: drop it
    return {key: v for key, (v, mag) in out.items() if abs(v) > max(1e-11*mag, 1e-13)}


def rows(kind, part, lcol, lrow, m, b0=('pol', 1)):
    '''
    The projected row (dict of project()) of one block: kind 'u'/'v' (Lorentz force) with b part 'F'/'G', or kind 'f'/'g'
    (induction) with u part 'P'/'T'; the column field has degree lcol, the row degree lrow. b0 = ('pol', n) or ('tor', n):
    the background field (poloidal h or toroidal k, degree n).
    '''
    global B0_KIND
    B0_KIND = b0
    if kind in ('u', 'v'):
        # momentum rows: the force per unit mass is 𝐆/ρ0, and kore's rows are ρ0 r̂⋅∇×∇×(.) (u) and ρ0 r̂⋅∇×(.) (v)
        Gv = lorentz(bfield(lcol, m, part), m)
        rho = sy.exp(lr)
        Gr = scal(1/rho, Gv)
        e = curl(curl(Gr, m), m)[0] if kind == 'u' else curl(Gr, m)[0]
        return project(rho*e, lrow, m)
    # induction rows: r̂⋅∇×(u×B0) (f) and r̂⋅∇×∇×(u×B0) (g)
    E = cross(velocity(lcol, m, part), B0())
    e = curl(E, m)[0] if kind == 'f' else curl(curl(E, m), m)[0]
    return project(e, lrow, m)


def compare(a, b, label):
    '''a = c*b term by term? prints the fitted c and the worst relative mismatch (used by derive_check.py)'''
    keys = set(a) | set(b)
    big = max(abs(v) for v in a.values())
    ratios = [a[k]/b[k] for k in keys if k in a and k in b and abs(a[k]) > 1e-8*big]
    c = ratios[0] if ratios else 0
    err = max(abs(a.get(k, 0) - c*b.get(k, 0)) for k in keys)/big
    print('%-58s factor %-24s mismatch %.1e %s' % (label, np.round(c, 12), err, 'ok' if err < 1e-10 else 'DIFFERENT'))
    return c, err


def as_dict(expr):
    '''a closed form (sympy, in the symbols P0, P1, h0, lr1, ... and r) as the dict of project(), for compare()'''
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
    b0 = sys.argv[sys.argv.index('--b0') + 1] if '--b0' in sys.argv else 'pol1'   # NEW (2026-10-29)
    if b0 not in ('pol1', 'pol2', 'tor1', 'tor2'):
        sys.exit('--b0 %r: pol1, pol2, tor1 or tor2' % b0)
    out = rows(row, col, lc, lr_, mm, b0=(b0[:3], int(b0[3])))
    if '--rho' in sys.argv:
        out = {k: v for k, v in out.items() if any(n[0] == 'lr' for n, p in k[0])}
    for (mono, rp), c in sorted(out.items(), key=lambda kv: (kv[0][1], str(kv[0][0]))):
        name = ' '.join('%s%d%s' % (n, d, '' if p == 1 else '^%d' % p) for (n, d), p in mono)
        print('%+.12f %+.12fi   r^%-3d %s' % (c.real, c.imag, rp, name))
    print('(%d terms)' % len(out))
