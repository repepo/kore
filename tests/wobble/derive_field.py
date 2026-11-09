#!/usr/bin/env python3
'''
usage: derive_field.py   (sympy and numpy, a few seconds, no run)

The field terms of bin/wobble.py (magnetic = 1, NEW 2026-11-09) on their own, for polynomial fields (random coefficients) in
a spheroid of flattening 0.1 and in a sphere: b̃ = Σ ∇×∇×(f_l Y_l r) + ∇×(g_l Y_l r) (m = 1, kore's Schmidt Y_l times e^{iφ}),
B̃0 = Σ ∇×∇×(h_n P_n r) + ∇×(k_n P_n r), with f_l, g_l ~ r^l and h_n, k_n ~ r^n times polynomials in r², regular at r = 0.
  1. _wall_fields and _lorentz, point by point on the spheres r = 1 and 0.7: b̃, ∇×(Q b̃), B̃0, ∇×(Q B̃0) and the covariant
     Lorentz force ∇×(Q b̃) × B̃0 + ∇×(Q B̃0) × b̃ against sympy in Cartesian coordinates (Q = diag(a², a², c²)), 1e-12;
  2. _maxwell, the Maxwell stress through the wall, against the volume torque of the Lorentz force in the body,
     ∫ X × F dV = ∫ (D α) × (D⁻¹ F̃) dα over the unit ball (the divergence theorem; Gauss-Legendre in r), 1e-11;
  3. _pressure_lorentz's projection: 1 for ∇_H(Y₂¹ e^{iφ}), 0 for ∇_H Y₄¹ and for r̂ × ∇_H Y₂¹.
Exit code 1 if a check fails.
'''
import os, sys
import numpy as np
import sympy as sp

BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bin')
sys.path.insert(0, BIN)
src = open(os.path.join(BIN, 'wobble.py')).read()
ns = {'np': np, '_N1': -1/np.sqrt(2)}
exec(src[src.index('def _field_grid'):src.index('def field_background')], ns)
_field_grid, _wall_fields, _lorentz, _maxwell, _pressure_lorentz = (ns[k] for k in (
    '_field_grid', '_wall_fields', '_lorentz', '_maxwell', '_pressure_lorentz'))

fails = []
def check(name, err, tol):
    ok = err < tol
    print('  %-74s %.1e (%s %.0e)' % (name, err, 'ok, <' if ok else 'FAIL, >=', tol))
    if not ok:
        fails.append(name)

rng = np.random.default_rng(7)
X, Y, Z, rs = sp.symbols('x y z r', real=True)
R2 = X**2 + Y**2 + Z**2

def solid(l, m):
    '''r^l Y_l^m e^{imφ} (kore's Schmidt harmonic, Condon-Shortley phase) as a polynomial in x, y, z; m 0 or 1'''
    t = sp.symbols('t')
    if m == 0:
        p = sp.Poly(sp.legendre(l, t), t)
        pre, deg = 1, l
    else:
        p = sp.Poly(sp.diff(sp.legendre(l, t), t), t)
        pre, deg = -sp.sqrt(sp.factorial(l - 1)/sp.factorial(l + 1))*(X + sp.I*Y), l - 1
    return sp.expand(pre*sum(c*Z**k*R2**((deg - k)//2) for (k,), c in p.terms()))

def curl(v):
    return [sp.diff(v[2], Y) - sp.diff(v[1], Z), sp.diff(v[0], Z) - sp.diff(v[2], X), sp.diff(v[1], X) - sp.diff(v[0], Y)]

def cross(u, v):
    return [u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0]]

def field(pol, tor, m):
    '''Σ ∇×∇×(Φ r) + ∇×(Ψ r) for {degree: coefficients of the polynomial in r² that multiplies r^l}'''
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

def radial(l, cs):
    '''f(r) = r^l Σ c_k r^(2k) as a sympy expression of rs'''
    return rs**l*sum(c*rs**(2*k) for k, c in enumerate(cs))

pol = {l: list(rng.normal(size=2) + 1j*rng.normal(size=2)) for l in (1, 2, 3, 4)}
tor = {l: list(rng.normal(size=2) + 1j*rng.normal(size=2)) for l in (1, 2, 3, 4)}
bpol = {1: [0.5, 0.3], 2: [0.2, -0.4], 3: [0.1, 0.25]}
btor = {2: [0.3, -0.3], 1: [0.2, 0.1]}
b_sym, B_sym = field(pol, tor, 1), field(bpol, btor, 0)
lf = np.array(sorted(pol)); lg = np.array(sorted(tor))
B0_degrees = max(list(bpol) + list(btor))
gr = _field_grid(12, B0_degrees + 1, 40)
x = gr['x']; s = gr['s']

def jets(R):
    Fj = np.array([[complex(sp.diff(radial(l, pol[l]), rs, d).subs(rs, R)) for d in range(3)] for l in lf])
    Gj = np.array([[complex(sp.diff(radial(l, tor[l]), rs, d).subs(rs, R)) for d in range(2)] for l in lg])
    B0 = {'h': {n: tuple(float(sp.diff(radial(n, c), rs, d).subs(rs, R)) for d in range(3)) for n, c in bpol.items()},
          'k': {n: tuple(float(sp.diff(radial(n, c), rs, d).subs(rs, R)) for d in range(2)) for n, c in btor.items()}}
    return Fj, Gj, B0

def spherical(v, R):
    '''(r, θ, φ) components at φ = 0 on the sphere R, at the grid's x = cos θ, of a Cartesian sympy vector'''
    fx = [sp.lambdify((X, Y, Z), c, 'numpy') for c in v]
    px, pz = R*s, R*x
    vx, vy, vz = (np.broadcast_to(f(px, 0*px, pz), px.shape).astype(complex) for f in fx)
    return s*vx + x*vz, x*vx - s*vz, vy

print('1. the fields and the covariant Lorentz force, point by point, against sympy')
for fl_ in (0.1, 0.0):
    a, c = (1 - fl_)**(-1/3), (1 - fl_)**(2/3)
    Q = [a*a, a*a, c*c]
    Jb = curl([q*v for q, v in zip(Q, b_sym)])
    JB = curl([q*v for q, v in zip(Q, B_sym)])
    F_sym = [sp.expand(u + v) for u, v in zip(cross(Jb, B_sym), cross(JB, b_sym))]
    for R in (1.0, 0.7):
        Fj, Gj, B0 = jets(R)
        fl = _wall_fields(gr, R, a, c, lf, Fj, lg, Gj, B0)
        for name, ours, ref in (('b', fl['b'], b_sym), ('J', fl['J'], Jb), ('B0', fl['B'], B_sym), ('J0', fl['J0'], JB),
                                ('F', _lorentz(fl), F_sym)):
            refc = spherical(ref, R)
            scale = max(np.abs(np.array(refc)).max(), 1e-300)
            err = max(np.abs(np.asarray(o) - rc).max() for o, rc in zip(ours, refc))/scale
            check('f = %g, R = %g: %s' % (fl_, R, name), err, 1e-12)

print('2. the Maxwell stress through the wall against the volume torque of the Lorentz force')
for fl_ in (0.1, 0.0):
    a, c = (1 - fl_)**(-1/3), (1 - fl_)**(2/3)
    Fj, Gj, B0 = jets(1.0)
    wall = _maxwell(gr, _wall_fields(gr, 1.0, a, c, lf, Fj, lg, Gj, B0), 1.0, a, c)
    xr, wr = np.polynomial.legendre.leggauss(24)
    vol = 0j
    for R, wR in zip(0.5*(xr + 1), 0.5*wr):
        Fj, Gj, B0 = jets(R)
        Fr, Ft, Fp = _lorentz(_wall_fields(gr, R, a, c, lf, Fj, lg, Gj, B0))
        Fz, Fs = x*Fr - s*Ft, s*Fr + x*Ft
        # (D R r̂) × (D⁻¹ F̃), its x - i y part after the φ integral: 2π i R [(a/c) s F_z - (c/a) x (F_s - i F_φ)]
        vol += wR*R**2*2*np.pi*1j*R*(((a/c)*s*Fz - (c/a)*x*(Fs - 1j*Fp)) @ gr['w'])
    vol /= ns['_N1']
    check('f = %g: |wall - volume|/|volume| (|volume| = %.3g)' % (fl_, abs(vol)), abs(wall - vol)/abs(vol), 1e-11)

print('3. the projection on the gradient of Y_2^1')
# _pressure_lorentz takes the force from _lorentz: fields whose force is the given one (J = r̂, B = (0, Fp, -Ft))
def proj(Ft, Fp):
    z = 0*x
    fl = {'J': (1 + z, z, z), 'B': (z, Fp, -Ft), 'J0': (z, z, z), 'b': (z, z, z)}
    return _pressure_lorentz(gr, fl)
V = gr['V']; dY = gr['dY']
check('∇_H Y₂¹ gives 1', abs(proj(dY[2], 1j*V[2]) - 1), 1e-13)
check('∇_H Y₄¹ gives 0', abs(proj(dY[4], 1j*V[4])), 1e-13)
check('r̂ × ∇_H Y₂¹ gives 0', abs(proj(-1j*V[2], dY[2])), 1e-13)

print('\nderive_field: ' + ('all checks passed' if not fails else 'FAILED: ' + '; '.join(fails)))
sys.exit(1 if fails else 0)
