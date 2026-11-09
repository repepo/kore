#!/usr/bin/env python3
'''
usage: derive.py   (sympy; about a minute)

The wall functionals of bin/wobble.py from their definitions, with sympy derivatives and Gauss quadrature on the sphere, against
the tables _PRESSURE and _VISCOUS there (1e-10). NEW (2026-11-09).

For the flow of one (kind, l) with m = 1 in kore's conventions (u = ∇×∇×(P Y r) + ∇×(T Y r), Y_l the Schmidt harmonic with the
Condon-Shortley phase, times e^{iφ}), with f, f′, f″, f‴ its radial function and derivatives at r = 1:
  _PRESSURE[(kind, l)][name]   (P2, P4, P6, T1, T3, T5; the higher degrees give nothing) the coefficient of Y₂¹ (Schmidt norm 4π/5) in the horizontal divergence on the sphere r = 1 of
                               ∇²w ('lap'), ∂z²w ('dzz'), ẑ (ẑ⋅∇²w) ('zlapz') and ẑ ∂z²w_z ('zdzzz');
  _VISCOUS[(kind, l)][name]    ∮ sin²θ (r̂ × ∂w/∂r)₋ dΩ ('sin2') and ∮ cos²θ (...)₋ dΩ ('cos2'), V₋ = V_x - i V_y.
'''
import os, re
import numpy as np
import sympy as sp

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bin', 'wobble.py')).read()
tables = {'np': np}
exec(src[src.index('_S30 ='):src.index('def axes')], tables)

r, th, ph = sp.symbols('r theta phi', positive=True)
F = sp.Function('F')
x = sp.cos(th)
st, ct, sph, cph = sp.sin(th), sp.cos(th), sp.sin(ph), sp.cos(ph)
J = sp.symbols('f0:6')

def Y(l):
    return sp.sqrt(sp.factorial(l - 1)/sp.factorial(l + 1))*sp.assoc_legendre(l, 1, x)*sp.exp(sp.I*ph)

def field(kind, l):
    f, Yl = F(r), Y(l)
    if kind == 'P':
        return (l*(l + 1)*f/r*Yl, sp.diff(r*f, r)/r*sp.diff(Yl, th), sp.diff(r*f, r)/r*sp.diff(Yl, ph)/st)
    return (0, f*sp.diff(Yl, ph)/st, -f*sp.diff(Yl, th))

def to_cart(v):
    vr, vt, vp = v
    return (vr*st*cph + vt*ct*cph - vp*sph, vr*st*sph + vt*ct*sph + vp*cph, vr*ct - vt*st)

def to_sph(v):
    vx, vy, vz = v
    return (vx*st*cph + vy*st*sph + vz*ct, vx*ct*cph + vy*ct*sph - vz*st, -vx*sph + vy*cph)

dz = lambda g: ct*sp.diff(g, r) - st/r*sp.diff(g, th)
lap = lambda g: sp.diff(r**2*sp.diff(g, r), r)/r**2 + sp.diff(st*sp.diff(g, th), th)/(r**2*st) + sp.diff(g, ph, 2)/(r**2*st**2)

def at_wall(e):
    for k in range(5, 0, -1):
        e = e.subs(sp.Derivative(F(r), (r, k)), J[k])
    return e.subs(F(r), J[0]).subs(r, 1)

xg, wg = np.polynomial.legendre.leggauss(40)
TH, PH = np.meshgrid(np.arccos(xg), 2*np.pi*np.arange(16)/16, indexing='ij')

def sphere_int(e, n):
    out = []
    for k in range(n):
        v = sp.lambdify((th, ph), sp.diff(e, J[k]), 'numpy')(TH, PH)
        out.append(np.sum(wg[:, None]*np.broadcast_to(np.asarray(v, dtype=complex), TH.shape))*2*np.pi/PH.shape[1])
    return np.array(out)

worst = 0.
for (kind, l) in tables['_PRESSURE']:
    wsp = field(kind, l)
    w = to_cart(wsp)
    Lw = [lap(g) for g in w]
    Zw = [dz(dz(g)) for g in w]
    for name, Vc in (('lap', Lw), ('dzz', Zw), ('zlapz', [0, 0, Lw[2]]), ('zdzzz', [0, 0, Zw[2]])):
        Vr, Vt, Vp = to_sph(Vc)
        div = sp.diff(st*Vt, th)/(r*st) + sp.diff(Vp, ph)/(r*st)
        v = sphere_int(at_wall(div*sp.conjugate(Y(2))), 4)/(4*np.pi/5)
        err = np.max(abs(v - np.array(tables['_PRESSURE'][(kind, l)][name], dtype=complex)))
        worst = max(worst, err)
        print('  %s%d %-6s %s   (table off by %.1e)' % (kind, l, name, np.round(v, 10), err), flush=True)
    Gt, Gp = -sp.diff(wsp[2], r), sp.diff(wsp[1], r)          # r̂ × ∂w/∂r
    Gm = (Gt*ct - sp.I*Gp)*sp.exp(-sp.I*ph)                    # its x - i y component
    for name, q in (('sin2', st**2), ('cos2', ct**2)):
        v = sphere_int(at_wall(q*Gm), 3)
        err = np.max(abs(v - np.array(tables['_VISCOUS'][(kind, l)][name], dtype=complex)))
        worst = max(worst, err)
        print('  %s%d %-6s %s   (table off by %.1e)' % (kind, l, name, np.round(v, 10), err), flush=True)
print('derive.py: ' + ('ok' if worst < 1e-10 else 'FAIL') + ' (largest difference %.1e)' % worst)
raise SystemExit(0 if worst < 1e-10 else 1)
