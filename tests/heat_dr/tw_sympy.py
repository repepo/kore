#!/usr/bin/env python3
'''
tw_sympy.py: sympy derivation of the thermal-wind entropy S_tw = sum_n sigma_n(r) P_n(cos theta) from
(1/r) dS_tw/dtheta = (r sin(theta)/(B g)) G^2 d/dz F,  F = (1 + dOmega)^2 - 1 (thermal_wind = 2) or 2 dOmega (thermal_wind = 1),
dOmega = f0(r) + f2(r) P2(cos theta), and check of the closed forms used in bin/radial_profiles.py (sigma2, sigma4, sigma6).
'''
import sympy as sp
r, x, B, G = sp.symbols('r x B G', positive=True)
f0, f2, g = sp.Function('f0')(r), sp.Function('f2')(r), sp.Function('g')(r)
P = [sp.legendre(n, x) for n in range(9)]
dO = f0 + f2*P[2]
# with x = cos(theta): d/dtheta = -sin d/dx, d/dz = x d/dr + (1 - x^2)/r d/dx  =>  dS/dx = -(r^2 G^2/(B g)) [x F_r + (1 - x^2) F_x/r]
def sigmas(F):
    dSdx = -(r**2*G**2/(B*g))*(x*sp.diff(F, r) + (1 - x**2)*sp.diff(F, x)/r)
    S = sp.integrate(sp.expand(dSdx), (x, -1, x))
    return [sp.simplify(sp.Rational(2*n + 1, 2)*sp.integrate(sp.expand(S*P[n]), (x, -1, 1))) for n in range(9)]
fp0, fp2 = sp.diff(f0, r), sp.diff(f2, r)
# closed forms as coded in radial_profiles.sigma2/4/6 (G^2 times eq. 3' of docs/heat-equation.md)
lin = {2: -(2*r**2*G**2/(B*g))*(fp0/3 + fp2/21 + sp.Rational(4, 7)*f2/r),
       4: -(6*r**2*G**2/(35*B*g))*(fp2 - 2*f2/r), 6: 0}
ex = {2: -(2*r**2*G**2/(B*g))*((1 + f0)*(fp0/3 + fp2/21 + sp.Rational(4, 7)*f2/r) + f2*(fp0 + fp2)/21),
      4: -(6*r**2*G**2/(35*B*g))*((1 + f0)*(fp2 - 2*f2/r) + f2*fp0 + sp.Rational(4, 11)*f2*fp2 + sp.Rational(14, 11)*f2**2/r),
      6: -(4*r**2*G**2/(77*B*g))*f2*(fp2 - 2*f2/r)}
nfail = 0
for name, F, ref in (('thermal_wind = 1', 2*dO, lin), ('thermal_wind = 2', (1 + dO)**2 - 1, ex)):
    s = sigmas(F)
    for n in range(1, 9):
        d = sp.simplify(s[n] - ref.get(n, 0))
        ok = (d == 0); nfail += not ok
        print('%-4s %s  sigma%d' % ('ok' if ok else 'FAIL', name, n) + ('' if ok else '  difference: %s' % d))
    # the thermal-wind balance itself, dS/dx + (r^2 G^2/(B g)) [x F_r + (1 - x^2) F_x/r] = 0, with S = sum sigma_n P_n
    S = sum(ref.get(n, 0)*P[n] for n in (2, 4, 6))
    res = sp.simplify(sp.diff(S, x) + (r**2*G**2/(B*g))*(x*sp.diff(F, r) + (1 - x**2)*sp.diff(F, x)/r))
    print('%-4s %s  balance residual with the closed forms: %s' % ('ok' if res == 0 else 'FAIL', name, res)); nfail += (res != 0)
# barotropic check: cylindrical dOmega = eps s^2 = eps r^2 (1 - x^2), f0 = 2 eps r^2/3, f2 = -2 eps r^2/3
eps = sp.symbols('eps')
cyl = {f0: 2*eps*r**2/3, f2: -2*eps*r**2/3}
for n in (2, 4, 6):
    v = sp.simplify(ex[n].subs(cyl).doit())
    print('%-4s cylindrical, exact sigma%d = %s' % ('ok' if v == 0 else 'FAIL', n, v)); nfail += (v != 0)
print('%d failures' % nfail)
