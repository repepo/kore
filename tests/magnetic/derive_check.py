#!/usr/bin/env python3
'''
derive_check.py: derive.py's rows against closed forms (needs sympy; about 3 minutes).
  1. uniform density: conductive_IC's operators (39487bbf, operators.py: induction, lorentz, dipolar B0), up to one factor
     (-1, the A-matrix sign; every row of kore is the conductive_IC row over L³ besides);
  2. the (ln ρ0)' terms: the closed forms implemented in bin/operators.py (rows f: d/dr -> d/dr + λ1 on P; rows g: the
     diagonal term 2im[r h λ1 P' + r h λ2 P + (1 - L/2) r h' λ1 P - (L/2) h λ1 P]; rows u: λ1 r⁴ (∇₁⋅𝐆_H)_l, operators.lorentz_rho);
  3. no (ln ρ0)' terms in the rows v and in the T columns;
  4. the toroidal B0 of degrees 1 and 2 (NEW 2026-10-28): derive.rows(b0=('tor', n)), any ρ0, against operators.induction_tor,
     lorentz_tor (n = 1) and induction_tor2, lorentz_tor2 (n = 2) (their rows times -L³/r^p; Hendrik apart), and no couplings beyond those.
Degrees l = 3..6, m = 0..3 (about 40 minutes on one core). Exit 1 on a mismatch.
'''
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import derive as d, sympy as sy
from derive import r
BAD = []
def compare(a, b, label):
    c, err = d.compare(a, b, label)
    if err > 1e-10 or abs(c + 1) > 1e-10: BAD.append(label)

# ------------------------------------------------------------------ conductive_IC's closed forms
# conductive_IC 39487bbf closed forms (operators.py, dipolar B0, not cdipole), in derive.py's symbols; D^k on the unknown X -> Xk
import sympy as sy
S = {n: sy.Symbol(n) for n in ['P0','P1','P2','P3','T0','T1','T2','F0','F1','F2','F3','G0','G1','G2','h0','h1','h2','h3','lr1','lr2','lr3']}
globals().update(S)
def ind_f(l, m, comp, dl):   # r^3 r.curl(B0 x u) (A sign), column l+dl
    if comp == 'P':
        if dl == -1:
            lc = l; C = sy.sqrt(l**2-m**2)*(l**2-1)/(2*l-1); return C*((l-2)*h0*P0 + l*r*h1*P0 - 2*r*h0*P1)
        if dl == 1:
            C = sy.sqrt((l+1)**2-m**2)*l*(l+2)/(2*l+3); return C*(-(l+3)*h0*P0 - (l+1)*r*h1*P0 - 2*r*h0*P1)
    if comp == 'T' and dl == 0:
        return -2*sy.I*m*r*h0*T0
def ind_g(l, m, comp, dl):
    L = l*(l+1)
    if comp == 'P' and dl == 0:
        return 2*sy.I*m*(h0*P1 + r*h1*P1 - (l**2+l+1)*h0*P0/r + h1*P0 + sy.Rational(L,2)*r*h2*P0 + r*h0*P2)
    if comp == 'T':
        if dl == -1:
            C  = (2*l+1)*sy.sqrt(sy.Rational(l*(l**2-1)*(l**2-m**2), 4*l**2-1))
            C1 = sy.sqrt(sy.Rational(l**2-1, 4*l**3-l)); C2 = sy.sqrt(sy.Rational(l*(l**2-1), 4*l**2-1))
            return C*(C2*h0*T0 - 2*C1*r*h0*T1 + (C2-2*C1)*r*h1*T0)
        if dl == 1:
            C = l*(l+2)*sy.sqrt(1+2*l+l**2-m**2)/(3+2*l)
            return C*(-2*r*h0*T1 - (l+1)*h0*T0 - (l+3)*r*h1*T0)
def lor_u(l, m, comp, dl):  # r^5 r.curlcurl(G) times -1 (A sign)
    L = l*(l+1)
    if comp == 'F':
        if dl == -1:
            C = sy.sqrt(l**2-m**2)*(l**2-1)/(2*l-1)
            return C*(-2*(l**2+2)*r*h0*F1 - 2*(l-2)*r**2*h1*F1 - (l-4)*r**2*h0*F2 - (l-2)*r**3*h1*F2
                      + L*(l+2)*h0*F0 + L*(l-4)*r*h1*F0 + l*r**2*h2*F0 + l*r**3*h3*F0 + 2*r**3*h0*F3)
        if dl == 1:
            C = sy.sqrt((1+l+m)*(1+l-m))*l*(l+2)/(2*l+3)
            return C*(-2*(l**2+2*l+3)*r*h0*F1 + 2*(l+3)*r**2*h1*F1 + (l+5)*r**2*h0*F2 + (l+3)*r**3*h1*F2
                      - L*(l-1)*h0*F0 - L*(l+5)*r*h1*F0 - (l+1)*r**2*h2*F0 - (l+1)*r**3*h3*F0 + 2*r**3*h0*F3)
    if comp == 'G' and dl == 0:
        return 2*sy.I*m*(-r*h0*G0 - (l**2+l-1)*r**2*h1*G0 + r**2*h0*G1 + r**3*h1*G1 + r**3*h0*G2)
def lor_v(l, m, comp, dl):  # r^3 r.curl(G) times -1
    L = l*(l+1)
    if comp == 'F' and dl == 0:
        return sy.I*m*(4*h0*F1 - L*(2*h1*F0 + r*h2*F0) + 2*r*h0*F2)
    if comp == 'G':
        if dl == -1:
            C = sy.sqrt((l-m)*(l+m))*(l**2-1)/(2*l-1); return C*((l-2)*h0*G0 + l*r*h1*G0 - 2*r*h0*G1)
        if dl == 1:
            C = -sy.sqrt((l+m+1)*(l+1-m))*l*(l+2)/(2*l+3); return C*((l+3)*h0*G0 + (l+1)*r*h1*G0 + 2*r*h0*G1)


# ------------------------------------------------------------------ 1. uniform density
uni = {d.lr: 0}
def unif(dct):  # drop the lambda terms (rows computed with general rho0)
    return {k: v for k, v in dct.items() if not any(n[0] == 'lr' for n, p in k[0])}
cases = [('f','P',ind_f,3), ('f','T',ind_f,3), ('g','P',ind_g,3), ('g','T',ind_g,3), ('u','F',lor_u,5), ('u','G',lor_u,5), ('v','F',lor_v,3), ('v','G',lor_v,3)]
for (row, part, form, p) in cases:
    for l, m in [(3,1), (4,2), (5,0), (6,3)]:
        for dl in (-1, 0, 1):
            cf = form(l, m, part, dl)
            lc = l + dl
            if cf is None or cf == 0 or lc < max(m, 1): continue
            sym = unif(d.rows(row, part, lc, l, m))
            compare(sym, d.as_dict(cf/r**p), '%s row l=%d m=%d, %s col l%+d' % (row, l, m, part, dl))


# ------------------------------------------------------------------ 2. the (ln rho0)' terms
def split(dct):
    lam = {k: v for k, v in dct.items() if any(n[0] == 'lr' for n, p in k[0])}
    return lam
def yc(lr_, lc, m):   # <Y_lr | cos Y_lc> in kore's normalisation
    if lc == lr_ + 1: return sy.sqrt((lr_+1)**2 - m**2)/(2*lr_+3)
    if lc == lr_ - 1: return sy.sqrt(lr_**2 - m**2)/(2*lr_-1)
    return 0
def f_extra(l, m, dl):   # notes: D -> D + lambda1 in the f row
    return ind_f(l, m, 'P', dl).subs(P1, P1 + lr1*P0) - ind_f(l, m, 'P', dl)
def g_extra(l, m, dl):   # notes: diagonal only for l0 = 1
    if dl != 0: return 0
    L = l*(l+1)
    return 2*sy.I*m*(r*h0*lr1*P1 + r*h0*lr2*P0 + (1 - sy.Rational(L,2))*r*h1*lr1*P0 - sy.Rational(L,2)*h0*lr1*P0)
def u_extra(l, m, part, dl):   # notes: -(lambda1/r) (div_1 G_H)_l', times -r^5 (A sign, cic scale)
    lc = l + dl; L, Lp, L0 = lc*(lc+1), l*(l+1), 2
    j0 = h2 + 2*h1/r - 2*h0/r**2
    if part == 'G':
        z = -1 if dl == 0 else 0
        g, g1 = G0, G1
        div = (sy.I*m*z/r**2)*(L*(h0 + r*h1)*g + L0*h0*(g + r*g1))
    else:
        y = yc(l, lc, m)
        DF = F2 + 2*F1/r - L*F0/r**2
        div = (y/(2*r))*(L0*h0*DF*(L0 - L - Lp) + L*j0*F0*(L - L0 - Lp))
    return -r**5*(-(lr1/r))*div
for (row, part, p) in [('f','P',3), ('g','P',3), ('u','F',5), ('u','G',5)]:
    for l, m in [(3,1), (4,2), (5,0), (6,3)]:
        for dl in (-1, 0, 1):
            lc = l + dl
            if lc < max(m, 1): continue
            sym = split(d.rows(row, part, lc, l, m))
            cand = {'f': lambda: f_extra(l, m, dl) if dl else 0, 'g': lambda: g_extra(l, m, dl)}[row]() if row in 'fg' else u_extra(l, m, part, dl)
            cand = sy.expand(cand)
            if not sym and cand == 0: continue
            if cand == 0:
                BAD.append('unexpected lambda terms'); print('%s row l=%d m=%d %s col l%+d: lambda terms present, none expected' % (row, l, m, part, dl)); continue
            if not sym:
                BAD.append('missing lambda terms'); print('%s row l=%d m=%d %s col l%+d: no lambda terms, expected some' % (row, l, m, part, dl)); continue
            compare(sym, d.as_dict(cand/r**p), '%s row l=%d m=%d, %s col l%+d (lambda terms)' % (row, l, m, part, dl))


# ------------------------------------------------------------------ 4. NEW (2026-10-28): toroidal B0 of degree 1
# the closed forms of operators.induction_tor and lorentz_tor (times -L^3/r^p, and without Hendrik), against derive.rows(b0=('tor', 1))
k0, k1, k2 = sy.symbols('k0 k1 k2')
def tor_form(row, part, l, m, dl):
    L = l*(l+1); Cm = sy.sqrt(l**2 - m**2)/(2*l - 1); Cp = sy.sqrt((l+1)**2 - m**2)/(2*l + 3); I = sy.I
    if (row, part, dl) == ('f', 'P', 0):  return -I*m*L*r*k0*P0
    if (row, part, dl) == ('g', 'P', -1): return -Cm*l*(l-1)*(l+1)*(k0*P0 + r*k0*lr1*P0 - r*k1*P0)
    if (row, part, dl) == ('g', 'P', 1):  return Cp*l*(l+1)*(l+2)*(k0*P0 + r*k0*lr1*P0 - r*k1*P0)
    if (row, part, dl) == ('g', 'T', 0):  return -I*m*L*r*k0*T0
    if (row, part, dl) == ('u', 'F', 0):
        return -I*m*((L**2 - 2*L - 2)*r*k0*F0 - (L+2)*r**2*k0*lr1*F0 + 2*r**2*k1*F0 - 2*(L-1)*r**2*k0*F1 - L*r**3*k1*lr1*F0
                     + L*r**3*k2*F0 - 2*r**3*k0*lr1*F1 + 2*r**3*k1*F1 - (L-2)*r**3*k0*F2)
    if (row, part, dl) == ('u', 'G', -1):
        return -Cm*(2*l*(l**2-1)*r**2*k0*G0 + (l**2-1)*(l+2)*r**3*k0*lr1*G0 - 2*(l**2-1)*r**3*k1*G0 - 2*(l**2-1)*r**3*k0*G1)
    if (row, part, dl) == ('u', 'G', 1):
        return Cp*(2*l*(l+1)*(l+2)*r**2*k0*G0 + l*(l-1)*(l+2)*r**3*k0*lr1*G0 + 2*l*(l+2)*r**3*k1*G0 + 2*l*(l+2)*r**3*k0*G1)
    if (row, part, dl) == ('v', 'F', -1): return -Cm*((l-2)*(l**2-1)*k0*F0 + l*(l**2-1)*r*k1*F0 - 2*(l**2-1)*r*k0*F1)
    if (row, part, dl) == ('v', 'F', 1):  return Cp*(l*(l+2)*(l+3)*k0*F0 + l*(l+1)*(l+2)*r*k1*F0 + 2*l*(l+2)*r*k0*F1)
    if (row, part, dl) == ('v', 'G', 0):  return -I*m*(L-2)*r*k0*G0
    return 0
def quad(l, m, dl):
    return {0: sy.Rational(1, (2*l - 1)*(2*l + 3)), -1: sy.I*m*sy.sqrt(l**2 - m**2)/(2*l - 1), 1: sy.I*m*sy.sqrt((l+1)**2 - m**2)/(2*l + 3),
            -2: sy.sqrt((l**2 - m**2)*((l-1)**2 - m**2))/((2*l - 1)*(2*l - 3)), 2: sy.sqrt(((l+1)**2 - m**2)*((l+2)**2 - m**2))/((2*l + 3)*(2*l + 5))}[dl]
def tor2_form(row, part, l, m, dl):   # operators.induction_tor2, lorentz_tor2 (times -L^3, without Hendrik), labels as r-power monomials
    L = l*(l+1); A = quad(l, m, dl) if abs(dl) <= 2 else 0; Q = L - 3*m**2
    g0 = k0*P0 + r*k0*lr1*P0
    if (row, part) == ('f', 'P') and dl in (-1, 1): return -A*3*(l*(l-1) if dl == -1 else (l+1)*(l+2))*r*k0*P0
    if (row, part) == ('g', 'P'):
        if dl == -2: return -A*(3*l*(l-2)*(l+1)*g0 - 3*(l-2)*(l-1)*(l+1)*r*k1*P0 + 3*(l-2)*(l+1)*r*k0*P1)
        if dl == 0:  return -A*Q*(-3*L*g0 - 3*L*r*k1*P0 - 6*L*r*k0*P1)
        if dl == 2:  return -A*(-3*l*(l+1)*(l+3)*g0 + 3*l*(l+2)*(l+3)*r*k1*P0 + 3*l*(l+3)*r*k0*P1)
    if (row, part) == ('g', 'T') and dl in (-1, 1): return -A*3*L*r*k0*T0
    if (row, part) == ('u', 'F') and dl in (-1, 1):
        c = ([3*(l**4 - 3*l**2 - 6), -3*(l**2 - l + 6), -6*(l-3), -3*l*(l-1), 3*l*(l-1), -6*(l-1)*(l+3), -18, -6*(l-3), -3*(l-2)*(l+3)] if dl == -1 else
             [3*(l**4 + 4*l**3 + 3*l**2 - 2*l - 8), -3*(l**2 + 3*l + 8), 6*(l+4), -3*(l+1)*(l+2), 3*(l+1)*(l+2), -6*(l-2)*(l+2), -18, 6*(l+4), -3*(l-2)*(l+3)])
        mono = [r*k0*F0, r**2*k0*lr1*F0, r**2*k1*F0, r**3*k1*lr1*F0, r**3*k2*F0, r**2*k0*F1, r**3*k0*lr1*F1, r**3*k1*F1, r**3*k0*F2]
        return -A*sum(ci*mi for ci, mi in zip(c, mono))
    if (row, part) == ('u', 'G') and dl in (-2, 0, 2):
        c = {-2: [6*l*(l-2)*(l+1), 3*(l-2)*(l+1)*(l+2), -6*(l-2)*(l+1), -6*(l-2)*(l+1)], 0: [-6*L*Q, 9*(L-2)*Q, -6*(2*L-3)*Q, -6*(2*L-3)*Q],
             2: [-6*l*(l+1)*(l+3), -3*l*(l-1)*(l+3), -6*l*(l+3), -6*l*(l+3)]}[dl]
        mono = [r**2*k0*G0, r**3*k0*lr1*G0, r**3*k1*G0, r**3*k0*G1]
        return -A*sum(ci*mi for ci, mi in zip(c, mono))
    if (row, part) == ('v', 'F') and dl in (-2, 0, 2):
        c = {-2: [3*(l-4)*(l-2)*(l+1), 3*(l-2)*(l-1)*(l+1), -9*(l-2)*(l+1)], 0: [-3*(L-6)*Q, 3*L*Q, -6*(L-3)*Q],
             2: [-3*l*(l+3)*(l+5), -3*l*(l+2)*(l+3), -9*l*(l+3)]}[dl]
        return -A*(c[0]*k0*F0 + c[1]*r*k1*F0 + c[2]*r*k0*F1)
    if (row, part) == ('v', 'G') and dl in (-1, 1): return -A*3*((l-3)*(l+2) if dl == -1 else (l-1)*(l+4))*r*k0*G0
    return 0
for n, form, dls in ((1, tor_form, (-1, 0, 1)), (2, tor2_form, (-2, -1, 0, 1, 2))):
  for (row, part, p) in [('f','P',3), ('f','T',3), ('g','P',3), ('g','T',3), ('u','F',5), ('u','G',5), ('v','F',3), ('v','G',3)]:
    for l, m in [(3,1), (4,2), (5,0), (6,3)]:
        for dl in dls:
            lc = l + dl
            if lc < max(m, 1): continue
            sym = d.rows(row, part, lc, l, m, b0=('tor', n))
            cand = sy.expand(form(row, part, l, m, dl))
            if cand == 0:
                if sym: BAD.append('toroidal: unexpected terms'); print('toroidal n=%d %s row l=%d m=%d %s col l%+d: terms present, none expected' % (n, row, l, m, part, dl))
                continue
            compare(sym, d.as_dict(cand/r**p), 'toroidal B0 n=%d: %s row l=%d m=%d, %s col l%+d' % (n, row, l, m, part, dl))

# ------------------------------------------------------------------ 3. none in the rows v and the T columns
for row, part in [('v','F'),('v','G'),('f','T'),('g','T')]:
    for l, m in [(3,1),(4,0)]:
        for dl in (-1,0,1):
            out = d.rows(row, part, l+dl, l, m)
            if [k for k in out if any(n[0]=='lr' for n,p in k[0])]:
                BAD.append('lambda terms in %s %s' % (row, part)); print('LAMBDA TERMS in row', row, 'column', part, l, m, dl)
print('rows v and T columns: no (ln rho0) terms' if not any('lambda terms in' in b for b in BAD) else '')
print('derive_check.py:', 'ok' if not BAD else 'FAIL: %s' % BAD)
sys.exit(1 if BAD else 0)
