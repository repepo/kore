#!/usr/bin/env python3
'''
derive_check.py: derive.py's rows against closed forms (needs sympy; about 40 minutes on one core).

WHY. bin/operators.py holds the magnetic couplings as closed forms in l and m, written by hand (ported from conductive_IC) or
fitted from derive.py (tools/derive/fit_rows.py). This script checks that each closed form equals the row derive.py gets
from the definitions, term by term, so that a typo, a wrong sign or a missing (ln ρ0)' term in operators.py is caught.
HOW. For each (row, column, Δl) and a few (l, m), derive.rows() gives the exact projection, a closed form below gives the
candidate, both as {(monomial, power of r): coefficient}, and derive.compare() requires them to agree to 1e-10 with the
overall factor -1 (kore's A matrix holds minus the right-hand side). The degrees checked (l 3-6, m 0-3) lie inside the
range the fitted forms were obtained from: this checks that the code equals the derivation there; the evidence at large l
is listed in docs/derivations.md (part A, Verification).
The checks:
  1. uniform density: conductive_IC's operators (39487bbf, operators.py: induction, lorentz, dipolar B0), up to one factor
     (-1, the A-matrix sign; every row of kore is the conductive_IC row over L³ besides);
  2. the (ln ρ0)' terms: the closed forms implemented in bin/operators.py (rows f: d/dr -> d/dr + λ1 on P; rows g: the
     diagonal term 2im[r h λ1 P' + r h λ2 P + (1 - L/2) r h' λ1 P - (L/2) h λ1 P]; rows u: λ1 r⁴ (∇₁⋅𝐆_H)_l, operators.lorentz_rho);
  3. no (ln ρ0)' terms in the rows v and in the T columns;
  4. the toroidal B0 of degrees 1 and 2 (NEW 2026-10-28): derive.rows(b0=('tor', n)), any ρ0, against operators.induction_tor,
     lorentz_tor (n = 1) and induction_tor2, lorentz_tor2 (n = 2) (their rows times -L³/r^p; Hendrik apart), and no couplings beyond those;
  5. the poloidal B0 of degree 2 (NEW 2026-10-28, 'Luo_S2', FDM with B0_l = 2): derive.rows(b0=('pol', 2)) against the tables of
     operators.induction_pol2, lorentz_pol2, read from bin/operators.py.
Degrees l = 3..6, m = 0..3 (about 40 minutes on one core). Exit 1 on a mismatch.
'''
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import derive as d, sympy as sy
from derive import r
BAD = []   # labels of the failed checks
def compare(a, b, label):
    '''derive.compare, recording a failure: the factor must be -1 (the A sign) and the mismatch below 1e-10'''
    c, err = d.compare(a, b, label)
    if err > 1e-10 or abs(c + 1) > 1e-10: BAD.append(label)

# ------------------------------------------------------------------ conductive_IC's closed forms
# conductive_IC 39487bbf closed forms (operators.py, dipolar B0, not cdipole), in derive.py's symbols; D^k on the unknown X -> Xk
import sympy as sy
S = {n: sy.Symbol(n) for n in ['P0','P1','P2','P3','T0','T1','T2','F0','F1','F2','F3','G0','G1','G2','h0','h1','h2','h3','lr1','lr2','lr3']}
globals().update(S)
def ind_f(l, m, comp, dl):   # r^3 r.curl(B0 x u) (A sign), column l+dl
    '''conductive_IC's induction row f (uniform density, dipolar B0) for the column comp ('P' or 'T') of degree l + dl'''
    if comp == 'P':
        if dl == -1:
            lc = l; C = sy.sqrt(l**2-m**2)*(l**2-1)/(2*l-1); return C*((l-2)*h0*P0 + l*r*h1*P0 - 2*r*h0*P1)
        if dl == 1:
            C = sy.sqrt((l+1)**2-m**2)*l*(l+2)/(2*l+3); return C*(-(l+3)*h0*P0 - (l+1)*r*h1*P0 - 2*r*h0*P1)
    if comp == 'T' and dl == 0:
        return -2*sy.I*m*r*h0*T0
def ind_g(l, m, comp, dl):
    '''conductive_IC's induction row g, as ind_f'''
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
    '''conductive_IC's Lorentz row u (uniform density, dipolar B0) for the column comp ('F' or 'G') of degree l + dl'''
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
    '''conductive_IC's Lorentz row v, as lor_u'''
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
    '''the uniform-density part of a derived row: the terms without (ln ρ0) derivatives'''
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
    '''the variable-density part of a derived row: the terms with (ln ρ0) derivatives'''
    lam = {k: v for k, v in dct.items() if any(n[0] == 'lr' for n, p in k[0])}
    return lam
def yc(lr_, lc, m):   # <Y_lr | cos Y_lc> in kore's normalisation
    '''the coupling coefficient of cos θ between the degrees lc and lr_ (the angular factor of a dipolar field)'''
    if lc == lr_ + 1: return sy.sqrt((lr_+1)**2 - m**2)/(2*lr_+3)
    if lc == lr_ - 1: return sy.sqrt(lr_**2 - m**2)/(2*lr_-1)
    return 0
def f_extra(l, m, dl):   # notes: D -> D + lambda1 in the f row
    '''the (ln ρ0)' terms of the f rows: every d/dr on P becomes d/dr + λ1 (the consoidal part of u is P' + P/r + λ1 P)'''
    return ind_f(l, m, 'P', dl).subs(P1, P1 + lr1*P0) - ind_f(l, m, 'P', dl)
def g_extra(l, m, dl):   # notes: diagonal only for l0 = 1
    '''the (ln ρ0)' terms of the g rows (same degree only), from -B0 ∇⋅u = B0 λ1 u_r and the λ1 P of the consoidal part'''
    if dl != 0: return 0
    L = l*(l+1)
    return 2*sy.I*m*(r*h0*lr1*P1 + r*h0*lr2*P0 + (1 - sy.Rational(L,2))*r*h1*lr1*P0 - sy.Rational(L,2)*h0*lr1*P0)
def u_extra(l, m, part, dl):   # notes: -(lambda1/r) (div_1 G_H)_l', times -r^5 (A sign, cic scale)
    '''the (ln ρ0)' terms of the u rows: ρ0 r̂⋅∇×∇×(ρ0⁻¹𝐆) = r̂⋅∇×∇×𝐆 - (λ1/r) ∇₁⋅𝐆_H, for the column part 'F' or 'G' '''
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
    '''the closed form of operators.induction_tor or lorentz_tor (degree-1 toroidal B0) for one block, as a sympy expression
    in derive.py's symbols (times -L^3, without Hendrik); 0 where kore has no coupling'''
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
    '''the angular factors of a degree-2 background field (operators.quad_factors), exact'''
    return {0: sy.Rational(1, (2*l - 1)*(2*l + 3)), -1: sy.I*m*sy.sqrt(l**2 - m**2)/(2*l - 1), 1: sy.I*m*sy.sqrt((l+1)**2 - m**2)/(2*l + 3),
            -2: sy.sqrt((l**2 - m**2)*((l-1)**2 - m**2))/((2*l - 1)*(2*l - 3)), 2: sy.sqrt(((l+1)**2 - m**2)*((l+2)**2 - m**2))/((2*l + 3)*(2*l + 5))}[dl]
def tor2_form(row, part, l, m, dl):   # operators.induction_tor2, lorentz_tor2 (times -L^3, without Hendrik), labels as r-power monomials
    '''the closed form of operators.induction_tor2 or lorentz_tor2 (degree-2 toroidal B0) for one block, as tor_form'''
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

# ------------------------------------------------------------------ 5. NEW (2026-10-28): poloidal B0 of degree 2
# operators.induction_pol2, lorentz_pol2, read from the source of bin/operators.py (each (row, column, Δl) entry of their tables,
# labels as r-power monomials: u{X}bpo{j}[lho{k}]ohr1_D{d} -> r^(5-X) h^(j) [λ_k] F^(d), f, g: r^(2-X)), times -L^3/quad_factors and
# without Hendrik, against derive.rows(b0=('pol', 2)), any ρ0; and no couplings beyond those
import ast, re
# the tables are read from the source of operators.py (not copied here), so the check sees exactly what kore runs
SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bin', 'operators.py')).read()
POL2 = {}
for fn in ast.parse(SRC).body:
    if isinstance(fn, ast.FunctionDef) and fn.name in ('induction_pol2', 'lorentz_pol2'):
        tab = [n for n in ast.walk(fn) if isinstance(n, ast.Dict)][0]
        for k, v in zip(tab.keys, tab.values):
            POL2[ast.literal_eval(k)] = ast.unparse(v.body)
def label_mono(lab):
    '''a label u{X}bpo{j}[lho{k}]ohr1_D{d} (or f, g, v) -> (power of r of kore's row, j, k, d): the monomial r^p h^(j) λ_k X^(d)'''
    row, X, j, k, dd = re.match(r'([uvfg])(\d)bpo(\d)(?:lho(\d))?(?:ohr1)?_D(\d)$', lab).groups()
    rp = {'u': 5, 'v': 3, 'f': 2, 'g': 2}[row]
    col = {'u': 'F', 'v': 'F', 'f': 'P', 'g': 'P'}[row]   # replaced by the actual column below
    return rp - int(X), int(j), (int(k) if k else 0), int(dd)
def pol2_form(row, part, l, m, dl):
    '''the table entry of induction_pol2 or lorentz_pol2 for one block, as a sympy expression in derive.py's symbols, times
    the angular factor and -1 (the operator returns -quad_factors * entry / L^3); 0 where the table has no entry'''
    e = POL2.get((row, part, dl))
    if e is None: return 0
    loc = {'l': sy.Integer(l), 'm': sy.Integer(m), 'L': sy.Integer(l*(l+1))}
    for lab in set(re.findall(r'[uvfg]\d\w*_D\d', e)):
        rp, j, k, dd = label_mono(lab)
        loc[lab] = r**rp*S['h%d' % j]*(S['lr%d' % k] if k else 1)*S['%s%d' % (part, dd)]
    return -quad(l, m, dl)*sy.sympify(e, locals=loc)
for (row, part, p) in [('f','P',3), ('f','T',3), ('g','P',3), ('g','T',3), ('u','F',5), ('u','G',5), ('v','F',3), ('v','G',3)]:
    for l, m in [(3,1), (4,2), (5,0), (6,3)]:
        for dl in (-2, -1, 0, 1, 2):
            lc = l + dl
            if lc < max(m, 1): continue
            sym = d.rows(row, part, lc, l, m, b0=('pol', 2))
            cand = sy.expand(pol2_form(row, part, l, m, dl))
            if cand == 0:
                if sym: BAD.append('poloidal n=2: unexpected terms'); print('poloidal n=2 %s row l=%d m=%d %s col l%+d: terms present, none expected' % (row, l, m, part, dl))
                continue
            compare(sym, d.as_dict(cand/r**p), 'poloidal B0 n=2: %s row l=%d m=%d, %s col l%+d' % (row, l, m, part, dl))

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
