#!/usr/bin/env python3
'''
gen_table.py: the operator code of fitted magnetic rows (NEW 2026-10-29; docs/derivations.md is the guide).

Reads the output of fit_rows.py and prints, for the rows f, g (induction) and u, v (Lorentz force), the entries of a table
{(row, column, Δl): lambda: <polynomial in l, m> * <label> + ...} in kore's labels, as in operators.induction_pol2 and
lorentz_pol2, and the list of labels with the vector parity of their column (vP, vT, vF, vG) for submatrices.py.
Label of a monomial  X^(d) B^(j) [(ln ρ0)^(k)] r^p  (X the unknown, B = h or k):
    rows u, v:  {row}{-p}{code}{j}[lho{k}]ohr1_D{d}       rows f, g:  {row}{-(p+1)}{code}{j}[lho{k}]_D{d}
with code 'bpo' (cnorm h, poloidal B0) or 'bto' (cnorm_tor k, toroidal B0): kore's row = derive.py's row times -r⁵/L³ (u) or
-r³/L³ (v, f, g), so r^p becomes r^(5+p) = r^(rpower_u - X) and r^(3+p) = r^(2 - X) (Hendrik apart). The operator
multiplies each table entry by -quad_factors(l, m)[Δl]/L³ (n = 2; use the n = 1 factors of fit_rows.A for a degree-1 field)
and, for the Lorentz force, by Hendrik. 'L' in the expressions is l(l+1). A 'None' line in the input stops it.
usage: python3 tools/derive/gen_table.py FIT_LOG
'''
import re, sys, collections
import sympy as sy


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    l, m = sy.symbols('l m')
    terms = collections.OrderedDict()
    for line in open(sys.argv[1]):
        mt = re.match(r'(\w) (\w) dl([+-]\d)\s+(\S+) r\^(-?\d+)\s+\[a \+ b m\^2\] = (.*)$', line.strip())
        if not mt:
            continue
        row, col, dl, mono, p, expr = mt.groups()
        if expr == 'None':
            sys.exit('no closed form for %s' % line.strip())
        dl, p = int(dl), int(p)
        f = dict((s[:-1], int(s[-1])) for s in mono.split('*'))
        code, j = ('bpo', f['h']) if 'h' in f else ('bto', f['k'])
        lr = f.get('lr', 0)
        x = -p if row in 'uv' else -(p + 1)
        lab = '%s%d%s%d%s%s_D%d' % (row, x, code, j, 'lho%d' % lr if lr else '', 'ohr1' if row in 'uv' else '', f[col])
        terms.setdefault((row, col, dl), []).append((lab, sy.sympify(expr, locals={'l': l, 'm': m})))

    def pystr(e):
        return str(sy.factor(e)).replace('l**2 + l - 3*m**2', 'L - 3*m**2').replace('l**2 + l - 3', 'L - 3')

    for kind, rows in (('induction (rows f, g)', 'fg'), ('Lorentz force (rows u, v)', 'uv')):
        print('# %s' % kind)
        for (row, col, dl), tl in terms.items():
            if row in rows:
                print("        ('%s', '%s', %d): lambda: %s," % (row, col, dl, ' + '.join('(%s)*%s' % (pystr(e), lab) for lab, e in tl)))
    par = {'F': 'vF', 'G': 'vG', 'P': 'vP', 'T': 'vT'}
    labs = collections.OrderedDict()
    for (row, col, dl), tl in terms.items():
        for lab, _ in tl:
            labs.setdefault(lab, set()).add(par[col])
    bad = {k: v for k, v in labs.items() if len(v) > 1}
    print('# labels (submatrices.py: labl += [...], arg2 += [the parity] for each)%s' % ('; CONFLICTS: %s' % bad if bad else ''))
    for v in ('vP', 'vT', 'vF', 'vG'):
        print('#  %s: %s' % (v, [k for k, s in labs.items() if v in s]))


if __name__ == '__main__':
    main()
