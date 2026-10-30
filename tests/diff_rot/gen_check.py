#!/usr/bin/env python3
'''
gen_check.py: the couplings of bin/diff_rot.py (the recurrence algebra of bin/angular.py) against the closed forms of
tests/diff_rot/diff_rot_coefficients.py for f_0 and f_2: every coefficient, m = 0..mmax, all l <= lmax and |Δl| <= 3;
for n = 4..nmax, that the coefficients vanish outside the Δl ranges assembled by kore (|Δl| <= n, even, for P→P and
T→T; |Δl| <= n + 1, odd, for P↔T); and, if ΔΩ of the run has degrees 0 and 2 only and thermal_wind is 1 or 2, the
thermal-wind σ2, σ4, σ6 of diff_rot.sigma_coefficients against their closed forms (eq. 2.10). Run in a run directory
(uses bin/ and its lmax, or the lmax given):

    python3 gen_check.py [mmax [nmax [lmax]]]          defaults 8, 16, the run's lmax

The couplings are compared relative to the largest closed-form coefficient of the same row block (u or v row, P or T
column, target degree l), the scale at which the coefficients enter a row of A: a few coefficients are small
differences of O(L) terms in the algebra (e.g. df0_2C_D1P = 2im), so their own relative error grows like L eps.
WHY. The run-time algebra of diff_rot.py replaced hand-derived closed forms for P0 and P2 (still in
diff_rot_coefficients.py). Agreement to round-off at every l, m with those independent forms checks the algebra where both
exist; for higher degrees n, where no closed form exists, the check is that no coupling falls outside the Δl kore assembles
(a coupling there would be silently dropped), and angular_check.py checks the algebra itself at any n.
'''
import sys, os
sys.path.insert(0, os.path.abspath('bin'))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import diff_rot_coefficients as dc
from parameters import par
import diff_rot as dr

mmax = int(sys.argv[1]) if len(sys.argv) > 1 else 8
nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 16
if len(sys.argv) > 3:
    par.lmax = int(sys.argv[3])
lmax = par.lmax
tol = 1e-12

def closed_name(name, n):
    '''the name of the closed form in diff_rot_coefficients.py of diff_rot's coefficient `name` for P_n (f0_... for n = 0)'''
    pre = 'f0' if n == 0 else 'f'
    for a in ('d2f_', 'df_', 'f_'):
        if name.startswith(a):
            return a.replace('f', pre, 1) + name[len(a):]

def block(name):   # the row block a coefficient belongs to: u (2C) or v (1C) row, P or T column
    '''the row block of a coefficient, e.g. 'u,P': the scale its error is measured against'''
    return ('u' if '_2C_' in name else 'v') + ',' + name[-1]

worst = 0.
print('closed forms vs diff_rot.py, lmax %d, m = 0..%d (relative to the largest coefficient of the row block)' % (lmax, mmax))
for n in (0, 2):
    names = [k for k in dr.momentum_coefficients(n, 0) if hasattr(dc, closed_name(k, n))]
    blocks = sorted(set(block(k) for k in names))
    for b in blocks:
        bn = [k for k in names if block(k) == b]
        err_b, where = 0., None
        for m in range(mmax + 1):
            for l in range(max(m, 1), lmax + 1):
                err, sc = 0., 0.
                for name in bn:
                    fn = getattr(dc, closed_name(name, n))
                    for dl in range(-3, 4):
                        if l + dl < max(m, 1) or l + dl > lmax:
                            continue
                        c = complex(fn(l, m, dl))
                        err = max(err, abs(dr.momentum_coef(name, n, l, m, dl) - c)); sc = max(sc, abs(c))
                if sc > 0 and err/sc > err_b:
                    err_b, where = err/sc, (m, l)
        worst = max(worst, err_b)
        print('  n=%d %s rows: max |diff_rot - closed| / row scale = %.1e%s' % (n, b, err_b, '  (m, l) = %s' % (where,) if where else ''))

print('Δl selection rules, n = 4..%d' % nmax)
for n in range(4, nmax + 1, 2):
    bad = 0.
    for m in (0, 1, 3, mmax):
        K = dr.momentum_coefficients(n, m)
        for name, M in K.items():
            M = M.toarray()
            sec = 'v' if '_1C_' in name else 'u'
            comp = 'upol' if name.endswith('P') else 'utor'
            dmax = dr.max_dl(n, sec, comp)
            par_ok = dmax % 2
            sc = np.abs(M).max()
            l = np.arange(M.shape[0])[:, None]; ls = np.arange(M.shape[1])[None, :]
            dl = np.abs(ls - l)
            out = ((dl > dmax) | (dl % 2 != par_ok)) & (l >= max(m, 1)) & (l <= lmax) & (ls >= max(m, 1))
            if out.any():
                bad = max(bad, np.abs(M[out]).max()/(sc if sc > 0 else 1))
    worst = max(worst, bad)
    print('  n=%2d  largest coefficient outside the assembled Δl range (relative) %.1e' % (n, bad))

if par.diff_rot and par.thermal and par.thermal_wind in (1, 2) and set(dr.degrees()) <= {0, 2}:
    import utils as ut
    import radial_profiles as rap
    r = np.cos(np.pi*(np.arange(par.N) + 0.5)/par.N)            # Chebyshev nodes of the radial domain
    r = 0.5*(ut.rcmb - par.ricb)*(r + 1) + par.ricb if par.ricb > 0 else r
    f = {n: (dr.fX(n, r, 0), dr.fX(n, r, 1)) if n in dr.degrees() else (0*r, 0*r) for n in (0, 2)}
    sc = dc.sigma_closed(r, f[0][0], f[0][1], f[2][0], f[2][1], rap.prf.gravity(r, 0), par.Gaspard, par.Beyonce,
                         par.thermal_wind)
    sa = dr.sigma_coefficients(r)/par.thermal_wind_amplitude
    # relative to the natural scale of S_tw, r² G²/(Beyonce g) |∂ΔΩ| (S_tw itself vanishes for a cylindrical rotation)
    q = r**2*par.Gaspard**2/(par.Beyonce*rap.prf.gravity(r, 0))
    scale = max(np.max(np.abs(q)*(np.abs(f[0][1]) + np.abs(f[2][1]) + np.abs(f[2][0]/r))), 1e-300)
    errs = [np.abs((sa[n] if n < sa.shape[0] else 0*r) - sc[k]).max()/scale for k, n in enumerate((2, 4, 6))]
    errs.append(np.abs(sa[[k for k in range(sa.shape[0]) if k not in (2, 4, 6)]]).max()/scale)
    worst = max(worst, max(errs))
    print('thermal wind (thermal_wind %d): σ2, σ4, σ6 of diff_rot vs closed forms, other degrees (must be 0), relative to r²G²|∂ΔΩ|/(Beyonce g): %s'
          % (par.thermal_wind, ' '.join('%.1e' % e for e in errs)))

print('worst %.1e  ->  %s' % (worst, 'ok' if worst < tol else 'FAIL'))
sys.exit(0 if worst < tol else 1)
