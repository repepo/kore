# size of each term of the complex energy and entropy balances, per eigenmode
# modes from real/imag_*.field + eigenvalues0.dat (kore output), or from eig.npz (csolve.py) if present
import sys, os, numpy as np
sys.argv = ['ebal.py', 'none']
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ebal.py')).read().split("if __name__")[0])
if os.path.isfile('eig.npz'):
    z = np.load('eig.npz'); lams = z['lam']; vecs = z['vecs']; nev = len(lams)
    def mode(i):
        sol = vecs[:, i]
        P = sol[:ut.n].reshape((nl, N1)); T = sol[ut.n:2*ut.n].reshape((nl, N1))
        S = sol[2*ut.n:3*ut.n].reshape((nl, N1)) if par.thermal else None
        return P, T, S
else:
    ru = np.loadtxt('real_flow.field').reshape((2*ut.n, -1)); iu = np.loadtxt('imag_flow.field').reshape((2*ut.n, -1))
    eig = np.loadtxt('eigenvalues0.dat').reshape((-1, 2)); nev = ru.shape[1]; lams = eig[:, 0] + 1j*eig[:, 1]
    if par.thermal:
        rt = np.loadtxt('real_thermal.field').reshape((ut.n, -1)); it = np.loadtxt('imag_thermal.field').reshape((ut.n, -1))
    def mode(i):
        sol = ru[:, i] + 1j*iu[:, i]
        P = sol[:ut.n].reshape((nl, N1)); T = sol[ut.n:2*ut.n].reshape((nl, N1))
        S = (rt[:, i] + 1j*it[:, i]).reshape((nl, N1)) if par.thermal else None
        return P, T, S
sel = [int(a) for a in os.environ.get('MODES', '').split(',') if a != ''] or list(range(nev))
fields = [mode(i) for i in sel]
X = cross_pairings(fields, fields)
dr = ' + <R>' if par.diff_rot else ''   # R = <u, (U0.grad)u + (u.grad)U0>, the differential-rotation term (ebal.dr_blocks)
print('momentum: lambda<I> + <C>%s - <V> - <B> = 0, normalised by |lambda| K.  Real part: sigma K, ReV, ReB%s ; imaginary part: omega K, ImC, ImB%s'
      % (dr, ', -ReR' if par.diff_rot else '', ', -ImR' if par.diff_rot else ''))
print(' mode     sigma      omega   | sigma K/|lam|K   ReV/|lam|K   ReB/|lam|K %s| omega K/|lam|K  ImC/|lam|K   ImB/|lam|K %s|  res'
      % (('  -ReR/|lam|K ', '  -ImR/|lam|K ') if par.diff_rot else ('', '')))
for j, i in enumerate(sel):
    lam = lams[i]; x = X[j][j]; K = x['I'].real; n = abs(lam)*K
    res = abs(lam*x['I'] + x['C'] + x['R'] - x['V'] - x['B'])/n
    print(' %2d  %10.6f %10.6f | %12.3e %12.3e %12.3e %s| %12.3e %12.3e %12.3e %s| %.1e' % (i, lam.real, lam.imag, lam.real*K/n, x['V'].real/n, x['B'].real/n,
          ('%12.3e ' % (-x['R'].real/n)) if par.diff_rot else '', lam.imag*K/n, x['C'].imag/n, x['B'].imag/n,
          ('%12.3e ' % (-x['R'].imag/n)) if par.diff_rot else '', res))
if par.thermal:
    hd = par.diff_rot
    print('\nentropy: lambda<E> - <A> - <D>%s = 0, normalised by |lambda| E.  Real: sigma E, ReA, ReD%s ; imaginary: omega E, ImA, ImD%s'
          % (' - <H>' if hd else '', ', ReH' if hd else '', ', ImH' if hd else ''))
    print(' mode  | sigma E/|lam|E   ReA/|lam|E   ReD/|lam|E %s| omega E/|lam|E  ImA/|lam|E   ImD/|lam|E %s|  res'
          % (('  ReH/|lam|E ', '  ImH/|lam|E ') if hd else ('', '')))
    for j, i in enumerate(sel):
        lam = lams[i]; x = X[j][j]; E = x['E'].real; n = abs(lam)*E
        res = abs(lam*x['E'] - x['A'] - x['D'] - x['H'])/n
        print(' %2d    | %12.3e %12.3e %12.3e %s| %12.3e %12.3e %12.3e %s| %.1e' % (i, lam.real*E/n, x['A'].real/n, x['D'].real/n,
              ('%12.3e ' % (x['H'].real/n)) if hd else '', lam.imag*E/n, x['A'].imag/n, x['D'].imag/n, ('%12.3e ' % (x['H'].imag/n)) if hd else '', res))
    print('\nper mode: KE, TE, and the independent checks  ReC/|C|  ImV/|V|  ImD/|D|  (should be 0)')
    for j, i in enumerate(sel):
        x = X[j][j]
        print(' %2d   KE %.4e  TE %.4e   ReC/|C| %9.2e   ImV/|V| %9.2e   ImD/|D| %9.2e' % (i, x['I'].real, x['E'].real, x['C'].real/abs(x['C']), x['V'].imag/abs(x['V']), x['D'].imag/max(abs(x['D']), 1e-300)))
