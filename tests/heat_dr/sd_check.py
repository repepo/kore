#!/usr/bin/env python3
'''
sd_check.py: spin_doctor's balances for the modes of eig.npz (csolve.py), with and without the differential-rotation
terms, and those terms against the matrix-level pairings of ebal.py. Run in a run directory (diff_rot 0 or 1).
Per mode: resid1, resid2, resens, resid3 as spin_doctor computes them ('on'), the same without Wdr, Wdr_im, Ensdr,
Wdr_thm ('off'), and the differences Wdr + i Wdr_im vs -2R and Wdr_thm vs 2 Re H (R, H of ebal.py), relative to the
scales 2|lambda|KE and 2|lambda|TE of the balances (spin_doctor's KE and TE are ebal's I and E, checked in the last columns).
'''
import sys, os
import numpy as np
sys.argv = ['ebal.py', 'none']
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ebal.py')).read().split("if __name__")[0])
import utils4pp as upp

z = np.load('eig.npz'); lams = z['lam']; vecs = z['vecs']
res = lambda terms: abs(sum(terms))/max(abs(t) for t in terms)
worst = 0.   # verdict: Wdr + i Wdr_im vs -2R and Wdr_thm vs 2 Re H (the resid columns are resolution-limited, 1e-4 at N = 64)
print('N %d lmax %d m %d symm %d ricb %g thermal %d diff_rot %d type %s amplitude %g thermal_wind %d bci %d bco %d'
      % (N, par.lmax, m, par.symm, ricb, par.thermal, par.diff_rot, par.diff_rot_type, par.diff_rot_ampl, par.thermal_wind, par.bci, par.bco))
print(' mode    sigma      omega    |  resid1 on/off      |  resid2 on/off      |  resens on/off      |  resid3 on/off      |'
      ' dWdr       dWdr_thm       | Wdr/2|w|KE  Wdr_im/2|w|KE  Wdr_thm/2|w|TE | KE/I-1   TE/E-1')
for i, lam in enumerate(lams):
    sol = vecs[:, i]
    usol = upp.expand_reshape_sol(sol[:2*ut.n], par.symm)
    tsol = upp.expand_reshape_sol(sol[2*ut.n:3*ut.n], par.symm) if par.thermal else 0
    [ udgn, tdgn ] = upp.diagnose(usol, tsol, ricb, rcmb, 1)
    [ KE, Dkin, Ensvel, Enscor, Ensvif, Ensbuo, Wthm, Wdr, Wcor_im, Dkin_im, Wthm_im, Wdr_im, Ensdr ] = np.sum(udgn, 0)
    s, w = lam.real, lam.imag
    r1 = [ res([2*s*KE, -Dkin, -Wthm, -Wdr]), res([2*s*KE, -Dkin, -Wthm]) ]
    r2 = [ res([2*w*KE, Wcor_im, -Dkin_im, -Wthm_im, -Wdr_im]), res([2*w*KE, Wcor_im, -Dkin_im, -Wthm_im]) ]
    re = [ res([Ensvel*s, Enscor, -Ensbuo, -Ensvif, -Ensdr]), res([Ensvel*s, Enscor, -Ensbuo, -Ensvif]) ]
    P = sol[:ut.n].reshape((nl, N1)); T = sol[ut.n:2*ut.n].reshape((nl, N1))
    S = sol[2*ut.n:3*ut.n].reshape((nl, N1)) if par.thermal else None
    x = cross_pairings([(P, T, S)], [(P, T, S)])[0][0]
    dR = abs((Wdr + 1j*Wdr_im) + 2*x['R'])/(2*abs(lam)*KE) if par.diff_rot else np.nan
    line = ' %3d  %9.6f  %9.6f  | %8.1e  %8.1e  | %8.1e  %8.1e  | %8.1e  %8.1e  |' % (i, s, w, *r1, *r2, *re)
    if par.thermal:
        [ TE, Dthm, Wadv, Wdr_thm ] = np.sum(tdgn, 0)
        Dthm *= par.ThermaD
        r3 = [ res([2*s*TE, -Dthm, -Wadv, -Wdr_thm]), res([2*s*TE, -Dthm, -Wadv]) ]
        dH = abs(Wdr_thm - 2*x['H'].real)/(2*abs(lam)*TE) if par.diff_rot else np.nan
        line += ' %8.1e  %8.1e  | %9.1e  %9.1e      |' % (*r3, dR, dH)
        line += ' %9.2e  %9.2e  %9.2e      | %8.1e %8.1e' % (Wdr/(2*abs(w)*KE), Wdr_im/(2*abs(w)*KE), Wdr_thm/(2*abs(w)*TE),
                                                          KE/x['I'].real - 1, TE/x['E'].real - 1)
    else:
        line += '    -         -      | %9.1e      -         |' % dR
        line += ' %9.2e  %9.2e      -          | %8.1e    -' % (Wdr/(2*abs(w)*KE), Wdr_im/(2*abs(w)*KE), KE/x['I'].real - 1)
    print(line)
    if par.diff_rot:
        worst = max([worst] + [v if np.isfinite(v) else np.inf for v in ([dR, dH] if par.thermal else [dR])])
if par.diff_rot:
    print('worst dWdr, dWdr_thm %.1e  ->  %s' % (worst, 'ok' if worst < 1e-5 else 'FAIL'))
