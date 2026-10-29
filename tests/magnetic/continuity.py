#!/usr/bin/env python3
'''
continuity.py [eps ...]: the variable-density magnetic eigenvalues go over smoothly to the Boussinesq ones. 'Wu 2005' with
r_out = sqrt(eps) has ρ0 = (1 - eps r²)², λ1 = (ln ρ0)' = -4 eps r/(1 - eps r²): the stratification is O(eps). For each eps it
builds and solves the default run of mk_run.py (axial B0, m 1, N 96) and the same with structure 'Boussinesq', and prints the
change of the least damped eigenvalue, which must be about linear in eps (the ratio change/eps tends to a constant).
'''
import os, sys, subprocess, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
eps = [float(e) for e in sys.argv[1:]] or [1e-4, 1e-3, 1e-2, 1e-1]
def lam(name, extra):
    subprocess.run([sys.executable, H + '/mk_run.py', name, extra, 'solve'], check=True, stdout=subprocess.DEVNULL)
    e = np.loadtxt(name + '/eigenvalues0.dat').reshape(-1, 2)
    return e[:, 0] + 1j*e[:, 1]
l0all = lam('cont_uniform', "par.structure = 'Boussinesq'")
l0 = l0all[np.argmax(l0all.real)]
print('uniform: %+.12f %+.12fi' % (l0.real, l0.imag))
ratios = []
for e in eps:
    la = lam('cont_%g' % e, 'par.r_out = %r' % np.sqrt(e))
    l = la[np.argmin(abs(la - l0))]
    ratios.append((l - l0)/e)
    print('eps %-7g lambda %+.12f %+.12fi   (lambda - lambda_uniform)/eps = %+.6f %+.6fi' % (e, l.real, l.imag, ratios[-1].real, ratios[-1].imag))
ok = len(ratios) < 2 or abs(ratios[0] - ratios[1])/abs(ratios[0]) < 0.05
print('continuity.py: (lambda - lambda_uniform)/eps at the two smallest eps differ by %.1e relative -> %s'
      % (abs(ratios[0] - ratios[1])/abs(ratios[0]) if len(ratios) > 1 else 0, 'ok' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
