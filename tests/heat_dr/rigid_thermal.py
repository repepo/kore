#!/usr/bin/env python3
# Rigid-offset identity with the heat equation on, for any Gaspard G: a rigid offset Ω = G (1 + dO) seen from the frame
# rotating at G is the frame rotating at G (1 + dO) plus a Doppler shift, so
#     lambda(diff_rot = 1, 'rigid' dO, Gaspard = G) = lambda(diff_rot = 0, Gaspard = G (1 + dO)) - i m G dO
# with ViscosD, ThermaD, Beyonce unchanged. Every eigenvalue of each run is matched to the nearest one of the other.
# usage: rigid_thermal.py BASE_DIR DR_DIR dO m G
import sys, numpy as np
base, drd, dO, m, G = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
lb = np.load(base + '/eig.npz')['lam']; ld = np.load(drd + '/eig.npz')['lam']
print('baseline run: %s (diff_rot = 0, Gaspard = G (1 + dO))   diff-rot run: %s (rigid, dO = %g, m = %d, Gaspard = G = %g)' % (base, drd, dO, m, G))
print('%-26s %-26s %-26s %s' % ('baseline lambda', 'expected', 'nearest diff-rot lambda', '|diff|/|lambda|'))
worst = 0
for l0 in sorted(lb, key=lambda z: -z.real):
    ex = l0 - 1j*m*G*dO
    j = np.argmin(abs(ld - ex)); d = abs(ld[j] - ex)/abs(ex); worst = max(worst, d)
    print('%-26s %-26s %-26s %.1e' % ('%.8f%+.8fi' % (l0.real, l0.imag), '%.8f%+.8fi' % (ex.real, ex.imag), '%.8f%+.8fi' % (ld[j].real, ld[j].imag), d))
back = max(np.min(abs(lb - 1j*m*G*dO - z))/abs(z) for z in ld)   # and every diff-rot eigenvalue has its partner
print('worst |diff|/|lambda| = %.1e over %d baseline modes; every diff-rot mode matched to %.1e  ->  %s'
      % (worst, len(lb), back, 'ok' if max(worst, back) < 1e-8 else 'FAIL'))
