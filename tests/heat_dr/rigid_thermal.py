#!/usr/bin/env python3
# Rigid-offset identity with the heat equation on: lambda(dO) = (1+dO) lambda(0; E/(1+dO), kappa/(1+dO), Beyonce/(1+dO)^2) - i m dO
# usage: rigid_thermal.py BASE_DIR DR_DIR dO m
import sys, numpy as np
base, drd, dO, m = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
lb = np.load(base + '/eig.npz')['lam']; ld = np.load(drd + '/eig.npz')['lam']
print('baseline run: %s   diff-rot run (rigid, dO = %g, m = %d, thermal = 1, thermal_wind = 1): %s' % (base, dO, m, drd))
print('%-22s %-22s %-22s %s' % ('baseline lambda', 'expected', 'nearest diff-rot lambda', '|diff|'))
worst = 0
for l0 in sorted(lb, key=lambda z: -z.real):
    ex = (1 + dO)*l0 - 1j*m*dO
    j = np.argmin(abs(ld - ex)); d = abs(ld[j] - ex); worst = max(worst, d)
    print('%-22s %-22s %-22s %.1e' % ('%.6f%+.6fi' % (l0.real, l0.imag), '%.6f%+.6fi' % (ex.real, ex.imag), '%.6f%+.6fi' % (ld[j].real, ld[j].imag), d))
print('worst |diff| = %.1e over %d modes' % (worst, len(lb)))
