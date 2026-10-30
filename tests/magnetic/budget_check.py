#!/usr/bin/env python3
'''
budget_check.py [tol]: in a magnetic run directory after spin_doctor.py, the energy budgets of every solution, from flow.dat
(residσ, residω: the power of the Lorentz force included) and magnetic.dat (resid𝐛 in the fluid; resid𝐄, the full budget with
the inner core, the outside potential field and the Ohmic losses, no-slip walls only). With a variable ρ0 these close only if
the (ln ρ0)' terms of the Lorentz rows and of the induction rows agree with spin_doctor's own formulas (utils4pp). The last
row of each file is the last solution. tol defaults to 1e-8; residω is skipped for real eigenvalues (all its terms are
round-off there).

WHY. The energy budgets check the magnetic operators independently of how they were derived: for an eigenmode λ = σ + iω,
2σ (kinetic + magnetic energy) must equal the sum of the powers of every force and source, and spin_doctor computes those
powers from the eigenvectors with its own formulas (for a toroidal or degree-2 field on an (r, θ) grid,
utils4pp.magnetic_powers_grid), not from kore's matrices. A wrong coefficient at any degree the mode contains leaves a
residual, so this check reaches every degree of the run (up to lmax), not only the few degrees derive_check.py looks at.
It checks integrals over whole eigenvectors, not each coefficient on its own.
'''
import sys, numpy as np
tol = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-8
# the last n rows of flow.dat and magnetic.dat are this solve's n solutions (columns as in spin_doctor's flow_cols, magnetic_cols)
ev = np.loadtxt('eigenvalues0.dat').reshape(-1, 2); n = len(ev)
fl = np.loadtxt('flow.dat', ndmin=2)[-n:]; mg = np.loadtxt('magnetic.dat', ndmin=2)[-n:]
r1, r2, rb, rE = fl[:, 16], fl[:, 15], mg[:, 3], mg[:, 10]
r2 = np.where(np.abs(ev[:, 1]) > 1e-10, r2, 0.)
worst = max(np.nanmax(r1), np.nanmax(r2), np.nanmax(rb), np.nanmax(rE))
for i in range(n):
    print('  %+.10f %+.10fi   residσ %.2e  residω %.2e  resid𝐛 %.2e  resid𝐄 %.2e' % (ev[i, 0], ev[i, 1], r1[i], r2[i], rb[i], rE[i]))
print('budget_check.py: worst %.2e -> %s' % (worst, 'ok' if worst < tol else 'FAIL'))
sys.exit(0 if worst < tol else 1)
