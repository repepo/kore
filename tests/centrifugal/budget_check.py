#!/usr/bin/env python3
'''
budget_check.py [tol]: after spin_doctor.py, in a run directory: the budgets of every solution close below tol (default 1e-8)
(NEW 2026-11-05). With centrifugal = 1, spin_doctor weights the kinetic energy with M (KE of the deformed body), the
Coriolis power with z̃ and the entropy "energy" with J (utils4pp.centrifugal_terms, by quadrature on the grid, apart from
the matrices): residσ and residω (flow.dat, columns 17 and 16) and resid𝑠 (thermal.dat, column 4) close only if these
weights are those of B and A. Without them residσ, residω are of order f, resid𝑠 of order f times the damping.
'''
import sys
import numpy as np
tol = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-8
flow = np.atleast_2d(np.loadtxt('flow.dat'))
worst = {'residσ': np.nanmax(flow[:, 16]), 'residω': np.nanmax(flow[:, 15])}
try:
    worst['resid𝑠'] = np.nanmax(np.atleast_2d(np.loadtxt('thermal.dat'))[:, 3])
except OSError:
    pass
bad = [k for k, v in worst.items() if not v < tol]
print('budget_check.py: %s (%d solutions: %s; tol %g)' % ('FAIL ' + ', '.join(bad) if bad else 'ok', flow.shape[0],
      ', '.join('%s %.1e' % kv for kv in worst.items()), tol))
sys.exit(1 if bad else 0)
