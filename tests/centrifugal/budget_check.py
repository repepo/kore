#!/usr/bin/env python3
'''
budget_check.py [tol] [HALF]: after spin_doctor.py, in a run directory: the budgets of every solution close below tol (default
1e-8) (NEW 2026-11-05). With centrifugal = 1, spin_doctor weights the kinetic energy with M (KE of the deformed body), the
Coriolis power with z̃ and the entropy "energy" with J (utils4pp.centrifugal_terms, by quadrature on the grid, apart from
the matrices): residσ and residω (flow.dat, columns 17 and 16) and resid𝑠 (thermal.dat, column 4) close only if these
weights are those of B and A. Without them residσ, residω are of order f, resid𝑠 of order f times the damping.
With magnetic = 1 (NEW, phase 1b) also resid𝐛 (magnetic.dat, column 4), which closes only with the metric part of the
induction, and residσ, residω only with the metric part of the Lorentz force (utils4pp.centrifugal_magnetic_terms). The
energy budget resid𝐄 (column 11) closes at second order in f only: the terms kore keeps are those of first order, and
their products are not cancelled in the energy. With HALF, the run directory of the same case at half the flattening,
resid𝐄 must be 4 times that of HALF, solution by solution: an order log2(ratio) within 0.4 of 2 (a term of first order
would give 1). Also solve.py's normalisation: KE = 1, with a field KE + Hendrik (ME + MEc) = 1, the energies of the
deformed body (their metric parts are computed on rank 0 from every degree: with several ranks they must count once).
NEW (2026-11-06, phase 2): the dissipation is that of the deformed body too. spin_doctor adds the metric parts of the viscous
power, of the power of a differential rotation and of the entropy diffusion (utils4pp.centrifugal_dissipation_terms) and,
with a field, of the magnetic diffusion (Mdfs, Ohmc of centrifugal_magnetic_terms); without them the residuals are of order
f times the part of the eigenvalue that the diffusion (or the differential rotation) makes.
'''
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath('bin'))
from parameters import par
tol = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-8
half = sys.argv[2] if len(sys.argv) > 2 else None
flow = np.atleast_2d(np.loadtxt('flow.dat'))
worst = {'residσ': np.nanmax(flow[:, 16]), 'residω': np.nanmax(flow[:, 15])}
try:
    worst['resid𝑠'] = np.nanmax(np.atleast_2d(np.loadtxt('thermal.dat'))[:, 3])
except OSError:
    pass
bad = [k for k, v in worst.items() if not v < tol]
mag, order = None, ''
try:
    mag = np.atleast_2d(np.loadtxt('magnetic.dat'))
except OSError:
    pass
norm = flow[:, 0] + (par.Hendrik*(mag[:, 0] + (mag[:, 11] if mag.shape[1] > 11 else 0)) if mag is not None else 0)
worst['|norm - 1|'] = np.nanmax(abs(norm - 1))
if not worst['|norm - 1|'] < tol:
    bad.append('normalisation')
if mag is not None:
    worst['resid𝐛'] = np.nanmax(mag[:, 3])
    if not worst['resid𝐛'] < tol:
        bad.append('resid𝐛')
    if half is not None:   # each solution against the nearest one of HALF (eigenvalues.dat: the rows of magnetic.dat)
        mh = np.atleast_2d(np.loadtxt(half + '/magnetic.dat'))
        ev = lambda d, n: (lambda e: e[-n:, 0] + 1j*e[-n:, 1])(np.atleast_2d(np.loadtxt(d + 'eigenvalues.dat')))
        lam, lamh = ev('', mag.shape[0]), ev(half + '/', mh.shape[0])
        ratio = mag[:, 10]/np.array([mh[np.argmin(abs(lamh - l)), 10] for l in lam])
        order = '; resid𝐄 %s, its order from half the flattening %s (2: second order)' % (
            ' '.join('%.1e' % v for v in mag[:, 10]), ' '.join('%.2f' % v for v in np.log2(ratio)))
        if not np.all(abs(np.log2(ratio) - 2) < 0.4):
            bad.append('resid𝐄 order')
print('budget_check.py: %s (%d solutions: %s; tol %g%s)' % ('FAIL ' + ', '.join(bad) if bad else 'ok', flow.shape[0],
      ', '.join('%s %.1e' % kv for kv in worst.items()), tol, order))
sys.exit(1 if bad else 0)
