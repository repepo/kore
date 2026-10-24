#!/usr/bin/env python3
'''
mkfreqs.py grid OUT1 OUT2 | res EIGRUN OUT [GRID_FILES...]: frequency lists for solve.py (rsf_frequencies, kore units: w = Im
lambda = -F/frame, F in nHz, frame 456).
  grid: a uniform background grid F = -336.0 .. -329.5 nHz (step 0.25) and the Rossby n = 0 m 2 mode -331.932 -0.264i sampled at
        F_R + sigma (-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4), sorted and split alternately into OUT1 and OUT2.
  res:  every eigenvalue of EIGRUN/eigenvalues0.dat with F in [-336, -329.5] and |g| < 2 nHz, on resonance (w = Im lambda),
        minus those already in the GRID_FILES (to 1e-7 nHz); also writes OUT.modes (F, g, w, sigma in kore units).
'''
import sys, numpy as np
FR = 456.0
def save(fn, F):
    np.savetxt(fn, -np.asarray(F)/FR, fmt='%.12e')
if sys.argv[1] == 'grid':
    FRo, gRo = -331.932, -0.264
    F = np.concatenate([np.arange(-336.0, -329.5 + 1e-9, 0.25), FRo + abs(gRo)*np.array([-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4])])
    F = np.unique(np.round(F, 6))
    save(sys.argv[2], F[0::2]); save(sys.argv[3], F[1::2])
    print('grid: %d frequencies, F %.3f .. %.3f nHz -> %s (%d), %s (%d)' % (F.size, F.min(), F.max(), sys.argv[2], F[0::2].size, sys.argv[3], F[1::2].size))
else:
    ev = np.loadtxt(sys.argv[2] + '/eigenvalues0.dat').reshape((-1, 2))
    F, g = -ev[:, 1]*FR, ev[:, 0]*FR
    k = (F >= -336) & (F <= -329.5) & (np.abs(g) < 2)
    have = np.concatenate([np.atleast_1d(-np.loadtxt(f)*FR) for f in sys.argv[4:]]) if len(sys.argv) > 4 else np.array([])
    k &= np.array([not np.any(np.abs(have - x) < 1e-7) for x in F])
    o = np.argsort(F[k])
    save(sys.argv[3], F[k][o])
    np.savetxt(sys.argv[3] + '.modes', np.column_stack([F[k][o], g[k][o], ev[k, 1][o], ev[k, 0][o]]), fmt='%.12e',
               header='F_nHz g_nHz w_kore sigma_kore')
    print('res: %d modes on resonance (of %d eigenvalues) -> %s' % (k.sum(), len(F), sys.argv[3]))
