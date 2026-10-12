#!/usr/bin/env python3
# usage (in a forced run directory, forcing != 0, thermal = 1): forced_check.py ../EIG [../EIG2 ...]
# EIG = the same run with forcing = 0. The forced A must be A_eig - i wf B_eig up to the scalar Anorm (A_eig, B_eig are
# divided by it in assemble.py, A_forced is not): finds that scalar by least squares and prints, per section (u, v, h),
# max over rows of max|A_forced - c (A_eig - i wf B_eig)| / max|A_forced| in that row.
# Extra EIG runs (e.g. diff_rot = 0) show what a mismatch gives.
import sys, numpy as np, scipy.sparse as ss
sys.path.insert(0, 'bin')
from parameters import par
ld = lambda f: (lambda z: ss.csr_matrix((z['data'], z['indices'], z['indptr']), shape=z['shape']))(np.load(f))
wf = par.forcing_frequency
Af = ld('A.npz')
b = ld('B_forced.npz')
n = Af.shape[0] // 3
print('forcing', par.forcing, 'wf', wf, 'diff_rot', par.diff_rot, 'thermal_wind', par.thermal_wind,
      '| B_forced nonzero rows', b.nonzero()[0])
for d in sys.argv[1:]:
    M = ld(d + '/A.npz') - 1j*wf*ld(d + '/B.npz')
    c = (M.conj().multiply(Af)).sum() / (M.conj().multiply(M)).sum()   # = Anorm of the eigenvalue run
    R = Af - c*M
    rel = abs(R).max(axis=1).toarray().ravel() / np.maximum(abs(Af).max(axis=1).toarray().ravel(), 1e-300)
    res = [rel[k*n:(k+1)*n].max() for k in range(3)]
    print('%-14s c = %.6e%+.1ei   residual u %.1e  v %.1e  h %.1e' % (d, c.real, c.imag, *res))
