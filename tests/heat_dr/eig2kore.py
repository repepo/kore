#!/usr/bin/env python3
'''
eig2kore.py: writes the eigenpairs of eig.npz (csolve.py) as kore's solve.py does, so that bin/spin_doctor.py
and the other postprocessing scripts can run without PETSc/SLEPc: eigenvalues0.dat, real_/imag_flow.field and,
with thermal = 1, real_/imag_thermal.field (one solution per column). Run in the run directory.
'''
import sys, os
sys.path.insert(0, os.path.abspath('bin'))
import numpy as np
from parameters import par
import utils as ut
z = np.load('eig.npz'); lam = z['lam']; vecs = z['vecs']
np.savetxt('eigenvalues0.dat', np.column_stack([lam.real, lam.imag]))
blocks = [('flow', 2*ut.n)] + ([('thermal', ut.n)] if par.thermal else [])
off = 0
for name, size in blocks:
    blk = vecs[off:off + size, :]; off += size
    np.savetxt('real_' + name + '.field', np.real(blk))
    np.savetxt('imag_' + name + '.field', np.imag(blk))
print('wrote %d solutions' % len(lam))
