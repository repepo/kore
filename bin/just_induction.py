#!/usr/bin/env python3
'''
NEW (2026-11-06, from conductive_IC): kinematic induction. The field b that a given flow u induces in the background field
B0, without the Lorentz force of b on the flow. In a run directory with magnetic = 1 (and hydro = 1), of an eigenvalue
problem (forcing = 0) or, NEW 2026-11-06, of a forced problem:

> ./bin/submatrices.py ncpus
> mpiexec -n ncpus ./bin/assemble.py          # A.npz and B.npz, with the rows of the field
> ./bin/just_induction.py [lambda]           # A_ind.npz and u_ind.npz
> mpiexec -n ncpus ./bin/solve_ind.py         # real_magnetic.field, imag_magnetic.field

The flow is read from real_flow.field and imag_flow.field, one solution per column, as another run wrote them with the same
N, lmax, m, symm and ricb: a forced run of the flow alone (magnetic = 0), for instance. For a flow with the time dependence
exp(lambda t) the rows of the field in A and B (sections f, g, and fic, gic of a conducting inner core, fm, gm of a
resolved mantle) give

    (A_bb - lambda B_bb) b = -A_bu u,

the induction equation with u given. lambda is the command-line argument, a Python complex number (e.g. -0.01+0.52j, the
eigenvalue of a mode), or by default i forcing_frequency: the forced runs' exp(i w t), w = forcing_frequency, and a steady
flow for w = 0. A_ind.npz holds A_bb - lambda B_bb and u_ind.npz the right-hand sides -A_bu u, one column per flow
solution. assemble.py divides A and B by the same norm, which leaves b unchanged; b is in the units of the flow given.
In the directory of a forced problem A already holds A - i w B (w = forcing_frequency), so the default lambda = i w needs
no B.npz; another lambda takes A - i w B - (lambda - i w) B, with B.npz, which of the forced problems only forcing = 11
writes: in another forced run it stops (a B.npz there would be left from another assembly).
'''
import os
import sys
sys.path.insert(1, 'bin/')
import numpy as np
import scipy.sparse as ss

from parameters import par
import utils as ut

if not (par.magnetic and par.hydro):
    sys.exit('just_induction.py needs magnetic = 1 and hydro = 1: it reads the rows of the field and the columns of the flow '
             'in A.npz')
lam = complex(sys.argv[1]) if len(sys.argv) > 1 else 1j*par.forcing_frequency

u = np.loadtxt('real_flow.field') + 1j*np.loadtxt('imag_flow.field')
if u.size % (2*ut.n):
    sys.exit('real_flow.field has %d values, not a multiple of the 2 n = %d of this run: the flow must come from a run with '
             'the same N, lmax, m, symm and ricb' % (u.size, 2*ut.n))
u = u.reshape((2*ut.n, -1))

A = ut.load_csr('A.npz')
if A.shape[0] != ut.sizmat:
    sys.exit('A.npz has %d rows, this run %d: run submatrices.py and assemble.py with these parameters' % (A.shape[0], ut.sizmat))
dlam = lam - 1j*ut.wf   # NEW (2026-11-06): A of a forced problem is A - i w B (ut.wf = 0 for forcing = 0)
if ut.bessel and dlam != 0:   # NEW (2026-11-06): its rows hold at the forcing frequency only
    sys.exit("innercore 'conducting, Bessel' holds for lambda = i forcing_frequency = %s only (k of the core's field j_l(k r) "
             "is that frequency's); use 'conducting, Chebys' for another lambda" % (1j*ut.wf))
if dlam != 0 and par.forcing not in (0, 11):   # NEW (2026-11-06): of the forced runs only forcing 11 writes B.npz
    sys.exit('lambda = %s: forcing = %d writes no B.npz (forcing = 0 and 11 do), so only lambda = i forcing_frequency = %s '
             'is possible here' % (lam, par.forcing, 1j*ut.wf))
if (par.forcing == 0 or dlam != 0) and not os.path.isfile('B.npz'):
    sys.exit('lambda = %s needs B.npz, which this run has not: run assemble.py' % (lam,))
B = ut.load_csr('B.npz') if (par.forcing == 0 or dlam != 0) else None
if B is not None and B.shape != A.shape:
    sys.exit('B.npz is %d x %d, A.npz %d x %d: run submatrices.py and assemble.py with these parameters'
             % (B.shape + A.shape))

# the field: sections f, g (and fic, gic, fm, gm), after the flow and the entropy; NEW (2026-11-06): before the rotating
# solids (rotdyn = 1), left out here: for m = 0 their rotation induces no field and leaves the wall conditions unchanged;
# NEW (2026-11-08): and before the composition (compositional = 1), which the flow carries but which induces no field
ib = np.arange(ut.offset['f'], ut.offset['c'])
Ab = A[ib]
M = (Ab[:, ib] - dlam*B[ib][:, ib]).tocsr() if B is not None else Ab[:, ib].tocsr()
M.sort_indices()
rhs = ss.csr_matrix(-(Ab[:, :2*ut.n] @ u))   # the flow: the first 2 n columns

np.savez('A_ind.npz', data=M.data, indices=M.indices, indptr=M.indptr, shape=M.shape)
np.savez('u_ind.npz', data=rhs.data, indices=rhs.indices, indptr=rhs.indptr, shape=rhs.shape)
print('just_induction: %d flow solution(s), lambda = %s; A_ind.npz (%d x %d) and u_ind.npz written'
      % (u.shape[1], lam, M.shape[0], M.shape[1]))
