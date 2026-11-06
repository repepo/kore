#!/usr/bin/env python3
'''
NEW (2026-11-06, from conductive_IC): kinematic induction. The field b that a given flow u induces in the background field
B0, without the Lorentz force of b on the flow. In a run directory with magnetic = 1 (and hydro = 1) and forcing = 0:

> ./bin/submatrices.py ncpus
> mpiexec -n ncpus ./bin/assemble.py          # A.npz and B.npz, with the rows of the field
> ./bin/just_induction.py [lambda]           # A_ind.npz and u_ind.npz
> mpiexec -n ncpus ./bin/solve_ind.py         # real_magnetic.field, imag_magnetic.field

The flow is read from real_flow.field and imag_flow.field, one solution per column, as another run wrote them with the same
N, lmax, m, symm and ricb: a forced run of the flow alone (magnetic = 0), for instance. For a flow with the time dependence
exp(lambda t) the rows of the field in A and B (sections f, g, and fic, gic of a conducting inner core) give

    (A_bb - lambda B_bb) b = -A_bu u,

the induction equation with u given. lambda is the command-line argument, a Python complex number (e.g. -0.01+0.52j, the
eigenvalue of a mode), or by default i forcing_frequency: the forced runs' exp(i w t), w = forcing_frequency, and a steady
flow for w = 0. A_ind.npz holds A_bb - lambda B_bb and u_ind.npz the right-hand sides -A_bu u, one column per flow
solution. assemble.py divides A and B by the same norm, which leaves b unchanged; b is in the units of the flow given.
'''
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
B = ut.load_csr('B.npz')
if A.shape[0] != ut.sizmat:
    sys.exit('A.npz has %d rows, this run %d: run submatrices.py and assemble.py with these parameters' % (A.shape[0], ut.sizmat))

ib = np.arange(ut.offset['f'], ut.sizmat)   # the field: sections f, g (and fic, gic), last in A (ut.offset)
Ab = A[ib]
M = (Ab[:, ib] - lam*B[ib][:, ib]).tocsr()
M.sort_indices()
rhs = ss.csr_matrix(-(Ab[:, :2*ut.n] @ u))   # the flow: the first 2 n columns

np.savez('A_ind.npz', data=M.data, indices=M.indices, indptr=M.indptr, shape=M.shape)
np.savez('u_ind.npz', data=rhs.data, indices=rhs.indices, indptr=rhs.indptr, shape=rhs.shape)
print('just_induction: %d flow solution(s), lambda = %s; A_ind.npz (%d x %d) and u_ind.npz written'
      % (u.shape[1], lam, M.shape[0], M.shape[1]))
