import numpy as np
import scipy.sparse as spr
#import scipy.io as sio

import sys
sys.path.insert(1,'bin/')
sys.path.insert(1,'koreviz/')

import parameters as par
import utils as ut

# read the flow field
realu = np.loadtxt('real_flow.field');
imagu = np.loadtxt('imag_flow.field');
u = realu[:ut.n*2] + 1j*imagu[:ut.n*2];

# reads A matrix (it must include the induction equation)
A = ut.load_csr('A.npz')

# slice A to select only the induction equation part
ind = A[ ut.n*2 : ut.n*4,       :ut.n*2 ];
iwD = A[ ut.n*2 : ut.n*4, ut.n*2:ut.n*4 ];
#indu = spr.csr_matrix(-ind*u).transpose()
indu = -ind*u

indu = spr.csr_matrix(indu.reshape(-1,1))

# solve for the induced magnetic field, scipy's spsolve needs a lot of memory, ok for small problems
#b = spr.linalg.spsolve(iwD,indu)  

np.savez('u_ind.npz', data=indu.data, indices=indu.indices, indptr=indu.indptr, shape=indu.shape)
#sio.mmwrite('u_ind.mtx',indu)


np.savez('A_ind.npz', data=iwD.data, indices=iwD.indices, indptr=iwD.indptr, shape=iwD.shape)

