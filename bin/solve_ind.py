#!/usr/bin/env python3
'''
kore solver script for only the induction equation
 
Solves the induction equation for every flow solution found in u_ind.npz
(one right hand side per column, written by just_induction.py).
 
To use, first export desired solver options:
> export opts='...'
 
and then execute:
> mpiexec -n ncpus ./bin/solve_ind.py $opts
'''
 
from timeit import default_timer as timer
t_start = timer()  # wall clock from here, before scipy/petsc/utils are imported
import sys

from solve_nopp import load_mat

from petsc4py import PETSc
from slepc4py import SLEPc
import scipy.io as sio
import scipy.sparse as ss
import numpy as np
 
from parameters import par
import utils as ut
 
 
def main():
    
    Print = PETSc.Sys.Print
    rank = PETSc.COMM_WORLD.getRank()
    size = PETSc.COMM_WORLD.getSize()
    opts = PETSc.Options()
 
    # solver options from parameters.py, unless already given on the command line.
    # this is a linear (forced) problem, so the eps_/st_ options are skipped
    skip = ('eps_', 'st_')
    for key, val in par.petsc_opts.items():
        if not key.startswith(skip) and not opts.hasName(key):
            opts.setValue(key, val)
    
    if rank == 0:
        tic = timer()
        
    # ------------------------------------------------------------------ reads matrix A
    MA = load_mat('A_ind.npz')
    # done reading and preparing A
    
    
    # ------------------------------------------------------------ reads right hand side vectors
    # one column per flow solution
        
    b0 = ut.load_csr('u_ind.npz')
    #b0 = sio.mmread('u_ind.mtx')
    #b0 = b0.tocsr()
    nsol = b0.shape[1]
    
    x, bvec = MA.createVecs()
    Istart,Iend = bvec.getOwnershipRange()
    
    # -------------------------------------------------------------- setup solver
    K = PETSc.KSP()
    K.create(PETSc.COMM_WORLD)
    K.setOperators(MA)
    K.setTolerances(rtol=par.tol,max_it=par.maxit)
    K.setFromOptions()
 
    # scatter to rank 0, created once and reused for every solution
    tozero,VR = PETSc.Scatter.toZero(x)
 
    if rank == 0:
        bsol = np.zeros((2*ut.n, nsol), dtype=complex)  # one induced field per column
 
    # -------------------------------------------------------------- solve & collect, one per flow solution
    for j in range(nsol):
 
        bvec.set(0)
        bvec.setValues(range(Istart,Iend),b0[Istart:Iend,j].toarray())
        bvec.assemblyBegin()
        bvec.assemblyEnd()
 
        x.set(0)
        K.solve(bvec, x)
 
        # collect result on rank 0
        tozero.scatter(x, VR)
        if rank == 0:
            bsol[:,j] = VR.getArray()[ : 2*ut.n ]
 
    del b0
 
    # cleanup
    tozero.destroy()
    VR.destroy()
    K.destroy()
    MA.destroy()
    bvec.destroy()
    x.destroy()
    
    if rank == 0:
 
        if not np.all(np.isfinite(bsol)):
            success = 0
            print('Solver crashed, got nan\'s!')
        else:
            success = 1 # got actual numbers ... but it could still be a bad solution ;)
            print(nsol, 'solution(s) computed')
            # one solution per column, same order as the flow solutions
            np.savetxt('real_magnetic.field',np.real(bsol))
            np.savetxt('imag_magnetic.field',np.imag(bsol))
 
        toc2 = timer()
        print('Solve done in',toc2-t_start,'seconds')
    
    # ------------------------------------------------------------------ done
    return 0
    
    
    
if __name__ == "__main__": 
    sys.exit(main())
