#!/usr/bin/env python3
'''
kore solver script

To use, first export desired solver options:
> export opts='...'

and then execute:
> mpiexec -n ncpus ./bin/solve.py $opts
'''

import sys
import slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
import scipy.io as sio
import scipy.sparse as ss
from timeit import default_timer as timer
import numpy as np

import parameters as par
import utils as ut
import utils_pp as upp



def main():
    
    Print = PETSc.Sys.Print
    rank = PETSc.COMM_WORLD.getRank()
    size = PETSc.COMM_WORLD.getSize()
    opts = PETSc.Options()
    
    if rank == 0:
        tic = timer()
        
    # ------------------------------------------------------------------ reads matrix A
    A = ut.load_csr('A_ind.npz')
    nb_l,nb_c = A.shape
    
    MA = PETSc.Mat()
    MA.create(PETSc.COMM_WORLD)
    MA.setSizes([nb_l,nb_l])
    MA.setType('mpiaij')
    MA.setFromOptions()
    MA.setUp()
    
    Istart,Iend = MA.getOwnershipRange()
    indptrA = A[Istart:Iend,:].indptr
    indicesA = A[Istart:Iend,:].indices
    dataA = A[Istart:Iend,:].data
    del A
    
    MA.setPreallocationCSR(csr=(indptrA,indicesA))
    MA.setValuesCSR(indptrA,indicesA,dataA)
    MA.assemblyBegin()
    MA.assemblyEnd()
    del indptrA,indicesA,dataA
    # done reading and preparing A
    
    
    # ------------------------------------------------------------ reads right hand side vector
        
    b0 = ut.load_csr('u_ind.npz')
    #b0 = sio.mmread('u_ind.mtx')
    #b0 = b0.tocsr()
    
    x, bvec = MA.createVecs()

    #b.set(0)
    Istart,Iend = bvec.getOwnershipRange()
    bvec.setValues(range(Istart,Iend),b0[Istart:Iend,0].toarray())
    bvec.assemblyBegin()
    bvec.assemblyEnd()
    del b0
    
    # -------------------------------------------------------------- setup, solve & collect
    K = PETSc.KSP()
    K.create(PETSc.COMM_WORLD)
    K.setOperators(MA)
    K.setTolerances(rtol=par.tol,max_it=par.maxit)
    K.setFromOptions()
    
    # solve
    K.solve(bvec, x)
    
    # collect result
    tozero,VR = PETSc.Scatter.toZero(x)
    tozero.begin(x,VR)
    tozero.end(x,VR)
    tozero.destroy()

    # cleanup
    K.destroy()
    MA.destroy()
    bvec.destroy()
    x.destroy()
    
    if rank == 0:
        
        #if par.hydro == 1:
        #    offset = 0            
        #    ru = np.reshape(np.real(VR[ offset : offset + 2*ut.n ]),(-1,1))
        #    iu = np.reshape(np.imag(VR[ offset : offset + 2*ut.n ]),(-1,1))
        
        if par.magnetic == 1:
            #offset = par.hydro * 2*ut.n
            offset = 0
            rb = np.reshape(np.real(VR[ offset : offset + 2*ut.n ]),(-1,1))
            ib = np.reshape(np.imag(VR[ offset : offset + 2*ut.n ]),(-1,1))
            
        #if par.thermal == 1:
        #    offset = par.hydro * 2*ut.n + par.magnetic * 2*ut.n
        #    rtemp = np.reshape(np.real(VR[ offset : offset + ut.n ]),(-1,1))
        #    itemp = np.reshape(np.imag(VR[ offset : offset + ut.n ]),(-1,1))
                
        if np.sum([np.isnan(rb), np.isnan(ib), np.isinf(rb), np.isinf(ib)]) > 0:
            success = 0
            print('Solver crashed, got nan\'s!')
        else:
            success = 1 # got actual numbers ... but it could still be a bad solution ;)
            print('Solution computed')
            np.savetxt('real_magnetic.field',rb)
            np.savetxt('imag_magnetic.field',ib)
            
            toc1 = timer()

        toc2 = timer()
        print('Solve done in',toc2-tic,'seconds')
    
    # ------------------------------------------------------------------ done
    return 0
    
    
    
if __name__ == "__main__": 
    sys.exit(main())

    
