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
    A = ut.load_csr('A.npz')
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
    
    if par.forcing == 0: # --------------------------------------------- if eigenvalue problem, reads matrix B
        
        B = ut.load_csr('B.npz')
        nb_l,nb_c = B.shape
        nbl = opts.getInt('nbl',nb_l)
        
        MB = PETSc.Mat()
        MB.create(PETSc.COMM_WORLD)
        MB.setSizes([nbl,nbl])
        MB.setType('mpiaij')
        MB.setFromOptions()
        MB.setUp()
        
        Istart,Iend = MB.getOwnershipRange()
        indptrB = B[Istart:Iend,:].indptr
        indicesB = B[Istart:Iend,:].indices
        dataB = B[Istart:Iend,:].data
        del B
        
        MB.setPreallocationCSR(csr=(indptrB,indicesB))
        MB.setValuesCSR(indptrB,indicesB,dataB)
        MB.assemblyBegin()
        MB.assemblyEnd()
        del indptrB,indicesB,dataB
        # done reading and preparing B
        
        
        # -------------------------------------------------------------- setup eigenvalue solver
        E = SLEPc.EPS()
        E.create(SLEPc.COMM_WORLD)
        E.setOperators(MA,MB)
        E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        #E.setDimensions(nev,ncv)
        E.setDimensions(par.nev)
        E.setTolerances(par.tol,par.maxit)
        
        wep = par.which_eigenpairs
        if wep == 'LM':
            E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
        elif wep == 'SM':
            E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)
        elif wep == 'LR':
            E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        elif wep == 'SR':
            E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
        elif wep == 'LI':
            E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_IMAGINARY)
        elif wep == 'SI':
            E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_IMAGINARY)
        elif wep == 'TM':
            E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        elif wep == 'TR':
            E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
        elif wep == 'TI':
            E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_IMAGINARY)
        
        E.setTarget(par.tau)
        E.setFromOptions()
        # done setting up solver
    
        E.solve() # ---------------------------------------------------- solve and collect solution
        
        class solution:
            pass  # class intentionally empty, this is just to have sol.*stuff*
            
        sol = solution()
        
        # recover results
        sol.its                   = E.getIterationNumber()
        sol.neps_type             = E.getType()
        sol.tol, sol.maxit        = E.getTolerances()
        sol.nev, sol.ncv, sol.mpd = E.getDimensions()
        sol.nconv                 = E.getConverged()
        sol.k                     = np.zeros((1,sol.nconv),dtype=complex)
        sol.vec                   = np.zeros((nb_l,sol.nconv),dtype=complex)
        sol.tau                   = E.getTarget()
        
        if sol.nconv > 0:
            
            # initialize eigenvector placeholder
            v = MA.createVecLeft()
            
            for i in range(0,sol.nconv):
                # gets eigenvalue and eigenvector for each solution found
                # note that petsc uses complex scalars, so k and v are generally complex
                # no need for separate real and imag (hence the None below)
                k = E.getEigenpair(i, v, None)
                # collects and assembles the vector in the zeroth processor 
                tozero,V = PETSc.Scatter.toZero(v)
                tozero.begin(v,V)
                tozero.end(v,V)
                tozero.destroy()
        
                sol.k[0,i] = k
                if rank == 0:
                    sol.vec[0:,i] = V[0:]
        
            if rank == 0:
            
                # Eigenvalues, 1 row per solution found, column 0 is the real part, column 1 is the imaginary part
                eigval = np.hstack([np.real(sol.k).transpose(),np.imag(sol.k).transpose()])
                
                # Eigenvectors, each column is a solution
                rEigv = np.copy(np.real(sol.vec))
                iEigv = np.copy(np.imag(sol.vec))
                
                # each solution for u has 2*ut.n coeffs
                ru = rEigv[ :2*ut.n, : ]
                iu = iEigv[ :2*ut.n, : ]
                
                # each solution for b has 2*ut.n coefs
                if par.magnetic == 1:
                    offset = 2*ut.n
                    rb = rEigv[ offset : offset + 2*ut.n, : ]
                    ib = iEigv[ offset : offset + 2*ut.n, : ]
                    
                # each solution for the temperature has ut.n coeffs 
                if par.thermal == 1:
                    offset = 2*ut.n + par.magnetic * 2*ut.n 
                    rtemp = rEigv[ offset : offset + ut.n, : ]
                    itemp = iEigv[ offset : offset + ut.n, : ]
                    
                success = sol.nconv
                
        else:
            
            if rank == 0:
                success = 0
                print('No converged solution found')
                np.savetxt('no_conv_solution',[0])
                
        MA.destroy()
        MB.destroy()
                
                
                
                
    else: # ------------------------------------------------------------ if forced problem, reads forcing vector
        
        b0 = ut.load_csr('B_forced.npz')
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
            
            ru = np.reshape(np.real(VR[:2*ut.n]),(-1,1))
            iu = np.reshape(np.imag(VR[:2*ut.n]),(-1,1))
            
            if par.magnetic == 1:
                rb = np.reshape(np.real(VR[2*ut.n:]),(-1,1))
                ib = np.reshape(np.imag(VR[2*ut.n:]),(-1,1))
                    
            if np.sum([np.isnan(ru), np.isnan(iu), np.isinf(ru), np.isinf(iu)]) > 0:
                success = 0
                print('Solver crashed, got nan\'s!')
            else:
                success = 1 # got actual numbers ... but it could still be a bad solution ;)
                print('Solution(s) computed')
        
    #PETSc.COMM_WORLD.Barrier()
    
    
    # ------------------------------------------------------------------ Postprocessing: compute energy, dissipation, etc.
    
    if rank == 0:
        
        if success > 0:
            
            toc1 = timer()
            
            kid = np.zeros((success,7))
            Dint_partial = np.zeros((success,3))
            p2t = np.zeros(success)
            resid1 = np.zeros(success)
            resid2 = np.zeros(success)
            y = np.zeros(success)
            vtorq = np.zeros(success,dtype=complex) 
            
            if par.magnetic == 1:
                ohm = np.zeros((success,4))         
                Dohm_partial = np.zeros((success,3))
                o2v = np.zeros(success)
                
            if par.thermal == 1:
                therm = np.zeros((success,1))   
                
            params = np.zeros((success,21))
            
            thk = np.sqrt(par.Ek)
            R1 = par.ricb + 15*thk
            R2 = ut.rcmb - 15*thk
            R3 = ut.rcmb - 30*thk
            
            print('Ek = 10**{:<8.4f}'.format(np.log10(par.Ek)))
        
            print('Post-processing:')    
            print('--- -------------- -------------- ---------- ---------- ---------- ---------- ----------')
            print('Sol    Damping        Frequency     Resid1     Resid2    ohm2visc    tor2pol   |trq|/A ')
            print('--- -------------- -------------- ---------- ---------- ---------- ---------- ----------')
            
            if par.track_target == 1:  # eigenvalue tracking enabled
                #read target data
                x = np.loadtxt('track_target')
            
            for i in range(success):
                
                #print('Processing solution',i)
                
                a = np.copy(ru[:,i])
                b = np.copy(iu[:,i])
                
                if par.forcing == 0:
                    w = eigval[i,1]
                    sigma = eigval[i,0]
                else:
                    w = ut.wf
                    sigma = 0
                
                #print('a = ',a[:10]/a[0])
                
                kid[i,:] = upp.ken_dis( a, b, par.N, par.lmax, par.m, par.symm, \
                par.ricb, ut.rcmb, par.ncpus, w, par.projection, par.forcing, par.ricb, ut.rcmb)

                # inner core boundary layer
                #k1 = upp.ken_dis( a, b, par.N, par.lmax, par.m, par.symm, \
                #par.ricb, ut.rcmb, par.ncpus, w, par.projection, par.forcing, par.ricb, R1)
                
                # core-mantle boundary layer1
                #k2 = upp.ken_dis( a, b, par.N, par.lmax, par.m, par.symm, \
                #par.ricb, ut.rcmb, par.ncpus, w, par.projection, par.forcing, R2, ut.rcmb)
                
                # core-mantle boundary layer2
                #k3 = upp.ken_dis( a, b, par.N, par.lmax, par.m, par.symm, \
                #par.ricb, ut.rcmb, par.ncpus, w, par.projection, par.forcing, R3, ut.rcmb)
                
                Dint_partial[i,0] = 0 #k1[2]*par.Ek
                Dint_partial[i,1] = 0 #k2[2]*par.Ek
                Dint_partial[i,2] = 0 #k3[2]*par.Ek
                
                KP = kid[i,0]
                KT = kid[i,1]
                p2t[i] = KP/KT
                KE = KP + KT
                #print('')
                #print('K = ',KE)
                #print('')
                Dint = kid[i,2]*par.Ek
                Dkin = kid[i,3]*par.Ek
                
                repow = kid[i,5]
                
                expsol = upp.expand_sol(a+1j*b)
                #print(np.shape(a+1j*b))
                #print(np.shape(ut.gamma_visc(0,0,0)))
                #print(np.shape(expsol))
                vtorq[i] = np.dot(par.Ek*ut.gamma_visc(0,0,0),expsol)
                #print(np.shape(vtorq))
                                
                #err1[i] = abs(-Dint/Dkin -1)
                resid1[i] = abs( Dint + Dkin ) / max( abs(Dint), abs(Dkin) )
                
                if par.track_target == 1:   
                    # compute distance (mismatch) to tracking target
                    y0 = abs( (x[0]-sigma)/sigma )      # damping mismatch
                    y1 = abs( (x[1]-w)/w )              # frequency mismatch
                    y2 = abs( (x[2]-p2t[i])/p2t[i] )    # poloidal to toroidal energy ratio mismatch
                    y[i] = y0 + y1 + y2                 # overall mismatch
                       
                
                if par.magnetic == 1:
                    
                    a = np.copy(rb[:,i])
                    b = np.copy(ib[:,i])
                    
                    ohm[i,:] = upp.ohm_dis( a, b, par.N, par.lmax, par.m, -par.symm, par.ricb, ut.rcmb, par.ncpus, par.ricb, ut.rcmb)
                    # use -symm above because magnetic field has the opposite
                    # symmetry as the flow field --if applied field is antisymm (vertical uniform).
                    
                    o1 = upp.ohm_dis( a, b, par.N, par.lmax, par.m, -par.symm, par.ricb, ut.rcmb, par.ncpus, par.ricb, R1)
                    o2 = upp.ohm_dis( a, b, par.N, par.lmax, par.m, -par.symm, par.ricb, ut.rcmb, par.ncpus, R2, ut.rcmb)
                    o3 = upp.ohm_dis( a, b, par.N, par.lmax, par.m, -par.symm, par.ricb, ut.rcmb, par.ncpus, R3, ut.rcmb)
                    
                    Dohm_partial[i,0] = (o1[2] + o1[3])*par.Le2*par.Em
                    Dohm_partial[i,1] = (o2[2] + o2[3])*par.Le2*par.Em
                    Dohm_partial[i,2] = (o3[2] + o3[3])*par.Le2*par.Em
                    
                    Dohm = (ohm[i,2]+ohm[i,3])*par.Le2*par.Em   
                    o2v[i] = Dohm/Dint
                    
                    if par.track_target == 1:
                        y3 = abs( (x[3]-o2v[i])/o2v[i] )
                        y[i] += y3  #include ohmic to viscous dissipation ratio in the tracking
                
                else:
                    Dohm = 0
                
                
                if par.thermal == 1:
                    atemp = np.copy(rtemp[:,i])
                    btemp = np.copy(itemp[:,i])
                    therm[i,:] = upp.thermal_dis( atemp, btemp, a, b, par.N, par.lmax, par.m, par.symm, par.ricb, ut.rcmb, par.ncpus, par.ricb, ut.rcmb)
                    if par.heating == 'two zone' or par.heating == 'user defined' :
                        Dtemp = therm[i,0]
                        #print('Dtemp=',Dtemp)
                    else:
                        Dtemp = therm[i,0]*par.Brunt**2
                else:
                    Dtemp = 0
                                    
                
                if par.forcing != 0:
                    if repow != 0:                          # body forcing (input power should match total dissipation) Needs rewriting ...
                        resid2[i] = abs( (repow-(Dohm-Dkin))/repow )
                    else:                                   # boundary flow forcing (input power to be implemented)
                        resid2[i] = -1.0    
                elif par.forcing == 0:                      # eigenvalue problem (damping should match total dissipation)
                    resid2[i] = abs( 2*sigma*KE - Dkin - Dtemp + Dohm ) / max( abs(2*sigma*KE), abs(Dkin), abs(Dohm), abs(Dtemp) )
                # Note that above I'm using -Dohm as a replacement of the power associated with the Lorentz force (they don't necessarily match)
                
                # print('Dkin  =' ,Dkin)
                # print('Dint  =' ,Dint)
                # print('Dohm  =' ,Dohm)
                # print('Dtemp =' ,Dtemp)
                # print('2sK   = ',2*sigma*KE)
                # print('resid2 = ',resid2[i])
                
                print('{:2d}   {: 12.9f}   {: 12.9f}   {:8.2e}   {:8.2e}   {:8.2e}   {:8.2e}   {:8.2e}'.format(i, sigma, w, resid1[i], resid2[i], Dohm/Dint, KT/KP, np.abs(vtorq[i])/np.sqrt(KE) ))
                
                
                params[i,:] = np.array([par.Ek, par.m, par.symm, par.ricb, par.bci, par.bco, par.projection, par.forcing, \
                par.forcing_amplitude, par.forcing_frequency, par.magnetic, par.Em, par.Le2, par.N, par.lmax, toc1-tic, par.ncpus, par.tol, par.thermal, par.Prandtl, par.Brunt])
                
            print('--- -------------- -------------- ---------- ---------- ---------- ---------- ----------')
            
            # find closest eigenvalue to tracking target and write to target file
            if (par.track_target == 1)&(par.forcing == 0):
                j = y==min(y)
                if par.magnetic == 1:
                    with open('track_target','wb') as tg:
                        np.savetxt( tg, np.c_[ eigval[j,0], eigval[j,1], p2t[j], o2v[j] ] )
                else:
                    with open('track_target','wb') as tg:
                        np.savetxt( tg, np.c_[ eigval[j,0], eigval[j,1], p2t[j] ] )
                print('Closest to target is solution', np.where(j==1)[0][0])
                if err2[j]>0.1:
                    np.savetxt('big_error', np.c_[err1[j],err2[j]] )
            
            
            # use this when writing the first target, it finds the solution with smallest p2t (useful to track the spin-over mode)
            elif (par.track_target == 2)&(par.forcing == 0):
                # here we select the solution with smallest poloidal to toroidal energy ratio and write the track_target file:
                j = p2t==min(p2t)
                if par.magnetic == 1:
                    with open('track_target','wb') as tg:
                        np.savetxt( tg, np.c_[ eigval[j,0], eigval[j,1], p2t[j], o2v[j] ] )
                else:
                    with open('track_target','wb') as tg:
                        np.savetxt( tg, np.c_[ eigval[j,0], eigval[j,1], p2t[j] ] )
                
                
                
            # ---------------------------------------------------------- write post-processed data and parameters to disk
            #print('Writing results')
            
            with open('params.dat','ab') as dpar:
                np.savetxt(dpar, params, \
                #fmt=['%.9e','%d','%d','%.9e','%d','%d','%d','%d','%.9e','%.9e','%d','%.9e','%.9e','%d','%d','%.2f', '%d', '%.2e'])  
                fmt=['%.9e','%d','%d','%.9e','%d','%d','%d','%d','%.9e','%.9e','%d','%.9e','%.9e','%d','%d','%.2f', '%d', '%.2e', '%d', '%.9e', '%.9e'])
            
            with open('flow.dat','ab') as dflo:
                np.savetxt(dflo, np.c_[kid, Dint_partial, np.real(vtorq), np.imag(vtorq)])
            
            if par.magnetic == 1:
                with open('magnetic.dat','ab') as dmag:
                    np.savetxt(dmag, np.c_[ohm, Dohm_partial])
                    
            if par.thermal ==1:
                with open('thermal.dat','ab') as dtmp:
                    np.savetxt(dtmp, therm) 
    
            if par.forcing == 0:
                with open('eigenvalues.dat','ab') as deig:
                    np.savetxt(deig, eigval)
            
            
            # ---------------------------------------------------------- write solution vector('s) to disk
            
            if par.write_solution == 1: # one solution per columns
                np.savetxt('real_flow.field',ru)
                np.savetxt('imag_flow.field',iu)
                if par.magnetic == 1:
                    np.savetxt('real_magnetic.field',rb)
                    np.savetxt('imag_magnetic.field',ib)
                if par.thermal == 1:
                    np.savetxt('real_temperature.field',rtemp)
                    np.savetxt('imag_temperature.field',itemp)  
                    
                    
                    
        toc2 = timer()
        print('Solve done in',toc2-tic,'seconds')
    
    # ------------------------------------------------------------------ done
    return 0
    
    
    
if __name__ == "__main__": 
    sys.exit(main())

    
