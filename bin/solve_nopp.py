#!/usr/bin/env python3
'''
kore solver script. Writes solutions to disk.

To use, first export desired solver options:
> export opts='...'

and then execute:
> mpiexec -n ncpus ./bin/solve_nopp.py $opts

You can use the postprocess.py script after the solutions are written to disk.
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

                if par.hydro == 1:
                    ru = rEigv[ut.offset_u : ut.offset_v + ut.size_v, : ]
                    iu = iEigv[ut.offset_u : ut.offset_v + ut.size_v, : ]

                if par.magnetic == 1:
                    rb = rEigv[ut.offset_f : ut.offset_g + ut.size_g, : ]
                    ib = iEigv[ut.offset_f : ut.offset_g + ut.size_g, : ]

                if par.rotdyn:
                    sol_rotdyn = (rEigv[ut.offset_rotdyn : ut.offset_rotdyn + ut.size_rotdyn, : ]
                                  + 1j*iEigv[ut.offset_rotdyn : ut.offset_rotdyn + ut.size_rotdyn, : ])

                if ut.icflag:
                    rb_ic = rEigv[ut.offset_fic : ut.offset_gic + ut.size_gic, : ]
                    ib_ic = iEigv[ut.offset_fic : ut.offset_gic + ut.size_gic, : ]

                if ut.resolved_mantle:
                    rb_mantle = rEigv[ut.offset_p : ut.offset_q + ut.size_q, : ]
                    ib_mantle = iEigv[ut.offset_p : ut.offset_q + ut.size_q, : ]

                if par.thermal == 1:
                    rtemp = rEigv[ut.offset_h : ut.offset_h + ut.size_h, : ]
                    itemp = iEigv[ut.offset_h : ut.offset_h + ut.size_h, : ]

                if par.compositional == 1:
                    rcomp = rEigv[ut.offset_i : ut.offset_i + ut.size_i, : ]
                    icomp = iEigv[ut.offset_i : ut.offset_i + ut.size_i, : ]

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

            if par.hydro == 1:
                ru = np.reshape(np.real(VR[ut.offset_u : ut.offset_v + ut.size_v]),(-1,1))
                iu = np.reshape(np.imag(VR[ut.offset_u : ut.offset_v + ut.size_v]),(-1,1))

            if par.magnetic == 1:
                rb = np.reshape(np.real(VR[ut.offset_f : ut.offset_g + ut.size_g]),(-1,1))
                ib = np.reshape(np.imag(VR[ut.offset_f : ut.offset_g + ut.size_g]),(-1,1))

                if par.rotdyn:
                    sol_rotdyn = np.reshape(
                        VR[ut.offset_rotdyn : ut.offset_rotdyn + ut.size_rotdyn],(-1,1))

                if ut.icflag:
                    rb_ic = np.reshape(np.real(VR[ut.offset_fic : ut.offset_gic + ut.size_gic]),(-1,1))
                    ib_ic = np.reshape(np.imag(VR[ut.offset_fic : ut.offset_gic + ut.size_gic]),(-1,1))

                if ut.resolved_mantle:
                    rb_mantle = np.reshape(np.real(VR[ut.offset_p : ut.offset_q + ut.size_q]),(-1,1))
                    ib_mantle = np.reshape(np.imag(VR[ut.offset_p : ut.offset_q + ut.size_q]),(-1,1))

            if par.thermal == 1:
                rtemp = np.reshape(np.real(VR[ut.offset_h : ut.offset_h + ut.size_h]),(-1,1))
                itemp = np.reshape(np.imag(VR[ut.offset_h : ut.offset_h + ut.size_h]),(-1,1))

            if par.compositional == 1:
                rcomp = np.reshape(np.real(VR[ut.offset_i : ut.offset_i + ut.size_i]),(-1,1))
                icomp = np.reshape(np.imag(VR[ut.offset_i : ut.offset_i + ut.size_i]),(-1,1))

            if np.sum([np.isnan(ru), np.isnan(iu), np.isinf(ru), np.isinf(iu)]) > 0:
                success = 0
                print('Solver crashed, got nan\'s!')
            else:
                success = 1 # got actual numbers ... but it could still be a bad solution ;)
                print('Solution(s) computed')

    #PETSc.COMM_WORLD.Barrier()


    # ------------------------------------------------------------------ write solution vector(s) to disk

    if rank == 0:

        if success > 0:

            if par.forcing == 0:
                with open('eigenvalues0.dat','wb') as deig:
                    np.savetxt(deig, eigval)

            # one solution per column
            if par.hydro:
                with open('real_flow.field','wb') as dflo1:
                    np.savetxt(dflo1, ru)
                with open('imag_flow.field','wb') as dflo2:
                    np.savetxt(dflo2, iu)

            if par.magnetic:
                with open('real_magnetic.field','wb') as dmag1:
                    np.savetxt(dmag1, rb)
                with open('imag_magnetic.field','wb') as dmag2:
                    np.savetxt(dmag2, ib)

            if par.rotdyn:
                with open('rotdyn.field','wb') as drotdyn:
                    np.savetxt(drotdyn, sol_rotdyn)

            if ut.icflag:
                with open('real_magnetic_ic.field','wb') as dmag1_ic:
                    np.savetxt(dmag1_ic, rb_ic)
                with open('imag_magnetic_ic.field','wb') as dmag2_ic:
                    np.savetxt(dmag2_ic, ib_ic)

            if ut.resolved_mantle:
                with open('real_magnetic_mantle.field','wb') as dmag1_mantle:
                    np.savetxt(dmag1_mantle, rb_mantle)
                with open('imag_magnetic_mantle.field','wb') as dmag2_mantle:
                    np.savetxt(dmag2_mantle, ib_mantle)

            if par.thermal:
                with open('real_temperature.field','wb') as dtemp1:
                    np.savetxt(dtemp1, rtemp)
                with open('imag_temperature.field','wb') as dtemp2:
                    np.savetxt(dtemp2, itemp)

            if par.compositional:
                with open('real_composition.field','wb') as dcomp1:
                    np.savetxt(dcomp1, rcomp)
                with open('imag_composition.field','wb') as dcomp2:
                    np.savetxt(dcomp2, icomp)

        toc2 = timer()
        print('Solve done in',toc2-tic,'seconds')
        with open('timing.dat','ab') as dtim:
                    np.savetxt(dtim, np.array([toc2-tic]))

    # ------------------------------------------------------------------ done
    return 0



if __name__ == "__main__":
    sys.exit(main())

