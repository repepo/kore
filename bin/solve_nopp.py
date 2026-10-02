#!/usr/bin/env python3
'''
kore solver script. Writes solutions to disk.

To use, first export desired runtime solver options:
> export opts='...'

and then execute:
> mpiexec -n ncpus ./bin/solve_nopp.py $opts

You can use the spin_doctor.py script after the solutions are written to disk.
'''

from timeit import default_timer as timer
t_start = timer()  # wall clock from here, before scipy/petsc/utils are imported
import sys
import slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
import scipy.io as sio
import scipy.sparse as ss
import numpy as np

import parameters as par
import utils as ut



def load_mat(fname):
    '''
    Reads a CSR matrix from an .npz file written by assemble.py and returns it as a
    distributed PETSc matrix. The file is memory-mapped, so each rank reads only its own rows.
    '''
    M = ut.load_npz_mmap(fname)
    n = int(M['shape'][0])
    # rows owned by this rank, same default distribution as PETSc vectors
    Istart, Iend = PETSc.Vec().createMPI(n, comm=PETSc.COMM_WORLD).getOwnershipRange()
    indptr = np.array(M['indptr'][Istart:Iend+1])
    lo, hi = indptr[0], indptr[-1]
    indices = np.array(M['indices'][lo:hi])
    data = np.array(M['data'][lo:hi])
    del M
    return PETSc.Mat().createAIJ(size=((Iend-Istart, n), (Iend-Istart, n)),
                                 csr=(indptr-lo, indices, data),
                                 comm=PETSc.COMM_WORLD)


def split_fields(vec):
    '''
    Splits solution vector(s) vec (one solution per column) into the individual fields,
    in the order the unknowns are stacked in A and B (see ut.sizmat).
    Returns a dict {field name: block of rows}, only for the fields present.
    '''
    blocks = [ ('flow',        2*ut.n   * par.hydro),
               ('magnetic',    2*ut.n   * par.magnetic),
               ('rotdyn',      3        * par.rotdyn),
               ('magnetic_ic', 2*ut.nic * ut.icflag),
               ('temperature', ut.n     * par.thermal),
               ('composition', ut.n     * par.compositional) ]
    fields = {}
    offset = 0
    for name, size in blocks:
        if size > 0:
            fields[name] = vec[ offset : offset + size ]
        offset += size
    return fields


def main():

    Print = PETSc.Sys.Print
    rank = PETSc.COMM_WORLD.getRank()
    size = PETSc.COMM_WORLD.getSize()
    opts = PETSc.Options()

    # solver options from parameters.py, unless already given on the command line.
    # eps_/st_ options are for eigenvalue problems, ksp_/pc_/mat_ ones for forced problems
    skip = ('ksp_', 'pc_', 'mat_') if par.forcing == 0 else ('eps_', 'st_')
    for key, val in getattr(par, 'petsc_opts', {}).items():
        if not key.startswith(skip) and not opts.hasName(key):
            opts.setValue(key, val)

    if rank == 0:
        tic = timer()

    # ------------------------------------------------------------------ reads matrix A
    MA = load_mat('A.npz')
    nb_l = MA.getSize()[0]

    if par.forcing == 0: # --------------------------------------------- if eigenvalue problem, reads matrix B

        MB = load_mat('B.npz')


        # -------------------------------------------------------------- setup eigenvalue solver
        E = SLEPc.EPS()
        E.create(SLEPc.COMM_WORLD)
        E.setOperators(MA,MB)
        E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        #E.setDimensions(nev,ncv)
        E.setDimensions(par.nev)
        E.setTolerances(par.tol,par.maxit)

        # 'TM' -> TARGET_MAGNITUDE, 'LR' -> LARGEST_REAL, etc.
        which = { 'L':'LARGEST', 'S':'SMALLEST', 'T':'TARGET' }[par.which_eigenpairs[0]] + '_' + \
                { 'M':'MAGNITUDE', 'R':'REAL', 'I':'IMAGINARY' }[par.which_eigenpairs[1]]
        E.setWhichEigenpairs(getattr(SLEPc.EPS.Which, which))

        E.setTarget(par.tau)
        E.setFromOptions()
        # done setting up solver

        E.solve() # ---------------------------------------------------- solve and collect solution

        nconv = E.getConverged()

        if nconv > 0:

            v = MA.createVecLeft()
            tozero, V = PETSc.Scatter.toZero(v)  # created once, reused for every eigenvector
            eigval = np.zeros((nconv, 2))        # column 0 real part, column 1 imaginary part
            vecs = np.zeros((nb_l, nconv), dtype=complex) if rank == 0 else None

            for i in range(nconv):
                k = E.getEigenpair(i, v)   # complex scalars, so no separate imaginary vector
                tozero.scatter(v, V)       # gather the eigenvector on rank 0
                eigval[i] = k.real, k.imag
                if rank == 0:
                    vecs[:, i] = V.getArray()

            tozero.destroy(); V.destroy(); v.destroy()

            if rank == 0:
                fields = split_fields(vecs)  # each column is a solution
                success = nconv

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

            VR = np.reshape(VR[:], (-1,1))
            if np.all(np.isfinite(VR)):
                success = 1 # got actual numbers ... but it could still be a bad solution ;)
                fields = split_fields(VR)
                print('Solution(s) computed')
            else:
                success = 0
                print('Solver crashed, got nan\'s!')

    #PETSc.COMM_WORLD.Barrier()


    # ------------------------------------------------------------------ write solution vector(s) to disk

    if rank == 0:

        if success > 0:

            if par.forcing == 0:
                with open('eigenvalues0.dat','wb') as deig:
                    np.savetxt(deig, eigval)

            # one solution per column
            for name, blk in fields.items():
                if name == 'rotdyn':  # written as complex numbers
                    np.savetxt('rotdyn.field', blk)
                else:
                    np.savetxt('real_'+name+'.field', np.real(blk))
                    np.savetxt('imag_'+name+'.field', np.imag(blk))

        toc2 = timer()
        print('Solve done in',toc2-t_start,'seconds')
        with open('timing.dat','ab') as dtim:
                    np.savetxt(dtim, np.array([toc2-t_start]))  # total solve_nopp time, incl. imports

    # ------------------------------------------------------------------ done
    return 0



if __name__ == "__main__":
    sys.exit(main())


