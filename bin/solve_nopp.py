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

from parameters import par
import utils as ut



def load_mat(fname):
    '''
    Reads a CSR matrix from an .npz file written by assemble.py and returns it as a
    distributed PETSc matrix. The file is memory-mapped, so each rank reads only its own rows.
    '''
    return to_mat(*load_rows(fname))


# NEW (2026-10-06): load_mat split into load_rows + to_mat, so that the local rows can be
# rescaled (prescale = 1) before the PETSc matrix is built.
def load_rows(fname):
    '''
    Reads this rank's rows of the CSR matrix in fname (memory-mapped .npz from assemble.py).
    Returns (n, Istart, Iend, indptr, indices, data), with indptr starting at 0.
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
    return n, Istart, Iend, indptr-lo, indices, data


def to_mat(n, Istart, Iend, indptr, indices, data):
    '''
    Builds a distributed PETSc matrix from this rank's rows, as returned by load_rows.
    '''
    return PETSc.Mat().createAIJ(size=((Iend-Istart, n), (Iend-Istart, n)),
                                 csr=(indptr, indices, data),
                                 comm=PETSc.COMM_WORLD)


# NEW (2026-10-06): Ruiz pre-scaling of the eigenvalue problem, used when par.prescale = 1.
def ruiz_scaling(Arows, Brows, tau, iters=10):
    '''
    Ruiz row+column equilibration of M = A - tau*B (infinity norm). Each iteration takes
    r_i = sqrt(max_j |M_ij|), c_j = sqrt(max_i |M_ij|) of the current scaled matrix and sets
    M <- diag(1/r) M diag(1/c). Arows and Brows are this rank's rows (from load_rows).
    Returns Dr (this rank's rows) and Dc (all columns, on every rank) such that diag(Dr) M diag(Dc)
    is the scaled matrix. Works on the local rows only; the column maxima are combined with one
    MPI Allreduce per iteration, so memory stays distributed.
    '''
    from mpi4py import MPI
    comm = PETSc.COMM_WORLD.tompi4py()
    n, Istart, Iend = Arows[:3]
    nloc = Iend - Istart
    csr = lambda X: ss.csr_matrix((X[5], X[4], X[3]), shape=(nloc, n))
    M = (csr(Arows) - tau*csr(Brows)).tocsr()
    a0 = np.abs(M.data)
    ptr, cols = M.indptr, M.indices
    rows = np.repeat(np.arange(nloc), np.diff(ptr))
    del M
    nonempty = ptr[1:] > ptr[:-1]
    dr = np.ones(nloc)
    dc = np.ones(n)
    for k in range(iters):
        a = a0*dr[rows]
        a *= dc[cols]
        r = np.zeros(nloc)
        r[nonempty] = np.maximum.reduceat(a, ptr[:-1][nonempty])
        c = np.zeros(n)
        np.maximum.at(c, cols, a)
        comm.Allreduce(MPI.IN_PLACE, c, op=MPI.MAX)
        del a
        r[r == 0] = 1.0  # empty rows/columns are left alone
        c[c == 0] = 1.0
        dr /= np.sqrt(r)
        dc /= np.sqrt(c)
    return dr, dc


def scale_rows(X, dr, dc):
    '''
    In place: this rank's rows X (from load_rows) become diag(dr) X diag(dc).
    '''
    n, Istart, Iend, indptr, indices, data = X
    rows = np.repeat(np.arange(Iend-Istart), np.diff(indptr))
    X[5] = data * (dr[rows] * dc[indices])  # data may be real (B) or complex (A)


# NEW (2026-10-06): energy normalisation of the eigenvectors, before they are written.
def normalise_energy(vecs):
    '''
    In place: scales each eigenvector (one per column) so that its kinetic + magnetic energy is 1,
    with KE and ME as spin_doctor computes them (outer core only, from ricb to rcmb; the IC field
    and the rotdyn rotation rates are scaled along but their energies are not counted).
    The complex phase is left as SLEPc returns it. Vectors with zero KE + ME are left unchanged.
    Called on all ranks with the full vecs on each; the l-components are shared among the ranks
    and their energies summed with an MPI Allreduce.
    '''
    import utils4pp as upp
    from mpi4py import MPI
    comm = PETSc.COMM_WORLD.tompi4py()
    rank, size = comm.Get_rank(), comm.Get_size()
    ll = ut.ell(par.m, par.lmax, par.symm)[2]
    t0 = timer()
    for i in range(vecs.shape[1]):
        f = split_fields(vecs[:, i])
        usol2 = upp.expand_reshape_sol(f['flow'], par.symm) if 'flow' in f else 0
        bsol2 = upp.expand_reshape_sol(f['magnetic'], ut.bsymm) if 'magnetic' in f else 0
        KE, ME = upp.kin_mag_energy(usol2, bsol2, par.ricb, ut.rcmb, ll[rank::size])
        energy = comm.allreduce(KE + ME, op=MPI.SUM)
        if energy > 0:
            vecs[:, i] /= np.sqrt(energy)
    PETSc.Sys.Print('Eigenvectors normalised to KE + ME = 1 in', round(timer()-t0, 2), 'seconds')


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
    for key, val in par.petsc_opts.items():
        if not key.startswith(skip) and not opts.hasName(key):
            opts.setValue(key, val)

    if rank == 0:
        tic = timer()

    # NEW (2026-10-06): optional Ruiz pre-scaling (par.prescale), eigenvalue problems only
    prescale = par.prescale
    if prescale and par.forcing != 0:
        Print('Note: prescale applies to eigenvalue problems only; not used for this forced problem.')
        prescale = 0

    # ------------------------------------------------------------------ reads matrix A
    if prescale:  # NEW (2026-10-06): A and B are read as local rows, rescaled, then built
        t_ps = timer()
        Arows = list(load_rows('A.npz'))
        Brows = list(load_rows('B.npz'))
        dr, dc = ruiz_scaling(Arows, Brows, par.tau)
        scale_rows(Arows, dr, dc)
        scale_rows(Brows, dr, dc)
        MA = to_mat(*Arows)
        MB = to_mat(*Brows)
        del Arows, Brows, dr
        Print('Ruiz pre-scaling of A - tau*B done in', round(timer()-t_ps, 2), 'seconds '
              '(the residuals printed below are for the scaled problem)')
    else:
        MA = load_mat('A.npz')
    nb_l = MA.getSize()[0]

    if par.forcing == 0: # --------------------------------------------- if eigenvalue problem, reads matrix B

        if not prescale:
            MB = load_mat('B.npz')


        # -------------------------------------------------------------- setup eigenvalue solver
        E = SLEPc.EPS()
        E.create(PETSc.COMM_WORLD)
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

            # NEW (2026-10-06): every rank gets a copy of the eigenvectors for the energy normalisation
            if rank != 0:
                vecs = np.empty((nb_l, nconv), dtype=complex)
            PETSc.COMM_WORLD.tompi4py().Bcast(vecs, root=0)
            if prescale:  # NEW (2026-10-06): eigenvectors of the scaled problem -> x = Dc*y
                vecs *= dc[:, None]
            normalise_energy(vecs)  # NEW (2026-10-06): KE + ME = 1 for every eigenvector

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


