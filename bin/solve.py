#!/usr/bin/env python3
'''
kore solver script. Writes solutions to disk.

To use, optionally export extra runtime solver options:
> export opts='...'

and then execute:
> mpiexec -n ncpus ./bin/solve.py $opts

Solver options are taken from par.petsc_opts (see params_default.py); options given on the
command line take precedence.

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



def load_rhs(fname):
    '''
    This rank's rows of the right-hand sides in fname (B_forced.npz, one per column), as a dense array, and their
    number: memory-mapped like A, so that a large rsf_nsamp is not read in full by every rank.
    '''
    ncol = int(ut.load_npz_mmap(fname)['shape'][1])
    n, Istart, Iend, indptr, indices, data = load_rows(fname)
    return ss.csr_matrix((data, indices, indptr), shape=(Iend-Istart, ncol)).toarray(), ncol


def load_mat(fname):
    '''
    Reads a CSR matrix from an .npz file written by assemble.py and returns it as a
    distributed PETSc matrix. The file is memory-mapped, so each rank reads only its own rows.
    '''
    return to_mat(*load_rows(fname))


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


# Ruiz pre-scaling of the eigenvalue problem, used when par.prescale = 1 (or None, automatic: see params_default.py).
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


def normalise_energy(vecs):
    '''
    In place: scales each eigenvector (one per column) so that its kinetic energy ∫ ½ ρ 𝐮⋅𝐮 dV is 1,
    with KE as spin_doctor computes it (from ricb to rcmb, Gauss-Legendre quadrature, rho0 weight).
    The other fields (thermal, ...) are scaled along. The complex phase is left as SLEPc returns it.
    Vectors with zero KE (or runs without flow) are left unchanged.
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
        usol2 = upp.expand_reshape_sol(split_fields(vecs[:, i])['flow'], par.symm)
        KE = upp.kinetic_energy(usol2, par.ricb, ut.rcmb, ll[rank::size])
        KE = comm.allreduce(KE, op=MPI.SUM)
        if KE > 0:
            vecs[:, i] /= np.sqrt(KE)
    PETSc.Sys.Print('Eigenvectors normalised to kinetic energy = 1 in', round(timer()-t0, 2), 'seconds')


def split_fields(vec):
    '''
    Splits solution vector(s) vec (one solution per column) into the individual fields,
    in the order the unknowns are stacked in A and B (see ut.sizmat).
    Returns a dict {field name: block of rows}, only for the fields present.
    '''
    blocks = [ ('flow',    2*ut.n),
               ('thermal', ut.n * par.thermal) ]
    fields = {}
    offset = 0
    for name, size in blocks:
        if size > 0:
            fields[name] = vec[ offset : offset + size ]
        offset += size
    return fields


def forced_scan(t_start):
    '''
    Frequency scan of the random Reynolds-stress forcing (forcing = 11 with par.rsf_frequencies). For each frequency w
    the forced matrix A - i (w - forcing_frequency) B is factorised once (A.npz is assembled at forcing_frequency,
    B.npz is the mass matrix, not normalised; both from assemble.py) and solved for every column (realisation) of
    B_forced.npz. Appends one row per frequency, as it goes, to
      rsf_spectrum.dat  w, mean KE over the realisations and its standard error (then, with rsf_split, the same for the
                        kinetic energy below r = rsf_split)
      rsf_ke.dat        w, the KE of each realisation (then, with rsf_split, each KE below rsf_split)
    after a '#' line with nsamp, seed and rsf_split for each scan (np.loadtxt skips it). KE is spin_doctor's kinetic
    energy (utils4pp.kinetic_energy). Its mean is the two-sided power spectral density of the kinetic energy, so for
    m != 0 int (mean KE) dw/2pi is the mean kinetic energy of the randomly forced flow (kore units); for m = 0 (a real
    flow, even spectrum) it is twice that. Both for the solved symm class only. No fields are written.
    '''
    import utils4pp as upp
    from mpi4py import MPI
    Print = PETSc.Sys.Print
    comm = PETSc.COMM_WORLD.tompi4py()
    rank, size = comm.Get_rank(), comm.Get_size()

    Arows = load_rows('A.npz')
    Brows = load_rows('B.npz')
    n, Istart, Iend = Arows[:3]
    csr = lambda X: ss.csr_matrix((X[5], X[4], X[3]), shape=(Iend-Istart, n))
    Aloc, Bloc = csr(Arows), csr(Brows)
    del Arows, Brows
    b0, nrhs = load_rhs('B_forced.npz')  # this rank's rows, one realisation per column
    ll = ut.ell(par.m, par.lmax, par.symm)[2]
    rs = getattr(par, 'rsf_split', None)
    freqs = np.atleast_1d(np.asarray(par.rsf_frequencies, dtype=float))
    Print('Frequency scan of the random forcing:', freqs.size, 'frequencies,', nrhs, 'realisations each')
    if rank == 0:   # a header line for each scan: rows of different scans may differ in columns
        for fname, head in (('rsf_spectrum.dat', 'w  mean_KE  stderr' + ('  mean_KE_below_split  stderr' if rs is not None else '')),
                            ('rsf_ke.dat', 'w  KE of each realisation' + ('  then KE below split' if rs is not None else ''))):
            with open(fname, 'a') as f:
                f.write('# %s   (forcing = 11, nsamp %d, seed %d, rsf_split %s)\n' % (head, nrhs, par.rsf_seed, rs))

    vec = np.empty(n, dtype=complex)
    for w in freqs:
        t_w = timer()
        M = (Aloc - 1j*(w - ut.wf)*Bloc).tocsr()
        M.sort_indices()
        MA = to_mat(n, Istart, Iend, M.indptr, M.indices, M.data)
        del M
        x, bvec = MA.createVecs()
        K = PETSc.KSP()
        K.create(PETSc.COMM_WORLD)
        K.setOperators(MA)
        K.setTolerances(rtol=par.tol,max_it=par.maxit)
        K.setFromOptions()
        tozero, V = PETSc.Scatter.toZero(x)
        ke = np.zeros(nrhs)
        ke_in = np.zeros(nrhs)
        for j in range(nrhs):
            bvec.setValues(range(Istart,Iend), b0[:, j])
            bvec.assemblyBegin()
            bvec.assemblyEnd()
            K.solve(bvec, x)
            tozero.scatter(x, V)
            if rank == 0:
                vec[:] = V.getArray()
            comm.Bcast(vec, root=0)
            usol = upp.expand_reshape_sol(split_fields(vec)['flow'], par.symm)
            ke[j] = comm.allreduce(upp.kinetic_energy(usol, par.ricb, ut.rcmb, ll[rank::size]), op=MPI.SUM)
            if rs is not None:
                ke_in[j] = comm.allreduce(upp.kinetic_energy(usol, par.ricb, rs, ll[rank::size]), op=MPI.SUM)
        tozero.destroy(); V.destroy(); K.destroy(); MA.destroy(); x.destroy(); bvec.destroy()

        se = lambda a: np.std(a, ddof=1)/np.sqrt(a.size) if a.size > 1 else np.nan
        row = [w, ke.mean(), se(ke)] + ([ke_in.mean(), se(ke_in)] if rs is not None else [])
        if rank == 0:
            with open('rsf_spectrum.dat', 'ab') as f:
                np.savetxt(f, np.array([row]), fmt='%.12e')
            with open('rsf_ke.dat', 'ab') as f:
                np.savetxt(f, np.concatenate([[w], ke] + ([ke_in] if rs is not None else []))[None, :], fmt='%.12e')
        Print('w = %.9e   mean KE %.6e +- %.1e%s   (%.1f s)' % (w, row[1], row[2],
              '   below %g: %.6e +- %.1e' % (rs, row[3], row[4]) if rs is not None else '', timer()-t_w))
        if not np.all(np.isfinite(ke)):
            Print('Solver crashed at w = %g, got nan\'s!' % w)

    if rank == 0:
        toc2 = timer()
        print('Solve done in',toc2-t_start,'seconds')
        with open('timing.dat','ab') as dtim:
            np.savetxt(dtim, np.array([toc2-t_start]))
    return 0



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

    # optional Ruiz pre-scaling (par.prescale), eigenvalue problems only
    prescale = getattr(par, 'prescale', 0)
    if prescale is None:  # automatic: Ruiz helps everywhere except thermal runs without entropy diffusion
        prescale = 0 if (par.thermal and par.ThermaD == 0) else 1
    tau = par.rtau + par.itau*1j
    if prescale and par.forcing != 0:
        Print('Note: prescale applies to eigenvalue problems only; not used for this forced problem.')
        prescale = 0
    scan = par.forcing == 11 and getattr(par, 'rsf_frequencies', None) is not None

    # ------------------------------------------------------------------ reads matrix A
    if scan:  # frequency scan of the random forcing: A - i (w - wf) B is built for each w in forced_scan
        return forced_scan(t_start)
    if prescale:  # A and B are read as local rows, rescaled, then built
        t_ps = timer()
        Arows = list(load_rows('A.npz'))
        Brows = list(load_rows('B.npz'))
        dr, dc = ruiz_scaling(Arows, Brows, tau)
        scale_rows(Arows, dr, dc)
        scale_rows(Brows, dr, dc)
        if par.drop_tol > 0:  # drop entries of the scaled A and |tau| B below drop_tol
            for name, X, fac in (('A', Arows, 1.0), ('B', Brows, max(abs(tau), 1.0))):
                rows = np.repeat(np.arange(X[2] - X[1]), np.diff(X[3]))
                keep = np.abs(X[5])*fac >= par.drop_tol
                X[3] = np.r_[0, np.cumsum(np.bincount(rows[keep], minlength=X[2] - X[1]))].astype(X[3].dtype)
                X[4], X[5] = X[4][keep], X[5][keep]
                nnz = PETSc.COMM_WORLD.tompi4py().allreduce(np.array([keep.size, keep.sum()]))
                Print('drop_tol %g: %s nnz %d -> %d' % (par.drop_tol, name, nnz[0], nnz[1]))
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

        E.setTarget(tau)
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

            # every rank gets a copy of the eigenvectors for the energy normalisation
            if rank != 0:
                vecs = np.empty((nb_l, nconv), dtype=complex)
            PETSc.COMM_WORLD.tompi4py().Bcast(vecs, root=0)
            if prescale:  # eigenvectors of the scaled problem -> x = Dc*y
                vecs *= dc[:, None]
            normalise_energy(vecs)  # kinetic energy = 1 for every eigenvector

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




    else: # ------------------------------------------------------------ if forced problem, reads forcing vector(s)

        b0, nrhs = load_rhs('B_forced.npz')  # this rank's rows, one right-hand side per column (forcing = 11: rsf_nsamp)
        x, bvec = MA.createVecs()

        #b.set(0)
        Istart,Iend = bvec.getOwnershipRange()

        # -------------------------------------------------------------- setup, solve & collect
        K = PETSc.KSP()
        K.create(PETSc.COMM_WORLD)
        K.setOperators(MA)
        K.setTolerances(rtol=par.tol,max_it=par.maxit)
        K.setFromOptions()

        # solve, one right-hand side after the other: the LU factorisation of the first solve is reused
        tozero, V = PETSc.Scatter.toZero(x)
        VR = np.zeros((nb_l, nrhs), dtype=complex) if rank == 0 else None
        for j in range(nrhs):
            if j == 1:
                t_1 = timer()
            bvec.setValues(range(Istart,Iend), b0[:, j])
            bvec.assemblyBegin()
            bvec.assemblyEnd()
            K.solve(bvec, x)
            tozero.scatter(x, V)  # collect result on rank 0
            if rank == 0:
                VR[:, j] = V.getArray()
        if nrhs > 1:
            Print(nrhs, 'right-hand sides; after the first solve (with the LU factorisation)',
                  round((timer()-t_1)/(nrhs-1), 3), 'seconds per solve')
        del b0

        # cleanup
        tozero.destroy()
        V.destroy()
        K.destroy()
        MA.destroy()
        bvec.destroy()
        x.destroy()

        if rank == 0:

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
                np.savetxt('real_'+name+'.field', np.real(blk))
                np.savetxt('imag_'+name+'.field', np.imag(blk))

        toc2 = timer()
        print('Solve done in',toc2-t_start,'seconds')
        with open('timing.dat','ab') as dtim:
                    np.savetxt(dtim, np.array([toc2-t_start]))  # total solve time, incl. imports

    # ------------------------------------------------------------------ done
    return 0



if __name__ == "__main__":
    sys.exit(main())


