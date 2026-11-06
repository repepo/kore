#!/usr/bin/env python3
'''
NEW (2026-11-06, from conductive_IC): solves the induction equation alone, for every flow that just_induction.py wrote
(A_ind.npz, u_ind.npz), and writes the induced field as solve.py writes a solution: real_magnetic.field and
imag_magnetic.field (with a conducting inner core also real_magnetic_ic.field and imag_magnetic_ic.field, with a resolved
mantle real_magnetic_mantle.field and imag_magnetic_mantle.field), one column per flow solution, in the order of the flow's
columns.

> mpiexec -n ncpus ./bin/solve_ind.py [petsc options]

The solver options are those of the forced problems in par.petsc_opts (ksp_, pc_, mat_: one LU with MUMPS, reused for
every flow); options on the command line take precedence.
'''
from timeit import default_timer as timer
t_start = timer()
import sys
sys.path.insert(1, 'bin/')
from solve import PETSc, load_mat, load_rhs   # (solve.py initialises slepc4py with the command line)
import numpy as np

from parameters import par
import utils as ut


def main():
    rank = PETSc.COMM_WORLD.getRank()
    opts = PETSc.Options()
    for key, val in getattr(par, 'petsc_opts', {}).items():   # the options of the forced problems, unless given
        if not key.startswith(('eps_', 'st_')) and not opts.hasName(key):
            opts.setValue(key, val)

    MA = load_mat('A_ind.npz')
    b0, nrhs = load_rhs('u_ind.npz')   # this rank's rows, one flow per column
    x, bvec = MA.createVecs()
    Istart, Iend = bvec.getOwnershipRange()
    K = PETSc.KSP()
    K.create(PETSc.COMM_WORLD)
    K.setOperators(MA)
    K.setTolerances(rtol=par.tol, max_it=par.maxit)
    K.setFromOptions()

    tozero, V = PETSc.Scatter.toZero(x)
    sol = np.zeros((MA.getSize()[0], nrhs), dtype=complex) if rank == 0 else None
    for j in range(nrhs):
        bvec.setValues(range(Istart, Iend), b0[:, j])
        bvec.assemblyBegin()
        bvec.assemblyEnd()
        K.solve(bvec, x)
        tozero.scatter(x, V)
        if rank == 0:
            sol[:, j] = V.getArray()
    tozero.destroy(); V.destroy(); K.destroy(); MA.destroy(); bvec.destroy(); x.destroy()

    if rank == 0:
        if not np.all(np.isfinite(sol)):
            print('Solver crashed, got nan\'s!')
            return 1
        blocks = ([('magnetic', sol[:2*ut.n])] + ([('magnetic_ic', sol[2*ut.n:2*ut.n + 2*ut.nic])] if ut.cic else [])
                  + ([('magnetic_mantle', sol[2*ut.n + 2*ut.nic:])] if ut.resolved_mantle else []))   # NEW (2026-11-06)
        for name, blk in blocks:
            np.savetxt('real_' + name + '.field', np.real(blk))
            np.savetxt('imag_' + name + '.field', np.imag(blk))
        print(nrhs, 'induced field(s) computed in', round(timer() - t_start, 2), 'seconds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
