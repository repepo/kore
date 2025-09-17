#!/usr/bin/env python3
'''
kore script to get magnetic field data

To use:
> ./bin/get_magnetic.py ncpus
where ncpus is the number of cores
'''

import sys
sys.path.insert(1,'bin/')
sys.path.insert(1,'koreviz/')
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

    tic = timer()

    success = 1
    i = 0

    ohm = np.zeros((success, 4))
    Dohm_partial = np.zeros((success, 3))
    mtorq = np.zeros(success, dtype=complex)

    rb = np.loadtxt('real_magnetic.field')
    ib = np.loadtxt('imag_magnetic.field')

    rmag = np.copy(rb)
    imag = np.copy(ib)

    ohm[i, :] = upp.ohm_dis(rmag, imag, par.N, par.lmax, par.m, -par.symm, par.ricb, ut.rcmb, par.ncpus, par.ricb,
                            ut.rcmb)

    Dohm_partial[i, 0] = 0  # (o1[2] + o1[3])*par.Le2*par.Em
    Dohm_partial[i, 1] = 0  # (o2[2] + o2[3])*par.Le2*par.Em
    Dohm_partial[i, 2] = 0  # (o3[2] + o3[3])*par.Le2*par.Em

    if np.all(ut.gamma_magnetic() == 0):
        mtorq[i] = 0
    else:
        mtorq[i] = par.Le2 * np.dot(ut.gamma_magnetic(), upp.expand_sol(rmag + 1j * imag))

    with open('magnetic.dat','ab') as dmag:
        np.savetxt(dmag, np.c_[ohm, Dohm_partial, np.real(mtorq), np.imag(mtorq)])

    p = np.loadtxt('params.dat')

    p[10] = par.magnetic
    p[11] = par.Em
    p[12] = par.Le2

    p = p[:24]

    if par.mantle == 'insulator':
        mantle_mag_bc = 0
    elif par.mantle == 'TWA':
        mantle_mag_bc = 1

    params = np.zeros((1,29))
    params[0,:] = np.r_[p, mantle_mag_bc, par.c_cmb, par.c1_cmb, par.mu, ut.B0_norm()[0]]

    with open('params.dat', 'wb') as dpar:
        np.savetxt(dpar, params, fmt=['%.9e', '%d', '%d', '%.9e', '%d', '%d', '%d', '%d', '%.9e', '%.9e', '%d',
                                      '%.9e', '%.9e', '%d', '%d', '%.2f', '%d', '%.2e', '%d', '%.9e', '%.9e',
                                      '%.9e', '%.9e', '%.9e', '%d', '%.9e', '%.9e', '%.9e', '%.9e'])
    toc = timer()

    print('Magnetic field data generated and written to disk in', toc - tic, 'seconds')

if __name__ == "__main__": 
    sys.exit(main())

    
