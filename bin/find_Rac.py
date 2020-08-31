#!/usr/bin/env python3

import numpy as np
import os
import sys
from mpi4py import MPI
from scipy.optimize import newton, brentq

opts='-st_type sinvert -eps_error_relative'

mmin = 12
mmax = 12
marr = np.arange(mmin,mmax+1)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ncpus = comm.Get_size()
mperproc = int(len(marr)/ncpus)

comm.scatter(marr,root=0)

Ra = 3.4e5

def get_sigma(Ra,ncpus, opts):
    Ra = 10**Ra
    os.system('sed -i "s/RaUsr/%f/" parameters.py' %Ra)
    os.system('mpiexec -n %d ./assemble.py' %ncpus)    
    os.system('mpiexec -n %d ./solve_nopp.py %s' %(ncpus,opts))
    eig = np.loadtxt('eigenvalues.dat')
    Idx = np.argmax(eig[:,0])
    sigma_c = eig[Idx,0]

    print(sigma_c)

    return sigma_c

# Function copied from SINGE - looks for Ra bounds
def bracket_brentq(f, x1, x2=None, dx=0.3, tol=1e-6, maxiter=200, args=None):
    y1 = f(x1, *args)
    print(y1)
    dx = abs(dx)         # dx must be positive.
    if x2 is not None:   # check that we actually have a bracket
        y2 = get_sigma(x2, *args)
        if y2*y1 > 0:        # we don't have a bracket !!
            if (abs(y2) < abs(y1)):
                x1,y1 = x2,y2        # start from the value closest to the root
            x2 = None        # search needed.
    if x2 is None:        # search for a bracket
        x2 = x1
        if y1>0:  dx = -dx        # up or down ?
        while True:
            x2 += dx
            y2 = f(x2, *args)
            if y2*y1 < 0: break
            print(x2)
            x1,y1 = x2,y2
    # Now that we know that the root is between x1 and x2, we use Brent's method:
    x0 = brentq(f, x1, x2, maxiter=maxiter, xtol=tol, rtol=tol, args=args)
    return x0

for m in range(mperproc):
    mIdx = rank*mperproc + m
    mdir = 'Rac_m' + str(marr[mIdx])
    os.mkdir(mdir)
    print("Rank %d -> directory %d" %(rank,marr[mIdx]))
    os.chdir(mdir)
    os.system('cp ../params_conv.py parameters.py')
    os.system('cp ../assemble.py ../utils*.py ../solve_nopp.py ../submatrices.py ../bc_variables.py .')
    os.system('sed -i "s/mUsr/%d/" parameters.py' %marr[mIdx]) 
    os.system('sed -i "s/RaUsr/%f/" parameters.py' %Ra)
    os.system('./submatrices.py %d' %ncpus)
    Rac = bracket_brentq(get_sigma,np.log10(Ra),args=(ncpus,opts))
    print(10**Rac)