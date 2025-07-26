#!/usr/bin/env python3

import numpy as np
import os
import sys
from scipy.optimize import brentq
from timeit import default_timer as timer
import datetime

sys.path.insert(1,os.getcwd()+'/bin')
import parameters as par

'''
Script to find the critical Rayleigh number for unstable convection.
Needs a Ra_gap variable in parameters.py.
'''

opts='-st_type sinvert -st_pc_factor_mat_solver_type mumps -mat_mumps_icntl_14 1000 -eps_true_residual -eps_balance twoside'
optsRes = opts + ' -eps_error_relative ::ascii_info_detail'

Ramin = 1.6e6

mmin = 9
mmax = 9

marr = np.arange(mmin,mmax+1)

def runKoreRes(Rac,opts): # Print residuals once Rac is found
    Ra = 10**Rac

    os.system('sed -i "0,/Ra_gap.*/s//Ra_gap=%f/" ./bin/parameters.py' %Ra)
    os.system('mpiexec -n %d ./bin/assemble.py' %par.ncpus)
    os.system('mpiexec -n %d ./bin/solve.py %s' %(par.ncpus,opts))
    # os.system('./bin/postprocess.py')

def get_sigma(Ra,ncpus, opts):
    Ra = 10**Ra

    print("Ra = %e" %Ra,flush=True)

    if Ra in ra_cache:
        return ra_cache[Ra]
    else:
        os.system('sed -i "0,/Ra_gap.*/s//Ra_gap=%f/" ./bin/parameters.py' %Ra)
        os.system('mpiexec -n %d ./bin/assemble.py > /dev/null' %ncpus)
        os.system('mpiexec -n %d ./bin/solve.py %s > /dev/null' %(ncpus,opts))
        eig0 = np.loadtxt('eigenvalues0.dat')
        eig = np.reshape(eig0, (-1, 2))
        Idx = np.argmax(eig[:,0])
        sigma_c = eig[Idx,0]
        ra_cache[Ra] = sigma_c

        os.remove('eigenvalues0.dat')

        return sigma_c


# Function copied from SINGE - looks for Ra bounds
def bracket_brentq(f, x1, x2=None, dx=0.01, tol=1e-6, maxiter=200, args=None):
    y1 = f(x1, *args)
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
            x1,y1 = x2,y2
    # Now that we know that the root is between x1 and x2, we use Brent's method:
    x0 = brentq(f, x1, x2, maxiter=maxiter, xtol=tol, rtol=tol, args=args)
    return x0


print("\n",flush=True)
print("###############################",flush=True)
print("# Kore linear convection mode #",flush=True)
print("###############################",flush=True)
print("",flush=True)

tic1 = timer()

# -------------------------------------------------------------------------------- Compute Rac

for m in marr:

    print("=======",flush=True)
    print(" m = %d" %m,flush=True)
    print("=======\n",flush=True)

    ra_cache = {}

    os.system('sed -i "0,/m =.*/s//m = %d/" ./bin/parameters.py' %m)
    os.system('./bin/submatrices.py %d > /dev/null' %par.ncpus)
    Rac = bracket_brentq(get_sigma,np.log10(Ramin),args=(par.ncpus,opts))
    runKoreRes(Rac,optsRes)
    Rac=10**Rac
    eig0 = np.loadtxt('eigenvalues0.dat')
    eig = np.reshape(eig0, (-1, 2))
    idx_c = np.argmax(eig[:,0])
    sigma_c,omega_c = eig[idx_c,:]

    X = np.array([par.Ek , par.ricb , Rac , int(m) ,  omega_c])
    fmt = ['%.3e','%.2f','%.5e','%d','%.5e']

    with open('critical_params.dat','a') as dcrit:
        np.savetxt(dcrit, X.reshape(1,X.shape[0]), fmt=fmt)

    toc2 = timer()

    tsinglem = str(datetime.timedelta(seconds=toc2 - tic1))

    print("Rac for m=%d computed in %s\n\n" %(m, tsinglem))

toc1 = timer()
tform = str(datetime.timedelta(seconds=toc1 - tic1))

print("\n======================================",flush=True)
print("Rac for m=%d to m=%d found in %s" %(mmin,mmax,tform),flush=True)
print("======================================\n\n",flush=True)
