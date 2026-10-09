#!/usr/bin/env python3

import numpy as np
import os
import sys
import importlib
from scipy.optimize import brentq
from timeit import default_timer as timer
import datetime

sys.path.insert(1,'bin/')
from parameters import par

'''
Script to find the critical Rayleigh number for unstable convection.
Needs modification to solve.py to write eigenvalues2.dat (with -wb options)
Needs also a Ra_gap variable in parameters.py
'''

opts='-st_type sinvert -st_pc_factor_mat_solver_type mumps -mat_mumps_icntl_14 1000 -eps_true_residual -eps_balance twoside'

ra_cache = {}

Ramin = 2.96e10


def get_sigma(Ra,ncpus, opts):
    Ra = 10**Ra

    print("Ra = %e" %Ra)

    if Ra in ra_cache:
        return ra_cache[Ra]
    else:
        os.system('sed -i "0,/Ra_gap.*/s//Ra_gap=%f/" bin/parameters.py' %Ra)
        os.system('mpiexec -n %d ./bin/assemble.py > out0' %ncpus)
        os.system('mpiexec -n %d ./bin/solve.py %s >> out1' %(ncpus,opts))
        eig0 = np.loadtxt('eigenvalues2.dat')
        eig = np.reshape(eig0, (-1, 2))
        Idx = np.argmax(eig[:,0])
        sigma_c = eig[Idx,0]
        ra_cache[Ra] = sigma_c

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


print("###############################",flush=True)
print("# Kore linear convection mode #",flush=True)
print("###############################",flush=True)
print("",flush=True)
tic1 = timer()
print("=======",flush=True)
print("m = %d" %par.m,flush=True)
print("=======\n",flush=True)

#os.system('sed -i "s/RaUsr/%f/" bin/parameters.py' %Ramin)
os.system('./bin/submatrices.py %d > out00' %par.ncpus)

# -------------------------------------------------------------------------------- Compute Rac
ra_cache = {}

Rac = bracket_brentq(get_sigma,np.log10(Ramin),args=(par.ncpus,opts))
Rac=10**Rac
eig0 = np.loadtxt('eigenvalues2.dat')
eig = np.reshape(eig0, (-1, 2))
idx_c = np.argmax(eig[:,0])
sigma_c,omega_c = eig[idx_c,:]

with open('critical.dat','ab') as dcrit:
	np.savetxt(dcrit, [par.Ek_gap , par.ricb , Rac , int(par.m) , sigma_c , omega_c])	

toc1 = timer()
tform = str(datetime.timedelta(seconds=toc1 - tic1))

print("\n======================================",flush=True)
print("Rac for m=%d found in %s" %(par.m,tform),flush=True)
print("==================================\n\n")
    
