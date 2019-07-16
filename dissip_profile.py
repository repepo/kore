import sys
import numpy as np

import parameters as par
import utils as ut
import utils_pp as upp


sol = 0
n = 200
ncpus = 14
nbl = 50  

thk = sqrt(par.Ek)

Ra = par.ricb
Rb = ut.rcmb

Rb1 = linspace( Ra, Ra + nbl*thk, n)
Rb2 = linspace( Rb - nbl*thk, Rb, n)

eigval = np.loadtxt('eigenvalues.dat')
a = np.loadtxt('real_flow.field',usecols=sol)
b = np.loadtxt('imag_flow.field',usecols=sol)

sigma, w = eigval[sol,:]

kid1 = zeros((n,7))
kid2 = zeros((n,7))


for i in range(n):
	print(i,'/',n)
	kid1[i,:] = upp.ken_dis( a, b, par.N, par.lmax, par.m, par.symm, par.ricb, ut.rcmb, ncpus, w, par.projection, par.forcing, Ra, Rb1[i])
	kid2[i,:] = upp.ken_dis( a, b, par.N, par.lmax, par.m, par.symm, par.ricb, ut.rcmb, ncpus, w, par.projection, par.forcing, Ra, Rb2[i])\
	
#K = kid[:,0]+kid[:,1]
Dkin = kid[:,3]*par.Ek
Dint = kid[:,2]*par.Ek

