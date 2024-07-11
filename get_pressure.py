#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as ss
import numpy.polynomial.chebyshev as ch

import sys
sys.path.insert(1,'bin/')
import utils as ut
import parameters as par
import utils4pp as upp

'''

Compute pressure coefficients from the flow coefficients
Use as:

python3 get_pressure.py nsol

nsol   : solution number

'''

solnum = int(sys.argv[1])

lmax = par.lmax
m    = par.m
symm = par.symm
N    = par.N
Ek   = par.Ek
ricb = par.ricb
rcmb = 1
n    = ut.n

nR = int(sys.argv[2]) # number of radial points

gap = rcmb-ricb
r = np.linspace(ricb,rcmb,nR)
if ricb == 0:
    r = r[1:]
    nR = nR - 1
    x = r/gap
else :
    x = 2.*(r-ricb)/gap - 1. 

chx = ch.chebvander(x,par.N-1) # this matrix has nR rows and N-1 cols

ru = np.loadtxt('real_flow.field',usecols=solnum).reshape((2*ut.n,-1))
iu = np.loadtxt('imag_flow.field',usecols=solnum).reshape((2*ut.n,-1))

rflow = np.copy(ru[:,0])
iflow = np.copy(iu[:,0])
# Expand solution
[Plj, Tlj] = upp.expand_reshape_sol( rflow + 1j*iflow, par.symm)
[ lp, lt, ll] = ut.ell(par.m, par.lmax, par.symm)
    

dPlj = np.zeros(np.shape(Plj),dtype=complex)
rdPlj = np.zeros(np.shape(Plj),dtype=complex)
rQlj = np.zeros(np.shape(Plj),dtype=complex)
rSlj = np.zeros(np.shape(Plj),dtype=complex)

ir2Plj = np.zeros(np.shape(Plj),dtype=complex)
irdPlj = np.zeros(np.shape(Plj),dtype=complex)
d2Plj = np.zeros(np.shape(Plj),dtype=complex)
rd3Plj = np.zeros(np.shape(Plj),dtype=complex)

rTlj = np.zeros(np.shape(Tlj),dtype=complex)

tol = 1e-9
# Chebyshev coefficients of powers of r
r0  = ut.chebco(0, par.N, tol, ricb, rcmb)
r1  = ut.chebco(1, par.N, tol, ricb, rcmb)
r2  = ut.chebco(2, par.N, tol, ricb, rcmb)

# Chebyshev coefficients of powers of 1/r
ir = ut.chebco_f(lambda r:1/r,N,ricb,rcmb,tol)
ir2 = ut.chebco_f(lambda r:1/r**2,N,ricb,rcmb,tol)

for k, l in enumerate(lp):
    dPlj[k,:] = ut.Dcheb(Plj[k,:], ricb, rcmb)
    rdPlj[k,:] = ut.cheb2Product(r1,dPlj[k,:],tol)
    rQlj[k,:] = l*(l+1)*Plj[k,:]
    rSlj[k,:] = (rdPlj[k,:]+Plj[k,:])
    if par.Ek > 0:
        ir2Plj[k,:] = ut.cheb2Product(ir2,Plj[k,:],tol)
        irdPlj[k,:] = ut.cheb2Product(ir,dPlj[k,:],tol)
        d2Plj[k,:] = ut.Dcheb(dPlj[k,:], ricb, rcmb)
        rd3Plj[k,:] = ut.cheb2Product(r1,ut.Dcheb(d2Plj[k,:], ricb, rcmb),tol)

for k, l in enumerate(lt):
    rTlj[k,:] = ut.cheb2Product(r1,Tlj[k,:],tol)

# reconstructing the pressure coefficients from the flow's
plj = np.zeros(np.shape(Plj),dtype=complex)
for k, l in enumerate(lp):
    # diagonal terms
    plj[k,:] = -1j*ut.wf*rSlj[k,:]
    plj[k,:] += +2*1j*m/(l*(l+1))*(rQlj[k,:]+rSlj[k,:])
    if par.Ek > 0:
        plj[k,:] += par.Ek * (rd3Plj[k,:] + 3*d2Plj[k,:] - (l*(l+1)) * irdPlj[k,:] + (l*(l+1)) * ir2Plj[k,:])
    # off-diagonal terms
    if l-1 in lt:
        idx = np.searchsorted(lt,l-1) # find the index of l-1 in lt
        plj[k,:] += -2*(l-1)/(2*l-1)/l*np.sqrt((l-m)*(l+m))*rTlj[idx,:]
    if l+1 in lt:
        idx = np.searchsorted(lt,l+1) # find the index of l+1 in lt
        plj[k,:] += -2*(l+2)/(2*l+3)/(l+1)*np.sqrt((l+m+1)*(l-m+1))*rTlj[idx,:]

# switch to spatial domain
plr = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
np.matmul( plj, chx.T, plr )

# save to file
np.savetxt('real_pressure.field',np.real(plj).flatten())
np.savetxt('imag_pressure.field',np.imag(plj).flatten())
