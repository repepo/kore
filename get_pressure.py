#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as ss
import numpy.polynomial.chebyshev as ch

import sys
sys.path.insert(1,'bin/')
import utils as ut
import parameters as par

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

a = np.loadtxt('real_flow.field',usecols=solnum)
b = np.loadtxt('imag_flow.field',usecols=solnum)

if m > 0 :
    symm1 = symm
    if symm == 1:
        m_top = m
        m_bot = m+1						# equatorially symmetric case (symm=1)
        lmax_top = lmax
        lmax_bot = lmax+1
    elif symm == -1:
        m_top = m+1
        m_bot = m				# equatorially antisymmetric case (symm=-1)
        lmax_top = lmax+1
        lmax_bot = lmax
elif m == 0 :
    symm1 = -symm 
    if symm == 1:
        m_top = 2
        m_bot = 1						# equatorially symmetric case (symm=1)
        lmax_top = lmax+2
        lmax_bot = lmax+1
    elif symm == -1:
        m_top = 1
        m_bot = 2				# equatorially antisymmetric case (symm=-1)
        lmax_top = lmax+1
        lmax_bot = lmax+2

ll = np.arange(m_top,lmax_top,2)
L = ss.diags(ll*(ll+1),0)
    
Plj0 = a[:n] + 1j*b[:n] 		#  N elements on each l block
Tlj0 = a[n:n+n] + 1j*b[n:n+n] 	#  N elements on each l block

Plj0  = np.reshape(Plj0,(int((lmax-m+1)/2),ut.N1))
Tlj0  = np.reshape(Tlj0,(int((lmax-m+1)/2),ut.N1))

Plj = np.zeros((int((lmax-m+1)/2),N),dtype=complex)
Tlj = np.zeros((int((lmax-m+1)/2),N),dtype=complex)

if ricb == 0 :
    iP = (m + 1 - ut.s)%2
    iT = (m + ut.s)%2
    for k in np.arange(int((lmax-m+1)/2)) :
        Plj[k,iP::2] = Plj0[k,:]
        Tlj[k,iT::2] = Tlj0[k,:]
else :
    Plj = Plj0
    Tlj = Tlj0

dPlj = np.zeros(np.shape(Plj),dtype=complex)
rdPlj = np.zeros(np.shape(Plj),dtype=complex)
rQlj = np.zeros(np.shape(Plj),dtype=complex)
rSlj = np.zeros(np.shape(Plj),dtype=complex)
rTlj = np.zeros(np.shape(Plj),dtype=complex)

ir2Plj = np.zeros(np.shape(Plj),dtype=complex)
irdPlj = np.zeros(np.shape(Plj),dtype=complex)
d2Plj = np.zeros(np.shape(Plj),dtype=complex)
rd3Plj = np.zeros(np.shape(Plj),dtype=complex)

tol = 1e-9
# Chebyshev coefficients of powers of r
r0  = ut.chebco(0, par.N, tol, par.ricb, ut.rcmb)
r1  = ut.chebco(1, par.N, tol, par.ricb, ut.rcmb)
r2  = ut.chebco(2, par.N, tol, par.ricb, ut.rcmb)
# r3  = ut.chebco(3, par.N, tol, par.ricb, ut.rcmb)
# r4  = ut.chebco(4, par.N, tol, par.ricb, ut.rcmb)
# r5  = ut.chebco(5, par.N, tol, par.ricb, ut.rcmb)
# r6  = ut.chebco(6, par.N, tol, par.ricb, ut.rcmb)

# # Chebyshev coefficients of powers of 1/r
ir = ut.chebco_f(lambda r:1/r,N,ricb,rcmb,1e-9)
ir2 = ut.chebco_f(lambda r:1/r**2,N,ricb,rcmb,1e-9)

for k in range(np.size(ll)):
    dPlj[k,:] = ut.Dcheb(Plj[k,:], ricb, rcmb)
    rdPlj[k,:] = ut.cheb2Product(r1,dPlj[k,:],tol)
    rQlj[k,:] = ll[k]*(ll[k]+1)*Plj[k,:]
    rSlj[k,:] = (rdPlj[k,:]+Plj[k,:])
    rTlj[k,:] = ut.cheb2Product(r1,Tlj[k,:],tol)
    # d2Plj[k,:] = ut.Dcheb(dPlj[k,:], ricb, rcmb)
    if par.Ek > 0:
        ir2Plj[k,:] = ut.cheb2Product(ir2,Plj[k,:],tol)
        irdPlj[k,:] = ut.cheb2Product(ir,dPlj[k,:],tol)
        d2Plj[k,:] = ut.Dcheb(dPlj[k,:], ricb, rcmb)
        rd3Plj[k,:] = ut.cheb2Product(r1,ut.Dcheb(d2Plj[k,:], ricb, rcmb),tol)


Tlr = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)

# np.matmul( Plj, chx.T, Plr )
# np.matmul( dPlj, chx.T, dP )
# # np.matmul( d2Plj, chx.T, d2P )
# # np.matmul( d3Plj, chx.T, d3P )

np.matmul( Tlj, chx.T, Tlr )

# rI = ss.diags(r**-1,0)
# rP = Plr * rI
# Qlr = ss.diags(ll*(ll+1),0) * rP
# Slr = rP + dP

# rS = Slr * rI
# rQ = Qlr * rI
# rT = Tlr * rI

# reconstructing the pressure coefficients from the flow's
# note: implemented naively. Should work for equat symmetric solutions only

plj = np.zeros(np.shape(Plj),dtype=complex)
for k in range(np.size(ll)):
    plj[k,:] = -1j*ut.wf*rSlj[k,:]
    plj[k,:] += +2*1j*m/(ll[k]*(ll[k]+1))*(rQlj[k,:]+rSlj[k,:])
    if ll[k] > m:
        plj[k,:] += -2*(ll[k]-1)/(2*ll[k]-1)/ll[k]*np.sqrt((ll[k]-m)*(ll[k]+m))*rTlj[k-1,:]
    if ll[k] < lmax:
        plj[k,:] += -2*(ll[k]+2)/(2*ll[k]+3)/(ll[k]+1)*np.sqrt((ll[k]+m+1)*(ll[k]-m+1))*rTlj[k,:]
    if par.Ek > 0:
        plj[k,:] += par.Ek * (rd3Plj[k,:] + 3*d2Plj[k,:] - (ll[k]*(ll[k]+1)) * irdPlj[k,:] + (ll[k]*(ll[k]+1)) * ir2Plj[k,:])

plr = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
np.matmul( plj, chx.T, plr )

np.savetxt('real_pressure.field',np.real(plj).flatten())
np.savetxt('imag_pressure.field',np.imag(plj).flatten())

