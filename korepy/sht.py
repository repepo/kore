import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.tri as tri
import numpy.polynomial.chebyshev as ch
import sys
sys.path.append('/home/ankit/kore/bin')

import utils as ut

import shtns

class sol:
	def __init__(self,solnum,lmax,m,symm,N,Ek,ricb,rcmb,n,nr,ntheta,nphi):

		self.solnum = solnum
		self.lmax   = lmax
		self.m      = m
		self.symm   = symm
		self.N      = N
		self.Ek     = Ek
		self.ricb   = ricb
		self.rcmb   = rcmb
		self.n      = n
		self.nr     = nr
		self.ntheta = ntheta
		self.nphi   = nphi

	def get_sol(self,datDir):

		gap = self.rcmb-self.ricb
		r = np.linspace(self.ricb,self.rcmb,self.nr)

		x = 2.*(r-self.ricb)/gap - 1. 

		chx = ch.chebvander(x,self.N-1) # this matrix has nr rows and N-1 cols

		a = np.loadtxt(datDir+'real_flow.field',usecols=self.solnum)
		b = np.loadtxt(datDir+'imag_flow.field',usecols=self.solnum)

		if self.m > 0 :
			symm1 = self.symm
			if self.symm == 1:
				m_top = self.m
				m_bot = self.m+1						# equatorially self.symmetric case (self.symm=1)
				lmax_top = self.lmax
				lmax_bot = self.lmax+1
			elif self.symm == -1:
				m_top = self.m+1
				m_bot = self.m				# equatorially antiself.symmetric case (self.symm=-1)
				lmax_top = self.lmax+1
				lmax_bot = self.lmax
		elif self.m == 0 :
			symm1 = -self.symm 
			if self.symm == 1:
				m_top = 2
				m_bot = 1						# equatorially self.symmetric case (self.symm=1)
				lmax_top = self.lmax+2
				lmax_bot = self.lmax+1
			elif self.symm == -1:
				m_top = 1
				m_bot = 2				# equatorially antiself.symmetric case (self.symm=-1)
				lmax_top = self.lmax+1
				lmax_bot = self.lmax+2

		if self.ricb == 0:
			r = r[1:]
			self.nr = self.nr - 1
			
		gap = self.rcmb-self.ricb
		self.r = np.linspace(self.ricb,self.rcmb,self.nr)
		x = 2.*(self.r-self.ricb)/gap - 1.

		# matrix with Chebishev polynomials at every x point for all degrees:
		chx = ch.chebvander(x,self.N-1) # this matrix has nr rows and N-1 cols
			
		Plj0 = a[:self.n] + 1j*b[:self.n] 		#  N elements on each l block
		Tlj0 = a[self.n:self.n+self.n] + 1j*b[self.n:self.n+self.n] 	#  N elements on each l block

		Plj  = np.reshape(Plj0,(int((self.lmax-self.m+1)/2),self.N))
		Tlj  = np.reshape(Tlj0,(int((self.lmax-self.m+1)/2),self.N))
		dPlj = np.zeros(np.shape(Plj),dtype=complex)

		Plr = np.zeros((int((self.lmax-self.m+1)/2), self.nr),dtype=complex)
		dP  = np.zeros((int((self.lmax-self.m+1)/2), self.nr),dtype=complex)
		rP  = np.zeros((int((self.lmax-self.m+1)/2), self.nr),dtype=complex)
		Qlr = np.zeros((int((self.lmax-self.m+1)/2), self.nr),dtype=complex)
		Slr = np.zeros((int((self.lmax-self.m+1)/2), self.nr),dtype=complex)
		Tlr = np.zeros((int((self.lmax-self.m+1)/2), self.nr),dtype=complex)

		np.matmul( Plj, chx.T, Plr )
		np.matmul( Tlj, chx.T, Tlr )

		#rI = ss.diags(r**-1,0)

		ll = np.arange(m_top,lmax_top,2)
		#L = ss.diags(ll*(ll+1),0)

		for k in range(np.size(ll)):
			dPlj[k,:] = ut.Dcheb(Plj[k,:], self.ricb, self.rcmb)

		np.matmul(dPlj, chx.T, dP)

		rP  = Plr * ss.diags(r**-1,0)
		Qlr = ss.diags(ll*(ll+1),0) * rP
		Slr = rP + dP

		# start index for l. Do not confuse with indices for the Cheb expansion!
		idP = int( (1-symm1)/2 )
		idT = int( (1+symm1)/2 )

		plx = idP+self.lmax-self.m+1
		tlx = idT+self.lmax-self.m+1

		# Set up arrays in SHTns style

		Qtmp = np.zeros([self.lmax - self.m + 1,self.nr],dtype=complex)
		Stmp = np.zeros([self.lmax - self.m + 1,self.nr],dtype=complex)
		Ttmp = np.zeros([self.lmax - self.m + 1,self.nr],dtype=complex)

		Qtmp[idP:plx:2, :] = Qlr
		Stmp[idP:plx:2, :] = Slr
		Ttmp[idT:tlx:2, :] = Tlr


		# SHTns init

		polar_opt = 1e-10

		norm=shtns.sht_schmidt | shtns.SHT_NO_CS_PHASE

		sh = shtns.sht(self.lmax,mmax=self.m,norm=norm,nthreads=1)
		ntheta, nphi = sh.set_grid(self.ntheta, self.nphi, polar_opt=polar_opt)

		S = np.zeros([sh.nlm,self.nr],dtype=complex)
		Q = np.zeros([sh.nlm,self.nr],dtype=complex)
		T = np.zeros([sh.nlm,self.nr],dtype=complex)

		ur     = np.zeros([ntheta,nphi,self.nr]) 
		utheta = np.zeros([ntheta,nphi,self.nr])
		uphi   = np.zeros([ntheta,nphi,self.nr])

		mmask = sh.m == self.m

		Q[mmask, :] = Qtmp 
		S[mmask, :] = Stmp 
		T[mmask, :] = Ttmp 

		for ir in range(self.nr):
			ur[...,ir],utheta[...,ir],uphi[...,ir] = sh.synth(Q[:,ir], S[:,ir], T[:,ir])


		# Transpose to form (nphi, ntheta, self.nr) - old magic style habit

		ur     = np.transpose(ur, (1,0,2))
		utheta = np.transpose(utheta, (1,0,2))
		uphi   = np.transpose(uphi, (1,0,2))

		theta = np.arccos(sh.cos_theta)
		phi   = np.linspace(0.,2*np.pi,nphi)

		return r,theta,phi,ur,utheta,uphi