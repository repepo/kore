import numpy as np
import scipy.sparse as ss
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy.polynomial.chebyshev as ch

sys.path.insert(1,'bin/')

import utils as ut
import parameters as par

'''
Script to plot magnetic field's and electric current density's radial functions
Use as:
python3 tools/electric.py solnum nR R1 R2
solnum  is the solution number
nR number of radial points
R1, R2 are the radial endpoints
'''


solnum = int(sys.argv[1])

lmax = par.lmax
m    = par.m
symm = par.symm
N    = par.N
Ek   = par.Ek
ricb = par.ricb
rcmb = 1

n    = int(N*(lmax-m+1)/2)

nR = int(sys.argv[2]) # number of radial points

R1 = float(sys.argv[3])
R2 = float(sys.argv[4])

reu = np.loadtxt('real_flow.field',usecols=solnum)
imu = np.loadtxt('imag_flow.field',usecols=solnum)

reb = np.loadtxt('real_magnetic.field',usecols=solnum)
imb = np.loadtxt('imag_magnetic.field',usecols=solnum)

		
Plj0 = reu[:n] + 1j*imu[:n] 		#  N elements on each l block
Tlj0 = reu[n:n+n] + 1j*imu[n:n+n] 	#  N elements on each l block
Plj   = np.reshape(Plj0,(-1,N))
Tlj   = np.reshape(Tlj0,(-1,N))

Flj0 = reb[:n] + 1j*imb[:n] 		#  N elements on each l block
Glj0 = reb[n:n+n] + 1j*imb[n:n+n] 	#  N elements on each l block
Flj   = np.reshape(Flj0,(-1,N))
Glj   = np.reshape(Glj0,(-1,N))

lm1 = 2*np.shape(Plj)[0] # this should be =lmax-m+1
lmm = lm1-1

# These are the l-indices for u and b
s = int(symm*0.5+0.5) # s=0 if u is antisymm, s=1 if u is symm
if m>0:
	lup = np.arange( m+1-s, m+1-s +lmm, 2) # u pol
	lut = np.arange( m+s  , m+s   +lmm, 2) # u tor
	#lbf = np.arange( m+s  , m+s   +lmm, 2) # b pol
	#lbg = np.arange( m+1-s, m+1-s +lmm, 2) # b tor
elif m==0:
	lup = np.arange( 1+s, 1+s +lmm, 2) # u pol
	lut = np.arange( 2-s, 2-s +lmm, 2) # u tor
	#lbf = np.arange( 2-s, 2-s +lmm, 2) # b pol
	#lbg = np.arange( 1+s, 1+s +lmm, 2) # b tor
lbf = lut
lbg = lup	


if ricb == 0:
	r = r[1:]
	nR = nR - 1
	
gap = rcmb-ricb
r = np.linspace(R1,R2,nR)
x = 2.*(r-ricb)/gap - 1. 

# matrix with Chebyshev polynomials at every x point for all degrees:
chx = ch.chebvander(x,par.N-1) # this matrix has nR rows and N-1 cols
	

# The magnetic field 
dFlj  = np.zeros(np.shape(Flj),dtype=complex)
d2Flj = np.zeros(np.shape(Flj),dtype=complex)
dGlj  = np.zeros(np.shape(Glj),dtype=complex)
F   = np.zeros((int(lm1/2), nR),dtype=complex)
F1  = np.zeros((int(lm1/2), nR),dtype=complex)
F2  = np.zeros((int(lm1/2), nR),dtype=complex)
G   = np.zeros((int(lm1/2), nR),dtype=complex)
G1  = np.zeros((int(lm1/2), nR),dtype=complex)

np.matmul( Flj, chx.T, F )
np.matmul( Glj, chx.T, G )

for k in range(np.size(lbf)):
	dFlj[k,:]  = ut.Dcheb( Flj[k,:], ricb, rcmb)
	
for k in range(np.size(lbf)):                                                                                                             
	d2Flj[k,:] = ut.Dcheb(dFlj[k,:], ricb, rcmb)
         
for k in range(np.size(lbg)):                                                                                                             
	dGlj[k,:]  = ut.Dcheb( Glj[k,:], ricb, rcmb)

np.matmul(dFlj , chx.T, F1)
np.matmul(d2Flj, chx.T, F2)
np.matmul(dGlj , chx.T, G1)


# The flow velocity field
dPlj  = np.zeros(np.shape(Plj),dtype=complex)
P   = np.zeros((int(lm1/2), nR),dtype=complex)
P1  = np.zeros((int(lm1/2), nR),dtype=complex)
T   = np.zeros((int(lm1/2), nR),dtype=complex)

np.matmul( Plj, chx.T, P )
np.matmul( Tlj, chx.T, T )

for k in range(np.size(lup)):
	dPlj[k,:]  = ut.Dcheb( Plj[k,:], ricb, rcmb)
np.matmul(dPlj , chx.T, P1)

rI  = ss.diags(r**-1,0)
r2I = ss.diags(r**-2,0)

Lf = ss.diags(lbf*(lbf+1),0)
Lg = ss.diags(lbg*(lbg+1),0)
Lp = ss.diags(lup*(lup+1),0)
Lt = ss.diags(lut*(lut+1),0)


# the current density
jrad = np.zeros(np.shape(G),dtype=complex)
jcon = np.zeros(np.shape(G),dtype=complex)
jtor = np.zeros(np.shape(F),dtype=complex)

jrad = par.Em*  Lg*G*rI
jcon = par.Em*( G1 + G*rI )
jtor = par.Em*( Lf*F*r2I -2*F1*rI -F2 )

u = np.loadtxt('flow.dat')  		# flow data
if len(u.shape)==1:
	u = u.reshape((-1,len(u)))
KP = u[:,0] 						# Poloidal kinetic energy
KT = u[:,1] 						# Toroidal kinetic energy
K  = KP + KT
t2p = KT/KP

A = np.sqrt(K);

matplotlib.rc('text', usetex=True);

plt.figure()
plt.plot(r,abs(sum(jrad,0))/A[solnum],label='radial')
plt.plot(r,abs(sum(jcon,0))/A[solnum],label='consoidal')
plt.plot(r,abs(sum(jtor,0))/A[solnum],label='toroidal')
plt.yscale('log')
plt.title(r'$E_\eta\,\nabla\times {\bf b}$')
plt.xlabel(r'$r$')
plt.legend()

plt.figure()
plt.plot(r,abs(sum(Q,0))/A[solnum],label='radial')
plt.plot(r,abs(sum(S,0))/A[solnum],label='consoidal')
plt.plot(r,abs(sum(T,0))/A[solnum],label='toroidal')
plt.yscale('log')
plt.title(r'Magnetic field $\bf b$')
plt.xlabel(r'$r$')
plt.legend()

'''
plt.figure()
plt.plot(r,abs(F[:3,:]).T)
#plt.plot(r,abs(G[:3,:])/A[solnum])
plt.yscale('log')
plt.title(r'F (poloidal) components, b-field, l=1,3,5')


plt.figure()
#plt.plot(r,abs(F[:3,:])/A[solnum])
plt.plot(r,abs(G[:3,:]).T)
plt.yscale('log')
plt.title(r'G (toroidal) components, b-field, l=2,4,6')
'''

plt.show()















