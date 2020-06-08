import numpy as np
import scipy.sparse as ss
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy.polynomial.chebyshev as ch
import utils as ut
import parameters as par

'''

Script to plot meridional cuts of the flow
Use as:

python3 plot_flow.py nsol nR ntheta theta0 theta1

nsol   : solution number
nR     : number of points in radius


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

#nR = int(sys.argv[2]) # number of radial points
#R1 = float(sys.argv[2]) # From radius R1
#R2 = float(sys.argv[3]) # To radius R2

R1 = ricb
R2 = rcmb

a = np.loadtxt('real_magnetic.field',usecols=solnum)
b = np.loadtxt('imag_magnetic.field',usecols=solnum)

bsymm = -par.symm	# applied field is antisymm, so it must have opposite symm as flow field


if m > 0 :
	bsymm1 = bsymm
	if bsymm == 1:
		m_top = m
		m_bot = m+1				# equatorially symmetric case (symm=1)
		lmax_top = lmax
		lmax_bot = lmax+1
	elif bsymm == -1:
		m_top = m+1
		m_bot = m				# equatorially antisymmetric case (symm=-1)
		lmax_top = lmax+1
		lmax_bot = lmax
elif m == 0 :
	bsymm1 = -bsymm 
	if bsymm == 1:
		m_top = 2
		m_bot = 1				# equatorially symmetric case (symm=1)
		lmax_top = lmax+2
		lmax_bot = lmax+1
	elif bsymm == -1:
		m_top = 1
		m_bot = 2				# equatorially antisymmetric case (symm=-1)
		lmax_top = lmax+1
		lmax_bot = lmax+2

lp = np.arange(m_top, lmax_top, 2)
lt = np.arange(m_bot, lmax_bot, 2)

'''
m1 = max(m,1)
if m == 0 :
	lmax1 = lmax+1
else:
	lmax1 = lmax 
l1 = np.arange(m1,lmax1+1) # vector with all l's allowed whether for T or P

# start index for l. Do not confuse with indices for the Cheb expansion!
idP = int( (1-bsymm1)/2 )
idT = int( (1+bsymm1)/2 )

plx = idP+lmax-m+1
tlx = idT+lmax-m+1

lp = l1[idP:plx:2]	# l's for poloidals
lt = l1[idT:tlx:2]	# l's for toroidals
'''


Lp = ss.diags(lp*(lp+1),0)
Lt = ss.diags(lt*(lt+1),0)
Inv_Lp = ss.diags(1/(lp*(lp+1)),0)
Inv_Lt = ss.diags(1/(lt*(lt+1)),0)


#r = np.linspace(ricb,rcmb,nR)
#r = np.linspace(R1,R2,nR)
#x = 2.*(r-ricb)/(rcmb-ricb) - 1.


# xk are the colocation points, from -1 to 1
i = np.arange(0,N)
xk = np.cos( (i+0.5)*np.pi/N )
# global x0
x = ( (R2-R1)*xk + (R1+R2) - (ricb+rcmb) )/(rcmb-ricb)

sqx = np.sqrt(1-xk**2)

# rk are the radial colocation points, from Ra to Rb
# global rk 
r = 0.5*(rcmb-ricb)*( x + 1 ) + ricb

nR = size(r)


# matrix with Chebishev polynomials at every x point for all degrees:
chx = ch.chebvander(x,par.N-1) # this matrix has nR rows and N-1 cols
	
Plj0 = a[:n] + 1j*b[:n] 		#  N elements on each l block
Tlj0 = a[n:n+n] + 1j*b[n:n+n] 	#  N elements on each l block

Plj  = np.reshape(Plj0,(int((lmax-m+1)/2),N))
Tlj  = np.reshape(Tlj0,(int((lmax-m+1)/2),N))

d1Plj = np.zeros(np.shape(Plj),dtype=complex)
d2Plj = np.zeros(np.shape(Plj),dtype=complex)


d1Tlj = np.zeros(np.shape(Tlj),dtype=complex)
d2Tlj = np.zeros(np.shape(Tlj),dtype=complex)

P0 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
P1 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
P2 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)

T0 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
T1 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)

Q0 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
S0 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)

np.matmul( Plj, chx.T, P0 )
np.matmul( Tlj, chx.T, T0 )

Inv_r = ss.diags(1/r,0)

for k in range(np.size(lp)):
	d1Plj[k,:] = ut.Dcheb(   Plj[k,:], ricb, rcmb)
for k in range(np.size(lp)):
	d2Plj[k,:] = ut.Dcheb( d1Plj[k,:], ricb, rcmb)

for k in range(np.size(lp)):
	d1Tlj[k,:] = ut.Dcheb(   Tlj[k,:], ricb, rcmb)
for k in range(np.size(lp)):
	d2Tlj[k,:] = ut.Dcheb( d1Tlj[k,:], ricb, rcmb)

np.matmul(d1Plj, chx.T, P1)
np.matmul(d2Plj, chx.T, P2)

np.matmul(d1Tlj, chx.T, T1)


Q0 = Lp * P0 * Inv_r
Q1 = ( Lp * P1 -   Q0 ) * Inv_r

S0 = P1 + P0 * Inv_r
S1 = P2 + Inv_Lp * Q1

M_pol = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
M_tor = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)

Dohm_pol = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
Dohm_tor = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)



## Poloidal components

for k,l in enumerate(lp):
	
	L = l*(l+1)
	
	q0 = Q0[k,:]
	
	s0 = S0[k,:]
	s1 = S1[k,:]
	
	f0 = 4*np.pi/(2*l+1)
	
	
	# magnetic field energy: (1/2)* b^2
	
	f1 = r**2 * np.abs( q0 )**2
	f2 = r**2*L * np.abs( s0 )**2
	
	M_pol[k,:] = f0*( f1+f2 )
	

	# Ohmic dissipation: 
	
	f1 = L* np.abs( q0 - s0 - r*s1 )**2
	Dohm_pol[k,:] = f0*( f1 )
	


## Toroidal components

for k,l in enumerate(lt):
	
	L = l*(l+1)
	
	t0 = T0[k,:]
	t1 = T1[k,:]
	
	f0 = 4*np.pi/(2*l+1)	
	
	# magnetic field energy: (1/2)* b^2
	
	f1 = (r**2)*L * np.abs( t0 )**2
	M_tor[k,:] = f0*f1
	
	# Ohmic dissipation: 

	f1 = L* np.abs( r*t1 + t0 )**2
	f2 = (L**2) * np.abs( t0 )**2 
	Dohm_tor[k,:] = f0*( f1+f2 )
	

M = (pi/N)*(R2-R1)*0.5*sum( sqx*( M_pol + M_tor ) )
print('M=',M)

Dohm = (pi/N)*(R2-R1)*0.5*sum( sqx*( Dohm_pol + Dohm_tor ) )*par.Le2*par.Em
print('Dohm=',Dohm)
	

figure()
plot(r,np.real(sum(Dohm_pol+Dohm_tor,0)*par.Le2*par.Em),label=r'$D_\mathrm{Ohm}$')
plot(r,np.real(sum(M_pol+M_tor,0)),label=r'$M$')
yscale('symlog',linthreshy=0.01,linscaley=0.5)   
legend()
show()
