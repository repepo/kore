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
#R1 = float(sys.argv[3]) # From radius R1
#R2 = float(sys.argv[4]) # To radius R2

R1 = rcmb - 30*np.sqrt(Ek)
R2 = rcmb
nR = 1000

w = np.loadtxt('eigenvalues.dat')
if len(w.shape)==1:	
	w = w.reshape((-1,len(w)))

'''
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


m1 = max(m,1)
if m == 0 :
	lmax1 = lmax+1
else:
	lmax1 = lmax 
l1 = np.arange(m1,lmax1+1) # vector with all l's allowed whether for T or P

# start index for l. Do not confuse with indices for the Cheb expansion!
idP = int( (1-symm1)/2 )
idT = int( (1+symm1)/2 )

plx = idP+lmax-m+1
tlx = idT+lmax-m+1

lp = l1[idP:plx:2]	# l's for poloidals
lt = l1[idT:tlx:2]	# l's for toroidals

cp = 4*np.pi*ss.diags(1/(2*lp+1),0)
ct = 4*np.pi*ss.diags(1/(2*lt+1),0)

Lp = ss.diags(lp*(lp+1),0)
Lt = ss.diags(lt*(lt+1),0)
Inv_Lp = ss.diags(1/(lp*(lp+1)),0)
Inv_Lt = ss.diags(1/(lt*(lt+1)),0)



#r = np.linspace(ricb,rcmb,nR)
r = np.linspace(R1,R2,nR)
x1 = 2.*(r-ricb)/(rcmb-ricb) - 1.


# xk are the colocation points, from -1 to 1
#i = np.arange(0,N)
#xk = np.cos( (i+0.5)*np.pi/N )
# global x0
#x = ( (R2-R1)*xk + (R1+R2) - (ricb+rcmb) )/(rcmb-ricb)
#x1 = r_[1,x,-1]

#sqx = np.sqrt(1-xk**2)

# rk are the radial colocation points, from Ra to Rb
# global rk 
#r = 0.5*(rcmb-ricb)*( x1 + 1 ) + ricb

#nR = size(r)




# matrix with Chebishev polynomials at every x point for all degrees:
chx = ch.chebvander(x1,par.N-1) # this matrix has nR rows and N-1 cols
	
Plj0 = a[:n] + 1j*b[:n] 		#  N elements on each l block
Tlj0 = a[n:n+n] + 1j*b[n:n+n] 	#  N elements on each l block

Plj  = np.reshape(Plj0,(int((lmax-m+1)/2),N))
Tlj  = np.reshape(Tlj0,(int((lmax-m+1)/2),N))

d1Plj = np.zeros(np.shape(Plj),dtype=complex)
d2Plj = np.zeros(np.shape(Plj),dtype=complex)
d3Plj = np.zeros(np.shape(Plj),dtype=complex)

d1Tlj = np.zeros(np.shape(Tlj),dtype=complex)
d2Tlj = np.zeros(np.shape(Tlj),dtype=complex)

P0 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
P1 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
P2 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
P3 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)

T0 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
T1 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
T2 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)

Q0 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
S0 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)

np.matmul( Plj, chx.T, P0 )
np.matmul( Tlj, chx.T, T0 )

Inv_r = ss.diags(1/r,0)
r1 = ss.diags(r,0)
r2 = ss.diags(r**2,0)

for k in range(np.size(lp)):
	d1Plj[k,:] = ut.Dcheb(   Plj[k,:], ricb, rcmb)
for k in range(np.size(lp)):
	d2Plj[k,:] = ut.Dcheb( d1Plj[k,:], ricb, rcmb)
for k in range(np.size(lp)):
	d3Plj[k,:] = ut.Dcheb( d2Plj[k,:], ricb, rcmb)

for k in range(np.size(lp)):
	d1Tlj[k,:] = ut.Dcheb(   Tlj[k,:], ricb, rcmb)
for k in range(np.size(lp)):
	d2Tlj[k,:] = ut.Dcheb( d1Tlj[k,:], ricb, rcmb)

np.matmul(d1Plj, chx.T, P1)
np.matmul(d2Plj, chx.T, P2)
np.matmul(d3Plj, chx.T, P3)

np.matmul(d1Tlj, chx.T, T1)
np.matmul(d2Tlj, chx.T, T2)

Q0 = Lp * P0 * Inv_r
Q1 = ( Lp * P1 -   Q0 ) * Inv_r
Q2 = ( Lp * P2 - 2*Q1 ) * Inv_r

S0 = P1 + P0 * Inv_r
S1 = P2 + Inv_Lp * Q1
S2 = P3 + Inv_Lp * Q2


# kinetic energy
# Kp = cp*( abs(Q0)**2 + Lp*abs(S0)**2 ) * r2 
# Kt = ct*( Lt*abs(T0)**2 ) * r2
# K = sum(Kp,0)+sum(Kt,0)

K_pol = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
K_tor = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
Dkin_pol = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
Dkin_tor = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
Dint_pol = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
Dint_tor = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
hv_pol = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
hv_tor = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)


## Poloidal components

for k,l in enumerate(lp):
	
	L = l*(l+1)
	
	q0 = Q0[k,:]; q0c = np.conj(q0)
	q1 = Q1[k,:]; q1c = np.conj(q1)
	q2 = Q2[k,:]; q2c = np.conj(q2)
	
	s0 = S0[k,:]; s0c = np.conj(s0)
	s1 = S1[k,:]; s1c = np.conj(s1)
	s2 = S2[k,:]; s2c = np.conj(s2)
	
	f0 = 4*np.pi/(2*l+1)
	
	# horizontal velocity

	f1 = r**2*L*s0*s0c
	
	hv_pol[k,:] = f0*f1
	
	
	# kinetic energy: (1/2)* u^2
	
	f1 = r**2*(q0*q0c + L*s0*s0c)
	
	K_pol[k,:] = f0*f1
	


	# kinetic energy dissipation: (1/2)* u.nabla^2 u
	
	f1 = L*r**2 * np.real( s0c*s2 + s0*s2c )
	f2 = 2*r*L * np.real( s0c*s1 + s0*s1c )
	f3 = -2*L**2 * np.abs( s0 )**2 - 2*(l**2+l+2) * np.abs( q0 )**2
	f4 = 2*r * np.real( q0c*q1 + q0*q1c ) + r**2 * np.real( q0c*q2 + q0*q2c )
	f5 = 4*L * np.real( q0c*s0 + q0*s0c )
	Dkin_pol[k,:] = (1/2)*f0*( f1+f2+f3+f4+f5 )
	
	#f1 = L*r**2 * s0c*s2
	#f2 = 2 * r * L * s0c*s1
	#f3 = -L**2 *( s0c*s0 ) - (l**2+l+2) * ( q0c*q0 )
	#f4 = 2*r * q0c*q1 + r2 * q0c*q2
	#f5 = 2*L *( q0c*s0 + q0*s0c )
	#Dkin_pol[k,:] = f0*( f1+f2+f3+f4+f5 )
	

	# internal energy dissipation: (symm\nabla u):(symm\nabla u)
	
	#f1 = 2*q1*q1c*r**2 + (-2*q0 + L*s0)*(-2*q0c + L*s0c)
	#f2 = L*(q0 - s0 + r*s1)*(q0c - s0c + r*s1c)
	#f3 = (-1 + l)*L*(2 + l)*(s0*s0c)
	#Dint_pol[k,:] = f0*( f1+f2+f3 )
	
	
	f1 = L*np.abs(q0 + r*s1 - s0)**2
	f2 = 3*np.abs(r*q1)**2
	f3 = L*(l-1)*(l+2)*np.abs(s0)**2
	Dint_pol[k,:] = f0*( f1+f2+f3 )
	
	

## Toroidal components

for k,l in enumerate(lt):
	
	L = l*(l+1)
	
	t0 = T0[k,:]; t0c = np.conj(t0)
	t1 = T1[k,:]; t1c = np.conj(t1)
	t2 = T2[k,:]; t2c = np.conj(t2)
	
	f0 = 4*np.pi/(2*l+1)
	
	# horizontal velocity
	
	f1 = r**2*L*t0*t0c
	hv_tor[k,:] = f0*f1
	
	
	# kinetic energy: (1/2)* u^2
	
	f1 = r**2*(L*t0*t0c)
	
	K_tor[k,:] = f0*f1
	

	# kinetic energy dissipation: (1/2)* u.nabla^2 u

	f1 = -2*L**2 * np.abs( t0 )**2
	f2 = 2*r*L * np.real( t0c*t1 + t0*t1c )
	f3 = L*r**2 * np.real( t0c*t2 + t0*t2c )
	Dkin_tor[k,:] = (1/2)*f0*( f1+f2+f3 )
	
	#f1 = L*r**2 * t0c*t2 
	#f2 = 2*r*L * t0c*t1
	#f3 = -L**2 * t0c*t0
	#Dkin_tor[k,:] = f0*( f1+f2+f3 )
	
	
	# internal energy dissipation: (symm\nabla u):(symm\nabla u)

	#f1 = L*(-t0 + r*t1)*(-t0c + r*t1c)
	#f2 = (-1 + l)*L*(2 + l)*(t0*t0c)
	#Dint_tor[k,:] = f0*( f1+f2 )
	
	
	f1 = L*np.abs( r*t1-t0 )**2
	f2 = L*(l-1)*(l+2)*np.abs( t0 )**2
	Dint_tor[k,:] = f0*( f1+f2 )
	

#K = (pi/N)*(R2-R1)*0.5*sum( sqx*(K_pol[:,1:-1]+K_tor[:,1:-1]) )

#Dkin = (pi/N)*(R2-R1)*0.5*sum( sqx*(Dkin_pol[:,1:-1]+Dkin_tor[:,1:-1]) )*par.Ek
#print('Dkin=',Dkin)
	
#Dint = (pi/N)*(R2-R1)*0.5*sum( sqx*(Dint_pol[:,1:-1]+Dint_tor[:,1:-1]) )*par.Ek
#print('Dint=',Dint)

#err1 = abs(-Dint/Dkin -1)
#print('err1=',err1)

hv = np.sqrt(np.real(sum(hv_pol+hv_tor,0)))


dint = np.real(sum(Dint_pol+Dint_tor,0)*par.Ek)
#rb = r[r>1-20*sqrt(Ek)]
#hb = hv[r>1-20*sqrt(Ek)]
#h0 = rb[hb==max(hb)]
#g0 = r[(dint>=0.01*dint[0])&(r>1-20*sqrt(Ek))][-1]

idx = np.r_[True, hv[1:] > hv[:-1]] & np.r_[hv[:-1] > hv[1:], True]
h0 = r[idx][-1]
g0 = r[(dint>=0.001*dint[-1])][0]



figure()
plot(r,c_[dint,hv])
#plot(r,np.sqrt(np.real(sum(hv_pol+hv_tor,0))),label=r'$|u_h|$')
#plot(r,np.real(sum(Dint_pol+Dint_tor,0)/K),label=r'$D_\mathrm{int}$')
#plot(r,np.real(sum(Dkin_pol+Dkin_tor,0)/K),label=r'$D_\mathrm{kin}$')
#plot(r,np.real(sum(K_pol+K_tor,0)),label=r'$K$')
#yscale('symlog',linthreshy=10,linscaley=0.5)   
#legend()
yscale('log')
show()



print('h0=',h0)
print('g0=',g0)





'''


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

Lp = ss.diags(lp*(lp+1),0)
Lt = ss.diags(lt*(lt+1),0)
Inv_Lp = ss.diags(1/(lp*(lp+1)),0)
Inv_Lt = ss.diags(1/(lt*(lt+1)),0)

r = np.linspace(R1,R2,nR)
x1 = 2.*(r-ricb)/(rcmb-ricb) - 1.


# matrix with Chebishev polynomials at every x point for all degrees:
chx = ch.chebvander(x1,par.N-1) # this matrix has nR rows and N-1 cols


Plj0 = a[:n] + 1j*b[:n] 		#  N elements on each l block
Tlj0 = a[n:n+n] + 1j*b[n:n+n] 	#  N elements on each l block

Plj  = np.reshape(Plj0,(int((lmax-m+1)/2),N))
Tlj  = np.reshape(Tlj0,(int((lmax-m+1)/2),N))



d1Plj = np.zeros(np.shape(Plj),dtype=complex)
d2Plj = np.zeros(np.shape(Plj),dtype=complex)
#d3Plj = np.zeros(np.shape(Plj),dtype=complex)

d1Tlj = np.zeros(np.shape(Tlj),dtype=complex)
d2Tlj = np.zeros(np.shape(Tlj),dtype=complex)

P0 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
P1 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
P2 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
#P3 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)

T0 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
T1 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
#T2 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)

Q0 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
S0 = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)

Inv_r = ss.diags(1/r,0)
r1 = ss.diags(r,0)
r2 = ss.diags(r**2,0)

np.matmul( Plj, chx.T, P0 )
np.matmul( Tlj, chx.T, T0 )


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
	

#M = (pi/N)*(R2-R1)*0.5*sum( sqx*( M_pol[:,1:-1] + M_tor[:,1:-1] ) )
#print('M=',M)

#Dohm = (pi/N)*(R2-R1)*0.5*sum( sqx*( Dohm_pol[:,1:-1] + Dohm_tor[:,1:-1] ) )*par.Le2*par.Em
#print('Dohm=',Dohm)
	
#err2 = abs( 1+(Dohm-Dkin)/(w[solnum,0]*K) )
#print('err2=',err2)


dohm = np.real(sum(Dohm_pol+Dohm_tor,0)*par.Le2*par.Em)


plot(r,dohm)

#plot(r,np.real(sum(Dohm_pol+Dohm_tor,0)*par.Le2*par.Em),label=r'$D_\mathrm{Ohm}$')
#plot(r,np.real(sum(M_pol+M_tor,0)),label=r'$M$')

#yscale('symlog',linthreshy=0.01,linscaley=0.5)   
#legend()
show()















