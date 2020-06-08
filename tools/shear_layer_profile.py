import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import sys
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.tri as tri
import numpy.polynomial.chebyshev as ch

import utils as ut
import parameters as par

'''

Script to plot meridional cuts of the flow
Use as:

python3 shear_layer_profile.py nsol z0 nl

nsol: solution number
z0	: profile intersection with z axis
nl	: number of points along profile

it will compute the profile between
-5*E**(1/3) and 5*E**(1/3)
as measured from the characteristic
coming from the inner core critical latitude

'''


solnum = int(sys.argv[1])
z0 = float(sys.argv[2])
nl = int(sys.argv[3])
w0 = ut.wf

lmax = par.lmax
m    = par.m
symm = par.symm
N    = par.N
Ek   = par.Ek
ricb = par.ricb
rcmb = 1
n    = ut.n

gap = rcmb-ricb

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


	
Plj0 = a[:n] + 1j*b[:n] 		#  N elements on each l block
Tlj0 = a[n:n+n] + 1j*b[n:n+n] 	#  N elements on each l block

Plj  = np.reshape(Plj0,(int((lmax-m+1)/2),N))
Tlj  = np.reshape(Tlj0,(int((lmax-m+1)/2),N))
dPlj = np.zeros(np.shape(Plj),dtype=complex)

Plr = np.zeros((int((lmax-m+1)/2), nl),dtype=complex)
dP  = np.zeros((int((lmax-m+1)/2), nl),dtype=complex)
rP  = np.zeros((int((lmax-m+1)/2), nl),dtype=complex)
Qlr = np.zeros((int((lmax-m+1)/2), nl),dtype=complex)
Slr = np.zeros((int((lmax-m+1)/2), nl),dtype=complex)
Tlr = np.zeros((int((lmax-m+1)/2), nl),dtype=complex)

ll = np.arange(m_top,lmax_top,2)
L = ss.diags(ll*(ll+1),0)

for k in range(np.size(ll)):
	dPlj[k,:] = ut.Dcheb(Plj[k,:], ricb, rcmb)


m1 = max(m,1)
if m == 0 :
	lmax1 = lmax+1
else:
	lmax1 = lmax 
l1 = np.arange(m1,lmax1+1) # vector with all l's allowed whether for T or P

clm = np.zeros((lmax-m+2,1))
for i,l in enumerate(l1):
	clm[i] = np.sqrt((l-m)*(l+m))

# start index for l. Do not confuse with indices for the Cheb expansion!
idP = int( (1-symm1)/2 )
idT = int( (1+symm1)/2 )

plx = idP+lmax-m+1
tlx = idT+lmax-m+1


phi = 0. # select meridional cut
theta_c = np.arcsin(w0/2)

#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['image.cmap'] = 'rainbow'

#fig=plt.figure(figsize=(12,5))
#ax1=fig.add_subplot(111)
#ax1.set_title(r'Velocity profile',size=14)

ur     = np.zeros( nl, dtype=complex)
utheta = np.zeros( nl, dtype=complex)
uphi   = np.zeros( nl, dtype=complex)


#for k,z0 in enumerate(np.linspace(za,zb,nz)):
	
l0 = ricb - z0*(w0/2)

la = -5*Ek**(1/3)
lb =  5*Ek**(1/3)

lc = np.linspace(la,lb,nl) + l0

theta = np.array([np.arccos( (z0 + l3*np.sin(theta_c))/np.sqrt(z0**2+l3**2+2*z0*l3*np.sin(theta_c))) for l3 in lc])

r = np.array([z0*(np.cos(tht)+np.sin(tht)*np.tan(tht+theta_c)) for tht in theta])

x = np.array([2.*(rr-ricb)/gap - 1. for rr in r])

chx = ch.chebvander(x,par.N-1) # this matrix has nl rows and N-1 cols


np.matmul( Plj, chx.T, Plr )
np.matmul( Tlj, chx.T, Tlr )

rI = ss.diags(r**-1,0)

np.matmul(dPlj, chx.T, dP)

rP  = Plr * ss.diags(r**-1,0)
Qlr = ss.diags(ll*(ll+1),0) * rP
Slr = rP + dP
	
for j in range(nl):
	
	ylm = np.r_[ut.Ylm_full(lmax, m, theta[j], phi),0]	
		
	ur[j] = np.dot( Qlr[:,j], ylm[idP:plx:2] )	

	tmp1 = np.dot(   -(l1[idP:plx:2]+1) * Slr[:,j]/np.tan(theta[j]), ylm[idP:plx:2]     )
	tmp2 = np.dot( clm[idP+1:plx+1:2,0] * Slr[:,j]/np.sin(theta[j]), ylm[idP+1:plx+1:2] )
	tmp3 = np.dot(                 1j*m * Tlr[:,j]/np.sin(theta[j]), ylm[idT:tlx:2]     )
	utheta[j] = tmp1+tmp2+tmp3
		
	tmp1 = np.dot(     (l1[idT:tlx:2]+1) * Tlr[:,j]/np.tan(theta[j]), ylm[idT:tlx:2]     )
	tmp2 = np.dot( -clm[idT+1:tlx+1:2,0] * Tlr[:,j]/np.sin(theta[j]), ylm[idT+1:tlx+1:2] )
	tmp3 = np.dot(                  1j*m * Slr[:,j]/np.sin(theta[j]), ylm[idP:plx:2]     )
	uphi[j] = tmp1+tmp2+tmp3
		
#ax1.plot(lc-l0,np.absolute(ur[:,k]),'r-',label=r'$|u_r|$')
#ax1.plot(lc-l0,np.absolute(utheta[:,k]),'g-',label=r'$|u_\theta|$')
#ax1.plot(lc-l0,np.absolute(uphi[:,k]),'b-',label=r'$|u_\phi|$')

#ax1.legend()

#plt.tight_layout()
#plt.show()

u = np.absolute(ur)

umax = np.max(u)

print('Max ur =',umax)

k0 = u==umax
dtat = lc[k0]-l0
print('Pos =',dtat[0])

k = u>=0.7*umax
tmp = lc[k]
slw7 = np.max(tmp)-np.min(tmp)
print('Width(0.7) =',slw7)

k = u>=0.8*umax
tmp = lc[k]
slw8 = np.max(tmp)-np.min(tmp)
print('Width(0.8) =',slw8)

k = u>=0.9*umax
tmp = lc[k]
slw9 = np.max(tmp)-np.min(tmp)
print('Width(0.9) =',slw9)

with open('slayer.dat','ab') as sla:
	np.savetxt(sla,np.c_[umax, dtat[0], slw7, slw8, slw9])


