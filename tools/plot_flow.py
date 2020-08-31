import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import sys
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.tri as tri
import numpy.polynomial.chebyshev as ch

sys.path.insert(1,'bin/')

import utils as ut
import parameters as par

'''

Script to plot meridional cuts of the flow
Use as:

python3 plot_flow.py nsol nR ntheta theta0 theta1

nsol   : solution number
nR     : number of points in radius
ntheta : number of points in the theta direction
theta0 : starting colatitude
theta1 : final colatitude

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
Ntheta = int(sys.argv[3]) # number of points in the theta direction

gap = rcmb-ricb
r = np.linspace(ricb,rcmb,nR)

x = 2.*(r-ricb)/gap - 1. 

chx = ch.chebvander(x,par.N-1) # this matrix has nR rows and N-1 cols

phi = 0. # select meridional cut

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

if ricb == 0:
	r = r[1:]
	nR = nR - 1
gap = rcmb-ricb
r = np.linspace(ricb,rcmb,nR)
x = 2.*(r-ricb)/gap - 1.

# matrix with Chebishev polynomials at every x point for all degrees:
chx = ch.chebvander(x,par.N-1) # this matrix has nR rows and N-1 cols
	
Plj0 = a[:n] + 1j*b[:n] 		#  N elements on each l block
Tlj0 = a[n:n+n] + 1j*b[n:n+n] 	#  N elements on each l block

Plj  = np.reshape(Plj0,(int((lmax-m+1)/2),N))
Tlj  = np.reshape(Tlj0,(int((lmax-m+1)/2),N))
dPlj = np.zeros(np.shape(Plj),dtype=complex)

Plr = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
dP  = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
rP  = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
Qlr = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
Slr = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)
Tlr = np.zeros((int((lmax-m+1)/2), nR),dtype=complex)

np.matmul( Plj, chx.T, Plr )
np.matmul( Tlj, chx.T, Tlr )

rI = ss.diags(r**-1,0)

ll = np.arange(m_top,lmax_top,2)
L = ss.diags(ll*(ll+1),0)

for k in range(np.size(ll)):
	dPlj[k,:] = ut.Dcheb(Plj[k,:], ricb, rcmb)

np.matmul(dPlj, chx.T, dP)

rP  = Plr * ss.diags(r**-1,0)
Qlr = ss.diags(ll*(ll+1),0) * rP
Slr = rP + dP



theta = np.linspace(float(sys.argv[4])*np.pi/180,float(sys.argv[5])*np.pi/180,Ntheta+2)
theta = theta[1:-1]

s = np.zeros( nR*Ntheta )
z = np.zeros( nR*Ntheta )

#ur2 = np.zeros( (nR)*Ntheta )
#ut2 = np.zeros( (nR)*Ntheta )
#up2 = np.zeros( (nR)*Ntheta )

ur     = np.zeros( (nR)*Ntheta, dtype=complex)
utheta = np.zeros( (nR)*Ntheta, dtype=complex)
uphi   = np.zeros( (nR)*Ntheta, dtype=complex)



#ylm = zeros((lmax-m+1,1),dtype=complex128)
#lv = zeros((lmax-m+1,1))

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


k=0
for kt in range(Ntheta):

	ylm = np.r_[ut.Ylm_full(lmax, m, theta[kt], phi),0]	
	for kr in range(0,nR):
		
		s[k]   = r[kr]*np.sin(theta[kt])
		z[k]   = r[kr]*np.cos(theta[kt])
		
		ur[k] = np.dot( Qlr[:,kr], ylm[idP:plx:2] )
		#ur2[k] = absolute(dot( Qlm[:,kr], ylm[idP:plx:2] ))**2		

		tmp1 = np.dot(   -(l1[idP:plx:2]+1) * Slr[:,kr]/np.tan(theta[kt]), ylm[idP:plx:2]     )
		tmp2 = np.dot( clm[idP+1:plx+1:2,0] * Slr[:,kr]/np.sin(theta[kt]), ylm[idP+1:plx+1:2] )
		tmp3 = np.dot(                 1j*m * Tlr[:,kr]/np.sin(theta[kt]), ylm[idT:tlx:2]     )
		utheta[k] = tmp1+tmp2+tmp3
		#ut2[k] = absolute(tmp1+tmp2+tmp3)**2
		
		tmp1 = np.dot(     (l1[idT:tlx:2]+1) * Tlr[:,kr]/np.tan(theta[kt]), ylm[idT:tlx:2]     )
		tmp2 = np.dot( -clm[idT+1:tlx+1:2,0] * Tlr[:,kr]/np.sin(theta[kt]), ylm[idT+1:tlx+1:2] )
		tmp3 = np.dot(                  1j*m * Slr[:,kr]/np.sin(theta[kt]), ylm[idP:plx:2]     )
		uphi[k] = tmp1+tmp2+tmp3
		#up2[k] = absolute(tmp1+tmp2+tmp3)**2
			
		#uz[k] = ur[k]*cos(theta[kt]) - ut[k]*sin(theta[kt])		
		k=k+1


a = 1.
c = 1.
id_in = np.where((s**2/(a**2)) + (z**2/(c**2)) < 1.)
s1 = s[id_in]
z1 = z[id_in]

triang = tri.Triangulation(s1, z1)
# Mask off unwanted triangles (inner core)
xmid = s1[triang.triangles].mean(axis=1)
x2 = xmid*xmid
ymid = z1[triang.triangles].mean(axis=1)
y2 = ymid*ymid
mask = np.where( (x2 + y2 <= ricb**2), 1, 0)
triang.set_mask(mask)


#matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['image.cmap'] = 'rainbow'

fig=plt.figure(figsize=(16,7))
# ------------------------------------------------------------------- ur
ax1=fig.add_subplot(131)
ax1.set_title(r'$|u_r|$',size=18)
im1=ax1.tricontourf( triang, np.absolute(ur[id_in]), 70)
for c in im1.collections:
              c.set_edgecolor('face')   
ax1.set_aspect('equal')
plt.colorbar(im1)

# --------------------------------------------------------------- utheta
ax2=fig.add_subplot(132)
ax2.set_title(r'$|u_\theta|$',size=18)
im2=ax2.tricontourf( triang, np.absolute(utheta[id_in]), 70)
for c in im2.collections:
              c.set_edgecolor('face')
ax2.set_aspect('equal')
plt.colorbar(im2)

# ----------------------------------------------------------------- uphi
ax3=fig.add_subplot(133)
ax3.set_title(r'$|u_\phi|$',size=18)
im3=ax3.tricontourf( triang, np.absolute(uphi[id_in]), 70)
for c in im3.collections:
              c.set_edgecolor('face')
ax3.set_aspect('equal')
plt.colorbar(im3)
# ----------------------------------------------------------------------
plt.tight_layout()
plt.show()
