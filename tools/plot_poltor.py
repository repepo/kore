import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import sys
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.tri as tri
import numpy.polynomial.chebyshev as ch
import cmasher as cmr

sys.path.insert(1,'bin/')

import utils as ut
import parameters as par

from matplotlib import rc
rc('text', usetex=True) 

cmap = 'cool'
cmap2 = cmr.neon_r

'''

Script to plot meridional cuts of the flow
Use as:

python3 plot_poltor.py nsol nR ntheta theta0 theta1

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
if ricb == 0:
    r = r[1:]
    nR = nR - 1
    x = r/gap
else :
    x = 2.*(r-ricb)/gap - 1. 

chx = ch.chebvander(x,par.N-1) # this matrix has nR rows and N-1 cols

#--------------------------------
phi = 0. # select meridional cut
#--------------------------------

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




# matrix with Chebishev polynomials at every x point for all degrees:
#chx = ch.chebvander(x,par.N-1) # this matrix has nR rows and N-1 cols
	
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

pol = np.zeros( (nR)*Ntheta, dtype=complex)
tor = np.zeros( (nR)*Ntheta, dtype=complex)


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
		
		pol[k] = np.dot( Plr[:,kr], ylm[idP:plx:2] )
		ur[k] = np.dot( Qlr[:,kr], ylm[idP:plx:2] )
		#ur2[k] = absolute(dot( Qlm[:,kr], ylm[idP:plx:2] ))**2		

		tmp1 = np.dot(   -(l1[idP:plx:2]+1) * Slr[:,kr]/np.tan(theta[kt]), ylm[idP:plx:2]     )
		tmp2 = np.dot( clm[idP+1:plx+1:2,0] * Slr[:,kr]/np.sin(theta[kt]), ylm[idP+1:plx+1:2] )
		tmp3 = np.dot(                 1j*m * Tlr[:,kr]/np.sin(theta[kt]), ylm[idT:tlx:2]     )
		utheta[k] = tmp1+tmp2+tmp3
		#ut2[k] = absolute(tmp1+tmp2+tmp3)**2
		
		tor[k] = np.dot( Tlr[:,kr], ylm[idT:tlx:2] )
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
#matplotlib.rcParams['image.cmap'] = 'rainbow'

#fig=plt.figure(figsize=(14,5))
fig, ax = plt.subplots(nrows=1,ncols=5,sharey='row',figsize=(13,5))
#fig.subplots_adjust(hspace=0.01)



# ------------------------------------------------------------------- ur
#ax[0,0]=fig.add_subplot(151)
#ax1.set_title(r'$|u_r|$',size=22)
ax[0].text(0.31,0,r'$|\mathbf{\hat r}\cdot\mathbf{u}_0|$',size=17)
im1=ax[0].tricontourf( triang, np.absolute(ur[id_in]), 70, cmap=cmap)
for c in im1.collections:
              c.set_edgecolor('face')   
ax[0].plot(r[0]*np.sin(theta),r[0]*np.cos(theta),'k',lw=0.4)
ax[0].plot(r[-1]*np.sin(theta),r[-1]*np.cos(theta),'k',lw=0.4)
ax[0].plot([0,0], [ r.min(),r.max() ], 'k', lw=0.4)
ax[0].plot([0,0], [ -r.max(),-r.min() ], 'k', lw=0.4)
ax[0].set_aspect('equal')
ax[0].set_axis_off()

cax = ax[0].inset_axes([0.02,-0.55,0.02,1.1],transform=ax[0].transData)
plt.colorbar(im1,aspect=70,ax=ax[0],cax=cax)

# --------------------------------------------------------------- utheta
#ax2=fig.add_subplot(152)
#ax2.set_title(r'$|u_\theta|$',size=22)
ax[1].text(0.31,0,r'$|\mathbf{\hat \theta}\cdot\mathbf{u}_0|$',size=17)
im2=ax[1].tricontourf( triang, np.absolute(utheta[id_in]), 70, cmap=cmap)
for c in im2.collections:
              c.set_edgecolor('face')
ax[1].plot(r[0]*np.sin(theta),r[0]*np.cos(theta),'k',lw=0.4)
ax[1].plot(r[-1]*np.sin(theta),r[-1]*np.cos(theta),'k',lw=0.4)
ax[1].plot([0,0], [ r.min(),r.max() ], 'k', lw=0.4)
ax[1].plot([0,0], [ -r.max(),-r.min() ], 'k', lw=0.4)
ax[1].set_aspect('equal')
ax[1].set_axis_off()

cax = ax[1].inset_axes([0.02,-0.55,0.02,1.1],transform=ax[1].transData)
plt.colorbar(im2,aspect=70,ax=ax[1],cax=cax)

# ----------------------------------------------------------------- uphi
#ax3=fig.add_subplot(153)
#ax3.set_title(r'$|u_\phi|$',size=22)
ax[2].text(0.31,0,r'$|\mathbf{\hat \phi}\cdot\mathbf{u}_0|$',size=17)
im3=ax[2].tricontourf( triang, np.absolute(uphi[id_in]), 70, cmap=cmap)
for c in im3.collections:
              c.set_edgecolor('face')
ax[2].plot(r[0]*np.sin(theta),r[0]*np.cos(theta),'k',lw=0.4)
ax[2].plot(r[-1]*np.sin(theta),r[-1]*np.cos(theta),'k',lw=0.4)
ax[2].plot([0,0], [ r.min(),r.max() ], 'k', lw=0.4)
ax[2].plot([0,0], [ -r.max(),-r.min() ], 'k', lw=0.4)
ax[2].set_aspect('equal')
ax[2].set_axis_off()

cax = ax[2].inset_axes([0.02,-0.55,0.02,1.1],transform=ax[2].transData)
plt.colorbar(im3,aspect=70,ax=ax[2],cax=cax)


# ------------------------------------------------------------------- pol
#ax4=fig.add_subplot(154)
#ax1.set_title(r'$|u_r|$',size=22)
ax[3].text(0.38,0,r'$|\mathcal{P}|$',size=17)
im4=ax[3].tricontourf( triang, np.absolute(pol[id_in]), 70, cmap=cmap2)
#im4=ax4.tricontourf( triang, np.real(pol[id_in]), 70, cmap=cmap)
for c in im4.collections:
              c.set_edgecolor('face')   
ax[3].plot(r[0]*np.sin(theta),r[0]*np.cos(theta),'k',lw=0.4)
ax[3].plot(r[-1]*np.sin(theta),r[-1]*np.cos(theta),'k',lw=0.4)
ax[3].plot([0,0], [ r.min(),r.max() ], 'k', lw=0.4)
ax[3].plot([0,0], [ -r.max(),-r.min() ], 'k', lw=0.4)
ax[3].set_aspect('equal')
ax[3].set_axis_off()

cax = ax[3].inset_axes([0.02,-0.55,0.02,1.1],transform=ax[3].transData)
plt.colorbar(im4,aspect=70,ax=ax[3],cax=cax)



# ----------------------------------------------------------------- tor
#ax5=fig.add_subplot(155)
#ax3.set_title(r'$|u_\phi|$',size=22)
ax[4].text(0.35,0,r'$|\mathcal{T}|$',size=17)
im5=ax[4].tricontourf( triang, np.absolute(tor[id_in]), 70, cmap=cmap2)
#im5=ax5.tricontourf( triang, np.real(tor[id_in]), 70, cmap=cmap)
for c in im5.collections:
              c.set_edgecolor('face')
ax[4].plot(r[0]*np.sin(theta),r[0]*np.cos(theta),'k',lw=0.4)
ax[4].plot(r[-1]*np.sin(theta),r[-1]*np.cos(theta),'k',lw=0.4)
ax[4].plot([0,0], [ r.min(),r.max() ], 'k', lw=0.4)
ax[4].plot([0,0], [ -r.max(),-r.min() ], 'k', lw=0.4)
ax[4].set_aspect('equal')
ax[4].set_axis_off()

cax = ax[4].inset_axes([0.02,-0.55,0.02,1.1],transform=ax[4].transData)
plt.colorbar(im5,aspect=70,ax=ax[4],cax=cax)


# ----------------------------------------------------------------------
plt.tight_layout()
plt.show()


