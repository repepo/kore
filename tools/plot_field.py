import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy.polynomial.chebyshev as ch

sys.path.insert(1,'bin/')

import utils as ut
import parameters as par
import utils_pp as upp

'''
Script to plot meridional cuts of a solution field
Use as:

python3 plot_field.py nsol nR ntheta theta0 theta1 field opt

nsol   : solution number
nR     : number of points in radius
ntheta : number of points in the theta direction
theta0 : starting colatitude
theta1 : final colatitude
field  : whether to plot flow velocity or magnetic field ('u' or 'b')
opt    : 'raw' for real part (phase and phi dependent!), or 'abs' for the magnitude
'''

plt.rc('text', usetex=True)

solnum = int(sys.argv[1])

if sys.argv[6] == 'u':
    a0 = np.loadtxt('real_flow.field',usecols=solnum)
    b0 = np.loadtxt('imag_flow.field',usecols=solnum)
    vsymm = par.symm
    if sys.argv[7] == 'raw':
        titlelabels = [r'$\mathbf{\hat r}\cdot\mathrm{Re}\left(\mathbf{u_0}\right)$', \
                       r'$\mathbf{\hat \theta}\cdot\mathrm{Re}\left(\mathbf{u_0}\right)$', \
                       r'$\mathbf{\hat \phi}\cdot\mathrm{Re}\left(\mathbf{u_0}\right)$']
    elif sys.argv[7] == 'abs':
        titlelabels = [r'$\mathbf{\hat r}\cdot\left|\mathbf{u_0}\right|$', \
                       r'$\mathbf{\hat \theta}\cdot\left|\mathbf{u_0}\right|$', \
                       r'$\mathbf{\hat \phi}\cdot\left|\mathbf{u_0}\right|$'] 
    cmap = 'rainbow'
elif sys.argv[6] == 'b':
    a0 = np.loadtxt('real_magnetic.field',usecols=solnum)
    b0 = np.loadtxt('imag_magnetic.field',usecols=solnum)
    if sys.argv[7] == 'raw':
        titlelabels = [r'$\mathbf{\hat r}\cdot\mathrm{Re}\left(\mathbf{b_0}\right)$', \
                       r'$\mathbf{\hat \theta}\cdot\mathrm{Re}\left(\mathbf{b_0}\right)$', \
                       r'$\mathbf{\hat \phi}\cdot\mathrm{Re}\left(\mathbf{b_0}\right)$']
    elif sys.argv[7] == 'abs':
        titlelabels = [r'$\mathbf{\hat r}\cdot\left|\mathbf{b_0}\right|$', \
                       r'$\mathbf{\hat \theta}\cdot\left|\mathbf{b_0}\right|$', \
                       r'$\mathbf{\hat \phi}\cdot\left|\mathbf{b_0}\right|$'] 
    vsymm = ut.bsymm
    cmap = 'plasma'


lmax = par.lmax
m    = par.m
symm = par.symm
N    = par.N
Ek   = par.Ek
ricb = par.ricb
rcmb = 1
n    = ut.n
n0   = ut.n0

nR = int(sys.argv[2]) # number of radial points
Ntheta = int(sys.argv[3]) # number of points in the theta direction

# setup radial grid
gap = rcmb-ricb
r = np.linspace(ricb,rcmb,nR)
if ricb == 0:
	r = r[1:]
	nR = nR - 1
x = upp.xcheb(r,ricb,rcmb) 


phi = 0. # select meridional cut


# matrix with Chebyshev polynomials at every x point for all degrees:
chx = ch.chebvander(x,par.N-1) # this matrix has nR rows and N-1 cols

# expand solution in case ricb=0
aib = upp.expand_sol(a0+1j*b0,vsymm)
a = np.real(aib)
b = np.imag(aib)

Plj0 = a[:n0] + 1j*b[:n0] 		#  N elements on each l block
Tlj0 = a[n0:n0+n0] + 1j*b[n0:n0+n0] 	#  N elements on each l block

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

ll0 = ut.ell(m,lmax,vsymm)
llpol = ll0[0]
lltor = ll0[1]
ll    = ll0[2]

for k in range(np.size(llpol)):
	dPlj[k,:] = ut.Dcheb(Plj[k,:], ricb, rcmb)

np.matmul(dPlj, chx.T, dP)

rP  = Plr * ss.diags(r**-1,0)
Qlr = ss.diags(llpol*(llpol+1),0) * rP
Slr = rP + dP

# setup the latitudinal grid
theta = np.linspace(float(sys.argv[4])*np.pi/180,float(sys.argv[5])*np.pi/180,Ntheta+2)
theta = theta[1:-1]

# Spherical harmonics for all theta at once: rows are theta, columns are l (same layout as ylm in the old loop,
# including the trailing zero)
ylm = np.zeros((Ntheta, lmax-m+2), dtype=complex)
ylm[:,:-1] = ut.Ylm_full_vec(lmax, m, theta, phi)

clm = np.zeros(lmax-m+2)
clm[:np.size(ll)] = np.sqrt((ll-m)*(ll+m))

# start index for l. Do not confuse with indices for the Cheb expansion!
sy = int( vsymm*0.5 + 0.5 ) # sy=0 if antisymm, sy=1 if symm
idP = (np.sign(m)+sy  )%2
idT = (np.sign(m)+sy+1)%2
plx = idP+lmax-m+1
tlx = idT+lmax-m+1

YP  = ylm[:, idP:plx:2]         # Ntheta x (number of poloidal l's)
YP1 = ylm[:, idP+1:plx+1:2]
YT  = ylm[:, idT:tlx:2]
YT1 = ylm[:, idT+1:tlx+1:2]
cot = (1/np.tan(theta))[:,None]
csc = (1/np.sin(theta))[:,None]

# all (theta, r) points at once; rows are theta, columns are r
ur     = YP @ Qlr
utheta = cot*((YP*(-(llpol+1))) @ Slr) + csc*((YP1*clm[idP+1:plx+1:2]) @ Slr) + csc*((1j*m*YT) @ Tlr)
uphi   = cot*((YT*(lltor+1))    @ Tlr) + csc*((YT1*(-clm[idT+1:tlx+1:2])) @ Tlr) + csc*((1j*m*YP) @ Slr)

S = np.sin(theta)[:,None]*r[None,:]
Z = np.cos(theta)[:,None]*r[None,:]


fig=plt.figure(figsize=(14,7))
# ------------------------------------------------------------------- ur
ax1=fig.add_subplot(131)
ax1.set_title(titlelabels[0],size=20)
#ax1.text(0.1,0,titlelabels[0],size=20)
if sys.argv[7] == 'raw':
    im1=ax1.contourf( S, Z, np.real(ur), 70, cmap=cmap)
elif sys.argv[7] == 'abs':
    im1=ax1.contourf( S, Z, np.absolute(ur), 70, cmap=cmap)
#for c in im1.collections:
#              c.set_edgecolor('face')
im1.set_edgecolor('face')
ax1.set_aspect('equal')
plt.colorbar(im1,aspect=70)

# --------------------------------------------------------------- utheta
ax2=fig.add_subplot(132)
ax2.set_title(titlelabels[1],size=20)
#ax2.text(0.1,0,titlelabels[1],size=20)
if sys.argv[7] == 'raw':
    im2=ax2.contourf( S, Z, np.real(utheta), 70, cmap=cmap)
elif sys.argv[7] == 'abs':
    im2=ax2.contourf( S, Z, np.absolute(utheta), 70, cmap=cmap)
#for c in im2.collections:
#              c.set_edgecolor('face')
im2.set_edgecolor('face')
ax2.set_aspect('equal')
plt.colorbar(im2,aspect=70)

# ----------------------------------------------------------------- uphi
ax3=fig.add_subplot(133)
ax3.set_title(titlelabels[2],size=20)
#ax3.text(0.1,0,titlelabels[2],size=20)
if sys.argv[7] == 'raw':
    im3=ax3.contourf( S, Z, np.real(uphi), 70, cmap=cmap)
elif sys.argv[7] == 'abs':
    im3=ax3.contourf( S, Z, np.absolute(uphi), 70, cmap=cmap)
#for c in im3.collections:
#              c.set_edgecolor('face')
im3.set_edgecolor('face')
ax3.set_aspect('equal')
plt.colorbar(im3,aspect=70)

# ----------------------------------------------------------------------
plt.tight_layout()
plt.show()
