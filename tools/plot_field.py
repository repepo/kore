import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import sys
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.tri as tri
import numpy.polynomial.chebyshev as ch
import multiprocessing as mp

sys.path.insert(1,'bin/')

import utils as ut
from parameters import par
import utils4pp as upp
import radial_profiles as rap

'''
Script to plot meridional cuts of a solution field
Use as:

python3 plot_field.py nR ntheta theta0 theta1 field opt nsol

nR     : number of points in radius
ntheta : number of points in the theta direction
theta0 : starting colatitude
theta1 : final colatitude
field  : whether to plot flow velocity or magnetic field ('u' or 'b')
opt    : 'raw' for real part (phase and phi dependent!), or 'abs' for the magnitude
nsol   : solution index
'''

plt.rc('text', usetex=True)

solnum = int(sys.argv[7])

if sys.argv[5] in ['u', 'vel', 'mf', 'vis', 'vif', 'cor', 'curl_u', 'curl_vel', 'curl_cor', 'curl_vis', 'curl_vif']:

    ru = np.loadtxt('real_flow.field',usecols=solnum)
    iu = np.loadtxt('imag_flow.field',usecols=solnum)
    vsymm = par.symm
    usol = upp.expand_reshape_sol( ru + 1j*iu, vsymm)
    tsol = None
    cmap = 'rainbow'

elif sys.argv[5] in ['buo', 'curlbuo']:

    rt = np.loadtxt('real_thermal.field',usecols=solnum)
    it = np.loadtxt('imag_thermal.field',usecols=solnum)
    vsymm = par.symm
    tsol = upp.expand_reshape_sol( rt + 1j*it, vsymm)
    usol = None
    cmap = 'rainbow'

elif sys.argv[5] in ['test1', 'test2']:

    ru = np.loadtxt('real_flow.field',usecols=solnum)
    iu = np.loadtxt('imag_flow.field',usecols=solnum)
    vsymm = par.symm
    usol = upp.expand_reshape_sol( ru + 1j*iu, vsymm)
    rt = np.loadtxt('real_thermal.field',usecols=solnum)
    it = np.loadtxt('imag_thermal.field',usecols=solnum)
    tsol = upp.expand_reshape_sol( rt + 1j*it, vsymm)
    cmap = 'rainbow'

elif sys.argv[5] == 'b':
    a0 = np.loadtxt('real_magnetic.field',usecols=solnum)
    b0 = np.loadtxt('imag_magnetic.field',usecols=solnum)
    vsymm = ut.bsymm
    cmap = 'plasma'
    tl0 = r'\mathbf{b}'

eigval = np.loadtxt('eigenvalues0.dat').reshape((-1,2))
sigma = eigval[solnum,0]

if sys.argv[5] in ['u', 'vel']:
    tl0 = r'\mathbf{u}'
elif sys.argv[5] == 'mf':
    tl0 = r'\rho\,\mathbf{u}'
elif sys.argv[5] in ['vis','vif']:
    tl0 = r'\mathbf{F_\nu/\rho}'
elif sys.argv[5] == 'cor':
    tl0 = r'2\mathbf{\hat z}\times \mathbf{u}'
elif sys.argv[5] == 'buo':
    tl0 = r'\mathbf{F_g}'
elif sys.argv[5] in ['curl_u', 'curl_vel']:
    tl0 = r'\nabla\times\mathbf{u}'
elif sys.argv[5] in ['curl_vis','curl_vif']:
    tl0 = r'\nabla\times(\mathbf{F_\nu/\rho})'
elif sys.argv[5] == 'curl_cor':
    tl0 = r'\nabla\times(2\mathbf{\hat z}\times \mathbf{u})'
elif sys.argv[5] == 'curl_buo':
    tl0 = r'\nabla\times(\mathbf{F_g})'
elif sys.argv[5] == 'test1':
    tl0 = r'\nabla\times (\sigma\,\mathbf{u})'
elif sys.argv[5] == 'test2':
    tl0 = r'\nabla\times (\mathbf{F_g}+\mathbf{F_\nu}-\mathbf{F_\Omega})'

if sys.argv[6] == 'raw':
    bra1 = r'$\mathbf{\hat r}\cdot\mathrm{Re}\left('
    bra2 = r'$\mathbf{\hat \theta}\cdot\mathrm{Re}\left('
    bra3 = r'$\mathbf{\hat \phi}\cdot\mathrm{Re}\left('
    ket = r'\right)$'

elif sys.argv[6] == 'abs':
    bra1 = r'$\mathbf{\hat r}\cdot\left|'
    bra2 = r'$\mathbf{\hat \theta}\cdot\left|'
    bra3 = r'$\mathbf{\hat \phi}\cdot\left|'
    ket = r'\right|$'
                    
titlelabels = [bra1 + tl0 + ket, bra2 + tl0 + ket, bra3 + tl0 + ket]

lmax = par.lmax
m    = par.m
symm = par.symm
N    = par.N
Ek   = par.Ek
ricb = par.ricb
rcmb = 1
n    = ut.n
n0   = ut.n0

nR = int(sys.argv[1]) # number of radial points
Ntheta = int(sys.argv[2]) # number of points in the theta direction

ncpus   = mp.cpu_count()
ntht_pp = int(np.round(Ntheta/ncpus))  # number of theta points per cpu
totheta = ntht_pp * ncpus     # the actual number of total theta points  

# setup radial grid
gap = rcmb-ricb
r = np.linspace(ricb,rcmb,nR)
if ricb == 0:
	r = r[1:]
	nR = nR - 1

phi = 0. # select meridional cut

ll0 = ut.ell(m,lmax,vsymm)
llpol = ll0[0]
lltor = ll0[1]
ll    = ll0[2]


# --------------------------------------------------------------
if sys.argv[5] == 'test2':
    out10 = upp.diagnose_4plot(ncpus, usol, tsol, r, 'curl_buo')
    out11 = upp.diagnose_4plot(ncpus, usol, tsol, r, 'curl_vis')
    out12 = upp.diagnose_4plot(ncpus, usol, tsol, r, 'curl_cor')
    #out13 = upp.diagnose_4plot(ncpus, usol, tsol, r, 'curlvel')
    out1 = out10 + out11 - out12 # - sigma*out13
elif sys.argv[5] == 'test1':
    out1 = sigma * upp.diagnose_4plot(ncpus, usol, tsol, r, 'curl_vel')
else:
    out1 = upp.diagnose_4plot(ncpus, usol, tsol, r, sys.argv[5])
# --------------------------------------------------------------




if sys.argv[5][:4] in ['curl', 'test']:
    vsymm = -1*vsymm  # change the symmetry if we are plotting the curl of something

# start index for l. Do not confuse with indices for the Cheb expansion!
sy = int( vsymm*0.5 + 0.5 ) # sy=0 if antisymm, sy=1 if symm
idP = (np.sign(m)+sy  )%2
idT = (np.sign(m)+sy+1)%2
plx = idP+lmax-m+1
tlx = idT+lmax-m+1

Qlr = out1[idP:plx:2,0,:]
Slr = out1[idP:plx:2,1,:]
Tlr = out1[idT:tlx:2,2,:]

ll0 = ut.ell(m,lmax,vsymm)
llpol = ll0[0]
lltor = ll0[1]
ll    = ll0[2]

# setup the latitudinal grid
theta = np.linspace(float(sys.argv[3])*np.pi/180, float(sys.argv[4])*np.pi/180, totheta+2)
theta = theta[1:-1]
theta2 = np.reshape(theta,(-1,ncpus),copy=True)

clm = np.zeros((lmax-m+2,1))
for i,l in enumerate(ll):
	clm[i] = np.sqrt((l-m)*(l+m))

# start index for l. Do not confuse with indices for the Cheb expansion!
sy = int( vsymm*0.5 + 0.5 ) # sy=0 if antisymm, sy=1 if symm
idP = (np.sign(m)+sy  )%2
idT = (np.sign(m)+sy+1)%2
plx = idP+lmax-m+1
tlx = idT+lmax-m+1


def pieceofcake( theta_pp ):

    ntht_pp = len(theta_pp)
    out = np.zeros( (nR*ntht_pp,5), dtype=complex ) 

    k=0
    for kt in range(ntht_pp):

        tht = theta_pp[kt]
        ylm = np.r_[ut.Ylm_full(lmax, m, tht, phi),0]	
        for kr,rr in enumerate(r):
            
            out[k,0]   = rr*np.sin(tht)  #s
            out[k,1]   = rr*np.cos(tht)  #z
            
            out[k,2] = np.dot( Qlr[:,kr], ylm[idP:plx:2] )  #ur	

            tmp1 = np.dot(           -(llpol+1) * Slr[:,kr]/np.tan(tht), ylm[idP:plx:2]     )
            tmp2 = np.dot( clm[idP+1:plx+1:2,0] * Slr[:,kr]/np.sin(tht), ylm[idP+1:plx+1:2] )
            tmp3 = np.dot(                 1j*m * Tlr[:,kr]/np.sin(tht), ylm[idT:tlx:2]     )
            out[k,3] = tmp1+tmp2+tmp3  #utheta
            
            tmp1 = np.dot(             (lltor+1) * Tlr[:,kr]/np.tan(tht), ylm[idT:tlx:2]     )
            tmp2 = np.dot( -clm[idT+1:tlx+1:2,0] * Tlr[:,kr]/np.sin(tht), ylm[idT+1:tlx+1:2] )
            tmp3 = np.dot(                  1j*m * Slr[:,kr]/np.sin(tht), ylm[idP:plx:2]     )
            out[k,4] = tmp1+tmp2+tmp3  #uphi
            	
            k=k+1

    return out


pool  = mp.Pool(processes=ncpus)
popov = [ pool.apply_async( pieceofcake, [theta2[:,i]]) for i in range(ncpus) ]
out0 = [ pp.get() for pp in popov ]
pool.close()
pool.join()

out1 = np.vstack(out0)
(s0,z0,ur,utheta,uphi) = np.unstack(out1, axis=1)
s = np.real(s0)
z = np.real(z0)

# Mask the inner core
a = 1.
c = 1.
id_in = np.where((s**2/(a**2)) + (z**2/(c**2)) < 1.)
s1 = s[id_in]
z1 = z[id_in]
triang = tri.Triangulation(s1, z1)
xmid = s1[triang.triangles].mean(axis=1)
x2 = xmid*xmid
ymid = z1[triang.triangles].mean(axis=1)
y2 = ymid*ymid
mask = np.where( (x2 + y2 <= ricb**2), 1, 0)
triang.set_mask(mask)


fig=plt.figure(figsize=(14,7))
# ------------------------------------------------------------------- ur
ax1=fig.add_subplot(131)
ax1.set_title(titlelabels[0],size=20)
#ax1.text(0.1,0,titlelabels[0],size=20)
if sys.argv[6] == 'raw':
    im1=ax1.tricontourf( triang, np.real(ur[id_in]), 70, cmap=cmap)
elif sys.argv[6] == 'abs':
    im1=ax1.tricontourf( triang, np.absolute(ur[id_in]), 70, cmap=cmap)
#for c in im1.collections:
#              c.set_edgecolor('face')
im1.set_edgecolor('face')
ax1.set_aspect('equal')
plt.colorbar(im1,aspect=70)

# --------------------------------------------------------------- utheta
ax2=fig.add_subplot(132)
ax2.set_title(titlelabels[1],size=20)
#ax2.text(0.1,0,titlelabels[1],size=20)
if sys.argv[6] == 'raw':
    im2=ax2.tricontourf( triang, np.real(utheta[id_in]), 70, cmap=cmap)
elif sys.argv[6] == 'abs':
    im2=ax2.tricontourf( triang, np.absolute(utheta[id_in]), 70, cmap=cmap)
#for c in im2.collections:
#              c.set_edgecolor('face')
im2.set_edgecolor('face')
ax2.set_aspect('equal')
plt.colorbar(im2,aspect=70)

# ----------------------------------------------------------------- uphi
ax3=fig.add_subplot(133)
ax3.set_title(titlelabels[2],size=20)
#ax3.text(0.1,0,titlelabels[2],size=20)
if sys.argv[6] == 'raw':
    im3=ax3.tricontourf( triang, np.real(uphi[id_in]), 70, cmap=cmap)
elif sys.argv[6] == 'abs':
    im3=ax3.tricontourf( triang, np.absolute(uphi[id_in]), 70, cmap=cmap)
#for c in im3.collections:
#              c.set_edgecolor('face')
im2.set_edgecolor('face')
ax3.set_aspect('equal')
plt.colorbar(im3,aspect=70)

# ----------------------------------------------------------------------
plt.tight_layout()
plt.show()
