import numpy as np
import numpy.polynomial.chebyshev as ch
import scipy.sparse as ss
import scipy.special as scsp
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as colors
import cmasher as cmr
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import Dcheb
from utils import Ylm_full
from utils import ell
from utils4pp import xcheb
from utils4fig import expand_sol


# colorblind safe
plt.style.use('tableau-colorblind10')

# latex fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


'''
Script to plot meridional cuts of a solution field
Use as:

python3 path/to/plot_field.py nsol theta0 theta1 coord field opt

nsol   : solution number
theta0 : starting colatitude
theta1 : final colatitude
coord  : 'cyl', 'sph' for cylindrical or spherical coordinates
field  : whether to plot flow velocity or magnetic field ('u' or 'b')
opt    : 'raw' for real part (phase and phi dependent!), or 'abs' for the magnitude

Requires the PYTHONPATH to include the ../bin folder:

export PYTHONPATH=$PYTHONPATH:/path/to/bin
'''

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


# --- INITIALIZATION ---
# load input arguments
solnum = int(sys.argv[1])
theta0 = float(sys.argv[2])
theta1 = float(sys.argv[3])
coord  = sys.argv[4]
field  = sys.argv[5]
opt    = sys.argv[6]

# load parameter data from solve.py generated files
p = np.loadtxt('params.dat')
if np.ndim(p) > 1:
    p = np.loadtxt('params.dat')[solnum, :]

m     = int(p[5])
symm  = int(p[6])

ricb  = p[7]
rcmb  = 1

lmax  = int(p[47])
N     = int(p[46])
N1    = int(N/2) * int(1 + np.sign(ricb)) + int((N%2)*np.sign(ricb))
n0    = int(N*(lmax-m+1)/2)

nr    = N-1
ntta  = lmax-1

# set up the evenly spaced radial grid
r = np.linspace(ricb, rcmb, nr)

if ricb == 0:
    r = r[1:]
    nr = nr - 1
x = xcheb(r,ricb,rcmb)

# select meridional cut
phi = 0

# matrix with Chebyshev polynomials at every x0 point for all degrees:
chx = ch.chebvander(x, N-1) # this matrix has nr rows and N-1 cols

# set up evenly spaced latitudinal grid
theta = np.linspace(theta0*np.pi/180, theta1*np.pi/180, ntta+2)
theta = theta[1:-1]

if field == 'u':
    # read field from disk
    a0 = np.loadtxt("real_flow.field", usecols=solnum)
    b0 = np.loadtxt("imag_flow.field", usecols=solnum)
    vsymm = symm

    # define colormaps and axes titles
    if opt == "raw":
        cmap = 'cmr.prinsenvlag'
        if coord == "sph":
            titlelabels = [r'$\mathbf{\hat r}\cdot\mathrm{Re}\left(\mathbf{u_0}\right)$',
                           r'$\mathbf{\hat \theta}\cdot\mathrm{Re}\left(\mathbf{u_0}\right)$',
                           r'$\mathbf{\hat \phi}\cdot\mathrm{Re}\left(\mathbf{u_0}\right)$']
        elif coord == "cyl":
            titlelabels = [r'$\mathbf{\hat s}\cdot\mathrm{Re}\left(\mathbf{u_0}\right)$',
                           r'$\mathbf{\hat z}\cdot\mathrm{Re}\left(\mathbf{u_0}\right)$',
                           r'$\mathbf{\hat \phi}\cdot\mathrm{Re}\left(\mathbf{u_0}\right)$']
    elif opt == "abs":
        cmap = 'plasma'
        if coord == "sph":
            titlelabels = [r'$\mathbf{\hat r}\cdot\left|\mathbf{u_0}\right|$',
                           r'$\mathbf{\hat \theta}\cdot\left|\mathbf{u_0}\right|$',
                           r'$\mathbf{\hat \phi}\cdot\left|\mathbf{u_0}\right|$']
        elif coord == "cyl":
            titlelabels = [r'$\mathbf{\hat s}\cdot\left|\mathbf{u_0}\right|$',
                           r'$\mathbf{\hat z}\cdot\left|\mathbf{u_0}\right|$',
                           r'$\mathbf{\hat \phi}\cdot\left|\mathbf{u_0}\right|$']
elif field == 'b':
    # read field from disk
    a0 = np.loadtxt("real_magnetic.field",usecols=solnum)
    b0 = np.loadtxt("imag_magnetic.field",usecols=solnum)
    B0_type = int(p[15])
    if B0_type in np.arange(4):
        B0_symm = -1
    elif B0_type == 4:
        B0_symm = 1
    elif B0_type == 5:
        B0_l = p[17]
        B0_symm = int((-1)**(B0_l))
    vsymm = symm * B0_symm

    # define colormaps and axes titles
    if opt == "raw":
        cmap = 'cmr.waterlily'
        if coord == "sph":
            titlelabels = [r'$\mathbf{\hat r}\cdot\mathrm{Re}\left(\mathbf{b_0}\right)$', \
                           r'$\mathbf{\hat \theta}\cdot\mathrm{Re}\left(\mathbf{b_0}\right)$', \
                           r'$\mathbf{\hat \phi}\cdot\mathrm{Re}\left(\mathbf{b_0}\right)$']
        elif coord == "cyl":
            titlelabels = [r'$\mathbf{\hat s}\cdot\mathrm{Re}\left(\mathbf{b_0}\right)$', \
                           r'$\mathbf{\hat z}\cdot\mathrm{Re}\left(\mathbf{b_0}\right)$', \
                           r'$\mathbf{\hat \phi}\cdot\mathrm{Re}\left(\mathbf{b_0}\right)$']
    elif opt == "abs":
        if coord == "sph":
            titlelabels = [r'$\mathbf{\hat r}\cdot\left|\mathbf{b_0}\right|$', \
                           r'$\mathbf{\hat \theta}\cdot\left|\mathbf{b_0}\right|$', \
                           r'$\mathbf{\hat \phi}\cdot\left|\mathbf{b_0}\right|$']
        elif coord == "cyl":
            titlelabels = [r'$\mathbf{\hat s}\cdot\left|\mathbf{b_0}\right|$', \
                           r'$\mathbf{\hat z}\cdot\left|\mathbf{b_0}\right|$', \
                           r'$\mathbf{\hat \phi}\cdot\left|\mathbf{b_0}\right|$']
        cmap = "viridis"

# initialize indices
ll0 = ell(m, lmax, vsymm)
llpol = ll0[0]
lltor = ll0[1]
ll = ll0[2]

# --- COMPUTATION ---
# expand solution in case ricb == 0
aib = expand_sol(a0+1j*b0,vsymm, ricb, m, lmax, N)
a = np.real(aib)
b = np.imag(aib)

#rearrange and separate poloidal and toroidal parts
Plj0 = a[:n0] + 1j*b[:n0] 		        #  N elements on each l block
Tlj0 = a[n0:n0+n0] + 1j*b[n0:n0+n0] 	#  N elements on each l block
lm1 = lmax-m+1

Plj = np.reshape(Plj0, (int(lm1/2), N))
Tlj = np.reshape(Tlj0, (int(lm1/2), N))
dPlj = np.zeros(np.shape(Plj),dtype=complex)

# initialize arrays
Plr = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
dP  = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
rP  = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
Qlr = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
Slr = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
Tlr = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)

# populate Plr and Tlr
np.matmul(Plj, chx.T, Plr)
np.matmul(Tlj, chx.T, Tlr)

# compute derivative Plj
for k in range(np.size(llpol)):
	dPlj[k,:] = Dcheb(Plj[k,:], ricb, rcmb)
np.matmul(dPlj, chx.T, dP)

# compute multiplications
rI = ss.diags(r**-1,0)
lI = ss.diags(llpol*(llpol+1),0)

rP  = Plr * rI
Qlr = lI * rP
Slr = rP + dP

# initialize solution arrays
s = np.zeros(nr * ntta)
z = np.zeros(nr * ntta)

ur = np.zeros((nr) * ntta,dtype=complex)
utheta = np.zeros((nr) * ntta,dtype=complex)
uphi = np.zeros((nr) * ntta,dtype=complex)

if coord == 'cyl':
    uz = np.zeros((nr) * ntta,dtype=complex)
    us = np.zeros((nr) * ntta,dtype=complex)

# initialize spherical harmonics coefficients.
clm = np.zeros((lmax-m+2,1))
for i,l in enumerate(ll):
    clm[i] = np.sqrt((l-m)*(l+m))

# start index for l. Do not confuse with indices for the Cheb expansion!
sy = int( vsymm*0.5 + 0.5 ) # sy=0 if antisymm, sy=1 if symm
idP = (np.sign(m)+sy  )%2
idT = (np.sign(m)+sy+1)%2
plx = idP+lmax-m+1
tlx = idT+lmax-m+1

# compute ur, utheta, uphi
k = 0
for kt in range(ntta):
    ylm = np.r_[Ylm_full(lmax, m, theta[kt], phi),0]
    for kr in range(0,nr):
        s[k]   = r[kr]*np.sin(theta[kt])
        z[k]   = r[kr]*np.cos(theta[kt])

        ur[k] = np.dot( Qlr[:,kr], ylm[idP:plx:2] )

        tmp1 = np.dot(           -(llpol+1) * Slr[:,kr]/np.tan(theta[kt]), ylm[idP:plx:2]     )
        tmp2 = np.dot( clm[idP+1:plx+1:2,0] * Slr[:,kr]/np.sin(theta[kt]), ylm[idP+1:plx+1:2] )
        tmp3 = np.dot(                 1j*m * Tlr[:,kr]/np.sin(theta[kt]), ylm[idT:tlx:2]     )
        utheta[k] = tmp1+tmp2+tmp3

        tmp1 = np.dot(             (lltor+1) * Tlr[:,kr]/np.tan(theta[kt]), ylm[idT:tlx:2]     )
        tmp2 = np.dot( -clm[idT+1:tlx+1:2,0] * Tlr[:,kr]/np.sin(theta[kt]), ylm[idT+1:tlx+1:2] )
        tmp3 = np.dot(                  1j*m * Slr[:,kr]/np.sin(theta[kt]), ylm[idP:plx:2]     )
        uphi[k] = tmp1+tmp2+tmp3

        if coord == 'cyl':
            uz[k] = ur[k] * np.cos(theta[kt]) - utheta[k] * np.sin(theta[kt])
            us[k] = ur[k] * np.sin(theta[kt]) + utheta[k] * np.cos(theta[kt])

        k = k + 1

# only compute indices inside the core mantle boundary (|r| < rcmb)
id_in = np.where(s**2 + z**2 < rcmb)
S = s[id_in]
Z = z[id_in]

triang = tri.Triangulation(S, Z)

# mask off unwanted triangles inside the inner core (|r| < ricb)
xmid = S[triang.triangles].mean(axis=1)
ymid = Z[triang.triangles].mean(axis=1)
#ricb = 0.95
mask = np.where((xmid**2 + ymid**2 <= ricb ** 2), 1, 0)
triang.set_mask(mask)

# --- VISUALIZATION ---
# set-up figure
fig = plt.figure(figsize=(12,4))

# plot r/s
ax1 = fig.add_subplot(131)
ax1.set_aspect('equal')
ax1.get_xaxis().set_visible(True)
ax1.get_yaxis().set_visible(True)
ax1.set_title(titlelabels[0],size=14)

if opt == "raw":
    norm1 = MidpointNormalize(midpoint=0)
    if coord == 'cyl':
        im1 = ax1.tricontourf(triang, np.real(us[id_in]), 70, norm=norm1, cmap=cmap)
    if coord == 'sph':
        im1 = ax1.tricontourf(triang, np.real(ur[id_in]), 70, norm=norm1, cmap=cmap)
if opt == "abs":
    if coord == 'cyl':
        im1 = ax1.tricontourf(triang, np.absolute(us[id_in]), 70, cmap=cmap)
    if coord == 'sph':
        im1 = ax1.tricontourf(triang, np.absolute(ur[id_in]), 70, cmap=cmap)
ax1.plot(r[0]*np.sin(theta), r[0]*np.cos(theta),'k',lw=0.4)
ax1.plot(r[-1]*np.sin(theta), r[-1]*np.cos(theta),'k',lw=0.4)

div1 = make_axes_locatable(ax1)
cax1 = div1.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax1)

# plot theta/z
ax2 = fig.add_subplot(132)
ax2.set_aspect('equal')
ax2.get_xaxis().set_visible(True)
ax2.get_yaxis().set_visible(True)
ax2.set_title(titlelabels[1],size=14)

if opt == "raw":
    norm2 = MidpointNormalize(midpoint=0)
    if coord == 'cyl':
        im2 = ax2.tricontourf(triang, np.real(uz[id_in]), 70, norm=norm2, cmap=cmap)
    if coord == 'sph':
        im2 = ax2.tricontourf(triang, np.real(utheta[id_in]), 70, norm=norm2, cmap=cmap)
if opt == "abs":
    if coord == 'cyl':
        im2 = ax2.tricontourf(triang, np.absolute(uz[id_in]), 70, cmap=cmap)
    if coord == 'sph':
        im2 = ax2.tricontourf(triang, np.absolute(utheta[id_in]), 70, cmap=cmap)
ax2.plot(r[0]*np.sin(theta), r[0]*np.cos(theta),'k',lw=0.4)
ax2.plot(r[-1]*np.sin(theta), r[-1]*np.cos(theta),'k',lw=0.4)

div2 = make_axes_locatable(ax2)
cax2 = div2.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2, cax=cax2)

# plot phi
ax3 = fig.add_subplot(133)
ax3.set_aspect('equal')
ax3.get_xaxis().set_visible(True)
ax3.get_yaxis().set_visible(True)
ax3.set_title(titlelabels[2],size=14)

if opt == "raw":
    norm3 = MidpointNormalize(midpoint=0)
    im3 = ax3.tricontourf(triang, np.real(uphi[id_in]), 70, norm=norm3, cmap=cmap)
if opt == "abs":
    im3 = ax3.tricontourf(triang, np.absolute(uphi[id_in]), 70, cmap=cmap)
ax3.plot(r[0]*np.sin(theta), r[0]*np.cos(theta),'k',lw=0.4)
ax3.plot(r[-1]*np.sin(theta), r[-1]*np.cos(theta),'k',lw=0.4)

div3 = make_axes_locatable(ax3)
cax3 = div3.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im3, cax=cax3)

# add custom plotting arguments

# show/save figure
plt.tight_layout()
plt.show()
#plt.savefig('field.png'.format(ricb))