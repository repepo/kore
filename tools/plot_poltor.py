import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import sys
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.tri as tri
import numpy.polynomial.chebyshev as ch
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

Script to plot cuts of the poloidal and toroidal fields
Use as:

python3 path/to/plot_poltor.py nsol theta0 theta1 field

nsol   : solution number
angle0 : starting angle (theta or phi depending on the type of cut)
angle1 : final angle (theta or phi depending on the type of cut)
field  : whether to plot flow velocity or magnetic field ('u' or 'b')
cut    : whether to plot, 'merid', a meridonial cut (fixed phi) or, 'equat', an equatorial cut (theta = pi/2)

Requires the PYTHONPATH to include the ../bin folder:

export PYTHONPATH=$PYTHONPATH:/path/to/bin
'''

# --- INITIALIZATION ---
# load input arguments
solnum = int(sys.argv[1])
angle0 = float(sys.argv[2])
angle1 = float(sys.argv[3])
field  = sys.argv[4]
cut    = sys.argv[5]

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
n     = int(N1*(lmax-m+1)/2)
n0    = int(N*(lmax-m+1)/2)

nr    = N-1
ntta  = lmax-1

# set up the evenly spaced radial grid
r = np.linspace(ricb,rcmb,nr)

if ricb == 0:
	r = r[1:]
	nr = nr - 1
x = xcheb(r,ricb,rcmb)

# select meridional cut
phi = 0

# matrix with Chebyshev polynomials at every x0 point for all degrees:
chx = ch.chebvander(x, N-1) # this matrix has nr rows and N-1 cols

if cut == 'merid':
    # select meridional cut
    nphi = 1
    phi = np.array([0.])
    # set up evenly spaced latitudinal grid
    ntta = lmax - 1
    theta = np.linspace(angle0*np.pi/180, angle1*np.pi/180, ntta+2)
    theta = theta[1:-1]
if cut == 'equat':
    # select equator
    ntta = 1
    theta = np.array([np.pi/2])
    # set up evenly spaced azimuthal grid
    nphi = lmax - 1
    phi = np.linspace(angle0*np.pi/180, angle1*np.pi/180, nphi+2)
    phi = phi[1:-1]

if field == 'u':
	# read field from disk
	a0 = np.loadtxt("real_flow.field", usecols=solnum)
	b0 = np.loadtxt("imag_flow.field", usecols=solnum)
	vsymm = symm
	titlelabel = r'Velocity field $\mathbf{u_0}$'
elif field == 'b':
	# read field from disk
	a0 = np.loadtxt("real_magnetic.field", usecols=solnum)
	b0 = np.loadtxt("imag_magnetic.field", usecols=solnum)
	B0_type = int(p[15])
	if B0_type in np.arange(4):
		B0_symm = -1
	elif B0_type == 4:
		B0_symm = 1
	elif B0_type == 5:
		B0_l = p[17]
		B0_symm = int((-1) ** (B0_l))
	vsymm = symm * B0_symm
	titlelabel = r'Magnetic field $\mathbf{b_0}$'

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
x = np.zeros(nr * max(ntta,nphi))
y = np.zeros(nr * max(ntta,nphi))

pol = np.zeros((nr)*max(ntta,nphi), dtype=complex)
tor = np.zeros((nr)*max(ntta,nphi), dtype=complex)

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

# compute pol, tor
k = 0
for kp in range(nphi):
	for kt in range(ntta):
		ylm = np.r_[Ylm_full(lmax, m, theta[kt], phi[kp]),0]
		for kr in range(0,nr):
			if cut == 'merid':
				x[k] = r[kr] * np.sin(theta[kt])
				y[k] = r[kr] * np.cos(theta[kt])
			elif cut == 'equat':
				x[k] = r[kr] * np.sin(theta[kt]) * np.cos(phi[kp])
				y[k] = r[kr] * np.sin(theta[kt]) * np.sin(phi[kp])

			pol[k] = np.dot(Plr[:,kr], ylm[idP:plx:2])
			tor[k] = np.dot( Tlr[:,kr], ylm[idT:tlx:2] )

			k = k+1

# only compute indices inside the core mantle boundary (|r| < rcmb)
id_in = np.where(x**2 + y**2 < rcmb)
X = x[id_in]
Y = y[id_in]

triang = tri.Triangulation(X, Y)

# mask off unwanted triangles inside the inner core (|r| < ricb)
xmid = X[triang.triangles].mean(axis=1)
ymid = Y[triang.triangles].mean(axis=1)
mask = np.where((xmid**2 + ymid**2 <= ricb ** 2), 1, 0)
triang.set_mask(mask)

# --- VISUALIZATION ---
# set-up figure
fig = plt.figure(figsize=(8,4))
fig.suptitle(titlelabel, fontsize=16)

# plot pol
ax1 = fig.add_subplot(121)
ax1.set_aspect('equal')
ax1.get_xaxis().set_visible(True)
ax1.get_yaxis().set_visible(True)

ax1.set_title('$|\mathcal{P}|$', size=14)
im1 = ax1.tricontourf(triang, np.absolute(pol[id_in]), 70, cmap='cmr.gem')
ax1.plot(r[0]*np.sin(theta), r[0]*np.cos(theta),'k',lw=0.4)
ax1.plot(r[-1]*np.sin(theta), r[-1]*np.cos(theta),'k',lw=0.4)

div1 = make_axes_locatable(ax1)
cax1 = div1.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax1)

# plot tor
ax2 = fig.add_subplot(122)
ax2.set_aspect('equal')
ax2.get_xaxis().set_visible(True)
ax2.get_yaxis().set_visible(True)

ax2.set_title('$|\mathcal{T}|$', size=14)
im2 = ax2.tricontourf(triang, np.absolute(tor[id_in]), 70, cmap='cmr.pepper')
ax2.plot(r[0]*np.sin(theta), r[0]*np.cos(theta),'k',lw=0.4)
ax2.plot(r[-1]*np.sin(theta), r[-1]*np.cos(theta),'k',lw=0.4)

div2 = make_axes_locatable(ax2)
cax2 = div2.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2, cax=cax2)

# add custom plotting arguments

# show/save figure
plt.tight_layout()
plt.show()
#plt.savefig('poltor.png'.format(ricb))
