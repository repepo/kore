import numpy as np
import numpy.polynomial.chebyshev as ch
import scipy.sparse as ss
import scipy.special as scsp
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cmasher as cmr
import matplotlib.colors as colors
from matplotlib.axis import Axis
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

from utils import Dcheb
from utils import Ylm_full
from utils import Ylm
from utils import ell
from utils4pp import xcheb
from utils4fig import expand_sol

# colorblind safe
plt.style.use('tableau-colorblind10')

# latex fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

'''
Script to plot meridional cuts of the flows energy density
Use as:
python3 path/to/plot_energy.py solnum theta0 theta1 field

solnum : solution number
theta0 : starting colatitude
theta1 : final colatitude
field  : whether to plot flow velocity or magnetic field ('u' or 'b')

Requires the PYTHONPATH to include the ../bin folder:

export PYTHONPATH=$PYTHONPATH:/path/to/bin
'''

# --- INITIALIZATION ---
# load input arguments
solnum = int(sys.argv[1])
theta0 = float(sys.argv[2])
theta1 = float(sys.argv[3])
field  = sys.argv[4]

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
phi = 0.

# matrix with Chebyshev polynomials at every x point for all degrees:
chx = ch.chebvander(x, N-1) # this matrix has nr rows and N-1 cols

# set up evenly spaced latitudinal grid
theta = np.linspace(theta0*np.pi/180, theta1*np.pi/180, ntta+2)
theta = theta[1:-1]

if field == 'u':
    # read field from disk
    a0 = np.loadtxt("real_flow.field", usecols=solnum)
    b0 = np.loadtxt("imag_flow.field", usecols=solnum)
    vsymm = symm
    titlelabel = r'Kinetic energy'
    cmap = 'magma'
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
    titlelabel = r'Magnetic energy'
    cmap = 'cividis'

# initialize indices
ll0   = ell(m, lmax, vsymm)
llpol = ll0[0]
lltor = ll0[1]
ll    = ll0[2]

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

ur2 = np.zeros((nr) * ntta)
ut2 = np.zeros((nr) * ntta)
up2 = np.zeros((nr) * ntta)

# initialize spherical harmonics coefficients.
clm = np.zeros((lm1+1,1))
for i,l in enumerate(ll):
	clm[i] = np.sqrt((l-m)*(l+m))

# start index for l. Do not confuse with indices for the Cheb expansion!
sy = int( vsymm*0.5 + 0.5 ) # sy=0 if antisymm, sy=1 if symm
idP = (np.sign(m)+sy  )%2
idT = (np.sign(m)+sy+1)%2
plx = idP+lmax-m+1
tlx = idT+lmax-m+1

# compute ur, utheta, uphi squared
k = 0
for kt in range(ntta):
    ylm = np.r_[Ylm_full(lmax, m, theta[kt], phi),0]
    for kr in range(0, nr):
        s[k] = r[kr] * np.sin(theta[kt])
        z[k] = r[kr] * np.cos(theta[kt])

        ur2[k] = np.abs(np.dot(Qlr[:, kr], ylm[idP:plx:2])) ** 2

        tmp1 = np.dot(-(llpol + 1) * Slr[:, kr] / np.tan(theta[kt]), ylm[idP:plx:2])
        tmp2 = np.dot(clm[idP + 1:plx + 1:2, 0] * Slr[:, kr] / np.sin(theta[kt]), ylm[idP + 1:plx + 1:2])
        tmp3 = np.dot(1j * m * Tlr[:, kr] / np.sin(theta[kt]), ylm[idT:tlx:2])
        ut2[k] = np.abs(tmp1 + tmp2 + tmp3) ** 2

        tmp1 = np.dot(             (lltor+1) * Tlr[:,kr]/np.tan(theta[kt]), ylm[idT:tlx:2]     )
        tmp2 = np.dot( -clm[idT+1:tlx+1:2,0] * Tlr[:,kr]/np.sin(theta[kt]), ylm[idT+1:tlx+1:2] )
        tmp3 = np.dot(                  1j*m * Slr[:,kr]/np.sin(theta[kt]), ylm[idP:plx:2]     )
        up2[k] = np.abs(tmp1 + tmp2 + tmp3) ** 2

        k = k + 1

# only compute indices inside the core mantle boundary (|r| < rcmb)
id_in = np.where(s**2 + z**2 < rcmb)
S = s[id_in]
Z = z[id_in]

triang = tri.Triangulation(S, Z)

# mask off unwanted triangles inside the inner core (|r| < ricb)
xmid = S[triang.triangles].mean(axis=1)
ymid = Z[triang.triangles].mean(axis=1)
mask = np.where((xmid**2 + ymid**2 <= ricb ** 2), 1, 0)
triang.set_mask(mask)

# --- VISUALIZATION ---
# set-up figure
fig = plt.figure(figsize=(6, 6))

# plot kinetic energy density
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.axis('off')

# set zero-values to very small
res = ur2[id_in]+ut2[id_in]+up2[id_in]
res[res == 0] = 1e-17
im = ax.tricontourf(triang, np.log10(res), 70, cmap=cmap)

ax.plot(r[0]*np.sin(theta), r[0]*np.cos(theta),'k',lw=0.4)
ax.plot(r[-1]*np.sin(theta), r[-1]*np.cos(theta),'k',lw=0.4)

div = make_axes_locatable(ax)
cax = div.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax)
cbar.set_ticks(cbar.get_ticks()[1:-1])
cbar.set_ticklabels('{:1.2e}'.format(-np.exp(t)) for t in cbar.get_ticks())
cbar.set_label(titlelabel, fontsize=14)

# add custom plotting arguments


# show/save figure
plt.tight_layout()
plt.show()
#plt.savefig('energy.png'.format(ricb), dpi=300)