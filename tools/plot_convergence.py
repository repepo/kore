import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as ch

from utils import ell
from utils4pp import xcheb
from utils4fig import expand_sol

# colorblind safe
plt.style.use('tableau-colorblind10')

# latex fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


'''
Script to compute the ell-spectrum at a given radius
Use as:
python3 /path/to/plot_convergence.py nsol field radius

nsol   : solution number
field  : 'u', 'b' for flow or magnetic field
radius : radial location at which the spectral energy is computed

Requires the PYTHONPATH to include the ../bin folder:

export PYTHONPATH=$PYTHONPATH:/path/to/bin
'''

# --- INITIALIZATION ---
# load input arguments
solnum = int(sys.argv[1])
field  = sys.argv[2]
radius = None
if len(sys.argv) == 4:
    radius = float(sys.argv[3])

# load parameter values from params.dat file
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

# set up the evenly spaced radial grid
r = np.linspace(ricb,rcmb,nr)

if ricb == 0:
    r = r[1:]
    nr = nr - 1
x = xcheb(r,ricb,rcmb)

# matrix with Chebyshev polynomials at every x point for all degrees:
chx = ch.chebvander(x, N-1)  # this matrix has nr rows and N-1 cols

# read fields from disk
if field == 'u':
    a0 = np.loadtxt('real_flow.field',usecols=solnum)
    b0 = np.loadtxt('imag_flow.field',usecols=solnum)
    vsymm = symm
elif field == 'b':
    a0 = np.loadtxt('real_magnetic.field',usecols=solnum)
    b0 = np.loadtxt('imag_magnetic.field',usecols=solnum)
    vsymm = -symm  # because external mag field is antisymmetric wrt the equator (if dipole or axial)
       
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

# rearrange and separate poloidal and toroidal parts
Plj0 = a[:n0] + 1j*b[:n0] 		        #  N elements on each l block
Tlj0 = a[n0:n0+n0] + 1j*b[n0:n0+n0] 	#  N elements on each l block
lm1 = lmax-m+1

Plj = np.reshape(Plj0, (int(lm1/2), N))
Tlj = np.reshape(Tlj0, (int(lm1/2), N))

# initialize arrays
Plr = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
Tlr = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)

# populate Plr and Tlr
np.matmul(Plj, chx.T, Plr)
np.matmul(Tlj, chx.T, Tlr)

if radius:
    ir = np.argmin(abs(r-radius))
    Pld = Plr[:, ir]
    Tld = Tlr[:, ir]

    txt = f'Spectral amplitude at $r={radius}$'
else:
    Pld = np.sum(Plr, axis=1)
    Tld = np.sum(Tlr, axis=1)

    txt = f'Spectral amplitude in full domain'

# plot only the coefficients that are nonzero
kp = abs(Pld) > 0
kt = abs(Tld) > 0

# --- VISUALIZATION ---
# set-up figure
plt.figure(figsize=(5, 3.5))
plt.yscale('log')
plt.xlim(0, lm1)

# plot ell spectrum
plt.plot(llpol[kp], abs(Pld[kp]), 'o-', ms=5, lw=1, label=r'Poloidal')
plt.plot(lltor[kt], abs(Tld[kt]), 'o-', ms=5, lw=1, label=r'Toroidal')
plt.xlabel(r'Angular degree $\ell$', size=12)
plt.ylabel(txt, size=12)

plt.legend(fontsize=12)

# add custom plotting arguments

# show/save figure
plt.savefig('convergence.png')

plt.tight_layout()
plt.show()

