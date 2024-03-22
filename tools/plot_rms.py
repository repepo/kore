import numpy as np
import scipy.sparse as ss
import sys
import matplotlib.pyplot as plt
import matplotlib
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
Script to plot the root-mean-square profile of the radial velocity or radial magnetic field.  
Use as:

python3 path/to/plot_rms.py solnum r0 r1 field

nsol   : solution number
r0     : starting radius
r1     : final radius
field  : whether to plot flow velocity or magnetic field ('u' or 'b')

Requires the PYTHONPATH to include the ../bin folder:

export PYTHONPATH=$PYTHONPATH:/path/to/bin
'''

# --- INITIALIZATION ---
# load input arguments
solnum = int(sys.argv[1])
r0 = float(sys.argv[2])
r1 = float(sys.argv[3])
field = sys.argv[4]

# load parameter data from solve.py generated files
p = np.loadtxt('params.dat')
if np.ndim(p) > 1:
    p = np.loadtxt('params.dat')[solnum, :]

m = int(p[5])
symm = int(p[6])

ricb = p[7]
rcmb = 1

lmax = int(p[47])
N = int(p[46])
n0 = int(N * (lmax - m + 1) / 2)

nr = N - 1

# set up the evenly spaced radial grid
r = np.linspace(ricb, rcmb, nr)

if ricb == 0:
    r = r[1:]
    nr = nr - 1
r_inv = 1/r
x = xcheb(r, ricb, rcmb)

# matrix with Chebyshev polynomials at every x point for all degrees:
chx = ch.chebvander(x, N - 1)  # this matrix has nr rows and N-1 cols

# read field from disk
if field == 'u':
    # read field data from disk
    a0 = np.loadtxt('real_flow.field', usecols=solnum)
    b0 = np.loadtxt('imag_flow.field', usecols=solnum)
    vsymm = symm
elif field == 'b':
    # read field data from disk
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

# initialize indices
ll0 = ell(m, lmax, vsymm)
llpol = ll0[0]
lltor = ll0[1]
ll = ll0[2]

# --- COMPUTATION ---
# expand solution in case ricb == 0
aib = expand_sol(a0 + 1j * b0, vsymm, ricb, m, lmax, N)
a = np.real(aib)
b = np.imag(aib)

# rearrange and separate poloidal and toroidal parts
Plj0 = a[:n0] + 1j * b[:n0]  # N elements on each l block
lm1 = lmax - m + 1

Plj = np.reshape(Plj0, (int(lm1 / 2), N))

# initialize arrays
P0 = np.zeros((int((lmax - m + 1) / 2), nr), dtype=complex)
Q0 = np.zeros((int((lmax - m + 1) / 2), nr), dtype=complex)

# populate matrices
np.matmul(Plj, chx.T, P0)

# compute multiplications
rI = ss.diags(r ** -1, 0)
lI = ss.diags(llpol * (llpol + 1), 0)

# compute radial, consoidal and toroidal scalar feilds
Q0 = lI * P0 * rI

# initialize solution array:
rms = np.zeros((int((lmax - m + 1) / 2), nr))

# compute poloidal components
for k, l in enumerate(llpol):
    L = l * (l + 1)

    q0 = Q0[k, :]

    f0 = 1 / (2 * l + 1)
    f1 = np.absolute(q0) ** 2

    rms[k, :] = r_inv * np.sqrt(f0*f1)

# --- VISUALIZATION ---
# set-up figure
fig = plt.figure(figsize=(6, 6))

# plot energy
ax1 = fig.add_subplot(111)

ax1.plot(r, sum(rms, 0))

ax1.set_yscale('log')
ax1.set_xlabel(r'$r$', size=12)
ax1.set_ylabel(r'${}_\mathrm{}$'.format(field, '{r.m.s.}'), size=12)

# add custom plotting arguments


# show/save figure
plt.tight_layout()
plt.show()
# plt.savefig('{}rms.png'.format(field))

