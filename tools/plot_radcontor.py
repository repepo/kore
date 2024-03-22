import numpy as np
import scipy.sparse as ss
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy.polynomial.chebyshev as ch

from utils import ell
from utils import Dcheb
from utils4pp import xcheb
from utils4fig import expand_sol

# colorblind safe
plt.style.use('tableau-colorblind10')

# latex fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

'''
Script to plot the radial profile of the radial, consoidal and toroidal kinetic energy 
and their spectral energy at a specified radius
Use as:

python3 path/to/plot_radcontor.py solnum r0 r1 field radius

nsol   : solution number
r0     : starting radius
r1     : final radius
field  : whether to plot flow velocity or magnetic field ('u' or 'b')
radius : radial location at which the spectral energy is computed

Requires the PYTHONPATH to include the ../bin folder:

export PYTHONPATH=$PYTHONPATH:/path/to/bin
'''

# --- INITIALIZATION ---
# load input arguments
solnum = int(sys.argv[1])
r0 = float(sys.argv[2])
r1 = float(sys.argv[3])
field = sys.argv[4]
radius = float(sys.argv[5])

# load parameter data from solve.py generated files
p = np.loadtxt('params.dat')
multi = False
if np.ndim(p) > 1:
    multi = True
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
r_sqr = r**2
x = xcheb(r,ricb,rcmb)

# matrix with Chebyshev polynomials at every x point for all degrees:
chx = ch.chebvander(x, N-1) # this matrix has nr rows and N-1 cols

# read field from disk
if field == 'u':
    # read field data from disk
    a0 = np.loadtxt('real_flow.field', usecols=solnum)
    b0 = np.loadtxt('imag_flow.field', usecols=solnum)
    vsymm = symm

    if multi:
        f = np.loadtxt('flow.dat')[solnum, :]
    else:
        f = np.loadtxt('flow.dat')
    total = f[0]

    titlelabel = 'total kinetic energy'

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

    if multi:
        f = np.loadtxt('magnetic.dat')[solnum, :]
    else:
        f = np.loadtxt('magnetic.dat')
    total = f[0]

    titlelabel = 'total magnetic energy'

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

d1Plj = np.zeros(np.shape(Plj),dtype=complex)
d2Plj = np.zeros(np.shape(Plj),dtype=complex)
d3Plj = np.zeros(np.shape(Plj),dtype=complex)

d1Tlj = np.zeros(np.shape(Tlj),dtype=complex)
d2Tlj = np.zeros(np.shape(Tlj),dtype=complex)

# initialize arrays
P0 = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
P1 = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
P2 = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
P3 = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)

T0 = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
T1 = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
T2 = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)

Q0 = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
S0 = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)

# populate matrices
np.matmul( Plj, chx.T, P0 )
np.matmul( Tlj, chx.T, T0 )

# compute derivative Plj
for k in range(np.size(llpol)):
    d1Plj[k, :] = Dcheb(Plj[k, :], ricb, rcmb)
    d2Plj[k, :] = Dcheb(d1Plj[k, :], ricb, rcmb)
    d3Plj[k, :] = Dcheb(d2Plj[k, :], ricb, rcmb)
np.matmul(d1Plj, chx.T, P1)
np.matmul(d2Plj, chx.T, P2)
np.matmul(d3Plj, chx.T, P3)

# compute derivatives Tlj
for k in range(np.size(lltor)):
    d1Tlj[k,:] = Dcheb(   Tlj[k,:], ricb, rcmb)
    d2Tlj[k,:] = Dcheb( d1Tlj[k,:], ricb, rcmb)
np.matmul(d1Tlj, chx.T, T1)
np.matmul(d2Tlj, chx.T, T2)

# compute multiplications
rI = ss.diags(r**-1,0)
lI = ss.diags(llpol*(llpol+1),0)
lI_inv = ss.diags(1/(llpol*(llpol+1)),0)

# compute radial, consoidal and toroidal scalar feilds
Q0 = lI * P0 * rI
Q1 = ( lI * P1 -   Q0 ) * rI
Q2 = ( lI * P2 - 2*Q1 ) * rI

S0 = P1 + P0 * rI
S1 = P2 + lI_inv * Q1
S2 = P3 + lI_inv * Q2

# initialize solution arrays:
pol = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
rad = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
con = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
tor = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)

# compute poloidal components
for k,l in enumerate(llpol):
    L = l*(l+1)
    
    q0 = Q0[k,:]
    q1 = Q1[k,:]
    q2 = Q2[k,:] 
    
    s0 = S0[k,:]
    s1 = S1[k,:]
    s2 = S2[k,:]
    
    f0 = 4*np.pi/(2*l+1)    

    f1 = r_sqr*np.absolute( q0 )**2
    f2 = r_sqr*L*np.absolute( s0 )**2
    pol[k,:] = f0*(f1+f2)
    rad[k,:] = f0*(f1)
    con[k,:] = f0*(f2)
    
# compute toroidal components
for k,l in enumerate(lltor):
    
    L = l*(l+1)
    
    t0 = T0[k,:]
    t1 = T1[k,:]
    t2 = T2[k,:]
    
    f0 = 4*np.pi/(2*l+1)

    f1 = (r_sqr)*L*np.absolute(t0)**2
    tor[k,:] = f0*f1

tot_r = np.real(sum(pol+tor,0))
rad_r = np.real(sum(rad,0))
con_r = np.real(sum(con,0))
tor_r = np.real(sum(tor,0))

# --- VISUALIZATION ---
# set-up figure
fig = plt.figure(figsize=(4, 6))

# plot energy
ax1 = fig.add_subplot(211)

ax1.plot(r, tot_r/total,'--',lw=1,color='k', label=titlelabel, zorder=1)
ax1.plot(r, rad_r/total, label=r'radial', zorder=0)
ax1.plot(r, con_r/total, label=r'consoidal', zorder=0)
ax1.plot(r, tor_r/total, label=r'toroidal', zorder=0)

ax1.set_xlabel(r'$r$',size=12)
ax1.legend(fontsize=10)

# plot spectral energy
ax2 = fig.add_subplot(212)

k = np.argmin(np.abs(r-radius))
ax2.plot(llpol,np.abs(rad[:,k])/tot_r[k], '.-',lw=0.3,label=r'radial')
ax2.plot(llpol,np.abs(con[:,k])/tot_r[k], '.-',lw=0.3,label=r'consoidal')
ax2.plot(lltor,np.abs(tor[:,k])/tot_r[k], '.-',lw=0.3,label=r'toroidal')

ax2.set_yscale('log')
ax2.set_xlabel(r'Spherical harmonic degree $\ell$',size=12)
ax2.set_ylabel(r'Spectral energy at $r={}$'.format(radius),size=12)

# add custom plotting arguments

# show/save figure
plt.tight_layout()
plt.show()
#plt.savefig('radcontor.png'.format(ricb))

