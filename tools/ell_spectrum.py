import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as ch

sys.path.insert(1,'bin/')

import utils as ut
import parameters as par

'''
Script to compute the ell-spectrum at a given radius
Use as:
python3 tools/ell_spectrum.py nsol radius field

nsol   : solution number
radius : radial location
field  : 'u' or 'b' for flow or magnetic field
'''

solnum = int(sys.argv[1])
radius = float(sys.argv[2])
field  = sys.argv[3]

m           = par.m
lmax        = par.lmax
ricb        = par.ricb
rcmb        = 1
gap         = rcmb - ricb
n           = ut.n
nr          = par.N


# set the radial grid
i = np.arange(0,nr)
x = np.cos( (i+0.5)*np.pi/nr )
r = 0.5*gap*(x+1) + ricb;

if ricb == 0 :
    x0 = 0.5 + x/2
else :
    x0 = x

# matrix with Chebyshev polynomials at every x point for all degrees:
chx = ch.chebvander(x0,par.N-1) # this matrix has nr rows and N-1 cols

# read fields from disk
if field == 'u':
    a = np.loadtxt('real_flow.field',usecols=solnum)
    b = np.loadtxt('imag_flow.field',usecols=solnum)
    vsymm = par.symm
elif field == 'b':
    a = np.loadtxt('real_magnetic.field',usecols=solnum)
    b = np.loadtxt('imag_magnetic.field',usecols=solnum)
    vsymm = par.symm * ut.symmB0 
       
# Rearrange and separate poloidal and toroidal parts
Plj0 = a[:n] + 1j*b[:n]         #  N elements on each l block
Tlj0 = a[n:n+n] + 1j*b[n:n+n]   #  N elements on each l block

lm1  = lmax-m+1    
Plj0  = np.reshape(Plj0,(int(lm1/2),ut.N1))
Tlj0  = np.reshape(Tlj0,(int(lm1/2),ut.N1))

Plj = np.zeros((int(lm1/2),par.N),dtype=complex)
Tlj = np.zeros((int(lm1/2),par.N),dtype=complex)

if ricb == 0 :
    iP = (m + 1 - ut.s)%2
    iT = (m + ut.s)%2
    for k in np.arange(int(lm1/2)) :
        Plj[k,iP::2] = Plj0[k,:]
        Tlj[k,iT::2] = Tlj0[k,:]
else :
    Plj = Plj0
    Tlj = Tlj0

s = int(vsymm*0.5+0.5) # s=0 if antisymm, s=1 if symm
if m>0:
    idp = np.arange( 1-s, lm1, 2)
    idt = np.arange( s  , lm1, 2)
    ll  = np.arange( m, lmax+1 )
elif m==0:
    idp = np.arange( s  , lm1, 2)
    idt = np.arange( 1-s, lm1, 2)
    ll  = np.arange( m+1, lmax+2 )
    
# init arrays
Plr  = np.zeros( (lm1, nr), dtype=complex )
Tlr  = np.zeros( (lm1, nr), dtype=complex )

# populate Plr and Tlr
Plr[idp,:] = np.matmul( Plj, chx.T)
Tlr[idt,:] = np.matmul( Tlj, chx.T)

ir = np.argmin(abs(r-radius))

# plot only the coeffs that are not zero
kp = abs(Plr[:,ir]) > 0
kt = abs(Tlr[:,ir]) > 0 

plt.figure()
plt.yscale('log')
plt.xlim(0,50)
plt.plot(ll[kp],abs(Plr[kp,ir]),'o-',ms=3,lw=1,label=r'Poloidal')
plt.plot(ll[kt],abs(Tlr[kt,ir]),'o-',ms=3,lw=1,label=r'Toroidal')
plt.xlabel(r'Angular degree $\ell$',size=14)
plt.ylabel(f'Spectral amplitude at $r={radius}$',size=14)
plt.legend()

plt.tight_layout()
plt.show()
