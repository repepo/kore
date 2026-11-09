import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as ch

sys.path.insert(1,'bin/')

import utils as ut
from parameters import par

'''
Script to load a given solution in ipython
Use as:
%run -i tools/load_solution.py nsol field

nsol   : solution number
field  : 'u' or 'b' for flow or magnetic field
'''

solnum = int(sys.argv[1])
field  = sys.argv[2]

m           = par.m
lmax        = par.lmax
ricb        = par.ricb
rcmb        = ut.rcmb
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
    _sol = ut.load_field('flow', solnum)   # NEW (2026-11-08): flow.npy or the text files
    a, b = _sol.real, _sol.imag
    vsymm = par.symm
elif field == 'b':
    _sol = ut.load_field('magnetic', solnum)   # NEW (2026-11-08): magnetic.npy or the text files
    a, b = _sol.real, _sol.imag
    if ut.cic:
        _sol = ut.load_field('magnetic_ic', solnum)   # NEW (2026-11-08): magnetic_ic.npy or the text files
        a_ic, b_ic = _sol.real, _sol.imag
    vsymm = par.symm * ut.symmB0 
       
# Rearrange and separate poloidal and toroidal parts
Plj0 = a[:n] + 1j*b[:n]         #  N elements on each l block
Tlj0 = a[n:n+n] + 1j*b[n:n+n]   #  N elements on each l block

lm1  = lmax-m+1    
Plj0  = np.reshape(Plj0,(int(lm1/2),ut.N1))
Tlj0  = np.reshape(Tlj0,(int(lm1/2),ut.N1))

Plj = np.zeros((int(lm1/2),par.N),dtype=complex)
Tlj = np.zeros((int(lm1/2),par.N),dtype=complex)

# FIX (2026-11-08): the parity of the Chebyshev coefficients in a full sphere and in the inner core follows the symmetry of
# the field loaded (symm*symmB0 for 'b'), as in utils4pp.expand_sol; ut.s is the flow's
s = int(vsymm*0.5+0.5) # s=0 if antisymm, s=1 if symm

if ricb == 0 :
    iP = (m + 1 - s)%2
    iT = (m + s)%2
    for k in np.arange(int(lm1/2)) :
        Plj[k,iP::2] = Plj0[k,:]
        Tlj[k,iT::2] = Tlj0[k,:]
else :
    Plj = Plj0
    Tlj = Tlj0

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



if field == 'b' and ut.cic:  # loads the solution in the inner core.

    nic = ut.nic
    nr_ic = par.N_cic

    # set the radial grid
    i_ic = np.arange(0,nr_ic)
    x_ic = np.cos( (i_ic+0.5)*np.pi/nr_ic )
    x0_ic = x_ic[x_ic>0]
    r_ic = ricb*x0_ic 

    # matrix with Chebyshev polynomials at every x point for all degrees:
    chx_ic = ch.chebvander(x0_ic,par.N_cic-1)  # this matrix has nr_ic/2 rows and N_cic-1 cols
    # Rearrange and separate poloidal and toroidal parts
    offset = 0
    Plj0_ic = a_ic[offset    :offset+  nic] + 1j*b_ic[offset    :offset+  nic]   # N elements on each l block
    Tlj0_ic = a_ic[offset+nic:offset+2*nic] + 1j*b_ic[offset+nic:offset+2*nic]   # N elements on each l block

    lm1_ic  = lmax-m+1   # FIX (2026-11-08): the inner core has the degrees of the fluid (par.lmax_cic is conductive_IC's)
    Plj0_ic  = np.reshape(Plj0_ic,(int(lm1_ic/2),ut.Nic))
    Tlj0_ic  = np.reshape(Tlj0_ic,(int(lm1_ic/2),ut.Nic))

    Plj_ic = np.zeros((int(lm1_ic/2),par.N_cic),dtype=complex)
    Tlj_ic = np.zeros((int(lm1_ic/2),par.N_cic),dtype=complex)


    iP_ic = (m + 1 - s)%2
    iT_ic = (m + s)%2
    for k in np.arange(int(lm1_ic/2)) :
        Plj_ic[k,iP_ic::2] = Plj0_ic[k,:]
        Tlj_ic[k,iT_ic::2] = Tlj0_ic[k,:]
 

    if m>0:
        idp_ic = np.arange( 1-s, lm1_ic, 2)
        idt_ic = np.arange( s  , lm1_ic, 2)
        ll_ic  = np.arange( m, lmax+1 )
    elif m==0:
        idp_ic = np.arange( s  , lm1_ic, 2)
        idt_ic = np.arange( 1-s, lm1_ic, 2)
        ll_ic  = np.arange( m+1, lmax+2 )
        
    # init arrays
    Plr_ic  = np.zeros( (lm1_ic, int(nr_ic/2)), dtype=complex )
    Tlr_ic  = np.zeros( (lm1_ic, int(nr_ic/2)), dtype=complex )

    # populate Plr and Tlr
    Plr_ic[idp_ic,:] = np.matmul( Plj_ic, chx_ic.T)
    Tlr_ic[idt_ic,:] = np.matmul( Tlj_ic, chx_ic.T)
