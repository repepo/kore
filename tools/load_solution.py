import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as ch

sys.path.insert(1,'bin/')

import utils as ut
import parameters as par

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
    a = np.loadtxt('real_flow.field',usecols=solnum)
    b = np.loadtxt('imag_flow.field',usecols=solnum)
    vsymm = par.symm
elif field == 'b':
    a = np.loadtxt('real_magnetic.field',usecols=solnum)
    b = np.loadtxt('imag_magnetic.field',usecols=solnum)
    if ut.icflag:
        a_ic = np.loadtxt('real_magnetic_ic.field',usecols=solnum)
        b_ic = np.loadtxt('imag_magnetic_ic.field',usecols=solnum)
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



if field == 'b' and ut.icflag:  # loads the solution in the inner core. 

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

    lm1_ic  = par.lmax_cic-m+1    
    Plj0_ic  = np.reshape(Plj0_ic,(int(lm1_ic/2),ut.Nic))
    Tlj0_ic  = np.reshape(Tlj0_ic,(int(lm1_ic/2),ut.Nic))

    Plj_ic = np.zeros((int(lm1_ic/2),par.N_cic),dtype=complex)
    Tlj_ic = np.zeros((int(lm1_ic/2),par.N_cic),dtype=complex)


    iP_ic = (m + 1 - ut.s)%2
    iT_ic = (m + ut.s)%2
    for k in np.arange(int(lm1_ic/2)) :
        Plj_ic[k,iP_ic::2] = Plj0_ic[k,:]
        Tlj_ic[k,iT_ic::2] = Tlj0_ic[k,:]
 

    s = int(vsymm*0.5+0.5) # s=0 if antisymm, s=1 if symm
    if m>0:
        idp_ic = np.arange( 1-s, lm1_ic, 2)
        idt_ic = np.arange( s  , lm1_ic, 2)
        ll_ic  = np.arange( m, par.lmax_cic+1 )
    elif m==0:
        idp_ic = np.arange( s  , lm1_ic, 2)
        idt_ic = np.arange( 1-s, lm1_ic, 2)
        ll_ic  = np.arange( m+1, par.lmax_cic+2 )
        
    # init arrays
    Plr_ic  = np.zeros( (lm1_ic, int(nr_ic/2)), dtype=complex )
    Tlr_ic  = np.zeros( (lm1_ic, int(nr_ic/2)), dtype=complex )

    # populate Plr and Tlr
    Plr_ic[idp_ic,:] = np.matmul( Plj_ic, chx_ic.T)
    Tlr_ic[idt_ic,:] = np.matmul( Tlj_ic, chx_ic.T)