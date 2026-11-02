# Variables involved with the boundary conditions

import numpy as np
import utils as ut
from parameters import par



def Tk(x, N, lamb_max, R1, R2) :
    '''
    Chebyshev polynomial from order 0 to N (as rows)
    and its derivatives ** with respect to r **, up to lamb_max (as columns),
    evaluated at x=-1 (r=R1) or x=1 (r=R2).
    '''
    
    if R1 == 0 :
        ric = -R2
    else :
        ric = R1
    
    out = np.zeros((N+1,lamb_max+1))
    
    for k in range(0,N+1):
        
        out[k,0] = x**k
        
        tmp = 1.
        for i in range(0, lamb_max):
            tmp = tmp * ( k**2 - i**2 )/( 2*i + 1 )
            out[k,i+1] = x**(k+i+1) * tmp * (2/(R2 - ric))**(i+1)
        
    return out
    

# to use in the b.c. and the torque calculation
Ta     = Tk(-1, par.N-1, 4, par.ricb, ut.rcmb)
Tb     = Tk( 1, par.N-1, 5, par.ricb, ut.rcmb)

if ut.cic:
    Tb_cic = Tk( 1, par.N_cic-1, 2, 0, par.ricb)

    if ut.symmB0 == -1:  # antisymmetric B0
        ixf = ( par.m + ut.s )%2
        ixg = ( par.m + 1 - ut.s )%2
    elif ut.symmB0 == 1:  # symmetric B0
        ixf = ( par.m + 1 - ut.s )%2
        ixg = ( par.m + ut.s )%2
    Tbfic = Tb_cic[ixf::2,:]
    Tbgic = Tb_cic[ixg::2,:]

if ut.resolved_mantle:  # NEW (2026-10-30): the conducting mantle layer, at the CMB and at r_surface
    Tm_cmb  = Tk(-1, par.N_mantle-1, 1, ut.rcmb, par.r_surface)
    Tm_surf = Tk( 1, par.N_mantle-1, 1, ut.rcmb, par.r_surface)



# Poloidals, Toroidals and derivatives at r = ricb

P0_icb = Ta[:,0]
P1_icb = Ta[:,1]
P2_icb = Ta[:,2]

T0_icb = Ta[:,0]
T1_icb = Ta[:,1]



# Poloidals, Toroidals and derivatives at r = rcmb

P0_cmb = Tb[:,0]
P1_cmb = Tb[:,1]
P2_cmb = Tb[:,2]

T0_cmb = Tb[:,0]
T1_cmb = Tb[:,1]
