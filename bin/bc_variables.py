# Variables involved with the boundary conditions

import numpy as np
import utils as ut
from parameters import par
import radial_profiles as rap


def Tk(x, N, lamb_max, R1=None, R2=None) :
    '''
    Chebyshev polynomial from order 0 to N (as rows)
    and its derivatives ** with respect to r **, up to lamb_max (as columns),
    evaluated at x=-1 (r=ricb) or x=1 (r=rcmb).
    NEW (2026-10-28): on the domain [R1, R2], or [-R2, R2] if R1 = 0 (default: the fluid's, as before)
    '''
    if R1 is None:
        R1, R2 = par.ricb, ut.rcmb

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
    
    


R  = ut.rcmb
Ri = par.ricb       

# Log density and up to 2nd derivative at the surface
lhb0 = rap.logrhoX(R,0)
lhb1 = rap.logrhoX(R,1)
lhb2 = rap.logrhoX(R,2)

if par.ricb > 0:
    # Density and up to 2nd derivative at the ICB
    lha0 = rap.logrhoX(Ri,0)
    lha1 = rap.logrhoX(Ri,1)
    lha2 = rap.logrhoX(Ri,2)


# to use in the b.c. and the torque calculation
Ta = Tk(-1, par.N-1, 4)
Tb = Tk( 1, par.N-1, 5)


if ut.cic:  # NEW (2026-10-28): the inner-core field at r = ricb, its Chebyshev domain [-ricb, ricb], one parity per section
    Tb_cic = Tk( 1, par.N_cic-1, 2, 0, par.ricb)
    ixf = ( par.m + ut.s )%2 if ut.symmB0 == -1 else ( par.m + 1 - ut.s )%2   # parity of f (as the vector parity vF)
    ixg = 1 - ixf
    Tbfic = Tb_cic[ixf::2,:]
    Tbgic = Tb_cic[ixg::2,:]
