# Variables involved with the boundary conditions

import numpy as np
import utils as ut
from parameters import par
import radial_profiles as rap


def Tk(x, N, lamb_max) :
    '''
    Chebyshev polynomial from order 0 to N (as rows)
    and its derivatives ** with respect to r **, up to lamb_max (as columns),
    evaluated at x=-1 (r=ricb) or x=1 (r=rcmb).
    '''
    
    if par.ricb == 0 :
        ric = -ut.rcmb
    else :
        ric = par.ricb
    
    out = np.zeros((N+1,lamb_max+1))
    
    for k in range(0,N+1):
        
        out[k,0] = x**k
        
        tmp = 1.
        for i in range(0, lamb_max):
            tmp = tmp * ( k**2 - i**2 )/( 2*i + 1 )
            out[k,i+1] = x**(k+i+1) * tmp * (2/(ut.rcmb - ric))**(i+1)
        
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
