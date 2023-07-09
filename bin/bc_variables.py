# Variables involved with the boundary conditions

import numpy as np
import utils as ut
from parameters import par



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
    
    
def Tcenter(N) :
    '''
    Chebyshev polynomial evaluated at x=0, from order 0 to N
    Useful to set boundary condition at the center if no inner core present
    i.e. when the Chebyshev domain is [-1,1]
    '''
    
    out = np.zeros((N+1,1))
    for k in range (0,N+1):
        out[k] = np.cos( k * np.pi/2 )
    out[abs(out)<0.1] = 0
    
    return out
        
        



# to use in the b.c. and the torque calculation
Ta = Tk(-1, par.N-1, 4)
Tb = Tk( 1, par.N-1, 5)
Tc = Tcenter(par.N-1)

'''
# For the ellipsoidal case 
if (par.bco == 0 or par.bco == 1):
    mua1 = 1. # 
    mua2 = 1. # will use spherical boundaries but will compute torques
    mua3 = 1. # as in an spheroid if alpha is different from zero
elif bco == 2:
    mua1 = 1.
    mua2 = 0.
    mua3 = 0.
elif bco == 3:
    mua1 = 1.
    mua2 = 1.
    mua3 = 0.
elif bco == 4:
    mua1 = 1.
    mua2 = 1.
    mua3 = 1.

a1 = mua1*ut.alpha
a2 = mua2*ut.alpha**2
a3 = mua3*ut.alpha**3

akj = np.zeros((4,4))   
# terms (r-1)^k / k! for the Taylor expansion, this one sets the long semiaxis a = 1
akj[0,:] = [ 1.                            , 0.                          , 0.                         , 0.            ]
akj[1,:] = [ -a1/3. - a2/5. - (13*a3)/105. , (-2*a1)/3. - a2/7. + a3/21. , (12*a2)/35. + (96*a3)/385. , (-40*a3)/231. ]
akj[2,:] = [ a2/10. + (3*a3)/35.           , (2*a2)/7. + a3/7.           , (4*a2)/35. - (48*a3)/385.  , (-8*a3)/77.   ]
akj[3,:] = [ -a3/42.                       , (-5*a3)/63.                 , (-4*a3)/77.                , (-8*a3)/693.  ] 

'''



# Poloidals, Toroidals and derivatives at r = ricb   

P0_icb = Ta[:,0]
P1_icb = Ta[:,1]
P2_icb = Ta[:,2]
P3_icb = Ta[:,3]
P4_icb = Ta[:,4]
 
T0_icb = Ta[:,0]
T1_icb = Ta[:,1]
T2_icb = Ta[:,2]
T3_icb = Ta[:,3]

# 0 component and its derivatives, it only has poloidals:
u0_icb = np.zeros((par.N, 4))
u0_icb[:,0] =  P0_icb 
u0_icb[:,1] =  P1_icb - P0_icb 
u0_icb[:,2] =  P2_icb - 2*P1_icb + 2*P0_icb 
u0_icb[:,3] =  P3_icb - 3*P2_icb + 6*P1_icb - 6*P0_icb

# +-1 component and derivatives. Poloidal part:
u1P_icb = np.zeros((par.N, 4))
u1P_icb[:,0] =  P1_icb + P0_icb 
u1P_icb[:,1] =  P2_icb + P1_icb - P0_icb 
u1P_icb[:,2] =  P3_icb + P2_icb - 2*P1_icb + 2*P0_icb 
u1P_icb[:,3] =  P4_icb + P3_icb - 3*P2_icb + 6*P1_icb - 6*P0_icb

# +-1 component and derivatives. Toroidal part:
u1T_icb = np.zeros((par.N, 4))
u1T_icb[:,0] =  T0_icb
u1T_icb[:,1] =  T1_icb 
u1T_icb[:,2] =  T2_icb 
u1T_icb[:,3] =  T3_icb



# Poloidals, Toroidals and derivatives at r = rcmb  

P0_cmb = Tb[:,0]
P1_cmb = Tb[:,1]
P2_cmb = Tb[:,2]
P3_cmb = Tb[:,3]
P4_cmb = Tb[:,4]
 
T0_cmb = Tb[:,0]
T1_cmb = Tb[:,1]
T2_cmb = Tb[:,2]
T3_cmb = Tb[:,3]

# 0 component and its derivatives, it only has poloidals:
u0_cmb = np.zeros((par.N, 4))
u0_cmb[:,0] =  P0_cmb 
u0_cmb[:,1] =  P1_cmb - P0_cmb 
u0_cmb[:,2] =  P2_cmb - 2*P1_cmb + 2*P0_cmb 
u0_cmb[:,3] =  P3_cmb - 3*P2_cmb + 6*P1_cmb - 6*P0_cmb

# +-1 component and derivatives. Poloidal part:
u1P_cmb = np.zeros((par.N, 4))
u1P_cmb[:,0] =  P1_cmb + P0_cmb 
u1P_cmb[:,1] =  P2_cmb + P1_cmb - P0_cmb 
u1P_cmb[:,2] =  P3_cmb + P2_cmb - 2*P1_cmb + 2*P0_cmb 
u1P_cmb[:,3] =  P4_cmb + P3_cmb - 3*P2_cmb + 6*P1_cmb - 6*P0_cmb

# +-1 component and derivatives. Toroidal part:
u1T_cmb = np.zeros((par.N, 4))
u1T_cmb[:,0] =  T0_cmb
u1T_cmb[:,1] =  T1_cmb 
u1T_cmb[:,2] =  T2_cmb 
u1T_cmb[:,3] =  T3_cmb




