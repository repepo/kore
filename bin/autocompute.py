#!/usr/bin/env python3

import numpy as np
import scipy.sparse as ss
# from scipy.linalg import solve
from scipy.sparse.linalg import spsolve
import numpy.polynomial.chebyshev as ch
import bc_variables as bv
import parameters as par
import utils as ut
import radial_profiles as rap


def balance_heat_flux():

    bc_val_bot = par.bci_thermal_val
    bc_val_top = par.bco_thermal_val

    tol = 1e-9

    cd_eps = ut.chebco_f(rap.epsilon_h,par.N,par.ricb,ut.rcmb,tol)
    epsInt = ch.chebint(cd_eps)

    return eps0, bc_val_top, bc_val_bot

def get_equilibrium_entropy():
    '''
    Function to compute equilibrium entropy gradient profile by solving
    Div(rho T kappa grad S) = Q
    '''

    # Balance heat fluxes
    eps0, bc_val_top, bc_val_bot = balance_heat_flux()

    tol = 1e-9

    r0  = ut.chebco(0, par.N, tol, par.ricb, ut.rcmb)
    r1  = ut.chebco(1, par.N, tol, par.ricb, ut.rcmb)
    cd_eps = ut.chebco_f(rap.epsilon_h,par.N,par.ricb,ut.rcmb,tol,args=eps0)

    D1 = ut.Dlam(1, par.N)
    D2 = ut.Dlam(2, par.N)
    S0 = ut.Slam(0, par.N) # From the Chebyshev basis ( C^(0) basis ) to C^(1) basis
    S1 = ut.Slam(1, par.N) # From C^(1) basis to C^(2) basis

    S10 = S1*S0

    r0_D1      = ut.Mlam(S10*r0,2,0) * (S1*D1)
    r1_D2      = ut.Mlam(S10*r1,2,0) * D2
    eps_h      = S10*cd_eps

    if par.anelastic:
        cd_lnT = ut.chebify( rap.log_temperature, 1, tol)
        cd_lho = ut.chebify( rap.log_density, 1, tol)
        cd_lnk = ut.chebify( rap.log_thermal_diffusivity, 1, tol)

        r1_lnT1 = ut.cheb2Product( r1, cd_lnT[:,1], tol)
        r1_lho1 = ut.cheb2Product( r1, cd_lho[:,1], tol)
        r1_lnk1 = ut.cheb2Product( r1, cd_lnk[:,1], tol)

        r1_lnT1_D1 = ut.Mlam(S10*r1_lnT1,2,0) * (S1*D1)
        r1_lho1_D1 = ut.Mlam(S10*r1_lho1,2,0) * (S1*D1)
        r1_lnk1_D1 = ut.Mlam(S10*r1_lnk1,2,0) * (S1*D1)


    z2 = ss.csr_matrix((2,ut.N1))
    chop = 2

    r0_D1      = ss.vstack( [ z2, r0_D1[:-chop,:] ], format='csr' )
    r1_D2      = ss.vstack( [ z2, r1_D2[:-chop,:] ], format='csr' )

    if par.anelastic:
        r1_lnT1_D1 = ss.vstack( [ z2, r1_lnT1_D1[:-chop,:] ], format='csr' )
        r1_lho1_D1 = ss.vstack( [ z2, r1_lho1_D1[:-chop,:] ], format='csr' )
        r1_lnk1_D1 = ss.vstack( [ z2, r1_lnk1_D1[:-chop,:] ], format='csr' )

    if par.anelastic:
        Amat = 2*r0_D1 + r1_lnT1_D1 + r1_lho1_D1 + r1_lnk1_D1 + r1_D2
    else:
        Amat = 2*r0_D1 + r1_D2

    if par.bci_thermal == 0: #Constant entropy
        Amat[0,:] = bv.Ta[:,0]
    elif par.bci_thermal == 1: #Constant flux
        Amat[0,:] = bv.Ta[:,1]

    if par.bco_thermal == 0: #Constant entropy
        Amat[1,:] = bv.Tb[:,0]
    elif par.bco_thermal == 1: #Constant flux
        Amat[1,:] = bv.Tb[:,1]

    bci = par.bci_thermal_val
    bco = par.bco_thermal_val

    Bmat = np.r_[bci,bco,eps_h[:-chop]]

    # Amat = Amat.todense()

    Scheb = spsolve(Amat,Bmat)

    dScheb = ut.Dcheb(Scheb,par.ricb,ut.rcmb)

    return dScheb


def gravCoeff():
    '''
    Integrates density profile in Cheb space and gives Cheb
    coefficients of gravity profile, normalized to the value
    at the outer boundary. This works. We checked. Again.
    '''
    ck = ut.chebco_f(rap.density,par.N,par.ricb,ut.rcmb,par.tol_tc)

    # x0 = -(par.ricb + ut.rcmb)/(ut.rcmb - par.ricb)
    if par.ricb > 0:
        gk = (ut.rcmb - par.ricb)/2. * ch.chebint(ck,lbnd=-1)

        g_ref = ch.chebval(1,gk)
        out = gk/g_ref

        out[0] += par.g_icb # Value of g at ricb normalized by value at rcmb
    else:
        gk = ut.rcmb * ch.chebint(ck,lbnd=0) # g at origin goes to zero
        g_ref = ch.chebval(1,gk)
        out = gk/g_ref

    return out[:par.N]