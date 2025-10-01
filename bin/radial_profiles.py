import numpy as np
import utils as ut
import parameters as par
#import pygyre as gy
#import scipy.interpolate as si
#import mesa_reader as mr



def density(r, rpower):  # rᵃ ρ

    out = np.zeros_like(r)
    out = (1-r**2)**par.density_beta

    return (r**rpower)*out
    
    
    
def viscosity(r, rpower):  # rᵃ ν

    out = np.ones_like(r)

    return (r**rpower)*out



def dynamic_viscosity(r, rpower):  # μ = ρ ν

    out = np.zeros_like(r)
    out = density(r,rpower) * viscosity(r,0)

    return out



def densityX(r, Dorder):  # derivatives of ρ(r)
    
    tol = 1e-12
    out = ut.fundit( density, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out



def rhoX(r, *args): # rᵃ ρᵇ

    (rpower, rhopower) = args
    out = np.zeros_like(r)
    out = density(r, 0)**rhopower

    return (r**rpower)*out



def lhoX(r, lhoorder):  # ρⁿ (ln ρ)⁽ⁿ⁾
    '''
    Returns the lhoorder derivative of (ln ρ)
    multiplied by the lhoorder power of ρ
    in order to cancel any ρ in the denominator.
    '''

    out = np.zeros_like(r)
    dd = densityX(r, lhoorder)
    d0 = dd[:,0]

    if lhoorder == 1:
        d1 = dd[:,1]
        out = d1
    elif lhoorder == 2:
        d1 = dd[:,1]; d2 = dd[:,2]
        out = d0 * d2 - d1**2
    elif lhoorder == 3:
        d1 = dd[:,1]; d2 = dd[:,2]; d3 = dd[:,3]
        out = 2*d1**3 -3*d0*d1*d2 +(d0**2)*d3
    elif lhoorder == 4:
        d1 = dd[:,1]; d2 = dd[:,2]; d3 = dd[:,3]; d4 = dd[:,4]
        out = -6*d1**4 +12*d0*(d1**2)*d2 -4*(d0**2)*d1*d3 - 3*(d0**2)*(d2**2) +(d0**3)*d4

    return out



def rhoXlhoX(r, *args):  # ρᵃ (ln ρ)⁽ᵇ⁾

    out = np.zeros_like(r)
    (rhopower, lhoorder) = args
    delta = rhopower - lhoorder
    if delta == 0:
        out = lhoX(r,lhoorder)
    elif delta>0:
        out = rhoX(r, 0, delta) * lhoX(r, lhoorder)

    return out



def muX(r, Dorder):

    tol = 1e-12
    out = ut.fundit( dynamic_viscosity, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

    return out



def burrito(r, *args):

    #print('burrito args=',args)
    (secx, rpower, rhopower, muorder, lhoorder, dx) = args

    if (lhoorder == None) and (muorder == None):
        out = rhoX(r, rpower, rhopower)
    
    elif (lhoorder == None) and (muorder >= 0):
        out = rhoX(r, rpower, rhopower) * muX(r, muorder)

    elif (lhoorder in [1,2,3,4]) and (muorder == None):
        out = (r**rpower) * rhoXlhoX(r, rhopower, lhoorder)

    elif (lhoorder in [1,2,3,4]) and (muorder >=0):
        out = (r**rpower) * muX(r, muorder) * rhoXlhoX(r, rhopower, lhoorder)

    return out