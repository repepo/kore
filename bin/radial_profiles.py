import numpy as np
import utils as ut
import parameters as par

# -----------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------- Structure profiles
# -----------------------------------------------------------------------------------------------------



def density(r, rpower):  # rᵃ ρ(r)

    out = np.zeros_like(r)

    if par.background == 0:

        out = ut.load_mesa(r,'density')

    elif par.background == 1:
        ### DEFINE BACKGROUND PROFILE OF DENSITY ###

        density_beta = 1
        out = (1 - r ** 2) ** density_beta

        ############################################
    elif par.background == 2:

        rad = np.loadtxt('radius.dat')
        out = ut.interp(r, rad, np.loadtxt('density.dat'))

    return (r**rpower)*out
    


def gravity(r, rpower):  # rᵃ g(r)

    out = np.zeros_like(r)

    if par.background == 0:

        out = ut.load_mesa(r,'gravity')

    elif par.background == 1:
        ### DEFINE BACKGROUND PROFILE OF GRAVITY ###

        out = np.ones_like(r)

        ############################################
    elif par.background == 2:

        rad = np.loadtxt('radius.dat')
        out = ut.interp(r, rad, np.loadtxt('gravity.dat'), even=False)

    return (r**rpower)*out



def bg_entropy_gradient(r, rpower):   # rᵃ dS/dr, Note that N²(r) = g(r) dS/dr in dimensionless units

    out = np.zeros_like(r)

    if par.background == 0:

        out = ut.load_mesa(r,'entropy_gradient')

    else:

        if par.def_entropy == 0:

            if not gravity(r, 0).any(0):

                BV2 = np.zeros_like(r)

                if par.background == 1:
                    ### DEFINE BACKGROUND PROFILE OF BRUNT-VÄISÄLÄ FREQUENCY ###

                    BV2 = np.ones_like(r)

                    ############################################################
                elif par.background == 2:

                    rad = np.loadtxt('radius.dat')
                    BV2 = ut.interp(r, rad, np.loadtxt('BV_frequency.dat'))

                out = BV2 / gravity(r, 0)

            else:
                print('Error! gravity profile cannot be zero anywhere')

        elif par.def_entropy == 1:

            if par.background == 1:
                ### DEFINE BACKGROUND GRADIENT OF ENTROPY ###

                out = np.ones_like(r)

                #############################################
            elif par.background == 2:

                rad = np.loadtxt('radius.dat')
                out = ut.interp(r, rad, np.loadtxt('entropy_gradient.dat'), even=False)

    return (r**rpower)*out



def bg_pressure(r, rpower):   # rᵃ p(r), Note that T(r) = p(r) / rho(r) in dimensionless units

    out = np.zeros_like(r)

    if par.background == 0:
        out = ut.load_mesa(r, 'pressure')

    else:

        if par.def_pressure == 0:

            temp = np.zeros_like(r)

            if par.background == 1:
                ### DEFINE BACKGROUND PROFILE OF TEMPERATURE ###

                temp = np.ones_like(r)

                ############################################################
            elif par.background == 2:

                rad = np.loadtxt('radius.dat')
                temp = ut.interp(r, rad, np.loadtxt('temperature.dat'))

            out = density(r, 0)*temp

        elif par.def_pressure == 1:

            if par.background == 1:
                ### DEFINE BACKGROUND GRADIENT OF PRESSURE ###

                out = np.ones_like(r)

                ##############################################
            elif par.background == 2:

                rad = np.loadtxt('radius.dat')
                out = ut.interp(r, rad, np.loadtxt('pressure.dat'))

    return (r**rpower)*out


def viscosity(r, rpower):  # rᵃ v(r)

    out = np.zeros_like(r)

    if par.background == 1 or (par.background == 0 and par.def_viscosity == 1):
        ### DEFINE BACKGROUND PROFILE OF KINEMATIC VISCOSITY ###

        out = np.ones_like(r)

        ########################################################
    elif par.background == 2 or (par.background == 0 and par.def_viscosity == 2):

        rad = np.loadtxt('radius.dat')
        out = ut.interp(r, rad, np.loadtxt('viscosity.dat'))

    return (r**rpower)*out



def thermal_diffusivity(r, rpower):   # rᵃ κ(r)

    out = np.zeros_like(r)

    if par.background == 1 or (par.background == 0 and par.def_thermal_diffusivity == 1):
        ### DEFINE BACKGROUND PROFILE OF THERMAL DIFFUSIVITY ###

        out = np.ones_like(r)

        ########################################################
    elif par.background == 2 or (par.background == 0 and par.def_thermal_diffusivity == 2):

        rad = np.loadtxt('radius.dat')
        out = ut.interp(r, rad, np.loadtxt('thermal_diffusivity.dat'))

    return (r**rpower)*out



# -----------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------ Derived profiles
# -----------------------------------------------------------------------------------------------------



def rho_T(r, rpower):   # ρT

    out = bg_pressure(r, 0)

    return (r**rpower)*out



def rho_T_dSdr(r, rpower):   # ρTdS/dr

    out = rho_T(r, 0) * bg_entropy_gradient(r, 0)

    return (r**rpower)*out



def kappa_rho_T(r, rpower):   # κρT

    out = thermal_diffusivity(r, 0) * rho_T(r, 0)

    return (r**rpower)*out



def dynamic_viscosity(r, rpower):  # μ = ρν

    out = np.zeros_like(r)
    out = density(r,rpower) * viscosity(r,0)

    return out



def densityX(r, Dorder):  # ρ⁽ⁿ⁾, radial derivatives of ρ(r)
    
    tol = 1e-12
    out = ut.fundit( density, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out



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



# -----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------- Profile functions for burrito
# -----------------------------------------------------------------------------------------------------



def rhoX(r, *args):   # rᵃ ρᵇ  powers of ρ

    (rpower, rhopower) = args
    out = np.zeros_like(r)
    out = density(r, 0)**rhopower

    return (r**rpower)*out



def rhoXlhoX(r, *args):   # ρᵃ (ln ρ)⁽ᵇ⁾  derivatives of ρ

    out = np.zeros_like(r)
    (rhopower, lhoorder) = args
    delta = rhopower - lhoorder
    if delta == 0:
        out = lhoX(r,lhoorder)
    elif delta>0:
        out = rhoX(r, 0, delta) * lhoX(r, lhoorder)

    return out



def muX(r, Dorder):   # Dynamic viscosity μ(r) = ρ(r)ν(r)

    tol = 1e-12
    out = ut.fundit( dynamic_viscosity, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

    return out



def krTX(r, Dorder):   # κ(r)ρ(r)T(r)

    tol = 1e-12
    out = ut.fundit( kappa_rho_T, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

    return out



def rTSX(r, Dorder):   # ρ(r)T(r)dS/dr

    tol = 1e-12
    out = ut.fundit( rho_T_dSdr, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

    return out



def graX(r, Dorder):   # g(r)

    tol = 1e-12
    out = ut.fundit( gravity, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

    return out



def roTX(r, Dorder):   # ρ(r)T(r)

    tol = 1e-12
    out = ut.fundit( rho_T, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

    return out


# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

proffdir = { 'lho':rhoXlhoX, 'moe':muX, 'rho':rhoX, 'gra':graX, 'roT':roTX  , 'rTS':rTSX  ,'krT':krTX }

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------



def burrito(r, *args):

    (secx, rpower, rhopower, func1, dorder1, func2, dorder2, dx) = args

    out0 = np.ones_like(r)
    out1 = np.ones_like(r)
    out2 = np.ones_like(r)

    if 'lho' not in (func1, func2):

        out0 = rhoX(r, rpower, rhopower)

        if func1 is not None:

            out1 = proffdir[func1](r, dorder1)

        if func2 is not None:

            out2 = proffdir[func2](r, dorder2)

    elif (func1 == 'lho') and (func2 == None):

        out0 = r**rpower
        out1 = rhoXlhoX(r, rhopower, dorder1)

    elif (func1 is not None) and (func2 == 'lho'):

        out0 = r**rpower
        out1 = proffdir[func1](r, dorder1)
        out2 = rhoXlhoX(r, rhopower, dorder2)

    return out0*out1*out2
    


# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
