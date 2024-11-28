import numpy as np
import utils as ut
import parameters as par
#import pygyre as gy
import scipy.interpolate as si
import mesa_reader as mr


#-------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------- Mesa-derived background profiles
#-------------------------------------------------------------------------------------------------------------

def density(r):
    '''
    Density, normalized by its value at the star's center.
    '''
    #m = gy.read_model(par.mesa_file)
    #xm = m['x']
    #ym = m['rho/rho_0']
    prf = mr.MesaData(par.mesa_file)
    xm = np.flip(prf.data('radius_cm'))
    ym = np.flip(prf.data('density'))
    interp = si.Akima1DInterpolator(xm/xm[-1], ym/ym[0])
    out = np.zeros_like(r)
    for i,x in enumerate(r):
        if x>=0:
            out[i] = interp(x)
        else:
            out[i] = interp(-x)  # even function of r
    return out  #even


def gravity(r):
    '''
    Magnitude of the gravitational acceleration, normalized by its value at the star's surface.
    '''
    #m = gy.read_model(par.mesa_file)
    #xm = m['x']
    #ym = m['dtheta']
    prf = mr.MesaData(par.mesa_file)
    xm = np.flip(prf.data('radius_cm'))
    ym = np.flip(prf.data('grav'))
    interp = si.Akima1DInterpolator(xm/xm[-1], ym/ym[-1])
    out = np.zeros_like(r)
    for i,x in enumerate(r):
        if x>=0:
            out[i] = interp(x)
        else:
            out[i] = -interp(-x)  # odd function of r
    return out  # odd


def temperature(r):
    '''
    Temperature, normalized by its value at the star's center.
    '''
    #m = gy.read_model(par.mesa_file)
    #xm = m['x']
    #ym = m['theta']
    prf = mr.MesaData(par.mesa_file)
    xm = np.flip(prf.data('radius_cm'))
    ym = np.flip(prf.data('temperature'))
    interp = si.Akima1DInterpolator(xm/xm[-1], ym/ym[0])
    out = np.zeros_like(r)
    for i,x in enumerate(r):
        if x>=0:
            out[i] = interp(x)
        else:
            out[i] = interp(-x)  # even function of r
    return out


def BruVa2(r):
    '''
    The squared, dimensionless Brunt-Vaisala frequency, in units of GM/R^3.
    '''
    #m = gy.read_model(par.mesa_file)
    #xm = m['x']
    #ym = m['dtheta']

    prf = mr.MesaData(par.mesa_file)
    xm = np.flip(prf.data('radius_cm'))
    ym = 3 * np.flip(prf.data('brunt_N2_dimensionless'))  # MESA uses 3*GM/R^3 instead of GM/R^3
    ym[xm/xm[-1]>par.r_cutoff] = 0  # zero out the atmosphere, stinkin atmosphere
    ym[ym<0]=0  # zero out convective zones

    interp = si.Akima1DInterpolator(xm/xm[-1], ym)
    out = np.zeros_like(r)
    for i,x in enumerate(r):
        if x>=0:
            out[i] = interp(x)
        else:
            out[i] = interp(-x)  # even function of r
    #out = r**2-r**4  # even
    return out  # even

# ---------------------------------------------------------------------------------------
# The functions below are directly linked to the ones above, no user intervention needed.
# ---------------------------------------------------------------------------------------

def thermal_diffusivity(r):
    out = np.ones_like(r)
    return out

def log_thermal_diffusivity(r):
    out = np.log(thermal_diffusivity(r))
    return out

def buoFac(r):

    #Profile of buoyancy = rho*alpha*T*g
    #Ideal gas : alpha = 1/T

    out = density(r)*gravity(r)
    return out

def krT(r):
    '''
    Thermal diffusivity * density * temperature
    '''
    out = thermal_diffusivity(r) * density(r) * temperature(r)
    return out  # even*even*even = even function of r


def roT(r):
    '''
    density * temperature
    '''
    out = density(r) * temperature(r)
    return out  # even*even = even function of r


# def dTdr(r):
#     '''
#     Gradient of temperature.
#     '''
#     #dck = ut.chebify(tempe,1,par.tol)[:,1]
#     #out = ut.funcheb( dck, r, par.ricb, ut.rcmb, 0)
#     m = gy.read_model(par.mesa_file)
#     xm = m['x']
#     ym = m['dtheta'] * m['z'][-1]
#     interp = si.make_interp_spline(xc, ym, k=3)
#     out = np.zeros_like(r)
#     for i,x in enumerate(r):
#         if x>=0:
#             out[i] = interp(x)
#         else:
#             out[i] = -interp(-x)  # odd
#     return out  # odd function of r


def TdS(r):
    '''
    r * temperature * entropy gradient (Glatzmaier2014, eq. 12.9),
    par.gamma is the adiabatic index (e.g. use par.gamma=5/3 for a monoatomic perfect gas)
    '''
    # m = gy.read_model(par.mesa_file)
    # z0 = m['z'][-1]
    # dtheta0 = m['dtheta'][-1]
    # n = m['n_poly'][-1]  # we assume there's just one single polytrope
    # gamma0 = (n+1) * z0 * (-dtheta0) * (1-1/par.gamma)
    # out = dTdr(r) + gamma0 * gravity(r)  # odd+odd = odd

    prf = mr.MesaData(par.mesa_file)

    xm = np.flip(prf.data('radius_cm'))
    r0 = xm[-1]  # star radius

    g = np.flip(prf.data('grav'))
    g0 = g[-1]  # gravity at the surface

    T = np.flip(prf.data('temperature'))
    T0 = T[0]  # temperature at the center

    N2 = np.flip(prf.data('brunt_N2'))/(g0/r0)  # dimensionless BV freq squared (i.e. in units of GM/R^3)
    N2[xm/r0>par.r_cutoff] = 0  # zero out the atmosphere, stinkin atmosphere

    ym = N2 * (T/T0) / (g/g0)  # if all variables dimensionless then T*(ds/dr) = N2*T/g
    ym[ym<0]=0  # zero out convective zones

    interp = si.Akima1DInterpolator(xm/r0, ym)
    out = np.zeros_like(r)
    for i,x in enumerate(r):
        if x>=0:
            out[i] = interp(x)
        else:
            out[i] = -interp(-x)  # N2*T/g is an odd function of r

    return out  # odd function of r


def tds(r):
    out = BruVa2(r) * temperature(r) / gravity(r)
    return out  # even * even / odd = odd

def log_density(r):
    out = np.log(density(r))
    return out

def viscosity(r):
    return np.zeros_like(r)

def log_temperature(r):
    out = np.log(temperature(r))
    return out

def kappa_rho(r):
    out = thermal_diffusivity(r)*density(r)
    return out
