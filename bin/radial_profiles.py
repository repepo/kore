import numpy as np
import utils as ut
import parameters as par
#import pygyre as gy
#import scipy.interpolate as si
#import mesa_reader as mr



def density(r, args):  # rᵃ ρ

    rpower = args
    out = np.zeros_like(r)
    out = (1-r**2)**par.density_beta

    return (r**rpower)*out



def ddensity(r, args):  # rᵃ ρ'

    [Dorder, rpower, tol] = args 
    dck = ut.chebco_rf( density, 0, par.N, par.ricb, ut.rcmb, tol, 0)
    out = ut.funcheb( dck, r, par.ricb, ut.rcmb, Dorder)

    return (out.T*(r**rpower)).T



def d1_density(r, args):
    
    [Dorder, rpower, tol] = args 
    out = ddensity(r, [1,0,tol])[:,1]

    return out


def d2_density(r, args):
    
    [Dorder, rpower, tol] = args 
    out = ddensity(r, [2,0,tol])[:,2]

    return out


def density2(r, args): # rᵃ ρ²

    rpower = args
    out = np.zeros_like(r)
    out = density(r, 0) * density(r, rpower)

    return out



def logrho1(r, args):  # rᵃ ρ ρ'

    [rpower, tol] = args 
    out = np.zeros_like(r)
    out = density(r, rpower) * ddensity(r, [1, 0, tol])[:,1]

    return out



def logrho2(r, args):  # rᵃ (ρ ρ'' - ρ'ρ')

    [rpower, tol] = args
    out = np.zeros_like(r)
    out = density(r, 0) * ddensity(r, [2, 0, tol])[:,2] - ( ddensity(r, [1, 0, tol])[:,1] )**2

    return (out.T*(r**rpower)).T



# def twozone(r,args):
#     '''
#     Symmetrized dT/dr (dimensionless), extended to negative r
#     rc is the transition radius, h the transition width, sym is 1 or -1
#     depending on the radial parity desired
#     ** Neutrally buoyant inner zone, stratified outer zone ** (Vidal2015)
#     '''
#     rc  = args[0]
#     h   = args[1]
#     sym = args[2]

#     out = np.zeros_like(r)
#     for i,x in enumerate(r):
#         if x >= 0 :
#             out[i] = (1 + np.tanh( 2*(x-rc)/h  ))/2
#         elif x < 0 :
#             out[i] = sym*(1 + np.tanh( 2*(abs(x)-rc)/h  ))/2
#     return out



# def BVprof(r,args=None):
#     '''
#     Symmetrized dT/dr (dimensionless), extended to negative r.
#     Define this function so that it is an odd function of r
#     '''
#     out = np.zeros_like(r)
#     for i,x in enumerate(r):
#         out[i] = x         # dT/dr propto r like in Dintrans1999
#         #out[i] = x*abs(x)  # dT/dr propto r^2
#         #out[i] = x**3      # dT/dr propto r^3
#         #rc = args[0]
#         #h  = args[1]
#         #if abs(x) < rc/2 :
#         #    out[i] = np.tanh( 4*x/h )
#         #elif x >= rc/2 :
#         #    out[i] = 0.5*(1 - np.tanh( 4*(x-rc)/h  ))
#         #elif x <= -rc/2 :
#         #    out[i] = -0.5*(1 - np.tanh( 4*(abs(x)-rc)/h  ))
#     return out



# #-------------------------------------------------------------------------------------------------------------
# # -------------------------------------------------------------------------- User-provided background profiles
# #-------------------------------------------------------------------------------------------------------------

# def density(r):
#     '''
#     Density, normalized by its value at the star's center.
#     '''
#     #m = gy.read_model(par.model)
#     #xm = m['x']
#     #ym = m['rho/rho_0']
#     prf = mr.MesaData(par.model)
#     xm = np.flip(prf.data('radius_cm')) 
#     ym = np.flip(prf.data('density'))
#     interp = si.Akima1DInterpolator(xm/xm[-1], ym/ym[0])
#     out = np.zeros_like(r)
#     for i,x in enumerate(r):
#         if x>=0:
#             out[i] = interp(x)
#         else:
#             out[i] = interp(-x)  # even function of r
#     return out  #even


# def gravity(r):
#     '''
#     Magnitude of the gravitational acceleration, normalized by its value at the star's surface. 
#     '''
#     #m = gy.read_model(par.model)
#     #xm = m['x']
#     #ym = m['dtheta']
#     prf = mr.MesaData(par.model)
#     xm = np.flip(prf.data('radius_cm'))
#     ym = np.flip(prf.data('grav'))
#     interp = si.Akima1DInterpolator(xm/xm[-1], ym/ym[-1])
#     out = np.zeros_like(r)
#     for i,x in enumerate(r):
#         if x>=0:
#             out[i] = interp(x)
#         else:
#             out[i] = -interp(-x)  # odd function of r
#     return out  # odd

    
# def temperature(r):
#     '''
#     Temperature, normalized by its value at the star's center.
#     '''
#     #m = gy.read_model(par.model)
#     #xm = m['x']
#     #ym = m['theta']
#     prf = mr.MesaData(par.model)
#     xm = np.flip(prf.data('radius_cm'))
#     ym = np.flip(prf.data('temperature'))
#     interp = si.Akima1DInterpolator(xm/xm[-1], ym/ym[0])
#     out = np.zeros_like(r)
#     for i,x in enumerate(r):
#         if x>=0:
#             out[i] = interp(x)
#         else:
#             out[i] = interp(-x)  # even function of r
#     return out


# def BruVa2(r):
#     '''
#     The squared, dimensionless Brunt-Vaisala frequency, in units of GM/R^3. 
#     '''
#     #m = gy.read_model(par.model)
#     #xm = m['x']
#     #ym = m['dtheta']

#     prf = mr.MesaData(par.model)
#     xm = np.flip(prf.data('radius_cm'))
#     ym = 3 * np.flip(prf.data('brunt_N2_dimensionless'))  # MESA uses 3*GM/R^3 instead of GM/R^3
#     ym[xm/xm[-1]>par.r_cutoff] = 0  # zero out the atmosphere, stinkin atmosphere
#     ym[ym<0]=0  # zero out convective zones

#     interp = si.Akima1DInterpolator(xm/xm[-1], ym)  
#     out = np.zeros_like(r)
#     for i,x in enumerate(r):
#         if x>=0:
#             out[i] = interp(x)
#         else:
#             out[i] = interp(-x)  # even function of r
#     #out = r**2-r**4  # even
#     return out  # even


# def viscosity(r):
#     '''
#     Kinematic viscosity (aka momentum diffusivity)
#     '''
#     out = np.zeros_like(r)
#     return out  # even function of r

        
# def kappa(r):
#     '''
#     Thermal diffusivity
#     '''
#     out = np.ones_like(r)  # even function of r
#     return out


# # ---------------------------------------------------------------------------------------
# # The functions below are directly linked to the ones above, no user intervention needed.
# # ---------------------------------------------------------------------------------------

# def rog(r):
#     '''
#     density * gravitational acceleration
#     '''
#     out = density(r) * gravity(r)  # even * odd = odd
#     return out # odd function of r

    
# def krT(r):
#     '''
#     Thermal diffusivity * density * temperature
#     '''
#     out = kappa(r) * density(r) * temperature(r)
#     return out  # even*even*even = even function of r

    
# def roT(r):
#     '''
#     density * temperature
#     '''
#     out = density(r) * temperature(r)
#     return out  # even*even = even function of r


# # def dTdr(r):
# #     '''
# #     Gradient of temperature.
# #     '''
# #     #dck = ut.chebify(tempe,1,par.tol)[:,1]
# #     #out = ut.funcheb( dck, r, par.ricb, ut.rcmb, 0)
# #     m = gy.read_model(par.model)
# #     xm = m['x']
# #     ym = m['dtheta'] * m['z'][-1] 
# #     interp = si.make_interp_spline(xc, ym, k=3)
# #     out = np.zeros_like(r)
# #     for i,x in enumerate(r):
# #         if x>=0:
# #             out[i] = interp(x)
# #         else:
# #             out[i] = -interp(-x)  # odd
# #     return out  # odd function of r


# def TdS(r):
#     '''
#     r * temperature * entropy gradient (Glatzmaier2014, eq. 12.9),
#     par.gamma is the adiabatic index (e.g. use par.gamma=5/3 for a monoatomic perfect gas)
#     '''
#     # m = gy.read_model(par.model)
#     # z0 = m['z'][-1]
#     # dtheta0 = m['dtheta'][-1]
#     # n = m['n_poly'][-1]  # we assume there's just one single polytrope
#     # gamma0 = (n+1) * z0 * (-dtheta0) * (1-1/par.gamma)
#     # out = dTdr(r) + gamma0 * gravity(r)  # odd+odd = odd
    
#     prf = mr.MesaData(par.model)
    
#     xm = np.flip(prf.data('radius_cm'))
#     r0 = xm[-1]  # star radius
    
#     g = np.flip(prf.data('grav'))
#     g0 = g[-1]  # gravity at the surface
    
#     T = np.flip(prf.data('temperature'))
#     T0 = T[0]  # temperature at the center
    
#     N2 = np.flip(prf.data('brunt_N2'))/(g0/r0)  # dimensionless BV freq squared (i.e. in units of GM/R^3)
#     N2[xm/r0>par.r_cutoff] = 0  # zero out the atmosphere, stinkin atmosphere
    
#     ym = N2 * (T/T0) / (g/g0)  # if all variables dimensionless then T*(ds/dr) = N2*T/g
#     ym[ym<0]=0  # zero out convective zones
    
#     interp = si.Akima1DInterpolator(xm/r0, ym)
#     out = np.zeros_like(r)
#     for i,x in enumerate(r):
#         if x>=0:
#             out[i] = interp(x)
#         else:
#             out[i] = -interp(-x)  # N2*T/g is an odd function of r
    
#     return out  # odd function of r


# def tds(r):
#     out = BruVa2(r) * temperature(r) / gravity(r)
#     return out  # even * even / odd = odd

# # --------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------




# #------------------------------------------
# # Magnetic : Variable conductivity
# #------------------------------------------

# def conductivity(r):
#     '''
#     This function needs to be an even function of r when ricb=0
#     '''
#     out = np.ones_like(r)
#     return out


# def magnetic_diffusivity(r):
#     out = 1./conductivity(r)
#     return out


# def eta_rho(r):
#     out = magnetic_diffusivity(r)*density(r)
#     return out
