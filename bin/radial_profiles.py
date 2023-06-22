import numpy as np
import utils as ut
import parameters as par
import numpy.polynomial.chebyshev as ch



def twozone(r,args):
    '''
    Symmetrized dT/dr (dimensionless), extended to negative r
    rc is the transition radius, h the transition width, sym is 1 or -1
    depending on the radial parity desired
    ** Neutrally buoyant inner zone, stratified outer zone ** (Vidal2015)
    '''
    rc  = args[0]
    h   = args[1]
    sym = args[2]

    out = np.zeros_like(r)
    for i,x in enumerate(r):
        if x >= 0 :
            out[i] = (1 + np.tanh( 2*(x-rc)/h  ))/2
        elif x < 0 :
            out[i] = sym*(1 + np.tanh( 2*(abs(x)-rc)/h  ))/2
    return out



def BVprof(r,args=None):
    '''
    Symmetrized dT/dr (dimensionless), extended to negative r.
    Define this function so that it is an odd function of r
    '''
    out = np.zeros_like(r)
    for i,x in enumerate(r):
        out[i] = x         # dT/dr propto r like in Dintrans1999
        #out[i] = x*abs(x)  # dT/dr propto r^2
        #out[i] = x**3      # dT/dr propto r^3
        #rc = args[0]
        #h  = args[1]
        #if abs(x) < rc/2 :
        #    out[i] = np.tanh( 4*x/h )
        #elif x >= rc/2 :
        #    out[i] = 0.5*(1 - np.tanh( 4*(x-rc)/h  ))
        #elif x <= -rc/2 :
        #    out[i] = -0.5*(1 - np.tanh( 4*(abs(x)-rc)/h  ))
    return out



#------------------------------------------
# Anelastic profiles
#------------------------------------------

def entropy_gradient(r):
    out = np.zeros_like(r)
    return out



def density(r):
    out = np.ones_like(r)
    return out



def log_density(r):
    out = np.log(density(r))
    return out



def temperature(r):
    out = np.ones_like(r)
    return out



def log_temperature(r):
    out = np.log(temperature(r))
    return out



def alpha(r):
    out = np.ones_like(r)
    return out



def viscosity(r):  # kinematic nu
    out = np.ones_like(r)
    return out



def thermal_diffusivity(r):
    out = np.ones_like(r)
    return out



def kappa_rho(r):
    out = thermal_diffusivity(r)*density(r)
    return out



def buoFac(r):
    '''
    Profile of buoyancy = rho*alpha*T*g
    gravity is multiplied later in Cheb space
    '''
    out = density(r)*alpha(r)*temperature(r)#*gravity(r)
    return out


def gravCoeff():
    '''
    Integrates density profile in Cheb space and gives Cheb
    coefficients of gravity profile, normalized to the value
    at the outer boundary. This works. We checked.
    '''
    ck = ut.chebco_f(density,par.N,par.ricb,ut.rcmb,par.tol_tc)

    x0 = -(par.ricb + ut.rcmb)/(ut.rcmb - par.ricb)
    gk = (ut.rcmb - par.ricb)/2. * ch.chebint(ck,lbnd=x0)

    g0 = ch.chebval(1,gk)
    out = gk/g0

    return out[:par.N]



#------------------------------------------
# Magnetic : Variable conductivity
#------------------------------------------

def conductivity(r):
    '''
    This function needs to be an even function of r when ricb=0
    '''
    out = np.ones_like(r)
    return out



def magnetic_diffusivity(r):
    out = 1./conductivity(r)
    return out



def eta_rho(r):
    out = magnetic_diffusivity(r)*density(r)
    return out


#------------------------------------------
