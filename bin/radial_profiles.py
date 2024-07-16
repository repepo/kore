import numpy as np
import utils as ut
import parameters as par



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

#---------------------------------------------------------
# Get required parameters according to interior model
#---------------------------------------------------------

def getPolyParams():
    '''
    This function defines useful parameters required to compute polytropic profiles
    defined by rho = T^n , p = rho^(1 + 1/n) and ideas gas law : p = rho T
    '''

    radratio = par.ricb/ut.rcmb
    Nrhofac = np.exp(par.Nrho/par.polind)
    c0 = ( (-2*par.g2 - (2*par.g0 + par.g1 - 2*par.g2)*Nrhofac*radratio
            + 2*par.g0*radratio**2 + par.g1*radratio**3)/
            ((-1 + radratio)*(2*par.g2 + radratio*(2*par.g0 + par.g1 + par.g1*radratio))) )
    c1 = ( (-2*(-1 + Nrhofac)*radratio)/
          ((-1 + radratio)*(2*par.g2 + radratio*(2*par.g0 + par.g1 + par.g1*radratio))*ut.rcmb) )

    return c0,c1,Nrhofac


def getEOSparams():
    '''
    This function allows the user to set custom profiles defined by polynomial
    coefficients. Coefficients are ordered from highest order to constant, as
    required by numpy.polyval .
    '''
    # Set constant values for any missing profiles

    coeffDens = [1]; coeffTemp = [1]; coeffAlpha = [1]; coeffGrav = [1]

    if par.interior_model == 'jupiter':
        coeffDens = [-3428.5223691617484,9297.45806689145,-9053.015978040878,
                3692.8869368199303,-970.5219501076829,455.9692566566673]
        coeffTemp = [-234.89315693284604,629.7550411434713,-620.7785004558405,
                264.5131393963175,-60.42859069469556,23.325397786820883]
        coeffAlpha = [  7772.115489487784, -32570.155177385783,57445.909371048925,
                -55346.03609488902, 31580.21652524248,-10792.31056102935,
                2120.720011656385,-212.80140120580023,2.8282797790308654]
        coeffGrav = [  359.26949772994396, -1647.4951402273955, 3114.971302818801,
                      -3121.9995586185805, 1768.8806269264637, -552.869900057094,
                       78.85393983296116, 1.8991569550910268, -0.5216321175523105]
    elif par.interior_model == 'saturn':
        coeffTemp= [ 2746322.598433222, -12819667.849508610, 24913613.036177561,
        -26159709.956658602, 16027297.768664116, -5718555.197998112,
        1089237.928095523, -77880.376328818 ]
        coeffDens= [ 2746322.598433222, -12819667.849508610, 24913613.036177561,
        -26159709.956658602, 16027297.768664116, -5718555.197998112]
        coeffGrav= [ 2746322.598433222, -12819667.849508610, 24913613.036177561,
        1089237.928095523, -77880.376328818 ]
        coeffAlpha= [ 2746322.598433222, -12819667.849508610, 24913613.036177561,
        -26159709.956658602, 16027297.768664116, -5718555.197998112,
        1089237.928095523, -77880.376328818 ]
    # elif par.interior_model == 'pns': # To be implemented
    #     pass

    return coeffDens, coeffTemp, coeffAlpha, coeffGrav


if par.interior_model == 'polytrope':
    c0,c1,Nrhofac = getPolyParams()

    def gint(r): # Indefinite integral of gravity
        return par.g0*r + (par.g1*r**2)/(2.*ut.rcmb) - (par.g2*ut.rcmb**2)/r

    # Special case: Gravity g(r) = 1/r^2

    if par.g2 == 1 and (par.g0+par.g1+par.g2) == 1:
        def zeta(r):
            return -c1 * gint(r) + c0
else:
    coeffDens, coeffTemp, coeffAlpha, coeffGrav = getEOSparams()

#-------------------------------------------------------------
# Compute profiles depending on interior model
#-------------------------------------------------------------


def entropy_gradient(r,args=None):
    if args is not None:
        ampStrat = args[0]
        rStrat   = args[1]
        thickStrat=args[2]
        slopeStrat=args[3]

        out = ( 0.25 * (ampStrat+1.) * (1.+np.tanh(slopeStrat*(r-rStrat))) *
                                    (1.-np.tanh(slopeStrat*(r-rStrat-thickStrat)))
                                        - 1. )
    elif ( par.interior_model == 'polytrope' and
          par.g2 == 1 and (par.g0+par.g1+par.g2) == 1): #Analytical solution is only possible when gravity is 1/r^2, there is probably a better way to write this.

        out = ( ((-1 + Nrhofac)*Nrhofac**par.polind * par.polind*par.ricb)/
               ((-1 + Nrhofac**par.polind)*r**2*(-1 + par.ricb/ut.rcmb)) /
                zeta(r)**(par.polind+1) )
    else:
        out = np.zeros_like(r)
    return out


def temperature(r):
    if par.interior_model is not None:
        if par.interior_model == 'polytrope':
            out = -c1*gint(r) + c0
        else:
            out = np.polyval(coeffTemp,par.r_cutoff * r)
    else:
        out = np.ones_like(r)
    return out

def log_temperature(r):
    out = np.log(temperature(r))
    return out



def density(r):
    if par.interior_model is not None:
        if par.interior_model == 'polytrope':
            out = temperature(r)**par.polind
        else:
            out = np.polyval(coeffDens,par.r_cutoff * r)
    else:
        out = np.ones_like(r)
    return out

def log_density(r):
    out = np.log(density(r))
    return out


def alpha(r):
    if par.interior_model is not None:
        if par.interior_model == 'polytrope':
            out = 1/temperature(r)
        else:
            out = np.exp(np.polyval(coeffAlpha,par.r_cutoff * r)) #Fit is to ln alpha
    else:
        out = np.ones_like(r)
    return out



def viscosity(r):  # kinematic nu
    out = np.ones_like(r)
    return out



def thermal_diffusivity(r):
    out = np.ones_like(r)
    return out

def log_thermal_diffusivity(r):
    out = np.log(thermal_diffusivity(r))
    return out


def kappa_rho(r):
    out = thermal_diffusivity(r)*density(r)
    return out

def heat_source(r,eps0=0):
    out = eps0
    return out

def epsilon_h(r,eps0=0):
    out = (eps0*r*heat_source(r,eps0)/
           (density(r)*temperature(r)*
            thermal_diffusivity(r)))
    return out


def gravity(r):
    if par.interior_model is not None:
        if par.interior_model == 'polytrope':
            out = par.g0 + par.g1 * r/ut.rcmb + par.g2 * ut.rcmb**2/r**2
        else:
            out = np.polyval(coeffGrav,par.r_cutoff * r)
    else:
        out = r
    return out


def buoFac(r):
    '''
    Profile of buoyancy = rho*alpha*T*g
    If autocomputed, gravity is multiplied later in Cheb space
    '''
    if par.autograv:
        out = density(r)*alpha(r)*temperature(r)
    else:
        out = density(r)*alpha(r)*temperature(r)*gravity(r)
    return out

def h_mag(r):
    if par.B0 == "axial":
        out = 0.5*r
    elif par.B0 == "dipole":
        out = 0.5/r**2
    elif par.B0 == "Luo_S1":
        out = r*(5-3*r**2)
    elif par.B0 == "G21 dipole":
        out = r/6 - r**3/10
    elif par.B0 == "Luo_S2":
        out = r**2*(157-296*r**2+143*r**4)
    else: 
        # Print error message if other choice is made (FDM not coded yet)
        print("Error: B0 not recognized")
    
    cnorm = ut.B0_norm()
    return cnorm * out


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

def write_profiles():
    i = np.arange(0, par.N)
    xi = np.cos(np.pi * (i + 0.5) / par.N)

    if par.ricb > 0:
        ri = (par.ricb + (ut.rcmb - par.ricb) * (xi + 1) / 2.)
    elif par.ricb == 0 :
        ri = ut.rcmb * xi

    rho0 = density(ri)
    temp0= temperature(ri)
    alpha0=alpha(ri)
    grav = gravity(ri)
    dsdr = entropy_gradient(ri)
    cond = conductivity(ri)

    X = np.array([ri,rho0,temp0,alpha0,grav,dsdr,cond]).transpose()

    header = ["r", "rho0", "temp0", "alpha0", "grav", "dsdr", "cond"]

    # try:
    #     import pandas as pd
    #     df = pd.DataFrame(X,columns=header)
    #     df.to_csv('profiles.dat',sep=' ',float_format='%.4f',index=False)
    # except:
    #     np.savetxt('profiles.dat',X,fmt='%.4f',header=' '.join(header),delimiter=' ')

    np.savetxt('profiles.dat',X,fmt='%.4f',header=' '.join(header),delimiter=' ')
#------------------------------------------
