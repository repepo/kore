import numpy as np
import scipy.special as ss
import utils as ut
from parameters import par
import smoothed_sun as sms



# -------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------ Structure profiles
# -------------------------------------------------------------------------------------------------------------

class user_defined_profiles():  # --------------------------------------------------- As user-defined functions
    '''
    Background profiles defined directly as functions. Modify as needed.
    '''

    def density(self, r, rpower):  # ------------------ ρ(r)
        # r1 = r*par.aux0;
        # out = np.ones_like(r)
        # out = 1-r1**2                      # Wu2005    β=1
        # out = (1-r1**2)**2                 # Wu2005    β=2
        # out = np.sin(np.pi*r1)/(np.pi*r1)  # Polytrope n=1
        out = np.exp(self.logrho(r,0))
        # --------------------------------------------------
        return (r**rpower)*out


    def logrho(self, r, rpower):  # ------ log of the density: log(ρ(r))
        r1 = r*par.aux0;
        # out = np.log(np.ones_like(r))
        # out = np.log(1-r1**2)                          # Wu2005    β=1
        out = np.log( (1-r1**2)**2 )                   # Wu2005    β=2
        # out = np.log( np.sin(np.pi*r1)/(np.pi*r1) )    # Polytrope n=1
        # out = sms.log_density(r1)
        # --------------------------------------------------------------
        return (r**rpower)*out


    def gravity(self, r, rpower):  # ------------------------------------------------------ g(r)
        r1 = r*par.aux0
        # out = (5*r1 - 3*r1**3)/2.                                              # Wu2005    β=1
        out = (r1*(35 - 42*r1**2 + 15*r1**4))/8.                               # Wu2005    β=2
        # out = (-(np.pi*r1*np.cos(np.pi*r1)) + np.sin(np.pi*r1))/(np.pi*r1**2)  # Polytrope n=1 
        # --------------------------------------------------------------------------------------
        return (r**rpower)*out


    def pressure(self, r, rpower):  # Normalized to 1 at r=0 ----------------- p(r)
        r1 = r*par.aux0
        # out = -0.5*((-2 + r1**2)*(-1 + r1**2)**2)                 # Wu2005    β=1
        out = -((-1 + r1**2)**3*(26 - 27*r1**2 + 9*r1**4))/26.    # Wu2005    β=2
        # out = np.sin(np.pi*r1)**2/(np.pi**2*r1**2)                # Polytrope n=1  
        # -------------------------------------------------------------------------
        return (r**rpower)*out


    def dlog_p(self,r):  # ----------------------------------------------------------------------- d(ln p)/dr
        r1 = r*par.aux0
        # out = (2*r1*(-5 + 3*r1**2))/(2 - 3*r1**2 + r1**4)                                   # Wu2005    β=1
        out = (6*r1*(35 - 42*r1**2 + 15*r1**4))/((-1 + r1**2)*(26 - 27*r1**2 + 9*r1**4))    # Wu2005    β=2
        # out = (-2/r1) + (2*np.pi/np.tan(np.pi*r1))                                          # Polytrope n=1
        # --------------------------------------------------------------------------------------------------- 
        return out
        
        
    def dlog_rho(self,r):  # --------------------------- d(ln ρ)/dr
        r1 = r*par.aux0
        # out = (2*r1)/(-1 + r1**2)                 # Wu2005    β=1
        out = (4*r1)/(-1 + r1**2)                 # Wu2005    β=2
        # out = (-1/r1) + (np.pi/np.tan(np.pi*r1))  # Polytrope n=1
        # ---------------------------------------------------------
        return out
        
        
    # def Gamma1_isentropic(self,r):  # This is the isentropic Γ₁  
    #     out = self.dlog_p(r)/self.dlog_rho(r)
    #     return out
    # def Gamma1(self,r,a,b,c):  # The resulting first adiabatic coefficient Γ₁
    #     out = self.Gamma1_isentropic(r) + ut.erf_transition(r,a,b,c)
    #     return out
    # def gradS(self,r,a,b,c):  # The background entropy gradient
    #     out = np.zeros_like(r)
    #     x = abs(r)<1
    #     out[x] = ( self.dlog_p(r[x])/self.Gamma1(r[x],a,b,c) ) - self.dlog_rho(r[x])
    #     return out
    def gradS(self, r, x1, w1, x2, w2, A):
        return ut.erf_top_hat(r, x1,w1,x2,w2,A)


    def pdSdr(self, r, rpower):
        # ------------------------ p(r) dS/dr
        out = np.zeros_like(r)
        x = abs(r)<1
        out[x] = self.pressure(r[x],0)*self.gradS(r[x], par.aux1, par.aux2, par.aux3, par.aux4, par.aux5)
        # -----------------------------------
        return (r**rpower)*out


    def viscosity(self, r, rpower):
        # -------------------------------------------------------------------- v(r)
        out = np.ones_like(r)
        #out = 1/self.density(r,0)
        #out = ( par.visc0 + 0.5*(1-par.visc0)*(1 + ss.erf((r-par.rvisc)/par.hvisc)) )/self.density(r,0)
        # -------------------------------------------------------------------------
        return (r**rpower)*out


    def thermal_diffusivity(self, r, rpower):
        # ------------------------------ κ(r)
        out = np.ones_like(r)
        # -----------------------------------
        return (r**rpower)*out



class profiles_from_file():  # -------------------------------------------------------- As read from model file
    '''
    Background profiles as read from a model file, types can be mesa, gyre or astropy table
    '''

    def density(self, r, rpower):
        # ------------------------------ ρ(r)
        out = ut.load_model(r,'density')
        # -----------------------------------
        #print('hello')
        return (r**rpower)*out


    def gravity(self, r, rpower):
        # ------------------------------ g(r)
        out = ut.load_model(r,'gravity')
        # -----------------------------------
        return (r**rpower)*out


    def pressure(self, r, rpower):
        # ------------------------------ p(r)
        out = ut.load_model(r, 'pressure')
        # -----------------------------------
        return (r**rpower)*out


    # def dlog_p(self,r):
    #     tol = 1e-12; Dorder = 1
    #     dp = ut.fonzie(prf.pressure, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)
    #     out = dp[:,1]/dp[:,0]
    #     return out
    # def dlog_rho(self,r):
    #     tol = 1e-12; Dorder = 1
    #     drho = ut.fonzie(prf.density, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)
    #     out = drho[:,1]/drho[:,0]
    #     return out   
    # def Gamma1_isentropic(self,r):  # This is the isentropic Γ₁  
    #     out = self.dlog_p(r)/self.dlog_rho(r)
    #     #out = (4/3)*np.ones_like(r)
    #     return out
    # def faux(self,r,a,b,c):  # cosine function (soft edges)
    #     out = np.zeros_like(r)
    #     x = (r>a)&(r<b)
    #     out[x] = (c/2)*(1-np.cos(2*np.pi*(r[x]-a)/(b-a)))
    #     out[r<0] = np.flipud(out[r>0])  # make it even
    #     return out
    # def Gamma1(self,r,a,b,c):  # The resulting first adiabatic coefficient Γ₁
    #     out = self.Gamma1_isentropic(r) + self.faux(r,a,b,c)
    #     return out
    # def gradS(self,r,a,b,c):  # The background entropy gradient
    #     out = np.zeros_like(r)
    #     x = abs(r)<1
    #     out[x] = ( self.dlog_p(r[x])/self.Gamma1(r[x],a,b,c) ) - self.dlog_rho(r[x])
    #     return out


    # def gradS(self, r):

    #     A = 2.12
    #     a1=0.045; ka1=1/0.04
    #     a2=0.713; ka2=1/0.025
    #     B = 0.40
    #     b1=0.04; kb1=1/0.025
    #     b2=0.12; kb2=1/0.04
    #     D = 0.40
    #     d1=0.22; kd1=1/0.09
    #     d2=0.620; kd2=1/0.09
    #     outA = ut.erf_transition(r,a1,ka1,A)-ut.erf_transition(r,a2,ka2,A)
    #     outB = ut.erf_transition(r,b1,kb1,B)-ut.erf_transition(r,b2,kb2,B)
    #     outD = ut.erf_transition(r,d1,kd1,D)-ut.erf_transition(r,d2,kd2,D)

    #     return -(outA+outB+outD)


    def gradS(self, r, x1, w1, x2, w2, A):
        return ut.erf_top_hat(r, x1,w1,x2,w2,A)


    def pdSdr(self, r, rpower):
        # ------------------------ p(r) dS/dr
        #out = ut.load_model(r,'pdSdr')
        out = np.zeros_like(r)
        x = abs(r)<1
        out[x] = self.pressure(r[x],0)*self.gradS(r[x], par.aux1, par.aux2, par.aux3, par.aux4, par.aux5)
        #out = self.pressure(r,0)*self.gradS(r) 
        # -----------------------------------
        return (r**rpower)*out


    def viscosity(self, r, rpower):  # def here only
        # ------------------------------ v(r)
        out = np.ones_like(r)
        #out = 1/self.density(r,0)
        # -----------------------------------
        return (r**rpower)*out


    def thermal_diffusivity(self, r, rpower):  # def here only
        # ------------------------------ κ(r)
        out = np.ones_like(r)
        # -----------------------------------
        return (r**rpower)*out



class Boussinesq_profiles():  # -------------------------------------------------------- As Boussinesq profiles 
    '''
    Background profiles as required by the Boussinesq aprroximation
    '''

    def density(self, r, rpower):
        # ------------------------------ ρ(r)
        out = np.ones_like(r)
        # -----------------------------------
        return (r**rpower)*out


    def gravity(self, r, rpower):
        # ------------------------------ g(r)
        out = r
        # -----------------------------------
        return (r**rpower)*out


    def pressure(self, r, rpower):
        # ------------------------------ p(r)
        out = np.ones_like(r)
        # -----------------------------------
        return (r**rpower)*out


    def pdSdr(self, r, rpower):
        out = np.zeros_like(r)
        if par.heating == 'internal':
            out = r
        elif par.heating == 'differential':
            out = 1/r**2
        return (r**rpower)*out


    def viscosity(self, r, rpower):
        # ------------------------------ v(r)
        out = np.ones_like(r)
        # -----------------------------------
        return (r**rpower)*out


    def thermal_diffusivity(self, r, rpower):
        # ------------------------------ κ(r)
        out = np.ones_like(r)
        # -----------------------------------
        return (r**rpower)*out



# -------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------- Instantiate the appropriate profile class
# -------------------------------------------------------------------------------------------------------------
if par.model_type == 'user def':
    prf = user_defined_profiles()
elif par.model_type in ['mesa', 'gsm', 'poly', 'astropy table']:
    prf = profiles_from_file()
elif par.model_type == 'Boussinesq':
    prf = Boussinesq_profiles()



# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------- Derived profiles
# -------------------------------------------------------------------------------------------------------------

# def kappress(r, rpower):   # κ p

#     out = prf.thermal_diffusivity(r, 0) * prf.pressure(r, 0)
#     return (r**rpower)*out


# def densityX(r, Dorder):  # ρ⁽ⁿ⁾, radial derivatives of ρ(r)

#     tol = 1e-14
#     out = ut.fonzie( prf.density, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)
#     return out


# def lhoX(r, lhoorder):  # ρⁿ (ln ρ)⁽ⁿ⁾
#     '''
#     Returns the lhoorder derivative of (ln ρ)
#     multiplied by the lhoorder power of ρ
#     in order to cancel any ρ in the denominator.
#     '''

#     out = np.zeros_like(r)
#     dd = densityX(r, lhoorder)
#     d0 = dd[:,0]

#     if lhoorder == 0:
#         out = d0
#     elif lhoorder == 1:
#         d1 = dd[:,1]
#         out = d1
#     elif lhoorder == 2:
#         d1 = dd[:,1]; d2 = dd[:,2]
#         out = d0 * d2 - d1**2
#     elif lhoorder == 3:
#         d1 = dd[:,1]; d2 = dd[:,2]; d3 = dd[:,3]
#         out = 2*d1**3 -3*d0*d1*d2 +(d0**2)*d3
#     elif lhoorder == 4:
#         d1 = dd[:,1]; d2 = dd[:,2]; d3 = dd[:,3]; d4 = dd[:,4]
#         out = -6*d1**4 +12*d0*(d1**2)*d2 -4*(d0**2)*d1*d3 - 3*(d0**2)*(d2**2) +(d0**3)*d4

#     return out


# -------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------- Profile functions for burrito
# -------------------------------------------------------------------------------------------------------------

# def rhoX(r, *args):   # rᵃ ρᵇ  powers of ρ

#     (rpower, rhopower) = args
#     out = np.zeros_like(r)
#     out = prf.density(r, 0)**rhopower

#     return (r**rpower)*out


# def rhoXlhoX(r, *args):   # ρᵃ (ln ρ)⁽ᵇ⁾  derivatives of ρ

#     out = np.zeros_like(r)
#     (rhopower, lhoorder) = args
#     delta = rhopower - lhoorder
#     if delta == 0:
#         out = lhoX(r,lhoorder)
#     else:
#         out = rhoX(r, 0, delta) * lhoX(r, lhoorder)

#     return out


def logrhoX(r, Dorder):

    tol = 1e-14
    out = ut.angine( prf.logrho, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out


def viscoX(r, Dorder):   # Kinematic viscosity

    tol = 1e-14
    out = ut.angine( prf.viscosity, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out


def kappressX(r, Dorder):   # κ(r)ρ(r)T(r)

    tol = 1e-14
    out = ut.angine( prf.thermal_diffusivity(r, 0) * prf.pressure(r, 0), r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out


def pdSdrX(r, Dorder):   # ρ(r)T(r)dS/dr

    tol = 1e-14
    out = ut.angine( prf.pdSdr, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out


def graviX(r, Dorder):   # g(r)

    tol = 1e-14
    out = ut.angine( prf.gravity, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out


def pressX(r, Dorder):   # ρ(r)T(r)

    tol = 1e-14
    out = ut.angine( prf.pressure, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out



# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
#proffdir = { 'lho':rhoXlhoX, 'moe':muX, 'rho':rhoX, 'gra':graviX, 'pss':pressX, 'pdS':pdSdrX, 'kps':kappressX }
proffdir = { 'lho':logrhoX, 'vsc':viscoX, 'gra':graviX, 'pss':pressX, 'pdS':pdSdrX, 'kps':kappressX }
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------


def burrito(r, *args):

    (secx, rpower0, rpower1, func1, dorder1, func2, dorder2, dx) = args

    out01 = r**rpower0
    out02 = prf.density(r,0)**rpower1
    out1  = np.ones_like(r)
    out2  = np.ones_like(r)

    if func1 is not None:

        if func1 == 'lh1':
            out1 = logrhoX(r,1) * logrhoX(r, dorder1)
        elif func1 == 'lh2':
            out1 = logrhoX(r,2) * logrhoX(r, dorder1)
        else:
            out1 = proffdir[func1](r, dorder1)

    if func2 is not None:

        out2 = proffdir[func2](r, dorder2)   

    out = out01*out02*out1*out2

    # simple test for singular behavior at r=rcmb
    # slope = abs( ( out[r==r[1]] - out[r==r[0]] ) / ( r[1] - r[0] ) )
    # #print(args, out[r==r[0]], slope)
    # if slope>1200:
    #     print('Possible divergence at r=1', args, slope) 

    return out


# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------



'''
def visfo_u_D0(r, l):

    L = l*(l+1.)
    u = par.rpower_u
    out = ( -1/3 * L * r**(-5 + u) * (3 * r * (r * (-2 + L - r * lho[:, 1] + r**2 * lho[:, 2]) * vsc[:, 2] 
            + vsc[:, 1] * (2 * (1 + L) + r * (-r * lho[:, 1]**2 - lho[:, 1] * (2 + L - r**2 * lho[:, 2]) 
            + 2 * r * (lho[:, 2] + r * lho[:, 3])))) + vsc[:, 0] * (3 * (-1 + l) * (2 + l) * L 
            + r * (-2 * L * r * lho[:, 1]**2 + 3 * lho[:, 1] * (2 - r**2 * lho[:, 2] + r**3 * lho[:, 3]) 
            + 3 * r * (-2 * lho[:, 2] + r**2 * lho[:, 2]**2 + r * (3 * lho[:, 3] + r * lho[:, 4]))))) )
    return out


def visfo_u_D1(r, l):

    L = l*(l+1.)
    u = par.rpower_u
    out = ( L * r**(-3 + u) * (vsc[:, 1] * (2 * (1 + L) - r * (lho[:, 1] * (2 + r * lho[:, 1]) 
            + 4 * r * lho[:, 2])) - r**2 * lho[:, 1] * vsc[:, 2] + vsc[:, 0] * (lho[:, 1] * (2 * (1 + L) 
            - 3 * r**2 * lho[:, 2]) - 3 * r * (2 * lho[:, 2] + r * lho[:, 3]))) )
    return out


def visfo_u_D2(r, l):

    L = l*(l+1.)
    u = par.rpower_u
    out = ( L * r**(-3 + u) * (vsc[:, 0] * (2 * L - r * (lho[:, 1] * (4 + r * lho[:, 1]) 
            + 4 * r * lho[:, 2])) - r * ((4 + 3 * r * lho[:, 1]) * vsc[:, 1] + r * vsc[:, 2])) )
    return out


def visfo_u_D3(r, l):

    L = l*(l+1.)
    u = par.rpower_u
    out = ( -2 * L * r**(-2 + u) * (vsc[:, 0] * (2 + r * lho[:, 1]) + r * vsc[:, 1]) )
    return out


def visfo_u_D4(r, l):

    L = l*(l+1.)
    u = par.rpower_u
    out = ( -L * r**(-1 + u) * vsc[:, 0] )
    return out


def visfo_v_D0(r, l):

    L = l*(l+1.)
    v = par.rpower_v
    out = ( -L * r**(-3 + v) * (vsc[:, 0] * (L + r * lho[:, 1]) + r * vsc[:, 1]) )
    return out


def visfo_v_D1(r, l):

    L = l*(l+1.)
    v = par.rpower_v
    out = ( L * r**(-2 + v) * (vsc[:, 0] * (2 + r * lho[:, 1]) + r * vsc[:, 1]) )
    return out


def visfo_v_D2(r, l):

    L = l*(l+1.)
    v = par.rpower_v
    out = ( L * r**(-1 + v) * vsc[:, 0] )
    return out

'''