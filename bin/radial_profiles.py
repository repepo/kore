import numpy as np
import utils as ut
from parameters import par



# -------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------ Structure profiles
# -------------------------------------------------------------------------------------------------------------

class user_defined_profiles():  # --------------------------------------------------- As user-defined functions
    '''
    Background profiles defined directly as functions. Modify as needed.
    '''

    def density(self, r, rpower):  # ---------------- ρ(r)
        r1 = r*par.aux0;
        # out = (1-r1**2)**2               # Wu2005    β=2
        out = np.sin(np.pi*r1)/(np.pi*r1)  # Polytrope n=1
        # ------------------------------------------------
        return (r**rpower)*out


    def gravity(self, r, rpower):  # ---------------------------------------------------- g(r)
        r1 = r*par.aux0
        # out = (r1*(35 - 42*r1**2 + 15*r1**4))/8.                             # Wu2005    β=2
        out = (-(np.pi*r1*np.cos(np.pi*r1)) + np.sin(np.pi*r1))/(np.pi*r1**2)  # Polytrope n=1 
        # ------------------------------------------------------------------------------------
        return (r**rpower)*out


    def pressure(self, r, rpower):  # Normalized to 1 at r=0 --------------- p(r)
        r1 = r*par.aux0
        # out = -((-1 + r1**2)**3*(26 - 27*r1**2 + 9*r1**4))/26.  # Wu2005    β=2
        out = np.sin(np.pi*r1)**2/(np.pi**2*r1**2)                # Polytrope n=1  
        # -----------------------------------------------------------------------
        return (r**rpower)*out


    def dlog_p(self,r):  # --------------------------------------------------------------------- d(ln p)/dr
        r1 = r*par.aux0
        # out = (6*r1*(35 - 42*r1**2 + 15*r1**4))/((-1 + r1**2)*(26 - 27*r1**2 + 9*r1**4))  # Wu2005    β=2
        out = (-2/r1) + (2*np.pi/np.tan(np.pi*r1))                                          # Polytrope n=1
        # ------------------------------------------------------------------------------------------------- 
        return out
        
        
    def dlog_rho(self,r):  # ------------------------- d(ln ρ)/dr
        r1 = r*par.aux0
        # out = (4*r1)/(-1 + r1**2)               # Wu2005    β=2
        out = (-1/r1) + (np.pi/np.tan(np.pi*r1))  # Polytrope n=1
        # -------------------------------------------------------
        return out
        
        
    def Gamma1_isentropic(self,r):  # This is the isentropic Γ₁  
        out = self.dlog_p(r)/self.dlog_rho(r)
        return out
    def faux1(self,r,a,b,c):  # inverted parabola (hard edges)
        out = np.zeros_like(r)
        x = (r>a)&(r<b)
        out[x] = -c*(4/b**2)*(r[x]-a)*(r[x]-b)
        out[r<0] = np.flipud(out[r>0])  # make it even
        return out
    def faux2(self,r,a,b,c):  # cosine function (soft edges)
        out = np.zeros_like(r)
        x = (r>a)&(r<b)
        out[x] = (c/2)*(1-np.cos(2*np.pi*(r[x]-a)/(b-a)))
        out[r<0] = np.flipud(out[r>0])  # make it even
        return out
    def faux(self,r,a,b,c):  # mixed parabola + cosine. par.aux4=1 is cosine, par.aux4=0 is parabola.
        out = self.faux1(r,a,b,c) * (1-par.aux4) + self.faux2(r,a,b,c) * par.aux4
        return out
    def Gamma1(self,r):  # The resulting first adiabatic coefficient Γ₁
        out = self.Gamma1_isentropic(r) + self.faux(r,par.aux1,par.aux2,par.aux3)
        return out
    def gradS(self,r):  # The background entropy gradient
        out = np.zeros_like(r)
        x = abs(r)<1
        out[x] = ( self.dlog_p(r[x])/self.Gamma1(r[x]) ) - self.dlog_rho(r[x])
        return out


    def pdSdr(self, r, rpower):
        # ------------------------ p(r) dS/dr
        out = np.zeros_like(r)
        x = abs(r)<1
        out[x] = self.pressure(r[x],0)*self.gradS(r[x])
        # -----------------------------------
        return (r**rpower)*out


    def viscosity(self, r, rpower):
        # ------------------------------ v(r)
        #out = np.ones_like(r)
        out = 1/self.density(r,0)
        # -----------------------------------
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


    def pdSdr(self, r, rpower):
        # ------------------------ p(r) dS/dr
        out = ut.load_model(r,'pdSdr')
        # -----------------------------------
        return (r**rpower)*out


    def viscosity(self, r, rpower):  # def here only
        # ------------------------------ v(r)
        out = np.ones_like(r)
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

def kappress(r, rpower):   # κ p

    out = prf.thermal_diffusivity(r, 0) * prf.pressure(r, 0)
    return (r**rpower)*out


def dynamic_viscosity(r, rpower):  # μ = ρν

    #out = prf.density(r,0) * prf.viscosity(r,0)
    out = np.ones_like(r) 
    return (r**rpower)*out


def densityX(r, Dorder):  # ρ⁽ⁿ⁾, radial derivatives of ρ(r)

    tol = 1e-12
    out = ut.fonzie( prf.density, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)
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



# -------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------- Profile functions for burrito
# -------------------------------------------------------------------------------------------------------------

def rhoX(r, *args):   # rᵃ ρᵇ  powers of ρ

    (rpower, rhopower) = args
    out = np.zeros_like(r)
    out = prf.density(r, 0)**rhopower

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
    out = ut.fonzie( dynamic_viscosity, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

    return out


def kappressX(r, Dorder):   # κ(r)ρ(r)T(r)

    tol = 1e-12
    out = ut.fonzie( kappress, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

    return out


def pdSdrX(r, Dorder):   # ρ(r)T(r)dS/dr

    tol = 1e-12
    out = ut.fonzie( prf.pdSdr, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

    return out


def graviX(r, Dorder):   # g(r)

    tol = 1e-12
    out = ut.fonzie( prf.gravity, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

    return out


def pressX(r, Dorder):   # ρ(r)T(r)

    tol = 1e-12
    out = ut.fonzie( prf.pressure, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

    return out



# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
proffdir = { 'lho':rhoXlhoX, 'moe':muX, 'rho':rhoX, 'gra':graviX, 'pss':pressX, 'pdS':pdSdrX, 'kps':kappressX }
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------


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


# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
