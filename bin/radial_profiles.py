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
        # odd extension to r<0 (only used when ricb=0, where submatrices treats 'pdS' as odd)
        out[x] = self.pressure(r[x],0)*np.sign(r[x])*self.gradS(np.abs(r[x]), par.aux1, par.aux2, par.aux3, par.aux4, par.aux5)
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
    
    def aub(self, r, rpower): 
        """
        Differential rotation radial profile for Y20
        """
        dr_type = par.diff_rot_type
        if dr_type=="Y20":
            out = par.diff_rot_amplitude*np.ones_like(r)
        
        elif dr_type=="Y20-wall-bounded":
            out = par.diff_rot_amplitude*(1-r)*(r-par.ricb)

        elif dr_type=="shellular": #[Baruteau, Rieutord 2012] : Ω(r) = Ω_ref * (r / R)**σ 
            # Ω_ref = 1 (= Ω_0) -> Set par.timescale = "rotation"
            # σ parameter
            out = np.zeros_like(r)

        elif dr_type=="shellular_boussinesq": #[Mirouh et al. 2016] : Ω(r) = Ω_ref * (1 + (1/2) * (N**2) * (1 - (r / R)**2))
            # Ω_ref = 1 (= Ω_0) -> Set par.timescale = "rotation"
            out = np.zeros_like(r)

        elif dr_type=="cylindrical": #[Baruteau, Rieutord 2012] : Ω(r, θ) = Ω_ref * [1 + (ε * (r / R)**2 * sin(θ)**2)]
            # Ω_ref = 1 (= Ω_0) -> Set par.timescale = "rotation"
            epsilon = par.diff_rot_amplitude
            out = -((2/3)*epsilon)*r**2

        elif dr_type=="conical": #[Guenel et al., 2016] : Ω(r, θ) = Ω_ref * [1 + (ε * sin(θ)**2)]
            # Ω_ref = 1 (= Ω_0) -> Set par.timescale = "rotation"
            epsilon = par.diff_rot_amplitude
            out = -((2/3)*epsilon)

        elif dr_type=="solar":
            rtc = 0.71
            h = 0.1
            W0 = 435 #[nHz]
            b = -86.04/W0 # Fitted on data
            out = b * (0.5 * (1 + np.tanh((2 * (r - rtc))/h)))

        return (r**rpower)*out
    
    def abu(self, r, rpower): 
        """
        Differential rotation radial profile for Y00
        """
        dr_type = par.diff_rot_type

        if dr_type=="Y20":
            out = np.zeros_like(r)
        
        elif dr_type=="Y20-wall-bounded":
            out = np.zeros_like(r)

        elif dr_type=="shellular": #[Baruteau, Rieutord 2012] : Ω(r) = Ω_ref * (r / R)**σ 
            # Ω_ref = 1 (= Ω_0) -> Set par.timescale = "rotation"
            # σ parameter
            sigma = par.diff_rot_amplitude
            out = (r**sigma) - 1

        elif dr_type=="shellular_boussinesq": #[Mirouh et al. 2016] : Ω(r) = Ω_ref * (1 + (1/2) * (N**2) * (1 - (r / R)**2))
            # Ω_ref = 1 (= Ω_0) -> Set par.timescale = "rotation"
            N02 = par.diff_rot_amplitude # Brunt-Väisälä frequency squared
            out = (N02/2)*(1 - r**2)

        elif dr_type=="cylindrical": #[Baruteau, Rieutord 2012] : Ω(r, θ) = Ω_ref * [1 + (ε * (r / R)**2 * sin(θ)**2)]
            # Ω_ref = 1 (= Ω_0) -> Set par.timescale = "rotation"
            epsilon = par.diff_rot_amplitude
            out = ((2/3)*epsilon)*r**2
        
        elif dr_type=="conical": #[Guenel et al., 2016] : Ω(r, θ) = Ω_ref * [1 + (ε * sin(θ)**2)]
            # Ω_ref = 1 (= Ω_0) -> Set par.timescale = "rotation"
            epsilon = par.diff_rot_amplitude
            out = ((2/3)*epsilon)
        
        elif dr_type=="solar":
            rtc = 0.71
            h = 0.1
            W0 = 435 #[nHz]
            a = -7.695/W0 # Fitted on data
            out = a * (0.5 * (1 + np.tanh((2 * (r - rtc))/h)))

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

    def dlog_p(self,r):
        tol = 1e-12; Dorder = 1
        dp = ut.fonzie(prf.pressure, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)
        out = dp[:,1]/dp[:,0]
        return out
    def dlog_rho(self,r):
        tol = 1e-12; Dorder = 1
        drho = ut.fonzie(prf.density, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)
        out = drho[:,1]/drho[:,0]
        return out   
    def Gamma1_isentropic(self,r):  # This is the isentropic Γ₁  
        out = self.dlog_p(r)/self.dlog_rho(r)
        #out = (4/3)*np.ones_like(r)
        return out
    def faux(self,r,a,b,c):  # cosine function (soft edges)
        out = np.zeros_like(r)
        x = (r>a)&(r<b)
        out[x] = (c/2)*(1-np.cos(2*np.pi*(r[x]-a)/(b-a)))
        out[r<0] = np.flipud(out[r>0])  # make it even
        return out
    def Gamma1(self,r,a,b,c):  # The resulting first adiabatic coefficient Γ₁
        out = self.Gamma1_isentropic(r) + self.faux(r,a,b,c)
        return out
    def gradS(self,r,a,b,c):  # The background entropy gradient
        out = np.zeros_like(r)
        x = abs(r)<1
        out[x] = ( self.dlog_p(r[x])/self.Gamma1(r[x],a,b,c) ) - self.dlog_rho(r[x])
        return out


    def gradS(self, r, x1, w1, x2, w2, A):
        return ut.erf_top_hat(r, x1,w1,x2,w2,A)


    def pdSdr(self, r, rpower):
        # ------------------------ p(r) dS/dr
        #out = ut.load_model(r,'pdSdr')
        out = np.zeros_like(r)
        x = abs(r)<1
        # odd extension to r<0 (only used when ricb=0, where submatrices treats 'pdS' as odd)
        out[x] = self.pressure(r[x],0)*np.sign(r[x])*self.gradS(np.abs(r[x]), par.aux1, par.aux2, par.aux3, par.aux4, par.aux5)
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


def aubX(r, Dorder):
    tol = 1e-14
    out = ut.fonzie( prf.aub, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]
    return out
    
def svp(r, rpower):
    return (r**rpower)*r*aubX(r, 1)

def pls(r, rpower):
    return (r**rpower)*(r**2)*aubX(r, 2)

def abuX(r, Dorder):
    tol = 1e-14
    out = ut.fonzie( prf.abu, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]
    return out
    
def spv(r, rpower):
    return (r**rpower)*r*abuX(r, 1)

def psl(r, rpower):
    return (r**rpower)*(r**2)*abuX(r, 2)

# -------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------- Profile functions for burrito
# -------------------------------------------------------------------------------------------------------------


def logrhoX(r, Dorder):

    tol = 1e-14
    out = ut.angine( prf.logrho, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out


def viscoX(r, Dorder):   # Kinematic viscosity

    tol = 1e-14
    out = ut.angine( prf.viscosity, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out


def kappressX(r, Dorder):   # κ(r)p(r), p stands for ρ(r)T(r)

    tol = 1e-14
    kps = lambda rr, rpower: prf.thermal_diffusivity(rr, rpower) * prf.pressure(rr, 0)  # ut.angine needs a function f(r, rpower)
    out = ut.angine( kps, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out


def pdSdrX(r, Dorder):   # ρ(r)T(r)dS/dr

    tol = 1e-14
    out = ut.angine( prf.pdSdr, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out


def gradSX(r, Dorder):   # dS/dr, odd extension to r<0 (used when ThermaD=0)

    tol = 1e-14
    dS  = lambda rr, rpower: (rr**rpower)*np.sign(rr)*prf.gradS(np.abs(rr), par.aux1, par.aux2, par.aux3, par.aux4, par.aux5)
    out = ut.angine( dS, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out


def graviX(r, Dorder):   # g(r)

    tol = 1e-14
    out = ut.angine( prf.gravity, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out


def pressX(r, Dorder):   # ρ(r)T(r)

    tol = 1e-14
    out = ut.angine( prf.pressure, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out

def svpX(r, Dorder):
    tol = 1e-14
    out = ut.fonzie( svp, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

    return out

def plsX(r, Dorder):
    tol = 1e-14
    out = ut.fonzie( pls, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

    return out

def spvX(r, Dorder):
    tol = 1e-14
    out = ut.fonzie( spv, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

    return out

def pslX(r, Dorder):
    tol = 1e-14
    out = ut.fonzie( psl, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

    return out
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
proffdir = { 'lho':logrhoX, 'vsc':viscoX, 'gra':graviX, 'pss':pressX, 'pdS':pdSdrX, 'dSd':gradSX, 'kps':kappressX, 'aub':aubX, 'svp':svpX, 'pls':plsX, 'abu':abuX, 'spv':spvX, 'psl':pslX }
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


    return out


# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
