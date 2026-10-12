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
        # The background entropy gradient dS/dr, here an erf top hat between x1 and x2.
        # With ricb = 0 it enters as the odd extension sign(r)*gradS(|r|) (pdSdr, gradSX), which is smooth only if
        # dS/dr -> 0 at r = 0: keep x1 >= ~6*w1 (e.g. x1 = 0.2, w1 = 0.03) or use a profile odd in r. With x1 = w1 = 0.03
        # (dS/dr(0) = 0.2) the jump at r = 0 spoiled all ricb = 0 thermal runs (ThermaD = 0: residσ ~1; ThermaD > 0: ~1e-5).
        # With ricb > 0, dS/dr != 0 at ricb and ThermaD = 0 gives spurious modes trapped at the inner boundary.
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
        if dr_type=="rigid":   # Ω = Ω0 (1 + ΔΩ): f2 = 0 (test profile, exact Doppler-shift identity)
            out = np.zeros_like(r)
        elif dr_type=="Y20":
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

        if dr_type=="rigid":   # Ω = Ω0 (1 + ΔΩ): f0 = ΔΩ
            out = par.diff_rot_amplitude*np.ones_like(r)
        elif dr_type=="Y20":
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


class solar_profiles(user_defined_profiles):  # ------------------------------------- Smooth fit to solar Model S
    '''
    Density, gravity and pressure of the Sun from the smooth Model S fit in smoothed_sun.py, x = r*aux0 being the
    fractional solar radius (r = 1 is x = aux0, so the unit of length is aux0 R_sun; ρ is normalised to the fit's
    central value, ρ(0) = 0.95, g to its value at x = 1, g(r = 1) = 1.02, and p to p(0) = 1), with a simple entropy
    gradient in place of Model S's: the erf top hat made odd in r (utils.erf_top_hat_odd),
    dS/dr = A/2 [erf((r-x1)/w1) + erf((r+x1)/w1)] erf_box(r, x2, w2),  aux1..aux5 = x1, w1, x2, w2, A.
    With x1 = 0 it rises as A erf(r/w1) from the centre, is flat through the radiative zone and falls to zero across
    the tachocline at x2, so it can be used with ricb = 0. Model S (dS/dr = As/x from bin/model_s.at, s = S/cp, r in
    solar radii) is fitted to 10% rms by x1 = 0, w1 = 0.03, x2 = 0.68, w2 = 0.06, A = 2.6; its convection zone is
    adiabatic. The central erf(r/w1) needs N of about 11/w1 to be resolved to round-off (N ~ 380 for w1 = 0.03; at
    N = 256 the dS/dr and p dS/dr expansions are truncated at 3e-9 and 4e-7 of their largest coefficient, the other
    profiles are at round-off by k ~ 150). With g normalised at x = 1, N^2 = Beyonce g dS/dr requires
    Beyonce = g_surf/(aux0 R_sun Omega^2) = 4.90e4 for Omega = 2 pi 453.1 nHz, giving N/Omega up to about 1000 in the
    radiative zone, and ν = 1e12 cm^2/s is ViscosD = ν/(Omega (aux0 R_sun)^2) = 7.4e-5.
    The diffusivities follow the same division: ν/ν_cz and κ/κ_cz are 1 in the convection zone and visc0, therm0 in
    the radiative zone, constant in each with the erf step of dS/dr at x2 = aux3 of width w2 = aux4 (which therefore
    must be set even when thermal = 0, unless visc0 = therm0 = 1), so ViscosD and ThermaD are the convection-zone
    values. Solar values (see sun/solar-profile.md): visc0 ~ 1e-11, therm0 ~ 1e-6 to 1e-5, far below what can be
    resolved, so in practice they are numerical floors. visc_pow and therm_pow multiply both by (ρ(x2)/ρ(r))**pow:
    0 is a constant kinematic diffusivity (recommended), 1 a constant dynamic one (ρν, ρκ constant), normalised at
    x2 so that ViscosD, ThermaD are then the values at the tachocline; a density power is limited to ~1e-6 in (κp)'
    by the round-off of p near the surface.
    pdSdr and the differential rotation profiles are inherited from user_defined_profiles.
    '''

    def logrho(self, r, rpower):  # ------------------------------- log(ρ(r)), even in r
        return (r**rpower)*sms.log_density(r*par.aux0)


    def gravity(self, r, rpower):  # -------------- g(r)/g_surf, odd in r as needed when ricb = 0
        return (r**rpower)*sms.gravity(r*par.aux0)


    def pressure(self, r, rpower):  # ------------------- p(r), normalised to 1 at r = 0, even in r
        return (r**rpower)*sms.pressure(r*par.aux0)


    def dlog_p(self,r):  # ----------------------------------------------------------- d(ln p)/dr
        return par.aux0*sms.dlog_pressure(r*par.aux0)


    def dlog_rho(self,r):  # --------------------------------------------------------- d(ln ρ)/dr
        return par.aux0*sms.dlog_density(r*par.aux0)


    def gradS(self, r, x1, w1, x2, w2, A):  # ------------ dS/dr, erf top hat made odd in r
        return ut.erf_top_hat_odd(r, x1,w1,x2,w2,A)


    def cz_step(self, r):  # ----- 0 in the radiative zone, 1 in the convection zone: the upper edge of dS/dr, even in r
        return 1 - ut.erf_box(r, par.aux3, par.aux4)


    def rho_pow(self, r, p):  # --------------------------------------------- (ρ(x2)/ρ(r))**p, even in r
        if p == 0:
            return 1.0
        return np.exp( -p*( sms.log_density(r*par.aux0) - sms.log_density(par.aux3*par.aux0) ) )


    def viscosity(self, r, rpower):  # --------------- ν(r)/ν_cz: visc0 below the tachocline, 1 above, times (ρ(x2)/ρ)**visc_pow
        H = 1.0 if par.visc0 == 1 else self.cz_step(r)  # no step (and no aux3, aux4 needed) when there is no contrast
        out = ( H + (1 - H)*par.visc0 )*self.rho_pow(r, par.visc_pow)
        return (r**rpower)*out


    def thermal_diffusivity(self, r, rpower):  # ----- κ(r)/κ_cz: therm0 below the tachocline, 1 above, times (ρ(x2)/ρ)**therm_pow
        H = 1.0 if par.therm0 == 1 else self.cz_step(r)
        out = ( H + (1 - H)*par.therm0 )*self.rho_pow(r, par.therm_pow)
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
        # The background entropy gradient dS/dr, here an erf top hat between x1 and x2.
        # With ricb = 0 it enters as the odd extension sign(r)*gradS(|r|) (pdSdr, gradSX), which is smooth only if
        # dS/dr -> 0 at r = 0: keep x1 >= ~6*w1 (e.g. x1 = 0.2, w1 = 0.03) or use a profile odd in r. With x1 = w1 = 0.03
        # (dS/dr(0) = 0.2) the jump at r = 0 spoiled all ricb = 0 thermal runs (ThermaD = 0: residσ ~1; ThermaD > 0: ~1e-5).
        # With ricb > 0, dS/dr != 0 at ricb and ThermaD = 0 gives spurious modes trapped at the inner boundary.
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
elif par.model_type == 'smoothed sun':
    prf = solar_profiles()
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

# Thermal-wind background entropy S_tw = sigma2(r) P2(cos θ) + sigma4(r) P4(cos θ) implied by Ω = Ω0 [1 + f0(r) + f2(r) P2],
# to first order in ΔΩ (docs/heat-equation.md, eq. 3): (1/r) dS_tw/dθ = (r sin θ/(Beyonce g)) d(2ΔΩ)/dz.
# Even in r (f even, f' and g odd); sigma2', sigma4' are odd, hence 'tp2', 'tp4' in the parity lists of submatrices.py.
def sigma2(r, rpower):
    g = prf.gravity(r, 0)
    out = -(2*r**2/(par.Beyonce*g)) * ( abuX(r, 1)/3 + aubX(r, 1)/21 + (4/7)*aubX(r, 0)/r )
    return (r**rpower)*out

def sigma4(r, rpower):
    g = prf.gravity(r, 0)
    out = -(6*r**2/(35*par.Beyonce*g)) * ( aubX(r, 1) - 2*aubX(r, 0)/r )
    return (r**rpower)*out

def sigma2X(r, Dorder):
    tol = 1e-14
    return ut.fonzie( sigma2, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

def sigma4X(r, Dorder):
    tol = 1e-14
    return ut.fonzie( sigma4, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

def tw2(r, rpower):   # p(r) sigma2(r), p = ρT as in 'pss'
    return (r**rpower)*prf.pressure(r, 0)*sigma2X(r, 0)

def tw4(r, rpower):   # p(r) sigma4(r)
    return (r**rpower)*prf.pressure(r, 0)*sigma4X(r, 0)

def tp2(r, rpower):   # p(r) sigma2'(r)
    return (r**rpower)*prf.pressure(r, 0)*sigma2X(r, 1)

def tp4(r, rpower):   # p(r) sigma4'(r)
    return (r**rpower)*prf.pressure(r, 0)*sigma4X(r, 1)

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

def tw2X(r, Dorder):
    tol = 1e-14
    return ut.fonzie( tw2, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

def tw4X(r, Dorder):
    tol = 1e-14
    return ut.fonzie( tw4, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

def tp2X(r, Dorder):
    tol = 1e-14
    return ut.fonzie( tp2, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

def tp4X(r, Dorder):
    tol = 1e-14
    return ut.fonzie( tp4, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
proffdir = { 'lho':logrhoX, 'vsc':viscoX, 'gra':graviX, 'pss':pressX, 'pdS':pdSdrX, 'dSd':gradSX, 'kps':kappressX, 'aub':aubX, 'svp':svpX, 'pls':plsX, 'abu':abuX, 'spv':spvX, 'psl':pslX,
             'tw2':tw2X, 'tw4':tw4X, 'tp2':tp2X, 'tp4':tp4X }
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
