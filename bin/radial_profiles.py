import numpy as np
import sys
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

        elif dr_type=="solar":  # fit to helioseismic Ω (times diff_rot_amplitude), tachocline at rtc = 0.71 R☉ of width h = 0.1 R☉
            x = r*par.aux0 if par.model_type == 'smoothed sun' else r  # fractional solar radius ('user def': r is taken as r/R☉)
            rtc = 0.71
            h = 0.1
            W0 = 435 #[nHz]
            b = -86.04/W0 # Fitted on data
            out = par.diff_rot_amplitude * b * (0.5 * (1 + np.tanh((2 * (x - rtc))/h)))

        elif dr_type in ("solar_P4", "tabulated"):
            out = f_new(2, r)

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
        
        elif dr_type=="solar":  # as in aub
            x = r*par.aux0 if par.model_type == 'smoothed sun' else r
            rtc = 0.71
            h = 0.1
            W0 = 435 #[nHz]
            a = -7.695/W0 # Fitted on data
            out = par.diff_rot_amplitude * a * (0.5 * (1 + np.tanh((2 * (x - rtc))/h)))

        elif dr_type in ("solar_P4", "tabulated"):
            out = f_new(0, r)

        return (r**rpower)*out


class solar_profiles(user_defined_profiles):  # ------------------------- Parts shared by the 'smoothed sun' structures
    '''
    The parts shared by the 'smoothed sun' structures (dS/dr, the diffusivities, the differential rotation); ρ, p and g
    come from the subclass chosen by structure (see structure_name). The rest of this docstring describes the original
    'smoothed sun' structure, the 8-term fit (eight_term_profiles).
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
    With solar_rz = 1, dS/dr and the diffusivity step are instead those of the radiative interior of Mukhopadhyay et al.
    2025, their eqs. (8)-(10) (params_default.py, docs/heat-equation.md §9).
    pdSdr and the differential rotation profiles are inherited from user_defined_profiles.
    '''

    def gradS(self, r, x1, w1, x2, w2, A):  # ------------ dS/dr, erf top hat made odd in r
        if par.solar_rz:   # Mukhopadhyay et al. 2025 eq. (10), times erf(x/w0) so that it is odd and smooth at r = 0
            x = r*par.aux0
            x1, x2, d = par.solar_rz_steps
            out = par.solar_rz_dSdr*0.25*(1 - ss.erf((x - x1)/d))*(1 - ss.erf((x - x2)/d))*ss.erf(x/par.solar_rz_core)
            if par.solar_rz_delta_cz != 0 or par.solar_rz_delta_sf != 0:   # convection-zone superadiabaticity δ (Bekki 2024, eq. 2):
                # (1/c_p) ds0/dr = δ d(ln p0)/dr above the overshoot step, δ < 0 subadiabatic; odd in r as d(ln p0)/dr is
                xa = np.abs(x); xos, dos = par.solar_rz_os
                delta = par.solar_rz_delta_cz + (par.solar_rz_delta_sf - par.solar_rz_delta_cz)*np.exp(-((xa - 0.985)/0.015)**2)
                out = out + delta*self.dlog_p(r)*0.5*(1 + ss.erf((xa - xos)/dos))
            return out
        return ut.erf_top_hat_odd(r, x1,w1,x2,w2,A)


    def cz_step(self, r):  # ----- 0 in the radiative zone, 1 in the convection zone: the upper edge of dS/dr, even in r
        if par.solar_rz:   # their eqs. (8), (9)
            xos, dos = par.solar_rz_os
            return 0.5*(1 + ss.erf((np.abs(r)*par.aux0 - xos)/dos))
        return 1 - ut.erf_box(r, par.aux3, par.aux4)


    def rho_pow(self, r, p):  # --------------------------------------------- (ρ(x2)/ρ(r))**p, even in r
        if p == 0:
            return 1.0
        return np.exp( -p*( self.logrho(r, 0) - self.logrho(par.aux3, 0) ) )


    def viscosity(self, r, rpower):  # --------------- ν(r)/ν_cz: visc0 below the tachocline, 1 above, times (ρ(x2)/ρ)**visc_pow
        H = 1.0 if par.visc0 == 1 else self.cz_step(r)  # no step (and no aux3, aux4 needed) when there is no contrast
        out = ( H + (1 - H)*par.visc0 )*self.rho_pow(r, par.visc_pow)
        return (r**rpower)*out


    def thermal_diffusivity(self, r, rpower):  # ----- κ(r)/κ_cz: therm0 below the tachocline, 1 above, times (ρ(x2)/ρ)**therm_pow
        H = 1.0 if par.therm0 == 1 else self.cz_step(r)
        out = ( H + (1 - H)*par.therm0 )*self.rho_pow(r, par.therm_pow)
        return (r**rpower)*out


class eight_term_profiles(solar_profiles):  # -------------------------------- structure = '8-term fit' (legacy)
    '''
    ρ, p and g of the smooth Model S fit in smoothed_sun.py: 8 even terms in ln ρ, g and p from hydrostatic balance (x = r aux0;
    ρ, p normalised at the centre, g at x = 1).
    Everything else as solar_profiles.
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


class modelS_profiles(solar_profiles):  # ----------------------- structure = 'old model S prototype' (legacy)
    '''
    ρ and p from Chebyshev fits of ln ρ and ln p of bin/model_s.at, g from smoothed_sun.py (normalised at x = 1), so the
    three are not in hydrostatic balance (structure = 'old model S prototype', the old solar_density = 'model S').
    solar_profiles with ln ρ and ln p from Chebyshev fits of bin/model_s.at (columns As, c_1, ρ/ρ_c, p/p_c, x) over the
    window solar_density_window in x = r aux0, degree solar_density_deg. Everything else
    (dS/dr, ν, κ, the differential rotation) as solar_profiles. ρ and p are normalised to their central values, as the
    8-term ones (the overall constants drop out of the equations). The shell fits are not even in r, so ricb = 0 uses
    solar_density_window = 'sun': Model S mirrored to x < 0 and fitted on [-W, W], W = aux0 + 0.005, with degree
    solar_density_sun_deg and the odd coefficients dropped, so that ln ρ and ln p are even and smooth at r = 0.
    '''
    def __init__(self):
        import os
        w = par.solar_density_window
        if w is None:
            w = [par.ricb*par.aux0 - 0.01, par.aux0 + 0.005] if par.ricb > 0 else 'sun'
        T = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_s.at'), skiprows=1)
        x, lrho, lp = T[:, 4], np.log(T[:, 2]), np.log(T[:, 3])
        C = np.polynomial.chebyshev.Chebyshev
        if isinstance(w, str):
            if w != 'sun':
                sys.exit("solar_density_window must be [x_lo, x_hi], None or 'sun'")
            W = par.aux0 + 0.005
            deg = par.solar_density_sun_deg
            k = x <= W
            xx = np.concatenate([-x[k], x[k]])
            def evenfit(y):
                c = C.fit(xx, np.concatenate([y[k], y[k]]), deg, domain=[-W, W]).coef.copy()
                c[1::2] = 0
                return C(c, domain=[-W, W])
            self.lrho, self.lp = evenfit(lrho), evenfit(lp)
            w = [-W, W]
        else:
            if par.ricb == 0:
                sys.exit("structure 'old model S prototype' with ricb = 0 needs solar_density_window = None or 'sun' (an even fit)")
            deg = par.solar_density_deg
            k = (x >= w[0]) & (x <= w[1])
            if k.sum() < 2*deg:
                sys.exit('solar_density_window %s has too few Model S points' % str(w))
            self.lrho = C.fit(x[k], lrho[k], deg, domain=list(w))
            self.lp   = C.fit(x[k], lp[k], deg, domain=list(w))
        self.window = w
        print('model S density: window', w, 'degree', deg, 'max fit error ln rho %.1e, ln p %.1e' %
              (np.abs(self.lrho(x[k]) - lrho[k]).max(), np.abs(self.lp(x[k]) - lp[k]).max()))

    def logrho(self, r, rpower):
        return (r**rpower)*self.lrho(r*par.aux0)

    def pressure(self, r, rpower):
        return (r**rpower)*np.exp(self.lp(r*par.aux0))

    def dlog_p(self, r):
        return par.aux0*self.lp.deriv()(r*par.aux0)

    def dlog_rho(self, r):
        return par.aux0*self.lrho.deriv()(r*par.aux0)

    def gravity(self, r, rpower):   # as eight_term_profiles
        return (r**rpower)*sms.gravity(r*par.aux0)


class modelS_hydrostatic_profiles(solar_profiles):  # ------------------------- structure = 'model S hydrostatic'
    '''
    ρ from an even fit of ln ρ of bin/model_s.at, g from its mass integral and p from hydrostatic balance dp/dx = -ρ g.
    Everything else as solar_profiles.
    ln ρ is one even polynomial of degree structure_deg over the whole Sun (whatever ricb), fitted on 0 ≤ x ≤ aux0
    (iteratively reweighted least squares, close to minimax in ln ρ). Then g = m(x)/(x² m(1)), m the mass integral of
    that ρ (Model S above x = aux0 for m(1)), normalised at the surface as smoothed_sun.py, and p from dp/dx = -K ρ g
    with K the Model S value of -p'/(ρ g), starting from Model S at x = aux0. ρ and p are normalised at the centre.
    With structure_deg = 30 the largest errors against Model S on 0 ≤ x ≤ 0.985 are 1.0% in ρ, 0.5% in p and 0.75% in g;
    with 50, 0.14%, 0.06% and 0.10%.
    '''
    def __init__(self):
        import os
        C = np.polynomial.chebyshev.Chebyshev
        T = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_s.at'), skiprows=1)
        x, rho, p = T[:, 4], T[:, 2], T[:, 3]
        g = x/T[:, 1]
        W, k = par.aux0 + 0.001, x <= par.aux0
        V = np.polynomial.chebyshev.chebvander(x[k]/W, par.structure_deg)[:, 0::2]   # even polynomials T_2j(x/W)
        wt = np.ones(k.sum())
        for i in range(200):
            c = np.linalg.lstsq(V*wt[:, None], np.log(rho[k])*wt, rcond=None)[0]
            e = np.abs(V @ c - np.log(rho[k]))
            wt = wt*(1 + e/e.max())
            wt = wt/wt.max()
        c2 = np.zeros(par.structure_deg + 1)
        c2[0::2] = c
        lrho = C(c2, domain=[-W, W])                                   # ln(ρ/ρ_c of Model S)
        rc = C.interpolate(lambda y: np.exp(lrho(np.abs(y))), 500, domain=[-W, W])
        xs, ws = np.polynomial.legendre.leggauss(200)
        xs, ws = (xs + 1)/2, ws/2
        mx = lambda y: y**3*((rc(np.outer(y, xs))*xs**2) @ ws)       # m(x) = x³ ∫₀¹ s² ρ(s x) ds (units of ρ_c R³)
        ks = x >= par.aux0                                         # Model S above aux0, only for m(1)
        m1 = mx(np.array([par.aux0]))[0] + np.sum(np.diff(x[ks])*(x[ks][1:]**2*rho[ks][1:] + x[ks][:-1]**2*rho[ks][:-1])/2)
        self.g = C.interpolate(lambda y: y*((rc(np.outer(y, xs))*xs**2) @ ws)/m1, 500, domain=[-W, W])
        s = (x > 0.05) & (x < 0.95)
        K = np.median(-np.gradient(p, x)[s]/(rho[s]*g[s]))
        def P(y):   # p_ModelS(aux0) + K ∫_|y|^aux0 ρ g dx, by quadrature from the surface (keeps the relative accuracy of small p)
            ya = np.abs(np.atleast_1d(y))[:, None]
            t = ya + (par.aux0 - ya)*xs
            return np.interp(par.aux0, x, p) + K*(par.aux0 - ya[:, 0])*((np.exp(lrho(t))*self.g(t)) @ ws)
        self.lp = C.interpolate(lambda y: np.log(P(y)/P(0.0)), 500, domain=[-W, W])   # p(0) = 1
        for f, odd in (('g', 1), ('lp', 0)):    # cut each series (g odd, ln p even) where its coefficients first drop below
            c = getattr(self, f).coef               # 1e-15 of the largest: beyond is round-off
            kc = np.nonzero(np.abs(c[odd::2]) < 1e-15*np.abs(c).max())[0]
            setattr(self, f, C(c[:2*kc[0] + odd] if kc.size else c, domain=[-W, W]))
        self.lrho = lrho - lrho(0)                                     # ρ(0) = 1
        kk = (x > 0.02) & k
        print("structure 'model S hydrostatic': degree", par.structure_deg, 'max error rho %.1e, p %.1e, g %.1e' %
              tuple(np.abs(f(x[kk])/ref[kk] - 1).max() for f, ref in ((lambda y: np.exp(lrho(y)), rho), (P, p), (self.g, g))))

    def logrho(self, r, rpower):
        return (r**rpower)*self.lrho(r*par.aux0)

    def pressure(self, r, rpower):
        return (r**rpower)*np.exp(self.lp(r*par.aux0))

    def gravity(self, r, rpower):
        return (r**rpower)*self.g(r*par.aux0)

    def dlog_p(self, r):
        return par.aux0*self.lp.deriv()(r*par.aux0)

    def dlog_rho(self, r):
        return par.aux0*self.lrho.deriv()(r*par.aux0)


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
# The 'smoothed sun' structures (ρ, p, g): par.structure and its class. The order is that of the index written to params.dat
# (spin_doctor.solar_densities); append new ones at the end. solar_density_names: the old par.solar_density values.
structures = { '8-term fit': eight_term_profiles, 'old model S prototype': modelS_profiles,
               'model S hydrostatic': modelS_hydrostatic_profiles }
solar_density_names = { 'fit': '8-term fit', 'model S': 'old model S prototype' }

def structure_name():   # ------------------------------ the 'smoothed sun' structure in effect (par.structure or the old solar_density)
    '''
    A key of structures. par.structure = None takes it from the old name par.solar_density (solar_density_names); a
    solar_density other than its default 'fit' must then agree with par.structure.
    '''
    if par.solar_density not in solar_density_names:
        sys.exit('solar_density must be one of %s' % list(solar_density_names))
    if par.structure is None:
        return solar_density_names[par.solar_density]
    if par.structure not in structures:
        sys.exit('structure must be one of %s' % list(structures))
    if par.solar_density != 'fit' and solar_density_names[par.solar_density] != par.structure:
        sys.exit("solar_density = %r conflicts with structure = %r; set only structure" % (par.solar_density, par.structure))
    return par.structure


if par.model_type == 'user def':
    prf = user_defined_profiles()
elif par.model_type == 'smoothed sun':
    prf = structures[structure_name()]()
elif par.model_type in ['mesa', 'gsm', 'poly', 'astropy table']:
    prf = profiles_from_file()
elif par.model_type == 'Boussinesq':
    prf = Boussinesq_profiles()


# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------- Derived profiles
# -------------------------------------------------------------------------------------------------------------


# With ricb = 0 the differential rotation profiles are evaluated at |r|: they are even functions of r by construction
# (like logrho), and submatrices.py keeps only their even part (otherwise e.g. half of the 'solar' tachocline step is lost,
# and a non-integer 'shellular' σ gives NaN at r < 0).
# 'solar_P4' and 'tabulated' (docs/heat-equation.md §8): f_n(r), ΔΩ = Ω/Ω_ref - 1 = Σ_n f_n P_n(cos θ), times diff_rot_amplitude
def _dr_x(r):   # fractional solar radius
    return r*par.aux0 if par.model_type == 'smoothed sun' else r

_tab = {}
def _tabulated_fits():
    '''
    'tabulated': f_n (n = 0, 2, ..., diff_rot_nmax) of ΔΩ = ome/diff_rot_frame - 1 at each table radius by Gauss-Legendre
    quadrature (64 nodes) of a cubic spline in cos θ, then a least-squares Chebyshev fit in x of degree diff_rot_rdeg
    (at most half the number of table radii used) over the table radii within 0.01 of the shell, or with
    diff_rot_window = 'sun' (the default when ricb = 0) one even fit over the whole Sun. Returns {n: Chebyshev}.
    '''
    if _tab:
        return _tab
    from scipy.interpolate import CubicSpline
    D = np.load(par.diff_rot_file)
    ome, rt, th = D['ome'], D['r'], D['theta']          # ome[θ, r] in nHz, θ in rad, r in R☉
    o = np.argsort(np.cos(th))
    xc = np.cos(th)[o]
    ns = list(range(0, par.diff_rot_nmax + 1, 2))
    xg, wg = np.polynomial.legendre.leggauss(64)
    P = np.array([np.polynomial.legendre.Legendre.basis(n)(xg) for n in ns])
    dO = CubicSpline(xc, ome[o, :]/par.diff_rot_frame - 1, axis=0)(xg)     # (64, nr)
    fn = (((2*np.array(ns) + 1)/2)[:, None]*P*wg) @ dO                       # (len(ns), nr)
    xa, xb = _dr_x(np.array([par.ricb, 1.0]))
    win = par.diff_rot_window
    if win is None and par.ricb == 0:
        win = 'sun'
    if win is not None:   # one fit over the whole Sun, even in r, the same for any ricb
        if win != 'sun':
            sys.exit("diff_rot_window must be None or 'sun'")
        L = xb + 0.01
        k = rt <= L
        r1, f1 = rt[k], fn[:, k]
        if par.diff_rot_hold is not None:   # hold f_n at its value at x_h below x_h
            j = np.argmin(np.abs(r1 - par.diff_rot_hold))
            S = 0.5*(1 + ss.erf((r1 - par.diff_rot_hold + 0.01)/0.01))
            f1 = S*f1 + (1 - S)*f1[:, j:j+1]
        dx = r1[1] - r1[0]
        rp = np.arange(r1[0] - dx, 0, -dx)[::-1]                 # below the first table radius: its value
        r1 = np.concatenate([rp, r1]); f1 = np.concatenate([np.repeat(f1[:, :1], rp.size, axis=1), f1], axis=1)
        xx = np.concatenate([-r1[::-1], r1])
        C = np.polynomial.chebyshev.Chebyshev
        for i, n in enumerate(ns):
            c = C.fit(xx, np.concatenate([f1[i, ::-1], f1[i]]), par.diff_rot_sun_deg, domain=[-L, L]).coef.copy()
            c[1::2] = 0
            _tab[n] = C(c, domain=[-L, L])
        res = [np.abs(_tab[n](r1) - f1[i]).max()*par.diff_rot_frame for i, n in enumerate(ns)]
        print("diff_rot_type 'tabulated': %s, whole-Sun even fit of degree %d on [0, %.3f]%s; max |fit - f_n| (nHz):" %
              (par.diff_rot_file, par.diff_rot_sun_deg, L, '' if par.diff_rot_hold is None else
               ', f_n held below %.3f' % par.diff_rot_hold), ' '.join('%.2e' % v for v in res))
        return _tab
    k = (rt >= xa - 0.01) & (rt <= xb + 0.01)
    deg = min(par.diff_rot_rdeg, int(k.sum())//2)
    C = np.polynomial.chebyshev.Chebyshev
    for i, n in enumerate(ns):
        _tab[n] = C.fit(rt[k], fn[i, k], deg, domain=[xa, xb])
    # report: fit residual of each f_n and the truncation error of Ω at the table nodes inside the shell (nHz)
    res = [np.abs(_tab[n](rt[k]) - fn[i, k]).max()*par.diff_rot_frame for i, n in enumerate(ns)]
    rec = sum(np.outer(np.polynomial.legendre.Legendre.basis(n)(np.cos(th)), _tab[n](rt[k])) for n in ns)
    trunc = np.abs(par.diff_rot_frame*(1 + rec) - ome[:, k]).max()
    print("diff_rot_type 'tabulated': %s, %d radii in [%.3f, %.3f], degree %d; max |fit - f_n| (nHz):" %
          (par.diff_rot_file, k.sum(), xa, xb, deg), ' '.join('%.2e' % v for v in res),
          '; max |Ω(P0..P%d) - Ω| = %.2f nHz' % (par.diff_rot_nmax, trunc))
    return _tab

def f_new(n, r):
    '''f_n(r) of 'solar_P4' and 'tabulated', times diff_rot_amplitude (the a_n are applied in fX, abuX, aubX)'''
    x = _dr_x(r)
    if par.diff_rot_type == 'solar_P4':
        c0, c2, c4, orz, rtc, dtc = par.diff_rot_solar
        st = 0.5*(1 + ss.erf((x - rtc)/dtc))
        On = {0: c0 + c2/3 + c4/5, 2: 2*c2/3 + 4*c4/7, 4: 8*c4/35}   # cos²θ = (1 + 2 P2)/3, cos⁴θ = 1/5 + 4 P2/7 + 8 P4/35
        if n == 0:
            out = (orz + (On[0] - orz)*st)/par.diff_rot_frame - 1
        else:
            out = On.get(n, 0)*st/par.diff_rot_frame
    else:
        fits = _tabulated_fits()
        out = fits[n](x) if n in fits else 0*x
    return par.diff_rot_amplitude*out

def _amp(n):   # a_n, the factor of the P_n term of ΔΩ (1 if diff_rot_term_amp is shorter)
    a = par.diff_rot_term_amp
    return a[n//2] if n//2 < len(a) else 1.0

def aubX(r, Dorder):
    tol = 1e-14
    f = prf.aub if par.ricb > 0 else lambda rr, rpower: (rr**rpower)*prf.aub(np.abs(rr), 0)
    out = ut.fonzie( f, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]
    return out*_amp(2)
    
def svp(r, rpower):
    return (r**rpower)*r*aubX(r, 1)

def pls(r, rpower):
    return (r**rpower)*(r**2)*aubX(r, 2)

def abuX(r, Dorder):
    tol = 1e-14
    f = prf.abu if par.ricb > 0 else lambda rr, rpower: (rr**rpower)*prf.abu(np.abs(rr), 0)
    out = ut.fonzie( f, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]
    return out*_amp(0)

def fX(n, r, Dorder):
    '''a_n f_n(r) and its radial derivatives, any even n (abuX, aubX for n = 0, 2; f_new for n >= 4, at |r| if ricb = 0)'''
    if n == 0:
        return abuX(r, Dorder)
    if n == 2:
        return aubX(r, Dorder)
    tol = 1e-14
    if par.ricb > 0:
        f = lambda rr, rpower: (rr**rpower)*f_new(n, rr)
    else:
        f = lambda rr, rpower: (rr**rpower)*f_new(n, np.abs(rr))
    return ut.fonzie( f, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]*_amp(n)
    
def spv(r, rpower):
    return (r**rpower)*r*abuX(r, 1)

def psl(r, rpower):
    return (r**rpower)*(r**2)*abuX(r, 2)

# Thermal-wind background entropy S_tw = σ2(r) P2(cos θ) + σ4(r) P4(cos θ) [+ σ6(r) P6(cos θ)] implied by Ω = Ω0 [1 + f0(r) + f2(r) P2],
# (1/r) dS_tw/dθ = (r sin θ/(Beyonce g)) Gaspard² d[(1+ΔΩ)² - 1]/dz (docs/heat-equation.md, eq. 2): to first order in ΔΩ
# ((1+ΔΩ)² - 1 -> 2ΔΩ, eq. 3) if thermal_wind = 1, exact if thermal_wind = 2 (tests/heat_dr/tw_sympy.py).
# Even in r (f even, f' and g odd); σn' are odd, hence 'tp2', 'tp4', 'tp6' in the parity lists of submatrices.py.
# thermal_wind = 3: S_tw prescribed, ∂S_tw/∂θ = dSdtheta sin 2θ = -(2/3) dSdtheta dP2/dθ, so σ2 = -(2/3) dSdtheta and σ4 = 0.
_sig_cache = {}
def sigma_all(r):
    '''
    σ_n(r), n = 0 ... tw_nmax (σ_0 = 0), of the thermal-wind entropy for ΔΩ = Σ_n a_n f_n P_n with degrees above 2 (ut.dr_high),
    from ∂S/∂x = -(r² G²/(Beyonce g)) [x ∂F/∂r + (1 - x²)(∂F/∂x)/r], x = cos θ, F = 2ΔΩ (thermal_wind = 1) or (1+ΔΩ)² - 1 (2):
    Legendre coefficients of ∂S/∂x by Gauss quadrature (exact, a polynomial of degree < tw_nmax), integrated in x.
    Times thermal_wind_amplitude. Rows n, columns r.
    '''
    key = (r.tobytes(), r.shape)
    if key in _sig_cache:
        return _sig_cache[key]
    Lg = np.polynomial.legendre
    ns = ut.tw_nmax
    xg, wg = Lg.leggauss(ns + 4)
    nd = [0, 2] + ut.dr_high
    dO  = sum(np.outer(fX(n, r, 0), Lg.Legendre.basis(n)(xg)) for n in nd)
    dOr = sum(np.outer(fX(n, r, 1), Lg.Legendre.basis(n)(xg)) for n in nd)
    dOx = sum(np.outer(fX(n, r, 0), Lg.Legendre.basis(n).deriv()(xg)) for n in nd)
    a = (1 + dO) if par.thermal_wind == 2 else 1
    rr = r[:, None]
    dSdx = -(rr**2*par.Gaspard**2/(par.Beyonce*prf.gravity(r, 0)[:, None]))*(xg*2*a*dOr + (1 - xg**2)*2*a*dOx/rr)
    d = ((dSdx*wg) @ Lg.legvander(xg, ns - 1))*((2*np.arange(ns) + 1)/2)
    c = Lg.legint(d, lbnd=-1, axis=1)
    c[:, 0] = 0
    out = par.thermal_wind_amplitude*c.T
    _sig_cache[key] = out
    return out

def sigma_n(n, r, rpower):
    '''σ_n of S_tw, any even n (0 above ut.tw_nmax)'''
    if ut.dr_high and par.thermal_wind in (1, 2):
        s = sigma_all(r)
        return (r**rpower)*(s[n] if n < len(s) else 0*r)
    return {2: sigma2, 4: sigma4, 6: sigma6}[n](r, rpower) if n in (2, 4, 6) else 0*r

def sigma2(r, rpower):
    if par.thermal_wind == 3:
        return (r**rpower)*(-2/3)*par.dSdtheta*np.ones_like(r)
    if ut.dr_high:
        return sigma_n(2, r, rpower)
    g = prf.gravity(r, 0)
    f0, f0p, f2, f2p = abuX(r, 0), abuX(r, 1), aubX(r, 0), aubX(r, 1)
    out = f0p/3 + f2p/21 + (4/7)*f2/r
    if par.thermal_wind == 2:
        out = (1 + f0)*out + f2*(f0p + f2p)/21
    out = -(2*r**2*par.Gaspard**2/(par.Beyonce*g)) * out
    if par.thermal_wind_amplitude != 1:
        out = par.thermal_wind_amplitude*out
    return (r**rpower)*out

def sigma4(r, rpower):
    if par.thermal_wind == 3:
        return 0*r
    if ut.dr_high:
        return sigma_n(4, r, rpower)
    g = prf.gravity(r, 0)
    f0, f0p, f2, f2p = abuX(r, 0), abuX(r, 1), aubX(r, 0), aubX(r, 1)
    out = f2p - 2*f2/r
    if par.thermal_wind == 2:
        out = (1 + f0)*out + f2*f0p + (4/11)*f2*f2p + (14/11)*f2**2/r
    out = -(6*r**2*par.Gaspard**2/(35*par.Beyonce*g)) * out
    if par.thermal_wind_amplitude != 1:
        out = par.thermal_wind_amplitude*out
    return (r**rpower)*out

def sigma6(r, rpower):   # thermal_wind = 2 only (or with ut.dr_high)
    if ut.dr_high:
        return sigma_n(6, r, rpower)
    g = prf.gravity(r, 0)
    f2, f2p = aubX(r, 0), aubX(r, 1)
    out = -(4*r**2*par.Gaspard**2/(77*par.Beyonce*g)) * f2*(f2p - 2*f2/r)
    if par.thermal_wind_amplitude != 1:
        out = par.thermal_wind_amplitude*out
    return (r**rpower)*out

def sigma2X(r, Dorder):
    tol = 1e-14
    return ut.fonzie( sigma2, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

def sigma4X(r, Dorder):
    tol = 1e-14
    return ut.fonzie( sigma4, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

def sigma6X(r, Dorder):
    tol = 1e-14
    return ut.fonzie( sigma6, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

def ptw(r):   # the p of tw2..tp6; 1 if ThermaD = 0 (ut.adiab_tw), where the heat rows are divided by p
    return np.ones_like(r) if ut.adiab_tw else prf.pressure(r, 0)

def tw2(r, rpower):   # p(r) sigma2(r), p = ρT as in 'pss'
    return (r**rpower)*ptw(r)*sigma2X(r, 0)

def tw4(r, rpower):   # p(r) sigma4(r)
    return (r**rpower)*ptw(r)*sigma4X(r, 0)

def tp2(r, rpower):   # p(r) sigma2'(r)
    return (r**rpower)*ptw(r)*sigma2X(r, 1)

def tp4(r, rpower):   # p(r) sigma4'(r)
    return (r**rpower)*ptw(r)*sigma4X(r, 1)

def tw6(r, rpower):   # p(r) sigma6(r)
    return (r**rpower)*ptw(r)*sigma6X(r, 0)

def tp6(r, rpower):   # p(r) sigma6'(r)
    return (r**rpower)*ptw(r)*sigma6X(r, 1)

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

def tw6X(r, Dorder):
    tol = 1e-14
    return ut.fonzie( tw6, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]

def tp6X(r, Dorder):
    tol = 1e-14
    return ut.fonzie( tp6, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
proffdir = { 'lho':logrhoX, 'vsc':viscoX, 'gra':graviX, 'pss':pressX, 'pdS':pdSdrX, 'dSd':gradSX, 'kps':kappressX, 'aub':aubX, 'svp':svpX, 'pls':plsX, 'abu':abuX, 'spv':spvX, 'psl':pslX,
             'tw2':tw2X, 'tw4':tw4X, 'tp2':tp2X, 'tp4':tp4X, 'tw6':tw6X, 'tp6':tp6X }

def _add_generic_profiles():
    '''profiles of the degrees above 2 (ut.dr_high): a_n f_n, r a_n f_n', r² a_n f_n'' (codes ut.dr_code(n)), and p σn, p σn'
    for n >= 8 (ut.tw_code(n)); as aubX, svpX, plsX and tw2X, tp2X'''
    tol = 1e-14
    for n in ut.dr_high:
        a, sv, pl = ut.dr_code(n)
        svn = lambda r, rpower, n=n: (r**rpower)*r*fX(n, r, 1)
        pln = lambda r, rpower, n=n: (r**rpower)*(r**2)*fX(n, r, 2)
        proffdir[a]  = lambda r, Dorder, n=n: fX(n, r, Dorder)
        proffdir[sv] = lambda r, Dorder, f=svn: ut.fonzie( f, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]
        proffdir[pl] = lambda r, Dorder, f=pln: ut.fonzie( f, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]
    for n in ut.tw_degrees:
        if n <= 6:
            continue
        tw, tp = ut.tw_code(n)
        sX = lambda r, Dorder, n=n: ut.fonzie( lambda rr, rpower: sigma_n(n, rr, rpower), r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]
        twn = lambda r, rpower, sX=sX: (r**rpower)*ptw(r)*sX(r, 0)
        tpn = lambda r, rpower, sX=sX: (r**rpower)*ptw(r)*sX(r, 1)
        proffdir[tw] = lambda r, Dorder, f=twn: ut.fonzie( f, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]
        proffdir[tp] = lambda r, Dorder, f=tpn: ut.fonzie( f, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)[:,-1]
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
            if func1 not in proffdir:   # degrees above 2 of ΔΩ, S_tw (needs utils, imported after this module)
                _add_generic_profiles()
            out1 = proffdir[func1](r, dorder1)

    if func2 is not None:

        if func2 not in proffdir:
            _add_generic_profiles()
        out2 = proffdir[func2](r, dorder2)   

    out = out01*out02*out1*out2


    return out


# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
