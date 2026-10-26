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
        r1 = r*par.aux0
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


class solar_profiles(user_defined_profiles):  # ------------------------- Parts shared by the 'smoothed sun' structures
    '''
    The parts shared by the 'smoothed sun' structures (dS/dr, the diffusivities); ρ, p and g
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
    pdSdr is inherited from user_defined_profiles; the differential rotation profiles are in diff_rot.py.
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
    (dS/dr, ν, κ) as solar_profiles. ρ and p are normalised to their central values, as the
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


class Boussinesq_profiles(user_defined_profiles):  # ---------------------------------------- As Boussinesq profiles
    '''
    Boussinesq reference state: uniform density and pressure (ln ρ = 0, p = 1, so p0 dS/dr = dS/dr), g = r, ν = κ = 1.
    dS/dr (the background temperature gradient) is set by par.heating: 'internal' r (uniform heat sources; with g = r,
    the buoyancy Beyonce g s and the advection r u_r of Rieutord 2006 and Mirouh et al. 2016, Beyonce = N0²), 'differential'
    1/r² (conduction between two walls; needs ricb > 0), 'top hat' the erf top hat of aux1..aux5 as in 'user def'.
    '''

    def density(self, r, rpower): 
        return (r**rpower)*np.ones_like(r)
    
    
    def logrho(self, r, rpower):
        return (r**rpower)*np.zeros_like(r)


    def gravity(self, r, rpower):
        return (r**rpower)*r


    def pressure(self, r, rpower):
        return (r**rpower)*np.ones_like(r)


    def dlog_p(self, r):
        return np.zeros_like(r)


    def dlog_rho(self, r):
        return np.zeros_like(r)


    def gradS(self, r, x1, w1, x2, w2, A):
        if par.heating == 'internal':
            return np.array(r, dtype=float)
        elif par.heating == 'differential':
            return 1/r**2
        elif par.heating == 'top hat':
            return ut.erf_top_hat(r, x1, w1, x2, w2, A)
        sys.exit("heating must be 'internal', 'differential' or 'top hat'")

    def viscosity(self, r, rpower):
        return (r**rpower)*np.ones_like(r)

    def thermal_diffusivity(self, r, rpower):
        return (r**rpower)*np.ones_like(r)


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

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
proffdir = { 'lho':logrhoX, 'vsc':viscoX, 'gra':graviX, 'pss':pressX, 'pdS':pdSdrX, 'dSd':gradSX, 'kps':kappressX }
# the differential-rotation codes (zf, zd, zs, tw, tp + degree) are added by burrito from diff_rot.profile_functions()
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
            if func1 not in proffdir:   # differential rotation: zf, zd, zs, tw, tp (diff_rot.py, imported here: it needs utils)
                import diff_rot
                proffdir.update(diff_rot.profile_functions())
            out1 = proffdir[func1](r, dorder1)

    if func2 is not None:

        if func2 not in proffdir:
            import diff_rot
            proffdir.update(diff_rot.profile_functions())
        out2 = proffdir[func2](r, dorder2)   

    out = out01*out02*out1*out2


    return out


# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
