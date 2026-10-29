import numpy as np
import sys
import utils as ut
from parameters import par
import smoothed_sun as sms
import background as bg
import background_field as bf

# -------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------ Structure profiles
# -------------------------------------------------------------------------------------------------------------

class structure():  # ------------------------------------------------ What every structure shares: dS/dr, ν and κ
    '''
    ρ0, p0 and g (and d(ln ρ0)/dr, d(ln p0)/dr) come from the subclass that par.structure chooses (structures, below),
    dS/dr and the diffusivities from background.py (par.stratification, par.diffusivity). x = r r_out is the radius in
    star radii. p stands for ρ0 T0.
    '''

    def density(self, r, rpower):  # ------------------ ρ(r)
        return (r**rpower)*np.exp(self.logrho(r, 0))


    def gradS(self, r):  # ------------------ dS/dr at r >= 0
        return bg.gradS(self, r)


    def pdSdr(self, r, rpower):
        # ------------------------ p(r) dS/dr
        out = np.zeros_like(r)
        x = abs(r)<1
        # odd extension to r<0 (only used when ricb=0, where submatrices treats 'pdS' as odd)
        out[x] = self.pressure(r[x],0)*np.sign(r[x])*self.gradS(np.abs(r[x]))
        # -----------------------------------
        return (r**rpower)*out


    def viscosity(self, r, rpower):  # ------------------ ν(r)/ViscosD
        return (r**rpower)*bg.diffusivity(self, r, 'nu')


    def thermal_diffusivity(self, r, rpower):  # ------------------ κ(r)/ThermaD
        return (r**rpower)*bg.diffusivity(self, r, 'kappa')


    def magnetic_diffusivity(self, r, rpower):  # ------------------ η(r)/MagnetD, NEW (2026-10-28)
        return (r**rpower)*bg.diffusivity(self, r, 'eta')


class wu2005_profiles(structure):  # --------------------------------------------------- structure = 'Wu 2005'
    '''
    ρ = (1 - x²)², x = r r_out (Wu 2005, β = 2), with its g and p (normalised to 1 at r = 0); r_out < 1 is the "peel" that
    keeps ρ > 0 at the outer wall. The commented lines are Wu 2005 with β = 1 and the polytrope n = 1.
    '''

    def logrho(self, r, rpower):  # ------ log of the density: log(ρ(r))
        r1 = r*par.r_out
        # out = np.log(1-r1**2)                          # Wu2005    β=1
        out = np.log( (1-r1**2)**2 )                   # Wu2005    β=2
        # out = np.log( np.sin(np.pi*r1)/(np.pi*r1) )    # Polytrope n=1
        # --------------------------------------------------------------
        return (r**rpower)*out


    def gravity(self, r, rpower):  # ------------------------------------------------------ g(r)
        r1 = r*par.r_out
        # out = (5*r1 - 3*r1**3)/2.                                              # Wu2005    β=1
        out = (r1*(35 - 42*r1**2 + 15*r1**4))/8.                               # Wu2005    β=2
        # out = (-(np.pi*r1*np.cos(np.pi*r1)) + np.sin(np.pi*r1))/(np.pi*r1**2)  # Polytrope n=1 
        # --------------------------------------------------------------------------------------
        return (r**rpower)*out


    def pressure(self, r, rpower):  # Normalized to 1 at r=0 ----------------- p(r)
        r1 = r*par.r_out
        # out = -0.5*((-2 + r1**2)*(-1 + r1**2)**2)                 # Wu2005    β=1
        out = -((-1 + r1**2)**3*(26 - 27*r1**2 + 9*r1**4))/26.    # Wu2005    β=2
        # out = np.sin(np.pi*r1)**2/(np.pi**2*r1**2)                # Polytrope n=1  
        # -------------------------------------------------------------------------
        return (r**rpower)*out


    def dlog_p(self,r):  # -------------------------------------------- d(ln p)/dr = r_out d(ln p)/dx
        r1 = r*par.r_out
        # out = (2*r1*(-5 + 3*r1**2))/(2 - 3*r1**2 + r1**4)                                   # Wu2005    β=1
        out = (6*r1*(35 - 42*r1**2 + 15*r1**4))/((-1 + r1**2)*(26 - 27*r1**2 + 9*r1**4))    # Wu2005    β=2
        # out = (-2/r1) + (2*np.pi/np.tan(np.pi*r1))                                          # Polytrope n=1
        # --------------------------------------------------------------------------------------------------- 
        return par.r_out*out
        
        
    def dlog_rho(self,r):  # --------------------------- d(ln ρ)/dr = r_out d(ln ρ)/dx
        r1 = r*par.r_out
        # out = (2*r1)/(-1 + r1**2)                 # Wu2005    β=1
        out = (4*r1)/(-1 + r1**2)                 # Wu2005    β=2
        # out = (-1/r1) + (np.pi/np.tan(np.pi*r1))  # Polytrope n=1
        # ---------------------------------------------------------
        return par.r_out*out


class eight_term_profiles(structure):  # -------------------------------- structure = '8-term fit' (legacy)
    '''
    ρ, p and g of the smooth Model S fit in smoothed_sun.py: 8 even terms in ln ρ, g and p from hydrostatic balance (x = r r_out;
    ρ, p normalised at the centre, g at x = 1). The unit of length is r_out R_sun. ρ is the fit's, ρ(0) = 0.95 of Model S,
    g(x = 1) = 1.02.
    With g normalised at x = 1, N^2 = Beyonce g dS/dr requires Beyonce = g_surf/(r_out R_sun Omega^2) = 4.90e4 for
    Omega = 2 pi 453.1 nHz, giving N/Omega up to about 1000 in the radiative zone, and ν = 1e12 cm^2/s is
    ViscosD = ν/(Omega (r_out R_sun)^2) = 7.4e-5 (r_out = 0.99; bin/units.py computes these).
    Model S's dS/dr (As/x from bin/model_s.at, s = S/cp, r in solar radii) is fitted to 10% rms by stratification =
    'odd top hat' with aux1..aux5 = x1, w1, x2, w2, A = 0, 0.03, 0.68, 0.06, 2.6; its convection zone is adiabatic. The
    central erf(r/w1) needs N of about 11/w1 to be resolved to round-off (N ~ 380 for w1 = 0.03; at N = 256 the dS/dr and
    p dS/dr expansions are truncated at 3e-9 and 4e-7 of their largest coefficient, the other profiles are at round-off
    by k ~ 150). Solar diffusivities (sun/solar-profile.md): visc0 ~ 1e-11, therm0 ~ 1e-6 to 1e-5, far below what can be
    resolved, so in practice they are numerical floors.
    '''

    def logrho(self, r, rpower):  # ------------------------------- log(ρ(r)), even in r
        return (r**rpower)*sms.log_density(r*par.r_out)


    def gravity(self, r, rpower):  # -------------- g(r)/g_surf, odd in r as needed when ricb = 0
        return (r**rpower)*sms.gravity(r*par.r_out)


    def pressure(self, r, rpower):  # ------------------- p(r), normalised to 1 at r = 0, even in r
        return (r**rpower)*sms.pressure(r*par.r_out)


    def dlog_p(self,r):  # ----------------------------------------------------------- d(ln p)/dr
        return par.r_out*sms.dlog_pressure(r*par.r_out)


    def dlog_rho(self,r):  # --------------------------------------------------------- d(ln ρ)/dr
        return par.r_out*sms.dlog_density(r*par.r_out)


class modelS_profiles(structure):  # ----------------------- structure = 'old model S prototype' (legacy)
    '''
    ρ and p from Chebyshev fits of ln ρ and ln p of bin/model_s.at, g from smoothed_sun.py (normalised at x = 1), so the
    three are not in hydrostatic balance (structure = 'old model S prototype').
    ln ρ and ln p from Chebyshev fits of bin/model_s.at (columns As, c_1, ρ/ρ_c, p/p_c, x) over the
    window solar_density_window in x = r r_out, degree solar_density_deg. ρ and p are normalised to their central values, as the
    8-term ones (the overall constants drop out of the equations). The shell fits are not even in r, so ricb = 0 uses
    solar_density_window = 'sun': Model S mirrored to x < 0 and fitted on [-W, W], W = r_out + 0.005, with degree
    solar_density_sun_deg and the odd coefficients dropped, so that ln ρ and ln p are even and smooth at r = 0.
    '''
    def __init__(self):
        import os
        w = par.solar_density_window
        if w is None:
            w = [par.ricb*par.r_out - 0.01, par.r_out + 0.005] if par.ricb > 0 else 'sun'
        T = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_s.at'), skiprows=1)
        x, lrho, lp = T[:, 4], np.log(T[:, 2]), np.log(T[:, 3])
        C = np.polynomial.chebyshev.Chebyshev
        if isinstance(w, str):
            if w != 'sun':
                sys.exit("solar_density_window must be [x_lo, x_hi], None or 'sun'")
            W = par.r_out + 0.005
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
        return (r**rpower)*self.lrho(r*par.r_out)

    def pressure(self, r, rpower):
        return (r**rpower)*np.exp(self.lp(r*par.r_out))

    def dlog_p(self, r):
        return par.r_out*self.lp.deriv()(r*par.r_out)

    def dlog_rho(self, r):
        return par.r_out*self.lrho.deriv()(r*par.r_out)

    def gravity(self, r, rpower):   # as eight_term_profiles
        return (r**rpower)*sms.gravity(r*par.r_out)


class modelS_hydrostatic_profiles(structure):  # ------------------------- structure = 'model S hydrostatic'
    '''
    ρ from an even fit of ln ρ of bin/model_s.at, g from its mass integral and p from hydrostatic balance dp/dx = -ρ g.
    ln ρ is one even polynomial of degree structure_deg over the whole Sun (whatever ricb), fitted on 0 ≤ x ≤ r_out
    (iteratively reweighted least squares, close to minimax in ln ρ). Then g = m(x)/(x² m(1)), m the mass integral of
    that ρ (Model S above x = r_out for m(1)), normalised at the surface as smoothed_sun.py, and p from dp/dx = -K ρ g
    with K the Model S value of -p'/(ρ g), starting from Model S at x = r_out. ρ and p are normalised at the centre.
    With structure_deg = 30 the largest errors against Model S on 0 ≤ x ≤ 0.985 are 1.0% in ρ, 0.5% in p and 0.75% in g;
    with 50, 0.14%, 0.06% and 0.10%.
    '''
    def __init__(self):
        import os
        C = np.polynomial.chebyshev.Chebyshev
        T = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_s.at'), skiprows=1)
        x, rho, p = T[:, 4], T[:, 2], T[:, 3]
        g = x/T[:, 1]
        W, k = par.r_out + 0.001, x <= par.r_out
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
        ks = x >= par.r_out                                         # Model S above r_out, only for m(1)
        m1 = mx(np.array([par.r_out]))[0] + np.sum(np.diff(x[ks])*(x[ks][1:]**2*rho[ks][1:] + x[ks][:-1]**2*rho[ks][:-1])/2)
        self.g = C.interpolate(lambda y: y*((rc(np.outer(y, xs))*xs**2) @ ws)/m1, 500, domain=[-W, W])
        s = (x > 0.05) & (x < 0.95)
        K = np.median(-np.gradient(p, x)[s]/(rho[s]*g[s]))
        def P(y):   # p_ModelS(r_out) + K ∫_|y|^r_out ρ g dx, by quadrature from the surface (keeps the relative accuracy of small p)
            ya = np.abs(np.atleast_1d(y))[:, None]
            t = ya + (par.r_out - ya)*xs
            return np.interp(par.r_out, x, p) + K*(par.r_out - ya[:, 0])*((np.exp(lrho(t))*self.g(t)) @ ws)
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
        return (r**rpower)*self.lrho(r*par.r_out)

    def pressure(self, r, rpower):
        return (r**rpower)*np.exp(self.lp(r*par.r_out))

    def gravity(self, r, rpower):
        return (r**rpower)*self.g(r*par.r_out)

    def dlog_p(self, r):
        return par.r_out*self.lp.deriv()(r*par.r_out)

    def dlog_rho(self, r):
        return par.r_out*self.lrho.deriv()(r*par.r_out)


class boussinesq_profiles(structure):  # ------------------------------------------- structure = 'Boussinesq'
    '''
    Boussinesq reference state: uniform density and pressure (ln ρ = 0, p = 1, so p0 dS/dr = dS/dr), g = r. With
    stratification 'linear' (dS/dr = r) the buoyancy Beyonce g s and the advection r u_r are those of Rieutord 2006 and
    Mirouh et al. 2016, Beyonce = N0².
    '''

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


# -------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------- Instantiate the appropriate profile class
# -------------------------------------------------------------------------------------------------------------
# par.structure and its class (background.STRUCTURES gives the order of the index written to params.dat)
structures = { '8-term fit': eight_term_profiles, 'old model S prototype': modelS_profiles,
               'model S hydrostatic': modelS_hydrostatic_profiles, 'Wu 2005': wu2005_profiles, 'Boussinesq': boussinesq_profiles }

bg.check()
prf = structures[par.structure]()


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
    dS  = lambda rr, rpower: (rr**rpower)*np.sign(rr)*prf.gradS(np.abs(rr))
    out = ut.angine( dS, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out


def graviX(r, Dorder):   # g(r)

    tol = 1e-14
    out = ut.angine( prf.gravity, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out


def etaX(r, Dorder):   # NEW (2026-10-28): magnetic diffusivity η(r)/MagnetD

    tol = 1e-14
    out = ut.angine( prf.magnetic_diffusivity, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out


def pressX(r, Dorder):   # ρ(r)T(r)

    tol = 1e-14
    out = ut.angine( prf.pressure, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
proffdir = { 'lho':logrhoX, 'vsc':viscoX, 'gra':graviX, 'pss':pressX, 'pdS':pdSdrX, 'dSd':gradSX, 'kps':kappressX }
# NEW (2026-10-28): magnetic = 1: 'eta' the magnetic diffusivity, 'bpo' cnorm h(r) of the background field (background_field.py)
proffdir.update({ 'eta':etaX, 'bpo':bf.bpoX, 'bto':bf.btoX })   # bto: cnorm_tor k(r), toroidal B0
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
