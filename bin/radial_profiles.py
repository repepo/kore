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


    def gradC(self, r):  # ------------------ NEW (2026-11-08): dC/dr at r >= 0 (compositional = 1)
        return bg.gradC(self, r)


    def rdCdr(self, r, rpower):
        # ------------------------ NEW (2026-11-08): ρ(r) dC/dr, as pdSdr
        out = np.zeros_like(r)
        x = abs(r)<1
        # odd extension to r<0 (only used when ricb=0, where submatrices treats 'rdC' as odd)
        out[x] = self.density(r[x],0)*np.sign(r[x])*self.gradC(np.abs(r[x]))
        # -----------------------------------
        return (r**rpower)*out


    def viscosity(self, r, rpower):  # ------------------ ν(r)/ViscosD
        return (r**rpower)*bg.diffusivity(self, r, 'nu')


    def thermal_diffusivity(self, r, rpower):  # ------------------ κ(r)/ThermaD
        return (r**rpower)*bg.diffusivity(self, r, 'kappa')


    def magnetic_diffusivity(self, r, rpower):  # ------------------ η(r)/MagnetD, NEW (2026-10-28)
        return (r**rpower)*bg.diffusivity(self, r, 'eta')


    def comp_diffusivity(self, r, rpower):  # ------------------ D(r)/CompoD, NEW (2026-11-08)
        return (r**rpower)*bg.diffusivity(self, r, 'D')


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


class polytrope_profiles(structure):  # ------------------------------------------------ structure = 'polytrope'
    '''
    NEW (2026-11-23): the self-gravitating polytrope of index n = polytrope_n (0 <= n <= 4.9): ρ = θⁿ, p = θⁿ⁺¹ (p = ρ0 T0, both
    1 at the centre) and g = θ'(ξ₁x)/θ'(ξ₁) (1 at the surface), θ(ξ) the Lane-Emden function, θ'' + 2θ'/ξ + θⁿ = 0,
    θ(0) = 1, θ'(0) = 0, ξ = ξ₁ x, x = r r_out, ξ₁ its first zero (the surface, x = 1). Then dp/dx = -K ρ g with
    K = -(n + 1) ξ₁ θ'(ξ₁). r_out < 1 keeps ρ > 0 at the outer wall, as with 'Wu 2005'. n = 1 is θ = sin ξ/ξ, ξ₁ = π: the
    commented lines of wu2005_profiles. θ and ξ₁ θ'(ξ₁ x) come from an integration of the Lane-Emden equation (DOP853, from
    its series at ξ = 1e-3, to its zero) and, on |x| <= W = r_out + 0.001 (halfway to the surface for r_out > 0.998), from
    even Chebyshev interpolants of ln(θ/(1 - x²)) and ln(-ξ₁θ'(ξ₁x)/x), which keep θ and θ' to a relative accuracy where
    they are small (a DCT of degree 511, doubled up to 8191: at r_out 0.99, 511 or 1023 up to n = 4.5, 4095 for n = 4.9).
    Against mpmath, p and ρ are within 2e-13 (relative) up to r_out 0.99 (1e-13 up to n = 4.5) and g within 3e-13 of its
    maximum (θ'(ξ₁), read at the event, sets it); beyond, all three within about 1e-15/(1 - r_out), the round-off of θ near
    its zero (for g, from r_out about 0.9999995, the truncation of its interpolant at degree 8191: 1.5e-10 at r_out
    0.9999999). The integration itself is used beyond W (g and ρ* up to x = 1 for Clairaut's equation, centrifugal.py,
    which takes the exact ρ* = ρ/ρ̄ of rho_star).
    '''
    def __init__(self):
        from scipy.integrate import solve_ivp
        import scipy.fft as sft
        n = float(par.polytrope_n)
        self.n = n
        def rhs(xi, v):
            return [v[1], -np.maximum(v[0], 0.)**n - 2*v[1]/xi]
        def zero(xi, v):
            return v[0]
        zero.terminal, zero.direction = True, -1
        x0 = 1e-3   # θ = 1 - ξ²/6 + n ξ⁴/120 - n(8n - 5) ξ⁶/15120 + ...: the next term is below 3e-27 there (3e-23 in θ')
        # max_step 0.02: interp samples the dense output, which with DOP853's own step sizes is 10 to 100 times less accurate
        # than the steps (θ' off by up to 3e-13 of its maximum near ξ ~ 1 for n = 1.5 to 4.9, a noise in the Chebyshev
        # coefficients above kore's cuts). With it θ is within 1e-14 and θ' within 3e-14 of its maximum on |x| <= W (against
        # mpmath; 5e-13 within 1e-9 of the zero, where the event's θ'(ξ₁) and so g are off by up to 3e-13), in 130 to 8600
        # steps (0.01 to 0.7 s, n = 4.9)
        sol = solve_ivp(rhs, (x0, 1e4), [self._series(x0, n, 0), self._series(x0, n, 1)], method='DOP853', rtol=2.5e-14,
                        atol=1e-18, events=zero, dense_output=True, max_step=0.02)
        if not sol.success or sol.t_events[0].size == 0:
            sys.exit("structure 'polytrope': the Lane-Emden integration for n = %r failed (%s)" % (n, sol.message))
        self.xi1, self.dth1 = sol.t_events[0][0], sol.y_events[0][0][1]       # ξ₁ and θ'(ξ₁)
        self.sol, self.x0 = sol, x0
        # W < 1: L and L2 below have a term (1 - x)ⁿ⁺¹ at the surface (θ one in (1 - x)ⁿ⁺²), singular for a non-integer n:
        # with W = 1 their coefficients fall only algebraically (those of L2 still 1e-15 to 2e-14 at degree 8191 for n of
        # about 0.005 to 0.35)
        self.W = W = par.r_out + min(0.001, (1 - par.r_out)/2)
        def interp(func):
            # an even function on |x| <= W, sampled at bf.cheb_interp's points taken at W y: degree 511, doubled up to 8191
            # until the largest of its last 64 coefficients is below 1e-15 of the largest (or of 1), or below 1e-12 and no
            # longer falls by 4 from one degree to the next (the samples' round-off, about 1e-16/(1 - x) relative near the
            # surface: from r_out about 0.998), or is below 1e-12 at 8191 (for L from r_out about 0.99999, the larger
            # round-off there; for L2, whose samples have none, from about 0.9999995 and for small n only, a truncation,
            # which leaves g within about 1e-15/(1 - r_out), as p)
            prev = np.inf
            for deg in (511, 1023, 2047, 4095, 8191):
                m = deg + 1
                y = np.sin(0.5*np.pi*(m - 1 - 2*np.arange(m))/m)
                c = sft.dct(func(W*y))/m
                c[0] /= 2
                c[1::2] = 0.                                        # even, exactly
                a = np.abs(c)
                tail = a[-64:].max()/max(a.max(), 1.)
                if tail < 1e-15 or prev/4 < tail < 1e-12 or (deg == 8191 and tail < 1e-12):
                    break
                prev = tail
            else:
                sys.exit("structure 'polytrope': θ of n = %r is not resolved by 8192 Chebyshev polynomials on |x| <= %g "
                         "(r_out too close to 1 for this n)" % (n, W))
            # no cut: the coefficients end in round-off. A cut at 1e-14 (of θ, in a first version) left a ripple of that
            # size, above kore's own 1e-15 cuts, which filled the operators (1.7 times the nonzeros for n = 4.5, r_out 0.99,
            # ricb 0.35 at N = 256)
            return np.polynomial.chebyshev.Chebyshev(c, domain=[-W, W])
        # Interpolants of ln(θ/(1 - x²)) and ln(-ξ₁θ'(ξ₁x)/x), both even and finite up to the surface, carry θ and θ' with
        # the relative accuracy of the integration where they are small (the envelope of a condensed polytrope, near the
        # wall). Interpolants of θ and θ' themselves are accurate only in absolute terms; their round-off, relative to a
        # small θ, passed kore's cuts of ln ρ, d ln ρ/dr, ... and filled the operators: 2.4 times the nonzeros of an exact θ
        # for n = 4.9, r_out 0.9, ricb 0.35 at N = 256, against 1.18 times with these
        self.L = interp(lambda y: np.log(self._ode(np.abs(y), 0)/((1 - np.abs(y))*(1 + np.abs(y)))))
        self.L2 = interp(lambda y: np.log(-self._ode(np.abs(y), 1)/np.abs(y)))

    @staticmethod
    def _series(s, n, k):   # θ (k = 0) or θ' (k = 1) near ξ = 0
        if k == 0:
            return 1 - s**2/6 + n*s**4/120 - n*(8*n - 5)*s**6/15120
        return -s/3 + n*s**3/30 - n*(8*n - 5)*s**5/2520

    def _ode(self, x, k):
        '''θ(ξ₁ x) (k = 0) or ξ₁ θ'(ξ₁ x) (k = 1) at x >= 0 from the integration (its series below ξ = 1e-3)'''
        xi = self.xi1*np.asarray(x, dtype=float)
        lo = xi < self.x0
        out = np.empty(np.shape(xi))
        if not lo.all():   # (OdeSolution fails on no points at all)
            out[~lo] = self.sol.sol(np.clip(xi[~lo], self.x0, self.xi1))[k]
        out[lo] = self._series(xi[lo], self.n, k)
        return out if k == 0 else self.xi1*out

    def _theta(self, x):   # ln θ and ξ₁ θ' at x = r r_out: the interpolants up to W, the integration beyond
        x = np.asarray(x, dtype=float)
        ax = np.abs(x)
        a = ax <= self.W
        lt, d = np.empty(np.shape(x)), np.empty(np.shape(x))
        lt[a] = np.log((1 - ax[a])*(1 + ax[a])) + self.L(x[a])
        d[a] = -x[a]*np.exp(self.L2(x[a]))
        if not a.all():
            with np.errstate(divide='ignore'):
                lt[~a] = np.log(np.maximum(self._ode(ax[~a], 0), 0.))   # -inf at the surface (θ there is ±1e-16)
            d[~a] = np.sign(x[~a])*self._ode(ax[~a], 1)
        return lt, d

    def dlog_theta(self, r):   # ------------------------------------------- d(ln θ)/dr = r_out ξ₁ θ'/θ, odd in r
        lt, d = self._theta(r*par.r_out)
        return par.r_out*d*np.exp(-lt)

    def logrho(self, r, rpower):   # ------------------------------------------------ ln ρ = n ln θ, even in r
        return (r**rpower)*self.n*self._theta(r*par.r_out)[0]

    def pressure(self, r, rpower):   # ---------------------------------------------- p = θⁿ⁺¹, 1 at r = 0
        return (r**rpower)*np.exp((self.n + 1)*self._theta(r*par.r_out)[0])

    def gravity(self, r, rpower):   # ----------------------------------- g = θ'(ξ₁ x)/θ'(ξ₁), 1 at x = 1, odd in r
        return (r**rpower)*self._theta(r*par.r_out)[1]/(self.xi1*self.dth1)

    def rho_star(self, x):
        '''
        NEW (2026-11-24): ρ* = ρ/ρ̄ = -ξθⁿ/(3θ') at x (ρ̄ the mean density inside x; 1 at the centre, 0 at the surface for
        n > 0), for Radau's equation (centrifugal.radau), in place of its ρ* from g/x at degree 511, which is off at the
        surface for a non-integer n below 1 (the (1 - x)ⁿ⁺¹ term of g) and in the core of n above 4.5. From the integration
        (radau calls it at one x at a time, too often for the interpolants of degree up to 8191)
        '''
        x = np.abs(np.asarray(x, dtype=float))
        out = np.ones(x.shape)
        z = x > 0
        th = np.where(x[z] < 1, np.maximum(self._ode(x[z], 0), 0.), 0.)   # θ(ξ₁) = 0 (the integration's is ±1e-16, whose
        out[z] = -self.xi1**2*x[z]*th**self.n/(3*self._ode(x[z], 1))      # n-th power is not small for a small n)
        return out

    def dlog_p(self, r):   # --------------------------------------------------- d(ln p)/dr = (n + 1) d(ln θ)/dr
        return (self.n + 1)*self.dlog_theta(r)

    def dlog_rho(self, r):   # ------------------------------------------------- d(ln ρ)/dr = n d(ln θ)/dr
        return self.n*self.dlog_theta(r)


class jones2011_profiles(structure):  # -------------------------------------------- structure = 'Jones 2011'
    '''
    NEW (2026-11-23): the polytrope of the anelastic dynamo benchmark (Jones et al. 2011, Icarus 216, 120; MagIC, Rayleigh),
    a shell only (ricb > 0): gravity of a central mass, g = 1/r² (1 at the outer wall r = 1), and ζ = c0 + c1 d/r,
    ρ0 = ζⁿ, p0 = ζⁿ⁺¹ (= ρ0 T0), n = polytrope_n, with β = ricb, d = 1 - β, ζ_o = (β + 1)/(β exp(N_ρ/n) + 1),
    c0 = (2ζ_o - β - 1)/(1 - β), c1 = (1 + β)(1 - ζ_o)/(1 - β)², N_ρ = polytrope_Nrho = ln(ρ_i/ρ_o) the number of density
    scale heights across the shell: ζ = 1 at mid-depth, ζ_o at the outer wall. Hydrostatic:
    dp/dr = -K ρ g with K = (n + 1) c1 d. The reference state is adiabatic; the benchmark's convection is driven by the
    conduction state between the walls (stratification 'conduction').
    '''
    def __init__(self):
        n, b = float(par.polytrope_n), par.ricb
        self.n = n
        zo = (b + 1)/(b*np.exp(par.polytrope_Nrho/n) + 1)
        self.c0 = (2*zo - b - 1)/(1 - b)
        self.c1d = (1 + b)*(1 - zo)/(1 - b)        # c1 d
        self.zo, self.zi = zo, self.c0 + self.c1d/b  # ζ at the outer and the inner wall

    def zeta(self, r):   # c0 + c1 d/r as ζ_o + c1 d (1 - r)/r, free of cancellation near r = 1 when ζ_o is small
        return self.zo + self.c1d*(1 - r)/r

    def dlog_zeta(self, r):   # ----------------------------------------- d(ln ζ)/dr = -c1 d/(r² ζ)
        return -self.c1d/(r**2*self.zeta(r))

    def logrho(self, r, rpower):
        return (r**rpower)*self.n*np.log(self.zeta(r))

    def pressure(self, r, rpower):
        return (r**rpower)*self.zeta(r)**(self.n + 1)

    def gravity(self, r, rpower):
        return (r**rpower)/r**2

    def dlog_p(self, r):
        return (self.n + 1)*self.dlog_zeta(r)

    def dlog_rho(self, r):
        return self.n*self.dlog_zeta(r)


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
        # FIX (2026-11-08, phase 6d): the three interpolants of degree 500 by a DCT (background_field.cheb_interp, as B0's
        # profiles since phase 6c). numpy's Chebyshev.interpolate (chebvander's recurrence) left round-off of up to 2e-14 (g)
        # and 1.3e-13 (ln p) of the largest coefficient at degrees 300 to 500, so the cut below landed early (degree 191
        # instead of 195 for g and 200 instead of 208 for ln p at r_out 0.985, structure_deg 30) or, for ln p, never (e.g.
        # r_out 0.99 with structure_deg 40): d ln p/dr was then 2.5e-10 off, and the operators of p filled up at N >= 450
        rc = bf.cheb_interp(lambda y: np.exp(lrho(np.abs(y))), 500, [-W, W])
        xs, ws = np.polynomial.legendre.leggauss(200)
        xs, ws = (xs + 1)/2, ws/2
        mx = lambda y: y**3*((rc(np.outer(y, xs))*xs**2) @ ws)       # m(x) = x³ ∫₀¹ s² ρ(s x) ds (units of ρ_c R³)
        ks = x >= par.r_out                                         # Model S above r_out, only for m(1)
        m1 = mx(np.array([par.r_out]))[0] + np.sum(np.diff(x[ks])*(x[ks][1:]**2*rho[ks][1:] + x[ks][:-1]**2*rho[ks][:-1])/2)
        self.g = bf.cheb_interp(lambda y: y*((rc(np.outer(y, xs))*xs**2) @ ws)/m1, 500, [-W, W])
        s = (x > 0.05) & (x < 0.95)
        K = np.median(-np.gradient(p, x)[s]/(rho[s]*g[s]))
        def P(y):   # p_ModelS(r_out) + K ∫_|y|^r_out ρ g dx, by quadrature from the surface (keeps the relative accuracy of small p)
            ya = np.abs(np.atleast_1d(y))[:, None]
            t = ya + (par.r_out - ya)*xs
            return np.interp(par.r_out, x, p) + K*(par.r_out - ya[:, 0])*((np.exp(lrho(t))*self.g(t)) @ ws)
        self.lp = bf.cheb_interp(lambda y: np.log(P(y)/P(0.0)), 500, [-W, W])   # p(0) = 1
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
               'model S hydrostatic': modelS_hydrostatic_profiles, 'Wu 2005': wu2005_profiles, 'Boussinesq': boussinesq_profiles,
               'polytrope': polytrope_profiles, 'Jones 2011': jones2011_profiles }

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


def densX(r, Dorder):   # NEW (2026-11-08): ρ(r), the composition's time derivative

    tol = 1e-14
    out = ut.angine( prf.density, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out


def rdCdrX(r, Dorder):   # NEW (2026-11-08): ρ(r) dC/dr, the composition's advection

    tol = 1e-14
    out = ut.angine( prf.rdCdr, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out


def gradCX(r, Dorder):   # NEW (2026-11-08): dC/dr, odd extension to r<0 (used when CompoD=0)

    tol = 1e-14
    dC  = lambda rr, rpower: (rr**rpower)*np.sign(rr)*prf.gradC(np.abs(rr))
    out = ut.angine( dC, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out


def DrhoX(r, Dorder):   # NEW (2026-11-08): D(r)ρ(r), the composition's diffusion

    tol = 1e-14
    Drh = lambda rr, rpower: prf.comp_diffusivity(rr, rpower) * prf.density(rr, 0)  # ut.angine needs a function f(r, rpower)
    out = ut.angine( Drh, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

    return out


def graviX(r, Dorder, effective=True):   # g(r)
    '''NEW (2026-11-05): with centrifugal = 1 and centrifugal_q > 0, the effective gravity g - (2/3) Ω² r of the deformed
    body (its centrifugal monopole, centrifugal.g_monopole), which the buoyancy takes (code 'gra'); effective=False gives
    the structure's own g, which the thermal wind takes (as diff_rot.sigma_coefficients). The same otherwise.'''

    tol = 1e-14
    grav = prf.gravity
    if effective and par.centrifugal and par.centrifugal_q != 0:
        import centrifugal as cf
        grav = lambda rr, rpower: prf.gravity(rr, rpower) + (rr**rpower)*cf.g_monopole(rr)
    out = ut.angine( grav, r, par.N, par.ricb, ut.rcmb, Dorder, tol, 0)

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
# NEW (2026-11-08): compositional = 1: 'dns' ρ, 'rdC' ρ dC/dr, 'dCd' dC/dr, 'Drh' D ρ (as 'pss', 'pdS', 'dSd', 'kps' for s)
proffdir.update({ 'dns':densX, 'rdC':rdCdrX, 'dCd':gradCX, 'Drh':DrhoX })
# added by burrito on first use (_more_codes): the differential-rotation codes (zf, zd, zs, tw, tp + degree) from
# diff_rot.profile_functions(), the background field degree by degree (hp, kt + degree) from mag_algebra.profile_functions()
# and (NEW 2026-11-05) the centrifugal map (ce0..ce3) from centrifugal.profile_functions() and, with magnetic = 1, the
# products of its amplitudes and the background field degree by degree (E, N, C, Q + p, t + degree) from
# centrifugal.magnetic_profile_functions(); NEW (2026-11-06, phase 2) with the profiles of the diffusion (E, N, C, Q, K + vs,
# kp, et: Evs, Nkp, Ket, ...) and with the shapes of ΔΩ (E, N, C, Q, K + z + degree: Ez0, Nz2, ...) from
# centrifugal.diffusion_profile_functions() and centrifugal.rotation_profile_functions() (NEW 2026-11-08: Dr, D ρ, as well:
# EDr, NDr, ...), and (NEW 2026-11-08) the background flow's (w + q, s, t, h, d, c, e + degree) from bgflow.profile_functions()
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------


def _more_codes(c):
    '''the profile codes added on first use: the differential rotation's (diff_rot.py), NEW (2026-10-30), the background
    field's degree by degree, hp? and kt? (mag_algebra.py) and, NEW (2026-11-05), the centrifugal map's ce0..ce4 and its
    products with the background field, Ep?, Nt?, ..., with the profiles of the diffusion, Evs, Nkp, Ket, EDr, ... and (NEW
    2026-11-06) with the shapes of ΔΩ, Ez0, Nz2, ... (centrifugal.py); imported here because they need utils'''
    if c[:2] in ('hp', 'kt'):
        import mag_algebra
        proffdir.update(mag_algebra.profile_functions())
    elif c[:2] == 'ce':   # NEW (2026-11-05)
        import centrifugal
        proffdir.update(centrifugal.profile_functions())
    elif c[0] in 'ENCQ' and c[1] in 'pt':   # NEW (2026-11-05): products of the map's amplitudes and the background field
        import centrifugal
        proffdir.update(centrifugal.magnetic_profile_functions())
    elif c[0] in 'ENCQK' and c[1:] in ('vs', 'kp', 'et', 'Dr'):   # NEW (2026-11-05, phase 2): products of the map's amplitudes
        import centrifugal                       # and the profiles of the diffusion, ν, κ p, η, D ρ (Evs, Nkp, ...; Dr NEW 2026-11-08)
        proffdir.update(centrifugal.diffusion_profile_functions())
    elif c[0] in 'ENCQK' and c[1] == 'z':   # NEW (2026-11-06, phase 2): products of the map's amplitudes and the shapes of ΔΩ
        import centrifugal
        proffdir.update(centrifugal.rotation_profile_functions())
    elif c[0] in 'wW':   # NEW (2026-11-08): the background flow degree by degree, wq?, ws?, wt?, wh?, wd?, wc?, we? (bgflow.py;
        # NEW 2026-11-18: Wq036, Wt095, ... above degree 35)
        import bgflow
        proffdir.update(bgflow.profile_functions())
    else:
        import diff_rot
        proffdir.update(diff_rot.profile_functions())

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
            if func1 not in proffdir:   # zf, zd, zs, tw, tp (diff_rot.py) or hp, kt (mag_algebra.py): _more_codes
                _more_codes(func1)
            out1 = proffdir[func1](r, dorder1)

    if func2 is not None:

        if func2 not in proffdir:
            _more_codes(func2)
        out2 = proffdir[func2](r, dorder2)   

    out = out01*out02*out1*out2


    return out


# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
