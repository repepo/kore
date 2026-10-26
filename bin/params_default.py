import numpy as np
#import targets as tg


class default_params():

    def __init__(self):

        # ----------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------ Physics modules       
        # ----------------------------------------------------------------------------------------------------------------------
        self.thermal = 0  # set to 1 to include the thermal (entropy) equation; the Navier-Stokes equation is always solved


        # ----------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------- Solution symmetry and inner core size
        # ----------------------------------------------------------------------------------------------------------------------
        self.m    = 0  # Azimuthal wave number
        self.symm = 1  # Equatorial symmetry, 1 for symmetric, -1 for antisymmetric
        self.ricb = 0 # Solid inner core radius 


        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------- Star/planet model parameters
        # ----------------------------------------------------------------------------------------------------------------------
        self.model = 'poly.n1_gamma3.h5' # MESA, GYRE, or astropy table model file

        self.model_type = 'poly'              # For polytropic profiles supplied by Gyre
        # self.model_type = 'mesa'              # For MESA file format profiles
        # self.model_type = 'gsm'               # For Gyre file format profiles
        # self.model_type = 'astropy table'     # For explicitly defined profiles (supply r, ρ and when required g, (T or p), (N^2 or dS), in array format)
        # self.model_type = 'Boussinesq'        #  for Boussinesq reference state, else anelastic reference state
        # self.model_type = 'user def'          # For implicitly defined profiles (adjust ρ and when required g, p, pdS in radial_profiles.py)
        # self.model_type = 'smoothed sun'      # Smooth fits to solar Model S (structure, below) with a simple dS/dr, see radial_profiles.solar_profiles

        # 'smoothed sun' structure (ρ, p, g): structure = 'model S hydrostatic' for new runs, full sphere or shell (an even fit
        # of ln ρ of degree structure_deg over the whole Sun, g from its mass integral, p from dp/dr = -ρ g; radial_profiles.
        # modelS_hydrostatic_profiles). structure_deg 30 is within 1.0 %, 0.5 %, 0.75 % of Model S in ρ, p, g, and 50 within
        # 0.14 %, 0.06 %, 0.10 %, where HFR m 8 stops changing (30 is 0.15 nHz off there in ω, 0.9 nHz in σ). Degrees of 40
        # and above need about 16 more lmax for the HFR ℓ-convergence; use 50 to match papers that use Model S itself.
        # Legacy, kept so that older parameters.py files reproduce their results: '8-term fit' (smoothed_sun.py, rough but
        # hydrostatic) and 'old model S prototype' (fits of bin/model_s.at, below; ρ and p fitted separately and g from the
        # 8-term fit, so not hydrostatic; to be removed, with solar_density below, once 'model S hydrostatic' is validated).
        # None: from the old name solar_density ('fit', the default, is '8-term fit'; 'model S' is 'old model S prototype').
        self.structure = None
        self.structure_deg = 30

        # solar_density (legacy), the old name of structure ('smoothed sun' density and pressure): 'fit' (smoothed_sun.py, the default)
        # or 'model S' (structure 'old model S prototype': Chebyshev fits of ln ρ, ln p of
        # bin/model_s.at of degree solar_density_deg over solar_density_window = [x_lo, x_hi] in x = r aux0; None: the shell
        # [ricb aux0, aux0] widened by 0.01 below and 0.005 above if ricb > 0, 'sun' if ricb = 0; 'sun': one fit over the whole
        # Sun, even in r (Model S mirrored to x < 0, odd coefficients dropped), of degree solar_density_sun_deg, on
        # [-(aux0 + 0.005), aux0 + 0.005], the same for any ricb; ln rho to 1.7e-3, mostly at the base of the convection zone,
        # where the shell fit reaches 2e-4). The 'fit' density is up to 48 % off near the
        # surface, which moves n = 1 Rossby, HFR and columnar modes by 1.5-8 nHz (docs/heat-equation.md §6). g is unchanged.
        self.solar_density = 'fit'
        self.solar_density_deg = 30
        self.solar_density_window = None
        self.solar_density_sun_deg = 60

        # profile smoothing power
        self.smopo = 0

        # ----------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------- Auxiliary variables        
        # ----------------------------------------------------------------------------------------------------------------------
        self.Ek     = 0
        self.Etherm = 0

        self.aux0   = 0
        self.aux1   = 0
        self.aux2   = 0
        self.aux3   = 0
        self.aux4   = 0
        self.aux5   = 0

        # Diffusivity profiles of model_type = 'smoothed sun' (radial_profiles.solar_profiles): ν and κ are ViscosD and ThermaD
        # (the convection-zone values) times ν/ν_cz = visc0 and κ/κ_cz = therm0 in the radiative zone and 1 in the convection
        # zone, constant in each with the erf step of dS/dr at x2 = aux3, width w2 = aux4 (needed also when thermal = 0 unless
        # visc0 = therm0 = 1). Solar: visc0 ~ 1e-11, therm0 ~ 1e-6 to 1e-5 (ν ~ 10, κ_rad ~ 1e5 to 1e7 cm²/s against
        # ν_t ~ κ_t ~ 1e12), so in practice they are numerical floors; see sun/solar-profile.md. visc_pow and therm_pow multiply
        # ν and κ by (ρ(x2)/ρ(r))**pow: 0 keeps the kinematic diffusivities constant (recommended), 1 the dynamic ones (ρν, ρκ),
        # ViscosD and ThermaD being then the values at x2. rvisc and hvisc are not used by this model.
        self.visc0     = 1.0
        self.therm0    = 1.0
        self.visc_pow  = 0.0
        self.therm_pow = 0.0
        self.rvisc     = 0.0
        self.hvisc     = 0.0

        # Radiative interior of Mukhopadhyay, Bekki, Zhu & Gizon 2025 (arXiv 2512.12691, their eqs. 8-10) for 'smoothed sun',
        # used when solar_rz = 1 in place of the aux1..aux5 top hat and of the aux3/aux4 diffusivity step (docs/heat-equation.md §9):
        #   dS/dr = solar_rz_dSdr/4 [1 - erf((x - x1)/d)] [1 - erf((x - x2)/d)] erf(x/w0),  [x1, x2, d] = solar_rz_steps,
        #   w0 = solar_rz_core, x = r aux0 in R☉; erf(x/w0) makes it odd and smooth at r = 0 (no effect above x ~ 3 w0).
        #   solar_rz_dSdr = (∂s0/∂r)_RZ aux0 R☉/c_p: 2.615 for their 1.32e-2 erg/g/K/cm with c_p = 3.46e8 and aux0 = 0.985.
        #   ν/ν_cz and κ/κ_cz go from visc0, therm0 below to 1 above as (1 + erf((x - x_OS)/d_OS))/2, [x_OS, d_OS] = solar_rz_os
        #   (their ν_RZ/ν_CZ = 0.01, κ_RZ/κ_CZ = 0.04).
        #   solar_rz_delta_cz, solar_rz_delta_sf add a convection-zone superadiabaticity δ (Bekki 2024, eq. 2):
        #   dS/dr += δ(x) d(ln p0)/dr (1 + erf((|x| - x_OS)/d_OS))/2, δ(x) = δ_cz + (δ_sf - δ_cz) exp(-((|x| - 0.985)/0.015)^2),
        #   δ < 0 being subadiabatic (Bekki's model 3-sub: δ_cz = -5e-7, δ_sf = 1e-3).
        self.solar_rz       = 0
        self.solar_rz_delta_cz = 0.0
        self.solar_rz_delta_sf = 0.0
        self.solar_rz_dSdr  = 2.615
        self.solar_rz_steps = [0.61, 0.68, 0.04]
        self.solar_rz_core  = 0.05
        self.solar_rz_os    = [0.70, 0.04]


        # ----------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------- Hydrodynamic parameters
        # ----------------------------------------------------------------------------------------------------------------------
        # Inner core mechanical boundary conditions
        # Use 0 for stress-free, 1 for no-slip or forced boundary flow. Ignored if ricb = 0
        self.bci = 0

        # CMB mechanical boundary conditions
        # Use 0 for stress-free, 1 for no-slip or forced boundary flow
        self.bco = 0

        # Differential rotation parameters. Set to 0 for solid body rotation. Ω_ref is the frame rotation rate (Gaspard in units of
        # the time scale) and ΔΩ is relative to it, so any timescale works: U0 = Gaspard ΔΩ r sinθ e_φ. User guide: docs/differential-rotation.md.
        # Only model_type = 'user def' and 'smoothed sun' define a rotation profile. ---------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        self.diff_rot = 0 # If 1, the code will solve for a perturbation to a background state with differential rotation. 
        # ΔΩ = diff_rot_ampl Σ_n f_n(r) P_n(cos θ), n even (bin/diff_rot.py); diff_rot_type chooses the shapes f_n:
        self.diff_rot_type = 'Y20'  # f2 = 1: ΔΩ = a P2(cos θ), a = diff_rot_ampl (P2 unnormalised, despite the name)
        # self.diff_rot_type = 'rigid'  # f0 = 1: a rigid offset, test profile: λ(a, Gaspard) = λ(diff_rot = 0, Gaspard (1 + a)) - i m Gaspard a exactly
        # self.diff_rot_type = 'Y20-wall-bounded'  # f2 = (1 - r)(r - ricb)
        # self.diff_rot_type = 'shellular'  # [Baruteau & Rieutord 2013] Ω ∝ (r/R)^σ: f0 = r^σ - 1, σ = diff_rot_param
        # self.diff_rot_type = 'shellular_boussinesq'  # [Mirouh et al. 2016] f0 = (1 - r²)/2, diff_rot_ampl = N0²
        # self.diff_rot_type = 'cylindrical'  # [Baruteau & Rieutord 2013] ΔΩ = a r² sin²θ: f0 = (2/3) r², f2 = -(2/3) r²
        # self.diff_rot_type = 'conical'  # [Guenel et al. 2016] ΔΩ = a sin²θ: f0 = 2/3, f2 = -2/3
        # self.diff_rot_type = 'solar'  # fit to the helioseismic Ω, P0 + P2 stepping up across a tanh tachocline at x = 0.71
        #   of width 0.1, relative to 435 nHz: the frame (Ω_ref) must rotate at 435 nHz (ViscosD, ThermaD, Beyonce for that rate).
        #   x is the fractional solar radius, r aux0 with model_type = 'smoothed sun' (r otherwise). Needs N ~ 400 with ricb = 0.
        # self.diff_rot_type = 'solar_P4'  # Ω/2π = Ω_RZ + [Ω_CZ(θ) - Ω_RZ] (1 + erf((x - rtc)/dtc))/2, Ω_CZ(θ) = c0 + c2 cos²θ
        #   + c4 cos⁴θ (Mukhopadhyay et al. 2026; diff_rot_solar below); ΔΩ = Ω/Ω_ref - 1, Ω_ref/2π = diff_rot_frame nHz: P0, P2, P4
        # self.diff_rot_type = 'tabulated'  # Ω(r, θ) from diff_rot_file (.npz with ome[θ, r] in nHz, r in R☉, theta in rad, as the
        #   Mukhopadhyay et al. 2025 Edmond file data/diff_rot.npz; tests/diff_rot/ls_to_edmond_layout.py converts the Larson & Schou
        #   2018 table), projected on P0, P2, ..., P_diff_rot_nmax at each table radius and fitted in r; ΔΩ = Ω/Ω_ref - 1
        # self.diff_rot_type = 'user'  # f_n given as numpy expressions of r, x and p = diff_rot_param in diff_rot_profiles
        #   'Y20-wall-bounded' and 'shellular' (unless σ is even and > 0) are not smooth at r = 0: use ricb > 0.
        self.diff_rot_ampl = 1.0  # the factor of ΔΩ, for every type (1 for 'solar', 'solar_P4', 'tabulated': the profile itself)
        self.diff_rot_param = 0.0  # the shape parameter of the types that have one: σ for 'shellular'; p in 'user' expressions
        self.diff_rot_term_amp = [1.0]*9  # a_0, a_2, ..., a_16: factors of the P0, P2, ..., P16 terms of ΔΩ, any type, after
        # diff_rot_ampl (missing entries: 1), e.g. [1, 1, 0] drops the P4 part of 'solar_P4'; S_tw follows the scaled ΔΩ
        self.diff_rot_profiles = {}  # 'user': {n: 'expression'}, the shape f_n of each even degree n as a numpy expression of r
        # (kore radius), x (fractional solar radius) and p (diff_rot_param), with sin, cos, tan, arcsin, arccos, arctan, sinh,
        # cosh, tanh, exp, log, log10, sqrt, abs, sign, pi, where, minimum, maximum, erf, erfc; e.g. a cylindrical rotation
        # {0: '(2/3)*r**2', 2: '-(2/3)*r**2'}. No '#' (the expressions are written to params.dat as a comment).
        # The thermal-wind entropy (thermal_wind = 1, 2) is that of ΔΩ (with its factor diff_rot_ampl), times thermal_wind_amplitude.

        self.thermal_wind = 1  # With thermal = 1 and diff_rot = 1 the heat equation gets the advection i m Gaspard ΔΩ s and,
        # if thermal_wind = 1 (needs Beyonce > 0), the advection of the thermal-wind background entropy S_tw(r,θ)
        # that Ω(r,θ) implies, to first order in ΔΩ (docs/heat-equation.md, eq. 3); thermal_wind = 2 uses the exact (1+ΔΩ)² balance,
        # which adds O(ΔΩ²) to the σn and degrees up to 2 n_max + 2 (diff_rot.sigma_coefficients). thermal_wind = 0 keeps the advection term only
        # (e.g. Mirouh et al. 2016, Ω(r) imposed). With ThermaD = 0 and thermal_wind >= 1 the heat rows are r² (eq)/p in a C^(1) basis.
        # thermal_wind = 3 prescribes S_tw instead, independent of Ω (Bekki et al. 2024): ∂S_tw/∂θ = dSdtheta sin 2θ, radially
        # uniform, i.e. S_tw = -(2/3) dSdtheta P2(cos θ); dSdtheta < 0 gives warm poles. For uniform rotation use diff_rot = 1 with
        # diff_rot_type = 'rigid' and diff_rot_ampl = 0.
        # spin_doctor's residuals include these terms (flow.dat: Wdr, Wdr_im; thermal.dat: Wdr_thm); tests/heat_dr/terms.py gives the matrix-level balance.
        self.dSdtheta = 0.0  # thermal_wind = 3 only: ∂S_tw/∂θ = dSdtheta sin 2θ, S_tw in the units of s (δs/c_p), so dSdtheta = Δθs/c_p
        # (Bekki et al. 2022, 2024 quote Δθs in erg/g/K with c_p = 4.17e8 erg/g/K, fully ionised hydrogen; Model S has ~3.46e8)
        self.thermal_wind_amplitude = 1.0  # thermal_wind = 1, 2: factor of the thermal-wind entropy S_tw (1: the balanced value)
        self.thermal_wind_radial = 1  # 1: keep u_r ∂S_tw/∂r in the heat equation (consistent); 0: drop it, as Bekki et al. 2022 and
        # Mukhopadhyay et al. 2025 (their background has ∂s0/∂r = 0 only; with S_tw(r, θ) the term is O(S_tw/r), not small)
        self.diff_rot_frame = 456.0  # 'solar_P4', 'tabulated': Ω_ref/2π in nHz, the rotation rate of the frame (time unit 1/Ω_ref)
        self.diff_rot_solar = [464.0, -76.0, -50.0, 433.5, 0.71, 0.045]  # 'solar_P4': c0, c2, c4, Ω_RZ (nHz), rtc, dtc
        self.diff_rot_file = ''  # 'tabulated': path of the .npz (ome, r, theta), absolute or relative to the run directory
        self.diff_rot_nmax = 8  # 'tabulated': highest even degree n kept in Ω (up to 16 with thermal_wind = 2); the heat rows couple Δl up to 2 nmax + 2
        # with thermal_wind = 2. For the HMI profile nmax = 8 leaves up to 7 nHz at 75° latitude near the surface, 12 about 2.6
        self.diff_rot_rdeg = 24  # 'tabulated': degree of the least-squares Chebyshev fit of each f_n in r over the shell
        self.diff_rot_window = None  # 'tabulated': None is the shell fit above if ricb > 0 and 'sun' if ricb = 0; 'sun' is one fit
        # over the whole Sun, [-L, L] with L = aux0 + 0.01 for 'smoothed sun', even in r (table mirrored, odd coefficients dropped),
        # of degree diff_rot_sun_deg, the same for any ricb; below the first table radius (0.1 R☉ in the Mukhopadhyay et al. file)
        # each f_n is held at its value there
        self.diff_rot_sun_deg = 60  # 'tabulated' with diff_rot_window = 'sun': degree (60: ~0.01 nHz on the Edmond file)
        self.diff_rot_hold = None  # 'tabulated' with diff_rot_window = 'sun': x_h (R☉) to hold f_n at f_n(x_h) for x < x_h, blended
        # by (1 + erf((x - x_h + 0.01)/0.01))/2; None: the table as it is (held only below its first radius)

        # ----------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------- Thermal parameters
        # ----------------------------------------------------------------------------------------------------------------------

        # model_type 'Boussinesq': the background dS/dr, 'internal' r (uniform heat sources, N² ∝ r² with g = r), 'differential'
        # 1/r² (conduction between the walls, ricb > 0) or 'top hat' (the erf top hat of aux1..aux5, as 'user def')
        self.heating = 'internal'

        # Thermal boundary conditions: 0 for Dirichlet, 1 for Neumann
        # (the unknown is the entropy s = δs/c_p: 0 is fixed entropy s=0, 1 is zero diffusive entropy flux s'=0; used only if ThermaD>0)
        self.bci_thermal = 0
        self.bco_thermal = 0


        # ----------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------- Forcing parameters
        # ----------------------------------------------------------------------------------------------------------------------
        self.forcing = 0  # Uncomment this line for eigenvalue problems
        # self.forcing = 6  # Buffett2010 ICB radial velocity boundary forcing, m=1,antisymm
        # self.forcing = 7  # Longitudinal libration boundary forcing, m={0, 2}, symm, no-slip
        # self.forcing = 9  # Radial, symmetric, m=2 boundary flow forcing, bci=1.
        # self.forcing = 10 # Radial, symmetric, order m boundary flow forcing at the cmb (l=m), unit amplitude.
        # self.forcing = 11 # Random Reynolds-stress body force (convective noise), any m, symm; rsf_* below.

        # Forcing frequency (ignored if forcing == 0)
        self.forcing_frequency = 1.0  # negative is prograde

        # Forcing amplitude (boundary forcings 6, 7, 9, 10; forcing 11 ignores it: its power scales with rsf_scale)
        self.forcing_amplitude_cmb = 1.0
        self.forcing_amplitude_icb = 0.0

        # Random Reynolds-stress body force (forcing = 11, bin/reynolds_stress.py): f = rho0^-1 div(rho0 s), s a random
        # symmetric traceless stress, white in time, with covariance 4 Theta nu_t/rho0 (Landau-Lifshitz form), the
        # fluctuation-dissipation partner of kore's viscous force: in the white limit a damped mode gets kinetic energy
        # Theta_eff f_t, f_t the fraction of its damping due to nu_t (the finite l_c and rsf_angular lower it by O(1)
        # factors for modes of scale near l_c). Each column of B_forced.npz is one realisation; the mean |y|^2 over
        # the columns is the two-sided power spectral density of y (int .. dw/2pi = variance), in kore units; for m = 0
        # (a real flow, even spectrum) the kinetic energy is half the integral over all w. The energy is that of the
        # solved symm class. Needs ViscosD > 0; the entropy rows are not forced.
        self.rsf_nsamp    = 1      # realisations, solved with one LU factorisation per frequency (back-substitutions)
        self.rsf_seed     = 0      # realisation j uses numpy.random.default_rng([rsf_seed, j]), whatever rsf_nsamp
        self.rsf_theta    = 'nu'   # turbulent temperature Theta, about the energy of one eddy, in kore units (rho0 r0^5 / tau^2):
                                   #   'nu':  3 rho0 nu_t^2 l_c, eddies of size l_c consistent with nu_t = u_c l_c/3 (u_c = 3 nu_t/l_c)
                                   #   'mlt': rho0 u_c^2 l_c^3/3 with the mixing-length u_c = (alpha F_c/(10 rho0))^(1/3),
                                   #          F_c = rsf_lum/(4 pi r^2) nu_t/max(nu_t) (crude convective fraction; needs visc_pow = 0)
                                   #   a number: constant Theta (tests)
                                   # 'nu' grows as nu_t^2; 'mlt' does not depend on nu. Observed Rossby-mode amplitudes need
                                   # Theta_eff ~ (0.6-3)e34 erg, (0.3-1.6)e-11 in kore units (Sun as for rsf_lum): about
                                   # 10 (1e12 cm^2/s / nu_t)^2 x 'nu' (nu_t = 1e12 cm^2/s is ViscosD ~ 7.4e-5 there) and
                                   # 1/20 x 'mlt' (project note; a white-noise estimate: with the default l_c and rsf_angular
                                   # the same modes are forced several times less, so rsf_scale goes up as much).
                                   # The 1/3 of both laws (Theta = rho0 u_c^2 l_c^3/3) is a convention: Gaussian closure gives
                                   # 1/6 for a correlation time tau_c, 1/3 for exp(-|t|/tau_c); rsf_scale absorbs such factors.
        self.rsf_scale    = 1.0    # multiplies Theta (the response power scales with it)
        self.rsf_alpha    = 2.0    # eddy size l_c = (rsf_lc_max^-4 + (rsf_alpha H_p)^-4)^(-1/4), H_p = -p/p' (kore's pressure
        self.rsf_lc_max   = 0.25   # profile): a smooth min of rsf_alpha H_p and rsf_lc_max (H_p is infinite at r = 0)
        self.rsf_lc       = None   # a constant l_c instead (kore length units)
        self.rsf_nu_floor = None   # nu_t = ViscosD (nu/nu0 - rsf_nu_floor), clipped at 0; None: visc0 for 'smoothed sun'
                                   # with visc0 < 1 (the radiative-zone floor; approximate if visc_pow != 0), 0 otherwise
        self.rsf_lum      = None   # luminosity L/(rho_u r0^5 Omega^3) for rsf_theta = 'mlt', rho_u the density unit
                                   # (rho_c for 'model S'): Sun, rho_c 153.9 g/cm^3, r0 = 0.985 R_sun, Omega = 2 pi 456 nHz: 7.0e-7
        self.rsf_angular  = 1      # 1: variance times exp(-L l_c^2/(2 r^2)) (eddies of finite horizontal size); 0: white in angle
        self.rsf_taper    = 0.04   # the stress goes smoothly (erfc) to zero over this width below rcmb (and above ricb > 0); > 0;
                                   # its erfc width is rsf_taper/8: raise it or N if assemble.py warns of a Chebyshev tail
        self.rsf_taper_in = 0.2    # ricb = 0: the same above r = 0, > 0 (kept wide: Chebyshev nodes are sparse at r = 0)
        self.rsf_rcut     = None   # a radius (kore units): no stress below it (1e-8 there), full stress above rsf_rcut +
        self.rsf_rcut_width = 0.04 # rsf_rcut_width (erfc of width rsf_rcut_width/8). Use it, not a higher rsf_nu_floor, to
                                   # keep the forcing out of the tail of a viscosity step: clipping nu_t leaves a corner the
                                   # rows cannot resolve. In a full sphere near r = 0.7 a 0.04 edge needs N ~ 1280, a 0.08
                                   # edge N ~ 900 (assemble.py warns of a Chebyshev tail when it is under-resolved)
        self.rsf_frequencies = None  # frequency scan: forcing frequencies (kore units, negative prograde); solve.py factors
                                     # A - i(w - forcing_frequency) B for each (B.npz, written by assemble.py for forcing 11)
                                     # and appends to rsf_spectrum.dat and rsf_ke.dat instead of writing the fields
        self.rsf_split    = None   # a radius: rsf_spectrum.dat also gives the kinetic energy below it (e.g. 0.71/aux0)


        # ----------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------- Unit of time and force switches
        # ----------------------------------------------------------------------------------------------------------------------
        self.timescale = 'free fall'
        self.Gaspard   = 1  # Omega*Tau                 Coriolis force factor. Set to 1 for unit time Tau = 1/Omega
        self.Beyonce   = 0  # (N0*Tau)**2               Buoyancy force factor. Set to 1 for unit time Tau = 1/N0 = sqrt(r0/g0)
        self.ViscosD   = 0  # nu0 * Tau / r0**2         Viscous force factor. Set to 1 for viscous diffusion time scale. This is the Ekman number if Tau = 1/Omega
        self.ThermaD   = 0  # kappa0 * Tau / r0**2      Thermal diffusion factor. Set to 1 for thermal diffusion time scale
        # thermal = 1 with ThermaD = 0 (adiabatic entropy) works but admits spurious modes trapped at the inner boundary
        # whenever dS/dr != 0 there (their eigenvalues move with N, spin_doctor residσ ~1). A tiny diffusivity removes the
        # trapped modes near the shift: ThermaD = 1e-9 reproduces the ThermaD = 0 eigenvalues to 3-7e-8 (the shift is linear
        # in ThermaD, about 33-67 x ThermaD depending on the mode, from N = 480 runs of the anel10 thermal case at
        # ThermaD = 0, 1e-7, 1e-5; N = 640, prescale = 1 only confirmed them). It still leaves
        # spurious growing modes at the ICB unless dS/dr = 0 at the inner boundary.
        # Recommended, also for ricb > 0: let dS/dr -> 0 at the inner boundary (for the erf top hat, x1 >= ricb + 3*w1).
        # That removes both the trapped and the growing spurious modes. If ThermaD = 0 is kept with the ramp at the wall,
        # raise nev by 1-2, since spurious ICB modes take slots.
        # With diff_rot = 1 and thermal_wind >= 1 (heat rows r² (eq)/p in C^(1), utils.adiab_tw) and fixed-entropy, stress-free
        # walls the ThermaD -> 0 limit has a thermal boundary layer ~ (ThermaD/|λ|)^½: a tiny ThermaD is not equivalent to 0
        # below ~1e-7 at N = 128 (docs/thermad0.md §d).


        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------- Resolution
        # ----------------------------------------------------------------------------------------------------------------------
        # Number of cpus
        self.ncpus = 2

        # Chebyshev polynomial truncation level. Use function def at top or set manually. N must be even if ricb = 0.
        # With ricb = 0 only N/2 polynomials of one parity are used, so use about twice the N of a shell run.
        # Thermal runs: on the anel10 thermal setup (ricb 0.02, layer edges 0.03 wide), N = 480 under-resolves the more
        # damped modes at ThermaD 0 and 1e-9 (residσ up to 1e-4); N = 560 gives residσ <= 3.4e-8 and N = 640 <= 3.4e-11.
        # Judge resolution with spin_doctor's residσ, residens and cvfmax against a higher-N run: the P/T Chebyshev tails
        # and the ℓ-Convergence can look converged when the entropy is not.
        self.N = 120

        # Spherical harmonic truncation lmax and approx lmax/N ratio:
        self.g = 1.0
        self.lmax = int( 2*self.ncpus*( np.floor_divide( self.g*self.N, 2*self.ncpus ) ) + self.m - 1 )
        # If manually setting the max angular degree lmax, then it must be even if m is odd,
        # and lmax-m+1 should be divisible by 2*ncpus
        # self.lmax = (2*self.ncpus*1 + self.m - 1)

        # Powers of r multiplying the r̂⋅∇×∇× (u) and r̂⋅∇× (v) momentum equations (viscous case). The operator
        # labels use rx up to 5 (u) and 3 (v), so these are the smallest values keeping every operator a polynomial
        # in r, as required when ricb = 0. Larger values do not change the eigenvalues but degrade the residuals.
        self.rpower_u = 5
        self.rpower_v = 3
        # Powers of the density ρ₀ multiplying the same equations
        self.rhopower_u = 1
        self.rhopower_v = 1
        # Powers of r and ρ₀ for the pressure equation (not read by the current pipeline)
        self.rpower_pp   = 4
        self.rhopower_pp = 4


        # ----------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------- SLEPc solver options
        # ----------------------------------------------------------------------------------------------------------------------
        # Target eigenvalue tau = rtau + i*itau (shift for the shift-and-invert solver)
        self.rtau = 0.0
        self.itau = 1.0

        self.which_eigenpairs = 'TM'  # Use 'TM' for shift-and-invert
        # L/S/T & M/R/I
        # L largest, S smallest, T target
        # M magnitude, R real, I imaginary

        # Number of desired eigenvalues
        self.nev = 5

        # Number of vectors in Krylov space for solver
        # self.ncv = 100

        # Maximum iterations to converge to an eigenvector
        self.maxit = 50

        # Tolerance for solver
        self.tol = 1e-16

        # PETSc/SLEPc/MUMPS runtime options, loaded by solve.py into the PETSc options database
        # (option name without the leading '-'; use '' for flags without a value).
        # solve.py passes only the group matching the problem type (forcing == 0 or not).
        # Anything given on the command line (e.g. mpiexec -n 4 ./bin/solve.py -eps_balance twoside) takes precedence.
        self.petsc_opts = {

            # ---------------------------------------- eigenvalue problems (forcing == 0): eps_*, st_*
            'st_type'                      : 'sinvert',             # shift-and-invert around tau (use with 'TM')
            'st_pc_factor_mat_solver_type' : 'mumps',               # direct LU solve with MUMPS
            'st_mat_mumps_cntl_1'          : 0.01,                  # pivot threshold, MUMPS's default; 1e-8 cost accuracy in thermal runs,
                                                                    # 0.1-0.5 did not help either
            'st_mat_mumps_icntl_35'        : 0,                     # block low-rank (BLR) factorization off: within ~15% of BLR's time at N <= 480, never
                                                                    # less accurate; set 2 (BLR on) for large N, ~13% less memory and up to
                                                                    # ~2x faster, then check spin_doctor residuals (BLR + Ruiz degraded
                                                                    # thermal ThermaD = 0 eigenvectors 10x more than Ruiz alone)
            'st_mat_mumps_cntl_7'          : 1e-14,                 # BLR tolerance, used only with icntl_35 = 2; 1e-12 broke eigenvectors at N>=640-840
            # 'st_mat_mumps_icntl_8'       : 77,                    # MUMPS's own scaling (default 77, automatic); keep it, 0 (off) fails
                                                                    # (INFOG(1)=-9) or gives useless eigenvectors
            # 'st_mat_mumps_icntl_10'      : 3,                     # up to 3 steps of iterative refinement (untested); N = 640 thermal
                                                                    # ThermaD = 1e-9 prescale = 1 eigenvalues were up to 1.1e-6 off a
                                                                    # solve with an iteratively refined LU, while spin_doctor looked clean
            'eps_error_relative'           : '::ascii_info_detail', # print relative errors after the solve
            # 'eps_balance'                : 'twoside',             # cleaner eigenvalues for final runs, ~+50% time
            # 'st_mat_mumps_icntl_14'      : 50,                    # extra MUMPS workspace (%), only if -9 still appears
            # 'st_mat_mumps_icntl_22'      : 1,                     # out-of-core factors, cuts memory ~half
            # 'st_mat_mumps_ooc_tmpdir'    : '/nvm/scratch',        # put OOC files on a real disk, not tmpfs /tmp
            # 'st_mat_mumps_icntl_28'      : 2,                     # parallel ordering ...
            # 'st_mat_mumps_icntl_29'      : 2,                     # ... with ParMETIS

            # ---------------------------------------- forced problems (forcing > 0): ksp_*, pc_*, mat_*
            'ksp_type'                     : 'preonly',             # direct solve, no Krylov iterations
            'pc_type'                      : 'lu',                  # (default GMRES/ILU does not converge)
            'pc_factor_mat_solver_type'    : 'mumps',               # LU with MUMPS
            # 'mat_mumps_cntl_1'           : 1e-6,                  # pivot threshold, if MUMPS reports INFOG(1)=-9
            # 'mat_mumps_icntl_22'         : 1,                     # out-of-core factors, cuts memory ~half
            # 'mat_mumps_ooc_tmpdir'       : '/nvm/scratch',        # put OOC files on a real disk, not tmpfs /tmp
        }

        # Pre-scaling of the eigenvalue problem (forcing == 0 only; ignored for forced problems).
        # prescale = None (default): automatic, on (1) except for thermal = 1 with ThermaD = 0, where it is off (0).
        # Set 0 or 1 explicitly to override.
        # Measured effect on spin_doctor residuals (anelastic_log, 2026-10):
        #   - hydro only (thermal = 0): better (anel10: residσ ~1e-11 -> ~1e-12, ℓ-Convergence ~1e-13 -> ~1e-15)
        #   - thermal, ThermaD > 0: 10-1e6x better (gi1: residσ ~1e-10 -> ~1e-14); for tiny ThermaD (1e-7, 1e-9) without it
        #     the entropy gets spurious high-ℓ noise at the outer boundary that leaks into the flow
        #   - thermal, ThermaD = 0: 10-1e4x worse (the column scaling Dc damages the entropy part; row-only scaling was
        #     still worse than none), hence off in the automatic setting.
        # With prescale = 1, solve.py applies Ruiz row+column equilibration to A - tau*B (10 iterations)
        # and solves Dr*A*Dc y = lambda Dr*B*Dc y instead. Eigenvalues are unchanged; eigenvectors are mapped
        # back (x = Dc*y) before they are written, so spin_doctor and postprocessing see unscaled vectors.
        # Lowers the condition number of A - tau*B by 6-8 orders of magnitude and gave cleaner eigenvectors
        # (conductive_IC torsional-mode tests: spin_doctor resid u ~1e-7 instead of ~1e-4 for modes 1-2 at N = 976).
        # Costs 7-8 s at N = 976 on 8 ranks, before the factorization; no change in peak memory.
        # solve.py's ||Ax-kBx||/||kx|| is then printed for the scaled problem; judge accuracy with spin_doctor.
        self.prescale = None

        # Drop of negligible matrix entries, to save memory (0, the default: off). With drop_tol > 0 the Chebyshev coefficients
        # of the radial profiles are cut at max(1e-15, drop_tol) of the largest (utils.gimmedachebs, 1e-15 otherwise), entries
        # of each operator submatrix below drop_tol times its largest are removed, and after the pre-scaling entries of A and
        # max(|tau|, 1) B below drop_tol are removed. drop_tol = 1e-14 gave 35 % less memory at the same eigenvalues (< 1e-3 nHz) in
        # thermal 'smoothed sun' runs.
        self.drop_tol = 0


        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------


    def set_scales(self):
        '''
        Sets some basic dimensionless parameters based on a chosen time scale
        '''

        if self.timescale == "rotation":  # unit time Tau = 1/Omega

            self.Gaspard = 1            # Omega*Tau                 
            self.Beyonce = 0            # (N0*Tau)**2               
            self.ViscosD = self.Ek      # nu0 * Tau / r0**2         
            self.ThermaD = self.Etherm  # kappa0 * Tau / r0**2      

        elif self.timescale == "viscous":  # viscous diffusion time scale

            self.Gaspard = 1/self.Ek            # Omega*Tau
            self.Beyonce = 0                    # (N0*Tau)**2
            self.ViscosD = 1                    # nu0 * Tau / r0**2
            self.ThermaD = self.Etherm/self.Ek  # kappa0 * Tau / r0**2
        
        
        elif self.timescale == "free fall":  # unit time Tau = 1/N0 = sqrt(r0/g0)

            self.Gaspard = 0  # Omega*Tau
            self.Beyonce = 1  # (N0*Tau)**2
            self.ViscosD = 0  # nu0 * Tau / r0**2
            self.ThermaD = 0  # kappa0 * Tau / r0**2


    def Ncheb(self, Ek):
        '''
        Returns the truncation level N for the Chebyshev expansion according to the Ekman number
        Please EXPERIMENT AND ADAPT to your particular problem. N must be even.
        '''
        if Ek !=0 :
            out = int(17*Ek**-0.2)
        else:
            out = 48  #

        return max(48, out + out%2)


    def ellmax(self, ncpus, g, m, N):
        '''
        Returns the lmax given a radial truncation level N,
        ncpus, azimuthal wave number m, and an approx. N/lmax ratio g.
        '''
        #out = int(2*ncpus + m - 1)
        out = int( 2*ncpus*( np.floor_divide( g*N, 2*ncpus ) ) + m - 1 )

        return out
    
    def set_eigv_frame(self, frame):
        if frame=="inertial":
            self.rtau = self.rtau
            self.itau = self.itau + self.m
        elif frame=="rotating":
            self.rtau = self.rtau
            self.itau = self.itau
