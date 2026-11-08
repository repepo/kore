import numpy as np
#import targets as tg


class default_params():

    def __init__(self):

        # ----------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------ Physics modules       
        # ----------------------------------------------------------------------------------------------------------------------
        self.thermal = 0  # set to 1 to include the thermal (entropy) equation
        # NEW (2026-10-28): set to 1 to include the induction equation and the Lorentz force (Magnetic parameters below,
        # docs/magnetic.md)
        self.magnetic = 0
        # NEW (2026-11-06, as conductive_IC): 1 solves the momentum equation for the flow. 0 leaves the flow out, for the free
        # decay of the field (magnetic = 1: diffusion, with the walls and a conducting inner core, and the induction by a
        # differential rotation) or of the entropy (thermal = 1); eigenvalue problems only (forcing = 0)
        self.hydro = 1


        # ----------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------- Solution symmetry and inner core size
        # ----------------------------------------------------------------------------------------------------------------------
        self.m    = 0  # Azimuthal wave number
        self.symm = 1  # Equatorial symmetry, 1 for symmetric, -1 for antisymmetric
        self.ricb = 0 # Solid inner core radius 


        # ----------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------- Background model
        # ----------------------------------------------------------------------------------------------------------------------
        # Three parts, each chosen by name (bin/background.py); the rotation is under Hydrodynamic parameters (diff_rot).
        # r_out is the outer wall (r = 1) in star radii, so x = r r_out is the radius in star radii and the unit of length is
        # r_out star radii.
        self.r_out = 1.0

        # structure: ρ0, p0 (p stands for ρ0 T0) and g (radial_profiles.py); no default:
        #   'model S hydrostatic'    the Sun, for new runs: an even fit of ln ρ of Model S of degree structure_deg over the whole
        #                            Sun (whatever ricb), g from its mass integral, p from dp/dr = -ρ g. structure_deg 30 is
        #                            within 1.0 %, 0.5 %, 0.75 % of Model S in ρ, p, g, and 50 within 0.14 %, 0.06 %, 0.10 %,
        #                            where HFR m 8 stops changing (30 is 0.15 nHz off there in ω, 0.9 nHz in σ). Degrees of 40
        #                            and above need about 16 more lmax for the HFR ℓ-convergence; use 50 to match papers that
        #                            use Model S itself.
        #   'Wu 2005'                ρ = (1 - x²)² (Wu 2005, β = 2) with its g and p; r_out < 1 (e.g. 0.99) keeps ρ > 0 at the wall
        #   'Boussinesq'             ρ0 = p0 = 1, g = r: the Boussinesq equations ('uniform' until 2026-10-28)
        #   legacy, kept so that older parameter files reproduce their results:
        #   '8-term fit'             smoothed_sun.py: rough but hydrostatic; ρ is up to 48 % off near the surface, which moves
        #                            n = 1 Rossby, HFR and columnar modes by 1.5-8 nHz (docs/background.md)
        #   'old model S prototype'  Chebyshev fits of ln ρ, ln p of bin/model_s.at of degree solar_density_deg over
        #                            solar_density_window = [x_lo, x_hi]; None: the shell [ricb r_out, r_out] widened by 0.01
        #                            below and 0.005 above if ricb > 0, 'sun' if ricb = 0; 'sun': one fit over the whole Sun, even
        #                            in r (Model S mirrored to x < 0, odd coefficients dropped), of degree solar_density_sun_deg
        #                            on [-(r_out + 0.005), r_out + 0.005], the same for any ricb (ln rho to 1.7e-3, mostly at the
        #                            base of the convection zone, where the shell fit reaches 2e-4); g of the 8-term fit, so not
        #                            hydrostatic. To be removed, with solar_density_*, once 'model S hydrostatic' is validated.
        self.structure = None
        self.structure_deg = 30
        self.solar_density_deg = 30
        self.solar_density_window = None
        self.solar_density_sun_deg = 60

        # stratification: dS/dr, S = δs/c_p, for r >= 0 (kore makes it odd in r); needed with thermal = 1, no default:
        #   'top hat'         erf top hat between x1 and x2 (in r), aux1..aux5 = x1, w1, x2, w2, A (utils.erf_top_hat)
        #   'odd top hat'     the same made odd in r (utils.erf_top_hat_odd): with x1 = 0 it rises as A erf(r/w1) from the
        #                     centre, for ricb = 0 (Model S's dS/dr to 10 % rms: aux1..aux5 = 0, 0.03, 0.68, 0.06, 2.6)
        #   'solar_rz'        the radiative interior of Mukhopadhyay, Bekki, Zhu & Gizon 2025 (arXiv 2512.12691, their eq. 10;
        #                     docs/background.md): dS/dr = solar_rz_dSdr/4 [1 - erf((x - x1)/d)] [1 - erf((x - x2)/d)]
        #                     erf(x/w0), [x1, x2, d] = solar_rz_steps, w0 = solar_rz_core; erf(x/w0) makes it odd and smooth
        #                     at r = 0 (no effect above x ~ 3 w0). solar_rz_dSdr = (∂s0/∂r)_RZ r_out R☉/c_p: 2.615 for their
        #                     1.32e-2 erg/g/K/cm with c_p = 3.46e8 and r_out = 0.985. solar_rz_delta_cz, solar_rz_delta_sf add
        #                     a convection-zone superadiabaticity δ (Bekki 2024, eq. 2): dS/dr += δ(x) d(ln p0)/dr
        #                     (1 + erf((|x| - x_OS)/d_OS))/2, δ(x) = δ_cz + (δ_sf - δ_cz) exp(-((|x| - 0.985)/0.015)^2),
        #                     [x_OS, d_OS] = solar_rz_os, δ < 0 being subadiabatic (Bekki's model 3-sub: δ_cz = -5e-7,
        #                     δ_sf = 1e-3)
        #   'linear'          r: with 'Boussinesq', N² = Beyonce r² (uniform heat sources; Rieutord 2006, Mirouh et al. 2016)
        #   'inverse square'  1/r²: conduction between the walls (needs ricb > 0)
        #   'user'            stratification_profile, a numpy expression of r, x and p = stratification_param in units of 1/L
        #                     (the names and the x of diff_rot_profiles), e.g. '2.5*exp(-((r - 0.75)/0.06)**2)'
        self.stratification = None
        self.stratification_profile = ''
        self.stratification_param = 0.0
        self.solar_rz_dSdr  = 2.615
        self.solar_rz_steps = [0.61, 0.68, 0.04]
        self.solar_rz_core  = 0.05
        self.solar_rz_os    = [0.70, 0.04]
        self.solar_rz_delta_cz = 0.0
        self.solar_rz_delta_sf = 0.0

        # diffusivity: ν/ViscosD and κ/ThermaD:
        #   'uniform'   1
        #   'step'      visc0, therm0 below and 1 above the upper edge of the erf box at x2 = aux3 of width aux4 (needed also
        #               when thermal = 0 unless visc0 = therm0 = 1), so ViscosD and ThermaD are the convection-zone values
        #   'solar_rz'  visc0, therm0 below and 1 above as (1 + erf((x - x_OS)/d_OS))/2, [x_OS, d_OS] = solar_rz_os
        #               (Mukhopadhyay et al. 2025 eqs. 8-9; their ν_RZ/ν_CZ = 0.01, κ_RZ/κ_CZ = 0.04)
        #   'user'      diffusivity_profiles = {'nu': '...', 'kappa': '...'}, numpy expressions of r, x and p = diffusivity_param,
        #               evaluated at |r| (even in a full sphere)
        # Solar: visc0 ~ 1e-11, therm0 ~ 1e-6 to 1e-5 (ν ~ 10, κ_rad ~ 1e5 to 1e7 cm²/s against ν_t ~ κ_t ~ 1e12), so in practice
        # they are numerical floors; see sun/solar-profile.md. visc_pow and therm_pow multiply ν and κ of any type by
        # (ρ(x2)/ρ(r))**pow, x2 = aux3: 0 keeps the kinematic diffusivities constant (recommended), 1 the dynamic ones (ρν, ρκ),
        # ViscosD and ThermaD being then the values at x2. rvisc and hvisc are not used.
        self.diffusivity = 'uniform'
        self.diffusivity_profiles = {}
        self.diffusivity_param = 0.0
        self.visc0     = 1.0
        self.therm0    = 1.0
        self.visc_pow  = 0.0
        self.therm_pow = 0.0
        self.rvisc     = 0.0
        self.hvisc     = 0.0

        # the erf top hat and step ('top hat', 'odd top hat', 'step'): aux1..aux5 = x1, w1, x2, w2, A
        self.aux1   = 0
        self.aux2   = 0
        self.aux3   = 0
        self.aux4   = 0
        self.aux5   = 0

        # profile smoothing power
        self.smopo = 0

        # ----------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------- Auxiliary variables
        # ----------------------------------------------------------------------------------------------------------------------
        self.Ek     = 0
        self.Etherm = 0
        # NEW (2026-10-28): magnetic inputs of set_scales, with the rotation rate Ω as Ek: the Lehnert number
        # Le = B0/(Ω r0 sqrt(ρ0 μ0)) and the magnetic Ekman number Em = η0/(Ω r0²), so Em = Ek/Pm and the Elsasser number
        # is Λ = Le²/Em
        self.Le     = 0
        self.Em     = 0


        # ----------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------- Hydrodynamic parameters
        # ----------------------------------------------------------------------------------------------------------------------
        # Inner core mechanical boundary conditions
        # Use 0 for stress-free, 1 for no-slip or forced boundary flow. Ignored if ricb = 0
        self.bci = 0

        # CMB mechanical boundary conditions
        # Use 0 for stress-free, 1 for no-slip or forced boundary flow
        self.bco = 0

        # NEW (2026-11-06, from conductive_IC): a drag at a stress-free outer wall (bco = 0, ViscosD > 0), the form drag of the
        # flow over the wall's topography (docs/drag.md, bin/topo.py): the wall exerts on the fluid the tangential stress
        # -K (u_t - v_m), v_m the velocity of the mantle (rotdyn = 1; 0 otherwise), between stress-free (K = 0) and no-slip
        # (K -> inf). K in units of rho_w Omega r0, rho_w the fluid's density at the wall. kappa_cmb: a constant K, the same in
        # every direction (kappa_cmb = kappa/(rho_w Omega r0) for a drag of stress kappa u: the Earth's topographic drag of
        # Rekier et al. calibrated at the diurnal frequency, kappa = 11.3 kg m^-2 s^-1, is 4.5e-6). background.check_drag stops
        # on what it does not cover.
        self.kappa_cmb = 0.0
        # topo_drag = 'rekier' (None: off): instead of kappa_cmb, the latitude-dependent tensor diag(kappa_tt, kappa_pp)(theta) of
        # the internal waves the flow radiates over the topography (Rekier et al. 2025, arXiv:2507.01671, eqs. 20, 21, B1, B2,
        # with the factor 4 pi^2 that the printed B1 lacks). Inputs: the rms height and the roll-off wavelength of the
        # topography in units of r0 (the outer radius of the fluid), and its Hurst exponent (> 1/2): the Earth's CMB (1.51 km,
        # 1500 km at r0 = 3480 km) is topo_hrms = 4.35e-4, topo_lambda0 = 0.431, topo_H = 0.8. N, the buoyancy frequency at the
        # wall, is that of the stratification (0 without buoyancy or where it is unstable) unless topo_N (N/Omega) is set; the
        # frequency of the flow is |itau|/Gaspard, the solver's target (forced problems: |forcing_frequency|/Gaspard), unless
        # topo_omega (w0/Omega) is set.
        self.topo_drag    = None
        self.topo_hrms    = 0.0
        self.topo_lambda0 = 0.431
        self.topo_H       = 0.8
        self.topo_N       = None
        self.topo_omega   = None
        self.topo_K       = ''   # test hook: a numpy expression of x = cos(theta) giving (kappa_tt, kappa_pp) instead of B1

        # Differential rotation parameters. Set to 0 for solid body rotation. Ω_ref is the frame rotation rate (Gaspard in units of
        # the time scale) and ΔΩ is relative to it, so any timescale works: U0 = Gaspard ΔΩ r sinθ e_φ. User guide: docs/background.md (Rotation).
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
        #   x is the fractional solar radius, r r_out with a solar structure (r otherwise). Needs N ~ 400 with ricb = 0.
        # self.diff_rot_type = 'solar_P4'  # Ω/2π = Ω_RZ + [Ω_CZ(θ) - Ω_RZ] (1 + erf((x - rtc)/dtc))/2, Ω_CZ(θ) = c0 + c2 cos²θ
        #   + c4 cos⁴θ (Mukhopadhyay et al. 2026; diff_rot_solar below); ΔΩ = Ω/Ω_ref - 1, Ω_ref/2π = diff_rot_frame nHz: P0, P2, P4
        # self.diff_rot_type = 'tabulated'  # Ω(r, θ) from diff_rot_file (.npz with ome[θ, r] in nHz, r in R☉, theta in rad, as the
        #   file data/diff_rot.npz of the Mukhopadhyay et al. 2025 data set; tests/diff_rot/ls_to_tabulated.py converts the Larson & Schou
        #   2018 table), projected on P0, P2, ..., P_diff_rot_nmax at each table radius and fitted in r; ΔΩ = Ω/Ω_ref - 1
        # self.diff_rot_type = 'user'  # f_n given as numpy expressions of r, x and p = diff_rot_param in diff_rot_profiles
        # self.diff_rot_type = 'ferraro'  # NEW (2026-10-30): ΔΩ = F(ψ) from the poloidal background field (magnetic = 1), ψ its flux
        #   function and F = diff_rot_F: a steady background with B0 (Ferraro); even degrees up to diff_rot_nmax. The reverse
        #   of B0 = 'ferraro' (the field from a given rotation); the two cannot be combined
        #   At ricb = 0, 'Y20' and 'conical' are not regular (their P2 term does not vanish at r = 0), and
        #   'Y20-wall-bounded' and 'shellular' (unless σ is even and > 0) are not smooth there (a kink): use ricb > 0.
        #   submatrices.py warns about them and about any other profile that is not regular at the centre (NEW 2026-11-04;
        #   docs/numerical-method.md, Regularity at the centre).
        self.diff_rot_ampl = 1.0  # the factor of ΔΩ, for every type (1 for 'solar', 'solar_P4', 'tabulated': the profile itself)
        self.diff_rot_F = 'psi'  # NEW (2026-10-30): 'ferraro': F, a numpy expression of psi (the flux function of B0) and p
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
        # that Ω(r,θ) implies, to first order in ΔΩ (docs/differential_rotation_details.md §2.4); thermal_wind = 2 uses the exact (1+ΔΩ)² balance,
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
        # over the whole Sun, [-L, L] with L = r_out + 0.01 for the solar structures, even in r (table mirrored, odd coefficients dropped),
        # of degree diff_rot_sun_deg, the same for any ricb; below the first table radius (0.1 R☉ in the Mukhopadhyay et al. file)
        # each f_n is held at its value there
        self.diff_rot_sun_deg = 60  # 'tabulated' with diff_rot_window = 'sun': degree (60: ~0.01 nHz on the Mukhopadhyay et al. 2025 profile)

        # NEW (2026-11-05): centrifugal deformation of the body (bin/centrifugal.py; user guide: docs/background.md, Centrifugal
        # deformation; equations: docs/centrifugal-equations.md). The rotating body is flattened: its level surfaces are
        # r (1 - (2/3) f(r) P2(cos θ)) at first order in their flattening f. kore then solves in coordinates that follow them: the
        # walls are level surfaces, every background profile is a function of their mean radius r, and the inertia, the Coriolis
        # force and the time derivative of s gain P2 factors (couplings |Δl| <= 3). FIX (2026-11-06, phase 2): exact at first order
        # in f, also with a field (phase 1b), the viscous force (no-slip and stress-free walls), the thermal and the magnetic
        # diffusion and a differential rotation, which is given on the level surfaces (phase 2); with thermal_wind = 1 or 2 the
        # entropy that balances it is that of the sphere. It stops with forcing != 0, bco or bci = 2 with viscosity, a conducting
        # inner core with mu_i2o != 1, and a viscous full sphere with m != 0 and rpower_v < 4.
        self.centrifugal = 0
        self.centrifugal_shape = 'clairaut'  # f(r) from Clairaut's equation with the structure's g (its self-gravity), of amplitude
        #   set by centrifugal_q at the surface, or by centrifugal_f at the outer wall if that is set
        # self.centrifugal_shape = 'uniform'  # f = centrifugal_f everywhere (a homogeneous body, as the Maclaurin spheroid)
        # self.centrifugal_shape = 'user'  # f = centrifugal_profile, a numpy expression of r, x and p = centrifugal_param
        # self.centrifugal_shape = 'spheroid'  # NEW (2026-11-06): f = centrifugal_f everywhere, kept exactly at any f < 1 (the
        #   stretched sphere of 'uniform' without the first-order approximation): a homogeneous fluid ('Boussinesq', uniform
        #   diffusivities, centrifugal_q = 0), with or without a field, no differential rotation (docs/centrifugal-equations.md)
        self.centrifugal_q = 0.0  # q = Ω² R³/(G M), centrifugal over gravitational acceleration at the equator of the surface
        #   x = 1 (the outer wall for 'Boussinesq'): Earth 3.5e-3, Jupiter 0.089, Saturn 0.155. The g of 'model S hydrostatic'
        #   ends at the outer wall: 'clairaut' applies its surface condition there, with q r_out³ (the mass above the wall
        #   neglected). It also gives the buoyancy the effective gravity g - (2/3) Ω² r (any shape).
        self.centrifugal_f = None  # flattening at the outer wall, (equatorial - polar radius)/equatorial radius: the value for
        #   'uniform' and 'spheroid', and for 'clairaut' the scale of its shape in place of centrifugal_q (e.g. the Earth's
        #   CMB, about 2.5e-3)
        self.centrifugal_profile = ''  # 'user': f(r) as a numpy expression of r (kore radius), x (r r_out with a solar structure,
        #   r otherwise) and p, with the functions of diff_rot_profiles, at |r|; no '#' (it is written to params.dat as a comment)
        self.centrifugal_param = 0.0  # 'user': p

        # ----------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------- Thermal parameters
        # ----------------------------------------------------------------------------------------------------------------------

        # Thermal boundary conditions: 0 for Dirichlet, 1 for Neumann
        # (the unknown is the entropy s = δs/c_p: 0 is fixed entropy s=0, 1 is zero diffusive entropy flux s'=0; used only if ThermaD>0)
        self.bci_thermal = 0
        self.bco_thermal = 0


        # ----------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------- Magnetic parameters
        # ----------------------------------------------------------------------------------------------------------------------
        # NEW (2026-10-28): used only with magnetic = 1. Ported by name from the conductive_IC branch; user guide docs/magnetic.md.
        # b = ∇×∇×(f r) + ∇×(g r), with the induction equation ∂b/∂t = ∇×(u×B0) + MagnetD ∇²b and the Lorentz force per unit
        # mass Hendrik ρ0⁻¹[(∇×b)×B0 + (∇×B0)×b] (Hendrik, MagnetD: set_scales from Le, Em).
        # Shell or full sphere, any structure, with or without a differential rotation; an insulating mantle or a thin conducting
        # wall (NEW 2026-11-06), eigenvalue problems, a uniform magnetic diffusivity; ViscosD = 0 only in a full sphere.
        # background.check_magnetic() stops on anything else (FIX 2026-11-03: this header still described the first step of
        # the port).

        # Imposed background field B0 = cnorm ∇×∇×(h(r) Y_l0 r), axisymmetric and poloidal (background_field.py)
        self.B0 = 'axial'          # axial, uniform field along the spin axis (h = r/2, l = 1)
        # self.B0 = None           # NEW (2026-10-28): no poloidal field (with a toroidal one, B0_tor below)
        # self.B0 = 'G21 dipole'   # NEW (2026-10-28): h = r/6 - r³/10 (Gerick et al. 2021), l = 1
        # self.B0 = 'Luo_S1'       # NEW (2026-10-28): h = (5 - 3r²) r (Luo & Jackson 2022), l = 1
        # self.B0 = 'Luo_S2'       # NEW (2026-10-28): h = (157 - 296r² + 143r⁴) r² (Luo & Jackson 2022), l = 2, quadrupole
        # self.B0 = 'FDM'          # free poloidal decay mode of the shell (Zhang & Fearn 1994, 1995; Schmitt 2012), degree B0_l
        # self.B0 = 'dipole'       # NEW (2026-11-07, from conductive_IC): a dipole at the centre, h = 1/(2r²), l = 1, current-free
        #                          # in the fluid; singular at r = 0, so it needs ricb > 0 (its source is in the inner core)
        # self.B0 = 'confined'     # NEW (2026-10-30): a field confined to B0_conf_r1 <= r <= B0_conf_r2, degree B0_l (1 or 2),
        #                          # h = r^l (r2² - r²)^p (r1 = 0) or r^l (r - r1)^p (r2 - r)^p: zero outside, with h and its first p - 1
        #                          # derivatives zero at the edges, so no surface currents (p = B0_conf_p >= 3: h''' enters the rows).
        #                          # Put it where ΔΩ is uniform to have a steady background with a differential rotation (Ferraro)
        self.beta = 3.0            # 'FDM': first guess of its radial wavenumber β (the root of background_field.findbeta)
        self.B0_l = 1              # 'FDM', 'confined': its degree l
        self.B0_conf_r1 = 0.0      # NEW (2026-10-30): 'confined': the inner edge (0: down to the centre; >= ricb with an inner core)
        self.B0_conf_r2 = 0.6      #                   the outer edge (<= 1)
        # self.B0 = 'general'      # NEW (2026-10-30): any poloidal field Σ_n ∇×∇×(h_n Y_n0 r), B0_profiles = {n: 'numpy expression of r, p'}
        #                          # (degrees of one parity; h_n ~ r^n at r = 0 in a full sphere; p = B0_param); built by mag_algebra.py
        self.B0_profiles = {}
        self.B0_tor_profiles = {}  # NEW (2026-10-30): B0_tor = 'general': {n: 'expression'} for k_n of Σ_n ∇×(k_n Y_n0 r), p = B0_tor_param
        self.B0_param = 0.0
        # self.B0 = 'ferraro'      # NEW (2026-10-30): a poloidal field whose flux function is ψ = G(ΔΩ) (needs diff_rot = 1): its field
        #                          # lines lie on the surfaces of constant rotation (Ferraro), so it is a steady background with U0.
        #                          # G = B0_ferraro_G, a numpy expression of w = ΔΩ and p = B0_param; odd degrees up to B0_ferraro_nmax.
        #                          # G should vanish on the values ΔΩ takes on the axis (ψ = 0 there) and, for a field confined
        #                          # inside, at the outer wall: background.ferraro_report prints how well that holds
        self.B0_ferraro_G = 'w'
        self.B0_ferraro_nmax = 15
        # NEW (2026-10-30): B0_wound = 1 adds to B0 the steady toroidal field that the differential rotation winds from the poloidal
        # B0 (needs a poloidal B0 and diff_rot = 1): diffusion balances the winding, B0_wound_eta MagnetD (∇² - 1/s²)B_φ = -Gaspard s B_p⋅∇ΔΩ,
        # solved degree by degree (background_field.wound_profiles) and added to any B0_tor. Its size is set by the physics, about
        # Gaspard ΔΩ/(B0_wound_eta MagnetD) times B_p; cnorm still normalises the poloidal field alone. B0_wound_eta = η_background/η:
        # the diffusivity that balances the winding in units of MagnetD (e.g. a turbulent one, > 1, gives a weaker wound field).
        # Ignored (docs/magnetic.md): the ohmic decay of B_p and the Lorentz force of B0 on the background flow.
        self.B0_wound = 0
        self.B0_wound_eta = 1.0
        # NEW (2026-11-03): with diff_rot = 1 and a poloidal B0 (and no B0_wound), what to do when B0⋅∇ΔΩ ≠ 0, i.e. when the two
        # are not a steady background together (background.ferraro_mismatch, above 1e-7): 'warn' prints the mismatch and goes on
        # (the linear problem then assumes the background is maintained by something it leaves out), 'refuse' stops
        self.ferraro_policy = 'warn'
        self.B0_conf_p  = 12       #                   the order of the zeros of h at the edges (12: Chebyshev series to 1e-14 by N ~ 192)
        # NEW (2026-10-30): where the couplings of the Lorentz force and of the induction come from: 'closed' (the closed forms of
        # operators.py, degree 1 and 2 only) or 'algebra' (mag_algebra.py, built from the recurrence algebra for any degrees; the
        # types 'general' and 'ferraro' always use it). Both give the same eigenvalues to round-off for the closed-form types.
        self.mag_couplings = 'closed'

        # NEW (2026-10-28): what the full-sphere fields ('G21 dipole', 'Luo_S1', 'Luo_S2') do with an inner core (ricb > 0), as
        # conductive_IC. Such a field has currents below ricb, and nothing keeps them steady in a solid inner core of finite
        # conductivity (no flow there, hence no EMF: an axisymmetric current decays on the core's magnetic diffusion time).
        #   'potential' (default): removes them. background_field.B0_beta_ic adds beta r^-(l+1) to h, beta =
        #       ricb^(l+1) (ricb h' - l h)/(2l + 1) at the ICB, which carries no current and keeps the CMB matching: the currents
        #       in the fluid stay the same and B0 continues into the core as the potential field h(ricb) (r/ricb)^l. The steady
        #       field for an insulating core and for any finite conductivity. (To keep exactly the fluid currents of the
        #       full-sphere field, give cnorm as a number rather than 'mag_energy'.)
        #   'frozen currents': h unchanged, so the currents below ricb stay in the core (FIX 2026-11-06: renamed as in
        #       conductive_IC, was 'full sphere', which now stops with this name). Self-consistent only for a perfectly
        #       conducting core, or as a snapshot of currents frozen in it, when its free-decay time (~0.006 Lu sigma_i2o
        #       Alfven times, Lu = Le/Em, for l = 2 at ricb = 0.35; about 3000 years for l = 2 in the Earth) is much longer
        #       than the periods sought. It does not tend to 'potential' as sigma_i2o decreases.
        # No effect for 'axial', 'FDM' and other fields that already match a potential field at the ICB, nor for 'dipole' (its
        # source is in the core), nor with ricb = 0.
        self.B0_ic = 'potential'
        # self.B0_ic = 'frozen currents'

        # Normalisation of B0 (cnorm):
        self.cnorm = 1             # a number: cnorm itself
        # self.cnorm = 'rms_cmb'     # the rms of the radial B0 over the CMB is 1
        # self.cnorm = 'mag_energy'  # ∫ B0² dV over the fluid shell is 1 (Luo & Jackson 2022)
        # self.cnorm = 'Schmitt2012' # ∫ B0² dV over the fluid shell is 4

        # relative cut of the Chebyshev series of r^p h^(k) (coefficients below tol_B0 times the largest of each series are
        # dropped): it sets the band width of the magnetic blocks and the fill-in of the LU factors (as conductive_IC)
        self.tol_B0 = 1e-8

        # NEW (2026-10-28): a toroidal (azimuthal) background field B0_tor = cnorm_tor ∇×(k(r) Y_l0 r), B_φ = -cnorm_tor k ∂θY_l0
        # (for l = 1: B_φ = cnorm_tor k sinθ). Alone (B0 = None) or added to the poloidal B0 when both have the same equatorial
        # symmetry (a toroidal field of degree l has (-1)^(l+1), a poloidal one (-1)^l: the dipole and B0_tor_l = 1 cannot be combined)
        self.B0_tor = None
        # self.B0_tor = 'Malkus'   # k = r: B0 = cnorm_tor s φ̂ (Malkus 1967), current J0 = 2 cnorm_tor ẑ; nonzero at the walls (surface
        #                          # currents there, whose push on a fluid slipping along the wall (bco or bci 0 or 2) the stress rows
        #                          # hold; an inviscid fluid (ViscosD = 0) with an insulator beyond stops; see docs/magnetic.md)
        # self.B0_tor = 'wall'     # k = r (r - ricb)(1 - r)/(1 - ricb)²: vanishes at both walls
        # self.B0_tor = 'user'     # k = B0_tor_profile, a numpy expression of r and p = B0_tor_param (sin ... tanh, exp, log, sqrt,
        #                          # where, erf, pi), e.g. 'r*(1 - r)**2' (smooth: its first two derivatives enter the rows)
        self.B0_tor_profile = ''
        self.B0_tor_param = 0.0
        self.B0_tor_l = 1           # its degree
        self.cnorm_tor = 1          # a number, 'mag_energy' (∫ B0_tor² dV = 1 over the shell) or 'Schmitt2012' (= 4)

        # Magnetic boundary conditions at the inner boundary and at the outer boundary
        self.innercore = 'insulator'
        # self.innercore = 'conducting, Chebys'  # NEW (2026-10-28): a conducting solid inner core; its field is part of the
        #                                        # unknowns (fic, gic); NEW 2026-11-07: bci = 0 or 1 (a fluid slipping along
        #                                        # it carries its u×B0 into the core, assemble.bc_b_cic)
        # self.innercore = 'conducting, Bessel'  # NEW (2026-11-06, from conductive_IC): the same core in forced problems only
        #                                        # (forcing != 0): its field, j_l(k r) at the forcing frequency, is solved
        #                                        # exactly and leaves only the rows of the inner wall (no fic, gic, N_cic)
        # self.innercore = 'TWA'                 # NEW (2026-11-06, from conductive_IC): a thin conducting layer on an
        #                                        # insulating inner core (c_icb, c1_icb, muw_icb below)
        self.mantle    = 'insulator'
        # self.mantle    = 'TWA'                 # NEW (2026-11-06, from conductive_IC): a thin conducting layer at the base
        #                                        # of an insulating mantle (c_cmb, c1_cmb, muw_cmb, B0_cmb below)
        # self.mantle    = 'resolved'            # NEW (2026-11-06, from conductive_IC): a conducting layer from the outer
        #                                        # wall to r_surface, its field solved for (r_surface, N_mantle, sigma_mantle)

        # NEW (2026-11-06, from conductive_IC): the thin conducting layer ('TWA', the thin-wall approximation of Roberts,
        # Glatzmaier & Clune 2010; docs/magnetic.md, Thin conducting layer): a layer of thickness h, conductivity sigma_w and
        # permeability mu_w between the fluid and an insulator, described by c = h mu_w/(r mu_f) and c1 = h sigma_w/(r sigma_f)
        # at the wall's radius r (assemble.bc_b_spherical). With c = c1 = 0 and mu = 1 the layer is an insulator.
        self.c_cmb   = 1e-5    # c of a thin conducting layer at the outer wall (mantle = 'TWA')
        self.c1_cmb  = 1e-5    # c1, the layer's conductance relative to the fluid's
        # the layer's permeability relative to vacuum, p = mu_w/mu0. With c it sets the thickness h/r = c mu/p and the O(h)
        # terms of the poloidal condition. p = 1 (default, a non-magnetic layer as at the Earth's CMB): these cancel the c term
        # at leading order, and only c1 acts. np.inf: without them (h -> 0 with h mu_w fixed, Guervilly et al. 2013), right
        # for mu_w >> mu0
        self.muw_cmb = 1.0
        self.c_icb   = 0.0     # the same for a thin layer on the inner core (innercore = 'TWA')
        self.c1_icb  = 0.0
        self.muw_icb = 1.0
        self.mu      = 1.0     # permeability of the fluid relative to vacuum: it enters the thin-wall conditions only
        # How B0 meets a thin layer at the outer wall (mantle = 'TWA'). A steady B0 drives no current in the layer (c1 does
        # not act on it), but c does: B0 has to satisfy (1/mu + l c)(r h)' + (l + e) h = 0 there, e = c mu l ((l+1)/p - 1)/p,
        # which for p = 1 and mu = 1 is the insulating condition (B0 unchanged).
        #   'wall' (default): background_field adds alpha r^l to h, alpha set by that condition, for the fields of internal
        #       currents ('G21 dipole', 'Luo_S1', 'Luo_S2', 'FDM', 'dipole'); the added term carries no current and keeps the
        #       matching at the inner boundary. The other fields are used as given ('axial' is imposed from outside the core).
        #   'plain': h unchanged; B0 then meets the layer's condition only for p = 1 and mu = 1, or c = 0 and mu = 1.
        self.B0_cmb = 'wall'
        # self.B0_cmb = 'plain'

        # NEW (2026-11-06, from conductive_IC, after PR #17 by A. Nuyt): the resolved conducting mantle (mantle = 'resolved'), a
        # layer from the outer wall (r = 1) to r_surface, insulating beyond, at rest (with a differential rotation it turns with
        # ΔΩ(1), uniform in latitude there; with forced_walls 'deforming' it deforms with a forcing that moves the outer wall),
        # whose field is expanded in N_mantle Chebyshev polynomials on [1, r_surface] (docs/magnetic.md, Resolved conducting
        # mantle). Unlike 'TWA' it holds for any thickness and frequency, and
        # for any smooth conductivity profile that N_mantle resolves (background.check_mantle stops otherwise). Needs mu = 1.
        self.r_surface = 1.83   # outer radius of the layer; 1.83 is the whole mantle of the Earth
        self.N_mantle  = 64     # Chebyshev polynomials in the layer
        # the layer's conductivity relative to the fluid's: a number (a uniform layer), or a numpy expression of r and
        # p = sigma_mantle_param with the names of the 'user' profiles, e.g. '1e-2*exp(-p*(r - 1))'
        self.sigma_mantle = 1.0
        self.sigma_mantle_param = 0.0

        # NEW (2026-10-28): innercore = 'conducting, Chebys' (N_cic) and, NEW 2026-11-06, 'conducting, Bessel' (mu_i2o, sigma_i2o)
        # (the inner-core field has the degrees of the fluid's, up to lmax: conductive_IC's lmax_cic is not ported, its ICB
        # matching couples the two degree by degree)
        self.N_cic     = 64    # Chebyshev polynomials of the inner-core field on [-ricb, ricb], N_cic/2 of one parity per l
                               # (conductive_IC: 48 gave broken eigenvectors at sigma_i2o = 1e-2)
        self.mu_i2o    = 1.0   # magnetic permeability ratio solid inner core / fluid
        self.sigma_i2o = 1.0   # electrical conductivity ratio solid inner core / fluid (magnetic diffusivity MagnetD/(mu_i2o sigma_i2o) inside)


        # ----------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------ Rotating solids
        # ----------------------------------------------------------------------------------------------------------------------
        # NEW (2026-11-06, from conductive_IC): rotdyn = 1 lets the mantle and the inner core turn as rigid bodies about the
        # rotation axis under the torques of the fluid, viscous and magnetic, and under their gravitational coupling, with
        # three more unknowns: their rotation rates omega_M, omega_IC relative to the frame and the angle phi between them
        # (docs/rotating-solids.md). The fluid follows them through the no-slip walls. Axisymmetric, equatorially symmetric
        # eigenvalue problems (m = 0, symm = 1, forcing = 0) with an inner core, ViscosD > 0, no-slip walls (bci = bco = 1),
        # a rotating frame (Gaspard > 0), no differential rotation and a spherical body; a toroidal B0 must vanish at the walls
        # (background.check_rotdyn stops otherwise).
        self.rotdyn = 0
        # axial moments of inertia of the mantle and of the inner core, in units of rho r0^5 (rho the density unit of the
        # structure: rho0 = 1 for 'Boussinesq'). The defaults are the Earth's (r0 the radius of the core).
        self.MoIZ_M  = 11.16
        self.MoIZ_IC = 9.2e-3
        # gravitational coupling between the inner core and the mantle, the torque K phi divided by rho r0^5 Omega^2
        # (K = gTorque Gaspard^2 in the units of the run)
        self.gTorque = 5e-8
        # Omega tau_IC, the viscous relaxation time of the inner core towards gravitational alignment times the rotation rate:
        # 2.3e3 is tau_IC = 1 year for the Earth; np.inf: no relaxation
        self.OmgtauIC = 2.3e3
        # 0 switches off the viscous (vtrq) or the magnetic (mtrq) torques on the solids (the fluid still feels its walls)
        self.vtrq = 1
        self.mtrq = 1


        # ----------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------- Forcing parameters
        # ----------------------------------------------------------------------------------------------------------------------
        self.forcing = 0  # Uncomment this line for eigenvalue problems
        # self.forcing = 6  # Buffett2010 ICB radial velocity boundary forcing, m=1,antisymm, ricb > 0
        # self.forcing = 7  # Longitudinal libration boundary forcing, m={0, 2}, symm, no-slip (NEW 2026-11-07: m = 0 also with bci = 0)
        # self.forcing = 9  # Radial, symmetric, m=2 boundary flow forcing, bci=1.
        # self.forcing = 10 # Radial, symmetric, order m boundary flow forcing at the cmb (l=m), unit amplitude.
        # self.forcing = 11 # Random Reynolds-stress body force (convective noise), any m, symm; rsf_* below.
        # self.forcing = 12 # Steady rotation of the walls about z (spherical Couette flow), m=0, symm, forcing_frequency = 0; a turning wall is no-slip (docs/couette-flow.md)
        # self.forcing = 13 # Entropy forcing_amplitude_icb P2(cos θ) at the inner wall (its gradient with bci_thermal = 1), m=0, symm.
        # self.forcing = 14 # Tide: radial velocity forcing_amplitude Y_2^m at the walls, (m, symm) = (0, 1), (1, -1), (2, 1); bci = bco = 1 (docs/tidal-forcing.md)

        # Forcing frequency (ignored if forcing == 0)
        self.forcing_frequency = 1.0  # negative is prograde

        # Forcing amplitude (boundary forcings 6, 7, 9, 10, 13, 14; for 12 the rotation rates of the walls, in units of 1/tau;
        # forcing 11 ignores it: its power scales with rsf_scale)
        self.forcing_amplitude_cmb = 1.0
        self.forcing_amplitude_icb = 0.0

        # NEW (2026-11-06): with a field (magnetic = 1), what a radial boundary flow (forcings 6, 7 with m != 0, 9, 10, 14)
        # stands for at a wall with a conducting solid behind it; the wall stays at the sphere either way (docs/magnetic.md)
        self.forced_walls = None           # the picture of the forcing: 'rigid' for forcings 6 and 7, 'deforming' for 9, 10, 14
        # self.forced_walls = 'rigid'      # the solid is rigid and at rest, only its shape moves past the sphere (a librating
                                           # triaxial mantle, the flattened inner core of forcing 6): the flow passes through
        # self.forced_walls = 'deforming'  # the solid deforms with the boundary (a tide): its surface moves with the fluid,
                                           # and inside a resolved mantle or a conducting inner core its motion induces field
        # Either way a wall that the forcing moves needs one permeability on both sides where B0 reaches it: mu = 1, mu_i2o = 1
        # with a conducting inner core, and a thin layer's muw = 1 where the poloidal B0 reaches the wall if c != 0 or the
        # layer conducts at rest ('rigid'), and where the toroidal B0 does if c1 != 0 (background.check_forced_magnetic)

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
        self.rsf_nu_floor = None   # nu_t = ViscosD (nu/nu0 - rsf_nu_floor), clipped at 0; None: visc0 for diffusivity
                                   # 'step' or 'solar_rz' with visc0 < 1 (the radiative-zone floor; approximate if visc_pow != 0), 0 otherwise
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
        self.rsf_split    = None   # a radius: rsf_spectrum.dat also gives the kinetic energy below it (e.g. 0.71/r_out)


        # ----------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------- Unit of time and force switches
        # ----------------------------------------------------------------------------------------------------------------------
        self.timescale = 'free fall'
        self.Gaspard   = 1  # Omega*Tau                 Coriolis force factor. Set to 1 for unit time Tau = 1/Omega
        self.Beyonce   = 0  # (N0*Tau)**2               Buoyancy force factor. Set to 1 for unit time Tau = 1/N0 = sqrt(r0/g0)
        self.ViscosD   = 0  # nu0 * Tau / r0**2         Viscous force factor. Set to 1 for viscous diffusion time scale. This is the Ekman number if Tau = 1/Omega
        self.ThermaD   = 0  # kappa0 * Tau / r0**2      Thermal diffusion factor. Set to 1 for thermal diffusion time scale
        # NEW (2026-10-28): magnetic = 1 only; set_scales sets them as (Gaspard Le)² and Gaspard Em
        self.Hendrik   = 0  # (Tau B0/r0)**2/(rho0 mu0)  Lorentz force factor, (Gaspard*Le)**2. 1 for the Alfvén time scale
        self.MagnetD   = 0  # eta0 * Tau / r0**2         Magnetic diffusion factor, Gaspard*Em. 1 for the magnetic diffusion time scale
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
        # below ~1e-7 at N = 128 (docs/background.md, ThermaD = 0).


        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------- Resolution
        # ----------------------------------------------------------------------------------------------------------------------
        # Number of cpus
        self.ncpus = 2

        # Chebyshev polynomial truncation level. Use function def at top or set manually. N must be even (kore stops otherwise).
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
                                                                    # 0.1-0.5 did not help the ThermaD = 0 tests; 0.1 halves the failures of
                                                                    # nearly parallel modes, such as r-mode ladders (docs/solver.md)
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
            # 'eps_balance'                : 'twoside',             # cleaner eigenvalues for final runs, ~+50% time; erratic with r-mode
                                                                    # ladders (docs/solver.md)
            # 'st_mat_mumps_icntl_14'      : 50,                    # extra MUMPS workspace (%), only if -9 still appears
            # 'st_mat_mumps_icntl_22'      : 1,                     # out-of-core factors, cuts memory ~half
            # 'st_mat_mumps_ooc_tmpdir'    : '/nvm/scratch',        # put OOC files on a real disk, not tmpfs /tmp
            # 'st_mat_mumps_icntl_28'      : 2,                     # parallel ordering ...
            # 'st_mat_mumps_icntl_29'      : 2,                     # ... with ParMETIS

            # ---------------------------------------- forced problems (forcing > 0): ksp_*, pc_*, mat_*
            'ksp_type'                     : 'preonly',             # direct solve, no Krylov iterations
            'pc_type'                      : 'lu',                  # (default GMRES/ILU does not converge)
            'pc_factor_mat_solver_type'    : 'mumps',               # LU with MUMPS
            # 'mat_mumps_icntl_14'         : 100,                   # extra MUMPS workspace (%): a forced LU that runs short returns Inf
                                                                    # (NaN results) instead of stopping (docs/solver.md)
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
        # thermal solar runs.
        self.drop_tol = 0

        # NEW (2026-11-03): submatrices.py writes every operator to one file, submatrices.npz, with the parameters it used
        # (operators.py loads it and stops if this run's parameters differ: bin/submatrix_io.py). 1 also writes one Matrix Market
        # file per operator, <label>.mtx, for outside tools; kore itself no longer reads them.
        self.write_mtx = 0


        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------


    def check_names(self):
        '''
        Warns about the names set in parameters.py that kore does not define here (a misspelt name would otherwise be
        ignored without notice); run-local names of builders are listed too. Called by submatrices.py.
        '''
        unknown = sorted(set(vars(self)) - set(vars(default_params())))
        if unknown:
            print('Warning: parameters.py sets names that kore does not read:', ', '.join(unknown))


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

        # NEW (2026-10-28): the magnetic time scales, unit time the Alfvén time r0 sqrt(ρ0 μ0)/B0 = 1/(Ω Le) or the magnetic
        # diffusion time r0²/η0 = 1/(Ω Em)
        elif self.timescale == "Alfven":

            self.Gaspard = 1/self.Le            # Omega*Tau
            self.Beyonce = 0                    # (N0*Tau)**2
            self.ViscosD = self.Ek/self.Le      # nu0 * Tau / r0**2
            self.ThermaD = self.Etherm/self.Le  # kappa0 * Tau / r0**2

        elif self.timescale == "magnetic":

            self.Gaspard = 1/self.Em            # Omega*Tau
            self.Beyonce = 0                    # (N0*Tau)**2
            self.ViscosD = self.Ek/self.Em      # nu0 * Tau / r0**2
            self.ThermaD = self.Etherm/self.Em  # kappa0 * Tau / r0**2

        # NEW (2026-10-28): the magnetic factors for any time scale (Le, Em are defined with Ω, as Ek). The removed magnetic
        # code of this branch (before 8eb16917) had Hendrik = Le for 'rotation': the Lorentz force goes with Le².
        self.Hendrik = (self.Gaspard*self.Le)**2  # (Tau B0/r0)**2/(rho0 mu0)
        self.MagnetD = self.Gaspard*self.Em       # eta0 * Tau / r0**2


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
