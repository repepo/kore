from params_default import default_params
import numpy as np
import units                        # bin/units.py: physical units to kore's (kore itself is dimensionless)

# ---------------------------------------
# --------------- Sets default parameters
# ---------------------------------------
par = default_params()
par.timescale = 'rotation'
par.set_scales()
# ---------------------------------------

# ---------------------------------------
# --------------- Manual parameter adjust
# ---------------------------------------
# A smooth Sun in the full sphere (ricb = 0): Model S density with g and p in hydrostatic balance, the radiative interior of
# Mukhopadhyay et al. 2025b (dS/dr, ν_RZ = 1e10 and κ_RZ = 4e10 cm²/s), and the observed differential rotation (HMI
# 2010-2026 mean, Legendre degrees up to 4) with its exact thermal wind, and a slightly subadiabatic convection zone
# (δ_cz close to the fit to the RDA HFR frequencies of Hanson et al. 2022; at the MCA fit,
# -2.1e-7, the m = 8 HFR mixes with deep-core r modes). Copy to bin/parameters.py.
# The m = 8 high-frequency retrograde (HFR) mode near the target: σ = -33.18, ω = -271.19 nHz in the 456 nHz frame
# (λ = -0.07275 + 0.59471i), with 0.55 % of its kinetic energy below 0.70 R☉; residσ 8.7e-5, resid𝑠 3.3e-5 and
# ℓ-convergence 1.5e-5; 21 GB peak and about 5 minutes with 4 ranks.

nu_cz = 1e12                        # ν = κ in the convection zone, cm²/s

par.m           = 8
par.symm        = 1
par.ricb        = 0.0
par.thermal     = 1
par.r_out       = 0.985             # outer wall in R☉, the unit of length
par.structure   = 'model S hydrostatic'
par.structure_deg = 30

par.stratification = 'solar_rz'     # radiative interior: dS/dr steps at 0.61, 0.68 R☉ (width 0.04)
par.solar_rz_steps = [0.61, 0.68, 0.04]
par.solar_rz_delta_cz = -4e-7       # convection-zone superadiabaticity (< 0 subadiabatic), near the HFR RDA fit
par.solar_rz_delta_sf = 0.0
par.diffusivity = 'solar_rz'        # ν, κ step at 0.70 R☉ (width 0.04)
par.solar_rz_os = [0.70, 0.04]
par.visc0       = 1e10/nu_cz        # ν_RZ/ν_CZ
par.therm0      = 4e10/nu_cz        # κ_RZ/κ_CZ

par.diff_rot    = 1
par.diff_rot_type = 'tabulated'
par.diff_rot_file = 'bin/jsoc_hmi_2010_2026.npz'  # JSOC HMI mean of the 83 72-day inversions of hmi.V_sht_2drls (NACOEFF 36), 2010-04-30 to 2026-06-29
par.diff_rot_nmax = 4
par.diff_rot_window = 'sun'
par.diff_rot_sun_deg = 60
par.thermal_wind = 2
par.thermal_wind_radial = 0

units.set_physical(par, frame_nHz=456.0, nu=nu_cz, kappa=nu_cz)   # Gaspard, Beyonce, ViscosD, ThermaD, diff_rot_frame

par.bci         = 0                 # stress-free, impenetrable, ∂S/∂r = 0
par.bco         = 0
par.bci_thermal = 1
par.bco_thermal = 1

par.ncpus       = 4
par.N           = 640
par.lmax        = 79

units.set_target(par, omega=-271.2, sigma=-33.2)                 # itau, rtau from ω, σ in nHz
par.nev         = 20
par.which_eigenpairs = 'TM'

par.drop_tol    = 1e-14
par.petsc_opts['st_mat_mumps_icntl_14'] = 50
par.petsc_opts['st_mat_mumps_icntl_7']  = 5    # METIS ordering
