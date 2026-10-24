from params_default import default_params
import numpy as np

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
# (δ_cz fitted to the HFR frequencies of Hanson et al. 2022, docs/heat-equation.md). Copy to bin/parameters.py.
# The m = 8 high-frequency retrograde (HFR) mode near the target: σ = -33.35, ω = -254.83 nHz in the 456 nHz frame
# (λ = -0.0731 + 0.5588i), spin_doctor residuals below 6e-5; 23 GB peak and about 4 minutes with 4 ranks.

RSUN, GSURF = 6.9599e10, 2.7361e4   # cm, cm/s²
frame = 456.0                       # rotating frame, nHz
Omega = 2*np.pi*frame*1e-9          # rad/s
L = 0.985*RSUN                      # unit of length, aux0 R☉
nu_cz = 1e12                        # ν = κ in the convection zone, cm²/s

par.m           = 8
par.symm        = 1
par.ricb        = 0.0
par.thermal     = 1
par.model_type  = 'smoothed sun'
par.structure   = 'model S hydrostatic'
par.structure_deg = 30
par.aux0        = 0.985

par.solar_rz    = 1                 # radiative interior: dS/dr steps at 0.61, 0.68 R☉ (width 0.04), ν, κ step at 0.70 (0.04)
par.solar_rz_steps = [0.61, 0.68, 0.04]
par.solar_rz_os = [0.70, 0.04]
par.visc0       = 1e10/nu_cz        # ν_RZ/ν_CZ
par.therm0      = 4e10/nu_cz        # κ_RZ/κ_CZ
par.solar_rz_delta_cz = -2.1e-7     # convection-zone superadiabaticity (< 0 subadiabatic), Hanson et al. 2022 MCA fit
par.solar_rz_delta_sf = 0.0

par.diff_rot    = 1
par.diff_rot_type = 'tabulated'
par.diff_rot_file = 'bin/jsoc_hmi_2010_2026.npz'  # JSOC HMI mean of the 83 72-day inversions of hmi.V_sht_2drls (NACOEFF 36), 2010-04-30 to 2026-06-29
par.diff_rot_nmax = 4
par.diff_rot_window = 'sun'
par.diff_rot_sun_deg = 60
par.diff_rot_hold = 0.6
par.diff_rot_frame = frame
par.thermal_wind = 2
par.thermal_wind_radial = 0

par.Gaspard     = 1.0
par.Beyonce     = GSURF/(L*Omega**2)
par.ViscosD     = nu_cz/(Omega*L**2)
par.ThermaD     = par.ViscosD

par.bci         = 0                 # stress-free, impenetrable, ∂S/∂r = 0
par.bco         = 0
par.bci_thermal = 1
par.bco_thermal = 1

par.ncpus       = 4
par.N           = 640
par.lmax        = 79

par.rtau        = -33.5/frame       # target, nHz
par.itau        = 254.8/frame
par.nev         = 20
par.which_eigenpairs = 'TM'

par.drop_tol    = 1e-14
par.petsc_opts['st_mat_mumps_icntl_14'] = 50
par.petsc_opts['st_mat_mumps_icntl_7']  = 5    # METIS ordering
