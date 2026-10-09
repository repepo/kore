from params_default import default_params
#import json
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
# Gravito-inertial mode damped by both viscosity and buoyancy work (gi1):
# lambda = -0.049653 + 2.037155i (shift 2.04i). Of sigma*K, viscous 44 %, buoyancy 56 %.
# Raising/lowering ThermaD moves the split (Th = 2e-4, 3e-4, 5e-4, 1e-3 -> buoyancy 21, 39, 56, 73 %).
par.ricb        = 0.35
par.m           = 2
par.symm        = 1
par.thermal     = 1
par.model_type  = 'user def'
par.aux0        = 0.99  # r peel cutoff
par.aux1        = 0.5   #x1
par.aux2        = 0.03  #w1
par.aux3        = 0.85  #x2
par.aux4        = 0.03  #w2
par.aux5        = 2.5   #A
par.Gaspard     = 1.
par.Beyonce     = 10.
par.ViscosD     = 1e-4
par.ThermaD     = 5e-4

par.visc0       = 1.0   # core/envelope viscosity ratio
par.rvisc       = 0.60   # transition radius
par.hvisc       = 0.04   # transition width

par.diff_rot    = 0
par.diff_rot_type = "conical" # Possible types : Y20, Y20-wall-bounded, shellular, cylindrical, conical, shellular_boussinesq and solar
par.diff_rot_amplitude = -0.25

par.bci         = 0
par.bco         = 0
par.bci_thermal = 0
par.bco_thermal = 0
par.ncpus       = 10
par.N           = 192
par.lmax        = 121
par.rtau        = 0     # σ : damping factor (negative is damped)
par.itau        = 2.04  # ⍵ : frequency (negative is prograde)
par.smopo       = 0
par.nev         = 4
par.which_eigenpairs = 'TM'

par.rpower_u    = 5
par.rhopower_u  = 1

par.rpower_v    = 3
par.rhopower_v  = 1  

par.rpower_pp   = 4
par.rhopower_pp = 4

# ---------------------------------------

# ---------------------------------------
# -------------- Write parameters to file
# ---------------------------------------
# pars_dict = vars(par)
# with open('params.json', 'w') as f:
#     json.dump(pars_dict, f, indent=4)
#
# To load from a file
# with open('params.json', 'r') as f:
#     pars_dict = json.load(f)
#     for key, value in pars_dict.items():
#         setattr(par, key, value)
# ---------------------------------------
