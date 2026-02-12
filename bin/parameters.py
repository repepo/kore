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
par.ricb        = 0.1
par.m           = 3
par.symm        = -1
par.thermal     = 0
par.model_type  = 'user def'
#par.model       = 'poly.n3_isentropic.h5'
par.aux0        = 0.96   # r peel cutoff
par.aux1        = -0.71  # r₁
par.aux2        = 0.71  # r₂
par.aux3        = 2.0   # Amplitude of Γ₁ deviation
par.aux4        = 1.0   # hard edge = 0,  soft edge = 1
par.aux5        = 0
par.Gaspard     = 1
par.Beyonce     = 1e4  #(1.0/0.3)**2
par.ViscosD     = 1e-4
par.ThermaD     = 0

par.visc0       = 1.0   # core/envelope viscosity ratio
par.rvisc       = 0.60   # transition radius
par.hvisc       = 0.04   # transition width

par.bci         = 0
par.bco         = 0
par.bci_thermal = 0
par.bco_thermal = 0
par.ncpus       = 10
par.N           = 10*28
g               = 1.0
par.lmax        = par.ellmax(par.ncpus, g, par.m, par.N)
rnd1            =-0.555094
rnd2            =-0.270064
par.rtau        = -0.001 #+ rnd1*1e-2
par.itau        = 0.4995   #+ rnd2*0.001 
par.smopo       = 0
par.nev         = 7
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
