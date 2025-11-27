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
par.m           = 2
par.symm        = 1
par.thermal     = 0
par.model_type  = 'user def'
par.aux0        = 1.0   # r peel cutoff
par.aux1        = -0.7  # r₁
par.aux2        = 0.7   # r₂
par.aux3        = 2.0   # Amplitude of Γ₁ deviation
par.aux4        = 1.0   # hard edge = 0,  soft edge = 1
par.aux5        = 0
par.Gaspard     = 1.0
par.Beyonce     = (1.0/0.3)**2
par.ViscosD     = 1e-4
par.ThermaD     = 0
par.bco         = 1
par.bco_thermal = 0
par.ncpus       = 10
par.N           = 640
g               = 0.5
par.lmax        = par.ellmax(par.ncpus, g, par.m, par.N)
rnd1            = 0
rnd2            = 0
par.rtau        = -2.5e-4  # + rnd1*1e-4
par.itau        = 1.1      # + rnd2*0.5 
par.smopo       = 0
par.nev         = 5
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
