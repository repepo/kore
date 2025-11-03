from params_default import default_params
import json


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
par.m       = 0
par.symm    = 1
par.thermal = 1
par.model   = 'poly.n1_gamma3.h5'
par.Gaspard = 1
par.Beyonce = (1/0.3)**2
par.ncpus   = 2
par.N       = 96
par.lmax    = 2*par.ncpus*12 + par.m - 1
par.rtau    = 0
par.itau    = 0.7955/0.3
par.smopo   = 0
# ---------------------------------------


# ---------------------------------------
# -------------- Write parameters to file
# ---------------------------------------
pars_dict = vars(par)
with open('params.json', 'w') as f:
    json.dump(pars_dict, f, indent=4)

# To load from a file
# with open('params.json', 'r') as f:
#     pars_dict = json.load(f)
#     for key, value in pars_dict.items():
#         setattr(par, key, value)
# ---------------------------------------
