from params_default import default_params
import json

## Set default parameters

par = default_params()

par.timescale = "rotation"
par.set_scales()

par.Ek   = 1e-4
par.bci  = 1
par.bco  = 1
par.rtau = 0
par.itau = 1
# Write parameters to file

pars_dict = vars(par)

with open('params.json', 'w') as f:
    json.dump(pars_dict, f, indent=4)