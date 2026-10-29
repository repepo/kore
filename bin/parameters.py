from params_default import default_params, pmak, ssak
import numpy as np

# ---------------------------------------
# --------------- Sets default parameters
# ---------------------------------------
# All parameters, their defaults and the available options are in params_default.py.
# Set here only those that differ from the defaults.
par = default_params()
# ---------------------------------------

# ---------------------------------------
# --------------- Manual parameter adjust
# ---------------------------------------
# Torsional-mode setup: FDM field with a thin conducting wall at the CMB
par.Ek       = 1e-3
par.magnetic = 1
par.B0       = 'FDM'     # torsional waves need B_s != 0, the axial field has none
par.mantle   = 'TWA'
par.c_cmb    = 1e-2
par.c1_cmb   = 1.5e-2    # magnetic torque ~ viscous torque
par.Lambda   = 10        # Le = sqrt(Lambda*Ek/Pm) = 3.2e-3
par.Pm       = 0.1       # Lundquist number Le/Em = 3.2e3

# Spin over mode (no inner core)
par.ricb     = 0
par.m        = 1
par.symm     = -1

# Export parameters as JSON
# par.export_json("parameters.json")

# Use JSON file to load parameters if needed
# par.load_json("parameters.json")

# ---------------------------------------

# ---------------------------------------
# ---- Derived parameters (call it last)
# ---------------------------------------
par.set_scales()
# ---------------------------------------
