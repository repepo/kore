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
par.Ek       = 10**-7
par.magnetic = 1
par.B0       = 'FDM'     # torsional waves need B_s != 0, the axial field has none
par.mantle   = 'TWA'
par.c_cmb    = 1e-2
par.c1_cmb   = 1.5e-2    # magnetic torque ~ viscous torque
par.Lambda   = 10        # Le = sqrt(Lambda*Ek/Pm) = 3.2e-3
par.Pm       = 0.1       # Lundquist number Le/Em = 3.2e3

par.rtau     = -0.0046   # aims at the mode lambda = -0.00463 + 0.0115i
par.itau     = 0.0115
# ---------------------------------------

# ---------------------------------------
# ---- Derived parameters (call it last)
# ---------------------------------------
par.set_scales()
# ---------------------------------------
