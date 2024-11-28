#!/usr/bin/env python3

import numpy as np
from scipy.io import savemat
import parameters as par
import utils as ut

keys = []
values = []
tol = 1e-9

twozone    = ((par.thermal == 1) and (par.heating == 'two zone'))               # boolean
userdef    = ((par.thermal == 1) and (par.heating == 'user defined'))           # boolean
cdipole    = ((par.magnetic == 1) and (par.B0 == 'dipole') and (par.ricb > 0))  # boolean
inviscid   = (((par.Ek == 0) or (par.ViscosD == 0)) and (par.ricb == 0))        # boolean
quadrupole = ((par.B0 == 'Luo_S2') or ((par.B0 == 'FDM') and (par.B0_l == 2)))  # boolean
anelastic  = (par.anelastic==1)
boussinesq = par.anelastic==0 and par.thermal==1

if anelastic:
    if par.interior_model=='mesa':
        import mesa_profiles as rap
    else:
        import radial_profiles as rap
else:
    import radial_profiles as rap

if boussinesq:
    if twozone:
        cd_ent = ut.chebco_rf(rap.twozone, rpower=1, N=par.N, ricb=par.ricb, rcmb=ut.rcmb, tol=tol, args=par.args).reshape([par.N,1])
    elif userdef:
        cd_ent = ut.chebco_rf(rap.BVprof, rpower=1, N=par.N, ricb=par.ricb, rcmb=ut.rcmb, tol=tol, args=par.args).reshape([par.N,1])
    if twozone or userdef:
        keys.append('cd_ent')
        values.append(cd_ent)

elif anelastic:  ##

    cd_rho = ut.chebify( rap.density, 2, tol)
    cd_lho = ut.chebify( rap.log_density, 4, tol)
    cd_vsc = ut.chebify( rap.viscosity, 2, tol)
    cd_krT = ut.chebify( rap.krT, 1, tol)
    cd_roT = ut.chebify( rap.roT, 0, tol)

    keys.append('cd_rho')
    keys.append('cd_lho')
    keys.append('cd_vsc')
    keys.append('cd_krT')
    keys.append('cd_roT')

    values.append(cd_rho)
    values.append(cd_lho)
    values.append(cd_vsc)
    values.append(cd_krT)
    values.append(cd_roT)

    if par.thermal == 1:

        cd_lnT = ut.chebify( rap.log_temperature, 1, tol)
        cd_kho = ut.chebify( rap.kappa_rho, 1, tol)
        cd_tds = ut.chebify( rap.tds, 0, tol)

        if par.autograv:
            cd_buo = ut.chebco_rf(rap.buoFac,0,par.N,par.ricb,ut.rcmb,tol)
            cd_buo = ut.cheb2Product(cd_buo,ac.gravCoeff(),tol).reshape([par.N,1])
        else:
            cd_buo = ut.chebco_rf(rap.buoFac,0,par.N,par.ricb,ut.rcmb,tol).reshape([par.N,1])

        keys.append('cd_lnT')
        keys.append('cd_kho')
        keys.append('cd_tds')
        keys.append('cd_buo')

        values.append(cd_lnT)
        values.append(cd_kho)
        values.append(cd_tds)
        values.append(cd_buo)

if par.magnetic:

    cd_eta = ut.chebify( rap.magnetic_diffusivity, 1, tol)

    keys.append('cd_eta')
    values.append(cd_eta)

    if par.anelastic:
        cd_eho = ut.chebify( rap.eta_rho, 1, tol)

        keys.append('cd_eho')
        values.append(cd_eho)

dict_prof = dict(zip(keys,values))

savemat('radProfs.mat',dict_prof)
rap.write_profiles()

print("Profiles generated:",keys)