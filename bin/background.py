'''
The background besides the rotation (diff_rot.py), each part chosen by name in parameters.py (params_default.py):
    structure       ρ0, p0 and g: a class of radial_profiles.py
    stratification  dS/dr, S = δs/c_p (thermal = 1)
    diffusivity     ν/ViscosD and κ/ThermaD
x = r r_out is the radius in star radii (r_out: the outer wall). The 'user' types take numpy expressions of r, x and one
parameter p, as diff_rot_profiles does, and params.dat records them after '#' (description).
Only parameters is imported at the top: radial_profiles.py calls check() before it builds its profiles (prf), and the
functions here take that structure as their first argument.
'''
import sys
import numpy as np
import scipy.special as ss
from parameters import par

# As written to params.dat (spin_doctor): the index in each tuple. Append new types at the end.
STRUCTURES = ('8-term fit', 'old model S prototype', 'model S hydrostatic', 'Wu 2005', 'Boussinesq')
SOLAR = STRUCTURES[:3]   # the solar structures: the rotation also uses x = r r_out (diff_rot.solar_x)
STRATIFICATIONS = ('top hat', 'odd top hat', 'solar_rz', 'linear', 'inverse square', 'user')
DIFFUSIVITIES = ('uniform', 'step', 'solar_rz', 'user')

# names that are no longer read and what to set instead (without this stop they would be ignored without notice)
REMOVED = {
    'model_type': "structure, stratification and diffusivity instead: 'user def' is 'Wu 2005', 'top hat', 'uniform'; "
                  "'smoothed sun' is a solar structure ('8-term fit' if structure was not set) with 'solar_rz', 'solar_rz' "
                  "(solar_rz = 1) or 'odd top hat', 'step' (solar_rz = 0); 'Boussinesq' is 'Boussinesq', the stratification "
                  "that replaces heating, 'uniform'",
    'solar_rz': "stratification = 'solar_rz' and diffusivity = 'solar_rz' instead (solar_rz = 0: 'odd top hat' and 'step')",
    'heating': "stratification instead: 'internal' is 'linear', 'differential' is 'inverse square', 'top hat' is 'top hat'",
    'solar_density': "structure instead: 'fit' is '8-term fit', 'model S' is 'old model S prototype'",
    'aux0': 'r_out instead (the same number)',
}


def _names(types):
    return ', '.join(repr(t) for t in types)

def _expression(expr, p, name):
    '''f(r) of a 'user' expression in r, x and p: the numpy names of diff_rot_profiles, x as there (diff_rot.solar_x)'''
    import diff_rot as dr
    code = compile(expr, '<%s>' % name, 'eval')
    def f(r):
        r = np.asarray(r, dtype=float)
        val = eval(code, {'__builtins__': {}}, dict(dr._USER_NS, r=r, x=dr.solar_x(r), p=p))
        return np.broadcast_to(np.asarray(val, dtype=float), r.shape).copy()
    return f

def _check_expression(e, p, name):
    if not isinstance(e, str) or not e or '#' in e:
        sys.exit('%s: a numpy expression of r, x, p as a string without "#" (it is written to params.dat), got %r' % (name, e))
    try:
        _expression(e, p, name)(np.linspace(0.1, 1, 4))
    except Exception as err:
        sys.exit('%s = %r: %s: %s' % (name, e, type(err).__name__, err))


def check():
    '''stops on settings that cannot work'''
    for name, new in REMOVED.items():
        if hasattr(par, name):
            sys.exit('par.%s is no longer read: set %s' % (name, new))
    if par.structure == 'uniform':   # its name from 2026-10-26 to 2026-10-28
        sys.exit("structure 'uniform' is now called 'Boussinesq' (the same ρ0 = p0 = 1, g = r)")
    if par.structure not in STRUCTURES:
        sys.exit('structure %r: set one of %s' % (par.structure, _names(STRUCTURES)))
    if par.structure == 'Wu 2005' and not 0 < par.r_out < 1:
        sys.exit("structure 'Wu 2005' needs 0 < r_out < 1: its density vanishes at x = 1")
    if par.thermal:
        if par.stratification not in STRATIFICATIONS:
            sys.exit('stratification %r: set one of %s' % (par.stratification, _names(STRATIFICATIONS)))
        if par.stratification == 'inverse square' and par.ricb == 0:
            sys.exit("stratification 'inverse square' needs ricb > 0")
        if (par.solar_rz_delta_cz != 0 or par.solar_rz_delta_sf != 0) and par.stratification != 'solar_rz':
            sys.exit("solar_rz_delta_cz, solar_rz_delta_sf are part of stratification 'solar_rz'")
        if par.stratification == 'user':
            _check_expression(par.stratification_profile, par.stratification_param, 'stratification_profile')
    if par.diffusivity not in DIFFUSIVITIES:
        sys.exit('diffusivity %r: set one of %s' % (par.diffusivity, _names(DIFFUSIVITIES)))
    if par.diffusivity == 'user':
        if sorted(par.diffusivity_profiles) != ['kappa', 'nu']:
            sys.exit("diffusivity 'user' needs diffusivity_profiles = {'nu': '...', 'kappa': '...'}")
        for k in ('nu', 'kappa'):
            _check_expression(par.diffusivity_profiles[k], par.diffusivity_param, "diffusivity_profiles['%s']" % k)


def description():
    '''the 'user' expressions as text for params.dat (a '#' comment), empty if there are none'''
    out = []
    if par.thermal and par.stratification == 'user':
        out.append('stratification: %s ; p = %r' % (par.stratification_profile, par.stratification_param))
    if par.diffusivity == 'user':
        out.append('diffusivity: nu = %s ; kappa = %s ; p = %r' % (par.diffusivity_profiles['nu'],
                   par.diffusivity_profiles['kappa'], par.diffusivity_param))
    return ' ; '.join(out)


# ----------------------------------------------------------------------------------------------------- stratification

def gradS(prf, r):
    '''
    dS/dr at r >= 0 (kore uses it as an odd function of r: radial_profiles.pdSdr, gradSX); prf, the structure, gives
    d(ln p0)/dr for the δ of 'solar_rz'.
    '''
    import utils as ut
    t = par.stratification
    if t == 'top hat':          # erf top hat between x1 and x2, aux1..aux5 = x1, w1, x2, w2, A (in r)
        # With ricb = 0 its odd extension is smooth only if dS/dr -> 0 at r = 0: keep x1 >= ~6 w1 (e.g. x1 = 0.2, w1 = 0.03)
        # or use 'odd top hat'. With x1 = w1 = 0.03 (dS/dr(0) = 0.2) the jump at r = 0 spoiled all ricb = 0 thermal runs
        # (ThermaD = 0: residσ ~1; ThermaD > 0: ~1e-5). With ricb > 0, dS/dr != 0 at ricb and ThermaD = 0 gives spurious
        # modes trapped at the inner boundary.
        return ut.erf_top_hat(r, par.aux1, par.aux2, par.aux3, par.aux4, par.aux5)
    if t == 'odd top hat':      # the top hat made odd in r (utils.erf_top_hat_odd): with x1 = 0 it rises as A erf(r/w1)
        return ut.erf_top_hat_odd(r, par.aux1, par.aux2, par.aux3, par.aux4, par.aux5)
    if t == 'linear':           # r: with g = r, N² = Beyonce r² (uniform heat sources; Rieutord 2006, Mirouh et al. 2016)
        return np.array(r, dtype=float)
    if t == 'inverse square':   # 1/r²: conduction between the walls (ricb > 0)
        return 1/r**2
    if t == 'user':             # stratification_profile in r, x, p = stratification_param
        return _expression(par.stratification_profile, par.stratification_param, 'stratification_profile')(r)
    # 'solar_rz': Mukhopadhyay et al. 2025 eq. (10), times erf(x/w0) so that it is odd and smooth at r = 0
    x = r*par.r_out
    x1, x2, d = par.solar_rz_steps
    out = par.solar_rz_dSdr*0.25*(1 - ss.erf((x - x1)/d))*(1 - ss.erf((x - x2)/d))*ss.erf(x/par.solar_rz_core)
    if par.solar_rz_delta_cz != 0 or par.solar_rz_delta_sf != 0:   # convection-zone superadiabaticity δ (Bekki 2024, eq. 2):
        # (1/c_p) ds0/dr = δ d(ln p0)/dr above the overshoot step, δ < 0 subadiabatic; odd in r as d(ln p0)/dr is
        xa = np.abs(x); xos, dos = par.solar_rz_os
        delta = par.solar_rz_delta_cz + (par.solar_rz_delta_sf - par.solar_rz_delta_cz)*np.exp(-((xa - 0.985)/0.015)**2)
        out = out + delta*prf.dlog_p(r)*0.5*(1 + ss.erf((xa - xos)/dos))
    return out


# -------------------------------------------------------------------------------------------------------- diffusivity

def cz_step(r):
    '''0 in the radiative zone, 1 in the convection zone, even in r: 'solar_rz' their eqs. (8), (9); 'step' the upper edge
    of the erf box at aux3, width aux4'''
    if par.diffusivity == 'solar_rz':
        xos, dos = par.solar_rz_os
        return 0.5*(1 + ss.erf((np.abs(r)*par.r_out - xos)/dos))
    import utils as ut
    return 1 - ut.erf_box(r, par.aux3, par.aux4)

def rho_pow(prf, r, p):
    '''(ρ0(x2)/ρ0(r))**p, x2 = aux3, even in r'''
    if p == 0:
        return 1.0
    return np.exp( -p*( prf.logrho(r, 0) - prf.logrho(par.aux3, 0) ) )

def diffusivity(prf, r, which):
    '''ν/ViscosD (which = 'nu') or κ/ThermaD ('kappa') at r, even in r, times (ρ0(aux3)/ρ0)**visc_pow (therm_pow)'''
    c, pw = (par.visc0, par.visc_pow) if which == 'nu' else (par.therm0, par.therm_pow)
    t = par.diffusivity
    if t == 'uniform':
        out = np.ones_like(r)
    elif t == 'user':           # at |r|, so even in r (ricb = 0) whatever the expression, as the 'user' dS/dr and ΔΩ
        out = _expression(par.diffusivity_profiles[which], par.diffusivity_param, "diffusivity_profiles['%s']" % which)(np.abs(r))
    else:                       # 'step', 'solar_rz': visc0 (therm0) below the step, 1 above
        H = 1.0 if c == 1 else cz_step(r)   # no step (and no aux3, aux4 needed) when there is no contrast
        out = H + (1 - H)*c
    return out*rho_pow(prf, r, pw)

def nu_floor():
    '''the part of ν/ViscosD that is not turbulent (forcing 11): visc0 below the step of 'step' and 'solar_rz' if < 1'''
    return par.visc0 if par.diffusivity in ('step', 'solar_rz') and par.visc0 < 1 else 0.
