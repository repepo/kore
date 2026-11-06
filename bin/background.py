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
MAGNETIC_WALLS = ('insulator', 'TWA', 'resolved')   # NEW (2026-10-28): mantle (magnetic = 1); B0 types: background_field.B0_TYPES
INNER_CORES = ('insulator', 'conducting, Chebys', 'TWA')   # NEW (2026-10-28): innercore (magnetic = 1); 'TWA' NEW 2026-11-06
B0_CMBS = ('wall', 'plain')   # NEW (2026-11-06): B0_cmb (magnetic = 1, mantle = 'TWA'), as written to params.dat
MAG_COUPLINGS = ('closed', 'algebra')            # NEW (2026-11-03): mag_couplings, B0_ic and ferraro_policy (magnetic = 1),
B0_ICS = ('potential', 'frozen currents')        # also written to params.dat as these indices ('frozen currents' was 'full sphere')
FERRARO_POLICIES = ('warn', 'refuse')

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
    if par.hydro not in (0, 1):   # NEW (2026-11-06): hydro = 0, no flow (as conductive_IC)
        sys.exit('hydro %r: 1 (the flow is solved) or 0 (no flow)' % (par.hydro,))
    if par.hydro == 0 and not (par.magnetic or par.thermal):
        sys.exit('hydro = 0 needs magnetic = 1 or thermal = 1: without the flow nothing else is solved')
    if par.hydro == 0 and par.forcing != 0:
        sys.exit('hydro = 0 is for eigenvalue problems (forcing = 0): the free decay of the field or of the entropy')
    if par.N % 2:   # NEW (2026-11-04): the operators of submatrices.py assume an even N (an odd one stopped there with a scipy error)
        sys.exit('N must be even, got N = %r (the operators assume it, in a shell too; a full sphere uses N/2 Chebyshev '
                 'polynomials of each parity)' % (par.N,))
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
    if par.magnetic:
        check_magnetic()
    if par.centrifugal:   # NEW (2026-11-05)
        check_centrifugal()
    if par.diffusivity not in DIFFUSIVITIES:
        sys.exit('diffusivity %r: set one of %s' % (par.diffusivity, _names(DIFFUSIVITIES)))
    if par.diffusivity == 'user':
        if sorted(par.diffusivity_profiles) != ['kappa', 'nu']:
            sys.exit("diffusivity 'user' needs diffusivity_profiles = {'nu': '...', 'kappa': '...'}")
        for k in ('nu', 'kappa'):
            _check_expression(par.diffusivity_profiles[k], par.diffusivity_param, "diffusivity_profiles['%s']" % k)


def check_centrifugal():
    '''
    NEW (2026-11-05): stops on the settings that the centrifugal deformation (centrifugal = 1, centrifugal.py) cannot do yet:
    a forcing (the forced problems need the metric terms in A and boundary flows on the deformed walls), the zero radial
    stress walls (bco, bci = 2; the stress-free and no-slip ones are done, phase 2) and, with magnetic = 1 (NEW 2026-11-05,
    phase 1b), a conducting inner core of another permeability than the fluid's (its matching rows would need the metric);
    NEW (2026-11-06, phase 2) rpower_v < 4 in a viscous full sphere with m != 0 when the body is deformed (the toroidal rows
    of the metric viscous force have a term P/r); and on incomplete shapes. NEW (2026-11-06): the exact spheroid
    (centrifugal_shape 'spheroid') is a homogeneous fluid without a differential rotation (NEW 2026-11-06, step 2: with or
    without a field).
    centrifugal.report() (submatrices.py) prints the flattening used and the warnings on the terms left spherical.
    '''
    import centrifugal as cf
    if par.centrifugal != 1:
        sys.exit('centrifugal must be 0 or 1, got %r' % (par.centrifugal,))
    if par.magnetic and par.ricb > 0 and par.innercore == 'conducting, Chebys' and par.mu_i2o != 1:
        sys.exit('centrifugal = 1 with a conducting inner core needs mu_i2o = 1: with another permeability the matching of the '
                 'tangential field on the deformed inner boundary has metric terms that are not implemented')
    if par.forcing != 0:
        sys.exit('centrifugal = 1 is implemented for eigenvalue problems only (forcing = 0): the forced problems need the '
                 'metric terms of the time derivative in A and their boundary flows on the deformed walls')
    if par.ViscosD > 0 and (par.bco == 2 or (par.ricb > 0 and par.bci == 2)):   # FIX (2026-11-05, phase 2): stress-free walls done
        sys.exit('centrifugal = 1 takes no-slip or stress-free walls (bco, bci = 1 or 0): the zero radial stress rows (2) on a '
                 'deformed wall need metric terms that are not implemented')
    if par.centrifugal_shape not in cf.SHAPES:
        sys.exit('centrifugal_shape %r: set one of %s' % (par.centrifugal_shape, _names(cf.SHAPES)))
    q, f = par.centrifugal_q, par.centrifugal_f
    if not isinstance(q, (int, float)) or q < 0:
        sys.exit('centrifugal_q must be a number >= 0, got %r' % (q,))
    if f is not None and not isinstance(f, (int, float)):
        sys.exit('centrifugal_f must be None or a number, got %r' % (f,))
    if par.centrifugal_shape == 'uniform' and f is None:
        sys.exit("centrifugal_shape 'uniform' needs centrifugal_f, the flattening")
    if par.centrifugal_shape == 'spheroid':   # NEW (2026-11-06): the exact spheroid (centrifugal.py, section 8)
        if f is None or not f < 1:
            sys.exit("centrifugal_shape 'spheroid' needs centrifugal_f, the flattening, below 1 (got %r)" % (f,))
        if par.structure != 'Boussinesq' or par.diffusivity != 'uniform':
            sys.exit("centrifugal_shape 'spheroid' is a homogeneous fluid: it needs structure 'Boussinesq' and diffusivity "
                     "'uniform' (got %r, %r)" % (par.structure, par.diffusivity))
        if par.diff_rot:   # (NEW 2026-11-06, step 2: a field is allowed)
            sys.exit("centrifugal_shape 'spheroid' is implemented without a differential rotation (diff_rot = 0)")
        if q != 0:
            sys.exit("centrifugal_shape 'spheroid' takes its flattening from centrifugal_f and the effective gravity from the "
                     "structure's g = r (that of a homogeneous spheroid): set centrifugal_q = 0")
    if par.centrifugal_shape == 'user':
        if f is not None:
            sys.exit("centrifugal_shape 'user' takes the flattening from centrifugal_profile: set centrifugal_f = None")
        _check_expression(par.centrifugal_profile, par.centrifugal_param, 'centrifugal_profile')
    if par.ViscosD > 0 and par.ricb == 0 and par.m != 0 and par.rpower_v < 4 and cf.deformed():   # NEW (2026-11-06, phase 2)
        sys.exit('centrifugal = 1 in a full sphere with viscosity and m != 0 needs rpower_v >= 4 (got %r): the metric part of '
                 'the viscous force has a term P/r in the toroidal rows, which are polynomials in r only when multiplied by r^4 '
                 '(set par.rpower_v = 4)' % (par.rpower_v,))


# NEW (2026-11-06, from conductive_IC, after PR #17 by A. Nuyt): the resolved conducting mantle (mantle = 'resolved'), a layer
# on [1, r_surface] whose field is part of the unknowns (sections fm, gm; assemble.bc_b_mantle). Its conductivity relative to
# the fluid's is the number sigma_mantle, or the numpy expression sigma_mantle of r and p = sigma_mantle_param (with the names of
# the 'user' profiles, background_field.USER_NS).

_sigma_m = []

def sigma_mantle(r):
    '''NEW (2026-11-06): the conductivity of the resolved mantle at r (1 <= r <= r_surface), relative to the fluid's'''
    if not _sigma_m:
        if isinstance(par.sigma_mantle, str):
            import background_field as bf
            code = compile(par.sigma_mantle, '<sigma_mantle>', 'eval')
            def f(r):
                with np.errstate(all='ignore'):
                    s = eval(code, {'__builtins__': {}}, dict(bf.USER_NS, r=r, p=par.sigma_mantle_param))
                return np.broadcast_to(np.asarray(s, dtype=float), np.shape(r))
            _sigma_m.append(f)
        else:
            s0 = float(par.sigma_mantle)
            _sigma_m.append(lambda r: np.full(np.shape(r), s0))
    return _sigma_m[0](np.asarray(r, dtype=float))


_dlnsigma_m = []

def dlnsigma_mantle(r):
    '''
    NEW (2026-11-06): (ln sigma)' in the resolved mantle, from a Chebyshev interpolant of ln sigma on [1, r_surface] without its
    round-off tail, which the derivative would amplify (0 for a uniform layer, a number)
    '''
    if not isinstance(par.sigma_mantle, str):
        return np.zeros(np.shape(r))
    if not _dlnsigma_m:
        lns = np.polynomial.Chebyshev.interpolate(lambda r: np.log(sigma_mantle(r)), max(128, 4*par.N_mantle),
                                                  domain=[1.0, par.r_surface])
        _dlnsigma_m.append(lns.trim(1e-13*np.abs(lns.coef).max()).deriv())
    return _dlnsigma_m[0](np.asarray(r, dtype=float))


def mantle_conductance():
    '''NEW (2026-11-06): ∫ sigma dr from 1 to r_surface: the c1_cmb of a thin wall with the conductance of the resolved mantle'''
    x, w = np.polynomial.legendre.leggauss(200)
    h = par.r_surface - 1.0
    return 0.5*h*np.sum(w*sigma_mantle(1.0 + 0.5*h*(x + 1)))


def check_mantle():
    '''
    NEW (2026-11-06, conductive_IC's check_mantle): stops on the settings that the resolved mantle (mantle = 'resolved') does
    not cover, before any matrix is built. r_surface > 1; N_mantle an integer >= 4; sigma_mantle a number, or an expression
    without '#' (it is written to params.dat), finite and > 0 on [1, r_surface] and resolved by N_mantle Chebyshev polynomials:
    B holds the N_mantle-term series of r² sigma and the toroidal rows that of r² (ln sigma)', which oscillate where they do not
    resolve the profile (sigma in B can dip below 0, giving spurious growing modes, and (ln sigma)' is wrong, giving wrong
    toroidal modes); a step or a kink is never resolved. Also mu = 1 (a magnetic mantle or fluid would change the background
    field at the wall), a spherical body (centrifugal = 0) and an impenetrable wall (bco = 0 or 1: the u×B0 of a slipping fluid
    assumes u_r = 0 there).
    '''
    num = (int, float, np.integer, np.floating)
    if not (isinstance(par.r_surface, num) and np.isfinite(par.r_surface) and par.r_surface > 1):
        sys.exit("mantle = 'resolved' needs a finite r_surface > 1 (the outer wall), got %r" % (par.r_surface,))
    if not (isinstance(par.N_mantle, (int, np.integer)) and not isinstance(par.N_mantle, bool) and par.N_mantle >= 4):
        sys.exit('N_mantle = %r: an integer >= 4' % (par.N_mantle,))
    if isinstance(par.sigma_mantle, str):
        if not par.sigma_mantle or '#' in par.sigma_mantle:
            sys.exit("sigma_mantle: a number or a numpy expression of r and p without '#' (it is written to params.dat), "
                     "got %r" % (par.sigma_mantle,))
    elif not isinstance(par.sigma_mantle, num) or isinstance(par.sigma_mantle, bool):
        sys.exit('sigma_mantle = %r: a number or a numpy expression of r and p' % (par.sigma_mantle,))
    try:
        s = sigma_mantle(np.linspace(1.0, par.r_surface, 401))
    except Exception as err:
        sys.exit('sigma_mantle = %r: %s: %s' % (par.sigma_mantle, type(err).__name__, err))
    if not (np.all(np.isfinite(s)) and np.all(s > 0)):
        sys.exit('sigma_mantle = %r must be finite and > 0 from r = 1 to r_surface' % (par.sigma_mantle,))
    if isinstance(par.sigma_mantle, str):   # the coefficients beyond N_mantle of r² sigma (relative) and ln sigma (absolute)
        N4 = 4*par.N_mantle
        to_r = lambda x: 1.0 + 0.5*(par.r_surface - 1.0)*(x + 1)
        c = np.abs(np.polynomial.chebyshev.chebinterpolate(lambda x: to_r(x)**2*sigma_mantle(to_r(x)), N4 - 1))
        with np.errstate(all='ignore'):
            cl = np.abs(np.polynomial.chebyshev.chebinterpolate(lambda x: np.log(sigma_mantle(to_r(x))), N4 - 1))
        tail = c[par.N_mantle:].max()/c.max(), cl[par.N_mantle:].max()
        if not max(tail) <= 1e-6:
            sys.exit('sigma_mantle = %r is not resolved by N_mantle = %d Chebyshev polynomials: the coefficients beyond '
                     'N_mantle reach %.1e of the largest for r**2 sigma and %.1e for ln(sigma), both must be <= 1e-6. Increase '
                     'N_mantle, or smooth the profile (a step or a kink is never resolved)'
                     % (par.sigma_mantle, par.N_mantle, tail[0], tail[1]))
    if par.mu != 1:
        sys.exit("mantle = 'resolved' assumes the permeability of vacuum in the fluid and in the layer (mu = 1)")
    if par.centrifugal:
        sys.exit("mantle = 'resolved' needs a spherical body (centrifugal = 0)")
    if par.hydro and par.bco == 2:
        sys.exit("mantle = 'resolved' needs an impenetrable outer wall (bco = 0 or 1): the u×B0 of a fluid slipping along it "
                 "assumes u_r = 0 there")


def check_twa():
    '''
    NEW (2026-11-06): stops on thin conducting walls (mantle or innercore 'TWA', from conductive_IC) outside what the wall rows
    of assemble.bc_b_spherical cover. Their parameters must be numbers: c, c1 >= 0, muw > 0 (np.inf allowed), mu > 0. The fluid's
    permeability mu enters the thin-wall conditions only, so mu != 1 needs a thin wall (or a conducting inner core) at every
    wall of the fluid: an insulating wall with mu != 1 is a thin wall with c = c1 = 0. A thin wall needs an impenetrable
    boundary (bco, bci = 0 or 1: the terms of a slipping fluid assume u_r = 0 there) and a spherical body (centrifugal = 0).
    With a differential rotation, ΔΩ must be uniform in latitude at the thin wall (background.magnetic_diffrot): the wall
    co-rotates with it, as a conducting inner core does.
    '''
    twa_o = par.mantle == 'TWA'
    twa_i = par.innercore == 'TWA' and par.ricb > 0
    num = (int, float, np.integer, np.floating)
    for name in ('c_cmb', 'c1_cmb', 'muw_cmb', 'c_icb', 'c1_icb', 'muw_icb', 'mu'):
        v = getattr(par, name)
        if not isinstance(v, num) or isinstance(v, bool) or np.isnan(v) or v < 0 or (name.startswith('mu') and v == 0):
            sys.exit('%s = %r: a number, >= 0 for c and c1 and > 0 for muw and mu' % (name, v))
    for name in ('c_cmb', 'c1_cmb', 'c_icb', 'c1_icb', 'mu'):
        if np.isinf(getattr(par, name)):
            sys.exit('%s must be finite (only muw_cmb, muw_icb may be np.inf)' % name)
    if par.mu != 1 and not (twa_o and (par.ricb == 0 or par.innercore in ('TWA', 'conducting, Chebys'))):
        sys.exit("mu = %r enters only the thin-wall conditions: every wall of the fluid must then be a thin wall ('TWA'; an "
                 "insulating wall with mu != 1 is a thin wall with c = c1 = 0) or a conducting inner core" % (par.mu,))
    if not (twa_o or twa_i):
        return
    if par.centrifugal:
        sys.exit("a thin conducting wall ('TWA') needs a spherical body (centrifugal = 0)")
    if par.hydro and ((twa_o and par.bco == 2) or (twa_i and par.bci == 2)):
        sys.exit("a thin conducting wall ('TWA') needs an impenetrable wall (bco, bci = 0 or 1): the terms of a fluid slipping "
                 "along it assume u_r = 0 there")


def check_magnetic():
    '''
    NEW (2026-10-28): stops on magnetic settings that are not implemented (yet).
    '''
    import background_field as bf
    if par.B0 is None and par.B0_tor is None:
        sys.exit('magnetic = 1 needs a background field: B0 (poloidal) and/or B0_tor (toroidal)')
    if par.B0 is not None and par.B0 not in bf.B0_TYPES:
        sys.exit('B0 %r: set None or one of %s' % (par.B0, _names(bf.B0_TYPES)))
    if par.B0_tor is not None:   # NEW (2026-10-28): toroidal B0
        if par.B0_tor not in bf.B0_TOR_TYPES:
            sys.exit('B0_tor %r: set None or one of %s' % (par.B0_tor, _names(bf.B0_TOR_TYPES)))
        if par.B0_tor == 'general':   # NEW (2026-10-30)
            _check_general(par.B0_tor_profiles, 'B0_tor_profiles')
        elif par.B0_tor_l not in (1, 2):
            sys.exit('B0_tor_l = %r: only 1 and 2 are implemented' % par.B0_tor_l)
        if par.B0_tor == 'user':
            if not isinstance(par.B0_tor_profile, str) or not par.B0_tor_profile or '#' in par.B0_tor_profile:
                sys.exit("B0_tor 'user' needs B0_tor_profile, a numpy expression of r and p (without '#'), got %r" % (par.B0_tor_profile,))
            try:
                bf.k(np.linspace(max(par.ricb, 0.1), 1, 4), 0)
            except Exception as err:
                sys.exit('B0_tor_profile = %r: %s: %s' % (par.B0_tor_profile, type(err).__name__, err))
        # FIX (2026-11-06): numpy numbers too (np.int64 is not an int), as conductive_IC 4bfb389
        if not (isinstance(par.cnorm_tor, (int, float, np.integer, np.floating)) or par.cnorm_tor in ('mag_energy', 'Schmitt2012')):
            sys.exit("cnorm_tor %r: a number, 'mag_energy' or 'Schmitt2012'" % (par.cnorm_tor,))
        if par.B0 is not None and bf.symm_pol != bf.symm_tor:
            sys.exit('B0 %r and B0_tor %r have opposite equatorial symmetries: they cannot be combined in one eigenproblem'
                     % (par.B0, par.B0_tor))
    if par.B0 == 'general':   # NEW (2026-10-30)
        _check_general(par.B0_profiles, 'B0_profiles')
        if par.cnorm == 'rms_cmb':   # FIX (2026-11-03): stopped here instead of crashing in background_field.B0_norm
            sys.exit("B0 'general': use cnorm 'mag_energy' or a number")
    if par.B0 == 'ferraro':   # NEW (2026-10-30)
        if not par.diff_rot:
            sys.exit("B0 'ferraro' needs a differential rotation (diff_rot = 1): its flux function is G(ΔΩ)")
        if par.diff_rot_type == 'ferraro':
            sys.exit("diff_rot_type 'ferraro' with B0 'ferraro' is circular (each is built from the other): choose one")
        if par.B0_ferraro_nmax < 1 or par.B0_ferraro_nmax % 2 == 0:
            sys.exit('B0_ferraro_nmax must be odd and >= 1 (the field has only odd degrees)')
        if not isinstance(par.B0_ferraro_G, str) or not par.B0_ferraro_G or '#' in par.B0_ferraro_G:   # FIX (2026-11-03): params.dat
            sys.exit("B0_ferraro_G: a numpy expression of w and p as a string without '#' (it is written to params.dat), got %r"
                     % (par.B0_ferraro_G,))
        if par.cnorm == 'rms_cmb':
            sys.exit("B0 'ferraro': use cnorm 'mag_energy' or a number")
    if getattr(par, 'B0_wound', 0):   # NEW (2026-10-30): the wound toroidal field (background_field.wound_profiles)
        if par.B0 is None or not par.diff_rot:
            sys.exit('B0_wound = 1 winds a toroidal field from the poloidal B0 with the differential rotation: it needs a poloidal B0 and diff_rot = 1')
        if not par.B0_wound_eta > 0:
            sys.exit('B0_wound_eta must be > 0 (the diffusivity that balances the winding, in units of MagnetD)')
        if par.ricb > 0 and par.innercore not in ('insulator', 'conducting, Chebys'):
            sys.exit('B0_wound = 1: innercore %r is not supported (insulator or conducting, Chebys)' % (par.innercore,))
        if par.mantle != 'insulator':   # NEW (2026-11-06): the wound field is solved with k = 0 at the outer wall
            sys.exit("B0_wound = 1 needs an insulating mantle: the wound field is solved with B_phi = 0 at the outer wall")
    if par.B0 == 'confined':   # NEW (2026-10-30)
        r1, r2 = par.B0_conf_r1, par.B0_conf_r2
        # FIX (2026-11-03): p >= 3, not 2: the momentum rows hold h''', which with p = 2 has a delta at the edges (h'' jumps)
        # that the profiles, evaluated point by point, would drop
        if par.B0_l not in (1, 2) or par.B0_conf_p < 3 or not (0 <= r1 < r2 <= 1):
            sys.exit("B0 'confined' needs B0_l = 1 or 2, B0_conf_p >= 3 and 0 <= B0_conf_r1 < B0_conf_r2 <= 1")
        if par.ricb > 0 and r1 < par.ricb:
            sys.exit("B0 'confined' with an inner core: the region must lie in the fluid, B0_conf_r1 >= ricb")
        if par.cnorm == 'rms_cmb':
            sys.exit("B0 'confined': cnorm 'rms_cmb' is not defined (B0 is zero at the CMB); use 'mag_energy' or a number")
    if par.B0 == 'FDM' and par.B0_l not in (1, 2):   # FIX (2026-10-28): the quadrupole couplings are implemented (operators.lorentz_pol2)
        sys.exit("B0 'FDM' with B0_l = %r: only B0_l = 1 or 2 is implemented" % par.B0_l)
    if not (isinstance(par.cnorm, (int, float, np.integer, np.floating)) or par.cnorm in ('rms_cmb', 'mag_energy', 'Schmitt2012')):
        sys.exit("cnorm %r: a number, 'rms_cmb', 'mag_energy' or 'Schmitt2012'" % (par.cnorm,))
    # FIX (2026-11-03): these two were not checked, and a misspelt value silently meant 'closed' or 'full sphere'
    if getattr(par, 'mag_couplings', 'closed') not in MAG_COUPLINGS:
        sys.exit('mag_couplings %r: set one of %s' % (par.mag_couplings, _names(MAG_COUPLINGS)))
    if par.B0_ic == 'full sphere':   # FIX (2026-11-06): renamed as in conductive_IC (1c9ba86); the index in params.dat is the same
        sys.exit("B0_ic = 'full sphere' is now called 'frozen currents' (h unchanged: the currents of the full-sphere field "
                 "below ricb stay in the inner core)")
    if par.B0_ic not in B0_ICS:
        sys.exit('B0_ic %r: set one of %s' % (par.B0_ic, _names(B0_ICS)))
    if par.ferraro_policy not in FERRARO_POLICIES:   # NEW (2026-11-03): the parameter that replaces background.FERRARO_POLICY
        sys.exit('ferraro_policy %r: set one of %s' % (par.ferraro_policy, _names(FERRARO_POLICIES)))
    if par.mantle not in MAGNETIC_WALLS:
        sys.exit('mantle %r: set one of %s' % (par.mantle, _names(MAGNETIC_WALLS)))
    if par.innercore not in INNER_CORES:
        sys.exit('innercore %r: set one of %s' % (par.innercore, _names(INNER_CORES)))
    if par.innercore == 'conducting, Chebys':
        if par.bci != 1:
            sys.exit("innercore 'conducting, Chebys' needs a no-slip inner boundary (bci = 1): the ICB matching assumes u = 0 there")
        if par.N_cic % 2 or par.N_cic < 4:
            sys.exit('N_cic must be even and >= 4, got %r' % par.N_cic)
    if par.B0_cmb not in B0_CMBS:   # NEW (2026-11-06)
        sys.exit('B0_cmb %r: set one of %s' % (par.B0_cmb, _names(B0_CMBS)))
    if par.mantle == 'resolved':   # NEW (2026-11-06): the resolved conducting mantle
        check_mantle()
    check_twa()   # NEW (2026-11-06): thin conducting walls
    if par.ricb == 0:   # NEW (2026-10-28): the full sphere
        if par.innercore != 'insulator':
            sys.exit("innercore %r needs ricb > 0" % (par.innercore,))
        if par.B0_tor is not None and par.B0_tor_l == 2 and par.B0_tor == 'Malkus':
            sys.exit("B0_tor 'Malkus' with B0_tor_l = 2 is not regular at r = 0 (k = r would need to go as r²): use 'wall' or 'user' with ricb = 0")
    if par.ViscosD == 0 and par.ricb > 0:   # NEW (2026-10-28): the inviscid path (bases, one bc row) exists for the full sphere only
        sys.exit('magnetic = 1 with ViscosD = 0 needs ricb = 0 (the inviscid formulation is for the full sphere only)')
    if par.forcing != 0:
        sys.exit('magnetic = 1 is implemented for eigenvalue problems only (forcing = 0)')
    # FIX (2026-11-03): an error of cnorm, cnorm_tor at the import of background_field that no check above explains (before
    # magnetic_diffrot, which uses them)
    for name, err in (('cnorm', bf.cnorm_error), ('cnorm_tor', bf.cnorm_tor_error)):
        if err is not None:
            sys.exit('%s could not be computed: %s: %s' % (name, type(err).__name__, err))
    if par.diff_rot:   # NEW (2026-10-30): the induction by the background flow, ∇×(U0×b), is in (diff_rot.induction_block)
        magnetic_diffrot()
    if par.Hendrik <= 0 or par.MagnetD <= 0:
        sys.exit('magnetic = 1 needs Le > 0 and Em > 0 (Hendrik = %r, MagnetD = %r: call set_scales() after setting them)'
                 % (par.Hendrik, par.MagnetD))


def _check_general(prof, name):
    '''NEW (2026-10-30): the degrees and expressions of a 'general' background field'''
    import background_field as bf
    if not isinstance(prof, dict) or not prof:
        sys.exit("B0 'general' needs %s = {n: 'numpy expression of r, p'}" % name)
    degs = list(prof)   # FIX (2026-11-03): sorted only once they are known to be integers (sorting mixed keys raised)
    if any(not isinstance(n, (int, np.integer)) or n < 1 for n in degs) or len(set(n % 2 for n in degs)) > 1:
        sys.exit('%s: the degrees must be integers >= 1 of one parity (one equatorial symmetry), got %r' % (name, degs))
    degs = sorted(degs)
    for n in degs:
        if not isinstance(prof[n], str) or not prof[n] or '#' in prof[n]:   # FIX (2026-11-03): they are written to params.dat
            sys.exit("%s[%d]: a numpy expression of r and p as a string without '#' (it is written to params.dat), got %r"
                     % (name, n, prof[n]))
        try:
            bf.gen_profile('h' if name == 'B0_profiles' else 'k', n, np.linspace(max(par.ricb, 0.1), 1, 4), 0)
        except Exception as err:
            sys.exit('%s[%d] = %r: %s: %s' % (name, n, prof[n], type(err).__name__, err))


def ferraro_mismatch():
    '''
    NEW (2026-10-30): max |B0⋅∇ΔΩ| / (max |B0| max |r ∇ΔΩ|) over the fluid, for the poloidal B0 and the differential rotation.
    A steady axisymmetric background needs ∇×(U0×B0) = r sinθ (B0⋅∇ΔΩ) e_φ = 0 (Ferraro's law of isorotation): ΔΩ constant
    along the field lines of the poloidal B0 (a toroidal B0 is always consistent: U0×B0 = 0). Kore's linear problem does not
    contain this term (it belongs to the background), so a mismatch means the background is assumed to be maintained by
    something the model leaves out. Evaluated from the shapes f_n (diff_rot.raw, r-derivative by a 4th-order difference) and
    the poloidal profiles (background_field.pol_profiles) on a grid; 0 without a poloidal B0.
    '''
    import diff_rot as dr
    import background_field as bf
    if par.B0 is None:
        return 0.0
    hd = 1e-3   # the step of the 4th-order difference of ∂ᵣΔΩ (error ~ hd⁴), inside the fluid
    r = np.linspace(max(par.ricb, 0.02) + 2*hd, 1.0 - 2*hd, 200)
    x, _ = np.polynomial.legendre.leggauss(48)
    sth = np.sqrt(1 - x**2)
    P = np.polynomial.legendre.Legendre
    d1 = lambda n: (-dr.raw(n, r + 2*hd) + 8*dr.raw(n, r + hd) - 8*dr.raw(n, r - hd) + dr.raw(n, r - 2*hd))/(12*hd)
    dO_r = sum(np.outer(d1(n), P.basis(n)(x)) for n in dr.degrees())
    dO_t = sum(np.outer(dr.raw(n, r), -sth*P.basis(n).deriv()(x)) for n in dr.degrees())   # ∂θ = -sinθ d/dx
    Br = Bt = 0
    for n0, f in bf.pol_profiles().items():   # NEW (2026-10-30): every degree (Y_n⁰ = P_n(cos θ))
        h0, h1 = f(r, 0), f(r, 1)
        Br = Br + np.outer(n0*(n0 + 1.)*h0/r, P.basis(n0)(x))
        Bt = Bt + np.outer(h1 + h0/r, -sth*P.basis(n0).deriv()(x))
    bdo = Br*dO_r + Bt*dO_t/r[:, None]
    scale = np.sqrt(Br**2 + Bt**2).max()*np.sqrt((dO_r*r[:, None])**2 + dO_t**2).max()
    return float(np.abs(bdo*r[:, None]).max()/scale) if scale > 0 else 0.0

def ferraro_report():
    '''
    NEW (2026-10-30): how well the assumptions of B0 'ferraro' hold (background_field: ψ = G(ΔΩ), h_n its projection):
    max |ψ| on the axis and at the outer wall, relative to max |ψ| (both should be ~0), and the relative error of ψ rebuilt
    from the degrees up to B0_ferraro_nmax (the truncation, which also sets max |B0⋅∇ΔΩ|).
    '''
    import background_field as bf
    r = np.linspace(max(par.ricb, 0.02), 1.0, 120)
    x, _ = np.polynomial.legendre.leggauss(64)
    psi = bf._ferraro_psi(r, x)
    big = max(np.abs(psi).max(), 1e-300)
    axis = np.abs(bf._ferraro_psi(r, np.array([-1., 1.]))).max()/big
    wall = np.abs(bf._ferraro_psi(np.array([1.0]), x)).max()/big
    P = np.polynomial.legendre.Legendre
    rebuilt = sum(np.outer(r*bf._ferraro_raw(n, r), (1 - x**2)*P.basis(n).deriv()(x)) for n in bf.ferraro_degrees())
    print("B0 'ferraro': G = %r, degrees 1..%d: max |ψ| on the axis %.1e, at r = 1 %.1e (of max |ψ|); ψ rebuilt from the degrees "
          'to %.1e' % (par.B0_ferraro_G, par.B0_ferraro_nmax, axis, wall, np.abs(rebuilt - psi).max()/big))

_warned = []
def magnetic_diffrot():
    '''
    NEW (2026-10-30): the checks of magnetic = 1 with diff_rot = 1:
      - a conducting inner core must co-rotate rigidly: ΔΩ(ricb, θ) = c (f_n(ricb) = 0 for n > 0, diff_rot.rigid_ricb); its
        field then gains the Doppler term -i m G c b and the ICB matching is unchanged; a latitude-dependent ΔΩ(ricb) would
        shear a solid core, which kore does not model; NEW (2026-11-06): likewise a thin conducting wall, at its radius, and
        the resolved mantle at r = 1;
      - a poloidal B0 and ΔΩ should satisfy Ferraro's law (ferraro_mismatch); par.ferraro_policy says whether a mismatch above
        1e-7 stops the run ('refuse') or only prints a warning ('warn', the default: the linear problem then assumes the
        background is maintained). NEW (2026-11-03): a parameter, formerly the constant background.FERRARO_POLICY.
    '''
    import diff_rot as dr
    if par.innercore == 'conducting, Chebys':
        c, ok = dr.rigid_ricb()
        if not ok:
            sys.exit('magnetic = 1, diff_rot = 1 with a conducting inner core needs a rigid rotation of the inner boundary '
                     '(f_n(ricb) = 0 for every n > 0): ΔΩ(ricb, θ) varies with latitude')
    # NEW (2026-11-06): a thin conducting wall co-rotates with ΔΩ at its radius, which must then be uniform in latitude; in the
    # frame of the wall its rows are those without the rotation (U0×b at the wall involves only b_r, which is continuous).
    # Likewise the resolved mantle (mantle = 'resolved'), whose field then gains the Doppler term -i m G ΔΩ(1) b
    for on, r, name in ((par.mantle in ('TWA', 'resolved'), 1.0, 'outer'),
                        (par.innercore == 'TWA' and par.ricb > 0, par.ricb, 'inner')):
        if on and not dr.rigid_at(r)[1]:
            sys.exit("magnetic = 1, diff_rot = 1 with a conducting wall (%r) at the %s boundary needs a rigid rotation there "
                     "(f_n = 0 for every n > 0): ΔΩ varies with latitude at r = %g"
                     % (par.mantle if name == 'outer' else par.innercore, name, r))
    mis = ferraro_mismatch()
    if par.B0 == 'ferraro' and not _warned:
        ferraro_report()
    import background_field as bf
    if bf.wound_on():   # NEW (2026-10-30): B0⋅∇Ω ≠ 0 is the point here: report how well the wound field balances it instead
        if not _warned:
            print('B0_wound: toroidal degrees %s; steady induction residual max |η(∇² - 1/s²)B_φ + G s B_p⋅∇ΔΩ| / max |G s B_p⋅∇ΔΩ| = %.2e'
                  % (sorted(bf.wound_profiles()), bf.wound_residual()))
        _warned.append(1)
        return
    if par.B0 is not None and not _warned:   # the report, once per process
        print('Ferraro check (B0 %r, diff_rot_type %r): max |B0⋅∇ΔΩ| r / (max |B0| max |r∇ΔΩ|) = %.2e' % (par.B0, par.diff_rot_type, mis))
    if mis > 1e-7:
        msg = ('the poloidal B0 and the differential rotation are not a steady background together: max |B0⋅∇ΔΩ| r = '
               '%.2e of max |B0| max |r∇ΔΩ| (Ferraro); the linear problem assumes the background is maintained' % mis)
        if par.ferraro_policy == 'refuse':
            sys.exit(msg + " (ferraro_policy = 'refuse')")
        if not _warned:
            print('WARNING: ' + msg)
    _warned.append(1)


def description():
    '''the 'user' expressions as text for params.dat (a '#' comment), empty if there are none'''
    out = []
    if par.magnetic and par.B0_tor == 'user':   # NEW (2026-10-28)
        out.append('B0_tor: k = %s ; p = %r' % (par.B0_tor_profile, par.B0_tor_param))
    if par.magnetic:   # NEW (2026-11-03): the expressions of the 'general' and 'ferraro' fields
        if par.B0 == 'general':
            out.append('B0: ' + ' ; '.join('h%d = %s' % (n, par.B0_profiles[n]) for n in sorted(par.B0_profiles))
                       + ' ; p = %r' % par.B0_param)
        if par.B0 == 'ferraro':
            out.append('B0: G = %s ; p = %r' % (par.B0_ferraro_G, par.B0_param))
        if par.B0_tor == 'general':
            out.append('B0_tor: ' + ' ; '.join('k%d = %s' % (n, par.B0_tor_profiles[n]) for n in sorted(par.B0_tor_profiles))
                       + ' ; p = %r' % par.B0_tor_param)
    if par.magnetic and par.mantle == 'resolved' and isinstance(par.sigma_mantle, str):   # NEW (2026-11-06)
        out.append('sigma_mantle: %s ; p = %r' % (par.sigma_mantle, par.sigma_mantle_param))
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
    '''ν/ViscosD (which = 'nu') or κ/ThermaD ('kappa') at r, even in r, times (ρ0(aux3)/ρ0)**visc_pow (therm_pow);
    η/MagnetD ('eta') is 1 for now, whatever the diffusivity type'''
    if which == 'eta':  # NEW (2026-10-28): the hook for a magnetic diffusivity profile (the operators have η and η')
        return np.ones_like(r)
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


# ---------------------------------------------------------------------------------- NEW (2026-11-04): the centre, ricb = 0

def _on(f, r):
    '''f at the radii r, as an array of the shape of r (a profile can be a constant, e.g. a 'step' diffusivity without a step)'''
    return np.asarray(f(r), dtype=float) + np.zeros(np.shape(r))

def irregular_part(f, n, scale, r0=0.02, J=30, M=100):
    '''
    How far the radial profile f(r) is from regular at r = 0, as fractions of scale: (L, K). WHY: in a full sphere a
    background term of degree n (f(r) P_n(cos θ), the poloidal or toroidal scalar of a field of degree n, or n = 1 for the
    radial vectors g and dS/dr) is smooth at the centre only if f is r^n times a series in r². kore evaluates the profiles
    at |r| with the parity (-1)^n, which keeps the parity but neither removes the powers below r^n (the modes then have them
    too: they are not regular at r = 0) nor smooths a kink of f(|r|) (the series in N then converge slowly); see
    docs/numerical-method.md, Regularity at the centre. HOW: the largest residuals of least-squares fits to f at M Chebyshev
    points of (0, r0) of r^n Σ_j c_j T_2j(r/r0), j < J (L), and of the same with r^(n mod 2) in place of r^n, a series of the
    parity of n (K, the kink). The fits resolve far finer scales than kore's basis (erf(r/0.005) gives round-off), so a power
    below r^n counts with its full size, while a kink counts only as far as the fit cannot follow it (4(1 - r) r: 8e-4).
    '''
    s = 0.5*(1 + np.cos(np.pi*(np.arange(M) + 0.5)/M))
    y = _on(f, r0*s)
    T = np.polynomial.chebyshev.chebvander(s, 2*J - 2)[:, 0::2]
    res = lambda V: np.abs(y - V @ np.linalg.lstsq(V, y, rcond=None)[0]).max()/scale
    return res(s[:, None]**n*T), res(s[:, None]**(n % 2)*T)

def _form(n):
    return 'a series in r²' if n == 0 else ('r' if n == 1 else 'r^%d' % n) + ' times a series in r²'

def centre_check(prf):
    '''
    A WARNING for each background profile that is not regular at r = 0 (called by submatrices.py when ricb = 0), from its
    irregular_part relative to the largest profile of its family (the degrees of ΔΩ, of S_tw, of the poloidal and of the
    toroidal B0) or to its own maximum (ln ρ0, p0, g, dS/dr, ν, κ): a kink above 1e-8 (the series converge slowly in N),
    and powers below r^n above 1e-8 and ten times the kink (the modes are not regular there); prf is the structure
    (radial_profiles.prf). The rotation types that are irregular by construction get a message of their own, and their
    S_tw is not checked again. Returns the number of warnings.
    '''
    import diff_rot as dr
    import background_field as bf
    rg = np.linspace(0, 1, 401)[1:]
    nwarn = [0]
    def family(profiles, ref):   # profiles: [(name, n, f)]; the scale is the largest |f| of them all
        scale = max([np.abs(_on(f, rg)).max() for _, _, f in profiles] + [0.])
        bad = False
        for name, n, f in profiles:
            L, K = irregular_part(f, n, scale) if scale > 0 else (0., 0.)
            # (with the absolute floor of the parity check in submatrices.py: a family that vanishes identically, e.g. the S_tw
            # of the 'cylindrical' rotation, is all round-off)
            if K > 1e-8 and K*scale > 1e-13:
                print('WARNING: %s is not smooth at r = 0: within r < 0.02 its %s extension to r < 0 differs from %s by %.1e of '
                      '%s (a kink or a jump), so the full sphere (ricb = 0) converges slowly in N' %
                      (name, 'odd' if n % 2 else 'even', _form(n % 2), K, ref))
                bad = True; nwarn[0] += 1
            if L > max(1e-8, 10*K) and L*scale > 1e-13:
                print('WARNING: %s is not regular at r = 0: within r < 0.02 it differs from %s by %.1e of %s, so in the full '
                      'sphere (ricb = 0) the modes are not regular there either' % (name, _form(n), L, ref))
                bad = True; nwarn[0] += 1
        return bad
    rot_bad = False
    if par.diff_rot:
        t = par.diff_rot_type
        if t in ('Y20', 'conical'):
            print("WARNING: diff_rot_type %r is not regular at r = 0 (its P2 term does not vanish there as r²): in the full "
                  'sphere (ricb = 0) the modes are not regular at the centre either' % t)
            rot_bad = True; nwarn[0] += 1
        elif t == 'Y20-wall-bounded' or (t == 'shellular' and not (par.diff_rot_param % 2 == 0 and par.diff_rot_param > 0)):
            # ΔΩ is used at |r| (diff_rot.fX): (1 - r) r and r**σ - 1 have a kink at r = 0
            print('WARNING: diff_rot_type', t, 'is not smooth at r = 0, the full sphere (ricb = 0) converges slowly in N')
            rot_bad = True; nwarn[0] += 1
        else:
            rot_bad = family([('f_%d of the rotation (diff_rot_type %r)' % (n, t), n, lambda r, n=n: dr.raw(n, r))
                              for n in dr.degrees()], 'max |f_n|')
        if par.thermal and (par.thermal_wind == 3 or (par.thermal_wind in (1, 2) and not rot_bad)):
            family([('σ_%d of the thermal wind (thermal_wind %d)' % (n, par.thermal_wind), n, lambda r, n=n: dr.sigma(n, r, 0))
                    for n in dr.tw_degrees()], 'max |σ_n|')
    if par.magnetic:
        pol, tor = bf.raw_profiles()
        family([('h_%d of B0 %r' % (n, par.B0), n, f) for n, f in pol.items()], 'max |h_n|')
        what = ', '.join(([] if par.B0_tor is None else ['B0_tor %r' % (par.B0_tor,)]) + (['B0_wound'] if bf.wound_on() else []))
        family([('k_%d of the toroidal field (%s)' % (n, what), n, f) for n, f in tor.items()], 'max |k_n|')
    singles = [('ln ρ0', 0, lambda r: prf.logrho(r, 0), 'structure'), ('p0', 0, lambda r: prf.pressure(r, 0), 'structure'),
               ('g', 1, lambda r: prf.gravity(r, 0), 'structure')]
    if par.thermal:
        singles += [('dS/dr', 1, prf.gradS, 'stratification')]
    if par.ViscosD > 0:
        singles += [('ν', 0, lambda r: diffusivity(prf, r, 'nu'), 'diffusivity')]
    if par.thermal and par.ThermaD > 0:
        singles += [('κ', 0, lambda r: diffusivity(prf, r, 'kappa'), 'diffusivity')]
    for name, n, f, kind in singles:
        family([('%s (%s %r)' % (name, kind, getattr(par, kind)), n, f)], 'max |%s|' % name)
    if par.centrifugal:   # NEW (2026-11-05): the flattening of the level surfaces must be even and smooth at r = 0
        import centrifugal as cf
        family([("the flattening f (centrifugal_shape %r)" % par.centrifugal_shape, 0, cf.flattening)], 'max |f|')
    return nwarn[0]
