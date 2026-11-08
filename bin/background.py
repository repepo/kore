'''
The background besides the rotation (diff_rot.py), each part chosen by name in parameters.py (params_default.py):
    structure       ρ0, p0 and g: a class of radial_profiles.py
    stratification  dS/dr, S = δs/c_p (thermal = 1); NEW (2026-11-08) comp_stratification, dC/dr (compositional = 1)
    diffusivity     ν/ViscosD, κ/ThermaD and (NEW 2026-11-08) D/CompoD
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
STRATIFICATIONS = ('top hat', 'odd top hat', 'solar_rz', 'linear', 'inverse square', 'user', 'two zone')   # ('two zone': NEW
# 2026-11-08, conductive_IC's)
COMP_STRATIFICATIONS = ('linear', 'inverse square', 'two zone', 'user')   # NEW (2026-11-08): comp_stratification, dC/dr
DIFFUSIVITIES = ('uniform', 'step', 'solar_rz', 'user')
MAGNETIC_WALLS = ('insulator', 'TWA', 'resolved')   # NEW (2026-10-28): mantle (magnetic = 1); B0 types: background_field.B0_TYPES
INNER_CORES = ('insulator', 'conducting, Chebys', 'TWA', 'conducting, Bessel')   # NEW (2026-10-28): innercore (magnetic = 1);
# 'TWA' and 'conducting, Bessel' (forced problems only) NEW 2026-11-06; params.dat writes the index
CONDUCTING_CORES = ('conducting, Chebys', 'conducting, Bessel')   # NEW (2026-11-06): the inner cores of finite conductivity
B0_CMBS = ('wall', 'plain')   # NEW (2026-11-06): B0_cmb (magnetic = 1, mantle = 'TWA'), as written to params.dat
MAG_COUPLINGS = ('closed', 'algebra')            # NEW (2026-11-03): mag_couplings, B0_ic and ferraro_policy (magnetic = 1),
B0_ICS = ('potential', 'frozen currents')        # also written to params.dat as these indices ('frozen currents' was 'full sphere')
FERRARO_POLICIES = ('warn', 'refuse')
FORCINGS = (1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14)   # NEW (2026-11-08): the forced problems of assemble.py (0: eigenvalue problems)
# NEW (2026-11-08): the forcings of conductive_IC that were not ported, and what to use instead (check_forcing stops on them)
NOT_PORTED = {
    3: "conductive_IC's body force of Rovira-Navarro et al. (2019), whose rows were written in the wrong basis there, is not "
       "ported: forcing 14 imposes the radial motion of the walls of a tide",
    4: "conductive_IC's test case of forcing 1's force times constants of the Enceladus fits is not ported: use forcing 1",
    5: "conductive_IC's test case of the Coriolis force on grad(Y_2^0/r) is not ported: forcing 1 with m = 0 is the "
       "Coriolis force on the equilibrium tide",
}

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
    if par.compositional not in (0, 1):   # NEW (2026-11-08)
        sys.exit('compositional %r: 1 (the composition is solved) or 0' % (par.compositional,))
    if par.hydro == 0 and not (par.magnetic or par.thermal or par.compositional):
        sys.exit('hydro = 0 needs magnetic = 1, thermal = 1 or compositional = 1: without the flow nothing else is solved')
    if par.hydro == 0 and par.forcing != 0:
        sys.exit('hydro = 0 is for eigenvalue problems (forcing = 0): the free decay of the field, of the entropy or of the '
                 'composition')
    if par.forcing != 0:   # NEW (2026-11-08)
        check_forcing()
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
        if par.stratification == 'two zone':   # NEW (2026-11-08)
            _check_two_zone('two_zone')
    if par.compositional:   # NEW (2026-11-08)
        check_composition()
    if par.magnetic:
        check_magnetic()
    if par.centrifugal:   # NEW (2026-11-05)
        check_centrifugal()
    if par.diffusivity not in DIFFUSIVITIES:
        sys.exit('diffusivity %r: set one of %s' % (par.diffusivity, _names(DIFFUSIVITIES)))
    if par.diffusivity == 'user':
        # NEW (2026-11-08): and 'D', the composition's, with compositional = 1 (allowed, and unused, without it)
        need = ['kappa', 'nu'] + (['D'] if par.compositional else [])
        if not isinstance(par.diffusivity_profiles, dict) or not set(need) <= set(par.diffusivity_profiles) \
                or not set(par.diffusivity_profiles) <= {'nu', 'kappa', 'D'}:
            sys.exit("diffusivity 'user' needs diffusivity_profiles = {'nu': '...', 'kappa': '...'%s}" %
                     (", 'D': '...'" if par.compositional else " (and 'D': '...' with compositional = 1)"))
        for k in sorted(par.diffusivity_profiles):
            _check_expression(par.diffusivity_profiles[k], par.diffusivity_param, "diffusivity_profiles['%s']" % k)
    if par.rotdyn not in (0, 1):   # NEW (2026-11-06): the rotating solids
        sys.exit('rotdyn %r: 0 (the mantle and the inner core turn with the frame) or 1 (they turn under the torques)' % (par.rotdyn,))
    if par.rotdyn:
        check_rotdyn()
    check_drag()   # NEW (2026-11-06): the drag at the outer wall (kappa_cmb, topo_drag)


def _check_two_zone(name):
    '''NEW (2026-11-08): the 'two zone' step [rc, h] of the entropy (two_zone) or of the composition (comp_two_zone)'''
    v = getattr(par, name)
    try:
        rc, h = [float(x) for x in v]
    except (TypeError, ValueError):
        sys.exit("%s = %r: [rc, h], the radius of the step of the 'two zone' gradient and its width" % (name, v))
    if not (np.isfinite(rc) and np.isfinite(h) and h > 0):
        sys.exit("%s = %r: [rc, h] with a width h > 0" % (name, v))


def check_composition():
    '''
    NEW (2026-11-08, from conductive_IC): stops on the settings that the composition (compositional = 1) cannot take: a
    gradient type that does not exist, 'inverse square' without an inner core (1/r² is singular at r = 0), an expression
    that does not evaluate, a negative diffusion factor.
    '''
    if par.comp_stratification not in COMP_STRATIFICATIONS:
        sys.exit('comp_stratification %r: set one of %s' % (par.comp_stratification, _names(COMP_STRATIFICATIONS)))
    if par.comp_stratification == 'inverse square' and par.ricb == 0:
        sys.exit("comp_stratification 'inverse square' needs ricb > 0")
    if par.comp_stratification == 'user':
        _check_expression(par.comp_stratification_profile, par.comp_stratification_param, 'comp_stratification_profile')
    if par.comp_stratification == 'two zone':
        _check_two_zone('comp_two_zone')
    for name in ('CompoD', 'BeyonceC'):
        v = getattr(par, name)
        if not (isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and np.isfinite(v)):
            sys.exit('%s = %r: a finite number' % (name, v))
    if par.CompoD < 0:
        sys.exit('CompoD = %r: the compositional diffusion factor is >= 0 (0: no diffusion of the composition)' % (par.CompoD,))
    if par.bci_compositional not in (0, 1) or par.bco_compositional not in (0, 1):
        sys.exit('bci_compositional, bco_compositional: 0 (fixed composition) or 1 (zero flux), got %r, %r'
                 % (par.bci_compositional, par.bco_compositional))


def check_forcing():
    '''
    NEW (2026-11-08): stops on a forcing that does not exist (assemble.py would write no right-hand side, and solve.py would
    stop on the missing B_forced.npz), and on the settings that forcing 12, the steady rotation of the walls (from
    conductive_IC), does not cover. Forcing 12 drives an axisymmetric, equatorially symmetric flow (m = 0, symm = 1) through
    the viscosity of the fluid, steadily (forcing_frequency = 0). A wall that turns must be no-slip: a stress-free wall
    does not carry the fluid (and the rows of a fluid slipping along a conducting solid hold the solid at rest). The walls
    stay in place (bc 0 or 1): a steady flow through a wall of zero radial stress would move it without end. With
    thermal = 1 it needs ThermaD > 0: without diffusion the steady entropy equation, u_r dS/dr = 0, leaves the entropy
    undetermined, and (NEW 2026-11-08) with compositional = 1 CompoD > 0 for the same reason. Forcings 13 and 14 check
    their settings in assemble.py; forcings 1, 2 and 8 check_tides.
    '''
    if par.forcing in NOT_PORTED:
        sys.exit('forcing %d: %s' % (par.forcing, NOT_PORTED[par.forcing]))
    if par.forcing not in FORCINGS:
        sys.exit('forcing %r: 0 (an eigenvalue problem) or one of the forced problems %s' % (par.forcing, _names(FORCINGS)))
    if par.forcing in (1, 2, 8):
        check_tides()
    if par.forcing != 12:
        return
    if par.m != 0 or par.symm != 1:
        sys.exit('forcing 12 needs m = 0 and symm = 1: walls that turn about z drive an axisymmetric, symmetric flow')
    if par.forcing_frequency != 0:
        sys.exit('forcing 12 is a steady rotation of the walls: it needs forcing_frequency = 0 (got %r). Walls that turn back '
                 'and forth are the libration, forcing 7' % (par.forcing_frequency,))
    if par.ViscosD == 0:
        sys.exit('forcing 12 needs a viscous fluid (ViscosD > 0): the walls drive the fluid by its viscosity')
    if par.ricb == 0 and par.forcing_amplitude_icb != 0:
        sys.exit('forcing 12 in a full sphere (ricb = 0) has no inner core to turn: set forcing_amplitude_icb = 0')
    if par.thermal and par.ThermaD == 0:
        sys.exit('forcing 12 with thermal = 1 needs ThermaD > 0: without diffusion the steady entropy equation, '
                 'u_r dS/dr = 0, leaves the entropy undetermined')
    if par.compositional and par.CompoD == 0:   # NEW (2026-11-08)
        sys.exit('forcing 12 with compositional = 1 needs CompoD > 0: without diffusion the steady composition equation, '
                 'u_r dC/dr = 0, leaves the composition undetermined')
    walls = [('outer', 'bco', 'forcing_amplitude_cmb')] + ([('inner', 'bci', 'forcing_amplitude_icb')] if par.ricb > 0 else [])
    for wall, bc, rate in walls:
        if getattr(par, bc) not in (0, 1):
            sys.exit('forcing 12 needs walls that stay in place, %s = 0 or 1 (got %r): a steady flow through a wall of zero '
                     'radial stress would move it without end' % (bc, getattr(par, bc)))
        if getattr(par, rate) != 0 and getattr(par, bc) != 1:
            sys.exit('forcing 12: the %s wall turns (%s = %r) but is stress-free (%s = 0), so it does not carry the fluid. '
                     'Make it no-slip (%s = 1), or set %s = 0' % (wall, rate, getattr(par, rate), bc, bc, rate))
    if all(getattr(par, rate) == 0 for wall, bc, rate in walls):
        sys.exit('forcing 12 needs a wall that turns: forcing_amplitude_cmb or forcing_amplitude_icb other than 0')


def check_tides():
    '''
    NEW (2026-11-08, from conductive_IC): stops on the settings that forcings 1, 2 and 8 do not cover. Each has a frequency
    (forcing_frequency != 0).
      1, the wave-like tide (Lin & Ogilvie 2018): (m, symm) = (0, 1), (1, -1) or (2, 1); the equilibrium tide ∇(X Y₂ᵐ) is a
         potential flow, solenoidal and with no viscous force, only for a uniform density ('Boussinesq'), with no
         stratification to displace (thermal = 0, compositional = 0) and no field to induce (magnetic = 0); it moves the
         outer wall radially,
         so the walls of the flow solved for are impenetrable (bco, bci 0 or 1) and the drag at the outer wall, which would
         act on the slip of the whole tide, is left out. For m = 2 the Coriolis force on it is a gradient in a full sphere,
         which drives nothing: ricb > 0.
      2, the eccentricity tide of Enceladus (Rekier et al. 2019): m = 0 or 2 and symm = 1, no-slip walls (bci = bco = 1,
         which the fits move radially and along themselves) and ricb from 16/21 to 1, where the fits hold
         (background.check_forced_magnetic: no conducting solid behind the walls).
      8, the libration in the frame of the mantle: m = 0, symm = 1; the walls are solids that stay in place (bco, bci 0 or
         1: the mantle, and the inner core that turns with it); the inner core's own libration relative to the mantle
         (forcing_amplitude_icb) needs an inner core.
    An inviscid fluid (ViscosD = 0, forcings 1 and 8) needs a full sphere: kore's inviscid formulation (its bases and its one
    wall row) is for ricb = 0 only.
    '''
    f = par.forcing
    if par.forcing_frequency == 0:
        sys.exit('forcing %d needs forcing_frequency != 0: it is a periodic forcing' % f)
    if par.ViscosD == 0 and par.ricb > 0:
        sys.exit('forcing %d with an inviscid fluid (ViscosD = 0) needs ricb = 0: the inviscid formulation is for the full '
                 'sphere only' % f)
    if f == 1:
        if (par.m, par.symm) not in ((0, 1), (1, -1), (2, 1)):
            sys.exit('forcing 1 (the wave-like tide) needs (m, symm) = (0, 1), (1, -1) or (2, 1), the degree-2 tide')
        if par.m == 2 and par.ricb == 0:
            sys.exit('forcing 1 with m = 2 needs an inner core (ricb > 0): in a full sphere the Coriolis force on the '
                     'equilibrium tide r² Y_2^2 is a gradient, which drives no flow')
        if par.structure != 'Boussinesq' or par.thermal or par.compositional or par.magnetic:
            sys.exit("forcing 1 (the wave-like tide) needs structure 'Boussinesq', thermal = 0, compositional = 0 and "
                     "magnetic = 0: its equilibrium tide is the potential flow of a uniform fluid with no stratification "
                     "and no field")
        if par.bco not in (0, 1) or (par.ricb > 0 and par.bci not in (0, 1)):
            sys.exit('forcing 1 needs impenetrable walls (bco, bci = 0 or 1): the radial motion of the outer wall is that of '
                     'the equilibrium tide')
        if par.kappa_cmb != 0 or par.topo_drag is not None:
            sys.exit('forcing 1 with a drag at the outer wall (kappa_cmb, topo_drag) is not covered: the drag would act on '
                     'the slip of the whole tide, the equilibrium tide included')
    elif f == 2:
        if (par.m, par.symm) not in ((0, 1), (2, 1)):
            sys.exit('forcing 2 (the eccentricity tide of Enceladus) needs m = 0 or 2 and symm = 1')
        if not 16/21 <= par.ricb < 1:
            sys.exit('forcing 2: the fits of the tide hold for ricb from 16/21 = %.4f to 1 (got %r)' % (16/21, par.ricb))
        if par.bco != 1 or par.bci != 1 or par.ViscosD == 0:
            sys.exit('forcing 2 needs no-slip walls (bci = bco = 1, ViscosD > 0): the tide moves them radially and along '
                     'themselves')
    elif f == 8:
        if par.m != 0 or par.symm != 1:
            sys.exit('forcing 8 (the libration in the frame of the mantle) needs m = 0 and symm = 1')
        if par.bco not in (0, 1) or (par.ricb > 0 and par.bci not in (0, 1)):
            sys.exit('forcing 8 needs walls that stay in place (bco, bci = 0 or 1): they are the mantle that librates and the '
                     'inner core that turns with it')
        if par.ricb == 0 and par.forcing_amplitude_icb != 0:
            sys.exit('forcing 8 in a full sphere (ricb = 0) has no inner core to librate relative to the mantle: set '
                     'forcing_amplitude_icb = 0')


def check_drag():
    '''
    NEW (2026-11-06, conductive_IC's check_drag): stops on the settings that the drag at the outer wall (kappa_cmb, topo_drag;
    topo.py) does not cover. The drag replaces the stress-free condition (bco = 0) by a Robin condition on the viscous stress,
    in units of the rotation rate (K/(rho_w Omega r0)). Left out: a wall that moves radially (forcings 9 and 10), a deformed
    body (the drag on the metric of the walls), and for topo_drag = 'rekier' a differential rotation (the frequency of the
    flow relative to the wall would depend on latitude) and a frequency scan of forcing 11 without topo_omega (the drag is
    built for one frequency). Only parameters are read here (radial_profiles.py calls check() before it builds its
    profiles).
    '''
    num = (int, float, np.integer, np.floating)
    def number(v, ok=lambda v: True):
        return isinstance(v, num) and not isinstance(v, bool) and np.isfinite(v) and ok(v)
    if par.topo_drag not in (None, 'rekier'):
        sys.exit("topo_drag = %r: None (no topographic drag) or 'rekier'" % (par.topo_drag,))
    if not number(par.kappa_cmb, lambda v: v >= 0):
        sys.exit('kappa_cmb = %r: the drag coefficient is a finite number >= 0 (0: stress-free)' % (par.kappa_cmb,))
    if par.kappa_cmb == 0 and par.topo_drag is None:
        return
    if par.topo_drag is not None and par.kappa_cmb != 0:
        sys.exit('topo_drag and kappa_cmb cannot be combined: set kappa_cmb = 0 with topo_drag')
    if not (par.hydro and par.ViscosD > 0 and par.bco == 0):
        sys.exit('the drag at the outer wall (kappa_cmb, topo_drag) needs the flow (hydro = 1), a viscous fluid (ViscosD > 0) '
                 'and a stress-free outer wall (bco = 0), whose condition it changes')
    if not par.Gaspard > 0:
        sys.exit('the drag at the outer wall needs a rotating frame (Gaspard > 0): it is in units of the rotation rate')
    if par.centrifugal:
        sys.exit('the drag at the outer wall needs a spherical body (centrifugal = 0)')
    if par.forcing in (9, 10):
        sys.exit('the drag at the outer wall with forcing %d is not covered: the wall moves radially' % par.forcing)
    if par.topo_drag is None:
        return
    if par.diff_rot:
        sys.exit("topo_drag = 'rekier' with a differential rotation (diff_rot = 1) is not covered: the frequency of the flow "
                 "relative to the wall would depend on latitude")
    for name in ('topo_N', 'topo_omega'):
        v = getattr(par, name)
        if v is not None and not number(v, lambda v: v >= 0):
            sys.exit('%s = %r: None (from the run) or a number >= 0, in units of the rotation rate' % (name, v))
    if not isinstance(par.topo_K, str):
        sys.exit('topo_K = %r: a numpy expression of x (a test hook), or the empty string' % (par.topo_K,))
    if par.topo_K:
        x = np.linspace(-1, 1, 101)
        try:
            kt, kp = eval(par.topo_K, {'np': np, '__builtins__': {}}, {'x': x})
            kk = np.array([np.broadcast_to(np.asarray(kt, dtype=float), x.shape),
                           np.broadcast_to(np.asarray(kp, dtype=float), x.shape)])
        except Exception as e:
            sys.exit('topo_K = %r: not a pair of numpy expressions of x (%s)' % (par.topo_K, e))
        if not (np.all(np.isfinite(kk)) and np.all(kk >= 0)):
            sys.exit('topo_K = %r: kappa_tt and kappa_pp must be finite and >= 0' % (par.topo_K,))
        return
    if not (number(par.topo_hrms, lambda v: v > 0) and number(par.topo_lambda0, lambda v: v > 0)
            and number(par.topo_H, lambda v: v > 0.5)):
        sys.exit("topo_drag = 'rekier' needs topo_hrms > 0, topo_lambda0 > 0 and topo_H > 1/2 (eq. B2), got %r, %r, %r"
                 % (par.topo_hrms, par.topo_lambda0, par.topo_H))
    w = par.topo_omega if par.topo_omega is not None else (par.forcing_frequency if par.forcing != 0 else par.itau)
    if w == 0:
        sys.exit("topo_drag = 'rekier' needs a frequency of the flow other than 0 (the drag goes as 1/w0): set itau, the "
                 "forcing frequency or topo_omega")
    if par.forcing == 11 and getattr(par, 'rsf_frequencies', None) is not None and par.topo_omega is None:
        sys.exit("topo_drag = 'rekier' with a frequency scan (rsf_frequencies) needs topo_omega: the drag is built for one "
                 "frequency of the flow, not for each frequency of the scan")


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
    if (par.magnetic and par.hydro and par.ricb > 0 and par.innercore == 'conducting, Chebys' and par.B0 is not None
            and par.bci == 0):   # NEW (2026-11-07): the slipping fluid of utils.cic_slips
        sys.exit('centrifugal = 1 with a conducting inner core and a poloidal B0 needs a no-slip inner wall (bci = 1): the '
                 'u×B0 of a fluid slipping along the deformed inner boundary has metric terms that are not implemented')
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
    round-off tail, which the derivative would amplify (0 for a uniform layer, a number). FIX (2026-11-07): the interpolant
    by a DCT (background_field.cheb_interp), whose round-off stays below the cut at any N_mantle
    '''
    import background_field as bf
    if not isinstance(par.sigma_mantle, str):
        return np.zeros(np.shape(r))
    if not _dlnsigma_m:
        lns = bf.cheb_interp(lambda r: np.log(sigma_mantle(r)), max(128, 4*par.N_mantle), [1.0, par.r_surface])
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
    field at the wall), a spherical body (centrifugal = 0) and bco = 0 or 1 (with bco = 2 the fluid crosses the wall, and the
    rows follow a radial flow there only where a forcing moves the wall: forced_walls).
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
        sys.exit("mantle = 'resolved' needs bco = 0 or 1: with zero radial stress at the outer wall (bco = 2) the fluid "
                 "crosses it, and the mantle's rows follow a radial flow there only where a forcing moves the wall "
                 "(forced_walls)")


def check_twa():
    '''
    NEW (2026-11-06): stops on thin conducting walls (mantle or innercore 'TWA', from conductive_IC) outside what the wall rows
    of assemble.bc_b_spherical cover. Their parameters must be numbers: c, c1 >= 0, muw > 0 (np.inf allowed), mu > 0. The fluid's
    permeability mu enters the thin-wall conditions only, so mu != 1 needs a thin wall (or a conducting inner core) at every
    wall of the fluid: an insulating wall with mu != 1 is a thin wall with c = c1 = 0. A thin wall needs bco, bci = 0 or 1
    (with 2 the fluid crosses the wall, and the rows follow a radial flow there only where a forcing moves the wall:
    forced_walls) and a spherical body (centrifugal = 0).
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
    if par.mu != 1 and not (twa_o and (par.ricb == 0 or par.innercore in ('TWA',) + CONDUCTING_CORES)):
        sys.exit("mu = %r enters only the thin-wall conditions: every wall of the fluid must then have a thin conducting "
                 "layer ('TWA'; an insulating wall with mu != 1 is a layer with c = c1 = 0) or a conducting inner core"
                 % (par.mu,))
    if not (twa_o or twa_i):
        return
    if par.centrifugal:
        sys.exit("a thin conducting layer ('TWA') needs a spherical body (centrifugal = 0)")
    if par.hydro and ((twa_o and par.bco == 2) or (twa_i and par.bci == 2)):
        sys.exit("a thin conducting layer ('TWA') needs bco, bci = 0 or 1 at its wall: with zero radial stress (2) the fluid "
                 "crosses the wall, and the layer's rows follow a radial flow there only where a forcing moves the wall "
                 "(forced_walls)")


def check_rotdyn():
    '''
    NEW (2026-11-06, conductive_IC's check_rotdyn): stops on the settings that the rotating solids (rotdyn = 1) do not cover.
    The solids drag the fluid through the no-slip walls, where the toroidal flow of degree 1 is r omega (assemble.py), and
    their torques act on that flow and on the toroidal field next to the walls, which hold the axial parts only for m = 0,
    symm = 1 (rotdyn.py). NEW (2026-11-06, as conductive_IC): the outer wall may also be stress-free (bco = 0), the mantle
    then pulling on the fluid through the drag (kappa_cmb, topo_drag; topo.py) or not at all, and magnetically. Left out: a
    stress-free inner wall, the stresses of a differential rotation and of a toroidal B0 at the walls (its current sheet
    there), a deformed body (spherical torques), and Gaspard = 0 (gTorque and OmgtauIC are in units of the rotation rate).
    '''
    import background_field as bf
    if not (par.hydro and par.m == 0 and par.symm == 1 and par.ricb > 0 and par.forcing == 0):
        sys.exit('rotdyn = 1 needs the flow (hydro = 1), m = 0, symm = 1, an inner core (ricb > 0) and an eigenvalue problem '
                 '(forcing = 0)')
    if not (par.ViscosD > 0 and par.bci == 1 and par.bco in (0, 1)):
        sys.exit('rotdyn = 1 needs a viscous fluid (ViscosD > 0) with a no-slip inner wall (bci = 1), and an outer wall that is '
                 'no-slip or stress-free (bco = 1 or 0): the mantle and the inner core drag the fluid through them')
    if not par.Gaspard > 0:
        sys.exit('rotdyn = 1 needs a rotating frame (Gaspard > 0): gTorque and OmgtauIC are in units of the rotation rate')
    if par.diff_rot:
        sys.exit('rotdyn = 1 with a differential rotation (diff_rot = 1) is not covered: the solids would turn with ΔΩ at the '
                 'walls, and the stresses of the background flow are left out')
    if par.centrifugal:
        sys.exit('rotdyn = 1 needs a spherical body (centrifugal = 0): the torques are those of spherical walls')
    num = (int, float, np.integer, np.floating)
    for name, ok in (('MoIZ_M', lambda v: 0 < v < np.inf), ('MoIZ_IC', lambda v: 0 < v < np.inf),
                     ('gTorque', lambda v: 0 <= v < np.inf), ('OmgtauIC', lambda v: v > 0)):
        v = getattr(par, name)
        if not (isinstance(v, num) and not isinstance(v, bool) and ok(v)):
            sys.exit('%s = %r: a number, > 0 for the moments of inertia and OmgtauIC (np.inf: no relaxation), >= 0 for '
                     'gTorque, finite but OmgtauIC' % (name, v))
    for name in ('vtrq', 'mtrq'):
        if getattr(par, name) not in (0, 1):
            sys.exit('%s = %r: 1 (the torque acts on the solids) or 0 (switched off)' % (name, getattr(par, name)))
    if par.magnetic:
        rr = np.linspace(par.ricb, 1.0, 201)
        for n, k in bf.tor_profiles().items():
            kk = np.abs(k(rr, 0))
            if max(kk[0], kk[-1]) > 1e-10*max(kk.max(), 1e-300):
                sys.exit("rotdyn = 1 needs a toroidal B0 that vanishes at both walls (e.g. B0_tor = 'wall'): the torque on "
                         "its current sheet there is left out (degree %d: %.1e at the inner wall, %.1e at the outer)"
                         % (n, kk[0], kk[-1]))


def moves_radially(wall):
    '''
    NEW (2026-11-06): whether the forcing prescribes a radial flow (uᵣ ≠ 0) at the wall 'outer' or 'inner', which stands for
    the motion of the boundary (the wall itself stays at the sphere): forcing 10 at the outer wall; forcings 9, 14 and 7 with
    m != 0 where their amplitude (forcing_amplitude_cmb, forcing_amplitude_icb) is not 0; forcing 6 at the inner wall;
    NEW (2026-11-08) forcing 2 at both walls where forcing_amplitude_cmb is not 0 (one amplitude for both)
    '''
    f = par.forcing
    if f == 2:
        return par.forcing_amplitude_cmb != 0 and (wall == 'outer' or par.ricb > 0)
    if wall == 'outer':
        return f == 10 or ((f in (9, 14) or (f == 7 and par.m != 0)) and par.forcing_amplitude_cmb != 0)
    return par.ricb > 0 and par.forcing_amplitude_icb != 0 and (f in (6, 9, 14) or (f == 7 and par.m != 0))


def forced_walls():
    '''
    NEW (2026-11-06): what a radial boundary flow (moves_radially) stands for at a wall with a conducting solid behind it:
    par.forced_walls, or with None (the default) the picture of the forcing, 'rigid' for forcings 6 and 7 (a rigid solid whose
    shape moves past the sphere: the flattened inner core of forcing 6, the librating triaxial mantle of forcing 7 with m = 2)
    and 'deforming' for forcings 9, 10 and 14 (a boundary that moves, a tide)
    '''
    fw = getattr(par, 'forced_walls', None)
    if fw is None:
        return 'rigid' if par.forcing in (6, 7) else 'deforming'
    return fw


def flows_through(wall):
    '''
    NEW (2026-11-06): whether a radial boundary flow at the wall 'outer' or 'inner' (moves_radially) passes through it into a
    rigid solid at rest (forced_walls 'rigid'), whose rows then take the flow's u×B0 (assemble.bc_b_radial)
    '''
    return moves_radially(wall) and forced_walls() == 'rigid'


def deforms(wall):
    '''
    NEW (2026-11-06): whether a thick conducting solid behind the wall 'outer' or 'inner' deforms with a radial boundary flow
    there (forced_walls 'deforming'): the resolved mantle, or a conducting inner core ('conducting, Chebys'). Its rows then take
    the induction ∇×(v×B0) of its own motion (assemble.deform_rows): v from deform_profile, B0 the background continued inside
    the solid (background_field.solid_profiles). A thin conducting layer ('TWA') has no inside: its rows hold the motion already.
    '''
    if not (par.magnetic and par.hydro and moves_radially(wall) and forced_walls() == 'deforming'):
        return False
    if wall == 'outer':
        return par.mantle == 'resolved'
    return par.ricb > 0 and par.innercore == 'conducting, Chebys'


def thin_drop(wall):
    '''
    NEW (2026-11-06): whether a thin conducting layer ('TWA') at the wall 'outer' or 'inner' deforms with a radial boundary flow
    there (forced_walls 'deforming') while a toroidal B0 is present. Its toroidal part is continuous at the wall (no
    permeability jump where it reaches a wall that moves: check_forced_magnetic) and falls to 0 across the layer (an
    insulator beyond it): the layer, moving with the fluid, carries half of it on average, so its
    rows take half the u×B0 of the toroidal part of a flow through it (assemble.bc_b_radial, tor_half), the limit of a thin
    resolved layer that deforms (deforms). The poloidal part varies by O(h) across the layer and adds nothing at that order.
    '''
    import background_field as bf
    if not (par.magnetic and par.hydro and moves_radially(wall) and forced_walls() == 'deforming' and bf.tor_profiles()):
        return False
    if wall == 'outer':
        return par.mantle == 'TWA' and par.c1_cmb != 0
    return par.ricb > 0 and par.innercore == 'TWA' and par.c1_icb != 0


def deform_profile(wall, l):
    '''
    NEW (2026-11-06): the motion of a thick conducting solid that deforms with its wall (deforms), degree l: its velocity is
    ∇×∇×(P(r_w) φ Y_l 𝐫), P(r_w) the flow's poloidal scalar at the wall and φ(r_w) = 1, so that the radial velocities match
    there; returns φ as f(r, d), its d-th derivative.
      'outer' (the resolved mantle, r_w = 1): φ = 1/r, so v = uᵣ(1)/r² 𝐫̂, purely radial: no tangential motion anywhere,
               solenoidal, the volume of every shell kept.
      'inner' (a conducting inner core, r_w = ricb): φ = ((l+3) x^l - (l+1) x^(l+2))/2, x = r/ricb, the simplest profile
               that is regular at the centre (x^l and x^(l+2)) with no tangential motion at the wall (φ' + φ/r = 0 there).
    Both match a no-slip wall, where the fluid's tangential velocity is 0. Along a slipping wall (NEW 2026-11-07: bci = 0 with
    a conducting inner core too, as bco = 0 with the resolved mantle) that velocity is the slip relative to the solid, whose
    u×B0 enters the wall's electric-field row (assemble.bc_b_slip).
    '''
    import background_field as bf
    if wall == 'outer':
        return bf._powers([(1.0, -1)], 1.0)
    return bf._powers([((l + 3)/2., l), (-(l + 1)/2., l + 2)], par.ricb)


def b0_at_wall(wall, parts='pt'):
    '''
    NEW (2026-11-06): whether B0 reaches the wall 'outer' or 'inner': with 'p' in parts its poloidal part (h or h' of some
    degree is not 0 there), with 't' its toroidal part (k), to 1e-6 of the largest of h, h' and k over the fluid (B0_tor
    'wall' is 0 at the walls, 'confined' outside its region). Where they are 0, that part of B0 is 0 at the wall, and a
    permeability that jumps there moves nothing of it.
    The slope h' of a profile interpolated at degree n = max(64, 2N) ('general', 'ferraro': background_field.cheb_interp,
    its coefficients below 1e-13 of the largest cut) carries round-off below 1e-13 n² 2/(1 - ricb) of it at an end, about
    1e-9 at small N and up to some 5e-6 at N 2000, so the tolerance on h' is ten times that bound where this exceeds 1e-6.
    A profile that the interpolant does not resolve is off by more: a field meant to vanish at a moving wall needs an N that
    resolves it.
    '''
    import background_field as bf
    try:
        pol = [(f, d) for f in bf.pol_profiles().values() for d in (0, 1)]
        tor = [(f, 0) for f in bf.tor_profiles().values()]
        test = (pol if 'p' in parts else []) + (tor if 't' in parts else [])
        if not test:
            return False
        n = 400   # Chebyshev points inside the fluid (none at r = 0)
        x = par.ricb + (1 - par.ricb)*0.5*(1 + np.cos(np.pi*(np.arange(n) + 0.5)/n))
        rw = np.array([1.0 if wall == 'outer' else float(par.ricb)])
        big = max(np.max(np.abs(f(x, d))) for f, d in pol + tor)
        tol = (1e-6, max(1e-6, 1e-12*max(64, 2*par.N)**2*2/(1 - par.ricb)))   # h and k; h'
        return not all(np.max(np.abs(f(rw, d))) <= tol[d]*big for f, d in test)   # (a NaN counts as reaching the wall)
    except Exception:
        return True


def tor_at_wall(wall):
    '''
    NEW (2026-11-07): (n, |k| at the wall 'outer' or 'inner', max |k| over the fluid) for the degree n of B0_tor's own
    profiles (background_field.tor_profiles without the wound field) whose |k| at the wall is the largest part of its own
    largest value, on the 201 points from ricb to 1 that check_rotdyn takes; None without B0_tor. Only the toroidal profiles
    enter (not b0_at_wall's cut, which the poloidal field sets), and an error in evaluating them is not caught. The wound
    field (B0_wound) is 0 at an insulating wall and at a thin layer by construction (wound_profiles: k = 0 at r = 1, and at
    ricb unless the core conducts), to the round-off of its Chebyshev fit, which grows with N (4e-9 of its largest value
    at N 800 in a full sphere): it has no sheet where sheet_jump looks for one.
    '''
    import background_field as bf
    rr = np.linspace(par.ricb, 1.0, 201)
    out = None
    for n, k in bf.tor_profiles(wound=False).items():
        kk = np.abs(k(rr, 0))
        kw, big = float(kk[-1 if wall == 'outer' else 0]), max(float(kk.max()), 1e-300)
        if out is None or kw/big > out[1]/out[2]:
            out = (n, kw, big)
    return out


def wall_bare(wall):
    '''
    NEW (2026-11-07, phase 6c): whether the solid beyond the wall 'outer' or 'inner' does not conduct: an insulator, or a thin
    layer that does not conduct ('TWA' with c1 = 0). B0 then ends at the wall in a current sheet on the fluid's side wherever
    it does not meet the field beyond (sheet_jump). A conducting inner core, the resolved mantle and a conducting layer take
    the field on and hold any drop of it themselves, with no sheet in the fluid.
    '''
    if wall == 'outer':
        return par.mantle == 'insulator' or (par.mantle == 'TWA' and par.c1_cmb == 0)
    return par.ricb > 0 and (par.innercore == 'insulator' or (par.innercore == 'TWA' and par.c1_icb == 0))


def wall_slips(wall):
    '''NEW (2026-11-07, phase 6c): whether the fluid slips along the wall 'outer' or 'inner': a stress-free wall (bco or bci 0,
    or 2 with zero radial stress) or an inviscid fluid (ViscosD = 0)'''
    if not par.hydro:
        return False
    if wall == 'outer':
        return par.ViscosD == 0 or par.bco in (0, 2)
    return par.ricb > 0 and (par.ViscosD == 0 or par.bci in (0, 2))


def pol_continues(wall):
    '''
    NEW (2026-11-07, phase 6c): whether the poloidal B0's own field goes on beyond the wall 'outer' or 'inner': 'axial' at both
    walls (imposed from beyond the outer wall, where its sources lie; a uniform field, it is a potential field in an
    insulating core too), 'dipole' (its source at the centre) and the fields of B0_ic = 'frozen currents' (currents in the
    core) at the inner wall, whose sources lie beyond it; 'confined', which is 0 at the walls and beyond. There B0 has no
    sheet as long as the permeability does not jump at the wall (sheet_jump). The other fields, those of sources inside the
    wall ('general' and 'ferraro', whatever their profiles, taken as fields of currents in the fluid; at the outer wall also
    'dipole' and the fields of 'frozen currents'), meet the solid beyond with the field that its steady condition sets: the
    potential field of those sources through an insulator or a thin layer.
    '''
    if par.B0 in ('axial', 'confined'):
        return True
    return wall == 'inner' and (par.B0 == 'dipole' or (par.B0 in ('G21 dipole', 'Luo_S1', 'Luo_S2')
                                                        and par.B0_ic == 'frozen currents'))


def wall_permeable(wall):
    '''NEW (2026-11-07, phase 6c): whether the permeability jumps at the wall 'outer' or 'inner' or across a thin layer on it:
    mu != 1 (the fluid against the layer or the insulator beyond), or a thin layer ('TWA') with a thickness (c != 0) whose
    permeability is not that of the vacuum beyond (muw != 1). A layer with muw = 1 around a fluid with mu = 1 changes no
    field that goes on through it.'''
    o = wall == 'outer'
    twa = (par.mantle if o else par.innercore) == 'TWA'
    c, p = (par.c_cmb, par.muw_cmb) if o else (par.c_icb, par.muw_icb)
    return par.mu != 1 or (twa and c != 0 and p != 1)


def _pol_wall_row(wall, n, h0, h1):
    '''NEW (2026-11-07, phase 6c): (R, D) of the steady condition of the wall on the poloidal profile h of degree n (h0 = h,
    h1 = h' there): R the f row of assemble.bc_b_twa with c1 = 0 (the rows of the insulating walls for c = 0 and mu = 1:
    r h' + (n+1) h outside, n h - r h' inside), D = 1/mu + n c (outer), the factor of (r h)' in R, or 1/mu + (n+1) c
    (inner), its factor in -R.'''
    o = wall == 'outer'
    twa = (par.mantle if o else par.innercore) == 'TWA'
    r = 1.0 if o else float(par.ricb)
    c, p = ((par.c_cmb, par.muw_cmb) if o else (par.c_icb, par.muw_icb)) if twa else (0.0, 1.0)
    mu = par.mu
    rh1 = h0 + r*h1   # (r h)'
    if o:
        D = 1/mu + n*c
        R = D*rh1 + n*h0
        if c != 0 and not np.isinf(p):
            R = R + (c*mu*n/p)*((n + 1)/p - 1)*h0
    else:
        D = 1/mu + (n + 1)*c
        R = (n + 1)*h0 - D*rh1
        if c != 0 and not np.isinf(p):
            R = R + (c*mu*(n + 1)/p)*(n/p + 1)*h0
    return R, D


def sheet_jump(wall):
    '''
    NEW (2026-11-07, phase 6c): the jump of B0 along the wall 'outer' or 'inner' where it ends in a current sheet on the
    fluid's side, ΔB0_t = B0_t(fluid) - mu H0_t(beyond) (the field beyond as kore's B0 in the fluid, with the permeability
    mu of the fluid relative to the solid's), whose push on the fluid along the wall is ∓Hendrik b_r ΔB0_t per area at the
    outer and the inner wall (assemble.bc_u_sheet; docs/magnetic-equations.md, Current sheets at slipping walls): ({n: k_n},
    {n: a_n}) of the degrees n where it is not 0, {} and {} where the solid beyond conducts (wall_bare).
      toroidal:  B0_phi = k_n P_n' sinθ falls to 0 beyond (no current there, nor in a layer that does not conduct): k_n at
                 the wall, for B0_tor's own profiles, of the degrees whose |k| there is above 1e-10 of their largest value
                 over the fluid (tor_at_wall's test). The wound field (B0_wound) is 0 at such a wall by construction.
      poloidal:  ΔB0_t = a_n ∇₁Y_n: with the steady condition R = 0 of the wall on h_n (_pol_wall_row), which B0 meets where
                 it has no sheet, a_n = R/(r D) outside and -R/(r D) inside (for an insulator (r h' + (n+1) h)/r and
                 (r h' - n h)/r, the jump of B0_θ to the potential field), for the fields that meet the solid beyond with
                 the field its condition sets (not pol_continues), and only the degrees with |a_n| above the round-off of h'
                 at the wall, max(1e-10, 1e-12 n_i² 2/(1 - ricb)) of the largest |B0| of that degree over the fluid (n_i =
                 max(64, 2N) the degree of the interpolants of 'general' and 'ferraro', b0_at_wall): the fields that meet
                 the condition by construction do so to round-off (FIX 2026-11-07: the interpolants by a DCT,
                 background_field.cheb_interp; numpy's left round-off above this tolerance from N about 450). A field that
                 goes on beyond (pol_continues) has no sheet where the permeability does not jump; where it does,
                 check_magnetic stops.
    '''
    import background_field as bf
    if not (par.magnetic and wall_bare(wall)):
        return {}, {}
    rw = 1.0 if wall == 'outer' else float(par.ricb)
    x = np.array([rw])
    tor = {}
    rr = np.linspace(par.ricb, 1.0, 201)
    for n, k in bf.tor_profiles(wound=False).items():
        kk = np.abs(k(rr, 0))
        kw = float(k(x, 0)[0])
        if abs(kw) > 1e-10*max(float(kk.max()), 1e-300):
            tor[n] = kw
    pol = {}
    if par.B0 is not None and not pol_continues(wall):
        ra = rr[rr > 0]
        ni = max(64, 2*par.N)
        tol = max(1e-10, 1e-12*ni**2*2/(1 - par.ricb))
        for n, h in bf.pol_profiles().items():
            big = float(np.max(np.abs(n*(n + 1)*h(ra, 0)/ra) + np.abs(h(ra, 1) + h(ra, 0)/ra)))
            R, D = _pol_wall_row(wall, n, float(h(x, 0)[0]), float(h(x, 1)[0]))
            a = (R if wall == 'outer' else -R)/(rw*D)
            if abs(a) > tol*max(big, 1e-300):
                pol[n] = a
    return tor, pol


def sheet_push(wall):
    '''NEW (2026-11-07, phase 6c): whether B0's current sheet at the wall 'outer' or 'inner' pushes a fluid slipping along it
    (wall_slips and a jump, sheet_jump), so that the stress rows of that wall take the push (assemble.bc_u_sheet)'''
    if not (par.magnetic and wall_slips(wall)):
        return False
    tor, pol = sheet_jump(wall)
    return bool(tor or pol)


def check_forced_magnetic():
    '''
    NEW (2026-11-06): stops on the forced problems with a field (magnetic = 1) that are not covered. Every wall stays at the
    sphere. A radial boundary flow (moves_radially) stands for a boundary that moves, and forced_walls (forced_walls())
    says how the conducting solid behind the wall moves with it, which its tangential electric field depends on:
      'rigid':     the solid is at rest (or turns about z) and only its shape moves past the sphere; the flow passes through
                   the wall, and its u×B0 enters the solid's rows (flows_through, assemble.bc_b_radial). Any solid.
      'deforming': the solid deforms with the boundary (a tide): its surface moves with the fluid, so u×B0 cancels in the
                   matching (the rows without it, as conductive_IC), and the motion inside a thick solid induces field there
                   (deforms, deform_profile, assemble.deform_rows). Not with innercore 'conducting, Bessel' behind a wall
                   that moves (its field would no longer be a Bessel function).
    A wall that moves, where B0 reaches it (b0_at_wall), needs one permeability on both sides of every surface there whose
    jump the rows would miss, under either forced_walls: a surface where the permeability jumps would carry the jump of the
    magnetization of B0 along as it moves, which the rows leave out. So mu_i2o = 1 with a conducting inner core, and mu = 1
    (the fluid against the layer or the insulator beyond). The permeability muw of a thin layer ('TWA'; its rows hold it
    only through c) must be 1 where the part of B0 that it acts on reaches the wall:
      - a layer with a thickness (c != 0): the poloidal B0, and the toroidal B0 too if the layer conducts (c1 != 0);
      - a conducting layer at rest under the flow through it ('rigid', flows_through): both, since the u×B0 of its rows is
        that of its own field, muw/mu times the fluid's;
      - a conducting layer that deforms with the boundary ('deforming') across the drop of a toroidal B0 (thin_drop): the
        toroidal B0, of which it carries muw/(2 mu), half for muw = mu.
    Elsewhere muw acts on no part of B0 at the wall and takes any value: in an insulating layer (c = c1 = 0), in a layer with
    a thickness that does not conduct under a toroidal B0 alone (it holds none of it), and in a conducting sheet (c = 0)
    that deforms with the boundary under a poloidal B0 (in its own frame it carries the fluid's electric field, whatever its
    own field). Walls that stay in place take any permeability.
    The rows of innercore 'conducting, Bessel' hold at forcing_frequency only (k of its field j_l(k r)), so a frequency scan
    (forcing 11 with rsf_frequencies), which shifts A by multiples of B, cannot use them.
    NEW (2026-11-08): forcing 2 moves the walls along themselves too, which no row of a conducting solid holds: it needs
    insulating walls (an insulator, or a thin layer that does not conduct, c1 = 0).
    '''
    if par.forcing == 2:   # NEW (2026-11-08): the tide moves the walls along themselves too
        solids = []
        if par.mantle == 'resolved' or (par.mantle == 'TWA' and par.c1_cmb != 0):
            solids.append('mantle %r' % par.mantle)
        if par.innercore in CONDUCTING_CORES or (par.innercore == 'TWA' and par.c1_icb != 0):
            solids.append('innercore %r' % par.innercore)
        if solids:
            sys.exit("forcing 2 with a conducting solid behind a wall (%s) is not covered: the tide moves the walls along "
                     "themselves, and the electric field of a solid that deforms that way is not implemented. Use insulating "
                     "walls ('insulator', or 'TWA' with c1 = 0)" % ', '.join(solids))
    if forced_walls() == 'deforming' and par.ricb > 0 and moves_radially('inner'):
        if par.innercore == 'conducting, Bessel':
            sys.exit("forced_walls = 'deforming' with innercore 'conducting, Bessel': forcing = %d moves the inner wall "
                     "radially, and the field that the core's motion induces is not a Bessel function. Use innercore "
                     "'conducting, Chebys', or forced_walls = 'rigid' (the core at rest)" % par.forcing)
    for wall in ('outer', 'inner'):
        if not moves_radially(wall):
            continue
        solid = par.mantle if wall == 'outer' else par.innercore
        names = [('mu_i2o' if solid in CONDUCTING_CORES else 'mu', 'pt')]
        if solid == 'TWA':   # the parts of B0 that the layer's permeability acts on
            c, c1, muw = ('c_cmb', 'c1_cmb', 'muw_cmb') if wall == 'outer' else ('c_icb', 'c1_icb', 'muw_icb')
            thick, conducts = getattr(par, c) != 0, getattr(par, c1) != 0
            parts = ('p' if thick else '') + ('t' if conducts and (thick or thin_drop(wall)) else '')
            if conducts and flows_through(wall):
                parts = 'pt'
            if parts:
                names.append((muw, parts))
        jumps = ['%s = %r' % (n, getattr(par, n)) for n, part in names if getattr(par, n) != 1 and b0_at_wall(wall, part)]
        if jumps:
            sys.exit("forcing = %d moves the %s wall radially where B0 reaches it: %s must be 1, one permeability on both "
                     "sides of every surface there (a surface where the permeability jumps would carry the jump of the "
                     "magnetization of B0 along as it moves). Walls that stay in place take any permeability"
                     % (par.forcing, wall, ' and '.join(jumps)))
    if (par.innercore == 'conducting, Bessel' and par.ricb > 0 and par.forcing == 11
            and getattr(par, 'rsf_frequencies', None) is not None):
        sys.exit("innercore 'conducting, Bessel' with a frequency scan (forcing 11, rsf_frequencies): its rows hold at "
                 "forcing_frequency only. Use 'conducting, Chebys'")
    if (par.forcing in (7, 8) and par.m == 0 and par.ricb > 0 and par.bci == 0 and par.forcing_amplitude_icb != 0 and par.hydro
            and par.B0 is not None and (par.innercore in CONDUCTING_CORES or (par.innercore == 'TWA' and par.c1_icb != 0))):
        # NEW (2026-11-07): the slip of utils.cic_slips, twa_slips('inner'), relative to a core that librates (NEW 2026-11-08:
        # forcing 8 too, whose inner core librates relative to the mantle by forcing_amplitude_icb)
        sys.exit('forcing %d (m = 0) with forcing_amplitude_icb != 0 behind a stress-free inner wall (bci = 0) along a '
                 'conducting inner core or a conducting thin layer on it: the libration of the core reaches the fluid there '
                 'through the u×B0 of the slip relative to the core alone, which is not implemented. Set bci = 1, or '
                 'forcing_amplitude_icb = 0' % par.forcing)


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
                finite = np.all(np.isfinite(bf.k_user().coef))
            except Exception as err:
                sys.exit('B0_tor_profile = %r: %s: %s' % (par.B0_tor_profile, type(err).__name__, err))
            if not finite:   # FIX (2026-11-07): as for 'general' (_check_general), it gave a zero field, silently
                _not_finite('B0_tor_profile', par.B0_tor_profile)
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
        if par.ricb > 0 and par.innercore not in ('insulator',) + CONDUCTING_CORES:
            sys.exit('B0_wound = 1: innercore %r is not supported (insulator, conducting, Chebys or conducting, Bessel)'
                     % (par.innercore,))
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
    if par.innercore in CONDUCTING_CORES and par.ricb > 0:
        # NEW (2026-11-07, as conductive_IC): a fluid slipping along the core (bci = 0) is allowed, with its u×B0 in the
        # core's electric-field row (utils.cic_slips); it stopped here before (it needed bci = 1)
        if par.hydro and par.bci == 2:
            sys.exit("innercore %r needs bci = 0 or 1: with zero radial stress (2) the fluid crosses the inner wall, and the "
                     "core's rows follow a radial flow there only where a forcing moves the wall (forced_walls)"
                     % (par.innercore,))
        num = (int, float, np.integer, np.floating)   # NEW (2026-11-06): they divide (operators, bc_b_cic, bc_b_bessel)
        for name in ('mu_i2o', 'sigma_i2o'):
            v = getattr(par, name)
            if not (isinstance(v, num) and not isinstance(v, bool) and np.isfinite(v) and v > 0):
                sys.exit('%s = %r: a finite number > 0' % (name, v))
    if par.innercore == 'conducting, Chebys' and par.ricb > 0:
        if par.N_cic % 2 or par.N_cic < 4:
            sys.exit('N_cic must be even and >= 4, got %r' % par.N_cic)
    if par.innercore == 'conducting, Bessel' and par.ricb > 0 and par.forcing == 0:   # NEW (2026-11-06, as conductive_IC)
        sys.exit("innercore 'conducting, Bessel' needs a forced problem (forcing != 0): the field in the inner core oscillates "
                 "at the forcing frequency. Use 'conducting, Chebys' for eigenvalue problems")
    if par.B0_cmb not in B0_CMBS:   # NEW (2026-11-06)
        sys.exit('B0_cmb %r: set one of %s' % (par.B0_cmb, _names(B0_CMBS)))
    if par.mantle == 'resolved':   # NEW (2026-11-06): the resolved conducting mantle
        check_mantle()
    check_twa()   # NEW (2026-11-06): thin conducting walls
    if par.ricb == 0:   # NEW (2026-10-28): the full sphere
        if par.innercore != 'insulator':
            sys.exit("innercore %r needs ricb > 0" % (par.innercore,))
        if par.B0 == 'dipole':   # NEW (2026-11-07, as conductive_IC)
            sys.exit("B0 'dipole' is singular at r = 0: it needs an inner core (ricb > 0), which holds its source")
        if par.B0_tor is not None and par.B0_tor_l == 2 and par.B0_tor == 'Malkus':
            sys.exit("B0_tor 'Malkus' with B0_tor_l = 2 is not regular at r = 0 (k = r would need to go as r²): use 'wall' or 'user' with ricb = 0")
    if par.ViscosD == 0 and par.ricb > 0:   # NEW (2026-10-28): the inviscid path (bases, one bc row) exists for the full sphere only
        sys.exit('magnetic = 1 with ViscosD = 0 needs ricb = 0 (the inviscid formulation is for the full sphere only)')
    if getattr(par, 'forced_walls', None) not in (None, 'rigid', 'deforming'):   # NEW (2026-11-06), in every magnetic run
        sys.exit("forced_walls = %r: None (the picture of the forcing: 'rigid' for forcings 6 and 7, 'deforming' for 9, 10 "
                 "and 14), 'rigid' (the solids at rest, the boundary flow passes through the walls) or 'deforming' (the "
                 "solids deform with the boundary)" % (par.forced_walls,))
    if par.forcing != 0:   # NEW (2026-11-06, as conductive_IC): forced problems with a field
        check_forced_magnetic()
    # FIX (2026-11-03): an error of cnorm, cnorm_tor at the import of background_field that no check above explains (before
    # magnetic_diffrot, which uses them)
    for name, err in (('cnorm', bf.cnorm_error), ('cnorm_tor', bf.cnorm_tor_error)):
        if err is not None:
            sys.exit('%s could not be computed: %s: %s' % (name, type(err).__name__, err))
    if par.B0 is not None and not np.isfinite(bf.cnorm):   # FIX (2026-11-07): a field that is zero or not finite
        sys.exit('cnorm = %r for B0 %r: the field is zero, or not finite, over the fluid, so it cannot be normalised'
                 % (bf.cnorm, par.B0))
    if par.B0_tor is not None and not np.isfinite(bf.cnorm_tor):   # FIX (2026-11-07): the same for the toroidal part
        sys.exit('cnorm_tor = %r for B0_tor %r: the field is zero, or not finite, over the fluid, so it cannot be '
                 'normalised' % (bf.cnorm_tor, par.B0_tor))
    # NEW (2026-11-07, phase 6c): B0's current sheet at a slipping wall (sheet_jump) pushes the fluid along it, and the
    # stress rows of a viscous fluid take that push (assemble.bc_u_sheet). Stops: an inviscid fluid, which cannot hold it; a
    # deformed body (the metric part of the push is not derived); and a field that goes on beyond the wall (pol_continues)
    # where the permeability jumps, whose field beyond, and so the sheet, kore does not know. With rotdyn = 1,
    # check_rotdyn stops on a toroidal B0 at either wall whatever the wall (the torque on the sheet is left out)
    for wall in ('outer', 'inner'):
        if not (wall_slips(wall) and wall_bare(wall)):
            continue
        o = wall == 'outer'
        bc, c1 = ('bco', 'c1_cmb') if o else ('bci', 'c1_icb')
        beyond = ('an insulator' if (par.mantle if o else par.innercore) == 'insulator'
                  else 'a thin layer that does not conduct (%s = 0)' % c1)
        if (par.B0 not in (None, 'confined') and pol_continues(wall) and wall_permeable(wall)
                and b0_at_wall(wall, 'p')):
            twa = (par.mantle if o else par.innercore) == 'TWA'
            c, p = ('c_cmb', 'muw_cmb') if o else ('c_icb', 'muw_icb')
            imposed = par.B0 == 'axial' and not o   # (imposed from beyond the outer wall, it goes on through the core)
            sys.exit("B0 %r%s goes on beyond the %s wall%s, and the fluid slips along that wall (%s) with %s beyond and a "
                     "permeability that jumps there (mu = %r%s): the field beyond the wall, which sets the current sheet "
                     "there and its push on the fluid, depends on %s, which kore does not model. Use mu = 1%s, %s, or a "
                     "conductor beyond"
                     % (par.B0, " (B0_ic 'frozen currents')" if (not o and par.B0 in ('G21 dipole', 'Luo_S1', 'Luo_S2')
                                                                   and par.B0_ic == 'frozen currents') else '',
                        wall, ' (the field imposed from beyond the outer wall, through the core)' if imposed
                        else ', where its sources are',
                        'ViscosD = 0' if par.ViscosD == 0 else '%s = %d' % (bc, getattr(par, bc)), beyond, par.mu,
                        ", %s = %r, %s = %r" % (c, getattr(par, c), p, getattr(par, p)) if twa else '',
                        'how the jump bends the imposed field' if imposed else 'how those sources meet the jump',
                        " and %s = 0 or %s = 1" % (c, p) if twa else '',
                        ('a viscous fluid with a no-slip wall (ViscosD > 0, %s = 1)' if par.ViscosD == 0
                         else 'a no-slip wall (%s = 1)') % bc))
        tor, pol = sheet_jump(wall)
        if not (tor or pol):
            continue
        what = []
        if tor:
            n, kw, kmax = tor_at_wall(wall)
            what.append('the toroidal B0 is not 0 there (degree %d: |k| %.1e, %.1e at most in the fluid)' % (n, kw, kmax))
        if pol:
            n = max(pol, key=lambda d: abs(pol[d]))
            what.append("the poloidal B0 does not meet the field beyond (degree %d: B0_θ jumps by %.1e ∂θP_%d)"
                        % (n, pol[n], n))
        what = ' and '.join(what)
        if par.ViscosD == 0:   # (a full sphere and its outer wall: with an inner core the inviscid path stopped above)
            small = ''   # a toroidal profile meant to vanish there that does so only to a small part of the field
            if tor:
                n, kw, kmax = tor_at_wall(wall)
                if kw < 1e-4*kmax and par.B0_tor in ('user', 'general'):
                    small = (". If the toroidal profile is meant to vanish at the wall, it does so only to %.0e of the "
                             "field: its zero may be inexact (a constant given to a few digits), its interpolant may not "
                             "resolve it at this N, or the profile may not be smooth at r = 0 (it must be a series in r of "
                             "the parity of its degree, e.g. 'r*(1 - r**2)**2' for degree 1)" % (kw/kmax))
            sys.exit("B0 ends at the %s wall in a current sheet (%s), with %s beyond, and the fluid is inviscid "
                     "(ViscosD = 0): the sheet pushes the fluid along the wall (-Hendrik b_r ΔB0_t per area), which an "
                     "inviscid fluid cannot hold (the push would drive a layer of no thickness along the wall, which its "
                     "equations do not resolve). Use a viscous fluid (ViscosD > 0, whose stress at the wall takes the push), "
                     "a field that meets the wall (a toroidal B0 that vanishes there, e.g. B0_tor = 'wall', and a poloidal "
                     "one that %s), or a conductor beyond (%s'TWA' with c1_cmb != 0), which holds the sheet itself%s"
                     % (wall, what, beyond,
                        "meets the steady condition of the wall, where the permeability jumps (B0_cmb = 'wall', the "
                        "default, adjusts 'G21 dipole', 'Luo_S1', 'Luo_S2' and 'FDM' to it)" if wall_permeable(wall)
                        else 'matches the potential field',
                        "mantle 'resolved', or " if par.mu == 1 else '', small))
        if par.centrifugal:
            sys.exit("B0 ends at the %s wall in a current sheet (%s), with %s beyond, and the fluid slips along that wall "
                     "(%s = %d) in a deformed body (centrifugal = 1): the push of the sheet on the deformed wall, with its "
                     "metric part, is not derived. Use centrifugal = 0, a no-slip wall (%s = 1), or a field that meets the "
                     "wall" % (wall, what, beyond, bc, getattr(par, bc), bc))
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
            finite = np.all(np.isfinite(bf._gen_interp('h' if name == 'B0_profiles' else 'k', n).coef))
        except Exception as err:
            sys.exit('%s[%d] = %r: %s: %s' % (name, n, prof[n], type(err).__name__, err))
        if not finite:   # FIX (2026-11-07): a profile singular in the domain (r**-2 with ricb = 0) gave a zero field, silently
            _not_finite('%s[%d]' % (name, n), prof[n])


def _not_finite(name, expr):
    '''
    FIX (2026-11-07): stops on a 'general' or 'user' profile that is not finite at some point of its Chebyshev interpolation
    (background_field._gen_interp, k_user: the nodes of [ricb, 1], or of [-1, 1] at |r| in a full sphere, r = 0 among them),
    where it gave NaN coefficients and so no field at all. Only these points are tested: a profile singular between them
    passes, with an interpolant that is finite but wrong.
    '''
    sys.exit('%s = %r is not finite at some of the points where it is interpolated (%s)%s'
             % (name, expr, '[-1, 1] at |r|, r = 0 included' if par.ricb == 0 else '[ricb, 1]',
                ': if the field is regular but the expression is 0/0 at r = 0, write it in a form that numpy evaluates '
                'there (a series, or where(r == 0, 0, ...)); a field singular at r = 0 needs an inner core (ricb > 0)'
                if par.ricb == 0 else ''))


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
    if par.innercore in CONDUCTING_CORES:   # NEW (2026-11-06): 'conducting, Bessel' too (its k in utils.bessel_ic)
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
    import background_field as bf
    if par.B0 == 'ferraro':   # FIX (2026-11-07): its cnorm is ferraro_cnorm, not bf.cnorm (1), which check_magnetic tests
        with np.errstate(divide='ignore', invalid='ignore'):
            for n in bf.ferraro_degrees():   # builds the interpolants of h_n (background_field._fer)
                bf.ferraro_profile(n, np.array([1.0]), 0)
            cf = bf.ferraro_cnorm()
        if not all(np.all(np.isfinite(c.coef)) for c in bf._fer.values()):   # as _check_general for 'general'
            sys.exit("B0 'ferraro' (G = %r) is not finite over the fluid: G must be finite on the values that ΔΩ takes "
                     "there" % (par.B0_ferraro_G,))
        if not np.isfinite(cf):
            sys.exit("cnorm = %r for B0 'ferraro' (G = %r): the field is zero, or not finite, over the fluid, so it cannot "
                     "be normalised" % (cf, par.B0_ferraro_G))
    mis = ferraro_mismatch()
    if par.B0 == 'ferraro' and not _warned:
        ferraro_report()
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
    if par.compositional and par.comp_stratification == 'user':   # NEW (2026-11-08)
        out.append('comp_stratification: %s ; p = %r' % (par.comp_stratification_profile, par.comp_stratification_param))
    if par.diffusivity == 'user':
        out.append('diffusivity: nu = %s ; kappa = %s%s ; p = %r' % (par.diffusivity_profiles['nu'],
                   par.diffusivity_profiles['kappa'],
                   ' ; D = %s' % par.diffusivity_profiles['D'] if par.compositional else '', par.diffusivity_param))
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
    if t == 'two zone':         # NEW (2026-11-08): conductive_IC's, a neutral inner zone below a stratified outer one
        return two_zone(r, *par.two_zone)
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


def two_zone(r, rc, h):
    '''NEW (2026-11-08): the 'two zone' gradient (1 + tanh(2(r - rc)/h))/2 at r >= 0, conductive_IC's (utils.twozone there)'''
    return 0.5*(1 + np.tanh(2*(np.asarray(r, dtype=float) - rc)/h))

def gradC(prf, r):
    '''
    NEW (2026-11-08): dC/dr at r >= 0, the gradient of the background composition (compositional = 1), of amplitude 1
    (BeyonceC sets the size of N_c² = BeyonceC g dC/dr); kore uses it as an odd function of r (radial_profiles.rdCdr,
    gradCX), as dS/dr. prf, the structure, is not used (the same signature as gradS).
    '''
    t = par.comp_stratification
    if t == 'linear':           # r: N_c² = BeyonceC r² with 'Boussinesq' (conductive_IC's 'internal')
        return np.array(r, dtype=float)
    if t == 'inverse square':   # 1/r²: diffusion between the walls (ricb > 0; conductive_IC's 'differential')
        return 1/r**2
    if t == 'two zone':
        return two_zone(r, *par.comp_two_zone)
    return _expression(par.comp_stratification_profile, par.comp_stratification_param, 'comp_stratification_profile')(r)


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
    '''ν/ViscosD (which = 'nu'), κ/ThermaD ('kappa') or (NEW 2026-11-08) D/CompoD ('D') at r, even in r, times
    (ρ0(aux3)/ρ0)**visc_pow (therm_pow, comp_pow); η/MagnetD ('eta') is 1 for now, whatever the diffusivity type'''
    if which == 'eta':  # NEW (2026-10-28): the hook for a magnetic diffusivity profile (the operators have η and η')
        return np.ones_like(r)
    c, pw = {'nu': (par.visc0, par.visc_pow), 'kappa': (par.therm0, par.therm_pow), 'D': (par.comp0, par.comp_pow)}[which]
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
    if par.compositional:   # NEW (2026-11-08): the composition's gradient and diffusivity
        singles += [('dC/dr', 1, prf.gradC, 'comp_stratification')]
        if par.CompoD > 0:
            singles += [('D', 0, lambda r: diffusivity(prf, r, 'D'), 'diffusivity')]
    for name, n, f, kind in singles:
        family([('%s (%s %r)' % (name, kind, getattr(par, kind)), n, f)], 'max |%s|' % name)
    if par.centrifugal:   # NEW (2026-11-05): the flattening of the level surfaces must be even and smooth at r = 0
        import centrifugal as cf
        family([("the flattening f (centrifugal_shape %r)" % par.centrifugal_shape, 0, cf.flattening)], 'max |f|')
    return nwarn[0]
