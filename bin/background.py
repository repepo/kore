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
MAGNETIC_WALLS = ('insulator',)   # NEW (2026-10-28): mantle (magnetic = 1); B0 types: background_field.B0_TYPES
INNER_CORES = ('insulator', 'conducting, Chebys')   # NEW (2026-10-28): innercore (magnetic = 1)

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
    if par.magnetic:
        check_magnetic()
    if par.diffusivity not in DIFFUSIVITIES:
        sys.exit('diffusivity %r: set one of %s' % (par.diffusivity, _names(DIFFUSIVITIES)))
    if par.diffusivity == 'user':
        if sorted(par.diffusivity_profiles) != ['kappa', 'nu']:
            sys.exit("diffusivity 'user' needs diffusivity_profiles = {'nu': '...', 'kappa': '...'}")
        for k in ('nu', 'kappa'):
            _check_expression(par.diffusivity_profiles[k], par.diffusivity_param, "diffusivity_profiles['%s']" % k)


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
        if not (isinstance(par.cnorm_tor, (int, float)) or par.cnorm_tor in ('mag_energy', 'Schmitt2012')):
            sys.exit("cnorm_tor %r: a number, 'mag_energy' or 'Schmitt2012'" % (par.cnorm_tor,))
        if par.B0 is not None and bf.symm_pol != bf.symm_tor:
            sys.exit('B0 %r and B0_tor %r have opposite equatorial symmetries: they cannot be combined in one eigenproblem'
                     % (par.B0, par.B0_tor))
    if par.B0 == 'general':   # NEW (2026-10-30)
        _check_general(par.B0_profiles, 'B0_profiles')
    if par.B0 == 'ferraro':   # NEW (2026-10-30)
        if not par.diff_rot:
            sys.exit("B0 'ferraro' needs a differential rotation (diff_rot = 1): its flux function is G(ΔΩ)")
        if par.diff_rot_type == 'ferraro':
            sys.exit("diff_rot_type 'ferraro' with B0 'ferraro' is circular (each is built from the other): choose one")
        if par.B0_ferraro_nmax < 1 or par.B0_ferraro_nmax % 2 == 0:
            sys.exit('B0_ferraro_nmax must be odd and >= 1 (the field has only odd degrees)')
        if par.cnorm == 'rms_cmb':
            sys.exit("B0 'ferraro': use cnorm 'mag_energy' or a number")
    if getattr(par, 'B0_wound', 0):   # NEW (2026-10-30): the wound toroidal field (background_field.wound_profiles)
        if par.B0 is None or not par.diff_rot:
            sys.exit('B0_wound = 1 winds a toroidal field from the poloidal B0 with the differential rotation: it needs a poloidal B0 and diff_rot = 1')
        if not par.B0_wound_eta > 0:
            sys.exit('B0_wound_eta must be > 0 (the diffusivity that balances the winding, in units of MagnetD)')
        if par.ricb > 0 and par.innercore not in ('insulator', 'conducting, Chebys'):
            sys.exit('B0_wound = 1: innercore %r is not supported (insulator or conducting, Chebys)' % (par.innercore,))
    if par.B0 == 'confined':   # NEW (2026-10-30)
        r1, r2 = par.B0_conf_r1, par.B0_conf_r2
        if par.B0_l not in (1, 2) or par.B0_conf_p < 2 or not (0 <= r1 < r2 <= 1):
            sys.exit("B0 'confined' needs B0_l = 1 or 2, B0_conf_p >= 2 and 0 <= B0_conf_r1 < B0_conf_r2 <= 1")
        if par.ricb > 0 and r1 < par.ricb:
            sys.exit("B0 'confined' with an inner core: the region must lie in the fluid, B0_conf_r1 >= ricb")
        if par.cnorm == 'rms_cmb':
            sys.exit("B0 'confined': cnorm 'rms_cmb' is not defined (B0 is zero at the CMB); use 'mag_energy' or a number")
    if par.B0 == 'FDM' and par.B0_l not in (1, 2):   # FIX (2026-10-28): the quadrupole couplings are implemented (operators.lorentz_pol2)
        sys.exit("B0 'FDM' with B0_l = %r: only B0_l = 1 or 2 is implemented" % par.B0_l)
    if not (isinstance(par.cnorm, (int, float)) or par.cnorm in ('rms_cmb', 'mag_energy', 'Schmitt2012')):
        sys.exit("cnorm %r: a number, 'rms_cmb', 'mag_energy' or 'Schmitt2012'" % (par.cnorm,))
    if par.mantle not in MAGNETIC_WALLS:
        sys.exit('mantle %r: set one of %s' % (par.mantle, _names(MAGNETIC_WALLS)))
    if par.innercore not in INNER_CORES:
        sys.exit('innercore %r: set one of %s' % (par.innercore, _names(INNER_CORES)))
    if par.innercore == 'conducting, Chebys':
        if par.bci != 1:
            sys.exit("innercore 'conducting, Chebys' needs a no-slip inner boundary (bci = 1): the ICB matching assumes u = 0 there")
        if par.N_cic % 2 or par.N_cic < 4:
            sys.exit('N_cic must be even and >= 4, got %r' % par.N_cic)
    if par.ricb == 0:   # NEW (2026-10-28): the full sphere
        if par.innercore != 'insulator':
            sys.exit("innercore %r needs ricb > 0" % (par.innercore,))
        if par.B0_tor is not None and par.B0_tor_l == 2 and par.B0_tor == 'Malkus':
            sys.exit("B0_tor 'Malkus' with B0_tor_l = 2 is not regular at r = 0 (k = r would need to go as r²): use 'wall' or 'user' with ricb = 0")
    if par.ViscosD == 0 and par.ricb > 0:   # NEW (2026-10-28): the inviscid path (bases, one bc row) exists for the full sphere only
        sys.exit('magnetic = 1 with ViscosD = 0 needs ricb = 0 (the inviscid formulation is for the full sphere only)')
    if par.forcing != 0:
        sys.exit('magnetic = 1 is implemented for eigenvalue problems only (forcing = 0)')
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
    degs = sorted(prof)
    if any(not isinstance(n, (int, np.integer)) or n < 1 for n in degs) or len(set(n % 2 for n in degs)) > 1:
        sys.exit('%s: the degrees must be integers >= 1 of one parity (one equatorial symmetry), got %r' % (name, degs))
    for n in degs:
        try:
            bf.gen_profile('h' if name == 'B0_profiles' else 'k', n, np.linspace(max(par.ricb, 0.1), 1, 4), 0)
        except Exception as err:
            sys.exit('%s[%d] = %r: %s: %s' % (name, n, prof[n], type(err).__name__, err))


# NEW (2026-10-30): what to do when a poloidal B0 and the differential rotation are not a steady background together.
# 'warn': print the mismatch and go on (the linear problem assumes the background is maintained); 'refuse': stop.
FERRARO_POLICY = 'warn'

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
        shear a solid core, which kore does not model;
      - a poloidal B0 and ΔΩ should satisfy Ferraro's law (ferraro_mismatch); FERRARO_POLICY says whether a mismatch above
        1e-7 stops the run or only prints a warning.
    '''
    import diff_rot as dr
    if par.innercore == 'conducting, Chebys':
        c, ok = dr.rigid_ricb()
        if not ok:
            sys.exit('magnetic = 1, diff_rot = 1 with a conducting inner core needs a rigid rotation of the inner boundary '
                     '(f_n(ricb) = 0 for every n > 0): ΔΩ(ricb, θ) varies with latitude')
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
        if FERRARO_POLICY == 'refuse':
            sys.exit(msg)
        if not _warned:
            print('WARNING: ' + msg)
    _warned.append(1)


def description():
    '''the 'user' expressions as text for params.dat (a '#' comment), empty if there are none'''
    out = []
    if par.magnetic and par.B0_tor == 'user':   # NEW (2026-10-28)
        out.append('B0_tor: k = %s ; p = %r' % (par.B0_tor_profile, par.B0_tor_param))
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
