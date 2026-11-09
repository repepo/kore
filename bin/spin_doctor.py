#!/usr/bin/env python3
'''
kore postprocessing script

Usage:
> python3 ./bin/spin_doctor.py ncpus
'''

import sys
sys.path.insert(1,'bin/')
from timeit import default_timer as timer
import os.path
import numpy as np
from parameters import par
import utils as ut
import utils4pp as upp
import diff_rot as dr
import radial_profiles as rap
import background as bg
import background_field as bf
import centrifugal as cf   # NEW (2026-11-05)
import rotdyn as rd   # NEW (2026-11-06): the rotating solids and the angular momentum budget
import topo   # NEW (2026-11-06): the drag at the outer wall
import bgflow as bgf   # NEW (2026-11-08): a background flow (bgflow = 1)



# diff_rot_type as written to params.dat: its index in diff_rot.TYPES, -1 if not listed
def diff_rot_code():
    return dr.TYPES.index(par.diff_rot_type) if par.diff_rot_type in dr.TYPES else -1

# the background as written to params.dat: the index of structure, stratification and diffusivity in the tuples of
# background.py (-1 for a stratification of None, with thermal = 0)
def structure_code():
    return bg.STRUCTURES.index(par.structure)

def stratification_code():
    return bg.STRATIFICATIONS.index(par.stratification) if par.stratification in bg.STRATIFICATIONS else -1

def diffusivity_code():
    return bg.DIFFUSIVITIES.index(par.diffusivity)

def comp_stratification_code():   # NEW (2026-11-08): -1 without the composition (compositional = 0)
    return bg.COMP_STRATIFICATIONS.index(par.comp_stratification) if (par.compositional and
                                                                       par.comp_stratification in bg.COMP_STRATIFICATIONS) else -1

def B0_code():  # NEW (2026-10-28): B0 as its index in background_field.B0_TYPES, -1 without the field
    return bf.B0_TYPES.index(par.B0) if (par.magnetic and par.B0 in bf.B0_TYPES) else -1

def cnorm_used():  # FIX (2026-11-03): for B0 'ferraro', bf.cnorm is a placeholder (1); its cnorm is computed on first use
    return bf.ferraro_cnorm() if (par.magnetic and par.B0 == 'ferraro') else bf.cnorm

def index_in(value, names):  # NEW (2026-11-03): the index of a name in its tuple, -1 if it is not there
    return names.index(value) if value in names else -1

def solar_density_window(k):
    '''end k (0, 1) of par.solar_density_window, nan if None (the default window), -/+(r_out + 0.005) if 'sun' (even fit)'''
    w = par.solar_density_window
    if isinstance(w, str):
        return (2*k - 1)*(par.r_out + 0.005)
    return np.nan if w is None else w[k]

def row_note():
    '''the '#' comment ending each params.dat row: the 'user' expressions of the background and of the rotation
    (background.description, diff_rot.description) and the 'tabulated' file with the first 12 hex digits of its sha256;
    NEW (2026-11-08) the background flow's expressions or its file and sha256 (bgflow.description)'''
    notes = [bg.description()] if bg.description() else []
    if par.diff_rot and dr.description():
        notes.append(dr.description())
    if cf.description():   # NEW (2026-11-05)
        notes.append(cf.description())
    if par.bgflow:   # NEW (2026-11-08)
        notes.append(bgf.description())
    if par.diff_rot and par.diff_rot_type == 'tabulated':
        import hashlib
        with open(par.diff_rot_file, 'rb') as f:
            notes.append('tabulated: %s sha256 %s' % (par.diff_rot_file, hashlib.sha256(f.read()).hexdigest()[:12]))
    return (' # ' + ' ; '.join(notes)) if notes else ''

def diff_rot_window_code():
    '''the 'tabulated' fit used: 0 over the shell, 1 over the whole Sun (diff_rot_window = 'sun', the default when ricb = 0)'''
    if not (par.diff_rot and par.diff_rot_type == 'tabulated'):
        return 0
    return int(par.diff_rot_window == 'sun' or (par.diff_rot_window is None and par.ricb == 0))

def solar_density_degree():
    '''degree of the Model S fits actually used: solar_density_sun_deg for the even whole-Sun fit ('sun', or None with ricb = 0);
    structure_deg for structure = 'model S hydrostatic' '''
    if par.structure == 'model S hydrostatic':
        return par.structure_deg
    w = par.solar_density_window
    return par.solar_density_sun_deg if (isinstance(w, str) or (w is None and par.ricb == 0)) else par.solar_density_deg



def mantle_value(fun):
    '''NEW (2026-11-06): a number of the resolved mantle for params.dat, 0 without it (mantle != 'resolved')'''
    return float(np.asarray(fun())) if ut.resolved_mantle else 0.0

def param_table(elapsed):
    '''
    Run parameters written as one row of params.dat, as (name, value, format) tuples.
    The order here is the column order in params.dat. To add a parameter, append a
    tuple at the end so existing columns keep their positions.
    '''
    topo_on = par.topo_drag is not None   # NEW (2026-11-06)
    return [
        ( 'thermal',                  par.thermal,                 '%d' ),
        ( 'm',                        par.m,                       '%d' ),
        ( 'symm',                     par.symm,                    '%d' ),
        ( 'ricb',                     par.ricb,                    '%.9e' ),
        ( 'bci',                      par.bci,                     '%d' ),
        ( 'bco',                      par.bco,                     '%d' ),
        ( 'forcing',                  par.forcing,                 '%d' ),
        ( 'forcing_frequency',        par.forcing_frequency,       '%.9e' ),
        ( 'forcing_amplitude_cmb',    par.forcing_amplitude_cmb,   '%.9e' ),
        ( 'forcing_amplitude_icb',    par.forcing_amplitude_icb,   '%.9e' ),
        ( 'Gaspard',                  par.Gaspard,                 '%.9e' ),
        ( 'Beyonce',                  par.Beyonce,                 '%.9e' ),
        ( 'ViscosD',                  par.ViscosD,                 '%.9e' ),
        ( 'ThermaD',                  par.ThermaD,                 '%.9e' ),
        ( 'ncpus',                    par.ncpus,                   '%d' ),
        ( 'N',                        par.N,                       '%d' ),
        ( 'lmax',                     par.lmax,                    '%d' ),
        ( 'runtime',                  elapsed,                     '%.9e' ),
        ( 'r_out',                    par.r_out,                   '%.9e' ),  # aux0 before background.py
        ( 'aux1',                     par.aux1,                    '%.9e' ),
        ( 'aux2',                     par.aux2,                    '%.9e' ),
        ( 'aux3',                     par.aux3,                    '%.9e' ),
        ( 'aux4',                     par.aux4,                    '%.9e' ),
        ( 'aux5',                     par.aux5,                    '%.9e' ),
        ( 'visc0',                    par.visc0,                   '%.9e' ),
        ( 'hvisc',                    par.hvisc,                   '%.9e' ),
        ( 'rvisc',                    par.rvisc,                   '%.9e' ),
        ( 'rpower_u',                 par.rpower_u,                '%.9f' ),
        ( 'rhopower_u',               par.rhopower_u,              '%.9f' ),
        ( 'rpower_v',                 par.rpower_v,                '%.9f' ),
        ( 'rhopower_v',               par.rhopower_v,              '%.9f' ),
        ( 'rpower_pp',                par.rpower_pp,               '%.9f' ),
        ( 'rhopower_pp',              par.rhopower_pp,             '%.9f' ),
        ( 'therm0',                   par.therm0,                  '%.9e' ),
        ( 'visc_pow',                 par.visc_pow,                '%.9e' ),
        ( 'therm_pow',                par.therm_pow,               '%.9e' ),
        ( 'diff_rot',                 par.diff_rot,                '%d' ),
        ( 'diff_rot_ampl',            par.diff_rot_ampl,           '%.9e' ),
        ( 'thermal_wind',             par.thermal_wind,            '%d' ),
        ( 'diff_rot_type',            diff_rot_code(),             '%d' ),  # index into diff_rot.TYPES ('user': its profiles end the row as a # comment)
        ( 'dSdtheta',                 par.dSdtheta,                '%.9e' ),  # thermal_wind = 3
    ] + [
        ( 'diff_rot_term_amp_%d' % (2*k), dr.term_amp(2*k),        '%.9e' ) for k in range(9)   # a_0, a_2, ..., a_16
    ] + [
        ( 'thermal_wind_amplitude',   par.thermal_wind_amplitude,  '%.9e' ),
        ( 'thermal_wind_radial',      par.thermal_wind_radial,     '%d' ),
        ( 'diff_rot_frame',           par.diff_rot_frame,          '%.9e' ),  # 'solar_P4', 'tabulated'
        ( 'diff_rot_nmax',            par.diff_rot_nmax,           '%d' ),    # 'tabulated'
        ( 'diff_rot_rdeg',            par.diff_rot_sun_deg if diff_rot_window_code() else par.diff_rot_rdeg, '%d' ),  # 'tabulated': the degree used
        ( 'diff_rot_generic',         0,                           '%d' ),    # removed option, always 0 (keeps the columns after it in place)
    ] + [
        ( 'diff_rot_solar_%d' % k,    par.diff_rot_solar[k],       '%.9e' ) for k in range(6)   # 'solar_P4': c0, c2, c4, Ω_RZ, rtc, dtc
    ] + [
        ( 'structure',                structure_code(),            '%d' ),    # index into background.STRUCTURES
        ( 'solar_density_deg',        solar_density_degree(),      '%d' ),    # the degree used ('sun': solar_density_sun_deg)
        ( 'solar_density_window_lo',  solar_density_window(0),     '%.9e' ),  # nan: the default window
        ( 'solar_density_window_hi',  solar_density_window(1),     '%.9e' ),
        ( 'solar_rz',                 int(par.stratification == 'solar_rz'), '%d' ),  # 1: stratification 'solar_rz'
        ( 'solar_rz_dSdr',            par.solar_rz_dSdr,           '%.9e' ),
        ( 'solar_rz_x1',              par.solar_rz_steps[0],       '%.9e' ),
        ( 'solar_rz_x2',              par.solar_rz_steps[1],       '%.9e' ),
        ( 'solar_rz_d',               par.solar_rz_steps[2],       '%.9e' ),
        ( 'solar_rz_core',            par.solar_rz_core,           '%.9e' ),
        ( 'solar_rz_xos',             par.solar_rz_os[0],          '%.9e' ),
        ( 'solar_rz_dos',             par.solar_rz_os[1],          '%.9e' ),
        ( 'diff_rot_window',          diff_rot_window_code(),      '%d' ),    # 'tabulated': 0 shell fit, 1 whole Sun ('sun')
        ( 'diff_rot_hold',            np.nan,                      '%.9e' ),  # removed option, always nan (keeps the columns after it in place)
        ( 'solar_rz_delta_cz',        par.solar_rz_delta_cz,       '%.9e' ),  # convection-zone superadiabaticity
        ( 'solar_rz_delta_sf',        par.solar_rz_delta_sf,       '%.9e' ),
        ( 'drop_tol',                 par.drop_tol,                '%.9e' ),
        ( 'diff_rot_param',           par.diff_rot_param,          '%.9e' ),  # 'shellular': σ; 'user': p
        ( 'stratification',           stratification_code(),       '%d' ),    # index into background.STRATIFICATIONS
        ( 'diffusivity',              diffusivity_code(),          '%d' ),    # index into background.DIFFUSIVITIES
        ( 'magnetic',                 par.magnetic,                '%d' ),    # NEW (2026-10-28), and the magnetic parameters:
        ( 'B0',                       B0_code(),                   '%d' ),    # index into background_field.B0_TYPES
        ( 'B0_l',                     bf.B0_l,                     '%d' ),
        ( 'cnorm',                    cnorm_used(),                '%.9e' ),  # as a number (0 without the field)
        ( 'Hendrik',                  par.Hendrik,                 '%.9e' ),
        ( 'MagnetD',                  par.MagnetD,                 '%.9e' ),
        ( 'innercore',                bg.INNER_CORES.index(par.innercore) if par.innercore in bg.INNER_CORES else -1, '%d' ),  # NEW (2026-10-28)
        ( 'N_cic',                    par.N_cic,                   '%d' ),
        ( 'mu_i2o',                   par.mu_i2o,                  '%.9e' ),
        ( 'sigma_i2o',                par.sigma_i2o,               '%.9e' ),
        ( 'B0_tor',                   bf.B0_TOR_TYPES.index(par.B0_tor) if (par.magnetic and par.B0_tor in bf.B0_TOR_TYPES) else -1, '%d' ),  # NEW (2026-10-28)
        ( 'B0_tor_l',                 par.B0_tor_l,                '%d' ),
        ( 'cnorm_tor',                bf.cnorm_tor,                '%.9e' ),  # as a number (0 without a toroidal field)
        ( 'B0_wound',                 int(bf.wound_on()),          '%d' ),    # NEW (2026-11-03): 1 if the wound field was added
        ( 'B0_wound_eta',             par.B0_wound_eta,            '%.9e' ),
        ( 'B0_conf_r1',               par.B0_conf_r1,              '%.9e' ),  # 'confined'
        ( 'B0_conf_r2',               par.B0_conf_r2,              '%.9e' ),
        ( 'B0_conf_p',                par.B0_conf_p,               '%d' ),
        ( 'B0_param',                 par.B0_param,                '%.9e' ),  # p of 'general' and 'ferraro' (expressions in the # note)
        ( 'B0_tor_param',             par.B0_tor_param,            '%.9e' ),  # p of B0_tor 'user' and 'general'
        ( 'B0_ferraro_nmax',          par.B0_ferraro_nmax,         '%d' ),
        ( 'mag_couplings',            int(ut.mag_algebra),         '%d' ),    # as used: 0 the closed forms, 1 mag_algebra.py
        ( 'tol_B0',                   par.tol_B0,                  '%.9e' ),
        ( 'B0_ic',                    index_in(par.B0_ic, bg.B0_ICS), '%d' ),   # index into background.B0_ICS
        ( 'ferraro_policy',           index_in(par.ferraro_policy, bg.FERRARO_POLICIES), '%d' ),   # background.FERRARO_POLICIES
        ( 'centrifugal',              par.centrifugal,             '%d' ),    # NEW (2026-11-05), and the centrifugal parameters:
        ( 'centrifugal_shape',        index_in(par.centrifugal_shape, cf.SHAPES), '%d' ),   # index into centrifugal.SHAPES
        ( 'centrifugal_q',            par.centrifugal_q,           '%.9e' ),
        ( 'centrifugal_f',            float(cf.flattening(np.array([1.0]))[0]) if par.centrifugal else 0.0, '%.9e' ),  # f at the outer wall, as used
        ( 'centrifugal_param',        par.centrifugal_param,       '%.9e' ),  # p of 'user' (its expression in the # note)
        ( 'hydro',                    par.hydro,                   '%d' ),    # NEW (2026-11-06): 0 without the flow
        ( 'mantle',                   index_in(par.mantle, bg.MAGNETIC_WALLS), '%d' ),   # NEW (2026-11-06): the thin walls,
        ( 'c_cmb',                    par.c_cmb,                   '%.9e' ),  #   index into background.MAGNETIC_WALLS
        ( 'c1_cmb',                   par.c1_cmb,                  '%.9e' ),
        ( 'muw_cmb',                  par.muw_cmb,                 '%.9e' ),
        ( 'c_icb',                    par.c_icb,                   '%.9e' ),
        ( 'c1_icb',                   par.c1_icb,                  '%.9e' ),
        ( 'muw_icb',                  par.muw_icb,                 '%.9e' ),
        ( 'mu',                       par.mu,                      '%.9e' ),
        ( 'B0_cmb',                   index_in(par.B0_cmb, bg.B0_CMBS), '%d' ),   # index into background.B0_CMBS
        ( 'r_surface',                par.r_surface,               '%.9e' ),  # NEW (2026-11-06): the resolved mantle
        ( 'N_mantle',                 par.N_mantle,                '%d' ),
        ( 'sigma_mantle_cmb',         mantle_value(lambda: bg.sigma_mantle(ut.rcmb)), '%.9e' ),  # sigma at r = 1, as used
        ( 'sigma_mantle_param',       par.sigma_mantle_param,      '%.9e' ),  # p of an expression (in the # note)
        ( 'c1_mantle',                mantle_value(bg.mantle_conductance), '%.9e' ),   # ∫ sigma dr, the c1 of a thin wall
        ( 'rotdyn',                   par.rotdyn,                  '%d' ),    # NEW (2026-11-06): the rotating solids
        ( 'MoIZ_M',                   par.MoIZ_M,                  '%.9e' ),
        ( 'MoIZ_IC',                  par.MoIZ_IC,                 '%.9e' ),
        ( 'gTorque',                  par.gTorque,                 '%.9e' ),
        ( 'OmgtauIC',                 par.OmgtauIC,                '%.9e' ),
        ( 'vtrq',                     par.vtrq,                    '%d' ),
        ( 'mtrq',                     par.mtrq,                    '%d' ),
        # NEW (2026-11-06): the drag at the outer wall (topo.py): kappa_cmb, topo_drag (1: 'rekier', 0: None), and with
        # 'rekier' (0 otherwise) its inputs and the N/Omega and w0/Omega it used
        ( 'kappa_cmb',                par.kappa_cmb,               '%.9e' ),
        ( 'topo_drag',                int(topo_on),                '%d' ),
        ( 'topo_hrms',                par.topo_hrms if topo_on else 0.0,     '%.9e' ),
        ( 'topo_lambda0',             par.topo_lambda0 if topo_on else 0.0,  '%.9e' ),
        ( 'topo_H',                   par.topo_H if topo_on else 0.0,        '%.9e' ),
        ( 'topo_N_cmb',               topo.N_cmb() if topo_on else 0.0,      '%.9e' ),
        ( 'topo_omega',               topo.omega0() if topo_on else 0.0,     '%.9e' ),
        # NEW (2026-11-08, from conductive_IC): the composition and the 'two zone' gradients (two_zone: the entropy's)
        ( 'compositional',            par.compositional,           '%d' ),
        ( 'comp_stratification',      comp_stratification_code(),  '%d' ),    # index into background.COMP_STRATIFICATIONS
        ( 'comp_stratification_param', par.comp_stratification_param, '%.9e' ),  # p of 'user' (its expression in the # note)
        ( 'BeyonceC',                 par.BeyonceC,                '%.9e' ),
        ( 'CompoD',                   par.CompoD,                  '%.9e' ),
        ( 'comp0',                    par.comp0,                   '%.9e' ),
        ( 'comp_pow',                 par.comp_pow,                '%.9e' ),
        ( 'bci_compositional',        par.bci_compositional,       '%d' ),
        ( 'bco_compositional',        par.bco_compositional,       '%d' ),
        ( 'comp_two_zone_rc',         par.comp_two_zone[0],        '%.9e' ),
        ( 'comp_two_zone_h',          par.comp_two_zone[1],        '%.9e' ),
        ( 'two_zone_rc',              par.two_zone[0],             '%.9e' ),
        ( 'two_zone_h',               par.two_zone[1],             '%.9e' ),
        # NEW (2026-11-08, from conductive_IC): a background flow (bgflow.py; its expressions or its file in the # note)
        ( 'bgflow',                   par.bgflow,                  '%d' ),
        ( 'bgflow_Ro',                par.bgflow_Ro if par.bgflow else 0.0, '%.9e' ),
    ]



def ratio(a, b):
    '''
    a/b, or nan when b is zero (e.g. Tor/Pol when there is no flow)
    '''
    return a/b if b != 0 else np.nan



IRREG0_NOTE = 1e-6   # NEW (2026-11-04): main() counts the solutions with irreg₀ above this, after the table

def table_widths(cols):
    '''
    Width of each column of the summary table: the wider of its header and a formatted
    sample value, plus some padding. Fixed up front so rows can be printed as they come.
    '''
    # sample values sized for the widest numbers expected: up to 99 solutions, |σ|,|ω| < 10000
    sample = lambda fmt: 99 if fmt.endswith('d}') else (-9999.0 if fmt.endswith('f}') else 1.0)
    return [ max(len(head), len(fmt.format(sample(fmt)))) + 2 for head, fmt, _ in cols ]



def table_header(cols, widths):
    '''
    Header line and the ‾‾‾ underline of the summary table
    '''
    head = ' ' + ' '.join( h.center(wd) for (h, _, _), wd in zip(cols, widths) )
    line = ' ' + ' '.join( '‾'*wd for wd in widths )
    return head, line



def table_row(cols, widths, i):
    '''
    Row of the summary table for solution i
    '''
    return ' ' + ' '.join( fmt.format(get(i)).center(wd) for (_, fmt, get), wd in zip(cols, widths) )



def main(ncpus):

    # ------------------------------------------------------------------ Postprocessing: compute energy, dissipation, etc.
    tic = timer()

    fname_ev = 'eigenvalues0.dat'
    fname_tm = 'timing.dat'

    if os.path.isfile(fname_ev):
        eigval = np.loadtxt(fname_ev).reshape((-1,2))

    timing = 0.0  # solve time, read from timing.dat if present
    if os.path.isfile(fname_tm):
        timing = np.loadtxt(fname_tm)
        if np.size(timing)>1:
            timing = timing[-1]

    # NEW (2026-11-08, from conductive_IC): the solutions from flow.npy, ... or from real_flow.field, imag_flow.field, ...
    # (utils.load_field: the two formats give the same numbers); a missing file stops with its name
    def field(name, rows):
        x = ut.load_field(name)
        if x is None:
            sys.exit('spin_doctor.py: no solution file of the %s (%s.npy, or real_%s.field and imag_%s.field) in %s'
                     % (name, name, name, name, os.getcwd()))
        return x.reshape((rows, -1))

    if par.hydro:
        u_all = field('flow', 2*ut.n)
        ru, iu = u_all.real.copy(), u_all.imag.copy()
    if par.thermal:
        t_all = field('thermal', ut.n)
        rt, it = t_all.real.copy(), t_all.imag.copy()
    if par.magnetic:  # NEW (2026-10-28)
        b_all = field('magnetic', 2*ut.n)
        rb, ib = b_all.real.copy(), b_all.imag.copy()
    bic = None
    if ut.cic:  # NEW (2026-10-28): the field of a conducting inner core
        bic = field('magnetic_ic', 2*ut.nic)
    bman = None
    if ut.resolved_mantle:  # NEW (2026-11-06): the field of the resolved mantle
        bman = field('magnetic_mantle', 2*ut.nman)
    rot = None
    if ut.rotdyn:  # NEW (2026-11-06): omega_M, omega_IC, phi of the rotating solids
        rot = field('rotdyn', 3)
    if par.compositional:  # NEW (2026-11-08): the composition
        rc = field('composition', ut.n)

    # NEW (2026-11-06): without the flow (hydro = 0) the solutions are counted in the field's (or the entropy's) file, and the
    # flow's diagnostics run on a zero flow (upp.zero_flow): they give zeros, and residσ, residω, residens are not computed
    # (NEW 2026-11-08: or the composition's)
    success = np.shape(ru if par.hydro else (rb if par.magnetic else (rt if par.thermal else rc)))[1]

    # ------------------------------------------------------------------------------------------------------------------
    # hydrodynamical variables to be processed
    KE          = np.zeros(success)
    KP          = np.zeros(success)
    KT          = np.zeros(success)
    Dkin        = np.zeros(success)
    Ensvel      = np.zeros(success)
    Enscor      = np.zeros(success)
    Ensvif      = np.zeros(success)
    Ensbuo      = np.zeros(success)
    cuvismax    = np.zeros(success)
    cuvismax_r  = np.zeros(success)
    cuvismax_l  = np.zeros(success, dtype=int)
    Wthm        = np.zeros(success)
    Wcor_im     = np.zeros(success)  # imaginary parts of the Coriolis, viscous and buoyancy powers
    Dkin_im     = np.zeros(success)
    Wthm_im     = np.zeros(success)
    Wdr         = np.zeros(success)  # power of the differential-rotation force, real and imaginary parts
    Wdr_im      = np.zeros(success)
    Ensdr       = np.zeros(success)
    Wp_icb      = np.zeros(success)  # NEW (2026-11-04): power of the pressure at the inner and outer walls (real part),
    Wp_cmb      = np.zeros(success)  #   and the imaginary part of their sum (upp.wall_powers)
    Wp_im       = np.zeros(success)
    Wv_icb      = np.zeros(success)  # power of the viscous stress at the inner and outer walls (upp.wall_powers)
    Wv_cmb      = np.zeros(success)
    Dvisc       = np.zeros(success)  # viscous dissipation ∫ 𝛔:∇𝐮 dV (upp.viscous_dissipation)
    Wf          = np.zeros(success)  # NEW (2026-11-08): forcings 1 and 8, the power of the body force (upp.body_force_terms),
    Wf_im       = np.zeros(success)  #   its imaginary part, its enstrophy term and its torque on the fluid
    Ensf        = np.zeros(success)
    Tf          = np.zeros(success, dtype=complex)
    Wbg         = np.zeros(success)  # NEW (2026-11-08): a background flow (bgflow = 1), the power of its force on the flow,
    Wbg_im      = np.zeros(success)  #   its imaginary part and its enstrophy term (upp.bgflow_setup)
    Ensbg       = np.zeros(success)
    residD      = np.full(success, np.nan)  # relative residual of Dkin = Wv_icb + Wv_cmb - Dvisc
    ldom        = np.zeros(success,dtype=int)
    lwidth      = np.zeros(success,dtype=int)
    lconv       = np.zeros(success)
    irreg0      = np.full(success, np.nan)  # NEW (2026-11-04): ricb = 0, the part of the flow and field not regular at r = 0

    # thermal variables to be processed
    TE          = np.zeros(success)
    Wadv_thm    = np.zeros(success)
    Dthm        = np.zeros(success)
    Wdr_thm     = np.zeros(success)
    Wbg_thm     = np.zeros(success)  # NEW (2026-11-08): -2 Re ∫ p₀ s* (𝐔⋅∇s + 𝐮⋅∇S) dV, a background flow

    # NEW (2026-11-08): the composition (compositional = 1): its budget (upp.composition_worker) and the power of its buoyancy
    CE          = np.zeros(success)  # ½ ∫ ρ₀ |c|² dV
    Wadv_cmp    = np.zeros(success)  # -∫ ρ₀ (dC/dr) uᵣ c* dV
    Dcmp        = np.zeros(success)  # CompoD ∫ c* ∇⋅(D ρ₀ ∇c) dV
    Wdr_cmp     = np.zeros(success)  # -2 Re ∫ ρ₀ c* i m G ΔΩ c dV
    Wbg_cmp     = np.zeros(success)  # -2 Re ∫ ρ₀ c* (𝐔⋅∇c + 𝐮⋅∇C) dV, a background flow (NEW 2026-11-08)
    Wcmp        = np.zeros(success)  # 2 Re ∫ ρ 𝐮*⋅(BeyonceC g c 𝐫̂) dV (in residσ), and its imaginary part (in residω)
    Wcmp_im     = np.zeros(success)
    resid4      = np.zeros(success)  # resid𝑐, the budget of the composition

    # magnetic variables to be processed (NEW 2026-10-28)
    ME          = np.zeros(success)  # ∫ ½ 𝐛⋅𝐛 dV (as conductive_IC's magnetic.dat: without Hendrik)
    Mdfs        = np.zeros(success)  # MagnetD ∫ 𝐛⋅∇²𝐛 dV
    Indu        = np.zeros(success)  # ∫ 𝐛⋅∇×(𝐮×𝐁₀) dV
    Wlor        = np.zeros(success)  # Hendrik 2 Re ∫ 𝐮*⋅𝐅 dV, the power of the Lorentz force, and its imaginary part
    Wlor_im     = np.zeros(success)
    residb      = np.zeros(success)
    MEic        = np.zeros(success)  # ½ ∫ |𝐛|² inside the inner core (conducting, or the potential field of an insulating one)
    MEext       = np.zeros(success)  # ½ ∫ |𝐛|² of the potential field outside the outer wall
    Ohm_f       = np.zeros(success)  # MagnetD ∫ |∇×𝐛|² dV over the fluid
    Ohm_ic      = np.zeros(success)  # MagnetD η_i ∫ |∇×𝐛|² dV over a conducting inner core
    residE      = np.full(success, np.nan)  # the full energy budget (impenetrable walls; nan where it does not apply)
    Iwall       = np.zeros(success)  # NEW (2026-10-30): the wall terms in Indu (utils4pp.INDU_WALLS)
    E_w         = np.zeros(success)  # NEW (2026-11-06): thin walls, the energy stored in them (without Hendrik) and their Ohmic
    Ohm_w       = np.zeros(success)  #   loss (MagnetD, upp.em_budget)
    twaerr      = np.zeros((success, 4))   # NEW (2026-11-06): h/δ_w and ε_TWA at the outer and the inner wall (upp.twa_error)
    use_twa     = bool(ut.twa_o or ut.twa_i)
    use_walls   = use_twa or bool(ut.resolved_mantle)   # NEW (2026-11-06): E_w, Ohm_w are those of the resolved mantle too
    # NEW (2026-11-06, from conductive_IC): the angular momentum of the fluid and the torques on its walls, axial (m = 0, symm = 1)
    # or equatorial (m = 1, symm = -1), rotdyn.py, for a spherical body; resid𝐋 the balance λL = (the torques on the fluid),
    # where rd.L_case()
    do_torques  = bool(par.hydro and not par.centrifugal and ((par.m == 0 and par.symm == 1) or (par.m == 1 and par.symm == -1)))
    L_case      = rd.L_case()
    angmom      = np.zeros(success, dtype=complex)  # L (rotdyn.angular_momentum)
    vtorq       = np.zeros(success, dtype=complex)  # viscous torque on the mantle (on everything beyond the outer wall)
    vtorq_ic    = np.zeros(success, dtype=complex)  #   and on the inner core
    mtorq       = np.zeros(success, dtype=complex)  # magnetic torque on the mantle (the Maxwell stress on the fluid's side)
    mtorq_ic    = np.zeros(success, dtype=complex)  #   and on the inner core
    residL      = np.full(success, np.nan)
    # NEW (2026-11-06, from conductive_IC): the drag at a stress-free outer wall (topo.py): Dtopo, the power it dissipates, and
    # dsig = -Dtopo/(2 E), the damping rate it implies (E the energy of the mode); per unit kappa_cmb without a drag
    do_drag     = bool(par.hydro and par.ViscosD > 0 and par.bco == 0 and par.Gaspard > 0 and not par.centrifugal)
    Dtopo       = np.zeros(success)
    dsig        = np.zeros(success)
    MEc         = np.zeros(success)  # NEW (2026-11-05): centrifugal = 1, ∫ 𝐛̃*⋅(M - I) 𝐛̃ dV: ME + MEc is the field's energy in the
    InduM       = np.zeros(success)  #   deformed fluid; InduM, Ohmc the induction work and Ohmic loss (MagnetD) on the metric part
    Ohmc        = np.zeros(success)  #   of the field, for resid𝐄 (upp.centrifugal_magnetic_terms)

    # residual errors to be processed
    resid1      = np.zeros(success)
    resid2      = np.zeros(success)
    resid3      = np.zeros(success)
    resens      = np.zeros(success)

    # parameter values to be saved
    params      = []                 # one row of param_table() per solution
    # ------------------------------------------------------------------------------------------------------------------------

    sigmas      = np.zeros(success)  # Re λ of each solution (kore units; bin/units.py converts λ to ω, σ in nHz)
    freqs       = np.zeros(success)  # Im λ of each solution (forced: the forcing frequency)

    # Summary table printed while processing: (header, format, value of solution i).
    # To print a new quantity, add a line here.
    table_cols = [
        ( '★',             '{:d}',      lambda i: i ),
        ( 'Re λ',          '{: .9f}',   lambda i: sigmas[i] ),
        ( 'Im λ',          '{: .9f}',   lambda i: freqs[i] ),
    ] + ([   # NEW (2026-11-06): the flow's columns only with the flow (hydro = 1)
        ( '𝒯/𝒫',           '{:8.2e}',   lambda i: ratio(KT[i], KP[i]) ),
        ( 'residσ',        '{:8.2e}',   lambda i: resid1[i] ),
        ( 'residω',        '{:8.2e}',   lambda i: resid2[i] ),
    ] if par.hydro else []) + ([
        ( 'resid𝑠',        '{:8.2e}',   lambda i: resid3[i] ),
    ] if par.thermal else []) + ([   # NEW (2026-11-08): the budget of the composition
        ( 'resid𝑐',        '{:8.2e}',   lambda i: resid4[i] ),
    ] if par.compositional else []) + ([   # NEW (2026-11-04): forced runs, the power the walls put in and the viscous dissipation
        ( 'P_in',          '{: .6e}',   lambda i: Wp_icb[i] + Wp_cmb[i] + Wv_icb[i] + Wv_cmb[i] + Wf[i] ),
        ( 'Dν',            '{: .6e}',   lambda i: Dvisc[i] ),
    ] if par.forcing > 0 else []) + ([
        ( 'resid𝐛',        '{:8.2e}',   lambda i: residb[i] ),
        ( 'resid𝐄',        '{:8.2e}',   lambda i: residE[i] ),
    ] if par.magnetic else []) + ([   # NEW (2026-11-06): the estimated error of the thin walls, the larger of the two
        ( 'εTWA',          '{:8.2e}',   lambda i: max(twaerr[i,1], twaerr[i,3]) ),
    ] if use_twa else []) + ([   # NEW (2026-11-06): the angular momentum budget of the fluid (rotdyn.py)
        ( 'resid𝐋',        '{:8.2e}',   lambda i: residL[i] ),
    ] if L_case else []) + ([   # NEW (2026-11-06): the drag at the outer wall (per unit kappa_cmb without one: '/κ')
        ( 'Dtopo' + ('' if topo.on() else '/κ'),   '{:8.2e}',   lambda i: Dtopo[i] ),
        ( 'δσtopo' + ('' if topo.on() else '/κ'),  '{:9.2e}',   lambda i: dsig[i] ),
    ] if do_drag else []) + ([
        # FIX (2026-11-06): in the units of KE, Hendrik (ME + MEc)/KE, as resid𝐄 and the normalisation KE + Hendrik ME = 1
        # (as conductive_IC 7660861; it was ME/KE, off by the factor Hendrik). Not without the flow (hydro = 0)
        ( 'Mag/Kin',       '{:8.2e}',   lambda i: ratio(par.Hendrik*(ME[i] + MEc[i]), KE[i]) ),
    ] if par.magnetic and par.hydro else []) + [
        ( 'Peak ℓ',        '{:d}',      lambda i: ldom[i] ),
        ( 'ℓ-Width',       '{:d}',      lambda i: lwidth[i] ),
        ( 'ℓ-Convergence', '{:8.2e}',   lambda i: lconv[i] ),
    ] + ([
        ( 'cvf_r',         '{:5.3f}',   lambda i: cuvismax_r[i] ),
        ( 'cvf_l',         '{:d}',      lambda i: cuvismax_l[i] ),
        ( 'cvfmax',        '{:8.2e}',   lambda i: cuvismax[i] ),
        ( 'residens',      '{:8.2e}',   lambda i: resens[i] ),
    ] if par.hydro else []) + ([   # NEW (2026-11-04): a full sphere, the Taylor terms at r = 0 that a regular solution lacks
        ( 'irreg₀',        '{:8.2e}',   lambda i: irreg0[i] ),   # (upp.centre_irregularity)
    ] if par.ricb == 0 else [])
    table_w = table_widths(table_cols)
    table_head, table_line = table_header(table_cols, table_w)

    print('\n' + table_head)
    print(table_line)


    # FIX (2026-11-08): the forced runs whose walls drive the flow along themselves, the libration (forcing 7, m = 0) and the
    # steady rotation of the walls (12). There Dkin = Wv - Dvisc is a small difference of two large terms, the power of the
    # walls' viscous stress and the dissipation, so residσ and resid𝐄 are relative to Dvisc too (as conductive_IC 4bfb389)
    walls_drive = par.forcing == 12 or (par.forcing == 7 and par.m == 0)
    # NEW (2026-11-08): the body force of a known flow, forcings 1 and 8 (assemble.lifted_rhs): its power, enstrophy term and
    # torque (upp.body_force_terms) enter residσ, residω, residens, resid𝐄 and resid𝐋, and P_in
    body_force = par.forcing in (1, 8)

    # Begin processing all solutions
    for i in range(success):

        if par.forcing == 0:
            w = eigval[i,1]
            sigma = eigval[i,0]
        else:
            w = ut.wf
            sigma = 0
        sigmas[i], freqs[i] = sigma, w

        t_sol2 = 0

        if par.hydro:
            rflow = np.copy(ru[:,i])
            iflow = np.copy(iu[:,i])
            # Expand solution
            u_sol2 = upp.expand_reshape_sol( rflow + 1j*iflow, par.symm)
        else:
            u_sol2 = upp.zero_flow()   # NEW (2026-11-06)
        [ lp, lt, ll] = ut.ell(par.m, par.lmax, par.symm)
        lpi = np.searchsorted(ll,lp);  # Poloidal indices
        lti = np.searchsorted(ll,lt);  # Toroidal indices


        if par.thermal:
            rthm = np.copy(rt[:,i])
            ithm = np.copy(it[:,i])
            # Expand solution
            t_sol2  = upp.expand_reshape_sol( rthm + 1j*ithm, par.symm)


        if par.magnetic:  # NEW (2026-10-28)
            b_sol2 = upp.expand_reshape_sol( np.copy(rb[:,i]) + 1j*np.copy(ib[:,i]), ut.bsymm)

        c_sol2 = upp.expand_reshape_sol( np.copy(rc[:,i]), par.symm ) if par.compositional else 0   # NEW (2026-11-08)


        # identify solutions (NEW 2026-11-06: without the flow, the field, or the entropy; NEW 2026-11-08: or the composition)
        if par.hydro:
            [ ldom[i], lwidth[i], lconv[i] ] = upp.identify( u_sol2 )
        elif par.magnetic or par.thermal:
            [ ldom[i], lwidth[i], lconv[i] ] = upp.identify( b_sol2, ut.bsymm ) if par.magnetic else upp.identify( t_sol2 )
        else:
            [ ldom[i], lwidth[i], lconv[i] ] = upp.identify( c_sol2 )

        if par.ricb == 0:  # NEW (2026-11-04): how far the flow (and the field) is from regular at r = 0
            groups = [ [ (u_sol2[0], lp), (u_sol2[1], lt) ] ] if par.hydro else []
            if par.magnetic:
                lbp, lbt = ut.ell(par.m, par.lmax, ut.bsymm)[:2]
                groups.append( [ (b_sol2[0], lbp), (b_sol2[1], lbt) ] )
            if par.thermal and not par.hydro:   # NEW (2026-11-06): the entropy, without the flow
                groups.append( [ (t_sol2, lp) ] )
            if par.compositional and not par.hydro:   # NEW (2026-11-08): the composition, without the flow
                groups.append( [ (c_sol2, lp) ] )
            irreg0[i] = upp.centre_irregularity(groups)

        # diagnose solutions (serially since 2026-10-20, see upp.diagnose)
        [ udgn, tdgn ] = upp.diagnose( u_sol2, t_sol2, par.ricb, ut.rcmb, int(ncpus), csol=c_sol2 )
        if par.compositional:  # NEW (2026-11-08): the budget of the composition, on the grid and tables of diagnose
            cdgn = upp.diagnose_composition()

        if par.centrifugal:  # NEW (2026-11-05): the metric parts of KE, of the Coriolis power and of TE, on the grid of diagnose
            [ KEc, Wcorc, TEc ] = upp.centrifugal_terms( par.ricb, ut.rcmb )
            # NEW (2026-11-06, phase 2): those of the viscous power, of the differential rotation and of the entropy diffusion
            [ Dkinc, Wdrc, Dthmc ] = upp.centrifugal_dissipation_terms( par.ricb, ut.rcmb )
            if par.compositional:   # NEW (2026-11-08): those of CE and of the diffusion of the composition
                [ CEc, Dcmpc ] = upp.centrifugal_composition_terms( par.ricb, ut.rcmb )

        if par.magnetic:  # NEW (2026-10-28): on the grid and flow tables that diagnose has just set
            [ ME[i], Mdfs0, Indu[i], Wlor0, Wlor_im0 ] = np.sum( upp.diagnose_magnetic( b_sol2 ), 0)
            # NEW (2026-10-30): the wall terms of Indu that leave the fluid for the energy beyond the walls. FIX (2026-11-06):
            # not U0×b where a conducting solid turns with ΔΩ (upp.INDU_ROT_SOLID: the induction work of its rotation)
            Iwall[i]   = upp.INDU_WALLS - upp.INDU_ROT_SOLID
            Mdfs[i]    = par.MagnetD * Mdfs0
            Wlor[i]    = par.Hendrik * Wlor0
            Wlor_im[i] = par.Hendrik * Wlor_im0
            # the parts of the magnetic energy budget outside the fluid, and the Ohmic losses (upp.em_budget, conductive_IC's)
            Ohm_f0, MEic[i], Ohm_ic0, MEext[i], E_w[i], Ohm_w0 = upp.em_budget( b_sol2, None if bic is None else bic[:,i],
                                                                                u_sol2 if par.hydro else None,
                                                                                None if bman is None else bman[:,i] )
            Ohm_f[i]  = par.MagnetD * Ohm_f0
            Ohm_ic[i] = par.MagnetD * Ohm_ic0
            Ohm_w[i]  = par.MagnetD * Ohm_w0   # NEW (2026-11-06): thin walls or the resolved mantle (0 without)
            if par.hydro and (bg.deforms('inner') or bg.deforms('outer')):
                # NEW (2026-11-06): the work of the motion of a thick conducting solid that deforms with its wall (forced_walls
                # 'deforming') on its own field, a volume integral over the solid (upp.deform_work), part of the induction work
                Iwall[i] -= upp.deform_work( u_sol2, None if bic is None else bic[:,i], None if bman is None else bman[:,i] )
            if use_twa:
                twaerr[i] = upp.twa_error(sigma + 1j*w)
            if par.centrifugal:  # NEW (2026-11-05, phase 1b): the metric parts, on the same grid (upp.centrifugal_magnetic_terms)
                cm = upp.centrifugal_magnetic_terms( b_sol2, None if bic is None else bic[:,i] )
                Wlor[i]    += par.Hendrik * np.real(cm['Wlor'])
                Wlor_im[i] += par.Hendrik * np.imag(cm['Wlor'])
                Indu[i]    += cm['Indu']
                Iwall[i]   += cm['Iwall']
                MEc[i], InduM[i], Ohmc[i] = cm['MEc'], cm['InduM'], par.MagnetD * cm['Ohmc']
                Mdfs[i]   += par.MagnetD * cm['Mdfs']         # NEW (2026-11-06, phase 2): the metric diffusion
                MEic[i], MEext[i] = cm['MEic'], cm['MEext']   # the potential fields of the deformed exterior and inner core,
                Ohm_ic[i]  = par.MagnetD * cm['Ohm_ic']       # and a conducting inner core with its metric part

        # NEW (2026-11-04): the viscous dissipation at the grid of diagnose, then the work of the wall stresses
        # (wall_powers moves the grid globals to the walls). NEW (2026-11-06): none without the flow
        if par.ViscosD>0 and par.hydro:
            Dvisc[i] = upp.viscous_dissipation()
        if par.hydro:
            [ Wp0, Wv0 ] = upp.wall_powers(u_sol2, t_sol2, sigma + 1j*w, b_sol2 if par.magnetic else None, csol=c_sol2)
            [ Wp_icb[i], Wp_cmb[i] ] = np.real(Wp0)
            Wp_im[i] = np.sum(np.imag(Wp0))
            [ Wv_icb[i], Wv_cmb[i] ] = Wv0
        if body_force:   # NEW (2026-11-08): forcings 1 and 8 (after wall_powers: it resets the grid and the tables)
            [ Wf0, Ensf[i], Tf[i] ] = upp.body_force_terms(u_sol2, sigma + 1j*w, t_sol2, c_sol2)
            Wf[i], Wf_im[i] = np.real(Wf0), np.imag(Wf0)

        if do_torques:   # NEW (2026-11-06): the angular momentum of the fluid and the torques on its walls (rotdyn.py)
            angmom[i] = rd.angular_momentum(u_sol2)
            vtorq[i], vtorq_ic[i] = rd.viscous_torques(u_sol2)
            if par.magnetic:
                mtorq[i], mtorq_ic[i] = rd.wall_torques(b_sol2)

        if par.ViscosD>0 and par.hydro:
            Ra = par.ricb
            Rb = ut.rcmb
            ii = np.arange(0,par.N)
            xk = np.cos( (ii+0.5)*np.pi/par.N )
            rk = np.flipud(0.5*(Rb-Ra)*( xk + 1 ) + Ra)
            cuvis = upp.diagnose_4plot(int(ncpus), u_sol2, t_sol2, rk, 'curl_vis')
            cvt = abs(cuvis[:,2,0,:])
            if par.ricb == 0:
                # a degree-l component is ∝ r^l, so near r = 0 the evaluated series is round-off which the
                # derivatives and 1/r factors of the curl amplify (1e8 and more at the innermost node): skip r < 0.01
                cvt[:, rk < 0.01] = 0
            idmax = np.unravel_index(np.argmax(cvt), cvt.shape)
            # the max value of the toroidal component of the curl of the viscous force is
            cuvismax[i] = abs(cuvis[idmax[0],2,0,idmax[1]])
            cuvismax_l[i] = idmax[0]
            cuvismax_r[i] = rk[idmax[1]]



        KP[i] = np.sum( udgn[lpi,0])  # Poloidal kinetic energy
        KT[i] = np.sum( udgn[lti,0])  # Toroidal kinetic energy

        [ KE[i], Dkin0, Ensvel[i], Enscor[i], Ensvif[i], Ensbuo[i], Wthm0, Wdr[i],
          Wcor_im[i], Dkin_im[i], Wthm_im[i], Wdr_im[i], Ensdr[i], Wcmp[i], Wcmp_im[i], Wbg[i], Wbg_im[i], Ensbg[i] ] = np.sum( udgn, 0)
        Dkin[i] = Dkin0
        # (NEW 2026-11-08: Ensbg, the enstrophy term of a background flow)
        resens[i] = abs(Ensvel[i]*sigma+Enscor[i]-Ensbuo[i]-Ensvif[i]-Ensdr[i]-Ensf[i]-Ensbg[i]) / max((abs(Ensvel[i]*sigma),abs(Enscor[i]),abs(Ensbuo[i]),abs(Ensvif[i]),abs(Ensdr[i]),abs(Ensf[i]),abs(Ensbg[i]))) \
                    if par.hydro else np.nan   # NEW (2026-11-06): no flow, no enstrophy budget
        if par.centrifugal:  # NEW (2026-11-05): KE of the deformed body (∫ ρ0 w*⋅M w dV), Coriolis power with z̃ = F⁻¹ẑ
            KE[i] += KEc
            Wcor_im[i] += np.imag(Wcorc)
            resens[i] = np.nan   # the enstrophy budget has no metric terms: not computed
            # NEW (2026-11-06, phase 2): the powers of the viscous force and of the differential rotation of the deformed
            # body; Dvisc and the wall powers stay those of the sphere, so residD is not computed
            Dkin[i] += np.real(Dkinc); Dkin_im[i] += np.imag(Dkinc)
            Wdr[i] += np.real(Wdrc); Wdr_im[i] += np.imag(Wdrc)
        Wthm[i] = Wthm0  # rate of working of buoyancy, Beyonce already included (upp.buoyancy)


        if par.thermal:

            [ TE[i], Dthm0, Wadv_thm[i], Wdr_thm[i], Wbg_thm[i] ] = np.sum( tdgn, 0)
            Dthm[i] = Dthm0 * par.ThermaD
            if par.centrifugal:  # NEW (2026-11-05): ∫ p0 J |s|² dV, the entropy "energy" of the deformed body
                TE[i] += TEc
                Dthm[i] += par.ThermaD * Dthmc   # NEW (2026-11-06, phase 2): its metric diffusion

        if par.compositional:   # NEW (2026-11-08): the budget of the composition, as that of the entropy

            [ CE[i], Dcmp0, Wadv_cmp[i], Wdr_cmp[i], Wbg_cmp[i] ] = np.sum( cdgn, 0)
            Dcmp[i] = Dcmp0 * par.CompoD
            if par.centrifugal:  # ∫ ρ0 J |c|² dV and the metric diffusion
                CE[i] += CEc
                Dcmp[i] += par.CompoD * Dcmpc


        # --------------------------------------------------------- Computing residuals to check the power balance:
        # KE is kinetic energy
        # TE is the entropy "energy" (1/2) ∫ p s² dV

        # Dkin is the power of the viscous force ∫ 𝐮⋅(∇⋅𝛔) dV = Wv - Dvisc (minus the viscous dissipation Dvisc if 𝐮 = 0 at the walls)
        # Dthm is the entropy "dissipation" via ThermaD ∫ s ∇⋅(κp∇s) dV = -ThermaD ∫ κp|∇s|² dV (+ surface term, zero for kore's thermal BCs)

        # Wthm is the rate of working of the buoyancy force (thermal) via ∫ ρ 𝐮⋅(Beyonce g s 𝐫̂) dV

        # Wadv is the entropy advection "power" via -∫ p (dS/dr) uᵣ s dV

        # Wdr is the power of the differential-rotation force 𝐅 = -G[(𝐔₀′⋅∇)𝐮 + (𝐮⋅∇)𝐔₀′], 2 Re ∫ ρ 𝐮*⋅𝐅 dV, Wdr_im its 2 Im,
        #     Ensdr its enstrophy term (zero without diff_rot)
        # Wdr_thm is -2 Re ∫ p s* [ i m G ΔΩ s + 𝐮⋅∇S_tw ] dV, the differential-rotation terms of the heat equation

        # Wp is the power of the pressure at the walls, -2 Re ∮ ρ₀ Π uᵣ* dA (Wp_im its imaginary part), zero unless uᵣ ≠ 0
        #     at a wall (forced boundary flows, bc 2); Dkin = Wv - Dvisc, Wv the power of the viscous stress at the walls
        # resid1 is the relative residual of 2*sigma*KE - Dkin - Wthm - Wdr - Wp = 0 (residσ, real part of the power balance)
        # resid2 is the relative residual of 2*w*KE + Wcor_im - Dkin_im - Wthm_im - Wdr_im - Wp_im = 0 (residω, its imaginary part;
        #        Wcor_im = 2 Im ∫ ρ 𝐮*⋅(2𝐳×𝐮) dV, Dkin_im = 2 Im ∫ 𝐮*⋅(∇⋅𝛔) dV, Wthm_im = 2 Im ∫ ρ 𝐮*⋅(Beyonce g s 𝐫̂) dV)
        # resid3 is the relative residual of 2*sigma*TE - Dthm - Wadv - Wdr_thm = 0
        # NEW (2026-11-08): Wcmp (Wcmp_im) is the power of the buoyancy of the composition, in resid1 (resid2) and resid𝐄, and
        #     resid4 (resid𝑐) the relative residual of 2*sigma*CE - Dcmp - Wadv_cmp - Wdr_cmp = 0, the budget of the composition
        # NEW (2026-11-08): Wf (Wf_im) is the power of the body force of forcings 1 and 8, in resid1 (resid2) and in P_in,
        #     Ensf its enstrophy term and Tf its torque in resid𝐋 (upp.body_force_terms; zero for the other forcings)
        # NEW (2026-11-08): Wbg (Wbg_im) is the power of the force of a background flow 𝐔 (bgflow = 1), -[(𝐔⋅∇)𝐮 + (𝐮⋅∇)𝐔],
        #     in resid1 (resid2) and resid𝐄, Ensbg its enstrophy term in resens; Wbg_thm (Wbg_cmp) -2 Re ∫ p₀ s* (𝐔⋅∇s + 𝐮⋅∇S)
        #     dV (the same with ρ₀, c, C) in resid3 (resid4) (upp.bgflow_setup; zero without)
        # ---------------------------------------------------------------------------------------------------------


        # NEW (2026-10-28): with magnetic = 1, Wlor (Wlor_im) is the power of the Lorentz force in residσ (residω), and
        # residb (resid𝐛) is the relative residual of 2*sigma*ME - Indu - Mdfs = 0 (the induction equation dotted with 𝐛*, in
        # the fluid; as conductive_IC's resid𝐛)
        Wp = Wp_icb[i] + Wp_cmb[i]
        if par.hydro:   # NEW (2026-11-06): nan without the flow
            # FIX (2026-11-08): relative to the viscous dissipation Dvisc too where the walls drive the flow (walls_drive):
            # without a field, buoyancy or a differential rotation every other term is small, and residσ was the round-off of
            # Dkin relative to itself, about 1
            scale1 = max( abs(2*sigma*KE[i]), abs(Dkin[i]), abs(Wthm[i]), abs(Wdr[i]), abs(Wlor[i]), abs(Wp), abs(Wf[i]),
                          abs(Dvisc[i]) if walls_drive else 0.0, abs(Wcmp[i]), abs(Wbg[i]) )
            resid1[i] = abs( 2*sigma*KE[i] - Dkin[i] - Wthm[i] - Wcmp[i] - Wdr[i] - Wlor[i] - Wp - Wf[i] - Wbg[i] )/scale1
            # FIX (2026-11-08): 0 when every term is 0, a steady flow of m = 0 (forcing 12, forcing 13 at frequency 0), whose
            # fields are real: the imaginary part of the balance holds trivially (it was 0/0)
            scale2 = max( abs(2*w*KE[i]), abs(Wcor_im[i]), abs(Dkin_im[i]), abs(Wthm_im[i]), abs(Wdr_im[i]), abs(Wlor_im[i]),
                          abs(Wp_im[i]), abs(Wf_im[i]), abs(Wcmp_im[i]), abs(Wbg_im[i]) )
            resid2[i] = abs( 2*w*KE[i] + Wcor_im[i] - Dkin_im[i] - Wthm_im[i] - Wcmp_im[i] - Wdr_im[i] - Wlor_im[i] - Wp_im[i]
                             - Wf_im[i] - Wbg_im[i] )/scale2 if scale2 > 0 else 0.0
        else:
            resid1[i] = resid2[i] = np.nan
        if par.ViscosD>0 and not par.centrifugal and par.hydro:   # NEW (2026-11-06, phase 2): not with centrifugal = 1 (Dvisc, Wv spherical)
            Wv = Wv_icb[i] + Wv_cmb[i]
            residD[i] = abs( Dkin[i] - Wv + Dvisc[i] ) / max( abs(Dkin[i]), abs(Wv), abs(Dvisc[i]) )
        # NEW (2026-11-06, from conductive_IC): residL (resid𝐋), the angular momentum of the fluid, λL = the torques on it: minus
        # those on the mantle and the inner core (viscous, and the Maxwell stress at the walls), and i Gaspard L, the Coriolis
        # torque, in the equatorial case (conductive_IC's sign: L is L_x - i L_y there). Relative to the largest term and to
        # |λ| ∫ ρ0 r³ |T| (rotdyn.angular_momentum), so that a flow with no torque on it (stress-free or inviscid walls, m = 0)
        # tests that it carries no net angular momentum
        if L_case:
            lam = sigma + 1j*w
            on_fluid = [ -vtorq[i], -vtorq_ic[i], -mtorq[i], -mtorq_ic[i], Tf[i] ] + ([ 1j*par.Gaspard*angmom[i] ] if par.m == 1 else [])
            Labs = rd.angular_momentum(u_sol2, absval=True)
            scale = max([ abs(lam)*Labs, par.Gaspard*Labs if par.m == 1 else 0.0 ] + [ abs(t) for t in on_fluid ])
            residL[i] = abs( lam*angmom[i] - sum(on_fluid) )/scale if scale > 0 else 0.0
        if par.magnetic:
            # FIX (2026-11-08): 0 when every term is 0, a field that the flow leaves unchanged (it was 0/0): the Malkus field
            # B0 ∝ s φ̂ under an axisymmetric flow (forcing 12), which neither shears it nor carries B0/s, uniform, around
            scaleb = max( abs(2*sigma*ME[i]), abs(Indu[i]), abs(Mdfs[i]) )
            residb[i] = abs( 2*sigma*ME[i] - Indu[i] - Mdfs[i] )/scaleb if scaleb > 0 else 0.0
            # residE (resid𝐄, as conductive_IC): 2σ(KE + Hendrik (ME + MEic + MEext)) = Dkin + Wthm + Wlor + Hendrik Indu
            # - Hendrik (Ohm_f + Ohm_ic): the Lorentz work and the induction cancel except for the work of the background
            # currents, and the Ohmic losses replace ∫ 𝐛⋅∇²𝐛. It tests the wall and ICB conditions, which resid𝐛 cannot see.
            # Needed no-slip walls (the spectral Indu holds the u×B0 flux through slipping walls, which it cannot tell apart).
            # NEW (2026-10-30): with a differential rotation, the power of its force on the flow (Wdr) enters too, and the
            # induction work is the volume part ∫ 𝐄⋅(∇×𝐛)* alone (Indu minus its wall terms upp.INDU_WALLS): U0×b at a wall
            # carries magnetic energy through it, into the field outside, which ME_ext and ME_ic already count.
            # FIX (2026-11-01): bci only where there is an inner wall (at ricb = 0 it is ignored, and the default bci = 0 left
            # resid𝐄 at nan for every full-sphere run). NEW (2026-11-06): without the flow (hydro = 0) for any walls, and
            # without Hendrik (only magnetic terms are left: 2σ(ME + MEic + MEext) = Indu - Iwall - Ohm_f - Ohm_ic)
            # NEW (2026-11-06, as conductive_IC): a fluid slipping along impenetrable walls (stress-free, or ViscosD = 0) too, with
            # the wall terms of u×B0 apart (upp.diagnose_magnetic computes INDU_WALLS then); not with centrifugal = 1. Thin walls
            # (mantle, innercore 'TWA') add the energy stored in them and their Ohmic loss (E_w, Ohm_w; upp.em_budget), which
            # close the budget exactly for c = 0 and to O(c c1) otherwise; not with mu != 1 (the energies would need it). The
            # resolved mantle adds its energy and Ohmic loss the same way (exact).
            noslip = par.bco == 1 and (par.bci == 1 or par.ricb == 0)
            imperm = par.ViscosD == 0 or (par.bco in (0, 1) and (par.ricb == 0 or par.bci in (0, 1)))
            # NEW (2026-11-06): with rotating solids (rotdyn = 1) only with both torques on them (vtrq = mtrq = 1)
            rotok = not ut.rotdyn or (par.vtrq == 1 and par.mtrq == 1)
            # NEW (2026-11-06): walls that move radially too (forced problems, a free wall bco or bci = 2), with the work of the
            # pressure there (Wpe below)
            moving = upp.walls_move()
            if (noslip or not par.hydro or ((imperm or moving) and not par.centrifugal)) and not (use_twa and par.mu != 1) and rotok:
                H = par.Hendrik if par.hydro else 1.0
                # NEW (2026-11-05): with centrifugal = 1 the energy of the deformed body, ∫ 𝐛̃*⋅M 𝐛̃ dV in the fluid (MEc), whose
                # rate is the work 2 Re ∫ 𝐄⋅(∇×M 𝐛̃)* dV (InduM) minus the Ohmic loss (Ohmc); closes at second order in f
                Etot = KE[i] + H*( ME[i] + MEc[i] + MEic[i] + MEext[i] )
                Wex  = Wlor[i] + H*(Indu[i] - Iwall[i] + InduM[i]) + Wdr[i] + Wbg[i]   # (NEW 2026-11-08: Wbg)
                Ohm  = H*( Ohm_f[i] + Ohmc[i] + Ohm_ic[i] )
                if use_walls:   # NEW (2026-11-06)
                    Etot += H*E_w[i]
                    Ohm  += H*Ohm_w[i]
                # NEW (2026-11-06, as conductive_IC): the rotating solids add their rotational energies and the gravitational
                # energy, I_M |ω_M|² + I_IC |ω_IC|² + K |φ|² in the units of KE (a rigid rotation ω of the fluid has KE = I |ω|²),
                # the power of the viscous torques on them, which is minus the work of the moving walls in Dkin, and the
                # dissipation of the inner core's relaxation, 2 K (Gaspard/OmgtauIC) |φ|². Their magnetic torques only move energy
                # between the field and the solids: the induction work leaves out the u×B0 flux at their walls (Iwall), which
                # is the field's share of it (utils4pp.INDU_WALLS)
                Wsol = 0.0
                if ut.rotdyn:
                    om, oi, ph = rot[:, i]
                    K = rd.coupling()
                    Etot += par.MoIZ_M*abs(om)**2 + par.MoIZ_IC*abs(oi)**2 + K*abs(ph)**2
                    Wsol = 2*np.real(np.conj(om)*vtorq[i] + np.conj(oi)*vtorq_ic[i]) - 2*K*rd.relaxation()*abs(ph)**2
                # NEW (2026-11-06): the work of the pressure where a wall moves radially (forced problems, bco or bci = 2), as
                # in residσ (upp.walls_move: only there, so that the other runs keep their numbers)
                Wpe = Wp if upp.walls_move() else 0.0
                # FIX (2026-11-08): relative to the viscous dissipation Dvisc too where the walls drive the flow (walls_drive), as
                # residσ: where the field takes little of the walls' power every other term is small (b = 0 for the Malkus field)
                # NEW (2026-11-08): the power of the body force of forcings 1 and 8 (Wf)
                residE[i] = ( abs( 2*sigma*Etot - Dkin[i] - Wthm[i] - Wcmp[i] - Wex - Wsol - Wpe - Wf[i] + Ohm )
                              / max( abs(2*sigma*Etot), abs(Dkin[i]), abs(Wthm[i]), abs(Wex), Ohm, abs(Wdr[i]), abs(Wsol),
                                     abs(Wpe), abs(Wf[i]), abs(Dvisc[i]) if walls_drive else 0.0, abs(Wcmp[i]), abs(Wbg[i]) ) )

        if do_drag:   # NEW (2026-11-06): the power the drag dissipates (per unit kappa_cmb without one) and its damping rate
            Dtopo[i] = topo.dissipation(u_sol2, rot[0, i] if ut.rotdyn else 0)
            E = KE[i]
            if par.magnetic:
                E += par.Hendrik*( ME[i] + MEc[i] + MEic[i] + MEext[i] + E_w[i] )
            if ut.rotdyn:
                E += par.MoIZ_M*abs(rot[0, i])**2 + par.MoIZ_IC*abs(rot[1, i])**2 + rd.coupling()*abs(rot[2, i])**2
            dsig[i] = -Dtopo[i]/(2*E) if E > 0 else np.nan

        if par.thermal:   # FIX (2026-11-08): 0 when every term is 0 (an entropy of 0, as resid𝐛; it was 0/0)
            scale3 = max( abs(2*sigma*TE[i]), abs(Dthm[i]), abs(Wadv_thm[i]), abs(Wdr_thm[i]), abs(Wbg_thm[i]) )
            resid3[i] = abs( 2*sigma*TE[i] - Dthm[i] - Wadv_thm[i] - Wdr_thm[i] - Wbg_thm[i] )/scale3 if scale3 > 0 else 0.0

        if par.compositional:   # NEW (2026-11-08): resid𝑐, as resid𝑠
            scale4 = max( abs(2*sigma*CE[i]), abs(Dcmp[i]), abs(Wadv_cmp[i]), abs(Wdr_cmp[i]), abs(Wbg_cmp[i]) )
            resid4[i] = abs( 2*sigma*CE[i] - Dcmp[i] - Wadv_cmp[i] - Wdr_cmp[i] - Wbg_cmp[i] )/scale4 if scale4 > 0 else 0.0

        # ------------------------------------------------------------------------------------------------------------------
        print(table_row(table_cols, table_w, i))
        # ------------------------------------------------------------------------------------------------------------------

        params.append( [ v for _, v, _ in param_table(timing + timer() - tic) ] )

    # ------------------------------------------------------------------------------------------------------------------------
    print(table_line + '\n')
    nirr = int(np.sum(irreg0 > IRREG0_NOTE))
    if nirr:   # NEW (2026-11-04)
        print('%d of the %d solutions have irreg₀ > %.0e: not regular at r = 0 (spurious or unresolved modes, or a background that is'
              ' not regular there; docs/numerical-method.md, Regularity at the centre)\n' % (nirr, success, IRREG0_NOTE))


    # ---------------------------------------------------------- write post-processed data and parameters to disk

    with open('params.dat','ab') as dpar:   # each row ends with row_note(), after '#'
        np.savetxt(dpar, np.array(params),
        fmt=[ f for _, _, f in param_table(0) ], footer='', comments='', newline=row_note() + '\n')

    # flow.dat columns, in order: (name, values, format). To add a column, append a tuple at the end.
    flow_cols = [
        ( 'KE',         KE,         '%.9e' ),
        ( 'KP',         KP,         '%.9e' ),
        ( 'KT',         KT,         '%.9e' ),
        ( 'Dkin',       Dkin,       '%.9e' ),
        ( 'ldom',       ldom,       '%d' ),
        ( 'lwidth',     lwidth,     '%d' ),
        ( 'lconv',      lconv,      '%.3e' ),
        ( 'Ensvel',     Ensvel,     '%.9e' ),
        ( 'Enscor',     Enscor,     '%.9e' ),
        ( 'Ensvif',     Ensvif,     '%.9e' ),
        ( 'Ensbuo',     Ensbuo,     '%.9e' ),
        ( 'cuvismax_r', cuvismax_r, '%.3e' ),
        ( 'cuvismax_l', cuvismax_l, '%d' ),
        ( 'cuvismax',   cuvismax,   '%.3e' ),
        ( 'resens',     resens,     '%.9e' ),  # enstrophy balance
        ( 'resid2',     resid2,     '%.9e' ),  # frequency balance, residω
        ( 'resid1',     resid1,     '%.9e' ),  # power balance, residσ
        ( 'Wdr',        Wdr,        '%.9e' ),  # differential-rotation power
        ( 'Wdr_im',     Wdr_im,     '%.9e' ),  # and its imaginary part
        ( 'Ensdr',      Ensdr,      '%.9e' ),  # differential-rotation enstrophy term (in resens)
        ( 'Wp_icb',     Wp_icb,     '%.9e' ),  # NEW (2026-11-04): power of the pressure at the inner wall,
        ( 'Wp_cmb',     Wp_cmb,     '%.9e' ),  #   at the outer wall,
        ( 'Wp_im',      Wp_im,      '%.9e' ),  #   and the imaginary part of their sum (in residω)
        ( 'Wv_icb',     Wv_icb,     '%.9e' ),  # power of the viscous stress at the inner wall,
        ( 'Wv_cmb',     Wv_cmb,     '%.9e' ),  #   and at the outer wall
        ( 'Dvisc',      Dvisc,      '%.9e' ),  # viscous dissipation ∫ 𝛔:∇𝐮 dV ≥ 0
        ( 'residD',     residD,     '%.9e' ),  # relative residual of Dkin = Wv_icb + Wv_cmb - Dvisc
        ( 'irreg0',     irreg0,     '%.3e' ),  # NEW (2026-11-04): ricb = 0, irreg₀ (nan with an inner core)
        ( 'angmom_re',  np.real(angmom),   '%.9e' ),  # NEW (2026-11-06): the angular momentum of the fluid (rotdyn.py), axial
        ( 'angmom_im',  np.imag(angmom),   '%.9e' ),  #   for m = 0, symm = 1, equatorial for m = 1, symm = -1 (0 otherwise),
        ( 'vtorq_re',   np.real(vtorq),    '%.9e' ),  #   the viscous torque on the mantle,
        ( 'vtorq_im',   np.imag(vtorq),    '%.9e' ),
        ( 'vtorq_ic_re', np.real(vtorq_ic), '%.9e' ), #   on the inner core,
        ( 'vtorq_ic_im', np.imag(vtorq_ic), '%.9e' ),
        ( 'mtorq_re',   np.real(mtorq),    '%.9e' ),  #   the magnetic torque on the mantle,
        ( 'mtorq_im',   np.imag(mtorq),    '%.9e' ),
        ( 'mtorq_ic_re', np.real(mtorq_ic), '%.9e' ), #   on the inner core,
        ( 'mtorq_ic_im', np.imag(mtorq_ic), '%.9e' ),
        ( 'residL',     residL,     '%.9e' ),  #   and resid𝐋 (nan where it does not apply)
        ( 'Dtopo',      Dtopo,      '%.9e' ),  # NEW (2026-11-06): the power the drag at the outer wall dissipates,
        ( 'dsig',       dsig,       '%.9e' ),  #   and the damping rate it implies (per unit kappa_cmb without a drag; 0 with
                                               #   a no-slip outer wall)
        ( 'Wf',         Wf,         '%.9e' ),  # NEW (2026-11-08): forcings 1 and 8, the power of the body force (in P_in),
        ( 'Wf_im',      Wf_im,      '%.9e' ),  #   and its imaginary part (0 for the other forcings)
        ( 'Wcmp',       Wcmp,       '%.9e' ),  # NEW (2026-11-08): the power of the buoyancy of the composition (in residσ),
        ( 'Wcmp_im',    Wcmp_im,    '%.9e' ),  #   and its imaginary part (in residω; 0 without the composition)
        ( 'Wbg',        Wbg,        '%.9e' ),  # NEW (2026-11-08): the power of the force of a background flow (in residσ),
        ( 'Wbg_im',     Wbg_im,     '%.9e' ),  #   its imaginary part (in residω)
        ( 'Ensbg',      Ensbg,      '%.9e' ),  #   and its enstrophy term (in resens; 0 without bgflow)
    ]
    with open('flow.dat','ab') as dflo:
        np.savetxt(dflo, np.column_stack([ v for _, v, _ in flow_cols ]),
        fmt=[ f for _, _, f in flow_cols ])


    if par.thermal:
        # thermal.dat columns, in order: (name, values, format). To add a column, append a tuple at the end.
        thermal_cols = [
            ( 'TE',         TE,         '%.18e' ),
            ( 'Wadv_thm',   Wadv_thm,   '%.18e' ),
            ( 'Dthm',       Dthm,       '%.18e' ),  # times ThermaD
            ( 'resid3',     resid3,     '%.18e' ),  # entropy budget, resid𝑠
            ( 'Wthm',       Wthm,       '%.18e' ),  # buoyancy work
            ( 'resid1',     resid1,     '%.18e' ),  # KE budget, residσ
            ( 'Wdr_thm',    Wdr_thm,    '%.18e' ),  # differential-rotation terms of the heat equation
            ( 'Wbg_thm',    Wbg_thm,    '%.18e' ),  # NEW (2026-11-08): those of a background flow (0 without bgflow)
        ]
        with open('thermal.dat','ab') as dtmp:
            np.savetxt(dtmp, np.column_stack([ v for _, v, _ in thermal_cols ]),
            fmt=[ f for _, _, f in thermal_cols ])


    if par.compositional:
        # NEW (2026-11-08): composition.dat columns, in order (as thermal.dat): (name, values, format). To add a column, append
        # a tuple at the end.
        composition_cols = [
            ( 'CE',         CE,         '%.18e' ),
            ( 'Wadv_cmp',   Wadv_cmp,   '%.18e' ),
            ( 'Dcmp',       Dcmp,       '%.18e' ),  # times CompoD
            ( 'resid4',     resid4,     '%.18e' ),  # budget of the composition, resid𝑐
            ( 'Wcmp',       Wcmp,       '%.18e' ),  # the work of its buoyancy (in residσ)
            ( 'resid1',     resid1,     '%.18e' ),  # KE budget, residσ
            ( 'Wdr_cmp',    Wdr_cmp,    '%.18e' ),  # its advection by the differential rotation
            ( 'Wbg_cmp',    Wbg_cmp,    '%.18e' ),  # NEW (2026-11-08): and by a background flow (0 without bgflow)
        ]
        with open('composition.dat','ab') as dcmp:
            np.savetxt(dcmp, np.column_stack([ v for _, v, _ in composition_cols ]),
            fmt=[ f for _, _, f in composition_cols ])


    if par.magnetic:
        # NEW (2026-10-28): magnetic.dat columns, in order: (name, values, format). To add a column, append a tuple at the end.
        magnetic_cols = [
            ( 'ME',         ME,         '%.9e' ),  # ∫ ½ 𝐛⋅𝐛 dV, without Hendrik (Hendrik*ME is in the units of KE)
            ( 'Mdfs',       Mdfs,       '%.9e' ),  # MagnetD ∫ 𝐛⋅∇²𝐛 dV
            ( 'Indu',       Indu,       '%.9e' ),  # ∫ 𝐛⋅∇×(𝐮×𝐁₀) dV
            ( 'residb',     residb,     '%.9e' ),  # induction budget, resid𝐛
            ( 'Wlor',       Wlor,       '%.9e' ),  # power of the Lorentz force (in residσ)
            ( 'Wlor_im',    Wlor_im,    '%.9e' ),  # and its imaginary part (in residω)
            ( 'MEic',       MEic,       '%.9e' ),  # ½ ∫ |𝐛|² inside the inner core, without Hendrik
            ( 'MEext',      MEext,      '%.9e' ),  # ½ ∫ |𝐛|² outside the outer wall, without Hendrik
            ( 'Ohm_f',      Ohm_f,      '%.9e' ),  # MagnetD ∫ |∇×𝐛|² over the fluid
            ( 'Ohm_ic',     Ohm_ic,     '%.9e' ),  # MagnetD η_i ∫ |∇×𝐛|² over a conducting inner core
            ( 'residE',     residE,     '%.9e' ),  # full energy budget, resid𝐄 (nan where it does not apply)
        ] + ([
            ( 'MEc',        MEc,        '%.9e' ),  # NEW (2026-11-05): centrifugal = 1, the metric part of ME (ME + MEc: the
        ] if (par.centrifugal or use_walls) else []) + ([   # energy of the field in the deformed fluid; 0 with a wall)
            ( 'E_w',        E_w,        '%.9e' ),  # NEW (2026-11-06): thin walls (or the resolved mantle), the energy stored
            ( 'Ohm_w',      Ohm_w,      '%.9e' ),  #   in them, without Hendrik, their Ohmic loss (MagnetD), and the estimated
            ( 'eps_twa_o',  twaerr[:,1], '%.9e' ), #   error ε_TWA of the outer and of the inner thin wall (0 where there is
            ( 'eps_twa_i',  twaerr[:,3], '%.9e' ), #   none)
        ] if use_walls else [])
        with open('magnetic.dat','ab') as dmag:
            np.savetxt(dmag, np.column_stack([ v for _, v, _ in magnetic_cols ]),
            fmt=[ f for _, _, f in magnetic_cols ])


    if ut.rotdyn:   # NEW (2026-11-06, as conductive_IC's rotdyn.dat): the rotating solids, real and imaginary parts of each
        rot_cols = [ ( 'omega_M',  rot[0] ),             # the rotation rate of the mantle, relative to the frame
                     ( 'omega_IC', rot[1] ),             # of the inner core
                     ( 'phi',      rot[2] ),             # the angle of the inner core relative to the mantle
                     ( 'Kphi',     rd.coupling()*rot[2] ),   # the gravitational torque on the mantle (minus that on the inner core)
                     ( 'vtorq',    vtorq ),              # the viscous torque on the mantle
                     ( 'mtorq',    mtorq ),              # the magnetic torque on the mantle
                     ( 'vtorq_ic', vtorq_ic ),           # the viscous torque on the inner core
                     ( 'mtorq_ic', mtorq_ic ) ]          # the magnetic torque on the inner core
        with open('rotdyn.dat','ab') as drd:
            np.savetxt(drd, np.column_stack([ f(v) for _, v in rot_cols for f in (np.real, np.imag) ]), fmt='%.9e')


    if par.forcing == 0:
        with open('eigenvalues.dat','ab') as deig:
            np.savetxt(deig, eigval)

    # NEW (2026-11-08, from conductive_IC): eigenmode tracking (track_target, params_default.py): the eigenvalue nearest the
    # target (track_target = 1), or that of the solution with the smallest KP/KT (2: to start a track, e.g. of the spin-over
    # mode), is written to the file track_target, the target of the next run (utils.py reads it)
    if par.forcing == 0 and success > 0 and (par.track_target == 1 or (par.track_target == 2 and par.hydro)):
        lam = eigval[:success, 0] + 1j*eigval[:success, 1]
        if par.track_target == 1:
            j = int(np.argmin(np.abs(lam - (par.rtau + 1j*par.itau))))
        else:
            j = int(np.argmin([ KP[k]/KT[k] if KT[k] > 0 else np.inf for k in range(success) ]))
        np.savetxt('track_target', [[ lam[j].real, lam[j].imag ]])
        print('Solution %d written to track_target as the next target\n' % j)

    print('Total time postprocessing:', timer()-tic, 'seconds')

    # ------------------------------------------------------------------ done
    return 0



if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
