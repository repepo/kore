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
    (background.description, diff_rot.description) and the 'tabulated' file with the first 12 hex digits of its sha256'''
    notes = [bg.description()] if bg.description() else []
    if par.diff_rot and dr.description():
        notes.append(dr.description())
    if cf.description():   # NEW (2026-11-05)
        notes.append(cf.description())
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



def param_table(elapsed):
    '''
    Run parameters written as one row of params.dat, as (name, value, format) tuples.
    The order here is the column order in params.dat. To add a parameter, append a
    tuple at the end so existing columns keep their positions.
    '''
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

    fname_ru = 'real_flow.field'
    fname_iu = 'imag_flow.field'

    fname_rt = 'real_thermal.field'
    fname_it = 'imag_thermal.field'

    if os.path.isfile(fname_ru) and os.path.isfile(fname_iu):
        ru = np.loadtxt(fname_ru).reshape((2*ut.n,-1))
        iu = np.loadtxt(fname_iu).reshape((2*ut.n,-1))
    if os.path.isfile(fname_rt) and os.path.isfile(fname_it):
        rt = np.loadtxt(fname_rt).reshape((ut.n,-1))
        it = np.loadtxt(fname_it).reshape((ut.n,-1))
    if par.magnetic:  # NEW (2026-10-28)
        rb = np.loadtxt('real_magnetic.field').reshape((2*ut.n,-1))
        ib = np.loadtxt('imag_magnetic.field').reshape((2*ut.n,-1))
    bic = None
    if ut.cic:  # NEW (2026-10-28): the field of a conducting inner core
        bic = ( np.loadtxt('real_magnetic_ic.field') + 1j*np.loadtxt('imag_magnetic_ic.field') ).reshape((2*ut.nic,-1))

    # NEW (2026-11-06): without the flow (hydro = 0) the solutions are counted in the field's (or the entropy's) file, and the
    # flow's diagnostics run on a zero flow (upp.zero_flow): they give zeros, and residσ, residω, residens are not computed
    success = np.shape(ru if par.hydro else (rb if par.magnetic else rt))[1]

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
    ] if par.thermal else []) + ([   # NEW (2026-11-04): forced runs, the power the walls put in and the viscous dissipation
        ( 'P_in',          '{: .6e}',   lambda i: Wp_icb[i] + Wp_cmb[i] + Wv_icb[i] + Wv_cmb[i] ),
        ( 'Dν',            '{: .6e}',   lambda i: Dvisc[i] ),
    ] if par.forcing > 0 else []) + ([
        ( 'resid𝐛',        '{:8.2e}',   lambda i: residb[i] ),
        ( 'resid𝐄',        '{:8.2e}',   lambda i: residE[i] ),
    ] if par.magnetic else []) + ([   # NEW (2026-11-06): the estimated error of the thin walls, the larger of the two
        ( 'εTWA',          '{:8.2e}',   lambda i: max(twaerr[i,1], twaerr[i,3]) ),
    ] if use_twa else []) + ([
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


        # identify solutions (NEW 2026-11-06: without the flow, the field, or the entropy)
        if par.hydro:
            [ ldom[i], lwidth[i], lconv[i] ] = upp.identify( u_sol2 )
        else:
            [ ldom[i], lwidth[i], lconv[i] ] = upp.identify( b_sol2, ut.bsymm ) if par.magnetic else upp.identify( t_sol2 )

        if par.ricb == 0:  # NEW (2026-11-04): how far the flow (and the field) is from regular at r = 0
            groups = [ [ (u_sol2[0], lp), (u_sol2[1], lt) ] ] if par.hydro else []
            if par.magnetic:
                lbp, lbt = ut.ell(par.m, par.lmax, ut.bsymm)[:2]
                groups.append( [ (b_sol2[0], lbp), (b_sol2[1], lbt) ] )
            if par.thermal and not par.hydro:   # NEW (2026-11-06): the entropy, without the flow
                groups.append( [ (t_sol2, lp) ] )
            irreg0[i] = upp.centre_irregularity(groups)

        # diagnose solutions (serially since 2026-10-20, see upp.diagnose)
        [ udgn, tdgn ] = upp.diagnose( u_sol2, t_sol2, par.ricb, ut.rcmb, int(ncpus) )

        if par.centrifugal:  # NEW (2026-11-05): the metric parts of KE, of the Coriolis power and of TE, on the grid of diagnose
            [ KEc, Wcorc, TEc ] = upp.centrifugal_terms( par.ricb, ut.rcmb )
            # NEW (2026-11-06, phase 2): those of the viscous power, of the differential rotation and of the entropy diffusion
            [ Dkinc, Wdrc, Dthmc ] = upp.centrifugal_dissipation_terms( par.ricb, ut.rcmb )

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
                                                                                u_sol2 if par.hydro else None )
            Ohm_f[i]  = par.MagnetD * Ohm_f0
            Ohm_ic[i] = par.MagnetD * Ohm_ic0
            Ohm_w[i]  = par.MagnetD * Ohm_w0   # NEW (2026-11-06): thin walls (0 without)
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
            [ Wp0, Wv0 ] = upp.wall_powers(u_sol2, t_sol2, sigma + 1j*w)
            [ Wp_icb[i], Wp_cmb[i] ] = np.real(Wp0)
            Wp_im[i] = np.sum(np.imag(Wp0))
            [ Wv_icb[i], Wv_cmb[i] ] = Wv0

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
          Wcor_im[i], Dkin_im[i], Wthm_im[i], Wdr_im[i], Ensdr[i] ] = np.sum( udgn, 0)
        Dkin[i] = Dkin0
        resens[i] = abs(Ensvel[i]*sigma+Enscor[i]-Ensbuo[i]-Ensvif[i]-Ensdr[i]) / max((abs(Ensvel[i]*sigma),abs(Enscor[i]),abs(Ensbuo[i]),abs(Ensvif[i]),abs(Ensdr[i]))) \
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

            [ TE[i], Dthm0, Wadv_thm[i], Wdr_thm[i] ] = np.sum( tdgn, 0)
            Dthm[i] = Dthm0 * par.ThermaD
            if par.centrifugal:  # NEW (2026-11-05): ∫ p0 J |s|² dV, the entropy "energy" of the deformed body
                TE[i] += TEc
                Dthm[i] += par.ThermaD * Dthmc   # NEW (2026-11-06, phase 2): its metric diffusion


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
        # ---------------------------------------------------------------------------------------------------------


        # NEW (2026-10-28): with magnetic = 1, Wlor (Wlor_im) is the power of the Lorentz force in residσ (residω), and
        # residb (resid𝐛) is the relative residual of 2*sigma*ME - Indu - Mdfs = 0 (the induction equation dotted with 𝐛*, in
        # the fluid; as conductive_IC's resid𝐛)
        Wp = Wp_icb[i] + Wp_cmb[i]
        if par.hydro:   # NEW (2026-11-06): nan without the flow
            resid1[i] = ( abs( 2*sigma*KE[i] - Dkin[i] - Wthm[i] - Wdr[i] - Wlor[i] - Wp )
                          / max( abs(2*sigma*KE[i]), abs(Dkin[i]), abs(Wthm[i]), abs(Wdr[i]), abs(Wlor[i]), abs(Wp) ) )
            resid2[i] = ( abs( 2*w*KE[i] + Wcor_im[i] - Dkin_im[i] - Wthm_im[i] - Wdr_im[i] - Wlor_im[i] - Wp_im[i] )
                          / max( abs(2*w*KE[i]), abs(Wcor_im[i]), abs(Dkin_im[i]), abs(Wthm_im[i]), abs(Wdr_im[i]), abs(Wlor_im[i]),
                                 abs(Wp_im[i]) ) )
        else:
            resid1[i] = resid2[i] = np.nan
        if par.ViscosD>0 and not par.centrifugal and par.hydro:   # NEW (2026-11-06, phase 2): not with centrifugal = 1 (Dvisc, Wv spherical)
            Wv = Wv_icb[i] + Wv_cmb[i]
            residD[i] = abs( Dkin[i] - Wv + Dvisc[i] ) / max( abs(Dkin[i]), abs(Wv), abs(Dvisc[i]) )
        if par.magnetic:
            residb[i] = abs( 2*sigma*ME[i] - Indu[i] - Mdfs[i] ) / max( abs(2*sigma*ME[i]), abs(Indu[i]), abs(Mdfs[i]) )
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
            # close the budget exactly for c = 0 and to O(c c1) otherwise; not with mu != 1 (the energies would need it).
            noslip = par.bco == 1 and (par.bci == 1 or par.ricb == 0)
            imperm = par.ViscosD == 0 or (par.bco in (0, 1) and (par.ricb == 0 or par.bci in (0, 1)))
            if (noslip or not par.hydro or (imperm and not par.centrifugal)) and not (use_twa and par.mu != 1):
                H = par.Hendrik if par.hydro else 1.0
                # NEW (2026-11-05): with centrifugal = 1 the energy of the deformed body, ∫ 𝐛̃*⋅M 𝐛̃ dV in the fluid (MEc), whose
                # rate is the work 2 Re ∫ 𝐄⋅(∇×M 𝐛̃)* dV (InduM) minus the Ohmic loss (Ohmc); closes at second order in f
                Etot = KE[i] + H*( ME[i] + MEc[i] + MEic[i] + MEext[i] )
                Wex  = Wlor[i] + H*(Indu[i] - Iwall[i] + InduM[i]) + Wdr[i]
                Ohm  = H*( Ohm_f[i] + Ohmc[i] + Ohm_ic[i] )
                if use_twa:   # NEW (2026-11-06)
                    Etot += H*E_w[i]
                    Ohm  += H*Ohm_w[i]
                residE[i] = ( abs( 2*sigma*Etot - Dkin[i] - Wthm[i] - Wex + Ohm )
                              / max( abs(2*sigma*Etot), abs(Dkin[i]), abs(Wthm[i]), abs(Wex), Ohm, abs(Wdr[i]) ) )

        if par.thermal:
            resid3[i] = ( abs( 2*sigma*TE[i] - Dthm[i] - Wadv_thm[i] - Wdr_thm[i] )
                          / max( abs(2*sigma*TE[i]), abs(Dthm[i]), abs(Wadv_thm[i]), abs(Wdr_thm[i]) ) )

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
        ]
        with open('thermal.dat','ab') as dtmp:
            np.savetxt(dtmp, np.column_stack([ v for _, v, _ in thermal_cols ]),
            fmt=[ f for _, _, f in thermal_cols ])


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
        ] if (par.centrifugal or use_twa) else []) + ([   # energy of the field in the deformed fluid; 0 with a thin wall)
            ( 'E_w',        E_w,        '%.9e' ),  # NEW (2026-11-06): thin walls, the energy stored in them, without Hendrik,
            ( 'Ohm_w',      Ohm_w,      '%.9e' ),  #   their Ohmic loss (MagnetD), and the estimated error ε_TWA of the
            ( 'eps_twa_o',  twaerr[:,1], '%.9e' ), #   outer and of the inner wall (0 where there is none)
            ( 'eps_twa_i',  twaerr[:,3], '%.9e' ),
        ] if use_twa else [])
        with open('magnetic.dat','ab') as dmag:
            np.savetxt(dmag, np.column_stack([ v for _, v, _ in magnetic_cols ]),
            fmt=[ f for _, _, f in magnetic_cols ])


    if par.forcing == 0:
        with open('eigenvalues.dat','ab') as deig:
            np.savetxt(deig, eigval)

    print('Total time postprocessing:', timer()-tic, 'seconds')

    # ------------------------------------------------------------------ done
    return 0



if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
