#!/usr/bin/env python3
'''
kore postprocessing script

Usage:
> ./bin/spin_doctor.py ncpus
'''

from timeit import default_timer as timer
t_start = timer()  # wall clock from here, before scipy/utils are imported
import sys
sys.path.insert(1,'bin/')
import scipy.io as sio
import scipy.sparse as ss
import os.path
import multiprocessing as mp
import numpy as np
from parameters import par
import utils as ut
import utils4pp as upp
import bgflow



def param_table(elapsed):
    '''
    Run parameters written as one row of params.dat, as (name, value, format) tuples.
    The order here is the column order in params.dat. To add a parameter, append a
    tuple at the end so existing columns keep their positions.
    '''
    return [
        ( 'hydro',                    par.hydro,                   '%d' ),
        ( 'magnetic',                 par.magnetic,                '%d' ),
        ( 'thermal',                  par.thermal,                 '%d' ),
        ( 'compositional',            par.compositional,           '%d' ),
        ( 'Ek',                       par.Ek,                      '%.9e' ),
        ( 'm',                        par.m,                       '%d' ),
        ( 'symm',                     par.symm,                    '%d' ),
        ( 'ricb',                     par.ricb,                    '%.9e' ),
        ( 'bci',                      par.bci,                     '%d' ),
        ( 'bco',                      par.bco,                     '%d' ),
        ( 'forcing',                  par.forcing,                 '%d' ),
        ( 'forcing_frequency',        par.forcing_frequency,       '%.9e' ),
        ( 'forcing_amplitude_cmb',    par.forcing_amplitude_cmb,   '%.9e' ),
        ( 'forcing_amplitude_icb',    par.forcing_amplitude_icb,   '%.9e' ),
        ( 'projection',               par.projection,              '%d' ),
        ( 'B0type',                   ut.B0type,                   '%d' ),
        ( 'beta_actual',              ut.beta_actual,              '%.9e' ),
        ( 'B0_l',                     ut.B0_l,                     '%d' ),
        ( 'innercore_mag_bc',         ut.innercore_mag_bc,         '%d' ),
        ( 'c_icb',                    par.c_icb,                   '%.9e' ),
        ( 'c1_icb',                   par.c1_icb,                  '%.9e' ),
        ( 'mantle_mag_bc',            ut.mantle_mag_bc,            '%d' ),
        ( 'c_cmb',                    par.c_cmb,                   '%.9e' ),
        ( 'c1_cmb',                   par.c1_cmb,                  '%.9e' ),
        ( 'mu',                       par.mu,                      '%.9e' ),
        ( 'Em',                       par.Em,                      '%.9e' ),
        ( 'Le2',                      par.Le2,                     '%.9e' ),
        ( 'B0_norm',                  ut.B0_norm(),                '%.9e' ),
        ( 'Etherm',                   par.Etherm,                  '%.9e' ),
        ( 'heating',                  ut.heating,                  '%d' ),
        ( 'BV2',                      par.BV2,                     '%.9e' ),
        ( 'rc',                       par.rc,                      '%.9e' ),
        ( 'h',                        par.h,                       '%.9e' ),
        ( 'rsy',                      par.rsy,                     '%d' ),
        ( 'bci_thermal',              par.bci_thermal,             '%d' ),
        ( 'bco_thermal',              par.bco_thermal,             '%d' ),
        ( 'Ecomp',                    par.Ecomp,                   '%.9e' ),
        ( 'compositional_background', ut.compositional_background, '%d' ),
        ( 'BV2_comp',                 par.BV2_comp,                '%.9e' ),
        ( 'rcc',                      par.rcc,                     '%.9e' ),
        ( 'hc',                       par.hc,                      '%.9e' ),
        ( 'rsyc',                     par.rsyc,                    '%d' ),
        ( 'bci_compositional',        par.bci_compositional,       '%d' ),
        ( 'bco_compositional',        par.bco_compositional,       '%d' ),
        ( 'OmgTau',                   par.OmgTau,                  '%.9e' ),
        ( 'ncpus',                    par.ncpus,                   '%d' ),
        ( 'N',                        par.N,                       '%d' ),
        ( 'lmax',                     par.lmax,                    '%d' ),
        ( 'runtime',                  elapsed,                     '%.2f' ),
        ( 'mu_i2o',                   par.mu_i2o,                  '%.9e' ),
        ( 'sigma_i2o',                par.sigma_i2o,               '%.9e' ),
        ( 'aux1',                     par.aux1,                    '%.9e' ),
        ( 'aux2',                     par.aux2,                    '%.9e' ),
        ( 'rotdyn',                   par.rotdyn,                  '%d' ),
        ( 'MoIZ_M',                   par.MoIZ_M,                  '%.9e' ),
        ( 'MoIZ_IC',                  par.MoIZ_IC,                 '%.9e' ),
        ( 'OmgtauIC',                 par.OmgtauIC,                '%.9e' ),
        ( 'gTorque',                  par.gTorque,                 '%.9e' ),
        ( 'bgflow',                   par.bgflow,                  '%d' ),    # NEW (2026-10-23)
        ( 'bgflow_Ro',                par.bgflow_Ro,               '%.9e' ),
        ( 'B0_tor',                   ut.B0_tor_type,              '%d' ),    # NEW (2026-10-30): index in ut.B0_tor_list, -1 without
        ( 'B0_tor_l',                 par.B0_tor_l,                '%d' ),
        ( 'cnorm_tor',                ut.cnorm_tor,                '%.9e' ),  # as a number (0 without a toroidal field)
    ]



def ratio(a, b):
    '''
    a/b, or nan when b is zero (e.g. Tor/Pol when there is no flow)
    '''
    return a/b if b != 0 else np.nan



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

    fname_ev = 'eigenvalues0.dat'
    fname_tm = 'timing.dat'

    if os.path.isfile(fname_ev):
        eigval = np.loadtxt(fname_ev).reshape((-1,2))

    timing = 0.0  # solve_nopp time, read from timing.dat if present
    if os.path.isfile(fname_tm):
        timing = np.loadtxt(fname_tm)
        if np.size(timing)>1:
            timing = timing[-1]

    fname_ru = 'real_flow.field'
    fname_iu = 'imag_flow.field'

    fname_rb = 'real_magnetic.field'
    fname_ib = 'imag_magnetic.field'

    fname_rt = 'real_temperature.field'
    fname_it = 'imag_temperature.field'

    fname_rc = 'real_composition.field'
    fname_ic = 'imag_composition.field'

    fname_dy = 'rotdyn.field'

    # NEW (2026-10-16): the solution files are read in parallel, by up to ncpus processes
    # (np.loadtxt takes 2.5 s for each 250 MB file at N = 1008, nev = 10)
    pairs  = [ (fname_ru, fname_iu), (fname_rb, fname_ib), (fname_rt, fname_it), (fname_rc, fname_ic) ]
    toread = [ f for pair in pairs if all(os.path.isfile(f) for f in pair) for f in pair ]
    with mp.Pool(processes=max(1, min(int(ncpus), len(toread)))) as pool:
        fields = dict(zip(toread, pool.map(np.loadtxt, toread)))

    if fname_ru in fields:
        ru = fields[fname_ru].reshape((2*ut.n,-1))
        iu = fields[fname_iu].reshape((2*ut.n,-1))
    if fname_rb in fields:
        rb = fields[fname_rb].reshape((2*ut.n,-1))
        ib = fields[fname_ib].reshape((2*ut.n,-1))
    if fname_rt in fields:
        rt = fields[fname_rt].reshape((ut.n,-1))
        it = fields[fname_it].reshape((ut.n,-1))
    if fname_rc in fields:
        rc = fields[fname_rc].reshape((ut.n,-1))
        ic = fields[fname_ic].reshape((ut.n,-1))
    if os.path.isfile(fname_dy):
        sol_rotdyn = np.loadtxt(fname_dy, dtype=complex).reshape((3,-1))
    # NEW (2026-10-18): the field inside a conducting inner core, for the full energy budget (resid E)
    bic = None
    if par.magnetic and ut.icflag and os.path.isfile('real_magnetic_ic.field'):
        bic = ( np.loadtxt('real_magnetic_ic.field') + 1j*np.loadtxt('imag_magnetic_ic.field') ).reshape((2*ut.nic,-1))

    if par.hydro == 1:
        success = np.shape(ru)[1]
    elif par.magnetic == 1:
        success = np.shape(rb)[1]

    # ------------------------------------------------------------------------------------------------------------------------
    KE          = np.zeros(success)
    KP          = np.zeros(success)
    KT          = np.zeros(success)
    Dkin        = np.zeros(success)
    Dint        = np.zeros(success)
    Wlor        = np.zeros(success)
    Wthm        = np.zeros(success)
    Wcmp        = np.zeros(success)
    Wbg         = np.zeros(success)  # NEW (2026-10-23): power of the background flow (bgflow = 1)
    vtorq       = np.zeros(success,dtype=complex)  # viscous torque on the mantle
    vtorq_ic    = np.zeros(success,dtype=complex)  # viscous torque on the IC
    ME          = np.zeros(success)
    Mdfs        = np.zeros(success)
    Indu        = np.zeros(success)
    mtorq       = np.zeros(success,dtype=complex)  # electromagnetic torque on the mantle
    mtorq_ic    = np.zeros(success,dtype=complex)  # electromagnetic torque on the IC   
    wtorq       = np.zeros(success,dtype=complex)  # NEW (2026-10-30): Lorentz torques on the fluid at its walls that mtorq, mtorq_ic leave out
    TE          = np.zeros(success)
    Wadv_thm    = np.zeros(success)
    Dthm        = np.zeros(success)
    CE          = np.zeros(success)
    Wadv_cmp    = np.zeros(success)
    Dcmp        = np.zeros(success)
    resid0      = np.zeros(success)
    resid1      = np.zeros(success)
    resid2      = np.zeros(success)
    resid3      = np.zeros(success)
    residL      = np.full(success, np.nan)    # angular momentum balance of the fluid (m=0, symm=1 axial; m=1, symm=-1 equatorial)
    residE      = np.full(success, np.nan)    # NEW (2026-10-18): full energy budget (fluid, inner core, walls, outside)
    budget      = np.full((success, 7), np.nan)  # NEW (2026-10-18): ME_ic, E_w, ME_ext, Ohm_f, Ohm_ic, Ohm_w, W_ex
    y           = np.zeros(success)                # for eigenmode tracking
    brms        = np.zeros(success)
    elldom      = np.zeros(success)
    geo         = np.full(success, np.nan)    # NEW (2026-10-11): geostrophic fraction of u_phi (m=0, symm=1 only)
    omg_mantle  = np.zeros(success,dtype=complex)
    omg_incore  = np.zeros(success,dtype=complex)
    misalignmt  = np.zeros(success,dtype=complex)
    gravtorq    = np.zeros(success,dtype=complex)
    angmom     = np.zeros(success,dtype=complex)
    params      = []                               # one row of param_table() per solution
    # ------------------------------------------------------------------------------------------------------------------------

    sigmas      = np.zeros(success)                # damping of each solution
    freqs       = np.zeros(success)                # frequency of each solution

    # Summary table printed while processing: (header, format, value of solution i).
    # To print a new quantity, add a line here.
    table_cols = [
        ( '★',           '{:d}',      lambda i: i ),
        ( 'Damping σ',   '{: .8f}',   lambda i: sigmas[i] ),
        ( 'Frequency ω', '{: .8f}',   lambda i: freqs[i] ),
        ( 'resid0',      '{:8.2e}',   lambda i: resid0[i] ),
        ( 'resid𝐮',      '{:8.2e}',   lambda i: resid1[i] ),
        ( 'resid𝐛',      '{:8.2e}',   lambda i: resid2[i] ),
        ( 'residθ',      '{:8.2e}',   lambda i: resid3[i] ),
        ( 'resid𝐋',      '{:8.2e}',   lambda i: residL[i] ),
        ( 'resid𝐄',      '{:8.2e}',   lambda i: residE[i] ),  # NEW (2026-10-18)
        ( 'Tor/Pol',     '{:8.2e}',   lambda i: ratio(KT[i], KP[i]) ),
        ( 'Mag/Kin',     '{:8.2e}',   lambda i: ratio(ME[i], KE[i]) ),
        ( 'Geo',         '{:.4f}',    lambda i: geo[i] ),  # NEW (2026-10-11)
        ( '|𝚪|mag',      '{:8.2e}',   lambda i: np.abs(mtorq[i]) ),
        ( '|𝚪|magIC',    '{:8.2e}',   lambda i: np.abs(mtorq_ic[i]) ),
    ]
    table_w = table_widths(table_cols)
    table_head, table_line = table_header(table_cols, table_w)

    print('\n' + table_head)
    print(table_line)


    if par.track_target == 1:  # eigenvalue tracking enabled
        #read target data
        x = np.loadtxt('track_target')


    # Loop invariants: these don't depend on the solution, so build them once
    if par.hydro:
        [ lp, lt, ll] = ut.ell(par.m, par.lmax, par.symm)
        lpi = np.searchsorted(ll,lp);  # Poloidal indices
        lti = np.searchsorted(ll,lt);  # Toroidal indices

    # FIX (2026-10-30): parentheses around the (m, symm) cases, as for do_mtorq, no inner-core torque in a full sphere, and
    # none with Ek = 0 (bci, bco are then ignored). Without them every m=1, symm=-1 run took a viscous torque also at
    # stress-free walls, and a full sphere one at r = 0: with no torque at all L is zero to round-off, and resid L was O(1)
    # (now 0, as for m=0 with stress-free walls)
    do_vtorq = par.hydro and (par.Ek != 0) and (par.bco == 1) and (((par.m==0) and (par.symm==1)) or ((par.m==1) and (par.symm==-1)))
    do_vtorq_ic = par.hydro and (par.Ek != 0) and (par.bci == 1) and (par.ricb > 0) and (((par.m==0) and (par.symm==1)) or ((par.m==1) and (par.symm==-1)))

    if do_vtorq:
        gvisc     = ut.gamma_visc(0,0,0)[0,:]    # viscous torque on the mantle
    if do_vtorq_ic:
        gvisc_icb = ut.gamma_visc_icb(par.ricb)[0,:]    # viscous torque on the IC

    # FIX (2026-10-29): parentheses around the (m, symm) cases; without them every m=1, symm=-1 run had do_mtorq True
    # (hydro or insulating mantle included) and crashed in gamma_magnetic
    # NEW (2026-10-30): the magnetic torques on the mantle and the inner core are those of a poloidal B0 (B0_r*b_phi). A toroidal
    # B0 with k != 0 at a wall (insulating or not) adds the stress b_r*B0_phi on the fluid there: it enters resid L (wtorq), but
    # not the torques on the mantle and the inner core, which also get the force on its current sheet (rotdyn = 1 stops on it)
    pol_B0      = par.magnetic and (par.B0 is not None)
    do_mtorq    = pol_B0 and (par.mantle == 'TWA') and (((par.m==0) and (par.symm==1)) or ((par.m==1) and (par.symm==-1)))
    do_mtorq_ic = pol_B0 and (par.innercore in ['conducting, Chebys', 'TWA']) and ((par.m==0) and (par.symm==1))
    # FIX (2026-10-30): for m=1, symm=-1 a poloidal B0 also exerts torques that gamma_magnetic (TWA mantle only) and
    # gamma_magnetic_ic (m=0 only) leave out: through an insulating mantle when B0 is imposed from outside ('axial': the
    # torque m x B0 on the induced dipole m), and through the ICB when the inner core conducts, carries B0's currents
    # (B0_ic other than 'potential', as in ut.B0_beta_ic) or holds its point source ('dipole': the torque m0 x b(0)).
    # resid L takes them from the Maxwell stress on the fluid side of that wall (wtorq). For m=0 they vanish
    # (b_phi = 0 at an insulating wall) or gamma_magnetic_ic has them.
    eq_B0       = pol_B0 and (par.forcing == 0) and (par.m == 1) and (par.symm == -1)
    pol_walls   = [ eq_B0 and (par.ricb > 0) and ( par.innercore != 'insulator' or par.B0 == 'dipole' or
                                               (ut.B0_ic != 'potential' and par.B0 in ut.B0_ic_list) ),
                    eq_B0 and (par.mantle != 'TWA') and (par.B0 not in ut.B0_cmb_list) ]  # [ICB, CMB]
    tor_walls   = ut.k_at_walls() if (par.magnetic and par.B0_tor is not None) else [False, False]
    do_wtorq    = par.magnetic and par.hydro and (any(pol_walls) or any(tor_walls))
    if do_mtorq:
        gmag    = ut.gamma_magnetic()[0,:]     # row vector, like gvisc (avoids a numpy deprecation when storing the torque)
    if do_mtorq_ic:
        gmag_ic = ut.gamma_magnetic_ic()[0,:]

    # The quadrature grid, once for all solutions. NEW (2026-10-16): no pool, diagnose is serial now (see there)
    upp.setup_grid(par.ricb, ut.rcmb)


    # Begin processing all solutions
    for i in range(success):

        if par.forcing == 0:
            w = eigval[i,1]
            sigma = eigval[i,0]
        else:
            w = ut.wf
            sigma = 0
        sigmas[i], freqs[i] = sigma, w

        [ u_sol2, b_sol2, t_sol2, c_sol2 ] = [0,0,0,0]

        if par.hydro:
            rflow = np.copy(ru[:,i])
            iflow = np.copy(iu[:,i])
            # Expand solution
            u_sol2 = upp.expand_reshape_sol( rflow + 1j*iflow, par.symm)
            u_sol  = upp.expand_sol( rflow + 1j*iflow, par.symm)  # this one for the torque

        if par.magnetic:
            rmag = np.copy(rb[:,i])
            imag = np.copy(ib[:,i])
            # Expand solution
            b_sol2 = upp.expand_reshape_sol( rmag + 1j*imag, ut.bsymm)
            b_sol  = upp.expand_sol( rmag + 1j*imag, ut.bsymm)  # this one for the torque
                        
        if par.thermal:
            rthm = np.copy(rt[:,i])
            ithm = np.copy(it[:,i])
            # Expand solution
            t_sol2  = upp.expand_reshape_sol( rthm + 1j*ithm, par.symm)
            
        if par.compositional:
            rcmp = np.copy(rc[:,i])
            icmp = np.copy(ic[:,i])
            # Expand solution
            c_sol2  = upp.expand_reshape_sol( rcmp + 1j*icmp, par.symm)		   			


        # diagnose solutions
        [ udgn, bdgn, tdgn, cdgn ] = upp.diagnose( u_sol2, b_sol2, t_sol2, c_sol2, par.ricb, ut.rcmb, int(ncpus), sigma+1j*w )


        if par.hydro:
            
            KP[i] = np.sum( udgn[lpi,0])  # Poloidal kinetic energy
            KT[i] = np.sum( udgn[lti,0])  # Toroidal kinetic energy
            
            [ KE[i], Dkin0, Dint0, Wlor0, Wthm0, Wcmp0, _ ] = np.sum( udgn, 0)
            Dkin[i] = par.OmgTau * par.Ek * Dkin0
            Dint[i] = par.OmgTau * par.Ek * Dint0
            Wlor[i] = par.OmgTau**2 * par.Le2 * Wlor0
            Wthm[i] = par.OmgTau**2 * par.BV2 * Wthm0
            Wcmp[i] = par.OmgTau**2 * par.BV2_comp * Wcmp0
            if par.bgflow:
                Wbg[i] = bgflow.power(u_sol2)
            
            # Viscous torques
            if do_vtorq:
                vtorq[i] = par.Ek * par.OmgTau * np.dot( gvisc, u_sol)  # constants checked 2026-10-02 against angrymom_z (angular momentum conservation)
            if do_vtorq_ic:
                vtorq_ic[i] = par.Ek * par.OmgTau * np.dot( gvisc_icb, u_sol)

            # Angular momentum in the z direction
            angmom[i] = upp.angrymom(u_sol2)

            # NEW (2026-10-11): fraction of the u_phi energy in its z-average (1 for a z-invariant u_phi)
            geo[i] = upp.geostrophic_fraction(u_sol2)

            #press0[i] = udgn[6][0], not sure why this was here? 


        if par.magnetic:

            [ ME[i], Mdfs0, Indu[i], brms[i] ] = np.sum( bdgn, 0)
            # [ME[i], Mdfs0, Indu[i]] = np.sum(bdgn, 0)
            Mdfs[i] = Mdfs0 * par.OmgTau * par.Em

            if do_mtorq:
                mtorq[i] = par.Le2 * (par.OmgTau**2) * np.dot( gmag, b_sol )

            if do_mtorq_ic:
                mtorq_ic[i] = par.Le2 * (par.OmgTau**2) * np.dot( gmag_ic, b_sol )

            if do_wtorq:
                wtorq[i] = par.Le2 * (par.OmgTau**2) * upp.torque_walls(b_sol2, pol_walls, tor_walls)  # on the fluid

        if par.rotdyn:

            omg_mantle[i] = sol_rotdyn[0,i]
            omg_incore[i] = sol_rotdyn[1,i]
            misalignmt[i] = sol_rotdyn[2,i]
            gravtorq[i] = par.gTorque * (par.OmgTau**2) * misalignmt[i]

        if par.thermal:

            [ TE[i], Dthm0, Wadv_thm[i] ] = np.sum( tdgn, 0) 
            Dthm[i] = Dthm0 * par.OmgTau * par.Etherm


        if par.compositional:
            
            [ CE[i], Dcmp0, Wadv_cmp[i] ] = np.sum( cdgn, 0)
            Dcmp[i] = Dcmp0 * par.OmgTau * par.Ecomp


        # --------------------------------------------------------- Computing residuals to check the power balance:
        # pss is the rate of working of stresses at the boundary
        # pvf is the rate of working of external volume force
        # Dint is the rate of change of internal energy
        # Dkin is the kinetic energy dissipation (viscous dissipation) via ∫𝐮⋅∇²𝐮 dV
        # Wthm is the rate of working of the buoyancy force (thermal)
        # Dohm is the Ohmic dissipation or Joule heating via ∫|∇×𝐛|² dV
        # Mdfs is the magnetic diffusion via ∫𝐛⋅∇²𝐛 dV
        # Dthm is the thermal "dissipation" via ∫ θ ∇²θ dV
        # Wthm_adv is the thermal advection "power" via ∫ (-𝐮⋅∇T) θ dV
        # KE is kinetic energy
        # ME is magnetic energy
        # TE is the thermal "energy" (1/2) ∫ θ² dV
        # resid0 is the relative residual of Dkin + Dint - pss = 0
        # Wbg is the rate of working of the background flow (bgflow = 1), -2 Re ∫ u*·(u·∇)U dV
        # resid1 is the relative residual of 2*sigma*KE - Dkin - Wlor -Wthm - pvf - Wbg = 0
        # resid2 is the relative residual of 2*sigma*ME - Indu - Mdfs = 0
        # resid3 is the relative residual of 2*sigma*TE - Dthm - Wadv_thm = 0
        # ---------------------------------------------------------------------------------------------------------

        [repow, pss, pvf] = [0, 0, 0]  # power from the forcing needs to be computed, not coded yet. 

        if par.forcing == 0:
            pss = 0
            pvf = 0
            if par.rotdyn and par.hydro and par.Ek != 0:
                # NEW (2026-10-18): power of the moving mantle and inner core on the fluid, minus the viscous torques times
                # their rotation rates (factor 2 as for the energies, int |u|^2), in the units of Dint0 and Dkin0
                pss = -2*np.real( vtorq[i]*np.conj(omg_mantle[i]) + vtorq_ic[i]*np.conj(omg_incore[i]) ) / (par.OmgTau*par.Ek)
        elif par.forcing == 1:
            pss = 0
            pvf = repow
        elif par.forcing == 7: # Libration as boundary flow forcing
            pvf = 0      # power of volume forces (Poincare)
            pss = repow  # power of stresses
        elif par.forcing == 8:  # Libration as a volume force
            pss = 0      # power of stresses
            pvf = repow  # power of volume forces (Poincare)
        elif par.forcing == 9: # Radial boundary flow forcing
            pvf = 0      # power of volume forces (Poincare)
            pss = repow  # power of stresses
        elif par.forcing == 12: # Steady differential rotation of the boundaries
            # power of the walls on the fluid, minus the viscous torques times the wall rotation rates, as for rotdyn above;
            # the complex amplitude of a wall's rotation rate is forcing_amplitude/2 (field + c.c., see assemble.py)
            pvf = 0
            pss = -2*np.real( vtorq[i]*par.forcing_amplitude_cmb/2 + vtorq_ic[i]*par.forcing_amplitude_icb/2 ) / (par.OmgTau*par.Ek)

        if par.Ek != 0 and par.hydro == 1:
            resid0[i] = abs( Dint0 + Dkin0 - pss ) / max( abs(Dint0), abs(Dkin0), abs(pss) )
        else:
            resid0[i] = np.nan

        if par.hydro:
            resid1[i] = abs( 2*sigma*KE[i] - Dkin[i] - Wlor[i] + Wthm[i] - Wbg[i] )/ \
                             max(abs(2*sigma*KE[i]), abs(Dkin[i]), abs(Wlor[i]), abs(Wthm[i]), abs(Wbg[i]))
            if par.forcing == 12:  # steady, no body force: the integral of u.lap(u) vanishes; measured against the dissipation
                resid1[i] = abs( Dkin[i] ) / abs( Dint[i] )
        
        if par.magnetic:
            resid2[i] = abs( 2*sigma*ME[i] - Indu[i] - Mdfs[i] ) / \
                             max( abs(2*sigma*ME[i]), abs(Indu[i]), abs(Mdfs[i]))
            
        if par.thermal:
            resid3[i] = abs( 2*sigma*TE[i] - Dthm[i] - Wadv_thm[i] ) / \
                             max( abs(2*sigma*TE[i]), abs(Dthm[i]), abs(Wadv_thm[i]))

        # Angular momentum balance of the fluid: dL/dt = lambda*L must equal the torque on the fluid,
        # which is minus the viscous and magnetic torques it exerts on the mantle and the inner core, plus the wall
        # stresses wtorq (gravitational torques act only between mantle and inner core). Only for eigenvalue problems with
        # m=0, symm=1 (axial) and m=1, symm=-1 (equatorial, with the Coriolis torque), where angrymom and the torques are
        # defined; body forcing would add a torque not included here.
        # Also for the steady forcing 12 (w = sigma = 0): the torques on the mantle and the inner core cancel.
        # FIX (2026-10-30): nan outside those cases (it was 0 for other m, symm, and for forced runs without a torque flag),
        # and 0 when no torque acts at the walls: the balance is then (lambda - i m OmgTau) L = 0, which is not tested
        L_case = par.hydro and ( (par.forcing in (0, 12) and par.m == 0 and par.symm == 1) or
                                 (par.forcing == 0 and par.m == 1 and par.symm == -1) )
        if not L_case:
            pass  # stays nan
        elif not (do_vtorq or do_mtorq or do_vtorq_ic or do_mtorq_ic or do_wtorq):
            residL[i] = 0

        elif par.hydro and par.forcing in (0, 12) and (par.m == 0 and par.symm == 1):
            dLdt    = (sigma + 1j*w) * angmom[i]
            torques = [ vtorq[i], vtorq_ic[i], mtorq[i], mtorq_ic[i], -wtorq[i] ]
            scale   = max( [abs(dLdt)] + [abs(t) for t in torques] )
            if scale > 0:
                residL[i] = abs( dLdt + sum(torques) ) / scale
        
        elif par.hydro and par.forcing == 0 and (par.m == 1 and par.symm == -1):
            dLdt    = (sigma + 1j*w) * angmom[i]
            ctorq   = -par.OmgTau * 1j * angmom[i]
            torques = [ vtorq[i], vtorq_ic[i], mtorq[i], mtorq_ic[i], -wtorq[i], ctorq ]
            scale   = max( [abs(dLdt)] + [abs(t) for t in torques] )
            if scale > 0:
                residL[i] = abs( dLdt + sum(torques) ) / scale

            '''
            reals = [np.real(dLdt)] + [np.real(t) for t in torques]
            imags = [np.imag(dLdt)] + [np.imag(t) for t in torques]
            print('{:1.2e}   {:1.2e}   {:1.2e}   {:1.2e}   {:1.2e}   {:1.2e}   {:1.2e}'.format(reals[0], reals[1], reals[2], reals[3], reals[4], reals[5], sum(reals)))
            print('{:1.2e}   {:1.2e}   {:1.2e}   {:1.2e}   {:1.2e}   {:1.2e}   {:1.2e}'.format(imags[0], imags[1], imags[2], imags[3], imags[4], imags[5], sum(imags)))
            '''

        # NEW (2026-10-18): full energy budget, 2 sigma (KE + ME + ME_ic + E_w + ME_ext) = Dkin + Wlor - Wthm + Indu
        # - Ohm_f - Ohm_ic - Ohm_w, with the magnetic terms in kinetic-energy units (times OmgTau**2*Le2). It adds
        # what resid b cannot see: the field and Ohmic loss of a conducting inner core and of thin walls, the
        # potential fields beyond insulating boundaries, and so the matching conditions at the ICB and CMB.
        # Here Ohmic losses are integrals of |curl b|^2, while resid b uses b.lap(b) in the fluid only. The Lorentz
        # work and the induction term cancel in the sum except for the work of the background currents (J0 x b).
        # FIX (2026-10-30): slipping walls (stress-free, or Ek = 0) carry the u x B0 part of the Poynting flux, which
        # Indu includes (upp.slip_power, subtracted below); before, resid E needed no-slip walls. A thin wall still
        # needs a no-slip boundary (rotdyn = 1 stops on slipping walls, utils.check_rotdyn). Not for forced problems,
        # 'conducting, Bessel', or mu != 1 with thin walls.
        # With rotdyn = 1 the rotation energies of the mantle and the inner core and the gravitational energy join the
        # budget; the torques only move energy between fluid and solids, so the viscous term is the strain-rate
        # dissipation (Dint, without the work of the moving walls), the work of the moving walls through u x B0 is the
        # magnetic torque times the rotation rate, and the inner core's viscous relaxation dissipates K*|phi|^2/tau_IC.
        noslip = (par.bco == 1) and (par.ricb == 0 or par.bci == 1) and par.Ek > 0
        slipok = not ( (par.ricb > 0 and (par.bci == 0 or par.Ek == 0) and par.innercore == 'TWA') or
                       ((par.bco == 0 or par.Ek == 0) and par.mantle == 'TWA') )  # FIX (2026-10-30)
        rotok  = (not par.rotdyn) or (par.vtrq == 1 and par.mtrq == 1)  # switched-off torques do not conserve energy
        if par.magnetic and par.forcing == 0 and rotok and par.innercore != 'conducting, Bessel' and \
           (noslip or slipok or not par.hydro) and not (par.mu != 1 and 'TWA' in (par.innercore, par.mantle)):
            s2 = par.OmgTau**2 * par.Le2
            eta = par.OmgTau * par.Em
            Ohm_f0, MEic0, Ohm_ic0, Ew0, Ohm_w0, MEext0 = upp.em_budget(b_sol2, None if bic is None else bic[:,i])
            Wex  = (Wlor[i] if par.hydro else 0) + s2*Indu[i]
            if par.hydro and not noslip:  # FIX (2026-10-30)
                Wex -= s2*upp.slip_power(u_sol2, b_sol2)
            Etot = (KE[i] if par.hydro else 0) + s2*( ME[i] + MEic0 + Ew0 + MEext0 )
            Ohm  = s2*eta*( Ohm_f0 + Ohm_ic0 + Ohm_w0 )
            Dk   = (Dkin[i] - Wthm[i]) if par.hydro else 0
            if par.rotdyn:
                # spin_doctor's energies are int |u|^2 (twice the kinetic energy of the complex amplitude), so the solids
                # enter as I*|omega|^2 and K*|phi|^2, and the powers as 2*Re(torque*conj(omega))
                Kg    = par.gTorque * par.OmgTau**2
                Etot += par.MoIZ_M*abs(omg_mantle[i])**2 + par.MoIZ_IC*abs(omg_incore[i])**2 + Kg*abs(misalignmt[i])**2
                Dk    = -Dint[i] - Wthm[i] - 2*Kg*(par.OmgTau/par.OmgtauIC)*abs(misalignmt[i])**2
                Wex  += 2*np.real( mtorq[i]*np.conj(omg_mantle[i]) + mtorq_ic[i]*np.conj(omg_incore[i]) )
            scale = max( abs(2*sigma*Etot), abs(Dk), Ohm, abs(Wex) )
            residE[i] = abs( 2*sigma*Etot - Dk - Wex + Ohm ) / scale
            budget[i] = [ s2*MEic0, s2*Ew0, s2*MEext0, s2*eta*Ohm_f0, s2*eta*Ohm_ic0, s2*eta*Ohm_w0, Wex ]

        # ------------------------------------------------------------------------------------------------------------------
        print(table_row(table_cols, table_w, i))
        # ------------------------------------------------------------------------------------------------------------------
        #print(' ')

        
        params.append( [ v for _, v, _ in param_table(0) ] )  # runtime filled in before writing params.dat

    # ------------------------------------------------------------------------------------------------------------------------
    print(table_line + '\n')


    '''
    # find closest eigenvalue to tracking target and write to target file
    if (par.track_target == 1)&(par.forcing == 0):
        j = y==min(y)
        if par.magnetic == 1:
            with open('track_target','wb') as tg:
                np.savetxt( tg, np.c_[ eigval[j,0], eigval[j,1], p2t[j], o2v[j] ] )
        else:
            with open('track_target','wb') as tg:
                np.savetxt( tg, np.c_[ eigval[j,0], eigval[j,1], p2t[j] ] )
        print('Closest to target is solution', np.where(j==1)[0][0])
        #if err2[j]>0.1:
        #    np.savetxt('big_error', np.c_[err1[j],err2[j]] )


    # use this when writing the first target, it finds the solution with smallest p2t (useful to track the spin-over mode)
    elif (par.track_target == 2)&(par.forcing == 0):
        # here we select the solution with smallest poloidal to toroidal energy ratio and write the track_target file:
        j = p2t==min(p2t)
        if par.magnetic == 1:
            with open('track_target','wb') as tg:
                np.savetxt( tg, np.c_[ eigval[j,0], eigval[j,1], p2t[j], o2v[j] ] )
        else:
            with open('track_target','wb') as tg:
                np.savetxt( tg, np.c_[ eigval[j,0], eigval[j,1], p2t[j] ] )

    '''

    # ---------------------------------------------------------- write post-processed data and parameters to disk

    # runtime = full solve_nopp time + full spin_doctor time (both incl. imports), same for every solution
    irt = [ name for name, _, _ in param_table(0) ].index('runtime')
    runtime = timing + timer() - t_start
    for row in params:
        row[irt] = runtime

    with open('params.dat','ab') as dpar:
        np.savetxt(dpar, np.array(params),
        fmt=[ f for _, _, f in param_table(0) ])

    if par.hydro:   
        with open('flow.dat','ab') as dflo:
            np.savetxt(dflo, np.c_[ KE,   KP,   KT,   Dkin,
                                    Dint, Wlor, Wthm, Wcmp,
                                    resid0, resid1,
                                    np.real(vtorq), np.imag(vtorq),
                                    np.real(vtorq_ic), np.imag(vtorq_ic),
                                    elldom, np.real(angmom), np.imag(angmom), residL, geo, # NEW (2026-10-11): geostrophic fraction, column 19
                                    Wbg ])  # NEW (2026-10-23): background flow power, column 20

    if par.magnetic:
        with open('magnetic.dat','ab') as dmag:
            np.savetxt(dmag, np.c_[ ME, Mdfs, Indu, resid2,
                                    np.real(mtorq), np.imag(mtorq),
                                    np.real(mtorq_ic), np.imag(mtorq_ic), brms])

    if par.magnetic:  # NEW (2026-10-18): terms of the full energy budget, in kinetic-energy units, and resid E
        with open('budget.dat','ab') as dbud:
            np.savetxt(dbud, np.c_[ budget, residE ])

    if par.rotdyn:
        with open('rotdyn.dat','ab') as drd:
            np.savetxt(drd, np.c_[ omg_mantle, omg_incore, misalignmt, gravtorq, vtorq, mtorq, vtorq_ic, mtorq_ic ])

    if par.thermal:
        with open('thermal.dat','ab') as dtmp:
            np.savetxt(dtmp, np.c_[ TE, Wadv_thm, Dthm, resid3 ])

    if par.compositional:
        with open('compositional.dat','ab') as dcmp:
            np.savetxt(dcmp, np.c_[ CE, Wadv_cmp, Dcmp ])

    if not par.forcing:
        with open('eigenvalues.dat','ab') as deig:
            np.savetxt(deig, eigval)

    print('Total time postprocessing:', timer()-t_start, 'seconds')

    # ------------------------------------------------------------------ done
    return 0



if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
