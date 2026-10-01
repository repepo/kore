#!/usr/bin/env python3
'''
kore postprocessing script

Usage:
> python3 ./bin/solution_doctor.py ncpus
'''

import sys
sys.path.insert(1,'bin/')
import scipy.io as sio
import scipy.sparse as ss
from timeit import default_timer as timer
import os.path
import multiprocessing as mp
import numpy as np
import parameters as par
import utils as ut
import utils4pp as upp



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
    tic = timer()

    fname_ev = 'eigenvalues0.dat'
    fname_tm = 'timing.dat'

    if os.path.isfile(fname_ev):
        eigval = np.loadtxt(fname_ev).reshape((-1,2))

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

    if os.path.isfile(fname_ru) and os.path.isfile(fname_iu):
        ru = np.loadtxt(fname_ru).reshape((2*ut.n,-1))
        iu = np.loadtxt(fname_iu).reshape((2*ut.n,-1))
    if os.path.isfile(fname_rb) and os.path.isfile(fname_ib):
        rb = np.loadtxt(fname_rb).reshape((2*ut.n,-1))
        ib = np.loadtxt(fname_ib).reshape((2*ut.n,-1))
    if os.path.isfile(fname_rt) and os.path.isfile(fname_it):
        rt = np.loadtxt(fname_rt).reshape((ut.n,-1))
        it = np.loadtxt(fname_it).reshape((ut.n,-1))
    if os.path.isfile(fname_rc) and os.path.isfile(fname_ic):
        rc = np.loadtxt(fname_rc).reshape((ut.n,-1))
        ic = np.loadtxt(fname_ic).reshape((ut.n,-1))
    if os.path.isfile(fname_dy):
        sol_rotdyn = np.loadtxt(fname_dy, dtype=complex).reshape((3,-1))

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
    vtorq       = np.zeros(success,dtype=complex)  # viscous torque on the mantle
    vtorq_ic    = np.zeros(success,dtype=complex)  # viscous torque on the IC
    ME          = np.zeros(success)
    Mdfs        = np.zeros(success)
    Indu        = np.zeros(success)
    mtorq       = np.zeros(success,dtype=complex)  # electromagnetic torque on the mantle
    mtorq_ic    = np.zeros(success,dtype=complex)  # electromagnetic torque on the IC   
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
    y           = np.zeros(success)                # for eigenmode tracking
    press0      = np.zeros(success)
    elldom      = np.zeros(success)
    omg_mantle  = np.zeros(success,dtype=complex)
    omg_incore  = np.zeros(success,dtype=complex)
    misalignmt  = np.zeros(success,dtype=complex)
    gravtorq    = np.zeros(success,dtype=complex)
    angmomz     = np.zeros(success,dtype=complex)
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
        ( 'Tor/Pol',     '{:8.2e}',   lambda i: ratio(KT[i], KP[i]) ),
        ( 'Mag/Kin',     '{:8.2e}',   lambda i: ratio(ME[i], KE[i]) ),
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
        gvisc     = ut.gamma_visc(0,0,0)[0,:]            # viscous torque on the mantle
        gvisc_icb = ut.gamma_visc_icb(par.ricb)[0,:]    # viscous torque on the IC

    do_mtorq    = par.magnetic and (par.mantle == 'TWA') and (par.m==0) and (par.symm==1)
    do_mtorq_ic = par.magnetic and (par.innercore in ['conducting, Chebys', 'TWA']) and ((par.m==0) and (par.symm==1))
    if do_mtorq:
        gmag    = ut.gamma_magnetic()
    if do_mtorq_ic:
        gmag_ic = ut.gamma_magnetic_ic()

    # One pool for all solutions. The quadrature grid must be set before the pool is created,
    # since the workers are forked and only see the module globals as they were at that time.
    upp.setup_grid(par.ricb, ut.rcmb)
    pool = mp.Pool(processes=int(ncpus))


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


        # diagnose solutions, in parallel
        [ udgn, bdgn, tdgn, cdgn ] = upp.diagnose( u_sol2, b_sol2, t_sol2, c_sol2, par.ricb, ut.rcmb, int(ncpus), sigma+1j*w, pool=pool )


        if par.hydro:
            
            KP[i] = np.sum( udgn[lpi,0])  # Poloidal kinetic energy
            KT[i] = np.sum( udgn[lti,0])  # Toroidal kinetic energy
            
            [ KE[i], Dkin0, Dint0, Wlor0, Wthm0, Wcmp0, press0[i] ] = np.sum( udgn, 0)
            Dkin[i] = par.OmgTau * par.Ek * Dkin0
            Dint[i] = par.OmgTau * par.Ek * Dint0
            Wlor[i] = par.OmgTau**2 * par.Le2 * Wlor0
            Wthm[i] = par.OmgTau**2 * par.BV2 * Wthm0
            Wcmp[i] = par.OmgTau**2 * par.BV2_comp * Wcmp0
            #print('Wlor=',Wlor[i])
            
            # Viscous torques
            vtorq[i] = par.Ek * par.OmgTau * np.dot( gvisc, u_sol)  # need to double check the constants here
            vtorq_ic[i] = par.Ek * par.OmgTau * np.dot( gvisc_icb, u_sol)

            # Angular momentum in the z direction
            angmomz[i] = upp.angrymom_z(u_sol2)

            #press0[i] = udgn[6][0], not sure why this was here? 


        if par.magnetic:

            [ ME0, Mdfs0, Indu[i] ] = np.sum( bdgn, 0)
            ME[i]   = ME0   * par.OmgTau**2 * par.Le2
            #Dohm = Dohm0 * par.OmgTau**3 * par.Le2 * par.Em
            Mdfs[i] = par.OmgTau * par.Em * Mdfs0

            if do_mtorq:
                mtorq[i] = par.Le2 * (par.OmgTau**2) * np.dot( gmag, b_sol )

            if do_mtorq_ic:
                mtorq_ic[i] = par.Le2 * (par.OmgTau**2) * np.dot( gmag_ic, b_sol )

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
        # resid1 is the relative residual of 2*sigma*KE - Dkin - Wlor -Wthm - pvf = 0
        # resid2 is the relative residual of 2*sigma*ME - Indu - Mdfs = 0
        # resid3 is the relative residual of 2*sigma*TE - Dthm - Wadv_thm = 0
        # ---------------------------------------------------------------------------------------------------------

        [repow, pss, pvf] = [0, 0, 0]  # power from the forcing needs to be computed, not coded yet. 

        if par.forcing == 0:
            pss = 0
            pvf = 0
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

        if par.Ek != 0 and par.hydro == 1:
            resid0[i] = abs( Dint0 + Dkin0 - pss ) / max( abs(Dint0), abs(Dkin0), abs(pss) )
        else:
            resid0[i] = np.nan

        if par.hydro:
            resid1[i] = abs( 2*sigma*KE[i] - Dkin[i] - Wlor[i] + Wthm[i] )/ \
                             max(abs(2*sigma*KE[i]), abs(Dkin[i]), abs(Wlor[i]), abs(Wthm[i]))
            #print('2σK = ',2*sigma*KE[i], '-Dkin = ', -Dkin[i], '-Wlor = ', -Wlor[i])
        
        if par.magnetic:
            resid2[i] = abs( 2*sigma*ME0 - Indu[i] - Mdfs[i] ) / \
                             max( abs(2*sigma*ME0), abs(Indu[i]), abs(Mdfs[i]))
            #print('2σM = ',2*sigma*ME0, '-Indu = ',-Indu[i], '-Mdfs = ', -Mdfs[i])
            
        if par.thermal:
            resid3[i] = abs( 2*sigma*TE[i] - Dthm[i] - Wadv_thm[i] ) / \
                             max( abs(2*sigma*TE[i]), abs(Dthm[i]), abs(Wadv_thm[i]))
        
    
        # ------------------------------------------------------------------------------------------------------------------
        print(table_row(table_cols, table_w, i))
        # ------------------------------------------------------------------------------------------------------------------
        #print(' ')

        toc = timer()
        
        params.append( [ v for _, v, _ in param_table(timing+toc-tic) ] )

    pool.close()
    pool.join()

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
                                    press0, elldom, np.real(angmomz), np.imag(angmomz) ])

    if par.magnetic:
        with open('magnetic.dat','ab') as dmag:
            np.savetxt(dmag, np.c_[ ME, Mdfs, Indu, resid2,
                                    np.real(mtorq), np.imag(mtorq),
                                    np.real(mtorq_ic), np.imag(mtorq_ic)])

    if par.rotdyn:
        with open('rotdyn.dat','ab') as drd:
            np.savetxt(drd, np.c_[ omg_mantle, omg_incore, misalignmt, gravtorq, vtorq, mtorq, vtorq_ic, mtorq_ic ])

    if par.thermal:
        with open('thermal.dat','ab') as dtmp:
            np.savetxt(dtmp, np.c_[ TE, Wadv_thm, Dthm, resid3 ])

    if par.compositional:
        with open('compositional.dat','ab') as dcmp:
            np.savetxt(dcmp, np.c_[ CE, Wadv_cmp, Dcmp ])

    if par.forcing == 0:
        with open('eigenvalues.dat','ab') as deig:
            np.savetxt(deig, eigval)

    # ------------------------------------------------------------------ done
    return 0



if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
