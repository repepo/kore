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
        ( 'aux0',                     par.aux0,                    '%.9e' ),
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

    success = np.shape(ru)[1]

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
    ldom        = np.zeros(success,dtype=int)
    lwidth      = np.zeros(success,dtype=int)
    lconv       = np.zeros(success)

    # thermal variables to be processed
    TE          = np.zeros(success)
    Wadv_thm    = np.zeros(success)
    Dthm        = np.zeros(success)

    # residual errors to be processed
    resid1      = np.zeros(success)
    resid2      = np.zeros(success)
    resid3      = np.zeros(success)
    resens      = np.zeros(success)

    # parameter values to be saved
    params      = []                 # one row of param_table() per solution
    # ------------------------------------------------------------------------------------------------------------------------

    sigmas      = np.zeros(success)  # damping of each solution
    freqs       = np.zeros(success)  # frequency of each solution

    # Summary table printed while processing: (header, format, value of solution i).
    # To print a new quantity, add a line here.
    table_cols = [
        ( '★',             '{:d}',      lambda i: i ),
        ( 'Damping σ',     '{: .9f}',   lambda i: sigmas[i] ),
        ( 'Frequency ω',   '{: .9f}',   lambda i: freqs[i] ),
        ( '𝒯/𝒫',           '{:8.2e}',   lambda i: ratio(KT[i], KP[i]) ),
        ( 'residσ',        '{:8.2e}',   lambda i: resid1[i] ),
        ( 'residω',        '{:8.2e}',   lambda i: resid2[i] ),
    ] + ([
        ( 'resid𝑠',        '{:8.2e}',   lambda i: resid3[i] ),
    ] if par.thermal else []) + [
        ( 'Peak ℓ',        '{:d}',      lambda i: ldom[i] ),
        ( 'ℓ-Width',       '{:d}',      lambda i: lwidth[i] ),
        ( 'ℓ-Convergence', '{:8.2e}',   lambda i: lconv[i] ),
        ( 'cvf_r',         '{:5.3f}',   lambda i: cuvismax_r[i] ),
        ( 'cvf_l',         '{:d}',      lambda i: cuvismax_l[i] ),
        ( 'cvfmax',        '{:8.2e}',   lambda i: cuvismax[i] ),
        ( 'residens',      '{:8.2e}',   lambda i: resens[i] ),
    ]
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

        rflow = np.copy(ru[:,i])
        iflow = np.copy(iu[:,i])
        # Expand solution
        u_sol2 = upp.expand_reshape_sol( rflow + 1j*iflow, par.symm)
        [ lp, lt, ll] = ut.ell(par.m, par.lmax, par.symm)
        lpi = np.searchsorted(ll,lp);  # Poloidal indices
        lti = np.searchsorted(ll,lt);  # Toroidal indices


        if par.thermal:
            rthm = np.copy(rt[:,i])
            ithm = np.copy(it[:,i])
            # Expand solution
            t_sol2  = upp.expand_reshape_sol( rthm + 1j*ithm, par.symm)


        # identify solutions
        [ ldom[i], lwidth[i], lconv[i] ] = upp.identify( u_sol2 )

        # diagnose solutions, in parallel
        [ udgn, tdgn ] = upp.diagnose( u_sol2, t_sol2, par.ricb, ut.rcmb, int(ncpus) )

        if par.ViscosD>0:
            Ra = par.ricb
            Rb = ut.rcmb
            ii = np.arange(0,par.N)
            xk = np.cos( (ii+0.5)*np.pi/par.N )
            rk = np.flipud(0.5*(Rb-Ra)*( xk + 1 ) + Ra)
            cuvis = upp.diagnose_4plot(int(ncpus), u_sol2, t_sol2, rk, 'curl_vis')
            idmax = np.unravel_index(np.argmax(abs(cuvis[:,2,0,:])), cuvis[:,2,0,:].shape)
            # the max value of the toroidal component of the curl of the viscous force is
            cuvismax[i] = abs(cuvis[idmax[0],2,0,idmax[1]])
            cuvismax_l[i] = idmax[0]
            cuvismax_r[i] = rk[idmax[1]]



        KP[i] = np.sum( udgn[lpi,0])  # Poloidal kinetic energy
        KT[i] = np.sum( udgn[lti,0])  # Toroidal kinetic energy

        [ KE[i], Dkin0, Ensvel[i], Enscor[i], Ensvif[i], Ensbuo[i], Wthm0, Wdr0,
          Wcor_im[i], Dkin_im[i], Wthm_im[i] ] = np.sum( udgn, 0)
        Dkin[i] = Dkin0
        resens[i] = abs(Ensvel[i]*sigma+Enscor[i]-Ensbuo[i]-Ensvif[i]) / max((abs(Ensvel[i]*sigma),abs(Enscor[i]),abs(Ensbuo[i]),abs(Ensvif[i])))
        Wthm[i] = Wthm0  # rate of working of buoyancy, Beyonce already included (upp.buoyancy)


        if par.thermal:

            [ TE[i], Dthm0, Wadv_thm[i] ] = np.sum( tdgn, 0)
            Dthm[i] = Dthm0 * par.ThermaD


        # --------------------------------------------------------- Computing residuals to check the power balance:
        # KE is kinetic energy
        # TE is the entropy "energy" (1/2) ∫ p s² dV

        # Dkin is the kinetic energy dissipation (viscous dissipation) via ∫𝐮⋅∇²𝐮 dV
        # Dthm is the entropy "dissipation" via ThermaD ∫ s ∇⋅(κp∇s) dV = -ThermaD ∫ κp|∇s|² dV (+ surface term, zero for kore's thermal BCs)

        # Wthm is the rate of working of the buoyancy force (thermal) via ∫ ρ 𝐮⋅(Beyonce g s 𝐫̂) dV

        # Wadv is the entropy advection "power" via -∫ p (dS/dr) uᵣ s dV

        # resid1 is the relative residual of 2*sigma*KE - Dkin - Wthm = 0 (residσ, real part of the power balance)
        # resid2 is the relative residual of 2*w*KE + Wcor_im - Dkin_im - Wthm_im = 0 (residω, its imaginary part;
        #        Wcor_im = 2 Im ∫ ρ 𝐮*⋅(2𝐳×𝐮) dV, Dkin_im = 2 Im ∫ 𝐮*⋅(∇⋅𝛔) dV, Wthm_im = 2 Im ∫ ρ 𝐮*⋅(Beyonce g s 𝐫̂) dV)
        # resid3 is the relative residual of 2*sigma*TE - Dthm - Wadv = 0
        # ---------------------------------------------------------------------------------------------------------


        resid1[i] = ( abs( 2*sigma*KE[i] - Dkin[i] - Wthm[i] ) / max( abs(2*sigma*KE[i]), abs(Dkin[i]), abs(Wthm[i]) ) )
        resid2[i] = ( abs( 2*w*KE[i] + Wcor_im[i] - Dkin_im[i] - Wthm_im[i] ) / max( abs(2*w*KE[i]), abs(Wcor_im[i]), abs(Dkin_im[i]), abs(Wthm_im[i]) ) )

        if par.thermal:
            resid3[i] = ( abs( 2*sigma*TE[i] - Dthm[i] - Wadv_thm[i] ) / max( abs(2*sigma*TE[i]), abs(Dthm[i]), abs(Wadv_thm[i]) ) )

        # ------------------------------------------------------------------------------------------------------------------
        print(table_row(table_cols, table_w, i))
        # ------------------------------------------------------------------------------------------------------------------

        params.append( [ v for _, v, _ in param_table(timing + timer() - tic) ] )

    # ------------------------------------------------------------------------------------------------------------------------
    print(table_line + '\n')


    # ---------------------------------------------------------- write post-processed data and parameters to disk

    with open('params.dat','ab') as dpar:
        np.savetxt(dpar, np.array(params),
        fmt=[ f for _, _, f in param_table(0) ])

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
        ]
        with open('thermal.dat','ab') as dtmp:
            np.savetxt(dtmp, np.column_stack([ v for _, v, _ in thermal_cols ]),
            fmt=[ f for _, _, f in thermal_cols ])


    if par.forcing == 0:
        with open('eigenvalues.dat','ab') as deig:
            np.savetxt(deig, eigval)

    print('Total time postprocessing:', timer()-tic, 'seconds')

    # ------------------------------------------------------------------ done
    return 0



if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
