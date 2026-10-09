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
    ldom        = np.zeros(success,dtype=int)
    lwidth      = np.zeros(success,dtype=int)
    lconv       = np.zeros(success)

    # thermal variables to be processed
    TE          = np.zeros(success)
    Wadv_thm    = np.zeros(success)
    Dthm        = np.zeros(success)

    # residual errors to be processed
    resid1      = np.zeros(success)
    resid3      = np.zeros(success)
    resens      = np.zeros(success)

    # parameter values to be saved
    params      = np.zeros((success,33))
    # ------------------------------------------------------------------------------------------------------------------------

    hdr_s = '    resid𝑠 ' if par.thermal else ''
    bar_s = ' ‾‾‾‾‾‾‾‾‾‾' if par.thermal else ''
    print('\n  ★     Damping σ     Frequency ω     𝒯/𝒫       resid𝐮 ' + hdr_s + '   Peak ℓ ℓ-Width ℓ-Convergence   cvf_r   cvf_l    cvfmax     residual')
    print(  ' ‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾' + bar_s + ' ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾ ')


    # Begin processing all solutions
    for i in range(success):

        if par.forcing == 0:
            w = eigval[i,1]
            sigma = eigval[i,0]
        else:
            w = ut.wf
            sigma = 0

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

        [ KE[i], Dkin0, Ensvel[i], Enscor[i], Ensvif[i], Ensbuo[i], Wthm0, Wdr0 ] = np.sum( udgn, 0)
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

        # resid1 is the relative residual of 2*sigma*KE - Dkin - Wthm = 0
        # resid3 is the relative residual of 2*sigma*TE - Dthm - Wadv = 0
        # ---------------------------------------------------------------------------------------------------------


        resid1[i] = ( abs( 2*sigma*KE[i] - Dkin[i] - Wthm[i] ) / max( abs(2*sigma*KE[i]), abs(Dkin[i]), abs(Wthm[i]) ) )

        if par.thermal:
            resid3[i] = ( abs( 2*sigma*TE[i] - Dthm[i] - Wadv_thm[i] ) / max( abs(2*sigma*TE[i]), abs(Dthm[i]), abs(Wadv_thm[i]) ) )

        col_s = '   {:8.2e}'.format(resid3[i]) if par.thermal else ''

        # -------i-------sigma-------w-----------KT/KP----resid1-----ldom-----lwidth-----lconv-------cuv_r-----cuv_l-----cuvm------resens------------------------------------------------------------------------------------------------------------------------------------
        print(' {:2d}   {: 12.9f}   {: 12.9f}   {:8.2e}   {:8.2e}{}    {:4d}    {:4d}     {:8.2e}     {:5.3f}   {:4d}    {:8.2e}    {:8.2e}'.format(i, sigma, w, KT[i]/KP[i], resid1[i], col_s, ldom[i], lwidth[i], lconv[i], cuvismax_r[i], cuvismax_l[i], cuvismax[i], resens[i]  ))
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        toc = timer()
        
        params[i,:] = np.array([
                                par.thermal,                    #0

                                par.m,                          #1
                                par.symm,                       #2
                                par.ricb,                       #3
                                par.bci,                        #4
                                par.bco,                        #5

                                par.forcing,                    #6
                                par.forcing_frequency,          #7
                                par.forcing_amplitude_cmb,      #8
                                par.forcing_amplitude_icb,      #9

                                par.Gaspard,                    #10
                                par.Beyonce,                    #11
                                par.ViscosD,                    #12
                                par.ThermaD,                    #13

                                par.ncpus,                      #14
                                par.N,                          #15
                                par.lmax,                       #16

                                timing+toc-tic,                 #17

                                par.aux0,                       #18
                                par.aux1,                       #19
                                par.aux2,                       #20
                                par.aux3,                       #21
                                par.aux4,                       #22
                                par.aux5,                       #23

                                par.visc0,                      #24
                                par.hvisc,                      #25
                                par.rvisc,                      #26

                                par.rpower_u,                   #27
                                par.rhopower_u,                 #28
                                par.rpower_v,                   #29
                                par.rhopower_v,                 #30

                                par.rpower_pp,                  #31
                                par.rhopower_pp                 #32
                                ])  # 33 total

    # ------------------------------------------------------------------------------------------------------------------------
    print(  ' ‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾' + bar_s + ' ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾ ')


    # ---------------------------------------------------------- write post-processed data and parameters to disk

    with open('params.dat','ab') as dpar:
        np.savetxt(dpar, params,
        fmt=[
            '%d',   '%d',   '%d',   '%.9e', '%d',   '%d',
            '%d',   '%.9e', '%.9e', '%.9e',
            '%.9e', '%.9e', '%.9e', '%.9e',
            '%d',   '%d',   '%d',   '%.9e',
            '%.9e', '%.9e', '%.9e', '%.9e', '%.9e', '%.9e',
            '%.9e', '%.9e', '%.9e',
            '%.9f', '%.9f', '%.9f', '%.9f', '%.9f', '%.9f'
            ])

    with open('flow.dat','ab') as dflo:
       np.savetxt(dflo, np.c_[ KE, KP, KT, Dkin, ldom, lwidth, lconv, Ensvel, Enscor, Ensvif, Ensbuo, cuvismax_r, cuvismax_l, cuvismax, resens ], fmt=['%.9e', '%.9e', '%.9e', '%.9e', '%d', '%d', '%.3e', '%.9e', '%.9e', '%.9e', '%.9e', '%.3e', '%d', '%.3e', '%.9e' ])


    if par.thermal:
        # columns: TE, Wadv, Dthm (times ThermaD), resid3 (entropy budget), Wthm (buoyancy work), resid1 (KE budget)
        with open('thermal.dat','ab') as dtmp:
            np.savetxt(dtmp, np.c_[ TE, Wadv_thm, Dthm, resid3, Wthm, resid1 ])


    if par.forcing == 0:
        with open('eigenvalues.dat','ab') as deig:
            np.savetxt(deig, eigval)

    # ------------------------------------------------------------------ done
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
