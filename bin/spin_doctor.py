#!/usr/bin/env python3
'''
kore postprocessing script

Usage:
> python3 ./bin/spin_doctor.py ncpus
'''

import sys
sys.path.insert(1,'bin/')
import scipy.io as sio
import scipy.sparse as ss
from timeit import default_timer as timer
import os.path
import numpy as np
from parameters import par
import utils as ut
import utils4pp as upp

import radial_profiles as rad


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

    fname_rt = 'real_thermal.field'
    fname_it = 'imag_thermal.field'

    fname_rc = 'real_composition.field'
    fname_ic = 'imag_composition.field'

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

    if par.hydro == 1:
        success = np.shape(ru)[1]
    elif par.magnetic == 1:
        success = np.shape(rb)[1]

    # ------------------------------------------------------------------------------------------------------------------
    # hydrodynamical variables to be processed
    KE          = np.zeros(success)
    KP          = np.zeros(success)
    KT          = np.zeros(success)
    Dkin        = np.zeros(success)
    Dint        = np.zeros(success)
    Ensvel      = np.zeros(success)
    Enscor      = np.zeros(success)
    Ensvif      = np.zeros(success)
    Ensbuo      = np.zeros(success)    
    cuvismax    = np.zeros(success)
    cuvismax_r  = np.zeros(success)
    cuvismax_l  = np.zeros(success, dtype=int)
    Wlor        = np.zeros(success)
    Wthm        = np.zeros(success)
    Wcmp        = np.zeros(success)
    vtorq       = np.zeros(success,dtype=complex)  # viscous torque on the mantle
    vtorq_icb   = np.zeros(success,dtype=complex)  # viscous torque on the inner core
    ldom        = np.zeros(success,dtype=int)
    lwidth      = np.zeros(success,dtype=int)
    lconv       = np.zeros(success)

    # magnetic variables to be processed
    ME          = np.zeros(success)
    Mdfs        = np.zeros(success)
    Indu        = np.zeros(success)
    mtorq       = np.zeros(success,dtype=complex)  # electromagnetic torque on the mantle

    # thermal variables to be processed
    TE          = np.zeros(success)
    Wadv_thm    = np.zeros(success)
    Dthm        = np.zeros(success)

    # compositional variables to be processed
    CE          = np.zeros(success)
    Wadv_cmp    = np.zeros(success)
    Dcmp        = np.zeros(success)

    # residual errors to be processed
    resid0      = np.zeros(success)
    resid1      = np.zeros(success)
    resid2      = np.zeros(success)
    resid3      = np.zeros(success)
    resens      = np.zeros(success)

    # tracking variables to be processed
    y           = np.zeros(success)                # for eigenmode tracking

    # parameter values to be saved
    params      = np.zeros((success,39))
    # ------------------------------------------------------------------------------------------------------------------------

    print('\n  ★     Damping σ     Frequency ω     𝒯/𝒫       resid𝐮    Peak ℓ ℓ-Width ℓ-Convergence   cvf_r   cvf_l    cvfmax     residual')
    print(  ' ‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾ ')


    if par.track_target == 1:  # eigenvalue tracking enabled
        #read target data
        x = np.loadtxt('track_target')

    # Begin processing all solutions
    for i in range(success):

        if par.forcing == 0:
            w = eigval[i,1]
            sigma = eigval[i,0]
        else:
            w = ut.wf
            sigma = 0

        [ u_sol2, b_sol2, t_sol2, c_sol2 ] = [0,0,0,0]

        if par.hydro:
            rflow = np.copy(ru[:,i])
            iflow = np.copy(iu[:,i])
            # Expand solution
            u_sol2 = upp.expand_reshape_sol( rflow + 1j*iflow, par.symm)
            u_sol  = upp.expand_sol( rflow + 1j*iflow, par.symm)  # this one for the torque
            [ lp, lt, ll] = ut.ell(par.m, par.lmax, par.symm)
            lpi = np.searchsorted(ll,lp);  # Poloidal indices
            lti = np.searchsorted(ll,lt);  # Toroidal indices

        # if par.magnetic:
        #     rmag = np.copy(rb[:,i])
        #     imag = np.copy(ib[:,i])
        #     # Expand solution
        #     b_sol2 = upp.expand_reshape_sol( rmag + 1j*imag, ut.bsymm)
        #     b_sol  = upp.expand_sol( rmag + 1j*imag, ut.bsymm)  # this one for the torque
                        
        if par.thermal:
            rthm = np.copy(rt[:,i])
            ithm = np.copy(it[:,i])
            # Expand solution
            t_sol2  = upp.expand_reshape_sol( rthm + 1j*ithm, par.symm)
            
        # if par.compositional:
        #     rcmp = np.copy(rc[:,i])
        #     icmp = np.copy(ic[:,i])
        #     # Expand solution
        #     c_sol2  = upp.expand_reshape_sol( rcmp + 1j*icmp, par.symm)		   			

        # identify solutions
        [ ldom[i], lwidth[i], lconv[i] ] = upp.identify( u_sol2 )

        # diagnose solutions, in parallel
        [ udgn, bdgn, tdgn, cdgn ] = upp.diagnose( u_sol2, b_sol2, t_sol2, c_sol2, par.ricb, ut.rcmb, int(ncpus) )

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
        #print(idmax[0], cuvismax_r )




        if par.hydro:
            
            KP[i] = np.sum( udgn[lpi,0])  # Poloidal kinetic energy
            KT[i] = np.sum( udgn[lti,0])  # Toroidal kinetic energy

            [ KE[i], Dkin0, Ensvel[i], Enscor[i], Ensvif[i], Ensbuo[i], Wlor0, Wthm0, Wcmp0 ] = np.sum( udgn, 0)
            Dkin[i] = Dkin0
            resens[i] = abs(Ensvel[i]*sigma+Enscor[i]-Ensbuo[i]-Ensvif[i]) / max((abs(Ensvel[i]*sigma),abs(Enscor[i]),abs(Ensbuo[i]),abs(Ensvif[i])))
            # Dint[i] = par.ViscosD * Dint0
            # Wlor[i] = 0#par.OmgTau**2 * par.Le2 * Wlor0
            # Wthm[i] = par.Beyonce * Wthm0
            # Wcmp[i] = 0#par.OmgTau**2 * par.BV2_comp * Wcmp0

            # # Viscous torques
            # vtorq[i] = 0#par.Ek * np.dot( ut.gamma_visc(0,0,0), u_sol)  # need to double check the constants here
            # vtorq_icb[i] = 0#par.Ek * np.dot( ut.gamma_visc_icb(par.ricb), u_sol)


        # if par.magnetic:

        #     [ ME0, Dohm0, Indu0 ] = np.sum( bdgn, 0)
        #     ME[i]   = 0#ME0   * par.OmgTau**2 * par.Le2
        #     #Dohm = Dohm0 * par.OmgTau**3 * par.Le2 * par.Em
        #     Indu[i] = 0#par.OmgTau * par.Em * Indu0

        #     if ((par.mantle == 'TWA') and (par.m==0) and (par.symm==1)):
        #         mtorq[i] = par.Le2 * np.dot( ut.gamma_magnetic(), b_sol )  # need to double check the constants here


        # if par.compositional:

        #     [ TE[i], Dthm0, Wadv_thm[i] ] = np.sum( tdgn, 0) 
        #     Dthm[i] = Dthm0 * par.Etherm


        # if par.compositional:
            
        #     [ CE[i], Dcmp0, Wadv_cmp[i] ] = np.sum( cdgn, 0)
        #     Dcmp[i] = Dcmp0 * par.Ecomp 


        # --------------------------------------------------------- Computing residuals to check the power balance:
        # KE is kinetic energy
        # ME is magnetic energy
        # TE is the thermal "energy" (p/2) ∫ S'² dV

        # Dint is the rate of change of internal energy
        # Dkin is the kinetic energy dissipation (viscous dissipation) via ∫𝐮⋅∇²𝐮 dV
        # Dohm is the Ohmic dissipation or Joule heating via ∫|∇×𝐛|² dV
        # Dthm is the thermal "dissipation" via ∫ S' ∇⋅κp∇S' dV

        # Wthm is the rate of working of the Lorentz force
        # Wthm is the rate of working of the buoyancy force (thermal)
        # Wcmp is the rate of working of the buoyancy force (compositional)

        # Indu is the magnetic induction "power"
        # Wadv is the thermal advection "power" via ∫ (-𝐮⋅r p dS'/dr ) dV

        # resid0 is the relative residual of Dkin + Dint = 0
        # resid1 is the relative residual of 2*sigma*KE - Dkin - Wlor -Wthm = 0
        # resid2 is the relative residual of 2*sigma*ME - Dohm - Indu  = 0
        # resid3 is the relative residual of 2*sigma*TE - Dthm - Wadv = 0
        # ---------------------------------------------------------------------------------------------------------

        # if par.ViscosD != 0 and par.hydro == 1:
        #     resid0[i] = abs( Dint0 + Dkin0 ) / max( abs(Dint0), abs(Dkin0) )
        # else:
        #     resid0[i] = np.nan

        if par.hydro:
            resid1[i] = ( abs( 2*sigma*KE[i] - Dkin[i] ) / max( abs(2*sigma*KE[i]), abs(Dkin[i])) )     
                         
        

        # -------i-------sigma-------w-----------KT/KP----resid1-----ldom-----lwidth-----lconv-------cuv_r-----cuv_l-----cuvm------resens------------------------------------------------------------------------------------------------------------------------------------
        print(' {:2d}   {: 12.9f}   {: 12.9f}   {:8.2e}   {:8.2e}    {:4d}    {:4d}     {:8.2e}     {:5.3f}   {:4d}    {:8.2e}    {:8.2e}'.format(i, sigma, w, KT[i]/KP[i], resid1[i], ldom[i], lwidth[i], lconv[i], cuvismax_r[i], cuvismax_l[i], cuvismax[i], resens[i]  ))
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        toc = timer()
        
        params[i,:] = np.array([
                                par.hydro,                      #0
                                par.magnetic,                   #1
                                par.thermal,                    #2
                                par.compositional,              #3

                                par.m,                          #4
                                par.symm,                       #5
                                par.ricb,                       #6
                                par.bci,                        #7
                                par.bco,                        #8

                                par.forcing,                    #9
                                par.forcing_frequency,          #10
                                par.forcing_amplitude_cmb,      #11
                                par.forcing_amplitude_icb,      #12
                                par.projection,                 #13

                                par.Gaspard,                    #14
                                par.Beyonce,                    #15
                                par.Hendrik,                    #16
                                par.ViscosD,                    #17
                                par.ThermaD,                    #18
                                par.MagnetD,                    #19

                                par.ncpus,                      #20
                                par.N,                          #21
                                par.lmax,                       #22

                                timing+toc-tic,                 #23

                                par.aux0,                       #24
                                par.aux1,                       #25
                                par.aux2,                       #26
                                par.aux3,                       #27
                                par.aux4,                       #28
                                par.aux5,                       #29

				                par.visc0,			            #30
			                    par.hvisc,			            #31
				                par.rvisc,			            #32

                                par.rpower_u,                   #33
                                par.rhopower_u,                 #34
                                par.rpower_v,                   #35
                                par.rhopower_v,                 #36

                                par.rpower_pp,                  #37
                                par.rhopower_pp                 #38
                                ])  # 39 total

    # ------------------------------------------------------------------------------------------------------------------------
    print(  ' ‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾ ')

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
        np.savetxt(dpar, params,
        fmt=[
            '%d',   '%d',   '%d',   '%d',
            
            '%d',   '%d',   '%.9e', '%d',
            
            '%d',   '%d',   '%.9e', '%.9e',
        
            '%.9e', '%d',   '%.9e', '%.9e',
             
            '%.9e', '%.9e', '%.9e', '%.9e',
        
            '%d',   '%d',   '%d',   '%.9e',
             
            '%.9e', '%.9e', '%.9e', '%.9e',
            
            '%.9e', '%.9e', '%.9e', '%.9e',

            '%.9e', '%.9f', '%.9f', '%.9f',

            '%.9f', '%.9f', '%.9f'
            ])

    if par.hydro:   
        with open('flow.dat','ab') as dflo:
           # np.savetxt(dflo, np.c_[ KE,   KP,   KT,   Dkin,
           #                         Dint, Wlor, Wthm, Wcmp,
           #                         resid0, resid1,
           #                         np.real(vtorq), np.imag(vtorq),
           #                         np.real(vtorq_icb), np.imag(vtorq_icb)])
           np.savetxt(dflo, np.c_[ KE, KP, KT, Dkin, ldom, lwidth, lconv, Ensvel, Enscor, Ensvif, Ensbuo, cuvismax_r, cuvismax_l, cuvismax, resens ], fmt=['%.9e', '%.9e', '%.9e', '%.9e', '%d', '%d', '%.3e', '%.9e', '%.9e', '%.9e', '%.9e', '%.3e', '%d', '%.3e', '%.9e' ])

    # if par.magnetic:
    #     with open('magnetic.dat','ab') as dmag:
    #         np.savetxt(dmag, np.c_[ ME, Mdfs, Indu, resid2,
    #                                 np.real(mtorq), np.imag(mtorq)])

    # if par.thermal:
    #     with open('thermal.dat','ab') as dtmp:
    #         np.savetxt(dtmp, np.c_[ TE, Wadv_thm, Dthm, resid3 ])

    # if par.compositional:
    #     with open('compositional.dat','ab') as dcmp:
    #         np.savetxt(dcmp, np.c_[ CE, Wadv_cmp, Dcmp ])

    if par.forcing == 0:
        with open('eigenvalues.dat','ab') as deig:
            np.savetxt(deig, eigval)

    # ------------------------------------------------------------------ done
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
