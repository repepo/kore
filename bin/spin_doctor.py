#!/usr/bin/env python3
'''
kore postprocessing script

Usage:
> ./bin/spin_doctor.py ncpus
'''

import sys
sys.path.insert(1,'bin/')
import scipy.io as sio
import scipy.sparse as ss
from timeit import default_timer as timer
import os.path
import numpy as np
import parameters as par
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

    fname_rb = 'real_magnetic.field'
    fname_ib = 'imag_magnetic.field'

    fname_rt = 'real_temperature.field'
    fname_it = 'imag_temperature.field'

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
    brms        = np.zeros(success)
    y           = np.zeros(success)                # for eigenmode tracking
    press0      = np.zeros(success)
    elldom      = np.zeros(success)
    params      = np.zeros((success,53))
    # ------------------------------------------------------------------------------------------------------------------------

    print('\n  ★     Damping σ     Frequency ω    resid0     resid𝐮     resid𝐛     residθ     Tor/Pol    Mag/Kin    |𝚪|mag    |𝚪|visc ')
    print(  ' ‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾')


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
            
            # Viscous torques
            vtorq[i] = par.OmgTau * par.Ek * np.dot( ut.gamma_visc(0,0,0), u_sol)  # need to double check the constants here
            vtorq_ic[i] = par.OmgTau * par.Ek * np.dot( ut.gamma_visc_icb(par.ricb), u_sol)

            press0[i] = udgn[6][0]


        if par.magnetic:

            [ ME0, Mdfs0, Indu0, brms[i] ] = np.sum( bdgn, 0)
            ME[i]   = ME0   * par.OmgTau**2 * par.Le2
            Indu[i] = Indu0 * par.OmgTau**2 * par.Le2
            Mdfs[i] = par.OmgTau**3 * par.Le2 * par.Em * Mdfs0

            # Magnetic torques
            mtorq[i] = par.OmgTau**2 * par.Le2 * np.dot( ut.gamma_magnetic(), b_sol )  # need to double check the constants here
            mtorq_ic[i] = par.OmgTau**2 * par.Le2 * np.dot( ut.gamma_magnetic_ic(), b_sol )

        if par.thermal:

            [ TE0, Dthm0, Wadv_thm0 ] = np.sum( tdgn, 0)
            TE[i]       = TE0       * par.OmgTau**2
            Wadv_thm[i] = Wadv_thm0 * par.OmgTau**2
            Dthm[i]     = Dthm0     * par.OmgTau**3 * par.Etherm


        if par.compositional:

            [ CE0, Dcmp0, Wadv_cmp0 ] = np.sum( cdgn, 0)
            CE[i]       = CE0 * par.OmgTau ** 2
            Wadv_cmp[i] = Wadv_cmp0 * par.OmgTau ** 2
            Dcmp[i]     = Dcmp0 * par.OmgTau ** 3 * par.Ecomp

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

        if par.magnetic:
            resid2[i] = abs( 2*sigma*ME[i] - Indu[i] - Mdfs[i] ) / \
                             max( abs(2*sigma*ME[i]), abs(Indu[i]), abs(Mdfs[i]))

        if par.thermal:
            resid3[i] = abs( 2*sigma*TE[i] - Dthm[i] - Wadv_thm[i] ) / \
                             max( abs(2*sigma*TE[i]), abs(Dthm[i]), abs(Wadv_thm[i]))

        # ------------------------------------------------------------------------------------------------------------------
        print(' {:2d}   {: 12.7f}   {: 12.7f}   {:8.2e}   {:8.2e}   {:8.2e}   {:8.2e}   {:8.2e}   {:8.2e}   {:8.2e}   {:8.2e}'.format( \
               i, sigma, w, resid0[i], resid1[i], resid2[i], resid3[i], KT[i]/KP[i], ME[i]/KE[i], np.abs(mtorq[i])/np.sqrt(KE[i])/par.OmgTau, np.abs(vtorq[i])/np.sqrt(KE[i])/par.OmgTau))
        # ------------------------------------------------------------------------------------------------------------------

        toc = timer()
        
        params[i,:] = np.array([                          
                                par.hydro,
                                par.magnetic,
                                par.thermal,
                                par.compositional,
                                
                                par.Ek,
                                par.m,
                                par.symm,
                                par.ricb,

                                par.bci,
                                par.bco,
                                par.forcing,
                                par.forcing_frequency,
                                
                                par.forcing_amplitude_cmb,
                                par.forcing_amplitude_icb,
                                par.projection,
                                ut.B0type,
                                
                                ut.beta_actual,
                                ut.B0_l,
                                ut.innercore_mag_bc,
                                par.c_icb,
                                
                                par.c1_icb,
                                ut.mantle_mag_bc,
                                par.c_cmb,
                                par.c1_cmb,
                                
                                par.mu,
                                par.Em,
                                par.Le2,
                                ut.B0_norm(),
                                
                                par.Etherm,
                                ut.heating,
                                par.BV2,
                                par.rc,
                                
                                par.h,
                                par.rsy,
                                par.bci_thermal,
                                par.bco_thermal,
                                
                                par.Ecomp,
                                ut.compositional_background,
                                par.BV2_comp,
                                par.rcc,
                                
                                par.hc,
                                par.rsyc,
                                par.bci_compositional,
                                par.bco_compositional,
                                
                                par.OmgTau,
                                par.ncpus,
                                par.N,
                                par.lmax,
                                
                                timing+toc-tic,
                                par.mu_i2o,
                                par.sigma_i2o,
                                par.aux1,
                                par.aux2
                                ])  # 53 total

    # ------------------------------------------------------------------------------------------------------------------------
    print(' ‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾\n')


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
            
            '%.9e', '%d',   '%d',   '%.9e',
            
            '%d',   '%d',   '%d',   '%.9e',
            
            '%.9e', '%.9e', '%d',   '%d' ,

            '%.9e', '%d',   '%d',   '%.9e',

            '%.9e', '%d',   '%.9e', '%.9e',
             
            '%.9e', '%.9e', '%.9e', '%.9e',

            '%.9e', '%d',   '%.9e', '%.9e',
           
            '%.9e', '%.9e', '%d',   '%d',

            '%.9e', '%d',   '%.9e', '%.9e',

            '%.9e', '%d',   '%d',   '%d',

            '%.9e', '%d',   '%d',   '%d',
             
            '%.2f', '%.9e', '%.9e' , '%.9e', '%.9e' ])

    if par.hydro:   
        with open('flow.dat','ab') as dflo:
            np.savetxt(dflo, np.c_[ KE,   KP,   KT,   Dkin,
                                    Dint, Wlor, Wthm, Wcmp,
                                    resid0, resid1,
                                    np.real(vtorq), np.imag(vtorq),
                                    np.real(vtorq_ic), np.imag(vtorq_ic),
                                    press0, elldom ])

    if par.magnetic:
        with open('magnetic.dat','ab') as dmag:
            np.savetxt(dmag, np.c_[ ME, Mdfs, Indu, resid2,
                                    np.real(mtorq), np.imag(mtorq),
                                    np.real(mtorq_ic), np.imag(mtorq_ic), brms])

    if par.thermal:
        with open('thermal.dat','ab') as dtmp:
            np.savetxt(dtmp, np.c_[ TE, Wadv_thm, Dthm, resid3 ])

    if par.compositional:
        with open('compositional.dat','ab') as dcmp:
            np.savetxt(dcmp, np.c_[ CE, Wadv_cmp, Dcmp ])

    if not par.forcing:
        with open('eigenvalues.dat', 'ab') as deig:
            np.savetxt(deig, np.c_[ eigval])

    # ------------------------------------------------------------------ done
    return 0



if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
