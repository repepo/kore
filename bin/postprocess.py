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
from parameters import par
import utils as ut
import utils4pp as upp



def main(ncpus):

    # ------------------------------------------------------------------ Postprocessing: compute energy, dissipation, etc.
    #tic = timer()

    # fname_tm = 'timing.dat'

    # if os.path.isfile(fname_tm):
    #     timing = np.loadtxt(fname_tm)
    #     if np.size(timing)>1:
    #         timing = timing[-1]

    fname_ev = 'eigenvalues0.dat'
    if os.path.isfile(fname_ev):
        eigval = np.loadtxt(fname_ev).reshape((-1,2))

    fname_ru = 'real_flow.field'
    fname_iu = 'imag_flow.field'

    # fname_rb = 'real_magnetic.field'
    # fname_ib = 'imag_magnetic.field'

    if os.path.isfile(fname_ru) and os.path.isfile(fname_iu):
        ru = np.loadtxt(fname_ru).reshape((2*ut.n,-1))
        iu = np.loadtxt(fname_iu).reshape((2*ut.n,-1))
    # if os.path.isfile(fname_rb) and os.path.isfile(fname_ib):
    #     rb = np.loadtxt(fname_rb).reshape((2*ut.n,-1))
    #     ib = np.loadtxt(fname_ib).reshape((2*ut.n,-1))

    if par.hydro == 1:
        success = np.shape(ru)[1]
    # elif par.magnetic == 1:
    #     success = np.shape(rb)[1]

    # ------------------------------------------------------------------------------------------------------------------------
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

    KE          = np.zeros(success)
    KP          = np.zeros(success)
    KT          = np.zeros(success)
    Ro          = np.zeros(success)
    # brmsCMB     = np.zeros(success)
    # brmsOut     = np.zeros(success)
    # press0      = np.zeros(success)
    #params      = np.zeros((success,53))
    # ------------------------------------------------------------------------------------------------------------------------

    # print('\n  ★    m    symm    ω      Ek     Pm    η       K         KP/K       KT/K        Ro       |p_2m|    Br_rms_out[nT]')
    # print(  ' ‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾‾ ‾‾‾‾‾‾ ‾‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾')
    
    
    print('\n  ★    m    symm     ω       σ      Ek     η     K      KP/K       KT/K        Ro        DR Type      DR Amp  ')
    print(  ' ‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾ ‾‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾')


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
            [lp, lt, ll] = ut.ell(par.m, par.lmax, par.symm)
            lpi = np.searchsorted(ll,lp);  # Poloidal indices
            lti = np.searchsorted(ll,lt);  # Toroidal indices

        # if par.magnetic:
        #     rmag = np.copy(rb[:,i])
        #     imag = np.copy(ib[:,i])
        #     # Expand solution
        #     b_sol2 = upp.expand_reshape_sol( rmag + 1j*imag, ut.bsymm)
        #     b_sol  = upp.expand_sol( rmag + 1j*imag, ut.bsymm)  # this one for the torque 			

    
        # diagnose solutions, in parallel
        [ udgn, _, _, _ ] = upp.diagnose( u_sol2, b_sol2, t_sol2, c_sol2, par.ricb, ut.rcmb, int(ncpus))


        if par.hydro:
            
            KP[i] = np.sum( udgn[lpi,0])  # Poloidal kinetic energy
            KT[i] = np.sum( udgn[lti,0])  # Toroidal kinetic energy

            KE[i] = KP[i] + KT[i] # Total kinetic energy
            
            #[KE[i], Dkin0, Dint0, Wlor0, Wthm0, Wcmp0] = np.sum( udgn, 0)
            # Dkin[i] = par.OmgTau * par.Ek * Dkin0
            # Dint[i] = par.OmgTau * par.Ek * Dint0
            # Wlor[i] = par.OmgTau**2 * par.Le2 * Wlor0
            # Wthm[i] = par.OmgTau**2 * par.BV2 * Wthm0
            # Wcmp[i] = par.OmgTau**2 * par.BV2_comp * Wcmp0
            #press0[i] = np.abs(upp.pressure4pp(2, sigma+1j*w, u_sol2)[0])  # get the pressure coefficient |p_2m| at CMB

            Ro[i]= np.sqrt((3/(2*np.pi)) * KE[i] / (1 - par.ricb**3))
            
            # Viscous torques
            # vtorq[i] = par.OmgTau * par.Ek * np.dot( ut.gamma_visc(0,0,0), u_sol)[0]  # need to double check the constants here
            # vtorq_ic[i] = par.OmgTau * par.Ek * np.dot( ut.gamma_visc_icb(par.ricb), u_sol)[0]

            # press0[i] = udgn[6][0]


        # if par.magnetic:

        #     [ ME0, Mdfs0, Indu0, brmsCMB[i], brmsOut[i]] = np.sum( bdgn, 0)
        #     # ME[i]   = ME0   * par.OmgTau**2 * par.Le2
        #     # Indu[i] = Indu0 * par.OmgTau**2 * par.Le2
        #     # Mdfs[i] = par.OmgTau**3 * par.Le2 * par.Em * Mdfs0

        #     # # Magnetic torques
        #     # mtorq[i] = par.OmgTau**2 * par.Le2 * np.dot( ut.gamma_magnetic(), b_sol )[0]  # need to double check the constants here
        #     # mtorq_ic[i] = par.OmgTau**2 * par.Le2 * np.dot( ut.gamma_magnetic_ic(), b_sol )[0]

        #     brmsCMB[i] = par.B0_scale * np.sqrt(brmsCMB[i])
        #     brmsOut[i] = par.B0_scale * np.sqrt(brmsOut[i])
        
        # ------------------------------------------------------------------------------------------------------------------
        print('  {:2d}    {:8.2e}    {:8.2e}    {:8.2e}      {:8.2e}      {:8.2e}     {:8.2e}       {:8.2e}         {:8.2e}       {:8.2e}      {:8.2e}      {}       {:8.2e}'.format( \
               i, par.m, par.symm, w, sigma, par.Ek, par.ricb, KE[i], KP[i]/KE[i], KT[i]/KE[i], Ro[i], par.diff_rot_type, par.diff_rot_amplitude) )
        # ------------------------------------------------------------------------------------------------------------------

        #toc = timer()

    # ------------------------------------------------------------------------------------------------------------------------
    print(  ' ‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾‾ ‾‾‾‾‾‾ ‾‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾')
    return 0



if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))