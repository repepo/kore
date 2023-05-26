#!/usr/bin/env python3
'''
kore postprocessing script

Usage:
> python3 ./bin/postprocess.py

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
import utils_pp as upp



def main():
    
    # ------------------------------------------------------------------ Postprocessing: compute energy, dissipation, etc.
    tic = timer()
    
    fname_ev = 'eigenvalues0.dat'
    fname_tm = 'timing.dat'

    if os.path.isfile(fname_ev):
        eigval = np.loadtxt(fname_ev)    
        
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
        ru = np.loadtxt(fname_ru)
        iu = np.loadtxt(fname_iu)
    
    if os.path.isfile(fname_rb) and os.path.isfile(fname_ib):
        rb = np.loadtxt(fname_rb)
        ib = np.loadtxt(fname_ib)
        
    if os.path.isfile(fname_rt) and os.path.isfile(fname_it):
        rt = np.loadtxt(fname_rt)
        it = np.loadtxt(fname_it)
    
    if os.path.isfile(fname_rc) and os.path.isfile(fname_ic):
        rc = np.loadtxt(fname_rc)
        ic = np.loadtxt(fname_ic)
    
    success = np.shape(ru)[1]

    
    kid          = np.zeros((success,7))
    Dint_partial = np.zeros((success,3))
    p2t          = np.zeros(success)
    resid1       = np.zeros(success)
    resid2       = np.zeros(success)
    resid3       = np.zeros(success)
    y            = np.zeros(success)
    vtorq        = np.zeros(success,dtype=complex)
    vtorq_icb    = np.zeros(success,dtype=complex)
    mtorq        = np.zeros(success,dtype=complex)
    o2v          = np.zeros(success)
    
    if par.magnetic == 1:
        ohm = np.zeros((success,4))         
        Dohm_partial = np.zeros((success,3))
        
    if par.thermal == 1:
        therm = np.zeros((success,1))
        
    if par.compositional == 1:
        comp = np.zeros((success,4))

    params = np.zeros((success,33))
    #params = np.zeros((success,30))
    

        
    if par.Ek != 0:
        print('Ek = 10**{:<8.4f}'.format(np.log10(par.Ek)))
    else:
        print('Ek = 0')

    print('Post-processing:')    
    print('--- -------------- -------------- ---------- ---------- ---------- ---------- ---------- ----------')
    print('Sol    Damping        Frequency     Resid1     Resid2    ohm2visc    tor2pol   visc trq   mag trq  ')
    print('--- -------------- -------------- ---------- ---------- ---------- ---------- ---------- ----------')
    
    if par.track_target == 1:  # eigenvalue tracking enabled
        #read target data
        x = np.loadtxt('track_target')
    
    
    
    
    for i in range(success):
        
        #print('Processing solution',i)
        
        rflow = np.copy(ru[:,i])
        iflow = np.copy(iu[:,i])
        
        if par.forcing == 0:
            w = eigval[i,1]
            sigma = eigval[i,0]
        else:
            w = ut.wf
            sigma = 0
        
        kid[i,:] = upp.ken_dis( rflow, iflow, par.N, par.lmax, par.m, par.symm, \
        par.ricb, ut.rcmb, par.ncpus, w, par.projection, par.forcing, par.ricb, ut.rcmb)
        
        Dint_partial[i,0] = 0 #k1[2]*par.Ek
        Dint_partial[i,1] = 0 #k2[2]*par.Ek
        Dint_partial[i,2] = 0 #k3[2]*par.Ek
        
        KP = kid[i,0]
        KT = kid[i,1]
        p2t[i] = KP/KT
        KE = KP + KT
    
        Dint = kid[i,2]*par.Ek
        Dkin = kid[i,3]*par.Ek
        
        repow = kid[i,5]
        
        expsol = upp.expand_sol(rflow+1j*iflow, par.symm)
        
        vtorq[i] = par.Ek * np.dot( ut.gamma_visc(0,0,0), expsol)
        vtorq_icb[i] = par.Ek * np.dot( ut.gamma_visc_icb(par.ricb), expsol)
        
        if par.track_target == 1:   
            # compute distance (mismatch) to tracking target
            y0 = abs( (x[0]-sigma)/sigma )      # damping mismatch
            y1 = abs( (x[1]-w)/w )              # frequency mismatch
            y2 = abs( (x[2]-p2t[i])/p2t[i] )    # poloidal to toroidal energy ratio mismatch
            y[i] = y0 + y1 + y2                 # overall mismatch
               
        
        if par.magnetic == 1:
            
            rmag = np.copy(rb[:,i])
            imag = np.copy(ib[:,i])
            
            ohm[i,:] = upp.ohm_dis( rmag, imag, par.N, par.lmax, par.m, ut.bsymm, par.ricb, ut.rcmb, par.ncpus, par.ricb, ut.rcmb )
            
            Dohm_partial[i,0] = 0  #(o1[2] + o1[3])*par.Le2*par.Em
            Dohm_partial[i,1] = 0  #(o2[2] + o2[3])*par.Le2*par.Em
            Dohm_partial[i,2] = 0  #(o3[2] + o3[3])*par.Le2*par.Em
            
            Dohm = (ohm[i,2]+ohm[i,3])*par.Le2*par.Em   
            if Dint != 0:
                o2v[i] = Dohm/Dint
            else:
                o2v[i] = np.inf
            
            ME = (ohm[i,0]+ohm[i,1]) # Magnetic energy
            
            if par.mantle == 'TWA':
                mtorq[i] = par.Le2 * np.dot( ut.gamma_magnetic(), upp.expand_sol(rmag+1j*imag, par.symm*ut.symmB0) ) 
            
            if par.track_target == 1:
                y3 = abs( (x[3]-o2v[i])/o2v[i] )
                y[i] += y3  #include ohmic to viscous dissipation ratio in the tracking
        
        else:
            
            Dohm = 0
            ME = 0
        
        
        if par.thermal == 1:

            atemp = np.copy(rt[:,i])
            btemp = np.copy(it[:,i])
            therm[i,:] = upp.thermal_dis( atemp, btemp, rflow, iflow, par.N, par.lmax, par.m, par.symm, par.ricb, ut.rcmb, par.ncpus, par.ricb, ut.rcmb)

            Dbuoy = therm[i,0]*(-par.BV2)
            TE = therm[i,1]
            Dtemp = therm[i,2]*par.Etherm
            Dadv = therm[i,3]

        else:

            Dbuoy = 0
            TE = 0
            Dtemp = 0
            Dadv = 0

        if par.compositional== 1:

            acomp = np.copy(rc[:,i])
            bcomp = np.copy(ic[:,i])
            comp[i,:] = upp.thermal_dis( acomp, bcomp, rflow, iflow, par.N, par.lmax, par.m, par.symm, par.ricb, ut.rcmb, par.ncpus, par.ricb, ut.rcmb, thermal=False)

            Dbuoy_comp = comp[i,0]*(-par.BV2_comp)
            CE = comp[i,1]
            Dcomp = comp[i,2]*par.Ecomp
            Dadv_comp = comp[i,3]

        else:

            Dbuoy_comp = 0
            CE = 0
            Dcomp = 0
            Dadv_comp = 0
                            
                            
        # ------------------------------------------------------ Computing residuals to check the power balance:
        #
        # pss is the rate of working of stresses at the boundary
        # pvf is the rate of working of external volume force
        # Dint is the rate of change of internal energy
        # Dkin is the kinetic energy dissipation (viscous dissipation)
        # Dtemp is the rate of working of the buoyancy force
        # Dohm is the Ohmic dissipation or Joule heating
        # KE is kinetic energy
        # ME is magnetic energy
        # resid1 is the relative residual of Dkin + Dint - pss = 0
        # resid2 is the relative residual of 2*sigma*(KE + Le2*ME) - Dkin - Dtemp + Dohm - pvf = 0
        
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
        
        if par.Ek != 0:
            resid1[i] = abs( Dint + Dkin - pss ) / max( abs(Dint), abs(Dkin), abs(pss) )
        else:
            resid1[i] = np.nan
        resid2[i] = abs( 2*sigma*(KE + par.Le2*ME) - Dkin - Dtemp + Dohm - pvf ) / \
                    max( abs(2*sigma*(KE + par.Le2*ME)), abs(Dkin), abs(Dohm), abs(Dtemp), abs(pvf) )
        
        if par.thermal == 1:
            resid3[i] = abs(2*sigma*(TE) - Dadv - Dtemp) / max(abs(2*sigma*(TE)), abs(Dadv), abs(Dtemp))
        elif (par.thermal == 0) & (par.compositional == 1):
            resid3[i] = abs(2*sigma*(CE) - Dadv_comp - Dcomp)/max(abs(2*sigma*(CE)), abs(Dadv_comp), abs(Dcomp))
        else:
            resid3[i] = np.nan
        
        # print('Dkin  =' ,Dkin)
        # print('Dint  =' ,Dint)
        # print('Dohm  =' ,Dohm)
        # print('Dtemp =' ,Dtemp)
        # print('pss   = ',pss)
        # print('pvf   = ',pvf)
        # print('resid1 = ',resid1[i])
        # print('resid2 = ',resid2[i])
        # print('2sigmaK = ',2*sigma*KE)
        # print('2Le2sigmaM = ',2*sigma*ME*Le2)
        
        
        # ------------------------------------------------------------------------------------------------------
        
        #print('{:2d}   {: 12.9f}   {: 12.9f}   {:8.2e}   {:8.2e}   {:8.2e}   {:8.2e}   {:8.2e}   {:8.2e}'.format(i, sigma,\
        # w, resid1[i], resid2[i], o2v[i], KT/KP, np.abs(vtorq[i])/np.sqrt(KE), 2*np.real(mtorq[i]) ))
        print('{:2d}   {: 12.9f}   {: 12.9f}   {:8.2e}   {:8.2e}   {:8.2e}   {:8.2e}   {:8.2e}   {:8.2e}'.format(i, sigma,\
         w, resid1[i], resid2[i], o2v[i], KT/KP, 2*np.abs(vtorq[i])/np.sqrt(KE), 2*np.abs(mtorq[i])/np.sqrt(KE) ))
        
        #params[i,:] = np.array([par.Ek, par.m, par.symm, par.ricb, par.bci, par.bco, par.projection, par.forcing, \
        # par.forcing_amplitude_cmb, par.forcing_frequency, par.magnetic, par.Em, par.Le2, par.N, par.lmax, toc1-tic, \
        # par.ncpus, par.tol, par.thermal, par.Prandtl, par.Brunt, par.forcing_amplitude_icb, par.rc, par.h ])
         
        if par.mantle == 'insulator':
            mantle_mag_bc = 0
        elif par.mantle == 'TWA':
            mantle_mag_bc = 1
        
        toc = timer()
        
        params[i,:] = np.array([par.Ek, par.m, par.symm, par.ricb, par.bci, par.bco, par.projection, par.forcing,
                 par.forcing_amplitude_cmb, par.forcing_frequency, par.magnetic, par.Em, par.Le2, par.N, par.lmax, timing+toc-tic,
                 par.ncpus, par.tol, par.thermal, par.Etherm, par.BV2, par.compositional, par.Ecomp, par.BV2_comp,
                 par.forcing_amplitude_icb, par.rc, par.h, mantle_mag_bc, par.c_cmb, par.c1_cmb, par.mu, ut.B0_norm(), par.OmgTau ])
        
    print('--- -------------- -------------- ---------- ---------- ---------- ---------- ---------- ----------')
    
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
        
        
        
        
    # ---------------------------------------------------------- write post-processed data and parameters to disk
    #print('Writing results')
    
    with open('params.dat','ab') as dpar:
        np.savetxt(dpar, params, 
        fmt=['%.9e','%d'  ,'%d'  ,'%.9e' ,'%d'   ,'%d'  ,'%d'  ,'%d'  ,
             '%.9e','%.9e','%d'  ,'%.9e' ,'%.9e' ,'%d'  ,'%d'  ,'%.2f',
             '%d'  ,'%.2e', '%d' ,'%.9e' ,'%.9e' ,'%d'  ,'%.9e','%.9e',
             '%.9e','%.9e','%.9e','%d'   ,'%.9e' ,'%.9e','%.9e','%.9e','%d'])
    
    with open('flow.dat','ab') as dflo:
        np.savetxt(dflo, np.c_[kid, Dint_partial, np.real(vtorq), np.imag(vtorq), np.real(vtorq_icb), np.imag(vtorq_icb)])
    
    if par.magnetic == 1:
        with open('magnetic.dat','ab') as dmag:
            np.savetxt(dmag, np.c_[ohm, Dohm_partial, np.real(mtorq), np.imag(mtorq)])
            
    if par.thermal == 1:
        with open('thermal.dat','ab') as dtmp:
            np.savetxt(dtmp, therm) 

    if par.compositional == 1:
        with open('compositional.dat','ab') as dcmp:
            np.savetxt(dcmp, np.c_[comp])

    if par.forcing == 0:
        with open('eigenvalues.dat','ab') as deig:
            np.savetxt(deig, eigval)    
    
    # ------------------------------------------------------------------ done
    return 0
    
    
    
if __name__ == "__main__": 
    sys.exit(main())

    
