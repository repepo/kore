#!/usr/bin/env python3
'''
kore generates submatrices

To use:
> ./bin/submatrices.py ncpus
where ncpus is the number of cores

This program generates (in parallel) the operators involved in the differential
equations as submatrices. They will be written to disk as .mtx files.
These submatrices are required for the assembly of the main matrices A and B.
'''

from timeit import default_timer as timer
import multiprocessing as mp
import scipy.sparse as ss
import scipy.io as sio
import numpy as np
import warnings
import sys
import parameters as par
import utils as ut
import radial_profiles as rap



def main(ncpus):

    warnings.simplefilter('ignore', ss.SparseEfficiencyWarning)

    twozone    = ((par.thermal == 1) and (par.heating == 'two zone'))               # boolean
    userdef    = ((par.thermal == 1) and (par.heating == 'user defined'))           # boolean
    cdipole    = ((par.magnetic == 1) and (par.B0 == 'dipole') and (par.ricb > 0))  # boolean
    inviscid   = ((par.Ek == 0) and (par.ricb == 0))                                # boolean
    quadrupole = ((par.B0 == 'Luo_S2') or ((par.B0 == 'FDM') and (par.B0_l == 2)))  # boolean

    # radial_profiles = ['eta', 'rho', 'tem','buo']

    tic = timer()
    print('N =', par.N,', lmax =', par.lmax)

    # vector parity for poloidal and toroidals. If ricb>0 we don't use parities, not needed.
    # if vP = 1 then we need only even parity Cheb polynomials
    # if vP = -1 we need only odd parity Chebs
    vP = int( (1 - 2*(par.m%2)) * par.symm )
    vT = -vP
    # for the magnetic field the parities are opposite to that of the flow if B0 is equatorially antisymmetric.
    vF = ut.symmB0*vP
    vG = -vF

    tol = 1e-9
    # Chebyshev coefficients of powers of r
    r0  = ut.chebco(0, par.N, tol, par.ricb, ut.rcmb)
    r1  = ut.chebco(1, par.N, tol, par.ricb, ut.rcmb)
    r2  = ut.chebco(2, par.N, tol, par.ricb, ut.rcmb)
    r3  = ut.chebco(3, par.N, tol, par.ricb, ut.rcmb)
    r4  = ut.chebco(4, par.N, tol, par.ricb, ut.rcmb)
    r5  = ut.chebco(5, par.N, tol, par.ricb, ut.rcmb)
    r6  = ut.chebco(6, par.N, tol, par.ricb, ut.rcmb)

    rp = [r0, r1, r2, r3, r4, r5, r6]

    # these are the cheb coeffs of r*twozone or r*BVprof.
    if twozone:
        cd_ent = ut.chebco_rf(rap.twozone, rpower=1, N=par.N, ricb=par.ricb, rcmb=ut.rcmb, tol=tol, args=par.args).reshape([par.N,1])
    elif userdef:
        cd_ent = ut.chebco_rf( rap.BVprof, rpower=1, N=par.N, ricb=par.ricb, rcmb=ut.rcmb, tol=tol, args=par.args).reshape([par.N,1])

    elif par.anelastic:
        #rd_rho = ut.get_radial_derivatives(rap.log_density,4,4,tol) # Density : Requires derivatives and radial powers up to fourth order
        # rd_tem = ut.get_radial_derivatives(ut.temperature,2,2,tol) # Temperature : Requires derivatives and radial powers up to second order
        # rd_buo = ut.get_radial_derivatives(ut.buoFac,)
        #rd_kho = ut.get_radial_derivatives(rap.kappa_rho,2,1,tol) # thermaldiffusivity * density
        #rd_lnT = ut.get_radial_derivatives(rap.log_temperature,2,1,tol) # Log(Temperature)

        cd_rho = ut.chebify( rap.density, 2, tol)
        cd_kho = ut.chebify( rap.kappa_rho, 1, tol)
        cd_lho = ut.chebify( rap.log_density, 4, tol)
        cd_lnT = ut.chebify( rap.log_temperature, 1, tol)
        cd_vsc = ut.chebify(rap.viscosity,2,tol)

        cd_ent = ut.chebco_rf( rap.entropy_gradient, rpower=0, N=par.N, ricb=par.ricb, rcmb=ut.rcmb, tol=tol, args=par.dent_args).reshape([par.N,1])

        cd_buo = ut.chebco_rf(rap.buoFac,0,par.N,par.ricb,ut.rcmb,tol)
        cd_buo = ut.cheb2Product(cd_buo,rap.gravCoeff(),tol).reshape([par.N,1])

    if par.magnetic == 1 :

        cnorm = ut.B0_norm()  # Normalization

        rpw = [ 0, 1, 2, 3, 4, 5, -1]  # powers of r needed for the h function
        rdh = [ [ [] for j in range(4) ] for i in range(7) ]
        for i,rpw1  in enumerate( rpw ):
            for j in range(4):
                args = [ par.beta, par.B0_l, par.ricb, rpw1, j ]
                rdh[i][j] = cnorm * ut.chebco_h( args, par.B0, par.N, 1, tol)
        # rdh is a 2D list where each element is the set of Chebyshev coeffs
        # of the function (r**rpw)*(d/dr)**j*(h_l(r))
        # in the last row (row 6) the power of r is -1
        # columns are derivative order, first column (col 0) is for the function h itself

        #rd_eta = ut.get_radial_derivatives(rap.magnetic_diffusivity,2,1,tol)  # Magnetic diffusivity profile
        cd_eta = ut.chebify( rap.magnetic_diffusivity, 1, tol)
        cd_eho = ut.chebify( rap.eta_rho, 1, tol)

    # Gegenbauer basis transformations
    S0 = ut.Slam(0, par.N) # From the Chebyshev basis ( C^(0) basis ) to C^(1) basis
    S1 = ut.Slam(1, par.N) # From C^(1) basis to C^(2) basis
    S2 = ut.Slam(2, par.N) # From C^(2) basis to C^(3) basis
    S3 = ut.Slam(3, par.N) # From C^(3) basis to C^(4) basis

    # Matrices to compute derivatives
    # The result will be in a higher Gegenbauer order basis according to
    # the order of the derivative, e.g. D3 will change the basis from C^(0) to C^(3)
    D1 = ut.Dlam(1, par.N)
    D2 = ut.Dlam(2, par.N)
    D3 = ut.Dlam(3, par.N)
    D4 = ut.Dlam(4, par.N)
    D  = [ 1, D1, D2, D3, D4 ]

    # Auxiliary basis transformations
    S10   = S1*S0
    S21   = S2*S1
    S210  = S2*S10
    S32   = S3*S2
    S321  = S32*S1
    S3210 = S321*S0
    S     = [ 1, S0, S10, S210, S3210 ]
    G4    = [ 1, S3, S32, S321, S3210 ]
    G3    = [ 1, S2, S21, S210 ]
    G2    = [ 1, S1, S10 ]
    G1    = [ 1, S0 ]
    G     = [ G1, G2, G3, G4 ]  # fixed this for the inviscid case

    # Sets the Gegenbauer basis order for each section
    if ((par.magnetic == 1) and ('conductor' in par.innercore)) :
        gebasis = [  4,   2,   3,   2,   2,   2  ]
    else:
        gebasis = [  4,   2,   2,   2,   2,   2  ]
    section     = [ 'u', 'v', 'f', 'g', 'h', 'i' ]

    if inviscid:
        gebasis[0] = 2  # only up to second derivatives in section u
        gebasis[1] = 1  # up to first derivatives in section v

    # Zero matrices, used when making room for bc's
    N1 = int((1 + np.sign(par.ricb)) * int(par.N/2))
    z4 = ss.csr_matrix((4,N1))
    z3 = ss.csr_matrix((3,N1))
    z2 = ss.csr_matrix((2,N1))
    z1 = ss.csr_matrix((1,N1))
    Z = [ z1, z2, z3, z4 ]

    '''
    We want the product matrices MXY (power X of r times something in the C^(Y) basis)
    these are the most time consuming to compute
    The derivative operator DY will be included later.
    These matrices are computed with the ut.Mlam function
    its arguments are r^(X) in the C^(Y) basis, Y, and vector parity
    e.g. M43 = ut.Mlam( S210*r4, 3, -1)
    We generate now a list of labels (labl) of the product matrices needed together with the corresponding vector_parity list (arg2)
    '''

    labl = []
    arg2 = []

    if par.hydro == 1:
        # -------------------------------------------------------------------------------------------------------------------------------------------
        # Matrices needed for the Navier-Stokes equation, double curl equations ------------------------------------------- NavStok 2curl - section u
        # -------------------------------------------------------------------------------------------------------------------------------------------

        # u
        arg2  += [     vP ,     vP ,     vP  ]
        labl_u = [ 'r2_D0', 'r3_D1', 'r4_D2' ]

        # Coriolis
        arg2   += [     vT ,     vT  ]
        labl_u += [ 'r3_D0', 'r4_D1' ]

        # Viscous diffusion
        arg2   += [     vP ,     vP ,     vP ,     vP  ]
        labl_u += [ 'r0_D0', 'r2_D2', 'r3_D3', 'r4_D4' ]

        # More viscous diffusion, anelastic terms
        if par.anelastic:
            arg2   += [       vP   ,        vP    ,       vP    ,       vP    ,       vP    ,
                              vP   ,        vP    ,       vP    ,       vP     ]
            labl_u += [ 'r1_lho1_D0', 'r2_lho2_D0', 'r3_lho3_D0', 'r2_lho1_D1', 'r3_lho2_D1',
                        'r3_lho1_D2', 'r4_lho2_D2', 'r4_lho3_D1', 'r4_lho1_D3' ]

            # And even more viscous diffusion,
            if par.variable_viscosity:
                labl_u += ['r0_vsc0_D0', 'r1_vsc0_lho1_D0', 'r1_vsc1_D0', 'r2_vsc0_D2',
                        'r2_vsc0_lho1_D1', 'r2_vsc0_lho2_D0', 'r2_vsc1_lho1_D0',
                        'r2_vsc2_D0', 'r3_vsc0_D3', 'r3_vsc0_lho1_D2', 'r3_vsc0_lho2_D1',
                        'r3_vsc0_lho3_D0', 'r3_vsc1_D2', 'r3_vsc1_lho1_D1', 'r3_vsc1_lho2_D0',
                        'r3_vsc2_lho1_D0', 'r4_vsc0_D4', 'r4_vsc0_lho1_D3', 'r4_vsc0_lho2_D2',
                        'r4_vsc0_lho3_D1', 'r4_vsc1_D3', 'r4_vsc1_lho1_D2', 'r4_vsc1_lho2_D1',
                        'r4_vsc2_D2', 'r4_vsc2_lho1_D1']

        if par.magnetic == 1 :
            # add Lorentz force
            arg2   += [     vF    ,     vF    ,     vF    ,     vF    ,     vF    ,     vF    ,     vF    ,
                            vF    ,     vF    ,     vG    ,     vG    ,     vG    ,     vG    ,     vG     ]  # vF for bpol, vG for btor
            labl_u += [ 'r1_h0_D1', 'r2_h1_D1', 'r2_h0_D2', 'r3_h1_D2', 'r0_h0_D0', 'r1_h1_D0', 'r2_h2_D0',
                        'r3_h3_D0', 'r3_h0_D3', 'r1_h0_D0', 'r2_h1_D0', 'r2_h0_D1', 'r3_h1_D1', 'r3_h0_D2' ]

            if quadrupole :
                arg2   += [     vF    ,     vG    ]
                labl_u += [ 'r3_h2_D1', 'r3_h2_D0']

        if (par.thermal == 1) or (par.compositional==1) :
            # add Buoyancy force

            if par.anelastic:
                labl_u += ['r3_buo0_D0']
            else:
                arg2   += [     vP  ]
                labl_u += [ 'r4_D0' ]

        labl += ut.labelit( labl_u, section='u', rplus=2*cdipole)


        # -------------------------------------------------------------------------------------------------------------------------------------------
        # Matrices needed for the Navier-Stokes equation, single curl equations ------------------------------------------- NavStok 1curl - section v
        # -------------------------------------------------------------------------------------------------------------------------------------------

        # u
        arg2  += [     vT  ]
        labl_v = [ 'r2_D0' ]

        # Coriolis
        arg2   += [     vP ,     vP  ]
        labl_v += [ 'r1_D0', 'r2_D1' ]

        # Viscous diffusion
        arg2   += [     vT ,     vT ,     vT  ]
        labl_v += [ 'r0_D0', 'r1_D1', 'r2_D2' ]

        # More viscous diffusion, anelastic terms
        if par.anelastic:
            arg2   += [       vT    ,       vT    ,       vT     ]
            labl_v += [ 'r1_lho1_D0', 'r2_lho2_D0', 'r2_lho1_D1' ]

            if par.variable_viscosity:
                labl_v += ['r0_vsc0_D0', 'r1_vsc0_D1', 'r1_vsc0_lho1_D0', 'r2_vsc0_D2',
                           'r2_vsc0_lho1_D1', 'r2_vsc1_D1', 'r2_vsc1_lho1_D0']

        if par.magnetic == 1 :
            # Lorentz force
            labl_v += [ 'r0_h0_D1', 'r0_h1_D0', 'r1_h2_D0', 'r1_h0_D2', 'r0_h0_D0', 'r1_h1_D0', 'r1_h0_D1' ]
            arg2   += [     vF    ,     vF    ,     vF    ,     vF    ,     vG    ,     vG    ,     vG     ]

        labl += ut.labelit( labl_v, section='v', rplus=3*cdipole)


    if par.magnetic == 1 :
        # -------------------------------------------------------------------------------------------------------------------------------------------
        # Matrices needed for the Induction equation, no-curl or consoidal equations -------------------------------------- Induct nocurl - section f
        # -------------------------------------------------------------------------------------------------------------------------------------------

        #if ((par.ricb > 0) and ('conductor' in par.innercore)) :  # conducting inner core, consoidal component, needs work, don't use!
        #    labl += [ 'f21',  'f32', 'f31',  'f20',   'f33',  'f22', 'f11', 'f00' ]

        # b
        if not par.anelastic:
            labl_f  = [ 'r2_D0' ]
        else:
            labl_f  = [ 'r2_rho0_D0' ]
        arg2   += [     vF  ]

        # induction
        labl_f += [ 'r0_h0_D0', 'r1_h1_D0', 'r1_h0_D1', 'r1_h0_D0' ]
        arg2   += [     vP    ,     vP    ,     vP    ,     vT     ]

        # magnetic diffusion
        if par.anelastic:
            labl_f += [ 'r0_eho0_D0', 'r1_eho0_D1', 'r2_eho0_D2' ]
        else:
            labl_f += [ 'r0_eta0_D0', 'r1_eta0_D1', 'r2_eta0_D2' ]
        arg2   += [      vF     ,      vF     ,      vF      ]

        labl += ut.labelit( labl_f, section='f', rplus=2*cdipole)


        # -------------------------------------------------------------------------------------------------------------------------------------------
        # Matrices needed for the Induction equation, single curl equations ------------------------------------------------ Induct 1curl - section g
        # -------------------------------------------------------------------------------------------------------------------------------------------

        # b
        if not par.anelastic:
            labl_g  = [ 'r2_D0' ]
        else:
            labl_g  = [ 'r2_rho0_D0' ]
        arg2   += [    vG   ]

        # induction
        labl_g += [ 'r0_h0_D1', 'r1_h1_D1', 'q1_h0_D0', 'r0_h1_D0', 'r1_h2_D0', 'r1_h0_D2',
                    'r0_h0_D0', 'r1_h0_D1', 'r1_h1_D0' ]  # 'q1_h0_D0' is for (1/r)*h(r)
        arg2   += [     vP    ,     vP    ,     vP     ,    vP    ,     vP    ,     vP    ,
                        vT    ,     vT    ,     vT     ]
        if par.anelastic:
            labl_g += [ 'r0_h0_lho1_D0', 'r1_h1_lho1_D0', 'r1_h0_lho1_D0', 'r1_h0_lho1_D1' ]
            arg2   += [        vP      ,        vT       ,        vT  ,         vT    ] #Check the last one

        # magnetic diffusion
        if par.anelastic:
            labl_g += [ 'r0_eho0_D0', 'r1_eho0_D1', 'r2_eho0_D2','r1_eta1_rho0_D0','r2_eta1_rho0_D1' ]
        else:
            labl_g += [ 'r0_eta0_D0', 'r1_eta0_D1', 'r2_eta0_D2', 'r1_eta1_D0', 'r2_eta1_D1' ]
        arg2   += [      vG     ,      vG     ,      vG     ,      vG     ,      vG      ]

        labl += ut.labelit( labl_g, section='g', rplus=3*cdipole)
        # might need to add r0_D0_g for Lin2017 forcing:
        # labl += [ 'r0_D0_g' ]
        # arg2 += [     vG    ]


    if par.thermal == 1 :
        # -------------------------------------------------------------------------------------------------------------------------------------------
        # Matrices needed for the heat equation ------------------------------------------------------------------------------------ Heat - section h
        # -------------------------------------------------------------------------------------------------------------------------------------------
        if par.anelastic:

            #Advection
            labl_h = ['r0_drS0_D0','r1_drS0_D0']

            # difus = - L*r0_kho0_D0_h + 2*r1_kho0_D1_h + r2_kho0_D2_h + r2_kho0_lnT1_D1_h + r2_kho1_D1_h
            labl_h += [ 'r0_kho0_D0', 'r1_kho0_D1', 'r2_kho0_D2', 'r2_kho0_lnT1_D1', 'r2_kho1_D1' ]
            labl_h += ['r2_rho0_D0'] #ds/dt
        else:
            if par.heating == 'differential' :

                labl_h = [ 'r0_D0', 'r1_D0', 'r2_D1',  'r3_D2', 'r3_D0' ]  # here the operators have mixed symmetries!?, this is potentially a problem when ricb == 0
                arg2  += [     vP ,     vP ,     vP ,      vP ,     vP  ]

            elif (par.heating == 'internal' or (twozone or userdef) ) :

                labl_h = [ 'r2_D0', 'r0_D0', 'r1_D1', 'r2_D2' ]
                if not cdipole:
                    arg2 += [  vP ,     vP ,     vP ,     vP  ]

                if (twozone or userdef) :

                    # this is for ut.twozone or ut.BVprof, we have r0 in the label because there is an r already included in the functions
                    labl_h += [ 'r0_drS0_D0' ]
                    if not cdipole:
                        arg2 += [   vP  ]



        labl += ut.labelit( labl_h, section='h', rplus=0)


    if par.compositional == 1 :
        # -------------------------------------------------------------------------------------------------------------------------------------------
        # Matrices needed for the compositional equation ------------------------------------------------------------------ Compositional - section i
        # -------------------------------------------------------------------------------------------------------------------------------------------

        if par.comp_background == 'differential' :

            labl_i = [ 'r0_D0', 'r1_D0', 'r2_D1',  'r3_D2', 'r3_D0' ]  # here the operators have mixed symmetries, this is a problem if ricb == 0
            arg2  += [     vP ,     vP ,     vP ,      vP ,     vP  ]

        elif (par.comp_background == 'internal' or (twozone or userdef) ) :

            labl_i = [ 'r2_D0', 'r0_D0', 'r1_D1',  'r2_D2' ]
            if not cdipole:
                arg2 += [ vP ,     vP ,     vP ,      vP  ]

        labl += ut.labelit( labl_i, section='i', rplus=0)



    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Pre-process the list with multiplication matrices labels to avoid duplicates --------------------------------------------------------------
    # Also generate the list of arguments parg0 and parg1 ---------------------------------------------------------------------------------------

    plabl = []
    parg0 = []
    parg1 = []
    parg2 = []

    if cdipole or (par.ricb > 0) :  # set vector_parity = 0, i.e. is not needed
        arg2 = np.size(labl)*[0]

    for k,labl1 in enumerate(labl) :

        [ lablx, rx, hx, dx, secx, profid1, dp1, profid2, dp2 ] = ut.decode_label(labl1)

        idx = [ j for j,x in enumerate(plabl) if x == lablx ]  # find indices of same labels as lablx in plabl
        vpx = [ parg2[i] for i in idx ]  # find the vector_parities of those

        if not(arg2[k] in vpx) :       # add to the processing list if not already there

            plabl += [ lablx]

            if len(lablx) == 5 :  # rX_DX

                c0arg = rp[rx]  # power of r in the C^(0) basis

            elif len(lablx) == 8 :  # rX_hX_DX

                c0arg = rdh[rx][hx]  # note that rpw[rx=6] = -1

            elif len(lablx) == 10 :  # rX_proX_DX


                if   profid1 == 'eta':  ck1 = cd_eta
                elif profid1 == 'rho':  ck1 = cd_rho
                elif profid1 == 'drS':  ck1 = cd_ent
                elif profid1 == 'lho':  ck1 = cd_lho
                elif profid1 == 'buo':  ck1 = cd_buo
                elif profid1 == 'vsc':  ck1 = cd_vsc
                elif profid1 == 'eho':  ck1 = cd_eho
                elif profid1 == 'kho':  ck1 = cd_kho

                c0arg = ut.cheb2Product( rp[rx], ck1[:,dp1], tol)

            elif len(lablx) == 13 : # rX_hX_profX_DX

                if profid1 == 'lho':  ck1 = cd_lho

                c0arg = ut.cheb2Product( rdh[rx][hx], ck1[:,dp1], tol)

            elif len(lablx) == 15 :  # rX_proX_proX_DX

                # exec('ck1 = cd_%s' %profid1)
                # exec('ck2 = cd_%s' %profid2)
                if profid1 == 'kho' and profid2 == 'lnT':
                    ck1 = cd_kho
                    ck2 = cd_lnT
                if profid1 == 'vsc' and profid2 == 'lho':
                    ck1 = cd_vsc
                    ck2 = cd_lho
                if profid1 == 'eta' and profid2 == 'rho':
                    ck1 = cd_eta
                    ck2 = cd_rho

                c0arg = ut.cheb3Product( rp[rx], ck1[:,dp1], ck2[:,dp2], tol)

            parg0 += [ S[dx]*c0arg ]    # Gegenbauer basis change from C^(0) to C^(dx)
            parg1 += [ dx ]             # dx is derivative order
            parg2 += [ arg2[k] ]        # vector_parity

            del c0arg # To prevent re-use of c0arg in next iteration


    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Generate the Mlam matrices in parallel ----------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------------
    pool = mp.Pool( processes = int(ncpus) )
    tmp = [ pool.apply_async( ut.Mlam, args = ( parg0[k], parg1[k], parg2[k]) ) for k in range(np.size(parg0,0)) ]
    # recover resulting list of matrices
    matlist = [tmp1.get() for tmp1 in tmp]
    pool.close()
    pool.join()
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------------


    # Now we need to multiply the matrices on the right by the appropriate derivative matrix,
    # and change basis accordingly:
    for k,labl1 in enumerate(labl) :


        [ lablx, rx, hx, dx, secx, profid1, dp1, profid2, dp2 ] = ut.decode_label(labl1)

        gbx   = gebasis[section.index(secx)]  # order of the Gegenbauer basis according to the section, must be after decode_label!

        # Multiply by appropriate derivative matrix on the right and change to C^(4), C^(3) or C^(2) basis depending on section
        idx = [ j for j,x in enumerate(plabl) if ((x == lablx) and (parg2[j] == arg2[k])) ]  # find matrix index in plabl
        # and compute finally the full operator -------------------------------------------------------------------------------------------------
        matrix = G[gbx-1][gbx-dx] * matlist[idx[0]] * D[dx]
        # ---------------------------------------------------------------------------------------------------------------------------------------


        # If no inner core then remove unneeded rows and cols
        if par.ricb == 0 :

            if profid1 == 'drS' :  # this one just for the twozone or BVprof function, choose accordingly here!
                operator_parity = 1  # build the twozone or BVprof functions such that the *operator* parity is 1. Operator must be even

            elif hx is not None:
                if ut.symmB0 == -1:
                    operator_parity = (-1)**( hx + 1 + rpw[rx] + dx )  # h(r) is odd if B0 antisymmetric (axial, G21 dipole, or l=1 FDM)
                elif ut.symmB0 == 1:
                    operator_parity = (-1)**( hx + rpw[rx] + dx )

            elif len(lablx) == 5 :
                operator_parity = 1-((rx+dx)%2)*2

            elif len(lablx) == 10 and profid1 == 'eta':  # the eta profile must be an even funnction of r
                operator_parity = 1-(( rx + dp1 + dx )%2)*2

            ####
            # TO DO: assign operator_parity to operators with longer labels, i.e. involving 'eta' or 'rho'
            ####

            vector_parity   = arg2[k]
            overall_parity  = vector_parity * operator_parity
            matrix = ut.remroco( matrix, overall_parity, vector_parity)

        # Make room for boundary conditions and write to disk
        if par.ricb == 0 :
            chop = int(gbx/2)
        else :
            chop = gbx

        if chop > 0:
            matrix = ss.vstack( [ Z[chop-1], matrix[:-chop,:] ], format='csr' )

        sio.mmwrite( labl1, matrix )

    # -------------------------------------------------------------------------------------------------------------------------------------------

    toc = timer()
    print('Submatrices generated and written to disk in', toc-tic, 'seconds')

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
