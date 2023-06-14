#!/usr/bin/env python3
'''
kore generates submatrices

To use:
> ./bin/submatrices.py ncpus
where ncpus is the number of cores

This program generates in parallel the operators involved in the differential
equations as submatrices. These submatrices are required for the assembly
of the main matrices A and B.
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

def get_radial_derivatives(func,rorder,Dorder,tol):

    '''
    This function computes terms of the form r^n d^m/dr^m of a
    radial profile in Chebyshev space.

    Parameters
    ----------
    func   : function
        Radial profile in the form of a function (can be found in utils)
    rorder : integer
        Highest order of radial power
    Dorder : integer
        Highest order of radial derivative
    tol    : real
        Tolerance for Chebyshev transforms for radial powers

    Returns
    -------
    rd_prof : 2D list
        List such that rd_prof[i][j] defines the Chebyshev coefficients of
        r^i d^j/dr^j of the radial profile
    '''

    # Make sure these are integers
    rorder = int(rorder)
    Dorder = int(Dorder)

    rd_prof = [ [ [] for j in range(Dorder+1) ] for i in range(rorder+1) ] #List for Cheb coeffs to r^n D^m profile
    dnprof = [ [] for i in range(Dorder+1) ] #List for Cheb coeffs of nth derivative of profile
    # Cheb coeffs of profile
    dnprof[0] = ut.chebco_f( func, par.N, par.ricb, ut.rcmb, par.tol_tc )

    for i in range(rorder+1):
        rn  = ut.chebco(i, par.N, tol, par.ricb, ut.rcmb) #Cheb coeffs of r^i
        rd_prof[i][0] =  ut.chebProduct(dnprof[0],rn,par.N,par.tol_tc) #Cheb coeffs of r^i profile
        for j in range(1,Dorder+1):
        # Cheb coeffs of r^i D^j profile
            if i==0:
                # These only need to be computed once
                dnprof[j] = ut.Dcheb(dnprof[j-1],par.ricb,ut.rcmb)
            rd_prof[i][j] = ut.chebProduct(dnprof[j],rn,par.N,par.tol_tc)

    return rd_prof



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
    r1  = ut.chebco(1, par.N, tol, par.ricb, ut.rcmb)
    r2  = ut.chebco(2, par.N, tol, par.ricb, ut.rcmb)
    r3  = ut.chebco(3, par.N, tol, par.ricb, ut.rcmb)
    r4  = ut.chebco(4, par.N, tol, par.ricb, ut.rcmb)
    r5  = ut.chebco(5, par.N, tol, par.ricb, ut.rcmb)
    r6  = ut.chebco(6, par.N, tol, par.ricb, ut.rcmb)

    if twozone :
        r2z = ut.chebco_twozone(par.args, par.N, par.ricb, ut.rcmb, par.tol_tc)
        rp = [1, r1, r2, r3, r4, r5, r6, r2z]
    elif userdef :
        rbv = ut.chebco_BVprof(par.args, par.N, par.ricb, ut.rcmb, par.tol_tc)
        rp = [1, r1, r2, r3, r4, r5, r6, rbv]
    else :
        rp = [1, r1, r2, r3, r4, r5, r6]


    if par.anelastic:
        rd_rho = get_radial_derivatives(ut.log_density,4,4,tol) # Density : Requires derivatives and radial powers up to fourth order
        # rd_tem = get_radial_derivatives(ut.temperature,2,2,tol) # Temperature : Requires derivatives and radial powers up to second order
        # rd_buo = get_radial_derivatives(ut.buoFac,)


    if par.magnetic == 1 :

        rdh = [ [ [] for j in range(4) ] for i in range(7) ]
        rpw = [ 0, 1, 2, 3, 4, 5, -1]  # powers of r needed for the h function

        rd_eta = get_radial_derivatives(ut.mag_diffus,2,1,tol)  # Magnetic diffusivity profile

        cnorm = ut.B0_norm()  # Normalization

        for i,rpw1  in enumerate( rpw ):
            for j in range(4):
                args = [ par.beta, par.B0_l, par.ricb, rpw1, j ]
                rdh[i][j] = cnorm * ut.chebco_h( args, par.B0, par.N, 1, tol)
        # rdh is a 2D list where each element is the set of Chebyshev coeffs
        # of the function (r**rpw)*(d/dr)**j*(h_l(r))
        # in the last row the power of r is -1
        # columns are derivative order, first column (col 0) is for the function h itself


    # Basis transformations
    S0 = ut.Slam(0, par.N) # From Chebyshev basis ( C^(0) basis ) to C^(1) basis
    S1 = ut.Slam(1, par.N) # From C^(1) basis to C^(2) basis
    S2 = ut.Slam(2, par.N) # From C^(2) basis to C^(3) basis
    S3 = ut.Slam(3, par.N) # From C^(3) basis to C^(4) basis

    # Derivatives
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
            labl_u += [ 'r1_rho1_D0', 'r2_rho2_D0', 'r3_rho3_D0', 'r2_rho1_D1', 'r3_rho2_D1',
                        'r3_rho1_D2', 'r4_rho2_D2', 'r4_rho3_D1', 'r4_rho1_D3' ]

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
            arg2   += [     vP  ]
            labl_u += [ 'r4_D0' ]


        for labl1 in labl_u:

            if cdipole:  # increase the power of r by 2:
                old_rpow = labl1[:2]
                new_rpow = 'r'+str( int(old_rpow[1])+2 )
                labl1 = labl1.replace(old_rpow, new_rpow, 1)

            # append "_u" to all section u labels
            labl += [ labl1+'_u' ]

 
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
            labl_v += [ 'r1_rho1_D0', 'r2_rho2_D0', 'r2_rho1_D1' ]
    
        if par.magnetic == 1 :
            # Lorentz force
            labl_v += [ 'r0_h0_D1', 'r0_h1_D0', 'r1_h2_D0', 'r1_h0_D2', 'r0_h0_D0', 'r1_h1_D0', 'r1_h0_D1' ]
            arg2   += [     vF    ,     vF    ,     vF    ,     vF    ,     vG    ,     vG    ,     vG     ]
            

        for labl1 in labl_v:

            if cdipole:  # increase the power of r by 3:
                old_rpow = labl1[:2]
                new_rpow = 'r'+str( int(old_rpow[1])+3 )
                labl1 = labl1.replace(old_rpow, new_rpow, 1)

            # append "_v" to all section v labels
            labl += [ labl1+'_v' ]



    if par.magnetic == 1 :
        # -------------------------------------------------------------------------------------------------------------------------------------------
        # Matrices needed for the Induction equation, no-curl or consoidal equations -------------------------------------- Induct nocurl - section f
        # -------------------------------------------------------------------------------------------------------------------------------------------
        
        #if ((par.ricb > 0) and ('conductor' in par.innercore)) :  # conducting inner core, consoidal component, needs work, don't use!
        #    labl += [ 'f21',  'f32', 'f31',  'f20',   'f33',  'f22', 'f11', 'f00' ]
            
        # b
        labl_f  = [ 'r2_D0' ]
        arg2   += [     vF  ]
        
        # induction
        labl_f += [ 'r0_h0_D0', 'r1_h1_D0', 'r1_h0_D1', 'r1_h0_D0' ]
        arg2   += [     vP    ,     vP    ,     vP    ,     vT     ]
        
        # magnetic diffusion
        labl_f += [ 'r0_eta0_D0', 'r1_eta0_D1', 'r2_eta0_D2' ]  
        arg2   += [      vF     ,      vF     ,      vF      ]
        
        
        for labl1 in labl_f:

            if cdipole:  # increase the power of r by 2:
                old_rpow = labl1[:2]
                new_rpow = 'r'+str( int(old_rpow[1])+2 )
                labl1 = labl1.replace(old_rpow, new_rpow, 1)

            # append "_f" to all section f labels
            labl += [ labl1+'_f' ]
        

        # -------------------------------------------------------------------------------------------------------------------------------------------
        # Matrices needed for the Induction equation, single curl equations ------------------------------------------------ Induct 1curl - section g
        # -------------------------------------------------------------------------------------------------------------------------------------------

        if cdipole :
            # b
            labl += [ 'g50', 'g00' ]  # g00 needed in case of Lin2017 forcing
            # induction
            labl += [ 'g301', 'g411',  'g200', 'g310', 'g420', 'g402', 'g300', 'g401', 'g410' ]  # e.g. 'g200' is for (r**2)*h(r)
            # magnetic diffusion
            labl += [ 'g30', 'g41',  'g52' ]
        
            
        # b
        labl_g  = [ 'r2_D0' ]  # 
        arg2   += [    vG   ]
        
        # induction
        labl_g += [ 'r0_h0_D1', 'r1_h1_D1', 'r6_h0_D0', 'r0_h1_D0', 'r1_h2_D0', 'r1_h0_D2',
                    'r0_h0_D0', 'r1_h0_D1', 'r1_h1_D0' ]  # 'r6_h0_D0' is for (1/r)*h(r)
        arg2   += [     vP    ,     vP    ,     vP     ,    vP    ,     vP    ,     vP    ,
                        vT    ,     vT    ,     vT     ]
        
        # magnetic diffusion
        labl_g += [ 'r0_eta0_D0', 'r1_eta0_D1', 'r2_eta0_D2', 'r1_eta1_D0', 'r2_eta1_D1' ]
        arg2   += [      vG     ,      vG     ,      vG     ,      vG     ,      vG      ]
        

        for labl1 in labl_g:

            if cdipole:  # increase the power of r by 3:
                old_rpow = labl1[:2]
                new_rpow = 'r'+str( int(old_rpow[1])+3 )
                if new_rpow == 'r9':  # for (1/r)*h(r)
                    new_rpow = 'r2'
                labl1 = labl1.replace(old_rpow, new_rpow, 1)

            # append "_g" to all section g labels
            labl += [ labl1+'_g' ]
        
        


    if par.thermal == 1 :
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Matrices needed for the heat equation ------------------------------------------------------------------------------------ Heat - section h
    # -------------------------------------------------------------------------------------------------------------------------------------------

        if par.heating == 'differential' :

            labl_h = [ 'r0_D0', 'r1_D0', 'r2_D1',  'r3_D2', 'r3_D0' ]  # here the operators have mixed symmetries!?, this is potentially a problem when ricb == 0
            arg2  += [     vP ,     vP ,     vP ,      vP ,     vP  ]

        elif (par.heating == 'internal' or (twozone or userdef) ) :

            labl_h = [ 'r2_D0', 'r0_D0', 'r1_D1', 'r2_D2' ]
            if not cdipole:
                arg2 += [  vP ,     vP ,     vP ,     vP  ]

            if (twozone or userdef) :

                labl_h += [ 'r7_D0' ]  # this is for ut.twozone or ut.BVprof
                if not cdipole:
                    arg2 += [   vP  ]
                
                
        for labl1 in labl_h:
            # append "_h" to all section h labels
            labl += [ labl1+'_h' ]           
                    
                    
                    

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


        for labl1 in labl_i:
            # append "_i" to all section i labels
            labl += [ labl1+'_i' ]  




    # -------------------------------------------------------------------------------------------------------------------------------------------




    # Pre-process the list with multiplication matrices labels to avoid duplicates --------------------------------------------------------------
    # Also generate the list of arguments parg0 and parg1
    plabl = []
    parg0 = []
    parg1 = []
    parg2 = []

    if cdipole or (par.ricb > 0) :  # set vector_parity = 0, i.e. is not needed
        arg2 = np.size(labl)*[0]
    #argnopar = np.size(labl)*[0]


    for k,labl1 in enumerate(labl) :

        lablx = labl1[1:]  # strip the leading character in label (representing the section), lablx is 2 or 3 digits long

        idx = [ j for j,x in enumerate(plabl) if x == lablx ]  # find indices of same labels as lablx in plabl
        vpx = [ parg2[i] for i in idx ]  # find the vector_parities of those

        if not(arg2[k] in vpx) :       # add to the processing list if not already there

            # if lablx has two digits then the first one is the power of r, last one is the derivative order
            # if lablx has three digits then the middle one is the derivative order of the function h(r), first and second digits as above
            if len(lablx) == 6 :
                rx = int(lablx[3])
            else:
                rx = int(lablx[0])
            dx = int(lablx[-1])

            if len(lablx) == 2 and rx > 0 :

                plabl += [ lablx ]          # labels in plabl are two or three digit strings, we process here the two digit ones only
                parg0 += [ S[dx]*rp[rx] ]   # power of r in the C^(dx) basis
                parg1 += [ dx ]             # dx is derivative order
                parg2 += [ arg2[k] ]        # vector_parity

            if len(lablx) == 3 :

                hx = int(lablx[1])
                plabl += [ lablx ]
                parg0 += [ S[dx] * rdh[rx][hx] ]
                parg1 += [ dx ]
                parg2 += [ arg2[k] ]

            if len(lablx) == 6 :

                prof_id = lablx[:3]  # this describes which radial profile is needed

                if prof_id == 'eta':
                    rprof = rd_eta
                if par.anelastic:
                    if prof_id == 'rho':
                        rprof = rd_rho

                profx = int(lablx[4])
                plabl += [ lablx ]
                parg0 += [ S[dx] * rprof[rx][profx] ]
                parg1 += [ dx ]
                parg2 += [ arg2[k] ]


    # Now generate the matrices in parallel -----------------------------------------------------------------------------------------------------
    pool = mp.Pool( processes = int(ncpus) )
    tmp = [ pool.apply_async( ut.Mlam, args = ( parg0[k], parg1[k], parg2[k]) ) for k in range(np.size(parg0,0)) ]
    #tmp = [ pool.apply_async( ut.Mlam, args = ( parg0[k], parg1[k], argnopar[k]) ) for k in range(np.size(parg0,0)) ]

    # recover results
    matlist = [tmp1.get() for tmp1 in tmp]
    pool.close()
    pool.join()
    # -------------------------------------------------------------------------------------------------------------------------------------------


    # finishing steps
    for k,labl1 in enumerate(labl) :

        lablx = labl1[1:]
        secx = labl1[0]  # the section
        if len(lablx) == 6 :
            rx = int(lablx[3])
        else:
            rx = int(lablx[0])
        dx   = int(lablx[-1])
        gbx  = gebasis[section.index(secx)]

        # Multiply by appropriate derivative operator on the right and change to C^(4), C^(3) or C^(2) basis depending on section
        if lablx == '00' :  # this one is the identity, just changes basis
            matrix = G[gbx-1][gbx-dx]
        else :
            idx = [ j for j,x in enumerate(plabl) if ((x == lablx) and (parg2[j] == arg2[k])) ]  # find matrix index in plabl
            matrix = G[gbx-1][gbx-dx] * matlist[idx[0]] * D[dx]

        # If no inner core then remove unneeded rows and cols
        if par.ricb == 0 :

            if rx == 7 :  # this one just for the twozone or BVprof function, choose accordingly here!
                operator_parity = 1  # build the twozone or BVprof functions such that the *operator* parity is 1. Operator must be even

            elif len(lablx) == 3 :
                hx = int(lablx[1])
                if ut.symmB0 == -1:
                    operator_parity = (-1)**( hx + 1 + rpw[rx] + dx )  # h(r) is odd if B0 antisymmetric (axial, G21 dipole, or l=1 FDM)
                elif ut.symmB0 == 1:
                    operator_parity = (-1)**( hx + rpw[rx] + dx )

            elif len(lablx) == 2 :
                operator_parity = 1-((rx+dx)%2)*2

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
    print('Blocks generated and written to disk in', toc-tic, 'seconds')

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
