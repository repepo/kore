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
import os
from parameters import par
import utils as ut



def main(ncpus):

    warnings.simplefilter('ignore', ss.SparseEfficiencyWarning)

    inviscid   = ((par.ViscosD == 0) and (par.ricb == 0))        # boolean

    tic = timer()
    
    if int(ncpus) == 0:
        #cpuc = len(os.sched_getaffinity(0))
        cpuc = mp.cpu_count()
        print('cpu count =', cpuc)
        ncpus = cpuc

    print('N =', par.N,', lmax =', par.lmax)

    # vector parity for poloidal and toroidals. If ricb>0 we don't use parities, not needed.
    # if vP = 1 then we need only even parity Cheb polynomials
    # if vP = -1 we need only odd parity Chebs
    vP = int( (1 - 2*(par.m%2)) * par.symm )
    vT = -vP
    # for the magnetic field the parities are opposite to that of the flow if B0 is equatorially antisymmetric.
    #vF = ut.symmB0*vP
    #vG = -vF

    vS = vP  # this is the vector parity of the entropy perturbation

    tol = 1e-9
    
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
    G0    = [ 1 ]
    G     = [ G0, G1, G2, G3, G4 ]  # fixed this for the inviscid and no thermal diffusion cases

    # Sets the Gegenbauer basis order for each section
    if ((par.magnetic == 1) and ('conductor' in par.innercore)) :
        gebasis = [  4,   2,   3,   2,   2,   2  ]
    else:
        gebasis = [  4,   2,   2,   2,   2,   2  ]
    section     = [ 'u', 'v', 'f', 'g', 'h', 'i' ]

    if inviscid:
        gebasis[0] = 2  # only up to second derivatives in section u
        gebasis[1] = 1  # up to first derivatives in section v

    if par.ThermaD == 0:
        gebasis[4] = 0  # No thermal diffusion, C^(0) basis is enough, no need for thermal bc's

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

    labl  = []
    arg2  = []

    if par.hydro == 1:
        # -------------------------------------------------------------------------------------------------------------------------------------------
        # Matrix labels needed for the Navier-Stokes equation, double curl equations -------------------------------------- NavStok 2curl - section u
        # -------------------------------------------------------------------------------------------------------------------------------------------

        # inertia and Coriolis diag terms
        arg2 += [ vP ]*6
        labl  = [ 'u3_D0', 'u2_D1', 'u1_D2', 'u2lho1_D0', 'u1lho2_D0', 'u1lho1_D1' ]
        
        # Coriolis off diag terms
        arg2 += [   vT   ,   vT    ]
        labl += [ 'u2_D0', 'u1_D1' ]

        # Viscous diffusion
        if par.ViscosD > 0:
            arg2 += [ vP ]*44
            labl += [ 'u1lh13vsc0_D0', 'u1lh22vsc0_D0', 'u1lho4vsc0_D0', 'u2lh12vsc0_D0',
                      'u2lho3vsc0_D0', 'u3lh11vsc0_D0', 'u3lho2vsc0_D0', 'u4lho1vsc0_D0',
                          'u5vsc0_D0',     'u5vsc0_D0', 'u1lh12vsc1_D0', 'u1lho3vsc1_D0',
                      'u2lh11vsc1_D0', 'u2lho2vsc1_D0', 'u3lho1vsc1_D0', 'u3lho1vsc1_D0',
                          'u4vsc1_D0',     'u4vsc1_D0', 'u1lho2vsc2_D0', 'u2lho1vsc2_D0',
                          'u3vsc2_D0',     'u3vsc2_D0',
                      'u1lh12vsc0_D1', 'u1lho3vsc0_D1', 'u2lho2vsc0_D1', 'u3lho1vsc0_D1',
                      'u3lho1vsc0_D1', 'u1lh11vsc1_D1', 'u1lho2vsc1_D1', 'u2lho1vsc1_D1',
                          'u3vsc1_D1',     'u3vsc1_D1', 'u1lho1vsc2_D1',
                      'u1lh11vsc0_D2', 'u1lho2vsc0_D2', 'u2lho1vsc0_D2',     'u3vsc0_D2', 
                      'u1lho1vsc1_D2',     'u2vsc1_D2',     'u1vsc2_D2',
                      'u1lho1vsc0_D3',     'u2vsc0_D3',     'u1vsc1_D3',     'u1vsc0_D4'    ]

        # Buoyancy force
        if par.thermal == 1:
            arg2 += [ vP ]  # poloidal parity here because the entropy perturbation follows the same parity as the radial velocity
            labl += [ 'u2gra0_D0' ]

        # -------------------------------------------------------------------------------------------------------------------------------------------
        # Matrix labels needed for the Navier-Stokes equation, single curl equations -------------------------------------- NavStok 1curl - section v
        # -------------------------------------------------------------------------------------------------------------------------------------------

        # inertia and Coriolis diag terms
        arg2 += [   vT    ]
        labl += [ 'v1_D0' ]

        # Coriolis off diag
        arg2 += [   vP   ,     vP     ,   vP    ]
        labl += [ 'v2_D0', 'v1lho1_D0', 'v1_D1' ]

        # Viscous diffusion
        if par.ViscosD > 0:
            arg2 += [ vT ]*7
            labl += [ 'v2lho1vsc0_D0', 'v3vsc0_D0', 'v2vsc1_D0',
                      'v1lho1vsc0_D1', 'v2vsc0_D1', 'v1vsc1_D1',
                          'v1vsc0_D2' ]

    if par.thermal == 1:                  
        # -------------------------------------------------------------------------------------------------------------------------------------------
        # Matrix labels needed for the thermal equation ---------------------------------------------------------------------------- Heat - section h
        # -------------------------------------------------------------------------------------------------------------------------------------------

        # entropy perturbation
        arg2 += [ vP ]
        labl += [ 'h0pss0_D0' ]

        # thermal advection
        arg2 += [ vP ]
        labl += [ 'h1pdS0_D0' ]

        # thermal diffusion
        if par.ThermaD > 0:
            arg2 += [ vP ]*4
            labl += [ 'h2kps0_D0', 'h1kps0_D1', 'h0kps1_D1', 'h0kps0_D2' ]



    # -------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------- Generates the argument lists for Mlam
    # -------------------------------------------------------------------------------------------------------------------------------------------
    opkey = []  # all 'reduced' operator id list, might have duplicates
    pkey  = []  # unique 'reduced' operator id list
    parg0 = []  # derivative order
    parg1 = []  # Cheb coeffs go here
    parg2 = []  # vector_parity

    if par.ricb > 0:  # set vector_parity = 0, i.e. is not needed
        arg2 = np.size(labl)*[0]

    # This loop populates the lists pkey, parg1, parg2
    for k,labl1 in enumerate(labl) :
        (secx, rpower, rhopower, func1, dorder1, func2, dorder2, dx) = ut.decode_label(labl1)
        key1 = (rpower, rhopower, func1, dorder1, func2, dorder2, dx, arg2[k])  # 'reduced' operator identifier
        opkey += [ key1 ]
        if not(key1 in pkey):  # if identifier not in the pkey list then we compute the matrix
            pkey  += [ key1 ]
            parg1 += [ dx ]             # dx is derivative order
            parg2 += [ arg2[k] ]        # vector_parity

    # For each of the unique operator id's we generate in parallel the Chebyshev
    # coefficients, and change the Gegenbauer basis from C^(0) to C^(dx).
    # This populates the parg0 list. 
    pool1 = mp.Pool( processes = int(ncpus) )
    tmp = [ pool1.apply_async( ut.chegevara, args = ( pkey1, opkey, labl, S ) ) for pkey1 in pkey ]
    parg0 = [tmp1.get() for tmp1 in tmp]
    pool1.close()
    pool1.join()
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------------



    # -------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------- Generate the Mlam matrices in parallel
    # -------------------------------------------------------------------------------------------------------------------------------------------
    pool2 = mp.Pool( processes = int(ncpus) )
    tmp = [ pool2.apply_async( ut.Mlam, args = ( parg0[k], parg1[k], parg2[k]) ) for k in range(np.size(parg0,0)) ]
    # recover resulting list of matrices
    matlist = [tmp1.get() for tmp1 in tmp]
    pool2.close()
    pool2.join()
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------------



    # Now we need to multiply the matrices on the right by the appropriate derivative matrix,
    # and change basis accordingly:
    for k,labl1 in enumerate(labl) :
        #print(labl1)
        secx = labl1[0]
        (rpower, rhopower, func1, dorder1, func2, dorder2, dx, vector_parity) = opkey[k]
        gbx = gebasis[section.index(secx)]  # order of the Gegenbauer basis according to the section

        # Multiply by appropriate derivative matrix on the right and change to C^(4), C^(3) or C^(2) basis depending on section
        idx = [ j for j,x in enumerate(pkey) if (x == opkey[k]) ]  # find matrix index in pkey
        
        matrix = G[gbx][gbx-dx] * matlist[idx[0]] * D[dx]

        if par.ricb == 0 :  # --------------------------------------------------------- If no solid inner core then remove unneeded rows and cols

            adj = int(func1 in ['gra', 'pdS', 'lh1'])   # adjusts operator parity for these profiles 
            operator_parity = 1-(( rpower + (dorder1 or 0) + (dorder2 or 0) + dx + adj )%2)*2  # we use 'or 0' to give 0 when dorder is None
            #print(labl1, operator_parity, adj, rpower, vector_parity)
            overall_parity  = vector_parity * operator_parity
            matrix = ut.remroco( matrix, overall_parity, vector_parity)
            chop = int(gbx/2)

        else:
            
            chop = gbx
        # ---------------------------------------------------------------------------------------------------------------------------------------


        if chop > 0:  # ------------------------------------------------------------------- Makes room for boundary conditions and writes to disk
            matrix = ss.vstack( [ Z[chop-1], matrix[:-chop,:] ], format='csr' )
            #pass
        sio.mmwrite( labl1+'.mtx', matrix )
        # ---------------------------------------------------------------------------------------------------------------------------------------



    toc = timer()
    print('Generated and written', np.size(labl), 'operator submatrices in', toc-tic, 'seconds')

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
