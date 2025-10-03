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

    inviscid   = ((par.ViscosD == 0) and (par.ricb == 0))        # boolean

    tic = timer()
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

    tol = 1e-6
    # Chebyshev coefficients of powers of r
    r0  = ut.chebco(0, par.N, tol, par.ricb, ut.rcmb)
    r1  = ut.chebco(1, par.N, tol, par.ricb, ut.rcmb)
    r2  = ut.chebco(2, par.N, tol, par.ricb, ut.rcmb)
    r3  = ut.chebco(3, par.N, tol, par.ricb, ut.rcmb)
    r4  = ut.chebco(4, par.N, tol, par.ricb, ut.rcmb)
    r5  = ut.chebco(5, par.N, tol, par.ricb, ut.rcmb)
    r6  = ut.chebco(6, par.N, tol, par.ricb, ut.rcmb)

    rp = [r0, r1, r2, r3, r4, r5, r6]
 
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
    opkey = []

    if par.hydro == 1:
        # -------------------------------------------------------------------------------------------------------------------------------------------
        # Matrices needed for the Navier-Stokes equation, double curl equations ------------------------------------------- NavStok 2curl - section u
        # -------------------------------------------------------------------------------------------------------------------------------------------

        # inertia and Coriolis diag terms
        arg2 += [ vP ]*6
        labl  = [ 'u3_D0', 'u2_D1', 'u1_D2', 'u2lho1_D0', 'u1lho2_D0', 'u1lho1_D1' ]
        
        # Coriolis off diag terms
        arg2 += [   vT   ,   vT    ]
        labl += [ 'u2_D0', 'u1_D1' ]

        # Viscous diffusion
        if par.ViscosD > 0:
            arg2 += [ vP ]*29
            labl += [ 'u1moe0lho4_D0', 'u1moe1lho3_D0', 'u1moe2lho2_D0', 
                      'u2moe0lho3_D0', 'u2moe1lho2_D0', 'u2moe2lho1_D0',     
                      'u3moe0lho2_D0', 'u3moe1lho1_D0', 'u3moe2_D0', 
                      'u4moe0lho1_D0', 'u4moe1_D0'    , 'u5moe0_D0',

                      'u1moe0lho3_D1', 'u1moe1lho2_D1', 'u1moe2lho1_D1',
                      'u2moe0lho2_D1', 'u2moe1lho1_D1', 'u3moe0lho1_D1', 
                      'u3moe1_D1',

                      'u1moe0lho2_D2', 'u1moe1lho1_D2', 'u1moe2_D2',
                      'u2moe0lho1_D2', 'u2moe1_D2'    , 'u3moe0_D2',
                            
                      'u1moe0lho1_D3', 'u1moe1_D3'    , 'u2moe0_D3',
                        
                      'u1moe0_D4' ] 

        # -------------------------------------------------------------------------------------------------------------------------------------------
        # Matrices needed for the Navier-Stokes equation, single curl equations ------------------------------------------- NavStok 1curl - section v
        # -------------------------------------------------------------------------------------------------------------------------------------------

        # inertia and Coriolis diag terms
        arg2 += [   vT    ]
        labl += [ 'v1_D0' ]

        # Coriolis off diag
        arg2 += [   vP   ,     vP     ,   vP    ]
        labl += [ 'v2_D0', 'v1lho1_D0', 'v1_D1' ]

        # Viscous diffusion
        if par.ViscosD >0:
            arg2 += [ vT ]*5 
            labl += [ 'v2moe1_D0', 'v3moe0_D0',
                      'v1moe1_D1', 'v2moe0_D1',
                      'v1moe0_D2' ]

    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Pre-process the list with multiplication matrices labels to avoid duplicates --------------------------------------------------------------
    # Also generate the list of arguments parg0 and parg1 ---------------------------------------------------------------------------------------

    pkey  = []
    parg0 = []
    parg1 = []
    parg2 = []

    if par.ricb > 0:  # set vector_parity = 0, i.e. is not needed
        arg2 = np.size(labl)*[0]

    for k,labl1 in enumerate(labl) :

        (secx, rpower, rhopower, func1, dorder1, func2, dorder2, dx) = ut.decode_label(labl1)
        key1 = (rpower, rhopower, func1, dorder1, func2, dorder2, dx, arg2[k])  # reduced operator identifier
        opkey += [ key1 ]
        #print(key1)

        if not(key1 in pkey):  # if identifier not in the pkey list then we compute the matrix

            # Get the Chebishev coefficients c0arg for this operator
            c0arg = ut.gimmedachebs( labl1 )

            pkey  += [ key1 ]
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

        print(labl1)
        secx = labl1[0]
        #(rpower, rhopower, muorder, lhoorder, dx, vector_parity) = opkey[k]
        (rpower, rhopower, func1, dorder1, func2, dorder2, dx, vector_parity) = opkey[k]
        gbx = gebasis[section.index(secx)]  # order of the Gegenbauer basis according to the section

        # Multiply by appropriate derivative matrix on the right and change to C^(4), C^(3) or C^(2) basis depending on section
        idx = [ j for j,x in enumerate(pkey) if (x == opkey[k]) ]  # find matrix index in pkey
        matrix = G[gbx][gbx-dx] * matlist[idx[0]] * D[dx]
        # ---------------------------------------------------------------------------------------------------------------------------------------


        # If no solid inner core then remove unneeded rows and cols
        if par.ricb == 0 :

            #operator_parity = 1-(( rpower + (muorder or 0) + (lhoorder or 0) + dx )%2)*2  # we use 'or 0' to give 0 when muorder or lhoorder are None
            operator_parity = 1-(( rpower + (dorder1 or 0) + (dorder2 or 0) + dx )%2)*2  # we use 'or 0' to give 0 when dorder is None
            overall_parity  = vector_parity * operator_parity
            matrix = ut.remroco( matrix, overall_parity, vector_parity)

        # Make room for boundary conditions and write to disk
        if par.ricb == 0 :
            chop = int(gbx/2)
        else :
            chop = gbx

        if chop > 0:
            matrix = ss.vstack( [ Z[chop-1], matrix[:-chop,:] ], format='csr' )

        sio.mmwrite( labl1+'.mtx', matrix )

    # -------------------------------------------------------------------------------------------------------------------------------------------

    toc = timer()
    print('Submatrices generated and written to disk in', toc-tic, 'seconds')

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
