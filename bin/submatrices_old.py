#!/usr/bin/env python3
'''
kore generates submatrices

To use:
> ./bin/submatrices.py ncpus
where ncpus is the number of cores

This program generates the operators involved in the differential
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


def main(ncpus):
    
    warnings.simplefilter('ignore', ss.SparseEfficiencyWarning)
    
    twozone = ((par.thermal == 1) and (par.heating == 'two zone'))  # boolean
    userdef = ((par.thermal == 1) and (par.heating == 'user defined'))  # boolean

    tic = timer()
    print('N =', par.N,', lmax =', par.lmax)
    
    # vector parity for poloidal and toroidals
    vP = int( (1 - 2*(par.m%2)) * par.symm )
    vT = -vP
    # for magnetic field
    vF = ut.symmB0*vP
    #vF = vP
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
        r2z = ut.chebco_twozone(par.args, par.N, par.ricb, ut.rcmb, 1e-6)
        rp = [1, r1, r2, r3, r4, r5, r6, r2z]
    elif userdef :
        rbv = ut.chebco_BVprof(par.args, par.N, par.ricb, ut.rcmb, 1e-5)
        rp = [1, r1, r2, r3, r4, r5, r6, rbv]
    else :
        rp = [1, r1, r2, r3, r4, r5, r6]
    
    
    if par.magnetic == 1 :
        
        rdh = [ [ [] for j in range(4) ] for i in range(7) ]
        rpw = [ 0, 1, 2, 3, 4, 5, -1]  # powers of r needed for the h function
        
        for i,rpw1  in enumerate( rpw ):
            for j in range(4):
        
                args = [ par.beta, par.B0_l, par.ricb, rpw1, j ]
                rdh[i][j] = ut.chebco_h( args, par.B0, par.N, 1, tol)
                
        # rh is a 2D list where each element is the set of Chebyshev coeffs
        # of the function (r**rpw)*(d/dr)***(h_l(r))
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
    G     = [ G2, G3, G4 ]
        
    # Sets the Gegenbauer basis order for each section 
    if ((par.magnetic == 1) and ('conductor' in par.innercore)) :
        gebasis = [  4,   2,   3,   2,   2  ]
    else:
        gebasis = [  4,   2,   2,   2,   2  ]  
    section     = [ 'u', 'v', 'f', 'g', 'h' ]
    
    # Used when making room for bc's
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
    
    
    
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Matrices needed for the Navier-Stokes equation, double curl equations ------------------------------------------- NavStok 2curl - section u 
    # -------------------------------------------------------------------------------------------------------------------------------------------
    
    # u
    if ((par.magnetic == 1) and (par.B0 == 'dipole') and (par.ricb > 0)) :
        labl = [ 'u40', 'u51', 'u62' ]
        arg2 = [   vP ,   vP ,   vP  ]
    else:
        labl = [ 'u20', 'u31', 'u42' ]
        arg2 = [   vP ,   vP ,   vP  ]
    
    # Coriolis
    if ((par.magnetic == 1) and (par.B0 == 'dipole') and (par.ricb > 0)) :
        labl += [ 'u40', 'u51', 'u62', 'u50', 'u61' ]
        arg2 += [   vP ,   vP ,   vP ,   vT ,   vT  ]
    else:
        labl += [ 'u20', 'u31', 'u42', 'u30', 'u41' ]
        arg2 += [   vP ,   vP ,   vP ,   vT ,   vT  ]
    
    # Viscous diffusion
    if ((par.magnetic == 1) and (par.B0 == 'dipole') and (par.ricb > 0)) :
        labl += [ 'u20', 'u42', 'u53', 'u64' ]
        arg2 += [   vP ,   vP ,   vP ,   vP  ]    
    else:
        labl += [ 'u00', 'u22', 'u33', 'u44' ]
        arg2 += [   vP ,   vP ,   vP ,   vP  ]
    
        
    if par.magnetic == 1 :
        
        if par.B0 in ['axial', 'G21 dipole', 'FDM'] :
        
            #labl += [ 'u31', 'u44', 'u30', 'u41', 'u10', 'u21', 'u32', 'u43', 'u00' ]
            #arg2 += [   vP ,   vP ,   vT ,   vT ,   vF ,   vF ,   vF ,   vF ,   vP  ]
            
            # Lorentz force
            labl += [ 'u101', 'u211', 'u202', 'u312', 'u000', 'u110', 'u220', 'u330', 'u303', 'u100', 'u210', 'u201', 'u311', 'u302' ] 
            arg2 += [   vF  ,   vF  ,   vF  ,   vF  ,   vF  ,   vF  ,   vF  ,   vF  ,   vF  ,   vG  ,   vG  ,   vG  ,   vG  ,   vG   ]  # vF for bpol, vG for btor


        elif ((par.B0 == 'dipole') and (par.ricb > 0)) :
        
            #labl += [ 'u40', 'u51', 'u53', 'u64', 'u50', 'u61', 'u11', 'u10', 'u21', 'u32', 'u62', 'u00' ]
            labl += [ 'u301', 'u411', 'u402', 'u512', 'u200', 'u310', 'u420', 'u530', 'u503', 'u300', 'u410', 'u401', 'u511', 'u502' ] 
            arg2 += [   vF  ,   vF  ,   vF  ,   vF  ,   vF  ,   vF  ,   vF  ,   vF  ,   vF  ,   vG  ,   vG  ,   vG  ,   vG  ,   vG   ]  # vF for bpol, vG for btor
            
            
    if par.thermal == 1 :
        
        if ((par.magnetic == 0) or ((par.magnetic == 1) and (par.B0 == 'axial'))) :
        
            labl += [ 'u40' ]
            arg2 += [   vP  ]
            
        elif ((par.magnetic == 1) and (par.B0 == 'dipole') and (par.ricb > 0)) :
        
            labl += [ 'u60' ]



    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Matrices needed for the Navier-Stokes equation, single curl equations ------------------------------------------- NavStok 1curl - section v
    # -------------------------------------------------------------------------------------------------------------------------------------------
    
    if ((par.magnetic == 1) and (par.B0 == 'dipole') and (par.ricb > 0)) :
        #       [   u    corio  corio  vdiff  vdiff  vdiff ]
        labl += [ 'v50', 'v40', 'v51', 'v30', 'v41', 'v52' ]  
        arg2 += [   vT ,   vP ,   vP ,   vT ,   vT ,   vT  ]
    else:
        #       [   u    corio  corio  vdiff  vdiff  vdiff ]
        labl += [ 'v20', 'v10', 'v21', 'v00', 'v11', 'v22' ]  
        arg2 += [   vT ,   vP ,   vP ,   vT ,   vT ,   vT  ]
    
    
    if par.magnetic == 1 :
        
        if par.B0 in ['axial', 'G21 dipole', 'FDM'] :
            
            # Lorentz force
            labl += [ 'v001', 'v010', 'v120', 'v102', 'v000', 'v110', 'v101' ]
            arg2 += [   vF  ,   vF  ,   vF  ,   vF  ,   vG  ,   vG  ,   vG   ]
    
        elif par.B0 == 'dipole' and par.ricb > 0 :
    
            #labl += [ 'v30', 'v40', 'v41', 'v50', 'v51',  'v52' ]   
            labl += [ 'v301', 'v310', 'v420', 'v402', 'v300', 'v410', 'v401' ]
            arg2 += [   vF  ,   vF  ,   vF  ,   vF  ,   vG  ,   vG  ,   vG   ]


    if par.magnetic == 1 :  
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Matrices needed for the Induction equation, no-curl or consoidal equations -------------------------------------- Induct nocurl - section f
    # -------------------------------------------------------------------------------------------------------------------------------------------
    
        if par.B0 in ['axial', 'G21 dipole', 'FDM'] :
        
            if ((par.ricb > 0) and ('conductor' in par.innercore)) :
                
                # Needs fixing!
                labl += [ 'f21',  'f32', 'f31',  'f20',   'f33',  'f22', 'f11', 'f00' ]  # for consoidal component          
                
            else :
                
                #  b 
                labl += [ 'f20' ]
                arg2 += [   vF  ]
                
                # induction
                labl += [ 'f000', 'f110', 'f101', 'f100' ]
                arg2 += [   vP  ,   vP  ,   vP  ,   vT   ]
                
                # magnetic diffusion
                labl += [ 'f00', 'f11', 'f22' ]
                arg2 += [   vF ,   vF ,   vF  ]
                
                #labl += [ 'f10', 'f21', 'f20', 'f00', 'f11', 'f22', 'f20' ]  # for no-curl radial component
                #arg2 += [   vP ,   vP ,   vT ,   vF ,   vF ,   vF ,   vF  ]
        
        
        elif ((par.B0 == 'dipole') and (par.ricb > 0)) :  # as above but times r^2
            
            #labl += [ 'f11', 'f00', 'f10', 'f20', 'f31', 'f42', 'f40' ]

            #  b 
            labl += [ 'f40' ]
            arg2 += [   vF  ]
            
            # induction
            labl += [ 'f200', 'f310', 'f301', 'f300' ]
            arg2 += [   vP  ,   vP  ,   vP  ,   vT   ]
            
            # magnetic diffusion
            labl += [ 'f20', 'f31', 'f42' ]
            arg2 += [   vF ,   vF ,   vF  ]
        

    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Matrices needed for the Induction equation, single curl equations (section g) ------------------------------------------------ Induct 1curl 
    # -------------------------------------------------------------------------------------------------------------------------------------------

        if par.B0 in ['axial', 'G21 dipole', 'FDM'] :

            # b
            labl += [ 'g20' ]
            arg2 += [   vG  ]
            
            # induction
            labl += [ 'g001', 'g111',  'g600', 'g010', 'g120', 'g102', 'g000', 'g101', 'g110' ]  # 'g600' is for (1/r)*h(r)
            arg2 += [   vP  ,   vP  ,    vP  ,   vP  ,   vP  ,   vP  ,   vT  ,   vT  ,   vT   ]    

            # magnetic diffusion
            labl += [ 'g00', 'g11',  'g22' ]
            arg2 += [   vG ,   vG ,    vG  ]   

            #labl += [ 'g00', 'g11',  'g22', 'g10', 'g21', 'g20' ]
            #arg2 += [   vP ,   vP ,    vP ,   vT ,   vT ,   vG  ]   
        
        elif ((par.B0 == 'dipole') and (par.ricb > 0)) :  # as above but times r^3
            
            #labl += [ 'g00', 'g11',  'g22', 'g10', 'g21', 'g30', 'g41',  'g52', 'g50' ]

            # b
            labl += [ 'g50' ]
            arg2 += [   vG  ]
            
            # induction
            labl += [ 'g301', 'g411',  'g200', 'g310', 'g420', 'g402', 'g300', 'g401', 'g410' ]  # 'g200' is for (r**2)*h(r)
            arg2 += [   vP  ,   vP  ,    vP  ,   vP  ,   vP  ,   vP  ,   vT  ,   vT  ,   vT   ]    

            # magnetic diffusion
            labl += [ 'g30', 'g41',  'g52' ]
            arg2 += [   vG ,   vG ,    vG  ]   



        

    if par.thermal == 1 :
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Matrices needed for the heat equation (section h) ------------------------------------------------------------------------------------ Heat 
    # -------------------------------------------------------------------------------------------------------------------------------------------
        
        if par.heating == 'differential' :
            
            labl += [ 'h00', 'h10', 'h21',  'h32' ] 
            arg2 += [   vP ,   vP ,   vP ,    vP  ]

        elif (par.heating == 'internal' or (twozone or userdef) ) :
            
            labl += [ 'h20', 'h00', 'h11',  'h22' ]
            arg2 += [   vP ,   vP ,   vP ,    vP  ]
            
            if (twozone or userdef) :
                
                labl += [ 'h70' ]
                arg2 += [   vP  ]
            
    # -------------------------------------------------------------------------------------------------------------------------------------------




    # Pre-process the list with multiplication matrices labels to avoid duplicates --------------------------------------------------------------
    # Also generate the list of arguments parg0 and parg1
    plabl = []
    parg0 = []
    parg1 = []
    parg2 = []
    
    if par.ricb > 0 :  # set vector_parity = 0, i.e. is not needed
        arg2 = np.size(labl)*[0]
    argcero = np.size(labl)*[0]
    
    
    for k,labl1 in enumerate(labl) :
        
        lablx = labl1[1:]  # strip the leading character in label (representing the section), lablx is 2 or 3 digits long
        idx = [ j for j,x in enumerate(plabl) if x == lablx]  # find indices of same labels as lablx in plabl
        vpx = [parg2[i] for i in idx]  # find the vector_parities of those
        
        if not(arg2[k] in vpx) :       # add to the processing list if not already there                                      
        
            # lablx has two or three digits, first one of the power of r, last one is the derivative order
            # if three digits present then the middle one is the derivative order of the function h
            rx = int(lablx[ 0])
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
                
                # print('label = ', plabl[-1])
                # print('parg1 = ', parg1[-1])
                # print('parg2 = ', parg2[-1])
                # print('parg0 = ', parg0[-1])
    
    #print(np.size(plabl),np.shape(parg0))
    
    # Now generate the matrices in parallel -----------------------------------------------------------------------------------------------------
    pool = mp.Pool( processes = int(ncpus) )
    tmp = [ pool.apply_async( ut.Mlam, args = ( parg0[k], parg1[k], parg2[k]) ) for k in range(np.size(parg0,0)) ]
    #tmp = [ pool.apply_async( ut.Mlam, args = ( parg0[k], parg1[k], argcero[k]) ) for k in range(np.size(parg0,0)) ]
    
    # recover results
    matlist = [tmp1.get() for tmp1 in tmp]
    pool.close()
    pool.join()
    # -------------------------------------------------------------------------------------------------------------------------------------------

    # finishing steps
    for k,labl1 in enumerate(labl) :
        
        lablx = labl1[1:]
        secx = labl1[0]  
        rx   = int(lablx[0])
        dx   = int(lablx[-1])
        gbx  = gebasis[section.index(secx)]
        
        #print(labl1,'rx =',rx)
        
        
        # Multiply by appropriate derivative operator on the right and change to C^(4), C^(3) or C^(2) basis depending on section
        if lablx == '00' :  # this one is the identity, just changes basis
            matrix = G[gbx-2][gbx-dx] 
        else :
            idx = [ j for j,x in enumerate(plabl) if ((x == lablx) and (parg2[j] == arg2[k])) ]  # find matrix index in plabl
            matrix = G[gbx-2][gbx-dx] * matlist[idx[0]] * D[dx] 

        # If no inner core then remove unneeded rows and cols
        if par.ricb == 0 :
            
            if rx == 7 :  # this one just for the twozone or BVprof function, choose accordingly here!
                operator_parity = 1  # build the twozone or BVprof functions such that the *operator* parity is 1. Odd parity (-1) does not seem to work properly.
            
            elif len(lablx) == 3 :
                hx = int(lablx[1])
                operator_parity = (-1)**( hx + rpw[rx] + dx )
            
            elif len(lablx) == 2 :
                operator_parity = 1-((rx+dx)%2)*2
            
            vector_parity   = arg2[k]
            overall_parity  = vector_parity * operator_parity
            
            matrix = ut.remroco( matrix, overall_parity, vector_parity)

        # Make room for boundary conditions and write to disk
        #if par.ricb == 0 and (secx != 'h' ):
        if par.ricb == 0 :
            chop = int(gbx/2)
        else :
            chop = gbx
        #print(chop)
        #print(np.shape(Z[chop-1]),np.shape(matrix[:-chop,:]))
        matrix = ss.vstack( [ Z[chop-1], matrix[:-chop,:] ], format='csr' )
        sio.mmwrite( labl1, matrix )
        
    # -------------------------------------------------------------------------------------------------------------------------------------------
        
    toc = timer()
    print('Blocks generated and written to disk in', toc-tic, 'seconds')

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
