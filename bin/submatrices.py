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
import diff_rot as dr



def main(ncpus):

    warnings.simplefilter('ignore', ss.SparseEfficiencyWarning)

    par.check_names()

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

    vS = vP  # this is the vector parity of the entropy perturbation
    # NEW (2026-10-28): the magnetic field has the opposite parities when B0 is equatorially antisymmetric
    vF = ut.symmB0*vP
    vG = -vF

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
    gebasis = [  4,   2,   2,   2,   2  ]  # NEW (2026-10-28): f, g (magnetic field) in C^(2), 2 bc rows each, as conductive_IC
    section = [ 'u', 'v', 'h', 'f', 'g' ]

    if inviscid:
        gebasis[0] = 2  # only up to second derivatives in section u
        gebasis[1] = 1  # up to first derivatives in section v
        if par.magnetic:   # NEW (2026-10-28): the Lorentz force has third derivatives of f in section u and second ones of g in
            gebasis[0] = 3  # section v; section v still has no bc rows (chop = 0 below), section u one (P = 0)
            gebasis[1] = 2

    if par.ThermaD == 0:
        gebasis[2] = 1 if ut.adiab_tw else 0  # No thermal diffusion, C^(0) basis is enough (C^(1) for the thermal wind's P'), no need for thermal bc's

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

        if par.ThermaD > 0:
            # entropy perturbation rʰ p s and thermal advection -rʰ p uᵣ dS/dr
            arg2 += [ vP ]*2
            labl += [ 'h0pss0_D0', 'h1pdS0_D0' ]
        else:
            # no diffusion: p cancels, so the heat equation is used divided by p: r s and -r uᵣ dS/dr
            # (p ~ 1e-7 near the surface makes the p-weighted rows ill-conditioned there); r² s and -r² uᵣ dS/dr with the thermal wind (ut.adiab_tw)
            arg2 += [ vP ]*2
            labl += [ 'h0_D0', 'h1dSd0_D0' ]

        # thermal diffusion
        if par.ThermaD > 0:
            arg2 += [ vP ]*4
            labl += [ 'h2kps0_D0', 'h1kps0_D1', 'h0kps1_D1', 'h0kps0_D2' ]

    if par.magnetic == 1:
        # -------------------------------------------------------------------------------------------------------------------------------------------
        # NEW (2026-10-28): magnetic field, B0 of degree 1 ------------------------------------------------- Lorentz force (u, v), induction (f, g)
        # -------------------------------------------------------------------------------------------------------------------------------------------
        # 'bpo' with order k is cnorm h^(k) (background_field.py). 'ohr1' lowers the ρ power by one: the Lorentz force per
        # unit mass has a 1/ρ, so its rows are r^p (not r^p ρ) times the force. The conductive_IC labels are given in the
        # comments ('u101': r¹ h D1 in its r⁴-weighted u rows, here u(5-1) since rpower_u = 5; v: 3 - p; f, g: 2 - p).

        # Lorentz force, section u: bpol columns ...
        arg2 += [ vF ]*9
        labl += [ 'u4bpo0ohr1_D1', 'u3bpo1ohr1_D1', 'u3bpo0ohr1_D2', 'u2bpo1ohr1_D2', 'u5bpo0ohr1_D0',   # u101 u211 u202 u312 u000
                  'u4bpo1ohr1_D0', 'u3bpo2ohr1_D0', 'u2bpo3ohr1_D0', 'u2bpo0ohr1_D3' ]                   # u110 u220 u330 u303
        # ... and btor columns
        arg2 += [ vG ]*5
        labl += [ 'u4bpo0ohr1_D0', 'u3bpo1ohr1_D0', 'u3bpo0ohr1_D1', 'u2bpo1ohr1_D1', 'u2bpo0ohr1_D2' ]  # u100 u210 u201 u311 u302

        # NEW (2026-10-28): variable ρ0, the λ1 (∇₁⋅𝐆_H) term of the u rows (labels with lho1 vanish for structure 'Boussinesq')
        arg2 += [ vF ]*5
        labl += [ 'u2bpo0lho1ohr1_D2', 'u3bpo0lho1ohr1_D1', 'u4bpo0lho1ohr1_D0', 'u2bpo2lho1ohr1_D0', 'u3bpo1lho1ohr1_D0' ]
        arg2 += [ vG ]*3
        labl += [ 'u3bpo0lho1ohr1_D0', 'u2bpo1lho1ohr1_D0', 'u2bpo0lho1ohr1_D1' ]

        # Lorentz force, section v: bpol, btor columns
        arg2 += [ vF ]*4
        labl += [ 'v3bpo0ohr1_D1', 'v3bpo1ohr1_D0', 'v2bpo2ohr1_D0', 'v2bpo0ohr1_D2' ]  # v001 v010 v120 v102
        arg2 += [ vG ]*3
        labl += [ 'v3bpo0ohr1_D0', 'v2bpo1ohr1_D0', 'v2bpo0ohr1_D1' ]                   # v000 v110 v101
        if ut.bf.B0_l == 2:
            # NEW (2026-10-28): poloidal B0 of degree 2 (operators.lorentz_pol2, induction_pol2), the labels it adds to those above
            arg2 += [ vF, vG ]
            labl += [ 'u2bpo2ohr1_D1', 'u2bpo2ohr1_D0' ]

        # induction equation, section f (r² r̂⋅): b, induction from P and T, magnetic diffusion
        arg2 += [ vF ]
        labl += [ 'f0_D0' ]                                             # f20
        arg2 += [ vP ]*3 + [ vT ]
        labl += [ 'f2bpo0_D0', 'f1bpo1_D0', 'f1bpo0_D1', 'f1bpo0_D0' ]  # f000 f110 f101 | f100
        arg2 += [ vP ]
        labl += [ 'f1bpo0lho1_D0' ]                                     # NEW (2026-10-28): variable ρ0
        arg2 += [ vF ]*3
        labl += [ 'f2eta0_D0', 'f1eta0_D1', 'f0eta0_D2' ]               # feta000 feta101 feta202

        # induction equation, section g (r² r̂⋅∇×): b, induction from P and T, magnetic diffusion
        arg2 += [ vG ]
        labl += [ 'g0_D0' ]                                             # g20
        arg2 += [ vP ]*6 + [ vT ]*3
        labl += [ 'g2bpo0_D1', 'g1bpo1_D1', 'g3bpo0_D0', 'g2bpo1_D0', 'g1bpo2_D0', 'g1bpo0_D2',   # g001 g111 g600 g010 g120 g102
                  'g2bpo0_D0', 'g1bpo0_D1', 'g1bpo1_D0' ]                                       # g000 g101 g110
        arg2 += [ vP ]*4
        labl += [ 'g1bpo0lho1_D1', 'g1bpo0lho2_D0', 'g1bpo1lho1_D0', 'g2bpo0lho1_D0' ]   # NEW (2026-10-28): variable ρ0
        arg2 += [ vG ]*5
        labl += [ 'g2eta0_D0', 'g1eta0_D1', 'g0eta0_D2', 'g1eta1_D0', 'g0eta1_D1' ]  # geta000 geta101 geta202 geta110 geta211

        if par.B0_tor is not None:
            # NEW (2026-10-28): toroidal B0 of degree 1, 'bto' = cnorm_tor k^(j) (operators.lorentz_tor, induction_tor; derived in
            # tests/magnetic/derive.py). Without a poloidal B0 the labels above are zero matrices (cnorm = 0).
            arg2 += [ vP ]*4 + [ vT ]
            labl += [ 'f1bto0_D0', 'g2bto0_D0', 'g1bto0lho1_D0', 'g1bto1_D0', 'g1bto0_D0' ]
            arg2 += [ vP ]
            labl += [ 'g1bto0_D1' ]   # degree 2 only
            arg2 += [ vF ]*9
            labl += [ 'u4bto0ohr1_D0', 'u3bto0lho1ohr1_D0', 'u3bto1ohr1_D0', 'u2bto1lho1ohr1_D0', 'u2bto2ohr1_D0',
                      'u3bto0ohr1_D1', 'u2bto0lho1ohr1_D1', 'u2bto1ohr1_D1', 'u2bto0ohr1_D2' ]
            arg2 += [ vG ]*4
            labl += [ 'u3bto0ohr1_D0', 'u2bto0lho1ohr1_D0', 'u2bto1ohr1_D0', 'u2bto0ohr1_D1' ]
            arg2 += [ vF ]*3 + [ vG ]
            labl += [ 'v3bto0ohr1_D0', 'v2bto1ohr1_D0', 'v2bto0ohr1_D1', 'v2bto0ohr1_D0' ]

        if ut.mag_algebra:
            # NEW (2026-10-30): the couplings of the Lorentz force and of the induction from the recurrence algebra
            # (mag_algebra.py, any background degrees) in place of the closed forms: their labels replace those with bpo, bto
            import mag_algebra as ma
            keep = [ i for i, lab in enumerate(labl) if 'bpo' not in lab and 'bto' not in lab ]
            labl, arg2 = [ labl[i] for i in keep ], [ arg2[i] for i in keep ]
            for lab, X in ma.labels():
                labl += [ lab ]
                arg2 += [ {'P': vP, 'T': vT, 'F': vF, 'G': vG}[X] ]

    if par.diff_rot == 1:
        # Differential rotation (diff_rot.py): the radial operators of the force of the background flow (sections u, v) and of
        # the heat-equation terms (section h, with thermal = 1), each with the vector parity of its column (vP for P and s, vT for T)
        dr.check()
        for lab, kind in dr.labels():
            labl += [ lab ]
            arg2 += [ {'P': vP, 'T': vT, 'F': vF, 'G': vG}[kind] ]   # NEW (2026-10-30): F, G for the induction by the flow

    # -------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------- Generates the argument lists for Mlam
    # -------------------------------------------------------------------------------------------------------------------------------------------
    opkey = []  # all 'reduced' operator id list, might have duplicates
    pkey  = []  # unique 'reduced' operator id list
    parg0 = []  # derivative order
    parg1 = []  # Cheb coeffs go here
    parg2 = []  # vector_parity
    parg3 = []  # parity of the profile coefficients (used only when ricb=0), same rule as for remroco below

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
            adj = int(func1 in ut.odd_profiles) + int(func2 in ut.odd_profiles)   # odd profiles (g, p dS/dr, ..., the σn'); FIX (2026-10-28): both slots
            parg3 += [ 1-(( rpower + (dorder1 or 0) + (dorder2 or 0) + adj )%2)*2 ]

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
    if par.ricb == 0 and par.diff_rot == 1 and (par.diff_rot_type == 'Y20-wall-bounded' or
            (par.diff_rot_type == 'shellular' and not (par.diff_rot_param % 2 == 0 and par.diff_rot_param > 0))):
        # ΔΩ is used at |r| (diff_rot.fX): (1 - r) r and r**σ - 1 have a kink at r = 0
        print('WARNING: diff_rot_type', par.diff_rot_type, 'is not smooth at r = 0, the full sphere (ricb = 0) converges slowly in N')
    if par.ricb == 0:  # profiles must have the parity assumed above, otherwise only part of them is kept
        for k,pkey1 in enumerate(pkey):
            c = np.abs(parg0[k])
            cwrong = c[int((1+parg3[k])/2)::2]
            # (absolute floor: a profile that vanishes identically, e.g. S_tw of the 'cylindrical' rotation, is all round-off)
            if np.max(c) > 0 and np.max(cwrong) > max(1e-8*np.max(c), 1e-13):
                print('WARNING: profile coefficients of', labl[opkey.index(pkey1)], 'do not have the assumed parity:',
                      'max wrong-parity/max = {:.2e}. Only the part with the assumed parity is used.'.format(np.max(cwrong)/np.max(c)))
    tmp = [ pool2.apply_async( ut.Mlam, args = ( parg0[k], parg1[k], parg2[k], parg3[k]) ) for k in range(np.size(parg0,0)) ]
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
        
        if dx > gbx:  # G[gbx][gbx-dx] would index from the end of the list: a wrong basis, silently
            raise ValueError('%s: derivative order %d needs at least the C^(%d) basis, section %s has C^(%d)' % (labl1, dx, dx, secx, gbx))
        matrix = G[gbx][gbx-dx] * matlist[idx[0]] * D[dx]

        if par.ricb == 0 :  # --------------------------------------------------------- If no solid inner core then remove unneeded rows and cols

            adj = int(func1 in ut.odd_profiles) + int(func2 in ut.odd_profiles)   # odd profiles, both slots (FIX 2026-10-28)
            operator_parity = 1-(( rpower + (dorder1 or 0) + (dorder2 or 0) + dx + adj )%2)*2  # we use 'or 0' to give 0 when dorder is None
            #print(labl1, operator_parity, adj, rpower, vector_parity)
            overall_parity  = vector_parity * operator_parity
            matrix = ut.remroco( matrix, overall_parity, vector_parity)
            chop = int(gbx/2)

        else:
            
            chop = gbx

        if secx == 'h' and par.ThermaD == 0:  # no thermal bc rows without diffusion, whatever the basis
            chop = 0
        if secx == 'v' and inviscid:  # NEW (2026-10-28): no bc rows in section v without viscosity, whatever the basis (C^(2) when magnetic)
            chop = 0
        # ---------------------------------------------------------------------------------------------------------------------------------------


        if chop > 0:  # ------------------------------------------------------------------- Makes room for boundary conditions and writes to disk
            matrix = ss.vstack( [ Z[chop-1], matrix[:-chop,:] ], format='csr' )
            #pass
        if par.drop_tol > 0:  # drop entries below drop_tol times the largest of this submatrix
            matrix = ss.csr_matrix(matrix)
            if matrix.nnz > 0:
                matrix.data[np.abs(matrix.data) < par.drop_tol*np.abs(matrix.data).max()] = 0
                matrix.eliminate_zeros()
        sio.mmwrite( labl1+'.mtx', matrix )
        # ---------------------------------------------------------------------------------------------------------------------------------------



    if ut.cic:
        # -------------------------------------------------------------------------------------------------------------------------------------------
        # NEW (2026-10-28): the field inside a conducting inner core (sections fic, gic), ported from conductive_IC's submatrices.py -----------------
        # -------------------------------------------------------------------------------------------------------------------------------------------
        # Chebyshev series on [-ricb, ricb], one parity per section (vF for f, vG for g), rows r² r̂⋅ (fic) and r² r̂⋅∇× (gic) of
        # ∂b/∂t = ∇²b (times the diffusivity) in the C^(2) basis; one bc row per block after the parity reduction. These are
        # not labels (decode_label and burrito are for the fluid shell): the names below are loaded by operators.py as any .mtx.
        Nic = par.N_cic
        R1, R2 = 0, par.ricb
        r1ic = ut.chebco_f( lambda r: r,    Nic, R1, R2, 1e-15 )  # Chebyshev coefficients of r and r² on [-ricb, ricb]
        r2ic = ut.chebco_f( lambda r: r**2, Nic, R1, R2, 1e-15 )
        S0ic = ut.Slam(0, Nic)
        S1ic = ut.Slam(1, Nic)
        D1ic = ut.Dlam(1, Nic, R1, R2)
        D2ic = ut.Dlam(2, Nic, R1, R2)
        M20 = ut.Mlam(               r2ic, 0, 0)  # ×r² in the C^(0) basis
        M11 = ut.Mlam(        S0ic * r1ic, 1, 0)  # ×r  in the C^(1) basis
        M22 = ut.Mlam( S1ic * S0ic * r2ic, 2, 0)  # ×r² in the C^(2) basis
        ops = { 'r0_D0': S1ic * S0ic, 'r2_D0': S1ic * S0ic * M20, 'r1_D1': S1ic * M11 * D1ic, 'r2_D2': M22 * D2ic }
        zic2 = ss.csr_matrix((2, Nic))
        for name, mat in ops.items():
            mat = ss.vstack( [ zic2, mat[:-2,:] ], format='csr' )  # room for the bc rows at the top
            sio.mmwrite( name + 'f_ic.mtx', ut.remroco( mat, vF, vF ) )  # operator parity 1: one bc row is left
            sio.mmwrite( name + 'g_ic.mtx', ut.remroco( mat, vG, vG ) )

    toc = timer()
    print('Generated and written', np.size(labl), 'operator submatrices in', toc-tic, 'seconds')
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
