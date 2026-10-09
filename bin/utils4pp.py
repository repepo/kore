# The diagnose function in this module invokes in parallel the functions
# flow_worker and thermal_worker.
# It provides energy, dissipation, power, etc for a given solution.

import multiprocessing as mp
import numpy.polynomial.chebyshev as ch
import numpy as np
from parameters import par
import utils as ut
import radial_profiles as rap

if par.diff_rot : import diff_rot_coefficients as dr

mp.set_start_method('fork')



def xcheb(r, ricb, rcmb):
    # returns points in the appropriate domain of the Cheb polynomial solutions
    # Domain [-1,1] corresponds to [ ricb,rcmb] if ricb>0
    # Domain [-1,1] corresponds to [-rcmb,rcmb] if ricb==0

    r1 = rcmb
    r0 = ricb + (np.sign(ricb)-1)*rcmb  # r0=-rcmb if ricb==0; r0=ricb if ricmb>0 
    out = 2*(r-r0)/(r1-r0) - 1

    return out



def funcheb(ck0, r, ricb, rcmb, n):
    '''
    Returns the function represented by the Chebyshev coeffs ck0, evaluated at the radii r.
    If r is None then uses the rk (i.e. the associated x0) points defined globally.
    First column is the function itself, second column is its derivative with respect to r,
    and so on up to the n-th derivative. Rows correspond to the radial points.
    Use this only when the Cheb coeffs are the full set, i.e. after using expand_sol if ricb=0.
    '''

    if np.sum(r==None)==1:
            x00 = x0  # use the globally defined x0
    else:
        x00 = xcheb(r, ricb, rcmb)  # use the explicit radial points given as argument
    
    out = np.zeros((np.size(x00), n+1), ck0.dtype)  # n+1 cols
    out[:,0] = ch.chebval(x00, ck0)  # the function itself
    
    ck0 = ut.ironit(ck0, 0)

    if n>0:
        dk = ut.Dn_cheb(ck0, ricb, rcmb, n)  # coeffs for the derivatives, n cols
        for j in range(1,n+1):
            out[:,j] = ch.chebval(x00, dk[:,j-1])  # and the derivatives

    return out









def rad_quad(f, Ra, Rb, wk):
    '''
    Computes the radial integral of f over [Ra,Rb] as sum(wk*f)*(Rb-Ra)/2.
    Assumes f is sampled at the nodes xk in [-1,1] (mapped to [Ra,Rb]) whose quadrature weights are wk.
    diagnose() uses Gauss-Legendre nodes and weights (exact for polynomials of degree < 2*Nq).
    '''

    out = np.sum( wk * f ) * (Rb-Ra)/2

    return out



def expand_sol(sol,vsymm):
    '''
    Expands the ricb=0 solution with ut.N1 coeffs to have full N coeffs,
    filling with zeros according to the equatorial symmetry.
    vsymm=-1 for equatorially antisymmetric, vsymm=1 for symmetric
    '''
 
    if par.ricb == 0 :

        lm1 = par.lmax-par.m+1

        # separate poloidal and toroidal coeffs
        P0 = sol[ 0    : ut.n   ]
        T0 = sol[ ut.n : 2*ut.n ]

        # these are the cheb coefficients, reorganized
        Plj0 = np.reshape(P0,(int(lm1/2),ut.N1))
        Tlj0 = np.reshape(T0,(int(lm1/2),ut.N1))

        # create new arrays
        Plj = np.zeros((int(lm1/2),par.N),dtype=complex)
        Tlj = np.zeros((int(lm1/2),par.N),dtype=complex)

        # assign according to symmetry
        s = int( (vsymm+1)/2 )  # s=0 if vsymm=-1, s=1 if vsymm=1
        iP = (par.m + 1 - s)%2  # even/odd Cheb polynomial for poloidals according to the parity of m+1-s
        iT = (par.m + s)%2
        for k in np.arange(int(lm1/2)) :
            Plj[k,iP::2] = Plj0[k,:]
            Tlj[k,iT::2] = Tlj0[k,:]

        # rebuild solution vector
        P2 = np.ravel(Plj)
        T2 = np.ravel(Tlj)
        out = np.r_[P2,T2]

    else :
        out = sol

    return out



def expand_reshape_sol(sol, vsymm):
    '''
    Expands the ricb=0 solution with ut.N1 coeffs to have full N coeffs,
    filling with zeros according to the equatorial symmetry.
    vsymm=-1 for equatorially antisymmetric, vsymm=1 for symmetric
    Returns a list with two 2D arrays, for poloidal and toroidal coeffs.
    rows for l, columns for Cheb order
    '''
 
    lm1 = par.lmax-par.m+1
    
    scalar_field = np.size(sol) == ut.n  # sol is a scalar field if true

    # separate poloidal and toroidal coeffs
    P0 = sol[ 0    : ut.n   ]
    if not scalar_field:
        T0 = sol[ ut.n : 2*ut.n ]
    
    if par.ricb==0:  # need to expand from N1 to N coeffs

        Plj0 = np.reshape(P0,(int(lm1/2),ut.N1))
        if not scalar_field:
            Tlj0 = np.reshape(T0,(int(lm1/2),ut.N1))

        # create new expanded arrays
        Plj = np.zeros((int(lm1/2),par.N),dtype=complex)
        if not scalar_field:
            Tlj = np.zeros((int(lm1/2),par.N),dtype=complex)

        # assign elements according to symmetry
        s = int( (vsymm+1)/2 )  # s=0 if vsymm=-1, s=1 if vsymm=1
        iP = (par.m + 1 - s)%2  # even/odd Cheb polynomial for poloidals according to the parity of m+1-s
        iT = (par.m + s)%2
        for k in np.arange(int(lm1/2)) :
            Plj[k,iP::2] = Plj0[k,:]
            if not scalar_field:	
                Tlj[k,iT::2] = Tlj0[k,:]

    else:  # No need to expand, just reshape
        
        Plj = np.reshape(P0,(int(lm1/2),par.N))
        if not scalar_field:
            Tlj = np.reshape(T0,(int(lm1/2),par.N))

    if not scalar_field:
        out = [Plj, Tlj]
    else:
        out = Plj
        
    return out



def cheb2space_pol(l, lp, P, ns, radii):
    '''
    Returns qlm's (radial) and slm's (consoidal) L-components up to derivatives of order ns (<=3)
    '''

    if np.sum(radii==None)==1:
        rr = rk
    else:
        rr = radii

    rr2 = rr**2
    rr3 = rr**3
    rr4 = rr**4
    rr5 = rr**5

    # rho = rap.densityX(rr, 1)

    # lho1 = rap.lhoX(rr,1)/rho[:,0]
    # lho2 = rap.lhoX(rr,2)/rho[:,0]**2
    # lho3 = rap.lhoX(rr,3)/rho[:,0]**3
    # lho4 = rap.lhoX(rr,4)/rho[:,0]**4

    L     = l*(l+1)
    idx   = list(lp).index(l) 
    f_pol = funcheb(P[idx,:], r=radii, ricb=par.ricb, rcmb=ut.rcmb, n=ns+1)
    
    plm0 = f_pol[:,0]
    plm1 = f_pol[:,1]
    
    qlm = []
    slm = []
    
    qlm0  = (L*plm0)/rr
    qlm.append(qlm0)

    slm0 = plm1 + plm0*(lho1 + 1/rr)
    slm.append(slm0)

    if ns>0:

        plm2 = f_pol[:,2]

        qlm1 = (L*(-plm0 + plm1*rr))/rr2
        qlm.append(qlm1)

        slm1 = plm2 + plm0*(lho2 - (1/rr2)) + plm1*(lho1 + 1/rr)
        slm.append(slm1)

    if ns>1:

        plm3 = f_pol[:,3]

        qlm2 = (L*(2*plm0 + rr*(-2*plm1 + plm2*rr)))/rr3
        qlm.append(qlm2)

        slm2 = plm3 + plm0*(lho3 + 2/rr3) + 2*plm1*(lho2 - (1/rr2)) + plm2*(lho1 + 1/rr)
        slm.append(slm2)

    if ns>2:

        plm4 = f_pol[:,4]
    
        qlm3 = (L*(-6*plm0 + rr*(6*plm1 + rr*(-3*plm2 + plm3*rr))))/rr4
        qlm.append(qlm3)

        slm3 = plm4 + plm0*(lho4 - 6/rr4) + 3*plm1*(lho3 + 2/rr3) + 3*plm2*(lho2 - 1/rr2) + plm3*(lho1 + 1/rr)
        slm.append(slm3)

    if ns>3:

        plm5 = f_pol[:,5]

        qlm4 = (24*L*plm0)/rr5 - (24*L*plm1)/rr4 + (12*L*plm2)/rr3 - (4*L*plm3)/rr2 + (L*plm4)/rr
        qlm.append(qlm4)

        slm4 = plm5 + plm0*(lho5 + 24/rr5) + 4*plm1*(lho4 - 6/rr4) + 6*plm2*(lho3 + 2/rr3) + 4*plm3*(lho2 - 1/rr2) + plm4*(lho1 + 1/rr)
        slm.append(slm4)

    return [qlm, slm]






def cheb2space_tor(L, lt, T, ns, radii):
    '''
    Returns tlm's (toroidal) L-components up to derivatives of order ns (<=3)
    '''

    idx   = list(lt).index(L) 
    f_tor = funcheb(T[idx,:], r=radii, ricb=par.ricb, rcmb=ut.rcmb, n=ns)

    tlm = []
    for i in range(ns+1):
        tlm.append(f_tor[:,i])

    return tlm






























def curl(l, qlm, slm, tlm):
    '''
    Returns the l-component of the curl of the input vector.
    Needs also the first and second radial derivative of the input vector.
    '''
    L = l*(l+1.)

    if len(qlm) == 2:
        [qlm0, qlm1] = qlm
        [slm0, slm1] = slm
        [tlm0, tlm1] = tlm

    elif len(qlm) == 3:
        [qlm0, qlm1, qlm2] = qlm
        [slm0, slm1, slm2] = slm
        [tlm0, tlm1, tlm2] = tlm

    out_rad0 = L*tlm0/rk
    out_con0 = tlm1 + tlm0/rk
    out_tor0 = (qlm0-slm0)/rk - slm1

    if len(qlm) == 2:
        out_rad = [out_rad0]
        out_con = [out_con0]
        out_tor = [out_tor0]

    elif len(qlm) == 3:
        out_rad1 = L*tlm1/rk - L*tlm0/r2
        out_con1 = tlm2 + tlm1/rk - tlm0/r2
        out_tor1 = (qlm1-slm1)/rk - (qlm0-slm0)/r2 - slm2
        out_rad = [out_rad0, out_rad1]
        out_con = [out_con0, out_con1]
        out_tor = [out_tor0, out_tor1]

    return [ out_rad, out_con, out_tor ]



def curl2(l, qlm, slm, tlm):
    '''
    Returns the l-component of the curl of the curl (yes, twice) of the input vector.
    Needs the first and second radial derivative of the input vector.
    '''
    L = l*(l+1.)

    [ [qlm0,qlm1], [slm0,slm1], [tlm0,tlm1] ] = curl(l, qlm, slm, tlm)

    out_rad = L*tlm0/rk
    out_con = tlm1 + tlm0/rk
    out_tor = (qlm0-slm0)/rk - slm1

    return [ out_rad, out_con, out_tor ]



    


def dotprod_pol(l, qlma, slma, qlmb, slmb):
    '''
    Returns the integrand to compute the volume integral of the dot product of two vector fields, poloidal l-component
    '''
    f0 = 4*np.pi/(2*l+1)
    f1 = r2 * 2 * np.real( qlma * np.conj( qlmb ) )
    f2 = r2 * l*(l+1) * 2 * np.real( slma * np.conj( slmb ) )
    return f0*(f1+f2)



def dotprod_tor(l, tlma, tlmb):
    '''
    Returns the integrand to compute the volume integral of the dot product of two vector fields, toroidal l-component
    '''
    f0 = 4*np.pi/(2*l+1)
    f1 = r2 * l*(l+1) * 2 * np.real( tlma * np.conj( tlmb ) )
    return f0*f1



def cdot_pol(l, qlma, slma, qlmb, slmb):
    '''
    Complex version of dotprod_pol: integrand of the volume integral of conj(a)⋅b, poloidal l-component.
    dotprod_pol is 2*Re of this.
    '''
    f0 = 4*np.pi/(2*l+1)
    f1 = r2 * np.conj( qlma ) * qlmb
    f2 = r2 * l*(l+1) * np.conj( slma ) * slmb
    return f0*(f1+f2)



def cdot_tor(l, tlma, tlmb):
    '''
    Complex version of dotprod_tor: integrand of the volume integral of conj(a)⋅b, toroidal l-component.
    dotprod_tor is 2*Re of this.
    '''
    f0 = 4*np.pi/(2*l+1)
    f1 = r2 * l*(l+1) * np.conj( tlma ) * tlmb
    return f0*f1









def dotprod_scal(l, hlma, hlmb):
    '''
    Returns the integrand to compute the volume integral of the product of two scalar fields, l-component.
    Same convention as dotprod_pol and dotprod_tor: 2 Re(a b*), with ∫|Yₗᵐ|² dΩ = 4π/(2l+1).
    '''
    f0 = 4*np.pi/(2*l+1)
    f1 = r2 * 2 * np.real( hlma * np.conj( hlmb ) )
    return f0*f1



def entropy_energy(l, hlm0):
    '''
    Returns the integrand to compute the entropy "energy" ½ ∫ p₀ s² dV, l-component
    '''
    return 0.5 * pss0 * dotprod_scal(l, hlm0, hlm0)



def entropy_advect(l, hlm0, qlm0):
    '''
    Returns the integrand to compute the entropy advection term -∫ p₀ (dS/dr) uᵣ s dV, l-component.
    qlm0 is the radial velocity uᵣ = L P/r.
    '''
    return -pdS0 * dotprod_scal(l, qlm0, hlm0)



def entropy_dissip(l, hlm0, hlm1):
    '''
    Returns the integrand to compute -∫ κ p₀ ∇s⋅∇s dV, l-component.
    ∫ s ∇⋅(κ p₀ ∇s) dV is this plus the surface term [ r² κ p₀ s ∂s/∂r ] (added in thermal_worker).
    '''
    L = l*(l+1)
    f0 = 4*np.pi/(2*l+1)
    f1 = r2 * np.abs( hlm1 )**2 + L * np.abs( hlm0 )**2
    return -kps0 * f0 * 2 * f1



def diff_rot_power(l, lp, lt, P, T):
    '''
    Returns the integrand to compute the volume integral of ∫ ( F ⋅ 𝐮 ) dV, l-component
    with F = (𝐮 ⋅ ∇Ω) r sin(θ) eϕ;  the differential rotation interaction between the perturbation and the base flow. 
    
    [Inefficient implementation]
    '''
    
    wdr = 0j

    m = par.m
    L = l*(l+1)
    f0 = (4*np.pi/(2*l+1))*(np.sqrt(L/2))*r2
    f1 = 0
    f2 = 0
    f4 = 0
    f5 = 0
    f6 = 0

    aub = rap.aubX(rk, 0)
    svp = rap.svpX(rk, 0)
    spv = rap.spvX(rk, 0)

    slm0 = 0
    tlm0 = 0
    qlm0_dl = 0
    slm0_dl = 0
    tlm0_dl = 0

    if l in lp : 
        [ [ _ ], [ slm0 ] ] = cheb2space_pol(l, lp, P, 0)
    elif l in lt : 
        [ tlm0 ] = cheb2space_tor(l, lt, T, 0)

    for offdiag in [-3, -2, -1, 0, 1, 2, 3]: 
        if l+offdiag in lp : 
            [ [ qlm0_dl ], [ slm0_dl ] ] = cheb2space_pol(l+offdiag, lp, P, 0)
        elif l+offdiag in lt : 
            [ tlm0_dl ] = cheb2space_tor(l+offdiag, lt, T, 0)

        c1 = (svp * dr.w_SQ_20(l, m, offdiag)) + (spv * dr.w_SQ_00(l, m, offdiag))
        c2 = aub * dr.w_SS_20(l, m, offdiag)
        c3 = aub * dr.w_ST_20(l, m, offdiag)
        c4 = (svp * dr.w_TQ_20(l, m, offdiag)) + (spv * dr.w_TQ_00(l, m, offdiag))
        c5 = aub * dr.w_TS_20(l, m, offdiag)
        c6 = aub * dr.w_TT_20(l, m, offdiag)

        f1 = c1 * (np.conj(slm0) * qlm0_dl) # conj(S_l)*Q_dl
        f2 = c2 * (np.conj(slm0) * slm0_dl) # conj(S_l)*S_dl
        f3 = c3 * (np.conj(slm0) * tlm0_dl) # conj(S_l)*T_dl
        f4 = c4 * (np.conj(tlm0) * qlm0_dl) # conj(T_l)*Q_dl
        f5 = c5 * (np.conj(tlm0) * slm0_dl) # conj(T_l)*S_dl
        f6 = c6 * (np.conj(tlm0) * tlm0_dl) # conj(T_l)*T_dl

        wdr += f0 * (f1 + f2 + f3 + f4 + f5 + f6)   

    # for offdiag in [-3, -2, -1, 0, 1, 2, 3]:
    #     if l in lp : 
    #         # S_l, Q_l != 0, T_l = 0
    #         [ [ _ ], [ slm0 ] ] = cheb2space_pol(l, lp, P, 0)
    #         if l + offdiag in lp : 
    #             # S_dl, Q_dl != 0, T_dl = 0 -> Non-zero terms : conj(S_l)*Q_dl, conj(S_l)*S_dl
    #             [ [ qlm0_dl ], [ slm0_dl ] ] = cheb2space_pol(l+offdiag, lp, P, 0)

    #             c1 = (svp * dr.w_SQ_20(l, m, offdiag)) + (spv * dr.w_SQ_00(l, m, offdiag))
    #             c2 = aub * dr.w_SS_20(l, m, offdiag)

    #             f1 = c1 * (np.conj(slm0) * qlm0_dl) # conj(S_l)*Q_dl
    #             f2 = c2 * (np.conj(slm0) * slm0_dl) # conj(S_l)*S_dl
    #         elif l + offdiag in lt :
    #             # S_dl, Q_dl = 0, T_dl != 0 -> Non-zero terms : conj(S_l)*T_dl
    #             [ tlm0_dl ] = cheb2space_tor(l+offdiag, lt, T, 0)

    #             c1 = aub * dr.w_ST_20(l, m, offdiag)

    #             f1 = c1 * (np.conj(slm0) * tlm0_dl) # conj(S_l)*T_dl

    #     elif l in lt : 
    #         # S_l, Q_l = 0, T_l != 0
    #         [ tlm0 ] = cheb2space_tor(l, lt, T, 0)
    #         if l + offdiag in lp : 
    #             # S_dl, Q_dl != 0, T_dl = 0 -> Non-zero terms : conj(T_l)*Q_dl, conj(T_l)*S_dl
    #             [ [ qlm0_dl ], [ slm0_dl ] ] = cheb2space_pol(l+offdiag, lp, P, 0)

    #             c1 = (svp * dr.w_TQ_20(l, m, offdiag)) + (spv * dr.w_TQ_00(l, m, offdiag))
    #             c2 = aub * dr.w_TS_20(l, m, offdiag)

    #             f1 = c1 * (np.conj(tlm0) * qlm0_dl) # conj(T_l)*Q_dl
    #             f2 = c2 * (np.conj(tlm0) * slm0_dl) # conj(T_l)*S_dl
    #         elif l + offdiag in lt :
    #             # S_dl, Q_dl = 0, T_dl != 0 -> Non-zero terms : conj(T_l)*T_dl
    #             [ tlm0_dl ] = cheb2space_tor(l+offdiag, lt, T, 0)

    #             c1 = aub * dr.w_TT_20(l, m, offdiag)

    #             f1 = c1 * (np.conj(tlm0) * tlm0_dl) # conj(T_l)*T_dl

    #     wdr += f0 * (f1 + f2)

    return 2*np.real(wdr)

def flow_worker( l ):
    '''
    Computes the power balance from the momentum (the Navier-Stokes) equation, l-component.
    Includes the kinetic energy, the kinetic energy dissipation, the rate of working (power) of the
    buoyancy force, the enstrophy budget terms, the differential rotation power, and the imaginary parts
    of the Coriolis, viscous and buoyancy powers for the frequency balance.
    '''

    Ra = par.ricb
    Rb = ut.rcmb

    kinep = np.zeros_like( rk, dtype='complex64')
    kinet = np.zeros_like( rk, dtype='complex64')
    kindp = np.zeros_like( rk, dtype='complex64')
    kindt = np.zeros_like( rk, dtype='complex64')
    enstro_vel_p = np.zeros_like( rk, dtype='complex64')
    enstro_vel_t = np.zeros_like( rk, dtype='complex64')
    enstro_cor_p = np.zeros_like( rk, dtype='complex64')
    enstro_cor_t = np.zeros_like( rk, dtype='complex64')
    enstro_vif_p = np.zeros_like( rk, dtype='complex64')
    enstro_vif_t = np.zeros_like( rk, dtype='complex64')
    enstro_buo_p = np.zeros_like( rk, dtype='complex64')
    enstro_buo_t = np.zeros_like( rk, dtype='complex64')

    
    wdr = 0

    [   velq,   vels,   velt ] = velocity(l)     # velocity 𝐮
    [ cuvelq, cuvels, cuvelt ] = curl(l, velq, vels, velt)  # its curl ∇×𝐮

    [   corq,   cors,   cort ] = coriolis(l)     # Coriolis force 𝐳×𝐮
    [ cucorq, cucors, cucort ] = curl(l, corq, cors, cort)  # its curl ∇×(𝐳×𝐮)

    if par.ViscosD>0:
        [   vifq,   vifs,   vift ] = visforce(l)     # viscous force divided by the density (∇⋅𝛔)/ρ
        [ cuvifq, cuvifs, cuvift ] = curl(l, vifq, vifs, vift)  # its curl ∇×((∇⋅𝛔)/ρ)
    else:
        [   vifq,   vifs,   vift ] = [0,0,0]
        [ cuvifq, cuvifs, cuvift ] = [0,0,0]


    if par.thermal:
        [   buoq,   buos,   buot ] = buoyancy(l)     # buoyancy force
        [ cubuoq, cubuos, cubuot ] = curl(l, buoq, buos, buot)  # its curl


    # (∇×𝐮)⋅(∇×𝐮)
    enstro_vel_p = dotprod_pol(l, cuvelq[0], cuvels[0], cuvelq[0], cuvels[0] ) * const0**2
    enstro_vel_t = dotprod_tor(l, cuvelt[0], cuvelt[0] ) * const0**2

    # (∇×𝐮)⋅(∇×(𝐳×𝐮))
    enstro_cor_p = dotprod_pol(l, cuvelq[0], cuvels[0], cucorq[0], cucors[0] ) * const0**2
    enstro_cor_t = dotprod_tor(l, cuvelt[0], cucort[0] ) * const0**2

    # (∇×𝐮)⋅(∇×((∇⋅𝛔)/ρ))
    if par.ViscosD>0:
        enstro_vif_p = dotprod_pol(l, cuvelq[0], cuvels[0], cuvifq[0], cuvifs[0] ) * const0**2
        enstro_vif_t = dotprod_tor(l, cuvelt[0], cuvift[0] ) * const0**2
    else:
        [ enstro_vif_p, enstro_vif_t ] = [0,0]

    if par.thermal:
        enstro_buo_p = dotprod_pol(l, cuvelq[0], cuvels[0], cubuoq[0], cubuos[0] ) * const0**2
        enstro_buo_t = dotprod_tor(l, cuvelt[0], cubuot[0] ) * const0**2


    # Kinetic energy ½ρ𝐮⋅𝐮
    kinep = 0.5*dotprod_pol(l, velq[0], vels[0], velq[0], vels[0] )*rho0
    kinet = 0.5*dotprod_tor(l, velt[0], velt[0])*rho0

    # kinetic energy dissipation ρ𝐮⋅((∇⋅𝛔)/ρ) = 𝐮⋅(∇⋅𝛔) aka power of viscous force
    if par.ViscosD>0:
        kindp = dotprod_pol(l, velq[0], vels[0], vifq[0], vifs[0])*rho0
        kindt = dotprod_tor(l, velt[0], vift[0])*rho0
    else:
        kindp = 0
        kindt = 0

    # rate of working of the buoyancy force ρ𝐮⋅(Beyonce g s 𝐫̂), the buoyancy force per unit mass as in operators.buoyancy
    if par.thermal:
        wbuop = dotprod_pol(l, velq[0], vels[0], buoq[0], buos[0])*rho0
    else:
        wbuop = 0

    # Complex powers 2 ∫ ρ 𝐮*⋅𝐅 dV (their real parts are Dkin and Wthm above) for the frequency balance:
    # the imaginary part of λ ∫ ρ |𝐮|² dV = ∫ ρ 𝐮*⋅( -2𝐳×𝐮 + (∇⋅𝛔)/ρ + buoyancy ) dV. The Coriolis power is
    # purely imaginary; the viscous and buoyancy powers are real only when their operators are Hermitian.
    pcor = 2*( cdot_pol(l, velq[0], vels[0], corq[0], cors[0]) + cdot_tor(l, velt[0], cort[0]) )*rho0
    if par.ViscosD>0:
        pvif = 2*( cdot_pol(l, velq[0], vels[0], vifq[0], vifs[0]) + cdot_tor(l, velt[0], vift[0]) )*rho0
    else:
        pvif = 0
    if par.thermal:
        pbuo = 2*cdot_pol(l, velq[0], vels[0], buoq[0], buos[0])*rho0
    else:
        pbuo = 0


    # Integrals
    Kene_l = rad_quad( kinep + kinet, Ra, Rb, wk)  # ∫ ½ ρ 𝐮⋅𝐮 dV 
    Dkin_l = rad_quad( kindp + kindt, Ra, Rb, wk)  # ∫ 𝐮⋅(∇⋅𝛔) dV
    Wthm_l = rad_quad( wbuop, Ra, Rb, wk )         # ∫ ρ 𝐮⋅(Beyonce g s 𝐫̂) dV

    Wcor_im_l = np.imag( rad_quad( pcor, Ra, Rb, wk) )  # 2 Im ∫ ρ 𝐮*⋅(2𝐳×𝐮) dV
    Dkin_im_l = np.imag( rad_quad( pvif, Ra, Rb, wk) )  # 2 Im ∫ 𝐮*⋅(∇⋅𝛔) dV
    Wthm_im_l = np.imag( rad_quad( pbuo, Ra, Rb, wk) )  # 2 Im ∫ ρ 𝐮*⋅(Beyonce g s 𝐫̂) dV

    Enstro_vel_l = rad_quad( enstro_vel_p + enstro_vel_t, Ra, Rb, wk)
    Enstro_cor_l = rad_quad( enstro_cor_p + enstro_cor_t, Ra, Rb, wk)
    Enstro_vif_l = rad_quad( enstro_vif_p + enstro_vif_t, Ra, Rb, wk)
    if par.thermal:
        Enstro_buo_l = rad_quad( enstro_buo_p + enstro_buo_t, Ra, Rb, wk)
    else:
        Enstro_buo_l = 0


    Wdr_l = 0
    if par.diff_rot:
        wdr = diff_rot_power(l, lp, lt, P, T)
        Wdr_l = rad_quad( wdr, Ra, Rb, wk )
    
    return [ Kene_l, Dkin_l, Enstro_vel_l, Enstro_cor_l, Enstro_vif_l, Enstro_buo_l, Wthm_l, Wdr_l,
             Wcor_im_l, Dkin_im_l, Wthm_im_l ]



def worker_4plot( l ):
    '''
    This function returns the l-degree [rad,con,tor] components of the field
    defined by the global variable 'field'.
    '''

    if field in ['vel','curl_vel','2curl_vel']:
        [ [rad0, rad1, rad2], [con0, con1, con2], [tor0, tor1, tor2] ] = velocity(l)
    elif field == 'mf':
        [ rad0, con0, tor0 ] = massflux(l)
    
    elif field in ['vis','curl_vis','2curl_vis']:
        [ [rad0, rad1, rad2], [con0, con1, con2], [tor0, tor1, tor2] ] = visforce(l)
 
    elif field in ['cor','curl_cor','2curl_cor']:
        [ [rad0, rad1, rad2], [con0, con1, con2], [tor0, tor1, tor2] ] = coriolis(l)

    elif field in ['buo', 'curl_buo','2curl_buo']:
        [ [rad0, rad1, rad2], [con0, con1, con2], [tor0, tor1, tor2] ] = buoyancy(l)
 
    if field[:5]=='curl_':
        [out_rad, out_con, out_tor] = curl(l, [rad0, rad1, rad2], [con0, con1, con2], [tor0, tor1, tor2])
    elif field[:6]=='2curl_':
        [out_rad, out_con, out_tor] = curl2(l, [rad0, rad1, rad2], [con0, con1, con2], [tor0, tor1, tor2])
    else:
        [out_rad, out_con, out_tor] = [ rad0, con0, tor0 ]

    return [ out_rad, out_con, out_tor ]






def thermal_worker( l ):
    '''
    Entropy budget, l-component (l a poloidal l). The heat equation solved in section h (its rows are
    this equation times rʰ, see operators.entropy, thermal_advection, thermal_diffusion) is
        p₀ ∂s/∂t = -p₀ uᵣ dS/dr + ThermaD ∇⋅(κ p₀ ∇s)
    Multiplied by s* (weight 1, p₀ is already in the equation) and integrated over the fluid volume it gives,
    for an eigenmode ∝ exp((σ+iω)t),
        2σ TE = Wadv + ThermaD Dthm,   with
        TE   = ½ ∫ p₀ |s|² dV                            the entropy "energy"
        Wadv = -∫ p₀ (dS/dr) uᵣ s* dV                    the advection of the background entropy
        Dthm = ∫ s* ∇⋅(κ p₀ ∇s) dV = -∫ κ p₀ |∇s|² dV + [ r² κ p₀ s* ∂s/∂r ]   the entropy diffusion
    (real parts, same 2Re and Yₗᵐ conventions as flow_worker). The surface term vanishes for fixed-entropy
    or fixed-flux walls and at r=0. Uses the global solutions usol2, tsol2. Returns [ TE_l, Dthm_l, Wadv_l ],
    Dthm_l without the ThermaD factor.
    The profiles are the physical ones (setup_grid), so the budget residual also flags a mismatch between
    the intended and the assembled heat equation (e.g. a diffusion operator with a wrong coefficient, or,
    when ricb=0, a profile without the parity that submatrices assumes).
    '''

    Ra = par.ricb
    Rb = ut.rcmb

    lp  = ut.ell( par.m, par.lmax, par.symm)[0]
    idx = list(lp).index(l)

    f_s  = funcheb( tsol2[idx,:], r=rk, ricb=par.ricb, rcmb=ut.rcmb, n=1 )
    hlm0 = f_s[:,0]  # s
    hlm1 = f_s[:,1]  # ∂s/∂r

    [ velq, _, _ ] = velocity(l)  # velq[0] is uᵣ

    Tene_l = rad_quad( entropy_energy(l, hlm0), Ra, Rb, wk )
    Wadv_l = rad_quad( entropy_advect(l, hlm0, velq[0]), Ra, Rb, wk )

    Dthm_l = 0
    if par.ThermaD > 0:
        Dthm_l = rad_quad( entropy_dissip(l, hlm0, hlm1), Ra, Rb, wk )
        # surface term [ r² κ p₀ s* ∂s/∂r ] from Ra to Rb
        rb   = np.array([ Ra, Rb ])
        f_sb = funcheb( tsol2[idx,:], r=rb, ricb=par.ricb, rcmb=ut.rcmb, n=1 )
        kpsb = rap.prf.thermal_diffusivity(rb, 0) * rap.pressX(rb, 0)
        sfc  = (4*np.pi/(2*l+1)) * rb**2 * kpsb * 2*np.real( np.conj(f_sb[:,0]) * f_sb[:,1] )
        Dthm_l += sfc[1] - sfc[0]

    return [ Tene_l, Dthm_l, Wadv_l ]



def velocity( l ):
    '''
    Returns the rad, con, tor components of the l-component of the flow velocity,
    and their first radial derivatives. Sampled at given radii (global rk if radii is None).
    '''
        
    ll = ut.ell( par.m, par.lmax, par.symm)
    lp  = ll[0]  # l's for poloidals
    lt  = ll[1]  # l's for toroidals

    P = np.copy(usol2[0])
    T = np.copy(usol2[1])

    qlm0 = np.zeros_like(rk, dtype='complex128')
    slm0 = np.zeros_like(rk, dtype='complex128')
    tlm0 = np.zeros_like(rk, dtype='complex128')

    qlm1 = np.zeros_like(rk, dtype='complex128')
    slm1 = np.zeros_like(rk, dtype='complex128')
    tlm1 = np.zeros_like(rk, dtype='complex128')

    qlm2 = np.zeros_like(rk, dtype='complex128')
    slm2 = np.zeros_like(rk, dtype='complex128')
    tlm2 = np.zeros_like(rk, dtype='complex128') 

    if l in lp:

        [ [qlm0, qlm1, qlm2], [slm0, slm1, slm2] ] = cheb2space_pol(l, lp, P, 2, rk)

    elif l in lt:

        [ tlm0, tlm1, tlm2 ] = cheb2space_tor(l, lt, T, 2, rk)
    
    return [ [qlm0, qlm1, qlm2], [slm0, slm1, slm2], [tlm0, tlm1, tlm2] ]



def massflux( l ):

    [ [qlm0, _, _], [slm0, _, _], [tlm0, _, _] ] = velocity(l)

    return [ qlm0*rho0, slm0*rho0, tlm0*rho0 ]






def coriolis( l ):
    '''
    Returns the rad,con,tor components of the l-component of the Coriolis force.
    Returns also the first radial derivative. Sampled at the radii rk defined globally.
    '''
    L = l*(l+1.)
    m = par.m

    ll = ut.ell( par.m, par.lmax, par.symm)
    lp  = ll[0]  # l's for poloidals
    lt  = ll[1]  # l's for toroidals

    P = np.copy(usol2[0])
    T = np.copy(usol2[1])

    rad0 = np.zeros_like(rk, dtype='complex128') 
    con0 = np.zeros_like(rk, dtype='complex128')
    tor0 = np.zeros_like(rk, dtype='complex128')

    rad1 = np.zeros_like(rk, dtype='complex128') 
    con1 = np.zeros_like(rk, dtype='complex128')
    tor1 = np.zeros_like(rk, dtype='complex128')

    rad2 = np.zeros_like(rk, dtype='complex128')
    con2 = np.zeros_like(rk, dtype='complex128')
    tor2 = np.zeros_like(rk, dtype='complex128') 

    if l-1 in lt:
        [tlm0, tlm1, tlm2] = cheb2space_tor(l-1, lt, T, 2, rk)

        rad0 += ((-1 + l)*np.sqrt(l**2 - m**2)*tlm0)/(-1 + 2*l)
        rad1 += ((-1 + l)*np.sqrt(l**2 - m**2)*tlm1)/(-1 + 2*l)
        rad2 += ((-1 + l)*np.sqrt(l**2 - m**2)*tlm2)/(-1 + 2*l)

        con0 += ((-1 + l)*np.sqrt(l**2 - m**2)*tlm0)/(l*(-1 + 2*l))
        con1 += ((-1 + l)*np.sqrt(l**2 - m**2)*tlm1)/(l*(-1 + 2*l))
        con2 += ((-1 + l)*np.sqrt(l**2 - m**2)*tlm2)/(l*(-1 + 2*l))

    elif l-1 in lp:
        [ [qlm0, qlm1, qlm2], [slm0, slm1, slm2] ] = cheb2space_pol(l-1, lp, P, 2, rk)

        tor0 += (np.sqrt(l**2 - m**2)*(qlm0 + slm0 - l*slm0))/(l*(-1 + 2*l))
        tor1 += (np.sqrt(l**2 - m**2)*(qlm1 + slm1 - l*slm1))/(l*(-1 + 2*l))
        tor2 += (np.sqrt(l**2 - m**2)*(qlm2 + slm2 - l*slm2))/(l*(-1 + 2*l))
        

    if l in lp:
        [ [qlm0, qlm1, qlm2], [slm0, slm1, slm2] ] = cheb2space_pol(l, lp, P, 2, rk)

        rad0 += -1j*m*slm0
        rad1 += -1j*m*slm1
        rad2 += -1j*m*slm2

        con0 += (-1j*m*(qlm0 + slm0))/(l*(1 + l))
        con1 += (-1j*m*(qlm1 + slm1))/(l*(1 + l))
        con2 += (-1j*m*(qlm2 + slm2))/(l*(1 + l))

    elif l in lt:
        [tlm0, tlm1, tlm2] = cheb2space_tor(l, lt, T, 2, rk)

        tor0 += (-1j*m*tlm0)/(l + l**2)
        tor1 += (-1j*m*tlm1)/(l + l**2)
        tor2 += (-1j*m*tlm2)/(l + l**2)


    if l+1 in lt:
        [tlm0, tlm1, tlm2] = cheb2space_tor(l+1, lt, T, 2, rk)

        rad0 += -(((2 + l)*np.sqrt((1 + l - m)*(1 + l + m))*tlm0)/(3 + 2*l))
        rad1 += -(((2 + l)*np.sqrt((1 + l - m)*(1 + l + m))*tlm1)/(3 + 2*l))
        rad2 += -(((2 + l)*np.sqrt((1 + l - m)*(1 + l + m))*tlm2)/(3 + 2*l))

        con0 += ((2 + l)*np.sqrt((1 + l - m)*(1 + l + m))*tlm0)/((1 + l)*(3 + 2*l))
        con1 += ((2 + l)*np.sqrt((1 + l - m)*(1 + l + m))*tlm1)/((1 + l)*(3 + 2*l))
        con2 += ((2 + l)*np.sqrt((1 + l - m)*(1 + l + m))*tlm2)/((1 + l)*(3 + 2*l))

    elif l+1 in lp:
        [ [qlm0, qlm1, qlm2], [ slm0, slm1, slm2] ] = cheb2space_pol(l+1, lp, P, 2, rk)

        tor0 += -((np.sqrt((1 + l - m)*(1 + l + m))*(qlm0 + (2 + l)*slm0))/((1 + l)*(3 + 2*l)))
        tor1 += -((np.sqrt((1 + l - m)*(1 + l + m))*(qlm1 + (2 + l)*slm1))/((1 + l)*(3 + 2*l)))
        tor2 += -((np.sqrt((1 + l - m)*(1 + l + m))*(qlm2 + (2 + l)*slm2))/((1 + l)*(3 + 2*l)))

    return [ [rad0*2*par.Gaspard, rad1*2*par.Gaspard, rad2*2*par.Gaspard],
             [con0*2*par.Gaspard, con1*2*par.Gaspard, con2*2*par.Gaspard],
             [tor0*2*par.Gaspard, tor1*2*par.Gaspard, tor2*2*par.Gaspard] ]




        


def visforce( l ):
    '''
    Returns the three (rad,con,tor) components of the l-component of the viscous force divided by the density,
    (∇⋅σ)/ρ, and their first and second radial derivatives. Sampled at the radii rk defined globally.
    '''

    ll = ut.ell( par.m, par.lmax, par.symm)
    lp  = ll[0]  # l's for poloidals
    lt  = ll[1]  # l's for toroidals

    P = np.copy(usol2[0])
    T = np.copy(usol2[1])

    L = l*(l+1.)

    if l in lp:

        [ [qlm0, qlm1, qlm2, qlm3, qlm4], [slm0, slm1, slm2, slm3, slm4] ] = cheb2space_pol(l, lp, P, 4, rk)

        rad0 = ( 4*qlm2*r2*vsc0 + 7*L*slm0*vsc0 + 2*L*lho1*rk*slm0*vsc0 - L*rk*slm1*vsc0 + 2*L*rk*slm0*vsc1 
                 + 4*qlm1*rk*(2*vsc0 + lho1*rk*vsc0 + rk*vsc1) - qlm0*((8 + 3*L + 4*lho1*rk)*vsc0 + 4*rk*vsc1))/(3.*r2)

        con0 = (3*lho1*rk*(qlm0 - slm0 + rk*slm1)*vsc0 + (8*qlm0 - 4*L*slm0 + rk*(qlm1 + 6*slm1 + 3*rk*slm2))*vsc0 + 3*rk*(qlm0 - slm0 + rk*slm1)*vsc1)/(3.*r2)

        tor0 = np.zeros_like(rk, dtype='complex128')


        rad1 = ( ( 2*qlm0*(8 + 3*L - 2*lho2*r2 + 2*lho1*rk) + 2*L*(-7 + lho2*r2 - lho1*rk)*slm0 
                   + rk*( -(qlm1*(16 + 3*L - 4*lho2*r2 + 4*lho1*rk)) + 2*L*(4 + lho1*rk)*slm1 + rk*(4*qlm3*rk + 4*qlm2*(2 + lho1*rk) - L*slm2) ) )*vsc0 
                 + rk*( -(qlm0*(4 + 3*L + 4*lho1*rk)*vsc1) + L*(5 + 2*lho1*rk)*slm0*vsc1 + rk*(8*qlm2*rk + L*slm1)*vsc1 
                        - 4*qlm0*rk*vsc2 + 2*L*rk*slm0*vsc2 + 4*qlm1*rk*(vsc1 + lho1*rk*vsc1 + rk*vsc2) ) ) / (3.*r3)

        con1 = ( ( qlm0*(-16 + 3*lho2*r2 - 3*lho1*rk) + (8*L - 3*lho2*r2 + 3*lho1*rk)*slm0 
                   + rk*( qlm1*(7 + 3*lho1*rk) - (6 + 4*L - 3*lho2*r2 + 3*lho1*rk)*slm1 + rk*(qlm2 + 3*(2 + lho1*rk)*slm2 + 3*rk*slm3)) )*vsc0 
                 + rk*( qlm0*((5 + 3*lho1*rk)*vsc1 + 3*rk*vsc2) - slm0*((-3 + 4*L + 3*lho1*rk)*vsc1 + 3*rk*vsc2) 
                        + rk*((4*qlm1 + 3*(1 + lho1*rk)*slm1 + 6*rk*slm2)*vsc1 + 3*rk*slm1*vsc2) ) ) / (3.*r3)
        
        tor1 = np.zeros_like(rk, dtype='complex128')


        rad2 = ( ( -2*qlm0*(24 + 9*L - 4*lho2*r2 + 6*lh12*r3 - 6*lho1*lho2*r3 + 2*lho3*r3 + 4*lho1*rk) 
                   - 2*L*(-21 + 2*lho2*r2 - 3*lh12*r3 + 3*lho1*lho2*r3 - lho3*r3 - 2*lho1*rk)*slm0 
                   + rk*( 4*qlm1*(12 + 3*L - 2*lho2*r2 + 3*lh12*r3 - 3*lho1*lho2*r3 + lho3*r3 + 2*lho1*rk) 
                          + 2*L*(-15 + 2*lho2*r2 - 2*lho1*rk)*slm1 + rk*(-(qlm2*(24 + 3*L - 8*lho2*r2 + 4*lho1*rk)) 
                   + L*(9 + 2*lho1*rk)*slm2 + rk*(4*qlm4*rk + 4*qlm3*(2 + lho1*rk) - L*slm3))) )*vsc0 
                 + rk*( 4*qlm0*(6 + 3*L - 2*lho2*r2 + 2*lho1*rk)*vsc1 + 4*L*(-6 + lho2*r2 - lho1*rk)*slm0*vsc1 
                        + 2*rk*( -(qlm1*(12 + 3*L - 4*lho2*r2 + 4*lho1*rk)) + 2*rk*(3*qlm3*rk + qlm2*(3 + 2*lho1*rk)) + 2*L*(3 + lho1*rk)*slm1)*vsc1 
                        + L*rk*slm0*((3 + 2*lho1*rk)*vsc2 + 2*rk*vsc3) - qlm0*rk*(3*L*vsc2 + 4*rk*(lho1*vsc2 + vsc3)) 
                        + r2*(3*(4*qlm2*rk + L*slm1)*vsc2 + 4*qlm1*rk*(lho1*vsc2 + vsc3) ) ) ) / (3.*r4)

        con2 = ( ( -3*qlm0*(-16 + 2*lho2*r2 - 3*lh12*r3 + 3*lho1*lho2*r3 - lho3*r3 - 2*lho1*rk) - 3*(8*L - 2*lho2*r2 + 3*lh12*r3 - 3*lho1*lho2*r3 + lho3*r3 + 2*lho1*rk)*slm0 
                   + rk*( 6*qlm1*(-5 + lho2*r2 - lho1*rk) + (12 + 16*L - 6*lho2*r2 + 9*lh12*r3 - 9*lho1*lho2*r3 + 3*lho3*r3 + 6*lho1*rk)*slm1 
                          + rk*( 3*qlm2*(2 + lho1*rk) - (12 + 4*L - 6*lho2*r2 + 3*lho1*rk)*slm2 + rk*(qlm3 + 6*slm3 + 3*lho1*rk*slm3 + 3*rk*slm4) )))*vsc0 
                 + rk*( qlm0*(-26 + 6*lho2*r2 - 6*lho1*rk)*vsc1 + 2*(-3 + 8*L - 3*lho2*r2 + 3*lho1*rk)*slm0*vsc1 
                        + rk*( qlm1*(8 + 6*lho1*rk) - 2*(3 + 4*L - 3*lho2*r2 + 3*lho1*rk)*slm1 + rk*(5*qlm2 + (9 + 6*lho1*rk)*slm2 + 9*rk*slm3) )*vsc1 
                        + qlm0*rk*((2 + 3*lho1*rk)*vsc2 + 3*rk*vsc3) - rk*slm0*((-6 + 4*L + 3*lho1*rk)*vsc2 + 3*rk*vsc3) 
                        + r2*(7*qlm1*vsc2 + 9*rk*slm2*vsc2 + 3*rk*slm1*(lho1*vsc2 + vsc3)) ) ) / (3.*r4)

        tor2 = np.zeros_like(rk, dtype='complex128') 


    elif l in lt:
        
        [ tlm0, tlm1, tlm2, tlm3, tlm4 ] = cheb2space_tor(l, lt, T, 4, rk)

        rad0 = np.zeros_like(rk, dtype='complex128')
        con0 = np.zeros_like(rk, dtype='complex128')
        
        tor0 = ( -(L*tlm0*vsc0) + rk*(-(lho1*tlm0*vsc0) + 2*tlm1*vsc0 + lho1*rk*tlm1*vsc0 + rk*tlm2*vsc0 - tlm0*vsc1 + rk*tlm1*vsc1) ) / r2

        rad1 = np.zeros_like(rk, dtype='complex128')
        con1 = np.zeros_like(rk, dtype='complex128')
        
        tor1 = ( ((2*L - lho2*r2 + lho1*rk)*tlm0 + rk*(-((2 + L - lho2*r2 + lho1*rk)*tlm1) + rk*((2 + lho1*rk)*tlm2 + rk*tlm3)))*vsc0 
                 + rk*(-(tlm0*((-1 + L + lho1*rk)*vsc1 + rk*vsc2)) + rk*(((1 + lho1*rk)*tlm1 + 2*rk*tlm2)*vsc1 + rk*tlm1*vsc2) ) ) / r3

        rad2 = np.zeros_like(rk, dtype='complex128')
        con2 = np.zeros_like(rk, dtype='complex128')

        tor2 = ( ( -((6*L - 2*lho2*r2 + 3*lh12*r3 - 3*lho1*lho2*r3 + lho3*r3 + 2*lho1*rk)*tlm0) 
                   + rk*( (4 + 4*L - 2*lho2*r2 + 3*lh12*r3 - 3*lho1*lho2*r3 + lho3*r3 + 2*lho1*rk)*tlm1 
                          + rk*( -((4 + L - 2*lho2*r2 + lho1*rk)*tlm2) + rk*(2*tlm3 + lho1*rk*tlm3 + rk*tlm4) ) ) )*vsc0 
                 + rk*( 2*(-1 + 2*L - lho2*r2 + lho1*rk)*tlm0*vsc1 - 2*rk*(1 + L - lho2*r2 + lho1*rk)*tlm1*vsc1 + r2*((3 + 2*lho1*rk)*tlm2 + 3*rk*tlm3)*vsc1 
                        + 3*r3*tlm2*vsc2 + r3*tlm1*(lho1*vsc2 + vsc3) - rk*tlm0*((-2 + L + lho1*rk)*vsc2 + rk*vsc3) ) ) / r4

    return [ [rad0*par.ViscosD,rad1*par.ViscosD,rad2*par.ViscosD],
             [con0*par.ViscosD,con1*par.ViscosD,con2*par.ViscosD],
             [tor0*par.ViscosD,tor1*par.ViscosD,tor2*par.ViscosD] ]














def buoyancy(l):
    '''
    Returns the l-degree (rad,con,tor) components of the buoyancy force, and its radial derivatives.
    Use it to compute the rate of working (power) of the thermal buoyancy
    '''

    ll = ut.ell( par.m, par.lmax, par.symm)
    lp  = ll[0]  # l's for poloidals
    lt  = ll[1]  # l's for toroidals

    out_rad0 = np.zeros_like(rk, dtype='complex128')
    out_con0 = np.zeros_like(rk, dtype='complex128')
    out_tor0 = np.zeros_like(rk, dtype='complex128')
    out_rad1 = np.zeros_like(rk, dtype='complex128')
    out_con1 = np.zeros_like(rk, dtype='complex128')
    out_tor1 = np.zeros_like(rk, dtype='complex128')
    out_rad2 = np.zeros_like(rk, dtype='complex128')
    out_con2 = np.zeros_like(rk, dtype='complex128')
    out_tor2 = np.zeros_like(rk, dtype='complex128')

    if l in lp:
        idx   = list(lp).index(l)
        f_pol = funcheb( tsol2[idx,:], r=rk, ricb=par.ricb, rcmb=ut.rcmb, n=2 )
        out_rad0 = f_pol[:,0] * rap.graviX( rk, 0)
        out_rad1 = f_pol[:,1] * rap.graviX( rk, 0) + f_pol[:,0] * rap.graviX( rk, 1)
        out_rad2 = f_pol[:,2] * rap.graviX( rk, 0) + f_pol[:,1] * rap.graviX( rk, 1) + f_pol[:,1] * rap.graviX( rk, 1) + f_pol[:,0] * rap.graviX( rk, 2)
        
    return [ [out_rad0*par.Beyonce, out_rad1*par.Beyonce, out_rad2*par.Beyonce],
             [out_con0*par.Beyonce, out_con1*par.Beyonce, out_con2*par.Beyonce],
             [out_tor0*par.Beyonce, out_tor1*par.Beyonce, out_tor2*par.Beyonce] ]






def setup_grid(Ra, Rb):
    '''
    Sets, as module globals, the Gauss-Legendre radial grid (rk, wk, ...) over [Ra,Rb] and the background
    profiles sampled on it (rho0, lho1..lho5, vsc0..vsc3, const0, ...), as needed by the workers below.
    Called by diagnose() and kinetic_energy().
    '''
    # xk, wk are the nodes and weights for the radial integrals, Gauss-Legendre quadrature.
    # Always go from -1 to 1. The number of nodes Nq is decoupled from par.N: the integrands are
    # products of Chebyshev series of degree < N and smooth background profiles, sampled exactly at
    # any radius by funcheb. Nq = 3N/2 integrates exactly polynomials of degree < 3N.
    global wk
    Nq = (3*par.N + 1)//2
    xk, wk = np.polynomial.legendre.leggauss(Nq)

    # rk are the corresponding radial points in the desired integration interval: from Ra to Rb
    global rk
    rk = 0.5*(Rb-Ra)*( xk + 1 ) + Ra

    # x0 are the points in the appropriate domain of the Chebyshev polynomial solutions
    global x0
    x0 = xcheb(rk, par.ricb, 1)

    # the following are needed to compute the integrals (i.e. the quadratures)
    global r2
    r2 = rk**2
    global r3
    r3 = rk**3
    global r4
    r4 = rk**4

    global rho0
    rho0 = np.exp(rap.logrhoX(rk, 0))

    global lho1
    lho1 = rap.logrhoX(rk,1)
    global lho2
    lho2 = rap.logrhoX(rk,2)
    global lho3
    lho3 = rap.logrhoX(rk,3)
    global lho4
    lho4 = rap.logrhoX(rk,4)
    global lho5
    lho5 = rap.logrhoX(rk,5)


    global lh12
    lh12 = lho1*lho2


    [ rpower, rhopower ] = [ par.rpower_pp, par.rhopower_pp ]
    global const0
    const0 = (rk**rpower)*(rho0**rhopower)
    global vsc0
    vsc0 = rap.viscoX( rk, 0)
    global vsc1
    vsc1 = rap.viscoX( rk, 1)
    global vsc2
    vsc2 = rap.viscoX( rk, 2)
    global vsc3
    vsc3 = rap.viscoX( rk, 3)

    if par.thermal:
        # background profiles for the entropy budget (thermal_worker)
        global pss0
        pss0 = rap.pressX( rk, 0)                                 # p₀
        global pdS0
        pdS0 = rap.pdSdrX( rk, 0)                                 # p₀ dS/dr
        global kps0
        kps0 = rap.prf.thermal_diffusivity( rk, 0) * pss0         # κ p₀ (equal to rap.kappressX( rk, 0))



def kinetic_energy(usol, Ra, Rb, ls=None):
    '''
    Returns the kinetic energy ∫ ½ ρ 𝐮⋅𝐮 dV of the flow solution usol (as given by expand_reshape_sol),
    integrated from Ra to Rb, with the same integrand and quadrature as flow_worker (i.e. spin_doctor's KE).
    ls restricts the sum to some l values (default: all), so that the work can be split among MPI ranks.
    '''
    global usol2
    usol2 = usol
    setup_grid(Ra, Rb)
    if ls is None:
        ls = ut.ell(par.m, par.lmax, par.symm)[2]
    out = 0.0
    for l in ls:
        [ velq, vels, velt ] = velocity(l)
        kinep = 0.5*dotprod_pol(l, velq[0], vels[0], velq[0], vels[0] )*rho0
        kinet = 0.5*dotprod_tor(l, velt[0], velt[0])*rho0
        out += rad_quad( kinep + kinet, Ra, Rb, wk)
    return out



def diagnose( usol, tsol, Ra, Rb, ncpus):
    '''
    Computes kinetic energy, internal and kinetic energy dissipation,
    and input power from body forces. Integrated From r=Ra to r=Rb, and
    angularly over the whole sphere. Processed in parallel using ncpus.
    '''

    out_t = 0
    global usol2
    usol2 = usol
    global tsol2
    tsol2 = tsol


    setup_grid(Ra, Rb)  # quadrature grid and background profiles, as module globals

    [ lp_u, lt_u, ll ] = ut.ell(par.m, par.lmax, par.symm)  # the l-indices of the flow field
    
    # process each l-component in parallel
    pool = mp.Pool(processes=ncpus)


    ppu = [ pool.apply_async( flow_worker,
            args=( l, )) for l in ll ]
    out_u = np.array([pp0.get() for pp0 in ppu])
    

    if par.thermal:
        ppt = [ pool.apply_async( thermal_worker,
                args=( l, )) for l in lp_u ]
        out_t = np.array([pp0.get() for pp0 in ppt])


    pool.close()
    pool.join()

    return [ out_u, out_t ]
    


def diagnose_4plot( ncpus, usol, tsol, radii, field0):
    '''
    
    '''

    global field
    field =  field0

    global usol2
    usol2 = usol
    global tsol2
    tsol2 = tsol

    global rk
    rk = radii
    global r2
    r2 = rk**2
    global r3
    r3 = rk**3
    global r4
    r4 = rk**4

    global rho0
    rho0 = np.exp(rap.logrhoX(rk, 0))

    global lho1
    lho1 = rap.logrhoX(rk,1)
    global lho2
    lho2 = rap.logrhoX(rk,2)
    global lho3
    lho3 = rap.logrhoX(rk,3)
    global lho4
    lho4 = rap.logrhoX(rk,4)
    global lho5
    lho5 = rap.logrhoX(rk,5)
    global lh12
    lh12 = lho1*lho2

    # Kinematic viscosity and its derivatives
    global vsc0
    vsc0 = rap.viscoX( rk, 0)
    global vsc1
    vsc1 = rap.viscoX( rk, 1)
    global vsc2
    vsc2 = rap.viscoX( rk, 2)
    global vsc3
    vsc3 = rap.viscoX( rk, 3)

    [ lp_u, lt_u, ll ] = ut.ell(par.m, par.lmax, par.symm)  # the l-indices of the flow field
    
    # process each l-component in parallel
    pool = mp.Pool(processes=ncpus)

    ppu = [ pool.apply_async( worker_4plot, args=( l, )) for l in ll ]
    out = np.array([pp0.get() for pp0 in ppu])
    
    pool.close()
    pool.join()

    return out



def identify(sol2):

    threshold = 0.1

    P = np.abs(sol2[0])
    T = np.abs(sol2[1])

    lm1 = par.lmax - par.m + 1
    s   = int( par.symm*0.5 + 0.5 ) # s=0 if antisymm, s=1 if symm
    idp = np.arange( (np.sign(par.m)+s  )%2, lm1, 2, dtype=int)
    idt = np.arange( (np.sign(par.m)+s+1)%2, lm1, 2, dtype=int)
    ll  = np.arange( par.m+1-np.sign(par.m), par.lmax+2-np.sign(par.m), dtype=int)

    amps      = np.zeros_like(ll,dtype=float)
    amps[idp] = np.sum(P, axis=1)
    amps[idt] = np.sum(T, axis=1)

    maxid1      = np.argmax( amps )
    max_ell     = ll[ maxid1 ]
    maxids      = amps > threshold * amps[ maxid1 ]
    spread      = ll[maxids][-1] - ll[maxids][0]
    convergence = amps[-1] / amps[ maxid1 ]

    return max_ell, spread, convergence
