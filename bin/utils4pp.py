# The diagnose function in this module invokes in parallel the functions
# flow_worker, magnetic_worker, and thermal_worker.
# It provides energy, dissipation, power, etc for a given solution.

import multiprocessing as mp
import numpy.polynomial.chebyshev as ch
import numpy as np
import scipy.special as scsp
import scipy.sparse as ss
from parameters import par
import utils as ut
import radial_profiles as rap

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
        # The following needs debugging, it is super slow and gives incorrect results
        # for j in range(1,n+1):
        #     out[:,j] = Dgeg(ck0, r, ricb, rcmb, n)[:,0]

    return out



def Dgeg(ck0, r, ricb, rcmb, n):
    '''
    Returns the n-th order derivative of the function represented by the Cheb coeffs ck0.
    The derivative is returned in radial space, sampled at radii r.
    '''

    out = np.zeros( (np.size(r), 1), ck0.dtype)

    lamb = n
    N = np.size(ck0)

    # this is essentially a copy from ut.Dlam
    if ricb == 0:
        const1 = (1/rcmb)**lamb  # ok when rcmb is not 1
    else:
        const1 = (2./(rcmb-ricb))**lamb
    const2 = scsp.factorial(lamb-1.)*2**(lamb-1.)
    tmp = lamb + np.arange(0,N-lamb)

    # The Gegenbauer derivative operator, 
    leopold = const1*const2*ss.diags( tmp, lamb, format='csr', dtype='float64').todense()
    # Leopold Gegenbauer died aged 54

    ck0 = np.reshape(ck0, (-1,1))
    
    dgn = np.matmul(leopold, ck0)  # the derivative coeffs, in the Gegenbauer(lamb) basis
    x = xcheb(r, ricb, rcmb)

    for k in range(N):
        out[:,0] += scsp.eval_gegenbauer(k, lamb, x) * dgn[k,0]

    return out



def cg_quad(f, Ra, Rb, N, sqx):
    '''
    Computes the radial integral of f as a Chebyshev-Gauss quadrature.
    Assumes f is sampled over [Ra,Rb], with sqx=np.sqrt(1-xk**2),
    where xk are the radial grid points for the Chebyshev-Guauss quadrature. 
    '''

    out = (np.pi/N) * np.sum( sqx * f ) * (Rb-Ra)/2
 
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



# def cheb4pp_pol(l, lp, P):
#     '''
#     Returns qlm's (radial) and slm's (consoidal) L-components and first derivative.
#     *** Each component times (rk**4)*(rho0**4) ***
#     '''

#     L    = l*(l+1)
#     idx   = list(lp).index(l) 
#     f_pol = funcheb(P[idx,:], r=None, ricb=par.ricb, rcmb=ut.rcmb, n=4)
    
#     plm0 = f_pol[:,0]
#     plm1 = f_pol[:,1]
#     plm2 = f_pol[:,2]
#     plm3 = f_pol[:,3]
#     plm4 = f_pol[:,4]
#     rho04 = rho0**4
        
#     qlm0 = (L*plm0)*r3*rho04
#     slm0 = plm1*r4*rho04 + plm0*(r4*lho14 + r3*rho04)
    
#     qlm1 = L*(-plm0 + plm1*rk)*r2*rho04
#     slm1 = plm2*r4*rho04 + plm0*(r4*lho24 - r2*rho04) + plm1*(r4*lho14 + r3*rho04)

#     qlm2 = (L*(2*plm0 + rk*(-2*plm1 + plm2*rk)))*rk*rho04
#     slm2 = plm3*r4*rho04 + plm0*(r4*lho34 + 2*rk*rho04) + 2*plm1*(r4*lho24 - (r2*rho04)) + plm2*(r4*lho14 + r3*rho04)

#     qlm3 = (L*(-6*plm0 + rk*(6*plm1 + rk*(-3*plm2 + plm3*rk))))*rho04
#     slm3 = plm4*r4*rho04 + plm0*(r4*lho44 - 6*rho04) + 3*plm1*(r4*lho34 + 2*rk*rho04) + 3*plm2*(r4*lho24 - r2*rho04) + plm3*(r4*lho14 + r3*rho04)

#     return [ [qlm0, qlm1, qlm2, qlm3], [slm0, slm1, slm2, slm3] ]



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



# def cheb4pp_tor(L, lt, T):
#     '''
#     Returns tlm's (toroidal) L-components up to derivatives of order ns (<=3)
#     *** Each component times (rk**4)*(rho0**4) ***
#     '''
    
#     const = r4*rho0**4
#     [tlm0, tlm1, tlm2, tlm3] = cheb2space_tor(L, lt, T, 3, rk)

#     return [tlm0*const, tlm1*const, tlm2*const, tlm3*const]



def energy_pol(l, qlm0, slm0):
    '''
    Returns the integrand to compute the poloidal energy, kinetic or magnetic, l-component
    (1/2) ∫ ρ 𝐮⋅𝐮 dV or (1/2) ∫ 𝐛⋅𝐛 dV
    '''
    f0 = 4*np.pi*rho0/(2*l+1)
    f1 = r2 * np.absolute( qlm0 )**2
    f2 = r2 * l*(l+1) * np.absolute( slm0 )**2  # r2 is rk**2, a global variable
    return f0*(f1+f2)



def energy_tor(l, tlm0):
    '''
    Returns the integrand to compute the toroidal energy, kinetic or magnetic, l-component
    (1/2) ∫ ρ 𝐮⋅𝐮 dV or (1/2) ∫ 𝐛⋅𝐛 dV
    '''
    f0 = 4*np.pi*rho0/(2*l+1)
    f1 = r2 * l*(l+1) * np.absolute(tlm0)**2  # r2 is rk**2, a global variable
    return f0*f1



def diffus_pol(l, qlm0, qlm1, qlm2, slm0, slm1, slm2):
    '''
    Returns the integrand to compute diffusion, poloidal l-component
    ∫ 𝐮⋅∇²𝐮 dV
    '''
    L = l*(l+1)
    f0 = 4*np.pi/(2*l+1)
    f1 = L * r2 * np.conj(slm0) * slm2
    f2 = 2 * rk * L * np.conj(slm0) * slm1
    f3 = -(L**2)*( np.conj(slm0)*slm0 ) - (l**2+l+2) * ( np.conj(qlm0)*qlm0 )
    f4 = 2 * rk * np.conj(qlm0)*qlm1 + r2 * np.conj(qlm0) * qlm2
    f5 = 2 * L *( np.conj(qlm0)*slm0 + qlm0*np.conj(slm0) )
    return 2*np.real( f0*( f1+f2+f3+f4+f5 ) )



def diffus_tor(l, tlm0, tlm1, tlm2):
    '''
    Returns the integrand to compute diffusion, toroidal l-component
    ∫ 𝐮⋅∇²𝐮 dV
    '''
    L = l*(l+1)
    f0 = 4*np.pi/(2*l+1)
    f1 = L * r2 * np.conj(tlm0) * tlm2
    f2 = 2 * rk * L * np.conj(tlm0) * tlm1
    f3 = -(L**2)*( np.conj(tlm0)*tlm0 )
    return 2*np.real( f0*(f1+f2+f3) )



def internl_dissip_pol(l, qlm0, qlm1, slm0, slm1):
    '''
    Returns the integrand to compute the internal energy dissipation, poloidal l-component
    '''    
    L = l*(l+1)
    f0 = 4*np.pi/(2*l+1)
    f1 = L*np.absolute(qlm0 + rk*slm1 - slm0)**2
    f2 = 3*np.absolute(rk*qlm1)**2
    f3 = L*(l-1)*(l+2)*np.absolute(slm0)**2
    return 2*f0*( f1+f2+f3 )



def internl_dissip_tor(l, tlm0, tlm1):
    '''
    Returns the integrand to compute the internal energy dissipation, toroidal l-component
    '''
    L = l*(l+1)
    f0 = 4*np.pi/(2*l+1)
    f1 = L*np.absolute( rk*tlm1-tlm0 )**2
    f2 = L*(l-1)*(l+2)*np.absolute( tlm0 )**2    
    return 2*f0*( f1+f2 )



def ohmic_dissip_pol(l, qlm0, slm0, slm1):
    '''
    Returns the integrand to compute ∫ |∇×𝐛|² dV, poloidal l-component
    '''
    f0 = 8*np.pi* l*(l+1)/(2*l+1)
    f1 = np.absolute( qlm0 - slm0 - rk*slm1 )**2
    return f0*f1



def ohmic_dissip_tor(l, tlm0, tlm1):
    '''
    Returns the integrand to compute ∫ |∇×𝐛|² dV , toroidal l-component
    '''
    L = l*(l+1)
    f0 = 8*np.pi*L/(2*l+1)
    f1 = np.absolute( rk*tlm1 + tlm0 )**2
    f2 = L*np.absolute( tlm0 )**2
    return f0*(f1+f2)



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



def curl4pp(l, qlm, slm, tlm, rpower):
    '''
    Returns the l-component of the curl of the input vector. *** times rk**rpower ***
    Needs also the first radial derivative of the input vector.
    '''
    L = l*(l+1.)

    [qlm0, qlm1] = qlm
    [slm0, slm1] = slm
    [tlm0, tlm1] = tlm

    rp1 = rk**(rpower-1)
    rp0 = rk**(rpower)

    out_rad = L*tlm0*rp1
    out_con = tlm1*rp0 + tlm0*rp1
    out_tor = (qlm0-slm0)*rp1 - slm1*rp0

    return [out_rad, out_con, out_tor]
    


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



def lorentz_power_pol(l, qlm0, slm0, qlmb, slmb):
    '''
    Returns the integrand to compute the rate of working of the Lorentz force, poloidal l-component
    '''
    f0 = 4*np.pi/(2*l+1)
    f1 = r2 * 2 * np.real( qlm0 * np.conj( qlmb ) )
    f2 = r2 * l*(l+1) * 2 * np.real( slm0 * np.conj( slmb ) )
    return f0*(f1+f2)



def lorentz_power_tor(l, tlm0, tlmb):
    '''
    Returns the integrand to compute the rate of working of the Lorentz force, toroidal l-component
    '''
    f0 = 4*np.pi/(2*l+1)
    f1 = r2 * l*(l+1) * 2 * np.real( tlm0 * np.conj( tlmb ) )
    return f0*f1



def buoyancy_power(l, plm0, hlm0 ):
    '''
    Returns the integrand to compute rate of working of the buoyancy force, l-component, thermal or compositional
    '''
    f0 = 4*np.pi/(2*l+1)
    f1 = r2 * l*(l+1) * 2*np.real( np.conj(plm0) * hlm0 )
    return f0*f1



def thermal_energy(l, hlm0):
    '''
    Returns the integrand to compute the volume integral of (1/2) ∫ θ² dV, l-component
    '''
    f0 = 4*np.pi/(2*l+1)
    f1 = r2 * np.abs( hlm0 )**2
    return f0*f1



def thermal_dissip(l, hlm0, hlm1, hlm2):
    '''
    Returns the integrand to compute ∫ θ ∇²θ dV, l-component
    '''
    f0 = 4*np.pi/(2*l+1)
    f1 = 2 * rk * 2*np.real( hlm0 * np.conj(hlm1) )
    f2 = r2 * 2*np.real( hlm0 * np.conj(hlm2) )
    f3 = -2*l*(l+1) * np.abs(hlm0)**2
    return f0*(f1+f2+f3)



def thermal_advect(l, hlm0, plm0, flag):
    '''
    Returns the integrand to compute the volume integral of ∫ (-𝐮⋅∇T) θ dV, l-component
    For thermal or compositional depending on the flag argument
    '''

    f0 = 4*np.pi/(2*l+1)
    f1 = l*(l+1) * 2*np.real( np.conj(plm0) * hlm0 )

    if flag == 'thermal':
    
        if par.heating   == "internal":
            fr = r2
        elif par.heating == "differential":
            fr = 1/rk
        elif par.heating == "two zone":
            fr = rk * ut.twozone(rk, par.args)
        elif par.heating == "user defined":
            fr = rk * ut.BVprof(rk, par.args)

    elif flag == 'compositional':

        if par.comp_background  == "internal":
            fr = r2
        elif par.comp_background == "differential":
            fr = 1/rk        

    return f0*fr*f1



def flow_worker( l ):
    '''
    Computes the power balance from the momentum (the Navier-Stokes) equation.
    Includes kinetic energy, kinetic dissipation, internal dissipation, and the
    rate of working (power) of the Lorentz forces and buyancy forces (thermal and compositional).
    l-component
    '''

    Ra = par.ricb
    Rb = ut.rcmb
    N = par.N

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

    #[ intdp, intdt ] = [0, 0]
    #[ wlorp, wlort ] = [0, 0]
    #[ wther, wcomp ] = [0, 0]
    
    #L = l*(l+1)

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
    #kinep = energy_pol(l, uq, us)
    #kinet = energy_tor(l, ut)
    kinep = 0.5*dotprod_pol(l, velq[0], vels[0], velq[0], vels[0] )*rho0
    kinet = 0.5*dotprod_tor(l, velt[0], velt[0])*rho0

    # kinetic energy dissipation 𝐮⋅((∇⋅𝛔)/ρ) aka power of viscous force
    if par.ViscosD>0:
        kindp = dotprod_pol(l, velq[0], vels[0], vifq[0], vifs[0])
        kindt = dotprod_tor(l, velt[0], vift[0])
    else:
        kindp = 0
        kindt = 0

    # if par.magnetic:
    #     [ qlmb, slmb, tlmb ] = lorentz4pp(l, b_sol2)  # the l-component of the Lorentz force
        
    # if par.thermal:
    #     hlm0 = buoyancy4pp(l, lp, t_sol2)  # the l-component of the thermal buoyancy force
        
    # if par.compositional:
    #     clm0 = buoyancy4pp(l, lp, c_sol2)  # the l-component of the compositional buoyancy force
        

    # if l in lp:

        # [ [ qlm0, qlm1, qlm2 ], [ slm0, slm1, slm2 ] ] = cheb2space_pol(l, lp, P, 2)
        
        # kinep = energy_pol(l, qlm0, slm0 )
        # kindp = diffus_pol(l, qlm0, qlm1, qlm2, slm0, slm1, slm2 )
        # intdp = internl_dissip_pol(l, qlm0, qlm1, slm0, slm1 )
        # if par.magnetic:
        #     wlorp = lorentz_power_pol(l, qlm0, slm0, qlmb, slmb )
        # if par.thermal:
        #     wther = buoyancy_power(l, qlm0*rk/(l*(l+1)), hlm0 )
        # if par.compositional:
        #     wcomp = buoyancy_power(l, qlm0*rk/(l*(l+1)), clm0 ) 

    # elif l in lt:

        # [ tlm0, tlm1, tlm2 ] = cheb2space_tor(l, lt, T, 2)

        # kinet = energy_tor(l, tlm0)
        # kindt = diffus_tor(l, tlm0, tlm1, tlm2)
        # intdt = internl_dissip_tor(l, tlm0, tlm1)
        # if par.magnetic:
        #     wlort = lorentz_power_tor(l, tlm0, tlmb)

    # Integrals
    Kene_l = cg_quad( kinep + kinet, Ra, Rb, N, sqx)  # ∫ ½ ρ 𝐮⋅𝐮 dV 
    Dkin_l = cg_quad( kindp + kindt, Ra, Rb, N, sqx)  # ∫ 𝐮⋅(∇⋅𝛔) dV
    # Dint_l = cg_quad( intdp + intdt, Ra, Rb, N, sqx)
    # Wlor_l = cg_quad( wlorp + wlort, Ra, Rb, N, sqx)
    # Wthm_l = cg_quad( wther, Ra, Rb, N, sqx )
    # Wcmp_l = cg_quad( wcomp, Ra, Rb, N, sqx )

    Enstro_vel_l = cg_quad( enstro_vel_p + enstro_vel_t, Ra, Rb, N, sqx)
    Enstro_cor_l = cg_quad( enstro_cor_p + enstro_cor_t, Ra, Rb, N, sqx)
    Enstro_vif_l = cg_quad( enstro_vif_p + enstro_vif_t, Ra, Rb, N, sqx)
    if par.thermal:
        Enstro_buo_l = cg_quad( enstro_buo_p + enstro_buo_t, Ra, Rb, N, sqx)
    else:
        Enstro_buo_l = 0

    #test_l = -2*cg_quad( test_p + test_t, Ra, Rb, N, sqx)

    # return [ Kene_l, Dkin_l, Dint_l, Wlor_l, Wthm_l, Wcmp_l ]
    return [ Kene_l, Dkin_l, Enstro_vel_l, Enstro_cor_l, Enstro_vif_l, Enstro_buo_l, 0, 0, 0 ]



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



def magnetic_worker(l, lp, lt, b_sol2, u_sol2, Ra, Rb, N, sqx):
    '''
    Returns the l-component of the magnetic energy (1/2) ∫ 𝐛⋅𝐛 dV,
    the magnetic diffusion via ∫ 𝐛⋅∇²𝐛 dV, and the l-component
    of the induction term (integrated too).
    '''
    
    F = b_sol2[0]
    G = b_sol2[1]
 
    [ menep, menet] = [0, 0]
    [ mdfsp, mdfst] = [0, 0]
    [ indup, indut] = [0, 0]
    
    L = l*(l+1)

    if par.hydro:

        [ qlmi, slmi, tlmi ] = induction4pp(l, u_sol2)  # the l-component of the induction term 

    if l in lp:

        [ [qlm0, qlm1, qlm2], [slm0, slm1, slm2] ] = cheb2space_pol(l, lp, F, 2)

        menep = energy_pol(l, qlm0, slm0)
        mdfsp = diffus_pol(l, qlm0, qlm1, qlm2, slm0, slm1, slm2 )
        indup = dotprod_pol(l, qlm0, slm0, qlmi, slmi)

    elif l in lt:

        [tlm0, tlm1, tlm2] = cheb2space_tor(l, lt, G, 2)
        
        menet = energy_tor(l, tlm0)
        mdfst = diffus_tor(l, tlm0, tlm1, tlm2)
        indut = dotprod_tor(l, tlm0, tlmi)

   # Integrals
    Mene_l = cg_quad( menep + menet, Ra, Rb, N, sqx)
    Mdfs_l = cg_quad( mdfsp + mdfst, Ra, Rb, N, sqx)
    Indu_l = cg_quad( indup + indut, Ra, Rb, N, sqx)

    return [ Mene_l, Mdfs_l, Indu_l ]



def thermal_worker(l, lp, t_sol2, u_sol2, Ra, Rb, N, sqx, flag):
    '''
    Returns the l-component of the thermal "energy" i.e. (1/2) ∫ θ² dV,
    the thermal "dissipation" i.e. ∫ θ ∇²θ dV,
    and the thermal advection "power" i.e. ∫ (-𝐮⋅∇T) θ dV
    integrated over the fluid volume.
    '''

    [ thene , thdis, thadv ] = [0, 0, 0]

    if l in lp:

        [ hlm0, hlm1, hlm2 ] = cheb2space_tor(l, lp, t_sol2, 2)  # _tor is the one needed here, for the temperature (a scalar)

        thene = thermal_energy(l, hlm0)
        thdis = thermal_dissip(l, hlm0, hlm1, hlm2)
        if par.hydro:
            [ [qlm0], [_] ] = cheb2space_pol(l, lp, u_sol2[0], 0)  # _pol is the one needed here, for the velocity
            thadv = thermal_advect(l, hlm0, qlm0*rk/(l*(l+1)), flag)
        
    # Integrals
    Tene_l = cg_quad( thene, Ra, Rb, N, sqx )
    Dthm_l = cg_quad( thdis, Ra, Rb, N, sqx )
    Wadv_l = cg_quad( thadv, Ra, Rb, N, sqx )

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



# def velocity4pp( l ):
#     '''
#     Returns the rad, con, tor components of the l-component of the flow velocity,
#     and their first radial derivatives. Sampled at given radii (global rk if radii is None).
    
#     '''
        
#     ll = ut.ell( par.m, par.lmax, par.symm)
#     lp  = ll[0]  # l's for poloidals
#     lt  = ll[1]  # l's for toroidals

#     P = np.copy(usol2[0])
#     T = np.copy(usol2[1])

#     qlm0 = np.zeros_like(rk, dtype='complex128')
#     slm0 = np.zeros_like(rk, dtype='complex128')
#     tlm0 = np.zeros_like(rk, dtype='complex128')

#     qlm1 = np.zeros_like(rk, dtype='complex128')
#     slm1 = np.zeros_like(rk, dtype='complex128')
#     tlm1 = np.zeros_like(rk, dtype='complex128')

#     if l in lp:

#         #[ [qlm0, qlm1, _, _ ], [slm0, slm1, _, _ ] ] = cheb4pp_pol(l, lp, P)
#         [ [qlm0, qlm1], [slm0, slm1 ] ] = cheb2space_pol(l, lp, P, 1, rk)

#     elif l in lt:

#         #[ tlm0, tlm1, _, _ ] = cheb4pp_tor(l, lt, T)
#         [ tlm0, tlm1 ] = cheb2space_tor(l, lt, T, 1, rk)
    
#     return [ [qlm0, qlm1], [slm0, slm1], [tlm0, tlm1] ]



# def velocity_curl( l ):
#     '''

#     '''
        
#     ll = ut.ell( par.m, par.lmax, par.symm)
#     lp  = ll[0]  # l's for poloidals
#     lt  = ll[1]  # l's for toroidals

#     L = l*(l+1.)

#     P = np.copy(usol2[0])
#     T = np.copy(usol2[1])

#     rad0 = np.zeros_like(rk, dtype='complex128')
#     con0 = np.zeros_like(rk, dtype='complex128')
#     tor0 = np.zeros_like(rk, dtype='complex128')

#     if l in lp:

#         [ [qlm0, qlm1], [slm0, slm1] ] = cheb2space_pol(l, lp, P, 1, rk)
#         tor0 += (qlm0 - slm0 - rk*slm1)/rk

#     elif l in lt:

#         [ tlm0, tlm1 ] = cheb2space_tor(l, lt, T, 1, rk)
#         rad0 += (L*tlm0)/rk
#         con0 += tlm0/rk + tlm1
    
#     return [ rad0, con0, tor0 ]



# def massflux4pp( l, u_sol2):
#     '''
#     Returns the rad, con, tor components of the l-component of the mass flux ρu,
#     and their first radial derivatives. Sampled at the radii rk defined globally.
#     '''

#     [ [qlm0,qlm1], [slm0,slm1], [tlm0,tlm1] ] = velocity4pp( l, u_sol2)

#     rad0 = rho0*qlm0
#     con0 = rho0*slm0
#     tor0 = rho0*tlm0

#     rad1 = rho1*qlm0 + rho0*qlm1
#     con1 = rho1*slm0 + rho0*slm1
#     tor1 = rho1*tlm0 + rho0*tlm1

#     return [ [rad0,rad1], [con0,con1], [tor0,tor1] ]



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




# def coriolis4pp( l ):
#     '''
#     Returns the rad,con,tor components of the l-component of the Coriolis force.
#     Returns also the first radial derivative. Sampled at the radii rk defined globally.
#     '''

#     #const = (rho0**2)*(r3)  # this factor to match visforce4pp
#     const = 1.
#     L = l*(l+1.)
#     m = par.m

#     ll = ut.ell( par.m, par.lmax, par.symm)
#     lp  = ll[0]  # l's for poloidals
#     lt  = ll[1]  # l's for toroidals

#     P = np.copy(usol2[0])
#     T = np.copy(usol2[1])

#     rad0 = np.zeros_like(rk, dtype='complex128') 
#     con0 = np.zeros_like(rk, dtype='complex128')
#     tor0 = np.zeros_like(rk, dtype='complex128')

#     rad1 = np.zeros_like(rk, dtype='complex128') 
#     con1 = np.zeros_like(rk, dtype='complex128')
#     tor1 = np.zeros_like(rk, dtype='complex128')

#     if l-1 in lt:
#         [tlm0, tlm1, _, _ ] = cheb4pp_tor(l-1, lt, T)

#         rad0 += ((-1 + l)*np.sqrt(l**2 - m**2)*tlm0)/(-1 + 2*l)
#         rad1 += ((-1 + l)*np.sqrt(l**2 - m**2)*tlm1)/(-1 + 2*l)

#         con0 += ((-1 + l)*np.sqrt(l**2 - m**2)*tlm0)/(l*(-1 + 2*l))
#         con1 += ((-1 + l)*np.sqrt(l**2 - m**2)*tlm1)/(l*(-1 + 2*l))

#     elif l-1 in lp:
#         [ [qlm0, qlm1, _, _ ], [slm0, slm1, _, _ ] ] = cheb4pp_pol(l-1, lp, P)

#         tor0 += (np.sqrt(l**2 - m**2)*(qlm0 + slm0 - l*slm0))/(l*(-1 + 2*l))
#         tor1 += (np.sqrt(l**2 - m**2)*(qlm1 + slm1 - l*slm1))/(l*(-1 + 2*l))
        

#     if l in lp:
#         [ [qlm0, qlm1, _, _ ], [slm0, slm1, _, _ ] ] = cheb4pp_pol(l, lp, P)

#         rad0 += -1j*m*slm0
#         rad1 += -1j*m*slm1

#         con0 += (-1j*m*(qlm0 + slm0))/(l*(1 + l))
#         con1 += (-1j*m*(qlm1 + slm1))/(l*(1 + l))

#     elif l in lt:
#         [tlm0, tlm1, _, _ ] = cheb4pp_tor(l, lt, T)

#         tor0 += (-1j*m*tlm0)/(l + l**2)
#         tor1 += (-1j*m*tlm1)/(l + l**2)


#     if l+1 in lt:
#         [tlm0, tlm1, _, _ ] = cheb4pp_tor(l+1, lt, T)

#         rad0 += -(((2 + l)*np.sqrt((1 + l - m)*(1 + l + m))*tlm0)/(3 + 2*l))
#         rad1 += -(((2 + l)*np.sqrt((1 + l - m)*(1 + l + m))*tlm1)/(3 + 2*l))

#         con0 += ((2 + l)*np.sqrt((1 + l - m)*(1 + l + m))*tlm0)/((1 + l)*(3 + 2*l))
#         con1 += ((2 + l)*np.sqrt((1 + l - m)*(1 + l + m))*tlm1)/((1 + l)*(3 + 2*l))

#     elif l+1 in lp:
#         [ [qlm0, qlm1, _, _ ], [ slm0, slm1, _, _ ] ] = cheb4pp_pol(l+1, lp, P)

#         tor0 += -((np.sqrt((1 + l - m)*(1 + l + m))*(qlm0 + (2 + l)*slm0))/((1 + l)*(3 + 2*l)))
#         tor1 += -((np.sqrt((1 + l - m)*(1 + l + m))*(qlm1 + (2 + l)*slm1))/((1 + l)*(3 + 2*l)))


#     return [ [rad0*2*par.Gaspard*const, rad1*2*par.Gaspard*const],
#              [con0*2*par.Gaspard*const, con1*2*par.Gaspard*const],
#              [tor0*2*par.Gaspard*const, tor1*2*par.Gaspard*const] ]



# def coriolis_curl( l ):

#     m = par.m

#     ll = ut.ell( par.m, par.lmax, par.symm)
#     lp  = ll[0]  # l's for poloidals
#     lt  = ll[1]  # l's for toroidals

#     P = np.copy(usol2[0])
#     T = np.copy(usol2[1])

#     rad0 = np.zeros_like(rk, dtype='complex128') 
#     con0 = np.zeros_like(rk, dtype='complex128')
#     tor0 = np.zeros_like(rk, dtype='complex128')

#     if l-1 in lt:
#         [tlm0, tlm1] = cheb2space_tor(l-1, lt, T, 1, rk)
#         tor0 += ((-1 + l)*np.sqrt(l**2 - m**2)*((-1 + l)*tlm0 - rk*tlm1))/(l*(-1 + 2*l)*rk)
#         #tor0 += -(((1 + 2*l)*((-np.sqrt((l*(-1 + l**2))/(-1 + 4*l**2)) + np.sqrt((1 - l**2)/(l - 4*l**3)))*tlm0 + np.sqrt((1 - l**2)/(l - 4*l**3))*rk*tlm1))/(np.sqrt((l*(1 + l)*(-1 + 4*l**2))/((-1 + l)*(l**2 - m**2)))*rk))
        
#     elif l-1 in lp:
#         [[qlm0, qlm1], [slm0, slm1]] = cheb2space_pol(l-1, lp, P, 1, rk)
#         rad0 += ((1 + l)*np.sqrt(l**2 - m**2)*(qlm0 + slm0 - l*slm0))/((-1 + 2*l)*rk)
#         con0 += (np.sqrt(l**2 - m**2)*(qlm0 + qlm1*rk - (-1 + l)*(slm0 + rk*slm1)))/(l*(-1 + 2*l)*rk)
        
#     if l in lp:
#         [[qlm0, qlm1], [slm0, slm1]] = cheb2space_pol(l, lp, P, 1, rk)
#         tor0 += (1j*m*(qlm0 - (-1 + l + l**2)*slm0 + rk*(qlm1 + slm1)))/(l*(1 + l)*rk)
        
#     elif l in lt:
#         [tlm0, tlm1] = cheb2space_tor(l, lt, T, 1, rk)
#         rad0 += -1j*m*tlm0/rk
#         con0 += (-1j*m*(tlm0 + rk*tlm1))/(l*(1 + l)*rk)
        
#     if l+1 in lt:
#         [tlm0, tlm1] = cheb2space_tor(l+1, lt, T, 1, rk)
#         tor0 += -(((2 + l)*np.sqrt((1 + l - m)*(1 + l + m))*((2 + l)*tlm0 + rk*tlm1))/((1 + l)*(3 + 2*l)*rk))
#         #tor0 += -(((1 + 2*l)*np.sqrt(((2 + l)*(1 + l - m)*(1 + l + m))/(3 + 4*l*(2 + l)))*((np.sqrt((l*(2 + l))/(3 + 11*l + 12*l**2 + 4*l**3)) + np.sqrt((l*(1 + l)*(2 + l))/(3 + 4*l*(2 + l))))*tlm0 + np.sqrt((l*(2 + l))/(3 + 11*l + 12*l**2 + 4*l**3))*rk*tlm1))/(np.sqrt(l*(1 + l))*rk))
        
#     elif l+1 in lp:
#         [[qlm0, qlm1], [slm0, slm1]] = cheb2space_pol(l+1, lp, P, 1, rk)
#         rad0 += -((l*np.sqrt((1 + l - m)*(1 + l + m))*(qlm0 + (2 + l)*slm0))/((3 + 2*l)*rk))
#         con0 += -((np.sqrt((1 + l - m)*(1 + l + m))*(qlm0 + qlm1*rk + (2 + l)*(slm0 + rk*slm1)))/((1 + l)*(3 + 2*l)*rk))

#     return [rad0*2*par.Gaspard, con0*2*par.Gaspard, tor0*2*par.Gaspard]
        


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




# def visforce4pp( l ):
#     '''
#     Returns the three (rad,con,tor) components of the l-component of the viscous force divided by the density,
#     (∇⋅σ)/ρ, and their first radial derivatives. Sampled at the radii rk defined globally.
#     '''

#     ll = ut.ell( par.m, par.lmax, par.symm)
#     lp  = ll[0]  # l's for poloidals
#     lt  = ll[1]  # l's for toroidals

#     P = np.copy(usol2[0])
#     T = np.copy(usol2[1])

#     L = l*(l+1.)

#     if l in lp:

#         [ [qlm0, qlm1, qlm2, qlm3], [slm0, slm1, slm2, slm3] ] = cheb4pp_pol(l, lp, P)

#         rad0 = ( 2*mu1*rk*(-2*qlm0 + 2*qlm1*rk + L*slm0) - mu0*((8 + 3*L)*qlm0 - 7*L*slm0 + rk*(-8*qlm1 - 4*qlm2*rk + L*slm1)) ) / (3.*rho0*r2)
#         con0 = ( 3*mu1*rk*(qlm0 - slm0 + rk*slm1) + mu0*(8*qlm0 - 4*L*slm0 + rk*(qlm1 + 6*slm1 + 3*rk*slm2)) ) / (3.*rho0*r2)
#         tor0 = np.zeros_like(rk, dtype='complex128')

#         rad1 = ( -2*mu1*rho1*r2*(-2*qlm0 + 2*qlm1*rk + L*slm0) + mu0*rho1*rk*((8 + 3*L)*qlm0 - 7*L*slm0 + rk*(-8*qlm1 - 4*qlm2*rk + L*slm1)) 
#                  + rho0*rk*(2*mu2*rk*(-2*qlm0 + 2*qlm1*rk + L*slm0) + mu1*(-((4 + 3*L)*qlm0) + 4*qlm1*rk + 8*qlm2*r2 + 5*L*slm0 + L*rk*slm1)) 
#                  + mu0*rho0*(2*(8 + 3*L)*qlm0 - 14*L*slm0 + rk*(-((16 + 3*L)*qlm1) + 8*L*slm1 + rk*(8*qlm2 + 4*qlm3*rk - L*slm2))) ) / (3.*rho0**2*r3)
#         con1 = ( -3*mu1*rho1*r2*(qlm0 - slm0 + rk*slm1) - mu0*rho1*rk*(8*qlm0 - 4*L*slm0 + rk*(qlm1 + 6*slm1 + 3*rk*slm2)) 
#                  + rho0*rk*(3*mu2*rk*(qlm0 - slm0 + rk*slm1) + mu1*(5*qlm0 + 4*qlm1*rk + 3*slm0 - 4*L*slm0 + 3*rk*slm1 + 6*r2*slm2)) 
#                  + mu0*rho0*(-16*qlm0 + 8*L*slm0 + rk*(7*qlm1 - 2*(3 + 2*L)*slm1 + rk*(qlm2 + 6*slm2 + 3*rk*slm3))) ) / (3.*rho0**2*r3)
#         tor1 = np.zeros_like(rk, dtype='complex128')
        
#     elif l in lt:
        
#         [ tlm0, tlm1, tlm2, tlm3 ] = cheb4pp_tor(l, lt, T)

#         rad0 = np.zeros_like(rk, dtype='complex128')
#         con0 = np.zeros_like(rk, dtype='complex128')
#         tor0 = ( -(L*mu0*tlm0) + rk*(-(mu1*tlm0) + 2*mu0*tlm1 + mu1*rk*tlm1 + mu0*rk*tlm2) ) / (rho0*r2)

#         rad1 = np.zeros_like(rk, dtype='complex128')
#         con1 = np.zeros_like(rk, dtype='complex128')
#         tor1 = ( mu1*rho1*r2*(tlm0 - rk*tlm1) + mu0*rho1*rk*(L*tlm0 - rk*(2*tlm1 + rk*tlm2)) 
#                  + rho0*rk*(mu2*rk*(-tlm0 + rk*tlm1) + mu1*(tlm0 - L*tlm0 + rk*(tlm1 + 2*rk*tlm2))) 
#                  + mu0*rho0*(2*L*tlm0 + rk*(-((2 + L)*tlm1) + rk*(2*tlm2 + rk*tlm3)))) / (rho0**2*r3)

#     return [ [rad0*par.ViscosD,rad1*par.ViscosD],
#              [con0*par.ViscosD,con1*par.ViscosD],
#              [tor0*par.ViscosD,tor1*par.ViscosD] ]




# def visforce4pp( l ):
#     '''
#     Returns the three (rad,con,tor) components of the l-component of the viscous force divided by the density,
#     (∇⋅σ)/ρ, and their first radial derivatives. Sampled at the radii rk defined globally.
#     '''

#     ll = ut.ell( par.m, par.lmax, par.symm)
#     lp  = ll[0]  # l's for poloidals
#     lt  = ll[1]  # l's for toroidals

#     P = np.copy(usol2[0])
#     T = np.copy(usol2[1])

#     L = l*(l+1.)

#     if l in lp:

#         [ [qlm0, qlm1, qlm2, qlm3], [slm0, slm1, slm2, slm3] ] = cheb4pp_pol(l, lp, P)

#         rad0 = ( 2*mu1*rk*(-2*qlm0 + 2*qlm1*rk + L*slm0) - mu0*((8 + 3*L)*qlm0 - 7*L*slm0 + rk*(-8*qlm1 - 4*qlm2*rk + L*slm1)) ) * (rho0*rk/3.)  # / (3.*rho0*r2)
#         con0 = ( 3*mu1*rk*(qlm0 - slm0 + rk*slm1) + mu0*(8*qlm0 - 4*L*slm0 + rk*(qlm1 + 6*slm1 + 3*rk*slm2)) ) * (rho0*rk/3.)  # / (3.*rho0*r2)
#         tor0 = np.zeros_like(rk, dtype='complex128')

#         rad1 = ( -2*mu1*rho1*r2*(-2*qlm0 + 2*qlm1*rk + L*slm0) + mu0*rho1*rk*((8 + 3*L)*qlm0 - 7*L*slm0 + rk*(-8*qlm1 - 4*qlm2*rk + L*slm1)) 
#                  + rho0*rk*(2*mu2*rk*(-2*qlm0 + 2*qlm1*rk + L*slm0) + mu1*(-((4 + 3*L)*qlm0) + 4*qlm1*rk + 8*qlm2*r2 + 5*L*slm0 + L*rk*slm1)) 
#                  + mu0*rho0*(2*(8 + 3*L)*qlm0 - 14*L*slm0 + rk*(-((16 + 3*L)*qlm1) + 8*L*slm1 + rk*(8*qlm2 + 4*qlm3*rk - L*slm2))) )/3.  # / (3.*rho0**2*r3)
#         con1 = ( -3*mu1*rho1*r2*(qlm0 - slm0 + rk*slm1) - mu0*rho1*rk*(8*qlm0 - 4*L*slm0 + rk*(qlm1 + 6*slm1 + 3*rk*slm2)) 
#                  + rho0*rk*(3*mu2*rk*(qlm0 - slm0 + rk*slm1) + mu1*(5*qlm0 + 4*qlm1*rk + 3*slm0 - 4*L*slm0 + 3*rk*slm1 + 6*r2*slm2)) 
#                  + mu0*rho0*(-16*qlm0 + 8*L*slm0 + rk*(7*qlm1 - 2*(3 + 2*L)*slm1 + rk*(qlm2 + 6*slm2 + 3*rk*slm3))) )/3.  # / (3.*rho0**2*r3)
#         tor1 = np.zeros_like(rk, dtype='complex128')
        
#     elif l in lt:
        
#         [ tlm0, tlm1, tlm2, tlm3 ] = cheb4pp_tor(l, lt, T)

#         rad0 = np.zeros_like(rk, dtype='complex128')
#         con0 = np.zeros_like(rk, dtype='complex128')
#         tor0 = ( -(L*mu0*tlm0) + rk*(-(mu1*tlm0) + 2*mu0*tlm1 + mu1*rk*tlm1 + mu0*rk*tlm2) ) * (rho0*rk)  # / (rho0*r2)

#         rad1 = np.zeros_like(rk, dtype='complex128')
#         con1 = np.zeros_like(rk, dtype='complex128')
#         tor1 = ( mu1*rho1*r2*(tlm0 - rk*tlm1) + mu0*rho1*rk*(L*tlm0 - rk*(2*tlm1 + rk*tlm2)) 
#                  + rho0*rk*(mu2*rk*(-tlm0 + rk*tlm1) + mu1*(tlm0 - L*tlm0 + rk*(tlm1 + 2*rk*tlm2))) 
#                  + mu0*rho0*(2*L*tlm0 + rk*(-((2 + L)*tlm1) + rk*(2*tlm2 + rk*tlm3))))  # / (rho0**2*r3)

#     return [ [rad0*par.ViscosD,rad1*par.ViscosD],
#              [con0*par.ViscosD,con1*par.ViscosD],
#              [tor0*par.ViscosD,tor1*par.ViscosD] ]





# def visforce_curl( l ):

#     ll = ut.ell( par.m, par.lmax, par.symm)
#     lp  = ll[0]  # l's for poloidals
#     lt  = ll[1]  # l's for toroidals

#     P = np.copy(usol2[0])
#     T = np.copy(usol2[1])

#     rad0 = np.zeros_like(rk, dtype='complex128') 
#     con0 = np.zeros_like(rk, dtype='complex128')
#     tor0 = np.zeros_like(rk, dtype='complex128')

#     L = l*(l+1.)

#     if l in lp:

#         [ [qlm0, qlm1, qlm2, qlm3], [slm0, slm1, slm2, slm3] ] = cheb2space_pol(l, lp, P, 3, rk)
#         tor0 += (3*mu1*r2*rho1*(qlm0 - slm0 + rk*slm1) + mu0*rho1*rk*(8*qlm0 - 4*L*slm0 + rk*(qlm1 + 6*slm1 + 3*rk*slm2)) 
#                  - 3*rho0*rk*(mu2*rk*(qlm0 - slm0 + rk*slm1) + 2*mu1*(2*qlm0 - L*slm0 + rk*(slm1 + rk*slm2))) 
#                  - 3*mu0*rho0*(L*(qlm0 - slm0 - rk*slm1) + rk**2*(-qlm2 + 3*slm2 + rk*slm3)))/(3.*rho0**2*rk**3)
     
#     elif l in lt:
        
#         [ tlm0, tlm1, tlm2, tlm3 ] = cheb2space_tor(l, lt, T, 3, rk)
#         rad0 += (L*(mu1*rk*(-tlm0 + rk*tlm1) + mu0*(-(L*tlm0) + rk*(2*tlm1 + rk*tlm2))))/(rho0*rk**3)
#         con0 += (rk*(mu1*rho1*rk*(tlm0 - rk*tlm1) + rho0*(-((L*mu1 + mu2*rk)*tlm0) + rk*(mu2*rk*tlm1 + 2*mu1*(tlm1 + rk*tlm2)))) 
#                  + mu0*(rho1*rk*(L*tlm0 - rk*(2*tlm1 + rk*tlm2)) + rho0*(L*tlm0 + rk*(-(L*tlm1) + rk*(3*tlm2 + rk*tlm3)))))/(rho0**2*rk**3)

#     return [ rad0*par.ViscosD, con0*par.ViscosD, tor0*par.ViscosD ]




def lorentz4pp( l, b_sol2 ):
    '''
    Returns the l-component of the Lorentz force.
    Use it to compute the rate of working (power) of the Lorentz force. For quadrupolar B0
    '''
    
    m   = par.m
    ll0 = ut.ell( m, par.lmax, ut.bsymm)
    lp  = ll0[0]  # l's for poloidals
    lt  = ll0[1]  # l's for toroidals
    
    P = b_sol2[0]
    T = b_sol2[1]

    h0 = ut.h0(rk, par.B0, [par.beta, par.B0_l, par.ricb, 0])
    h1 = ut.h1(rk, par.B0, [par.beta, par.B0_l, par.ricb, 0])
    h2 = ut.h2(rk, par.B0, [par.beta, par.B0_l, par.ricb, 0])

    cnorm = ut.B0_norm()

    out_rad = np.zeros_like(rk, dtype='complex128')
    out_con = np.zeros_like(rk, dtype='complex128')
    out_tor = np.zeros_like(rk, dtype='complex128')
    
    if l-2 in lp:

        [ [ qlm0, _ ], [ slm0, slm1 ] ] = cheb2space_pol(l-2, lp, P, 1)

        C_rad    = 3*(-2 + l)*np.sqrt((-1 + l - m)*(l - m)*(-1 + l + m)*(l + m))/(3 + 4*(-2 + l)*l)
        out_rad += C_rad*( -h0*(qlm0/r2 + 5*slm0/r2 - slm1/rk) + h2*slm0 + h1*(-qlm0/rk + 3*slm0/rk + slm1) )

        C_con    = 3*np.sqrt((-1 + l - m)*(l - m)*(-1 + l + m)*(l + m))/(3*l + 4*(-2 + l)*l**2)
        out_con += C_con*( -3*h0*l*qlm0/r2 + qlm0*(2*h1/rk + h2) + 3*h0*(-2 + l)*(slm0/r2 + slm1/rk) )

    elif l-2 in lt:

        [ tlm0, tlm1 ] = cheb2space_tor(l-2, lt, T, 1)

        C_tor    = -3*(-2 + l)*np.sqrt((-1 + l - m)*(l - m)*(-1 + l + m)*(l + m))/(l*(3 + 4*(-2 + l)*l))
        out_tor += C_tor*( h0*(-4 + l)*tlm0/r2 + h1*(-1 + l)*tlm0/rk - 3*h0*tlm1/rk )

    if l-1 in lt:

        [ tlm0, tlm1 ] = cheb2space_tor(l-1, lt, T, 1)
           
        C_rad    = 3j*m*np.sqrt(l**2-m**2)/(2*l-1) 
        out_rad += C_rad*( h0*(-5*tlm0/r2 + tlm1/rk) + 3*h1*tlm0/rk + h2*tlm0 + h1*tlm1 )

        C_con    = 3j*m*np.sqrt(l**2 - m**2)/(l*(1 + l)*(-1 + 2*l))
        out_con += C_con*( h0*(6 + (-1 + l)*l)*tlm0/r2 + h1*(-1 + l)*l*tlm0/rk + 6*h0*tlm1/rk )

    elif l-1 in lp:

        [ [ qlm0, _ ], [ slm0, slm1 ] ] = cheb2space_pol(l-1, lp, P, 1)

        C_tor    =  3j*m*np.sqrt(l**2 - m**2)/(l*(1 + l)*(-1 + 2*l))      
        out_tor += C_tor*( qlm0*(2*h1/rk + h2) - 6*h0*(slm0/r2 + slm1/rk) )

    if l in lp:

        [ [ qlm0, _ ], [ slm0, slm1 ] ] = cheb2space_pol(l, lp, P, 1)

        C_rad    = -3*(l+l**2-3*m**2)/(-3+4*l*(l+1))
        out_rad += C_rad*( -h0*(qlm0/r2 + 5*slm0/r2 - slm1/rk) + h2*slm0 + h1*(-qlm0/rk + 3*slm0/rk + slm1) )

        C_con    = 3*(l + l**2 - 3*m**2)/(l*(1 + l)*(-3 + 4*l*(1 + l)))
        out_con += C_con*( -2*h0*l*(1 + l)*qlm0/r2 + qlm0*(2*h1/rk + h2) + 2*h0*(-3 + l + l**2)*(slm0/r2 + slm1/rk) )

    elif l in lt:

        [ tlm0, tlm1 ] = cheb2space_tor(l, lt, T, 1)
    
        C_tor    = 3*(l + l**2 - 3*m**2)/(l*(1 + l)*(-3 + 4*l*(1 + l)))
        out_tor += C_tor*( h0*(-6 + l + l**2)*tlm0/r2 - h1*l*(1 + l)*tlm0/rk + 2*h0*(-3 + l + l**2)*tlm1/rk )

    if l+1 in lt:

        [ tlm0, tlm1 ] = cheb2space_tor(l+1, lt, T, 1)
    
        C_rad    = 3j*m*np.sqrt((l+1-m)*(l+1+m))/(2*l+3)
        out_rad += C_rad*( h0*(-5*tlm0/r2 + tlm1/rk) + 3*h1*tlm0/rk + h2*tlm0 + h1*tlm1 )

        C_con    = 3j*m*np.sqrt((1 + l - m)*(1 + l + m))/(l*(1 + l)*(3 + 2*l))
        out_con += C_con*( h0*(8 + l*(3 + l))*tlm0/r2 + h1*(1 + l)*(2 + l)*tlm0/rk + 6*h0*tlm1/rk )

    elif l+1 in lp:

        [ [ qlm0, _ ], [ slm0, slm1 ] ] = cheb2space_pol(l+1, lp, P, 1)

        C_tor    = 3j*m*np.sqrt((1 + l - m)*(1 + l + m))/(l*(1 + l)*(3 + 2*l))
        out_tor += C_tor*( qlm0*(2*h1/rk + h2) - 6*h0*(slm0/r2 + slm1/rk) )
    
    if l+2 in lp:

        [ [ qlm0, _ ], [ slm0, slm1 ] ] = cheb2space_pol(l+2, lp, P, 1)

        C_rad    = -3*(l+3)*np.sqrt((1+l-m)*(2+l-m)*(1+l+m)*(2+l+m))/((2*l+3)*(2*l+5))
        out_rad += C_rad*( -h0*(qlm0/r2 + 5*slm0/r2 - slm1/rk) + h2*slm0 + h1*(-qlm0/rk + 3*slm0/rk + slm1) )

        C_con    = -3*np.sqrt((1 + l - m)*(2 + l - m)*(1 + l + m)*(2 + l + m))/((1 + l)*(3 + 2*l)*(5 + 2*l))
        out_con += C_con*( 3*h0*(1 + l)*qlm0/r2 + qlm0*(2*h1/rk + h2) - 3*h0*(3 + l)*(slm0/r2 + slm1/rk) )

    elif l+2 in lt:

        [ tlm0, tlm1 ] = cheb2space_tor(l+2, lt, T, 1)
    
        C_tor    = 3*(l+3)*np.sqrt((1 + l - m)*(2 + l - m)*(1 + l + m)*(2 + l + m))/((1 + l)*(3 + 2*l)*(5 + 2*l))
        out_tor += C_tor*( h0*(l+5)*tlm0/r2 + h1*(l+2)*tlm0/rk + 3*h0*tlm1/rk )

    return [out_rad * cnorm, out_con * cnorm, out_tor * cnorm]



def induction4pp( l, u_sol2 ):
    '''
    Returns the l-component of the induction term ∇×(𝐮×𝐁₀), quadrupolar B0
    '''
    
    m   = par.m
    ll0 = ut.ell( m, par.lmax, par.symm)
    lp  = ll0[0]  # l's for poloidals
    lt  = ll0[1]  # l's for toroidals
    ricb = par.ricb
    rcmb = ut.rcmb
    
    P = u_sol2[0]
    T = u_sol2[1]

    h0 = ut.h0(rk, par.B0, [par.beta, par.B0_l, ricb, 0])
    h1 = ut.h1(rk, par.B0, [par.beta, par.B0_l, ricb, 0])
    h2 = ut.h2(rk, par.B0, [par.beta, par.B0_l, ricb, 0])

    cnorm = ut.B0_norm()

    out_rad = np.zeros_like(rk, dtype='complex128')
    out_con = np.zeros_like(rk, dtype='complex128')
    out_tor = np.zeros_like(rk, dtype='complex128')
    
    if l-2 in lp:

        [ [ qlm0, qlm1], [slm0, slm1] ] =  cheb2space_pol(l-2, lp, P, 1)

        out_rad += (3*(1 + l)*np.sqrt((-1 + l - m)*(l - m)*(-1 + l + m)*(l + m))*(-(h0*qlm0) - h1*qlm0*rk + 3*h0*(-2 + l)*slm0))/((3 + 4*(-2 + l)*l)*r2)

        out_con += (3*np.sqrt((-1 + l - m)*(l - m)*(-1 + l + m)*(l + m))*(-(qlm1*(h0 + h1*rk)) - qlm0*(2*h1 + h2*rk) + 3*h1*(-2 + l)*slm0 + 3*h0*(-2 + l)*slm1))/(l*(3 + 4*(-2 + l)*l)*rk)

    elif l-2 in lt:

        [ tlm0, tlm1]  =  cheb2space_tor(l-2, lt, T, 1)

        out_tor += (-3*(-2 + l)*np.sqrt((-1 + l - m)*(l - m)*(-1 + l + m)*(l + m))*(h0*l*tlm0 + h1*(-3 + l)*rk*tlm0 - 3*h0*rk*tlm1))/(l*(3 + 4*(-2 + l)*l)*r2)

    if l-1 in lt:

        [ tlm0, tlm1]  =  cheb2space_tor(l-1, lt, T, 1)
           
        out_rad  += (18j*h0*m*np.sqrt(l**2 - m**2)*tlm0)/((-1 + 2*l)*r2)

        out_con += (18j*m*np.sqrt(l**2 - m**2)*(h1*tlm0 + h0*tlm1))/(l*(-1 + l + 2*l**2)*rk)

    elif l-1 in lp:

        [ [ qlm0, qlm1], [slm0, slm1] ] =  cheb2space_pol(l-1, lp, P, 1)

        out_tor += (3j*m*np.sqrt(l**2 - m**2)*(-(qlm0*rk*(2*h1 + h2*rk)) + h0*l*(1 + l)*slm0 + h1*rk*(-(qlm1*rk) + (-6 + l + l**2)*slm0) - h0*rk*(qlm1 + 6*slm1)))/(l*(1 + l)*(-1 + 2*l)*r2)

    if l in lp:

        [ [ qlm0, qlm1], [slm0, slm1] ] =  cheb2space_pol(l, lp, P, 1)

        out_rad += (-3*(l + l**2 - 3*m**2)*(h1*qlm0*rk + h0*(qlm0 - 2*(-3 + l + l**2)*slm0)))/((-3 + 4*l*(1 + l))*r2)

        out_con += (-3*(l + l**2 - 3*m**2)*(qlm1*(h0 + h1*rk) + qlm0*(2*h1 + h2*rk) - 2*h1*(-3 + l + l**2)*slm0 - 2*h0*(-3 + l + l**2)*slm1))/(l*(1 + l)*(-3 + 4*l*(1 + l))*rk)

    elif l in lt:

        [ tlm0, tlm1]  =  cheb2space_tor(l, lt, T, 1)
    
        out_tor += (3*(l + l**2 - 3*m**2)*(h0*l*(1 + l)*tlm0 + 3*h1*(-2 + l + l**2)*rk*tlm0 + 2*h0*(-3 + l + l**2)*rk*tlm1))/(l*(1 + l)*(-3 + 4*l*(1 + l))*r2)

    if l+1 in lt:

        [ tlm0, tlm1]  =  cheb2space_tor(l+1, lt, T, 1)
    
        out_rad += (18j*h0*m*np.sqrt((1 + l - m)*(1 + l + m))*tlm0)/((3 + 2*l)*r2)

        out_con += (18j*m*np.sqrt((1 + l - m)*(1 + l + m))*(h1*tlm0 + h0*tlm1))/(l*(1 + l)*(3 + 2*l)*rk)

    elif l+1 in lp:

        [ [ qlm0, qlm1], [slm0, slm1] ] =  cheb2space_pol(l+1, lp, P, 1)

        out_tor += (3j*m*np.sqrt((1 + l - m)*(1 + l + m))*(-(qlm0*rk*(2*h1 + h2*rk)) + h0*l*(1 + l)*slm0 + h1*rk*(-(qlm1*rk) + (-6 + l + l**2)*slm0) - h0*rk*(qlm1 + 6*slm1)))/(l*(1 + l)*(3 + 2*l)*r2) 
    
    if l+2 in lp:

        [ [ qlm0, qlm1], [slm0, slm1] ] =  cheb2space_pol(l+2, lp, P, 1)

        out_rad += (3*l*np.sqrt((1 + l - m)*(2 + l - m)*(1 + l + m)*(2 + l + m))*(h1*qlm0*rk + h0*(qlm0 + 3*(3 + l)*slm0)))/((3 + 2*l)*(5 + 2*l)*r2)

        out_con += (3*np.sqrt((1 + l - m)*(2 + l - m)*(1 + l + m)*(2 + l + m))*(h2*qlm0*rk + h1*(2*qlm0 + qlm1*rk + 3*(3 + l)*slm0) + h0*(qlm1 + 3*(3 + l)*slm1)))/((1 + l)*(3 + 2*l)*(5 + 2*l)*rk)

    elif l+2 in lt:

        [ tlm0, tlm1]  =  cheb2space_tor(l+2, lt, T, 1)
    
        out_tor += (3*(3 + l)*np.sqrt((1 + l - m)*(2 + l - m)*(1 + l + m)*(2 + l + m))*(h1*(4 + l)*rk*tlm0 + h0*(tlm0 + l*tlm0 + 3*rk*tlm1)))/((1 + l)*(3 + 2*l)*(5 + 2*l)*r2)

    return [out_rad * cnorm, out_con * cnorm, out_tor * cnorm]



def buoyancy(l):
    '''
    Returns the l-degree (rad,con,tor) components of the buoyancy force, and its radial derivatives.
    Use it to compute the rate of working (power) of buoyancy, either thermal or compositional
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



# def buoyancy4pp(l):
#     '''
#     Returns the l-degree (rad,con,tor) components of the buoyancy force, and its radial derivatives.
#     Use it to compute the rate of working (power) of buoyancy, either thermal or compositional
#     '''

#     #const0 = (rho0**4)*(r4)  # to match cheb4pp_pol
#     #const1 = (rho0**2)*(r3)  # to match visforce4pp
#     #const1 = 1.
#     #const = const0*const1

#     ll = ut.ell( par.m, par.lmax, par.symm)
#     lp  = ll[0]  # l's for poloidals
#     lt  = ll[1]  # l's for toroidals

#     out_rad0 = np.zeros_like(rk, dtype='complex128')
#     out_con0 = np.zeros_like(rk, dtype='complex128')
#     out_tor0 = np.zeros_like(rk, dtype='complex128')
#     out_rad1 = np.zeros_like(rk, dtype='complex128')
#     out_con1 = np.zeros_like(rk, dtype='complex128')
#     out_tor1 = np.zeros_like(rk, dtype='complex128')
    
#     if l in lp:
#         idx   = list(lp).index(l)
#         f_pol = funcheb( tsol2[idx,:], r=rk, ricb=par.ricb, rcmb=ut.rcmb, n=1 )
#         out_rad0  = f_pol[:,0] * rap.graviX( rk, 0)
#         out_rad1  = f_pol[:,1] * rap.graviX( rk, 0) + f_pol[:,0] * rap.graviX( rk, 1)
        
#     return [ [out_rad0*par.Beyonce, out_rad1*par.Beyonce],
#              [out_con0*par.Beyonce, out_con1*par.Beyonce],
#              [out_tor0*par.Beyonce, out_tor1*par.Beyonce] ]



def diagnose( usol, bsol2, tsol, csol2, Ra, Rb, ncpus):
    '''
    Computes kinetic energy, internal and kinetic energy dissipation,
    and input power from body forces. Integrated From r=Ra to r=Rb, and
    angularly over the whole sphere. Processed in parallel using ncpus.
    '''
    [out_u, out_b] = [0,0]
    [out_t, out_c] = [0,0]

    global usol2
    usol2 = usol
    global tsol2
    tsol2 = tsol


    # xk are the grid points for the integration using Gauss-Chebyshev quadratures.
    # Always go from -1 to 1
    i = np.arange(0,par.N)
    xk = np.cos( (i+0.5)*np.pi/par.N )

    # rk are the corresponding radial points in the desired integration interval: from Ra to Rb
    global rk
    rk = 0.5*(Rb-Ra)*( xk + 1 ) + Ra

    # x0 are the points in the appropriate domain of the Chebyshev polynomial solutions
    global x0
    x0 = xcheb(rk, par.ricb, 1)

    # the following are needed to compute the integrals (i.e. the quadratures)
    global sqx
    sqx = np.sqrt(1-xk**2)
    global r2
    r2 = rk**2
    global r3
    r3 = rk**3
    global r4
    r4 = rk**4

    global rho0
    rho0 = np.exp(rap.logrhoX(rk, 0))
    # global rho1
    # rho1 = np.exp(rap.logrhoX(rk, 1))

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

    # global lho14
    # lho14 = rap.lhoX(rk,1)*rho0**3
    # global lho24
    # lho24 = rap.lhoX(rk,2)*rho0**2
    # global lho34
    # lho34 = rap.lhoX(rk,3)*rho0
    # global lho44
    # lho44 = rap.lhoX(rk,4)

    [ rpower, rhopower ] = [ par.rpower_pp, par.rhopower_pp ]
    global const0
    const0 = (rk**rpower)*(rho0**rhopower)
    # global const1
    # const1 = const0/(r4*rho0**4)
    # global const2
    # const2 = ( const1 )**2

    # global lho12
    # lho12 = rap.lhoX(rk,1)*rho0
    # global lho22
    # lho22 = rap.lhoX(rk,2)
    # global lho32
    # lho32 = rap.lhoX(rk,3)/rho0
    # global lho42
    # lho42 = rap.lhoX(rk,4)/rho0**2   

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
    #[ lp_b, lt_b, _  ] = ut.ell(par.m, par.lmax, ut.bsymm)  # the l-indices of the magnetic field
    
    # process each l-component in parallel
    pool = mp.Pool(processes=ncpus)

    #print(np.shape(usol2), np.shape(tsol2))

    if par.hydro:
        ppu = [ pool.apply_async( flow_worker,
                args=( l, )) for l in ll ]
        out_u = np.array([pp0.get() for pp0 in ppu])
    
    # if par.magnetic:
    #     ppb = [ pool.apply_async( magnetic_worker,
    #             args=( l, lp_b, lt_b, bsol2, usol2, Ra, Rb, par.N, sqx)) for l in ll ]   
    #     out_b = np.array([pp0.get() for pp0 in ppb])

    # if par.thermal:
    #     ppt = [ pool.apply_async( thermal_worker,
    #             args=( l, lp_u, tsol2, usol2, Ra, Rb, par.N, sqx, 'thermal' )) for l in lp_u ]   
    #     out_t = np.array([pp0.get() for pp0 in ppt])

    # if par.compositional:
    #     # we use again the thermal_worker but with the compositional solution as argument
    #     ppc = [ pool.apply_async( thermal_worker,
    #             args=( l, lp_u, csol2, usol2, Ra, Rb, par.N, sqx, 'compositional' )) for l in lp_u ]   
    #     out_c = np.array([pp0.get() for pp0 in ppc])

    pool.close()
    pool.join()

    return [ out_u, out_b, out_t, out_c ]
    


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
    # global rho1
    # rho1 = np.exp(rap.logrhoX(rk, 1))

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
