# The diagnose function in this module invokes in parallel the functions
# flow_worker, magnetic_worker, and thermal_worker.
# It provides energy, dissipation, power, etc for a given solution.

import multiprocessing as mp
import numpy.polynomial.chebyshev as ch
import numpy as np
import parameters as par
import utils as ut

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

    if r == None:
        x00 = x0  # use the globally defined x0
    else:
        x00 = xcheb(r, ricb, rcmb)  # use the explicit radial points given as argument
    
    out = np.zeros((np.size(x00), n+1), ck0.dtype)  # n+1 cols
    out[:,0] = ch.chebval(x00, ck0)  # the function itself
    
    if n>0:
        dk = ut.Dn_cheb(ck0, ricb, rcmb, n)  # coeffs for the derivatives, n cols
        for j in range(1,n+1):
            out[:,j] = ch.chebval(x00, dk[:,j-1])  # and the derivatives

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



def cheb2space_pol(L, lp, P, ns):
    '''
    Returns qlm's (radial) and slm's (consoidal) L-components up to derivatives of order ns (<3)
    '''

    L1    = L*(L+1)
    idx   = list(lp).index(L) 
    f_pol = funcheb(P[idx,:], r=None, ricb=par.ricb, rcmb=ut.rcmb, n=ns+1)
    
    plm = []
    qlm = []
    slm = []

    for i in range(ns+2):
        plm.append(f_pol[:,i])
    
    qlm0  = L1*plm[0]/rk
    qlm.append(qlm0)

    slm0 = plm[1] + (plm[0]/rk)
    slm.append(slm0)

    if ns>0:

        qlm1 = (L1*plm[1] - qlm0)/rk
        qlm.append(qlm1)

        slm1 = plm[2] + (qlm1/L1)
        slm.append(slm1)

    if ns>1:

        qlm2 = (L1*plm[2]-2*qlm1)/rk
        qlm.append(qlm2)

        slm2 = plm[3] + (qlm2/L1)
        slm.append(slm2)

    return [qlm, slm]



def cheb2space_tor(L, lt, T, ns):
    '''
    Returns tlm's (toroidal) L-components up to derivatives of order ns (<3)
    '''

    idx   = list(lt).index(L) 
    f_tor = funcheb(T[idx,:], r=None, ricb=par.ricb, rcmb=ut.rcmb, n=ns)

    tlm = []
    for i in range(ns+1):
        tlm.append(f_tor[:,i])

    return tlm



def energy_pol(l, qlm0, slm0):
    '''
    Returns the integrand to compute the poloidal energy, kinetic or magnetic, l-component
    (1/2) ∫ 𝐮⋅𝐮 dV or (1/2) ∫ 𝐛⋅𝐛 dV
    '''
    f0 = 4*np.pi/(2*l+1)
    f1 = r2 * np.absolute( qlm0 )**2
    f2 = r2 * l*(l+1) * np.absolute( slm0 )**2  # r2 is rk**2, a global variable
    return f0*(f1+f2)



def energy_tor(l, tlm0):
    '''
    Returns the integrand to compute the toroidal energy, kinetic or magnetic, l-component
    (1/2) ∫ 𝐮⋅𝐮 dV or (1/2) ∫ 𝐛⋅𝐛 dV
    '''
    f0 = 4*np.pi/(2*l+1)
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



def flow_worker( l, lp, lt, u_sol2, b_sol2, t_sol2, c_sol2, Ra, Rb, N, sqx ):
    '''
    Computes the power balance from the momentum (the Navier-Stokes) equation.
    Includes kinetic energy, kinetic dissipation, internal dissipation, and the
    rate of working (power) of the Lorentz forces and buyancy forces (thermal and compositional).
    l-component
    '''
    
    P = u_sol2[0]
    T = u_sol2[1]
  
    [ kinep, kinet ] = [0, 0]
    [ kindp, kindt ] = [0, 0]
    [ intdp, intdt ] = [0, 0]
    [ wlorp, wlort ] = [0, 0]
    [ wther, wcomp ] = [0, 0]
    
    L = l*(l+1)

    if par.magnetic:
        [ qlmb, slmb, tlmb ] = lorentz4pp(l, b_sol2)  # the l-component of the Lorentz force
        
    if par.thermal:
        hlm0 = buoyancy4pp(l, lp, t_sol2)  # the l-component of the thermal buoyancy force
        
    if par.compositional:
        clm0 = buoyancy4pp(l, lp, c_sol2)  # the l-component of the compositional buoyancy force
        

    if l in lp:

        [ [ qlm0, qlm1, qlm2 ], [ slm0, slm1, slm2 ] ] = cheb2space_pol(l, lp, P, 2)
        
        kinep = energy_pol(l, qlm0, slm0 )
        kindp = diffus_pol(l, qlm0, qlm1, qlm2, slm0, slm1, slm2 )
        intdp = internl_dissip_pol(l, qlm0, qlm1, slm0, slm1 )
        if par.magnetic:
            wlorp = lorentz_power_pol(l, qlm0, slm0, qlmb, slmb )
        if par.thermal:
            wther = buoyancy_power(l, qlm0*rk/(l*(l+1)), hlm0 )
        if par.compositional:
            wcomp = buoyancy_power(l, qlm0*rk/(l*(l+1)), clm0 ) 

    elif l in lt:

        [ tlm0, tlm1, tlm2 ] = cheb2space_tor(l, lt, T, 2)

        kinet = energy_tor(l, tlm0)
        kindt = diffus_tor(l, tlm0, tlm1, tlm2)
        intdt = internl_dissip_tor(l, tlm0, tlm1)
        if par.magnetic:
            wlort = lorentz_power_tor(l, tlm0, tlmb)

    # Integrals
    Kene_l = cg_quad( kinep + kinet, Ra, Rb, N, sqx)
    Dkin_l = cg_quad( kindp + kindt, Ra, Rb, N, sqx)
    Dint_l = cg_quad( intdp + intdt, Ra, Rb, N, sqx)
    Wlor_l = cg_quad( wlorp + wlort, Ra, Rb, N, sqx)
    Wthm_l = cg_quad( wther, Ra, Rb, N, sqx )
    Wcmp_l = cg_quad( wcomp, Ra, Rb, N, sqx )   

    return [ Kene_l, Dkin_l, Dint_l, Wlor_l, Wthm_l, Wcmp_l ]



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



def buoyancy4pp(l, lp, tsol2):
    '''
    Returns the l-component of the buoyancy force.
    Use it to compute the rate of working (power) of buoyancy, either thermal or compositional
    '''

    hlm0 = 0
    
    if l in lp:
        idx   = list(lp).index(l) 
        f_pol = funcheb( tsol2[idx,:], r=None, ricb=par.ricb, rcmb=ut.rcmb, n=0 )
        hlm0  = f_pol[:,0]
        
    return hlm0



def diagnose( usol2, bsol2, tsol2, csol2, Ra, Rb, ncpus):
    '''
    Computes kinetic energy, internal and kinetic energy dissipation,
    and input power from body forces. Integrated From r=Ra to r=Rb, and
    angularly over the whole sphere. Processed in parallel using ncpus.
    '''
    [out_u, out_b] = [0,0]
    [out_t, out_c] = [0,0]

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

    [ lp_u, lt_u, ll ] = ut.ell(par.m, par.lmax, par.symm)  # the l-indices of the flow field
    [ lp_b, lt_b, _  ] = ut.ell(par.m, par.lmax, ut.bsymm)  # the l-indices of the magnetic field
    
    # process each l-component in parallel
    pool = mp.Pool(processes=ncpus)

    #print(np.shape(usol2), np.shape(tsol2))

    if par.hydro:
        ppu = [ pool.apply_async( flow_worker,
                args=( l, lp_u, lt_u, usol2, bsol2, tsol2, csol2, Ra, Rb, par.N, sqx)) for l in ll ]
        out_u = np.array([pp0.get() for pp0 in ppu])
    
    if par.magnetic:
        ppb = [ pool.apply_async( magnetic_worker,
                args=( l, lp_b, lt_b, bsol2, usol2, Ra, Rb, par.N, sqx)) for l in ll ]   
        out_b = np.array([pp0.get() for pp0 in ppb])

    if par.thermal:
        ppt = [ pool.apply_async( thermal_worker,
                args=( l, lp_u, tsol2, usol2, Ra, Rb, par.N, sqx, 'thermal' )) for l in lp_u ]   
        out_t = np.array([pp0.get() for pp0 in ppt])

    if par.compositional:
        # we use again the thermal_worker but with the compositional solution as argument
        ppc = [ pool.apply_async( thermal_worker,
                args=( l, lp_u, csol2, usol2, Ra, Rb, par.N, sqx, 'compositional' )) for l in lp_u ]   
        out_c = np.array([pp0.get() for pp0 in ppc])

    pool.close()
    pool.join()

    return [ out_u, out_b, out_t, out_c ]
