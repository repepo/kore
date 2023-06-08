# utility defs for postprocessing solutions

import multiprocessing as mp
import scipy.fftpack as sft
import numpy.polynomial.chebyshev as ch
import numpy as np
import parameters as par
import utils as ut
import bc_variables as bc

mp.set_start_method('fork')



def xcheb(r,ricb,rcmb):
	# returns points in the appropriate domain of the Cheb polynomial solutions
    # Domain [-1,1] corresponds to [ ricb,rcmb] if ricb>0
    # Domain [-1,1] corresponds to [-rcmb,rcmb] if ricb=0
	Rb = rcmb
	Ra = ricb + (np.sign(ricb)-1)*rcmb
	out = 2*(r-Ra)/(Rb-Ra) - 1

	return out



def funcheb(ck0, r, ricb, rcmb):
    '''
    Returns the function represented by the Chebyshev coeffs ck0, evaluated at the radii r.
    If r is None then uses the x0 points defined globally.
    First column is the function itself, second column is its derivative with respect to r,
    third column is the second derivative, and fourth column is the third derivative.
    Rows correspond to the radial points.
    Use this only when the Cheb coeffs are the full set, i.e. after using expand_sol if ricb=0.
    '''

    ck1 = ut.Dcheb(ck0, ricb, rcmb)  # coeffs for the 1st. derivative
    ck2 = ut.Dcheb(ck1, ricb, rcmb)  # coeffs for the 2nd. derivative
    ck3 = ut.Dcheb(ck2, ricb, rcmb)  # coeffs for the 3rd. derivative

    if r == None:
        x00 = x0  # use the globally defined x0
    else:
        x00 = xcheb(r, ricb, rcmb)  # use the radial points given as argument

    out = np.zeros((np.size(x00), np.size(ck0)), ck0.dtype)
    out[:,0] = ch.chebval(x00, ck0)
    out[:,1] = ch.chebval(x00, ck1)
    out[:,2] = ch.chebval(x00, ck2)
    out[:,3] = ch.chebval(x00, ck3)

    return out



def cg_quad(f, Ra, Rb, N, sqx):
    '''
    Computes the radial integral of f as a Chebyshev-Gauss quadrature.
    Assumes f is sampled over [Ra,Rb], with sqx=np.sqrt(1-xk**2),
    where xk are the radial grid points for the Chebyshev-Guauss quadrature. 
    '''
    out = (np.pi/N) * np.sum( sqx * f ) * (Rb-Ra)/2
    return out



def kinetic_energy_pol(l, qlm0, slm0):
    '''
    Returns the integrand to compute the poloidal kinetic energy, l-component
    '''
    f0 = 4*np.pi/(2*l+1)
    f1 = r2 * np.absolute( qlm0 )**2
    f2 = r2 * l*(l+1) * np.absolute( slm0 )**2  # r2 is rk**2, a global variable
    return f0*(f1+f2)



def kinetic_dissip_pol(l, qlm0, qlm1, qlm2, slm0, slm1, slm2):
    '''
    Returns the integrand to compute the kinetic energy dissipation rate, poloidal l-component
    '''
    L = l*(l+1)
    f0 = 4*np.pi/(2*l+1)
    f1 = L * r2 * np.conj(slm0) * slm2
    f2 = 2 * rk * L * np.conj(slm0) * slm1
    f3 = -(L**2)*( np.conj(slm0)*slm0 ) - (l**2+l+2) * ( np.conj(qlm0)*qlm0 )
    f4 = 2 * rk * np.conj(qlm0)*qlm1 + r2 * np.conj(qlm0) * qlm2
    f5 = 2 * L *( np.conj(qlm0)*slm0 + qlm0*np.conj(slm0) )
    return par.OmgTau * par.Ek * 2*np.real( f0*( f1+f2+f3+f4+f5 ) )



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



def kinetic_energy_tor(l, tlm0):
    '''
    Returns the integrand to compute the toroidal kinetic energy, l-component
    '''
    f0 = 4*np.pi/(2*l+1)
    f1 = r2 * l*(l+1) * np.absolute(tlm0)**2  # r2 is rk**2, a global variable
    return f0*f1



def internl_dissip_tor(l, tlm0, tlm1):
    '''
    Returns the integrand to compute the internal energy dissipation, toroidal l-component
    '''
    L = l*(l+1)
    f0 = 4*np.pi/(2*l+1)
    f1 = L*np.absolute( rk*tlm1-tlm0 )**2
    f2 = L*(l-1)*(l+2)*np.absolute( tlm0 )**2    
    return 2*f0*( f1+f2 )



def kinetic_dissip_tor(l, tlm0, tlm1, tlm2):
    '''
    Returns the integrand to compute the kinetic energy dissipation, toroidal l-component
    '''
    L = l*(l+1)
    f0 = 4*np.pi/(2*l+1)
    f1 = L * r2 * np.conj(tlm0) * tlm2
    f2 = 2 * rk * L * np.conj(tlm0) * tlm1
    f3 = -(L**2)*( np.conj(tlm0)*tlm0 )
    return par.OmgTau * par.Ek * 2*np.real( f0*(f1+f2+f3) )


                    
def expand_sol(sol,vsymm):
    '''
    Expands the ricb=0 solution with ut.N1 coeffs to have full N coeffs,
    filling with zeros according to the equatorial symmetry.
    vsymm=-1 for equatorially antisymmetric, vsymm=1 for symmetric
    '''
    if par.ricb == 0 :

        N  = par.N
        N1 = ut.N1 # N/2 if no IC, N if present
        n  = ut.n

        lm1 = par.lmax-par.m+1

        P0 = sol[:n]
        T0 = sol[n:n+n]

        # these are the cheb coefficients, reorganized
        Plj0 = np.reshape(P0,(int(lm1/2),N1))
        Tlj0 = np.reshape(T0,(int(lm1/2),N1))

        Plj = np.zeros((int(lm1/2),par.N),dtype=complex)
        Tlj = np.zeros((int(lm1/2),par.N),dtype=complex)

        s = int( (vsymm+1)/2 )  # s=0 if vsymm=-1, s=1 if vsymm=1

        iP = (par.m + 1 - s)%2  # even/odd Cheb polynomial for poloidals according to the parity of m+1-s
        iT = (par.m + s)%2
        for k in np.arange(int(lm1/2)) :
            Plj[k,iP::2] = Plj0[k,:]
            Tlj[k,iT::2] = Tlj0[k,:]

        P2 = np.ravel(Plj)
        T2 = np.ravel(Tlj)

        out = np.r_[P2,T2]

    else :
        out = sol

    return out



def thermal_worker(l, Hk0, Pk0, N, ricb, rcmb, Ra, Rb, thermal):

    L  = l*(l+1)

    Hk = np.zeros((1,N),dtype=complex)
    Pk = np.zeros((1,N),dtype=complex)

    if ricb == 0 :
        iP = ( par.m + (1-ut.s) )%2
        Hk[0,iP::2] = Hk0
        Pk[0,iP::2] = Pk0
    else :
        Hk[0,:] = Hk0
        Pk[0,:] = Pk0

    dHk  = ut.Dcheb(Hk[0,:],ricb,rcmb)   # cheb coeffs of the first derivative
    d2Hk = ut.Dcheb(dHk,ricb,rcmb)  # 2nd derivative

    hlm0 = np.zeros(np.shape(x0),dtype=complex)
    hlm1 = np.zeros(np.shape(x0),dtype=complex)
    hlm2 = np.zeros(np.shape(x0),dtype=complex)

    hlm0 = ch.chebval(x0, Hk[0,:])
    hlm1 = ch.chebval(x0, dHk)
    hlm2 = ch.chebval(x0, d2Hk)

    plm0 = np.zeros(np.shape(x0),dtype=complex)

    plm0 = ch.chebval(x0, Pk[0,:])

    f0 = 4 * np.pi / (2 * l + 1)

    # -------------------------------------------------------------------------- buoyancy power

    f1 = r2 * L * np.conj(plm0) * hlm0
    f2 = r2 * L * np.conj(hlm0) * plm0
    Dbuoy_l = 0.5*(Rb-Ra)*(np.pi/N)*np.sum( sqx*f0*(f1+f2) )

    # -------------------------------------------------------------------------- thermal energy

    f1 = r2 * np.conj(hlm0)*hlm0
    Ten_l = 0.5*(Rb-Ra)*(np.pi/N)*np.sum( sqx*f0*(f1) )

    # -------------------------------------------------------------------------- thermal dissipation

    f1 = 2 * rk * np.conj(hlm0) * hlm1
    f2 = r2 * np.conj(hlm0) * hlm2
    f3 = -2 * L * np.conj(hlm0) * hlm0
    f4 = 2 * rk * np.conj(hlm1) * hlm0
    f5 = r2 * np.conj(hlm2) * hlm0
    Dtemp_l = 0.5 * (Rb - Ra) * (np.pi / N) * np.sum(sqx * f0 * (f1 + f2 + f3 + f4 + f5))

    # -------------------------------------------------------------------------- advection term heat equation

    if thermal:

        if par.heating == "internal":
            f1 = r2 * L * np.conj(plm0) * hlm0
            f2 = r2 * L * np.conj(hlm0) * plm0
        elif par.heating == "differential":
            f1 = L * np.conj(plm0) * hlm0/rk
            f2 = L * np.conj(hlm0) * plm0/rk
        elif par.heating == "two zone":
            fr = rk * ut.twozone(rk, par.args)
            f1 = fr * L * np.conj(plm0) * hlm0
            f2 = fr * L * np.conj(hlm0) * plm0
        elif par.heating == "user defined":
            fr = rk * ut.BVprof(rk, par.args)
            f1 = fr * L * np.real(np.conj(plm0) * hlm0)
            f2 = fr * L * np.conj(hlm0) * plm0

    else:

        if par.comp_background == "internal":
            f1 = r2 * L * np.conj(plm0) * hlm0
            f2 = r2 * L * np.conj(hlm0) * plm0
        elif par.comp_background == "differential":
            f1 = L * np.conj(plm0) * hlm0/rk
            f2 = L * np.conj(hlm0) * plm0/rk

    Dadv_l = 0.5 * (Rb - Ra) * (np.pi / N) * np.sum(sqx * f0 * (f1 + f2))

    return [np.real(Dbuoy_l), np.real(Ten_l), np.real(Dtemp_l), np.real(Dadv_l)]



def pol_worker( l, Pk, N, ricb, rcmb, Ra, Rb): # ------------
    '''
    Here we compute various integrals that involve poloidal components, degree l
    We use Chebyshev-Gauss quadratures
    '''
    # plm's are the poloidal scalars and its derivatives evaluated
    # at the grid points in the Cheb polynomial domain of the solution    
    f_pol = funcheb(Pk, r=None, ricb=ricb, rcmb=rcmb)
    plm0 = f_pol[:,0]
    plm1 = f_pol[:,1]
    plm2 = f_pol[:,2]
    plm3 = f_pol[:,3]

    # the radial scalars, rk goes from Ra to Rb
    L = l*(l+1)
    qlm0 = L*plm0/rk
    qlm1 = (L*plm1 - qlm0)/rk
    qlm2 = (L*plm2-2*qlm1)/rk

    # the consoidal scalars
    slm0 = plm1 + (plm0/rk)
    slm1 = plm2 + (qlm1/L)
    slm2 = plm3 + (qlm2/L)

    # Integrands
    f_kep = kinetic_energy_pol(l, qlm0, slm0)  
    f_idp = internl_dissip_pol(l, qlm0, qlm1, slm0, slm1)
    f_kdp = kinetic_dissip_pol(l, qlm0, qlm1, qlm2, slm0, slm1, slm2)
    
    # Integrals
    Kene_pol_l = cg_quad(f_kep, Ra, Rb, N, sqx)
    Dint_pol_l = cg_quad(f_idp, Ra, Rb, N, sqx)
    Dkin_pol_l = cg_quad(f_kdp, Ra, Rb, N, sqx)
        
    power_pol_l = 0  ### To fix later
   
    return [Kene_pol_l, Dint_pol_l, np.real(Dkin_pol_l), np.imag(Dkin_pol_l),\
     np.real(power_pol_l), np.imag(power_pol_l)]



def tor_worker( l, Tk, N, ricb, rcmb, Ra, Rb): # ------------
    '''
    Here we compute various integrals that involve toroidal components, degree l
    We use Chebyshev-Gauss quadratures
    '''
    # tlm's are the toroidal scalars and derivatives evaluated
    # at points in the Cheb polynomial domain of the solution    
    f_tor = funcheb(Tk, r=None, ricb=ricb, rcmb=rcmb)
    tlm0 = f_tor[:,0]
    tlm1 = f_tor[:,1]
    tlm2 = f_tor[:,2]

    # Integrands
    f_ket = kinetic_energy_tor(l, tlm0)
    f_idt = internl_dissip_tor(l, tlm0, tlm1)
    f_kdt = kinetic_dissip_tor(l, tlm0, tlm1, tlm2)
    
    # Integrals
    Kene_tor_l = cg_quad(f_ket, Ra, Rb, N, sqx)
    Dint_tor_l = cg_quad(f_idt, Ra, Rb, N, sqx)
    Dkin_tor_l = cg_quad(f_kdt, Ra, Rb, N, sqx)
 
    power_tor_l = 0  ### To fix later   

    return [Kene_tor_l, Dint_tor_l, np.real(Dkin_tor_l), np.imag(Dkin_tor_l),\
     np.real(power_tor_l), np.imag(power_tor_l)]



def pol_ohm( l, Pk0, N, ricb, rcmb, Ra, Rb): # ------------------------------------------

    Pk = np.zeros((1,N),dtype=complex)

    vsymm = ut.bsymm  # b field symmetry follows from both u and B0
    s = int( (vsymm+1)/2 ) # s=0 if b is antisymm, s=1 if b is symm

    if ricb == 0 :
        iP = ( par.m + 1 - s )%2
        Pk[0,iP::2] = Pk0
    else :
        Pk[0,:] = Pk0

    L  = l*(l+1)

    dPk  = ut.Dcheb(Pk[0,:],ricb,rcmb)
    d2Pk = ut.Dcheb(dPk,ricb,rcmb)

    plm0 = np.zeros(np.shape(x0),dtype=complex)
    plm1 = np.zeros(np.shape(x0),dtype=complex)
    plm2 = np.zeros(np.shape(x0),dtype=complex)

    # this is where the evaluation happens
    plm0 = ch.chebval(x0, Pk[0,:])
    plm1 = ch.chebval(x0, dPk)
    plm2 = ch.chebval(x0, d2Pk)
    '''
    plm0 = np.dot(chx,Pk,plm0)
    plm1 = np.dot(chx,dPk,plm1)
    plm2 = np.dot(chx,d2Pk,plm2)
    '''

    qlm0 = L*plm0/rk
    qlm1 = (L*plm1 - qlm0)/rk

    slm0 = plm1 + (plm0/rk)
    slm1 = plm2 + (qlm1/L)

    # -------------------------------------------------------------------------- magnetic field energy, poloidal

    f0 = 4*np.pi/(2*l+1)
    f1 = r2*np.absolute( qlm0 )**2
    f2 = r2*L*np.absolute( slm0 )**2

    benergy_pol_l = 0.5*(Rb-Ra)*(np.pi/N)*np.sum( sqx*f0*( f1+f2 ) )


    # -------------------------------------------------------------------------- Ohmic dissipation, poloidal

    f0 = 8*np.pi*L/(2*l+1)
    f1 = np.absolute( qlm0 - slm0 - rk*slm1 )**2

    odis_pol_l = 0.5*(Rb-Ra)*(np.pi/N)*np.sum( sqx*f0*f1 )



    return [benergy_pol_l, odis_pol_l]



def tor_ohm( l, Tk0, N, ricb, rcmb, Ra, Rb): # ------------------------------------------


    Tk = np.zeros((1,N),dtype=complex)

    vsymm = ut.bsymm  # b field symmetry follows from both u and B0
    s = int( (vsymm+1)/2 ) # s=0 if b is antisymm, s=1 if b is symm

    # iT is the starting index of the Chebishev polynomials
    # if required Cheb degrees are even: 0,2,4,.. then iT=0
    # if degrees are odd then iT=1

    # required degrees are even if m+1-s is even and vice versa

    if ricb == 0 :
        iT = (par.m + s)%2
        Tk[0,iT::2] = Tk0
    else :
        Tk[0,:] = Tk0

    L  = l*(l+1)

    # Tk is the full set of cheb coefficients for a given l
    # we use the full domain [-1,1] if an inner core is present

    dTk  = ut.Dcheb(Tk[0,:],ricb,rcmb)

    tlm0 = np.zeros(np.shape(x0),dtype=complex)
    tlm1 = np.zeros(np.shape(x0),dtype=complex)

    tlm0 = ch.chebval(x0, Tk[0,:])
    tlm1 = ch.chebval(x0, dTk)
    '''
    tlm0 = np.dot(chx,Tk,tlm0)
    tlm1 = np.dot(chx,dTk,tlm1)
    '''

    # -------------------------------------------------------------------------- magnetic field energy, toroidal

    f0 = 4*np.pi/(2*l+1)
    f1 = r2*L*np.absolute( tlm0 )**2

    benergy_tor_l = 0.5*(Rb-Ra)*(np.pi/N)*np.sum( sqx*f0*( f1 ) )

    # -------------------------------------------------------------------------- Ohmic dissipation, toroidal

    f0 = 8*np.pi*L/(2*l+1)
    f1 = np.absolute( rk*tlm1 + tlm0 )**2
    f2 = L*np.absolute( tlm0 )**2

    odis_tor_l = 0.5*(Rb-Ra)*(np.pi/N)*np.sum( sqx*f0*( f1+f2 ) )


    return [benergy_tor_l, odis_tor_l]



def ken_dis( u_sol, Ra, Rb, ncpus):
    '''
    Computes total kinetic energy, internal and kinetic energy dissipation,
    and input power from body forces.
    '''

    # xk are the grid points for the integration (Gauss-Chebyshev quadrature), from -1 to 1
    i = np.arange(0,par.N)
    xk = np.cos( (i+0.5)*np.pi/par.N )

    # rk are the corresponding radial points, from Ra to Rb
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

    # separate poloidal and toroidal coeffs
    Pk0 = u_sol[ 0    : ut.n   ]
    Tk0 = u_sol[ ut.n : 2*ut.n ]

    # these are the cheb coefficients, reorganized as a 2D array
    # rows are for l's, columns for cheb coeff order
    # rows will be processed in parallel
    Pk0 = np.reshape(Pk0,(int((par.lmax-par.m+1)/2),par.N))
    Tk0 = np.reshape(Tk0,(int((par.lmax-par.m+1)/2),par.N))

    # the l numbers for poloidals and toroidals
    ll0 = ut.ell(par.m, par.lmax, par.symm)
    llpol = ll0[0]
    lltor = ll0[1]

    # process each l component in parallel
    pool = mp.Pool(processes=ncpus)
    p = [ pool.apply_async(pol_worker,
          args=( l, Pk0[row,:], par.N, par.ricb, ut.rcmb, Ra, Rb)) 
          for row,l in enumerate(llpol) ]
    t = [ pool.apply_async(tor_worker, 
          args=( l, Tk0[row,:], par.N, par.ricb, ut.rcmb, Ra, Rb))
          for row,l in enumerate(lltor) ]

    # Sum all l-contributions
    res_pol = np.sum([p1.get() for p1 in p],0)
    res_tor = np.sum([t1.get() for t1 in t],0)

    pool.close()
    pool.join()

    KP = res_pol[0]
    KT = res_tor[0]

    internal_dis = res_pol[1]+res_tor[1]
    rekin_dis = res_pol[2]+res_tor[2]
    imkin_dis = res_pol[3]+res_tor[3]

    repower = res_pol[4]+res_tor[4]
    impower = res_pol[5]+res_tor[5]

    return [KP, KT, internal_dis, rekin_dis, imkin_dis, repower, impower]



def ohm_dis( a, b, N, lmax, m, bsymm, ricb, rcmb, ncpus, Ra, Rb):
    '''
    Computes the total energy in the induced magnetic field and the Ohmic dissipation.
    bsymm is the symmetry of the *induced magnetic field*, which is
    opposed to that of the flow if the applied field is antisymmetric.
    '''

    # xk are the grid points for the integration (Gauss-Chebyshev quadrature), from -1 to 1
    i = np.arange(0,N)
    xk = np.cos( (i+0.5)*np.pi/N )

    # rk are the corresponding radial points, from Ra to Rb
    global rk
    rk = 0.5*(Rb-Ra)*( xk + 1 ) + Ra

    # x0 are the points in the domain of the Cheb poynomial solutions
    global x0
    x0 = xcheb(rk, par.ricb, 1)

    # the following are needed to compute the integrals
    global sqx
    sqx = np.sqrt(1-xk**2)
    global r2
    r2 = rk**2
    global r3
    r3 = rk**3
    global r4
    r4 = rk**4

    n = ut.n
    N1 = ut.N1

    ev0 = a + 1j * b
    Pk0 = ev0[:n]
    Tk0 = ev0[n:n+n]

    # these are the cheb coefficients, reorganized
    Pk0 = np.reshape(Pk0,(int((lmax-m+1)/2),N1))
    Tk0 = np.reshape(Tk0,(int((lmax-m+1)/2),N1))

    ll = ut.ell(m,lmax,bsymm)
    llpol = ll[0]
    lltor = ll[1]

    # process each l component in parallel
    pool = mp.Pool(processes=ncpus)
    p = [ pool.apply_async(pol_ohm,args=(l, Pk0[k,:], N, ricb, rcmb, Ra, Rb)) for k,l in enumerate(llpol) ]
    t = [ pool.apply_async(tor_ohm,args=(l, Tk0[k,:], N, ricb, rcmb, Ra, Rb)) for k,l in enumerate(lltor) ]

    res_pol = np.sum([p1.get() for p1 in p],0)
    res_tor = np.sum([t1.get() for t1 in t],0)

    pool.close()
    pool.join()

    MagEnerPol = res_pol[0]
    MagEnerTor = res_tor[0]

    OhmDissPol = res_pol[1]
    OhmDissTor = res_tor[1]


    return np.real([MagEnerPol, MagEnerTor, OhmDissPol, OhmDissTor])



def thermal_dis( atemp, btemp, au, bu, N, lmax, m, symm, ricb, rcmb, ncpus, Ra, Rb, thermal=True):
    '''
    Returns the power associated with the buoyancy force.
    In the future will also compute thermal dissipation and other
    quantities related to the temperature equation.
    '''

    # # xk are the colocation points, from -1 to 1
    # i = np.arange(0,N)
    # xk = np.cos( (i+0.5)*np.pi/N )
    # global x0
    # x0 = ( (Rb-Ra)*xk + (Ra+Rb) - (ricb+rcmb) )/(rcmb-ricb)

    # # rk are the radial colocation points, from ricb to rcmb
    # global rk
    # rk = 0.5*(rcmb-ricb)*( x0 + 1 ) + ricb
    # # the following are needed to compute the integrals
    # global sqx
    # sqx = np.sqrt(1-xk**2)
    # global r2
    # r2 = rk**2


    # xk are the colocation points for the integration, from -1 to 1
    i = np.arange(0,N)
    xk = np.cos( (i+0.5)*np.pi/N )

    '''
    # x0 are points in [-1,1] mapped to [ricb,rcmb] domain (if ricb>0)
    # or mapped to [-rcmb,rcmb] (if ricb=0)
    global x0
    if ricb > 0 :
        x0 = ( (Rb-Ra)*xk + (Ra+Rb) - (ricb+rcmb) )/(rcmb-ricb)
    else :
        x0 = ( (Rb-Ra)*xk + (Ra+Rb) )/(2*rcmb)

    # rk are the radial colocation points, from Ra to Rb
    global rk
    if ricb > 0:
        rk = 0.5*(rcmb-ricb)*( x0 + 1 ) + ricb
    else :
        rk = rcmb*x0
    '''

    global x0
    x0 = ( (Rb-Ra)*xk + (Ra+Rb) - (ricb+rcmb) )/(rcmb-ricb)
    global rk
    rk = 0.5*(rcmb-ricb)*( x0 + 1 ) + ricb


    # the following are needed to compute the integrals
    global sqx
    sqx = np.sqrt(1-xk**2)
    global r2
    r2 = rk**2
    #print('rk[-1]=',rk[-1])


    # l-indices for u. lup are also the indices for the temperature
    #lmm = 2*np.shape(Plj)[0] -1 # this should be =lmax-m
    lmm = lmax-m
    s = int(symm*0.5+0.5) # s=0 if u is antisymm, s=1 if u is symm
    if m>0:
        lup = np.arange( m+1-s, m+1-s +lmm, 2) # u pol
        #lut = np.arange( m+s  , m+s   +lmm, 2) # u tor
    elif m==0:
        lup = np.arange( 1+s, 1+s +lmm, 2) # u pol
        #lut = np.arange( 2-s, 2-s +lmm, 2) # u tor

    N1 = int( (N/2) * (1 + np.sign(ricb)) )  # N/2 if no IC, N if present
    n = int(N1*(lmax-m+1)/2)

    evtemp = atemp + 1j*btemp
    Hk0 = evtemp[0:  n]

    evu = au + 1j*bu
    Pk0 = evu[0:  n]

    # these are the cheb coefficients, reorganized
    Hk2 = np.reshape(Hk0,(int((lmax-m+1)/2),N1))
    Pk2 = np.reshape(Pk0,(int((lmax-m+1)/2),N1))

    # process each l component in parallel
    pool = mp.Pool(processes=ncpus)
    p = [ pool.apply_async(thermal_worker,args=(l, Hk2[k,:], Pk2[k,:], N, ricb, rcmb, Ra, Rb, thermal)) for k,l in enumerate(lup) ]
    res_pol = np.sum([p1.get() for p1 in p],0)

    pool.close()
    pool.join()

    return res_pol
