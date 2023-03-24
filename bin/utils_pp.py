# utility for postprocessing solutions

import multiprocessing as mp
import scipy.fftpack as sft
import numpy.polynomial.chebyshev as ch
import numpy as np

import parameters as par
import utils as ut
import bc_variables as bc

mp.set_start_method('fork')


def xcheb(r,ricb,rcmb):
	# returns points in the domain of the Cheb polynomial solutions
    # [-1,1] corresponds to [ ricb,rcmb] if ricb>0
    # [-1,1] corresponds to [-rcmb,rcmb] if ricb=0
	Rb = rcmb
	Ra = ricb + (np.sign(ricb)-1)*rcmb
	out = 2*(r-Ra)/(Rb-Ra) - 1

	return out


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


def thermal_worker(l, Hk0, Pk0, N, ricb, rcmb, Ra, Rb):

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

    hlm0 = np.zeros(np.shape(x0),dtype=complex)
    hlm0 = ch.chebval(x0, Hk[0,:])

    plm0 = np.zeros(np.shape(x0),dtype=complex)
    plm0 = ch.chebval(x0, Pk[0,:])

    # -------------------------------------------------------------------------- buoyancy power
    f0 = 4*np.pi*L/(2*l+1)
    f1 = r2 * 2 * np.real( np.conj(plm0) * hlm0 )
    buoyancy_power_l = 0.5*(Rb-Ra)*(np.pi/N)*np.sum( sqx*f0*f1 )

    #if l<10:
    #    print(l,buoyancy_power_l)

    return [buoyancy_power_l]


def pol_worker( l, Pk0, N, m, ricb, rcmb, w, projection, forcing, Ra, Rb): # ------------
    '''
    Here we compute various integrals that involve poloidal components, degree l
    We use Chebyshev-Gauss quadratures
    '''
    Pk = np.zeros((1,N),dtype=complex)

    # iP is the starting index of the Chebishev polynomials
    # if required Cheb degrees are even: 0,2,4,.. then iP=0
    # if degrees are odd then iP=1

    # required degrees are even if m+1-s is even and vice versa

    if ricb == 0 :
        iP = ( m + 1 - ut.s )%2
        Pk[0,iP::2] = Pk0  # expand the solution so that it has N coeffs
    else :
        Pk[0,:] = Pk0

    L  = l*(l+1)

    dPk  = ut.Dcheb(Pk[0,:],ricb,rcmb)   # cheb coeffs of the first derivative
    d2Pk = ut.Dcheb(dPk,ricb,rcmb)  # 2nd derivative
    d3Pk = ut.Dcheb(d2Pk,ricb,rcmb) # 3rd derivative

    # plm's are the poloidal scalars evaluated at points in the Cheb polynomial domain of the solution

    plm0 = np.zeros(np.shape(x0),dtype=complex)
    plm1 = np.zeros(np.shape(x0),dtype=complex)
    plm2 = np.zeros(np.shape(x0),dtype=complex)
    plm3 = np.zeros(np.shape(x0),dtype=complex)

    plm0 = ch.chebval(x0, Pk[0,:])
    plm1 = ch.chebval(x0, dPk)
    plm2 = ch.chebval(x0, d2Pk)
    plm3 = ch.chebval(x0, d3Pk)
    '''
    plm0 = np.dot(chx,Pk,plm0)
    plm1 = np.dot(chx,dPk,plm1)
    plm2 = np.dot(chx,d2Pk,plm2)
    plm3 = np.dot(chx,d3Pk,plm3)
    '''

    # the radial scalars, rk goes from Ra to Rb
    qlm0 = L*plm0/rk
    qlm1 = (L*plm1 - qlm0)/rk
    qlm2 = (L*plm2-2*qlm1)/rk

    # the consoidal scalars
    slm0 = plm1 + (plm0/rk)
    slm1 = plm2 + (qlm1/L)
    slm2 = plm3 + (qlm2/L)



    # -------------------------------------------------------------------------- kinetic energy, poloidal
    # Volume integral of (1/2)* u.u

    f0 = 4*np.pi/(2*l+1)
    f1 = r2*np.absolute( qlm0 )**2
    f2 = r2*L*np.absolute( slm0 )**2

    # for the integrals we use Chebyshev-Gauss quadratures
    Ken_pol_l = (np.pi/N)*(Rb-Ra)*0.5*np.sum( sqx*f0*( f1+f2) )


    # -------------------------------------------------------------------------- internal energy dissipation, poloidal
    # Volume integral of (symm\nabla u):(symm\nabla u)

    f0 = 4*np.pi/(2*l+1)
    f1 = L*np.absolute(qlm0 + rk*slm1 - slm0)**2
    f2 = 3*np.absolute(rk*qlm1)**2
    f3 = L*(l-1)*(l+2)*np.absolute(slm0)**2
    # integral is a positive real number, take 2* to match Dkin
    Dint_pol_l = 2*(np.pi/N)*(Rb-Ra)*0.5*np.sum( sqx*f0*( f1+f2+f3 ) )


    # -------------------------------------------------------------------------- kinetic energy dissipation rate, poloidal
    # Volume integral of u.nabla^2 u, (as a complex number, use 2*real part)

    f0 = 4*np.pi/(2*l+1)
    f1 = L * r2 * np.conj(slm0) * slm2
    f2 = 2 * rk * L * np.conj(slm0) * slm1
    f3 = -(L**2)*( np.conj(slm0)*slm0 ) - (l**2+l+2) * ( np.conj(qlm0)*qlm0 )
    f4 = 2 * rk * np.conj(qlm0)*qlm1 + r2 * np.conj(qlm0) * qlm2
    f5 = 2 * L *( np.conj(qlm0)*slm0 + qlm0*np.conj(slm0) )
    # kinetic energy dissipation is 2*real part of the integral
    Dkin_pol_l = 2*np.real( (np.pi/N)*(Rb-Ra)*0.5*np.sum( sqx*f0*( f1+f2+f3+f4+f5 ) ) )
    '''
    if l<4:
        print('')
        print('K = ', Ken_pol_l)
        print('Dint = ',Dint_pol_l)
        print('Dkin =',Dkin_pol_l)
        print('')
    '''

    if (projection == 1 and forcing == 0) or forcing == 1: # ------------------- power from Lin2018 forcing, poloidal

        if l==2 and m==2:
            f0 = (8*np.pi/35)/(ricb**5-1)
            #f1 = 7j* np.conj(qlm0) *( r3   +  (ricb**5)/r2 )
            #f2 = 7j* np.conj(slm0) *( 3*r3 - 2*(ricb**5)/r2 )
            f1 = 7j * r3 * ( np.conj(qlm0) + 3*np.conj(slm0) )
            f2 = 7j *(ricb**5)* ( np.conj(qlm0) - 2*np.conj(slm0) )/r2
            power_pol_l =  (np.pi/N)*(Rb-Ra)*0.5*np.sum( sqx*f0*( f1+f2 ) )
        else:
            power_pol_l =  0

    elif (projection == 3 and forcing == 0) or forcing == 3: # ----------------- power from Marc's forcing, poloidal

        C0 = 1j; C1 = 1; C2 = 1j

        if l == 2:
            if m == 0:
                f0 = (4*np.pi/105)*C0
                f1 = -21j*w*r4*np.conj(qlm0)
            elif m == 1:
                f0 = (4*np.pi/105)*C1
                f1 = -21j*w*r4*np.conj(qlm0) + 42j*r4*np.conj(slm0)
            elif m == 2:
                f0 = (-4j*np.pi/35)*C2
                f1 = 7*w*r4*np.conj(qlm0) - 28*r4*np.conj(slm0)
            else:
                f0 = 0
                f1 = 0
            power_pol_l = (np.pi/N)*(Rb-Ra)*0.5*np.sum( sqx*f0*( f1 ) )
        else:
            power_pol_l = 0

    elif (projection == 4 and forcing == 0) or forcing == 4: # ----------------- power from forcing test 1, poloidal

        X = ut.ftest1(ricb)
        XA = X[0]
        XB = X[1]
        XC = X[2]

        if l==2 and m==2:
            f0 = -8j*np.pi*XC/35
            f1 = 7 * XA * r3 * ( np.conj(qlm0) + 3*np.conj(slm0) )
            f2 = 7 * XB * ( np.conj(qlm0) - 2*np.conj(slm0) )/r2
            power_pol_l = (np.pi/N)*(Rb-Ra)*0.5*np.sum( sqx*f0*( f1 + f2 ) )
        else:
            power_pol_l =  0

    elif (projection == 5 and forcing == 0) or forcing == 5: # ----------------- power from forcing test 2, poloidal
        power_pol_l = 0

    else:
        power_pol_l = 0


    return [Ken_pol_l, Dint_pol_l, np.real(Dkin_pol_l), np.imag(Dkin_pol_l),\
     np.real(power_pol_l), np.imag(power_pol_l)]




def tor_worker( l, Tk0, N, m, ricb, rcmb, w, projection, forcing, Ra, Rb): # ------------
    '''
    Here we compute integrals that involve toroidal components, degree l
    Same way as for the poloidals above
    '''

    Tk = np.zeros((1,N),dtype=complex)

    if ricb == 0 :
        iT = (m + ut.s)%2
        Tk[0,iT::2] = Tk0  # expand the solution so that it has N coeffs
    else :
        Tk[0,:] = Tk0

    L  = l*(l+1)

    dTk  = ut.Dcheb(Tk[0,:],ricb,rcmb)
    d2Tk = ut.Dcheb(dTk,ricb,rcmb)

    tlm0 = np.zeros(np.shape(x0),dtype=complex)
    tlm1 = np.zeros(np.shape(x0),dtype=complex)
    tlm2 = np.zeros(np.shape(x0),dtype=complex)

    tlm0 = ch.chebval(x0, Tk[0,:])
    tlm1 = ch.chebval(x0, dTk)
    tlm2 = ch.chebval(x0, d2Tk)
    '''
    tlm0 = np.dot(chx,Tk,tlm0)
    tlm1 = np.dot(chx,dTk,tlm1)
    tlm2 = np.dot(chx,d2Tk,tlm2)
    '''


    # -------------------------------------------------------------------------- kinetic Energy, toroidal

    f0 = 4*np.pi/(2*l+1)
    f1 = (r2)*L*np.absolute(tlm0)**2
    # integral is:
    Ken_tor_l = (np.pi/N)*(Rb-Ra)*0.5 * np.sum( sqx*f0*( f1 ) )
    #if l<5:
    #    print('l, KT = ',l,Ken_tor_l)
    #    #print('ken_tor_l = ', Ken_tor_l)


    # -------------------------------------------------------------------------- internal energy dissipation, toroidal

    f0 = 4*np.pi/(2*l+1)
    f1 = L*np.absolute( rk*tlm1-tlm0 )**2
    f2 = L*(l-1)*(l+2)*np.absolute( tlm0 )**2
    # integral is
    Dint_tor_l =  2* (np.pi/N)*(Rb-Ra)*0.5 * np.sum( sqx*f0*( f1+f2 ) )


    # -------------------------------------------------------------------------- kinetic energy dissipation rate, toroidal

    f0 = 4*np.pi/(2*l+1)
    f1 = L * r2 * np.conj(tlm0) * tlm2
    f2 = 2 * rk * L * np.conj(tlm0) * tlm1
    f3 = -(L**2)*( np.conj(tlm0)*tlm0 )
    # integral is:
    Dkin_tor_l = 2*np.real( (np.pi/N) * (Rb-Ra)*0.5 * np.sum( sqx*f0*( f1+f2+f3 ) ) )
    '''
    if l<4:
        print('')
        print('Ktor = ', Ken_tor_l)
        print('Dint = ',Dint_tor_l)
        print('Dkin =',Dkin_tor_l)
        print('')
    '''
    if (projection == 1 and forcing == 0) or forcing == 1: # ------------------- power from Lin2018 forcing, toroidal

        if l==3 and m==2:
            f0 = (8*np.pi/35)/(ricb**5-1)
            f1 = 10*np.sqrt(5)*(ricb**5)* np.conj(tlm0) /r2
            power_tor_l =  (np.pi/N)*(Rb-Ra)*0.5*np.sum( sqx*f0*( f1 ) )
        else:
            power_tor_l =  0


    elif (projection == 3 and forcing == 0) or forcing == 3: # ----------------- power from Marc's forcing, toroidal

        C0 = 1j; C1 = 1; C2 = 1j

        if l == 1:
            if m == 0:
                f0 = (4*np.pi/105)*C0
                f1 = 28*r4*np.conj(tlm0)
            elif m == 1:
                f0 = (4*np.pi/105)*C1
                f1 = 14*np.sqrt(3)*r4*np.conj(tlm0)
            else:
                f0 = 0
                f1 = 0
            power_tor_l = (np.pi/N)*(Rb-Ra)*0.5*np.sum( sqx*f0*( f1 ) )

        elif l == 3:
            if m == 0:
                f0 = (4*np.pi/105)*C0
                f1 = -72*r4*np.conj(tlm0)
            elif m == 1:
                f0 = (4*np.pi/105)*C1
                f1 = -48*np.sqrt(2)*r4*np.conj(tlm0)
            elif m == 2:
                f0 = (-4j*np.pi/35)*C2
                f1 = -8j*np.sqrt(5)*r4*np.conj(tlm0)
            else:
                f0 = 0
                f1 = 0
            power_tor_l = (np.pi/N)*(Rb-Ra)*0.5*np.sum( sqx*f0*( f1 ) )

        else:
            power_tor_l = 0


    elif (projection == 4 and forcing == 0) or forcing == 4: # ----------------- power from forcing test 1, toroidal

        X = ut.ftest1(ricb)
        XA = X[0]
        XB = X[1]
        XC = X[2]

        if l==3 and m==2:
            f0 = -8j*np.pi*XC/35
            f1 = -10j*np.sqrt(5)*XB*np.conj(tlm0)/r2
            power_tor_l = (np.pi/N)*(Rb-Ra)*0.5*np.sum( sqx*f0*( f1 ) )
        else:
            power_tor_l =  0


    elif (projection == 5 and forcing == 0) or forcing == 5: # ----------------- Power from forcing test 2, toroidal

        if l==1 and m==0:
            f0 = -16*np.pi/105
            f1 = 7 * np.conj(tlm0)
            power_tor_l = (np.pi/N)*(Rb-Ra)*0.5*np.sum( sqx*f0*( f1 ) )
            print('Tor Power =',power_tor_l)
        elif l==3 and m==0:
            f0 = -16*np.pi/105
            f1 = 27* np.conj(tlm0)
            power_tor_l = (np.pi/N)*(Rb-Ra)*0.5*np.sum( sqx*f0*( f1 ) )
            print('Tor Power =',power_tor_l)
        else:
            power_tor_l =  0


    elif forcing == 7:  # ------------------------------------- Power from longitudinal libration, boundary flow forcing

        if l==1 and m==0:

            T0icb = bc.Ta[:,0]
            T0cmb = bc.Tb[:,0]
            T1icb = bc.Ta[:,1]
            T1cmb = bc.Tb[:,1]

            icb_torque = (16*np.pi/3)*(par.ricb**2)*np.imag( np.dot( par.ricb*T1icb - T0icb, Tk[0,:] ) )
            cmb_torque = (16*np.pi/3)*np.imag( np.dot( T1cmb - T0cmb, Tk[0,:] ) )

            pow_icb = (par.forcing_frequency)*par.forcing_amplitude_icb * icb_torque * par.Ek
            #print('pow_icb=',pow_icb)
            pow_cmb = (par.forcing_frequency)*par.forcing_amplitude_cmb * cmb_torque * par.Ek
            #print('pow_cmb=',pow_cmb)

            power_tor_l = np.real(pow_cmb) - np.real(pow_icb)  # net power is cmb power - icb power

        else:
            power_tor_l = 0


    elif forcing == 8:  # ----------------------------- Power from Poincaré force (longitudinal libration), body forcing

        if l==1 and m==0:
            f0 = 8*np.pi/3
            f1 = 2*np.real( tlm0 ) * r3
            power_tor_l = (np.pi/N)*(Rb-Ra)*0.5*np.sum( sqx*f0*( f1 ) ) * (par.forcing_frequency**2)*par.forcing_amplitude_cmb
        else:
            power_tor_l = 0


    else:
        power_tor_l = 0


    return [Ken_tor_l, Dint_tor_l, np.real(Dkin_tor_l), np.imag(Dkin_tor_l),\
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



def ken_dis( a, b, N, lmax, m, symm, ricb, rcmb, ncpus, w, projection, forcing, Ra, Rb):
    '''
    Computes total kinetic energy, internal and kinetic energy dissipation,
    and input power from body forces.
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

    # the following are needed to compute the integrals (i.e. the quadratures)
    global sqx
    sqx = np.sqrt(1-xk**2)
    global r2
    r2 = rk**2
    global r3
    r3 = rk**3
    global r4
    r4 = rk**4

    n=ut.n
    N1=ut.N1

    ev0 = a + 1j * b
    Pk0 = ev0[:n]
    Tk0 = ev0[n:n+n]

    # these are the cheb coefficients, reorganized
    Pk0 = np.reshape(Pk0,(int((lmax-m+1)/2),N1))
    Tk0 = np.reshape(Tk0,(int((lmax-m+1)/2),N1))

    ll0 = ut.ell(m,lmax,symm)
    llpol = ll0[0]
    lltor = ll0[1]


    # process each l component in parallel
    pool = mp.Pool(processes=ncpus)

    p = [ pool.apply_async(pol_worker, args=( l, Pk0[k,:], N, m, ricb, rcmb, w, projection, forcing, Ra, Rb))\
     for k,l in enumerate(llpol) ]

    t = [ pool.apply_async(tor_worker, args=( l, Tk0[k,:], N, m, ricb, rcmb, w, projection, forcing, Ra, Rb))\
     for k,l in enumerate(lltor) ]

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
    p = [ pool.apply_async(thermal_worker,args=(l, Hk2[k,:], Pk2[k,:], N, ricb, rcmb, Ra, Rb)) for k,l in enumerate(lup) ]
    res_pol = np.sum([p1.get() for p1 in p],0)

    pool.close()
    pool.join()

    if thermal:
        buoy_power = -(par.Ra/par.Prandtl)*(par.Ek**2)*res_pol[0]
    else:
        buoy_power = -(par.Ra_comp/par.Prandtl)*(par.Ek**2)*res_pol[0]

    return [buoy_power]
