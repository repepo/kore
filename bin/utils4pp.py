# The diagnose function in this module invokes the functions
# flow_worker, magnetic_worker, and thermal_worker for each l (serially since 2026-10-16).
# It provides energy, dissipation, power, etc for a given solution.

import multiprocessing as mp
import numpy.polynomial.chebyshev as ch
import numpy as np
from parameters import par
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



# NEW (2026-10-16): all radial functions of a solution at once, for diagnose.
def cheb2space_table(C, nd):
    '''
    Returns the radial functions with Chebyshev coefficients C (one row per l, as from expand_reshape_sol)
    and their r-derivatives up to order nd, at the quadrature points of setup_grid, as an array of
    shape (nd+1, rows of C, points): out[j,i,:] is the j-th derivative of row i.
    Same as funcheb row by row, to rounding: the derivative coefficients come from the same ut.Dcheb
    recurrence, but the Chebyshev sums are one matrix product per derivative (chebvander at x0) instead
    of one chebval (a Python loop over N) per row. The workers of diagnose need each row several times
    (lorentz4pp, induction4pp and pressure4pp also use the neighbouring l's), so they take these tables
    in place of the coefficients (see radial).
    '''
    V   = ch.chebvander(x0, np.shape(C)[1]-1)  # (points, N): T_k at the points
    dk  = np.ascontiguousarray(np.transpose(C))  # (N, rows): one expansion per column, as ut.Dcheb takes them
    out = np.zeros((nd+1, np.shape(C)[0], np.size(x0)), dtype=complex)
    for j in range(nd+1):
        if j > 0:
            dk = ut.Dcheb(dk, par.ricb, ut.rcmb)  # coefficients of the next derivative
        # two real products, half the work of one complex product; contiguous copies so that BLAS is used
        out[j].real = ( V @ np.ascontiguousarray(np.real(dk)) ).T
        out[j].imag = ( V @ np.ascontiguousarray(np.imag(dk)) ).T
    return out



def radial(F, idx, n):
    '''
    NEW (2026-10-16): radial function idx of F and its first n r-derivatives at the points rk, as rows 0..n.
    F holds Chebyshev coefficients (one row per l, evaluated here with funcheb) or is a table
    from cheb2space_table (looked up).
    '''
    if np.ndim(F) == 3:
        if n >= np.shape(F)[0]:
            raise ValueError('radial: the table has only %d derivatives' % (np.shape(F)[0]-1))
        return F[:n+1, idx, :]
    return funcheb(F[idx,:], r=None, ricb=par.ricb, rcmb=ut.rcmb, n=n).T



def cg_quad(f, Ra, Rb, N, sqx):
    '''
    Computes the radial integral of f over [Ra,Rb], with f sampled at the grid points of setup_grid.
    NEW (2026-10-06): the grid is now Gauss-Legendre, and sqx holds the weights times N/pi (see
    setup_grid), so this formula gives sum(w_k*f_k)*(Rb-Ra)/2. Before, the grid was Chebyshev-Gauss
    with sqx = sqrt(1-xk**2). 
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
    f_pol = radial(P, idx, ns+1)  # NEW (2026-10-16): P can be a table from cheb2space_table
    
    plm = []
    qlm = []
    slm = []

    for i in range(ns+2):
        plm.append(f_pol[i])
    
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
    f_tor = radial(T, idx, ns)  # NEW (2026-10-16): T can be a table from cheb2space_table

    tlm = []
    for i in range(ns+1):
        tlm.append(f_tor[i])

    return tlm

def rms(l, qlm0):
    '''
    Returns the r.m.s. value of the radial velocity or magnetic field, l-component
    √ 1/(4πr²)  ∫ (𝐮ᵣ)² dV or √ 1/(4πr²)  ∫ (𝐛ᵣ)² dV
    '''
    f0 = 1/(2*l+1)
    f1 = np.absolute( qlm0 )**2
    return np.sqrt(f0*f1)/rk


def angrymom(u_sol2):
    '''
    Returns the angular momentum in the z direction
    (m=0) in the z direction
    (m=1) in the equatorial direction
    '''
    out = 0
    [ lp, lt, _ ] = ut.ell( par.m, par.lmax, par.symm)

    if par.m == 0 and par.symm == 1:
        T = u_sol2[1]  # gets the toroidal component
        idx = list(lt).index(1)  # this index should always be zero
        if idx != 0:
            print('Hello, I am a bug')
        else:
            t10 = funcheb( T[idx,:], r=None, ricb=par.ricb, rcmb=ut.rcmb, n=0 )[:,0]
            out = cg_quad( (8*np.pi/3)*r3*t10, par.ricb, ut.rcmb, par.N, sqx )
    
    elif par.m == 1 and par.symm == -1:
        T = u_sol2[1]
        idx = list(lt).index(1)  # this index should always be zero
        if idx != 0:
            print('Hello, I am a bug')
        else:
            t11 = funcheb(T[idx, :], r=None, ricb=par.ricb, rcmb=ut.rcmb, n=0)[:, 0]
            out = cg_quad((16 * np.pi / 3) * r3 * t11, par.ricb, ut.rcmb, par.N, sqx)

    return out



# NEW (2026-10-11): z-invariance of u_phi, written by spin_doctor to flow.dat and its summary table.
_geo_grid = None  # (key, grid) of the cylindrical grid of geostrophic_fraction, built once per resolution


def geostrophic_fraction(u_sol2, ns=240, nz=48):
    '''
    Returns the fraction of the energy of u_phi held by its average along the rotation axis:
        geo = ∫|<u_phi>|² dV / ∫|u_phi|² dV,   <u_phi>(s) = (1/H) ∫ u_phi dz along each fluid column of height H.
    The column average and the rest are orthogonal, so 1-geo is the energy fraction of the z-dependent part
    of u_phi; geo = 1 for a z-invariant (geostrophic) u_phi, as in a torsional mode.
    Only for m=0, symm=1, where u_phi = -sum_l t_l(r) dP_l(cos θ)/dθ comes from the toroidal scalars alone
    and is equatorially symmetric; returns nan otherwise. Integrated over the northern half of the fluid,
    on ns midpoint cells in s and nz Gauss-Legendre points in z along each column, from the inner core
    (or the equator, outside the tangent cylinder) up to the CMB. The z-points do not resolve the Ekman
    layers, so geo measures the interior; doubling ns and nz changes it by less than 1e-4 (torsional modes,
    Ek = 1e-7 to 1e-9). u_sol2 comes from expand_reshape_sol.
    '''
    global _geo_grid

    if not (par.m == 0 and par.symm == 1):
        return np.nan

    key = (par.N, par.lmax, par.ricb, ns, nz)
    if _geo_grid is None or _geo_grid[0] != key:
        lt = ut.ell( par.m, par.lmax, par.symm)[1]                     # toroidal l's, in the order of u_sol2[1]
        s  = ut.rcmb*(np.arange(ns) + 0.5)/ns                           # cell centres in s
        z0 = np.sqrt(np.maximum(par.ricb**2 - s**2, 0))                 # column bottom: ICB inside the TC, else z = 0
        H  = np.sqrt(ut.rcmb**2 - s**2) - z0                            # column height (northern half)
        zg, wg = np.polynomial.legendre.leggauss(nz)
        Z  = z0[:,None] + H[:,None]*(zg[None,:] + 1)/2
        W  = H[:,None]*wg[None,:]/2                                     # dz weights
        R  = np.sqrt(s[:,None]**2 + Z**2)
        chx = ch.chebvander( xcheb(R, par.ricb, ut.rcmb).ravel(), par.N-1 )        # Chebyshev T_k at the points
        # dP_l(cos θ)/dθ = P_l^1(cos θ). NEW (2026-10-16): by the upward recurrence in l, for all l at once,
        # instead of scipy.special.lpmv(1, l, cos θ) l by l (17 s at lmax = 1007); same values to 2e-12 relative.
        x   = (Z/R).ravel()                                                       # cos θ
        P1  = np.zeros((max(par.lmax, 1)+1, x.size))
        P1[1] = -np.sqrt(1 - x**2)                                                # P_1^1, Condon-Shortley phase as lpmv
        for l in range(1, par.lmax):
            P1[l+1] = ( (2*l+1)*x*P1[l] - (l+1)*P1[l-1] )/l                       # l P_{l+1}^1 = (2l+1) x P_l^1 - (l+1) P_{l-1}^1
        dP  = P1[lt]
        _geo_grid = ( key, (s, H, W, chx, dP) )
    s, H, W, chx, dP = _geo_grid[1]

    T = u_sol2[1]  # toroidal Chebyshev coefficients, one row per l
    # NEW (2026-10-16): real and imaginary parts as two real products (T @ chx.T made a complex copy of chx)
    Tx    = np.ascontiguousarray(np.real(T)) @ chx.T + 1j*( np.ascontiguousarray(np.imag(T)) @ chx.T )
    uphi  = -np.sum( Tx*dP, axis=0 ).reshape(W.shape)
    ubar  = np.sum( uphi*W, axis=1 )/H                                  # z-average along each column
    e_tot = np.sum( np.abs(uphi)**2 * W * s[:,None] )                   # volume element ∝ s ds dz
    e_geo = np.sum( np.abs(ubar)**2 * H * s )

    return e_geo/e_tot if e_tot > 0 else np.nan



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



def flow_worker( l, lp, lt, u_sol2, b_sol2, t_sol2, c_sol2, Ra, Rb, N, sqx, eigval ):
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

    press_l = pressure4pp(l, eigval, u_sol2)[0]  # the l component of the pressure at the cmb

    return [ Kene_l, Dkin_l, Dint_l, Wlor_l, Wthm_l, Wcmp_l, np.abs(press_l) ]



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
    [ brrms, cmb  ] = [0, 0]

    L = l*(l+1)

    if par.hydro:

        [ qlmi, slmi, tlmi ] = induction4pp(l, u_sol2)  # the l-component of the induction term 

    if l in lp:

        [ [qlm0, qlm1, qlm2], [slm0, slm1, slm2] ] = cheb2space_pol(l, lp, F, 2)

        menep = energy_pol(l, qlm0, slm0)
        mdfsp = diffus_pol(l, qlm0, qlm1, qlm2, slm0, slm1, slm2 )
        indup = dotprod_pol(l, qlm0, slm0, qlmi, slmi)
        brrms = rms(l, qlm0)

    elif l in lt:

        [tlm0, tlm1, tlm2] = cheb2space_tor(l, lt, G, 2)
        
        menet = energy_tor(l, tlm0)
        mdfst = diffus_tor(l, tlm0, tlm1, tlm2)
        indut = dotprod_tor(l, tlm0, tlmi)

   # Integrals
    Mene_l = cg_quad( menep + menet, Ra, Rb, N, sqx)
    Mdfs_l = cg_quad( mdfsp + mdfst, Ra, Rb, N, sqx)
    Indu_l = cg_quad( indup + indut, Ra, Rb, N, sqx)

   # Pick root-mean-square value at CMB radius
    cmb    = np.zeros(N)
    cmb[0] = 1
    Brms_l = sum(brrms*cmb)

    return [ Mene_l, Mdfs_l, Indu_l , Brms_l ]



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

    if   ut.B0_l == 1:
        out = lorentz4pp_dipo(l, b_sol2)
    elif ut.B0_l == 2:
        out = lorentz4pp_quad(l, b_sol2)

    return out



def lorentz4pp_dipo( l, b_sol2 ):
    '''
    Returns the l-component of the Lorentz force.
    Use it to compute the rate of working (power) of the Lorentz force. For dipolar B0 only.
    '''
    
    m   = par.m
    ll0 = ut.ell( m, par.lmax, ut.bsymm)
    lp  = ll0[0]  # l's for poloidals
    lt  = ll0[1]  # l's for toroidals
    
    P = b_sol2[0]
    T = b_sol2[1]

    [ h0, h1, h2, cnorm ] = b0_profiles()  # NEW (2026-10-16): computed once per grid, not for every l

    out_rad = np.zeros_like(rk, dtype='complex128')
    out_con = np.zeros_like(rk, dtype='complex128')
    out_tor = np.zeros_like(rk, dtype='complex128')

    if l-1 in lp:

        [ [ qlm0, _ ], [ slm0, slm1 ] ] = cheb2space_pol(l-1, lp, P, 1)

        C_rad    = (-1 + l)*np.sqrt(l**2 - m**2)/(-1 + 2*l)
        out_rad += C_rad*( -(h0*(qlm0/r2 + slm0/r2 - slm1/rk)) + (h2*slm0 + h1*(-qlm0/rk + 3*slm0/rk + slm1)) )

        C_con    = np.sqrt(l**2 - m**2)/(l*(-1 + 2*l))
        out_con += C_con*( (-2*h0*l*qlm0/r2 + qlm0*(2*h1/rk + h2) + 2*h0*(-1 + l)*(slm0/r2 + slm1/rk)) )

    elif l-1 in lt:

        [ tlm0, tlm1 ] = cheb2space_tor(l-1, lt, T, 1)

        C_tor =  -(-1 + l)*np.sqrt(l**2 - m**2)/(l*(-1 + 2*l))
        out_tor += C_tor*( h0*(-2 + l)*tlm0/r2 + h1*l*tlm0/rk - 2*h0*tlm1/rk )

    if l in lt:
        
        [ tlm0, tlm1 ] = cheb2space_tor(l, lt, T, 1)
        
        out_rad += 1j*m*( h0*(-tlm0/r2 + tlm1/rk) + 3*h1*tlm0/rk + h2*tlm0 + h1*tlm1 )

        out_con += 1j*m*( h0*(2 + l + l**2)*tlm0/r2 + h1*l*(1 + l)*tlm0/rk + 2*h0*tlm1/rk ) / (l*(1 + l))

    elif l in lp:

        [ [ qlm0, _ ], [ slm0, slm1 ] ] = cheb2space_pol(l, lp, P, 1)

        out_tor += 1j*m*( qlm0*(2*h1/rk + h2) - 2*h0*(slm0/r2 + slm1/rk) ) / (l*(1 + l))

    if l+1 in lp:

        [ [ qlm0, _ ], [ slm0, slm1 ] ] = cheb2space_pol(l+1, lp, P, 1)

        C_rad = (2 + l)*np.sqrt((1 + l - m)*(1 + l + m)) / (3 + 2*l)
        out_rad += C_rad*( h0*(qlm0/r2 + slm0/r2 - slm1/rk) - (h2*slm0 + h1*(-qlm0/rk + 3*slm0/rk + slm1)) )
        
        C_con = np.sqrt((1 + l - m)*(1 + l + m)) / ((1 + l)*(3 + 2*l))
        out_con += C_con*( -2*h0*(1 + l)*qlm0/r2 - qlm0*(2*h1/rk + h2) + 2*h0*(2 + l)*(slm0/r2 + slm1/rk) )

    elif l+1 in lt:

        [ tlm0, tlm1 ] = cheb2space_tor(l+1, lt, T, 1)

        C_tor = (2 + l)*np.sqrt((1 + l - m)*(1 + l + m)) / ((1 + l)*(3 + 2*l))
        out_tor += C_tor*( (h0*(3 + l)*tlm0/r2 + h1*(1 + l)*tlm0/rk + 2*h0*tlm1/rk) )

    return [out_rad * cnorm, out_con * cnorm, out_tor * cnorm]



def lorentz4pp_quad( l, b_sol2 ):
    '''
    Returns the l-component of the Lorentz force.
    Use it to compute the rate of working (power) of the Lorentz force. For quadrupolar B0 only.
    '''
    
    m   = par.m
    ll0 = ut.ell( m, par.lmax, ut.bsymm)
    lp  = ll0[0]  # l's for poloidals
    lt  = ll0[1]  # l's for toroidals
    
    P = b_sol2[0]
    T = b_sol2[1]

    [ h0, h1, h2, cnorm ] = b0_profiles()  # NEW (2026-10-16): computed once per grid, not for every l

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
 
    if   ut.B0_l == 1:
        out = induction4pp_dipo(l, u_sol2)
    elif ut.B0_l == 2:
        out = induction4pp_quad(l, u_sol2)

    return out   



def induction4pp_dipo( l, u_sol2 ):
    '''
    Returns the l-component of the induction term ∇×(𝐮×𝐁₀), dipolar B0
    '''
    
    m   = par.m
    ll0 = ut.ell( m, par.lmax, par.symm)
    lp  = ll0[0]  # l's for poloidals
    lt  = ll0[1]  # l's for toroidals
    ricb = par.ricb
    rcmb = ut.rcmb
    
    P = u_sol2[0]
    T = u_sol2[1]

    [ h0, h1, h2, cnorm ] = b0_profiles()  # NEW (2026-10-16): computed once per grid, not for every l

    out_rad = np.zeros_like(rk, dtype='complex128')
    out_con = np.zeros_like(rk, dtype='complex128')
    out_tor = np.zeros_like(rk, dtype='complex128')
    
    if l-1 in lp:

        [ [ qlm0, qlm1], [slm0, slm1] ] =  cheb2space_pol(l-1, lp, P, 1)
           
        out_rad += -(((1 + l)*np.sqrt(l**2 - m**2)*(h1*qlm0*rk + h0*(qlm0 - 2*(-1 + l)*slm0)))/((-1 + 2*l)*r2))

        out_con += (np.sqrt(l**2 - m**2)*(-(qlm1*(h0 + h1*rk)) - qlm0*(2*h1 + h2*rk) + 2*h1*(-1 + l)*slm0 + 2*h0*(-1 + l)*slm1))/(l*(-1 + 2*l)*rk)

    elif l-1 in lt:

        [ tlm0, tlm1]  =  cheb2space_tor(l-1, lt, T, 1)

        out_tor += -(((1 + 2*l)*(h1*(np.sqrt((l*(-1 + l**2))/(-1 + 4*l**2)) - 2*np.sqrt((1 - l**2)/(l - 4*l**3)))*rk*tlm0 + h0*(np.sqrt((l*(-1 + l**2))/(-1 + 4*l**2))*tlm0 - 2*np.sqrt((1 - l**2)/(l - 4*l**3))*rk*tlm1)))/(np.sqrt((l*(1 + l)*(-1 + 4*l**2))/((-1 + l)*(l**2 - m**2)))*r2))

    if l in lt:

        [ tlm0, tlm1]  =  cheb2space_tor(l, lt, T, 1)

        out_rad += (2j*h0*m*tlm0)/r2

        out_con += ((2j)*m*(h1*tlm0 + h0*tlm1))/(l*(1 + l)*rk)

    elif l in lp:

        [ [ qlm0, qlm1], [slm0, slm1] ] =  cheb2space_pol(l, lp, P, 1)

        out_tor += ((-1j)*m*(h2*qlm0*r2 - h0*l*(1 + l)*slm0 + h1*rk*(2*qlm0 + qlm1*rk - (-2 + l + l**2)*slm0) + h0*rk*(qlm1 + 2*slm1)))/(l*(1 + l)*r2)

    if l+1 in lp:

        [ [ qlm0, qlm1], [slm0, slm1] ] =  cheb2space_pol(l+1, lp, P, 1)

        out_rad += (l*np.sqrt((1 + l - m)*(1 + l + m))*(h1*qlm0*rk + h0*(qlm0 + 2*(2 + l)*slm0)))/((3 + 2*l)*r2)

        out_con += (np.sqrt((1 + l - m)*(1 + l + m))*(h2*qlm0*rk + h1*(2*qlm0 + qlm1*rk + 2*(2 + l)*slm0) + h0*(qlm1 + 2*(2 + l)*slm1)))/((1 + l)*(3 + 2*l)*rk)

    elif l+1 in lt:

        [ tlm0, tlm1]  =  cheb2space_tor(l+1, lt, T, 1)

        out_tor +=  ((1 + 2*l)*np.sqrt(((2 + l)*(1 + l - m)*(1 + l + m))/(3 + 4*l*(2 + l)))*(h1*(2*np.sqrt((l*(2 + l))/(3 + 11*l + 12*l**2 + 4*l**3)) + np.sqrt((l*(1 + l)*(2 + l))/(3 + 4*l*(2 + l))))*rk*tlm0 + h0*(np.sqrt((l*(1 + l)*(2 + l))/(3 + 4*l*(2 + l)))*tlm0 + 2*np.sqrt((l*(2 + l))/(3 + 11*l + 12*l**2 + 4*l**3))*rk*tlm1)))/(np.sqrt(l*(1 + l))*r2)
    
    return [out_rad * cnorm, out_con * cnorm, out_tor * cnorm]



def induction4pp_quad( l, u_sol2 ):
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

    [ h0, h1, h2, cnorm ] = b0_profiles()  # NEW (2026-10-16): computed once per grid, not for every l

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
        hlm0  = radial(tsol2, idx, 0)[0]  # NEW (2026-10-16): tsol2 can be a table from cheb2space_table
        
    return hlm0



def pressure4pp(l, eigval, u_sol2):
    '''
    Returns the l-component of the pressure. Does NOT account for magnetic pressure yet
    '''

    m, Ek = par.m, par.Ek

    ll0 = ut.ell( m, par.lmax, par.symm)
    lp  = ll0[0]  # l's for poloidals
    lt  = ll0[1]  # l's for toroidals
    L = l*(l+1)

    P = u_sol2[0]
    T = u_sol2[1]

    out = np.zeros_like(rk, dtype='complex128')

    if l-1 in lt:

        [ tlm0, _ ] = cheb2space_tor(l-1, lt, T, 1)

        C = -2*(l-1)*np.sqrt((l-m)*(l+m))/(l*(2*l-1))
        out += C*rk*tlm0

    if l in lp:

        [ [qlm0, _ , _ ], [ slm0, slm1, slm2 ]] = cheb2space_pol(l, lp, P, 2)

        out += ((2j*m*rk/L) + 2*Ek/rk)*qlm0 + ((2j*m*rk/L) - (Ek*L/rk) - eigval*rk )*slm0 + 2*Ek*slm1 + rk*Ek*slm2

    if l+1 in lt:

        [ tlm0, _ ]  =  cheb2space_tor(l+1, lt, T, 1)

        C = -2*(l+2)*np.sqrt((l+1-m)*(l+1+m))/((l+1)*(2*l+3))
        out += C*rk*tlm0

    return out



_grid = None  # (Ra, Rb, N) of the quadrature grid currently in the module globals
_b0   = None  # NEW (2026-10-16): [h0, h1, h2, cnorm] of B0 on that grid, see b0_profiles


def setup_grid(Ra, Rb):
    '''
    Sets the module-level quadrature grid (rk, x0, sqx, r2, r3, r4) used by the workers.
    '''
    global rk, x0, sqx, r2, r3, r4, _grid, _b0

    # NEW (2026-10-06): Gauss-Legendre nodes xk and weights wk (from -1 to 1) replace the
    # Chebyshev-Gauss nodes. cg_quad used Chebyshev-Gauss with the integrand times sqrt(1-xk**2) to
    # cancel the Chebyshev weight; that product is not smooth at the ends, so the quadrature was only
    # second order and spin_doctor's resid0 (and part of resid u) measured the quadrature error
    # (resid0 ~ 1/N**2, e.g. 1.5e-2 at N = 200). Gauss-Legendre on the same N points is exact for
    # polynomials up to degree 2N-1: resid0 drops to ~1e-11 at the same cost.
    xk, wk = np.polynomial.legendre.leggauss(par.N)

    # rk are the corresponding radial points in the desired integration interval: from Ra to Rb
    rk = 0.5*(Rb-Ra)*( xk + 1 ) + Ra

    # x0 are the points in the appropriate domain of the Chebyshev polynomial solutions
    x0 = xcheb(rk, par.ricb, 1)

    # the following are needed to compute the integrals (i.e. the quadratures)
    sqx = wk*par.N/np.pi  # NEW (2026-10-06): quadrature weights, scaled so that cg_quad's formula is unchanged
    r2 = rk**2
    r3 = rk**3
    r4 = rk**4

    _grid = (Ra, Rb, par.N)
    _b0   = None  # NEW (2026-10-16): B0 profiles are recomputed on the new grid



def b0_profiles():
    '''
    NEW (2026-10-16): returns [h0, h1, h2, cnorm], the radial functions of B0 at the points rk and ut.B0_norm().
    Computed once per grid: lorentz4pp and induction4pp need them for every l, and ut.B0_norm() alone
    takes 10 ms with cnorm = 'mag_energy' (it builds a Gauss-Legendre rule each time).
    '''
    global _b0
    if _b0 is None:
        args = [ par.beta, par.B0_l, par.ricb, 0 ]
        _b0  = [ ut.h0(rk, par.B0, args), ut.h1(rk, par.B0, args), ut.h2(rk, par.B0, args), ut.B0_norm() ]
    return _b0



def diagnose( usol2, bsol2, tsol2, csol2, Ra, Rb, ncpus, eigval, pool=None):
    '''
    Computes kinetic energy, internal and kinetic energy dissipation,
    and input power from body forces. Integrated From r=Ra to r=Rb, and
    angularly over the whole sphere. Returns one row per l for each field.
    NEW (2026-10-16): serial. The radial functions of every l and their derivatives are evaluated once
    (cheb2space_table) and the workers, run one l after another, read them from these tables. Before,
    each l was a task for a pool of ncpus processes that received the full Chebyshev coefficients
    (32 MB at N = 1008) and evaluated each radial function about five times (the neighbouring l's in
    lorentz4pp, induction4pp, pressure4pp): 143 s per solution at N = 1008 on 8 processes, now 2 s.
    Results agree with the old ones to rounding (~1e-14 relative). ncpus and pool are no longer used.
    '''
    [out_u, out_b] = [0,0]
    [out_t, out_c] = [0,0]

    if _grid != (Ra, Rb, par.N):
        setup_grid(Ra, Rb)

    [ lp_u, lt_u, ll ] = ut.ell(par.m, par.lmax, par.symm)  # the l-indices of the flow field
    [ lp_b, lt_b, _  ] = ut.ell(par.m, par.lmax, ut.bsymm)  # the l-indices of the magnetic field

    # radial functions at the points rk: poloidal ones up to the 3rd derivative, toroidal and scalar ones up to the 2nd
    utab = [ cheb2space_table(usol2[0], 3), cheb2space_table(usol2[1], 2) ] if par.hydro else 0
    btab = [ cheb2space_table(bsol2[0], 3), cheb2space_table(bsol2[1], 2) ] if par.magnetic else 0
    ttab = cheb2space_table(tsol2, 2) if par.thermal else 0
    ctab = cheb2space_table(csol2, 2) if par.compositional else 0

    if par.hydro:
        out_u = np.array([ flow_worker( l, lp_u, lt_u, utab, btab, ttab, ctab, Ra, Rb, par.N, sqx, eigval) for l in ll ])

    if par.magnetic:
        out_b = np.array([ magnetic_worker( l, lp_b, lt_b, btab, utab, Ra, Rb, par.N, sqx) for l in ll ])

    if par.thermal:
        out_t = np.array([ thermal_worker( l, lp_u, ttab, utab, Ra, Rb, par.N, sqx, 'thermal' ) for l in lp_u ])

    if par.compositional:
        # we use again the thermal_worker but with the compositional solution as argument
        out_c = np.array([ thermal_worker( l, lp_u, ctab, utab, Ra, Rb, par.N, sqx, 'compositional' ) for l in lp_u ])

    return [ out_u, out_b, out_t, out_c ]



# NEW (2026-10-06): kinetic + magnetic energy of one solution, used by solve_nopp.py to normalise eigenvectors.
def kin_mag_energy(usol2, bsol2, Ra, Rb, ls=None):
    '''
    Returns [KE, ME] of one solution, integrated from r=Ra to r=Rb, with the same definitions
    and routines as diagnose/spin_doctor: KE = (1/2) ∫ 𝐮⋅𝐮 dV (flow_worker), and
    ME = Le2*OmgTau**2 * (1/2) ∫ 𝐛⋅𝐛 dV (magnetic_worker, scaled as in spin_doctor).
    usol2, bsol2 come from expand_reshape_sol (0 if the field is absent). Serial; computes only
    the energies, not the dissipation or power terms, so it is much cheaper than diagnose.
    ls: the l values to include (default all), so that callers can split the sum among processes.
    '''
    setup_grid(Ra, Rb)
    [ lp_u, lt_u, ll ] = ut.ell(par.m, par.lmax, par.symm)
    [ lp_b, lt_b, _  ] = ut.ell(par.m, par.lmax, ut.bsymm)
    KE, ME0 = 0.0, 0.0
    for l in (ll if ls is None else ls):
        if par.hydro:
            if l in lp_u:
                [ [qlm0], [slm0] ] = cheb2space_pol(l, lp_u, usol2[0], 0)
                KE += cg_quad( energy_pol(l, qlm0, slm0), Ra, Rb, par.N, sqx)
            elif l in lt_u:
                [tlm0] = cheb2space_tor(l, lt_u, usol2[1], 0)
                KE += cg_quad( energy_tor(l, tlm0), Ra, Rb, par.N, sqx)
        if par.magnetic:
            if l in lp_b:
                [ [qlm0], [slm0] ] = cheb2space_pol(l, lp_b, bsol2[0], 0)
                ME0 += cg_quad( energy_pol(l, qlm0, slm0), Ra, Rb, par.N, sqx)
            elif l in lt_b:
                [tlm0] = cheb2space_tor(l, lt_b, bsol2[1], 0)
                ME0 += cg_quad( energy_tor(l, tlm0), Ra, Rb, par.N, sqx)
    return [ KE, ME0 * par.OmgTau**2 * par.Le2 if par.magnetic else 0.0 ]



# NEW (2026-10-18): the magnetic terms of the full energy budget (spin_doctor's resid E).
def em_budget(bsol2, bic=None):
    '''
    Returns [Ohm_f, ME_ic, Ohm_ic, E_w, Ohm_w, ME_ext] for one solution, the terms of the magnetic energy budget
    that spin_doctor's fluid-only balance (resid b) leaves out, in the units of magnetic_worker's energy and of
    its integral of b.lap(b) (spin_doctor multiplies the dissipations by OmgTau*Em, and everything by
    OmgTau**2*Le2 to compare with the kinetic energy):
      Ohm_f  : int |curl b|^2 over the fluid (its Ohmic loss is eta times this, eta = 1 in the fluid)
      ME_ic  : (1/2) int |b|^2 inside the inner core: the field of a conducting inner core ('conducting, Chebys',
               bic = its part of the solution vector), or the potential field below an insulating inner core or a
               thin wall ('insulator', 'TWA')
      Ohm_ic : eta_i * int |curl b|^2 over a conducting inner core, eta_i = 1/(mu_i2o*sigma_i2o)
      E_w    : magnetic energy stored in thin walls (innercore and/or mantle = 'TWA'): (c*r/2) int |b_t|^2 dS over
               the poloidal field's b_t, the energy that goes with the c term of the wall condition (the toroidal
               condition has no c term, so the model stores no toroidal energy in the wall)
      Ohm_w  : Ohmic loss of the thin-wall currents, int |K|^2 dS/(c1*r), K the jump of b_t across the wall:
               |g|^2/c1 (toroidal) and c1*|r lap_l f|^2 (poloidal, from the wall conditions)
      ME_ext : energy of the potential field outside the CMB (insulating mantle, or outside a thin wall)
    The wall's b_r jump (c term) and current (c1 term) follow from the thin-wall conditions of bc_b_thinlayer,
    so the boundary terms need no derivatives beyond the second. Assumes mu = 1 (fluid permeability) with walls.
    For ricb = 0 the inner-core terms are zero.
    '''
    r_i, r_o = par.ricb, ut.rcmb
    [ lp, lt, _ ] = ut.ell(par.m, par.lmax, ut.bsymm)
    f0p = 4*np.pi/(2*lp+1); f0t = 4*np.pi/(2*lt+1)
    Lp = lp*(lp+1.); Lt = lt*(lt+1.)

    # Ohmic loss in the fluid, from the tables of the radial functions on the quadrature grid
    Fp = cheb2space_table(bsol2[0], 2); Gt = cheb2space_table(bsol2[1], 1)
    Ohm_f = 0.0
    for k, l in enumerate(lp):
        f, f1, f2 = Fp[0, k], Fp[1, k], Fp[2, k]
        Ohm_f += cg_quad( ohmic_dissip_pol(l, Lp[k]*f/rk, f1 + f/rk, f2 + f1/rk - f/r2), r_i, r_o, par.N, sqx)
    for k, l in enumerate(lt):
        Ohm_f += cg_quad( ohmic_dissip_tor(l, Gt[0, k], Gt[1, k]), r_i, r_o, par.N, sqx)

    # values and r-derivatives of f and g at the boundaries (x = -1 at the ICB, x = 1 at the CMB)
    def bvals(C, nd, x):
        dk = np.ascontiguousarray(np.transpose(C)); out = []
        for j in range(nd+1):
            if j > 0: dk = ut.Dcheb(dk, r_i, r_o)
            out.append(ch.chebval(x, dk))
        return out

    ME_ic = Ohm_ic = E_w = Ohm_w = ME_ext = 0.0

    # ------------------------------------------------------------------------------------------------ CMB side
    f, f1, f2 = bvals(bsol2[0], 2, 1.0); g = bvals(bsol2[1], 0, 1.0)[0]
    r = r_o
    F, F1 = r*f, f + r*f1
    DF = 2*f1 + r*f2 - Lp*f/r            # r*lap_l(f) = F'' - L F/r^2
    if par.mantle == 'TWA':
        c, c1 = par.c_cmb, par.c1_cmb
        Fout = -(r*F1 + c1*r**2*DF)/lp     # field beyond the wall, from the wall condition
        s = F1/r                           # tangential poloidal field on the fluid side
        E_w += np.sum( c*r**3*f0p*Lp*np.abs(s)**2 )
        if c1 > 0:
            Ohm_w += np.sum( 2*f0p*r*Lp*c1*np.abs(DF)**2 ) + np.sum( 2*f0t*r*Lt*np.abs(g)**2/c1 )
    else:
        Fout = F
    ME_ext = np.sum( f0p*Lp*lp*np.abs(Fout/r)**2 )   # potential field r^-(l+1) outside r = rcmb = 1

    # ------------------------------------------------------------------------------------------------ ICB side
    if r_i > 0:
        r = r_i
        if par.innercore in ['insulator', 'TWA']:
            f, f1, f2 = bvals(bsol2[0], 2, -1.0); g = bvals(bsol2[1], 0, -1.0)[0]
            F, F1 = r*f, f + r*f1
            DF = 2*f1 + r*f2 - Lp*f/r
            if par.innercore == 'TWA':
                c, c1 = par.c_icb, par.c1_icb
                Fin = (r*F1 - c1*r**2*DF)/(lp+1)  # field below the wall, from the wall condition
                s = F1/r
                E_w += np.sum( c*r**3*f0p*Lp*np.abs(s)**2 )
                if c1 > 0:
                    Ohm_w += np.sum( 2*f0p*r*Lp*c1*np.abs(DF)**2 ) + np.sum( 2*f0t*r*Lt*np.abs(g)**2/c1 )
            else:
                Fin = F
            ME_ic = np.sum( f0p*Lp*(lp+1)*r*np.abs(Fin/r)**2 )   # potential field r^l inside r = ricb

        elif par.innercore == 'conducting, Chebys' and bic is not None:
            # the inner-core field: Chebyshev series in x = r/ricb on [-1, 1], one parity per l
            Nic = ut.Nic; nb = int((par.lmax_cic - par.m + 1)/2)
            lpc, ltc = ut.ell(par.m, par.lmax_cic, ut.bsymm)[:2]
            if ut.symmB0 == -1: ixf, ixg = (par.m + ut.s)%2, (par.m + 1 - ut.s)%2
            else:               ixf, ixg = (par.m + 1 - ut.s)%2, (par.m + ut.s)%2
            xg, wg = np.polynomial.legendre.leggauss(max(2*par.N_cic, 64))
            ri = 0.5*r*(xg + 1); wi = 0.5*r*wg
            def icfun(a, ix, nd):
                cc = np.zeros(par.N_cic, complex); cc[ix::2] = a; out = []
                for j in range(nd+1):
                    out.append(ch.chebval(ri/r, cc)); cc = ch.chebder(cc)/r
                return out
            eta_i = 1/(par.mu_i2o*par.sigma_i2o)
            for k, l in enumerate(lpc):
                L = l*(l+1.); a0 = 4*np.pi/(2*l+1)
                f, f1, f2 = icfun(bic[k*Nic:(k+1)*Nic], ixf, 2)
                q0, s0 = L*f/ri, f1 + f/ri
                ME_ic  += np.sum( wi*a0*ri**2*( np.abs(q0)**2 + L*np.abs(s0)**2 ) )
                Ohm_ic += eta_i*np.sum( wi*2*a0*L*ri**2*np.abs(f2 + 2*f1/ri - L*f/ri**2)**2 )
            for k, l in enumerate(ltc):
                L = l*(l+1.); a0 = 4*np.pi/(2*l+1)
                g, g1 = icfun(bic[nb*Nic + k*Nic: nb*Nic + (k+1)*Nic], ixg, 1)
                ME_ic  += np.sum( wi*a0*ri**2*L*np.abs(g)**2 )
                Ohm_ic += eta_i*np.sum( wi*2*a0*L*( L*np.abs(g)**2 + np.abs(ri*g1 + g)**2 ) )
        else:
            ME_ic = np.nan  # 'conducting, Bessel' (forced problems): not coded

    return [ Ohm_f, ME_ic, Ohm_ic, E_w, Ohm_w, ME_ext ]
