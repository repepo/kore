# The diagnose function in this module invokes the functions
# flow_worker and thermal_worker for each l (serially since 2026-10-20).
# It provides energy, dissipation, power, etc for a given solution.

import multiprocessing as mp
import numpy.polynomial.chebyshev as ch
import numpy as np
from parameters import par
import utils as ut
import radial_profiles as rap
import angular
import diff_rot as dr
import background_field as bf


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









# NEW (2026-10-20): all radial functions of a solution at once, ported from the conductive_IC branch (2026-10-16).
def cheb2space_table(C, nd, r):
    '''
    Returns the radial functions with Chebyshev coefficients C (one row per l, as from expand_reshape_sol)
    and their r-derivatives up to order nd, at the radii r, as an array of shape (nd+1, rows of C, radii):
    out[j,i,:] is the j-th derivative of row i.
    Same as funcheb row by row, to rounding: the derivative coefficients come from the same ut.Dcheb
    recurrence, but the Chebyshev sums are one matrix product per derivative (chebvander at r) instead
    of one chebval (a Python loop over N) per row. The workers need each row several times (coriolis
    also uses the neighbouring l's), so they take these tables in place of the coefficients (see radial).
    '''
    V   = ch.chebvander(xcheb(r, par.ricb, ut.rcmb), np.shape(C)[1]-1)  # (radii, N): T_k at the radii
    dk  = np.ascontiguousarray(np.transpose(C))  # (N, rows): one expansion per column, as ut.Dcheb takes them
    out = np.zeros((nd+1, np.shape(C)[0], np.size(r)), dtype=complex)
    for j in range(nd+1):
        if j > 0:
            dk = ut.Dcheb(dk, par.ricb, ut.rcmb)  # coefficients of the next derivative
        # two real products, half the work of one complex product; contiguous copies so that BLAS is used
        out[j].real = ( V @ np.ascontiguousarray(np.real(dk)) ).T
        out[j].imag = ( V @ np.ascontiguousarray(np.imag(dk)) ).T
    return out



def radial(F, idx, n, radii):
    '''
    NEW (2026-10-20): radial function idx of F and its first n r-derivatives, as rows 0..n.
    F holds Chebyshev coefficients (one row per l, evaluated here with funcheb at radii) or is a table
    from cheb2space_table (looked up; set_solution builds the tables at the current rk).
    '''
    if np.ndim(F) == 3:
        if n >= np.shape(F)[0]:
            raise ValueError('radial: the table has only %d derivatives' % (np.shape(F)[0]-1))
        return F[:n+1, idx, :]
    return funcheb(F[idx,:], r=radii, ricb=par.ricb, rcmb=ut.rcmb, n=n).T



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
    f_pol = radial(P, idx, ns+1, radii)  # NEW (2026-10-20): P can be a table from cheb2space_table
    
    plm0 = f_pol[0]
    plm1 = f_pol[1]
    
    qlm = []
    slm = []
    
    qlm0  = (L*plm0)/rr
    qlm.append(qlm0)

    slm0 = plm1 + plm0*(lho1 + 1/rr)
    slm.append(slm0)

    if ns>0:

        plm2 = f_pol[2]

        qlm1 = (L*(-plm0 + plm1*rr))/rr2
        qlm.append(qlm1)

        slm1 = plm2 + plm0*(lho2 - (1/rr2)) + plm1*(lho1 + 1/rr)
        slm.append(slm1)

    if ns>1:

        plm3 = f_pol[3]

        qlm2 = (L*(2*plm0 + rr*(-2*plm1 + plm2*rr)))/rr3
        qlm.append(qlm2)

        slm2 = plm3 + plm0*(lho3 + 2/rr3) + 2*plm1*(lho2 - (1/rr2)) + plm2*(lho1 + 1/rr)
        slm.append(slm2)

    if ns>2:

        plm4 = f_pol[4]
    
        qlm3 = (L*(-6*plm0 + rr*(6*plm1 + rr*(-3*plm2 + plm3*rr))))/rr4
        qlm.append(qlm3)

        slm3 = plm4 + plm0*(lho4 - 6/rr4) + 3*plm1*(lho3 + 2/rr3) + 3*plm2*(lho2 - 1/rr2) + plm3*(lho1 + 1/rr)
        slm.append(slm3)

    if ns>3:

        plm5 = f_pol[5]

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
    f_tor = radial(T, idx, ns, radii)  # NEW (2026-10-20): T can be a table from cheb2space_table

    tlm = []
    for i in range(ns+1):
        tlm.append(f_tor[i])

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



def thermal_wind_entropy():
    '''
    Returns [σₙ, σₙ′] at the radii rk for n in diff_rot.tw_degrees(), S_tw = Σₙ σₙ(r) Pₙ(cosθ) the thermal-wind
    background entropy (docs/differential_rotation_details.md, eq. 2.9), computed here independently of diff_rot.py's
    algebra (numpy Legendre series at each radius) so that the entropy budget checks the assembled heat equation:
        ∂S/∂x = -c q [x F_r + (1-x²) F_x/r],   c = G²/Beyonce, q = r²/g,   F = (1+ΔΩ)² - 1 (thermal_wind = 2) or 2ΔΩ (1),
    integrated in x, and its r derivative written out; times thermal_wind_amplitude. thermal_wind = 3: prescribed
    ∂S_tw/∂θ = dSdtheta sin 2θ, so σ₂ = -(2/3) dSdtheta. Only the profiles f_n come from diff_rot.
    '''
    degs = dr.tw_degrees()
    sig  = np.zeros((len(degs), len(rk)))
    sig1 = np.zeros_like(sig)
    if par.thermal_wind == 3:
        sig[degs.index(2)] = -(2/3)*par.dSdtheta
        return [ list(sig), list(sig1) ]
    Lg = np.polynomial.legendre.Legendre
    nd = dr.degrees()
    f = { n: [ dr.fX(n, rk, d) for d in (0,1,2) ] for n in nd }
    g0, g1 = rap.graviX(rk, 0), rap.graviX(rk, 1)
    c = par.Gaspard**2/par.Beyonce
    e = int(par.thermal_wind == 2)
    X, W = Lg([0, 1]), Lg([2/3, 0, -2/3])   # x, 1 - x²
    for k, r in enumerate(rk):
        O   = sum( Lg.basis(n)*f[n][0][k] for n in nd )   # ΔΩ, ∂ᵣΔΩ, ∂ᵣ²ΔΩ as series in x
        Or  = sum( Lg.basis(n)*f[n][1][k] for n in nd )
        Orr = sum( Lg.basis(n)*f[n][2][k] for n in nd )
        Ox, Oxr = O.deriv(), Or.deriv()
        a = 1 + e*O
        Fr, Fx = 2*a*Or, 2*a*Ox
        Frr = 2*a*Orr + 2*e*Or*Or
        Fxr = 2*a*Oxr + 2*e*Or*Ox
        q, q1 = r**2/g0[k], 2*r/g0[k] - r**2*g1[k]/g0[k]**2
        B  = X*Fr + W*Fx/r
        B1 = X*Frr + W*(Fxr/r - Fx/r**2)
        S  = (-c*q*B).integ(lbnd=-1)
        S1 = (-c*(q1*B + q*B1)).integ(lbnd=-1)
        for j, n in enumerate(degs):
            sig[j, k]  = S.coef[n] if n < len(S.coef) else 0
            sig1[j, k] = S1.coef[n] if n < len(S1.coef) else 0
    return [ list(par.thermal_wind_amplitude*sig), list(par.thermal_wind_amplitude*sig1) ]



_dr_bg = None  # NEW (2026-10-20): (key, diff_rot_background()) for the current radial grid, see diff_rot_setup


def diff_rot_background(ll):
    '''
    NEW (2026-10-20): the parts of diff_rot_setup that do not depend on the solution, at the current radii rk:
    the Gauss-Legendre grid x = cosθ and weights wx, sinθ, Y, ∂_θY and i m Y/sinθ for the l's ll (angular.ylm_grid),
    ΔΩ = Σₙ fₙ Pₙ and its derivatives on the (rk, cosθ) grid (diff_rot.fX), and, with thermal = 1 and thermal_wind,
    the (n, σₙ, σₙ′) of S_tw (thermal_wind_entropy). diagnose computes them once per grid, not per solution.
    '''
    m = par.m
    x, wx = np.polynomial.legendre.leggauss( par.lmax + 10 + max(dr.tw_nmax(), dr.n_max()) )
    sth = np.sqrt(1 - x**2)  # sinθ
    # as ut.Ylm, its θ derivative and i m Y/sinθ, by the recurrence of angular.ylm_grid (any l; scipy is NaN from l = 646)
    [ Yg, dYg, Zg ] = angular.ylm_grid(m, int(ll.max()), x)
    Y   = Yg[ll].astype(complex)
    dY  = dYg[ll].astype(complex)
    imY = Zg[ll]

    r   = rk[:,None]
    dO  = [ 0*r*x for d in (0,1,2) ]                  # ΔΩ, ∂ᵣΔΩ, ∂ᵣ²ΔΩ
    dOt = [ 0*r*x for d in (0,1) ]                    # ∂_θΔΩ, ∂ᵣ∂_θΔΩ
    for n in dr.degrees():
        Pn = np.polynomial.legendre.Legendre.basis(n)
        fn = [ dr.fX(n, rk, d)[:,None] for d in (0,1,2) ]
        dO  = [ dO[d] + fn[d]*Pn(x) for d in (0,1,2) ]
        dOt = [ dOt[d] - sth*Pn.deriv()(x)*fn[d] for d in (0,1) ]

    tw = []
    if par.thermal and par.thermal_wind:
        [ sig, sig1 ] = thermal_wind_entropy()
        tw = list( zip( dr.tw_degrees(), sig, sig1 ) )

    return [ x, wx, sth, Y, dY, imY, dO, dOt, tw ]



def diff_rot_setup():
    '''
    Differential rotation (diff_rot = 1). Sets, as module globals for flow_worker and thermal_worker:
    drforce[l] = [ [q, ∂ᵣq], [s, ∂ᵣs], [t, ∂ᵣt] ], the l-components of the force of the background flow on the perturbation
        𝐅 = -G [ (𝐔₀′⋅∇)𝐮 + (𝐮⋅∇)𝐔₀′ ] = -G [ ΔΩ (i m 𝐮 + 2𝐳×𝐮) + (𝐮⋅∇ΔΩ) r sinθ 𝐞ᵩ ],   𝐔₀′ = ΔΩ r sinθ 𝐞ᵩ,
    (right-hand side of the momentum equation per unit mass, G = Gaspard, ΔΩ = Σₙ fₙ(r) Pₙ(cosθ), diff_rot.fX) and of ∂ᵣ𝐅;
    with thermal = 1, drheat[l], the l-components of the left-hand-side terms p₀ [ i m G ΔΩ s + 𝐮⋅∇S_tw ] of the heat
    equation (S_tw = 0 if thermal_wind = 0).
    𝐮 is evaluated on the (rk, cosθ) grid from the l-components of velocity(),
        uᵣ = Σ q Y,   u_θ = Σ s ∂_θY + t i m Y/sinθ,   uᵩ = Σ s i m Y/sinθ - t ∂_θY,     Y = ut.Ylm,
    and the products are projected back on Y, (∂_θY, i m Y/sinθ), (i m Y/sinθ, -∂_θY) by Gauss-Legendre quadrature in
    cosθ, exact here (the integrands are polynomials in cosθ of degree < 2 lmax + 10).
    '''
    global drforce
    global drheat

    m = par.m
    G = par.Gaspard
    [ lp, lt, ll ] = ut.ell( par.m, par.lmax, par.symm)
    ll = np.asarray(ll)

    # NEW (2026-10-20): what does not depend on the solution (angular grid, Y, ΔΩ and S_tw on the grid) is computed
    # once per radial grid and kept in _dr_bg (2.5 s per solution before, at N = 192, lmax = 64, thermal_wind = 2)
    global _dr_bg
    key = ( rk.tobytes(), repr(sorted(vars(par).items())) )  # the grid and every parameter
    if _dr_bg is None or _dr_bg[0] != key:
        _dr_bg = ( key, diff_rot_background(ll) )
    [ x, wx, sth, Y, dY, imY, dO, dOt, tw ] = _dr_bg[1]

    # 𝐮 and ∂ᵣ𝐮 on the grid: u[d] = [uᵣ, u_θ, uᵩ] (d = 0) or their radial derivatives (d = 1), rows rk, columns cosθ
    # NEW (2026-10-20): sums over l as matrix products (rows rk, columns l) @ (rows l, columns cosθ), not np.outer per l
    u   = np.zeros( (2, 3, len(rk), len(x)), dtype=complex )
    vel = [ velocity(l) for l in ll ]
    for d in (0,1):
        [ q, s, t ] = [ np.array([ v[c][d] for v in vel ]).T for c in (0,1,2) ]
        u[d,0] = q @ Y
        u[d,1] = s @ dY + t @ imY
        u[d,2] = s @ imY - t @ dY

    r   = rk[:,None]

    def force(v, dO0, dOr, dOth):
        # -G [ ΔΩ (i m 𝐯 + 2𝐳×𝐯) + (r vᵣ ∂ᵣΔΩ + v_θ ∂_θΔΩ) sinθ 𝐞ᵩ ],  𝐳×𝐯 = ( -sinθ vᵩ, -cosθ vᵩ, sinθ vᵣ + cosθ v_θ )
        [ vr, vt, vp ] = v
        return -G*np.array([ dO0*(1j*m*vr - 2*sth*vp),
                             dO0*(1j*m*vt - 2*x*vp),
                             dO0*(1j*m*vp + 2*(sth*vr + x*vt)) + (r*vr*dOr + vt*dOth)*sth ])

    F0 = force(u[0], dO[0], dO[1], dOt[0])
    F1 = force(u[1], dO[0], dO[1], dOt[0]) + force(u[0], dO[1], dO[2], dOt[1])  # ∂ᵣ𝐅, but for
    F1[2] += -G*u[0,0]*dO[1]*sth                                                 # ∂ᵣ of the factor r in r vᵣ ∂ᵣΔΩ

    nl = 4*np.pi/(2*ll+1)  # ∫ |Y|² dΩ
    L  = ll*(ll+1)
    proj = lambda f, Z: 2*np.pi*(f*wx) @ np.conj(Z).T  # ∫ f Z* dΩ, rows rk, columns l
    q = [ proj(F[0], Y)/nl for F in (F0, F1) ]
    s = [ ( proj(F[1], dY) + proj(F[2], imY) )/(L*nl) for F in (F0, F1) ]
    t = [ ( proj(F[1], imY) - proj(F[2], dY) )/(L*nl) for F in (F0, F1) ]
    drforce = { l: [ [q[0][:,k], q[1][:,k]], [s[0][:,k], s[1][:,k]], [t[0][:,k], t[1][:,k]] ] for k,l in enumerate(ll) }

    if par.thermal:
        # s on the grid, ttab rows are the l's of lp. NEW (2026-10-20): from the table, one matrix product
        sg = np.transpose(ttab[0]) @ Y[[ list(ll).index(l) for l in lp ]]
        h = 1j*m*G*dO[0]*sg
        if par.thermal_wind:
            # 𝐮⋅∇S_tw = uᵣ Σ σₙ′ Pₙ - (u_θ/r) Σ σₙ sinθ Pₙ′(cosθ), σₙ and σₙ′ from diff_rot_background
            for n, sn, sn1 in tw:
                Pn = np.polynomial.legendre.Legendre.basis(n)
                if par.thermal_wind_radial:
                    h += u[0,0]*sn1[:,None]*Pn(x) - (u[0,1]/r)*sn[:,None]*sth*Pn.deriv()(x)
                else:   # no uᵣ ∂S_tw/∂r
                    h += - (u[0,1]/r)*sn[:,None]*sth*Pn.deriv()(x)
        hl = proj( pss0[:,None]*h, Y )/nl
        drheat = { l: hl[:,k] for k,l in enumerate(ll) }



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


    # differential rotation: complex power 2 ∫ ρ 𝐮*⋅𝐅 dV of the force 𝐅 = -G[(𝐔₀′⋅∇)𝐮 + (𝐮⋅∇)𝐔₀′] (diff_rot_setup),
    # and its enstrophy term (∇×𝐮)⋅(∇×𝐅)
    [ Wdr_l, Wdr_im_l, Enstro_dr_l ] = [0, 0, 0]
    if par.diff_rot:
        [   drq,   drs,   drt ] = drforce[l]
        [ cudrq, cudrs, cudrt ] = curl(l, drq, drs, drt)
        pdr = 2*( cdot_pol(l, velq[0], vels[0], drq[0], drs[0]) + cdot_tor(l, velt[0], drt[0]) )*rho0
        Wdr_l    = np.real( rad_quad( pdr, Ra, Rb, wk) )
        Wdr_im_l = np.imag( rad_quad( pdr, Ra, Rb, wk) )
        enstro_dr = ( dotprod_pol(l, cuvelq[0], cuvels[0], cudrq[0], cudrs[0]) + dotprod_tor(l, cuvelt[0], cudrt[0]) ) * const0**2
        Enstro_dr_l = rad_quad( enstro_dr, Ra, Rb, wk)

    return [ Kene_l, Dkin_l, Enstro_vel_l, Enstro_cor_l, Enstro_vif_l, Enstro_buo_l, Wthm_l, Wdr_l,
             Wcor_im_l, Dkin_im_l, Wthm_im_l, Wdr_im_l, Enstro_dr_l ]



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
    or fixed-flux walls and at r=0. With diff_rot = 1 the left-hand side gains p₀ [ i m G ΔΩ s + 𝐮⋅∇S_tw ] and the
    budget the term Wdr = -2 Re ∫ p₀ s* [ i m G ΔΩ s + 𝐮⋅∇S_tw ] dV (from diff_rot_setup).
    Uses the global solutions usol2, tsol2. Returns [ TE_l, Dthm_l, Wadv_l, Wdr_l ], Dthm_l without the ThermaD factor.
    The profiles are the physical ones (setup_grid), so the budget residual also flags a mismatch between
    the intended and the assembled heat equation (e.g. a diffusion operator with a wrong coefficient, or,
    when ricb=0, a profile without the parity that submatrices assumes).
    '''

    Ra = par.ricb
    Rb = ut.rcmb

    lp  = ut.ell( par.m, par.lmax, par.symm)[0]
    idx = list(lp).index(l)

    f_s  = radial( ttab, idx, 1, rk )  # NEW (2026-10-20): from the table
    hlm0 = f_s[0]  # s
    hlm1 = f_s[1]  # ∂s/∂r

    [ velq, _, _ ] = velocity(l)  # velq[0] is uᵣ

    Tene_l = rad_quad( entropy_energy(l, hlm0), Ra, Rb, wk )
    Wadv_l = rad_quad( entropy_advect(l, hlm0, velq[0]), Ra, Rb, wk )

    Dthm_l = 0
    if par.ThermaD > 0:
        Dthm_l = rad_quad( entropy_dissip(l, hlm0, hlm1), Ra, Rb, wk )
        # surface term [ r² κ p₀ s* ∂s/∂r ] from Ra to Rb
        [ rb, kpsb, tbnd ] = tsfc  # NEW (2026-10-20): s, ∂s/∂r and κ p₀ at Ra, Rb, from set_solution
        f_sb = radial( tbnd, idx, 1, rb )
        sfc  = (4*np.pi/(2*l+1)) * rb**2 * kpsb * 2*np.real( np.conj(f_sb[0]) * f_sb[1] )
        Dthm_l += sfc[1] - sfc[0]

    # differential rotation: -2 Re ∫ p₀ s* [ i m G ΔΩ s + 𝐮⋅∇S_tw ] dV (diff_rot_setup)
    Wdr_l = 0
    if par.diff_rot:
        Wdr_l = -rad_quad( dotprod_scal(l, drheat[l], hlm0), Ra, Rb, wk )

    return [ Tene_l, Dthm_l, Wadv_l, Wdr_l ]



def velocity( l ):
    '''
    Returns the rad, con, tor components of the l-component of the flow velocity,
    and their first radial derivatives. Sampled at given radii (global rk if radii is None).
    '''
        
    ll = ut.ell( par.m, par.lmax, par.symm)
    lp  = ll[0]  # l's for poloidals
    lt  = ll[1]  # l's for toroidals

    [ P, T ] = utab  # NEW (2026-10-20): tables of the radial functions (set_solution), not copies of usol2

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

    [ P, T ] = utab  # NEW (2026-10-20): tables of the radial functions (set_solution), not copies of usol2

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

    [ P, T ] = utab  # NEW (2026-10-20): tables of the radial functions (set_solution), not copies of usol2

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
        f_pol = radial( ttab, idx, 2, rk )  # NEW (2026-10-20): from the table, and g from set_solution
        [ grv0, grv1, grv2 ] = grv
        out_rad0 = f_pol[0] * grv0
        out_rad1 = f_pol[1] * grv0 + f_pol[0] * grv1
        out_rad2 = f_pol[2] * grv0 + f_pol[1] * grv1 + f_pol[1] * grv1 + f_pol[0] * grv2
        
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



def set_solution(usol, tsol, ndp=5, ndt=4):
    '''
    NEW (2026-10-20): sets the solution usol2, tsol2 (as given by expand_reshape_sol) as module globals for the
    workers, together with the tables of its radial functions at the current radii rk (cheb2space_table):
    utab = [ poloidal up to the ndp-th r-derivative, toroidal up to the ndt-th ] (visforce needs 5 and 4,
    velocity alone 3 and 2) and, with thermal = 1 and an entropy solution, ttab (up to the 2nd derivative),
    g and its derivatives at rk (grv, for buoyancy) and s, ∂s/∂r and κ p₀ at the walls (tsfc, for thermal_worker).
    Call it after rk is set (setup_grid, diagnose_4plot).
    '''
    global usol2, tsol2, utab, ttab, grv, tsfc
    usol2 = usol
    tsol2 = tsol
    utab  = [ cheb2space_table(usol[0], ndp, rk), cheb2space_table(usol[1], ndt, rk) ]
    if par.thermal and np.ndim(tsol) == 2:
        ttab = cheb2space_table(tsol, 2, rk)
        grv  = [ rap.graviX(rk, d) for d in (0,1,2) ]
        rb   = np.array([ par.ricb, ut.rcmb ])
        tsfc = [ rb, rap.prf.thermal_diffusivity(rb, 0) * rap.pressX(rb, 0), cheb2space_table(tsol, 1, rb) ]



def kinetic_energy(usol, Ra, Rb, ls=None):
    '''
    Returns the kinetic energy ∫ ½ ρ 𝐮⋅𝐮 dV of the flow solution usol (as given by expand_reshape_sol),
    integrated from Ra to Rb, with the same integrand and quadrature as flow_worker (i.e. spin_doctor's KE).
    ls restricts the sum to some l values (default: all), so that the work can be split among MPI ranks.
    '''
    setup_grid(Ra, Rb)
    set_solution(usol, 0, 3, 2)  # NEW (2026-10-20): velocity reads the radial functions from tables
    if ls is None:
        ls = ut.ell(par.m, par.lmax, par.symm)[2]
    out = 0.0
    for l in ls:
        [ velq, vels, velt ] = velocity(l)
        kinep = 0.5*dotprod_pol(l, velq[0], vels[0], velq[0], vels[0] )*rho0
        kinet = 0.5*dotprod_tor(l, velt[0], velt[0])*rho0
        out += rad_quad( kinep + kinet, Ra, Rb, wk)
    return out



# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------- NEW (2026-10-28): magnetic field (magnetic = 1)
# ----------------------------------------------------------------------------------------------------------------------
# Ported from conductive_IC (utils4pp.py at 39487bbf: energy_pol/tor, diffus_pol/tor, magnetic_worker, lorentz4pp_dipo,
# induction4pp_dipo), dipolar B0. b = ∇×∇×(f r) + ∇×(g r) carries no ρ: its components are q = L f/r, s = f' + f/r, t = g.
# The Lorentz force and the induction term are the Boussinesq ones (structure 'uniform', as background.check() requires):
# induction4pp takes the consoidal part of u from velocity tables that have the (ln ρ0)' term, but the formulas assume ρ0 = 1.



def set_bsolution(bsol, nd=3):
    '''
    Sets the magnetic solution bsol2 (expand_reshape_sol(..., ut.bsymm)) and the tables btab of f and g and their
    r-derivatives up to order nd at the current radii rk, as module globals. Call it after setup_grid.
    '''
    global bsol2, btab
    bsol2 = bsol
    btab  = [ cheb2space_table(bsol[0], nd, rk), cheb2space_table(bsol[1], nd, rk) ]



def bfield_pol(l, ns):
    '''
    [ [q, q', ...], [s, s', ...] ] up to the ns-th r-derivative (ns <= 2) of the poloidal l-component of b (from btab)
    '''
    lp = ut.ell(par.m, par.lmax, ut.bsymm)[0]
    f  = btab[0][:, list(lp).index(l), :]
    L  = l*(l+1)
    q  = [ L*f[0]/rk ]
    s  = [ f[1] + f[0]/rk ]
    if ns > 0:
        q.append( L*(f[1]/rk - f[0]/r2) )
        s.append( f[2] + f[1]/rk - f[0]/r2 )
    if ns > 1:
        q.append( L*(f[2]/rk - 2*f[1]/r2 + 2*f[0]/r3) )
        s.append( f[3] + f[2]/rk - 2*f[1]/r2 + 2*f[0]/r3 )
    return [ q, s ]



def bfield_tor(l, ns):
    '''
    [ t, t', ... ] up to the ns-th r-derivative of the toroidal l-component of b (from btab)
    '''
    lt = ut.ell(par.m, par.lmax, ut.bsymm)[1]
    return [ btab[1][j, list(lt).index(l), :] for j in range(ns+1) ]



def magnetic_energy(bsol, Ra, Rb, ls=None):
    '''
    Returns ∫ ½ 𝐛⋅𝐛 dV of the magnetic solution bsol (expand_reshape_sol with ut.bsymm), from Ra to Rb (no Hendrik
    factor), with the quadrature of kinetic_energy. ls restricts the sum to some l values (default: all).
    '''
    setup_grid(Ra, Rb)
    set_bsolution(bsol, 1)
    lp, lt, ll = ut.ell(par.m, par.lmax, ut.bsymm)
    out = 0.0
    for l in (ll if ls is None else ls):
        if l in lp:
            [ [q0], [s0] ] = bfield_pol(l, 0)
            out += rad_quad( 0.5*dotprod_pol(l, q0, s0, q0, s0), Ra, Rb, wk)
        elif l in lt:
            t0 = bfield_tor(l, 0)[0]
            out += rad_quad( 0.5*dotprod_tor(l, t0, t0), Ra, Rb, wk)
    return out



def diffus_pol(l, qlm0, qlm1, qlm2, slm0, slm1, slm2):
    '''
    Integrand of ∫ 𝐛⋅∇²𝐛 dV (2 Re of it), poloidal l-component, for a solenoidal field (conductive_IC)
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
    Integrand of ∫ 𝐛⋅∇²𝐛 dV (2 Re of it), toroidal l-component (conductive_IC)
    '''
    L = l*(l+1)
    f0 = 4*np.pi/(2*l+1)
    f1 = L * r2 * np.conj(tlm0) * tlm2
    f2 = 2 * rk * L * np.conj(tlm0) * tlm1
    f3 = -(L**2)*( np.conj(tlm0)*tlm0 )
    return 2*np.real( f0*(f1+f2+f3) )



def lorentz4pp( l ):
    '''
    The l-component [rad, con, tor] of the Lorentz force (∇×b)×B0 + (∇×B0)×b at rk (without Hendrik), dipolar B0
    (conductive_IC's lorentz4pp_dipo), from btab
    '''
    m   = par.m
    lp, lt, _ = ut.ell( m, par.lmax, ut.bsymm)
    h0, h1, h2 = [ bf.h(rk, k) for k in (0, 1, 2) ]

    out_rad = np.zeros_like(rk, dtype='complex128')
    out_con = np.zeros_like(rk, dtype='complex128')
    out_tor = np.zeros_like(rk, dtype='complex128')

    if l-1 in lp:
        [ [ qlm0, _ ], [ slm0, slm1 ] ] = bfield_pol(l-1, 1)
        C_rad    = (-1 + l)*np.sqrt(l**2 - m**2)/(-1 + 2*l)
        out_rad += C_rad*( -(h0*(qlm0/r2 + slm0/r2 - slm1/rk)) + (h2*slm0 + h1*(-qlm0/rk + 3*slm0/rk + slm1)) )
        C_con    = np.sqrt(l**2 - m**2)/(l*(-1 + 2*l))
        out_con += C_con*( (-2*h0*l*qlm0/r2 + qlm0*(2*h1/rk + h2) + 2*h0*(-1 + l)*(slm0/r2 + slm1/rk)) )
    elif l-1 in lt:
        [ tlm0, tlm1 ] = bfield_tor(l-1, 1)
        C_tor =  -(-1 + l)*np.sqrt(l**2 - m**2)/(l*(-1 + 2*l))
        out_tor += C_tor*( h0*(-2 + l)*tlm0/r2 + h1*l*tlm0/rk - 2*h0*tlm1/rk )

    if l in lt:
        [ tlm0, tlm1 ] = bfield_tor(l, 1)
        out_rad += 1j*m*( h0*(-tlm0/r2 + tlm1/rk) + 3*h1*tlm0/rk + h2*tlm0 + h1*tlm1 )
        out_con += 1j*m*( h0*(2 + l + l**2)*tlm0/r2 + h1*l*(1 + l)*tlm0/rk + 2*h0*tlm1/rk ) / (l*(1 + l))
    elif l in lp:
        [ [ qlm0, _ ], [ slm0, slm1 ] ] = bfield_pol(l, 1)
        out_tor += 1j*m*( qlm0*(2*h1/rk + h2) - 2*h0*(slm0/r2 + slm1/rk) ) / (l*(1 + l))

    if l+1 in lp:
        [ [ qlm0, _ ], [ slm0, slm1 ] ] = bfield_pol(l+1, 1)
        C_rad = (2 + l)*np.sqrt((1 + l - m)*(1 + l + m)) / (3 + 2*l)
        out_rad += C_rad*( h0*(qlm0/r2 + slm0/r2 - slm1/rk) - (h2*slm0 + h1*(-qlm0/rk + 3*slm0/rk + slm1)) )
        C_con = np.sqrt((1 + l - m)*(1 + l + m)) / ((1 + l)*(3 + 2*l))
        out_con += C_con*( -2*h0*(1 + l)*qlm0/r2 - qlm0*(2*h1/rk + h2) + 2*h0*(2 + l)*(slm0/r2 + slm1/rk) )
    elif l+1 in lt:
        [ tlm0, tlm1 ] = bfield_tor(l+1, 1)
        C_tor = (2 + l)*np.sqrt((1 + l - m)*(1 + l + m)) / ((1 + l)*(3 + 2*l))
        out_tor += C_tor*( (h0*(3 + l)*tlm0/r2 + h1*(1 + l)*tlm0/rk + 2*h0*tlm1/rk) )

    return [ out_rad*bf.cnorm, out_con*bf.cnorm, out_tor*bf.cnorm ]



def induction4pp( l ):
    '''
    The l-component [rad, con, tor] of the induction term ∇×(u×B0) at rk, dipolar B0 (conductive_IC's
    induction4pp_dipo), from the velocity tables utab (set_solution); ρ0 = 1 assumed
    '''
    m   = par.m
    lp, lt, _ = ut.ell( m, par.lmax, par.symm)
    [ P, T ] = utab
    h0, h1, h2 = [ bf.h(rk, k) for k in (0, 1, 2) ]

    out_rad = np.zeros_like(rk, dtype='complex128')
    out_con = np.zeros_like(rk, dtype='complex128')
    out_tor = np.zeros_like(rk, dtype='complex128')

    if l-1 in lp:
        [ [ qlm0, qlm1], [slm0, slm1] ] =  cheb2space_pol(l-1, lp, P, 1, rk)
        out_rad += -(((1 + l)*np.sqrt(l**2 - m**2)*(h1*qlm0*rk + h0*(qlm0 - 2*(-1 + l)*slm0)))/((-1 + 2*l)*r2))
        out_con += (np.sqrt(l**2 - m**2)*(-(qlm1*(h0 + h1*rk)) - qlm0*(2*h1 + h2*rk) + 2*h1*(-1 + l)*slm0 + 2*h0*(-1 + l)*slm1))/(l*(-1 + 2*l)*rk)
    elif l-1 in lt:
        [ tlm0, tlm1]  =  cheb2space_tor(l-1, lt, T, 1, rk)
        out_tor += -(((1 + 2*l)*(h1*(np.sqrt((l*(-1 + l**2))/(-1 + 4*l**2)) - 2*np.sqrt((1 - l**2)/(l - 4*l**3)))*rk*tlm0 + h0*(np.sqrt((l*(-1 + l**2))/(-1 + 4*l**2))*tlm0 - 2*np.sqrt((1 - l**2)/(l - 4*l**3))*rk*tlm1)))/(np.sqrt((l*(1 + l)*(-1 + 4*l**2))/((-1 + l)*(l**2 - m**2)))*r2))

    if l in lt:
        [ tlm0, tlm1]  =  cheb2space_tor(l, lt, T, 1, rk)
        out_rad += (2j*h0*m*tlm0)/r2
        out_con += ((2j)*m*(h1*tlm0 + h0*tlm1))/(l*(1 + l)*rk)
    elif l in lp:
        [ [ qlm0, qlm1], [slm0, slm1] ] =  cheb2space_pol(l, lp, P, 1, rk)
        out_tor += ((-1j)*m*(h2*qlm0*r2 - h0*l*(1 + l)*slm0 + h1*rk*(2*qlm0 + qlm1*rk - (-2 + l + l**2)*slm0) + h0*rk*(qlm1 + 2*slm1)))/(l*(1 + l)*r2)

    if l+1 in lp:
        [ [ qlm0, qlm1], [slm0, slm1] ] =  cheb2space_pol(l+1, lp, P, 1, rk)
        out_rad += (l*np.sqrt((1 + l - m)*(1 + l + m))*(h1*qlm0*rk + h0*(qlm0 + 2*(2 + l)*slm0)))/((3 + 2*l)*r2)
        out_con += (np.sqrt((1 + l - m)*(1 + l + m))*(h2*qlm0*rk + h1*(2*qlm0 + qlm1*rk + 2*(2 + l)*slm0) + h0*(qlm1 + 2*(2 + l)*slm1)))/((1 + l)*(3 + 2*l)*rk)
    elif l+1 in lt:
        [ tlm0, tlm1]  =  cheb2space_tor(l+1, lt, T, 1, rk)
        out_tor +=  ((1 + 2*l)*np.sqrt(((2 + l)*(1 + l - m)*(1 + l + m))/(3 + 4*l*(2 + l)))*(h1*(2*np.sqrt((l*(2 + l))/(3 + 11*l + 12*l**2 + 4*l**3)) + np.sqrt((l*(1 + l)*(2 + l))/(3 + 4*l*(2 + l))))*rk*tlm0 + h0*(np.sqrt((l*(1 + l)*(2 + l))/(3 + 4*l*(2 + l)))*tlm0 + 2*np.sqrt((l*(2 + l))/(3 + 11*l + 12*l**2 + 4*l**3))*rk*tlm1)))/(np.sqrt(l*(1 + l))*r2)

    return [ out_rad*bf.cnorm, out_con*bf.cnorm, out_tor*bf.cnorm ]



def magnetic_worker( l ):
    '''
    l-component of the magnetic budget terms: [ ME = ∫ ½ 𝐛⋅𝐛 dV, ∫ 𝐛⋅∇²𝐛 dV, ∫ 𝐛⋅∇×(𝐮×𝐁₀) dV (both 2 Re), and the complex
    power of the Lorentz force 2 ∫ 𝐮*⋅𝐅 dV as its real and imaginary parts ] (no Hendrik or MagnetD factors; the Lorentz
    force per unit mass is ρ0⁻¹ Hendrik 𝐅, so its power has no ρ0 weight). Needs set_solution and set_bsolution.
    '''
    Ra = par.ricb
    Rb = ut.rcmb
    lp, lt, _ = ut.ell(par.m, par.lmax, ut.bsymm)

    [ qi, si, ti ] = induction4pp(l)
    if l in lp:
        [ [q0, q1, q2], [s0, s1, s2] ] = bfield_pol(l, 2)
        mene = 0.5*dotprod_pol(l, q0, s0, q0, s0)
        mdfs = diffus_pol(l, q0, q1, q2, s0, s1, s2)
        indu = dotprod_pol(l, q0, s0, qi, si)
    else:
        [ t0, t1, t2 ] = bfield_tor(l, 2)
        mene = 0.5*dotprod_tor(l, t0, t0)
        mdfs = diffus_tor(l, t0, t1, t2)
        indu = dotprod_tor(l, t0, ti)

    [ fq, fs, ft ] = lorentz4pp(l)
    [ [vq, _, _], [vs, _, _], [vt, _, _] ] = velocity(l)
    plor = 2*( cdot_pol(l, vq, vs, fq, fs) + cdot_tor(l, vt, ft) )

    return [ rad_quad(mene, Ra, Rb, wk), rad_quad(mdfs, Ra, Rb, wk), rad_quad(indu, Ra, Rb, wk),
             np.real(rad_quad(plor, Ra, Rb, wk)), np.imag(rad_quad(plor, Ra, Rb, wk)) ]



def ohmic_dissip_pol(l, qlm0, slm0, slm1):
    '''
    Integrand of ∫ |∇×𝐛|² dV, poloidal l-component (conductive_IC)
    '''
    f0 = 8*np.pi* l*(l+1)/(2*l+1)
    f1 = np.absolute( qlm0 - slm0 - rk*slm1 )**2
    return f0*f1



def ohmic_dissip_tor(l, tlm0, tlm1):
    '''
    Integrand of ∫ |∇×𝐛|² dV, toroidal l-component (conductive_IC)
    '''
    L = l*(l+1)
    f0 = 8*np.pi*L/(2*l+1)
    f1 = np.absolute( rk*tlm1 + tlm0 )**2
    f2 = L*np.absolute( tlm0 )**2
    return f0*(f1+f2)



def em_budget(bsol, bic=None):
    '''
    NEW (2026-10-28): the terms of the magnetic energy budget that the fluid-only resid𝐛 leaves out, ported from conductive_IC's
    em_budget for an insulating mantle and an insulating or conducting ('conducting, Chebys') inner core (no thin walls):
    [ Ohm_f, ME_ic, Ohm_ic, ME_ext ], without Hendrik or MagnetD:
      Ohm_f  : ∫ |∇×𝐛|² over the fluid
      ME_ic  : ½ ∫ |𝐛|²/mu_i2o inside the inner core: the field of a conducting inner core (bic: its part of the solution
               vector), or the potential field below an insulating one
      Ohm_ic : η_i ∫ |∇×𝐛|²/mu_i2o over a conducting inner core, η_i = 1/(mu_i2o sigma_i2o)
      ME_ext : ½ ∫ |𝐛|² of the potential field outside the outer wall
    Needs setup_grid (the fluid's quadrature grid).
    '''
    r_i, r_o = par.ricb, ut.rcmb
    [ lp, lt, _ ] = ut.ell(par.m, par.lmax, ut.bsymm)
    f0p = 4*np.pi/(2*lp+1)
    Lp = lp*(lp+1.)

    Fp = cheb2space_table(bsol[0], 2, rk); Gt = cheb2space_table(bsol[1], 1, rk)
    Ohm_f = 0.0
    for k, l in enumerate(lp):
        f, f1, f2 = Fp[0, k], Fp[1, k], Fp[2, k]
        Ohm_f += rad_quad( ohmic_dissip_pol(l, Lp[k]*f/rk, f1 + f/rk, f2 + f1/rk - f/r2), r_i, r_o, wk)
    for k, l in enumerate(lt):
        Ohm_f += rad_quad( ohmic_dissip_tor(l, Gt[0, k], Gt[1, k]), r_i, r_o, wk)

    def bvals(C, nd, x):  # values and r-derivatives of the rows of C at x = -1 (inner wall) or 1 (outer wall)
        dk = np.ascontiguousarray(np.transpose(C)); out = []
        for j in range(nd+1):
            if j > 0: dk = ut.Dcheb(dk, r_i, r_o)
            out.append(ch.chebval(x, dk))
        return out

    ME_ic = Ohm_ic = 0.0
    f = bvals(bsol[0], 0, 1.0)[0]
    ME_ext = np.sum( f0p*Lp*lp*np.abs(f)**2 ) * r_o   # potential field r^-(l+1) beyond r_o (F = r f there, r_o = 1)

    if par.innercore == 'insulator':
        f = bvals(bsol[0], 0, -1.0)[0]
        ME_ic = np.sum( f0p*Lp*(lp+1)*r_i*np.abs(f)**2 )   # potential field r^l below r_i
    elif bic is not None:
        # the inner-core field: Chebyshev series in x = r/ricb on [-1, 1], one parity per l
        Nic = ut.Nic; nb = int((par.lmax - par.m + 1)/2)
        if ut.symmB0 == -1: ixf, ixg = (par.m + ut.s)%2, (par.m + 1 - ut.s)%2
        else:               ixf, ixg = (par.m + 1 - ut.s)%2, (par.m + ut.s)%2
        xg, wg = np.polynomial.legendre.leggauss(max(2*par.N_cic, 64))
        ri = 0.5*r_i*(xg + 1); wi = 0.5*r_i*wg
        def icfun(a, ix, nd):
            cc = np.zeros(par.N_cic, complex); cc[ix::2] = a; out = []
            for j in range(nd+1):
                out.append(ch.chebval(ri/r_i, cc)); cc = ch.chebder(cc)/r_i
            return out
        eta_i = 1/(par.mu_i2o*par.sigma_i2o)
        for k, l in enumerate(lp):
            L = l*(l+1.); a0 = 4*np.pi/(2*l+1)
            f, f1, f2 = icfun(bic[k*Nic:(k+1)*Nic], ixf, 2)
            q0, s0 = L*f/ri, f1 + f/ri
            ME_ic  += np.sum( wi*a0*ri**2*( np.abs(q0)**2 + L*np.abs(s0)**2 ) )
            Ohm_ic += eta_i*np.sum( wi*2*a0*L*ri**2*np.abs(f2 + 2*f1/ri - L*f/ri**2)**2 )
        for k, l in enumerate(lt):
            L = l*(l+1.); a0 = 4*np.pi/(2*l+1)
            g, g1 = icfun(bic[nb*Nic + k*Nic: nb*Nic + (k+1)*Nic], ixg, 1)
            ME_ic  += np.sum( wi*a0*ri**2*L*np.abs(g)**2 )
            Ohm_ic += eta_i*np.sum( wi*2*a0*L*( L*np.abs(g)**2 + np.abs(ri*g1 + g)**2 ) )
        # FIX (2026-10-28): with mu_i2o != 1 the inner core stores ½ ∫ 𝐛⋅𝐇 = ½ ∫ |𝐛|²/mu_i2o and loses ∫ 𝐄⋅𝐉 = η_i ∫ |∇×𝐛|²/mu_i2o
        # (𝐇 = 𝐛/mu_i2o, 𝐉 = ∇×𝐇): conductive_IC's em_budget omits the 1/mu_i2o, and its resid𝐄 is then 1e-2 at mu_i2o = 2
        ME_ic, Ohm_ic = ME_ic/par.mu_i2o, Ohm_ic/par.mu_i2o
    else:
        ME_ic = Ohm_ic = np.nan

    return [ Ohm_f, ME_ic, Ohm_ic, ME_ext ]



def diagnose_magnetic( bsol ):
    '''
    NEW (2026-10-28): magnetic_worker for every l, as an array (one row per l). Call it right after diagnose (same grid
    and flow solution).
    '''
    set_bsolution(bsol)
    ll = ut.ell(par.m, par.lmax, ut.bsymm)[2]
    return np.array([ magnetic_worker(l) for l in ll ])



def diagnose( usol, tsol, Ra, Rb, ncpus):
    '''
    Computes kinetic energy, internal and kinetic energy dissipation,
    and input power from body forces. Integrated From r=Ra to r=Rb, and
    angularly over the whole sphere.
    NEW (2026-10-20): serial, as in the conductive_IC branch (2026-10-16). The radial functions of every l and
    their derivatives are evaluated once per solution (set_solution, cheb2space_table) and the workers, run
    one l after another, read them from these tables. Before, each l was a task for a pool of ncpus processes,
    and each worker evaluated its radial functions with chebval (a Python loop over N) several times over
    (velocity, coriolis with the neighbouring l's, visforce, buoyancy). ncpus is no longer used.
    '''

    out_t = 0

    setup_grid(Ra, Rb)  # quadrature grid and background profiles, as module globals
    set_solution(usol, tsol)  # the solution and the tables of its radial functions at rk, as module globals
    if par.diff_rot:
        diff_rot_setup()  # differential-rotation force and heat-equation terms, as module globals

    [ lp_u, lt_u, ll ] = ut.ell(par.m, par.lmax, par.symm)  # the l-indices of the flow field

    out_u = np.array([ flow_worker(l) for l in ll ])

    if par.thermal:
        out_t = np.array([ thermal_worker(l) for l in lp_u ])

    return [ out_u, out_t ]
    


def diagnose_4plot( ncpus, usol, tsol, radii, field0):
    '''
    Returns the [rad,con,tor] components of the field field0 (see worker_4plot), one row per l, at the radii.
    NEW (2026-10-20): serial, from tables of the radial functions at the radii (set_solution); ncpus is no longer used.
    '''

    global field
    field =  field0

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

    set_solution(usol, tsol)  # NEW (2026-10-20): the solution and the tables of its radial functions at the radii

    [ lp_u, lt_u, ll ] = ut.ell(par.m, par.lmax, par.symm)  # the l-indices of the flow field

    out = np.array([ worker_4plot(l) for l in ll ])

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
