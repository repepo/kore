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
import background as bg   # NEW (2026-11-06): the conductivity of the resolved mantle (mantle_budget)
import centrifugal as cf   # NEW (2026-11-05)
import bgflow as bgf   # NEW (2026-11-08): a background flow (bgflow = 1)


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
    # NEW (2026-11-05): the structure's g, as diff_rot.sigma_coefficients (not the effective g of centrifugal = 1)
    g0, g1 = rap.graviX(rk, 0, False), rap.graviX(rk, 1, False)
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



_dr_bg = {}  # NEW (2026-10-20): {key: diff_rot_background()} for the radial grids in use, see diff_rot_setup


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
    equation (S_tw = 0 if thermal_wind = 0); NEW (2026-11-08) with compositional = 1 and a solution of the composition,
    drcomp[l], those of ρ₀ i m G ΔΩ c in its equation.
    𝐮 is evaluated on the (rk, cosθ) grid from the l-components of velocity(),
        uᵣ = Σ q Y,   u_θ = Σ s ∂_θY + t i m Y/sinθ,   uᵩ = Σ s i m Y/sinθ - t ∂_θY,     Y = ut.Ylm,
    and the products are projected back on Y, (∂_θY, i m Y/sinθ), (i m Y/sinθ, -∂_θY) by Gauss-Legendre quadrature in
    cosθ, exact here (the integrands are polynomials in cosθ of degree < 2 lmax + 10).
    '''
    global drforce
    global drheat
    global drcomp

    m = par.m
    G = par.Gaspard
    [ lp, lt, ll ] = ut.ell( par.m, par.lmax, par.symm)
    ll = np.asarray(ll)

    # NEW (2026-10-20): what does not depend on the solution (angular grid, Y, ΔΩ and S_tw on the grid) is computed
    # once per radial grid and kept in _dr_bg (2.5 s per solution before, at N = 192, lmax = 64, thermal_wind = 2)
    # (NEW 2026-11-04: two grids are kept, the quadrature nodes of diagnose and the walls of wall_powers)
    key = ( rk.tobytes(), repr(sorted(vars(par).items())) )  # the grid and every parameter
    if key not in _dr_bg:
        if len(_dr_bg) >= 2:
            _dr_bg.clear()
        _dr_bg[key] = diff_rot_background(ll)
    [ x, wx, sth, Y, dY, imY, dO, dOt, tw ] = _dr_bg[key]

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

    if comp_on():   # NEW (2026-11-08): ρ₀ i m G ΔΩ c, the composition's (no part balances ΔΩ: C is spherical)
        cg = np.transpose(ctab[0]) @ Y[[ list(ll).index(l) for l in lp ]]
        cl = proj( rho0[:,None]*(1j*m*G*dO[0]*cg), Y )/nl
        drcomp = { l: cl[:,k] for k,l in enumerate(ll) }



_bg_bg = {}  # NEW (2026-11-08): {key: bgflow_background()} for the radial grids in use, see bgflow_setup


def bgflow_background(ll):
    '''
    NEW (2026-11-08): the parts of bgflow_setup that do not depend on the solution, at the current radii rk: the
    Gauss-Legendre grid x = cosθ and weights wx, Y, ∂_θY and i m Y/sinθ for the l's ll (angular.ylm_grid), the background
    flow 𝐔, ∂ᵣ𝐔, ∂ᵣ²𝐔 and its vorticity 𝛀_U, ∂ᵣ𝛀_U on the (rk, cosθ) grid (bgflow.flow_grid), and the gradients ∇S, ∇C of
    the background's entropy and composition (bgflow.scalar_grid; None without thermal, compositional).
    '''
    x, wx = np.polynomial.legendre.leggauss( par.lmax + 12 + bgf.nmax() )   # exact for the products of 𝐔 (degree <= nmax + 1)
    [ Yg, dYg, Zg ] = angular.ylm_grid(par.m, int(ll.max()), x)             # with 𝐮 and Y* (degree <= lmax + 1 each)
    Ug = bgf.flow_grid(rk, x, nd=2)
    gS = bgf.scalar_grid('S', rk, x) if par.thermal else None
    gC = bgf.scalar_grid('C', rk, x) if par.compositional else None
    return [ x, wx, Yg[ll].astype(complex), dYg[ll].astype(complex), Zg[ll], Ug, gS, gC ]



def bgflow_setup():
    '''
    NEW (2026-11-08): the background flow (bgflow = 1, bgflow.py). Sets, as module globals for the workers, as diff_rot_setup:
    bgforce[l] = [ [q, ∂ᵣq], [s, ∂ᵣs], [t, ∂ᵣt] ], the l-components of the force of the background on the perturbation per
    unit mass and of its r-derivative,
        𝐅 = -[ (𝐔⋅∇)𝐮 + (𝐮⋅∇)𝐔 ] = -∇(𝐔⋅𝐮) + 𝐔×𝛚 + 𝐮×𝛀_U,   𝛚 = ∇×𝐮,  𝛀_U = ∇×𝐔
    (the right-hand side of the momentum equation; the gradient, which the rows of A leave out with their curls, is included
    so that the pressure of pressure_force is the physical one), from 𝐮, 𝛚, the background (bgflow_background) and their
    r-derivatives on the (rk, cosθ) grid, projected as in diff_rot_setup (the gradient of φ = 𝐔⋅𝐮: q = φ′, s = φ/r);
    with thermal = 1, bgheat[l], the l-components of the left-hand-side terms p₀ (𝐔⋅∇s + 𝐮⋅∇S) of the heat equation, and
    with a solution of the composition, bgcomp[l], those of ρ₀ (𝐔⋅∇c + 𝐮⋅∇C) in its equation.
    '''
    global bgforce
    global bgheat
    global bgcomp

    [ lp, lt, ll ] = ut.ell( par.m, par.lmax, par.symm)
    ll = np.asarray(ll)
    key = ( rk.tobytes(), repr(sorted(vars(par).items())) )
    if key not in _bg_bg:
        if len(_bg_bg) >= 2:
            _bg_bg.clear()
        _bg_bg[key] = bgflow_background(ll)
    [ x, wx, Y, dY, imY, Ug, gS, gC ] = _bg_bg[key]

    def grid(q, s, t):   # (rows l, columns rk) -> the (r, θ, φ) components on (rk, cosθ)
        return [ q.T @ Y, s.T @ dY + t.T @ imY, s.T @ imY - t.T @ dY ]
    vel = [ velocity(l) for l in ll ]
    u = [ grid(*[ np.array([ v[c][d] for v in vel ]) for c in (0,1,2) ]) for d in (0,1,2) ]   # 𝐮, ∂ᵣ𝐮, ∂ᵣ²𝐮
    cu = [ curl(l, *v) for l, v in zip(ll, vel) ]
    w = [ grid(*[ np.array([ c[k][d] for c in cu ]) for k in (0,1,2) ]) for d in (0,1) ]      # 𝛚, ∂ᵣ𝛚

    cross = lambda a, b: [ a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0] ]
    add = lambda *vs: [ sum(c) for c in zip(*vs) ]
    dot = lambda a, b: a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    U, W = Ug['U'], Ug['W']
    F0 = add( cross(U[0], w[0]), cross(u[0], W[0]) )
    F1 = add( cross(U[1], w[0]), cross(U[0], w[1]), cross(u[1], W[0]), cross(u[0], W[1]) )
    phi = [ dot(U[0], u[0]), dot(U[1], u[0]) + dot(U[0], u[1]), dot(U[2], u[0]) + 2*dot(U[1], u[1]) + dot(U[0], u[2]) ]

    nl = 4*np.pi/(2*ll+1)  # ∫ |Y|² dΩ
    L  = ll*(ll+1)
    r  = rk[:,None]
    proj = lambda f, Z: 2*np.pi*(f*wx) @ np.conj(Z).T  # ∫ f Z* dΩ, rows rk, columns l
    ph = [ proj(p, Y)/nl for p in phi ]               # φ, ∂ᵣφ, ∂ᵣ²φ
    q = [ proj(F[0], Y)/nl for F in (F0, F1) ]
    s = [ ( proj(F[1], dY) + proj(F[2], imY) )/(L*nl) for F in (F0, F1) ]
    t = [ ( proj(F[1], imY) - proj(F[2], dY) )/(L*nl) for F in (F0, F1) ]
    q = [ q[0] - ph[1], q[1] - ph[2] ]
    s = [ s[0] - ph[0]/r, s[1] - ph[1]/r + ph[0]/r**2 ]
    bgforce = { l: [ [q[0][:,k], q[1][:,k]], [s[0][:,k], s[1][:,k]], [t[0][:,k], t[1][:,k]] ] for k,l in enumerate(ll) }

    ip = [ list(ll).index(l) for l in lp ]
    def advection(tab, grad):   # 𝐔⋅∇X + 𝐮⋅∇X0 on the grid, X from its table (rows the l's of lp), ∇X0 the background's
        X0 = np.transpose(tab[0]) @ Y[ip]
        X1 = np.transpose(tab[1]) @ Y[ip]
        gX = [ X1, (np.transpose(tab[0]) @ dY[ip])/r, (np.transpose(tab[0]) @ imY[ip])/r ]
        return dot(U[0], gX) + dot(u[0], grad)
    if par.thermal and np.ndim(tsol2) == 2:
        hl = proj( pss0[:,None]*advection(ttab, gS), Y )/nl
        bgheat = { l: hl[:,k] for k,l in enumerate(ll) }
    if comp_on():
        cl = proj( rho0[:,None]*advection(ctab, gC), Y )/nl
        bgcomp = { l: cl[:,k] for k,l in enumerate(ll) }



def flow_worker( l ):
    '''
    Computes the power balance from the momentum (the Navier-Stokes) equation, l-component.
    Includes the kinetic energy, the kinetic energy dissipation, the rate of working (power) of the
    buoyancy force, the enstrophy budget terms, the differential rotation power, and the imaginary parts
    of the Coriolis, viscous and buoyancy powers for the frequency balance.
    NEW (2026-11-08): with compositional = 1 the buoyancy force is that of the entropy and of the composition (the enstrophy
    term Enstro_buo of both), the power of the entropy's is Wthm and the composition's Wcmp (and Wcmp_im), the last two values.
    NEW (2026-11-08): with a background flow (bgflow = 1) the power of its force, Wbg and Wbg_im, and its enstrophy term
    Enstro_bg, the last three values (0 without).
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


    # buoyancy force: that of the entropy (bth), NEW (2026-11-08) of the composition (bcp) and their sum
    bth = buoyancy(l, 'thermal') if par.thermal else None
    bcp = buoyancy(l, 'comp') if comp_on() else None
    if bth is not None and bcp is not None:
        buo = [ [ bth[k][d] + bcp[k][d] for d in (0,1,2) ] for k in (0,1,2) ]
    else:
        buo = bth if bth is not None else bcp
    if buo is not None:
        [   buoq,   buos,   buot ] = buo
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

    if buo is not None:
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
        wbuop = dotprod_pol(l, velq[0], vels[0], bth[0][0], bth[1][0])*rho0
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
        pbuo = 2*cdot_pol(l, velq[0], vels[0], bth[0][0], bth[1][0])*rho0
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
    if buo is not None:
        Enstro_buo_l = rad_quad( enstro_buo_p + enstro_buo_t, Ra, Rb, wk)
    else:
        Enstro_buo_l = 0

    # NEW (2026-11-08): the power of the buoyancy of the composition, 2 ∫ ρ 𝐮*⋅(BeyonceC g c 𝐫̂) dV: real part Wcmp, imaginary
    # part Wcmp_im
    [ Wcmp_l, Wcmp_im_l ] = [ 0, 0 ]
    if bcp is not None:
        pcmp = 2*cdot_pol(l, velq[0], vels[0], bcp[0][0], bcp[1][0])*rho0
        Wcmp_l    = np.real( rad_quad( pcmp, Ra, Rb, wk ) )   # (as Wthm: 2 Re cdot_pol is dotprod_pol)
        Wcmp_im_l = np.imag( rad_quad( pcmp, Ra, Rb, wk ) )


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

    # NEW (2026-11-08): a background flow, the complex power 2 ∫ ρ 𝐮*⋅𝐅 dV of its force 𝐅 = -[(𝐔⋅∇)𝐮 + (𝐮⋅∇)𝐔] (bgflow_setup)
    # and its enstrophy term, as those of the differential rotation; the last three values
    [ Wbg_l, Wbg_im_l, Enstro_bg_l ] = [0, 0, 0]
    if par.bgflow:
        [   bgq,   bgs,   bgt ] = bgforce[l]
        [ cubgq, cubgs, cubgt ] = curl(l, bgq, bgs, bgt)
        pbg = 2*( cdot_pol(l, velq[0], vels[0], bgq[0], bgs[0]) + cdot_tor(l, velt[0], bgt[0]) )*rho0
        Wbg_l    = np.real( rad_quad( pbg, Ra, Rb, wk) )
        Wbg_im_l = np.imag( rad_quad( pbg, Ra, Rb, wk) )
        enstro_bg = ( dotprod_pol(l, cuvelq[0], cuvels[0], cubgq[0], cubgs[0]) + dotprod_tor(l, cuvelt[0], cubgt[0]) ) * const0**2
        Enstro_bg_l = rad_quad( enstro_bg, Ra, Rb, wk)

    return [ Kene_l, Dkin_l, Enstro_vel_l, Enstro_cor_l, Enstro_vif_l, Enstro_buo_l, Wthm_l, Wdr_l,
             Wcor_im_l, Dkin_im_l, Wthm_im_l, Wdr_im_l, Enstro_dr_l, Wcmp_l, Wcmp_im_l, Wbg_l, Wbg_im_l, Enstro_bg_l ]



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

    elif field == 'b':   # NEW (2026-11-08): the magnetic field, from the tables of set_bsolution (diagnose_4plot's bsol)
        lp_b, lt_b, _ = ut.ell(par.m, par.lmax, ut.bsymm)
        rad0 = con0 = tor0 = np.zeros_like(rk, dtype='complex128')
        if l in lp_b:
            [ [rad0], [con0] ] = bfield_pol(l, 0)
        elif l in lt_b:
            tor0 = bfield_tor(l, 0)[0]
 
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
    NEW (2026-11-08): with a background flow (bgflow = 1) the term Wbg = -2 Re ∫ p₀ s* (𝐔⋅∇s + 𝐮⋅∇S) dV (bgflow_setup).
    Uses the global solutions usol2, tsol2. Returns [ TE_l, Dthm_l, Wadv_l, Wdr_l, Wbg_l ], Dthm_l without the ThermaD factor.
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

    # NEW (2026-11-08): a background flow, -2 Re ∫ p₀ s* (𝐔⋅∇s + 𝐮⋅∇S) dV (bgflow_setup)
    Wbg_l = 0
    if par.bgflow:
        Wbg_l = -rad_quad( dotprod_scal(l, bgheat[l], hlm0), Ra, Rb, wk )

    return [ Tene_l, Dthm_l, Wadv_l, Wdr_l, Wbg_l ]



def composition_worker( l ):
    '''
    NEW (2026-11-08): the budget of the composition, l-component (l a poloidal l), as thermal_worker's of the entropy with
    ρ₀ in place of p₀. The equation of section c (operators.composition, comp_advection, comp_diffusion),
        ρ₀ ∂c/∂t = -ρ₀ uᵣ dC/dr + CompoD ∇⋅(D ρ₀ ∇c)   (- ρ₀ i m G ΔΩ c with diff_rot = 1),
    times c* and integrated over the fluid gives 2σ CE = Wadv + CompoD Dcmp + Wdr, with
        CE   = ½ ∫ ρ₀ |c|² dV                           the composition's "energy"
        Wadv = -∫ ρ₀ (dC/dr) uᵣ c* dV                   the advection of the background composition
        Dcmp = ∫ c* ∇⋅(D ρ₀ ∇c) dV = -∫ D ρ₀ |∇c|² dV + [ r² D ρ₀ c* ∂c/∂r ]   its diffusion
        Wdr  = -2 Re ∫ ρ₀ c* i m G ΔΩ c dV              (diff_rot_setup, drcomp)
        Wbg  = -2 Re ∫ ρ₀ c* (𝐔⋅∇c + 𝐮⋅∇C) dV          (a background flow, bgflow_setup, bgcomp)
    (real parts, the 2Re convention of thermal_worker). Uses the tables of set_solution (csol). Returns
    [ CE_l, Dcmp_l, Wadv_l, Wdr_l, Wbg_l ], Dcmp_l without the CompoD factor.
    '''
    Ra = par.ricb
    Rb = ut.rcmb

    lp  = ut.ell( par.m, par.lmax, par.symm)[0]
    idx = list(lp).index(l)
    L   = l*(l+1)
    nl  = 4*np.pi/(2*l+1)

    f_c = radial( ctab, idx, 1, rk )
    c0, c1 = f_c[0], f_c[1]   # c, ∂c/∂r

    [ velq, _, _ ] = velocity(l)  # velq[0] is uᵣ

    Cene_l = rad_quad( 0.5*rho0*dotprod_scal(l, c0, c0), Ra, Rb, wk )
    Wadv_l = rad_quad( -rdC0*dotprod_scal(l, velq[0], c0), Ra, Rb, wk )

    Dcmp_l = 0
    if par.CompoD > 0:
        Dcmp_l = rad_quad( -Drh0*nl*2*( r2*np.abs(c1)**2 + L*np.abs(c0)**2 ), Ra, Rb, wk )
        [ rb, Drhb, cbnd ] = csfc   # the surface term [ r² D ρ₀ c* ∂c/∂r ] from Ra to Rb
        f_cb = radial( cbnd, idx, 1, rb )
        sfc  = nl * rb**2 * Drhb * 2*np.real( np.conj(f_cb[0]) * f_cb[1] )
        Dcmp_l += sfc[1] - sfc[0]

    Wdr_l = 0
    if par.diff_rot:
        Wdr_l = -rad_quad( dotprod_scal(l, drcomp[l], c0), Ra, Rb, wk )

    Wbg_l = 0   # NEW (2026-11-08): a background flow, -2 Re ∫ ρ₀ c* (𝐔⋅∇c + 𝐮⋅∇C) dV (bgflow_setup)
    if par.bgflow:
        Wbg_l = -rad_quad( dotprod_scal(l, bgcomp[l], c0), Ra, Rb, wk )

    return [ Cene_l, Dcmp_l, Wadv_l, Wdr_l, Wbg_l ]



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



def pressure_force(l, eigval):
    '''
    NEW (2026-11-04): the pressure gradient ∇Π = 𝐅 - λ𝐮, l-component (l of the poloidals), from the momentum equation
    per unit mass λ𝐮 = 𝐅 - ∇Π, with Π = p′/ρ₀ (LBR) and 𝐅 every other force, as in flow_worker: the Coriolis force
    -coriolis, (∇⋅𝛔)/ρ₀ (visforce), the buoyancy (radial, so not in the consoidal part), the differential rotation
    (drforce) and (NEW 2026-11-08) a background flow (bgforce, with its gradient part). eigval = λ = σ + iω, the fields
    ∝ e^{λt} (forced runs: i forcing_frequency). Returns [ [q, ∂ᵣq], [s, ∂ᵣs] ] of ∇Π at rk: Π_l = r s (and Π_l′ = q
    where the momentum rows are solved). Without the Lorentz force, which wall_powers adds at the walls where it is needed
    (NEW 2026-11-06: lorentz_walls).
    '''
    [ velq, vels, _ ] = velocity(l)
    [ corq, cors, _ ] = coriolis(l)
    q = [ -corq[d] - eigval*velq[d] for d in (0,1) ]
    s = [ -cors[d] - eigval*vels[d] for d in (0,1) ]
    forces = []
    if par.ViscosD > 0:
        forces.append( visforce(l) )
    if par.thermal or comp_on():   # (NEW 2026-11-08: and the composition's)
        forces.append( buoyancy(l) )
    if par.diff_rot:
        forces.append( drforce[l] )
    if par.bgflow:   # NEW (2026-11-08)
        forces.append( bgforce[l] )
    for [ fq, fs, _ ] in forces:
        q = [ q[d] + fq[d] for d in (0,1) ]
        s = [ s[d] + fs[d] for d in (0,1) ]
    return [ q, s ]



def walls_move():
    '''
    NEW (2026-11-06): whether a wall can move radially, so that the pressure works on the fluid there (wall_powers): a forced
    problem (a boundary flow with uᵣ ≠ 0 at a wall) or a wall with zero radial stress (bco or bci = 2)
    '''
    return bool(par.forcing != 0 or par.bco == 2 or (par.ricb > 0 and par.bci == 2))



def lorentz_walls(bsol):
    '''
    NEW (2026-11-06): the consoidal part s_l of the Lorentz force 𝐅 = (∇×𝐛)×𝐁₀ + (∇×𝐁₀)×𝐛 (no Hendrik, per unit volume) at the
    current radii rk, {l: s_l} for the degrees l of the flow's poloidal scalar: the part of 𝐅 that the pressure balances there
    (pressure_force leaves it out). From the field bsol (expand_reshape_sol with ut.bsymm), B0 and ∇×B0 of any degrees, on a
    Gauss-Legendre grid in cos θ (as magnetic_powers_grid), projected on r∇Y_l: s_l = ∮ 𝐅⋅(r∇Y_l)* dΩ / ∮ |r∇Y_l|² dΩ.
    '''
    m = par.m
    lmx = par.lmax + 1
    nB = max([3] + list(bf.pol_profiles()) + list(bf.tor_profiles()))
    xg, wx = np.polynomial.legendre.leggauss(lmx + nB + 12)
    Yg, dYg, Zg = angular.ylm_grid(m, lmx, xg)
    Y0, dY0, _ = angular.ylm_grid(0, nB, xg)
    lp_u = ut.ell(m, par.lmax, par.symm)[0]
    lp_b, lt_b, ll = ut.ell(m, par.lmax, ut.bsymm)
    set_bsolution(bsol, 2)   # f, f', f'' for b and ∇×b
    z = np.zeros_like(rk, dtype=complex)
    b = [0*np.outer(rk, xg)]*3; j = [0*np.outer(rk, xg)]*3
    for l in ll:
        if l in lp_b:
            [ [q, q1], [s_, s1] ] = bfield_pol(l, 1)
            bq = (q, s_, z); [ cr, cc, ct ] = curl(l, [q, q1], [s_, s1], [z, z])
        else:
            [ t, t1 ] = bfield_tor(l, 1)
            bq = (z, z, t); [ cr, cc, ct ] = curl(l, [z, z], [z, z], [t, t1])
        for F, (qq, ss_, tt) in ((b, bq), (j, (cr[0], cc[0], ct[0]))):
            F[0] = F[0] + np.outer(qq, Yg[l])
            F[1] = F[1] + np.outer(ss_, dYg[l]) + np.outer(tt, Zg[l])
            F[2] = F[2] + np.outer(ss_, Zg[l]) - np.outer(tt, dYg[l])
    B0 = [0*np.outer(rk, xg)]*3; J0 = [0*np.outer(rk, xg)]*3
    for n, f in bf.pol_profiles().items():   # cnorm ∇×∇×(h Y_n 𝐫), as magnetic_powers_grid
        L0 = n*(n + 1.); hh = [ f(rk, d) for d in (0, 1, 2) ]
        Dh = hh[2] + 2*hh[1]/rk - L0*hh[0]/rk**2
        B0 = [ B0[0] + np.outer(L0*hh[0]/rk, Y0[n]), B0[1] + np.outer(hh[1] + hh[0]/rk, dY0[n]), B0[2] ]
        J0 = [ J0[0], J0[1], J0[2] + np.outer(Dh, dY0[n]) ]
    for n, f in bf.tor_profiles().items():   # cnorm_tor ∇×(k Y_n 𝐫)
        L0 = n*(n + 1.); kk = [ f(rk, d) for d in (0, 1) ]
        B0 = [ B0[0], B0[1], B0[2] - np.outer(kk[0], dY0[n]) ]
        J0 = [ J0[0] + np.outer(L0*kk[0]/rk, Y0[n]), J0[1] + np.outer(kk[1] + kk[0]/rk, dY0[n]), J0[2] ]
    cross = lambda a, c: [ a[1]*c[2] - a[2]*c[1], a[2]*c[0] - a[0]*c[2], a[0]*c[1] - a[1]*c[0] ]
    G = [ a + c for a, c in zip(cross(j, B0), cross(J0, b)) ]
    out = {}
    for l in lp_u:
        nrm = np.sum(wx*(np.abs(dYg[l])**2 + np.abs(Zg[l])**2))
        out[l] = np.sum(wx[None, :]*(G[1]*np.conj(dYg[l])[None, :] + G[2]*np.conj(Zg[l])[None, :]), axis=1)/nrm
    return out



def wall_powers(usol, tsol, eigval, bsol=None, csol=0):
    '''
    NEW (2026-11-04): the power of the wall stresses on the fluid, for [inner, outer] wall (inner 0 in a full sphere),
        pressure  Wp = -2 ∮ ρ₀ Π uᵣ* dA (complex: its real part closes residσ, its imaginary part residω),
        viscous   Wv =  2 Re ∮ 𝐮*⋅𝛔⋅𝐧 dA,   𝛔 = ρ₀ ν ViscosD [ ∇𝐮 + (∇𝐮)ᵀ - ⅔(∇⋅𝐮) 𝐈 ],   ∇⋅𝐮 = -(ln ρ₀)′ uᵣ,
    with 𝐧 = -𝐫̂ at the inner wall; in the convention of dotprod_pol (the physical field is 𝐮 e^{imφ} + c.c.).
    𝛔⋅𝐫̂ = ρ₀ ν ViscosD [ (2q′ + ⅔(ln ρ₀)′q) Y 𝐫̂ + (s′ - s/r + q/r) r∇Y + (t′ - t/r) r∇Y×𝐫̂ ]. Π from pressure_force at
    the walls (eigval as there). Dkin = ∫ 𝐮⋅(∇⋅𝛔) dV = Wv - Dν (viscous_dissipation). Both are zero where uᵣ = 0 (Wp) or 𝐮 = 0 (Wv).
    Sets the grid globals at the walls: call after the other diagnostics (diagnose, diagnose_magnetic).
    NEW (2026-11-06): bsol, the field (magnetic = 1): where a wall can move radially (walls_move) Π includes the Lorentz force's
    part, Hendrik s_l/ρ₀ (lorentz_walls), which a forced problem with a field needs. NEW (2026-11-08): csol, the composition.
    '''
    rw = np.array([ par.ricb, ut.rcmb ]) if par.ricb > 0 else np.array([ ut.rcmb ])
    set_radii(rw)
    set_solution(usol, tsol, csol=csol)
    if par.diff_rot:
        diff_rot_setup()
    if par.bgflow:   # NEW (2026-11-08)
        bgflow_setup()
    [ lp, lt, ll ] = ut.ell(par.m, par.lmax, par.symm)
    nu = rho0*vsc0*par.ViscosD
    Wp = np.zeros(len(rw), dtype=complex)
    Wv = np.zeros(len(rw))
    lor = lorentz_walls(bsol) if (bsol is not None and walls_move()) else {}   # NEW (2026-11-06)
    for l in ll:
        [ velq, vels, velt ] = velocity(l)
        if l in lp:
            Pi = rk*pressure_force(l, eigval)[1][0]
            if l in lor:
                Pi = Pi + rk*par.Hendrik*lor[l]/rho0
            Wp += -2*rho0*cdot_pol(l, velq[0], vels[0], Pi, 0*Pi)
            tq = 2*velq[1] + (2/3)*lho1*velq[0]
            ts = vels[1] - vels[0]/rk + velq[0]/rk
            Wv += nu*dotprod_pol(l, velq[0], vels[0], tq, ts)
        else:
            tt = velt[1] - velt[0]/rk
            Wv += nu*dotprod_tor(l, velt[0], tt)
    if par.ricb > 0:
        return [ Wp*np.array([-1, 1]), Wv*np.array([-1, 1]) ]
    return [ np.concatenate(([0], Wp)), np.concatenate(([0], Wv)) ]



def viscous_dissipation():
    '''
    NEW (2026-11-04): the viscous dissipation Dν = ∫ 𝛔:∇𝐮 dV = ∫ 2 ρ₀ ν ViscosD 𝐞_d:𝐞_d dV ≥ 0, 𝐞_d = 𝐞 - ⅓(∇⋅𝐮)𝐈 the deviatoric
    part of the rate of strain 𝐞 = ½[∇𝐮 + (∇𝐮)ᵀ] (𝛔 = 2 ρ₀ ν ViscosD 𝐞_d), in the convention of dotprod_pol (the physical
    field is 𝐮 e^{imφ} + c.c., so 2 𝐞_d:𝐞_d* for the complex amplitudes). Computed on an (rk, cosθ) grid from the velocity
    (Gauss-Legendre in cosθ, exact for these polynomials), independently of visforce: Dkin = Wv - Dν (wall_powers) checks
    both. Call after diagnose (at its grid).
    '''
    m = par.m
    [ lp, lt, ll ] = ut.ell(par.m, par.lmax, par.symm)
    x, wx = np.polynomial.legendre.leggauss( par.lmax + 12 )
    sth = np.sqrt(1 - x**2)
    [ Yg, dYg, Zg ] = angular.ylm_grid(m, int(max(ll)), x)    # Y, ∂θY, i m Y/sinθ, rows l
    Vg = Zg/(1j*m) if m else 0*Yg                             # Y/sinθ (only used multiplied by m)
    L  = (np.arange(np.shape(Yg)[0])*(np.arange(np.shape(Yg)[0]) + 1.))[:,None]
    d2Yg = -x*dYg/sth - L*Yg + m**2*Vg/sth                     # ∂θ²Y, from the Legendre equation
    dZg  = 1j*m*( dYg - x*Vg )/sth                             # ∂θ(i m Y/sinθ)

    # u and its r- and θ-derivatives on the grid (rows rk, columns cosθ): u_r, u_θ, u_φ = Σ q Y, s ∂θY + t Z, s Z - t ∂θY
    u, ur, ut_ = [ np.zeros((3, len(rk), len(x)), dtype=complex) for k in (0,1,2) ]
    for l in ll:
        [ q, s, t ] = velocity(l)
        Y, dY, Z, d2Y, dZ = Yg[l], dYg[l], Zg[l], d2Yg[l], dZg[l]
        for d, A in ((0, u), (1, ur)):
            A[0] += np.outer(q[d], Y)
            A[1] += np.outer(s[d], dY) + np.outer(t[d], Z)
            A[2] += np.outer(s[d], Z) - np.outer(t[d], dY)
        ut_[0] += np.outer(q[0], dY)
        ut_[1] += np.outer(s[0], d2Y) + np.outer(t[0], dZ)
        ut_[2] += np.outer(s[0], dZ) - np.outer(t[0], d2Y)

    r, st, ct, im = rk[:,None], sth[None,:], x[None,:], 1j*m
    err = ur[0]
    ett = ( ut_[1] + u[0] )/r
    epp = ( im*u[2] + ct*u[1] )/(r*st) + u[0]/r
    ert = 0.5*( ur[1] - u[1]/r + ut_[0]/r )
    erp = 0.5*( im*u[0]/(r*st) + ur[2] - u[2]/r )
    etp = 0.5*( ut_[2] - ct*u[2]/st + im*u[1]/st )/r
    div = err + ett + epp
    ee  = ( abs(err - div/3)**2 + abs(ett - div/3)**2 + abs(epp - div/3)**2
            + 2*( abs(ert)**2 + abs(erp)**2 + abs(etp)**2 ) )

    f = 2 * 2*rho0[:,None]*vsc0[:,None]*par.ViscosD * ee       # 2 (c.c.) × 2 ρ₀ ν ViscosD 𝐞_d:𝐞_d*
    return 2*np.pi*np.sum( (wk*rk**2)[:,None]*wx[None,:]*f )*(ut.rcmb - par.ricb)/2



def body_force_terms(usol, eigval, tsol=0, csol=0):
    '''
    NEW (2026-11-08): forcings 1 and 8, the terms of the body force 𝐟 that the known flow 𝐱 (utils.lifted_flow) puts in the
    momentum equation of the flow 𝐮 solved for (assemble.lifted_rhs): 𝐟 = -[λ𝐱 + 2 Gaspard 𝐳×𝐱 - (∇⋅𝛔(𝐱))/ρ₀ - 𝐅(𝐱)], the
    equations of 𝐱 itself (𝐅 the force of a differential rotation, diff_rot_setup, and NEW 2026-11-08 of a background flow,
    bgflow_setup; 𝐱 carries no entropy and no field), with
    λ = eigval = i forcing_frequency. Returns
      Wf,   the complex power 2 ∫ ρ₀ 𝐮*⋅𝐟 dV (its real part in residσ, its imaginary part in residω);
      Ensf, its enstrophy term ∫ (∇×𝐮)⋅(∇×𝐟) dV, weighted as flow_worker's (residens);
      Tf,   the torque of 𝐟 on the fluid, factor ∫ ρ₀ r³ t₁ dr from the toroidal part t₁ of 𝐟 of degree 1 (rotdyn.angular_momentum's
            normalisation, axial for m = 0 and equatorial for m = 1; resid𝐋).
    The gradients in 𝐟 (the inertia and the viscous force of the equilibrium tide, the Coriolis force on the rigid libration)
    do no work between impenetrable walls, have no torque and no curl: they leave the three terms unchanged. On the
    quadrature grid of diagnose, by the routines of flow_worker; call after diagnose. It leaves the tables of the solution
    usol, tsol (the entropy, as given by expand_reshape_sol, with thermal = 1) and (NEW 2026-11-08) csol (the composition)
    at that grid.
    '''
    Ra, Rb = par.ricb, ut.rcmb
    [ lp, lt, ll ] = ut.ell(par.m, par.lmax, par.symm)
    xsol = expand_reshape_sol( ut.lifted_flow()[:2*ut.n], par.symm )
    setup_grid(Ra, Rb)
    set_solution(xsol, np.zeros((len(lp), par.N), complex) if par.thermal else 0)   # 𝐱 carries no entropy (diff_rot_setup)
    if par.diff_rot:
        diff_rot_setup()   # drforce of 𝐱
    if par.bgflow:
        bgflow_setup()     # bgforce of 𝐱 (NEW 2026-11-08)
    F = {}
    for l in ll:
        v, c = velocity(l), coriolis(l)
        f = [ [ -(eigval*v[k][d] + c[k][d]) for d in (0, 1) ] for k in (0, 1, 2) ]
        if par.ViscosD > 0:
            vf = visforce(l)
            f = [ [ f[k][d] + vf[k][d] for d in (0, 1) ] for k in (0, 1, 2) ]
        if par.diff_rot:
            f = [ [ f[k][d] + drforce[l][k][d] for d in (0, 1) ] for k in (0, 1, 2) ]
        if par.bgflow:
            f = [ [ f[k][d] + bgforce[l][k][d] for d in (0, 1) ] for k in (0, 1, 2) ]
        F[l] = f
    set_solution(usol, tsol, csol=csol)
    Wf, Ensf, Tf = 0j, 0.0, 0j
    for l in ll:
        [ velq, vels, velt ] = velocity(l)
        [ fq, fs, ft ] = F[l]
        Wf += rad_quad( 2*( cdot_pol(l, velq[0], vels[0], fq[0], fs[0]) + cdot_tor(l, velt[0], ft[0]) )*rho0, Ra, Rb, wk )
        [ cuq, cus, cut ] = curl(l, velq, vels, velt)
        [ cfq, cfs, cft ] = curl(l, fq, fs, ft)
        Ensf += rad_quad( ( dotprod_pol(l, cuq[0], cus[0], cfq[0], cfs[0]) + dotprod_tor(l, cut[0], cft[0]) )*const0**2,
                          Ra, Rb, wk )
        if l == 1 and l in lt:
            import rotdyn as rd
            Tf = rd._factor()*rad_quad( rho0*rk**3*ft[0], Ra, Rb, wk )
    return [ Wf, Ensf, Tf ]




def comp_on():
    '''NEW (2026-11-08): whether the tables of a solution of the composition are set (compositional = 1, set_solution)'''
    return bool(par.compositional) and 'csol2' in globals() and np.ndim(csol2) == 2



def buoyancy(l, which='all'):
    '''
    Returns the l-degree (rad,con,tor) components of the buoyancy force, and its radial derivatives.
    Use it to compute the rate of working (power) of the thermal buoyancy
    NEW (2026-11-08): which = 'thermal', the entropy's Beyonce g s 𝐫̂, 'comp', the composition's BeyonceC g c 𝐫̂ (with its
    tables, comp_on), or 'all', the buoyancy force, their sum (each alone when the other is absent)
    '''
    if which == 'all':
        th = buoyancy(l, 'thermal') if par.thermal else None
        cp = buoyancy(l, 'comp') if comp_on() else None
        if th is not None and cp is not None:
            return [ [ th[k][d] + cp[k][d] for d in (0,1,2) ] for k in (0,1,2) ]
        if th is not None or cp is not None:
            return th if th is not None else cp
    tab, fac = (ctab, par.BeyonceC) if which == 'comp' else (ttab if par.thermal else None, par.Beyonce)

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

    if l in lp and tab is not None:
        idx   = list(lp).index(l)
        f_pol = radial( tab, idx, 2, rk )  # NEW (2026-10-20): from the table, and g from set_solution
        [ grv0, grv1, grv2 ] = grv
        out_rad0 = f_pol[0] * grv0
        out_rad1 = f_pol[1] * grv0 + f_pol[0] * grv1
        out_rad2 = f_pol[2] * grv0 + f_pol[1] * grv1 + f_pol[1] * grv1 + f_pol[0] * grv2
        
    return [ [out_rad0*fac, out_rad1*fac, out_rad2*fac],
             [out_con0*fac, out_con1*fac, out_con2*fac],
             [out_tor0*fac, out_tor1*fac, out_tor2*fac] ]






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
    set_radii( 0.5*(Rb-Ra)*( xk + 1 ) + Ra )



def set_radii(radii):
    '''
    NEW (2026-11-04): sets the radii rk and the background profiles sampled there (rho0, lho1..lho5, vsc0..vsc3, const0,
    pss0, ...) as module globals, for the workers below. setup_grid calls it at the quadrature nodes, diagnose_4plot at
    the plot radii and wall_powers at the walls.
    '''
    global rk
    rk = radii

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

    if par.compositional:
        # NEW (2026-11-08): background profiles for the budget of the composition (composition_worker), ρ₀ is rho0
        global rdC0
        rdC0 = rap.rdCdrX( rk, 0)                                 # ρ₀ dC/dr
        global Drh0
        Drh0 = rap.prf.comp_diffusivity( rk, 0) * rho0            # D ρ₀ (equal to rap.DrhoX( rk, 0))



def set_solution(usol, tsol, ndp=5, ndt=4, csol=0):
    '''
    NEW (2026-10-20): sets the solution usol2, tsol2 (as given by expand_reshape_sol) as module globals for the
    workers, together with the tables of its radial functions at the current radii rk (cheb2space_table):
    utab = [ poloidal up to the ndp-th r-derivative, toroidal up to the ndt-th ] (visforce needs 5 and 4,
    velocity alone 3 and 2) and, with thermal = 1 and an entropy solution, ttab (up to the 2nd derivative),
    g and its derivatives at rk (grv, for buoyancy) and s, ∂s/∂r and κ p₀ at the walls (tsfc, for thermal_worker).
    NEW (2026-11-08): with compositional = 1 and a solution of the composition csol (expand_reshape_sol), csol2 and the same
    for it: ctab, grv, and c, ∂c/∂r and D ρ₀ at the walls (csfc, for composition_worker).
    Call it after rk is set (setup_grid, diagnose_4plot).
    '''
    global usol2, tsol2, utab, ttab, grv, tsfc, csol2, ctab, csfc
    usol2 = usol
    tsol2 = tsol
    csol2 = csol
    utab  = [ cheb2space_table(usol[0], ndp, rk), cheb2space_table(usol[1], ndt, rk) ]
    if par.thermal and np.ndim(tsol) == 2:
        ttab = cheb2space_table(tsol, 2, rk)
        grv  = [ rap.graviX(rk, d) for d in (0,1,2) ]
        rb   = np.array([ par.ricb, ut.rcmb ])
        tsfc = [ rb, rap.prf.thermal_diffusivity(rb, 0) * rap.pressX(rb, 0), cheb2space_table(tsol, 1, rb) ]
    if par.compositional and np.ndim(csol) == 2:
        ctab = cheb2space_table(csol, 2, rk)
        grv  = [ rap.graviX(rk, d) for d in (0,1,2) ]
        rb   = np.array([ par.ricb, ut.rcmb ])
        csfc = [ rb, rap.prf.comp_diffusivity(rb, 0) * rap.densX(rb, 0), cheb2space_table(csol, 1, rb) ]



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



def zero_flow():
    '''
    NEW (2026-11-06): a zero flow, in the shape that expand_reshape_sol gives, for the post-processing of a run without the
    flow (hydro = 0): the flow's diagnostics then give zeros and set the grid and tables as with a flow.
    '''
    nl = int((par.lmax - par.m + 1)/2)
    return [ np.zeros((nl, par.N), dtype=complex), np.zeros((nl, par.N), dtype=complex) ]



def thermal_energy(tsol, Ra, Rb, ls=None):
    '''
    NEW (2026-11-06): the entropy "energy" TE = ½ ∫ p₀ |s|² dV of the entropy solution tsol (expand_reshape_sol), from Ra to
    Rb, with the integrand and the quadrature of thermal_worker (spin_doctor's TE). ls restricts the sum to some l values
    (default: all; only the l's of the entropy count). For solve.py's normalisation without the flow (hydro = 0); the tables
    it leaves (a zero flow, tsol) serve centrifugal_terms for the metric part TEc.
    '''
    setup_grid(Ra, Rb)
    set_solution(zero_flow(), tsol, 3, 2)
    lp = list(ut.ell(par.m, par.lmax, par.symm)[0])
    out = 0.0
    for l in (lp if ls is None else ls):
        if l in lp:
            hlm0 = radial(ttab, lp.index(l), 0, rk)[0]
            out += rad_quad(entropy_energy(l, hlm0), Ra, Rb, wk)
    return out



def composition_energy(csol, Ra, Rb, ls=None):
    '''
    NEW (2026-11-08): CE = ½ ∫ ρ₀ |c|² dV of the solution of the composition csol (expand_reshape_sol), from Ra to Rb, with
    the integrand and the quadrature of composition_worker; ls as thermal_energy. For solve.py's normalisation without the
    flow (hydro = 0); the tables it leaves (a zero flow, csol) serve centrifugal_composition_terms.
    '''
    setup_grid(Ra, Rb)
    set_solution(zero_flow(), 0, 3, 2, csol=csol)
    lp = list(ut.ell(par.m, par.lmax, par.symm)[0])
    out = 0.0
    for l in (lp if ls is None else ls):
        if l in lp:
            c0 = radial(ctab, lp.index(l), 0, rk)[0]
            out += rad_quad(0.5*rho0*dotprod_scal(l, c0, c0), Ra, Rb, wk)
    return out



def centrifugal_terms(Ra, Rb):
    '''
    NEW (2026-11-05): with centrifugal = 1 (centrifugal.py), the metric parts of the budgets, from the solution tables at the
    current radii rk (after setup_grid and set_solution, as kinetic_energy and diagnose leave them), in the conventions of
    flow_worker and thermal_worker (KE = ∫ ρ0 |u|² dV in their units, the complex powers 2 ∫ ρ0 u*⋅F dV):
        KEc  = ∫ ρ0 w*⋅(M - I) w dV                the metric part of the kinetic energy: KE + KEc = ∫ ρ0 w*⋅M w dV is the
                                                   kinetic energy of the deformed body, the one B weights
        Wcor = 2 ∫ ρ0 w*⋅[2 Gaspard (z̃ - ẑ)×w] dV  the metric part of the Coriolis power (imaginary: it does no work)
        TEc  = ∫ p0 (J - 1) |s|² dV = ∫ p0 η P2 |s|² dV   (0 without an entropy solution)
    The metric terms are those of the operators, applied to the l-components (q, s, t) of w at each radius with the angular
    couplings of centrifugal.primitives and the radial factors ε, η of the profile codes ce0, ce1. Returns [KEc, Wcor, TEc].
    NEW (2026-11-06): the exact spheroid (centrifugal_shape 'spheroid'), with the couplings of centrifugal.
    spheroid_primitives and no radial factor; TEc = 0 (J = 1).
    '''
    m = par.m
    lp, lt, ll = ut.ell(par.m, par.lmax, par.symm)
    lw = cf.lw_of()
    nr = len(rk)
    W = {k: np.zeros((lw + 1, nr), dtype=complex) for k in 'qst'}   # rows: degree 0..lw, columns: radii
    for l in ll:
        [ q, s, t ] = velocity(l)
        W['q'][l], W['s'][l], W['t'][l] = q[0], s[0], t[0]
    prof = cf.profile_functions()
    amp = {'e': prof['ce0'](rk, 0), 'h': prof['ce1'](rk, 0)}
    lv = np.arange(lw + 1)
    nl, Lv = 4*np.pi/(2*lv + 1), lv*(lv + 1.)
    def apply(kind):   # the metric term as l-components at each radius
        out = {k: np.zeros_like(W['q']) for k in 'qst'}
        if cf.exact():   # NEW (2026-11-06): the exact spheroid, constant couplings
            C = cf.spheroid_primitives(kind, m, lw)[0]
            for k in 'qst':
                out[k] += C[k]['q'] @ W['q'] + C[k]['a'] @ W['s'] + C[k]['T'] @ W['t']
            return out
        for a, f in amp.items():
            C = cf.primitives(kind, a, m, lw)[0]
            for k in 'qst':
                out[k] += (C[k]['q'] @ W['q'] + C[k]['a'] @ W['s'] + C[k]['T'] @ W['t'])*f[None, :]
        return out
    def vdot(F):       # ∫ ρ0 w*⋅F dV, as cdot_pol + cdot_tor summed over l
        g = nl[:, None]*(np.conj(W['q'])*F['q'] + Lv[:, None]*(np.conj(W['s'])*F['s'] + np.conj(W['t'])*F['t']))
        return rad_quad(np.sum(g, 0)*r2*rho0, Ra, Rb, wk)
    KEc  = np.real(vdot(apply('inertia')))
    Wcor = 2*2*par.Gaspard*vdot(apply('coriolis'))
    TEc  = 0.0
    if par.thermal and 'ttab' in globals() and np.ndim(tsol2) == 2 and not cf.exact():   # (NEW 2026-11-06: J = 1, none)
        Sg = np.zeros((lw + 1, nr), dtype=complex)
        for k, l in enumerate(lp):
            Sg[l] = radial(ttab, k, 0, rk)[0]
        P2 = angular.Algebra.get(m, lw).legendre(2)[0]
        g = nl[:, None]*np.conj(Sg)*(P2 @ Sg)*amp['h'][None, :]
        TEc = np.real(rad_quad(np.sum(g, 0)*r2*pss0, Ra, Rb, wk))
    return [ KEc, Wcor, TEc ]



# NEW (2026-11-06, phase 2): the metric parts of the dissipative and differential-rotation budgets with centrifugal = 1,
# from the expressions of centrifugal.py (sections 6 and 7) evaluated on the grid. They are the operators before their
# projection on kore's rows, up to a gradient, whose power vanishes (∇⋅(ρ0 w) = 0 and w_r = 0 at the walls).
def _cf_table(C, ls, nd, rr, lw):
    '''[(lw + 1, nr) array for d = 0..nd]: the l-components of a scalar unknown and of its r-derivatives at the radii rr, from
    its Chebyshev coefficients C (rows: the degrees ls)'''
    T = cheb2space_table(C, nd, rr)
    out = [np.zeros((lw + 1, len(rr)), dtype=complex) for d in range(nd + 1)]
    for k, l in enumerate(ls):
        for d in range(nd + 1):
            out[d][l] = T[d, k]
    return out

def _cf_order(*es):
    '''the highest r-derivative of an unknown in the expressions es'''
    return max([d for e in es for (p, X, d, a, j, lam) in e if X is not None] + [0])

def _cf_eval(e, tabs, rr, amp, coef, vector=True):
    '''
    The l-components at the radii rr of an expression e of centrifugal.py (section 6), Σ r^p a(r) c^(j)(r) Λ(r) M X^(d) over
    its monomials (p, X, d, a, j, lam): tabs[X][d] the l-components of the unknown X^(d), amp[a] the amplitude a at rr,
    coef(j) the j-th derivative of the profile in the slot c, Λ the product of the (ln ρ0) derivatives λ_k. A vector
    expression gives {'q', 's', 't'}, a scalar one {'s'}.
    '''
    shape = tabs[next(iter(tabs))][0].shape
    keys = 'qst' if vector else 's'
    out = {k: np.zeros(shape, dtype=complex) for k in keys}
    cj, lk = {}, {}
    for (p, X, d, a, j, lam), M in e.items():
        rad = rr**p
        if a is not None:
            rad = rad*amp[a]
        if j is not None:
            if j not in cj:
                cj[j] = coef(j)
            rad = rad*cj[j]
        for k in lam:
            if k not in lk:
                lk[k] = rap.logrhoX(rr, k)
            rad = rad*lk[k]
        V = tabs[X][d]
        for k in keys:
            Mk = getattr(M, k) if vector else M
            if Mk.nnz:
                out[k] += (Mk @ V)*rad[None, :]
    return out

def _cf_vdot(W, F, weight, Ra, Rb):
    '''∫ weight W*⋅F dV (cdot_pol + cdot_tor summed over l, as centrifugal_terms' vdot)'''
    lv = np.arange(W['q'].shape[0]); nl, Lv = (4*np.pi/(2*lv + 1))[:, None], (lv*(lv + 1.))[:, None]
    g = nl*(np.conj(W['q'])*F['q'] + Lv*(np.conj(W['s'])*F['s'] + np.conj(W['t'])*F['t']))
    return rad_quad(np.sum(g, 0)*r2*weight, Ra, Rb, wk)

_cf_expr = {}
def _cf_expressions(kind, lw):
    '''the expressions of centrifugal.py for the budgets, built once: 'flow' (w of P, T), 'visc' (δV of P, T), 'heat'
    (-∇⋅(κp (M - I)∇s)), NEW (2026-11-08) 'comp' (-∇⋅(D ρ0 (M - I)∇c)), 'rot' ({n: δW of P, T}), 'mag' (the metric part of
    the diffusion's field of F, G, as magnetic_diffusion_metric, and its curl)'''
    key = (kind, lw)
    if key not in _cf_expr:
        A = angular.Algebra.get(par.m, lw)
        if kind == 'flow':
            _cf_expr[key] = cf._sum(cf._flow(A, 'P'), cf._flow(A, 'T'))
        elif kind == 'visc' and cf.exact():   # NEW (2026-11-06): the exact spheroid's, times ν (the profile slot)
            _cf_expr[key] = cf._rmul(cf.NU, cf._sum(cf.spheroid_terms(A, 'P', 'visc'), cf.spheroid_terms(A, 'T', 'visc')))
        elif kind == 'heat' and cf.exact():   # NEW (2026-11-06): times κ p
            _cf_expr[key] = cf._rmul(cf.NU, cf.spheroid_terms(A, 'S', 'heat'))
        elif kind == 'comp' and cf.exact():   # NEW (2026-11-08): the composition's, times D ρ0
            _cf_expr[key] = cf._rmul(cf.NU, cf.spheroid_terms(A, 'C', 'heat'))
        elif kind == 'comp':
            _cf_expr[key] = cf.thermal_diffusion_metric(A, lw, 'C')
        elif kind == 'visc':
            _cf_expr[key] = cf._sum(cf.viscous_force(A, 'P'), cf.viscous_force(A, 'T'))
        elif kind == 'heat':
            _cf_expr[key] = cf.thermal_diffusion_metric(A, lw)
        elif kind == 'rot':
            _cf_expr[key] = {n: cf._sum(cf.rotation_force(A, 'P', n, lw), cf.rotation_force(A, 'T', n, lw)) for n in dr.degrees()}
        elif kind == 'mag':
            if cf.exact():   # NEW (2026-11-06, step 2): the exact spheroid's, times η (the profile slot)
                e = cf._rmul(cf.NU, cf._sum(cf.spheroid_terms(A, 'F', 'mag'), cf.spheroid_terms(A, 'G', 'mag')))
            else:
                e = cf._sum(cf.magnetic_diffusion_metric(A, 'F', lw), cf.magnetic_diffusion_metric(A, 'G', lw))
            _cf_expr[key] = (e, cf._curl(A, e))
    return _cf_expr[key]

def centrifugal_dissipation_terms(Ra, Rb):
    '''
    NEW (2026-11-06, phase 2): with centrifugal = 1, the metric parts of the viscous power, of the power of the differential
    rotation and of the entropy diffusion, from the solution at the current radii rk (after diagnose, before wall_powers
    moves the grid), in the conventions of flow_worker and thermal_worker:
        Dkin = 2 ∫ ρ0 w*⋅(ViscosD δV(w)) dV       complex: its real part goes to Dkin, its imaginary part to Dkin_im
        Wdr  = 2 ∫ ρ0 w*⋅(-Gaspard δW(w)) dV      complex, to Wdr and Wdr_im (zero without diff_rot)
        Dthm = 2 Re ∫ s* δ dV,  δ = -∇⋅(κp (M - I)∇s)   without ThermaD, as thermal_worker (zero without ThermaD)
    δV and δW are the metric parts of the viscous force and of the force of the differential rotation (centrifugal.
    viscous_force, rotation_force), δ that of the entropy diffusion (thermal_diffusion_metric), on the degrees that the
    operators use. The advection of the entropy by ΔΩ has a metric part (i m G η P2 ΔΩ s) whose power is imaginary: none
    here. Returns [Dkin_c, Wdr_c, Dthm_c].
    NEW (2026-11-06): the exact spheroid (centrifugal_shape 'spheroid'): δV and δ of centrifugal.spheroid_terms, exact.
    '''
    lp, lt, _ = ut.ell(par.m, par.lmax, par.symm)
    prof = cf.profile_functions()
    amp = {a: prof[c](rk, 0) for a, (c, _) in cf.AMPS.items()}
    Dkin_c = Wdr_c = 0j
    Dthm_c = 0.0
    if par.ViscosD > 0 or par.diff_rot:
        lw = par.lmax + 16 + (dr.n_max() if par.diff_rot else 0)
        ev = {'visc': _cf_expressions('visc', lw)} if par.ViscosD > 0 else {}
        if par.diff_rot:
            ev['rot'] = _cf_expressions('rot', lw)
        w_e = _cf_expressions('flow', lw)
        nd = _cf_order(w_e, *[e for k, e in ev.items() if k != 'rot'], *(ev['rot'].values() if 'rot' in ev else []))
        tabs = {'P': _cf_table(usol2[0], lp, nd, rk, lw), 'T': _cf_table(usol2[1], lt, nd, rk, lw)}
        W = _cf_eval(w_e, tabs, rk, amp, None)
        if 'visc' in ev:
            V = _cf_eval(ev['visc'], tabs, rk, amp, lambda j: rap.viscoX(rk, j))
            Dkin_c = 2*par.ViscosD*_cf_vdot(W, V, rho0, Ra, Rb)
        if 'rot' in ev:
            for n, e in ev['rot'].items():
                F = _cf_eval(e, tabs, rk, amp, lambda j, n=n: dr.fX(n, rk, j))
                Wdr_c += -2*par.Gaspard*_cf_vdot(W, F, rho0, Ra, Rb)
    if par.thermal and par.ThermaD > 0 and 'ttab' in globals() and np.ndim(tsol2) == 2:
        lw = par.lmax + 16
        e = _cf_expressions('heat', lw)
        S = {'S': _cf_table(tsol2, lp, _cf_order(e), rk, lw)}
        D = _cf_eval(e, S, rk, amp, lambda j: rap.kappressX(rk, j), vector=False)['s']
        lv = np.arange(lw + 1); nl = (4*np.pi/(2*lv + 1))[:, None]
        Dthm_c = 2*np.real(rad_quad(np.sum(nl*np.conj(S['S'][0])*D, 0)*r2, Ra, Rb, wk))
    return [ Dkin_c, Wdr_c, Dthm_c ]

def centrifugal_composition_terms(Ra, Rb):
    '''
    NEW (2026-11-08): with centrifugal = 1 and compositional = 1, the metric parts of the budget of the composition, from its
    tables at the current radii rk (set_solution with csol), as TEc (centrifugal_terms) and Dthm_c (centrifugal_dissipation_
    terms) for the entropy, with ρ₀ in place of p₀:
        CEc  = ∫ ρ₀ (J - 1) |c|² dV = ∫ ρ₀ η P2 |c|² dV   (0 in the exact spheroid, J = 1)
        Dcmp = 2 Re ∫ c* δ dV,  δ = -∇⋅(D ρ₀ (M - I)∇c)   without CompoD (0 without CompoD)
    Returns [CEc, Dcmp_c].
    '''
    m = par.m
    lp = ut.ell(par.m, par.lmax, par.symm)[0]
    CEc, Dcmp_c = 0.0, 0.0
    if not comp_on():
        return [ CEc, Dcmp_c ]
    lw = cf.lw_of()
    nr = len(rk)
    lv = np.arange(lw + 1)
    nlv = 4*np.pi/(2*lv + 1)
    prof = cf.profile_functions()
    if not cf.exact():
        Cg = np.zeros((lw + 1, nr), dtype=complex)
        for k, l in enumerate(lp):
            Cg[l] = radial(ctab, k, 0, rk)[0]
        P2 = angular.Algebra.get(m, lw).legendre(2)[0]
        g = nlv[:, None]*np.conj(Cg)*(P2 @ Cg)*prof['ce1'](rk, 0)[None, :]
        CEc = np.real(rad_quad(np.sum(g, 0)*r2*rho0, Ra, Rb, wk))
    if par.CompoD > 0:
        lw = par.lmax + 16
        amp = {a: prof[c](rk, 0) for a, (c, _) in cf.AMPS.items()}
        e = _cf_expressions('comp', lw)
        C = {'C': _cf_table(csol2, lp, _cf_order(e), rk, lw)}
        D = _cf_eval(e, C, rk, amp, lambda j: rap.DrhoX(rk, j), vector=False)['s']
        nl = (4*np.pi/(2*np.arange(lw + 1) + 1))[:, None]
        Dcmp_c = 2*np.real(rad_quad(np.sum(nl*np.conj(C['C'][0])*D, 0)*r2, Ra, Rb, wk))
    return [ CEc, Dcmp_c ]



# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------- NEW (2026-10-28): magnetic field (magnetic = 1)
# ----------------------------------------------------------------------------------------------------------------------
# Ported from conductive_IC (utils4pp.py at 39487bbf: energy_pol/tor, diffus_pol/tor, magnetic_worker, lorentz4pp_dipo,
# induction4pp_dipo), dipolar B0. b = ∇×∇×(f r) + ∇×(g r) carries no ρ: its components are q = L f/r, s = f' + f/r, t = g.
# The Lorentz force and the induction term are the Boussinesq ones (per unit volume, no ρ0):
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

    spectral = not use_grid()   # NEW (2026-10-28): otherwise the induction and Lorentz terms come from magnetic_powers_grid
    zero = np.zeros_like(rk, dtype=complex)
    [ qi, si, ti ] = induction4pp(l) if spectral else [ zero, zero, zero ]
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

    [ fq, fs, ft ] = lorentz4pp(l) if spectral else [ zero, zero, zero ]
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



def em_budget(bsol, bic=None, usol=None, bman=None):
    '''
    NEW (2026-10-28): the terms of the magnetic energy budget that the fluid-only resid𝐛 leaves out, ported from conductive_IC's
    em_budget for an insulating mantle and an insulating or conducting ('conducting, Chebys') inner core:
    [ Ohm_f, ME_ic, Ohm_ic, ME_ext, E_w, Ohm_w ], without Hendrik or MagnetD:
      Ohm_f  : ∫ |∇×𝐛|² over the fluid
      ME_ic  : ½ ∫ |𝐛|²/mu_i2o inside the inner core: the field of a conducting inner core (bic: its part of the solution
               vector; NEW 2026-11-06: 'conducting, Bessel' from the fluid's field at the inner wall, bessel_budget), or the
               potential field below an insulating one (below r - h with a thin wall)
      Ohm_ic : η_i ∫ |∇×𝐛|²/mu_i2o over a conducting inner core, η_i = 1/(mu_i2o sigma_i2o)
      ME_ext : ½ ∫ |𝐛|² of the potential field outside the outer wall (beyond r + h with a thin wall)
    NEW (2026-11-06), with thin conducting walls (mantle, innercore 'TWA'; conductive_IC's terms, 0 without them):
      E_w    : the magnetic energy stored in the walls: (c r/2) ∫ |𝐛_t|² dS of the poloidal field, the energy of the c term
               (the toroidal condition has no c term, so the wall stores no toroidal energy), plus (h/(2p)) ∫ |b_r|² dS,
               h = c r/p the wall's thickness (none for p = np.inf)
      Ohm_w  : the Ohmic loss of the wall currents, ∫ |𝐊|² dS/(c1 r), 𝐊 the jump of 𝐛_t across the wall: c1 |r ∇²_l f|² from
               the poloidal condition and |g|²/c1 from the toroidal one
    The field beyond the wall follows from the wall conditions (assemble.bc_b_twa), so no derivative beyond the second is
    needed. usol (the flow, expand_reshape_sol): with the fluid slipping along a thin wall, the wall's poloidal current takes
    the radial induction of the slip (slip_lapF), as the wall rows do, and of a radial boundary flow through a wall at rest
    (radial_lapF; forced_walls = 'rigid'), or half its toroidal part where the layer deforms across a toroidal B0
    (background.thin_drop). Assumes mu = 1 with thin walls.
    NEW (2026-11-06): with the resolved mantle (mantle = 'resolved'; bman, its part of the solution vector) E_w and Ohm_w are
    the layer's ½ ∫ |𝐛|² and ∫ |∇×𝐛|²/sigma (mantle_budget), and ME_ext the potential field beyond r_surface (nan without bman).
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
    E_w = Ohm_w = 0.0   # NEW (2026-11-06): thin walls
    if ut.twa_o:
        ME_ext, E_w, Ohm_w = twa_budget(bsol, 'outer', bvals, usol)
    elif ut.resolved_mantle:   # NEW (2026-11-06)
        E_w, Ohm_w, ME_ext = mantle_budget(bman) if bman is not None else (np.nan, np.nan, np.nan)
    else:
        f = bvals(bsol[0], 0, 1.0)[0]
        ME_ext = np.sum( f0p*Lp*lp*np.abs(f)**2 ) * r_o   # potential field r^-(l+1) beyond r_o (F = r f there, r_o = 1)

    if r_i == 0:   # NEW (2026-10-28): a full sphere, no inner core
        pass
    elif ut.twa_i:   # NEW (2026-11-06)
        ME_ic, Ew, Ohmw = twa_budget(bsol, 'inner', bvals, usol)
        E_w, Ohm_w = E_w + Ew, Ohm_w + Ohmw
    elif par.innercore == 'insulator':
        f = bvals(bsol[0], 0, -1.0)[0]
        ME_ic = np.sum( f0p*Lp*(lp+1)*r_i*np.abs(f)**2 )   # potential field r^l below r_i
    elif ut.bessel:   # NEW (2026-11-06): 'conducting, Bessel', in closed form
        ME_ic, Ohm_ic = bessel_budget(bsol, bvals)
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

    return [ Ohm_f, ME_ic, Ohm_ic, ME_ext, E_w, Ohm_w ]



def bessel_budget(bsol, bvals):
    '''
    NEW (2026-11-06): [ME_ic, Ohm_ic] of the inner core 'conducting, Bessel' in the units of em_budget (its integrands of a
    conducting inner core), in closed form, exact for any skin depth. Inside the core f = f_R j_l(k r)/j_l(k R) and
    g = g_R j_l(k r)/j_l(k R), R = ricb, f_R = f(R) and g_R = mu_i2o g(R) from the fluid's side (bc_b_cic), k and xd = R f'/f
    at R from utils.bessel_ic. Both solve ∇²_l f = -k² f with k² imaginary, so with F = r f, F'' = -k² r F + L F/r², and
    integrating |F'|² by parts,
        ∫ (L |f|² + |(r f)'|²) dr = R f_R* (r f)'(R) + k² ∫ r² |f|² dr = R (1 + xd) |f_R|² + k² I,
    whose imaginary part gives I = ∫ r² |f|² dr = -R Im(xd) |f_R|²/Im(k²) (Lommel's integral; R³/(2l+3) |f_R|² at k = 0) and
    whose real part is R (1 + Re xd) |f_R|². Then, with a_l = 4π/(2l+1), η_i = 1/(mu_i2o sigma_i2o) and ∇²_l f = -k² f,
        ME_ic  = Σ a_l L [ R (1 + Re xd) |f_R|² ] + Σ a_l L I_g,
        Ohm_ic = η_i Σ 2 a_l L |k|⁴ I_f + η_i Σ 2 a_l L R (1 + Re xd) |g_R|²,
    both divided by mu_i2o (𝐇 = 𝐛/mu_i2o), as for 'conducting, Chebys'.
    '''
    R = par.ricb
    lp, lt = ut.ell(par.m, par.lmax, ut.bsymm)[:2]
    eta_i = 1/(par.mu_i2o*par.sigma_i2o)
    fR = bvals(bsol[0], 0, -1.0)[0]
    gR = par.mu_i2o*bvals(bsol[1], 0, -1.0)[0]
    def parts(l):   # |k|⁴, R (1 + Re xd) and I/|b_R|²
        k, xd = ut.bessel_ic(l)
        k2 = k*k
        I = R**3/(2*l + 3) if k == 0 else -R*np.imag(xd)/np.imag(k2)
        return abs(k2)**2, R*(1 + np.real(xd)), I
    ME_ic = Ohm_ic = 0.0
    for c, l in enumerate(lp):
        L = l*(l + 1.); a0 = 4*np.pi/(2*l + 1); k4, S, I = parts(l)
        ME_ic  += a0*L*S*abs(fR[c])**2
        Ohm_ic += eta_i*2*a0*L*k4*I*abs(fR[c])**2
    for c, l in enumerate(lt):
        L = l*(l + 1.); a0 = 4*np.pi/(2*l + 1); k4, S, I = parts(l)
        ME_ic  += a0*L*I*abs(gR[c])**2
        Ohm_ic += eta_i*2*a0*L*S*abs(gR[c])**2
    return ME_ic/par.mu_i2o, Ohm_ic/par.mu_i2o



def mantle_budget(bman):
    '''
    NEW (2026-11-06, conductive_IC's mantle_budget): [ME_m, Ohm_m, ME_ext] of the resolved mantle (mantle = 'resolved') from bman,
    its part of one solution vector (N_mantle Chebyshev coefficients on [1, r_surface] per l, fm then gm), in the units of
    em_budget: ½ ∫ |𝐛|² and ∫ |∇×𝐛|²/sigma over the layer, and the energy of the potential field r^-(l+1) beyond r_surface. The
    integrands are those of a conducting inner core (em_budget).
    '''
    Nm = par.N_mantle; nb = int((par.lmax - par.m + 1)/2)
    R1, R2 = ut.rcmb, par.r_surface
    lpm, ltm = ut.ell(par.m, par.lmax, ut.bsymm)[:2]
    xg, wg = np.polynomial.legendre.leggauss(max(2*Nm, 64))
    rm = R1 + 0.5*(R2 - R1)*(xg + 1); wm = 0.5*(R2 - R1)*wg
    eta = 1/bg.sigma_mantle(rm)
    def mfun(a, nd):   # the radial function and its first nd r-derivatives on rm
        cc = np.array(a, dtype=complex); out = []
        for j in range(nd+1):
            out.append(ch.chebval(xg, cc)); cc = ch.chebder(cc)*2/(R2 - R1)
        return out
    ME_m = Ohm_m = ME_ext = 0.0
    for k, l in enumerate(lpm):
        L = l*(l+1.); a0 = 4*np.pi/(2*l+1)
        a = bman[k*Nm:(k+1)*Nm]
        f, f1, f2 = mfun(a, 2)
        ME_m   += np.sum( wm*a0*rm**2*( np.abs(L*f/rm)**2 + L*np.abs(f1 + f/rm)**2 ) )
        Ohm_m  += np.sum( wm*eta*2*a0*L*rm**2*np.abs(f2 + 2*f1/rm - L*f/rm**2)**2 )
        ME_ext += a0*L*l*R2*np.abs(np.sum(a))**2   # f(r_surface) is the sum of the coefficients
    for k, l in enumerate(ltm):
        L = l*(l+1.); a0 = 4*np.pi/(2*l+1)
        g, g1 = mfun(bman[nb*Nm + k*Nm: nb*Nm + (k+1)*Nm], 1)
        ME_m   += np.sum( wm*a0*rm**2*L*np.abs(g)**2 )
        Ohm_m  += np.sum( wm*eta*2*a0*L*( L*np.abs(g)**2 + np.abs(rm*g1 + g)**2 ) )
    return [ ME_m, Ohm_m, ME_ext ]



def deform_work(usol, bic=None, bman=None):
    '''
    NEW (2026-11-06): with forced_walls 'deforming' (background.deforms), the work of the motion of a thick conducting solid
    that deforms with its wall on the solid's field, 2 Re ∫ (𝐯×𝐁₀)⋅(∇×𝐛)* dV over the resolved mantle (bman, its part of the
    solution vector) and a conducting inner core (bic), in the units of em_budget (no MagnetD): 𝐯 = ∇×∇×(P(r_w) φ Y 𝐫), φ from
    background.deform_profile and P(r_w) from the flow usol (expand_reshape_sol) at the wall, 𝐁₀ continued inside the solid
    (background_field.solid_profiles), 𝐛 from the solid's Chebyshev coefficients. On a Gauss grid in r and cos θ, independent
    of the rows (assemble.deform_rows). The full budget (resid𝐄) adds it to the induction work (spin_doctor): with the solid's
    surface moving with the fluid, the tangential electric field is continuous at the wall and the fluxes of 𝐄×𝐛* through it
    cancel, so the field's energy changes by the work of the fluid's motion and of the solid's, less the Ohmic losses.
    '''
    m = par.m
    lp_u = ut.ell(m, par.lmax, par.symm)[0]
    lp_b, lt_b = ut.ell(m, par.lmax, ut.bsymm)[:2]
    nb = int((par.lmax - m + 1)/2)
    lmx = par.lmax + 1
    cross = lambda a, c: [ a[1]*c[2] - a[2]*c[1], a[2]*c[0] - a[0]*c[2], a[0]*c[1] - a[1]*c[0] ]
    total = 0.0
    for wall in ('inner', 'outer'):
        if not bg.deforms(wall):
            continue
        C = bic if wall == 'inner' else bman
        if C is None:
            return np.nan
        pol, tor = bf.solid_profiles(wall)
        nB = max([1] + list(pol) + list(tor))
        xg, wx = np.polynomial.legendre.leggauss(lmx + nB + 12)
        Yg, dYg, Zg = angular.ylm_grid(m, lmx, xg)
        Y0, dY0, _ = angular.ylm_grid(0, nB, xg)
        if wall == 'inner':   # the core's series in x = r/ricb on [-1, 1], one parity per section (as em_budget)
            rw, Nb, Nc = par.ricb, ut.Nic, par.N_cic
            xr, wr = np.polynomial.legendre.leggauss(max(2*Nc, par.lmax + 2*nB + 20, 64))
            rr, wrr = 0.5*rw*(xr + 1), 0.5*rw*wr
            if ut.symmB0 == -1: ixf, ixg = (m + ut.s)%2, (m + 1 - ut.s)%2
            else:               ixf, ixg = (m + 1 - ut.s)%2, (m + ut.s)%2
            def sfun(a, ix, nd):
                cc = np.zeros(Nc, complex); cc[ix::2] = a; out = []
                for j in range(nd + 1):
                    out.append(ch.chebval(rr/rw, cc)); cc = ch.chebder(cc)/rw
                return out
        else:   # the mantle's series on [1, r_surface], all the polynomials (as mantle_budget)
            rw, Nb, Rs = ut.rcmb, par.N_mantle, par.r_surface
            xr, wr = np.polynomial.legendre.leggauss(max(2*Nb, 64))
            rr, wrr = rw + 0.5*(Rs - rw)*(xr + 1), 0.5*(Rs - rw)*wr
            ixf = ixg = None
            def sfun(a, ix, nd):
                cc = np.array(a, dtype=complex); out = []
                for j in range(nd + 1):
                    out.append(ch.chebval(xr, cc)); cc = ch.chebder(cc)*2/(Rs - rw)
                return out
        def synth(qs):   # {l: (q, s, t)} on rr -> (F_r, F_θ, F_φ) on (rr, x), F = q Y 𝐫̂ + s r∇Y + t r∇Y×𝐫̂
            Fr = Ft = Fp = 0*np.outer(rr, xg)
            for l, (q, s, t) in qs.items():
                Fr = Fr + np.outer(q, Yg[l]); Ft = Ft + np.outer(s, dYg[l]) + np.outer(t, Zg[l])
                Fp = Fp + np.outer(s, Zg[l]) - np.outer(t, dYg[l])
            return [ Fr, Ft, Fp ]
        z = np.zeros_like(rr, dtype=complex)
        J = {}   # ∇×𝐛: from f, t = -(f'' + 2f'/r - L f/r²); from g, q = L g/r, s = g' + g/r
        for k, l in enumerate(lp_b):
            L = l*(l + 1.)
            f, f1, f2 = sfun(C[k*Nb:(k + 1)*Nb], ixf, 2)
            J[l] = (z, z, -(f2 + 2*f1/rr - L*f/rr**2))
        for k, l in enumerate(lt_b):
            L = l*(l + 1.)
            g, g1 = sfun(C[nb*Nb + k*Nb: nb*Nb + (k + 1)*Nb], ixg, 1)
            J[l] = (L*g/rr, g1 + g/rr, z)
        P0 = cheb2space_table(usol[0], 0, np.array([float(rw)]))[0][:, 0]   # P at the wall, each degree of lp_u
        V = {}   # 𝐯 = ∇×∇×(P(r_w) φ Y 𝐫): q = L P_s/r, s = P_s' + P_s/r
        for k, l in enumerate(lp_u):
            phi = bg.deform_profile(wall, l)
            Ps, Ps1 = P0[k]*phi(rr, 0), P0[k]*phi(rr, 1)
            V[l] = (l*(l + 1.)*Ps/rr, Ps1 + Ps/rr, z)
        B0 = [ 0*np.outer(rr, xg) ]*3   # 𝐁₀ inside the solid
        for n, f in pol.items():
            hh = [ f(rr, d) for d in (0, 1) ]
            B0 = [ B0[0] + np.outer(n*(n + 1.)*hh[0]/rr, Y0[n]), B0[1] + np.outer(hh[1] + hh[0]/rr, dY0[n]), B0[2] ]
        for n, f in tor.items():
            B0 = [ B0[0], B0[1], B0[2] - np.outer(f(rr, 0), dY0[n]) ]
        W, j = cross(synth(V), B0), synth(J)
        dens = sum(w*np.conj(c) for w, c in zip(W, j))
        total += 2*np.real(2*np.pi*np.sum((wrr*rr**2)[:, None]*wx[None, :]*dens))
    return total



def twa_budget(bsol, wall, bvals, usol=None):
    '''
    NEW (2026-11-06, the thin-wall part of conductive_IC's em_budget): [ ME, E_w, Ohm_w ] of the thin wall 'outer' or 'inner':
    the energy of the potential field beyond it (ME_ext beyond r + h, or ME_ic below r - h), and the wall's E_w and Ohm_w
    (em_budget). bvals(C, nd, x): the values and r-derivatives of the rows of C at x = 1 (outer) or -1 (inner).
    '''
    [ lp, lt, _ ] = ut.ell(par.m, par.lmax, ut.bsymm)
    f0p = 4*np.pi/(2*lp+1); f0t = 4*np.pi/(2*lt+1)
    Lp = lp*(lp+1.); Lt = lt*(lt+1.)
    if wall == 'outer':
        r, c, c1, p, x, sl = ut.rcmb, par.c_cmb, par.c1_cmb, par.muw_cmb, 1.0, 1
    else:
        r, c, c1, p, x, sl = par.ricb, par.c_icb, par.c1_icb, par.muw_icb, -1.0, -1
    f, f1, f2 = bvals(bsol[0], 2, x); g = bvals(bsol[1], 0, x)[0]
    F, F1 = r*f, f + r*f1
    DF = 2*f1 + r*f2 - Lp*f/r            # r ∇²_l f = F'' - L F/r²
    if usol is not None and ut.twa_slips(wall):
        DF = DF + slip_lapF(usol, wall)
    if usol is not None and c1 != 0 and bg.flows_through(wall):   # NEW (2026-11-06): a radial boundary flow through the wall
        DF = DF + radial_lapF(usol, wall)
    elif usol is not None and bg.thin_drop(wall):   # or the layer deforming across the drop of the toroidal B0: half of it
        DF = DF + 0.5*radial_lapF(usol, wall, tor_only=True)
    E_w = Ohm_w = 0.0
    hw = 0.0                               # the wall's thickness
    if np.isinf(p):   # the field beyond the wall from the wall condition (outer: F' + l F/r + c1 r D F = 0 there)
        Fx = -sl*(r*F1 + sl*c1*r**2*DF)/(lp if wall == 'outer' else lp + 1)
    else:             # at r ± h, F ± h F' with h F' = c r F' (conductive_IC FIX 2026-10-21)
        hw = c*r/p
        Fx = F + sl*c*r*F1
        E_w += np.sum( (hw/p)*f0p*Lp**2*np.abs(f)**2 )   # the radial field in the wall
    E_w += np.sum( c*r**3*f0p*Lp*np.abs(F1/r)**2 )        # the tangential poloidal field, the c term
    if c1 > 0:
        Ohm_w += np.sum( 2*f0p*r*Lp*c1*np.abs(DF)**2 ) + np.sum( 2*f0t*r*Lt*np.abs(g)**2/c1 )
    if wall == 'outer':
        ME = np.sum( f0p*Lp*lp*np.abs(Fx/r)**2 ) * r/(r + hw)          # potential field r^-(l+1) beyond r + h
    else:
        ME = np.sum( f0p*Lp*(lp+1)*r*np.abs(Fx/r)**2 ) * r/(r - hw)    # potential field r^l below r - h
    return [ ME, E_w, Ohm_w ]



def slip_lapF(usol, wall):
    '''
    NEW (2026-11-06, conductive_IC's slip_lapF, here for any poloidal B0): for each poloidal degree l of b, r²/(L MagnetD) times
    [r̂⋅∇×(u×B0)]_l at a thin wall along which the fluid slips ('outer' or 'inner'), the radial induction of the slip that the
    poloidal wall row adds to D F (assemble.bc_b_slip). With u_r = 0, r̂⋅∇×(u×B0) = -(B0_r ∇_H⋅u_t + ∂θB0_r u_θ)/r, computed on
    a Gauss grid in cos θ and projected on Y_l: a check of the angular algebra of the assembly. usol from expand_reshape_sol.
    '''
    lpu, ltu, _ = ut.ell(par.m, par.lmax, par.symm)
    lpb = ut.ell(par.m, par.lmax, ut.bsymm)[0]
    r = ut.rcmb if wall == 'outer' else par.ricb
    prof = bf.pol_profiles()
    nB = max(prof)
    x, w = np.polynomial.legendre.leggauss(par.lmax + nB + 9)
    Y, dY, Z = angular.ylm_grid(par.m, par.lmax + 1, x)
    Y0, dY0, _ = angular.ylm_grid(0, nB, x)
    rr = np.array([r], dtype=float)
    Pt, Tt = cheb2space_table(usol[0], 1, rr), cheb2space_table(usol[1], 0, rr)
    lh = rap.logrhoX(rr, 1)[0]
    s_ = Pt[1][:, 0] + Pt[0][:, 0]/r + lh*Pt[0][:, 0]      # u_t = s r∇Y + t r∇Y×r̂ of each degree
    t_ = Tt[0][:, 0]
    uth  = s_ @ dY[lpu] + t_ @ Z[ltu]                        # u_θ
    divu = (-lpu*(lpu+1)*s_) @ Y[lpu]                       # r ∇_H⋅u_t
    Br = sum( n*(n+1)*f(rr, 0)[0]/r*Y0[n] for n, f in prof.items() )
    dBr = sum( n*(n+1)*f(rr, 0)[0]/r*dY0[n] for n, f in prof.items() )
    X = -( Br*divu + dBr*uth )/r
    Xl = np.array([ (2*l+1)/2*np.sum(w*X*Y[l]) for l in lpb ])
    return r**2/(lpb*(lpb+1)*par.MagnetD)*Xl



def radial_lapF(usol, wall, tor_only=False):
    '''
    NEW (2026-11-06): as slip_lapF, for the radial boundary flow of a forced problem (background.flows_through): for each
    poloidal degree l of b, r²/(L MagnetD) [r̂⋅∇×(uᵣ r̂×B0)]_l at the wall, the radial induction of the flow that the poloidal
    row of a thin conducting layer at rest adds to D F (assemble.bc_b_radial). r̂⋅∇×(uᵣ r̂×B0) = ∇_H⋅(uᵣ B0_t) =
    uᵣ ∇_H⋅B0_t + B0_t⋅∇_H uᵣ, ∇_H⋅B0_t = -(r² B0_r)'/r² (∇⋅B0 = 0), for a poloidal and a toroidal B0, computed on a Gauss
    grid in cos θ and projected on Y_l: a check of the angular algebra of the assembly. usol from expand_reshape_sol.
    NEW (2026-11-06): tor_only = True, from the toroidal B0 alone (a thin layer that deforms across its drop, thin_drop).
    '''
    lpu = ut.ell(par.m, par.lmax, par.symm)[0]
    lpb = ut.ell(par.m, par.lmax, ut.bsymm)[0]
    r = ut.rcmb if wall == 'outer' else par.ricb
    pol, tor = ({} if tor_only else bf.pol_profiles()), bf.tor_profiles()
    nB = max([1] + list(pol) + list(tor))
    x, w = np.polynomial.legendre.leggauss(par.lmax + nB + 9)
    Y, dY, Z = angular.ylm_grid(par.m, par.lmax + 1, x)
    Y0, dY0, _ = angular.ylm_grid(0, nB, x)
    rr = np.array([r], dtype=float)
    P0 = cheb2space_table(usol[0], 0, rr)[0][:, 0]
    ur_k = lpu*(lpu + 1)*P0/r                              # uᵣ = L P/r of each degree
    ur, dur, zur = ur_k @ Y[lpu], ur_k @ dY[lpu], ur_k @ Z[lpu]   # uᵣ, ∂θuᵣ, i m uᵣ/sinθ
    divB = sum( -n*(n+1)*(f(rr, 0)[0] + r*f(rr, 1)[0])/r**2*Y0[n] for n, f in pol.items() )   # ∇_H⋅B0_t
    Bth  = sum( (f(rr, 1)[0] + f(rr, 0)[0]/r)*dY0[n] for n, f in pol.items() )             # B0_θ
    Bph  = sum( -f(rr, 0)[0]*dY0[n] for n, f in tor.items() )                              # B0_φ
    X = ur*divB + (Bth*dur + Bph*zur)/r
    Xl = np.array([ (2*l+1)/2*np.sum(w*X*Y[l]) for l in lpb ])
    return r**2/(lpb*(lpb+1)*par.MagnetD)*Xl



def twa_error(lam):
    '''
    NEW (2026-11-06, conductive_IC's twa_error): how far a solution of eigenvalue lam (in the run's time unit) is from the thin
    wall limit: [h/δ_w, ε_TWA] at the outer and at the inner wall, 0 where there is no thin wall (spin_doctor writes ε_TWA to
    magnetic.dat). Inside the wall b'' = k² b with k² = lam/η_w, and the thin-wall conditions keep the linear profile across
    it: the first term left out, the wall's own induction, is ε_TWA = |k h|²/3 = (2/3)(h/δ_w)² of the wall's part, δ_w =
    sqrt(2 η_w/|lam|) the wall's skin depth. As c c1 = h² mu_w sigma_w/(r² mu_f sigma_f) = h² η_f/(r² η_w), η_f = MagnetD,
        (h/δ_w)² = c c1 r² |lam|/(2 MagnetD)     (r = 1 or ricb), for any muw,
    with lam in the frame of the wall: lam + i m Gaspard ΔΩ(r) when the wall turns with a differential rotation (NEW 2026-11-08:
    and + i m times the rate of a background flow there, bgflow.rigid_rate).
    The weaker conditions h/r << 1 and l h/r << 1 are not checked.
    '''
    out = []
    for on, c, c1, r in ((ut.twa_o, par.c_cmb, par.c1_cmb, ut.rcmb), (ut.twa_i, par.c_icb, par.c1_icb, par.ricb)):
        if on and par.MagnetD != 0:
            lw = lam + (1j*par.m*par.Gaspard*dr.rigid_at(r)[0] if par.diff_rot else 0)   # in the wall's frame
            if par.bgflow:   # NEW (2026-11-08)
                lw = lw + 1j*par.m*bgf.rigid_rate(r)[0]
            hd2 = c*c1*r**2*abs(lw)/(2*par.MagnetD)
            out += [np.sqrt(hd2), 2*hd2/3]
        else:
            out += [0., 0.]
    return np.array(out)



# ----------------------------------------------------------------------------------------------------------------------
# NEW (2026-11-05, phase 1b): the metric parts of the magnetic budgets with centrifugal = 1 (centrifugal.py, sections 4, 5)
# ----------------------------------------------------------------------------------------------------------------------
def _lcomp_metric(V, V1, amps, rr, m, lw):
    '''
    (M - I) V and its r-derivative as l-components on the degrees 0..lw (dicts q, s, t of (lw + 1, nr) arrays), from those of
    V and V' at the radii rr, with the amplitudes amps = (ε, η, r η') there and the primitives of centrifugal.py (section 2):
    d/dr [ε Ce V + η Ch V] = (η/r) Ce V + ε Ce V' + (r η'/r) Ch V + η Ch V'
    NEW (2026-11-06, step 2): the exact spheroid, (Q - I) V and (Q - I) V' with the constant couplings of
    centrifugal.spheroid_primitives (amps not used)
    '''
    if cf.exact():
        C = cf.spheroid_primitives('inertia', m, lw)[0]
        app = lambda U: {k: C[k]['q'] @ U['q'] + C[k]['a'] @ U['s'] + C[k]['T'] @ U['t'] for k in 'qst'}
        return app(V), app(V1)
    e, h, c = amps
    Ce, Ch = cf.primitives('inertia', 'e', m, lw)[0], cf.primitives('inertia', 'h', m, lw)[0]
    app = lambda C, U: {k: C[k]['q'] @ U['q'] + C[k]['a'] @ U['s'] + C[k]['T'] @ U['t'] for k in 'qst'}
    eV, hV, eV1, hV1 = app(Ce, V), app(Ch, V), app(Ce, V1), app(Ch, V1)
    return ({k: e*eV[k] + h*hV[k] for k in 'qst'},
            {k: (h/rr)*eV[k] + e*eV1[k] + (c/rr)*hV[k] + h*hV1[k] for k in 'qst'})

def _lcomp_curl(V, V1, rr):
    '''∇× on l-components (as curl): (q, s, t) -> (L t/r, t' + t/r, (q - s)/r - s')'''
    lv = np.arange(V['q'].shape[0]); L = (lv*(lv + 1.))[:, None]
    return {'q': L*V['t']/rr, 's': V1['t'] + V['t']/rr, 't': (V['q'] - V['s'])/rr - V1['s']}

def _lcomp_dot(U, V, rr, w):
    '''∫ U*⋅V dV over the radii rr with the quadrature weights w (cdot_pol + cdot_tor summed over l)'''
    lv = np.arange(U['q'].shape[0]); nl = (4*np.pi/(2*lv + 1))[:, None]; L = (lv*(lv + 1.))[:, None]
    g = nl*(np.conj(U['q'])*V['q'] + L*(np.conj(U['s'])*V['s'] + np.conj(U['t'])*V['t']))
    return np.sum(np.sum(g, 0)*rr**2*w)

def _lcomp_field(bsol, rr, lw):
    '''b̃ and b̃' as l-components on the degrees 0..lw at the radii rr, from the solution bsol (f, g)'''
    lp, lt, _ = ut.ell(par.m, par.lmax, ut.bsymm)
    z = lambda: {k: np.zeros((lw + 1, len(rr)), dtype=complex) for k in 'qst'}
    B, B1 = z(), z()
    Ft, Gt = cheb2space_table(bsol[0], 2, rr), cheb2space_table(bsol[1], 1, rr)
    for k, l in enumerate(lp):
        L = l*(l + 1.); F0, F1, F2 = Ft[0, k], Ft[1, k], Ft[2, k]
        B['q'][l], B['s'][l] = L*F0/rr, F1 + F0/rr
        B1['q'][l], B1['s'][l] = L*(F1/rr - F0/rr**2), F2 + F1/rr - F0/rr**2
    for k, l in enumerate(lt):
        B['t'][l], B1['t'][l] = Gt[0, k], Gt[1, k]
    return B, B1

def centrifugal_magnetic_energy(bsol, Ra, Rb):
    '''
    ∫ b̃*⋅(M - I) b̃ dV from Ra to Rb, the metric part of the field's energy (ME + this is ∫ |b|² dV of the body), on the grid of
    setup_grid (solve.normalise_energy, after magnetic_energy)
    '''
    lw = cf.lw_of()
    prof = cf.profile_functions()
    B, B1 = _lcomp_field(bsol, rk, lw)
    Mb, _ = _lcomp_metric(B, B1, [prof[c](rk, 0) for c in ('ce0', 'ce1', 'ce2')], rk, par.m, lw)
    return np.real(_lcomp_dot(B, Mb, rk, wk*(Rb - Ra)/2))

def centrifugal_magnetic_terms(bsol, bic=None):
    '''
    With centrifugal = 1 and magnetic = 1, the metric parts of the magnetic budgets (centrifugal.py, sections 4 and 5), in the
    conventions of magnetic_worker and em_budget (no Hendrik or MagnetD), on the quadrature grid of diagnose (call it after
    diagnose and diagnose_magnetic, before wall_powers moves the grid). b̃ is the field of the reference space (the flux
    J F⁻¹ b), w the flow, j0 = ∇×b̃, jc = ∇×((M - I) b̃), B0, J0 = ∇×B0 and J0c = ∇×((M - I) B0), all of the reference space;
    δ𝐆 and δ𝐄 are the first-order metric parts of the Lorentz force and of the induction's field, as the operators have them.
    Returns a dict:
        Wlor   : 2 ∫ w*⋅δ𝐆 dV (complex), added to the Lorentz power of residσ and residω
        Indu   : 2 Re ∫ b̃*⋅∇×δ𝐄 dV, added to the induction work of resid𝐛 (the volume part 2 Re ∫ δ𝐄⋅j0* dV plus Iwall)
        Iwall  : its wall terms 2 Re ∮ (δ𝐄×b̃*)⋅r̂ dS (zero at no-slip walls)
        MEc    : ∫ b̃*⋅(M - I) b̃ dV, the metric part of the field's energy: ME + MEc = ∫ |b|² dV in the deformed fluid
        InduM  : 2 Re ∫ jc*⋅𝐄 dV, 𝐄 = w×B0 + δ𝐄 + U0×b̃ the induction's field: the work on the metric part of the field
        Ohmc   : 2 Re ∫ jc*⋅j0 dV (times MagnetD: the metric part of the Ohmic loss with the spherical diffusion)
        MEext  : the energy of the potential field outside the deformed body (in place of em_budget's ME_ext)
        MEic   : that of the field inside the inner core: the potential field of an insulating one, ∫ b̃*⋅M b̃ dV of a conducting
                 one (in place of em_budget's ME_ic)
        Ohm_ic : a conducting inner core's η_i 2 Re ∫ (j0 + jc)*⋅j0 dV (in place of em_budget's Ohm_ic, without MagnetD)
                 and, NEW (2026-11-06, phase 2), with its metric diffusion η_i 2 Re ∫ j0*⋅[jc + (M - I) j0] dV
        Mdfs   : NEW (2026-11-06, phase 2) 2 Re ∫ b̃*⋅(-∇×δ𝐄_d) dV, the work of the metric part δ𝐄_d = η [(M - I) j0 + jc] of
                 the diffusion's field (without MagnetD), added to Mdfs in resid𝐛; Ohmc then also has 2 Re ∫ j0*⋅δ𝐄_d dV/MagnetD
    With them the budgets of the deformed body are: resid𝐛 with Indu, as in the sphere (the B rows of the field are those of
    the sphere: ∂ₜb̃ = -∇×(Fᵀ𝐄) exactly); residσ, residω with Wlor; resid𝐄 with the energy ∫ b̃*⋅M b̃ dV (ME + MEc), whose rate
    is the work 2 Re ∫ 𝐄⋅(∇×M b̃)* dV (Indu - Iwall + InduM) minus the Ohmic loss (Ohm_f + MagnetD Ohmc), the energy of the
    potential fields of the deformed exterior and inner core, and that of a conducting inner core with its metric part.
    resid𝐛, residσ and residω close to round-off; resid𝐄 to the second order in the flattening (the operators and the wall
    rows are of first order).
    '''
    m = par.m
    Ra, Rb = par.ricb, ut.rcmb
    lw = cf.lw_of()
    prof = cf.profile_functions()
    amp = lambda rr: [prof[c](rr, 0) for c in ('ce0', 'ce1', 'ce2')]
    lp_u, lt_u, _ = ut.ell(m, par.lmax, par.symm)
    nB = max([3] + list(bf.pol_profiles()) + list(bf.tor_profiles()))
    lw0 = nB + 4                                       # the degrees of B0 and of (M - I) B0, m = 0
    D = max(nB, dr.n_max() + 1 if par.diff_rot else 0)
    xg, wx = np.polynomial.legendre.leggauss(lw + 16 + max(0, D - 22))
    Ym, Y0 = angular.ylm_grid(m, lw, xg), angular.ylm_grid(0, lw0, xg)
    P2x = 1.5*xg**2 - 0.5

    def flow(rr):   # w as l-components (with its (ln ρ0)' part, as velocity) at the radii rr
        W = {k: np.zeros((lw + 1, len(rr)), dtype=complex) for k in 'qst'}
        lh = rap.logrhoX(rr, 1)
        Pt, Tt = cheb2space_table(usol2[0], 1, rr), cheb2space_table(usol2[1], 0, rr)
        for k, l in enumerate(lp_u):
            W['q'][l], W['s'][l] = l*(l + 1.)*Pt[0, k]/rr, Pt[1, k] + Pt[0, k]/rr + lh*Pt[0, k]
        for k, l in enumerate(lt_u):
            W['t'][l] = Tt[0, k]
        return W
    def background(rr):   # B0 and B0' as l-components on the degrees 0..lw0 at m = 0 (as magnetic_powers_grid)
        V = {k: np.zeros((lw0 + 1, len(rr))) for k in 'qst'}; V1 = {k: np.zeros((lw0 + 1, len(rr))) for k in 'qst'}
        for n, f in bf.pol_profiles().items():
            L0 = n*(n + 1.); h0, h1, h2 = [ f(rr, d) for d in (0, 1, 2) ]
            V['q'][n] += L0*h0/rr; V['s'][n] += h1 + h0/rr
            V1['q'][n] += L0*(h1/rr - h0/rr**2); V1['s'][n] += h2 + h1/rr - h0/rr**2
        for n, f in bf.tor_profiles().items():
            k0, k1 = [ f(rr, d) for d in (0, 1) ]
            V['t'][n] += k0; V1['t'][n] += k1
        return V, V1
    def synth(V, Yt):   # l-components -> (F_r, F_θ, F_φ) on (r, x), F = q Y r̂ + s r∇Y + t r∇Y×r̂
        Y, dY, Z = Yt
        return [ V['q'].T @ Y, V['s'].T @ dY + V['t'].T @ Z, V['s'].T @ Z - V['t'].T @ dY ]
    def cross(a, c):
        return [ a[1]*c[2] - a[2]*c[1], a[2]*c[0] - a[0]*c[2], a[0]*c[1] - a[1]*c[0] ]
    def vol(f):   # ∫ f dV over the fluid, f on (rk, x), the e^{imφ} factors already combined (2π)
        return 2*np.pi*np.sum((wk*rk**2)[:, None]*wx[None, :]*f)*(Rb - Ra)/2
    def rotation(rr):   # U0 = Gaspard ΔΩ r sinθ e_φ (as magnetic_powers_grid)
        z0 = 0*np.outer(rr, xg)
        if not par.diff_rot:
            return [ z0, z0, z0 ]
        dO = sum(np.outer(dr.fX(n, rr, 0), np.polynomial.legendre.Legendre.basis(n)(xg)) for n in dr.degrees())
        return [ z0, z0, par.Gaspard*dO*np.outer(rr, np.sqrt(1 - xg**2)) ]

    out = {}
    # the fluid: l-components, their metric parts and curls, then the products on the (r, x) grid
    a_k = amp(rk)
    B, B1 = _lcomp_field(bsol, rk, lw)
    Mb, Mb1 = _lcomp_metric(B, B1, a_k, rk, m, lw)
    j0, jc = _lcomp_curl(B, B1, rk), _lcomp_curl(Mb, Mb1, rk)
    V0, V01 = background(rk)
    M0, M01 = _lcomp_metric(V0, V01, a_k, rk, 0, lw0)
    J0, J0c = _lcomp_curl(V0, V01, rk), _lcomp_curl(M0, M01, rk)
    w, b, jg, jcg = [ synth(V, Ym) for V in (flow(rk), B, j0, jc) ]
    B0g, J0g, J0cg = [ synth(V, Y0) for V in (V0, J0, J0c) ]
    hP2 = a_k[1][:, None]*P2x[None, :] if not cf.exact() else 0*P2x[None, :]   # J - 1 (NEW 2026-11-06, step 2: 0 exactly)
    G = [ -hP2*(p + q) + s + t for p, q, s, t in zip(cross(J0g, b), cross(jg, B0g), cross(J0cg, b), cross(jcg, B0g)) ]
    out['Wlor'] = 2*vol(sum(np.conj(u)*g for u, g in zip(w, G)))
    dE = [ -hP2*v for v in cross(w, B0g) ]
    E = [ p + q + s for p, q, s in zip(cross(w, B0g), dE, cross(rotation(rk), b)) ]
    out['InduM'] = 2*np.real(vol(sum(e*np.conj(j) for e, j in zip(E, jcg))))
    wq = wk*(Rb - Ra)/2
    out['MEc'] = np.real(_lcomp_dot(B, Mb, rk, wq))
    out['Ohmc'] = 2*np.real(_lcomp_dot(jc, j0, rk, wq))
    # NEW (2026-11-06, phase 2): the metric part of the diffusion's field, δ𝐄 = η [(M - I) j0 + jc] = -e (e as
    # centrifugal.magnetic_diffusion_metric, per unit MagnetD with the profile etaX): its work on the fluid's field,
    # Mdfs = 2 Re ∫ b̃*⋅∇×e dV (resid𝐛), and the Ohmic loss of the deformed fluid, 2 Re ∫ 𝐄_diff⋅(∇×M b̃)* dV, which is
    # Ohm_f + MagnetD [2 Re ∫ jc*⋅j0 dV + 2 Re ∫ j0*⋅(-e) dV] at first order (the second term is new: Ohmc)
    out['Mdfs'] = 0.0
    if par.MagnetD > 0:
        e, ce = _cf_expressions('mag', lw)
        lpb, ltb, _ = ut.ell(m, par.lmax, ut.bsymm)
        nd = _cf_order(e, ce)
        tabs = {'F': _cf_table(bsol[0], lpb, nd, rk, lw), 'G': _cf_table(bsol[1], ltb, nd, rk, lw)}
        ampk = {a: prof[c](rk, 0) for a, (c, _) in cf.AMPS.items()}
        etaj = lambda j: rap.etaX(rk, j)
        Ek, CEk = _cf_eval(e, tabs, rk, ampk, etaj), _cf_eval(ce, tabs, rk, ampk, etaj)
        out['Mdfs'] = 2*np.real(_lcomp_dot(B, CEk, rk, wq))
        if cf.exact():   # NEW (2026-11-06, step 2): the exact spheroid's loss 2 ∫ j*⋅Q j dV, j = ∇×(Q b̃) = j0 + jc
            jt = {k: j0[k] + jc[k] for k in 'qst'}
            out['Ohmc'] = 2*np.real(_lcomp_dot(jt, jt, rk, wq) + _lcomp_dot(jt, _lcomp_metric(jt, jt, None, rk, m, lw)[0], rk, wq)
                                    - _lcomp_dot(j0, j0, rk, wq))
        else:
            out['Ohmc'] += 2*np.real(_lcomp_dot(j0, {k: -v for k, v in Ek.items()}, rk, wq))
    # the induction work of δ𝐄: its volume part and the wall terms 2 Re ∮ (δ𝐄×b̃*)⋅r̂ dS (w = 0 at no-slip walls)
    rw = np.array([Ra if Ra > 0 else 0.5, Rb])   # (a dummy inner radius in a full sphere, its term is not used)
    Bw = _lcomp_field(bsol, rw, lw)[0]
    ww, bw, B0w = synth(flow(rw), Ym), synth(Bw, Ym), synth(background(rw)[0], Y0)
    hw = amp(rw)[1][:, None]*P2x[None, :] if not cf.exact() else 0*P2x[None, :]   # (NEW 2026-11-06, step 2: J = 1)
    dEw = [ -hw*v for v in cross(ww, B0w) ]
    walls = 2*np.pi*np.sum(wx[None, :]*(dEw[1]*np.conj(bw[2]) - dEw[2]*np.conj(bw[1]))*(rw**2)[:, None], axis=1)
    out['Iwall'] = 2*np.real(walls[1] - (walls[0] if Ra > 0 else 0))
    out['Indu'] = 2*np.real(vol(sum(e*np.conj(j) for e, j in zip(dE, jg)))) + out['Iwall']

    # the potential fields of the insulating exterior (and inner core): the coefficients A_l of ∇φ at the wall
    # (centrifugal.wall_potential), the energy of the sphere's potential field (Σ (l + 1) |A_l|² R³ outside, Σ l |A_l|² R³
    # inside, kore's ∫ |Y|² dΩ) and the shell between the sphere and the wall, ∮ |∇φ|² ξ_r dA with ξ_r = ε R P2
    lv = np.arange(lw + 1); nl = 4*np.pi/(2*lv + 1)
    def potential_energy(wall, k):
        R = rw[k]
        if cf.exact():   # NEW (2026-11-06, step 2): the exact spheroid, ∓R³ Σ λ_l |A_l|² ∫|Y_l|² dΩ (centrifugal.spheroid_potential)
            A, lam = cf.spheroid_potential(wall, m, lw, R, Bw['q'][:, k], Bw['s'][:, k], Bw['t'][:, k])
            return (-1. if wall == 'outer' else 1.)*np.sum(nl*lam*np.abs(A)**2)*R**3
        e0, h0 = [ float(a[k]) for a in amp(rw)[:2] ]
        A = cf.wall_potential(wall, m, lw, e0, h0, R, Bw['q'][:, k], Bw['s'][:, k], Bw['t'][:, k])
        kq = -(lv + 1.) if wall == 'outer' else lv*1.
        sph = np.sum(nl*np.abs(kq)*np.abs(A)**2)*R**3
        g = synth({'q': (kq*A)[:, None], 's': A[:, None], 't': 0*A[:, None]}, Ym)
        shell = e0*R**3*2*np.pi*np.sum(wx*P2x*sum(np.abs(c[0])**2 for c in g))
        return sph - shell if wall == 'outer' else sph + shell
    out['MEext'] = potential_energy('outer', 1)
    out['MEic'] = out['Ohm_ic'] = 0.0
    if Ra == 0:
        pass
    elif par.innercore == 'insulator':
        out['MEic'] = potential_energy('inner', 0)
    elif bic is not None:
        # a conducting inner core (mu_i2o = 1): its field as em_budget reads it, ∫ b̃*⋅M b̃ dV and η_i 2 Re ∫ (j0 + jc)*⋅j0 dV,
        # with ε, η inside it from centrifugal.amplitudes
        Nic = ut.Nic; nb = int((par.lmax - par.m + 1)/2)
        if ut.symmB0 == -1: ixf, ixg = (par.m + ut.s)%2, (par.m + 1 - ut.s)%2
        else:               ixf, ixg = (par.m + 1 - ut.s)%2, (par.m + ut.s)%2
        xi, wi = np.polynomial.legendre.leggauss(max(2*par.N_cic, 64))
        ri = 0.5*Ra*(xi + 1); wi = 0.5*Ra*wi
        def icfun(a, ix, nd):
            cc = np.zeros(par.N_cic, complex); cc[ix::2] = a; res = []
            for j in range(nd+1):
                res.append(ch.chebval(ri/Ra, cc)); cc = ch.chebder(cc)/Ra
            return res
        lp, lt, _ = ut.ell(m, par.lmax, ut.bsymm)
        Bi = {k: np.zeros((lw + 1, len(ri)), dtype=complex) for k in 'qst'}
        Bi1 = {k: np.zeros((lw + 1, len(ri)), dtype=complex) for k in 'qst'}
        for k, l in enumerate(lp):
            L = l*(l + 1.); f, f1, f2 = icfun(bic[k*Nic:(k+1)*Nic], ixf, 2)
            Bi['q'][l], Bi['s'][l] = L*f/ri, f1 + f/ri
            Bi1['q'][l], Bi1['s'][l] = L*(f1/ri - f/ri**2), f2 + f1/ri - f/ri**2
        for k, l in enumerate(lt):
            g, g1 = icfun(bic[nb*Nic + k*Nic: nb*Nic + (k+1)*Nic], ixg, 1)
            Bi['t'][l], Bi1['t'][l] = g, g1
        Mi, Mi1 = _lcomp_metric(Bi, Bi1, cf.amplitudes(ri), ri, m, lw)
        j0i = _lcomp_curl(Bi, Bi1, ri)
        out['MEic'] = np.real(_lcomp_dot(Bi, Bi, ri, wi) + _lcomp_dot(Bi, Mi, ri, wi))
        ohm = _lcomp_dot(j0i, j0i, ri, wi) + _lcomp_dot(_lcomp_curl(Mi, Mi1, ri), j0i, ri, wi)
        if cf.exact():   # NEW (2026-11-06, step 2): the exact spheroid's ∫ j*⋅Q j dV, j = ∇×(Q b̃)
            ji = {k: j0i[k] + v for k, v in _lcomp_curl(Mi, Mi1, ri).items()}
            ohm = _lcomp_dot(ji, ji, ri, wi) + _lcomp_dot(ji, _lcomp_metric(ji, ji, None, ri, m, lw)[0], ri, wi)
        elif cf.ic_metric_on():
            # NEW (2026-11-06, phase 2): the core's metric diffusion (uniform η_i), the loss 2 Re ∫ j0*⋅(-e) dV as in the fluid
            e = _cf_expressions('mag', lw)[0]
            nd = _cf_order(e)
            tabs = {X: [np.zeros((lw + 1, len(ri)), dtype=complex) for d in range(nd + 1)] for X in 'FG'}
            for k, l in enumerate(lp):
                for d, v in enumerate(icfun(bic[k*Nic:(k+1)*Nic], ixf, nd)):
                    tabs['F'][d][l] = v
            for k, l in enumerate(lt):
                for d, v in enumerate(icfun(bic[nb*Nic + k*Nic: nb*Nic + (k+1)*Nic], ixg, nd)):
                    tabs['G'][d][l] = v
            ampi = {a: f(ri) for a, f in cf.amplitude_series().items()}
            Ei = _cf_eval(e, tabs, ri, ampi, lambda j: np.ones_like(ri) if j == 0 else np.zeros_like(ri))
            ohm += _lcomp_dot(j0i, {k: -v for k, v in Ei.items()}, ri, wi)
        out['Ohm_ic'] = 2*np.real(ohm)/(par.mu_i2o*par.sigma_i2o)
    else:
        out['MEic'] = out['Ohm_ic'] = np.nan
    return out



def diagnose_magnetic( bsol ):
    '''
    NEW (2026-10-28): magnetic_worker for every l, as an array (one row per l). Call it right after diagnose (same grid
    and flow solution).
    '''
    global INDU_WALLS, INDU_ROT_SOLID
    INDU_WALLS = 0.0   # the spectral formulas: no-slip walls, where the wall terms vanish
    INDU_ROT_SOLID = 0.0
    set_bsolution(bsol)
    ll = ut.ell(par.m, par.lmax, ut.bsymm)[2]
    out = np.array([ magnetic_worker(l) for l in ll ])
    if use_grid():   # NEW (2026-10-28): toroidal B0 (or FORCE_GRID): the induction and Lorentz terms from the (r, θ) grid
        indu, plor = magnetic_powers_grid()
        out[0, 2] += indu; out[0, 3] += plor.real; out[0, 4] += plor.imag
    elif slipping_walls() or ut.rotdyn or walls_move():   # NEW (2026-11-06): the spectral Indu includes the wall terms;
        magnetic_powers_grid(walls_only=True)   # the full budget needs them apart (rotdyn = 1: the no-slip walls move with the
                                                # solids; forced problems and bco, bci = 2: the walls that move, walls_move)
    return out



def slipping_walls():
    '''
    NEW (2026-11-06): whether the fluid slips along a wall (bco = 0, or bci = 0 with an inner core, or ViscosD = 0), where u×B0
    carries magnetic energy through the wall (INDU_WALLS). With the flow (hydro = 1) only.
    '''
    return bool(par.hydro and (par.ViscosD == 0 or par.bco == 0 or (par.ricb > 0 and par.bci == 0)))



# NEW (2026-10-30): the wall terms 2 Re ∮ (𝐄×𝐛*)⋅r̂ dS included in the last Indu of magnetic_powers_grid. The fluid budget
# (resid𝐛) needs them; the full budget (resid𝐄) needs the volume work ∫ 𝐄⋅(∇×𝐛)* alone, because the flux through a wall
# leaves the fluid's magnetic energy for the energy outside (an insulating mantle or core) or inside (spin_doctor.py). They
# vanish for no-slip walls without a differential rotation; U0×b does not vanish where ΔΩ is not zero at a wall.
# NEW (2026-11-06): nor does u×B0 at the no-slip walls of rotating solids (rotdyn = 1). Their flux is the induction work of
# the solid's rotation in the field (Re ∫ (𝐕×𝐁₀)⋅(∇×𝐛)* dV over it, 𝐕 = ω ẑ×𝐫), which the solid's rotational energy pays
# for: it is minus the power of the magnetic torque on the solid, so resid𝐄 takes it out of the induction work, as here.
INDU_WALLS = 0.0
# NEW (2026-11-06): the part of INDU_WALLS from U0×b at the walls where a conducting solid turns with ΔΩ (a conducting inner
# core, a thin conducting wall: ΔΩ is uniform in latitude there, background.magnetic_diffrot). That flux does not leave for
# the energy beyond the wall: it is the induction work of the turning solid, Re ∫ (𝐕×𝐛)⋅(∇×𝐛)* dV over it, which equals the
# flux because Re ∫ 𝐛*⋅∇×(𝐕×𝐛) dV = 0 for a rigid rotation 𝐕 and the potential field beyond carries none of it. The full
# budget (resid𝐄) keeps it in the induction work (spin_doctor: Iwall = INDU_WALLS - INDU_ROT_SOLID). It was missing for a
# conducting inner core turning with ΔΩ(ricb) ≠ 0 (resid𝐄 ~1e-4 for eigenvalues that are right).
INDU_ROT_SOLID = 0.0

# NEW (2026-10-28): True (or the environment variable KORE_MAG_GRID=1): magnetic_powers_grid also for a poloidal B0 alone, a
# cross-check of the spectral formulas lorentz4pp, induction4pp (tests/magnetic)
import os
FORCE_GRID = os.environ.get('KORE_MAG_GRID', '0') == '1'

def use_grid():
    # lorentz4pp, induction4pp: dipole only, and without the induction by a differential rotation (NEW 2026-10-30) or by a
    # background flow (NEW 2026-11-08)
    return (FORCE_GRID or par.B0_tor is not None or (par.B0 is not None and bf.B0_l != 1) or bool(par.diff_rot) or ut.mag_algebra
            or bool(par.bgflow))


def magnetic_powers_grid(walls_only=False):
    '''
    NEW (2026-10-28): the induction work Indu = 2 Re ∫ 𝐛*⋅∇×(𝐮×𝐁₀ + 𝐔₀×𝐛) dV (the 𝐔₀×𝐛 of a differential rotation: NEW
    2026-10-30; NEW 2026-11-08: 𝐔₀ also holds a background flow, bgflow = 1, bgflow.flow_grid) and the complex power of the
    Lorentz force
    2 ∫ 𝐮*⋅[(∇×𝐛)×𝐁₀ + (∇×𝐁₀)×𝐛] dV (no Hendrik), for any B0 (poloidal and/or toroidal, background_field), computed on an
    (r, θ) grid from the vector fields: u (with its (ln ρ0)' part), b and ∇×b synthesised from their l-components
    (velocity, bfield_pol/tor, curl) on Schmidt harmonics (angular.ylm_grid), B0 and ∇×B0 from h, k. Independent of
    lorentz4pp, induction4pp and of the operators. The induction work uses ∫ 𝐛*⋅∇×𝐄 = ∫ 𝐄⋅(∇×𝐛)* dV + [∮ (𝐄×𝐛*)⋅r̂ dS] at the
    walls, 𝐄 = 𝐮×𝐁₀, so it holds whatever the walls (𝐄 = 0 there with no-slip). Needs setup_grid, set_solution, set_bsolution.
    NEW (2026-11-06): walls_only = True computes the wall terms alone (INDU_WALLS, also returned), for the full budget of a fluid
    slipping along its walls when the induction work comes from the spectral formulas (diagnose_magnetic).
    '''
    m = par.m
    lmx = par.lmax + 1
    nB = max([3] + list(bf.pol_profiles()) + list(bf.tor_profiles()))   # NEW (2026-10-30): any background degree
    # FIX (2026-11-03): the products u*⋅(J0×b), E⋅J* have degree in cos θ up to 2 lmx + the degree D of the background (B0, or
    # U0 = ΔΩ r sinθ e_φ of degree n_max + 1); lmx + 12 Gauss nodes integrate them exactly up to D = 22, more nodes beyond
    D = max(nB, dr.n_max() + 1 if par.diff_rot else 0, bgf.nmax() + 1 if par.bgflow else 0)   # (NEW 2026-11-08: U of bgflow)
    xg, wx = np.polynomial.legendre.leggauss(lmx + 12 + max(0, D - 22))
    sg = np.sqrt(1 - xg**2)
    Yg, dYg, Zg = angular.ylm_grid(m, lmx, xg)       # Y_l, ∂θY_l, i m Y_l/sinθ at the nodes, rows l
    Y0, dY0, _ = angular.ylm_grid(0, nB, xg)         # m = 0, for B0
    lp_u, lt_u, _ = ut.ell(m, par.lmax, par.symm)
    lp_b, lt_b, ll = ut.ell(m, par.lmax, ut.bsymm)

    def synth(qs):   # {l: (q, s, t)} at the radii -> (F_r, F_θ, F_φ) on the (r, x) grid: F = q Y r̂ + s r∇Y + t r∇Y×r̂
        Fr = Ft = Fp = 0
        for l, (q, s, t) in qs.items():
            Fr = Fr + np.outer(q, Yg[l]); Ft = Ft + np.outer(s, dYg[l]) + np.outer(t, Zg[l]); Fp = Fp + np.outer(s, Zg[l]) - np.outer(t, dYg[l])
        return Fr, Ft, Fp

    z = np.zeros_like(rk, dtype=complex)
    U, B, J = {}, {}, {}
    for l in ([] if walls_only else ll):   # NEW (2026-11-06): not for the wall terms alone
        if l in lp_u:
            [ [q], [s] ] = cheb2space_pol(l, lp_u, utab[0], 0, rk); U[l] = (q, s, z)
        elif l in lt_u:
            [ t ] = cheb2space_tor(l, lt_u, utab[1], 0, rk); U[l] = (z, z, t)
        if l in lp_b:
            [ [q, q1], [s, s1] ] = bfield_pol(l, 1); B[l] = (q, s, z)
            [ cr, cc, ct ] = curl(l, [q, q1], [s, s1], [z, z]); J[l] = (cr[0], cc[0], ct[0])
        elif l in lt_b:
            [ t, t1 ] = bfield_tor(l, 1); B[l] = (z, z, t)
            [ cr, cc, ct ] = curl(l, [z, z], [z, z], [t, t1]); J[l] = (cr[0], cc[0], ct[0])
    u, b, j = synth(U), synth(B), synth(J)

    def background(rr):   # B0 and ∇×B0 at the radii rr: lists of (r, θ, φ) components on (rr, x)
        Bv = [0*np.outer(rr, xg)]*3; Jv = [0*np.outer(rr, xg)]*3
        # NEW (2026-10-30): every degree n of the background (background_field.pol_profiles, tor_profiles)
        for n, f in bf.pol_profiles().items():   # cnorm ∇×∇×(h Y_n 𝐫): B_r = L h Y/r, B_θ = (h' + h/r) ∂θY; ∇×B0 = (D h) ∂θY φ̂
            L0 = n*(n + 1.); hh = [ f(rr, d) for d in (0, 1, 2) ]
            Dh = hh[2] + 2*hh[1]/rr - L0*hh[0]/rr**2
            Bv = [ Bv[0] + np.outer(L0*hh[0]/rr, Y0[n]), Bv[1] + np.outer(hh[1] + hh[0]/rr, dY0[n]), Bv[2] ]
            Jv = [ Jv[0], Jv[1], Jv[2] + np.outer(Dh, dY0[n]) ]
        for n, f in bf.tor_profiles().items():   # cnorm_tor ∇×(k Y_n 𝐫): B_φ = -k ∂θY; ∇×B0: r L k Y/r, θ (k' + k/r) ∂θY
            L0 = n*(n + 1.); kk = [ f(rr, d) for d in (0, 1) ]
            Bv = [ Bv[0], Bv[1], Bv[2] - np.outer(kk[0], dY0[n]) ]
            Jv = [ Jv[0] + np.outer(L0*kk[0]/rr, Y0[n]), Jv[1] + np.outer(kk[1] + kk[0]/rr, dY0[n]), Jv[2] ]
        return Bv, Jv

    def cross(a, c):
        return [ a[1]*c[2] - a[2]*c[1], a[2]*c[0] - a[0]*c[2], a[0]*c[1] - a[1]*c[0] ]
    def vol(f):   # ∫ f dV, f on (rk, x), the e^{imφ} factors already combined (2π)
        return 2*np.pi*np.sum((wk*rk**2)[:, None]*wx[None, :]*f)*(ut.rcmb - par.ricb)/2

    def rotation(rr):   # NEW (2026-10-30): the background flow U0 = Gaspard ΔΩ r sinθ e_φ at the radii rr, (r, θ, φ) on (rr, x)
        z0 = 0*np.outer(rr, xg)
        out = [ z0, z0, z0 ]
        if par.diff_rot:
            dO = sum(np.outer(dr.fX(n, rr, 0), np.polynomial.legendre.Legendre.basis(n)(xg)) for n in dr.degrees())
            out = [ z0, z0, par.Gaspard*dO*np.outer(rr, sg) ]
        if par.bgflow:   # NEW (2026-11-08): and the background flow of bgflow.py, its three components
            out = [ a + c for a, c in zip(out, bgf.flow_grid(np.asarray(rr, dtype=float), xg, nd=0)['U'][0]) ]
        return out

    if not walls_only:
        B0v, J0v = background(rk)
        G = [ a + c for a, c in zip(cross(j, B0v), cross(J0v, b)) ]
        plor = 2*vol(sum(np.conj(uc)*gc for uc, gc in zip(u, G)))
        # the electric field of the induction: 𝐄 = 𝐮×𝐁₀ + 𝐔₀×𝐛 (NEW 2026-10-30: the second term, with a differential rotation)
        E = [ a + c for a, c in zip(cross(u, B0v), cross(rotation(rk), b)) ]
        indu = 2*np.real(vol(sum(ec*np.conj(jc) for ec, jc in zip(E, j))))
    # wall terms 2 Re ∮ (E x b*).r̂ dS, from u and b at the walls (zero for no-slip walls, where u = 0)
    rw = np.array([par.ricb if par.ricb > 0 else 0.5, ut.rcmb])   # (a dummy inner radius in a full sphere, its term is not used)
    lh = rap.logrhoX(rw, 1)
    Ut = [ cheb2space_table(usol2[0], 1, rw), cheb2space_table(usol2[1], 0, rw) ]
    Bt = [ cheb2space_table(bsol2[0], 1, rw), cheb2space_table(bsol2[1], 0, rw) ]
    Uw, Bw = {}, {}
    zw = np.zeros(2, dtype=complex)
    for l in ll:
        L = l*(l + 1.)
        if l in lp_u:
            k_ = list(lp_u).index(l); P0, P1 = Ut[0][0, k_], Ut[0][1, k_]
            Uw[l] = (L*P0/rw, P1 + P0/rw + lh*P0, zw)
        elif l in lt_u:
            Uw[l] = (zw, zw, Ut[1][0, list(lt_u).index(l)])
        if l in lp_b:
            k_ = list(lp_b).index(l); F0, F1 = Bt[0][0, k_], Bt[0][1, k_]
            Bw[l] = (L*F0/rw, F1 + F0/rw, zw)
        elif l in lt_b:
            Bw[l] = (zw, zw, Bt[1][0, list(lt_b).index(l)])
    uw, bw = synth(Uw), synth(Bw)
    Ew = [ a + c for a, c in zip(cross(uw, background(rw)[0]), cross(rotation(rw), bw)) ]   # U0 is not zero at the walls
    flux = (Ew[1]*np.conj(bw[2]) - Ew[2]*np.conj(bw[1]))          # (E x b*).r̂
    walls = 2*np.pi*np.sum(wx[None, :]*flux*(rw**2)[:, None], axis=1)
    global INDU_WALLS, INDU_ROT_SOLID
    INDU_WALLS = 2*np.real(walls[1] - (walls[0] if par.ricb > 0 else 0))   # no inner wall in a full sphere
    # NEW (2026-11-06): the U0×b part at a wall with a turning conducting solid behind it (INDU_ROT_SOLID)
    solid = [ par.ricb > 0 and (ut.cic or ut.bessel or ut.twa_i), ut.twa_o or ut.resolved_mantle ]   # (NEW 2026-11-06: the
    if (par.diff_rot or par.bgflow) and any(solid):                                         # resolved mantle, the Bessel core)
        Er = cross(rotation(rw), bw)
        rot = 2*np.pi*np.sum(wx[None, :]*(Er[1]*np.conj(bw[2]) - Er[2]*np.conj(bw[1]))*(rw**2)[:, None], axis=1)
        INDU_ROT_SOLID = 2*np.real((rot[1] if solid[1] else 0) - (rot[0] if solid[0] else 0))
    # NEW (2026-11-06): likewise at a wall where the solid behind it moves with the fluid, with velocity 𝐕 (a forced problem):
    # u_t where the wall is no-slip (the libration), plus uᵣ r̂ at a thin conducting layer with forced_walls 'deforming' (the
    # layer deforms with the boundary). What stays in the induction work follows the solid's rows. A thin conducting layer
    # moves with the fluid (bc_b_twa), so 𝐕×B0 drives no current of its own in it, and the forcing works against the force on
    # its surface current 𝐊 = 𝐧×(𝐛_beyond - 𝐛): 2 Re ∮ (𝐕×B0)⋅𝐊* dS, 𝐛_beyond the potential field just beyond the wall. A
    # conducting inner core (Chebys or Bessel) and the resolved mantle have rows without 𝐕×B0 (bc_b_cic, bc_b_bessel,
    # bc_b_mantle): next to them the tangential electric field jumps by the consoidal part of the u_t×B0 of the libration's
    # rigid turning, which induces nothing inside the solid (its toroidal part is 0), and whose flux
    # 2 Re ∮ ((𝐕×B0)×𝐛_tor*)⋅𝐧 dS is the work of that turning, 𝐛_tor the part of b from g (n: out of the fluid). Where such a
    # thick solid deforms with the boundary (background.deforms) its motion inside is modelled, and the work of its uᵣ is the
    # volume integral deform_work (spin_doctor). With forced_walls 'rigid' a radial boundary flow passes through the wall into
    # the solid at rest, whose rows take its u×B0 (assemble.bc_b_radial): its flux leaves the fluid for the solid's field, as
    # at a slipping wall, and stays in INDU_WALLS
    if par.forcing != 0 and any(solid):
        ns = np.array([ float(par.bci == 1 and par.ViscosD > 0), float(par.bco == 1 and par.ViscosD > 0) ])[:, None]
        rad = float(bg.forced_walls() == 'deforming')*np.array([ float(ut.twa_i), float(ut.twa_o) ])[:, None]
        B0w = background(rw)[0]
        # the mean B0 across a thin layer that deforms: the toroidal part falls to 0 across it (background.thin_drop)
        B0m = [ B0w[0], B0w[1], B0w[2] + 0.5*sum(np.outer(f(rw, 0), dY0[n]) for n, f in bf.tor_profiles().items()) ]
        z0 = 0*uw[0]
        Ev = [ a + c for a, c in zip(cross([ uw[0]*rad, z0, z0 ], B0m), cross([ z0, uw[1]*ns, uw[2]*ns ], B0w)) ]
        bt = synth({ l: v for l, v in Bw.items() if l in lt_b })
        Bb = {}
        for l in lp_b:   # f ~ r^l inside the inner wall, r^-(l+1) beyond the outer one: s = (l+1) f/r, -l f/r
            F0 = Bt[0][0, list(lp_b).index(l)]
            Bb[l] = (l*(l + 1.)*F0/rw, np.array([ (l + 1)*F0[0]/rw[0], -l*F0[1]/rw[1] ]), zw)
        bb = synth(Bb)
        thin = [ ut.twa_i, ut.twa_o ]
        mov = []
        for w in (0, 1):
            d = [ bw[c][w] - bb[c][w] for c in range(3) ] if thin[w] else [ bt[c][w] for c in range(3) ]
            mov.append( 2*np.pi*rw[w]**2*np.sum(wx*(Ev[1][w]*np.conj(d[2]) - Ev[2][w]*np.conj(d[1]))) )
        INDU_ROT_SOLID += 2*np.real((mov[1] if solid[1] else 0) - (mov[0] if solid[0] else 0))
    if walls_only:
        return INDU_WALLS
    indu += INDU_WALLS
    return indu, plor



def diagnose( usol, tsol, Ra, Rb, ncpus, csol=0):
    '''
    Computes kinetic energy, internal and kinetic energy dissipation,
    and input power from body forces. Integrated From r=Ra to r=Rb, and
    angularly over the whole sphere.
    NEW (2026-10-20): serial, as in the conductive_IC branch (2026-10-16). The radial functions of every l and
    their derivatives are evaluated once per solution (set_solution, cheb2space_table) and the workers, run
    one l after another, read them from these tables. Before, each l was a task for a pool of ncpus processes,
    and each worker evaluated its radial functions with chebval (a Python loop over N) several times over
    (velocity, coriolis with the neighbouring l's, visforce, buoyancy). ncpus is no longer used.
    NEW (2026-11-08): csol, the solution of the composition (compositional = 1), for the buoyancy and for
    diagnose_composition, which takes the tables set here.
    '''

    out_t = 0

    setup_grid(Ra, Rb)  # quadrature grid and background profiles, as module globals
    set_solution(usol, tsol, csol=csol)  # the solution and the tables of its radial functions at rk, as module globals
    if par.diff_rot:
        diff_rot_setup()  # differential-rotation force and heat-equation terms, as module globals
    if par.bgflow:
        bgflow_setup()    # NEW (2026-11-08): those of a background flow

    [ lp_u, lt_u, ll ] = ut.ell(par.m, par.lmax, par.symm)  # the l-indices of the flow field

    out_u = np.array([ flow_worker(l) for l in ll ])

    if par.thermal:
        out_t = np.array([ thermal_worker(l) for l in lp_u ])

    return [ out_u, out_t ]



def diagnose_composition():
    '''
    NEW (2026-11-08): the budget of the composition (composition_worker), one row per l of the poloidals, on the grid and the
    tables that diagnose has just set (with csol)
    '''
    lp = ut.ell(par.m, par.lmax, par.symm)[0]
    return np.array([ composition_worker(l) for l in lp ])
    


def diagnose_4plot( ncpus, usol, tsol, radii, field0, csol=0, bsol=None):
    '''
    Returns the [rad,con,tor] components of the field field0 (see worker_4plot), one row per l, at the radii (a curl,
    'curl_...', with its first r-derivative: [l, component, value 0 or derivative 1, radius], as spin_doctor reads it).
    NEW (2026-10-20): serial, from tables of the radial functions at the radii (set_solution); ncpus is no longer used.
    NEW (2026-11-08): csol, the composition, whose buoyancy 'buo' includes; bsol, the magnetic field (expand_reshape_sol
    with ut.bsymm), for field0 = 'b' (usol may then be zero_flow()).
    '''

    global field
    field =  field0

    set_radii(radii)  # NEW (2026-11-04): the radii and the background profiles there

    set_solution(usol, tsol, csol=csol)  # NEW (2026-10-20): the solution and the tables of its radial functions at the radii
    if bsol is not None:
        set_bsolution(bsol, 1)   # NEW (2026-11-08): f, g and their first r-derivatives at the radii

    [ lp_u, lt_u, ll ] = ut.ell(par.m, par.lmax, par.symm)  # the l-indices of the flow field

    out = np.array([ worker_4plot(l) for l in ll ])

    return out



def identify(sol2, vsymm=None):
    '''
    [the dominant degree, the width of the band of degrees above 0.1 of its amplitude, the amplitude of the highest degree
    relative to the dominant one] of the solution sol2 (expand_reshape_sol). NEW (2026-11-06): vsymm, the equatorial symmetry
    of sol2 (default par.symm, the flow's; ut.bsymm for the magnetic field), and a scalar field (the entropy, one array with the
    degrees of P), for the runs without the flow (hydro = 0)
    '''

    threshold = 0.1

    scalar = not isinstance(sol2, (list, tuple))
    P = np.abs(sol2 if scalar else sol2[0])
    T = None if scalar else np.abs(sol2[1])

    lm1 = par.lmax - par.m + 1
    s   = int( (par.symm if vsymm is None else vsymm)*0.5 + 0.5 ) # s=0 if antisymm, s=1 if symm
    idp = np.arange( (np.sign(par.m)+s  )%2, lm1, 2, dtype=int)
    idt = np.arange( (np.sign(par.m)+s+1)%2, lm1, 2, dtype=int)
    ll  = np.arange( par.m+1-np.sign(par.m), par.lmax+2-np.sign(par.m), dtype=int)

    amps      = np.zeros_like(ll,dtype=float)
    amps[idp] = np.sum(P, axis=1)
    if not scalar:
        amps[idt] = np.sum(T, axis=1)

    maxid1      = np.argmax( amps )
    max_ell     = ll[ maxid1 ]
    maxids      = amps > threshold * amps[ maxid1 ]
    spread      = ll[maxids][-1] - ll[maxids][0]
    convergence = amps[ idp[-1] if scalar else -1 ] / amps[ maxid1 ]   # (a scalar has no toroidal degrees)

    return max_ell, spread, convergence



_centre_taylor = {}   # NEW (2026-11-04): (N, lmax) -> the Taylor table and grid of centre_irregularity

def centre_irregularity(groups):
    '''
    NEW (2026-11-04): ricb = 0, how far a solution is from regular at r = 0. A degree-l component of a regular field goes as
    r^l there, but kore's radial basis keeps only the parity of l, so the Taylor terms a_p r^p, p < l, of that parity can
    appear: in spurious or unresolved modes, or with a background that is not regular at the centre (docs/numerical-method.md,
    Regularity at the centre). Returns the largest |a_p| h^p, h = π/N (about the spacing of the Chebyshev points at the
    centre), over the components of a group, relative to the largest |X_l(r)| of the group, and the largest over the groups.
    groups: [[(C, ls), ...], ...], e.g. [[(P, lp), (T, lt)]] for the flow, C the Chebyshev coefficients (one row per l, as
    from expand_reshape_sol) and ls their degrees. WHY h^p: a_p alone amplifies the radial truncation by up to N^p/p!, while
    the h^p T_k^(p)(0)/p! of one T_k, k < N, add up to at most cosh π ≈ 12, so a resolved regular solution stays at round-off
    for any N and l, and a solution irregular at the scale of the grid gives about 1.
    '''
    P = int(max(max(ls) for group in groups for _, ls in group))
    if (par.N, P) not in _centre_taylor:
        N, h = par.N, np.pi/par.N
        t = np.zeros((P, N))   # t[p, k] = h^p T_k^(p)(0)/p!, the coefficient of y^p in T_k(h y) (x = r/rcmb at ricb = 0)
        t[0, 0] = 1.
        if N > 1 and P > 1:
            t[1, 1] = h
        for k in range(1, N - 1):   # T_(k+1) = 2x T_k - T_(k-1)
            t[1:, k + 1] = 2*h*t[:-1, k]
            t[:, k + 1] -= t[:, k - 1]
        _centre_taylor.clear()
        _centre_taylor[(par.N, P)] = (t, ch.chebvander(np.linspace(0, 1, 2*N + 1), N - 1))
    t, V = _centre_taylor[(par.N, P)]
    p = np.arange(P)
    out = 0.0
    for group in groups:
        big = max(np.abs(V @ np.asarray(C).T).max() for C, _ in group)
        if big == 0:
            continue
        worst = 0.0
        for C, ls in group:
            l = np.asarray(ls)[:, None]
            low = (p < l) & ((l - p) % 2 == 0)   # the powers a regular solution lacks
            if low.any():
                worst = max(worst, np.abs((np.asarray(C) @ t.T)[low]).max())
        out = max(out, worst/big)
    return out
