import scipy.linalg as las
import scipy.optimize as so
import scipy.sparse as ss
import scipy.special as scsp
import scipy.fftpack as sft
import numpy.polynomial.chebyshev as ch
import numpy as np
import parameters as par
from typing import Union, Callable, Any
from warnings import warn

'''
A library of various function definitions and utilities
'''

# ----------------------------------------------------------------------------------------------------------------------
# First some global variables: -----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if par.forcing == 0:
    wf = 0
else:
    wf = par.forcing_frequency

rcmb   = 1

N1     = int(par.N/2) * int(1 + np.sign(par.ricb)) + int((par.N%2)*np.sign(par.ricb)) # N/2 if no IC, N if present
n      = int(N1*(par.lmax-par.m+1)/2)
n0     = int(par.N*(par.lmax-par.m+1)/2)
m      = par.m
lmax   = par.lmax
vsymm  = par.symm

symm1 = (2*np.sign(par.m) - 1) * par.symm  # symm1=par.symm if m>0, symm1 = -par.symm if m=0

# this gives the size (rows or columns) of the main matrices
sizmat = 2*n*par.hydro + 2*n*par.magnetic + n*par.thermal + n*par.compositional

s = int( (vsymm+1)/2 ) # s=0 if antisymm, s=1 if symm
m_top = m + 1-s
m_bot = m + s
if m_top == 0: m_top = 2
if m_bot == 0: m_bot = 2
lmax_top = lmax + 1 + (1-2*np.sign(m))*s
lmax_bot = lmax + 1 + (1-2*np.sign(m))*(1-s)

beta_actual = 0
if par.B0 in ['axial','dipole','G21 dipole','Luo_S1']:
    symmB0 = -1
    B0_l   =  1
elif par.B0 == 'Luo_S2':
    symmB0 = 1
    B0_l   = 2
elif par.B0 == 'FDM':
    symmB0 = int((-1)**par.B0_l)
    B0_l   = par.B0_l

bsymm = par.symm * symmB0  # induced magnetic field (b) symmetry follows from u and B0

B0list = ['axial', 'dipole', 'G21 dipole', 'Luo_S1', 'Luo_S2', 'FDM']
B0type = B0list.index(par.B0)

if par.innercore == 'insulator':
    innercore_mag_bc = 0
elif par.innercore == 'TWA':
    innercore_mag_bc = 1

if par.mantle == 'insulator':
    mantle_mag_bc = 0
elif par.mantle == 'TWA':
    mantle_mag_bc = 1

thermal_heating_list = ['internal', 'differential', 'two zone', 'user defined']
heating = thermal_heating_list.index(par.heating)

compositional_background_list = ['internal', 'differential']
compositional_background = compositional_background_list.index(par.comp_background)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------



def decode_label( labl ):
    '''
    Returns the stripped label, the indices rx, hx, dx, the section,
    and up to two profile identifiers with their corresponding derivative order
    [ lablx, rx, hx, dx, section, profid1, dp1, profid2, dp2 ]
    '''

    [ lablx, rx, hx, dx, section, profid1, dp1, profid2, dp2 ] = [ None ]*9

    lablx = labl[:-2]  # label without the section

    if labl[:2] == 'q1':
        rx = 6  # this is an index, not a power, it corresponds to r**-1
    elif labl[0] == 'r':
        rx = int(labl[1])  # index of the power of r

    dx = int(labl[-3])  # operator's derivative order
    section = labl[-1]    # section

    if   len(labl) in [10, 15, 20]:  # h is there

        hx = int(labl[4])

        if len(labl) in [ 15, 20]:   # h and at least 1 profile
            profid1 = labl[6:9]
            dp1 = int(labl[9])

            if len(labl) == 20:      # h and 2 profiles, e.g. 'r1_h0_rho2_eta1_D0_f'
                profid1 = labl[11:14]
                dp1 = int(labl[14])

    elif len(labl) in [12, 17]:  # no h and at least 1 profile

        profid1 = labl[3:6]
        dp1 = int(labl[6])

        if len(labl) == 17:  # no h and 2 profiles, e.g. 'r0_rho1_eta3_D2_u'

            profid2 = labl[8:11]
            dp2 = int(labl[11])

    return [ lablx, rx, hx, dx, section, profid1, dp1, profid2, dp2 ]



def labelit( labl, section, rplus=0):
    '''
    Appends a section string to each label in the list labl.
    Optionally, it increases the r power in each label by rplus.
    '''

    out = []

    for labl1 in labl:

        if rplus>0:  # increase the power of r in the label by rplus:

            old_rpow = labl1[:2]
            r_or_q = old_rpow[0]
            if old_rpow == 'q1':
                orpw = -1
                r_or_q = 'r'
            else:
                orpw = int(old_rpow[1])

            new_rpow = r_or_q + str( orpw + rplus )
            labl1 = labl1.replace(old_rpow, new_rpow, 1)

        # append the appropriate section string
        out += [ labl1 + '_' + section ]

    return out



def packit( lista_local, mtx, row, col):
    '''
    Appends sparse matrix data, row, and col info to lista_local
    '''
    mtx.eliminate_zeros()
    mtx = mtx.tocoo()
    blk = [mtx.data, mtx.row + row, mtx.col + col]
    for q in [0,1,2]:
        lista_local[q]= np.concatenate( ( lista_local[q], blk[q] ) )

    return lista_local



def ell( m, lmax, vsymm) :
    # Returns the l values for the poloidal flow (section u) and l values for toroidal flow (section v)
    # ll are *all* the l values and (idp,idt) are the indices for poloidals and toroidals respectively
    lm1 = lmax - m + 1
    s   = int( vsymm*0.5 + 0.5 ) # s=0 if antisymm, s=1 if symm
    idp = np.arange( (np.sign(m)+s  )%2, lm1, 2, dtype=int)
    idt = np.arange( (np.sign(m)+s+1)%2, lm1, 2, dtype=int)
    ll  = np.arange( m+1-np.sign(m), lmax+2-np.sign(m), dtype=int)

    return [ ll[idp], ll[idt], ll ]



def remroco(matrix, overall_parity, vector_parity):
    '''
    Removes rows and cols from matrix according to parity
    overall_parity determines rows
    vector_parity determines cols
    '''
    idj = int((1-overall_parity)/2) # overall_parity = 1 removes odd row indices
    idk = int((1-vector_parity)/2)  # vector_parity = 1 removes odd col indices

    return matrix[ idj::2, idk::2 ]



def chebco_f(func,N,ricb,rcmb,tol,args=None):
    '''
    Returns the first N Chebyshev coefficients
    from 0 to N-1, of func(r)
    '''
    i = np.arange(0, N)
    xi = np.cos(np.pi * (i + 0.5) / N)

    if ricb > 0:
        ri = (ricb + (rcmb - ricb) * (xi + 1) / 2.)
    elif ricb == 0 :
        ri = rcmb * xi

    if args is None:
        tmp = sft.dct(func(ri))
    else:
        tmp = sft.dct(func(ri,args))

    out = tmp / N
    out[0] = out[0] / 2.
    out[np.absolute(out) <= tol] = 0.
    return out



def chebco_rf(func,rpower,N,ricb,rcmb,tol,args=None):
    '''
    Returns the first N Chebyshev coefficients
    from 0 to N-1, of the function
    r**rpower * func(r)
    '''
    i = np.arange(0, N)
    xi = np.cos(np.pi * (i + 0.5) / N)

    if ricb > 0:
        ri = (ricb + (rcmb - ricb) * (xi + 1) / 2.)
    elif ricb == 0 :
        ri = rcmb * xi

    if args is None:
        tmp = sft.dct(ri**rpower * func(ri))
    else:
        tmp = sft.dct(ri**rpower * func(ri,args))

    out = tmp / N
    out[0] = out[0] / 2.
    out[np.absolute(out) <= tol] = 0.
    return out



def chebco(powr, N, tol, ricb, rcmb):
    '''
    Returns the first N Chebyshev coefficients
    from 0 to N-1, of the function
    ( ricb + (rcmb-ricb)*( x + 1 )/2. )**powr
    '''
    i = np.arange(0,N)
    xi = np.cos(np.pi*(i+0.5)/N)

    if ricb == 0:                                    # No inner core ---> Chebyshev domain [-1,1] mapped to [-rcmb, rcmb]
        ai = ( rcmb*xi )**powr
    else:
        ai = ( ricb + (rcmb-ricb)*(xi+1)/2. )**powr  # With inner core -> Chebyshev domain [-1,1] mapped to [ ricb, rcmb]

    out = sft.dct(ai)/N
    out[0]=out[0]/2.
    out[np.absolute(out)<=tol]=0.

    return out



def Dcheb(ck, ricb, rcmb):
    '''
    The derivative of a Chebyshev expansion with coefficients ck
    returns the coefficients of the derivative in the Chebyshev basis
    assumes ck computed for r in the domain [ricb,rcmb] (if ricb>0)
    or r in [-rcmb,rcmb] if ricb=0.
    '''
    c = np.copy(ck)
    c[0] = 2.*c[0]
    s =  np.size(c)
    out = np.zeros_like(c)  #,dtype=np.complex128)
    out[-2] = 2.*(s-1.)*c[-1]
    for k in range(s-3,-1,-1):
        out[k] = out[k+2] + 2.*(k+1)*ck[k+1]
    out[0] = out[0]/2.

    if ricb == 0 :
        out1 = out/rcmb
    else :
        out1 = 2*out/(rcmb-ricb)

    return out1



def Dn_cheb(ck, ricb, rcmb, Dorder):
    '''
    Returns the Chebyshev coefficients of the derivatives (up to order n)
    of a Chebyshev expansion with coefficients ck. Assumes ck is computed
    for r in the domain [ricb,rcmb] (if ricb>0) or r in [-rcmb,rcmb] if ricb=0.
    First column correspond to the first derivative, last column to the
    n-th derivative.
    '''
    c = np.copy(ck)
    s = np.size(c)
    out = np.zeros((s,Dorder), ck.dtype)
    out[:,0] = Dcheb(c, ricb, rcmb)
    for j in range(1,Dorder):
        out[:,j] = Dcheb( out[:, j-1], ricb, rcmb)

    return out



def chebify(func, Dorder, tol):
    '''
    Returns the Chebyshev coeffs of function func,
    and its derivatives (as columns) up to order Dorder.
    '''
    c0 = chebco_rf( func, 0, par.N, par.ricb, rcmb, tol)
    out = np.c_[ c0, Dn_cheb(c0, par.ricb, rcmb, Dorder) ]

    return out



def cheb3Product(ck1, ck2, ck3, tol):
    '''
    Computes the product of three Chebyshev series
    '''
    out = Mlam(ck1,0,0) * ( Mlam(ck2,0,0) * ck3 )
    out[ np.absolute(out) <= tol ] = 0.0
    return out



def cheb2Product(ck1, ck2, tol):
    '''
    Computes the product of two Chebyshev series
    '''
    out = Mlam(ck1,0,0) * ck2
    out[ np.absolute(out) <= tol ] = 0.0
    return out




def get_radial_derivatives( func, rorder, Dorder, tol):
    '''
    This function computes terms of the form r^n d^m/dr^m of a
    radial profile in Chebyshev space.

    Parameters
    ----------
    func   : function
        Radial profile in the form of a function (can be found in utils)
    rorder : integer
        Highest order of radial power
    Dorder : integer
        Highest order of radial derivative
    tol    : real
        Tolerance for Chebyshev transforms for radial powers

    Returns
    -------
    rd_prof : 2D list
        List such that rd_prof[i][j] defines the Chebyshev coefficients of
        r^i d^j/dr^j of the radial profile
    '''

    # Make sure these are integers
    rorder = int(rorder)
    Dorder = int(Dorder)

    rd_prof = [ [ [] for j in range(Dorder+1) ] for i in range(rorder+1) ] #List for Cheb coeffs to r^n D^m profile
    dnprof = [ [] for i in range(Dorder+1) ] #List for Cheb coeffs of nth derivative of profile
    # Cheb coeffs of profile
    dnprof[0] = chebco_f( func, par.N, par.ricb, rcmb, par.tol_tc )

    for i in range(rorder+1):
        rn  = chebco(i, par.N, tol, par.ricb, rcmb) #Cheb coeffs of r^i
        rd_prof[i][0] =  chebProduct(dnprof[0],rn,par.N,par.tol_tc) #Cheb coeffs of r^i profile
        for j in range(1,Dorder+1):
        # Cheb coeffs of r^i D^j profile
            if i==0:
                # These only need to be computed once
                dnprof[j] = Dcheb(dnprof[j-1],par.ricb,rcmb)
            rd_prof[i][j] = chebProduct(dnprof[j],rn,par.N,par.tol_tc)

    return rd_prof



def jl_smx(l,x,d):
    '''
    Spherical Bessel function of the first kind and derivatives,
    small argument (x<<1) only, 0 <= d <= 3
    '''
    c1 =  (2**l)*scsp.factorial(l)/scsp.factorial(2*l+1)
    c2 = -(2**l)*scsp.factorial(l+1)/scsp.factorial(2*l+3)
    c3 =  (2**l)*scsp.factorial(l+2)/(2*scsp.factorial(2*l+5))

    if d == 0:
        out = c1*x**l + c2*x**(l+2) + c3*x**(l+4)

    elif d == 1:
        out = c1*l*x**(l-1) + c2*(l+2)*x**(l+1) + c3*(l+4)*x**(l+3)

    elif d == 2:
        if l>=2:
            out = c1*l*(l-1)*x**(l-2) + c2*(l+2)*(l+1)*x**l + c3*(l+4)*(l+3)*x**(l+2)
        elif l==1:
            out = c2*(l+2)*(l+1)*x**l + c3*(l+4)*(l+3)*x**(l+2)

    elif d == 3:
        if l>=3:
            out = c1*l*(l-1)*(l-2)*x**(l-3) + c2*(l+2)*(l+1)*l*x**(l-1) + c3*(l+4)*(l+3)*(l+2)*x**(l+1)
        else:
            out = c2*(l+2)*(l+1)*l*x**(l-1) + c3*(l+4)*(l+3)*(l+2)*x**(l+1)

    return out



def jl(l,x,d):
    '''
    Spherical Bessel function of the first kind
    d is zero or 1
    '''
    out = np.zeros_like(x)

    k = x<1e-3
    out[k]  = jl_smx(l,x[k],d)
    out[~k] = scsp.spherical_jn(l,x[~k],derivative=d)

    return out



def nl(l,x,d):
    '''
    Spherical Bessel function of the second kind
    '''
    return scsp.spherical_yn(l,x,derivative=d)



def dlogjl(l,x):
    '''
    Log derivative of spherical Bessel function of the first kind
    '''
    # functional form of the numerator and denominator in the continued fraction
    def num(k,x):
        return -1
    def denom(k,x):
        return (1 + 2*k) / x

    # Lentz-Thompson algorithm
    def lentz_thompson(a, b, b0, eps=1e-15, acc=1e-12):
        if b0 == 0:
            f0 = eps
        else:
            f0 = b0
        c0 = f0;d0 = 0
        c = c0;d = d0;f = f0
        for i in range(len(a)):
            c = b[i] + a[i] / c
            if c == 0:
                c = eps
            d = b[i] + a[i] * d
            if d == 0:
                d = eps
            d = 1 / d
            Delta = c * d
            f = f * Delta
            if abs(Delta - 1) < acc:
                break
        return f

    # first 100 numerators and denominators in continued fraction
    a = [num(k,x) for k in range(l+1, l+101)]
    b = [denom(k,x) for k in range(l+1, l+101)]

    # first term in the sum (in front of first quotient)
    b0 = l / x

    return lentz_thompson(a, b, b0, eps=1e-30, acc=1e-14)



def findbeta(args):
    '''
    root finding for beta, needed for the Free Decay Modes
    '''
    beta0 = args[0]
    ell   = args[1]
    ricb  = args[2]

    def f0(x, ric, l):
        if ric>0:
            # Zhang & Fearn, GAFD (1995), page 196.
            return jl(l+1,x*ric,0) * nl(l-1,x,0) - jl(l-1,x,0) * nl(l+1,x*ric,0)
        elif ric==0:
            # Gubbins & Roberts (1987), page 49.
            return jl(l-1,x,0)

    sol = so.root( f0, beta0, args=(ricb, ell) )
    beta1 = sol.x[0]

    return beta1
if par.B0 == 'FDM':
    beta_actual = findbeta([par.beta, B0_l, par.ricb])



def h0(rr, kind, args):
    '''
    Radial poloidal function for the background magnetic field times
    a power of r
    args[0] = guess for beta (FDM, radial complexity)
    args[1] = order l (FDM, l=1 is dipole)
    args[2] = ricb
    args[3] = power of r

    If extending to r<0 then this function has parity (-1)**(l+rp)
    '''
    b    = args[0]
    l    = args[1]
    ricb = args[2]
    rp   = args[3]

    r = rr[rr>0]

    if   kind == 'axial':       # axial uniform field in the z direction
        l = 1
        out = (1/2)*r**(1+rp)

    elif kind == 'dipole' and ricb > 0 :      # dipole, singular at r=0
        l = 1
        out = (1/2)*r**(-2+rp)

    elif kind == 'G21 dipole':  # Felix's dipole (Gerick 2021)
        l = 1
        out = (1/6)*r**(1+rp) - (1/10)*r**(3+rp)

    elif kind == 'Luo_S1':
        l = 1
        out = (5 - 3*r**2)*r**(1+rp)

    elif kind == 'Luo_S2':
        l = 2
        out = (157-296*r**2+143*r**4)*r**(2+rp)

    elif kind == 'FDM':         # poloidal Free Decay Mode
        b = findbeta(args)
        x = b*r

        if ricb==0:
            # Gubbins & Roberts (1987), page 49, eq. 3.67
            out = jl(l,x,0)*r**rp
        else:
            # Zhang & Fearn, GAFD (1995), page 196, eq. 2.7
            out = ( jl(l,x,0)*nl(-1 + l,b,0) - jl(-1 + l,b,0)*nl(l,x,0) )*r**rp

    out2 = np.zeros_like(rr)
    out2[rr>0] = out
    if (ricb == 0) and (np.size(rr[rr>0]) == np.size(rr[rr<0])):
        out2[rr<0] = np.flipud(out)*(-1)**(l+rp)

    return out2



def h1(rr, kind, args):
    '''
    First radial derivative of the function h0, times a power of r
    '''
    b    = args[0]
    l    = args[1]
    ricb = args[2]
    rp   = args[3]

    r = rr[rr>0]

    if kind == 'axial':         # axial uniform field in the z direction
        l = 1
        out = (1/2)*r**rp

    elif kind == 'dipole' and ricb > 0 :      # dipole, singular at r=0
        l = 1
        out = -r**(-3+rp)

    elif kind == 'G21 dipole':  # Felix's dipole (Gerick 2021)
        l = 1
        out = (1/6)*r**rp - (3/10)*r**(2+rp)

    elif kind == 'Luo_S1':
        l = 1
        out = (5 - 9*r**2)*r**rp

    elif kind == 'Luo_S2':
        l = 2
        out = 2*r**(1 + rp)*(157 - 592*r**2 + 429*r**4)

    elif kind == 'FDM':
        b = findbeta(args)
        x = b*r
        if ricb==0:
            out = b*jl(l,x,1)*r**rp
        else:
            out = ( b*(jl(l,x,1)*nl(-1 + l,b,0) - jl(-1 + l,b,0)*nl(l,x,1)) )*r**rp

    out2 = np.zeros_like(rr)
    out2[rr>0] = out
    if (ricb == 0) and (np.size(rr[rr>0]) == np.size(rr[rr<0])):
        out2[rr<0] = np.flipud(out)*(-1)**(l-1+rp)

    return out2



def h2(rr, kind, args):
    '''
    Second radial derivative of the function h0, times a power of r
    '''
    b    = args[0]
    l    = args[1]
    ricb = args[2]
    rp   = args[3]

    r = rr[rr>0]

    if kind == 'axial':         # axial uniform field in the z direction
        l = 1
        out = np.zeros_like(r)

    elif kind == 'dipole' and ricb > 0 :      # dipole, singular at r=0
        l = 1
        out = 3*r**(-4+rp)

    elif kind == 'G21 dipole':  # Felix's dipole (Gerick 2021)
        l = 1
        out = (6/10)*r**(1+rp)

    elif kind == 'Luo_S1':
        l = 1
        out = -18*r**(1+rp)

    elif kind == 'Luo_S2':
        l = 2
        out = 2*r**rp*(157 - 1776*r**2 + 2145*r**4)

    elif kind == 'FDM':
        b = findbeta(args)
        x = b*r
        if ricb==0:
            k = x<1e-3
            out = np.zeros_like(r)
            out[ k] = b**2*jl_smx(l,x[k],2)*r[k]**rp
            out[~k] = b**2*jl(-1 + l,x[~k],1)*r[~k]**rp + ((1 + l)*(jl(l,x[~k],0) - x[~k]*jl(l,x[~k],1)))*r[~k]**(-2+rp)
        else:
            out= ((x**2*jl(-1 + l,x,1) + (1 + l)*(jl(l,x,0) - x*jl(l,x,1)))*nl(-1 + l,b,0) \
             - jl(-1 + l,b,0)*(x**2*nl(-1 + l,x,1) + (1 + l)*(nl(l,x,0) - b*r*nl(l,x,1))))*r**(-2+rp)

    out2 = np.zeros_like(rr)
    out2[rr>0] = out
    if (ricb == 0) and (np.size(rr[rr>0]) == np.size(rr[rr<0])):
        out2[rr<0] = np.flipud(out)*(-1)**(l+rp)

    return out2



def h3(rr, kind, args):
    '''
    Third radial derivative of the function h0, times a power of r
    '''
    b    = args[0]
    l    = args[1]
    ricb = args[2]
    rp   = args[3]

    r = rr[rr>0]

    if kind == 'axial':         # axial uniform field in the z direction
        l = 1
        out = np.zeros_like(r)

    elif kind == 'dipole' and ricb > 0 :      # dipole, singular at r=0
        l = 1
        out = -12*r**(-5+rp)

    elif kind == 'G21 dipole':  # Felix's dipole (Gerick 2021)
        l = 1
        out = (6/10)*r**rp

    elif kind == 'Luo_S1':
        l = 1
        out = -18*r**rp

    elif kind == 'Luo_S2':
        l = 2
        out = 24*r**(1 + rp)*(-296 + 715*r**2)

    elif kind == 'FDM':

        b = findbeta(args)
        x = b*r

        if l>=2:

            if ricb==0:

                k = x<1e-3
                x0 = x[ k]  # small x
                x1 = x[~k]  # the rest

                out = np.zeros_like(r)
                out[ k] = b**3*jl_smx(l,x0,3)*r[k]**rp

                out[~k] = (x1**3*jl(-2 + l,x1,1) + l*x1*jl(-1 + l,x1,0) - x1**2*jl(-1 + l,x1,1) \
                 - 2*l*x1**2*jl(-1 + l,x1,1) - 3*jl(l,x1,0) - 4*l*jl(l,x1,0) - l**2*jl(l,x1,0) \
                 + 3*x1*jl(l,x1,1) + 4*l*x1*jl(l,x1,1) + l**2*x1*jl(l,x1,1))*r[~k]**(-3+rp)

            else:

                out = ((b**3*r**3*jl(-2 + l,b*r,1) + b*l*r*jl(-1 + l,b*r,0) - b**2*r**2*jl(-1 + l,b*r,1) \
                 - 2*b**2*l*r**2*jl(-1 + l,b*r,1) - 3*jl(l,b*r,0) - 4*l*jl(l,b*r,0) - l**2*jl(l,b*r,0) + 3*b*r*jl(l,b*r,1) \
                 + 4*b*l*r*jl(l,b*r,1) + b*l**2*r*jl(l,b*r,1))*nl(-1 + l,b,0) + jl(-1 + l,b,0)*(-(b**3*r**3*nl(-2 + l,b*r,1)) \
                 - b*l*r*nl(-1 + l,b*r,0) + b**2*r**2*nl(-1 + l,b*r,1) + 2*b**2*l*r**2*nl(-1 + l,b*r,1) + 3*nl(l,b*r,0) \
                 + 4*l*nl(l,b*r,0) + l**2*nl(l,b*r,0) - 3*b*r*nl(l,b*r,1) - 4*b*l*r*nl(l,b*r,1) - b*l**2*r*nl(l,b*r,1)))*r**(-3+rp)

        elif l==1:

            if ricb==0:

                k = x<1e-3
                x0 = x[ k]  # small x
                x1 = x[~k]  # the rest

                out = np.zeros_like(r)
                out[ k] = b**3*jl_smx(l,x0,3)*r[k]**rp

                out[~k] = (-2*x1**2*jl(0,x1,1) - 8*jl(1,x1,0) + x1*(8 - x1**2)*jl(1,x1,1))*r[~k]**(-3+rp)

            else:

                out = (-2*b**2*r**2*jl(0,b*r,1)*nl(0,b,0) - 8*jl(1,b*r,0)*nl(0,b,0) + 8*b*r*jl(1,b*r,1)*nl(0,b,0) \
                 - b**3*r**3*jl(1,b*r,1)*nl(0,b,0) + 2*b**2*r**2*jl(0,b,0)*nl(0,b*r,1) + 8*jl(0,b,0)*nl(1,b*r,0) \
                 - 8*b*r*jl(0,b,0)*nl(1,b*r,1) + b**3*r**3*jl(0,b,0)*nl(1,b*r,1))*r**(-3+rp)

    out2 = np.zeros_like(rr)
    out2[rr>0] = out
    if (ricb == 0) and (np.size(rr[rr>0]) == np.size(rr[rr<0])):
        out2[rr<0] = np.flipud(out)*(-1)**(l-1+rp)

    return out2



def chebco_h(args, kind, N, rcmb, tol):
    '''
    Computes the Chebyshev coeffs of the h0 function used to build B0,
    and derivatives, times a power of r.
    '''

    #beta = args[0]  # beta
    #l    = args[1]  # l
    ricb = args[2]  # ricb
    #rx   = args[3]  # power of r

    dx   = args[4]  # derivative order

    i = np.arange(0, N)
    xi = np.cos(np.pi * (i + 0.5) / N)

    if ricb > 0:
        ri = (ricb + (rcmb - ricb) * (xi + 1) / 2.)
    elif ricb == 0 :
        ri = rcmb * xi

    if   dx == 0 :
        fi = h0(ri, kind, args[:4])
    elif dx == 1 :
        fi = h1(ri, kind, args[:4])
    elif dx == 2 :
        fi = h2(ri, kind, args[:4])
    elif dx == 3 :
        fi = h3(ri, kind, args[:4])

    out = sft.dct(fi) / N
    out[0] = out[0] / 2.
    out[np.absolute(out) <= tol] = 0.
    return out



def B0_norm():
    '''
    Returns the normalization constant of the applied magnetic field
    '''

    if par.magnetic == 1:

        ricb = par.ricb
        args = [ par.beta, par.B0_l, ricb, 0 ]
        kind = par.B0

        l = B0_l
        L = l*(l+1)

        if par.cnorm == 'rms_cmb':  # rms of radial magnetic field at the cmb is set to 1

            rk = np.array([1.0])
            out = np.sqrt(2*l+1) / ( l*(l+1) * h0(rk, kind, args) )
            out = out[0]

        elif par.cnorm in ['mag_energy', 'Schmitt2012']:  # total magnetic energy is set to 1 or 2

            N = 240
            i = np.arange(0,N)
            xk = np.cos( (i+0.5)*np.pi/N )  # colocation points, from -1 to 1
            sqx = np.sqrt(1-xk**2)
            rk = 0.5*(1-ricb)*( xk + 1 ) + ricb
            r2 = rk**2

            y0 = h0(rk, kind, args)
            y1 = h1(rk, kind, args)

            f0 = 4*np.pi*L/(2*l+1)
            f1 = (L+1)*y0**2
            f2 = 2*rk*y0*y1
            f3 = r2*y1**2

            integ = (np.pi/N) * ( (1-ricb)/2 ) * np.sum( sqx*f0*( f1+f2+f3 ) )

            if par.cnorm == 'mag_energy':
                out = 1/np.sqrt(integ)
            elif par.cnorm == 'Schmitt2012':
                out = 2/np.sqrt(integ)

        else:

            out = par.cnorm

    else:

        out = 0

    return out



def Dlam(derivative_order: int,
         truncation_order: int) \
        -> ss.csr_matrix:
    '''
    Returns the :math:`\\mathbf{\\mathcal D}_\\lambda` matrix of derivatives in Gegenbauer space. Direct
    implementation of its definition in Olver and Townsend (2013). The final if-else clause takes care of the fact
    that Gegenbauer polynomials are evaluated at a coordinate x that is a function of r, and therefore the chain rule needs to be applied.

    :param derivative_order: The order :math:`\\lambda` of the derivative.
    :param truncation_order: The maximum order of the expansion of the result. Sets the size of the operator.

    '''

    truncation_order = truncation_order - 1  # This line is here because I (Jorge) use "truncation order" to mean something slightly different from what the code means by "N".

    if derivative_order < 0:
        raise ValueError('(derivative_operator): Negative derivative order. Invalid.')

    elif derivative_order > truncation_order:
        warn('(derivative_operator): Derivative order higher than truncation order. Result is the zero array.')
        to_return = np.zeros([truncation_order+1, truncation_order+1])

    elif derivative_order == 0:
        to_return = np.eye(truncation_order+1)

    else:
        diagonal = np.array(range(derivative_order, truncation_order + 1))
        to_return = ss.diags(diagonal, offsets=derivative_order).A
        to_return = 2.0 ** (derivative_order - 1) * scsp.factorial(derivative_order - 1) * to_return

    if par.ricb == 0:
        to_return = to_return * (1/rcmb)**derivative_order  # ok when rcmb is not 1
    else:
        to_return = to_return * (2./(rcmb-par.ricb))**derivative_order

    return ss.csr_matrix(to_return)



def Slam(basis_index: int,
         truncation_order: int) \
        -> ss.csr_matrix:
    '''
    Direct implementation of :math:`\\mathbf{\\mathcal{S}}_\\lambda` from Olver and Townsend (2013). Pre-multiplies a
    vector of coefficients of a function in the :math:`C^{(\\lambda)}` and returns the vector of coefficients in the
    :math:`C^{(\\lambda+1)}` basis.

    :param basis_index: The Gegenbauer family index :math;`\\lambda`.
    :param truncation_order: The maximum order of the expansion of the result. Sets the size of the operator.

    '''

    truncation_order = truncation_order - 1 # This line is here because I (Jorge) use "truncation order" to mean something slightly different from what the code means by "N".

    if basis_index < 0:
        raise ValueError('(rotation_operator): Negative basis order. Invalid.')

    if basis_index == 0:

        diagonal0 = 0.5 * np.ones(truncation_order + 1)
        diagonal0[0] = 1.0
        diagonal2 = -0.5 * np.ones(truncation_order - 1)

    else:

        diagonal0 = basis_index / (basis_index + np.array(range(truncation_order + 1)))
        diagonal2 = -basis_index / (basis_index + np.array(range(2, truncation_order + 1)))

    return ss.diags([diagonal0, diagonal2], offsets=[0, 2], format = 'csr')



def starting_multiplication_coefficient(basis_index: int,
                                        subindex: int,
                                        idx1: int,
                                        idx2: int) \
        -> float:
    '''
    Computes the starting :math:`c_s^\\lambda(j,k)` coefficient for the matrix operator entry. This is the direct
    implementation of Eq.(3.9) from Olver and Townsend (2013). Note that what is called :math:`j` in this function
    according to the definition of :math:`c_s^\\lambda(j,k)` corresponds to the **column** of the matrix,
    confusingly denoted :math:`k` outside of this function

    :param basis_index: The basis index, :math:`\\lambda`.
    :param subindex: The :math:`s` subindex.
    :param idx1: The :math:`j` index.
    :param idx2: The :math:`k` index.
    '''

    to_return = (idx1 + idx2 + basis_index - 2*subindex) / (idx1 + idx2 + basis_index - subindex)

    for t in range(subindex):
        to_return = to_return * (basis_index+t) / (1+t)
        to_return = to_return * (2*basis_index+idx1+idx2-2*subindex+t) / (basis_index+idx1+idx2-2*subindex+t)

    for t in range(idx1-subindex):
        to_return = to_return * (basis_index+t) / (1+t)
        to_return = to_return * (idx2-subindex+1+t) / (idx2-subindex+basis_index+t)

    return to_return



def multiplication_coefficients(basis_index: int,
                                range_of_subindex: np.ndarray,
                                idx1: int,
                                range_of_idx2: np.ndarray) \
        -> np.ndarray:
    '''
    Computes all the coefficients :math:`c_s^\\lambda(j,k)` for the matrix operator entry. This is the direct
    implementation of the recursion right below Eq.(3.9) from Olver and Townsend (2013). Note that what is called
    :math:`j` in this function according to the definition of :math:`c_s^\\lambda(j,k)` corresponds to the **column** of
    the matrix, confusingly denoted :math:`k` outside of this function

    :param basis_index: The basis index, :math:`\\lambda`.
    :param range_of_subindex: All :math:`s` involved in the sum.
    :param idx1: The :math:`j` involved in the sum
    :param range_of_idx2: All :math:`k` involved in the sum.
    '''

    to_return = np.zeros(len(range_of_subindex))
    to_return[0] = starting_multiplication_coefficient(basis_index,
                                                       int(range_of_subindex[0]),
                                                       idx1,
                                                       int(range_of_idx2[0]))

    for t in range(1, len(range_of_subindex)):
        s = range_of_subindex[t-1]
        idx2 = range_of_idx2[t-1]

        factor = (idx1 + idx2 + basis_index - s) / (idx1 + idx2 + basis_index - s + 1)
        factor = factor * (basis_index + s) / (s + 1)
        factor = factor * (idx1 - s) / (basis_index + idx1 - s - 1)
        factor = factor * (2 * basis_index + idx1 + idx2 - s) / (basis_index + idx1 + idx2 - s)
        factor = factor * (idx2 - s + basis_index) / (idx2 - s + 1)

        to_return[t] = to_return[t - 1] * factor

    return to_return



def Mlam(coefficients: np.ndarray,
         basis_index: int,
         vector_parity: int,
         truncation_order: Union[int|None] = None) \
        -> Union[np.ndarray|ss.csr_array]:
    '''
    This function computes the :math:`\\mathcal{\\mathbf{M}}_\\lambda` operator, as described by Olver and Townsend (
    2013).

    - If :math:`\\lambda = 0`, the operator reduces to a Toeplitz + almost Hankel operators. This is the direct implementation of the equation right below Eq.(2.7) of Olver and Townsend (2013).
    - If :math:`\\lambda = 1`, the operator reduces to a Toeplitz + Hankel operators. However, it can be also be written as the sum of Toeplitz operators, each one adding to the previous outside of the first row and column. This is what has been done in this implementation.
    - Otherwise, each term in the operator is given by the expression between Eq.(3.7) and Eq.(3.8) of Olver and Townsend (2013).

    The present implementation also considers the parity of all elements. In the application of interest,
    this operator will be pre-multiplying the :math:`\\lambda`-th derivative of an eigenvector, so the parity of (a)
    the eigenvector itself, (b) the derivative and (c) the function represented in this multiplication operator are
    all taken into account to set certain rows and columns to zero.

    In the cases for :math:`\\lambda = 0,1`, this has been done by creating the full operators with the Scipy
    Toeplitz and Hankel functions, and then setting the appropriate rows and columns to 0. In the general case,
    where the operator is to be filled element by element, the useful elements have been identified at the begining
    and only those have been computed, skipping all the rest.

    :param basis_index: The index of the Gegenbauer family, :math:`\\lambda`.
    :param coefficients: The :math:`\\lambda`-th Gegenabuer coefficients of the factor represented in
    :math:`\\mathbf{\\mnathcal{M}[a]}, :math:`a_k`.
    :param vector_parity: The parity of the eigenvector.
    :param truncation_order: The truncation order, setting the size of the operator. If ``None``, it infers the order
    from the size of ``coefficients``. Defaults to None.

    '''

    if basis_index < 0:
        raise ValueError('(multiplication_operator): Negative basis order. Invalid.')

    if vector_parity not in [-1, 0, 1]:
        raise ValueError('(multiplication_operator): Invalid vector parity. Only allowed is 0, 1 or -1. Got '
                         + str(vector_parity) + '.')

    if truncation_order is None:
        truncation_order = len(coefficients)-1 # There are no issues with "truncation order vs. N" here because it is all
        # handled internally and there is no interfacing with the outside.

    if sum(abs(coefficients)) == 0.0:

        to_return = np.zeros([truncation_order+1, truncation_order+1])

    else:

        # The parities of all individual elements. The parity of the factor represented in this multiplication
        # operator is assessed using (the parity of) its last non-zero coefficient.
        last_nonzero = np.nonzero(coefficients)[0][-1]
        derivative_parity = int(1 - 2 * (basis_index % 2))
        function_parity = int(1 - 2 * (last_nonzero % 2))
        operator_parity = vector_parity * derivative_parity * function_parity

        if basis_index == 0:

            # Auxiliary padding. Always good to have enough zeros in case they are needed.
            coefficients = np.pad(coefficients,(0, int(2 * truncation_order + 1) - len(coefficients)))

            # Toeplitz matrix
            rowcol = coefficients[:truncation_order + 1]
            rowcol[0] = 2.0 * rowcol[0]
            toeplitz = las.toeplitz(rowcol)

            # Hankel matrix
            col = coefficients[:truncation_order + 1]
            row = coefficients[truncation_order:int(2 * truncation_order + 1)]
            hankel = las.hankel(col, row)
            hankel[0, :] = np.zeros(truncation_order + 1)

            to_return = 0.5 * (toeplitz + hankel)

        elif basis_index == 1:

            # Auxiliary padding. Always good to have enough zeros in case they are needed.
            coefficients = np.pad(coefficients, (0, int(2 * truncation_order + 1) - len(coefficients)))
            to_return = las.toeplitz(coefficients[:truncation_order+1])
            for idx in range(1,truncation_order+1):
                to_return[idx:,idx:] = to_return[idx:,idx:] + las.toeplitz(coefficients[2*idx:truncation_order+idx+1])

        else:

            # In the general case, parity is taken care of at the very beginning.
            if vector_parity * derivative_parity == 1:
                column_range = range(0, truncation_order + 1, 2)
            elif vector_parity * derivative_parity == -1:
                column_range = range(1, truncation_order + 1, 2)
            else:
                column_range = range(truncation_order + 1)

            if operator_parity == 1:
                row_range = range(0, truncation_order + 1, 2)
            elif vector_parity == -1:
                row_range = range(1, truncation_order + 1, 2)
            else:
                row_range = range(truncation_order + 1)

            coefficients = coefficients[:last_nonzero+1]

            # Now we loop over all useful entries of the operator.
            to_return = np.zeros([truncation_order+1,truncation_order+1])
            for row in row_range:
                for col in column_range:

                    # Range of s as defined by Olver and Townsend (2013). Between Eq.(3.7) and Eq.(3.8).
                    range_of_s = np.array(range(max(0,col-row),col+1))
                    range_of_p = 2 * range_of_s + row - col
                    # The range of those indices are truncated according to the maximum "significant" order of the
                    # provided factor coefficients.
                    range_of_s = range_of_s[range_of_p < len(coefficients)]
                    range_of_p = range_of_p[range_of_p < len(coefficients)]

                    if len(range_of_s) == 0:
                        to_return[row, col] = 0.0

                    else:
                        array_of_multiplication_coefficients = multiplication_coefficients(basis_index,
                                                                                           range_of_s,
                                                                                           col,
                                                                                           range_of_p)

                        to_return[row,col] = np.sum(coefficients[range_of_p] * array_of_multiplication_coefficients)

        if (basis_index == 0 or basis_index == 1) and (vector_parity != 0):

            # If the Toeplitz + Hankel simplifications were used, the parity is taken care of at the end by removing
            # all unnecessary rows/columns.

            if vector_parity * derivative_parity == 1:
                columns_to_remove = np.array(range(1, truncation_order + 1, 2)).astype(int)
            else:
                columns_to_remove = np.array(range(0, truncation_order + 1, 2)).astype(int)

            if operator_parity == 1:
                rows_to_remove = np.array(range(1, truncation_order + 1, 2)).astype(int)
            else:
                rows_to_remove = np.array(range(0, truncation_order + 1, 2)).astype(int)

            to_return[rows_to_remove, :] = np.zeros([len(rows_to_remove), truncation_order + 1])
            to_return[:, columns_to_remove] = np.zeros([truncation_order + 1, len(columns_to_remove)])

    return ss.csr_matrix(to_return)



def chebProduct(ck,dk,tol):
    '''
    Computes the Chebyshev expansion of a product of
    two Chebyshev series defined by ck and dk
    '''
    out = np.zeros_like(ck)
    out = Mlam(ck,0,0)*dk
    out[np.absolute(out) <= tol] = 0.

    return out



def marc_tide(omega, l, m, loc, N, ricb, rcmb):
    '''
    Tidal body force as used by Rovira-Navarro et al, 2018
    '''

    C0 = 1j; C1 = 1; C2 = 1j

    if l==2 and loc == 'top':
        r5 = chebco(5, N-4, 2e-16, ricb, rcmb)

        if m == 0:
            out = -6j*C0*omega*r5
        if m == 1:
            out = -6j*C1*(omega+1)*r5
        if m == 2:
            out = -6j*C2*(omega+2)*r5

    elif loc == 'bot':
        r4 = chebco(4, N-2, 2e-16, ricb, rcmb)

        if m == 0:
            if l == 1:
                out = (4/5)*C0*r4
            elif l == 3:
                out = (-24/5)*C0*r4

        elif m == 1:
            if l == 1:
                out = (2/5)*np.sqrt(3)*C1*r4
            elif l == 3:
                out = -(16/5)*np.sqrt(2)*C1*r4

        elif m == 2:
            if l == 3:
                out = -(8/np.sqrt(5))*C2*r4

    return out



def eccen_tide(m, eta, boundary):
    '''
    Eccentricity tide as boundary forcing.
    Computed by Jeremy
    eta = ricb
    '''

    if boundary == 'cmb':

        if m == -2:

            P = (3.00927601455473e-7 + 9.923006738722803e-7*(-0.8809523809523809 + eta) + 2.0022902696912132e-6*(-0.8809523809523809 + eta)**2 \
            + 2.078150971132207e-6*(-0.8809523809523809 + eta)**3)/(0.9796806966104893 + 10.138063592395676*(-0.8809523809523809 + eta) \
            + 22.793605369253733*(-0.8809523809523809 + eta)**2 + 49.52238501116611*(-0.8809523809523809 + eta)**3)

            dP = (3.6923409583928235e-8 + 5.525597059314317e-7*(-0.8809523809523809 + eta) + 1.491072486300637e-6*(-0.8809523809523809 + eta)**2 \
            + 1.7131360325476716e-6*(-0.8809523809523809 + eta)**3)/(0.990016066975477 + 10.903688391808103*(-0.8809523809523809 + eta) \
            + 28.973467794986213*(-0.8809523809523809 + eta)**2 + 54.35051022571307*(-0.8809523809523809 + eta)**3)

        elif m == 0:

            P = (-1.0530272472650882e-7 - 3.4723290320109935e-7*(-0.8809523809523809 + eta) - 7.006556396692487e-7*(-0.8809523809523809 + eta)**2 \
            - 7.272013553918712e-7*(-0.8809523809523809 + eta)**3)/(0.9796806966105374 + 10.138063592396957*(-0.8809523809523809 + eta) \
            + 22.793605369262107*(-0.8809523809523809 + eta)**2 + 49.52238501117491*(-0.8809523809523809 + eta)**3)

            dP = (-1.2920501863486681e-8 - 1.933556188506115e-7*(-0.8809523809523809 + eta) - 5.217666801343201e-7*(-0.8809523809523809 + eta)**2 \
            - 5.994727342457128e-7*(-0.8809523809523809 + eta)**3)/(0.990016066975396 + 10.903688391807878*(-0.8809523809523809 + eta) \
            + 28.973467794991336*(-0.8809523809523809 + eta)**2 + 54.35051022572941*(-0.8809523809523809 + eta)**3)

        elif m == 2:

            P = (-4.2989657350784934e-8 - 1.4175723912463045e-7*(-0.8809523809523809 + eta) - 2.860414670987557e-7*(-0.8809523809523809 + eta)**2 \
            - 2.9687871016172726e-7*(-0.8809523809523809 + eta)**3)/(0.9796806966105591 + 10.1380635923966*(-0.8809523809523809 + eta) \
            + 22.793605369256564*(-0.8809523809523809 + eta)**2 + 49.522385011165795*(-0.8809523809523809 + eta)**3)

            dP = (-5.274772797703058e-9 - 7.893710084733917e-8*(-0.8809523809523809 + eta) - 2.1301035518586083e-7*(-0.8809523809523809 + eta)**2 \
            - 2.4473371893559314e-7*(-0.8809523809523809 + eta)**3)/(0.990016066975294 + 10.903688391807277*(-0.8809523809523809 + eta) \
            + 28.973467794994058*(-0.8809523809523809 + eta)**2 + 54.35051022573984*(-0.8809523809523809 + eta)**3)

    elif boundary == 'icb':

        if m == -2:

            P = (1.450645275222846e-9 + 1.6226580438374377e-8*(-0.8809523809523809 + eta) + 3.8162533600883e-8*(-0.8809523809523809 + eta)**2 \
            + 2.1621482543044536e-8*(-0.8809523809523809 + eta)**3)/(1.0040474189168584 + 9.816014125318514*(-0.8809523809523809 + eta) \
            + 12.467747302017761*(-0.8809523809523809 + eta)**2 - 0.13326826070343542*(-0.8809523809523809 + eta)**3)

            dP = (5.690951704457902e-11 + 6.365766130462465e-10*(-0.8809523809523809 + eta) + 1.4971346844873185e-9*(-0.8809523809523809 + eta)**2 \
            + 8.482212366654875e-10*(-0.8809523809523809 + eta)**3)/(1.0040474189168667 + 9.816014125318599*(-0.8809523809523809 + eta) \
            + 12.467747302017914*(-0.8809523809523809 + eta)**2 - 0.13326826070340814*(-0.8809523809523809 + eta)**3)

        elif m == 0:

            P = (-5.076201031536069e-10 - 5.678120334892032e-9*(-0.8809523809523809 + eta) - 1.3354104944854432e-8*(-0.8809523809523809 + eta)**2 \
            - 7.565942816136093e-9*(-0.8809523809523809 + eta)**3)/(1.0040474189168602 + 9.81601412531852*(-0.8809523809523809 + eta) \
            + 12.467747302017683*(-0.8809523809523809 + eta)**2 - 0.13326826070345546*(-0.8809523809523809 + eta)**3)

            dP = (-1.991418260963427e-11 - 2.22755412021775e-10*(-0.8809523809523809 + eta) - 5.238880076023669e-10*(-0.8809523809523809 + eta)**2 \
            - 2.96815602688989e-10*(-0.8809523809523809 + eta)**3)/(1.0040474189168587 + 9.816014125318501*(-0.8809523809523809 + eta) \
            + 12.46774730201766*(-0.8809523809523809 + eta)**2 - 0.13326826070345907*(-0.8809523809523809 + eta)**3)

        elif m == 2:

            P = (-2.072350393175482e-10 - 2.3180829197677505e-9*(-0.8809523809523809 + eta) - 5.451790514411792e-9*(-0.8809523809523809 + eta)**2 \
            - 3.0887832204348696e-9*(-0.8809523809523809 + eta)**3)/(1.0040474189168525 + 9.816014125318436*(-0.8809523809523809 + eta) \
            + 12.46774730201756*(-0.8809523809523809 + eta)**2 - 0.13326826070347508*(-0.8809523809523809 + eta)**3)

            dP = (-8.129931006368303e-12 - 9.093951614946213e-11*(-0.8809523809523809 + eta) - 2.138763834981828e-10*(-0.8809523809523809 + eta)**2 \
            - 1.2117446238077878e-10*(-0.8809523809523809 + eta)**3)/(1.0040474189168571 + 9.816014125318477*(-0.8809523809523809 + eta) \
            + 12.467747302017562*(-0.8809523809523809 + eta)**2 - 0.13326826070347858*(-0.8809523809523809 + eta)**3)

    return np.array([P,dP])



def ftest1(ricb):
    '''
    Tidal body force constants for Enceladus
    Computed by Jeremy
    '''

    A = (2.776146590187793e-8 - 1.1936189474389837e-7*ricb + 1.7976942241610017e-7*ricb**2 - 1.0431683766634872e-7*ricb**3 \
    + 1.0203567472726217e-8*ricb**4 + 5.944217795551174e-9*ricb**5)/(2.039370654683117 - 12.426922103588517*ricb \
    + 29.534185532382605*ricb**2 - 33.58608820683908*ricb**3 + 17.080450537622824*ricb**4 - 1.6409964140402016*ricb**5 - 1.*ricb**6)

    B = (7.525664633338715e-9 - 5.8895735957722854e-8*ricb + 1.8268212761498169e-7*ricb**2 - 2.798206106175817e-7*ricb**3 \
    + 2.1103777701038198e-7*ricb**4 - 6.252914009737955e-8*ricb**5)/(2.039370654683117 - 12.426922103588517*ricb \
    + 29.534185532382605*ricb**2 - 33.58608820683908*ricb**3 + 17.080450537622824*ricb**4 - 1.6409964140402016*ricb**5 - 1.*ricb**6)

    C = 1e6

    #A = 1.0
    #B = ricb**5
    #C = 1/(1-ricb**5)

    return np.array([A,B,C])



def Ylm(l, m, theta, phi):
    # The Spherical Harmonics, seminormalized
    out = scsp.sph_harm(m, l, phi, theta)
    return out*np.sqrt(4*np.pi/(2*l+1))



def Ylm_full(lmax, m, theta, phi):
    # array of Spherical Harmonics with a range of l's
    m1 = max(m,1)
    if m == 0 :
        lmax1 = lmax+1
    else:
        lmax1 = lmax
    out = np.zeros(lmax-m+1,dtype=np.complex128)
    for l in np.arange(m1,lmax1+1):
        out[l-m1]=Ylm(l,m,theta,phi)
    return out



def Ylm_symm(lmax, m, theta, phi, symm, scalar):
    out = np.zeros((lmax-m+1)/2,dtype=np.complex128)
    if (symm == 1 and scalar == 'pol') or (symm == -1 and scalar == 'tor'):
        m_sym = m
        lmax_sym = lmax
    else :
        m_sym = m+1
        lmax_sym = lmax+1
    for l in np.arange(m_sym,lmax_sym,2.):
        out[(l-m_sym)/2]=Ylm(l,m,theta,phi)
    return out



def load_csr(filename):
    # utility to load sparse matrices efficiently
    loader = np.load(filename)
    return ss.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])




def Tk(x, N, lamb_max):
    '''
    Chebyshev polynomial from order 0 to N (as rows)
    and its derivatives ** with respect to r **, up to lamb_max (as columns),
    evaluated at x=-1 (r=ricb) or x=1 (r=rcmb).
    '''

    if par.ricb == 0 :
        ric = -rcmb
    else :
        ric = par.ricb

    out = np.zeros((N+1,lamb_max+1))

    for k in range(0,N+1):

        out[k,0] = x**k

        tmp = 1.
        for i in range(0, lamb_max):
            tmp = tmp * ( k**2 - i**2 )/( 2.*i + 1. )
            out[k,i+1] = x**(k+i+1.) * tmp * (2./(rcmb-ric))**(i+1)

    return out



def gamma_visc(a1,a2,a3):

    out = np.zeros((1,n0+n0),dtype=complex)


    #Tb3 = copy(Tb2)
    #Tb3[nc:,:] = 0
    #Tb3 = np.copy(bv.Tb)
    Tb3 = Tk( 1, par.N-1, 5)


    P0 = Tb3[:,0]
    P1 = Tb3[:,1]
    P2 = Tb3[:,2]
    P3 = Tb3[:,3]
    P4 = Tb3[:,4]
    P5 = Tb3[:,5]

    T0 = Tb3[:,0]
    T1 = Tb3[:,1]
    T2 = Tb3[:,2]
    T3 = Tb3[:,3]
    T4 = Tb3[:,4]


    # this is for a spheroid with long semiaxis a=1

    I = 1j

    pol2 = (  a1*((0. + 24.624956739107787*I)*P0 - (0. + 24.624956739107787*I)*P1 + (0. + 12.312478369553894 *I)*P2
                + (0. + 4.104159456517965 *I)*P3)
            + a2*((0. + 5.2767764440945255*I)*P0 - (0. + 5.2767764440945255*I)*P1 + (0. + 2.6383882220472628 *I)*P2
                - (0. + 9.67409014750663  *I)*P3 - (0. + 1.758925481364842 *I)*P4)
            + a3*((0. - 1.758925481364842 *I)*P0 + (0. + 1.758925481364842 *I)*P1 - (0. + 0.879462740682421  *I)*P2
                + (0. + 0.2931542468941404*I)*P3 + (0. + 3.517850962729684 *I)*P4 + (0. + 0.4885904114902339 *I)*P5) )

    pol4 = (  a2*((0. - 42.817918360629186*I)*P0 + (0. + 42.817918360629186*I)*P1 + (0. + 3.5681598633857656 *I)*P2
                - (0. + 6.422687754094378 *I)*P3 - (0. + 0.7136319726771531*I)*P4)
            + a3*((0. - 31.140304262275777*I)*P0 + (0. + 31.140304262275777*I)*P1 - (0. + 20.111446502719772 *I)*P2
                + (0. + 2.465274087430165 *I)*P3 + (0. + 3.6979111311452484*I)*P4 + (0. + 0.324378169398706  *I)*P5))

    pol6 = (  a3*((0. + 45.56049760657571*I)*P0 - (0. + 45.56049760657571  *I)*P1 - (0. + 22.780248803287854 *I)*P2
                + (0. + 3.254321257612551*I)*P3 + (0. + 1.30172850304502   *I)*P4 + (0. + 0.07231825016916779*I)*P5))

    tol1 = ( -11.847687835088976     *T0 + 11.847687835088976*T1
            + a1*(9.47815026807118   *T0 - 9.47815026807118  *T1 - 4.73907513403559  *T2)
            + a2*(2.031032200300967  *T0 - 2.031032200300967 *T1 + 5.077580500752418 *T2 + 1.5232741502257257 *T3)
            + a3*(0.4513404889557705 *T0 - 0.4513404889557705*T1 - 1.8241678095295728*T3 - 0.37611707412980877*T4))

    tol3 = (  a1*(39.799940335196546 *T0 - 6.633323389199425 *T1 - 3.3166616945997127*T2)
            + a2*(-19.899970167598273*T0 - 13.26664677839885 *T1 + 8.291654236499282 *T2 + 1.6583308472998564 *T3)
            + a3*(-14.472705576435107*T0 + 7.93988708707204  *T1 + 2.010097996727099 *T2 - 3.5679239441906003 *T3 - 0.5025244991817748  *T4))

    tol5 = (  a2*(-57.208391908193846*T0 - 9.534731984698974 *T1 + 3.8138927938795897*T2 + 0.4767365992349487 *T3)
            + a3*(4.400645531399526  *T0 + 34.960683943896235*T1 + 0.6845448604399262*T2 - 2.725955426394706  *T3 - 0.24448030729997366 *T4))

    tol7 = (  a3*(59.85704517989213  *T0 + 24.316924604331177*T1 - 0.8016568550878411*T2 - 0.7571203631385165 *T3 - 0.044536491949324505*T4))


    # assemble the torque (row vector)

    for l in np.arange(m_top,lmax_top,2.):

        colP = int( (par.N)*(l-m_top)/2 )

        if l==2 and par.m==1:
            out[0,colP:colP+par.N] = pol2
        elif l==4 and par.m==1:
            out[0,colP:colP+par.N] = pol4
        elif l==6 and par.m==1:
            out[0,colP:colP+par.N] = pol6

    for l in np.arange(m_bot,lmax_bot,2.):

        colT = n0 + int( (par.N)*(l-m_bot)/2 )

        if l==1 and par.m==1:
            out[0,colT:colT+par.N] = tol1
        elif l==3 and par.m==1:
            out[0,colT:colT+par.N] = tol3
        elif l==5 and par.m==1:
            out[0,colT:colT+par.N] = tol5
        elif l==7 and par.m==1:
            out[0,colT:colT+par.N] = tol7

    # axial torque for a spherical cmb, take 2*real after multiplying by the solution vector
    if par.m == 0 and par.symm == 1:
        R = 1  #rcmb
        # axial torque depends on the l=1, m=0 toroidal component only
        out[0,n0:n0+par.N] = (8*np.pi/3)*(R**2)*( R*T1 - T0 )

    return out



def gamma_visc_icb(ricb):
    '''
    Axial viscous torque on the inner core, spherical. Take 2*real after multiplying by the solution vector
    '''

    out = np.zeros((1,n0+n0),dtype=complex)

    if par.m == 0 and par.symm == 1 and par.ricb > 0:

        T = Tk( -1, par.N-1, 1)
        T0 = T[:,0]
        T1 = T[:,1]
        R = ricb
        # axial torque depends on the l=1, m=0 toroidal component only
        out[0,n0:n0+par.N] = (8*np.pi/3)*(R**2)*( R*T1 - T0 )

    return out



def gamma_magnetic():
    '''
    Axial magnetic torque on the mantle (spherical) when there is a thin conductive layer at bottom. Needs m=0 and symm=1.
    '''

    if (par.magnetic==1 and par.m == 0 and par.symm==1 and par.mantle=='TWA'):

        out = np.zeros((1,n0+n0),dtype=complex)
        G = Tk( 1, par.N-1, 0)[:,0]
        R = np.array([1.0])
        h_cmb = B0_norm() * h0(R, par.B0, [par.beta, par.B0_l, par.ricb, 0])

        if B0_l == 1:  # Either uniform axial or dipole background field, induced magnetic field b is thus antisymmetric

            # the torque is prop. to the l=2 toroidal component of b
            out[0,n0:n0+par.N] = (16*np.pi/5) * G * h_cmb

        elif B0_l == 2:  # Quadrupole background field, induced magnetic field b is thus symmetric

            # torque prop. to l=1 and l=3 toroidal component of b
            out[0,n0:n0+par.N]          = -(16*np.pi/5)     * G * h_cmb # l=1
            out[0,n0+par.N: n0+2*par.N] =  (16*18*np.pi/35) * G * h_cmb # l=3

    else:

        out = 0

    # Take the product between the output of this function and the solution for b to obtain the dimensionless torque
    # Then multiply by Elsasser*R_cmb^3*rho*eta to make the torque dimensional
    # (rho is the density and eta is the magnetic diffusivity, both of the fluid core).

    return out

