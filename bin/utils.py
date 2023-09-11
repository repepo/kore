from scipy.linalg import toeplitz
from scipy.linalg import hankel
import scipy.optimize as so
import scipy.sparse as ss
import scipy.special as scsp
import scipy.fftpack as sft
import scipy.misc as sm
import numpy.polynomial.chebyshev as ch
import numpy as np

import parameters as par

'''
Various function definitions and utilities
'''

# First some global variables: -----------------------------------------------------------------------------------------

if par.forcing == 0:
    wf = 0
else:
    wf = par.forcing_frequency

rcmb   = 1

N1     = int(par.N*(0.5 + 0.5*np.sign(par.ricb)))   # N/2 if no IC, N if present
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

# ----------------------------------------------------------------------------------------------------------------------



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



def twozone(r,args):
    '''
    Symmetrized dT/dr (dimensionless), extended to negative r
    rc is the transition radius, h the transition width, sym is 1 or -1
    depending on the radial parity desired
    ** Neutrally buoyant inner zone, stratified outer zone ** (Vidal2015)
    '''
    rc  = args[0]
    h   = args[1]
    sym = args[2]

    out = np.zeros_like(r)
    for i,x in enumerate(r):
        if x >= 0 :
            out[i] = (1 + np.tanh( 2*(x-rc)/h  ))/2
        elif x < 0 :
            out[i] = sym*(1 + np.tanh( 2*(abs(x)-rc)/h  ))/2
    return out



def BVprof(r,args):
    '''
    Symmetrized dT/dr (dimensionless), extended to negative r.
    Define this function so that it is an odd function of r
    '''
    out = np.zeros_like(r)
    for i,x in enumerate(r):
        out[i] = x         # dT/dr propto r like in Dintrans1999
        #out[i] = x*abs(x)  # dT/dr propto r^2
        #out[i] = x**3      # dT/dr propto r^3
        #rc = args[0]
        #h  = args[1]
        #if abs(x) < rc/2 :
        #    out[i] = np.tanh( 4*x/h )
        #elif x >= rc/2 :
        #    out[i] = 0.5*(1 - np.tanh( 4*(x-rc)/h  ))
        #elif x <= -rc/2 :
        #    out[i] = -0.5*(1 - np.tanh( 4*(abs(x)-rc)/h  ))
    return out



def conductivity(r):
    '''
    This function needs to be an even function of r when ricb=0
    '''
    return np.ones_like(r)



def mag_diffus(r):
    return 1./conductivity(r)



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



def chebco_f(func,rpower,N,ricb,rcmb,tol,args=None):
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

    tmp = sft.dct(ri**rpower * func(ri))

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



def chebco_twozone(args, N, ricb, rcmb, tol):
    '''
    Returns the first N Chebyshev coefficients
    from 0 to N-1, of the function
    r * twozone(r,rc,h,sym)
    where rc is the radius of the neutral core
    and h is the transition thickness
    sym is the desired parity, in case of no inner core
    '''
    i = np.arange(0, N)
    xi = np.cos(np.pi * (i + 0.5) / N)

    if ricb > 0:
        ri = (ricb + (rcmb - ricb) * (xi + 1) / 2.)
    elif ricb == 0 :
        ri = rcmb * xi
    #args = [rc, h, sym]
    fi = ri * twozone(ri,args)  # the ri here comes from r^2 u_r = r^2*[ l(l+1) P/r ]

    tmp = sft.dct(fi)

    out = tmp / N
    out[0] = out[0] / 2.
    out[np.absolute(out) <= tol] = 0.
    return out



def chebco_BVprof(args, N, ricb, rcmb, tol):
    '''
    Returns the first N Chebyshev coefficients
    from 0 to N-1, of the function
    r * BVprof(r,args)
    '''
    i = np.arange(0, N)
    xi = np.cos(np.pi * (i + 0.5) / N)

    if ricb > 0:
        ri = (ricb + (rcmb - ricb) * (xi + 1) / 2.)
    elif ricb == 0 :
        ri = rcmb * xi
    fi = ri * BVprof(ri,args)  # this function should be even, i.e., BVprof should be odd.

    tmp = sft.dct(fi)

    out = tmp / N
    out[0] = out[0] / 2.
    out[np.absolute(out) <= tol] = 0.
    return out



def chebco_h(args, kind, N, rcmb, tol):
    '''
    Computes the Chebyshev coeffs of the h0 function and derivatives
    times a power of r
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



def Dlam(lamb,N):
    '''
    Order lamb (>=1) derivative matrix, size N*N
    '''
    if par.ricb == 0:
        const1 = (1/rcmb)**lamb  # ok when rcmb is not 1
    else:
        const1 = (2./(rcmb-par.ricb))**lamb
    const2 = scsp.factorial(lamb-1.)*2**(lamb-1.)
    tmp = lamb + np.arange(0,N-lamb)

    return const1*const2*ss.diags(tmp,lamb, format='csr')



def Slam(lamb,N):
    '''
    Converts C^(lamb) series coefficients to C^(lamb+1) series
    '''

    if lamb == 0:
        diag0 = 0.5*np.ones(N); diag0[0]=1.
        diag1 = -0.5*np.ones(N-2)
    else:
        tmp = np.arange(0.,N)
        diag0 = lamb/(lamb+tmp)
        diag1 = -lamb/(lamb+tmp[2:])

    return ss.diags([diag0,diag1],[0,2], format='csr')



def csl0( s, lamb, j, k):
    '''
    Computes the c_s^lambda(j,k) needed for the Mlam (multiplication) matrix
    '''

    p1=1; p3=1
    for t in range(0,s):
        p1 = p1*(lamb+t)/float(1+t)
        p3 = p3*(2*lamb+j+k-2*s+t)/float(lamb+j+k-2*s+t)

    p2=1; p4=1
    for t in range(0,j-s):
        p2 = p2*(lamb+t)/float(1+t)
        p4 = p4*(k-s+1+t)/float(k-s+lamb+t)

    return p1*p2*p3*p4*(j+k+lamb-2.*s)/float(j+k+lamb-s)



def csl(svec,lamb,j,k):
    '''
    recursion for c_s^lambda, starting from c_svec[0]^lambda(j,k)
    svec must be a vector of s values
    **do not confuse with the (j,k) entry of the Mlam matrix**
    '''
    out = np.zeros(np.shape(svec))
    out[0] = csl0(svec[0], lamb, j, k)
    for i,s in enumerate(svec[0:-1],1) :
        tmp1 = (j+k+lamb-s)*(lamb+s)*(j-s)*(2*lamb+j+k-s)*(k-s+lamb)
        tmp2 = (j+k+lamb-s+1)*(s+1)*(lamb+j-s-1)*(lamb+j+k-s)*(k-s+1)
        out[i] = out[i-1]*tmp1/float(tmp2)
        k=k+2

    return out



def Mlam(a0,lamb,vector_parity):
    '''
    Multiplication matrix. a0 are the cofficients in the C^(lamb) basis and lamb
    is the order of the C^(lamb) basis. (This basis should match the
    one from the highest derivative order appearing in the equation)
    '''

    if np.sum(abs(a0)) > 0 :

        N = np.size(a0)
        bw = max(np.nonzero(a0)[0])

        a1 = np.zeros(2*N)
        a1[:N] = a0

        #if a0.dtype == np.complex128:
        #    print(a0)

        if vector_parity != 0: # no inner core case

            # Overall operator parity given by a0 parity * lambda parity
            # check a0 parity like this: first nonzero a0 coefficient
            # a0 is the full vector of coefficients, including even and odd, size N
            tmp = np.nonzero(a0)[0]
            ix = tmp[-1] # index of *last* non zero coefficient
            rpower_parity = 1 - 2*(ix%2)
            lamb_parity = 1 - 2*(lamb%2)
            operator_parity = rpower_parity * lamb_parity
            overall_parity = vector_parity * operator_parity
            # rows to be deleted determined by overall_parity (after multiplying with DX and the eigenvector)
            # j even when overall_parity = 1 and vice versa
            # columns to be deleted determined by vector_parity (after multiplying with DX and the eigenvector)
            # k even when vector_parity = 1 and vice versa
            idj = int((1-overall_parity)/2)
            idk = int((1-vector_parity*lamb_parity)/2)
            jrange = range(idj,N,2)

        else: # vector_parity = 0, inner core case

            jrange = range(0,N)


        if lamb > 0:

            out = ss.dok_matrix((N,N))
            for j in jrange:

                k1 = max( 0, j-bw-1 )
                k2 = min( N, j+bw+2 )
                ka = range(k1,k2)

                if vector_parity != 0:
                    krange = ka[ka[idk]%2::2]
                else:
                    krange = ka

                for k in krange:

                    s0 = max(0,k-j)
                    s = np.arange(s0,k+1)
                    idx = 2*s+j-k
                    a = a1[idx]

                    if s0 == 0:
                        cvec = csl(s,lamb,k,j-k)
                    elif s0 == k-j:
                        cvec = csl(s,lamb,k,k-j)

                    out[j,k] = np.dot(a,cvec)

            out = out.tocsr()

        else:

            a2 = np.copy(a0);
            a2[0] = 2*a2[0]
            tmp1 = toeplitz(a2)
            tmp2 = hankel(a2)
            tmp2[0,:]=np.zeros(N)

            tmp = 0.5*(tmp1+tmp2)

            if vector_parity != 0:
                idj0 = int((1+overall_parity)/2)
                idk0 = int((1+vector_parity*lamb_parity)/2)
                for j in range(idj0,N,2):
                    tmp[j,:]=np.zeros(N)
                for k in range(idk0,N,2):
                    tmp[:,k]=np.zeros(N)

            out = ss.csr_matrix(tmp)
        '''
        The case lamb = 1 should be reducible too to a Hankel+Toeplitz
        not done yet
        '''

    else:

        out = 0

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



def Dn_cheb(ck, ricb, rcmb, n):
    '''
    Returns the Chebyshev coefficients of the derivatives (up to order n)
    of a Chebyshev expansion with coefficients ck. Assumes ck is computed
    for r in the domain [ricb,rcmb] (if ricb>0) or r in [-rcmb,rcmb] if ricb=0.
    First column correspond to the first derivative, last column to the
    n-th derivative.
    '''
    c = np.copy(ck)
    s = np.size(c)
    out = np.zeros((s,n), ck.dtype)
    out[:,0] = Dcheb(c, ricb, rcmb)
    for j in range(1,n):
        out[:,j] = Dcheb( out[:, j-1], ricb, rcmb)

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

