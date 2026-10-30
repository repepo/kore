from scipy.linalg import toeplitz
from scipy.linalg import hankel
import scipy.sparse as ss
import scipy.special as scsp
import scipy.fftpack as sft
import scipy.interpolate as si
import numpy.polynomial.chebyshev as ch
import numpy as np
from parameters import par
import radial_profiles as rap
import angular
import background_field as bf
import sys

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

# NEW (2026-10-28): equatorial symmetry of the background field B0 and of the induced field b (magnetic = 1). With an
# antisymmetric B0 (symmB0 = -1, 'axial' or a dipole) b has the symmetry opposite to u, so the poloidal field f has the l's
# of the toroidal flow and g those of the poloidal flow.
symmB0 = bf.symmB0
bsymm  = par.symm * symmB0

# NEW (2026-10-28): a conducting inner core whose field is part of the unknowns (sections fic, gic): Nic Chebyshev
# coefficients of one parity per l on [-ricb, ricb] (N_cic/2), nic per section, the same l's as f, g
cic    = int(bool(par.magnetic and par.innercore == 'conducting, Chebys' and par.ricb > 0))
Nic    = int(par.N_cic/2)
nic    = int(Nic*(par.lmax-par.m+1)/2) * cic

# this gives the size (rows or columns) of the main matrices
sizmat = 2*n + n*par.thermal + 2*n*par.magnetic + 2*nic  # NEW (2026-10-28): + the magnetic field (+ inner core)

# NEW (2026-10-28): first row (and column) of each section in A and B: P, T, s, then f, g after s, then fic, gic, so that
# the flow and the entropy keep their places and non-magnetic matrices are unchanged
offset = { 'u': 0, 'v': n, 'h': 2*n, 'f': (2 + par.thermal)*n, 'g': (3 + par.thermal)*n,
           'fic': (4 + par.thermal)*n, 'gic': (4 + par.thermal)*n + nic }

# ThermaD = 0 with the thermal-wind advection (docs/differential_rotation_details.md §7.2): the section-h rows are r² (eq)/p instead of
# r (eq)/p, in the C^(1) basis (u_θ has P'), still without thermal bc rows; the thermal-wind profiles are then σn, σn'
adiab_tw = bool(par.thermal == 1 and par.ThermaD == 0 and par.diff_rot == 1 and par.thermal_wind >= 1)

# profiles that are odd in r (parity of the operators with ricb = 0): g, p dS/dr, (ln ρ)'^2-type products, dS/dr and the
# derivatives σn' of the thermal-wind entropy (codes tp + degree character, diff_rot.codes)
odd_profiles = ['gra', 'pdS', 'lh1', 'dSd'] + ['tp' + c for c in '0123456789abcdefghijklmnopqrstuvwxyz']
# NEW (2026-10-28): the background field's h (k) of degree n has the parity (-1)^n at ricb = 0
if par.magnetic:
    if par.B0 is not None and bf.B0_l % 2:
        odd_profiles.append('bpo')
    if par.B0_tor is not None and par.B0_tor_l % 2:
        odd_profiles.append('bto')
# NEW (2026-10-30): the magnetic couplings from the recurrence algebra (mag_algebra.py) instead of the closed forms of
# operators.py: with par.mag_couplings = 'algebra' (any B0 type), and always for the types that need it ('general', 'ferraro')
mag_algebra = bool(par.magnetic) and (getattr(par, 'mag_couplings', 'closed') == 'algebra' or par.B0 in ('general', 'ferraro')
                                     or par.B0_tor == 'general')
if mag_algebra:
    import mag_algebra as _ma
    odd_profiles += _ma.odd_codes()
# the Δl of the magnetic couplings assemble.py loops over
mag_dl = list(_ma.dl_range()) if mag_algebra else [-2, -1, 0, 1, 2]

def dr_offd(l, ls, row, col):
    '''
    block offset, in units of N1 as assemble.py uses it, of the column block of the unknown col ('P', 'S' or 'T') of degree ls
    for the row of degree l of section row ('u', 'h': rows in the poloidal l set; 'v': toroidal)
    '''
    lp, lt, _ = ell(par.m, par.lmax, par.symm)
    rows = lt if row == 'v' else lp
    cols = lt if col == 'T' else lp
    return int(np.where(cols == ls)[0][0] - np.where(rows == l)[0][0])

s = int( (vsymm+1)/2 ) # s=0 if antisymm, s=1 if symm
m_top = m + 1-s
m_bot = m + s
if m_top == 0: m_top = 2
if m_bot == 0: m_bot = 2
lmax_top = lmax + 1 + (1-2*np.sign(m))*s
lmax_bot = lmax + 1 + (1-2*np.sign(m))*(1-s)


def decode_label(labl):

    (section, rpower, rhopower, func1, dorder1, func2, dorder2, dx) = (None, None, None, None, None, None, None, None)

    howlong = len(labl)
    section = labl[0]        # this is 'u' or 'v' or 'h' (or 'f', 'g' with magnetic = 1)
    rx      = int(labl[1])   # related to rpower
    dx      = int(labl[-1])  # operators' derivative order

    if section == 'u':
        if par.ViscosD == 0 and par.magnetic:   # NEW (2026-10-28): inviscid with a magnetic field, r⁵ ρ² (the Lorentz force has h/r²)
            rpower = par.rpower_u - rx ; rhopower = 2
        elif par.ViscosD == 0:
            rpower = 3 - rx ; rhopower = 2   # Inviscid, we multiply the r̂⋅∇×∇× equations by r³ ρ**rhopower
        else:
            rpower = par.rpower_u - rx ; rhopower = par.rhopower_u   # Viscous, we multiply the r̂⋅∇×∇× equations by (r**rpower_u)*(ρ**rhopower)

    elif section == 'v':
        if par.ViscosD == 0 and par.magnetic:   # NEW (2026-10-28): inviscid with a magnetic field, r³ ρ
            rpower = par.rpower_v - rx ; rhopower = 1
        elif par.ViscosD == 0:
            rpower = 2 - rx ; rhopower = 1   # Inviscid, we multiply the r̂⋅∇× equations by r² ρ
        else:
            rpower = par.rpower_v - rx ; rhopower = par.rhopower_v   # Viscous, we multiply the r̂⋅∇× equations by r³ ρ

    elif section == 'h':
        if par.ThermaD == 0 and not adiab_tw:
            rpower = 1 - rx ; rhopower = 0   # Adiabatic motion, we multiply the thermal equation by r
        else:
            rpower = 2 - rx ; rhopower = 0   # Non-adiabatic motion (or adiabatic with the thermal wind), we multiply the thermal equation by r²

    elif section in ['f', 'g']:   # NEW (2026-10-28): the induction equation, r² r̂⋅ (f) and r² r̂⋅∇× (g), not multiplied by ρ
        rpower = 2 - rx ; rhopower = 0

    if howlong in [9,13,17]:   # sXfu1X_DX or sXfu1Xfu2X_DX
        func1   = labl[2:5]
        dorder1 = int(labl[5])

        if howlong == 13:   # sXfu1Xfu2X_DX
            func2   = labl[6:9]
            dorder2 = int(labl[9])
            if (func2 == 'ohr') and (dorder2==1):
                rhopower = rhopower - 1
                (func2, dorder2) = (None, None)

        elif howlong == 17:   # sXfu1Xfu2XohrX_DX
            func2   = labl[6:9]
            dorder2 = int(labl[9])
            if labl[10:14] == 'ohr1':
                rhopower = rhopower - 1

    return (section, rpower, rhopower, func1, dorder1, func2, dorder2, dx)


def gimmedachebs( labl ):
    '''
    Returns the Chebyshev coefficients of the operator identified by labl with the form
    sX_DX
    sXfu1X_DX
    sXfu1Xfu2X_DX
    s can be 'u' or 'v' or 'h' and the X's are single digit integers (can be all different)
    '''

    tol = max(1e-15, par.drop_tol)  # relative to the largest coefficient, see chebco_f
    if ('bpo' in labl or 'bto' in labl or labl[2:4] in ('hp', 'kt')) and 'lho' not in labl:   # (hp, kt: NEW 2026-10-30)  # NEW (2026-10-28): the background field's own cut (tol_B0, as conductive_IC);
        tol = max(tol, par.tol_B0)            # not for its products with the density profile (they can be long near the surface)
    args = decode_label(labl)  # (section, rpower, rhopower, func1, dorder1, func2, dorder2, dx)
    c0arg = chebco_f( rap.burrito, par.N, par.ricb, rcmb, tol, *args)
    # print('burrito', labl, args)

    return c0arg


def chegevara( pkey1 , opkey, labl, S):
    '''
    Generates Chebyshev coeffs for a given operator label
    and changes the Gegenbauer basis as needed by the operator
    '''

    k = opkey.index(pkey1)
    labl1 = labl[k]

    c0arg = gimmedachebs(labl1)
    dx    = pkey1[6]
    out   = S[dx]*c0arg

    return out


def packit( lista_local, mtx, row, col):
    '''
    Appends sparse matrix data, row, and col info to lista_local.
    lista_local is [data, row, col], each a list of arrays, joined only once at the end with unpackit
    (concatenating at every call copies everything accumulated so far, O(n^2) overall)
    '''
    mtx.eliminate_zeros()
    mtx = mtx.tocoo()
    blk = [mtx.data, mtx.row + row, mtx.col + col]
    for q in [0,1,2]:
        lista_local[q].append( blk[q] )

    return lista_local


def unpackit( lista_local ):
    '''
    Joins the lists of arrays built with packit into three arrays: data, row, col
    '''
    if len(lista_local[0]) == 0:  # no entries on this rank
        return [ np.zeros(0), np.zeros(0, dtype=int), np.zeros(0, dtype=int) ]
    return [ np.concatenate( lista_local[q] ) for q in [0,1,2] ]


def ell( m, lmax, vsymm) :
    '''
    Returns the l values for the poloidal flow (section u) and l values for toroidal flow (section v)
    ll are *all* the l values and (idp,idt) are the indices for poloidals and toroidals respectively
    '''
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


def chebco_f( func, N, ricb, rcmb, tol, *args):
    '''
    Returns the first N Chebyshev coefficients
    from 0 to N-1, of func(r). Coefficients smaller than
    tol times the largest one (in absolute value) are set to zero.
    '''
    i = np.arange(0, N)
    xi = np.cos(np.pi * (i + 0.5) / N)

    if ricb > 0:
        ri = (ricb + (rcmb - ricb) * (xi + 1) / 2.)
    elif ricb == 0 :
        ri = rcmb * xi

    tmp = sft.dct( func(ri,*args) )

    out = tmp / N
    out[0] = out[0] / 2.
    out[np.absolute(out) <= tol * np.max(np.absolute(out))] = 0.
    return out


def Dcheb(ck, ricb, rcmb):
    '''
    The derivative of a Chebyshev expansion with coefficients ck
    returns the coefficients of the derivative in the Chebyshev basis
    assumes ck computed for r in the domain [ricb,rcmb] (if ricb>0)
    or r in [-rcmb,rcmb] if ricb=0.
    If ck is 2D, each column is one expansion (used by utils4pp.cheb2space_table).
    '''
    c = np.copy(ck)
    c[0] = 2.*c[0]
    s =  len(c)  # NEW (2026-10-20): len instead of np.size, so a 2D ck (one expansion per column) works too
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


def xcheb(r, ricb, rcmb):
    '''
    returns points in the appropriate domain of the Cheb polynomial solutions
    Domain [-1,1] corresponds to [ ricb,rcmb] if ricb>0
    Domain [-1,1] corresponds to [-rcmb,rcmb] if ricb==0
    '''

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

    x00 = xcheb(r, ricb, rcmb)  # use the explicit radial points given as argument

    out = np.zeros((np.size(x00), n+1), ck0.dtype)  # n+1 cols
    out[:,0] = ch.chebval(x00, ck0)  # the function itself

    if n>0:
        dk = Dn_cheb(ck0, ricb, rcmb, n)  # coeffs for the derivatives, n cols
        for j in range(1,n+1):
            out[:,j] = ch.chebval(x00, dk[:,j-1])  # and the derivatives

    return out


def fonzie( func, r, N, ricb, rcmb, Dorder, tol, *args):
    out = np.zeros((np.size(r), 1))

    ck  = chebco_f( func, N, ricb, rcmb, tol, args)
    if id(func) in [ id(rap.prf.density), id(rap.prf.pdSdr), id(rap.prf.pressure), id(rap.prf.gravity) ]:
        ck = ironit(ck, par.smopo)
    out = funcheb(ck, r, ricb, rcmb, Dorder)

    return out


def angine( func, r, N, ricb, rcmb, Dorder, tol, *args):
    '''
    Returns func(r) or its Dorder derivative.
    '''

    out = np.zeros_like(r)
    if Dorder == 0:
        out = func(r, *args)
    elif Dorder>0:
        ck  = chebco_f( func, N, ricb, rcmb, tol, args)  # get Cheb coeffs
        out = funcheb(ck, r, ricb, rcmb, Dorder)[:,-1]   # compute derivative

    return out


def ironit(coeffs, strength):

    x = np.linspace(0,1,np.size(coeffs))
    y = 1-x
    y = (y/y[0])**strength
    out = coeffs * y
    
    return out


def erf_transition(r, r0, scaling_factor, amplitude):
    '''
    A nice smooth erf-based transition function at r=r0
    the scaling factor controls how sharp the transition is,
    the higher the factor the sharper the transition.
    '''
    out = np.zeros_like(r)
    k = r>0
    out[k] = scsp.erfc((r[k]-r0)*scaling_factor) * amplitude/2
    if min(r)<0:
        out[~k] = np.flipud(out[k])
    return out


def erf_top_hat(x, x1, w1, x2, w2, A):
    return A * 0.5 * (scsp.erf((x - x1) / w1) - scsp.erf((x - x2) / w2))


def erf_box(x, x2, w2):
    '''
    Smooth even box, 1 for |x| < x2 and 0 for |x| > x2, with erf transitions of width w2 at x = ±x2:
    erfc((x-x2)/w2) erfc(-(x+x2)/w2)/4. On x >= 0 it equals the single step erfc((x-x2)/w2)/2 to within
    exp(-(x2/w2)^2), and it is even by construction, so it has no kink at x = 0 whatever x2/w2.
    '''
    return 0.25 * scsp.erfc((x - x2) / w2) * scsp.erfc(-(x + x2) / w2)


def erf_top_hat_odd(x, x1, w1, x2, w2, A):
    '''
    The erf top hat between x1 and x2 made odd in x:
    A/2 [erf((x-x1)/w1) + erf((x+x1)/w1)] erf_box(x, x2, w2),
    an odd factor times an even one, so it is exactly odd. On x >= 0 it is erf_top_hat when x1 >> w1,
    but it vanishes smoothly at x = 0 for any x1 (x1 = 0 gives A erf(x/w1) erf_box(x, x2, w2)), so the
    odd extension sign(x) f(|x|) that kore uses when ricb = 0 is smooth at the centre. The central
    erf(x/w1) needs about 11/w1 Chebyshev modes on [-1, 1] to reach round-off (N ~ 380 for w1 = 0.03).
    '''
    return A * 0.5 * (scsp.erf((x - x1) / w1) + scsp.erf((x + x1) / w1)) * erf_box(x, x2, w2)


def interp(x0, x, y, even=True):

    akima = si.Akima1DInterpolator(x, y)
    out = np.zeros_like(x0)

    if np.min(x)<0:
        out = akima(x0)
    else:
        out[x0>=0] = akima(x0[x0>=0])
        if even:
            out[x0<0] = akima(-x0[x0<0])
        else:
            out[x0<0] = -akima(-x0[x0<0])

    return out


def Dlam(lamb,N,R1=None,R2=None):
    '''
    Order lamb (>=1) derivative matrix, size N*N
    NEW (2026-10-28): for the Chebyshev domain [R1, R2], or [-R2, R2] if R1 = 0 (default: [ricb, rcmb], as before)
    '''
    if R1 is None:
        R1, R2 = par.ricb, rcmb
    if R1 == 0:
        const1 = (1/R2)**lamb  # ok when R2 is not 1
    else:
        const1 = (2./(R2-R1))**lamb
    const2 = scsp.factorial(lamb-1.)*2**(lamb-1.)
    tmp = lamb + np.arange(0,N-lamb)

    return const1*const2*ss.diags(tmp,lamb, format='csr', dtype='float64')


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


def Mlam(a0,lamb,vector_parity,a0_parity=None):
    '''
    Multiplication matrix. a0 are the cofficients in the C^(lamb) basis and lamb
    is the order of the C^(lamb) basis. (This basis should match the
    one from the highest derivative order appearing in the equation)
    '''

    N = np.size(a0)

    if np.sum(abs(a0)) > 0 :

        bw = max(np.nonzero(a0)[0])

        a1 = np.zeros(2*N)
        a1[:N] = a0


        if vector_parity != 0: # no inner core case

            # Overall operator parity given by a0 parity * lambda parity
            # check a0 parity like this: first nonzero a0 coefficient
            # a0 is the full vector of coefficients, including even and odd, size N
            #tmp = np.nonzero(a0)[0]
            #ix = tmp[-1] # index of *last* non zero coefficient
            if a0_parity is None:  # guess it from the largest coefficient (wrong for profiles of mixed parity)
                ix = np.argmax(abs(a0))  # index of largest a0 coeff     #2*((argmax(abs(c0)))%2)-1
                a0_parity = 1 - 2*(ix%2)
            lamb_parity = 1 - 2*(lamb%2)
            operator_parity = a0_parity * lamb_parity
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

            # Vectorised over all entries (j,k) with |j-k| <= bw, the others are zero.
            # Only the s terms with 2*s+j-k <= bw contribute (a1 is zero beyond bw), at most bw//2+1 of them.
            # Assumes integer lamb, as for all the C^(lamb) bases used in Kore.
            jj  = np.array(jrange)
            off = np.arange(-bw, bw+1)
            J = np.repeat(jj, off.size)
            K = (jj[:,None] + off[None,:]).ravel()
            keep = (K >= 0) & (K < N)
            if vector_parity != 0:
                keep &= (K%2 == idk)
            J = J[keep]
            K = K[keep]

            d     = np.abs(J-K)
            s0    = np.maximum(0, K-J)
            nterm = np.minimum(K, s0 + (bw-d)//2) - s0  # number of terms after the first one

            # c_{s0}^lamb(K,d), each product telescoped to lamb or lamb-1 factors
            jf = K.astype(float)
            kf = d.astype(float)
            s  = s0.astype(float)
            n  = jf - s
            a  = lamb + jf + kf - 2*s
            p  = np.ones_like(s)
            for i in range(1,lamb): p *= (s+i)/i                    # (lamb)_s / s!
            for i in range(1,lamb): p *= (n+i)/i                    # (lamb)_n / n!
            for i in range(lamb):   p *= (a+s+i)/(a+i)              # (a+lamb)_s / (a)_s
            for i in range(lamb-1): p *= (kf-s+1+i)/(kf-s+1+n+i)    # (kf-s+1)_n / (kf-s+lamb)_n
            c = p*(jf+kf+lamb-2*s)/(jf+kf+lamb-s)

            val = a1[2*s0+J-K]*c

            # forward recursion in s, only for the entries that still have nonzero terms
            for q in range(1, bw//2+1):
                act = nterm >= q
                if not act.any():
                    break
                sa = s[act]; ja = jf[act]; ka = kf[act]
                tmp1 = (ja+ka+lamb-sa)*(lamb+sa)*(ja-sa)*(2*lamb+ja+ka-sa)*(ka-sa+lamb)
                tmp2 = (ja+ka+lamb-sa+1)*(sa+1)*(lamb+ja-sa-1)*(lamb+ja+ka-sa)*(ka-sa+1)
                c[act]    = c[act]*tmp1/tmp2
                s[act]   += 1
                kf[act]  += 2
                val[act] += a1[(2*s[act]+J[act]-K[act]).astype(int)]*c[act]

            out = ss.csr_matrix((val, (J, K)), shape=(N,N))
            out.eliminate_zeros()

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

        out = ss.csr_matrix((N,N))

    return out


def Ylm(l, m, theta, phi):
    # The Spherical Harmonics, seminormalized: sqrt(4π/(2l+1)) times scipy's sph_harm_y(l, m, theta, phi), by the
    # recurrence of angular.ylm_grid (scipy's sph_harm_y returns NaN from l = 646 on)
    return angular.ylm(l, m, theta, phi)


def Ylm_full(lmax, m, theta, phi):
    # array of Spherical Harmonics with a range of l's (theta, phi scalars)
    m1 = max(m,1)
    if m == 0 :
        lmax1 = lmax+1
    else:
        lmax1 = lmax
    out = np.zeros(lmax-m+1,dtype=np.complex128)
    if m < 0:
        for l in np.arange(m1,lmax1+1):
            out[l-m1]=Ylm(l,m,theta,phi)
        return out
    Y = angular.ylm_grid(m, lmax1, np.cos(theta))[0][:, 0]*np.exp(1j*m*phi)
    out[:lmax1-m1+1] = Y[m1:lmax1+1]
    return out


def load_csr(filename):
    # utility to load sparse matrices efficiently
    loader = np.load(filename)
    return ss.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def load_npz_mmap(filename):
    '''
    Memory-maps the arrays stored in an uncompressed .npz file (as written by np.savez),
    so that slicing them reads only the requested part from disk.
    Returns a dict {array name: read-only np.memmap}.
    '''
    import zipfile, struct
    out = {}
    with zipfile.ZipFile(filename) as z, open(filename, 'rb') as f:
        for info in z.infolist():
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(filename + ' is compressed, cannot memory-map it')
            # skip the zip local file header to reach the .npy data
            f.seek(info.header_offset)
            nlen, xlen = struct.unpack('<HH', f.read(30)[26:30])
            f.seek(info.header_offset + 30 + nlen + xlen)
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            else:
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            out[info.filename[:-4]] = np.memmap(filename, dtype=dtype, mode='r', shape=shape,
                                                offset=f.tell(), order='F' if fortran else 'C')
    return out
