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

# this gives the size (rows or columns) of the main matrices
sizmat = 2*n + n*par.thermal

s = int( (vsymm+1)/2 ) # s=0 if antisymm, s=1 if symm
m_top = m + 1-s
m_bot = m + s
if m_top == 0: m_top = 2
if m_bot == 0: m_bot = 2
lmax_top = lmax + 1 + (1-2*np.sign(m))*s
lmax_bot = lmax + 1 + (1-2*np.sign(m))*(1-s)


# ----------------------------------------------------------------------------------------------------------------------
# Loads a model from file: ---------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if par.model_type == 'astropy table':
    try:
        from astropy.table import Table
    except ImportError:
        print('Astropy is not installed. Please install it or choose another model_type.')
        sys.exit()
    profile = Table.read(par.model, format='ascii')
elif par.model_type in ['poly', 'mesa', 'gsm']:
    try:
        import pygyre as gy
    except ImportError:
        print('PyGYRE is not installed. Please install it or choose another model_type.')
        sys.exit()
    profile = gy.read_model(par.model)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def decode_label(labl):

    (section, rpower, rhopower, func1, dorder1, func2, dorder2, dx) = (None, None, None, None, None, None, None, None)

    howlong = len(labl)
    section = labl[0]        # this is 'u' or 'v' or 'h'
    rx      = int(labl[1])   # related to rpower
    dx      = int(labl[-1])  # operators' derivative order

    if section == 'u':
        if par.ViscosD == 0:
            rpower = 3 - rx ; rhopower = 2   # Inviscid, we multiply the r̂⋅∇×∇× equations by r³ ρ**rhopower
        else:
            rpower = par.rpower_u - rx ; rhopower = par.rhopower_u   # Viscous, we multiply the r̂⋅∇×∇× equations by (r**rpower_u)*(ρ**rhopower)

    elif section == 'v':
        if par.ViscosD == 0:
            rpower = 2 - rx ; rhopower = 1   # Inviscid, we multiply the r̂⋅∇× equations by r² ρ
        else:
            rpower = par.rpower_v - rx ; rhopower = par.rhopower_v   # Viscous, we multiply the r̂⋅∇× equations by r³ ρ

    elif section == 'h':
        if par.ThermaD == 0:
            rpower = 1 - rx ; rhopower = 0   # Adiabatic motion, we multiply the thermal equation by r
        else:
            rpower = 2 - rx ; rhopower = 0   # Non-adiabatic motion, we multiply the thermal equation by r²

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

    tol = 1e-9
    args = decode_label(labl)  # (section, rpower, rhopower, func1, dorder1, func2, dorder2, dx)
    c0arg = chebco_f( rap.burrito, par.N, par.ricb, rcmb, tol, *args)
    print('burrito', labl, args)

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
    from 0 to N-1, of func(r)
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
    out[np.absolute(out) <= tol] = 0.
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


def load_model(r, var):

    out = np.zeros_like(r)
    z = True  # True for even functions of r

    if par.model_type in ['astropy table', 'poly']:

        x0 = profile['x']
        x = x0[x0<=par.aux0]; x=x/x[-1]
        y = np.zeros_like(x)
        y0 = np.zeros_like(x0)

        if var == 'density':
            y0 = profile['rho/rho_0']; y=y0[x0<=par.aux0]

        elif var == 'pressure':
            y0 = profile['P/P_0']; y=y0[x0<=par.aux0]

        elif var == 'gravity':
            y0 = x0/profile['c_1']; y=y0[x0<=par.aux0]
            z = False  # odd function of r

        elif var == 'pdSdr':
            y0[1:-1] = profile['P/P_0'][1:-1] * profile['As'][1:-1] / x0[1:-1]; y=y0[x0<=par.aux0]
            z = False  # odd function of r

    elif par.model_type == 'mesa' or par.model_type == 'gsm':

        G_star = 6.67430e-8                 # gravitational constant in cm^3 g^-1 s^-2
        M_star = profile.meta['M_star']     # stellar mass in g
        R_star = profile.meta['R_star']     # stellar radius in cm
        x = profile['r'] / R_star
        y = np.zeros_like(x)

        if var == 'density':
            y = profile['rho'] * R_star**3 / M_star   # dimensionless density

        if var == 'pressure':
            y = profile['P'] * R_star**4 / M_star**2 / G_star # dimensionless pressure

        if (var == 'gravity') or (var == 'pdSdr'):
            if profile.meta['version'] > 20:
                mass = profile['M_r'] / M_star  # dimensionless mass for newest MESA file formats
            else:
                mass = 1 / (1 + 1/profile['w'])  # dimensionless mass for older MESA file formats

            y[1:] = ( mass[1:] / x[1:]**2 )  # dimensionless gravity

            if var == 'pdSdr':
                BV2 = profile['N^2'] * R_star**3 / M_star / G_star  # dimensionless Brunt-Väisälä frequency
                pressure = profile['P'] * R_star**4 / M_star**2 / G_star # dimensionless pressure
                y[1:] = BV2[1:] * pressure[1:] / y[1:]

            z = False

    out = interp(r, x, y, even=z)

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
    # The Spherical Harmonics, seminormalized
    #out = scsp.sph_harm(m, l, phi, theta)   ### for scipy older than 1.15.3
    out = scsp.sph_harm_y(l,m,theta,phi)    ### for scipy 1.15.3 or newer  
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
