import numpy as np
import scipy.sparse as ss
try:
    import shtns
except ImportError:  # kmode then keeps only the spectral coefficients, potextra uses scipy
    shtns = None


def ell_idx(m, lmax, vsymm):
    '''
    Returns the degrees l of the rows of the (l, r) arrays of a field with azimuthal number m and equatorial
    symmetry vsymm, and the row indices of its poloidal (or scalar) and toroidal parts
    '''
    lm1 = lmax - m + 1
    s = int(vsymm*0.5+0.5) # s=0 if antisymm, s=1 if symm

    if m > 0:
        idp = np.arange( 1-s, lm1, 2)
        idt = np.arange( s  , lm1, 2)
        ll  = np.arange( m, lmax+1 )
    else:
        idp = np.arange( s  , lm1, 2)
        idt = np.arange( 1-s, lm1, 2)
        ll  = np.arange( m+1, lmax+2 )

    return ll, idp, idt


def spec2spat_vec(M,ut,chx,Plj,Tlj,vsymm,nthreads,
                  vort=False,transform=True):

    lm1  = M.lmax-M.m+1

    # init arrays
    Plr  = np.zeros( (lm1, M.nr), dtype=complex )
    Qlr  = np.zeros( (lm1, M.nr), dtype=complex )
    Slr  = np.zeros( (lm1, M.nr), dtype=complex )
    Tlr  = np.zeros( (lm1, M.nr), dtype=complex )
    dP   = np.zeros( (lm1, M.nr), dtype=complex )
    rP   = np.zeros( (lm1, M.nr), dtype=complex )
    dPlj = np.zeros(  np.shape(Plj), dtype=complex )
    if vort:
        ddPlj = np.zeros(  np.shape(Plj), dtype=complex )
        dTlj  = np.zeros(  np.shape(Tlj), dtype=complex )
        dT    = np.zeros( (lm1, M.nr), dtype=complex )
        ddP   = np.zeros( (lm1, M.nr), dtype=complex )

    # These are the l values (ll) and indices (idp,idt)
    ll, idp, idt = ell_idx(M.m, M.lmax, vsymm)

    # populate Plr and Tlr
    Plr[idp,:] = np.matmul( Plj, chx.T)
    Tlr[idt,:] = np.matmul( Tlj, chx.T)

    # populate dPlj and dP
    for k in range(int(lm1/2)):
        dPlj[k,:] = ut.Dcheb(Plj[k,:], M.ricb, M.rcmb)
        dP[idp,:] = np.matmul(dPlj, chx.T)
    if vort:
        for k in range(int(lm1/2)):
            dTlj[k,:] = ut.Dcheb(Tlj[k,:], M.ricb, M.rcmb)
            ddPlj[k,:] = ut.Dcheb(dPlj[k,:], M.ricb, M.rcmb)
            dT[idt,:] = np.matmul(dTlj, chx.T)
            ddP[idp,:] = np.matmul(ddPlj, chx.T)

    # populate Qlr and Slr
    rI = ss.diags(M.r**-1,0)
    L  = ss.diags(ll*(ll+1),0)

    if vort:
        r2I = ss.diags(M.r**-2,0)
        r2P = Plr * r2I

        Qlr = L * Tlr  # l(l+1) * T
        Slr = dT + Tlr * rI  # T' + T/r
        Tlr = L * r2P - 2 * dP * rI - ddP # l(l+1) * P/r^2 - 2*P'/r - P"

    else:

        rP  = Plr * rI  # P/r
        Qlr = L * rP    # l(l+1)*P/r
        Slr = rP + dP   # P' + P/r

    # Now in these Q, S, T arrays, the first lmax+1 indices are for m=0
    # and the remaining lmax+1-m are for mres.
    # (this is the SHTns way with m=mres when m is not zero)

    lmax2 = int( M.lmax + 1 - np.sign(M.m) )  # the true max value of l
#    nlm = ( np.sign(M.m)+1 ) * (lmax2+1) - M.m

    if M.m == 0 :  #pad with zeros for the l=0 component
        ql = np.r_[ np.zeros((1,M.nr)) ,Qlr ]
        sl = np.r_[ np.zeros((1,M.nr)) ,Slr ]
        tl = np.r_[ np.zeros((1,M.nr)) ,Tlr ]
        pl = np.r_[ np.zeros((1,M.nr)) ,Plr ]
    else :
        ql = Qlr
        sl = Slr
        tl = Tlr
        pl = Plr #Only for output, not needed for transforms


    # SHTns init. Schmidt seminormalized with the Condon-Shortley phase, as utils.Ylm_full in bin/
    norm = shtns.sht_schmidt
    mmax = int( np.sign(M.m) )
    mres = max(1,M.m)
    sh   = shtns.sht( lmax2, mmax=mmax, mres=mres, norm=norm, nthreads=nthreads )
    M.sh = sh

    # Create spectral arrays in SHTns style

    Q = np.zeros([M.nr, sh.nlm], dtype=complex)
    S = np.zeros([M.nr, sh.nlm], dtype=complex)
    T = np.zeros([M.nr, sh.nlm], dtype=complex)
    P = np.zeros([M.nr, sh.nlm], dtype=complex)

    mask = sh.m == M.m

    Q[:,mask] = ql.T
    S[:,mask] = sl.T
    T[:,mask] = tl.T
    P[:,mask] = pl.T

    if transform:
        ntheta, nphi = sh.set_grid( M.ntheta+M.ntheta%2, M.nphi, polar_opt=1e-10)
        M.theta = np.arccos(sh.cos_theta)
        M.phi   = np.linspace(0., 2*np.pi, M.nphi*mres+1, endpoint=True)
        M.ntheta = ntheta
        M.nphi   = nphi

        # init the spatial component arrays
        ur     = np.zeros([M.nr, ntheta, nphi] )
        utheta = np.zeros([M.nr, ntheta, nphi] )
        uphi   = np.zeros([M.nr, ntheta, nphi] )

    # the final call to shtns for each radius
        for ir in range(M.nr):
            ur[ir,...], utheta[ir,...],  uphi[ir,...] = sh.synth( Q[ir,:], S[ir,:], T[ir,:])

        return Q,S,P,T,ur,utheta,uphi
    else:
        return Q,S,P,T


def spec2spat_scal(M,chx,Plj,vsymm,nthreads,transform=True):

    lm1  = M.lmax-M.m+1

    # init arrays
    Plr  = np.zeros( (lm1, M.nr), dtype=complex )

    # These are the l values (ll) and indices (idp)
    ll, idp, idt = ell_idx(M.m, M.lmax, vsymm)

    # populate Plr
    Plr[idp,:] = np.matmul( Plj, chx.T)

    # (this is the SHTns way with m=mres when m is not zero)

    lmax2 = int( M.lmax + 1 - np.sign(M.m) )  # the true max value of l
    # nlm = ( np.sign(M.m)+1 ) * (lmax2+1) - M.m

    if M.m == 0 :  #pad with zeros for the l=0 component
        ql = np.r_[ np.zeros((1,M.nr)) ,Plr ]
    else :
        ql = Plr

    norm = shtns.sht_schmidt  # with the Condon-Shortley phase, as utils.Ylm_full in bin/
    mmax = int( np.sign(M.m) )
    mres = max(1,M.m)
    sh   = shtns.sht( lmax2, mmax=mmax, mres=mres, norm=norm, nthreads=nthreads )
    M.sh = sh

    Q = np.zeros([M.nr, sh.nlm], dtype=complex)
    mask = sh.m == M.m
    Q[:,mask] = ql.T

    if transform:
        ntheta, nphi = sh.set_grid( M.ntheta+M.ntheta%2, M.nphi, polar_opt=1e-10)
        M.theta = np.arccos(sh.cos_theta)
        M.phi   = np.linspace(0., 2*np.pi, M.nphi*mres+1, endpoint=True)

        # init the spatial component arrays
        M.ntheta = ntheta
        M.nphi   = nphi
        M.sh     = sh
        scal     = np.zeros([M.nr, ntheta, nphi] )
        # the final call to shtns for each radius
        for ir in range(M.nr):
            scal[ir,...] = sh.synth( Q[ir,:])
        return Q,scal
    else:
        return [Q]

def get_ang_momentum(M,epsilon_cmb):
    '''
    Poloidal and toroidal contributions to the axial torque on a non-spherical CMB r = rcmb*(1 + epsilon).
    M is a kmode of the flow (field='u'); epsilon_cmb holds the spherical harmonic coefficients of the
    topography epsilon in the layout and normalisation of M.sh (e.g. from M.sh.analys).
    '''

    l   = M.sh.l
    m   = M.sh.m
    r   = M.r
    Slm = M.Slm[0,:]
    Tlm = M.Tlm[0,:]

    Gamma_tor = np.zeros(M.sh.nlm,dtype=np.complex128)

    Gamma_pol = 1j * m * Slm * M.rcmb * np.conjugate(epsilon_cmb)
    clm1 = (l+2)/(2*l+3) * np.sqrt((l+m+1)*(l-m+1))
    clm2 = (l-1)/(2*l-1) * np.sqrt((l+m)*(l-m))

    for mm in [0,M.m]:
        for ell in range(mm,M.lmax+1):
            k = M.sh.idx(ell,mm)
            if ell == mm:
                Gamma_tor[k] = M.rcmb * ( clm1[k] * Tlm[M.sh.idx(ell+1,mm)])
            elif ell == M.lmax:
                Gamma_tor[k] = M.rcmb * ( -clm2[k] * Tlm[M.sh.idx(ell-1,mm)] )
            else:
                Gamma_tor[k] = M.rcmb * ( clm1[k] * Tlm[M.sh.idx(ell+1,mm)]
                                         -clm2[k] * Tlm[M.sh.idx(ell-1,mm)] )
            Gamma_tor[k] *= np.conjugate(epsilon_cmb[k])

    torq_pollm = np.real( 4*np.pi/(2*l+1) * (Gamma_pol)) # elementwise (array) multiplication
    torq_torlm = np.real( 4*np.pi/(2*l+1) * (Gamma_tor))
    mask = M.sh.m == 0
    torq_pollm[~mask] *= 2
    torq_torlm[~mask] *= 2
    torq_pol = np.sum(torq_pollm)
    torq_tor = np.sum(torq_torlm)

    return torq_pol, torq_tor

def get_coriolis_torque(M,epsilon_cmb):
    '''
    Radial, consoidal and toroidal contributions to the Coriolis torque on a non-spherical CMB
    r = rcmb*(1 + epsilon). Arguments as in get_ang_momentum.
    '''

    l   = M.sh.l
    m   = M.sh.m
    r   = M.r
    Qlm = M.Qlm[0,:]
    Slm = M.Slm[0,:]
    Tlm = M.Tlm[0,:]

    Gamma_rad = np.zeros(M.sh.nlm,dtype=np.complex128)
    Gamma_con = np.zeros(M.sh.nlm,dtype=np.complex128)
    Gamma_tor = np.zeros(M.sh.nlm,dtype=np.complex128)

    clm_rad_p2 = -2/(2*l+3)/(2*l+5) * np.sqrt((l+m+1)*(l-m+1)) * np.sqrt((l+m+2)*(l-m+2))
    clm_rad_0 = 4*(l**2+l-1+m**2)/(4*l*(l+1)-3)
    clm_rad_m2 = 2/(4*l*(l-2)+3) * np.sqrt((l+m)*(l-m)) * np.sqrt((l-1)**2-m**2)

    for mm in [0,M.m]:
        for ell in range(mm,M.lmax+1):
            k = M.sh.idx(ell,mm)
            if ell <= mm+1:
                Gamma_rad[k] = M.rcmb * ( clm_rad_p2[k] * Qlm[M.sh.idx(ell+2,mm)]
                                          +clm_rad_0[k] * Qlm[M.sh.idx(ell,mm)]   )
            elif ell >= M.lmax-1:
                Gamma_rad[k] = M.rcmb * ( clm_rad_m2[k] * Qlm[M.sh.idx(ell-2,mm)]
                                          +clm_rad_0[k] * Qlm[M.sh.idx(ell,mm)]    )
            else:
                Gamma_rad[k] = M.rcmb * ( clm_rad_p2[k] * Qlm[M.sh.idx(ell+2,mm)]
                                          +clm_rad_0[k] * Qlm[M.sh.idx(ell,mm)]
                                         +clm_rad_m2[k] * Qlm[M.sh.idx(ell-2,mm)] )
            Gamma_rad[k] *= np.conjugate(epsilon_cmb[k])

    clm_con_p2 = -2*(l+3)/(2*l+3)/(2*l+5) * np.sqrt((l+m+1)*(l-m+1)) * np.sqrt((l+m+2)*(l-m+2))
    clm_con_0 = -2*(l+l**2-3*m**2)/(4*l*(l+1)-3)
    clm_con_m2 = 2*(l-2)/(4*l*(l-2)+3) * np.sqrt((l+m)*(l-m)) * np.sqrt((l-1)**2-m**2)

    for mm in [0,M.m]:
        for ell in range(mm,M.lmax+1):
            k = M.sh.idx(ell,mm)
            if ell <= mm+1:
                Gamma_con[k] = M.rcmb * ( clm_con_p2[k] * Slm[M.sh.idx(ell+2,mm)]
                                          +clm_con_0[k] * Slm[M.sh.idx(ell,mm)]   )
            elif ell >= M.lmax-1:
                Gamma_con[k] = M.rcmb * ( clm_con_m2[k] * Slm[M.sh.idx(ell-2,mm)]
                                          +clm_con_0[k] * Slm[M.sh.idx(ell,mm)]    )
            else:
                Gamma_con[k] = M.rcmb * ( clm_con_p2[k] * Slm[M.sh.idx(ell+2,mm)]
                                          +clm_con_0[k] * Slm[M.sh.idx(ell,mm)]
                                         +clm_con_m2[k] * Slm[M.sh.idx(ell-2,mm)] )
            Gamma_con[k] *= np.conjugate(epsilon_cmb[k])

    clm_tor_p1 = 2*1j*m*(l+2)/(2*l+3) * np.sqrt((l+m+1)*(l-m+1))
    clm_tor_m1 = 2*1j*m*(l-1)/(2*l-1) * np.sqrt((l+m)*(l-m))

    for mm in [0,M.m]:
        for ell in range(mm,M.lmax+1):
            k = M.sh.idx(ell,mm)
            if ell == mm:
                Gamma_tor[k] = M.rcmb * ( clm_tor_p1[k] * Tlm[M.sh.idx(ell+1,mm)])
            elif ell == M.lmax:
                Gamma_tor[k] = M.rcmb * ( clm_tor_m1[k] * Tlm[M.sh.idx(ell-1,mm)] )
            else:
                Gamma_tor[k] = M.rcmb * ( clm_tor_p1[k] * Tlm[M.sh.idx(ell+1,mm)]
                                         +clm_tor_m1[k] * Tlm[M.sh.idx(ell-1,mm)] )
            Gamma_tor[k] *= np.conjugate(epsilon_cmb[k])

    torq_radlm = np.real( 4*np.pi/(2*l+1) * (Gamma_rad))
    torq_conlm = np.real( 4*np.pi/(2*l+1) * (Gamma_con))
    torq_torlm = np.real( 4*np.pi/(2*l+1) * (Gamma_tor))
    mask = M.sh.m == 0
    torq_radlm[~mask] *= 2
    torq_conlm[~mask] *= 2
    torq_torlm[~mask] *= 2
    torq_rad = np.sum(torq_radlm)
    torq_con = np.sum(torq_conlm)
    torq_tor = np.sum(torq_torlm)

    return torq_rad, torq_con, torq_tor


def potential_field(Pcmb, Picb, ll, m, rcmb, ricb, rout, theta, nphi, nthreads=1, backend='auto'):
    '''
    Potential magnetic field outside the CMB (insulating mantle) and inside the ICB (insulating inner core),
    from the poloidal scalar on each boundary. Each radius in rout is taken by itself: r >= rcmb is outside,
    r <= ricb is inside the inner core. Radii in the fluid shell, ricb < r < rcmb, are not allowed.

    With b = curl curl (P r) + curl (T r) (r the position vector, not the unit vector), the field is poloidal
    with P_l(r) = P_l(rcmb)*(rcmb/r)**(l+1) outside, and P_l(r) = P_l(ricb)*(r/ricb)**l inside, so that, for
    each degree l,
        b_r = l(l+1) P_l/r Y_l^m,   (b_theta, b_phi) = d(r P_l)/dr / r (d/dtheta, 1/sin(theta) d/dphi) Y_l^m,
    with d(r P_l)/dr / r = -l P_l/r outside and (l+1) P_l/r inside,
    with Y_l^m the Schmidt seminormalized spherical harmonics with the Condon-Shortley phase (as SHTns'
    sht_schmidt and utils.Ylm_full in bin/), and the real field taken as SHTns does (twice the real part for m > 0).

    Pcmb, Picb : P_l(rcmb) and P_l(ricb), one per degree in ll (complex). Picb is not used if no r <= ricb
    theta      : colatitudes of the grid; nphi points in longitude over 2*pi/max(1,m)
    backend    : 'shtns', 'scipy' or 'auto' (SHTns if available)

    Returns br, btheta, bphi with shape (len(rout), len(theta), nphi).
    '''

    rout = np.atleast_1d(rout)
    ll   = np.asarray(ll)
    mres = max(1, m)

    tol = 1e-12  # radii this close to a boundary count as on it
    outside = rout >= rcmb*(1 - tol)
    inside  = (rout <= ricb*(1 + tol)) & (rout > 0)
    if not np.all(outside | inside):
        raise ValueError('rout must be >= rcmb = %g or in (0, ricb = %g]; the fluid shell is not a potential '
                         'field region' % (rcmb, ricb))

    def coeffs(r, out):  # P_l(r) and d(r P_l)/dr / P_l
        if out:
            return Pcmb * (rcmb/r)**(ll+1), -ll
        else:
            return Picb * (r/ricb)**ll, ll + 1

    if backend == 'auto':
        backend = 'shtns' if shtns is not None else 'scipy'

    br = np.zeros([len(rout), len(theta), nphi])
    bt = np.zeros_like(br)
    bp = np.zeros_like(br)

    if backend == 'shtns':

        lmax2 = int(ll.max())
        sh = shtns.sht( lmax2, mmax=int(np.sign(m)), mres=mres, norm=shtns.sht_schmidt, nthreads=nthreads )
        ntheta, nphi_sh = sh.set_grid( len(theta), nphi, polar_opt=1e-10 )
        assert np.allclose(np.arccos(sh.cos_theta), theta) and nphi_sh == nphi, 'grid differs from the SHTns one'
        k = np.array([ sh.idx(int(l), m) for l in ll ])
        zero = np.zeros(sh.nlm, dtype=complex)

        for i, r in enumerate(rout):
            P, sfac = coeffs(r, outside[i])
            Q = zero.copy(); Q[k] = ll*(ll+1)*P/r
            S = zero.copy(); S[k] = sfac*P/r
            br[i], bt[i], bp[i] = sh.synth(Q, S, zero)

    elif backend == 'scipy':

        from scipy.special import sph_harm_y  # scipy >= 1.15; returns nan for l >= ~646

        phi = 2*np.pi*np.arange(nphi)/(nphi*mres)
        th2, ph2 = np.meshgrid(theta, phi, indexing='ij')
        fac = 1 if m == 0 else 2                                   # real field, as SHTns
        Ylm, dYt = [], []
        for l in ll:
            y, dy = sph_harm_y(int(l), m, th2, ph2, diff_n=1)       # orthonormal, with Condon-Shortley phase
            norm = np.sqrt(4*np.pi/(2*l+1))                       # -> Schmidt seminormalized
            Ylm.append(norm*y); dYt.append(norm*dy[..., 0])

        for i, r in enumerate(rout):
            P, sfac = coeffs(r, outside[i])
            for j, l in enumerate(ll):
                br[i] += fac*np.real( l*(l+1)*P[j]/r * Ylm[j] )
                bt[i] += fac*np.real( sfac[j]*P[j]/r * dYt[j] )
                bp[i] += fac*np.real( sfac[j]*P[j]/r * 1j*m*Ylm[j]/np.sin(th2) )

    else:
        raise ValueError("backend must be 'auto', 'shtns' or 'scipy'")

    return br, bt, bp
