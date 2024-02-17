import numpy as np
import scipy.sparse as ss
import shtns


def spec2spat_vec(M,ut,par,chx,a,b,vsymm,nthreads,
                  vort=False,transform=True):

    # Rearrange and separate poloidal and toroidal parts

    Plj0 = a[:M.n] + 1j*b[:M.n]         #  N elements on each l block
    Tlj0 = a[M.n:2*M.n] + 1j*b[M.n:2*M.n]   #  N elements on each l block

    lm1  = M.lmax-M.m+1
    Plj0  = np.reshape(Plj0,(int(lm1/2),ut.N1))
    Tlj0  = np.reshape(Tlj0,(int(lm1/2),ut.N1))

    Plj = np.zeros((int(lm1/2),par.N),dtype=complex)
    Tlj = np.zeros((int(lm1/2),par.N),dtype=complex)

    if M.ricb == 0 :
        iP = (M.m + 1 - ut.s)%2
        iT = (M.m + ut.s)%2
        for k in np.arange(int(lm1/2)) :
            Plj[k,iP::2] = Plj0[k,:]
            Tlj[k,iT::2] = Tlj0[k,:]
    else :
        Plj = Plj0
        Tlj = Tlj0

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
    s = int(vsymm*0.5+0.5) # s=0 if antisymm, s=1 if symm

    if M.m > 0:
        idp = np.arange( 1-s, lm1, 2)
        idt = np.arange( s  , lm1, 2)
        ll  = np.arange( M.m, M.lmax+1 )
    elif M.m == 0:
        idp = np.arange( s  , lm1, 2)
        idt = np.arange( 1-s, lm1, 2)
        ll  = np.arange( M.m+1, M.lmax+2 )

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
    nlm = ( np.sign(M.m)+1 ) * (lmax2+1) - M.m
    Q = np.zeros([M.nr, nlm], dtype=complex)
    S = np.zeros([M.nr, nlm], dtype=complex)
    T = np.zeros([M.nr, nlm], dtype=complex)

    if M.m == 0 :  #pad with zeros for the l=0 component
        ql = np.r_[ np.zeros((1,M.nr)) ,Qlr ]
        sl = np.r_[ np.zeros((1,M.nr)) ,Slr ]
        tl = np.r_[ np.zeros((1,M.nr)) ,Tlr ]
    else :
        ql = Qlr
        sl = Slr
        tl = Tlr

    Q[:, np.sign(M.m)*(lmax2+1):] = ql.T
    S[:, np.sign(M.m)*(lmax2+1):] = sl.T
    T[:, np.sign(M.m)*(lmax2+1):] = tl.T
    M.ell = np.arange(M.m,M.lmax+1)


    # SHTns init
    #norm = shtns.sht_schmidt | shtns.SHT_NO_CS_PHASE
    norm = shtns.sht_schmidt
    mmax = int( np.sign(M.m) )
    mres = max(1,M.m)
    sh   = shtns.sht( lmax2, mmax=mmax, mres=mres, norm=norm, nthreads=nthreads )
    M.sh = sh

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

        return Q,S,T,ur,utheta,uphi
    else:
        return Q,S,T


def spec2spat_scal(M,ut,par,chx,a,b,vsymm,nthreads,transform=True):

    # Rearrange and separate poloidal and toroidal parts

    Plj0 = a + 1j*b
    lm1  = M.lmax-M.m+1
    Plj0  = np.reshape(Plj0,(int(lm1/2),ut.N1))

    Plj = np.zeros((int(lm1/2),par.N),dtype=complex)

    if M.ricb == 0 :
        iP = (M.m + 1 - ut.s)%2
        iT = (M.m + ut.s)%2
        for k in np.arange(int(lm1/2)) :
            Plj[k,iP::2] = Plj0[k,:]
    else :
        Plj = Plj0

    # init arrays
    Plr  = np.zeros( (lm1, M.nr), dtype=complex )

    # These are the l values (ll) and indices (idp,idt)
    s = int(vsymm*0.5+0.5) # s=0 if antisymm, s=1 if symm

    if M.m > 0:
        idp = np.arange( 1-s, lm1, 2)
    elif M.m == 0:
        idp = np.arange( s  , lm1, 2)

    # populate Plr and Tlr
    Plr[idp,:] = np.matmul( Plj, chx.T)

    # (this is the SHTns way with m=mres when m is not zero)

    lmax2 = int( M.lmax + 1 - np.sign(M.m) )  # the true max value of l
    nlm = ( np.sign(M.m)+1 ) * (lmax2+1) - M.m
    Q = np.zeros([M.nr, nlm], dtype=complex)

    if M.m == 0 :  #pad with zeros for the l=0 component
        ql = np.r_[ np.zeros((1,M.nr)) ,Plr ]
    else :
        ql = Plr

    Q[:, np.sign(M.m)*(lmax2+1):] = ql.T
    M.ell = np.arange(M.m,M.lmax+1)
    norm = shtns.sht_schmidt
    mmax = int( np.sign(M.m) )
    mres = max(1,M.m)
    sh   = shtns.sht( lmax2, mmax=mmax, mres=mres, norm=norm, nthreads=nthreads )
    M.sh = sh

    if transform:
        # SHTns init
        #norm = shtns.sht_schmidt | shtns.SHT_NO_CS_PHASE
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

    l   = M.sh.l
    m   = M.sh.m
    r   = M.r
    Slm = M.S[0,:]
    Tlm = M.T[0,:]

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

    l   = M.sh.l
    m   = M.sh.m
    r   = M.r
    Qlm = M.Q[0,:]
    Slm = M.S[0,:]
    Tlm = M.T[0,:]

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
    torq_tor = np.sum(torq_conlm)

    return torq_rad, torq_con, torq_tor

