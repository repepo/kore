import numpy as np
import scipy.sparse as ss

import utils as ut


def expand_sol(sol, vsymm, ricb, m, lmax, N):
    '''
    Expands the ricb=0 solution with ut.N1 coeffs to have full N coeffs,
    filling with zeros according to the equatorial symmetry.
    vsymm=-1 for equatorially antisymmetric, vsymm=1 for symmetric
    '''

    if ricb == 0:

        lm1 = lmax - m + 1
        N1 = int(N / 2) * int(1 + np.sign(ricb)) + int((N % 2) * np.sign(ricb))
        n = int(N1*(lmax-m+1)/2)
        scalar_field = np.size(sol) == n

        # separate poloidal and toroidal coeffs
        P0 = sol[0: n]
        if not scalar_field:
            T0 = sol[n: 2 * n]

        # these are the cheb coefficients, reorganized
        Plj0 = np.reshape(P0, (int(lm1 / 2), N1))
        if not scalar_field:
            Tlj0 = np.reshape(T0, (int(lm1 / 2), N1))

        # create new arrays
        Plj = np.zeros((int(lm1 / 2), N), dtype=complex)
        if not scalar_field:
            Tlj = np.zeros((int(lm1 / 2), N), dtype=complex)

        # assign according to symmetry
        s = int((vsymm + 1) / 2)  # s=0 if vsymm=-1, s=1 if vsymm=1
        iP = (m + 1 - s) % 2  # even/odd Cheb polynomial for poloidals according to the parity of m+1-s
        iT = (m + s) % 2
        for k in np.arange(int(lm1 / 2)):
            Plj[k, iP::2] = Plj0[k, :]
            if not scalar_field:
                Tlj[k, iT::2] = Tlj0[k, :]

        # rebuild solution vector
        P2 = np.ravel(Plj)
        if not scalar_field:
            T2 = np.ravel(Tlj)
            out = np.r_[P2, T2]
        else:
            out = np.r_[P2]

    else:
        out = sol

    return out

def vector_field(a0, b0, vsymm, ricb, rcmb, m, lmax, N, n0, nr, chx, r):
    # initialize indices
    ll0 = ut.ell(m, lmax, vsymm)
    llpol = ll0[0]
    lltor = ll0[1]
    ll = ll0[2]

    # expand solution in case ricb == 0
    aib = expand_sol(a0 + 1j * b0, vsymm, ricb, m, lmax, N)
    a = np.real(aib)
    b = np.imag(aib)

    # rearrange and separate poloidal and toroidal parts
    Plj0 = a[:n0] + 1j * b[:n0]  # N elements on each l block
    Tlj0 = a[n0:n0 + n0] + 1j * b[n0:n0 + n0]  # N elements on each l block
    lm1 = lmax - m + 1

    Plj = np.reshape(Plj0, (int(lm1 / 2), N))
    Tlj = np.reshape(Tlj0, (int(lm1 / 2), N))

    d1Plj = np.zeros(np.shape(Plj), dtype=complex)
    d2Plj = np.zeros(np.shape(Plj), dtype=complex)
    d3Plj = np.zeros(np.shape(Plj), dtype=complex)

    d1Tlj = np.zeros(np.shape(Tlj), dtype=complex)
    d2Tlj = np.zeros(np.shape(Tlj), dtype=complex)

    # initialize arrays
    P0 = np.zeros((int((lmax - m + 1) / 2), nr), dtype=complex)
    P1 = np.zeros((int((lmax - m + 1) / 2), nr), dtype=complex)
    P2 = np.zeros((int((lmax - m + 1) / 2), nr), dtype=complex)
    P3 = np.zeros((int((lmax - m + 1) / 2), nr), dtype=complex)

    T0 = np.zeros((int((lmax - m + 1) / 2), nr), dtype=complex)
    T1 = np.zeros((int((lmax - m + 1) / 2), nr), dtype=complex)
    T2 = np.zeros((int((lmax - m + 1) / 2), nr), dtype=complex)

    Q0 = np.zeros((int((lmax - m + 1) / 2), nr), dtype=complex)
    S0 = np.zeros((int((lmax - m + 1) / 2), nr), dtype=complex)

    # populate matrices
    np.matmul(Plj, chx.T, P0)
    np.matmul(Tlj, chx.T, T0)

    # compute derivative Plj
    for k in range(np.size(llpol)):
        d1Plj[k, :] = ut.Dcheb(Plj[k, :], ricb, rcmb)
        d2Plj[k, :] = ut.Dcheb(d1Plj[k, :], ricb, rcmb)
        d3Plj[k, :] = ut.Dcheb(d2Plj[k, :], ricb, rcmb)
    np.matmul(d1Plj, chx.T, P1)
    np.matmul(d2Plj, chx.T, P2)
    np.matmul(d3Plj, chx.T, P3)

    # compute derivatives Tlj
    for k in range(np.size(lltor)):
        d1Tlj[k, :] = ut.Dcheb(Tlj[k, :], ricb, rcmb)
        d2Tlj[k, :] = ut.Dcheb(d1Tlj[k, :], ricb, rcmb)
    np.matmul(d1Tlj, chx.T, T1)
    np.matmul(d2Tlj, chx.T, T2)

    # compute multiplications
    rI = ss.diags(r ** -1, 0)
    lI = ss.diags(llpol * (llpol + 1), 0)
    lI_inv = ss.diags(1 / (llpol * (llpol + 1)), 0)

    # compute radial, consoidal and toroidal scalar feilds
    Q0 = lI * P0 * rI
    Q1 = (lI * P1 - Q0) * rI
    Q2 = (lI * P2 - 2 * Q1) * rI

    S0 = P1 + P0 * rI
    S1 = P2 + lI_inv * Q1
    S2 = P3 + lI_inv * Q2

    return Q0, Q1, Q2, S0, S1, S2, T0, T1, T2

def scalar_field(a0, b0, vsymm, ricb, rcmb, m, lmax, N, n0, nr, chx):
    # initialize indices
    ll0 = ut.ell(m, lmax, vsymm)
    llpol = ll0[0]

    # expand solution in case ricb == 0
    aib = expand_sol(a0 + 1j * b0, vsymm, ricb, m, lmax, N)
    a = np.real(aib)
    b = np.imag(aib)

    # rearrange and separate poloidal and toroidal parts
    Hk0 = a[0:n0] + 1j*b[0:n0]
    lm1 = lmax - m + 1

    Hk = np.reshape(Hk0, (int(lm1/2), N))

    d1Hk = np.zeros(np.shape(Hk), dtype=complex)
    d2Hk = np.zeros(np.shape(Hk), dtype=complex)

    # initialize solution arrays
    H0 = np.zeros((int((lmax-m+1)/2), nr), dtype=complex)
    H1 = np.zeros((int((lmax-m+1)/2), nr), dtype=complex)
    H2 = np.zeros((int((lmax-m+1)/2), nr), dtype=complex)

    # populate matrices
    np.matmul(Hk, chx.T, H0)

    for k in range(np.size(llpol)):
        d1Hk[k, :] = ut.Dcheb(Hk[k, :], ricb, rcmb)
        d2Hk[k, :] = ut.Dcheb(d1Hk[k, :], ricb, rcmb)
    np.matmul(d1Hk, chx.T, H1)
    np.matmul(d2Hk, chx.T, H2)

    return H0, H1, H2


def lorentz_pol(l, k, r, p, Q0, S0, S1, T0, T1):
    out_rad = np.zeros_like(r, dtype='complex128')
    out_con = np.zeros_like(r, dtype='complex128')

    lp = np.shape(Q0)[0]

    m    = int(p[5])
    beta = p[16]
    B0_l = int(p[17])
    ricb = p[7]

    B0list = ['axial', 'dipole', 'G21 dipole', 'Luo_S1', 'Luo_S2', 'FDM']
    B0 = B0list[int(p[15])]
    cnorm = p[27]

    h0 = ut.h0(r, B0, [beta, B0_l, ricb, 0])
    h1 = ut.h1(r, B0, [beta, B0_l, ricb, 0])
    h2 = ut.h2(r, B0, [beta, B0_l, ricb, 0])

    r_sqr = r**2

    qlm0 = Q0[k - 2, :]
    slm0 = S0[k - 2, :]
    slm1 = S1[k - 2, :]

    C_rad = 3 * (-2 + l) * np.sqrt((-1 + l - m) * (l - m) * (-1 + l + m) * (l + m)) / (3 + 4 * (-2 + l) * l)
    out_rad += C_rad*(-h0 * (qlm0/r_sqr + 5*slm0/r - slm1/r) + h2 * slm0 + h1 * (-qlm0/r + 3*slm0/r + slm1))

    C_con = 3 * np.sqrt((-1 + l - m) * (l - m) * (-1 + l + m) * (l + m)) / (3 * l + 4 * (-2 + l) * l ** 2)
    out_con += C_con*(-3*h0*l*qlm0/r_sqr + qlm0 * (2*h1/r + h2) + 3*h0 * (-2 + l) * (slm0/r_sqr + slm1/r))

    tlm0 = T0[k-1, :]
    tlm1 = T1[k-1, :]

    C_rad = 3j * m * np.sqrt(l ** 2 - m ** 2) / (2 * l - 1)
    out_rad += C_rad * (h0 * (-5 * tlm0 / r_sqr + tlm1 / r) + 3 * h1 * tlm0 / r + h2 * tlm0 + h1 * tlm1)

    C_con = 3j * m * np.sqrt(l ** 2 - m ** 2) / (l * (1 + l) * (-1 + 2 * l))
    out_con += C_con * (h0 * (6 + (-1 + l) * l) * tlm0 / r_sqr + h1 * (-1 + l) * l * tlm0 / r + 6 * h0 * tlm1 / r)

    qlm0 = Q0[k, :]
    slm0 = S0[k, :]
    slm1 = S1[k, :]

    C_rad = -3 * (l + l ** 2 - 3 * m ** 2) / (-3 + 4 * l * (l + 1))
    out_rad += C_rad * (-h0 * (qlm0/r_sqr + 5 * slm0/r_sqr - slm1/r) + h2*slm0 + h1 * (-qlm0/r + 3 * slm0/r + slm1))

    C_con = 3 * (l + l ** 2 - 3 * m ** 2) / (l * (1 + l) * (-3 + 4 * l * (1 + l)))
    out_con += C_con*(-2*h0*l*(1 + l) * qlm0/r_sqr + qlm0*(2*h1/r + h2) + 2*h0*(-3 + l + l**2) * (slm0/r_sqr + slm1/r))

    if k+1 < lp:
        tlm0 = T0[k + 1, :]
        tlm1 = T1[k + 1, :]

        C_rad = 3j * m * np.sqrt((l + 1 - m) * (l + 1 + m)) / (2 * l + 3)
        out_rad += C_rad * (h0 * (-5 * tlm0 / r_sqr + tlm1 / r) + 3 * h1 * tlm0 / r + h2 * tlm0 + h1 * tlm1)

        C_con = 3j * m * np.sqrt((1 + l - m) * (1 + l + m)) / (l * (1 + l) * (3 + 2 * l))
        out_con += C_con * (h0 * (8 + l * (3 + l)) * tlm0 / r_sqr + h1 * (1 + l) * (2 + l) * tlm0 / r + 6 * h0 * tlm1 / r)

    if k+2 < lp:
        qlm0 = Q0[k + 2, :]
        slm0 = S0[k + 2, :]
        slm1 = S1[k + 2, :]

        C_rad = -3 * (l + 3) * np.sqrt((1 + l - m) * (2 + l - m) * (1 + l + m) * (2 + l + m)) / ((2 * l + 3) * (2 * l + 5))
        out_rad += C_rad * (-h0 * (qlm0/r_sqr + 5 * slm0/r_sqr - slm1/r) + h2 * slm0 + h1 * (-qlm0/r + 3 * slm0/r + slm1))

        C_con = -3 * np.sqrt((1 + l - m) * (2 + l - m) * (1 + l + m) * (2 + l + m)) / ((1 + l) * (3 + 2 * l) * (5 + 2 * l))
        out_con += C_con * (3*h0*(1 + l) * qlm0/r_sqr + qlm0 * (2*h1/r + h2) - 3 * h0 * (3 + l) * (slm0/r_sqr + slm1/r))

    return out_rad*cnorm, out_con*cnorm


def lorentz_tor(l, k, r, p, Q0, S0, S1, T0, T1):
    out_tor = np.zeros_like(r, dtype='complex128')

    lt = np.shape(T0)[0]

    m    = int(p[5])
    beta = p[16]
    B0_l = int(p[17])
    ricb = p[7]

    B0list = ['axial', 'dipole', 'G21 dipole', 'Luo_S1', 'Luo_S2', 'FDM']
    B0 = B0list[int(p[15])]
    cnorm = p[27]

    h0 = ut.h0(r, B0, [beta, B0_l, ricb, 0])
    h1 = ut.h1(r, B0, [beta, B0_l, ricb, 0])
    h2 = ut.h2(r, B0, [beta, B0_l, ricb, 0])

    r_sqr = r**2

    tlm0 = T0[k-2, :]
    tlm1 = T1[k-2, :]

    C_tor = -3 * (-2 + l) * np.sqrt((-1 + l - m) * (l - m) * (-1 + l + m) * (l + m)) / (l * (3 + 4 * (-2 + l) * l))
    out_tor += C_tor * (h0 * (-4 + l) * tlm0 / r_sqr + h1 * (-1 + l) * tlm0 / r - 3 * h0 * tlm1 / r_sqr)

    qlm0 = Q0[k-1, :]
    slm0 = S0[k-1, :]
    slm1 = S1[k-1, :]

    C_tor = 3j * m * np.sqrt(l ** 2 - m ** 2) / (l * (1 + l) * (-1 + 2 * l))
    out_tor += C_tor * (qlm0 * (2 * h1 / r + h2) - 6 * h0 * (slm0 / r_sqr + slm1 / r))

    tlm0 = T0[k, :]
    tlm1 = T1[k, :]

    C_tor = 3 * (l + l ** 2 - 3 * m ** 2) / (l * (1 + l) * (-3 + 4 * l * (1 + l)))
    out_tor += C_tor * (h0 * (-6 + l + l ** 2) * tlm0/r_sqr - h1*l*(1 + l)*tlm0/r + 2*h0*(-3 + l + l**2) * tlm1/r)

    if k+1 < lt:
        qlm0 = Q0[k + 1, :]
        slm0 = S0[k + 1, :]
        slm1 = S1[k + 1, :]

        C_tor = 3j * m * np.sqrt((1 + l - m) * (1 + l + m)) / (l * (1 + l) * (3 + 2 * l))
        out_tor += C_tor * (qlm0 * (2 * h1 / r + h2) - 6 * h0 * (slm0 / r_sqr + slm1 / r))

    if k+2 < lt:
        tlm0 = T0[k + 2, :]
        tlm1 = T1[k + 2, :]

        C_tor = 3*(l + 3)*np.sqrt((1 + l - m)*(2 + l - m)*(1 + l + m)*(2 + l + m))/((1 + l)*(3 + 2 * l)*(5 + 2*l))
        out_tor += C_tor * (h0 * (l + 5) * tlm0 / r_sqr + h1 * (l + 2) * tlm0 / r + 3 * h0 * tlm1 / r)

    return cnorm * out_tor

def induction_pol(l, k, r, p, Q0, Q1, S0, S1, T0, T1):
    out_rad = np.zeros_like(r, dtype='complex128')
    out_con = np.zeros_like(r, dtype='complex128')

    lp = np.shape(Q0)[0]

    m    = int(p[5])
    beta = p[16]
    B0_l = int(p[17])
    ricb = p[7]

    B0list = ['axial', 'dipole', 'G21 dipole', 'Luo_S1', 'Luo_S2', 'FDM']
    B0 = B0list[int(p[15])]
    cnorm = p[27]

    h0 = ut.h0(r, B0, [beta, B0_l, ricb, 0])
    h1 = ut.h1(r, B0, [beta, B0_l, ricb, 0])
    h2 = ut.h2(r, B0, [beta, B0_l, ricb, 0])

    r_sqr = r**2

    qlm0 = Q0[k - 2, :]
    qlm1 = Q1[k - 2, :]
    slm0 = S0[k - 2, :]
    slm1 = S1[k - 2, :]

    out_rad += (3 * (1 + l) * np.sqrt((-1 + l - m) * (l - m) * (-1 + l + m) * (l + m)) * (
                -(h0 * qlm0) - h1 * qlm0 * r + 3 * h0 * (-2 + l) * slm0)) / ((3 + 4 * (-2 + l) * l) * r_sqr)

    out_con += (3 * np.sqrt((-1 + l - m) * (l - m) * (-1 + l + m) * (l + m)) * (
                -(qlm1 * (h0 + h1 * r)) - qlm0 * (2 * h1 + h2 * r) + 3 * h1 * (-2 + l) * slm0 + 3 * h0 * (
                    -2 + l) * slm1)) / (l * (3 + 4 * (-2 + l) * l) * r)

    tlm0 = T0[k - 1, :]
    tlm1 = T1[k - 1, :]

    out_rad += (18j * h0 * m * np.sqrt(l ** 2 - m ** 2) * tlm0) / ((-1 + 2 * l) * r_sqr)

    out_con += (18j * m * np.sqrt(l ** 2 - m ** 2) * (h1 * tlm0 + h0 * tlm1)) / (l * (-1 + l + 2 * l ** 2) * r)

    qlm0 = Q0[k, :]
    qlm1 = Q1[k, :]
    slm0 = S0[k, :]
    slm1 = S1[k, :]

    out_rad += (-3 * (l + l ** 2 - 3 * m ** 2) * (h1 * qlm0 * r + h0 * (qlm0 - 2 * (-3 + l + l ** 2) * slm0))) / (
                (-3 + 4 * l * (1 + l)) * r_sqr)

    out_con += (-3 * (l + l ** 2 - 3 * m ** 2) * (
                qlm1 * (h0 + h1 * r) + qlm0 * (2 * h1 + h2 * r) - 2 * h1 * (-3 + l + l ** 2) * slm0 - 2 * h0 * (
                    -3 + l + l ** 2) * slm1)) / (l * (1 + l) * (-3 + 4 * l * (1 + l)) * r)

    if k + 1 < lp:
        tlm0 = T0[k + 1, :]
        tlm1 = T1[k + 1, :]

        out_rad += (18j*h0*m*np.sqrt((1 + l - m)*(1 + l + m))*tlm0)/((3 + 2*l)*r_sqr)

        out_con += (18j*m*np.sqrt((1 + l - m)*(1 + l + m))*(h1*tlm0 + h0*tlm1))/(l*(1 + l)*(3 + 2*l)*r)

    if k + 2 < lp:
        qlm0 = Q0[k + 2, :]
        qlm1 = Q1[k + 2, :]
        slm0 = S0[k + 2, :]
        slm1 = S1[k + 2, :]

        out_rad += (3*l*np.sqrt((1 + l - m)*(2 + l - m)*(1 + l + m)*(2 + l + m))*(h1*qlm0*r + h0*(qlm0 + 3*(3 + l)*slm0)))/((3 + 2*l)*(5 + 2*l)*r_sqr)

        out_con += (3*np.sqrt((1 + l - m)*(2 + l - m)*(1 + l + m)*(2 + l + m))*(h2*qlm0*r + h1*(2*qlm0 + qlm1*r + 3*(3 + l)*slm0) + h0*(qlm1 + 3*(3 + l)*slm1)))/((1 + l)*(3 + 2*l)*(5 + 2*l)*r)

    return out_rad*cnorm, out_con*cnorm

def induction_tor(l, k, r, p, Q0, Q1, S0, S1, T0, T1):
    out_tor = np.zeros_like(r, dtype='complex128')

    lt = np.shape(T0)[0]

    m    = int(p[5])
    beta = p[16]
    B0_l = int(p[17])
    ricb = p[7]

    B0list = ['axial', 'dipole', 'G21 dipole', 'Luo_S1', 'Luo_S2', 'FDM']
    B0 = B0list[int(p[15])]
    cnorm = p[27]

    h0 = ut.h0(r, B0, [beta, B0_l, ricb, 0])
    h1 = ut.h1(r, B0, [beta, B0_l, ricb, 0])
    h2 = ut.h2(r, B0, [beta, B0_l, ricb, 0])

    r_sqr = r**2

    tlm0 = T0[k - 2, :]
    tlm1 = T1[k - 2, :]

    out_tor += (-3 * (-2 + l) * np.sqrt((-1 + l - m) * (l - m) * (-1 + l + m) * (l + m)) * (
                h0 * l * tlm0 + h1 * (-3 + l) * r * tlm0 - 3 * h0 * r * tlm1)) / (l * (3 + 4 * (-2 + l) * l) * r_sqr)

    qlm0 = Q0[k - 1, :]
    qlm1 = Q1[k - 1, :]
    slm0 = S0[k - 1, :]
    slm1 = S1[k - 1, :]

    out_tor += (3j * m * np.sqrt(l ** 2 - m ** 2) * (
                -(qlm0 * r * (2 * h1 + h2 * r)) + h0 * l * (1 + l) * slm0 + h1 * r * (
                    -(qlm1 * r) + (-6 + l + l ** 2) * slm0) - h0 * r * (qlm1 + 6 * slm1))) / (
                           l * (1 + l) * (-1 + 2 * l) * r_sqr)

    tlm0 = T0[k, :]
    tlm1 = T1[k, :]

    out_tor += (3 * (l + l ** 2 - 3 * m ** 2) * (
                h0 * l * (1 + l) * tlm0 + 3 * h1 * (-2 + l + l ** 2) * r * tlm0 + 2 * h0 * (
                    -3 + l + l ** 2) * r * tlm1)) / (l * (1 + l) * (-3 + 4 * l * (1 + l)) * r_sqr)

    if k+1 < lt:
        qlm0 = Q0[k + 1, :]
        qlm1 = Q1[k + 1, :]
        slm0 = S0[k + 1, :]
        slm1 = S1[k + 1, :]

        out_tor += (3j * m * np.sqrt((1 + l - m) * (1 + l + m)) * (
                    -(qlm0 * r * (2 * h1 + h2 * r)) + h0 * l * (1 + l) * slm0 + h1 * r * (
                        -(qlm1 * r) + (-6 + l + l ** 2) * slm0) - h0 * r * (qlm1 + 6 * slm1))) / (
                               l * (1 + l) * (3 + 2 * l) * r_sqr)

    if k+2 < lt:
        tlm0 = T0[k + 2, :]
        tlm1 = T1[k + 2, :]

        out_tor += (3 * (3 + l) * np.sqrt((1 + l - m) * (2 + l - m) * (1 + l + m) * (2 + l + m)) * (
                    h1 * (4 + l) * r * tlm0 + h0 * (tlm0 + l * tlm0 + 3 * r * tlm1))) / (
                               (1 + l) * (3 + 2 * l) * (5 + 2 * l) * r_sqr)

    return cnorm * out_tor