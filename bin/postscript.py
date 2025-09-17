#!/usr/bin/env python3
'''
kore post-postprocessing script

Usage:
> ./bin/postscript.py ncpus
'''

import sys

import numpy as np
import numpy.polynomial.chebyshev as ch

import scipy.sparse as ss
import scipy.special as scsp

from utils4pp import xcheb
from utils import Dcheb
from utils import ell

def main(ncpus):
    params = np.loadtxt('params.dat')
    if np.ndim(params) == 1:
        params = np.array([params])
    success = np.shape(params)[0]

    coeffs = np.zeros((success, 2))

    print('\n  ★     Transmission coefficient       Mac-like coefficient ')
    print(' ‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ')

    for i in range(success):
        # --- INITIALIZATION ---
        # init arguments
        nr = 500
        nz = 500
        s = 0.65
        theta = np.pi/2

        # load parameter data from solve.py generated files
        p = params[i, :]

        Ek = p[4]
        m = int(p[5])
        symm = int(p[6])

        ricb = p[7]
        rcmb = 1

        lmax = int(p[47])
        N = int(p[46])
        n0 = int(N * (lmax - m + 1) / 2)

        # read flow field from disk
        a0 = np.loadtxt("real_flow.field", usecols=i)
        b0 = np.loadtxt("imag_flow.field", usecols=i)
        vsymm = symm

        # initialise indices
        ll0 = ell(m, lmax, vsymm)
        llpol = ll0[0]
        lltor = ll0[1]
        ll = ll0[2]

        # --- COMPUTATION ---
        # expand solution in case ricb == 0
        # expand solution in case ricb == 0
        aib = expand_sol(a0 + 1j * b0, vsymm, ricb, m, lmax, N)
        a = np.real(aib)
        b = np.imag(aib)

        Plj0 = a[:n0] + 1j * b[:n0]  # N elements on each l block
        Tlj0 = a[n0:n0 + n0] + 1j * b[n0:n0 + n0]  # N elements on each l block
        lm1 = lmax - m + 1

        Plj = np.reshape(Plj0, (int(lm1 / 2), N))
        Tlj = np.reshape(Tlj0, (int(lm1 / 2), N))
        dPlj = np.zeros(np.shape(Plj), dtype=complex)

        z, up, us = plot_cyl_line(s, nz, N, m, vsymm, ricb, rcmb, Plj, Tlj, dPlj, lm1, llpol, lltor, ll)

        r, urad, utta, uphi = plot_sph_line(theta, nr, N, m, vsymm, ricb, rcmb, Plj, Tlj, dPlj, lm1, llpol, lltor, ll)

        up_masked = np.ma.array(up, mask=s ** 2 + z ** 2 > 1)
        us_masked = np.ma.array(us, mask=s ** 2 + z ** 2 > 1)
        z_masked = np.ma.array(z, mask=s ** 2 + z ** 2 > 1)

        ind = np.argmax(z_masked[z_masked < np.sqrt(1 - s**2) - np.sqrt(Ek)*(rcmb - ricb)])
        res1 = (up_masked ** 2 + us_masked ** 2)

        coeffs[i, 0] = res1[ind]/res1[0]

        res2 = urad ** 2 + uphi ** 2 + utta ** 2

        coeffs[i, 1] = np.max(res2[r > 0.96]) / np.max(res2[r < 0.9])

        print(' {:2d}         {: 12.7f}                  {: 12.7f}'.format(i, coeffs[i,0], coeffs[i,1]))
    print(' ‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ')

    with open('coeffs.dat', 'ab') as dcfs:
        np.savetxt(dcfs, coeffs)

    return 0

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

def plot_cyl_line(s, nz, N, m, vsymm, ricb, rcmb, Plj, Tlj, dPlj, lm1, llpol, lltor, ll):
    # --- INITIALIZATION ---
    # set up the evenly spaced cylindrical grid
    z = np.linspace(0,1, nz+1)
    z = z[:-1]

    # compute spherical grid
    r = np.sqrt(s**2 + z**2)
    r[r > 1] = 1
    theta = np.arctan2(s, z)

    x = xcheb(r, 0, 1)

    # matrix with Chebyshev polynomials at every x point for all degrees:
    chx = ch.chebvander(x, N-1) # this matrix has nr rows and N-1 cols

    # --- COMPUTATION ---
    # initialize arrays
    Plr = np.zeros((int((lm1)/2), nz),dtype=complex)
    dP  = np.zeros((int((lm1)/2), nz),dtype=complex)
    rP  = np.zeros((int((lm1)/2), nz),dtype=complex)
    Qlr = np.zeros((int((lm1)/2), nz),dtype=complex)
    Slr = np.zeros((int((lm1)/2), nz),dtype=complex)
    Tlr = np.zeros((int((lm1)/2), nz),dtype=complex)

    # populate Plr and Tlr
    np.matmul(Plj, chx.T, Plr)
    np.matmul(Tlj, chx.T, Tlr)

    # compute derivative Plj
    for k in range(np.size(llpol)):
        dPlj[k,:] = Dcheb(Plj[k,:], ricb, rcmb)
    np.matmul(dPlj, chx.T, dP)

    # compute multiplications
    rI = ss.diags(r ** -1, 0)
    lI = ss.diags(llpol * (llpol + 1), 0)

    rP = Plr * rI
    Qlr = lI * rP
    Slr = rP + dP

    # initialize solution arrays
    ur = np.zeros(nz)
    ut = np.zeros(nz)
    up = np.zeros(nz)
    us = np.zeros(nz)
    uz = np.zeros(nz)

    # initialize spherical harmonics coefficients.
    clm = np.zeros((lm1+1,1))
    Ylm_cache = np.zeros((lm1+1, nz),dtype=complex)
    for i,l in enumerate(ll):
        clm[i] = np.sqrt((l-m)*(l+m))
        Ylm_cache[i, :] = np.sqrt(4 * np.pi / (2 * l + 1)) * scsp.sph_harm(m, l, 0, theta)

    # start index for l. Do not confuse with indices for the Cheb expansion!
    sy = int( vsymm*0.5 + 0.5 ) # sy=0 if antisymm, sy=1 if symm
    idP = (np.sign(m)+sy  )%2
    idT = (np.sign(m)+sy+1)%2
    plx = idP+lm1
    tlx = idT+lm1

    sinT = np.sin(theta)
    tanT = np.tan(theta)
    cosT = np.cos(theta)

    for kz in range(nz):
        ylm = np.r_[Ylm_cache[:, kz], 0]
        ur[kz] = np.abs(Qlr[:, kz] @ ylm[idP:plx:2])

        tmp1 = -(llpol + 1) * Slr[:, kz] / tanT[kz] @ ylm[idP:plx:2]
        tmp2 = clm[idP + 1:plx + 1:2, 0] * Slr[:, kz] / sinT[kz] @ ylm[idP + 1:plx + 1:2]
        tmp3 = 1j * m * Tlr[:, kz] / sinT[kz] @ ylm[idT:tlx:2]
        ut[kz] = np.abs(tmp1 + tmp2 + tmp3)

        tmp1 = (lltor + 1) * Tlr[:, kz] / tanT[kz] @ ylm[idT:tlx:2]
        tmp2 = -clm[idT + 1:tlx + 1:2, 0] * Tlr[:, kz] / sinT[kz] @ ylm[idT + 1:tlx + 1:2]
        tmp3 = 1j * m * Slr[:, kz] / sinT[kz] @ ylm[idP:plx:2]
        up[kz] = np.abs(tmp1 + tmp2 + tmp3)

        uz[kz] = ur[kz] * cosT[kz] - ut[kz] * sinT[kz]
        us[kz] = ur[kz] * sinT[kz] + ut[kz] * cosT[kz]

    return z, up, us

def plot_sph_line(theta, nr, N, m, vsymm, ricb, rcmb, Plj, Tlj, dPlj, lm1, llpol, lltor, ll):
    # --- INITIALIZATION ---
    # set up the evenly spaced radial grid
    r = np.linspace(0,1, nr+1)
    r = r[1:]

    x = xcheb(r, 0, 1)

    # matrix with Chebyshev polynomials at every x point for all degrees:
    chx = ch.chebvander(x, N-1) # this matrix has nr rows and N-1 cols

    # --- COMPUTATION ---
    # initialize arrays
    Plr = np.zeros((int((lm1)/2), nr),dtype=complex)
    dP  = np.zeros((int((lm1)/2), nr),dtype=complex)
    rP  = np.zeros((int((lm1)/2), nr),dtype=complex)
    Qlr = np.zeros((int((lm1)/2), nr),dtype=complex)
    Slr = np.zeros((int((lm1)/2), nr),dtype=complex)
    Tlr = np.zeros((int((lm1)/2), nr),dtype=complex)

    # populate Plr and Tlr
    np.matmul(Plj, chx.T, Plr)
    np.matmul(Tlj, chx.T, Tlr)

    # compute derivative Plj
    for k in range(np.size(llpol)):
        dPlj[k,:] = Dcheb(Plj[k,:], ricb, rcmb)
    np.matmul(dPlj, chx.T, dP)

    # compute multiplications
    rI = ss.diags(r ** -1, 0)
    lI = ss.diags(llpol * (llpol + 1), 0)

    rP = Plr * rI
    Qlr = lI * rP
    Slr = rP + dP

    # initialize solution arrays
    ur = np.zeros(nr)
    ut = np.zeros(nr)
    up = np.zeros(nr)

    # initialize spherical harmonics coefficients.
    clm = np.zeros((lm1+1,1))
    Ylm_cache = np.zeros(lm1+1,dtype=complex)
    for i,l in enumerate(ll):
        clm[i] = np.sqrt((l-m)*(l+m))
        Ylm_cache[i] = np.sqrt(4 * np.pi / (2 * l + 1)) * scsp.sph_harm(m, l, 0, theta)
    ylm = np.r_[Ylm_cache, 0]

    # start index for l. Do not confuse with indices for the Cheb expansion!
    sy = int( vsymm*0.5 + 0.5 ) # sy=0 if antisymm, sy=1 if symm
    idP = (np.sign(m)+sy  )%2
    idT = (np.sign(m)+sy+1)%2
    plx = idP+lm1
    tlx = idT+lm1

    sinT = np.sin(theta)
    tanT = np.tan(theta)

    for kr in range(nr):
        ur[kr] = np.abs(Qlr[:, kr] @ ylm[idP:plx:2])

        tmp1 = -(llpol + 1) * Slr[:, kr] / tanT @ ylm[idP:plx:2]
        tmp2 = clm[idP + 1:plx + 1:2, 0] * Slr[:, kr] / sinT @ ylm[idP + 1:plx + 1:2]
        tmp3 = 1j * m * Tlr[:, kr] / sinT @ ylm[idT:tlx:2]
        ut[kr] = np.abs(tmp1 + tmp2 + tmp3)

        tmp1 = (lltor + 1) * Tlr[:, kr] / tanT @ ylm[idT:tlx:2]
        tmp2 = -clm[idT + 1:tlx + 1:2, 0] * Tlr[:, kr] / sinT @ ylm[idT + 1:tlx + 1:2]
        tmp3 = 1j * m * Slr[:, kr] / sinT @ ylm[idP:plx:2]
        up[kr] = np.abs(tmp1 + tmp2 + tmp3)

    return r, ur, ut, up



if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
