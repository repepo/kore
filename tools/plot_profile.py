import numpy as np
import scipy.sparse as ss
import sys
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as ch
import scipy.integrate as integrate
from os.path import exists

import utils as ut
import utils4pp as upp
import utils4fig as ufig

# colorblind safe
plt.style.use('tableau-colorblind10')

# latex fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

'''
Computes the radial profile of the energies, forces and dissipation
Use as: 

python3 path/to/plot_profile.py solnum r0 r1 field

solnum : solution number
r0     : starting radius
r1     : final radius
field  : whether to plot flow velocity, magnetic field, temperature or composition

Requires the PYTHONPATH to include the ../bin folder:

export PYTHONPATH=$PYTHONPATH:/path/to/bin
'''

# --- INITIALIZATION ---
# load input parameters
solnum = int(sys.argv[1])
r0     = float(sys.argv[2])
r1     = float(sys.argv[3])
field  = sys.argv[4]

# load parameter data from solve.py generated files
p = np.loadtxt('params.dat')
if np.ndim(p) > 1:
    p = np.loadtxt('params.dat')[solnum, :]

m     = int(p[5])
symm  = int(p[6])

ricb  = p[7]
rcmb  = 1

lmax  = int(p[47])
N     = int(p[46])
n0    = int(N*(lmax-m+1)/2)

nr    = N-1

omgtau  = p[44]
Ek      = p[4]
Em      = p[25]
Et      = p[28]
Ec      = p[36]
Le2     = p[26]
thm_BV2 = p[30]
cmp_BV2 = p[38]

if field == 't':
    heating = int(p[29])
    rc = p[31]
    h = p[32]
    rsy = p[33]
    args = [rc, h, rsy]

if field == 'c':
    heating = int(p[37])
    rc = p[39]
    h = p[40]
    rsy = p[41]
    args = [rc, h, rsy]

# set up the evenly spaced radial grid
r = np.linspace(ricb,rcmb,nr)

if ricb == 0:
    r = r[1:]
    nr = nr - 1
r_sqr = r**2
r_inv = ss.diags(1/r, 0)
x = upp.xcheb(r,ricb,rcmb)

# matrix with Chebyshev polynomials at every x point for all degrees:
chx = ch.chebvander(x, N-1) # this matrix has nr rows and N-1 cols

# create useful boolean variables
magnetic = (p[1] == 1)
thermal = (p[2] == 1)
composition = (p[3] == 1)

# set-up figure
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.set_xlabel(r'$r$', size=12)

# --- COMPUTATION ---
# read flow field data from disk
ru = np.loadtxt('real_flow.field', usecols=solnum)
iu = np.loadtxt('imag_flow.field', usecols=solnum)

Q0, Q1, Q2, S0, S1, S2, T0, T1, T2 = ufig.vector_field(ru, iu, symm, ricb, rcmb, m, lmax, N, n0, nr, chx, r)

if field == "u":
    # initialize indices
    ll0 = ut.ell(m, lmax, symm)
    llpol = ll0[0]
    lltor = ll0[1]
    ll = ll0[2]

    # prepare solution arrays
    ke_pol = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
    ke_tor = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)

    wl_pol = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
    wl_tor = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
    wt_pol = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
    wc_pol = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)

    dk_pol = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
    dk_tor = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)

    if magnetic:
        # read magnetic field data from disk
        rb = np.loadtxt('real_magnetic.field', usecols=solnum)
        ib = np.loadtxt('imag_magnetic.field', usecols=solnum)

        B0_type = int(p[15])
        if B0_type in np.arange(4):
            B0_symm = -1
        elif B0_type == 4:
            B0_symm = 1
        elif B0_type == 5:
            B0_l = p[17]
            B0_symm = int((-1) ** (B0_l))
        bsymm = symm*B0_symm

        E0, E1, E2, F0, F1, F2, G0, G1, G2 = ufig.vector_field(rb, ib, bsymm, ricb, rcmb, m, lmax, N, n0, nr, chx, r)

    if thermal:
        # read thermal field data from disk
        rt = np.loadtxt('real_temperature.field', usecols=solnum)
        it = np.loadtxt('imag_temperature.field', usecols=solnum)

        H0, H1, H2 = ufig.scalar_field(rt, it, symm, ricb, rcmb, m, lmax, N, n0, nr, chx)

    if composition:
        # read compositional field data from disk
        rc = np.loadtxt('real_composition.field', usecols=solnum)
        ic = np.loadtxt('imag_composition.field', usecols=solnum)

        C0, C1, C2 = ufig.scalar_field(rc, ic, symm, ricb, rcmb, lmax, N, n0, nr, chx)

    # compute poloidal components
    for k, l in enumerate(llpol):
        qlm0 = Q0[k, :]
        qlm1 = Q1[k, :]
        qlm2 = Q2[k, :]
        slm0 = S0[k, :]
        slm1 = S1[k, :]
        slm2 = S2[k, :]

        L = l * (l + 1)
        f0 = 4 * np.pi / (2 * l + 1)

        # compute poloidal kinetic energy
        f1 = r_sqr * np.absolute(qlm0) ** 2
        f2 = r_sqr * L * np.absolute(slm0) ** 2
        ke_pol[k, :] = f0*(f1 + f2)

        # compute poloidal kinetic dissipation
        f1 = L * r_sqr * np.conj(slm0) * slm2
        f2 = 2 * r * L * np.conj(slm0) * slm1
        f3 = -(L ** 2) * (np.conj(slm0) * slm0) - (l ** 2 + l + 2) * (np.conj(qlm0) * qlm0)
        f4 = 2 * r * np.conj(qlm0) * qlm1 + r * np.conj(qlm0) * qlm2
        f5 = 2 * L * (np.conj(qlm0) * slm0 + qlm0 * np.conj(slm0))
        dk_pol[k, :] = 2 * np.real(f0 * (f1 + f2 + f3 + f4 + f5))

        if magnetic:
            qlmb, slmb = ufig.lorentz_pol(l, k, r, p, E0, F0, F1, G0, G1)

            # compute poloidal lorentz force
            f1 = r_sqr * 2 * np.real(qlm0 * np.conj(qlmb))
            f2 = r_sqr * L * 2 * np.real(slm0 * np.conj(slmb))
            wl_pol[k, :] = f0*(f1 + f2)

        if thermal:
            hlm0 = H0[k, :]
            plm0 = Q0[k, :] * r / L

            # compute thermal buoyancy force
            f1 = r_sqr * L * 2 * np.real(np.conj(plm0) * hlm0)
            wt_pol[k, :] = f0*f1

        if composition:
            clm0 = C0[k, :]
            plm0 = Q0[k, :] * r / L

            # compute compositional buoyancy force
            f1 = r_sqr * L * 2 * np.real(np.conj(plm0) * clm0)
            wt_pol[k, :] = f0 * f1

    # compute toroidal components
    for k, l in enumerate(lltor):
        tlm0 = T0[k, :]
        tlm1 = T1[k, :]
        tlm2 = T2[k, :]

        L = l * (l + 1)
        f0 = 4 * np.pi / (2 * l + 1)

        # compute toroidal kinetic energy
        f1 = r_sqr * L * np.absolute(tlm0) ** 2
        ke_tor[k, :] = f0 * f1

        # compute toroidal kinetic energy dissipation
        f1 = L * r_sqr * np.conj(tlm0) * tlm2
        f2 = 2 * r * L * np.conj(tlm0) * tlm1
        f3 = -(L ** 2) * (np.conj(tlm0) * tlm0)
        dk_tor[k, :] = 2 * np.real(f0 * (f1 + f2 + f3))

        if magnetic:
            tlmb = ufig.lorentz_tor(l, k, r, p, E0, F0, F1, G0, G1)

            # compute toroidal lorentz force
            f1 = r_sqr * L * 2 * np.real(tlm0 * np.conj(tlmb))
            wl_tor[k, :] = f0 * f1


    ke = np.real(sum(ke_pol + ke_tor, 0))
    dkin = omgtau * Ek * np.real(sum(dk_pol + dk_tor, 0))

    ax.semilogy(r, ke, label=r'KE')
    ax.semilogy(r, abs(dkin), label=r'$|\mathcal{D}_\mathrm{kin}|$')

    if magnetic:
        wlor = omgtau**2 * Le2 * np.real(sum(wl_pol+wl_tor,0))
        ax.semilogy(r, abs(wlor), label=r'$|\mathcal{W}_\mathrm{lor}|$')

    if thermal:
        wtmp = omgtau ** 2 * thm_BV2 * np.real(sum(wt_pol, 0))
        ax.semilogy(r, abs(wtmp), label=r'$|\mathcal{W}_\mathrm{tmp}|$')

    if composition:
        wcmp = omgtau ** 2 * cmp_BV2 * np.real(sum(wc_pol, 0))
        ax.semilogy(r, abs(wcmp), label=r'$|\mathcal{W}_\mathrm{cmp}|$')

if field == "b":
    # prepare solution arrays
    me_pol = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
    me_tor = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)

    wi_pol = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
    wi_tor = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)

    do_pol = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)
    do_tor = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)

    # read magnetic field data from disk
    rb = np.loadtxt('real_magnetic.field', usecols=solnum)
    ib = np.loadtxt('imag_magnetic.field', usecols=solnum)

    B0_type = int(p[15])
    if B0_type in np.arange(4):
        B0_symm = -1
    elif B0_type == 4:
        B0_symm = 1
    elif B0_type == 5:
        B0_l = p[17]
        B0_symm = int((-1) ** (B0_l))
    bsymm = symm * B0_symm

    # initialize indices
    ll0 = ut.ell(m, lmax, bsymm)
    llpol = ll0[0]
    lltor = ll0[1]
    ll = ll0[2]

    E0, E1, E2, F0, F1, F2, G0, G1, G2 = ufig.vector_field(rb, ib, bsymm, ricb, rcmb, m, lmax, N, n0, nr, chx, r)

    # compute poloidal components
    for k, l in enumerate(llpol):
        elm0 = E0[k, :]
        elm1 = E1[k, :]
        elm2 = E2[k, :]
        flm0 = F0[k, :]
        flm1 = F1[k, :]
        flm2 = F2[k, :]

        qlmi, slmi = ufig.induction_pol(l, k, r, p, E0, E1, F0, F1, G0, G1)

        L = l * (l + 1)
        f0 = 4 * np.pi / (2 * l + 1)

        # compute poloidal magnetic energy
        f1 = r_sqr * np.absolute(elm0) ** 2
        f2 = r_sqr * L * np.absolute(flm0) ** 2
        me_pol[k, :] = f0*(f1 + f2)

        # compute poloidal induction force
        f1 = r_sqr * 2 * np.real(elm0 * np.conj(qlmi))
        f2 = r_sqr * L * 2 * np.real(flm0 * np.conj(slmi))
        wi_pol[k, :] =  f0 * (f1 + f2)

        # compute poloidal ohmic dissipation
        f1 = np.absolute(elm0 - flm0 - r * flm1) ** 2
        do_pol[k, :] = 2 * L * f0 * f1

    # compute toroidal components
    for k, l in enumerate(lltor):
        glm0 = G0[k, :]
        glm1 = G1[k, :]
        glm2 = G2[k, :]

        tlmi = ufig.induction_tor(l, k, r, p, E0, E1, F0, F1, G0, G1)

        L = l * (l + 1)
        f0 = 4 * np.pi / (2 * l + 1)

        # compute toroidal magnetic energy
        f1 = r_sqr * L * np.absolute(glm0) ** 2
        me_tor[k, :] = f0 * f1

        # compute toroidal induction force
        f1 = r_sqr * L * 2 * np.real(glm0 * np.conj(tlmi))
        wi_tor[k, :] = f0 * f1

        # compute toroidal ohmic dissipation
        f1 = np.absolute(r * glm1 + glm0) ** 2
        f2 = L * np.absolute(glm0) ** 2
        do_tor[k, :] = 2 * L * f0 * (f1 + f2)

    me = np.real(sum(me_pol + me_tor, 0))
    wind = np.real(sum(wi_pol + wi_tor, 0))
    dohm = omgtau * Em * np.real(sum(do_pol + do_tor, 0))

    ax.semilogy(r, me, label=r'ME')
    ax.semilogy(r, abs(wind), label=r'$|\mathcal{W}_\mathrm{ind}|$')
    ax.semilogy(r, abs(dohm), label=r'$|\mathcal{D}_\mathrm{ohm}|$')

if field == "t":
    # initialize indices
    ll0 = ut.ell(m, lmax, symm)
    llpol = ll0[0]

    # prepare solution arrays
    te_pol = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)

    wa_pol = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)

    dt_pol = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)

    rt = np.loadtxt('real_temperature.field', usecols=solnum)
    it = np.loadtxt('imag_temperature.field', usecols=solnum)

    H0, H1, H2 = ufig.scalar_field(rt, it, symm, ricb, rcmb, m, lmax, N, n0, nr, chx)

    for k, l in enumerate(llpol):
        hlm0 = H0[k, :]
        hlm1 = H1[k, :]
        hlm2 = H2[k, :]

        f0 = 4 * np.pi / (2 * l + 1)
        L = l*(l+1)

        plm0 = Q0[k, :] * r / L

        # compute thermal energy
        f1 = r_sqr * np.abs(hlm0) ** 2
        te_pol[k, :] =  f0 * f1

        # compute advection force
        f1 = L * 2 * np.real(np.conj(plm0) * hlm0)
        if heating == 0:
            fr = r_sqr
        elif heating == 1:
            fr = r_inv
        elif heating == 2:
            fr = r * ut.twozone(r, args)
        elif heating == 3:
            fr = r * ut.BVprof(r, args)
        wa_pol[k, :] = f0 * fr * f1

        # compute thermal dissipation
        f1 = 2 * r * 2 * np.real(hlm0 * np.conj(hlm1))
        f2 = r_sqr * 2 * np.real(hlm0 * np.conj(hlm2))
        f3 = -2 * L * np.abs(hlm0) ** 2
        dt_pol[k, :] = f0 * (f1 + f2 + f3)

    te = np.real(sum(te_pol, 0))
    wadv = np.real(sum(wa_pol, 0))
    dthm = omgtau ** 2 * Et * np.real(sum(dt_pol, 0))

    ax.semilogy(r, te, label=r'TE')
    ax.semilogy(r, abs(wadv), label=r'$|\mathcal{W}_\mathrm{adv}|$')
    ax.semilogy(r, abs(dthm), label=r'$|\mathcal{D}_\mathrm{thm}|$')

if field == "c":
    # initialize indices
    ll0 = ut.ell(m, lmax, symm)
    llpol = ll0[0]

    # prepare solution arrays
    ce_pol = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)

    wa_pol = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)

    dc_pol = np.zeros((int((lmax-m+1)/2), nr),dtype=complex)

    rc = np.loadtxt('real_composition.field', usecols=solnum)
    ic = np.loadtxt('imag_composition.field', usecols=solnum)

    C0, C1, C2 = ufig.scalar_field(rc, ic, symm, ricb, rcmb, m, lmax, N, n0, nr, chx)

    for k, l in enumerate(llpol):
        clm0 = C0[k, :]
        clm1 = C1[k, :]
        clm2 = C2[k, :]

        f0 = 4 * np.pi / (2 * l + 1)
        L = l * (l + 1)

        plm0 = Q0[k, :] * r / L

        # compute thermal energy
        f1 = r_sqr * np.abs(clm0) ** 2
        ce_pol[k, :] = f0 * f1

        # compute advection force
        f1 = L * 2 * np.real(np.conj(plm0) * clm0)
        if heating == 0:
            fr = r_sqr
        elif heating == 1:
            fr = r_inv
        elif heating == 2:
            fr = r * ut.twozone(r, args)
        elif heating == 3:
            fr = r * ut.BVprof(r, args)
        wa_pol[k, :] = f0 * fr * f1

        # compute thermal dissipation
        f1 = 2 * r * 2 * np.real(clm0 * np.conj(clm1))
        f2 = r_sqr * 2 * np.real(clm0 * np.conj(clm2))
        f3 = -2 * L * np.abs(clm0) ** 2
        dc_pol[k, :] = f0 * (f1 + f2 + f3)

    ce = np.real(sum(ce_pol, 0))
    wadv = np.real(sum(wa_pol, 0))
    dcmp = omgtau ** 2 * Et * np.real(sum(dc_pol, 0))

    ax.semilogy(r, ce, label=r'CE')
    ax.semilogy(r, abs(wadv), label=r'$|\mathcal{W}_\mathrm{adv}|$')
    ax.semilogy(r, abs(dcmp), label=r'$|\mathcal{D}_\mathrm{cmp}|$')

# --- VISUALIZATION ---
# plot figure
ax.legend()

# show/save figure
plt.tight_layout()
plt.show()
#plt.savefig('profile.png'.format(ricb))
