'''
Random Reynolds-stress body force: the right-hand sides of the forced problem for forcing = 11.

Convection acts on the large-scale flow through the fluctuating Reynolds stress. Here it is the random force per
unit mass
    f = rho0^-1 div(rho0 s),
s a random symmetric traceless tensor (the fluctuating part of u'u'), white in time and in space up to the eddy size
l_c, with the Landau-Lifshitz covariance
    E[ s_ij(x,t) s_kl(x',t')* ] = 4 Theta nu_t/rho0  Pi_ijkl  delta(x - x') delta(t - t'),
Pi the projector onto symmetric traceless tensors, nu_t the turbulent part of kore's viscosity and Theta(r) a
"turbulent temperature", about the energy of one eddy (rsf_theta). This is the fluctuation-dissipation partner of
kore's viscous force rho0^-1 div(2 rho0 nu d): for any test field a, in the white limit,
    E|<a, f>|^2 = 4 int rho0 Theta nu_t d(a)*:d(a) dV,   <a, b> = int rho0 a*.b dV,
so a damped mode of unit kinetic energy reaches E|c|^2 = Theta_eff f_t, f_t the fraction of its damping due to
nu_t (exactly so for a normal operator; with shear the forced solve gives the non-normal corrections). The finite
eddy size and the angular factor below only lower this: by O(1) factors for modes whose scale is near l_c (about
0.5 for the test fields of tests/rsf at l_c 0.05), so compare a forced run with Theta_eff f_t only in that sense.
Project note: kore/convective-forcing/reynolds-stress-forcing.md (derivation, checks, choice of Theta).

Per degree l (Y = Y_l^m as utils.Ylm, n_l = 4 pi/(2l+1), L = l(l+1)), s = sum_k c_k(r) T_k with the tensor harmonics
    T1 = Y rr,  T2 = rB + Br,  T3 = rC + Cr,  T4 = Y (I - rr),  T5, T6 = the traceless surface Hessian of Y and its
    rotation by 90 degrees,  c4 = -c1/2 (traceless),   B = r grad Y, C = r x B.
The c_k are independent (across k, l, m and frequency), with
    E[ c_k(r) c_k(r')* ] = q chi_k/(r^2 n_l) delta(r - r'),   q = 4 Theta nu_t/rho0,
    chi_1,2,3,5,6 = 2/3, 1/(2L), 1/(2L), 2/(L(L-2)), 2/(L(L-2))     (no c5, c6 at l = 1).
With h1, h2 = (ln rho0)', (ln rho0)'' the force f = fP Y r + fB B + fC C has
    fP = c1' + (3/r + h1) c1 - L c2/r
    fB = -c1/(2r) + c2' + (3/r + h1) c2 - (L-2) c5/(2r)
    fC = c3' + (3/r + h1) c3 - (L-2) c6/(2r)
and the rows, with the scaling of the inertia rows (r^rpower_u rho0^rhopower_u (r.curl curl f)_l/L^3 and
r^rpower_v rho0^rhopower_v (r.curl f)_l/L^3), are
    b_u = r^(rpower_u - 2) rho0^rhopower_u [fP - (r fB)']/L^2   (rows of P_l, l in ut.ell()[0]: c1, c2, c5)
    b_v = -r^(rpower_v - 1) rho0^rhopower_v fC/L^2              (rows of T_l, l in ut.ell()[1]: c3, c6).
The entropy rows are zero. The overall sign of each block does not matter (the noise is symmetric).

Discretisation. The noise is a sum over centres r_j, spaced h_j = sigma_j/2,
    c_k(r) = tau(r) A_lk(r) S(r)^(-1/2) sum_j sqrt(h_j) xi_j g_j(r),   S(r) = sum_j h_j g_j(r),
    A_lk = sqrt(q chi_k/(r^2 n_l)) [exp(-L l_c^2/(4 r^2))],
g_j a unit-area Gaussian of width sigma_j = l_c(r_j)/sqrt(2), xi_j complex circular Gaussians (E|xi|^2 = 1), the
bracket with rsf_angular = 1 (the horizontal correlation of eddies of size l_c). The amplitude A is taken at r, not
at the centres, so the stress vanishes wherever nu_t does: none leaks into a radiative interior. The centres run
from 3 sigma below the inner wall (or r = 0) to 3 sigma above rcmb, so that S = 1 inside for constant l_c (it
deviates by tens of % where l_c changes over l_c, near the surface; S^(-1/2) corrects that), and
    E[c(r) c(r')*] = tau tau' A A' (S S')^(-1/2) sum_j h_j g_j(r) g_j(r') = tau tau' A A' x (unit-area Gaussian of
    width l_c in r - r')
(exactly for constant l_c): delta-correlated on scales above l_c. By Cauchy-Schwarz, for any field a
    E|<a, f>|^2 <= 4 int rho0 Theta nu_t tau^2 [exp(-L l_c^2/(2 r^2))] |d(a)|^2 dV
(tests/rsf/noise_check.py B5), with equality in the white limit. tau is a smooth (erfc) taper to zero over
rsf_taper below the outer wall (and above an inner core, or over rsf_taper_in above r = 0 in a full sphere), so that
no surface traction is needed in the stress-free rows; it reduces the effective Theta nu_t to Theta nu_t tau^2, and,
within about l_c of a wall, further (the correlation partners lie in the taper). With rsf_rcut, tau also goes smoothly
from 1 above rsf_rcut + rsf_rcut_width to 1e-8 at rsf_rcut (the same erfc shape), which keeps the forcing out of a
region where nu_t is small but not zero (the tail of a viscosity step); clipping nu_t there with a higher rsf_nu_floor
would leave a corner in the stress, whose second derivative the rows need and Chebyshev polynomials cannot resolve.

Background (background()): rho0 and its log-derivatives, nu_t = ViscosD (nu/nu0 - rsf_nu_floor) and their derivatives
from kore's own profile helpers (radial_profiles.viscoX, logrhoX, pressX); nu_t below 1e-14 of its maximum counts as
zero. l_c = rsf_lc, or (rsf_lc_max^-4 + (rsf_alpha H_p)^-4)^(-1/4), a smooth min of rsf_alpha H_p and rsf_lc_max
(H_p = -p/p', infinite at r = 0). sqrt(q) = C nu_t^a l_c^b rho0^c r^d for each choice of Theta (_theta_law), so A,
A' and A'' follow analytically.

Normalisation in time. One column of B_forced.npz is the Fourier amplitude of the force at the forcing frequency, and
E|y|^2 over many columns is the two-sided power spectral density of any linear output y, in kore units, such that
int E|y(w)|^2 dw/2pi is the variance of y. For m != 0, the mean of the solutions' kinetic energies at w
(rsf_spectrum.dat) integrates over w/2pi to the mean kinetic energy of the real flow u_m Y_m + c.c. (kore's KE of its
m component, spin_doctor, utils4pp.kinetic_energy). For m = 0 the stress and the flow are real, the spectrum is even
in w and its +w and -w halves are the same motion: the integral over all w is twice the mean kinetic energy (halve
it, or integrate over w > 0). Either way it is the energy of the solved symm class: the noise drives both classes
independently, so the energy at m is the sum of the symm = +1 and -1 runs. White noise is the same in every rotating
frame.

Column j of B_forced.npz uses numpy.random.default_rng([rsf_seed, j]), so it does not depend on rsf_nsamp.
Parameters: rsf_* in params_default.py.
'''

import sys
import numpy as np
import scipy.fftpack as sft
import scipy.special as ssp
import scipy.sparse as ss

from parameters import par
import utils as ut
import radial_profiles as rap

LC_SOFT = 4     # l_c = (rsf_lc_max^-4 + (rsf_alpha H_p)^-4)^(-1/4)
NU_CUT = 1e-14  # nu_t below this fraction of its maximum counts as zero
S_EPS = 1e-30   # S + S_EPS in the normalisation S^(-1/2), finite where no centre reaches
_cache = {}


def chi(k, L):
    '''variance factor chi_k of the tensor-harmonic coefficient c_k (k = 1, 2, 3, 5, 6)'''
    return {1: 2/3, 2: 1/(2*L), 3: 1/(2*L), 5: 2/(L*(L - 2)) if L > 2 else 0., 6: 2/(L*(L - 2)) if L > 2 else 0.}[k]


def nu_floor():
    '''the part of kore's viscosity (in units of ViscosD) that is not turbulent'''
    if par.rsf_nu_floor is not None:
        return par.rsf_nu_floor
    import background as bg
    return bg.nu_floor()   # visc0 below the step of 'step' and 'solar_rz' (if < 1), 0 otherwise


def nut_max():
    '''the maximum of nu_t = ViscosD (nu/nu0 - floor) over the domain (fine grid)'''
    key = ('nut_max', par.ViscosD, nu_floor())
    if key not in _cache:
        rf = np.linspace(par.ricb, ut.rcmb, 20001)[1:]
        _cache[key] = max(np.max(par.ViscosD*(rap.viscoX(rf, 0) - nu_floor())), 0.)
    return _cache[key]


def _theta_law():
    '''
    sqrt(q) = C nu_t^a l_c^b rho0^c r^d, q = 4 Theta nu_t/rho0, for the chosen rsf_theta: returns (C, a, b, c, d),
    rsf_scale included
    '''
    th = par.rsf_theta
    if isinstance(th, str) and th == 'nu':        # Theta = 3 rho0 nu_t^2 l_c (u_c = 3 nu_t/l_c): q = 12 nu_t^3 l_c
        C, a, b, c, d = np.sqrt(12.), 1.5, 0.5, 0., 0.
    elif isinstance(th, str) and th == 'mlt':     # Theta = rho0 u_c^2 l_c^3/3, u_c^3 = alpha F_c/(10 rho0)
        if par.rsf_lum is None:
            sys.exit("rsf_theta = 'mlt' needs rsf_lum, the luminosity in kore units (params_default.py)")
        if getattr(par, 'visc_pow', 0) != 0:
            sys.exit("rsf_theta = 'mlt' takes the convective fraction of the flux as nu_t/max(nu_t), which needs visc_pow = 0")
        nref = nut_max()   # F_c = rsf_lum/(4 pi r^2) nu_t/nref
        C = np.sqrt(4/3)*(par.rsf_alpha*par.rsf_lum/(40*np.pi*nref))**(1/3) if nref > 0 else 0.
        a, b, c, d = 5/6, 1.5, -1/3, -2/3
    elif isinstance(th, str):
        sys.exit("rsf_theta must be 'nu', 'mlt' or a number")
    else:                                         # constant Theta: q = 4 Theta nu_t/rho0
        C, a, b, c, d = 2*np.sqrt(float(th)), 0.5, 0., -0.5, 0.
    return C*np.sqrt(par.rsf_scale), a, b, c, d


def eddy_size(r):
    '''the eddy size l_c and its first two derivatives at r'''
    r = np.asarray(r, dtype=float)
    if par.rsf_lc is not None:
        return par.rsf_lc*np.ones_like(r), np.zeros_like(r), np.zeros_like(r)
    p0, p1, p2, p3 = [rap.pressX(r, d) for d in range(4)]
    g0 = -p1/p0/par.rsf_alpha                                       # 1/(alpha H_p) and its derivatives
    g1 = (-p2/p0 + (p1/p0)**2)/par.rsf_alpha
    g2 = (-p3/p0 + 3*p1*p2/p0**2 - 2*(p1/p0)**3)/par.rsf_alpha
    k = LC_SOFT
    u0 = par.rsf_lc_max**(-k) + g0**k
    u1 = k*g0**(k - 1)*g1
    u2 = k*(k - 1)*g0**(k - 2)*g1**2 + k*g0**(k - 1)*g2
    lc0 = u0**(-1/k)
    lc1 = -u0**(-1/k - 1)*u1/k
    lc2 = (1/k)*(1/k + 1)*u0**(-1/k - 2)*u1**2 - u0**(-1/k - 1)*u2/k
    return lc0, lc1, lc2


def background(r):
    '''
    Background at radii r > 0 (kore units): dict with rho0, h1 = (ln rho0)', h2 = (ln rho0)'', nut (turbulent
    viscosity), lc, lc1, lc2 (eddy size and its derivatives), theta (turbulent temperature, times rsf_scale; 0 where
    nu_t is) and s, s1, s2: s = sqrt(q), q = 4 Theta nu_t/rho0 the variance of the stress per unit volume and time,
    and its first two derivatives (zero where nu_t is).
    '''
    r = np.asarray(r, dtype=float)
    rho0 = rap.prf.density(r, 0)   # as rap.burrito
    out = {'rho0': rho0, 'h1': rap.logrhoX(r, 1), 'h2': rap.logrhoX(r, 2)}
    nu0 = par.ViscosD*(rap.viscoX(r, 0) - nu_floor())
    nu1, nu2 = par.ViscosD*rap.viscoX(r, 1), par.ViscosD*rap.viscoX(r, 2)
    on = nu0 > NU_CUT*nut_max()
    nn = np.where(on, nu0, 1.)     # safe denominators
    lc, lc1, lc2 = eddy_size(r)
    C, a, b, c, d = _theta_law()
    s = np.where(on, C*nn**a*lc**b*rho0**c*r**d, 0.)
    ls1 = a*nu1/nn + b*lc1/lc + c*out['h1'] + d/r                                     # (ln s)'
    ls2 = a*(nu2/nn - (nu1/nn)**2) + b*(lc2/lc - (lc1/lc)**2) + c*out['h2'] - d/r**2  # (ln s)''
    out['s'], out['s1'], out['s2'] = s, s*ls1, s*(ls2 + ls1**2)
    out['nut'] = np.where(on, nu0, 0.)
    out['lc'], out['lc1'], out['lc2'] = lc, lc1, lc2
    out['theta'] = np.where(on, rho0*s**2/(4*nn), 0.)   # Theta = rho0 q/(4 nu_t)
    return out


def envelope(bg, r, l, k):
    '''
    A = sqrt(chi_k/n_l) s/r [exp(-L l_c^2/(4 r^2)) with rsf_angular] and its first two derivatives at r, from the
    background bg at the same r
    '''
    r = np.asarray(r, dtype=float)
    L = l*(l + 1.)
    K = np.sqrt(chi(k, L)*(2*l + 1)/(4*np.pi))
    lf1, lf2 = -1/r, 1/r**2   # (ln f)', (ln f)'' of f = 1/r [exp(-(L/4) l_c^2/r^2)]
    if par.rsf_angular:
        lc, lc1, lc2 = bg['lc'], bg['lc1'], bg['lc2']
        g0 = lc**2/r**2
        g1 = 2*lc*lc1/r**2 - 2*lc**2/r**3
        g2 = 2*lc1**2/r**2 + 2*lc*lc2/r**2 - 8*lc*lc1/r**3 + 6*lc**2/r**4
        f0 = np.exp(-L*g0/4)/r
        lf1, lf2 = lf1 - L*g1/4, lf2 - L*g2/4
    else:
        f0 = 1/r
    f1, f2 = f0*lf1, f0*(lf2 + lf1**2)
    s, s1, s2 = bg['s'], bg['s1'], bg['s2']
    return K*s*f0, K*(s1*f0 + s*f1), K*(s2*f0 + 2*s1*f1 + s*f2)


def normalisation(S):
    '''S^(-1/2) and its first two derivatives, from S = sum_j h_j g_j(r) and its derivatives'''
    S0, S1, S2 = S
    x = S0 + S_EPS
    return x**-0.5, -0.5*x**-1.5*S1, 0.75*x**-2.5*S1**2 - 0.5*x**-1.5*S2


def _erfc_step(z):
    '''(1/2) erfc(z) and its first two derivatives with respect to z'''
    e = np.exp(-z**2)/np.sqrt(np.pi)
    return 0.5*ssp.erfc(z), -e, 2*z*e


def taper(r):
    '''the taper tau(r) of the stress and its first two derivatives'''
    r = np.asarray(r, dtype=float)
    T = par.rsf_taper
    Ti = par.rsf_taper if par.ricb > 0 else par.rsf_taper_in
    if not (T > 0 and Ti > 0):
        sys.exit('rsf_taper (and rsf_taper_in with ricb = 0) must be > 0: the stress has to vanish at the walls '
                 '(and at r = 0), where the stress-free rows carry no surface traction')
    w = T/8
    s0, s1, s2 = _erfc_step((r - (ut.rcmb - T/2))/w)   # 1 below rcmb - T, 1e-8 at rcmb
    t0, t1, t2 = s0, s1/w, s2/w**2
    wi = Ti/8
    i0, i1, i2 = _erfc_step(((par.ricb + Ti/2) - r)/wi)   # 1e-8 at ricb (or r = 0), 1 above ricb + Ti
    i1, i2 = -i1/wi, i2/wi**2
    t0, t1, t2 = t0*i0, t1*i0 + t0*i1, t2*i0 + 2*t1*i1 + t0*i2
    if par.rsf_rcut is not None:
        Wc = par.rsf_rcut_width
        if not (Wc > 0 and par.ricb <= par.rsf_rcut < ut.rcmb - Wc):
            sys.exit('rsf_rcut must lie in [ricb, rcmb - rsf_rcut_width) and rsf_rcut_width must be > 0')
        wc = Wc/8
        c0, c1, c2 = _erfc_step(((par.rsf_rcut + Wc/2) - r)/wc)   # 1e-8 at rsf_rcut, 1 above rsf_rcut + rsf_rcut_width
        c1, c2 = -c1/wc, c2/wc**2
        t0, t1, t2 = t0*c0, t1*c0 + t0*c1, t2*c0 + 2*t1*c1 + t0*c2
    return t0, t1, t2


def centres():
    '''
    Noise centres r_j with spacing h_j = sigma_j/2, sigma_j = l_c(r_j)/sqrt(2), from 3 sigma below ricb (or r = 0)
    to 3 sigma above rcmb, l_c taken at the nearest point of the domain outside it. Centres whose kernels (within
    6 sigma) only reach where the stress amplitude tau s/r is below 1e-7 of its maximum are dropped.
    '''
    lo, hi = par.ricb, ut.rcmb
    rf = np.linspace(lo, hi, 20001)[1:]
    sf = eddy_size(rf)[0]/np.sqrt(2)
    sig = lambda x: np.interp(np.clip(x, rf[0], hi), rf, sf)
    rj = []
    x = lo - 3*sig(lo)
    while x < hi + 3*sig(hi):
        rj.append(x)
        x += sig(x)/2
        if len(rj) > 200000:
            sys.exit('reynolds_stress: more than 2e5 noise centres, l_c too small (rsf_lc, rsf_alpha)')
    rj = np.array(rj)
    sj = sig(rj)
    hj = np.gradient(rj) if rj.size > 1 else sj/2
    env = taper(rf)[0]*background(rf)['s']/rf
    keep = np.zeros(rj.size, bool)
    if env.max() > 0:
        i0 = np.searchsorted(rf, rj - 6*sj)
        i1 = np.searchsorted(rf, rj + 6*sj)
        keep = np.array([i1[j] > i0[j] and env[i0[j]:i1[j]].max() > 1e-7*env.max() for j in range(rj.size)], bool)
    return {'r': rj[keep], 'h': hj[keep], 'sig': sj[keep], 'lc': np.sqrt(2)*sj[keep]}


class Builder:
    '''
    Everything needed to turn noise into the rows of kore's sections u and v: the Chebyshev nodes (r > 0 half
    when ricb = 0), the background, taper and amplitudes there, the noise kernels and the basis changes.
    '''

    def __init__(self):
        if par.ViscosD == 0:
            sys.exit('forcing = 11 needs ViscosD > 0 (the noise is the partner of the viscous force)')
        self.N = par.N
        x = np.cos(np.pi*(np.arange(self.N) + 0.5)/self.N)   # the nodes of utils.chebco_f
        if par.ricb > 0:
            self.r = par.ricb + (ut.rcmb - par.ricb)*(x + 1)/2
        else:
            if self.N % 2:
                sys.exit('forcing = 11 with ricb = 0 needs N even')
            self.r = ut.rcmb*x[:self.N//2]   # r > 0; the other half by parity
            if par.rpower_u < 5 or par.rpower_v < 3:
                sys.exit('forcing = 11 with ricb = 0 needs rpower_u >= 5 and rpower_v >= 3 (params_default.py)')
        self.bg = background(self.r)
        self.tau = taper(self.r)
        self.c = centres()
        d = self.r[:, None] - self.c['r'][None, :]
        s = self.c['sig'][None, :]
        g = np.exp(-d**2/(2*s**2))/(np.sqrt(2*np.pi)*s)
        self.G = [g, -d/s**2*g, (d**2/s**4 - 1/s**2)*g]   # kernels and their first two r-derivatives
        self.nrm = normalisation([G @ self.c['h'] for G in self.G])
        S0, S1, S2, S3 = [ut.Slam(k, self.N) for k in range(4)]
        self.S = {'u': S3*S2*S1*S0, 'v': S1*S0}   # C^(0) -> C^(4) (section u), C^(2) (section v), as submatrices.py
        self.gbx = {'u': 4, 'v': 2}
        vP = int((1 - 2*(par.m % 2))*par.symm)
        # parity of the rows of P (inertia 'u3_D0', r^rpower_u) and T ('v1_D0', r^rpower_v), as submatrices.py
        self.rowpar = {'u': vP*(1 - 2*((par.rpower_u - 5) % 2)), 'v': -vP*(1 - 2*((par.rpower_v - 3) % 2))}
        self.env = {}
        self.tail = 0.

    def amplitudes(self, l, k):
        '''sqrt(h) at the centres: the weight of xi_j (the amplitude A_lk is applied at the nodes, coeff())'''
        return np.sqrt(self.c['h'])

    def envelope(self, l, k):
        '''tau A_lk S^(-1/2) and its first two derivatives at the nodes'''
        if (l, k) not in self.env:
            A0, A1, A2 = envelope(self.bg, self.r, l, k)
            t0, t1, t2 = self.tau
            n0, n1, n2 = self.nrm
            t0, t1, t2 = t0*n0, t1*n0 + t0*n1, t2*n0 + 2*t1*n1 + t0*n2
            self.env[(l, k)] = (t0*A0, t1*A0 + t0*A1, t2*A0 + 2*t1*A1 + t0*A2)
        return self.env[(l, k)]

    def coeff(self, W, l, k):
        '''c_k, c_k', c_k'' at the nodes for the noise weights W (centres x columns), degree l'''
        S = [G @ W for G in self.G]
        E0, E1, E2 = [e[:, None] for e in self.envelope(l, k)]
        return E0*S[0], E1*S[0] + E0*S[1], E2*S[0] + 2*E1*S[1] + E0*S[2]

    def forces_u(self, l, W1, W2, W5):
        '''fP, fB, fB' at the nodes from the noise weights of c1, c2, c5 (None for zero)'''
        r = self.r[:, None]
        h1, h2 = self.bg['h1'][:, None], self.bg['h2'][:, None]
        L = l*(l + 1.)
        z = lambda W, k: [0., 0., 0.] if W is None else self.coeff(W, l, k)
        c1, c1p, _ = z(W1, 1)
        c2, c2p, c2pp = z(W2, 2)
        c5, c5p, _ = z(W5, 5)
        fP = c1p + (3/r + h1)*c1 - L*c2/r
        fB = -c1/(2*r) + c2p + (3/r + h1)*c2 - (L - 2)*c5/(2*r)
        dfB = (c1/(2*r**2) - c1p/(2*r) + c2pp + (3/r + h1)*c2p + (h2 - 3/r**2)*c2
               + (L - 2)*c5/(2*r**2) - (L - 2)*c5p/(2*r))
        return fP, fB, dfB

    def forces_v(self, l, W3, W6):
        '''fC at the nodes from the noise weights of c3, c6 (None for zero)'''
        r = self.r[:, None]
        h1 = self.bg['h1'][:, None]
        L = l*(l + 1.)
        z = lambda W, k: [0., 0., 0.] if W is None else self.coeff(W, l, k)
        c3, c3p, _ = z(W3, 3)
        c6, _, _ = z(W6, 6)
        return c3p + (3/r + h1)*c3 - (L - 2)*c6/(2*r)

    def rows_u(self, l, fP, fB, dfB):
        '''section-u rows of P_l (N1 x columns) for the force components fP, fB, fB' at the nodes'''
        r, rho0 = self.r[:, None], self.bg['rho0'][:, None]
        fP, fB, dfB = [np.reshape(f, (self.r.size, -1)) for f in (fP, fB, dfB)]   # one column per realisation
        L = l*(l + 1.)
        return self.to_section(r**(par.rpower_u - 2)*rho0**par.rhopower_u*(fP - fB - r*dfB)/L**2, 'u')

    def rows_v(self, l, fC):
        '''section-v rows of T_l for the force component fC at the nodes'''
        r, rho0 = self.r[:, None], self.bg['rho0'][:, None]
        fC = np.reshape(fC, (self.r.size, -1))
        L = l*(l + 1.)
        return self.to_section(-r**(par.rpower_v - 1)*rho0**par.rhopower_v*fC/L**2, 'v')

    def to_section(self, vals, sec):
        '''
        Values at the nodes (r > 0 half if ricb = 0) -> rows of section sec, as submatrices.py lays out an operator's
        output: Chebyshev coefficients (utils.chebco_f, without its cut), C^(0) -> C^(4) or C^(2), the rows of the
        section's parity (ricb = 0), then the bc rows (zero) on top and the last rows dropped.
        '''
        vals = np.asarray(vals)
        if vals.ndim == 1:
            vals = vals[:, None]
        if par.ricb == 0:
            vals = np.vstack([vals, self.rowpar[sec]*vals[::-1]])   # x_{N-1-i} = -x_i
        c = sft.dct(vals, axis=0)/self.N
        c[0] /= 2
        amax = np.abs(c).max()
        if amax > 0:
            self.tail = max(self.tail, np.abs(c[int(0.9*self.N):]).max()/amax)
        y = self.S[sec] @ c
        N1 = ut.N1
        if par.ricb == 0:
            y = y[(1 - self.rowpar[sec])//2::2]
            chop = self.gbx[sec]//2
        else:
            chop = self.gbx[sec]
        out = np.zeros((N1, y.shape[1]), dtype=complex)
        out[chop:] = y[:N1 - chop]
        return out


def rhs(nsamp, seed, verbose=True):
    '''
    nsamp random right-hand sides (columns) of size ut.sizmat for forcing = 11, as a CSR matrix. Column j uses
    numpy.random.default_rng([seed, j]).
    '''
    lp, lt, _ = ut.ell(par.m, par.lmax, par.symm)
    bd = Builder()
    nc = bd.c['r'].size
    out = np.zeros((ut.sizmat, nsamp), dtype=complex)
    if nc == 0:
        print('reynolds_stress: Theta nu_t is zero everywhere, the right-hand sides are zero')
        return ss.csr_matrix(out)
    ku = [1, 2, 5]
    kv = [3, 6]
    Xu = np.zeros((len(lp), 3, nc, nsamp), dtype=complex)
    Xv = np.zeros((len(lt), 2, nc, nsamp), dtype=complex)
    for j in range(nsamp):
        rng = np.random.default_rng([seed, j])
        Xu[..., j] = (rng.standard_normal((len(lp), 3, nc)) + 1j*rng.standard_normal((len(lp), 3, nc)))/np.sqrt(2)
        Xv[..., j] = (rng.standard_normal((len(lt), 2, nc)) + 1j*rng.standard_normal((len(lt), 2, nc)))/np.sqrt(2)
    for i, l in enumerate(lp):
        W = [bd.amplitudes(l, k)[:, None]*Xu[i, q] if chi(k, l*(l + 1.)) > 0 else None for q, k in enumerate(ku)]
        out[i*ut.N1:(i + 1)*ut.N1] = bd.rows_u(l, *bd.forces_u(l, *W))
    for i, l in enumerate(lt):
        W = [bd.amplitudes(l, k)[:, None]*Xv[i, q] if chi(k, l*(l + 1.)) > 0 else None for q, k in enumerate(kv)]
        out[ut.n + i*ut.N1:ut.n + (i + 1)*ut.N1] = bd.rows_v(l, bd.forces_v(l, *W))
    if verbose:
        c, bg = bd.c, bd.bg
        tn = bg['theta']*bg['nut']
        print('--------------------------------------------')
        print(' Random Reynolds-stress forcing (forcing = 11): %d realisation(s), seed %d' % (nsamp, seed))
        print(' %d noise centres on r = %.4f .. %.4f, l_c = %.3g .. %.3g, nu_t floor %.3g ViscosD'
              % (nc, c['r'][0], c['r'][-1], c['lc'].min(), c['lc'].max(), nu_floor()))
        print(' max Theta nu_t = %.3e at r = %.3f (theta = %s, scale %g)'
              % (tn.max(), bd.r[np.argmax(tn)], par.rsf_theta, par.rsf_scale))
        print(' Chebyshev tail of the rows (max over the last 10%% / max): %.1e%s'
              % (bd.tail, '  WARNING: under-resolved, raise N or l_c' if bd.tail > 1e-6 else ''))
        print('--------------------------------------------')
    return ss.csr_matrix(out)
