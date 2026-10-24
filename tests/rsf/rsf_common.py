'''
Shared pieces of the tests of the random Reynolds-stress forcing (forcing = 11, bin/reynolds_stress.py).
Run from a kore run directory (its bin/, parameters.py and, where operators are used, the *.mtx files).

- the energy pairing of kore's rows with a field, as bin/ru_insane.py:
    <a, F> = int (rho0 a)*.F dV = sum_l n_l L^3 [ int rho0^(1-rhopower_u) P_l* r^(3-rpower_u) row_u(F)_l dr
                                                + int rho0^(1-rhopower_v) T_l* r^(3-rpower_v) row_v(F)_l dr ]
- smooth test fields that vanish near the walls (and near r = 0), as kore coefficient vectors
- their strain rate in tensor harmonics: for u = U Y r + V B + W C (U = L P/r, V = P' + P (1/r + (ln rho0)'), W = -T)
    e = U' T1 + (V' - V/r + U/r)/2 T2 + (W' - W/r)/2 T3 + (2U - L V)/(2r) T4 + V/r T5 + W/r T6
  (project note, checked with sympy), the angular integral of d*:d (d = e - tr(e) I/3) and the weak form
    <a, f> = -int rho0 e(a)*:s dV  for s = sum_k c_k T_k, c4 = -c1/2.
'''
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'bin'))
import numpy as np
import numpy.polynomial.chebyshev as ch
from parameters import par
import utils as ut
import radial_profiles as rap

N, N1, ricb = par.N, ut.N1, par.ricb
lp, lt, ll = ut.ell(par.m, par.lmax, par.symm)
vP = int((1 - 2*(par.m % 2))*par.symm)
vT = -vP
s_ = int((par.symm + 1)/2)
istart = {'P': (par.m + 1 - s_) % 2, 'T': (par.m + s_) % 2}   # first Chebyshev index of P, T (ricb = 0)
gb = {'u': 4, 'v': 2}
chop = {s: (gb[s] if ricb > 0 else gb[s]//2) for s in gb}
# parity of the rows of P and T (ricb = 0): that of the inertia rows, r^rpower_u and r^rpower_v (Builder.rowpar)
rowpar = {'u': vP*(1 - 2*((par.rpower_u - 5) % 2)), 'v': vT*(1 - 2*((par.rpower_v - 3) % 2))}
idj = {s: (1 - rowpar[s])//2 for s in rowpar}
dxdr = 1. if ricb == 0 else 2/(ut.rcmb - ricb)
NORM = lambda L: np.array([1., 2*L, 2*L, 2., L*(L - 2)/2, L*(L - 2)/2])   # int |T_k|^2 dOmega / n_l, k = 1..6


class Quad:
    '''Gauss-Legendre quadrature on [ricb, rcmb] ([0, rcmb] if ricb = 0) and the Chebyshev/Gegenbauer values there'''

    def __init__(self, nq):
        xk, wk = np.polynomial.legendre.leggauss(nq)
        lo = ricb
        self.r = 0.5*(ut.rcmb - lo)*(xk + 1) + lo
        self.w = 0.5*(ut.rcmb - lo)*wk
        self.xc = self.r if ricb == 0 else (self.r - ricb)*dxdr - 1
        self.rho0 = np.exp(rap.logrhoX(self.r, 0))
        self.h1 = rap.logrhoX(self.r, 1)
        self.h2 = rap.logrhoX(self.r, 2)
        self.nu = par.ViscosD*rap.viscoX(self.r, 0)
        self.wgt = {'u': self.rho0**(1 - par.rhopower_u)*self.r**(3 - par.rpower_u),
                    'v': self.rho0**(1 - par.rhopower_v)*self.r**(3 - par.rpower_v)}
        self.E = {}
        for lamb in (2, 4):
            E = np.zeros((self.r.size, N)); E[:, 0] = 1.; E[:, 1] = 2*lamb*self.xc
            for k in range(2, N):
                E[:, k] = (2*(k + lamb - 1)*self.xc*E[:, k-1] - (k + 2*lamb - 2)*E[:, k-2])/k
            self.E[lamb] = E

    def vals_row(self, y, sec):
        '''values at the nodes of section-sec rows y (N1, or N1 x K), bc rows on top'''
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, None]
        nk = N1 - chop[sec]
        c = np.zeros((N, y.shape[1]), dtype=complex)
        if ricb > 0:
            c[:nk] = y[chop[sec]:]
        else:
            c[idj[sec] + 2*np.arange(nk)] = y[chop[sec]:]
        return self.E[gb[sec]] @ c

    def pair(self, sec, l, left_vals, y):
        '''n_l L^3 int w(r) conj(left) y dr for the rows y (one column per output)'''
        L = l*(l + 1.)
        nl = 4*np.pi/(2*l + 1)
        return nl*L**3*np.sum((self.w*self.wgt[sec]*np.conj(left_vals))[:, None]*self.vals_row(y, sec), axis=0)

    def field(self, coef, kind):
        '''values and first two r-derivatives at the nodes of P or T given by its kore coefficients (N1)'''
        c = np.zeros(N, dtype=complex)
        if ricb > 0:
            c[:] = coef
        else:
            c[istart[kind]::2] = coef
        return [ch.chebval(self.xc, ch.chebder(c, k) if k else c)*dxdr**k for k in range(3)]


def chebfit_parity(fun, kind):
    '''kore coefficients (N1) of fun(r) (the parity of P or T when ricb = 0), from utils.chebco_f's nodes'''
    c = ut.chebco_f(fun, N, ricb, ut.rcmb, 0.)
    if ricb > 0:
        return c.astype(complex)
    wrong = np.abs(c[1 - istart[kind]::2]).max()/np.abs(c).max()
    if wrong > 1e-12:
        print('chebfit_parity: wrong-parity part %.1e' % wrong)
    return c[istart[kind]::2].astype(complex)


def window(r, r0=None, w=None):
    '''
    smooth window, even in r, below 1e-7 at the walls (and at r = 0); r0, w: centre and width of a localised one
    (default: a broad one in the middle of the shell, around r^2 = 0.45 in a sphere, 5e-8 at r = 0, where the pairing
    weight r^(3 - rpower) of non-default powers would amplify the truncation of the rows)
    '''
    if r0 is None:
        if ricb > 0:
            r0, w = 0.5*(ricb + ut.rcmb), (ut.rcmb - ricb)/8
        else:
            return np.exp(-((r**2 - 0.45)/0.11)**2)
    if ricb > 0:
        return np.exp(-((r - r0)/w)**2)
    return np.exp(-((r**2 - r0**2)/(2*r0*w))**2)


def radii():
    '''centres and width of localised test fields across the domain, all below 1e-7 at the walls'''
    if ricb > 0:
        D = ut.rcmb - ricb
        return [ricb + f*D for f in (0.25, 0.4, 0.55, 0.7, 0.8)], 0.04*D
    return [0.3, 0.45, 0.55, 0.65, 0.75, 0.85], 0.04


def test_fields(rng, ncoef=4, r0=None, w=None):
    '''
    random smooth fields r^l window(r) poly(r^2), vanishing near the walls (regular and of the right parity at r = 0):
    dicts l -> kore coefficients of P (l in lp) and of T (l in lt)
    '''
    P, T = {}, {}
    for kind, ls, out in (('P', lp, P), ('T', lt, T)):
        for l in ls:
            a = rng.standard_normal(ncoef) + 1j*rng.standard_normal(ncoef)
            f = lambda r: window(r, r0, w)*r**l*np.polynomial.polynomial.polyval(r**2, a)
            out[l] = chebfit_parity(f, kind)
    return P, T


def strain(q, l, Pv, Tv):
    '''e1..e6 at the nodes of q for P (values and 2 derivatives, or None) and T (values and 1 derivative, or None)'''
    r, h1, h2 = q.r, q.h1, q.h2
    L = l*(l + 1.)
    z = np.zeros(r.size, dtype=complex)
    U = dU = V = dV = W = dW = z
    if Pv is not None:
        P, dP, d2P = Pv
        U = L*P/r
        dU = L*(dP/r - P/r**2)
        V = dP + P*(1/r + h1)
        dV = d2P + dP*(1/r + h1) + P*(h2 - 1/r**2)
    if Tv is not None:
        W = -Tv[0]
        dW = -Tv[1]
    return [dU, (dV - V/r + U/r)/2, (dW - W/r)/2, (2*U - L*V)/(2*r), V/r, W/r]


def dd(l, e):
    '''int d*:d dOmega / n_l (d the traceless part of e)'''
    L = l*(l + 1.)
    n = NORM(L)
    return sum(n[k]*np.abs(e[k])**2 for k in range(6)) - np.abs(e[0] + 2*e[3])**2/3


def dd_channels(l, e):
    '''
    the same split by the stress coefficient that pairs with each part: k = 1 (c1, with c4 = -c1/2): (2/3)|e1 - e4|^2;
    k = 2, 3, 5, 6: NORM_k |e_k|^2. They add up to dd(l, e).
    '''
    L = l*(l + 1.)
    n = NORM(L)
    out = {1: (2/3)*np.abs(e[0] - e[3])**2}
    for k in (2, 3, 5, 6):
        out[k] = n[k-1]*np.abs(e[k-1])**2
    return out


def weak(q, l, e, c):
    '''-int rho0 e*:s dV for s = sum c_k T_k (c: dict k -> values at the nodes, columns), c4 = -c1/2'''
    L = l*(l + 1.)
    n = NORM(L)
    nl = 4*np.pi/(2*l + 1)
    tot = 0.
    for k, ck in c.items():
        if k == 1:
            tot = tot + np.conj(e[0] - e[3])[:, None]*ck   # c1 (T1 - T4/2): |T1|^2 = 1, |T4|^2/2 = 1
        else:
            tot = tot + n[k-1]*np.conj(e[k-1])[:, None]*ck
    return -nl*np.sum((q.w*q.rho0*q.r**2)[:, None]*tot, axis=0)
