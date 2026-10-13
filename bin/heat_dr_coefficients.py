'''
Angular coupling coefficients of the differential-rotation terms in the heat equation (docs/heat-equation.md, section 2).

With Schmidt harmonics S_l^m (ut.Ylm) and x = cos θ, every coupling comes from two matrices on the degree index at
fixed m (row = target l', column = source l):
    x S_l = p_l S_{l+1} + q_l S_{l-1},       p_l = sqrt((l+1)² - m²)/(2l+1),  q_l = sqrt(l² - m²)/(2l+1)     -> X
    (1 - x²) dS_l/dx = (l+1) q_l S_{l-1} - l p_l S_{l+1}                                                     -> D
and the terms need Pn(X) (advection, u_r dS_tw/dr), Pn'(X) D (poloidal u_θ dS_tw/dθ) and Pn'(X) (toroidal
u_θ dS_tw/dθ) for n = 2, 4 (and n = 6 with the exact thermal wind, thermal_wind = 2). The matrices are built with
8 degrees more than lmax, because entries formed from a truncated X are wrong within a few degrees of the truncation
(P6(X) and P6'(X) D need 6 raising steps, lp reaches lmax + 1 for m = 0).
'''
import numpy as np
from parameters import par
import utils as ut

_cache = {}

def _matrices(m):
    if m in _cache:
        return _cache[m]
    nhi = max(ut.tw_nmax, ut.dr_nmax)   # degrees above 6 only with ut.dr_high (docs/heat-equation.md §8)
    lw = par.lmax + max(8, nhi + 4)
    X = np.zeros((lw + 1, lw + 1)); D = np.zeros((lw + 1, lw + 1))
    for l in range(m, lw + 1):
        if l + 1 <= lw:
            X[l + 1, l] = np.sqrt((l + 1)**2 - m**2)/(2*l + 1)
            D[l + 1, l] = -l*X[l + 1, l]
        if l - 1 >= m:
            X[l - 1, l] = np.sqrt(l**2 - m**2)/(2*l + 1)
            D[l - 1, l] = (l + 1)*X[l - 1, l]
    I  = np.eye(lw + 1)
    X2 = X @ X
    P2  = (3*X2 - I)/2
    P4  = (35*X2 @ X2 - 30*X2 + 3*I)/8
    dP2 = 3*X
    dP4 = (35*X2 @ X - 15*X)/2
    X4  = X2 @ X2
    P6  = (231*X4 @ X2 - 315*X4 + 105*X2 - 5*I)/16
    dP6 = (693*X4 @ X - 630*X2 @ X + 105*X)/8
    _cache[m] = {'P2': P2, 'P4': P4, 'P6': P6, 'dP2': dP2, 'dP4': dP4, 'dP6': dP6,
                 'dP2D': dP2 @ D, 'dP4D': dP4 @ D, 'dP6D': dP6 @ D}
    if nhi > 6:   # P_n(X), P_n'(X) by the recurrences (k+1) P_{k+1} = (2k+1) X P_k - k P_{k-1},  P_{k+1}' = (k+1) P_k + X P_k'
        Pk, Pkm, dPk = X.copy(), I.copy(), I.copy()   # P1, P0, P1'
        for k in range(1, nhi):
            Pk, Pkm, dPk = ((2*k + 1)*X @ Pk - k*Pkm)/(k + 1), Pk, (k + 1)*Pk + X @ dPk
            n = k + 1
            if n > 6 and n % 2 == 0:
                _cache[m].update({'P%d' % n: Pk, 'dP%d' % n: dPk, 'dP%dD' % n: dPk @ D})
    return _cache[m]

def coef(name, l, m, offdiag):
    '''coefficient of the target degree l in (name applied to the source degree l + offdiag)'''
    M = _matrices(m)[name]
    ls = l + offdiag
    if ls < m or ls < 0 or ls >= M.shape[0] or l >= M.shape[0]:
        return 0.0
    return M[l, ls]
