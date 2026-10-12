'''
Angular coupling coefficients of the differential-rotation terms in the heat equation (docs/heat-equation.md, section 2).

With Schmidt harmonics S_l^m (ut.Ylm) and x = cos θ, every coupling comes from two matrices on the degree index at
fixed m (row = target l', column = source l):
    x S_l = p_l S_{l+1} + q_l S_{l-1},       p_l = sqrt((l+1)² - m²)/(2l+1),  q_l = sqrt(l² - m²)/(2l+1)     -> X
    (1 - x²) dS_l/dx = (l+1) q_l S_{l-1} - l p_l S_{l+1}                                                     -> D
and the terms need P2(X), P4(X) (advection, u_r dS_tw/dr), P2'(X) D, P4'(X) D (poloidal u_θ dS_tw/dθ) and
P2'(X), P4'(X) (toroidal u_θ dS_tw/dθ). The matrices are built with 8 degrees more than lmax, because entries
formed from a truncated X are wrong within a few degrees of the truncation (P4(X) needs 4 raising steps, lp reaches
lmax + 1 for m = 0).
'''
import numpy as np
from parameters import par

_cache = {}

def _matrices(m):
    if m in _cache:
        return _cache[m]
    lw = par.lmax + 8
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
    _cache[m] = {'P2': P2, 'P4': P4, 'dP2': dP2, 'dP4': dP4, 'dP2D': dP2 @ D, 'dP4D': dP4 @ D}
    return _cache[m]

def coef(name, l, m, offdiag):
    '''coefficient of the target degree l in (name applied to the source degree l + offdiag)'''
    M = _matrices(m)[name]
    ls = l + offdiag
    if ls < m or ls < 0 or ls >= M.shape[0] or l >= M.shape[0]:
        return 0.0
    return M[l, ls]
