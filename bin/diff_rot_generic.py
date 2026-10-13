'''
Angular coupling coefficients of the differential-rotation force for ΔΩ = f_n(r) P_n(cos θ), any n (docs/heat-equation.md §8).

The background flow U0 = G ΔΩ r sinθ e_φ acts on the perturbation u through
    W = G [ ΔΩ (i m u + 2 ẑ×u) + sinθ (r ∂ᵣΔΩ u_r + ∂_θΔΩ u_θ) e_φ ],
with u = ρ0⁻¹∇×∇×(ρ0 P r) + ∇×(T r) of degree l: u_r = L P Y/r, u_θ = a ∂_θY + i m T Y/sinθ, u_φ = i m a Y/sinθ - T ∂_θY,
a = (rP)'/r + h' P (h = ln ρ0). Each component of W is a radial shape (q = L P/r, a, T, or q with r f_n' in place of f_n)
times an angular function; projecting W of source degree l on the target degree l' (W = Σ [qW Y r̂ + sW r∇Y + tW r̂×r∇Y])
gives 12 primitive matrices C[k][shape][l', l], k = q, s, t, by Gauss-Legendre quadrature in cos θ, exact for these
polynomial integrands. kore's rows are row_v = r̂·∇×W ∝ L' tW and row_u = r̂·∇×∇×W ∝ (L'/r²)[qW - ∂ᵣ(r sW)], so the
coefficients of the radial operators (the labels of submatrices.py) follow by the fixed algebra of `coefs` below, in the
naming and normalisation of diff_rot_coefficients.py (which has them in closed form for n = 0 and 2; the two agree to
~1e-12, tests/diff_rot/gen_check.py). The matrices are built for degrees up to lmax + n + 6 and cached per (n, m).
'''
import numpy as np
import scipy.special as scsp
from parameters import par

_cache = {}
_harm = {}

def _harmonics(m, lw, nq):
    '''Gauss-Legendre nodes, weights and Schmidt Y_l^m (as utils.Ylm at φ = 0), ∂_θY, i m Y/sinθ for l = 0..lw'''
    key = (m, lw, nq)
    if key in _harm:
        return _harm[key]
    x, w = np.polynomial.legendre.leggauss(nq)
    th = np.arccos(x)
    sth = np.sqrt(1 - x**2)
    Y = np.zeros((lw + 1, nq), dtype=complex)
    dY = np.zeros_like(Y)
    Z = np.zeros_like(Y)
    for l in range(m, lw + 1):
        y, dy = scsp.sph_harm_y(l, m, th, 0.0, diff_n=1)
        c = np.sqrt(4*np.pi/(2*l + 1))
        Y[l] = y*c
        dY[l] = dy[:, 0]*c
        if m:
            Z[l] = 1j*m*Y[l]/sth
    _harm[key] = (x, w, sth, Y, dY, Z)
    return _harm[key]

def _primitives(n, m, lw):
    nq = 2*lw + n + 12
    x, w, sth, Y, dY, Z = _harmonics(m, lw, nq)
    Pn = np.polynomial.legendre.Legendre.basis(n)
    pn, dpn = Pn(x), Pn.deriv()(x)
    l = np.arange(lw + 1)
    L = l*(l + 1.)
    nl = 4*np.pi/(2*l + 1)
    z = 0*Y
    # components of W/G per unit radial shape, rows = source l, columns = nodes
    Wr = {'q': 1j*m*pn*Y, 'a': -2j*m*pn*Y, 'T': 2*sth*pn*dY, 'dq': z}
    Wt = {'q': z, 'a': pn*(1j*m*dY - 2*x*Z), 'T': pn*(1j*m*Z + 2*x*dY), 'dq': z}
    Wp = {'q': 2*sth*pn*Y, 'a': pn*(1j*m*Z + 2*x*dY) - sth**2*dpn*dY, 'T': pn*(-1j*m*dY + 2*x*Z) - sth**2*dpn*Z, 'dq': sth*pn*Y}
    proj = lambda F, B: 2*np.pi*(F*w) @ np.conj(B).T   # [source l, target l'] of ∫ F B* dΩ
    with np.errstate(divide='ignore', invalid='ignore'):
        C = {'q': {}, 's': {}, 't': {}}
        for sh in ('q', 'a', 'T', 'dq'):
            C['q'][sh] = (proj(Wr[sh], Y)/nl[None, :]).T
            C['s'][sh] = ((proj(Wt[sh], dY) + proj(Wp[sh], Z))/(L*nl)[None, :]).T
            C['t'][sh] = ((proj(Wt[sh], Z) - proj(Wp[sh], dY))/(L*nl)[None, :]).T
    for k in C:
        for sh in C[k]:
            C[k][sh][~np.isfinite(C[k][sh])] = 0
    return C, L

def _matrices(n, m):
    key = (n, m)
    if key in _cache:
        return _cache[key]
    lw = par.lmax + n + 6
    C, L = _primitives(n, m, lw)
    Ls, Lt = L[None, :], L[:, None]   # source, target
    Cq, Cs, Ct = C['q'], C['s'], C['t']
    K = {'f_2C_D0P': Ls*Cq['q'] + Cq['a'], 'f_2C_D1P': Cq['a'] - (Ls*Cs['q'] + 2*Cs['a']), 'f_2C_lho1_D0P': Cq['a'] - Cs['a'],
         'f_2C_lho2_D0P': -Cs['a'], 'f_2C_lho1_D1P': -Cs['a'], 'f_2C_D2P': -Cs['a'],
         'df_2C_D0P': Ls*Cq['dq'] - (Ls*Cs['q'] + Cs['a'] + Ls*Cs['dq']), 'df_2C_D1P': -(Cs['a'] + Ls*Cs['dq']),
         'df_2C_lho1_D0P': -Cs['a'], 'd2f_2C_D0P': -Ls*Cs['dq'],
         'f_2C_D0T': Cq['T'] - Cs['T'], 'f_2C_D1T': -Cs['T'], 'df_2C_D0T': -Cs['T'],
         'f_1C_D0P': Ls*Ct['q'] + Ct['a'], 'f_1C_D1P': Ct['a'], 'f_1C_lho1_D0P': Ct['a'], 'df_1C_D0P': Ls*Ct['dq'],
         'f_1C_D0T': Ct['T']}
    K = {k: Lt*v for k, v in K.items()}   # the normalisation of diff_rot_coefficients.py (operators divide by L'^3)
    _cache[key] = K
    return K

def coef(name, n, l, m, offdiag):
    '''coefficient `name` (diff_rot_coefficients.py naming with f = f_n) for the target degree l and source degree l + offdiag'''
    M = _matrices(n, m)[name]
    ls = l + offdiag
    if ls < m or ls < 0 or ls >= M.shape[1] or l >= M.shape[0]:
        return 0.0
    return M[l, ls]

# radial labels of the force for f_n (codes a, s, p = utils.dr_code(n)), per (section, component): (coefficient, label template)
LABELS = {
    ('u', 'upol'): [('f_2C_D0P', 'u3{a}0_D0'), ('f_2C_D1P', 'u2{a}0_D1'), ('f_2C_lho1_D0P', 'u2{a}0lho1_D0'),
                    ('f_2C_lho2_D0P', 'u1{a}0lho2_D0'), ('f_2C_lho1_D1P', 'u1{a}0lho1_D1'), ('f_2C_D2P', 'u1{a}0_D2'),
                    ('df_2C_D0P', 'u3{s}0_D0'), ('df_2C_D1P', 'u2{s}0_D1'), ('df_2C_lho1_D0P', 'u2{s}0lho1_D0'),
                    ('d2f_2C_D0P', 'u3{p}0_D0')],
    ('u', 'utor'): [('f_2C_D0T', 'u2{a}0_D0'), ('f_2C_D1T', 'u1{a}0_D1'), ('df_2C_D0T', 'u2{s}0_D0')],
    ('v', 'upol'): [('f_1C_D0P', 'v2{a}0_D0'), ('f_1C_lho1_D0P', 'v1{a}0lho1_D0'), ('f_1C_D1P', 'v1{a}0_D1'), ('df_1C_D0P', 'v2{s}0_D0')],
    ('v', 'utor'): [('f_1C_D0T', 'v1{a}0_D0')],
}

def labels(n, section, component):
    '''[(coefficient name, label)] of the f_n force terms of a row block'''
    import utils as ut
    a, s, p = ut.dr_code(n)
    return [(c, t.format(a=a, s=s, p=p)) for c, t in LABELS[(section, component)]]

def max_dl(n, section, component):
    '''largest |Δl| of the f_n couplings: n for P→P and T→T, n + 1 for P↔T'''
    same = (section == 'u') == (component == 'upol')
    return n if same else n + 1
