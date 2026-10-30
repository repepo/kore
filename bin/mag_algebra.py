'''
NEW (2026-10-30): the magnetic couplings for any axisymmetric background field, on the recurrence algebra of angular.py.

WHY. operators.py holds the couplings of the Lorentz force and of the induction as closed forms in l and m, one set per
degree of B0 (degree 1 and 2, poloidal and toroidal), found once with sympy (tests/magnetic/derive.py,
tools/derive/fit_rows.py). A field that is a sum of many degrees (for example one built from a differential rotation,
background_field 'ferraro', or given degree by degree, 'general') would need a new set for every degree. This module
builds the same couplings at run time instead, for any degrees n, from the exact angular algebra that the differential
rotation uses (diff_rot.py): the background is axisymmetric, so every angular factor is a polynomial in x = cos θ, and every
operation is a short product of sparse banded matrices. Nothing is fitted; the result is exact up to round-off for any n
and l. With par.mag_couplings = 'algebra' it replaces the closed forms for every B0 type (a check: the eigenvalues agree to
round-off, tests/magnetic/algebra_check.py); 'general' and 'ferraro' need it.

METHOD. The background is B0 = Σ_n cnorm ∇×∇×(h_n Y_n⁰ 𝐫) + Σ_n cnorm_tor ∇×(k_n Y_n⁰ 𝐫) (Y_n⁰ = P_n(cos θ), kore's Schmidt
harmonic at m = 0). With ∂θP_n = -sinθ P_n'(x), its components and its current J0 = ∇×B0 are, degree by degree,
    poloidal h:  B0 = L0 h/r P_n r̂ - (h' + h/r) P_n' sinθ θ̂,    J0 = -(D h) P_n' sinθ φ̂,      D h = h'' + 2h'/r - L0 h/r²
    toroidal k:  B0 = k P_n' sinθ φ̂,                              J0 = L0 k/r P_n r̂ - (k' + k/r) P_n' sinθ θ̂
(L0 = n(n+1)). A perturbation of degree l is a Field (angular.py) with components on R_l, S_l, T_l whose radial parts are
monomials r^p X^(d) (X = P, T, F, G):
    u from P: q = L P/r, s = P' + P/r + λ1 P;   u from T: t = T;   b from F: q = L F/r, s = F' + F/r;   b from G: t = G;
    J = ∇×b from F: t = -(F'' + 2F'/r - L F/r²);   from G: q = L G/r, s = G' + G/r.
The products need V × F for the three kinds of background vectors c(x) r̂, c(x) sinθ θ̂ = c (x r̂ - ẑ) and
c(x) sinθ φ̂ = c ẑ×r̂, with c a polynomial in x:
    r̂ × (q, s, t) = (0, t, -s);   ẑ × F (angular.zcross);   (ẑ×r̂) × F = r̂ (ẑ⋅F) - ẑ (r̂⋅F),  ẑ⋅F = x F_r - sinθ F_θ,
    ẑ σ = σ x r̂ + r̂ × (ẑ × (σ r̂))      (since -σ sinθ θ̂ = r̂ × (σ sinθ φ̂)),
and the multiplication by c(x) (angular.mult with the matrices c(X), c'(X)). The Lorentz force is 𝐆 = J0 × b - B0 × J and the
induction 𝐄 = u × B0 = -B0 × u; each is a sum of (radial monomial) × (Field of sparse matrices [target, source]).
The rows (as derive.py, times -r⁵/L'³ (u) or -r³/L'³ (v, f, g), and Hendrik for the force): for a field with components
(q, s, t) at the target degree,
    r̂⋅∇×∇×V = L'(q - (r s)')/r²,    r̂⋅∇×V = L' t/r,    and ρ0 r̂⋅∇×∇×(V/ρ0) = L'(q - (r s)' + λ1 r s)/r²,
so the rows need one radial derivative of the monomials (product rule). Each final monomial r^P X^(d) B^(j) [λ_k] becomes a
label: {row}{rpow - P}{code}{j}[lho{k}][ohr1]_D{d}, code 'hp' + degree (cnorm h_n) or 'kt' + degree (cnorm_tor k_n),
rpow 5 (u), 3 (v), 2 (f, g); its coefficient is a sparse matrix [target l', source l], stored by diagonal (Δl).
'''
import sys
import numpy as np
import scipy.sparse as sp
from parameters import par
import angular

DIGITS = '0123456789abcdefghijklmnopqrstuvwxyz'
ROWPOW = {'u': 5, 'v': 3, 'f': 2, 'g': 2}


# ---------------------------------------------------------------------------------------------- the background degrees
def degrees():
    '''{'h': [n, ...], 'k': [n, ...]}: the degrees of the poloidal and the toroidal background (background_field)'''
    import background_field as bf
    return {'h': sorted(bf.pol_profiles()), 'k': sorted(bf.tor_profiles())}

def nmax():
    '''the largest degree of the background, so |Δl| <= nmax + 1'''
    d = degrees()
    return max(d['h'] + d['k'] + [0])

def code(B, n):
    '''the profile code of the background part B ('h' or 'k') of degree n'''
    if n >= len(DIGITS):
        sys.exit('mag_algebra: degree %d of B0 above the largest the profile codes allow (%d)' % (n, len(DIGITS) - 1))
    return ('hp' if B == 'h' else 'kt') + DIGITS[n]

def profile_functions():
    '''{code: f(r, Dorder)} of every background profile, for radial_profiles.burrito'''
    import background_field as bf
    out = {}
    for n, f in bf.pol_profiles().items():
        out[code('h', n)] = f
    for n, f in bf.tor_profiles().items():
        out[code('k', n)] = f
    return out

def odd_codes():
    '''the profile codes of odd parity at ricb = 0: h_n, k_n of odd n (utils.odd_profiles)'''
    d = degrees()
    return [code('h', n) for n in d['h'] if n % 2] + [code('k', n) for n in d['k'] if n % 2]


# ---------------------------------------------------------------------------------- radial monomials and their algebra
# A monomial is (p, X, d, B, n, j, lam): r^p X^(d) B_n^(j) λ_lam, X in 'PTFG', B in 'hk' (None: no background factor),
# lam = 0 (none) or k >= 1 for the k-th r-derivative of ln ρ0. An expression is {monomial: sparse matrix}.

def _add(e, key, M):
    '''e[key] += M (an expression is a dict monomial -> sparse matrix)'''
    e[key] = e[key] + M if key in e else M

def _scale(e, f):
    '''f times an expression'''
    return {k: f*v for k, v in e.items()}

def _times(e, mono):
    '''the product of an expression (perturbation monomials, no background) and a background monomial (c, p, B, n, j)'''
    c, p, B, n, j = mono
    out = {}
    for (p0, X, d, B0, n0, j0, lam), M in e.items():
        _add(out, (p0 + p, X, d, B, n, j, lam), c*M)
    return out

def _shift(e, c, p):
    '''c r^p times an expression'''
    return {(k[0] + p,) + k[1:]: c*M for k, M in e.items()}

def _deriv(e):
    '''d/dr of an expression (product rule on r^p, X^(d), B^(j), λ_lam)'''
    out = {}
    for (p, X, d, B, n, j, lam), M in e.items():
        if p != 0:
            _add(out, (p - 1, X, d, B, n, j, lam), p*M)
        _add(out, (p, X, d + 1, B, n, j, lam), M)
        if B is not None:
            _add(out, (p, X, d, B, n, j + 1, lam), M)
        if lam:
            _add(out, (p, X, d, B, n, j, lam + 1), M)
    return out

def _lam(e):
    '''λ1 times an expression'''
    out = {}
    for (p, X, d, B, n, j, lam), M in e.items():
        if lam:
            sys.exit('mag_algebra: a product of two (ln ρ0) derivatives has no label')
        _add(out, (p, X, d, B, n, j, 1), M)
    return out


# ----------------------------------------------------------------------------------------------- the angular operations
def _poly_mats(A, c):
    '''c(X) and c'(X) for a numpy Polynomial c in x (Horner)'''
    def ev(q):
        M = 0*A.I
        for a in q.coef[::-1]:
            M = A.X @ M + a*A.I
        return M.tocsr()
    return ev(c), ev(c.deriv())

def _rcross(F):
    '''r̂ × F: r̂ × S_l = -T_l, r̂ × T_l = S_l, r̂ × R_l = 0'''
    return angular.Field(0*F.q, F.t, -F.s)

def _vcross(A, kind, c, F):
    '''V × F for V = c(x) r̂ ('r'), c(x) sinθ θ̂ ('st') or c(x) sinθ φ̂ ('sp'); c a numpy Polynomial in x'''
    x = np.polynomial.Polynomial([0, 1])
    if kind == 'r':
        return A.mult(_rcross(F), *_poly_mats(A, c))
    if kind == 'st':   # c sinθ θ̂ = c (x r̂ - ẑ)
        return A.mult(_rcross(F), *_poly_mats(A, c*x)) - A.mult(A.zcross(F), *_poly_mats(A, c))
    # c sinθ φ̂ = c ẑ×r̂:  (ẑ×r̂)×F = r̂ (ẑ⋅F) - ẑ (r̂⋅F)
    zdotF = A.X @ F.q - A.sin_theta(F)
    sig = F.q
    zsig = angular.Field(A.X @ sig, 0*sig, 0*sig) + _rcross(A.zcross_r(sig))
    W = angular.Field(zdotF, 0*sig, 0*sig) - zsig
    return A.mult(W, *_poly_mats(A, c))


# ------------------------------------------------------------------------------------------- the perturbation fields
def _pert(A, X, kind):
    '''{monomial: Field} of the perturbation made of the unknown X: kind 'u' (velocity), 'b' (field) or 'J' (∇×b)'''
    I, L, Z = A.I, A.L, 0*A.I
    F = lambda q=None, s=None, t=None: angular.Field(Z if q is None else q, Z if s is None else s, Z if t is None else t)
    m = lambda p, d, lam=0: (p, X, d, None, None, None, lam)
    if kind == 'u' and X == 'P':
        return {m(-1, 0): F(q=L) + F(s=I), m(0, 1): F(s=I), m(0, 0, 1): F(s=I)}
    if (kind, X) in (('u', 'T'), ('b', 'G')):
        return {m(0, 0): F(t=I)}
    if kind == 'b' and X == 'F':
        return {m(-1, 0): F(q=L) + F(s=I), m(0, 1): F(s=I)}
    if kind == 'J' and X == 'F':
        return {m(0, 2): F(t=-I), m(-1, 1): F(t=-2*I), m(-2, 0): F(t=L)}
    if kind == 'J' and X == 'G':
        return {m(-1, 0): F(q=L) + F(s=I), m(0, 1): F(s=I)}
    return {}

def _background(n, B):
    '''the background vectors of degree n: {'B0': [...], 'J0': [...]}, each a list of (kind, c(x), [(coef, p, B, n, j)])'''
    Pn = np.polynomial.Legendre.basis(n).convert(kind=np.polynomial.Polynomial)
    dPn = Pn.deriv()
    L0 = n*(n + 1.)
    if B == 'h':
        return {'B0': [('r', Pn, [(L0, -1, 'h', n, 0)]), ('st', -dPn, [(1., 0, 'h', n, 1), (1., -1, 'h', n, 0)])],
                'J0': [('sp', -dPn, [(1., 0, 'h', n, 2), (2., -1, 'h', n, 1), (-L0, -2, 'h', n, 0)])]}
    return {'B0': [('sp', dPn, [(1., 0, 'k', n, 0)])],
            'J0': [('r', Pn, [(L0, -1, 'k', n, 0)]), ('st', -dPn, [(1., 0, 'k', n, 1), (1., -1, 'k', n, 0)])]}

def _product(A, vecs, pert, sign):
    '''sign Σ V × pert over the background vectors vecs: {component: expression}'''
    out = {'q': {}, 's': {}, 't': {}}
    for kind, c, rad in vecs:
        for pm, F in pert.items():
            W = _vcross(A, kind, c, F)
            for k in 'qst':
                Mk = (sign*W[k]).tocsr()
                if Mk.nnz == 0:
                    continue
                for mono in rad:
                    for key, M in _times({pm: Mk}, mono).items():
                        _add(out[k], key, M)
    return out


# --------------------------------------------------------------------------------------------------------- the rows
_coef, _band = {}, {}
def coefficients(m):
    '''
    {(row, column): {label: sparse matrix [target l', source l]}} for every background degree (cached): the rows 'u', 'v'
    (Lorentz force, columns 'F', 'G') and 'f', 'g' (induction, columns 'P', 'T'), in kore's A sign, without Hendrik and
    without the 1/L'^3 of the row (added by block()).
    '''
    if m in _coef:
        return _coef[m]
    out = build(m, degrees(), par.lmax + nmax() + 8)
    _coef[m] = out
    _band[m] = {rc: {lab: angular.band(M, nmax() + 2) for lab, M in d.items()} for rc, d in out.items()}
    return out

def build(m, degs, lw):
    '''the coefficients of coefficients() for the background degrees degs = {'h': [...], 'k': [...]} on the degrees 0..lw
    (also used by tests/magnetic/algebra_check.py without a run)'''
    A = angular.Algebra.get(m, lw)
    Lt = lambda M: A.L @ M
    out = {}
    for B in 'hk':
        for n in degs.get(B, []):
            bg = _background(n, B)
            for X in 'FG':   # Lorentz force G = J0 × b - B0 × J
                G = _product(A, bg['J0'], _pert(A, X, 'b'), 1.)
                G2 = _product(A, bg['B0'], _pert(A, X, 'J'), -1.)
                for k in 'qst':
                    for key, M in G2[k].items():
                        _add(G[k], key, M)
                # u: -r⁵/L'³ ρ0 r̂⋅∇×∇×(G/ρ0) = -(r³ L'/L'³) [q - s - r s' + λ1 r s];   v: -(r² L'/L'³) t
                s_r = _shift(G['s'], 1., 1)                 # r s
                urow = dict(G['q'])
                for key, M in _deriv(s_r).items():         # - (r s)'
                    _add(urow, key, -M)
                for key, M in _lam(s_r).items():           # + λ1 r s
                    _add(urow, key, M)
                _store(out, 'u', X, _shift(urow, -1., 3), Lt)
                _store(out, 'v', X, _shift(G['t'], -1., 2), Lt)
            for X in 'PT':   # induction E = u × B0 = -B0 × u
                E = _product(A, bg['B0'], _pert(A, X, 'u'), -1.)
                # f: -r³/L'³ r̂⋅∇×E = -(r² L'/L'³) t;   g: -r³/L'³ r̂⋅∇×∇×E = -(r L'/L'³)[q - s - r s']
                _store(out, 'f', X, _shift(E['t'], -1., 2), Lt)
                s_r = _shift(E['s'], 1., 1)
                grow = dict(E['q'])
                for key, M in _deriv(s_r).items():         # q - (r s)'
                    _add(grow, key, -M)
                _store(out, 'g', X, _shift(grow, -1., 1), Lt)
    return out

def _store(out, row, X, e, Lt):
    '''the row's monomials as labels, times L' of the target, added to out[(row, X)]'''
    d = out.setdefault((row, X), {})
    for (p, Xn, dd, B, n, j, lam), M in e.items():
        if M.nnz == 0 or abs(M).max() == 0:
            continue
        rx = ROWPOW[row] - p
        if not 0 <= rx <= 9 or B is None:
            sys.exit('mag_algebra: monomial r^%d %s^(%d) %s has no label in row %s' % (p, Xn, dd, B, row))
        lab = '%s%d%s%d%s%s_D%d' % (row, rx, code(B, n), j, 'lho%d' % lam if lam else '', 'ohr1' if row in 'uv' else '', dd)
        _add(d, lab, Lt(M).tocsr())

def labels():
    '''[(label, 'P', 'T', 'F' or 'G')]: every radial operator of the couplings and its column, for submatrices.py'''
    out = []
    for (row, X), d in coefficients(par.m).items():
        out += [(lab, X) for lab in d]
    return out

def dl_range():
    '''the Δl of the magnetic couplings: -(nmax + 1)..(nmax + 1)'''
    return range(-nmax() - 1, nmax() + 2)

def block(l, section, component, offdiag, M):
    '''
    The block of the Lorentz force (section 'u', 'v'; component 'bpol', 'btor') or of the induction (section 'f', 'g';
    component 'upol', 'utor') for the row degree l and the source degree l + offdiag: Σ coefficient × radial operator / L'^3
    (times Hendrik for the force), and the column block offset in units of N1. [0, 0] if nothing couples this Δl.
    '''
    import utils as ut
    X = {'bpol': 'F', 'btor': 'G', 'upol': 'P', 'utor': 'T'}[component]
    coefficients(par.m)
    bands = _band[par.m].get((section, X), {})
    out = 0
    for lab, bnd in bands.items():
        B = (bnd.shape[0] - 1)//2
        ls = l + offdiag
        if abs(offdiag) > B or ls < par.m or ls >= bnd.shape[1] or l >= bnd.shape[1]:
            continue
        c = bnd[offdiag + B, l]
        if c != 0:
            out = out + c*M[lab]
    if np.isscalar(out):
        return [0, 0]
    lp, lt = ut.ell(par.m, par.lmax, par.symm)[:2]
    lF, lG = ut.ell(par.m, par.lmax, ut.bsymm)[:2]
    rows = {'u': lp, 'v': lt, 'f': lF, 'g': lG}[section]
    cols = {'F': lF, 'G': lG, 'P': lp, 'T': lt}[X]
    offd = int(np.where(cols == l + offdiag)[0][0] - np.where(rows == l)[0][0])
    scale = par.Hendrik if section in 'uv' else 1.
    return [scale*out/(l*(l + 1.))**3, offd]
