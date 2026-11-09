#!/usr/bin/env python3
'''
NEW (2026-11-08, from conductive_IC's bgflow.py): an axisymmetric background flow U(r, θ) with poloidal and toroidal parts
(bgflow = 1): the steady flow of a forcing-12 run (a spherical Couette flow, linear or nonlinear), or one given by
expressions, with the entropy and the composition that come with it.

WHY. diff_rot = 1 gives kore a background rotation, the toroidal flow Gaspard ΔΩ r sinθ e_φ with even degrees of ΔΩ. The flow
that turning walls drive also has a meridional circulation and toroidal degrees that no rotation profile holds, and its
waves and its stability are what this is for. conductive_IC builds the terms of such a flow by quadrature in θ and dense
radial operators, for a uniform density in a shell, with the flow alone. Here they are built as the rest of kore's
couplings are: the angular coefficients exactly, from the recurrence algebra of angular.py (as mag_algebra.py does for the
background field), and the radial parts as labels of submatrices.py. So they hold for any density, in a full sphere, with
the entropy, the composition and the magnetic field, at any lmax.

THE BACKGROUND, per unit Ro = bgflow_Ro (the rate of the flow in units of the frame's rotation rate) and with G = Gaspard,
    U = Ro G Σ_n [ρ0⁻¹∇×∇×(ρ0 P_n Y_n 𝐫) + ∇×(T_n Y_n 𝐫)],   S = Ro Σ_n σ_n Y_n,   C = Ro Σ_n γ_n Y_n,   Y_n = P_n(cos θ),
so that ∇⋅(ρ0 U) = 0 (with Gaspard = 0, a frame at rest, G is replaced by 1: Ro is then the rate in units of 1/τ). S and C
are the parts of the entropy and of the composition that the flow brought about in the run that made it (optional). The
scalars come from bgflow_file (an .npz of Chebyshev series: written by this module from a forcing-12 run, or by
conductive_IC) or from bgflow_profiles (numpy expressions of r). With the components (q, s, t) of angular.py, U and its
vorticity Ω_U = ∇×U are, degree by degree,
    from P_n:  U: q = L0 P_n/r, s = P_n' + P_n/r + λ1 P_n          Ω_U: t = q/r - s' - s/r
    from T_n:  U: t = T_n                                          Ω_U: q = L0 T_n/r, s = T_n' + T_n/r
(L0 = n(n+1), λ1 = (ln ρ0)'). These radial functions times Ro G are the profile codes wq, ws, wt (+ the degree character
of mag_algebra.DIGITS); p0 σ_n, p0 σ_n', ρ0 γ_n, ρ0 γ_n' times Ro are wh, wd, wc, we (without p0 or ρ0 where the rows are
divided by it: utils.adiab_tw, adiab_c).

THE TERMS, the perturbation equations linearised about U, S and C:
    momentum   (U⋅∇)u + (u⋅∇)U = ∇(U⋅u) - U×ω - u×Ω_U   (ω = ∇×u; the gradient leaves the rows, which take the curl)
    entropy    p0 (U⋅∇s + u⋅∇S),   composition   ρ0 (U⋅∇c + u⋅∇C),   induction   ∇×(U×b)
Each is a sum of (radial monomial) × (Field of sparse matrices [target degree, source degree]), built with the operations
of mag_algebra (V × F, and here V⋅F for the scalars), in kore's rows: r⁵ r̂⋅∇×∇× (u), r³ r̂⋅∇× (v), r² (h, c), and those of
the induction (f, g, as mag_algebra). A monomial r^p X^(d) B_n^(j) [λ_k] of a row is the label
    {row}{ROWPOW[row] - p}w{B}{n}{j}[lho{k} | pss0 | dns0]_D{d}
(pss0, dns0: the p0 or ρ0 of U⋅∇s, U⋅∇c in rows that are not divided by it). Its coefficient is a sparse matrix [target l',
source l] stored by diagonal; a degree n of U couples l to |Δl| <= n + 1.
The assembly (assemble.bgflow_blocks) places the blocks in the rows u, v, h, c, f, g. A conducting solid at a wall (an
inner core, a thin layer, the resolved mantle) turns with U there, which must be a rigid rotation (check()): its field
gains the Doppler term of that rate (rigid_rate), as with a differential rotation.

AS A SCRIPT, in the folder of a forcing-12 run (m = 0, symm = 1), it writes bgflow_file from the run's steady flow, with
the degrees n <= nmax:
    python3 bin/bgflow.py nmax              the linear flow, from the solution in the folder, per unit Ro
    python3 bin/bgflow.py nmax nonlinear    the nonlinear flow at Ro = forcing amplitude/Gaspard, by Newton's method
                                            (newton(); it holds at bgflow_Ro = Ro only)
The amplitude is forcing_amplitude_icb, or forcing_amplitude_cmb when the inner wall is at rest. Not from a run with a
background field (magnetic = 1 and a B0): its steady state has a field induced by the flow, which the file would leave out.
Only parameters, angular and mag_algebra are imported at the top (utils imports this module for odd_codes).
'''
import os
import sys
import numpy as np
from parameters import par
import angular
import mag_algebra as ma

DIGITS = ma.DIGITS
KINDS = 'PTSC'   # the background's scalars: P_n, T_n (the flow), σ_n (the entropy), γ_n (the composition)
ROWPOW = {'u': 5, 'v': 3, 'h': 2, 'c': 2, 'f': 2, 'g': 2}
COLUMN = {'upol': 'P', 'utor': 'T', 'temp': 'S', 'comp': 'C', 'bpol': 'F', 'btor': 'G'}   # assemble's names
NMAX = len(DIGITS) - 1   # the largest degree the profile codes allow


# ======================================================================================================================
# 1. The background
# ======================================================================================================================

def rate_unit():
    '''the factor of U per unit Ro: Gaspard (Ro is a rate in units of the frame's rotation rate), 1 with Gaspard = 0 (a frame
    at rest: Ro is a rate in units of 1/τ)'''
    return par.Gaspard if par.Gaspard != 0 else 1.0

def _domain():
    '''the domain of the background's Chebyshev series: [ricb, 1], or [-1, 1] in a full sphere'''
    return [-1.0, 1.0] if par.ricb == 0 else [par.ricb, 1.0]

def _parse_key(key):
    '''(kind, n) of a key of bgflow_profiles: 'P2', 'T1', 'S0', 'C12', ...'''
    if not (isinstance(key, str) and len(key) >= 2 and key[0] in KINDS and key[1:].isdigit()):
        sys.exit("bgflow_profiles: the keys are the kind (P, T, S or C) and the degree, as 'P2', 'T1', 'S2', got %r" % (key,))
    return key[0], int(key[1:])

def _user_function(expr, key, n):
    '''f(r) of an expression of bgflow_profiles: numpy only (diff_rot's names), r, ricb bound; at |r|, times (-1)^n for r < 0
    (the parity of a regular scalar of degree n), when ricb = 0'''
    import diff_rot as dr
    if not isinstance(expr, str) or '#' in expr or not expr.strip():
        sys.exit('bgflow_profiles[%r]: a numpy expression of r as a string without "#" (it is written to params.dat), got %r'
                 % (key, expr))
    code = compile(expr, '<bgflow_profiles>', 'eval')
    def f(r):
        r = np.asarray(r, dtype=float)
        ra = np.abs(r) if par.ricb == 0 else r
        val = eval(code, {'__builtins__': {}}, dict(dr._USER_NS, r=ra, ricb=par.ricb))
        out = np.broadcast_to(np.asarray(val, dtype=float), r.shape).copy()
        return out*np.where(r < 0, (-1.0)**n, 1.0) if par.ricb == 0 else out
    return f

_file = {}
def load():
    '''the arrays of bgflow_file (cached)'''
    if not _file:
        if not os.path.isfile(par.bgflow_file):
            sys.exit('bgflow = 1: no background file %r (bgflow_file) in %s: write it with bin/bgflow.py in the folder of a '
                     'forcing-12 run, or give the background as bgflow_profiles' % (par.bgflow_file, os.getcwd()))
        with np.load(par.bgflow_file) as z:
            _file.update({k: z[k] for k in z.files})
        for k in ('nP', 'P', 'nT', 'T', 'ricb'):
            if k not in _file:
                sys.exit('bgflow_file %r has no %r: it needs nP, P, nT, T and ricb (bin/bgflow.py)' % (par.bgflow_file, k))
    return _file

def _lobatto(f, deg, dom):
    '''the Chebyshev series of degree deg on dom that interpolates f at the deg + 1 Chebyshev-Lobatto points, the ends of dom
    among them: a background that vanishes at a wall has a series that vanishes there to round-off, smooth or not (the
    points of the first kind of Chebyshev.interpolate miss the ends, and a kink in a full sphere moved the value there)'''
    import scipy.fft
    x = np.cos(np.pi*np.arange(deg + 1)/deg)
    c = scipy.fft.dct(f(0.5*(dom[0] + dom[1]) + 0.5*(dom[1] - dom[0])*x), type=1)/deg
    c[0] /= 2
    c[-1] /= 2
    return np.polynomial.Chebyshev(c, domain=dom)

_src = {}
def source():
    '''
    {kind: {n: Chebyshev}}: the background's scalars per unit Ro, P_n, T_n, σ_n, γ_n, as numpy Chebyshev series on [ricb,
    1] (on [-1, 1], of parity (-1)^n, in a full sphere): from bgflow_profiles if it is not empty (fitted at par.N points),
    otherwise from bgflow_file; cached. P_0 and T_0 are left out (they make no flow), as are degrees whose series are zero.
    '''
    if _src:
        return _src
    if par.bgflow not in (0, 1):   # (utils calls this at its import, before check())
        sys.exit('bgflow %r: 0 (no background flow) or 1 (bgflow.py)' % (par.bgflow,))
    out = {k: {} for k in KINDS}
    C = np.polynomial.Chebyshev
    dom = _domain()
    if par.bgflow_profiles:
        for key in sorted(par.bgflow_profiles):
            k, n = _parse_key(key)
            if n > NMAX:
                sys.exit('bgflow_profiles[%r]: degree %d above the largest the profile codes allow (%d)' % (key, n, NMAX))
            out[k][n] = _lobatto(_user_function(par.bgflow_profiles[key], key, n), max(par.N, 16) - 1, dom)
    else:
        z = load()
        for k in KINDS:
            ns = np.atleast_1d(z.get('n' + k, np.zeros(0, dtype=int)))
            if ns.size:
                for n, c in zip(ns, np.atleast_2d(z[k])):
                    out[k][int(n)] = C(np.asarray(c, dtype=float), domain=dom)
    for k in 'PT':
        out[k].pop(0, None)
    for k in KINDS:
        for n in list(out[k]):
            if not np.any(out[k][n].coef):
                del out[k][n]
    _src.update(out)
    return _src

def raw(K, n):
    '''the scalar of kind K and degree n per unit Ro at r >= 0 as it is given, for background.centre_check: its expression of
    bgflow_profiles (not its Chebyshev interpolant, which would hide a kink of the expression at r = 0), or its series of
    bgflow_file'''
    for key, expr in par.bgflow_profiles.items():
        if _parse_key(key) == (K, n):
            return _user_function(expr, key, n)
    return source()[K][n]

def reset():
    '''forget the background and the radial operators built from it (newton(): a new file at each step); the angular
    coefficients stay, cached by the degrees (coefficients)'''
    for d in (_file, _src, _fits, _stack):
        d.clear()

def degrees():
    '''{kind: [n, ...]}: the degrees of the background in use (S with thermal = 1, C with compositional = 1)'''
    src = source()
    use = {'P': True, 'T': True, 'S': bool(par.thermal), 'C': bool(par.compositional)}
    return {k: sorted(src[k]) if use[k] else [] for k in KINDS}

def nmax():
    '''the largest degree of the background in use, so that |Δl| <= nmax + 1'''
    return max([0] + [n for ns in degrees().values() for n in ns])


# ======================================================================================================================
# 2. The profile codes
# ======================================================================================================================
# B: q, s, t (U from P_n: q, s; from T_n: t, times Ro G), h, d (p0 σ_n, p0 σ_n'), c, e (ρ0 γ_n, ρ0 γ_n'), times Ro

def code(B, n):
    '''the profile code of (B, n): 'w' B and the degree character'''
    if not 0 <= n <= NMAX:
        sys.exit('bgflow: degree %d of the background above the largest the profile codes allow (%d)' % (n, NMAX))
    return 'w' + B + DIGITS[n]

def _codes():
    '''[(B, n)] of every profile code in use'''
    d = degrees()
    return ([(B, n) for n in d['P'] for B in 'qs'] + [('t', n) for n in d['T']] +
            [(B, n) for n in d['S'] for B in 'hd'] + [(B, n) for n in d['C'] for B in 'ce'])

def _value(B, n, r):
    '''the radial function of the code (B, n) at the radii r, from the background's series'''
    import utils as ut
    import radial_profiles as rap
    src = source()
    r = np.asarray(r, dtype=float)
    A = par.bgflow_Ro*rate_unit()
    if B == 't':
        return A*src['T'][n](r)
    if B in 'qs':
        P = src['P'][n](r)
        if B == 'q':
            return A*n*(n + 1.)*P/r
        return A*(src['P'][n].deriv()(r) + P/r + rap.logrhoX(r, 1)*P)
    k = 'S' if B in 'hd' else 'C'
    f = src[k][n] if B in 'hc' else src[k][n].deriv()
    if k == 'S':
        w = 1.0 if ut.adiab_tw else rap.prf.pressure(r, 0)
    else:
        w = 1.0 if ut.adiab_c else rap.prf.density(r, 0)
    return par.bgflow_Ro*w*f(r)

_fits = {}
def _fit(B, n):
    '''Chebyshev coefficients of the code (B, n) on kore's radial grid (as diff_rot._fit), cached'''
    import utils as ut
    if (B, n) not in _fits:
        _fits[(B, n)] = ut.chebco_f(lambda r, rpower: (r**rpower)*_value(B, n, r), par.N, par.ricb, ut.rcmb, 1e-14, 0)
    return _fits[(B, n)]

def profile(B, n, r, Dorder=0):
    '''the code (B, n) or its Dorder-th r-derivative at the radii r, from its fit'''
    import utils as ut
    return ut.funcheb(_fit(B, n), np.atleast_1d(np.asarray(r, dtype=float)), par.ricb, ut.rcmb, Dorder)[:, -1]

def profile_functions():
    '''{code: f(r, Dorder)} of every profile code, for radial_profiles.burrito'''
    return {code(B, n): (lambda r, Dorder, B=B, n=n: profile(B, n, r, Dorder)) for B, n in _codes()}

def odd_codes():
    '''the codes of odd parity at ricb = 0 (utils.odd_profiles): P_n, T_n, σ_n, γ_n have the parity (-1)^n, so q, s (P_n/r, P_n')
    and d, e (the derivatives) are odd for even n, and t, h, c for odd n'''
    return [code(B, n) for B, n in _codes() if (n + (B in 'qsde')) % 2]


# ======================================================================================================================
# 3. The angular couplings
# ======================================================================================================================

def _vectors(K, n):
    '''the background vectors of the scalar K of degree n, as mag_algebra._background: lists of (kind, c(x), [(coef, p, B, n,
    j)]), kind 'r' (c r̂), 'st' (c sinθ θ̂) or 'sp' (c sinθ φ̂): {'U': U, 'W': Ω_U} for P, T; {'G': ∇S} for S, ∇C for C'''
    Pn = np.polynomial.Legendre.basis(n)
    dPn = Pn.deriv()
    L0 = n*(n + 1.)
    if K == 'P':   # U = q Y r̂ + s r∇Y,  r∇Y = -sinθ P_n' θ̂;  Ω_U = t r∇Y×r̂ = t sinθ P_n' φ̂
        return {'U': [('r', Pn, [(1., 0, 'q', n, 0)]), ('st', -dPn, [(1., 0, 's', n, 0)])],
                'W': [('sp', dPn, [(1., -1, 'q', n, 0), (-1., -1, 's', n, 0), (-1., 0, 's', n, 1)])]}
    if K == 'T':
        return {'U': [('sp', dPn, [(1., 0, 't', n, 0)])],
                'W': [('r', Pn, [(L0, -1, 't', n, 0)]), ('st', -dPn, [(1., 0, 't', n, 1), (1., -1, 't', n, 0)])]}
    g, h = ('d', 'h') if K == 'S' else ('e', 'c')   # ∇S = σ' Y r̂ + (σ/r) r∇Y
    out = [('r', Pn, [(1., 0, g, n, 0)])]
    if n > 0:
        out.append(('st', -dPn, [(1., -1, h, n, 0)]))
    return {'G': out}

def _pert(A, X, kind):
    '''{monomial: Field} of a perturbation: kind 'u' (u from P, T) and 'b' (b from F, G) as mag_algebra._pert; 'w' (ω = ∇×u
    from P, T); 'g' (∇X of the scalar X = S or C: q = X', s = X/r)'''
    I, L, Z = A.I, A.L, 0*A.I
    F = lambda q=None, s=None, t=None: angular.Field(Z if q is None else q, Z if s is None else s, Z if t is None else t)
    m = lambda p, d, lam=0: (p, X, d, None, None, None, lam)
    if kind == 'w' and X == 'P':   # t = q/r - s' - s/r with q = L P/r, s = P' + P/r + λ1 P
        return {m(-2, 0): F(t=L), m(0, 2): F(t=-I), m(-1, 1): F(t=-2*I), m(-1, 0, 1): F(t=-I), m(0, 0, 2): F(t=-I),
                m(0, 1, 1): F(t=-I)}
    if kind == 'w' and X == 'T':   # q = L T/r, s = T' + T/r
        return {m(-1, 0): F(q=L) + F(s=I), m(0, 1): F(s=I)}
    if kind == 'g':
        return {m(0, 1): F(q=I), m(-1, 0): F(s=I)}
    return ma._pert(A, X, kind)

def _vdot(A, kind, c, F):
    '''V⋅F, a scalar (matrix [target, source]), for V = c(x) r̂ ('r'), c sinθ θ̂ ('st'), c sinθ φ̂ ('sp'):
    sinθ F_θ = -D s + i m t (angular.sin_theta),  sinθ F_φ = i m s + D t'''
    M = ma._poly_mats(A, c)[0]
    if kind == 'r':
        return M @ F.q
    if kind == 'st':
        return M @ A.sin_theta(F)
    return M @ (1j*A.m*F.s + A.D @ F.t)

def _dot(A, vecs, pert, sign):
    '''sign Σ V⋅pert over the background vectors vecs: an expression {monomial: matrix}'''
    out = {}
    for kind, c, rad in vecs:
        for pm, F in pert.items():
            M = (sign*_vdot(A, kind, c, F)).tocsr()
            if M.nnz == 0:
                continue
            for mono in rad:
                for key, Mk in ma._times({pm: M}, mono).items():
                    ma._add(out, key, Mk)
    return out

def _curl_rows(G):
    '''{q, s, t} of a vector -> (q - (r s)', t): what r̂⋅∇×∇× and r̂⋅∇× take of it (times L'/r², L'/r)'''
    row = dict(G['q'])
    for key, M in ma._deriv(ma._shift(G['s'], 1., 1)).items():
        ma._add(row, key, -M)
    return row, G['t']

def build(m, degs, lw, phys):
    '''
    {(row, column): {label: sparse matrix [target l', source l]}} of the background of degrees degs = {kind: [n, ...]} on the
    degrees 0..lw, in kore's A sign, times L' of the target in the rows u, v, f, g, without the 1/L'^3 of the row (added by
    block()). phys: {'hydro', 'thermal', 'compositional', 'magnetic': bool, 'heat': 'pss0' or '', 'comp': 'dns0' or ''}
    (the p0, ρ0 of U⋅∇s, U⋅∇c, '' where the rows are divided by it). Also used by tests/bgflow without a run.
    '''
    A = angular.Algebra.get(m, lw)
    Lt = lambda M: A.L @ M
    out = {}
    pu = {X: _pert(A, X, 'u') for X in 'PT'}
    pw = {X: _pert(A, X, 'w') for X in 'PT'}
    for K in 'PT':
        for n in degs.get(K, []):
            V = _vectors(K, n)
            if phys['hydro']:
                for X in 'PT':   # F' = Ω_U × u - U × ω, on the left of the momentum equation (as the Coriolis force)
                    F = ma._product(A, V['W'], pu[X], 1.)
                    for k, e in ma._product(A, V['U'], pw[X], -1.).items():
                        for key, M in e.items():
                            ma._add(F[k], key, M)
                    urow, vrow = _curl_rows(F)   # u: r⁵ r̂⋅∇×∇×F' = r³ L'[q - (r s)'];  v: r³ r̂⋅∇×F' = r² L' t
                    _store(out, 'u', X, ma._shift(urow, 1., 3), Lt)
                    _store(out, 'v', X, ma._shift(vrow, 1., 2), Lt)
            for sec, X, on, w in (('h', 'S', phys['thermal'], phys['heat']), ('c', 'C', phys['compositional'], phys['comp'])):
                if on:   # -r² w U⋅∇s (the right-hand side, as the heat rows of A hold)
                    _store(out, sec, X, ma._shift(_dot(A, V['U'], _pert(A, X, 'g'), -1.), 1., 2), None, w)
            if phys['magnetic']:
                for X in 'FG':   # E = U × b on the right of the induction equation: f -r² L' t, g -r L'[q - (r s)']
                    grow, frow = _curl_rows(ma._product(A, V['U'], _pert(A, X, 'b'), 1.))
                    _store(out, 'f', X, ma._shift(frow, -1., 2), Lt)
                    _store(out, 'g', X, ma._shift(grow, -1., 1), Lt)
    for K, sec, on in (('S', 'h', phys['thermal']), ('C', 'c', phys['compositional'])):
        if on and phys['hydro']:
            for n in degs.get(K, []):
                G = _vectors(K, n)['G']
                for X in 'PT':   # -r² u⋅∇S (S with its p0 in the codes h, d)
                    _store(out, sec, X, ma._shift(_dot(A, G, pu[X], -1.), 1., 2), None)
    return ma._prune(out)

def _store(out, row, X, e, Lt, extra=''):
    '''the row's monomials as labels (times L' of the target with Lt), added to out[(row, X)]; extra: the second profile
    (pss0, dns0) of the monomials without a λ'''
    d = out.setdefault((row, X), {})
    for (p, Xn, dd, B, n, j, lam), M in e.items():
        if M.nnz == 0 or abs(M).max() == 0:
            continue
        rx = ROWPOW[row] - p
        if not 0 <= rx <= 9 or B is None or (lam and extra):
            sys.exit('bgflow: monomial r^%d %s^(%d) %s%s λ%d has no label in row %s' % (p, Xn, dd, B, n, lam, row))
        lab = '%s%d%s%d%s_D%d' % (row, rx, code(B, n), j, 'lho%d' % lam if lam else extra, dd)
        ma._add(d, lab, (Lt(M) if Lt is not None else M).tocsr())

def physics():
    '''the phys argument of build() for this run'''
    import utils as ut
    return {'hydro': bool(par.hydro), 'thermal': bool(par.thermal), 'compositional': bool(par.compositional),
            'magnetic': bool(par.magnetic), 'heat': '' if ut.adiab_tw else 'pss0', 'comp': '' if ut.adiab_c else 'dns0'}

_coef, _band, _stack = {}, {}, {}
def _ckey(m):
    '''what the coefficients of build() depend on: the order m, the degrees of the background, lmax and the physics; not the
    values of the background, which are in the profile codes (newton() keeps the coefficients from one iteration to the next)'''
    d = degrees()
    return (m, tuple(tuple(d[k]) for k in KINDS), par.lmax, tuple(sorted(physics().items())))

def coefficients(m):
    '''the coefficients of build() for this run's background at the order m, cached; also stored by diagonal'''
    key = _ckey(m)
    if key not in _coef:
        nm = nmax()
        _coef[key] = build(m, degrees(), par.lmax + nm + 8, physics())
        _band[key] = {rc: {lab: angular.band(M, nm + 2) for lab, M in d.items()} for rc, d in _coef[key].items()}
    return _coef[key]

def labels():
    '''[(label, column)]: every radial operator of the background's terms and its column ('P', 'T', 'S', 'C', 'F', 'G'), for
    submatrices.py'''
    return [(lab, X) for (row, X), d in coefficients(par.m).items() for lab in d]

def dl_range():
    '''the Δl of the blocks: -(nmax + 1)..(nmax + 1)'''
    return range(-nmax() - 1, nmax() + 2)

def _stacked(section, X, M):
    '''
    The radial operators M[label] of the rows of section and the column X, stacked for block(): (bands, S, indices, indptr,
    shape), with bands[k, Δl + B, l] the coefficient of the k-th label (angular.band), the union of the sparsity patterns of
    the operators as the CSR indices and indptr, and S the sparse matrix [entry of the pattern, k] of the entries of the k-th
    operator, so that a block Σ_k c_k M[label_k] has the values S @ c on the pattern: one product for each block, in place of
    a sum of sparse matrices. None if no label couples them. Cached, by the operators themselves (newton() reloads them).
    '''
    import scipy.sparse as ss
    ck = _ckey(par.m)
    bands = _band[ck].get((section, X), {})
    labs = sorted(bands)
    if not labs:
        return None
    key = (ck, section, X, tuple(id(M[lab]) for lab in labs))
    if key not in _stack:
        mats = [ss.coo_matrix(M[lab]) for lab in labs]
        nr, nc = mats[0].shape
        assert all(Mk.shape == (nr, nc) for Mk in mats) and len({bands[lab].shape for lab in labs}) == 1
        lin = [Mk.row.astype(np.int64)*nc + Mk.col for Mk in mats]
        pat = np.unique(np.concatenate(lin))   # sorted: row by row, the columns in order (the CSR order)
        S = ss.csr_matrix((np.concatenate([Mk.data for Mk in mats]),
                           (np.concatenate([np.searchsorted(pat, li) for li in lin]),
                            np.concatenate([np.full(len(li), k) for k, li in enumerate(lin)]))), shape=(len(pat), len(labs)))
        indptr = np.searchsorted(pat//nc, np.arange(nr + 1))
        _stack[key] = (np.array([bands[lab] for lab in labs]), S, (pat % nc).astype(np.int32), indptr.astype(np.int32),
                       (nr, nc))
    return _stack[key]

def block(l, section, component, offdiag, M):
    '''
    The block of the background's terms for the row degree l of section 'u', 'v', 'h', 'c', 'f' or 'g', the column
    component ('upol', 'utor', 'temp', 'comp', 'bpol', 'btor') and the source degree l + offdiag: Σ coefficient × radial
    operator M[label] / L'^3 (_stacked), with the column block offset in units of N1 (the degrees of F, G those of utils.ell
    with the field's symmetry ut.bsymm). [0, 0] if nothing couples this Δl.
    '''
    import scipy.sparse as ss
    import utils as ut
    X = COLUMN[component]
    coefficients(par.m)
    st = _stacked(section, X, M)
    if st is None:
        return [0, 0]
    bands, S, indices, indptr, shape = st
    B = (bands.shape[1] - 1)//2
    ls = l + offdiag
    if abs(offdiag) > B or ls < par.m or ls >= bands.shape[2] or l >= bands.shape[2]:
        return [0, 0]
    c = bands[:, offdiag + B, l]
    if not np.any(c):
        return [0, 0]
    out = ss.csr_matrix((S @ c, indices, indptr), shape=shape, copy=True)   # (a copy: eliminate_zeros works in place, and
    out.eliminate_zeros()                                                    # the pattern is the cache's)
    lp, lt = ut.ell(par.m, par.lmax, par.symm)[:2]
    lF, lG = ut.ell(par.m, par.lmax, ut.bsymm)[:2]
    rows = {'u': lp, 'v': lt, 'h': lp, 'c': lp, 'f': lF, 'g': lG}[section]
    cols = {'P': lp, 'T': lt, 'S': lp, 'C': lp, 'F': lF, 'G': lG}[X]
    offd = int(np.where(cols == l + offdiag)[0][0] - np.where(rows == l)[0][0])
    return [out/(l*(l + 1.))**3, offd]


# ======================================================================================================================
# 4. The walls, the checks
# ======================================================================================================================

def rigid_rate(r, tol=1e-10):
    '''
    (rate, ok): the angular velocity of U at the wall of radius r relative to the frame, Ro G T_1(r)/r (in the run's units), and
    whether U is a rigid rotation there: no flow through it (P_n(r) = 0), none along it but the rotation (P_n'(r) = 0, and
    T_n(r) = 0 for n other than 1), to tol relative to the largest |P_n|, |T_n| in the fluid. A conducting solid behind the
    wall turns with it (assemble.py: the Doppler term of its field).
    '''
    src = source()
    rg = np.linspace(max(par.ricb, 1e-3), 1, 64)
    big = max([1e-300] + [np.abs(f(rg)).max() for k in 'PT' for f in src[k].values()])
    rr = np.array([float(r)])
    T1 = src['T'].get(1)
    rate = par.bgflow_Ro*rate_unit()*float(T1(rr)[0])/r if (T1 is not None and r > 0) else 0.0
    bad = [abs(P(rr)[0]) + r*abs(P.deriv()(rr)[0]) for P in src['P'].values()] + \
          [abs(T(rr)[0]) for n, T in src['T'].items() if n != 1]
    return rate, all(v <= tol*big for v in bad)

def _wall_radii():
    '''the radii of the walls: [1], and ricb with an inner core'''
    return [1.0] + ([par.ricb] if par.ricb > 0 else [])

def _density_mismatch(z):
    '''the largest relative difference between the density bgflow_file was written for and this run's (0 without poloidal
    parts, which alone depend on it); a file without one (conductive_IC's) was written for a uniform density'''
    import radial_profiles as rap
    rr = np.asarray(z['rho_r'], dtype=float) if 'rho_r' in z else np.linspace(max(par.ricb, 0.05), 1, 8)
    ref = np.asarray(z['rho'], dtype=float) if 'rho' in z else np.ones_like(rr)
    now = rap.prf.density(rr, 0)
    return float(np.abs(now - ref).max()/np.abs(ref).max())

def check():
    '''stops on what the background flow does not cover, or a background that cannot be used (submatrices.py calls it)'''
    import background as bg
    if par.bgflow not in (0, 1):
        sys.exit('bgflow %r: 0 (no background flow) or 1 (bgflow.py)' % (par.bgflow,))
    if not par.bgflow:
        return
    if par.centrifugal:
        sys.exit('bgflow = 1 with centrifugal = 1 is not covered: the terms of the background flow on the deformed body '
                 '(its metric parts) are not built')
    if par.rotdyn:
        sys.exit('bgflow = 1 with rotdyn = 1 is not covered: the torques of the background flow on the turning solids are '
                 'not built')
    if par.topo_drag == 'rekier':
        sys.exit("bgflow = 1 with topo_drag 'rekier' is not covered: the drag is built for the frequency of the flow "
                 "relative to the wall, which a background flow there would change with latitude")
    if par.ViscosD == 0 and par.ricb > 0:
        sys.exit('bgflow = 1 with ViscosD = 0 needs ricb = 0 (the inviscid formulation is for the full sphere only)')
    if not (isinstance(par.bgflow_Ro, (int, float, np.integer, np.floating)) and np.isfinite(par.bgflow_Ro)):
        sys.exit('bgflow_Ro = %r: a finite number' % (par.bgflow_Ro,))
    if not isinstance(par.bgflow_profiles, dict):
        sys.exit("bgflow_profiles: a dict such as {'P2': 'expression of r', 'T1': '...'} (empty: bgflow_file), got %r"
                 % (par.bgflow_profiles,))
    seen = {}
    for key in par.bgflow_profiles:
        k, n = _parse_key(key)
        if (k, n) in seen:
            sys.exit('bgflow_profiles: %r and %r are the same scalar' % (seen[(k, n)], key))
        seen[(k, n)] = key
        if (k == 'S' and not par.thermal) or (k == 'C' and not par.compositional):
            sys.exit('bgflow_profiles[%r]: the %s of the background needs %s = 1' %
                     (key, 'entropy' if k == 'S' else 'composition', 'thermal' if k == 'S' else 'compositional'))
    if not par.bgflow_profiles:
        z = load()
        if abs(float(z['ricb']) - par.ricb) > 1e-12:
            sys.exit('bgflow_file %r is for ricb = %r, the run has ricb = %r' % (par.bgflow_file, float(z['ricb']), par.ricb))
        if 'Ro' in z and abs(float(z['Ro']) - par.bgflow_Ro) > 1e-9*max(1.0, abs(float(z['Ro']))):
            sys.exit('bgflow_file %r holds the nonlinear flow at Ro = %r, which holds at that Ro only: set bgflow_Ro = %r'
                     % (par.bgflow_file, float(z['Ro']), float(z['Ro'])))
    src = source()
    for k in KINDS:
        for n, f in src[k].items():
            if not np.all(np.isfinite(f.coef)):
                sys.exit('bgflow: %s_%d of the background is not finite' % (k, n))
            if n > NMAX:
                sys.exit('bgflow: degree %d of the background above the largest the profile codes allow (%d)' % (n, NMAX))
    if not par.bgflow_profiles and src['P'] and _density_mismatch(load()) > 1e-8:
        sys.exit("bgflow_file %r was written for another density (%s): its poloidal flow ρ0⁻¹∇×∇×(ρ0 P 𝐫) is that "
                 "density's (∇⋅(ρ0 U) = 0), so it is not a flow of this run's" %
                 (par.bgflow_file, 'a uniform one, conductive_IC' if 'rho' not in load() else 'its structure'))
    odd = [('P', n) for n in src['P'] if n % 2] + [('T', n) for n in src['T'] if n % 2 == 0] + \
          [(k, n) for k in 'SC' for n in degrees()[k] if n % 2]
    if odd:
        sys.exit('bgflow: the background must be symmetric about the equator (P_n, σ_n, γ_n of even n, T_n of odd n), as '
                 'kore assumes for the perturbations (symm); it has %s' % ', '.join('%s_%d' % kn for kn in odd))
    big = max([1e-300] + [np.abs(f(np.linspace(max(par.ricb, 1e-3), 1, 64))).max() for k in 'PT' for f in src[k].values()])
    for r in _wall_radii():   # (relative to the whole flow: a rotation's file has round-off poloidal parts)
        through = [n for n, P in src['P'].items() if abs(P(np.array([r]))[0]) > 1e-10*big]
        if through:
            sys.exit('bgflow: the background flows through the wall r = %g (P_n(%g) != 0 for n = %s): kore\'s walls are '
                     'impenetrable' % (r, r, through))
    if par.magnetic:
        walls = [(par.mantle in ('TWA', 'resolved'), 1.0, 'the outer wall (mantle %r)' % par.mantle),
                 (par.ricb > 0 and (par.innercore in bg.CONDUCTING_CORES or par.innercore == 'TWA'), par.ricb,
                  'the inner wall (innercore %r)' % par.innercore)]
        for on, r, name in walls:
            if on and not rigid_rate(r)[1]:
                sys.exit('bgflow = 1 with a conducting solid at %s needs a rigid rotation of the background there '
                         '(P_n = P_n\' = 0 and T_n = 0 but for n = 1, at r = %g): a solid turns rigidly' % (name, r))
        steadiness()

_warned = []
def steadiness():
    '''
    With a background field B0 (magnetic = 1): how far B0 is from steady under the background flow, max |∇×(U×B0)| r /
    (max |U| max |B0|) over the fluid (with the differential rotation's flow too, diff_rot = 1), reported once; above 1e-7 a
    warning, or a stop with ferraro_policy = 'refuse' (as background.magnetic_diffrot): the linear problem assumes the
    background is maintained. On a grid of radii and Gauss-Legendre nodes in cos θ, from the profiles and their derivatives.
    '''
    import background_field as bf
    if par.B0 is None and par.B0_tor is None:
        return 0.0
    nB = max([1] + list(bf.pol_profiles()) + list(bf.tor_profiles()))
    x, _ = np.polynomial.legendre.leggauss(nmax() + nB + 8)
    rr = np.linspace(par.ricb + 0.02*(1 - par.ricb), 0.98, 48)
    Uf = flow_grid(rr, x, nd=1, dtheta=True)                 # U, ∂rU and ∂θU
    Bv, dBr, dBt = _b0_grid(rr, x)
    sth, cth = np.sqrt(1 - x**2)[None, :], x[None, :]
    r = rr[:, None]
    cross = lambda a, c: [a[1]*c[2] - a[2]*c[1], a[2]*c[0] - a[0]*c[2], a[0]*c[1] - a[1]*c[0]]
    E = cross(Uf['U'][0], Bv)
    Er = [a + c for a, c in zip(cross(Uf['U'][1], Bv), cross(Uf['U'][0], dBr))]      # ∂r E
    Et = [a + c for a, c in zip(cross(Uf['Ut'], Bv), cross(Uf['U'][0], dBt))]        # ∂θ E
    curl = [(cth/sth*E[2] + Et[2])/r, -(E[2]/r + Er[2]), E[1]/r + Er[1] - Et[0]/r]
    U = np.sqrt(sum(np.abs(c)**2 for c in Uf['U'][0])).max()
    B = np.sqrt(sum(np.abs(c)**2 for c in Bv)).max()
    mis = float(np.sqrt(sum(np.abs(c)**2 for c in curl)).max()/max(U*B, 1e-300)) if U*B > 0 else 0.0
    if not _warned:
        print('bgflow: the background field under the background flow, max |∇×(U×B0)| r / (max |U| max |B0|) = %.2e' % mis)
    if mis > 1e-7:
        msg = ('the background field B0 and the background flow are not a steady background together (|∇×(U×B0)| r = %.2e '
               'of |U| |B0|); the linear problem assumes the background is maintained' % mis)
        if par.ferraro_policy == 'refuse':
            sys.exit(msg + " (ferraro_policy = 'refuse')")
        if not _warned:
            print('WARNING: ' + msg)
    _warned.append(1)
    return mis

def _b0_grid(rr, x):
    '''B0 and its r- and θ-derivatives on the grid (rr, x): three lists of (r, θ, φ) components'''
    import background_field as bf
    P = np.polynomial.Legendre
    sth = np.sqrt(1 - x**2)
    r = rr[:, None]
    z = 0*np.outer(rr, x)
    B, Br, Bt = [z, z, z], [z, z, z], [z, z, z]
    for n, f in bf.pol_profiles().items():   # B_r = L0 h/r P_n, B_θ = -(h' + h/r) sinθ P_n'
        L0 = n*(n + 1.)
        h = [f(rr, d)[:, None] for d in (0, 1, 2)]
        Pn, dPn, d2Pn = P.basis(n)(x), P.basis(n).deriv()(x), P.basis(n).deriv(2)(x)
        a, da = h[1] + h[0]/r, h[2] + h[1]/r - h[0]/r**2
        B = [B[0] + L0*h[0]/r*Pn, B[1] - a*sth*dPn, B[2]]
        Br = [Br[0] + L0*(h[1]/r - h[0]/r**2)*Pn, Br[1] - da*sth*dPn, Br[2]]
        Bt = [Bt[0] - L0*h[0]/r*sth*dPn, Bt[1] - a*(x*dPn - sth**2*d2Pn), Bt[2]]   # ∂θ P_n = -sinθ P_n', ∂θ(sinθ P_n') = x P_n' - sin²θ P_n''
    for n, f in bf.tor_profiles().items():   # B_φ = k sinθ P_n'
        k = [f(rr, d)[:, None] for d in (0, 1)]
        dPn, d2Pn = P.basis(n).deriv()(x), P.basis(n).deriv(2)(x)
        B = [B[0], B[1], B[2] + k[0]*sth*dPn]
        Br = [Br[0], Br[1], Br[2] + k[1]*sth*dPn]
        Bt = [Bt[0], Bt[1], Bt[2] + k[0]*(x*dPn - sth**2*d2Pn)]
    return B, Br, Bt

def description():
    '''the background as text for params.dat (a '#' comment): its expressions, or its file and the first 12 hex digits of its
    sha256'''
    if not par.bgflow:
        return ''
    if par.bgflow_profiles:
        return 'bgflow: ' + ' ; '.join('%s = %s' % (k, par.bgflow_profiles[k]) for k in sorted(par.bgflow_profiles))
    import hashlib
    with open(par.bgflow_file, 'rb') as f:
        return 'bgflow: %s sha256 %s' % (par.bgflow_file, hashlib.sha256(f.read()).hexdigest()[:12])


# ======================================================================================================================
# 5. The background on a grid (spin_doctor)
# ======================================================================================================================

def flow_grid(rr, x, nd=2, dtheta=False):
    '''
    U on the grid of radii rr and x = cos θ: {'U': [U, ∂rU, ..., ∂r^nd U], 'W': [Ω_U, ∂rΩ_U]} and with dtheta {'Ut': ∂θU},
    each a list of (r, θ, φ) components, arrays [len(rr), len(x)], from the profile codes (in the run's units, with Ro G).
    The differential rotation's flow is added with diff_rot = 1 (to U and its r-derivatives only: steadiness()).
    '''
    P = np.polynomial.Legendre
    sth = np.sqrt(1 - x**2)
    r = rr[:, None]
    z = 0*np.outer(rr, x)
    U = [[z, z, z] for d in range(nd + 1)]
    W = [[z, z, z] for d in range(2)]
    Ut = [z, z, z]
    pr = lambda B, n, d: profile(B, n, rr, d)[:, None]
    d = degrees()
    for n in d['P']:
        Pn, dPn, d2Pn = P.basis(n)(x), P.basis(n).deriv()(x), P.basis(n).deriv(2)(x)
        for k in range(nd + 1):
            U[k] = [U[k][0] + pr('q', n, k)*Pn, U[k][1] - pr('s', n, k)*sth*dPn, U[k][2]]
        q, s = [pr('q', n, k) for k in (0, 1, 2)], [pr('s', n, k) for k in (0, 1, 2)]
        w0 = q[0]/r - s[1] - s[0]/r
        w1 = q[1]/r - q[0]/r**2 - s[2] - s[1]/r + s[0]/r**2
        W = [[W[0][0], W[0][1], W[0][2] + w0*sth*dPn], [W[1][0], W[1][1], W[1][2] + w1*sth*dPn]]
        Ut = [Ut[0] - q[0]*sth*dPn, Ut[1] - s[0]*(x*dPn - sth**2*d2Pn), Ut[2]]
    for n in d['T']:
        L0 = n*(n + 1.)
        Pn, dPn, d2Pn = P.basis(n)(x), P.basis(n).deriv()(x), P.basis(n).deriv(2)(x)
        for k in range(nd + 1):
            U[k] = [U[k][0], U[k][1], U[k][2] + pr('t', n, k)*sth*dPn]
        t = [pr('t', n, k) for k in (0, 1, 2)]
        W = [[W[0][0] + L0*t[0]/r*Pn, W[0][1] - (t[1] + t[0]/r)*sth*dPn, W[0][2]],
             [W[1][0] + L0*(t[1]/r - t[0]/r**2)*Pn, W[1][1] - (t[2] + t[1]/r - t[0]/r**2)*sth*dPn, W[1][2]]]
        Ut = [Ut[0], Ut[1], Ut[2] + t[0]*(x*dPn - sth**2*d2Pn)]
    if dtheta and par.diff_rot:   # the rotation G ΔΩ r sinθ e_φ (for steadiness() only)
        import diff_rot as dr
        for n in dr.degrees():
            Pn, dPn = P.basis(n)(x), P.basis(n).deriv()(x)
            for k in range(nd + 1):   # ∂r^k (r f_n)
                f = dr.fX(n, rr, k)[:, None]*r + (k*dr.fX(n, rr, k - 1)[:, None] if k else 0)
                U[k] = [U[k][0], U[k][1], U[k][2] + par.Gaspard*f*sth*Pn]
            Ut = [Ut[0], Ut[1], Ut[2] + par.Gaspard*dr.fX(n, rr, 0)[:, None]*r*(x*Pn - sth**2*dPn)]   # ∂θ(sinθ P_n)
    out = {'U': U, 'W': W}
    if dtheta:
        out['Ut'] = Ut
    return out

def scalar_grid(K, rr, x):
    '''∇S (K = 'S') or ∇C (K = 'C') of the background, without p0 or ρ0, on the grid (rr, x): [r, θ, φ] components'''
    P = np.polynomial.Legendre
    sth = np.sqrt(1 - x**2)
    z = 0*np.outer(rr, x)
    out = [z, z, z]
    for n in degrees()[K]:
        f = source()[K][n]
        a, da = par.bgflow_Ro*f(rr)[:, None], par.bgflow_Ro*f.deriv()(rr)[:, None]
        out = [out[0] + da*P.basis(n)(x), out[1] - a/rr[:, None]*sth*P.basis(n).deriv()(x), out[2]]
    return out


# ======================================================================================================================
# 6. The writer, Newton's method (python3 bin/bgflow.py nmax [nonlinear])
# ======================================================================================================================

def _read_field(name):
    '''the first solution of field name that solve.py wrote: name.npy or real_name.field, imag_name.field (utils.load_field)'''
    import utils as ut
    x = ut.load_field(name, 0)
    if x is None:
        sys.exit('bgflow.py: no solution of the %s in this folder (%s.npy, or the text files): run the forcing-12 problem first'
                 % (name, name))
    return x

def _full_series(blk, ls):
    '''the rows of Chebyshev coefficients of a block of kore's solution (len(ls) rows of N1), on the full N in a full sphere
    (the stored coefficients of one parity, (-1)^l, at their places)'''
    import utils as ut
    c = np.asarray(blk).reshape(len(ls), ut.N1)
    if par.ricb > 0:
        return c
    out = np.zeros((len(ls), par.N), dtype=c.dtype)
    for k, l in enumerate(ls):
        out[k, int(l) % 2::2] = c[k]
    return out

def write(x, amp, nmax, Ro=None, fname='bgflow.npz'):
    '''
    Writes fname from kore's solution vector x of a forcing-12 run (the fields are x + c.c., m = 0), with the degrees n <=
    nmax: P_n, T_n per unit Ro (divided by amp, the forcing amplitude in the run's units, so that Ro G P_n is the flow of rate
    Ro), σ_n and γ_n per unit Ro (divided by amp/G), the density at 8 radii (the poloidal flow belongs with it), ricb, which
    wall turns and the other's rate relative to it; Ro marks a nonlinear flow (it holds at bgflow_Ro = Ro only).
    '''
    import utils as ut
    import radial_profiles as rap
    lp, lt = ut.ell(0, par.lmax, 1)[:2]
    Rounit = amp/rate_unit()
    out = {'ricb': par.ricb}
    out['nP'], out['P'] = lp[lp <= nmax], 2*np.real(_full_series(x[:ut.n], lp))[lp <= nmax]/amp
    out['nT'], out['T'] = lt[lt <= nmax], 2*np.real(_full_series(x[ut.n:2*ut.n], lt))[lt <= nmax]/amp
    if par.thermal:
        o = ut.offset['h']
        out['nS'], out['S'] = lp[lp <= nmax], 2*np.real(_full_series(x[o:o + ut.n], lp))[lp <= nmax]/Rounit
    if par.compositional:
        o = ut.offset['c']
        out['nC'], out['C'] = lp[lp <= nmax], 2*np.real(_full_series(x[o:o + ut.n], lp))[lp <= nmax]/Rounit
    out['rho_r'] = np.linspace(max(par.ricb, 0.05), 1, 8)
    out['rho'] = rap.prf.density(out['rho_r'], 0)
    inner = par.ricb > 0 and par.forcing_amplitude_icb != 0
    out['wall'] = 'inner' if inner else 'outer'
    other = par.forcing_amplitude_cmb if inner else (par.forcing_amplitude_icb if par.ricb > 0 else 0.0)
    out['ratio'] = other/amp
    if Ro is not None:
        out['Ro'] = Ro
    np.savez(fname, **out)

def _forced_solution():
    '''kore's solution vector of the forced run in the folder, in the order of A (solve.split_fields)'''
    import utils as ut
    x = np.zeros(ut.sizmat, dtype=complex)
    x[:2*ut.n] = _read_field('flow')
    if par.thermal:
        x[ut.offset['h']:ut.offset['h'] + ut.n] = _read_field('thermal')
    if par.compositional:
        x[ut.offset['c']:ut.offset['c'] + ut.n] = _read_field('composition')
    return x

def _assemble():
    '''A (sparse) and the right-hand side b of the forced problem, from submatrices.py (with the run's ncpus processes) and
    assemble.py (one rank) run here with the current parameters. operators.py reads submatrices.npz when it is imported, so
    it is imported (or reloaded) after submatrices.main has written it.'''
    import importlib
    import utils as ut
    import submatrices
    import radial_profiles as rap
    for k in [k for k in rap.proffdir if k[0] == 'w']:   # the background's codes, rebuilt from the new background
        del rap.proffdir[k]
    ut.adiab_tw, ut.adiab_c = ut.adiabatic_rows()   # (they depend on bgflow, which newton() changes)
    if par.bgflow and par.ricb == 0:   # the parities of the background's codes (utils adds them at import with bgflow = 1)
        ut.odd_profiles.extend(c for c in odd_codes() if c not in ut.odd_profiles)
    submatrices.main(max(1, int(par.ncpus)))   # (the pool of the run's ncpus processes)
    if 'operators' in sys.modules:
        importlib.reload(sys.modules['operators'])
    import operators
    import assemble
    assemble.op = operators
    assemble.main()
    return ut.load_csr('A.npz'), _load_npz('B_forced.npz').toarray().ravel()

def _load_npz(f):
    '''a CSR matrix saved as assemble.py saves them (data, indices, indptr, shape)'''
    import scipy.sparse as sp
    with np.load(f) as z:
        return sp.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))

def newton(nmax, dRo=0.15, tol=1e-9, fname='bgflow.npz'):
    '''
    The nonlinear steady flow of a forcing-12 problem at Ro = amp/G (amp the forcing amplitude). kore's x (the fields are
    U = 2x, with the entropy and the composition) solves A x + ½ B(2x) x = b, with A x = b the linear problem and B(V) the
    terms of the background V; B(2x) is also the Jacobian of the quadratic term (it is symmetric in its two flows), so a
    Newton step is
        (A + B_k) x_k+1 = b + ½ B_k x_k,    B_k = B(2 x_k),
    with every degree of x_k in B_k. B_k is kore's own background matrix: A assembled with bgflow = 1 and the background
    x_k (written to fname at Ro = R, the step's), minus A assembled with bgflow_Ro = 0 (the same rows and bases). Ro is
    reached from 0 in steps of at most dRo; the matrices are nearly dense, so the solves are dense. Writes fname with the
    degrees n <= nmax. The folder's submatrices.npz, A.npz and B_forced.npz are those of the run again at the end.
    '''
    import scipy.linalg as sl
    amp = _amplitude()
    Ro = amp/rate_unit()
    own = (par.bgflow, par.bgflow_file, par.bgflow_profiles, par.bgflow_Ro)   # the run's, restored at the end
    xlin = _forced_solution()   # (it imports utils, which sets its flags with the run's own parameters)
    par.bgflow, par.bgflow_file, par.bgflow_profiles = 1, fname, {}
    write(xlin, amp, par.lmax + 1, None, fname)        # any background with the run's degrees, for the Ro = 0 matrix
    par.bgflow_Ro = 0.0
    reset()
    A0, b = _assemble()
    if abs(A0.imag).max() > 1e-12*abs(A0).max() or abs(b.imag).max() > 1e-12*abs(b).max():
        sys.exit('bgflow.newton: the forced problem of m = 0 is expected to be real')
    A0d, b = A0.real.toarray(), b.real
    x1 = sl.solve(A0d, b)/Ro                            # the linear flow per unit Ro
    x, R0 = x1.copy(), 1.0
    nstep = max(1, int(np.ceil(abs(Ro)/dRo)))
    for R in Ro*np.arange(1, nstep + 1)/nstep:
        x = R*x1 + (R/R0)**2*(x - R0*x1)                # first guess, with the O(Ro²) part of the last step
        for k in range(30):
            par.bgflow_Ro = R
            write(x, R*rate_unit(), par.lmax + 1, R, fname)   # U_k = 2 x_k, every degree
            reset()
            Ak = _assemble()[0].real.toarray()
            Bk = Ak - A0d
            xn = sl.solve(Ak, (R/Ro)*b + 0.5*(Bk @ x))
            dx, x = np.linalg.norm(xn - x)/np.linalg.norm(xn), xn
            print('Ro = %.6g, step %d: |x_k+1 - x_k|/|x_k+1| = %.1e' % (R, k, dx), flush=True)
            if dx < tol:
                break
        else:
            sys.exit('bgflow.newton: no convergence at Ro = %g (smaller steps dRo?)' % R)
        R0 = R
    print('|U - Ro U_lin|/|Ro U_lin| = %.4f' % (np.linalg.norm(x - Ro*x1)/np.linalg.norm(Ro*x1)))
    write(x, amp, nmax, Ro, fname)
    par.bgflow, par.bgflow_file, par.bgflow_profiles, par.bgflow_Ro = own
    reset()
    _assemble()   # the folder's operators and matrices, of the run itself again
    return x

def _amplitude():
    '''the forcing amplitude of the wall the file is per unit of: the inner wall's, or the outer wall's when it is at rest'''
    return par.forcing_amplitude_icb if (par.ricb > 0 and par.forcing_amplitude_icb != 0) else par.forcing_amplitude_cmb

def _check_writer():
    '''stops on a run whose steady state the file cannot hold'''
    if not (par.forcing == 12 and par.forcing_frequency == 0 and par.m == 0 and par.symm == 1):
        sys.exit('bgflow.py writes the steady flow of a forcing-12 run (forcing 12, forcing_frequency 0, m 0, symm 1)')
    if _amplitude() == 0:
        sys.exit('bgflow.py: no wall turns (forcing_amplitude_icb = forcing_amplitude_cmb = 0)')
    if par.magnetic and (par.B0 is not None or par.B0_tor is not None):
        sys.exit('bgflow.py: the run has a background field (magnetic = 1, B0 %r, B0_tor %r): its steady state has a field '
                 'that the flow induces from B0, of the order of the flow, which the background file would leave out'
                 % (par.B0, par.B0_tor))
    if par.centrifugal:
        sys.exit('bgflow.py: not from a run with centrifugal = 1 (bgflow = 1 does not cover the deformed body)')


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[2:] not in ([], ['nonlinear']):
        sys.exit('usage: python3 bin/bgflow.py nmax [nonlinear]   (in the folder of a forcing-12 run)')
    nmax_ = int(sys.argv[1])
    # the module itself, not this script's copy (__main__): submatrices.py, operators.py and assemble.py import it, and
    # Newton's method must reset the background they read (reset), not a copy of it
    import bgflow
    bgflow._check_writer()
    if sys.argv[2:] == ['nonlinear']:
        bgflow.newton(nmax_)
    else:
        bgflow.write(bgflow._forced_solution(), bgflow._amplitude(), nmax_)
    print('%s written, with the degrees n <= %d' % ('bgflow.npz', nmax_))
