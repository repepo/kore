'''
Differential rotation (docs/differential_rotation_details.md): everything it adds to kore, in one place.

WHY THIS WAY (docs/derivations.md, part B). A rotation that varies with latitude couples each degree l of the perturbation
to l ± 1, l ± 2, ..., up to the degree of ΔΩ (plus one between P and T). Kore builds its matrices block by block, a radial
operator times a coefficient for each (l, Δl), so it needs those coefficients for every l up to lmax and for every even
degree n of ΔΩ. They are computed here at run time, with the exact recurrence algebra of angular.py: ΔΩ is a series of
Legendre polynomials P_n(cos θ), multiplying by P_n(cos θ) is a polynomial in the matrix X of the recurrence for cos θ, and
every term of the force (ΔΩ i m u, ΔΩ 2ẑ×u, the shear terms) is a short product of such sparse matrices. This works for any
profile and any n without new algebra, and at any lmax (no quadrature, no overflowing harmonics); closed forms for P0 and P2
are kept only as a check (tests/diff_rot/diff_rot_coefficients.py, gen_check.py).

The background rotates at Ω = Ω_ref [1 + ΔΩ(r, θ)] with
    ΔΩ = diff_rot_ampl Σ_n f_n(r) P_n(cos θ),   n even,
seen in the frame of Ω_ref as the flow U0 = Gaspard ΔΩ r sinθ e_φ, together with the latitudinal entropy
S_tw = Σ_n σ_n(r) P_n(cos θ) of thermal-wind balance (or prescribed). They add
    momentum rows (sections u, v):  W = Gaspard [ΔΩ (i m u + 2 ẑ×u) + sinθ (r ∂ᵣΔΩ u_r + ∂_θΔΩ u_θ) e_φ]
    heat rows (section h):          p0 [i m Gaspard ΔΩ s + u_r ∂ᵣS_tw + (u_θ/r) ∂_θS_tw]
Sections of this module:
  1. profiles: the shapes f_n of the predefined types, of 'tabulated' (a table of Ω(r, θ)) and of 'user' (numpy
     expressions in par.diff_rot_profiles), and their Chebyshev fits for submatrices.py (profile codes zf, zd, zs:
     f_n, r f_n', r² f_n'');
  2. thermal wind: the σ_n by the Legendre-series algebra of angular.py (profile codes tw, tp: p0 σ_n, p0 σ_n');
  3. angular couplings: of the momentum force (the 12 primitives and the coefficients of the radial operators) and of
     the heat terms (P_n(X), P_n'(X), P_n'(X) D), all from the recurrence matrices of angular.py;
  4. the terms: the labels for submatrices.py, the blocks for operators.py and the Δl ranges for assemble.py.
Only parameters and angular are imported at the top: utils and radial_profiles are imported where needed (radial_profiles
imports this module from burrito), so the angular part can be imported and tested without a run directory.
'''
import sys
import numpy as np
import scipy.special as scsp
from parameters import par
import angular

# diff_rot_type as written to params.dat (spin_doctor): its index in this tuple. Append new types at the end.
TYPES = ('Y20', 'Y20-wall-bounded', 'shellular', 'shellular_boussinesq', 'cylindrical', 'conical', 'solar', 'rigid',
         'solar_P4', 'tabulated', 'user')
DIGITS = '0123456789abcdefghijklmnopqrstuvwxyz'   # the degree character of the profile codes


# ======================================================================================================================
# 1. Profiles
# ======================================================================================================================

def solar_x(r):
    '''fractional solar radius: r r_out with a solar structure (length unit r_out R☉), r otherwise'''
    import background as bg
    return r*par.r_out if par.structure in bg.SOLAR else r

_tab = {}
def _tabulated_fits():
    '''
    'tabulated': f_n (n = 0, 2, ..., diff_rot_nmax) of Ω/Ω_ref - 1 = ome/diff_rot_frame - 1 at each table radius by
    Gauss-Legendre quadrature (64 nodes) of a cubic spline in cos θ, then a least-squares Chebyshev fit in x of degree
    diff_rot_rdeg (at most half the number of table radii used) over the table radii within 0.01 of the shell, or with
    diff_rot_window = 'sun' (the default when ricb = 0) one even fit over the whole Sun. Returns {n: Chebyshev}.
    '''
    if _tab:
        return _tab
    from scipy.interpolate import CubicSpline
    D = np.load(par.diff_rot_file)
    ome, rt, th = D['ome'], D['r'], D['theta']          # ome[θ, r] in nHz, θ in rad, r in R☉
    o = np.argsort(np.cos(th))
    xc = np.cos(th)[o]
    ns = list(range(0, par.diff_rot_nmax + 1, 2))
    xg, wg = np.polynomial.legendre.leggauss(64)
    P = np.array([np.polynomial.legendre.Legendre.basis(n)(xg) for n in ns])
    dO = CubicSpline(xc, ome[o, :]/par.diff_rot_frame - 1, axis=0)(xg)     # (64, nr)
    fn = (((2*np.array(ns) + 1)/2)[:, None]*P*wg) @ dO                       # (len(ns), nr)
    xa, xb = solar_x(np.array([par.ricb, 1.0]))
    win = par.diff_rot_window
    if win is None and par.ricb == 0:
        win = 'sun'
    C = np.polynomial.chebyshev.Chebyshev
    if win is not None:   # one fit over the whole Sun, even in r, the same for any ricb
        if win != 'sun':
            sys.exit("diff_rot_window must be None or 'sun'")
        L = xb + 0.01
        k = rt <= L
        r1, f1 = rt[k], fn[:, k]
        dx = r1[1] - r1[0]
        rp = np.arange(r1[0] - dx, 0, -dx)[::-1]                 # below the first table radius: its value
        r1 = np.concatenate([rp, r1]); f1 = np.concatenate([np.repeat(f1[:, :1], rp.size, axis=1), f1], axis=1)
        xx = np.concatenate([-r1[::-1], r1])
        for i, n in enumerate(ns):
            c = C.fit(xx, np.concatenate([f1[i, ::-1], f1[i]]), par.diff_rot_sun_deg, domain=[-L, L]).coef.copy()
            c[1::2] = 0
            _tab[n] = C(c, domain=[-L, L])
        res = [np.abs(_tab[n](r1) - f1[i]).max()*par.diff_rot_frame for i, n in enumerate(ns)]
        print("diff_rot_type 'tabulated': %s, whole-Sun even fit of degree %d on [0, %.3f]; max |fit - f_n| (nHz):" %
              (par.diff_rot_file, par.diff_rot_sun_deg, L), ' '.join('%.2e' % v for v in res))
        return _tab
    k = (rt >= xa - 0.01) & (rt <= xb + 0.01)
    deg = min(par.diff_rot_rdeg, int(k.sum())//2)
    for i, n in enumerate(ns):
        _tab[n] = C.fit(rt[k], fn[i, k], deg, domain=[xa, xb])
    # report: fit residual of each f_n and the truncation error of Ω at the table nodes inside the shell (nHz)
    res = [np.abs(_tab[n](rt[k]) - fn[i, k]).max()*par.diff_rot_frame for i, n in enumerate(ns)]
    rec = sum(np.outer(np.polynomial.legendre.Legendre.basis(n)(np.cos(th)), _tab[n](rt[k])) for n in ns)
    trunc = np.abs(par.diff_rot_frame*(1 + rec) - ome[:, k]).max()
    print("diff_rot_type 'tabulated': %s, %d radii in [%.3f, %.3f], degree %d; max |fit - f_n| (nHz):" %
          (par.diff_rot_file, k.sum(), xa, xb, deg), ' '.join('%.2e' % v for v in res),
          '; max |Ω(P0..P%d) - Ω| = %.2f nHz' % (par.diff_rot_nmax, trunc))
    return _tab

_USER_NS = {k: getattr(np, k) for k in ('sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh',
                                         'exp', 'log', 'log10', 'sqrt', 'abs', 'sign', 'pi', 'where', 'minimum', 'maximum')}
_USER_NS['erf'], _USER_NS['erfc'] = scsp.erf, scsp.erfc

def _user_shape(expr):
    '''f(r) of a 'user' profile: the numpy expression expr in r, x = solar_x(r) and p = diff_rot_param'''
    code = compile(expr, '<diff_rot_profiles>', 'eval')
    def f(r):
        '''f_n(r): the user's expression evaluated with numpy only (no builtins), r, x and p bound'''
        r = np.asarray(r, dtype=float)
        val = eval(code, {'__builtins__': {}}, dict(_USER_NS, r=r, x=solar_x(r), p=par.diff_rot_param))
        return np.broadcast_to(np.asarray(val, dtype=float), r.shape).copy()
    return f

def shapes():
    '''{n: f_n(r)}: the shape of each Legendre degree of ΔΩ for par.diff_rot_type, before the factor diff_rot_ampl'''
    t, p = par.diff_rot_type, par.diff_rot_param
    one = lambda r: np.ones_like(np.asarray(r, dtype=float))
    if t == 'rigid':                   # ΔΩ = 1: a rigid offset of the frame rate (exact Doppler identity)
        return {0: one}
    if t == 'Y20':                     # P2(cos θ), unnormalised
        return {2: one}
    if t == 'Y20-wall-bounded':        # (1 - r)(r - ricb) P2(cos θ)
        return {2: lambda r: (1 - r)*(r - par.ricb)}
    if t == 'shellular':               # Ω ∝ r^σ, σ = diff_rot_param (Baruteau & Rieutord 2013)
        return {0: lambda r: r**p - 1}
    if t == 'shellular_boussinesq':    # (1 - r²)/2 (Mirouh et al. 2016: diff_rot_ampl = N0²)
        return {0: lambda r: 0.5*(1 - r**2)}
    if t == 'cylindrical':             # r² sin²θ = (2/3) r² (P0 - P2) (Baruteau & Rieutord 2013)
        return {0: lambda r: (2/3)*r**2, 2: lambda r: -(2/3)*r**2}
    if t == 'conical':                 # sin²θ = (2/3)(P0 - P2) (Guenel et al. 2016)
        return {0: lambda r: (2/3)*one(r), 2: lambda r: -(2/3)*one(r)}
    if t == 'solar':                   # helioseismic fit, frame 435 nHz, tanh tachocline at x = 0.71 of width 0.1
        H = lambda r: 0.5*(1 + np.tanh(2*(solar_x(r) - 0.71)/0.1))
        return {0: lambda r: (-7.695/435)*H(r), 2: lambda r: (-86.04/435)*H(r)}
    if t == 'solar_P4':                # Ω/2π = Ω_RZ + [Ω_CZ(θ) - Ω_RZ] (1 + erf((x - rtc)/dtc))/2, Ω_CZ = c0 + c2 cos²θ + c4 cos⁴θ
        c0, c2, c4, orz, rtc, dtc = par.diff_rot_solar
        st = lambda r: 0.5*(1 + scsp.erf((solar_x(r) - rtc)/dtc))
        On = {0: c0 + c2/3 + c4/5, 2: 2*c2/3 + 4*c4/7, 4: 8*c4/35}   # cos²θ = (1 + 2 P2)/3, cos⁴θ = 1/5 + 4 P2/7 + 8 P4/35
        out = {0: lambda r: (orz + (On[0] - orz)*st(r))/par.diff_rot_frame - 1}
        for n in (2, 4):
            out[n] = lambda r, n=n: On[n]*st(r)/par.diff_rot_frame
        return out
    if t == 'tabulated':               # Ω(r, θ)/Ω_ref - 1 from diff_rot_file
        if par.diff_rot_nmax < 0 or par.diff_rot_nmax % 2:
            sys.exit('diff_rot_nmax must be even and >= 0')
        fits = _tabulated_fits()
        return {n: (lambda r, n=n: fits[n](solar_x(r))) for n in sorted(fits)}
    if t == 'user':                    # numpy expressions of r, x, p for each even degree
        prof = par.diff_rot_profiles
        if not prof:
            sys.exit("diff_rot_type 'user' needs par.diff_rot_profiles = {n: 'expression of r, x, p', ...}")
        for n, e in prof.items():
            if not (isinstance(n, (int, np.integer)) and n >= 0 and n % 2 == 0):
                sys.exit('diff_rot_profiles: the degrees must be even integers >= 0, got %r' % (n,))
            if not isinstance(e, str) or '#' in e:
                sys.exit('diff_rot_profiles[%d]: a string without "#" (it is written to params.dat), got %r' % (n, e))
        return {int(n): _user_shape(prof[n]) for n in sorted(prof)}
    sys.exit('diff_rot_type %r: unknown, the types are %s' % (t, ', '.join(TYPES)))

def degrees():
    '''the even degrees n of ΔΩ'''
    return sorted(shapes())

def n_max():
    '''the highest degree of ΔΩ'''
    return max(degrees())

def tw_degrees():
    '''the degrees of S_tw: up to n_max + 2 (thermal_wind = 1, linear in ΔΩ), 2 n_max + 2 (2, the exact (1 + ΔΩ)²),
    2 (3, prescribed), none (0)'''
    tw = par.thermal_wind if par.thermal else 0
    top = {0: 0, 1: n_max() + 2, 2: 2*n_max() + 2, 3: 2}[tw]
    return list(range(2, top + 1, 2))

def tw_nmax():
    '''the highest degree of S_tw (0 without the thermal wind)'''
    d = tw_degrees()
    return max(d) if d else 0

def codes(n):
    '''profile codes of f_n, r f_n', r² f_n'' (zf, zd, zs) and of p0 σ_n, p0 σ_n' (tw, tp) for the degree n'''
    if n >= len(DIGITS):
        sys.exit('differential rotation: degree %d above the largest one the profile codes allow (%d)' % (n, len(DIGITS) - 1))
    c = DIGITS[n]
    return {'f': 'zf' + c, 'd': 'zd' + c, 's': 'zs' + c, 'tw': 'tw' + c, 'tp': 'tp' + c}

def check():
    '''stops on settings that cannot work (called by submatrices.py)'''
    for name, new in (('diff_rot_amplitude', "diff_rot_ampl (and diff_rot_param for the σ of 'shellular')"),
                      ('diff_rot_generic', 'nothing: there is one coupling builder now (tests/diff_rot/gen_check.py checks it)'),
                      ('diff_rot_hold', "nothing: the table is used as it is (the hold copied the table's latitudinal shear "
                                        'at x_h into the whole core)')):
        if hasattr(par, name):   # removed parameters: they would otherwise be ignored without notice
            sys.exit('par.%s is no longer read: use %s' % (name, new))
    if par.thermal and par.thermal_wind not in (0, 1, 2, 3):
        sys.exit('thermal_wind must be 0, 1, 2 or 3')
    if par.thermal and par.thermal_wind in (1, 2) and par.Beyonce == 0:
        sys.exit('thermal_wind = 1, 2 needs Beyonce > 0 (S_tw is O(1/Beyonce)): set thermal_wind = 0 or set Beyonce after set_scales()')
    for n in degrees() + tw_degrees():
        codes(n)

def description():
    '''the profile definition as text for params.dat (a '#' comment): the 'user' expressions, empty otherwise'''
    if par.diff_rot_type != 'user':
        return ''
    return 'user: ' + ' ; '.join('zf%d = %s' % (n, par.diff_rot_profiles[n]) for n in sorted(par.diff_rot_profiles)) + \
           ' ; p = %r' % par.diff_rot_param

# Chebyshev fits on kore's radial grid (as every profile: utils.fonzie), even in r when ricb = 0 (evaluated at |r|)
_fits = {}
def _fit(key, func):
    '''Chebyshev coefficients of func(r, rpower) on kore's radial domain, cached by key'''
    import utils as ut
    if key not in _fits:
        _fits[key] = ut.chebco_f(func, par.N, par.ricb, ut.rcmb, 1e-14, 0)   # func(r, rpower = 0), as utils.fonzie
    return _fits[key]

def _eval(c, r, Dorder):
    '''the Chebyshev series c (from _fit) or its Dorder-th r-derivative at the radii r'''
    import utils as ut
    return ut.funcheb(c, r, par.ricb, ut.rcmb, Dorder)[:, -1]

def term_amp(n):
    '''a_n, the factor of the P_n term of ΔΩ: par.diff_rot_term_amp[n/2] (1 for missing entries)'''
    a = par.diff_rot_term_amp
    return a[n//2] if n//2 < len(a) else 1.0

def fX(n, r, Dorder):
    '''diff_rot_ampl a_n f_n(r) and its radial derivatives, from the Chebyshev fit of the shape (at |r| if ricb = 0)'''
    f = shapes()[n]
    c = _fit(('f', n), lambda rr, rpower: (rr**rpower)*f(np.abs(rr) if par.ricb == 0 else rr))
    return _eval(c, r, Dorder)*par.diff_rot_ampl*term_amp(n)

def raw(n, r):
    '''diff_rot_ampl a_n f_n(r) evaluated directly (no fit), at |r| if ricb = 0'''
    return par.diff_rot_ampl*term_amp(n)*shapes()[n](np.abs(r) if par.ricb == 0 else r)


# ======================================================================================================================
# 2. Thermal wind
# ======================================================================================================================

def _solve_D(h, m=0):
    '''σ with D σ = h (D of angular.py at order m = 0, the Legendre polynomials), σ_0 = 0: rows = degrees, forward
    substitution of h_k = (k+2) q_{k+1} σ_{k+1} - (k-1) p_{k-1} σ_{k-1}'''
    K = h.shape[0] - 1
    p, q = angular.coefficients(m, K + 1)
    s = np.zeros_like(h)
    for k in range(K):
        prev = (k - 1)*p[k - 1]*s[k - 1] if k >= 1 else 0.
        s[k + 1] = (h[k] + prev)/((k + 2)*q[k + 1])
    return s

_sig = {}
def sigma_coefficients(r):
    '''
    Legendre coefficients σ_k(r) of S_tw, k = 0..tw_nmax (rows; σ_0 = 0), at the radii r. Thermal-wind balance (2.9),
        ∂_x S_tw = -(r² G²/(Beyonce g)) [x ∂ᵣF + (1 - x²) ∂_x F/r],   F = 2ΔΩ (thermal_wind 1) or (1 + ΔΩ)² - 1 (2),
    written in Legendre space with the m = 0 matrices of angular.py (multiplication by x is X, by (1 - x²) d/dx is D, by
    a series is Σ_n coefficient_n P_n(X)), so that D σ = (1 - x²) ∂_x S_tw is solved for σ exactly; times
    thermal_wind_amplitude. thermal_wind = 3: σ_2 = -(2/3) dSdtheta (∂_θS_tw = dSdtheta sin 2θ).
    '''
    import radial_profiles as rap
    r = np.asarray(r, dtype=float)
    key = (r.tobytes(), r.shape)
    if key in _sig:
        return _sig[key]
    K = tw_nmax()
    out = np.zeros((K + 1, r.size))
    if par.thermal_wind == 3:
        out[2] = -(2/3)*par.dSdtheta
    elif par.thermal_wind in (1, 2):
        lw = 2*n_max() + 8
        A = angular.Algebra.get(0, lw)
        nd = degrees()
        O, Or = np.zeros((lw + 1, r.size)), np.zeros((lw + 1, r.size))
        for n in nd:
            O[n], Or[n] = fX(n, r, 0), fX(n, r, 1)
        def mult(Ccoef, Bcoef):   # product of two series, B of the degrees of ΔΩ
            '''the Legendre coefficients of the product of two Legendre series (radius by radius): Σ_n B_n P_n(X) C'''
            return sum((A.legendre(n)[0] @ Ccoef)*Bcoef[n][None, :] for n in nd)
        if par.thermal_wind == 2:   # F = 2ΔΩ + ΔΩ², ∂ᵣF = 2(1 + ΔΩ) ∂ᵣΔΩ
            F, Fr = 2*O + mult(O, O), 2*Or + 2*mult(Or, O)
        else:
            F, Fr = 2*O, 2*Or
        rhs = A.X @ Fr + (A.D @ F)/r[None, :]                        # x ∂ᵣF + (1 - x²) ∂_x F/r
        q = r**2*par.Gaspard**2/(par.Beyonce*rap.prf.gravity(r, 0))
        h = -(rhs - A.X @ (A.X @ rhs))*q[None, :]                    # (1 - x²) ∂_x S_tw
        out = _solve_D(h)[:K + 1]*par.thermal_wind_amplitude
    _sig[key] = out
    return out

def sigma(n, r, rpower):
    '''r^rpower σ_n(r)'''
    s = sigma_coefficients(r)
    return (np.asarray(r)**rpower)*(s[n] if n < s.shape[0] else 0*np.asarray(r, dtype=float))

def sigmaX(n, r, Dorder):
    '''σ_n and its radial derivatives from its Chebyshev fit on kore's radial grid'''
    return _eval(_fit(('sigma', n), lambda rr, rpower: sigma(n, rr, rpower)), r, Dorder)

def _ptw(r):
    '''the p0 of the tw, tp profiles; 1 if ThermaD = 0 with the thermal wind (utils.adiab_tw: the heat rows divided by p0)'''
    import utils as ut
    import radial_profiles as rap
    return np.ones_like(r) if ut.adiab_tw else rap.prf.pressure(r, 0)

def profile_functions():
    '''{code: f(r, Dorder)} of every differential-rotation profile code, for radial_profiles.burrito'''
    import utils as ut
    out = {}
    fit = lambda key, g: (lambda r, Dorder: _eval(_fit(key, g), r, Dorder))
    for n in degrees():
        c = codes(n)
        out[c['f']] = lambda r, Dorder, n=n: fX(n, r, Dorder)
        out[c['d']] = fit(('d', n), lambda r, rpower, n=n: (r**rpower)*r*fX(n, r, 1))
        out[c['s']] = fit(('s', n), lambda r, rpower, n=n: (r**rpower)*r**2*fX(n, r, 2))
    for n in tw_degrees():
        c = codes(n)
        out[c['tw']] = fit(('tw', n), lambda r, rpower, n=n: (r**rpower)*_ptw(r)*sigmaX(n, r, 0))
        out[c['tp']] = fit(('tp', n), lambda r, rpower, n=n: (r**rpower)*_ptw(r)*sigmaX(n, r, 1))
    return out


# ======================================================================================================================
# 3. Angular couplings
# ======================================================================================================================

def _primitives(n, m, lw):
    '''
    The 12 primitive matrices C[k][shape] of the force for ΔΩ = f(r) P_n(x) (§3.3): the projection k = 'q', 's', 't' of
    W per unit G and per unit radial shape ('q': f L P/r, 'a': f a, 'T': f T, 'dq': (r f') L P/r), on the degrees 0..lw,
    sparse, row = target, column = source:
        W_sh = P_n (i m u_sh + 2 ẑ×u_sh) + ẑ×[-P_n'(x) sinθ u_θ,sh r̂],   W_dq = ẑ×[P_n(x) Y r̂]
    '''
    A = angular.Algebra.get(m, lw)
    Pn, dPn = A.legendre(n)
    I = A.I
    shapes_ = {'q': A.field(q=I), 'a': A.field(s=I), 'T': A.field(t=I)}
    C = {'q': {}, 's': {}, 't': {}}
    for sh, u in shapes_.items():
        W = A.mult(1j*m*u + 2*A.zcross(u), Pn, dPn)        # P_n (i m u + 2 ẑ×u)
        if sh != 'q':                                       # sinθ ∂θΔΩ u_θ e_φ = ẑ×r̂ (-P_n'(x) sinθ u_θ), f_n factor
            W = W + A.zcross_r(-(dPn @ A.sin_theta(u)))
        for k in 'qst':
            C[k][sh] = W[k].tocsr()
    W = A.zcross_r(Pn)                                      # sinθ r ∂ᵣΔΩ u_r e_φ: shape 'dq' (r f_n' in place of f_n)
    for k in 'qst':
        C[k]['dq'] = W[k].tocsr()
    return C, A

_mom, _mom_band = {}, {}
def momentum_coefficients(n, m):
    '''the coefficients of the radial operators for f_n (§3.4, the naming of tests/diff_rot/diff_rot_coefficients.py), as
    sparse matrices [target l', source l] on the degrees up to lmax + n + 6, cached; also stored by diagonal'''
    key = (n, m)
    if key in _mom:
        return _mom[key]
    lw = par.lmax + n + 6
    C, A = _primitives(n, m, lw)
    Ls = lambda M: M @ A.L     # times L of the source degree
    Cq, Cs, Ct = C['q'], C['s'], C['t']
    K = {'f_2C_D0P': Ls(Cq['q']) + Cq['a'], 'f_2C_D1P': Cq['a'] - (Ls(Cs['q']) + 2*Cs['a']), 'f_2C_lho1_D0P': Cq['a'] - Cs['a'],
         'f_2C_lho2_D0P': -Cs['a'], 'f_2C_lho1_D1P': -Cs['a'], 'f_2C_D2P': -Cs['a'],
         'df_2C_D0P': Ls(Cq['dq']) - (Ls(Cs['q']) + Cs['a'] + Ls(Cs['dq'])), 'df_2C_D1P': -(Cs['a'] + Ls(Cs['dq'])),
         'df_2C_lho1_D0P': -Cs['a'], 'd2f_2C_D0P': -Ls(Cs['dq']),
         'f_2C_D0T': Cq['T'] - Cs['T'], 'f_2C_D1T': -Cs['T'], 'df_2C_D0T': -Cs['T'],
         'f_1C_D0P': Ls(Ct['q']) + Ct['a'], 'f_1C_D1P': Ct['a'], 'f_1C_lho1_D0P': Ct['a'], 'df_1C_D0P': Ls(Ct['dq']),
         'f_1C_D0T': Ct['T']}
    K = {k: (A.L @ v).tocsr() for k, v in K.items()}   # times L' of the target: the normalisation of the closed forms
    _mom[key] = K
    _mom_band[key] = {k: angular.band(v, n + 3) for k, v in K.items()}
    return K

def _lookup(bands, l, m, offdiag):
    '''one coefficient from the diagonals of a coupling matrix (angular.band): target degree l, source l + offdiag; 0 outside
    the matrix, below |m| or beyond the band'''
    B = (bands.shape[0] - 1)//2
    ls = l + offdiag
    if ls < m or ls < 0 or ls >= bands.shape[1] or l >= bands.shape[1] or abs(offdiag) > B:
        return 0.0
    return bands[offdiag + B, l]

def momentum_coef(name, n, l, m, offdiag):
    '''coefficient `name` for f_n, target degree l, source degree l + offdiag'''
    momentum_coefficients(n, m)
    return _lookup(_mom_band[(n, m)][name], l, m, offdiag)

_heat = {}
def heat_coefficients(m):
    '''P_n(X), P_n'(X) and P_n'(X) D ('Pn', 'dPn', 'dPnD') for every even n of ΔΩ and S_tw, stored by diagonal, on the
    degrees up to lmax + max(8, n + 4) (products of truncated X are wrong in their last few degrees)'''
    if m in _heat:
        return _heat[m]
    nhi = max([0] + degrees() + tw_degrees())
    A = angular.Algebra.get(m, par.lmax + max(8, nhi + 4))
    out = {}
    for n in range(0, nhi + 1, 2):
        Pn, dPn = A.legendre(n)
        B = n + 1
        out['P%d' % n], out['dP%d' % n], out['dP%dD' % n] = (angular.band(Pn, B), angular.band(dPn, B),
                                                             angular.band(dPn @ A.D, B))
    _heat[m] = out
    return out

def heat_coef(name, l, m, offdiag):
    '''coefficient of the target degree l in (name applied to the source degree l + offdiag)'''
    return _lookup(heat_coefficients(m)[name], l, m, offdiag)


# ======================================================================================================================
# 4. The terms
# ======================================================================================================================

# momentum: the radial labels of the f_n force per (section, component): (coefficient, label template); {a}, {s}, {p}
# are the codes of f_n, r f_n', r² f_n''
MOMENTUM = {
    ('u', 'upol'): [('f_2C_D0P', 'u3{a}0_D0'), ('f_2C_D1P', 'u2{a}0_D1'), ('f_2C_lho1_D0P', 'u2{a}0lho1_D0'),
                    ('f_2C_lho2_D0P', 'u1{a}0lho2_D0'), ('f_2C_lho1_D1P', 'u1{a}0lho1_D1'), ('f_2C_D2P', 'u1{a}0_D2'),
                    ('df_2C_D0P', 'u3{s}0_D0'), ('df_2C_D1P', 'u2{s}0_D1'), ('df_2C_lho1_D0P', 'u2{s}0lho1_D0'),
                    ('d2f_2C_D0P', 'u3{p}0_D0')],
    ('u', 'utor'): [('f_2C_D0T', 'u2{a}0_D0'), ('f_2C_D1T', 'u1{a}0_D1'), ('df_2C_D0T', 'u2{s}0_D0')],
    ('v', 'upol'): [('f_1C_D0P', 'v2{a}0_D0'), ('f_1C_lho1_D0P', 'v1{a}0lho1_D0'), ('f_1C_D1P', 'v1{a}0_D1'), ('df_1C_D0P', 'v2{s}0_D0')],
    ('v', 'utor'): [('f_1C_D0T', 'v1{a}0_D0')],
}

def momentum_labels(n, section, component):
    '''[(coefficient name, label)] of the f_n force terms of a row block'''
    c = codes(n)
    return [(k, t.format(a=c['f'], s=c['d'], p=c['s'])) for k, t in MOMENTUM[(section, component)]]

def max_dl(n, section, component):
    '''largest |Δl| of the f_n couplings: n for P→P and T→T, n + 1 for P↔T'''
    return n if (section == 'u') == (component == 'upol') else n + 1

def _heat_labels():
    '''the section-h terms (§7.2): {'temp': [(n, label)], 'tp': [(n, label)], 'tw': [(n, (h2, h1_D1, h1lho1))], 'tor': [(n, label)]}
    rows r² p0 (eq) (ThermaD > 0), r (eq)/p0 or r² (eq)/p0 (ThermaD = 0 without / with the thermal wind, utils.adiab_tw)'''
    out = {'temp': [], 'tp': [], 'tw': [], 'tor': []}
    for n in degrees():                                   # (a) i m G ΔΩ s
        out['temp'].append((n, ('h0pss0%s0_D0' if par.ThermaD > 0 else 'h0%s0_D0') % codes(n)['f']))
    for n in tw_degrees():                                # u.grad S_tw
        c = codes(n)
        if par.thermal_wind_radial:                       # (b) u_r ∂ᵣS_tw -> L r p0 σn' P
            out['tp'].append((n, 'h1%s0_D0' % c['tp']))
        out['tw'].append((n, ('h2%s0_D0' % c['tw'], 'h1%s0_D1' % c['tw'], 'h1%s0lho1_D0' % c['tw'])))   # (c) p0 σn (P + rP' + r h'P)
        out['tor'].append((n, 'h1%s0_D0' % c['tw']))      # (c') -i m r p0 σn T
    return out

def labels():
    '''[(label, 'P', 'T', 'F' or 'G')]: every radial operator of the differential-rotation terms and the parity class of its
    column (P for the P and s columns, T for T, F and G for the poloidal and toroidal magnetic field), for submatrices.py'''
    out = []
    for n in degrees():
        for (sec, comp) in MOMENTUM:
            out += [(lab, 'P' if comp == 'upol' else 'T') for _, lab in momentum_labels(n, sec, comp)]
    if par.thermal:
        h = _heat_labels()
        out += [(lab, 'P') for _, lab in h['temp'] + h['tp']]
        out += [(lab, 'P') for _, labs in h['tw'] for lab in labs]
        out += [(lab, 'T') for _, lab in h['tor']]
    if par.magnetic:   # NEW (2026-10-30): the induction by the background flow, columns F (poloidal b) and G (toroidal b)
        for n in degrees():
            for (sec, comp) in INDUCTION:
                out += [(lab, 'F' if comp == 'bpol' else 'G') for _, lab in induction_labels(n, sec, comp)]
    return out

def dl_range(section, component):
    '''the Δl = l_source - l_target of the blocks assemble.py places (rows: section u, v, h; columns: upol, utor, temp)'''
    if section in ('u', 'v'):
        nd = n_max()
        same = (section == 'u') == (component == 'upol')
        return range(-nd, nd + 1, 2) if same else range(-nd - 1, nd + 2, 2)
    if component == 'temp':
        return range(-n_max(), n_max() + 1, 2)
    nw = tw_nmax()
    if not (par.thermal_wind and nw):
        return range(0)
    return range(-nw, nw + 1, 2) if component == 'upol' else range(1 - nw, nw, 2)

def momentum_block(l, section, component, offdiag, M):
    '''
    The block of the force W for the row degree l (section 'u' or 'v'), the column 'upol' (P) or 'utor' (T) and the
    source degree l + offdiag: Gaspard Σ_n Σ coefficient × radial operator / L'^3, the operators M[label] being the
    *.mtx loaded by operators.py. Returns [matrix, offd] (offd: the column block offset in units of N1, utils.dr_offd),
    or [0, 0] if no degree couples this Δl.
    '''
    import utils as ut
    m = par.m
    out = 0
    for n in degrees():
        if abs(offdiag) <= max_dl(n, section, component) and (offdiag + n) % 2 == int((section == 'u') != (component == 'upol')):
            for c, lab in momentum_labels(n, section, component):
                out = out + momentum_coef(c, n, l, m, offdiag)*M[lab]
    if np.isscalar(out):
        return [0, 0]
    offd = ut.dr_offd(l, l + offdiag, section, 'P' if component == 'upol' else 'T')
    return [par.Gaspard*out/(l*(l + 1.))**3, offd]

def heat_block(l, component, offdiag, M):
    '''
    The block of the heat-equation terms for the row degree l, the column 'temp' (s), 'upol' (P) or 'utor' (T) and the
    source degree l + offdiag (§7.2): (a) i m G Σ_n P_n(X) [rʰ p0 f_n] s; (b) L_s Σ_n P_n(X) [r p0 σ_n'] P;
    (c) Σ_n P_n'(X) D [p0 σ_n (P + r P' + r h' P)]; (c') -i m Σ_n P_n'(X) [r p0 σ_n] T. Divided by L^3 and with the sign of
    the A-matrix right-hand side (the heat rows of A hold -(left-hand-side terms)). Returns [matrix, offd] or [0, 0].
    '''
    import utils as ut
    m = par.m
    L = l*(l + 1.)
    lp, lt, _ = ut.ell(par.m, par.lmax, par.symm)
    ls = l + offdiag
    h = _heat_labels()
    out, kind = 0, 'P'
    if component == 'temp' and offdiag % 2 == 0:
        for n, lab in h['temp']:
            if abs(offdiag) <= n:
                out = out + 1j*m*heat_coef('P%d' % n, l, m, offdiag)*M[lab]
        out = par.Gaspard*out
    elif component == 'upol' and par.thermal_wind and offdiag % 2 == 0 and abs(offdiag) <= tw_nmax():
        Ls = ls*(ls + 1.)   # L of the source degree, from u_r = L P/r
        for n, lab in h['tp']:
            out = out + Ls*heat_coef('P%d' % n, l, m, offdiag)*M[lab]
        for n, (l2, l1d, l1h) in h['tw']:
            out = out + heat_coef('dP%dD' % n, l, m, offdiag)*(M[l2] + M[l1d] + M[l1h])
    elif component == 'utor' and par.thermal_wind and offdiag % 2 == 1 and abs(offdiag) < tw_nmax():
        kind = 'T'
        for n, lab in h['tor']:
            out = out - 1j*m*heat_coef('dP%d' % n, l, m, offdiag)*M[lab]
    if np.isscalar(out):
        return [0, 0]
    offd = int(np.where((lt if kind == 'T' else lp) == ls)[0][0] - np.where(lp == l)[0][0])
    return [-out/L**3, offd]



# ======================================================================================================================
# 5. NEW (2026-10-30): the induction by the background flow, ∇×(U0×b), in the induction equation (magnetic = 1)
# ======================================================================================================================
# WHY. With a magnetic field, the differential rotation also shears the perturbation b: the induction equation gains
# ∇×(U0×b), U0 = G ΔΩ r sinθ e_φ (G = Gaspard). It is the Ω-effect (a poloidal b sheared into a toroidal one) and, for a
# rigid ΔΩ = c, the Doppler shift -i m G c of the field. With ∇⋅U0 = ∇⋅b = 0 and U0 = G ΔΩ ẑ×𝐫,
#     ∇×(U0×b) = (b⋅∇)U0 - (U0⋅∇)b = G [ -i m ΔΩ b + sinθ (r ∂ᵣΔΩ b_r + ∂θΔΩ b_θ) e_φ ]
# ((U0⋅∇)b = G ΔΩ (i m b + ẑ×b) and (b⋅∇)U0 = G ΔΩ ẑ×b + G (b⋅∇ΔΩ) ẑ×𝐫: the ẑ×b terms cancel). It has the same angular
# structure as the force W of section 3 without its 2ẑ×u term, so it is built with the same algebra, for any degree n of
# ΔΩ (docs/derivations.md, part B; docs/magnetic.md). b = ∇×∇×(F Y 𝐫) + ∇×(G Y 𝐫) has the shapes q = L F/r, s = F' + F/r
# (no (ln ρ0)': b carries no ρ0), t = G.
# Rows (as derive.py: kore's induction rows are -r³/L'³ times r̂⋅(.) (f) and r̂⋅∇×(.) (g) of the equation's right-hand side):
#   r̂⋅ of e_φ terms vanishes and -i m ΔΩ b keeps the type, so the f rows see only -i m ΔΩ b_r: the q-component of the
#   primitives, from the shape q; r̂⋅∇× of a field with components (q, s, t) is L' t/r, so the g rows see the t-components.
#   f row, F column:  -Ls(Cq_q) . f r² F                                     label f0{f}0_D0
#   g row, F column:  -L'(Ls(Ct_q) + Ct_a) . f r F,  -L' Ct_a . f r² F',  -L' Ls(Ct_dq) . (r f') r F
#                                                                            labels g1{f}0_D0, g0{f}0_D1, g1{d}0_D0
#   g row, G column:  -L' Ct_T . f r² G                                      label g0{f}0_D0
# (Ls: times L of the source degree; f = diff_rot_ampl a_n f_n). The f row from G and the components that would need
# negative powers of r (Cq_a, Cq_T, Cq_dq) are zero: the e_φ terms have no radial component.
# Selection rules: F→f and G→g keep the type, Δl = 0, ±2, ..., ±n; F→g changes it, Δl = ±1, ±3, ..., ±(n + 1).

def _induction_primitives(n, m, lw):
    '''
    The primitive matrices C[k][shape] of ∇×(U0×b) per unit G for ΔΩ = f(r) P_n(x) (k = 'q', 's', 't'; shapes 'q': f L F/r,
    'a': f (F' + F/r), 'T': f G, 'dq': (r f') L F/r), on the degrees 0..lw, sparse, row = target, column = source:
        W_sh = -i m P_n b_sh + ẑ×[-P_n'(x) sinθ b_θ,sh r̂],   W_dq = ẑ×[P_n(x) Y r̂]
    as _primitives, whose force has P_n (i m u + 2ẑ×u) in place of -i m P_n b.
    '''
    A = angular.Algebra.get(m, lw)
    Pn, dPn = A.legendre(n)
    I = A.I
    shapes_ = {'q': A.field(q=I), 'a': A.field(s=I), 'T': A.field(t=I)}
    C = {'q': {}, 's': {}, 't': {}}
    for sh, b in shapes_.items():
        W = A.mult(b, Pn, dPn)*(-1j*m)                      # -i m ΔΩ b
        if sh != 'q':                                       # sinθ ∂θΔΩ b_θ e_φ = ẑ×r̂ (-P_n'(x) sinθ b_θ), f_n factor
            W = W + A.zcross_r(-(dPn @ A.sin_theta(b)))
        for k in 'qst':
            C[k][sh] = W[k].tocsr()
    W = A.zcross_r(Pn)                                      # sinθ r ∂ᵣΔΩ b_r e_φ: shape 'dq' (r f_n' in place of f_n)
    for k in 'qst':
        C[k]['dq'] = W[k].tocsr()
    return C, A

_ind, _ind_band = {}, {}
def induction_coefficients(n, m):
    '''the coefficients of the radial operators of ∇×(U0×b) for f_n (the table above), as sparse matrices [target l',
    source l] on the degrees up to lmax + n + 6, cached; also stored by diagonal'''
    key = (n, m)
    if key in _ind:
        return _ind[key]
    lw = par.lmax + n + 6
    C, A = _induction_primitives(n, m, lw)
    Ls = lambda M: M @ A.L     # times L of the source degree
    Lt = lambda M: A.L @ M     # times L' of the target degree
    Cq, Ct = C['q'], C['t']
    K = {'fF_D0': -Ls(Cq['q']),
         'gF_D0': -Lt(Ls(Ct['q']) + Ct['a']), 'gF_D1': -Lt(Ct['a']), 'gFd_D0': -Lt(Ls(Ct['dq'])),
         'gG_D0': -Lt(Ct['T'])}
    K = {k: v.tocsr() for k, v in K.items()}
    _ind[key] = K
    _ind_band[key] = {k: angular.band(v, n + 3) for k, v in K.items()}
    return K

def induction_coef(name, n, l, m, offdiag):
    '''coefficient `name` of ∇×(U0×b) for f_n, target degree l, source degree l + offdiag'''
    induction_coefficients(n, m)
    return _lookup(_ind_band[(n, m)][name], l, m, offdiag)

# the radial labels of the induction by f_n per (section, column): (coefficient, label template); {a}, {s} are the codes
# of f_n and r f_n'. The rows of section f are r^(2 - X), those of g as well (utils.decode_label).
INDUCTION = {
    ('f', 'bpol'): [('fF_D0', 'f0{a}0_D0')],
    ('g', 'bpol'): [('gF_D0', 'g1{a}0_D0'), ('gF_D1', 'g0{a}0_D1'), ('gFd_D0', 'g1{s}0_D0')],
    ('g', 'btor'): [('gG_D0', 'g0{a}0_D0')],
}

def induction_labels(n, section, component):
    '''[(coefficient name, label)] of the f_n induction terms of a row block (section 'f' or 'g', column 'bpol' or 'btor')'''
    c = codes(n)
    return [(k, t.format(a=c['f'], s=c['d'])) for k, t in INDUCTION.get((section, component), [])]

def induction_dl_range(section, component):
    '''the Δl of the induction blocks assemble.py places: F→f and G→g even, |Δl| <= n_max; F→g odd, |Δl| <= n_max + 1;
    G→f none'''
    nd = n_max()
    if (section, component) in (('f', 'bpol'), ('g', 'btor')):
        return range(-nd, nd + 1, 2)
    if (section, component) == ('g', 'bpol'):
        return range(-nd - 1, nd + 2, 2)
    return range(0)

def induction_block(l, section, component, offdiag, M):
    '''
    The block of ∇×(U0×b) for the row degree l (section 'f' or 'g'), the column 'bpol' (F) or 'btor' (G) and the source degree
    l + offdiag: Gaspard Σ_n Σ coefficient × radial operator / L'^3 (the sign of kore's A: minus the right-hand side), with
    the column block offset in units of N1 (the l's of F and G are those of utils.ell with the field's symmetry ut.bsymm).
    Returns [matrix, offd], or [0, 0] if no degree couples this Δl.
    '''
    import utils as ut
    m = par.m
    out = 0
    for n in degrees():
        for c, lab in induction_labels(n, section, component):
            if abs(offdiag) <= n + 1:
                out = out + induction_coef(c, n, l, m, offdiag)*M[lab]
    if np.isscalar(out):
        return [0, 0]
    lF, lG = ut.ell(par.m, par.lmax, ut.bsymm)[:2]
    rows = lF if section == 'f' else lG
    cols = lF if component == 'bpol' else lG
    offd = int(np.where(cols == l + offdiag)[0][0] - np.where(rows == l)[0][0])
    return [par.Gaspard*out/(l*(l + 1.))**3, offd]

def rigid_ricb(tol=1e-10):
    '''
    (c, ok): the rotation ΔΩ(ricb, θ) of the inner boundary, c = diff_rot_ampl a_0 f_0(ricb), and whether it is rigid, i.e.
    f_n(ricb) = 0 for every n > 0 (to tol relative to the largest |f_n|). A conducting inner core co-rotates with c: its field
    then gains the Doppler term -i m G c b (assemble.py), and nothing else changes (the matching conditions involve only
    b_r in U0×b, which is continuous).
    '''
    ri = np.array([par.ricb])
    vals = {n: float(raw(n, ri)[0]) for n in degrees()}
    big = max([1e-300] + [abs(raw(n, np.linspace(max(par.ricb, 1e-3), 1, 50))).max() for n in degrees()])
    c = vals.get(0, 0.0)
    ok = all(abs(v) <= tol*big for n, v in vals.items() if n > 0)
    return c, ok
