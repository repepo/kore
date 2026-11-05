'''
NEW (2026-11-05): the centrifugal deformation of the body (centrifugal = 1), at first order in its flattening: everything
it adds to kore, in one place (docs/background.md, Centrifugal deformation; docs/centrifugal-equations.md).

WHY THIS WAY. A rotating star or planet is flattened: its level surfaces, on which the effective potential (gravity plus
centrifugal) is constant, are spheroids, and the background is constant on them, not on spheres. Kore writes the mass flux
as ∇×∇×(ρ0 P r) + ρ0 ∇×(T r), which conserves mass only if ρ0 depends on one radius; so instead of adding a θ-dependent part
to every background profile, the problem is solved in a reference space whose spheres r = const are the level surfaces. A
point α of the reference space is at X = α + ξ(α) in the body, with
    ξ = ε(r) ∇(r² P2(cos θ)/2) = ε(r) r [P2 r̂ + ½ (dP2/dθ) θ̂],   ε = -(2/3) f,
f(r) the flattening of the level surface of mean radius r (oblate: f > 0); the level surface r is r(1 + ε P2) at first
order, and the θ part of ξ keeps the map smooth at the centre (ε is even in r). Every background profile, the walls (level
surfaces) and kore's labels stay as they are. P and T describe the mass flux of the reference space, w = J F⁻¹ u (F = I + ∇ξ,
J = det F), so that ∇⋅(ρ0 w) = 0 is exact mass conservation in the body, and the impenetrable and no-slip rows are unchanged.
The momentum equation, written with covariant components (Fᵀ times it), keeps a pure gradient of the reduced pressure,
which kore's curl rows remove, and changes in two terms; the heat equation, times J, only in its time derivative:
    ∂ₜu  -> M ∂ₜw,     M = FᵀF/J              (the B matrix, momentum rows)
    2ẑ×u -> 2 z̃×w,     z̃ = F⁻¹ẑ                 (the A matrix, Coriolis)
    ∂ₜs  -> J ∂ₜs                              (the B matrix, heat rows)
The buoyancy keeps its form with the gravity of the effective potential, g - (2/3) Ω² r (g_monopole, radial_profiles.graviX).
At first order, with η = r ε', x = cos θ and w_z = ẑ⋅w:
    (M - I) w   = ε (3 w_z ẑ - w) + η [ (3/2) x (w_z r̂ + w_r ẑ) - w_r r̂ - P2 w ]
    (z̃ - ẑ) × w = -ε ẑ×w - η [ (3/2) x² ẑ×w - (x/2) r̂×w ]
    J - 1       = η P2
Each is a multiplication by polynomials in cos θ of degree 2 at most, with the radial factors ε and η: the same kind of term
as the force of a differential rotation, so they are built with the same recurrence algebra (angular.py) and projected on
kore's rows as in diff_rot.py (diff_rot.momentum_coefficients without its r ∂ᵣΔΩ terms). They couple a degree l to l ± 2 of
the same kind (P to P, T to T, s to s) and to l ± 1, l ± 3 of the other kind (P to T).
LEFT OUT (background.check_centrifugal stops or warns): the metric terms of the viscous force, of the thermal diffusion and
of the stress-free rows (relative size ε times the diffusion: no-slip runs are right in the bulk, with damping rates off by a
fraction of order ε), of the differential rotation and of the magnetic terms.

Sections:
  1. the shape f(r): 'clairaut' (Clairaut's equation from the structure's g, solved in Radau's form), 'uniform' (constant f),
     'user' (a numpy expression); the profile codes ce0, ce1, ce2 (ε, r ε', r (r ε')', even in r); the monopole of g; the map
     of points and flows to the body for plots (to_physical);
  2. the angular couplings: the primitives of (M - I) w and (z̃ - ẑ) × w and the coefficients of the radial operators;
  3. the terms: the labels for submatrices.py, the Δl ranges for assemble.py, the blocks for operators.py.
Only parameters and angular are imported at the top: utils and radial_profiles are imported where needed (radial_profiles
imports this module from burrito), so the algebra can be imported and tested without a run directory.
'''
import sys
import numpy as np
from parameters import par
import angular

SHAPES = ('clairaut', 'uniform', 'user')   # centrifugal_shape as written to params.dat: its index here
CODES = {'e': ('ce0', 'ce1'), 'h': ('ce1', 'ce2')}   # amplitude ε: codes of ε and r ε'; amplitude η = r ε': of η and r η'


# ======================================================================================================================
# 1. The shape
# ======================================================================================================================

def x_scale():
    '''the structure's radius x (in units of the body's radius) per kore radius r: x = r r_out for 'Wu 2005' and the solar
    structures, x = r for 'Boussinesq' (its outer wall is the surface)'''
    return 1.0 if par.structure == 'Boussinesq' else par.r_out

def x_top():
    '''where Clairaut's surface condition is applied: the surface x = 1, or the outer wall for 'model S hydrostatic', whose
    g is defined only up to r_out (the mass above the wall is then neglected, as it is in kore's q there)'''
    return par.r_out if par.structure == 'model S hydrostatic' else 1.0

def radau(g, xt, n=256):
    '''
    Clairaut's equation for a body whose gravity is g(x) (any normalisation, odd in x), up to its surface xt, in Radau's form:
        x η' = 6 - 6 ρ*(η + 1) - η² + η,   η = x f'/f,   ρ* = ρ/ρ̄ = (x g' + 2g)/(3g),   η(0) = 0
    (ρ̄ the mean density inside x; ρ* from g alone, by Poisson's equation, so the structure's g must be its self-gravity),
    integrated from the centre (series η = -(6/7)(ρ* - 1) near it) with ln(f/f_c)' = η/x alongside. ρ* = 1 + x h'/(3h)
    with h = g/x from an even Chebyshev interpolant of degree n. Returns (eta, lnf, rho_s): callables of x in [0, xt] (at |x|)
    for η, ln(f/f_c) and ρ*. The surface condition x f' + 2f = (5/2) q at xt then gives f_c (shape_functions).
    A homogeneous body (g ∝ x) has η = 0 (the Maclaurin f = 5q/4); a polytrope of index 1 has f(xt)/q = 7.5/π².
    '''
    from scipy.integrate import solve_ivp
    C = np.polynomial.chebyshev.Chebyshev
    h = C.interpolate(lambda y: np.asarray(g(np.abs(y)), dtype=float)/np.abs(y), 2*n - 1, domain=[-xt, xt])   # g/x, even
    # (an odd degree: an even number of Chebyshev points, so x = 0 is not one of them)
    hd = h.deriv()
    rho_s = lambda y: 1 + np.abs(y)*hd(np.abs(y))/(3*h(np.abs(y)))
    def rhs(y, v):
        e = v[0]
        return [(6 - 6*rho_s(y)*(e + 1) - e*e + e)/y, e/y]
    x0 = 1e-4*xt
    e0 = -(6/7)*(rho_s(x0) - 1)
    sol = solve_ivp(rhs, (x0, xt), [e0, e0/2], method='DOP853', rtol=1e-12, atol=1e-14, dense_output=True)
    if not sol.success:
        sys.exit("centrifugal_shape 'clairaut': the integration of Clairaut's equation failed (%s)" % sol.message)
    def at(k):
        def f(y):
            y = np.abs(np.asarray(y, dtype=float))
            out = np.empty(y.shape)
            lo = y < x0
            out[~lo] = sol.sol(np.clip(y[~lo], x0, xt))[k]
            out[lo] = e0*(y[lo]/x0)**2/(2 if k else 1)          # the series near the centre: η = c x², ln(f/f_c) = c x²/2
            return out
        return f
    return at(0), at(1), rho_s

_shape = {}
def shape_functions():
    '''
    {'f': f, 'e': ε, 'h': r ε', 'k': r (r ε')'}: callables of the kore radius r (at |r|), and 'f_top' (the flattening at
    the top x_top of the structure, nan if not 'clairaut') and 'eta_top' (η there), for par.centrifugal_shape:
      'clairaut'  f from radau() with the structure's g; f_top = 5 q_top/(2 (η_top + 2)), q_top = centrifugal_q x_top³ (the
                  surface condition; x_top³ because q is defined at the surface, with the mass inside x_top); with
                  centrifugal_f set, the same shape scaled to centrifugal_f at the outer wall. r ε' and r (r ε')' from η exactly.
      'uniform'   f = centrifugal_f
      'user'      f = centrifugal_profile, a numpy expression of r, x and p = centrifugal_param, as the other 'user'
                  expressions (background._expression: x = r r_out with a solar structure, r otherwise), at |r|; r ε' and
                  r (r ε')' from the derivatives of its Chebyshev fit (profile_functions)
    '''
    if _shape:
        return _shape
    t = par.centrifugal_shape
    one = lambda r: np.ones(np.shape(r))
    if t == 'clairaut':
        import radial_profiles as rap
        xs, xt = x_scale(), x_top()
        eta, lnf, rho_s = radau(lambda x: rap.prf.gravity(x/xs, 0), xt)
        eta_t = float(eta(np.array([xt]))[0])
        if par.centrifugal_f is None:
            f_ref, x_ref = 2.5*par.centrifugal_q*xt**3/(eta_t + 2), xt
        else:
            f_ref, x_ref = par.centrifugal_f, xs
        lnf_ref = float(lnf(np.array([x_ref]))[0])
        f = lambda r: f_ref*np.exp(lnf(np.abs(r)*xs) - lnf_ref)
        e = lambda r: -(2/3)*f(r)
        _shape.update({'f': f, 'e': e,
                       'h': lambda r: e(r)*eta(np.abs(r)*xs),
                       'k': lambda r: e(r)*(6 - 6*rho_s(np.abs(r)*xs)*(eta(np.abs(r)*xs) + 1) + eta(np.abs(r)*xs)),
                       'f_top': f_ref*np.exp(float(lnf(np.array([xt]))[0]) - lnf_ref), 'eta_top': eta_t})
    elif t == 'uniform':
        f0 = par.centrifugal_f
        _shape.update({'f': lambda r: f0*one(r), 'e': lambda r: -(2/3)*f0*one(r), 'h': lambda r: 0*one(r),
                       'k': lambda r: 0*one(r), 'f_top': np.nan, 'eta_top': np.nan})
    elif t == 'user':
        import background as bg
        fu = bg._expression(par.centrifugal_profile, par.centrifugal_param, 'centrifugal_profile')
        f = lambda r: fu(np.abs(np.asarray(r, dtype=float)))
        _shape.update({'f': f, 'e': lambda r: -(2/3)*f(r), 'h': None, 'k': None, 'f_top': np.nan, 'eta_top': np.nan})
    else:
        sys.exit('centrifugal_shape %r: unknown, the shapes are %s' % (t, ', '.join(SHAPES)))
    return _shape

def flattening(r):
    '''f(r), the flattening of the level surface of mean radius r (kore units)'''
    return shape_functions()['f'](np.asarray(r, dtype=float))

# Chebyshev fits on kore's radial grid (as diff_rot._fit), even in r when ricb = 0 (the shapes are evaluated at |r|)
_fits = {}
def _fit(key, func):
    '''Chebyshev coefficients of func(r, rpower) on kore's radial domain, cached by key'''
    import utils as ut
    if key not in _fits:
        _fits[key] = ut.chebco_f(func, par.N, par.ricb, ut.rcmb, 1e-14, 0)
    return _fits[key]

def _eval(c, r, Dorder):
    '''the Chebyshev series c (from _fit) or its Dorder-th r-derivative at the radii r'''
    import utils as ut
    return ut.funcheb(c, np.asarray(r, dtype=float), par.ricb, ut.rcmb, Dorder)[:, -1]

def profile_functions():
    '''
    {code: f(r, Dorder)} of the profile codes, for radial_profiles.burrito: ce0 = ε, ce1 = r ε', ce2 = r (r ε')' (each even
    in r), from Chebyshev fits of the exact shapes ('clairaut', 'uniform') or of the derivatives of the fit of ε ('user')
    '''
    S = shape_functions()
    pw = lambda r, rpower: np.asarray(r, dtype=float)**rpower
    out = {'ce0': lambda r, Dorder: _eval(_fit('ce0', lambda rr, rpower: pw(rr, rpower)*S['e'](rr)), r, Dorder)}
    if S['h'] is not None:
        out['ce1'] = lambda r, Dorder: _eval(_fit('ce1', lambda rr, rpower: pw(rr, rpower)*S['h'](rr)), r, Dorder)
        out['ce2'] = lambda r, Dorder: _eval(_fit('ce2', lambda rr, rpower: pw(rr, rpower)*S['k'](rr)), r, Dorder)
    else:
        out['ce1'] = lambda r, Dorder: _eval(_fit('ce1', lambda rr, rpower: pw(rr, rpower)*rr*out['ce0'](rr, 1)), r, Dorder)
        out['ce2'] = lambda r, Dorder: _eval(_fit('ce2', lambda rr, rpower: pw(rr, rpower)*rr*out['ce1'](rr, 1)), r, Dorder)
    return out

def g_monopole(r):
    '''
    The centrifugal monopole of the effective gravity, in the units of the structure's g, at the kore radii r (odd in r):
    -(2/3) Ω² x R = -(2/3) q_top g(x_top) x/x_top, x = r x_scale(), q_top = centrifugal_q x_top³ (0 when centrifugal_q = 0).
    On the level surfaces the effective potential is Φ0(r) - Ω² r²/3 at first order (its P2 parts cancel there), so the
    buoyancy uses dΦ_eff/dr = g - (2/3) Ω² r; radial_profiles.graviX adds it to g.
    '''
    import radial_profiles as rap
    r = np.asarray(r, dtype=float)
    q = par.centrifugal_q
    if q == 0:
        return 0*r
    xs, xt = x_scale(), x_top()
    gt = float(np.asarray(rap.prf.gravity(np.array([xt/xs]), 0))[0])
    return -(2/3)*q*xt**3*gt*(r*xs)/xt

def to_physical(r, theta, wr=None, wt=None, wp=None):
    '''
    For plots (tools/plot_field.py): where the reference point (r, θ) is in the body and, given the components (w_r, w_θ,
    w_φ) of a field w of the reference space there, the velocity u = F w/J of the body in the spherical basis of that point,
    at first order in f (ε, η = r ε' the profile codes ce0, ce1, at |r|):
        s = r sinθ (1 - ε/2),  z = r cosθ (1 + ε)                         X = α + ξ, ξ = ε (-x/2, -y/2, z) on each sphere
        F w/J = w + ε ((3/2) w_z ẑ - w/2) + η w_r ((3/2) cosθ ẑ - r̂/2) - η P2 w
    and the basis turned by the colatitude change δθ = -(3/2) ε sinθ cosθ of the point. Returns (s, z), or
    (s, z, u_r, u_θ, u_φ) with the components (arrays broadcast together). A scalar (s, buoyancy) needs (s, z) only.
    '''
    prof = profile_functions()
    r = np.asarray(r, dtype=float)
    shp = np.broadcast(r, theta).shape
    ra = np.abs(np.broadcast_to(r, shp)).ravel()
    e = prof['ce0'](ra, 0).reshape(shp)
    h = prof['ce1'](ra, 0).reshape(shp)
    c, sn = np.cos(theta), np.sin(theta)
    s, z = r*sn*(1 - e/2), r*c*(1 + e)
    if wr is None:
        return s, z
    P2 = 1.5*c*c - 0.5
    wz = c*wr - sn*wt
    Ur = wr + e*(1.5*wz*c - wr/2)                                   # the η terms cancel in the radial component
    Ut = wt - e*(1.5*wz*sn + wt/2) - h*(1.5*c*sn*wr + P2*wt)
    Up = wp - e*wp/2 - h*P2*wp
    dth = -1.5*e*sn*c
    return s, z, Ur + dth*Ut, Ut - dth*Ur, Up

def description():
    '''the 'user' expression as text for params.dat (a '#' comment), empty otherwise'''
    if par.centrifugal and par.centrifugal_shape == 'user':
        return 'centrifugal user: f = %s ; p = %r' % (par.centrifugal_profile, par.centrifugal_param)
    return ''

def report():
    '''prints the flattening used (submatrices.py): at the walls and, for 'clairaut', at the top of the structure with f/q;
    and a WARNING for each term left spherical (background.check_centrifugal stops on what cannot be done)'''
    S = shape_functions()
    walls = [1.0] + ([par.ricb] if par.ricb > 0 else [])
    msg = "centrifugal_shape %r: flattening %s" % (par.centrifugal_shape, ', '.join(
        '%.6e at r = %g' % (float(flattening(np.array([r]))[0]), r) for r in walls))
    if par.centrifugal_shape == 'clairaut':
        msg += '; at x = %g (top of the structure) f = %.6e, a f\'/f = %.6f' % (x_top(), S['f_top'], S['eta_top'])
        if par.centrifugal_q > 0:
            msg += ', f/q = %.6f' % (S['f_top']/(par.centrifugal_q*x_top()**3))
    if par.centrifugal_q > 0 and par.thermal:
        msg += '; buoyancy with the effective gravity, g(1) %+.3e from the centrifugal monopole' % float(g_monopole(np.array([1.0]))[0])
    print(msg)
    fmax = np.abs(flattening(np.linspace(max(par.ricb, 0.0), 1, 201))).max()
    if fmax > 0.1:
        print('WARNING: centrifugal = 1: flattening up to %.3f, beyond about 0.1 the first order in f loses accuracy' % fmax)
    left = (['the viscous force'] if par.ViscosD > 0 else []) + (['the thermal diffusion'] if par.thermal and par.ThermaD > 0 else [])
    if left:
        print('WARNING: centrifugal = 1: %s of the sphere (the metric terms of the diffusion are not implemented): damping '
              'rates are off by a fraction of order f' % ' and '.join(left))
    if par.diff_rot:
        print('WARNING: centrifugal = 1: the differential-rotation terms are those of the sphere (the terms of order f ΔΩ are '
              'left out)')


# ======================================================================================================================
# 2. Angular couplings
# ======================================================================================================================

def _ops(A):
    '''the operations of the metric on fields of the algebra A: ẑ⋅F (a scalar), c ẑ, c r̂ (fields of a scalar c), r̂ × F'''
    m, X, Li, Dd = A.m, A.X, A.Li, A.Dd
    zdot = lambda F: X @ F.q - A.sin_theta(F)                            # ẑ⋅F = x F_r - sinθ F_θ
    zhat = lambda c: angular.Field(X @ c, Li @ (Dd @ c), Li @ (1j*m*c))  # c ẑ = c x r̂ - c sinθ θ̂
    rhat = lambda c: A.field(q=c)                                        # c r̂
    rcross = lambda F: A.field(s=F.t, t=-1*F.s)                          # r̂ × S_l = -T_l, r̂ × T_l = S_l
    return zdot, zhat, rhat, rcross

def primitives(kind, amp, m, lw):
    '''
    The primitive matrices C[k][shape] of the metric terms per unit radial factor, on the degrees 0..lw (k = 'q', 's', 't'
    the components of the result, shapes 'q': w_r, 'a': the consoidal coefficient of w, 'T': its toroidal one, as in
    diff_rot._primitives; row = target degree, column = source degree, sparse):
        kind 'inertia', amp 'e':   3 w_z ẑ - w                                 (factor ε)
        kind 'inertia', amp 'h':   (3/2) x (w_z r̂ + w_r ẑ) - w_r r̂ - P2 w    (factor η = r ε')
        kind 'coriolis', amp 'e':  -ẑ×w                                         (factor ε)
        kind 'coriolis', amp 'h':  -(3/2) x² ẑ×w + (x/2) r̂×w                   (factor η)
    Returns (C, A), A the algebra.
    '''
    A = angular.Algebra.get(m, lw)
    P2, dP2 = A.legendre(2)
    X, I = A.X, A.I
    zdot, zhat, rhat, rcross = _ops(A)
    C = {'q': {}, 's': {}, 't': {}}
    for sh, w in {'q': A.field(q=I), 'a': A.field(s=I), 'T': A.field(t=I)}.items():
        if (kind, amp) == ('inertia', 'e'):
            W = 3*zhat(zdot(w)) - w
        elif (kind, amp) == ('inertia', 'h'):
            W = rhat(1.5*(X @ zdot(w))) + zhat(1.5*(X @ w.q)) - rhat(w.q) - A.mult(w, P2, dP2)
        elif (kind, amp) == ('coriolis', 'e'):
            W = -1*A.zcross(w)
        else:
            W = -1.5*A.mult(A.zcross(w), X @ X, 2*X) + 0.5*A.mult(rcross(w), X, I)
        for k in 'qst':
            C[k][sh] = W[k].tocsr()
    return C, A

def lw_of():
    '''the degrees 0..lw on which the couplings are built: kore's highest degree (lmax + 1) plus a margin for the products'''
    return par.lmax + 8

_K, _Kb = {}, {}
def momentum_coefficients(kind, amp, m):
    '''
    The coefficients of the radial operators (the names of diff_rot.momentum_coefficients) for one metric term, as sparse
    matrices [target l', source l] times L' of the target, cached; also stored by diagonal. A field W with components
    (q, s, t) per unit radial factor f enters the rows u as r³ (q - (r s)')/L'² and v as r² t/L'², here with f = ε or η
    and its r-derivative r f' (codes CODES[amp]); unlike the force of a differential rotation there is no r ∂ᵣ term.
    '''
    key = (kind, amp, m)
    if key in _K:
        return _K[key]
    C, A = primitives(kind, amp, m, lw_of())
    Ls = lambda M: M @ A.L     # times L of the source degree
    Cq, Cs, Ct = C['q'], C['s'], C['t']
    K = {'f_2C_D0P': Ls(Cq['q']) + Cq['a'], 'f_2C_D1P': Cq['a'] - (Ls(Cs['q']) + 2*Cs['a']), 'f_2C_lho1_D0P': Cq['a'] - Cs['a'],
         'f_2C_lho2_D0P': -1*Cs['a'], 'f_2C_lho1_D1P': -1*Cs['a'], 'f_2C_D2P': -1*Cs['a'],
         'df_2C_D0P': -1*(Ls(Cs['q']) + Cs['a']), 'df_2C_D1P': -1*Cs['a'], 'df_2C_lho1_D0P': -1*Cs['a'],
         'f_2C_D0T': Cq['T'] - Cs['T'], 'f_2C_D1T': -1*Cs['T'], 'df_2C_D0T': -1*Cs['T'],
         'f_1C_D0P': Ls(Ct['q']) + Ct['a'], 'f_1C_D1P': Ct['a'], 'f_1C_lho1_D0P': Ct['a'],
         'f_1C_D0T': Ct['T']}
    K = {k: (A.L @ v).tocsr() for k, v in K.items()}
    _K[key] = K
    _Kb[key] = {k: angular.band(v, 4) for k, v in K.items()}
    return K

_heat = {}
def heat_coefficients(m):
    '''P2(X) by diagonal (angular.band), for J - 1 = η P2 in the heat rows'''
    if m not in _heat:
        A = angular.Algebra.get(m, lw_of())
        _heat[m] = angular.band(A.legendre(2)[0], 4)
    return _heat[m]

def _lookup(bands, l, m, offdiag):
    '''one coefficient from the diagonals of a coupling matrix (angular.band): target degree l, source l + offdiag; 0 outside'''
    B = (bands.shape[0] - 1)//2
    ls = l + offdiag
    if ls < m or ls < 0 or ls >= bands.shape[1] or l >= bands.shape[1] or abs(offdiag) > B:
        return 0.0
    return bands[offdiag + B, l]


# ======================================================================================================================
# 3. The terms
# ======================================================================================================================

# the radial labels per (section, component): (coefficient, label template); {a} is the code of the radial factor f, {s} that
# of r f' (CODES). As diff_rot.MOMENTUM without its r ∂ᵣΔΩ terms (d2f, df_1C_D0P).
MOMENTUM = {
    ('u', 'upol'): [('f_2C_D0P', 'u3{a}0_D0'), ('f_2C_D1P', 'u2{a}0_D1'), ('f_2C_lho1_D0P', 'u2{a}0lho1_D0'),
                    ('f_2C_lho2_D0P', 'u1{a}0lho2_D0'), ('f_2C_lho1_D1P', 'u1{a}0lho1_D1'), ('f_2C_D2P', 'u1{a}0_D2'),
                    ('df_2C_D0P', 'u3{s}0_D0'), ('df_2C_D1P', 'u2{s}0_D1'), ('df_2C_lho1_D0P', 'u2{s}0lho1_D0')],
    ('u', 'utor'): [('f_2C_D0T', 'u2{a}0_D0'), ('f_2C_D1T', 'u1{a}0_D1'), ('df_2C_D0T', 'u2{s}0_D0')],
    ('v', 'upol'): [('f_1C_D0P', 'v2{a}0_D0'), ('f_1C_lho1_D0P', 'v1{a}0lho1_D0'), ('f_1C_D1P', 'v1{a}0_D1')],
    ('v', 'utor'): [('f_1C_D0T', 'v1{a}0_D0')],
}

def momentum_labels(amp, section, component):
    '''[(coefficient name, label)] of the terms of radial factor amp ('e': ε, 'h': η) in a row block'''
    a, s = CODES[amp]
    return [(k, t.format(a=a, s=s)) for k, t in MOMENTUM[(section, component)]]

def heat_label():
    '''the radial operator of η s in the heat rows: that of operators.entropy times the profile ce1'''
    return ('h0pss0%s0_D0' if par.ThermaD > 0 else 'h0%s0_D0') % CODES['h'][0]

def labels():
    '''[(label, 'P' or 'T')]: every radial operator of the centrifugal terms and the parity class of its column, for
    submatrices.py (each once; inertia and Coriolis share them)'''
    out = []
    for amp in CODES:
        for (sec, comp) in MOMENTUM:
            out += [(lab, 'P' if comp == 'upol' else 'T') for _, lab in momentum_labels(amp, sec, comp)]
    if par.thermal:
        out.append((heat_label(), 'P'))
    seen = set()
    return [x for x in out if not (x[0] in seen or seen.add(x[0]))]

def dl_range(section, component):
    '''the Δl = l_source - l_target of the blocks assemble.py places: 0, ±2 between degrees of the same kind (u rows and P
    columns, v and T, h and s), ±1, ±3 between P and T'''
    same = {('u', 'upol'): True, ('v', 'utor'): True, ('h', 'temp'): True, ('u', 'utor'): False, ('v', 'upol'): False}
    return range(-2, 3, 2) if same[(section, component)] else range(-3, 4, 2)

def momentum_block(kind, l, section, component, offdiag, M):
    '''
    The block of a metric term for the row degree l (section 'u' or 'v'), the column 'upol' (P) or 'utor' (T) and the source
    degree l + offdiag: Σ_amp Σ coefficient × radial operator / L'^3, the operators M[label] being those of operators.py.
    kind 'inertia': the rows of (M - I) w (B gets minus this, as it gets minus operators.inertia); kind 'coriolis': those of
    2 Gaspard (z̃ - ẑ)×w (A gets it, as operators.coriolis). Returns [matrix, offd] (offd: the column block offset in units of
    N1, utils.dr_offd), or [0, 0] if nothing couples this Δl.
    '''
    import utils as ut
    m = par.m
    if offdiag not in dl_range(section, component):
        return [0, 0]
    out = 0
    for amp in CODES:
        momentum_coefficients(kind, amp, m)
        bands = _Kb[(kind, amp, m)]
        for c, lab in momentum_labels(amp, section, component):
            v = _lookup(bands[c], l, m, offdiag)
            if v != 0:
                out = out + v*M[lab]
    if np.isscalar(out):
        return [0, 0]
    offd = ut.dr_offd(l, l + offdiag, section, 'P' if component == 'upol' else 'T')
    fac = 1.0 if kind == 'inertia' else 2*par.Gaspard
    return [fac*out/(l*(l + 1.))**3, offd]

def entropy_block(l, offdiag, M):
    '''
    The block of (J - 1) s = η P2 s in the heat rows (B, the sign of operators.entropy, whose radial operator it multiplies
    by η) for the row degree l and the source degree l + offdiag (0, ±2): [matrix, offd] or [0, 0].
    '''
    import utils as ut
    m = par.m
    if offdiag not in dl_range('h', 'temp'):
        return [0, 0]
    c = _lookup(heat_coefficients(m), l, m, offdiag)
    if c == 0:
        return [0, 0]
    lp = ut.ell(par.m, par.lmax, par.symm)[0]
    offd = int(np.where(lp == l + offdiag)[0][0] - np.where(lp == l)[0][0])
    return [c*M[heat_label()]/(l*(l + 1.))**3, offd]
