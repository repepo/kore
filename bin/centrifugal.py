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
With a field (magnetic = 1, NEW phase 1b, sections 4 and 5) the field is a flux of the reference space too, b̃ = J F⁻¹ b,
and so is the background: kore's formula for B0 gives B̃0 on the level surfaces, and the field of the body is F B̃0/J. Then
∇⋅b̃ = 0 is exact, the Lorentz force takes its covariant form (1/J)[∇×(M b̃) × B̃0 + ∇×(M B̃0) × b̃] and the induction
∂ₜb̃ = ∇×(w × B̃0/J); their first-order metric parts are built with the recurrence algebra of mag_algebra.py. At an
insulating wall the field is matched to the potential field outside on the deformed wall; at a conducting inner core of the
fluid's permeability (mu_i2o = 1) b̃ is continuous, and the row of the tangential electric field gains the metric part of the
diffusion (section 6).
NEW (2026-11-06, phase 2, sections 6 and 7): the dissipative terms take their covariant forms too, the viscous force, the
thermal and the magnetic diffusion (in the fluid and in a conducting inner core) and the stress-free and fixed-flux wall rows;
a differential rotation, given in the reference space as every profile (the body point X(r, θ) turns at ΔΩ(r, θ), so a ΔΩ
of r alone is constant on the level surfaces), adds the metric parts of its force and of its advection of the entropy (its
induction has none).
NEW (2026-11-06, section 8): the exact spheroid, centrifugal_shape 'spheroid'. Every level surface has the flattening f and
the map is the linear stretch X = diag(a, a, c) α (c/a = 1 - f, a² c = 1), whose metric is constant: its terms are kept
exactly, at any f, for a homogeneous fluid without a differential rotation; NEW (2026-11-06, step 2) also with a field: the
Lorentz force, the magnetic diffusion, the insulating walls matched to the potential field in the spheroidal coordinates of
the wall, and a conducting inner core.
LEFT OUT (background.check_centrifugal stops, report() warns): forced problems, the zero radial stress walls (bco, bci = 2),
a conducting inner core of another permeability (mu_i2o ≠ 1), the thermal-wind entropy (thermal_wind = 1, 2), which keeps
the balance of the sphere, and the second order in f (but for the exact spheroid).

Sections:
  1. the shape f(r): 'clairaut' (Clairaut's equation from the structure's g, solved in Radau's form), 'uniform' (constant f),
     'user' (a numpy expression); the profile codes ce0, ce1, ce2 (ε, r ε', r (r ε')', even in r) and ce3 (r (ce2)', for the
     magnetic terms); the monopole of g; the map of points and flows to the body for plots (to_physical);
  2. the angular couplings: the primitives of (M - I) w and (z̃ - ẑ) × w and the coefficients of the radial operators;
  3. the terms: the labels for submatrices.py, the Δl ranges for assemble.py, the blocks for operators.py;
  4. the magnetic terms: the profile codes of ε, η times the profiles of B0, the metric parts of the Lorentz force and of
     the induction (magnetic_build) and their blocks for operators.py;
  5. the insulating walls: the metric parts of the matching rows (wall_coefficients, wall_rows for assemble.py), the
     potential field outside for spin_doctor (wall_potential) and ε, η inside the inner core (amplitudes);
  6. the dissipative terms: the metric parts of the viscous force and of the thermal and magnetic diffusion
     (diffusion_build), of the stress-free and fixed-flux wall rows (wall_dissipative) and, in a conducting inner core, of
     its diffusion and of the electric field's row at the ICB (diffusion_block_ic, icb_rows);
  7. the differential rotation: the metric parts of its force (rotation_force, rotation_build) and of its advection of the
     entropy (rotation_heat_coefficients), and their blocks for operators.py;
  8. the exact spheroid ('spheroid'): the metric parts of the linear stretch, exactly (spheroid_terms, spheroid_build,
     spheroid_walls), their blocks (spheroid_block, through momentum_block and diffusion_block) and the primitives of the
     inertia and the Coriolis force for the budgets (spheroid_primitives); with a field (step 2), the Lorentz force
     (spheroid_magnetic_build), the insulating wall rows (spheroid_lambda, spheroid_field_walls), the potential field for the
     budgets (spheroid_potential) and the ICB row of a conducting inner core (spheroid_icb).
Only parameters and angular are imported at the top: utils and radial_profiles are imported where needed (radial_profiles
imports this module from burrito), so the algebra can be imported and tested without a run directory.
'''
import sys, re
import numpy as np
from parameters import par
import angular

SHAPES = ('clairaut', 'uniform', 'user', 'spheroid')   # centrifugal_shape as written to params.dat: its index here
# (NEW 2026-11-06: 'spheroid', the exact spheroid of section 8)
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
    with h = g/x from an even Chebyshev interpolant of degree 2n - 1 (FIX 2026-11-08: by a DCT, its coefficients below
    1e-15 of the largest dropped). Returns (eta, lnf, rho_s): callables of x in [0, xt] (at |x|) for η, ln(f/f_c) and ρ*.
    The surface condition x f' + 2f = (5/2) q at xt then gives f_c (shape_functions).
    A homogeneous body (g ∝ x) has η = 0 (the Maclaurin f = 5q/4); a polytrope of index 1 has f(xt)/q = 7.5/π².
    '''
    from scipy.integrate import solve_ivp
    import background_field as bf
    h = bf.cheb_interp(lambda y: np.asarray(g(np.abs(y)), dtype=float)/np.abs(y), 2*n - 1, [-xt, xt])   # g/x, even
    # (an odd degree: an even number of Chebyshev points, so x = 0 is not one of them)
    # FIX (2026-11-08, phase 6d): a DCT (cheb_interp, as B0's profiles since phase 6c) instead of numpy's
    # Chebyshev.interpolate, whose round-off (2e-17 j to 5e-17 j of the largest coefficient up to j = 511) h' amplified at
    # x = xt: ρ* was 1.1e-7 off there for every structure (a homogeneous body too, where ρ* = 1), r (r ε')' 2.5e-6 of ε
    # off at the outer wall of 'model S hydrostatic' (x_top = r_out), and its fit ce4 there 43 times its size off at N 450
    # (4.6e-3 now). The cut is at 1e-15, not at the profiles' 1e-13: the g/x of the solar structures have content down to
    # 1e-15 of their largest coefficient (to degree 130 for '8-term fit', about 200 for 'model S hydrostatic'), which a cut
    # at 1e-13 truncates
    h.coef[np.abs(h.coef) < 1e-15*np.abs(h.coef).max()] = 0.
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
      'spheroid'  f = centrifugal_f, kept exactly (section 8; NEW 2026-11-06): these are the profiles of its first order, for
                  the plots and the reports only
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
    elif t in ('uniform', 'spheroid'):   # NEW (2026-11-06): 'spheroid' (section 8) has the profiles of 'uniform', its first order
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

_deformed = []
def deformed():
    '''NEW (2026-11-06, phase 2): whether the body is deformed at all, f ≠ 0 somewhere in 0 <= r <= 1. Without a deformation
    (q = 0, or 'uniform' with centrifugal_f = 0) every metric term vanishes and kore keeps the sphere's bases and rows (the
    C^(3) basis of the sections v and g, the rows r³ of g and gic, rpower_v >= 4 in a viscous full sphere), so that its
    matrices are exactly those of centrifugal = 0. The dissipative terms and those of the differential rotation are left out
    then too (diffusion_kinds, rotation_on, wall_dissipative_rows), being zero.
    FIX (2026-11-06): it is called before the structure is built (background.check_centrifugal, utils), so 'clairaut', whose
    f is its scale (centrifugal_f, or centrifugal_q) times a positive shape from the structure's g, is decided by the scale'''
    if not _deformed:
        t, f = par.centrifugal_shape, par.centrifugal_f
        if t == 'clairaut':
            d = f != 0 if f is not None else par.centrifugal_q != 0
        elif t in ('uniform', 'spheroid'):   # NEW (2026-11-06): 'spheroid'
            d = f is not None and f != 0
        else:
            d = np.any(flattening(np.linspace(0., 1., 41)) != 0)
        _deformed.append(bool(d))
    return _deformed[0]

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
    in r), from Chebyshev fits of the exact shapes ('clairaut', 'uniform') or of the derivatives of the fit of ε ('user');
    NEW (2026-11-05, magnetic terms) ce3 = r (ce2)', from the derivative of the fit of ce2
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
    out['ce3'] = lambda r, Dorder: _eval(_fit('ce3', lambda rr, rpower: pw(rr, rpower)*rr*out['ce2'](rr, 1)), r, Dorder)
    # NEW (2026-11-05, phase 2) ce4 = r (ce3)', for the dissipative terms
    out['ce4'] = lambda r, Dorder: _eval(_fit('ce4', lambda rr, rpower: pw(rr, rpower)*rr*out['ce3'](rr, 1)), r, Dorder)
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
    NEW (2026-11-06): the exact spheroid (section 8), exactly: s = a r sinθ, z = c r cosθ, u = F w (u_s = a w_s, u_z = c w_z,
    u_φ = a w_φ), in the basis of the colatitude atan2(a sinθ, c cosθ) of the point.
    '''
    if par.centrifugal_shape == 'spheroid':
        a, c = spheroid_axes()
        r = np.asarray(r, dtype=float)
        ct, st = np.cos(theta), np.sin(theta)
        s, z = a*r*st, c*r*ct
        if wr is None:
            return s, z
        Us, Uz = a*(st*wr + ct*wt), c*(ct*wr - st*wt)
        th = np.arctan2(a*st, c*ct)                 # the colatitude of the point (r > 0)
        return s, z, np.sin(th)*Us + np.cos(th)*Uz, np.cos(th)*Us - np.sin(th)*Uz, a*wp + 0*Us
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
    if par.centrifugal_shape == 'spheroid':   # NEW (2026-11-06): section 8
        msg += '; kept exactly: the level surface r = 1 has the radii a = %.12f (equator), c = %.12f (pole)' % spheroid_axes()
    print(msg)
    fmax = np.abs(flattening(np.linspace(max(par.ricb, 0.0), 1, 201))).max()
    if fmax > 0.1 and par.centrifugal_shape != 'spheroid':
        print('WARNING: centrifugal = 1: flattening up to %.3f, beyond about 0.1 the first order in f loses accuracy' % fmax)
    # FIX (2026-11-06, phase 2): the viscous force, the diffusion and the differential rotation have their metric terms now
    if par.diff_rot and par.thermal and par.thermal_wind in (1, 2):
        print('WARNING: centrifugal = 1: the thermal-wind entropy S_tw balances ΔΩ in the sphere (the terms of order f ΔΩ of '
              'the balance are left out); its advection and the other terms of ΔΩ are those of the deformed body')
    if par.magnetic and exact():   # NEW (2026-11-06, step 2)
        print('centrifugal = 1: B0 is given on the level surfaces (the reference space) and carried by the map: the field in '
              'the body is diag(a, a, c) times the spherical formula')
    elif par.magnetic:
        print('centrifugal = 1: B0 is given on the level surfaces (the reference space) and carried by the map: the field in '
              'the body differs from the spherical formula at order f')


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

def heat_label(section='h'):
    '''the radial operator of η s in the heat rows: that of operators.entropy times the profile ce1; NEW (2026-11-08) section
    'c', that of η c in the rows of the composition (operators.composition times ce1)'''
    if section == 'c':
        return ('c0dns0%s0_D0' if par.CompoD > 0 else 'c0%s0_D0') % CODES['h'][0]
    return ('h0pss0%s0_D0' if par.ThermaD > 0 else 'h0%s0_D0') % CODES['h'][0]

def labels():
    '''[(label, 'P' or 'T')]: every radial operator of the centrifugal terms and the parity class of its column, for
    submatrices.py (each once; inertia and Coriolis share them)'''
    if exact():   # NEW (2026-11-06): the exact spheroid's (section 8), with J = 1 (no heat label)
        seen = set()
        return [x for x in spheroid_labels(('inertia', 'coriolis')) if not (x[0] in seen or seen.add(x[0]))]
    out = []
    for amp in CODES:
        for (sec, comp) in MOMENTUM:
            out += [(lab, 'P' if comp == 'upol' else 'T') for _, lab in momentum_labels(amp, sec, comp)]
    if par.thermal:
        out.append((heat_label(), 'P'))
    if par.compositional:   # NEW (2026-11-08)
        out.append((heat_label('c'), 'C'))
    seen = set()
    return [x for x in out if not (x[0] in seen or seen.add(x[0]))]

def dl_range(section, component):
    '''the Δl = l_source - l_target of the blocks assemble.py places: 0, ±2 between degrees of the same kind (u rows and P
    columns, v and T, h and s; NEW phase 2: f and F, g and G; NEW 2026-11-08: c and the composition), ±1, ±3 between P and T
    (f and G, g and F)'''
    same = {('u', 'upol'): True, ('v', 'utor'): True, ('h', 'temp'): True, ('u', 'utor'): False, ('v', 'upol'): False,
            ('f', 'bpol'): True, ('g', 'btor'): True, ('f', 'btor'): False, ('g', 'bpol'): False, ('c', 'comp'): True}
    section, component = section.replace('ic', ''), component.replace('_ic', '')   # NEW (2026-11-06): a conducting core's
    if exact():   # NEW (2026-11-06): the exact spheroid (section 8), ±4 and ±3 (its viscous force and wall rows)
        return range(-4, 5, 2) if same[(section, component)] else range(-3, 4, 2)
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
    if exact():   # NEW (2026-11-06): the exact spheroid (section 8)
        return spheroid_block(kind, l, section, component, offdiag, M)
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

def entropy_block(l, offdiag, M, section='h'):
    '''
    The block of (J - 1) s = η P2 s in the heat rows (B, the sign of operators.entropy, whose radial operator it multiplies
    by η) for the row degree l and the source degree l + offdiag (0, ±2): [matrix, offd] or [0, 0]. NEW (2026-11-08): section
    'c', (J - 1) c in the rows of the composition (the sign of operators.composition).
    '''
    import utils as ut
    m = par.m
    if offdiag not in dl_range(section, 'temp' if section == 'h' else 'comp') or exact():   # NEW (2026-11-06): J = 1 in the
        return [0, 0]                                                                          # exact spheroid
    c = _lookup(heat_coefficients(m), l, m, offdiag)
    if c == 0:
        return [0, 0]
    lp = ut.ell(par.m, par.lmax, par.symm)[0]
    offd = int(np.where(lp == l + offdiag)[0][0] - np.where(lp == l)[0][0])
    return [c*M[heat_label(section)]/(l*(l + 1.))**3, offd]


# ======================================================================================================================
# 4. The magnetic terms (NEW 2026-11-05, phase 1b)
# ======================================================================================================================
# The field is a flux of the reference space too, b̃ = J F⁻¹ b, and so is the background: kore's formula for B0 is B̃0, and the
# field of the body is F B̃0/J. So ∇⋅b̃ = 0 is exact, f and g keep their meaning, and the B rows of f, g (∂ₜb̃) do not change.
# With the current j̃ = J F⁻¹ ∇×b = ∇×(M b̃) and Fᵀ(a×c) = (ã×c̃)/J (a = F ã/J, c = F c̃/J):
#     covariant Lorentz force (over ρ0, times Hendrik as now):   (1/J) [∇×(M b̃) × B̃0 + ∇×(M B̃0) × b̃]
#     induction:                                                  ∂ₜb̃ = ∇×(w × B̃0/J) + the magnetic diffusion
# The metric part of the diffusion is in section 6 (FIX 2026-11-06, phase 2; phase 1b left it out). At first order, with
# M - I and J - 1 = η P2 of section 1, the metric parts are (B0, J0 = ∇×B0, b, J = ∇×b all of the reference space)
#     δ𝐆 = -η P2 (J0×b - B0×J) + [∇×((M - I)B0)]×b - B0×[∇×((M - I)b)],      δ𝐄 = -η P2 (w×B0)   (induction ∇×δ𝐄)
# and they enter kore's rows as the force 𝐆 and the field 𝐄 = u×B0 of mag_algebra.py. They are built here with its recurrence
# algebra, the background a sum of vectors c(x) r̂, c(x) sinθ θ̂, c(x) sinθ φ̂ (mag_algebra._background) and the perturbation
# a sum of Fields; a radial factor is a monomial r^p X^(d) [a B_n^(j)] λ_k whose amplitude a is one of AMPS, and the product
# a B_n^(j) is one profile code (product_code: 'E', 'N', 'C', 'Q' then 'p' + degree for h_n, 't' + degree for k_n), the
# derivative digit j of the label applying to B_n only. An amplitude's derivative is the next one over r (ε' = η/r, ...).
AMPS = {'E': ('ce0', 'N'), 'N': ('ce1', 'C'), 'C': ('ce2', 'Q'), 'Q': ('ce3', 'K'),   # amplitude: (its code, that of r a')
        'K': ('ce4', None)}   # NEW (2026-11-05, phase 2): ce4 = r (ce3)', for the dissipative terms (section 6)
# x = cos θ and P2 as Legendre series (FIX 2026-11-08, phase 6e: they meet the background's P_n, now Legendre series too,
# whose values on the matrix X mag_algebra._poly_mats sums over the P_k(X) of angular's recurrence)
_xp = np.polynomial.Legendre([0., 1.])
_P2p = np.polynomial.Legendre([0., 0., 1.])

def product_code(a, B, n):
    '''the profile code of the product of the amplitude a (AMPS) and the background profile B ('h' or 'k') of degree n'''
    import mag_algebra as ma
    if n >= len(ma.DIGITS):
        sys.exit('centrifugal: degree %d of B0 above the largest the profile codes allow (%d)' % (n, len(ma.DIGITS) - 1))
    return a + ('p' if B == 'h' else 't') + ma.DIGITS[n]

def magnetic_profile_functions():
    '''{product code: f(r, Dorder)} for radial_profiles.burrito: a(r) B_n^(Dorder)(r), the amplitude a from profile_functions
    and B_n (cnorm included) from mag_algebra.profile_functions'''
    import mag_algebra as ma
    amp = profile_functions()
    out = {}
    for B, ns in ma.degrees().items():
        for n in ns:
            Bf = ma.profile_functions()[ma.code(B, n)]
            for a, (c, _) in AMPS.items():
                out[product_code(a, B, n)] = lambda r, Dorder, A=amp[c], Bf=Bf: A(r, 0)*Bf(r, Dorder)
    return out

def magnetic_odd_codes():
    '''the product codes of odd parity at ricb = 0 (utils.odd_profiles): the amplitudes are even, B_n has the parity (-1)^n;
    NEW (2026-11-06, step 2) the exact spheroid's terms have mag_algebra's codes of B_n alone (section 8)'''
    import mag_algebra as ma
    if exact():
        return ma.odd_codes()
    return [product_code(a, B, n) for B, ns in ma.degrees().items() for n in ns if n % 2 for a in AMPS]

# --- monomials (p, X, d, B, n, j, lam, a): r^p X^(d) [a B_n^(j)] λ_lam (mag_algebra's with the amplitude a, None if none);
# an expression is {monomial: sparse matrix [target, source]}; a radial factor of the background is (coef, p, B, n, j, a)
def _add(e, key, M):
    e[key] = e[key] + M if key in e else M

def _shift(e, c, p):
    '''c r^p times an expression'''
    return {(k[0] + p,) + k[1:]: c*M for k, M in e.items()}

def _deriv(e):
    '''d/dr of an expression (product rule on r^p, X^(d), B^(j), λ_lam and the amplitude)'''
    out = {}
    for (p, X, d, B, n, j, lam, a), M in e.items():
        if p != 0:
            _add(out, (p - 1, X, d, B, n, j, lam, a), p*M)
        _add(out, (p, X, d + 1, B, n, j, lam, a), M)
        if B is not None:
            _add(out, (p, X, d, B, n, j + 1, lam, a), M)
        if lam:
            _add(out, (p, X, d, B, n, j, lam + 1, a), M)
        if a is not None:
            if AMPS[a][1] is None:
                sys.exit('centrifugal: a derivative of the amplitude %s has no profile code' % a)
            _add(out, (p - 1, X, d, B, n, j, lam, AMPS[a][1]), M)
    return out

def _lam(e):
    '''λ1 times an expression'''
    out = {}
    for (p, X, d, B, n, j, lam, a), M in e.items():
        if lam:
            sys.exit('centrifugal: a product of two (ln ρ0) derivatives has no label')
        _add(out, (p, X, d, B, n, j, 1, a), M)
    return out

def _rad_shift(rad, k):
    return [(c, p + k, B, n, j, a) for (c, p, B, n, j, a) in rad]

def _rad_deriv(rad):
    '''d/dr of a background radial factor'''
    out = []
    for (c, p, B, n, j, a) in rad:
        if p != 0:
            out.append((p*c, p - 1, B, n, j, a))
        out.append((c, p, B, n, j + 1, a))
        if a is not None:
            out.append((c, p - 1, B, n, j, AMPS[a][1]))
    return out

def _with_amp(rad, a):
    out = []
    for (c, p, B, n, j, a0) in rad:
        if a0 is not None:
            sys.exit('centrifugal: two amplitudes in one radial factor')
        out.append((c, p, B, n, j, a))
    return out

# --- the background vectors: [(kind, c(x), radial factor)], kind 'r' (c r̂), 'st' (c sinθ θ̂), 'sp' (c sinθ φ̂)
def _background(n, B):
    '''mag_algebra._background (B0 and J0 of degree n) with the amplitude slot'''
    import mag_algebra as ma
    return {k: [(kind, c, [mono + (None,) for mono in rad]) for kind, c, rad in v] for k, v in ma._background(n, B).items()}

def _scaled(vecs, poly, a):
    '''poly(x) a(r) times background vectors'''
    return [(kind, poly*c, _with_amp(rad, a)) for kind, c, rad in vecs]

def _metric_vectors(vecs, a):
    '''(M - I) V per unit amplitude of background vectors V: a 'E', ε (3 V_z ẑ - V); a 'N', η [(3/2) x (V_z r̂ + V_r ẑ) - V_r r̂
    - P2 V] (section 1), with ẑ = x r̂ - sinθ θ̂, V_z = x V_r - sinθ V_θ'''
    x = _xp
    out = []
    for kind, c, rad in vecs:
        Vr = c if kind == 'r' else 0*x
        Vz = x*c if kind == 'r' else (-1*(1 - x*x)*c if kind == 'st' else 0*x)
        if a == 'E':
            terms = [('r', 3*x*Vz), ('st', -3*Vz), (kind, -1*c)]
        else:
            terms = [('r', 1.5*x*Vz + 1.5*x*x*Vr - Vr), ('st', -1.5*x*Vr), (kind, -1*_P2p*c)]
        ra = _with_amp(rad, a)
        out += [(k2, c2, ra) for k2, c2 in terms if np.any(c2.coef != 0)]
    return out

def _curl_vectors(vecs):
    '''∇× of axisymmetric background vectors:
        a c r̂ -> (a/r) c' sinθ φ̂;   b c sinθ θ̂ -> ((r b)'/r) c sinθ φ̂;   d c sinθ φ̂ -> -(d/r) [(1 - x²) c]' r̂ - ((r d)'/r) c sinθ θ̂'''
    x = _xp
    out = []
    for kind, c, rad in vecs:
        if kind == 'r':
            out.append(('sp', c.deriv(), _rad_shift(rad, -1)))
        elif kind == 'st':
            out.append(('sp', c, _rad_shift(_rad_deriv(_rad_shift(rad, 1)), -1)))
        else:
            out.append(('r', -1*(c*(1 - x*x)).deriv(), _rad_shift(rad, -1)))
            out.append(('st', -1*c, _rad_shift(_rad_deriv(_rad_shift(rad, 1)), -1)))
    return [v for v in out if np.any(v[1].coef != 0)]

# --- the perturbation: {monomial: Field}
def _pert(A, X, kind):
    '''mag_algebra._pert (u from P, T; b from F, G; J = ∇×b) with the amplitude slot'''
    import mag_algebra as ma
    return {k + (None,): F for k, F in ma._pert(A, X, kind).items()}

def _metric_pert(A, pert, a, lw):
    '''(M - I) of a perturbation per unit amplitude (a 'E': ε, 'N': η), with the primitives of the inertia (section 2)'''
    C, _ = primitives('inertia', 'e' if a == 'E' else 'h', A.m, lw)
    out = {}
    for (p, X, d, B, n, j, lam, a0), F in pert.items():
        if a0 is not None:
            sys.exit('centrifugal: two amplitudes in one perturbation')
        out[(p, X, d, B, n, j, lam, a)] = angular.Field(*[C[k]['q'] @ F.q + C[k]['a'] @ F.s + C[k]['T'] @ F.t for k in 'qst'])
    return out

def _curl_pert(A, pert):
    '''∇× of a perturbation: (q, s, t) -> (L t/r, (r t)'/r, (q - (r s)')/r) on R_l, S_l, T_l'''
    comp = {'q': {}, 's': {}, 't': {}}
    for key, F in pert.items():
        for k in 'qst':
            _add(comp[k], key, F[k])
    out = {'q': {}, 's': {}, 't': {}}
    for key, M in comp['t'].items():
        _add(out['q'], key, A.L @ M)
    for key, M in _deriv(_shift(comp['t'], 1., 1)).items():
        _add(out['s'], key, M)
    for key, M in comp['q'].items():
        _add(out['t'], key, M)
    for key, M in _deriv(_shift(comp['s'], 1., 1)).items():
        _add(out['t'], key, -M)
    Z = A.zero()
    keys = dict.fromkeys([*out['q'], *out['s'], *out['t']])   # FIX (2026-11-06): an ordered union, not a set, whose order
    #   follows the string hashes, which change from run to run (the sums then differed in the last bits between builds)
    return {(k[0] - 1,) + k[1:]: angular.Field(*[out[c].get(k, Z) for c in 'qst']) for k in keys}   # the 1/r

def _product(A, vecs, pert, sign):
    '''sign Σ V × pert over background vectors V: {component: expression}'''
    import mag_algebra as ma
    out = {'q': {}, 's': {}, 't': {}}
    for kind, c, rad in vecs:
        for (p0, X, d, B0, n0, j0, lam, a0), F in pert.items():
            if B0 is not None:
                sys.exit('centrifugal: two background factors in one monomial')
            W = ma._vcross(A, kind, c, F)
            for k in 'qst':
                Mk = (sign*W[k]).tocsr()
                if Mk.nnz == 0:
                    continue
                for (cc, p, B, n, j, a) in rad:
                    if a is not None and a0 is not None:
                        sys.exit('centrifugal: two amplitudes in one monomial')
                    _add(out[k], (p0 + p, X, d, B, n, j, lam, a if a is not None else a0), cc*Mk)
    return out

def _merge(G, H):
    for k in 'qst':
        for key, M in H[k].items():
            _add(G[k], key, M)
    return G

def _store(out, row, X, e, Lt):
    '''the row's monomials as labels '{row}{rx}{product code}{j}[lho{k}][ohr1]_D{d}', times L' of the target'''
    import mag_algebra as ma
    d = out.setdefault((row, X), {})
    for (p, Xn, dd, B, n, j, lam, a), M in e.items():
        if M.nnz == 0 or abs(M).max() == 0:
            continue
        rx = ma.ROWPOW[row] - p
        if not 0 <= rx <= 9 or B is None or a is None or j > 9 or dd > 9:
            sys.exit('centrifugal: monomial r^%d %s^(%d) %s%s^(%d) has no label in row %s' % (p, Xn, dd, a, B, j, row))
        lab = '%s%d%s%d%s%s_D%d' % (row, rx, product_code(a, B, n), j, 'lho%d' % lam if lam else '',
                                    'ohr1' if row in 'uv' else '', dd)
        _add(d, lab, Lt(M).tocsr())

def magnetic_build(m, degs, lw):
    '''
    {(row, column): {label: sparse matrix [target l', source l]}}: the metric parts of the Lorentz force (rows 'u', 'v',
    columns 'F', 'G') and of the induction (rows 'f', 'g', columns 'P', 'T') for the background degrees degs = {'h': [...],
    'k': [...]} on the degrees 0..lw, in kore's A sign, without Hendrik and the 1/L'^3 of the row (as mag_algebra.build;
    also used by tests/centrifugal/derive_mag.py without a run)
    '''
    import mag_algebra as ma
    A = angular.Algebra.get(m, lw)
    Lt = lambda M: A.L @ M
    out = {}
    for B in 'hk':
        for n in degs.get(B, []):
            bg = _background(n, B)
            bN = {k: _scaled(v, -1*_P2p, 'N') for k, v in bg.items()}             # -η P2 B0, -η P2 J0
            Jm = {a: _curl_vectors(_metric_vectors(bg['B0'], a)) for a in 'EN'}   # ∇×((M - I)B0), per amplitude
            for X in 'FG':   # δ𝐆
                b, J = _pert(A, X, 'b'), _pert(A, X, 'J')
                G = _product(A, bN['J0'], b, 1.)
                _merge(G, _product(A, bN['B0'], J, -1.))
                for a in 'EN':
                    _merge(G, _product(A, Jm[a], b, 1.))
                    _merge(G, _product(A, bg['B0'], _curl_pert(A, _metric_pert(A, b, a, lw)), -1.))
                # u: -r³ [q - (r s)' + λ1 r s];   v: -r² t   (mag_algebra.build)
                s_r = _shift(G['s'], 1., 1)
                urow = dict(G['q'])
                for key, M in _deriv(s_r).items():
                    _add(urow, key, -M)
                for key, M in _lam(s_r).items():
                    _add(urow, key, M)
                _store(out, 'u', X, _shift(urow, -1., 3), Lt)
                _store(out, 'v', X, _shift(G['t'], -1., 2), Lt)
            for X in 'PT':   # δ𝐄 = u × (-η P2 B0)
                E = _product(A, bN['B0'], _pert(A, X, 'u'), -1.)
                _store(out, 'f', X, _shift(E['t'], -1., 2), Lt)          # f: -r² t
                s_r = _shift(E['s'], 1., 1)
                grow = dict(E['q'])
                for key, M in _deriv(s_r).items():                        # g: -r [q - (r s)']
                    _add(grow, key, -M)
                _store(out, 'g', X, _shift(grow, -1., 1), Lt)
    return ma._prune(out)

def _bandwidth(blocks):
    '''the largest |Δl| of the nonzero coefficients'''
    B = 0
    for d in blocks.values():
        for M in d.values():
            M = M.tocoo()
            if M.nnz:
                B = max(B, int(np.abs(M.row - M.col).max()))
    return B

_mcoef, _mband = {}, {}
def magnetic_coefficients(m):
    '''magnetic_build for this run's background (cached), also stored by diagonal; NEW (2026-11-06, step 2) the exact
    spheroid's spheroid_magnetic_build (section 8)'''
    import mag_algebra as ma
    if m not in _mcoef:
        build = spheroid_magnetic_build if exact() else magnetic_build
        out = build(m, ma.degrees(), par.lmax + ma.nmax() + 16)
        _mcoef[m] = out
        B = _bandwidth(out)
        _mband[m] = (B, {rc: {lab: angular.band(M, B) for lab, M in d.items()} for rc, d in out.items()})
    return _mcoef[m]

def magnetic_labels():
    '''[(label, 'P', 'T', 'F' or 'G')]: the radial operators of the metric magnetic terms and their columns, for submatrices.py'''
    out = []
    for (row, X), d in magnetic_coefficients(par.m).items():
        out += [(lab, X) for lab in d]
    return out

def magnetic_dl_range():
    '''the Δl of the metric magnetic couplings, -B..B (B the bandwidth of the coefficients: nmax + 3 at most)'''
    magnetic_coefficients(par.m)
    B = _mband[par.m][0]
    return range(-B, B + 1)

def magnetic_block(l, section, component, offdiag, M):
    '''
    The block of the metric part of the Lorentz force (section 'u', 'v'; component 'bpol', 'btor') or of the induction
    (section 'f', 'g'; component 'upol', 'utor') for the row degree l and the source degree l + offdiag, as mag_algebra.block:
    [Σ coefficient × radial operator / L'^3 (times Hendrik for the force), the column block offset in units of N1], or
    [0, 0] if nothing couples this Δl.
    '''
    import utils as ut
    X = {'bpol': 'F', 'btor': 'G', 'upol': 'P', 'utor': 'T'}[component]
    magnetic_coefficients(par.m)
    bands = _mband[par.m][1].get((section, X), {})
    out = 0
    for lab, bnd in bands.items():
        v = _lookup(bnd, l, par.m, offdiag)
        if v != 0:
            out = out + v*M[lab]
    if np.isscalar(out):
        return [0, 0]
    lp, lt = ut.ell(par.m, par.lmax, par.symm)[:2]
    lF, lG = ut.ell(par.m, par.lmax, ut.bsymm)[:2]
    rows = {'u': lp, 'v': lt, 'f': lF, 'g': lG}[section]
    cols = {'F': lF, 'G': lG, 'P': lp, 'T': lt}[X]
    offd = int(np.where(cols == l + offdiag)[0][0] - np.where(rows == l)[0][0])
    scale = par.Hendrik if section in 'uv' else 1.
    return [scale*out/(l*(l + 1.))**3, offd]


# ======================================================================================================================
# 5. The insulating walls (NEW 2026-11-05, phase 1b)
# ======================================================================================================================
# Outside an insulating wall the field is ∇φ, ∇²φ = 0, in the body. The wall is the level surface r = R of the map, and the
# exact conditions there, in the reference space, are:
#     the flux through the wall:       b̃_r = [J F⁻¹ ∇φ(X)]_r = [∇φ - ∇×(ξ×∇φ)]_r          (Nanson; first order, ∇²φ = 0)
#     the tangential field:            [M b̃]_tan = ∇_tan Φ,  Φ(α) = φ(X(α)) = φ + ξ⋅∇φ     (Fᵀ b, continuous)
# (∇φ, ξ⋅∇φ and the curl evaluated at r = R: the radial component of a curl needs the values on the sphere only). The
# tangential gradient has no toroidal part, so with m_s, m_t the S and T parts of (M - I) b̃ at R, δs = (ξ⋅∇φ)_l/R and
# δq = -[∇×(ξ×∇φ)]_r, eliminating the potential φ = Σ a_l r^-(l+1) Y_l (outside the CMB) or a_l r^l Y_l (inside an insulating
# inner core), whose metric terms take its spherical value a_l R^-(l+2) (or a_l R^(l-1)) = c_l = F_l' + F_l/R, gives
#     outer:  (l+1) F + R F' + R (m_s - δs) - R δq/(l+1) = 0,      G + m_t = 0
#     inner:  l F - R F' - R (m_s - δs) - R δq/l = 0,              G + m_t = 0
# rows that couple F_l to F_(l±2) and G_(l±1, l±3), and G_l to G_(l±2) and F_(l±1, l±3). At a conducting inner core on the
# same map (background.check_centrifugal: mu_i2o = 1) b̃_r is continuous, with it the tangential M b̃ continuous is the
# tangential b̃ continuous, so the rows that match the field keep their form; FIX (2026-11-06, phase 2): the row of the
# tangential electric field gains the metric part of the diffusion (section 6, icb_rows).
def wall_coefficients(wall, m, lw, eps, eta, R):
    '''
    The metric parts of the wall rows at the radius R of the wall 'outer' or 'inner' with ε = eps, η = eta there: {(row,
    source): sparse matrix [target l, source k]} for the rows 'f', 'g' and the sources 'F0' (F_k(R)), 'F1' (F_k'(R)), 'G0'
    (G_k(R)), on the degrees 0..lw; the rows are those of assemble.bc_b_spherical (f: (l+1) F + R F' outside, l F - R F'
    inside; g: G). Also used by tests/centrifugal/bc_check.py without a run.
    '''
    import mag_algebra as ma
    A = angular.Algebra.get(m, lw)
    I, L, X = A.I, A.L, A.X
    lv = np.arange(lw + 1, dtype=float)
    C = {a: primitives('inertia', a, m, lw)[0] for a in 'eh'}
    amp = {'e': eps, 'h': eta}
    msum = lambda k, sh: amp['e']*C['e'][k][sh] + amp['h']*C['h'][k][sh]
    # (M - I) b̃ at R with q = L F/R, s = F' + F/R, t = G: its S (m_s) and T (m_t) parts per source value
    mS = {'F0': msum('s', 'q') @ L/R + msum('s', 'a')/R, 'F1': msum('s', 'a'), 'G0': msum('s', 'T')}
    mT = {'F0': msum('t', 'q') @ L/R + msum('t', 'a')/R, 'F1': msum('t', 'a'), 'G0': msum('t', 'T')}
    Ds, Dq = _wall_D(A, wall)                                                                  # δs = ε Ds c, δq = -ε L Dq c
    if wall == 'outer':
        sgn, den = 1., sp_diags(1/(lv + 1))
    else:
        sgn, den = -1., sp_diags(np.where(lv > 0, 1/np.where(lv > 0, lv, 1.), 0.))
    # R (m_s - δs) (outer, minus inside) - R δq/den, with c = F' + F/R
    phi = -eps*sgn*Ds + eps*(den @ L @ Dq)                                                    # times R c
    out = {('f', 'F0'): sgn*R*mS['F0'] + phi, ('f', 'F1'): sgn*R*mS['F1'] + R*phi, ('f', 'G0'): sgn*R*mS['G0'],
           ('g', 'F0'): mT['F0'], ('g', 'F1'): mT['F1'], ('g', 'G0'): mT['G0']}
    return {k: sp_csr(v) for k, v in out.items()}

def _wall_D(A, wall):
    '''Ds and Dq of the wall rows on the algebra A: with the potential field ∇φ at R per unit c_l = F_l' + F_l/R (q = -(l+1) c
    outside the CMB, l c inside the ICB; s = c) and ξ = ε R (P2 r̂ - (3/2) x sinθ θ̂) at R, δs = (ξ⋅∇φ)/R = ε Ds c and
    δq = -[∇×(ξ×∇φ)]_r = -ε L Dq c (Dq: the T part of (P2 r̂ - (3/2) x sinθ θ̂)×∇φ)'''
    import mag_algebra as ma
    lv = np.arange(A.lw + 1, dtype=float)
    V = A.field(q=sp_diags(-(lv + 1) if wall == 'outer' else lv) @ A.I, s=A.I)
    P2, _ = A.legendre(2)
    Ds = P2 @ V.q - 1.5*(A.X @ A.sin_theta(V))
    Dq = (ma._vcross(A, 'r', _P2p, V) + ma._vcross(A, 'st', -1.5*_xp, V)).t
    return Ds, Dq

def wall_potential(wall, m, lw, eps, eta, R, q, s, t):
    '''
    The potential field that matches the field b̃ at the wall (post-processing, utils4pp.centrifugal_magnetic_terms): from the
    l-components q, s, t of b̃ at r = R (arrays on the degrees 0..lw), the coefficients A_l of ∇φ at R (∇φ = Σ A_l (-(l+1) Y_l
    r̂ + r∇Y_l) outside the CMB, Σ A_l (l Y_l r̂ + r∇Y_l) inside an insulating inner core), from the tangential matching at
    first order: A = s + m_s - δs, m_s the S part of (M - I) b̃ and δs = ε Ds s (s = c, the spherical A)
    '''
    A = angular.Algebra.get(m, lw)
    ms = 0
    for a, amp in (('e', eps), ('h', eta)):
        C = primitives('inertia', a, m, lw)[0]
        ms = ms + amp*(C['s']['q'] @ q + C['s']['a'] @ s + C['s']['T'] @ t)
    return s + ms - eps*(_wall_D(A, wall)[0] @ s)

_amp_all = {}
def amplitude_series():
    '''
    {'E': ε, 'N': η = r ε', 'C': r η', 'Q': r C', 'K': r Q'} (the amplitudes of AMPS) as Chebyshev series on [-1, 1], also
    defined inside the inner core, where the profile codes (fits on the fluid's domain) are not: from an even interpolant
    of the shape's ε (a conducting inner core: its metric diffusion, section 6, and its post-processing).
    FIX (2026-11-08, phase 6d): the interpolant by background_field.cheb_interp (a DCT), its coefficients below 1e-13 of the
    largest dropped, as B0's profiles (phase 6c). numpy's Chebyshev.interpolate left round-off of about 1e-16 j at degree j,
    which the chain amplified: Q (r³ d³ε/dr³, in the rows gic) was off in the core by 1e-8 of max |ε| at N 160, 2e-6 at N 450
    and 2e-5 at N 800, and the core's operators were dense. The DCT alone still leaves 1e-8 at N 800; the cut alone keeps
    numpy's round-off from N about 450. A 'clairaut' ε has a floor of its own, from its ODE (Q about 2e-9 of max |ε|)
    '''
    if not _amp_all:
        import background_field as bf
        C = np.polynomial.chebyshev.Chebyshev
        S = shape_functions()
        a = bf.cheb_interp(lambda x: S['e'](np.abs(x)), 2*max(par.N, 64) + 1, [-1., 1.])
        a.coef[np.abs(a.coef) < 1e-13*np.abs(a.coef).max()] = 0.   # the round-off, which the derivatives amplify
        x = C.identity(domain=[-1, 1])
        for k in 'ENCQK':   # NEW (2026-11-06): the whole chain (was ε, η, r η')
            _amp_all[k] = a
            a = x*a.deriv()
    return _amp_all

def amplitudes(r):
    '''(ε, η = r ε', r η') at any radii 0 <= r <= 1, also inside the inner core (amplitude_series; post-processing of a
    conducting inner core, utils4pp.centrifugal_magnetic_terms)'''
    A = amplitude_series()
    r = np.asarray(r, dtype=float)
    return A['E'](r), A['N'](r), A['C'](r)

def sp_diags(v):
    import scipy.sparse as sps
    return sps.diags(np.asarray(v, dtype=float)).tocsr()

def sp_csr(M):
    import scipy.sparse as sps
    return sps.csr_matrix(M)

_wall = {}
def wall_rows(wall):
    '''wall_coefficients for this run at the wall 'outer' (r = 1) or 'inner' (r = ricb, insulating inner core), cached, by
    diagonal: (B, {(row, source): bands}); NEW (2026-11-06, step 2) the exact spheroid's spheroid_field_walls (section 8)'''
    import utils as ut
    if wall not in _wall:
        R = ut.rcmb if wall == 'outer' else par.ricb
        if exact():
            W = spheroid_field_walls(wall, par.m, par.lmax + 12, R)
        else:
            prof = profile_functions()
            eps = float(prof['ce0'](np.array([R]), 0)[0])
            eta = float(prof['ce1'](np.array([R]), 0)[0])
            W = wall_coefficients(wall, par.m, par.lmax + 12, eps, eta, R)
        B = _bandwidth({0: W})
        _wall[wall] = (B, {k: angular.band(M, B) for k, M in W.items()})
    return _wall[wall]


# ======================================================================================================================
# 6. The dissipative terms (NEW 2026-11-05, phase 2)
# ======================================================================================================================
# The viscous force, the thermal and the magnetic diffusion are operators of the body. Pulled back to the reference space
# they gain first-order parts, as the inertia did, but with derivatives: they couple a degree l to l ± 2 of the same kind and
# to l ± 1, l ± 3 of the other kind, through radial operators made of the amplitudes of the map, the coefficient profile (ν,
# κ p or η) and the (ln ρ0) derivatives. From the map alone (tests/centrifugal/derive_diff.py checks each formula):
#   viscous force per unit mass, V(u) = (1/ρ0) ∇⋅(ρ0 ν σ(u)), σ(u) = ∇u + (∇u)ᵀ - (2/3)(∇⋅u) I. Its covariant form is
#   Fᵀ V(F w/J) = V(w) + δV with, up to a gradient (which kore's curl rows remove),
#       δV = -ξ×(∇×V(w)) + V(∇×(ξ×w) + λ1 w_r ξ) - ξ_r ν' ∇⋅σ(w) - ∇(ξ_r κ)⋅σ(w),      κ = ν' + ν λ1:
#   the first two terms are the change of coordinates (the Lie derivative of the force along ξ, with ∇⋅w = -λ1 w_r), the last
#   two that of the profiles, which are constant on the level surfaces and not on the spheres;
#   heat rows (the equation times J): J ∇_X⋅(κp ∇_X s) = ∇⋅(κp M⁻¹ ∇s), first order δ = -∇⋅(κp (M - I) ∇s); NEW (2026-11-08)
#   the rows of the composition the same with D ρ0 and c;
#   induction: the electric field of the diffusion is Fᵀ(η ∇_X×b) = η M ∇×(M b̃), first order δ = η [(M - I) ∇×b̃ + ∇×((M - I) b̃)];
#   walls: stress-free, the tangential part of Fᵀ σ_X F⁻ᵀ r̂ (the traction on the level surface, in the tangent vectors F t);
#   fixed flux, r̂⋅M⁻¹∇s = 0 (the flux through the level surface).
# They are built with the recurrence algebra of section 2 on expressions {monomial: Field (or sparse matrix)}, a monomial
# (p, X, d, a, j, lam) being r^p X^(d) a c^(j) Λ: a power of r, the d-th derivative of the unknown X (P, T, S for the entropy,
# C for the composition, F, G; None for a radial factor alone), an amplitude a of the map (AMPS), the j-th derivative of the
# coefficient profile c and a product Λ of (ln ρ0) derivatives, the sorted tuple of their orders. A label gets the product
# code a + 'vs', 'kp', 'Dr' or 'et' (a c, DIFF_CODES) with the digit j, and Λ as 'lho' k (λ_k), or 'lh1' k, 'lh2' k (λ1 λ_k,
# λ2 λ_k) in the first slot.
# Terms whose order is above the row's (r⁵ ρ0 r̂⋅∇×∇× has fourth derivatives of P) appear term by term and cancel in the sum,
# to round-off (dropped by mag_algebra._prune, and checked: _check_orders). The toroidal rows (v, g) keep third derivatives of
# P (F) for m ≠ 0, the part of ∂ᵣ² of the poloidal flow that the P2 factors turn toroidal: submatrices.py then builds those
# sections in the C^(3) basis (orders()).
DIFF_CODES = {'vs': 'vsc', 'kp': 'kps', 'et': 'eta', 'Dr': 'Drh'}   # the product code's suffix: the code of the profile it
# multiplies (NEW 2026-11-08: 'Dr', D ρ0 of the composition)
ROWPOW2 = {'u': 5, 'v': 3, 'h': 2, 'f': 2, 'g': 2, 'c': 2}     # the power of r of the rows (decode_label: rpower = ROWPOW2 - rx)

def diffusion_profile_functions():
    '''{product code: f(r, Dorder)} for radial_profiles.burrito: a(r) c^(Dorder)(r), the amplitude a (AMPS, from
    profile_functions) and the coefficient profile c of radial_profiles (vsc: ν, kps: κ p, eta: η, NEW 2026-11-08 Drh: D ρ0)'''
    import radial_profiles as rap
    amp = profile_functions()
    out = {}
    for a, (ca, _) in AMPS.items():
        for sfx, base in DIFF_CODES.items():
            out[a + sfx] = lambda r, Dorder, A=amp[ca], base=base: A(r, 0)*rap.proffdir[base](r, Dorder)
    return out

# --- expressions of monomials (p, X, d, a, j, lam)
def _sum(*es):
    out = {}
    for e in es:
        for k, M in e.items():
            _add(out, k, M)
    return out

def _sc(e, c):
    return {k: c*M for k, M in e.items()}

def _map(e, fn):
    return {k: fn(M) for k, M in e.items()}

def _dr(e):
    '''d/dr of an expression: the product rule on r^p, X^(d) (X not None), the amplitude (a' = (next amplitude)/r), the
    profile c^(j) (j not None) and each λ_k of Λ (λ_k' = λ_(k+1))'''
    out = {}
    for (p, X, d, a, j, lam), M in e.items():
        if p != 0:
            _add(out, (p - 1, X, d, a, j, lam), p*M)
        if X is not None:
            _add(out, (p, X, d + 1, a, j, lam), M)
        if a is not None:
            if AMPS[a][1] is None:
                sys.exit('centrifugal: a derivative of the amplitude %s has no profile code' % a)
            _add(out, (p - 1, X, d, AMPS[a][1], j, lam), M)
        if j is not None:
            _add(out, (p, X, d, a, j + 1, lam), M)
        for i in range(len(lam)):
            _add(out, (p, X, d, a, j, tuple(sorted(lam[:i] + (lam[i] + 1,) + lam[i + 1:]))), M)
    return out

def _rf(*terms):
    '''a radial factor {(p, None, 0, a, j, lam): c} from (c, p, a, j, lam)'''
    out = {}
    for c, p, a, j, lam in terms:
        _add(out, (p, None, 0, a, j, tuple(sorted(lam))), c)
    return out

def _rmul(f, e):
    '''the product of the radial factor f and the expression e'''
    out = {}
    for (p1, _, _, a1, j1, l1), c in f.items():
        for (p, X, d, a, j, lam), M in e.items():
            if a1 is not None and a is not None:
                sys.exit('centrifugal: two amplitudes in one term')
            if j1 is not None and j is not None:
                sys.exit('centrifugal: two profile factors in one term')
            _add(out, (p + p1, X, d, a if a1 is None else a1, j if j1 is None else j1, tuple(sorted(lam + l1))), c*M)
    return out

# --- vector calculus on vector expressions {monomial: Field (q, s, t) on R_l, S_l, T_l} and scalar ones {monomial: matrix}
def _parts(e):
    '''the components q, s, t of a vector expression, as three scalar expressions'''
    return tuple({k: F[c] for k, F in e.items()} for c in 'qst')

def _vec(A, q=None, s=None, t=None):
    '''a vector expression from scalar ones (components left out are zero)'''
    Z = A.zero()
    q, s, t = q or {}, s or {}, t or {}
    return {k: angular.Field(q.get(k, Z), s.get(k, Z), t.get(k, Z)) for k in dict.fromkeys([*q, *s, *t])}   # FIX (2026-11-06):
    #   an ordered union, as in _curl_pert

def _Lt(A, e):
    return _map(e, lambda M: A.L @ M)

def _div(A, e):
    '''∇⋅v = q' + 2q/r - L s/r'''
    q, s, _ = _parts(e)
    return _sum(_dr(q), _shift(q, 2., -1), _shift(_Lt(A, s), -1., -1))

def _grad(A, g):
    '''∇g = g' R + (g/r) S'''
    return _sum(_vec(A, q=_dr(g)), _vec(A, s=_shift(g, 1., -1)))

def _curl(A, e):
    '''∇×v: (q, s, t) -> (L t/r, (r t)'/r, (q - (r s)')/r)'''
    q, s, t = _parts(e)
    return _vec(A, q=_shift(_Lt(A, t), 1., -1), s=_shift(_dr(_shift(t, 1., 1)), 1., -1),
                t=_sum(_shift(q, 1., -1), _shift(_dr(_shift(s, 1., 1)), -1., -1)))

def _divsig(A, e):
    '''∇⋅σ(v) = ∇²v + (1/3)∇(∇⋅v) = (4/3)∇(∇⋅v) - ∇×∇×v'''
    return _sum(_sc(_grad(A, _div(A, e)), 4/3), _sc(_curl(A, _curl(A, e)), -1.))

def _rsig(A, e):
    '''r̂⋅σ(v) = (2q' - (2/3)∇⋅v) R + (s' + (q - s)/r) S + (t' - t/r) T'''
    q, s, t = _parts(e)
    return _vec(A, q=_sum(_sc(_dr(q), 2.), _sc(_div(A, e), -2/3)), s=_sum(_dr(s), _shift(q, 1., -1), _shift(s, -1., -1)),
                t=_sum(_dr(t), _shift(t, -1., -1)))

def _zdot(A, e):
    '''ẑ⋅v = x v_r - sinθ v_θ (scalar)'''
    return _map(e, lambda F: A.X @ F.q - A.sin_theta(F))

def _zhat(A, g):
    '''g ẑ for a scalar g'''
    return _map(g, lambda c: angular.Field(A.X @ c, A.Li @ (A.Dd @ c), A.Li @ (1j*A.m*c)))

def _zcr(A, e):
    return _map(e, A.zcross)

def _rcr(e):
    '''r̂ × v: r̂ × S_l = -T_l, r̂ × T_l = S_l'''
    return _map(e, lambda F: angular.Field(0*F.q, F.t, -1*F.s))

def _poly(A, e, c):
    '''c(x) v, c a numpy Legendre series in x = cos θ'''
    import mag_algebra as ma
    g, dg = ma._poly_mats(A, c)
    return _map(e, lambda F: A.mult(F, g, dg))

def _zsig(A, e):
    '''ẑ⋅σ(v) = 2∇v_z - ẑ×(∇×v) - (2/3)(∇⋅v) ẑ'''
    return _sum(_sc(_grad(A, _zdot(A, e)), 2.), _sc(_zcr(A, _curl(A, e)), -1.), _sc(_zhat(A, _div(A, e)), -2/3))

def _H(A, e):
    '''H v = (3/2) v_z ẑ - v/2, H = ∇∇(r² P2/2) (ξ = ε H 𝐫)'''
    return _sum(_sc(_zhat(A, _zdot(A, e)), 1.5), _sc(e, -0.5))

def _Hr(A, g):
    '''g H r̂ = g [(3/2) x ẑ - r̂/2] for a scalar g'''
    return _sum(_sc(_zhat(A, _map(g, lambda c: A.X @ c)), 1.5), _vec(A, q=_sc(g, -0.5)))

EPS_R = _rf((1., 1, 'E', None, ()))       # ε r: ξ = ε r H r̂, ξ_r = ε r P2
LAM1 = _rf((1., 0, None, None, (1,)))     # λ1

def _xi_cross(A, e):
    '''ξ × v = ε r [(3/2) x ẑ×v - (1/2) r̂×v]'''
    return _rmul(EPS_R, _sum(_poly(A, _zcr(A, e), 1.5*_xp), _sc(_rcr(e), -0.5)))

def _metric(A, e, lw):
    '''(M - I) v (section 1): the amplitudes ε ('E') and η ('N') times the primitives of the inertia'''
    out = {}
    for a, amp in (('E', 'e'), ('N', 'h')):
        C = primitives('inertia', amp, A.m, lw)[0]
        for (p, X, d, a0, j, lam), F in e.items():
            if a0 is not None:
                sys.exit('centrifugal: two amplitudes in one term')
            _add(out, (p, X, d, a, j, lam), angular.Field(*[C[k]['q'] @ F.q + C[k]['a'] @ F.s + C[k]['T'] @ F.t for k in 'qst']))
    return out

# --- the unknowns
def _flow(A, X):
    '''w of the unknown X: P, q = L P/r, s = P' + P/r + λ1 P; T, t = T'''
    I, L = A.I, A.L
    if X == 'P':
        return _vec(A, q={(-1, 'P', 0, None, None, ()): L},
                    s={(-1, 'P', 0, None, None, ()): I, (0, 'P', 1, None, None, ()): I, (0, 'P', 0, None, None, (1,)): I})
    return _vec(A, t={(0, 'T', 0, None, None, ()): I})

def _field(A, X):
    '''b̃ of the unknown X: F, q = L F/r, s = F' + F/r; G, t = G'''
    I, L = A.I, A.L
    if X == 'F':
        return _vec(A, q={(-1, 'F', 0, None, None, ()): L}, s={(-1, 'F', 0, None, None, ()): I, (0, 'F', 1, None, None, ()): I})
    return _vec(A, t={(0, 'G', 0, None, None, ()): I})

NU = _rf((1., 0, None, 0, ()))                                    # ν
KAPPA = _rf((1., 0, None, 1, ()), (1., 0, None, 0, (1,)))         # κ = ν' + ν λ1 = (ρ0 ν)'/ρ0

def _visc(A, e):
    '''the viscous force per unit mass of a flow v: V(v) = ν ∇⋅σ(v) + κ r̂⋅σ(v)'''
    return _sum(_rmul(NU, _divsig(A, e)), _rmul(KAPPA, _rsig(A, e)))

def viscous_force(A, X):
    '''δV of the flow of the unknown X ('P' or 'T'), up to a gradient (the formula above), as a vector expression'''
    w = _flow(A, X)
    out = _sc(_xi_cross(A, _curl(A, _visc(A, w))), -1.)                                        # -ξ×(∇×V(w))
    du = _sum(_curl(A, _xi_cross(A, w)), _rmul(EPS_R, _Hr(A, _rmul(LAM1, _parts(w)[0]))))   # ∇×(ξ×w) + λ1 w_r ξ
    out = _sum(out, _visc(A, du))
    out = _sum(out, _rmul(_rf((-1., 1, 'E', 1, ())), _poly(A, _divsig(A, w), _P2p)))         # -ξ_r ν' ∇⋅σ(w)
    g = _rf((1., 1, 'E', 1, ()), (1., 1, 'E', 0, (1,)))                                       # ξ_r κ = g P2, g = ε r κ
    rs = _rsig(A, w)
    # -∇(g P2)⋅σ(w) = -[g' P2 - 3 (g/r) x²] r̂⋅σ(w) - 3 (g/r) x ẑ⋅σ(w)
    return _sum(out, _sc(_rmul(_dr(g), _poly(A, rs, _P2p)), -1.), _rmul(_shift(g, 3., -1), _poly(A, rs, _xp*_xp)),
                _rmul(_shift(g, -3., -1), _poly(A, _zsig(A, w), _xp)))

def thermal_diffusion_metric(A, lw, X='S'):
    '''-∇⋅(κp (M - I) ∇s), the first-order part of J ∇⋅(κp ∇s) (scalar expression; ∇s: q = s', s/r on S); NEW (2026-11-08)
    X = 'C', the same for the composition c (κp is then D ρ0)'''
    gs = _vec(A, q={(0, X, 1, None, None, ()): A.I}, s={(-1, X, 0, None, None, ()): A.I})
    return _sc(_div(A, _rmul(NU, _metric(A, gs, lw))), -1.)      # NU: the profile slot, here κ p (D ρ0)

def magnetic_diffusion_metric(A, X, lw):
    '''the first-order part of the electric field of the diffusion, -η [(M - I) ∇×b̃ + ∇×((M - I) b̃)], b̃ of the unknown X'''
    b = _field(A, X)
    return _sc(_rmul(NU, _sum(_metric(A, _curl(A, b), lw), _curl(A, _metric(A, b, lw)))), -1.)   # NU: here η

# --- the rows and their labels
def _label(row, rx, a, sfx, j, lam, d):
    '''the label of r^(rowpower - rx) [a c^(j)] Λ X^(d) in the row: '{row}{rx}{a}{sfx}{j}[lho{k}]_D{d}' or, with a product of
    two (ln ρ0) derivatives, '{row}{rx}lh{i}{k}{a}{sfx}{j}_D{d}' (burrito takes such products in the first slot); None if it
    has none'''
    prod = '%s%s%d' % (a, sfx, j)
    if len(lam) == 0:
        mid = prod
    elif len(lam) == 1:
        mid = prod + 'lho%d' % lam[0]
    elif len(lam) == 2 and lam[0] in (1, 2) and lam[1] <= 9:
        mid = 'lh%d%d' % lam + prod
    else:
        return None
    return '%s%d%s_D%d' % (row, rx, mid, d)

def _store2(out, row, X, e, Lt, sfx):
    '''the row's monomials as labels, times Lt (L' of the target, or 1 for the heat rows), added to out[(row, X)]'''
    d_ = out.setdefault((row, X), {})
    for (p, Xn, dd, a, j, lam), M in e.items():
        M = M.tocsr()
        if M.nnz == 0 or abs(M).max() == 0:
            continue
        rx = ROWPOW2[row] - p
        lab = _label(row, rx, a, sfx, j, lam, dd) if (0 <= rx <= 9 and a is not None and j is not None and j <= 9
                                                       and dd <= 9 and Xn == X) else None
        if lab is None:
            sys.exit('centrifugal: the term r^%d %s^(%d) %s c^(%s) λ%s has no label in row %s' % (p, Xn, dd, a, j, lam, row))
        _add(d_, lab, Lt(M).tocsr())

def _row_u(e):
    '''r³ [q - (r s)'] of a force per unit mass (times L', ρ0 and 1/L'³ in the block: the row r⁵ ρ0 r̂⋅∇×∇×)'''
    q, s, _ = _parts(e)
    return _shift(_sum(q, _sc(_dr(_shift(s, 1., 1)), -1.)), 1., 3)

def _row_v(e):
    '''r² t (the row r³ ρ0 r̂⋅∇×)'''
    return _shift(_parts(e)[2], 1., 2)

def diffusion_build(m, lw, kinds=('visc', 'heat', 'mag')):
    '''
    {(row, column): {label: sparse matrix [target l', source l]}} of the first-order dissipative terms on the degrees 0..lw:
    'visc' the viscous force δV in the rows u, v (columns P, T; the sign of operators.viscous_diffusion, without ViscosD and
    the 1/L'^3), 'heat' -∇⋅(κp (M - I) ∇s) in the rows h (column S; the sign of operators.thermal_diffusion, without ThermaD
    and 1/L^3), 'mag' the field of the diffusion in the rows f, g (columns F, G; kore's A sign, as magnetic_build, without
    MagnetD and the 1/L'^3), NEW (2026-11-08) 'comp' -∇⋅(D ρ0 (M - I) ∇c) in the rows c (column C; the sign of
    operators.comp_diffusion, without CompoD and 1/L^3); pruned (mag_algebra._prune). Also used by
    tests/centrifugal/derive_diff.py without a run.
    '''
    import mag_algebra as ma
    A = angular.Algebra.get(m, lw)
    Lt = lambda M: A.L @ M
    out = {}
    if 'visc' in kinds:
        for X in 'PT':
            V = viscous_force(A, X)
            _store2(out, 'u', X, _row_u(V), Lt, 'vs')
            _store2(out, 'v', X, _row_v(V), Lt, 'vs')
    if 'heat' in kinds:
        _store2(out, 'h', 'S', _shift(thermal_diffusion_metric(A, lw), 1., 2), lambda M: M, 'kp')   # r² (the row)
    if 'comp' in kinds:   # NEW (2026-11-08)
        _store2(out, 'c', 'C', _shift(thermal_diffusion_metric(A, lw, 'C'), 1., 2), lambda M: M, 'Dr')   # r² (the row)
    if 'mag' in kinds:
        for X in 'FG':
            E = magnetic_diffusion_metric(A, X, lw)
            q, s, t = _parts(E)
            _store2(out, 'f', X, _shift(t, -1., 2), Lt, 'et')                                       # f: -r² t
            _store2(out, 'g', X, _shift(_sum(q, _sc(_dr(_shift(s, 1., 1)), -1.)), -1., 1), Lt, 'et')   # g: -r [q - (r s)']
    out = ma._prune(out)
    _check_orders(out)
    return out

MAXD = {'u': 4, 'v': 3, 'h': 2, 'f': 2, 'g': 3, 'c': 2}
def _check_orders(out):
    '''stops if a term above the order of its row is left (they cancel in exact arithmetic: see the section's header)'''
    for (row, X), d in out.items():
        for lab in d:
            if int(lab[-1]) > MAXD[row]:
                sys.exit('centrifugal: the metric term %s (column %s) is above the order of its row: the cancellation failed' % (lab, X))

# --- the stress-free and fixed-flux wall rows
E1, N1 = _rf((1., 0, 'E', None, ())), _rf((1., 0, 'N', None, ()))   # ε, η

def wall_traction(A, X):
    '''
    The first-order part of the traction Fᵀ σ_X F⁻ᵀ r̂ on a level surface, per unit ν (vector expression; its S and T parts
    make the stress-free rows), for the flow of the unknown X (tests/centrifugal/derive_diff.py, part A):
        r̂⋅σ(δu0) - (a⋅∇)w - ε H b + ε H (r̂⋅σ(w)) - (3/2) ε x ẑ⋅σ(w) + (ε/2 - η P2) r̂⋅σ(w)   (+ radial parts),
    δu0 = (∇ξ)w - (∇⋅ξ)w = ε H w + η w_r H r̂ - η P2 w (the velocity of the body minus w, at the same reference point),
    a = ∂ᵣξ = (ε + η) H r̂, (ẑ⋅∇)w = ∇w_z - ẑ×(∇×w), b = (∇w)ᵀ r̂ = ∇w_r - (w - w_r r̂)/r
    '''
    w = _flow(A, X)
    q, s, t = _parts(w)
    rs = _rsig(A, w)
    du0 = _sum(_rmul(E1, _H(A, w)), _rmul(N1, _Hr(A, q)), _sc(_rmul(N1, _poly(A, w, _P2p)), -1.))
    zgrad = _sum(_grad(A, _zdot(A, w)), _sc(_zcr(A, _curl(A, w)), -1.))
    adw = _rmul(_sum(E1, N1), _sum(_poly(A, zgrad, 1.5*_xp), _sc(_dr(w), -0.5)))
    b = _sum(_grad(A, q), _sc(_shift(_vec(A, s=s, t=t), 1., -1), -1.))
    return _sum(_rsig(A, du0), _sc(adw, -1.), _sc(_rmul(E1, _H(A, b)), -1.), _rmul(E1, _H(A, rs)),
                _sc(_rmul(E1, _poly(A, _zsig(A, w), _xp)), -1.5), _rmul(_sc(E1, 0.5), rs), _sc(_rmul(N1, _poly(A, rs, _P2p)), -1.))

def wall_flux(A, lw):
    '''-r̂⋅(M - I)∇s, the first-order part of the flux r̂⋅M⁻¹∇s through a level surface (scalar expression)'''
    gs = _vec(A, q={(0, 'S', 1, None, None, ()): A.I}, s={(-1, 'S', 0, None, None, ()): A.I})
    return _sc(_parts(_metric(A, gs, lw))[0], -1.)

def _at(e, R, vals):
    '''{(X, d): matrix}: the expression at r = R, with vals = {amplitude: a(R), ('lam', k): λ_k(R)} (no profile factor)'''
    out = {}
    for (p, X, d, a, j, lam), M in e.items():
        if j is not None:
            sys.exit('centrifugal: a profile factor in a wall row')
        c = R**p*(1. if a is None else vals[a])
        for k in lam:
            c = c*vals[('lam', k)]
        _add(out, (X, d), c*M)
    return out

def wall_values(R):
    '''the amplitudes and the (ln ρ0) derivatives at the radius R, for wall_dissipative'''
    import radial_profiles as rap
    prof = profile_functions()
    vals = {a: float(prof[c](np.array([R]), 0)[0]) for a, (c, _) in AMPS.items()}
    vals.update({('lam', k): float(np.asarray(rap.logrhoX(np.array([R]), k)).ravel()[0]) for k in range(1, 4)})
    return vals

def wall_dissipative(m, lw, R, vals):
    '''
    The metric parts of the stress-free and fixed-flux wall rows at the radius R: {(row, source): sparse matrix [l, l']},
    rows 'u' (R² times the S part of the traction: kore's row of zero consoidal stress), 'v' (R times its T part: zero
    toroidal stress) and 'h' (the flux, kore's row s' = 0); sources 'P0', 'P1', 'P2', 'T0', 'T1', 'S0', 'S1' (the column's
    value and derivatives at R); vals as wall_values. Also used by tests/centrifugal/derive_diff.py without a run.
    '''
    A = angular.Algebra.get(m, lw)
    out = {}
    for X in 'PT':
        for (Xn, d), F in _at(wall_traction(A, X), R, vals).items():
            _add(out, ('u', '%s%d' % (Xn, d)), R**2*F.s)
            _add(out, ('v', '%s%d' % (Xn, d)), R*F.t)
    for (Xn, d), M in _at(wall_flux(A, lw), R, vals).items():
        _add(out, ('h', '%s%d' % (Xn, d)), M)
    return {k: sp_csr(v) for k, v in out.items()}

# --- in a run: the coefficients (cached, by diagonal), the labels, the blocks and the wall rows
def diffusion_kinds():
    '''the dissipative terms of this run: 'visc' (ViscosD > 0, NEW 2026-11-09: and the flow, hydro = 1), 'heat' (thermal
    diffusion), 'mag' (a field with diffusion), NEW (2026-11-08) 'comp' (the composition with diffusion); none if the body
    is not deformed (deformed())'''
    if not deformed():   # NEW (2026-11-06)
        return ()
    return tuple(k for k, on in (('visc', par.hydro and par.ViscosD > 0), ('heat', par.thermal and par.ThermaD > 0),
                                 ('mag', par.magnetic and par.MagnetD > 0),
                                 ('comp', par.compositional and par.CompoD > 0)) if on)

_dcoef, _dband = {}, {}
def diffusion_coefficients(m):
    '''diffusion_build for this run (cached), also stored by diagonal'''
    if m not in _dcoef:
        out = diffusion_build(m, par.lmax + 16, diffusion_kinds())
        B = _bandwidth(out)
        if B > 2:
            sys.exit('centrifugal: the metric diffusion couples degrees %d apart, more than assemble.py places' % B)
        _dcoef[m] = out
        _dband[m] = {rc: {lab: angular.band(M, B) for lab, M in d.items()} for rc, d in out.items()}
    return _dcoef[m]

def diffusion_labels():
    '''[(label, 'P', 'T', 'S', 'F', 'G' or 'C')]: the radial operators of the metric diffusion and their columns, for
    submatrices.py'''
    if exact():   # NEW (2026-11-06): the exact spheroid's (section 8; step 2: 'mag'; NEW 2026-11-08: 'comp')
        return spheroid_labels(('visc', 'heat', 'mag', 'comp'))
    out = []
    for (row, X), d in diffusion_coefficients(par.m).items():
        out += [(lab, X) for lab in d]
    return out

def diffusion_orders():
    '''{section: the highest derivative in its metric diffusion}: v and g reach 3 for m ≠ 0 (P''' and F'''), and
    submatrices.py then builds them in the C^(3) basis'''
    out = {}
    blocks = diffusion_coefficients(par.m).items() if not exact() else \
        [(rc, d) for kind in ('visc', 'heat', 'mag', 'comp') if kind in spheroid_coefficients(par.m)
         for rc, d in spheroid_coefficients(par.m)[kind][0].items()]   # NEW (2026-11-06): the exact spheroid's
    for (row, X), d in blocks:
        for lab in d:
            out[row] = max(out.get(row, 0), int(lab[-1]))
    return out

def diffusion_block(l, section, component, offdiag, M):
    '''
    The block of the metric diffusion for the row degree l of section 'u', 'v' (the viscous force: ViscosD times the sign of
    operators.viscous_diffusion, which A gets with a minus), 'h' (the thermal diffusion: ThermaD, the sign of
    thermal_diffusion), 'f', 'g' (the magnetic diffusion: MagnetD, the A sign) or, NEW (2026-11-08), 'c' (the diffusion of the
    composition: CompoD, the sign of comp_diffusion), the column 'upol', 'utor', 'temp', 'bpol', 'btor' or 'comp' and the
    source degree l + offdiag: [Σ coefficient × radial operator / L'^3, the column block offset in units of N1], or [0, 0] if
    nothing couples this Δl.
    '''
    import utils as ut
    if exact():   # NEW (2026-11-06): the exact spheroid (section 8; step 2: 'mag' in f, g; NEW 2026-11-08: 'comp' in c)
        return spheroid_block({'h': 'heat', 'c': 'comp', 'f': 'mag', 'g': 'mag'}.get(section, 'visc'), l, section, component,
                              offdiag, M)
    X = {'upol': 'P', 'utor': 'T', 'temp': 'S', 'bpol': 'F', 'btor': 'G', 'comp': 'C'}[component]
    diffusion_coefficients(par.m)
    out = 0
    for lab, bnd in _dband[par.m].get((section, X), {}).items():
        v = _lookup(bnd, l, par.m, offdiag)
        if v != 0:
            out = out + v*M[lab]
    if np.isscalar(out):
        return [0, 0]
    lp, lt = ut.ell(par.m, par.lmax, par.symm)[:2]
    lF, lG = (ut.ell(par.m, par.lmax, ut.bsymm)[:2] if par.magnetic else (None, None))
    rows = {'u': lp, 'v': lt, 'h': lp, 'f': lF, 'g': lG, 'c': lp}[section]
    cols = {'P': lp, 'T': lt, 'S': lp, 'F': lF, 'G': lG, 'C': lp}[X]
    offd = int(np.where(cols == l + offdiag)[0][0] - np.where(rows == l)[0][0])
    scale = {'u': par.ViscosD, 'v': par.ViscosD, 'h': par.ThermaD, 'f': par.MagnetD, 'g': par.MagnetD, 'c': par.CompoD}[section]
    return [scale*out/(l*(l + 1.))**3, offd]

_dwall = {}
def wall_dissipative_rows(wall):
    '''wall_dissipative for this run at the wall 'outer' (r = rcmb) or 'inner' (r = ricb), cached, by diagonal:
    (B, {(row, source): bands})'''
    import utils as ut
    if not deformed():   # NEW (2026-11-06): no metric part on the walls of a sphere
        return 0, {}
    if wall not in _dwall:
        R = ut.rcmb if wall == 'outer' else par.ricb
        if exact():   # NEW (2026-11-06): the exact spheroid's rows (section 8)
            W = spheroid_walls(par.m, par.lmax + 12, R)
        else:
            W = wall_dissipative(par.m, par.lmax + 12, R, wall_values(R))
        B = _bandwidth({0: W})
        _dwall[wall] = (B, {k: angular.band(M, B) for k, M in W.items()})
    return _dwall[wall]

# --- a conducting inner core (NEW 2026-11-06): its metric diffusion and the electric field's row at the ICB
# Inside the core the field only diffuses, ∂ₜb̃ = -∇×(η_i M ∇×(M b̃)) with the uniform η_i = 1/(mu_i2o sigma_i2o), so its rows
# fic, gic get the metric terms of the fluid's rows f, g (diffusion_build) with η' = 0: the labels with the profile's own
# value (j = 0), their amplitudes taken from amplitude_series (the map is smooth across the ICB) and built on the core's
# Chebyshev series in submatrices.py. The core's centre is in its domain, so the rows gic, which have a term F/r for m ≠ 0,
# are r³ r̂⋅∇× there (utils.rpower_gic, as rpower_g in a full sphere), and in the C^(3) basis (F''').
# At the ICB the field b̃ is continuous (b is, and F, J are; the core has the fluid's permeability), so the rows that match
# it keep their form. The tangential electric field is continuous too; its toroidal part follows from the radial induction
# equation on both sides (u = 0 at the no-slip ICB), its consoidal part is the row of g that assemble.bc_b_cic writes for the
# spherical diffusion, R [S part of 𝐄]_o - η_i R [S part of 𝐄]_i = 0, which with 𝐄 = M ∇×(M b̃) = ∇×b̃ + δ𝐄 per unit η,
# δ𝐄 = (M - I)∇×b̃ + ∇×((M - I) b̃), gains R [S part of δ𝐄] on both sides (with F, F', F'' and G, G' there).
LABEL_IC = re.compile(r'([fg])(\d)([ENCQK])et0_D(\d)$')
LABEL_IC_SPH = re.compile(r'([fg])(\d)eta0_D(\d)$')   # NEW (2026-11-06, step 2): the exact spheroid's (no amplitude)

def ic_metric_on():
    '''the metric magnetic diffusion of a conducting inner core and its ICB row: centrifugal = 1, a field with diffusion, a
    conducting inner core, and a deformation (deformed())'''
    import utils as ut
    return bool(par.centrifugal == 1 and par.magnetic and par.MagnetD > 0 and ut.cic and deformed())

def ic_labels():
    '''[(label, row 'f' or 'g', column 'F' or 'G', power of r, amplitude, derivative order)]: the operators of the metric
    diffusion inside a conducting inner core, those of the fluid's rows with the profile's value (j = 0; η' = 0 inside);
    the power of r of the rows fic (2) and gic (utils.rpower_gic) minus rx'''
    import utils as ut
    out = []
    if exact():   # NEW (2026-11-06, step 2): the exact spheroid's 'f{rx}eta0_D{d}', 'g{rx}eta0_D{d}' (amplitude None: 1)
        for (row, X), d in spheroid_coefficients(par.m)['mag'][0].items():
            for lab in d:
                mt = LABEL_IC_SPH.match(lab)
                if mt is None:
                    sys.exit('centrifugal: the exact diffusion label %s has no counterpart in the inner core' % lab)
                out.append((lab, row, X, (2 if row == 'f' else ut.rpower_gic) - int(mt.group(2)), None, int(mt.group(3))))
        return out
    for (row, X), d in diffusion_coefficients(par.m).items():
        if row not in 'fg':
            continue
        for lab in d:
            mt = LABEL_IC.match(lab)
            if mt is None:
                if not re.match(r'[fg]\d[ENCQK]et[1-9]_D\d$', lab):   # η' terms vanish in the core: any other label is unexpected
                    sys.exit('centrifugal: the metric diffusion label %s has no counterpart in the inner core' % lab)
                continue
            rp = (2 if row == 'f' else ut.rpower_gic) - int(mt.group(2))
            out.append((lab, row, X, rp, mt.group(3), int(mt.group(4))))
    return out

def diffusion_block_ic(l, section, component, offdiag, M):
    '''
    The block of the metric diffusion inside a conducting inner core for the row degree l of section 'fic' or 'gic', the
    column 'bpol_ic' or 'btor_ic' and the source degree l + offdiag (the A sign, as diffusion_block): [MagnetD η_i Σ
    coefficient × the core's operator (label + '_ic') / L'^3, the column block offset in units of Nic], or [0, 0].
    '''
    import utils as ut
    row = {'fic': 'f', 'gic': 'g'}[section]
    X = {'bpol_ic': 'F', 'btor_ic': 'G'}[component]
    if exact():   # NEW (2026-11-06, step 2): the exact spheroid's (section 8)
        bands = spheroid_coefficients(par.m)['mag'][1].get((row, X), {})
    else:
        diffusion_coefficients(par.m)
        bands = {lab: bnd for lab, bnd in _dband[par.m].get((row, X), {}).items() if LABEL_IC.match(lab) is not None}
    out = 0
    for lab, bnd in bands.items():
        v = _lookup(bnd, l, par.m, offdiag)
        if v != 0:
            out = out + v*M[lab + '_ic']
    if np.isscalar(out):
        return [0, 0]
    lF, lG = ut.ell(par.m, par.lmax, ut.bsymm)[:2]
    rows, cols = {'f': lF, 'g': lG}[row], {'F': lF, 'G': lG}[X]
    offd = int(np.where(cols == l + offdiag)[0][0] - np.where(rows == l)[0][0])
    return [par.MagnetD*out/(par.mu_i2o*par.sigma_i2o)/(l*(l + 1.))**3, offd]

def icb_electric(m, lw, R, vals):
    '''
    The metric part of the ICB row of the electric field: {(X, d): sparse matrix [l, l']}, R times the S part of δ𝐄 per unit
    η at r = R (the formula above) for the column X ('F' or 'G') of degree l', its d-th derivative there; vals as
    wall_values. Also used by tests/centrifugal/derive_diff.py without a run.
    '''
    A = angular.Algebra.get(m, lw)
    out = {}
    for X in 'FG':
        b = _field(A, X)
        E = _sum(_metric(A, _curl(A, b), lw), _curl(A, _metric(A, b, lw)))
        for (Xn, d), F in _at(E, R, vals).items():
            _add(out, (Xn, d), R*F.s)
    return {k: sp_csr(v) for k, v in out.items()}

_icb = {}
def icb_rows():
    '''icb_electric for this run at r = ricb (the amplitudes there from the fluid's profile codes; the map is smooth across),
    cached, by diagonal: (B, {(X, d): bands})'''
    if not _icb:
        if exact():   # NEW (2026-11-06, step 2): the exact spheroid's (section 8)
            W = spheroid_icb(par.m, par.lmax + 12, par.ricb)
        else:
            W = icb_electric(par.m, par.lmax + 12, par.ricb, wall_values(par.ricb))
        B = _bandwidth({0: W})
        _icb.update({'B': B, 'W': {k: angular.band(M, B) for k, M in W.items()}})
    return _icb['B'], _icb['W']


# ======================================================================================================================
# 7. The differential rotation (NEW 2026-11-06, phase 2)
# ======================================================================================================================
# The background flow of the body is U0 = G ΔΩ ẑ×X (G = Gaspard), with diff_rot's ΔΩ(r, θ) given in the reference space as
# every other profile: the body point X(r, θ) turns at ΔΩ(r, θ) (a ΔΩ of r alone is constant on the level surfaces). The
# map commutes with the rotations about the axis, F⁻¹(ẑ×X) = ẑ×α, so the flow's flux is w0 = J F⁻¹ U0 = J U00,
# U00 = G ΔΩ ẑ×α its spherical formula. Pulled back (tests/centrifugal/derive_diff.py,
# part A5):
#   momentum: the force W(u) = (U0⋅∇)u + (u⋅∇)U0 of diff_rot.py is, in covariant form and up to a gradient,
#       W(w) + δW,   δW = -U00×∇×((M - I) w) + (ε + η P2) w×ω0 - η G ΔΩ (ẑ×w - x r̂×w),   ω0 = ∇×U00
#   (from -[w0×∇×(M w) + w×∇×(M w0)]/J, with (J M - I) U00 = -ε U00 for an azimuthal flow);
#   heat (its rows are the equation times J): J U0⋅∇_X s = (1 + η P2) U00⋅∇s, the metric part i m G η P2 ΔΩ s;
#   induction: the electric field of the flow pulls back exactly, Fᵀ(U0×b) = U00×b̃, with no metric part.
# With ΔΩ = Σ_n f_n(r) P_n(x) (diff_rot.fX: f_n with its amplitudes), ω0 = G [(2 f_n + r f_n') P_n ẑ - (r f_n' x P_n
# + f_n (1 - x²) P_n') r̂]. The terms are built per degree n with the algebra of section 6, f_n in the profile slot: product
# codes a + 'z' + the degree's character (rot_code). W(u) is algebraic in u, so the rows keep the orders of diff_rot's (u: P'',
# T'; v: P', T), the higher derivatives of the curl form cancelling to round-off (_check_orders). They couple l to l ± (n + 2).
MAXD_ROT = {'u': 2, 'v': 1}

def rot_code(a, n):
    '''the profile code of the amplitude a (AMPS) times f_n, the degree-n shape of ΔΩ'''
    import diff_rot as dr
    return a + 'z' + dr.DIGITS[n]

def rotation_on():
    '''the metric terms of the differential rotation: centrifugal = 1 with diff_rot = 1, and a deformation (deformed())'''
    return bool(par.centrifugal == 1 and par.diff_rot and deformed())

def rotation_profile_functions():
    '''{product code: f(r, Dorder)} for radial_profiles.burrito: a(r) f_n^(Dorder)(r), the amplitude a (profile_functions)
    and diff_rot.fX'''
    import diff_rot as dr
    amp = profile_functions()
    out = {}
    for n in dr.degrees():
        for a, (ca, _) in AMPS.items():
            out[rot_code(a, n)] = lambda r, Dorder, A=amp[ca], n=n: A(r, 0)*dr.fX(n, r, Dorder)
    return out

def _legendre(n):
    '''P_n as a numpy Legendre series in x (FIX 2026-11-08, phase 6e: not a Polynomial from leg2poly, whose coefficients
    grow as about (1 + √2)^n: see mag_algebra._poly_mats)'''
    return np.polynomial.Legendre.basis(n)

def rotation_force(A, X, n, lw):
    '''δW of the flow of the unknown X ('P' or 'T') for ΔΩ = c(r) P_n(x), per unit G (c in the profile slot), up to a
    gradient (the formula above), as a vector expression'''
    w = _flow(A, X)
    Pn = _legendre(n)
    c, rc1 = _rf((1., 0, None, 0, ())), _rf((1., 1, None, 1, ()))          # c, r c'
    v = _curl(A, _metric(A, w, lw))                                        # ∇×((M - I) w)
    # -U00×v = -c r P_n (v_z r̂ - v_r ẑ)
    t1 = _rmul(_rf((-1., 1, None, 0, ())), _poly(A, _sum(_vec(A, q=_zdot(A, v)), _sc(_zhat(A, _parts(v)[0]), -1.)), Pn))
    # w×ω0 = -g1 ẑ×w - g2 r̂×w,  g1 = (2c + r c') P_n,  g2 = -(r c') x P_n - c (1 - x²) P_n'
    zw, rw = _zcr(A, w), _rcr(w)
    wxo = _sum(_sc(_rmul(c, _poly(A, zw, Pn)), -2.), _sc(_rmul(rc1, _poly(A, zw, Pn)), -1.),
               _rmul(rc1, _poly(A, rw, _xp*Pn)), _rmul(c, _poly(A, rw, (1 - _xp*_xp)*Pn.deriv())))
    t2 = _sum(_rmul(E1, wxo), _rmul(N1, _poly(A, wxo, _P2p)))              # (ε + η P2) w×ω0
    t3 = _sc(_rmul(N1, _rmul(c, _poly(A, _sum(zw, _sc(_poly(A, rw, _xp), -1.)), Pn))), -1.)   # -η c P_n (ẑ×w - x r̂×w)
    return _sum(t1, t2, t3)

def rotation_build(m, lw, degrees):
    '''
    {(row, column): {label: sparse matrix [target l', source l]}} of δW in the rows u, v (columns P, T) for the degrees n of
    ΔΩ (the sign of diff_rot.momentum_block, without Gaspard and the 1/L'^3), pruned and checked (MAXD_ROT). Also used by
    tests/centrifugal/derive_diff.py without a run.
    '''
    import mag_algebra as ma, diff_rot as dr
    A = angular.Algebra.get(m, lw)
    Lt = lambda M: A.L @ M
    out = {}
    for n in degrees:
        for X in 'PT':
            W = rotation_force(A, X, n, lw)
            _store2(out, 'u', X, _row_u(W), Lt, 'z' + dr.DIGITS[n])
            _store2(out, 'v', X, _row_v(W), Lt, 'z' + dr.DIGITS[n])
    out = ma._prune(out)
    for (row, X), d in out.items():
        for lab in d:
            if int(lab[-1]) > MAXD_ROT[row]:
                sys.exit('centrifugal: the metric term %s of the differential rotation (column %s) is above the order of '
                         'its row: the cancellation failed' % (lab, X))
    return out

_rcoef, _rband = {}, {}
def rotation_coefficients(m):
    '''rotation_build for this run (cached), also stored by diagonal'''
    import diff_rot as dr
    if m not in _rcoef:
        out = rotation_build(m, par.lmax + dr.n_max() + 10, dr.degrees())
        B = _bandwidth(out)
        if B > dr.n_max() + 3:
            sys.exit('centrifugal: the metric terms of the differential rotation couple degrees %d apart, more than '
                     'assemble.py places' % B)
        _rcoef[m] = out
        _rband[m] = {rc: {lab: angular.band(M, B) for lab, M in d.items()} for rc, d in out.items()}
    return _rcoef[m]

def rotation_heat_label(n, section='h'):
    '''the radial operator of η f_n s in the heat rows: that of diff_rot's i m G ΔΩ s with the profile η f_n; NEW (2026-11-08)
    section 'c', that of η f_n c in the rows of the composition'''
    if section == 'c':
        return ('c0dns0%s0_D0' if par.CompoD > 0 else 'c0%s0_D0') % rot_code('N', n)
    return ('h0pss0%s0_D0' if par.ThermaD > 0 else 'h0%s0_D0') % rot_code('N', n)

_rheat = {}
def rotation_heat_coefficients(m):
    '''(P2 P_n)(X) by diagonal for every degree n of ΔΩ'''
    import diff_rot as dr
    if m not in _rheat:
        A = angular.Algebra.get(m, par.lmax + dr.n_max() + 10)
        P2 = A.legendre(2)[0]
        _rheat[m] = {n: angular.band(P2 @ A.legendre(n)[0], n + 2) for n in dr.degrees()}
    return _rheat[m]

def rotation_labels():
    '''[(label, 'P', 'T', 'S' or 'C')]: the radial operators of the metric terms of the differential rotation and their
    columns, for submatrices.py'''
    import diff_rot as dr
    out = []
    if not rotation_on():   # NEW (2026-11-06): none without a deformation
        return out
    for (row, X), d in rotation_coefficients(par.m).items():
        out += [(lab, X) for lab in d]
    if par.thermal:
        out += [(rotation_heat_label(n), 'S') for n in dr.degrees()]
    if par.compositional:   # NEW (2026-11-08)
        out += [(rotation_heat_label(n, 'c'), 'C') for n in dr.degrees()]
    return out

def rotation_dl_range(section, component):
    '''the Δl of the blocks: 0, ±2, ..., ±(n_max + 2) within a kind (u and P, v and T, h and s, c and the composition),
    ±1, ..., ±(n_max + 3) across'''
    import diff_rot as dr
    nd = dr.n_max() + 2
    same = (section in ('h', 'c')) or ((section == 'u') == (component == 'upol'))
    return range(-nd, nd + 1, 2) if same else range(-nd - 1, nd + 2, 2)

def rotation_block(l, section, component, offdiag, M):
    '''
    The block of the metric terms of the differential rotation for the row degree l and the source degree l + offdiag:
    section 'u', 'v' (column 'upol', 'utor'): δW with the sign of diff_rot.momentum_block (A gets it with +), Gaspard Σ
    coefficient × radial operator / L'^3; section 'h' (column 'temp'): i m G η P2 ΔΩ s with the sign of diff_rot.heat_block;
    NEW (2026-11-08) section 'c' (column 'comp'): i m G η P2 ΔΩ c with the sign of diff_rot.comp_block. Returns [matrix, offd]
    or [0, 0].
    '''
    import utils as ut, diff_rot as dr
    m = par.m
    if offdiag not in rotation_dl_range(section, component) or not rotation_on():   # NEW (2026-11-06): rotation_on
        return [0, 0]
    L = l*(l + 1.)
    out = 0
    if section in ('h', 'c'):
        for n, bnd in rotation_heat_coefficients(m).items():
            v = _lookup(bnd, l, m, offdiag)
            if v != 0:
                out = out + 1j*m*v*M[rotation_heat_label(n, section)]
        if np.isscalar(out):
            return [0, 0]
        lp = ut.ell(par.m, par.lmax, par.symm)[0]
        offd = int(np.where(lp == l + offdiag)[0][0] - np.where(lp == l)[0][0])
        return [-par.Gaspard*out/L**3, offd]
    X = 'P' if component == 'upol' else 'T'
    rotation_coefficients(m)
    for lab, bnd in _rband[m].get((section, X), {}).items():
        v = _lookup(bnd, l, m, offdiag)
        if v != 0:
            out = out + v*M[lab]
    if np.isscalar(out):
        return [0, 0]
    offd = ut.dr_offd(l, l + offdiag, section, X)
    return [par.Gaspard*out/L**3, offd]


# ======================================================================================================================
# 8. The exact spheroid (NEW 2026-11-06, centrifugal_shape 'spheroid')
# ======================================================================================================================
# A body whose level surfaces all have one flattening f is the image of the reference space under a linear map, X = F α with
# F = diag(a, a, c), c/a = 1 - f and a² c = 1 (the level surface r is the spheroid of the volume of the sphere r): the stretch
# of section 1 kept exactly (at first order in f, a = 1 + f/3 and c = 1 - 2f/3, the map of 'uniform'). Its metric is
# constant, J = 1, M = FᵀF = Q = diag(a², a², c²) and z̃ = ẑ/c, and the Laplacian of the body is, in the reference space,
# D = ∇⋅(Q⁻¹∇) = ∇²/a² + (1/c² - 1/a²) ∂z². So the terms of sections 1 and 6 hold at any f, with no expansion in it
# (Z v = ẑ (ẑ⋅v), (ẑ⋅∇)v = ∇(ẑ⋅v) - ẑ×(∇×v), ∂z² v = (ẑ⋅∇)²v, x = cos θ):
#     inertia        (M - I) w = (a² - 1) w + (c² - a²) Z w
#     Coriolis       (z̃ - ẑ)×w = (1/c - 1) ẑ×w
#     viscous force  Fᵀ ν∇_X²(F w) - ν∇²w = ν [(a²/c² - 1) ∂z² w + (c² - a²) Z D w]                   (ν uniform)
#     heat           J ∇_X⋅(κp ∇_X s) - ∇⋅(κp ∇s) = κp [(1/a² - 1) ∇²s + (1/c² - 1/a²) ∂z² s]            (κ p uniform)
#                    (NEW 2026-11-08: and the same for the composition c, with D ρ0, its flux rows as those of s)
#     stress-free    the traction Fᵀσ_X F⁻ᵀ r̂ = Q (∇w) Q⁻¹ r̂ + (∇w)ᵀ r̂ ((∇w)_ij = ∂_j w_i) less the sphere's r̂⋅σ(w):
#                    Q (∇w) Q⁻¹ r̂ - ∂ᵣw = (a²/c² - 1) x (ẑ⋅∇)w + (c² - a²) Z [∂ᵣw/a² + (1/c² - 1/a²) x (ẑ⋅∇)w]
#     fixed flux     r̂⋅Q⁻¹∇s - ∂ᵣs = (1/a² - 1) ∂ᵣs + (1/c² - 1/a²) x ∂z s
# The time derivative of s and its advection keep their spherical form (J = 1, w⋅∇S0), and so does the buoyancy: with
# kore's g = r it acts along F⁻ᵀ𝐫 = ∇_X(r²/2) in the body, normal to the level surfaces, as the effective gravity of a
# homogeneous body in hydrostatic equilibrium (a Maclaurin spheroid: its effective potential depends on r alone). The
# couplings are built with the algebra of section 6 on expressions of constant coefficients and take kore's own labels (no
# profile, or the uniform ν, κ p, η and D ρ0: 'vsc0', 'kps0', 'eta0', 'Drh0'). The products of Z and ∂z² couple a degree l
# to l ± 4 of the same kind and to l ± 3 of the other. The fluid is homogeneous (structure 'Boussinesq', diffusivity 'uniform'): its
# expressions have no (ln ρ0) terms, and no profile derivatives. Checks: tests/centrifugal/spheroid_check.py.
# NEW (2026-11-06, step 2) the field. b̃ = J F⁻¹ b = F⁻¹ b and the background B̃0 (kore's formula; F B̃0 in the body) are fluxes
# of the reference space (section 4), so with J = 1 and M = Q the induction ∂ₜb̃ = ∇×(w × B̃0) is that of the sphere, and
#     Lorentz force  ∇×(Q b̃) × B̃0 + ∇×(Q B̃0) × b̃ less the sphere's: (a² - 1)(J0×b - B0×J) + δ {[∇×(Z B0)]×b - B0×[∇×(Z b)]}
#     diffusion      the field of ∂ₜb̃ = -∇×(η Q ∇×(Q b̃)) less the sphere's, per unit η:
#                    (a⁴ - 1) ∇×b̃ + a² δ [Z ∇×b̃ + ∇×(Z b̃)] + δ² Z ∇×(Z b̃)
# (δ = c² - a², Q - I = (a² - 1) I + δ Z; b, B0, J = ∇×b and J0 = ∇×B0 of the reference space). The force takes mag_algebra's
# labels (the profiles hp, kt of B0), the diffusion 'eta0'; through B0 of degree n the force couples l to l ± (n + 3).
# The walls are the level surfaces r = R, spheroids of semi-axes a R, c R. In the spheroidal coordinates confocal with the
# wall, s = k R √(1 + ν²) √(1 - x²), z = k R ν x outside an oblate body (k² = a² - c², ν ≥ 0; s = k R √(ν² - 1) √(1 - x²),
# z = k R ν x for a prolate one, k² = c² - a², ν ≥ 1), the wall is ν0 = c/k with x = cos θ: on it the spheroidal angle is
# the reference colatitude. So the potential field outside, φ = Σ c_l Y_l R_l(ν)/R_l(ν0) with R_l = Q_l^m(iν) (Q_l^m(ν)
# prolate; P_l^m inside an insulating inner core, regular there), is Φ = Σ c_l Y_l on the wall, and its flux per unit
# reference area is b̃_r = (a²/(k R)) ∂ν φ (that of ∇φ through the surface ν0 per dx dφ is k R (1 ± ν0²) ∂ν φ, and
# 1 ± ν0² = a²/k²). With the tangential field Fᵀb = Q b̃ = ∇_tan Φ:
#     q_l(b̃) = λ_l s_l(Q b̃),  t_l(Q b̃) = 0  at r = R,     λ_l = (a²/k) d ln R_l(ν0)/dν
# (-(l + 1) and l in the sphere), that is, with m_s, m_t the S and T parts of (Q - I) b̃ at R (q = L F/R, s = F' + F/R, t = G),
#     outside  (l - μ) F - μ R (F' + m_s) = 0,  μ = λ_l/(l + 1)      ((l + 1) F + R F' for μ = -1)
#     inside   (l + 1 - ν) F - ν R (F' + m_s) = 0,  ν = λ_l/l        (l F - R F' for ν = 1)
#     and G + m_t = 0.
# At a conducting inner core (mu_i2o = 1) b̃ is continuous, and the row of the tangential electric field gains the metric
# part of the diffusion's field (section 6, with the formula above); the core's diffusion takes the fluid's terms with η_i.
SPHEROID_KINDS = ('inertia', 'coriolis', 'visc', 'heat', 'mag')   # (NEW 2026-11-06, step 2: 'mag')

def spheroid_axes(f=None):
    '''(a, c), the equatorial and polar radii of the level surface r = 1 of the spheroid of flattening f (centrifugal_f):
    c/a = 1 - f and a² c = 1'''
    f = par.centrifugal_f if f is None else f
    return (1 - f)**(-1/3), (1 - f)**(2/3)

def exact():
    '''the exact spheroid: centrifugal = 1 with centrifugal_shape 'spheroid' and a deformation (deformed())'''
    return bool(par.centrifugal == 1 and par.centrifugal_shape == 'spheroid' and deformed())

def _hom(e):
    '''an expression without its (ln ρ0) terms: that of a homogeneous fluid (λ_k = 0)'''
    return {k: M for k, M in e.items() if not k[5]}

def _zgrad(A, e):
    '''(ẑ⋅∇)v = ∇(ẑ⋅v) - ẑ×(∇×v) of a vector expression'''
    return _sum(_grad(A, _zdot(A, e)), _sc(_zcr(A, _curl(A, e)), -1.))

def _Z(A, e):
    '''Z v = ẑ (ẑ⋅v)'''
    return _zhat(A, _zdot(A, e))

def _sdz(A, g):
    '''∂z g = ẑ⋅∇g of a scalar expression'''
    return _zdot(A, _grad(A, g))

def spheroid_terms(A, X, kind, f=None):
    '''
    The exact metric terms of the unknown X ('P', 'T': the flow w; 'S': the entropy s) on the algebra A, as expressions
    (the formulas above, flattening f or centrifugal_f): kind 'inertia' (M - I) w, 'coriolis' (z̃ - ẑ)×w, 'visc' the viscous
    force per unit ν, 'traction' the stress-free wall rows per unit ν (vector expressions); 'heat' the diffusion per unit
    κp, 'flux' the fixed-flux wall rows (scalar expressions, X = 'S'; NEW 2026-11-08, or 'C', the composition, per unit D ρ0);
    NEW (2026-11-06, step 2) X = 'F', 'G' (the field b̃), kind 'mag': the diffusion's field per unit η in kore's sign (minus
    the formula above, as magnetic_diffusion_metric)
    '''
    a, c = spheroid_axes(f)
    a2, c2 = a*a, c*c
    if X in ('S', 'C'):
        s = {(0, X, 0, None, None, ()): A.I}
        if kind == 'heat':
            return _sum(_sc(_div(A, _grad(A, s)), 1/a2 - 1), _sc(_sdz(A, _sdz(A, s)), 1/c2 - 1/a2))
        if kind == 'flux':
            return _sum(_sc(_dr(s), 1/a2 - 1), _sc(_map(_sdz(A, s), lambda M: A.X @ M), 1/c2 - 1/a2))
    elif X in 'FG':   # NEW (2026-11-06, step 2)
        b = _field(A, X)
        if kind == 'mag':
            j, cz = _curl(A, b), _curl(A, _Z(A, b))
            return _sc(_sum(_sc(j, a2*a2 - 1), _sc(_sum(_Z(A, j), cz), a2*(c2 - a2)), _sc(_Z(A, cz), (c2 - a2)**2)), -1.)
    else:
        w = _hom(_flow(A, X))
        if kind == 'inertia':
            return _sum(_sc(w, a2 - 1), _sc(_Z(A, w), c2 - a2))
        if kind == 'coriolis':
            return _sc(_zcr(A, w), 1/c - 1)
        if kind == 'visc':
            dz = _zgrad(A, _zgrad(A, w))
            Dw = _sum(_sc(_curl(A, _curl(A, w)), -1/a2), _sc(dz, 1/c2 - 1/a2))    # D w (∇²w = -∇×∇×w, ∇⋅w = 0)
            return _sum(_sc(dz, a2/c2 - 1), _sc(_Z(A, Dw), c2 - a2))
        if kind == 'traction':
            xz = _poly(A, _zgrad(A, w), _xp)
            return _sum(_sc(xz, a2/c2 - 1), _sc(_Z(A, _sum(_sc(_dr(w), 1/a2), _sc(xz, 1/c2 - 1/a2))), c2 - a2))
    sys.exit('centrifugal: no exact term %r of %r' % (kind, X))

def _store_sph(out, row, X, e, Lt, code):
    '''the row's monomials (constant coefficients: no amplitude, profile or λ) as kore's own labels
    '{row}{rx}[{code}0]_D{d}', times Lt (L' of the target, or 1 for the heat rows), added to out[(row, X)]'''
    d_ = out.setdefault((row, X), {})
    for (p, Xn, dd, a, j, lam), M in e.items():
        M = M.tocsr()
        if M.nnz == 0 or abs(M).max() == 0:
            continue
        rx = ROWPOW2[row] - p
        if not (0 <= rx <= 9 and a is None and j is None and not lam and Xn == X and dd <= 9):
            sys.exit('centrifugal: the exact term r^%d %s^(%d) has no label in row %s' % (p, Xn, dd, row))
        _add(d_, '%s%d%s_D%d' % (row, rx, '' if code is None else code + '0', dd), Lt(M).tocsr())

def spheroid_build(m, lw, kinds=SPHEROID_KINDS, f=None):
    '''
    {kind: {(row, column): {label: sparse matrix [target l', source l]}}} of the exact metric terms on the degrees 0..lw:
    'inertia' and 'coriolis' in the rows u, v (columns P, T; (M - I) w and (z̃ - ẑ)×w as section 3's momentum_block, without
    2 Gaspard and the 1/L'^3), 'visc' in u, v (the sign of operators.viscous_diffusion, without ViscosD and the 1/L'^3) and
    'heat' in h (column S; the sign of operators.thermal_diffusion, without ThermaD and the 1/L^3), NEW (2026-11-06, step 2)
    'mag' in f, g (columns F, G; kore's A sign, as diffusion_build, without MagnetD and the 1/L'^3), NEW (2026-11-08) 'comp'
    in c (column C; the sign of operators.comp_diffusion, without CompoD and the 1/L^3); pruned and checked (_check_orders).
    Also used by tests/centrifugal/spheroid_check.py without a run.
    '''
    import mag_algebra as ma
    A = angular.Algebra.get(m, lw)
    Lt = lambda M: A.L @ M
    out = {}
    for kind in kinds:
        o = {}
        if kind == 'heat':
            _store_sph(o, 'h', 'S', _shift(spheroid_terms(A, 'S', 'heat', f), 1., 2), lambda M: M, 'kps')   # r² (the row)
        elif kind == 'comp':   # NEW (2026-11-08)
            _store_sph(o, 'c', 'C', _shift(spheroid_terms(A, 'C', 'heat', f), 1., 2), lambda M: M, 'Drh')  # r² (the row)
        elif kind == 'mag':   # NEW (2026-11-06, step 2): f: -r² t, g: -r [q - (r s)'] (as diffusion_build)
            for X in 'FG':
                q, s, t = _parts(spheroid_terms(A, X, 'mag', f))
                _store_sph(o, 'f', X, _shift(t, -1., 2), Lt, 'eta')
                _store_sph(o, 'g', X, _shift(_sum(q, _sc(_dr(_shift(s, 1., 1)), -1.)), -1., 1), Lt, 'eta')
        else:
            for X in 'PT':
                e = spheroid_terms(A, X, kind, f)
                code = 'vsc' if kind == 'visc' else None
                _store_sph(o, 'u', X, _row_u(e), Lt, code)
                _store_sph(o, 'v', X, _row_v(e), Lt, code)
        out[kind] = ma._prune(o)
        _check_orders(out[kind])
    return out

def spheroid_walls(m, lw, R, f=None):
    '''the exact metric parts of the stress-free and fixed-flux wall rows at the radius R, as wall_dissipative: {(row,
    source): sparse matrix [l, l']}, rows 'u' (R² times the S part of the traction), 'v' (R times its T part) and 'h' (the
    flux), sources 'P0', 'P1', 'P2', 'T0', 'T1', 'S0', 'S1'. Also used by tests/centrifugal/spheroid_check.py.'''
    A = angular.Algebra.get(m, lw)
    out = {}
    for X in 'PT':
        for (Xn, d), F in _at(spheroid_terms(A, X, 'traction', f), R, {}).items():
            _add(out, ('u', '%s%d' % (Xn, d)), R**2*F.s)
            _add(out, ('v', '%s%d' % (Xn, d)), R*F.t)
    for (Xn, d), M in _at(spheroid_terms(A, 'S', 'flux', f), R, {}).items():
        _add(out, ('h', '%s%d' % (Xn, d)), M)
    for (row, src) in out:
        if src not in ('P0', 'P1', 'P2', 'T0', 'T1', 'S0', 'S1'):
            sys.exit('centrifugal: the exact wall row %s has a term in %s, which assemble.py does not place' % (row, src))
    return {k: sp_csr(v) for k, v in out.items()}

_sph = {}
def spheroid_coefficients(m):
    '''spheroid_build for this run (cached): {kind: (out, bands)} with the kinds of this run ('visc' with ViscosD > 0 and,
    NEW 2026-11-09, the flow, 'heat'
    with thermal diffusion, NEW (2026-11-06, step 2) 'mag' with a field and MagnetD > 0, NEW (2026-11-08) 'comp' with the
    composition and CompoD > 0), the bands by diagonal (angular.band, |Δl| <= 4)'''
    if m not in _sph:
        kinds = ['inertia', 'coriolis'] + (['visc'] if par.hydro and par.ViscosD > 0 else []) + \
                (['heat'] if par.thermal and par.ThermaD > 0 else []) + (['mag'] if par.magnetic and par.MagnetD > 0 else []) + \
                (['comp'] if par.compositional and par.CompoD > 0 else [])
        _sph[m] = {}
        for kind, out in spheroid_build(m, par.lmax + 16, kinds).items():
            B = _bandwidth(out)
            if B > 4:
                sys.exit('centrifugal: the exact term %s couples degrees %d apart, more than assemble.py places' % (kind, B))
            _sph[m][kind] = (out, {rc: {lab: angular.band(M, 4) for lab, M in d.items()} for rc, d in out.items()})
    return _sph[m]

def spheroid_labels(kinds):
    '''[(label, 'P', 'T', 'S', 'F', 'G' or 'C')]: the radial operators of the exact terms of the kinds given (those of this
    run), for submatrices.py'''
    out = []
    for kind, (o, _) in spheroid_coefficients(par.m).items():
        if kind in kinds:
            for (row, X), d in o.items():
                out += [(lab, X) for lab in d]
    return out

def spheroid_block(kind, l, section, component, offdiag, M):
    '''
    The block of the exact metric term kind ('inertia', 'coriolis': momentum_block's conventions; 'visc', 'heat', 'mag',
    'comp': diffusion_block's) for the row degree l of section 'u', 'v', 'h' (NEW 2026-11-06, step 2: or 'f', 'g'; NEW
    2026-11-08: or 'c'), the column 'upol', 'utor', 'temp' (or 'bpol', 'btor', 'comp') and the source degree l + offdiag:
    [factor Σ coefficient × radial operator / L'^3, the column block offset in units of N1] (factor 1, 2 Gaspard, ViscosD,
    ThermaD, MagnetD or CompoD), or [0, 0] if nothing couples this Δl.
    '''
    import utils as ut
    C = spheroid_coefficients(par.m)
    if kind not in C:
        return [0, 0]
    X = {'upol': 'P', 'utor': 'T', 'temp': 'S', 'bpol': 'F', 'btor': 'G', 'comp': 'C'}[component]
    out = 0
    for lab, bnd in C[kind][1].get((section, X), {}).items():
        v = _lookup(bnd, l, par.m, offdiag)
        if v != 0:
            out = out + v*M[lab]
    if np.isscalar(out):
        return [0, 0]
    if section in 'fg':   # NEW (2026-11-06, step 2): the field's rows and columns (as diffusion_block)
        lF, lG = ut.ell(par.m, par.lmax, ut.bsymm)[:2]
        rows, cols = {'f': lF, 'g': lG}[section], {'F': lF, 'G': lG}[X]
        offd = int(np.where(cols == l + offdiag)[0][0] - np.where(rows == l)[0][0])
    else:
        offd = ut.dr_offd(l, l + offdiag, section, X)
    fac = {'inertia': 1., 'coriolis': 2*par.Gaspard, 'visc': par.ViscosD, 'heat': par.ThermaD, 'mag': par.MagnetD,
           'comp': par.CompoD}[kind]
    return [fac*out/(l*(l + 1.))**3, offd]

def spheroid_primitives(kind, m, lw, f=None):
    '''
    The primitive matrices C[k][shape] of the exact inertia and Coriolis terms on the degrees 0..lw, as primitives()
    (k = 'q', 's', 't' the components of the result, shapes 'q', 'a', 'T' those of w): kind 'inertia' (M - I) w = (a² - 1) w
    + (c² - a²) Z w, kind 'coriolis' (z̃ - ẑ)×w = (1/c - 1) ẑ×w, with no radial factor (for the budgets:
    utils4pp.centrifugal_terms). Returns (C, A).
    '''
    A = angular.Algebra.get(m, lw)
    a, c = spheroid_axes(f)
    zdot, zhat, rhat, rcross = _ops(A)
    C = {'q': {}, 's': {}, 't': {}}
    for sh, w in {'q': A.field(q=A.I), 'a': A.field(s=A.I), 'T': A.field(t=A.I)}.items():
        W = (a*a - 1)*w + (c*c - a*a)*zhat(zdot(w)) if kind == 'inertia' else (1/c - 1)*A.zcross(w)
        for k in 'qst':
            C[k][sh] = W[k].tocsr()
    return C, A


# --- the field (NEW 2026-11-06, step 2): the Lorentz force, the insulating walls, the ICB row of a conducting inner core
def _zvecs(vecs):
    '''Z V = ẑ (ẑ⋅V) of axisymmetric background vectors (section 4's kinds): V_z = x V_r - sinθ V_θ, ẑ = x r̂ - sinθ θ̂'''
    x = _xp
    out = []
    for kind, c, rad in vecs:
        if kind != 'sp':
            Vz = x*c if kind == 'r' else -1*(1 - x*x)*c
            out += [('r', x*Vz, rad), ('st', -1*Vz, rad)]
    return out

def _store_ma(out, row, X, e, Lt):
    '''the row's monomials (constant coefficients: no amplitude, homogeneous fluid) as mag_algebra's labels
    '{row}{rx}{hp|kt}{n}{j}ohr1_D{d}' ('ohr1' in the rows u, v), times L' of the target, added to out[(row, X)]'''
    import mag_algebra as ma
    d_ = out.setdefault((row, X), {})
    for (p, Xn, dd, B, n, j, lam, a), M in e.items():
        M = M.tocsr()
        if M.nnz == 0 or abs(M).max() == 0:
            continue
        rx = ma.ROWPOW[row] - p
        if not (0 <= rx <= 9 and B is not None and a is None and not lam and j <= 9 and dd <= 9 and Xn == X):
            sys.exit('centrifugal: the exact term r^%d %s^(%d) %s^(%s) has no label in row %s' % (p, Xn, dd, B, j, row))
        _add(d_, '%s%d%s%d%s_D%d' % (row, rx, ma.code(B, n), j, 'ohr1' if row in 'uv' else '', dd), Lt(M).tocsr())

def spheroid_magnetic_build(m, degs, lw, f=None):
    '''
    {(row, column): {label: sparse matrix [target l', source l]}}: the exact metric part of the Lorentz force (the formula of
    the header) in the rows 'u', 'v' (columns 'F', 'G') for the background degrees degs = {'h': [...], 'k': [...]} on the
    degrees 0..lw, with mag_algebra's labels (the coefficients are constant), in kore's A sign, without Hendrik and the
    1/L'^3 of the row (as magnetic_build; the induction has no metric part, J = 1). Also used by
    tests/centrifugal/spheroid_check.py without a run.
    '''
    import mag_algebra as ma
    A = angular.Algebra.get(m, lw)
    a, c = spheroid_axes(f)
    k1, dl = a*a - 1, c*c - a*a
    zdot, zhat, _, _ = _ops(A)
    Lt = lambda M: A.L @ M
    out = {}
    for B in 'hk':
        for n in degs.get(B, []):
            bg = _background(n, B)
            jz = _curl_vectors(_zvecs(bg['B0']))                          # ∇×(Z B0)
            for X in 'FG':
                b, J = _pert(A, X, 'b'), _pert(A, X, 'J')
                zb = {key: zhat(zdot(F)) for key, F in b.items()}         # Z b
                G = _product(A, bg['J0'], b, k1)                           # (a² - 1)(J0×b - B0×J)
                _merge(G, _product(A, bg['B0'], J, -k1))
                _merge(G, _product(A, jz, b, dl))                          # δ {[∇×(Z B0)]×b - B0×[∇×(Z b)]}
                _merge(G, _product(A, bg['B0'], _curl_pert(A, zb), -dl))
                urow = dict(G['q'])                                        # u: -r³ [q - (r s)'] (ρ0 = 1);   v: -r² t
                for key, M in _deriv(_shift(G['s'], 1., 1)).items():
                    _add(urow, key, -M)
                _store_ma(out, 'u', X, _shift(urow, -1., 3), Lt)
                _store_ma(out, 'v', X, _shift(G['t'], -1., 2), Lt)
    return ma._prune(out)

def spheroid_lambda(wall, m, lw, f=None):
    '''
    λ_l on the degrees 0..lw of the insulating wall rows (the header: q_l(b̃) = λ_l s_l(Q b̃) at the wall), λ_l = (a²/k)
    d ln R_l(ν0)/dν with R_l = Q_l^m outside the outer wall ('outer') and P_l^m inside an insulating inner core ('inner'), of
    the argument z = iν (oblate) or ν (prolate), by recurrence in l with the derivatives (z² - 1) R_l' = l z R_l - (l + m)
    R_(l-1) = (l - m + 1) R_(l+1) - (l + 1) z R_l: the ratios P_l/P_(l-1) forward from l = m (P is the dominant solution),
    Q_(l+1)/Q_l backward from far above lw (Q is the minimal one). The sphere's -(l + 1) and l below l = m and for f = 0.
    Checked against mpmath (legenp, legenq) and the potential fields of the spheroid: tests/centrifugal/spheroid_check.py.
    '''
    a, c = spheroid_axes(f)
    lv = np.arange(lw + 1)
    lam = -(lv + 1.) if wall == 'outer' else lv*1.
    if a == c:
        return lam
    k = np.sqrt(abs(a*a - c*c))
    z, dz = (1j*c/k, 1j) if a > c else (c/k + 0j, 1.)
    if wall == 'inner':
        t = None
        for l in range(m, lw + 1):
            if l == m:
                d = m*z/(z*z - 1)
            else:
                t = (2*m + 1)*z if l == m + 1 else ((2*l - 1)*z - (l - 1 + m)/t)/(l - m)
                d = (l*z - (l + m)/t)/(z*z - 1)
            lam[l] = (a*a/k)*(dz*d).real
    else:
        rho = abs(z + np.sqrt(z*z - 1))                                    # the ratio of the two solutions per degree
        top = lw + 20 + int(min(np.ceil(40/np.log(max(rho*rho, 1 + 1e-12))), 10**6))
        s = 0j
        for l in range(top, m - 1, -1):   # s = Q_(l+1)/Q_l; then Q_l/Q_(l-1) = (l + m)/((2l + 1) z - (l - m + 1) s)
            if l <= lw:
                lam[l] = (a*a/k)*(dz*((l + 1)*z - (l - m + 1)*s)/(1 - z*z)).real
            s = (l + m)/((2*l + 1)*z - (l - m + 1)*s)
    return lam

def spheroid_field_walls(wall, m, lw, R, f=None):
    '''
    The exact insulating wall rows at the radius R of the wall 'outer' or 'inner' (an insulating inner core), as their parts
    beyond the sphere's rows (assemble.bc_b_spherical: f, (l + 1) F + R F' outside, l F - R F' inside; g, G), in the format of
    wall_coefficients: {(row, source): sparse matrix [l, l']}, rows 'f', 'g', sources 'F0', 'F1', 'G0'. From the header's
        outside  (l - μ) F - μ R (F' + m_s) = 0,  μ = λ_l/(l + 1);    inside  (l + 1 - ν) F - ν R (F' + m_s) = 0,  ν = λ_l/l;
        G + m_t = 0,
    λ_l of spheroid_lambda, m_s and m_t the S and T parts of (Q - I) b̃ at R (spheroid_primitives) with q = L F/R,
    s = F' + F/R, t = G. Also used by tests/centrifugal/spheroid_check.py without a run.
    '''
    C, A = spheroid_primitives('inertia', m, lw, f)
    L = A.L
    lv = np.arange(lw + 1, dtype=float)
    lam = spheroid_lambda(wall, m, lw, f)
    mS = {'F0': C['s']['q'] @ L/R + C['s']['a']/R, 'F1': C['s']['a'], 'G0': C['s']['T']}
    mT = {'F0': C['t']['q'] @ L/R + C['t']['a']/R, 'F1': C['t']['a'], 'G0': C['t']['T']}
    if wall == 'outer':
        mu = lam/(lv + 1)                                              # -1 in the sphere
        out = {('f', 'F0'): sp_diags(-(1 + mu)), ('f', 'F1'): sp_diags(-R*(1 + mu))}
    else:
        mu = np.where(lv > 0, lam/np.where(lv > 0, lv, 1.), 1.)        # 1 in the sphere
        out = {('f', 'F0'): sp_diags(1 - mu), ('f', 'F1'): sp_diags(R*(1 - mu))}
    D = sp_diags(-R*mu)
    for k in ('F0', 'F1', 'G0'):
        out[('f', k)] = D @ mS[k] + out[('f', k)] if ('f', k) in out else D @ mS[k]
        out[('g', k)] = mT[k]
    return {k: sp_csr(v) for k, v in out.items()}

def spheroid_potential(wall, m, lw, R, q, s, t, f=None):
    '''
    The potential field that matches the field b̃ at the wall (post-processing, utils4pp.centrifugal_magnetic_terms): from
    the l-components q, s, t of b̃ at r = R (arrays on the degrees 0..lw), (A, λ) with A_l = s_l(Q b̃), the coefficients of the
    tangential field ∇_tan Φ = Σ A_l r∇Y_l at R (Φ = R Σ A_l Y_l on the wall), and λ_l of spheroid_lambda: the wall rows make
    q_l = λ_l A_l, and the energy of the potential field is -R³ Σ λ_l |A_l|² ∫|Y_l|² dΩ outside, R³ Σ λ_l |A_l|² ∫|Y_l|² dΩ
    inside an insulating inner core (∫ |∇φ|² dV = ∓∮ Φ* b̃_r R² dΩ)
    '''
    C = spheroid_primitives('inertia', m, lw, f)[0]
    return s + C['s']['q'] @ q + C['s']['a'] @ s + C['s']['T'] @ t, spheroid_lambda(wall, m, lw, f)

def spheroid_icb(m, lw, R, f=None):
    '''the exact metric part of the ICB row of the electric field of a conducting inner core, as icb_electric: {(X, d):
    sparse matrix [l, l']}, R times the S part of δ𝐄 = Q ∇×(Q b̃) - ∇×b̃ per unit η at r = R (minus spheroid_terms' 'mag')'''
    A = angular.Algebra.get(m, lw)
    out = {}
    for X in 'FG':
        for (Xn, d), F in _at(_sc(spheroid_terms(A, X, 'mag', f), -1.), R, {}).items():
            _add(out, (Xn, d), R*F.s)
    return {k: sp_csr(v) for k, v in out.items()}


# --- NEW (2026-11-09): conducting mantles on the exact spheroid: the resolved mantle and the thin conducting layer
# The resolved mantle (mantle = 'resolved') fills the level surfaces 1 <= r <= r_surface, spheroids of the fluid's flattening.
# At rest its field only diffuses (FIX 2026-11-09: a fluid slipping along it adds its u×B0 to the row at r = 1, as in the
# sphere, assemble.bc_b_slip), σ ∂ₜb̃ = -∇×E + (ln σ)' r̂ × E per MagnetD, E = Q ∇×(Q b̃) the
# covariant electric field (Fᵀ times the body's, per unit η) and σ = σ(r) constant on the level surfaces. Its rows are those
# of the sphere (assemble.bc_b_mantle, divided by η) plus the metric parts:
#     diffusion   the fluid's (spheroid_build 'mag': -∇×δE, δE = Q ∇×(Q b̃) - ∇×b̃) and, in the rows gm, the part of
#                 (ln σ)' r̂ × δE, -r² (ln σ)' s(V) with V = -δE (A sign; its sphere part is the -(ln σ)'(r² g' + r g) of
#                 operators.magnetic_diffusion): labels 'f{rx}eta0_D{d}', 'g{rx}eta0_D{d}', 'g{rx}dls0_D{d}' (r^(2 - rx),
#                 times (ln σ)' for dls, the mantle's rows being at r² like the fluid's in a shell), built on the mantle's
#                 Chebyshev series (submatrices.py, + '_man'); with F''' terms the rows gm are in the C^(3) basis;
#     at r = 1    b̃ is continuous (the rows f - fm, (r f)' - (r fm)', g - gm keep their form) and so is the tangential E: the
#                 row σ(1) (r g)' - (r gm)' gains σ(1) R s(δE) of the fluid minus R s(δE) of the mantle (spheroid_icb), as
#                 at a conducting inner core;
#     at r_surface the insulating wall rows of the spheroid (spheroid_field_walls at R = r_surface: the level surface r_surface
#                 is homothetic to the wall, so λ_l is the same).
# The thin conducting layer (mantle = 'TWA', c_cmb = 0, mu = 1) is the thin limit of that mantle: the layer 1 <= r <= 1 + h,
# whose thickness in the body is a h at the equator and c h at the poles, with σ_w h = c1 R fixed as h -> 0 (c1 = c1_cmb).
# There its current is a sheet K̃ = ∫ J̃ dr along the wall, J̃ = σ_w Q⁻¹ E (the flux of the current, J̃ = ∇×(Q b̃)), with the
# tangential E of the fluid next to it (FIX 2026-11-09: with a slipping fluid, its u×B0 too: spheroid_twa_slip) and its radial part set by J̃_r = 0 inside the layer: (Q⁻¹ E)_r = 0 gives,
# with x = cos θ and ẑ_H = ẑ - x r̂,
#     K̃ = (c1 R/a²) [E_H + V(x) ẑ_H (ẑ_H⋅E_H)],     V(x) = (a² - c²)/(c² + (a² - c²) x²)
# (V = 0 in the sphere: K̃ = c1 R (∇×b̃)_H). V is not a polynomial: its Legendre series (to 1e-17 of its first coefficient,
# degree 46 at f = 0.3, 28 at f = 0.1, 12 at f = 1/300) widens the coupling of the wall rows to that degree. The
# sheet makes the tangential H = Q b̃ jump to the potential field ∇Φ outside, ∇Φ_H - H_H = K̃ × r̂, and b̃_r is continuous:
#     f:  (l - μ) F - μ R (F' + m_s - k_t) = 0,     g:  G + m_t + k_s = 0
# with μ, m_s, m_t of the insulating wall (spheroid_field_walls) and k_s, k_t the S and T parts of K̃ (in the sphere,
# μ = -1: (l + 1) F + R F' + c1 R² ∇²_l F = 0 and G + c1 (G + R G') = 0, assemble.bc_b_twa with c = 0, mu = 1). Checks:
# tests/mantle/derive_spheroid.py (the sheet against Q⁻¹E pointwise) and tests/mantle/spheroid_mantle_check.py (runs).

def twa_weight(f=None):
    '''
    V(x) = (a² - c²)/(c² + (a² - c²) x²) of the thin layer's sheet current as a numpy Legendre series (0 in the sphere), its
    coefficients below 1e-17 of the first dropped. V = 1/(x² ± x0²) with x0 = c/k (k² = |a² - c²|; + oblate, - prolate),
    and from 1/(t - x) = Σ (2n + 1) P_n(x) Q_n(t) (t off [-1, 1]) its even coefficients are V_n = -(2n + 1) Q_n(t)/t at
    t = i x0 (oblate) or x0 (prolate): Q_n by the ratios Q_n/Q_(n-1) = n/((2n + 1) t - (n + 1) Q_(n+1)/Q_n) from far above
    (Q is the minimal solution, as in spheroid_lambda) and Q_0(t) = ½ ln((t + 1)/(t - 1)). (A quadrature of V by numpy's
    Gauss-Legendre nodes left the coefficients 5e-14 off.)
    '''
    a, c = spheroid_axes(f)
    k2 = a*a - c*c
    if k2 == 0:
        return np.polynomial.Legendre([0.])
    x0 = c/np.sqrt(abs(k2))
    t = 1j*x0 if k2 > 0 else x0 + 0j
    rho = abs(t + np.sqrt(t*t - 1))                       # the coefficients fall as rho^-n
    top = int(min(2000, np.ceil(45/np.log(rho)) + 20))
    ratio, s = np.zeros(top + 1, dtype=complex), 0j
    for n in range(top, 0, -1):                           # ratio[n] = Q_n/Q_(n-1)
        s = n/((2*n + 1)*t - (n + 1)*s)
        ratio[n] = s
    Qn = 0.5*np.log((t + 1)/(t - 1))
    co = np.zeros(top + 1)
    for n in range(top + 1):
        if n > 0:
            Qn = Qn*ratio[n]
        if n % 2 == 0:
            co[n] = (-(2*n + 1)*Qn/t).real
    co[np.abs(co) < 1e-17*abs(co[0])] = 0.
    return np.polynomial.Legendre(np.trim_zeros(co, 'b'))

def spheroid_twa(m, lw, R, c1, f=None):
    '''
    The rows of the thin conducting layer (c = 0, mu = 1) on the exact spheroid at the wall r = R as their parts beyond the
    sphere's thin-wall rows (assemble.bc_b_twa: f, (l + 1) F + R F' + c1 R² ∇²_l F; g, G + c1 (G + R G')), in the format of
    spheroid_field_walls with the sources 'F0', 'F1', 'F2', 'G0', 'G1': those of the insulating spheroid plus μ R k_t + R k_t°
    (f) and k_s - k_s° (g), k_s, k_t the S and T parts of the sheet K̃ of the header and k_s°, k_t° those of the sphere's
    c1 R (∇×b̃)_H. lw: build the algebra with a margin above the largest degree used (the weight V has its own degree).
    '''
    a, c = spheroid_axes(f)
    A = angular.Algebra.get(m, lw)
    lv = np.arange(lw + 1, dtype=float)
    mu = spheroid_lambda('outer', m, lw, f)/(lv + 1)
    out = dict(spheroid_field_walls('outer', m, lw, R, f))
    V = twa_weight(f)
    tang = lambda e: _map(e, lambda F: angular.Field(0*F.q, F.s, F.t))
    for X in 'FG':
        j = _curl(A, _field(A, X))
        E = _sum(j, _sc(spheroid_terms(A, X, 'mag', f), -1.))                        # Q ∇×(Q b̃)
        EH = tang(E)
        K = _sc(_sum(EH, _poly(A, tang(_zhat(A, _zdot(A, EH))), V)), c1*R/(a*a))      # K̃
        K0 = _sc(tang(j), c1*R)                                                       # the sphere's
        for (Xn, d), F in _at(K, R, {}).items():
            _add(out, ('f', '%s%d' % (Xn, d)), sp_diags(R*mu) @ F.t)
            _add(out, ('g', '%s%d' % (Xn, d)), F.s)
        for (Xn, d), F in _at(K0, R, {}).items():
            _add(out, ('f', '%s%d' % (Xn, d)), R*F.t)
            _add(out, ('g', '%s%d' % (Xn, d)), -F.s)
    return {k: sp_csr(v) for k, v in out.items()}

def spheroid_twa_slip(m, lw, R, c1, br, f=None):
    '''
    NEW (2026-11-09): a fluid slipping along the thin conducting layer on the exact spheroid (stress-free, bco = 0, or
    inviscid; utils.twa_slips): the terms of its u×B0 in the layer's rows, which replace the sphere's (assemble.bc_b_slip),
    in the format of spheroid_twa with the sources 'P0', 'P1', 'T0' (the flow's columns at R). The covariant electric field
    of the fluid is E = MagnetD Q ∇×(Q b̃) - w×B̃0 (Fᵀ(u×B) = w×B̃0 at J = 1, w = F⁻¹u, kore's P and T, and B̃0 = F⁻¹B0, kore's
    B0 formula), and the tangential E is continued into the layer at rest (condition C.I, as in the sphere): the sheet of
    spheroid_twa's header gains, per MagnetD (the rows of the layer being those of E/MagnetD),
        K̃_slip = (c1 R/a²) [S + V(x) ẑ_H (ẑ_H⋅S)],     S = -(w×B̃0)_H = B0_r (r̂×w_t)
    (w_r = 0 at the wall): f gains μ R k_t, g gains k_s of K̃_slip, without the 1/MagnetD. br: B0_r(R, x) as a numpy Legendre
    series (Σ n(n+1) h_n(R)/R P_n, background_field.pol_profiles). In the sphere (V = 0, a = 1, μ = -1) these are bc_b_slip's
    (G' - R S_l/MagnetD in the g row times c1, D F + R² X_l/(L MagnetD) in the f row times c1).
    '''
    a, c = spheroid_axes(f)
    A = angular.Algebra.get(m, lw)
    lv = np.arange(lw + 1, dtype=float)
    mu = spheroid_lambda('outer', m, lw, f)/(lv + 1)
    V = twa_weight(f)
    tang = lambda e: _map(e, lambda F: angular.Field(0*F.q, F.s, F.t))
    out = {}
    for X in 'PT':
        S = _poly(A, _rcr(tang(_flow(A, X))), br)
        if V.degree() > 0 or V.coef[0] != 0:
            S = _sum(S, _poly(A, tang(_zhat(A, _zdot(A, S))), V))
        K = _sc(S, c1*R/(a*a))
        for (Xn, d), F in _at(K, R, {('lam', 1): 0.}).items():   # λ1 = 0: the spheroid is homogeneous
            _add(out, ('f', '%s%d' % (Xn, d)), sp_diags(R*mu) @ F.t)
            _add(out, ('g', '%s%d' % (Xn, d)), F.s)
    return {k: sp_csr(v) for k, v in out.items()}

def mantle_metric_on():
    '''the metric parts of the resolved mantle: the exact spheroid with a field, diffusion and mantle = 'resolved' '''
    return bool(exact() and par.magnetic and par.MagnetD > 0 and par.mantle == 'resolved')

def twa_metric_on():
    '''the rows of the thin conducting layer on the exact spheroid (spheroid_twa): mantle = 'TWA' there'''
    return bool(exact() and par.magnetic and par.mantle == 'TWA')

LABEL_MAN = re.compile(r'([fg])(\d)(eta|dls)0_D(\d)$')

def spheroid_mantle_build(m, lw, sigma_varies=True, f=None):
    '''
    {(row, column): {label: sparse matrix [target l', source l]}} of the metric diffusion of the resolved mantle on the
    degrees 0..lw (the header): the fluid's 'mag' terms of spheroid_build and, if sigma_varies, the (ln σ)' part of the rows
    g ('g{rx}dls0_D{d}'), in kore's A sign, without MagnetD and the 1/L'^3 of the row. Also used by
    tests/mantle/derive_spheroid.py.
    '''
    import mag_algebra as ma
    out = {k: dict(v) for k, v in spheroid_build(m, lw, ('mag',), f)['mag'].items()}
    if sigma_varies:
        A = angular.Algebra.get(m, lw)
        o = {}
        for X in 'FG':
            s = _parts(spheroid_terms(A, X, 'mag', f))[1]                 # s(V), V = -δE
            _store_sph(o, 'g', X, _shift(s, -1., 2), lambda M: A.L @ M, 'dls')
        for k, d in ma._prune(o).items():
            out.setdefault(k, {}).update(d)
    _check_orders(out)
    return out

_man = {}
def mantle_coefficients():
    '''spheroid_mantle_build for this run (cached, the (ln σ)' part only for a sigma_mantle that is an expression):
    (out, {(row, column): {label: bands}}), |Δl| <= 4'''
    if not _man:
        out = spheroid_mantle_build(par.m, par.lmax + 16, isinstance(par.sigma_mantle, str))
        if _bandwidth(out) > 4:
            sys.exit('centrifugal: the metric diffusion of the mantle couples degrees more than 4 apart')
        _man.update({'out': out, 'bands': {rc: {lab: angular.band(M, 4) for lab, M in d.items()} for rc, d in out.items()}})
    return _man['out'], _man['bands']

def mantle_labels():
    '''[(label, row 'f' or 'g', column 'F' or 'G', power of r, profile None or 'dls', derivative order)]: the operators of the
    metric diffusion in the resolved mantle, for submatrices.py (as label + '_man')'''
    out = []
    for (row, X), d in mantle_coefficients()[0].items():
        for lab in d:
            mt = LABEL_MAN.match(lab)
            if mt is None or mt.group(1) != row:
                sys.exit('centrifugal: the mantle label %s has no operator' % lab)
            out.append((lab, row, X, 2 - int(mt.group(2)), None if mt.group(3) == 'eta' else 'dls', int(mt.group(4))))
    return out

def diffusion_block_m(l, section, component, offdiag, M):
    '''
    The block of the metric diffusion inside the resolved mantle for the row degree l of section 'fm' or 'gm', the column
    'bpol_m' or 'btor_m' and the source degree l + offdiag (the A sign; the rows divided by η, as the sphere's): [MagnetD Σ
    coefficient × the mantle's operator (label + '_man') / L'^3, the column block offset in units of N_mantle], or [0, 0].
    '''
    import utils as ut
    row = {'fm': 'f', 'gm': 'g'}[section]
    X = {'bpol_m': 'F', 'btor_m': 'G'}[component]
    out = 0
    for lab, bnd in mantle_coefficients()[1].get((row, X), {}).items():
        v = _lookup(bnd, l, par.m, offdiag)
        if v != 0:
            out = out + v*M[lab + '_man']
    if np.isscalar(out):
        return [0, 0]
    lF, lG = ut.ell(par.m, par.lmax, ut.bsymm)[:2]
    rows, cols = {'f': lF, 'g': lG}[row], {'F': lF, 'G': lG}[X]
    offd = int(np.where(cols == l + offdiag)[0][0] - np.where(rows == l)[0][0])
    return [par.MagnetD*out/(l*(l + 1.))**3, offd]

_mwall = {}
def mantle_rows():
    '''the metric parts of the resolved mantle's wall rows (cached, by diagonal): 'cmb', the electric-field row at r = 1
    (spheroid_icb: {(X, d): bands}, R s(δE)), and 'surface', the insulating rows at r_surface (spheroid_field_walls:
    {(row, source): bands}); each as (B, bands)'''
    import utils as ut
    if not _mwall:
        for key, W in (('cmb', spheroid_icb(par.m, par.lmax + 12, ut.rcmb)),
                       ('surface', spheroid_field_walls('outer', par.m, par.lmax + 12, par.r_surface))):
            B = _bandwidth({0: W})
            _mwall[key] = (B, {k: angular.band(M, B) for k, M in W.items()})
    return _mwall

_twa_slip = {}
def twa_slip_rows(br):
    '''spheroid_twa_slip for this run at the outer wall (c1_cmb, br the B0_r series there), cached: {(row, source): csr}'''
    import utils as ut
    if not _twa_slip:
        _twa_slip.update(spheroid_twa_slip(par.m, par.lmax + len(twa_weight().coef) + br.degree() + 12, ut.rcmb,
                                           par.c1_cmb, br))
    return _twa_slip

_twa = {}
def twa_rows():
    '''spheroid_twa for this run at the outer wall (c1_cmb), cached, by diagonal: (B, {(row, source): bands})'''
    import utils as ut
    if not _twa:
        W = spheroid_twa(par.m, par.lmax + len(twa_weight().coef) + 12, ut.rcmb, par.c1_cmb)
        B = _bandwidth({0: W})
        _twa['rows'] = (B, {k: angular.band(M, B) for k, M in W.items()})
    return _twa['rows']
