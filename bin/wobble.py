'''
NEW (2026-11-09): the wobble of the mantle (wobble = 1). The mantle turns as a rigid body about an equatorial axis, under the
pressure and the viscous torques of the fluid it holds: the m = 1 coupling of the rigid-body dynamics of the mantle with the
flow of the core of Triana, Rekier, Trinh & Dehant (2019, GJI 218, 1071; arXiv:1904.11837). With a flattened wall the
pressure of the flow has a torque on the mantle, and the free core nutation (the spin-over of the fluid), the Chandler wobble
(the free wobble of the whole body) and the inertial modes of the core become modes of one system.

WHY THIS WAY. The fluid is described in the frame of the mantle, as in the paper: its wall is then at rest, the no-slip rows
are those of a fixed wall, and the mantle's angular acceleration enters as the Poincaré force. The flattened wall is kore's
exact spheroid (centrifugal = 1, centrifugal_shape 'spheroid', centrifugal.py section 8): the map X = D α, D = diag(a, a, c),
c/a = 1 - f, a² c = 1, which makes the wall the reference sphere r = 1 at any flattening. Both torques are then integrals on
that sphere, with no Taylor expansion of the wall (the paper's Appendix A): the pressure torque from the reduced pressure on
the wall, which the horizontal momentum balance there gives from the viscous force alone (the flow is zero on the wall), and
the viscous torque from ∂w/∂r on the wall. A spherical body (centrifugal = 0) is the case a = c = 1, where the pressure has no
torque.

THE EQUATIONS (Gaspard = G the rotation rate of the frame, ν = ViscosD, kore's units, ρ = 1). The mantle turns at G ẑ + M
(M equatorial, small). In the frame of the mantle the fluid obeys
    ∂ₜu + 2G ẑ×u + ∂ₜM × X = -∇p + ν∇²u,       p = P - G (ẑ×X)⋅(M×X) + const,
P the pressure: the centrifugal force of the tilted rotation is a gradient and goes into p. In the reference space (kore's
covariant rows, Fᵀ times the equation) the Poincaré force is Fᵀ(∂ₜM × Dα) = ∂ₜ(D⁻¹M) × α, a rigid rotation of α at D⁻¹M:
it enters the rows of T of degree 1 only, as the inertia of the flow T₁ = r Mc/(2a) (b_rows, column of Mc). Its torque on
the mantle, with the outward normal n of the fluid,
    Γ = ∮ P X×n dS - ν ∫ X×∇²u dV
(the paper's eq. 3). On the spheroid, X×n dS = (a/c - c/a) z (y, -x, 0) dΩ in the coordinates α of the unit sphere, so
    Γ_P = (a/c - c/a) ∮ P z (y, -x, 0) dΩ,
which takes the part of degree 2 of P only. Its part G (ẑ×X)⋅(M×X) = -G a c z (Mₓ x + M_y y) gives
    Γ_cent = -G (4π/15)(a² - c²) M×ẑ = -G (C_f - A_f) M×ẑ,
the torque that makes the fluid turn with a tilted axis (C_f, A_f the moments of inertia of the fluid). The reduced pressure
on the wall: there w = 0, so the inertia and the Coriolis force vanish and ∇_H p = [ν V - ∂ₜ(D⁻¹M) × α]_H, V = Fᵀ∇²u =
D² L w the covariant viscous force (L = ∇²/a² + (1/c² - 1/a²)∂z² the body's Laplacian in the reference space, J = 1). The
rigid rotation is divergence-free on the sphere, so p₂ (the coefficient of Y₂¹, kore's Schmidt harmonic) is
    p₂ = -(ν/6) [∇_H⋅V_H]₂ = -(ν/6) (G1 + a²β G2 + (c²/a² - 1) G3 + (c² - a²)β G4),    β = 1/c² - 1/a²,
with G1..G4 the parts of degree 2 of the horizontal divergence of ∇²w, ∂z²w, ẑ(ẑ⋅∇²w), ẑ ∂z²w_z on the wall, linear forms in
the values and derivatives of P₂, P₄, P₆, T₁, T₃, T₅ at r = 1 (_PRESSURE, derived with sympy, tests/wobble/derive.py). In a sphere
p₂ = ν (3P₂″ + P₂‴) at the wall, the paper's Π₂₁ (eq. B.2) there. The viscous torque, ∫ X×∇²u dV = D⁻¹ ∫ α×Lw dV (the
cofactor of D, det D = 1), by parts with w = 0 on the wall:
    Γ_ν = -(ν/a) ∮ (sin²θ/a² + cos²θ/c²) r̂ × ∂w/∂r dΩ   (equatorial part; _VISCOUS: T₁′, T₃′, P₂″ at r = 1).
The mantle (moments A_M about an equatorial axis, C_M about ẑ, e = (C_M - A_M)/A_M):
    A_M ∂ₜM + G (C_M - A_M) M×ẑ = Γ.
With the complex amplitudes of m = 1 (the convention of resid𝐋, rotdyn.angular_momentum: the x - i y part of a vector
divided by the normalisation of Y₁¹ at the equator, -1/√2), (M×ẑ)₋ = i M₋, and the unknown Mc = M₋/(-1/√2) is the rate
whose rigid rotation is T₁ = r Mc/2:
    λ A_M Mc + i G [e A_M + (4π/15)(a² - c²)] Mc = Γ_P' + Γ_ν                        (the row of Mc: a_rows, b_rows)
with Γ_P' = -i (a/c - c/a) sqrt(2/3) (4π/5) p₂/(-1/√2), the torque of the reduced pressure. The free Chandler wobble of the
mantle alone is λ = -i G e (prograde, Im λ < 0 in kore's convention, where the spin-over is near +i).

Budget (budget): the angular momentum of the fluid, H_f = I_f(Gẑ + M) + L, L = ∫X×u dV = D⁻¹∫α×w dV, in the frame of the
mantle: λ(A_f M + L) + G(C_f - A_f) M×ẑ + G ẑ×L = -Γ; its residual checks the two torques together.

NEW (2026-11-09): THE FIELD (magnetic = 1). B0 is steady in the frame of the mantle (it turns with the mantle's axis), and
the fluid's flow relative to that frame induces b; the induction, the Lorentz force and the wall rows are kore's. The field
adds two torques on the mantle:
  - the Lorentz part of the pressure on the wall, Γ_P'(Lorentz) = pressure_factor() p₂(Lorentz): on the no-slip wall the
    horizontal balance becomes ∇_H p = [ν V + F - ∂ₜ(D⁻¹M) × α]_H with F = Hendrik [∇×(Q b̃) × B̃0 + ∇×(Q B̃0) × b̃] the
    covariant Lorentz force of the exact spheroid (centrifugal.py, section 8; Q = D², b̃ = D⁻¹b, B̃0 = D⁻¹B0 the fluxes of the
    reference space), so p₂ gains the projection of F_H on ∇_H Y₂¹ over its norm 6 ∫|Y₂¹|² dΩ (none in a sphere);
  - Γ_B, the torque of the field on everything beyond the fluid's side of the wall, minus the Maxwell stress through it on the
    fluid, Hendrik ∮ X × [B0 (b⋅n) + b (B0⋅n) - (B0⋅b) n] dS. It acts on the currents of a conducting mantle ('TWA' or
    'resolved'), and on the sources of any part of B0 that the fluid does not carry, which turn with the
    mantle: an external part ('axial', imposed from outside) and the wall currents of a field that does not meet the
    potential field of the spheroid. With an insulating mantle and a field of internal currents that meets it, Γ_B = 0 (the
    stress through any surface outside the currents is that at infinity), so the field adds only the pressure. NEW
    (2026-11-09): a toroidal B0 that does not vanish at the wall 'leaks' into a conducting mantle, carried there by its
    steady currents (no sheet at the wall: B0 and b are continuous through it); Γ_B is then the torque on those currents and
    on the mantle's induced ones, ∫ X × (J0 × b + j × B0) dV over the mantle, whatever B0 does inside it (the stress at its
    outer surface is that of the potential fields beyond).
Both are linear forms on f, f', f'' and g, g' at the wall (field_rows), computed on a Gauss-Legendre grid in θ (_wall_fields,
pure numpy, checked against the divergence theorem in tests/wobble/derive_field.py); ∇×(Q v) = a² ∇×v + (c² - a²) ∇v_z × ẑ.
The mantle's row gains Γ_P'(Lorentz) + Γ_B, and the budget of the fluid the torque of the field on it, -Γ_B: it then checks
the Lorentz part of the pressure (from the force at the wall) against the Maxwell stress (from the fields at the wall).

NEW (2026-11-09): A FLUID THAT SLIPS ALONG THE WALL (a stress-free wall, bco = 0, or an inviscid fluid, ViscosD = 0; slips()).
The flow along the wall, w_H ≠ 0, adds to the horizontal balance there its inertia and its Coriolis force, in the covariant
form of the exact spheroid (Fᵀ∂ₜu = M ∂ₜw, M = D², and Fᵀ(2G ẑ×u) = (2G/c) ẑ×w at det D = 1):
    ∇_H p = [ν V + F - λ M w - (2G/c) ẑ×w - ∂ₜ(D⁻¹M) × α]_H,
so that p₂ gains (λ/6)(a² [w] + (c² - a²) [zwz]) + (G/(3c)) [zxw], the degree-2 parts of the horizontal divergence of w, ẑ w_z
and ẑ×w on the wall (_PRESSURE_SLIP, coefficients of f, f′; the part in λ goes to the row of Mc in B, inertia_rows). The
viscous torque, ∫ α×Lw dV by Green's identity without w = 0 on the wall (∇²α = 0, ∂z²α = 0):
    ∫ α×∇²w dV = ∮ (r̂×∂w/∂r - r̂×w) dΩ,     ∫ α×∂z²w dV = ∮ x (r̂×∂z w - ẑ×w) dΩ,
whose terms beyond the no-slip ones (∂z w = x ∂w/∂r there) are _VISCOUS_SLIP: the traction's normal part has a torque on the
spheroid, and in a stress-free sphere the terms cancel (T₁′ = T₁ at the wall: no torque). slip_weights gathers them.

LEFT OUT (background.check_wobble stops): an inner core (full sphere only), with an insulating mantle a toroidal B0 that
does not vanish at the wall (its current sheet), with a slipping fluid also a sheet of the poloidal B0 that pushes it (background.sheet_push), buoyancy
(a moving mass adds gravitational torques), the other shapes of centrifugal = 1 (whose wall is not the sphere of a linear
map), forcings.
'''
import numpy as np
from parameters import par
import utils as ut

_N1 = -1/np.sqrt(2)   # Y_1^1 at the equator, kore's normalisation (rotdyn.wall_torques)

# The degree-2 parts of the horizontal divergence on the wall (G1..G4 of the docstring) of the flow of each (kind, l), as the
# coefficients of (f, f′, f″, f‴) at r = 1 (f = P_l or T_l, kore's Schmidt harmonics). From tests/wobble/derive.py.
_S30 = np.sqrt(30.)
_PRESSURE = {
    ('P', 2): {'lap':   [-36, 36, -18, -6],
               'dzz':   [-96/7, 96/7, -48/7, -16/7],
               'zlapz': [-78/7, 102/7, -9, -19/7],
               'zdzzz': [-22/7, 2, -3/7, -5/21]},
    ('P', 4): {'lap':   [0, 0, 0, 0],
               'dzz':   [100*_S30/21, -100*_S30/21, -200*_S30/63, -20*_S30/63],
               'zlapz': [-400*_S30/21, -80*_S30/21, 20*_S30/9, 20*_S30/63],
               'zdzzz': [-510*_S30/77, -46*_S30/11, -124*_S30/231, -2*_S30/231]},
    ('T', 1): {'lap':   [0, 0, 0, 0],
               'dzz':   [0, 0, -2j/np.sqrt(3), 0],
               'zlapz': [-2j*np.sqrt(3), 2j*np.sqrt(3), 1j*np.sqrt(3), 0],
               'zdzzz': [-6j*np.sqrt(3)/7, 6j*np.sqrt(3)/7, 1j*np.sqrt(3)/21, 0]},
    ('T', 3): {'lap':   [0, 0, 0, 0],
               'dzz':   [0, -20j*np.sqrt(2)/7, -4j*np.sqrt(2)/7, 0],
               'zlapz': [48j*np.sqrt(2)/7, -8j*np.sqrt(2)/7, -4j*np.sqrt(2)/7, 0],
               'zdzzz': [32j*np.sqrt(2)/7, 2j*np.sqrt(2)/3, -2j*np.sqrt(2)/21, 0]},
    ('P', 6): {'lap': [0, 0, 0, 0], 'dzz': [0, 0, 0, 0], 'zlapz': [0, 0, 0, 0],
               'zdzzz': [2800*np.sqrt(7)/143, 2320*np.sqrt(7)/143, 480*np.sqrt(7)/143, 80*np.sqrt(7)/429]},
    ('T', 5): {'lap': [0, 0, 0, 0], 'dzz': [0, 0, 0, 0], 'zlapz': [0, 0, 0, 0],
               'zdzzz': [-128j*np.sqrt(5)/77, -16j*np.sqrt(5)/21, -16j*np.sqrt(5)/231, 0]},
}
# ∮ sin²θ (r̂ × ∂w/∂r)₋ dΩ and ∮ cos²θ (...)₋ dΩ on the wall, as coefficients of (f, f′, f″) at r = 1
_PI = np.pi
_VISCOUS = {
    ('P', 2): {'sin2': [8j*np.sqrt(6)*_PI/15, -8j*np.sqrt(6)*_PI/15, -8j*np.sqrt(6)*_PI/15],
               'cos2': [-8j*np.sqrt(6)*_PI/15, 8j*np.sqrt(6)*_PI/15, 8j*np.sqrt(6)*_PI/15]},
    ('P', 4): {'sin2': [0, 0, 0], 'cos2': [0, 0, 0]},
    ('T', 1): {'sin2': [0, -8*np.sqrt(2)*_PI/5, 0], 'cos2': [0, -16*np.sqrt(2)*_PI/15, 0]},
    ('T', 3): {'sin2': [0, 64*np.sqrt(3)*_PI/105, 0], 'cos2': [0, -64*np.sqrt(3)*_PI/105, 0]},
    ('P', 6): {'sin2': [0, 0, 0], 'cos2': [0, 0, 0]},
    ('T', 5): {'sin2': [0, 0, 0], 'cos2': [0, 0, 0]},
}

# NEW (2026-11-09): a fluid that slips along the wall (bco = 0, or ViscosD = 0; slips()). The degree-2 parts on the wall of the
# horizontal divergence of w ('w'), ẑ w_z ('zwz') and ẑ × w ('zxw'), as coefficients of (f, f′) at r = 1, for the inertia and
# the Coriolis force in the reduced pressure; and the terms in w of the viscous torque's boundary integrals (zero on a no-slip
# wall), ∮(-r̂ × w)₋ dΩ ('lapw', of ∇²) and ∮ x (r̂ × (∂z w - x ∂r w) - ẑ × w)₋ dΩ ('dzzw', of ∂z²), coefficients of (f, f′).
# From tests/wobble/derive.py.
_PRESSURE_SLIP = {
    ('P', 2): {'w': [-6, -6], 'zwz': [-25/7, -19/7], 'zxw': [7j, 1j]},
    ('P', 4): {'w': [0, 0], 'zwz': [100*_S30/63, 20*_S30/63], 'zxw': [0, 0]},
    ('T', 1): {'w': [0, 0], 'zwz': [1j*np.sqrt(3), 0], 'zxw': [-np.sqrt(3), 0]},
    ('T', 3): {'w': [0, 0], 'zwz': [-4j*np.sqrt(2)/7, 0], 'zxw': [-16*np.sqrt(2)/7, 0]},
}
_VISCOUS_SLIP = {
    ('P', 2): {'lapw': [0, 0], 'dzzw': [-8j*np.sqrt(6)*_PI/3, 8j*np.sqrt(6)*_PI/15]},
    ('T', 1): {'lapw': [8*np.sqrt(2)*_PI/3, 0], 'dzzw': [16*np.sqrt(2)*_PI/15, 0]},
    ('T', 3): {'lapw': [0, 0], 'dzzw': [-256*np.sqrt(3)*_PI/105, 0]},
}


def axes():
    '''(a, c) of the wall: the exact spheroid's (centrifugal.spheroid_axes), or (1, 1) for a spherical body'''
    if par.centrifugal:
        import centrifugal as cf
        return cf.spheroid_axes()
    return 1.0, 1.0


def fluid_inertia():
    '''(A_f, C_f): the moments of inertia of the homogeneous fluid (ρ = 1) about an equatorial axis and about ẑ'''
    a, c = axes()
    return (4*np.pi/15)*(a**2 + c**2), (8*np.pi/15)*a**2


def pressure_weights():
    '''{(kind, l): coefficients of (f, f′, f″, f‴) at r = 1}: p₂ on the wall (reduced pressure, Y₂¹ coefficient)'''
    a, c = axes()
    beta = 1/c**2 - 1/a**2
    w = {'lap': 1.0, 'dzz': a**2*beta, 'zlapz': c**2/a**2 - 1, 'zdzzz': (c**2 - a**2)*beta}
    out = {}
    for key, parts in _PRESSURE.items():
        out[key] = -(par.ViscosD/6)*sum(w[k]*np.array(v, dtype=complex) for k, v in parts.items())
    return out


def pressure_factor():
    '''Γ_P' (conv.) = pressure_factor() p₂: -i (a/c - c/a) sqrt(2/3) (4π/5)/(-1/√2)'''
    a, c = axes()
    return -1j*(a/c - c/a)*np.sqrt(2/3)*(4*np.pi/5)/_N1


def viscous_weights():
    '''{(kind, l): coefficients of (f, f′, f″) at r = 1}: the viscous torque on the mantle, conv.'''
    a, c = axes()
    out = {}
    for key, parts in _VISCOUS.items():
        v = np.array(parts['sin2'], dtype=complex)/a**2 + np.array(parts['cos2'], dtype=complex)/c**2
        out[key] = -(par.ViscosD/a)*v/_N1
    return out


def slips():
    '''NEW (2026-11-09): whether the fluid slips along the wall: a stress-free wall (bco = 0) or an inviscid fluid'''
    return bool(par.bco == 0 or par.ViscosD == 0)


def slip_weights():
    '''
    NEW (2026-11-09): {(kind, l): (cA, cB)}, the torques on the mantle of a flow that slips along the wall (slips()), conv.,
    as coefficients of (f, f′) at r = 1: cA, the Coriolis force's part of Γ_P', pressure_factor() (G/(3c)) [zxw], plus the
    terms in w of Γ_ν, -(ν/a)([lapw]/a² + β [dzzw]); cB, the factor of λ, the inertia's part of Γ_P', pressure_factor()
    (1/6)(a² [w] + (c² - a²) [zwz]) (the header of this file). Empty on a no-slip wall, where these vanish.
    '''
    if not slips():
        return {}
    a, c = axes()
    beta = 1/c**2 - 1/a**2
    pf, G = pressure_factor(), par.Gaspard
    out = {}
    for key in set(_PRESSURE_SLIP) | set(_VISCOUS_SLIP):
        cA, cB = np.zeros(2, dtype=complex), np.zeros(2, dtype=complex)
        if key in _PRESSURE_SLIP:
            t = {k: np.array(v, dtype=complex) for k, v in _PRESSURE_SLIP[key].items()}
            cA += pf*(G/(3*c))*t['zxw']
            cB += pf*(a*a*t['w'] + (c*c - a*a)*t['zwz'])/6
        if key in _VISCOUS_SLIP:
            t = {k: np.array(v, dtype=complex) for k, v in _VISCOUS_SLIP[key].items()}
            cA += -(par.ViscosD/a)*(t['lapw']/a**2 + beta*t['dzzw'])/_N1
        out[key] = (cA, cB)
    return out


def centrifugal_coefficient():
    '''Γ_cent (conv.) = centrifugal_coefficient() Mc'''
    a, c = axes()
    return -1j*par.Gaspard*(4*np.pi/15)*(a**2 - c**2)


def _column(kind, l):
    '''first column of the Chebyshev block of P (kind 'P') or T ('T') of degree l, and the Chebyshev polynomials of that
    parity and their r-derivatives at r = 1 (rows: polynomials, columns: derivative order)'''
    import bc_variables as bv
    if kind == 'P':
        col = ut.offset['u'] + int(ut.N1*(l - ut.m_top)/2)
        ix = (par.m + 1 - ut.s) % 2
    else:
        col = ut.offset['v'] + int(ut.N1*(l - ut.m_bot)/2)
        ix = (par.m + ut.s) % 2
    Tb = bv.Tb if par.ricb > 0 else bv.Tb[ix::2, :]
    return col, Tb


def torque_rows():
    '''[(column array, coefficient array)]: the torque of the reduced pressure and the viscous torque, Γ_P' + Γ_ν, as a
    linear form on the Chebyshev coefficients of P₂ to P₆ and T₁ to T₅ (only the degrees within lmax)'''
    lp, lt, _ = ut.ell(par.m, par.lmax, par.symm)
    pw, vw, pf = pressure_weights(), viscous_weights(), pressure_factor()
    sw = slip_weights()
    out = []
    for (kind, l) in _PRESSURE:
        if l not in (lp if kind == 'P' else lt):
            continue
        col, Tb = _column(kind, l)
        coef = np.zeros(ut.N1, dtype=complex)
        for k, wk in enumerate(pw[(kind, l)]):
            coef += pf*wk*Tb[:, k]
        for k, wk in enumerate(vw[(kind, l)]):
            coef += wk*Tb[:, k]
        if (kind, l) in sw:   # NEW (2026-11-09): a slipping fluid's Coriolis force and viscous wall terms
            coef += Tb[:, :2] @ sw[(kind, l)][0]
        out.append((col + np.arange(ut.N1), coef))
    return out


def inertia_rows():
    '''NEW (2026-11-09): [(column array, coefficient array)]: the factor of λ in the torque of a slipping fluid (slip_weights,
    cB), the inertia of its flow along the wall in the reduced pressure, on the Chebyshev coefficients of P₂, P₄, T₁, T₃'''
    lp, lt, _ = ut.ell(par.m, par.lmax, par.symm)
    out = []
    for (kind, l), (_, cB) in slip_weights().items():
        if l not in (lp if kind == 'P' else lt) or not np.any(cB):
            continue
        col, Tb = _column(kind, l)
        out.append((col + np.arange(ut.N1), Tb[:, :2] @ cB))
    return out


# ------------------------------------------------------------------------------------- the field (magnetic = 1, NEW 2026-11-09)
# The fields on the sphere r = R of the reference space at the Gauss-Legendre points x = cos θ, in kore's conventions: the
# perturbation b̃ = Σ ∇×∇×(f_l Y_l r) + ∇×(g_l Y_l r) (Y_l the Schmidt harmonic of m = 1 times e^{iφ}; b_r = L f/r Y,
# b_θ = (f' + f/r) ∂θY + g i Y/sinθ, b_φ = (f' + f/r) i Y/sinθ - g ∂θY), the background B̃0 = Σ ∇×∇×(h_n P_n r) + ∇×(k_n P_n r).
# Pure functions of numpy arrays (angular.py only), which tests/wobble/derive_field.py checks without a run.

def _field_grid(lmax, nb, npts):
    '''Gauss-Legendre points and weights in x = cos θ, and on them Y_l, ∂θY_l, Y_l/sinθ (m = 1, l <= lmax) and P_n, ∂θP_n
    (m = 0, n <= nb)'''
    import angular
    x, w = np.polynomial.legendre.leggauss(npts)
    Y, dY, Z = angular.ylm_grid(1, lmax, x)
    P, dP, _ = angular.ylm_grid(0, nb, x)
    return {'x': x, 'w': w, 's': np.sqrt(1 - x*x), 'Y': Y, 'dY': dY, 'V': Z.imag, 'P': P, 'dP': dP}


def _zcross(x, s, R, v, vs, sdt, m):
    '''∇v_z × ẑ = (s g_φ, x g_φ, -s g_r - x g_θ), components (r, θ, φ), for g = ∇v_z, v_z = x v_r - s v_θ (s = sinθ), from
    v = (v_r, v_θ, ∂r v_r, ∂r v_θ, ∂θ v_r), vs = v_r/sinθ (m = 1 only) and sdt = sinθ ∂θ v_θ on the sphere R; m 1 or 0'''
    vr, vt, dvr, dvt, tvr = v
    vz = x*vr - s*vt
    gr = x*dvr - s*dvt
    gt = (-s*vr + x*tvr - x*vt - sdt)/R
    if m == 0:
        return 0*gr, 0*gr, -s*gr - x*gt
    vzs = x*vs - vt                                       # v_z/sinθ (v_r/sinθ is regular for m = 1)
    return 1j*vz/R, 1j*x*vzs/R, -s*gr - x*gt


def _wall_fields(gr, R, a, c, lf, Fj, lg, Gj, B0):
    '''
    The perturbation and the background on the sphere R, for the field of degrees lf (poloidal, jets Fj[..., k, :] = f, f', f''
    at R of the degree lf[k]) and lg (toroidal, Gj[..., k, :] = g, g'), batched over the leading axis of Fj, Gj, and the
    background B0 = {'h': {n: (h, h', h'')}, 'k': {n: (k, k')}} at R: {'b': (b_r, b_θ, b_φ), 'J': ∇×(Q b̃), 'B': B̃0,
    'J0': ∇×(Q B̃0), 'Bz': B̃0_z, 'bz': b̃_z}, components (r, θ, φ) of the reference space, Q = diag(a², a², c²) (a = c = 1:
    the sphere). ∇×(Q v) = a² ∇×v + (c² - a²) ∇v_z × ẑ.
    '''
    x, s = gr['x'], gr['s']
    dl = c*c - a*a
    E = lambda u, A: u @ A
    lf, lg = np.asarray(lf, dtype=int), np.asarray(lg, dtype=int)
    Lf, Lg = lf*(lf + 1.), lg*(lg + 1.)
    F0, F1, F2 = Fj[..., 0], Fj[..., 1], Fj[..., 2]
    G0, G1 = Gj[..., 0], Gj[..., 1]
    Yf, dYf, Vf = gr['Y'][lf], gr['dY'][lf], gr['V'][lf]
    Yg, dYg, Vg = gr['Y'][lg], gr['dY'][lg], gr['V'][lg]
    S0 = F1 + F0/R                          # f' + f/r
    S1 = F2 + F1/R - F0/R**2                # its r-derivative
    Df = F2 + 2*F1/R - Lf*F0/R**2           # ∇²f, minus the toroidal scalar of ∇×b
    q = Lf*F0/R
    br, brs = E(q, Yf), E(q, Vf)
    bt = E(S0, dYf) + 1j*E(G0, Vg)
    bp = 1j*E(S0, Vf) - E(G0, dYg)
    dbr = E(Lf*(F1/R - F0/R**2), Yf)
    dbt = E(S1, dYf) + 1j*E(G1, Vg)
    tbr = E(q, dYf)
    sdt = E(S0, -x*dYf - Lf[:, None]*s*Yf + Vf) + 1j*E(G0, dYg - x*Vg)   # sinθ ∂θ b_θ (Legendre's equation, m = 1)
    zc = _zcross(x, s, R, (br, bt, dbr, dbt, tbr), brs, sdt, 1)
    Gp = G1 + G0/R
    J = (a*a*E(Lg*G0/R, Yg) + dl*zc[0],
         a*a*(E(Gp, dYg) - 1j*E(Df, Vf)) + dl*zc[1],
         a*a*(1j*E(Gp, Vg) + E(Df, dYf)) + dl*zc[2])
    z = 0*x
    Br, Bt, Bp, dBr, dBt, tBr, sdB, J0r, J0t, J0p = (z.copy() for _ in range(10))
    for n, (h0, h1, h2) in B0.get('h', {}).items():
        Ln, P, dP = n*(n + 1.), gr['P'][n], gr['dP'][n]
        Br = Br + Ln*h0/R*P
        Bt = Bt + (h1 + h0/R)*dP
        dBr = dBr + Ln*(h1/R - h0/R**2)*P
        dBt = dBt + (h2 + h1/R - h0/R**2)*dP
        tBr = tBr + Ln*h0/R*dP
        sdB = sdB + (h1 + h0/R)*(-x*dP - Ln*s*P)
        J0p = J0p + (h2 + 2*h1/R - Ln*h0/R**2)*dP
    for n, (k0, k1) in B0.get('k', {}).items():
        Ln, P, dP = n*(n + 1.), gr['P'][n], gr['dP'][n]
        Bp = Bp - k0*dP
        J0r = J0r + Ln*k0/R*P
        J0t = J0t + (k1 + k0/R)*dP
    zc0 = _zcross(x, s, R, (Br, Bt, dBr, dBt, tBr), None, sdB, 0)
    J0 = (a*a*J0r + dl*zc0[0], a*a*J0t + dl*zc0[1], a*a*J0p + dl*zc0[2])
    return {'b': (br, bt, bp), 'J': J, 'B': (Br, Bt, Bp), 'J0': J0, 'Bz': x*Br - s*Bt, 'bz': x*br - s*bt}


def _lorentz(fl):
    '''the covariant Lorentz force ∇×(Q b̃) × B̃0 + ∇×(Q B̃0) × b̃ (without Hendrik), components (r, θ, φ)'''
    (Jr, Jt, Jp), (Br, Bt, Bp) = fl['J'], fl['B']
    (Kr, Kt, Kp), (br, bt, bp) = fl['J0'], fl['b']
    return (Jt*Bp - Jp*Bt + Kt*bp - Kp*bt,
            Jp*Br - Jr*Bp + Kp*br - Kr*bp,
            Jr*Bt - Jt*Br + Kr*bt - Kt*br)


def _pressure_lorentz(gr, fl):
    '''p₂ of the Lorentz force on the wall (without Hendrik): the coefficient of Y₂¹ of p with ∇_H p = F_H, the projection of
    the horizontal force on ∇_H(Y₂ e^{iφ}) over its norm 6 ∫Y₂² dΩ'''
    _, Ft, Fp = _lorentz(fl)
    w = gr['w']
    return (Ft*gr['dY'][2] - 1j*Fp*gr['V'][2]) @ w/(6*np.sum(w*gr['Y'][2]**2))


def _maxwell(gr, fl, R, a, c):
    '''
    The torque of the field on the fluid inside the wall r = R (without Hendrik), the Maxwell stress through it,
    ∮ X × [B0 (b⋅n) + b (B0⋅n) - (B0⋅b) n] dS, in the body X = D R r̂, n dS = R² D⁻ᵀ r̂ dΩ, B = D B̃ (D = diag(a, a, c), det 1):
        R³ ∮ { D⁻ᵀ(r̂×B̃0) b̃_r + D⁻ᵀ(r̂×b̃) B̃0_r - (B̃0⋅Q b̃) (a/c - c/a) z (y, -x, 0) } dΩ,
    as L_x - i L_y divided by Y₁¹ at the equator (conv.); in the sphere that of rotdyn.wall_torques
    '''
    x, s, w = gr['x'], gr['s'], gr['w']
    (br, bt, bp), (Br, Bt, Bp) = fl['b'], fl['B']
    dot = a*a*(Br*br + Bt*bt + Bp*bp) + (c*c - a*a)*fl['Bz']*fl['bz']
    dens = ((-x*Bp - 1j*Bt)*br + (-x*bp - 1j*bt)*Br)/a - 1j*(a/c - c/a)*x*s*dot
    return 2*np.pi*R**3/_N1*(dens @ w)


def field_background(R):
    '''B0 at r = R for _wall_fields: {'h': {n: (h, h', h'')}, 'k': {n: (k, k')}} (cnorm included), and the largest degree'''
    import background_field as bf
    rr = np.array([R], dtype=float)
    val = lambda f, d: float(np.ravel(f(rr, d))[0])
    B0 = {'h': {n: tuple(val(f, d) for d in range(3)) for n, f in bf.pol_profiles().items()},
          'k': {n: tuple(val(f, d) for d in range(2)) for n, f in bf.tor_profiles().items()}}
    return B0, max([1] + list(B0['h']) + list(B0['k']))


_FIELD = {}

def field_weights():
    '''
    {(sec, l): (cP, cM)}: the Lorentz part of p₂ (cP, with Hendrik) and the torque of the field on the fluid (cM, with Hendrik,
    conv.), as linear forms on (f, f', f'') (sec 'f') or (g, g') ('g') of the degree l at the wall; only the degrees that
    couple (|Δl| <= nb + 6 about the degrees 2 and 1), and the coefficients above round-off (cached)
    '''
    if 'w' in _FIELD:
        return _FIELD['w']
    a, c = axes()
    R = ut.rcmb
    B0, nb = field_background(R)
    lp, lt, _ = ut.ell(par.m, par.lmax, ut.bsymm)
    top = min(par.lmax, 2 + nb + 6)
    lf, lg = lp[lp <= top], lt[lt <= top]
    gr = _field_grid(top + 1, nb, top + 2*nb + 16)
    out = {}
    for sec, ls, nd in (('f', lf, 3), ('g', lg, 2)):
        K = len(ls)*nd
        Fj = np.zeros((K, len(lf), 3), dtype=complex)
        Gj = np.zeros((K, len(lg), 2), dtype=complex)
        for k in range(len(ls)):
            for d in range(nd):
                (Fj if sec == 'f' else Gj)[k*nd + d, k, d] = 1.
        fl = _wall_fields(gr, R, a, c, lf, Fj, lg, Gj, B0)
        cP = par.Hendrik*_pressure_lorentz(gr, fl).reshape(len(ls), nd)
        cM = par.Hendrik*_maxwell(gr, fl, R, a, c).reshape(len(ls), nd)
        for C in (cP, cM):   # the degrees outside the couplings: round-off, which the derivatives at the wall would amplify
            C[np.abs(C) < 1e-11*np.maximum(np.abs(C).max(axis=0), 1e-300)] = 0
        for k, l in enumerate(ls):
            out[(sec, int(l))] = (cP[k], cM[k])
    _FIELD['w'] = out
    return out


def _field_column(sec, l):
    '''first column of the Chebyshev block of f (sec 'f') or g ('g') of degree l, and the Chebyshev polynomials of its parity
    and their r-derivatives at r = 1 (as assemble.bc_b_spherical)'''
    import bc_variables as bv
    lfirst = ut.m_bot if (sec == 'f') == (ut.symmB0 == -1) else ut.m_top
    col = ut.offset[sec] + int(ut.N1*(l - lfirst)/2)
    Tb = bv.Tb if par.ricb > 0 else bv.Tb[l % 2::2, :]
    return col, Tb


def field_rows():
    '''[(column array, coefficient array)]: the torques of the field on the mantle, Γ_P'(Lorentz) + Γ_B, with Γ_P'(Lorentz)
    = pressure_factor() p₂(Lorentz) and Γ_B = -(the torque of the field on the fluid), as a linear form on the Chebyshev
    coefficients of f and g'''
    if not par.magnetic:
        return []
    pf = pressure_factor()
    out = []
    for (sec, l), (cP, cM) in field_weights().items():
        cw = pf*cP - cM
        if not np.any(cw):
            continue
        col, Tb = _field_column(sec, l)
        out.append((col + np.arange(ut.N1), Tb[:, :cw.size] @ cw))
    return out


def a_rows(rank0):
    '''
    The row of A for Mc (A = -(the right-hand side), as everywhere in assemble.py): i G [e A_M + (4π/15)(a² - c²)] on Mc
    and minus the torques of the flow and (NEW 2026-11-09) of the field; empty on the ranks other than rank0
    '''
    if not (ut.wobble and rank0):
        return [np.array([], dtype=complex), np.array([], dtype=int), np.array([], dtype=int)]
    iW = ut.offset['wob']
    data = [np.array([1j*par.Gaspard*par.dynflat_M*par.MoIX_M - centrifugal_coefficient()])]
    rows = [np.array([iW])]
    cols = [np.array([iW])]
    for c, v in torque_rows() + field_rows():
        data.append(-v); rows.append(np.full(c.size, iW)); cols.append(c)
    return [np.concatenate(data), np.concatenate(rows), np.concatenate(cols)]


def b_rows(rank0, inertia_v1):
    '''
    The entries of B (B = -(the factor of λ)): -A_M on Mc, and the Poincaré force in the rows of T of degree 1, -(the
    inertia of the rigid rotation T₁ = r/(2a) per unit Mc), its column; inertia_v1 = op.inertia(1, 'v', 'utor', 0)
    '''
    if not (ut.wobble and rank0):
        return [np.array([]), np.array([], dtype=int), np.array([], dtype=int)]
    iW = ut.offset['wob']
    a, _ = axes()
    cT = ut.chebco_f(lambda r: r/(2*a), par.N, par.ricb, ut.rcmb, 1e-15)
    if par.ricb == 0:
        cT = cT[(par.m + ut.s) % 2::2]
    col = np.asarray(-(inertia_v1 @ cT)).ravel()
    row0 = ut.offset['v'] + int(ut.N1*(1 - ut.m_bot)/2)
    nz = np.nonzero(col)[0]
    data, rows, cols = [[-par.MoIX_M], col[nz]], [[iW], row0 + nz], [[iW], np.full(nz.size, iW)]
    for c, v in inertia_rows():   # NEW (2026-11-09): λ times the inertia's torque of a slipping fluid, on the row of Mc
        data.append(v); rows.append(np.full(c.size, iW)); cols.append(c)
    return [np.concatenate(data), np.concatenate(rows), np.concatenate(cols)]


# ------------------------------------------------------------------------------------------------- post-processing
def budget(flow, Mc, lam, field=None):
    '''
    The torques on the mantle and the angular momentum budget of the fluid for one solution: flow, the flow's unknowns (2n),
    Mc, the eigenvalue lam, and (NEW 2026-11-09, magnetic = 1) field, the field's (f, g: 2n). Returns a dict: 'pressure'
    (Γ_P', reduced, with the Lorentz part), 'centrifugal' (Γ_cent), 'viscous' (Γ_ν), 'total', 'L' (the fluid's, conv.),
    'resid' |λ(A_f Mc + L) + iG(C_f - A_f)Mc - iG L + Γ| / (|λ| |L| + |Γ|), and with a field 'pressure_lorentz' (the Lorentz
    part of Γ_P') and 'magnetic' (Γ_B, minus the torque of the field on the fluid, from the Maxwell stress through the wall;
    in 'total'): the budget then holds with the torque of the field on the fluid, -Γ_B, and checks Γ_P' (from the force on
    the wall) against it (from the stress through the wall).
    '''
    x = np.zeros(ut.sizmat, dtype=complex)
    x[:2*ut.n] = flow
    x[ut.offset['wob']] = Mc
    if field is not None:
        x[ut.offset['f']:ut.offset['f'] + 2*ut.n] = field
    pw, vw, pf = pressure_weights(), viscous_weights(), pressure_factor()
    lp, lt, _ = ut.ell(par.m, par.lmax, par.symm)
    gp = gv = 0j
    for (kind, l) in _PRESSURE:
        if l not in (lp if kind == 'P' else lt):
            continue
        col, Tb = _column(kind, l)
        coefs = x[col:col + ut.N1]
        jets = Tb.T @ coefs
        gp += pf*np.dot(pw[(kind, l)], jets[:4])
        gv += np.dot(vw[(kind, l)], jets[:3])
    for (kind, l), (cA, cB) in slip_weights().items():   # NEW (2026-11-09): a slipping fluid (Coriolis and inertia in Γ_P',
        if l not in (lp if kind == 'P' else lt):         # the wall terms of Γ_ν: those of cA that are not pressure)
            continue
        col, Tb = _column(kind, l)
        jets = Tb.T[:2] @ x[col:col + ut.N1]
        cP = pf*(par.Gaspard/(3*axes()[1]))*np.array(_PRESSURE_SLIP[(kind, l)]['zxw'], dtype=complex) \
            if (kind, l) in _PRESSURE_SLIP else np.zeros(2, dtype=complex)
        gp += np.dot(cP, jets) + lam*np.dot(cB, jets)
        gv += np.dot(cA - cP, jets)
    gc = centrifugal_coefficient()*Mc
    a, c = axes()
    col, _ = _column('T', 1)
    t1 = x[col:col + ut.N1]
    if par.ricb == 0:   # one parity per l in a full sphere: back to all the Chebyshev polynomials
        full = np.zeros(par.N, dtype=complex)
        full[(par.m + ut.s) % 2::2] = t1
        t1 = full
    xg, wg = np.polynomial.legendre.leggauss(par.N + 24)
    r = 0.5*(ut.rcmb - par.ricb)*(xg + 1) + par.ricb
    T = ut.funcheb(t1, r, par.ricb, ut.rcmb, 0)[:, 0]
    L = (16*np.pi/3)/a*0.5*(ut.rcmb - par.ricb)*np.sum(wg*r**3*T)
    Af, Cf = fluid_inertia()
    G = par.Gaspard
    extra = {}
    gb = 0j
    if field is not None and par.magnetic:
        gl = 0j
        for (sec, l), (cP, cM) in field_weights().items():
            col, Tb = _field_column(sec, l)
            jets = Tb.T[:cP.size] @ x[col:col + ut.N1]
            gl += pf*np.dot(cP, jets)
            gb -= np.dot(cM, jets)
        gp += gl
        extra = {'pressure_lorentz': gl, 'magnetic': gb}
    tot = gp + gc + gv + gb
    lhs = lam*(Af*Mc + L) + 1j*G*(Cf - Af)*Mc - 1j*G*L
    return {'pressure': gp, 'centrifugal': gc, 'viscous': gv, 'total': tot, 'L': L,
            'resid': abs(lhs + tot)/(abs(lam)*abs(L) + abs(tot) + abs(gb) + 1e-300), **extra}


def energies(flow, Mc, lam, field=None):
    '''
    Energies and powers of one solution for the comparison with Triana et al. (2019), with the sphere's metric (their error
    is of order f): all are ∫|.|² of the complex amplitudes (no factor 1/2), so that only their ratios are meant.
      KE      the kinetic energy of the flow in the frame of the mantle;
      KE_T11  that of its toroidal part of degree 1;
      KE_srf  that of the flow in the steadily rotating frame (the frame turning at Gaspard about the axis fixed in space):
              the flow plus the rigid rotation Ms of the mantle relative to it, Ms = Mc λ/(λ - i G) (T₁ + r Ms/2);
      KM_srf  the kinetic energy of the mantle in that frame, A_M |Ms|²/4 (its rigid rotation's on the same scale);
      Dkin    ν ∫|∇×u|², the viscous dissipation on a no-slip wall (not with a slipping fluid);
      DP      the power of the Poincaré force on the flow, |λ Mc* L|/4 (L the angular momentum of the flow, conv.).
    With the torques of budget (field: the field's unknowns with magnetic = 1, NEW 2026-11-09).
    '''
    import numpy.polynomial.chebyshev as ch
    lp, lt, _ = ut.ell(par.m, par.lmax, par.symm)
    nb, N1 = len(lp), ut.N1
    xg, wg = np.polynomial.legendre.leggauss(par.N + 32)
    r = 0.5*(xg + 1)*ut.rcmb; wr = 0.5*wg*ut.rcmb
    ip, it = (par.m + 1 - ut.s) % 2, (par.m + ut.s) % 2
    def prof(c, ix):
        full = np.zeros(par.N, dtype=complex)
        full[ix::2] = c
        return ut.funcheb(full, r, 0, ut.rcmb, 2)
    G = par.Gaspard
    Ms = Mc*lam/(lam - 1j*G)
    KE = KE_T11 = KE_srf = Dkin = 0.
    for k, l in enumerate(lp):
        F = prof(flow[k*N1:(k + 1)*N1], ip)
        P, dP, d2P = F[:, 0], F[:, 1], F[:, 2]
        L, nl = l*(l + 1), 4*np.pi/(2*l + 1)
        e = L*nl*np.sum(wr*(L*abs(P)**2 + abs(P + r*dP)**2))
        KE += e; KE_srf += e
        Dkin += L*nl*np.sum(wr*abs(d2P*r + 2*dP - L*P/r)**2)
    for k, l in enumerate(lt):
        F = prof(flow[(nb + k)*N1:(nb + k + 1)*N1], it)
        T, dT = F[:, 0], F[:, 1]
        L, nl = l*(l + 1), 4*np.pi/(2*l + 1)
        e = L*nl*np.sum(wr*abs(T)**2*r**2)
        KE += e
        Dkin += L*nl*np.sum(wr*(L*abs(T)**2 + abs(T + r*dT)**2))
        if l == 1:
            KE_T11 = e
            KE_srf += L*nl*np.sum(wr*abs(T + r*Ms/2)**2*r**2)
        else:
            KE_srf += e
    b = budget(flow, Mc, lam, field)
    return {'KE': KE, 'KE_T11': KE_T11, 'KE_srf': KE_srf, 'KM_srf': par.MoIX_M*abs(Ms)**2/4,
            'Dkin': par.ViscosD*Dkin, 'DP': abs(lam*np.conj(Mc)*b['L'])/4, **b}
