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

LEFT OUT (background.check_wobble stops): an inner core (full sphere only), a field (its Lorentz force at the wall and the
Maxwell stress on a conducting mantle would add torques), buoyancy (a moving mass adds gravitational torques), the other
shapes of centrifugal = 1 (whose wall is not the sphere of a linear map), stress-free walls, forcings.
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
        out.append((col + np.arange(ut.N1), coef))
    return out


def a_rows(rank0):
    '''
    The row of A for Mc (A = -(the right-hand side), as everywhere in assemble.py): i G [e A_M + (4π/15)(a² - c²)] on Mc
    and minus the torques of the flow; empty on the ranks other than rank0
    '''
    if not (ut.wobble and rank0):
        return [np.array([], dtype=complex), np.array([], dtype=int), np.array([], dtype=int)]
    iW = ut.offset['wob']
    data = [np.array([1j*par.Gaspard*par.dynflat_M*par.MoIX_M - centrifugal_coefficient()])]
    rows = [np.array([iW])]
    cols = [np.array([iW])]
    for c, v in torque_rows():
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
    return [np.concatenate([[-par.MoIX_M], col[nz]]), np.concatenate([[iW], row0 + nz]),
            np.concatenate([[iW], np.full(nz.size, iW)])]


# ------------------------------------------------------------------------------------------------- post-processing
def budget(flow, Mc, lam):
    '''
    The torques on the mantle and the angular momentum budget of the fluid for one solution: flow, the flow's unknowns (2n),
    Mc, the eigenvalue lam. Returns a dict: 'pressure' (Γ_P', reduced), 'centrifugal' (Γ_cent), 'viscous' (Γ_ν), 'total',
    'L' (the fluid's, conv.), 'resid' |λ(A_f Mc + L) + iG(C_f - A_f)Mc - iG L + Γ| / (|λ| |L| + |Γ|).
    '''
    x = np.zeros(ut.sizmat, dtype=complex)
    x[:2*ut.n] = flow
    x[ut.offset['wob']] = Mc
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
    tot = gp + gc + gv
    lhs = lam*(Af*Mc + L) + 1j*G*(Cf - Af)*Mc - 1j*G*L
    return {'pressure': gp, 'centrifugal': gc, 'viscous': gv, 'total': tot, 'L': L,
            'resid': abs(lhs + tot)/(abs(lam)*abs(L) + abs(tot) + 1e-300)}


def energies(flow, Mc, lam):
    '''
    Energies and powers of one solution for the comparison with Triana et al. (2019), with the sphere's metric (their error
    is of order f): all are ∫|.|² of the complex amplitudes (no factor 1/2), so that only their ratios are meant.
      KE      the kinetic energy of the flow in the frame of the mantle;
      KE_T11  that of its toroidal part of degree 1;
      KE_srf  that of the flow in the steadily rotating frame (the frame turning at Gaspard about the axis fixed in space):
              the flow plus the rigid rotation Ms of the mantle relative to it, Ms = Mc λ/(λ - i G) (T₁ + r Ms/2);
      KM_srf  the kinetic energy of the mantle in that frame, A_M |Ms|²/4 (its rigid rotation's on the same scale);
      Dkin    the viscous dissipation ν ∫|∇×u|² (no-slip wall);
      DP      the power of the Poincaré force on the flow, |λ Mc* L|/4 (L the angular momentum of the flow, conv.).
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
    b = budget(flow, Mc, lam)
    return {'KE': KE, 'KE_T11': KE_T11, 'KE_srf': KE_srf, 'KM_srf': par.MoIX_M*abs(Ms)**2/4,
            'Dkin': par.ViscosD*Dkin, 'DP': abs(lam*np.conj(Mc)*b['L'])/4, **b}
