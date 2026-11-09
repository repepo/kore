'''
NEW (2026-11-06, from conductive_IC: its topo.py, the drag rows of its assemble.py, utils.check_drag and
utils4pp.cmb_slip_sq, cmb_slip_K): a drag at a stress-free outer wall (bco = 0), the form drag of the flow over the
wall's topography. The wall exerts on the fluid the tangential stress -K (u_t - v_m), v_m = omega_M z x r the velocity of
the mantle (rotdyn = 1; 0 otherwise), a Robin condition between stress-free (K = 0) and no-slip (K -> inf). K is in units
of rho_w Omega r0, rho_w the fluid's density at the wall: the constant kappa_cmb, the same in every direction, or, with
topo_drag = 'rekier', the tensor diag(kappa_tt, kappa_pp)(theta) of the internal waves the flow radiates over the
topography (Rekier, Triana, Barik, Abdulah & Kang 2025, arXiv:2507.01671, eqs. 20, 21, B1, B2, with the factor 4 pi^2 that
the printed B1 of v4 lacks: it is lost between A19 and 10, and with it a sinusoidal ridge gives Bell's (1975)
rho U^2 h^2 k sqrt(N^2 - w^2)/4).

For a tangential flow of amplitudes U_x (zonal) and U_y (meridional) over topography of spectrum S(k), B1 with that factor
gives the mean power per unit area
    P = rho I_k/(2 |w0|) int_0^2pi (U_x^2 cos^2 a + U_y^2 sin^2 a) g(a) da,
    g = sqrt(max(0, (w0^2 - N^2)(f^2 - w0^2) + w0^2 ft^2 sin^2 a)),  f = 2 Omega cos(theta), ft = 2 Omega sin(theta),
    I_k = int S k^2 dk = h_rms^2 k0 Gamma(H - 1/2)/(4 sqrt(pi) Gamma(H)),  k0 = 2 pi/lambda0          (B2)
(theta the colatitude). There is no zonal-meridional cross term, so a linear drag of the same mean power K <u^2> is the
diagonal tensor acting on (u_theta, u_phi), in units of rho Omega r0:
    kappa_pp = (I_k/r0)/|w0/Omega| int cos^2 a g/Omega^2 da,  kappa_tt = the same with sin^2 a.
rho and r0 cancel: the inputs are topo_hrms = h_rms/r0, topo_lambda0 = lambda0/r0 and topo_H, N the buoyancy frequency at the
wall (N_cmb) and w0 the frequency of the flow relative to the mantle (omega0).

The rows. Per degree l the tangential flow at the wall is u_t = s_l grad1(Y_l) + t_l grad1(Y_l) x r_hat (grad1 the gradient on
the unit sphere; s = P' + P/r + (ln rho0)' P, t = T), and the stress rows of assemble.bc_u_spherical are r^2 (section u,
second outer row) and r (section v, first outer row) times the parts of the viscous stress sigma_t/(rho0 nu ViscosD) on
grad1(Y_l) and grad1(Y_l) x r_hat. In the units of the run the condition is
    rho0 nu ViscosD sigma_t/(rho0 nu ViscosD) = -Gaspard rho_w K (u_t - v_m),  i.e.  sigma_t + q K (u_t - v_m) = 0,
    q = Gaspard/(ViscosD nu(1))   (rho0 = rho_w cancels; nu the viscosity profile of the structure),
and projected on grad1(Y_l) and grad1(Y_l) x r_hat it couples the row of degree l to s and t of every degree l':
    section u:  (stress row)_l + r^2 q sum_l' ( A_ss[l, l'] s_l' + A_st[l, l'] t_l' ) = 0,
    section v:  (stress row)_l + r   q sum_l' ( A_ts[l, l'] s_l' + A_tt[l, l'] t_l' ) = 0,
    A_xy[l, l'] = oint conj(x_l).K.y_l' dS / oint |x_l|^2 dS,  a_l = grad1(Y_l) (for s), b_l = grad1(Y_l) x r_hat (for t),
on the unit sphere, by Gauss-Legendre quadrature (couplings). A constant K gives A_ss = A_tt = K, A_st = A_ts = 0 and no
coupling between degrees; K(theta) couples l to l +- 2, ... within P and within T, and for m != 0 P_l to T_(l+-1), ...
(the whole tensor also when it is isotropic but depends on theta). With the rotating mantle (m = 0) t_1 is T_1 - r omega_M:
the column of omega_M.

spin_doctor: Dtopo = 2 Gaspard rho_w oint (u_t - v_m)*.K.(u_t - v_m) dS, the power the drag dissipates, in the units of its
energies (the complex amplitude's int rho0 |u|^2, whose rate is 2 sigma times it), and dsig = -Dtopo/(2 E) the damping rate
it implies (E the energy of the mode). For a stress-free outer wall without a drag both are per unit kappa_cmb (K = 1): the
damping a weak drag would add, to first order.

NEW (2026-11-09): a deformed body (centrifugal = 1). The wall is then the level surface r = 1 of the map X(alpha), the
unknown the reference flux w = J F^-1 u (centrifugal.py), and the stress rows hold the traction of that surface,
[F^T sigma_X N]_t with N = F^-T r_hat = grad r (not normalised; centrifugal.wall_dissipative_rows). Times |N|, the drag
of the body, -K (u_t - v_m) per unit area of the surface, adds to them t.K_eff.w with
    K_eff = (|N|/J) F^T K F = diag(h_t kappa_tt(x_n), h_p kappa_pp(x_n))  on (w_theta, w_phi),
    h_t = |N| |F theta_hat|^2/J,  h_p = |N| |F phi_hat|^2/J,
exact for the axisymmetric map: F takes theta_hat to the meridian of the surface and phi_hat to phi_hat, so that K_eff is
diagonal in the reference theta, phi too, and the couplings above take it as they take K(theta). By Nanson's formula
(dS_X = J |N| dS) the same K_eff gives the power the drag dissipates on the body's wall. x_n is the cosine of the
colatitude of the wall's normal, where the drag of the topography takes its local vertical (f = 2 Omega x_n,
ft = 2 Omega sqrt(1 - x_n^2)), and the buoyancy frequency there is N_cmb |N| (the effective gravity and dS/dn both scale
with |grad r|). wall_metric gives h_t, h_p, x_n and |N|:
    first order (eps, eta the shape's amplitudes ce0, ce1 at the wall; P2 = (3x^2 - 1)/2):
        h_t = 1 + eps - (3 eps + 2 eta) P2,  h_p = 1 - eps - (eps + 2 eta) P2,  |N| = 1 - (eps + eta) P2,
        x_n = x - (3/2) eps x (1 - x^2);
    the exact spheroid (semi-axes a, a, c, a^2 c = 1; Q = a^2 x^2 + c^2 (1 - x^2)):
        h_t = a Q^(3/2),  h_p = a^3 Q^(1/2),  |N| = a Q^(1/2),  x_n = a x/Q^(1/2).
K is taken at x_n node by node, never expanded about x: the drag of the topography has kinks at the critical latitudes.
A constant kappa_cmb then couples the degrees too (K_eff depends on theta): l to l +- 2 and, for m != 0, P_l to T_(l+-1)
at first order, every degree of the parity for the exact spheroid. topo_hrms and topo_lambda0 are relative to the mean
radius, and the test hook topo_K is a function of x_n. The rotating solids stay spherical (rotdyn = 1 stops with
centrifugal = 1), so v_m = 0 here.
'''
import numpy as np
from scipy.special import gamma
from parameters import par
import utils as ut

_cache = {}


def on():
    '''whether the outer wall has a drag (kappa_cmb or topo_drag)'''
    return bool(par.kappa_cmb != 0 or par.topo_drag is not None)


def wall_profiles():
    '''[rho0, nu] at the outer wall (the density and the viscosity profile of the structure, as rotdyn.viscous_factor)'''
    import radial_profiles as rap
    rr = np.array([float(ut.rcmb)])
    return [float(np.exp(np.ravel(rap.logrhoX(rr, 0))[0])), float(np.ravel(rap.viscoX(rr, 0))[0])]


def ratio():
    '''q = Gaspard/(ViscosD nu(1)), the drag coefficient of the run's units over that of the viscous stress at the wall'''
    return par.Gaspard/(par.ViscosD*wall_profiles()[1])


def N_cmb():
    '''
    N/Omega at the outer wall: topo_N if set, else sqrt(Beyonce g dS/dr)/Gaspard of the stratification (N^2 = Beyonce g dS/dr,
    docs/anelastic.md), 0 without buoyancy (thermal = 0) and where the stratification is unstable (N^2 < 0). NEW (2026-11-08):
    with the composition (compositional = 1) N^2 has its part too, BeyonceC g dC/dr. NEW (2026-11-09): on a deformed body
    (centrifugal = 1) this is N of the reference sphere (with the effective gravity of the buoyancy, radial_profiles.graviX),
    which wall_tensor multiplies by |grad r| along the wall
    '''
    if par.topo_N is not None:
        return float(par.topo_N)
    if not (par.thermal or par.compositional):
        return 0.0
    import radial_profiles as rap
    rr = np.array([float(ut.rcmb)])
    g = float(np.ravel(rap.prf.gravity(rr, 0))[0])
    if par.centrifugal and par.centrifugal_q != 0:   # NEW (2026-11-09): the effective gravity of the buoyancy (graviX)
        import centrifugal as cf
        g += float(np.ravel(cf.g_monopole(rr))[0])
    n2 = par.Beyonce*g*float(np.ravel(rap.prf.gradS(rr))[0]) if par.thermal else 0.0
    if par.compositional:
        n2 += par.BeyonceC*g*float(np.ravel(rap.prf.gradC(rr))[0])
    return float(np.sqrt(max(n2, 0.0)))/par.Gaspard


def omega0():
    '''
    |w0|/Omega, the frequency of the flow relative to the mantle: topo_omega if set, else |itau|/Gaspard, the frequency of the
    solver's target (eigenvalue problems), or |forcing_frequency|/Gaspard (forced problems); the run's frequencies are in units
    of 1/tau, and Omega tau = Gaspard
    '''
    if par.topo_omega is not None:
        return abs(float(par.topo_omega))
    w = par.forcing_frequency if par.forcing != 0 else par.itau
    return abs(w)/par.Gaspard


def Ik_R():
    '''I_k/r0 of eq. (B2) from topo_hrms = h_rms/r0, topo_lambda0 = lambda0/r0 and topo_H'''
    k0R = 2*np.pi/par.topo_lambda0
    return par.topo_hrms**2*k0R*gamma(par.topo_H - 0.5)/(4*np.sqrt(np.pi)*gamma(par.topo_H))


def hook(x):
    '''the test hook topo_K, a numpy expression of x = cos(theta) giving (kappa_tt, kappa_pp), as two arrays of x's shape'''
    kt, kp = eval(par.topo_K, {'np': np, '__builtins__': {}}, {'x': x})
    return (np.broadcast_to(np.asarray(kt, dtype=float), x.shape).copy(),
            np.broadcast_to(np.asarray(kp, dtype=float), x.shape).copy())


def kappa(x, na=2048, n=None):
    '''
    [kappa_tt, kappa_pp] (meridional, zonal; units rho_w Omega r0) at x = cos(colatitude): kappa_cmb in every direction, or
    eq. (B1) with the factor 4 pi^2 (topo_drag = 'rekier'; the a-integral by the midpoint rule, exact to round-off for the
    trigonometric polynomial under the root where it does not vanish), or the test hook topo_K. NEW (2026-11-09): n, the
    buoyancy frequency N/Omega at each x (default N_cmb(); a deformed wall's varies, wall_tensor)
    '''
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if par.topo_drag is None:
        k = par.kappa_cmb if par.kappa_cmb != 0 else 1.0   # 1: per unit kappa_cmb (spin_doctor, a stress-free wall)
        return k + 0*x, k + 0*x
    if par.topo_K:
        return hook(x)
    w0 = omega0()
    n = np.broadcast_to(np.asarray(N_cmb() if n is None else n, dtype=float), x.shape)[:, None]
    a = (np.arange(na) + 0.5)*2*np.pi/na
    f2, ft2 = 4*x**2, 4*(1 - x**2)
    g = np.sqrt(np.maximum(0.0, (w0**2 - n**2)*(f2[:, None] - w0**2) + w0**2*ft2[:, None]*np.sin(a)**2))
    c = Ik_R()/w0*(2*np.pi/na)   # (B1) times 4 pi^2: no 1/(4 pi^2) in front of the integral
    return c*np.sum(np.sin(a)**2*g, axis=1), c*np.sum(np.cos(a)**2*g, axis=1)


def deformed():
    '''NEW (2026-11-09): whether the outer wall is that of a deformed body (centrifugal = 1 with a deformation)'''
    if not par.centrifugal:
        return False
    import centrifugal as cf
    return bool(cf.deformed())


def wall_metric(x):
    '''
    NEW (2026-11-09): the deformed outer wall at the reference nodes x = cos(theta): [h_t, h_p, x_n, |N|] of the module
    docstring (K_eff = diag(h_t kappa_tt(x_n), h_p kappa_pp(x_n)), x_n the cosine of the colatitude of the wall's normal,
    |N| = |grad r|), exactly for the spheroid (centrifugal_shape 'spheroid'), to first order in the shape otherwise; [1, 1, x, 1]
    for a sphere
    '''
    x = np.atleast_1d(np.asarray(x, dtype=float))
    one = np.ones_like(x)
    if not deformed():
        return one, one, x, one
    import centrifugal as cf
    if cf.exact():
        a, c = cf.spheroid_axes()
        Q = a*a*x*x + c*c*(1 - x*x)
        return a*Q**1.5, a**3*np.sqrt(Q), a*x/np.sqrt(Q), a*np.sqrt(Q)
    v = cf.wall_values(float(ut.rcmb))
    e, h = v['E'], v['N']
    P2 = 1.5*x*x - 0.5
    return (1 + e - (3*e + 2*h)*P2, 1 - e - (e + 2*h)*P2, x - 1.5*e*x*(1 - x*x), 1 - (e + h)*P2)


def wall_tensor(x):
    '''
    [K_tt, K_pp] at the reference nodes x: kappa(x) on a sphere; NEW (2026-11-09) on a deformed body the effective tensor
    h_t kappa_tt(x_n), h_p kappa_pp(x_n), with the buoyancy frequency N_cmb |N| of each node (module docstring)
    '''
    if not deformed():
        return kappa(x)
    ht, hp, xn, gr = wall_metric(x)
    n = N_cmb()*gr if (par.topo_drag is not None and not par.topo_K) else None
    kt, kp = kappa(xn, n=n)
    return ht*kt, hp*kp


def grid():
    '''
    The outer wall on Gauss-Legendre nodes in x = cos(theta), cached: the nodes x, the weights, dY_l/dtheta and
    Z_l = i m Y_l/sin(theta) there (l = 0..lmax + 1, angular.ylm_grid), and kappa_tt, kappa_pp (NEW 2026-11-09: those of
    K_eff on a deformed body, wall_tensor). Many nodes,
    max(4 (lmax + 2), 2000), since K has kinks where the root of (B1) vanishes (the critical latitudes). The rule of
    conductive_IC (numpy's leggauss), from scipy's roots_legendre: the couplings agree to 5e-12, ten times faster
    '''
    if 'grid' not in _cache:
        import angular
        from scipy.special import roots_legendre
        x, w = roots_legendre(max(4*(par.lmax + 2), 2000))
        Y, dY, Z = angular.ylm_grid(par.m, par.lmax + 1, x)
        _cache['grid'] = (x, w, dY, Z) + tuple(wall_tensor(x))
    return _cache['grid']


def couplings():
    '''
    The coupling matrices A_ss, A_st, A_ts, A_tt [l, l'] (l, l' = 0..lmax + 1) of the module docstring at the outer wall for
    par.m, cached; zero rows for l < max(m, 1). Couplings that vanish by symmetry are set to 0 (their round-off removed)
    '''
    if 'A' in _cache:
        return _cache['A']
    x, w, dY, Z, kt, kp = grid()                   # dY = dY/dtheta (real), Z = i m Y/sin(theta)
    W = 2*np.pi*w
    nrm = np.sum(W*(dY**2 + np.abs(Z)**2), axis=1)  # oint |grad1 Y_l|^2 = oint |grad1 Y_l x r_hat|^2
    nrm[nrm == 0] = 1.0
    # a = (dY, Z), b = (Z, -dY) on (theta, phi): conj(a_l).K.a_l' = kt dY_l dY_l' + kp conj(Z_l) Z_l', etc.
    Ass = ((W*kt*dY) @ dY.T + (W*kp*np.conj(Z)) @ Z.T)/nrm[:, None]
    Ast = ((W*kt*dY) @ Z.T - (W*kp*np.conj(Z)) @ dY.T)/nrm[:, None]
    Ats = ((W*kt*np.conj(Z)) @ dY.T - (W*kp*dY) @ Z.T)/nrm[:, None]
    Att = ((W*kt*np.conj(Z)) @ Z.T + (W*kp*dY) @ dY.T)/nrm[:, None]
    tol = 1e-13*max(abs(Ass).max(), abs(Att).max(), 1e-300)
    for M in (Ass, Ast, Ats, Att):
        M[abs(M) < tol] = 0
    _cache['A'] = (Ass, Ast, Ats, Att)
    return _cache['A']


def rows(l, sec):
    '''
    The drag's entries in the stress row of the outer wall of degree l, section 'u' (P_l) or 'v' (T_l), as [data, row, col]:
    r^2 q (section u) or r q (section v) times the couplings to s and t of every degree, and the omega_M column with the
    rotating mantle (module docstring); the stress-free part of the row is bc_u_spherical's
    '''
    import bc_variables as bv
    r = float(ut.rcmb)
    if par.ricb > 0:
        Tu = Tv = bv.Tb
    else:   # one parity per kind in a full sphere, as in bc_u_spherical
        Tu = bv.Tb[( par.m + 1 - ut.s )%2::2, :]
        Tv = bv.Tb[( par.m + ut.s )%2::2, :]
    lh1 = bv.lhb1
    lp, lt = ut.ell(par.m, par.lmax, par.symm)[:2]
    if sec == 'u':
        row = ut.offset['u'] + int(ut.N1*(l - ut.m_top)/2) + 1   # the second outer row of the block (the first is P = 0)
        fac = r**2*ratio()
    else:
        row = ut.offset['v'] + int(ut.N1*(l - ut.m_bot)/2)
        fac = r*ratio()
    if par.topo_drag is None and not deformed():   # a constant K on a sphere: the degree itself, A_ss = A_tt = kappa_cmb
        Cs = {l: par.kappa_cmb} if sec == 'u' else {}
        Ct = {l: par.kappa_cmb} if sec == 'v' else {}
    else:
        Ass, Ast, Ats, Att = couplings()
        Cs = dict(enumerate((Ass if sec == 'u' else Ats)[l]))
        Ct = dict(enumerate((Ast if sec == 'u' else Att)[l]))
    data, cols = [], []
    for k in lp:
        if Cs.get(k, 0) != 0:
            data.append( fac*Cs[k]*(Tu[:,1] + Tu[:,0]/r + lh1*Tu[:,0]) )
            cols.append( ut.offset['u'] + int(ut.N1*(k - ut.m_top)/2) + np.arange(Tu.shape[0]) )
    for k in lt:
        if Ct.get(k, 0) != 0:
            data.append( fac*Ct[k]*Tv[:,0] )
            cols.append( ut.offset['v'] + int(ut.N1*(k - ut.m_bot)/2) + np.arange(Tv.shape[0]) )
            if ut.rotdyn and k == 1:   # the slip relative to the mantle: T_1 - r omega_M
                data.append( np.array([ -fac*Ct[k]*r ]) )
                cols.append( np.array([ ut.offset['rot'] ]) )
    if not data:
        return [np.zeros(0, dtype=complex), np.zeros(0, dtype=int), np.zeros(0, dtype=int)]
    data = np.concatenate(data).astype(complex)
    return [data, np.full(data.size, row), np.concatenate(cols)]


# ------------------------------------------------------------------------------------------------- spin_doctor: Dtopo, dsig
def dissipation(usol2, omega_M=0):
    '''
    Dtopo = 2 Gaspard rho_w oint (u_t - v_m)*.K.(u_t - v_m) dS over the outer wall for the flow usol2 (utils4pp.expand_reshape_sol),
    v_m = omega_M z x r; K = 1 in every direction (per unit kappa_cmb) for a stress-free wall without a drag. NEW (2026-11-09):
    on a deformed body the integral over the body's wall, by K_eff over the reference sphere (module docstring)
    '''
    import numpy.polynomial.chebyshev as ch
    import bc_variables as bv
    lp, lt = ut.ell(par.m, par.lmax, par.symm)[:2]
    P, T = usol2[0].T, usol2[1].T   # one column per l
    r = float(ut.rcmb)
    xr = ut.xcheb(np.array([r]), par.ricb, ut.rcmb)
    x, w, dY, Z, kt, kp = grid()
    p0 = ch.chebval(xr, P)[:, 0]
    s = ch.chebval(xr, ut.Dcheb(P, par.ricb, ut.rcmb))[:, 0] + p0/r + bv.lhb1*p0
    t = ch.chebval(xr, T)[:, 0].astype(complex)
    if par.m == 0 and 1 in list(lt):
        t[list(lt).index(1)] -= r*omega_M   # the mantle's rotation, T_1 = r omega_M
    uth = s @ dY[lp] + t @ Z[lt]
    uph = s @ Z[lp] - t @ dY[lt]
    return 2*par.Gaspard*wall_profiles()[0]*r**2*2*np.pi*np.sum(w*(kt*np.abs(uth)**2 + kp*np.abs(uph)**2))
