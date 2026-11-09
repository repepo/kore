'''
NEW (2026-11-06, from conductive_IC: its assemble.py rows, utils.gamma_visc, gamma_magnetic and utils4pp.angrymom,
torque_walls): the rotating solids (rotdyn = 1), and the angular momentum of the fluid that spin_doctor checks (resid𝐋).

rotdyn = 1 lets the mantle and the inner core turn as rigid bodies about the rotation axis, at the rates omega_M and omega_IC
relative to the frame, with phi the angle of the inner core relative to the mantle. For a mode exp(lambda t)

    lambda I_M  omega_M  = G_M  + K phi
    lambda I_IC omega_IC = G_IC - K phi
    lambda phi           = omega_IC - omega_M - (Gaspard/OmgtauIC) phi

with I_M, I_IC = MoIZ_M, MoIZ_IC, K = gTorque Gaspard^2 and G_M, G_IC the axial torques of the fluid on each solid: viscous
(times vtrq) and magnetic (times mtrq). The fluid follows the solids through its no-slip walls: the toroidal flow of degree 1,
u_phi = T sin(theta), is T(1) = omega_M and T(ricb) = ricb omega_IC there (a rigid rotation omega has T = omega r). NEW
(2026-11-06, as conductive_IC): the outer wall may be stress-free (bco = 0) instead; the fluid slips along the mantle there,
and a drag (kappa_cmb, topo_drag; topo.py) acts on the slip T_1 - r omega_M, its stress being the viscous torque on the
mantle (none without a drag).

The torques are in the convention of the complex amplitudes, in which the axial angular momentum of the fluid is
    L = ∫ rho0 s u_phi dV = (8π/3) ∫ rho0 r³ T dr        (T of degree 1 alone; Kore's Y_1^0 = cos(theta)):
    viscous    on the mantle  -(8π/3) R² c (R T' - T) at R = 1,   on the inner core  +(8π/3) R² c (R T' - T) at R = ricb,
               c = rho0 nu ViscosD (sigma_rphi = c (T' - T/r) sin(theta));
    magnetic   on the mantle  -Hendrik ∮ s B0_r b_phi dS at r = 1,   on the inner core  +Hendrik ∮ s B0_r b_phi dS at r = ricb,
the Maxwell stress on the fluid's side of each wall (with B0_phi = 0 there, background.check_rotdyn), which is the torque on
everything beyond it: the thin wall, the resolved mantle or the conducting inner core, whichever conducts (b_phi = 0 at an
insulating wall). With b_phi = -Σ g_l ∂θY_l and B0_r = Σ n(n+1) h_n Y_n/r (the degrees n of background_field.pol_profiles,
cnorm included)
    ∮ s B0_r b_phi dS = Σ_l C_l(r) g_l(r),   C_l = 2π r² Σ_n n(n+1) h_n 2l(l+1)/((2l+1)(2n+1)) (δ_{n,l-1} - δ_{n,l+1}),
which for a dipole is (16π/5) r² h_1 g_2, conductive_IC's gamma_magnetic.

For spin_doctor the same balance for the fluid alone, lambda L = (the torques on the fluid), resid𝐋: axial for m = 0, symm = 1,
and equatorial for m = 1, symm = -1, where L = (16π/3) ∫ rho0 r³ T dr is L_x - i L_y divided by Kore's normalisation of
Y_1^1 (-sin(theta)/sqrt(2)), as conductive_IC's angrymom, and the Coriolis torque i Gaspard L joins the torques. Its magnetic
torques come from the Maxwell stress on a grid in theta, for any B0 and walls (wall_torques).
'''
import numpy as np
import numpy.polynomial.chebyshev as ch
from parameters import par
import utils as ut
import radial_profiles as rap
import background_field as bf
import angular



def coupling():
    '''K = gTorque Gaspard², the gravitational torque per radian between the inner core and the mantle, in the run's units'''
    return par.gTorque*par.Gaspard**2


def relaxation():
    '''Gaspard/OmgtauIC, the rate of the inner core's viscous relaxation towards alignment (0 with OmgtauIC = np.inf)'''
    return par.Gaspard/par.OmgtauIC


def viscous_factor(r):
    '''rho0 nu ViscosD at the radius r (sigma_rphi = rho0 nu ViscosD (T' - T/r) sin(theta) for m = 0)'''
    rr = np.array([r], dtype=float)
    return float(np.exp(np.ravel(rap.logrhoX(rr, 0))[0])*np.ravel(rap.viscoX(rr, 0))[0])*par.ViscosD


def magnetic_factors(r):
    '''{l: C_l(r)}: ∮ s B0_r b_phi dS over the sphere of radius r is Σ_l C_l(r) g_l(r) for an axisymmetric b (module docstring)'''
    out = {}
    rr = np.array([r], dtype=float)
    for n, f in bf.pol_profiles().items():
        h = float(np.ravel(f(rr, 0))[0])
        for l, sgn in ((n + 1, 1.0), (n - 1, -1.0)):   # δ_{n,l-1} (l = n + 1) and -δ_{n,l+1} (l = n - 1)
            if l >= 1:
                out[l] = out.get(l, 0.0) + sgn*2*np.pi*r**2*n*(n + 1)*h*2*l*(l + 1)/((2*l + 1)*(2*n + 1))
    return out


def a_rows(rank0):
    '''
    The rows of A for the three unknowns (A = -(the right-hand side), as everywhere in assemble.py), and the entries of the
    solids' rotation in the no-slip rows of T of degree 1, as [data, row, col]; empty on the ranks other than rank0
    '''
    if not (ut.rotdyn and rank0):
        return [np.array([], dtype=complex), np.array([], dtype=int), np.array([], dtype=int)]
    r0 = ut.offset['rot']                         # omega_M, omega_IC, phi
    iM, iI, ip = r0, r0 + 1, r0 + 2
    ri = par.ricb
    data, rows, cols = [], [], []
    def add(d, r, c):
        d = np.atleast_1d(np.asarray(d, dtype=complex))
        data.append(d); rows.append(np.full(d.size, r)); cols.append(np.atleast_1d(np.asarray(c, dtype=int)))
    # the walls move with the solids: T(1) - omega_M = 0, T(ricb) - ricb omega_IC = 0 in the no-slip rows of T of degree 1
    # (assemble.bc_u_spherical, section v: the outer wall's row first, then the inner one's). NEW (2026-11-06): a stress-free
    # outer wall (bco = 0) is not dragged along, but through the drag on the slip T_1 - omega_M (topo.rows), if any
    tcol = ut.offset['v'] + int(ut.N1*(1 - ut.m_bot)/2)
    if par.bco == 1:
        add(-ut.rcmb, tcol, iM)
    add(-ri, tcol + 1, iI)
    # viscous torques, from T of degree 1 at each wall: A gets minus the torque
    import bc_variables as bv
    Ta, Tb = bv.Ta, bv.Tb   # the fluid's Chebyshev polynomials and their r-derivatives at the inner and the outer wall
    tc = tcol + np.arange(ut.N1)
    R = ut.rcmb
    add(-par.vtrq*(-(8*np.pi/3)*R**2*viscous_factor(R)*(R*Tb[:,1] - Tb[:,0])), iM, tc)
    add(-par.vtrq*( (8*np.pi/3)*ri**2*viscous_factor(ri)*(ri*Ta[:,1] - Ta[:,0])), iI, tc)
    # magnetic torques, from g at each wall (the l's of g: those of P if B0 is antisymmetric, of T if symmetric)
    if par.magnetic and par.mtrq:
        lg = ut.ell(par.m, par.lmax, ut.bsymm)[1]
        mg = ut.m_bot if ut.symmB0 == 1 else ut.m_top
        for r, Tw, sgn, row in ((R, Tb, -1.0, iM), (ri, Ta, 1.0, iI)):
            for l, C in magnetic_factors(r).items():
                if l in lg and C != 0:
                    gc = ut.offset['g'] + int(ut.N1*(l - mg)/2) + np.arange(ut.N1)
                    add(-sgn*par.Hendrik*C*Tw[:,0], row, gc)
    # the gravitational torques ±K phi and the angle's equation lambda phi = omega_IC - omega_M - (Gaspard/OmgtauIC) phi
    K = coupling()
    add(-K, iM, ip)
    add(K, iI, ip)
    add([1.0, -1.0, relaxation()], ip, [iM, iI, ip])
    return [np.concatenate(data), np.concatenate(rows), np.concatenate(cols)]


def b_rows(rank0):
    '''The rows of B for the three unknowns (B = -(the factor of lambda), as everywhere in assemble.py)'''
    if not (ut.rotdyn and rank0):
        return [np.array([]), np.array([], dtype=int), np.array([], dtype=int)]
    r0 = ut.offset['rot']
    idx = np.arange(r0, r0 + 3)
    return [-np.array([par.MoIZ_M, par.MoIZ_IC, 1.0]), idx, idx]



# ----------------------------------------------------------------------------------------- spin_doctor: resid𝐋 and rotdyn.dat
def L_case():
    '''
    Whether spin_doctor checks the angular momentum of the fluid (resid𝐋): eigenvalue problems with the flow, m = 0, symm = 1
    (axial) or m = 1, symm = -1 (equatorial, without a differential rotation, whose stresses would add an equatorial torque),
    and a spherical body. For m = 0 the advection by a differential rotation has no net torque (∫ rho0 u⋅∇(s U0) dV = 0
    between impenetrable walls). NEW (2026-11-08): a background flow (bgflow = 1) has none for any m: 𝐫×[(𝐔⋅∇)𝐮 + (𝐮⋅∇)𝐔] =
    (𝐔⋅∇)(𝐫×𝐮) + (𝐮⋅∇)(𝐫×𝐔), whose integral with rho0 is a flux through the walls, 0 with ∇⋅(rho0 𝐔) = ∇⋅(rho0 𝐮) = 0 and
    neither flow crossing them (the equatorial budget closes to round-off with one, tests/bgflow). NEW (2026-11-08): also
    the axial balance of the forced problems whose walls move only along themselves and that put no force in the fluid:
    the libration (forcing 7, m = 0), the steady rotation of the walls (12) and the thermal forcing (13). A radial
    boundary flow would add the Coriolis torque of the mass it carries through the walls, and a random body force (11)
    its own torque. The body forces of forcings 1 (axial or equatorial) and 8 (axial),
    whose walls are at rest, add theirs (utils4pp.body_force_terms). FIX (2026-11-09): for the same reason not with a wall
    of zero radial stress (bco or bci = 2) of a viscous fluid, which the fluid crosses.
    '''
    axial = par.m == 0 and par.symm == 1
    equatorial = par.m == 1 and par.symm == -1 and not par.diff_rot
    forced = (par.forcing in (7, 8, 12, 13) and axial) or (par.forcing == 1 and (axial or equatorial))
    # FIX (2026-11-09): nor with a wall of zero radial stress (bco or bci = 2), which the fluid crosses (resid𝐋 ~1 for
    # resolved modes): an inviscid fluid's one wall row is no penetration, whatever bco
    imperm = par.ViscosD == 0 or (par.bco in (0, 1) and (par.ricb == 0 or par.bci in (0, 1)))
    return bool(par.hydro and not par.centrifugal and imperm and ((par.forcing == 0 and (axial or equatorial)) or forced))


def _factor():
    '''8π/3 (axial) or 16π/3 (equatorial): L = factor ∫ rho0 r³ T dr, and the viscous torques with the same factor'''
    return 8*np.pi/3 if par.m == 0 else 16*np.pi/3


def _t1(usol2):
    '''the Chebyshev coefficients of T of degree 1 (the first toroidal l for m = 0, symm = 1 and m = 1, symm = -1)'''
    lt = list(ut.ell(par.m, par.lmax, par.symm)[1])
    return usol2[1][lt.index(1), :]


def angular_momentum(usol2, absval=False):
    '''
    L = factor ∫ rho0 r³ T dr of the flow usol2 (utils4pp.expand_reshape_sol), by Gauss-Legendre quadrature in r from ricb to
    1; with absval = True, factor ∫ rho0 r³ |T| dr >= |L|, the scale of resid𝐋 (a flow can carry no net angular momentum)
    '''
    x, w = np.polynomial.legendre.leggauss(par.N + 24)
    r = 0.5*(ut.rcmb - par.ricb)*(x + 1) + par.ricb
    T = ut.funcheb(_t1(usol2), r, par.ricb, ut.rcmb, 0)[:, 0]
    rho = np.exp(np.ravel(rap.logrhoX(r, 0)))
    return _factor()*0.5*(ut.rcmb - par.ricb)*np.sum(w*rho*r**3*(np.abs(T) if absval else T))


def viscous_torques(usol2):
    '''[on the mantle, on the inner core]: the viscous torques of the flow usol2 (0 without viscosity or inner core)'''
    out = np.zeros(2, dtype=complex)
    if par.ViscosD == 0:
        return out
    t1 = _t1(usol2)
    for k, (r, sgn) in enumerate(((ut.rcmb, -1.0), (par.ricb, 1.0))):
        if r > 0:
            T, dT = ut.funcheb(t1, np.array([r], dtype=float), par.ricb, ut.rcmb, 1)[0]
            out[k] = sgn*_factor()*r**2*viscous_factor(r)*(r*dT - T)
    return out


def wall_torques(bsol2):
    '''
    [on the mantle, on the inner core]: the magnetic torques, minus the Maxwell stress on the fluid through each wall,
    Hendrik ∮ r r̂ × [B0_r 𝐛 + b_r 𝐁0] dS (the B0⋅b term has no torque), from the field bsol2 (expand_reshape_sol) on the
    fluid's side, on a Gauss-Legendre grid in cos(theta): poloidal and toroidal B0 of any degrees, and any walls. Axial for
    m = 0, equatorial (L_x - i L_y divided by the normalisation of Y_1^1, as angular_momentum) for m = 1.
    '''
    out = np.zeros(2, dtype=complex)
    if not par.magnetic:
        return out
    lp, lt, _ = ut.ell(par.m, par.lmax, ut.bsymm)
    pol, tor = bf.pol_profiles(), bf.tor_profiles()
    n0 = max([1] + list(pol) + list(tor))
    x, w = np.polynomial.legendre.leggauss(par.lmax + n0 + 8)
    Y, dY, Z = angular.ylm_grid(par.m, par.lmax + 1, x)
    Y0, dY0, _ = angular.ylm_grid(0, n0, x)
    norm = float(angular.ylm_grid(1, 1, np.array([0.0]))[0][1][0]) if par.m == 1 else 1.0   # Y_1^1 at the equator
    P, T = bsol2[0].T, bsol2[1].T   # one column per l
    for k, (r, sgn) in enumerate(((ut.rcmb, 1.0), (par.ricb, -1.0))):   # sgn: the outward normal of the fluid
        if r == 0:
            continue
        rr = np.array([r], dtype=float)
        xr = ut.xcheb(rr, par.ricb, ut.rcmb)
        f  = ch.chebval(xr, P)[:, 0]
        df = ch.chebval(xr, ut.Dcheb(P, par.ricb, ut.rcmb))[:, 0]
        g  = ch.chebval(xr, T)[:, 0]
        s  = df + f/r
        br = (lp*(lp + 1)*f/r) @ Y[lp]
        bt = s @ dY[lp] + g @ Z[lt]
        bp = s @ Z[lp] - g @ dY[lt]
        B0r = sum(n*(n + 1)*np.ravel(h(rr, 0))[0]/r*Y0[n] for n, h in pol.items()) if pol else 0*x
        B0t = sum((np.ravel(h(rr, 1))[0] + np.ravel(h(rr, 0))[0]/r)*dY0[n] for n, h in pol.items()) if pol else 0*x
        B0p = -sum(np.ravel(kk(rr, 0))[0]*dY0[n] for n, kk in tor.items()) if tor else 0*x
        tth = -r*(br*B0p + B0r*bp)   # theta and phi components of r r̂ × (B0_r b + b_r B0)
        tph =  r*(br*B0t + B0r*bt)
        if par.m == 0:
            on_fluid = sgn*2*np.pi*r**2*np.sum(w*np.sqrt(1 - x**2)*(-tth))
        else:
            on_fluid = sgn*(2*np.pi/norm)*r**2*np.sum(w*(x*tth - 1j*tph))
        out[k] = -par.Hendrik*on_fluid
    return out
