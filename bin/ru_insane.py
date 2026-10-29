#!/usr/bin/env python3
'''
ru_insane.py: are the operators sane?  Symmetry tests of kore's anelastic (LBR) operator
blocks under the energy and entropy pairings, on random smooth fields that satisfy the
boundary conditions.  No eigensolve, no MPI, no SLEPc: it needs only the *.mtx submatrices,
so it runs between submatrices.py and assemble.py, in seconds.

To use (from the run directory, after ./bin/submatrices.py):
> ./bin/ru_insane.py [nvec] [--tol TOL] [--kmax K] [--decay D] [--nq Q] [--lmax-test L] [--seed S]

Exit code 0 if every test is below TOL (default 1e-6), 1 otherwise, so it can gate a run script
(./bin/ru_insane.py || exit 1).  Run it again whenever an operator, a profile, the resolution,
a boundary condition or the viscous/thermal switches change; the same configuration need not
be retested.

What is tested
--------------
With u = rho0^-1 curl curl (rho0 P r) + curl (T r), kore's rows are, per l and section,
  row_u(F) = r^rpower_u rho0^rhopower_u (r.curl curl F)_l / L^3   (section u, rows for P_l)
  row_v(F) = r^rpower_v rho0^rhopower_v (r.curl F)_l / L^3        (section v, rows for T_l)
  row_h(.) = r^h (entropy equation)_l / L^3       (section h, rows for s_l)
and for any force F per unit mass, with P = 0 on the walls,
  int (rho0 u)*.F dV = sum_l n_l L^3 [ int rho0^(1-rhopower_u) P_l* r^(3-rpower_u) row_u(F)_l dr
                                      + int rho0^(1-rhopower_v) T_l* r^(3-rpower_v) row_v(F)_l dr ],
n_l = 4 pi/(2l+1), L = l(l+1).  Under this pairing the LBR energy equation requires
  inertia   <x,Iy> =  conj<y,Ix>, <x,Ix> = KE(x) > 0
  Coriolis  <x,Cy> = -conj<y,Cx>              (does no work; its (ln rho0)' terms come from div u != 0)
  viscous   <x,Vy> =  conj<y,Vx>, <x,Vx> < 0  (rho0^-1 div(rho0 nu S) with its derivatives of ln rho0 up to fourth order,
                                               together with the stress-free rows)
and, with the s* weight on the heat rows, the entropy block Hermitian positive and the diffusion
block Hermitian negative (divergence form, thermal bc rows).  The buoyancy, advection and
entropy pairings are also compared with direct quadratures of the same fields, which checks
the r-powers and L-factors of those rows, and KE with utils4pp.kinetic_energy.

With diff_rot = 1 (background U0 = G ΔΩ r sinθ e_φ, ΔΩ = Σ_n f_n P_n, G = Gaspard) the blocks of
operators.differential_rotation (R) and operators.thermal_diffrot (heat, thermal = 1), packed as assemble.py packs them,
are paired with the poloidal (P, s) and toroidal (T) parts of the fields, every part against every part, and compared
with direct (r, cos θ) quadratures of the 3-D fields (uᵣ = L P Y/r, u_θ = Q ∂θY/r + i m T Y/sinθ,
u_φ = i m Q Y/(r sinθ) - T ∂θY, Q = (rP)' + r (ln rho0)' P, Y Schmidt as ut.Ylm):
  inertia, Coriolis, diff-rot  <x,Iy>, <x,Cy>, <x,Ry> against int rho0 x*.y, int rho0 x*.(2G z x y) and
                               int rho0 x*.G[(U0'.grad)y + (y.grad)U0'], U0' = U0/G (same sign as Coriolis in A);
                               errors relative to the Cauchy-Schwarz bound sqrt(KE(x)KE(y)) (times 2G, G max|ΔΩ|)
  heat advection               'temp' blocks against -int p0 x_s* i m G ΔΩ y_s (the heat blocks carry the sign of A),
                               and anti-Hermitian: <x,Ht y> = -conj<y,Ht x> (relative to m G max|ΔΩ| sqrt(E(x)E(y)))
  heat thermal wind            'upol', 'utor' blocks against -int p0 x_s* u(y).grad S_tw, S_tw = Σ σn Pn(cos θ) recomputed
                               from the f_n and g (thermal_wind = 1: linear in ΔΩ; 2: exact (1+ΔΩ)²; 3: prescribed)
  rigid (diff_rot_type 'rigid', ΔΩ = d): R = d (i m G I + C), Ht = -i m G d E and the thermal-wind blocks vanish.

A sign error, a missing rho0, a diffusion term not in divergence form or an inconsistent bc row
shows up at 1e-3 to 1 against a floor of 1e-12 (1e-8 for the viscous block at low l).  An error
that preserves the symmetry (a wrong but symmetric coefficient, a wrong nu profile) is invisible
here; that is what the comparison of the energy budget with spin_doctor is for.  The diff-rot comparisons are not
symmetry tests: a 1 % error in one coefficient of one Δl term shows up at 1e-4 to 1.

The random fields have Chebyshev coefficients up to degree kmax (default N1/4) decaying as
exp(-k/decay) (default kmax/8), with the lowest coefficients adjusted in the least-squares
sense so that the bc rows hold exactly (a correction at high degree leaves bc residuals of 1e-6
because the stress-free rows carry (ln rho0)'' and k^4 factors, and that pollutes the test).
Not handled: the inviscid formulation (ViscosD = 0, rows multiplied by rho0^2) and compositional blocks.

NEW (2026-10-28): with magnetic = 1 (insulating walls or a conducting inner core), on random fields that satisfy the field's
wall conditions (insulating walls: (l+1)f + rf' = 0 outside, lf - rf' = 0 inside, g = 0), with b = curl curl(f r) + curl(g r):
  b time derivative   <x, b y>_r = n_l L^3 int conj(x_f) row_f(b y) dr + (g alike): Hermitian positive
  magnetic diffusion  under the same pairing: Hermitian negative (with the insulating rows the wall terms are Hermitian and
                      negative too; not with a conducting inner core, whose matching rows couple to fic, gic: skipped then)
  Lorentz/induction   for B0 = 'axial' (no background current) and no-slip walls, the energy identity
                      int rho0 u*.F_L dV = -Hendrik conj(int b*.curl(u x B0) dV), F_L the Lorentz force per unit mass:
                      <u, Lorentz b> (the u, v pairing above) against Hendrik conj(E_b(b, induction u)), where
                      E_b(x, X) = n_l L^3 int [ D_l conj(x_f) row_f(X) - conj(x_g) row_g(X) ] dr = int x*.(curl(u x B0)) dV for
                      the induction rows (A sign, rows r^3 r.(.)/L^3 and r^3 r.curl(.)/L^3), D_l f = f'' + 2f'/r - L f/r^2;
                      it holds because u = 0 on the walls, so (u x B0) and r.curl(u x B0) vanish there. Relative to the
                      Cauchy-Schwarz-like scale |<u, Lorentz b>| + Hendrik |E_b|. With variable rho0 this tests the (ln rho0)'
                      terms of both the induction (P columns) and the Lorentz force (u rows) against each other.
'''

import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.getcwd(), 'bin'))                    # the run's own bin/ first
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.sparse as ss
import numpy.polynomial.chebyshev as ch
from parameters import par
import utils as ut
import radial_profiles as rap
import operators as op
import bc_variables as bv
import diff_rot as dr

ap = argparse.ArgumentParser(description='symmetry tests of the kore anelastic operator blocks')
ap.add_argument('nvec', nargs='?', type=int, default=3, help='number of random fields (default 3)')
ap.add_argument('--tol', type=float, default=1e-6, help='pass threshold for every relative error (default 1e-6)')
ap.add_argument('--kmax', type=int, default=0, help='highest Chebyshev degree of the random fields (default N1/4)')
ap.add_argument('--decay', type=float, default=0., help='coefficients decay as exp(-k/decay) (default kmax/8)')
ap.add_argument('--nq', type=int, default=0, help='Gauss-Legendre quadrature nodes (default 2N+2)')
ap.add_argument('--lmax-test', type=int, default=10**6, help='zero the fields for l above this')
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()

N    = par.N
N1   = ut.N1
ricb = par.ricb
rcmb = ut.rcmb
lp, lt, ll = ut.ell(par.m, par.lmax, par.symm)
nl   = len(lp)

if par.ViscosD == 0:
    print('ru_insane.py: SKIPPED, the inviscid formulation (rows multiplied by rho0^2) is not handled')
    sys.exit(0)

divp0 = hasattr(op, 'h0_D0')   # ThermaD=0 heat rows divided by p0

gb   = {'u': 4, 'v': 2, 'h': 2 if par.ThermaD > 0 else int(ut.adiab_tw), 'f': 2, 'g': 2}   # Gegenbauer basis of each section (C^(1) for ThermaD = 0 with the thermal wind)
chop = {s: (gb[s] if ricb > 0 else gb[s]//2) for s in gb}         # rows reserved for bc's
if par.ThermaD == 0:
    chop['h'] = 0                                                 # no thermal bc rows, whatever the basis

vP = int((1 - 2*(par.m % 2))*par.symm)   # Chebyshev parity of P (ricb=0 only)
vT = -vP
vS = vP
s_ = int((par.symm + 1)/2)
iP = (par.m + 1 - s_) % 2                # first Chebyshev index kept for P, T, s when ricb=0
iT = (par.m + s_) % 2
iS = iP
istart = {'P': iP, 'T': iT, 'S': iS}


def row_parity(label, vpar):
    (sec, rpower, _, f1, d1, f2, d2, dx) = ut.decode_label(label)
    adj = int(f1 in ut.odd_profiles)
    opp = 1 - ((rpower + (d1 or 0) + (d2 or 0) + dx + adj) % 2)*2
    return vpar*opp

hlab = 'h0_D0' if divp0 else 'h0pss0_D0'
idj = {'u': (1 - row_parity('u3_D0', vP))//2,
       'v': (1 - row_parity('v1_D0', vT))//2,
       'h': (1 - row_parity(hlab, vS))//2}

# ---------------------------------------------------------------------------------------- quadrature
Nq = args.nq if args.nq > 0 else 2*N + 2       # even, so r=0 is never a node when ricb=0
xk, wk = np.polynomial.legendre.leggauss(Nq)
if ricb > 0:
    rk  = 0.5*(rcmb - ricb)*(xk + 1) + ricb
    jac = 0.5*(rcmb - ricb)
else:
    rk  = rcmb*xk                              # integrands are even in r: int_0^1 = 1/2 int_-1^1
    jac = 0.5*rcmb

rho0 = np.exp(rap.logrhoX(rk, 0))
if par.thermal:
    pss0 = rap.pressX(rk, 0)
    gra0 = rap.graviX(rk, 0)
    dS0  = rap.prf.gradS(np.abs(rk))*np.sign(rk)

# pairing weights (functions of r) for each section
wgt = {'u': rho0**(1 - par.rhopower_u) * rk**(3 - par.rpower_u),
       'v': rho0**(1 - par.rhopower_v) * rk**(3 - par.rpower_v)}
if par.thermal:
    if par.ThermaD > 0:
        wgt['h'] = np.ones_like(rk)            # rows are r^2 (eq)/L^3
    elif divp0:
        wgt['h'] = pss0 if ut.adiab_tw else pss0*rk   # rows are r (eq/p0)/L^3, r^2 (eq/p0)/L^3 with the thermal wind
    else:
        wgt['h'] = rk                          # rows are r (eq)/L^3

# Chebyshev values at the nodes, and values of a C^(lamb) series at the nodes (three-term recurrence)
Tmat = ch.chebvander(xk, N - 1)                # (Nq, N)
Emap = {}
for lamb in set(gb.values()):
    if lamb == 0:
        Emap[lamb] = Tmat
        continue
    E = np.zeros((Nq, N)); E[:, 0] = 1.; E[:, 1] = 2*lamb*xk
    for k in range(2, N):
        E[:, k] = (2*(k + lamb - 1)*xk*E[:, k-1] - (k + 2*lamb - 2)*E[:, k-2])/k
    Emap[lamb] = E


def full_coeffs(vec, start):
    '''raw coefficient vector (length N1) -> full Chebyshev coefficients (length N)'''
    if ricb > 0:
        return vec
    c = np.zeros(N, dtype=complex)
    c[start::2] = vec
    return c


def vals_unknown(vec, kind):
    '''values at the nodes of an unknown (P, T or S) given by its raw coefficients'''
    return Tmat @ full_coeffs(vec, istart[kind])


def vals_row(y, sec):
    '''values at the nodes of an operator output y (raw, with the bc rows on top)'''
    nk = N1 - chop[sec]
    c  = np.zeros(N, dtype=complex)
    if ricb > 0:
        c[:nk] = y[chop[sec]:]
    else:
        c[idj[sec] + 2*np.arange(nk)] = y[chop[sec]:]
    return Emap[gb[sec]] @ c


def pair(sec, l, left_vals, y):
    '''n_l L^3 int w(r) conj(left) y dr, the energy (or entropy) pairing of section sec'''
    L   = l*(l + 1.)
    nl_ = 4*np.pi/(2*l + 1)
    return nl_ * L**3 * jac * np.sum(wk * wgt[sec] * np.conj(left_vals) * vals_row(y, sec))


# ---------------------------------------------------------------------------- operator applications
KEYS = ['I', 'C', 'V', 'B', 'E', 'A', 'D', 'R', 'Ht', 'Hw']

def dr_blocks(l, sec):
    '''
    differential-rotation blocks of row l in section sec (diff_rot = 1), as assemble.py packs them: list of
    (mtx, offd, kind), kind the unknown ('P', 'T' or 'S') of the column block k + offd. Sections u, v:
    operators.differential_rotation; section h: operators.thermal_diffrot ('temp' always, 'upol'/'utor' if thermal_wind)
    '''
    if not par.diff_rot:
        return []
    comp = {'u': [('upol', dr.dl_range('u', 'upol'), 'P'), ('utor', dr.dl_range('u', 'utor'), 'T')],
            'v': [('upol', dr.dl_range('v', 'upol'), 'P'), ('utor', dr.dl_range('v', 'utor'), 'T')],
            'h': [('temp', dr.dl_range('h', 'temp'), 'S'), ('upol', dr.dl_range('h', 'upol'), 'P'),
                  ('utor', dr.dl_range('h', 'utor'), 'T')]}[sec]
    fun = op.thermal_diffrot if sec == 'h' else op.differential_rotation
    out = [tuple(fun(l, sec, name, i)) + (kind,) for name, offs, kind in comp for i in offs if l + i in (lt if kind == 'T' else lp)]
    return [b for b in out if not np.isscalar(b[0])]   # an operator returns a plain 0 for a Δl it does not have


def apply_blocks(blocks, fld, k):
    '''sum of the blocks (mtx, offd, kind) applied to the column blocks k + offd of the field (P, T, S)'''
    unk = dict(zip('PTS', fld))
    return sum((mtx @ unk[kind][k + offd] for mtx, offd, kind in blocks), np.zeros(N1, dtype=complex))


def cross_pairings(left, right):
    '''
    left, right: lists of fields (P, T, S), each an (nl, N1) raw coefficient array.
    out[a][b] = dict of pairings <left_a, Op right_b> summed over l:
      I = int (rho0 u)*.u (= KE),  C = int (rho0 u)*.(2 z x u),  V = int (rho0 u)*.F_visc,
      B = int (rho0 u)*.(Beyonce g s r^),  and with the s* weight on the heat rows:
      E = int p0 |s|^2,  A = -int p0 S' u_r s*,  D = ThermaD int s* div(kappa p0 grad s);
    with diff_rot = 1 also R = int (rho0 u)*.G[(U0'.grad)u + (u.grad)U0'] (momentum diff-rot blocks) and the heat
    diff-rot blocks Ht = -int p0 s* i m G dOmega s ('temp') and Hw = -int p0 s* u.grad S_tw ('upol', 'utor').
    '''
    na, nb = len(left), len(right)
    out = [[{k: 0j for k in KEYS} for b in range(nb)] for a in range(na)]
    for k, l in enumerate(lp):                                   # section u, rows for P_l
        I  = op.inertia(l, 'u', 'upol', 0)
        Cd = op.coriolis(l, 'u', 'upol', 0)[0]
        Co = [(op.coriolis(l, 'u', 'utor', i), i) for i in [-1, 1] if l + i in lt]
        V  = op.viscous_diffusion(l, 'u', 'upol', 0)
        Bu = op.buoyancy(l, 'u', '', 0) if par.thermal else None
        Rb = dr_blocks(l, 'u')
        ys = []
        for (P, T, S) in right:
            yC = Cd @ P[k]
            for (mtx, offd), i in Co:
                yC = yC + mtx @ T[k + offd]
            ys.append((I @ P[k], yC, V @ P[k], Bu @ S[k] if par.thermal else None, apply_blocks(Rb, (P, T, S), k)))
        for a, (P, T, S) in enumerate(left):
            Pv = vals_unknown(P[k], 'P')
            for b in range(nb):
                out[a][b]['I'] += pair('u', l, Pv, ys[b][0])
                out[a][b]['C'] += pair('u', l, Pv, ys[b][1])
                out[a][b]['V'] += pair('u', l, Pv, ys[b][2])
                out[a][b]['R'] += pair('u', l, Pv, ys[b][4])
                if par.thermal:
                    out[a][b]['B'] += pair('u', l, Pv, ys[b][3])
    for k, l in enumerate(lt):                                   # section v, rows for T_l
        I  = op.inertia(l, 'v', 'utor', 0)
        Cd = op.coriolis(l, 'v', 'utor', 0)[0]
        Co = [(op.coriolis(l, 'v', 'upol', i), i) for i in [-1, 1] if l + i in lp]
        V  = op.viscous_diffusion(l, 'v', 'utor', 0)
        Rb = dr_blocks(l, 'v')
        ys = []
        for (P, T, S) in right:
            yC = Cd @ T[k]
            for (mtx, offd), i in Co:
                yC = yC + mtx @ P[k + offd]
            ys.append((I @ T[k], yC, V @ T[k], apply_blocks(Rb, (P, T, S), k)))
        for a, (P, T, S) in enumerate(left):
            Tv = vals_unknown(T[k], 'T')
            for b in range(nb):
                out[a][b]['I'] += pair('v', l, Tv, ys[b][0])
                out[a][b]['C'] += pair('v', l, Tv, ys[b][1])
                out[a][b]['V'] += pair('v', l, Tv, ys[b][2])
                out[a][b]['R'] += pair('v', l, Tv, ys[b][3])
    if par.thermal:                                              # section h, rows for s_l
        for k, l in enumerate(lp):
            E = op.entropy(l, 'h', '', 0)
            A = op.thermal_advection(l, 'h', 'upol', 0)
            D = op.thermal_diffusion(l, 'h', '', 0) if par.ThermaD > 0 else None
            Hb = dr_blocks(l, 'h')
            Ht = [b for b in Hb if b[2] == 'S']; Hw = [b for b in Hb if b[2] != 'S']
            ys = []
            for (P, T, S) in right:
                ys.append((E @ S[k], A @ P[k], D @ S[k] if par.ThermaD > 0 else None,
                           apply_blocks(Ht, (P, T, S), k), apply_blocks(Hw, (P, T, S), k)))
            for a, (P, T, S) in enumerate(left):
                Sv = vals_unknown(S[k], 'S')
                for b in range(nb):
                    out[a][b]['E'] += pair('h', l, Sv, ys[b][0])
                    out[a][b]['A'] += pair('h', l, Sv, ys[b][1])
                    if par.ThermaD > 0:
                        out[a][b]['D'] += pair('h', l, Sv, ys[b][2])
                    out[a][b]['Ht'] += pair('h', l, Sv, ys[b][3])
                    out[a][b]['Hw'] += pair('h', l, Sv, ys[b][4])
    return out


def direct_quadratures(P, T, S):
    '''
    From the fields alone (no operators), to check the r-powers and L-factors of the buoyancy,
    advection and entropy rows:  W_buo = int rho0 Beyonce g u_r* s dV,  W_adv = -int p0 S' u_r s* dV,
    TE = int p0 |s|^2 dV,  with u_r = L P / r.
    '''
    wb = 0j; wa = 0j; te = 0j
    for k, l in enumerate(lp):
        L = l*(l + 1.); nl_ = 4*np.pi/(2*l + 1)
        ur = L*vals_unknown(P[k], 'P')/rk
        sv = vals_unknown(S[k], 'S')
        wb += nl_ * jac * np.sum(wk * rho0 * par.Beyonce * gra0 * np.conj(ur) * sv * rk**2)
        wa += -nl_ * jac * np.sum(wk * pss0 * dS0 * ur * np.conj(sv) * rk**2)
        te += nl_ * jac * np.sum(wk * pss0 * np.abs(sv)**2 * rk**2)
    return wb, wa, te


# ------------------------------------------------------------- differential rotation: (r, θ) grid quadratures
if par.diff_rot:
    m  = par.m
    G  = par.Gaspard
    xg, wx = np.polynomial.legendre.leggauss(par.lmax + 12 + dr.tw_nmax() + dr.n_max())   # x = cos θ
    sg = np.sqrt(1 - xg**2)
    th = np.arccos(xg)
    def Ysch(l, mm):   # Schmidt harmonic Y_l^mm(θ) at φ = 0, as ut.Ylm
        return ut.Ylm(l, mm, th, 0.).real if l >= mm else 0*xg
    def dYsch(l):      # dY_l^m/dθ = m cotθ Y_l^m + √((l-m)(l+m+1)) Y_l^{m+1}
        return m*xg/sg*Ysch(l, m) + np.sqrt((l - m)*(l + m + 1.))*Ysch(l, m + 1)
    Yp, dYp = np.array([Ysch(l, m) for l in lp]), np.array([dYsch(l) for l in lp])
    Yt, dYt = np.array([Ysch(l, m) for l in lt]), np.array([dYsch(l) for l in lt])
    dxdr = 2/(rcmb - ricb) if ricb > 0 else 1/rcmb
    h1   = rap.logrhoX(rk, 1)
    def fX(fun, r, d):   # ricb = 0: the physical, even f_n at |r| (as heat_check.py), not the code's fit at r
        return fun(np.abs(r), d)*np.sign(r)**d if ricb == 0 else fun(r, d)
    # ΔΩ = Σ_n f_n(r) P_n(cos θ) and its derivatives, (Nq, Nx): only the profiles come from diff_rot, the angular part
    # (numpy Legendre polynomials on the grid) is independent of its algebra
    Lg   = np.polynomial.legendre.Legendre
    nd   = dr.degrees()
    fn_  = {n: (lambda r, d, n=n: dr.fX(n, r, d)) for n in nd}
    dOm  = sum(fX(fn_[n], rk, 0)[:, None]*Lg.basis(n)(xg) for n in nd)
    dOmr = sum(fX(fn_[n], rk, 1)[:, None]*Lg.basis(n)(xg) for n in nd)
    dOmt = sum(-sg*Lg.basis(n).deriv()(xg)*fX(fn_[n], rk, 0)[:, None] for n in nd)
    if par.thermal and par.thermal_wind:
        r0 = ricb if ricb > 0 else -rcmb           # ∂ᵣS_tw from a Chebyshev interpolant on 2N nodes (r = 0 is not one)
        xc = np.cos(np.pi*(np.arange(2*N) + 0.5)/(2*N))
        def dSdx(r, x):
            '''∂S_tw/∂x at the radii r (nr,) and x (nr, nx) or (nx,), independently of diff_rot.sigma_coefficients:
            thermal_wind 1, 2: -(r² G²/(Beyonce g)) [x F_r + (1 - x²) F_x/r], F = 2ΔΩ (1) or (1 + ΔΩ)² - 1 (2), times
            thermal_wind_amplitude; 3: ∂θS_tw = dSdtheta sin 2θ, so ∂S_tw/∂x = -2 dSdtheta x'''
            x = np.broadcast_to(x, (len(r), np.shape(x)[-1]))
            if par.thermal_wind == 3:
                return -2*par.dSdtheta*x
            f = {n: [fX(fn_[n], r, d)[:, None] for d in (0, 1)] for n in nd}
            O  = sum(f[n][0]*Lg.basis(n)(x) for n in nd)
            Or = sum(f[n][1]*Lg.basis(n)(x) for n in nd)
            Ox = sum(f[n][0]*Lg.basis(n).deriv()(x) for n in nd)
            a = (1 + O) if par.thermal_wind == 2 else 1
            q = (r**2*G**2/(par.Beyonce*rap.graviX(r, 0)))[:, None]
            return -q*(x*2*a*Or + (1 - x**2)*2*a*Ox/r[:, None])*par.thermal_wind_amplitude
        u_, w_ = np.polynomial.legendre.leggauss(dr.tw_nmax() + 4)
        def S_of(r):   # S(r, xg) = ∫_{-1}^{x} ∂S/∂x dx' (Gauss-Legendre on each [-1, x], exact) without its P0 part
            out = np.zeros((len(r), len(xg)))
            for j, xj in enumerate(xg):
                t = -1 + (xj + 1)*(u_ + 1)/2
                out[:, j] = (dSdx(r, t) @ w_)*(xj + 1)/2
            return out - (out @ wx)[:, None]/2
        Stw_t = -sg*dSdx(rk, xg)                   # ∂θS_tw
        Sc = ch.chebfit(xc, S_of(0.5*(rcmb - r0)*(xc + 1) + r0), 2*N - 1)
        Stw_r = ch.chebval((2*rk - r0 - rcmb)/(rcmb - r0), ch.chebder(Sc)).T*2/(rcmb - r0)
        if not par.thermal_wind_radial:   # the heat equation without uᵣ ∂S_tw/∂r
            Stw_r = 0*Stw_r
        # size of ∇S_tw before the cancellations, G² r²/(Beyonce g) |∂F| (S_tw = 0 for 'cylindrical': round-off only)
        aS = abs(G**2*rk**2/(par.Beyonce*rap.graviX(rk, 0)))[:, None]*2*abs(1 + (par.thermal_wind == 2)*dOm)*(abs(dOmr) + abs(dOmt/rk[:, None])) \
            if par.thermal_wind != 3 else abs(par.dSdtheta)*np.ones_like(dOm)
    dV = 2*np.pi*jac*(wk*rk**2)[:, None]*wx[None, :]        # r² dr d(cos θ) dφ


def grid_fields(fld):
    '''uᵣ, u_θ, u_φ, s of a field (P, T, S) at the (r, θ) nodes (factor e^{imφ} dropped), with
    uᵣ = L P Y/r, u_θ = Q dY/dθ/r + i m T Y/sinθ, u_φ = i m Q Y/(r sinθ) - T dY/dθ, Q = (rP)' + r (ln ρ0)' P'''
    P, T, S = fld
    Pv = np.array([vals_unknown(p, 'P') for p in P])
    Pd = np.array([ch.chebval(xk, ch.chebder(full_coeffs(p, iP)))*dxdr for p in P])
    Q  = (Pv + rk*Pd + rk*h1*Pv)/rk
    Tv = np.array([vals_unknown(t, 'T') for t in T])
    ur  = (lp*(lp + 1.))[:, None]*Pv/rk
    uth = Q.T @ dYp + 1j*m*(Tv.T @ Yt)/sg
    uph = 1j*m*(Q.T @ Yp)/sg - Tv.T @ dYt
    s   = np.array([vals_unknown(q, 'S') for q in S]).T @ Yp if par.thermal else None
    return ur.T @ Yp, uth, uph, s


def grid_pairings(gx, gy):
    '''
    direct quadratures, from grid_fields of x and y, of  I = int rho0 x*.y,  C = int rho0 x*.(2G z x y),
    R = int rho0 x*.G[(U0'.grad)y + (y.grad)U0'],  U0' = dOmega r sinθ e_φ, which in components is
      G { dOmega (i m y + 2 z x y) + sinθ (r ∂ᵣdOmega y_r + ∂θdOmega y_θ) e_φ },  z x y = (-sinθ y_φ, -cosθ y_φ, cosθ y_θ + sinθ y_r),
    and with thermal = 1  Ht = int p0 x_s* i m G dOmega y_s,  Hw = int p0 x_s* (y_r ∂ᵣS_tw + y_θ ∂θS_tw/r).
    '''
    (xr, xt, xp, xs), (yr, yt, yp, ys) = gx, gy
    zy = (-sg*yp, -xg*yp, xg*yt + sg*yr)
    def dot(f):
        return np.sum(dV*rho0[:, None]*(np.conj(xr)*f[0] + np.conj(xt)*f[1] + np.conj(xp)*f[2]))
    out = {'I': dot((yr, yt, yp)), 'C': 2*G*dot(zy),
           'R': G*dot([dOm*(1j*m*yy + 2*zz) for yy, zz in zip((yr, yt, yp), zy)]) +
                G*dot((0, 0, sg*(rk[:, None]*dOmr*yr + dOmt*yt)))}
    if par.thermal:
        out['Ht'] = np.sum(dV*pss0[:, None]*np.conj(xs)*1j*m*G*dOm*ys)
        out['Hw'] = np.sum(dV*pss0[:, None]*np.conj(xs)*(yr*Stw_r + yt*Stw_t/rk[:, None])) if par.thermal_wind else 0j
        out['Hwa'] = np.sum(dV*pss0[:, None]*abs(xs)*(abs(yr) + abs(yt/rk[:, None]))*aS) if par.thermal_wind else 0.
    return out


# ------------------------------------------------------------------------------------- bc rows
def bc_rows(l, sec):
    '''dense bc rows for one l block, as in assemble.bc_u_spherical / bc_theta_spherical'''
    R = rcmb; Ri = ricb; L = l*(l + 1.)
    ixu = (par.m + 1 - ut.s) % 2
    ixv = (par.m + ut.s) % 2
    Tbu = bv.Tb if ricb > 0 else bv.Tb[ixu::2, :]
    Tbv = bv.Tb if ricb > 0 else bv.Tb[ixv::2, :]
    rows = []
    if sec == 'u':
        if par.bco in [0, 2]:
            if par.bco == 0:
                rows.append(Tbu[:, 0])
            else:
                rows.append(Tbu[:, 0]*(bv.lhb1*R - 3) + Tbu[:, 1]*3*R)
            rows.append(Tbu[:, 2]*R**2 + Tbu[:, 1]*R**2*bv.lhb1 + Tbu[:, 0]*(R**2*bv.lhb2 + (L - 2) - R*bv.lhb1))
        elif par.bco == 1:
            rows.append(Tbu[:, 0])
            rows.append(Tbu[:, 1] + Tbu[:, 0]*(bv.lhb1 + 1/R))
        if ricb > 0:
            if par.bci in [0, 2]:
                if par.bci == 0:
                    rows.append(bv.Ta[:, 0])
                else:
                    rows.append(bv.Ta[:, 0]*(bv.lha1*Ri - 3) + bv.Ta[:, 1]*3*Ri)
                rows.append(bv.Ta[:, 2]*Ri**2 + bv.Ta[:, 1]*Ri**2*bv.lha1 + bv.Ta[:, 0]*(Ri**2*bv.lha2 + (L - 2) - Ri*bv.lha1))
            elif par.bci == 1:
                rows.append(bv.Ta[:, 0])
                rows.append(bv.Ta[:, 1] + bv.Ta[:, 0]*(bv.lha1 + 1/Ri))
    elif sec == 'v':
        if par.bco in [0, 2]:
            rows.append(R*Tbv[:, 1] - Tbv[:, 0])
        elif par.bco == 1:
            rows.append(Tbv[:, 0])
        if ricb > 0:
            if par.bci in [0, 2]:
                rows.append(Ri*bv.Ta[:, 1] - bv.Ta[:, 0])
            elif par.bci == 1:
                rows.append(bv.Ta[:, 0])
    elif sec == 'h' and par.ThermaD > 0:   # s = 0 (0) or s' = 0 (1)
        Tbh = bv.Tb if ricb > 0 else bv.Tb[ixu::2, :]
        rows.append(Tbh[:, par.bco_thermal])
        if ricb > 0:
            rows.append(bv.Ta[:, par.bci_thermal])
    if len(rows) == 0:
        return np.zeros((0, N1), dtype=complex)
    return np.array(rows, dtype=complex)


def enforce_bc(x, Bc):
    '''adjust the lowest nbc+6 coefficients of x (least squares) so that Bc x = 0 and x stays smooth'''
    nbc = Bc.shape[0]
    if nbc == 0:
        return x
    cols = np.arange(min(nbc + 6, len(x)))
    x = x.copy()
    x[cols] -= np.linalg.lstsq(Bc[:, cols], Bc @ x, rcond=None)[0]
    return x


# ------------------------------------------------------------------------- NEW (2026-10-28): magnetic blocks
def magnetic_checks(nvec, rng, kmax, decay, note):
    '''the magnetic tests of the docstring; note(name, value) records each relative error'''
    if ricb == 0:
        print('magnetic: SKIPPED (ricb = 0 is not implemented for magnetic = 1)'); return
    lf, lg = (lt, lp) if ut.symmB0 == -1 else (lp, lt)   # degrees of f (and fic), g
    win = np.exp(-np.arange(kmax)/decay)
    Tder = lambda c, d: ch.chebval(xk, ch.chebder(c, d))*(2/(rcmb - ricb))**d
    def bc_b(l, sec):
        if sec == 'f':
            return np.array([(l + 1)*bv.Tb[:, 0] + rcmb*bv.Tb[:, 1], l*bv.Ta[:, 0] - ricb*bv.Ta[:, 1]], dtype=complex)
        return np.array([bv.Tb[:, 0], bv.Ta[:, 0]], dtype=complex)
    def rnd(ls, sec):
        out = np.zeros((len(ls), N1), dtype=complex)
        for k, l in enumerate(ls):
            out[k, :kmax] = win*(rng.standard_normal(kmax) + 1j*rng.standard_normal(kmax))
            out[k] = enforce_bc(out[k], bc_b(l, sec))
        return out
    bf = [(rnd(lf, 'f'), rnd(lg, 'g')) for _ in range(nvec)]
    uf = [random_fields(rng, kmax, decay, 10**6)[:2] for _ in range(nvec)]
    def pair_r(sec, l, xv, y):   # n_l L^3 int conj(x) y dr, no weight
        L = l*(l + 1.)
        return 4*np.pi/(2*l + 1) * L**3 * jac * np.sum(wk * np.conj(xv) * vals_row(y, sec))
    def bpair(x, y, opname):     # <x, Op y>_r over f and g, Op = 'b' or 'diff'
        out = 0j
        for sec, comp, ls, i in (('f', 'bpol', lf, 0), ('g', 'btor', lg, 1)):
            for k, l in enumerate(ls):
                M = op.b(l, sec, comp, 0) if opname == 'b' else op.magnetic_diffusion(l, sec, comp, 0)
                out += pair_r(sec, l, Tmat @ x[i][k], M @ y[i][k])
        return out
    for a in range(nvec):
        for c in range(a, nvec):
            for name, sign in (('b', 1), ('diff', -1)):
                if name == 'diff' and ut.cic:
                    continue
                xy, yx = bpair(bf[a], bf[c], name), bpair(bf[c], bf[a], name)
                note('magnetic %s Hermitian' % name, abs(xy - np.conj(yx))/abs(xy))
                if a == c:
                    note('magnetic %s %s' % (name, 'positive' if sign > 0 else 'negative'), float(sign*xy.real <= 0))
    print('magnetic: b block Hermitian %.2e%s' % (worst_of('magnetic b Hermitian'),
          '' if ut.cic else ', diffusion Hermitian %.2e (negative: %s)' % (worst_of('magnetic diff Hermitian'),
          'yes' if worst_of('magnetic diff negative') == 0 else 'NO')))
    if par.B0 != 'axial' or par.bci != 1 or par.bco != 1:
        print('magnetic: Lorentz/induction identity skipped (needs B0 = \'axial\' and no-slip walls)'); return
    for a in range(nvec):
        F, Gf = bf[a]
        for c in range(nvec):
            P, T = uf[c]
            lor = 0j                                   # <u, Lorentz b> with the momentum pairing
            for k, l in enumerate(lp):
                y = np.zeros(N1, dtype=complex)
                for comp, ls, B, dls in (('bpol', lf, F, (-1, 1)), ('btor', lg, Gf, (0,))):
                    for i in dls:
                        if l + i in ls:
                            mtx, offd = op.lorentz(l, 'u', comp, i)
                            y = y + mtx @ B[k + offd]
                lor += pair('u', l, vals_unknown(P[k], 'P'), y)
            for k, l in enumerate(lt):
                y = np.zeros(N1, dtype=complex)
                for comp, ls, B, dls in (('bpol', lf, F, (0,)), ('btor', lg, Gf, (-1, 1))):
                    for i in dls:
                        if l + i in ls:
                            mtx, offd = op.lorentz(l, 'v', comp, i)
                            y = y + mtx @ B[k + offd]
                lor += pair('v', l, vals_unknown(T[k], 'T'), y)
            Eb = 0j                                    # E_b(b, induction u)
            for sec, ls, X, wfun in (('f', lf, F, lambda c_, L: Tder(c_, 2) + 2*Tder(c_, 1)/rk - L*Tmat @ c_/rk**2),
                                     ('g', lg, Gf, lambda c_, L: -(Tmat @ c_))):
                for k, l in enumerate(ls):
                    y = np.zeros(N1, dtype=complex)
                    for comp, lu, U, dls in (('upol', lp, P, (-1, 0, 1)), ('utor', lt, T, (-1, 0, 1))):
                        for i in dls:
                            if l + i in lu:
                                mtx, offd = op.induction(l, sec, comp, i)
                                if np.isscalar(mtx):
                                    continue
                                y = y + (mtx @ U[k + offd]).astype(complex)
                    Eb += pair_r(sec, l, wfun(X[k], l*(l + 1.)), y)
            H = par.Hendrik
            note('magnetic Lorentz/induction identity', abs(lor - H*np.conj(Eb))/(abs(lor) + H*abs(Eb)))
    print('magnetic: <u, Lorentz b> = Hendrik conj(int b*.curl(u x B0) dV): worst %.2e (scale |<u,Lb>| + Hendrik |E_b|)'
          % worst_of('magnetic Lorentz/induction identity'))


def worst_of(name):
    return WORST.get(name, np.nan)


def random_fields(rng, kmax, decay, lmax_test):
    '''random smooth Chebyshev coefficient fields satisfying the bc's, coefficients ~ exp(-k/decay)'''
    win = np.exp(-np.arange(kmax)/decay) if decay > 0 else np.ones(kmax)
    P = np.zeros((nl, N1), dtype=complex)
    T = np.zeros((nl, N1), dtype=complex)
    S = np.zeros((nl, N1), dtype=complex)
    for k, l in enumerate(lp):
        P[k, :kmax] = win*(rng.standard_normal(kmax) + 1j*rng.standard_normal(kmax))
        P[k] = enforce_bc(P[k], bc_rows(l, 'u'))
        S[k, :kmax] = win*(rng.standard_normal(kmax) + 1j*rng.standard_normal(kmax))
        S[k] = enforce_bc(S[k], bc_rows(l, 'h'))
        if l > lmax_test: P[k] = 0; S[k] = 0
    for k, l in enumerate(lt):
        T[k, :kmax] = win*(rng.standard_normal(kmax) + 1j*rng.standard_normal(kmax))
        T[k] = enforce_bc(T[k], bc_rows(l, 'v'))
        if l > lmax_test: T[k] = 0
    return P, T, S


# ---------------------------------------------------------------------------------------- driver
WORST = {}   # the worst relative error of each test (main, magnetic_checks)

def main():
    nvec  = args.nvec
    kmax  = args.kmax if args.kmax > 0 else N1//4
    decay = args.decay if args.decay > 0 else kmax/8
    rng   = np.random.default_rng(args.seed)
    print('ru_insane.py: N=%d, lmax=%d, m=%d, symm=%d, ricb=%g, bci=%d, bco=%d, thermal=%d, ThermaD=%g, divp0=%s'
          % (N, par.lmax, par.m, par.symm, ricb, par.bci, par.bco, par.thermal, par.ThermaD, divp0))
    fields = [random_fields(rng, kmax, decay, args.lmax_test) for _ in range(nvec)]
    t0 = time.time()
    X = cross_pairings(fields, fields)
    print('%d random smooth bc-satisfying fields (degree < %d, decay %g, Nq %d), pairings in %.1f s' % (nvec, kmax + 6, decay, Nq, time.time() - t0))
    print(' pair   inertia Herm.   Coriolis anti-Herm. /2sqrt(KExKEy)   viscous Herm.      KE(x)       Dkin(x)     ReC/|C|')
    worst = WORST
    def note(name, val):
        worst[name] = max(worst.get(name, 0.), val)
    for a in range(nvec):
        for b in range(a, nvec):
            xy, yx = X[a][b], X[b][a]
            eI  = abs(xy['I'] - np.conj(yx['I']))/abs(xy['I'])
            eC  = abs(xy['C'] + np.conj(yx['C']))/(2*np.sqrt(X[a][a]['I'].real*X[b][b]['I'].real))
            eV  = abs(xy['V'] - np.conj(yx['V']))/abs(xy['V'])
            note('inertia Hermitian', eI); note('Coriolis anti-Hermitian', eC); note('viscous Hermitian', eV)
            extra = ''
            if a == b:
                extra = '  %10.3e %10.3e  %9.2e' % (xy['I'].real, 2*xy['V'].real, xy['C'].real/abs(xy['C']))
                note('inertia positive', float(xy['I'].real <= 0)); note('viscous negative', float(xy['V'].real >= 0))
                note('Coriolis no work', abs(xy['C'].real)/(2*xy['I'].real))
            print(' (%d,%d)   %10.3e         %10.3e              %10.3e%s' % (a, b, eI, eC, eV, extra))
            if par.thermal:
                eE = abs(xy['E'] - np.conj(yx['E']))/abs(xy['E']); note('entropy Hermitian', eE)
                msg = '         entropy block Herm. %10.3e' % eE
                if a == b:
                    note('entropy positive', float(xy['E'].real <= 0))
                if par.ThermaD > 0:
                    eD = abs(xy['D'] - np.conj(yx['D']))/abs(xy['D']); note('diffusion Hermitian', eD)
                    msg += '   diffusion block Herm. %10.3e' % eD
                    if a == b:
                        msg += '   Dthm(s) = %10.3e' % (2*xy['D'].real)
                        note('diffusion negative', float(xy['D'].real >= 0))
                print(msg)
    if par.thermal:                                               # rows against direct quadratures of the fields
        for a in range(nvec):
            wb, wa, te = direct_quadratures(*fields[a])
            o = X[a][a]
            eB = abs(o['B'] - wb)/abs(wb); eA = abs(o['A'] - wa)/abs(wa); eE = abs(o['E'] - te)/abs(te)
            note('buoyancy rows vs quadrature', eB); note('advection rows vs quadrature', eA); note('entropy rows vs quadrature', eE)
            if a == 0:
                print('rows against direct quadratures (field 0): buoyancy %.2e   advection %.2e   entropy %.2e' % (eB, eA, eE))
    if par.diff_rot:   # differential rotation: every field split into its (P, s) and T parts, so that each coupling is compared on its own
        parts = [p for (P, T, S) in fields for p in ((P, 0*T, S), (0*P, T, 0*S))]
        Xs = cross_pairings(parts, parts)
        gf = [grid_fields(f) for f in parts]
        rel = lambda u, v: abs(u - v)/abs(v) if v != 0 else abs(u)
        rigid = (par.diff_rot_type == 'rigid')
        d  = par.diff_rot_ampl
        Om = np.max(abs(dOm)) or 1.
        for a in range(2*nvec):
            for b in range(2*nvec):
                o, g = Xs[a][b], grid_pairings(gf[a], gf[b])
                sI = np.sqrt(Xs[a][a]['I'].real*Xs[b][b]['I'].real)   # momentum errors relative to the Cauchy-Schwarz bounds
                for key, name, sc in [('I', 'inertia', 1), ('C', 'Coriolis', 2*G), ('R', 'diff-rot', G*Om)]:
                    note(name + ' rows vs grid', abs(o[key] - g[key])/(sc*sI))
                if rigid:   # Ω = Ω0 (1 + d): the diff-rot force is d (i m G u + 2G z x u) exactly
                    note('rigid: R = d(i m G I + C)', abs(o['R'] - d*(1j*m*G*o['I'] + o['C']))/(G*Om*sI))
                if par.thermal:   # the heat blocks carry the A-matrix sign: H = -(grid)
                    note('heat advection rows vs grid', rel(-o['Ht'], g['Ht']))
                    note('heat thermal-wind rows vs grid', abs(o['Hw'] + g['Hw'])/max(abs(g['Hw']), g['Hwa']) if g['Hwa'] > 0 else rel(-o['Hw'], g['Hw']))
                    sE = np.sqrt(Xs[a][a]['E'].real*Xs[b][b]['E'].real)
                    if sE > 0 and m != 0 and np.max(abs(dOm)) > 0:   # i m G ΔΩ s is anti-Hermitian, |<x,Ht y>| <= m G max|ΔΩ| sqrt(ExEy)
                        note('heat advection anti-Hermitian', abs(o['Ht'] + np.conj(Xs[b][a]['Ht']))/(m*G*Om*sE))
                    if rigid:
                        note('rigid: Ht = -i m G d E', rel(o['Ht'], -1j*m*G*d*o['E']))
        if rigid and par.thermal and par.thermal_wind != 3:   # S_tw = 0 (unless prescribed): the thermal-wind blocks must vanish (largest entry relative to the entropy blocks)
            note('rigid: thermal-wind blocks vanish', max([abs(b[0]).max() for l in lp for b in dr_blocks(l, 'h') if b[2] != 'S'], default=0.)
                                                      / max(abs(op.entropy(l, 'h', '', 0)).max() for l in lp))
        print('diff_rot (%s, Gaspard %g): rows vs (r, theta) grid quadrature, worst over pairs of P/T parts, /(Cauchy-Schwarz bound): inertia %.2e   Coriolis %.2e   diff-rot %.2e'
              % (par.diff_rot_type, G, worst['inertia rows vs grid'], worst['Coriolis rows vs grid'], worst['diff-rot rows vs grid']))
        if par.thermal:
            print('heat diff-rot (thermal_wind %d): advection i m G dOmega s vs grid %.2e   thermal wind u.grad S_tw vs grid %.2e   advection anti-Herm. %s'
                  % (par.thermal_wind, worst['heat advection rows vs grid'], worst['heat thermal-wind rows vs grid'],
                     '%.2e' % worst['heat advection anti-Hermitian'] if 'heat advection anti-Hermitian' in worst else 'n/a (m = 0)'))
        if rigid:
            print('rigid (d = %g): R = d(i m G I + C) %.2e' % (d, worst['rigid: R = d(i m G I + C)'])
                  + ('   Ht = -i m G d E %.2e   max|thermal-wind block|/max|entropy block| %.2e' % (worst['rigid: Ht = -i m G d E'],
                     worst.get('rigid: thermal-wind blocks vanish', np.nan)) if par.thermal else ''))
    if par.magnetic:   # NEW (2026-10-28)
        magnetic_checks(nvec, rng, kmax, decay, note)
    try:                                                         # KE against the independent physical-space code
        import utils4pp as upp
        P, T, S = fields[0]
        usol = [np.array([full_coeffs(P[k], iP) for k in range(nl)]), np.array([full_coeffs(T[k], iT) for k in range(nl)])]
        KE_pp = upp.kinetic_energy(usol, ricb, rcmb)
        eK = abs(X[0][0]['I'].real/KE_pp - 1); note('KE vs utils4pp', eK)
        print('KE of field 0: pairing %.12e, utils4pp.kinetic_energy %.12e, rel. diff %.2e' % (X[0][0]['I'].real, KE_pp, eK))
    except Exception as e:
        print('utils4pp comparison skipped:', e)
    bad = {k: v for k, v in worst.items() if not (v < args.tol)}
    if bad:
        print('ru_insane.py: FAIL (tol %g):' % args.tol + ''.join('  %s %.2e' % kv for kv in bad.items()))
        sys.exit(1)
    print('ru_insane.py: PASS, worst relative error %.2e (tol %g)' % (max(worst.values()), args.tol))


if __name__ == '__main__':
    main()
