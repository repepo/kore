#!/usr/bin/env python3
'''
ebal.py: matrix-level (spectral) energy and entropy budgets for kore anelastic (LBR) runs.

Run from a kore run directory (needs the *.mtx submatrices, bin/, and the *.field files
for the per-mode budget):

    python3 ebal.py            # per-mode budget + operator symmetry tests
    python3 ebal.py modes      # per-mode budget only
    python3 ebal.py symm       # operator symmetry tests only (no eigensolve needed)

Idea
----
kore solves the curl-curl (section u, rows for P) and curl (section v, rows for T) of the
momentum equation per unit mass, with u = rho0^-1 curl curl (rho0 P r) + curl (T r).
For any vector field F (and P = 0 on the walls):

    int (rho0 u)* . F dV  =  sum_l n_l L^3 [ int rho0 P_l* r^(3-rpower_u) row_u(F)_l dr
                                           + int rho0 T_l* r^(3-rpower_v) row_v(F)_l dr ]

where row_u(F) = r^rpower_u (r^.curl curl F)_l / L^3 is exactly what operators.py produces
(section u), row_v(F) = r^rpower_v (r^.curl F)_l / L^3 (section v), n_l = 4 pi/(2l+1) and
L = l(l+1).  So the power of every force in the energy equation is a sparse matrix-vector
product followed by one radial quadrature; no reconstruction of the fields in physical
space is needed, and every term that enters A or B gets a budget line automatically.

Section h (heat) rows are r^h (entropy equation) / L^3; pairing them with s_l* gives
the entropy budget 2 sigma TE = W_adv + ThermaD D_thm.

Conventions: KE = int rho0 |u|^2 dV and Dkin = 2 Re int rho0 u*.F_visc dV etc., i.e. the same
quantities as spin_doctor (whose dotprod_* carry a factor 2).
'''

import sys, os, time
sys.path.insert(0, os.path.abspath('bin'))
import numpy as np
import scipy.sparse as ss
import scipy.linalg as sla
import numpy.polynomial.chebyshev as ch
from parameters import par
import utils as ut
import radial_profiles as rap
import operators as op
import bc_variables as bv
import diff_rot as dr

N    = par.N
N1   = ut.N1
ricb = par.ricb
rcmb = ut.rcmb
lp, lt, ll = ut.ell(par.m, par.lmax, par.symm)
nl   = len(lp)
m    = par.m

inviscid = (par.ViscosD == 0)
if inviscid:
    sys.exit('ebal.py: the inviscid formulation (rows multiplied by rho^2) is not handled')

divp0 = hasattr(op, 'h0_D0')   # ThermaD=0 heat rows divided by p0 (divp0 patch)

gb   = {'u': 4, 'v': 2, 'h': 2 if par.ThermaD > 0 else int(ut.adiab_tw)}   # Gegenbauer basis of each section (C^(1) for ThermaD = 0 with the thermal wind)
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
Nq = int(os.environ.get('EBAL_NQ', 2*N + 2))   # even, so r=0 is never a node when ricb=0
xk, wk = np.polynomial.legendre.leggauss(Nq)
if ricb > 0:
    rk   = 0.5*(rcmb - ricb)*(xk + 1) + ricb
    jac  = 0.5*(rcmb - ricb)
else:
    rk   = rcmb*xk                             # integrands are even in r: int_0^1 = 1/2 int_-1^1
    jac  = 0.5*rcmb

rho0 = np.exp(rap.logrhoX(rk, 0))
if par.thermal:
    pss0 = rap.pressX(rk, 0)
    gra0 = rap.graviX(rk, 0)
    dS0  = rap.prf.gradS(np.abs(rk))*np.sign(rk)

# pairing weights (functions of r) for each section, see the docstring
wgt = {'u': rho0**(1 - par.rhopower_u) * rk**(3 - par.rpower_u),
       'v': rho0**(1 - par.rhopower_v) * rk**(3 - par.rpower_v)}
if par.thermal:
    if par.ThermaD > 0:
        wgt['h'] = np.ones_like(rk)            # rows are r^2 (eq)/L^3
    elif divp0:
        wgt['h'] = pss0 if ut.adiab_tw else pss0*rk   # rows are r (eq/p0)/L^3, r^2 (eq/p0)/L^3 with the thermal wind
    else:
        wgt['h'] = rk                          # rows are r (eq)/L^3

# Chebyshev values at the nodes, and C^(lamb) -> values maps
Tmat = ch.chebvander(xk, N - 1)                # (Nq, N)
Emap = {}
for lamb in set(gb.values()):
    S = ss.identity(N, format='csr')
    for j in range(lamb):
        S = ut.Slam(j, N) @ S
    if lamb > 0 and os.environ.get('EBAL_DIRECT', '1') == '1':
        # values of the Gegenbauer polynomials C^(lamb)_k at the nodes by the three-term recurrence
        # (stable); the alternative, inverting S, has condition ~ N^lamb and leaves ~1e-9 errors
        E = np.zeros((Nq, N)); E[:, 0] = 1.; E[:, 1] = 2*lamb*xk
        for k in range(2, N):
            E[:, k] = (2*(k + lamb - 1)*xk*E[:, k-1] - (k + 2*lamb - 2)*E[:, k-2])/k
        Emap[lamb] = E
    else:
        Sinv = sla.solve_triangular(S.toarray(), np.eye(N))   # S is upper triangular
        Emap[lamb] = Tmat @ Sinv               # values of a C^(lamb) series at the nodes


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
    L  = l*(l + 1.)
    nl_ = 4*np.pi/(2*l + 1)
    return nl_ * L**3 * jac * np.sum(wk * wgt[sec] * np.conj(left_vals) * vals_row(y, sec))


# ---------------------------------------------------------------------------- operator applications
KEYS = ['I', 'C', 'V', 'B', 'R', 'E', 'A', 'D', 'H']


def dr_blocks(l, sec):
    '''
    differential-rotation blocks of row l in section sec (diff_rot = 1): list of (mtx, offd, kind), kind the
    unknown ('P' or 'T') of the column block k + offd, as assemble.py adds them (R = (U0.grad)u + (u.grad)U0)
    '''
    if not par.diff_rot:
        return []
    comp = {'u': [('upol', dr.dl_range('u', 'upol'), lp, 'P'), ('utor', dr.dl_range('u', 'utor'), lt, 'T')],
            'v': [('upol', dr.dl_range('v', 'upol'), lp, 'P'), ('utor', dr.dl_range('v', 'utor'), lt, 'T')]}[sec]
    out = []
    for name, offs, ls, kind in comp:
        for i in offs:
            if l + i in ls:
                mtx, offd = op.differential_rotation(l, sec, name, i)
                if not np.isscalar(mtx):
                    out.append((mtx, offd, kind))
    return out

def heat_blocks(l):
    '''
    differential-rotation blocks of the heat row l (diff_rot = 1 and thermal = 1): list of (mtx, offd, kind) as
    assemble.py adds them, kind the unknown of the column block k + offd. The blocks carry the sign of the A-matrix
    right-hand side, so pairing them gives H = -<s, p0 [i m dOmega s + u.grad S_tw]> and the balance is
    lambda E - A - D - H = 0.
    '''
    if not (par.diff_rot and par.thermal):
        return []
    comp = [('temp', dr.dl_range('h', 'temp'), lp, 'S'), ('upol', dr.dl_range('h', 'upol'), lp, 'P'),
            ('utor', dr.dl_range('h', 'utor'), lt, 'T')]   # the last two are empty without the thermal wind
    out = []
    for name, offs, ls, kind in comp:
        for i in offs:
            if l + i in ls:
                mtx, offd = op.thermal_diffrot(l, 'h', name, i)
                if not np.isscalar(mtx):
                    out.append((mtx, offd, kind))
    return out

def cross_pairings(left, right):
    '''
    left, right: lists of fields (P, T, S), each P, T, S an (nl, N1) raw coefficient array (S may be None).
    Returns out[a][b] = dict of pairings <left_a, Op right_b> summed over l, for
      I = int (rho0 u)*.u  (= KE),  C = int (rho0 u)*.(2 z x u),  V = int (rho0 u)*.F_visc,
      B = int (rho0 u)*.(Beyonce g s r^),  and with the s* weight on the heat rows:
      E = int p0 |s|^2,  A = -int p0 S' u_r s*,  D = ThermaD int s* div(kappa p0 grad s).
    The operator blocks are built once per l and applied to every right field.
    '''
    na, nb = len(left), len(right)
    out = [[{k: 0j for k in KEYS} for b in range(nb)] for a in range(na)]
    # section u, rows for P_l
    for k, l in enumerate(lp):
        I = op.inertia(l, 'u', 'upol', 0)
        Cd = op.coriolis(l, 'u', 'upol', 0)[0]
        Co = [(op.coriolis(l, 'u', 'utor', i), i) for i in [-1, 1] if l + i in lt]
        V = op.viscous_diffusion(l, 'u', 'upol', 0)
        Bu = op.buoyancy(l, 'u', '', 0) if par.thermal else None
        Rb = dr_blocks(l, 'u')
        ys = []
        for (P, T, S) in right:
            yC = Cd @ P[k]
            for (mtx, offd), i in Co:
                yC = yC + mtx @ T[k + offd]
            yR = 0
            for mtx, offd, kind in Rb:
                yR = yR + mtx @ (P if kind == 'P' else T)[k + offd]
            ys.append((I @ P[k], yC, V @ P[k], Bu @ S[k] if par.thermal else None, yR))
        for a, (P, T, S) in enumerate(left):
            Pv = vals_unknown(P[k], 'P')
            for b in range(nb):
                out[a][b]['I'] += pair('u', l, Pv, ys[b][0])
                out[a][b]['C'] += pair('u', l, Pv, ys[b][1])
                out[a][b]['V'] += pair('u', l, Pv, ys[b][2])
                if par.thermal:
                    out[a][b]['B'] += pair('u', l, Pv, ys[b][3])
                if par.diff_rot:
                    out[a][b]['R'] += pair('u', l, Pv, ys[b][4])
    # section v, rows for T_l
    for k, l in enumerate(lt):
        I = op.inertia(l, 'v', 'utor', 0)
        Cd = op.coriolis(l, 'v', 'utor', 0)[0]
        Co = [(op.coriolis(l, 'v', 'upol', i), i) for i in [-1, 1] if l + i in lp]
        V = op.viscous_diffusion(l, 'v', 'utor', 0)
        Rb = dr_blocks(l, 'v')
        ys = []
        for (P, T, S) in right:
            yC = Cd @ T[k]
            for (mtx, offd), i in Co:
                yC = yC + mtx @ P[k + offd]
            yR = 0
            for mtx, offd, kind in Rb:
                yR = yR + mtx @ (P if kind == 'P' else T)[k + offd]
            ys.append((I @ T[k], yC, V @ T[k], yR))
        for a, (P, T, S) in enumerate(left):
            Tv = vals_unknown(T[k], 'T')
            for b in range(nb):
                out[a][b]['I'] += pair('v', l, Tv, ys[b][0])
                out[a][b]['C'] += pair('v', l, Tv, ys[b][1])
                out[a][b]['V'] += pair('v', l, Tv, ys[b][2])
                if par.diff_rot:
                    out[a][b]['R'] += pair('v', l, Tv, ys[b][3])
    # section h, rows for s_l
    if par.thermal:
        for k, l in enumerate(lp):
            E = op.entropy(l, 'h', '', 0)
            A = op.thermal_advection(l, 'h', 'upol', 0)
            D = op.thermal_diffusion(l, 'h', '', 0) if par.ThermaD > 0 else None
            Hb = heat_blocks(l)
            ys = []
            for (P, T, S) in right:
                yH = 0
                for mtx, offd, kind in Hb:
                    yH = yH + mtx @ {'P': P, 'T': T, 'S': S}[kind][k + offd]
                ys.append((E @ S[k], A @ P[k], D @ S[k] if par.ThermaD > 0 else None, yH))
            for a, (P, T, S) in enumerate(left):
                Sv = vals_unknown(S[k], 'S')
                for b in range(nb):
                    out[a][b]['E'] += pair('h', l, Sv, ys[b][0])
                    out[a][b]['A'] += pair('h', l, Sv, ys[b][1])
                    if par.ThermaD > 0:
                        out[a][b]['D'] += pair('h', l, Sv, ys[b][2])
                    if Hb:
                        out[a][b]['H'] += pair('h', l, Sv, ys[b][3])
    return out


def direct_quadratures(P, T, S):
    '''
    Direct quadratures from the fields alone (no operators), to check the r-powers and
    L-factors of the buoyancy, advection and entropy rows:
      W_buo = int rho0 Beyonce g u_r* s dV,  W_adv = -int p0 S' u_r s* dV,  TE = int p0 |s|^2 dV,
    with u_r = L P / r.
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


# ------------------------------------------------------------------------------------- bc rows
def bc_rows(l, sec):
    '''dense bc rows for one l block, as in assemble.bc_u_spherical (ricb>0 and ricb=0)'''
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
    elif sec == 'h' and par.ThermaD > 0:   # as in assemble.bc_theta_spherical: s=0 (0) or s'=0 (1)
        Tbh = bv.Tb if ricb > 0 else bv.Tb[ixu::2, :]
        rows.append(Tbh[:, par.bco_thermal])
        if ricb > 0:
            rows.append(bv.Ta[:, par.bci_thermal])
    if len(rows) == 0:
        return np.zeros((0, N1), dtype=complex)
    return np.array(rows, dtype=complex)


def enforce_bc(x, Bc, kmax=None):
    '''
    Adjust the lowest nbc coefficients of x so that Bc x = 0, so that a smooth x stays smooth
    (a Euclidean projection would put the correction into the highest Chebyshev orders, since
    the bc rows grow like k^2 and k^4; a correction at degree kmax adds a non-decaying tail there).
    '''
    nbc = Bc.shape[0]
    if nbc == 0:
        return x
    cols = np.arange(min(nbc + 6, len(x)))      # minimal-norm correction within the lowest orders
    x = x.copy()
    x[cols] -= np.linalg.lstsq(Bc[:, cols], Bc @ x, rcond=None)[0]
    return x


def random_fields(rng, kmax, decay=0.):
    '''random smooth (degree < kmax + 4) Chebyshev coefficient fields satisfying the velocity bc's;
    decay > 0 multiplies the coefficients by exp(-k/decay)'''
    win = np.exp(-np.arange(kmax)/decay) if decay > 0 else np.ones(kmax)
    lmax_test = int(os.environ.get('EBAL_LMAX_TEST', 10**6))   # restrict the fields to l <= lmax_test
    P = np.zeros((nl, N1), dtype=complex)
    T = np.zeros((nl, N1), dtype=complex)
    S = np.zeros((nl, N1), dtype=complex)
    for k, l in enumerate(lp):
        P[k, :kmax] = win*(rng.standard_normal(kmax) + 1j*rng.standard_normal(kmax))
        P[k] = enforce_bc(P[k], bc_rows(l, 'u'), kmax)
        S[k, :kmax] = win*(rng.standard_normal(kmax) + 1j*rng.standard_normal(kmax))
        S[k] = enforce_bc(S[k], bc_rows(l, 'h'), kmax)
    for k, l in enumerate(lt):
        T[k, :kmax] = win*(rng.standard_normal(kmax) + 1j*rng.standard_normal(kmax))
        T[k] = enforce_bc(T[k], bc_rows(l, 'v'), kmax)
    for k, l in enumerate(lp):
        if l > lmax_test: P[k] = 0; S[k] = 0
    for k, l in enumerate(lt):
        if l > lmax_test: T[k] = 0
    return P, T, S


# ---------------------------------------------------------------------------------------- drivers
def budget_modes():
    t0 = time.time()
    ru = np.loadtxt('real_flow.field').reshape((2*ut.n, -1))
    iu = np.loadtxt('imag_flow.field').reshape((2*ut.n, -1))
    eig = np.loadtxt('eigenvalues0.dat').reshape((-1, 2))
    nev = ru.shape[1]
    if par.thermal:
        rt = np.loadtxt('real_thermal.field').reshape((ut.n, -1))
        it = np.loadtxt('imag_thermal.field').reshape((ut.n, -1))
    flow = np.loadtxt('flow.dat') if os.path.isfile('flow.dat') else None
    thd  = np.loadtxt('thermal.dat') if (par.thermal and os.path.isfile('thermal.dat')) else None
    print('fields loaded in %.1f s' % (time.time() - t0))

    fields = []
    for i in range(nev):
        sol = ru[:, i] + 1j*iu[:, i]
        P = sol[:ut.n].reshape((nl, N1))
        T = sol[ut.n:2*ut.n].reshape((nl, N1))
        S = (rt[:, i] + 1j*it[:, i]).reshape((nl, N1)) if par.thermal else None
        fields.append((P, T, S))
    t0 = time.time()
    outs = [cross_pairings([f], [f])[0][0] for f in fields]
    t1 = time.time() - t0

    print('\nPer-mode matrix-level budget: pairings of the eigenvector with the operator rows.')
    print('KE = Re<u,u>, Dkin = 2Re<u,F_visc>, Wbuo = 2Re<u,F_buo> compared with spin_doctor (flow.dat, thermal.dat) as spec/sd - 1.')
    print('ReC/|C|: Coriolis work (0 if the Coriolis block is anti-Hermitian); ImV/|V|: imaginary part of the viscous')
    print('pairing (0 if Hermitian); res: |lambda I + C - V - B|/max(|.|), the complex balance of the momentum rows.\n')
    hdr = ' mode      sigma         omega       KE_spec     dKE/KE    dDkin/Dkin   ReC/|C|    ImV/|V|    ImI/|I|     res'
    if par.thermal:
        hdr += '    dWbuo/Wbuo  ImB/|B|     TE_spec    dTE/TE   dWadv/Wadv dDthm/Dthm   ImE/|E|    res_s'
    print(hdr)
    for i in range(nev):
        lam = eig[i, 0] + 1j*eig[i, 1]
        o = outs[i]
        KE = o['I'].real
        bal = lam*o['I'] + o['C'] + o['R'] - o['V'] - o['B']   # R = 0 unless diff_rot
        res = abs(bal)/max(abs(lam*o['I']), abs(o['C']), abs(o['V']), abs(o['B']), abs(o['R']))
        dKE = KE/flow[i, 0] - 1 if flow is not None else np.nan
        dDk = 2*o['V'].real/flow[i, 3] - 1 if flow is not None else np.nan
        line = ' %3d  %12.9f  %12.9f  %11.5e  %9.2e  %9.2e  %9.2e  %9.2e  %9.2e  %9.2e' % (
            i, lam.real, lam.imag, KE, dKE, dDk, o['C'].real/abs(o['C']), o['V'].imag/abs(o['V']),
            o['I'].imag/abs(o['I']), res)
        if par.thermal:
            TE = o['E'].real
            bal_s = lam*o['E'] - o['A'] - o['D'] - o['H']
            res_s = abs(bal_s)/max(abs(lam*o['E']), abs(o['A']), abs(o['D']), abs(o['H']))
            if thd is not None:
                dWb = 2*o['B'].real/thd[i, 4] - 1
                dTE = TE/(2*thd[i, 0]) - 1
                dWa = 2*o['A'].real/thd[i, 1] - 1
                dDt = 2*o['D'].real/thd[i, 2] - 1 if par.ThermaD > 0 else np.nan
            else:
                dWb = dTE = dWa = dDt = np.nan
            line += '  %9.2e  %9.2e  %11.5e %9.2e  %9.2e  %9.2e  %9.2e  %9.2e' % (
                dWb, o['B'].imag/abs(o['B']), TE, dTE, dWa, dDt, o['E'].imag/abs(o['E']), res_s)
        print(line)
    print('\n%d modes in %.2f s' % (nev, t1))
    if par.thermal:
        P, T, S = fields[-1]; o = outs[-1]
        wb, wa, te = direct_quadratures(P, T, S)
        print('direct quadrature check (last mode): Wbuo %.9e vs pairing %.9e; Wadv %.9e vs %.9e; TE %.9e vs %.9e'
              % (wb.real, o['B'].real, wa.real, o['A'].real, te.real, o['E'].real))


def symmetry_tests(nvec=3, seed=0):
    '''
    Bilinear symmetry of the operator blocks under the energy pairing, with random smooth
    fields x, y that satisfy the velocity (and thermal) bc's (no eigensolve needed):
      inertia  <x,Iy> = conj<y,Ix>  (Hermitian, positive: <x,Ix> = KE(x) > 0)
      Coriolis <x,Cy> = -conj<y,Cx> (anti-Hermitian: does no work)
      viscous  <x,Vy> = conj<y,Vx>  (Hermitian, negative: 2<x,Vx> = Dkin(x) < 0)
      entropy and diffusion blocks likewise, with the s* weight.
    The Coriolis error is printed relative to |<x,Cy>| and to the inertia scale 2 sqrt(KE(x)KE(y)):
    for random fields dominated by u_r (large l) the Coriolis pairing is ~1/L smaller than the
    inertia one (z x u_r r^ is orthogonal to u_r r^), and the first ratio overstates the error.
    EBAL_KMAX, EBAL_DECAY (coefficients ~ exp(-k/decay)) and EBAL_NQ override the defaults.
    '''
    rng = np.random.default_rng(seed)
    kmax  = int(os.environ.get('EBAL_KMAX', N1//4))
    decay = float(os.environ.get('EBAL_DECAY', kmax/8))
    fields = [random_fields(rng, kmax, decay) for _ in range(nvec)]
    t0 = time.time()
    X = cross_pairings(fields, fields)
    t1 = time.time() - t0
    print('\nOperator symmetry tests with %d random smooth bc-satisfying fields (degree < %d, decay %g, Nq %d), %.1f s' % (nvec, kmax + 4, decay, Nq, t1))
    print(' pair   inertia Herm.   Coriolis anti-Herm.: /|C|  /2sqrt(KExKEy)   viscous Herm.      KE(x)       Dkin(x)    ReC/|C|')
    for a in range(nvec):
        for b in range(a, nvec):
            xy, yx = X[a][b], X[b][a]
            eI = abs(xy['I'] - np.conj(yx['I']))/abs(xy['I'])
            eC = abs(xy['C'] + np.conj(yx['C']))/abs(xy['C'])
            eCI = abs(xy['C'] + np.conj(yx['C']))/(2*np.sqrt(X[a][a]['I'].real*X[b][b]['I'].real))
            eV = abs(xy['V'] - np.conj(yx['V']))/abs(xy['V'])
            extra = ''
            if a == b:
                extra = '  %10.3e %10.3e  %9.2e' % (xy['I'].real, 2*xy['V'].real, xy['C'].real/abs(xy['C']))
            print(' (%d,%d)   %10.3e         %10.3e  %10.3e      %10.3e%s' % (a, b, eI, eC, eCI, eV, extra))
            if par.thermal:
                eE = abs(xy['E'] - np.conj(yx['E']))/abs(xy['E'])
                msg = '         entropy block Herm. %10.3e' % eE
                if par.ThermaD > 0:
                    eD = abs(xy['D'] - np.conj(yx['D']))/abs(xy['D'])
                    msg += '   diffusion block Herm. %10.3e' % eD
                    if a == b:
                        msg += '   Dthm(s) = %10.3e' % (2*xy['D'].real)
                print(msg)

    # the same quantities from the independent physical-space code (utils4pp), for one random field
    try:
        import utils4pp as upp
        P, T, S = fields[0]
        usol = [np.array([full_coeffs(P[k], iP) for k in range(nl)]), np.array([full_coeffs(T[k], iT) for k in range(nl)])]
        KE_pp = upp.kinetic_energy(usol, ricb, rcmb)
        KE_sp = X[0][0]['I'].real
        print('\nKE of random field 0: pairing %.12e, utils4pp.kinetic_energy %.12e, rel. diff %.2e' % (KE_sp, KE_pp, KE_sp/KE_pp - 1))
    except Exception as e:
        print('utils4pp comparison skipped:', e)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    print('ebal.py: N=%d, lmax=%d, m=%d, ricb=%g, thermal=%d, ThermaD=%g, divp0=%s, Nq=%d' % (N, par.lmax, m, ricb, par.thermal, par.ThermaD, divp0, Nq))
    if what in ['all', 'modes']:
        budget_modes()
    if what in ['all', 'symm']:
        symmetry_tests()
