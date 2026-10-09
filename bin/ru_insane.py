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
  row_u(F) = r^rpower_u (r.curl curl F)_l / L^3   (section u, rows for P_l)
  row_v(F) = r^rpower_v (r.curl F)_l / L^3        (section v, rows for T_l)
  row_h(.) = r^h (entropy equation)_l / L^3       (section h, rows for s_l)
and for any force F per unit mass, with P = 0 on the walls,
  int (rho0 u)*.F dV = sum_l n_l L^3 [ int rho0 P_l* r^(3-rpower_u) row_u(F)_l dr
                                      + int rho0 T_l* r^(3-rpower_v) row_v(F)_l dr ],
n_l = 4 pi/(2l+1), L = l(l+1).  Under this pairing the LBR energy equation requires
  inertia   <x,Iy> =  conj<y,Ix>, <x,Ix> = KE(x) > 0
  Coriolis  <x,Cy> = -conj<y,Cx>              (does no work; its (ln rho0)' terms come from div u != 0)
  viscous   <x,Vy> =  conj<y,Vx>, <x,Vx> < 0  (rho0^-1 div(rho0 nu S) with its derivatives of ln rho0 up to fourth order,
                                               together with the stress-free rows)
and, with the s* weight on the heat rows, the entropy block Hermitian positive and the diffusion
block Hermitian negative (divergence form, thermal bc rows).  The buoyancy, advection and
entropy pairings are also compared with direct quadratures of the same fields, which checks
the r-powers and L-factors of those rows, and KE with utils4pp.kinetic_energy.

A sign error, a missing rho0, a diffusion term not in divergence form or an inconsistent bc row
shows up at 1e-3 to 1 against a floor of 1e-12 (1e-8 for the viscous block at low l).  An error
that preserves the symmetry (a wrong but symmetric coefficient, a wrong nu profile) is invisible
here; that is what the comparison of the energy budget with spin_doctor is for.

The random fields have Chebyshev coefficients up to degree kmax (default N1/4) decaying as
exp(-k/decay) (default kmax/8), with the lowest coefficients adjusted in the least-squares
sense so that the bc rows hold exactly (a correction at high degree leaves bc residuals of 1e-6
because the stress-free rows carry (ln rho0)'' and k^4 factors, and that pollutes the test).
Not handled: the inviscid formulation (ViscosD = 0, rows multiplied by rho0^2), magnetic and
compositional blocks.
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

gb   = {'u': 4, 'v': 2, 'h': 2 if par.ThermaD > 0 else 0}        # Gegenbauer basis of each section
chop = {s: (gb[s] if ricb > 0 else gb[s]//2) for s in gb}         # rows reserved for bc's

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
    adj = int(f1 in ['gra', 'pdS', 'lh1', 'dSd'])
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
    dS0  = rap.prf.gradS(np.abs(rk), par.aux1, par.aux2, par.aux3, par.aux4, par.aux5)*np.sign(rk)

# pairing weights (functions of r) for each section
wgt = {'u': rho0 * rk**(3 - par.rpower_u),
       'v': rho0 * rk**(3 - par.rpower_v)}
if par.thermal:
    if par.ThermaD > 0:
        wgt['h'] = np.ones_like(rk)            # rows are r^2 (eq)/L^3
    elif divp0:
        wgt['h'] = pss0*rk                     # rows are r (eq/p0)/L^3
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
KEYS = ['I', 'C', 'V', 'B', 'E', 'A', 'D']

def cross_pairings(left, right):
    '''
    left, right: lists of fields (P, T, S), each an (nl, N1) raw coefficient array.
    out[a][b] = dict of pairings <left_a, Op right_b> summed over l:
      I = int (rho0 u)*.u (= KE),  C = int (rho0 u)*.(2 z x u),  V = int (rho0 u)*.F_visc,
      B = int (rho0 u)*.(Beyonce g s r^),  and with the s* weight on the heat rows:
      E = int p0 |s|^2,  A = -int p0 S' u_r s*,  D = ThermaD int s* div(kappa p0 grad s).
    '''
    na, nb = len(left), len(right)
    out = [[{k: 0j for k in KEYS} for b in range(nb)] for a in range(na)]
    for k, l in enumerate(lp):                                   # section u, rows for P_l
        I  = op.inertia(l, 'u', 'upol', 0)
        Cd = op.coriolis(l, 'u', 'upol', 0)[0]
        Co = [(op.coriolis(l, 'u', 'utor', i), i) for i in [-1, 1] if l + i in lt]
        V  = op.viscous_diffusion(l, 'u', 'upol', 0)
        Bu = op.buoyancy(l, 'u', '', 0) if par.thermal else None
        ys = []
        for (P, T, S) in right:
            yC = Cd @ P[k]
            for (mtx, offd), i in Co:
                yC = yC + mtx @ T[k + offd]
            ys.append((I @ P[k], yC, V @ P[k], Bu @ S[k] if par.thermal else None))
        for a, (P, T, S) in enumerate(left):
            Pv = vals_unknown(P[k], 'P')
            for b in range(nb):
                out[a][b]['I'] += pair('u', l, Pv, ys[b][0])
                out[a][b]['C'] += pair('u', l, Pv, ys[b][1])
                out[a][b]['V'] += pair('u', l, Pv, ys[b][2])
                if par.thermal:
                    out[a][b]['B'] += pair('u', l, Pv, ys[b][3])
    for k, l in enumerate(lt):                                   # section v, rows for T_l
        I  = op.inertia(l, 'v', 'utor', 0)
        Cd = op.coriolis(l, 'v', 'utor', 0)[0]
        Co = [(op.coriolis(l, 'v', 'upol', i), i) for i in [-1, 1] if l + i in lp]
        V  = op.viscous_diffusion(l, 'v', 'utor', 0)
        ys = []
        for (P, T, S) in right:
            yC = Cd @ T[k]
            for (mtx, offd), i in Co:
                yC = yC + mtx @ P[k + offd]
            ys.append((I @ T[k], yC, V @ T[k]))
        for a, (P, T, S) in enumerate(left):
            Tv = vals_unknown(T[k], 'T')
            for b in range(nb):
                out[a][b]['I'] += pair('v', l, Tv, ys[b][0])
                out[a][b]['C'] += pair('v', l, Tv, ys[b][1])
                out[a][b]['V'] += pair('v', l, Tv, ys[b][2])
    if par.thermal:                                              # section h, rows for s_l
        for k, l in enumerate(lp):
            E = op.entropy(l, 'h', '', 0)
            A = op.thermal_advection(l, 'h', 'upol', 0)
            D = op.thermal_diffusion(l, 'h', '', 0) if par.ThermaD > 0 else None
            ys = []
            for (P, T, S) in right:
                ys.append((E @ S[k], A @ P[k], D @ S[k] if par.ThermaD > 0 else None))
            for a, (P, T, S) in enumerate(left):
                Sv = vals_unknown(S[k], 'S')
                for b in range(nb):
                    out[a][b]['E'] += pair('h', l, Sv, ys[b][0])
                    out[a][b]['A'] += pair('h', l, Sv, ys[b][1])
                    if par.ThermaD > 0:
                        out[a][b]['D'] += pair('h', l, Sv, ys[b][2])
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
    worst = {}
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
    try:                                                          # KE against the independent physical-space code
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
