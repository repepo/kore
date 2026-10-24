#!/usr/bin/env python3
'''
spectrum_check.py pick|fields|check: the forced solves of the random Reynolds-stress forcing (bin/solve.py with
forcing = 11) against the exact expected spectrum, computed here with scipy from the noise map. For a small run
(dense eigensolve, sizmat up to ~3000) with forcing = 11, after submatrices.py and assemble.py (A.npz at
forcing_frequency, B.npz, B_forced.npz). Run from the run directory; run_all.sh does the sequence.

The noise map M has one column per noise function (unit weight on one centre, one c_k, one l), so that every
realisation is b = M xi with xi complex circular, E|xi|^2 = 1. Then for the forced matrix F(w) = A - i (w - wf) B
    E[KE(w)] = sum over the columns of M of KE(F(w)^-1 M_col)      (exact, no sampling)
with KE the energy pairing of bin/ru_insane.py (checked against utils4pp.kinetic_energy).

  pick    dense eigensolve of A_rest x = lambda B x (A_rest = A + i wf B), picks the least damped mode with sigma < 0
          and no other eigenvalue within 4 |sigma|, writes rsf_freqs.npy (21 frequencies w_n + |sigma_n| (-5..5)) and mode.npz
  fields  after solve.py without rsf_frequencies: every column of the fields solves F(wf) x = b_j (residual), and
          column 0 of B_forced.npz is M xi_0 for the draws of numpy.random.default_rng([rsf_seed, 0])
  check   after solve.py with rsf_frequencies = rsf_freqs.npy: the KE of every realisation in rsf_ke.dat against
          utils4pp.kinetic_energy of F(w)^-1 b_j solved here (pass < 1e-8, relative to the largest), and the
          spectrum rsf_spectrum.dat (mean KE over rsf_nsamp realisations and its standard error) against the exact
          E[KE(w)]: passes if every |z| = |mean - exact|/stderr < 4.5 (the mean z^2 is printed; it is not judged,
          since the same realisations are used at every frequency). Also prints, for the mode, its energy from the
          left eigenvector,
              P_n = KE(x_n) E|y_n^H b|^2 / (|y_n^H B x_n|^2 2|sigma_n|),
          the normal-mode estimate E|<a_n, f>|^2/(2|sigma_n|) and the fluctuation-dissipation value Theta f_eff,
          f_eff = E|<a_n, f>|^2/(-2 Theta <a_n, V a_n>) (1 for white noise without taper; Theta includes rsf_scale).
'''
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'bin'))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.linalg as sla
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
from parameters import par
import utils as ut
import operators as op
import reynolds_stress as rs
import rsf_common as rc

def load(name):
    z = np.load(name)
    return ss.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))

A = load('A.npz').astype(complex)
B = load('B.npz').astype(complex)
wf = ut.wf
n = ut.n
q = rc.Quad(max(2*par.N + 2, 200))

def noise_map(bd):
    cols = []
    lp, lt = rc.lp, rc.lt
    for i, l in enumerate(lp):
        for k in (1, 2, 5):
            if rs.chi(k, l*(l + 1.)) == 0:
                continue
            W = np.diag(bd.amplitudes(l, k))
            Ws = [W if kk == k else None for kk in (1, 2, 5)]
            blk = np.zeros((ut.sizmat, W.shape[1]), dtype=complex)
            blk[i*ut.N1:(i + 1)*ut.N1] = bd.rows_u(l, *bd.forces_u(l, *Ws))
            cols.append(((i, k, 'u'), blk))
    for i, l in enumerate(lt):
        for k in (3, 6):
            if rs.chi(k, l*(l + 1.)) == 0:
                continue
            W = np.diag(bd.amplitudes(l, k))
            Ws = [W if kk == k else None for kk in (3, 6)]
            blk = np.zeros((ut.sizmat, W.shape[1]), dtype=complex)
            blk[n + i*ut.N1:n + (i + 1)*ut.N1] = bd.rows_v(l, bd.forces_v(l, *Ws))
            cols.append(((i, k, 'v'), blk))
    return cols

def ke(X):
    '''KE of each column of X (sizmat x K), energy pairing of the inertia rows'''
    X = np.asarray(X).reshape(ut.sizmat, -1)
    out = np.zeros(X.shape[1])
    for i, l in enumerate(rc.lp):
        p = X[i*ut.N1:(i + 1)*ut.N1]
        out += np.real(pair_cols('u', l, p, op.inertia(l, 'u', 'upol', 0) @ p, 'P'))
    for i, l in enumerate(rc.lt):
        t = X[n + i*ut.N1:n + (i + 1)*ut.N1]
        out += np.real(pair_cols('v', l, t, op.inertia(l, 'v', 'utor', 0) @ t, 'T'))
    return out

def pair_cols(sec, l, left, rows, kind):
    '''column by column pairing of the fields left (coefficients) with the rows (both N1 x K)'''
    c = np.zeros((par.N, left.shape[1]), dtype=complex)
    if par.ricb > 0:
        c[:] = left
    else:
        c[rc.istart[kind]::2] = left
    import numpy.polynomial.chebyshev as ch
    vals = ch.chebvander(q.xc, par.N - 1) @ c
    L = l*(l + 1.)
    return 4*np.pi/(2*l + 1)*L**3*np.sum((q.w*q.wgt[sec])[:, None]*np.conj(vals)*q.vals_row(rows, sec), axis=0)

def pairing(a, Y):
    '''<a, F> for the coefficient vector a and every column of rows Y (sizmat x K)'''
    out = 0.
    for i, l in enumerate(rc.lp):
        sl = slice(i*ut.N1, (i + 1)*ut.N1)
        out = out + pair_cols('u', l, np.repeat(a[sl, None], Y.shape[1], 1), Y[sl], 'P')
    for i, l in enumerate(rc.lt):
        sl = slice(n + i*ut.N1, n + (i + 1)*ut.N1)
        out = out + pair_cols('v', l, np.repeat(a[sl, None], Y.shape[1], 1), Y[sl], 'T')
    return out

def viscous(a):
    '''the viscous rows applied to the flow part of a'''
    y = np.zeros(ut.sizmat, dtype=complex)
    for i, l in enumerate(rc.lp):
        sl = slice(i*ut.N1, (i + 1)*ut.N1)
        y[sl] = op.viscous_diffusion(l, 'u', 'upol', 0) @ a[sl]
    for i, l in enumerate(rc.lt):
        sl = slice(n + i*ut.N1, n + (i + 1)*ut.N1)
        y[sl] = op.viscous_diffusion(l, 'v', 'utor', 0) @ a[sl]
    return y

mode = sys.argv[1] if len(sys.argv) > 1 else 'check'
bd = rs.Builder()
blocks = noise_map(bd)
M = np.hstack([b for _, b in blocks])
print('spectrum_check %s: N %d, lmax %d, m %d, symm %d, ricb %g, sizmat %d, %d noise functions'
      % (mode, par.N, par.lmax, par.m, par.symm, par.ricb, ut.sizmat, M.shape[1]))

if mode == 'pick':
    Arest = (A + 1j*wf*B).toarray()
    lam, vl, vr = sla.eig(Arest, B.toarray(), left=True, right=True)
    fin = np.isfinite(lam) & (np.abs(lam) < 1e3)
    gap = {}   # distance to the nearest other eigenvalue, for the damped modes isolated by 4 |sigma|
    for j in np.where(fin & (lam.real < 0) & (np.abs(lam.imag) < 1.8))[0]:
        g = np.min(np.abs(lam[fin & (np.arange(lam.size) != j)] - lam[j]))
        if g > 4*abs(lam[j].real):
            gap[j] = g
    if not gap:
        sys.exit('no isolated damped mode found')
    j = min(gap, key=lambda j: abs(lam[j].real))
    s, w = lam[j].real, lam[j].imag
    freqs = w + abs(s)*np.linspace(-5, 5, 21)
    np.save('rsf_freqs.npy', freqs)
    np.savez('mode.npz', lam=lam[j], x=vr[:, j], y=vl[:, j])
    print('  mode lambda = %.6f %+.6fi (nearest other eigenvalue %.4f away); scan %.5f .. %.5f'
          % (s, w, gap[j], freqs[0], freqs[-1]))
    sys.exit(0)

if mode == 'fields':
    Xr = np.loadtxt('real_flow.field', ndmin=2) + 1j*np.loadtxt('imag_flow.field', ndmin=2)
    if par.thermal:
        Xr = np.vstack([Xr, np.loadtxt('real_thermal.field', ndmin=2) + 1j*np.loadtxt('imag_thermal.field', ndmin=2)])
    bf = load('B_forced.npz').toarray()
    res = np.linalg.norm(A @ Xr - bf, axis=0)/np.linalg.norm(bf, axis=0)
    # column 0 from the noise map and the draws of rhs()
    rng = np.random.default_rng([par.rsf_seed, 0])
    nc = bd.c['r'].size
    Xu = (rng.standard_normal((len(rc.lp), 3, nc)) + 1j*rng.standard_normal((len(rc.lp), 3, nc)))/np.sqrt(2)
    Xv = (rng.standard_normal((len(rc.lt), 2, nc)) + 1j*rng.standard_normal((len(rc.lt), 2, nc)))/np.sqrt(2)
    xi = np.concatenate([(Xu[i, [1, 2, 5].index(k)] if s == 'u' else Xv[i, [3, 6].index(k)]) for (i, k, s), _ in blocks])
    e0 = np.linalg.norm(M @ xi - bf[:, 0])/np.linalg.norm(bf[:, 0])
    ok = res.max() < 1e-9 and e0 < 1e-12
    print('  %d solutions: max |A x - b|/|b| = %.1e;  B_forced column 0 = M xi_0 to %.1e   %s'
          % (Xr.shape[1], res.max(), e0, 'ok' if ok else 'FAIL'))
    sys.exit(0 if ok else 1)

# check
import utils4pp as upp
P, T = rc.test_fields(np.random.default_rng(3))
x0 = np.zeros(ut.sizmat, dtype=complex)
for i, l in enumerate(rc.lp):
    x0[i*ut.N1:(i + 1)*ut.N1] = P[l]
for i, l in enumerate(rc.lt):
    x0[n + i*ut.N1:n + (i + 1)*ut.N1] = T[l]
ke_pp = upp.kinetic_energy(upp.expand_reshape_sol(x0[:2*n], par.symm), par.ricb, ut.rcmb)
print('  KE pairing vs utils4pp.kinetic_energy on a smooth field: rel. diff %.1e' % abs(ke(x0)[0]/ke_pp - 1))
spec = np.loadtxt('rsf_spectrum.dat', ndmin=2)
kes = np.loadtxt('rsf_ke.dat', ndmin=2)
bf = load('B_forced.npz').toarray()
ll = ut.ell(par.m, par.lmax, par.symm)[2]
K = bf.shape[1]
z = []
dke = 0.
for row, krow in zip(spec, kes):
    w = row[0]
    lu = ssl.splu((A - 1j*(w - wf)*B).tocsc())
    X = lu.solve(M)
    Xb = lu.solve(bf)   # the realisations solve.py solved
    kb = np.array([upp.kinetic_energy(upp.expand_reshape_sol(Xb[:2*n, j], par.symm), par.ricb, ut.rcmb, ll)
                   for j in range(K)])
    dke = max(dke, abs(krow[0] - w)/abs(w) + np.abs(krow[1:1 + K] - kb).max()/np.abs(kb).max(),
              abs(row[1] - krow[1:1 + K].mean())/abs(row[1]))
    if not z:   # the pairing is the energy only if the boundary terms vanish (no-slip walls, or fields zero there)
        kp = [upp.kinetic_energy(upp.expand_reshape_sol(X[:2*n, j], par.symm), par.ricb, ut.rcmb) for j in range(3)]
        print('  KE pairing vs utils4pp.kinetic_energy on 3 forced solutions: rel. diff %.1e'
              % np.max(np.abs(ke(X[:, :3])/kp - 1)))
        print('        w            exact E[KE]      sampled mean KE    stderr       z')
    exact = ke(X).sum()
    z.append((row[1] - exact)/row[2])
    print('  %14.9f   %.6e     %.6e     %.1e   %+.2f' % (w, exact, row[1], row[2], z[-1]))
z = np.array(z)
ok = np.all(np.abs(z) < 4.5) and dke < 1e-8 and len(kes) == len(spec)
print('  KE of every realisation (rsf_ke.dat) against a scipy solve: max rel. diff %.1e' % dke)
print('  max |z| %.2f, mean z^2 %.2f (%d frequencies, rsf_nsamp %d)   %s' % (np.abs(z).max(), np.mean(z**2), z.size,
      par.rsf_nsamp, 'ok' if ok else 'FAIL'))

if os.path.exists('mode.npz'):
    md = np.load('mode.npz')
    lam, x, y = complex(md['lam']), md['x'], md['y']
    s = abs(lam.real)
    mu = np.vdot(y, B @ x)
    Eyb = np.sum(np.abs(M.conj().T @ y)**2)   # E|y^H b|^2
    Pn = ke(x)[0]*Eyb/(abs(mu)**2*2*s)
    a = x/np.sqrt(ke(x)[0])
    Eaf = np.sum(np.abs(pairing(a, M))**2)    # E|<a, f>|^2
    Vaa = np.real(pairing(a, viscous(a)[:, None])[0])
    theta = par.rsf_scale*float(par.rsf_theta) if not isinstance(par.rsf_theta, str) else np.nan
    print('  mode lambda = %.6f %+.6fi:  P_n (left eigenvector) %.4e,  normal-mode estimate %.4e,'
          ' Theta f_eff %.4e (f_eff %.4f, -<a,Va>/|sigma| %.4f)'
          % (lam.real, lam.imag, Pn, Eaf/(2*s), Eaf/(-2*Vaa), Eaf/(-2*Vaa*theta), -Vaa/s))
    lu = ssl.splu((A - 1j*(lam.imag - wf)*B).tocsc())
    peak = ke(lu.solve(M)).sum()
    print('  exact E[KE] at w_n %.4e = single-mode Lorentzian peak 2 P_n/|sigma| %.4e x %.4f'
          % (peak, 2*Pn/s, peak/(2*Pn/s)))
sys.exit(0 if ok else 1)
