#!/usr/bin/env python3
'''
asm_check.py: check the assembled A matrix (A.npz) of a diff_rot = 1 run against the grid side of heat_check.py.
The heat rows of A(diff_rot = 1) - A(diff_rot = 0), both runs otherwise identical, are exactly the new blocks as
assemble.py packed them; paired with conj(s) they must give -G, G the (r, theta) quadrature of
p0 conj(s) [i m Gaspard dOmega s + u_r dS_tw/dr + (u_theta/r) dS_tw/dtheta] from the same random fields.
Run in the diff_rot = 1 run directory: python3 asm_check.py PATH_TO_A_npz_OF_THE_diff_rot_0_RUN [ntrials]
With ThermaD = 0 and thermal_wind >= 1 the heat rows are r^2 (eq)/p in C^(1) (utils.adiab_tw), and a diff_rot = 0 (or
thermal_wind = 0) run has r (eq)/p in C^(0): the reference is then the same run with diff_rot_ampl = 0.
'''
import sys, os, numpy as np, scipy.sparse as ss, scipy.sparse.linalg
a0path, ntr_ = sys.argv[1], (int(sys.argv[2]) if len(sys.argv) > 2 else 3)
sys.argv = ['heat_check.py', str(ntr_)]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'heat_check.py')).read().split('# ---- run')[0])
def load_kore(name):   # A.npz as assemble.py writes it (data, indices, indptr, shape)
    z = np.load(name)
    return ss.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
A1 = load_kore('A.npz')
A0 = load_kore(a0path)
nb = int((par.lmax - par.m + 1)/2)   # block rows per section, as assemble.py
assert nb*N1 == ut.n and len(lp) == nb, 'kore needs lmax - m + 1 even (lp and lt with nb = (lmax - m + 1)/2 degrees each)'
assert A1.shape == A0.shape == (3*ut.n, 3*ut.n), (A1.shape, A0.shape, nb, N1)
h0 = 2*nb*N1
# assemble.py divides A and B by the Frobenius norm of A (a different number in each run): recover it from the
# entropy block of B, which is op.entropy(l) unscaled
def unscale(Apath, Bpath):
    A = load_kore(Apath); B = load_kore(Bpath)
    k = 0; l = lp[0]
    blk = B[h0 + k*N1 + chop['h']:h0 + (k + 1)*N1, h0 + k*N1:h0 + (k + 1)*N1]
    raw = ss.csr_matrix(op.entropy(l, 'h', '', 0))[chop['h']:, :]
    Anorm = ss.linalg.norm(raw)/ss.linalg.norm(blk)
    assert ss.linalg.norm(blk*Anorm - raw) < 1e-10*ss.linalg.norm(raw), 'B block is not a multiple of op.entropy (other heat rows?)'
    return A*Anorm, Anorm
A1, n1 = unscale('A.npz', 'B.npz')
A0, n0 = unscale(a0path, os.path.join(os.path.dirname(a0path), 'B.npz'))
print('A normalisation undone: Anorm = %.6g (diff_rot = 1), %.6g (diff_rot = 0)' % (n1, n0))
dA = (A1 - A0).tocsr()
dh = dA[h0:3*nb*N1, :]
print('m = %d, symm = %d, ricb = %g, lmax = %d, N = %d, diff_rot_type = %s, diff_rot_ampl = %g, thermal_wind = %d, Gaspard = %g, ThermaD = %g' % (m, par.symm, ricb, par.lmax, N, par.diff_rot_type, par.diff_rot_ampl, par.thermal_wind, par.Gaspard, par.ThermaD))
print('rows of A outside section h that differ: %d (momentum diff-rot blocks)' % np.count_nonzero(np.diff(dA[:h0, :].indptr)))
bcrows = np.concatenate([k*N1 + np.arange(chop['h']) for k in range(nb)])
worst = 0
if len(bcrows):   # none with ThermaD = 0; the pairing below skips these rows, so they enter the verdict here
    worst = abs(dh[bcrows, :]).max()/abs(A1[h0 + bcrows, :]).max()
    print('bc rows of section h: max |difference| / max |A| = %.1e (must be roundoff)' % worst)
for t in range(ntr_):
    P, T, S = random_fields(max(4, N1//3), 0.7)
    x = np.concatenate([P.ravel(), T.ravel(), S.ravel()])
    y = dh @ x
    XH = 0
    for k, l in enumerate(lp):
        XH += pair('h', l, vals_unknown(S[k], 'S'), y[k*N1:(k + 1)*N1])
    H = -XH
    G, Ga = grid_side(P, T, S)
    err = abs(H - G)/max(abs(G), Ga, 1e-300); worst = max(worst, err if np.isfinite(err) else np.inf)
    print(' trial %d: assembled %s  grid %s  rel diff %.1e' % (t, H, G, err))
print('worst relative difference %.1e  ->  %s' % (worst, 'ok' if worst < 1e-8 else 'FAIL'))
