#!/usr/bin/env python3
'''
Shift-invert eigensolve of the kore pair A x = lambda B x (A.npz, B.npz written by assemble.py)
with scipy only: OP = (A - s B)^{-1} B, standard Arnoldi on OP, lambda = s + 1/mu.
Run in a run directory. Options: csolve.py [shift_real shift_imag [k]]
Writes eig.npz with lam (k,) and vecs (sizmat, k), sorted by decreasing growth rate,
and prints the eigenvalues.
'''
import sys, time
import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl

def load(name):
    z = np.load(name)
    return ss.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))

sr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
si = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
k  = int(sys.argv[3]) if len(sys.argv) > 3 else 20
s  = sr + 1j*si

t0 = time.time()
A = load('A.npz'); B = load('B.npz').astype(complex)
n = A.shape[0]
lu = ssl.splu((A - s*B).tocsc())
OP = ssl.LinearOperator((n, n), matvec=lambda x: lu.solve(B @ x), dtype=complex)
mu, vecs = ssl.eigs(OP, k=k, which='LM', tol=1e-13, maxiter=5000)
lam = s + 1/mu
order = np.argsort(-lam.real)
lam, vecs = lam[order], vecs[:, order]
# residual of the generalised problem for each pair
res = np.array([np.linalg.norm(A @ vecs[:, j] - lam[j]*(B @ vecs[:, j]))/np.linalg.norm(A @ vecs[:, j]) for j in range(len(lam))])
np.savez('eig.npz', lam=lam, vecs=vecs, res=res, shift=s)
print('n = %d, shift = %s, k = %d, %.1f s' % (n, s, k, time.time() - t0))
if np.all(abs(lam - s) < 1e-9*max(abs(s), 1)):   # no equilibration here (solve.py has Ruiz scaling)
    print('WARNING: every eigenvalue equals the shift, A - sB is numerically singular: try another shift or solve.py')
for j in range(len(lam)):
    print('%3d  sigma = %12.6f  omega = %12.6f   |Ax-lBx|/|Ax| = %.1e' % (j, lam[j].real, lam[j].imag, res[j]))
