#!/usr/bin/env python3
'''
solve_bekki.py RUN freq [growth [k [ref]]]: shift-invert solve of a run built by mk_bekki.py near the mode of frequency
freq and growth rate growth (nHz), with scipy only. Frequencies are reported in a frame rotating at ref nHz (456, the
frame of the Mukhopadhyay et al. 2025 tables, by default), whatever the run's own frame: f = -ω Ω_ref/2π + m (frame - ref),
negative is retrograde; the growth rate is σ Ω_ref/2π. Ruiz-equilibrates A - s B as kore's solve.py does (prescale = 1).
Prints the k eigenvalues (default 10) nearest the shift and writes RUN/eig.npz (f, g, lam, vecs, res).
'''
import sys
import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl

if len(sys.argv) < 3:
    sys.exit(__doc__)
d, tf = sys.argv[1], float(sys.argv[2])
tg  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
k   = int(sys.argv[4]) if len(sys.argv) > 4 else 10
ref = float(sys.argv[5]) if len(sys.argv) > 5 else 456.0
info = dict(l.split() for l in open(d + '/bekki.txt'))
m, frame = int(info['m']), float(info['frame'])
off = m*(frame - ref)

def load(name):
    z = np.load(d + '/' + name)
    return ss.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))

A = load('A.npz'); B = load('B.npz').astype(complex)
n = A.shape[0]
s = tg/frame - 1j*(tf - off)/frame
M = (A - s*B).tocoo(); a0 = np.abs(M.data)
dr = np.ones(n); dc = np.ones(n)
for it in range(10):
    a = a0*dr[M.row]*dc[M.col]
    rm = np.zeros(n); np.maximum.at(rm, M.row, a); dr /= np.sqrt(np.where(rm > 0, rm, 1))
    a = a0*dr[M.row]*dc[M.col]
    cm = np.zeros(n); np.maximum.at(cm, M.col, a); dc /= np.sqrt(np.where(cm > 0, cm, 1))
Dr, Dc = ss.diags(dr), ss.diags(dc)
As = (Dr @ A @ Dc).tocsr(); Bs = (Dr @ B @ Dc).tocsr()
lu = ssl.splu((As - s*Bs).tocsc())
mu, y = ssl.eigs(ssl.LinearOperator((n, n), matvec=lambda x: lu.solve(Bs @ x), dtype=complex),
                 k=k, which='LM', tol=1e-13, maxiter=20000)
vecs = Dc @ y
lam = s + 1/mu
order = np.argsort(np.abs(lam - s))
lam, vecs = lam[order], vecs[:, order]
res = np.array([np.linalg.norm(A @ vecs[:, j] - lam[j]*(B @ vecs[:, j]))/np.linalg.norm(A @ vecs[:, j]) for j in range(k)])
f, g = -lam.imag*frame + off, lam.real*frame
np.savez(d + '/eig.npz', f=f, g=g, lam=lam, vecs=vecs, res=res)
print('%s: n = %d, m = %d, frame %g nHz, reported in a frame at %g nHz' % (d, n, m, frame, ref))
for j in range(k):
    print('%3d  f = %11.4f nHz  growth = %+9.4f nHz   |Ax-lBx|/|Ax| = %.1e' % (j, f[j], g[j], res[j]))
