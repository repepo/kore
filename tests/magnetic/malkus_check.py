#!/usr/bin/env python3
'''
malkus_check.py [Ek] [N] [lmax]: the Malkus field B0 = Le s φ̂ (B0 = None, B0_tor = 'Malkus', cnorm_tor = 1) against the ideal
theory of Malkus (1967). With this field the induction is i m Le u and the Lorentz force (i m Le²/λ)(i m u + 2ẑ×u) up to a
gradient, so the ideal problem reduces exactly to that of the inertial waves, in any container with impenetrable walls:
    ω = (λ/2) [1 ± sqrt(1 + 4 Le² m (m - λ)/λ²)],  λ the inertial-mode frequency (kore: Im of the eigenvalue),
and for small Le the frequency of the fast branch shifts by Le² m (m - λ)/λ (the spin-over, λ = m = 1, does not move).
For the m = 1, symm -1 modes near 0.75i (uniform density, insulating walls, Ek = Em) it builds the hydrodynamic run and three
Malkus runs (Le² = 1e-6, 4e-6, 1.6e-5) and checks that the shift is quadratic in Le (the same Δω/Le² for the three, to 1e-3)
and within tol (10 %) of m(m - λ)/λ for the least damped mode. Defaults Ek 1e-5, N 96, lmax 64 (2 to 8 % off, from the
viscous and Ohmic layers; Ek = 1e-6, N 192, lmax 128 gives 0.4 to 1 %).
'''
import os, sys, subprocess, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
Ek = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-5
N = int(sys.argv[2]) if len(sys.argv) > 2 else 96
lmax = int(sys.argv[3]) if len(sys.argv) > 3 else 64
LE2 = [1e-6, 4e-6, 1.6e-5]
base = ("par.structure = 'uniform'; par.Ek = %r; par.Em = %r; par.set_scales(); par.N = %d; par.lmax = %d; par.ncpus = 4; "
        "par.nev = 10; par.itau = 0.75; par.B0 = None; par.B0_tor = 'Malkus'; par.cnorm_tor = 1" % (Ek, Ek, N, lmax))
def lam(name, extra):
    subprocess.run([sys.executable, H + '/mk_run.py', name, base + '; ' + extra, 'solve'], check=True, stdout=subprocess.DEVNULL)
    e = np.loadtxt(name + '/eigenvalues0.dat').reshape(-1, 2)
    return e[:, 0] + 1j*e[:, 1]
hyd = lam('malkus_hyd', 'par.magnetic = 0')
res = [lam('malkus_%g' % le2, 'par.Le = np.sqrt(%r); par.set_scales(); par.N = %d; par.lmax = %d' % (le2, N, lmax)) for le2 in LE2]
ok = True
for i, z0 in enumerate(sorted(hyd, key=lambda z: -z.real)[:4]):
    li = z0.imag; ideal = (1 - li)/li     # m(m - λ)/λ with m = 1
    sh = [ (r_[np.argmin(abs(r_ - z0))] - z0).imag/le2 for r_, le2 in zip(res, LE2) ]
    quad = (max(sh) - min(sh))/abs(sh[0])
    err = abs(sh[0] - ideal)/abs(ideal)
    print('lambda %+.6f %+.6fi: d(Im)/Le^2 = %s (quadratic to %.0e), ideal m(m-lambda)/lambda = %+.4f, off by %.1f %%'
          % (z0.real, z0.imag, ' '.join('%+.4f' % s for s in sh), quad, ideal, 100*err))
    ok &= quad < 1e-3 and (i > 0 or err < 0.1)
print('malkus_check.py:', 'ok' if ok else 'FAIL')
sys.exit(0 if ok else 1)
