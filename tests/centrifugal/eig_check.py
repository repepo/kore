#!/usr/bin/env python3
'''
eig_check.py: what the centrifugal terms do to the eigenvalues, against exact results (NEW 2026-11-05).

Builds small runs with mk_run.py in the current directory (A.npz, B.npz; no SLEPc) and finds the eigenvalues of
A x = λ B x near a target by shift-invert (scipy splu and eigs). The checks, each printed as one verdict line:

  zero        centrifugal = 1 with f = 0 and q = 0 gives the matrices A and B of centrifugal = 0, entry by entry
  spin-over   the inviscid Boussinesq full sphere with a uniform f (a Maclaurin spheroid, m 1): the spin-over mode has
              λ = 2i a²/(a² + c²) = i (1 + f + f²/2 + ...) exactly (Greenspan; targets.som of Zhang 2004 without viscosity);
              at first order (Im λ - 1)/f -> 1 and Re λ = 0 (the second order of the code is f²/3, not f²/2)
  relabelling f = c (1 - r²)², which vanishes with its derivative at the wall: the body is still the sphere, only its
              points are labelled differently inside, so the eigenvalues must move at O(c²) only. On the well-resolved
              modes of the sphere (equal at two resolutions), the odd part (λ(c) - λ(-c))/2 is O(c³) and λ(c) - λ(0)
              is O(c²): this tests every metric term (inertia, Coriolis, the factors ε and η, all Δl) at once
  clairaut    'clairaut' with 'Boussinesq' (homogeneous: η = 0, f = 5q/4 at every r) gives the eigenvalues of 'uniform'
              with f = 5q/4 and the same q
  monopole    thermal Boussinesq (g = r): with q > 0 and f = 0, g_eff = (1 - 2q/3) r, so the eigenvalues equal those of
              Beyonce (1 - 2q/3) without the monopole
  ideal       inviscid and adiabatic (ViscosD = ThermaD = 0), stratified, deformed with η ≠ 0: the system conserves the
              energy ∫ ρ0 w*⋅M w + p0 J |s|² dV, so the resolved eigenvalues are imaginary. This tests that the metric
              blocks keep that structure (a non-Hermitian error, such as a coupling read with source and target swapped);
              the value of J is tested by the entropy budget of damped modes (budget_check.py)

Usage: python3 eig_check.py   (about a minute; exit code 1 if a check fails)
'''
import os, sys, subprocess
import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl

HERE = os.path.dirname(os.path.abspath(__file__))
fail = []

def build(name, extra):
    p = subprocess.run([sys.executable, os.path.join(HERE, 'mk_run.py'), name, extra], capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-3000:]); sys.exit('eig_check.py: %s could not be built' % name)
    return name

def load(d, M):
    z = np.load(os.path.join(d, M + '.npz'))
    return ss.csr_matrix((z['data'], z['indices'], z['indptr']), shape=z['shape'])

def eigs(d, sigma, k=6):
    '''the k eigenvalues of the run d nearest sigma, nearest first'''
    A, B = load(d, 'A').astype(complex), load(d, 'B').astype(complex)
    lu = ssl.splu((A - sigma*B).tocsc())
    op = ssl.LinearOperator(A.shape, matvec=lambda x: lu.solve(B @ x), dtype=complex)
    nu = ssl.eigs(op, k=k, which='LM', tol=1e-14, ncv=max(2*k + 1, 40), return_eigenvectors=False)
    lam = sigma + 1/nu
    return lam[np.argsort(abs(lam - sigma))]

def verdict(name, ok, msg):
    print('%-12s %s  %s' % (name + ':', 'ok  ' if ok else 'FAIL', msg))
    if not ok:
        fail.append(name)

def nearest(l, x):
    return l[np.argmin(abs(l - x))]

# ------------------------------------------------------------------------------------------------------------------ zero
build('z0', 'par.centrifugal = 0')
build('z1', '')
dmax = max(abs(load('z0', M).astype(complex) - load('z1', M).astype(complex)).max() for M in 'AB')
verdict('zero', dmax == 0, 'max |A, B (f = 0) - A, B (centrifugal = 0)| = %.1e' % dmax)

# ------------------------------------------------------------------------------------------------------------- spin-over
rows = []
for f in (1e-3, -1e-3, 1e-2, -1e-2):
    build('so', 'par.centrifugal_f = %r' % f)
    l = nearest(eigs('so', 1j*(1 + f), 2), 1j*(1 + f))
    rows.append((f, l))
ok = all(abs((l.imag - 1)/f - 1) < 0.5*abs(f) and abs(l.real) < 1e-12 for f, l in rows)
exact = lambda f: 2/(1 + (1 - f)**2)
verdict('spin-over', ok, '(Im λ - 1)/f - 1 = ' + ', '.join('%+.2e (f %+g)' % ((l.imag - 1)/f - 1, f) for f, l in rows)
        + '; Im λ - exact: ' + ', '.join('%+.1e' % (l.imag - exact(f)) for f, l in rows[2:]))

# ----------------------------------------------------------------------------------------------------------- relabelling
HI = 'par.N = 48; par.lmax = 40'
build('rl_lo', 'par.N = 36; par.lmax = 30')
build('rl_0', HI)
for c in (1e-2, -1e-2, 1e-3, -1e-3):
    build('rl_%+g' % c, HI + "; par.centrifugal_shape = 'user'; par.centrifugal_f = None; par.centrifugal_profile = 'p*(1 - r**2)**2'; "
          "par.centrifugal_param = %r" % c)
worst_odd, ratios, nmodes = 0., [], 0
for s in (1.0000001j, 0.3j, 0.7j, -0.5j, 1.5j):
    hi, lo = eigs('rl_0', s, 10), eigs('rl_lo', s, 10)
    for l0 in [h for h in hi if abs(lo - h).min() < 1e-11][:3]:
        nmodes += 1
        d = {c: nearest(eigs('rl_%+g' % c, l0 + 1e-7j, 3), l0) for c in (1e-2, -1e-2, 1e-3, -1e-3)}
        worst_odd = max(worst_odd, abs(d[1e-2] - d[-1e-2])/2/1e-2**2, abs(d[1e-3] - d[-1e-3])/2/1e-3**2)
        ratios.append(abs(d[1e-2] - l0)/abs(d[1e-3] - l0))
ok = nmodes >= 8 and worst_odd < 1e-2 and all(80 < r < 125 for r in ratios)
verdict('relabelling', ok, '%d resolved modes: |λ(c) - λ(-c)|/(2c²) <= %.1e (c = 1e-2, 1e-3), |λ(c) - λ0| ratio (1e-2 to 1e-3) '
        'in [%.1f, %.1f] (100 for O(c²))' % (nmodes, worst_odd, min(ratios), max(ratios)))

# -------------------------------------------------------------------------------------------------------------- clairaut
q = 0.04
build('cl_c', "par.centrifugal_shape = 'clairaut'; par.centrifugal_f = None; par.centrifugal_q = %r" % q)
build('cl_u', "par.centrifugal_f = %r; par.centrifugal_q = %r" % (1.25*q, q))
a, b = eigs('cl_c', 0.6j, 8), eigs('cl_u', 0.6j, 8)
dmax = max(abs(nearest(b, x) - x) for x in a)
verdict('clairaut', dmax < 1e-12, "homogeneous 'clairaut' against 'uniform' f = 5q/4 (q = %g): max |Δλ| = %.1e" % (q, dmax))

# -------------------------------------------------------------------------------------------------------------- monopole
TH = "par.thermal = 1; par.stratification = 'linear'; par.symm = 1; par.lmax = 30; par.N = 36"
q = 0.1
build('mo_q', TH + '; par.centrifugal_q = %r' % q)
build('mo_b', TH + '; par.centrifugal = 0; par.Beyonce = %r' % (1 - 2*q/3))
dmax = 0.
for s in (0.2j, 0.6j, 1.2j):
    a, b = eigs('mo_b', s, 6), eigs('mo_q', s, 6)
    dmax = max(dmax, max(abs(nearest(b, x) - x) for x in a))
verdict('monopole', dmax < 1e-12, 'q = %g, f = 0 against Beyonce (1 - 2q/3): max |Δλ| = %.1e' % (q, dmax))

# ----------------------------------------------------------------------------------------------------------------- ideal
# (a weak stratification, Beyonce 0.01: the strong one of 'monopole' crowds the spectrum with gravity modes, whose truncation
# makes conjugate pairs λ = ±a + ib in the sphere as well, the spurious centre modes of ThermaD = 0)
U = TH + "; par.Beyonce = 0.01; par.centrifugal_shape = 'user'; par.centrifugal_f = None; " \
         "par.centrifugal_profile = 'p*(1 + r**2 - 0.5*r**4)'; par.centrifugal_param = 0.01"
build('id_lo', U)
build('id_hi', U.replace('par.lmax = 30; par.N = 36', 'par.lmax = 40; par.N = 48'))
re_res, re_all, nmodes = 0., 0., 0
for s in (0.3j, 0.6j, 0.9j, 1.2j, -0.6j, -0.9j, -1.2j, -1.5j):
    hi, lo = eigs('id_hi', s, 10), eigs('id_lo', s, 10)
    good = [h for h in hi if abs(lo - h).min() < 1e-8]
    nmodes += len(good)
    re_res = max([re_res] + [abs(h.real) for h in good])
    re_all = max(re_all, abs(hi.real).max())
verdict('ideal', nmodes >= 8 and re_res < 1e-12, 'the deformed, stratified ideal fluid: %d resolved modes, max |Re λ| = %.1e '
        '(%.1e over all 80 eigenvalues found)' % (nmodes, re_res, re_all))

print('eig_check.py: %s' % ('FAIL ' + ', '.join(fail) if fail else 'ok'))
sys.exit(1 if fail else 0)
