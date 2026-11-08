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
  clairaut    'clairaut' with 'Boussinesq' (homogeneous: η = 0, f = 5q/4 at every r) gives the matrices A and B of
              'uniform' with f = 5q/4 and the same q, entry by entry, and so its eigenvalues (FIX 2026-11-08: with
              Clairaut's interpolant by a DCT, centrifugal.radau; the matrices differed by about 2e-12 of the row scale)
  monopole    thermal Boussinesq (g = r): with q > 0 and f = 0, g_eff = (1 - 2q/3) r, so the eigenvalues equal those of
              Beyonce (1 - 2q/3) without the monopole
  ideal       inviscid and adiabatic (ViscosD = ThermaD = 0), stratified, deformed with η ≠ 0: the system conserves the
              energy ∫ ρ0 w*⋅M w + p0 J |s|² dV, so the resolved eigenvalues are imaginary. This tests that the metric
              blocks keep that structure (a non-Hermitian error, such as a coupling read with source and target swapped);
              the value of J is tested by the entropy budget of damped modes (budget_check.py)
  zero field  (NEW, phase 1b) as zero with a field: a shell with a conducting inner core and the axial field, and the
              full sphere with a poloidal and a toroidal 'general' field
  relabelled  (NEW, phase 1b; FIX 2026-11-06, phase 2) the relabelling test with a field: the field of the body is
  field       F B̃0/J, so a sphere labelled as deformed inside (f = c (1 - r²)², c = ±1e-3) with kore's field B̃0 is the
              sphere with the field carried by the map. For the axial field (flux function s²/2) that is the poloidal
              h = (1 + ε) r/2; for the Malkus field s φ̂ it is s (1 - η P2) φ̂, the toroidal k1 = r (1 + η/5), k3 = -η r/5.
              The fluid is viscous (Ek = Em = 1e-3, no-slip walls), so that every metric term enters. Both runs must move
              the eigenvalues alike at first order: the parts of the shifts odd in c (the even part is kore's error of
              second order) agree to 1e-3 on the resolved modes (the eigenvalue equal at lmax 40 and 52 to 1e-9, and the
              shift of the mapped field to 1e-3), to 4e-6 (axial) and 3e-5 (Malkus). (At c = 1e-4 the eigensolver's noise,
              about 1e-11, is 1e-4 of the smallest shifts; at 3e-3 the error of third order in c of the odd part is
              2e-4 of them.) Leaving out the metric part of the magnetic diffusion puts them 3 % to 60 % off, that of the
              viscous force 4 % to 170 % (axial) and up to 160 times the shift (Malkus). (Phase 1b's version was inviscid,
              with the diffusion kept spherical, and agreed to 1-2 % and 3-4 %; leaving out δ𝐄, the factor 1/J of the
              force or ∇×((M - I)b)×B0, or flipping the sign of ∇×((M - I)B0)×b gave 5 % to 44 % and 14 % to 450 % there)
  relabelled  (NEW, phase 2) the free decay of a field (Le = 1e-6, so that the flow does not act back on it) in a sphere
  decay       labelled as deformed inside (c = ±1e-3): the decay rates must not move at first order, |λ(c) - λ(-c)|/2
              below 1e-9 |λ| (it is of order c³: 2e-13 to 4e-11 of |λ|). Three cases: a full sphere with an
              insulating wall; a conducting inner core of the fluid's conductivity, where the relabelling moves the inner
              boundary, which a uniform conductor does not see (this tests the metric diffusion of the core and the
              matching rows at the boundary); and one of conductivity 0.3, with f = c (r² - ricb²)² (1 - r²)², which
              leaves both boundaries in place

Usage: python3 eig_check.py   (two to three minutes on four cores; exit code 1 if a check fails)
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
dM = max(abs(load('cl_c', M).astype(complex) - load('cl_u', M).astype(complex)).max() for M in 'AB')
a, b = eigs('cl_c', 0.6j, 8), eigs('cl_u', 0.6j, 8)
dmax = max(abs(nearest(b, x) - x) for x in a)
verdict('clairaut', dM == 0 and dmax < 1e-12, "homogeneous 'clairaut' against 'uniform' f = 5q/4 (q = %g): max |A, B "
        "difference| = %.1e, max |Δλ| = %.1e" % (q, dM, dmax))

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

# ------------------------------------------------------------------------------------------------- zero field (phase 1b)
MZ = "par.magnetic = 1; par.Ek = 1e-3; par.Em = 1e-3; par.Le = 0.03; par.set_scales(); par.Gaspard = 1.; par.Beyonce = 1."
dz = []
for e in (MZ + "; par.ricb = 0.35; par.innercore = 'conducting, Chebys'",
          MZ + "; par.B0 = 'general'; par.B0_profiles = {1: 'r/6 - r**3/10', 3: '0.3*r**3*(1 - r**2)'}; "
               "par.B0_tor = 'general'; par.B0_tor_profiles = {2: 'r**2*(1 - r**2)'}"):
    build('zm0', e + '; par.centrifugal = 0')
    build('zm1', e)
    dz.append(max(abs(load('zm0', M).astype(complex) - load('zm1', M).astype(complex)).max() for M in 'AB'))
verdict('zero field', max(dz) == 0, 'max |A, B (f = 0) - A, B (centrifugal = 0)| = %.1e (conducting inner core, axial field), '
        '%.1e (full sphere, general poloidal and toroidal field)' % tuple(dz))

# ---------------------------------------------------------------------------------------------- relabelled field (phase 1b)
# FIX (2026-11-06, phase 2): a viscous fluid (every metric term enters), the shifts at c = ±1e-3 and their parts odd in c
# compared (the even part is kore's error of second order), on the modes whose shift is resolved; tolerance 1e-3 for both
# fields (was 3 % and 9 %, inviscid, with the magnetic diffusion kept spherical)
C, TOL = 1e-3, 1e-3
FR = ("par.magnetic = 1; par.Ek = 1e-3; par.Em = 1e-3; par.Le = 0.1; par.set_scales(); par.bci = 1; par.bco = 1; "
      "par.rpower_v = 4; par.N = 64; par.lmax = 40")
def cf_shape(c):
    return ("par.centrifugal_shape = 'user'; par.centrifugal_f = None; par.centrifugal_profile = 'p*(1 - r**2)**2'; "
            "par.centrifugal_param = %r" % c)
def eta(c):
    return '(8/3)*%r*r**2*(1 - r**2)' % c     # η = r ε' of ε = -(2/3) f, f = c (1 - r²)²
FIELDS = {   # name: (the field B̃0, the field F B̃0/J it gives the body at c, the target)
    'axial':  ("par.B0 = 'axial'",
               lambda c: "par.B0 = 'general'; par.B0_profiles = {1: '(1 - (2/3)*%r*(1 - r**2)**2)*r/2'}" % c, -0.02 + 1.0j),
    'Malkus': ("par.B0 = None; par.B0_tor = 'Malkus'",
               lambda c: "par.B0 = None; par.B0_tor = 'general'; par.B0_tor_profiles = {1: 'r*(1 + %s/5)', 3: '-(%s)*r/5'}"
                         % (eta(c), eta(c)), -0.02 + 0.97j)}
shift = lambda d, x: nearest(eigs(d, x + 1e-7j, 3), x) - x
for name, (fld, mapped, s) in FIELDS.items():
    build('fr_0', FR + '; ' + fld + '; par.centrifugal = 0')
    build('fr_hi', FR + '; ' + fld + '; par.centrifugal = 0; par.lmax = 52')
    for k, c in (('p', C), ('m', -C)):
        build('fr_cf' + k, FR + '; ' + fld + '; ' + cf_shape(c))
        build('fr_map' + k, FR + '; ' + mapped(c) + '; par.centrifugal = 0')
    build('fr_map_hi', FR + '; ' + mapped(C) + '; par.centrifugal = 0; par.lmax = 52')
    l0s, his = eigs('fr_0', s, 10), eigs('fr_hi', s, 16)
    rows = []
    for l0 in [x for x in l0s if abs(his - x).min() < 1e-9]:
        dmap = (shift('fr_mapp', l0) - shift('fr_mapm', l0))/2
        if abs(shift('fr_map_hi', nearest(his, l0)) - shift('fr_mapp', l0)) > 1e-3*abs(dmap):   # the shift is not resolved
            continue
        rows.append((l0, (shift('fr_cfp', l0) - shift('fr_cfm', l0))/2, dmap))
    rel = [abs(dcf - dmap)/abs(dmap) for l0, dcf, dmap in rows]
    verdict('field ' + name, len(rel) >= 4 and max(rel) < TOL, '%d resolved modes moved by %.1e to %.1e (c = ±%g): '
            '|Δλ(deformed) - Δλ(mapped field)|/|Δλ| of the parts odd in c <= %.1e (below %g)'
            % (len(rel), min([abs(r[2]) for r in rows] or [0.]), max([abs(r[2]) for r in rows] or [0.]), C,
               max(rel or [np.inf]), TOL))

# ------------------------------------------------------------------------------------------- relabelled decay (phase 2)
# NEW (2026-11-06, phase 2): the free decay of a field (Le = 1e-6, so the flow does not act back on it) in a sphere labelled
# as deformed inside: its decay rates must not move at first order in c
DEC = "par.magnetic = 1; par.B0 = 'axial'; par.Em = 1e-2; par.Le = 1e-6; par.N = 48; par.lmax = 24"
IC = "par.Ek = 1e-3; par.set_scales(); par.ricb = 0.35; par.innercore = 'conducting, Chebys'; par.mu_i2o = 1.; par.Nic = 24"
DECAY = {   # name: (the run, the relabelling f / c)
    'sphere': (DEC + "; par.Ek = 0; par.set_scales(); par.ViscosD = 0", 'p*(1 - r**2)**2'),
    'core': (DEC + '; ' + IC + '; par.sigma_i2o = 1.', 'p*(1 - r**2)**2'),
    'core 0.3': (DEC + '; ' + IC + '; par.sigma_i2o = 0.3', 'p*(r**2 - 0.35**2)**2*(1 - r**2)**2')}
CD, res = 1e-3, []
for name, (run, prof) in DECAY.items():
    build('dc_0', run + '; par.centrifugal = 0')
    for k, c in (('p', CD), ('m', -CD)):
        build('dc_' + k, run + "; par.centrifugal_shape = 'user'; par.centrifugal_f = None; par.centrifugal_profile = %r; "
              "par.centrifugal_param = %r" % (prof, c))
    l0s = []
    for s in (-0.1, -0.35, -0.6):
        l0s += [x for x in eigs('dc_0', s, 4) if abs(x.imag) < 1e-8 and min([abs(x - y) for y in l0s] + [1.]) > 1e-6]
    odd = [abs(shift('dc_p', l0) - shift('dc_m', l0))/2/abs(l0) for l0 in l0s]
    res.append((name, len(odd), max(odd or [np.inf])))
verdict('relabelled decay', all(n >= 4 and o < 1e-9 for _, n, o in res), 'the free decay of a field in a sphere labelled as deformed, '
        'c = ±%g: |λ(c) - λ(-c)|/(2|λ|) <= %s (O(c³))' % (CD, ', '.join('%.0e (%s, %d modes)' % (o, nm, n) for nm, n, o in res)))

print('eig_check.py: %s' % ('FAIL ' + ', '.join(fail) if fail else 'ok'))
sys.exit(1 if fail else 0)
