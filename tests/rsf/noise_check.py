#!/usr/bin/env python3
'''
noise_check.py [tol]: the random Reynolds-stress force of bin/reynolds_stress.py against kore's viscous operator
(fluctuation-dissipation), exactly (no sampling): every noise function (unit weight on one centre, for one c_k and
one l) is built and paired with random smooth test fields a that vanish near the walls (rsf_common.test_fields).
Run in a kore run directory after submatrices.py. The targets are built here from kore's profiles and the documented
laws (documented(), params_default.py), not from bin/reynolds_stress.py.

With the run's own rsf_* parameters:
  B0  the amplitude of the noise at the builder's nodes, tau A_lk S^-1/2 for every (l, k), against the documented
      formulas: A_lk = sqrt(chi_k (2l+1)/(4 pi) 4 Theta nu_t/rho0)/r [exp(-L l_c^2/(4 r^2))], Theta from rsf_theta
      ('nu', 'mlt' or a number) times rsf_scale, nu_t = ViscosD (nu/nu0 - floor), l_c, the erfc taper (and rsf_rcut)
      and S = sum_j h_j g_j.                                                                    pass: < 1e-12
      With a constant rsf_lc also S = 1 wherever tau > 1e-6 (the centres run past the walls).   pass: |S - 1| < 3e-3
  B1  strong = weak: for every noise function, <a, f> from kore's rows (energy pairing of bin/ru_insane.py) equals
      -int rho0 e(a)*:s dV computed from the stress itself (c_k = tau A_lk S^-1/2 sqrt(h_j) g_j at the nodes of a
      Gauss-Legendre quadrature), for smooth fields spread over the domain (rsf_common.test_fields). Checks the
      c_k -> f formulas, the derivatives of tau A S^-1/2 (background, eddy size, angular factor), the rows and their
      transform. Smooth fields do not see the Chebyshev truncation of single noise functions, which N 160 in a full
      sphere does not resolve at l_c 0.05 (the 'tail' printed below).                       pass: < tol (1e-6)
  B5  no forcing without a dissipative partner: for fields localised at several radii (rsf_common.radii, also where
      nu_t vanishes), E|<a, f>|^2 from kore's rows against 4 int rho0 Theta nu_t tau^2 [exp(-L l_c^2/(2 r^2))]
      |d(a)|^2 dV, which bounds it (Cauchy-Schwarz). Noise reaching below a convection zone would exceed it by orders
      of magnitude.                                                                     pass: every ratio < 1 + 1e-4
With rsf_theta = 1, rsf_scale = 2.5, rsf_angular = 0, rsf_nu_floor = 0 (so Theta nu_t = 2.5 nu) and the run's rsf_lc
(0.05 if None):
  B2  covariance in the white limit, per channel (c1, c2, c5 of section u; c3, c6 of section v) and per l: the sum
      over noise functions of |<a, f>|^2 (weak form, fine quadrature, l_c = 0.003) against the same channel and l of
      4 int rho0 2.5 nu tau^2 |d(a)|^2 dV (rsf_common.dd_channels), for every (channel, l) above 1e-6 of its channel.
      Checks chi_k, the amplitudes and rsf_scale, and the discretisation of the noise.  pass: every |ratio - 1| < 1e-2
  B3  kore's viscous force against the traceless strain: -2 Re<a, V a> = 4 int rho0 nu |d(a)|^2 dV, so the noise is
      the partner of kore's own damping.                                                    pass: < tol
  B4  all together at that l_c, from kore's rows: E|<a, f>|^2 / (-2 Theta Re<a, V a>), 1 - O((l_c/scale of a)^2):
      printed, not judged.
'''
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'bin'))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.special as ssp
from parameters import par
import utils as ut
import operators as op
import radial_profiles as rap
import reynolds_stress as rs
import rsf_common as rc

tol = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-6
SEC = (('u', rc.lp, (1, 2, 5)), ('v', rc.lt, (3, 6)))

def CHI(k, L):
    if k in (5, 6):
        return 2/(L*(L - 2)) if L > 2 else 0.
    return {1: 2/3, 2: 1/(2*L), 3: 1/(2*L)}[k]

def documented(r):
    '''rho0, nu_t, l_c, Theta (times rsf_scale) and the taper at r, from kore's profiles and the documented laws'''
    rho0 = np.exp(rap.logrhoX(r, 0))
    if par.rsf_nu_floor is not None:
        fl = par.rsf_nu_floor
    elif par.model_type == 'smoothed sun' and par.visc0 < 1:
        fl = par.visc0
    else:
        fl = 0.
    rf = np.linspace(par.ricb, ut.rcmb, 20001)[1:]
    numax = max((par.ViscosD*(rap.viscoX(rf, 0) - fl)).max(), 0.)
    nut = par.ViscosD*(rap.viscoX(r, 0) - fl)
    nut = np.where(nut > 1e-14*numax, nut, 0.)
    if par.rsf_lc is not None:
        lc = par.rsf_lc*np.ones_like(r)
    else:
        Hp = -rap.pressX(r, 0)/rap.pressX(r, 1)
        lc = (par.rsf_lc_max**-4 + (par.rsf_alpha*Hp)**-4)**-0.25
    th = par.rsf_theta
    if isinstance(th, str) and th == 'nu':
        theta = 3*rho0*nut**2*lc
    elif isinstance(th, str) and th == 'mlt':
        Fc = par.rsf_lum/(4*np.pi*r**2)*nut/numax
        theta = rho0*(par.rsf_alpha*Fc/(10*rho0))**(2/3)*lc**3/3
    else:
        theta = float(th)*np.ones_like(r)
    theta = par.rsf_scale*np.where(nut > 0, theta, 0.)
    T, Ti = par.rsf_taper, (par.rsf_taper if par.ricb > 0 else par.rsf_taper_in)
    tau = 0.25*ssp.erfc((r - ut.rcmb + T/2)/(T/8))*ssp.erfc((par.ricb + Ti/2 - r)/(Ti/8))
    if par.rsf_rcut is not None:   # 1e-8 at rsf_rcut, 1 above rsf_rcut + rsf_rcut_width
        Wc = par.rsf_rcut_width
        tau = tau*0.5*ssp.erfc((par.rsf_rcut + Wc/2 - r)/(Wc/8))
    return {'rho0': rho0, 'nut': nut, 'lc': lc, 'theta': theta, 'tau': tau}

def gauss(r, c):
    d = r[:, None] - c['r'][None, :]
    s = c['sig'][None, :]
    return np.exp(-d**2/(2*s**2))/(np.sqrt(2*np.pi)*s)

def amplitude_check(bd):
    '''B0: the worst relative difference of tau A S^-1/2 at the builder's nodes from the documented formulas'''
    dc = documented(bd.r)
    S = gauss(bd.r, bd.c) @ bd.c['h']
    worst = 0.
    for sec, ls, ks in SEC:
        for l in ls:
            L = l*(l + 1.)
            for k in ks:
                if CHI(k, L) == 0:
                    continue
                A = np.sqrt(CHI(k, L)*(2*l + 1)/(4*np.pi)*4*dc['theta']*dc['nut']/dc['rho0'])/bd.r
                if par.rsf_angular:
                    A *= np.exp(-L*dc['lc']**2/(4*bd.r**2))
                E = dc['tau']*A/np.sqrt(S)
                worst = max(worst, np.abs(bd.envelope(l, k)[0] - E).max()/np.abs(E).max())
    return worst

def fields_at(q, sec, l, F):
    '''values (and derivatives) of P or T of the field set F at the nodes of q, and its strain'''
    if sec == 'u':
        Fv = q.field(F[0][l], 'P')
        return Fv, rc.strain(q, l, Fv, None)
    Fv = q.field(F[1][l], 'T')[:2]
    return Fv, rc.strain(q, l, None, Fv)

def noise_pairings(bd, q, sets, strong):
    '''
    for every field set (P, T) in sets: the sum over noise functions of |<a, f>|^2, weak form at the nodes of q, per
    (sec, k, l), and (strong = True) from kore's rows, in total; with the worst strong-weak difference for sets[0],
    relative to the largest weak pairing of each (l, k)
    '''
    bgq = rs.background(q.r)
    G = gauss(q.r, bd.c)
    tq = rs.taper(q.r)[0]*rs.normalisation([G @ bd.c['h'], 0., 0.])[0]
    Ew = [dict() for _ in sets]
    Es = [0. for _ in sets]
    err = 0.
    for sec, ls, ks in SEC:
        for l in ls:
            fl = [fields_at(q, sec, l, F) for F in sets]
            for k in ks:
                if rs.chi(k, l*(l + 1.)) == 0:
                    continue
                a = bd.amplitudes(l, k)
                Cq = (tq*rs.envelope(bgq, q.r, l, k)[0])[:, None]*G*a[None, :]
                if strong:
                    W = np.diag(a)
                    if sec == 'u':
                        rows = bd.rows_u(l, *bd.forces_u(l, *[W if kk == k else None for kk in (1, 2, 5)]))
                    else:
                        rows = bd.rows_v(l, bd.forces_v(l, *[W if kk == k else None for kk in (3, 6)]))
                for i, (Fv, e) in enumerate(fl):
                    gw = rc.weak(q, l, e, {k: Cq})
                    Ew[i][(sec, k, l)] = np.sum(np.abs(gw)**2)
                    if strong:
                        gs = q.pair(sec, l, Fv[0], rows)
                        Es[i] += np.sum(np.abs(gs)**2)
                        if i == 0 and np.abs(gw).max() > 0:
                            err = max(err, np.abs(gs - gw).max()/np.abs(gw).max())
    return Ew, Es, err

def dissipation(q, F, weight):
    '''4 int rho0 weight(l) |d(a)|^2 dV per (sec, k, l), weight(l) at the nodes of q (Theta nu_t, ...)'''
    out = {}
    for sec, ls, ks in SEC:
        for l in ls:
            dch = rc.dd_channels(l, fields_at(q, sec, l, F)[1])
            for k in ks:
                out[(sec, k, l)] = 4*np.sum(q.w*q.rho0*weight(l)*q.r**2*4*np.pi/(2*l + 1)*dch[k])
    return out

rng = np.random.default_rng(2)
q = rc.Quad(max(4*par.N, 600))
r0s, wd = rc.radii()
loc = [rc.test_fields(rng, r0=r0, w=wd) for r0 in r0s]
smooth = rc.test_fields(rng)

# ---- with the run's parameters: B0 (amplitude), B1 (strong = weak) and B5 (the Cauchy-Schwarz bound, field by field)
bd = rs.Builder()
b0 = amplitude_check(bd)
_, Es, b1 = noise_pairings(bd, q, [smooth] + loc, True)
dq = documented(q.r)
ang = lambda l: np.exp(-l*(l + 1.)*dq['lc']**2/(2*q.r**2)) if par.rsf_angular else 1.
target = [sum(dissipation(q, F, lambda l: dq['theta']*dq['nut']*dq['tau']**2*ang(l)).values()) for F in loc]
S = gauss(q.r, bd.c) @ bd.c['h']
s1 = np.abs(S[dq['tau'] > 1e-6] - 1).max() if par.rsf_lc is not None else 0.
b5 = [Es[i + 1]/target[i] if target[i] > 0 else (0. if Es[i + 1] == 0 else np.inf) for i in range(len(loc))]
ok5 = all(x < 1 + 1e-4 for x in b5)
print('noise_check (N %d, lmax %d, m %d, symm %d, ricb %g; theta %s, scale %g, l_c %s, angular %d, nu floor %g; %d noise centres):'
      % (par.N, par.lmax, par.m, par.symm, par.ricb, par.rsf_theta, par.rsf_scale,
         par.rsf_lc if par.rsf_lc else 'alpha H_p', par.rsf_angular, rs.nu_floor(), bd.c['r'].size))
print('  B0 amplitude tau A S^-1/2 vs the documented laws, worst rel. diff  %.1e%s' % (b0,
      ';  max |S - 1| where tau > 1e-6: %.1e' % s1 if par.rsf_lc is not None else ''))
print('  B1 rows vs stress (weak form), worst rel. diff  %.1e' % b1)
print('  B5 E|<a,f>|^2 / 4 int rho0 Theta nu_t tau^2 |d|^2 for fields at r = %s:' % ', '.join('%.3g' % x for x in r0s))
print('     %s   (targets %s; sum_j h_j g_j = %.4f .. %.4f where tau > 0.5)' % (' '.join('%.4f' % x for x in b5),
      ' '.join('%.1e' % t for t in target), S[dq['tau'] > 0.5].min(), S[dq['tau'] > 0.5].max()))
tail1 = bd.tail

# ---- Theta = 1, scale 2.5, nu_t = nu, white in angle: B3, B4 at the run's l_c, B2 in the white limit
SCALE = 2.5
par.rsf_theta, par.rsf_angular, par.rsf_nu_floor, par.rsf_scale = 1.0, 0, 0., SCALE
if par.rsf_lc is None:
    par.rsf_lc = 0.05
P, T = rc.test_fields(rng)
Va = sum(q.pair('u', l, q.field(P[l], 'P')[0], op.viscous_diffusion(l, 'u', 'upol', 0) @ P[l])[0] for l in rc.lp) \
   + sum(q.pair('v', l, q.field(T[l], 'T')[0], op.viscous_diffusion(l, 'v', 'utor', 0) @ T[l])[0] for l in rc.lt)
Dnu = sum(dissipation(q, (P, T), lambda l: q.nu).values())
b3 = abs(-2*Va.real/Dnu - 1)
bd = rs.Builder()
Ew, Es, _ = noise_pairings(bd, q, [(P, T)], True)
b4 = Es[0]/(-2*SCALE*Va.real)

par.rsf_lc = 0.003
bd2 = rs.Builder()
q2 = rc.Quad(6000)
t2 = documented(q2.r)['tau']
E2 = noise_pairings(bd2, q2, [(P, T)], False)[0][0]
D2 = dissipation(q2, (P, T), lambda l: SCALE*1.0*q2.nu*t2**2)
chan = sorted(set((s_, k) for s_, k, l in D2))
b2 = {c: sum(E2[x] for x in E2 if x[:2] == c)/sum(D2[x] for x in D2 if x[:2] == c) for c in chan}
b2l = {}   # the worst |ratio - 1| per channel over its l with a non-negligible share
for c in chan:
    tot = sum(D2[x] for x in D2 if x[:2] == c)
    b2l[c] = max(abs(E2[x]/D2[x] - 1) for x in D2 if x[:2] == c and D2[x] > 1e-6*tot)
ok2 = all(abs(x - 1) < 1e-2 for x in b2.values()) and all(x < 1e-2 for x in b2l.values())

ok = b0 < 1e-12 and s1 < 3e-3 and b1 < tol and ok5 and ok2 and b3 < tol
print('  B2 sum|<a,f>|^2 / 4 int rho0 Theta nu_t tau^2 |d|^2 at l_c 0.003, Theta nu_t = %g nu (%d centres), per channel'
      ' (worst l): %s' % (SCALE, bd2.c['r'].size, '  '.join('%s%d %.6f (%.4f)' % (c[0], c[1], b2[c], b2l[c]) for c in chan)))
print('  B3 -2 Re<a,Va> / 4 int rho0 nu |d|^2 - 1:  %.1e' % b3)
print('  B4 E|<a,f>|^2 / (-2 Theta Re<a,Va>) at l_c %g (rows):  %.6f   (weak form %.6f)'
      % (bd.c['lc'][0], b4, sum(Ew[0].values())/(-2*SCALE*Va.real)))
print('  Chebyshev tail of the rows %.1e (run parameters), %.1e (Theta = 1)   %s' % (tail1, bd.tail, 'ok' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
