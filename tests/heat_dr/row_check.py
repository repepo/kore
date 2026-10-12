#!/usr/bin/env python3
'''
row_check.py: independent check of the section-h rows of the assembled A and B of a ThermaD = 0 run (run in its directory).

For random low-degree P_l, T_l, s_l (Chebyshev, right parity when ricb = 0) it builds x, computes A x and B x, and
reads every section-h row block as a series in the C^(gb) Gegenbauer basis (gb and the first kept row index tried in
all combinations), evaluated on an r grid. It compares with the physics evaluated directly in (r, theta):
    B row l' :  c r^w s_l'(r) / L'^3
    A row l' :  c r^w [ -u_r S0' - i m G dOmega s - u . grad S_tw ]_l' / L'^3          (ThermaD = 0, rows (eq)/p0)
with w = 2 (claimed for the new mode) or 1 (old mode), c fitted once from B (the 1/Anorm of assemble.py).
S_tw is integrated here in cos(theta) from the thermal-wind balance, eq. (2) of docs/heat-equation.md, with
F = 2 G^2 dOmega (thermal_wind 1) or G^2 [(1 + dOmega)^2 - 1] (thermal_wind 2), its l = 0 part removed (thermal_wind 3:
dS_tw/dtheta = dSdtheta sin 2 theta); it does not use
radial_profiles.sigma2/4/6, tw*, tp*, or any kore operator. Only the raw profile functions of radial_profiles.prf
(g, ln rho, dS/dr, f0, f2) and the matrix layout (ut.ell, ut.N1) are taken from kore.
Unlike heat_check.py, asm_check.py and ru_insane.py, which pair the rows with p0 s* (p0 ~ 1e-6 at the wall for the
test profiles), the comparison is unweighted (max over r of |row - reference| / max|reference|), so it sees the
near-surface part of the rows. P is taken with P = 0 at the walls (as every kore velocity bc imposes) and, as a
diagnostic, also free at the walls.
usage: python3 row_check.py [tol] [ntrials]     (ThermaD = 0 runs only; tol defaults to 1e-8)
'''
import sys, os, itertools
sys.path.insert(0, os.path.abspath('bin'))
import numpy as np, scipy.sparse as ss, scipy.special as sp
import numpy.polynomial.chebyshev as C
import numpy.polynomial.polynomial as Pw
import numpy.polynomial.legendre as Lg
from parameters import par
import utils as ut
import radial_profiles as rap

tol = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-8
ntr = int(sys.argv[2]) if len(sys.argv) > 2 else 2
rng = np.random.default_rng(12345)   # the same fields for both wall variants: reseeded in run()
N, N1, ricb, m, G, Bey = par.N, ut.N1, par.ricb, par.m, par.Gaspard, par.Beyonce
lp, lt, ll = [list(a) for a in ut.ell(m, par.lmax, par.symm)]
nb = len(lp)
assert par.thermal == 1 and par.ThermaD == 0, 'row_check.py: ThermaD = 0 runs only'
w_claim = 2 if ut.adiab_tw else 1
tw = par.thermal_wind if par.diff_rot else 0

def load(f):
    z = np.load(f)
    return ss.csr_matrix((z['data'], z['indices'], z['indptr']), shape=z['shape'])
A, B = load('A.npz'), load('B.npz')

# ---------------------------------------------------------------- grids
if ricb > 0:
    r = 0.5*(1 - ricb)*(np.cos(np.pi*(np.arange(300) + 0.5)/300)[::-1] + 1) + ricb
    xr = 2*(r - ricb)/(1 - ricb) - 1; dxdr = 2/(1 - ricb)
else:
    r = np.cos(np.pi*(np.arange(300) + 0.5)/300)[::-1]; r = r[r > 0.02]
    xr = r; dxdr = 1.
nr = len(r)
xq, wq = Lg.leggauss(par.lmax + 30)
th = np.arccos(xq); sth = np.sin(th)

def Yl(l):
    y, dy = sp.sph_harm_y(l, m, th, 0, diff_n=1)
    f = np.sqrt(4*np.pi/(2*l + 1))
    return y*f, dy[:, 0]*f
Y = {l: Yl(l) for l in ll}

# raw profiles (even extension at ricb = 0 as in kore), derivatives from high-degree Chebyshev fits on the domain
a = ricb if ricb > 0 else -1.
MF = int(os.environ.get('FC_M', 400)); xf = np.cos(np.pi*(np.arange(MF) + 0.5)/MF); rf = 0.5*(1 - a)*(xf + 1) + a
def fitd(func, d):
    c = C.chebfit(xf, func(rf), MF//2)
    for _ in range(d): c = C.chebder(c)*(2/(1 - a))
    return C.chebval(2*(r - a)/(1 - a) - 1, c)
ab = lambda rr: np.abs(rr) if ricb == 0 else rr
f0f = lambda rr: rap.prf.abu(ab(rr), 0)*np.ones_like(rr)
f2f = lambda rr: rap.prf.aub(ab(rr), 0)*np.ones_like(rr)
f0, f01 = fitd(f0f, 0), fitd(f0f, 1)
f2, f21 = fitd(f2f, 0), fitd(f2f, 1)
g = rap.prf.gravity(r, 0)
hp = (-4*par.aux0**2*r/(1 - (par.aux0*r)**2)) if par.model_type == 'user def' else fitd(lambda rr: rap.prf.logrho(rr, 0), 1)   # (ln rho)' of ln((1 - (aux0 r)^2)^2), analytic for 'user def' (prf.dlog_rho lacks the factor aux0)
dS = np.sign(r)*rap.prf.gradS(np.abs(r), par.aux1, par.aux2, par.aux3, par.aux4, par.aux5)

# ---------------------------------------------------------------- S_tw(r, x) by integration in x = cos(theta)
def stw_sigma(rr):
    '''Legendre coefficients sigma_n(rr) (n = 0..6, sigma_0 = 0) of S_tw from eq. (2), at the radii rr'''
    F0, F01 = (fitd_at(f0f, 0, rr), fitd_at(f0f, 1, rr))
    F2, F21 = (fitd_at(f2f, 0, rr), fitd_at(f2f, 1, rr))
    gg = rap.prf.gravity(rr, 0)
    out = np.zeros((len(rr), 8))
    for i in range(len(rr)):
        dO  = Pw.Polynomial([F0[i] - 0.5*F2[i], 0, 1.5*F2[i]])           # dOmega(x)
        dOr = Pw.Polynomial([F01[i] - 0.5*F21[i], 0, 1.5*F21[i]])
        if tw == 1:
            F, Fr = 2*dO, 2*dOr
        else:
            F, Fr = 2*dO + dO*dO, 2*dOr + 2*dO*dOr
        F, Fr = G**2*F, G**2*Fr
        X = Pw.Polynomial([0, 1]); omx2 = Pw.Polynomial([1, 0, -1])
        dSx = -(rr[i]**2/(Bey*gg[i]))*(X*Fr + omx2*F.deriv()/rr[i])  # dS_tw/dx = -(1/sin) dS_tw/dtheta
        if tw == 3:   # prescribed: dS_tw/dtheta = dSdtheta sin 2 theta
            dSx = -2*par.dSdtheta*X
        S = dSx.integ()
        lc = Lg.poly2leg(S.coef)
        lc = np.concatenate([lc, np.zeros(8 - len(lc))])
        lc[0] = 0.
        out[i] = lc
    return out
def fitd_at(func, d, rr):
    c = C.chebfit(xf, func(rf), MF//2)
    for _ in range(d): c = C.chebder(c)*(2/(1 - a))
    return C.chebval(2*(rr - a)/(1 - a) - 1, c)

if tw:
    sig_f = stw_sigma(rf)                                # on the fit grid, for the r derivative
    sig = np.zeros((nr, 8)); sig1 = np.zeros((nr, 8))
    for n in range(8):
        c = C.chebfit(xf, sig_f[:, n], MF//2)
        sig[:, n] = C.chebval(2*(r - a)/(1 - a) - 1, c)
        sig1[:, n] = C.chebval(2*(r - a)/(1 - a) - 1, C.chebder(c)*(2/(1 - a)))
    Sx  = np.zeros((nr, len(xq))); Sr = np.zeros((nr, len(xq)))
    for n in range(8):
        Pn = Lg.Legendre.basis(n)
        Sx += np.outer(sig[:, n], Pn.deriv()(xq)); Sr += np.outer(sig1[:, n], Pn(xq))
    Sth = -sth[None, :]*Sx                               # dS_tw/dtheta
    print('S_tw Legendre content max|sigma_n|, n = 1..7:', ' '.join('%.2e' % np.max(np.abs(sig[:, n])) for n in range(1, 8)))

# ---------------------------------------------------------------- layout
s_ = int((par.symm + 1)/2)
iP = (m + 1 - s_) % 2; iT = (m + s_) % 2
def colblock(kind, l):
    if kind == 'P': return lp.index(l)*N1
    if kind == 'T': return nb*N1 + lt.index(l)*N1
    return 2*nb*N1 + lp.index(l)*N1
def rand_cheb(start):
    c = np.zeros(N, dtype=complex)
    K = 14
    c[:K] = (rng.standard_normal(K) + 1j*rng.standard_normal(K))*np.exp(-0.2*np.arange(K))
    if ricb == 0:
        c[1 - start::2] = 0
    return c
def put(x, kind, l, c, start):
    b = colblock(kind, l)
    x[b:b + N1] = c if ricb > 0 else c[start::2]

def evalrows(v, gb, rstart):
    '''values on r of a section-h row block given as C^(gb) coefficients'''
    cf = np.zeros(N, dtype=complex)
    if ricb > 0: cf[:] = v
    else: cf[rstart::2] = v
    k = np.arange(N)
    E = sp.eval_chebyt(k[None, :], xr[:, None]) if gb == 0 else sp.eval_gegenbauer(k[None, :], gb, xr[:, None])
    return E @ cf

def proj(f, l):   # [f]_l on the Schmidt harmonic
    return 2*np.pi*(f*wq[None, :]) @ np.conj(Y[l][0]) / (4*np.pi/(2*l + 1))

def run(walls):
  global rng
  rng = np.random.default_rng(12345)
  res = {}
  for trial in range(ntr):
      x = np.zeros(A.shape[1], dtype=complex)
      Pv = {}; Tv = {}; Sv = {}
      for l in lp:
          cP = rand_cheb(iP); cS = rand_cheb(iP)
          if walls:   # P = 0 at both walls: times (1 - x^2)
              tmp = C.chebmul(cP, [0.5, 0, -0.5])[:N]; cP = np.zeros(N, complex); cP[:len(tmp)] = tmp
          put(x, 'P', l, cP, iP); put(x, 'S', l, cS, iP)
          Pv[l] = [C.chebval(xr, cP), C.chebval(xr, C.chebder(cP))*dxdr]
          Sv[l] = C.chebval(xr, cS)
      for l in lt:
          cT = rand_cheb(iT); put(x, 'T', l, cT, iT); Tv[l] = C.chebval(xr, cT)
      Ax, Bx = A @ x, B @ x
      # physics on the grid
      ur = np.zeros((nr, len(xq)), complex); ut_ = np.zeros_like(ur); sg = np.zeros_like(ur)
      for l in lp:
          L = l*(l + 1); y, dy = Y[l]
          ur += np.outer(L*Pv[l][0]/r, y)
          ut_ += np.outer((Pv[l][0] + r*Pv[l][1] + r*hp*Pv[l][0])/r, dy)
          sg += np.outer(Sv[l], y)
      for l in lt:
          y, dy = Y[l]
          ut_ += np.outer(Tv[l], 1j*m*y/sth)
      rhs = -ur*dS[:, None]
      if par.diff_rot:
          dO = f0[:, None] + f2[:, None]*(1.5*xq**2 - 0.5)[None, :]
          rhs = rhs - 1j*m*G*dO*sg
          if tw:
              rhs = rhs - (ur*Sr + (ut_/r[:, None])*Sth)
      # compare per row block, for every (gb, first row index, weight)
      for gb, rstart, w in itertools.product((0, 1, 2), (0, 1) if ricb == 0 else (0,), (1, 2)):
          eB = []; eA = []; cs = []
          for l in lp:
              L = l*(l + 1); b = 2*nb*N1 + lp.index(l)*N1
              fB = evalrows(Bx[b:b + N1], gb, rstart); fA = evalrows(Ax[b:b + N1], gb, rstart)
              refB = r**w*Sv[l]/L**3; refA = r**w*proj(rhs, l)/L**3
              cs.append((refB.conj() @ fB)/(refB.conj() @ refB))
              eB.append((fB, refB)); eA.append((fA, refA))
          c = np.median(np.real(cs))
          errB = max(np.max(np.abs(fB - c*rB)) for fB, rB in eB)/max(np.max(np.abs(c*rB)) for fB, rB in eB)
          errA = max(np.max(np.abs(fA - c*rA)) for fA, rA in eA)/max(np.max(np.abs(c*rA)) for fA, rA in eA)
          key = (gb, rstart, w)
          res[key] = max(res.get(key, (0, 0))[0], errB), max(res.get(key, (0, 0))[1], errA)

  return res

claim = (1 if ut.adiab_tw else 0, (iP if w_claim == 2 else 1 - iP) if ricb == 0 else 0, w_claim)
ok = True
for walls in (True, False):
    res = run(walls)
    eB, eA = res[claim]
    alt = min(max(v) for k, v in res.items() if k != claim)
    tag = 'P = 0 at the walls' if walls else 'P free at the walls (diagnostic)'
    print('%-34s claimed C^(%d), first row %d, r^%d:  B %.2e  A %.2e   best other (basis, row, weight) %.2e'
          % (tag, claim[0], claim[1], claim[2], eB, eA, alt))
    if walls:
        ok = max(eB, eA) < tol and alt > 1e-3
print('row_check: N=%d ricb=%g m=%d symm=%d thermal_wind=%d G=%g  -> %s' % (N, ricb, m, par.symm, tw, G, 'ok' if ok else 'FAIL'))
