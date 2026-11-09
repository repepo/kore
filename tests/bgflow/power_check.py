#!/usr/bin/env python3
'''
usage: python3 power_check.py RUNDIR   (a run of kore with bgflow = 1 and bgflow_profiles, solved and post-processed)

An independent check of spin_doctor's powers of a background flow (bgflow = 1): Wbg and Wbg_im (flow.dat), Wbg_thm
(thermal.dat) and Wbg_cmp (composition.dat) of the first solution, against the same integrals computed here on a grid of
(r, θ) from the textbook components of the advection in spherical coordinates,
    (A⋅∇)B = [A⋅∇B_r - (A_θ B_θ + A_φ B_φ)/r,  A⋅∇B_θ + (A_θ B_r - A_φ B_φ cot θ)/r,  A⋅∇B_φ + A_φ (B_r + B_θ cot θ)/r],
    A⋅∇f = A_r ∂r f + (A_θ/r) ∂θ f + (A_φ/(r sinθ)) ∂φ f,
    Wbg = -2 ∫ ρ₀ 𝐮*⋅[(𝐔⋅∇)𝐮 + (𝐮⋅∇)𝐔] dV (real part, and imaginary part for Wbg_im),
    Wbg_thm = -2 Re ∫ p₀ s* (𝐔⋅∇s + 𝐮⋅∇S) dV,   Wbg_cmp = -2 Re ∫ ρ₀ c* (𝐔⋅∇c + 𝐮⋅∇C) dV,
with 𝐔, S, C from the expressions of bgflow_profiles themselves (differentiated by numpy's Chebyshev series of 200 terms,
not by kore's profile codes) and 𝐮, s, c from the solution's Chebyshev coefficients. spin_doctor builds the force from
𝐔 × ω + 𝐮 × Ω_U and the curl of the background's vectors instead (utils4pp.bgflow_setup). Prints one JSON line with the
relative differences, and the kinetic energy ∫ ρ₀|𝐮|² dV here over flow.dat's (the normalisation, 1).
'''
import os, sys, json
import numpy as np
os.chdir(sys.argv[1])
sys.path.insert(0, 'bin')
from parameters import par   # noqa: E402
import utils as ut           # noqa: E402
import utils4pp as upp       # noqa: E402
import angular               # noqa: E402
import radial_profiles as rap  # noqa: E402

m, ri, ro = par.m, par.ricb, ut.rcmb
G = par.Gaspard if par.Gaspard != 0 else 1.0
Ro = par.bgflow_Ro
NR, NX = 4*par.N, 2*par.lmax + 40
xr, wr = np.polynomial.legendre.leggauss(NR)
rr = ri + (ro - ri)*(xr + 1)/2
wr = wr*(ro - ri)/2
x, wx = np.polynomial.legendre.leggauss(NX)
sth, cth = np.sqrt(1 - x**2), x
r = rr[:, None]
rho = rap.prf.density(rr, 0)[:, None]
lam1 = rap.logrhoX(rr, 1)[:, None]
lam2 = rap.logrhoX(rr, 2)[:, None]

# --- the background, from the expressions
def series(expr, n):
    '''Chebyshev series of an expression of bgflow_profiles on the fluid (|r| times sign(r)^n in a full sphere, as kore)'''
    f = lambda z: np.broadcast_to(eval(expr, {'np': np, '__builtins__': {}}, {'r': np.abs(z) if ri == 0 else z, 'ricb': ri}),
                                  np.shape(z))*(np.sign(z)**n if ri == 0 else 1)
    return np.polynomial.Chebyshev.interpolate(f, 199, domain=[-1, 1] if ri == 0 else [ri, 1])
z0 = np.zeros((NR, NX))
U = [z0.copy() for _ in range(3)]; Ur = [z0.copy() for _ in range(3)]; Ut = [z0.copy() for _ in range(3)]
gS, gC = [z0.copy(), z0.copy()], [z0.copy(), z0.copy()]   # (∂r, ∂θ) of S, C
Pl = np.polynomial.Legendre
for key, expr in sorted(par.bgflow_profiles.items()):
    k, n = key[0], int(key[1:])
    f = series(expr, n)
    F = [f.deriv(j)(rr)[:, None] if j else f(rr)[:, None] for j in range(4)]
    Y, dY = Pl.basis(n)(x), -sth*Pl.basis(n).deriv()(x)              # P_n(cos θ), ∂θ
    d2Y = -cth*Pl.basis(n).deriv()(x) + sth**2*Pl.basis(n).deriv(2)(x)
    if k == 'P':   # U_r = L P/r Y, U_θ = (P' + P/r + λ1 P) ∂θY
        L = n*(n + 1.)
        q, dq = L*F[0]/r, L*(F[1]/r - F[0]/r**2)
        s = F[1] + F[0]/r + lam1*F[0]
        ds = F[2] + F[1]/r - F[0]/r**2 + lam2*F[0] + lam1*F[1]
        a = Ro*G
        U[0] += a*q*Y; Ur[0] += a*dq*Y; Ut[0] += a*q*dY
        U[1] += a*s*dY; Ur[1] += a*ds*dY; Ut[1] += a*s*d2Y
    elif k == 'T':   # U_φ = -T ∂θY
        a = Ro*G
        U[2] += -a*F[0]*dY; Ur[2] += -a*F[1]*dY; Ut[2] += -a*F[0]*d2Y
    else:            # S or C = Ro f Y
        g = gS if k == 'S' else gC
        g[0] += Ro*F[1]*Y; g[1] += Ro*F[0]*dY

# --- the solution (the first one)
def read(name):   # name.npy or the text files (utils.load_field)
    return ut.load_field(name, 0)
lp, lt, _ = ut.ell(m, par.lmax, par.symm)
Ys, dYs, Zs = angular.ylm_grid(m, par.lmax + 2, x)          # Z = i m Y/sinθ
L_ = lambda l: l*(l + 1.)
def d2theta(l):   # ∂θ²Y from the Legendre equation
    return -cth/sth*dYs[l] + (m**2/sth**2 - L_(l))*Ys[l]
def dZ(l):        # ∂θ (i m Y/sinθ)
    return 1j*m*(dYs[l]/sth - cth*Ys[l]/sth**2)
cheb = lambda c, n: ut.funcheb(np.asarray(c), rr, ri, ro, n)
sol = read('flow')
Psol, Tsol = upp.expand_reshape_sol(sol, par.symm)
zc = np.zeros((NR, NX), complex)
u = [zc.copy() for _ in range(3)]; ur = [zc.copy() for _ in range(3)]; ut_ = [zc.copy() for _ in range(3)]
for c, l in zip(Psol, lp):
    f = cheb(c, 3)
    P, dP, d2P = f[:, 0][:, None], f[:, 1][:, None], f[:, 2][:, None]
    q, dq = L_(l)*P/r, L_(l)*(dP/r - P/r**2)
    s = dP + P/r + lam1*P
    ds = d2P + dP/r - P/r**2 + lam2*P + lam1*dP
    u[0] += q*Ys[l]; ur[0] += dq*Ys[l]; ut_[0] += q*dYs[l]
    u[1] += s*dYs[l]; ur[1] += ds*dYs[l]; ut_[1] += s*d2theta(l)
    u[2] += s*Zs[l]; ur[2] += ds*Zs[l]; ut_[2] += s*dZ(l)
for c, l in zip(Tsol, lt):
    f = cheb(c, 1)
    T, dT = f[:, 0][:, None], f[:, 1][:, None]
    u[1] += T*Zs[l]; ur[1] += dT*Zs[l]; ut_[1] += T*dZ(l)
    u[2] += -T*dYs[l]; ur[2] += -dT*dYs[l]; ut_[2] += -T*d2theta(l)
up = [1j*m*c for c in u]   # ∂φ

def adv(A, B, Br, Bt, Bp):
    '''(A⋅∇)B, the textbook components (Bp = ∂φB)'''
    cot = cth/sth
    grad = lambda k: A[0]*Br[k] + A[1]/r*Bt[k] + A[2]/(r*sth)*Bp[k]
    return [grad(0) - (A[1]*B[1] + A[2]*B[2])/r,
            grad(1) + (A[1]*B[0] - A[2]*B[2]*cot)/r,
            grad(2) + A[2]*(B[0] + B[1]*cot)/r]
zero = [0*U[0]]*3
Fv = [-(a + b) for a, b in zip(adv(U, u, ur, ut_, up), adv(u, U, Ur, Ut, zero))]
vol = lambda f: 2*np.pi*np.sum(wr[:, None]*wx[None, :]*r**2*f)
W = 2*vol(rho*sum(np.conj(a)*b for a, b in zip(u, Fv)))
KE = vol(rho*sum(np.abs(a)**2 for a in u))
row = lambda f: np.loadtxt(f, ndmin=2)[0]
fl = row('flow.dat')
out = {'KE/KE_sd': KE/fl[0], 'Wbg': abs(W.real - fl[45])/abs(fl[45]), 'Wbg_im': abs(W.imag - fl[46])/max(abs(fl[46]), 1e-300),
       'Wbg_sd': fl[45]}

def scalar(name, w, g):
    c = read(name)
    if c is None:
        return None
    C = upp.expand_reshape_sol(c, par.symm)
    C = C[0] if isinstance(C, (list, tuple)) else C
    X, Xr, Xt = zc.copy(), zc.copy(), zc.copy()
    for cc, l in zip(C, lp):
        f = cheb(cc, 1)
        X += f[:, 0][:, None]*Ys[l]; Xr += f[:, 1][:, None]*Ys[l]; Xt += f[:, 0][:, None]*dYs[l]
    advs = U[0]*Xr + U[1]/r*Xt + U[2]/(r*sth)*1j*m*X + u[0]*g[0] + u[1]/r*g[1]
    return -2*vol(w*np.conj(X)*advs).real
if par.thermal:
    Wt = scalar('thermal', rap.prf.pressure(rr, 0)[:, None], gS)
    th = row('thermal.dat')
    out['Wbg_thm'] = abs(Wt - th[7])/abs(th[7]); out['Wbg_thm_sd'] = th[7]
if par.compositional:
    Wc = scalar('composition', rho, gC)
    co = row('composition.dat')
    out['Wbg_cmp'] = abs(Wc - co[7])/abs(co[7]); out['Wbg_cmp_sd'] = co[7]
print(json.dumps(out))
