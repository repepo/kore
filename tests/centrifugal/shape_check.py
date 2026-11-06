#!/usr/bin/env python3
'''
shape_check.py: how a damped mode moves when the walls are deformed, exactly at finite Ek, against kore's centrifugal = 1
(NEW 2026-11-06, phase 2). This tests the dissipative terms of the deformed body where the deformation reaches the walls
(the relabelling tests cannot: there the map vanishes at the walls): the viscous force, the stress-free rows, the thermal
diffusion and the fixed-flux row.

The reference uses the mode of the sphere alone (f = 0, kore's spherical code). The first-order shift of an eigenvalue
when a wall moves along its normal by f V_n follows from the mode u and its adjoint û (Hadamard's formula); for the
rotating fluid the adjoint is the mirror image, û(x) = conj(R u(R x)), R: x -> -x, so all the products below are of u with
R u(R x), without conjugates:
  no-slip walls   λ' <û, u> = Ek ∮ V_n ∂n û*·∂n u dS
  fixed entropy   λ' ∫|s|² = κ ∮ V_n |∂n s|² dS
  fixed flux      λ' ∫|s|² = -∮ V_n (κ |∇Γ s|² + λ |s|²) dS
and, for kore's 'uniform' map X = F α, F = I + f diag(1/3, 1/3, -2/3) (linear: the spheroid a = 1 + f/3, c = 1 - 2f/3), a
volume formula that holds for no-slip and stress-free walls alike (the stress-free condition is natural in the weak form):
  λ' <û, u> = -[λ ∫ û*·2δF u + (2/3) ∫ û*·2ẑ×u + Ek ∫ 2 (δe(û)*:e(u) + e(û)*:δe(u))],
  δe(w) = sym(δF ∇w - ∇w δF), δF = diag(1/3, 1/3, -2/3),
the first-order part of the weak form pulled back by the map (∇_X u = F ∇w F⁻¹/J for u = F w/J). V_n = ξ_r = ε r P2 on each
wall (ε = -2f/3), outward from the fluid. The formulas are checked by a dilation (V_n = r on the walls: dλ/dR = -2 Ek dλ/dEk,
and -2λ for diffusion), and the two flow formulas against each other on no-slip walls (2e-12).

kore's dλ/df: runs at f = ±δ, ±2δ (δ = 5e-3) and Richardson's extrapolation (smaller steps leave the eigenvalues' last
digits in the derivative). The checks, at Ek = 1e-3, κ = 1e-2 (heat: Beyonce = 0, so the diffusion modes keep u = 0):
  no-slip       full sphere m = 1 (the spin-over) and m = 2, a shell (ricb 0.35, both walls deformed): kore against
                Hadamard and the volume formula (1e-10; leaving out the metric viscous terms puts kore 5e-2 off)
  stress-free   full sphere m = 1 and 2, a shell: kore against the volume formula (5e-10 to 3e-8); the spin-over, a
                rigid rotation, moves as in the inviscid fluid, λ' = i (no strain: Ek drops out)
  heat          fixed entropy and fixed flux, a full sphere and a shell: kore against Hadamard (4e-10 to 3e-9)
Each prints the relative difference; the tolerance is 1e-7.

Usage: python3 shape_check.py [NPROC]   (builds its runs with mk_run.py in the current directory; two and a half minutes on
four cores; NPROC parallel cases, default 4)
'''
import os, sys, json, subprocess
import numpy as np

HERE = os.path.abspath(__file__)
EK, KAPPA, DL, DE = 1e-3, 1e-2, 5e-3, 1e-4

# ------------------------------------------------------------------------------------------------- the worker (in a run)
def worker(kind, sigma, Ek):
    sys.path.insert(0, 'bin')
    from parameters import par
    import scipy.sparse as ss, scipy.sparse.linalg as ssl
    import utils as ut, utils4pp as upp, angular
    lam, x = eig('.', sigma, vectors=True)
    ri = par.ricb
    P2 = lambda c: 1.5*c**2 - 0.5
    lp, lt, _ = ut.ell(par.m, par.lmax, par.symm)
    xg, wg = np.polynomial.legendre.leggauss(par.lmax + 20)
    Yg, dYg, Zg = angular.ylm_grid(par.m, par.lmax + 1, xg)
    rg, wr = np.polynomial.legendre.leggauss(par.N + 20)
    rg, wr = ri + (1 - ri)*(rg + 1)/2, (1 - ri)*wr/2
    walls = [(1.0, 1.0)] + ([(ri, -1.0)] if ri > 0 else [])      # (radius, outward normal along r̂)
    rad = lambda C, l, ls, r, n: ut.funcheb(C[list(ls).index(l), :], r, par.ricb, ut.rcmb, n).T

    def flow_amp(r):
        '''the amplitudes of e^{imφ}: a_r, a_θ, a_φ and their r-derivatives on (r, cosθ = xg); kore's flow is
        Σ q Y r̂ + s r∇Y + t ∇Y × r⃗ (utils4pp.curl), q = L P/r, s = (r P)'/r, t = T'''
        P, T = upp.expand_reshape_sol(x, par.symm)
        out = {k: np.zeros((2, len(r), len(xg)), complex) for k in 'rtp'}
        for l in lp:
            p = rad(P, l, lp, r, 2)
            q = [l*(l + 1)*p[0]/r, l*(l + 1)*(p[1]/r - p[0]/r**2)]
            s = [p[1] + p[0]/r, p[2] + p[1]/r - p[0]/r**2]
            for d in range(2):
                out['r'][d] += np.outer(q[d], Yg[l]); out['t'][d] += np.outer(s[d], dYg[l]); out['p'][d] += np.outer(s[d], Zg[l])
        for l in lt:
            t = rad(T, l, lt, r, 1)
            for d in range(2):
                out['t'][d] += np.outer(t[d], Zg[l]); out['p'][d] -= np.outer(t[d], dYg[l])
        return out

    def hadamard_flow(V):
        '''Ek ∮ V_n ∂n û*·∂n u dS / <û, u>, V(c, rw) the displacement of the wall rw along r̂'''
        a = flow_amp(rg)
        vol = np.einsum('i,j,ij->', wr*rg**2, wg, a['r'][0]**2 + a['t'][0]**2 - a['p'][0]**2)
        wall = 0
        for rw, sg in walls:
            b = flow_amp(np.array([rw]))
            wall += sg*rw**2*np.einsum('j,j->', wg*V(xg, rw), (b['r'][1]**2 + b['t'][1]**2 - b['p'][1]**2)[0])
        return Ek*wall/vol

    def u_cart(X):
        '''the Cartesian velocity at the points X (npts, 3)'''
        P, T = upp.expand_reshape_sol(x, par.symm)
        r = np.sqrt((X**2).sum(1)); ct = X[:, 2]/r; ph = np.arctan2(X[:, 1], X[:, 0]); st = np.sqrt(1 - ct**2)
        Y, dY, Z = angular.ylm_grid(par.m, par.lmax + 1, ct)
        ar, at, ap = (np.zeros(len(r), complex) for _ in range(3))
        for l in lp:
            p = rad(P, l, lp, r, 1)
            q, s = l*(l + 1)*p[0]/r, p[1] + p[0]/r
            ar += q*Y[l]; at += s*dY[l]; ap += s*Z[l]
        for l in lt:
            t = rad(T, l, lt, r, 0)[0]
            at += t*Z[l]; ap -= t*dY[l]
        e = np.exp(1j*par.m*ph); cp, sp = np.cos(ph), np.sin(ph)
        return ((ar*e)[:, None]*np.stack([st*cp, st*sp, ct], 1) + (at*e)[:, None]*np.stack([ct*cp, ct*sp, -st], 1)
                + (ap*e)[:, None]*np.stack([-sp, cp, 0*cp], 1))

    def volume_flow():
        nphi = 4*abs(par.m) + 12
        ph = 2*np.pi*np.arange(nphi)/nphi
        R_, C_, F_ = np.meshgrid(rg, xg, ph, indexing='ij')
        W = ((wr*rg**2)[:, None, None]*wg[None, :, None]*(2*np.pi/nphi)*np.ones_like(R_)).ravel()
        S_ = np.sqrt(1 - C_**2)
        X = np.stack([R_*S_*np.cos(F_), R_*S_*np.sin(F_), R_*C_], -1).reshape(-1, 3)
        Rm = np.diag([-1., 1., 1.])
        fu, fv = u_cart, lambda X: u_cart(X @ Rm) @ Rm             # v = R u(R x) = û*
        def grad(fun, h=1e-4):                                      # ∂u_i/∂x_j, fourth-order differences
            G = np.zeros((len(X), 3, 3), complex)
            for j in range(3):
                e = np.zeros(3); e[j] = h
                G[:, :, j] = (8*(fun(X + e) - fun(X - e)) - (fun(X + 2*e) - fun(X - 2*e)))/(12*h)
            return G
        u, v, Gu, Gv = fu(X), fv(X), grad(fu), grad(fv)
        dF = np.diag([1/3, 1/3, -2/3])
        sym = lambda G: (G + np.swapaxes(G, 1, 2))/2
        de = lambda G: sym(np.einsum('ik,pkj->pij', dF, G) - np.einsum('pik,kj->pij', G, dF))
        zc = lambda w: np.stack([-w[:, 1], w[:, 0], 0*w[:, 0]], 1)
        m0 = W @ (v*u).sum(1)
        res0 = abs(lam*m0 + W @ (v*2*zc(u)).sum(1) + Ek*(W @ (2*(sym(Gv)*sym(Gu)).sum((1, 2)))))/abs(lam*m0)
        d = -(lam*(W @ (v*(2*u @ dF)).sum(1)) + (2/3)*(W @ (v*2*zc(u)).sum(1))
              + Ek*(W @ (2*((de(Gv)*sym(Gu)).sum((1, 2)) + (sym(Gv)*de(Gu)).sum((1, 2))))))/m0
        return d, res0

    def hadamard_heat(V):
        S = upp.expand_reshape_sol(x[ut.offset['h']:ut.offset['h'] + ut.n], par.symm)
        def amp(r):
            o = {k: np.zeros((len(r), len(xg)), complex) for k in ('s', 'sr', 'st', 'sp')}
            for l in lp:
                p = rad(S, l, lp, r, 1)
                o['s'] += np.outer(p[0], Yg[l]); o['sr'] += np.outer(p[1], Yg[l])
                o['st'] += np.outer(p[0], dYg[l]); o['sp'] += np.outer(p[0], Zg[l])
            return o
        a = amp(rg)
        vol = np.einsum('i,j,ij->', wr*rg**2, wg, abs(a['s'])**2)
        wall = 0
        for (rw, sg), bc in zip(walls, (par.bco_thermal, par.bci_thermal)):
            b = amp(np.array([rw]))
            if bc == 0:
                w = KAPPA*abs(b['sr'][0])**2
            else:
                w = -(KAPPA*(abs(b['st'][0])**2 + abs(b['sp'][0])**2)/rw**2 + lam*abs(b['s'][0])**2)
            wall += sg*rw**2*np.einsum('j,j->', wg*V(xg, rw), w)
        return wall/vol, np.linalg.norm(x[:ut.offset['h']])/np.linalg.norm(x)

    shape = lambda c, rw: -2/3*rw*P2(c)          # ξ_r = ε r P2 per unit f
    dil = lambda c, rw: rw + 0*c                 # ξ_r = r
    out = {'lam': [lam.real, lam.imag]}
    if kind == 'noslip':
        h, d = hadamard_flow(shape), hadamard_flow(dil)
        v, res0 = volume_flow()
        out.update(had=[h.real, h.imag], dil=[d.real, d.imag], vol=[v.real, v.imag], res0=res0)
    elif kind == 'free':
        v, res0 = volume_flow()
        out.update(vol=[v.real, v.imag], res0=res0)
    else:
        (h, fl), (d, _) = hadamard_heat(shape), hadamard_heat(dil)
        out.update(had=[h.real, h.imag], dil=[d.real, d.imag], flow=fl)
    print(json.dumps(out))

# ------------------------------------------------------------------------------------------------------------ the driver
def eig(d, sigma, k=4, vectors=False):
    import scipy.sparse as ss, scipy.sparse.linalg as ssl
    def load(M):
        z = np.load(os.path.join(d, M + '.npz'))
        return ss.csr_matrix((z['data'], z['indices'], z['indptr']), shape=z['shape']).astype(complex)
    A, B = load('A'), load('B')
    lu = ssl.splu((A - sigma*B).tocsc())
    op = ssl.LinearOperator(A.shape, matvec=lambda x: lu.solve(B @ x), dtype=complex)
    nu, X = ssl.eigs(op, k=k, which='LM', tol=1e-14, ncv=40)
    lam = sigma + 1/nu
    i = np.argmin(abs(lam - sigma))
    return (lam[i], X[:, i]) if vectors else lam[i]

def build(name, extra):
    if os.path.exists(os.path.join(name, 'A.npz')):
        return
    p = subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), 'mk_run.py'), name, extra], capture_output=True,
                       text=True)
    if p.returncode:
        raise RuntimeError('%s could not be built:\n%s' % (name, p.stdout[-3000:]))

# name: (kind, parameters, target at f = 0)
FLOW = "par.ViscosD = %r; par.rpower_v = 4; " % EK
HEAT = ("par.thermal = 1; par.stratification = 'linear'; par.Beyonce = 0.; par.ThermaD = %r; par.m = 1; par.symm = -1; "
        "par.N = 48; par.lmax = 30; " % KAPPA)
SHELL = "par.ricb = 0.35; par.bci = %d; par.bco = %d; "
CASES = {
    'noslip_m1':  ('noslip', FLOW + "par.bco = 1; par.N = 64; par.lmax = 40", -0.0989 + 1.0037j),
    'noslip_m2':  ('noslip', FLOW + "par.bco = 1; par.m = 2; par.symm = 1; par.N = 64; par.lmax = 41", -0.1523 + 1.2467j),
    'noslip_sh':  ('noslip', FLOW + SHELL % (1, 1) + "par.N = 64; par.lmax = 40", -0.1020 + 1.0057j),
    'free_m1':    ('free', FLOW + "par.bco = 0; par.symm = 1; par.N = 64; par.lmax = 40", -0.1354 + 0.7707j),
    'free_m2':    ('free', FLOW + "par.bco = 0; par.m = 2; par.N = 48; par.lmax = 31", -0.0068 + 0.6666j),
    'free_sh':    ('free', FLOW + SHELL % (0, 0) + "par.m = 2; par.symm = 1; par.N = 48; par.lmax = 31", -0.0473 + 1.2297j),
    'free_so':    ('free', FLOW + "par.bco = 0; par.N = 48; par.lmax = 30", 1j),
    'entropy':    ('heat', HEAT + "par.bco_thermal = 0", -KAPPA*5.76345919689**2),
    'flux':       ('heat', HEAT + "par.bco_thermal = 1", -KAPPA*3.34209365736**2),
    'entropy_sh': ('heat', HEAT + "par.bco_thermal = 0; par.bci_thermal = 0; par.ViscosD = 1e-2; " + SHELL % (1, 1), -0.375),
    'flux_sh':    ('heat', HEAT + "par.bco_thermal = 1; par.bci_thermal = 1; par.ViscosD = 1e-2; " + SHELL % (1, 1), -0.318),
}

def run_case(name):
    kind, extra, sigma = CASES[name]
    steps = {0: 0.0, 1: DL, -1: -DL, 2: 2*DL, -2: -2*DL}
    extra = extra.strip().rstrip(';')
    for k, f in steps.items():
        build('%s_%+d' % (name, k), extra + '; par.centrifugal_f = %r' % f)
    if kind == 'noslip':      # the dilation: Ek (1 ± 1e-4), a central difference
        for k in (1, -1):
            build('%s_E%+d' % (name, k), extra.replace('par.ViscosD = %r' % EK, 'par.ViscosD = %r' % (EK*(1 + k*DE))))
    p = subprocess.run([sys.executable, HERE, '--worker', kind, repr(sigma), repr(EK)], cwd='%s_+0' % name, capture_output=True,
                       text=True)
    if p.returncode:
        raise RuntimeError('%s: the worker failed:\n%s' % (name, p.stderr[-3000:]))
    w = json.loads(p.stdout.strip().splitlines()[-1])
    l0 = complex(*w['lam'])
    lam = {k: eig('%s_%+d' % (name, k), l0) for k in (1, -1, 2, -2)}
    kd = (4*(lam[1] - lam[-1])/(2*DL) - (lam[2] - lam[-2])/(4*DL))/3
    msg, rels = 'λ %.8f%+.8fi  dλ/df kore %.9f%+.9fi' % (l0.real, l0.imag, kd.real, kd.imag), []
    if 'had' in w:
        h = complex(*w['had']); rels.append(abs(kd - h)/abs(kd)); msg += ', Hadamard %.1e' % rels[-1]
        d = complex(*w['dil'])
        if kind == 'noslip':
            lE = {k: eig('%s_E%+d' % (name, k), l0) for k in (1, -1)}
            ref = -2*EK*(lE[1] - lE[-1])/(2*DE*EK)
        else:
            ref = -2*l0
        rels.append(abs(d - ref)/abs(ref)); msg += ' (dilation %.1e)' % rels[-1]
    if 'vol' in w:
        v = complex(*w['vol']); rels.append(abs(kd - v)/abs(kd)); msg += ', volume formula %.1e' % rels[-1]
        msg += ' (its weak form at f = 0: %.0e)' % w['res0']
    if kind == 'heat':
        msg += ', flow part of the mode %.0e' % w['flow']
    return name, max(rels), msg

if __name__ == '__main__':
    if '--worker' in sys.argv:
        i = sys.argv.index('--worker')
        worker(sys.argv[i + 1], complex(sys.argv[i + 2]), float(sys.argv[i + 3]))
        sys.exit(0)
    from multiprocessing import Pool
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    fail = []
    with Pool(nproc) as pool:
        for name, rel, msg in pool.imap(run_case, list(CASES)):
            ok = rel < 1e-7
            print('%-11s %s  %s' % (name + ':', 'ok  ' if ok else 'FAIL', msg))
            if not ok:
                fail.append(name)
    print('shape_check.py: %s' % ('FAIL ' + ', '.join(fail) if fail else 'ok'))
    sys.exit(1 if fail else 0)
