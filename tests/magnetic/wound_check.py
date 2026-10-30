#!/usr/bin/env python3
'''
wound_check.py: the wound toroidal field of B0_wound = 1 (background_field.wound_profiles), NEW 2026-10-30.

WHY. The wound field B_φ solves the φ-component of the steady induction equation, B0_wound_eta MagnetD (∇² - 1/s²)B_φ =
-Gaspard s B_p⋅∇ΔΩ, degree by degree on a Chebyshev grid. These checks do not reuse that construction:
  1. exact: a uniform axial B_p and ΔΩ = a r² give one degree, k_2 = A r⁴/14 + C r² + D r⁻³ (A = -2 G a B_z/(3 η)), with C, D
     from the boundary conditions (insulating walls; a conducting inner core, r k' + (1 - 3/sigma_i2o) k = 0; regular at r = 0);
  2. residual: (∇² - 1/s²)B_φ by finite differences in (r, θ) of B_φ(r, θ), and B_p from the flux function
     ψ = r Σ h_n (1 - x²) P_n' (B_r = ∂θψ/(r² sinθ), B_θ = -∂rψ/(r sinθ)) and ∇ΔΩ also by finite differences, at points off the
     collocation grid, relative to max |G s B_p⋅∇ΔΩ|/η (4th-order differences, step 1e-3: ~1e-9 is the floor);
  3. convergence: max |k_n(N) - k_n(2N)| relative to max |k_n| (wound_profiles uses max(96, 2N) + 1 nodes).
HOW. Each case is a run directory built from the repo's bin/ (tests/magnetic/mk_run.py's defaults, no matrices); this script
runs itself inside it with --inner. usage: python3 tests/magnetic/wound_check.py   (in a scratch directory; ~1 minute)
'''
import os, sys, subprocess
T = os.path.dirname(os.path.abspath(__file__))
DR = "par.diff_rot = 1; par.diff_rot_type = 'user'; par.diff_rot_profiles = {0: '0.05*r**2', 2: '-0.1*r**2', 4: '-0.03*r**4'}"
EX = "par.structure = 'Boussinesq'; par.diff_rot = 1; par.diff_rot_type = 'user'; par.diff_rot_profiles = {0: 'r**2'}; par.diff_rot_ampl = 0.1"
CASES = {   # extra parameters of mk_run.py; 'exact' cases also compare with the closed form
    'wd_ax':     (EX, 1),
    'wd_ax_cic': (EX + "; par.innercore = 'conducting, Chebys'; par.sigma_i2o = 3.0", 1),
    'wd_ax_fs':  (EX + "; par.ricb = 0", 1),
    'wd_gen':    (DR + "; par.B0 = 'general'; par.B0_profiles = {1: 'r/6 - r**3/10', 3: '0.3*r**3*(1 - r**2)'}", 0),
    'wd_s2_fs':  (DR + "; par.ricb = 0; par.B0 = 'Luo_S2'; par.symm = 1", 0),
    'wd_fdm_cic': ("par.diff_rot = 1; par.diff_rot_type = 'user'; par.diff_rot_profiles = {0: '0.1*(r - 0.35)**2', 2: '-0.2*(r - 0.35)**2*r'}"
                   "; par.B0 = 'FDM'; par.innercore = 'conducting, Chebys'; par.B0_wound_eta = 5.0", 0),
}

def inner(exact):
    sys.path.insert(0, 'bin')
    import numpy as np
    from parameters import par
    import background as bg, background_field as bf, diff_rot as dr
    bg.check(); dr.check()
    W = bf.wound_profiles()
    eta = par.B0_wound_eta*par.MagnetD
    P = np.polynomial.legendre.Legendre
    ri = par.ricb
    out = 'degrees %s' % sorted(W)
    if exact:   # 1. the closed form
        h = bf.pol_profiles()[1]
        Bz = 2*h(np.array([0.7]), 0)[0]/0.7
        A = -2*par.Gaspard*par.diff_rot_ampl*Bz/(3*eta)
        if ri == 0:
            C, D = -A/14, 0.
        else:
            q = 0. if par.innercore == 'insulator' else 1 - 3/par.sigma_i2o
            if par.innercore == 'insulator':
                M = np.array([[ri**2, ri**-3], [1., 1.]]); rhs = -A/14*np.array([ri**4, 1.])
            else:
                M = np.array([[(2 + q)*ri**2, (q - 3)*ri**-3], [1., 1.]]); rhs = -A/14*np.array([(4 + q)*ri**4, 1.])
            C, D = np.linalg.solve(M, rhs)
        r = np.linspace(max(ri, 1e-3), 1, 201)
        ex = A/14*r**4 + C*r**2 + D*r**-3
        out += '  exact %.1e' % (abs(W[2](r) - ex).max()/abs(ex).max())
    # 2. the residual by finite differences, off the grid
    hd = 1e-3
    r0 = np.linspace(max(ri, 0.) + 0.05, 0.95, 23) + 0.0071
    t0 = np.linspace(0.2, np.pi - 0.2, 19) + 0.0113
    pol = bf.pol_profiles()
    def Bphi(r, t):
        x = np.cos(t)
        return sum(c(r)*np.sin(t)*P.basis(n).deriv()(x) for n, c in W.items())
    def psi(r, t):
        x = np.cos(t)
        return sum(r*f(np.atleast_1d(r), 0)*(1 - x**2)*P.basis(n).deriv()(x) for n, f in pol.items())
    def dOm(r, t):
        return sum(dr.raw(n, np.atleast_1d(r))*P.basis(n)(np.cos(t)) for n in dr.degrees())
    def d1(f, r, t, ax):   # 4th-order first derivative
        e = (hd, 0) if ax == 0 else (0, hd)
        g = lambda k: f(r + k*e[0], t + k*e[1])
        return (-g(2) + 8*g(1) - 8*g(-1) + g(-2))/(12*hd)
    def d2(f, r, t, ax):
        e = (hd, 0) if ax == 0 else (0, hd)
        g = lambda k: f(r + k*e[0], t + k*e[1])
        return (-g(2) + 16*g(1) - 30*g(0) + 16*g(-1) - g(-2))/(12*hd**2)
    worst = 0; big = 0
    R, Th = np.meshgrid(r0, t0, indexing='ij')
    for r, t in zip(R.ravel(), Th.ravel()):
        s = r*np.sin(t)
        lap = (d2(Bphi, r, t, 0) + 2*d1(Bphi, r, t, 0)/r + (d2(Bphi, r, t, 1) + np.cos(t)/np.sin(t)*d1(Bphi, r, t, 1))/r**2
               - Bphi(r, t)/s**2)
        Br = d1(psi, r, t, 1)/(r**2*np.sin(t)); Bt = -d1(psi, r, t, 0)/(r*np.sin(t))
        src = par.Gaspard*s*(Br*d1(dOm, r, t, 0) + Bt*d1(dOm, r, t, 1)/r)/eta
        worst = max(worst, float(np.abs(lap + src).max())); big = max(big, float(np.abs(src).max()))
    out += '  FD residual %.1e' % (worst/big)
    # 3. convergence with N
    rr = np.linspace(-1 if ri == 0 else ri, 1, 301)
    # wound_profiles uses max(96, 2N) (+1) nodes: compare 97 nodes (N <= 48) and 193 (N = 96) with 385 (N = 192)
    errs = []
    for N in (48, 96, 192):
        par.N = N; bf._wnd.clear()
        errs.append(bf.wound_profiles())
    big = max(abs(c(rr)).max() for c in errs[-1].values())
    d = lambda a, b: max(abs(a[n](rr) - b[n](rr)).max() for n in b)/big
    out += '  nodes 97/193 %.1e, 193/385 %.1e' % (d(errs[0], errs[2]), d(errs[1], errs[2]))
    print(out)
    return worst/big

if __name__ == '__main__':
    if '--inner' in sys.argv:
        inner(int(sys.argv[-1])); sys.exit(0)
    bad = 0
    for name, (extra, exact) in CASES.items():
        subprocess.run([sys.executable, T + '/mk_run.py', name, extra + '; par.B0_wound = 1; par.N = 48'], check=True,
                       stdout=subprocess.DEVNULL)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), '--inner', str(exact)], cwd=name, capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith('degrees')]
        print('%-11s %s' % (name, line[0] if line else 'FAILED ' + (r.stdout + r.stderr)[-800:]))
        bad += not line
    sys.exit(1 if bad else 0)
