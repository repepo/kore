#!/usr/bin/env python3
'''
usage: stokes_check.py [DIR]      runs the cases below in DIR (default stokes_runs) and checks them
       stokes_check.py eval RUN   (internal) prints the comparison for one run, from inside the run directory

Forcing 14 (a radial velocity A Y_2^m at the walls) against the exact solution without rotation (Gaspard = 0) in an
incompressible fluid ('Boussinesq', uniform ν). The flow then obeys the unsteady Stokes equations λ𝐮 = -∇Π + ν∇²𝐮,
λ = i forcing_frequency, and only the degree l = 2 is excited. With 𝐮 = ∇×∇×(P Y 𝐫),

    D(λ - νD) P = 0,   D = d²/dr² + (2/r) d/dr - L/r²,   L = 6,
    P = a r^l + b r^-(l+1) + c h1_l(kr) + d h2_l(kr),   k² = -λ/ν,   Im k < 0,

with P = A r/L (uᵣ = L P/r = A) and P' + P/r = 0 (no tangential velocity) at each wall; in a full sphere a r^l and
c j_l(kr) (regular at r = 0) and the outer wall. The spherical Hankel functions are written in closed form, with their
exponentials taken at the wall where each is largest (h1 decays inwards from the outer wall, h2 outwards from the inner
one): j_l and y_l both grow as e^{|Im k| r}, and with thin Stokes layers their combination loses all its digits. The potential part a r^l + b r^-(l+1) carries the pressure, Π = -λ (r P_pot)',
the Bessel part none. The check compares, for each case:
  - P_2(r) of the run with P(r), and the size of every other degree (zero without rotation);
  - spin_doctor's KE, the pressure and viscous wall powers Wp, Wv of each wall, and Dvisc with their exact values
    (Dvisc = Wp + Wv summed over the walls, since 0 = Dkin + Wp and Dkin = Wv - Dvisc);
  - residσ, residω, residD.
The amplitudes are those of the complex coefficients (the physical field is that times e^{imφ + iwt} plus the complex
conjugate), and the powers use spin_doctor's convention (2 Re of the complex products).
'''
import os, sys, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-8

CASES = {   # name: parameter changes (Gaspard = 0, no rotation, l = 2 only: a small lmax)
    'shell_m0':  "par.m = 0; par.symm = 1; par.forcing_frequency = 1.0; par.forcing_amplitude_cmb = 1.0; par.forcing_amplitude_icb = 0.5",
    'shell_m1':  "par.m = 1; par.symm = -1; par.forcing_frequency = 2.0; par.forcing_amplitude_cmb = 1.0; par.forcing_amplitude_icb = -0.3",
    'shell_m2':  "par.m = 2; par.symm = 1; par.forcing_frequency = -0.5; par.forcing_amplitude_cmb = 0.7; par.forcing_amplitude_icb = 0.2",
    'sphere_m2': "par.ricb = 0; par.m = 2; par.symm = 1; par.forcing_frequency = 1.0; par.forcing_amplitude_cmb = 1.0; par.forcing_amplitude_icb = 0.0",
}
COMMON = "par.forcing = 14; par.Gaspard = 0.0; g = 0.05"


def evaluate():
    sys.path.insert(1, 'bin/')
    import numpy.polynomial.chebyshev as ch
    from scipy.special import spherical_jn
    from parameters import par
    import utils as ut
    import utils4pp as upp

    l, L = 2, 6.
    lam, nu = 1j*par.forcing_frequency, par.ViscosD
    k = np.sqrt(-lam/nu)
    k = -k if k.imag > 0 else k
    ro, ri = float(ut.rcmb), float(par.ricb)
    shell = ri > 0
    from math import factorial
    ck = [ factorial(l + j)/(factorial(j)*factorial(l - j)) for j in range(l + 1) ]

    def hankel(r, kind, rw):   # h_l(kr) e^{∓ik rw} (kind 1, 2) and its r-derivative: e^{±ik(r - rw)} times a finite sum
        sg = 1 if kind == 1 else -1
        z = k*r
        g  = (-sg*1j)**(l + 1) * sum( ck[j]*(sg*0.5j)**j * z**(-j - 1) for j in range(l + 1) )
        dg = (-sg*1j)**(l + 1) * sum( ck[j]*(sg*0.5j)**j * (-j - 1)*z**(-j - 2) for j in range(l + 1) )
        e = np.exp(sg*1j*k*(r - rw))
        return e*g, e*(sg*1j*k*g + k*dg)

    def fun(r):   # the four (two in a full sphere) solutions and their r-derivatives
        if shell:
            [ h1, dh1 ], [ h2, dh2 ] = hankel(r, 1, ro), hankel(r, 2, ri)
            f  = [ r**l, r**(-l-1), h1, h2 ]
            df = [ l*r**(l-1), -(l+1)*r**(-l-2), dh1, dh2 ]
        else:
            jo = spherical_jn(l, k*ro)
            f  = [ r**l, spherical_jn(l, k*r)/jo ]
            df = [ l*r**(l-1), k*spherical_jn(l, k*r, True)/jo ]
        return np.array(f), np.array(df)

    walls = [ (ro, par.forcing_amplitude_cmb) ] + ([ (ri, par.forcing_amplitude_icb) ] if shell else [])
    M, rhs = [], []
    for rw, A in walls:
        f, df = fun(rw)
        M += [ f, df + f/rw ]
        rhs += [ A*rw/L, 0 ]
    c = np.linalg.solve(np.array(M, dtype=complex), np.array(rhs, dtype=complex))
    cpot = c[:2] if shell else np.array([c[0], 0])

    def fields(r):   # P, P', q, s, q', s', Π at r
        f, df = fun(r)
        P, dP = c @ f, c @ df
        d2P = -2*dP/r + L*P/r**2 + (c[2:] @ (-k**2*f[2:]) if shell else c[1]*(-k**2*f[1]))   # D P = -k² × Bessel part
        q, s = L*P/r, dP + P/r
        dq, ds = L*(dP/r - P/r**2), d2P + dP/r - P/r**2
        Pi = -lam*( cpot[0]*(l+1)*r**l - cpot[1]*l*r**(-l-1) )
        return P, q, s, dq, ds, Pi

    # the run's P_2 and the other degrees
    u2 = upp.expand_reshape_sol(ut.load_field('flow').reshape((2*ut.n, -1))[:, 0], par.symm)   # flow.npy or the text files
    lp = list(ut.ell(par.m, par.lmax, par.symm)[0])
    i2 = lp.index(l)
    r = np.linspace(ri if shell else 1e-3, ro, 400)
    Pk = ch.chebval(upp.xcheb(r, par.ricb, ut.rcmb), u2[0][i2])
    Pa = fields(r)[0]
    errP = np.max(abs(Pk - Pa))/np.max(abs(Pa))
    others = max( np.max(abs(np.delete(u2[0], i2, 0))) if len(lp) > 1 else 0, np.max(abs(u2[1])) )/np.max(abs(u2[0][i2]))

    # exact KE and wall powers
    x, w = np.polynomial.legendre.leggauss(400)
    a = ri if shell else 0.
    rq = 0.5*(ro - a)*(x + 1) + a
    _, q, s, _, _, _ = fields(rq)
    n2 = 4*np.pi/(2*l + 1)
    KE = np.sum(w*n2*rq**2*(abs(q)**2 + L*abs(s)**2))*(ro - a)/2
    Wp, Wv = {}, {}
    for name, rw, sign in (('cmb', ro, 1), ('icb', ri, -1)):
        if name == 'icb' and not shell:
            Wp[name], Wv[name] = 0j, 0.
            continue
        _, q, s, dq, ds, Pi = fields(np.array([rw]))
        Wp[name] = sign*(-2*n2*rw**2*np.conj(q)*Pi)[0]
        tq, ts = 2*dq, ds - s/rw + q/rw
        Wv[name] = sign*2*nu*n2*rw**2*np.real(np.conj(q)*tq + L*np.conj(s)*ts)[0]
    Dv = np.real(Wp['cmb'] + Wp['icb']) + Wv['cmb'] + Wv['icb']

    f = np.loadtxt('flow.dat', ndmin=2)[-1]
    got = { 'KE': f[0], 'Wp_icb': f[20], 'Wp_cmb': f[21], 'Wp_im': f[22], 'Wv_icb': f[23], 'Wv_cmb': f[24], 'Dvisc': f[25] }
    exact = { 'KE': KE, 'Wp_icb': Wp['icb'].real, 'Wp_cmb': Wp['cmb'].real, 'Wp_im': (Wp['icb'] + Wp['cmb']).imag,
              'Wv_icb': Wv['icb'], 'Wv_cmb': Wv['cmb'], 'Dvisc': Dv }
    scale = max(abs(v) for v in exact.values())
    errs = { key: abs(got[key] - exact[key])/scale for key in got }
    worst = max(errP, max(errs.values()), f[16], f[15], f[26])
    print('  P_2 error %.1e, other degrees %.1e, residσ %.1e, residω %.1e, residD %.1e' % (errP, others, f[16], f[15], f[26]))
    for key in got:
        print('  %-7s run % .10e  exact % .10e  error %.1e' % (key, got[key], exact[key], errs[key]))
    print('RESULT %s' % ('PASS' if worst < TOL and others < TOL else 'FAIL'))


def main(top):
    sys.path.insert(0, HERE)
    import mk_run
    os.makedirs(top, exist_ok=True)
    ok = True
    for name, changes in CASES.items():
        d = os.path.join(top, name)
        mk_run.build(d, COMMON + '; ' + changes)
        out = subprocess.run([sys.executable, os.path.join(HERE, 'stokes_check.py'), 'eval'], cwd=d,
                             capture_output=True, text=True)
        print(name + ':\n' + out.stdout + out.stderr)
        ok = ok and 'RESULT PASS' in out.stdout
    print('stokes_check:', 'all passed' if ok else 'FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'eval':
        evaluate()
    else:
        sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'stokes_runs'))
