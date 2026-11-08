#!/usr/bin/env python3
'''
usage: decay_check.py [WORKDIR]   (default: tests/decay/runs)

Runs without the flow (hydro = 0) against exact results. Without the flow the field only diffuses: in a fluid of uniform
diffusivity eta each mode is a spherical Bessel function j_l(kr), y_l(kr), and it decays at the rate eta k^2, where k
makes the mode meet its boundary conditions. Every eigenvalue of these runs is such a rate (Im lambda = 0):
  shell, insulating walls      poloidal l  j_{l-1}(k) y_{l+1}(k ri) - y_{l-1}(k) j_{l+1}(k ri) = 0  (r f' + (l+1) f = 0 at the
                                           outer wall, r f' - l f = 0 at the inner one: the potential fields outside)
                               toroidal l  j_l(k) y_l(k ri) - y_l(k) j_l(k ri) = 0  (g = 0 at both walls)
  conducting inner core        the field inside is j_l(k sqrt(s) r) (s = sigma_i2o, mu_i2o = 1); f, f' and g, (r g)'/sigma
                               continuous at ri (a 3 x 3 determinant)
  full sphere                  poloidal j_{l-1}(k) = 0, toroidal j_l(k) = 0
The entropy of 'Boussinesq' with a uniform diffusivity decays in the same way at the rate ThermaD k^2: s = 0 at both walls
is the toroidal condition above, s' = 0 at the outer wall gives j_l'(k) y_l(k ri) - y_l'(k) j_l(k ri) = 0. Two more checks:
with a differential rotation, the field's modes of hydro = 0 (diffusion and induction by the rotation) are those of the
same run with the flow and a vanishing B0 (cnorm 1e-9: the field and the flow then decouple); and just_induction.py with
solve_ind.py give the field of a mode of a run whose Lorentz force vanishes (Hendrik 1e-30), from its flow and eigenvalue.
Each run's spin_doctor budgets (resid𝐛, resid𝐄, resid𝑠) must close too, to 1e-9. FIX (2026-11-08): they were held to
1e-10, which resid𝐛 of the conducting inner core met on one machine and not on another (8e-11, 2e-10): it takes the
magnetic diffusion from the integral of 𝐛⋅∇²𝐛, the third derivative of the poloidal scalar, whose round-off grows with
N (4e-10 at N 64, with the decay rates exact to 5e-14). Exit code 1 if an eigenvalue or a field is off by more than
1e-10, or a budget by more than 1e-9.
'''
import os, sys, subprocess
import numpy as np
from scipy.special import spherical_jn as J, spherical_yn as Y
from scipy.optimize import brentq

T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(T, '..', 'tidal'))
import mk_run   # noqa: E402  (the run builder of tests/tidal: 'Boussinesq', ricb 0.35, N 96, NCPUS ranks)

W = sys.argv[1] if len(sys.argv) > 1 else os.path.join(T, 'runs')
os.makedirs(W, exist_ok=True)
RI = 0.35


def roots(f, n, kmax=40.):
    '''the first n positive roots of f(k), bracketed on a fine grid'''
    ks = np.linspace(0.05, kmax, 40000)
    v = np.array([f(k) for k in ks])
    out = []
    for a, b, fa, fb in zip(ks[:-1], ks[1:], v[:-1], v[1:]):
        if np.sign(fa) != np.sign(fb):
            r = brentq(f, a, b, xtol=1e-15, rtol=1e-15)
            if abs(f(r)) < 1e-8*max(abs(fa), abs(fb)):   # a root, not a pole of y_l
                out.append(r)
        if len(out) == n:
            break
    return np.array(out)

def pol_ins(l, n):
    return roots(lambda k: J(l-1, k)*Y(l+1, k*RI) - Y(l-1, k)*J(l+1, k*RI), n)

def tor_ins(l, n):
    return roots(lambda k: J(l, k)*Y(l, k*RI) - Y(l, k)*J(l, k*RI), n)

def pol_cic(l, n, s):
    def f(k):
        ki = k*np.sqrt(s)
        return np.linalg.det([[J(l-1, k), Y(l-1, k), 0],
                              [J(l, k*RI), Y(l, k*RI), -J(l, ki*RI)],
                              [k*J(l, k*RI, True), k*Y(l, k*RI, True), -ki*J(l, ki*RI, True)]])
    return roots(f, n)

def tor_cic(l, n, s):
    def f(k):
        ki = k*np.sqrt(s)
        d = lambda F, kk: F(l, kk*RI) + kk*RI*F(l, kk*RI, True)   # (r F(kk r))' at ri
        return np.linalg.det([[J(l, k), Y(l, k), 0],
                              [J(l, k*RI), Y(l, k*RI), -J(l, ki*RI)],
                              [d(J, k), d(Y, k), -d(J, ki)/s]])
    return roots(f, n)

def flux_out(l, n):   # s' = 0 at the outer wall, s = 0 at the inner one
    return roots(lambda k: J(l, k, True)*Y(l, k*RI) - Y(l, k, True)*J(l, k*RI), n)


MAG = "par.hydro = 0; par.magnetic = 1; par.B0 = 'axial'; par.cnorm = 1; par.Hendrik = 1.0; par.MagnetD = 1.0; par.N = 48; g = 0.34"
THM = ("par.hydro = 0; par.thermal = 1; par.stratification = 'linear'; par.ThermaD = 1.0; par.bci_thermal = 0; par.N = 48; "
       "g = 0.34")
# name, changes, the exact rates k^2 (m = 0, symm = 1: the field has the poloidal l = 1, 3, the toroidal l = 2; m = 1,
# symm = 1: the entropy has l = 1, 3)
cases = [
    ('insulating', MAG + "; par.m = 0; par.symm = 1; par.rtau = -10.0; par.itau = 0.0; par.nev = 6",
     lambda: np.r_[pol_ins(1, 2), pol_ins(3, 1), tor_ins(2, 1)]**2),
    ('conducting_ic', MAG + "; par.m = 0; par.symm = 1; par.innercore = 'conducting, Chebys'; par.sigma_i2o = 0.3; "
                      "par.N_cic = 32; par.rtau = -10.0; par.itau = 0.0; par.nev = 6",
     lambda: np.r_[pol_cic(1, 2, 0.3), pol_cic(3, 1, 0.3), tor_cic(2, 1, 0.3)]**2),
    ('full_sphere', MAG + "; par.m = 0; par.symm = 1; par.ricb = 0; par.rtau = -10.0; par.itau = 0.0; par.nev = 6",
     lambda: np.r_[np.pi, 2*np.pi, roots(lambda k: J(2, k), 1)]**2),   # pol l = 1 (j_0), pol l = 3 = tor l = 2 (j_2)
    ('entropy', THM + "; par.m = 1; par.symm = 1; par.bco_thermal = 0; par.rtau = -25.0; par.itau = 0.0; par.nev = 4",
     lambda: np.r_[tor_ins(1, 2), tor_ins(3, 1)]**2),
    ('entropy_flux', THM + "; par.m = 1; par.symm = 1; par.bco_thermal = 1; par.rtau = -10.0; par.itau = 0.0; par.nev = 4",
     lambda: np.r_[flux_out(1, 2), flux_out(3, 1)]**2),
]


def eig(d):
    e = np.loadtxt(os.path.join(d, 'eigenvalues0.dat')).reshape(-1, 2)
    return e[:, 0] + 1j*e[:, 1]

def budgets(d):
    '''the largest of resid𝐛, resid𝐄 (magnetic.dat columns 4, 11) and resid𝑠 (thermal.dat column 4)'''
    out = 0.
    for f, cols in (('magnetic.dat', (3, 10)), ('thermal.dat', (3,))):
        p = os.path.join(d, f)
        if os.path.isfile(p):
            out = max(out, np.nanmax(np.loadtxt(p, ndmin=2)[:, cols]))
    return out


fail = 0
def report(name, err, budget):
    global fail
    ok = err < 1e-10 and budget < 1e-9
    fail |= not ok
    print('%-15s %s  (largest error %.1e, largest budget residual %.1e)' % (name, 'ok' if ok else 'FAIL', err, budget))

for name, extra, exact in cases:
    d = os.path.join(W, name)
    mk_run.build(d, extra)
    lam, ref = eig(d), -exact()
    report(name, max(np.min(abs(lam - r))/abs(r) for r in ref), budgets(d))

# the induction by a differential rotation: hydro = 0 against hydro = 1 with a vanishing B0 (cnorm 1e-9)
DR = ("par.magnetic = 1; par.B0 = 'axial'; par.Hendrik = 1.0; par.MagnetD = 1.0; par.m = 1; par.symm = 1; par.diff_rot = 1; "
      "par.diff_rot_type = 'conical'; par.diff_rot_ampl = 2.0; par.N = 48; g = 0.34")
d0, d1 = os.path.join(W, 'diffrot_hydro0'), os.path.join(W, 'diffrot_hydro1')
mk_run.build(d0, DR + "; par.hydro = 0; par.cnorm = 1; par.rtau = -20.0; par.itau = 0.0; par.nev = 3")
l0 = eig(d0)
err = 0.
for z in l0[:2]:
    mk_run.build(d1, DR + "; par.hydro = 1; par.cnorm = 1e-9; par.rtau = %r; par.itau = %r; par.nev = 1" % (z.real, z.imag))
    err = max(err, np.min(abs(eig(d1) - z))/abs(z))
report('diffrot', err, budgets(d0))

# kinematic induction: just_induction.py and solve_ind.py against the field of a mode without Lorentz force (Hendrik 1e-30)
d = os.path.join(W, 'induction')
mk_run.build(d, "par.magnetic = 1; par.B0 = 'axial'; par.cnorm = 1; par.Hendrik = 1e-30; par.MagnetD = 1e-2; par.m = 1; "
                "par.symm = 1; par.rtau = 0.0; par.itau = 0.5; par.nev = 2; par.N = 48; g = 0.34")
b = lambda: np.loadtxt(os.path.join(d, 'real_magnetic.field')) + 1j*np.loadtxt(os.path.join(d, 'imag_magnetic.field'))
beig, z = b(), eig(d)[0]
py, n = sys.executable, str(mk_run.NCPUS)
for cmd in ([py, 'bin/just_induction.py', repr(complex(z))], mk_run.MPIEXEC.split() + ['-n', n, py, 'bin/solve_ind.py']):
    p = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
    if p.returncode:
        sys.exit('%s: %s failed\n%s' % (d, cmd[-1], (p.stdout + p.stderr)[-3000:]))
report('induction', abs(b()[:, 0] - beig[:, 0]).max()/abs(beig[:, 0]).max(), 0.)

print('decay_check: ' + ('all cases ok' if not fail else 'FAILED'))
sys.exit(fail)
