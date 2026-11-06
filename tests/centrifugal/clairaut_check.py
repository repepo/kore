#!/usr/bin/env python3
'''
clairaut_check.py: the flattening of centrifugal_shape 'clairaut' against known results (NEW 2026-11-05).

  homogeneous   centrifugal.radau with g ∝ x: η = 0, f = 5q/4 at every radius (Maclaurin)
  polytrope     g of the polytrope of index 1 (ρ ∝ sin(πx)/(πx)): f/q = (1 + k2)/2 = 7.5/π² at the surface (k2 = 15/π² - 1)
  Wu 2005       a run with structure 'Wu 2005' (ρ = (1 - x²)², r_out 0.99): the shape of the run (shape_functions, from the
                structure's g through Radau's form) against Clairaut's second-order equation integrated here from ρ itself,
                    f'' + (6ρ/(x ρ̄)) f' + (6/x²)(ρ/ρ̄ - 1) f = 0,   x f' + 2f = (5/2) q at x = 1,   ρ̄ = (3/x³) ∫ ρ x² dx,
                at the surface and at x = 0.3, 0.6, 0.99; the profile codes ce1 = r ε', ce2 = r (r ε')' against finite
                differences of ε, and the Chebyshev fits of the codes against the shapes
  Sun           structure 'model S hydrostatic' (r_out 0.985, the mass above the wall neglected): f/q at the wall, printed
                with the check that it lies between 1/2 (a point mass) and 0.53 (k2 of the Sun is about 0.03)

Usage: python3 clairaut_check.py   (builds the runs cl_wu and cl_sun with mk_run.py in the current directory; under a minute)
'''
import os, sys, subprocess
import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
fail = []

def verdict(name, ok, msg):
    print('%-12s %s  %s' % (name + ':', 'ok  ' if ok else 'FAIL', msg))
    if not ok:
        fail.append(name)

def run_python(d, code):
    '''runs code in the run directory d (its own bin/ and parameters), returns its stdout'''
    p = subprocess.run([sys.executable, '-c', "import sys; sys.path.insert(0, 'bin')\n" + code], cwd=d, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-2000:], p.stderr[-3000:]); sys.exit('clairaut_check.py: the script failed in %s' % d)
    return p.stdout.strip().splitlines()[-1]   # (the last line: the structures print notes when they are set up)

def build(name, extra):
    p = subprocess.run([sys.executable, os.path.join(HERE, 'mk_run.py'), name, extra], capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-3000:]); sys.exit('clairaut_check.py: %s could not be built' % name)

# ------------------------------------------------------------------------------------------- homogeneous and polytrope
sys.path.insert(0, os.path.join(HERE, '..', '..', 'bin'))
import centrifugal as cf
eta, lnf, rho = cf.radau(lambda x: 2.0*x, 1.0)
et = float(eta(np.array([1.0]))[0])
emax = np.abs(eta(np.linspace(0, 1, 101))).max()
verdict('homogeneous', abs(2.5/(et + 2) - 1.25) < 1e-10 and emax < 1e-10, 'f/q = %.12f (5/4), max |η| = %.1e' % (2.5/(et + 2), emax))
g1 = lambda x: (np.sin(np.pi*x) - np.pi*x*np.cos(np.pi*x))/x**2
eta, lnf, rho = cf.radau(g1, 1.0)
fq = 2.5/(float(eta(np.array([1.0]))[0]) + 2)
verdict('polytrope', abs(fq - 7.5/np.pi**2) < 1e-9, 'n = 1: f/q = %.12f, 7.5/π² = %.12f' % (fq, 7.5/np.pi**2))

# ------------------------------------------------------------------------------------------------------------ Wu 2005
q = 0.089
# FIX (2026-11-06, phase 2): rpower_v = 4, which a viscous, deformed full sphere with m != 0 needs now (background.check_centrifugal)
build('cl_wu', "par.structure = 'Wu 2005'; par.r_out = 0.99; par.ricb = 0; par.ViscosD = 1e-3; par.centrifugal_shape = 'clairaut'; "
               "par.centrifugal_f = None; par.centrifugal_q = %r; par.N = 48; par.rpower_v = 4" % q)
out = run_python('cl_wu', '''
import numpy as np, centrifugal as cf
from parameters import par
S = cf.shape_functions(); P = cf.profile_functions()
x = np.array([0.3, 0.6, 0.99])
a = [S['f_top']/par.centrifugal_q, S['eta_top'], *(S['f'](x/par.r_out)/par.centrifugal_q)]
r = np.linspace(0.05, 1, 40); h = 1e-5
fd1 = np.abs(S['h'](r) - r*(S['e'](r + h) - S['e'](r - h))/(2*h)).max()
fd2 = np.abs(S['k'](r) - r*(S['h'](r + h) - S['h'](r - h))/(2*h)).max()
fit = max(np.abs(P['ce0'](r, 0) - S['e'](r)).max(), np.abs(P['ce1'](r, 0) - S['h'](r)).max(), np.abs(P['ce2'](r, 0) - S['k'](r)).max())
print(*a, fd1, fd2, fit, np.abs(S['e'](r)).max())
''').split()
v = [float(t) for t in out]
fq_k, eta_k, fx_k = v[0], v[1], v[2:5]
fd1, fd2, fit, emax = v[5:9]
# Clairaut's second-order equation from ρ = (1 - x²)²
rho = lambda x: (1 - x*x)**2
rbar = lambda x: 3*(x**3/3 - 2*x**5/5 + x**7/7)/x**3
def rhs(x, y):
    return [y[1], -(6*rho(x)/(rbar(x)*x))*y[1] - (6/x**2)*(rho(x)/rbar(x) - 1)*y[0]]
x0 = 1e-5
sol = solve_ivp(rhs, (x0, 1.0), [1.0, -(6/7)*(rho(x0)/rbar(x0) - 1)/x0], method='DOP853', rtol=1e-13, atol=1e-15, dense_output=True)
f1, fp1 = sol.y[:, -1]
c = 2.5/(fp1 + 2*f1)
fx = [c*sol.sol(x)[0] for x in (0.3, 0.6, 0.99)]
dev = max(abs(fq_k - c*f1), abs(eta_k - fp1/f1), *[abs(a - b) for a, b in zip(fx_k, fx)])
verdict('Wu 2005', dev < 1e-8, "f/q = %.10f at the surface (Clairaut from ρ: %.10f), a f'/f = %.8f, f/q at x = 0.3, 0.6, 0.99: "
        "%s; max deviation %.1e" % (fq_k, c*f1, eta_k, ', '.join('%.8f' % a for a in fx_k), dev))
verdict('codes', fd1 < 1e-6*emax and fd2 < 1e-6*emax and fit < 1e-8*emax,
        "r ε' and r (r ε')' against finite differences %.1e, %.1e (max |ε| %.1e); Chebyshev fits of ce0, ce1, ce2 (N 48) %.1e"
        % (fd1, fd2, emax, fit))

# ---------------------------------------------------------------------------------------------------------------- Sun
build('cl_sun', "par.structure = 'model S hydrostatic'; par.r_out = 0.985; par.ricb = 0; par.ViscosD = 1e-3; par.N = 48; "
                "par.centrifugal_shape = 'clairaut'; par.centrifugal_f = None; par.centrifugal_q = 2e-5; par.rpower_v = 4")
v = [float(t) for t in run_python('cl_sun', '''
import numpy as np, centrifugal as cf
from parameters import par
S = cf.shape_functions()
print(S['f_top']/(par.centrifugal_q*cf.x_top()**3), S['eta_top'], S['f'](np.array([0.7/par.r_out]))[0]/S['f_top'])
''').split()]
verdict('Sun', 0.5 < v[0] < 0.53, "model S hydrostatic: f/q = %.6f at the wall x = 0.985 (a f'/f = %.4f; f(0.7)/f(wall) = %.4f)"
        % tuple(v))

print('clairaut_check.py: %s' % ('FAIL ' + ', '.join(fail) if fail else 'ok'))
sys.exit(1 if fail else 0)
