#!/usr/bin/env python3
'''
Finds the critical Rayleigh number for the onset of convection at a given m, i.e. the Ra_gap at which the
growth rate of the least stable mode crosses zero. Used by test_convection_bouss.py and
test_thermocompositional.py.

bin/parameters.py in the case folder must have a line 'Ra_gap = ...', from which par.Ra is computed. Kore
is run from the case folder; Ra_gap is rewritten before each solve. find_Rac can also vary other lines
of bin/parameters.py (see its argument update), e.g. a thermal and a compositional Rayleigh number together.

Run from a case folder that already has bin/ (with the case parameter file as bin/parameters.py):
> ../find_Rac.py Ramin m [ncpus]
'''

import os
import re
import sys
import subprocess
import importlib.util
import numpy as np
from scipy.optimize import brentq

# Added to petsc_opts in parameters.py. The references were computed on main with -eps_true_residual as
# well, which does not converge here together with prescale = 1 (tol = 1e-15)
opts = '-eps_balance twoside'

# Format of the reference files: Ek, ricb, Rac, m, omega_c
fmt = '%.3e %.2f %.5e %d %.5e'



def set_param(name, value):
    '''
    Rewrites the line 'name = ...' of bin/parameters.py as 'name = value'
    '''
    with open('bin/parameters.py') as f:
        s = f.read()
    s, n = re.subn(r'^%s\s*=.*$' % re.escape(name), lambda _: '%s = %r' % (name, value), s, flags=re.M)
    assert n == 1, "bin/parameters.py needs exactly one '%s = ...' line" % name
    with open('bin/parameters.py', 'w') as f:
        f.write(s)



def solve(values, ncpus, quiet=True):
    '''
    Sets the parameter lines in the dictionary values, assembles and solves, and returns the eigenvalue with
    the largest growth rate as [sigma, omega]
    '''
    out = subprocess.DEVNULL if quiet else None
    for name, value in values.items():
        set_param(name, float(value))
    subprocess.run('mpiexec -n %d ./bin/assemble.py' % ncpus, shell=True, check=True, stdout=out)
    subprocess.run('mpiexec -n %d ./bin/solve_nopp.py %s' % (ncpus, opts), shell=True, check=True, stdout=out)
    eig = np.loadtxt('eigenvalues0.dat').reshape(-1, 2)
    os.remove('eigenvalues0.dat')
    return eig[np.argmax(eig[:, 0])]



def bracket_brentq(f, x1, dx=0.01, tol=1e-6, maxiter=200):
    '''
    Steps from x1 in the direction of the root by dx until the sign of f changes, then Brent's method.
    Copied from SINGE.
    '''
    y1 = f(x1)
    dx = -abs(dx) if y1 > 0 else abs(dx)
    x2 = x1
    while True:
        x2 += dx
        y2 = f(x2)
        if y2*y1 < 0:
            break
        x1, y1 = x2, y2
    return brentq(f, x1, x2, maxiter=maxiter, xtol=tol, rtol=tol)



def find_Rac(Ramin, m, ncpus, update=lambda Ra: {'Ra_gap': Ra}, log=True, dx=0.01):
    '''
    Returns [Ek, ricb, Rac, m, omega_c] for the parameters in bin/parameters.py, searching from Ramin
    (Rac and Ramin are gap Rayleigh numbers). m must already be set in bin/parameters.py.
    update(Ra) gives the parameter lines to set for a given Ra. The search steps by dx in log10(Ra), or in Ra
    if log = False (for an Ra that can be negative).
    '''
    # loaded from its path, not imported, so that a parameters module cached from another case is not used
    spec = importlib.util.spec_from_file_location('parameters', os.path.join('bin', 'parameters.py'))
    params = importlib.util.module_from_spec(spec)
    sys.path.insert(0, 'bin')  # for params_default
    spec.loader.exec_module(params)
    sys.path.pop(0)
    par = params.par
    assert par.m == m, 'm = %d in bin/parameters.py, expected %d' % (par.m, m)

    subprocess.run('./bin/submatrices.py %d' % ncpus, shell=True, check=True, stdout=subprocess.DEVNULL)

    Ra = (lambda x: 10**x) if log else (lambda x: x)
    cache = {}
    def sigma(x):  # growth rate of the least stable mode
        if x not in cache:
            cache[x] = solve(update(Ra(x)), ncpus)[0]
            print('Ra = % .6e   sigma = % .6e' % (Ra(x), cache[x]), flush=True)
        return cache[x]

    Rac = Ra(bracket_brentq(sigma, np.log10(Ramin) if log else Ramin, dx=dx))
    sigma_c, omega_c = solve(update(Rac), ncpus, quiet=False)

    return [par.Ek, par.ricb, Rac, m, omega_c]



if __name__ == '__main__':
    Ramin, m = float(sys.argv[1]), int(sys.argv[2])
    ncpus = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    print(fmt % tuple(find_Rac(Ramin, m, ncpus)))
