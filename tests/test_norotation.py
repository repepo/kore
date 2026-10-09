#!/usr/bin/env python3

import os
import numpy as np
from scipy.special import spherical_jn
from scipy.optimize import brentq
from koretest import KoreTest
from find_Rac import set_param



def k_tor(l, bco):
    '''
    Smallest k of the toroidal viscous decay mode of degree l in a full sphere, T = j_l(k*r), decay rate k**2
    (viscous time units). No-slip (bco = 1): j_l(k) = 0. Stress-free (bco = 0): d(T/r)/dr = 0 at r = 1,
    i.e. k*j_l'(k) = j_l(k).
    '''
    if bco == 1:
        f = lambda k: spherical_jn(l, k)
    else:
        f = lambda k: k*spherical_jn(l, k, derivative=True) - spherical_jn(l, k)
    k = np.linspace(0.5, 20, 2000)
    i = np.where(np.sign(f(k[:-1])) != np.sign(f(k[1:])))[0][0]
    return brentq(f, k[i], k[i+1], xtol=1e-14, rtol=1e-15)



class TestNoRotation(KoreTest):

    def solve(self, m, symm, settings={}):
        '''
        Solves with rotation = 0 at m and symm (and the other parameter lines in settings), with l = m ... m+3.
        Returns the eigenvalues found around the target.
        '''
        set_param('par.m', m)
        set_param('par.symm', symm)
        for name, value in settings.items():
            set_param(name, value)
        self.run('./bin/submatrices.py %d' % self.ncpus, quiet=True)
        self.run('mpiexec -n %d ./bin/assemble.py' % self.ncpus, quiet=True)
        self.run('mpiexec -n %d ./bin/solve_nopp.py' % self.ncpus, quiet=True)
        eig = np.loadtxt(os.path.join(self.dir, 'eigenvalues0.dat')).reshape(-1, 2)
        return eig[:, 0] + 1j*eig[:, 1]


    def test_toroidal_decay(self):
        '''
        Without rotation the toroidal flow decouples and decays freely. For l = 2 and 3, no-slip and stress-free,
        the decay rate must match the analytic one, -k**2 (see k_tor), and be the same for m = 0, 1 and 2
        (spherical symmetry). symm is chosen so that l is a toroidal degree.
        '''
        self.prepare('norotation', 'params.norotation')
        cwd = os.getcwd()
        try:
            os.chdir(self.dir)
            for bco in (1, 0):
                for l in (2, 3):
                    lam = -k_tor(l, bco)**2
                    found = []
                    for m in (0, 1, 2):
                        symm = 1 if (l - m) % 2 == 1 else -1  # toroidal l - m is odd for symm = 1
                        eig = self.solve(m, symm, {'par.bco': bco, 'par.rtau': 1.001*lam})
                        found.append(eig[np.argmin(np.abs(eig - lam))])
                    found = np.array(found)
                    np.testing.assert_allclose(found, lam, rtol=1e-10, atol=1e-10, err_msg='l=%d bco=%d' % (l, bco))
                    np.testing.assert_allclose(found, found[0], rtol=1e-12, atol=1e-12, err_msg='l=%d bco=%d' % (l, bco))
        finally:
            os.chdir(cwd)
            self.clean()
