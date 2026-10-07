#!/usr/bin/env python3

import os
import numpy as np
from koretest import KoreTest



class TestTorsionalS2(KoreTest):

    precision = 1e-7  # tolerance on |lambda - reference|/|reference| (run-to-run differences are ~1e-8)

    # Luo & Jackson (2022), table 1, L = 350: the first two torsional modes of S2, and how close kore gets
    # at the resolution of this test (N = 128, lmax = 127; it is 7.3e-7 and 4.7e-6 respectively)
    luo_jackson = [ (-0.0065952461 + 1.0335959942j, 2e-6),    # mode 1
                    (-0.0141833345 + 1.9095123626j, 1e-5) ]   # mode 2

    def test_torsional_s2(self):
        '''
        Torsional modes 1 and 2 of the quadrupolar background field S2 of Luo & Jackson (2022, Proc. R. Soc. A
        478, 20210982): inviscid full sphere, insulating boundary, Le = 1e-4, Lu = 2e4, m = 0, at N = 128,
        lmax = 127. The matrices are built once and solved twice: with the target in params.torsional_s2
        (mode 2) and with -eps_target near mode 1, since a mode far from the target is less accurate.
        Each eigenvalue must match the reference computed with kore at this resolution (one row per mode
        in reference.eig) and Luo & Jackson's converged value.
        '''
        self.prepare('torsional_s2', 'params.torsional_s2')
        found = []
        try:
            self.run('./bin/submatrices.py %d' % self.ncpus, quiet=True)
            self.run('mpiexec -n %d ./bin/assemble.py' % self.ncpus, quiet=True)
            for target in ('-eps_target -0.0066+1.0336i', ''):   # mode 1, then mode 2 (target in params)
                self.run('mpiexec -n %d ./bin/solve_nopp.py %s' % (self.ncpus, target))
                eig = np.loadtxt(os.path.join(self.dir, 'eigenvalues0.dat')).reshape(-1, 2)
                found.append(eig[:, 0] + 1j*eig[:, 1])
            ref = np.loadtxt(os.path.join(self.dir, 'reference.eig')).reshape(-1, 2)
        finally:
            self.clean()

        for lam, r, (lj, tol) in zip(found, ref, self.luo_jackson):
            r = complex(*r)
            z = lam[np.argmin(np.abs(lam - r))]  # the eigenvalue closest to the reference
            # relative to |lambda|: the damping is ~1/130 of the frequency, so it is not compared on its own
            assert abs(z - r)/abs(r) < self.precision, (z, r)
            assert abs(z - lj)/abs(lj) < tol, (z, lj)
