#!/usr/bin/env python3

import os
import re
import numpy as np
from koretest import KoreTest



class TestGravitoInertial(KoreTest):

    precision = 1e-6  # tolerance on |lambda - reference|/|reference| (run-to-run differences are ~1e-7)

    # Dintrans, Rieutord & Valdetarro (2022), figure 4: target values rtau, itau with the buoyancy frequency BV2, and
    # how close kore gets at the resolution of this test (N = 150, lmax = 311; it is 3e-3, 3e-3 and 6e-4 respectively)
    dintrans = [(2*-6.088e-3, 2*0.550, (2*0.2)**2, 3e-3),
                (2*-3.854e-3, 2*0.755, (2*1.0)**2, 3e-3),
                (2*-5.190e-3, 2*1.130, (2*2.5)**2, 6e-4)]

    def set_target(self, rtau, itau, BV2):
        '''
        Overwrites rtau, itau and BV2 in the copy of bin/parameters.py made by prepare()
        '''
        path = os.path.join(self.dir, 'bin', 'parameters.py')
        with open(path) as f:
            text = f.read()
        for name, value in [('rtau', rtau), ('itau', itau), ('BV2', BV2)]:
            text, n = re.subn(r'^par\.%s\s*=.*$' % name, 'par.%s = %r' % (name, value), text, flags=re.M)
            assert n == 1, 'par.%s not found exactly once in parameters.py' % name
        with open(path, 'w') as f:
            f.write(text)

    def test_dintrans99(self):
        '''
        H1, H2, E1 gravito-inertial mode at Ek=2e-7 (stress-free, fixed-temperature, ricb=0.35). For each entry of
        self.dintrans the matrices are rebuilt with the corresponding Brunt-Vaisala frequency and the system is solved
        for the corresponding target eigenvalue. Each eigenvalue must match the reference gravito-inertial waves
        computed with kore at this resolution (one row per mode in reference.eig)
        '''
        self.prepare('dintrans1999', 'params.dintrans99')
        eigfile = os.path.join(self.dir, 'eigenvalues0.dat')
        found = []
        try:
            for rtau, itau, BV2, _ in self.dintrans:
                self.set_target(rtau, itau, BV2)
                if os.path.exists(eigfile):
                    os.remove(eigfile)  # never pick up the result of the previous mode
                self.run('./bin/submatrices.py %d' % self.ncpus, quiet=True)
                self.run('mpiexec -n %d ./bin/assemble.py' % self.ncpus, quiet=True)
                self.run('mpiexec -n %d ./bin/solve_nopp.py' % self.ncpus)
                eig = np.loadtxt(eigfile).reshape(-1, 2)
                found.append(eig[:, 0] + 1j*eig[:, 1])
            ref = np.loadtxt(os.path.join(self.dir, 'reference.dintrans99')).reshape(-1, 2)
        finally:
            self.clean()

        for lam, r, (rtau, itau, _, tol) in zip(found, ref, self.dintrans):
            r = complex(*r)
            z = lam[np.argmin(np.abs(lam - r))]  # the eigenvalue closest to the reference
            # relative to |lambda|: the damping is ~1/130 of the frequency, so it is not compared on its own
            print(abs(z - r) / abs(r))
            din = rtau + 1j*itau
            print(abs(z - din) / abs(din))
            assert abs(z - r)/abs(r) < self.precision, (z, r)
            assert abs(z - din)/abs(din) < tol, (z, din)