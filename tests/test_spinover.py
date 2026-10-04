#!/usr/bin/env python3

import os
import numpy as np
from koretest import KoreTest



class TestSpinover(KoreTest):

    def test_spinover(self):
        '''
        Spin-over mode at Ek=1e-3 (hydrodynamic only, no-slip, ricb=0.35). The least damped of the
        eigenvalues found around the target must match the reference.
        '''
        self.prepare('spinover', 'params.spinover')
        try:
            self.run('./bin/submatrices.py %d' % self.ncpus, quiet=True)
            self.run('mpiexec -n %d ./bin/assemble.py' % self.ncpus, quiet=True)
            self.run('mpiexec -n %d ./bin/solve_nopp.py' % self.ncpus)
            eig = np.loadtxt(os.path.join(self.dir, 'eigenvalues0.dat')).reshape(-1, 2)
            ref = np.loadtxt(os.path.join(self.dir, 'reference.eig'))
        finally:
            self.clean()

        leading = eig[np.argmax(eig[:, 0])]  # largest real part, i.e. the least damped one
        np.testing.assert_allclose(leading, ref, rtol=self.precision, atol=1e-20)
