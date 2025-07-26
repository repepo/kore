#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from koretest import KoreTest

class TestSpinover(KoreTest):

    def test_spinover(self):
        """ Test the spinover mode at Ek=1e-3
        """
        self.dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spinover')
        os.system('cp -r %s/bin %s/' % (self.kore_dir,self.dir))

        os.chdir(self.dir)
        os.system('cp -r params.spinover ./bin/parameters.py')
        os.system("./bin/submatrices.py %d > /dev/null" % self.ncpus)
        os.system("mpiexec -n %d ./bin/assemble.py > /dev/null" % self.ncpus)
        os.system("mpiexec -n %d ./bin/solve.py %s" % (self.ncpus, self.solve_opts))
        datRef = np.loadtxt('reference.eig')
        datTmp = np.loadtxt('eigenvalues0.dat')
        idx = np.argmax(datTmp[:,0])
        datTmp = datTmp[idx,:]
        os.chdir(self.startDir)
        self.cleanDir(self.startDir, self.dir)

        return np.testing.assert_allclose(datRef, datTmp, rtol=self.precision,
                                   atol=1e-20)