import unittest
import numpy as np
import os
from glob import glob
import shutil
import time

def readData(file):
    return np.loadtxt(file)


class spinoverTest(unittest.TestCase):

    def __init__(self, testName, ncpus, solve_opts, dir, precision=1e-8):
        super(spinoverTest, self).__init__(testName)

        self.dir = dir
        self.precision = precision
        self.ncpus = ncpus
        self.solve_opts = solve_opts
        self.startDir = os.getcwd()
        self.description = "Spin-over mode at Ek=1e-3"
        self.aux_files = ['submatrices_new.py',
                          'assemble.py',
                          'operators_new.py',
                          'utils_pp.py',
                          'utils.py',
                          'solve_nopp.py',
                          'bc_variables.py',
                          'radial_profiles.py'
                          ]

    def setUp(self):
        # Cleaning when entering
        print('\nDirectory   :           %s' % self.dir)
        print('Description :           %s' % self.description)
        self.startTime = time.time()
        self.cleanDir(self.dir)
        os.chdir(self.dir)
        for file in self.aux_files:
            os.system('cp ../../bin/%s .' %file)
        os.system("./submatrices_new.py %d > /dev/null" %(self.ncpus))
        os.system("mpiexec -n %d ./assemble.py > /dev/null" %(self.ncpus))
        os.system("mpiexec -n %d ./solve_nopp.py %s > /dev/null" %(self.ncpus,self.solve_opts))

    def list2reason(self, exc_list):
        if exc_list and exc_list[-1][0] is self:
            return exc_list[-1][1]

    def cleanDir(self,dir):
        for f in glob("%s/*.mtx" %dir):
            os.remove(f)
        for f in glob("%s/*.npz" %dir):
            os.remove(f)
        for f in glob("%s/*.field" %dir):
            os.remove(f)
        for f in glob("%s/*.dat" %dir):
            if f != "%s/reference_eig.dat" %dir:
                os.remove(f)
        for file in self.aux_files:
            if os.path.exists("%s/%s" %(dir,file)):
                os.remove("%s/%s" %(dir,file))

        if os.path.exists("%s/__pycache__" %dir):
            shutil.rmtree("%s/__pycache__" %dir)

    def tearDown(self):
        # Cleaning when leaving
        os.chdir(self.startDir)
        self.cleanDir(self.dir)

        t = time.time()-self.startTime
        st = time.strftime("%M:%S", time.gmtime(t))
        print('Time used   :                            %s' % st)

        if hasattr(self, '_outcome'): # python 3.4+
            if hasattr(self._outcome, 'errors'):  # python 3.4-3.10
                result = self.defaultTestResult()
                self._feedErrorsToResult(result, self._outcome.errors)
            else:  # python 3.11+
                result = self._outcome.result
        else:  # python 2.7-3.3
            result = getattr(self, '_outcomeForDoCleanups',
                             self._resultForDoCleanups)

        error = self.list2reason(result.errors)
        failure = self.list2reason(result.failures)
        ok = not error and not failure

        if ok:
            print('Validating results..                     OK')
        else:
            if error:
                print('Validating results..                     ERROR!')
                print('\n')
                print(result.errors[-1][-1])
            if failure:
                print('Validating results..                     FAIL!')
                print('\n')
                print(result.failures[-1][-1])

    def outputFileDiff(self):
        datRef = readData('%s/reference_eig.dat' % self.dir)
        datTmp = readData('%s/eigenvalues0.dat' % self.dir)
        np.testing.assert_allclose(datRef, datTmp, rtol=self.precision,
                                   atol=1e-20)
