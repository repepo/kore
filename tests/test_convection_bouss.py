#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from koretest import KoreTest

class TestConvectionBouss(KoreTest):

    def test_jones(self):
        """ Test the onset of convection from Jones et al. 2000
        """
        self.dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jones2000')
        os.system('cp -r %s/bin %s/' % (self.kore_dir,self.dir))

        os.chdir(self.dir)
        os.system('cp params.jones ./bin/parameters.py')
        os.system('./find_Rac.py')
        datRef = np.loadtxt('reference.jones')
        datTmp = np.loadtxt('critical_params.dat')
        os.chdir(self.startDir)
        self.cleanDir(self.startDir, self.dir)

        return np.testing.assert_allclose(datRef, datTmp, rtol=self.precision,
                                   atol=1e-20)

    def test_dormy(self):
        """ Test the onset of convection from Dormy et al. 2004
        """
        self.dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dormy2004')
        os.system('cp -r %s/bin %s/' % (self.kore_dir,self.dir))

        os.chdir(self.dir)
        os.system('cp params.dormy04 ./bin/parameters.py')
        os.system('./find_Rac.py')
        datRef = np.loadtxt('reference.dormy04')
        datTmp = np.loadtxt('critical_params.dat')
        os.chdir(self.startDir)
        self.cleanDir(self.startDir, self.dir)

        return np.testing.assert_allclose(datRef, datTmp, rtol=self.precision,
                                   atol=1e-20)