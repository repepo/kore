#!/usr/bin/env python3

import os
import numpy as np
from koretest import KoreTest
from find_Rac import find_Rac, fmt



class TestConvectionBouss(KoreTest):

    # Rac comes from a root search on log10(Ra) with tolerance 1e-6; the result is rounded as in the
    # reference file (6 digits) before comparing
    precision = 1e-5

    def onset(self, case, params, Ramin, m):
        '''
        Critical gap Rayleigh number and drift frequency at m, compared with reference.<case>, which holds
        Ek, ricb, Rac, m, omega_c
        '''
        self.prepare(case, params)
        cwd = os.getcwd()
        try:
            os.chdir(self.dir)
            out = np.array((fmt % tuple(find_Rac(Ramin, m, self.ncpus))).split(), dtype=float)
            ref = np.loadtxt('reference.' + params.split('.')[-1])
        finally:
            os.chdir(cwd)
            self.clean()

        np.testing.assert_allclose(out, ref, rtol=self.precision, atol=1e-20)


    def test_jones(self):
        '''
        Onset of convection in a full sphere (Jones, Soward & Mussa 2000): internal heating, fixed heat flux,
        no-slip, Ta = 1e9 (Ek = 6.3e-5), Pr = 1, m = 9
        '''
        self.onset('jones2000', 'params.jones', 4.6e6, 9)


    def test_dormy(self):
        '''
        Onset of convection in a spherical shell (Dormy, Soward, Jones, Jault & Cardin 2004): ricb = 0.35,
        differential heating, fixed temperatures, no-slip, Ek = 2e-5, Pr = 1, m = 9
        '''
        self.onset('dormy2004', 'params.dormy04', 1.6e6, 9)
