#!/usr/bin/env python3

import os
import numpy as np
from koretest import KoreTest
from find_Rac import find_Rac, set_param



class TestThermoCompositional(KoreTest):

    # Each Rac comes from a root search on log10(Ra) with tolerance 1e-6 (about 1.5e-5 relative in Ra)
    precision = 5e-5

    def test_codensity(self):
        '''
        Onset of thermo-compositional convection with equal Prandtl and Schmidt numbers (Silva, Mather & Simitev
        2019, GAFD, section 5.1, journal p. 13): the problem then reduces to a single co-density, and the critical
        curve in the Rt-Rc plane is the straight line Rt + Rc = Ra0, i.e. (their eq. 18)

            Rac(alpha) = Ra0 / (cos(alpha) + sin(alpha)),

        with Rt = Ra*cos(alpha) and Rc = Ra*sin(alpha) (their eqs. 7-8, p. 7). Rac is computed at alpha = 0
        (driven by temperature only), pi/2 (by composition only) and pi/4 (both), and Ra0 and the drift
        frequency must be the same for all three. Pr = Sc = 1, tau = 1e4, eta = 0.35, m = 7.
        '''
        self.prepare('silva2019', 'params.silva')
        cwd = os.getcwd()
        out = []
        try:
            os.chdir(self.dir)
            set_param('Pr', 1.0)
            set_param('Sc', 1.0)
            for alpha in (0, np.pi/2, np.pi/4):
                # starting from their local estimate (15a), p. 12, Rcrit = 2.3e5
                Ek, ricb, Rac, m, omega = find_Rac(2.3e5, 7, self.ncpus,
                                                   update=lambda Ra: {'Rt': Ra*np.cos(alpha), 'Rc': Ra*np.sin(alpha)})
                out.append([Rac*(np.cos(alpha) + np.sin(alpha)), omega])
        finally:
            os.chdir(cwd)
            self.clean()

        out = np.array(out)
        for row in out[1:]:
            np.testing.assert_allclose(row, out[0], rtol=self.precision)
