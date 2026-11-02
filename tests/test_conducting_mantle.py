#!/usr/bin/env python3

import os
import numpy as np
from koretest import KoreTest
from find_Rac import set_param



class TestConductingMantle(KoreTest):

    precision = 1e-9

    # NEW (2026-10-30): slowest free-decay rates of the poloidal l = 1 and toroidal l = 2 fields (m = 0) in the case of
    # params.conducting_mantle. Uniform layer: exact, from spherical Bessel functions j_l, y_l in the fluid and in the
    # layer, matched at the CMB (b and the tangential electric field continuous), at the ICB and at r_surface (potential
    # fields). Exponential layer: the layer's equation integrated from r_surface (scipy DOP853, rtol 1e-13) and matched to
    # the exact fluid solution at the CMB. Kore gets them to 2e-11 at this resolution.
    cases = [ ( 0.3,                   { 'pol l=1': -8.220476516888839, 'tor l=2': -28.12194013329784 } ),
              ( '0.3*exp(-p*(r - 1))', { 'pol l=1': -9.521791176917253, 'tor l=2': -34.95451577224672 } ) ]   # p = 10

    def test_conducting_mantle(self):
        '''
        mantle = 'resolved': free decay of the field under a conducting mantle layer, uniform and with an exponential
        conductivity profile (the profile also enters through d(ln sigma)/dr in the toroidal equation)
        '''
        self.prepare('conducting_mantle', 'params.conducting_mantle')
        cwd = os.getcwd()
        try:
            os.chdir(self.dir)
            for sigma, refs in self.cases:
                set_param('par.sigma_mantle', sigma)
                self.run('./bin/submatrices.py %d' % self.ncpus, quiet=True)
                self.run('mpiexec -n %d ./bin/assemble.py' % self.ncpus, quiet=True)
                for name, lam in refs.items():
                    set_param('par.rtau', 1.001*lam)
                    self.run('mpiexec -n %d ./bin/solve_nopp.py' % self.ncpus, quiet=True)
                    eig = np.loadtxt('eigenvalues0.dat').reshape(-1, 2)
                    os.remove('eigenvalues0.dat')
                    z = eig[:, 0] + 1j*eig[:, 1]
                    z = z[np.argmin(np.abs(z - lam))]
                    assert abs(z - lam)/abs(lam) < self.precision, (sigma, name, z, lam)
        finally:
            os.chdir(cwd)
            self.clean()
