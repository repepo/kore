#!/usr/bin/env python3
'''
Helpers shared by the Kore tests.

Each test case lives in tests/<case>/ with a parameter file and a reference result. A test makes a
fresh copy of bin/ in that folder, installs the parameter file as bin/parameters.py, runs Kore there
and compares the result with the reference.

Run from this folder with:
> pytest . -v -s
'''

import os
import shutil
import subprocess
from glob import glob



class KoreTest:

    precision = 1e-8  # relative tolerance when comparing with the reference values
    ncpus     = 2     # number of MPI ranks used by the tests

    tests_dir = os.path.dirname(os.path.abspath(__file__))
    kore_dir  = os.path.dirname(tests_dir)


    def prepare(self, case, params):
        '''
        Sets up tests/<case>/ for a run: a fresh copy of bin/ with <params> as bin/parameters.py
        '''
        self.dir = os.path.join(self.tests_dir, case)
        self.clean()
        # no __pycache__, so that no stale compiled parameters.py can be picked up
        shutil.copytree(os.path.join(self.kore_dir, 'bin'), os.path.join(self.dir, 'bin'),
                        ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy(os.path.join(self.dir, params), os.path.join(self.dir, 'bin', 'parameters.py'))


    def run(self, cmd, quiet=False):
        '''
        Runs a shell command in the case folder. The test fails if the command fails.
        '''
        subprocess.run(cmd, shell=True, check=True, cwd=self.dir,
                       stdout=subprocess.DEVNULL if quiet else None)


    def clean(self):
        '''
        Removes everything a run leaves behind in the case folder
        '''
        for pattern in ['*.mtx', '*.npz', '*.field', '*.dat', 'no_conv_solution']:
            for f in glob(os.path.join(self.dir, pattern)):
                os.remove(f)
        shutil.rmtree(os.path.join(self.dir, 'bin'), ignore_errors=True)
