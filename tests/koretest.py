#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from glob import glob

class KoreTest:

    precision  = 1e-8
    ncpus      = 4
    solve_opts = "-st_type sinvert -eps_error_relative ::ascii_info_detail"
    startDir   = os.getcwd()
    kore_dir   = os.path.join(os.path.dirname(
                        os.path.abspath(__file__)),'..'
                        )

    print('  Running test suite using %d MPI ranks  ' % ncpus)
    print('-----------------------------------------')
    print('\n')
    print("Kore test suite")
    print("\n")
    print(" _   __               ")
    print("| | / /               ")
    print("| |/ /  ___  _ __ ___ ")
    print(r"|    \ / _ \| '__/ _ \\")
    print(r"| |\  \ (_) | | |  __/")
    print(r"\_| \_/\___/|_|  \___|")
    print("\n")

    def cleanDir(self,startDir,dir):
        """ Clean up the test directory
        """
        os.chdir(dir)
        for f in glob("%s/*.mtx" % dir):
            os.remove(f)
        for f in glob("%s/*.npz" % dir):
            os.remove(f)
        for f in glob("%s/*.field" % dir):
            os.remove(f)
        for f in glob("%s/*.dat" % dir):
            os.remove(f)
        shutil.rmtree('./bin')
        os.chdir(startDir)