#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import unittest
import spinover.unitTest


def getParser():
    """
    Get script option
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s '+__version__,
                        help="Show program's version number and exit.")
    parser.add_argument('--level', action='store', dest='test_level', type=int,
                        default=-1, help='Test level, use -2 for more info')
    parser.add_argument('--use-debug-flags', action='store_true',
                        dest='use_debug_flags',
                        default=False, help='Use compilation debug flags')
    parser.add_argument('--use-mpi', action='store_true', dest='use_mpi',
                        default=False, help='Use MPI')
    parser.add_argument('--use-openmp', action='store_true', dest='use_openmp',
                        default=False, help='Use the hybrid version')
    parser.add_argument('--use-mkl', action='store_true', dest='use_mkl',
                        default=False,
                        help='Use the MKL for FFTs and Lapack calls')
    parser.add_argument('--use-shtns', action='store_true', dest='use_shtns',
                        default=False, help='Use SHTns for Legendre transforms')
    parser.add_argument('--use-precond', action='store', dest='use_precond',
                        type=bool, default=True,
                        help='Use matrix preconditioning')
    parser.add_argument('--nranks', action='store', dest='nranks', type=int,
                        default=4, help='Specify the number of MPI ranks')
    parser.add_argument('--nthreads', action='store', dest='nthreads', type=int,
                        default=1,
                        help='Specify the number of threads (hybrid version)')
    parser.add_argument('--mpicmd', action='store', dest='mpicmd', type=str,
                        default='mpirun', help='Specify the mpi executable')

    return parser


def print_logo():
    print('\n')
    print("Kore unit test suite")
    print("\n")
    print(" _   __               ")
    print("| | / /               ")
    print("| |/ /  ___  _ __ ___ ")
    print("|    \ / _ \| '__/ _ \\")
    print("| |\  \ (_) | | |  __/")
    print("\_| \_/\___/|_|  \___|")
    print("\n")


def getSuite(startdir,ncpus, solve_opts, precision):
    """
    Construct test suite
    """
    suite = unittest.TestSuite()

    suite.addTest(spinover.unitTest.spinoverTest('outputFileDiff',
                                                 ncpus,
                                                 solve_opts,
                                                  '%s/spinover' %startdir,
                                                  precision=precision))

    return suite


if __name__ == '__main__':
    precision = 1e-8 # relative tolerance between expected and actual result
    startdir = os.getcwd()
    ncpus = 2
    solve_opts = '-st_type sinvert -eps_error_relative ::ascii_info_detail'

    # parser = getParser()
    # args = parser.parse_args()

    # Initialisation
    print_logo()

    # Run the auto-test suite
    print('  Running test suite  ')
    print('----------------------')
    suite = getSuite(startdir,ncpus, solve_opts, precision)
    runner = unittest.TextTestRunner(verbosity=0)
    ret = not runner.run(suite).wasSuccessful()

    sys.exit(ret)
