#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import unittest
import spinover.unitTest

__version__="0.2"

def getParser():
    """
    Get script option
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s '+__version__,
                        help="Show program's version number and exit.")
    parser.add_argument('--nranks', action='store', dest='ncpus', type=int,
                        default=2, help='Specify the number of MPI ranks')

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
    solve_opts = '-st_type sinvert -eps_error_relative ::ascii_info_detail'

    parser = getParser()
    args = parser.parse_args()

    ncpus = args.ncpus

    # Initialisation
    print_logo()

    # Run the auto-test suite
    print('  Running test suite using %d MPI ranks  ' %args.ncpus)
    print('-----------------------------------------')
    suite = getSuite(startdir,ncpus, solve_opts, precision)
    runner = unittest.TextTestRunner(verbosity=0)
    ret = not runner.run(suite).wasSuccessful()

    sys.exit(ret)
