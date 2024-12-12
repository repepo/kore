#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Application entry point."""

import os
import shutil
import sys
import logging
from anaconda_navigator import __about__
from anaconda_navigator.app.cli import parse_arguments
from anaconda_navigator.app.start import start_app
from anaconda_navigator.config import CONF_PATH
from anaconda_navigator.exceptions import exception_handler
from anaconda_navigator.utils.conda import is_conda_available
from anaconda_navigator.utils.logs import clean_logs
from anaconda_navigator.utils.misc import remove_lock, remove_pid


bis_debug = False  # pylint: disable=invalid-name


def is_debug_enabled():  # pylint: disable=missing-function-docstring
    return bis_debug


def set_debug(flag=True):  # pylint: disable=missing-function-docstring
    global bis_debug  # pylint: disable=global-statement,invalid-name
    bis_debug = flag


def main():  # cov-skip
    """Main application entry point."""
    global bis_debug  # pylint: disable=global-statement,invalid-name
    # Check if conda is available
    if not is_conda_available():
        path = os.path.abspath(os.path.dirname(sys.argv[0]))
        # print(path, len(sys.argv))
        msg = '''#
# Please activate the conda root enviroment properly before running the
# `anaconda-navigator` command.
'''
        win_msg = f'''#
# To activate the environment please open a Windows Command Prompt and run:
#
#   {path}\\activate root
'''

        unix_msg = f'''#
# To activate the environment please open a terminal and run:
#
#   . {path}/activate root
'''

        more_info = '''#
# For more information please see the documentation at:
#
#   https://docs.anaconda.com/free/navigator/
#'''
        if os.name == 'nt':
            print_msg = f'{msg}{win_msg}{more_info}'
        else:
            print_msg = f'{msg}{unix_msg}{more_info}'

        print(print_msg)

        return 1

    # Parse CLI arguments
    options = parse_arguments()

    # Return information on version
    if options.version:
        print(__about__.__version__)
        sys.exit(0)

    if options.log_level == logging.DEBUG:
        bis_debug = True

    # Reset Navigator conifg
    if options.reset:
        print('\nAnaconda Navigator configuration reset...\n\n')
        if os.path.isdir(CONF_PATH):
            try:
                shutil.rmtree(CONF_PATH)
                print('Anaconda Navigator configuration reset successful!\n')
                sys.exit(0)
            except Exception as e:  # pylint: disable=broad-except,invalid-name
                print('Anaconda Navigator configuration reset failed!!!\n')
                print(e)
                sys.exit(1)

    if options.removelock:
        print('\nRemoving Anaconda Navigator lock...\n\n')
        lock = remove_lock()
        pid = remove_pid()
        if lock and pid:
            print('Anaconda Navigator lock removal successful!\n')
            sys.exit(0)
        else:
            print('Anaconda Navigator lock removal failed!!!\n')
            sys.exit(1)

    # Clean old style logs
    clean_logs()

    # Import app
    return exception_handler(start_app, options)


if __name__ == '__main__':  # cov-skip
    main()
