# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright 2016 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Command line arguments."""

# Standard library imports
import argparse
import logging


def parse_arguments():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description='Navigator Updater.')
    parser.add_argument(
        '--version',
        help='Print version information',
        action='store_const',
        const=True,
        default=False,
        dest='version',
    )
    parser.add_argument(
        '--latest-version',
        help='Latest version available for update',
        action='store',
        dest='latest_version',
        const=None,
        default=None,
        nargs='?',
    )
    parser.add_argument(
        '--prefix',
        help='Environment prefix on which to perform the update',
        action='store',
        dest='prefix',
        const=None,
        default=None,
        nargs='?',
    )
    parser.add_argument(
        '--debug',
        help='Print debug statements',
        action='store_const',
        dest='log_level',
        const=logging.DEBUG,
        default=logging.WARNING,
    )

    return parser.parse_args()
