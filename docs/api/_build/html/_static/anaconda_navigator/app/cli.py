#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Command line arguments."""

import argparse
import logging


def parse_arguments():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description='Anaconda Navigator.')
    parser.add_argument(
        '--version',
        help='Print version information',
        action='store_const',
        const=True,
        default=False,
        dest='version',
    )
    parser.add_argument(
        '--reset',
        help='Reset Navigator configuration',
        action='store_const',
        dest='reset',
        const=True,
        default=False,
    )
    parser.add_argument(
        '--remove-lock',
        help='Remove Navigator lock',
        action='store_const',
        dest='removelock',
        const=True,
        default=False,
    )
    parser.add_argument(
        '--verbose',
        help='Print information statements',
        action='store_const',
        dest='log_level',
        const=logging.INFO,
        default=logging.INFO,
    )
    parser.add_argument(
        '--debug',
        help='Print debug statements',
        action='store_const',
        dest='log_level',
        const=logging.DEBUG,
        default=logging.INFO,
    )

    return parser.parse_args()
