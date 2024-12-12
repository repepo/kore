# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright 2016 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""This folder contains css for QSS styling of the application."""

from __future__ import absolute_import, division, print_function

# Standard library imports
import os.path as osp

DATA_PATH = osp.dirname(osp.realpath(__file__))
GLOBAL_STYLES_PATH = osp.join(DATA_PATH, 'styles.css')
