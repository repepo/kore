# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2016 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------
"""
This folder contains image files bundled with Anaconda Navigator package.

This folder is defined as a python module so that some convenience global
variables can be defined.
"""

from __future__ import absolute_import, division, print_function

# Standard library imports
import os.path as osp

IMAGE_PATH = osp.dirname(osp.realpath(__file__))
LOGO_PATH = osp.join(IMAGE_PATH, 'logos')

# --- Anaconda Logo
# -----------------------------------------------------------------------------
ANACONDA_ICON_16_PATH = osp.join(IMAGE_PATH, 'anaconda-icon-16x16.png')
ANACONDA_ICON_24_PATH = osp.join(IMAGE_PATH, 'anaconda-icon-24x24.png')
ANACONDA_ICON_32_PATH = osp.join(IMAGE_PATH, 'anaconda-icon-32x32.png')
ANACONDA_ICON_48_PATH = osp.join(IMAGE_PATH, 'anaconda-icon-48x48.png')
ANACONDA_ICON_64_PATH = osp.join(IMAGE_PATH, 'anaconda-icon-64x64.png')
ANACONDA_ICON_75_PATH = osp.join(IMAGE_PATH, 'anaconda-icon-75x75.png')
ANACONDA_ICON_128_PATH = osp.join(IMAGE_PATH, 'anaconda-icon-128x128.png')
ANACONDA_ICON_256_PATH = osp.join(IMAGE_PATH, 'anaconda-icon-256x256.png')
ANACONDA_ICON_512_PATH = osp.join(IMAGE_PATH, 'anaconda-icon-512x512.png')
ANACONDA_ICON_1024_PATH = osp.join(IMAGE_PATH, 'anaconda-icon-1024x1024.png')
ANACONDA_NAVIGATOR_LOGO = osp.join(IMAGE_PATH, 'anaconda-navigator-logo.svg')
ANACONDA_LOGO = osp.join(IMAGE_PATH, 'anaconda-logo.svg')
ANACONDA_NAVIGATOR_LOGO_PNG = osp.join(
    IMAGE_PATH, 'anaconda-navigator-logo.png'
)
