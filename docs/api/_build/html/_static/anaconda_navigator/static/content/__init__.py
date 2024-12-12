# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""
This folder contains data files bundled with Anaconda Navigator package.

This folder is defined as a python module so that some convenience global variables can be defined.
"""

from __future__ import annotations

__all__ = ['DATA_PATH', 'LINKS_INFO_PATH', 'BUNDLE_METADATA_COMP_PATH', 'EXTERNAL_APPS_CONF_PATH', 'CONF_METADATA_PATH']

import os
import typing

from anaconda_navigator.config import METADATA_PATH


DATA_PATH: typing.Final[str] = os.path.dirname(os.path.realpath(__file__))

LINKS_INFO_PATH: typing.Final[str] = os.path.join(DATA_PATH, 'links.json')
BUNDLE_METADATA_COMP_PATH: typing.Final[str] = os.path.join(DATA_PATH, 'metadata.json.bz2')
EXTERNAL_APPS_CONF_PATH: typing.Final[str] = os.path.join(DATA_PATH, 'external_apps.yml')

CONF_METADATA_PATH: typing.Final[str] = os.path.join(METADATA_PATH, 'metadata.json')
