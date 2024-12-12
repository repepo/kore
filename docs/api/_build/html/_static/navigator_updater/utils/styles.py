# -*- coding: utf-8 -*-

# pylint: disable=invalid-name

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Styles for the application."""

import os
from navigator_updater.static import images
from navigator_updater.static.css import GLOBAL_STYLES_PATH


def load_style_sheet():
    """Load css styles file and parse to include custom variables."""
    with open(GLOBAL_STYLES_PATH, 'rt', encoding='utf-8') as f:
        data = f.read()

    if os.name == 'nt':
        data = data.replace('$IMAGE_PATH', images.IMAGE_PATH.replace('\\', '/'))
    else:
        data = data.replace('$IMAGE_PATH', images.IMAGE_PATH)

    return data
