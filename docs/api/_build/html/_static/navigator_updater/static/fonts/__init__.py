# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2016 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------
"""Custom Fonts."""

from __future__ import absolute_import, division, print_function

# Standard library imports
import glob
import os

# Third party imports
from qtpy.QtGui import QFontDatabase

PATH = os.path.dirname(os.path.realpath(__file__))
FONTFILES = glob.glob(os.path.join(PATH, '*.ttf'))


def load_fonts(app):
    """Load custom fonts."""
    app.database = QFontDatabase()
    for font in FONTFILES:
        app.database.addApplicationFont(font)
