# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Constants used by package manager widget."""

import enum


class AppType(str, enum.Enum):
    """Options for the application types."""

    WEB = 'web_app'
    CONDA = 'conda_app'
    INSTALLABLE = 'installable'
