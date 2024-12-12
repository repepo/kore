# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""
This folder contains image files bundled with Anaconda Navigator package.

This folder is defined as a python module so that some convenience global
variables can be defined.
"""

from __future__ import annotations

__all__ = ()

import os

IMAGE_PATH = os.path.dirname(os.path.realpath(__file__))
CLOUD_BACKGROUNDS_PATH = os.path.join(IMAGE_PATH, 'cloud_backgrounds')
LOGO_PATH = os.path.join(IMAGE_PATH, 'logos')
ICONS_PATH = os.path.join(IMAGE_PATH, 'icons')

# --- Anaconda Logo
# -----------------------------------------------------------------------------
ANACONDA_LOGO = os.path.join(IMAGE_PATH, 'anaconda-logo.svg')
ANACONDA_LOGO_WHITE = os.path.join(IMAGE_PATH, 'anaconda-logo-white.svg')
ANACONDA_NAVIGATOR_LOGO = os.path.join(IMAGE_PATH, 'anaconda-navigator-logo.svg')
ANACONDA_ICON_256_PATH = os.path.join(IMAGE_PATH, 'anaconda-icon-256x256.png')
ANACONDA_SERVER_LOGIN_LOGO = os.path.join(LOGO_PATH, 'Logo-Server.png')
ANACONDA_ORG_EDITION_LOGIN_LOGO = os.path.join(LOGO_PATH, 'Anaconda_Logo_RGB_Org.png')
ANACONDA_PROFESSIONAL_LOGIN_LOGO = os.path.join(LOGO_PATH, 'Logo-Professional.png')
ANACONDA_ENTERPRISE_LOGIN_LOGO = os.path.join(LOGO_PATH, 'Logo-Enterprise.png')
ANACONDA_CLOUD_LOGIN_LOGO = os.path.join(LOGO_PATH, 'Anaconda_Logo_RGB_Cloud.png')

# NOTE: Check copyright on this image
VIDEO_ICON_PATH = os.path.join(IMAGE_PATH, 'default-content.png')
CLOSE_DIALOG_ICON_PATH = os.path.join(ICONS_PATH, 'close-icon.svg')
REFRESH_ICON_PATH = os.path.join(ICONS_PATH, 'refresh.svg')
EXCLAMATION_CIRCLE_PATH = os.path.join(ICONS_PATH, 'exclamation-circle.svg')
BLOCK_CLOSE_ICON_PATH = os.path.join(ICONS_PATH, 'block-close.svg')

# --- Application icons
# -----------------------------------------------------------------------------
GLUEVIZ_ICON_1024_PATH = os.path.join(IMAGE_PATH, 'glueviz-icon-1024x1024.png')
NOTEBOOK_ICON_1024_PATH = os.path.join(IMAGE_PATH, 'jupyter-icon-1024x1024.png')
ORANGE_ICON_1024_PATH = os.path.join(IMAGE_PATH, 'orange-icon-1024x1024.png')
QTCONSOLE_ICON_1024_PATH = os.path.join(IMAGE_PATH, 'qtconsole-icon-1024x1024.png')
SPYDER_ICON_1024_PATH = os.path.join(IMAGE_PATH, 'spyder-icon-1024x1024.png')
RODEO_ICON_1024_PATH = os.path.join(IMAGE_PATH, 'rodeo-icon-1024x1024.png')
VEUSZ_ICON_1024_PATH = os.path.join(IMAGE_PATH, 'veusz-icon-1024x1024.png')
RSTUDIO_ICON_1024_PATH = os.path.join(IMAGE_PATH, 'rstudio-icon-1024x1024.png')
JUPYTERLAB_ICON_1024_PATH = os.path.join(IMAGE_PATH, 'jupyterlab-icon-1024x1024.png')
VSCODE_ICON_1024_PATH = os.path.join(IMAGE_PATH, 'vscode-icon-1024x1024.png')
PYCHARM_ICON_1024_PATH = os.path.join(IMAGE_PATH, 'pycharm-icon.png')
PYCHARM_CE_ICON_1024_PATH = os.path.join(IMAGE_PATH, 'pycharm-ce-icon.png')
DATALORE_ICON_1024_PATH = os.path.join(IMAGE_PATH, 'datalore-icon.png')
QTCREATOR_ICON_1024_PATH = os.path.join(IMAGE_PATH, 'qtcreator-icon-1024x1024.png')

# --- Spinners
# -----------------------------------------------------------------------------
# http://preloaders.net/en/circular
SPINNER_16_PATH = os.path.join(IMAGE_PATH, 'spinner-16x16.gif')
SPINNER_32_PATH = os.path.join(IMAGE_PATH, 'spinner-32x32.gif')
SPINNER_GREEN_16_PATH = os.path.join(IMAGE_PATH, 'spinner-green-16x16.gif')
SPINNER_WHITE_16_PATH = os.path.join(IMAGE_PATH, 'spinner-white-16x16.gif')

# Conda Package Manager Table icons
# -----------------------------------------------------------------------------
MANAGER_INSTALLED = os.path.join(IMAGE_PATH, 'icons', 'check-box-checked-active.svg')
MANAGER_NOT_INSTALLED = os.path.join(IMAGE_PATH, 'icons', 'check-box-blank.svg')
MANAGER_ADD = os.path.join(IMAGE_PATH, 'icons', 'mark-install.svg')
MANAGER_REMOVE = os.path.join(IMAGE_PATH, 'icons', 'mark-remove.svg')
MANAGER_DOWNGRADE = os.path.join(IMAGE_PATH, 'icons', 'mark-downgrade.svg')
MANAGER_UPGRADE = os.path.join(IMAGE_PATH, 'icons', 'mark-upgrade.svg')
MANAGER_UPGRADE_ARROW = os.path.join(IMAGE_PATH, 'icons', 'update-app-active.svg')
MANAGER_SPACER = os.path.join(IMAGE_PATH, 'conda-manager-spacer.svg')
WARNING_ICON = os.path.join(IMAGE_PATH, 'icons', 'warning-active.svg')
INFO_ICON = os.path.join(IMAGE_PATH, 'icons', 'info-active.svg')
PYTHON_LOGO = os.path.join(IMAGE_PATH, 'python-logo.svg')

# Additional dialog icons
# -----------------------------------------------------------------------------
PASSWORD_HIDDEN = os.path.join(IMAGE_PATH, 'icons', 'eye-hidden.svg')
PASSWORD_VISIBLE = os.path.join(IMAGE_PATH, 'icons', 'eye-visible.svg')
BUSY = os.path.join(IMAGE_PATH, 'busy.gif')
