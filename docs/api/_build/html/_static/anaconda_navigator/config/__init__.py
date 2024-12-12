# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""
Module in charge of the configuration settings.

It uses a modified version of Python's configuration parser.
"""

import enum
import os
import platform
import sys
import typing
from anaconda_navigator.__about__ import __version__
from anaconda_navigator.config.base import get_conf_path, get_home_dir, is_gtk_desktop, is_ubuntu, SUBFOLDER
from anaconda_navigator.config.main import CONF


# FLAGS
TEST_CI = os.environ.get('TEST_CI', False)
MAC = sys.platform == 'darwin'


class AnacondaBrand(str, enum.Enum):
    """Represents Anaconda products."""
    ANACONDA_ORG = 'Anaconda.org'
    CLOUD = 'Anaconda Cloud'
    ENTERPRISE_EDITION = 'Enterprise 4 Repository'
    TEAM_EDITION = 'Anaconda Server'
    COMMERCIAL_EDITION = 'Anaconda Professional'
    DEFAULT = ANACONDA_ORG

    def __repr__(self):
        return self.value


if MAC:
    MAC_VERSION: str = platform.mac_ver()[0]
    MAC_VERSION_INFO: typing.Tuple[typing.Union[int, str], ...] = tuple(
        int(item) if item.isdigit() else item
        for item in MAC_VERSION.split('.')
    )
else:
    MAC_VERSION = ''
    MAC_VERSION_INFO = ()

WIN = os.name == 'nt'
try:
    WIN7 = platform.platform().lower().startswith('windows-7')
except Exception:  # pylint: disable=broad-except
    WIN7 = False

LINUX = sys.platform.startswith('linux')
UBUNTU = is_ubuntu()
GTK = is_gtk_desktop()
DEV = 'dev' in __version__
BITS = 8 * tuple.__itemsize__
BITS_64 = BITS == 64
BITS_32 = BITS == 32
OS_64_BIT = platform.machine().endswith('64')

# Paths
HOME_PATH = get_home_dir()
CONF_PATH = get_conf_path()
LAUNCH_SCRIPTS_PATH = os.path.join(CONF_PATH, 'scripts')
CONTENT_PATH = os.path.join(CONF_PATH, 'content')
CONTENT_JSON_PATH = os.path.join(CONTENT_PATH, 'content.json')
IMAGE_ICON_SIZE = (256, 256)
IMAGE_DATA_PATH = os.path.join(CONF_PATH, 'images')
CHANNELS_PATH = os.path.join(CONF_PATH, 'channels')
METADATA_PATH = os.path.join(CONF_PATH, 'metadata')
LOCKFILE = os.path.join(CONF_PATH, 'navigator.lock')
PIDFILE = os.path.join(CONF_PATH, 'navigator.pid')
AD_CACHE: typing.Final[str] = os.path.join(CONF_PATH, 'ad_cache')
CLOUD_CACHE: typing.Final[str] = os.path.join(CONF_PATH, 'cloud_cache')

DEFAULT_ANACONDA_MAIN_URL = 'https://anaconda.org'

VALID_DEV_TOOLS = ['notebook', 'qtconsole', 'spyder']
LOG_FOLDER = os.path.join(CONF_PATH, 'logs')
LOG_FILENAME = 'navigator.log'

MAX_LOG_FILE_SIZE = 2 * 1024 * 1024
