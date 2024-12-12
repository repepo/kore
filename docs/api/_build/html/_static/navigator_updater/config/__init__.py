# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright 2016 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Module in charge of the configuration settings."""

# Standard library imports
import os
import sys

# Local imports
from navigator_updater import __version__
from navigator_updater.config.base import get_conf_path, get_home_dir

# FLAGS
TEST_CI = os.environ.get('TEST_CI', False)
MAC = sys.platform == 'darwin'
WIN = os.name == 'nt'
LINUX = sys.platform.startswith('linux')
DEV = 'dev' in __version__

HOME_PATH = get_home_dir()
CONF_PATH = get_conf_path()
LAUNCH_SCRIPTS_PATH = os.path.join(CONF_PATH, 'scripts')
LOCKFILE = os.path.join(CONF_PATH, 'navigator_updater.lock')
NAVIGATOR_LOCKFILE = os.path.join(CONF_PATH, 'navigator.lock')
PIDFILE = os.path.join(CONF_PATH, 'navigator.pid')

# Logging
LOG_FOLDER = os.path.join(CONF_PATH, 'logs')
LOG_FILENAME = 'navigator_updater.log'
MAX_LOG_FILE_SIZE = 2 * 1024 * 1024
