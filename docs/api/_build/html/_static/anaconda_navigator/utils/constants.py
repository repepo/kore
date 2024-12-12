# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Constants used by package manager widget."""

from collections import OrderedDict
import enum
import gettext


_ = gettext.gettext

# Constants
COLUMNS = (
    COL_START,
    COL_ACTION,
    COL_NAME,
    COL_PACKAGE_TYPE,
    COL_DESCRIPTION,
    COL_VERSION,
    COL_STATUS,
    COL_URL,
    COL_ACTION_VERSION,
    COL_END,

    COL_UPGRADE,
    COL_INSTALL,
    COL_REMOVE,
    COL_DOWNGRADE,
) = list(range(0, 14))

ACTION_COLUMNS = [
    COL_ACTION,
    COL_ACTION_VERSION,
]

ACTIONS = (
    ACTION_NONE,
    ACTION_INSTALL,
    ACTION_REMOVE,
    ACTION_UPGRADE,
    ACTION_DOWNGRADE,
    ACTION_UPDATE,  # equivalent to conda update, no version specified
    ACTION_CREATE,
    ACTION_CLONE,
    ACTION_REMOVE_ENV,
    ACTION_IMPORT,
    ACTION_SEARCH,
    ACTION_EXPORT,
) = list(range(100, 112))

PACKAGE_TYPES = (
    CONDA_PACKAGE,
    PIP_PACKAGE,
) = ['     conda', '    pip']

PACKAGE_STATUS = (
    INSTALLED,
    NOT_INSTALLED,
    UPGRADABLE,
    DOWNGRADABLE,
    SELECTED,
    ALL,
    MIXGRADABLE,
) = list(range(200, 207))

COMBOBOX_VALUES_ORDERED = [
    'Installed',
    'Not installed',
    'Updatable',
    'Selected',
    'All',
]

COMBO_PACKAGE_STATUS = PACKAGE_STATUS[:]
COMBO_PACKAGE_STATUS.remove(DOWNGRADABLE)
COMBO_PACKAGE_STATUS.remove(MIXGRADABLE)
COMBOBOX_VALUES = OrderedDict(zip(COMBOBOX_VALUES_ORDERED, COMBO_PACKAGE_STATUS))
ROOT = 'root'
UPGRADE_SYMBOL = ' ⬆'

# Application actions
APPLICATION_INSTALL = 'install'
APPLICATION_UPDATE = 'update'
APPLICATION_REMOVE = 'remove'
APPLICATION_LAUNCH = 'launch'

# Widget names for senders
MAIN_UPDATE = 'main-update'
ENVIRONMENT_PACKAGE_MANAGER = 'environment-package-manager'
TAB_HOME = 'tab-home'
TAB_ENVIRONMENT = 'tab-environment'
TAB_LEARNING = 'tab-learning'
TAB_COMMUNITY = 'tab-community'

ACTION_2_WORD = {
    ACTION_NONE: 'No action',
    ACTION_INSTALL: 'Installing',
    ACTION_REMOVE: 'Removing',
    ACTION_UPGRADE: 'Updating',
    ACTION_DOWNGRADE: 'Downgrading',
    ACTION_CREATE: 'Creating',
    ACTION_CLONE: 'Cloning',
    ACTION_REMOVE_ENV: 'Removing',
    ACTION_IMPORT: 'Importing',
    ACTION_SEARCH: 'Searching',
}


class AppType(str, enum.Enum):
    """Options for the application types."""

    WEB = 'web_app'
    CONDA = 'conda_app'
    INSTALLABLE = 'installable'
