# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Default configuration options."""

__all__ = ['CONF']

import os
import sys
import typing
from . import base
from . import preferences
from . import user


# -----------------------------------------------------------------------------
# --- Defaults
# -----------------------------------------------------------------------------

DEFAULTS: typing.Final[typing.Sequence[typing.Tuple[str, typing.Mapping[str, typing.Any]]]] = [
    (
        'main',  # General
        {
            'name': 'Anaconda Navigator',
            'first_run': True,
            'hide_quit_dialog': False,
            'hide_running_apps_dialog': False,
            'hide_update_dialog': False,
            'hide_offline_dialog': True,
            'identity': '',
            'first_time_offline': True,
            'last_status_is_offline': False,
            'running_apps_to_close': ['anaconda-fusion'],  # Hidden opt
            'add_default_channels': True,
            'offline_mode': False,
            'default_env': os.environ.get('CONDA_PREFIX'),

            # --- Package Manager
            'conda_active_channels': None,

            # --- Anaconda Client Configuration, these values are not needed
            'logged_brand': None,
            'logged_api_url': None,
            'auth_domain': 'id.anaconda.cloud',
            'cloud_base_url': 'https://anaconda.cloud',
            'anaconda_api_url': 'https://api.anaconda.org',
            'anaconda_server_api_url': None,
            'anaconda_server_token': None,
            'anaconda_server_token_id': None,
            'anaconda_server_show_hidden_channels': False,
            'enterprise_4_repo_api_url': None,
            'anaconda_professional_url': 'https://repo.anaconda.cloud',
            'ssl_verification': True,
            'ssl_certificate': None,
            # Used by batch initial config
            'default_anaconda_api_url': None,
            'default_ssl_certificate': None,

            # --- Preferences
            'enable_high_dpi_scaling': not sys.platform.startswith('linux'),
            'provide_analytics': True,
            'show_application_environments': True,
            'show_application_launch_errors': True,

            # --- Advertisements url
            'advertisement_url': 'https://www.anaconda.com/api/navigator',
        },
    ),
    # --- Server validation preferences
    (
        'ssl',
        {
            'trusted_servers': [],
        }
    ),
    # --- External apps paths
    (
        'applications',
        {
            'dataspell_path': '',
            'pycharm_ce_path': '',
            'pycharm_pro_path': '',
            'vscode_path': '',
        }
    ),
    # --- Internal variables
    (
        'internal',
        {
            'welcome_state': preferences.WELCOME_DELAYS.first,
            'welcome_ts': 0,
        }
    ),
]

# -----------------------------------------------------------------------------
# --- Config instance
# -----------------------------------------------------------------------------

base.fix_recursive_folder()

# IMPORTANT NOTES:
# 1. If you want to *change* the default value of a current option, you need to
#    do a MINOR update in config version, e.g. from 1.0.0 to 1.1.0
# 2. If you want to *remove* options that are no longer needed in our codebase,
#    or if you want to *rename* options, then you need to do a MAJOR update in
#    version, e.g. from 1.0.0 to 2.0.0
# 3. You don't need to touch this value if you're just adding a new option
CONF_VERSION: typing.Final[str] = '2.0.0'
CONF: typing.Final[user.UserConfig] = user.UserConfig(
    name=base.CONFIG_NAME,
    defaults=DEFAULTS,
    version=CONF_VERSION,
    subfolder=base.SUBFOLDER,
    raw_mode=True,
)
