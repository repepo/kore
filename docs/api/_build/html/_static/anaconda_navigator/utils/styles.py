# -*- coding: utf-8 -*-

# pylint: disable=invalid-name

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Styles for the application."""

import ast
import contextlib
import os
import re
import string
from anaconda_navigator.static import images
from anaconda_navigator.static.css import GLOBAL_SASS_STYLES_PATH, GLOBAL_STYLES_PATH


BLUR_SIZE = 10
MANAGER_TABLE_STYLES = {
    'icon.action.not_installed': images.MANAGER_NOT_INSTALLED,
    'icon.action.installed': images.MANAGER_INSTALLED,
    'icon.action.remove': images.MANAGER_REMOVE,
    'icon.action.add': images.MANAGER_ADD,
    'icon.action.upgrade': images.MANAGER_UPGRADE,
    'icon.action.downgrade': images.MANAGER_DOWNGRADE,
    'icon.upgrade.arrow': images.MANAGER_UPGRADE_ARROW,
    'icon.spacer': images.MANAGER_SPACER,
    'icon.python': images.PYTHON_LOGO,
    'icon.anaconda': images.ANACONDA_LOGO,
    'color.background.remove': '#0000',
    'color.background.install': '#0000',
    'color.background.upgrade': '#0000',
    'color.background.downgrade': '#0000',
    'color.foreground.not.installed': '#666',
    'color.foreground.upgrade': '#00A3E0',
    'size.icons': (32, 32),
}


class SassVariables:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Enum to hold SASS defined variables."""

    def __init__(self) -> None:
        """Enum to hold SASS defined variables."""
        self.SHADOW_BLUR_RADIUS = 7  # Used for dialogs
        self.WIDGET_APPLICATION_TOTAL_HEIGHT = 200
        self.WIDGET_APPLICATION_TOTAL_WIDTH = 200
        self.WIDGET_CONTENT_PADDING = 5
        self.WIDGET_CONTENT_TOTAL_HEIGHT = 200
        self.WIDGET_CONTENT_TOTAL_WIDTH = 200
        self.WIDGET_CONTENT_PADDING = 5
        self.WIDGET_CONTENT_MARGIN = 5
        self.WIDGET_ENVIRONMENT_TOTAL_HEIGHT = 50
        self.WIDGET_IMPORT_ENVIRONMENT_TOTAL_HEIGHT = 55
        self.WIDGET_ENVIRONMENT_TOTAL_WIDTH = 25
        self.WIDGET_APPLICATION_TOTAL_WIDTH = 260
        self.WIDGET_APPLICATION_TOTAL_HEIGHT = 295
        self.WIDGET_CHANNEL_DIALOG_WIDTH = 400
        self.WIDGET_CHANNEL_TOTAL_WIDTH = 300
        self.WIDGET_CHANNEL_TOTAL_HEIGHT = 40
        self.WIDGET_CHANNEL_PADDING = 5
        self.WIDGET_RUNNING_APPS_WIDTH = 450
        self.WIDGET_RUNNING_APPS_TOTAL_WIDTH = 350
        self.WIDGET_RUNNING_APPS_TOTAL_HEIGHT = 55
        self.WIDGET_RUNNING_APPS_PADDING = 10
        self.WIDGET_LOGIN_CARD_TOTAL_WIDTH = 315
        self.WIDGET_LOGIN_CARD_TOTAL_HEIGHT = 200

    def __repr__(self):
        """Return a pretty formtated representation of the enum."""
        keys = []
        representation = 'SASS variables enum: \n'
        for key in self.__dict__:
            if key[0] in string.ascii_uppercase:
                keys.append(key)

        for key in sorted(keys):
            representation += f'    {key} = {self.__dict__[key]}\n'
        return representation


SASS_VARIABLES = SassVariables()


def load_sass_variables(data):
    """Parse Sass file styles and get custom values for used in code."""
    global SASS_VARIABLES  # pylint: disable=global-variable-not-assigned
    pattern = re.compile(r'[$]\S*:.*?;')
    variables = re.findall(pattern, data)
    for var in variables:
        name, value = var[1:-1].split(':')
        if name[0] in string.ascii_uppercase:
            value = value.strip()
            with contextlib.suppress(BaseException):
                value = ast.literal_eval(value)
            setattr(SASS_VARIABLES, name, value)
    return SASS_VARIABLES


def load_style_sheet():
    """Load css styles file and parse to include custom variables."""
    with open(GLOBAL_SASS_STYLES_PATH, 'rt', encoding='utf-8') as f:
        sass_data = f.read()

    load_sass_variables(sass_data)

    with open(GLOBAL_STYLES_PATH, 'rt', encoding='utf-8') as f:
        data = f.read()

    if os.name == 'nt':
        data = data.replace('$IMAGE_PATH', images.IMAGE_PATH.replace('\\', '/'))
    else:
        data = data.replace('$IMAGE_PATH', images.IMAGE_PATH)

    return data
