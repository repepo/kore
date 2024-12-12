# -*- coding: utf-8 -*-

# pylint: disable=missing-module-docstring

import os
import re
from navigator_updater.config import LINUX
from navigator_updater.utils.conda.launch import run_process
from navigator_updater.utils.logs import logger


def setup_scale_factor_for_linux():
    """
    Setup QT_SCALE_FACTOR parameter to be equal to the system one,
    because QT is not able to handle scaling for the X Window Systems
    """
    if LINUX:
        os.environ['QT_SCALE_FACTOR'] = get_scaling_factor_using_dbus()


def get_scaling_factor_using_dbus():
    """Returns system primary monitor scale factor, otherwise one will be returned."""
    scaling_factor = '1'

    stdout, stderr, error = run_process(
        [
            'dbus-send', '--session', '--print-reply', '--dest=org.gnome.Mutter.DisplayConfig',
            '/org/gnome/Mutter/DisplayConfig', 'org.gnome.Mutter.DisplayConfig.GetCurrentState'
        ]
    )
    if stderr or error:
        logger.warning('An exception occurred during fetching list of system display settings.')
    else:
        monitor_name = get_primary_monitor_name()
        if monitor_name:
            scaling_factor_res = re.search(
                r'struct \{\n(?:.*\n){2}.*double (?P<scaling_factor>\d(?:\.\d+)?)(?:.*\n){5}.*'
                fr'string \"{monitor_name}\"',
                stdout,
            )
            if scaling_factor_res:
                scaling_factor = scaling_factor_res.group('scaling_factor')
            else:
                logger.warning('Can\'t detect system scaling factor settings for primary monitor.')

    return scaling_factor


def get_primary_monitor_name():
    """Returns name of the primary monitor"""
    primary_monitor_name = None

    stdout, stderr, error = run_process(['xrandr', '--listactivemonitors'])
    if stderr or error:
        logger.warning('An exception occurred during fetching list of active monitors.')
    else:
        active_monitor_res = re.search(r'\d+\: \+\*(?P<primary_monitor>.*) \d*/\d+x\d+/\d+\+\d+\+\d +.*', stdout)
        if active_monitor_res:
            primary_monitor_name = active_monitor_res.group('primary_monitor')
        else:
            logger.warning('Can\'t detect primary monitor.')

    return primary_monitor_name
