# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Qt utilities."""

from __future__ import absolute_import, division, unicode_literals

import os
import re
from qtpy.QtCore import Qt, QTimer  # pylint: disable=no-name-in-module
from qtpy.QtWidgets import QAction, QApplication, QMenu  # pylint: disable=no-name-in-module
from navigator_updater.utils.linux_scaling import setup_scale_factor_for_linux


def qapplication(test_time=60):
    """Create QApplication instance."""
    app = QApplication.instance()

    if app is None:
        setup_scale_factor_for_linux()
        app = QApplication(['Anaconda-Navigator'])
        app.setApplicationName('Anaconda-Navigator')

    test_ci = os.environ.get('TEST_CI')

    if test_ci is not None:
        timer_shutdown = QTimer(app)
        timer_shutdown.timeout.connect(app.quit)
        timer_shutdown.start(test_time * 1000)
    return app


def add_actions(target, actions, insert_before=None):
    """Add actions to a menu."""
    previous_action = None
    target_actions = list(target.actions())
    if target_actions:
        previous_action = target_actions[-1]
        if previous_action.isSeparator():
            previous_action = None
    for action in actions:
        if (action is None) and (previous_action is not None):
            if insert_before is None:
                target.addSeparator()
            else:
                target.insertSeparator(insert_before)
        elif isinstance(action, QMenu):
            if insert_before is None:
                target.addMenu(action)
            else:
                target.insertMenu(insert_before, action)
        elif isinstance(action, QAction):
            if insert_before is None:
                target.addAction(action)
            else:
                target.insertAction(insert_before, action)
        previous_action = action


def create_action(  # pylint: disable=too-many-arguments
    parent,
    text,
    shortcut=None,
    icon=None,
    tip=None,
    toggled=None,
    triggered=None,
    data=None,
    menurole=None,
    context=Qt.WindowShortcut
):
    """Create a QAction."""
    action = QAction(text, parent)
    if triggered is not None:
        action.triggered.connect(triggered)
    if toggled is not None:
        action.toggled.connect(toggled)
        action.setCheckable(True)
    if icon is not None:
        action.setIcon(icon)
    if shortcut is not None:
        action.setShortcut(shortcut)
    if tip is not None:
        action.setToolTip(tip)
        action.setStatusTip(tip)
    if data is not None:
        action.setData(data)
    if menurole is not None:
        action.setMenuRole(menurole)
    # NOTE: Hard-code all shortcuts and choose context=Qt.WidgetShortcut (this will avoid calling shortcuts from another
    #   dockwidget since the context thing doesn't work quite well with these widgets)
    action.setShortcutContext(context)
    return action


def update_pointer(cursor=None):
    """Update application pointer."""
    if cursor is None:
        QApplication.restoreOverrideCursor()
    else:
        QApplication.setOverrideCursor(cursor)


def file_uri(fname):
    """Select the right file uri scheme according to the operating system."""
    if (os.name == 'nt') and re.search(r'^[a-zA-Z]:', fname):
        return 'file:///' + fname
    return 'file://' + fname
