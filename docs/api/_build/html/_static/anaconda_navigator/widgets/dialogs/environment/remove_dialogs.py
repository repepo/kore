# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Dialogs for removing environments."""

__all__ = ['RemoveDialog']

from qtpy import QtWidgets
from anaconda_navigator import widgets
from . import common


class RemoveDialog(common.EnvironmentActionsDialog):  # pylint: disable=too-few-public-methods
    """Remove existing environment dialog."""

    def __init__(self, parent=None, name=None, prefix=None):
        """Remove existing environment `name` dialog."""
        super().__init__(parent=parent)

        # Widgets
        self.label_text = widgets.LabelBase('Do you want to remove the environment?')
        self.label_name = widgets.LabelBase('Name:')
        self.label_name_value = widgets.LabelBase(name)
        self.label_location = widgets.LabelBase('Location:')
        self.label_prefix = widgets.LabelBase(prefix)

        self.button_cancel = widgets.ButtonNormal('Cancel')
        self.button_ok = widgets.ButtonDanger('Remove')

        # Setup
        self.align_labels([self.label_name, self.label_location])
        self.label_prefix.setObjectName('environment-location')
        self.setWindowTitle('Remove environment')
        self.setMinimumWidth(380)
        self.label_name.setMinimumWidth(60)
        self.label_location.setMinimumWidth(60)

        # Layouts
        layout_name = QtWidgets.QHBoxLayout()
        layout_name.addWidget(self.label_name)
        layout_name.addWidget(widgets.SpacerHorizontal())
        layout_name.addWidget(self.label_name_value)
        layout_name.addStretch()

        layout_location = QtWidgets.QHBoxLayout()
        layout_location.addWidget(self.label_location)
        layout_location.addWidget(widgets.SpacerHorizontal())
        layout_location.addWidget(self.label_prefix)
        layout_location.addStretch()

        layout_buttons = QtWidgets.QHBoxLayout()
        layout_buttons.addStretch()
        layout_buttons.addWidget(self.button_cancel)
        layout_buttons.addWidget(widgets.SpacerHorizontal())
        layout_buttons.addWidget(self.button_ok)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_name)
        layout.addWidget(widgets.SpacerVertical())
        layout.addLayout(layout_location)
        layout.addWidget(widgets.SpacerVertical())
        layout.addWidget(widgets.SpacerVertical())
        layout.addLayout(layout_buttons)
        self.setLayout(layout)

        # Signals
        self.button_ok.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)

        # Setup
        self.update_location()
        self.button_ok.setDisabled(False)
