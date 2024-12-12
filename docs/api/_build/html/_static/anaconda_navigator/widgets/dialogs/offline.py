# -*- coding: utf-8 -*-

# pylint: disable=invalid-name

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Update application dialog."""

from qtpy.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout  # pylint: disable=no-name-in-module
from anaconda_navigator.config import CONF
from anaconda_navigator.widgets import ButtonPrimary, QCheckBox, SpacerHorizontal, SpacerVertical
from anaconda_navigator.widgets.dialogs import DialogBase


class DialogOfflineMode(DialogBase):
    """Offline mode dialog."""

    WIDTH = 460
    _MESSAGE_BASE = (
        'Some of the functionality of Anaconda Navigator will be limited. '  # pylint: disable=implicit-str-concat
        'Conda environment creation will be subject to the packages currently available on your package '
        'cache.<br><br>'
    )
    _MESSAGE_LOC = (
        '<b>Offline mode</b> is indicated to the left of the login/logout '  # pylint: disable=implicit-str-concat
        'button on the top right corner of the main application window.<br><br>'
    )
    _MESSAGE_ENABLE = 'Offline mode will be disabled automatically when internet connectivity is restored.<br><br>'
    _MESSAGE_FORCE = (
        'You can also manually force <b>Offline mode</b> by enabling the setting on the application preferences.<br>'
    )
    _MESSAGE_EXTRA_PREF = 'By checking this option you will force <b>Offline mode</b>.<br>'

    MESSAGE_TOOL = _MESSAGE_BASE + _MESSAGE_ENABLE + _MESSAGE_FORCE
    MESSAGE_PREFERENCES = _MESSAGE_BASE + _MESSAGE_LOC + _MESSAGE_ENABLE + _MESSAGE_EXTRA_PREF
    MESSAGE_DIALOG = _MESSAGE_BASE + _MESSAGE_LOC + _MESSAGE_ENABLE + _MESSAGE_FORCE

    def __init__(
        self,
        parent=None,
        config=CONF,
    ):
        """Offline mode dialog."""
        super().__init__(parent=parent)

        self.label = QLabel(self.MESSAGE_DIALOG)
        self.button_ok = ButtonPrimary('Ok')
        self.checkbox_hide = QCheckBox("Don't show again")
        self.config = config

        # Widgets setup
        self.label.setWordWrap(True)
        self.setMinimumWidth(self.WIDTH)
        self.setMaximumWidth(self.WIDTH)
        self.setWindowTitle('Offline Mode')

        # Layouts
        layout_buttons = QHBoxLayout()
        layout_buttons.addStretch()
        layout_buttons.addWidget(self.checkbox_hide)
        layout_buttons.addWidget(SpacerHorizontal())
        layout_buttons.addWidget(self.button_ok)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout_buttons.addWidget(SpacerVertical())
        layout_buttons.addWidget(SpacerVertical())
        layout.addLayout(layout_buttons)
        self.setLayout(layout)

        # Signals
        self.button_ok.clicked.connect(self.handle_accept)

        # Setup
        self.button_ok.setFocus()
        self.setup()

    def setup(self):
        """Setup widget content."""
        hide_dialog = self.config.get('main', 'hide_offline_dialog')
        self.checkbox_hide.setChecked(bool(hide_dialog))

    def handle_accept(self):
        """Handle not showing updates on startup."""
        value = bool(self.checkbox_hide.checkState())
        self.config.set('main', 'hide_offline_dialog', value)
        self.accept()
