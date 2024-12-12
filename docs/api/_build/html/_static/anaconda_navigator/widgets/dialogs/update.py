# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Update application dialog."""

from qtpy.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout  # pylint: disable=no-name-in-module
from anaconda_navigator.config import CONF, WIN
from anaconda_navigator.widgets import ButtonNormal, ButtonPrimary, SpacerHorizontal, SpacerVertical
from anaconda_navigator.widgets.dialogs import DialogBase


class DialogUpdateApplication(DialogBase):
    """Update application dialog."""

    WIDTH = 580  # pylint: disable=invalid-name

    def __init__(  # pylint: disable=too-many-arguments
        self,
        version,
        config=CONF,
        startup=False,
        qa_testing=False,
        is_root_writable=False,
    ):
        """
        Update application dialog.

        Parameter
        ---------
        version: str
            New version of update available.
        """
        super().__init__()

        extra_text = ''
        if not is_root_writable and WIN:
            extra_text = '<br>(You will be prompted to elevate privileges)<br>'

        self.label = QLabel(
            'There\'s a new version of Anaconda Navigator available. '
            'We strongly recommend you to update. <br><br>'
            'If you click yes, Anaconda Navigator will close and then the '
            'Anaconda Navigator Updater will start.<br><br><br>'
            f'Do you wish to update to <b>Anaconda Navigator {version}</b> now?<br>{extra_text}<br>'
        )
        self.button_yes = ButtonPrimary('Yes')
        self.button_no = ButtonNormal('No, remind me later')
        self.button_no_show = ButtonNormal("No, don't show again")
        self.config = config

        if not startup:
            self.button_no_show.setVisible(False)
            self.button_no.setText('No')

        # Widgets setup
        self.label.setWordWrap(True)
        self.setMinimumWidth(self.WIDTH)
        self.setMaximumWidth(self.WIDTH)
        self.setWindowTitle('Update Application')

        # On QA testing addicon continuumcrew channel allows to test that
        # the update checking mechanism is working with a dummy package
        # version 1000.0.0, this disallows any installation when using that
        # check
        if qa_testing:
            self.button_yes.setDisabled(True)
            self.button_no.setDisabled(True)
            self.button_no_show.setDisabled(True)

        # Layouts
        layout_buttons = QHBoxLayout()
        layout_buttons.addStretch()
        layout_buttons.addWidget(self.button_no_show)
        layout_buttons.addWidget(SpacerHorizontal())
        layout_buttons.addWidget(self.button_no)
        layout_buttons.addWidget(SpacerHorizontal())
        layout_buttons.addWidget(self.button_yes)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout_buttons.addWidget(SpacerVertical())
        layout_buttons.addWidget(SpacerVertical())
        layout.addLayout(layout_buttons)
        self.setLayout(layout)

        # Signals
        self.button_yes.clicked.connect(self.accept)
        self.button_no.clicked.connect(self.reject)
        self.button_no_show.clicked.connect(self.no_show)

        self.button_yes.setFocus()

    def no_show(self):
        """Handle not showing updates on startup."""
        self.config.set('main', 'hide_update_dialog', True)
        self.reject()
