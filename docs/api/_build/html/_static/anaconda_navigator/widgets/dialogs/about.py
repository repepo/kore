# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""About Anaconda Navigator dialog."""

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout  # pylint: disable=no-name-in-module
from anaconda_navigator import __about__
from anaconda_navigator.static import images
from anaconda_navigator.widgets import (
    ButtonLabel, ButtonLink, ButtonNormal, QSvgWidget, SpacerHorizontal, SpacerVertical,
)
from anaconda_navigator.widgets.dialogs import DialogBase


class AboutDialog(DialogBase):
    """About dialog."""
    GITHUB_URL = 'https://github.com/ContinuumIO/anaconda-issues/issues'  # pylint: disable=invalid-name

    # Url, action, description
    sig_url_clicked = Signal(object, object, object)

    def __init__(self, *args, **kwargs):
        """About dialog."""
        super().__init__(*args, **kwargs)

        # Variables
        text = f'''<b>Anaconda Navigator {__about__.__version__}</b><br>
            <br>Copyright &copy; 2016 Anaconda, Inc.
            <p>Created by Anaconda
            <br>
            <p>For bug reports and feature requests, please visit our
            '''

        # Widgets
        self.widget_icon = QSvgWidget(images.ANACONDA_LOGO)
        self.label_about = QLabel(text)
        self.button_link = ButtonLink('Issue Tracker')
        self.button_label = ButtonLabel('on GitHub.')
        self.button_ok = ButtonNormal('Ok')

        # Widgets setup
        self.widget_icon.setFixedSize(self.widget_icon.size_for_width(100))
        self.button_ok.setMinimumWidth(70)
        self.button_ok.setDefault(True)
        self.setWindowTitle('About Anaconda Navigator')

        # Layouts
        layout_h = QHBoxLayout()
        layout_h.addWidget(self.widget_icon, 0, Qt.AlignTop)
        layout_h.addWidget(SpacerHorizontal())

        layout_content = QVBoxLayout()
        layout_content.addWidget(self.label_about, 0, Qt.AlignBottom)
        layout_content_h = QHBoxLayout()
        layout_content_h.addWidget(self.button_link, 0, Qt.AlignLeft)
        layout_content_h.addWidget(self.button_label, 0, Qt.AlignLeft)
        layout_content_h.addStretch(0)

        layout_content.addLayout(layout_content_h)
        layout_h.addLayout(layout_content)

        layout_buttons = QHBoxLayout()
        layout_buttons.addStretch()
        layout_buttons.addWidget(self.button_ok)

        layout_main = QVBoxLayout()
        layout_main.addLayout(layout_h)
        layout_main.addWidget(SpacerVertical())
        layout_main.addWidget(SpacerVertical())
        layout_main.addLayout(layout_buttons)
        self.setLayout(layout_main)

        # Signals
        self.button_link.clicked.connect(lambda: self.sig_url_clicked.emit(self.GITHUB_URL, 'content', 'click'))
        self.button_ok.clicked.connect(self.accept)

        # Setup
        self.button_ok.setFocus()
